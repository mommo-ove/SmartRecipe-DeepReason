# SmartRecipe 个性化营养约束餐单规划设计

**状态：** 已确认设计，尚未实施

**日期：** 2026-08-04

**目标用户：** 健康成年人的非医疗性日常体重管理

**核心场景：** 生成满足营养、安全、时间和预算约束的七日餐单，并支持局部换餐后的增量重规划。

## 1. 背景与问题定义

原SmartRecipe以菜谱问答和多路检索为主，已经具备LangGraph工作流、BM25、向量检索、Neo4j、Text2Cypher、Text2SQL、Evidence Ledger和依赖调度等基础模块。但“根据条件推荐两道菜”仍可被单个大模型完成，难以证明Multi-Agent、图数据库和异构查询的必要性。

新版将业务目标收敛为“个性化营养约束规划与验证”：系统不是凭语言模型常识编写餐单，而是将自然语言目标转换为结构化约束，从真实菜谱和营养数据中检索候选，通过确定性约束求解器生成七日组合，再对全部硬约束和引用证据进行验证。

核心区别是：

```text
普通菜谱问答：找到相关内容并回答
新版餐单规划：找到候选 → 补齐事实 → 求可行组合 → 验证 → 解释
```

## 2. 范围与非目标

### 2.1 核心范围

- 七日三餐规划；
- 每日热量区间和最低蛋白质；
- 过敏原零容忍；
- 单餐制作时间上限；
- 七日总预算；
- 菜谱和主要食材重复次数限制；
- 用户口味偏好；
- 局部换餐后的增量重规划；
- 无解冲突诊断；
- 检索、Text2Cypher、求解、Agent链路和回答的分层评测。

### 2.2 非目标

- 不处理糖尿病、肾病、孕期等医疗饮食建议；
- 不让LLM自行计算或猜测营养数据；
- 不以分布式向量数据库规模作为项目卖点；
- 不把Text-to-Image结果作为营养或安全证据；
- 第一阶段不要求CLIP图片召回和VLM营养标签解析阻塞主链路交付。

## 3. 当前数据现实与质量门禁

仓库当前约有181个Markdown菜谱和157张图片。“6000余个”更接近菜谱、食材、步骤、分类等图谱节点总量，不应描述为6000道菜。现有菜谱包含非标准用量，部分营养文本来源与单位不可追溯，不能直接用于精确约束求解。

采用两层数据策略：

```text
全部原始菜谱
├─ 可进入普通问答与检索
└─ 通过数据质量门禁的菜谱（planning_eligible=true）
   可进入七日规划求解器
```

第一阶段目标是整理60至100道可规划菜谱，后续扩展到约200道。营养数据采用“中文菜谱 + USDA FoodData Central + 中文食材别名映射”的方案：

1. 从Markdown解析食材、用量、份数和步骤；
2. 将“个、勺、少许”等单位归一化为克或毫升；
3. 将中文食材映射到稳定的canonical ingredient ID和FDC ID；
4. 保存每100克营养值、数据来源和数据版本；
5. 用确定性Python代码计算每份菜谱营养；
6. 无法映射或关键字段缺失时标记为`nutrition_incomplete`，不得进入求解。

参考：[USDA FoodData Central API](https://fdc.nal.usda.gov/api-guide/)。

## 4. 总体架构

```text
用户请求
   ↓
Router
   ├─ 普通做法/营养查询 → 直接能力路由
   ├─ 关键字段缺失 → Clarify
   └─ 七日规划/局部换餐 → 依赖调度工作流
          ↓
   Constraint Agent
          ↓
   Recipe Retrieval Agent
      ├─ BM25
      ├─ Chroma Dense Retrieval
      └─ Neo4j关系检索（按需）
          ↓ recipe_id级RRF + Rerank
   Nutrition Agent
          ↓
   OR-Tools CP-SAT Solver
          ↓
   Verification / Critic Gate
          ↓
   Response Agent
```

并非所有节点都包装为Agent：

- LLM负责自然语言理解、约束抽取、长尾关系查询规划和结果解释；
- Python负责依赖判断、数值复算、集合校验、过敏原拦截和证据检查；
- OR-Tools负责组合选择和可行性判断；
- MySQL、Neo4j和Chroma负责提供事实与索引。

## 5. 依赖调度与结构化Handoff

主链路的真实依赖为：

```text
PlanningRequest
  → CandidateRecipeSet
  → EnrichedRecipeSet
  → MealPlanResult
  → VerificationReport
  → FinalAnswer
```

候选菜谱内部的BM25、Dense和图检索可以并行；营养查询必须等待候选`recipe_ids`；求解器必须等待约束和完整营养事实；验证必须等待求解结果。调度器只发送依赖已经满足的ready tasks，任务完成后更新状态和结构化Handoff。

关键合同包括：

- `PlanningRequest`：热量、蛋白质、过敏原、预算、时间和偏好；
- `CandidateRecipeSet`：去重后的recipe IDs、各检索路径排名和证据；
- `EnrichedRecipeSet`：每份营养、成本、时间和数据版本；
- `MealPlanResult`：Solver状态、选中组合、目标函数和冲突信息；
- `VerificationReport`：硬约束检查、证据覆盖和Gate决定。

## 6. 数据存储职责

### 6.1 MySQL：事实源

MySQL保存精确、可聚合的规划事实：

- 菜谱、餐次类型、份数、制作时间、每份成本；
- 食材标准名、FDC ID、每100克营养和过敏原；
- 菜谱食材用量、单位归一化状态；
- 每份菜谱营养汇总；
- `planning_eligible`和`data_version`。

### 6.2 Chroma：默认Dense检索后端

Chroma保存菜谱Chunk向量和检索元数据：

- `chunk_id`、`recipe_id`、`chunk_type`；
- `planning_eligible`、`meal_type`；
- embedding模型版本和内容哈希。

当前数据规模只有几万级检索单元，Chroma足以完成本地Dense检索和元数据过滤。现有Milvus适配器保留，但不作为默认依赖，也不以“数据量很大”作为选型理由。

### 6.3 Neo4j：关系查询

Neo4j只承载真正需要遍历的关系：

```text
Recipe -[CONTAINS]-> Ingredient
Ingredient -[BELONGS_TO]-> Allergen
Ingredient -[SUBSTITUTE_FOR]-> Ingredient
Recipe -[SUITABLE_FOR]-> MealType
```

热量筛选与营养聚合仍走MySQL；Neo4j服务过敏原传播、食材替换和关联解释。

## 7. 检索与融合

在线检索顺序为：

1. 根据过敏原和其他硬条件得到安全候选范围；
2. BM25与Chroma Dense Retrieval并行召回；
3. 需要关系推理时调用固定Cypher或受控Text2Cypher；
4. 每条检索路径先按`recipe_id`去重，只保留本路径最高排名Chunk；
5. 使用recipe_id级RRF融合；
6. 对候选菜谱摘要和最佳证据Chunk执行Cross-Encoder Rerank；
7. 输出统一的`selected_recipe_ids`。

安全过滤不能依赖Rerank。含目标用户过敏原的菜谱即使排名较低，也不得进入求解候选集。

检索后端需要统一接口，例如：

```text
VectorStorePort
├─ ChromaVectorStore（默认开发与演示）
└─ MilvusVectorStore（保留现有实现）
```

## 8. Text2Cypher与Text2SQL边界

### 8.1 Text2Cypher

采用“固定模板优先、动态生成兜底”：

```text
问题分类
→ 固定模板未命中
→ 精简Schema + few-shot生成只读Cypher
→ EXPLAIN与Schema检查
→ 标签/关系/函数白名单
→ LIMIT与超时
→ 执行并保存结果证据
```

Text2Cypher服务长尾关系问题，例如“替换牛奶后，替代食材是否会引入新的过敏原，并有哪些早餐菜谱可使用”。动态查询失败时降级到模板或Clarify，不允许无限重写。

可参考Neo4j官方GraphRAG的`Text2CypherRetriever`与`VectorCypherRetriever`：[Neo4j GraphRAG Retriever](https://neo4j.com/docs/neo4j-graphrag-python/current/user_guide_rag.html)。

### 8.2 Text2SQL

固定营养读取使用参数化SQL，不为展示Agent能力而每次生成SQL。Text2SQL用于长尾分析，例如“本周餐单相比上周平均每天减少多少热量”。它必须受到：

- 用户与recipe ID闭集约束；
- AST只读校验；
- 表字段白名单；
- 行数、时间和结果规模限制；
- 有界纠错重试。

## 9. 约束求解

Constraint Agent输出结构化Schema，例如：

```json
{
  "days": 7,
  "daily_calories": {"min": 1400, "max": 1600},
  "daily_protein_min": 90,
  "allergens": ["peanut"],
  "max_cook_time_per_meal": 30,
  "weekly_budget": 300,
  "preferences": ["鸡胸肉", "清淡"],
  "max_recipe_repeat": 2
}
```

对每个“日期、餐次、菜谱”建立布尔变量：

```text
x[day, meal, recipe] ∈ {0, 1}
```

硬约束包括：

- 每顿必须满足餐次组合要求；
- 每日热量位于目标区间；
- 每日蛋白质达到下限；
- 过敏原菜谱不可选；
- 单餐制作时间不超过上限；
- 七日总成本不超过预算；
- 菜谱和主要食材重复次数受限。

软目标采用分阶段优化：

```text
可行性
  > 用户偏好
  > 菜品多样性
  > 成本与食材复用
```

CP-SAT使用整数，价格采用分，营养值按固定倍率转换。Solver返回`OPTIMAL`、`FEASIBLE`、`INFEASIBLE`、`MODEL_INVALID`或`UNKNOWN`，系统不得把状态混为“成功/失败”二值。

参考：[Google OR-Tools CP-SAT](https://developers.google.com/optimization/cp/cp_solver)。

## 10. 无解诊断与局部重规划

每组约束绑定assumption literal。若返回`INFEASIBLE`，使用`sufficient_assumptions_for_infeasibility()`获取足以导致无解的约束集合，生成结构化冲突报告。该集合不一定是唯一或最小冲突集，因此文案只表述为“足以导致无解的约束组合”。

系统只能建议放宽预算、偏好、热量容差等可调整项；过敏原等安全约束不能自动放宽。

用户局部换餐时：

1. 将未受影响餐次锁定；
2. 对新排除食材或新偏好添加约束；
3. 只重新求解受影响日期及相关营养平衡；
4. 验证未受影响餐次保持率与新约束满足率。

## 11. Evidence Ledger与Critic Gate

所有工具输出原子Evidence：

- 检索证据：recipe ID、Chunk、路径与排名；
- 营养证据：MySQL记录、数据版本和来源；
- 图证据：Cypher、返回节点和关系路径；
- 求解证据：输入约束、Solver状态、组合和每日合计；
- 验证证据：Python复算结果与违规列表。

Answer Agent为关键Claim引用Evidence ID。Critic按成本从低到高检查：

1. Evidence ID是否存在；
2. Claim引用的recipe ID是否属于最终餐单；
3. 数值与Python复算是否一致；
4. 是否存在过敏原或其他硬约束违规；
5. 来源与数据版本是否完整；
6. 仅对规则难覆盖的语义Claim调用LLM判断。

Gate状态定义为：

- `ALLOW`：验证通过；
- `RETRY`：底层方案正确，仅答案表达错误；
- `REPLAN`：候选或餐单不满足约束；
- `CLARIFY`：关键字段缺失或用户约束冲突；
- `DENY`：安全违规或关键证据缺失。

## 12. 多模态与Text-to-Image

Text-to-Image降为非核心展示能力：只有经过验证的餐单才能生成餐盘示意图，Prompt必须由已验证事实构造，图片标记为AI生成效果图，不参与营养判断。

更有业务价值的后续能力是营养标签结构化解析：

```text
包装营养成分表图片
→ VLM/OCR提取每100g、每份、kJ/kcal和蛋白质
→ Pydantic结构化输出
→ 单位换算与数值范围检查
→ 低置信度字段请求用户确认
→ 写入个人食物库
```

该能力可使用字段准确率、数值Exact Match、单位换算正确率和低置信度召回率评测。已有CLIP编码器、图片语料和Milvus图片仓储全部保留，但不阻塞核心MVP。

## 13. 失败处理

- 关键槽位缺失：进入Clarify；
- 候选不足：扩大Top-K或执行一次Query Rewrite；
- 营养字段缺失：剔除菜谱，不允许LLM补数；
- 动态Cypher不安全：拒绝执行并降级；
- Text2SQL越权：拒绝并记录Gate finding；
- Solver无解：输出冲突约束，请用户选择可放宽项；
- Solver超时：只有已经得到且验证通过的`FEASIBLE`方案才可返回；
- 验证失败：根据问题进入`RETRY`、`REPLAN`或`DENY`；
- 外部模型超时：有界重试，禁止无限循环。

## 14. 可复现评测

建立100条版本化Benchmark：

| 子集 | 数量 | 核心内容 |
|---|---:|---|
| 菜谱检索 | 40 | 关键词、语义、多条件和模糊需求 |
| Text2Cypher | 20 | 替换、过敏原和多跳关系 |
| 七日规划 | 30 | 20条有解、10条故意无解 |
| 局部重规划 | 10 | 换菜、预算变化、临时排除食材 |

至少30条由人工编写和复核，其余从真实数据按约束模板生成并抽查。LLM可用于同义改写，但不能生成Gold Label。

### 14.1 指标

- 检索：Recall@5/10、MRR、NDCG@10；
- Text2Cypher：语法合法率、执行成功率、结果集合准确率、安全拦截率；
- 求解：有解判断准确率、硬约束满足率、目标函数值、求解耗时；
- 重规划：未受影响餐次保持率、新约束满足率；
- Agent：任务顺序正确率、Handoff完整率、任务完成率、P50/P95延迟和Token成本；
- 回答：关键数值证据覆盖率和辅助Faithfulness指标。

过敏原违规数和硬约束违规数的发布门槛为0。

### 14.2 消融实验

```text
Dense
BM25
BM25 + Dense + RRF
BM25 + Dense + RRF + Rerank
```

Neo4j只在多跳关系子集上与无图方案比较。只有组件带来真实收益，才保留相应复杂度。

LLM案例重复运行3次并报告平均值与最差值；开发集用于调参，保留测试集只用于最终评估。每次运行输出包含数据、模型、Prompt、随机种子、任务DAG、工具输入输出、SQL/Cypher、Evidence IDs、延迟和Gate状态的JSON报告。

可参考：[Ragas Metrics](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/)。Ragas只作为辅助评价，不能替代确定性约束检查。

## 15. 实施顺序

1. 数据Schema、质量门禁和60至100道可规划菜谱；
2. OR-Tools求解、无解和局部换餐测试；
3. Chroma适配器、recipe ID级RRF与检索消融；
4. 结构化Handoff接入现有依赖调度器；
5. 过敏原/替代关系与受控Text2Cypher；
6. Evidence、Critic、Trace与100条Benchmark；
7. 真实联调和指标报告；
8. 之后再评估营养标签VLM与Text-to-Image。

实施继续采用“教学 + TDD”模式：每项任务先解释真实问题、比较方案并诊断基础理解；随后写失败测试、实现最小代码、运行最小实验、制造失败并观察；最后才形成面试表述。

## 16. 完成定义

只有同时满足以下条件，核心MVP才算完成：

- 至少一条七日规划请求完成真实端到端运行；
- 至少一条无解案例给出结构化冲突报告；
- 至少一次局部换餐保持未受影响餐次；
- Chroma Dense与BM25、RRF、Rerank完成真实消融；
- Text2Cypher在真实Neo4j上执行并通过结果集评测；
- MySQL、Neo4j、Chroma及模型调用完成真实联调；
- 100条Benchmark产生可复现JSON报告；
- 过敏原与硬约束违规数为0；
- 文档明确区分单元测试、Fake Client测试、真实集成测试和完整联调。
