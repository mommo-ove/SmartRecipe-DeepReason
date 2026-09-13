# FoodTrace Allergen Recall MVP Implementation Plan

> **For Claude:** Use `${SUPERPOWERS_SKILLS_ROOT}/skills/collaboration/executing-plans/SKILL.md` to implement this plan task-by-task.

**Goal:** 复用 SmartRecipe DeepReason 的成熟控制面，在不从零造框架的前提下实现“未标注过敏原召回调查”纵向闭环，并形成一周内可本地演示、可评测、可用于实习面试的 FoodTrace MVP。

**Architecture:** FoodTrace 是事件驱动的供应链调查 Agent，不以“多 Agent 数量”为卖点。事故解析后，系统先通过固定 Cypher 追踪原料批次到生产批次，再用参数化 SQL 查询库存和订单；SOP 检索可以并行，最后由一致性 Reviewer 和保守 Gate 输出 `ALLOW_REPORT / HUMAN_REVIEW / BLOCKED`。它复用 DeepReason 的结构化 Handoff、Evidence 和 trace，但领域模型、依赖图、风险策略和 benchmark 与 GeoReport 完全不同。

**Tech Stack:** Python、Pydantic、LangGraph/DeepReason、Neo4j Cypher、MySQL/SQLAlchemy、pytest、FastAPI（仅复用现有入口）、确定性 fixture 和 Gold Label。

---

## 新线程启动提示词

在新的 FoodTrace 线程中直接发送下面这段：

```text
请执行 FoodTrace 的 deadline MVP 计划：
F:\agent+项目\SmartRecipe-DeepReason\.worktrees\multimodal-dependency-upgrade\docs\plans\2026-07-26-foodtrace-allergen-recall-mvp.md

工作树：
F:\agent+项目\SmartRecipe-DeepReason\.worktrees\multimodal-dependency-upgrade

请使用 Superpowers 的 Executing Plans、TDD、Verification Before Completion；
逐任务执行，每个任务先 RED、再 GREEN、再独立代码审查。

重要：当前工作树已有未提交的 DeepReason scheduling、Handoff、Text2SQL scope 修改。
禁止 reset、checkout、覆盖或删除这些用户修改；先按 Task 0 审计、测试并建立检查点。

FoodTrace 只做“未标注花生过敏原召回”一种事故。
不要把它扩展成第二个 GeoReport 式四路并发 Multi-Agent，
不要加入 CLIP、图片 Milvus、微生物污染、跨国法规、K8s 或真实通知发送。

每完成一个任务，用中文告诉我：
1. 业务链路为什么这样设计；
2. Cypher/SQL/Evidence/Gate 分别解决什么问题；
3. 测试和指标证明了什么；
4. 面试官可能怎样追问，我应怎样回答。

先从 Task 0 开始，不允许跳过工作树保护。
```

## 当前真实状态

- 工作树：`F:\agent+项目\SmartRecipe-DeepReason\.worktrees\multimodal-dependency-upgrade`
- 分支：`feature/multimodal-dependency-upgrade`
- 已提交：
  - `d50adde feat: add licensed recipe image corpus`
  - `981baa0 feat: add lazy local CLIP encoder`
  - `460dce1 feat: add Milvus image vector store`
- 当前存在未提交修改：
  - DeepReason models、planning、orchestrator、scheduling；
  - direct capabilities；
  - Text2SQL generation/validation/state；
  - predefined Cypher 与 Text2Cypher prompt；
  - MySQL schema/sample data；
  - scheduling/Handoff/Text2SQL scope 测试。
- 这些改动属于用户已有资产，禁止丢弃。
- FoodTrace 专属领域代码、数据生成器、Gold cases 和 benchmark 尚未实现。

## 与 GeoReport 的强制差异

| 维度 | FoodTrace | GeoReport |
|---|---|---|
| 工作模式 | 事故驱动、在线调查 | 离线、长任务科研分析 |
| 主数据 | Neo4j 供应链图 + SQL 订单 | 日志 + CSV + PNG |
| 依赖形态 | 图追溯完成后才能查受影响订单 | 四个 Analyst 可并行 |
| 核心风险 | 漏召回、批次错配、图/SQL不一致 | 数值错误、科学结论越界 |
| Gate | 是否发布召回范围/转人工 | 是否允许科研结论/报告 |
| 主要指标 | impact-set recall、false-negative rate | numeric accuracy、claim grounding |

FoodTrace 简历中不要重复突出 Coordinator、Planner、四路并发和 PPT；重点写图谱追溯、参数化 SQL、多源一致性和低漏判决策。

## 范围冻结

### 唯一事故

```text
供应商通知某一原料批次可能含未标注花生；
系统需找出使用该原料的配方、产品、生产批次、门店库存和订单，
并根据证据完整性决定发布召回调查报告还是转人工复核。
```

### 本周必须完成

- 固定 seed 企业仿真数据和 Gold Label。
- 原料批次→配方→产品→生产批次的固定只读 Cypher。
- 基于生产批次 ID 的参数化只读 SQL。
- SOP 本地检索。
- 图谱/SQL结果一致性校验。
- Evidence Ledger 和保守 Gate。
- 一个 CLI 演示、一个 FastAPI smoke test。
- 10～30 条 deterministic benchmark。
- 简历 bullet、3 分钟讲稿和追问题树。

### 本周不做

- 动态 Text2Cypher。
- 复杂 GraphRAG。
- CLIP、图片语料和 Milvus。
- 多种污染事故。
- 全量 openFDA/Open Food Facts 导入。
- 真实邮件、短信或召回动作。
- Kubernetes、云部署和 UI 重写。

---

### Task 0: Protect and checkpoint the existing SmartRecipe worktree

**Files:**
- Inspect only: all currently modified/untracked files
- Do not add: `.vscode/`

**Step 1: Capture the exact current state**

```powershell
git status --short
git diff --check
git diff --stat
```

**Step 2: Read the existing plans and changed contracts**

必须阅读：

```text
docs/plans/2026-07-20-multimodal-dependency-upgrade.md
gustobot/application/deepreason/models.py
gustobot/application/deepreason/planning.py
gustobot/application/deepreason/orchestrator.py
gustobot/application/deepreason/scheduling.py
tests/test_deepreason_scheduling.py
tests/test_deepreason_handoff_integration.py
tests/test_text2sql_recipe_scope.py
```

**Step 3: Run the focused existing tests**

```powershell
uv run pytest -q tests/test_deepreason_models.py tests/test_deepreason_planning.py tests/test_deepreason_scheduling.py tests/test_deepreason_handoff_integration.py tests/test_text2sql_recipe_scope.py
```

**Step 4: Review failures without overwriting user work**

- 若失败，使用 Systematic Debugging。
- 不允许通过删测试、放宽断言或 reset 解决。
- 先完成独立代码审查。

**Step 5: Create a checkpoint commit only after review**

精确 `git add` 已审查的 DeepReason/Text2SQL/schema/test 文件，不添加 `.vscode/` 或无关 examples。

```powershell
git commit -m "feat: checkpoint dependency-aware DeepReason handoffs"
```

---

### Task 1: Define the FoodTrace domain contract and deterministic Gold data

**Files:**
- Create: `gustobot/application/foodtrace/__init__.py`
- Create: `gustobot/application/foodtrace/models.py`
- Create: `gustobot/application/foodtrace/fixtures.py`
- Create: `gustobot/data/foodtrace/schema.cypher`
- Create: `gustobot/data/foodtrace/seed_graph.cypher`
- Create: `gustobot/data/foodtrace/seed_orders.sql`
- Create: `benchmark/foodtrace/cases.json`
- Create: `tests/test_foodtrace_fixtures.py`
- Create: `tests/test_foodtrace_models.py`

**Step 1: Write failing model and reproducibility tests**

模型至少包括：

```text
IncidentNotice
IngredientLot
Recipe
Product
ProductionBatch
StoreInventory
OrderExposure
RecallScope
GoldRecallCase
```

断言相同 seed 生成完全相同的数据和 Gold impacted IDs。

**Step 2: Verify RED**

```powershell
uv run pytest -q tests/test_foodtrace_models.py tests/test_foodtrace_fixtures.py
```

**Step 3: Implement the smallest deterministic dataset**

- 1 个花生污染/未标注原料批次。
- 至少 2 条配方路径，其中 1 条不受影响作为负例。
- 至少 3 个生产批次、2 家门店、若干库存和订单。
- 明确图谱节点 ID 与 SQL 外键映射。
- 每个 benchmark case 保存 Gold impacted product/batch/store/order IDs。

**Step 4: Verify GREEN**

```powershell
uv run pytest -q tests/test_foodtrace_models.py tests/test_foodtrace_fixtures.py
```

**Step 5: Review and commit**

```powershell
git add gustobot/application/foodtrace gustobot/data/foodtrace benchmark/foodtrace tests/test_foodtrace_models.py tests/test_foodtrace_fixtures.py
git commit -m "feat: add deterministic FoodTrace allergen fixtures"
```

---

### Task 2: Implement fixed graph tracing and parameterized order lookup

**Files:**
- Create: `gustobot/application/foodtrace/graph_queries.py`
- Create: `gustobot/application/foodtrace/sql_queries.py`
- Create: `gustobot/application/foodtrace/repositories.py`
- Create: `tests/test_foodtrace_graph_queries.py`
- Create: `tests/test_foodtrace_sql_queries.py`
- Create: `tests/test_foodtrace_repositories.py`

**Step 1: Write failing graph-scope tests**

固定 Cypher 应只读地解析：

```text
ingredient_lot
→ recipe
→ product
→ production_batch
```

断言不会跨越无关原料批次，也不允许 LLM 生成任意 Cypher。

**Step 2: Write failing SQL-scope tests**

SQL 必须：

- 接收结构化 `production_batch_ids`；
- 使用绑定参数；
- 只允许 SELECT；
- 查询库存和订单影响集合；
- 拒绝空范围全表查询和写操作。

**Step 3: Verify RED**

```powershell
uv run pytest -q tests/test_foodtrace_graph_queries.py tests/test_foodtrace_sql_queries.py tests/test_foodtrace_repositories.py
```

**Step 4: Implement minimal repositories**

- 测试使用可重复 fixture。
- 集成演示复用现有 Neo4j/MySQL 配置。
- 数据库不可用时明确返回 `dependency_unavailable`，不得伪造成功。

**Step 5: Verify GREEN and optional Docker integration**

```powershell
uv run pytest -q tests/test_foodtrace_graph_queries.py tests/test_foodtrace_sql_queries.py tests/test_foodtrace_repositories.py
```

若本地 Docker 服务可用，再运行 integration marker；不可用不阻塞单元测试，但最终 demo 前必须至少真实跑通一次。

**Step 6: Review and commit**

```powershell
git add gustobot/application/foodtrace tests/test_foodtrace_graph_queries.py tests/test_foodtrace_sql_queries.py tests/test_foodtrace_repositories.py
git commit -m "feat: trace allergen batches across graph and orders"
```

---

### Task 3: Build the incident workflow, Evidence Ledger, and conservative Gate

**Files:**
- Create: `gustobot/application/foodtrace/workflow.py`
- Create: `gustobot/application/foodtrace/evidence.py`
- Create: `gustobot/application/foodtrace/reviewer.py`
- Create: `gustobot/application/foodtrace/gate.py`
- Create: `gustobot/application/foodtrace/sop.py`
- Create: `tests/test_foodtrace_workflow.py`
- Create: `tests/test_foodtrace_gate.py`

**Step 1: Write failing dependency tests**

依赖图固定为：

```text
incident_parse
├─> sop_lookup
└─> graph_scope
      └─> order_exposure
             └─> consistency_review
                    └─> decision_gate
```

`order_exposure` 不得在 `graph_scope` 前执行。

**Step 2: Write failing Gate tests**

状态：

```text
ALLOW_REPORT
HUMAN_REVIEW
BLOCKED
```

必须转人工的情况：

- 事故批次无法定位；
- 图谱和 SQL ID 映射不一致；
- Gold-required evidence 缺失；
- 任一依赖失败；
- 查询范围异常扩大；
- 报告包含没有 evidence ID 的影响数字。

**Step 3: Verify RED**

```powershell
uv run pytest -q tests/test_foodtrace_workflow.py tests/test_foodtrace_gate.py
```

**Step 4: Implement by adapting, not copying, DeepReason contracts**

- 复用 `Handoff`/Evidence 的稳定 ID 和失败表示。
- FoodTrace workflow 只注册本领域 tool handlers。
- 不创建四个平行“聊天 Agent”。
- 数量、集合运算和一致性校验由 Python 完成。

**Step 5: Verify GREEN and full suite**

```powershell
uv run pytest -q tests/test_foodtrace_workflow.py tests/test_foodtrace_gate.py
uv run pytest -q
```

**Step 6: Review and commit**

```powershell
git add gustobot/application/foodtrace tests/test_foodtrace_workflow.py tests/test_foodtrace_gate.py
git commit -m "feat: gate evidence-bound FoodTrace investigations"
```

---

### Task 4: Add CLI/API demo and structured incident trace

**Files:**
- Create: `gustobot/application/foodtrace/cli.py`
- Create: `scripts/run_foodtrace_demo.py`
- Create: `tests/test_foodtrace_demo.py`
- Modify: `main.py`
- Create: `docs/foodtrace-demo.md`

**Step 1: Write a failing end-to-end demo test**

输入一条未标注花生事故通知，断言 trace 显示：

```text
Incident Parser
SOP Lookup
Graph Scope
Order Exposure
Consistency Reviewer
Decision Gate
Recall Report
```

**Step 2: Verify RED**

```powershell
uv run pytest -q tests/test_foodtrace_demo.py
```

**Step 3: Implement one CLI and one thin API endpoint**

- CLI 是面试主入口。
- API 只做 request/response 转换，不复制 workflow。
- 输出受影响 product/batch/store/order ID 集合、证据和 Gate 状态。
- trace 不记录数据库凭据、绝对路径或原始异常。

**Step 4: Verify GREEN and run locally**

```powershell
uv run pytest -q tests/test_foodtrace_demo.py
uv run python scripts/run_foodtrace_demo.py --case allergen_peanut_001
```

**Step 5: Commit**

```powershell
git add gustobot/application/foodtrace/cli.py scripts/run_foodtrace_demo.py tests/test_foodtrace_demo.py main.py docs/foodtrace-demo.md
git commit -m "feat: add FoodTrace incident-response demo"
```

---

### Task 5: Benchmark recall scope, safety behavior, and failure recovery

**Files:**
- Create: `gustobot/application/foodtrace/benchmark.py`
- Create: `scripts/run_foodtrace_benchmark.py`
- Create: `tests/test_foodtrace_benchmark.py`
- Create: `benchmark/foodtrace/results/.gitkeep`

**Step 1: Write failing metric tests**

指标固定为：

```text
impacted_batch_precision
impacted_batch_recall
impacted_order_recall
false_negative_rate
evidence_coverage
inconsistency_gate_recall
unsafe_sql_block_rate
end_to_end_success_rate
p50_runtime_ms
p95_runtime_ms
```

**Step 2: Build 10～30 deterministic cases**

包含：

- 正常单路径召回；
- 一个原料影响多个配方；
- 不受影响负例；
- 批次不存在；
- 图谱边缺失；
- 图/SQL ID 不一致；
- SQL 依赖失败；
- SOP 缺失；
- 查询范围异常；
- 无证据影响数字。

**Step 3: Verify RED then GREEN**

```powershell
uv run pytest -q tests/test_foodtrace_benchmark.py
uv run python scripts/run_foodtrace_benchmark.py
```

保存原始 JSON 结果；简历只允许引用实际结果。

**Step 4: Commit**

```powershell
git add gustobot/application/foodtrace/benchmark.py scripts/run_foodtrace_benchmark.py tests/test_foodtrace_benchmark.py benchmark/foodtrace
git commit -m "test: benchmark FoodTrace recall safety"
```

---

### Task 6: Produce interview materials and final verification

**Files:**
- Create: `docs/interview/foodtrace-three-minute-script.md`
- Create: `docs/interview/foodtrace-question-tree.md`
- Create: `docs/interview/foodtrace-resume-bullets.md`
- Create: `docs/interview/foodtrace-failure-retrospective.md`
- Modify: `README.md`

**Step 1: Write the three-minute story**

必须按以下顺序：

```text
业务事故
为什么图数据库
为什么 SQL 必须依赖图追溯结果
为什么不用动态 Text2Cypher
如何绑定 Evidence
图/SQL冲突怎样进入 HUMAN_REVIEW
benchmark 实测结果
失败方案与取舍
```

**Step 2: Prepare the interview question tree**

至少回答：

- 为什么不是普通 RAG？
- 为什么不用四个并发 Agent？
- Neo4j 和关系库怎样分工？
- 如何防止 Cypher/SQL 越权？
- 为什么漏召回比多召回更危险？
- Gold Label 怎样生成，是否等于真实线上指标？
- 图谱与订单数据不一致怎么办？
- 真实数据和仿真数据怎样区分？
- 如果数据库宕机，系统为什么不能继续生成报告？

**Step 3: Final verification**

```powershell
uv run pytest -q
uv run python scripts/run_foodtrace_demo.py --case allergen_peanut_001
uv run python scripts/run_foodtrace_benchmark.py
git diff --check
```

**Step 4: Independent final review and commit**

```powershell
git add docs/interview README.md
git commit -m "docs: finish FoodTrace interview-ready MVP"
```

---

## 完成定义

只有同时满足以下条件，FoodTrace 才可以宣布完成：

- 一个命令可复现未标注花生事故调查。
- 图追溯范围和 SQL 订单范围均可回到 Gold Label。
- SQL 使用绑定参数且只读。
- 图谱/SQL不一致时 Gate 转 `HUMAN_REVIEW`，不伪造完整报告。
- 至少一个依赖失败案例被正确阻断。
- benchmark 保存原始结果，简历不伪造线上指标。
- 能明确解释哪些数据真实、哪些是固定 seed 仿真。
- 3 分钟讲稿不把项目讲成 GeoReport 的业务换皮。

## 时间盒

- 第 1 天：Task 0～1。
- 第 2 天：Task 2～3。
- 第 3 天：Task 4～6。
- 第 4～7 天：只允许修 Bug、真实跑通数据库、演练和修改表达，不允许新增事故类型或基础设施。

