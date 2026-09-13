# SmartRecipe 关系约束检索评测 v1

## 结论

这次评测验证的是“必须包含 A、B，同时排除 C”的关系约束检索，不是整个问答系统的端到端准确率。

- 真实语料：Food.com 派生公开数据，经字段完整性与规划可用性过滤后取 300 条菜谱。
- 真实图数据库：Neo4j 5.26 Community。
- 图谱规模：300 个 Recipe、624 个共享 Ingredient、2,372 条 `HAS_INGREDIENT` 边。
- 测试集：40 条结构化关系题，10 条 development、30 条冻结 test。
- Gold 生成：直接穷举原始菜谱的食材字段，要求同时命中两个必选食材且不含禁用食材；不依赖 LLM 生成标签。
- 冻结标识：测试文件 SHA-256 为 `964a82ad4521700e1fa339ad79b5d4d68ad311ab6fa933ed072f7d9b7de5c1c6`。

## Test 结果

| 配置 | Recall@5 | Recall@20 | MRR | NDCG@20 | 禁用食材违规率@20 | 平均延迟 | P95 延迟 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 40.56% | 89.44% | 25.22% | 42.91% | 30.33% | 11.36 ms | 12.34 ms |
| BGE-M3 Dense | 11.94% | 34.17% | 16.39% | 17.28% | 14.17% | 129.01 ms | 143.03 ms |
| BM25 + Dense + RRF | 26.39% | 77.78% | 37.12% | 41.17% | 26.00% | 137.47 ms | 157.69 ms |
| RRF + Neo4j 硬约束门禁 | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% | 164.39 ms | 181.87 ms |

## 为什么结果不是“加了向量就一定提高”

Dense 检索擅长语义相近，但不天然理解 `without C` 是禁止条件；BM25 甚至会把 C 当作正向关键词。RRF 只融合排名，也不能保证硬约束。因此当前实现先用参数化 Cypher 求出合法 `recipe_id` 集合，再让 BM25/BGE-M3/RRF 只在合法集合内排序。禁用食材不是软加分项，图查询为空时直接返回空结果，不降级为不安全候选。

本测试的 100% 是确定性关系查询在结构化 Gold 上的模块上限，不能写成“系统问答准确率 100%”。它最有价值的业务指标是：列明禁用食材违规率从 BM25 的 30.33% 降至 0%，同时 Recall@20 从 89.44% 提升到 100%。

## 可用于简历的严谨表述

基于 Neo4j 构建菜谱—食材共享图谱，设计参数化 Cypher 关系检索与 `recipe_id` 级硬约束门禁，并在合法候选集内融合 BM25、BGE-M3 与 RRF；在 30 条冻结关系约束测试集上，将 Recall@20 从 BM25 基线的 89.4% 提升至 100%，列明禁用食材违规率由 30.3% 降至 0%，P95 检索延迟为 181.9 ms。

## 复现

```powershell
docker compose up -d neo4j
uv run python scripts/build_meal_planning_graph.py --neo4j-url bolt://localhost:17687
uv run python scripts/run_graph_retrieval_benchmark.py `
  --model "F:\agent+项目\.hf-cache\hub\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181" `
  --neo4j-url bolt://localhost:17687
```

## 当前边界

- 已真实联调：Neo4j 导入、共享节点建图、食材交集/排除查询、BGE-M3 本地编码、BM25/Dense/RRF 消融。
- 已测试：数据集隔离、参数化 Cypher、禁用食材 fail-closed、`recipe_id` 去重与合法集重排。
- 未包含在本指标中：中文实体抽取、同义词归一化、复合食材中的隐含过敏原、交叉污染、LLM 最终答案生成。
- 公开语料只能支持“列明食材排除”，不能声称医学级过敏安全。
