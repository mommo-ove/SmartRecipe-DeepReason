# SmartRecipe Text2Cypher 专项评测

## 评测结论

当前线上链路采用“高频查询模板优先，动态 Text2Cypher 兜底”，而不是每个问题都调用 LLM 生成 Cypher。本轮用真实 Neo4j 和冻结数据评测了模板主链路，并在 development 集上对比了 DeepSeek 动态生成。

- 数据：40 条关系查询，10 条 development，30 条冻结 test。
- 安全集：15 条恶意或不合规 Cypher。
- 评测范围：在食材实体已结构化的前提下，验证 Cypher 生成/模板路由、Schema、EXPLAIN、执行结果和安全拦截。
- 不包含：中文实体抽取、同义词归一、复合过敏原推断、最终回答生成。

## 真实测试结果

| 配置 | 数据分割 | 逻辑精确匹配 | 执行成功率 | P50 | P95 |
|---|---:|---:|---:|---:|---:|
| 模板优先 + 全链路校验 | 冻结 test，30 条 | 100% | 100% | 34.64 ms | 56.00 ms |
| DeepSeek + Schema 约束 | development，10 条 | 100% | 100% | 20.27 s | 34.22 s |

独立安全集上，15 条不安全语句拒绝率为 100%，期望错误码命中率为 100%。冻结 test 中没有触发修复，因此 repair success 在该数据上是 N/A，不应解读为修复失败。

DeepSeek development 结果只是消融实验，不是冻结 test 结论。两组都命中 100% 时，模板主链路的 P50 约为 DeepSeek 动态生成的 1/585，也避免了无必要的 API 调用。

## 评测方法

每条样本保存用户问题、结构化参数、Gold `recipe_id`、期望路由和数据集哈希。执行时记录：

1. 语句是否能被解析为合法候选。
2. 只读安全检查是否通过。
3. 标签、关系和属性是否存在于当前 Schema。
4. Neo4j `EXPLAIN` 是否能构建执行计划。
5. 真实执行后 `recipe_id` 集合是否与 Gold 完全一致。
6. 失败时是否在限定次数内修复。

数据集哈希：

- cases: `2fac02ece6c893cba8f508bec70b3b495afa8e98800a3099e6d9b6d98a287592`
- security: `31094df3e1669341c04ef8cc2a2c33a3241bdcfd595311a28e3d41a5b53cb825`
- source corpus: `7b0feeaf02147fa555ed6947f769fb083734bab485a2bfc9d0e59b12c4bf2ef4`

## 工程决策

本数据中 40 条关系问题都能被参数化高频模板表达，所以它验证的是“模板主链路可靠性”，不能证明“任意中文问题的动态 Text2Cypher 都正确”。现在的选型是：

- 高频、结构稳定、安全敏感的查询：固定 Cypher 模板。
- 模板不覆盖的长尾查询：DeepSeek 动态生成，仍必须经过安全、Schema、参数、EXPLAIN 和有界修复。
- 动态链路需要新建“模板未覆盖的长尾问题”冻结集，不能用当前模板题集代替。

## 状态标记

- 已实现且有测试：评测集构建、development/test 隔离、few-shot 防泄漏、指标计算、安全集、前端 Trace 展示。
- 已真实联调：Neo4j 模板链路冻结 test；DeepSeek + Schema development。
- 已实现但单测使用 Fake Client：动态生成失败、自动修复和最大重试次数。
- 尚未完成：独立动态长尾冻结集、实际触发修复的真实成功率。

## 复现命令

```powershell
docker compose up -d neo4j
uv run python scripts/build_meal_planning_graph.py --neo4j-url bolt://localhost:17687
uv run python scripts/build_text2cypher_benchmark.py
uv run python scripts/run_text2cypher_benchmark.py `
  --split test `
  --config template_first_validated `
  --neo4j-url bolt://localhost:17687 `
  --output benchmark/meal_planning/text2cypher/runs/test-template.v1.json
```
