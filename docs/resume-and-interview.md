# 简历与面试材料

## 推荐项目名称

SmartRecipe - 基于分层 Multi-Agent 与多源检索的智能食谱决策系统

## 可直接使用的简历描述

- 基于 FastAPI 构建异步智能食谱服务，引入 Coordinator、Reviewer、Planner 与领域 Agent 的分层 Multi-Agent 架构，通过 LangChain structured output 生成结构化任务计划，支持复杂问题的多领域拆解与并发执行，并以规则路由保障模型异常时稳定降级。
- 将 GraphRAG、Text2Cypher、Text2SQL、视觉分析和文件处理封装为独立领域能力；菜谱链路融合 Neo4j 图查询、Milvus 向量检索、BM25 与 Rerank，统计链路通过 SQL validation 和只读执行双重拦截写操作。
- 设计 AgentTask、Handoff、EvidenceItem 等跨 Agent 数据合同，使用 Evidence Ledger 记录来源和稳定哈希；通过 Critic 审计与 `allow/retry/deny` Gate 实现失败隔离、有界重试和危险 SQL 二次拦截。
- 建立覆盖单领域与复合领域的 30 条离线路由评测集，当前规则兜底 exact domain-set match 为 30/30；新增自动化测试覆盖并发执行、路由复核、证据去重、安全拒绝与 API 输出。

## 指标边界

可以说：

> 在 30 条自建离线路由集上，规则兜底的领域集合完全匹配率为 100%，测试包含 recipe+analytics、vision+recipe、file+analytics 等复合意图。

不能把这个数字说成“线上意图准确率 100%”。端到端召回率、答案准确率和幻觉率必须在接入真实模型与数据库后重新运行服务 benchmark。

## 高频面试问答

### 为什么是 Multi-Agent，不是普通工作流？

固定工作流每次选择一个分支；迁移版先生成多个 `AgentTask`，不同领域 Agent 可以并发执行，结果通过 Handoff 汇合。Coordinator、Reviewer、领域执行器和 Critic 的目标、权限、输入输出都不同，因此是受控的分层 Multi-Agent，而不是多个相同 Prompt。

### Reviewer 有什么用？

Coordinator 负责第一次任务拆解，Reviewer 读取原问题和完整计划，检查是否漏掉统计、菜谱、图片或文件领域。Reviewer 只能补充或收紧计划，不能绕过安全策略；结构化输出失败时走规则复核。

### 为什么不直接让 LLM 一次回答？

复杂问题可能同时需要 Neo4j 图关系和 MySQL 聚合。一次生成难以分别验证 Cypher、SQL、证据和错误。拆成任务后可以并发、独立重试、限制权限，并准确定位失败节点。

### Evidence 和 RAG 文档有什么区别？

RAG 文档是检索候选；Evidence 是经过统一归一化、带来源、哈希和任务绑定的可审计依据。SQL 语句、图谱结果和文档都可以成为 Evidence。

### DeepReason 相比原版提升在哪里？

原版主要解决单意图路由和专业子图执行；迁移版增加复杂任务分解、多 Agent 并发、Reviewer 复核、结构化交接、证据账本、Critic 和统一 Gate。数据库和检索能力继续复用原有成熟实现。

