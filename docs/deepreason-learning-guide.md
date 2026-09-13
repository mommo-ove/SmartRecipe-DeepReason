# DeepReason 三天速成学习指南

## 第一天：看懂为什么需要上层编排

先读 `gustobot/application/deepreason/models.py` 和 `planning.py`。

必须掌握：

1. Coordinator 不直接回答，它把问题输出为 `ExecutionPlan`。
2. `with_structured_output(ExecutionPlan)` 把模型限制成结构化规划器。
3. Reviewer 使用独立的 `ReviewDecision` 再检查一次，只允许补任务或收紧计划。
4. 模型格式错误、超时或不可用时，`HeuristicPlanner` 负责规则兜底。

练习问题：

> 推荐一道低辣鸡肉菜，并统计同类菜的平均烹饪时长。

预期拆成 `recipe` 和 `analytics` 两个任务。不要直接让一个 LLM 同时生成 Cypher、SQL 和答案：拆开后每条链可以独立校验、测试、重试和审计。

## 第二天：看懂 Agent 不是一个 Python 文件名

读 `domain_agents.py` 和 `orchestrator.py`。

领域 Agent 由四部分构成：明确职责、输入合同、可调用能力、输出合同。`DomainAgentRegistry` 按领域找到执行器；`FunctionDomainAgent` 统一处理超时、异常和 Handoff；顶层 LangGraph 使用 `Send` 动态创建多个 `domain_agent` 分支，并在下一个 superstep 汇合。

这里要特别理解“控制面只决策一次”：Coordinator 和 Reviewer 产出的 `AgentTask.domain` 已经是经过校验的领域合同。Registry 根据该字段直接选择业务能力，不再调用旧 Router。Recipe Agent 直接进入 RAG 子图，Analytics Agent 直接进入 Text2SQL 子图，图片、文件和通用 Agent 直接进入对应节点。这样减少一次 LLM 调用，也避免上层判断为 `analytics`、下层又误判为 `recipe` 的冲突。

面试表达：

> Coordinator 和 Reviewer 负责控制面，领域 Agent 负责数据面。复杂问题会形成多个独立 AgentTask，并发执行后通过结构化 Handoff 汇合，因此它不是只能单路由的固定流水线。

同时复习原有三条菜谱查询能力：固定问题走 predefined Cypher，灵活图查询走 Text2Cypher，模糊推荐和语义问题走 GraphRAG。统计问题独立走 Text2SQL。

## 第三天：证据、门禁和评测

读 `evidence.py`、`orchestrator.py` 中的 `audit_handoffs()`、`decide_gate()`，最后看 `tests/test_deepreason_orchestrator.py`。

一句话记忆：

> Agent 产出结果，Evidence 记录依据，Critic 找问题，Gate 决定能不能发布。

重点回答：

- 为什么还要 Critic：工具执行成功不等于结果安全、完整、有依据。
- 为什么 Gate 不是 LLM：安全底线应由确定性代码执行，模型只能提供建议。
- 为什么限制重试：防止 Agent 无限循环、延迟和 Token 成本失控。
- 为什么 Evidence 用稳定哈希：便于去重、追踪和回放同一来源。

运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_deepreason_*.py -q
.\.venv\Scripts\python.exe scripts\run_deepreason_benchmark.py
```

当前离线评测集包含 30 条单领域和复合领域问题，指标为 exact domain-set match。该指标只验证规则兜底路由，不等同于端到端问答准确率。
