# DeepReason 3 分钟本地面试演示

## 演示目标

用一个复合领域问题完整走通现有 DeepReason 顶层 LangGraph：

> 推荐一道低辣鸡肉菜，并统计这类菜的平均烹饪时长。

该问题会拆为 `recipe` 与 `analytics` 两个任务，适合同时说明 Planner 的多领域拆解、`Send` 并发扇出、结构化 Handoff、Evidence Ledger 以及 SQL 安全 Gate。

演示默认使用项目已有的 `build_demo_registry()`，不需要 API Key、Docker 或数据库。领域结果是固定的离线样例，但 Coordinator、Planner、Reviewer、顶层 LangGraph、并发调度、Evidence Ledger、Critic 和 Gate 都运行真实代码路径；生产模式只需把 demo registry 换回 `build_gustobot_registry()`。

## 一条命令运行

在项目根目录执行：

```powershell
uv sync --frozen
uv run python scripts/run_deepreason_demo.py
```

已有 `.venv` 时也可以直接执行：

```powershell
.\.venv\Scripts\python.exe scripts\run_deepreason_demo.py
```

脚本按七段输出 Coordinator、Planner、Reviewer、并发领域 Agents、Evidence Ledger、Critic/安全 Gate 和 Synthesizer。Evidence 默认写入 `evidence/deepreason-demo.jsonl`，重复运行依靠稳定哈希自动去重；可用 `--ledger-path <path>` 指定其他位置。

## 演示与现有代码的对应关系

| Trace 段落 | 复用代码 | 现场要点 |
| --- | --- | --- |
| Coordinator / Planner / Reviewer | `planning.py`、`orchestrator.py` | 规则 fallback 保证离线确定性；模型可用时走 structured output |
| 并发领域 Agents | `orchestrator.py` 的 `Send`、`domain_agents.py` | runner 只包一层计时观察器，真正执行已有 demo agents |
| Evidence Ledger | `evidence.py`、`EvidenceItem` | 检索文档和只读 SQL 被统一成带稳定 ID 的证据 |
| Critic / Gate | `audit_handoffs()`、`decide_gate()` | 缺证据/失败可有限重试，写 SQL 直接 `DENY` |
| benchmark | 现有 30 条 `deepreason_benchmark_cases.json` | 只说明离线路由集，不泛化为线上答案准确率 |

## 3 分钟中文讲解稿

### 0:00–0:20 开场

“我演示的是 SmartRecipe DeepReason 的本地端到端链路。问题是：推荐一道低辣鸡肉菜，并统计同类菜的平均烹饪时长。它同时需要菜谱检索和统计分析，所以比单一路由更能体现分层 Multi-Agent。为了保证面试现场稳定，领域返回值使用仓库已有的离线 adapter，但顶层编排、并发、证据和安全判断都是真实代码。”

### 0:20–0:50 Coordinator、Planner、Reviewer

“Coordinator 先接收目标，它不直接回答，而是调用 Planner 生成结构化 `ExecutionPlan`。这里拆出 `recipe-1` 和 `analytics-2`，每个任务都有领域、指令和是否必须提供证据。当前没有配置模型，所以走确定性的 heuristic fallback；线上可以切换为 LLM structured output。随后 Reviewer 独立复核完整计划，检查是否漏领域。本例覆盖完整，因此状态是 `APPROVED`。”

### 0:50–1:25 并发领域 Agent

“Reviewer 通过后，LangGraph 用 `Send` 在同一个 superstep 扇出两个任务。Trace 中两个 Agent 的开始和结束区间真实重叠：Recipe Agent 复用菜谱检索能力，Analytics Agent 复用只读 Text2SQL 能力。它们各自返回统一的 `Handoff`，包含状态、摘要、结构化输出、证据和耗时。这样一个 Agent 失败不会抹掉另一个成功结果，也可以只重试失败任务。”

### 1:25–2:00 Evidence Ledger

“接着看 Evidence Ledger。菜谱 Agent 的检索文档和 Analytics Agent 的 SELECT 语句被归一化成两条 `EvidenceItem`，都绑定任务 ID、来源类型和稳定哈希。稳定 ID 让账本追加时可以去重，也让最终答案能追溯到具体文档或 SQL，而不只是保留一段模型文本。”

### 2:00–2:30 Critic 与安全 Gate

“Critic 在汇总前做确定性审计：检查任务失败、证据缺失，并二次扫描 SQL。这个查询只有 SELECT 且两项证据齐全，所以 Gate 是 `ALLOW`。如果出现临时失败或缺证据，Gate 会在预算内 `RETRY`；如果发现 DELETE、UPDATE 等写操作，则直接 `DENY`。安全底线由代码执行，不交给 LLM 自己判断。”

### 2:30–2:50 Synthesizer 与 benchmark

“只有通过 Gate 的成功 Handoff 才会进入 Synthesizer，最终合并为菜谱推荐和平均 25 分钟两个结论。脚本最后还复用了仓库现有 30 条路由 benchmark，本地结果是 30/30。这个指标只代表自建离线集合上的领域集合匹配率，不等同于线上答案准确率。”

### 2:50–3:00 收尾

“这套设计的核心是把控制面、领域执行、证据和安全发布分开：既复用原有 GraphRAG 与 Text2SQL，又获得并发、失败隔离、可审计和有界重试，而且没有重构已有架构。”

## 面试前验证

```powershell
uv run pytest tests/test_deepreason_demo.py tests/test_deepreason_orchestrator.py tests/test_deepreason_benchmark.py -q
uv run python scripts/run_deepreason_demo.py
```

需要展示拒绝分支时，可直接讲解现有测试 `test_unsafe_sql_evidence_is_denied`：它让 Analytics Agent 返回 `DELETE FROM recipes`，Critic 产生 `unsafe_sql` finding，Gate 最终为 `DENY`。

