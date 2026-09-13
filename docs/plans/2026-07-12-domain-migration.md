# SmartRecipe DeepReason Domain Migration Implementation Plan

> **For Claude:** Use the Executing Plans skill to implement this plan task-by-task.

**Goal:** Build a standalone SmartRecipe copy that uses DeepReason-style multi-agent planning, structured handoffs, evidence auditing, gates, and multi-domain execution while preserving GustoBot's existing recipe capabilities.

**Architecture:** Keep GustoBot's FastAPI service and existing GraphRAG, Text2Cypher, Text2SQL, image, and file implementations as domain capabilities. Add an asynchronous orchestration layer with coordinator/reviewer routing, task decomposition, concurrent domain-agent execution, evidence normalization, critic verification, risk gates, and final synthesis. The existing LangGraph remains available as a compatibility path.

**Tech Stack:** Python 3.12, FastAPI, Pydantic, LangChain structured output, LangGraph, Neo4j, Milvus, MySQL, Redis, pytest.

---

### Task 1: Domain contracts and workflow state

**Files:**
- Create: `gustobot/application/deepreason/models.py`
- Test: `tests/test_deepreason_models.py`

1. Write tests for structured task, handoff, evidence, gate, and workflow-result contracts.
2. Run the focused test and verify it fails because the package does not exist.
3. Implement enums and Pydantic models with validation and serializable defaults.
4. Run the focused test and verify it passes.

### Task 2: Coordinator, reviewer, and planner

**Files:**
- Create: `gustobot/application/deepreason/planning.py`
- Create: `gustobot/application/deepreason/prompts.py`
- Test: `tests/test_deepreason_planning.py`

1. Write tests for single-domain and cross-domain task decomposition.
2. Implement deterministic routing fallback for recipe retrieval, analytics, vision, file, and general tasks.
3. Add optional LLM structured-output coordinator/reviewer/planner calls.
4. Ensure invalid LLM output falls back without breaking the workflow.

### Task 3: GustoBot domain adapters

**Files:**
- Create: `gustobot/application/deepreason/domain_agents.py`
- Test: `tests/test_deepreason_domain_agents.py`

1. Define a domain-agent protocol and registry.
2. Wrap the existing RAG subgraph, Text2SQL subgraph, image handler, file handler, and general handler.
3. Normalize every result into a structured handoff and evidence list.
4. Add timeout and exception isolation so one failed subtask does not erase successful results.

### Task 4: Evidence, critic, gate, and orchestrator

**Files:**
- Create: `gustobot/application/deepreason/evidence.py`
- Create: `gustobot/application/deepreason/orchestrator.py`
- Test: `tests/test_deepreason_orchestrator.py`

1. Write tests proving two independent tasks execute concurrently and both results are retained.
2. Implement an append-only evidence ledger with stable hashes.
3. Implement critic findings for failed tasks, missing evidence, unsafe database output, and conflicting results.
4. Implement allow/retry/deny gate decisions with bounded retry.
5. Synthesize a final response from successful handoffs and expose a complete trace.

### Task 5: FastAPI integration and observability

**Files:**
- Create: `gustobot/interfaces/http/deepreason.py`
- Modify: `gustobot/interfaces/http/router.py`
- Modify: `main.py`
- Test: `tests/test_deepreason_api.py`

1. Add `/api/deepreason/chat` and `/api/deepreason/status` endpoints.
2. Preserve session, image, and file inputs.
3. Return route, plan, handoffs, evidence, gate decision, timing, and answer.
4. Add health/status output without exposing secrets.

### Task 6: Benchmark, documentation, and interview package

**Files:**
- Create: `tests/test_deepreason_benchmark.py`
- Create: `docs/deepreason-learning-guide.md`
- Create: `docs/architecture-deepreason.md`
- Create: `docs/resume-and-interview.md`
- Modify: `README.md`

1. Add reproducible routing, multi-agent completion, gate, and latency benchmark cases.
2. Document the code-level data flow and a three-day learning path.
3. Document defensible resume bullets and interview answers tied to code and benchmark output.
4. Run focused tests, the complete test suite, and a FastAPI smoke test.

