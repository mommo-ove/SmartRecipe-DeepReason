# SmartRecipe Constrained Meal Planner Implementation Plan

> **For Claude:** Use ${SUPERPOWERS_SKILLS_ROOT}/skills/collaboration/executing-plans/SKILL.md to implement this plan task-by-task.

**Goal:** Upgrade SmartRecipe from recipe Q&A into a tested seven-day meal-planning Agent that retrieves eligible recipes, solves hard nutrition/safety constraints, explains infeasible requests, supports local replanning, and binds its conclusions to evidence.

**Architecture:** Keep the existing DeepReason Router/Coordinator/ready-task scheduler. Add a deterministic meal-planning domain behind it: typed constraints → safety filter → BM25/Chroma/Neo4j candidate retrieval → recipe-level RRF → MySQL fact hydration → OR-Tools CP-SAT → deterministic verification → evidence-backed response. Chroma runs as a local Docker service and the application uses the HTTP-only client, avoiding native HNSW conflicts with the preserved legacy Milvus client. LLMs parse language and explain results; dependency rules, database safety, constraint solving, and release gates remain deterministic code.

**Tech Stack:** Python 3.12, Pydantic 2, LangGraph, OR-Tools CP-SAT, Chroma Server/HTTP Client, rank-bm25, SQLAlchemy/MySQL, Neo4j, pytest, Docker Compose.

---

## Delivery rules

- Preserve the existing CLIP encoder, image corpus, Milvus image repository, and DeepReason scheduler.
- Do not describe a capability as complete until its listed tests and real integration command pass.
- Unit tests may use deterministic fake embeddings or in-memory repositories, but their status must say `Fake Client`.
- Allergen and hard-constraint violations are release blockers; an LLM cannot override them.
- Build the first benchmark from versioned gold labels. LLMs may paraphrase user queries but cannot create gold answers.
- Commit after each task so that failures are easy to localize and revert.

## Status labels used during execution

- `Implemented + tested`: deterministic implementation and automated tests pass.
- `Implemented + Fake Client`: integration boundary exists, but the external service/model is replaced by a fake.
- `Real integration passed`: local Chroma/MySQL/Neo4j/API command has run successfully.
- `Planned only`: no completion claim is allowed.

---

### Task 1: Add meal-planning domain models and dependencies

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `gustobot/application/meal_planning/__init__.py`
- Create: `gustobot/application/meal_planning/models.py`
- Test: `tests/test_meal_planning_models.py`

**Step 1: Write the failing model tests**

Cover these invariants:

```python
def test_constraints_reject_inverted_calorie_range():
    with pytest.raises(ValidationError):
        MealPlanConstraints(daily_calories_min=1800, daily_calories_max=1500)


def test_recipe_is_not_eligible_without_normalized_nutrition():
    recipe = RecipeCandidate(
        recipe_id="r1",
        name="番茄炒蛋",
        calories_kcal=None,
        protein_g=20,
        total_minutes=15,
        estimated_cost_cents=1200,
        allergens=[],
        source_refs=["recipe:r1"],
    )
    assert recipe.planning_eligible is False
```

**Step 2: Run the test and confirm RED**

Run: `uv run pytest tests/test_meal_planning_models.py -q`

Expected: import failure because `meal_planning.models` does not exist.

**Step 3: Add dependencies**

Run:

```powershell
uv add "httpx>=0.27,<0.28" "ortools==9.9.3963" "chromadb-client>=1,<2"
```

**Step 4: Implement the smallest typed model layer**

Create Pydantic models/enums for:

- `MealSlot`: breakfast, lunch, dinner.
- `MealPlanConstraints`: calorie interval, minimum protein, excluded allergens, max cooking time, weekly budget, repetition limits.
- `RecipeCandidate`: normalized nutrition, time, cost, allergens, preference tags, source refs, `planning_eligible`.
- `MealAssignment`, `DailyPlan`, `MealPlan`.
- `SolveStatus`: optimal, feasible, infeasible, unknown.

Important: keep money as integer cents and scale nutrient decimals before passing them to CP-SAT.

**Step 5: Run GREEN and regression tests**

Run:

```powershell
uv run pytest tests/test_meal_planning_models.py -q
uv run pytest -q
```

Expected: model tests pass; existing 143-test baseline remains green.

**Step 6: Commit**

```powershell
git add pyproject.toml uv.lock gustobot/application/meal_planning tests/test_meal_planning_models.py
git commit -m "feat: add typed meal planning domain"
```

---

### Task 2: Create planning-eligible data quality gates and seed fixtures

**Files:**
- Create: `gustobot/application/meal_planning/quality.py`
- Create: `gustobot/application/meal_planning/fixtures.py`
- Create: `gustobot/data/meal_planning/recipes.v1.json`
- Create: `gustobot/data/meal_planning/manifest.v1.json`
- Test: `tests/test_meal_planning_quality.py`

**Step 1: Write failing tests**

Test that planning eligibility rejects:

- unknown calories or protein;
- quantities that were not normalized to grams/millilitres;
- unknown allergen status;
- missing source lineage;
- duplicate recipe IDs.

Test that non-eligible recipes remain searchable for ordinary Q&A but never enter the solver.

**Step 2: Run RED**

Run: `uv run pytest tests/test_meal_planning_quality.py -q`

Expected: import failure for `quality.py`.

**Step 3: Implement the validator and deterministic starter corpus**

Use a small, human-reviewable seed (12–20 recipes) for solver development. Each record must contain:

```json
{
  "recipe_id": "r101",
  "name": "西兰花鸡胸肉",
  "meal_types": ["lunch", "dinner"],
  "calories_kcal": 430,
  "protein_g": 42,
  "total_minutes": 25,
  "estimated_cost_cents": 1600,
  "allergens": [],
  "ingredients_normalized": [{"ingredient_id": "chicken_breast", "grams": 150}],
  "source_refs": ["recipe:r101", "nutrition:fdc:171077"],
  "planning_eligible": true
}
```

The manifest stores dataset version, record count, SHA-256, human reviewer, and generation time. Do not pretend this seed is the final 60–100 recipe dataset.

**Step 4: Run GREEN**

Run: `uv run pytest tests/test_meal_planning_quality.py -q`

**Step 5: Commit**

```powershell
git add gustobot/application/meal_planning gustobot/data/meal_planning tests/test_meal_planning_quality.py
git commit -m "feat: gate recipes by planning data quality"
```

---

### Task 3: Implement the feasible CP-SAT solver

**Files:**
- Create: `gustobot/application/meal_planning/solver.py`
- Test: `tests/test_meal_planning_solver.py`

**Step 1: Write a failing feasible-case test**

```python
def test_solver_builds_seven_day_plan_satisfying_hard_constraints():
    result = solver.solve(constraints, eligible_recipes)
    assert result.status in {SolveStatus.OPTIMAL, SolveStatus.FEASIBLE}
    assert len(result.plan.days) == 7
    assert all(constraints.daily_calories_min <= d.calories_kcal <= constraints.daily_calories_max for d in result.plan.days)
    assert all(d.protein_g >= constraints.daily_protein_min_g for d in result.plan.days)
    assert result.plan.total_cost_cents <= constraints.weekly_budget_cents
```

Also assert exactly one recipe per meal slot, allergen exclusion, time limit, and recipe repetition limit.

**Step 2: Run RED**

Run: `uv run pytest tests/test_meal_planning_solver.py -q`

Expected: solver import failure.

**Step 3: Implement binary decision variables and hard constraints**

Use `x[day, slot, recipe] ∈ {0,1}`. Add candidate variables only when the recipe supports the slot and has already passed the allergen/data-quality filter. Convert nutrient values to scaled integers.

Return status, selected assignments, solver wall time, objective values, and solver evidence. Never convert `UNKNOWN` into `INFEASIBLE`.

**Step 4: Run GREEN and a focused mutation**

Run:

```powershell
uv run pytest tests/test_meal_planning_solver.py -q
uv run pytest tests/test_meal_planning_solver.py -q -k allergen
```

Temporarily mark every lunch candidate with the excluded allergen and confirm the feasible test becomes infeasible; revert the test mutation afterward.

**Step 5: Commit**

```powershell
git add gustobot/application/meal_planning/solver.py tests/test_meal_planning_solver.py
git commit -m "feat: solve hard meal plan constraints with cp-sat"
```

---

### Task 4: Add staged optimization and local replanning

**Files:**
- Modify: `gustobot/application/meal_planning/solver.py`
- Create: `gustobot/application/meal_planning/replanning.py`
- Test: `tests/test_meal_planning_replanning.py`

**Step 1: Write failing tests**

Test that:

- hard feasibility always wins over preference;
- among feasible plans, preferred tags improve before variety/cost tie-breakers;
- replacing Tuesday dinner locks all other assignments;
- an impossible locked replan returns conflict details and does not silently rewrite the week.

**Step 2: Run RED**

Run: `uv run pytest tests/test_meal_planning_replanning.py -q`

**Step 3: Implement lexicographic/staged objectives**

Avoid one opaque weighted sum. Solve in stages:

1. satisfy all hard constraints;
2. maximize preference score while fixing feasibility;
3. minimize repeated recipes/main ingredients;
4. minimize cost and unnecessary changes.

For local replanning, add equality constraints for unaffected assignments and an explicit change-count objective.

**Step 4: Run GREEN**

Run: `uv run pytest tests/test_meal_planning_replanning.py -q`

**Step 5: Commit**

```powershell
git add gustobot/application/meal_planning/solver.py gustobot/application/meal_planning/replanning.py tests/test_meal_planning_replanning.py
git commit -m "feat: optimize preferences and support local replanning"
```

---

### Task 5: Diagnose infeasible plans and verify every solved plan

**Files:**
- Create: `gustobot/application/meal_planning/diagnostics.py`
- Create: `gustobot/application/meal_planning/verification.py`
- Test: `tests/test_meal_planning_verification.py`

**Step 1: Write failing tests**

Cover:

- feasible plan passes all deterministic checks;
- one injected peanut recipe yields `DENY`;
- one altered calorie value yields `REPLAN`;
- impossible calorie + protein + budget constraints return a sufficient conflict set;
- `UNKNOWN` yields `RETRY`, not a fabricated plan.

**Step 2: Run RED**

Run: `uv run pytest tests/test_meal_planning_verification.py -q`

**Step 3: Add assumption literals and the verification gate**

Attach assumption literals to user-visible constraint groups, then map `sufficient_assumptions_for_infeasibility()` back to explanations. Call it a **sufficient conflict set**, not a guaranteed minimum unsatisfiable core.

Recalculate calories, protein, time, cost, allergens, assignment count, and repetition from selected recipe facts. Return only deterministic statuses:

- `ALLOW`: all checks pass;
- `RETRY`: infrastructure/timeout/unknown;
- `REPLAN`: candidate plan violates a fixable constraint;
- `CLARIFY`: required constraint is missing or ambiguous;
- `DENY`: safety violation such as allergen exposure.

**Step 4: Run GREEN**

Run: `uv run pytest tests/test_meal_planning_verification.py -q`

**Step 5: Commit**

```powershell
git add gustobot/application/meal_planning/diagnostics.py gustobot/application/meal_planning/verification.py tests/test_meal_planning_verification.py
git commit -m "feat: verify plans and explain infeasible constraints"
```

---

### Task 6: Add normalized MySQL schema and repository boundary

**Files:**
- Create: `gustobot/data/migrations/002_meal_planning.sql`
- Modify: `gustobot/data/init_mysql.sql`
- Create: `gustobot/infrastructure/persistence/__init__.py`
- Create: `gustobot/infrastructure/persistence/meal_planning_repository.py`
- Test: `tests/test_meal_planning_repository.py`
- Test: `tests/integration/test_mysql_meal_planning_repository.py`

**Step 1: Write failing repository contract tests**

The repository must fetch only the requested `recipe_id` closed set and map rows to `RecipeCandidate`. It must not allow a planning query without explicit IDs.

**Step 2: Run RED**

Run: `uv run pytest tests/test_meal_planning_repository.py -q`

**Step 3: Add additive schema migration**

Add normalized per-serving nutrient facts, ingredient amounts in grams, allergen status, cost/time fields, source lineage, and planning eligibility. Preserve existing tables and data.

**Step 4: Implement parameterized fixed queries**

Use SQLAlchemy bind parameters and a repository method such as:

```python
def get_planning_facts(self, recipe_ids: Sequence[str]) -> list[RecipeCandidate]:
    if not recipe_ids:
        return []
    # SELECT ... WHERE recipe_id IN :recipe_ids; no LLM-generated SQL here.
```

**Step 5: Run unit tests, then real MySQL integration**

Run:

```powershell
uv run pytest tests/test_meal_planning_repository.py -q
docker compose up -d mysql
uv run pytest tests/integration/test_mysql_meal_planning_repository.py -q -m integration
```

Expected: unit test passes with an in-memory/fake row source; integration test passes only after MySQL migration and seed are loaded. Mark status accurately if Docker is unavailable.

**Step 6: Commit**

```powershell
git add gustobot/data gustobot/infrastructure/persistence tests/test_meal_planning_repository.py tests/integration/test_mysql_meal_planning_repository.py
git commit -m "feat: persist normalized meal planning facts"
```

---

### Task 7: Add Chroma as the default local dense recipe store

**Files:**
- Modify: `gustobot/config/settings.py`
- Modify: `.env.example`
- Modify: `docker-compose.yml`
- Create: `gustobot/infrastructure/retrieval/__init__.py`
- Create: `gustobot/infrastructure/retrieval/vector_store.py`
- Create: `gustobot/infrastructure/retrieval/chroma_recipe_store.py`
- Test: `tests/test_chroma_recipe_store.py`
- Test: `tests/integration/test_chroma_recipe_store.py`

**Step 1: Write failing adapter contract tests**

Use a deterministic fake embedding function and a recording fake Chroma client for the application-level contract. Test upsert, metadata filters, top-k ordering, and recipe/chunk metadata. Label this test `Implemented + Fake Client`.

**Step 2: Run RED**

Run: `uv run pytest tests/test_chroma_recipe_store.py -q`

**Step 3: Implement the port and adapter**

The application port exposes `upsert(chunks)` and `search(query, top_k, filters)`. The application uses `chromadb.HttpClient`; Chroma Server and its HNSW index run in Docker. The adapter stores:

- `chunk_id` as row/document ID;
- `recipe_id` as grouping metadata;
- human-readable chunk text;
- planning eligibility and meal type metadata;
- embedding vector.

Do not delete the existing Milvus adapter or image collection. Add a health check and persistent Docker volume for Chroma.

**Step 4: Run GREEN and the real persistence experiment**

Run:

```powershell
uv run pytest tests/test_chroma_recipe_store.py -q
docker compose up -d chroma
uv run pytest tests/integration/test_chroma_recipe_store.py -q -m integration
```

Restart the Chroma container and verify that the collection remains available. Then change one query vector in the test and predict which recipe moves to rank 1 before rerunning it.

**Step 5: Commit**

```powershell
git add gustobot/config/settings.py .env.example docker-compose.yml gustobot/infrastructure/retrieval tests/test_chroma_recipe_store.py tests/integration/test_chroma_recipe_store.py
git commit -m "feat: add local chroma recipe retrieval"
```

---

### Task 8: Implement recipe-level RRF and evidence-preserving retrieval

**Files:**
- Create: `gustobot/application/retrieval/__init__.py`
- Create: `gustobot/application/retrieval/models.py`
- Create: `gustobot/application/retrieval/fusion.py`
- Modify: `gustobot/application/agents/rag_sub_graph/components/graph_rag/rag_modules/hybrid_retrieval.py`
- Test: `tests/test_recipe_rrf.py`

**Step 1: Write failing ranking tests**

Cover the long-document bias case:

```python
def test_each_route_contributes_once_per_recipe():
    bm25 = [hit("r1", "c1", 1), hit("r1", "c2", 2), hit("r2", "c3", 3)]
    dense = [hit("r2", "c4", 1)]
    fused = reciprocal_rank_fusion({"bm25": bm25, "dense": dense}, k=60)
    assert fused[0].recipe_id == "r2"
    assert len(fused_by_id(fused, "r1").evidence_chunks) == 2
```

Also test missing routes, stable tie-breaking, and route-specific rank evidence.

**Step 2: Run RED**

Run: `uv run pytest tests/test_recipe_rrf.py -q`

**Step 3: Implement per-route dedupe and recipe-level fusion**

For each route, only the best-ranked chunk contributes `1 / (k + rank)` to a recipe. Preserve additional chunks as evidence, but never let them add extra RRF score. Replace `node_id` fusion in the legacy path with stable `recipe_id` grouping.

**Step 4: Run GREEN and legacy retrieval regression**

Run:

```powershell
uv run pytest tests/test_recipe_rrf.py -q
uv run pytest -q -k "retrieval or rag"
```

**Step 5: Commit**

```powershell
git add gustobot/application/retrieval gustobot/application/agents/rag_sub_graph/components/graph_rag/rag_modules/hybrid_retrieval.py tests/test_recipe_rrf.py
git commit -m "feat: fuse retrieval at recipe level"
```

---

### Task 9: Compose retrieval, fact hydration, solving, and verification

**Files:**
- Create: `gustobot/application/meal_planning/service.py`
- Create: `gustobot/application/meal_planning/ports.py`
- Test: `tests/test_meal_planning_service.py`

**Step 1: Write a failing orchestration test**

Use recording fakes to assert this exact order:

1. safety filter and retrieval;
2. recipe-level fusion;
3. hydrate only selected recipe IDs from MySQL;
4. solve;
5. verify;
6. return plan plus evidence.

Also test that empty retrieval stops before the solver and that failed verification never reaches answer generation.

**Step 2: Run RED**

Run: `uv run pytest tests/test_meal_planning_service.py -q`

**Step 3: Implement the application service**

Keep external systems behind typed ports. The service should return a domain result, not prose. Include selected recipe IDs, retrieval trace, hydrated fact IDs, solver status, verification decision, and evidence IDs.

**Step 4: Run GREEN**

Run: `uv run pytest tests/test_meal_planning_service.py -q`

**Step 5: Commit**

```powershell
git add gustobot/application/meal_planning/service.py gustobot/application/meal_planning/ports.py tests/test_meal_planning_service.py
git commit -m "feat: compose verified meal planning pipeline"
```

---

### Task 10: Integrate the meal-planning capability into DeepReason

**Files:**
- Modify: `gustobot/application/deepreason/models.py`
- Modify: `gustobot/application/deepreason/planning.py`
- Modify: `gustobot/application/deepreason/direct_capabilities.py`
- Modify: `gustobot/application/deepreason/domain_agents.py`
- Modify: `gustobot/application/deepreason/orchestrator.py`
- Test: `tests/test_deepreason_meal_planning.py`
- Test: `tests/test_deepreason_scheduling.py`

**Step 1: Write failing dependency tests**

Assert a complex plan has dependencies such as:

```text
recipe-retrieval ──→ nutrition-hydration ──→ meal-plan-solver ──→ verification
```

The first scheduling wave may dispatch independent BM25/Dense/graph retrieval work, but must not dispatch nutrition or the solver. Failed retrieval must skip/block downstream work with a structured dependency error.

**Step 2: Run RED**

Run:

```powershell
uv run pytest tests/test_deepreason_meal_planning.py -q
uv run pytest tests/test_deepreason_scheduling.py -q
```

**Step 3: Add the domain and deterministic executor**

Add `Domain.MEAL_PLANNING`. The Router may choose direct, plan, or clarify. The Coordinator creates task dependencies, but the existing deterministic scheduler alone decides which pending tasks are ready. The meal-planning domain executor calls `MealPlanningService`; it is not another free-form LLM Agent.

Extend handoff output with typed `selected_recipe_ids`, constraints, solve status, verification status, and evidence IDs. Never satisfy planning evidence requirements with generic `agent_output` evidence.

**Step 4: Run GREEN and all DeepReason tests**

Run: `uv run pytest tests/test_deepreason_meal_planning.py tests/test_deepreason_scheduling.py tests/test_deepreason_orchestrator.py -q`

**Step 5: Commit**

```powershell
git add gustobot/application/deepreason tests/test_deepreason_meal_planning.py tests/test_deepreason_scheduling.py
git commit -m "feat: integrate constrained planning into deepreason"
```

---

### Task 11: Upgrade evidence and critic decisions for planning

**Files:**
- Modify: `gustobot/application/deepreason/models.py`
- Modify: `gustobot/application/deepreason/evidence.py`
- Modify: `gustobot/application/deepreason/orchestrator.py`
- Create: `gustobot/application/meal_planning/evidence.py`
- Test: `tests/test_meal_planning_evidence.py`
- Test: `tests/test_deepreason_critic_gate.py`

**Step 1: Write failing evidence tests**

Assert that each answer claim about a recipe, nutrient total, budget, allergen, or solver decision references an atomic evidence ID. Verify that deterministic arithmetic and allergen checks run before any optional semantic LLM check.

**Step 2: Run RED**

Run: `uv run pytest tests/test_meal_planning_evidence.py tests/test_deepreason_critic_gate.py -q`

**Step 3: Implement typed evidence and richer gate statuses**

Add source types such as `recipe_chunk`, `mysql_nutrition`, `neo4j_relation`, `solver_assignment`, and `verification_check`. Extend gate handling carefully to `ALLOW/RETRY/REPLAN/CLARIFY/DENY` without changing existing non-planning behaviour.

For semantic claims, an LLM may classify support, but it receives only atomic claims and candidate evidence. The deterministic gate remains authoritative for numeric and safety claims.

**Step 4: Run GREEN**

Run: `uv run pytest tests/test_meal_planning_evidence.py tests/test_deepreason_critic_gate.py -q`

**Step 5: Commit**

```powershell
git add gustobot/application/deepreason gustobot/application/meal_planning/evidence.py tests/test_meal_planning_evidence.py tests/test_deepreason_critic_gate.py
git commit -m "feat: ground meal plan claims in atomic evidence"
```

---

### Task 12: Add Neo4j relationship schema and a safe Text2Cypher guard

**Files:**
- Create: `gustobot/data/neo4j/meal_planning_schema.cypher`
- Create: `gustobot/data/neo4j/meal_planning_seed.cypher`
- Create: `gustobot/application/agents/rag_sub_graph/components/text2cypher/cypher_guard.py`
- Modify: `gustobot/application/agents/rag_sub_graph/components/text2cypher/node.py`
- Test: `tests/test_cypher_guard.py`
- Test: `tests/integration/test_neo4j_meal_planning.py`

**Step 1: Write failing guard tests**

Allow parameterized read queries over approved labels/relations. Reject writes, unrestricted procedures, unknown labels/relations, missing `LIMIT`, string interpolation, and excessive limits.

**Step 2: Run RED**

Run: `uv run pytest tests/test_cypher_guard.py -q`

**Step 3: Implement defense in depth**

The execution order must be:

1. parse/tokenize and enforce read-only rules;
2. enforce schema/function/procedure allowlists;
3. require parameters and bounded `LIMIT`;
4. run Neo4j `EXPLAIN`;
5. execute with timeout/read account;
6. bounded repair or fixed-template fallback.

Use fixed Cypher templates for allergen/substitution/meal-type relations. Reserve dynamic Text2Cypher for long-tail relationship questions.

**Step 4: Run unit and real integration tests**

Run:

```powershell
uv run pytest tests/test_cypher_guard.py -q
docker compose up -d neo4j
uv run pytest tests/integration/test_neo4j_meal_planning.py -q -m integration
```

**Step 5: Commit**

```powershell
git add gustobot/data/neo4j gustobot/application/agents/rag_sub_graph/components/text2cypher tests/test_cypher_guard.py tests/integration/test_neo4j_meal_planning.py
git commit -m "feat: secure meal planning graph queries"
```

---

### Task 13: Build a versioned benchmark and ablation runner

**Files:**
- Create: `gustobot/application/meal_planning/benchmark.py`
- Create: `benchmark/meal_planning/cases.v1.jsonl`
- Create: `benchmark/meal_planning/manifest.v1.json`
- Create: `scripts/run_meal_planning_benchmark.py`
- Test: `tests/test_meal_planning_benchmark.py`

**Step 1: Write failing benchmark tests**

Verify:

- manifest version/hash/sample counts;
- no duplicate case IDs;
- retrieval cases have gold `recipe_ids`;
- planning cases have expected feasible/infeasible labels;
- results are computed from raw case outputs, never hard-coded;
- ablations have comparable case sets.

**Step 2: Run RED**

Run: `uv run pytest tests/test_meal_planning_benchmark.py -q`

**Step 3: Implement 100-case schema and runner**

Target composition:

- 40 retrieval cases;
- 20 Text2Cypher cases;
- 30 planning cases (20 feasible, 10 infeasible);
- 10 local replanning cases.

At least 30 cases must be human-authored/reviewed. Until all 100 are present, the manifest must expose the actual count and status `draft`; no resume metric may be reported.

Compute:

- retrieval Recall@5/10, MRR, NDCG@10 and RRF/rerank ablations;
- Text2Cypher syntax, execution, result-set match, and safety rejection;
- feasibility classification, hard-constraint satisfaction, allergen violations, objective and latency;
- local replan preservation/change count;
- evidence coverage, end-to-end success, P50/P95 latency, and token/call counts.

**Step 4: Run GREEN and generate raw results**

Run:

```powershell
uv run pytest tests/test_meal_planning_benchmark.py -q
uv run python scripts/run_meal_planning_benchmark.py --mode fixture --output benchmark/meal_planning/results/latest.json
```

The fixture mode proves metric plumbing, not real model quality. Label it `Fake Client`.

**Step 5: Commit**

```powershell
git add gustobot/application/meal_planning/benchmark.py benchmark/meal_planning scripts/run_meal_planning_benchmark.py tests/test_meal_planning_benchmark.py
git commit -m "feat: add versioned meal planning benchmark"
```

---

### Task 14: Add API/demo, trace output, and real smoke test

**Files:**
- Create: `gustobot/application/services/meal_planning_service.py`
- Create: `gustobot/interfaces/http/meal_planning.py`
- Modify: FastAPI router registration file discovered during implementation
- Create: `scripts/run_meal_planning_demo.py`
- Create: `docs/meal-planning-runtime.md`
- Test: `tests/test_meal_planning_api.py`
- Test: `tests/integration/test_meal_planning_e2e.py`

**Step 1: Write failing API tests**

Test one feasible request, one clarify request, one infeasible request, and one local replan. The response must expose `run_id`, solve/gate status, plan, evidence IDs, and a trace summary.

**Step 2: Run RED**

Run: `uv run pytest tests/test_meal_planning_api.py -q`

**Step 3: Implement the thin API and runnable demo**

Keep HTTP validation separate from domain logic. Persist trace events for routing, each scheduling wave, retrieval ranks, selected recipe IDs, SQL hydration, solver status, verification findings, retries, and final response.

**Step 4: Run unit and real smoke tests**

Run:

```powershell
uv run pytest tests/test_meal_planning_api.py -q
docker compose up -d mysql neo4j redis
uv run python scripts/run_meal_planning_demo.py --query "为我制定一份七天减脂餐单，每天1600到1800千卡，蛋白质至少100克，不含花生，每餐30分钟内，每周预算300元"
uv run pytest tests/integration/test_meal_planning_e2e.py -q -m integration
uv run pytest -q
```

Expected: API/domain tests pass, demo returns either a verified plan or a structured infeasibility explanation, and the full suite remains green.

**Step 5: Record truthful status**

Update `docs/meal-planning-runtime.md` with a matrix showing:

- deterministic unit-tested pieces;
- Fake Client boundaries;
- local services actually integrated;
- external LLM/reranker calls actually run;
- benchmark sample count and measured results.

**Step 6: Commit**

```powershell
git add gustobot/application/services/meal_planning_service.py gustobot/interfaces/http scripts/run_meal_planning_demo.py docs/meal-planning-runtime.md tests/test_meal_planning_api.py tests/integration/test_meal_planning_e2e.py
git commit -m "feat: expose verified meal planning workflow"
```

---

## Final verification gate

Run all of the following before calling the upgrade complete:

```powershell
uv run pytest -q
uv run python scripts/run_meal_planning_benchmark.py --mode real --output benchmark/meal_planning/results/real-latest.json
git diff --check
git status --short
```

Completion requires:

- zero allergen violations in benchmark planning cases;
- zero released hard-constraint violations;
- real MySQL, Neo4j, and local Chroma smoke tests passed;
- raw benchmark results committed or archived with version/hash;
- no claim that fixture/fake-client results are real integration metrics;
- existing multimodal and dependency-scheduling tests remain green.

## Deferred scope

Do not let these delay the core planner:

- CLIP-based meal-image similarity;
- VLM/OCR nutrition-label ingestion;
- generated meal-card illustrations;
- production Milvus cluster;
- medical nutrition plans;
- reinforcement learning.

They can be added only after the deterministic planning/evaluation loop is stable.
