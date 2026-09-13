# Validated Text2Cypher Retrieval Implementation Plan

> **For Claude:** Use `${SUPERPOWERS_SKILLS_ROOT}/skills/collaboration/executing-plans/SKILL.md` to implement this plan task-by-task.

**Goal:** Build a template-first, dynamically generated and deterministically validated Text2Cypher path for the current `Recipe-[:HAS_INGREDIENT]->Ingredient` graph, then connect its legal `recipe_id` output to hybrid retrieval, Evidence Ledger and meal planning.

**Architecture:** High-frequency structured requests use parameterized Cypher templates. Long-tail requests use DeepSeek with the live graph schema and relevant few-shot examples, but generated statements cannot execute until they pass read-only, schema, relationship, return-contract and Neo4j `EXPLAIN` checks. Failed statements may be repaired at most twice; successful results are normalized into `selected_recipe_ids`, rechecked by Graph Gate and recorded as evidence.

**Tech Stack:** Python 3.12, Pydantic, Neo4j Python Driver, LangChain `ChatOpenAI`/DeepSeek, pytest, existing BM25/BGE-M3/RRF retrieval, existing Evidence Ledger, OR-Tools CP-SAT, Gradio evaluation console.

---

## Measured baseline and target outputs

Already measured on 300 Food.com recipes and the frozen 40-case graph relation set:

- BM25 Recall@20: 89.44%; forbidden-ingredient violation@20: 30.33%.
- BGE-M3 Recall@20: 34.17%; violation@20: 14.17%.
- BM25 + Dense + RRF Recall@20: 77.78%; violation@20: 26.00%.
- Existing deterministic graph gate Recall@20: 100%; violation@20: 0.00%.

The implementation must generate, not assume, these new metrics:

- template route accuracy;
- Cypher generation format rate;
- schema-validation pass rate;
- `EXPLAIN` pass rate;
- execution success rate;
- logical result accuracy against Gold `recipe_id` sets;
- unsafe-query rejection rate;
- repair success rate and average repair count;
- end-to-end constraint violation rate and p50/p95 latency.

### Task 1: Define the new-schema Text2Cypher contracts

**Files:**
- Create: `gustobot/application/meal_planning/text2cypher/__init__.py`
- Create: `gustobot/application/meal_planning/text2cypher/models.py`
- Test: `tests/test_meal_text2cypher_models.py`

**Step 1: Write failing model tests**

Cover `CypherSource`, `CypherCandidate`, `ValidationIssue`, `ValidationReport`, `CypherExecutionResult` and `Text2CypherResult`. Require a non-empty statement, parameter dictionary, trace steps, normalized `selected_recipe_ids`, attempt count in `0..2`, and explicit failure status.

**Step 2: Verify RED**

Run: `uv run pytest tests/test_meal_text2cypher_models.py -q`

Expected: collection fails because the package does not exist.

**Step 3: Implement the minimal Pydantic contracts**

Do not import the legacy `Concept` schema. All result records must use stable `recipe_id`, never infer identity from recipe name.

**Step 4: Verify GREEN**

Run: `uv run pytest tests/test_meal_text2cypher_models.py -q`

Expected: all model tests pass.

### Task 2: Add parameterized fixed templates and deterministic template routing

**Files:**
- Create: `gustobot/application/meal_planning/text2cypher/templates.py`
- Test: `tests/test_meal_text2cypher_templates.py`

**Step 1: Write failing tests**

Cover these initial high-frequency shapes:

1. required ingredients;
2. required plus excluded ingredients;
3. recipe properties by `recipe_id`;
4. ingredient list by `recipe_id`;
5. aggregate recipe count;
6. structured filters for meal type, maximum minutes and minimum protein.

Assert that user values appear only in `parameters`, never interpolated into the statement. Assert that unknown/ambiguous requests return no template match and therefore route to dynamic generation.

**Step 2: Verify RED**

Run: `uv run pytest tests/test_meal_text2cypher_templates.py -q`

Expected: missing template API.

**Step 3: Implement the minimal template catalog**

Use the current labels and relationship only:

```cypher
(:Recipe)-[:HAS_INGREDIENT]->(:Ingredient)
```

Every recipe-returning template must return `recipe.recipe_id AS recipe_id` and apply a bounded `$top_k`.

**Step 4: Verify GREEN**

Run the targeted test file and confirm it passes.

### Task 3: Implement deterministic read-only and return-contract validation

**Files:**
- Create: `gustobot/application/meal_planning/text2cypher/safety.py`
- Test: `tests/test_meal_text2cypher_safety.py`

**Step 1: Write failing safety tests**

Reject multiple statements and write/admin clauses including `CREATE`, `MERGE`, `DELETE`, `DETACH DELETE`, `SET`, `REMOVE`, `DROP`, `LOAD CSV`, `CALL`, `FOREACH` and transaction commands. Include adversarial casing, comments and quoted text. Require bounded `LIMIT` for recipe-list queries and standardized `recipe_id` output.

**Step 2: Verify RED**

Run: `uv run pytest tests/test_meal_text2cypher_safety.py -q`

Expected: missing validator.

**Step 3: Implement lexical normalization and validation**

Strip comments and string literals before scanning clauses so a property such as `created_at` or a quoted word does not trigger a false positive. Return structured issues rather than booleans.

**Step 4: Verify GREEN**

Run the targeted safety tests and confirm every malicious case is rejected.

### Task 4: Capture and validate the live Neo4j schema

**Files:**
- Create: `gustobot/application/meal_planning/text2cypher/schema.py`
- Modify: `gustobot/application/meal_planning/graph_retrieval.py`
- Test: `tests/test_meal_text2cypher_schema.py`
- Integration test: `tests/integration/test_neo4j_meal_text2cypher.py`

**Step 1: Write failing tests**

Build a `GraphSchemaSnapshot` containing allowed labels, properties and directed relationships. Test rejection of unknown labels, relationships and properties. Test the valid direction `Recipe -> HAS_INGREDIENT -> Ingredient`.

**Step 2: Verify RED**

Run: `uv run pytest tests/test_meal_text2cypher_schema.py -q`

**Step 3: Implement schema acquisition without requiring APOC**

Use Neo4j metadata procedures supported by the installed database, normalize the response, and cache it with an explicit refresh method. The application must not rely on prompt text as the source of truth.

**Step 4: Add real integration coverage**

Run with `MEAL_PLANNING_NEO4J_URL=bolt://localhost:17687` and verify that the 300-recipe graph reports `Recipe`, `Ingredient` and `HAS_INGREDIENT`.

### Task 5: Add Neo4j EXPLAIN and execution adapters

**Files:**
- Create: `gustobot/application/meal_planning/text2cypher/neo4j_adapter.py`
- Test: `tests/test_meal_text2cypher_neo4j_adapter.py`
- Integration test: `tests/integration/test_neo4j_meal_text2cypher.py`

**Step 1: Write failing tests**

Require `EXPLAIN <statement>` to run with the same parameters as execution. Test syntax failure, missing parameters, no execution after failed validation, result row normalization and timeout/error classification.

**Step 2: Verify RED**

Run the targeted unit test.

**Step 3: Implement the adapter using the existing Neo4j driver**

Keep `explain()` and `execute()` separate so tests can prove an invalid statement is never executed.

**Step 4: Verify against the real container**

Execute one valid template and one deliberately malformed statement. The valid query must return real `recipe_id` rows; the malformed one must fail at `EXPLAIN`.

### Task 6: Add DeepSeek generation and bounded repair

**Files:**
- Create: `gustobot/application/meal_planning/text2cypher/generator.py`
- Create: `gustobot/application/meal_planning/text2cypher/prompts.py`
- Test: `tests/test_meal_text2cypher_generator.py`

**Step 1: Write failing tests using an injected deterministic model**

Test prompt inputs: question, current schema, relevant examples and output contract. Test repair inputs: original question, invalid statement, structured validation issues and fresh schema. Test removal of Markdown fences. Do not test LangChain mock call counts.

**Step 2: Verify RED**

Run the targeted generator test.

**Step 3: Implement a small model protocol and DeepSeek adapter**

Reuse `get_llm()` and the configured `deepseek-v4-flash`. Keep the model dependency injectable so validators and orchestration remain deterministic in unit tests.

**Step 4: Verify GREEN**

Run the targeted tests without making a real API call.

### Task 7: Orchestrate template-first generation, validation, repair and execution

**Files:**
- Create: `gustobot/application/meal_planning/text2cypher/service.py`
- Test: `tests/test_meal_text2cypher_service.py`

**Step 1: Write failing state-machine tests**

Cover:

- template hit bypasses the LLM;
- template miss invokes dynamic generation;
- validation failure triggers repair;
- every repaired statement reruns the complete validation chain;
- maximum two repairs;
- unsafe statements are never sent to repair or execution;
- successful rows produce ordered, deduplicated `selected_recipe_ids`;
- empty result is distinct from execution failure.

**Step 2: Verify RED**

Run the targeted service test.

**Step 3: Implement the minimal deterministic state machine**

Persist trace stages such as `template_route`, `generate`, `validate`, `explain`, `repair`, `execute` and `graph_gate` with duration and status.

**Step 4: Verify GREEN**

Run the service tests and all Text2Cypher unit tests.

### Task 8: Connect legal recipe IDs to BM25/Dense routing and Graph Gate

**Files:**
- Modify: `gustobot/application/meal_planning/graph_retrieval.py`
- Create: `gustobot/application/meal_planning/retrieval_router.py`
- Test: `tests/test_meal_retrieval_router.py`

**Step 1: Write failing route tests**

Expected policy:

- hard relation/negative constraint -> Text2Cypher/graph only;
- exact keyword request -> BM25;
- fuzzy similarity/preference -> BM25 plus Dense;
- Dense is not automatically fused into every request;
- graph-produced allowed IDs are a hard candidate boundary;
- RRF cannot reintroduce a forbidden ID.

**Step 2: Verify RED**

Run the targeted router tests.

**Step 3: Implement routing and hard candidate filtering**

Do not rename this GraphRAG. It is graph-constrained hybrid retrieval.

**Step 4: Verify GREEN**

Run graph retrieval, hybrid retrieval and routing test files.

### Task 9: Emit atomic evidence and perform deterministic fact checks

**Files:**
- Modify: `gustobot/application/deepreason/domain_agents.py`
- Modify: `gustobot/application/deepreason/models.py`
- Create: `gustobot/application/meal_planning/fact_verification.py`
- Test: `tests/test_meal_fact_verification.py`
- Modify: `tests/test_deepreason_domain_agents.py`

**Step 1: Write failing evidence tests**

Require evidence for recipe existence, ingredients, time, nutrition and the exact Cypher query/result source. Verify numeric equality, entity existence, selected-ID membership, excluded-ingredient absence and claim-to-evidence coverage.

**Step 2: Verify RED**

Run the targeted tests.

**Step 3: Implement deterministic checks first**

LLM-based semantic claim review is optional and must not replace number/entity/constraint checks.

**Step 4: Verify GREEN**

Run evidence, domain-agent and fact-verification tests.

### Task 10: Feed verified candidates into CP-SAT meal planning

**Files:**
- Modify: `gustobot/application/meal_planning/agent.py`
- Modify: `gustobot/application/meal_planning/candidate_pool.py`
- Test: `tests/test_meal_planning_agent.py`
- Test: `tests/test_meal_planning_candidate_pool.py`

**Step 1: Write failing integration-style unit tests**

Ensure CP-SAT receives only candidates whose required capabilities and graph constraints are verified. Test forbidden candidate removal, candidate-pool insufficiency, targeted supplement retrieval and infeasible planning.

**Step 2: Verify RED**

Run the two targeted test files.

**Step 3: Implement the candidate-source adapter**

Keep CP-SAT deterministic; never ask the LLM to decide whether a hard constraint may be ignored.

**Step 4: Verify GREEN**

Run all meal-planning tests.

### Task 11: Build the frozen Text2Cypher and safety benchmark

**Files:**
- Create: `benchmark/meal_planning/text2cypher/cases.v1.jsonl`
- Create: `benchmark/meal_planning/text2cypher/security_cases.v1.jsonl`
- Create: `scripts/run_text2cypher_benchmark.py`
- Create: `tests/test_meal_text2cypher_benchmark.py`
- Create: `docs/evaluation/text2cypher-evaluation.md`

**Step 1: Write failing benchmark tests**

Require immutable case IDs, split, question, Gold recipe IDs, expected route and dataset hash. Reuse the 40 frozen relation questions as the starting Gold set; add separately versioned malformed/unsafe statements. Prevent test cases from silently entering prompt examples.

**Step 2: Verify RED**

Run the benchmark tests.

**Step 3: Implement four ablation configurations**

1. DeepSeek direct generation;
2. DeepSeek plus live schema;
3. DeepSeek plus schema and retrieved few-shot;
4. template-first plus complete validation and repair.

Record raw outputs and traces so every percentage can be audited.

**Step 4: Run development, freeze, then test**

Tune only on the development split. Run the test split once after the design is frozen. Save the JSON report, model name, prompt version, corpus hash and benchmark hash.

### Task 12: Extend the Gradio evaluation console

**Files:**
- Modify: `evaluation_ui/smartrecipe_eval_ui/core.py`
- Modify: `evaluation_ui/smartrecipe_eval_ui/app.py`
- Modify: `evaluation_ui/tests/test_core.py`
- Modify: `evaluation_ui/tests/test_app.py`

**Step 1: Write failing UI service tests**

Require tables for the four Text2Cypher configurations, repair trace inspection, unsafe-case inspection and raw report download.

**Step 2: Verify RED**

Run: `evaluation_ui/.venv/Scripts/python.exe -m pytest tests -q` from `evaluation_ui`.

**Step 3: Implement the UI tab**

The UI must call the benchmark script and must not hard-code metrics.

**Step 4: Browser verification**

Run one template-only smoke evaluation and inspect one repaired dynamic case and one rejected unsafe case.

### Task 13: Full verification and interview documentation

**Files:**
- Modify: `docs/interview/smartrecipe-100-questions.md`
- Create: `docs/interview/smartrecipe-text2cypher-review.md`

**Step 1: Run fresh verification**

```powershell
uv run pytest -q
cd evaluation_ui
.\.venv\Scripts\python.exe -m pytest tests -q
```

Then run the real Neo4j integration and real DeepSeek test benchmark. Do not report metrics from Fake Clients as model metrics.

**Step 2: Document actual results and limitations**

Include the final confusion/error taxonomy: ambiguous questions, entity mapping, schema mismatch, semantic logic error, empty result and API/Neo4j failure.

**Step 3: Produce teaching and interview material**

Add architecture flow, one end-to-end case, 30-second explanation, 2-minute explanation and three layers of follow-up questions. Label every capability as unit-tested, Fake Client, real Neo4j integration or real DeepSeek evaluation.

