# Multimodal Retrieval and Dependency Scheduler Implementation Plan

> **For Claude:** Use `${SUPERPOWERS_SKILLS_ROOT}/skills/collaboration/executing-plans/SKILL.md` to implement this plan task-by-task.

**Goal:** Build a reproducible image-to-recipe retrieval path and make DeepReason execute dependent Agent tasks only after their upstream Handoffs are available.

**Architecture:** Download a small licensed evaluation corpus from Wikimedia Commons, encode catalog and query images with a lazy-loaded local CLIP ViT-B/32 adapter, and store 512-dimensional vectors in a dedicated Milvus collection keyed by the existing Neo4j `recipe_id`. At request time, fuse VLM-to-text retrieval and CLIP image retrieval at recipe level with RRF. Separately, replace the current one-shot `Send` fan-out with deterministic ready-task scheduling and structured dependency outputs.

**Tech Stack:** Python 3.12, Pydantic, Transformers/PyTorch CLIP, Milvus, LangGraph `Send`, FastAPI, pytest.

---

### Task 1: Reproducible recipe image corpus

**Files:**
- Create: `configs/recipe_image_sources.json`
- Create: `scripts/download_recipe_images.py`
- Create: `gustobot/application/multimodal/__init__.py`
- Create: `gustobot/application/multimodal/catalog.py`
- Test: `tests/test_recipe_image_catalog.py`
- Modify: `.gitignore`

**Step 1: Write the failing catalog tests**

Test that every manifest row contains `recipe_id`, `recipe_name`, `commons_query`, `split`, and `license_allowlist`; reject duplicate `(recipe_id, split, ordinal)` values and recipe IDs not found in `gustobot/data/kg_output/concepts.csv`.

**Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_recipe_image_catalog.py -q`

Expected: FAIL because the catalog module and manifest do not exist.

**Step 3: Implement the catalog and downloader**

Use these existing graph IDs as the first labeled classes:

```python
RECIPES = {
    "201003834": "宫保鸡丁",
    "201000320": "西红柿炒鸡蛋",
    "201001857": "扬州炒饭",
    "201002121": "提拉米苏",
    "201003189": "牛排",
    "201005342": "清蒸鲈鱼",
    "201000285": "凉拌黄瓜",
    "201001395": "手工水饺",
    "201003908": "麻婆豆腐",
    "201003271": "糖醋排骨",
}
```

The downloader must call the Wikimedia Commons MediaWiki API, request `imageinfo` plus `extmetadata`, accept only CC/Public Domain licenses, verify MIME and image decoding with Pillow, calculate SHA-256, and write a generated `attribution.jsonl`. Keep downloaded binaries under ignored `gustobot/data/recipe_images/`; commit only the source manifest and attribution metadata.

**Step 4: Run tests and a two-recipe smoke download**

Run:

```powershell
python -m pytest tests/test_recipe_image_catalog.py -q
python scripts/download_recipe_images.py --limit-recipes 2 --images-per-split 1
```

Expected: tests PASS; downloaded files decode as RGB images and attribution rows contain source page, author, license, and checksum.

**Step 5: Commit**

```bash
git add .gitignore configs/recipe_image_sources.json scripts/download_recipe_images.py gustobot/application/multimodal tests/test_recipe_image_catalog.py
git commit -m "feat: add reproducible recipe image corpus"
```

### Task 2: Local CLIP encoder with a testable contract

**Files:**
- Create: `gustobot/application/multimodal/encoders.py`
- Test: `tests/test_multimodal_encoders.py`
- Modify: `pyproject.toml`
- Modify: `.env.example`
- Modify: `gustobot/config/settings.py`

**Step 1: Write failing contract tests**

Define an `ImageEncoder` protocol and test a fake encoder without loading a model. Test that `LocalClipEncoder` validates file existence, lazily loads the model once, returns a 512-float vector, L2-normalizes it, selects CUDA when available, and falls back to CPU.

**Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_multimodal_encoders.py -q`

Expected: FAIL because `encoders.py` does not exist.

**Step 3: Implement the minimal encoder**

```python
class ImageEncoder(Protocol):
    dimension: int
    model_version: str
    def encode_image(self, image_path: Path) -> list[float]: ...

class LocalClipEncoder:
    dimension = 512
    def __init__(self, model_name="openai/clip-vit-base-patch32", device="auto"): ...
```

Import `torch`, `CLIPModel`, and `CLIPProcessor` only inside `_ensure_loaded()` so normal API startup does not require the heavy optional dependency. Add a `multimodal` optional dependency group containing compatible `torch` and `transformers` versions.

**Step 4: Run unit tests**

Run: `python -m pytest tests/test_multimodal_encoders.py -q`

Expected: PASS without downloading model weights because model loading is injected/mocked at the boundary.

**Step 5: Commit**

```bash
git add pyproject.toml .env.example gustobot/config/settings.py gustobot/application/multimodal/encoders.py tests/test_multimodal_encoders.py
git commit -m "feat: add local CLIP image encoder"
```

### Task 3: Dedicated Milvus image-vector repository

**Files:**
- Create: `gustobot/infrastructure/persistence/image_vector_store.py`
- Test: `tests/test_image_vector_store.py`

**Step 1: Write failing adapter tests**

With a fake `MilvusClient`, assert that the adapter creates `recipe_image_embeddings` with `image_id` as string primary key, `image_vector` as a 512-dimensional vector, COSINE metric, and dynamic metadata fields. Assert that search requests return typed hits with `recipe_id`, `image_path`, distance, source page, and model version.

**Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_image_vector_store.py -q`

Expected: FAIL because the adapter does not exist.

**Step 3: Implement the adapter**

Expose only four operations: `ensure_collection()`, `upsert(records)`, `search(vector, top_k)`, and `stats()`. Do not let domain code import `pymilvus` directly. Use image checksum plus model version to make rebuilds idempotent.

**Step 4: Run tests**

Run: `python -m pytest tests/test_image_vector_store.py -q`

Expected: PASS.

**Step 5: Commit**

```bash
git add gustobot/infrastructure/persistence/image_vector_store.py tests/test_image_vector_store.py
git commit -m "feat: add Milvus image vector repository"
```

### Task 4: Offline image indexing command

**Files:**
- Create: `gustobot/infrastructure/persistence/rebuild_image_index.py`
- Create: `scripts/rebuild_image_index.py`
- Test: `tests/test_rebuild_image_index.py`

**Step 1: Write the failing indexing tests**

Test that only `catalog` split images are indexed, corrupt or missing files are reported without aborting the entire batch, duplicate checksums are skipped, and `recipe_id`, `image_id`, model version, attribution, and checksum survive into store records.

**Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_rebuild_image_index.py -q`

Expected: FAIL because the command does not exist.

**Step 3: Implement batch indexing**

Keep the orchestration pure and dependency-injected:

```python
def rebuild_image_index(catalog, encoder, store, *, batch_size=16) -> IndexReport:
    ...
```

The CLI constructs the real CLIP encoder and Milvus adapter from settings and prints indexed/skipped/failed counts plus elapsed time.

**Step 4: Run tests**

Run: `python -m pytest tests/test_rebuild_image_index.py -q`

Expected: PASS.

**Step 5: Commit**

```bash
git add gustobot/infrastructure/persistence/rebuild_image_index.py scripts/rebuild_image_index.py tests/test_rebuild_image_index.py
git commit -m "feat: build recipe image index offline"
```

### Task 5: Recipe-level dual-route fusion

**Files:**
- Create: `gustobot/application/multimodal/models.py`
- Create: `gustobot/application/multimodal/fusion.py`
- Test: `tests/test_multimodal_fusion.py`

**Step 1: Write failing RRF tests**

Cover these cases:

- multiple images for one recipe contribute only the best rank from the visual route;
- multiple text chunks for one recipe contribute only the best rank from the text route;
- a recipe present in both routes can outrank a recipe ranked first in only one route;
- evidence from all routes is retained after deduplication;
- deterministic tie-breaking uses best rank then recipe ID.

**Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_multimodal_fusion.py -q`

Expected: FAIL because fusion code does not exist.

**Step 3: Implement recipe-level RRF**

```python
def fuse_ranked_candidates(
    routes: Mapping[str, Sequence[RetrievalCandidate]],
    *,
    rrf_k: int = 60,
    limit: int = 20,
) -> list[FusedRecipeCandidate]:
    ...
```

Never add raw CLIP cosine scores to BM25 or text-vector scores. Convert every route to rank before fusion and preserve route-specific raw scores only for explanation.

**Step 4: Run tests**

Run: `python -m pytest tests/test_multimodal_fusion.py -q`

Expected: PASS.

**Step 5: Commit**

```bash
git add gustobot/application/multimodal/models.py gustobot/application/multimodal/fusion.py tests/test_multimodal_fusion.py
git commit -m "feat: fuse visual and text candidates by recipe"
```

### Task 6: Multimodal recipe retrieval service and Vision Agent integration

**Files:**
- Create: `gustobot/application/multimodal/service.py`
- Create: `gustobot/application/multimodal/vision_analysis.py`
- Modify: `gustobot/application/deepreason/direct_capabilities.py`
- Modify: `gustobot/application/deepreason/domain_agents.py`
- Test: `tests/test_multimodal_service.py`
- Modify: `tests/test_deepreason_domain_agents.py`

**Step 1: Write failing service tests**

Use fake VLM, text retriever, image encoder, and image store. Assert that an uploaded image produces:

```json
{
  "visual_analysis": {"dish_candidates": [], "ingredients": [], "cooking_methods": []},
  "image_matches": [],
  "text_matches": [],
  "fused_candidates": [],
  "selected_recipe_ids": [],
  "documents": []
}
```

Test graceful degradation: if CLIP/Milvus fails, return VLM-text results; if VLM fails, return image-vector results; fail only when both routes fail.

**Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_multimodal_service.py tests/test_deepreason_domain_agents.py -q`

Expected: FAIL because the service is not wired.

**Step 3: Implement and wire the service**

The VLM must output a Pydantic `VisualAnalysis`, not free-form prose. The service uses its normalized query for the existing recipe retriever, encodes the original image for Milvus, fuses candidates, and emits documents carrying `recipe_id`, route, rank, score, source, and image attribution. Extend `_evidence_from_result()` to produce `image_retrieval` evidence for visual hits.

**Step 4: Run focused tests**

Run: `python -m pytest tests/test_multimodal_service.py tests/test_deepreason_domain_agents.py -q`

Expected: PASS.

**Step 5: Commit**

```bash
git add gustobot/application/multimodal gustobot/application/deepreason/direct_capabilities.py gustobot/application/deepreason/domain_agents.py tests/test_multimodal_service.py tests/test_deepreason_domain_agents.py
git commit -m "feat: connect dual-route retrieval to vision agent"
```

### Task 7: Real dependency-aware Agent scheduling

**Files:**
- Create: `gustobot/application/deepreason/scheduling.py`
- Modify: `gustobot/application/deepreason/models.py`
- Modify: `gustobot/application/deepreason/orchestrator.py`
- Modify: `gustobot/application/deepreason/direct_capabilities.py`
- Test: `tests/test_deepreason_scheduling.py`
- Modify: `tests/test_deepreason_orchestrator.py`

**Step 1: Write failing scheduler tests**

Test pure functions first:

- tasks without dependencies are ready together;
- dependent tasks are not ready until every upstream task succeeds;
- dependency outputs are keyed by upstream task ID;
- missing dependency IDs are rejected before execution;
- cycles produce `dependency_cycle`;
- exhausted upstream failures mark downstream tasks `SKIPPED` with `dependency_failed`;
- independent branches continue even when another branch fails;
- a failed ready task retries within budget before downstream execution.

Then add an orchestration test where recipe returns `selected_recipe_ids=["201003834", "201000320"]` and analytics asserts those exact IDs appear in `context["dependency_outputs"]`; assert analytics starts only after recipe completes.

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_deepreason_scheduling.py tests/test_deepreason_orchestrator.py -q`

Expected: FAIL because all tasks are still dispatched in one `Send` fan-out.

**Step 3: Implement scheduler state and graph loop**

Add explicit state fields:

```python
task_statuses: dict[str, TaskStatus]
task_attempts: dict[str, int]
ready_tasks: list[AgentTask]
dependency_errors: list[str]
```

Change the graph to:

```text
reviewer -> scheduler -> Send(ready domain_agent batch)
domain_agent -> scheduler
scheduler(no runnable work) -> critic_gate -> synthesizer
```

`scheduler` is deterministic. It reads Handoffs, updates statuses, retries only failed tasks within budget, marks blocked descendants skipped, and detects a cycle when pending tasks exist but none can become ready. Each `Send` includes only the selected task plus `dependency_outputs` from successful upstream Handoffs.

For analytics, pass `selected_recipe_ids` structurally from context and constrain Text2SQL to those IDs; do not ask the LLM to rediscover which recipes were selected.

**Step 4: Run focused tests**

Run: `python -m pytest tests/test_deepreason_scheduling.py tests/test_deepreason_orchestrator.py -q`

Expected: PASS.

**Step 5: Commit**

```bash
git add gustobot/application/deepreason/scheduling.py gustobot/application/deepreason/models.py gustobot/application/deepreason/orchestrator.py gustobot/application/deepreason/direct_capabilities.py tests/test_deepreason_scheduling.py tests/test_deepreason_orchestrator.py
git commit -m "feat: schedule agent tasks by dependencies"
```

### Task 8: Evaluation, runtime docs, and interview proof

**Files:**
- Create: `scripts/evaluate_image_retrieval.py`
- Create: `configs/image_retrieval_eval.json`
- Modify: `scripts/check_environment.py`
- Modify: `docs/runtime-setup.md`
- Modify: `docs/architecture-deepreason.md`
- Modify: `docs/resume-and-interview.md`
- Test: `tests/test_image_retrieval_evaluation.py`

**Step 1: Write failing metric tests**

Implement small gold-label fixtures and assert exact `Recall@1`, `Recall@5`, `MRR`, and per-route/fused metrics. The evaluation split must not contain catalog images or duplicate checksums.

**Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_image_retrieval_evaluation.py -q`

Expected: FAIL because evaluation code does not exist.

**Step 3: Implement evaluation and documentation**

The report must include dataset size, label source, model version, collection name, index type, metric, per-route recall, fused recall, p50/p95 latency, failures, and limitations. Document exact Docker Compose startup and clearly mark unverified infrastructure until WSL/Milvus are available.

**Step 4: Run full verification**

Run:

```powershell
python -m pytest -q
python scripts/check_environment.py --models
python scripts/evaluate_image_retrieval.py --dry-run
```

Expected: all tests PASS; external model checks PASS; dry-run validates catalog/evaluation contracts without requiring Milvus.

**Step 5: Commit**

```bash
git add scripts configs docs tests gustobot
git commit -m "docs: add multimodal evaluation and interview runbook"
```

