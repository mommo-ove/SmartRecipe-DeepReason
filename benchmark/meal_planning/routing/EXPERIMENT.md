# Meal Router evaluation protocol

The experiment measures the final gateway decision, not only the LLM label. A
meal-plan request may therefore call both the Router and the constraint extractor.
Latency, tokens, fallback usage and cost include both calls.

## Data boundary

- `cases.v2.jsonl`: 43 iteration cases. It may be inspected and used to change prompts.
- `frozen/cases.v1.jsonl`: 60 hash-pinned holdout cases. Do not change prompts or
  fallback rules from its errors. Resume metrics remain ineligible until labels are
  independently reviewed and the manifest is updated.

The runner rejects R1 and R2 on the frozen set.

## Baselines and three iterations

1. R0: zero-shot LLM baseline. No examples.
2. R1: add representative examples for each business intent.
3. R2: add boundary negatives for recommendation vs multi-day planning, composite
   retrieval plus nutrition, and missing conversational references.
4. R3: mine R2 development errors. BGE-M3 removes semantic duplicates and preserves
   distinct confusion patterns; only the selected errors become examples.

Run R0-R2 on the iteration set:

```powershell
uv run python scripts/run_meal_router_experiment.py --router llm --dataset development --prompt-version r0_zero_shot --output benchmark/meal_planning/routing/runs/dev-r0.json
uv run python scripts/run_meal_router_experiment.py --router llm --dataset development --prompt-version r1_few_shot --output benchmark/meal_planning/routing/runs/dev-r1.json
uv run python scripts/run_meal_router_experiment.py --router llm --dataset development --prompt-version r2_hard_negative --output benchmark/meal_planning/routing/runs/dev-r2.json
```

Mine R2 errors with the local BGE-M3 model, then run R3:

```powershell
uv run --extra retrieval-models python scripts/mine_meal_router_errors.py --input benchmark/meal_planning/routing/runs/dev-r2.json --output benchmark/meal_planning/routing/runs/dev-r2-mined.json --model-path "F:/agent+项目/.hf-cache/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"
uv run python scripts/run_meal_router_experiment.py --router llm --dataset development --prompt-version r3_mined_errors --mined-errors benchmark/meal_planning/routing/runs/dev-r2-mined.json --output benchmark/meal_planning/routing/runs/dev-r3.json
```

After R3 is locked, run it once on the frozen set:

```powershell
uv run python scripts/run_meal_router_experiment.py --router llm --dataset frozen --prompt-version r3_mined_errors --mined-errors benchmark/meal_planning/routing/runs/dev-r2-mined.json --output benchmark/meal_planning/routing/runs/frozen-r3-final.json
```

Report intent Macro-F1, final policy error rate, unsafe direct-route rate,
clarification rate, fallback rate, p95 latency, token count and model cost. Pricing
must be passed with the two `--*-cost-per-million` options from a dated provider
price sheet; the default zero value is not a claim that model calls are free.
