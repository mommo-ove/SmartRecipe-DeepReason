# FoodTrace allergen-recall demo

FoodTrace includes a deterministic, offline incident-response path for an
undeclared peanut notice. It traces the ingredient lot through recipes,
products, and production batches, then limits inventory and order lookup to
those batches. A conservative gate allows a report only when the required
evidence and impact counts agree.

Run the interview-friendly trace:

```powershell
uv run python scripts/run_foodtrace_demo.py --case allergen_peanut_001
```

Run the same result as structured JSON:

```powershell
uv run python scripts/run_foodtrace_demo.py --case allergen_peanut_001 --json
```

The demo data is a fixed-seed synthetic fixture. It does not claim to contain
production incidents or live customer data. The CLI and API use the same
`FoodTraceWorkflow`; the API endpoint is `POST /api/foodtrace/investigations`
and accepts an `IncidentNotice` JSON body.
