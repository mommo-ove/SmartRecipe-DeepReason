# SmartRecipe Evaluation Console

This Gradio page calls the existing benchmark scripts; it does not hard-code metric values.

## Start

```powershell
cd F:\agent+项目\SmartRecipe-DeepReason\.worktrees\multimodal-dependency-upgrade\evaluation_ui
uv run python run.py
```

Open <http://127.0.0.1:7860>. The page also attempts to open a browser automatically.

## Recommended first run

1. Click **检查环境** and confirm the BGE model and Neo4j are ready.
2. Keep only `bm25`, select `test`, and click **开始真实评测**. This is the fastest end-to-end smoke run.
3. Select all four configurations for the complete ablation. BGE-M3 on CPU is slower.
4. Use **逐题检查** to compare Gold hits and forbidden-ingredient violations.
5. Use **项目100问** to search the architecture and evaluation explanations.

The raw JSON report is written under
`benchmark/meal_planning/graph_relations/runs/ui/<split>/`.
