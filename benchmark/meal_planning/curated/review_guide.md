# SmartRecipe 检索 Gold 复核说明

这 40 条问题用于检索开发集，不是自动生成的线上指标。复核时逐条检查：

1. `query` 是否像真实用户会说的话；
2. `gold_recipe_name` 和食材是否确实满足问题；
3. `meal_type`、最大时间、最低蛋白质等过滤条件是否与原菜谱事实一致；
4. 如果不止一道菜都相关，在 `relevant_recipe_ids` 中补充所有可接受答案；
5. 复核完成后再冻结版本，开发过程中不得继续修改冻结测试集。

运行下面的命令可重新生成复核表：

```powershell
uv run python scripts/export_retrieval_review_sheet.py
```

复核文件为 `review_sheet.v1.csv`。只有人工复核并冻结后，指标才适合用于简历。
