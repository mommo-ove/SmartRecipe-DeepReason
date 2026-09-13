"""Beginner demo: pass selected recipe IDs from one task to another."""


def run_demo() -> None:
    # 1. 模拟菜谱 Agent 的输出。字典可以理解成一张带字段名的表单。
    recipe_output = {
        "answer": "推荐西红柿炒蛋和番茄鸡蛋汤",
        "selected_recipe_ids": ["201003834", "201004552"],
    }

    # 2. 调度器把上游输出放进 dependency_outputs。
    context = {
        "dependency_outputs": {
            "recipe-1": recipe_output,
        }
    }

    # 3. 营养任务按照依赖任务的名字，取出上游选中的菜谱 ID。
    recipe_ids = context["dependency_outputs"]["recipe-1"]["selected_recipe_ids"]

    # 4. 把 ID 放进 Text2SQL 的输入，限制它只能查询这两道菜。
    analytics_input = {
        "question": "计算这两道菜的总热量",
        "recipe_ids": recipe_ids,
    }

    print("菜谱 Agent 输出：", recipe_output)
    print("营养 Agent 收到：", analytics_input)


if __name__ == "__main__":
    run_demo()
