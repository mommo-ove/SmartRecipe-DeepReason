from gustobot.application.deepreason.models import AgentTask, Domain, ExecutionPlan, ReviewDecision
import pytest

from gustobot.application.deepreason.planning import HeuristicPlanner, StructuredPlanner


def test_compound_recipe_statistics_question_creates_two_tasks():
    plan = HeuristicPlanner().plan("推荐一道低辣鸡肉菜，并统计这类菜的平均烹饪时长")

    assert [task.domain for task in plan.tasks] == [Domain.RECIPE, Domain.ANALYTICS]
    assert plan.is_multi_agent is True


def test_selected_recipe_nutrition_plan_declares_limit_and_dependency():
    plan = HeuristicPlanner().plan("推荐两道菜，并计算这两道菜的总热量")

    recipe_task, analytics_task = plan.tasks
    assert recipe_task.domain == Domain.RECIPE
    assert recipe_task.result_limit == 2
    assert analytics_task.domain == Domain.ANALYTICS
    assert analytics_task.depends_on == [recipe_task.task_id]


def test_category_statistics_remain_independent_of_single_recommendation():
    plan = HeuristicPlanner().plan("推荐一道低辣鸡肉菜，并统计这类菜的平均烹饪时长")

    recipe_task, analytics_task = plan.tasks
    assert recipe_task.result_limit == 1
    assert analytics_task.depends_on == []


def test_image_path_routes_to_vision_even_without_image_keyword():
    plan = HeuristicPlanner().plan("分析一下", image_path="uploads/dish.jpg")

    assert [task.domain for task in plan.tasks] == [Domain.VISION]


def test_general_conversation_does_not_create_recipe_task():
    plan = HeuristicPlanner().plan("你好，谢谢你的帮助")

    assert [task.domain for task in plan.tasks] == [Domain.GENERAL]


def test_pure_recipe_count_question_only_creates_analytics_task():
    plan = HeuristicPlanner().plan("统计菜谱总数")

    assert [task.domain for task in plan.tasks] == [Domain.ANALYTICS]


def test_reviewer_adds_recipe_task_for_food_recommendation():
    planner = HeuristicPlanner()
    plan = planner.plan("推荐一道适合新手的鸡肉菜")
    reviewed = planner.review("推荐一道适合新手的鸡肉菜", plan)

    assert reviewed.review_status == "approved"
    assert reviewed.tasks[0].domain == Domain.RECIPE


@pytest.mark.asyncio
async def test_structured_planner_falls_back_when_llm_fails():
    class BrokenModel:
        def with_structured_output(self, schema):
            raise RuntimeError("provider unavailable")

    plan = await StructuredPlanner(model=BrokenModel()).aplan(
        "推荐鸡肉菜并统计平均时长"
    )

    assert [task.domain for task in plan.tasks] == [Domain.RECIPE, Domain.ANALYTICS]
    assert "fallback" in plan.rationale


@pytest.mark.asyncio
async def test_structured_planner_accepts_valid_llm_plan():
    expected = HeuristicPlanner().plan("统计菜谱数量")
    expected.rationale = "llm decomposition"

    class StructuredCall:
        async def ainvoke(self, messages):
            return expected

    class Model:
        def with_structured_output(self, schema):
            return StructuredCall()

    plan = await StructuredPlanner(model=Model()).aplan("统计菜谱数量")

    assert plan.rationale == "llm decomposition"
    assert plan.review_status == "approved"


@pytest.mark.asyncio
async def test_structured_reviewer_can_add_domain_missed_by_coordinator():
    coordinator_plan = ExecutionPlan(
        query="推荐鸡肉菜并统计平均时长",
        tasks=[AgentTask(task_id="recipe-1", domain=Domain.RECIPE, instruction="查询鸡肉菜")],
    )
    reviewer_decision = ReviewDecision(
        status="revise",
        required_domains=[Domain.RECIPE, Domain.ANALYTICS],
        findings=["coordinator missed analytics task"],
    )

    class StructuredCall:
        def __init__(self, value):
            self.value = value

        async def ainvoke(self, messages):
            return self.value

    class Model:
        def with_structured_output(self, schema):
            value = coordinator_plan if schema is ExecutionPlan else reviewer_decision
            return StructuredCall(value)

    plan = await StructuredPlanner(model=Model()).aplan("推荐鸡肉菜并统计平均时长")

    assert [task.domain for task in plan.tasks] == [Domain.RECIPE, Domain.ANALYTICS]
    assert plan.review_status == "revised"
    assert "missed analytics" in plan.reviewer_notes[0]
