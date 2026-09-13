import pytest

from gustobot.application.meal_planning.graph_retrieval import apply_recipe_id_gate
from gustobot.application.meal_planning.retrieval_router import (
    RetrievalBoundaryError,
    RetrievalQueryKind,
    RetrievalRequest,
    RetrievalRoute,
    RetrievalRouter,
)
from gustobot.application.meal_planning.text2cypher.models import (
    CypherRunStatus,
    Text2CypherResult,
)


class CountingRetriever:
    def __init__(self, ranking):
        self.ranking = list(ranking)
        self.calls = []

    def search(self, query, metadata_filters=None, *, top_k=20):
        self.calls.append((query, metadata_filters or {}, top_k))
        return self.ranking[:top_k]


def graph_result(status=CypherRunStatus.SUCCESS, recipe_ids=None):
    return Text2CypherResult(
        question="graph question",
        status=status,
        selected_recipe_ids=list(recipe_ids or []),
    )


def router(bm25=None, dense=None):
    return RetrievalRouter(
        bm25=bm25 or CountingRetriever(["bm25-a"]),
        dense=dense or CountingRetriever(["dense-a"]),
    )


def test_exact_keyword_request_uses_bm25_without_calling_dense():
    bm25 = CountingRetriever(["r1", "r2"])
    dense = CountingRetriever(["r2", "r3"])
    service = router(bm25, dense)
    request = RetrievalRequest(
        query="番茄炒蛋",
        kind=RetrievalQueryKind.EXACT_KEYWORD,
        top_k=2,
    )

    result = service.search(request)

    assert service.choose_route(request) is RetrievalRoute.BM25
    assert result == ["r1", "r2"]
    assert len(bm25.calls) == 1
    assert dense.calls == []


def test_fuzzy_preference_request_uses_bm25_dense_and_rrf():
    bm25 = CountingRetriever(["lexical", "shared"])
    dense = CountingRetriever(["semantic", "shared"])
    service = router(bm25, dense)
    request = RetrievalRequest(
        query="清淡又有饱腹感的菜",
        kind=RetrievalQueryKind.FUZZY_PREFERENCE,
        top_k=3,
    )

    result = service.search(request)

    assert service.choose_route(request) is RetrievalRoute.HYBRID
    assert result[0] == "shared"
    assert len(bm25.calls) == 1
    assert len(dense.calls) == 1


def test_hard_relation_request_uses_only_text2cypher_ids():
    bm25 = CountingRetriever(["unsafe", "safe"])
    dense = CountingRetriever(["unsafe", "safe"])
    service = router(bm25, dense)
    request = RetrievalRequest(
        query="含鸡蛋但不含花生",
        kind=RetrievalQueryKind.HARD_RELATION,
        top_k=5,
    )

    result = service.search(
        request,
        graph_result=graph_result(recipe_ids=["safe", "safe", "safe-2"]),
    )

    assert service.choose_route(request) is RetrievalRoute.GRAPH_ONLY
    assert result == ["safe", "safe-2"]
    assert bm25.calls == []
    assert dense.calls == []


def test_text_rankings_are_hard_filtered_by_text2cypher_boundary():
    bm25 = CountingRetriever(["forbidden", "safe-a", "safe-b"])
    dense = CountingRetriever(["forbidden", "safe-b", "safe-a"])
    service = router(bm25, dense)
    request = RetrievalRequest(
        query="清淡的鸡蛋菜",
        kind=RetrievalQueryKind.FUZZY_PREFERENCE,
        top_k=5,
    )

    result = service.search(
        request,
        graph_result=graph_result(recipe_ids=["safe-a", "safe-b"]),
    )

    assert set(result) == {"safe-a", "safe-b"}
    assert "forbidden" not in result


def test_empty_text2cypher_boundary_returns_empty_instead_of_full_text_search():
    bm25 = CountingRetriever(["would-have-been-returned"])
    service = router(bm25, CountingRetriever([]))
    request = RetrievalRequest(
        query="不存在的食材组合",
        kind=RetrievalQueryKind.EXACT_KEYWORD,
    )

    result = service.search(
        request,
        graph_result=graph_result(status=CypherRunStatus.EMPTY),
    )

    assert result == []
    assert bm25.calls == []


@pytest.mark.parametrize(
    "status",
    [
        CypherRunStatus.VALIDATION_FAILED,
        CypherRunStatus.EXECUTION_FAILED,
    ],
)
def test_failed_text2cypher_boundary_never_falls_back_to_full_corpus(status):
    request = RetrievalRequest(
        query="不含花生的菜",
        kind=RetrievalQueryKind.HARD_RELATION,
    )

    with pytest.raises(RetrievalBoundaryError):
        router().search(request, graph_result=graph_result(status=status))


def test_rrf_defense_in_depth_cannot_reintroduce_a_forbidden_id():
    ranking = apply_recipe_id_gate(
        ["forbidden", "safe", "forbidden", "safe-2"],
        ["safe", "safe-2"],
    )

    assert ranking == ["safe", "safe-2"]

