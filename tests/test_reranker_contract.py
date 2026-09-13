from gustobot.infrastructure.core.reranker import (
    _build_qwen3_payload,
    _extract_ranked_results,
)


def test_qwen3_rerank_payload_uses_flat_compatible_contract():
    payload = _build_qwen3_payload(
        "宫保鸡丁需要什么食材",
        ["鸡肉和花生", "番茄炒蛋"],
        model="qwen3-rerank",
        top_n=1,
    )

    assert payload == {
        "model": "qwen3-rerank",
        "query": "宫保鸡丁需要什么食材",
        "documents": ["鸡肉和花生", "番茄炒蛋"],
        "top_n": 1,
    }
    assert "input" not in payload
    assert "parameters" not in payload


def test_extracts_current_qwen3_top_level_response():
    ranked = _extract_ranked_results(
        {
            "results": [
                {"index": 1, "relevance_score": 0.2},
                {"index": 0, "relevance_score": 0.9},
            ]
        }
    )

    assert ranked == [(0, 0.9), (1, 0.2)]


def test_extracts_legacy_nested_response_for_backward_compatibility():
    ranked = _extract_ranked_results(
        {"output": {"results": [{"index": 2, "relevance_score": 0.7}]}}
    )

    assert ranked == [(2, 0.7)]
