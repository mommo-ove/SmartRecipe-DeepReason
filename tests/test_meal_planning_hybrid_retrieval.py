from __future__ import annotations

import numpy as np

from gustobot.application.meal_planning.external_corpus import ExternalRecipeNormalizer
from gustobot.application.meal_planning.hybrid_retrieval import (
    BgeM3Encoder,
    BgeReranker,
    DenseExternalRetriever,
    RRFExternalRetriever,
    RerankedExternalRetriever,
    build_external_retriever,
    rrf_fuse,
)
from gustobot.application.meal_planning.retrieval_baseline import ExternalBM25Retriever


def document(recipe_id: int, name: str, ingredients: str):
    return ExternalRecipeNormalizer().normalize(
        {
            "RecipeId": recipe_id,
            "Name": name,
            "TotalTime": "PT25M",
            "Description": "main meal",
            "RecipeCategory": "Chicken Breast",
            "Keywords": 'c("Lunch")',
            "RecipeIngredientParts": ingredients,
            "Calories": 500,
            "ProteinContent": 40,
            "RecipeInstructions": 'c("cook")',
        }
    )


class KeywordEncoder:
    def encode(self, texts):
        return np.asarray(
            [
                [
                    float("chicken" in text.casefold()),
                    float("beef" in text.casefold()),
                ]
                for text in texts
            ],
            dtype=np.float32,
        )


def test_dense_retriever_uses_cosine_similarity_and_structured_filters():
    chicken = document(1, "Chicken Bowl", 'c("chicken", "broccoli")')
    beef = document(2, "Beef Bowl", 'c("beef", "pepper")')
    retriever = DenseExternalRetriever([chicken, beef], KeywordEncoder())

    results = retriever.search(
        "chicken meal",
        {"meal_type": "lunch", "max_meal_minutes": 30},
        top_k=2,
    )

    assert results == ["foodcom-1", "foodcom-2"]


def test_rrf_rewards_candidates_supported_by_multiple_retrievers():
    fused = rrf_fuse(
        {
            "bm25": ["recipe-a", "x1", "x2", "x3", "recipe-b"],
            "dense": ["y1", "y2", "y3", "y4", "recipe-b"],
        },
        k=60,
    )

    assert fused.index("recipe-b") < fused.index("recipe-a")


class StaticRetriever:
    def search(self, query, metadata_filters=None, *, top_k=20):
        return ["foodcom-1", "foodcom-2"][:top_k]


class BeefFirstReranker:
    def score(self, query, documents):
        return [float("Beef" in text) for text in documents]


def test_second_stage_reranker_can_change_rrf_candidate_order():
    chicken = document(1, "Chicken Bowl", 'c("chicken", "broccoli")')
    beef = document(2, "Beef Bowl", 'c("beef", "pepper")')
    retriever = RerankedExternalRetriever(
        [chicken, beef],
        StaticRetriever(),
        BeefFirstReranker(),
        candidate_k=2,
    )

    assert retriever.search("high protein", top_k=2) == ["foodcom-2", "foodcom-1"]


class FakeSentenceTransformer:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return np.ones((len(texts), 3), dtype=np.float32)


def test_bge_m3_adapter_requests_normalized_embeddings():
    model = FakeSentenceTransformer()
    encoder = BgeM3Encoder(model=model)

    vectors = encoder.encode(["中文查询", "English recipe"])

    assert vectors.shape == (2, 3)
    assert model.calls[0][1]["normalize_embeddings"] is True


class FakeFlagReranker:
    def compute_score(self, pairs, normalize=True):
        assert normalize is True
        return [0.2, 0.9]


def test_bge_reranker_adapter_scores_query_document_pairs():
    reranker = BgeReranker(model=FakeFlagReranker())

    assert reranker.score("query", ["doc-a", "doc-b"]) == [0.2, 0.9]


def test_factory_builds_the_four_comparable_retrieval_configurations():
    chicken = document(1, "Chicken Bowl", 'c("chicken", "broccoli")')
    beef = document(2, "Beef Bowl", 'c("beef", "pepper")')
    documents = [chicken, beef]

    bm25 = build_external_retriever("bm25", documents)
    dense = build_external_retriever(
        "dense", documents, encoder=KeywordEncoder()
    )
    rrf = build_external_retriever(
        "rrf", documents, encoder=KeywordEncoder()
    )
    reranked = build_external_retriever(
        "rrf_rerank",
        documents,
        encoder=KeywordEncoder(),
        reranker=BeefFirstReranker(),
    )

    assert isinstance(bm25, ExternalBM25Retriever)
    assert isinstance(dense, DenseExternalRetriever)
    assert isinstance(rrf, RRFExternalRetriever)
    assert isinstance(reranked, RerankedExternalRetriever)
