from __future__ import annotations

import pytest

from gustobot.infrastructure.persistence.image_vector_store import (
    ImageVectorRecord,
    MilvusImageVectorStore,
)


class FakeMilvusClient:
    def __init__(self, *, has_collection: bool = False) -> None:
        self.collection_exists = has_collection
        self.created: list[dict] = []
        self.upserts: list[dict] = []
        self.searches: list[dict] = []

    def has_collection(self, collection_name: str) -> bool:
        return self.collection_exists

    def create_collection(self, **kwargs):
        self.created.append(kwargs)
        self.collection_exists = True

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)
        return {"upsert_count": len(kwargs["data"])}

    def search(self, **kwargs):
        self.searches.append(kwargs)
        return [[{
            "id": "img-1",
            "distance": 0.91,
            "entity": {
                "recipe_id": "201003834",
                "image_path": "gustobot/data/recipe_images/catalog/201003834/01.jpg",
                "source_page": "https://commons.wikimedia.org/wiki/File:Kung_Pao.jpg",
                "model_version": "openai/clip-vit-base-patch32",
                "checksum": "abc",
            },
        }]]

    def get_collection_stats(self, collection_name: str):
        return {"row_count": 12}


def test_ensure_collection_uses_explicit_image_schema_contract() -> None:
    client = FakeMilvusClient()
    store = MilvusImageVectorStore(client=client, dimension=512)

    store.ensure_collection()

    assert client.created == [{
        "collection_name": "recipe_image_embeddings",
        "dimension": 512,
        "primary_field_name": "image_id",
        "id_type": "string",
        "vector_field_name": "image_vector",
        "metric_type": "COSINE",
        "auto_id": False,
        "enable_dynamic_field": True,
        "max_length": 128,
    }]


def test_upsert_serializes_vector_and_recipe_metadata() -> None:
    client = FakeMilvusClient(has_collection=True)
    store = MilvusImageVectorStore(client=client, dimension=2)
    record = ImageVectorRecord(
        image_id="img-1",
        recipe_id="201003834",
        image_path="gustobot/data/recipe_images/catalog/201003834/01.jpg",
        image_vector=[0.6, 0.8],
        model_version="test/clip",
        checksum="abc",
        source_page="https://commons.wikimedia.org/wiki/File:Kung_Pao.jpg",
    )

    count = store.upsert([record])

    assert count == 1
    assert client.upserts[0]["collection_name"] == "recipe_image_embeddings"
    assert client.upserts[0]["data"][0]["recipe_id"] == "201003834"
    assert client.upserts[0]["data"][0]["image_vector"] == [0.6, 0.8]


def test_search_returns_typed_hits_and_requests_metadata() -> None:
    client = FakeMilvusClient(has_collection=True)
    store = MilvusImageVectorStore(client=client, dimension=2)

    hits = store.search([0.6, 0.8], top_k=5)

    assert hits[0].image_id == "img-1"
    assert hits[0].recipe_id == "201003834"
    assert hits[0].score == pytest.approx(0.91)
    assert client.searches[0]["anns_field"] == "image_vector"
    assert client.searches[0]["limit"] == 5
    assert "recipe_id" in client.searches[0]["output_fields"]


def test_store_rejects_vectors_with_wrong_dimension() -> None:
    store = MilvusImageVectorStore(client=FakeMilvusClient(has_collection=True), dimension=3)

    with pytest.raises(ValueError, match="expected 3 dimensions"):
        store.search([0.6, 0.8], top_k=5)


def test_stats_exposes_collection_and_row_count() -> None:
    store = MilvusImageVectorStore(client=FakeMilvusClient(has_collection=True), dimension=512)

    assert store.stats() == {
        "collection_name": "recipe_image_embeddings",
        "dimension": 512,
        "row_count": 12,
    }
