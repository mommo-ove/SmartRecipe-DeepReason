from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ImageVectorRecord(BaseModel):
    image_id: str
    recipe_id: str
    image_path: str
    image_vector: list[float]
    model_version: str
    checksum: str
    source_page: str = ""
    license: str = ""


class ImageSearchHit(BaseModel):
    image_id: str
    recipe_id: str
    image_path: str
    score: float
    source_page: str = ""
    model_version: str = ""
    checksum: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class MilvusImageVectorStore:
    def __init__(
        self,
        *,
        client: Any | None = None,
        uri: str | None = None,
        collection_name: str = "recipe_image_embeddings",
        dimension: int = 512,
    ) -> None:
        self.collection_name = collection_name
        self.dimension = dimension
        if client is None:
            from pymilvus import MilvusClient

            if not uri:
                raise ValueError("uri is required when client is not provided")
            client = MilvusClient(uri=uri)
        self.client = client

    def ensure_collection(self) -> None:
        if self.client.has_collection(self.collection_name):
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=self.dimension,
            primary_field_name="image_id",
            id_type="string",
            vector_field_name="image_vector",
            metric_type="COSINE",
            auto_id=False,
            enable_dynamic_field=True,
            max_length=128,
        )

    def upsert(self, records: list[ImageVectorRecord]) -> int:
        if not records:
            return 0
        self.ensure_collection()
        data: list[dict[str, Any]] = []
        for record in records:
            self._validate_vector(record.image_vector)
            data.append(record.model_dump())
        result = self.client.upsert(collection_name=self.collection_name, data=data)
        return int(result.get("upsert_count", len(data)))

    def search(self, vector: list[float], *, top_k: int = 10) -> list[ImageSearchHit]:
        self._validate_vector(vector)
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.ensure_collection()
        output_fields = [
            "recipe_id",
            "image_path",
            "source_page",
            "model_version",
            "checksum",
            "license",
        ]
        result = self.client.search(
            collection_name=self.collection_name,
            data=[vector],
            anns_field="image_vector",
            limit=top_k,
            output_fields=output_fields,
            search_params={"metric_type": "COSINE", "params": {"ef": max(64, top_k)}},
        )
        rows = result[0] if result else []
        hits: list[ImageSearchHit] = []
        for row in rows:
            entity = row.get("entity", {}) or {}
            metadata = {field: entity.get(field, row.get(field)) for field in output_fields}
            hits.append(
                ImageSearchHit(
                    image_id=str(row.get("id", entity.get("image_id", ""))),
                    recipe_id=str(metadata["recipe_id"]),
                    image_path=str(metadata["image_path"]),
                    score=float(row.get("distance", row.get("score", 0.0))),
                    source_page=str(metadata.get("source_page") or ""),
                    model_version=str(metadata.get("model_version") or ""),
                    checksum=str(metadata.get("checksum") or ""),
                    metadata={"license": metadata.get("license") or ""},
                )
            )
        return hits

    def stats(self) -> dict[str, Any]:
        self.ensure_collection()
        stats = self.client.get_collection_stats(self.collection_name)
        return {
            "collection_name": self.collection_name,
            "dimension": self.dimension,
            "row_count": int(stats.get("row_count", 0)),
        }

    def _validate_vector(self, vector: list[float]) -> None:
        if len(vector) != self.dimension:
            raise ValueError(f"expected {self.dimension} dimensions, got {len(vector)}")
