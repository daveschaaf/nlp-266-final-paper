"""Read the versioned frozen text-unit embeddings cached in Qdrant."""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, ScoredPoint, UpdateResult, SearchParams, Filter, FieldCondition, MatchValue, MatchAny, Range
import uuid
from .contracts import EmbeddingRecord, EmbeddingInput
import logging
from dearborn.logging_config import setup_logging
import numpy as np
import os
logger = logging.getLogger(__name__)
setup_logging()


EMBEDDING_MODEL = "google/embeddinggemma-300m"
QDRANT_URL = os.environ.get("DEARBORN_QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.environ.get("DEARBORN_QDRANT_COLLECTION", "extract_gemma_storage")

class ExtractEmbeddingStore():
    """Recover embedding records by a deterministic UUID derived from unit ID."""
    def __init__(self, collection):
        self.collection = collection
        self.client = QdrantClient(url=QDRANT_URL)
        self.init_collection()

    def init_collection(self):
        # Dataset construction must not silently create an empty store. Vectors
        # are written explicitly by embeddings.py before they can be consumed.
        if not self.client.collection_exists(self.collection):
            raise LookupError("Collection does not exist: {self.collection}")
        else:
            logger.info(f"Collection exists: {self.collection}")

    def fetch(self, unit_ids: list[str]) -> list[EmbeddingRecord]:
        """Fetch all requested vectors in caller ID order, or fail loudly."""
        if not unit_ids:
            return []
        # embeddings.py writes points under these exact UUIDv5 values.
        point_ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, unit_id))
            for unit_id in unit_ids
        ]
        points = self.client.retrieve(
            collection_name=self.collection,
            ids=point_ids,
            with_vectors=True,
            with_payload=True,
        )
        # Qdrant does not guarantee the original request order. Index by the
        # payload ID and reconstruct records below in unit_ids order.
        by_unit_id = {
            point.payload['unit_id']: point
            for point in points
        }
        missing_ids = [unit_id for unit_id in unit_ids if unit_id not in by_unit_id]
        if missing_ids:
            raise LookupError(
                f'Unable to find {len(missing_ids)} embedding(s) with ids {missing_ids}'
            )
        records = [
            EmbeddingRecord(
                input=EmbeddingInput(
                    unit_id=unit_id,
                    source_scope_id=by_unit_id[unit_id].payload['source_scope_id'],
                    modality=by_unit_id[unit_id].payload['modality'],
                    text=by_unit_id[unit_id].payload['text'],
                    ),
                encoder_id=by_unit_id[unit_id].payload['encoder_id'],
                embedding_dimension=by_unit_id[unit_id].payload['embedding_dimension'],
                normalization=by_unit_id[unit_id].payload['normalization'],
                vector=np.asarray(by_unit_id[unit_id].vector, dtype=np.float32),
                )
            for unit_id in unit_ids
        ]
        return records
