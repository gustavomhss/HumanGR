"""
HumanGR Qdrant Client

Cliente Qdrant que APENAS acessa collections com prefixo humangr_.
NUNCA acessa collections pipeline_* ou veritas_*.
"""

from typing import Any, Optional
from dataclasses import dataclass

from .config import get_config

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClient = None

try:
    from fastembed import TextEmbedding
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False
    TextEmbedding = None


@dataclass
class SearchResult:
    """Search result from Qdrant."""
    sprint_id: str
    title: str
    score: float
    content: str
    payload: dict


class HumanGRQdrantClient:
    """
    Qdrant client for HumanGR.

    CRITICAL SEPARATION:
    - Only accesses humangr_* collections
    - Never touches pipeline_* or veritas_*
    """

    def __init__(self):
        if not QDRANT_AVAILABLE:
            raise ImportError("qdrant-client not installed. Run: pip install qdrant-client")

        config = get_config()

        self._client = QdrantClient(
            host=config.qdrant_host,
            port=config.qdrant_port,
        )

        self._collection = config.qdrant_context_collection
        self._embedder = None

        # Validate collection name
        if not self._collection.startswith("humangr_"):
            raise ValueError(
                f"CRITICAL: Collection must start with 'humangr_', got '{self._collection}'"
            )

    def _get_embedder(self):
        """Lazy load embedder."""
        if self._embedder is None:
            if not FASTEMBED_AVAILABLE:
                raise ImportError("fastembed not installed. Run: pip install fastembed")
            self._embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        return self._embedder

    def health_check(self) -> dict[str, Any]:
        """Check Qdrant connection and collection."""
        try:
            collections = self._client.get_collections().collections
            collection_names = [c.name for c in collections]

            # Check our collection exists
            has_collection = self._collection in collection_names

            # Count humangr_ collections
            humangr_collections = [c for c in collection_names if c.startswith("humangr_")]

            # CRITICAL: Check for contamination
            has_pipeline = any(c.startswith("pipeline_") for c in collection_names)
            has_veritas = any(c.startswith("veritas_") for c in collection_names)

            return {
                "connected": True,
                "collection_exists": has_collection,
                "collection_name": self._collection,
                "humangr_collections": humangr_collections,
                "contamination_warning": has_pipeline or has_veritas,
                "contamination_details": {
                    "has_pipeline_collections": has_pipeline,
                    "has_veritas_collections": has_veritas,
                },
            }
        except Exception as e:
            return {
                "connected": False,
                "error": str(e),
            }

    def get_collection_stats(self) -> dict[str, Any]:
        """Get statistics about the HumanGR collection."""
        try:
            info = self._client.get_collection(self._collection)
            return {
                "collection": self._collection,
                "points_count": info.points_count,
                "status": info.status,
                "vector_size": info.config.params.vectors.size,
            }
        except Exception as e:
            return {
                "collection": self._collection,
                "error": str(e),
            }

    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[SearchResult]:
        """
        Search for context packs by semantic similarity.

        Args:
            query: Search query text
            limit: Maximum number of results

        Returns:
            List of SearchResult objects
        """
        embedder = self._get_embedder()

        # Get query embedding
        query_vector = list(embedder.embed([query]))[0].tolist()

        # Search
        results = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=limit,
        )

        return [
            SearchResult(
                sprint_id=r.payload.get("sprint_id", ""),
                title=r.payload.get("title", ""),
                score=r.score,
                content=r.payload.get("content", "")[:500],
                payload=r.payload,
            )
            for r in results.points
        ]

    def get_by_sprint(self, sprint_id: str) -> Optional[dict]:
        """
        Get context pack by sprint ID.

        Args:
            sprint_id: Sprint identifier (e.g., "S00")

        Returns:
            Context pack payload or None
        """
        results = self._client.scroll(
            collection_name=self._collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="sprint_id",
                        match=models.MatchValue(value=sprint_id),
                    ),
                ]
            ),
            limit=1,
            with_payload=True,
        )

        if results[0]:
            return results[0][0].payload
        return None


def get_qdrant_client() -> HumanGRQdrantClient:
    """Get HumanGR Qdrant client instance."""
    return HumanGRQdrantClient()
