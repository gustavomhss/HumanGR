"""Qdrant Hybrid Search Integration.

Enhances Qdrant integration with:
1. Hybrid search combining dense and sparse vectors (BM25)
2. Advanced collection management
3. Payload filtering for precise retrieval
4. Quantization for memory efficiency
5. Batch operations for performance

Key Features:
- Dense vectors: Semantic similarity using embeddings
- Sparse vectors: Keyword matching using BM25-style tokenization
- Hybrid fusion: Combines both for best results

Usage:
    from pipeline.rag.qdrant_hybrid import (
        QdrantHybridSearch,
        create_hybrid_collection,
        hybrid_search,
    )

    # Initialize
    qdrant = QdrantHybridSearch()

    # Create hybrid collection
    await qdrant.create_hybrid_collection("claims")

    # Add documents with hybrid vectors
    await qdrant.add_documents([
        {"id": "1", "text": "Climate change is real", "metadata": {"source": "IPCC"}},
    ])

    # Hybrid search
    results = await qdrant.hybrid_search(
        query="global warming evidence",
        top_k=10,
        filter={"source_type": "verified"},
    )
"""

from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional, Tuple, TypedDict
from datetime import datetime, timezone
from collections import Counter
import re

from pydantic import BaseModel, Field

# Phase 2 FIX: Import resilience infrastructure
from pipeline.retry_config import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitOpenError,
    CircuitState,
)
from pipeline.resilience.coordinator import get_stack_circuit_coordinator

# Stack enforcement decorator for critical operations (vector search requires qdrant)
try:
    from pipeline.langgraph.nemo_stack_rails import enforce_stacks, StackEnforcementError
    STACK_ENFORCEMENT_AVAILABLE = True
except ImportError:
    STACK_ENFORCEMENT_AVAILABLE = False
    # Provide no-op decorator if not available
    def enforce_stacks(action: str, required=None, recommended=None):
        def decorator(func):
            return func
        return decorator
    class StackEnforcementError(Exception):
        pass

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
DEFAULT_COLLECTION = os.getenv("QDRANT_DEFAULT_COLLECTION", "claims")
# EMBEDDING_SIZE is now dynamic - retrieved from EmbeddingProvider
# Default fallback if provider not available
_DEFAULT_EMBEDDING_SIZE = int(os.getenv("QDRANT_EMBEDDING_SIZE", "1024"))
SPARSE_VOCAB_SIZE = int(os.getenv("QDRANT_SPARSE_VOCAB_SIZE", "30000"))


def _get_embedding_size() -> int:
    """Get embedding size from provider or use default."""
    try:
        from pipeline.embedding_provider import get_embedding_dimensions
        return get_embedding_dimensions()
    except Exception:
        return _DEFAULT_EMBEDDING_SIZE

# OPT-10-005: Prefetch buffer for improved cache utilization
PREFETCH_BUFFER = int(os.getenv("QDRANT_PREFETCH_BUFFER", "5"))

# Check if Qdrant is available
try:
    from qdrant_client import QdrantClient, AsyncQdrantClient
    from qdrant_client.http import models as qmodels
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger.warning("Qdrant client not available. Install with: pip install qdrant-client")


# =============================================================================
# DATA MODELS
# =============================================================================


class HybridSearchResult(TypedDict):
    """Result from hybrid search."""
    id: str
    score: float
    dense_score: Optional[float]
    sparse_score: Optional[float]
    payload: Dict[str, Any]
    vector: Optional[List[float]]


class DocumentInput(TypedDict, total=False):
    """Input document for indexing."""
    id: str
    text: str
    embedding: Optional[List[float]]
    metadata: Dict[str, Any]


class CollectionStats(BaseModel):
    """Statistics about a collection."""
    name: str = Field(..., description="Collection name")
    points_count: int = Field(..., description="Number of points")
    vectors_count: int = Field(..., description="Number of vectors")
    indexed_vectors_count: int = Field(..., description="Indexed vectors")
    status: str = Field(..., description="Collection status")
    disk_data_size: int = Field(default=0, description="Disk size in bytes")
    ram_data_size: int = Field(default=0, description="RAM size in bytes")


class FilterCondition(BaseModel):
    """Filter condition for search."""
    field: str = Field(..., description="Field to filter on")
    operator: str = Field(default="eq", description="Operator: eq, ne, gt, lt, gte, lte, in, contains")
    value: Any = Field(..., description="Value to compare")


# =============================================================================
# SPARSE VECTOR UTILITIES
# =============================================================================


class SparseVectorizer:
    """Simple BM25-style sparse vectorizer.

    Converts text to sparse vectors using term frequency-based scoring.
    In production, would use a proper BM25 implementation.
    """

    def __init__(self, vocab_size: int = SPARSE_VOCAB_SIZE):
        self.vocab_size = vocab_size
        self._idf: Dict[str, float] = {}
        self._doc_count = 0

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into terms."""
        # Simple tokenization - would use better NLP in production
        text = text.lower()
        # Remove punctuation and split
        tokens = re.findall(r'\b[a-z]+\b', text)
        # Remove stopwords (simplified)
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                     'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                     'as', 'into', 'through', 'during', 'before', 'after', 'and',
                     'but', 'or', 'yet', 'so', 'that', 'this', 'these', 'those'}
        return [t for t in tokens if t not in stopwords and len(t) > 1]

    def _hash_term(self, term: str) -> int:
        """Hash term to vocabulary index."""
        return hash(term) % self.vocab_size

    def vectorize(self, text: str) -> Tuple[List[int], List[float]]:
        """Convert text to sparse vector.

        Returns:
            Tuple of (indices, values) for sparse representation
        """
        tokens = self.tokenize(text)
        if not tokens:
            return [], []

        # Count term frequencies
        tf = Counter(tokens)

        # Convert to sparse vector
        indices = []
        values = []

        for term, count in tf.items():
            idx = self._hash_term(term)
            # Simple TF scoring (log-normalized)
            tf_score = 1 + (count / len(tokens)) if count > 0 else 0
            indices.append(idx)
            values.append(tf_score)

        # Sort by index (required for some sparse formats)
        sorted_pairs = sorted(zip(indices, values))
        indices = [p[0] for p in sorted_pairs]
        values = [p[1] for p in sorted_pairs]

        return indices, values

    def update_idf(self, documents: List[str]) -> None:
        """Update IDF values from document collection."""
        self._doc_count += len(documents)
        for doc in documents:
            terms = set(self.tokenize(doc))
            for term in terms:
                self._idf[term] = self._idf.get(term, 0) + 1


# Global sparse vectorizer
_sparse_vectorizer = SparseVectorizer()


def get_sparse_vector(text: str) -> Tuple[List[int], List[float]]:
    """Get sparse vector for text."""
    return _sparse_vectorizer.vectorize(text)


# =============================================================================
# QDRANT HYBRID SEARCH
# =============================================================================


class QdrantHybridSearch:
    """Qdrant client with hybrid search capabilities.

    Combines dense (semantic) and sparse (keyword) search
    for optimal retrieval performance.
    """

    def __init__(
        self,
        url: str = QDRANT_URL,
        api_key: Optional[str] = QDRANT_API_KEY,
        embedding_size: Optional[int] = None,
    ):
        """Initialize Qdrant hybrid search client.

        Args:
            url: Qdrant server URL
            api_key: Optional API key for authentication
            embedding_size: Size of dense embedding vectors (auto-detected if None)
        """
        if not QDRANT_AVAILABLE:
            raise ImportError("Qdrant client not available")

        self.url = url
        self.api_key = api_key
        # Get embedding size from provider if not explicitly set
        self.embedding_size = embedding_size if embedding_size is not None else _get_embedding_size()

        # Initialize clients
        self._sync_client = QdrantClient(url=url, api_key=api_key)
        self._async_client: Optional[AsyncQdrantClient] = None

        # Sparse vectorizer
        self._sparse_vectorizer = SparseVectorizer()

        # Phase 2 FIX: Add CircuitBreaker for Qdrant operations
        self._circuit_breaker = CircuitBreakerRegistry.get_instance().get_or_create(
            circuit_id="qdrant_hybrid",
            failure_threshold=5,
            reset_timeout=30.0,
        )

        # Register with coordinator for unified health monitoring
        coordinator = get_stack_circuit_coordinator()
        coordinator.register_stack("qdrant_hybrid", self._circuit_breaker)

        logger.info(f"Qdrant hybrid search initialized: {url}")

    async def _get_async_client(self) -> AsyncQdrantClient:
        """Get or create async client."""
        if self._async_client is None:
            self._async_client = AsyncQdrantClient(url=self.url, api_key=self.api_key)
        return self._async_client

    async def create_hybrid_collection(
        self,
        name: str,
        sparse_vectors_name: str = "sparse",
        enable_quantization: bool = True,
    ) -> bool:
        """Create a collection with hybrid vector support.

        Args:
            name: Collection name
            sparse_vectors_name: Name for sparse vectors
            enable_quantization: Enable scalar quantization for memory efficiency

        Returns:
            True if successful
        """
        try:
            # Vector configurations
            vectors_config = {
                "dense": qmodels.VectorParams(
                    size=self.embedding_size,
                    distance=qmodels.Distance.COSINE,
                ),
            }

            # Sparse vector configuration
            sparse_vectors_config = {
                sparse_vectors_name: qmodels.SparseVectorParams(),
            }

            # Quantization config for memory efficiency
            quantization_config = None
            if enable_quantization:
                quantization_config = qmodels.ScalarQuantization(
                    scalar=qmodels.ScalarQuantizationConfig(
                        type=qmodels.ScalarType.INT8,
                        always_ram=True,
                    ),
                )

            # Create collection
            client = await self._get_async_client()
            await client.create_collection(
                collection_name=name,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_vectors_config,
                quantization_config=quantization_config,
            )

            logger.info(f"Created hybrid collection: {name}")
            return True

        except Exception as e:
            logger.error(f"Failed to create hybrid collection: {e}")
            return False

    async def add_documents(
        self,
        collection_name: str,
        documents: List[DocumentInput],
        batch_size: int = 100,
        get_embedding: Optional[callable] = None,
    ) -> int:
        """Add documents with hybrid vectors.

        Args:
            collection_name: Target collection
            documents: Documents to add
            batch_size: Batch size for uploads
            get_embedding: Optional function to generate embeddings

        Returns:
            Number of documents added
        """
        if not documents:
            return 0

        client = await self._get_async_client()
        total_added = 0

        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            points = []

            for doc in batch:
                doc_id = doc.get("id", str(hash(doc["text"]))[:16])
                text = doc["text"]
                metadata = doc.get("metadata", {})

                # Get dense embedding
                # C-009 FIX: Never use zero vectors - they produce meaningless search results
                if "embedding" in doc and doc["embedding"]:
                    dense_vector = doc["embedding"]
                elif get_embedding:
                    dense_vector = await get_embedding(text)
                else:
                    # Use configured embedding provider
                    try:
                        from pipeline.embedding_provider import get_embedding_provider
                        provider = get_embedding_provider()
                        dense_vector = provider.embed(text[:8000])  # Limit text size
                    except Exception as e:
                        logger.warning(
                            f"C-009: Skipping document {doc_id} - no embedding provider available: {e}. "
                            "Documents without embeddings will NOT be indexed."
                        )
                        continue  # SKIP document - don't use zero vector

                # Get sparse vector
                sparse_indices, sparse_values = self._sparse_vectorizer.vectorize(text)

                # Create point
                point = qmodels.PointStruct(
                    id=doc_id if isinstance(doc_id, int) else hash(doc_id) % (2**63),
                    vector={
                        "dense": dense_vector,
                    },
                    payload={
                        "text": text,
                        "doc_id": str(doc_id),
                        **metadata,
                        "_indexed_at": datetime.now(timezone.utc).isoformat(),
                    },
                )

                # Add sparse vector if we have one
                if sparse_indices:
                    point.vector["sparse"] = qmodels.SparseVector(
                        indices=sparse_indices,
                        values=sparse_values,
                    )

                points.append(point)

            # Upsert batch
            await client.upsert(
                collection_name=collection_name,
                points=points,
            )

            total_added += len(points)
            logger.debug(f"Added batch of {len(points)} documents")

        logger.info(f"Added {total_added} documents to {collection_name}")
        return total_added

    @enforce_stacks("search", required=["qdrant"], recommended=["langfuse"])
    async def hybrid_search(
        self,
        collection_name: str,
        query: str,
        query_embedding: Optional[List[float]] = None,
        top_k: int = 10,
        filter_conditions: Optional[List[FilterCondition]] = None,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        get_embedding: Optional[callable] = None,
    ) -> List[HybridSearchResult]:
        """Perform hybrid search combining dense and sparse vectors.

        Phase 2 FIX: Uses CircuitBreaker for resilience.

        Args:
            collection_name: Collection to search
            query: Query text
            query_embedding: Optional pre-computed query embedding
            top_k: Number of results to return
            filter_conditions: Optional filter conditions
            dense_weight: Weight for dense (semantic) results
            sparse_weight: Weight for sparse (keyword) results
            get_embedding: Optional function to generate embeddings

        Returns:
            List of search results (empty on circuit open or error)
        """
        # Phase 2: Check circuit breaker first
        if self._circuit_breaker.state == CircuitState.OPEN:
            logger.warning(
                "QDRANT-001: Hybrid search SKIPPED - Circuit breaker OPEN. "
                "Service is degraded. Returning empty results (fail-closed)."
            )
            return []

        try:
            client = await self._get_async_client()
        except Exception as e:
            self._circuit_breaker.record_failure()
            logger.error(
                f"QDRANT-001: Failed to get Qdrant client: {e}. "
                "Returning empty results (fail-closed)."
            )
            return []

        # Get query embedding
        # C-009 FIX: Never use zero vectors for search - they produce meaningless results
        if query_embedding is None:
            if get_embedding:
                query_embedding = await get_embedding(query)
            else:
                # Use configured embedding provider
                try:
                    from pipeline.embedding_provider import get_embedding_provider
                    provider = get_embedding_provider()
                    query_embedding = provider.embed(query[:8000])  # Limit text size
                except Exception as e:
                    logger.error(
                        f"C-009: Hybrid search FAILED - no embedding provider available: {e}. "
                        "Search requires embeddings. Returning empty results."
                    )
                    return []  # FAIL-CLOSED: don't search with zero vector

        # Get sparse query vector
        sparse_indices, sparse_values = self._sparse_vectorizer.vectorize(query)

        # Build filter if provided
        qdrant_filter = None
        if filter_conditions:
            must_conditions = []
            for cond in filter_conditions:
                if cond.operator == "eq":
                    must_conditions.append(
                        qmodels.FieldCondition(
                            key=cond.field,
                            match=qmodels.MatchValue(value=cond.value),
                        )
                    )
                elif cond.operator == "in":
                    must_conditions.append(
                        qmodels.FieldCondition(
                            key=cond.field,
                            match=qmodels.MatchAny(any=cond.value),
                        )
                    )
                elif cond.operator in ("gt", "lt", "gte", "lte"):
                    range_params = {}
                    if cond.operator == "gt":
                        range_params["gt"] = cond.value
                    elif cond.operator == "lt":
                        range_params["lt"] = cond.value
                    elif cond.operator == "gte":
                        range_params["gte"] = cond.value
                    elif cond.operator == "lte":
                        range_params["lte"] = cond.value
                    must_conditions.append(
                        qmodels.FieldCondition(
                            key=cond.field,
                            range=qmodels.Range(**range_params),
                        )
                    )

            if must_conditions:
                qdrant_filter = qmodels.Filter(must=must_conditions)

        # Perform searches with circuit breaker protection
        results = {}

        # OPT-10-005: Prefetch extra results for better cache utilization
        prefetch_limit = top_k * 2 + PREFETCH_BUFFER

        try:
            # Dense search
            dense_results = await client.search(
                collection_name=collection_name,
                query_vector=("dense", query_embedding),
                query_filter=qdrant_filter,
                limit=prefetch_limit,
                with_payload=True,
            )

            for hit in dense_results:
                point_id = str(hit.id)
                if point_id not in results:
                    results[point_id] = {
                        "id": point_id,
                        "dense_score": hit.score,
                        "sparse_score": 0.0,
                        "payload": hit.payload,
                        "vector": None,
                    }
                else:
                    results[point_id]["dense_score"] = hit.score

            # Sparse search (if we have sparse query vector)
            if sparse_indices:
                # OPT-10-005: Use same prefetch limit for sparse search
                sparse_results = await client.search(
                    collection_name=collection_name,
                    query_vector=qmodels.NamedSparseVector(
                        name="sparse",
                        vector=qmodels.SparseVector(
                            indices=sparse_indices,
                            values=sparse_values,
                        ),
                    ),
                    query_filter=qdrant_filter,
                    limit=prefetch_limit,
                    with_payload=True,
                )

                for hit in sparse_results:
                    point_id = str(hit.id)
                    if point_id not in results:
                        results[point_id] = {
                            "id": point_id,
                            "dense_score": 0.0,
                            "sparse_score": hit.score,
                            "payload": hit.payload,
                            "vector": None,
                        }
                    else:
                        results[point_id]["sparse_score"] = hit.score

            # Record success
            self._circuit_breaker.record_success()

        except CircuitOpenError:
            logger.warning(
                "QDRANT-001: Hybrid search BLOCKED - Circuit breaker triggered during call. "
                "Returning empty results (fail-closed)."
            )
            return []
        except Exception as e:
            # Record failure
            self._circuit_breaker.record_failure()
            logger.error(
                f"QDRANT-001: Hybrid search FAILED: {e}. "
                "Returning empty results (fail-closed)."
            )
            return []

        # Fuse results using weighted scoring
        for result in results.values():
            result["score"] = (
                dense_weight * (result["dense_score"] or 0.0) +
                sparse_weight * (result["sparse_score"] or 0.0)
            )

        # Sort by fused score and return top_k
        sorted_results = sorted(
            results.values(),
            key=lambda x: x["score"],
            reverse=True,
        )[:top_k]

        return sorted_results

    async def get_collection_stats(self, collection_name: str) -> Optional[CollectionStats]:
        """Get statistics for a collection.

        Args:
            collection_name: Collection name

        Returns:
            CollectionStats or None if not found
        """
        try:
            client = await self._get_async_client()
            info = await client.get_collection(collection_name)

            return CollectionStats(
                name=collection_name,
                points_count=info.points_count or 0,
                vectors_count=info.vectors_count or 0,
                indexed_vectors_count=info.indexed_vectors_count or 0,
                status=info.status.value if info.status else "unknown",
                disk_data_size=info.disk_data_size or 0,
                ram_data_size=info.ram_data_size or 0,
            )
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return None

    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection.

        Args:
            collection_name: Collection to delete

        Returns:
            True if successful
        """
        try:
            client = await self._get_async_client()
            await client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False

    async def update_collection_quantization(
        self,
        collection_name: str,
        quantization_type: str = "int8",
        always_ram: bool = True,
    ) -> bool:
        """Update collection quantization settings.

        Args:
            collection_name: Collection to update
            quantization_type: "int8" or "binary"
            always_ram: Keep quantized vectors in RAM

        Returns:
            True if successful
        """
        try:
            client = await self._get_async_client()

            if quantization_type == "int8":
                config = qmodels.ScalarQuantization(
                    scalar=qmodels.ScalarQuantizationConfig(
                        type=qmodels.ScalarType.INT8,
                        always_ram=always_ram,
                    ),
                )
            elif quantization_type == "binary":
                config = qmodels.BinaryQuantization(
                    binary=qmodels.BinaryQuantizationConfig(
                        always_ram=always_ram,
                    ),
                )
            else:
                raise ValueError(f"Unknown quantization type: {quantization_type}")

            await client.update_collection(
                collection_name=collection_name,
                quantization_config=config,
            )

            logger.info(f"Updated quantization for {collection_name}: {quantization_type}")
            return True

        except Exception as e:
            logger.error(f"Failed to update quantization: {e}")
            return False

    async def close(self) -> None:
        """Close client connections."""
        if self._async_client:
            await self._async_client.close()
            self._async_client = None
        logger.debug("Qdrant client closed")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_qdrant_client: Optional[QdrantHybridSearch] = None


def get_qdrant_hybrid() -> QdrantHybridSearch:
    """Get singleton Qdrant hybrid search client."""
    global _qdrant_client
    if _qdrant_client is None:
        if not QDRANT_AVAILABLE:
            raise ImportError("Qdrant client not available")
        _qdrant_client = QdrantHybridSearch()
    return _qdrant_client


async def create_hybrid_collection(
    name: str,
    **kwargs,
) -> bool:
    """Create a hybrid collection.

    Falls back gracefully if Qdrant is unavailable.
    """
    if not QDRANT_AVAILABLE:
        logger.warning("Qdrant not available for collection creation")
        return False

    try:
        client = get_qdrant_hybrid()
        return await client.create_hybrid_collection(name, **kwargs)
    except Exception as e:
        logger.error(f"Failed to create collection: {e}")
        return False


@enforce_stacks("search", required=["qdrant"], recommended=["langfuse"])
async def hybrid_search(
    collection_name: str,
    query: str,
    top_k: int = 10,
    filter_conditions: Optional[List[FilterCondition]] = None,
    **kwargs,
) -> Optional[List[HybridSearchResult]]:
    """Perform hybrid search.

    Falls back gracefully if Qdrant is unavailable.
    """
    if not QDRANT_AVAILABLE:
        logger.warning("Qdrant not available for hybrid search")
        return None

    try:
        client = get_qdrant_hybrid()
        return await client.hybrid_search(
            collection_name=collection_name,
            query=query,
            top_k=top_k,
            filter_conditions=filter_conditions,
            **kwargs,
        )
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        return None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "QDRANT_AVAILABLE",
    # Classes
    "QdrantHybridSearch",
    "SparseVectorizer",
    # Models
    "HybridSearchResult",
    "DocumentInput",
    "CollectionStats",
    "FilterCondition",
    # Functions
    "get_qdrant_hybrid",
    "create_hybrid_collection",
    "hybrid_search",
    "get_sparse_vector",
]
