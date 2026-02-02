"""MemoRAG - Memory-Augmented Retrieval with State Compression.

MemoRAG implements memory-efficient RAG by:
1. Compressing retrieval states into compact representations
2. Maintaining episodic memory of past retrievals
3. Learning retrieval patterns across sessions
4. Reducing redundant retrievals through memoization

Based on:
- "MemoRAG: Moving towards Next-Gen RAG Via Memory-Inspired Knowledge Discovery"
- Qian et al., 2024

Usage:
    from pipeline.rag.memo_rag import MemoRAG

    rag = MemoRAG()

    # Query with memory augmentation
    result = await rag.query_with_memory(
        query="What is the evidence for climate change?",
        use_episodic_memory=True,
    )

    # Compress state after sprint
    compressed = await rag.compress_sprint_state(sprint_id="S05")
"""

from __future__ import annotations

import os
import json
import hashlib
import logging
from typing import Any, Dict, List, Optional, Set
from datetime import datetime, timezone
from collections import defaultdict

from pydantic import BaseModel, Field

# Phase 2 FIX: Import resilience infrastructure
from pipeline.retry_config import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitOpenError,
    CircuitState,
)
from pipeline.resilience import RetrievalResult
from pipeline.resilience.coordinator import get_stack_circuit_coordinator

logger = logging.getLogger(__name__)

# =============================================================================
# QDRANT CLIENT (Lazy load - C-007 FIX)
# =============================================================================

_qdrant_client: Optional[Any] = None

def _get_qdrant_client() -> Optional[Any]:
    """Lazy load Qdrant client for retrieval."""
    global _qdrant_client
    if _qdrant_client is None:
        try:
            from pipeline.qdrant_client import get_qdrant_client
            _qdrant_client = get_qdrant_client()
            logger.info("MemoRAG: Qdrant client initialized successfully")
        except ImportError as e:
            logger.error(f"C-007: Qdrant client not available for MemoRAG: {e}")
            return None
        except Exception as e:
            logger.error(f"C-007: Failed to initialize Qdrant client for MemoRAG: {e}")
            return None
    return _qdrant_client


# =============================================================================
# CONFIGURATION
# =============================================================================

MEMO_CACHE_SIZE = int(os.getenv("MEMO_CACHE_SIZE", "1000"))
COMPRESSION_RATIO = float(os.getenv("MEMO_COMPRESSION_RATIO", "0.3"))
EPISODIC_MEMORY_WINDOW = int(os.getenv("EPISODIC_MEMORY_WINDOW", "100"))

# MemoRAG is available if dependencies are met
try:
    import numpy as np
    MEMO_RAG_AVAILABLE = True
except ImportError:
    MEMO_RAG_AVAILABLE = False
    np = None


# =============================================================================
# DATA MODELS
# =============================================================================


class MemoryEntry(BaseModel):
    """An entry in episodic memory."""

    query_hash: str = Field(..., description="Hash of the original query")
    query_text: str = Field(..., description="Original query text")
    compressed_state: Dict[str, Any] = Field(default_factory=dict)
    relevance_score: float = Field(default=0.0)
    access_count: int = Field(default=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    documents_retrieved: int = Field(default=0)
    retrieval_time_ms: float = Field(default=0.0)


class CompressedState(BaseModel):
    """Compressed retrieval state."""

    sprint_id: str = Field(..., description="Sprint this state belongs to")
    query_clusters: List[Dict[str, Any]] = Field(default_factory=list)
    document_signatures: List[str] = Field(default_factory=list)
    topic_distribution: Dict[str, float] = Field(default_factory=dict)
    compression_ratio: float = Field(default=0.0)
    original_size: int = Field(default=0)
    compressed_size: int = Field(default=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MemoRAGResult(BaseModel):
    """Result from MemoRAG query."""

    query: str = Field(...)
    answer: str = Field(default="")
    documents: List[Dict[str, Any]] = Field(default_factory=list)
    memory_hit: bool = Field(default=False)
    memory_similarity: float = Field(default=0.0)
    retrieval_time_ms: float = Field(default=0.0)
    from_cache: bool = Field(default=False)
    compression_applied: bool = Field(default=False)


class SprintStateCompression(BaseModel):
    """Compressed state for an entire sprint."""

    sprint_id: str = Field(...)
    total_queries: int = Field(default=0)
    unique_documents: int = Field(default=0)
    cluster_count: int = Field(default=0)
    compression_ratio: float = Field(default=0.0)
    topic_summary: Dict[str, float] = Field(default_factory=dict)
    key_entities: List[str] = Field(default_factory=list)
    memory_footprint_kb: float = Field(default=0.0)


# =============================================================================
# MEMO RAG IMPLEMENTATION
# =============================================================================


class MemoRAG:
    """Memory-Augmented RAG with state compression.

    MemoRAG maintains episodic memory of retrievals and compresses
    state for efficient long-term storage and retrieval.
    """

    def __init__(
        self,
        cache_size: int = MEMO_CACHE_SIZE,
        compression_ratio: float = COMPRESSION_RATIO,
        episodic_window: int = EPISODIC_MEMORY_WINDOW,
    ):
        self.cache_size = cache_size
        self.compression_ratio = compression_ratio
        self.episodic_window = episodic_window

        # Episodic memory (LRU-like)
        self._episodic_memory: Dict[str, MemoryEntry] = {}
        self._memory_access_order: List[str] = []

        # Sprint state tracking
        self._sprint_states: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # Statistics
        self._stats = {
            "total_queries": 0,
            "memory_hits": 0,
            "compressions": 0,
            "cache_evictions": 0,
        }

        # Phase 2 FIX: Add CircuitBreaker for Qdrant operations
        self._circuit_breaker = CircuitBreakerRegistry.get_instance().get_or_create(
            circuit_id="memo_rag_qdrant",
            failure_threshold=5,
            reset_timeout=30.0,
        )

        # Register with coordinator for unified health monitoring
        coordinator = get_stack_circuit_coordinator()
        coordinator.register_stack("memo_rag_qdrant", self._circuit_breaker)

        logger.info(f"MemoRAG initialized (cache_size={cache_size})")

    def _hash_query(self, query: str) -> str:
        """Generate hash for query."""
        return hashlib.sha256(query.lower().strip().encode()).hexdigest()[:16]

    def _update_access_order(self, query_hash: str) -> None:
        """Update LRU access order."""
        if query_hash in self._memory_access_order:
            self._memory_access_order.remove(query_hash)
        self._memory_access_order.append(query_hash)

        # Evict if over capacity
        while len(self._memory_access_order) > self.cache_size:
            evict_hash = self._memory_access_order.pop(0)
            if evict_hash in self._episodic_memory:
                del self._episodic_memory[evict_hash]
                self._stats["cache_evictions"] += 1

    async def query_with_memory(
        self,
        query: str,
        use_episodic_memory: bool = True,
        similarity_threshold: float = 0.85,
        sprint_id: Optional[str] = None,
    ) -> MemoRAGResult:
        """Query with episodic memory augmentation.

        Args:
            query: The query to process
            use_episodic_memory: Whether to check episodic memory first
            similarity_threshold: Threshold for memory hit
            sprint_id: Optional sprint ID for state tracking

        Returns:
            MemoRAGResult with answer and metadata
        """
        import time
        start_time = time.time()

        self._stats["total_queries"] += 1
        query_hash = self._hash_query(query)

        # Check episodic memory
        memory_hit = False
        memory_similarity = 0.0
        from_cache = False

        if use_episodic_memory and query_hash in self._episodic_memory:
            entry = self._episodic_memory[query_hash]
            entry.access_count += 1
            entry.last_accessed = datetime.now(timezone.utc)
            self._update_access_order(query_hash)

            memory_hit = True
            memory_similarity = 1.0  # Exact match
            from_cache = True
            self._stats["memory_hits"] += 1

            retrieval_time = (time.time() - start_time) * 1000

            return MemoRAGResult(
                query=query,
                answer=entry.compressed_state.get("answer", ""),
                documents=entry.compressed_state.get("documents", []),
                memory_hit=memory_hit,
                memory_similarity=memory_similarity,
                retrieval_time_ms=retrieval_time,
                from_cache=from_cache,
            )

        # Perform actual retrieval
        retrieval_result = await self._retrieve_documents(query)
        documents = retrieval_result.documents
        if not retrieval_result.success:
            logger.warning(
                f"MemoRAG: Retrieval service error - {retrieval_result.error_message}. "
                "Proceeding with empty documents."
            )
        answer = await self._generate_answer(query, documents)

        # Store in episodic memory
        compressed_state = {
            "answer": answer,
            "documents": documents[:5],  # Keep top 5
            "query_embedding_hash": query_hash,
        }

        entry = MemoryEntry(
            query_hash=query_hash,
            query_text=query,
            compressed_state=compressed_state,
            documents_retrieved=len(documents),
            retrieval_time_ms=(time.time() - start_time) * 1000,
        )

        self._episodic_memory[query_hash] = entry
        self._update_access_order(query_hash)

        # Track for sprint state
        if sprint_id:
            self._sprint_states[sprint_id].append({
                "query": query,
                "query_hash": query_hash,
                "documents": len(documents),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        retrieval_time = (time.time() - start_time) * 1000

        return MemoRAGResult(
            query=query,
            answer=answer,
            documents=documents,
            memory_hit=memory_hit,
            memory_similarity=memory_similarity,
            retrieval_time_ms=retrieval_time,
            from_cache=from_cache,
        )

    async def _retrieve_documents(self, query: str) -> RetrievalResult:
        """Retrieve documents for query.

        C-007 FIX: Uses real Qdrant retrieval instead of fake documents.
        Phase 2 FIX: Uses CircuitBreaker for resilience.

        Returns RetrievalResult:
        - empty() when service works but no results (success=True)
        - service_error() when service fails (success=False, triggers circuit)
        - NEVER returns fake documents.
        """
        # Phase 2: Check circuit breaker first
        if self._circuit_breaker.state == CircuitState.OPEN:
            logger.warning(
                "C-007: MemoRAG retrieval SKIPPED - Circuit breaker OPEN. "
                "Service is degraded. Returning service_error (fail-closed)."
            )
            return RetrievalResult.service_error(
                service_name="memo_rag_qdrant",
                error_message="Circuit breaker open - service degraded",
            )

        client = _get_qdrant_client()
        if client is None:
            # Record failure for circuit breaker
            self._circuit_breaker.record_failure()
            logger.error(
                "C-007: MemoRAG retrieval FAILED - Qdrant not available. "
                "Returning service_error (fail-closed). No fake documents will be generated."
            )
            return RetrievalResult.service_error(
                service_name="memo_rag_qdrant",
                error_message="Qdrant client not available",
            )

        try:
            results = client.search_similar(
                collection="memo_rag_documents",
                query_text=query,
                top_k=5,
            )

            documents = []
            for hit in results:
                documents.append({
                    "id": str(hit.id) if hasattr(hit, 'id') else str(hit.get("id", "unknown")),
                    "content": hit.payload.get("content", "") if hasattr(hit, 'payload') else hit.get("content", ""),
                    "score": float(hit.score) if hasattr(hit, 'score') else float(hit.get("score", 0.0)),
                })

            # Record success for circuit breaker
            self._circuit_breaker.record_success()
            logger.debug(f"MemoRAG: Retrieved {len(documents)} documents for query")

            if not documents:
                # Service worked but no results - this is success=True
                return RetrievalResult.empty(source="memo_rag_qdrant")

            return RetrievalResult(
                documents=documents,
                source="memo_rag_qdrant",
                success=True,
                error_message=None,
            )

        except CircuitOpenError:
            logger.warning(
                "C-007: MemoRAG retrieval BLOCKED - Circuit breaker triggered during call. "
                "Returning service_error (fail-closed)."
            )
            return RetrievalResult.service_error(
                service_name="memo_rag_qdrant",
                error_message="Circuit breaker opened during operation",
            )
        except Exception as e:
            # Record failure for circuit breaker
            self._circuit_breaker.record_failure()
            logger.error(
                f"C-007: MemoRAG retrieval FAILED - Qdrant search error: {e}. "
                "Returning service_error (fail-closed). No fake documents will be generated."
            )
            return RetrievalResult.service_error(
                service_name="memo_rag_qdrant",
                error_message=str(e),
            )

    async def _generate_answer(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """Generate answer from documents.

        C-007 FIX: Uses real LLM generation instead of fake answers.
        Returns empty string on failure (fail-closed) - NEVER returns fake answers.
        """
        if not documents:
            logger.warning(
                "C-007: MemoRAG generation called with no documents. "
                "Returning empty response (fail-closed)."
            )
            return ""

        try:
            from pipeline.claude_cli_llm import execute_agent_task

            context = "\n\n".join([
                f"Document {i+1}:\n{doc.get('content', '')}"
                for i, doc in enumerate(documents[:5])
            ])

            prompt = f"""Based on the following documents, answer the query.

Query: {query}

Documents:
{context}

Provide a concise, factual answer based ONLY on the information in the documents above."""

            result = execute_agent_task(
                task="GENERATE_RESPONSE",
                instruction=prompt,
                workspace_path=os.getcwd(),
                sprint_id="memo_rag",
            )

            if result.get("status") == "success" and result.get("output"):
                return result["output"]

            logger.error(
                f"C-007: MemoRAG LLM generation failed: {result.get('error', 'unknown')}. "
                "Returning empty response (fail-closed)."
            )
            return ""

        except ImportError:
            logger.error(
                "C-007: MemoRAG LLM not available (claude_cli_llm not installed). "
                "Returning empty response (fail-closed). No fake answers will be generated."
            )
            return ""
        except Exception as e:
            logger.error(
                f"C-007: MemoRAG generation FAILED: {e}. "
                "Returning empty response (fail-closed). No fake answers will be generated."
            )
            return ""

    async def compress_sprint_state(
        self,
        sprint_id: str,
        keep_top_k: int = 10,
    ) -> SprintStateCompression:
        """Compress all retrieval state from a sprint.

        Args:
            sprint_id: Sprint to compress
            keep_top_k: Number of top queries to keep detailed

        Returns:
            Compressed sprint state
        """
        self._stats["compressions"] += 1

        sprint_data = self._sprint_states.get(sprint_id, [])

        if not sprint_data:
            return SprintStateCompression(sprint_id=sprint_id)

        # Analyze queries
        unique_docs: Set[str] = set()
        topic_counts: Dict[str, int] = defaultdict(int)

        for entry in sprint_data:
            query = entry.get("query", "")
            # Simple topic extraction (would use NLP in production)
            words = query.lower().split()
            for word in words:
                if len(word) > 4:
                    topic_counts[word] += 1

        # Calculate topic distribution
        total_topics = sum(topic_counts.values()) or 1
        topic_distribution = {
            k: v / total_topics
            for k, v in sorted(topic_counts.items(), key=lambda x: -x[1])[:20]
        }

        # Extract key entities (top topics)
        key_entities = list(topic_distribution.keys())[:10]

        # Calculate compression ratio
        original_size = len(json.dumps(sprint_data))
        compressed_data = {
            "summary": f"Sprint {sprint_id}: {len(sprint_data)} queries",
            "topics": topic_distribution,
            "entities": key_entities,
        }
        compressed_size = len(json.dumps(compressed_data))
        compression_ratio = compressed_size / original_size if original_size > 0 else 0

        return SprintStateCompression(
            sprint_id=sprint_id,
            total_queries=len(sprint_data),
            unique_documents=len(unique_docs),
            cluster_count=min(len(sprint_data), 5),
            compression_ratio=compression_ratio,
            topic_summary=topic_distribution,
            key_entities=key_entities,
            memory_footprint_kb=compressed_size / 1024,
        )

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            **self._stats,
            "episodic_memory_size": len(self._episodic_memory),
            "cache_capacity": self.cache_size,
            "cache_utilization": len(self._episodic_memory) / self.cache_size,
            "sprint_states_tracked": len(self._sprint_states),
        }

    def clear_memory(self, sprint_id: Optional[str] = None) -> None:
        """Clear memory, optionally for specific sprint."""
        if sprint_id:
            if sprint_id in self._sprint_states:
                del self._sprint_states[sprint_id]
        else:
            self._episodic_memory.clear()
            self._memory_access_order.clear()
            self._sprint_states.clear()
            self._stats = {
                "total_queries": 0,
                "memory_hits": 0,
                "compressions": 0,
                "cache_evictions": 0,
            }


# =============================================================================
# SINGLETON AND CONVENIENCE FUNCTIONS
# =============================================================================

_memo_rag_instance: Optional[MemoRAG] = None


def get_memo_rag() -> MemoRAG:
    """Get or create MemoRAG singleton."""
    global _memo_rag_instance
    if _memo_rag_instance is None:
        _memo_rag_instance = MemoRAG()
    return _memo_rag_instance


async def query_with_memory(
    query: str,
    use_episodic_memory: bool = True,
    sprint_id: Optional[str] = None,
) -> MemoRAGResult:
    """Convenience function for memory-augmented query."""
    rag = get_memo_rag()
    return await rag.query_with_memory(
        query=query,
        use_episodic_memory=use_episodic_memory,
        sprint_id=sprint_id,
    )


async def compress_sprint_state(sprint_id: str) -> SprintStateCompression:
    """Convenience function for sprint state compression."""
    rag = get_memo_rag()
    return await rag.compress_sprint_state(sprint_id)


__all__ = [
    "MEMO_RAG_AVAILABLE",
    "MemoRAG",
    "MemoRAGResult",
    "MemoryEntry",
    "CompressedState",
    "SprintStateCompression",
    "get_memo_rag",
    "query_with_memory",
    "compress_sprint_state",
]
