"""Corrective RAG (CRAG) - Self-Correcting Retrieval Augmented Generation.

CRAG implements retrieval with self-correction by:
1. Evaluating quality of retrieved documents
2. Triggering web search for low-quality retrievals
3. Decomposing documents into knowledge strips
4. Filtering and recomposing relevant information

Based on:
- "Corrective Retrieval Augmented Generation" (CRAG)
- Yan et al., 2024

Usage:
    from pipeline.rag.corrective_rag import CorrectiveRAG

    rag = CorrectiveRAG()

    # Query with automatic correction
    result = await rag.query_with_correction(
        query="What is the latest on climate policy?",
        enable_web_fallback=True,
    )

    # Check correction actions taken
    print(result.action_taken)     # CORRECT, INCORRECT, or AMBIGUOUS
    print(result.web_search_used)  # Whether web fallback was triggered
"""

from __future__ import annotations

import os
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional
from enum import Enum

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
# QDRANT CLIENT (Lazy load - C-005 FIX)
# =============================================================================

_qdrant_client: Optional[Any] = None

def _get_qdrant_client() -> Optional[Any]:
    """Lazy load Qdrant client for retrieval."""
    global _qdrant_client
    if _qdrant_client is None:
        try:
            from pipeline.qdrant_client import get_qdrant_client
            _qdrant_client = get_qdrant_client()
            logger.info("CRAG: Qdrant client initialized successfully")
        except ImportError as e:
            logger.error(f"C-005: Qdrant client not available for CRAG: {e}")
            return None
        except Exception as e:
            logger.error(f"C-005: Failed to initialize Qdrant client for CRAG: {e}")
            return None
    return _qdrant_client


# =============================================================================
# CONFIGURATION
# =============================================================================

RETRIEVAL_QUALITY_THRESHOLD = float(os.getenv("CRAG_QUALITY_THRESHOLD", "0.5"))
AMBIGUITY_RANGE = (0.3, 0.7)  # Scores in this range trigger AMBIGUOUS
WEB_SEARCH_ENABLED = os.getenv("CRAG_WEB_SEARCH_ENABLED", "true").lower() == "true"
MAX_KNOWLEDGE_STRIPS = int(os.getenv("CRAG_MAX_STRIPS", "10"))

# OPT-10-012: Parallel processing configuration
STRIP_BATCH_WORKERS = int(os.getenv("CRAG_STRIP_BATCH_WORKERS", "5"))
STRIP_PROCESS_TIMEOUT = int(os.getenv("CRAG_STRIP_PROCESS_TIMEOUT", "10"))

# CRAG is available (native implementation)
CRAG_AVAILABLE = True


# =============================================================================
# RETRIEVAL EVALUATOR ACTIONS
# =============================================================================


class RetrievalAction(str, Enum):
    """Action based on retrieval quality evaluation."""
    CORRECT = "correct"       # Retrieved docs are relevant, use directly
    INCORRECT = "incorrect"   # Retrieved docs are irrelevant, trigger web search
    AMBIGUOUS = "ambiguous"   # Mixed quality, refine and supplement


class KnowledgeStripType(str, Enum):
    """Type of knowledge strip."""
    FACTUAL = "factual"
    DEFINITION = "definition"
    EXAMPLE = "example"
    STATISTIC = "statistic"
    CITATION = "citation"
    OPINION = "opinion"


# =============================================================================
# DATA MODELS
# =============================================================================


class KnowledgeStrip(BaseModel):
    """A decomposed strip of knowledge from a document."""

    content: str = Field(..., description="The knowledge content")
    strip_type: KnowledgeStripType = Field(default=KnowledgeStripType.FACTUAL)
    source_doc_id: str = Field(default="")
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    is_relevant: bool = Field(default=False)


class RetrievalEvaluation(BaseModel):
    """Evaluation of retrieval quality."""

    action: RetrievalAction = Field(default=RetrievalAction.AMBIGUOUS)
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0)
    per_doc_scores: List[float] = Field(default_factory=list)
    reasoning: str = Field(default="")
    needs_web_search: bool = Field(default=False)


class WebSearchResult(BaseModel):
    """Result from web search fallback."""

    query: str = Field(...)
    results: List[Dict[str, Any]] = Field(default_factory=list)
    search_time_ms: float = Field(default=0.0)
    source: str = Field(default="web")


class CRAGResult(BaseModel):
    """Complete result from Corrective RAG."""

    query: str = Field(...)
    answer: str = Field(default="")

    # Action taken
    action_taken: RetrievalAction = Field(default=RetrievalAction.CORRECT)
    evaluation: RetrievalEvaluation = Field(default_factory=RetrievalEvaluation)

    # Knowledge processing
    knowledge_strips: List[KnowledgeStrip] = Field(default_factory=list)
    relevant_strips_count: int = Field(default=0)
    filtered_strips_count: int = Field(default=0)

    # Sources
    original_documents: List[Dict[str, Any]] = Field(default_factory=list)
    web_search_used: bool = Field(default=False)
    web_results: Optional[WebSearchResult] = None

    # Timing
    total_time_ms: float = Field(default=0.0)
    retrieval_time_ms: float = Field(default=0.0)
    evaluation_time_ms: float = Field(default=0.0)
    correction_time_ms: float = Field(default=0.0)


# =============================================================================
# CORRECTIVE RAG IMPLEMENTATION
# =============================================================================


class CorrectiveRAG:
    """Corrective RAG with self-correction and web fallback.

    Implements the CRAG paradigm where:
    1. Retrieved documents are evaluated for quality
    2. Low-quality retrievals trigger web search
    3. Documents are decomposed into knowledge strips
    4. Relevant strips are filtered and recomposed
    """

    def __init__(
        self,
        quality_threshold: float = RETRIEVAL_QUALITY_THRESHOLD,
        enable_web_search: bool = WEB_SEARCH_ENABLED,
        max_strips: int = MAX_KNOWLEDGE_STRIPS,
    ):
        self.quality_threshold = quality_threshold
        self.enable_web_search = enable_web_search
        self.max_strips = max_strips

        self._stats = {
            "total_queries": 0,
            "correct_actions": 0,
            "incorrect_actions": 0,
            "ambiguous_actions": 0,
            "web_searches_triggered": 0,
            "strips_generated": 0,
            "strips_filtered": 0,
        }

        # Phase 2 FIX: Add CircuitBreaker for Qdrant operations
        self._circuit_breaker = CircuitBreakerRegistry.get_instance().get_or_create(
            circuit_id="crag_qdrant",
            failure_threshold=5,
            reset_timeout=30.0,
        )

        # Register with coordinator for unified health monitoring
        coordinator = get_stack_circuit_coordinator()
        coordinator.register_stack("crag_qdrant", self._circuit_breaker)

        logger.info(f"CorrectiveRAG initialized (threshold={quality_threshold})")

    async def query_with_correction(
        self,
        query: str,
        enable_web_fallback: bool = True,
        documents: Optional[List[Dict[str, Any]]] = None,
    ) -> CRAGResult:
        """Query with automatic correction based on retrieval quality.

        Args:
            query: The query to process
            enable_web_fallback: Whether to use web search for poor retrievals
            documents: Optional pre-retrieved documents

        Returns:
            CRAGResult with correction details
        """
        import time
        start_time = time.time()

        self._stats["total_queries"] += 1

        # Step 1: Retrieve documents
        retrieve_start = time.time()
        if documents:
            retrieved_docs = documents
            retrieval_success = True
        else:
            retrieval_result = await self._retrieve_documents(query)
            retrieved_docs = retrieval_result.documents
            retrieval_success = retrieval_result.success
            if not retrieval_success:
                logger.warning(
                    f"CRAG: Retrieval service error - {retrieval_result.error_message}. "
                    "Proceeding with empty documents."
                )
        retrieval_time = (time.time() - retrieve_start) * 1000

        # Step 2: Evaluate retrieval quality
        eval_start = time.time()
        evaluation = await self._evaluate_retrieval(query, retrieved_docs)
        eval_time = (time.time() - eval_start) * 1000

        # Step 3: Take corrective action based on evaluation
        correction_start = time.time()
        web_search_used = False
        web_results = None

        if evaluation.action == RetrievalAction.CORRECT:
            self._stats["correct_actions"] += 1
            # Use retrieved documents directly
            final_docs = retrieved_docs

        elif evaluation.action == RetrievalAction.INCORRECT:
            self._stats["incorrect_actions"] += 1
            # Trigger web search if enabled
            if enable_web_fallback and self.enable_web_search:
                web_results = await self._web_search(query)
                web_search_used = True
                self._stats["web_searches_triggered"] += 1
                # Use web results instead
                final_docs = web_results.results if web_results else []
            else:
                final_docs = []

        else:  # AMBIGUOUS
            self._stats["ambiguous_actions"] += 1
            # Combine retrieved docs with optional web search
            final_docs = retrieved_docs
            if enable_web_fallback and self.enable_web_search:
                web_results = await self._web_search(query)
                web_search_used = True
                self._stats["web_searches_triggered"] += 1
                # Merge web results with retrieved docs
                if web_results:
                    final_docs = final_docs + web_results.results

        # Step 4: Decompose documents into knowledge strips
        knowledge_strips = await self._decompose_to_strips(query, final_docs)
        self._stats["strips_generated"] += len(knowledge_strips)

        # Step 5: Filter relevant strips
        relevant_strips = [s for s in knowledge_strips if s.is_relevant]
        filtered_count = len(knowledge_strips) - len(relevant_strips)
        self._stats["strips_filtered"] += filtered_count

        correction_time = (time.time() - correction_start) * 1000

        # Step 6: Generate answer from relevant strips
        answer = await self._generate_answer(query, relevant_strips[:self.max_strips])

        total_time = (time.time() - start_time) * 1000

        return CRAGResult(
            query=query,
            answer=answer,
            action_taken=evaluation.action,
            evaluation=evaluation,
            knowledge_strips=relevant_strips[:self.max_strips],
            relevant_strips_count=len(relevant_strips),
            filtered_strips_count=filtered_count,
            original_documents=retrieved_docs,
            web_search_used=web_search_used,
            web_results=web_results,
            total_time_ms=total_time,
            retrieval_time_ms=retrieval_time,
            evaluation_time_ms=eval_time,
            correction_time_ms=correction_time,
        )

    async def _retrieve_documents(self, query: str) -> RetrievalResult:
        """Retrieve documents for query.

        C-005 FIX: Uses real Qdrant retrieval instead of fake documents.
        Phase 2 FIX: Uses CircuitBreaker for resilience.

        Returns RetrievalResult:
        - empty() when service works but no results (success=True)
        - service_error() when service fails (success=False, triggers circuit)
        - NEVER returns fake documents.
        """
        # Phase 2: Check circuit breaker first
        if self._circuit_breaker.state == CircuitState.OPEN:
            logger.warning(
                "C-005: CRAG retrieval SKIPPED - Circuit breaker OPEN. "
                "Service is degraded. Returning service_error (fail-closed)."
            )
            return RetrievalResult.service_error(
                service_name="crag_qdrant",
                error_message="Circuit breaker open - service degraded",
            )

        client = _get_qdrant_client()
        if client is None:
            # Record failure for circuit breaker
            self._circuit_breaker.record_failure()
            logger.error(
                "C-005: CRAG retrieval FAILED - Qdrant not available. "
                "Returning service_error (fail-closed). No fake documents will be generated."
            )
            return RetrievalResult.service_error(
                service_name="crag_qdrant",
                error_message="Qdrant client not available",
            )

        try:
            results = client.search_similar(
                collection="crag_documents",
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
            logger.debug(f"CRAG: Retrieved {len(documents)} documents for query")

            if not documents:
                # Service worked but no results - this is success=True
                return RetrievalResult.empty(source="crag_qdrant")

            return RetrievalResult(
                documents=documents,
                source="crag_qdrant",
                success=True,
                error_message=None,
            )

        except CircuitOpenError:
            logger.warning(
                "C-005: CRAG retrieval BLOCKED - Circuit breaker triggered during call. "
                "Returning service_error (fail-closed)."
            )
            return RetrievalResult.service_error(
                service_name="crag_qdrant",
                error_message="Circuit breaker opened during operation",
            )
        except Exception as e:
            # Record failure for circuit breaker
            self._circuit_breaker.record_failure()
            logger.error(
                f"C-005: CRAG retrieval FAILED - Qdrant search error: {e}. "
                "Returning service_error (fail-closed). No fake documents will be generated."
            )
            return RetrievalResult.service_error(
                service_name="crag_qdrant",
                error_message=str(e),
            )

    async def _evaluate_retrieval(
        self,
        query: str,
        documents: List[Dict[str, Any]],
    ) -> RetrievalEvaluation:
        """Evaluate quality of retrieved documents."""
        if not documents:
            return RetrievalEvaluation(
                action=RetrievalAction.INCORRECT,
                overall_score=0.0,
                reasoning="No documents retrieved",
                needs_web_search=True,
            )

        # Calculate per-document scores
        per_doc_scores = []
        query_words = set(query.lower().split())

        for doc in documents:
            content = doc.get("content", "").lower()
            doc_words = set(content.split())

            # Simple relevance calculation
            overlap = len(query_words & doc_words)
            base_score = doc.get("score", 0.5)
            relevance = min(1.0, overlap / max(len(query_words), 1) + base_score * 0.3)
            per_doc_scores.append(relevance)

        # Overall score is weighted average (higher weight for top docs)
        weights = [1.0 / (i + 1) for i in range(len(per_doc_scores))]
        total_weight = sum(weights)
        overall = sum(s * w for s, w in zip(per_doc_scores, weights)) / total_weight

        # Determine action
        if overall >= AMBIGUITY_RANGE[1]:
            action = RetrievalAction.CORRECT
            reasoning = f"High-quality retrieval (score={overall:.2f})"
        elif overall <= AMBIGUITY_RANGE[0]:
            action = RetrievalAction.INCORRECT
            reasoning = f"Low-quality retrieval (score={overall:.2f}), web search recommended"
        else:
            action = RetrievalAction.AMBIGUOUS
            reasoning = f"Mixed quality retrieval (score={overall:.2f}), refinement needed"

        return RetrievalEvaluation(
            action=action,
            overall_score=overall,
            per_doc_scores=per_doc_scores,
            reasoning=reasoning,
            needs_web_search=action != RetrievalAction.CORRECT,
        )

    async def _web_search(self, query: str) -> WebSearchResult:
        """Perform web search as fallback.

        C-006 FIX: Uses real Tavily API instead of fake results.
        Returns empty results on failure (fail-closed) - NEVER returns fake web results.
        """
        import time
        start = time.time()

        try:
            # Try to use Tavily API
            tavily_api_key = os.getenv("TAVILY_API_KEY")
            if not tavily_api_key:
                logger.error(
                    "C-006: CRAG web search FAILED - TAVILY_API_KEY not set. "
                    "Returning empty results (fail-closed). No fake web results will be generated."
                )
                return WebSearchResult(
                    query=query,
                    results=[],
                    search_time_ms=(time.time() - start) * 1000,
                    source="error_no_api_key",
                )

            from tavily import TavilyClient
            client = TavilyClient(api_key=tavily_api_key)

            response = client.search(query=query, max_results=3)

            results = []
            for item in response.get("results", []):
                results.append({
                    "id": item.get("url", ""),
                    "content": item.get("content", ""),
                    "url": item.get("url", ""),
                    "title": item.get("title", ""),
                })

            search_time = (time.time() - start) * 1000
            logger.debug(f"CRAG: Web search returned {len(results)} results")

            return WebSearchResult(
                query=query,
                results=results,
                search_time_ms=search_time,
                source="tavily",
            )

        except ImportError:
            logger.error(
                "C-006: CRAG web search FAILED - Tavily library not installed. "
                "Returning empty results (fail-closed). No fake web results will be generated."
            )
            return WebSearchResult(
                query=query,
                results=[],
                search_time_ms=(time.time() - start) * 1000,
                source="error_no_tavily",
            )
        except Exception as e:
            logger.error(
                f"C-006: CRAG web search FAILED: {e}. "
                "Returning empty results (fail-closed). No fake web results will be generated."
            )
            return WebSearchResult(
                query=query,
                results=[],
                search_time_ms=(time.time() - start) * 1000,
                source=f"error_{type(e).__name__}",
            )

    def _process_single_sentence(
        self,
        sent: str,
        doc_id: str,
        query_words: set,
    ) -> Optional[KnowledgeStrip]:
        """Process a single sentence into a knowledge strip.

        OPT-10-012: Extracted for parallel processing.
        """
        if len(sent.strip()) < 10:
            return None

        # Calculate relevance
        sent_words = set(sent.lower().split())
        overlap = len(query_words & sent_words)
        relevance = min(1.0, overlap / max(len(query_words), 1))

        # Classify strip type
        sent_lower = sent.lower()
        if any(w in sent_lower for w in ["is defined as", "means", "refers to"]):
            strip_type = KnowledgeStripType.DEFINITION
        elif any(w in sent_lower for w in ["%", "percent", "million", "billion"]):
            strip_type = KnowledgeStripType.STATISTIC
        elif any(w in sent_lower for w in ["for example", "such as", "e.g."]):
            strip_type = KnowledgeStripType.EXAMPLE
        elif any(w in sent_lower for w in ["according to", "study", "research"]):
            strip_type = KnowledgeStripType.CITATION
        else:
            strip_type = KnowledgeStripType.FACTUAL

        return KnowledgeStrip(
            content=sent.strip(),
            strip_type=strip_type,
            source_doc_id=doc_id,
            relevance_score=relevance,
            confidence=0.7,
            is_relevant=relevance >= 0.3,
        )

    async def _decompose_to_strips(
        self,
        query: str,
        documents: List[Dict[str, Any]],
    ) -> List[KnowledgeStrip]:
        """Decompose documents into knowledge strips.

        OPT-10-012: Uses ThreadPoolExecutor for parallel processing.
        """
        strips = []
        query_words = set(query.lower().split())

        # Collect all sentences with their metadata
        sentence_tasks = []
        for doc in documents:
            content = doc.get("content", "")
            doc_id = doc.get("id", "unknown")
            sentences = content.split(". ")
            for sent in sentences:
                sentence_tasks.append((sent, doc_id))

        # OPT-10-012: Process sentences in parallel batches
        if len(sentence_tasks) > STRIP_BATCH_WORKERS:
            with ThreadPoolExecutor(max_workers=STRIP_BATCH_WORKERS) as executor:
                futures = [
                    executor.submit(
                        self._process_single_sentence,
                        sent,
                        doc_id,
                        query_words,
                    )
                    for sent, doc_id in sentence_tasks
                ]

                for future in as_completed(futures, timeout=STRIP_PROCESS_TIMEOUT):
                    try:
                        result = future.result()
                        if result is not None:
                            strips.append(result)
                    except Exception as e:
                        logger.debug(f"OPT-10-012: Strip processing error: {e}")
        else:
            # For small batches, process sequentially to avoid overhead
            for sent, doc_id in sentence_tasks:
                result = self._process_single_sentence(sent, doc_id, query_words)
                if result is not None:
                    strips.append(result)

        # Sort by relevance
        strips.sort(key=lambda x: x.relevance_score, reverse=True)

        return strips

    async def _generate_answer(
        self,
        query: str,
        strips: List[KnowledgeStrip],
    ) -> str:
        """Generate answer from knowledge strips."""
        if not strips:
            return f"Unable to find relevant information for: {query}"

        # Recompose strips into answer
        strip_contents = [s.content for s in strips[:5]]
        combined = " ".join(strip_contents)

        return f"Based on the evidence: {combined[:500]}"

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        return {
            **self._stats,
            "quality_threshold": self.quality_threshold,
            "web_search_enabled": self.enable_web_search,
        }


# =============================================================================
# SINGLETON AND CONVENIENCE FUNCTIONS
# =============================================================================

_crag_instance: Optional[CorrectiveRAG] = None


def get_corrective_rag() -> CorrectiveRAG:
    """Get or create CorrectiveRAG singleton."""
    global _crag_instance
    if _crag_instance is None:
        _crag_instance = CorrectiveRAG()
    return _crag_instance


async def query_with_correction(
    query: str,
    enable_web_fallback: bool = True,
) -> CRAGResult:
    """Convenience function for corrective query."""
    rag = get_corrective_rag()
    return await rag.query_with_correction(
        query=query,
        enable_web_fallback=enable_web_fallback,
    )


__all__ = [
    "CRAG_AVAILABLE",
    "CorrectiveRAG",
    "CRAGResult",
    "RetrievalAction",
    "RetrievalEvaluation",
    "KnowledgeStrip",
    "KnowledgeStripType",
    "WebSearchResult",
    "get_corrective_rag",
    "query_with_correction",
]
