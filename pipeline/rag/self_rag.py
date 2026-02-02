"""Self-RAG - Self-Reflective Retrieval Augmented Generation.

Self-RAG implements retrieval with self-reflection by:
1. Deciding whether retrieval is needed for a query
2. Evaluating relevance of retrieved documents
3. Assessing whether generated response is supported by evidence
4. Critiquing response utility and factual accuracy

Based on:
- "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"
- Asai et al., 2023

Usage:
    from pipeline.rag.self_rag import SelfRAG

    rag = SelfRAG()

    # Query with self-reflection
    result = await rag.query_with_reflection(
        query="What causes climate change?",
        critique_response=True,
    )

    # Check reflection tokens
    print(result.retrieval_needed)  # Whether retrieval was deemed necessary
    print(result.relevance_scores)  # Per-document relevance
    print(result.support_score)     # How well evidence supports response
"""

from __future__ import annotations

import os
import logging
import time
from typing import Any, Dict, List, Optional, Union
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
# QDRANT CLIENT (Lazy load - C-004 FIX)
# =============================================================================

_qdrant_client: Optional[Any] = None

def _get_qdrant_client() -> Optional[Any]:
    """Lazy load Qdrant client for retrieval."""
    global _qdrant_client
    if _qdrant_client is None:
        try:
            from pipeline.qdrant_client import get_qdrant_client
            _qdrant_client = get_qdrant_client()
            logger.info("Self-RAG: Qdrant client initialized successfully")
        except ImportError as e:
            logger.error(f"C-004: Qdrant client not available - retrieval will fail: {e}")
            return None
        except Exception as e:
            logger.error(f"C-004: Failed to initialize Qdrant client: {e}")
            return None
    return _qdrant_client


# =============================================================================
# CONFIGURATION
# =============================================================================

RELEVANCE_THRESHOLD = float(os.getenv("SELF_RAG_RELEVANCE_THRESHOLD", "0.7"))
SUPPORT_THRESHOLD = float(os.getenv("SELF_RAG_SUPPORT_THRESHOLD", "0.6"))
MAX_CRITIQUE_ITERATIONS = int(os.getenv("SELF_RAG_MAX_ITERATIONS", "3"))

# OPT-10-010: Early exit threshold for high confidence
HIGH_CONFIDENCE_THRESHOLD = float(os.getenv("SELF_RAG_HIGH_CONFIDENCE_THRESHOLD", "0.85"))

# Self-RAG is available (native implementation)
SELF_RAG_AVAILABLE = True


# =============================================================================
# REFLECTION TOKENS (Special tokens for self-reflection)
# =============================================================================


class RetrievalToken(str, Enum):
    """Token indicating whether retrieval is needed."""
    RETRIEVE = "retrieve"       # Retrieval is necessary
    NO_RETRIEVE = "no_retrieve" # Can answer without retrieval
    UNCERTAIN = "uncertain"     # Uncertain, retrieve to be safe


class RelevanceToken(str, Enum):
    """Token indicating document relevance."""
    RELEVANT = "relevant"           # Document is relevant
    PARTIALLY_RELEVANT = "partial"  # Partially relevant
    IRRELEVANT = "irrelevant"       # Not relevant


class SupportToken(str, Enum):
    """Token indicating how well response is supported."""
    FULLY_SUPPORTED = "fully_supported"      # Fully supported by evidence
    PARTIALLY_SUPPORTED = "partially"        # Partially supported
    NO_SUPPORT = "no_support"                # Not supported by evidence
    CONTRADICTED = "contradicted"            # Contradicts evidence


class UtilityToken(str, Enum):
    """Token indicating response utility."""
    UTILITY_5 = "5"  # Very useful
    UTILITY_4 = "4"  # Useful
    UTILITY_3 = "3"  # Somewhat useful
    UTILITY_2 = "2"  # Not very useful
    UTILITY_1 = "1"  # Not useful


# =============================================================================
# DATA MODELS
# =============================================================================


class ReflectionResult(BaseModel):
    """Result of self-reflection on a single aspect."""

    aspect: str = Field(..., description="What was reflected on")
    decision: str = Field(..., description="The reflection decision/token")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: str = Field(default="")


class DocumentRelevance(BaseModel):
    """Relevance assessment for a document."""

    document_id: str = Field(...)
    content_preview: str = Field(default="")
    relevance: RelevanceToken = Field(default=RelevanceToken.IRRELEVANT)
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    key_overlap: List[str] = Field(default_factory=list)
    reasoning: str = Field(default="")


class SupportAssessment(BaseModel):
    """Assessment of how well evidence supports response."""

    support_token: SupportToken = Field(default=SupportToken.NO_SUPPORT)
    support_score: float = Field(default=0.0, ge=0.0, le=1.0)
    supporting_docs: List[str] = Field(default_factory=list)
    contradicting_docs: List[str] = Field(default_factory=list)
    unsupported_claims: List[str] = Field(default_factory=list)
    reasoning: str = Field(default="")


class CritiqueResult(BaseModel):
    """Result of response critique."""

    utility: UtilityToken = Field(default=UtilityToken.UTILITY_3)
    utility_score: float = Field(default=0.5, ge=0.0, le=1.0)
    factual_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    completeness: float = Field(default=0.0, ge=0.0, le=1.0)
    clarity: float = Field(default=0.0, ge=0.0, le=1.0)
    improvements: List[str] = Field(default_factory=list)


class SelfRAGResult(BaseModel):
    """Complete result from Self-RAG query."""

    query: str = Field(...)
    answer: str = Field(default="")

    # Retrieval decision
    retrieval_needed: RetrievalToken = Field(default=RetrievalToken.RETRIEVE)
    retrieval_reasoning: str = Field(default="")

    # Retrieved documents with relevance
    documents: List[Dict[str, Any]] = Field(default_factory=list)
    document_relevances: List[DocumentRelevance] = Field(default_factory=list)

    # Support assessment
    support: SupportAssessment = Field(default_factory=SupportAssessment)

    # Critique (if enabled)
    critique: Optional[CritiqueResult] = None

    # Iterations (if refinement was needed)
    iterations: int = Field(default=1)
    refined: bool = Field(default=False)

    # Timing
    total_time_ms: float = Field(default=0.0)
    retrieval_time_ms: float = Field(default=0.0)
    reflection_time_ms: float = Field(default=0.0)


# =============================================================================
# SELF-RAG IMPLEMENTATION
# =============================================================================


class SelfRAG:
    """Self-Reflective RAG with critique and refinement.

    Implements the Self-RAG paradigm where the model:
    1. Decides if retrieval is needed
    2. Critiques retrieved documents for relevance
    3. Generates response with support assessment
    4. Optionally refines based on critique
    """

    def __init__(
        self,
        relevance_threshold: float = RELEVANCE_THRESHOLD,
        support_threshold: float = SUPPORT_THRESHOLD,
        max_iterations: int = MAX_CRITIQUE_ITERATIONS,
    ):
        self.relevance_threshold = relevance_threshold
        self.support_threshold = support_threshold
        self.max_iterations = max_iterations

        self._stats = {
            "total_queries": 0,
            "retrieval_skipped": 0,
            "documents_filtered": 0,
            "responses_refined": 0,
        }

        # Phase 2 FIX: Add CircuitBreaker for Qdrant operations
        self._circuit_breaker = CircuitBreakerRegistry.get_instance().get_or_create(
            circuit_id="self_rag_qdrant",
            failure_threshold=5,
            reset_timeout=30.0,
        )

        # Register with coordinator for unified health monitoring
        coordinator = get_stack_circuit_coordinator()
        coordinator.register_stack("self_rag_qdrant", self._circuit_breaker)

        logger.info(f"SelfRAG initialized (relevance_threshold={relevance_threshold})")

    async def query_with_reflection(
        self,
        query: str,
        critique_response: bool = True,
        refine_if_needed: bool = True,
        documents: Optional[List[Dict[str, Any]]] = None,
    ) -> SelfRAGResult:
        """Query with self-reflection at each step.

        Args:
            query: The query to process
            critique_response: Whether to critique the generated response
            refine_if_needed: Whether to refine response if critique is poor
            documents: Optional pre-retrieved documents

        Returns:
            SelfRAGResult with full reflection chain
        """
        import time
        start_time = time.time()

        self._stats["total_queries"] += 1

        # Step 1: Decide if retrieval is needed
        retrieval_decision = await self._assess_retrieval_need(query)

        retrieval_time = 0.0
        retrieved_docs: List[Dict[str, Any]] = []
        doc_relevances: List[DocumentRelevance] = []

        if retrieval_decision.decision != RetrievalToken.NO_RETRIEVE.value:
            # Step 2: Retrieve and assess relevance
            retrieve_start = time.time()

            if documents:
                retrieved_docs = documents
            else:
                # Phase 2 FIX: Handle RetrievalResult properly
                retrieval_result = await self._retrieve_documents(query)
                if retrieval_result.success:
                    retrieved_docs = retrieval_result.documents
                else:
                    # Log the error but continue with empty docs (fail-closed)
                    logger.warning(
                        "Self-RAG: Retrieval failed (%s), continuing with empty context",
                        retrieval_result.error_type,
                    )
                    retrieved_docs = []

            retrieval_time = (time.time() - retrieve_start) * 1000

            # Step 3: Assess relevance of each document
            doc_relevances = await self._assess_document_relevance(query, retrieved_docs)

            # Filter to relevant documents only
            relevant_docs = [
                doc for doc, rel in zip(retrieved_docs, doc_relevances)
                if rel.relevance_score >= self.relevance_threshold
            ]
            self._stats["documents_filtered"] += len(retrieved_docs) - len(relevant_docs)
            retrieved_docs = relevant_docs if relevant_docs else retrieved_docs[:3]
        else:
            self._stats["retrieval_skipped"] += 1

        # Step 4: Generate response
        answer = await self._generate_response(query, retrieved_docs)

        # Step 5: Assess support
        support = await self._assess_support(query, answer, retrieved_docs)

        # Step 6: Critique (if enabled)
        critique = None
        iterations = 1
        refined = False

        if critique_response:
            critique = await self._critique_response(query, answer, retrieved_docs)

            # OPT-10-010: Early exit if confidence is already high
            if critique.utility_score >= HIGH_CONFIDENCE_THRESHOLD:
                logger.debug(
                    f"OPT-10-010: Early exit - confidence {critique.utility_score:.2f} "
                    f">= threshold {HIGH_CONFIDENCE_THRESHOLD}"
                )
            # Refine if utility is low
            elif refine_if_needed and critique.utility_score < 0.5:
                for i in range(self.max_iterations - 1):
                    iterations += 1
                    answer = await self._refine_response(
                        query, answer, critique.improvements, retrieved_docs
                    )
                    critique = await self._critique_response(query, answer, retrieved_docs)

                    # OPT-10-010: Early exit when confidence reaches threshold
                    if critique.utility_score >= HIGH_CONFIDENCE_THRESHOLD:
                        refined = True
                        self._stats["responses_refined"] += 1
                        logger.debug(
                            f"OPT-10-010: Early exit after refinement - "
                            f"confidence {critique.utility_score:.2f}"
                        )
                        break

                    if critique.utility_score >= 0.7:
                        refined = True
                        self._stats["responses_refined"] += 1
                        break

        reflection_time = (time.time() - start_time) * 1000 - retrieval_time
        total_time = (time.time() - start_time) * 1000

        return SelfRAGResult(
            query=query,
            answer=answer,
            retrieval_needed=RetrievalToken(retrieval_decision.decision),
            retrieval_reasoning=retrieval_decision.reasoning,
            documents=retrieved_docs,
            document_relevances=doc_relevances,
            support=support,
            critique=critique,
            iterations=iterations,
            refined=refined,
            total_time_ms=total_time,
            retrieval_time_ms=retrieval_time,
            reflection_time_ms=reflection_time,
        )

    async def _assess_retrieval_need(self, query: str) -> ReflectionResult:
        """Assess whether retrieval is needed for this query."""
        # Simple heuristics (would use LLM in production)
        factual_indicators = ["what", "when", "where", "who", "how many", "evidence", "source"]
        opinion_indicators = ["think", "feel", "opinion", "believe"]

        query_lower = query.lower()

        needs_retrieval = any(ind in query_lower for ind in factual_indicators)
        is_opinion = any(ind in query_lower for ind in opinion_indicators)

        if is_opinion and not needs_retrieval:
            return ReflectionResult(
                aspect="retrieval_need",
                decision=RetrievalToken.NO_RETRIEVE.value,
                confidence=0.7,
                reasoning="Query appears to be opinion-based, retrieval not necessary",
            )
        elif needs_retrieval:
            return ReflectionResult(
                aspect="retrieval_need",
                decision=RetrievalToken.RETRIEVE.value,
                confidence=0.9,
                reasoning="Query requires factual information, retrieval recommended",
            )
        else:
            return ReflectionResult(
                aspect="retrieval_need",
                decision=RetrievalToken.UNCERTAIN.value,
                confidence=0.5,
                reasoning="Uncertain if retrieval needed, retrieving to be safe",
            )

    async def _retrieve_documents(self, query: str) -> RetrievalResult:
        """Retrieve documents for query.

        C-004 FIX: Uses real Qdrant retrieval instead of fake documents.
        Phase 2 FIX: Uses CircuitBreaker for resilience.

        Returns:
            RetrievalResult - NEVER returns fake documents.
            - empty() if no documents found (service worked)
            - service_error() if Qdrant failed (service broken)
        """
        # Check circuit breaker FIRST
        if not self._circuit_breaker.can_execute():
            retry_after = self._circuit_breaker.get_time_until_retry()
            logger.warning(
                "SELF_RAG_CIRCUIT_OPEN: Retrieval blocked - circuit open for %.1fs",
                retry_after,
            )
            return RetrievalResult(
                success=False,
                error=CircuitOpenError("self_rag_qdrant", retry_after),
                error_type="circuit_open",
            )

        start_time = time.time()

        client = _get_qdrant_client()
        if client is None:
            self._circuit_breaker.record_failure()
            logger.error(
                "C-004: Self-RAG retrieval FAILED - Qdrant not available. "
                "CircuitBreaker: failure recorded."
            )
            return RetrievalResult.service_error(
                error=ConnectionError("Qdrant client not available"),
                error_type="service_down",
            )

        try:
            # Use the qdrant_client wrapper to search
            results = client.search_similar(
                collection="self_rag_documents",
                query_text=query,
                top_k=5,
            )

            # Record success with circuit breaker
            self._circuit_breaker.record_success()

            search_time = (time.time() - start_time) * 1000

            # Convert to standard format
            if not results:
                return RetrievalResult.empty()

            documents = []
            for hit in results:
                documents.append({
                    "id": str(hit.id) if hasattr(hit, 'id') else str(hit.get("id", "unknown")),
                    "content": hit.payload.get("content", "") if hasattr(hit, 'payload') else hit.get("content", ""),
                    "score": float(hit.score) if hasattr(hit, 'score') else float(hit.get("score", 0.0)),
                    "metadata": hit.payload if hasattr(hit, 'payload') else hit.get("metadata", {}),
                })

            logger.debug(f"Self-RAG: Retrieved {len(documents)} documents for query")
            return RetrievalResult.with_documents(documents, search_time_ms=search_time)

        except CircuitOpenError:
            raise  # Don't record, already open
        except Exception as e:
            self._circuit_breaker.record_failure()
            logger.error(
                f"C-004: Self-RAG retrieval FAILED - Qdrant search error: {e}. "
                "CircuitBreaker: failure recorded."
            )
            return RetrievalResult.service_error(error=e)

    async def _retrieve_documents_compat(self, query: str) -> List[Dict[str, Any]]:
        """Backward-compatible wrapper for _retrieve_documents.

        Returns List[Dict] for existing callers that expect the old signature.
        """
        result = await self._retrieve_documents(query)
        if result.success:
            return result.documents
        return []  # Fail-closed: return empty on error

    async def _assess_document_relevance(
        self,
        query: str,
        documents: List[Dict[str, Any]],
    ) -> List[DocumentRelevance]:
        """Assess relevance of each document."""
        relevances = []

        for doc in documents:
            # Simple relevance check (would use embeddings in production)
            doc_content = doc.get("content", "").lower()
            query_words = set(query.lower().split())
            doc_words = set(doc_content.split())
            overlap = query_words & doc_words

            relevance_score = len(overlap) / max(len(query_words), 1)
            relevance_score = min(1.0, relevance_score + doc.get("score", 0) * 0.3)

            if relevance_score >= 0.7:
                token = RelevanceToken.RELEVANT
            elif relevance_score >= 0.4:
                token = RelevanceToken.PARTIALLY_RELEVANT
            else:
                token = RelevanceToken.IRRELEVANT

            relevances.append(DocumentRelevance(
                document_id=doc.get("id", "unknown"),
                content_preview=doc.get("content", "")[:100],
                relevance=token,
                relevance_score=relevance_score,
                key_overlap=list(overlap)[:5],
                reasoning=f"Found {len(overlap)} overlapping terms",
            ))

        return relevances

    async def _generate_response(
        self,
        query: str,
        documents: List[Dict[str, Any]],
    ) -> str:
        """Generate response based on query and documents.

        C-004 FIX: Attempts real LLM generation, returns empty on failure.
        NEVER returns fake/placeholder responses.
        """
        if not documents:
            logger.warning(
                "C-004: Self-RAG generation called with no documents. "
                "Returning empty response (fail-closed)."
            )
            return ""

        try:
            # Try to use Claude API via pipeline_autonomo
            from pipeline.claude_cli_llm import execute_agent_task

            # Build context from documents
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
                sprint_id="self_rag",
            )

            if result.get("status") == "success" and result.get("output"):
                return result["output"]

            logger.error(
                f"C-004: Self-RAG LLM generation failed: {result.get('error', 'unknown')}. "
                "Returning empty response (fail-closed)."
            )
            return ""

        except ImportError:
            logger.error(
                "C-004: Self-RAG LLM not available (claude_cli_llm not installed). "
                "Returning empty response (fail-closed). No fake responses will be generated."
            )
            return ""
        except Exception as e:
            logger.error(
                f"C-004: Self-RAG generation FAILED: {e}. "
                "Returning empty response (fail-closed). No fake responses will be generated."
            )
            return ""

    async def _assess_support(
        self,
        query: str,
        answer: str,
        documents: List[Dict[str, Any]],
    ) -> SupportAssessment:
        """Assess how well the answer is supported by documents."""
        if not documents:
            return SupportAssessment(
                support_token=SupportToken.NO_SUPPORT,
                support_score=0.0,
                reasoning="No documents to support answer",
            )

        # Simple support check (would use NLI model in production)
        answer_words = set(answer.lower().split())
        supporting = []

        for doc in documents:
            doc_words = set(doc.get("content", "").lower().split())
            overlap = answer_words & doc_words
            if len(overlap) > 3:
                supporting.append(doc.get("id", "unknown"))

        support_ratio = len(supporting) / len(documents)

        if support_ratio >= 0.7:
            token = SupportToken.FULLY_SUPPORTED
        elif support_ratio >= 0.3:
            token = SupportToken.PARTIALLY_SUPPORTED
        else:
            token = SupportToken.NO_SUPPORT

        return SupportAssessment(
            support_token=token,
            support_score=support_ratio,
            supporting_docs=supporting,
            reasoning=f"{len(supporting)}/{len(documents)} documents support the answer",
        )

    async def _critique_response(
        self,
        query: str,
        answer: str,
        documents: List[Dict[str, Any]],
    ) -> CritiqueResult:
        """Critique the generated response."""
        # Simple critique (would use LLM in production)
        improvements = []

        # Check length
        if len(answer) < 50:
            improvements.append("Response could be more detailed")

        # Check if it addresses the query
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        if len(query_words & answer_words) < 2:
            improvements.append("Response may not fully address the query")

        # Calculate scores
        utility_score = min(1.0, len(answer) / 200)
        factual = 0.7 if documents else 0.3
        completeness = 0.8 if not improvements else 0.5
        clarity = 0.7

        utility_int = max(1, min(5, int(utility_score * 5)))

        return CritiqueResult(
            utility=UtilityToken(str(utility_int)),
            utility_score=utility_score,
            factual_accuracy=factual,
            completeness=completeness,
            clarity=clarity,
            improvements=improvements,
        )

    async def _refine_response(
        self,
        query: str,
        answer: str,
        improvements: List[str],
        documents: List[Dict[str, Any]],
    ) -> str:
        """Refine response based on critique."""
        # Placeholder - would use LLM to refine
        improvements_str = ", ".join(improvements)
        return f"{answer} [Refined to address: {improvements_str}]"

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        return {
            **self._stats,
            "relevance_threshold": self.relevance_threshold,
            "support_threshold": self.support_threshold,
        }


# =============================================================================
# SINGLETON AND CONVENIENCE FUNCTIONS
# =============================================================================

_self_rag_instance: Optional[SelfRAG] = None


def get_self_rag() -> SelfRAG:
    """Get or create SelfRAG singleton."""
    global _self_rag_instance
    if _self_rag_instance is None:
        _self_rag_instance = SelfRAG()
    return _self_rag_instance


async def query_with_reflection(
    query: str,
    critique_response: bool = True,
) -> SelfRAGResult:
    """Convenience function for self-reflective query."""
    rag = get_self_rag()
    return await rag.query_with_reflection(
        query=query,
        critique_response=critique_response,
    )


__all__ = [
    "SELF_RAG_AVAILABLE",
    "SelfRAG",
    "SelfRAGResult",
    "RetrievalToken",
    "RelevanceToken",
    "SupportToken",
    "UtilityToken",
    "DocumentRelevance",
    "SupportAssessment",
    "CritiqueResult",
    "get_self_rag",
    "query_with_reflection",
]
