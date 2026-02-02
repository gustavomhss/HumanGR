"""Active RAG - Proactive Retrieval Augmented Generation.

Active RAG goes beyond traditional RAG by:
1. Predicting what queries/information might be needed next
2. Proactively prefetching relevant context
3. Caching predictions for faster response times
4. Learning from query patterns to improve predictions

This is especially useful for claim verification where:
- Related claims often need to be checked
- Sources referenced in one claim appear in others
- Topics cluster together in verification workflows

Usage:
    from pipeline.rag.active_rag import ActiveRAG

    # Initialize
    rag = ActiveRAG()

    # Process current query and predict next
    result = await rag.query_with_prefetch(
        query="Is climate change real?",
        prefetch_depth=2,
    )

    # result.answer: Answer to current query
    # result.prefetched: Pre-loaded context for likely follow-ups
"""

from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import asyncio

from pydantic import BaseModel, Field

# Stack enforcement decorator for critical operations (RAG requires qdrant)
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

# Phase 2 FIX: CircuitBreaker for resilience
try:
    from pipeline.retry_config import (
        CircuitBreaker,
        CircuitBreakerRegistry,
        CircuitOpenError,
        CircuitState,
    )
    from pipeline.resilience.coordinator import get_stack_circuit_coordinator
    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    CIRCUIT_BREAKER_AVAILABLE = False
    CircuitBreaker = None
    CircuitBreakerRegistry = None
    CircuitOpenError = Exception
    CircuitState = None

# Qdrant client helper (reuse existing pattern)
try:
    from pipeline.rag.qdrant_hybrid import QdrantClientWrapper, _get_qdrant_client
    QDRANT_CLIENT_AVAILABLE = True
except ImportError:
    QDRANT_CLIENT_AVAILABLE = False
    _get_qdrant_client = lambda: None


# =============================================================================
# CONFIGURATION
# =============================================================================

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "humangr_claims")
PREDICTION_CACHE_TTL = int(os.getenv("PREDICTION_CACHE_TTL", "300"))  # 5 minutes
MAX_PREFETCH_QUERIES = int(os.getenv("MAX_PREFETCH_QUERIES", "5"))

# Active RAG is always available (native implementation)
ACTIVE_RAG_AVAILABLE = True


# =============================================================================
# LLM PROMPTS FOR QUERY PREDICTION
# =============================================================================

QUERY_PREDICTION_PROMPT = """You are an expert at predicting what information a user will need next based on their current query and context.

CURRENT QUERY: {current_query}

CONTEXT:
{context}

RECENT QUERIES (if any):
{recent_queries}

RETRIEVED DOCUMENTS (summaries):
{documents}

Based on the above, predict 3-5 likely next queries the user will need, ranked by probability.
Consider:
1. Natural follow-up questions
2. Related topics they might explore
3. Clarifying questions they might ask
4. Deeper dives into specific aspects

Output as JSON:
{{
  "predicted_queries": [
    {{
      "query": "What is the primary evidence for X?",
      "probability": 0.8,
      "rationale": "User asked about X, likely wants supporting evidence",
      "category": "evidence_seeking"
    }},
    {{
      "query": "Are there any contradicting views on X?",
      "probability": 0.6,
      "rationale": "Balanced research often seeks opposing viewpoints",
      "category": "counter_evidence"
    }}
  ]
}}"""

QUERY_REFINEMENT_PROMPT = """You are an expert at refining search queries to get better results.

ORIGINAL QUERY: {original_query}

RETRIEVED DOCUMENTS (may be insufficient):
{documents}

The current results may not fully answer the query. Generate refined sub-queries that would help retrieve more relevant information.

Consider:
1. Breaking down complex queries into simpler parts
2. Using different terminology or synonyms
3. Focusing on specific aspects not well covered
4. Adding context that might help retrieval

Output as JSON:
{{
  "refined_queries": [
    {{
      "query": "refined query 1",
      "focus": "what aspect this targets",
      "expected_improvement": "what gap this fills"
    }}
  ],
  "missing_information": ["list of information gaps identified"],
  "suggested_sources": ["types of sources that might help"]
}}"""

RELEVANCE_ASSESSMENT_PROMPT = """You are an expert at assessing document relevance to a query.

QUERY: {query}

DOCUMENT:
{document}

Assess the relevance of this document to the query on a scale of 0.0 to 1.0.

Consider:
1. Direct relevance: Does it directly answer the query?
2. Supporting information: Does it provide useful context?
3. Credibility: Is the information from a reliable source?
4. Recency: Is the information up to date?
5. Specificity: Does it address the specific aspects asked about?

Output as JSON:
{{
  "relevance_score": 0.85,
  "reasoning": "The document directly addresses...",
  "strengths": ["list of why it's relevant"],
  "weaknesses": ["list of limitations"],
  "suggested_follow_ups": ["queries to fill gaps"]
}}"""


# =============================================================================
# DATA MODELS
# =============================================================================


class QueryPrediction(BaseModel):
    """A predicted query that might come next."""

    query: str = Field(..., description="Predicted query text")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    reasoning: str = Field(default="", description="Why this query was predicted")
    category: str = Field(default="related", description="Category of prediction")
    priority: int = Field(default=1, ge=1, le=10, description="Prefetch priority")


class RAGContext(BaseModel):
    """Context retrieved for a query."""

    query: str = Field(..., description="The query this context is for")
    documents: List[Dict[str, Any]] = Field(default_factory=list)
    relevance_scores: List[float] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)
    fetch_time_ms: float = Field(default=0.0)
    cached: bool = Field(default=False)


class PrefetchResult(BaseModel):
    """Result from prefetching context."""

    predicted_queries: List[QueryPrediction] = Field(default_factory=list)
    prefetched_contexts: Dict[str, RAGContext] = Field(default_factory=dict)
    total_documents: int = Field(default=0)
    prefetch_time_ms: float = Field(default=0.0)


class QueryWithPrefetchResult(BaseModel):
    """Result from query with prefetching."""

    query: str
    answer: str
    context: RAGContext
    prefetched: PrefetchResult
    total_time_ms: float


class RetrievalFeedback(BaseModel):
    """Feedback for a retrieval result."""

    query: str = Field(..., description="Original query")
    document_id: str = Field(..., description="Document ID")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    was_helpful: bool = Field(default=True, description="Whether document was helpful")
    feedback_type: str = Field(default="implicit", description="implicit or explicit")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class IterativeRetrievalResult(BaseModel):
    """Result from iterative retrieval."""

    query: str = Field(..., description="Original query")
    iterations: int = Field(..., description="Number of iterations performed")
    final_context: RAGContext = Field(..., description="Final aggregated context")
    iteration_history: List[Dict[str, Any]] = Field(default_factory=list)
    refinement_queries: List[str] = Field(default_factory=list)
    total_documents: int = Field(..., description="Total unique documents retrieved")
    convergence_score: float = Field(..., ge=0.0, le=1.0, description="Convergence score")


@dataclass
class QueryPattern:
    """A pattern of query sequences."""

    sequence: Tuple[str, ...]
    count: int = 1
    last_seen: datetime = field(default_factory=datetime.utcnow)

    @property
    def weight(self) -> float:
        """Calculate weight based on recency and frequency."""
        age_hours = (datetime.now(timezone.utc) - self.last_seen).total_seconds() / 3600
        recency_factor = 1.0 / (1.0 + age_hours / 24)  # Decay over days
        return self.count * recency_factor


# =============================================================================
# ACTIVE RAG
# =============================================================================


class ActiveRAG:
    """Active RAG with query prediction and prefetching.

    Goes beyond traditional RAG by proactively predicting
    and prefetching context for likely follow-up queries.
    """

    def __init__(
        self,
        qdrant_url: str = QDRANT_URL,
        collection: str = QDRANT_COLLECTION,
        cache_ttl: int = PREDICTION_CACHE_TTL,
        max_prefetch: int = MAX_PREFETCH_QUERIES,
    ):
        """Initialize Active RAG.

        Args:
            qdrant_url: Qdrant server URL
            collection: Collection name
            cache_ttl: Cache TTL in seconds
            max_prefetch: Maximum queries to prefetch
        """
        self.qdrant_url = qdrant_url
        self.collection = collection
        self.cache_ttl = cache_ttl
        self.max_prefetch = max_prefetch

        # Query history for pattern learning
        self._query_history: List[Tuple[str, datetime]] = []
        self._query_patterns: Dict[Tuple[str, ...], QueryPattern] = {}

        # Prefetch cache
        self._context_cache: Dict[str, Tuple[RAGContext, datetime]] = {}

        # Topic clusters for prediction
        self._topic_clusters: Dict[str, List[str]] = defaultdict(list)

        # Phase 2 FIX: Initialize CircuitBreaker for Qdrant operations
        if CIRCUIT_BREAKER_AVAILABLE:
            self._circuit_breaker = CircuitBreakerRegistry.get_instance().get_or_create(
                circuit_id="active_rag_qdrant",
                failure_threshold=5,
                reset_timeout=30.0,
            )
            coordinator = get_stack_circuit_coordinator()
            coordinator.register_stack("active_rag_qdrant", self._circuit_breaker)
        else:
            self._circuit_breaker = None

        logger.info("Active RAG initialized")

    def _normalize_query(self, query: str) -> str:
        """Normalize query for pattern matching."""
        return query.lower().strip()

    def _update_patterns(self, query: str) -> None:
        """Update query patterns from history."""
        normalized = self._normalize_query(query)
        now = datetime.now(timezone.utc)

        # Add to history
        self._query_history.append((normalized, now))

        # Keep only recent history
        cutoff = now - timedelta(hours=24)
        self._query_history = [
            (q, t) for q, t in self._query_history if t > cutoff
        ]

        # Update patterns (bigrams and trigrams)
        for window_size in [2, 3]:
            if len(self._query_history) >= window_size:
                recent = tuple(q for q, _ in self._query_history[-window_size:])
                if recent in self._query_patterns:
                    self._query_patterns[recent].count += 1
                    self._query_patterns[recent].last_seen = now
                else:
                    self._query_patterns[recent] = QueryPattern(
                        sequence=recent,
                        count=1,
                        last_seen=now,
                    )

    async def predict_next_queries(
        self,
        current_query: str,
        top_k: int = 5,
    ) -> List[QueryPrediction]:
        """Predict what queries might come next.

        Uses multiple strategies:
        1. Historical patterns (what queries followed this before)
        2. Topic clustering (related topics)
        3. Semantic similarity (similar queries in history)

        Args:
            current_query: The current query
            top_k: Number of predictions to return

        Returns:
            List of predicted queries with confidence
        """
        predictions = []
        normalized = self._normalize_query(current_query)

        # Strategy 1: Historical patterns
        pattern_predictions = self._predict_from_patterns(normalized)
        predictions.extend(pattern_predictions)

        # Strategy 2: Topic-based predictions
        topic_predictions = self._predict_from_topics(normalized)
        predictions.extend(topic_predictions)

        # Strategy 3: Template-based predictions (for claim verification)
        template_predictions = self._predict_from_templates(normalized)
        predictions.extend(template_predictions)

        # Deduplicate and sort by confidence
        seen = set()
        unique_predictions = []
        for p in sorted(predictions, key=lambda x: x.confidence, reverse=True):
            if p.query not in seen:
                seen.add(p.query)
                unique_predictions.append(p)

        return unique_predictions[:top_k]

    def _predict_from_patterns(self, query: str) -> List[QueryPrediction]:
        """Predict from historical query patterns."""
        predictions = []

        for pattern, data in self._query_patterns.items():
            if query in pattern[:-1]:  # Query is in the pattern but not last
                idx = pattern.index(query)
                if idx < len(pattern) - 1:
                    next_query = pattern[idx + 1]
                    confidence = min(data.weight / 10, 0.9)  # Cap at 0.9
                    predictions.append(QueryPrediction(
                        query=next_query,
                        confidence=confidence,
                        reasoning="Historical pattern",
                        category="historical",
                        priority=1,
                    ))

        return predictions

    def _predict_from_topics(self, query: str) -> List[QueryPrediction]:
        """Predict from topic clusters."""
        predictions = []

        # Extract potential topic
        words = set(query.split())

        for topic, related_queries in self._topic_clusters.items():
            if topic in words:
                for related in related_queries[:3]:
                    if related != query:
                        predictions.append(QueryPrediction(
                            query=related,
                            confidence=0.6,
                            reasoning=f"Related to topic: {topic}",
                            category="topic",
                            priority=2,
                        ))

        return predictions

    def _predict_from_templates(self, query: str) -> List[QueryPrediction]:
        """Predict using claim verification templates."""
        predictions = []

        # Common follow-up patterns for claim verification
        templates = [
            ("is {} true", "evidence for {}", 0.7),
            ("is {} true", "who said {}", 0.6),
            ("source of {}", "credibility of {}", 0.65),
            ("claim about {}", "fact check {}", 0.7),
        ]

        for pattern, follow_up, conf in templates:
            # Extract topic from query
            words = query.split()
            if len(words) >= 2:
                topic = " ".join(words[-2:])  # Last two words as topic

                if any(w in query.lower() for w in ["is", "true", "false", "claim"]):
                    predictions.append(QueryPrediction(
                        query=follow_up.format(topic),
                        confidence=conf,
                        reasoning="Claim verification pattern",
                        category="template",
                        priority=2,
                    ))

        return predictions

    async def _predict_with_llm(
        self,
        current_query: str,
        context: Dict[str, Any],
        llm_client: Optional[Any] = None,
    ) -> List[QueryPrediction]:
        """Use LLM to predict likely next queries.

        This method provides more sophisticated predictions by using an LLM
        to analyze the current context and predict what information the user
        will likely need next.

        Args:
            current_query: The current query
            context: Dictionary containing:
                - current_context: Current conversation/session context
                - recent_queries: List of recent queries
                - documents: Retrieved documents (summaries)
            llm_client: Optional LLM client for generation

        Returns:
            List of predicted queries with confidence and rationale
        """
        if llm_client is None:
            # Fallback to heuristics if no LLM available
            logger.debug("No LLM client provided, falling back to heuristics")
            return []

        try:
            import json

            # Format the prompt
            prompt = QUERY_PREDICTION_PROMPT.format(
                current_query=current_query,
                context=context.get("current_context", "No additional context."),
                recent_queries="\n".join(context.get("recent_queries", [])) or "None",
                documents="\n".join([
                    f"- {doc.get('title', 'Untitled')}: {doc.get('summary', doc.get('content', '')[:200])}"
                    for doc in context.get("documents", [])[:5]
                ]) or "None retrieved yet.",
            )

            # Call LLM
            response = await llm_client.generate(prompt)

            # Parse response
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                result = json.loads(json_match.group())
                predictions = []

                for pred in result.get("predicted_queries", []):
                    predictions.append(QueryPrediction(
                        query=pred.get("query", ""),
                        confidence=float(pred.get("probability", 0.5)),
                        reasoning=pred.get("rationale", "LLM prediction"),
                        category=pred.get("category", "llm_predicted"),
                        priority=1,  # LLM predictions are high priority
                    ))

                logger.info(f"LLM predicted {len(predictions)} queries for: {current_query[:50]}...")
                return predictions

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM prediction response: {e}")
        except Exception as e:
            logger.warning(f"LLM prediction failed: {e}")

        return []

    async def predict_next_queries_with_llm(
        self,
        current_query: str,
        context: Optional[Dict[str, Any]] = None,
        llm_client: Optional[Any] = None,
        top_k: int = 5,
    ) -> List[QueryPrediction]:
        """Predict next queries using both heuristics and LLM.

        Combines heuristic-based predictions with LLM-based predictions
        for more accurate and diverse query suggestions.

        Args:
            current_query: The current query
            context: Optional context for LLM prediction
            llm_client: Optional LLM client
            top_k: Number of predictions to return

        Returns:
            List of predicted queries, combining heuristics and LLM
        """
        predictions = []

        # Get heuristic predictions
        heuristic_predictions = await self.predict_next_queries(current_query, top_k)
        predictions.extend(heuristic_predictions)

        # Get LLM predictions if client available
        if llm_client and context:
            llm_predictions = await self._predict_with_llm(
                current_query,
                context,
                llm_client,
            )
            predictions.extend(llm_predictions)

        # Deduplicate and sort by confidence
        seen = set()
        unique_predictions = []
        for p in sorted(predictions, key=lambda x: x.confidence, reverse=True):
            normalized = self._normalize_query(p.query)
            if normalized not in seen:
                seen.add(normalized)
                unique_predictions.append(p)

        return unique_predictions[:top_k]

    async def prefetch_context(
        self,
        predictions: List[QueryPrediction],
    ) -> PrefetchResult:
        """Prefetch context for predicted queries.

        Args:
            predictions: List of predicted queries

        Returns:
            PrefetchResult with prefetched contexts
        """
        start_time = datetime.now(timezone.utc)
        prefetched = {}
        total_docs = 0

        # Sort by priority and take top N
        sorted_predictions = sorted(
            predictions,
            key=lambda p: (p.priority, -p.confidence),
        )[:self.max_prefetch]

        # Prefetch in parallel
        tasks = []
        for pred in sorted_predictions:
            tasks.append(self._fetch_context(pred.query))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for pred, result in zip(sorted_predictions, results):
            if isinstance(result, RAGContext):
                prefetched[pred.query] = result
                total_docs += len(result.documents)

        prefetch_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        return PrefetchResult(
            predicted_queries=sorted_predictions,
            prefetched_contexts=prefetched,
            total_documents=total_docs,
            prefetch_time_ms=prefetch_time,
        )

    async def _fetch_context(self, query: str) -> RAGContext:
        """Fetch context for a query (with caching).

        Phase 2 FIX: Uses real Qdrant retrieval with CircuitBreaker protection.
        NEVER returns fake documents - returns empty context on failure.
        """
        normalized = self._normalize_query(query)
        now = datetime.now(timezone.utc)

        # Check cache
        if normalized in self._context_cache:
            context, cached_at = self._context_cache[normalized]
            if (now - cached_at).total_seconds() < self.cache_ttl:
                context.cached = True
                return context

        start = datetime.now(timezone.utc)

        # Phase 2 FIX: Check circuit breaker first
        if self._circuit_breaker and self._circuit_breaker.state == CircuitState.OPEN:
            logger.warning(
                "ACTIVE_RAG-001: Circuit breaker OPEN - returning empty context."
            )
            return RAGContext(
                query=query,
                documents=[],
                relevance_scores=[],
                sources=[],
                fetch_time_ms=0.0,
                cached=False,
            )

        # Phase 2 FIX: Fetch from real Qdrant
        documents = []
        relevance_scores = []
        sources = []

        if QDRANT_CLIENT_AVAILABLE:
            client = _get_qdrant_client()
            if client is not None:
                try:
                    results = client.search_similar(
                        collection=self.collection,
                        query_text=query,
                        top_k=5,
                    )

                    if self._circuit_breaker:
                        self._circuit_breaker.record_success()

                    for hit in results or []:
                        if hasattr(hit, 'payload'):
                            documents.append(hit.payload.get("content", ""))
                            relevance_scores.append(hit.score if hasattr(hit, 'score') else 0.0)
                            sources.append(hit.payload.get("source", "unknown"))
                        elif isinstance(hit, dict):
                            documents.append(hit.get("content", ""))
                            relevance_scores.append(hit.get("score", 0.0))
                            sources.append(hit.get("source", "unknown"))

                except Exception as e:
                    if self._circuit_breaker:
                        self._circuit_breaker.record_failure()
                    logger.warning(
                        f"ACTIVE_RAG-002: Qdrant search failed: {e}. Returning empty context."
                    )
            else:
                if self._circuit_breaker:
                    self._circuit_breaker.record_failure()
                logger.warning(
                    "ACTIVE_RAG-003: Qdrant client not available. Returning empty context."
                )
        else:
            logger.debug("ACTIVE_RAG: Qdrant client wrapper not available.")

        fetch_time = (datetime.now(timezone.utc) - start).total_seconds() * 1000

        context = RAGContext(
            query=query,
            documents=documents,
            relevance_scores=relevance_scores,
            sources=sources,
            fetch_time_ms=fetch_time,
            cached=False,
        )

        # Cache result
        self._context_cache[normalized] = (context, now)

        return context

    @enforce_stacks("search", required=["qdrant"], recommended=["langfuse"])
    async def query_with_prefetch(
        self,
        query: str,
        prefetch_depth: int = 1,
    ) -> QueryWithPrefetchResult:
        """Query with automatic prefetching of likely follow-ups.

        Args:
            query: The query to answer
            prefetch_depth: How many levels of predictions to prefetch

        Returns:
            Result with answer and prefetched context
        """
        start_time = datetime.now(timezone.utc)

        # Update patterns
        self._update_patterns(query)

        # Get context for current query
        context = await self._fetch_context(query)

        # Phase 2 FIX: Generate answer from context (not fake placeholder)
        # If documents found, summarize them; otherwise indicate no context found
        if context.documents:
            # Return a summary of found documents (real LLM would generate better answer)
            doc_summary = "; ".join(doc[:100] for doc in context.documents[:3] if doc)
            answer = f"Based on {len(context.documents)} documents: {doc_summary}..."
        else:
            # No documents found - be explicit about it
            answer = f"No relevant documents found for query: {query}"
            logger.debug(f"ACTIVE_RAG: No documents retrieved for query '{query[:50]}...'")

        # Predict and prefetch
        predictions = await self.predict_next_queries(query)
        prefetch_result = await self.prefetch_context(predictions)

        total_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        return QueryWithPrefetchResult(
            query=query,
            answer=answer,
            context=context,
            prefetched=prefetch_result,
            total_time_ms=total_time,
        )

    def add_topic_cluster(self, topic: str, related_queries: List[str]) -> None:
        """Add a topic cluster for prediction.

        Args:
            topic: The topic keyword
            related_queries: Queries related to this topic
        """
        self._topic_clusters[topic].extend(related_queries)

    def clear_cache(self) -> None:
        """Clear the context cache."""
        self._context_cache.clear()
        logger.debug("Context cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about Active RAG."""
        return {
            "history_size": len(self._query_history),
            "pattern_count": len(self._query_patterns),
            "cache_size": len(self._context_cache),
            "topic_count": len(self._topic_clusters),
            "feedback_count": len(getattr(self, '_feedback_history', [])),
        }

    # =========================================================================
    # ITERATIVE RETRIEVAL WITH FEEDBACK LOOPS
    # =========================================================================

    def record_feedback(
        self,
        query: str,
        document_id: str,
        relevance_score: float,
        was_helpful: bool = True,
        feedback_type: str = "implicit",
    ) -> RetrievalFeedback:
        """Record feedback for a retrieval result.

        This feedback is used to improve future retrievals.

        Args:
            query: The original query
            document_id: ID of the document
            relevance_score: Relevance score (0-1)
            was_helpful: Whether the document was helpful
            feedback_type: "implicit" (from clicks) or "explicit" (user rating)

        Returns:
            The recorded feedback
        """
        if not hasattr(self, '_feedback_history'):
            self._feedback_history: List[RetrievalFeedback] = []
            self._document_scores: Dict[str, List[float]] = defaultdict(list)

        feedback = RetrievalFeedback(
            query=query,
            document_id=document_id,
            relevance_score=relevance_score,
            was_helpful=was_helpful,
            feedback_type=feedback_type,
        )

        self._feedback_history.append(feedback)
        self._document_scores[document_id].append(relevance_score)

        logger.debug(f"Recorded feedback for doc {document_id}: {relevance_score}")
        return feedback

    def get_document_score_adjustment(self, document_id: str) -> float:
        """Get score adjustment for a document based on feedback history.

        Args:
            document_id: The document ID

        Returns:
            Score adjustment factor (1.0 = neutral, >1.0 = boost, <1.0 = reduce)
        """
        if not hasattr(self, '_document_scores'):
            return 1.0

        scores = self._document_scores.get(document_id, [])
        if not scores:
            return 1.0

        # Calculate average historical relevance
        avg_score = sum(scores) / len(scores)

        # Convert to adjustment factor (0.5 - 1.5 range)
        return 0.5 + avg_score

    @enforce_stacks("search", required=["qdrant"], recommended=["langfuse"])
    async def iterative_retrieve(
        self,
        query: str,
        max_iterations: int = 3,
        convergence_threshold: float = 0.8,
        refinement_model: Optional[callable] = None,
    ) -> IterativeRetrievalResult:
        """Perform iterative retrieval with query refinement.

        Each iteration:
        1. Retrieves documents for current query
        2. Analyzes gaps in retrieved information
        3. Generates refined sub-queries
        4. Aggregates all retrieved documents

        Continues until convergence or max iterations.

        Args:
            query: Initial query
            max_iterations: Maximum number of iterations
            convergence_threshold: Score at which to stop iterating
            refinement_model: Optional model for query refinement

        Returns:
            IterativeRetrievalResult with aggregated context
        """
        all_documents: Dict[str, Dict[str, Any]] = {}
        all_scores: Dict[str, float] = {}
        iteration_history = []
        refinement_queries = [query]
        current_queries = [query]
        convergence_score = 0.0

        for iteration in range(max_iterations):
            iteration_docs = []
            iteration_start = datetime.now(timezone.utc)

            # Retrieve for all current queries
            for q in current_queries:
                context = await self._fetch_context(q)

                for i, doc in enumerate(context.documents):
                    doc_id = doc.get("id", str(hash(str(doc)))[:16])

                    # Apply feedback-based score adjustment
                    base_score = context.relevance_scores[i] if i < len(context.relevance_scores) else 0.5
                    adjusted_score = base_score * self.get_document_score_adjustment(doc_id)

                    if doc_id not in all_documents:
                        all_documents[doc_id] = doc
                        all_scores[doc_id] = adjusted_score
                        iteration_docs.append(doc)
                    else:
                        # Keep highest score
                        all_scores[doc_id] = max(all_scores[doc_id], adjusted_score)

            # Calculate convergence based on new documents found
            new_doc_ratio = len(iteration_docs) / max(len(all_documents), 1)
            convergence_score = 1.0 - new_doc_ratio

            # Record iteration
            iteration_history.append({
                "iteration": iteration + 1,
                "queries": current_queries,
                "new_documents": len(iteration_docs),
                "total_documents": len(all_documents),
                "convergence_score": convergence_score,
                "duration_ms": (datetime.now(timezone.utc) - iteration_start).total_seconds() * 1000,
            })

            logger.debug(
                f"Iteration {iteration + 1}: {len(iteration_docs)} new docs, "
                f"convergence={convergence_score:.2f}"
            )

            # Check for convergence
            if convergence_score >= convergence_threshold:
                logger.info(f"Converged after {iteration + 1} iterations")
                break

            # Generate refined queries for next iteration
            if refinement_model:
                # Use provided model for refinement
                current_queries = await refinement_model(query, all_documents, iteration)
            else:
                # Simple refinement: extract key terms from retrieved docs
                current_queries = self._generate_refinement_queries(
                    query, list(all_documents.values())
                )

            refinement_queries.extend(current_queries)

            if not current_queries:
                break

        # Build final context
        sorted_docs = sorted(
            [(doc_id, doc, all_scores[doc_id]) for doc_id, doc in all_documents.items()],
            key=lambda x: x[2],
            reverse=True,
        )

        final_context = RAGContext(
            query=query,
            documents=[doc for _, doc, _ in sorted_docs],
            relevance_scores=[score for _, _, score in sorted_docs],
            sources=list(set(
                doc.get("source", "unknown") for doc in all_documents.values()
            )),
            cached=False,
        )

        return IterativeRetrievalResult(
            query=query,
            iterations=len(iteration_history),
            final_context=final_context,
            iteration_history=iteration_history,
            refinement_queries=refinement_queries,
            total_documents=len(all_documents),
            convergence_score=convergence_score,
        )

    def _generate_refinement_queries(
        self,
        original_query: str,
        documents: List[Dict[str, Any]],
        max_queries: int = 3,
    ) -> List[str]:
        """Generate refinement queries based on retrieved documents.

        Simple heuristic-based refinement. In production, would use LLM.

        Args:
            original_query: The original query
            documents: Retrieved documents
            max_queries: Maximum refinement queries to generate

        Returns:
            List of refinement queries
        """
        if not documents:
            return []

        # Extract key terms from documents
        all_terms = []
        original_terms = set(self._normalize_query(original_query).split())

        for doc in documents[:5]:  # Only look at top docs
            text = doc.get("text", "") or doc.get("content", "")
            if text:
                words = self._normalize_query(text).split()
                # Find terms that appear multiple times but aren't in original query
                term_counts = defaultdict(int)
                for word in words:
                    if len(word) > 3 and word not in original_terms:
                        term_counts[word] += 1

                # Add frequent terms
                for term, count in term_counts.items():
                    if count >= 2:
                        all_terms.append((term, count))

        # Sort by frequency and generate queries
        all_terms.sort(key=lambda x: x[1], reverse=True)
        refinement_queries = []

        for term, _ in all_terms[:max_queries]:
            refinement_queries.append(f"{original_query} {term}")

        return refinement_queries

    async def retrieve_with_feedback_loop(
        self,
        query: str,
        initial_context: Optional[RAGContext] = None,
        feedback_callback: Optional[callable] = None,
        max_refinements: int = 2,
    ) -> RAGContext:
        """Retrieve with active feedback loop.

        Allows for human-in-the-loop or automatic feedback
        to improve retrieval in real-time.

        Args:
            query: The query
            initial_context: Optional pre-fetched context
            feedback_callback: Async callback for feedback (doc -> score)
            max_refinements: Maximum refinement rounds

        Returns:
            Refined RAGContext
        """
        context = initial_context or await self._fetch_context(query)

        for round_num in range(max_refinements):
            if not context.documents:
                break

            # Get feedback for each document
            feedback_scores = []
            for doc in context.documents:
                if feedback_callback:
                    score = await feedback_callback(doc)
                    feedback_scores.append(score)

                    # Record feedback
                    doc_id = doc.get("id", str(hash(str(doc)))[:16])
                    self.record_feedback(
                        query=query,
                        document_id=doc_id,
                        relevance_score=score,
                        was_helpful=score > 0.5,
                        feedback_type="explicit",
                    )
                else:
                    # Use existing relevance scores as feedback
                    idx = context.documents.index(doc)
                    if idx < len(context.relevance_scores):
                        feedback_scores.append(context.relevance_scores[idx])
                    else:
                        feedback_scores.append(0.5)

            # Re-rank based on feedback
            doc_scores = list(zip(context.documents, feedback_scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)

            # Update context with re-ranked documents
            context = RAGContext(
                query=context.query,
                documents=[d for d, _ in doc_scores],
                relevance_scores=[s for _, s in doc_scores],
                sources=context.sources,
                fetch_time_ms=context.fetch_time_ms,
                cached=False,
            )

            # Check if top results are good enough
            if feedback_scores and feedback_scores[0] >= 0.8:
                logger.debug(f"Feedback loop converged after {round_num + 1} rounds")
                break

        return context


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_active_rag: Optional[ActiveRAG] = None


def get_active_rag() -> ActiveRAG:
    """Get singleton Active RAG instance."""
    global _active_rag
    if _active_rag is None:
        _active_rag = ActiveRAG()
    return _active_rag


async def predict_next_queries(
    current_query: str,
    top_k: int = 5,
) -> Optional[List[QueryPrediction]]:
    """Predict next queries.

    Falls back gracefully if Active RAG is unavailable.
    """
    try:
        rag = get_active_rag()
        return await rag.predict_next_queries(current_query, top_k)
    except Exception as e:
        logger.error(f"Query prediction failed: {e}")
        return None


async def prefetch_context(
    predictions: List[QueryPrediction],
) -> Optional[PrefetchResult]:
    """Prefetch context for predictions.

    Falls back gracefully if Active RAG is unavailable.
    """
    try:
        rag = get_active_rag()
        return await rag.prefetch_context(predictions)
    except Exception as e:
        logger.error(f"Context prefetch failed: {e}")
        return None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ACTIVE_RAG_AVAILABLE",
    "ActiveRAG",
    "QueryPrediction",
    "PrefetchResult",
    "RAGContext",
    "QueryWithPrefetchResult",
    "QueryPattern",
    "RetrievalFeedback",
    "IterativeRetrievalResult",
    "get_active_rag",
    "predict_next_queries",
    "prefetch_context",
    # LLM Prompts
    "QUERY_PREDICTION_PROMPT",
    "QUERY_REFINEMENT_PROMPT",
    "RELEVANCE_ASSESSMENT_PROMPT",
]
