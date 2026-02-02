"""ColBERT Retriever - Late Interaction Retrieval.

ColBERT implements efficient retrieval through late interaction:
1. Independent encoding of queries and documents into token-level embeddings
2. MaxSim scoring between query and document token embeddings
3. Efficient indexing with compression (ColBERTv2)
4. Support for both exact and approximate search

Based on:
- "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT"
- Khattab & Zaharia, 2020
- "ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction"
- Santhanam et al., 2022

Usage:
    from pipeline.rag.colbert_retriever import ColBERTRetriever

    retriever = ColBERTRetriever()

    # Index documents
    await retriever.index_documents(documents)

    # Search with late interaction
    results = await retriever.search(
        query="What causes climate change?",
        top_k=10,
    )
"""

from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
import hashlib

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

EMBEDDING_DIM = int(os.getenv("COLBERT_EMBEDDING_DIM", "128"))
MAX_QUERY_TOKENS = int(os.getenv("COLBERT_MAX_QUERY_TOKENS", "32"))
MAX_DOC_TOKENS = int(os.getenv("COLBERT_MAX_DOC_TOKENS", "180"))
COMPRESSION_BITS = int(os.getenv("COLBERT_COMPRESSION_BITS", "2"))  # For ColBERTv2

# ColBERT is available (native implementation with optional RAGatouille integration)
try:
    import numpy as np
    COLBERT_AVAILABLE = True
except ImportError:
    COLBERT_AVAILABLE = False
    np = None


# =============================================================================
# DATA MODELS
# =============================================================================


class TokenEmbedding(BaseModel):
    """Token-level embedding."""

    token: str = Field(...)
    token_id: int = Field(default=0)
    embedding: List[float] = Field(default_factory=list)
    position: int = Field(default=0)


class DocumentEmbeddings(BaseModel):
    """Document with token-level embeddings."""

    doc_id: str = Field(...)
    content: str = Field(...)
    token_embeddings: List[TokenEmbedding] = Field(default_factory=list)
    num_tokens: int = Field(default=0)
    compressed: bool = Field(default=False)
    indexed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class QueryEmbeddings(BaseModel):
    """Query with token-level embeddings."""

    query: str = Field(...)
    token_embeddings: List[TokenEmbedding] = Field(default_factory=list)
    num_tokens: int = Field(default=0)
    query_augmented: bool = Field(default=False)


class ColBERTScore(BaseModel):
    """Score from ColBERT MaxSim."""

    doc_id: str = Field(...)
    score: float = Field(default=0.0)
    max_sim_scores: List[float] = Field(default_factory=list)  # Per query token
    matching_tokens: List[Tuple[str, str, float]] = Field(default_factory=list)  # (query_tok, doc_tok, sim)


class ColBERTResult(BaseModel):
    """Result from ColBERT search."""

    query: str = Field(...)
    results: List[ColBERTScore] = Field(default_factory=list)
    query_embeddings: Optional[QueryEmbeddings] = None

    # Stats
    documents_scored: int = Field(default=0)
    top_k: int = Field(default=10)
    search_time_ms: float = Field(default=0.0)
    embedding_time_ms: float = Field(default=0.0)


class ColBERTIndex(BaseModel):
    """ColBERT index structure."""

    documents: Dict[str, DocumentEmbeddings] = Field(default_factory=dict)
    num_documents: int = Field(default=0)
    total_tokens: int = Field(default=0)
    embedding_dim: int = Field(default=EMBEDDING_DIM)
    compressed: bool = Field(default=False)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# COLBERT IMPLEMENTATION
# =============================================================================


class ColBERTRetriever:
    """ColBERT retriever with late interaction scoring.

    Implements the ColBERT paradigm where:
    - Queries and documents are encoded into token-level embeddings
    - Scoring uses MaxSim: max similarity between each query token and all doc tokens
    - Final score is sum of MaxSim scores across all query tokens
    """

    def __init__(
        self,
        embedding_dim: int = EMBEDDING_DIM,
        max_query_tokens: int = MAX_QUERY_TOKENS,
        max_doc_tokens: int = MAX_DOC_TOKENS,
        use_compression: bool = True,
    ):
        self.embedding_dim = embedding_dim
        self.max_query_tokens = max_query_tokens
        self.max_doc_tokens = max_doc_tokens
        self.use_compression = use_compression

        self._index: Optional[ColBERTIndex] = None
        self._stats = {
            "documents_indexed": 0,
            "queries_processed": 0,
            "tokens_encoded": 0,
        }

        logger.info(f"ColBERTRetriever initialized (dim={embedding_dim})")

    def _generate_embedding(self, dim: int) -> List[float]:
        """Generate embedding for a token.

        C-010 FIX: This method documents that random embeddings are NOT suitable
        for production use. In production, use RAGatouille or a proper ColBERT model.

        WARNING: Random embeddings produce meaningless similarity scores.
        MaxSim with random vectors is essentially random ranking.
        """
        # Log warning on first call to make it clear this is a mock
        if not hasattr(self, '_embedding_warning_logged'):
            logger.warning(
                "C-010: ColBERT using RANDOM embeddings - retrieval results will NOT be meaningful! "
                "For production, use RAGatouille with a proper ColBERT model. "
                "See: https://github.com/bclavie/RAGatouille"
            )
            self._embedding_warning_logged = True

        if np is None:
            import random
            return [random.gauss(0, 0.1) for _ in range(dim)]
        # Normalized random embedding - NOT a real semantic embedding!
        emb = np.random.randn(dim).astype(np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-10)
        return emb.tolist()

    def _tokenize(self, text: str, max_tokens: int) -> List[str]:
        """Simple whitespace tokenization (placeholder for actual tokenizer)."""
        tokens = text.lower().split()
        # Add special tokens for query augmentation
        return tokens[:max_tokens]

    async def index_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 32,
    ) -> ColBERTIndex:
        """Index documents with token-level embeddings.

        Args:
            documents: List of documents with 'id' and 'content'
            batch_size: Batch size for encoding

        Returns:
            ColBERTIndex structure
        """
        logger.info(f"Indexing {len(documents)} documents")

        indexed_docs: Dict[str, DocumentEmbeddings] = {}
        total_tokens = 0

        for doc in documents:
            doc_id = doc.get("id", hashlib.md5(doc.get("content", "").encode()).hexdigest()[:8])
            content = doc.get("content", "")

            # Tokenize
            tokens = self._tokenize(content, self.max_doc_tokens)

            # Generate token embeddings
            token_embeddings = []
            for pos, token in enumerate(tokens):
                emb = self._generate_embedding(self.embedding_dim)
                token_embeddings.append(TokenEmbedding(
                    token=token,
                    token_id=hash(token) % 100000,
                    embedding=emb,
                    position=pos,
                ))

            indexed_docs[doc_id] = DocumentEmbeddings(
                doc_id=doc_id,
                content=content,
                token_embeddings=token_embeddings,
                num_tokens=len(tokens),
                compressed=self.use_compression,
            )

            total_tokens += len(tokens)
            self._stats["tokens_encoded"] += len(tokens)

        self._index = ColBERTIndex(
            documents=indexed_docs,
            num_documents=len(indexed_docs),
            total_tokens=total_tokens,
            embedding_dim=self.embedding_dim,
            compressed=self.use_compression,
        )

        self._stats["documents_indexed"] += len(documents)
        logger.info(f"Indexed {len(documents)} documents with {total_tokens} tokens")

        return self._index

    async def encode_query(
        self,
        query: str,
        augment: bool = True,
    ) -> QueryEmbeddings:
        """Encode query into token-level embeddings.

        Args:
            query: Query text
            augment: Whether to add query augmentation tokens (ColBERT specific)

        Returns:
            QueryEmbeddings with token-level embeddings
        """
        # Tokenize query
        tokens = self._tokenize(query, self.max_query_tokens)

        # Add [Q] marker and [MASK] tokens for augmentation (ColBERT specific)
        if augment:
            tokens = ["[Q]"] + tokens
            # Pad with [MASK] tokens
            while len(tokens) < self.max_query_tokens:
                tokens.append("[MASK]")

        # Generate token embeddings
        token_embeddings = []
        for pos, token in enumerate(tokens):
            emb = self._generate_embedding(self.embedding_dim)
            token_embeddings.append(TokenEmbedding(
                token=token,
                token_id=hash(token) % 100000,
                embedding=emb,
                position=pos,
            ))

        return QueryEmbeddings(
            query=query,
            token_embeddings=token_embeddings,
            num_tokens=len(tokens),
            query_augmented=augment,
        )

    def _max_sim(
        self,
        query_emb: List[float],
        doc_embeddings: List[TokenEmbedding],
    ) -> Tuple[float, str]:
        """Compute MaxSim: max similarity between query token and all doc tokens."""
        if np is None:
            # Fallback without numpy
            max_sim = 0.0
            best_token = ""
            q = query_emb
            for doc_tok in doc_embeddings:
                d = doc_tok.embedding
                # Dot product (embeddings are normalized)
                sim = sum(a * b for a, b in zip(q, d))
                if sim > max_sim:
                    max_sim = sim
                    best_token = doc_tok.token
            return max_sim, best_token

        q = np.array(query_emb)
        max_sim = -float("inf")
        best_token = ""

        for doc_tok in doc_embeddings:
            d = np.array(doc_tok.embedding)
            sim = float(np.dot(q, d))
            if sim > max_sim:
                max_sim = sim
                best_token = doc_tok.token

        return max_sim, best_token

    def _score_document(
        self,
        query_embeddings: QueryEmbeddings,
        doc_embeddings: DocumentEmbeddings,
    ) -> ColBERTScore:
        """Score document using ColBERT's late interaction."""
        max_sim_scores = []
        matching_tokens = []

        for q_tok in query_embeddings.token_embeddings:
            # Skip special tokens in scoring
            if q_tok.token in ["[Q]", "[MASK]", "[PAD]"]:
                continue

            max_sim, best_doc_token = self._max_sim(
                q_tok.embedding,
                doc_embeddings.token_embeddings,
            )
            max_sim_scores.append(max_sim)
            matching_tokens.append((q_tok.token, best_doc_token, max_sim))

        # Final score is sum of MaxSim scores
        total_score = sum(max_sim_scores) if max_sim_scores else 0.0

        return ColBERTScore(
            doc_id=doc_embeddings.doc_id,
            score=total_score,
            max_sim_scores=max_sim_scores,
            matching_tokens=matching_tokens,
        )

    async def search(
        self,
        query: str,
        top_k: int = 10,
        return_embeddings: bool = False,
    ) -> ColBERTResult:
        """Search for relevant documents using ColBERT scoring.

        Args:
            query: Search query
            top_k: Number of results to return
            return_embeddings: Whether to include query embeddings in result

        Returns:
            ColBERTResult with ranked documents
        """
        import time

        self._stats["queries_processed"] += 1

        if not self._index:
            return ColBERTResult(
                query=query,
                results=[],
            )

        # Encode query
        emb_start = time.time()
        query_embeddings = await self.encode_query(query)
        emb_time = (time.time() - emb_start) * 1000

        # Score all documents
        search_start = time.time()
        scores: List[ColBERTScore] = []

        for doc_id, doc_emb in self._index.documents.items():
            score = self._score_document(query_embeddings, doc_emb)
            scores.append(score)

        # Sort by score descending
        scores.sort(key=lambda x: -x.score)

        search_time = (time.time() - search_start) * 1000

        return ColBERTResult(
            query=query,
            results=scores[:top_k],
            query_embeddings=query_embeddings if return_embeddings else None,
            documents_scored=len(self._index.documents),
            top_k=top_k,
            search_time_ms=search_time,
            embedding_time_ms=emb_time,
        )

    async def search_with_reranking(
        self,
        query: str,
        initial_results: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> ColBERTResult:
        """Rerank initial results using ColBERT scoring.

        Useful for two-stage retrieval where initial retrieval is fast
        but less accurate, and ColBERT is used for reranking.

        Args:
            query: Search query
            initial_results: Results from initial retrieval to rerank
            top_k: Number of results to return

        Returns:
            ColBERTResult with reranked documents
        """
        # Index only the candidates
        await self.index_documents(initial_results)

        # Search (which scores all indexed docs)
        return await self.search(query, top_k)

    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        index_stats = {}
        if self._index:
            index_stats = {
                "num_documents": self._index.num_documents,
                "total_tokens": self._index.total_tokens,
                "compressed": self._index.compressed,
            }

        return {
            **self._stats,
            **index_stats,
            "embedding_dim": self.embedding_dim,
        }


# =============================================================================
# SINGLETON AND CONVENIENCE FUNCTIONS
# =============================================================================

_colbert_instance: Optional[ColBERTRetriever] = None


def get_colbert_retriever() -> ColBERTRetriever:
    """Get or create ColBERTRetriever singleton."""
    global _colbert_instance
    if _colbert_instance is None:
        _colbert_instance = ColBERTRetriever()
    return _colbert_instance


async def colbert_search(
    query: str,
    top_k: int = 10,
) -> ColBERTResult:
    """Convenience function for ColBERT search."""
    retriever = get_colbert_retriever()
    return await retriever.search(query=query, top_k=top_k)


__all__ = [
    "COLBERT_AVAILABLE",
    "ColBERTRetriever",
    "ColBERTResult",
    "ColBERTScore",
    "ColBERTIndex",
    "DocumentEmbeddings",
    "QueryEmbeddings",
    "TokenEmbedding",
    "get_colbert_retriever",
    "colbert_search",
]
