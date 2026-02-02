"""Pipeline V2 RAG Module.

This module contains RAG (Retrieval Augmented Generation) services.

Components:
- active_rag: Proactive RAG with query prediction, prefetching, and feedback loops
- qdrant_hybrid: Hybrid search combining dense and sparse vectors

Usage:
    from pipeline.rag import (
        # Active RAG
        ActiveRAG,
        predict_next_queries,
        prefetch_context,
        # Hybrid Search
        QdrantHybridSearch,
        hybrid_search,
        create_hybrid_collection,
    )

    # Initialize Active RAG
    rag = ActiveRAG()

    # Predict what queries might come next
    predictions = await rag.predict_next_queries(current_query="climate change")

    # Prefetch context proactively
    contexts = await rag.prefetch_context(predictions)

    # Iterative retrieval with feedback
    result = await rag.iterative_retrieve(query="climate evidence", max_iterations=3)

    # Hybrid search (dense + sparse)
    results = await hybrid_search("claims", "global warming evidence")
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Import RAG Configuration (always available)
try:
    from pipeline.rag.rag_config import (
        # Individual Configs
        QdrantConfig,
        ActiveRAGConfig,
        SelfRAGConfig,
        CorrectiveRAGConfig,
        MemoRAGConfig,
        Mem0Config,
        LettaConfig,
        AMEMConfig,
        GraphRAGConfig,
        # Master Config
        RAGStacksConfig,
        # Getters
        get_rag_config,
        get_qdrant_config,
        get_active_rag_config,
        get_self_rag_config,
        get_corrective_rag_config,
        get_memo_rag_config,
        get_mem0_config,
        get_letta_config,
        get_amem_config,
        get_graphrag_config,
    )
    RAG_CONFIG_AVAILABLE = True
except ImportError as e:
    logger.debug(f"RAG config not available: {e}")
    RAG_CONFIG_AVAILABLE = False

# Import Active RAG
try:
    from pipeline.rag.active_rag import (
        ActiveRAG,
        QueryPrediction,
        PrefetchResult,
        RAGContext,
        RetrievalFeedback,
        IterativeRetrievalResult,
        predict_next_queries,
        prefetch_context,
        get_active_rag,
        ACTIVE_RAG_AVAILABLE,
    )
    _ACTIVE_RAG_IMPORT_SUCCESS = True
except ImportError as e:
    logger.debug(f"Active RAG not available: {e}")
    _ACTIVE_RAG_IMPORT_SUCCESS = False
    ACTIVE_RAG_AVAILABLE = False

# Import Qdrant Hybrid Search
try:
    from pipeline.rag.qdrant_hybrid import (
        QdrantHybridSearch,
        SparseVectorizer,
        HybridSearchResult,
        DocumentInput,
        CollectionStats,
        FilterCondition,
        get_qdrant_hybrid,
        create_hybrid_collection,
        hybrid_search,
        get_sparse_vector,
        QDRANT_AVAILABLE,
    )
    _QDRANT_IMPORT_SUCCESS = True
except ImportError as e:
    logger.debug(f"Qdrant hybrid search not available: {e}")
    _QDRANT_IMPORT_SUCCESS = False
    QDRANT_AVAILABLE = False

__all__ = [
    "ACTIVE_RAG_AVAILABLE",
    "QDRANT_AVAILABLE",
]

if _ACTIVE_RAG_IMPORT_SUCCESS:
    __all__.extend([
        "ActiveRAG",
        "QueryPrediction",
        "PrefetchResult",
        "RAGContext",
        "RetrievalFeedback",
        "IterativeRetrievalResult",
        "predict_next_queries",
        "prefetch_context",
        "get_active_rag",
    ])

if _QDRANT_IMPORT_SUCCESS:
    __all__.extend([
        "QdrantHybridSearch",
        "SparseVectorizer",
        "HybridSearchResult",
        "DocumentInput",
        "CollectionStats",
        "FilterCondition",
        "get_qdrant_hybrid",
        "create_hybrid_collection",
        "hybrid_search",
        "get_sparse_vector",
    ])

# Import MemoRAG
try:
    from pipeline.rag.memo_rag import (
        MemoRAG,
        MemoRAGResult,
        MemoryEntry,
        CompressedState,
        SprintStateCompression,
        get_memo_rag,
        MEMO_RAG_AVAILABLE,
    )
    _MEMO_RAG_IMPORT_SUCCESS = True
except ImportError as e:
    logger.debug(f"MemoRAG not available: {e}")
    _MEMO_RAG_IMPORT_SUCCESS = False
    MEMO_RAG_AVAILABLE = False

if _MEMO_RAG_IMPORT_SUCCESS:
    __all__.extend([
        "MemoRAG",
        "MemoRAGResult",
        "MemoryEntry",
        "CompressedState",
        "SprintStateCompression",
        "get_memo_rag",
        "MEMO_RAG_AVAILABLE",
    ])
else:
    __all__.append("MEMO_RAG_AVAILABLE")

# Import Self-RAG
try:
    from pipeline.rag.self_rag import (
        SelfRAG,
        SelfRAGResult,
        RetrievalToken,
        RelevanceToken,
        SupportToken,
        UtilityToken,
        DocumentRelevance,
        SupportAssessment,
        CritiqueResult,
        get_self_rag,
        SELF_RAG_AVAILABLE,
    )
    _SELF_RAG_IMPORT_SUCCESS = True
except ImportError as e:
    logger.debug(f"Self-RAG not available: {e}")
    _SELF_RAG_IMPORT_SUCCESS = False
    SELF_RAG_AVAILABLE = False

if _SELF_RAG_IMPORT_SUCCESS:
    __all__.extend([
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
        "SELF_RAG_AVAILABLE",
    ])
else:
    __all__.append("SELF_RAG_AVAILABLE")

# Import Corrective RAG (CRAG)
try:
    from pipeline.rag.corrective_rag import (
        CorrectiveRAG,
        CRAGResult,
        RetrievalAction,
        RetrievalEvaluation,
        KnowledgeStrip,
        KnowledgeStripType,
        WebSearchResult,
        get_corrective_rag,
        CRAG_AVAILABLE,
    )
    _CRAG_IMPORT_SUCCESS = True
except ImportError as e:
    logger.debug(f"Corrective RAG not available: {e}")
    _CRAG_IMPORT_SUCCESS = False
    CRAG_AVAILABLE = False

if _CRAG_IMPORT_SUCCESS:
    __all__.extend([
        "CorrectiveRAG",
        "CRAGResult",
        "RetrievalAction",
        "RetrievalEvaluation",
        "KnowledgeStrip",
        "KnowledgeStripType",
        "WebSearchResult",
        "get_corrective_rag",
        "CRAG_AVAILABLE",
    ])
else:
    __all__.append("CRAG_AVAILABLE")

# Import RAPTOR RAG
try:
    from pipeline.rag.raptor_rag import (
        RaptorRAG,
        RaptorTree,
        RaptorNode,
        RaptorQueryResult,
        TraversalStrategy,
        get_raptor_rag,
        RAPTOR_AVAILABLE,
    )
    _RAPTOR_IMPORT_SUCCESS = True
except ImportError as e:
    logger.debug(f"RAPTOR RAG not available: {e}")
    _RAPTOR_IMPORT_SUCCESS = False
    RAPTOR_AVAILABLE = False

if _RAPTOR_IMPORT_SUCCESS:
    __all__.extend([
        "RaptorRAG",
        "RaptorTree",
        "RaptorNode",
        "RaptorQueryResult",
        "TraversalStrategy",
        "get_raptor_rag",
        "RAPTOR_AVAILABLE",
    ])
else:
    __all__.append("RAPTOR_AVAILABLE")

# Import ColBERT Retriever
try:
    from pipeline.rag.colbert_retriever import (
        ColBERTRetriever,
        ColBERTResult,
        ColBERTScore,
        ColBERTIndex,
        DocumentEmbeddings,
        QueryEmbeddings,
        TokenEmbedding,
        get_colbert_retriever,
        COLBERT_AVAILABLE,
    )
    _COLBERT_IMPORT_SUCCESS = True
except ImportError as e:
    logger.debug(f"ColBERT retriever not available: {e}")
    _COLBERT_IMPORT_SUCCESS = False
    COLBERT_AVAILABLE = False

if _COLBERT_IMPORT_SUCCESS:
    __all__.extend([
        "ColBERTRetriever",
        "ColBERTResult",
        "ColBERTScore",
        "ColBERTIndex",
        "DocumentEmbeddings",
        "QueryEmbeddings",
        "TokenEmbedding",
        "get_colbert_retriever",
        "COLBERT_AVAILABLE",
    ])
else:
    __all__.append("COLBERT_AVAILABLE")

# Add RAG Config exports
if RAG_CONFIG_AVAILABLE:
    __all__.extend([
        "RAG_CONFIG_AVAILABLE",
        # Individual Configs
        "QdrantConfig",
        "ActiveRAGConfig",
        "SelfRAGConfig",
        "CorrectiveRAGConfig",
        "MemoRAGConfig",
        "Mem0Config",
        "LettaConfig",
        "AMEMConfig",
        "GraphRAGConfig",
        # Master Config
        "RAGStacksConfig",
        # Getters
        "get_rag_config",
        "get_qdrant_config",
        "get_active_rag_config",
        "get_self_rag_config",
        "get_corrective_rag_config",
        "get_memo_rag_config",
        "get_mem0_config",
        "get_letta_config",
        "get_amem_config",
        "get_graphrag_config",
    ])
else:
    __all__.append("RAG_CONFIG_AVAILABLE")
