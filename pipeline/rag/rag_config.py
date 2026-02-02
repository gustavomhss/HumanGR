"""RAG Stacks Centralized Configuration.

This module provides centralized configuration for all 9 RAG stacks,
loading values from environment variables and YAML configuration files.

All configurations follow the RAG_STACKS_OPTIMIZATION_CHECKLIST.md
with OPTIMIZED values (not defaults).

Stacks:
    1. Qdrant - Vector Database for Embeddings
    2. Active RAG - Proactive Retrieval with Gap Detection
    3. Self-RAG - Self-Reflective Retrieval
    4. Corrective RAG (CRAG) - Self-Correcting Retrieval
    5. MemoRAG - Memory-Augmented Retrieval
    6. Mem0 - Long-Term Memory
    7. Letta - Stateful Agent Memory
    8. A-MEM - Atomic Notes (Zettelkasten)
    9. GraphRAG - Graph-Enhanced Retrieval

Author: Agent 1 - RAG Stacks Implementation Specialist
Version: 1.0.0
Date: 2026-01-21
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_env_float(key: str, default: float) -> float:
    """Get float from environment variable.

    Args:
        key: Environment variable name
        default: Default value if not found or invalid

    Returns:
        Float value from environment or default
    """
    if not key or not isinstance(key, str):
        logger.warning(f"Invalid key provided to get_env_float: {key}")
        return default
    try:
        return float(os.getenv(key, str(default)))
    except (ValueError, TypeError) as e:
        logger.debug(f"Failed to parse env var {key}: {e}")
        return default


def get_env_int(key: str, default: int) -> int:
    """Get int from environment variable.

    Args:
        key: Environment variable name
        default: Default value if not found or invalid

    Returns:
        Int value from environment or default
    """
    if not key or not isinstance(key, str):
        logger.warning(f"Invalid key provided to get_env_int: {key}")
        return default
    try:
        return int(os.getenv(key, str(default)))
    except (ValueError, TypeError) as e:
        logger.debug(f"Failed to parse env var {key}: {e}")
        return default


def get_env_bool(key: str, default: bool) -> bool:
    """Get bool from environment variable.

    Args:
        key: Environment variable name
        default: Default value if not found or invalid

    Returns:
        Bool value from environment or default
    """
    if not key or not isinstance(key, str):
        logger.warning(f"Invalid key provided to get_env_bool: {key}")
        return default
    val = os.getenv(key, str(default)).lower()
    return val in ("true", "1", "yes", "on")


# =============================================================================
# STACK 1: QDRANT CONFIGURATION
# =============================================================================


@dataclass
class QdrantConfig:
    """Qdrant Vector Database Configuration.

    Optimized values per RAG_STACKS_OPTIMIZATION_CHECKLIST.md
    """

    # Connection
    url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333"))
    grpc_port: int = field(default_factory=lambda: get_env_int("QDRANT_GRPC_PORT", 6334))
    prefer_grpc: bool = field(default_factory=lambda: get_env_bool("QDRANT_PREFER_GRPC", True))
    timeout: float = field(default_factory=lambda: get_env_float("QDRANT_TIMEOUT", 45.0))

    # Collection
    collection: str = field(default_factory=lambda: os.getenv("QDRANT_COLLECTION", "pipeline_v2_claims"))

    # Hybrid Search Weights (OPTIMIZED)
    dense_weight: float = field(default_factory=lambda: get_env_float("QDRANT_DENSE_WEIGHT", 0.45))
    sparse_weight: float = field(default_factory=lambda: get_env_float("QDRANT_SPARSE_WEIGHT", 0.40))
    bm25_weight: float = field(default_factory=lambda: get_env_float("QDRANT_BM25_WEIGHT", 0.15))

    # Retrieval
    default_top_k: int = field(default_factory=lambda: get_env_int("QDRANT_DEFAULT_TOP_K", 15))
    score_threshold: float = field(default_factory=lambda: get_env_float("QDRANT_SCORE_THRESHOLD", 0.35))

    # Connection Pooling
    pool_size: int = 10
    batch_size: int = 100

    # HNSW Index
    hnsw_ef: int = 256
    hnsw_m: int = 32

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Prefetch
    prefetch_window: int = 5

    def get_weights_for_operation(self, operation: str) -> Dict[str, float]:
        """Get operation-specific weights."""
        weights_map = {
            "spec": {"dense": 0.50, "sparse": 0.35, "bm25": 0.15},
            "exec": {"dense": 0.40, "sparse": 0.45, "bm25": 0.15},
            "gate": {"dense": 0.45, "sparse": 0.40, "bm25": 0.15},
        }
        return weights_map.get(operation, {
            "dense": self.dense_weight,
            "sparse": self.sparse_weight,
            "bm25": self.bm25_weight,
        })


# =============================================================================
# STACK 2: ACTIVE RAG CONFIGURATION
# =============================================================================


@dataclass
class ActiveRAGConfig:
    """Active RAG Configuration - Proactive Retrieval.

    Optimized values per RAG_STACKS_OPTIMIZATION_CHECKLIST.md
    """

    enabled: bool = True

    # Cache
    prediction_cache_ttl: int = field(default_factory=lambda: get_env_int("PREDICTION_CACHE_TTL", 600))
    max_prefetch_queries: int = field(default_factory=lambda: get_env_int("MAX_PREFETCH_QUERIES", 8))

    # Iterative Retrieval
    max_iterations: int = field(default_factory=lambda: get_env_int("ACTIVE_RAG_MAX_ITERATIONS", 4))
    convergence_threshold: float = field(
        default_factory=lambda: get_env_float("ACTIVE_RAG_CONVERGENCE_THRESHOLD", 0.85)
    )
    max_refinements: int = 3
    feedback_weight: float = 0.7

    # Pattern Analysis
    pattern_window_hours: int = 48
    min_pattern_count: int = 2

    # LLM Prediction
    llm_prediction_enabled: bool = True
    llm_prediction_min_confidence: float = 0.6

    # Prefetch
    prefetch_timeout_ms: int = 2000
    prefetch_async: bool = True

    # Prediction Strategy Weights
    prediction_weights: Dict[str, float] = field(default_factory=lambda: {
        "historical": 0.30,
        "topic": 0.25,
        "template": 0.25,
        "llm": 0.20,
    })


# =============================================================================
# STACK 3: SELF-RAG CONFIGURATION
# =============================================================================


@dataclass
class SelfRAGConfig:
    """Self-RAG Configuration - Self-Reflective Retrieval.

    Optimized values per RAG_STACKS_OPTIMIZATION_CHECKLIST.md
    """

    enabled: bool = field(default_factory=lambda: get_env_bool("SELF_RAG_ENABLED", True))

    # Main Thresholds (LOWERED for better recall)
    relevance_threshold: float = field(
        default_factory=lambda: get_env_float("SELF_RAG_RELEVANCE_THRESHOLD", 0.65)
    )
    support_threshold: float = field(
        default_factory=lambda: get_env_float("SELF_RAG_SUPPORT_THRESHOLD", 0.55)
    )
    utility_threshold: float = field(
        default_factory=lambda: get_env_float("SELF_RAG_UTILITY_THRESHOLD", 0.55)
    )
    max_iterations: int = field(
        default_factory=lambda: get_env_int("SELF_RAG_MAX_ITERATIONS", 4)
    )

    # Reflection Token Thresholds
    retrieval_yes_threshold: float = 0.45
    relevance_relevant_threshold: float = 0.65
    support_fully_supported_threshold: float = 0.55
    support_partially_supported_threshold: float = 0.35
    utility_weight: float = 0.35

    # Reflection Weights
    reflection_weights: Dict[str, float] = field(default_factory=lambda: {
        "relevance": 0.35,
        "support": 0.40,
        "utility": 0.25,
    })

    # Early Exit
    early_exit_enabled: bool = True
    early_exit_all_tokens_pass: bool = True
    early_exit_convergence_detected: bool = True
    early_exit_max_utility: float = 0.9

    # Adaptive Iterations
    adaptive_iterations_enabled: bool = True
    high_quality_threshold: float = 0.8
    medium_quality_threshold: float = 0.6


# =============================================================================
# STACK 4: CORRECTIVE RAG (CRAG) CONFIGURATION
# =============================================================================


@dataclass
class CorrectiveRAGConfig:
    """Corrective RAG Configuration - Self-Correcting Retrieval.

    Optimized values per RAG_STACKS_OPTIMIZATION_CHECKLIST.md
    """

    enabled: bool = True

    # Quality Thresholds
    quality_threshold: float = field(
        default_factory=lambda: get_env_float("CRAG_QUALITY_THRESHOLD", 0.55)
    )

    # Ambiguity Range (NARROWED)
    ambiguity_low: float = field(
        default_factory=lambda: get_env_float("CRAG_AMBIGUITY_LOW", 0.40)
    )
    ambiguity_high: float = field(
        default_factory=lambda: get_env_float("CRAG_AMBIGUITY_HIGH", 0.65)
    )

    @property
    def ambiguity_range(self) -> Tuple[float, float]:
        """Get ambiguity range tuple."""
        return (self.ambiguity_low, self.ambiguity_high)

    # Knowledge Strips
    max_strips: int = field(
        default_factory=lambda: get_env_int("CRAG_MAX_STRIPS", 15)
    )
    strip_min_length: int = 25
    strip_relevance_threshold: float = 0.35
    strip_confidence_min: float = 0.60

    # Strip Type Weights
    strip_type_weights: Dict[str, float] = field(default_factory=lambda: {
        "factual": 1.0,
        "definition": 0.9,
        "statistic": 1.1,
        "citation": 1.2,
        "example": 0.8,
        "opinion": 0.5,
    })

    # Web Search
    web_search_enabled: bool = field(
        default_factory=lambda: get_env_bool("CRAG_WEB_SEARCH_ENABLED", True)
    )
    web_search_max_requests_per_minute: int = 10
    web_search_timeout: float = 5.0
    web_search_max_results: int = 5
    web_search_prefer_sources: List[str] = field(default_factory=lambda: [
        "wikipedia", "arxiv", "gov", "edu"
    ])

    # Hysteresis
    hysteresis_enabled: bool = True
    hysteresis_correct_stay_threshold: float = 0.55
    hysteresis_incorrect_stay_threshold: float = 0.50


# =============================================================================
# STACK 5: MEMORAG CONFIGURATION
# =============================================================================


@dataclass
class MemoRAGConfig:
    """MemoRAG Configuration - Memory-Augmented Retrieval.

    Optimized values per RAG_STACKS_OPTIMIZATION_CHECKLIST.md
    """

    enabled: bool = True

    # Cache (EXPANDED)
    cache_size: int = field(
        default_factory=lambda: get_env_int("MEMO_CACHE_SIZE", 5000)
    )
    compression_ratio: float = field(
        default_factory=lambda: get_env_float("MEMO_COMPRESSION_RATIO", 0.25)
    )

    # Episodic Memory
    episodic_memory_window: int = field(
        default_factory=lambda: get_env_int("EPISODIC_MEMORY_WINDOW", 250)
    )
    lru_eviction_batch: int = field(
        default_factory=lambda: get_env_int("MEMO_LRU_EVICTION_BATCH", 10)
    )

    # Compression Settings
    keep_top_k_queries: int = 25
    topic_extraction_limit: int = 30
    entity_extraction_limit: int = 20
    min_query_similarity: float = 0.3

    # Deduplication
    deduplication_enabled: bool = True
    dedup_similarity_threshold: float = 0.85

    # Access Pattern
    access_recency_weight: float = 0.6
    access_frequency_weight: float = 0.4


# =============================================================================
# STACK 6: MEM0 CONFIGURATION
# =============================================================================


@dataclass
class Mem0Config:
    """Mem0 Configuration - Long-Term Memory.

    Optimized values per RAG_STACKS_OPTIMIZATION_CHECKLIST.md
    """

    enabled: bool = True

    # Connection
    url: str = field(default_factory=lambda: os.getenv("MEM0_URL", "http://localhost:8080"))
    timeout: float = field(default_factory=lambda: get_env_float("MEM0_TIMEOUT", 45.0))

    # User
    default_user_id: str = field(
        default_factory=lambda: os.getenv("MEM0_DEFAULT_USER_ID", "pipeline_autonomo_v2")
    )

    # Memory Store
    memory_type: str = "hybrid"
    retention_days: int = field(
        default_factory=lambda: get_env_int("MEM0_RETENTION_DAYS", 90)
    )
    max_memories_per_user: int = field(
        default_factory=lambda: get_env_int("MEM0_MAX_MEMORIES", 2500)
    )
    embedding_model: str = "nomic-embed-text"

    # Categories
    categories: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "preferences": {"priority": 1, "retention": "permanent"},
        "few_shots": {"priority": 2, "retention": "90_days"},
        "interactions": {"priority": 3, "retention": "30_days"},
    })


# =============================================================================
# STACK 7: LETTA CONFIGURATION
# =============================================================================


@dataclass
class LettaConfig:
    """Letta Configuration - Stateful Agent Memory.

    Optimized values per RAG_STACKS_OPTIMIZATION_CHECKLIST.md
    """

    enabled: bool = True

    # Connection
    service_url: str = field(
        default_factory=lambda: os.getenv("LETTA_SERVICE_URL", "http://localhost:8283")
    )
    timeout: float = field(default_factory=lambda: get_env_float("LETTA_TIMEOUT", 90.0))
    default_model: str = field(
        default_factory=lambda: os.getenv("LETTA_DEFAULT_MODEL", "claude-3-5-sonnet")
    )

    # Memory Configuration
    persona_token_limit: int = 3000
    human_token_limit: int = 1500
    memory_pressure_threshold: float = 0.85
    recall_topk: int = 15

    # Personas
    personas: Dict[str, str] = field(default_factory=lambda: {
        "spec_master": """You are the Spec Master, responsible for creating precise specifications.
You have deep expertise in requirement analysis and technical writing.
You always ensure specifications are complete, testable, and unambiguous.
Key traits: Precise, thorough, detail-oriented, systematic.""",

        "ace_exec": """You are the Ace Exec, the elite code executor.
You write clean, efficient, well-tested code.
You follow best practices and patterns from the codebase.
Key traits: Efficient, practical, quality-focused, pragmatic.""",

        "qa_master": """You are the QA Master, guardian of quality.
You ensure all outputs meet the highest standards.
You identify edge cases and potential issues proactively.
Key traits: Critical, thorough, systematic, detail-oriented.""",
    })


# =============================================================================
# STACK 8: A-MEM CONFIGURATION
# =============================================================================


@dataclass
class AMEMConfig:
    """A-MEM Configuration - Atomic Notes (Zettelkasten).

    Optimized values per RAG_STACKS_OPTIMIZATION_CHECKLIST.md
    """

    enabled: bool = True

    # Linking
    link_threshold: float = field(
        default_factory=lambda: get_env_float("AMEM_LINK_THRESHOLD", 0.65)
    )
    max_links: int = field(
        default_factory=lambda: get_env_int("AMEM_MAX_LINKS", 15)
    )
    min_note_length: int = field(
        default_factory=lambda: get_env_int("AMEM_MIN_NOTE_LENGTH", 50)
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("AMEM_EMBEDDING_MODEL", "nomic-embed-text")
    )

    # Notes
    max_notes_per_topic: int = 200
    link_decay_days: int = 30
    auto_link_on_create: bool = True

    # Note Type Weights
    note_type_weights: Dict[str, float] = field(default_factory=lambda: {
        "error": 1.5,
        "correction": 1.4,
        "learning": 1.2,
        "pattern": 1.1,
        "observation": 0.8,
        "note": 0.7,
    })

    # Link Score Weights
    link_score_weights: Dict[str, float] = field(default_factory=lambda: {
        "semantic": 0.40,
        "topic_overlap": 0.25,
        "entity_overlap": 0.20,
        "type_compatibility": 0.15,
    })

    # Retrieval
    retrieval_graph_depth: int = 2
    retrieval_limit: int = 15
    retrieval_expand_top_links: int = 5


# =============================================================================
# STACK 9: GRAPHRAG CONFIGURATION
# =============================================================================


@dataclass
class GraphRAGConfig:
    """GraphRAG Configuration - Graph-Enhanced Retrieval.

    Optimized values per RAG_STACKS_OPTIMIZATION_CHECKLIST.md
    """

    enabled: bool = True

    # Connection
    service_url: str = field(
        default_factory=lambda: os.getenv("GRAPHRAG_SERVICE_URL", "http://localhost:50052")
    )
    timeout: float = field(default_factory=lambda: get_env_float("GRAPHRAG_TIMEOUT", 90.0))

    # Search
    search_type: str = field(
        default_factory=lambda: os.getenv("GRAPHRAG_SEARCH_TYPE", "hybrid")
    )
    top_k: int = field(default_factory=lambda: get_env_int("GRAPHRAG_TOP_K", 15))

    # Graph
    community_level: int = 2
    min_community_size: int = 3
    entity_types: List[str] = field(default_factory=lambda: ["claim", "source", "topic"])
    relationship_types: List[str] = field(default_factory=lambda: ["supports", "contradicts", "cites"])

    # Claim Verification
    claim_support_ratio_threshold: float = 2.0


# =============================================================================
# MASTER CONFIGURATION
# =============================================================================


@dataclass
class RAGStacksConfig:
    """Master configuration for all RAG stacks."""

    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    active_rag: ActiveRAGConfig = field(default_factory=ActiveRAGConfig)
    self_rag: SelfRAGConfig = field(default_factory=SelfRAGConfig)
    corrective_rag: CorrectiveRAGConfig = field(default_factory=CorrectiveRAGConfig)
    memo_rag: MemoRAGConfig = field(default_factory=MemoRAGConfig)
    mem0: Mem0Config = field(default_factory=Mem0Config)
    letta: LettaConfig = field(default_factory=LettaConfig)
    amem: AMEMConfig = field(default_factory=AMEMConfig)
    graphrag: GraphRAGConfig = field(default_factory=GraphRAGConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "RAGStacksConfig":
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            RAGStacksConfig loaded from file or defaults on error
        """
        if not YAML_AVAILABLE:
            logger.warning("PyYAML not available, using defaults")
            return cls()

        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)

            if data is None:
                logger.warning(f"Empty YAML file at {path}, using defaults")
                return cls()

            config = cls()

            # Apply YAML overrides to each stack
            if "qdrant" in data:
                for key, value in data["qdrant"].items():
                    if hasattr(config.qdrant, key):
                        setattr(config.qdrant, key, value)

            if "active_rag" in data:
                for key, value in data["active_rag"].items():
                    if hasattr(config.active_rag, key):
                        setattr(config.active_rag, key, value)

            if "self_rag" in data:
                for key, value in data["self_rag"].items():
                    if hasattr(config.self_rag, key):
                        setattr(config.self_rag, key, value)

            if "corrective_rag" in data:
                for key, value in data["corrective_rag"].items():
                    if hasattr(config.corrective_rag, key):
                        setattr(config.corrective_rag, key, value)

            if "memo_rag" in data:
                for key, value in data["memo_rag"].items():
                    if hasattr(config.memo_rag, key):
                        setattr(config.memo_rag, key, value)

            if "mem0" in data:
                for key, value in data["mem0"].items():
                    if hasattr(config.mem0, key):
                        setattr(config.mem0, key, value)

            if "letta" in data:
                for key, value in data["letta"].items():
                    if hasattr(config.letta, key):
                        setattr(config.letta, key, value)

            if "amem" in data:
                for key, value in data["amem"].items():
                    if hasattr(config.amem, key):
                        setattr(config.amem, key, value)

            if "graphrag" in data:
                for key, value in data["graphrag"].items():
                    if hasattr(config.graphrag, key):
                        setattr(config.graphrag, key, value)

            return config

        except FileNotFoundError:
            logger.warning(f"Config file not found at {path}, using defaults")
            return cls()
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error in {path}: {e}")
            return cls()
        except (OSError, IOError) as e:
            logger.error(f"IO error reading {path}: {e}")
            return cls()

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        from dataclasses import asdict
        return asdict(self)


# =============================================================================
# SINGLETON INSTANCE (Thread-Safe)
# =============================================================================

import threading

_config: Optional[RAGStacksConfig] = None
_config_lock = threading.Lock()


def get_rag_config() -> RAGStacksConfig:
    """Get singleton RAG stacks configuration (thread-safe).

    Returns:
        RAGStacksConfig singleton instance
    """
    global _config
    if _config is None:
        with _config_lock:
            # Double-check pattern for thread safety
            if _config is None:
                # Try to load from YAML if available
                yaml_path = Path(__file__).parent.parent.parent.parent / "configs" / "pipeline_autonomo" / "rag_stacks" / "rag_stacks_config.yaml"
                if yaml_path.exists():
                    _config = RAGStacksConfig.from_yaml(str(yaml_path))
                else:
                    _config = RAGStacksConfig()
    return _config


def get_qdrant_config() -> QdrantConfig:
    """Get Qdrant configuration."""
    return get_rag_config().qdrant


def get_active_rag_config() -> ActiveRAGConfig:
    """Get Active RAG configuration."""
    return get_rag_config().active_rag


def get_self_rag_config() -> SelfRAGConfig:
    """Get Self-RAG configuration."""
    return get_rag_config().self_rag


def get_corrective_rag_config() -> CorrectiveRAGConfig:
    """Get Corrective RAG configuration."""
    return get_rag_config().corrective_rag


def get_memo_rag_config() -> MemoRAGConfig:
    """Get MemoRAG configuration."""
    return get_rag_config().memo_rag


def get_mem0_config() -> Mem0Config:
    """Get Mem0 configuration."""
    return get_rag_config().mem0


def get_letta_config() -> LettaConfig:
    """Get Letta configuration."""
    return get_rag_config().letta


def get_amem_config() -> AMEMConfig:
    """Get A-MEM configuration."""
    return get_rag_config().amem


def get_graphrag_config() -> GraphRAGConfig:
    """Get GraphRAG configuration."""
    return get_rag_config().graphrag


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
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
]
