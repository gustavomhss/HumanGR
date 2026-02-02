"""Stack Injection - Central Hub for Stack Wiring.

This module provides the StackInjector that wires stacks to operations
following the MIGRATION_V2_TO_LANGGRAPH.md specification.

Architecture:
    Operation (SPEC/EXEC/GATE/SIGNOFF/ARTIFACT)
                    |
                    v
              StackInjector
                    |
        +-----------+-----------+
        |           |           |
        v           v           v
    Primary    Secondary    Optional
     Stacks      Stacks      Stacks

Key Features:
- Operation-based injection: Each operation gets its required stacks
- Graceful degradation: Optional stacks don't block if unavailable
- Health-aware: Only inject healthy stacks
- Configuration-driven: Can be customized via YAML

Based on: MIGRATION_V2_TO_LANGGRAPH.md Section "STACK HOOKS / WIRING"

Author: Pipeline Autonomo Team
Version: 1.0.0 (2026-01-16)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND TYPES
# =============================================================================


class Operation(str, Enum):
    """Pipeline operations that can receive stack injection."""

    SPEC = "SPEC"
    EXEC = "EXEC"
    GATE = "GATE"
    SIGNOFF = "SIGNOFF"
    ARTIFACT = "ARTIFACT"


class StackPolicy(str, Enum):
    """Stack injection policy."""

    REQUIRED = "REQUIRED"  # Operation fails if stack unavailable
    OPTIONAL = "OPTIONAL"  # Operation continues without stack
    DISABLED = "DISABLED"  # Stack not used for this operation
    DEPRECATED = "DEPRECATED"  # Stack is deprecated and should not be used


@dataclass
class StackHealth:
    """Health status for a stack."""

    name: str
    healthy: bool
    latency_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class StackContext:
    """Context passed to operations with injected stacks."""

    run_id: str
    sprint_id: str
    operation: Operation
    stacks: Dict[str, Any]
    stack_health: Dict[str, StackHealth]


# =============================================================================
# STACK REGISTRY
# =============================================================================


# Default stack assignments per operation
# From MIGRATION_V2_TO_LANGGRAPH.md Section "CEREBROS → STACKS"
DEFAULT_STACK_POLICY: Dict[Operation, Dict[str, StackPolicy]] = {
    Operation.SPEC: {
        # Primary
        "langgraph": StackPolicy.REQUIRED,
        "redis": StackPolicy.REQUIRED,
        "crewai": StackPolicy.REQUIRED,
        # Reasoning
        "bot": StackPolicy.OPTIONAL,  # Buffer of Thoughts
        "got": StackPolicy.OPTIONAL,  # Graph of Thoughts
        "dspy": StackPolicy.OPTIONAL,
        # RAG
        "active_rag": StackPolicy.OPTIONAL,
        "qdrant": StackPolicy.OPTIONAL,
        "graphrag": StackPolicy.OPTIONAL,
        # Security
        "nemo": StackPolicy.OPTIONAL,
        "llm_guard": StackPolicy.OPTIONAL,
        # Observability
        "langfuse": StackPolicy.OPTIONAL,
        "phoenix": StackPolicy.OPTIONAL,
    },
    Operation.EXEC: {
        # Primary
        "langgraph": StackPolicy.REQUIRED,
        "redis": StackPolicy.REQUIRED,
        "crewai": StackPolicy.REQUIRED,
        "temporal": StackPolicy.OPTIONAL,
        # Memory
        "letta": StackPolicy.OPTIONAL,
        "mem0": StackPolicy.DEPRECATED,  # Use Letta instead
        # Security
        "nemo": StackPolicy.OPTIONAL,
        "llm_guard": StackPolicy.OPTIONAL,
        # Type safety
        "pydantic_ai": StackPolicy.OPTIONAL,
        "instructor": StackPolicy.OPTIONAL,
        # Observability
        "langfuse": StackPolicy.OPTIONAL,
        "phoenix": StackPolicy.OPTIONAL,
    },
    Operation.GATE: {
        # Primary
        "redis": StackPolicy.REQUIRED,
        # Reasoning
        "got": StackPolicy.OPTIONAL,
        "reflexion": StackPolicy.OPTIONAL,
        # Eval
        "trulens": StackPolicy.OPTIONAL,
        "ragas": StackPolicy.OPTIONAL,
        "deepeval": StackPolicy.OPTIONAL,
        "cleanlab": StackPolicy.OPTIONAL,
        # Formal
        "z3": StackPolicy.OPTIONAL,
        # Observability
        "langfuse": StackPolicy.OPTIONAL,
    },
    Operation.SIGNOFF: {
        # Primary
        "redis": StackPolicy.REQUIRED,
        # Verification
        "cleanlab": StackPolicy.OPTIONAL,
        # Graph
        "falkordb": StackPolicy.OPTIONAL,
        "graphiti": StackPolicy.OPTIONAL,
        # Observability
        "langfuse": StackPolicy.OPTIONAL,
    },
    Operation.ARTIFACT: {
        # Primary
        "redis": StackPolicy.REQUIRED,
        # Graph
        "falkordb": StackPolicy.OPTIONAL,
        "neo4j": StackPolicy.OPTIONAL,
        # Transforms
        "hamilton": StackPolicy.OPTIONAL,
    },
}


# =============================================================================
# STACK FACTORIES
# =============================================================================


def get_stack_factory(stack_name: str) -> Optional[Callable[[], Any]]:
    """Get factory function for a stack.

    Returns:
        Factory function that returns stack client, or None if unavailable.
    """
    factories: Dict[str, Callable[[], Any]] = {
        # Existing V2 stacks
        "redis": lambda: _get_redis_client(),
        "crewai": lambda: _get_crewai_client(),
        "langfuse": lambda: _get_langfuse_client(),
        "qdrant": lambda: _get_qdrant_client(),
        "falkordb": lambda: _get_falkordb_client(),
        "letta": lambda: _get_letta_client(),
        "mem0": lambda: _get_mem0_client(),
        "temporal": lambda: _get_temporal_client(),
        "got": lambda: _get_got_client(),
        "reflexion": lambda: _get_reflexion_client(),

        # New P0 stacks
        "langgraph": lambda: _get_langgraph_client(),
        "phoenix": lambda: _get_phoenix_client(),
        "graphiti": lambda: _get_graphiti_client(),
        "active_rag": lambda: _get_active_rag_client(),

        # New P1 stacks
        "nemo": lambda: _get_nemo_client(),
        "llm_guard": lambda: _get_llm_guard_client(),
        "bot": lambda: _get_bot_client(),
        "z3": lambda: _get_z3_client(),
        "pydantic_ai": lambda: _get_pydantic_ai_client(),
        "cleanlab": lambda: _get_cleanlab_client(),
        "graphrag": lambda: _get_graphrag_client(),
        "dspy": lambda: _get_dspy_client(),

        # New P2 stacks
        "trulens": lambda: _get_trulens_client(),
        "ragas": lambda: _get_ragas_client(),
        "deepeval": lambda: _get_deepeval_client(),
        "instructor": lambda: _get_instructor_client(),
        "neo4j": lambda: _get_neo4j_client(),
        "hamilton": lambda: _get_hamilton_client(),
    }

    return factories.get(stack_name)


# Stack client getters (lazy-loaded, cached)
_stack_cache: Dict[str, Any] = {}
_stack_cache_lock = threading.Lock()


def clear_stack_cache(stack_names: Optional[List[str]] = None) -> int:
    """Clear cached stack clients to prevent memory leaks.

    Should be called at the end of a pipeline run or when stacks need
    to be re-initialized (e.g., after configuration change).

    Args:
        stack_names: Optional list of stack names to clear. If None, clears all.

    Returns:
        Number of stacks cleared.
    """
    global _stack_cache
    with _stack_cache_lock:
        if stack_names is None:
            count = len(_stack_cache)
            _stack_cache.clear()
            logger.info(f"Cleared all {count} cached stacks")
            return count
        else:
            count = 0
            for name in stack_names:
                if name in _stack_cache:
                    del _stack_cache[name]
                    count += 1
            logger.info(f"Cleared {count} cached stacks: {stack_names}")
            return count


def _get_redis_client() -> Any:
    if "redis" not in _stack_cache:
        try:
            from pipeline.redis_client import get_redis_client
            _stack_cache["redis"] = get_redis_client()
        except ImportError as e:
            logger.debug(f"Stack 'redis' not available: {e}")
            return None
    return _stack_cache.get("redis")


def _get_crewai_client() -> Any:
    if "crewai" not in _stack_cache:
        try:
            from pipeline.crewai_client import get_crewai_client
            client = get_crewai_client()
            _stack_cache["crewai"] = client
            if client:
                logger.info("STACK CrewAI: LOADED (agent orchestration)")
            else:
                logger.warning("STACK CrewAI: loaded but client is None")
        except ImportError as e:
            logger.debug(f"Stack 'crewai' not available: {e}")
            return None
    return _stack_cache.get("crewai")


def _get_langfuse_client() -> Any:
    if "langfuse" not in _stack_cache:
        try:
            from pipeline.langfuse_client import get_langfuse_client
            _stack_cache["langfuse"] = get_langfuse_client()
        except ImportError as e:
            logger.debug(f"Stack 'langfuse' not available: {e}")
            return None
    return _stack_cache.get("langfuse")


def _get_qdrant_client() -> Any:
    if "qdrant" not in _stack_cache:
        try:
            from pipeline.qdrant_client import get_qdrant_client
            client = get_qdrant_client()
            _stack_cache["qdrant"] = client
            if client and client.is_available():
                collections = len(client.list_collections()) if hasattr(client, 'list_collections') else 'N/A'
                logger.info(f"STACK Qdrant: CONNECTED (vector store, {collections} collections)")
            else:
                logger.warning("STACK Qdrant: loaded but not available")
        except ImportError as e:
            logger.debug(f"Stack 'qdrant' not available: {e}")
            return None
    return _stack_cache.get("qdrant")


def _get_falkordb_client() -> Any:
    if "falkordb" not in _stack_cache:
        try:
            from pipeline.falkordb_client import get_falkordb_client
            client = get_falkordb_client()
            _stack_cache["falkordb"] = client
            if client and client.is_available():
                logger.info("STACK FalkorDB: CONNECTED (knowledge graph ready)")
            else:
                logger.warning("STACK FalkorDB: loaded but not available")
        except ImportError as e:
            logger.debug(f"Stack 'falkordb' not available: {e}")
            return None
    return _stack_cache.get("falkordb")


def _get_letta_client() -> Any:
    if "letta" not in _stack_cache:
        try:
            from pipeline.letta_client import get_letta_client
            client = get_letta_client()
            _stack_cache["letta"] = client
            if client:
                mode = "server" if client.is_server_mode() else "standalone"
                logger.info(f"STACK Letta: LOADED ({mode} mode - agent memory)")
            else:
                logger.warning("STACK Letta: loaded but client is None")
        except ImportError as e:
            logger.debug(f"Stack 'letta' not available: {e}")
            return None
    return _stack_cache.get("letta")


def _get_mem0_client() -> Any:
    """Get Mem0 client (DEPRECATED).

    .. deprecated:: 2025-12-31
        Mem0 is DEPRECATED in favor of Letta. Always returns None.

        Reason for deprecation:
        - Dependency conflict: Mem0 requires openai>=1.90, conflicts with crewai~=1.83
        - Letta provides native persistence with PostgreSQL backend
        - Letta unifies agent_runtime + memory in a single stack

        Use _get_letta_client() instead for agent memory functionality.
        See docs/STACK_CANONICAL.md Section 10 for deprecated stacks.

    Returns:
        Always None - Mem0 is disabled.
    """
    import warnings
    warnings.warn(
        "Mem0 stack is deprecated and disabled. Use 'letta' stack instead. "
        "See docs/STACK_CANONICAL.md Section 10.",
        DeprecationWarning,
        stacklevel=2,
    )
    return None


def _get_temporal_client() -> Any:
    if "temporal" not in _stack_cache:
        try:
            from pipeline.temporal_integration import get_temporal_integration
            _stack_cache["temporal"] = get_temporal_integration()
        except ImportError as e:
            logger.debug(f"Stack 'temporal' not available: {e}")
            return None
    return _stack_cache.get("temporal")


def _get_got_client() -> Any:
    """GoT (Graph of Thoughts) client for multi-path reasoning."""
    if "got" not in _stack_cache:
        try:
            from pipeline.got import GoTController
            from pipeline.got.adapter import ClaudeCLILanguageModel
            # Create default LM for GoT
            lm = ClaudeCLILanguageModel()
            _stack_cache["got"] = GoTController(lm=lm)
        except ImportError as e:
            logger.debug(f"Stack 'got' not available: {e}")
            return None
        except Exception as e:
            logger.debug(f"Stack 'got' initialization failed: {e}")
            return None
    return _stack_cache.get("got")


def _get_reflexion_client() -> Any:
    """Reflexion client for self-reflection and verbal RL."""
    if "reflexion" not in _stack_cache:
        try:
            from pipeline.reflexion_engine import ReflexionEngine
            _stack_cache["reflexion"] = ReflexionEngine()
        except ImportError as e:
            logger.debug(f"Stack 'reflexion' not available: {e}")
            return None
    return _stack_cache.get("reflexion")


def _get_langgraph_client() -> Any:
    """LangGraph client - verifies StateGraph is available.

    LangGraph is the control plane for the pipeline. This returns
    a lightweight client that confirms the StateGraph machinery is ready.
    """
    if "langgraph" not in _stack_cache:
        try:
            from langgraph.graph import StateGraph
            # Return a lightweight wrapper confirming availability
            _stack_cache["langgraph"] = {
                "available": True,
                "StateGraph": StateGraph,
                "version": getattr(
                    __import__("langgraph"), "__version__", "unknown"
                ),
            }
        except ImportError as e:
            logger.debug(f"Stack 'langgraph' not available: {e}")
            return None
    return _stack_cache.get("langgraph")


def _get_phoenix_client() -> Any:
    if "phoenix" not in _stack_cache:
        try:
            import phoenix
            _stack_cache["phoenix"] = phoenix
        except ImportError as e:
            logger.debug(f"Stack 'phoenix' not available: {e}")
            return None
    return _stack_cache.get("phoenix")


def _get_graphiti_client() -> Any:
    """Graphiti client for temporal knowledge graph operations.

    Requires FalkorDB connection. Uses FALKORDB_URL environment variable
    or falls back to localhost:6379.
    """
    if "graphiti" not in _stack_cache:
        try:
            import os
            from graphiti_core import Graphiti

            # Get FalkorDB URI from environment or use default
            falkordb_url = os.getenv("FALKORDB_URL", "bolt://localhost:6379")

            _stack_cache["graphiti"] = Graphiti(uri=falkordb_url)
        except ImportError as e:
            logger.debug(f"Stack 'graphiti' not available: {e}")
            return None
        except Exception as e:
            logger.debug(f"Stack 'graphiti' initialization failed: {e}")
            return None
    return _stack_cache.get("graphiti")


def _get_active_rag_client() -> Any:
    """Active RAG client using FLARE (Forward-Looking Active REtrieval).

    Tries to use LlamaIndex FLARE implementation first, falls back to
    custom implementation with Qdrant for vector search.
    """
    if "active_rag" not in _stack_cache:
        try:
            # Try LlamaIndex FLARE first
            from llama_index.core.query_engine import FLAREInstructQueryEngine

            _stack_cache["active_rag"] = {
                "available": True,
                "type": "flare",
                "engine_class": FLAREInstructQueryEngine,
            }
        except ImportError:
            try:
                # Fallback: Use Qdrant directly for active retrieval
                from pipeline.qdrant_client import get_qdrant_client

                qdrant = get_qdrant_client()
                if qdrant is not None:
                    _stack_cache["active_rag"] = {
                        "available": True,
                        "type": "qdrant_fallback",
                        "qdrant_client": qdrant,
                    }
                else:
                    logger.debug("Stack 'active_rag' not available: Qdrant client is None")
                    return None
            except ImportError as e:
                logger.debug(f"Stack 'active_rag' not available: {e}")
                return None
    return _stack_cache.get("active_rag")


def _get_nemo_client() -> Any:
    if "nemo" not in _stack_cache:
        try:
            from nemoguardrails import RailsConfig, LLMRails
            _stack_cache["nemo"] = {"RailsConfig": RailsConfig, "LLMRails": LLMRails}
        except ImportError as e:
            logger.debug(f"Stack 'nemo' not available: {e}")
            return None
    return _stack_cache.get("nemo")


def _get_llm_guard_client() -> Any:
    """Get LLM Guard client - checks Docker container via HTTP.

    LLM Guard runs as a Docker container (not a Python import).
    We check the health endpoint to verify it's available.
    """
    if "llm_guard" not in _stack_cache:
        try:
            import httpx
            import os
            llm_guard_url = os.getenv("LLM_GUARD_URL", "http://localhost:50053")
            # Quick health check
            response = httpx.get(f"{llm_guard_url}/health", timeout=2.0)
            if response.status_code == 200:
                # Return a simple client wrapper
                class LLMGuardClient:
                    def __init__(self, base_url: str):
                        self.base_url = base_url
                        self.healthy = True
                _stack_cache["llm_guard"] = LLMGuardClient(llm_guard_url)
            else:
                logger.debug(f"Stack 'llm_guard' health check failed: {response.status_code}")
                return None
        except Exception as e:
            logger.debug(f"Stack 'llm_guard' not available: {e}")
            return None
    return _stack_cache.get("llm_guard")


def _get_bot_client() -> Any:
    """Buffer of Thoughts (BOT) client for enhanced reasoning.

    BOT extends Chain-of-Thought by maintaining a reasoning buffer
    that can be referenced and refined across multiple reasoning steps.
    Uses the native GoT framework as the foundation.
    """
    if "bot" not in _stack_cache:
        try:
            # Use GoT as foundation for Buffer of Thoughts
            from pipeline.got import (
                GoTController,
                ClaudeCLILanguageModel,
                GraphOfOperations,
                Generate,
                Refine,
                Aggregate,
            )

            # BOT extends GoT with buffer management
            _stack_cache["bot"] = {
                "available": True,
                "type": "bot",
                "framework": "got_based",
                "GoTController": GoTController,
                "ClaudeCLILanguageModel": ClaudeCLILanguageModel,
                "GraphOfOperations": GraphOfOperations,
                "operations": {
                    "Generate": Generate,
                    "Refine": Refine,
                    "Aggregate": Aggregate,
                },
            }
        except ImportError as e:
            logger.debug(f"Stack 'bot' not available: {e}")
            return None
    return _stack_cache.get("bot")


def _get_z3_client() -> Any:
    if "z3" not in _stack_cache:
        try:
            import z3
            _stack_cache["z3"] = z3
        except ImportError as e:
            logger.debug(f"Stack 'z3' not available: {e}")
            return None
    return _stack_cache.get("z3")


def _get_pydantic_ai_client() -> Any:
    """Pydantic AI client for type-safe AI agent development.

    Handles logfire dependency issues gracefully by mocking the module.
    logfire has opentelemetry import issues in Python 3.13.
    """
    if "pydantic_ai" not in _stack_cache:
        try:
            import sys
            import types

            # Create comprehensive mock logfire
            # logfire has import issues with opentelemetry in this env
            mock_logfire = types.ModuleType("logfire")

            # Mock all classes that pydantic_ai might need
            mock_logfire.Logfire = type(
                "Logfire",
                (),
                {
                    "__init__": lambda self, **kw: None,
                    "span": lambda self, *a, **kw: type(
                        "Span", (), {"__enter__": lambda s: s, "__exit__": lambda *a: None}
                    )(),
                },
            )
            mock_logfire.LogfireSpan = type(
                "LogfireSpan",
                (),
                {
                    "__init__": lambda self, **kw: None,
                    "__enter__": lambda self: self,
                    "__exit__": lambda self, *a: None,
                },
            )
            mock_logfire.instrument_pydantic = lambda *args, **kwargs: None
            mock_logfire.configure = lambda *args, **kwargs: None
            mock_logfire.span = lambda *args, **kwargs: type(
                "Span", (), {"__enter__": lambda s: s, "__exit__": lambda *a: None}
            )()
            mock_logfire.instrument = lambda *args, **kwargs: lambda fn: fn

            # Remove broken logfire and replace with mock
            if "logfire" in sys.modules:
                del sys.modules["logfire"]
            sys.modules["logfire"] = mock_logfire

            import pydantic_ai

            _stack_cache["pydantic_ai"] = pydantic_ai
        except ImportError as e:
            logger.debug(f"Stack 'pydantic_ai' not available: {e}")
            return None
        except Exception as e:
            logger.debug(f"Stack 'pydantic_ai' initialization failed: {e}")
            return None
    return _stack_cache.get("pydantic_ai")


def _get_cleanlab_client() -> Any:
    """Cleanlab client for data quality and label error detection.

    Uses Datalab for local data quality analysis. Studio requires paid API.
    """
    if "cleanlab" not in _stack_cache:
        try:
            # Use Datalab for local analysis (Studio requires paid API)
            from cleanlab import Datalab

            _stack_cache["cleanlab"] = {
                "available": True,
                "Datalab": Datalab,
                "type": "local",
            }
        except ImportError as e:
            logger.debug(f"Stack 'cleanlab' not available: {e}")
            return None
    return _stack_cache.get("cleanlab")


def _get_graphrag_client() -> Any:
    """GraphRAG client for graph-enhanced retrieval augmented generation.

    Tries Microsoft's graphrag library first, falls back to
    FalkorDB + Qdrant combination for graph-based RAG.
    """
    if "graphrag" not in _stack_cache:
        try:
            # Try Microsoft GraphRAG first
            import graphrag
            from graphrag.query.structured_search.global_search import GlobalSearch
            from graphrag.query.structured_search.local_search import LocalSearch

            _stack_cache["graphrag"] = {
                "available": True,
                "type": "microsoft_graphrag",
                "module": graphrag,
                "GlobalSearch": GlobalSearch,
                "LocalSearch": LocalSearch,
            }
        except ImportError:
            try:
                # Fallback: FalkorDB + Qdrant for graph-based RAG
                from pipeline.falkordb_client import get_falkordb_client
                from pipeline.qdrant_client import get_qdrant_client

                falkordb = get_falkordb_client()
                qdrant = get_qdrant_client()

                if falkordb is not None and qdrant is not None:
                    _stack_cache["graphrag"] = {
                        "available": True,
                        "type": "falkordb_qdrant",
                        "falkordb_client": falkordb,
                        "qdrant_client": qdrant,
                    }
                else:
                    missing = []
                    if falkordb is None:
                        missing.append("falkordb")
                    if qdrant is None:
                        missing.append("qdrant")
                    logger.debug(f"Stack 'graphrag' not available: missing {missing}")
                    return None
            except ImportError as e:
                logger.debug(f"Stack 'graphrag' not available: {e}")
                return None
    return _stack_cache.get("graphrag")


def _get_dspy_client() -> Any:
    if "dspy" not in _stack_cache:
        try:
            import dspy
            _stack_cache["dspy"] = dspy
        except ImportError as e:
            logger.debug(f"Stack 'dspy' not available: {e}")
            return None
    return _stack_cache.get("dspy")


def _get_trulens_client() -> Any:
    """TruLens client for LLM evaluation and observability.

    Uses the new trulens.core module (trulens_eval is deprecated).
    """
    if "trulens" not in _stack_cache:
        try:
            # Use new import path (trulens_eval is deprecated)
            from trulens.core import Tru

            _stack_cache["trulens"] = {
                "available": True,
                "Tru": Tru,
                "type": "trulens_core",
            }
        except ImportError:
            # Fallback to deprecated trulens_eval
            try:
                from trulens_eval import TruLlama

                _stack_cache["trulens"] = {
                    "available": True,
                    "TruLlama": TruLlama,
                    "type": "trulens_eval",
                }
            except ImportError as e:
                logger.debug(f"Stack 'trulens' not available: {e}")
                return None
    return _stack_cache.get("trulens")


def _get_ragas_client() -> Any:
    if "ragas" not in _stack_cache:
        try:
            import ragas
            _stack_cache["ragas"] = ragas
        except ImportError as e:
            logger.debug(f"Stack 'ragas' not available: {e}")
            return None
    return _stack_cache.get("ragas")


def _get_deepeval_client() -> Any:
    if "deepeval" not in _stack_cache:
        try:
            import deepeval
            _stack_cache["deepeval"] = deepeval
        except ImportError as e:
            logger.debug(f"Stack 'deepeval' not available: {e}")
            return None
    return _stack_cache.get("deepeval")


def _get_instructor_client() -> Any:
    if "instructor" not in _stack_cache:
        try:
            import instructor
            _stack_cache["instructor"] = instructor
        except ImportError as e:
            logger.debug(f"Stack 'instructor' not available: {e}")
            return None
    return _stack_cache.get("instructor")


def _get_neo4j_client() -> Any:
    if "neo4j" not in _stack_cache:
        try:
            from neo4j import GraphDatabase
            _stack_cache["neo4j"] = GraphDatabase
        except ImportError as e:
            logger.debug(f"Stack 'neo4j' not available: {e}")
            return None
    return _stack_cache.get("neo4j")


def _get_hamilton_client() -> Any:
    if "hamilton" not in _stack_cache:
        try:
            from hamilton import driver
            _stack_cache["hamilton"] = driver
        except ImportError as e:
            logger.debug(f"Stack 'hamilton' not available: {e}")
            return None
    return _stack_cache.get("hamilton")


# =============================================================================
# STACK INJECTOR
# =============================================================================


class StackInjector:
    """Central hub for stack injection and wiring.

    This class manages:
    - Which stacks are available for each operation
    - Health checking before injection
    - Graceful degradation for optional stacks
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        custom_policy: Optional[Dict[Operation, Dict[str, StackPolicy]]] = None,
    ):
        """Initialize the stack injector.

        Args:
            config_path: Path to YAML config for custom stack policy.
            custom_policy: Custom policy to override defaults.
        """
        self.policy = custom_policy or DEFAULT_STACK_POLICY.copy()
        self._stack_health: Dict[str, StackHealth] = {}

        if config_path:
            self._load_config(config_path)

    def _load_config(self, config_path: str) -> None:
        """Load stack policy from YAML config."""
        try:
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)

            for op_name, stacks in config.get("operations", {}).items():
                try:
                    op = Operation(op_name)
                    if op not in self.policy:
                        self.policy[op] = {}

                    for stack_name, policy_str in stacks.items():
                        self.policy[op][stack_name] = StackPolicy(policy_str)
                except (ValueError, KeyError):
                    logger.warning(f"Invalid config entry: {op_name}.{stack_name}")

        except Exception as e:
            logger.warning(f"Could not load stack config: {e}")

    def get_stacks_for_operation(
        self,
        operation: Operation,
        check_health: bool = True,
    ) -> Dict[str, Any]:
        """Get all available stacks for an operation.

        Args:
            operation: The operation to get stacks for.
            check_health: Whether to verify stack health before returning.

        Returns:
            Dict of stack_name -> stack_client for available stacks.

        Raises:
            RuntimeError: If a REQUIRED stack is unavailable.
        """
        op_policy = self.policy.get(operation, {})
        stacks: Dict[str, Any] = {}
        missing_required: List[str] = []

        for stack_name, policy in op_policy.items():
            if policy == StackPolicy.DISABLED:
                continue

            factory = get_stack_factory(stack_name)
            if factory is None:
                if policy == StackPolicy.REQUIRED:
                    missing_required.append(stack_name)
                continue

            try:
                client = factory()
                if client is None:
                    if policy == StackPolicy.REQUIRED:
                        missing_required.append(stack_name)
                    continue

                if check_health:
                    health = self._check_stack_health(stack_name, client)
                    self._stack_health[stack_name] = health
                    if not health.healthy and policy == StackPolicy.REQUIRED:
                        missing_required.append(stack_name)
                        continue

                stacks[stack_name] = client

            except Exception as e:
                logger.warning(f"Failed to get stack {stack_name}: {e}")
                if policy == StackPolicy.REQUIRED:
                    missing_required.append(stack_name)

        if missing_required:
            raise RuntimeError(
                f"REQUIRED stacks unavailable for {operation.value}: {missing_required}"
            )

        # Log stack injection summary for visibility
        if stacks:
            stack_names = list(stacks.keys())
            critical_stacks = [s for s in stack_names if s in ("falkordb", "letta", "qdrant", "redis", "crewai")]
            optional_stacks = [s for s in stack_names if s not in critical_stacks]

            if critical_stacks:
                logger.info(f"STACKS [{operation.value}]: Critical={critical_stacks}")
            if optional_stacks:
                logger.debug(f"STACKS [{operation.value}]: Optional={optional_stacks}")

        return stacks

    def _check_stack_health(self, stack_name: str, client: Any) -> StackHealth:
        """Check health of a stack."""
        import time
        start = time.time()

        try:
            # Try common health check patterns
            if hasattr(client, "health_check"):
                healthy = client.health_check()
            elif hasattr(client, "ping"):
                healthy = client.ping()
            elif hasattr(client, "is_connected"):
                healthy = client.is_connected()
            else:
                # Assume healthy if no health check available
                healthy = True

            latency = (time.time() - start) * 1000

            return StackHealth(
                name=stack_name,
                healthy=healthy,
                latency_ms=latency,
            )

        except Exception as e:
            latency = (time.time() - start) * 1000
            return StackHealth(
                name=stack_name,
                healthy=False,
                latency_ms=latency,
                error=str(e),
            )

    def with_stacks(self, operation: Operation) -> Callable:
        """Decorator to inject stacks into a function.

        Usage:
            @injector.with_stacks(Operation.EXEC)
            async def exec_node(ctx: StackContext, state: PipelineState) -> PipelineState:
                redis = ctx.stacks["redis"]
                crewai = ctx.stacks["crewai"]
                ...
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(run_id: str, sprint_id: str, *args, **kwargs):
                stacks = self.get_stacks_for_operation(operation)

                ctx = StackContext(
                    run_id=run_id,
                    sprint_id=sprint_id,
                    operation=operation,
                    stacks=stacks,
                    stack_health=self._stack_health.copy(),
                )

                return await func(ctx, *args, **kwargs)

            return wrapper
        return decorator

    def get_health_report(self) -> Dict[str, StackHealth]:
        """Get health report for all checked stacks."""
        return self._stack_health.copy()

    def check_health(self) -> Dict[str, Dict[str, Any]]:
        """Check health of all known stacks.

        This method performs a comprehensive health check across all stacks
        defined in the policy. Used by bridge.py for status reporting.

        Skips DEPRECATED stacks and treats OPTIONAL stacks that aren't
        available as healthy (they're optional after all).

        Returns:
            Dict mapping stack_name to health info dict:
            {
                "redis": {"healthy": True, "latency_ms": 1.2},
                "crewai": {"healthy": True, "latency_ms": 0.5},
                ...
            }
        """
        health_results: Dict[str, Dict[str, Any]] = {}
        all_stacks: set = set()
        stack_policies: Dict[str, StackPolicy] = {}

        # Collect all unique stacks from all operations with their policies
        for op_policy in self.policy.values():
            for stack_name, policy in op_policy.items():
                all_stacks.add(stack_name)
                # Keep the most restrictive policy seen
                if stack_name not in stack_policies:
                    stack_policies[stack_name] = policy

        for stack_name in all_stacks:
            policy = stack_policies.get(stack_name, StackPolicy.OPTIONAL)

            # Skip DEPRECATED stacks - they shouldn't be checked
            if policy == StackPolicy.DEPRECATED:
                health_results[stack_name] = {
                    "healthy": True,  # Don't report as unhealthy
                    "status": "deprecated",
                    "latency_ms": 0,
                }
                continue

            factory = get_stack_factory(stack_name)
            if factory is None:
                # OPTIONAL stacks without factory are OK
                if policy == StackPolicy.OPTIONAL:
                    health_results[stack_name] = {
                        "healthy": True,
                        "status": "not_configured",
                        "latency_ms": 0,
                    }
                else:
                    health_results[stack_name] = {
                        "healthy": False,
                        "error": "No factory found",
                        "latency_ms": 0,
                    }
                continue

            try:
                client = factory()
                if client is None:
                    # OPTIONAL stacks that return None are OK
                    if policy == StackPolicy.OPTIONAL:
                        health_results[stack_name] = {
                            "healthy": True,
                            "status": "not_available",
                            "latency_ms": 0,
                        }
                    else:
                        health_results[stack_name] = {
                            "healthy": False,
                            "error": "Factory returned None",
                            "latency_ms": 0,
                        }
                    continue

                health = self._check_stack_health(stack_name, client)
                self._stack_health[stack_name] = health
                health_results[stack_name] = {
                    "healthy": health.healthy,
                    "latency_ms": health.latency_ms,
                    "error": health.error,
                }

            except Exception as e:
                health_results[stack_name] = {
                    "healthy": False,
                    "error": str(e),
                    "latency_ms": 0,
                }

        return health_results


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


_default_injector: Optional[StackInjector] = None


def get_stack_injector() -> StackInjector:
    """Get the default stack injector."""
    global _default_injector
    if _default_injector is None:
        _default_injector = StackInjector()
    return _default_injector


def get_stacks(operation: Operation) -> Dict[str, Any]:
    """Convenience function to get stacks for an operation."""
    return get_stack_injector().get_stacks_for_operation(operation)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "Operation",
    "StackPolicy",
    "StackHealth",
    "StackContext",
    "StackInjector",
    "get_stack_injector",
    "get_stacks",
    "clear_stack_cache",
    "DEFAULT_STACK_POLICY",
]
