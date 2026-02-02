"""Pipeline V2 Agents Module.

This module contains AI agent coordination and shared memory for CrewAI.

Components:
- crewai_shared_memory: Shared memory for CrewAI agents (P2-012)
- crewai_coordination: Agent coordination and delegation

Note: Agent personas are loaded from runtime_cards/*.yml via runtime_card_loader.py
      and used by crewai_hierarchy.py. See configs/pipeline/runtime_cards/

Usage:
    from pipeline.agents import (
        # CrewAI coordination
        CrewCoordinator,
        TaskDelegator,
        COORDINATION_AVAILABLE,
        # CrewAI shared memory
        SharedMemoryManager,
        get_shared_memory_config,
    )

    # Configure CrewAI with shared memory
    memory_config = get_shared_memory_config(crew_id="qa_crew")
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Import pydantic agents
try:
    from pipeline.agents.pydantic_agents import (
        # Agents
        claim_verifier_agent,
        source_evaluator_agent,
        evidence_gatherer_agent,
        # Contexts
        VerificationContext,
        SourceContext,
        EvidenceContext,
        # Results
        ClaimVerificationResult,
        SourceEvaluationResult,
        EvidenceGatheringResult,
        # Agent Communication
        AgentMessage,
        AgentResponse,
        AgentCapability,
        MultiAgentContext,
        AgentOrchestrator,
        get_agent_orchestrator,
        # Tools
        search_knowledge_graph,
        check_source_credibility,
        find_related_claims,
        # Runner functions
        run_claim_verification,
        run_source_evaluation,
        run_evidence_gathering,
        coordinate_claim_verification,
        # Constants
        PYDANTIC_AI_AVAILABLE,
    )
    _PYDANTIC_AI_IMPORT_SUCCESS = True
except ImportError as e:
    logger.debug(f"Pydantic-AI agents not available: {e}")
    _PYDANTIC_AI_IMPORT_SUCCESS = False
    PYDANTIC_AI_AVAILABLE = False

__all__ = [
    "PYDANTIC_AI_AVAILABLE",
]

if _PYDANTIC_AI_IMPORT_SUCCESS:
    __all__.extend([
        # Agents
        "claim_verifier_agent",
        "source_evaluator_agent",
        "evidence_gatherer_agent",
        # Contexts
        "VerificationContext",
        "SourceContext",
        "EvidenceContext",
        # Results
        "ClaimVerificationResult",
        "SourceEvaluationResult",
        "EvidenceGatheringResult",
        # Agent Communication
        "AgentMessage",
        "AgentResponse",
        "AgentCapability",
        "MultiAgentContext",
        "AgentOrchestrator",
        "get_agent_orchestrator",
        # Tools
        "search_knowledge_graph",
        "check_source_credibility",
        "find_related_claims",
        # Runners
        "run_claim_verification",
        "run_source_evaluation",
        "run_evidence_gathering",
        "coordinate_claim_verification",
    ])

# P2-012: Import CrewAI shared memory
try:
    from pipeline.agents.crewai_shared_memory import (
        QDRANT_AVAILABLE as CREWAI_QDRANT_AVAILABLE,
        REDIS_AVAILABLE as CREWAI_REDIS_AVAILABLE,
        MemoryType,
        MemoryScope,
        MemoryEntry,
        EntityMemory,
        MemorySearchResult,
        SharedMemoryManager,
        get_shared_memory_config,
        create_memory_enabled_crew_config,
        get_shared_memory_manager,
    )
    _CREWAI_MEMORY_IMPORT_SUCCESS = True
    CREWAI_SHARED_MEMORY_AVAILABLE = True
except ImportError as e:
    logger.debug(f"CrewAI shared memory not available: {e}")
    _CREWAI_MEMORY_IMPORT_SUCCESS = False
    CREWAI_SHARED_MEMORY_AVAILABLE = False

__all__.append("CREWAI_SHARED_MEMORY_AVAILABLE")

if _CREWAI_MEMORY_IMPORT_SUCCESS:
    __all__.extend([
        "CREWAI_QDRANT_AVAILABLE",
        "CREWAI_REDIS_AVAILABLE",
        "MemoryType",
        "MemoryScope",
        "MemoryEntry",
        "EntityMemory",
        "MemorySearchResult",
        "SharedMemoryManager",
        "get_shared_memory_config",
        "create_memory_enabled_crew_config",
        "get_shared_memory_manager",
    ])

# P2-B8: Import CrewAI coordination module
try:
    from pipeline.agents.crewai_coordination import (
        # Classes
        CrewCoordinator,
        TaskDelegator,
        SharedKnowledgeBase,
        # Result types
        AgentCapabilities,
        TaskDefinition,
        DelegationResult,
        TaskExecutionResult,
        CoordinationMetrics,
        CrewResult,
        # Enums
        AgentRole,
        TaskPriority,
        TaskStatus,
        DelegationStrategy,
        # Functions
        get_crew_coordinator,
        create_coordinated_crew,
        delegate_task,
        get_coordination_metrics,
        # Constants
        COORDINATION_AVAILABLE,
        CREWAI_AVAILABLE,
    )
    _CREWAI_COORDINATION_IMPORT_SUCCESS = True
except ImportError as e:
    logger.debug(f"CrewAI coordination not available: {e}")
    _CREWAI_COORDINATION_IMPORT_SUCCESS = False
    COORDINATION_AVAILABLE = False

__all__.append("COORDINATION_AVAILABLE")

if _CREWAI_COORDINATION_IMPORT_SUCCESS:
    __all__.extend([
        # Classes
        "CrewCoordinator",
        "TaskDelegator",
        "SharedKnowledgeBase",
        # Result types
        "AgentCapabilities",
        "TaskDefinition",
        "DelegationResult",
        "TaskExecutionResult",
        "CoordinationMetrics",
        "CrewResult",
        # Enums
        "AgentRole",
        "TaskPriority",
        "TaskStatus",
        "DelegationStrategy",
        # Functions
        "get_crew_coordinator",
        "create_coordinated_crew",
        "delegate_task",
        "get_coordination_metrics",
        "CREWAI_AVAILABLE",
    ])

# =============================================================================
# LETTA INTEGRATION (F1-F3: 2026-02-01)
# =============================================================================
# Export Letta integration for agents-memory E2E tests
try:
    from pipeline.letta_integration import (
        LettaMemoryBridge,
        LettaCriticalError,
        get_memory_bridge,
        LETTA_MAX_RETRIES,
    )
    LETTA_INTEGRATION_AVAILABLE = True
    _LETTA_IMPORT_SUCCESS = True
except ImportError as e:
    logger.debug(f"Letta integration not available: {e}")
    LETTA_INTEGRATION_AVAILABLE = False
    _LETTA_IMPORT_SUCCESS = False

__all__.append("LETTA_INTEGRATION_AVAILABLE")

if _LETTA_IMPORT_SUCCESS:
    __all__.extend([
        "LettaMemoryBridge",
        "LettaCriticalError",
        "get_memory_bridge",
        "LETTA_MAX_RETRIES",
    ])

