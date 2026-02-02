"""CrewAI Hierarchy Integration for Pipeline Autonomo v2.0.

This module provides mapping between the pipeline's hierarchy.py and
CrewAI's agent/crew structure. It enables automatic crew creation
based on the organizational hierarchy.

Key Features:
    - Map hierarchy.py agents to CrewAI agents
    - Generate crews based on squad structure
    - Maintain hierarchy rules in CrewAI execution
    - Support for manager/worker patterns

Architecture:
    hierarchy.py (Source of Truth)
        ↓
    crewai_hierarchy.py (Mapping Layer)
        ↓
    CrewAI Crew (Execution Layer)

Author: Pipeline Autonomo Team
Version: 2.0.0 (2025-12-29)
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from pipeline.langgraph.stack_injection import StackContext

logger = logging.getLogger(__name__)


# =============================================================================
# P1-6.2 FIX: Degraded Features Tracker for Observability (2026-02-01)
# =============================================================================
# Tracks which features failed to import at module load time.
# This list is populated by import handlers and exposed via get_degraded_features().
# AUDIT REF: P1_ROOT_CAUSE_ANALYSIS - "Degradação invisível"
_degraded_features: list[str] = []
_degraded_features_lock: threading.Lock = threading.Lock()


def _register_degraded_feature(feature: str, reason: str) -> None:
    """Register a feature as degraded (failed to import).

    P1-6.2 FIX: Tracks degraded features for observability.

    Args:
        feature: Name of the feature that failed.
        reason: Human-readable reason for the failure.
    """
    with _degraded_features_lock:
        if feature not in _degraded_features:
            _degraded_features.append(feature)
            logger.warning(f"P1-6: Feature '{feature}' degraded: {reason}")


def get_degraded_features() -> list[str]:
    """Get list of degraded features.

    P1-6.2 FIX: Returns list of features that failed to import.
    Use this for observability and debugging.

    Returns:
        List of degraded feature names.
    """
    with _degraded_features_lock:
        return _degraded_features.copy()


def clear_degraded_features() -> None:
    """Clear the degraded features list (for testing)."""
    with _degraded_features_lock:
        _degraded_features.clear()


# =============================================================================
# OPT-11-001: Persona Cache for O(1) persona lookups
# =============================================================================
_persona_cache: dict[tuple[str, bool], dict[str, str]] = {}
_persona_cache_lock: threading.Lock = threading.Lock()


def _get_cached_persona(agent_id: str, include_memory: bool) -> dict[str, str] | None:
    """Get persona from cache if available.

    OPT-11-001: Cache personas to avoid repeated YAML I/O.

    Returns:
        Cached persona dict (copy), or None if not cached.
    """
    cache_key = (agent_id.lower(), include_memory)
    with _persona_cache_lock:
        if cache_key in _persona_cache:
            return _persona_cache[cache_key].copy()  # Return copy to prevent mutation
    return None


def _cache_persona(agent_id: str, include_memory: bool, persona: dict[str, str]) -> None:
    """Store persona in cache.

    OPT-11-001: Cache personas to avoid repeated YAML I/O.
    """
    cache_key = (agent_id.lower(), include_memory)
    with _persona_cache_lock:
        _persona_cache[cache_key] = persona.copy()  # Store copy


def clear_persona_cache() -> None:
    """Clear the persona cache.

    OPT-11-001: Call at end of sprint or when cerebro files change.
    """
    global _persona_cache
    with _persona_cache_lock:
        _persona_cache.clear()


# =============================================================================
# P0-02: PARALLEL PERSONA LOADING WITH SEMAPHORE
# =============================================================================

import asyncio

MAX_CONCURRENT_PERSONA_LOADS = 10  # Limit to prevent Redis connection pool exhaustion


async def get_personas_parallel(
    agent_ids: list[str],
    include_memory: bool = True,
) -> dict[str, dict[str, str]]:
    """Load multiple personas in parallel with concurrency limit.

    P0-02 FIX: Uses semaphore to prevent Redis connection pool exhaustion
    while loading multiple personas concurrently.

    Args:
        agent_ids: List of agent IDs to load personas for.
        include_memory: Whether to include memory context.

    Returns:
        Dict mapping agent_id to persona dict.

    Example:
        >>> personas = await get_personas_parallel(
        ...     ["spec_master", "qa_master", "ace_exec"]
        ... )
        >>> print(personas["spec_master"]["role"])
        'Specification Master'
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_PERSONA_LOADS)

    async def load_one(
        agent_id: str,
    ) -> tuple[str, dict[str, str] | None, Exception | None]:
        """Load a single persona with semaphore limiting."""
        async with semaphore:
            try:
                # Run sync get_persona in thread to avoid blocking
                persona = await asyncio.to_thread(
                    get_persona, agent_id, include_memory
                )
                return (agent_id, persona, None)
            except Exception as e:
                logger.warning(f"P0-02: Failed to load persona {agent_id}: {e}")
                return (agent_id, None, e)

    # Launch all tasks concurrently
    tasks = [load_one(aid) for aid in agent_ids]
    results = await asyncio.gather(*tasks)

    # Collect results, use defaults for failures
    personas: dict[str, dict[str, str]] = {}
    failures: list[str] = []

    for agent_id, persona, error in results:
        if error is not None or persona is None:
            failures.append(agent_id)
            # Use default persona for failed loads
            personas[agent_id] = {
                "role": agent_id,
                "goal": f"Perform tasks as {agent_id}",
                "backstory": f"You are {agent_id}.",
            }
        else:
            personas[agent_id] = persona

    if failures:
        logger.warning(f"P0-02: Used default personas for: {failures}")

    logger.debug(
        f"P0-02: Loaded {len(personas)} personas "
        f"({len(agent_ids) - len(failures)} success, {len(failures)} fallback)"
    )

    return personas


def get_personas_parallel_sync(
    agent_ids: list[str],
    include_memory: bool = True,
) -> dict[str, dict[str, str]]:
    """Sync wrapper for parallel persona loading.

    P0-02 FIX: Handles both async and sync calling contexts automatically.

    Args:
        agent_ids: List of agent IDs to load personas for.
        include_memory: Whether to include memory context.

    Returns:
        Dict mapping agent_id to persona dict.

    Example:
        >>> personas = get_personas_parallel_sync(
        ...     ["spec_master", "qa_master", "ace_exec"]
        ... )
        >>> print(personas["qa_master"]["role"])
        'QA Master'
    """
    try:
        loop = asyncio.get_running_loop()
        # Already in async context - run in thread to avoid blocking
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                asyncio.run,
                get_personas_parallel(agent_ids, include_memory)
            )
            return future.result(timeout=120)
    except RuntimeError:
        # No event loop - safe to run directly
        return asyncio.run(get_personas_parallel(agent_ids, include_memory))


from pipeline.crewai_client import (
    AgentDefinition,
    CrewAIClient,
    CrewResult,
    TaskDefinition,
    get_crewai_client,
)
from pipeline.hierarchy import (
    HIERARCHY_MAP,
    get_level,
    get_subordinates,
    get_supervisor,
    validate_agent_exists,
)

import logging

logger = logging.getLogger(__name__)

# GHOST CODE INTEGRATION (2026-01-30): Shared memory for CrewAI agents
# Enables knowledge persistence and sharing across agent executions
_shared_memory_config = None
SHARED_MEMORY_AVAILABLE = False


def get_crewai_shared_memory_config() -> dict | None:
    """Get shared memory configuration for CrewAI crews.

    GHOST CODE INTEGRATION: Uses pipeline.agents.crewai_shared_memory

    Returns:
        Memory config dict for CrewAI, or None if not available.
    """
    global _shared_memory_config, SHARED_MEMORY_AVAILABLE
    if _shared_memory_config is not None:
        return _shared_memory_config

    try:
        from pipeline.agents.crewai_shared_memory import get_shared_memory_config
        _shared_memory_config = get_shared_memory_config()
        SHARED_MEMORY_AVAILABLE = True
        logger.info("CrewAI shared memory configuration loaded")
        return _shared_memory_config
    except ImportError as e:
        # P1-6.1 FIX: Upgrade to warning and register degradation
        _register_degraded_feature("shared_memory", f"CrewAI shared memory not available: {e}")
        SHARED_MEMORY_AVAILABLE = False
        return None


# GHOST CODE INTEGRATION (2026-01-30): Multi-crew coordination
# Enables coordination between multiple crews with task delegation
_crew_coordinator = None
CREW_COORDINATION_AVAILABLE = False


def get_coordinated_crew_executor():
    """Get crew coordinator for multi-crew orchestration.

    GHOST CODE INTEGRATION: Uses pipeline.agents.crewai_coordination

    Returns:
        CrewCoordinator instance, or None if not available.
    """
    global _crew_coordinator, CREW_COORDINATION_AVAILABLE
    if _crew_coordinator is not None:
        return _crew_coordinator

    try:
        from pipeline.agents.crewai_coordination import get_crew_coordinator
        _crew_coordinator = get_crew_coordinator()
        CREW_COORDINATION_AVAILABLE = True
        logger.info("CrewAI coordination loaded (multi-crew support enabled)")
        return _crew_coordinator
    except ImportError as e:
        # P1-6.1 FIX: Upgrade to warning and register degradation
        _register_degraded_feature("crew_coordination", f"Multi-crew coordination not available: {e}")
        CREW_COORDINATION_AVAILABLE = False
        return None


# MCP Tools integration (lazy import to avoid circular dependencies)
_mcp_tools_module = None


def _get_mcp_tools_module():
    """Get mcp_tools module (lazy to avoid circular imports)."""
    global _mcp_tools_module
    if _mcp_tools_module is None:
        try:
            from pipeline import mcp_tools
            _mcp_tools_module = mcp_tools
        except ImportError as e:
            logger.debug(f"IMPORT: Module not available: {e}")
    return _mcp_tools_module

# Lazy import to avoid circular dependencies
_spec_kit_loader = None
_cerebro_yaml_loader = None
_stack_injector = None


def _get_spec_kit_loader():
    """Get SpecKitLoader instance (lazy to avoid circular imports)."""
    global _spec_kit_loader
    if _spec_kit_loader is None:
        try:
            from pipeline.spec_kit_loader import get_spec_kit_loader
            _spec_kit_loader = get_spec_kit_loader()
        except ImportError as e:
            logger.debug(f"IMPORT: Module not available: {e}")
    return _spec_kit_loader


def _get_cerebro_yaml_loader():
    """Get CerebroYAMLLoader instance (lazy to avoid circular imports)."""
    global _cerebro_yaml_loader
    if _cerebro_yaml_loader is None:
        try:
            from pipeline.cerebro_yaml_loader import get_cerebro_yaml_loader
            _cerebro_yaml_loader = get_cerebro_yaml_loader()
        except ImportError as e:
            logger.debug(f"IMPORT: Module not available: {e}")
    return _cerebro_yaml_loader


def _get_stack_injector():
    """Get StackInjector instance (lazy to avoid circular imports)."""
    global _stack_injector
    if _stack_injector is None:
        try:
            from pipeline.stack_injector import get_stack_injector
            _stack_injector = get_stack_injector()
        except ImportError as e:
            logger.debug(f"IMPORT: Module not available: {e}")
    return _stack_injector


# Stack Synergy Manager (lazy import)
_synergy_manager = None


def _get_synergy_manager():
    """Get StackSynergyManager instance (lazy to avoid circular imports)."""
    global _synergy_manager
    if _synergy_manager is None:
        try:
            from pipeline.langgraph.stack_synergy import get_synergy_manager
            _synergy_manager = get_synergy_manager()
        except ImportError as e:
            logger.debug(f"StackSynergyManager not available: {e}")
    return _synergy_manager


def _get_stack_context_for_agent(agent_id: str) -> str:
    """Get stack context section for agent backstory injection.

    This provides agents with information about available stacks,
    their capabilities, and how to use them effectively.

    Args:
        agent_id: The agent identifier.

    Returns:
        Stack context section as markdown string, or empty string if unavailable.
    """
    manager = _get_synergy_manager()
    if manager is None:
        return ""

    try:
        context = manager.get_agent_context(agent_id)
        return context.to_prompt_section()
    except Exception as e:
        logger.debug(f"Could not get stack context for {agent_id}: {e}")
        return ""


# =============================================================================
# Persona Templates (FALLBACK - Cerebros are preferred)
# =============================================================================

# NOTE: These templates are used as fallback when Cerebro docs are not available.
# The Spec Kit Cerebros (docs/Agents/L*/Cerebro *.md) are the source of truth.

PERSONA_TEMPLATES: dict[str, dict[str, str]] = {
    # Level 0 - System
    "pack_driver": {
        "role": "Pipeline Orchestrator",
        "goal": "Orchestrate pipeline execution and coordinate all agents",
        "backstory": """You are the Pipeline Orchestrator, responsible for managing
        the entire pipeline execution flow. You dispatch tasks to agents, manage
        phase transitions, and ensure smooth operation of the pipeline.""",
    },
    "ops_ctrl": {
        "role": "Operations Controller",
        "goal": "Monitor system health and enforce safety protocols",
        "backstory": """You are the Operations Controller, the watchdog of the
        pipeline. You monitor agent health, coordinate high-availability,
        and have veto power for safety-critical decisions.""",
    },
    "run_master": {
        "role": "Pipeline Caretaker",
        "goal": "Maintain pipeline stability and drive self-improvement",
        "backstory": """You are the Run Master, the caretaker of the pipeline.
        You implement self-improvement protocols, fix bugs, and ensure the
        pipeline evolves and improves over time.""",
    },
    # Level 1-2 - Executive
    "presidente": {
        "role": "Strategic Advisor",
        "goal": "Provide big-picture strategic guidance",
        "backstory": """You are the President, the highest executive in the
        business hierarchy. You maintain the big picture, set strategic
        direction, and ensure the CEO stays on track.""",
    },
    "ceo": {
        "role": "Chief Executive Officer",
        "goal": "Execute strategy and coordinate all operations",
        "backstory": """You are the CEO, responsible for executing the strategic
        vision. You coordinate VPs, make critical decisions, and ensure
        alignment between all parts of the organization.""",
    },
    # Level 3 - VPs
    "spec_vp": {
        "role": "VP of Specifications",
        "goal": "Oversee product definition and specification quality",
        "backstory": """You are the VP of Specifications, responsible for all
        product definition work. You ensure specifications are clear, complete,
        and aligned with business objectives.""",
    },
    "exec_vp": {
        "role": "VP of Execution",
        "goal": "Oversee implementation and quality assurance",
        "backstory": """You are the VP of Execution, responsible for all
        implementation work. You coordinate development teams and QA to
        deliver high-quality software.""",
    },
    # NOTE: qa_vp removed - exec_vp now handles both execution AND QA oversight
    # Level 4 - Masters
    "spec_master": {
        "role": "Specification Master",
        "goal": "Manage specification squad and ensure quality specs",
        "backstory": """You are the Spec Master, leading the specification team.
        You coordinate product owners, project managers, and task decomposers
        to create clear, actionable specifications.""",
    },
    "ace_exec": {
        "role": "Execution Master",
        "goal": "Manage execution squads and deliver implementations",
        "backstory": """You are the Ace Exec, leading all execution squads.
        You coordinate squad leads, manage resources, and ensure timely
        delivery of quality implementations.""",
    },
    "qa_master": {
        "role": "QA Master",
        "goal": "Ensure software quality through rigorous testing",
        "backstory": """You are the QA Master, guardian of software quality.
        You lead auditors, reviewers, and hunters to find and prevent bugs,
        ensuring every release meets the highest standards.""",
    },
    # Level 5 - Squad Leads
    "squad_lead_ops": {
        "role": "Operations Squad Lead",
        "goal": "Lead operations team in technical planning",
        "backstory": """You lead the operations squad, coordinating technical
        planners and dependency mappers to ensure smooth execution.""",
    },
    "squad_lead_sec": {
        "role": "Security Squad Lead",
        "goal": "Lead security team in identifying vulnerabilities",
        "backstory": """You lead the security squad, coordinating red team
        agents to identify and address security vulnerabilities.""",
    },
    # Level 6 - Workers
    "auditor": {
        "role": "Code Auditor",
        "goal": "Audit code for quality, standards compliance, and issues",
        "backstory": """You are a Code Auditor, responsible for thorough
        code review. You identify bugs, style issues, and deviations from
        standards with meticulous attention to detail.""",
    },
    "refinador": {
        "role": "Code Refinador",
        "goal": "Refine and improve code quality",
        "backstory": """You are a Code Refinador, focused on improving code
        quality through refactoring, optimization, and cleanup.""",
    },
    "technical_planner": {
        "role": "Technical Planner",
        "goal": "Plan technical implementation strategies",
        "backstory": """You are a Technical Planner, responsible for designing
        implementation approaches and technical architectures.""",
    },
    "product_owner": {
        "role": "Product Owner",
        "goal": "Define product requirements and priorities",
        "backstory": """You are a Product Owner, responsible for defining
        what needs to be built and prioritizing the backlog.""",
    },
    # =========================================================================
    # L5 Workers (GAP-L5-001 FIX: Added 2026-01-31)
    # =========================================================================
    "agente_auditor": {
        "role": "Agente Auditor",
        "goal": "Audit code and specs with meticulous attention to patterns and rules",
        "backstory": """You are the Agente Auditor, specialized in detailed review
        of code and specifications. You identify deviations from standards, missing
        documentation, and potential issues before they become problems. Your
        reviews are thorough and constructive.""",
    },
    "clean_reviewer": {
        "role": "Clean Code Reviewer",
        "goal": "Ensure code is clean, readable, and maintainable following SOLID principles",
        "backstory": """You are the Clean Reviewer, focused on clean code principles
        and software craftsmanship. You review code for readability, proper naming,
        single responsibility, and adherence to SOLID principles. Your goal is
        maintainable, elegant code.""",
    },
    "edge_case_hunter": {
        "role": "Edge Case Hunter",
        "goal": "Identify edge cases, boundary conditions, and uncovered scenarios",
        "backstory": """You are the Edge Case Hunter, specialized in finding failures
        at boundaries and limits. You think about null values, empty collections,
        maximum values, race conditions, and unexpected inputs. No edge case escapes
        your attention.""",
    },
    "gap_hunter": {
        "role": "Gap Hunter",
        "goal": "Identify gaps between specification and implementation",
        "backstory": """You are the Gap Hunter, responsible for finding discrepancies
        between what was specified and what was implemented. You compare RF/INV/EDGE
        requirements against code to ensure nothing was missed or incorrectly
        implemented.""",
    },
    "human_reviewer": {
        "role": "Human Layer Reviewer",
        "goal": "Validate human perspective and usability in deliverables",
        "backstory": """You are the Human Reviewer, specialized in validating
        user experience and human factors. You ensure that systems are usable,
        accessible, and designed with real users in mind. You advocate for the
        human perspective in all technical decisions.""",
    },
    "dependency_mapper": {
        "role": "Dependency Mapper",
        "goal": "Map and analyze dependencies between components and systems",
        "backstory": """You are the Dependency Mapper, expert in understanding
        how components connect and depend on each other. You identify circular
        dependencies, version conflicts, and import issues. Your maps guide
        safe refactoring and integration.""",
    },
    "task_decomposer": {
        "role": "Task Decomposer",
        "goal": "Break down complex requirements into granular, actionable tasks",
        "backstory": """You are the Task Decomposer, specialized in taking large
        requirements and breaking them into small, well-defined tasks. Each task
        you create is atomic, testable, and has clear acceptance criteria. You
        enable parallel execution through proper decomposition.""",
    },
    "red_team_agent": {
        "role": "Red Team Agent",
        "goal": "Identify security vulnerabilities through adversarial thinking",
        "backstory": """You are the Red Team Agent, a security specialist who thinks
        like an attacker. You probe systems for vulnerabilities, test authentication
        bypasses, and attempt injection attacks. Your goal is to find weaknesses
        before real attackers do.""",
    },
    "project_manager": {
        "role": "Project Manager",
        "goal": "Coordinate project activities and track progress",
        "backstory": """You are the Project Manager, responsible for coordinating
        work across teams and tracking deliverables. You manage timelines, identify
        blockers, and ensure clear communication between stakeholders. You keep
        projects on track.""",
    },
    "debt_tracker": {
        "role": "Tech Debt Tracker",
        "goal": "Track, prioritize, and advocate for technical debt resolution",
        "backstory": """You are the Debt Tracker, guardian of code quality over time.
        You identify technical debt, assess its impact, and prioritize which debts
        to pay down first. You balance short-term delivery with long-term
        maintainability.""",
    },
    "human_layer_specialist": {
        "role": "Human Layer Specialist",
        "goal": "Execute and coordinate the 7 Human Layers of verification",
        "backstory": """You are the Human Layer Specialist, expert in human-in-the-loop
        verification. You manage the 7 Human Layers: HL-1 (Critical Override), HL-2
        (Quality Assurance), HL-3 (Ethical Review), HL-4 (Domain Expert), HL-5
        (Stakeholder Validation), HL-6 (Legal Compliance), and HL-7 (Final Approval).""",
    },
    # =========================================================================
    # L6 Specialists (GAP-L6-001 FIX: Added 2026-01-31)
    # =========================================================================
    "oracle_architect": {
        "role": "Oracle Architect",
        "goal": "Design reliable and decentralized oracle systems",
        "backstory": """You are the Oracle Architect, expert in designing oracle
        systems that bridge off-chain data to on-chain smart contracts. You understand
        Chainlink, UMA, and custom oracle patterns. You design for reliability,
        decentralization, and manipulation resistance.""",
    },
    "blockchain_engineer": {
        "role": "Blockchain Engineer",
        "goal": "Implement secure and efficient smart contracts",
        "backstory": """You are the Blockchain Engineer, specialized in Solidity
        and EVM development. You write gas-efficient, secure smart contracts and
        understand DeFi patterns, reentrancy guards, and upgrade mechanisms. You
        follow best practices from OpenZeppelin and Trail of Bits.""",
    },
    "data_engineer": {
        "role": "Data Engineer",
        "goal": "Design and implement robust data pipelines and storage",
        "backstory": """You are the Data Engineer, expert in data infrastructure.
        You design ETL pipelines, optimize database queries, and ensure data quality.
        You work with PostgreSQL, Redis, and vector databases like Qdrant. Your
        pipelines are reliable and scalable.""",
    },
    "ui_designer": {
        "role": "UI Designer",
        "goal": "Create intuitive and beautiful user interfaces",
        "backstory": """You are the UI Designer, focused on creating interfaces
        that are both aesthetically pleasing and functionally effective. You
        understand color theory, typography, layout, and accessibility. You design
        for clarity and delight.""",
    },
    "ux_researcher": {
        "role": "UX Researcher",
        "goal": "Understand user needs through research and testing",
        "backstory": """You are the UX Researcher, dedicated to understanding
        how users think and behave. You conduct user interviews, usability tests,
        and analyze usage data. Your insights guide product decisions and ensure
        we build what users actually need.""",
    },
    "llm_orchestrator": {
        "role": "LLM Orchestrator",
        "goal": "Orchestrate and optimize LLM-based workflows",
        "backstory": """You are the LLM Orchestrator, expert in multi-model
        orchestration. You understand prompt engineering, chain-of-thought,
        RAG patterns, and agent architectures. You optimize for cost, latency,
        and quality across Claude, GPT, and open-source models.""",
    },
    "legal_tech_specialist": {
        "role": "Legal Tech Specialist",
        "goal": "Ensure legal compliance and regulatory adherence",
        "backstory": """You are the Legal Tech Specialist, bridging law and
        technology. You understand GDPR, securities regulations, and smart contract
        law. You ensure systems comply with relevant regulations and can advise
        on legal implications of technical decisions.""",
    },
    "web3_frontend": {
        "role": "Web3 Frontend Developer",
        "goal": "Build responsive and Web3-integrated frontend experiences",
        "backstory": """You are the Web3 Frontend Developer, expert in building
        dApps that connect to blockchain. You work with React, ethers.js, and
        wallet integrations. You create seamless experiences for both crypto-native
        and mainstream users.""",
    },
    # =========================================================================
    # Executive Support Agents (Added 2026-01-31)
    # =========================================================================
    "arbiter": {
        "role": "Conflict Arbiter",
        "goal": "Resolve conflicts between agents with fair and final decisions",
        "backstory": """You are the Arbiter, the final authority on conflicts
        between agents. When agents disagree on approach, priorities, or technical
        decisions, you gather evidence, consider all perspectives, and make binding
        decisions. Your rulings are fair, documented, and final.""",
    },
    "retrospective_master": {
        "role": "Retrospective Master",
        "goal": "Synthesize lessons learned and drive continuous improvement",
        "backstory": """You are the Retrospective Master, guardian of organizational
        learning. You analyze completed sprints, identify patterns in successes
        and failures, and synthesize actionable lessons. Your retrospectives drive
        real improvement in team performance.""",
    },
    "integration_officer": {
        "role": "Integration Officer",
        "goal": "Coordinate dependencies and integrations across squads",
        "backstory": """You are the Integration Officer, responsible for cross-squad
        coordination. You identify integration points, manage shared interfaces,
        and ensure components work together seamlessly. You prevent integration
        surprises and coordinate joint testing.""",
    },
    "external_liaison": {
        "role": "External Liaison",
        "goal": "Manage external vendor relationships and API integrations",
        "backstory": """You are the External Liaison, responsible for managing
        relationships with external services and vendors. You coordinate API
        integrations, manage vendor contracts, and ensure external dependencies
        are reliable and well-documented.""",
    },
    # =========================================================================
    # System Agents (Added 2026-01-31)
    # =========================================================================
    "resource_optimizer": {
        "role": "Resource Optimizer",
        "goal": "Optimize token usage and computational resources",
        "backstory": """You are the Resource Optimizer, responsible for efficient
        resource allocation. You monitor token usage, optimize context windows,
        and ensure computational resources are used effectively. You balance
        quality with cost efficiency.""",
    },
    "system_observer": {
        "role": "System Observer",
        "goal": "Detect anomalies and monitor system health",
        "backstory": """You are the System Observer, the watchful eye on system
        health. You monitor metrics, detect anomalies, and identify issues before
        they become critical. You provide early warning for resource exhaustion,
        performance degradation, and unusual patterns.""",
    },
    "human_approver": {
        "role": "Human Approver",
        "goal": "Provide human oversight for critical decisions",
        "backstory": """You are the Human Approver, representing human judgment
        in the system. You review and approve critical decisions that require
        human oversight, ensuring that automated systems remain aligned with
        human values and intentions.""",
    },
    # =========================================================================
    # Squad Leads (Added 2026-01-31)
    # =========================================================================
    "squad_lead_p1": {
        "role": "Squad Lead P1",
        "goal": "Lead Priority 1 squad in executing critical tasks",
        "backstory": """You lead the P1 Squad, responsible for the highest priority
        implementation tasks. You coordinate workers, manage task distribution,
        and ensure timely delivery of critical features.""",
    },
    "squad_lead_p2": {
        "role": "Squad Lead P2",
        "goal": "Lead Priority 2 squad in executing important tasks",
        "backstory": """You lead the P2 Squad, handling important but not critical
        implementation work. You balance quality with speed and coordinate
        effectively with your team.""",
    },
    "squad_lead_p3": {
        "role": "Squad Lead P3",
        "goal": "Lead Priority 3 squad in executing standard tasks",
        "backstory": """You lead the P3 Squad, managing standard priority work.
        You ensure consistent quality and maintain team morale while delivering
        on schedule.""",
    },
    "squad_lead_p4": {
        "role": "Squad Lead P4",
        "goal": "Lead Priority 4 squad in executing background tasks",
        "backstory": """You lead the P4 Squad, handling lower priority but still
        important work. You optimize for efficiency and handle tasks that improve
        overall system quality.""",
    },
    "squad_lead_p5": {
        "role": "Squad Lead P5",
        "goal": "Lead Priority 5 squad in executing maintenance tasks",
        "backstory": """You lead the P5 Squad, focused on maintenance and
        housekeeping tasks. You ensure technical debt is addressed and the
        codebase remains healthy.""",
    },
    "squad_lead_hlx": {
        "role": "Human Layer Squad Lead",
        "goal": "Lead the Human Layer squad in user-facing initiatives",
        "backstory": """You lead the Human Layer Squad, focused on user experience
        and human factors. You coordinate UI/UX designers, researchers, and
        frontend developers to create exceptional user experiences.""",
    },
    # =========================================================================
    # FASE 3 Additions (2026-01-31) - Missing persona templates
    # =========================================================================
    "sprint_planner": {
        "role": "Sprint Planner",
        "goal": "Plan and organize sprint work into actionable tasks with clear priorities",
        "backstory": """You are the Sprint Planner, responsible for translating
        high-level sprint goals into detailed, actionable work items. You work
        with spec_vp to ensure all requirements are properly decomposed and
        prioritized based on dependencies and business value.""",
    },
    "ace_orchestration": {
        "role": "Orchestration Coordinator",
        "goal": "Coordinate execution flow between workers ensuring smooth handoffs",
        "backstory": """You are the Orchestration Coordinator, working under ace_exec
        to manage the flow of execution tasks across worker agents. You ensure work
        is properly distributed, dependencies are respected, and handoffs between
        workers are seamless.""",
    },
    "human_layer": {
        "role": "Human Layer Coordinator",
        "goal": "Coordinate 7 Human Layer validations for human-centric quality",
        "backstory": """You are the Human Layer Coordinator, orchestrating the 7
        Human Layer validations (Usuario, Operador, Mantenedor, Decisor, Seguranca,
        Hacker, Simplificador). You ensure all deliverables meet human-centric
        quality standards. Rejections go to REWORK, not HALT.""",
    },
    "orchestrator": {
        "role": "System Orchestrator",
        "goal": "High-level orchestration of system components and agents",
        "backstory": """You are the System Orchestrator, responsible for high-level
        coordination of the entire system. You manage the interplay between different
        subsystems and ensure coherent operation of all components.""",
    },
    "run_supervisor": {
        "role": "Run Supervisor",
        "goal": "Monitor and supervise pipeline runs for health and progress",
        "backstory": """You are the Run Supervisor, monitoring pipeline runs to ensure
        they complete successfully. You detect stuck runs, resource issues, and
        coordinate recovery when needed.""",
    },
    "security_specialist": {
        "role": "Security Specialist",
        "goal": "Ensure security best practices are followed throughout the system",
        "backstory": """You are the Security Specialist, maintaining the security
        posture of the system. You review code for vulnerabilities, design security
        controls, and respond to security incidents.""",
    },
    "knowledge_engineer": {
        "role": "Knowledge Engineer",
        "goal": "Design and maintain knowledge graphs for evidence relationships",
        "backstory": """You are the Knowledge Engineer, responsible for knowledge
        representation. You design ontologies, manage the knowledge graph, and
        ensure relationships between facts are properly modeled.""",
    },
}


def get_persona(agent_id: str, include_memory: bool = True) -> dict[str, str]:
    """Get persona for an agent, preferring Cerebro YAML documents.

    This function:
    1. First tries to load the agent's Cerebro from YAML (6 files)
    2. Falls back to SpecKit Markdown loader
    3. Falls back to PERSONA_TEMPLATES

    When include_memory=True:
    - Injects stack configuration context
    - Injects persistent memory (Mem0 + Letta)
    - Injects knowledge graph context (FalkorDB)

    OPT-11-001: Now uses persona cache for O(1) repeated lookups.

    Args:
        agent_id: Agent ID from hierarchy.py.
        include_memory: Whether to include stacks and persistent memory in backstory.

    Returns:
        Dictionary with role, goal, and backstory.
    """
    agent_id = agent_id.lower()

    # OPT-11-001: Check cache first
    cached = _get_cached_persona(agent_id, include_memory)
    if cached is not None:
        return cached

    # =========================================================================
    # PRIORITY 1: Try to load from Cerebro YAML (NEW - 6 files per agent)
    # =========================================================================
    yaml_loader = _get_cerebro_yaml_loader()
    if yaml_loader is not None:
        try:
            cerebro = yaml_loader.load(agent_id)
            if cerebro is not None:
                # Extract persona from Cerebro YAML
                role = cerebro.role or agent_id
                goal = cerebro.mission or f"Execute tasks as {agent_id}"
                backstory = cerebro.vision_anchor or f"You are {agent_id}, following your Cerebro document."

                # STACK INJECTION: Add stack context and memory
                if include_memory:
                    injector = _get_stack_injector()
                    if injector is not None:
                        try:
                            context = injector.build_full_context(
                                cerebro,
                                include_memory=True,
                                include_kg=True,
                            )
                            backstory += "\n\n" + context.to_backstory_section()
                        except Exception as e:
                            # RED TEAM FIX VIO-006: Log errors, don't silently ignore
                            logger.debug(f"VIO-006: Stack injection failed for {agent_id}: {e}")

                # OPT-11-001: Cache before returning
                persona = {
                    "role": role,
                    "goal": goal,
                    "backstory": backstory,
                }
                _cache_persona(agent_id, include_memory, persona)
                return persona
        except FileNotFoundError:
            logger.debug(f"VIO-006: Cerebro YAML not found for {agent_id}, trying fallback")
        except Exception as e:
            # RED TEAM FIX VIO-006: Log errors, don't silently ignore
            logger.warning(f"VIO-006: Cerebro YAML loading failed for {agent_id}: {e}")

    # =========================================================================
    # PRIORITY 2: Try SpecKit Markdown loader (legacy)
    # =========================================================================
    # Legacy Letta memory (if new stack injector not available)
    memory_context = ""
    if include_memory:
        try:
            from pipeline.letta_integration import get_memory_bridge
            bridge = get_memory_bridge()
            memory_context = bridge.get_prompt_context(agent_id, max_items=5)
            if memory_context:
                memory_context = f"\n\n---\n## Your Persistent Memory\n{memory_context}\n---\n"
        except Exception as e:
            # RED TEAM FIX VIO-006: Log errors, don't silently ignore
            logger.debug(f"VIO-006: Letta memory not available for {agent_id}: {e}")

    spec_kit = _get_spec_kit_loader()
    if spec_kit is not None:
        cerebro = spec_kit.load_cerebro(agent_id)
        if cerebro is not None:
            identity = cerebro.identity
            role = identity.role if identity.role else agent_id
            goal = identity.mission
            if not goal and cerebro.vision_anchor:
                goal = cerebro.vision_anchor[:200] if len(cerebro.vision_anchor) > 200 else cerebro.vision_anchor
            if not goal:
                goal = f"Execute tasks as {agent_id}"

            backstory = cerebro.vision_anchor or f"You are {agent_id}, following your Cerebro document."
            backstory = backstory + memory_context

            # OPT-11-001: Cache before returning
            persona = {
                "role": role,
                "goal": goal,
                "backstory": backstory,
            }
            _cache_persona(agent_id, include_memory, persona)
            return persona

    # =========================================================================
    # FALLBACK: Templates and patterns
    # =========================================================================
    if agent_id in PERSONA_TEMPLATES:
        template = PERSONA_TEMPLATES[agent_id].copy()
        template["backstory"] = template["backstory"] + memory_context
        # OPT-11-001: Cache before returning
        _cache_persona(agent_id, include_memory, template)
        return template

    if agent_id.startswith("squad_lead_"):
        persona = {
            "role": f"Squad Lead ({agent_id})",
            "goal": "Lead and coordinate squad workers",
            "backstory": f"You are the squad lead for {agent_id}, coordinating workers to complete assigned tasks.{memory_context}",
        }
        # OPT-11-001: Cache before returning
        _cache_persona(agent_id, include_memory, persona)
        return persona

    if agent_id.startswith("worker_"):
        persona = {
            "role": f"Worker ({agent_id})",
            "goal": "Execute assigned tasks efficiently",
            "backstory": f"You are {agent_id}, a worker responsible for executing specific tasks assigned by your squad lead.{memory_context}",
        }
        # OPT-11-001: Cache before returning
        _cache_persona(agent_id, include_memory, persona)
        return persona

    # FALLBACK: Default persona
    persona = {
        "role": f"Pipeline Agent ({agent_id})",
        "goal": "Execute assigned tasks and support the pipeline",
        "backstory": f"You are {agent_id}, a pipeline agent. Execute your tasks diligently and report to your supervisor.{memory_context}",
    }
    # OPT-11-001: Cache before returning
    _cache_persona(agent_id, include_memory, persona)
    return persona


# =============================================================================
# OPT-11-010: Lazy Backstory Building
# =============================================================================


def _build_full_backstory(
    base: str,
    stack_context: str | None,
    separator: str = "\n\n---\n\n",
) -> str:
    """Build full backstory by concatenating base with optional context.

    OPT-11-010: Only concatenate when stack_context is not empty.

    Args:
        base: Base backstory string.
        stack_context: Optional stack context to append.
        separator: Separator between base and context.

    Returns:
        Full backstory string.
    """
    if not stack_context:
        return base
    return f"{base}{separator}{stack_context}"


def _needs_stack_context(agent_id: str) -> bool:
    """Check if agent needs stack context injection.

    OPT-11-010: Skip stack context for simple/fallback agents.

    Args:
        agent_id: The agent identifier.

    Returns:
        True if agent should have stack context injected.
    """
    # Workers typically don't need full stack context
    if agent_id.startswith("worker_"):
        return False
    # Dynamically spawned agents with task IDs don't need stack context
    if "_task_" in agent_id:
        return False
    return True


# =============================================================================
# Hierarchy to CrewAI Mapping
# =============================================================================


class HierarchyMapper:
    """Maps pipeline hierarchy to CrewAI structure.

    This class provides methods to create CrewAI agents and crews
    based on the pipeline's organizational hierarchy.
    """

    def __init__(self, crewai_client: Optional[CrewAIClient] = None) -> None:
        """Initialize hierarchy mapper.

        Args:
            crewai_client: CrewAI client instance.
        """
        self._client = crewai_client or get_crewai_client()

    def create_agent_definition(
        self,
        agent_id: str,
        include_mcp_tools: bool = True,
        include_stack_context: bool = True,
        preloaded_persona: dict[str, str] | None = None,
    ) -> AgentDefinition:
        """Create a CrewAI agent definition from hierarchy.

        P0-02 FIX: Now accepts preloaded_persona to enable batch persona loading.

        Args:
            agent_id: Agent ID from hierarchy.py.
            include_mcp_tools: Whether to include MCP tools for the agent.
            include_stack_context: Whether to inject stack context into backstory.
            preloaded_persona: Optional pre-loaded persona dict to use instead of
                loading from file. This enables parallel batch loading.

        Returns:
            AgentDefinition for CrewAI.

        Raises:
            ValueError: If agent not in hierarchy.
        """
        if not validate_agent_exists(agent_id):
            raise ValueError(f"Agent '{agent_id}' not found in hierarchy")

        # P0-02: Use preloaded persona if provided, otherwise load from file
        if preloaded_persona is not None:
            persona = preloaded_persona
        else:
            # F-228 FIX: Use runtime card loader for faster persona loading
            # The hierarchy is: runtime_card (fast) -> cerebro (full) -> templates (fallback)
            from pipeline.runtime_card_loader import get_persona_with_runtime_card
            persona = get_persona_with_runtime_card(agent_id)

        level = get_level(agent_id) or 0
        subordinates = get_subordinates(agent_id)

        # Get MCP tool names for the agent
        tools: list[str] = []
        if include_mcp_tools:
            mcp_tools = _get_mcp_tools_module()
            if mcp_tools is not None:
                try:
                    tool_names = mcp_tools.get_mcp_tool_names_for_agent(agent_id)
                    tools.extend(tool_names)
                except Exception as e:
                    # 2026-01-10 FIX: Log instead of silently swallowing
                    logger.debug(f"MCP tools not available for {agent_id}: {e}")

        # OPT-11-010: Lazy backstory building - only get stack context if needed
        backstory = persona["backstory"]
        if include_stack_context and _needs_stack_context(agent_id):
            stack_context = _get_stack_context_for_agent(agent_id)
            backstory = _build_full_backstory(backstory, stack_context)
            if stack_context:
                logger.debug(f"Injected stack context for agent {agent_id}")

        return AgentDefinition(
            agent_id=agent_id,
            role=persona["role"],
            goal=persona["goal"],
            backstory=backstory,
            tools=tools,
            allow_delegation=len(subordinates) > 0,
            verbose=False,
            max_iter=10 if level < 3 else 5,  # Higher levels get more iterations
        )

    def create_squad_crew(
        self,
        manager_id: str,
        task_descriptions: list[tuple[str, str]],
        use_parallel_loading: bool = True,
    ) -> tuple[list[AgentDefinition], list[TaskDefinition]]:
        """Create agent and task definitions for a squad.

        P0-02 FIX: Now supports parallel persona loading for improved performance.

        Args:
            manager_id: Squad manager (e.g., qa_master, ace_exec).
            task_descriptions: List of (description, expected_output) tuples.
            use_parallel_loading: Whether to load personas in parallel.
                Set to False for backwards compatibility or debugging.

        Returns:
            Tuple of (agent_definitions, task_definitions).

        Raises:
            ValueError: If manager not in hierarchy.
        """
        if not validate_agent_exists(manager_id):
            raise ValueError(f"Manager '{manager_id}' not found in hierarchy")

        agents: list[AgentDefinition] = []
        tasks: list[TaskDefinition] = []

        # Get subordinates first to know all agent IDs
        subordinates = list(get_subordinates(manager_id))
        all_agent_ids = [manager_id] + subordinates

        # P0-02: Batch load all personas in parallel (if enabled and > 1 agent)
        if use_parallel_loading and len(all_agent_ids) > 1:
            logger.info(f"P0-02: Loading {len(all_agent_ids)} personas in parallel")
            personas = get_personas_parallel_sync(all_agent_ids)
        else:
            # Fallback to sequential loading (backwards compatible)
            personas = {}

        # Create manager agent
        agents.append(self.create_agent_definition(
            manager_id,
            preloaded_persona=personas.get(manager_id),
        ))

        # Create subordinate agents
        for sub_id in subordinates:
            agents.append(self.create_agent_definition(
                sub_id,
                preloaded_persona=personas.get(sub_id),
            ))

        # Create tasks distributed among subordinates
        for i, (desc, expected) in enumerate(task_descriptions):
            # Round-robin assign to subordinates (or manager if no subs)
            if subordinates:
                assigned = subordinates[i % len(subordinates)]
            else:
                assigned = manager_id

            tasks.append(
                TaskDefinition(
                    task_id=f"task_{i}",
                    description=desc,
                    expected_output=expected,
                    agent_id=assigned,
                )
            )

        return agents, tasks

    def run_squad(
        self,
        crew_id: str,
        manager_id: str,
        task_descriptions: list[tuple[str, str]],
        requesting_agent: str | None = None,
    ) -> CrewResult:
        """Run a squad as a CrewAI crew.

        Args:
            crew_id: Unique identifier for this crew run.
            manager_id: Squad manager ID.
            task_descriptions: List of (description, expected_output) tuples.
            requesting_agent: Agent requesting this squad execution (for audit).

        Returns:
            CrewResult with execution details.

        Note:
            RED TEAM FIX H-01 (2026-01-04): Added requesting_agent validation.
            Per INV-014, all actions must have agent identity.
        """
        # RED TEAM FIX H-01: Validate caller authority
        if requesting_agent:
            caller_level = get_level(requesting_agent)
            manager_level = get_level(manager_id)
            if caller_level is not None and manager_level is not None:
                if caller_level > manager_level:
                    logger.warning(
                        f"HIERARCHY VIOLATION: {requesting_agent} (L{caller_level}) "
                        f"attempted to run squad managed by {manager_id} (L{manager_level})"
                    )
                    # Log but allow - this is for audit, not blocking
        else:
            logger.debug(f"run_squad called without requesting_agent for crew {crew_id}")

        agents, tasks = self.create_squad_crew(manager_id, task_descriptions)

        return self._client.run_crew(
            crew_id=crew_id,
            agents=agents,
            tasks=tasks,
            manager_id=manager_id,
            process="hierarchical",
        )

    # CRIT-004 FIX (2026-01-22): Veto power enforcement - DISTRIBUTED via Redis
    #
    # PREVIOUS BUG: _vetoed_operations was a class-level set, meaning each
    # Python process had its own set. In multi-process environments (e.g.,
    # multiple Celery workers, gunicorn workers, or subprocess spawns),
    # vetoes registered in one process were invisible to others, allowing
    # attackers to bypass vetoes by exploiting process isolation.
    #
    # FIX: Vetoes are now stored in Redis with the following design:
    # - Key: "veto:{operation_id}" with value containing veto details
    # - TTL: 24 hours by default (vetoes don't persist forever)
    # - All processes share the same Redis instance
    # - Thread-local lock only for local cache synchronization
    #
    # FALLBACK: If Redis is unavailable, we fall back to the class-level set
    # but log a CRITICAL warning. This maintains backward compatibility
    # while making the security gap visible.

    _vetoed_operations: set[str] = set()  # LOCAL FALLBACK ONLY - not authoritative
    _veto_lock: threading.Lock = threading.Lock()  # Thread safety for local fallback

    # Redis key prefix for distributed vetoes
    VETO_KEY_PREFIX = "pipeline:veto:"
    VETO_TTL_SECONDS = 86400  # 24 hours

    def _get_redis_client(self):
        """Get Redis client for distributed veto storage."""
        try:
            from pipeline.redis_client import get_redis_client
            client = get_redis_client()
            if client.is_available():
                return client
        except Exception as e:
            logger.debug(f"Redis not available for veto storage: {e}")
        return None

    def check_ops_ctrl_veto(self, operation_id: str, operation_type: str) -> tuple[bool, str]:
        """Check if ops_ctrl has vetoed an operation.

        CRIT-004 FIX (2026-01-22): Now uses Redis for distributed veto checking.
        Vetoes are visible across all processes and workers.

        RED TEAM FIX HL-08: Implements actual veto power for safety-critical decisions.

        Args:
            operation_id: Unique operation identifier.
            operation_type: Type of operation (e.g., 'gate_execution', 'phase_transition').

        Returns:
            Tuple of (is_vetoed, reason).
        """
        # CRIT-004 FIX: Check Redis first (distributed source of truth)
        redis_client = self._get_redis_client()
        if redis_client:
            try:
                veto_key = f"{self.VETO_KEY_PREFIX}{operation_id}"
                veto_data = redis_client.get_json(veto_key)
                if veto_data:
                    reason = veto_data.get("reason", f"Operation {operation_id} was vetoed")
                    vetoing_agent = veto_data.get("vetoing_agent", "unknown")
                    logger.info(f"CRIT-004: Veto found in Redis for {operation_id} by {vetoing_agent}")
                    return True, reason
            except Exception as e:
                logger.warning(f"CRIT-004: Redis veto check failed, using fallback: {e}")
        else:
            # CRIT-004: Log warning when Redis is unavailable
            logger.warning(
                "CRIT-004 SECURITY WARNING: Redis unavailable for distributed veto check. "
                "Using local fallback - vetoes may not be visible across processes!"
            )

        # Fallback to local set
        with self._veto_lock:
            if operation_id in self._vetoed_operations:
                return True, f"Operation {operation_id} was vetoed by ops_ctrl"

        # RED TEAM FIX VIO-004: security_gate_skip is PERMANENTLY BLOCKED
        # Security gates are inviolable per INV-006
        PERMANENTLY_BLOCKED_OPERATIONS = {
            'security_gate_skip',  # RED TEAM FIX VIO-004: NEVER allowed
        }

        if operation_type in PERMANENTLY_BLOCKED_OPERATIONS:
            logger.error(
                f"VIO-004 BLOCKED: Operation {operation_id} ({operation_type}) "
                "is PERMANENTLY BLOCKED. Security gates cannot be skipped under ANY circumstances."
            )
            return True, f"PERMANENT BLOCK: {operation_type} is prohibited per INV-006"

        # Check for safety-critical operation types that require ops_ctrl approval
        SAFETY_CRITICAL_OPERATIONS = {
            'production_deploy',
            'database_migration',
            'human_layer_override',
        }

        if operation_type in SAFETY_CRITICAL_OPERATIONS:
            logger.warning(
                f"SAFETY-CRITICAL: Operation {operation_id} ({operation_type}) "
                "requires ops_ctrl approval before proceeding"
            )
            # Return vetoed by default - requires explicit approval
            return True, f"Safety-critical operation {operation_type} requires ops_ctrl approval"

        return False, ""

    def register_veto(self, operation_id: str, reason: str, vetoing_agent: str = "ops_ctrl") -> None:
        """Register a veto for an operation.

        CRIT-004 FIX (2026-01-22): Vetoes are now stored in Redis for
        distributed visibility across all processes and workers.

        Only L0 agents (ops_ctrl, pack_driver, run_master) can veto.

        Args:
            operation_id: Operation to veto.
            reason: Reason for veto.
            vetoing_agent: Agent issuing the veto.

        Raises:
            ValueError: If vetoing agent lacks authority.
        """
        vetoing_level = get_level(vetoing_agent)
        if vetoing_level is None or vetoing_level > 0:
            raise ValueError(
                f"HL-08 VIOLATION: Only L0 agents can veto. "
                f"{vetoing_agent} (L{vetoing_level}) lacks authority."
            )

        # CRIT-004 FIX: Store veto in Redis (distributed)
        redis_client = self._get_redis_client()
        if redis_client:
            try:
                from datetime import datetime, timezone

                veto_data = {
                    "operation_id": operation_id,
                    "reason": reason,
                    "vetoing_agent": vetoing_agent,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                veto_key = f"{self.VETO_KEY_PREFIX}{operation_id}"
                redis_client.set_json(veto_key, veto_data, ex=self.VETO_TTL_SECONDS)
                logger.info(
                    f"CRIT-004: VETO REGISTERED in Redis: {operation_id} vetoed by {vetoing_agent}. "
                    f"Reason: {reason}. TTL: {self.VETO_TTL_SECONDS}s"
                )
            except Exception as e:
                logger.error(f"CRIT-004: Failed to store veto in Redis: {e}")
        else:
            logger.warning(
                "CRIT-004 SECURITY WARNING: Redis unavailable for veto storage. "
                "Veto will only be visible in this process!"
            )

        # Also store locally as fallback
        with self._veto_lock:
            self._vetoed_operations.add(operation_id)
        logger.info(f"VETO REGISTERED (local): {operation_id} vetoed by {vetoing_agent}. Reason: {reason}")

    def clear_veto(self, operation_id: str, clearing_agent: str) -> None:
        """Clear a veto (requires L0 authority).

        CRIT-004 FIX (2026-01-22): Clears veto from both Redis and local set.
        """
        clearing_level = get_level(clearing_agent)
        if clearing_level is None or clearing_level > 0:
            raise ValueError(f"Only L0 agents can clear vetoes. {clearing_agent} lacks authority.")

        # CRIT-004 FIX: Clear from Redis
        redis_client = self._get_redis_client()
        if redis_client:
            try:
                veto_key = f"{self.VETO_KEY_PREFIX}{operation_id}"
                redis_client.delete(veto_key)
                logger.info(f"CRIT-004: VETO CLEARED from Redis: {operation_id} cleared by {clearing_agent}")
            except Exception as e:
                logger.error(f"CRIT-004: Failed to clear veto from Redis: {e}")

        # Also clear locally
        with self._veto_lock:
            self._vetoed_operations.discard(operation_id)
        logger.info(f"VETO CLEARED (local): {operation_id} cleared by {clearing_agent}")

    def get_hierarchy_levels(self) -> dict[int, list[str]]:
        """Get all agents organized by hierarchy level.

        Returns:
            Dictionary mapping level to list of agent IDs.
        """
        levels: dict[int, list[str]] = {}

        for agent_id in HIERARCHY_MAP:
            level = get_level(agent_id)
            if level is not None:
                if level not in levels:
                    levels[level] = []
                levels[level].append(agent_id)

        return levels

    def get_squad_structure(self, master_id: str) -> dict[str, Any]:
        """Get the squad structure under a master.

        Args:
            master_id: Master agent ID (e.g., qa_master, ace_exec).

        Returns:
            Dictionary with squad structure.
        """
        if not validate_agent_exists(master_id):
            return {}

        subordinates_dict: dict[str, Any] = {}
        structure: dict[str, Any] = {
            "master": master_id,
            "master_level": get_level(master_id),
            "supervisor": get_supervisor(master_id),
            "subordinates": subordinates_dict,
        }

        # Get direct subordinates (squad leads or workers)
        subordinates = get_subordinates(master_id)
        for sub_id in subordinates:
            sub_subs = get_subordinates(sub_id)
            subordinates_dict[sub_id] = {
                "level": get_level(sub_id),
                "workers": sub_subs,
            }

        return structure


# =============================================================================
# Factory Functions
# =============================================================================


def _run_squad_with_degradation_tracking(
    mapper: HierarchyMapper,
    crew_id: str,
    vp_id: str,
    tasks: list[tuple[str, str]],
    crew_type: str,
) -> CrewResult:
    """Run squad with degradation tracking and logging.

    P1-6.4 FIX: Wraps mapper.run_squad to add degraded_features tracking
    and log a summary at the end of crew execution.

    Args:
        mapper: HierarchyMapper instance.
        crew_id: Unique crew identifier.
        vp_id: VP to route tasks through.
        tasks: List of (description, expected_output) tuples.
        crew_type: Type of crew ("qa", "spec", "exec") for logging.

    Returns:
        CrewResult with degraded_features populated.
    """
    # Get degraded features BEFORE execution (they accumulate during import)
    degraded = get_degraded_features()

    # Execute the squad
    result = mapper.run_squad(crew_id, vp_id, tasks)

    # Attach degraded features to result
    if degraded:
        result.degraded_features = degraded
        logger.warning(
            f"P1-6: [{crew_type}_crew] Completed with degraded features: {degraded}"
        )
    else:
        logger.info(f"[{crew_type}_crew] Completed with all features available")

    return result


def create_qa_crew(
    crew_id: str,
    review_targets: list[str],
    use_got: bool = True,
    # NEW: Parameters for parallel execution and context injection
    context_pack: dict | None = None,
    parallel_workers: bool = True,
    run_dir: str | None = None,
    sprint_id: str | None = None,
) -> CrewResult:
    """Create and run a QA crew with parallel worker support.

    Uses GoT (Graph of Thoughts) for multi-perspective validation
    when use_got=True (default). This improves QA coverage by generating
    multiple validation perspectives and reaching consensus.

    NEW (2026-01-28): Supports parallel worker execution via QAMasterOrchestrator.
    When parallel_workers=True (default), 8 subordinate workers run in parallel
    for ~8x speedup.

    Args:
        crew_id: Unique identifier for this crew run.
        review_targets: List of items to review.
        use_got: Whether to use GoT for multi-perspective validation (default True).
        context_pack: Sprint context with RF/INV/EDGE for validation (optional).
        parallel_workers: Whether to run workers in parallel (default True).
        run_dir: Run directory for artifacts (optional, uses temp if not provided).
        sprint_id: Sprint identifier (optional, extracted from crew_id if not provided).

    Returns:
        CrewResult with execution details.

    Reference: docs/pipeline/QA_MASTER_PARALLEL_DEEP_STUDY.md
    """
    import asyncio
    from pathlib import Path
    import tempfile

    mapper = HierarchyMapper()
    tasks: list[tuple[str, str]] = []

    # Extract sprint_id from crew_id if not provided (format: "S00_qa")
    if sprint_id is None:
        sprint_id = crew_id.split("_")[0] if "_" in crew_id else "S00"

    # Use temp dir if run_dir not provided
    if run_dir is None:
        run_dir = Path(tempfile.gettempdir()) / "pipeline_runs" / crew_id
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = Path(run_dir)

    # NEW: Use QAMasterOrchestrator for parallel execution
    if parallel_workers:
        try:
            from pipeline.qa_master_orchestrator import (
                QAMasterOrchestrator,
                run_qa_validation,
            )

            # Run orchestrator asynchronously
            async def _run_orchestrator():
                orchestrator = QAMasterOrchestrator(
                    run_dir=run_dir,
                    sprint_id=sprint_id,
                )
                return await orchestrator.run(
                    deliverables=review_targets,
                    context_pack=context_pack or {},
                    task_id=crew_id,
                    run_gates=False,  # Gates are run separately in workflow
                    run_workers=True,
                )

            # Run in event loop
            try:
                loop = asyncio.get_running_loop()
                # Already in async context, create task
                future = asyncio.ensure_future(_run_orchestrator())
                qa_result = loop.run_until_complete(future)
            except RuntimeError:
                # No running loop, create one
                qa_result = asyncio.run(_run_orchestrator())

            # Convert QAResult to CrewResult format
            return CrewResult(
                crew_id=crew_id,
                success=qa_result.passed,
                output=qa_result.to_dict(),
                tasks_completed=len(qa_result.worker_results),
                errors=[qa_result.decision.reason] if not qa_result.passed else [],
            )

        except ImportError as e:
            logger.warning(
                f"[create_qa_crew] QAMasterOrchestrator not available ({e}), "
                "falling back to sequential execution"
            )
            # Fall through to original implementation

    # Original implementation (sequential, with GoT)
    if use_got:
        try:
            from pipeline.got_integration import validate_with_got

            for target in review_targets:
                # Pre-validate with GoT to get validation criteria
                result = validate_with_got(
                    artifact=target,
                    criteria=["correctness", "completeness", "security", "performance"],
                    num_perspectives=3,
                )

                # Add task with GoT findings as context
                if result.findings:
                    findings_str = "; ".join(result.findings[:3])
                    tasks.append((
                        f"Review {target} for quality issues. GoT pre-analysis found: {findings_str}",
                        f"Detailed review report for {target} addressing GoT findings",
                    ))
                else:
                    tasks.append((
                        f"Review {target} for quality issues and bugs (GoT pre-analysis: PASS)",
                        f"Review report for {target}",
                    ))
        except (ImportError, RuntimeError) as e:
            # P1-6.1 FIX: Register GoT degradation for observability
            _register_degraded_feature("got_qa_validation", f"GoT not available for QA: {e}")
            # RED TEAM FIX: RuntimeError from D-02 fix must also be caught
            # GoT not available, use original approach
            tasks = [
                (f"Review {target} for quality issues and bugs", f"Review report for {target}")
                for target in review_targets
            ]
    else:
        tasks = [
            (f"Review {target} for quality issues and bugs", f"Review report for {target}")
            for target in review_targets
        ]

    # TASK-F1-003: QA is under exec_vp (unified VP for execution + QA)
    # qa_vp was removed - exec_vp handles both execution AND QA oversight
    # exec_vp will delegate to qa_master who is a subordinate
    # P1-6.4 FIX: Use wrapper for degradation tracking
    return _run_squad_with_degradation_tracking(mapper, crew_id, "exec_vp", tasks, "qa")


def create_spec_crew(
    crew_id: str,
    features: list[str],
    use_got: bool = True,
    stack_ctx: "StackContext | None" = None,  # P0-3 FIX: Accept stack context
) -> CrewResult:
    """Create and run a specification crew.

    Uses GoT (Graph of Thoughts) for multi-perspective spec decomposition
    when use_got=True (default). This improves spec quality by generating
    multiple decomposition approaches and selecting the best one.

    P0-3 FIX: Now accepts stack_ctx for stack injection.

    GHOST CODE INTEGRATION (2026-01-30):
    - Uses CrewAI Shared Memory for spec artifact persistence
    - Uses CrewAI Coordination for multi-agent orchestration

    Args:
        crew_id: Unique identifier for this crew run.
        features: List of features to specify.
        use_got: Whether to use GoT for spec decomposition (default True).
        stack_ctx: Optional stack context with injected stacks (P0-3).

    Returns:
        CrewResult with execution details.
    """
    # P0-3 FIX: Log available stacks
    if stack_ctx:
        available_stacks = list(stack_ctx.stacks.keys()) if stack_ctx.stacks else []
        logger.info(f"[create_spec_crew] Stack context available: {available_stacks}")
    mapper = HierarchyMapper()
    tasks: list[tuple[str, str]] = []

    # =========================================================================
    # GHOST CODE INTEGRATION: CrewAI Shared Memory (2026-01-30)
    # Enable knowledge persistence across spec agents
    # =========================================================================
    shared_memory_config = None
    if SHARED_MEMORY_AVAILABLE:
        try:
            shared_memory_config = get_crewai_shared_memory_config()
            if shared_memory_config:
                logger.info(
                    f"[create_spec_crew] Shared memory enabled for {crew_id}: "
                    f"STM={shared_memory_config.get('short_term_enabled', False)}, "
                    f"LTM={shared_memory_config.get('long_term_enabled', False)}"
                )
        except Exception as e:
            logger.warning(f"[create_spec_crew] Shared memory init failed (non-blocking): {e}")

    # =========================================================================
    # GHOST CODE INTEGRATION: CrewAI Coordination (2026-01-30)
    # Enable multi-agent coordination for parallel spec generation
    # =========================================================================
    coordinator = None
    if CREW_COORDINATION_AVAILABLE:
        try:
            coordinator = get_coordinated_crew_executor()
            if coordinator:
                logger.info(f"[create_spec_crew] Crew coordination enabled for {crew_id}")
        except Exception as e:
            logger.warning(f"[create_spec_crew] Crew coordination init failed (non-blocking): {e}")

    # Use GoT for improved spec decomposition
    if use_got:
        try:
            from pipeline.got_integration import decompose_spec_with_got

            for feature in features:
                result = decompose_spec_with_got(
                    requirement=feature,
                    context=f"Crew: {crew_id}",
                    num_perspectives=3,
                )

                if result.success and result.subtasks:
                    # Create tasks for each GoT-generated subtask
                    for subtask in result.subtasks:
                        tasks.append((
                            f"Write specification for: {subtask}",
                            f"Specification document for {subtask} (part of {feature})",
                        ))
                else:
                    # Fallback to original feature
                    tasks.append((
                        f"Write specification for {feature}",
                        f"Specification document for {feature}",
                    ))
        except (ImportError, RuntimeError) as e:
            # P1-6.1 FIX: Register GoT degradation for observability
            _register_degraded_feature("got_spec_decomposition", f"GoT not available for spec: {e}")
            # RED TEAM FIX: RuntimeError from D-02 fix must also be caught
            # GoT not available, use original approach
            tasks = [
                (f"Write specification for {feature}", f"Specification document for {feature}")
                for feature in features
            ]
    else:
        tasks = [
            (f"Write specification for {feature}", f"Specification document for {feature}")
            for feature in features
        ]

    # =========================================================================
    # GHOST CODE INTEGRATION: Use Coordinator if available (2026-01-30)
    # Enables parallel spec generation with cross-agent coordination
    # =========================================================================
    if coordinator is not None:
        try:
            import asyncio

            async def _run_coordinated_spec():
                # Create coordinated crew with shared memory
                coordinated_tasks = []
                for task_desc, expected_output in tasks:
                    coordinated_tasks.append({
                        "description": task_desc,
                        "expected_output": expected_output,
                        "agent_role": "spec_master",
                    })

                result = await coordinator.run_coordinated(
                    crew_id=crew_id,
                    tasks=coordinated_tasks,
                    memory_config=shared_memory_config,
                )
                return result

            # Execute async coordinator
            try:
                loop = asyncio.get_running_loop()
                future = asyncio.ensure_future(_run_coordinated_spec())
                coord_result = loop.run_until_complete(future)
            except RuntimeError:
                coord_result = asyncio.run(_run_coordinated_spec())

            # Convert coordinator result to CrewResult
            if coord_result:
                logger.info(
                    f"[create_spec_crew] Coordinated execution completed: "
                    f"tasks={len(tasks)}, success={coord_result.get('success', False)}"
                )
                return CrewResult(
                    crew_id=crew_id,
                    success=coord_result.get("success", False),
                    output=coord_result.get("output", {}),
                    tasks_completed=coord_result.get("tasks_completed", len(tasks)),
                    errors=coord_result.get("errors", []),
                )
        except Exception as e:
            logger.warning(
                f"[create_spec_crew] Coordinated execution failed, falling back to sequential: {e}"
            )
            # Fall through to sequential execution

    # =========================================================================
    # GHOST CODE INTEGRATION: Persist specs to shared memory (2026-01-30)
    # Store spec artifacts for cross-agent retrieval
    # =========================================================================
    if shared_memory_config and SHARED_MEMORY_AVAILABLE:
        try:
            from pipeline.agents.crewai_shared_memory import get_shared_memory_manager
            memory_manager = get_shared_memory_manager()
            if memory_manager:
                # Store feature list for downstream agents
                import asyncio
                asyncio.run(memory_manager.store_memory(
                    agent_id="spec_master",
                    key=f"{crew_id}_features",
                    value={"features": features, "task_count": len(tasks)},
                    scope="crew",
                ))
                logger.debug(f"[create_spec_crew] Stored {len(features)} features in shared memory")
        except Exception as e:
            logger.debug(f"[create_spec_crew] Shared memory store failed (non-blocking): {e}")

    # TASK-F1-001: Route through spec_vp instead of directly to spec_master
    # This ensures VP oversight in the spec phase
    # P1-6.4 FIX: Use wrapper for degradation tracking
    return _run_squad_with_degradation_tracking(mapper, crew_id, "spec_vp", tasks, "spec")


def create_exec_crew(
    crew_id: str,
    implementations: list[str],
    use_got: bool = True,
    context_pack: dict | None = None,
    granular_tasks: list[dict] | None = None,
    stack_ctx: "StackContext | None" = None,  # P0-3 FIX: Accept stack context
) -> CrewResult:
    """Create and run an execution crew.

    Uses GoT (Graph of Thoughts) for task planning when use_got=True (default).
    This improves execution by generating multiple implementation approaches,
    identifying risks, and selecting the best plan.

    GAP-4 FIX: Now accepts context_pack and granular_tasks for proper RF/INV/EDGE
    propagation to agents. When granular_tasks is provided, uses those instead
    of generic "Implement {item}" tasks.

    P0-3 FIX: Now accepts stack_ctx for stack injection into crew operations.
    Stacks like Redis, Qdrant, etc. can be used for RAG, caching, etc.

    GHOST CODE INTEGRATION (2026-01-30):
    - Uses CrewAI Shared Memory for context sharing between exec agents
    - Uses CrewAI Coordination for task delegation

    Args:
        crew_id: Unique identifier for this crew run.
        implementations: List of items to implement.
        use_got: Whether to use GoT for task planning (default True).
        context_pack: Optional context pack with RF/INV/EDGE data.
        granular_tasks: Optional list of pre-decomposed tasks with context.
            Each task dict should have:
            - deliverable: str
            - requirements: list[dict] (RF)
            - invariants: list[dict] (INV)
            - edge_cases: list[dict] (EDGE)
            - task_prompt: str (full prompt)
        stack_ctx: Optional stack context with injected stacks (P0-3).

    Returns:
        CrewResult with execution details.
    """
    # P0-3 FIX: Log available stacks
    if stack_ctx:
        available_stacks = list(stack_ctx.stacks.keys()) if stack_ctx.stacks else []
        logger.info(f"[create_exec_crew] Stack context available: {available_stacks}")
    mapper = HierarchyMapper()
    tasks: list[tuple[str, str]] = []

    # =========================================================================
    # GHOST CODE INTEGRATION: CrewAI Shared Memory (2026-01-30)
    # Enable context sharing between implementation agents
    # =========================================================================
    shared_memory_config = None
    if SHARED_MEMORY_AVAILABLE:
        try:
            shared_memory_config = get_crewai_shared_memory_config()
            if shared_memory_config:
                logger.info(
                    f"[create_exec_crew] Shared memory enabled for {crew_id}"
                )

                # Pre-load spec context from shared memory if available
                try:
                    from pipeline.agents.crewai_shared_memory import get_shared_memory_manager
                    import asyncio

                    memory_manager = get_shared_memory_manager()
                    if memory_manager:
                        # Try to retrieve spec context from previous spec crew
                        spec_crew_id = crew_id.replace("_exec", "_spec")
                        spec_context = asyncio.run(memory_manager.retrieve_memory(
                            agent_id="ace_exec",
                            key=f"{spec_crew_id}_features",
                        ))
                        if spec_context:
                            logger.debug(
                                f"[create_exec_crew] Retrieved spec context from shared memory"
                            )
                except Exception as e:
                    logger.debug(f"[create_exec_crew] Spec context retrieval failed: {e}")

        except Exception as e:
            logger.warning(f"[create_exec_crew] Shared memory init failed (non-blocking): {e}")

    # =========================================================================
    # GHOST CODE INTEGRATION: CrewAI Coordination (2026-01-30)
    # Enable task delegation for parallel implementation
    # =========================================================================
    coordinator = None
    if CREW_COORDINATION_AVAILABLE:
        try:
            coordinator = get_coordinated_crew_executor()
            if coordinator:
                logger.info(f"[create_exec_crew] Crew coordination enabled for {crew_id}")
        except Exception as e:
            logger.warning(f"[create_exec_crew] Crew coordination init failed (non-blocking): {e}")

    # GAP-4: If granular_tasks provided, use them with full context
    if granular_tasks:
        for task_data in granular_tasks:
            deliverable = task_data.get("deliverable", "unknown")
            task_prompt = task_data.get("task_prompt", f"Implement {deliverable}")

            # Format rich task description with RF/INV/EDGE
            requirements = task_data.get("requirements", [])
            invariants = task_data.get("invariants", [])
            edge_cases = task_data.get("edge_cases", [])

            description_parts = [f"## Deliverable: {deliverable}", ""]

            if requirements:
                description_parts.append("## Requirements to Implement:")
                for rf in requirements:
                    rf_id = rf.get("id", "RF-???")
                    rf_desc = rf.get("description", rf.get("text", ""))
                    description_parts.append(f"- [{rf_id}] {rf_desc}")
                description_parts.append("")

            if invariants:
                description_parts.append("## Invariants to Enforce:")
                for inv in invariants:
                    inv_id = inv.get("id", "INV-???")
                    inv_rule = inv.get("rule", inv.get("description", ""))
                    description_parts.append(f"- [{inv_id}] {inv_rule}")
                description_parts.append("")

            if edge_cases:
                description_parts.append("## Edge Cases to Handle:")
                for edge in edge_cases:
                    edge_id = edge.get("id", "EDGE-???")
                    edge_scenario = edge.get("scenario", edge.get("description", ""))
                    edge_expected = edge.get("expected", "")
                    desc = f"- [{edge_id}] {edge_scenario}"
                    if edge_expected:
                        desc += f" → Expected: {edge_expected}"
                    description_parts.append(desc)
                description_parts.append("")

            description_parts.append("## Implementation Instructions:")
            description_parts.append(task_prompt)

            full_description = "\n".join(description_parts)
            tasks.append((full_description, f"Implementation of {deliverable}"))

        # TASK-F1-002: Route through exec_vp instead of directly to ace_exec
        # This ensures VP oversight in the execution phase
        # P1-6.4 FIX: Use wrapper for degradation tracking
        return _run_squad_with_degradation_tracking(mapper, crew_id, "exec_vp", tasks, "exec")

    # Use GoT for improved task planning (original path, with optional context enrichment)
    if use_got:
        try:
            from pipeline.got_integration import plan_tasks_with_got

            for item in implementations:
                # Enrich constraints from context_pack if available
                constraints = ["Follow existing code patterns", "Include tests"]
                if context_pack:
                    invs = context_pack.get("invariants", [])
                    for inv in invs[:3]:  # Top 3 invariants as constraints
                        inv_rule = inv.get("rule", inv.get("description", ""))
                        if inv_rule:
                            constraints.append(f"INVARIANT: {inv_rule}")

                result = plan_tasks_with_got(
                    goal=f"Implement {item}",
                    constraints=constraints,
                    num_plans=3,
                )

                if result.success and result.steps:
                    # Create tasks for each planned step
                    for step in result.steps:
                        tasks.append((
                            step,
                            f"Completed: {step}",
                        ))

                    # Add risk mitigation tasks if needed
                    for risk in result.risks[:2]:  # Top 2 risks
                        tasks.append((
                            f"Mitigate risk: {risk}",
                            f"Risk mitigation implemented for: {risk}",
                        ))
                else:
                    # Fallback to original item
                    tasks.append((
                        f"Implement {item}",
                        f"Implementation of {item}",
                    ))
        except (ImportError, RuntimeError) as e:
            # P1-6.1 FIX: Register GoT degradation for observability
            _register_degraded_feature("got_exec_planning", f"GoT not available for exec: {e}")
            # RED TEAM FIX: RuntimeError from D-02 fix must also be caught
            # GoT not available, use original approach
            tasks = [
                (f"Implement {item}", f"Implementation of {item}")
                for item in implementations
            ]
    else:
        tasks = [
            (f"Implement {item}", f"Implementation of {item}")
            for item in implementations
        ]

    # =========================================================================
    # GHOST CODE INTEGRATION: Use Coordinator if available (2026-01-30)
    # Enables parallel execution with task delegation
    # =========================================================================
    if coordinator is not None:
        try:
            import asyncio

            async def _run_coordinated_exec():
                # Create coordinated tasks
                coordinated_tasks = []
                for task_desc, expected_output in tasks:
                    coordinated_tasks.append({
                        "description": task_desc,
                        "expected_output": expected_output,
                        "agent_role": "ace_exec",
                    })

                result = await coordinator.run_coordinated(
                    crew_id=crew_id,
                    tasks=coordinated_tasks,
                    memory_config=shared_memory_config,
                )
                return result

            # Execute async coordinator
            try:
                loop = asyncio.get_running_loop()
                future = asyncio.ensure_future(_run_coordinated_exec())
                coord_result = loop.run_until_complete(future)
            except RuntimeError:
                coord_result = asyncio.run(_run_coordinated_exec())

            # Convert coordinator result to CrewResult
            if coord_result:
                logger.info(
                    f"[create_exec_crew] Coordinated execution completed: "
                    f"tasks={len(tasks)}, success={coord_result.get('success', False)}"
                )
                return CrewResult(
                    crew_id=crew_id,
                    success=coord_result.get("success", False),
                    output=coord_result.get("output", {}),
                    tasks_completed=coord_result.get("tasks_completed", len(tasks)),
                    errors=coord_result.get("errors", []),
                )
        except Exception as e:
            logger.warning(
                f"[create_exec_crew] Coordinated execution failed, falling back to sequential: {e}"
            )
            # Fall through to sequential execution

    # =========================================================================
    # GHOST CODE INTEGRATION: Store execution context in shared memory (2026-01-30)
    # =========================================================================
    if shared_memory_config and SHARED_MEMORY_AVAILABLE:
        try:
            from pipeline.agents.crewai_shared_memory import get_shared_memory_manager
            import asyncio

            memory_manager = get_shared_memory_manager()
            if memory_manager:
                asyncio.run(memory_manager.store_memory(
                    agent_id="ace_exec",
                    key=f"{crew_id}_implementations",
                    value={"implementations": implementations, "task_count": len(tasks)},
                    scope="crew",
                ))
                logger.debug(f"[create_exec_crew] Stored {len(implementations)} implementations in shared memory")
        except Exception as e:
            logger.debug(f"[create_exec_crew] Shared memory store failed (non-blocking): {e}")

    # TASK-F1-002: Route through exec_vp instead of directly to ace_exec
    # This ensures VP oversight in the execution phase
    # P1-6.4 FIX: Use wrapper for degradation tracking
    return _run_squad_with_degradation_tracking(mapper, crew_id, "exec_vp", tasks, "exec")


# =============================================================================
# Individual Agent Factory Functions (GAP-NEW-002, GAP-NEW-006 FIX: 2026-01-31)
# =============================================================================
# These functions create individual CrewAI agents for CEO and VP roles.
# They can be used standalone or as building blocks for custom crews.


def create_ceo_agent():
    """Create CEO agent for orchestrating VPs.

    CEO is the Chief Executive Officer, responsible for orchestrating
    VPs and executing strategy defined by Presidente. CEO sits at L2
    in the hierarchy and manages spec_vp, exec_vp, and other
    executive support agents.

    Returns:
        CrewAI Agent configured as CEO.

    Raises:
        ImportError: If crewai is not available.

    Example:
        >>> ceo = create_ceo_agent()
        >>> print(ceo.role)
        'Chief Executive Officer'
    """
    from crewai import Agent

    persona = get_persona("ceo", include_memory=True)
    return Agent(
        role=persona.get("role", "Chief Executive Officer"),
        goal=persona.get("goal", "Execute strategy and coordinate VPs"),
        backstory=persona.get(
            "backstory",
            "You are the CEO, responsible for executing strategic vision "
            "and coordinating VPs to deliver high-quality results.",
        ),
        verbose=True,
        allow_delegation=True,
    )


def create_spec_vp_agent():
    """Create Spec VP agent for overseeing specification work.

    Spec VP (Vice President of Specifications) is responsible for
    overseeing product definition and specification quality. Reports
    to CEO and manages spec_master and sprint_planner.

    Returns:
        CrewAI Agent configured as Spec VP.

    Raises:
        ImportError: If crewai is not available.

    Example:
        >>> spec_vp = create_spec_vp_agent()
        >>> print(spec_vp.role)
        'VP of Specifications'
    """
    from crewai import Agent

    persona = get_persona("spec_vp", include_memory=True)
    return Agent(
        role=persona.get("role", "VP of Specifications"),
        goal=persona.get("goal", "Oversee product definition and specification quality"),
        backstory=persona.get(
            "backstory",
            "You are the VP of Specifications, responsible for all product "
            "definition work. You ensure specifications are clear, complete, "
            "and aligned with business objectives.",
        ),
        verbose=True,
        allow_delegation=True,
    )


def create_exec_vp_agent():
    """Create Exec VP agent for overseeing implementation work.

    Exec VP (Vice President of Execution) is responsible for overseeing
    implementation and delivery. Reports to CEO and manages ace_exec
    and squad leads.

    Returns:
        CrewAI Agent configured as Exec VP.

    Raises:
        ImportError: If crewai is not available.

    Example:
        >>> exec_vp = create_exec_vp_agent()
        >>> print(exec_vp.role)
        'VP of Execution'
    """
    from crewai import Agent

    persona = get_persona("exec_vp", include_memory=True)
    return Agent(
        role=persona.get("role", "VP of Execution"),
        goal=persona.get("goal", "Oversee implementation and quality assurance"),
        backstory=persona.get(
            "backstory",
            "You are the VP of Execution, responsible for all implementation "
            "work. You coordinate development teams and QA to deliver "
            "high-quality software.",
        ),
        verbose=True,
        allow_delegation=True,
    )


# NOTE: create_qa_vp_agent() removed - qa_vp was eliminated from architecture
# exec_vp now handles both execution AND QA oversight (TASK-VP-003)


# =============================================================================
# P1-4 FIX: Hierarchy Accessor Functions for Observability (2026-02-01)
# =============================================================================
# These functions expose hierarchy information to the workflow for logging
# and debugging purposes. Previously HierarchyMapper was internal-only.

# Hierarchy paths by crew type (for observability)
CREW_HIERARCHY_PATHS: dict[str, list[str]] = {
    "spec": ["ceo", "spec_vp", "spec_master"],
    "exec": ["ceo", "exec_vp", "ace_exec"],
    "qa": ["ceo", "exec_vp", "qa_master"],
    "review": ["ceo", "spec_vp", "spec_master"],
    "rework": ["ceo", "exec_vp", "ace_exec"],
}


def get_hierarchy_mapper() -> HierarchyMapper:
    """Get a HierarchyMapper instance for observability.

    P1-4 FIX: Exposes HierarchyMapper to workflow for logging and debugging.
    Previously the mapper was internal-only, making it impossible to track
    which agents were executing what.

    Returns:
        HierarchyMapper instance.

    Example:
        >>> mapper = get_hierarchy_mapper()
        >>> agent_def = mapper.create_agent_definition("spec_master")
        >>> print(agent_def.role)
    """
    return HierarchyMapper()


def get_crew_hierarchy_path(crew_type: str) -> list[str]:
    """Get the hierarchy path for a crew type.

    P1-4 FIX: Returns the agent hierarchy path for logging.
    This enables workflow nodes to report which agents are involved
    in each operation.

    Args:
        crew_type: Type of crew ("spec", "exec", "qa", "review", "rework").

    Returns:
        List of agent IDs in hierarchy order (top to bottom).

    Example:
        >>> path = get_crew_hierarchy_path("spec")
        >>> print(path)
        ['ceo', 'spec_vp', 'spec_master']
    """
    return CREW_HIERARCHY_PATHS.get(crew_type, ["unknown"])


def get_all_crew_hierarchies() -> dict[str, list[str]]:
    """Get all crew hierarchy paths.

    Returns:
        Dictionary mapping crew_type to hierarchy path.
    """
    return CREW_HIERARCHY_PATHS.copy()
