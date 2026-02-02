"""DOC-019: Canonical agent hierarchy definition for Pipeline Autônomo.

This module defines the organizational structure of all agents in the pipeline.
All agents must use this as the single source of truth for hierarchy relationships,
communication permissions, and escalation paths.

CRITICAL IMPORT BOUNDARY (GAP-5-006):
=====================================
DO NOT import from other pipeline_autonomo modules (except standard library).
This module MUST remain a pure data module to prevent circular import risks.

Import Chain Analysis:
  hierarchy.py (Level 0 - Pure Data)
  ↓
  ipc.py (Level 1 - Imports from hierarchy)
  ↓
  agent_bus.py (Level 2 - Imports from ipc)

Rationale:
- hierarchy.py is imported by ipc.py, which is imported by agent_bus.py
- If hierarchy.py imports from ipc.py or agent_bus.py, circular import deadlock occurs
- Keep this module dependency-free to maintain a clean import hierarchy

Allowed Imports:
- typing (standard library)
- datetime (only if needed, currently imported in escalate_message)

Forbidden Imports:
- .ipc
- .agent_bus
- Any other pipeline_autonomo module

HIERARCHY MAP (Complete):

Level 0 - System Controllers (above business hierarchy):
  Wave2-Falha5 FIX: Clear separation of responsibilities:

  - pack_driver: PIPELINE ORCHESTRATOR
    * Primary executor of pipeline flow
    * Dispatches tasks to agents
    * Manages phase transitions (SPEC → PLAN → EXEC → QA → VOTE)
    * Authority: Execution flow, task assignment

  - ops_ctrl: OPERATIONS WATCHDOG
    * Health monitoring (heartbeats, liveness)
    * HA coordination (leader election)
    * Safe halt enforcement
    * Authority: Health, safety, HA failover

  - run_master: PIPELINE CARETAKER
    * Self-improvement protocols
    * Fine-tuning and bug fixes
    * Reports directly to USER
    * Authority: Pipeline maintenance, self-updates

  CONFLICT RESOLUTION:
    - pack_driver controls "what runs"
    - ops_ctrl controls "if it can run" (health veto)
    - run_master controls "how it should evolve" (maintenance)
    - If ops_ctrl detects critical failure → SAFE_HALT (overrides pack_driver)
    - If pack_driver and ops_ctrl disagree → ops_ctrl wins (safety first)

Level 1 - Business Top (BIG PICTURE):
  - presidente: Reports to pack_driver, watches big picture, keeps CEO on track
  - ONLY communicates with CEO (single point of contact)

Level 2 - CEO (MID PICTURE):
  - ceo: Reports to presidente, orchestrates the "battlefield"
  - ONLY one who talks to presidente
  - Leads execution, coordinates VPs

Level 3 - VPs (Vice Presidents):
  - spec_vp: Reports to CEO, specification VP, oversees product definition
  - exec_vp: Reports to CEO, execution VP, oversees implementation

Level 4 - Masters:
  - spec_master: Reports to spec_vp, manages spec squads
  - sprint_planner: Reports to spec_vp
  - ace_exec: Reports to exec_vp, manages execution squads
  - qa_master: Reports to exec_vp, manages QA (under exec for quality control)

Level 5 - Squad Leads:
  - squad_lead_ops, squad_lead_sec: Report to ace_exec
  - squad_lead_p1 through squad_lead_p5: Report to ace_exec
  - squad_lead_hlx: Reports to ace_exec (Human Layer squad)

Level 6 - Worker Agents:
  - spec_team: product_owner, project_manager, task_decomposer (under spec_master)
  - qa_team: auditor, agente_auditor, refinador, etc. (under qa_master)
  - ops_team: technical_planner, dependency_mapper (under squad_lead_ops)
  - sec_team: red_team_agent (under squad_lead_sec)
  - squad workers: worker_p1_01, etc. (under respective squad_leads)

DESIGN PRINCIPLES:
1. Lowercase agent_ids for Python naming consistency
2. Clear supervisor relationship: every agent has exactly one supervisor
3. Subordinates tracked for delegation and escalation
4. Hierarchical levels for organizational clarity
5. No peer-to-peer communication (must escalate to common supervisor)

PEER DEFINITION (Wave3-Melhoria1):
A "peer" is defined as:
  - Agents with the SAME supervisor AND the SAME hierarchical level
  - Examples of peers:
    * auditor, refinador, clean_reviewer (all under qa_master, level 5)
    * squad_lead_p1, squad_lead_p2 (all under ace_exec, level 4)
  - NOT peers:
    * qa_master, ace_exec (different supervisors: exec_vp)
    * spec_master, ace_exec (different supervisors: spec_vp vs exec_vp)
  - Peers CANNOT communicate directly - must escalate to shared supervisor

COMMUNICATION RULES (IMP-001):
1. System orchestrators (level 0) can send to any agent
2. Supervisor can send to direct subordinates
3. Subordinate can send to direct supervisor
4. Peers cannot communicate directly (must escalate to supervisor)
5. All unknown agents reject messages

ESCALATION RULES:
1. If A cannot send to B directly, escalate to A's supervisor
2. Message is forwarded up the chain until common supervisor found
3. Escalation count is tracked in message metadata
4. Max escalation depth prevents infinite loops

LEVEL 0 CONFLICT RESOLUTION (Wave2-Falha5):
1. pack_driver: Controls execution flow and task dispatch
2. ops_ctrl: Controls health monitoring and safety
3. run_master: Controls self-improvement and maintenance
4. In case of conflict: ops_ctrl has veto power (safety first)
5. ops_ctrl can issue SAFE_HALT that overrides pack_driver
6. run_master reports to USER, operates independently for fine-tuning

USE CASES:
- Agent validation: validate_agent_exists(agent_id)
- Permission checks: can_send_to(from_id, to_id), can_receive_from(agent_id, sender_id)
- Supervisor lookup: get_supervisor(agent_id)
- Reporting structure: get_subordinates(agent_id)
- Escalation path: escalate_message(msg, current_agent)
- Organizational reporting: get_level(agent_id)

CRITICAL NOTES (IMP-16):
- ONLY CEO communicates with PRESIDENTE (single point of contact)
- Presidente watches BIG PICTURE, CEO manages MID PICTURE
- Each layer only knows what it needs to know, reports up filtered info
- This structure preserves context windows for AI agents

Author: Pipeline Autonomo Team
Version: 1.0.0 - DOC-019 (2025-12-24)
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional


# ============================================================================
# Wave3-Melhoria3: Worker Spawn Limits
# ============================================================================
# These limits prevent resource exhaustion from runaway worker spawning

WORKER_LIMITS = {
    # Maximum concurrent workers per squad
    "max_concurrent_per_squad": 5,

    # Maximum total workers across all squads
    "max_total_workers": 20,

    # Maximum retries per worker before escalation
    "max_retries_per_worker": 3,

    # Worker timeout in seconds (5 minutes default)
    "timeout_seconds": 300,

    # Memory limit per worker in MB (0 = no limit)
    "memory_limit_mb": 512,
}


def get_worker_limit(limit_name: str) -> int:
    """Get a worker limit value by name.

    Args:
        limit_name: One of the keys in WORKER_LIMITS

    Returns:
        The limit value, or 0 if not found
    """
    return WORKER_LIMITS.get(limit_name, 0)

# Canonical hierarchy map with lowercase agent_ids
HIERARCHY_MAP: Dict[str, Dict[str, Any]] = {
    # Level 0 - System orchestrators (above business hierarchy)
    "pack_driver": {
        "level": 0,
        "supervisor": None,  # No supervisor - system controller
        "subordinates": ["presidente", "system_observer"],  # System orchestration
        "virtual": True,  # Implemented as workflow function in orchestrator, not CrewAI agent
    },
    "ops_ctrl": {
        "level": 0,
        "supervisor": None,  # No supervisor - system controller
        "subordinates": ["presidente"],  # Operations control authority
        "virtual": True,  # Implemented as SafeHaltController class, not CrewAI agent
    },
    # Wave1-Amb4 FIX: Run Master - pipeline caretaker, reports to user (supervisor=None)
    "run_master": {
        "level": 0,
        "supervisor": None,  # Reports to user - operates with absolute autonomy
        "subordinates": [],  # Monitors and fixes pipeline, but doesn't directly manage agents
    },
    # Level 1 - Top of business hierarchy (BIG PICTURE)
    "presidente": {
        "level": 1,
        "supervisor": "pack_driver",  # Reports to system orchestrator
        "subordinates": ["ceo"],  # ONLY CEO talks to presidente
    },
    # Level 2 - CEO (MID PICTURE) - Orchestrates the battlefield
    "ceo": {
        "level": 2,
        "supervisor": "presidente",  # Only one who reports to presidente
        "subordinates": [
            "spec_vp",
            "exec_vp",
            # qa_vp removed - exec_vp handles both execution AND QA oversight
            "human_approver",
            "arbiter",  # REVISED: Conflict resolution
            "retrospective_master",  # REVISED: Lessons learned
            "integration_officer",  # REVISED: Cross-squad coordination
        ],
    },
    # Level 3 - VPs
    "spec_vp": {
        "level": 3,
        "supervisor": "ceo",  # Reports to CEO
        "subordinates": ["spec_master", "sprint_planner"],
    },
    "exec_vp": {
        "level": 3,
        "supervisor": "ceo",  # Reports to CEO
        "subordinates": [
            "ace_exec",
            "qa_master",  # QA under exec_vp for unified execution/quality control
            "external_liaison",  # REVISED: External vendors
        ],
    },
    # Level 4 - Masters
    "spec_master": {
        "level": 4,
        "supervisor": "spec_vp",
        "subordinates": ["product_owner", "project_manager", "task_decomposer"],
    },
    "sprint_planner": {
        "level": 4,
        "supervisor": "spec_vp",
        "subordinates": [],
    },
    "ace_exec": {
        "level": 4,
        "supervisor": "exec_vp",
        "subordinates": [  # Manages all execution squads
            "squad_lead_ops",
            "squad_lead_sec",
            "squad_lead_p1",
            "squad_lead_p2",
            "squad_lead_p3",
            "squad_lead_p4",
            "squad_lead_p5",
            "squad_lead_hlx",
        ],
    },
    "qa_master": {
        "level": 4,
        "supervisor": "exec_vp",  # Reports to exec_vp for unified execution/QA oversight
        "subordinates": [
            "auditor",
            "agente_auditor",
            "refinador",
            "clean_reviewer",
            "edge_case_hunter",
            "gap_hunter",
            "human_reviewer",  # Human Layer worker
            "debt_tracker",  # REVISED: Tech debt tracking
        ],
    },
    # Level 5 - Squad Leads (all report to ace_exec)
    "squad_lead_ops": {
        "level": 5,
        "supervisor": "ace_exec",
        "subordinates": [
            "technical_planner",
            "dependency_mapper",
            # L6 Specialists (F-160 FIX)
            "llm_orchestrator",
            "oracle_architect",
            "data_engineer",
        ],
    },
    "squad_lead_sec": {
        "level": 5,
        "supervisor": "ace_exec",
        "subordinates": [
            "red_team_agent",
            # L6 Specialists (F-160 FIX)
            "blockchain_engineer",
        ],
    },
    "squad_lead_p1": {
        "level": 5,
        "supervisor": "ace_exec",
        "subordinates": ["worker_p1_01", "worker_p1_02"],
    },
    "squad_lead_p2": {
        "level": 5,
        "supervisor": "ace_exec",
        "subordinates": ["worker_p2_01", "worker_p2_02"],
    },
    "squad_lead_p3": {
        "level": 5,
        "supervisor": "ace_exec",
        "subordinates": ["worker_p3_01", "worker_p3_02"],
    },
    "squad_lead_p4": {
        "level": 5,
        "supervisor": "ace_exec",
        "subordinates": ["worker_p4_01", "worker_p4_02"],
    },
    "squad_lead_p5": {
        "level": 5,
        "supervisor": "ace_exec",
        "subordinates": ["worker_p5_01", "worker_p5_02"],
    },
    "squad_lead_hlx": {
        "level": 5,
        "supervisor": "ace_exec",
        "subordinates": [
            "human_layer_specialist",  # REVISED: HLX workers
            # L6 Specialists (F-160 FIX)
            "ui_designer",
            "ux_researcher",
            "legal_tech_specialist",
            "web3_frontend",
        ],
    },
    # Level 6 - Workers (Spec team)
    "product_owner": {
        "level": 6,
        "supervisor": "spec_master",
        "subordinates": [],
    },
    "project_manager": {
        "level": 6,
        "supervisor": "spec_master",
        "subordinates": [],
    },
    "task_decomposer": {
        "level": 6,
        "supervisor": "spec_master",
        "subordinates": [],
    },
    # Level 6 - Workers (QA team)
    "auditor": {
        "level": 6,
        "supervisor": "qa_master",
        "subordinates": [],
    },
    "agente_auditor": {
        "level": 6,
        "supervisor": "qa_master",
        "subordinates": [],
    },
    "refinador": {
        "level": 6,
        "supervisor": "qa_master",
        "subordinates": [],
    },
    "clean_reviewer": {
        "level": 6,
        "supervisor": "qa_master",
        "subordinates": [],
    },
    "edge_case_hunter": {
        "level": 6,
        "supervisor": "qa_master",
        "subordinates": [],
    },
    "gap_hunter": {
        "level": 6,
        "supervisor": "qa_master",
        "subordinates": [],
    },
    # Level 6 - Workers (Ops team)
    "technical_planner": {
        "level": 6,
        "supervisor": "squad_lead_ops",
        "subordinates": [],
    },
    "dependency_mapper": {
        "level": 6,
        "supervisor": "squad_lead_ops",
        "subordinates": [],
    },
    # Level 6 - Workers (Security team)
    "red_team_agent": {
        "level": 6,
        "supervisor": "squad_lead_sec",
        "subordinates": [],
    },
    # Level 3 - Human Layer Agents (reports to CEO for independence)
    "human_approver": {
        "level": 3,
        "supervisor": "ceo",
        "subordinates": [],
        "virtual": True,  # Implemented as human_layer/ module, not CrewAI agent
    },
    # Level 6 - Human Layer Agents (worker level)
    "human_reviewer": {
        "level": 6,
        "supervisor": "qa_master",
        "subordinates": [],
    },
    # NEW AGENTS (REVISED Implementation 2025-12-26)
    # Level 2 - Executive Support (reports to CEO)
    "arbiter": {
        "level": 2,
        "supervisor": "ceo",
        "subordinates": [],
        "virtual": True,  # Implemented as conflict_resolver.py module, not CrewAI agent
    },
    "retrospective_master": {
        "level": 2,
        "supervisor": "ceo",
        "subordinates": [],
        "virtual": True,  # Conceptual - no implementation exists yet
    },
    # Level 3 - VP Level Support
    "integration_officer": {
        "level": 3,
        "supervisor": "ceo",
        "subordinates": [],
    },
    "external_liaison": {
        "level": 3,
        "supervisor": "exec_vp",
        "subordinates": [],
        "virtual": True,  # Conceptual - no implementation exists yet
    },
    # Level 0 - System Controllers (Operations Team)
    "resource_optimizer": {
        "level": 0,
        "supervisor": "ops_ctrl",  # FIX: Reports to operations controller
        "subordinates": [],
        "virtual": True,  # Implemented as TokenBudget class, not CrewAI agent
    },
    "system_observer": {
        "level": 0,  # FIX: Same level as other L0 system agents
        "supervisor": "ops_ctrl",  # FIX: Reports to operations controller
        "subordinates": [],
        "virtual": True,  # Conceptual - no implementation exists yet
    },
    # Level 4 - QA Specialists
    "debt_tracker": {
        "level": 4,
        "supervisor": "qa_master",
        "subordinates": [],
    },
    "human_layer_specialist": {
        "level": 5,
        "supervisor": "squad_lead_hlx",
        "subordinates": [],
    },
    # Level 6 - Workers (P1 Squad)
    "worker_p1_01": {
        "level": 6,
        "supervisor": "squad_lead_p1",
        "subordinates": [],
    },
    "worker_p1_02": {
        "level": 6,
        "supervisor": "squad_lead_p1",
        "subordinates": [],
    },
    # Level 6 - Workers (P2 Squad)
    "worker_p2_01": {
        "level": 6,
        "supervisor": "squad_lead_p2",
        "subordinates": [],
    },
    "worker_p2_02": {
        "level": 6,
        "supervisor": "squad_lead_p2",
        "subordinates": [],
    },
    # Level 6 - Workers (P3 Squad)
    "worker_p3_01": {
        "level": 6,
        "supervisor": "squad_lead_p3",
        "subordinates": [],
    },
    "worker_p3_02": {
        "level": 6,
        "supervisor": "squad_lead_p3",
        "subordinates": [],
    },
    # Level 6 - Workers (P4 Squad)
    "worker_p4_01": {
        "level": 6,
        "supervisor": "squad_lead_p4",
        "subordinates": [],
    },
    "worker_p4_02": {
        "level": 6,
        "supervisor": "squad_lead_p4",
        "subordinates": [],
    },
    # Level 6 - Workers (P5 Squad)
    "worker_p5_01": {
        "level": 6,
        "supervisor": "squad_lead_p5",
        "subordinates": [],
    },
    "worker_p5_02": {
        "level": 6,
        "supervisor": "squad_lead_p5",
        "subordinates": [],
    },
    # =========================================================================
    # Level 6 - L6 Specialists (F-160 FIX: Add canonical specialists)
    # =========================================================================
    # These specialists are invoked on-demand when their expertise is needed.
    # They report to appropriate squad leads based on their domain.
    # See: docs/Agents/L6_Specialists/ for full definitions.
    "llm_orchestrator": {
        "level": 6,
        "supervisor": "squad_lead_ops",  # Technical operations expertise
        "subordinates": [],
    },
    "ui_designer": {
        "level": 6,
        "supervisor": "squad_lead_hlx",  # Human-facing design
        "subordinates": [],
    },
    "ux_researcher": {
        "level": 6,
        "supervisor": "squad_lead_hlx",  # User experience research
        "subordinates": [],
    },
    "oracle_architect": {
        "level": 6,
        "supervisor": "squad_lead_ops",  # Technical architecture
        "subordinates": [],
    },
    "blockchain_engineer": {
        "level": 6,
        "supervisor": "squad_lead_sec",  # Security-sensitive blockchain
        "subordinates": [],
    },
    "data_engineer": {
        "level": 6,
        "supervisor": "squad_lead_ops",  # Data operations
        "subordinates": [],
    },
    "legal_tech_specialist": {
        "level": 6,
        "supervisor": "squad_lead_hlx",  # Human/legal layer
        "subordinates": [],
    },
    "web3_frontend": {
        "level": 6,
        "supervisor": "squad_lead_hlx",  # Frontend/UX domain
        "subordinates": [],
    },
    # 2026-01-10 FIX: Add missing agents that exist in SIGNOFF_HIERARCHY
    # These agents have Cerebro files but were missing from canonical hierarchy
    "orchestrator": {
        "level": 0,
        "supervisor": None,  # System-level agent
        "subordinates": [],
    },
    "run_supervisor": {
        "level": 0,
        "supervisor": None,  # System-level agent
        "subordinates": [],
        "virtual": True,  # Conceptual - no implementation exists yet
    },
    "ace_orchestration": {
        "level": 4,
        "supervisor": "ace_exec",  # Execution orchestration
        "subordinates": [],
    },
    "human_layer": {
        "level": 4,
        "supervisor": "squad_lead_hlx",  # Human layer coordination
        "subordinates": [],
    },
}

# Dynamic agent registry for runtime-registered agents (e.g., squad workers)
# GAP-11-010: Dynamic hierarchy registration for workers spawned at runtime
_DYNAMIC_AGENTS: Dict[str, Dict[str, Any]] = {}
# Thread-safety lock for concurrent access to _DYNAMIC_AGENTS
# AUDIT-2025-12-27: Added to prevent race conditions in multi-threaded scenarios
_DYNAMIC_AGENTS_LOCK = threading.RLock()

# BLACK TEAM FIX CHAOS-008: Limits and TTL for dynamic agents
MAX_DYNAMIC_AGENTS = 500  # Prevent unbounded growth
DYNAMIC_AGENT_TTL_SECONDS = 3600  # 1 hour - agents older than this are candidates for cleanup


def _cleanup_expired_dynamic_agents_locked() -> int:
    """Remove expired dynamic agents from registry.

    BLACK TEAM FIX CHAOS-008: Prevents unbounded growth of _DYNAMIC_AGENTS.

    MUST be called while holding _DYNAMIC_AGENTS_LOCK.

    Returns:
        Number of agents removed.
    """
    now = time.time()
    expired = [
        agent_id
        for agent_id, info in _DYNAMIC_AGENTS.items()
        if now - info.get("registered_at", 0) > DYNAMIC_AGENT_TTL_SECONDS
    ]
    for agent_id in expired:
        del _DYNAMIC_AGENTS[agent_id]
    return len(expired)


def cleanup_dynamic_agents() -> int:
    """Public API to cleanup expired dynamic agents.

    BLACK TEAM FIX CHAOS-008: Can be called periodically by external
    processes to prevent registry pollution.

    Returns:
        Number of agents removed.
    """
    with _DYNAMIC_AGENTS_LOCK:
        return _cleanup_expired_dynamic_agents_locked()


def _get_dynamic_agent_if_valid(agent_id: str) -> Optional[Dict[str, Any]]:
    """Get a dynamic agent's info if it exists and hasn't expired.

    DRIFT-004 FIX: Helper function that checks TTL on access.

    MUST be called while holding _DYNAMIC_AGENTS_LOCK.

    Args:
        agent_id: The agent identifier to look up

    Returns:
        Agent info dict if valid, None if not found or expired.
    """
    if agent_id not in _DYNAMIC_AGENTS:
        return None

    agent_info = _DYNAMIC_AGENTS[agent_id]
    registered_at = agent_info.get("registered_at", 0)

    if time.time() - registered_at > DYNAMIC_AGENT_TTL_SECONDS:
        # Agent expired - remove it (lazy cleanup)
        del _DYNAMIC_AGENTS[agent_id]
        return None

    return agent_info


def refresh_dynamic_agent_ttl(agent_id: str) -> bool:
    """Refresh the TTL for a dynamic agent (heartbeat-like behavior).

    DRIFT-004 FIX: Allows agents to stay alive by refreshing their TTL.

    Args:
        agent_id: The agent identifier

    Returns:
        True if agent was found and refreshed, False otherwise.
    """
    with _DYNAMIC_AGENTS_LOCK:
        if agent_id not in _DYNAMIC_AGENTS:
            return False
        _DYNAMIC_AGENTS[agent_id]["registered_at"] = time.time()
        return True


def register_dynamic_agent(
    agent_id: str,
    level: int,
    supervisor: str,
    subordinates: Optional[List[str]] = None,
    registrar_id: Optional[str] = None,
) -> None:
    """Register a dynamic agent in the runtime hierarchy.

    This is used for agents that are spawned dynamically at runtime,
    such as squad workers, which are not in the static HIERARCHY_MAP.

    RED TEAM FIX CRIT-002 (RED-12): Now validates registrar authority.

    Args:
        agent_id: The agent identifier (lowercase)
        level: Hierarchical level (typically 5 for workers)
        supervisor: The supervisor's agent_id
        subordinates: List of subordinate agent_ids (default: empty list)
        registrar_id: The agent performing the registration (for authority check)

    Raises:
        ValueError: If registrar lacks authority to create agent at specified level.

    Example:
        >>> register_dynamic_agent("ops_worker_task_001_abc123", 5, "squad_lead_ops", registrar_id="squad_lead_ops")
        >>> validate_agent_exists("ops_worker_task_001_abc123")
        True
    """
    if subordinates is None:
        subordinates = []

    # RED TEAM FIX CRIT-002 (RED-12): Validate registrar authority
    if registrar_id:
        registrar_level = get_level(registrar_id)
        if registrar_level is None:
            raise ValueError(
                f"CRIT-002: Unknown registrar {registrar_id}. Cannot register dynamic agent."
            )
        # Registrar can only create agents at LOWER privilege (higher level number)
        if level <= registrar_level:
            raise ValueError(
                f"CRIT-002: {registrar_id} (L{registrar_level}) cannot create agent at L{level}. "
                f"Agents can only create subordinates at lower privilege levels."
            )

    # Validate supervisor exists and is at appropriate level
    supervisor_level = get_level(supervisor)
    if supervisor_level is None:
        raise ValueError(f"CRIT-002: Unknown supervisor {supervisor}")
    if level <= supervisor_level:
        raise ValueError(
            f"CRIT-002: Dynamic agent level {level} must be lower privilege than supervisor L{supervisor_level}"
        )

    # Validate agent_id format to prevent path traversal or spoofing
    import re
    if not re.match(r'^[a-z][a-z0-9_]*$', agent_id):
        raise ValueError(
            f"CRIT-002: Invalid agent_id format: {agent_id}. "
            "Must be lowercase alphanumeric with underscores."
        )

    # Prevent shadowing static agents
    if agent_id in HIERARCHY_MAP:
        raise ValueError(
            f"CRIT-002: Cannot shadow static agent {agent_id}"
        )

    with _DYNAMIC_AGENTS_LOCK:
        # BLACK TEAM FIX CHAOS-008: Enforce max limit
        if len(_DYNAMIC_AGENTS) >= MAX_DYNAMIC_AGENTS:
            # Run cleanup first
            _cleanup_expired_dynamic_agents_locked()
            # If still at limit, reject
            if len(_DYNAMIC_AGENTS) >= MAX_DYNAMIC_AGENTS:
                raise ValueError(
                    f"CHAOS-008: Maximum dynamic agents ({MAX_DYNAMIC_AGENTS}) reached. "
                    "Cannot register more agents until cleanup."
                )

        _DYNAMIC_AGENTS[agent_id] = {
            "level": level,
            "supervisor": supervisor,
            "subordinates": subordinates,
            "registered_at": time.time(),  # BLACK TEAM FIX CHAOS-008: Track registration time
        }


def unregister_dynamic_agent(agent_id: str) -> None:
    """Unregister a dynamic agent from the runtime hierarchy.

    Should be called when a dynamic agent (e.g., worker) is terminated.

    Args:
        agent_id: The agent identifier to unregister

    Example:
        >>> unregister_dynamic_agent("ops_worker_task_001_abc123")
        >>> validate_agent_exists("ops_worker_task_001_abc123")
        False
    """
    with _DYNAMIC_AGENTS_LOCK:
        if agent_id in _DYNAMIC_AGENTS:
            del _DYNAMIC_AGENTS[agent_id]


def get_dynamic_agents() -> Dict[str, Dict[str, Any]]:
    """Return copy of all registered dynamic agents.

    Returns:
        Dictionary mapping agent_id to hierarchy info
    """
    with _DYNAMIC_AGENTS_LOCK:
        return _DYNAMIC_AGENTS.copy()


def get_supervisor(agent_id: str, check_ttl: bool = True) -> Optional[str]:
    """Return the supervisor agent_id for the given agent, or None if top-level.

    DRIFT-004 FIX: Now checks TTL for dynamic agents.

    Args:
        agent_id: The agent identifier (lowercase)
        check_ttl: If True, verify dynamic agents haven't expired (default True)

    Returns:
        The supervisor's agent_id, or None for top-level agents

    Example:
        >>> get_supervisor("qa_master")
        'presidente'
        >>> get_supervisor("presidente")
        None
    """
    # Check static hierarchy first
    hierarchy = HIERARCHY_MAP.get(agent_id)
    if not hierarchy:
        # Check dynamic agents with TTL check
        with _DYNAMIC_AGENTS_LOCK:
            if check_ttl:
                hierarchy = _get_dynamic_agent_if_valid(agent_id)
            else:
                hierarchy = _DYNAMIC_AGENTS.get(agent_id)
    if not hierarchy:
        return None
    return hierarchy.get("supervisor")


def get_subordinates(agent_id: str, check_ttl: bool = True) -> List[str]:
    """Return list of direct subordinate agent_ids for the given agent.

    DRIFT-004 FIX: Now checks TTL for dynamic agents.

    Args:
        agent_id: The agent identifier (lowercase)
        check_ttl: If True, verify dynamic agents haven't expired (default True)

    Returns:
        List of subordinate agent_ids (may be empty)

    Example:
        >>> get_subordinates("qa_master")
        ['auditor', 'agente_auditor', 'refinador', 'clean_reviewer', 'edge_case_hunter', 'gap_hunter']
    """
    # Check static hierarchy first
    hierarchy = HIERARCHY_MAP.get(agent_id)
    if not hierarchy:
        # Check dynamic agents with TTL check
        with _DYNAMIC_AGENTS_LOCK:
            if check_ttl:
                hierarchy = _get_dynamic_agent_if_valid(agent_id)
            else:
                hierarchy = _DYNAMIC_AGENTS.get(agent_id)
    if not hierarchy:
        return []
    subordinates = hierarchy.get("subordinates", [])
    return subordinates if isinstance(subordinates, list) else []


def get_level(agent_id: str, check_ttl: bool = True) -> Optional[int]:
    """Return the hierarchical level of the agent.

    DRIFT-004 FIX: Now checks TTL for dynamic agents.

    Args:
        agent_id: The agent identifier (lowercase)
        check_ttl: If True, verify dynamic agents haven't expired (default True)

    Returns:
        The hierarchy level (0-5), or None if agent not found

    Example:
        >>> get_level("presidente")
        1
        >>> get_level("qa_master")
        3
    """
    # Check static hierarchy first
    hierarchy = HIERARCHY_MAP.get(agent_id)
    if not hierarchy:
        # Check dynamic agents with TTL check
        with _DYNAMIC_AGENTS_LOCK:
            if check_ttl:
                hierarchy = _get_dynamic_agent_if_valid(agent_id)
            else:
                hierarchy = _DYNAMIC_AGENTS.get(agent_id)
    if not hierarchy:
        return None
    return hierarchy.get("level")


def are_peers(agent_a: str, agent_b: str) -> bool:
    """Wave3-Melhoria1: Check if two agents are peers.

    Peers are agents with:
    1. The SAME supervisor
    2. The SAME hierarchical level

    Peers cannot communicate directly - they must escalate to their
    shared supervisor.

    Args:
        agent_a: First agent ID
        agent_b: Second agent ID

    Returns:
        True if agents are peers, False otherwise

    Example:
        >>> are_peers("auditor", "refinador")
        True  # Both under qa_master, same level
        >>> are_peers("qa_master", "ace_exec")
        False  # Same level but different supervisors
        >>> are_peers("squad_lead_p1", "squad_lead_p2")
        True  # Both under ace_exec, same level
    """
    if agent_a == agent_b:
        return False  # Same agent is not a peer

    # Get supervisors
    sup_a = get_supervisor(agent_a)
    sup_b = get_supervisor(agent_b)

    # Must have same supervisor
    if sup_a != sup_b or sup_a is None:
        return False

    # Must have same level
    level_a = get_level(agent_a)
    level_b = get_level(agent_b)

    if level_a != level_b or level_a is None:
        return False

    return True


def can_send_to(from_agent: str, to_agent: str) -> bool:
    """Verify if from_agent can send a message to to_agent.

    Communication rules:
    1. System orchestrator (level 0) can send to any known agent
    2. Supervisor can send to direct subordinates
    3. Subordinate can send to direct supervisor
    4. Peers cannot communicate directly (must escalate)
    5. Unknown agents cannot send messages
    6. Self-messages are always allowed

    Args:
        from_agent: ID of the sending agent
        to_agent: ID of the receiving agent

    Returns:
        True if communication is permitted, False otherwise

    Example:
        >>> can_send_to("pack_driver", "qa_master")
        True  # System orchestrator can send to any agent
        >>> can_send_to("presidente", "qa_master")
        True
        >>> can_send_to("qa_master", "presidente")
        True
        >>> can_send_to("auditor", "refinador")
        False  # Peers cannot communicate directly
    """
    # Same agent always allowed (self-message)
    if from_agent == to_agent:
        return True

    # Unknown agents cannot send
    with _DYNAMIC_AGENTS_LOCK:
        from_in_dynamic = from_agent in _DYNAMIC_AGENTS
        to_in_dynamic = to_agent in _DYNAMIC_AGENTS

    if from_agent not in HIERARCHY_MAP and not from_in_dynamic:
        return False

    # Unknown recipients not allowed
    if to_agent not in HIERARCHY_MAP and not to_in_dynamic:
        return False

    # System orchestrator (level 0) can send to any known agent
    from_level = get_level(from_agent)
    if from_level == 0:
        return True

    # Check if to_agent is a direct subordinate
    subordinates = get_subordinates(from_agent)
    if to_agent in subordinates:
        return True

    # Check if to_agent is the direct supervisor
    supervisor = get_supervisor(from_agent)
    if supervisor == to_agent:
        return True

    # Otherwise, not allowed (peers or non-direct relationships)
    return False


def can_receive_from(agent_id: str, sender_id: str) -> bool:
    """Verify if agent_id can receive messages from sender_id.

    This is the inverse of can_send_to, useful for inbox validation.

    Args:
        agent_id: ID of the receiving agent
        sender_id: ID of the sending agent

    Returns:
        True if the agent can receive from sender, False otherwise

    Example:
        >>> can_receive_from("qa_master", "pack_driver")
        True  # Any agent can receive from system orchestrator
        >>> can_receive_from("qa_master", "presidente")
        True
        >>> can_receive_from("qa_master", "auditor")
        True
    """
    # Self-messages allowed
    if sender_id == agent_id:
        return True

    # Unknown sender not allowed
    with _DYNAMIC_AGENTS_LOCK:
        sender_in_dynamic = sender_id in _DYNAMIC_AGENTS
    if sender_id not in HIERARCHY_MAP and not sender_in_dynamic:
        return False

    # System orchestrator (level 0) can send to any agent
    sender_level = get_level(sender_id)
    if sender_level == 0:
        return True

    # Check if sender is direct supervisor
    supervisor = get_supervisor(agent_id)
    if supervisor == sender_id:
        return True

    # Check if sender is subordinate (for responses/delegations)
    subordinates = get_subordinates(agent_id)
    if sender_id in subordinates:
        return True

    return False


def validate_agent_exists(agent_id: str, check_ttl: bool = True) -> bool:
    """Check if an agent exists in the hierarchy.

    Checks both static HIERARCHY_MAP and dynamic runtime agents.

    DRIFT-004 FIX: Now checks TTL by default and removes expired agents.

    Args:
        agent_id: The agent identifier to validate
        check_ttl: If True, verify dynamic agents haven't expired (default True)

    Returns:
        True if agent exists (and is not expired), False otherwise

    Example:
        >>> validate_agent_exists("qa_master")
        True
        >>> validate_agent_exists("fake_agent")
        False
    """
    if agent_id in HIERARCHY_MAP:
        return True
    with _DYNAMIC_AGENTS_LOCK:
        if agent_id not in _DYNAMIC_AGENTS:
            return False

        # DRIFT-004 FIX: Check TTL on access
        if check_ttl:
            agent_info = _DYNAMIC_AGENTS[agent_id]
            registered_at = agent_info.get("registered_at", 0)
            if time.time() - registered_at > DYNAMIC_AGENT_TTL_SECONDS:
                # Agent expired - remove it
                del _DYNAMIC_AGENTS[agent_id]
                return False

        return True


def escalate_message(message: Dict[str, Any], current_agent: str) -> Optional[str]:
    """Return the agent_id of the supervisor to escalate the message to.

    Used when an agent needs to forward a message it cannot send directly
    (e.g., peer-to-peer communication).

    Args:
        message: Dictionary with the message to be escalated
        current_agent: ID of the agent performing the escalation

    Returns:
        agent_id of the supervisor, or None if no supervisor exists

    Side effects:
        Updates message metadata to track escalation history

    Example:
        >>> msg = {"task": "review_code", "sender": "auditor"}
        >>> escalate_message(msg, "auditor")
        'qa_master'

    Raises:
        ValueError: If escalation path is invalid (IPC-005)
    """
    # IPC-005: Validate escalation path - agent must exist
    if not validate_agent_exists(current_agent):
        raise ValueError(f"IPC-005: Cannot escalate from unknown agent: {current_agent}")

    supervisor = get_supervisor(current_agent)
    if not supervisor:
        return None

    # IPC-005: Validate supervisor exists in hierarchy
    if not validate_agent_exists(supervisor):
        raise ValueError(f"IPC-005: Invalid escalation path - supervisor {supervisor} does not exist for {current_agent}")

    # Update message metadata to indicate escalation
    from datetime import datetime, timezone
    message["escalated_by"] = current_agent
    message["escalated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if "escalation_count" not in message:
        message["escalation_count"] = 0
    message["escalation_count"] += 1

    return supervisor


def is_virtual_agent(agent_id: str) -> bool:
    """Check if an agent is virtual (conceptual, not instantiated as CrewAI).

    Virtual agents are defined in the hierarchy for documentation and
    organizational purposes but are implemented as modules, functions,
    or classes rather than CrewAI agents.

    Examples of virtual agents:
    - pack_driver: Implemented as workflow function in orchestrator
    - ops_ctrl: Implemented as SafeHaltController class
    - arbiter: Implemented as conflict_resolver.py module
    - human_approver: Implemented as human_layer/ module

    Args:
        agent_id: The agent identifier (lowercase)

    Returns:
        True if agent is virtual (conceptual/non-CrewAI), False if real

    Example:
        >>> is_virtual_agent("pack_driver")
        True
        >>> is_virtual_agent("spec_master")
        False
    """
    # Check static hierarchy first
    hierarchy = HIERARCHY_MAP.get(agent_id)
    if not hierarchy:
        # Check dynamic agents (dynamic agents are never virtual)
        with _DYNAMIC_AGENTS_LOCK:
            if agent_id in _DYNAMIC_AGENTS:
                return False
        return False
    return hierarchy.get("virtual", False)
