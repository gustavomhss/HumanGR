# cockpit/cockpit_transformer.py
"""
Transform pipeline state to cockpit node format for HumanGR.
PURE FUNCTIONS - no side effects, no I/O (except through state_reader).

This module transforms the raw pipeline state files into the format
expected by the cockpit frontend (nodes with status, metrics, logs, etc.)
"""
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

from . import state_reader
from . import resilience_reader

# =============================================================================
# PHASE TO NODES MAPPING
# =============================================================================
# Maps pipeline phases to which nodes should be active during that phase
#
# LINEAR FLOW (real pipeline execution):
#   INIT → SPEC → EXEC → GATES → SIGNOFF → ARTIFACTS → DONE
#
# Each phase has GRANULAR sub-nodes showing who's working:
#   INIT:      L0 run_master initializes, loads context
#   SPEC:      L3 spec_master decomposes, L5 auditor/dep_mapper assist
#   EXEC:      L3 ace_exec → L4 squad_leads → L5 workers (RF-001 to RF-004)
#   GATES:     L3 qa_master runs G0-G8, reflexion on failures
#   SIGNOFF:   L2 exec_vp reviews, L1 ceo approves
#   ARTIFACTS: L0 generates docs, metrics, handoff
#   DONE:      sprint complete

PHASE_TO_NODES = {
    # INIT: Pipeline initialization
    "INIT": ["init", "load"],

    # SPEC: Spec decomposition (L2 Spec VP → L3 Spec Master → L5 Workers)
    "SPEC": ["specvp", "spec", "inv", "deps"],

    # PLAN: Sprint planning (from SPEC, creates execution plan)
    "PLAN": ["plan"],

    # EXEC: Implementation (L2 Exec VP → L4 Squad Leads → L5 Workers → L3 Ace Exec)
    "EXEC": ["execvp", "spawn", "assign", "exec1", "exec2", "exec3", "exec4", "test"],

    # QA: Quality validation - LangGraph phase name (L3 QA Master → L5 Gates → L6 Reflexion)
    "QA": ["qa", "g0", "g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8", "reflex"],

    # GATES: Alias for QA (legacy cockpit phase name)
    "GATES": ["qa", "g0", "g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8", "reflex"],

    # VOTE: Voting and approval - LangGraph phase name (L3 Merge → L5 Review → Rework loop)
    "VOTE": ["merge", "review", "rework"],

    # SIGNOFF: Alias for VOTE (legacy cockpit phase name)
    "SIGNOFF": ["merge", "review", "rework"],

    # ARTIFACTS: Documentation and metrics (L0 → L1 CEO Signoff)
    "ARTIFACTS": ["docs", "art", "metrics", "signoff"],

    # HANDOFF: Transition to next sprint
    "HANDOFF": ["handoff"],

    # DONE: Sprint complete
    "DONE": ["done"],

    # HALT: Safe halt state (error/stop)
    "HALT": [],
}

# Phase order for determining what's "before" current phase
# This matches the LangGraph SprintPhase enum: INIT → SPEC → PLAN → EXEC → QA → VOTE → DONE
# Also includes legacy cockpit phase names as aliases
PHASE_ORDER = [
    "INIT", "SPEC", "PLAN", "EXEC", "QA", "VOTE", "ARTIFACTS", "HANDOFF", "DONE", "HALT"
]

# Legacy phase name mapping (cockpit names → LangGraph names)
PHASE_ALIASES = {
    "GATES": "QA",
    "SIGNOFF": "VOTE",
}

# =============================================================================
# GATE TO NODE MAPPING
# =============================================================================

GATE_TO_NODE = {
    "G0": "g0", "G1": "g1", "G2": "g2", "G3": "g3",
    "G4": "g4", "G5": "g5", "G6": "g6", "G7": "g7", "G8": "g8",
}

NODE_TO_GATE = {v: k for k, v in GATE_TO_NODE.items()}

# =============================================================================
# STATUS MAPPING
# =============================================================================

STATUS_MAP = {
    # Pipeline statuses -> Cockpit statuses
    "PASS": "complete",
    "PASSED": "complete",
    "SUCCESS": "complete",
    "COMPLETE": "complete",
    "COMPLETED": "complete",
    "DONE": "complete",

    "FAIL": "error",
    "FAILED": "error",
    "ERROR": "error",
    "REJECTED": "rejected",
    "BLOCK": "error",
    "BLOCKED": "error",

    "WARN": "warning",
    "WARNING": "warning",
    "REWORK": "rejected",

    "RUNNING": "active",
    "ACTIVE": "active",
    "IN_PROGRESS": "active",
    "EXECUTING": "active",

    "PENDING": "pending",
    "WAITING": "pending",
    "QUEUED": "pending",
    "SKIPPED": "pending",
}

# =============================================================================
# ALL NODE IDS - Granular nodes matching visual flow
# =============================================================================

ALL_NODE_IDS = [
    # L0 - System
    "init", "load", "docs", "art", "metrics", "handoff", "done",
    # L1 - Executive
    "signoff",
    # L2 - VPs
    "specvp", "execvp",
    # L3 - Masters
    "spec", "plan", "test", "qa", "merge",
    # L4 - Squad Leads
    "spawn", "assign",
    # L5 - Workers
    "inv", "deps", "exec1", "exec2", "exec3", "exec4",
    "g0", "g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8",
    "review", "rework",
    # L6 - Specialists
    "reflex",
]

# =============================================================================
# NODE METADATA - Who works and what they do (with layer indicators)
# =============================================================================

NODE_TITLES = {
    # L0 - System (Run Master, Infrastructure)
    "init": "Run Master",
    "load": "Load Context",
    "docs": "Gen Docs",
    "art": "Artifacts",
    "metrics": "Metrics",
    "handoff": "Handoff",
    "done": "Done",

    # L1 - Executive (CEO)
    "signoff": "CEO Signoff",

    # L2 - VPs (Strategic Direction)
    "specvp": "Spec VP",
    "execvp": "Exec VP",

    # L3 - Masters (Tactical Coordination)
    "spec": "Spec Master",
    "plan": "Sprint Plan",
    "test": "Ace Exec",
    "qa": "QA Master",
    "merge": "Merge",

    # L4 - Squad Leads (Team Coordination)
    "spawn": "Spawn",
    "assign": "Assign",

    # L5 - Workers (Execution)
    "inv": "Auditor",
    "deps": "Dep Mapper",
    "exec1": "RF-001",
    "exec2": "RF-002",
    "exec3": "RF-003",
    "exec4": "RF-004",
    "g0": "G0",
    "g1": "G1",
    "g2": "G2",
    "g3": "G3",
    "g4": "G4",
    "g5": "G5",
    "g6": "G6",
    "g7": "G7",
    "g8": "G8",
    "review": "Reviewer",
    "rework": "Rework",

    # L6 - Specialists (Advanced/AI)
    "reflex": "Reflexion",
}

# =============================================================================
# NODE DESCRIPTIONS - Descricao detalhada de cada node para o popup
# =============================================================================

NODE_DESCRIPTIONS: Dict[str, str] = {
    # L0 - System
    "init": "Pipeline initialization. Run Master validates environment, creates workspace, and checks stack health.",
    "load": "Loads sprint context pack from context packs. Parses requirements (RFs), invariants (INVs), and edge cases.",
    "docs": "Documentation generation. Creates API docs, README updates, and changelog entries from code changes.",
    "art": "Artifact writer. Packages deliverables and generates final output files for the sprint.",
    "metrics": "Metrics collection. Aggregates test coverage, gate scores, and performance measurements.",
    "handoff": "Sprint handoff. Prepares state for next sprint or creates final release package.",
    "done": "Sprint completion. Cleanup, archival, and final status update.",

    # L1 - Executive
    "signoff": "Executive sign-off. CEO performs final approval of sprint deliverables.",

    # L2 - VPs
    "specvp": "Specification VP. Oversees spec decomposition and ensures requirements quality.",
    "execvp": "Execution VP. Oversees implementation phase and coordinates squads.",

    # L3 - Masters
    "spec": "Specification decomposition. Spec Master breaks down requirements using Graph-of-Thoughts reasoning.",
    "plan": "Sprint planning. Estimates effort, sequences tasks, and allocates resources to squads.",
    "test": "Test execution. Ace Exec runs full test suite and collects coverage metrics.",
    "qa": "Quality assurance. QA Master aggregates gate results and validates sprint meets quality thresholds.",
    "merge": "Code merge. Integrates squad outputs and resolves conflicts.",

    # L4 - Squad Leads
    "spawn": "Squad spawning. Creates worker squads and assigns squad leads.",
    "assign": "Task assignment. Distributes tasks to workers based on skills and load.",

    # L5 - Workers
    "inv": "Invariant validation. Auditor checks all INV-* constraints against implementation.",
    "deps": "Dependency mapping. Builds import graph and detects circular dependencies.",
    "exec1": "Implementation squad 1. Workers implementing RF-001 and related tasks.",
    "exec2": "Implementation squad 2. Workers implementing RF-002 and related tasks.",
    "exec3": "Implementation squad 3. Workers implementing RF-003 and related tasks.",
    "exec4": "Implementation squad 4. Workers implementing RF-004 and related tasks.",
    "g0": "Gate G0: Schema & Grounding. Validates data schemas and ensures proper grounding.",
    "g1": "Gate G1: Semantic Coherence. Checks semantic consistency across codebase.",
    "g2": "Gate G2: Cross-Reference. Validates references between modules.",
    "g3": "Gate G3: Behavioral Testing. Verifies expected behavior via tests.",
    "g4": "Gate G4: Security & Blindagem. Security scanning and vulnerability checks.",
    "g5": "Gate G5: Coverage & Quality. Code coverage and quality metrics.",
    "g6": "Gate G6: Performance. Performance benchmarks and regression detection.",
    "g7": "Gate G7: Integration. Integration tests and API contract validation.",
    "g8": "Gate G8: Final Validation. Comprehensive final checks before approval.",
    "review": "Human review. Manual inspection and approval by reviewers.",
    "rework": "Rework handling. Processes gate failures and coordinates fixes.",

    # L6 - Specialists
    "reflex": "Reflexion engine. Analyzes failures and generates improvement strategies.",
}

# =============================================================================
# NODE TO AGENTS MAPPING - Quais agentes trabalham em cada node
# =============================================================================

NODE_TO_AGENTS: Dict[str, List[str]] = {
    # L0 - System
    "init": ["run_master"],
    "load": ["run_master", "pack_driver"],
    "docs": ["docgen"],
    "art": ["artifact_writer"],
    "metrics": ["metrics_collector"],
    "handoff": ["handoff_handler"],
    "done": ["run_master"],

    # L1 - Executive
    "signoff": ["ceo"],

    # L2 - VPs
    "specvp": ["spec_vp"],
    "execvp": ["exec_vp"],

    # L3 - Masters
    "spec": ["spec_master"],
    "plan": ["sprint_planner"],
    "test": ["ace_exec"],
    "qa": ["qa_master"],
    "merge": ["merge_handler"],

    # L4 - Squad Leads
    "spawn": ["squad_lead"],
    "assign": ["task_assigner"],

    # L5 - Workers
    "inv": ["auditor"],
    "deps": ["dependency_mapper"],
    "exec1": ["worker_rf001"],
    "exec2": ["worker_rf002"],
    "exec3": ["worker_rf003"],
    "exec4": ["worker_rf004"],
    "g0": ["gate_runner"],
    "g1": ["gate_runner"],
    "g2": ["gate_runner"],
    "g3": ["gate_runner"],
    "g4": ["gate_runner"],
    "g5": ["gate_runner"],
    "g6": ["gate_runner"],
    "g7": ["gate_runner"],
    "g8": ["gate_runner"],
    "review": ["human_reviewer", "clean_reviewer"],
    "rework": ["rework_handler"],

    # L6 - Specialists
    "reflex": ["reflexion_engine"],
}

# =============================================================================
# LAYER COLORS - Cor por camada da hierarquia (para avatars de agentes)
# =============================================================================

LAYER_COLORS: Dict[int, str] = {
    0: "#ef4444",  # L0 - Red
    1: "#f59e0b",  # L1 - Amber
    2: "#eab308",  # L2 - Yellow
    3: "#22c55e",  # L3 - Green
    4: "#06b6d4",  # L4 - Cyan
    5: "#3b82f6",  # L5 - Blue
    6: "#8b5cf6",  # L6 - Purple
}

# =============================================================================
# AGENT LAYER - Layer de cada agente na hierarquia
# =============================================================================

AGENT_LAYER: Dict[str, int] = {
    # L0
    "run_master": 0,
    "pack_driver": 0,
    "docgen": 0,
    "artifact_writer": 0,
    "metrics_collector": 0,
    "handoff_handler": 0,
    # L1
    "ceo": 1,
    # L2
    "spec_vp": 2,
    "exec_vp": 2,
    # L3
    "spec_master": 3,
    "sprint_planner": 3,
    "ace_exec": 3,
    "qa_master": 3,
    "merge_handler": 3,
    # L4
    "squad_lead": 4,
    "task_assigner": 4,
    # L5
    "auditor": 5,
    "dependency_mapper": 5,
    "worker_rf001": 5,
    "worker_rf002": 5,
    "worker_rf003": 5,
    "worker_rf004": 5,
    "gate_runner": 5,
    "human_reviewer": 5,
    "clean_reviewer": 5,
    "rework_handler": 5,
    # L6
    "reflexion_engine": 6,
}

# =============================================================================
# NODE SUBSTEPS - Substeps de cada node
# =============================================================================

NODE_SUBSTEPS: Dict[str, List[Dict[str, str]]] = {
    "init": [
        {"name": "Validate environment", "status": "pending"},
        {"name": "Create workspace", "status": "pending"},
        {"name": "Check stack health", "status": "pending"},
        {"name": "Initialize IPC", "status": "pending"},
    ],
    "load": [
        {"name": "Fetch context pack", "status": "pending"},
        {"name": "Validate pack schema", "status": "pending"},
        {"name": "Parse requirements", "status": "pending"},
        {"name": "Index dependencies", "status": "pending"},
    ],
    "spec": [
        {"name": "Load context", "status": "pending"},
        {"name": "Parse requirements", "status": "pending"},
        {"name": "Generate spec (GoT)", "status": "pending"},
        {"name": "Validate invariants", "status": "pending"},
    ],
    "inv": [
        {"name": "Load rules", "status": "pending"},
        {"name": "Validate RFs", "status": "pending"},
        {"name": "Check INV constraints", "status": "pending"},
        {"name": "Generate report", "status": "pending"},
    ],
    "deps": [
        {"name": "Analyze imports", "status": "pending"},
        {"name": "Build graph", "status": "pending"},
        {"name": "Detect cycles", "status": "pending"},
        {"name": "Export DAG", "status": "pending"},
    ],
    "plan": [
        {"name": "Assess effort", "status": "pending"},
        {"name": "Sequence tasks", "status": "pending"},
        {"name": "Allocate resources", "status": "pending"},
    ],
    "spawn": [
        {"name": "Create squads", "status": "pending"},
        {"name": "Assign leads", "status": "pending"},
        {"name": "Distribute tasks", "status": "pending"},
    ],
    "assign": [
        {"name": "Match skills", "status": "pending"},
        {"name": "Balance load", "status": "pending"},
        {"name": "Notify agents", "status": "pending"},
    ],
    "exec1": [
        {"name": "Setup workspace", "status": "pending"},
        {"name": "Implement RF-001", "status": "pending"},
        {"name": "Self-test", "status": "pending"},
        {"name": "Commit changes", "status": "pending"},
    ],
    "exec2": [
        {"name": "Setup workspace", "status": "pending"},
        {"name": "Implement RF-002", "status": "pending"},
        {"name": "Self-test", "status": "pending"},
        {"name": "Commit changes", "status": "pending"},
    ],
    "exec3": [
        {"name": "Setup workspace", "status": "pending"},
        {"name": "Implement RF-003", "status": "pending"},
        {"name": "Self-test", "status": "pending"},
        {"name": "Commit changes", "status": "pending"},
    ],
    "exec4": [
        {"name": "Setup workspace", "status": "pending"},
        {"name": "Implement RF-004", "status": "pending"},
        {"name": "Self-test", "status": "pending"},
        {"name": "Commit changes", "status": "pending"},
    ],
    "test": [
        {"name": "Collect tests", "status": "pending"},
        {"name": "Run suite", "status": "pending"},
        {"name": "Check coverage", "status": "pending"},
        {"name": "Generate report", "status": "pending"},
    ],
    "g0": [
        {"name": "Load schema rules", "status": "pending"},
        {"name": "Validate artifacts", "status": "pending"},
        {"name": "Score results", "status": "pending"},
        {"name": "Record decision", "status": "pending"},
    ],
    "g1": [
        {"name": "Load semantic rules", "status": "pending"},
        {"name": "Analyze coherence", "status": "pending"},
        {"name": "Score results", "status": "pending"},
        {"name": "Record decision", "status": "pending"},
    ],
    "g2": [
        {"name": "Load reference rules", "status": "pending"},
        {"name": "Check cross-refs", "status": "pending"},
        {"name": "Score results", "status": "pending"},
        {"name": "Record decision", "status": "pending"},
    ],
    "g3": [
        {"name": "Load test cases", "status": "pending"},
        {"name": "Run behavioral tests", "status": "pending"},
        {"name": "Score results", "status": "pending"},
        {"name": "Record decision", "status": "pending"},
    ],
    "g4": [
        {"name": "Load security rules", "status": "pending"},
        {"name": "Scan vulnerabilities", "status": "pending"},
        {"name": "Score results", "status": "pending"},
        {"name": "Record decision", "status": "pending"},
    ],
    "g5": [
        {"name": "Collect coverage", "status": "pending"},
        {"name": "Check thresholds", "status": "pending"},
        {"name": "Score results", "status": "pending"},
        {"name": "Record decision", "status": "pending"},
    ],
    "g6": [
        {"name": "Run benchmarks", "status": "pending"},
        {"name": "Check regression", "status": "pending"},
        {"name": "Score results", "status": "pending"},
        {"name": "Record decision", "status": "pending"},
    ],
    "g7": [
        {"name": "Run integrations", "status": "pending"},
        {"name": "Validate contracts", "status": "pending"},
        {"name": "Score results", "status": "pending"},
        {"name": "Record decision", "status": "pending"},
    ],
    "g8": [
        {"name": "Final validation", "status": "pending"},
        {"name": "Aggregate scores", "status": "pending"},
        {"name": "Make decision", "status": "pending"},
        {"name": "Record verdict", "status": "pending"},
    ],
    "reflex": [
        {"name": "Analyze failure", "status": "pending"},
        {"name": "Generate reflection", "status": "pending"},
        {"name": "Propose fix", "status": "pending"},
        {"name": "Queue rework", "status": "pending"},
    ],
    "qa": [
        {"name": "Aggregate results", "status": "pending"},
        {"name": "Check thresholds", "status": "pending"},
        {"name": "Summarize findings", "status": "pending"},
    ],
    "review": [
        {"name": "Human checklist", "status": "pending"},
        {"name": "Collect feedback", "status": "pending"},
        {"name": "Record approval", "status": "pending"},
    ],
    "signoff": [
        {"name": "Executive review", "status": "pending"},
        {"name": "Final approval", "status": "pending"},
        {"name": "Record sign-off", "status": "pending"},
    ],
    "merge": [
        {"name": "Collect branches", "status": "pending"},
        {"name": "Resolve conflicts", "status": "pending"},
        {"name": "Finalize merge", "status": "pending"},
    ],
    "handoff": [
        {"name": "Package artifacts", "status": "pending"},
        {"name": "Update state", "status": "pending"},
        {"name": "Notify next sprint", "status": "pending"},
    ],
    "done": [
        {"name": "Cleanup workspace", "status": "pending"},
        {"name": "Archive logs", "status": "pending"},
        {"name": "Record metrics", "status": "pending"},
    ],
    "specvp": [
        {"name": "Review requirements", "status": "pending"},
        {"name": "Approve spec plan", "status": "pending"},
    ],
    "execvp": [
        {"name": "Review implementation plan", "status": "pending"},
        {"name": "Approve squad allocation", "status": "pending"},
    ],
    "rework": [
        {"name": "Analyze failures", "status": "pending"},
        {"name": "Create fix tasks", "status": "pending"},
        {"name": "Re-queue for execution", "status": "pending"},
    ],
}

# =============================================================================
# NODE LOG KEYWORDS - Keywords para filtrar logs por node
# =============================================================================

NODE_LOG_KEYWORDS: Dict[str, List[str]] = {
    "init": ["run_master", "INIT", "workspace", "stack_check", "initialize", "startup"],
    "load": ["pack_driver", "context_pack", "load", "fetch", "parse", "S00", "S01"],
    "spec": ["spec_master", "SPEC", "decompos", "requirement", "GoT", "RF-"],
    "inv": ["auditor", "invariant", "INV-", "validate", "rule", "constraint"],
    "deps": ["dependency", "mapper", "import", "graph", "cycle", "DAG"],
    "plan": ["planner", "PLAN", "effort", "sequence", "allocate", "schedule"],
    "spawn": ["spawn", "squad", "create", "lead"],
    "assign": ["assign", "task", "worker", "distribute", "allocat"],
    "exec1": ["RF-001", "worker", "squad_1", "implement", "exec1"],
    "exec2": ["RF-002", "worker", "squad_2", "implement", "exec2"],
    "exec3": ["RF-003", "worker", "squad_3", "implement", "exec3"],
    "exec4": ["RF-004", "worker", "squad_4", "implement", "exec4"],
    "test": ["test", "pytest", "coverage", "suite", "ace_exec"],
    "g0": ["G0", "gate_0", "schema", "grounding"],
    "g1": ["G1", "gate_1", "semantic", "coherence"],
    "g2": ["G2", "gate_2", "cross-ref", "reference"],
    "g3": ["G3", "gate_3", "behavioral", "test"],
    "g4": ["G4", "gate_4", "security", "blindagem", "vuln"],
    "g5": ["G5", "gate_5", "coverage", "quality"],
    "g6": ["G6", "gate_6", "performance", "benchmark"],
    "g7": ["G7", "gate_7", "integration", "contract"],
    "g8": ["G8", "gate_8", "final", "validation"],
    "reflex": ["reflex", "REFLEX", "reflection", "failure", "improvement"],
    "qa": ["qa_master", "QA", "quality", "aggregate", "threshold"],
    "review": ["review", "human", "approval", "checklist", "manual"],
    "signoff": ["signoff", "CEO", "executive", "approval", "final"],
    "merge": ["merge", "conflict", "branch", "integrate"],
    "handoff": ["handoff", "package", "artifact", "release", "transition"],
    "done": ["done", "complete", "cleanup", "archive", "finish"],
    "specvp": ["spec_vp", "specification", "VP"],
    "execvp": ["exec_vp", "execution", "VP"],
    "rework": ["rework", "fix", "retry", "failed"],
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _normalize_status(status: Optional[str]) -> str:
    """Normalize a status string to cockpit format."""
    if not status:
        return "pending"
    return STATUS_MAP.get(status.upper(), "pending")


def _get_circuit_breaker_css_class(state: str) -> str:
    """Get CSS class for circuit breaker state."""
    return {
        "closed": "cb-closed",
        "open": "cb-open",
        "half_open": "cb-half-open",
    }.get(state.lower() if state else "", "cb-unknown")


def _get_alert_css_class(level: str) -> str:
    """Get CSS class for alert level."""
    return {
        "critical": "alert-critical",
        "warning": "alert-warning",
        "info": "alert-info",
        "none": "alert-none",
    }.get(level, "alert-unknown")


def _get_phase_index(phase: str) -> int:
    """Get the index of a phase in the phase order."""
    phase_upper = phase.upper() if phase else "INIT"
    # Resolve aliases first
    resolved_phase = PHASE_ALIASES.get(phase_upper, phase_upper)
    try:
        return PHASE_ORDER.index(resolved_phase)
    except ValueError:
        return -1


def _normalize_phase(phase: str) -> str:
    """Normalize phase name, resolving aliases."""
    phase_upper = phase.upper() if phase else "INIT"
    return PHASE_ALIASES.get(phase_upper, phase_upper)


def _is_phase_completed(phase: str, current_phase: str, completed_phases: List[str]) -> bool:
    """Check if a phase is completed relative to current progress."""
    normalized_phase = _normalize_phase(phase)
    normalized_completed = [_normalize_phase(p) for p in completed_phases]
    if normalized_phase in normalized_completed:
        return True
    current_idx = _get_phase_index(current_phase)
    phase_idx = _get_phase_index(phase)
    return phase_idx >= 0 and phase_idx < current_idx


def _is_phase_active(phase: str, current_phase: str) -> bool:
    """Check if a phase is currently active."""
    return _normalize_phase(phase) == _normalize_phase(current_phase)


# =============================================================================
# AGENT HELPER FUNCTIONS (for popup display)
# =============================================================================


def _get_agent_display_name(agent_id: str) -> str:
    """
    Converte agent_id para nome de exibicao.

    Ex: "spec_master" -> "Spec Master"
    """
    return agent_id.replace("_", " ").title()


def _get_agent_avatar(agent_id: str) -> str:
    """
    Gera avatar de 2 letras para um agente.

    Ex: "spec_master" -> "SM"
        "qa_master" -> "QM"
        "auditor" -> "AU"
    """
    parts = agent_id.split("_")
    if len(parts) >= 2:
        return (parts[0][0] + parts[1][0]).upper()
    return agent_id[:2].upper()


def _get_agent_color(agent_id: str) -> str:
    """
    Retorna cor do agente baseado em sua camada.

    Ex: "spec_master" (L3) -> "#22c55e" (green)
    """
    layer = AGENT_LAYER.get(agent_id, 5)  # Default: L5 (blue)
    return LAYER_COLORS.get(layer, "#3b82f6")


def _event_matches_node(event: Dict[str, Any], node_id: str) -> bool:
    """
    Verifica se evento pertence ao node.

    Criterios de matching:
    1. node_id direto
    2. gate_id (para nodes g0-g8)
    3. agent_id (se agente pertence ao node)
    4. keywords na mensagem
    """
    # Match por node_id direto
    if event.get("node_id") == node_id:
        return True

    # Match por gate_id
    if node_id.startswith("g") and len(node_id) == 2:
        gate_id = node_id.upper()
        if event.get("gate_id") == gate_id:
            return True

    # Match por agent_id
    event_agent = event.get("agent_id", "")
    node_agents = NODE_TO_AGENTS.get(node_id, [])
    if event_agent and event_agent in node_agents:
        return True

    # Match por keywords
    keywords = NODE_LOG_KEYWORDS.get(node_id, [])
    if keywords:
        event_str = str(event.get("message", "")) + str(event.get("msg", "")) + str(event.get("type", ""))
        event_str_lower = event_str.lower()
        for kw in keywords:
            if kw.lower() in event_str_lower:
                return True

    return False


def _event_to_level(event: Dict[str, Any]) -> str:
    """Converte tipo de evento para nivel de log."""
    event_type = str(event.get("type", "")).lower()
    status = str(event.get("status", "")).lower()
    level = str(event.get("level", "")).lower()

    # Usar level existente se valido
    if level in ["error", "warn", "warning", "success", "info"]:
        return "warn" if level == "warning" else level

    # Inferir de tipo/status
    if "error" in event_type or "fail" in event_type or status == "fail":
        return "error"
    if "warn" in event_type:
        return "warn"
    if "pass" in status or "success" in event_type or "complete" in event_type:
        return "success"
    return "info"


def _parse_log_line(line: str) -> tuple:
    """
    Parseia linha de log Python.

    Ex: "2026-01-27 20:22:30,123 - module - INFO - Message"

    Returns: (timestamp, level, message)
    """
    import re

    # Padrao: YYYY-MM-DD HH:MM:SS,mmm - module - LEVEL - message
    match = re.match(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - \S+ - (\w+) - (.+)",
        line
    )
    if match:
        timestamp = match.group(1).replace(" ", "T")
        level = match.group(2).lower()
        msg = match.group(3)

        # Normalizar level
        if level == "error":
            level = "error"
        elif level == "warning":
            level = "warn"
        elif level in ["info", "debug"]:
            level = "info"
        else:
            level = "info"

        return timestamp, level, msg

    # Fallback: linha sem formato padrao
    return "", "info", line[:100] if len(line) > 100 else line


# =============================================================================
# NODE POPUP FUNCTIONS (substeps, agents, enhanced logs)
# =============================================================================


def get_node_substeps(
    node_id: str,
    phase: str,
    events: List[Dict[str, Any]],
    completed_phases: List[str],
    gates: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Retorna substeps de um node com status atualizado.

    Args:
        node_id: ID do node (ex: "spec", "g0")
        phase: Fase atual do pipeline
        events: Lista de eventos do pipeline
        completed_phases: Lista de fases ja completadas
        gates: Resultados dos gates (para determinar status de gates)

    Returns:
        Lista de substeps no formato:
        [{"name": str, "status": str, "time": str}, ...]
    """
    if node_id not in NODE_SUBSTEPS:
        return []

    # Deep copy para nao modificar original
    substeps = [dict(s) for s in NODE_SUBSTEPS[node_id]]

    # Determinar status do node
    node_status = get_node_status(node_id, phase, gates, completed_phases, {})

    if node_status == "complete":
        # Todos substeps completos
        for s in substeps:
            s["status"] = "done"
            s["time"] = "-"
    elif node_status in ["error", "rejected"]:
        # Node falhou - marcar ultimo substep como error
        node_events = [e for e in events if _event_matches_node(e, node_id)]
        progress = min(len(node_events), len(substeps) - 1)
        for i, s in enumerate(substeps):
            if i < progress:
                s["status"] = "done"
                s["time"] = "-"
            elif i == progress:
                s["status"] = "error"
                s["time"] = "-"
            else:
                s["status"] = "pending"
                s["time"] = "-"
    elif node_status == "active":
        # Alguns completos, um ativo, resto pendente
        node_events = [e for e in events if _event_matches_node(e, node_id)]
        progress = min(len(node_events), len(substeps) - 1)

        for i, s in enumerate(substeps):
            if i < progress:
                s["status"] = "done"
                s["time"] = "-"
            elif i == progress:
                s["status"] = "active"
                s["time"] = "-"
            else:
                s["status"] = "pending"
                s["time"] = "-"
    else:
        # Todos pendentes
        for s in substeps:
            s["status"] = "pending"
            s["time"] = "-"

    return substeps


def get_node_agents(
    node_id: str,
    heartbeats: Dict[str, Dict[str, Any]],
    agent_statuses: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Retorna lista de agentes que trabalham neste node.

    Args:
        node_id: ID do node
        heartbeats: Heartbeats de todos os agentes
        agent_statuses: Status de todos os agentes

    Returns:
        Lista de agentes no formato:
        [{"name": str, "avatar": str, "color": str, "status": str, "task": str}, ...]
    """
    agent_ids = NODE_TO_AGENTS.get(node_id, [])
    if not agent_ids:
        return []

    result = []
    for agent_id in agent_ids:
        # Buscar dados do agente (heartbeat tem prioridade)
        agent_data = heartbeats.get(agent_id, {}) or agent_statuses.get(agent_id, {})

        # Determinar status
        raw_status = agent_data.get("status", "idle")
        if isinstance(raw_status, str) and raw_status.lower() in ["active", "running", "working", "busy"]:
            status = "active"
        else:
            status = "idle"

        # Extrair task atual
        task = agent_data.get("task", "") or agent_data.get("current_task", "") or ""

        result.append({
            "name": _get_agent_display_name(agent_id),
            "avatar": _get_agent_avatar(agent_id),
            "color": _get_agent_color(agent_id),
            "status": status,
            "task": task,
        })

    return result


def get_node_logs_enhanced(
    node_id: str,
    events: List[Dict[str, Any]],
    gates: Dict[str, Dict[str, Any]],
    run_log: List[str],
    redis_history: List[Dict[str, Any]],
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Retorna logs especificos de um node de multiplas fontes.

    Args:
        node_id: ID do node
        events: Eventos estruturados do pipeline
        gates: Resultados dos gates
        run_log: Linhas do arquivo run.log
        redis_history: Eventos do Redis
        limit: Maximo de logs a retornar

    Returns:
        Lista de logs ordenados por timestamp DESC:
        [{"time": str, "level": str, "msg": str}, ...]
    """
    logs = []

    # 1. Eventos estruturados
    for event in events:
        if _event_matches_node(event, node_id):
            msg = event.get("message", "") or event.get("msg", "") or str(event.get("type", ""))
            if msg:
                logs.append({
                    "time": str(event.get("timestamp", ""))[:19],  # ISO format truncado
                    "level": _event_to_level(event),
                    "msg": msg,
                })

    # 2. Resultados de gates (para nodes g0-g8)
    if node_id.startswith("g") and len(node_id) == 2:
        gate_id = node_id.upper()
        if gate_id in gates:
            gate_data = gates[gate_id]
            status = gate_data.get("status", "")
            level = "success" if status == "PASS" else "error" if status in ["FAIL", "BLOCK"] else "warn"
            score = gate_data.get("score", "N/A")
            logs.append({
                "time": str(gate_data.get("executed_at", ""))[:19],
                "level": level,
                "msg": f"Gate {gate_id}: {status} (score: {score})",
            })
            # Adicionar mensagens detalhadas do gate
            for msg in gate_data.get("messages", []):
                logs.append({
                    "time": str(gate_data.get("executed_at", ""))[:19],
                    "level": "info",
                    "msg": msg,
                })

    # 3. Linhas do run.log filtradas por keywords
    keywords = NODE_LOG_KEYWORDS.get(node_id, [])
    if keywords and run_log:
        for line in run_log:
            line_lower = line.lower()
            if any(kw.lower() in line_lower for kw in keywords):
                time, level, msg = _parse_log_line(line)
                if msg:
                    logs.append({"time": time, "level": level, "msg": msg})

    # 4. Redis history filtrado
    for event in redis_history:
        if _event_matches_node(event, node_id):
            msg = event.get("message", "") or event.get("msg", "") or str(event.get("type", ""))
            if msg:
                logs.append({
                    "time": str(event.get("timestamp", ""))[:19],
                    "level": _event_to_level(event),
                    "msg": msg,
                })

    # Remover duplicatas baseado em (time, msg)
    seen = set()
    unique_logs = []
    for log in logs:
        key = (log.get("time", ""), log.get("msg", ""))
        if key not in seen:
            seen.add(key)
            unique_logs.append(log)

    # Ordenar por timestamp DESC e limitar
    unique_logs = sorted(unique_logs, key=lambda x: x.get("time", "") or "", reverse=True)
    return unique_logs[:limit]


# =============================================================================
# RESILIENCE TRANSFORMATION FUNCTIONS
# =============================================================================


def transform_resilience_state() -> Dict[str, Any]:
    """
    Transform resilience module state into cockpit format.

    Returns dict with:
    - circuit_breakers: List of breaker cards for UI
    - oscillation: Alert banner data
    - retry_metrics: Retry statistics
    - overall_health: Boolean health indicator
    - timestamp: When state was read
    - available: Whether resilience module is available

    PURE FUNCTION with graceful error handling.
    """
    try:
        state = resilience_reader.get_resilience_state()
    except Exception:
        return {
            "circuit_breakers": [],
            "oscillation": {"active_patterns": [], "alert_level": "none", "has_alert": False},
            "retry_metrics": {"total_attempts": 0, "total_successes": 0, "total_failures": 0, "success_rate": 0.0},
            "overall_health": True,
            "timestamp": "",
            "available": False,
        }

    # Transform circuit breakers into UI cards
    cb_cards = []
    all_healthy = True
    for name, cb_data in state.get("circuit_breakers", {}).items():
        cb_state = cb_data.get("state", "unknown")
        is_healthy = cb_state.lower() == "closed"
        if not is_healthy:
            all_healthy = False
        cb_cards.append({
            "name": name,
            "state": cb_state.upper(),
            "is_healthy": is_healthy,
            "failure_count": cb_data.get("failure_count", 0),
            "success_count": cb_data.get("success_count", 0),
            "last_failure_time": cb_data.get("last_failure_time"),
            "css_class": _get_circuit_breaker_css_class(cb_state),
        })

    # Transform oscillation into alert format
    osc = state.get("oscillation", {})
    is_oscillating = osc.get("is_oscillating", False)
    pattern = osc.get("pattern", "none")

    if pattern == "runaway":
        alert_level = "critical"
        alert_message = "RUNAWAY detected: System in unstable state"
    elif pattern == "cycle":
        alert_level = "warning"
        alert_message = "CYCLE detected: Repeated state transitions"
    elif pattern == "abab":
        alert_level = "info"
        alert_message = "ABAB detected: Alternating state pattern"
    else:
        alert_level = "none"
        alert_message = ""

    oscillation_alert = {
        "active_patterns": [pattern] if is_oscillating else [],
        "alert_level": alert_level,
        "alert_message": alert_message,
        "has_alert": is_oscillating,
        "css_class": _get_alert_css_class(alert_level),
    }

    # Transform retry metrics
    retry = state.get("retry_metrics", {})
    retry_metrics = {
        "total_attempts": retry.get("total_attempts", 0),
        "total_successes": retry.get("total_successes", 0),
        "total_failures": retry.get("total_failures", 0),
        "success_rate": round(retry.get("success_rate", 0) * 100, 2),
    }

    # Determine overall health
    overall_health = all_healthy and not is_oscillating

    return {
        "circuit_breakers": cb_cards,
        "oscillation": oscillation_alert,
        "retry_metrics": retry_metrics,
        "overall_health": overall_health,
        "timestamp": state.get("timestamp", ""),
        "available": True,
    }


# =============================================================================
# MAIN TRANSFORMATION FUNCTIONS
# =============================================================================


def get_node_status(
    node_id: str,
    phase: str,
    gates: Dict[str, Dict],
    completed_phases: List[str],
    agent_statuses: Dict[str, Dict],
) -> str:
    """
    Determine node status based on pipeline state.

    Args:
        node_id: The node ID to check
        phase: Current pipeline phase
        gates: Gate results dict
        completed_phases: List of completed phases
        agent_statuses: Agent status dict

    Returns:
        Status string: pending, active, complete, warning, error, rejected
    """
    # Gate nodes - check gate results
    if node_id.startswith("g") and len(node_id) == 2:
        gate_id = node_id.upper()
        if gate_id in gates:
            return _normalize_status(gates[gate_id].get("status"))
        # If GATES phase is active and gate not run yet, show as pending
        if phase.upper() in ["GATES", "QA"]:
            return "pending"
        return "pending"

    # Find which phase this node belongs to
    for p, nodes in PHASE_TO_NODES.items():
        if node_id in nodes:
            if _is_phase_completed(p, phase, completed_phases):
                return "complete"
            if _is_phase_active(p, phase):
                return "active"
            return "pending"

    # Check agent status for exec nodes
    if node_id.startswith("exec") and agent_statuses:
        # Map exec nodes to potential agent IDs
        for agent_id, status in agent_statuses.items():
            if status.get("status", "").upper() in ["RUNNING", "ACTIVE"]:
                return "active"

    return "pending"


def get_node_metrics(
    node_id: str,
    gates: Dict[str, Dict],
    events: List[Dict],
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Get metrics for a specific node.

    Args:
        node_id: The node ID
        gates: Gate results
        events: Event log
        metrics: Pipeline metrics

    Returns:
        Dict with node-specific metrics
    """
    result = {}

    # Gate nodes - get gate metrics
    if node_id.startswith("g") and len(node_id) == 2:
        gate_id = node_id.upper()
        if gate_id in gates:
            gate_data = gates[gate_id]
            result = {
                "passed": gate_data.get("passed", False),
                "score": gate_data.get("score", 0),
                "executed_at": gate_data.get("executed_at", ""),
                "duration_ms": gate_data.get("duration_ms", 0),
            }
            # Add gate-specific details
            details = gate_data.get("details", {})
            if details:
                result.update(details)

    # Add general metrics if available
    if metrics:
        if node_id == "test":
            result["tests_passed"] = metrics.get("tests_passed", 0)
            result["tests_failed"] = metrics.get("tests_failed", 0)
            result["coverage"] = metrics.get("coverage", 0)

    return result


def get_node_logs(
    node_id: str,
    events: List[Dict],
    gates: Dict[str, Dict],
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Get log entries for a specific node.

    Args:
        node_id: The node ID
        events: Event log
        gates: Gate results
        limit: Maximum number of logs

    Returns:
        List of log entries
    """
    logs = []

    # Gate logs from gate results
    if node_id.startswith("g") and len(node_id) == 2:
        gate_id = node_id.upper()
        if gate_id in gates:
            gate_data = gates[gate_id]
            status = gate_data.get("status", "UNKNOWN")
            logs.append({
                "time": gate_data.get("executed_at", ""),
                "level": "success" if status == "PASS" else "error",
                "msg": f"Gate {gate_id}: {status}",
            })
            # Add any messages from gate
            for msg in gate_data.get("messages", []):
                logs.append({
                    "time": gate_data.get("executed_at", ""),
                    "level": "info",
                    "msg": msg,
                })

    # Filter events relevant to this node
    for event in events:
        event_type = event.get("type", "")
        event_node = event.get("node_id", "")
        event_gate = event.get("gate_id", "")

        # Match by node_id
        if event_node == node_id:
            logs.append({
                "time": event.get("timestamp", event.get("time", "")),
                "level": event.get("level", "info"),
                "msg": event.get("message", event.get("msg", str(event_type))),
            })
        # Match gate events
        elif node_id.startswith("g") and event_gate == node_id.upper():
            logs.append({
                "time": event.get("timestamp", event.get("time", "")),
                "level": event.get("level", "info"),
                "msg": event.get("message", event.get("msg", str(event_type))),
            })

    return logs[-limit:]


def transform_to_cockpit_state(run_dir: Path, project_root: Optional[Path] = None) -> Dict[str, Any]:
    """
    Transform pipeline state files to cockpit format.
    READ-ONLY: Only reads files, never writes.

    Args:
        run_dir: Path to the run directory
        project_root: Optional project root for checkpoint reading

    Returns:
        Dict with cockpit state:
        - nodes: Dict[node_id, {status, metrics, logs}]
        - phase: Current phase
        - run_id: Run identifier
        - sprint_id: Sprint identifier
        - updated_at: Last update timestamp
        - event_logs: List of event log entries for the log panel
    """
    # Read all state (read-only) - now includes checkpoint phase detection and Redis
    all_state = state_reader.get_all_state(run_dir, project_root)
    run_state = all_state["run_state"]
    heartbeats = all_state["heartbeats"]
    events = all_state["events"]
    gates = all_state["gates"]
    agent_statuses = all_state["agents"]
    metrics = all_state["metrics"]
    run_log = all_state.get("run_log", [])
    redis_history = all_state.get("redis_history", [])

    # Extract current phase
    phase = run_state.get("phase", run_state.get("current_phase", "INIT"))
    completed_phases = run_state.get("completed_phases", [])

    # Build nodes dict with enhanced popup data
    nodes = {}
    for node_id in ALL_NODE_IDS:
        status = get_node_status(
            node_id, phase, gates, completed_phases, agent_statuses
        )
        node_metrics = get_node_metrics(node_id, gates, events, metrics)

        # Enhanced logs from multiple sources (events, run.log, redis)
        node_logs = get_node_logs_enhanced(
            node_id, events, gates, run_log, redis_history, limit=20
        )

        # Substeps with progress based on phase and events
        node_substeps = get_node_substeps(
            node_id, phase, events, completed_phases, gates
        )

        # Agents working on this node
        node_agents = get_node_agents(
            node_id, heartbeats, agent_statuses
        )

        # Description from constant
        node_description = NODE_DESCRIPTIONS.get(node_id, "")

        nodes[node_id] = {
            "status": status,
            "title": NODE_TITLES.get(node_id, node_id),
            "description": node_description,
            "metrics": node_metrics,
            "substeps": node_substeps,
            "agents": node_agents,
            "logs": node_logs,
        }

    # Convert events to log lines for the output panel
    event_logs = []
    for event in events:
        msg = event.get("message", "")
        event_type = event.get("type", "")
        phase_name = event.get("phase", "")
        timestamp = event.get("timestamp", "")
        level = event.get("level", "info")

        if msg:
            log_line = f"[{phase_name}] {msg}" if phase_name else msg
            event_logs.append({
                "line": log_line,
                "level": level,
                "timestamp": timestamp,
                "type": event_type,
            })

    # Get resilience state
    resilience = transform_resilience_state()

    # Build result
    return {
        "nodes": nodes,
        "phase": phase,
        "run_id": run_state.get("run_id", run_dir.name if run_dir else ""),
        "sprint_id": run_state.get("sprint_id", ""),
        "updated_at": run_state.get("updated_at", datetime.now().isoformat()),
        "heartbeats": heartbeats,
        "event_logs": event_logs,  # For the log panel (structured events from ndjson)
        "run_log": run_log,  # For the log panel (raw log lines from run.log)
        "redis_history": redis_history,  # For the log panel (real-time events from Redis)
        "resilience": resilience,  # Resilience module state (circuit breakers, oscillation, etc.)
        "summary": {
            "total_nodes": len(ALL_NODE_IDS),
            "active_nodes": sum(1 for n in nodes.values() if n["status"] == "active"),
            "complete_nodes": sum(1 for n in nodes.values() if n["status"] == "complete"),
            "error_nodes": sum(1 for n in nodes.values() if n["status"] in ["error", "rejected"]),
            "gates_passed": sum(1 for g in gates.values() if g.get("status") == "PASS"),
            "gates_failed": sum(1 for g in gates.values() if g.get("status") == "FAIL"),
        },
    }


def get_node_connections() -> List[Dict[str, str]]:
    """
    Get the static node connections for rendering arrows.

    HIERARCHY FLOW (respects L0-L6 layers):
        INIT:      L0 init → L0 load
        SPEC:      L0 load → L2 specvp → L3 spec → L5 inv → L5 deps → L3 plan
        EXEC:      L3 plan → L2 execvp → L4 spawn → L4 assign → L5 workers → L3 test
        GATES:     L3 test → L3 qa → L5 gates → L6 reflex
        SIGNOFF:   L6 reflex → L3 merge → L5 review (rework loops)
        ARTIFACTS: L5 review → L0 docs → L0 art → L0 metrics → L1 signoff
        FINAL:     L1 signoff → L0 handoff → L0 done

    Returns:
        List of {from, to} connection dicts
    """
    return [
        # INIT Phase
        {"from": "init", "to": "load"},

        # SPEC Phase (L0 → L2 → L3 → L5 → L3)
        {"from": "load", "to": "specvp"},
        {"from": "specvp", "to": "spec"},
        {"from": "spec", "to": "inv"},
        {"from": "inv", "to": "deps"},
        {"from": "deps", "to": "plan"},

        # EXEC Phase (L3 → L2 → L4 → L5 → L3)
        {"from": "plan", "to": "execvp"},
        {"from": "execvp", "to": "spawn"},
        {"from": "spawn", "to": "assign"},
        {"from": "assign", "to": "exec1"},
        {"from": "assign", "to": "exec2"},
        {"from": "assign", "to": "exec3"},
        {"from": "assign", "to": "exec4"},
        {"from": "exec1", "to": "test"},
        {"from": "exec2", "to": "test"},
        {"from": "exec3", "to": "test"},
        {"from": "exec4", "to": "test"},

        # GATES Phase (L3 → L5 → L6)
        {"from": "test", "to": "qa"},
        {"from": "qa", "to": "g0"},
        {"from": "g0", "to": "g1"},
        {"from": "g1", "to": "g2"},
        {"from": "g2", "to": "g3"},
        {"from": "g3", "to": "g4"},
        {"from": "g4", "to": "g5"},
        {"from": "g5", "to": "g6"},
        {"from": "g6", "to": "g7"},
        {"from": "g7", "to": "g8"},
        {"from": "g8", "to": "reflex"},

        # SIGNOFF Phase (L6 → L3 → L5)
        {"from": "reflex", "to": "merge"},
        {"from": "merge", "to": "review"},
        {"from": "rework", "to": "spawn"},  # Loop back

        # ARTIFACTS Phase (L5 → L0 → L1)
        {"from": "review", "to": "docs"},
        {"from": "docs", "to": "art"},
        {"from": "art", "to": "metrics"},
        {"from": "metrics", "to": "signoff"},

        # FINAL Phase (L1 → L0)
        {"from": "signoff", "to": "handoff"},
        {"from": "handoff", "to": "done"},
    ]
