"""LangGraph Pipeline State Definition.

This module defines the PipelineState TypedDict for the LangGraph control plane,
following the schema from PIPELINE_V3_MASTER_PLAN.md Section 10.4.

The state is designed to:
- Support idempotency/retry/resume (LangGraph checkpointing)
- Preserve V2 contracts (handoff/approvals/signoff/gates)
- Not break dashboard/metrics (run_state.yml, event_log.ndjson, heartbeats)
- Integrate NEW stacks: Buffer of Thoughts (BoT), Active RAG, NeMo/LLM Guard

Rule of Gold: "state interno" (LangGraph) can be rich, but "state publicado"
(run_state.yml + event_log) must be simple and stable.

Author: Pipeline Autonomo Team
Version: 1.0.0 (2026-01-16)
Based on: MIGRATION_V2_TO_LANGGRAPH.md + PIPELINE_V3_MASTER_PLAN.md Section 10.4
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, TypedDict, Union
import uuid


# =============================================================================
# LAZY CONTEXT PACK (P0-05)
# =============================================================================


class LazyContextPack:
    """Lazy-loading context pack with dict-like interface.

    This class provides a dict-like interface to context pack data,
    loading the full pack data only when first accessed. This optimizes
    checkpoint serialization by avoiding serialization of large context
    packs that may not be needed during resume.

    P0-05: When serializing to checkpoint, only serializes a reference
    (pack_id + path) rather than the full content. On access, loads
    the full pack data from the pack path.

    Usage:
        # Create from existing pack data
        lazy = LazyContextPack.from_pack_state(context_pack_state)

        # Create from pack reference (for checkpoint resume)
        lazy = LazyContextPack(pack_id="S00", pack_path="/path/to/pack")

        # Access like a dict (triggers load if not loaded)
        objective = lazy["objective"]
        deliverables = lazy.get("deliverables", [])

        # Check if specific key exists
        if "functional_requirements" in lazy:
            # ...

        # Iteration (triggers full load)
        for key, value in lazy.items():
            # ...

    Attributes:
        pack_id: The sprint/pack identifier (e.g., "S00").
        pack_path: Path to the context pack file.
        _loaded: Whether full data has been loaded.
        _data: The loaded context pack data (ContextPackState).
        _loader_func: Optional custom loader function.
    """

    __slots__ = ("pack_id", "pack_path", "_loaded", "_data", "_loader_func", "_sha256")

    def __init__(
        self,
        pack_id: str,
        pack_path: str,
        sha256: str = "",
        loader_func: Optional[Any] = None,
    ):
        """Initialize a LazyContextPack.

        Args:
            pack_id: The pack identifier (e.g., "S00").
            pack_path: Path to the context pack file.
            sha256: Expected SHA256 hash for integrity check.
            loader_func: Optional custom loader function. If not provided,
                uses the default spec_kit_loader.
        """
        self.pack_id = pack_id
        self.pack_path = pack_path
        self._sha256 = sha256
        self._loaded = False
        self._data: Optional[ContextPackState] = None
        self._loader_func = loader_func

    @classmethod
    def from_pack_state(
        cls,
        pack_state: "ContextPackState",
        loader_func: Optional[Any] = None,
    ) -> "LazyContextPack":
        """Create a LazyContextPack from existing ContextPackState.

        The pack is immediately marked as loaded since we already have
        the full data.

        Args:
            pack_state: The existing context pack state dict.
            loader_func: Optional custom loader function for reloading.

        Returns:
            LazyContextPack with data already loaded.
        """
        instance = cls(
            pack_id=pack_state.get("pack_id", ""),
            pack_path=pack_state.get("pack_path", ""),
            sha256=pack_state.get("sha256", ""),
            loader_func=loader_func,
        )
        instance._data = pack_state
        instance._loaded = True
        return instance

    def _load(self) -> None:
        """Load the full context pack data.

        This method is called automatically when data is first accessed.
        Uses the spec_kit_loader to load and parse the context pack.
        """
        if self._loaded:
            return

        if self._loader_func is not None:
            # Use custom loader
            self._data = self._loader_func(self.pack_id, self.pack_path)
            self._loaded = True
            return

        # Default loader: use spec_kit_loader
        try:
            from pipeline.spec_kit_loader import load_context_pack

            pack = load_context_pack(self.pack_id)
            self._data = extract_context_pack_data(pack)
            self._loaded = True
        except ImportError:
            # Fallback: create minimal pack state
            import hashlib

            self._data = ContextPackState(
                pack_id=self.pack_id,
                pack_path=self.pack_path,
                sha256=self._sha256,
                dependencies=[],
                deliverables=[],
                objective=None,
                rf_count=0,
                inv_count=0,
                edge_count=0,
                functional_requirements=[],
                invariants=[],
                edge_cases=[],
                acceptance_criteria=[],
                dod_checklist=[],
                behaviors=[],
                deliverables_spec=[],
                rf_to_deliverable={},
                rf_to_test={},
                inv_to_enforcement={},
            )
            self._loaded = True

    def is_loaded(self) -> bool:
        """Check if the full data has been loaded."""
        return self._loaded

    def force_load(self) -> "LazyContextPack":
        """Force load the data and return self for chaining."""
        self._load()
        return self

    def to_dict(self) -> ContextPackState:
        """Get the full data as a dict.

        Triggers load if not already loaded.

        Returns:
            The full ContextPackState dict.
        """
        self._load()
        return self._data or {}

    def to_reference(self) -> Dict[str, str]:
        """Get a minimal reference dict for checkpoint serialization.

        Returns only the pack_id, pack_path, and sha256 - enough to
        reconstruct the LazyContextPack on resume.

        Returns:
            Dict with pack_id, pack_path, and sha256.
        """
        return {
            "pack_id": self.pack_id,
            "pack_path": self.pack_path,
            "sha256": self._sha256,
            "_lazy": "true",  # Marker for checkpoint deserialization
        }

    # Dict-like interface

    def __getitem__(self, key: str) -> Any:
        """Get an item, loading data if needed."""
        self._load()
        if self._data is None:
            raise KeyError(key)
        return self._data[key]  # type: ignore

    def __contains__(self, key: str) -> bool:
        """Check if key exists, loading data if needed."""
        self._load()
        if self._data is None:
            return False
        return key in self._data

    def __iter__(self) -> Iterator[str]:
        """Iterate over keys, loading data if needed."""
        self._load()
        if self._data is None:
            return iter([])
        return iter(self._data)

    def __len__(self) -> int:
        """Get number of keys, loading data if needed."""
        self._load()
        if self._data is None:
            return 0
        return len(self._data)

    def get(self, key: str, default: Any = None) -> Any:
        """Get an item with default, loading data if needed."""
        self._load()
        if self._data is None:
            return default
        return self._data.get(key, default)  # type: ignore

    def keys(self):
        """Get keys, loading data if needed."""
        self._load()
        if self._data is None:
            return [].keys()  # type: ignore
        return self._data.keys()

    def values(self):
        """Get values, loading data if needed."""
        self._load()
        if self._data is None:
            return [].values()  # type: ignore
        return self._data.values()

    def items(self):
        """Get items, loading data if needed."""
        self._load()
        if self._data is None:
            return [].items()  # type: ignore
        return self._data.items()

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "lazy"
        return f"LazyContextPack(pack_id={self.pack_id!r}, status={status})"

    def __str__(self) -> str:
        return self.__repr__()


# =============================================================================
# ENUMS
# =============================================================================


class SprintPhase(str, Enum):
    """Sprint execution phases.

    Invariant I3: INIT -> SPEC -> PLAN -> EXEC -> QA -> VOTE -> DONE
    (retries can go back, but register `attempt`)
    """

    INIT = "INIT"
    SPEC = "SPEC"
    PLAN = "PLAN"
    EXEC = "EXEC"
    QA = "QA"
    VOTE = "VOTE"
    DONE = "DONE"
    HALT = "HALT"  # Safe halt state


class PipelineStatus(str, Enum):
    """Overall pipeline status."""

    INITIALIZED = "initialized"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    HALTED = "halted"  # Safe halt triggered
    RECOVERED = "recovered"  # Resumed from checkpoint


class GateStatus(str, Enum):
    """Gate execution status.

    Invariant I4: gates must PASS before signoff is valid.
    """

    PASS = "PASS"
    FAIL = "FAIL"
    BLOCK = "BLOCK"
    PENDING = "PENDING"
    SKIPPED = "SKIPPED"


class TaskLevelStatus(str, Enum):
    """Task-level execution status for surgical rework.

    State Machine:
        PENDING -> BLOCKED (if has unresolved dependencies)
               -> EXECUTING
        EXECUTING -> VALIDATING
        VALIDATING -> PASSED (terminal)
                   -> FAILED
        FAILED -> REPAIRING -> VALIDATING (loop)
              -> ESCALATED (terminal, if max_attempts exceeded)
    """

    PENDING = "pending"
    BLOCKED = "blocked"
    EXECUTING = "executing"
    VALIDATING = "validating"
    PASSED = "passed"
    FAILED = "failed"
    REPAIRING = "repairing"
    ESCALATED = "escalated"

    def is_terminal(self) -> bool:
        """Return True if this is a terminal state."""
        return self in (TaskLevelStatus.PASSED, TaskLevelStatus.ESCALATED)


class ReworkPhase(str, Enum):
    """Phase within the rework process.

    Tracks where we are in the surgical rework loop:
    1. DIAGNOSING: Analyzing the failure to identify root cause
    2. PATCHING: Generating and applying surgical fix
    3. VALIDATING: Re-running validation on patched code
    """

    DIAGNOSING = "diagnosing"
    PATCHING = "patching"
    VALIDATING = "validating"


class FailureCategory(str, Enum):
    """Category of failure for routing rework strategy.

    Determines how the failure should be handled:
    - CODE: Can be auto-repaired by agents
    - TIMEOUT: Infrastructure issue - increase timeout
    - INFRASTRUCTURE: Requires ops intervention
    - SECURITY: Requires human review
    - INTEGRATION: Conflict between tasks
    - PERFORMANCE: Can be optimized by agents
    """

    CODE = "code"
    TIMEOUT = "timeout"
    INFRASTRUCTURE = "infrastructure"
    SECURITY = "security"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"


# =============================================================================
# SUB-STATE TYPEDDICTS
# =============================================================================


class IdentityState(TypedDict):
    """Identity fields for the sprint.

    These fields uniquely identify the current execution context.
    """

    run_id: str
    sprint_id: str
    attempt: int  # Increments on each real retry
    phase: str  # SprintPhase value


class PointerState(TypedDict):
    """External reference pointers.

    Pointers to directories, traces, and external sessions.
    """

    run_dir: str  # Absolute or relative to repo
    repo_root: str
    trace_id: Optional[str]  # Langfuse trace
    agentops_run_id: Optional[str]  # AgentOps (if enabled)
    literal_session_id: Optional[str]  # LiteralAI (if enabled)
    phoenix_session_id: Optional[str]  # Arize Phoenix (if enabled)


class PolicySnapshotState(TypedDict):
    """Policy snapshot for the run.

    Invariant I10: exists per run with timeouts/flags/retries,
    and policy changes are always explicit.

    This prevents "rediscovering" guardrails/timeouts after migration.
    """

    task_timeout_seconds: int
    sprint_timeout_seconds: int
    max_retry_attempts: int
    use_temporal: bool
    use_zap_validation: bool
    run_test_guardrail: bool
    cost_policy_ref: Optional[str]
    worker_limit_profile: Optional[str]

    # Guardrail subsystems (cannot "disappear" during migration)
    fraud_detection_enabled: bool
    executive_verification_enabled: bool
    completeness_gate_enabled: bool
    truthfulness_validation_enabled: bool
    rework_system_enabled: bool


class StackHealthEntry(TypedDict):
    """Health status for a single stack."""

    healthy: bool
    checked_at: str  # ISO timestamp
    error: Optional[str]
    latency_ms: Optional[float]


class StackHealthState(TypedDict):
    """Stack health snapshot.

    Tracks which stacks are required and their health status.
    """

    required_stacks: List[str]
    stack_health: Dict[str, StackHealthEntry]
    last_check_at: str  # ISO timestamp
    all_required_healthy: bool


class ContextPackState(TypedDict, total=False):
    """Context pack information.

    Contains the loaded context pack details for the sprint.
    Extended in GAP-1 to include full data, not just counts.
    """

    # Original fields (required)
    pack_id: str
    pack_path: str
    sha256: str
    dependencies: List[str]
    deliverables: List[str]
    objective: Optional[str]

    # Counts (for backward compatibility)
    rf_count: int  # Requirements Functional count
    inv_count: int  # Invariants count
    edge_count: int  # Edge cases count

    # GAP-1: Full data fields (optional for backward compatibility)
    # Functional Requirements: [{id, description, priority, files, test_ids}]
    functional_requirements: List[Dict[str, Any]]

    # Invariants: [{id, rule, enforcement, class_name}]
    invariants: List[Dict[str, Any]]

    # Edge Cases: [{id, case, expected_behavior, test_id}]
    edge_cases: List[Dict[str, Any]]

    # Acceptance Criteria: [{id, criteria, measurement, threshold}]
    acceptance_criteria: List[Dict[str, Any]]

    # Definition of Done checklist: [{item, required, category}]
    dod_checklist: List[Dict[str, Any]]

    # Behaviors/Gherkin: [{name, type, given, when, then, test_id}]
    behaviors: List[Dict[str, Any]]

    # Deliverables with spec: [{path, requirements, invariants, edge_cases}]
    deliverables_spec: List[Dict[str, Any]]

    # Traceability mappings
    rf_to_deliverable: Dict[str, str]
    rf_to_test: Dict[str, List[str]]
    inv_to_enforcement: Dict[str, Dict[str, str]]


class HandoffEntry(TypedDict):
    """A single handoff record.

    Handoffs are the communication mechanism from superior to subordinate.
    """

    from_agent: str
    to_agent: str
    what_i_want: str
    why_i_want: str
    expected_behavior: str
    recorded_at: str  # ISO timestamp
    correlation_id: Optional[str]


class ApprovalEntry(TypedDict):
    """A single approval record.

    Approvals are given by superiors to subordinates after reviewing work.
    """

    approver: str
    subordinate: str
    approved: bool
    justification: str
    approved_at: str  # ISO timestamp
    review_notes: Optional[str]
    artifacts_reviewed: List[str]


class SignoffEntry(TypedDict):
    """A single signoff record.

    Signoffs are the final confirmation that an agent completed their work.
    """

    agent_id: str
    approved: bool
    signed_at: str  # ISO timestamp
    what_i_did: str
    why_this_way: str
    how_it_works: str
    artifacts_verified: List[str]
    checks_passed: List[str]
    observations: Optional[str]


class MilestoneEntry(TypedDict):
    """A single milestone record."""

    milestone_id: str
    achieved_at: str  # ISO timestamp
    status: str  # "achieved", "pending", "failed"
    evidence: Optional[str]


class GovernanceState(TypedDict):
    """Governance state for handoffs, approvals, signoffs, and milestones.

    Invariant I5: supervisor cannot sign off if there is a subordinate
    without registered approval.
    """

    # Key format: "{from_agent}->{to_agent}"
    handoffs: Dict[str, HandoffEntry]

    # Key format: approvals[approver][subordinate]
    approvals: Dict[str, Dict[str, ApprovalEntry]]

    # Key format: signoffs[agent_id]
    signoffs: Dict[str, SignoffEntry]

    # Key format: milestones[agent_id][milestone_id]
    milestones: Dict[str, Dict[str, MilestoneEntry]]


class TaskEntry(TypedDict):
    """A single task execution record."""

    task_id: str
    agent_id: str
    status: str  # "pending", "running", "completed", "failed"
    result_ref: Optional[str]
    error: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]


class TemporalState(TypedDict):
    """Temporal workflow state (if Temporal is used)."""

    workflow_id: Optional[str]
    run_id: Optional[str]
    activity_count: int
    last_activity_at: Optional[str]


class ExecutionState(TypedDict):
    """Execution state for tasks.

    Contains CrewAI task results and Temporal workflow info.
    """

    tasks: Dict[str, TaskEntry]
    temporal: TemporalState
    crew_result: Optional[Dict[str, Any]]
    total_tasks: int
    completed_tasks: int
    failed_tasks: int


class GateState(TypedDict):
    """QA/Gates state.

    Invariant I4: no valid signoff exists if gates.status != PASS.
    Invariant I6: evidence bundle has MANIFEST.json; hash in bundle name
    must match calculated manifest.
    """

    gate_track: str  # A/B/C
    gate_selection_path: str
    gate_report_path: Optional[str]
    gate_status: str  # GateStatus value
    gates_passed: List[str]
    gates_failed: List[str]
    gates_skipped: List[str]
    evidence_bundle_dir: Optional[str]
    evidence_manifest_path: Optional[str]

    # Artifact paths
    qa_artifacts: Dict[str, str]  # quality_bar/gate_receipt/track_report/etc


class MemoryState(TypedDict):
    """Memory/Learning state (references, not giant blobs).

    These are pointers to external memory systems, not the actual data.
    """

    amem_notes: List[str]  # A-MEM note IDs/refs
    mem0_refs: List[str]  # Mem0 memory refs
    qdrant_refs: List[str]  # Qdrant collection/points refs
    letta_refs: List[str]  # Letta thread IDs


class NodeHistoryEntry(TypedDict):
    """A single node execution history entry.

    Invariant I2: each LangGraph node has deterministic idempotency_key;
    re-execution does not duplicate side effects.
    """

    node: str
    status: str  # "started", "completed", "failed", "skipped"
    started_at: str
    finished_at: Optional[str]
    idempotency_key: str
    error: Optional[str]


class EventResumeState(TypedDict):
    """State for event consumption and resume."""

    last_event_id_redis: Optional[str]  # If consuming Redis streams
    last_event_offset_file: Optional[int]  # If consuming ndjson
    node_history: List[NodeHistoryEntry]


class TimestampState(TypedDict):
    """Timestamp fields."""

    started_at: str  # ISO timestamp
    updated_at: str  # ISO timestamp
    completed_at: Optional[str]  # ISO timestamp


class ErrorEntry(TypedDict):
    """A single error record."""

    where: str  # Node/phase where error occurred
    error: str  # Error message
    ts: str  # ISO timestamp
    traceback: Optional[str]
    recoverable: bool


# =============================================================================
# NEW STACKS STATE (P0/P1 Integration)
# =============================================================================


class ThoughtTemplate(TypedDict):
    """A single Buffer of Thoughts template.

    BoT (Buffer of Thoughts) provides meta-buffer of thought-templates
    for pre-retrieval reasoning. Costs ~12% of GoT.
    """

    template_id: str
    problem_type: str  # e.g., "spec_decomposition", "qa_validation"
    template_content: str
    instantiated: bool
    instantiation_result: Optional[str]
    usage_count: int
    last_used_at: Optional[str]


class ThoughtBufferState(TypedDict):
    """Buffer of Thoughts (BoT) state.

    Integrated before GoT for pre-retrieval reasoning.
    See: MIGRATION_V2_TO_LANGGRAPH.md Section 3.1
    """

    enabled: bool
    meta_buffer: List[ThoughtTemplate]
    active_template_id: Optional[str]
    buffer_version: str
    last_buffer_update: Optional[str]

    # Metrics for BoT efficiency
    templates_retrieved: int
    templates_instantiated: int
    cost_savings_vs_got: Optional[float]  # Should be ~88%


class RAGContextState(TypedDict):
    """Active RAG (FLARE) state.

    Active RAG performs iterative retrieval during generation,
    not just at the start. Integrated in SPEC/QA/PLANNING nodes.
    See: MIGRATION_V2_TO_LANGGRAPH.md Section 2.1
    """

    enabled: bool
    current_query: Optional[str]
    retrieved_documents: List[str]  # Document IDs/refs
    retrieval_iterations: int
    max_iterations: int

    # Quality metrics
    relevance_scores: List[float]
    last_retrieval_at: Optional[str]

    # GraphRAG integration (if Neo4j is available)
    graph_context_used: bool
    graph_entities: List[str]
    graph_relations: List[str]


class NeMoRailResult(TypedDict):
    """Result from a NeMo Guardrails check."""

    rail_name: str
    passed: bool
    blocked_reason: Optional[str]
    confidence: float
    checked_at: str


class LLMGuardResult(TypedDict):
    """Result from an LLM Guard scan."""

    scanner_name: str
    passed: bool
    risk_score: float
    findings: List[str]
    scanned_at: str


class SecurityContextState(TypedDict):
    """Security context state (NeMo Guardrails + LLM Guard).

    Provides defense in depth with:
    - NeMo Guardrails: dialog flows, topical rails, safety
    - LLM Guard: security scanners, PII detection

    IMPORTANT (NF-013 FIX): Fields use Optional[bool] to support "not yet checked" state.
    - None = not yet checked (should be treated as FAIL)
    - True = checked and passed
    - False = checked and failed

    See: MIGRATION_V2_TO_LANGGRAPH.md Section 4.4, 4.5
    """

    # NeMo Guardrails state
    nemo_enabled: bool
    nemo_config_path: Optional[str]
    nemo_input_rails_passed: Optional[bool]  # NF-013: None = not yet checked
    nemo_output_rails_passed: Optional[bool]  # NF-013: None = not yet checked
    nemo_rail_results: List[NeMoRailResult]

    # LLM Guard state
    llm_guard_enabled: bool
    llm_guard_input_passed: Optional[bool]  # NF-013: None = not yet checked
    llm_guard_output_passed: Optional[bool]  # NF-013: None = not yet checked
    llm_guard_results: List[LLMGuardResult]

    # Combined security status
    all_checks_passed: Optional[bool]  # NF-013: None = not yet checked
    blocked_at_stage: Optional[str]  # "input", "output", or None
    security_score: float  # 0.0-1.0, higher is safer (starts at 0.0)


def is_security_verified(security_context: SecurityContextState) -> bool:
    """Check if all security checks have been run AND passed.

    NF-013 FIX: None means "not yet checked" and should be treated as FAILURE.
    Only True means the check passed.

    Args:
        security_context: The security context state to check

    Returns:
        True only if ALL checks have explicitly passed (True).
        False if any check is None (not yet run) or False (failed).
    """
    nemo_input_ok = security_context.get("nemo_input_rails_passed") is True
    nemo_output_ok = security_context.get("nemo_output_rails_passed") is True
    guard_input_ok = security_context.get("llm_guard_input_passed") is True
    guard_output_ok = security_context.get("llm_guard_output_passed") is True
    all_ok = security_context.get("all_checks_passed") is True

    return nemo_input_ok and nemo_output_ok and guard_input_ok and guard_output_ok and all_ok


def get_security_status(security_context: SecurityContextState) -> str:
    """Get human-readable security status.

    Args:
        security_context: The security context state to check

    Returns:
        "NOT_CHECKED" if any check is None
        "PASSED" if all checks are True
        "FAILED" if any check is False
    """
    checks = [
        security_context.get("nemo_input_rails_passed"),
        security_context.get("nemo_output_rails_passed"),
        security_context.get("llm_guard_input_passed"),
        security_context.get("llm_guard_output_passed"),
        security_context.get("all_checks_passed"),
    ]

    if any(c is None for c in checks):
        return "NOT_CHECKED"
    elif all(c is True for c in checks):
        return "PASSED"
    else:
        return "FAILED"


class CleanLabResult(TypedDict):
    """Result from Cleanlab TLM hallucination detection."""

    claim: str
    trustworthiness_score: float
    is_hallucination: bool
    checked_at: str


class HallucinationState(TypedDict):
    """Hallucination detection state (Cleanlab TLM).

    Cleanlab provides hallucination detection with confidence scores.
    See: MIGRATION_V2_TO_LANGGRAPH.md #12 (P1)
    """

    enabled: bool
    claims_checked: int
    hallucinations_detected: int
    results: List[CleanLabResult]
    average_trustworthiness: Optional[float]


class Z3InvariantResult(TypedDict):
    """Result from Z3 formal invariant verification."""

    invariant_name: str
    gate_id: str
    proved: bool
    proof_time_ms: float
    counter_example: Optional[str]
    verified_at: str


class FormalVerificationState(TypedDict):
    """Formal verification state (Z3).

    Z3 provides formal invariant verification for critical gates G0, G4, G8.
    See: MIGRATION_V2_TO_LANGGRAPH.md Section 3.4
    """

    enabled: bool
    invariants_verified: int
    invariants_failed: int
    results: List[Z3InvariantResult]
    last_verification_at: Optional[str]


# =============================================================================
# SPEC PHASE STATE (IronClad Integration)
# =============================================================================


class SpecPhaseState(TypedDict):
    """State for the spec phase (IronClad Spec Kit integration).

    Contains the outputs from all 4 IronClad subsystems:
    1. Masticator: Raw input -> EARS requirements
    2. Gap Filler: Enriched requirements with filled gaps
    3. Journey Completer: Generated user journeys
    4. Guardian: Validation results

    This state is populated by the init_node when raw_spec_input is provided.
    """

    # Phase 1: Masticator output
    ears_requirements: List[Dict[str, Any]]  # List of EARSRequirement.to_dict()
    clarification_questions: List[str]  # Questions for stakeholder

    # Phase 2: Gap Filler output
    gap_report: Optional[Dict[str, Any]]  # GapReport.to_dict()
    gaps_found: int
    gaps_auto_filled: int
    gaps_need_clarification: int

    # Phase 3: Journey Completer output
    journeys: List[Dict[str, Any]]  # List of JourneySpec.to_dict()
    journey_count: int
    happy_path_id: Optional[str]

    # Phase 4: Guardian output
    validation_report: Optional[Dict[str, Any]]  # ValidationReport.to_dict()
    validation_passed: bool
    gates_passed: int
    gates_failed: int
    gates_warnings: int

    # Metadata
    ironclad_version: str
    processed_at: Optional[str]  # ISO timestamp
    duration_ms: int


def create_empty_spec_phase_state() -> SpecPhaseState:
    """Create an empty SpecPhaseState for initialization.

    Returns:
        Empty SpecPhaseState with default values.
    """
    return SpecPhaseState(
        # Phase 1
        ears_requirements=[],
        clarification_questions=[],
        # Phase 2
        gap_report=None,
        gaps_found=0,
        gaps_auto_filled=0,
        gaps_need_clarification=0,
        # Phase 3
        journeys=[],
        journey_count=0,
        happy_path_id=None,
        # Phase 4
        validation_report=None,
        validation_passed=False,
        gates_passed=0,
        gates_failed=0,
        gates_warnings=0,
        # Metadata
        ironclad_version="3.1.0",
        processed_at=None,
        duration_ms=0,
    )


# =============================================================================
# SURGICAL REWORK TYPEDDICTS
# =============================================================================


class DiagnosisEntry(TypedDict):
    """A diagnosis entry for a task failure.

    Contains all information needed to generate a surgical patch.
    """

    diagnosis_id: str
    task_id: str
    file_path: str
    line_number: Optional[int]
    column_number: Optional[int]
    function_name: Optional[str]
    class_name: Optional[str]
    failure_category: str  # FailureCategory value
    severity: str  # low, medium, high, critical
    error_message: str
    root_cause: str
    expected_behavior: str
    actual_behavior: str
    code_snippet: str
    stack_trace: Optional[str]
    suggested_fix: str
    fix_confidence: float
    alternative_fixes: List[str]
    created_at: str  # ISO timestamp
    diagnosis_time_ms: int


class PatchEntry(TypedDict):
    """A patch entry representing a surgical fix.

    Represents the minimum change needed to fix a diagnosed issue.
    """

    patch_id: str
    task_id: str
    diagnosis_id: str
    file_path: str
    line_start: int
    line_end: int
    change_type: str  # add, remove, replace
    old_content: Optional[str]
    new_content: str
    context_before: str
    context_after: str
    explanation: str
    confidence: float
    created_at: str  # ISO timestamp
    applied_at: Optional[str]
    verified_at: Optional[str]


class TaskLevelEntry(TypedDict):
    """Entry tracking a single task in the surgical rework system.

    Maintains the full state of a task through execution and rework cycles.
    """

    task_id: str
    sprint_id: str
    sequence: int
    name: str
    description: str
    deliverables: List[str]
    depends_on: List[str]
    status: str  # TaskLevelStatus value
    attempts: int
    max_attempts: int
    code_generated: Optional[str]
    confidence_score: float
    execution_time_ms: int
    tokens_used: int
    diagnoses: List[DiagnosisEntry]
    patches: List[PatchEntry]
    repair_history: List[Dict[str, Any]]
    checkpoint_path: Optional[str]
    created_at: str  # ISO timestamp
    started_at: Optional[str]
    completed_at: Optional[str]
    last_validation_at: Optional[str]
    rework_phase: Optional[str]  # ReworkPhase value when in REPAIRING


class SurgicalReworkState(TypedDict):
    """State for the surgical rework system.

    Contains all task-level state for fine-grained execution and rework.

    Key Invariants:
    - Each task has independent lifecycle
    - Tasks can be repaired without affecting others
    - Parallel groups execute concurrently
    - Escalation is per-task, not per-sprint
    """

    enabled: bool
    tasks: Dict[str, TaskLevelEntry]  # task_id -> TaskLevelEntry
    parallel_groups: List[List[str]]  # Groups of task_ids that can run in parallel
    current_group_index: int
    tasks_pending: int
    tasks_executing: int
    tasks_passed: int
    tasks_failed: int
    tasks_escalated: int
    total_repairs: int
    total_patches_applied: int
    integration_validation_passed: bool
    integration_issues: List[Dict[str, Any]]
    last_task_completed: Optional[str]
    last_task_failed: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]


# =============================================================================
# MAIN PIPELINE STATE
# =============================================================================


class PipelineState(TypedDict):
    """Complete LangGraph Pipeline State.

    This TypedDict contains ALL fields required for the LangGraph control plane,
    following the schema from PIPELINE_V3_MASTER_PLAN.md Section 10.4.

    Sections:
    1. Identity (run_id, sprint_id, attempt, phase)
    2. Pointers (run_dir, trace_id, etc.)
    3. Policy Snapshot (timeouts, retries, flags)
    4. Stack Health (required stacks, health status)
    5. Context (context_pack, constitution_ref)
    6. Governance (handoffs, approvals, signoffs, milestones)
    7. Execution (tasks, temporal, crew_result)
    8. QA/Gates (gate_status, evidence, artifacts)
    9. Memory/Learning (amem, mem0, qdrant, letta refs)
    10. Events and Resume (node_history, event offsets)
    11. Timestamps (started_at, updated_at, completed_at)
    12. Errors (error list with context)
    13. NEW: Thought Buffer (BoT for pre-retrieval reasoning)
    14. NEW: RAG Context (Active RAG/FLARE state)
    15. NEW: Security Context (NeMo Guardrails + LLM Guard)
    16. NEW: Hallucination State (Cleanlab TLM)
    17. NEW: Formal Verification (Z3 invariants)

    Invariants (from Section 10.4.3):
    - I1 (Namespacing): any persistent key contains run_id + sprint_id
    - I2 (Idempotency): each node has deterministic idempotency_key
    - I3 (Phase order): INIT -> SPEC -> PLAN -> EXEC -> QA -> VOTE -> DONE
    - I4 (Gates before signoff): no valid signoff if gates.status != PASS
    - I5 (Approvals before signoff): supervisor cannot sign off without approvals
    - I6 (Evidence integrity): bundle has MANIFEST.json with matching hash
    - I7 (Artifacts existence): required artifacts exist for each phase
    - I8 (Dashboard contract): event_log uses schema_version, event_id, type
    - I9 (SAFE_HALT): if safe_halt is active, graph enters HALT
    - I10 (Policy snapshot): exists per run, changes are explicit
    - I11 (Observability non-blocking): sinks never block the pipeline
    """

    # =========================================================================
    # 1. IDENTITY
    # =========================================================================
    run_id: str
    sprint_id: str
    attempt: int
    phase: str  # SprintPhase value
    status: str  # PipelineStatus value

    # =========================================================================
    # 2. POINTERS
    # =========================================================================
    run_dir: str
    repo_root: str
    trace_id: Optional[str]
    agentops_run_id: Optional[str]
    literal_session_id: Optional[str]
    phoenix_session_id: Optional[str]

    # =========================================================================
    # 3. POLICY SNAPSHOT
    # =========================================================================
    policy: PolicySnapshotState

    # =========================================================================
    # 4. STACK HEALTH SNAPSHOT
    # =========================================================================
    required_stacks: List[str]
    stack_health: Dict[str, StackHealthEntry]

    # =========================================================================
    # 5. CONTEXT
    # =========================================================================
    context_pack: Optional[ContextPackState]
    constitution_ref: Optional[str]

    # =========================================================================
    # 6. GOVERNANCE
    # =========================================================================
    # Key: "{from_agent}->{to_agent}"
    handoffs: Dict[str, HandoffEntry]

    # Key: approvals[approver][subordinate]
    approvals: Dict[str, Dict[str, ApprovalEntry]]

    # Key: signoffs[agent_id]
    signoffs: Dict[str, SignoffEntry]

    # Key: milestones[agent_id][milestone_id]
    milestones: Dict[str, Dict[str, MilestoneEntry]]

    # =========================================================================
    # 7. EXECUTION
    # =========================================================================
    tasks: Dict[str, TaskEntry]
    temporal: TemporalState
    crew_result: Optional[Dict[str, Any]]
    # Granular tasks from spec decomposition (GAP-3 fix)
    # Each task has: deliverable, requirements, invariants, edge_cases, task_prompt
    granular_tasks: List[Dict[str, Any]]

    # =========================================================================
    # 8. QA/GATES
    # =========================================================================
    gate_track: str
    gate_selection_path: str
    gate_report_path: Optional[str]
    gate_status: str  # GateStatus value
    gates_passed: List[str]
    gates_failed: List[str]
    evidence_bundle_dir: Optional[str]
    evidence_manifest_path: Optional[str]
    qa_artifacts: Dict[str, str]

    # =========================================================================
    # 9. MEMORY/LEARNING (references only)
    # =========================================================================
    amem_notes: List[str]
    mem0_refs: List[str]
    qdrant_refs: List[str]
    letta_refs: List[str]

    # =========================================================================
    # 10. EVENTS AND RESUME
    # =========================================================================
    node_history: List[NodeHistoryEntry]
    last_event_id_redis: Optional[str]
    last_event_offset_file: Optional[int]

    # Resume flags (set by checkpoints to skip phases)
    _skip_exec: Optional[bool]  # Skip EXEC phase on resume
    _resume_from_gates: Optional[bool]  # Resume directly to gates/QA

    # =========================================================================
    # 11. TIMESTAMPS
    # =========================================================================
    started_at: str
    updated_at: str
    completed_at: Optional[str]

    # =========================================================================
    # 12. ERRORS
    # =========================================================================
    errors: List[ErrorEntry]

    # =========================================================================
    # 13. CHECKPOINTS (LangGraph specific)
    # =========================================================================
    checkpoints: List[str]  # List of checkpoint IDs
    last_checkpoint_at: Optional[str]
    checkpoint_storage: str  # "file", "redis", "hybrid"

    # =========================================================================
    # 14. NEW: THOUGHT BUFFER (Buffer of Thoughts - BoT)
    # =========================================================================
    thought_buffer: ThoughtBufferState

    # =========================================================================
    # 15. NEW: RAG CONTEXT (Active RAG / FLARE)
    # =========================================================================
    rag_context: RAGContextState

    # =========================================================================
    # 16. NEW: SECURITY CONTEXT (NeMo Guardrails + LLM Guard)
    # =========================================================================
    security_context: SecurityContextState

    # =========================================================================
    # 17. NEW: HALLUCINATION STATE (Cleanlab TLM)
    # =========================================================================
    hallucination_state: HallucinationState

    # =========================================================================
    # 18. NEW: FORMAL VERIFICATION (Z3)
    # =========================================================================
    formal_verification: FormalVerificationState

    # =========================================================================
    # 19. NEW: SURGICAL REWORK (Task-level execution and rework)
    # =========================================================================
    surgical_rework: SurgicalReworkState

    # =========================================================================
    # 20. NEW: SPEC PHASE (IronClad Spec Kit integration)
    # =========================================================================
    spec_phase: Optional[SpecPhaseState]

    # =========================================================================
    # 21. NEW: RAW SPEC INPUT (for IronClad processing)
    # =========================================================================
    raw_spec_input: Optional[str]  # Raw input for Masticator
    happy_path_journey: Optional[Dict[str, Any]]  # Happy path for JourneyCompleter


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def validate_identifier(value: str, name: str, pattern: Optional[str] = None) -> None:
    """Validate an identifier string.

    Args:
        value: The value to validate.
        name: The name of the identifier (for error messages).
        pattern: Optional regex pattern to match.

    Raises:
        ValueError: If validation fails.
    """
    import re

    if not value:
        raise ValueError(f"{name} cannot be empty")

    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string, got {type(value).__name__}")

    # Check for invalid characters that could cause issues
    if "\x00" in value or "\n" in value or "\r" in value:
        raise ValueError(f"{name} contains invalid characters (null/newline)")

    # Check reasonable length
    if len(value) > 256:
        raise ValueError(f"{name} too long ({len(value)} > 256 chars)")

    # Optional pattern matching
    if pattern:
        if not re.match(pattern, value):
            raise ValueError(f"{name} '{value}' does not match expected pattern '{pattern}'")


def create_initial_state(
    run_id: str,
    sprint_id: str,
    run_dir: str,
    repo_root: str = ".",
    policy: Optional[PolicySnapshotState] = None,
) -> PipelineState:
    """Create an initial PipelineState with sensible defaults.

    Args:
        run_id: Unique identifier for this pipeline run.
        sprint_id: Sprint identifier (e.g., "S00", "S01").
        run_dir: Path to the run output directory.
        repo_root: Path to the repository root.
        policy: Optional policy snapshot. Uses defaults if not provided.

    Returns:
        A fully initialized PipelineState ready for LangGraph execution.

    Raises:
        ValueError: If run_id or sprint_id are invalid.
    """
    # Input validation
    validate_identifier(run_id, "run_id")
    validate_identifier(sprint_id, "sprint_id", pattern=r"^S\d{2,3}[A-Z]?$")

    if not run_dir:
        raise ValueError("run_dir cannot be empty")

    now = datetime.now(timezone.utc).isoformat()

    if policy is None:
        policy = create_policy_snapshot()

    return PipelineState(
        # Identity
        run_id=run_id,
        sprint_id=sprint_id,
        attempt=1,
        phase=SprintPhase.INIT.value,
        status=PipelineStatus.INITIALIZED.value,

        # Pointers
        run_dir=run_dir,
        repo_root=repo_root,
        trace_id=None,
        agentops_run_id=None,
        literal_session_id=None,
        phoenix_session_id=None,

        # Policy
        policy=policy,

        # Stack Health
        required_stacks=[],
        stack_health={},

        # Context
        context_pack=None,
        constitution_ref=None,

        # Governance
        handoffs={},
        approvals={},
        signoffs={},
        milestones={},

        # Execution
        tasks={},
        temporal=TemporalState(
            workflow_id=None,
            run_id=None,
            activity_count=0,
            last_activity_at=None,
        ),
        crew_result=None,
        granular_tasks=[],  # GAP-3 fix: spec decomposition tasks

        # Gates
        gate_track="A",
        gate_selection_path="",
        gate_report_path=None,
        gate_status=GateStatus.PENDING.value,
        gates_passed=[],
        gates_failed=[],
        evidence_bundle_dir=None,
        evidence_manifest_path=None,
        qa_artifacts={},

        # Memory
        amem_notes=[],
        mem0_refs=[],
        qdrant_refs=[],
        letta_refs=[],

        # Events/Resume
        node_history=[],
        last_event_id_redis=None,
        last_event_offset_file=None,
        _skip_exec=None,  # Set by checkpoint to skip exec phase
        _resume_from_gates=None,  # Set by checkpoint to resume to gates

        # Timestamps
        started_at=now,
        updated_at=now,
        completed_at=None,

        # Errors
        errors=[],

        # Checkpoints
        checkpoints=[],
        last_checkpoint_at=None,
        checkpoint_storage="file",

        # NEW: Thought Buffer (BoT)
        thought_buffer=ThoughtBufferState(
            enabled=True,
            meta_buffer=[],
            active_template_id=None,
            buffer_version="1.0.0",
            last_buffer_update=None,
            templates_retrieved=0,
            templates_instantiated=0,
            cost_savings_vs_got=None,
        ),

        # NEW: RAG Context
        rag_context=RAGContextState(
            enabled=True,
            current_query=None,
            retrieved_documents=[],
            retrieval_iterations=0,
            max_iterations=3,
            relevance_scores=[],
            last_retrieval_at=None,
            graph_context_used=False,
            graph_entities=[],
            graph_relations=[],
        ),

        # NEW: Security Context
        # NF-013 FIX: FAIL-CLOSED defaults
        # None = not yet checked (treated as failure until proven otherwise)
        # security_score starts at 0.0 (worst case until proven otherwise)
        security_context=SecurityContextState(
            nemo_enabled=True,
            nemo_config_path=None,
            nemo_input_rails_passed=None,  # NF-013: None = not yet checked
            nemo_output_rails_passed=None,  # NF-013: None = not yet checked
            nemo_rail_results=[],
            llm_guard_enabled=True,
            llm_guard_input_passed=None,  # NF-013: None = not yet checked
            llm_guard_output_passed=None,  # NF-013: None = not yet checked
            llm_guard_results=[],
            all_checks_passed=None,  # NF-013: None = not yet checked
            blocked_at_stage=None,
            security_score=0.0,  # NF-013: Assume worst until proven otherwise
        ),

        # NEW: Hallucination State
        hallucination_state=HallucinationState(
            enabled=True,
            claims_checked=0,
            hallucinations_detected=0,
            results=[],
            average_trustworthiness=None,
        ),

        # NEW: Formal Verification
        formal_verification=FormalVerificationState(
            enabled=True,
            invariants_verified=0,
            invariants_failed=0,
            results=[],
            last_verification_at=None,
        ),

        # NEW: Surgical Rework
        surgical_rework=SurgicalReworkState(
            enabled=True,
            tasks={},
            parallel_groups=[],
            current_group_index=0,
            tasks_pending=0,
            tasks_executing=0,
            tasks_passed=0,
            tasks_failed=0,
            tasks_escalated=0,
            total_repairs=0,
            total_patches_applied=0,
            integration_validation_passed=False,
            integration_issues=[],
            last_task_completed=None,
            last_task_failed=None,
            started_at=None,
            completed_at=None,
        ),

        # NEW: Spec Phase (IronClad)
        spec_phase=None,  # Will be populated by init_node if raw_spec_input provided

        # NEW: Raw Spec Input
        raw_spec_input=None,
        happy_path_journey=None,
    )


def create_policy_snapshot(
    task_timeout_seconds: int = 3600,
    sprint_timeout_seconds: int = 14400,
    max_retry_attempts: int = 3,
    use_temporal: bool = True,
    use_zap_validation: bool = True,
    run_test_guardrail: bool = False,
    cost_policy_ref: Optional[str] = None,
    worker_limit_profile: Optional[str] = None,
    fraud_detection_enabled: bool = True,
    executive_verification_enabled: bool = True,
    completeness_gate_enabled: bool = True,
    truthfulness_validation_enabled: bool = True,
    rework_system_enabled: bool = True,
) -> PolicySnapshotState:
    """Create a policy snapshot with the given configuration.

    This snapshot freezes the policy at run start to prevent accidental
    drift during migration (Invariant I10).

    HIGH-008 Fix: Validates all policy values to prevent invalid configurations
    that would cause mysterious failures later in the pipeline.

    Args:
        task_timeout_seconds: Timeout per task (default: 1 hour). Must be > 0.
        sprint_timeout_seconds: Timeout per sprint (default: 4 hours). Must be >= task_timeout.
        max_retry_attempts: Maximum retry attempts. Must be >= 1.
        use_temporal: Whether to use Temporal for durable execution.
        use_zap_validation: Whether to use ZAP message validation.
        run_test_guardrail: Whether to run pytest before each sprint.
        cost_policy_ref: Reference to cost policy (if any).
        worker_limit_profile: Reference to worker limit profile (if any).
        fraud_detection_enabled: Enable fraud detection in signoffs.
        executive_verification_enabled: Enable executive verification.
        completeness_gate_enabled: Enable completeness gate.
        truthfulness_validation_enabled: Enable truthfulness validation.
        rework_system_enabled: Enable rework system (vs blocking).

    Returns:
        PolicySnapshotState with the specified configuration.

    Raises:
        ValueError: If any policy value is invalid.
    """
    # HIGH-008 Fix: Validate policy values
    if task_timeout_seconds <= 0:
        raise ValueError(f"task_timeout_seconds must be > 0, got {task_timeout_seconds}")
    if sprint_timeout_seconds <= 0:
        raise ValueError(f"sprint_timeout_seconds must be > 0, got {sprint_timeout_seconds}")
    if sprint_timeout_seconds < task_timeout_seconds:
        raise ValueError(
            f"sprint_timeout_seconds ({sprint_timeout_seconds}) must be >= "
            f"task_timeout_seconds ({task_timeout_seconds})"
        )
    if max_retry_attempts < 1:
        raise ValueError(f"max_retry_attempts must be >= 1, got {max_retry_attempts}")

    return PolicySnapshotState(
        task_timeout_seconds=task_timeout_seconds,
        sprint_timeout_seconds=sprint_timeout_seconds,
        max_retry_attempts=max_retry_attempts,
        use_temporal=use_temporal,
        use_zap_validation=use_zap_validation,
        run_test_guardrail=run_test_guardrail,
        cost_policy_ref=cost_policy_ref,
        worker_limit_profile=worker_limit_profile,
        fraud_detection_enabled=fraud_detection_enabled,
        executive_verification_enabled=executive_verification_enabled,
        completeness_gate_enabled=completeness_gate_enabled,
        truthfulness_validation_enabled=truthfulness_validation_enabled,
        rework_system_enabled=rework_system_enabled,
    )


def create_stack_health_entry(
    healthy: bool,
    error: Optional[str] = None,
    latency_ms: Optional[float] = None,
) -> StackHealthEntry:
    """Create a stack health entry.

    Args:
        healthy: Whether the stack is healthy.
        error: Error message if unhealthy.
        latency_ms: Health check latency in milliseconds.

    Returns:
        StackHealthEntry with the specified values.
    """
    return StackHealthEntry(
        healthy=healthy,
        checked_at=datetime.now(timezone.utc).isoformat(),
        error=error,
        latency_ms=latency_ms,
    )


def create_node_history_entry(
    node: str,
    status: str = "started",
    idempotency_key: Optional[str] = None,
    error: Optional[str] = None,
) -> NodeHistoryEntry:
    """Create a node history entry for LangGraph execution tracking.

    Args:
        node: Name of the node being executed.
        status: Current status ("started", "completed", "failed", "skipped").
        idempotency_key: Deterministic key for idempotency (auto-generated if None).
        error: Error message if failed.

    Returns:
        NodeHistoryEntry with the specified values.
    """
    now = datetime.now(timezone.utc).isoformat()

    if idempotency_key is None:
        idempotency_key = f"{node}_{uuid.uuid4().hex[:8]}"

    return NodeHistoryEntry(
        node=node,
        status=status,
        started_at=now,
        finished_at=now if status in ("completed", "failed", "skipped") else None,
        idempotency_key=idempotency_key,
        error=error,
    )


def create_error_entry(
    where: str,
    error: str,
    traceback: Optional[str] = None,
    recoverable: bool = True,
) -> ErrorEntry:
    """Create an error entry for tracking pipeline errors.

    Args:
        where: Node/phase where error occurred.
        error: Error message.
        traceback: Full traceback if available.
        recoverable: Whether the error is recoverable.

    Returns:
        ErrorEntry with the specified values.
    """
    return ErrorEntry(
        where=where,
        error=error,
        ts=datetime.now(timezone.utc).isoformat(),
        traceback=traceback,
        recoverable=recoverable,
    )


# =============================================================================
# STATE UPDATE HELPERS
# =============================================================================


def update_phase(state: PipelineState, new_phase: SprintPhase) -> PipelineState:
    """Update the phase in the state.

    Also updates the `updated_at` timestamp.

    Args:
        state: Current state.
        new_phase: New phase to set.

    Returns:
        Updated state (new dict, original unchanged).
    """
    return {
        **state,
        "phase": new_phase.value,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def update_status(state: PipelineState, new_status: PipelineStatus) -> PipelineState:
    """Update the status in the state.

    Also updates the `updated_at` timestamp.
    Sets `completed_at` if status is COMPLETED or FAILED.

    Args:
        state: Current state.
        new_status: New status to set.

    Returns:
        Updated state (new dict, original unchanged).
    """
    now = datetime.now(timezone.utc).isoformat()
    updates: Dict[str, Any] = {
        "status": new_status.value,
        "updated_at": now,
    }

    if new_status in (PipelineStatus.COMPLETED, PipelineStatus.FAILED, PipelineStatus.HALTED):
        updates["completed_at"] = now

    return {**state, **updates}


def add_error(state: PipelineState, error_entry: ErrorEntry) -> PipelineState:
    """Add an error entry to the state.

    Args:
        state: Current state.
        error_entry: Error entry to add.

    Returns:
        Updated state with the new error.
    """
    return {
        **state,
        "errors": [*state["errors"], error_entry],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def add_checkpoint(state: PipelineState, checkpoint_id: str) -> PipelineState:
    """Add a checkpoint ID to the state.

    Args:
        state: Current state.
        checkpoint_id: Checkpoint identifier to add.

    Returns:
        Updated state with the new checkpoint.
    """
    now = datetime.now(timezone.utc).isoformat()
    return {
        **state,
        "checkpoints": [*state["checkpoints"], checkpoint_id],
        "last_checkpoint_at": now,
        "updated_at": now,
    }


def record_handoff(
    state: PipelineState,
    from_agent: str,
    to_agent: str,
    what_i_want: str,
    why_i_want: str,
    expected_behavior: str,
    correlation_id: Optional[str] = None,
) -> PipelineState:
    """Record a handoff in the state.

    Args:
        state: Current state.
        from_agent: Agent initiating the handoff.
        to_agent: Agent receiving the handoff.
        what_i_want: What the supervisor wants done.
        why_i_want: Why this work is needed.
        expected_behavior: Expected behavior/outcome.
        correlation_id: Optional correlation ID.

    Returns:
        Updated state with the new handoff.
    """
    key = f"{from_agent}->{to_agent}"
    entry = HandoffEntry(
        from_agent=from_agent,
        to_agent=to_agent,
        what_i_want=what_i_want,
        why_i_want=why_i_want,
        expected_behavior=expected_behavior,
        recorded_at=datetime.now(timezone.utc).isoformat(),
        correlation_id=correlation_id,
    )

    return {
        **state,
        "handoffs": {**state["handoffs"], key: entry},
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def record_signoff(
    state: PipelineState,
    agent_id: str,
    approved: bool,
    what_i_did: str,
    why_this_way: str,
    how_it_works: str,
    artifacts_verified: Optional[List[str]] = None,
    checks_passed: Optional[List[str]] = None,
    observations: Optional[str] = None,
) -> PipelineState:
    """Record a signoff in the state.

    Args:
        state: Current state.
        agent_id: Agent providing the signoff.
        approved: Whether approved.
        what_i_did: Description of work done.
        why_this_way: Justification for approach.
        how_it_works: Technical explanation.
        artifacts_verified: List of verified artifacts.
        checks_passed: List of checks that passed.
        observations: Optional observations.

    Returns:
        Updated state with the new signoff.
    """
    entry = SignoffEntry(
        agent_id=agent_id,
        approved=approved,
        signed_at=datetime.now(timezone.utc).isoformat(),
        what_i_did=what_i_did,
        why_this_way=why_this_way,
        how_it_works=how_it_works,
        artifacts_verified=artifacts_verified or [],
        checks_passed=checks_passed or [],
        observations=observations,
    )

    return {
        **state,
        "signoffs": {**state["signoffs"], agent_id: entry},
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


# =============================================================================
# CONTEXT PACK HELPERS (GAP-1)
# =============================================================================


def extract_context_pack_data(context_pack: Any) -> ContextPackState:
    """Extract full data from ContextPack into ContextPackState dict.

    This function bridges the spec_kit_loader.ContextPack dataclass
    to the LangGraph state TypedDict, extracting all relevant data
    for propagation through the pipeline.

    Args:
        context_pack: A ContextPack instance from spec_kit_loader.

    Returns:
        ContextPackState dict with all extracted data.
    """
    import hashlib

    # Calculate SHA256 of raw content if available
    raw_content = getattr(context_pack, "raw_content", "")
    sha256 = hashlib.sha256(raw_content.encode()).hexdigest() if raw_content else ""

    # Extract functional requirements from v2 manifest or legacy
    functional_requirements: List[Dict[str, Any]] = []
    if hasattr(context_pack, "intent_manifest") and context_pack.intent_manifest:
        manifest = context_pack.intent_manifest
        for rf in getattr(manifest, "functional_requirements", []):
            if isinstance(rf, dict):
                functional_requirements.append(rf)
            else:
                functional_requirements.append({
                    "id": getattr(rf, "id", ""),
                    "description": getattr(rf, "description", str(rf)),
                    "priority": getattr(rf, "priority", "P1"),
                })

    # Extract invariants from both v1 and v2 formats
    invariants: List[Dict[str, Any]] = []
    # v1 format (dict)
    v1_invariants = getattr(context_pack, "invariants", {})
    if isinstance(v1_invariants, dict):
        for inv_id, rule in v1_invariants.items():
            invariants.append({"id": inv_id, "rule": rule})
    # v2 format (from contracts)
    if hasattr(context_pack, "intent_manifest") and context_pack.intent_manifest:
        contracts = getattr(context_pack.intent_manifest, "contracts", None)
        if contracts:
            for inv in getattr(contracts, "invariants", []):
                invariants.append({
                    "id": getattr(inv, "id", ""),
                    "rule": getattr(inv, "rule", ""),
                    "enforcement": getattr(inv, "enforcement", ""),
                    "class_name": getattr(inv, "class_name", ""),
                })

    # Extract edge cases from boundaries
    edge_cases: List[Dict[str, Any]] = []
    if hasattr(context_pack, "intent_manifest") and context_pack.intent_manifest:
        boundaries = getattr(context_pack.intent_manifest, "boundaries", None)
        if boundaries:
            for edge in getattr(boundaries, "edge_cases", []):
                edge_cases.append({
                    "id": getattr(edge, "id", ""),
                    "case": getattr(edge, "case", ""),
                    "expected_behavior": getattr(edge, "expected_behavior", ""),
                    "test_id": getattr(edge, "test_id", ""),
                })

    # Extract acceptance criteria from quality
    acceptance_criteria: List[Dict[str, Any]] = []
    if hasattr(context_pack, "intent_manifest") and context_pack.intent_manifest:
        quality = getattr(context_pack.intent_manifest, "quality", None)
        if quality:
            for ac in getattr(quality, "acceptance_criteria", []):
                acceptance_criteria.append({
                    "id": getattr(ac, "id", ""),
                    "criteria": getattr(ac, "criteria", ""),
                    "measurement": getattr(ac, "measurement", ""),
                    "threshold": getattr(ac, "threshold", ""),
                })

    # Extract DoD checklist from quality
    dod_checklist: List[Dict[str, Any]] = []
    if hasattr(context_pack, "intent_manifest") and context_pack.intent_manifest:
        quality = getattr(context_pack.intent_manifest, "quality", None)
        if quality:
            dod = getattr(quality, "definition_of_done", {})
            if isinstance(dod, dict):
                for category, items in dod.items():
                    for item in (items if isinstance(items, list) else [items]):
                        dod_checklist.append({
                            "item": item,
                            "required": True,
                            "category": category,
                        })

    # Extract behaviors from Gherkin features
    behaviors: List[Dict[str, Any]] = []
    if hasattr(context_pack, "intent_manifest") and context_pack.intent_manifest:
        behaviors_data = getattr(context_pack.intent_manifest, "behaviors", None)
        if behaviors_data:
            for feature in getattr(behaviors_data, "features", []):
                for scenario in getattr(feature, "scenarios", []):
                    behaviors.append({
                        "feature": getattr(feature, "name", ""),
                        "name": getattr(scenario, "name", ""),
                        "type": getattr(scenario, "type", "happy_path"),
                        "given": getattr(scenario, "given", []),
                        "when": getattr(scenario, "when", []),
                        "then": getattr(scenario, "then", []),
                        "test_id": getattr(scenario, "test_id", ""),
                    })

    # Build deliverables_spec with mapped requirements
    deliverables = getattr(context_pack, "deliverables", [])
    deliverables_spec: List[Dict[str, Any]] = []
    for deliverable in deliverables:
        spec = {
            "path": deliverable,
            "requirements": [],
            "invariants": [],
            "edge_cases": [],
        }
        # Map RF to deliverable based on traceability or naming
        for rf in functional_requirements:
            files = rf.get("files", [])
            if deliverable in files or not files:
                spec["requirements"].append(rf)
        deliverables_spec.append(spec)

    # Extract traceability mappings
    rf_to_deliverable: Dict[str, str] = {}
    rf_to_test: Dict[str, List[str]] = {}
    inv_to_enforcement: Dict[str, Dict[str, str]] = {}
    if hasattr(context_pack, "intent_manifest") and context_pack.intent_manifest:
        traceability = getattr(context_pack.intent_manifest, "traceability", None)
        if traceability:
            rf_to_deliverable = dict(getattr(traceability, "rf_to_deliverable", {}))
            rf_to_test = dict(getattr(traceability, "rf_to_test", {}))
            inv_to_enforcement = dict(getattr(traceability, "inv_to_enforcement", {}))

    return ContextPackState(
        pack_id=getattr(context_pack, "sprint_id", ""),
        pack_path=f"context_packs/{getattr(context_pack, 'sprint_id', '')}_CONTEXT.md",
        sha256=sha256,
        dependencies=list(getattr(context_pack, "dependencies", {}).keys()),
        deliverables=deliverables,
        objective=getattr(context_pack, "objective", ""),
        rf_count=len(functional_requirements),
        inv_count=len(invariants),
        edge_count=len(edge_cases),
        functional_requirements=functional_requirements,
        invariants=invariants,
        edge_cases=edge_cases,
        acceptance_criteria=acceptance_criteria,
        dod_checklist=dod_checklist,
        behaviors=behaviors,
        deliverables_spec=deliverables_spec,
        rf_to_deliverable=rf_to_deliverable,
        rf_to_test=rf_to_test,
        inv_to_enforcement=inv_to_enforcement,
    )


def get_requirements_for_deliverable(
    state: PipelineState,
    deliverable: str,
) -> Dict[str, List[Dict[str, Any]]]:
    """Get RF/INV/EDGE relevant to a specific deliverable.

    Args:
        state: Pipeline state with context_pack data.
        deliverable: Path to the deliverable file.

    Returns:
        Dict with keys 'requirements', 'invariants', 'edge_cases'.
    """
    context = state.get("context_pack", {})
    if not context:
        return {"requirements": [], "invariants": [], "edge_cases": []}

    deliverables_spec = context.get("deliverables_spec", [])

    # Find matching deliverable spec
    for spec in deliverables_spec:
        if spec.get("path") == deliverable:
            return {
                "requirements": spec.get("requirements", []),
                "invariants": spec.get("invariants", []),
                "edge_cases": spec.get("edge_cases", []),
            }

    # Fallback: return all if no specific mapping
    return {
        "requirements": context.get("functional_requirements", []),
        "invariants": context.get("invariants", []),
        "edge_cases": context.get("edge_cases", []),
    }


def format_context_for_agent(
    state: PipelineState,
    agent_role: str,
    deliverable: Optional[str] = None,
) -> str:
    """Format context pack data as prompt section for agent.

    Args:
        state: Pipeline state with context_pack data.
        agent_role: Role of the agent (e.g., 'ace_exec', 'qa_master').
        deliverable: Optional specific deliverable to focus on.

    Returns:
        Formatted markdown string for injection into agent prompt.
    """
    context = state.get("context_pack", {})
    if not context:
        return "## Context\nNo context pack loaded."

    parts = ["## Sprint Context"]
    parts.append(f"**Objective:** {context.get('objective', 'Not specified')}")
    parts.append(f"**Sprint:** {context.get('pack_id', 'Unknown')}")

    # Get relevant requirements
    if deliverable:
        reqs = get_requirements_for_deliverable(state, deliverable)
    else:
        reqs = {
            "requirements": context.get("functional_requirements", []),
            "invariants": context.get("invariants", []),
            "edge_cases": context.get("edge_cases", []),
        }

    # Format requirements based on agent role
    if agent_role in ("ace_exec", "spec_master", "squad_lead"):
        if reqs["requirements"]:
            parts.append("\n### Functional Requirements")
            for rf in reqs["requirements"][:10]:  # Limit to avoid context overflow
                parts.append(f"- **{rf.get('id', 'RF')}**: {rf.get('description', '')}")

        if reqs["invariants"]:
            parts.append("\n### Invariants to Enforce")
            for inv in reqs["invariants"][:10]:
                parts.append(f"- **{inv.get('id', 'INV')}**: {inv.get('rule', '')}")

        if reqs["edge_cases"]:
            parts.append("\n### Edge Cases to Handle")
            for edge in reqs["edge_cases"][:10]:
                parts.append(f"- **{edge.get('id', 'EDGE')}**: {edge.get('case', '')} -> {edge.get('expected_behavior', '')}")

    elif agent_role in ("qa_master", "auditor"):
        # QA agents need acceptance criteria and DoD
        ac = context.get("acceptance_criteria", [])
        if ac:
            parts.append("\n### Acceptance Criteria")
            for criterion in ac[:10]:
                parts.append(f"- **{criterion.get('id', 'AC')}**: {criterion.get('criteria', '')} (threshold: {criterion.get('threshold', 'N/A')})")

        dod = context.get("dod_checklist", [])
        if dod:
            parts.append("\n### Definition of Done")
            for item in dod[:10]:
                parts.append(f"- [ ] {item.get('item', '')} ({item.get('category', '')})")

    elif agent_role in ("ceo", "presidente"):
        # Executive agents need high-level summary
        parts.append(f"\n**Deliverables:** {len(context.get('deliverables', []))} files")
        parts.append(f"**Requirements:** {context.get('rf_count', 0)} RF")
        parts.append(f"**Invariants:** {context.get('inv_count', 0)} INV")
        parts.append(f"**Edge Cases:** {context.get('edge_count', 0)} EDGE")

    return "\n".join(parts)


# =============================================================================
# SURGICAL REWORK HELPERS
# =============================================================================


def create_task_level_entry(
    task_id: str,
    sprint_id: str,
    sequence: int,
    name: str,
    description: str,
    deliverables: Optional[List[str]] = None,
    depends_on: Optional[List[str]] = None,
    max_attempts: int = 3,
) -> TaskLevelEntry:
    """Create a new task-level entry for the surgical rework system.

    Args:
        task_id: Unique task identifier (e.g., "S00-T01").
        sprint_id: Sprint identifier (e.g., "S00").
        sequence: Execution sequence within sprint.
        name: Human-readable task name.
        description: Detailed task description.
        deliverables: List of files to generate.
        depends_on: List of task IDs this task depends on.
        max_attempts: Maximum repair attempts before escalation.

    Returns:
        TaskLevelEntry ready for insertion into surgical_rework.tasks.
    """
    now = datetime.now(timezone.utc).isoformat()

    return TaskLevelEntry(
        task_id=task_id,
        sprint_id=sprint_id,
        sequence=sequence,
        name=name,
        description=description,
        deliverables=deliverables or [],
        depends_on=depends_on or [],
        status=TaskLevelStatus.PENDING.value,
        attempts=0,
        max_attempts=max_attempts,
        code_generated=None,
        confidence_score=0.0,
        execution_time_ms=0,
        tokens_used=0,
        diagnoses=[],
        patches=[],
        repair_history=[],
        checkpoint_path=None,
        created_at=now,
        started_at=None,
        completed_at=None,
        last_validation_at=None,
        rework_phase=None,
    )


def update_task_level_status(
    state: PipelineState,
    task_id: str,
    new_status: TaskLevelStatus,
    code_generated: Optional[str] = None,
    confidence_score: Optional[float] = None,
    rework_phase: Optional[ReworkPhase] = None,
) -> PipelineState:
    """Update the status of a task in the surgical rework system.

    Args:
        state: Current pipeline state.
        task_id: ID of the task to update.
        new_status: New status for the task.
        code_generated: Optional code generated by the task.
        confidence_score: Optional confidence score.
        rework_phase: Optional rework phase (when status is REPAIRING).

    Returns:
        Updated state with the task status changed.

    Raises:
        KeyError: If task_id is not found in surgical_rework.tasks.
    """
    now = datetime.now(timezone.utc).isoformat()

    surgical_rework = state.get("surgical_rework", {})
    tasks = dict(surgical_rework.get("tasks", {}))

    if task_id not in tasks:
        raise KeyError(f"Task {task_id} not found in surgical_rework.tasks")

    task = dict(tasks[task_id])

    # Update status
    task["status"] = new_status.value

    # Update timestamps based on status
    if new_status == TaskLevelStatus.EXECUTING:
        task["started_at"] = now
        task["attempts"] = task.get("attempts", 0) + 1

    if new_status == TaskLevelStatus.VALIDATING:
        task["last_validation_at"] = now

    if new_status.is_terminal():
        task["completed_at"] = now

    # Update optional fields
    if code_generated is not None:
        task["code_generated"] = code_generated

    if confidence_score is not None:
        task["confidence_score"] = confidence_score

    if rework_phase is not None:
        task["rework_phase"] = rework_phase.value
    elif new_status != TaskLevelStatus.REPAIRING:
        task["rework_phase"] = None

    tasks[task_id] = task

    # Update counters
    pending = sum(1 for t in tasks.values() if t["status"] == TaskLevelStatus.PENDING.value)
    executing = sum(1 for t in tasks.values() if t["status"] == TaskLevelStatus.EXECUTING.value)
    passed = sum(1 for t in tasks.values() if t["status"] == TaskLevelStatus.PASSED.value)
    failed = sum(1 for t in tasks.values() if t["status"] == TaskLevelStatus.FAILED.value)
    escalated = sum(1 for t in tasks.values() if t["status"] == TaskLevelStatus.ESCALATED.value)

    updated_surgical_rework = SurgicalReworkState(
        **{
            **surgical_rework,
            "tasks": tasks,
            "tasks_pending": pending,
            "tasks_executing": executing,
            "tasks_passed": passed,
            "tasks_failed": failed,
            "tasks_escalated": escalated,
            "last_task_completed": task_id if new_status == TaskLevelStatus.PASSED else surgical_rework.get("last_task_completed"),
            "last_task_failed": task_id if new_status in (TaskLevelStatus.FAILED, TaskLevelStatus.ESCALATED) else surgical_rework.get("last_task_failed"),
        }
    )

    return {
        **state,
        "surgical_rework": updated_surgical_rework,
        "updated_at": now,
    }


def add_diagnosis_to_task(
    state: PipelineState,
    task_id: str,
    diagnosis: DiagnosisEntry,
) -> PipelineState:
    """Add a diagnosis entry to a task in the surgical rework system.

    Args:
        state: Current pipeline state.
        task_id: ID of the task.
        diagnosis: Diagnosis entry to add.

    Returns:
        Updated state with the diagnosis added.

    Raises:
        KeyError: If task_id is not found in surgical_rework.tasks.
    """
    now = datetime.now(timezone.utc).isoformat()

    surgical_rework = state.get("surgical_rework", {})
    tasks = dict(surgical_rework.get("tasks", {}))

    if task_id not in tasks:
        raise KeyError(f"Task {task_id} not found in surgical_rework.tasks")

    task = dict(tasks[task_id])
    diagnoses = list(task.get("diagnoses", []))
    diagnoses.append(diagnosis)
    task["diagnoses"] = diagnoses

    tasks[task_id] = task

    updated_surgical_rework = SurgicalReworkState(
        **{**surgical_rework, "tasks": tasks}
    )

    return {
        **state,
        "surgical_rework": updated_surgical_rework,
        "updated_at": now,
    }


def add_patch_to_task(
    state: PipelineState,
    task_id: str,
    patch: PatchEntry,
) -> PipelineState:
    """Add a patch entry to a task in the surgical rework system.

    Args:
        state: Current pipeline state.
        task_id: ID of the task.
        patch: Patch entry to add.

    Returns:
        Updated state with the patch added.

    Raises:
        KeyError: If task_id is not found in surgical_rework.tasks.
    """
    now = datetime.now(timezone.utc).isoformat()

    surgical_rework = state.get("surgical_rework", {})
    tasks = dict(surgical_rework.get("tasks", {}))

    if task_id not in tasks:
        raise KeyError(f"Task {task_id} not found in surgical_rework.tasks")

    task = dict(tasks[task_id])
    patches = list(task.get("patches", []))
    patches.append(patch)
    task["patches"] = patches

    tasks[task_id] = task

    # Update patch counter
    total_patches = surgical_rework.get("total_patches_applied", 0) + 1

    updated_surgical_rework = SurgicalReworkState(
        **{
            **surgical_rework,
            "tasks": tasks,
            "total_patches_applied": total_patches,
        }
    )

    return {
        **state,
        "surgical_rework": updated_surgical_rework,
        "updated_at": now,
    }


def set_parallel_groups(
    state: PipelineState,
    parallel_groups: List[List[str]],
) -> PipelineState:
    """Set the parallel execution groups in the surgical rework system.

    Args:
        state: Current pipeline state.
        parallel_groups: List of groups, each containing task IDs that can run in parallel.

    Returns:
        Updated state with parallel groups set.
    """
    now = datetime.now(timezone.utc).isoformat()

    surgical_rework = state.get("surgical_rework", {})

    updated_surgical_rework = SurgicalReworkState(
        **{
            **surgical_rework,
            "parallel_groups": parallel_groups,
            "current_group_index": 0,
            "started_at": now,
        }
    )

    return {
        **state,
        "surgical_rework": updated_surgical_rework,
        "updated_at": now,
    }


def advance_to_next_group(state: PipelineState) -> PipelineState:
    """Advance to the next parallel group in the surgical rework system.

    Args:
        state: Current pipeline state.

    Returns:
        Updated state with current_group_index incremented.
    """
    now = datetime.now(timezone.utc).isoformat()

    surgical_rework = state.get("surgical_rework", {})
    current_index = surgical_rework.get("current_group_index", 0)
    parallel_groups = surgical_rework.get("parallel_groups", [])

    new_index = current_index + 1

    # Check if we've completed all groups
    completed_at = now if new_index >= len(parallel_groups) else None

    updated_surgical_rework = SurgicalReworkState(
        **{
            **surgical_rework,
            "current_group_index": new_index,
            "completed_at": completed_at,
        }
    )

    return {
        **state,
        "surgical_rework": updated_surgical_rework,
        "updated_at": now,
    }


def get_current_group_tasks(state: PipelineState) -> List[str]:
    """Get the task IDs for the current parallel group.

    Args:
        state: Current pipeline state.

    Returns:
        List of task IDs in the current group, or empty list if done.
    """
    surgical_rework = state.get("surgical_rework", {})
    parallel_groups = surgical_rework.get("parallel_groups", [])
    current_index = surgical_rework.get("current_group_index", 0)

    if current_index >= len(parallel_groups):
        return []

    return parallel_groups[current_index]


def record_repair_attempt(
    state: PipelineState,
    task_id: str,
    diagnosis_id: str,
    patch_ids: List[str],
    success: bool,
    notes: Optional[str] = None,
) -> PipelineState:
    """Record a repair attempt in the task's repair history.

    Args:
        state: Current pipeline state.
        task_id: ID of the task.
        diagnosis_id: ID of the diagnosis that led to this repair.
        patch_ids: List of patch IDs applied.
        success: Whether the repair was successful.
        notes: Optional notes about the repair.

    Returns:
        Updated state with repair attempt recorded.

    Raises:
        KeyError: If task_id is not found in surgical_rework.tasks.
    """
    now = datetime.now(timezone.utc).isoformat()

    surgical_rework = state.get("surgical_rework", {})
    tasks = dict(surgical_rework.get("tasks", {}))

    if task_id not in tasks:
        raise KeyError(f"Task {task_id} not found in surgical_rework.tasks")

    task = dict(tasks[task_id])
    repair_history = list(task.get("repair_history", []))

    repair_entry = {
        "attempt": len(repair_history) + 1,
        "diagnosis_id": diagnosis_id,
        "patch_ids": patch_ids,
        "success": success,
        "notes": notes,
        "timestamp": now,
    }
    repair_history.append(repair_entry)
    task["repair_history"] = repair_history

    tasks[task_id] = task

    # Update total repairs counter
    total_repairs = surgical_rework.get("total_repairs", 0) + 1

    updated_surgical_rework = SurgicalReworkState(
        **{
            **surgical_rework,
            "tasks": tasks,
            "total_repairs": total_repairs,
        }
    )

    return {
        **state,
        "surgical_rework": updated_surgical_rework,
        "updated_at": now,
    }


def set_integration_validation_result(
    state: PipelineState,
    passed: bool,
    issues: Optional[List[Dict[str, Any]]] = None,
) -> PipelineState:
    """Set the integration validation result.

    Args:
        state: Current pipeline state.
        passed: Whether integration validation passed.
        issues: List of integration issues found.

    Returns:
        Updated state with integration validation result.
    """
    now = datetime.now(timezone.utc).isoformat()

    surgical_rework = state.get("surgical_rework", {})

    updated_surgical_rework = SurgicalReworkState(
        **{
            **surgical_rework,
            "integration_validation_passed": passed,
            "integration_issues": issues or [],
        }
    )

    return {
        **state,
        "surgical_rework": updated_surgical_rework,
        "updated_at": now,
    }
