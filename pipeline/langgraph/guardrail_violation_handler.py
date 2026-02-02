"""Guardrail Violation Handler for Pipeline V2.

This module implements the CORRECT guardrail philosophy:
- Guardrails BLOCK pipeline advancement when violations are detected
- The task is INVALIDATED entirely
- A detailed violation report is generated
- The task is sent back for REWORK with error awareness
- The agent learns from the error via Reflexion
- Learnings are persisted to A-MEM

Philosophy: "Guard rails on a highway don't stop the car - they redirect it safely"
But MORE IMPORTANTLY: "The car MUST go back and fix the issue, not continue damaged"

Architecture:
    Guardrail Violation Detected
              |
              v
    +-------------------+
    | Task Invalidation |  <- Marks task as INVALID
    +-------------------+
              |
              v
    +-------------------+
    | Violation Report  |  <- Full details of what went wrong
    +-------------------+
              |
              v
    +-------------------+
    | Reflexion Engine  |  <- Agent learns from error
    +-------------------+
              |
              v
    +-------------------+
    | A-MEM Persistence |  <- Learning is persisted
    +-------------------+
              |
              v
    +-------------------+
    | Rework Queue      |  <- Task sent back for rework
    +-------------------+
              |
              v
    Agent receives task with:
    - Original task context
    - Violation report
    - Error learnings
    - Retry instructions

Author: Pipeline Autonomo Team
Version: 1.0.0 (2026-01-16)
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict

logger = logging.getLogger(__name__)

# Bounded store limits to prevent unbounded memory growth
MAX_VIOLATION_REPORTS = 500
MAX_TASK_INVALIDATIONS = 500
MAX_REWORK_QUEUE_SIZE = 100


# =============================================================================
# ENUMS
# =============================================================================


class ViolationType(str, Enum):
    """Types of guardrail violations."""

    SECURITY = "security"           # Security violation (injection, jailbreak, etc.)
    INVARIANT = "invariant"         # Pipeline invariant violated
    TRUST = "trust"                 # Trust boundary violation
    CONTENT = "content"             # Content policy violation (PII, secrets, etc.)
    VALIDATION = "validation"       # Input/output validation failure
    GATE = "gate"                   # Gate check failure
    TIMEOUT = "timeout"             # Timeout exceeded
    RESOURCE = "resource"           # Resource limit exceeded
    CONSISTENCY = "consistency"     # State consistency violation


class ViolationSeverity(str, Enum):
    """Severity levels for violations."""

    CRITICAL = "critical"   # Must stop immediately, human review required
    HIGH = "high"           # Task invalidated, rework required
    MEDIUM = "medium"       # Task invalidated, can auto-retry with fixes
    LOW = "low"             # Warning, can continue with corrections
    INFO = "info"           # Informational, logged but no action


class TaskInvalidationReason(str, Enum):
    """Reasons for task invalidation."""

    GUARDRAIL_VIOLATION = "guardrail_violation"
    GATE_FAILURE = "gate_failure"
    TIMEOUT = "timeout"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    INVARIANT_BROKEN = "invariant_broken"
    SECURITY_BREACH = "security_breach"
    CONSISTENCY_ERROR = "consistency_error"
    EXTERNAL_FAILURE = "external_failure"


class ReworkPriority(str, Enum):
    """Priority for rework queue."""

    URGENT = "urgent"       # Process immediately
    HIGH = "high"           # Process next
    NORMAL = "normal"       # Process in order
    LOW = "low"             # Process when available


# =============================================================================
# RESULT TYPES
# =============================================================================


class ViolationDetails(TypedDict):
    """Detailed information about a single violation."""

    violation_id: str
    violation_type: str          # ViolationType value
    severity: str                # ViolationSeverity value
    message: str                 # Human-readable description
    guardrail_name: str          # Which guardrail was violated
    evidence: Dict[str, Any]     # Evidence of the violation
    context: Dict[str, Any]      # Context when violation occurred
    timestamp: str               # ISO timestamp

    # For agent awareness
    what_went_wrong: str         # Simple explanation
    why_it_matters: str          # Why this is a problem
    how_to_fix: str              # Suggested fix approach


class ViolationReport(TypedDict):
    """Complete violation report for a task."""

    report_id: str
    task_id: str
    run_id: str
    sprint_id: str
    agent_id: Optional[str]
    node_name: str

    # Violations
    violations: List[ViolationDetails]
    total_violations: int
    highest_severity: str

    # Task context
    original_input: Dict[str, Any]
    state_snapshot: Dict[str, Any]

    # Learning context (from Reflexion)
    error_analysis: Optional[Dict[str, Any]]
    proposed_fixes: List[str]
    prior_learnings: List[Dict[str, Any]]  # Relevant past learnings

    # Timestamps
    created_at: str

    # Instructions for rework
    rework_instructions: str


class TaskInvalidation(TypedDict):
    """Task invalidation record."""

    invalidation_id: str
    task_id: str
    run_id: str
    sprint_id: str

    # Reason
    reason: str                  # TaskInvalidationReason value
    violation_report_id: str

    # Original task info
    original_node: str
    original_phase: str
    original_attempt: int

    # Rework info
    rework_required: bool
    rework_priority: str         # ReworkPriority value
    rework_instructions: str
    max_rework_attempts: int
    current_rework_attempt: int

    # Timestamps
    invalidated_at: str

    # Learning reference
    learning_id: Optional[str]   # Reference to A-MEM learning entry


class ReworkTask(TypedDict):
    """Task queued for rework."""

    rework_id: str
    original_task_id: str
    run_id: str
    sprint_id: str

    # Priority
    priority: str                # ReworkPriority value

    # Context for agent
    violation_report: ViolationReport
    original_state: Dict[str, Any]
    error_learnings: List[Dict[str, Any]]

    # Instructions
    rework_instructions: str
    retry_hints: List[str]

    # Attempt tracking
    attempt_number: int
    max_attempts: int

    # Queue info
    queued_at: str
    started_at: Optional[str]
    completed_at: Optional[str]

    # Status
    status: str                  # "queued", "in_progress", "completed", "failed"


# =============================================================================
# VIOLATION BUILDER
# =============================================================================


@dataclass
class ViolationBuilder:
    """Builder for creating violation details."""

    violation_type: ViolationType
    severity: ViolationSeverity
    message: str
    guardrail_name: str

    evidence: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)

    what_went_wrong: str = ""
    why_it_matters: str = ""
    how_to_fix: str = ""

    def with_evidence(self, key: str, value: Any) -> "ViolationBuilder":
        """Add evidence to the violation."""
        self.evidence[key] = value
        return self

    def with_context(self, key: str, value: Any) -> "ViolationBuilder":
        """Add context to the violation."""
        self.context[key] = value
        return self

    def with_explanation(
        self,
        what_went_wrong: str,
        why_it_matters: str,
        how_to_fix: str,
    ) -> "ViolationBuilder":
        """Add human-readable explanation for the agent."""
        self.what_went_wrong = what_went_wrong
        self.why_it_matters = why_it_matters
        self.how_to_fix = how_to_fix
        return self

    def build(self) -> ViolationDetails:
        """Build the violation details."""
        return ViolationDetails(
            violation_id=f"viol_{uuid.uuid4().hex[:8]}",
            violation_type=self.violation_type.value,
            severity=self.severity.value,
            message=self.message,
            guardrail_name=self.guardrail_name,
            evidence=self.evidence,
            context=self.context,
            timestamp=datetime.now(timezone.utc).isoformat(),
            what_went_wrong=self.what_went_wrong or self.message,
            why_it_matters=self.why_it_matters or f"Guardrail '{self.guardrail_name}' protects pipeline integrity",
            how_to_fix=self.how_to_fix or "Review the violation details and fix the underlying issue",
        )


# =============================================================================
# GUARDRAIL VIOLATION HANDLER
# =============================================================================


class GuardrailViolationHandler:
    """Handles guardrail violations with task invalidation and rework.

    This is the CORRECT implementation that:
    1. BLOCKS pipeline advancement on violations
    2. INVALIDATES the entire task
    3. GENERATES detailed violation reports
    4. QUEUES task for REWORK with error awareness
    5. INTEGRATES with Reflexion for learning
    6. PERSISTS learnings to A-MEM
    """

    def __init__(
        self,
        max_rework_attempts: int = 3,
        learning_enabled: bool = True,
    ):
        """Initialize violation handler.

        Args:
            max_rework_attempts: Maximum rework attempts per task
            learning_enabled: Whether to enable Reflexion learning
        """
        self.max_rework_attempts = max_rework_attempts
        self.learning_enabled = learning_enabled

        # Thread safety lock for concurrent access
        self._lock = asyncio.Lock()

        # In-memory stores (would be Redis/DB in production)
        # These are bounded to prevent unbounded memory growth
        self._violation_reports: Dict[str, ViolationReport] = {}
        self._task_invalidations: Dict[str, TaskInvalidation] = {}
        self._rework_queue: List[ReworkTask] = []

        # Statistics
        self._stats = {
            "violations_detected": 0,
            "tasks_invalidated": 0,
            "reworks_queued": 0,
            "reworks_completed": 0,
            "reworks_failed": 0,
            "stores_pruned": 0,
        }

        # Reflexion engine (lazy load)
        self._reflexion_engine = None

    def _prune_stores_if_needed(self) -> None:
        """Prune stores if they exceed size limits.

        Called internally - caller must hold the lock.
        """
        # Prune violation reports (keep most recent)
        if len(self._violation_reports) > MAX_VIOLATION_REPORTS:
            sorted_reports = sorted(
                self._violation_reports.items(),
                key=lambda x: x[1]["created_at"],
                reverse=True,
            )
            self._violation_reports = dict(sorted_reports[:MAX_VIOLATION_REPORTS // 2])
            self._stats["stores_pruned"] += 1
            logger.info(f"Pruned violation reports to {len(self._violation_reports)}")

        # Prune task invalidations (keep most recent)
        if len(self._task_invalidations) > MAX_TASK_INVALIDATIONS:
            sorted_invalidations = sorted(
                self._task_invalidations.items(),
                key=lambda x: x[1]["invalidated_at"],
                reverse=True,
            )
            self._task_invalidations = dict(sorted_invalidations[:MAX_TASK_INVALIDATIONS // 2])
            self._stats["stores_pruned"] += 1
            logger.info(f"Pruned task invalidations to {len(self._task_invalidations)}")

        # Prune rework queue (remove completed/failed tasks)
        if len(self._rework_queue) > MAX_REWORK_QUEUE_SIZE:
            # Keep only queued and in_progress tasks
            active_tasks = [t for t in self._rework_queue if t["status"] in ("queued", "in_progress")]
            completed_tasks = [t for t in self._rework_queue if t["status"] not in ("queued", "in_progress")]

            # Keep some completed for history
            completed_tasks.sort(key=lambda x: x["completed_at"] or x["queued_at"], reverse=True)
            self._rework_queue = active_tasks + completed_tasks[:MAX_REWORK_QUEUE_SIZE // 4]
            self._stats["stores_pruned"] += 1
            logger.info(f"Pruned rework queue to {len(self._rework_queue)}")

    def _get_reflexion_engine(self):
        """Lazy load Reflexion engine."""
        if self._reflexion_engine is None:
            try:
                from pipeline.langgraph.reflexion.engine import get_reflexion_engine
                self._reflexion_engine = get_reflexion_engine()
            except ImportError:
                logger.warning("Reflexion engine not available")
        return self._reflexion_engine

    async def handle_violation(
        self,
        violations: List[ViolationDetails],
        task_id: str,
        run_id: str,
        sprint_id: str,
        node_name: str,
        state: Dict[str, Any],
        agent_id: Optional[str] = None,
    ) -> tuple[TaskInvalidation, ReworkTask]:
        """Handle guardrail violations.

        This is the main entry point when a guardrail detects a violation.
        Thread-safe: uses asyncio.Lock for concurrent access protection.

        Args:
            violations: List of violations detected
            task_id: ID of the task that violated
            run_id: Current run ID
            sprint_id: Current sprint ID
            node_name: Node where violation occurred
            state: Current pipeline state
            agent_id: ID of the agent that caused violation

        Returns:
            Tuple of (TaskInvalidation, ReworkTask) for pipeline handling
        """
        # 1. Determine highest severity (no lock needed - pure computation)
        severity_order = [
            ViolationSeverity.CRITICAL,
            ViolationSeverity.HIGH,
            ViolationSeverity.MEDIUM,
            ViolationSeverity.LOW,
            ViolationSeverity.INFO,
        ]
        highest_severity = ViolationSeverity.INFO
        for v in violations:
            v_severity = ViolationSeverity(v["severity"])
            if severity_order.index(v_severity) < severity_order.index(highest_severity):
                highest_severity = v_severity

        # 2. Run Reflexion analysis if enabled
        error_analysis = None
        proposed_fixes = []
        prior_learnings = []

        if self.learning_enabled:
            reflexion = self._get_reflexion_engine()
            if reflexion:
                try:
                    # Get error analysis
                    failure_context = {
                        "violations": [v["message"] for v in violations],
                        "node": node_name,
                        "task_id": task_id,
                        "agent_id": agent_id,
                        "state_keys": list(state.keys()),
                    }

                    result = await reflexion.reflect(failure_context)
                    error_analysis = {
                        "reflection_id": result["reflection_id"],
                        "final_analysis": result["final_analysis"],
                        "quality_level": result["quality_level"],
                    }

                    proposed_fixes = [
                        a["action"] for a in result["action_plan"]["actions"]
                    ]

                    # Get prior learnings
                    learnings = await reflexion.feedback_learner.get_relevant_learnings(
                        failure_context, min_confidence=0.5
                    )
                    prior_learnings = [
                        {"action": l["pattern"]["action"], "outcome": l["outcome"]}
                        for l in learnings[:5]
                    ]

                    # Learn from this violation
                    await reflexion.learn(
                        action=f"task_{node_name}",
                        outcome="failure",
                        context=failure_context,
                    )

                except Exception as e:
                    logger.warning(f"Reflexion analysis failed: {e}")

            # Persist learning to A-MEM
            try:
                from pipeline.langgraph.a_mem import (
                    save_violation_learning,
                    get_relevant_violation_learnings,
                )

                # Save this violation as a learning
                await save_violation_learning(
                    violation_type=violations[0]["violation_type"] if violations else "unknown",
                    message="; ".join([v["message"] for v in violations[:3]]),
                    context={
                        "node": node_name,
                        "task_id": task_id,
                        "agent_id": agent_id,
                        "violations_count": len(violations),
                        "highest_severity": highest_severity.value,
                    },
                    action_taken="task_invalidated_for_rework",
                    outcome="queued_for_rework",
                    run_id=run_id,
                    sprint_id=sprint_id,
                )

                # Get relevant prior learnings from A-MEM
                amem_learnings = await get_relevant_violation_learnings(
                    context={
                        "node": node_name,
                        "violation_type": violations[0]["violation_type"] if violations else None,
                    },
                    limit=3,
                )
                if amem_learnings:
                    prior_learnings.extend([
                        {"action": l.action_taken, "outcome": l.outcome, "lesson": l.lesson}
                        for l in amem_learnings
                    ])

            except ImportError:
                logger.debug("A-MEM not available, skipping learning persistence")
            except Exception as e:
                logger.warning(f"A-MEM persistence failed: {e}")

        # 3. Generate rework instructions
        rework_instructions = self._generate_rework_instructions(
            violations=violations,
            proposed_fixes=proposed_fixes,
            prior_learnings=prior_learnings,
        )

        # 4. Create objects (pure computation - no lock needed)
        report = ViolationReport(
            report_id=f"rpt_{uuid.uuid4().hex[:8]}",
            task_id=task_id,
            run_id=run_id,
            sprint_id=sprint_id,
            agent_id=agent_id,
            node_name=node_name,
            violations=violations,
            total_violations=len(violations),
            highest_severity=highest_severity.value,
            original_input=state.get("input_data", {}),
            state_snapshot=self._create_state_snapshot(state),
            error_analysis=error_analysis,
            proposed_fixes=proposed_fixes,
            prior_learnings=prior_learnings,
            created_at=datetime.now(timezone.utc).isoformat(),
            rework_instructions=rework_instructions,
        )

        # 5. Create task invalidation
        current_attempt = state.get("attempt", 1)
        invalidation = TaskInvalidation(
            invalidation_id=f"inv_{uuid.uuid4().hex[:8]}",
            task_id=task_id,
            run_id=run_id,
            sprint_id=sprint_id,
            reason=TaskInvalidationReason.GUARDRAIL_VIOLATION.value,
            violation_report_id=report["report_id"],
            original_node=node_name,
            original_phase=state.get("phase", "unknown"),
            original_attempt=current_attempt,
            rework_required=highest_severity in [
                ViolationSeverity.CRITICAL,
                ViolationSeverity.HIGH,
                ViolationSeverity.MEDIUM,
            ],
            rework_priority=self._determine_rework_priority(highest_severity),
            rework_instructions=rework_instructions,
            max_rework_attempts=self.max_rework_attempts,
            current_rework_attempt=current_attempt,
            invalidated_at=datetime.now(timezone.utc).isoformat(),
            learning_id=error_analysis.get("reflection_id") if error_analysis else None,
        )

        # 6. Create rework task
        rework = ReworkTask(
            rework_id=f"rwk_{uuid.uuid4().hex[:8]}",
            original_task_id=task_id,
            run_id=run_id,
            sprint_id=sprint_id,
            priority=invalidation["rework_priority"],
            violation_report=report,
            original_state=state,
            error_learnings=prior_learnings,
            rework_instructions=rework_instructions,
            retry_hints=proposed_fixes,
            attempt_number=current_attempt + 1,
            max_attempts=self.max_rework_attempts,
            queued_at=datetime.now(timezone.utc).isoformat(),
            started_at=None,
            completed_at=None,
            status="queued",
        )

        # 7. Store in shared state (lock required for thread safety)
        async with self._lock:
            self._stats["violations_detected"] += len(violations)
            self._violation_reports[report["report_id"]] = report
            self._task_invalidations[invalidation["invalidation_id"]] = invalidation
            self._stats["tasks_invalidated"] += 1
            self._rework_queue.append(rework)
            self._stats["reworks_queued"] += 1
            # Prune stores if they exceed size limits
            self._prune_stores_if_needed()

        logger.warning(
            f"Task {task_id} INVALIDATED due to {len(violations)} guardrail violations "
            f"(highest severity: {highest_severity.value}). "
            f"Rework queued with priority {rework['priority']}."
        )

        return invalidation, rework

    def _generate_rework_instructions(
        self,
        violations: List[ViolationDetails],
        proposed_fixes: List[str],
        prior_learnings: List[Dict[str, Any]],
    ) -> str:
        """Generate detailed rework instructions for the agent."""

        lines = [
            "=== REWORK INSTRUCTIONS ===",
            "",
            "Your previous task was INVALIDATED due to guardrail violations.",
            "You MUST fix the following issues before proceeding:",
            "",
        ]

        # Add violation details
        for i, v in enumerate(violations, 1):
            lines.extend([
                f"--- Violation {i}: {v['guardrail_name']} ---",
                f"What went wrong: {v['what_went_wrong']}",
                f"Why it matters: {v['why_it_matters']}",
                f"How to fix: {v['how_to_fix']}",
                "",
            ])

        # Add proposed fixes
        if proposed_fixes:
            lines.extend([
                "--- Recommended Actions ---",
            ])
            for fix in proposed_fixes[:5]:
                lines.append(f"- {fix}")
            lines.append("")

        # Add prior learnings
        if prior_learnings:
            lines.extend([
                "--- Learnings from Past Similar Errors ---",
            ])
            for learning in prior_learnings[:3]:
                lines.append(
                    f"- Action: {learning['action']} -> Outcome: {learning['outcome']}"
                )
            lines.append("")

        # Add general guidance
        lines.extend([
            "--- General Guidance ---",
            "1. Review ALL violations before attempting the task again",
            "2. Address the ROOT CAUSE, not just the symptoms",
            "3. Test your changes against the guardrails before submitting",
            "4. If stuck, escalate to your supervisor with details",
            "",
            "=========================",
        ])

        return "\n".join(lines)

    def _create_state_snapshot(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Create a minimal state snapshot for the report."""
        # Only include relevant fields, not the entire state
        snapshot_keys = [
            "run_id", "sprint_id", "phase", "status", "attempt",
            "gate_status", "gates_passed", "gates_failed",
        ]
        return {k: state.get(k) for k in snapshot_keys if k in state}

    def _determine_rework_priority(
        self,
        severity: ViolationSeverity,
    ) -> str:
        """Determine rework priority based on severity."""
        priority_map = {
            ViolationSeverity.CRITICAL: ReworkPriority.URGENT.value,
            ViolationSeverity.HIGH: ReworkPriority.HIGH.value,
            ViolationSeverity.MEDIUM: ReworkPriority.NORMAL.value,
            ViolationSeverity.LOW: ReworkPriority.LOW.value,
            ViolationSeverity.INFO: ReworkPriority.LOW.value,
        }
        return priority_map.get(severity, ReworkPriority.NORMAL.value)

    async def get_next_rework_task(self) -> Optional[ReworkTask]:
        """Get the next task from the rework queue.

        Returns tasks in priority order (URGENT > HIGH > NORMAL > LOW).
        Thread-safe: uses asyncio.Lock for concurrent access protection.
        """
        async with self._lock:
            if not self._rework_queue:
                return None

            # Sort by priority
            priority_order = {
                ReworkPriority.URGENT.value: 0,
                ReworkPriority.HIGH.value: 1,
                ReworkPriority.NORMAL.value: 2,
                ReworkPriority.LOW.value: 3,
            }

            self._rework_queue.sort(
                key=lambda t: (priority_order.get(t["priority"], 2), t["queued_at"])
            )

            # Get first queued task
            for task in self._rework_queue:
                if task["status"] == "queued":
                    task["status"] = "in_progress"
                    task["started_at"] = datetime.now(timezone.utc).isoformat()
                    return task

            return None

    async def mark_rework_completed(
        self,
        rework_id: str,
        success: bool,
    ) -> None:
        """Mark a rework task as completed.

        Thread-safe: uses asyncio.Lock for concurrent access protection.
        """
        async with self._lock:
            for task in self._rework_queue:
                if task["rework_id"] == rework_id:
                    task["status"] = "completed" if success else "failed"
                    task["completed_at"] = datetime.now(timezone.utc).isoformat()

                    if success:
                        self._stats["reworks_completed"] += 1
                    else:
                        self._stats["reworks_failed"] += 1

                    return

    def get_violation_report(self, report_id: str) -> Optional[ViolationReport]:
        """Get a violation report by ID."""
        return self._violation_reports.get(report_id)

    def get_stats(self) -> Dict[str, int]:
        """Get handler statistics."""
        return dict(self._stats)

    def get_pending_reworks(self) -> List[ReworkTask]:
        """Get all pending rework tasks."""
        return [t for t in self._rework_queue if t["status"] == "queued"]


# =============================================================================
# SINGLETON
# =============================================================================

_violation_handler: Optional[GuardrailViolationHandler] = None


def get_violation_handler(
    max_rework_attempts: int = 3,
    learning_enabled: bool = True,
) -> GuardrailViolationHandler:
    """Get singleton violation handler instance.

    Args:
        max_rework_attempts: Maximum rework attempts
        learning_enabled: Whether to enable learning

    Returns:
        GuardrailViolationHandler instance
    """
    global _violation_handler
    if _violation_handler is None:
        _violation_handler = GuardrailViolationHandler(
            max_rework_attempts=max_rework_attempts,
            learning_enabled=learning_enabled,
        )
    return _violation_handler


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_security_violation(
    message: str,
    guardrail_name: str,
    evidence: Optional[Dict[str, Any]] = None,
) -> ViolationDetails:
    """Create a security violation."""
    builder = ViolationBuilder(
        violation_type=ViolationType.SECURITY,
        severity=ViolationSeverity.HIGH,
        message=message,
        guardrail_name=guardrail_name,
    )

    if evidence:
        for k, v in evidence.items():
            builder.with_evidence(k, v)

    builder.with_explanation(
        what_went_wrong=f"Security violation detected: {message}",
        why_it_matters="Security violations can compromise pipeline integrity and data safety",
        how_to_fix="Review and sanitize input/output, ensure no malicious content passes through",
    )

    return builder.build()


def create_invariant_violation(
    invariant_id: str,
    message: str,
    evidence: Optional[Dict[str, Any]] = None,
) -> ViolationDetails:
    """Create an invariant violation."""
    builder = ViolationBuilder(
        violation_type=ViolationType.INVARIANT,
        severity=ViolationSeverity.HIGH,
        message=message,
        guardrail_name=f"Invariant_{invariant_id}",
    )

    if evidence:
        for k, v in evidence.items():
            builder.with_evidence(k, v)

    builder.with_explanation(
        what_went_wrong=f"Invariant {invariant_id} was violated: {message}",
        why_it_matters="Invariants ensure pipeline consistency and correctness",
        how_to_fix="Review the invariant requirements and ensure state meets all conditions",
    )

    return builder.build()


def create_validation_violation(
    field_name: str,
    message: str,
    expected: Any,
    actual: Any,
) -> ViolationDetails:
    """Create a validation violation."""
    builder = ViolationBuilder(
        violation_type=ViolationType.VALIDATION,
        severity=ViolationSeverity.MEDIUM,
        message=message,
        guardrail_name=f"Validator_{field_name}",
    )

    builder.with_evidence("expected", str(expected))
    builder.with_evidence("actual", str(actual))
    builder.with_evidence("field", field_name)

    builder.with_explanation(
        what_went_wrong=f"Validation failed for '{field_name}': expected {expected}, got {actual}",
        why_it_matters="Validation ensures data integrity throughout the pipeline",
        how_to_fix=f"Ensure '{field_name}' meets the expected format/value: {expected}",
    )

    return builder.build()


def create_gate_violation(
    gate_id: str,
    message: str,
    checks_failed: List[str],
) -> ViolationDetails:
    """Create a gate violation."""
    builder = ViolationBuilder(
        violation_type=ViolationType.GATE,
        severity=ViolationSeverity.HIGH,
        message=message,
        guardrail_name=f"Gate_{gate_id}",
    )

    builder.with_evidence("gate_id", gate_id)
    builder.with_evidence("checks_failed", checks_failed)

    builder.with_explanation(
        what_went_wrong=f"Gate {gate_id} failed: {', '.join(checks_failed)}",
        why_it_matters="Gates ensure quality and completeness before pipeline progression",
        how_to_fix=f"Address all failed checks: {', '.join(checks_failed)}",
    )

    return builder.build()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Classes
    "GuardrailViolationHandler",
    "ViolationBuilder",
    # Enums
    "ViolationType",
    "ViolationSeverity",
    "TaskInvalidationReason",
    "ReworkPriority",
    # Types
    "ViolationDetails",
    "ViolationReport",
    "TaskInvalidation",
    "ReworkTask",
    # Functions
    "get_violation_handler",
    "create_security_violation",
    "create_invariant_violation",
    "create_validation_violation",
    "create_gate_violation",
]
