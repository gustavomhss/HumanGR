"""Invariant Enforcement for LangGraph Pipeline.

This module defines and enforces the 11 core invariants identified in the
migration plan. Invariants are non-negotiable rules that must hold at all times.

Addresses: RED AGENT CRIT-01, CRIT-04 - Key namespacing and idempotency enforcement

Invariants:
    I1: Namespacing - Any persistent key contains run_id + sprint_id
    I2: Idempotency - Each node has deterministic idempotency_key; re-execution does not duplicate
    I3: Phase Order - INIT -> SPEC -> PLAN -> EXEC -> QA -> VOTE -> DONE
    I4: Gates Before Signoff - Gates must be executed and passed before signoffs
    I5: Executive Verification - Execs verify subordinate work before approving
    I6: Truthfulness - Signoff claims must match actual artifacts
    I7: Audit Trail - Every decision has evidence bundle with manifest
    I8: Event Schema - event_log uses schema_version, event_id, type
    I9: SAFE_HALT Priority - SAFE_HALT takes precedence over all other operations
    I10: Redis Canonical - Redis is canonical; file is mirror for resilience
    I11: Runaway Protection - Worker limits and cost controls prevent infinite loops

Author: Pipeline Autonomo Team
Version: 1.0.0 (2026-01-16)
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar

logger = logging.getLogger(__name__)

# Type variable for decorator
F = TypeVar('F', bound=Callable[..., 'InvariantCheckResult'])


# =============================================================================
# INVARIANT DEFINITIONS
# =============================================================================


class InvariantCode(str, Enum):
    """Invariant codes for identification and reporting."""

    I1_NAMESPACING = "I1"
    I2_IDEMPOTENCY = "I2"
    I3_PHASE_ORDER = "I3"
    I4_GATES_BEFORE_SIGNOFF = "I4"
    I5_EXECUTIVE_VERIFICATION = "I5"
    I6_TRUTHFULNESS = "I6"
    I7_AUDIT_TRAIL = "I7"
    I8_EVENT_SCHEMA = "I8"
    I9_SAFE_HALT_PRIORITY = "I9"
    I10_REDIS_CANONICAL = "I10"
    I11_RUNAWAY_PROTECTION = "I11"


@dataclass
class InvariantViolation:
    """Record of an invariant violation."""

    code: InvariantCode
    message: str
    key: Optional[str] = None
    expected: Optional[str] = None
    actual: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __str__(self) -> str:
        parts = [f"[{self.code.value}] {self.message}"]
        if self.key:
            parts.append(f"key={self.key}")
        if self.expected:
            parts.append(f"expected={self.expected}")
        if self.actual:
            parts.append(f"actual={self.actual}")
        return " | ".join(parts)


@dataclass
class InvariantCheckResult:
    """Result of invariant check."""

    passed: bool
    violations: List[InvariantViolation] = field(default_factory=list)
    checked_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def add_violation(self, violation: InvariantViolation) -> None:
        """Add a violation and mark as failed."""
        self.violations.append(violation)
        self.passed = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "violations": [str(v) for v in self.violations],
            "checked_at": self.checked_at,
        }


# =============================================================================
# INVARIANT VIOLATION ERROR - FOR STRICT ENFORCEMENT
# =============================================================================


class InvariantViolationError(Exception):
    """Raised when an invariant is violated and enforcement is strict.

    This exception should be raised instead of just returning a failed result
    when we need to BLOCK execution rather than just WARN.

    Usage:
        result = checker.check_phase_transition(...)
        if not result.passed:
            raise InvariantViolationError(result)
    """

    def __init__(self, result: InvariantCheckResult):
        """Initialize with the check result.

        Args:
            result: The failed InvariantCheckResult.
        """
        self.result = result
        self.violations = result.violations

        # Build message from violations
        violations_str = "; ".join([str(v) for v in result.violations])
        invariant_codes = set(v.code.value for v in result.violations)
        codes_str = ", ".join(sorted(invariant_codes))

        super().__init__(
            f"Invariant violation(s) [{codes_str}]: {violations_str}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "error": "InvariantViolationError",
            "message": str(self),
            "violations": [str(v) for v in self.violations],
            "result": self.result.to_dict(),
        }


def enforce_invariant(check_func: F) -> F:
    """Decorator that raises InvariantViolationError if check fails.

    This decorator transforms an advisory check into a blocking check.
    Use this when violations MUST stop execution.

    Usage:
        @enforce_invariant
        def check_something(...) -> InvariantCheckResult:
            ...

        # Now this will raise InvariantViolationError if check fails
        check_something(...)  # raises if not passed

    Note:
        The decorated function will still return InvariantCheckResult if passed.
        It only raises if the check fails.
    """
    @functools.wraps(check_func)
    def wrapper(*args, **kwargs) -> InvariantCheckResult:
        result = check_func(*args, **kwargs)
        if not result.passed:
            logger.error(
                f"INVARIANT VIOLATION (blocking): {[str(v) for v in result.violations]}"
            )
            raise InvariantViolationError(result)
        return result
    return wrapper  # type: ignore


def enforce_or_warn(
    result: InvariantCheckResult,
    strict: bool = True,
    context: Optional[str] = None,
) -> InvariantCheckResult:
    """Enforce or warn based on strict mode.

    This is a helper function to conditionally enforce invariants.

    Args:
        result: The check result.
        strict: If True, raises on violation. If False, just logs warning.
        context: Optional context for logging.

    Returns:
        The result (if passed or not strict).

    Raises:
        InvariantViolationError: If not passed and strict mode.
    """
    if not result.passed:
        ctx = f" [{context}]" if context else ""
        violations_str = "; ".join([str(v) for v in result.violations])

        if strict:
            logger.error(f"INVARIANT VIOLATION{ctx} (BLOCKING): {violations_str}")
            raise InvariantViolationError(result)
        else:
            logger.warning(f"INVARIANT VIOLATION{ctx} (warning only): {violations_str}")

    return result


# =============================================================================
# I1: NAMESPACING INVARIANT
# =============================================================================


class NamespacingEnforcer:
    """Enforces I1: Any persistent key contains run_id + sprint_id.

    Addresses CRIT-01: Cross-run contamination prevention.

    Key patterns that MUST include run_id:
    - milestone:{run_id}:{sprint_id}:{agent_id}:{milestone_id}
    - handoff:{run_id}:{sprint_id}:{from}:{to}
    - signoff:{run_id}:{sprint_id}:{agent_id}
    - approval:{run_id}:{sprint_id}:{agent_id}
    - checkpoint:{run_id}:{sprint_id}:{node}:{attempt}
    - event:{run_id}:{sprint_id}:{type}:{attempt}
    """

    # Key patterns that must be namespaced
    NAMESPACED_PREFIXES = [
        "milestone",
        "handoff",
        "signoff",
        "approval",
        "checkpoint",
        "event",
        "gate",
        "evidence",
        "artifact",
    ]

    @classmethod
    def validate_key(
        cls,
        key: str,
        run_id: str,
        sprint_id: str,
    ) -> InvariantCheckResult:
        """Validate that a key is properly namespaced.

        Args:
            key: The key to validate.
            run_id: Expected run_id.
            sprint_id: Expected sprint_id.

        Returns:
            InvariantCheckResult with any violations.
        """
        result = InvariantCheckResult(passed=True)

        # Check if key type requires namespacing
        key_type = key.split(":")[0] if ":" in key else key
        if key_type not in cls.NAMESPACED_PREFIXES:
            return result  # Key type doesn't require namespacing

        # Check for run_id
        if run_id not in key:
            result.add_violation(
                InvariantViolation(
                    code=InvariantCode.I1_NAMESPACING,
                    message="Key missing run_id - cross-run contamination risk",
                    key=key,
                    expected=f"Key should contain '{run_id}'",
                )
            )

        # Check for sprint_id
        if sprint_id not in key:
            result.add_violation(
                InvariantViolation(
                    code=InvariantCode.I1_NAMESPACING,
                    message="Key missing sprint_id - cross-sprint contamination risk",
                    key=key,
                    expected=f"Key should contain '{sprint_id}'",
                )
            )

        return result

    @classmethod
    def format_key(
        cls,
        key_type: str,
        run_id: str,
        sprint_id: str,
        *parts: str,
    ) -> str:
        """Format a properly namespaced key.

        Args:
            key_type: Type prefix (e.g., "milestone", "signoff").
            run_id: Run identifier.
            sprint_id: Sprint identifier.
            *parts: Additional key parts.

        Returns:
            Properly formatted key.
        """
        all_parts = [key_type, run_id, sprint_id] + list(parts)
        return ":".join(all_parts)


# =============================================================================
# I2: IDEMPOTENCY INVARIANT
# =============================================================================


class IdempotencyEnforcer:
    """Enforces I2: Each node has deterministic idempotency_key.

    Addresses CRIT-04: Prevents duplicate side effects on retry.

    Idempotency key format:
    {node}:{run_id}:{sprint_id}:attempt:{attempt}

    Usage:
        enforcer = IdempotencyEnforcer()
        if enforcer.should_execute(key):
            # Do work
            enforcer.mark_executed(key)
    """

    def __init__(self):
        """Initialize the enforcer."""
        self._executed_keys: Set[str] = set()

    def generate_key(
        self,
        node: str,
        run_id: str,
        sprint_id: str,
        attempt: int,
    ) -> str:
        """Generate deterministic idempotency key.

        Args:
            node: Node name (e.g., "INIT", "EXEC", "GATE").
            run_id: Run identifier.
            sprint_id: Sprint identifier.
            attempt: Attempt number.

        Returns:
            Deterministic idempotency key.
        """
        return f"{node}:{run_id}:{sprint_id}:attempt:{attempt}"

    def should_execute(self, key: str) -> bool:
        """Check if operation with this key should execute.

        Args:
            key: Idempotency key.

        Returns:
            True if should execute, False if already executed.
        """
        return key not in self._executed_keys

    def mark_executed(self, key: str) -> None:
        """Mark operation as executed.

        Args:
            key: Idempotency key.
        """
        self._executed_keys.add(key)
        logger.debug(f"Marked idempotency key as executed: {key}")

    def check_and_execute(
        self,
        key: str,
        operation_name: str = "operation",
    ) -> Tuple[bool, Optional[InvariantViolation]]:
        """Check if should execute and record decision.

        Args:
            key: Idempotency key.
            operation_name: Name for logging.

        Returns:
            Tuple of (should_execute, violation_if_duplicate).
        """
        if self.should_execute(key):
            self.mark_executed(key)
            return True, None

        violation = InvariantViolation(
            code=InvariantCode.I2_IDEMPOTENCY,
            message=f"Duplicate {operation_name} prevented by idempotency",
            key=key,
        )
        logger.info(f"I2: Prevented duplicate execution of {key}")
        return False, violation

    def clear(self) -> None:
        """Clear all recorded keys (for testing)."""
        self._executed_keys.clear()


# =============================================================================
# I3: PHASE ORDER INVARIANT
# =============================================================================


class PhaseOrderEnforcer:
    """Enforces I3: INIT -> SPEC -> PLAN -> EXEC -> QA -> VOTE -> DONE.

    Phases must follow strict order. Retries can go back but must
    increment attempt counter.
    """

    VALID_ORDER = ["INIT", "SPEC", "PLAN", "EXEC", "QA", "VOTE", "DONE", "HALT"]

    @classmethod
    def validate_transition(
        cls,
        from_phase: str,
        to_phase: str,
        is_retry: bool = False,
    ) -> InvariantCheckResult:
        """Validate phase transition.

        Args:
            from_phase: Current phase.
            to_phase: Target phase.
            is_retry: Whether this is a retry (allows backward transitions).

        Returns:
            InvariantCheckResult with any violations.
        """
        result = InvariantCheckResult(passed=True)

        # HALT can be reached from any phase
        if to_phase == "HALT":
            return result

        # Validate phases exist
        if from_phase not in cls.VALID_ORDER:
            result.add_violation(
                InvariantViolation(
                    code=InvariantCode.I3_PHASE_ORDER,
                    message=f"Invalid from_phase: {from_phase}",
                    expected=f"One of {cls.VALID_ORDER}",
                    actual=from_phase,
                )
            )
            return result

        if to_phase not in cls.VALID_ORDER:
            result.add_violation(
                InvariantViolation(
                    code=InvariantCode.I3_PHASE_ORDER,
                    message=f"Invalid to_phase: {to_phase}",
                    expected=f"One of {cls.VALID_ORDER}",
                    actual=to_phase,
                )
            )
            return result

        from_idx = cls.VALID_ORDER.index(from_phase)
        to_idx = cls.VALID_ORDER.index(to_phase)

        # Forward transition always allowed
        if to_idx == from_idx + 1:
            return result

        # Same phase allowed (stay)
        if to_idx == from_idx:
            return result

        # Backward only allowed on retry
        if to_idx < from_idx:
            if not is_retry:
                result.add_violation(
                    InvariantViolation(
                        code=InvariantCode.I3_PHASE_ORDER,
                        message=f"Backward transition without retry: {from_phase} -> {to_phase}",
                        expected="is_retry=True for backward transitions",
                    )
                )
            return result

        # Skip (forward by more than 1)
        if to_idx > from_idx + 1:
            result.add_violation(
                InvariantViolation(
                    code=InvariantCode.I3_PHASE_ORDER,
                    message=f"Phase skip not allowed: {from_phase} -> {to_phase}",
                    expected=cls.VALID_ORDER[from_idx + 1],
                    actual=to_phase,
                )
            )

        return result


# =============================================================================
# I4: GATES BEFORE SIGNOFF INVARIANT
# =============================================================================


class GatesBeforeSignoffEnforcer:
    """Enforces I4: Gates must be executed and passed before signoffs."""

    @classmethod
    def validate_signoff_allowed(
        cls,
        gate_status: str,
        gates_passed: List[str],
        required_gates: List[str],
    ) -> InvariantCheckResult:
        """Check if signoff is allowed based on gate status.

        Args:
            gate_status: Current gate status (PENDING, PASSED, FAILED).
            gates_passed: List of gates that passed.
            required_gates: List of required gates.

        Returns:
            InvariantCheckResult with any violations.
        """
        result = InvariantCheckResult(passed=True)

        # Gate status must be PASSED
        if gate_status != "PASSED":
            result.add_violation(
                InvariantViolation(
                    code=InvariantCode.I4_GATES_BEFORE_SIGNOFF,
                    message=f"Signoff blocked: gate_status is {gate_status}, not PASSED",
                    expected="PASSED",
                    actual=gate_status,
                )
            )
            return result

        # All required gates must have passed
        missing_gates = set(required_gates) - set(gates_passed)
        if missing_gates:
            result.add_violation(
                InvariantViolation(
                    code=InvariantCode.I4_GATES_BEFORE_SIGNOFF,
                    message=f"Signoff blocked: missing required gates {missing_gates}",
                    expected=str(required_gates),
                    actual=str(gates_passed),
                )
            )

        return result


# =============================================================================
# I5: EXECUTIVE VERIFICATION INVARIANT
# =============================================================================


class ExecutiveVerificationEnforcer:
    """Enforces I5: Execs verify subordinate work before approving.

    Hierarchy: CEO -> VP -> Master -> Lead -> Worker
    An executive can only sign off on work from direct subordinates.
    """

    HIERARCHY = {
        "CEO": ["VP_SPEC", "VP_EXEC", "VP_QA"],
        "VP_SPEC": ["SPEC_MASTER"],
        "VP_EXEC": ["ACE_EXEC"],
        "VP_QA": ["QA_MASTER"],
        "SPEC_MASTER": ["SQUAD_LEAD"],
        "ACE_EXEC": ["SQUAD_LEAD"],
        "QA_MASTER": ["SQUAD_LEAD"],
        "SQUAD_LEAD": ["WORKER"],
        "WORKER": [],
    }

    @classmethod
    def validate_signoff_authority(
        cls,
        signer_role: str,
        subordinate_role: str,
    ) -> InvariantCheckResult:
        """Validate that signer has authority over subordinate.

        Args:
            signer_role: Role of the signing agent.
            subordinate_role: Role of the agent whose work is being signed.

        Returns:
            InvariantCheckResult with any violations.
        """
        result = InvariantCheckResult(passed=True)

        if signer_role not in cls.HIERARCHY:
            result.add_violation(
                InvariantViolation(
                    code=InvariantCode.I5_EXECUTIVE_VERIFICATION,
                    message=f"Unknown signer role: {signer_role}",
                    expected=f"One of {list(cls.HIERARCHY.keys())}",
                    actual=signer_role,
                )
            )
            return result

        valid_subordinates = cls.HIERARCHY.get(signer_role, [])
        if subordinate_role not in valid_subordinates:
            result.add_violation(
                InvariantViolation(
                    code=InvariantCode.I5_EXECUTIVE_VERIFICATION,
                    message=f"{signer_role} cannot sign off on {subordinate_role} work",
                    expected=f"Subordinate should be one of {valid_subordinates}",
                    actual=subordinate_role,
                )
            )

        return result


# =============================================================================
# I6: TRUTHFULNESS INVARIANT
# =============================================================================


class TruthfulnessEnforcer:
    """Enforces I6: Signoff claims must match actual artifacts.

    A signoff claiming "tests pass" must be verified against actual test results.
    """

    @classmethod
    def validate_signoff_claims(
        cls,
        claims: Dict[str, Any],
        evidence: Dict[str, Any],
    ) -> InvariantCheckResult:
        """Validate signoff claims against evidence.

        Args:
            claims: Dict of claims (e.g., {"tests_pass": True, "coverage": 90}).
            evidence: Dict of evidence (e.g., {"test_result": {...}, "coverage_report": {...}}).

        Returns:
            InvariantCheckResult with any violations.
        """
        result = InvariantCheckResult(passed=True)

        # Check tests_pass claim
        if claims.get("tests_pass"):
            test_evidence = evidence.get("test_result", {})
            if not test_evidence.get("passed", False):
                result.add_violation(
                    InvariantViolation(
                        code=InvariantCode.I6_TRUTHFULNESS,
                        message="Claim 'tests_pass=True' contradicts test evidence",
                        expected="test_result.passed=True",
                        actual=f"test_result.passed={test_evidence.get('passed')}",
                    )
                )

        # Check coverage claim
        claimed_coverage = claims.get("coverage")
        if claimed_coverage is not None:
            actual_coverage = evidence.get("coverage_report", {}).get("line_coverage")
            if actual_coverage is not None and abs(claimed_coverage - actual_coverage) > 1:
                result.add_violation(
                    InvariantViolation(
                        code=InvariantCode.I6_TRUTHFULNESS,
                        message=f"Claimed coverage {claimed_coverage}% doesn't match actual {actual_coverage}%",
                        expected=str(claimed_coverage),
                        actual=str(actual_coverage),
                    )
                )

        # Check artifacts_created claim
        if claims.get("artifacts_created"):
            created_artifacts = evidence.get("artifacts", [])
            for artifact in claims["artifacts_created"]:
                if artifact not in created_artifacts:
                    result.add_violation(
                        InvariantViolation(
                            code=InvariantCode.I6_TRUTHFULNESS,
                            message=f"Claimed artifact '{artifact}' not found in evidence",
                            expected=artifact,
                            actual=str(created_artifacts),
                        )
                    )

        return result


# =============================================================================
# I7: AUDIT TRAIL INVARIANT
# =============================================================================


class AuditTrailEnforcer:
    """Enforces I7: Every decision has evidence bundle with manifest.

    Required evidence bundle structure:
    - manifest.json: List of all evidence files
    - decision.json: The decision record
    - evidence/: Directory with supporting files
    """

    REQUIRED_MANIFEST_FIELDS = ["decision_id", "timestamp", "evidence_files", "agent_id"]

    @classmethod
    def validate_evidence_bundle(
        cls,
        bundle_path: Path,
    ) -> InvariantCheckResult:
        """Validate evidence bundle structure.

        Args:
            bundle_path: Path to evidence bundle directory.

        Returns:
            InvariantCheckResult with any violations.
        """
        result = InvariantCheckResult(passed=True)

        # Check manifest exists
        manifest_path = bundle_path / "manifest.json"
        if not manifest_path.exists():
            result.add_violation(
                InvariantViolation(
                    code=InvariantCode.I7_AUDIT_TRAIL,
                    message="Evidence bundle missing manifest.json",
                    key=str(bundle_path),
                )
            )
            return result

        # Validate manifest structure
        try:
            import json
            with open(manifest_path) as f:
                manifest = json.load(f)

            for field in cls.REQUIRED_MANIFEST_FIELDS:
                if field not in manifest:
                    result.add_violation(
                        InvariantViolation(
                            code=InvariantCode.I7_AUDIT_TRAIL,
                            message=f"Manifest missing required field: {field}",
                            key=str(manifest_path),
                            expected=str(cls.REQUIRED_MANIFEST_FIELDS),
                        )
                    )

            # Verify evidence files exist
            for evidence_file in manifest.get("evidence_files", []):
                evidence_path = bundle_path / evidence_file
                if not evidence_path.exists():
                    result.add_violation(
                        InvariantViolation(
                            code=InvariantCode.I7_AUDIT_TRAIL,
                            message=f"Evidence file listed in manifest not found: {evidence_file}",
                            key=str(evidence_path),
                        )
                    )

        except json.JSONDecodeError as e:
            result.add_violation(
                InvariantViolation(
                    code=InvariantCode.I7_AUDIT_TRAIL,
                    message=f"Invalid manifest JSON: {e}",
                    key=str(manifest_path),
                )
            )

        return result


# =============================================================================
# I8: EVENT SCHEMA INVARIANT
# =============================================================================


class EventSchemaEnforcer:
    """Enforces I8: event_log uses schema_version, event_id, type.

    Every event in the pipeline MUST conform to a defined schema.
    This prevents malformed events from corrupting state or breaking
    event processing downstream.

    Required fields for ALL events:
    - event_id: Unique identifier for the event
    - event_type: Type of event (phase_transition, gate_result, etc.)
    - schema_version: Version of the event schema
    - timestamp: When the event occurred
    - run_id: Which run this event belongs to

    Additional required fields depend on event_type.
    """

    # Schema version for validation
    CURRENT_SCHEMA_VERSION = "1.0"

    # Base fields required for ALL events
    BASE_REQUIRED_FIELDS = {
        "event_id": str,
        "event_type": str,
        "schema_version": str,
        "timestamp": str,
        "run_id": str,
    }

    # Additional required fields per event type
    EVENT_TYPE_SCHEMAS: Dict[str, Dict[str, type]] = {
        "phase_transition": {
            "from_phase": str,
            "to_phase": str,
            "sprint_id": str,
        },
        "gate_result": {
            "gate_id": str,
            "passed": bool,
            "sprint_id": str,
        },
        "task_start": {
            "task_id": str,
            "node": str,
            "sprint_id": str,
        },
        "task_complete": {
            "task_id": str,
            "status": str,
            "sprint_id": str,
        },
        "violation": {
            "violation_id": str,
            "invariant_code": str,
            "severity": str,
        },
        "rework_queued": {
            "rework_id": str,
            "original_task_id": str,
            "attempt": int,
        },
        "checkpoint": {
            "checkpoint_id": str,
            "node": str,
            "attempt": int,
        },
        "signoff": {
            "signoff_id": str,
            "agent_id": str,
            "sprint_id": str,
        },
        "halt": {
            "reason": str,
            "halt_type": str,
        },
    }

    @classmethod
    def validate_event(
        cls,
        event: Dict[str, Any],
    ) -> InvariantCheckResult:
        """Validate an event against its schema.

        Args:
            event: The event dictionary to validate.

        Returns:
            InvariantCheckResult with any violations.
        """
        result = InvariantCheckResult(passed=True)

        # Check base required fields
        for field, expected_type in cls.BASE_REQUIRED_FIELDS.items():
            if field not in event:
                result.add_violation(
                    InvariantViolation(
                        code=InvariantCode.I8_EVENT_SCHEMA,
                        message=f"Event missing required base field: {field}",
                        expected=f"Field '{field}' of type {expected_type.__name__}",
                    )
                )
            elif not isinstance(event[field], expected_type):
                result.add_violation(
                    InvariantViolation(
                        code=InvariantCode.I8_EVENT_SCHEMA,
                        message=f"Event field '{field}' has wrong type",
                        expected=expected_type.__name__,
                        actual=type(event[field]).__name__,
                    )
                )

        # If base validation failed, don't continue
        if not result.passed:
            return result

        # Check schema version
        schema_version = event.get("schema_version", "")
        if schema_version != cls.CURRENT_SCHEMA_VERSION:
            result.add_violation(
                InvariantViolation(
                    code=InvariantCode.I8_EVENT_SCHEMA,
                    message=f"Event schema version mismatch",
                    expected=cls.CURRENT_SCHEMA_VERSION,
                    actual=schema_version,
                )
            )

        # Check event type specific fields
        event_type = event.get("event_type", "")
        type_schema = cls.EVENT_TYPE_SCHEMAS.get(event_type)

        if type_schema is None:
            # Unknown event type - this is a violation
            result.add_violation(
                InvariantViolation(
                    code=InvariantCode.I8_EVENT_SCHEMA,
                    message=f"Unknown event type: {event_type}",
                    expected=f"One of {list(cls.EVENT_TYPE_SCHEMAS.keys())}",
                    actual=event_type,
                )
            )
            return result

        # Validate type-specific fields
        for field, expected_type in type_schema.items():
            if field not in event:
                result.add_violation(
                    InvariantViolation(
                        code=InvariantCode.I8_EVENT_SCHEMA,
                        message=f"Event type '{event_type}' missing required field: {field}",
                        expected=f"Field '{field}' of type {expected_type.__name__}",
                    )
                )
            elif not isinstance(event[field], expected_type):
                result.add_violation(
                    InvariantViolation(
                        code=InvariantCode.I8_EVENT_SCHEMA,
                        message=f"Event field '{field}' has wrong type",
                        expected=expected_type.__name__,
                        actual=type(event[field]).__name__,
                    )
                )

        return result

    @classmethod
    def create_event(
        cls,
        event_type: str,
        run_id: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create a schema-compliant event.

        This is a helper to create events that will pass validation.

        Args:
            event_type: Type of event.
            run_id: Run identifier.
            **kwargs: Additional event fields.

        Returns:
            A properly formatted event dictionary.

        Raises:
            ValueError: If event_type is unknown.
        """
        import uuid

        if event_type not in cls.EVENT_TYPE_SCHEMAS:
            raise ValueError(f"Unknown event type: {event_type}")

        event = {
            "event_id": f"evt_{uuid.uuid4().hex[:12]}",
            "event_type": event_type,
            "schema_version": cls.CURRENT_SCHEMA_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            **kwargs,
        }

        return event


# =============================================================================
# I9: SAFE_HALT PRIORITY INVARIANT
# =============================================================================


class SafeHaltEnforcer:
    """Enforces I9: SAFE_HALT takes precedence over all other operations.

    When safe_halt is triggered:
    - All pending operations must complete or checkpoint
    - No new operations can be started
    - State must be persisted before shutdown
    """

    @classmethod
    def should_halt(
        cls,
        safe_halt_triggered: bool,
        current_operation: Optional[str] = None,
    ) -> Tuple[bool, Optional[InvariantViolation]]:
        """Check if operation should halt.

        Args:
            safe_halt_triggered: Whether SAFE_HALT has been triggered.
            current_operation: Current operation name (if any).

        Returns:
            Tuple of (should_halt, violation_if_continuing).
        """
        if safe_halt_triggered:
            if current_operation:
                violation = InvariantViolation(
                    code=InvariantCode.I9_SAFE_HALT_PRIORITY,
                    message=f"Operation '{current_operation}' must halt due to SAFE_HALT",
                    key=current_operation,
                )
                return True, violation
            return True, None
        return False, None

    @classmethod
    def validate_halt_state(
        cls,
        state_persisted: bool,
        checkpoint_created: bool,
        pending_operations: List[str],
    ) -> InvariantCheckResult:
        """Validate state after SAFE_HALT.

        Args:
            state_persisted: Whether state was persisted.
            checkpoint_created: Whether checkpoint was created.
            pending_operations: List of still-pending operations.

        Returns:
            InvariantCheckResult with any violations.
        """
        result = InvariantCheckResult(passed=True)

        if not state_persisted:
            result.add_violation(
                InvariantViolation(
                    code=InvariantCode.I9_SAFE_HALT_PRIORITY,
                    message="SAFE_HALT: State not persisted before shutdown",
                    expected="state_persisted=True",
                )
            )

        if not checkpoint_created:
            result.add_violation(
                InvariantViolation(
                    code=InvariantCode.I9_SAFE_HALT_PRIORITY,
                    message="SAFE_HALT: Checkpoint not created",
                    expected="checkpoint_created=True",
                )
            )

        if pending_operations:
            result.add_violation(
                InvariantViolation(
                    code=InvariantCode.I9_SAFE_HALT_PRIORITY,
                    message=f"SAFE_HALT: {len(pending_operations)} operations still pending",
                    expected="No pending operations",
                    actual=str(pending_operations),
                )
            )

        return result


# =============================================================================
# I10: REDIS CANONICAL INVARIANT
# =============================================================================


class RedisCanonicalEnforcer:
    """Enforces I10: Redis is canonical; file is mirror for resilience.

    Write path: Redis first, then file mirror
    Read path: Redis first, fallback to file if Redis unavailable
    """

    @classmethod
    def validate_write_order(
        cls,
        redis_written: bool,
        file_written: bool,
        redis_timestamp: Optional[str] = None,
        file_timestamp: Optional[str] = None,
    ) -> InvariantCheckResult:
        """Validate write order (Redis before file).

        Args:
            redis_written: Whether Redis write succeeded.
            file_written: Whether file write succeeded.
            redis_timestamp: Timestamp of Redis write.
            file_timestamp: Timestamp of file write.

        Returns:
            InvariantCheckResult with any violations.
        """
        result = InvariantCheckResult(passed=True)

        # Redis must be written
        if not redis_written:
            result.add_violation(
                InvariantViolation(
                    code=InvariantCode.I10_REDIS_CANONICAL,
                    message="Redis (canonical) write failed",
                    expected="redis_written=True",
                )
            )

        # If both written, Redis should be first
        if redis_timestamp and file_timestamp:
            if file_timestamp < redis_timestamp:
                result.add_violation(
                    InvariantViolation(
                        code=InvariantCode.I10_REDIS_CANONICAL,
                        message="File written before Redis (wrong order)",
                        expected="Redis timestamp < File timestamp",
                        actual=f"Redis: {redis_timestamp}, File: {file_timestamp}",
                    )
                )

        return result

    @classmethod
    def validate_consistency(
        cls,
        redis_data: Optional[Dict[str, Any]],
        file_data: Optional[Dict[str, Any]],
    ) -> InvariantCheckResult:
        """Validate Redis/file data consistency.

        Args:
            redis_data: Data from Redis.
            file_data: Data from file mirror.

        Returns:
            InvariantCheckResult with any violations.
        """
        result = InvariantCheckResult(passed=True)

        if redis_data is None and file_data is None:
            return result  # Both empty is valid

        if redis_data is not None and file_data is not None:
            # Compare key fields (not full equality due to timestamps)
            redis_version = redis_data.get("version")
            file_version = file_data.get("version")
            if redis_version != file_version:
                result.add_violation(
                    InvariantViolation(
                        code=InvariantCode.I10_REDIS_CANONICAL,
                        message="Version mismatch between Redis and file mirror",
                        expected=f"redis_version == file_version",
                        actual=f"Redis: {redis_version}, File: {file_version}",
                    )
                )

        return result


# =============================================================================
# I11: RUNAWAY PROTECTION INVARIANT
# =============================================================================


class RunawayProtectionEnforcer:
    """Enforces I11: Worker limits and cost controls prevent infinite loops.

    Limits:
    - Max retries per node: 3
    - Max total attempts per sprint: 10
    - Max API cost per sprint: $10
    - Max wall time per sprint: 30 minutes
    """

    MAX_RETRIES_PER_NODE = 3
    MAX_TOTAL_ATTEMPTS = 10
    MAX_COST_USD = 10.0
    MAX_WALL_TIME_SECONDS = 1800  # 30 minutes

    @classmethod
    def validate_retry_limit(
        cls,
        node: str,
        current_retries: int,
    ) -> InvariantCheckResult:
        """Validate retry count for a node.

        Args:
            node: Node name.
            current_retries: Current retry count.

        Returns:
            InvariantCheckResult with any violations.
        """
        result = InvariantCheckResult(passed=True)

        if current_retries >= cls.MAX_RETRIES_PER_NODE:
            result.add_violation(
                InvariantViolation(
                    code=InvariantCode.I11_RUNAWAY_PROTECTION,
                    message=f"Node '{node}' exceeded max retries ({cls.MAX_RETRIES_PER_NODE})",
                    key=node,
                    expected=f"retries < {cls.MAX_RETRIES_PER_NODE}",
                    actual=str(current_retries),
                )
            )

        return result

    @classmethod
    def validate_total_attempts(
        cls,
        total_attempts: int,
    ) -> InvariantCheckResult:
        """Validate total attempts across all nodes.

        Args:
            total_attempts: Total attempts in the sprint.

        Returns:
            InvariantCheckResult with any violations.
        """
        result = InvariantCheckResult(passed=True)

        if total_attempts >= cls.MAX_TOTAL_ATTEMPTS:
            result.add_violation(
                InvariantViolation(
                    code=InvariantCode.I11_RUNAWAY_PROTECTION,
                    message=f"Sprint exceeded max total attempts ({cls.MAX_TOTAL_ATTEMPTS})",
                    expected=f"attempts < {cls.MAX_TOTAL_ATTEMPTS}",
                    actual=str(total_attempts),
                )
            )

        return result

    @classmethod
    def validate_cost(
        cls,
        current_cost_usd: float,
    ) -> InvariantCheckResult:
        """Validate API cost.

        Args:
            current_cost_usd: Current cost in USD.

        Returns:
            InvariantCheckResult with any violations.
        """
        result = InvariantCheckResult(passed=True)

        if current_cost_usd >= cls.MAX_COST_USD:
            result.add_violation(
                InvariantViolation(
                    code=InvariantCode.I11_RUNAWAY_PROTECTION,
                    message=f"Sprint exceeded max cost (${cls.MAX_COST_USD})",
                    expected=f"cost < ${cls.MAX_COST_USD}",
                    actual=f"${current_cost_usd:.2f}",
                )
            )

        return result

    @classmethod
    def validate_wall_time(
        cls,
        elapsed_seconds: float,
    ) -> InvariantCheckResult:
        """Validate wall time.

        Args:
            elapsed_seconds: Elapsed time in seconds.

        Returns:
            InvariantCheckResult with any violations.
        """
        result = InvariantCheckResult(passed=True)

        if elapsed_seconds >= cls.MAX_WALL_TIME_SECONDS:
            result.add_violation(
                InvariantViolation(
                    code=InvariantCode.I11_RUNAWAY_PROTECTION,
                    message=f"Sprint exceeded max wall time ({cls.MAX_WALL_TIME_SECONDS}s)",
                    expected=f"elapsed < {cls.MAX_WALL_TIME_SECONDS}s",
                    actual=f"{elapsed_seconds:.1f}s",
                )
            )

        return result


# =============================================================================
# COMBINED INVARIANT CHECKER
# =============================================================================


class InvariantChecker:
    """Combined checker for all invariants.

    Usage:
        checker = InvariantChecker(run_id="run_001", sprint_id="S00")
        result = checker.check_all(state)
    """

    def __init__(
        self,
        run_id: str,
        sprint_id: str,
    ):
        """Initialize the checker.

        Args:
            run_id: Run identifier.
            sprint_id: Sprint identifier.
        """
        self.run_id = run_id
        self.sprint_id = sprint_id
        self.idempotency = IdempotencyEnforcer()

    def check_key_namespacing(self, key: str) -> InvariantCheckResult:
        """Check I1: Key namespacing."""
        return NamespacingEnforcer.validate_key(key, self.run_id, self.sprint_id)

    def check_phase_transition(
        self,
        from_phase: str,
        to_phase: str,
        is_retry: bool = False,
    ) -> InvariantCheckResult:
        """Check I3: Phase order."""
        return PhaseOrderEnforcer.validate_transition(from_phase, to_phase, is_retry)

    def check_signoff_allowed(
        self,
        gate_status: str,
        gates_passed: List[str],
        required_gates: Optional[List[str]] = None,
    ) -> InvariantCheckResult:
        """Check I4: Gates before signoff."""
        if required_gates is None:
            required_gates = ["G0"]  # Minimum required gate
        return GatesBeforeSignoffEnforcer.validate_signoff_allowed(
            gate_status, gates_passed, required_gates
        )

    def format_namespaced_key(self, key_type: str, *parts: str) -> str:
        """Format a key with proper namespacing (I1 compliant)."""
        return NamespacingEnforcer.format_key(
            key_type, self.run_id, self.sprint_id, *parts
        )

    def check_executive_authority(
        self,
        signer_role: str,
        subordinate_role: str,
    ) -> InvariantCheckResult:
        """Check I5: Executive verification."""
        return ExecutiveVerificationEnforcer.validate_signoff_authority(
            signer_role, subordinate_role
        )

    def check_signoff_truthfulness(
        self,
        claims: Dict[str, Any],
        evidence: Dict[str, Any],
    ) -> InvariantCheckResult:
        """Check I6: Truthfulness."""
        return TruthfulnessEnforcer.validate_signoff_claims(claims, evidence)

    def check_evidence_bundle(self, bundle_path: Path) -> InvariantCheckResult:
        """Check I7: Audit trail."""
        return AuditTrailEnforcer.validate_evidence_bundle(bundle_path)

    def check_event_schema(self, event: Dict[str, Any]) -> InvariantCheckResult:
        """Check I8: Event schema validation."""
        return EventSchemaEnforcer.validate_event(event)

    def create_event(self, event_type: str, **kwargs) -> Dict[str, Any]:
        """Create a schema-compliant event (I8 helper)."""
        return EventSchemaEnforcer.create_event(event_type, self.run_id, **kwargs)

    def check_safe_halt(
        self,
        safe_halt_triggered: bool,
        current_operation: Optional[str] = None,
    ) -> Tuple[bool, Optional[InvariantViolation]]:
        """Check I9: SAFE_HALT priority."""
        return SafeHaltEnforcer.should_halt(safe_halt_triggered, current_operation)

    def check_redis_write_order(
        self,
        redis_written: bool,
        file_written: bool,
        redis_timestamp: Optional[str] = None,
        file_timestamp: Optional[str] = None,
    ) -> InvariantCheckResult:
        """Check I10: Redis canonical (write order)."""
        return RedisCanonicalEnforcer.validate_write_order(
            redis_written, file_written, redis_timestamp, file_timestamp
        )

    def check_runaway_retry(
        self,
        node: str,
        current_retries: int,
    ) -> InvariantCheckResult:
        """Check I11: Runaway protection (retry limit)."""
        return RunawayProtectionEnforcer.validate_retry_limit(node, current_retries)

    def check_runaway_cost(self, current_cost_usd: float) -> InvariantCheckResult:
        """Check I11: Runaway protection (cost limit)."""
        return RunawayProtectionEnforcer.validate_cost(current_cost_usd)

    def check_runaway_time(self, elapsed_seconds: float) -> InvariantCheckResult:
        """Check I11: Runaway protection (time limit)."""
        return RunawayProtectionEnforcer.validate_wall_time(elapsed_seconds)


# =============================================================================
# SINGLETON ACCESS
# =============================================================================


_checker_instances: Dict[str, InvariantChecker] = {}


def get_invariant_checker(run_id: str, sprint_id: str) -> InvariantChecker:
    """Get or create invariant checker for a run/sprint.

    Args:
        run_id: Run identifier.
        sprint_id: Sprint identifier.

    Returns:
        InvariantChecker instance.
    """
    key = f"{run_id}:{sprint_id}"
    if key not in _checker_instances:
        _checker_instances[key] = InvariantChecker(run_id, sprint_id)
    return _checker_instances[key]


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Codes
    "InvariantCode",
    # Types
    "InvariantViolation",
    "InvariantCheckResult",
    # Exception for strict enforcement
    "InvariantViolationError",
    # Enforcement helpers
    "enforce_invariant",
    "enforce_or_warn",
    # Enforcers (I1-I4)
    "NamespacingEnforcer",
    "IdempotencyEnforcer",
    "PhaseOrderEnforcer",
    "GatesBeforeSignoffEnforcer",
    # Enforcers (I5-I11)
    "ExecutiveVerificationEnforcer",
    "TruthfulnessEnforcer",
    "AuditTrailEnforcer",
    "EventSchemaEnforcer",  # I8 - NEW
    "SafeHaltEnforcer",
    "RedisCanonicalEnforcer",
    "RunawayProtectionEnforcer",
    # Combined
    "InvariantChecker",
    "get_invariant_checker",
]
