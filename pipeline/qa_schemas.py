"""QA Master schemas and failure classification.

This module provides:
1. GateFailureType enum for classifying gate failures
2. classify_gate_failure() to determine failure type from gate result
3. get_failure_remediation() to get appropriate remediation strategy

The key insight is that different failure types need different handling:
- CODE failures: Agents can fix by modifying code
- TIMEOUT failures: Infrastructure issue - agents CANNOT fix
- INFRASTRUCTURE failures: Ops intervention needed
- SECURITY failures: Human review required
- PERFORMANCE failures: Agents can optimize

Created: 2026-01-28
Reference: docs/pipeline/ROOT_CAUSE_ANALYSIS_DEEP_STUDY.md
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone


class GateFailureType(Enum):
    """Classification of gate failure types.

    Different failure types require different remediation strategies:
    - CODE: Agents can fix by modifying code
    - TIMEOUT: Infrastructure issue - increase timeout or investigate
    - INFRASTRUCTURE: Ops intervention needed (command not found, permission denied)
    - SECURITY: Human review required (secrets detected, policy violation)
    - PERFORMANCE: Agents can optimize (coverage below threshold)
    """

    CODE = "code"
    TIMEOUT = "timeout"
    INFRASTRUCTURE = "infrastructure"
    SECURITY = "security"
    PERFORMANCE = "performance"


@dataclass
class FailureClassification:
    """Detailed classification of a gate failure."""

    failure_type: GateFailureType
    exit_code: int
    gate_id: str
    can_rework: bool
    delegate_to: Optional[str]
    escalate_to: Optional[str]
    max_attempts: int
    reason: str


def classify_gate_failure(gate_result: Dict[str, Any]) -> GateFailureType:
    """Classify a gate failure based on exit_code and other signals.

    Args:
        gate_result: Dictionary containing gate execution result with:
            - exit_code: Process exit code (0=success, 124=timeout, etc)
            - sph_hits: List of secret pattern hits
            - status: PASS/FAIL/BLOCK
            - details: Optional details string

    Returns:
        GateFailureType indicating what kind of failure occurred.

    Exit Code Mapping:
        0: Success (should not be called for this)
        124: Timeout - process killed after timeout
        126: Permission denied
        127: Command not found
        1-125: Generic failure (usually code issue)
    """
    exit_code = gate_result.get("exit_code", 1)
    sph_hits = gate_result.get("sph_hits", [])
    status = gate_result.get("status", "FAIL")
    details = gate_result.get("details", "")

    # Security violations take precedence
    if sph_hits or status == "BLOCK":
        # Check if it's a security block vs dependency block
        sph_str = str(sph_hits)
        if "SECURITY_BLOCK" in sph_str or "POST_SECURITY_BLOCK" in sph_str:
            return GateFailureType.SECURITY
        if "DEPENDENCY_VIOLATION" in sph_str:
            return GateFailureType.INFRASTRUCTURE
        # Generic SPH hit (secret detected)
        if any(
            pattern in sph_str
            for pattern in ["aws_access_key", "private_key", "bearer_token", "api_key"]
        ):
            return GateFailureType.SECURITY

    # Timeout detection (exit_code=124 is standard Unix timeout)
    if exit_code == 124:
        return GateFailureType.TIMEOUT

    # Infrastructure issues
    if exit_code == 126:  # Permission denied
        return GateFailureType.INFRASTRUCTURE
    if exit_code == 127:  # Command not found
        return GateFailureType.INFRASTRUCTURE

    # Performance issues (coverage below threshold, etc)
    # Check details string for coverage-related failures
    if details:
        details_lower = details.lower()
        if any(
            term in details_lower
            for term in ["coverage", "threshold", "below", "percentage"]
        ):
            return GateFailureType.PERFORMANCE

    # Default: assume code issue that agents can fix
    return GateFailureType.CODE


def get_failure_remediation(failure_type: GateFailureType) -> Dict[str, Any]:
    """Get remediation strategy for a failure type.

    Args:
        failure_type: The classified failure type

    Returns:
        Dictionary with:
            - can_rework: Whether agents can fix this
            - delegate_to: Who should handle (agent or human)
            - max_attempts: How many retries allowed
            - escalate_to: Who to escalate to if retries exhausted
            - action: Description of what to do
    """
    strategies = {
        GateFailureType.CODE: {
            "can_rework": True,
            "delegate_to": "ace_exec",
            "max_attempts": 3,
            "escalate_to": "qa_master",
            "action": "fix_code_and_retest",
            "description": "Code issue - agents can fix by analyzing failure and modifying code",
        },
        GateFailureType.TIMEOUT: {
            "can_rework": False,  # Agents can't fix timeout
            "delegate_to": None,
            "max_attempts": 1,  # Just retry once with longer timeout
            "escalate_to": "ops_ctrl",
            "action": "increase_timeout_or_investigate",
            "description": "Timeout - infrastructure issue, not a code problem",
        },
        GateFailureType.INFRASTRUCTURE: {
            "can_rework": False,
            "delegate_to": None,
            "max_attempts": 0,
            "escalate_to": "ops_ctrl",
            "action": "fix_infrastructure",
            "description": "Infrastructure issue - command not found or permission denied",
        },
        GateFailureType.SECURITY: {
            "can_rework": False,  # Needs human review
            "delegate_to": "human_reviewer",
            "max_attempts": 0,
            "escalate_to": "human_layer",
            "action": "human_review_required",
            "description": "Security violation - requires human review before proceeding",
        },
        GateFailureType.PERFORMANCE: {
            "can_rework": True,
            "delegate_to": "ace_exec",
            "max_attempts": 2,
            "escalate_to": "qa_master",
            "action": "optimize_and_retest",
            "description": "Performance/coverage issue - agents can add tests or optimize",
        },
    }
    return strategies.get(failure_type, strategies[GateFailureType.CODE])


def classify_all_gate_failures(
    gates_failed: List[Dict[str, Any]]
) -> Dict[str, FailureClassification]:
    """Classify all failed gates and return detailed classifications.

    Args:
        gates_failed: List of gate result dictionaries

    Returns:
        Dictionary mapping gate_id to FailureClassification
    """
    classifications = {}

    for gate in gates_failed:
        gate_id = gate.get("gate_id", "unknown")
        exit_code = gate.get("exit_code", 1)
        failure_type = classify_gate_failure(gate)
        remediation = get_failure_remediation(failure_type)

        classifications[gate_id] = FailureClassification(
            failure_type=failure_type,
            exit_code=exit_code,
            gate_id=gate_id,
            can_rework=remediation["can_rework"],
            delegate_to=remediation["delegate_to"],
            escalate_to=remediation["escalate_to"],
            max_attempts=remediation["max_attempts"],
            reason=remediation["description"],
        )

    return classifications


def get_dominant_failure_type(
    classifications: Dict[str, FailureClassification]
) -> GateFailureType:
    """Get the most severe/dominant failure type from a set of classifications.

    Severity order (highest to lowest):
    1. SECURITY - requires human review
    2. INFRASTRUCTURE - requires ops
    3. TIMEOUT - requires investigation
    4. PERFORMANCE - agents can fix
    5. CODE - agents can fix

    Args:
        classifications: Dictionary of gate_id to FailureClassification

    Returns:
        The most severe failure type found
    """
    if not classifications:
        return GateFailureType.CODE

    severity_order = [
        GateFailureType.SECURITY,
        GateFailureType.INFRASTRUCTURE,
        GateFailureType.TIMEOUT,
        GateFailureType.PERFORMANCE,
        GateFailureType.CODE,
    ]

    failure_types = {c.failure_type for c in classifications.values()}

    for severity in severity_order:
        if severity in failure_types:
            return severity

    return GateFailureType.CODE


# =============================================================================
# QA Task and Decision Schemas (for future QA Master integration)
# =============================================================================


class QATaskStatus(Enum):
    """Status of a QA task."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class QADecision(Enum):
    """QA Master decision types."""

    APPROVE = "approve"
    REJECT = "reject"
    REWORK = "rework"
    ESCALATE = "escalate"
    DEFER = "defer"


@dataclass
class QATaskResult:
    """Result of a single QA task execution."""

    task_id: str
    task_type: str  # "gate", "review", "audit", etc.
    status: QATaskStatus
    agent_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)
    evidence_paths: List[str] = field(default_factory=list)


@dataclass
class QAMasterDecision:
    """Final decision from QA Master after all validations."""

    decision: QADecision
    sprint_id: str
    run_id: str
    gates_passed: List[str]
    gates_failed: List[str]
    failure_classifications: Dict[str, str]  # gate_id -> failure_type
    dominant_failure_type: Optional[str]
    can_rework: bool
    delegate_to: Optional[str]
    escalate_to: Optional[str]
    rejection_reason: Optional[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "decision": self.decision.value,
            "sprint_id": self.sprint_id,
            "run_id": self.run_id,
            "gates_passed": self.gates_passed,
            "gates_failed": self.gates_failed,
            "failure_classifications": self.failure_classifications,
            "dominant_failure_type": self.dominant_failure_type,
            "can_rework": self.can_rework,
            "delegate_to": self.delegate_to,
            "escalate_to": self.escalate_to,
            "rejection_reason": self.rejection_reason,
            "timestamp": self.timestamp.isoformat(),
        }
