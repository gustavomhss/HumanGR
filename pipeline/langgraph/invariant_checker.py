"""Invariant Checker Module - Compatibility alias.

This module provides backwards compatibility for imports expecting
`invariant_checker` (singular) instead of `invariants` (plural).

All functionality is provided by the `invariants` module.
"""

from pipeline.langgraph.invariants import (
    # Main classes
    InvariantChecker,
    InvariantCheckResult,
    InvariantViolation,
    InvariantCode,
    InvariantViolationError,
    # Enforcers
    NamespacingEnforcer,
    IdempotencyEnforcer,
    PhaseOrderEnforcer,
    GatesBeforeSignoffEnforcer,
    ExecutiveVerificationEnforcer,
    TruthfulnessEnforcer,
    AuditTrailEnforcer,
    SafeHaltEnforcer,
    RedisCanonicalEnforcer,
    RunawayProtectionEnforcer,
    # Functions
    get_invariant_checker,
)

# Alias for backwards compatibility
InvariantResult = InvariantCheckResult

__all__ = [
    "InvariantChecker",
    "InvariantCheckResult",
    "InvariantResult",
    "InvariantViolation",
    "InvariantCode",
    "InvariantViolationError",
    "NamespacingEnforcer",
    "IdempotencyEnforcer",
    "PhaseOrderEnforcer",
    "GatesBeforeSignoffEnforcer",
    "ExecutiveVerificationEnforcer",
    "TruthfulnessEnforcer",
    "AuditTrailEnforcer",
    "SafeHaltEnforcer",
    "RedisCanonicalEnforcer",
    "RunawayProtectionEnforcer",
    "get_invariant_checker",
]
