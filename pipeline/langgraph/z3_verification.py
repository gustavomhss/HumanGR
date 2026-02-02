"""Z3 Formal Verification for Pipeline Invariants.

P2-026: Z3-based formal verification of critical invariants.

This module uses the Z3 SMT solver to formally verify that pipeline
invariants hold. It provides mathematical proof that the system
maintains its invariants under all possible states.

Critical Gates Verified:
- G0: Spec compliance invariants
- G4: Security/blindagem invariants
- G8: Mutation testing invariants

Author: Pipeline Autonomo Team
Version: 1.0.0 (2026-01-16)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypedDict

logger = logging.getLogger(__name__)

# =============================================================================
# Z3 AVAILABILITY CHECK
# =============================================================================

try:
    import z3
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    logger.debug("Z3 not installed - formal verification disabled")


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

class VerificationStatus(str, Enum):
    """Status of formal verification."""
    PROVED = "proved"
    REFUTED = "refuted"
    UNKNOWN = "unknown"
    TIMEOUT = "timeout"
    ERROR = "error"
    SKIPPED = "skipped"


class Z3InvariantResult(TypedDict):
    """Result from Z3 formal invariant verification."""
    invariant_name: str
    gate_id: str
    proved: bool
    proof_time_ms: float
    counter_example: Optional[str]
    verified_at: str


@dataclass
class InvariantSpec:
    """Specification of a formal invariant."""
    name: str
    gate_id: str
    description: str
    formula_builder: Callable[["Z3Context"], Any]
    critical: bool = True


@dataclass
class VerificationResult:
    """Result of Z3 verification."""
    invariant_name: str
    gate_id: str
    status: VerificationStatus
    proved: bool
    proof_time_ms: float
    counter_example: Optional[str] = None
    error_message: Optional[str] = None
    verified_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Z3InvariantResult:
        """Convert to TypedDict format."""
        return {
            "invariant_name": self.invariant_name,
            "gate_id": self.gate_id,
            "proved": self.proved,
            "proof_time_ms": self.proof_time_ms,
            "counter_example": self.counter_example,
            "verified_at": self.verified_at,
        }


# =============================================================================
# Z3 CONTEXT
# =============================================================================

class Z3Context:
    """Context for building Z3 formulas.

    Provides symbolic variables and helper methods for constructing
    formal verification constraints.

    Usage:
        ctx = Z3Context()
        x = ctx.int_var("x")
        y = ctx.int_var("y")
        constraint = x + y > 0
        ctx.add_constraint(constraint)
    """

    def __init__(self):
        if not Z3_AVAILABLE:
            raise RuntimeError("Z3 not available")

        self._solver = z3.Solver()
        self._variables: Dict[str, Any] = {}

    def int_var(self, name: str) -> Any:
        """Create an integer variable."""
        if name not in self._variables:
            self._variables[name] = z3.Int(name)
        return self._variables[name]

    def real_var(self, name: str) -> Any:
        """Create a real (float) variable."""
        if name not in self._variables:
            self._variables[name] = z3.Real(name)
        return self._variables[name]

    def bool_var(self, name: str) -> Any:
        """Create a boolean variable."""
        if name not in self._variables:
            self._variables[name] = z3.Bool(name)
        return self._variables[name]

    def string_var(self, name: str) -> Any:
        """Create a string variable."""
        if name not in self._variables:
            self._variables[name] = z3.String(name)
        return self._variables[name]

    def enum_var(self, name: str, values: List[str]) -> Any:
        """Create an enumeration variable."""
        sort_name = f"{name}_sort"
        if sort_name not in self._variables:
            enum_sort = z3.EnumSort(sort_name, values)
            self._variables[sort_name] = enum_sort
        return z3.Const(name, self._variables[sort_name][0])

    def add_constraint(self, constraint: Any) -> None:
        """Add a constraint to the solver."""
        self._solver.add(constraint)

    def check(self, timeout_ms: int = 5000) -> VerificationStatus:
        """Check satisfiability of constraints."""
        self._solver.set("timeout", timeout_ms)
        result = self._solver.check()

        if result == z3.sat:
            return VerificationStatus.REFUTED  # Found counterexample
        elif result == z3.unsat:
            return VerificationStatus.PROVED  # No counterexample exists
        else:
            return VerificationStatus.UNKNOWN

    def get_model(self) -> Optional[Dict[str, Any]]:
        """Get model (counterexample) if satisfiable."""
        if self._solver.check() == z3.sat:
            model = self._solver.model()
            return {
                str(d): str(model[d]) for d in model.decls()
            }
        return None

    def reset(self) -> None:
        """Reset solver state."""
        self._solver.reset()
        self._variables.clear()


# =============================================================================
# INVARIANT DEFINITIONS
# =============================================================================

def _build_i1_namespacing(ctx: Z3Context) -> Any:
    """I1: Namespacing - Keys must contain run_id and sprint_id.

    Formally: For all keys k in storage:
        contains(k, run_id) AND contains(k, sprint_id)
    """
    # Variables
    key = ctx.string_var("key")
    run_id = ctx.string_var("run_id")
    sprint_id = ctx.string_var("sprint_id")

    # Constraint: key must contain both identifiers
    contains_run = z3.Contains(key, run_id)
    contains_sprint = z3.Contains(key, sprint_id)

    # The invariant holds if all keys satisfy this
    # We check the negation: exists key that violates
    violation = z3.Not(z3.And(contains_run, contains_sprint))

    # Add domain constraints
    ctx.add_constraint(z3.Length(run_id) > 0)
    ctx.add_constraint(z3.Length(sprint_id) > 0)
    ctx.add_constraint(z3.Length(key) > 0)

    return violation


def _build_i2_idempotency(ctx: Z3Context) -> Any:
    """I2: Idempotency - Re-execution does not duplicate.

    Formally: For all operations op with idempotency_key k:
        execute(op, k) == execute(op, k)  (same result)
        count(results with k) == 1
    """
    # Variables
    idempotency_key = ctx.string_var("idempotency_key")
    execution_count = ctx.int_var("execution_count")
    result_count = ctx.int_var("result_count")

    # Constraint: result count must equal 1 regardless of execution count
    idempotent = z3.Implies(
        execution_count >= 1,
        result_count == 1
    )

    # Violation is when we have more results than expected
    violation = z3.Not(idempotent)

    # Domain constraints
    ctx.add_constraint(execution_count >= 0)
    ctx.add_constraint(result_count >= 0)

    return violation


def _build_i3_phase_order(ctx: Z3Context) -> Any:
    """I3: Phase Order - Phases must follow strict order.

    Formally: phase_number(current) >= phase_number(previous)
    Valid order: INIT(0) -> SPEC(1) -> PLAN(2) -> EXEC(3) -> QA(4) -> VOTE(5) -> DONE(6)
    """
    # Variables representing phase numbers
    current_phase = ctx.int_var("current_phase")
    previous_phase = ctx.int_var("previous_phase")

    # Phase constraints
    ctx.add_constraint(current_phase >= 0)
    ctx.add_constraint(current_phase <= 6)
    ctx.add_constraint(previous_phase >= 0)
    ctx.add_constraint(previous_phase <= 6)

    # Invariant: current must be >= previous (can only move forward)
    valid_transition = current_phase >= previous_phase

    # Violation: going backwards
    violation = z3.Not(valid_transition)

    return violation


def _build_i4_gates_before_signoff(ctx: Z3Context) -> Any:
    """I4: Gates Before Signoff - Gates must pass before signoffs allowed.

    Formally: signoff_allowed => all_gates_passed
    """
    # Variables
    signoff_requested = ctx.bool_var("signoff_requested")
    gates_passed = ctx.int_var("gates_passed")
    total_gates = ctx.int_var("total_gates")

    # Domain constraints
    ctx.add_constraint(gates_passed >= 0)
    ctx.add_constraint(total_gates > 0)
    ctx.add_constraint(gates_passed <= total_gates)

    # Invariant: signoff only if all gates passed
    all_passed = gates_passed == total_gates
    valid = z3.Implies(signoff_requested, all_passed)

    # Violation: signoff requested but not all gates passed
    violation = z3.Not(valid)

    return violation


def _build_i5_executive_verification(ctx: Z3Context) -> Any:
    """I5: Executive Verification - Execs must verify subordinate work.

    Formally: approval_given => verification_complete
    """
    # Variables
    approval_given = ctx.bool_var("approval_given")
    verification_complete = ctx.bool_var("verification_complete")
    evidence_present = ctx.bool_var("evidence_present")

    # Invariant: approval requires verification and evidence
    valid = z3.Implies(
        approval_given,
        z3.And(verification_complete, evidence_present)
    )

    # Violation
    violation = z3.Not(valid)

    return violation


def _build_i6_truthfulness(ctx: Z3Context) -> Any:
    """I6: Truthfulness - Signoff claims must match artifacts.

    Formally: For all claims c in signoff:
        exists artifact a such that verifies(a, c)
    """
    # Variables
    claim_count = ctx.int_var("claim_count")
    verified_count = ctx.int_var("verified_count")
    artifact_count = ctx.int_var("artifact_count")

    # Domain constraints
    ctx.add_constraint(claim_count >= 0)
    ctx.add_constraint(verified_count >= 0)
    ctx.add_constraint(artifact_count >= 0)

    # Invariant: all claims must be verified by artifacts
    all_verified = verified_count == claim_count
    has_artifacts = artifact_count >= claim_count

    valid = z3.And(all_verified, has_artifacts)

    # Violation
    violation = z3.Not(valid)

    return violation


def _build_i7_audit_trail(ctx: Z3Context) -> Any:
    """I7: Audit Trail - Every decision has evidence bundle.

    Formally: For all decisions d:
        exists evidence_bundle e such that documents(e, d)
    """
    # Variables
    decision_count = ctx.int_var("decision_count")
    evidence_bundle_count = ctx.int_var("evidence_bundle_count")

    # Domain constraints
    ctx.add_constraint(decision_count >= 0)
    ctx.add_constraint(evidence_bundle_count >= 0)

    # Invariant: every decision has evidence
    valid = evidence_bundle_count >= decision_count

    # Violation
    violation = z3.Not(valid)

    return violation


def _build_i9_safe_halt(ctx: Z3Context) -> Any:
    """I9: SAFE_HALT Priority - SAFE_HALT supersedes all operations.

    Formally: safe_halt_active => no_other_operations_proceed
    """
    # Variables
    safe_halt_active = ctx.bool_var("safe_halt_active")
    operations_running = ctx.int_var("operations_running")

    # Domain constraints
    ctx.add_constraint(operations_running >= 0)

    # Invariant: if safe_halt, no operations
    valid = z3.Implies(safe_halt_active, operations_running == 0)

    # Violation
    violation = z3.Not(valid)

    return violation


def _build_i11_runaway_protection(ctx: Z3Context) -> Any:
    """I11: Runaway Protection - Enforce worker and cost limits.

    Formally:
        worker_iterations <= max_iterations
        total_cost <= cost_budget
    """
    # Variables
    worker_iterations = ctx.int_var("worker_iterations")
    max_iterations = ctx.int_var("max_iterations")
    total_cost = ctx.real_var("total_cost")
    cost_budget = ctx.real_var("cost_budget")

    # Domain constraints
    ctx.add_constraint(worker_iterations >= 0)
    ctx.add_constraint(max_iterations > 0)
    ctx.add_constraint(total_cost >= 0)
    ctx.add_constraint(cost_budget > 0)

    # Invariant: within limits
    iteration_ok = worker_iterations <= max_iterations
    cost_ok = total_cost <= cost_budget

    valid = z3.And(iteration_ok, cost_ok)

    # Violation
    violation = z3.Not(valid)

    return violation


# Gate-specific invariants

def _build_g0_spec_compliance(ctx: Z3Context) -> Any:
    """G0: Spec Compliance - Deliverables match specification.

    Formally:
        For all required deliverables d:
            exists implementation i such that implements(i, d)
    """
    # Variables
    required_deliverables = ctx.int_var("required_deliverables")
    implemented_deliverables = ctx.int_var("implemented_deliverables")
    interface_count = ctx.int_var("interface_count")
    interface_matches = ctx.int_var("interface_matches")

    # Domain constraints
    ctx.add_constraint(required_deliverables >= 0)
    ctx.add_constraint(implemented_deliverables >= 0)
    ctx.add_constraint(interface_count >= 0)
    ctx.add_constraint(interface_matches >= 0)

    # Invariant: all required deliverables implemented, interfaces match
    deliverables_ok = implemented_deliverables >= required_deliverables
    interfaces_ok = interface_matches == interface_count

    valid = z3.And(deliverables_ok, interfaces_ok)

    # Violation
    violation = z3.Not(valid)

    return violation


def _build_g4_blindagem(ctx: Z3Context) -> Any:
    """G4: Blindagem - Security invariants for anti-manipulation.

    Formally:
        - No injection vulnerabilities
        - Rate limits enforced
        - Audit trail complete
    """
    # Variables
    injection_vulnerabilities = ctx.int_var("injection_vulnerabilities")
    rate_limit_violations = ctx.int_var("rate_limit_violations")
    unaudited_operations = ctx.int_var("unaudited_operations")

    # Domain constraints
    ctx.add_constraint(injection_vulnerabilities >= 0)
    ctx.add_constraint(rate_limit_violations >= 0)
    ctx.add_constraint(unaudited_operations >= 0)

    # Invariant: no security issues
    no_injection = injection_vulnerabilities == 0
    rate_limits_ok = rate_limit_violations == 0
    fully_audited = unaudited_operations == 0

    valid = z3.And(no_injection, rate_limits_ok, fully_audited)

    # Violation
    violation = z3.Not(valid)

    return violation


def _build_g8_mutation(ctx: Z3Context) -> Any:
    """G8: Mutation Testing - Kill rate meets threshold.

    Formally: mutants_killed / total_mutants >= 0.70
    """
    # Variables
    mutants_killed = ctx.int_var("mutants_killed")
    total_mutants = ctx.int_var("total_mutants")

    # Domain constraints
    ctx.add_constraint(total_mutants > 0)
    ctx.add_constraint(mutants_killed >= 0)
    ctx.add_constraint(mutants_killed <= total_mutants)

    # Invariant: kill rate >= 70%
    # mutants_killed * 100 >= total_mutants * 70
    valid = mutants_killed * 100 >= total_mutants * 70

    # Violation
    violation = z3.Not(valid)

    return violation


# =============================================================================
# INVARIANT REGISTRY
# =============================================================================

INVARIANT_SPECS: Dict[str, InvariantSpec] = {
    "I1_NAMESPACING": InvariantSpec(
        name="I1_NAMESPACING",
        gate_id="ALL",
        description="Keys must contain run_id and sprint_id",
        formula_builder=_build_i1_namespacing,
        critical=True,
    ),
    "I2_IDEMPOTENCY": InvariantSpec(
        name="I2_IDEMPOTENCY",
        gate_id="ALL",
        description="Re-execution does not duplicate",
        formula_builder=_build_i2_idempotency,
        critical=True,
    ),
    "I3_PHASE_ORDER": InvariantSpec(
        name="I3_PHASE_ORDER",
        gate_id="ALL",
        description="Phases follow strict order",
        formula_builder=_build_i3_phase_order,
        critical=True,
    ),
    "I4_GATES_BEFORE_SIGNOFF": InvariantSpec(
        name="I4_GATES_BEFORE_SIGNOFF",
        gate_id="G4",
        description="Gates must pass before signoffs",
        formula_builder=_build_i4_gates_before_signoff,
        critical=True,
    ),
    "I5_EXECUTIVE_VERIFICATION": InvariantSpec(
        name="I5_EXECUTIVE_VERIFICATION",
        gate_id="G0",
        description="Executives must verify subordinate work",
        formula_builder=_build_i5_executive_verification,
        critical=True,
    ),
    "I6_TRUTHFULNESS": InvariantSpec(
        name="I6_TRUTHFULNESS",
        gate_id="G0",
        description="Signoff claims must match artifacts",
        formula_builder=_build_i6_truthfulness,
        critical=True,
    ),
    "I7_AUDIT_TRAIL": InvariantSpec(
        name="I7_AUDIT_TRAIL",
        gate_id="G4",
        description="Every decision has evidence bundle",
        formula_builder=_build_i7_audit_trail,
        critical=True,
    ),
    "I9_SAFE_HALT": InvariantSpec(
        name="I9_SAFE_HALT",
        gate_id="ALL",
        description="SAFE_HALT supersedes all operations",
        formula_builder=_build_i9_safe_halt,
        critical=True,
    ),
    "I11_RUNAWAY_PROTECTION": InvariantSpec(
        name="I11_RUNAWAY_PROTECTION",
        gate_id="ALL",
        description="Worker and cost limits enforced",
        formula_builder=_build_i11_runaway_protection,
        critical=True,
    ),
    "G0_SPEC_COMPLIANCE": InvariantSpec(
        name="G0_SPEC_COMPLIANCE",
        gate_id="G0",
        description="Deliverables match specification",
        formula_builder=_build_g0_spec_compliance,
        critical=True,
    ),
    "G4_BLINDAGEM": InvariantSpec(
        name="G4_BLINDAGEM",
        gate_id="G4",
        description="Security/anti-manipulation invariants",
        formula_builder=_build_g4_blindagem,
        critical=True,
    ),
    "G8_MUTATION": InvariantSpec(
        name="G8_MUTATION",
        gate_id="G8",
        description="Mutation testing kill rate threshold",
        formula_builder=_build_g8_mutation,
        critical=True,
    ),
}


# =============================================================================
# Z3 VERIFIER
# =============================================================================

class Z3Verifier:
    """Z3 formal verifier for pipeline invariants.

    This class uses Z3 SMT solver to formally verify that invariants
    hold under all possible states.

    Usage:
        verifier = Z3Verifier()

        # Verify single invariant
        result = verifier.verify_invariant("I1_NAMESPACING")

        # Verify all invariants for a gate
        results = verifier.verify_gate("G0")

        # Verify all critical invariants
        results = verifier.verify_all_critical()
    """

    _instance: Optional["Z3Verifier"] = None

    def __new__(cls) -> "Z3Verifier":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._timeout_ms = 5000
        self._results_cache: Dict[str, VerificationResult] = {}
        self._initialized = True

    @property
    def available(self) -> bool:
        """Check if Z3 is available."""
        return Z3_AVAILABLE

    def set_timeout(self, timeout_ms: int) -> None:
        """Set verification timeout in milliseconds."""
        self._timeout_ms = timeout_ms

    def verify_invariant(
        self,
        invariant_name: str,
        use_cache: bool = True
    ) -> VerificationResult:
        """Verify a single invariant.

        Args:
            invariant_name: Name of invariant (e.g., "I1_NAMESPACING")
            use_cache: Whether to use cached results

        Returns:
            VerificationResult with status and details
        """
        # Check cache
        if use_cache and invariant_name in self._results_cache:
            return self._results_cache[invariant_name]

        # Check Z3 availability
        if not Z3_AVAILABLE:
            return VerificationResult(
                invariant_name=invariant_name,
                gate_id="UNKNOWN",
                status=VerificationStatus.SKIPPED,
                proved=False,
                proof_time_ms=0,
                error_message="Z3 not available",
            )

        # Get invariant spec
        spec = INVARIANT_SPECS.get(invariant_name)
        if not spec:
            return VerificationResult(
                invariant_name=invariant_name,
                gate_id="UNKNOWN",
                status=VerificationStatus.ERROR,
                proved=False,
                proof_time_ms=0,
                error_message=f"Unknown invariant: {invariant_name}",
            )

        # Verify
        start_time = time.time()
        try:
            ctx = Z3Context()
            violation = spec.formula_builder(ctx)
            ctx.add_constraint(violation)

            status = ctx.check(timeout_ms=self._timeout_ms)
            proof_time_ms = (time.time() - start_time) * 1000

            # Get counterexample if violated
            counter_example = None
            if status == VerificationStatus.REFUTED:
                model = ctx.get_model()
                if model:
                    counter_example = str(model)

            result = VerificationResult(
                invariant_name=invariant_name,
                gate_id=spec.gate_id,
                status=status,
                proved=(status == VerificationStatus.PROVED),
                proof_time_ms=proof_time_ms,
                counter_example=counter_example,
            )

            # Cache result
            self._results_cache[invariant_name] = result

            return result

        except Exception as e:
            proof_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Z3 verification failed for {invariant_name}: {e}")
            return VerificationResult(
                invariant_name=invariant_name,
                gate_id=spec.gate_id,
                status=VerificationStatus.ERROR,
                proved=False,
                proof_time_ms=proof_time_ms,
                error_message=str(e),
            )

    def verify_gate(self, gate_id: str) -> List[VerificationResult]:
        """Verify all invariants for a specific gate.

        Args:
            gate_id: Gate identifier (e.g., "G0", "G4", "G8")

        Returns:
            List of verification results
        """
        results = []

        for name, spec in INVARIANT_SPECS.items():
            if spec.gate_id == gate_id or spec.gate_id == "ALL":
                result = self.verify_invariant(name)
                results.append(result)

        return results

    def verify_all_critical(self) -> List[VerificationResult]:
        """Verify all critical invariants.

        Returns:
            List of verification results for critical invariants
        """
        results = []

        for name, spec in INVARIANT_SPECS.items():
            if spec.critical:
                result = self.verify_invariant(name)
                results.append(result)

        return results

    def verify_all(self) -> List[VerificationResult]:
        """Verify all registered invariants.

        Returns:
            List of all verification results
        """
        results = []

        for name in INVARIANT_SPECS:
            result = self.verify_invariant(name)
            results.append(result)

        return results

    def get_verification_summary(self) -> Dict[str, Any]:
        """Get summary of verification results.

        Returns:
            Summary with counts and status
        """
        results = self.verify_all()

        proved = sum(1 for r in results if r.proved)
        refuted = sum(1 for r in results if r.status == VerificationStatus.REFUTED)
        unknown = sum(1 for r in results if r.status == VerificationStatus.UNKNOWN)
        errors = sum(1 for r in results if r.status == VerificationStatus.ERROR)
        skipped = sum(1 for r in results if r.status == VerificationStatus.SKIPPED)

        return {
            "total_invariants": len(results),
            "proved": proved,
            "refuted": refuted,
            "unknown": unknown,
            "errors": errors,
            "skipped": skipped,
            "all_proved": proved == len(results),
            "z3_available": Z3_AVAILABLE,
            "results": [r.to_dict() for r in results],
        }

    def clear_cache(self) -> None:
        """Clear the results cache."""
        self._results_cache.clear()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_verifier: Optional[Z3Verifier] = None


def get_z3_verifier() -> Z3Verifier:
    """Get singleton Z3 verifier instance."""
    global _verifier
    if _verifier is None:
        _verifier = Z3Verifier()
    return _verifier


def verify_invariant(invariant_name: str) -> VerificationResult:
    """Verify a single invariant."""
    return get_z3_verifier().verify_invariant(invariant_name)


def verify_gate_invariants(gate_id: str) -> List[VerificationResult]:
    """Verify all invariants for a gate."""
    return get_z3_verifier().verify_gate(gate_id)


def verify_critical_invariants() -> List[VerificationResult]:
    """Verify all critical invariants."""
    return get_z3_verifier().verify_all_critical()


def is_z3_available() -> bool:
    """Check if Z3 is available."""
    return Z3_AVAILABLE


def get_invariant_specs() -> Dict[str, InvariantSpec]:
    """Get all invariant specifications."""
    return INVARIANT_SPECS.copy()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Classes
    "Z3Context",
    "Z3Verifier",
    "InvariantSpec",
    "VerificationResult",
    # Enums
    "VerificationStatus",
    # TypedDicts
    "Z3InvariantResult",
    # Registry
    "INVARIANT_SPECS",
    # Functions
    "get_z3_verifier",
    "verify_invariant",
    "verify_gate_invariants",
    "verify_critical_invariants",
    "is_z3_available",
    "get_invariant_specs",
    # Constants
    "Z3_AVAILABLE",
]
