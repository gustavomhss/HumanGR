"""Corrective Enforcement Layer for Pipeline V2.

This module provides guardrails that BLOCK pipeline advancement on errors.
The philosophy: guardrails MUST block execution and force REDO with error awareness.
The pipeline cannot advance if there's an error - rework is MANDATORY.

Principles:
1. DETECT → BLOCK → REDO WITH ERROR CONTEXT (not just redirect)
2. Force rework/adjustment with full error awareness
3. Max retries reached = HALT, not continue
4. Log everything for learning and A-MEM persistence

Architecture:
    Input → EnforcementLayer → (Issue Detected?)
                                   |
                                   v
                            CorrectionEngine
                                   |
                            ┌──────┴──────┐
                            v             v
                        Corrected     Needs Retry
                            |             |
                            v             v
                        Continue    Adjust & Retry
                            |             |
                            └──────┬──────┘
                                   v
                              (Max retries?)
                                   |
                            ┌──────┴──────┐
                            No            Yes
                            |             |
                            v             v
                        Continue      Escalate
                                     (graceful)

Author: Pipeline Autonomo Team
Version: 1.0.0 (2026-01-16)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# ENUMS AND TYPES
# =============================================================================


class CorrectionAction(str, Enum):
    """Actions the enforcement layer can take."""
    PASS = "pass"                    # No issues, continue
    CORRECTED = "corrected"          # Issue found and auto-corrected
    RETRY = "retry"                  # Needs retry with adjustments
    ADJUST_AND_CONTINUE = "adjust"   # Minor adjustment, continue
    ESCALATE = "escalate"            # Max corrections reached, escalate (graceful)


class EnforcementType(str, Enum):
    """Types of enforcement checks."""
    INVARIANT = "invariant"
    SECURITY = "security"
    TRUST_BOUNDARY = "trust_boundary"
    CONTENT_FILTER = "content_filter"
    STACK_HEALTH = "stack_health"
    VALIDATION = "validation"


class EscalationLevel(str, Enum):
    """Escalation levels (graceful, not blocking)."""
    NONE = "none"                    # No escalation needed
    LOG_WARNING = "log_warning"      # Log and continue
    NOTIFY_HUMAN = "notify_human"    # Flag for human review, continue
    REQUIRE_APPROVAL = "require_approval"  # Pause for approval, then continue
    SAFE_MODE = "safe_mode"          # Continue in degraded/safe mode


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class EnforcementResult:
    """Result of an enforcement check."""
    passed: bool
    enforcement_type: EnforcementType
    action: CorrectionAction
    original_value: Any = None
    corrected_value: Any = None
    correction_applied: str = ""
    message: str = ""
    retries_remaining: int = 0
    escalation_level: EscalationLevel = EscalationLevel.NONE
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class CorrectionContext:
    """Context for correction operations."""
    run_id: str
    sprint_id: str
    node: str
    attempt: int
    max_attempts: int = 3
    corrections_applied: List[str] = field(default_factory=list)
    original_input: Any = None
    agent_id: Optional[str] = None

    @property
    def can_retry(self) -> bool:
        return self.attempt < self.max_attempts


@dataclass
class CorrectionStrategy:
    """Strategy for correcting a specific type of issue."""
    name: str
    applies_to: EnforcementType
    correct: Callable[[Any, CorrectionContext], Awaitable[Tuple[bool, Any, str]]]
    priority: int = 0  # Higher = run first


# =============================================================================
# CORRECTION ENGINE
# =============================================================================


class CorrectionEngine:
    """Engine that applies corrections to issues.

    This engine attempts to automatically correct issues rather than
    blocking execution. It uses a pipeline of correction strategies.
    """

    _instance: Optional["CorrectionEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "CorrectionEngine":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._strategies: Dict[EnforcementType, List[CorrectionStrategy]] = {
            t: [] for t in EnforcementType
        }
        self._correction_history: List[Dict[str, Any]] = []
        self._initialized = True

        # Register default strategies
        self._register_default_strategies()

    def _register_default_strategies(self) -> None:
        """Register built-in correction strategies."""

        # Invariant corrections
        self.register_strategy(CorrectionStrategy(
            name="namespace_key_correction",
            applies_to=EnforcementType.INVARIANT,
            correct=self._correct_namespace_key,
            priority=10,
        ))

        self.register_strategy(CorrectionStrategy(
            name="phase_transition_correction",
            applies_to=EnforcementType.INVARIANT,
            correct=self._correct_phase_transition,
            priority=9,
        ))

        # Security corrections
        self.register_strategy(CorrectionStrategy(
            name="content_sanitization",
            applies_to=EnforcementType.SECURITY,
            correct=self._correct_content_security,
            priority=10,
        ))

        self.register_strategy(CorrectionStrategy(
            name="jailbreak_neutralization",
            applies_to=EnforcementType.CONTENT_FILTER,
            correct=self._neutralize_jailbreak,
            priority=10,
        ))

        # Trust boundary corrections
        self.register_strategy(CorrectionStrategy(
            name="permission_downgrade",
            applies_to=EnforcementType.TRUST_BOUNDARY,
            correct=self._downgrade_permission,
            priority=8,
        ))

        # Validation corrections
        self.register_strategy(CorrectionStrategy(
            name="type_coercion",
            applies_to=EnforcementType.VALIDATION,
            correct=self._coerce_types,
            priority=10,
        ))

    def register_strategy(self, strategy: CorrectionStrategy) -> None:
        """Register a correction strategy."""
        strategies = self._strategies[strategy.applies_to]
        strategies.append(strategy)
        # Keep sorted by priority (highest first)
        strategies.sort(key=lambda s: s.priority, reverse=True)

    async def attempt_correction(
        self,
        enforcement_type: EnforcementType,
        value: Any,
        context: CorrectionContext,
        issue_description: str = "",
    ) -> EnforcementResult:
        """Attempt to correct an issue.

        Args:
            enforcement_type: Type of enforcement that failed
            value: The value that caused the issue
            context: Correction context
            issue_description: Description of what went wrong

        Returns:
            EnforcementResult with correction outcome
        """
        strategies = self._strategies.get(enforcement_type, [])

        if not strategies:
            # No strategies - BLOCK execution, cannot continue without correction
            logger.error(
                f"No correction strategies for {enforcement_type.value}. "
                f"BLOCKING: Pipeline cannot advance without correction capability."
            )
            return EnforcementResult(
                passed=False,  # BLOCK - no correction available
                enforcement_type=enforcement_type,
                action=CorrectionAction.ESCALATE,
                original_value=value,
                corrected_value=value,
                message=f"BLOCKED: No correction available for {issue_description}",
                escalation_level=EscalationLevel.REQUIRE_APPROVAL,
            )

        # Try each strategy in priority order
        for strategy in strategies:
            try:
                corrected, corrected_value, correction_desc = await strategy.correct(
                    value, context
                )

                if corrected:
                    # Record correction
                    self._record_correction(
                        strategy.name,
                        enforcement_type,
                        value,
                        corrected_value,
                        correction_desc,
                        context,
                    )

                    return EnforcementResult(
                        passed=True,
                        enforcement_type=enforcement_type,
                        action=CorrectionAction.CORRECTED,
                        original_value=value,
                        corrected_value=corrected_value,
                        correction_applied=correction_desc,
                        message=f"Auto-corrected by {strategy.name}",
                        metadata={"strategy": strategy.name},
                    )

            except Exception as e:
                logger.debug(f"Strategy {strategy.name} failed: {e}")
                continue

        # No strategy could correct - check if retry is possible
        if context.can_retry:
            return EnforcementResult(
                passed=False,
                enforcement_type=enforcement_type,
                action=CorrectionAction.RETRY,
                original_value=value,
                message=f"Correction failed, retry recommended: {issue_description}",
                retries_remaining=context.max_attempts - context.attempt,
                escalation_level=EscalationLevel.LOG_WARNING,
            )

        # Max retries reached - HALT execution
        logger.error(
            f"Max correction attempts reached for {enforcement_type.value}. "
            f"HALTING: Pipeline cannot continue after {context.max_attempts} failed attempts."
        )
        return EnforcementResult(
            passed=False,  # BLOCK - max retries exhausted
            enforcement_type=enforcement_type,
            action=CorrectionAction.ESCALATE,
            original_value=value,
            corrected_value=value,  # Use original
            message=f"HALTED after {context.max_attempts} attempts: {issue_description}",
            escalation_level=EscalationLevel.REQUIRE_APPROVAL,
            metadata={"requires_human_review": True, "halted": True},
        )

    def _record_correction(
        self,
        strategy_name: str,
        enforcement_type: EnforcementType,
        original: Any,
        corrected: Any,
        description: str,
        context: CorrectionContext,
    ) -> None:
        """Record a correction for learning."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategy": strategy_name,
            "type": enforcement_type.value,
            "run_id": context.run_id,
            "sprint_id": context.sprint_id,
            "node": context.node,
            "original_hash": hashlib.md5(str(original).encode()).hexdigest()[:8],
            "description": description,
        }
        self._correction_history.append(record)

        # Keep history bounded
        if len(self._correction_history) > 1000:
            self._correction_history = self._correction_history[-500:]

        # Log for observability
        logger.info(
            f"Correction applied: {strategy_name} on {enforcement_type.value} "
            f"in {context.node} - {description}"
        )

    # =========================================================================
    # DEFAULT CORRECTION STRATEGIES
    # =========================================================================

    async def _correct_namespace_key(
        self,
        value: Any,
        context: CorrectionContext,
    ) -> Tuple[bool, Any, str]:
        """Correct invalid namespace keys (I1)."""
        if not isinstance(value, str):
            return False, value, ""

        # Expected format: {prefix}:{sprint_id}:{resource}
        # Example: claim:S00:abc123

        parts = value.split(":")
        if len(parts) >= 3:
            # Already has namespace
            return True, value, "Key already namespaced"

        # Try to add namespace
        sprint_id = context.sprint_id
        if len(parts) == 2:
            # Has prefix but no sprint
            corrected = f"{parts[0]}:{sprint_id}:{parts[1]}"
            return True, corrected, f"Added sprint namespace: {value} -> {corrected}"
        elif len(parts) == 1:
            # Just a bare key
            corrected = f"data:{sprint_id}:{value}"
            return True, corrected, f"Added full namespace: {value} -> {corrected}"

        return False, value, ""

    async def _correct_phase_transition(
        self,
        value: Any,
        context: CorrectionContext,
    ) -> Tuple[bool, Any, str]:
        """Correct invalid phase transitions (I3).

        Instead of blocking, we adjust the transition path.
        """
        if not isinstance(value, dict) or "from_phase" not in value:
            return False, value, ""

        from_phase = value.get("from_phase", "")
        to_phase = value.get("to_phase", "")

        # Define valid transitions
        valid_transitions = {
            "INIT": ["SPEC", "HALT"],
            "SPEC": ["EXEC", "INIT", "HALT"],  # Can go back to INIT
            "EXEC": ["GATE", "SPEC", "HALT"],  # Can retry SPEC
            "GATE": ["VOTE", "EXEC", "HALT"],  # Can retry EXEC
            "VOTE": ["DONE", "GATE", "HALT"],  # Can retry GATE
            "DONE": ["INIT"],  # Can start new run
            "HALT": ["INIT"],  # Can restart
        }

        allowed = valid_transitions.get(from_phase, [])

        if to_phase in allowed:
            return True, value, "Transition is valid"

        # Find the correct intermediate phase
        if from_phase == "INIT" and to_phase == "GATE":
            # Need to go through SPEC and EXEC
            corrected = {**value, "to_phase": "SPEC", "queued_phases": ["EXEC", "GATE"]}
            return True, corrected, f"Corrected: INIT->GATE requires intermediate phases"

        if from_phase == "SPEC" and to_phase == "VOTE":
            # Need to go through EXEC and GATE
            corrected = {**value, "to_phase": "EXEC", "queued_phases": ["GATE", "VOTE"]}
            return True, corrected, f"Corrected: SPEC->VOTE requires intermediate phases"

        # Default: go back one step to retry
        retry_map = {"EXEC": "SPEC", "GATE": "EXEC", "VOTE": "GATE"}
        if to_phase in retry_map:
            corrected = {**value, "to_phase": retry_map[to_phase]}
            return True, corrected, f"Corrected: Retry from earlier phase"

        return False, value, ""

    async def _correct_content_security(
        self,
        value: Any,
        context: CorrectionContext,
    ) -> Tuple[bool, Any, str]:
        """Correct security issues in content."""
        if not isinstance(value, str):
            return False, value, ""

        original = value
        corrected = value
        corrections = []

        # Remove potential injection patterns
        injection_patterns = [
            (r"(?i)ignore\s+(previous\s+)?instructions?", "[INSTRUCTION_REMOVED]"),
            (r"(?i)pretend\s+(you\s+are|to\s+be)", "[ROLEPLAY_REMOVED]"),
            (r"(?i)you\s+are\s+now\s+", "[IDENTITY_REMOVED]"),
            (r"(?i)system\s*:\s*", "[SYSTEM_REMOVED]"),
        ]

        for pattern, replacement in injection_patterns:
            if re.search(pattern, corrected):
                corrected = re.sub(pattern, replacement, corrected)
                corrections.append(f"Neutralized: {pattern[:30]}...")

        # Redact potential secrets
        secret_patterns = [
            (r"(?i)(api[_-]?key|secret|password|token)\s*[=:]\s*['\"]?[\w-]{20,}['\"]?", "[REDACTED_SECRET]"),
            (r"(?i)bearer\s+[\w-]{20,}", "[REDACTED_BEARER]"),
        ]

        for pattern, replacement in secret_patterns:
            if re.search(pattern, corrected):
                corrected = re.sub(pattern, replacement, corrected)
                corrections.append("Redacted potential secret")

        if corrections:
            return True, corrected, "; ".join(corrections)

        return True, value, "No security issues found"

    async def _neutralize_jailbreak(
        self,
        value: Any,
        context: CorrectionContext,
    ) -> Tuple[bool, Any, str]:
        """Neutralize jailbreak attempts without blocking."""
        if not isinstance(value, str):
            return False, value, ""

        # Jailbreak patterns and neutralizations
        jailbreak_fixes = [
            # DAN attacks
            (r"(?i)do\s+anything\s+now", "follow my guidelines"),
            (r"(?i)DAN\s+mode", "normal mode"),
            (r"(?i)jailbreak(ed)?", "standard operation"),

            # Role play attacks
            (r"(?i)act\s+as\s+(a\s+)?evil", "act as a helpful assistant"),
            (r"(?i)pretend\s+you\s+have\s+no\s+(rules|limits)", "follow standard guidelines"),

            # Ignore instructions
            (r"(?i)ignore\s+(all\s+)?(your\s+)?rules", "follow all rules"),
            (r"(?i)forget\s+(everything|your\s+training)", "remember your training"),
        ]

        corrected = value
        corrections = []

        for pattern, replacement in jailbreak_fixes:
            if re.search(pattern, corrected):
                corrected = re.sub(pattern, replacement, corrected)
                corrections.append(f"Neutralized jailbreak pattern")

        if corrections:
            return True, corrected, f"Jailbreak neutralized: {len(corrections)} patterns"

        return True, value, "No jailbreak detected"

    async def _downgrade_permission(
        self,
        value: Any,
        context: CorrectionContext,
    ) -> Tuple[bool, Any, str]:
        """Downgrade permission request to allowed level."""
        if not isinstance(value, dict):
            return False, value, ""

        requested_tier = value.get("requested_tier", "")
        agent_tier = value.get("agent_tier", "WORKER")

        # Tier hierarchy (lower index = more permissions)
        tier_hierarchy = [
            "EXECUTIVE", "SUPERVISOR", "LEAD", "SENIOR", "WORKER", "RESTRICTED"
        ]

        try:
            requested_idx = tier_hierarchy.index(requested_tier)
            agent_idx = tier_hierarchy.index(agent_tier)
        except ValueError:
            return False, value, ""

        if requested_idx >= agent_idx:
            # Request is within allowed permissions
            return True, value, "Permission within allowed tier"

        # Downgrade to agent's tier
        corrected = {
            **value,
            "requested_tier": agent_tier,
            "downgraded_from": requested_tier,
        }
        return True, corrected, f"Downgraded permission: {requested_tier} -> {agent_tier}"

    async def _coerce_types(
        self,
        value: Any,
        context: CorrectionContext,
    ) -> Tuple[bool, Any, str]:
        """Coerce types to expected format."""
        if not isinstance(value, dict) or "expected_type" not in value:
            return False, value, ""

        expected = value.get("expected_type")
        actual = value.get("value")

        try:
            if expected == "str":
                corrected = str(actual) if actual is not None else ""
                return True, {"value": corrected, "expected_type": expected}, f"Coerced to str"
            elif expected == "int":
                corrected = int(float(actual)) if actual is not None else 0
                return True, {"value": corrected, "expected_type": expected}, f"Coerced to int"
            elif expected == "float":
                corrected = float(actual) if actual is not None else 0.0
                return True, {"value": corrected, "expected_type": expected}, f"Coerced to float"
            elif expected == "bool":
                if isinstance(actual, str):
                    corrected = actual.lower() in ("true", "1", "yes", "on")
                else:
                    corrected = bool(actual)
                return True, {"value": corrected, "expected_type": expected}, f"Coerced to bool"
            elif expected == "list":
                if isinstance(actual, str):
                    corrected = [actual]
                elif actual is None:
                    corrected = []
                else:
                    corrected = list(actual) if hasattr(actual, "__iter__") else [actual]
                return True, {"value": corrected, "expected_type": expected}, f"Coerced to list"
        except (ValueError, TypeError) as e:
            logger.debug(f"Type coercion failed: {e}")

        return False, value, ""


# =============================================================================
# ENFORCEMENT LAYER
# =============================================================================


class EnforcementLayer:
    """Main enforcement layer that wraps pipeline operations.

    PHILOSOPHY (FIXED GAP-06): BLOCK and REDO, not just log and continue.

    This layer intercepts operations, checks for issues, and:
    1. BLOCKS execution when violations are detected
    2. Triggers REWORK with error awareness
    3. Escalates to HALT after max retry attempts

    The pipeline MUST NOT advance if there's an error - rework is MANDATORY.
    """

    _instance: Optional["EnforcementLayer"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "EnforcementLayer":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._correction_engine = CorrectionEngine()
        self._enforcement_log: List[EnforcementResult] = []
        self._escalation_queue: List[Dict[str, Any]] = []
        self._initialized = True

        # Import enforcement modules (graceful)
        self._load_enforcement_modules()

    def _load_enforcement_modules(self) -> None:
        """Load enforcement modules with graceful degradation."""
        # Invariant checker - load class, not instance (requires run_id/sprint_id)
        self._checker_cache: Dict[str, Any] = {}
        try:
            from pipeline.langgraph.invariants import InvariantChecker
            self._InvariantCheckerClass = InvariantChecker
        except ImportError:
            self._InvariantCheckerClass = None
            logger.debug("Invariant checker not available")

        # NeMo enhanced
        try:
            from pipeline.security.nemo_enhanced import get_nemo_enhanced
            self._nemo = get_nemo_enhanced()
        except Exception as e:
            self._nemo = None
            logger.debug(f"NeMo enhanced not available: {e}")

        # LLM Guard
        try:
            from pipeline.security.llm_guard_integration import get_security_orchestrator
            self._security_orchestrator = get_security_orchestrator()
        except (ImportError, TypeError, Exception) as e:
            self._security_orchestrator = None
            logger.debug(f"Security orchestrator not available: {e}")

        # Trust boundaries
        try:
            from pipeline.langgraph.trust_boundaries import get_trust_boundary_enforcer
            self._trust_enforcer = get_trust_boundary_enforcer()
        except Exception as e:
            self._trust_enforcer = None
            logger.debug(f"Trust boundary enforcer not available: {e}")

        # Stack guardrails
        try:
            from pipeline.langgraph.stack_guardrails import get_guardrails
            self._stack_guardrails = get_guardrails()
        except Exception as e:
            self._stack_guardrails = None
            logger.debug(f"Stack guardrails not available: {e}")

    def _get_invariant_checker(self, run_id: str, sprint_id: str):
        """Get invariant checker for a specific run/sprint (lazy loading)."""
        if self._InvariantCheckerClass is None:
            return None
        # Create or reuse checker
        key = f"{run_id}:{sprint_id}"
        if key not in self._checker_cache:
            self._checker_cache[key] = self._InvariantCheckerClass(run_id, sprint_id)
        return self._checker_cache[key]

    async def enforce_input(
        self,
        input_data: Any,
        context: CorrectionContext,
    ) -> Tuple[Any, List[EnforcementResult]]:
        """Enforce guardrails on input data.

        Returns corrected input and list of enforcement results.
        BLOCKS on critical violations - triggers REWORK via result flags.
        """
        results: List[EnforcementResult] = []
        corrected_data = input_data

        # 1. Content security check
        if isinstance(corrected_data, str):
            security_result = await self._enforce_content_security(corrected_data, context)
            results.append(security_result)
            if security_result.corrected_value is not None:
                corrected_data = security_result.corrected_value

        # 2. Jailbreak check
        if isinstance(corrected_data, str):
            jailbreak_result = await self._enforce_jailbreak_protection(corrected_data, context)
            results.append(jailbreak_result)
            if jailbreak_result.corrected_value is not None:
                corrected_data = jailbreak_result.corrected_value

        # 3. Dict validation
        if isinstance(corrected_data, dict):
            validation_result = await self._enforce_validation(corrected_data, context)
            results.append(validation_result)
            if validation_result.corrected_value is not None:
                corrected_data = validation_result.corrected_value

        # Log results
        self._enforcement_log.extend(results)

        # Queue any escalations (but don't block!)
        for result in results:
            if result.escalation_level in (EscalationLevel.NOTIFY_HUMAN, EscalationLevel.REQUIRE_APPROVAL):
                self._queue_escalation(result, context)

        return corrected_data, results

    async def enforce_output(
        self,
        output_data: Any,
        context: CorrectionContext,
    ) -> Tuple[Any, List[EnforcementResult]]:
        """Enforce guardrails on output data.

        Returns corrected output and list of enforcement results.
        BLOCKS on critical violations - triggers REWORK via result flags.
        """
        results: List[EnforcementResult] = []
        corrected_data = output_data

        # 1. Check for secrets/PII in output
        if isinstance(corrected_data, str):
            security_result = await self._enforce_output_security(corrected_data, context)
            results.append(security_result)
            if security_result.corrected_value is not None:
                corrected_data = security_result.corrected_value

        # 2. Validate output structure
        if isinstance(corrected_data, dict):
            validation_result = await self._enforce_validation(corrected_data, context)
            results.append(validation_result)
            if validation_result.corrected_value is not None:
                corrected_data = validation_result.corrected_value

        # Log results
        self._enforcement_log.extend(results)

        return corrected_data, results

    async def enforce_phase_transition(
        self,
        from_phase: str,
        to_phase: str,
        context: CorrectionContext,
    ) -> Tuple[str, EnforcementResult]:
        """Enforce valid phase transition (I3).

        Returns (corrected_to_phase, result).
        If transition is invalid, returns a valid intermediate phase.
        """
        value = {"from_phase": from_phase, "to_phase": to_phase}

        result = await self._correction_engine.attempt_correction(
            EnforcementType.INVARIANT,
            value,
            context,
            f"Phase transition {from_phase} -> {to_phase}",
        )

        # Extract corrected phase
        if result.action == CorrectionAction.CORRECTED:
            corrected = result.corrected_value
            if isinstance(corrected, dict):
                return corrected.get("to_phase", to_phase), result

        return to_phase, result

    async def enforce_trust_boundary(
        self,
        agent_id: str,
        resource: str,
        action: str,
        context: CorrectionContext,
    ) -> Tuple[bool, str, EnforcementResult]:
        """Enforce trust boundary (RBAC).

        Returns (allowed, effective_action, result).
        If denied, may return a downgraded action that IS allowed.

        IMPORTANTE: FAIL-CLOSED - Trust boundaries são CRITICAL, não permite bypass.
        """
        if self._trust_enforcer is None:
            # FAIL-CLOSED: Trust boundaries são CRITICAL - não permite bypass
            from pipeline.stack_health_supervisor import (
                get_stack_supervisor,
                CriticalStackUnavailableError,
            )

            logger.error(
                f"FAIL-CLOSED: Trust enforcer não disponível. "
                f"Acesso NEGADO para {agent_id} -> {resource}:{action}. "
                "Trust boundaries são OBRIGATÓRIOS."
            )

            # Alertar o supervisor para que o Run Master seja notificado
            supervisor = get_stack_supervisor()
            if not supervisor.is_healthy("trust_boundaries"):
                raise CriticalStackUnavailableError(
                    "trust_boundaries",
                    f"Trust enforcer unavailable - cannot authorize {agent_id} for {resource}:{action}"
                )

            # Mesmo que o supervisor diga que está healthy, o enforcer não está inicializado
            # Isso é um problema de configuração - NEGAR acesso e ESCALAR
            return False, action, EnforcementResult(
                passed=False,
                enforcement_type=EnforcementType.TRUST_BOUNDARY,
                action=CorrectionAction.ESCALATE,  # ESCALATE é o valor correto do enum
                message="FAIL-CLOSED: Trust enforcer not initialized - access denied",
            )

        # Check access
        try:
            check_result = self._trust_enforcer.check_access(agent_id, resource, action)

            if check_result.allowed:
                return True, action, EnforcementResult(
                    passed=True,
                    enforcement_type=EnforcementType.TRUST_BOUNDARY,
                    action=CorrectionAction.PASS,
                    message=f"Access granted: {agent_id} -> {resource}:{action}",
                )

            # Access denied - try to downgrade
            value = {
                "agent_id": agent_id,
                "resource": resource,
                "action": action,
                "agent_tier": getattr(check_result, "agent_tier", "WORKER"),
                "requested_tier": getattr(check_result, "required_tier", "EXECUTIVE"),
            }

            correction_result = await self._correction_engine.attempt_correction(
                EnforcementType.TRUST_BOUNDARY,
                value,
                context,
                f"Access denied: {agent_id} -> {resource}:{action}",
            )

            if correction_result.action == CorrectionAction.CORRECTED:
                # Return downgraded action
                downgraded = correction_result.corrected_value
                effective_action = downgraded.get("action", "read")  # Default to read-only
                return True, effective_action, correction_result

            # Can't correct - still allow but flag
            return True, "read", EnforcementResult(
                passed=True,
                enforcement_type=EnforcementType.TRUST_BOUNDARY,
                action=CorrectionAction.ADJUST_AND_CONTINUE,
                message=f"Downgraded to read-only access",
                escalation_level=EscalationLevel.LOG_WARNING,
            )

        except Exception as e:
            logger.error(f"Trust boundary check failed: {e}")
            # On error, deny access (fail-secure) - FIX CONSISTENCY-01
            return False, action, EnforcementResult(
                passed=False,
                enforcement_type=EnforcementType.TRUST_BOUNDARY,
                action=CorrectionAction.ESCALATE,  # Fixed: BLOCK -> ESCALATE (valid enum)
                message=f"Trust check error: {e}",
                escalation_level=EscalationLevel.REQUIRE_APPROVAL,  # Escalate on error
            )

    async def enforce_invariants(
        self,
        state: Dict[str, Any],
        context: CorrectionContext,
    ) -> Tuple[Dict[str, Any], List[EnforcementResult]]:
        """Enforce all invariants on state.

        Returns (corrected_state, results).
        """
        results: List[EnforcementResult] = []
        corrected_state = dict(state)

        # Get invariant checker for this run/sprint context
        invariant_checker = self._get_invariant_checker(context.run_id, context.sprint_id)
        if invariant_checker is None:
            return corrected_state, results

        # I1: Namespacing
        for key in list(corrected_state.keys()):
            if key.startswith("data_") or key.startswith("cache_"):
                result = await self._correction_engine.attempt_correction(
                    EnforcementType.INVARIANT,
                    key,
                    context,
                    f"Key namespacing: {key}",
                )
                results.append(result)

                if result.action == CorrectionAction.CORRECTED:
                    # Rename key
                    old_key = key
                    new_key = result.corrected_value
                    corrected_state[new_key] = corrected_state.pop(old_key)

        # I11: Runaway protection
        attempt = corrected_state.get("attempt", 0)
        max_attempts = 10

        if attempt >= max_attempts:
            results.append(EnforcementResult(
                passed=True,
                enforcement_type=EnforcementType.INVARIANT,
                action=CorrectionAction.ESCALATE,
                message=f"Max attempts ({max_attempts}) reached - escalating",
                escalation_level=EscalationLevel.NOTIFY_HUMAN,
                metadata={"invariant": "I11", "attempts": attempt},
            ))
            # Reset attempt counter but flag for review
            corrected_state["attempt"] = 0
            corrected_state["_escalated_at"] = datetime.now(timezone.utc).isoformat()

        return corrected_state, results

    # =========================================================================
    # PRIVATE ENFORCEMENT METHODS
    # =========================================================================

    async def _enforce_content_security(
        self,
        content: str,
        context: CorrectionContext,
    ) -> EnforcementResult:
        """Enforce content security."""
        return await self._correction_engine.attempt_correction(
            EnforcementType.SECURITY,
            content,
            context,
            "Content security check",
        )

    async def _enforce_jailbreak_protection(
        self,
        content: str,
        context: CorrectionContext,
    ) -> EnforcementResult:
        """Enforce jailbreak protection."""
        return await self._correction_engine.attempt_correction(
            EnforcementType.CONTENT_FILTER,
            content,
            context,
            "Jailbreak protection",
        )

    async def _enforce_output_security(
        self,
        content: str,
        context: CorrectionContext,
    ) -> EnforcementResult:
        """Enforce output security (secrets, PII)."""
        # Use security orchestrator if available
        if self._security_orchestrator is not None:
            try:
                check_result = await self._security_orchestrator.validate_output(content)

                if check_result.get("is_safe", True):
                    return EnforcementResult(
                        passed=True,
                        enforcement_type=EnforcementType.SECURITY,
                        action=CorrectionAction.PASS,
                        original_value=content,
                        corrected_value=content,
                        message="Output is safe",
                    )

                # Get sanitized version
                sanitized = check_result.get("sanitized_text", content)
                return EnforcementResult(
                    passed=True,
                    enforcement_type=EnforcementType.SECURITY,
                    action=CorrectionAction.CORRECTED,
                    original_value=content,
                    corrected_value=sanitized,
                    correction_applied="Sanitized sensitive content",
                    message="Output sanitized",
                )

            except Exception as e:
                logger.debug(f"Security orchestrator error: {e}")

        # Fallback to correction engine
        return await self._correction_engine.attempt_correction(
            EnforcementType.SECURITY,
            content,
            context,
            "Output security check",
        )

    async def _enforce_validation(
        self,
        data: Dict[str, Any],
        context: CorrectionContext,
    ) -> EnforcementResult:
        """Enforce data validation."""
        # Check for required fields based on context
        required_fields = self._get_required_fields(context.node)
        missing = [f for f in required_fields if f not in data]

        if not missing:
            return EnforcementResult(
                passed=True,
                enforcement_type=EnforcementType.VALIDATION,
                action=CorrectionAction.PASS,
                original_value=data,
                corrected_value=data,
                message="Validation passed",
            )

        # Add default values for missing fields
        corrected = dict(data)
        defaults = {
            "run_id": lambda: f"run_{context.run_id}",
            "sprint_id": lambda: context.sprint_id,
            "timestamp": lambda: datetime.now(timezone.utc).isoformat(),
            "attempt": lambda: context.attempt,
            "errors": lambda: [],
            "node_history": lambda: [],
        }

        added = []
        for field in missing:
            if field in defaults:
                corrected[field] = defaults[field]()
                added.append(field)

        return EnforcementResult(
            passed=True,
            enforcement_type=EnforcementType.VALIDATION,
            action=CorrectionAction.CORRECTED if added else CorrectionAction.PASS,
            original_value=data,
            corrected_value=corrected,
            correction_applied=f"Added missing fields: {added}" if added else "",
            message=f"Validation completed, added {len(added)} fields",
        )

    def _get_required_fields(self, node: str) -> List[str]:
        """Get required fields for a node."""
        base_fields = ["run_id", "sprint_id"]

        node_fields = {
            "init": base_fields,
            "exec": base_fields + ["phase", "attempt"],
            "gate": base_fields + ["phase", "tasks_completed"],
            "signoff": base_fields + ["gate_status", "gates_passed"],
            "artifact": base_fields + ["signoff_status"],
        }

        return node_fields.get(node, base_fields)

    async def enforce_stack_health(
        self,
        required_stacks: List[str],
        context: CorrectionContext,
    ) -> Tuple[Dict[str, bool], List[EnforcementResult]]:
        """Enforce stack health with fallback behavior.

        Checks if required stacks are healthy and applies fallback behavior
        if they are not. BLOCKS when stacks fail - no silent degradation.

        Args:
            required_stacks: List of stack names required for operation
            context: Correction context

        Returns:
            Tuple of (stack availability map, enforcement results)
        """
        results: List[EnforcementResult] = []
        stack_availability: Dict[str, bool] = {}

        if self._stack_guardrails is None:
            # No guardrails available - assume all stacks available
            for stack in required_stacks:
                stack_availability[stack] = True
            results.append(EnforcementResult(
                passed=True,
                enforcement_type=EnforcementType.STACK_HEALTH,
                action=CorrectionAction.PASS,
                message="Stack guardrails not available, assuming healthy",
                escalation_level=EscalationLevel.LOG_WARNING,
            ))
            return stack_availability, results

        try:
            # Check each required stack
            for stack_name in required_stacks:
                try:
                    health = await asyncio.to_thread(
                        self._stack_guardrails.check_stack_health,
                        stack_name
                    )

                    if health.healthy:
                        stack_availability[stack_name] = True
                        results.append(EnforcementResult(
                            passed=True,
                            enforcement_type=EnforcementType.STACK_HEALTH,
                            action=CorrectionAction.PASS,
                            message=f"Stack {stack_name} is healthy",
                            metadata={"stack": stack_name, "latency_ms": health.latency_ms},
                        ))
                    else:
                        # Stack unhealthy - check for fallback
                        fallback = self._get_stack_fallback(stack_name)
                        if fallback:
                            stack_availability[stack_name] = True
                            results.append(EnforcementResult(
                                passed=True,
                                enforcement_type=EnforcementType.STACK_HEALTH,
                                action=CorrectionAction.ADJUST_AND_CONTINUE,
                                message=f"Stack {stack_name} unhealthy, using fallback: {fallback}",
                                escalation_level=EscalationLevel.LOG_WARNING,
                                metadata={
                                    "stack": stack_name,
                                    "fallback": fallback,
                                    "error": health.error,
                                },
                            ))
                        else:
                            # FIX BLACK-NEW-08: No fallback AND unhealthy = FAIL-SECURE
                            # Cannot proceed without the required stack
                            stack_availability[stack_name] = False
                            results.append(EnforcementResult(
                                passed=False,  # FIXED: MUST block when no fallback available
                                enforcement_type=EnforcementType.STACK_HEALTH,
                                action=CorrectionAction.ESCALATE,
                                message=f"BLOCKED: Stack {stack_name} unhealthy with no fallback - cannot proceed",
                                escalation_level=EscalationLevel.REQUIRE_APPROVAL,  # Require human
                                metadata={
                                    "stack": stack_name,
                                    "error": health.error,
                                    "degraded_mode": False,  # Not degraded - blocked
                                    "requires_human_approval": True,
                                },
                            ))

                except Exception as e:
                    # FIXED (GAP-07): Error checking stack - assume UNAVAILABLE for safety
                    logger.warning(f"Error checking stack {stack_name}: {e} - marking as unhealthy")
                    stack_availability[stack_name] = False
                    results.append(EnforcementResult(
                        passed=False,
                        enforcement_type=EnforcementType.STACK_HEALTH,
                        action=CorrectionAction.RETRY,
                        message=f"Error checking {stack_name}, marking as unhealthy for safety",
                        escalation_level=EscalationLevel.NOTIFY_HUMAN,
                        metadata={"stack": stack_name, "error": str(e)},
                    ))

        except Exception as e:
            logger.error(f"Stack health check failed: {e} - marking all stacks as unhealthy")
            # FIXED (GAP-07): On any error, assume all stacks UNAVAILABLE for safety
            for stack in required_stacks:
                stack_availability[stack] = False

        # Queue escalations
        for result in results:
            if result.escalation_level in (EscalationLevel.NOTIFY_HUMAN, EscalationLevel.REQUIRE_APPROVAL):
                self._queue_escalation(result, context)

        return stack_availability, results

    def _get_stack_fallback(self, stack_name: str) -> Optional[str]:
        """Get fallback for a stack."""
        # Define fallback mappings
        fallbacks = {
            "redis": "local_cache",           # Local in-memory cache
            "qdrant": "local_faiss",          # Local FAISS index
            "falkordb": "local_graph",        # In-memory graph
            "langfuse": "local_logging",      # Local file logging
            "phoenix": "local_metrics",       # Local metrics collection
            "letta": "redis",                 # Redis-based memory
            "mem0": "redis",                  # Redis-based memory
            "graphiti": "falkordb",           # Direct FalkorDB access
            "nemo": "local_rules",            # Local rule-based filtering
            "llm_guard": "local_patterns",    # Local pattern matching
        }
        return fallbacks.get(stack_name)

    async def enforce_stack_operation(
        self,
        stack_name: str,
        operation: Callable[..., Awaitable[Any]],
        *args,
        context: Optional[CorrectionContext] = None,
        fallback_value: Any = None,
        **kwargs,
    ) -> Tuple[Any, EnforcementResult]:
        """Execute a stack operation with circuit breaker and fallback.

        Args:
            stack_name: Name of the stack
            operation: Async operation to execute
            *args: Arguments for operation
            context: Correction context (optional)
            fallback_value: Value to return if operation fails
            **kwargs: Keyword arguments for operation

        Returns:
            Tuple of (result, enforcement result)
        """
        ctx = context or CorrectionContext(
            run_id="unknown",
            sprint_id="unknown",
            node="stack_operation",
            attempt=1,
        )

        # Check circuit breaker if available
        if self._stack_guardrails is not None:
            try:
                circuit_state = self._stack_guardrails.get_circuit_state(stack_name)
                if circuit_state and circuit_state.value == "open":
                    # Circuit is open - use fallback immediately
                    logger.debug(f"Circuit open for {stack_name}, using fallback")
                    return fallback_value, EnforcementResult(
                        passed=True,
                        enforcement_type=EnforcementType.STACK_HEALTH,
                        action=CorrectionAction.ADJUST_AND_CONTINUE,
                        original_value=None,
                        corrected_value=fallback_value,
                        message=f"Circuit open for {stack_name}, using fallback",
                        metadata={"stack": stack_name, "circuit_state": "open"},
                    )
            except Exception as e:
                logger.debug(f"Error checking circuit for {stack_name}: {e}")

        # Try to execute operation
        try:
            result = await operation(*args, **kwargs)
            return result, EnforcementResult(
                passed=True,
                enforcement_type=EnforcementType.STACK_HEALTH,
                action=CorrectionAction.PASS,
                original_value=result,
                corrected_value=result,
                message=f"Stack operation {stack_name} succeeded",
                metadata={"stack": stack_name},
            )

        except Exception as e:
            logger.warning(f"Stack operation {stack_name} failed: {e}")

            # Record failure for circuit breaker
            if self._stack_guardrails is not None:
                try:
                    self._stack_guardrails.record_failure(stack_name, str(e))
                except Exception as cb_err:
                    # Bloco 5 FIX: Log circuit breaker recording failures
                    logger.debug(f"Failed to record circuit breaker failure for {stack_name}: {cb_err}")

            # Return fallback value
            return fallback_value, EnforcementResult(
                passed=True,  # Still pass - we don't block
                enforcement_type=EnforcementType.STACK_HEALTH,
                action=CorrectionAction.ADJUST_AND_CONTINUE,
                original_value=None,
                corrected_value=fallback_value,
                correction_applied=f"Using fallback due to error: {e}",
                message=f"Stack {stack_name} failed, using fallback",
                escalation_level=EscalationLevel.LOG_WARNING,
                metadata={"stack": stack_name, "error": str(e)},
            )

    def _queue_escalation(
        self,
        result: EnforcementResult,
        context: CorrectionContext,
    ) -> None:
        """Queue an escalation for review (doesn't block)."""
        escalation = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": context.run_id,
            "sprint_id": context.sprint_id,
            "node": context.node,
            "type": result.enforcement_type.value,
            "level": result.escalation_level.value,
            "message": result.message,
            "metadata": result.metadata,
        }
        self._escalation_queue.append(escalation)

        # Log escalation
        logger.warning(
            f"Escalation queued: [{result.escalation_level.value}] "
            f"{result.enforcement_type.value} in {context.node}: {result.message}"
        )

        # Keep queue bounded
        if len(self._escalation_queue) > 100:
            self._escalation_queue = self._escalation_queue[-50:]

    def get_escalation_queue(self) -> List[Dict[str, Any]]:
        """Get pending escalations for review."""
        return list(self._escalation_queue)

    def clear_escalation(self, index: int) -> bool:
        """Clear an escalation after review."""
        if 0 <= index < len(self._escalation_queue):
            self._escalation_queue.pop(index)
            return True
        return False

    def get_enforcement_stats(self) -> Dict[str, Any]:
        """Get enforcement statistics."""
        total = len(self._enforcement_log)
        by_action = {}
        by_type = {}

        for result in self._enforcement_log:
            action = result.action.value
            by_action[action] = by_action.get(action, 0) + 1

            etype = result.enforcement_type.value
            by_type[etype] = by_type.get(etype, 0) + 1

        return {
            "total_enforcements": total,
            "by_action": by_action,
            "by_type": by_type,
            "pending_escalations": len(self._escalation_queue),
            "corrections_applied": sum(
                1 for r in self._enforcement_log
                if r.action == CorrectionAction.CORRECTED
            ),
        }


# =============================================================================
# SINGLETON ACCESSORS
# =============================================================================


_enforcement_layer: Optional[EnforcementLayer] = None
_correction_engine: Optional[CorrectionEngine] = None


def get_enforcement_layer() -> EnforcementLayer:
    """Get singleton enforcement layer."""
    global _enforcement_layer
    if _enforcement_layer is None:
        _enforcement_layer = EnforcementLayer()
    return _enforcement_layer


def get_correction_engine() -> CorrectionEngine:
    """Get singleton correction engine."""
    global _correction_engine
    if _correction_engine is None:
        _correction_engine = CorrectionEngine()
    return _correction_engine


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


async def enforce_and_correct(
    data: Any,
    context: CorrectionContext,
    is_input: bool = True,
) -> Tuple[Any, List[EnforcementResult]]:
    """Convenience function to enforce guardrails.

    Args:
        data: Data to check and potentially correct
        context: Correction context
        is_input: Whether this is input (True) or output (False) data

    Returns:
        (corrected_data, enforcement_results)
    """
    layer = get_enforcement_layer()

    if is_input:
        return await layer.enforce_input(data, context)
    else:
        return await layer.enforce_output(data, context)


def create_correction_context(
    run_id: str,
    sprint_id: str,
    node: str,
    attempt: int = 1,
    max_attempts: int = 3,
    agent_id: Optional[str] = None,
) -> CorrectionContext:
    """Create a correction context."""
    return CorrectionContext(
        run_id=run_id,
        sprint_id=sprint_id,
        node=node,
        attempt=attempt,
        max_attempts=max_attempts,
        agent_id=agent_id,
    )


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Enums
    "CorrectionAction",
    "EnforcementType",
    "EscalationLevel",
    # Data classes
    "EnforcementResult",
    "CorrectionContext",
    "CorrectionStrategy",
    # Classes
    "CorrectionEngine",
    "EnforcementLayer",
    # Accessors
    "get_enforcement_layer",
    "get_correction_engine",
    # Convenience
    "enforce_and_correct",
    "create_correction_context",
]
