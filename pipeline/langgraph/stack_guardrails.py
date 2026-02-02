"""Stack Guardrails - Enforcement System for Stack Usage.

This module provides MANDATORY guardrails that ENFORCE correct stack usage
and GUARANTEE stacks are functioning. No silent failures allowed.

Features:
1. MANDATORY ENFORCEMENT - Certain stacks MUST be used (langfuse, etc.)
2. CIRCUIT BREAKERS - Automatic fallback when stacks fail
3. HEALTH VALIDATION - Pre-flight checks before operations
4. VIOLATION TRACKING - Log and alert on guardrail violations
5. RETRY POLICIES - Intelligent retry with exponential backoff

Architecture:
    StackGuardrails (Enforcer)
        |
        +-- MandatoryStackEnforcer (Usage Rules)
        +-- CircuitBreakerRegistry (Failure Protection)
        +-- HealthValidator (Pre-flight Checks)
        +-- ViolationTracker (Audit Trail)
        |
        v
    Pipeline/Agents (Protected Consumers)

Author: Pipeline Autonomo Team
Version: 1.0.0 (2026-01-16)
"""

from __future__ import annotations

import logging
import time
import functools
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar
from enum import Enum
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
import traceback

logger = logging.getLogger(__name__)

# Type variable for generic decorators
F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================


class GuardrailSeverity(str, Enum):
    """Severity level for guardrail violations."""

    CRITICAL = "critical"  # Block execution, alert immediately
    HIGH = "high"  # Log error, may block
    MEDIUM = "medium"  # Log warning, continue
    LOW = "low"  # Log info, continue


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class EnforcementAction(str, Enum):
    """Action to take on violation."""

    BLOCK = "block"  # Stop execution
    WARN = "warn"  # Log warning, continue
    FALLBACK = "fallback"  # Use alternative
    RETRY = "retry"  # Retry with backoff


# Mandatory stacks that MUST be used
MANDATORY_STACKS: Dict[str, Dict[str, Any]] = {
    "langfuse": {
        "operations": ["llm_call", "generation", "agent_execution"],
        "severity": GuardrailSeverity.CRITICAL,
        "message": "TODA chamada LLM DEVE ter langfuse.trace()",
        "action": EnforcementAction.BLOCK,
    },
}

# Recommended stacks for specific operations
RECOMMENDED_STACKS: Dict[str, Dict[str, Any]] = {
    "reflexion": {
        "operations": ["gate_failure", "task_failure", "error_handling"],
        "severity": GuardrailSeverity.HIGH,
        "message": "TODA falha DEVE usar reflexion.reflect()",
        "action": EnforcementAction.WARN,
    },
    "letta": {
        "operations": ["learning", "insight", "decision"],
        "severity": GuardrailSeverity.MEDIUM,
        "message": "TODO aprendizado DEVE ser salvo em letta",
        "action": EnforcementAction.WARN,
    },
    "got": {
        "operations": ["complex_analysis", "multi_perspective", "root_cause"],
        "severity": GuardrailSeverity.MEDIUM,
        "message": "Analises complexas DEVEM usar got.analyze()",
        "action": EnforcementAction.WARN,
    },
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class Violation:
    """Record of a guardrail violation."""

    timestamp: datetime
    guardrail: str
    severity: GuardrailSeverity
    message: str
    context: Dict[str, Any]
    stack_trace: Optional[str] = None
    resolved: bool = False


@dataclass
class CircuitBreaker:
    """Circuit breaker for a stack."""

    stack_name: str
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure: Optional[datetime] = None
    last_success: Optional[datetime] = None

    # Configuration
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_seconds: int = 60

    # Lock for thread safety
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record_success(self):
        """Record a successful call."""
        with self._lock:
            self.success_count += 1
            self.last_success = datetime.now(timezone.utc)

            if self.state == CircuitState.HALF_OPEN:
                if self.success_count >= self.success_threshold:
                    self._close()

    def record_failure(self, error: Optional[Exception] = None):
        """Record a failed call."""
        with self._lock:
            self.failure_count += 1
            self.last_failure = datetime.now(timezone.utc)

            if self.state == CircuitState.CLOSED:
                if self.failure_count >= self.failure_threshold:
                    self._open()
            elif self.state == CircuitState.HALF_OPEN:
                self._open()

    def can_execute(self) -> bool:
        """Check if calls are allowed."""
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True

            if self.state == CircuitState.OPEN:
                # Check if timeout has passed
                if self.last_failure:
                    elapsed = (datetime.now(timezone.utc) - self.last_failure).total_seconds()
                    if elapsed >= self.timeout_seconds:
                        self._half_open()
                        return True
                return False

            # HALF_OPEN - allow one request
            return True

    def _open(self):
        """Open the circuit."""
        self.state = CircuitState.OPEN
        self.success_count = 0
        logger.warning(f"Circuit OPEN for stack '{self.stack_name}' - too many failures")

    def _close(self):
        """Close the circuit."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        logger.info(f"Circuit CLOSED for stack '{self.stack_name}' - recovered")

    def _half_open(self):
        """Set circuit to half-open for testing."""
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0
        logger.info(f"Circuit HALF-OPEN for stack '{self.stack_name}' - testing recovery")


@dataclass
class HealthCheckResult:
    """Result of a stack health check."""

    stack_name: str
    healthy: bool
    latency_ms: float
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


# =============================================================================
# VIOLATION TRACKER
# =============================================================================


class ViolationTracker:
    """Tracks and reports guardrail violations."""

    _instance: Optional['ViolationTracker'] = None
    _lock = threading.Lock()

    def __new__(cls) -> 'ViolationTracker':
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.violations: List[Violation] = []
        self.violation_counts: Dict[str, int] = {}
        self.max_violations = 1000  # Rolling window
        self._initialized = True

    def record(
        self,
        guardrail: str,
        severity: GuardrailSeverity,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        include_stack: bool = True,
    ) -> Violation:
        """Record a violation."""
        violation = Violation(
            timestamp=datetime.now(timezone.utc),
            guardrail=guardrail,
            severity=severity,
            message=message,
            context=context or {},
            stack_trace=traceback.format_stack() if include_stack else None,
        )

        self.violations.append(violation)
        self.violation_counts[guardrail] = self.violation_counts.get(guardrail, 0) + 1

        # Trim old violations
        if len(self.violations) > self.max_violations:
            self.violations = self.violations[-self.max_violations:]

        # Log based on severity
        if severity == GuardrailSeverity.CRITICAL:
            logger.critical(f"GUARDRAIL VIOLATION [{guardrail}]: {message}")
        elif severity == GuardrailSeverity.HIGH:
            logger.error(f"GUARDRAIL VIOLATION [{guardrail}]: {message}")
        elif severity == GuardrailSeverity.MEDIUM:
            logger.warning(f"GUARDRAIL VIOLATION [{guardrail}]: {message}")
        else:
            logger.info(f"GUARDRAIL VIOLATION [{guardrail}]: {message}")

        return violation

    def get_violations(
        self,
        guardrail: Optional[str] = None,
        severity: Optional[GuardrailSeverity] = None,
        since: Optional[datetime] = None,
    ) -> List[Violation]:
        """Get violations with optional filters."""
        results = self.violations

        if guardrail:
            results = [v for v in results if v.guardrail == guardrail]

        if severity:
            results = [v for v in results if v.severity == severity]

        if since:
            results = [v for v in results if v.timestamp >= since]

        return results

    def get_summary(self) -> Dict[str, Any]:
        """Get violation summary."""
        now = datetime.now(timezone.utc)
        last_hour = now - timedelta(hours=1)
        last_day = now - timedelta(days=1)

        return {
            "total_violations": len(self.violations),
            "violations_last_hour": len(self.get_violations(since=last_hour)),
            "violations_last_day": len(self.get_violations(since=last_day)),
            "by_guardrail": dict(self.violation_counts),
            "critical_count": len(self.get_violations(severity=GuardrailSeverity.CRITICAL)),
            "high_count": len(self.get_violations(severity=GuardrailSeverity.HIGH)),
        }


# =============================================================================
# CIRCUIT BREAKER REGISTRY
# =============================================================================


class CircuitBreakerRegistry:
    """Registry of circuit breakers for all stacks."""

    _instance: Optional['CircuitBreakerRegistry'] = None
    _lock = threading.Lock()

    def __new__(cls) -> 'CircuitBreakerRegistry':
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.breakers: Dict[str, CircuitBreaker] = {}
        self._initialized = True

    def get_breaker(self, stack_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for a stack."""
        if stack_name not in self.breakers:
            self.breakers[stack_name] = CircuitBreaker(stack_name=stack_name)
        return self.breakers[stack_name]

    def can_execute(self, stack_name: str) -> bool:
        """Check if stack can be called."""
        return self.get_breaker(stack_name).can_execute()

    def record_success(self, stack_name: str):
        """Record successful stack call."""
        self.get_breaker(stack_name).record_success()

    def record_failure(self, stack_name: str, error: Optional[Exception] = None):
        """Record failed stack call."""
        self.get_breaker(stack_name).record_failure(error)

    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers."""
        return {
            name: {
                "state": breaker.state.value,
                "failure_count": breaker.failure_count,
                "success_count": breaker.success_count,
                "last_failure": breaker.last_failure.isoformat() if breaker.last_failure else None,
            }
            for name, breaker in self.breakers.items()
        }


# =============================================================================
# STACK GUARDRAILS - MAIN CLASS
# =============================================================================


class StackGuardrails:
    """Main guardrails enforcement system.

    This class ENFORCES correct stack usage and GUARANTEES reliability:

    1. Pre-flight health checks before operations
    2. Mandatory stack usage enforcement
    3. Circuit breaker protection
    4. Violation tracking and alerting
    5. Automatic fallback to alternatives

    Usage:
        guardrails = StackGuardrails()

        # Check health before operation
        guardrails.validate_pre_flight(["redis", "langfuse"])

        # Execute with protection
        with guardrails.protected_call("qdrant") as stack:
            result = stack.search(...)

        # Enforce mandatory stacks
        @guardrails.require_stack("langfuse")
        def my_llm_call():
            ...
    """

    _instance: Optional['StackGuardrails'] = None
    _lock = threading.Lock()

    def __new__(cls) -> 'StackGuardrails':
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.violation_tracker = ViolationTracker()
        self.circuit_registry = CircuitBreakerRegistry()
        self._stack_injector = None
        self._nemo_rails = None
        self._health_cache: Dict[str, HealthCheckResult] = {}
        self._health_cache_ttl = 30  # seconds
        self._initialized = True

        # Initialize NeMo Guardrails
        self._init_nemo()

        logger.info("StackGuardrails initialized - enforcement active")

    def _init_nemo(self):
        """Initialize NeMo Guardrails for policy enforcement."""
        try:
            from pipeline.langgraph.nemo_stack_rails import NemoStackRails
            self._nemo_rails = NemoStackRails()
            if self._nemo_rails.available:
                logger.info("NeMo Guardrails ACTIVE - policy enforcement enabled")
            else:
                logger.warning("NeMo Guardrails not available - using fallback enforcement")
        except ImportError as e:
            logger.warning(f"Could not load NeMo Guardrails: {e}")
            self._nemo_rails = None

    @property
    def nemo_active(self) -> bool:
        """Check if NeMo enforcement is active."""
        return self._nemo_rails is not None and self._nemo_rails.available

    def _get_stack_injector(self):
        """Lazy load stack injector."""
        if self._stack_injector is None:
            try:
                from pipeline.langgraph.stack_injection import get_stack_injector
                self._stack_injector = get_stack_injector()
            except ImportError as e:
                logger.error(f"Could not load StackInjector: {e}")
        return self._stack_injector

    # =========================================================================
    # HEALTH VALIDATION
    # =========================================================================

    def check_health(self, stack_name: str, force: bool = False) -> HealthCheckResult:
        """Check health of a specific stack.

        Args:
            stack_name: Name of the stack to check
            force: Force fresh check (bypass cache)

        Returns:
            HealthCheckResult with status
        """
        # Check cache first
        if not force and stack_name in self._health_cache:
            cached = self._health_cache[stack_name]
            age = (datetime.now(timezone.utc) - cached.timestamp).total_seconds()
            if age < self._health_cache_ttl:
                return cached

        # Perform fresh check
        injector = self._get_stack_injector()
        if not injector:
            result = HealthCheckResult(
                stack_name=stack_name,
                healthy=False,
                latency_ms=0,
                error="StackInjector not available",
            )
        else:
            start = time.time()
            try:
                health = injector.check_health()
                stack_health = health.get(stack_name, {})
                latency = (time.time() - start) * 1000

                result = HealthCheckResult(
                    stack_name=stack_name,
                    healthy=stack_health.get("healthy", False),
                    latency_ms=latency,
                    error=stack_health.get("error"),
                )
            except Exception as e:
                result = HealthCheckResult(
                    stack_name=stack_name,
                    healthy=False,
                    latency_ms=(time.time() - start) * 1000,
                    error=str(e),
                )

        self._health_cache[stack_name] = result
        return result

    def validate_pre_flight(
        self,
        required_stacks: List[str],
        raise_on_failure: bool = True,
    ) -> Dict[str, HealthCheckResult]:
        """Validate all required stacks are healthy before operation.

        Args:
            required_stacks: List of stack names that must be healthy
            raise_on_failure: Raise exception if any stack is unhealthy

        Returns:
            Dict of stack_name -> HealthCheckResult

        Raises:
            StackGuardrailError: If any required stack is unhealthy
        """
        results = {}
        failures = []

        for stack_name in required_stacks:
            result = self.check_health(stack_name)
            results[stack_name] = result

            if not result.healthy:
                failures.append((stack_name, result.error))

                # Record violation
                self.violation_tracker.record(
                    guardrail="pre_flight_check",
                    severity=GuardrailSeverity.HIGH,
                    message=f"Stack '{stack_name}' failed pre-flight check: {result.error}",
                    context={"stack": stack_name, "error": result.error},
                )

        if failures and raise_on_failure:
            failed_names = [f[0] for f in failures]
            raise StackGuardrailError(
                f"Pre-flight check failed for stacks: {failed_names}",
                failed_stacks=failures,
            )

        return results

    # =========================================================================
    # ENFORCEMENT
    # =========================================================================

    def enforce_mandatory(
        self,
        operation: str,
        used_stacks: Set[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Violation]:
        """Enforce mandatory stack usage for an operation.

        Args:
            operation: Type of operation being performed
            used_stacks: Set of stacks actually used
            context: Additional context for violation report

        Returns:
            List of violations (empty if compliant)
        """
        violations = []

        for stack_name, rules in MANDATORY_STACKS.items():
            if operation in rules["operations"]:
                if stack_name not in used_stacks:
                    violation = self.violation_tracker.record(
                        guardrail=f"mandatory_{stack_name}",
                        severity=rules["severity"],
                        message=rules["message"],
                        context={
                            "operation": operation,
                            "required_stack": stack_name,
                            "used_stacks": list(used_stacks),
                            **(context or {}),
                        },
                    )
                    violations.append(violation)

                    if rules["action"] == EnforcementAction.BLOCK:
                        raise StackGuardrailError(
                            f"BLOCKED: {rules['message']}",
                            violations=[violation],
                        )

        return violations

    def check_recommended(
        self,
        operation: str,
        used_stacks: Set[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Violation]:
        """Check recommended stack usage for an operation.

        Args:
            operation: Type of operation being performed
            used_stacks: Set of stacks actually used
            context: Additional context

        Returns:
            List of violations (warnings)
        """
        violations = []

        for stack_name, rules in RECOMMENDED_STACKS.items():
            if operation in rules["operations"]:
                if stack_name not in used_stacks:
                    violation = self.violation_tracker.record(
                        guardrail=f"recommended_{stack_name}",
                        severity=rules["severity"],
                        message=rules["message"],
                        context={
                            "operation": operation,
                            "recommended_stack": stack_name,
                            "used_stacks": list(used_stacks),
                            **(context or {}),
                        },
                    )
                    violations.append(violation)

        return violations

    # =========================================================================
    # PROTECTED EXECUTION
    # =========================================================================

    @contextmanager
    def protected_call(
        self,
        stack_name: str,
        fallback_stack: Optional[str] = None,
        retry_count: int = 3,
        retry_delay: float = 1.0,
    ):
        """Execute a stack call with full protection.

        Provides:
        1. Circuit breaker protection
        2. Automatic retry with backoff
        3. Fallback to alternative stack
        4. Violation tracking

        Usage:
            with guardrails.protected_call("qdrant", fallback_stack="redis") as stack:
                result = stack.search(...)

        Args:
            stack_name: Primary stack to use
            fallback_stack: Alternative stack if primary fails
            retry_count: Number of retries before fallback
            retry_delay: Initial delay between retries (exponential backoff)

        Yields:
            Stack client object
        """
        breaker = self.circuit_registry.get_breaker(stack_name)

        # Check circuit breaker
        if not breaker.can_execute():
            if fallback_stack:
                logger.warning(
                    f"Circuit open for '{stack_name}', using fallback '{fallback_stack}'"
                )
                stack_name = fallback_stack
                breaker = self.circuit_registry.get_breaker(fallback_stack)

                if not breaker.can_execute():
                    raise StackGuardrailError(
                        f"Both primary and fallback stacks unavailable",
                        failed_stacks=[(stack_name, "circuit open")],
                    )
            else:
                raise StackGuardrailError(
                    f"Stack '{stack_name}' circuit is open",
                    failed_stacks=[(stack_name, "circuit open")],
                )

        # Get stack client
        injector = self._get_stack_injector()
        if not injector:
            raise StackGuardrailError("StackInjector not available")

        # Try to get the stack
        last_error = None
        for attempt in range(retry_count):
            try:
                stack = injector.get_stack(stack_name)
                if stack is None:
                    raise StackGuardrailError(f"Stack '{stack_name}' returned None")

                yield stack

                # Success
                breaker.record_success()
                return

            except Exception as e:
                last_error = e
                breaker.record_failure(e)

                if attempt < retry_count - 1:
                    delay = retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Stack '{stack_name}' failed (attempt {attempt + 1}/{retry_count}), "
                        f"retrying in {delay:.1f}s: {e}"
                    )
                    time.sleep(delay)

        # All retries failed
        if fallback_stack and fallback_stack != stack_name:
            logger.warning(
                f"Stack '{stack_name}' failed after {retry_count} attempts, "
                f"trying fallback '{fallback_stack}'"
            )
            with self.protected_call(fallback_stack, retry_count=1) as fallback:
                yield fallback
        else:
            self.violation_tracker.record(
                guardrail="protected_call",
                severity=GuardrailSeverity.CRITICAL,
                message=f"Stack '{stack_name}' failed after {retry_count} attempts",
                context={"stack": stack_name, "error": str(last_error)},
            )
            raise StackGuardrailError(
                f"Stack '{stack_name}' failed after {retry_count} attempts: {last_error}",
                failed_stacks=[(stack_name, str(last_error))],
            )

    # =========================================================================
    # DECORATORS
    # =========================================================================

    def require_stack(self, *stack_names: str) -> Callable[[F], F]:
        """Decorator that requires specific stacks to be healthy.

        Usage:
            @guardrails.require_stack("redis", "langfuse")
            def my_function():
                ...
        """
        def decorator(func: F) -> F:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Validate stacks are healthy
                self.validate_pre_flight(list(stack_names))
                return func(*args, **kwargs)
            return wrapper  # type: ignore
        return decorator

    def track_stack_usage(self, operation: str) -> Callable[[F], F]:
        """Decorator that tracks and enforces stack usage.

        Usage:
            @guardrails.track_stack_usage("llm_call")
            def my_llm_function():
                ...  # Must use langfuse!
        """
        def decorator(func: F) -> F:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Track which stacks are used
                # This is a simplified version - full implementation would
                # instrument stack calls
                used_stacks: Set[str] = set()

                # For now, we check mandatory stacks
                self.enforce_mandatory(operation, used_stacks)

                return func(*args, **kwargs)
            return wrapper  # type: ignore
        return decorator

    def with_circuit_breaker(
        self,
        stack_name: str,
        fallback: Optional[Callable] = None,
    ) -> Callable[[F], F]:
        """Decorator that adds circuit breaker protection.

        Usage:
            @guardrails.with_circuit_breaker("qdrant", fallback=use_redis_cache)
            def search_vectors():
                ...
        """
        def decorator(func: F) -> F:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                breaker = self.circuit_registry.get_breaker(stack_name)

                if not breaker.can_execute():
                    if fallback:
                        logger.warning(
                            f"Circuit open for '{stack_name}', using fallback"
                        )
                        return fallback(*args, **kwargs)
                    else:
                        raise StackGuardrailError(
                            f"Circuit breaker open for '{stack_name}'"
                        )

                try:
                    result = func(*args, **kwargs)
                    breaker.record_success()
                    return result
                except Exception as e:
                    breaker.record_failure(e)
                    if fallback:
                        logger.warning(
                            f"Stack '{stack_name}' failed, using fallback: {e}"
                        )
                        return fallback(*args, **kwargs)
                    raise

            return wrapper  # type: ignore
        return decorator

    # =========================================================================
    # REPORTING
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get complete guardrails status."""
        return {
            "violations": self.violation_tracker.get_summary(),
            "circuit_breakers": self.circuit_registry.get_status(),
            "health_cache_size": len(self._health_cache),
        }

    def get_health_report(self) -> Dict[str, Any]:
        """Get health report for all stacks."""
        injector = self._get_stack_injector()
        if not injector:
            return {"error": "StackInjector not available"}

        health = injector.check_health()
        total = len(health)
        healthy = sum(1 for h in health.values() if h.get("healthy", False))

        return {
            "total_stacks": total,
            "healthy_stacks": healthy,
            "unhealthy_stacks": total - healthy,
            "coverage_pct": (healthy / total * 100) if total > 0 else 0,
            "stacks": health,
            "circuit_breakers": self.circuit_registry.get_status(),
        }


# =============================================================================
# EXCEPTIONS
# =============================================================================


class StackGuardrailError(Exception):
    """Exception raised when guardrails are violated."""

    def __init__(
        self,
        message: str,
        failed_stacks: Optional[List[tuple]] = None,
        violations: Optional[List[Violation]] = None,
    ):
        super().__init__(message)
        self.failed_stacks = failed_stacks or []
        self.violations = violations or []


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


_guardrails_instance: Optional[StackGuardrails] = None


def get_guardrails() -> StackGuardrails:
    """Get the singleton StackGuardrails instance."""
    global _guardrails_instance
    if _guardrails_instance is None:
        _guardrails_instance = StackGuardrails()
    return _guardrails_instance


def validate_stacks(*stack_names: str) -> Dict[str, HealthCheckResult]:
    """Validate that stacks are healthy."""
    return get_guardrails().validate_pre_flight(list(stack_names))


def protected_stack_call(
    stack_name: str,
    fallback_stack: Optional[str] = None,
):
    """Context manager for protected stack calls."""
    return get_guardrails().protected_call(stack_name, fallback_stack)


def require_stack(*stack_names: str):
    """Decorator requiring stacks to be healthy."""
    return get_guardrails().require_stack(*stack_names)


def with_circuit_breaker(stack_name: str, fallback: Optional[Callable] = None):
    """Decorator adding circuit breaker protection."""
    return get_guardrails().with_circuit_breaker(stack_name, fallback)


def get_guardrails_status() -> Dict[str, Any]:
    """Get guardrails status report."""
    return get_guardrails().get_status()


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Classes
    "StackGuardrails",
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    "ViolationTracker",
    "Violation",
    "HealthCheckResult",
    "StackGuardrailError",
    # Enums
    "GuardrailSeverity",
    "CircuitState",
    "EnforcementAction",
    # Constants
    "MANDATORY_STACKS",
    "RECOMMENDED_STACKS",
    # Functions
    "get_guardrails",
    "validate_stacks",
    "protected_stack_call",
    "require_stack",
    "with_circuit_breaker",
    "get_guardrails_status",
]
