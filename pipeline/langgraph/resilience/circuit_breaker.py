"""Circuit Breaker pattern for preventing cascade failures.

The circuit breaker prevents a failing service from overwhelming the system
by quickly failing requests when a service is known to be down.

State Machine:
    CLOSED (normal operation)
        │
        │ failure_count >= failure_threshold
        ▼
    OPEN (rejecting calls)
        │
        │ recovery_timeout elapsed
        ▼
    HALF_OPEN (testing recovery)
        │
        ├─ success × N → CLOSED
        │
        └─ failure → OPEN

Usage:
    breaker = CircuitBreaker("redis", failure_threshold=5, recovery_timeout_seconds=60)

    try:
        result = await breaker.call(lambda: redis_client.get("key"))
    except CircuitBreakerOpen as e:
        # Service is known to be down, fail fast
        logger.warning(f"Circuit open, retry in {e.time_until_half_open}s")
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    TypeVar,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")
AsyncOperation = Callable[[], Awaitable[T]]


class CircuitState(str, Enum):
    """State of a circuit breaker.

    CLOSED: Normal operation. Requests pass through. Failures are tracked.
    OPEN: Too many failures. Requests are rejected immediately.
    HALF_OPEN: Testing if service recovered. Limited requests allowed.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open and rejecting calls.

    Attributes:
        service_name: Name of the protected service.
        time_until_half_open: Seconds until circuit transitions to half-open.
        failure_count: Number of failures that triggered the opening.
    """

    def __init__(
        self,
        service_name: str,
        time_until_half_open: float,
        failure_count: int = 0,
    ):
        super().__init__(
            f"Circuit breaker open for '{service_name}'. "
            f"Retry in {time_until_half_open:.1f}s"
        )
        self.service_name = service_name
        self.time_until_half_open = time_until_half_open
        self.failure_count = failure_count


@dataclass
class CircuitBreakerStats:
    """Statistics for a circuit breaker."""

    service_name: str
    state: CircuitState
    failure_count: int
    success_count: int
    total_calls: int
    total_failures: int
    total_successes: int
    consecutive_failures: int
    consecutive_successes: int
    last_failure_time: Optional[datetime]
    last_success_time: Optional[datetime]
    time_in_current_state_seconds: float
    state_transitions: int

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "service_name": self.service_name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_calls": self.total_calls,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "last_failure_time": (
                self.last_failure_time.isoformat()
                if self.last_failure_time
                else None
            ),
            "last_success_time": (
                self.last_success_time.isoformat()
                if self.last_success_time
                else None
            ),
            "time_in_current_state_seconds": round(
                self.time_in_current_state_seconds, 2
            ),
            "state_transitions": self.state_transitions,
        }


class CircuitBreaker:
    """Circuit breaker for protecting against cascade failures.

    Prevents a failing service from overwhelming the system by:
    1. Tracking failures in CLOSED state
    2. Opening the circuit when failures exceed threshold
    3. Rejecting requests immediately when OPEN
    4. Periodically testing recovery in HALF_OPEN state

    Thread-safe and async-safe using asyncio.Lock.

    Attributes:
        service_name: Name of the protected service (for logging).
        failure_threshold: Number of failures to trigger OPEN state.
        recovery_timeout_seconds: Time to wait before testing recovery.
        half_open_success_threshold: Successes needed to close circuit.
    """

    def __init__(
        self,
        service_name: str,
        failure_threshold: int = 5,
        recovery_timeout_seconds: float = 60.0,
        half_open_success_threshold: int = 2,
    ):
        """Initialize circuit breaker.

        Args:
            service_name: Name for logging and identification.
            failure_threshold: Failures needed to open circuit.
            recovery_timeout_seconds: Seconds before testing recovery.
            half_open_success_threshold: Successes needed to close.
        """
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if recovery_timeout_seconds <= 0:
            raise ValueError("recovery_timeout_seconds must be > 0")
        if half_open_success_threshold < 1:
            raise ValueError("half_open_success_threshold must be >= 1")

        self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.recovery_timeout_seconds = recovery_timeout_seconds
        self.half_open_success_threshold = half_open_success_threshold

        # State
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count_half_open = 0
        self._last_failure_time: Optional[float] = None
        self._last_success_time: Optional[float] = None
        self._state_changed_at: float = time.time()

        # Statistics
        self._total_calls = 0
        self._total_failures = 0
        self._total_successes = 0
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._state_transitions = 0

        # Concurrency control
        self._lock = asyncio.Lock()

        logger.debug(
            "CircuitBreaker created: service=%s threshold=%d timeout=%.1fs",
            service_name,
            failure_threshold,
            recovery_timeout_seconds,
        )

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        return self._state

    @property
    def is_open(self) -> bool:
        """Whether circuit is currently open (rejecting calls)."""
        return self._state == CircuitState.OPEN

    @property
    def is_closed(self) -> bool:
        """Whether circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED

    @property
    def is_half_open(self) -> bool:
        """Whether circuit is half-open (testing recovery)."""
        return self._state == CircuitState.HALF_OPEN

    @property
    def failure_count(self) -> int:
        """Current failure count."""
        return self._failure_count

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state with logging."""
        old_state = self._state
        if old_state == new_state:
            return

        self._state = new_state
        self._state_changed_at = time.time()
        self._state_transitions += 1

        logger.info(
            "CIRCUIT_TRANSITION: service=%s %s -> %s (failures=%d)",
            self.service_name,
            old_state.value,
            new_state.value,
            self._failure_count,
        )

    async def record_success(self) -> None:
        """Record a successful call."""
        async with self._lock:
            self._total_successes += 1
            self._last_success_time = time.time()
            self._consecutive_successes += 1
            self._consecutive_failures = 0

            if self._state == CircuitState.HALF_OPEN:
                self._success_count_half_open += 1
                if self._success_count_half_open >= self.half_open_success_threshold:
                    logger.info(
                        "CIRCUIT_CLOSING: service=%s after %d successful calls",
                        self.service_name,
                        self._success_count_half_open,
                    )
                    self._transition_to(CircuitState.CLOSED)
                    self._failure_count = 0
                    self._success_count_half_open = 0

            elif self._state == CircuitState.CLOSED:
                # Decay failure count on success (gradual recovery)
                self._failure_count = max(0, self._failure_count - 1)

    async def record_failure(self, exception: Optional[Exception] = None) -> None:
        """Record a failed call.

        Args:
            exception: The exception that caused the failure (for logging).
        """
        async with self._lock:
            self._failure_count += 1
            self._total_failures += 1
            self._last_failure_time = time.time()
            self._consecutive_failures += 1
            self._consecutive_successes = 0

            if self._state == CircuitState.HALF_OPEN:
                logger.warning(
                    "CIRCUIT_REOPENING: service=%s failed during half-open test",
                    self.service_name,
                )
                self._transition_to(CircuitState.OPEN)
                self._success_count_half_open = 0

            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    logger.warning(
                        "CIRCUIT_OPENING: service=%s failures=%d threshold=%d error='%s'",
                        self.service_name,
                        self._failure_count,
                        self.failure_threshold,
                        str(exception)[:100] if exception else "unknown",
                    )
                    self._transition_to(CircuitState.OPEN)

    async def _maybe_transition_to_half_open(self) -> bool:
        """Check if we should transition from OPEN to HALF_OPEN.

        Returns:
            True if transitioned to HALF_OPEN.
        """
        if self._state != CircuitState.OPEN:
            return False

        if self._last_failure_time is None:
            return False

        elapsed = time.time() - self._last_failure_time
        if elapsed >= self.recovery_timeout_seconds:
            logger.info(
                "CIRCUIT_HALF_OPENING: service=%s recovery_timeout=%.1fs elapsed=%.1fs",
                self.service_name,
                self.recovery_timeout_seconds,
                elapsed,
            )
            self._transition_to(CircuitState.HALF_OPEN)
            self._success_count_half_open = 0
            return True

        return False

    def _get_time_until_half_open(self) -> float:
        """Get seconds until circuit transitions to half-open."""
        if self._state != CircuitState.OPEN or self._last_failure_time is None:
            return 0.0

        elapsed = time.time() - self._last_failure_time
        return max(0.0, self.recovery_timeout_seconds - elapsed)

    async def call(self, operation: AsyncOperation[T]) -> T:
        """Execute operation with circuit breaker protection.

        Args:
            operation: Async callable to execute.

        Returns:
            Result of the operation.

        Raises:
            CircuitBreakerOpen: If circuit is open.
            Exception: Any exception from the operation.

        Example:
            result = await breaker.call(lambda: redis.get("key"))
        """
        async with self._lock:
            self._total_calls += 1
            await self._maybe_transition_to_half_open()

            if self._state == CircuitState.OPEN:
                time_until_half_open = self._get_time_until_half_open()
                raise CircuitBreakerOpen(
                    self.service_name,
                    time_until_half_open,
                    self._failure_count,
                )

        # Execute operation outside the lock
        try:
            result = await operation()
            await self.record_success()
            return result
        except Exception as e:
            await self.record_failure(e)
            raise

    async def call_with_fallback(
        self,
        operation: AsyncOperation[T],
        fallback: T,
    ) -> T:
        """Execute operation with fallback when circuit is open.

        Args:
            operation: Async callable to execute.
            fallback: Value to return if circuit is open.

        Returns:
            Operation result or fallback value.
        """
        try:
            return await self.call(operation)
        except CircuitBreakerOpen:
            logger.debug(
                "CIRCUIT_FALLBACK: service=%s using fallback",
                self.service_name,
            )
            return fallback

    async def reset(self) -> None:
        """Manually reset circuit breaker to CLOSED state.

        Use this after known recovery (e.g., Run Master restarted service).
        """
        async with self._lock:
            logger.info(
                "CIRCUIT_MANUAL_RESET: service=%s previous_state=%s failures=%d",
                self.service_name,
                self._state.value,
                self._failure_count,
            )
            self._transition_to(CircuitState.CLOSED)
            self._failure_count = 0
            self._success_count_half_open = 0
            self._consecutive_failures = 0

    def get_stats(self) -> CircuitBreakerStats:
        """Get current statistics for this circuit breaker."""
        return CircuitBreakerStats(
            service_name=self.service_name,
            state=self._state,
            failure_count=self._failure_count,
            success_count=self._success_count_half_open,
            total_calls=self._total_calls,
            total_failures=self._total_failures,
            total_successes=self._total_successes,
            consecutive_failures=self._consecutive_failures,
            consecutive_successes=self._consecutive_successes,
            last_failure_time=(
                datetime.fromtimestamp(self._last_failure_time)
                if self._last_failure_time
                else None
            ),
            last_success_time=(
                datetime.fromtimestamp(self._last_success_time)
                if self._last_success_time
                else None
            ),
            time_in_current_state_seconds=time.time() - self._state_changed_at,
            state_transitions=self._state_transitions,
        )


class GlobalCircuitBreakerRegistry:
    """Singleton registry for all circuit breakers.

    Provides centralized management of circuit breakers:
    - Get or create circuit breakers by service name
    - Query states of all breakers
    - Reset all breakers (e.g., after Run Master recovery)

    Usage:
        registry = get_circuit_breaker_registry()
        redis_breaker = registry.get_breaker("redis")
        crewai_breaker = registry.get_breaker("crewai", failure_threshold=3)

        # Check all states
        states = registry.get_all_states()

        # Reset all after recovery
        await registry.reset_all()
    """

    _instance: Optional["GlobalCircuitBreakerRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._breakers: Dict[str, CircuitBreaker] = {}
                    instance._configs: Dict[str, Dict[str, Any]] = {}
                    cls._instance = instance
        return cls._instance

    def get_breaker(
        self,
        service_name: str,
        failure_threshold: int = 5,
        recovery_timeout_seconds: float = 60.0,
        half_open_success_threshold: int = 2,
    ) -> CircuitBreaker:
        """Get or create circuit breaker for a service.

        Args:
            service_name: Name of the service.
            failure_threshold: Failures to trigger OPEN (only for new breakers).
            recovery_timeout_seconds: Recovery timeout (only for new breakers).
            half_open_success_threshold: Successes to close (only for new breakers).

        Returns:
            Circuit breaker for the service.

        Note:
            Configuration parameters only apply when creating a new breaker.
            Existing breakers return with their original configuration.
        """
        if service_name not in self._breakers:
            self._breakers[service_name] = CircuitBreaker(
                service_name=service_name,
                failure_threshold=failure_threshold,
                recovery_timeout_seconds=recovery_timeout_seconds,
                half_open_success_threshold=half_open_success_threshold,
            )
            self._configs[service_name] = {
                "failure_threshold": failure_threshold,
                "recovery_timeout_seconds": recovery_timeout_seconds,
                "half_open_success_threshold": half_open_success_threshold,
            }

        return self._breakers[service_name]

    def get_all_states(self) -> Dict[str, CircuitState]:
        """Get states of all registered circuit breakers."""
        return {name: cb.state for name, cb in self._breakers.items()}

    def get_all_stats(self) -> Dict[str, CircuitBreakerStats]:
        """Get statistics for all registered circuit breakers."""
        return {name: cb.get_stats() for name, cb in self._breakers.items()}

    def get_open_circuits(self) -> List[str]:
        """Get names of all open circuit breakers."""
        return [name for name, cb in self._breakers.items() if cb.is_open]

    async def reset_all(self) -> None:
        """Reset all circuit breakers to CLOSED state."""
        logger.info(
            "REGISTRY_RESET_ALL: resetting %d circuit breakers",
            len(self._breakers),
        )
        for breaker in self._breakers.values():
            await breaker.reset()

    async def reset_breaker(self, service_name: str) -> bool:
        """Reset a specific circuit breaker.

        Args:
            service_name: Name of the service to reset.

        Returns:
            True if breaker was found and reset, False if not found.
        """
        if service_name in self._breakers:
            await self._breakers[service_name].reset()
            return True
        return False

    def remove_breaker(self, service_name: str) -> bool:
        """Remove a circuit breaker from the registry.

        Args:
            service_name: Name of the service to remove.

        Returns:
            True if breaker was found and removed, False if not found.
        """
        if service_name in self._breakers:
            del self._breakers[service_name]
            del self._configs[service_name]
            return True
        return False

    def clear(self) -> None:
        """Remove all circuit breakers from the registry.

        Warning: Use with caution. This resets all tracking.
        """
        logger.warning("REGISTRY_CLEAR: removing all circuit breakers")
        self._breakers.clear()
        self._configs.clear()


def get_circuit_breaker_registry() -> GlobalCircuitBreakerRegistry:
    """Get the global circuit breaker registry singleton."""
    return GlobalCircuitBreakerRegistry()


# =============================================================================
# Convenience Functions
# =============================================================================


async def with_circuit_breaker(
    service_name: str,
    operation: AsyncOperation[T],
    failure_threshold: int = 5,
    recovery_timeout_seconds: float = 60.0,
) -> T:
    """Execute operation with automatic circuit breaker.

    Convenience function that gets/creates a circuit breaker from
    the global registry and executes the operation.

    Args:
        service_name: Name of the service being called.
        operation: Async callable to execute.
        failure_threshold: Failures to trigger OPEN.
        recovery_timeout_seconds: Recovery timeout.

    Returns:
        Result of the operation.

    Raises:
        CircuitBreakerOpen: If circuit is open.
        Exception: Any exception from the operation.

    Example:
        result = await with_circuit_breaker(
            "redis",
            lambda: redis_client.get("key"),
        )
    """
    registry = get_circuit_breaker_registry()
    breaker = registry.get_breaker(
        service_name,
        failure_threshold=failure_threshold,
        recovery_timeout_seconds=recovery_timeout_seconds,
    )
    return await breaker.call(operation)


__all__ = [
    "CircuitState",
    "CircuitBreakerOpen",
    "CircuitBreakerStats",
    "CircuitBreaker",
    "GlobalCircuitBreakerRegistry",
    "get_circuit_breaker_registry",
    "with_circuit_breaker",
]
