"""Immutable retry configuration types.

This module defines the RetryConfig dataclass and preset configurations
for different types of operations. All configurations are immutable (frozen)
to prevent accidental modification during execution.

Presets:
    RETRY_TRANSIENT: For transient failures (network blips, temporary unavailability)
    RETRY_EXTERNAL: For external service calls (APIs, databases, message queues)
    RETRY_CRITICAL: For critical operations (state persistence, checkpoints)
    RETRY_NONE: For idempotent operations where retry provides no benefit
    RETRY_AGGRESSIVE: For latency-sensitive operations with fast recovery
"""

from dataclasses import dataclass, field
from typing import Tuple, Type


@dataclass(frozen=True)
class RetryConfig:
    """Immutable configuration for retry with exponential backoff.

    Attributes:
        max_retries: Maximum number of retry attempts (0 = no retries).
        base_delay_seconds: Initial delay between retries.
        max_delay_seconds: Maximum delay cap.
        exponential_base: Base for exponential backoff calculation.
        jitter: Whether to add random jitter (+-25%) to delays.
        retryable_exceptions: Exception types that trigger retries.

    Example:
        >>> config = RetryConfig(max_retries=3, base_delay_seconds=1.0)
        >>> # Delays: 1s, 2s, 4s (with optional jitter)

    Invariants:
        - max_retries >= 0
        - base_delay_seconds > 0
        - max_delay_seconds >= base_delay_seconds
        - exponential_base > 1
    """

    max_retries: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: Tuple[Type[Exception], ...] = field(
        default=(Exception,)
    )

    def __post_init__(self) -> None:
        """Validate configuration invariants."""
        if self.max_retries < 0:
            raise ValueError(
                f"max_retries must be >= 0, got {self.max_retries}"
            )
        if self.base_delay_seconds <= 0:
            raise ValueError(
                f"base_delay_seconds must be > 0, got {self.base_delay_seconds}"
            )
        if self.max_delay_seconds < self.base_delay_seconds:
            raise ValueError(
                f"max_delay_seconds ({self.max_delay_seconds}) must be >= "
                f"base_delay_seconds ({self.base_delay_seconds})"
            )
        if self.exponential_base <= 1:
            raise ValueError(
                f"exponential_base must be > 1, got {self.exponential_base}"
            )

    def with_overrides(self, **kwargs) -> "RetryConfig":
        """Create a new config with overridden values.

        Args:
            **kwargs: Fields to override.

        Returns:
            New RetryConfig with overridden values.

        Example:
            >>> config = RETRY_TRANSIENT.with_overrides(max_retries=5)
        """
        # Get current values as dict
        current = {
            "max_retries": self.max_retries,
            "base_delay_seconds": self.base_delay_seconds,
            "max_delay_seconds": self.max_delay_seconds,
            "exponential_base": self.exponential_base,
            "jitter": self.jitter,
            "retryable_exceptions": self.retryable_exceptions,
        }
        # Apply overrides
        current.update(kwargs)
        return RetryConfig(**current)


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

RETRY_TRANSIENT = RetryConfig(
    max_retries=3,
    base_delay_seconds=1.0,
    max_delay_seconds=10.0,
    retryable_exceptions=(ConnectionError, TimeoutError, OSError),
)
"""Use for transient failures: network blips, temporary unavailability.

Expected scenarios:
    - Brief network interruptions
    - Service momentarily overloaded
    - DNS resolution delays

Success probability with 30% failure rate: 99.19% (1 - 0.3^4)
Max total wait time: ~14s (1 + 2 + 4 + 7s cap with jitter)
"""

RETRY_EXTERNAL = RetryConfig(
    max_retries=5,
    base_delay_seconds=2.0,
    max_delay_seconds=60.0,
    retryable_exceptions=(ConnectionError, TimeoutError, OSError),
)
"""Use for external service calls: APIs, databases, message queues.

Expected scenarios:
    - Database connection pool exhausted
    - Redis cluster failover
    - External API rate limiting

Success probability with 30% failure rate: 99.98% (1 - 0.3^6)
Max total wait time: ~126s (2 + 4 + 8 + 16 + 32 + 60s cap with jitter)
"""

RETRY_CRITICAL = RetryConfig(
    max_retries=10,
    base_delay_seconds=0.5,
    max_delay_seconds=120.0,
    retryable_exceptions=(Exception,),
)
"""Use for critical operations that MUST succeed: state persistence, checkpoints.

Expected scenarios:
    - Disk write failures
    - Checkpoint persistence
    - State file updates

Success probability with 30% failure rate: 99.9999% (1 - 0.3^11)
Max total wait time: ~360s with aggressive retry
"""

RETRY_NONE = RetryConfig(
    max_retries=0,
    base_delay_seconds=1.0,
    max_delay_seconds=1.0,
)
"""Use for idempotent operations where retry provides no benefit.

Expected scenarios:
    - Pure validation (no side effects)
    - Non-recoverable errors
    - Operations that must fail-fast
"""

RETRY_AGGRESSIVE = RetryConfig(
    max_retries=3,
    base_delay_seconds=0.1,
    max_delay_seconds=1.0,
    jitter=False,
)
"""Use for latency-sensitive operations with fast recovery.

Expected scenarios:
    - Cache reads
    - Health checks
    - Status polling

Success probability with 30% failure rate: 99.19%
Max total wait time: ~0.7s (0.1 + 0.2 + 0.4)
"""

# Redis-specific presets
RETRY_REDIS = RetryConfig(
    max_retries=5,
    base_delay_seconds=0.5,
    max_delay_seconds=10.0,
    retryable_exceptions=(ConnectionError, TimeoutError, OSError),
)
"""Use for Redis operations: cache, pub/sub, streams."""

# CrewAI-specific presets
RETRY_CREWAI = RetryConfig(
    max_retries=3,
    base_delay_seconds=5.0,
    max_delay_seconds=60.0,
    retryable_exceptions=(Exception,),
)
"""Use for CrewAI operations: crew execution, agent spawning.

Note: CrewAI operations can take significant time, so we use longer delays.
"""

# File I/O presets
RETRY_FILE_IO = RetryConfig(
    max_retries=5,
    base_delay_seconds=0.2,
    max_delay_seconds=5.0,
    retryable_exceptions=(OSError, IOError, PermissionError),
)
"""Use for file I/O operations: read, write, atomic operations."""


__all__ = [
    "RetryConfig",
    "RETRY_TRANSIENT",
    "RETRY_EXTERNAL",
    "RETRY_CRITICAL",
    "RETRY_NONE",
    "RETRY_AGGRESSIVE",
    "RETRY_REDIS",
    "RETRY_CREWAI",
    "RETRY_FILE_IO",
]
