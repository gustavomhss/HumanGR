"""Core retry functions with exponential backoff.

This module provides async and sync retry executors with:
    - Exponential backoff with jitter
    - Configurable retryable exceptions
    - Retry result tracking for metrics
    - Callback hooks for logging/alerting

Usage:
    async def flaky_operation():
        if random.random() < 0.5:
            raise ConnectionError("Network error")
        return "success"

    result = await retry_async(
        flaky_operation,
        "my_operation",
        config=RETRY_TRANSIENT,
    )
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
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

from .retry_config import RetryConfig, RETRY_TRANSIENT

logger = logging.getLogger(__name__)

T = TypeVar("T")
AsyncOperation = Callable[[], Awaitable[T]]
SyncOperation = Callable[[], T]
OnRetryCallback = Callable[[int, Exception], Awaitable[None]]
OnRetrySyncCallback = Callable[[int, Exception], None]


def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate delay for retry attempt using exponential backoff.

    Formula: min(base * (exponential_base ^ attempt), max_delay) * jitter

    Args:
        attempt: Zero-indexed attempt number (0 = first retry).
        config: Retry configuration.

    Returns:
        Delay in seconds before next retry.

    Example:
        >>> config = RetryConfig(base_delay_seconds=1.0, exponential_base=2.0, jitter=False)
        >>> calculate_delay(0, config)  # 1.0s
        >>> calculate_delay(1, config)  # 2.0s
        >>> calculate_delay(2, config)  # 4.0s
    """
    # Base exponential calculation
    delay = config.base_delay_seconds * (config.exponential_base ** attempt)

    # Apply maximum cap
    delay = min(delay, config.max_delay_seconds)

    # Apply jitter if enabled (+-25%)
    if config.jitter:
        jitter_factor = 0.75 + random.random() * 0.5
        delay = delay * jitter_factor

    return delay


@dataclass
class RetryResult(Generic[T]):
    """Result of a retry operation with full tracking.

    Attributes:
        success: Whether the operation eventually succeeded.
        result: The result if successful, None otherwise.
        attempts: Total number of attempts made.
        total_delay_seconds: Total time spent waiting between retries.
        exceptions: List of exceptions encountered.
        operation_name: Name of the operation for logging.
        started_at: When the retry sequence started.
        finished_at: When the retry sequence finished.
    """

    success: bool
    result: Optional[T]
    attempts: int
    total_delay_seconds: float
    exceptions: List[Exception]
    operation_name: str
    started_at: datetime
    finished_at: datetime

    @property
    def duration_seconds(self) -> float:
        """Total duration including operation time and delays."""
        return (self.finished_at - self.started_at).total_seconds()

    @property
    def final_exception(self) -> Optional[Exception]:
        """The last exception encountered, if any."""
        return self.exceptions[-1] if self.exceptions else None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for logging/metrics."""
        return {
            "success": self.success,
            "attempts": self.attempts,
            "total_delay_seconds": round(self.total_delay_seconds, 3),
            "duration_seconds": round(self.duration_seconds, 3),
            "operation_name": self.operation_name,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "exception_count": len(self.exceptions),
            "final_exception": (
                str(self.final_exception)[:200]
                if self.final_exception
                else None
            ),
        }


async def retry_async(
    operation: AsyncOperation[T],
    operation_name: str,
    config: RetryConfig = RETRY_TRANSIENT,
    on_retry: Optional[OnRetryCallback] = None,
) -> T:
    """Execute async operation with exponential backoff retry.

    Args:
        operation: Async callable to execute (no arguments).
        operation_name: Name for logging purposes.
        config: Retry configuration.
        on_retry: Optional callback invoked before each retry.

    Returns:
        Result of successful operation.

    Raises:
        Exception: The last exception if all retries exhausted.

    Example:
        async def flaky_operation():
            if random.random() < 0.5:
                raise ConnectionError("Network error")
            return "success"

        result = await retry_async(
            flaky_operation,
            "my_operation",
            config=RETRY_TRANSIENT,
        )
    """
    last_exception: Optional[Exception] = None
    total_attempts = config.max_retries + 1

    for attempt in range(total_attempts):
        try:
            return await operation()
        except config.retryable_exceptions as e:
            last_exception = e

            if attempt == config.max_retries:
                logger.error(
                    "RETRY_EXHAUSTED: operation='%s' attempts=%d error='%s'",
                    operation_name,
                    total_attempts,
                    str(e)[:200],
                )
                raise

            delay = calculate_delay(attempt, config)

            logger.warning(
                "RETRY: operation='%s' attempt=%d/%d error='%s' delay=%.2fs",
                operation_name,
                attempt + 1,
                total_attempts,
                str(e)[:100],
                delay,
            )

            if on_retry is not None:
                await on_retry(attempt, e)

            await asyncio.sleep(delay)

    # Should not reach here, but safety net
    if last_exception is not None:
        raise last_exception
    raise RuntimeError(f"Unexpected state in retry_async for {operation_name}")


async def retry_async_with_result(
    operation: AsyncOperation[T],
    operation_name: str,
    config: RetryConfig = RETRY_TRANSIENT,
    on_retry: Optional[OnRetryCallback] = None,
) -> RetryResult[T]:
    """Execute async operation with full result tracking.

    Unlike retry_async, this never raises - it returns a RetryResult
    that indicates success or failure.

    Args:
        operation: Async callable to execute.
        operation_name: Name for logging purposes.
        config: Retry configuration.
        on_retry: Optional callback invoked before each retry.

    Returns:
        RetryResult with full tracking information.

    Example:
        result = await retry_async_with_result(operation, "my_op")
        if result.success:
            print(f"Success after {result.attempts} attempts")
        else:
            print(f"Failed: {result.final_exception}")
    """
    started_at = datetime.now()
    exceptions: List[Exception] = []
    total_delay = 0.0
    total_attempts = config.max_retries + 1
    result: Optional[T] = None
    success = False

    for attempt in range(total_attempts):
        try:
            result = await operation()
            success = True
            break
        except config.retryable_exceptions as e:
            exceptions.append(e)

            if attempt == config.max_retries:
                logger.error(
                    "RETRY_EXHAUSTED: operation='%s' attempts=%d error='%s'",
                    operation_name,
                    total_attempts,
                    str(e)[:200],
                )
                break

            delay = calculate_delay(attempt, config)
            total_delay += delay

            logger.warning(
                "RETRY: operation='%s' attempt=%d/%d error='%s' delay=%.2fs",
                operation_name,
                attempt + 1,
                total_attempts,
                str(e)[:100],
                delay,
            )

            if on_retry is not None:
                await on_retry(attempt, e)

            await asyncio.sleep(delay)

    finished_at = datetime.now()

    return RetryResult(
        success=success,
        result=result,
        attempts=len(exceptions) + (1 if success else 0),
        total_delay_seconds=total_delay,
        exceptions=exceptions,
        operation_name=operation_name,
        started_at=started_at,
        finished_at=finished_at,
    )


def retry_sync(
    operation: SyncOperation[T],
    operation_name: str,
    config: RetryConfig = RETRY_TRANSIENT,
    on_retry: Optional[OnRetrySyncCallback] = None,
) -> T:
    """Execute sync operation with exponential backoff retry.

    Args:
        operation: Sync callable to execute (no arguments).
        operation_name: Name for logging purposes.
        config: Retry configuration.
        on_retry: Optional callback invoked before each retry.

    Returns:
        Result of successful operation.

    Raises:
        Exception: The last exception if all retries exhausted.

    Example:
        def flaky_operation():
            if random.random() < 0.5:
                raise ConnectionError("Network error")
            return "success"

        result = retry_sync(
            flaky_operation,
            "my_operation",
            config=RETRY_TRANSIENT,
        )
    """
    last_exception: Optional[Exception] = None
    total_attempts = config.max_retries + 1

    for attempt in range(total_attempts):
        try:
            return operation()
        except config.retryable_exceptions as e:
            last_exception = e

            if attempt == config.max_retries:
                logger.error(
                    "RETRY_EXHAUSTED: operation='%s' attempts=%d error='%s'",
                    operation_name,
                    total_attempts,
                    str(e)[:200],
                )
                raise

            delay = calculate_delay(attempt, config)

            logger.warning(
                "RETRY: operation='%s' attempt=%d/%d error='%s' delay=%.2fs",
                operation_name,
                attempt + 1,
                total_attempts,
                str(e)[:100],
                delay,
            )

            if on_retry is not None:
                on_retry(attempt, e)

            time.sleep(delay)

    # Should not reach here, but safety net
    if last_exception is not None:
        raise last_exception
    raise RuntimeError(f"Unexpected state in retry_sync for {operation_name}")


def retry_sync_with_result(
    operation: SyncOperation[T],
    operation_name: str,
    config: RetryConfig = RETRY_TRANSIENT,
    on_retry: Optional[OnRetrySyncCallback] = None,
) -> RetryResult[T]:
    """Execute sync operation with full result tracking.

    Unlike retry_sync, this never raises - it returns a RetryResult
    that indicates success or failure.

    Args:
        operation: Sync callable to execute.
        operation_name: Name for logging purposes.
        config: Retry configuration.
        on_retry: Optional callback invoked before each retry.

    Returns:
        RetryResult with full tracking information.
    """
    started_at = datetime.now()
    exceptions: List[Exception] = []
    total_delay = 0.0
    total_attempts = config.max_retries + 1
    result: Optional[T] = None
    success = False

    for attempt in range(total_attempts):
        try:
            result = operation()
            success = True
            break
        except config.retryable_exceptions as e:
            exceptions.append(e)

            if attempt == config.max_retries:
                logger.error(
                    "RETRY_EXHAUSTED: operation='%s' attempts=%d error='%s'",
                    operation_name,
                    total_attempts,
                    str(e)[:200],
                )
                break

            delay = calculate_delay(attempt, config)
            total_delay += delay

            logger.warning(
                "RETRY: operation='%s' attempt=%d/%d error='%s' delay=%.2fs",
                operation_name,
                attempt + 1,
                total_attempts,
                str(e)[:100],
                delay,
            )

            if on_retry is not None:
                on_retry(attempt, e)

            time.sleep(delay)

    finished_at = datetime.now()

    return RetryResult(
        success=success,
        result=result,
        attempts=len(exceptions) + (1 if success else 0),
        total_delay_seconds=total_delay,
        exceptions=exceptions,
        operation_name=operation_name,
        started_at=started_at,
        finished_at=finished_at,
    )


__all__ = [
    "calculate_delay",
    "retry_async",
    "retry_async_with_result",
    "retry_sync",
    "retry_sync_with_result",
    "RetryResult",
    "AsyncOperation",
    "SyncOperation",
]
