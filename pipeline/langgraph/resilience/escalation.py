"""Run Master escalation for pipeline resilience.

This module provides the retry_with_run_master function that implements
the core resilience philosophy:

    1. Retry with exponential backoff
    2. If retries exhausted, escalate to Run Master
    3. Pipeline WAITS for Run Master resolution
    4. Retry once more after resolution
    5. If still fails, raise with full context

The Run Master is the "zelador" (caretaker) of the pipeline - it handles
infrastructure issues that agents cannot resolve themselves.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    TypeVar,
    Union,
)

from .retry import retry_async, calculate_delay, RetryResult
from .retry_config import RetryConfig, RETRY_TRANSIENT

if TYPE_CHECKING:
    from pipeline.langgraph.state import PipelineState

logger = logging.getLogger(__name__)

T = TypeVar("T")
AsyncOperation = Callable[[], Awaitable[T]]


class InterventionType(str, Enum):
    """Types of infrastructure issues requiring Run Master intervention.

    Each type maps to a specific handler in the Run Master.
    """

    # Stack/Container Issues
    STACK_UNAVAILABLE = "stack_unavailable"
    CONTAINER_DOWN = "container_down"
    CONTAINER_UNHEALTHY = "container_unhealthy"

    # Service Issues
    SERVICE_DOWN = "service_down"
    SERVICE_TIMEOUT = "service_timeout"
    SERVICE_ERROR = "service_error"

    # Resource Issues
    DISK_FULL = "disk_full"
    MEMORY_EXHAUSTED = "memory_exhausted"
    CPU_OVERLOAD = "cpu_overload"

    # Dependency Issues
    IMPORT_ERROR = "import_error"
    DEPENDENCY_MISSING = "dependency_missing"
    VERSION_MISMATCH = "version_mismatch"

    # Configuration Issues
    CONFIG_MISSING = "config_missing"
    CONFIG_INVALID = "config_invalid"
    CONTEXT_PACK_MISSING = "context_pack_missing"

    # Security Issues (require human approval)
    SECURITY_BLOCKED = "security_blocked"
    QUIETSTAR_BLOCKED = "quietstar_blocked"

    # Generic
    UNKNOWN = "unknown"


@dataclass
class InterventionRequest:
    """Request for Run Master intervention.

    This is the structured payload sent to the Run Master when
    the pipeline needs help resolving an infrastructure issue.
    """

    intervention_type: InterventionType
    error_message: str
    node: str  # Which workflow node encountered the issue
    sprint_id: str
    run_id: str

    # Additional context
    stack_name: Optional[str] = None
    service_name: Optional[str] = None
    module_name: Optional[str] = None
    file_path: Optional[str] = None

    # Retry history
    retry_attempts: int = 0
    total_retry_time_seconds: float = 0.0
    exceptions: List[str] = field(default_factory=list)

    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    escalated_at: Optional[datetime] = None

    # Suggested action (hint for Run Master)
    suggested_action: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for IPC/logging."""
        return {
            "intervention_type": self.intervention_type.value,
            "error_message": self.error_message[:500],  # Truncate long messages
            "node": self.node,
            "sprint_id": self.sprint_id,
            "run_id": self.run_id,
            "stack_name": self.stack_name,
            "service_name": self.service_name,
            "module_name": self.module_name,
            "file_path": self.file_path,
            "retry_attempts": self.retry_attempts,
            "total_retry_time_seconds": round(self.total_retry_time_seconds, 3),
            "exceptions": self.exceptions[-5:],  # Last 5 only
            "created_at": self.created_at.isoformat(),
            "escalated_at": (
                self.escalated_at.isoformat() if self.escalated_at else None
            ),
            "suggested_action": self.suggested_action,
        }


@dataclass
class InterventionResolution:
    """Resolution from Run Master after handling intervention.

    This is returned by the Run Master to indicate the outcome
    of its intervention attempt.
    """

    resolved: bool
    action_taken: str
    resolution_time_seconds: float

    # Details
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    # For retry guidance
    should_retry: bool = True
    new_config: Optional[Dict[str, Any]] = None


class EscalationError(Exception):
    """Error during Run Master escalation."""

    def __init__(
        self,
        message: str,
        request: InterventionRequest,
        resolution: Optional[InterventionResolution] = None,
    ):
        super().__init__(message)
        self.request = request
        self.resolution = resolution


async def _send_intervention_request(
    request: InterventionRequest,
    run_dir: Path,
) -> None:
    """Send intervention request to Run Master via IPC.

    This writes to the Run Master's alert queue file.
    """
    from .file_safety import append_ndjson_safe

    ipc_dir = run_dir / "state" / "ipc" / "alerts"
    ipc_dir.mkdir(parents=True, exist_ok=True)

    alert_file = ipc_dir / "pending_alerts.ndjson"
    request.escalated_at = datetime.now()

    append_ndjson_safe(alert_file, request.to_dict())

    logger.warning(
        "ESCALATED_TO_RUN_MASTER: type=%s node=%s sprint=%s",
        request.intervention_type.value,
        request.node,
        request.sprint_id,
    )


async def _wait_for_resolution(
    request: InterventionRequest,
    run_dir: Path,
    timeout_seconds: float = 300.0,
    poll_interval_seconds: float = 5.0,
) -> InterventionResolution:
    """Wait for Run Master to resolve the intervention.

    Args:
        request: The intervention request.
        run_dir: Pipeline run directory.
        timeout_seconds: Max time to wait for resolution.
        poll_interval_seconds: How often to check for resolution.

    Returns:
        InterventionResolution from Run Master.

    Raises:
        EscalationError: If timeout exceeded or resolution failed.
    """
    from .file_safety import read_json_safe

    resolution_file = (
        run_dir / "state" / "ipc" / "resolutions" /
        f"{request.sprint_id}_{request.node}_{request.created_at.timestamp():.0f}.json"
    )

    start_time = time.monotonic()

    logger.info(
        "WAITING_FOR_RESOLUTION: type=%s timeout=%.0fs",
        request.intervention_type.value,
        timeout_seconds,
    )

    while time.monotonic() - start_time < timeout_seconds:
        if resolution_file.exists():
            try:
                data = read_json_safe(resolution_file)
                resolution = InterventionResolution(
                    resolved=data.get("resolved", False),
                    action_taken=data.get("action_taken", "unknown"),
                    resolution_time_seconds=data.get("resolution_time_seconds", 0),
                    details=data.get("details", {}),
                    error_message=data.get("error_message"),
                    should_retry=data.get("should_retry", True),
                    new_config=data.get("new_config"),
                )

                logger.info(
                    "RESOLUTION_RECEIVED: resolved=%s action=%s time=%.2fs",
                    resolution.resolved,
                    resolution.action_taken,
                    resolution.resolution_time_seconds,
                )

                return resolution

            except Exception as e:
                logger.warning("Failed to read resolution file: %s", e)

        await asyncio.sleep(poll_interval_seconds)

    # Timeout - escalate to human
    raise EscalationError(
        f"Timeout waiting for Run Master resolution after {timeout_seconds}s",
        request=request,
    )


def _infer_intervention_type(
    exception: Exception,
    operation_name: str,
) -> InterventionType:
    """Infer intervention type from exception and operation context.

    Args:
        exception: The exception that triggered escalation.
        operation_name: Name of the failing operation.

    Returns:
        Best-guess InterventionType for the error.
    """
    exc_str = str(exception).lower()
    exc_type = type(exception).__name__.lower()

    # Import errors
    if isinstance(exception, ImportError) or "import" in exc_type:
        return InterventionType.IMPORT_ERROR

    # Connection errors
    if isinstance(exception, ConnectionError) or "connection" in exc_str:
        if "redis" in operation_name.lower() or "redis" in exc_str:
            return InterventionType.STACK_UNAVAILABLE
        if "crewai" in operation_name.lower():
            return InterventionType.SERVICE_DOWN
        return InterventionType.STACK_UNAVAILABLE

    # Timeout errors
    if isinstance(exception, TimeoutError) or "timeout" in exc_str:
        return InterventionType.SERVICE_TIMEOUT

    # Permission/Security
    if isinstance(exception, PermissionError) or "permission" in exc_str:
        return InterventionType.SECURITY_BLOCKED

    # File/IO errors
    if isinstance(exception, (FileNotFoundError, IOError)):
        if "context" in exc_str or "pack" in exc_str:
            return InterventionType.CONTEXT_PACK_MISSING
        if "config" in exc_str:
            return InterventionType.CONFIG_MISSING
        return InterventionType.CONFIG_MISSING

    # OS errors (disk, memory)
    if isinstance(exception, OSError):
        if "no space" in exc_str or "disk" in exc_str:
            return InterventionType.DISK_FULL
        if "memory" in exc_str:
            return InterventionType.MEMORY_EXHAUSTED

    return InterventionType.UNKNOWN


def _suggest_action(intervention_type: InterventionType) -> str:
    """Suggest an action for the Run Master based on intervention type."""
    suggestions = {
        InterventionType.STACK_UNAVAILABLE: "docker restart <container>",
        InterventionType.CONTAINER_DOWN: "docker start <container>",
        InterventionType.CONTAINER_UNHEALTHY: "docker restart <container>",
        InterventionType.SERVICE_DOWN: "Restart service or check health",
        InterventionType.SERVICE_TIMEOUT: "Check service load, increase timeout",
        InterventionType.DISK_FULL: "Free disk space in run directory",
        InterventionType.MEMORY_EXHAUSTED: "Kill memory-intensive processes",
        InterventionType.IMPORT_ERROR: "pip install <missing_package>",
        InterventionType.DEPENDENCY_MISSING: "Check requirements.txt",
        InterventionType.CONFIG_MISSING: "Regenerate config file",
        InterventionType.CONFIG_INVALID: "Validate config syntax",
        InterventionType.CONTEXT_PACK_MISSING: "Check context_packs/",
        InterventionType.SECURITY_BLOCKED: "Review with human - security decision required",
        InterventionType.QUIETSTAR_BLOCKED: "Review with human - safety decision required",
    }
    return suggestions.get(intervention_type, "Investigate error details")


async def retry_with_run_master(
    operation: AsyncOperation[T],
    operation_name: str,
    state: "PipelineState",
    node: str,
    intervention_type: Optional[InterventionType] = None,
    retry_config: RetryConfig = RETRY_TRANSIENT,
    rm_timeout_seconds: float = 300.0,
    rm_poll_interval_seconds: float = 5.0,
) -> T:
    """Execute operation with retry and Run Master escalation.

    This is the MAIN resilience function. Philosophy:
        1. Try operation with exponential backoff retries
        2. If all retries fail, escalate to Run Master
        3. WAIT for Run Master to resolve the issue
        4. Try ONE MORE TIME after resolution
        5. If still fails, raise with full context

    Args:
        operation: Async operation to execute.
        operation_name: Name for logging/metrics.
        state: Pipeline state (for run_dir, sprint_id, etc).
        node: Which workflow node is executing this.
        intervention_type: Override auto-detection of issue type.
        retry_config: Retry configuration.
        rm_timeout_seconds: Max time to wait for Run Master.
        rm_poll_interval_seconds: How often to poll for resolution.

    Returns:
        Result of successful operation.

    Raises:
        EscalationError: If Run Master cannot resolve.
        Exception: The final exception after all recovery attempts.

    Example:
        result = await retry_with_run_master(
            operation=lambda: redis_client.get("key"),
            operation_name="redis_get",
            state=state,
            node="exec",
        )
    """
    exceptions: List[Exception] = []
    total_delay = 0.0
    total_attempts = retry_config.max_retries + 1

    # Phase 1: Standard retry with exponential backoff
    for attempt in range(total_attempts):
        try:
            return await operation()
        except retry_config.retryable_exceptions as e:
            exceptions.append(e)

            if attempt < retry_config.max_retries:
                delay = calculate_delay(attempt, retry_config)
                total_delay += delay

                logger.warning(
                    "RETRY: operation='%s' attempt=%d/%d error='%s' delay=%.2fs",
                    operation_name,
                    attempt + 1,
                    total_attempts,
                    str(e)[:100],
                    delay,
                )

                await asyncio.sleep(delay)

    # Phase 2: Retries exhausted - escalate to Run Master
    logger.error(
        "RETRY_EXHAUSTED_ESCALATING: operation='%s' attempts=%d",
        operation_name,
        total_attempts,
    )

    # Get run directory from state
    run_dir = Path(state.get("run_dir", "/tmp/pipeline_run"))
    sprint_id = state.get("sprint_id", "unknown")
    run_id = state.get("run_id", "unknown")

    # Infer intervention type if not provided
    if intervention_type is None:
        intervention_type = _infer_intervention_type(
            exceptions[-1],
            operation_name,
        )

    # Create intervention request
    request = InterventionRequest(
        intervention_type=intervention_type,
        error_message=str(exceptions[-1])[:500],
        node=node,
        sprint_id=sprint_id,
        run_id=run_id,
        retry_attempts=total_attempts,
        total_retry_time_seconds=total_delay,
        exceptions=[str(e)[:200] for e in exceptions[-5:]],
        suggested_action=_suggest_action(intervention_type),
    )

    # Send to Run Master
    await _send_intervention_request(request, run_dir)

    # Phase 3: Wait for resolution
    try:
        resolution = await _wait_for_resolution(
            request,
            run_dir,
            timeout_seconds=rm_timeout_seconds,
            poll_interval_seconds=rm_poll_interval_seconds,
        )
    except EscalationError:
        # Timeout waiting for Run Master
        raise

    # Phase 4: Resolution received - try one more time
    if resolution.resolved and resolution.should_retry:
        logger.info(
            "POST_RESOLUTION_RETRY: operation='%s' action='%s'",
            operation_name,
            resolution.action_taken,
        )

        try:
            return await operation()
        except Exception as e:
            raise EscalationError(
                f"Operation '{operation_name}' failed after Run Master resolution: {e}",
                request=request,
                resolution=resolution,
            ) from e

    elif resolution.resolved and not resolution.should_retry:
        # Resolution says don't retry (e.g., permanent failure)
        raise EscalationError(
            f"Run Master resolved but advises not to retry: {resolution.error_message}",
            request=request,
            resolution=resolution,
        )

    else:
        # Resolution failed
        raise EscalationError(
            f"Run Master could not resolve: {resolution.error_message}",
            request=request,
            resolution=resolution,
        )


async def retry_and_continue(
    operation: AsyncOperation[T],
    operation_name: str,
    state: "PipelineState",
    node: str,
    default_result: T,
    retry_config: RetryConfig = RETRY_TRANSIENT,
) -> T:
    """Execute operation with retry, continue on failure.

    Unlike retry_with_run_master, this function:
        1. Tries the operation with retries
        2. If all retries fail, CONTINUES with default result
        3. Escalates to Run Master in BACKGROUND (non-blocking)

    Use this for non-critical operations like:
        - Observability (Langfuse, Phoenix)
        - Optional integrations
        - Metrics collection
        - Documentation generation

    Args:
        operation: Async operation to execute.
        operation_name: Name for logging.
        state: Pipeline state.
        node: Workflow node name.
        default_result: Value to return if operation fails.
        retry_config: Retry configuration.

    Returns:
        Operation result or default_result on failure.

    Example:
        # Log to Langfuse, continue if it fails
        trace_id = await retry_and_continue(
            operation=lambda: langfuse.trace(...),
            operation_name="langfuse_trace",
            state=state,
            node="exec",
            default_result=None,
        )
    """
    exceptions: List[Exception] = []
    total_attempts = retry_config.max_retries + 1

    for attempt in range(total_attempts):
        try:
            return await operation()
        except retry_config.retryable_exceptions as e:
            exceptions.append(e)

            if attempt < retry_config.max_retries:
                delay = calculate_delay(attempt, retry_config)

                logger.debug(
                    "RETRY_CONTINUE: operation='%s' attempt=%d/%d error='%s'",
                    operation_name,
                    attempt + 1,
                    total_attempts,
                    str(e)[:100],
                )

                await asyncio.sleep(delay)

    # All retries failed - log and continue
    logger.warning(
        "CONTINUE_WITH_DEFAULT: operation='%s' failed after %d attempts, using default",
        operation_name,
        total_attempts,
    )

    # Background escalation (non-blocking)
    run_dir = Path(state.get("run_dir", "/tmp/pipeline_run"))
    sprint_id = state.get("sprint_id", "unknown")
    run_id = state.get("run_id", "unknown")

    request = InterventionRequest(
        intervention_type=_infer_intervention_type(exceptions[-1], operation_name),
        error_message=str(exceptions[-1])[:500],
        node=node,
        sprint_id=sprint_id,
        run_id=run_id,
        retry_attempts=total_attempts,
        exceptions=[str(e)[:200] for e in exceptions[-5:]],
        suggested_action=_suggest_action(
            _infer_intervention_type(exceptions[-1], operation_name)
        ),
    )

    # Fire and forget - don't block pipeline
    asyncio.create_task(
        _send_intervention_request(request, run_dir)
    )

    return default_result


__all__ = [
    "InterventionType",
    "InterventionRequest",
    "InterventionResolution",
    "EscalationError",
    "retry_with_run_master",
    "retry_and_continue",
]
