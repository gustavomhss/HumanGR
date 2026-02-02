"""Claude Execution Guard - Singleton enforcement for daemon pattern.

This module provides a SINGLE entry point for all Claude CLI executions.
It enforces the daemon pattern mechanically - no execution can happen
without going through this guard.

PAT-031: Mechanical enforcement of daemon pattern.

Usage:
    from pipeline.claude_execution_guard import (
        get_execution_guard,
        execute_with_guard,
        GuardedExecution,
    )

    # Option 1: Use the guard directly
    guard = get_execution_guard()
    result = guard.execute(prompt="...", context="...")

    # Option 2: Use convenience function
    result = execute_with_guard(prompt="...", context="...")

    # Option 3: Use context manager for tracking
    with GuardedExecution("my_task") as exec:
        result = exec.run(prompt="...")

Architecture:
    - SINGLETON guard instance
    - All executions logged and tracked
    - Daemon flags ALWAYS applied (not configurable)
    - Pre/post execution hooks for validation
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Execution modes for Claude CLI."""

    DAEMON = "daemon"  # Full access (Write/Edit/Bash) - REQUIRED
    # NO OTHER MODES ALLOWED - daemon is mandatory


class ExecutionStatus(Enum):
    """Status of an execution."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    BLOCKED = "blocked"  # Blocked by guard


@dataclass
class ExecutionRecord:
    """Record of a Claude execution."""

    execution_id: str
    task_name: str
    caller: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    status: ExecutionStatus = ExecutionStatus.PENDING
    daemon_verified: bool = False
    result_summary: str = ""
    error: Optional[str] = None


@dataclass
class GuardConfig:
    """Configuration for execution guard."""

    # Logging
    log_all_executions: bool = True
    log_path: str = field(
        default_factory=lambda: os.getenv(
            "CLAUDE_GUARD_LOG_PATH",
            "/tmp/claude_execution_guard.log"
        )
    )

    # Limits
    max_concurrent_executions: int = 5
    execution_timeout_seconds: int = 600  # 10 min

    # Hooks
    pre_execution_hooks: list[Callable[[str, str], bool]] = field(default_factory=list)
    post_execution_hooks: list[Callable[[ExecutionRecord], None]] = field(default_factory=list)


class DaemonNotActiveError(Exception):
    """Raised when daemon pattern is not properly configured."""
    pass


class GuardViolationError(Exception):
    """Raised when execution guard rules are violated."""
    pass


class ClaudeExecutionGuard:
    """Singleton guard for all Claude CLI executions.

    This class ensures:
    1. All Claude executions go through a single point
    2. Daemon flags are ALWAYS applied
    3. All executions are logged and tracked
    4. Concurrent execution limits are enforced

    IMPORTANT: This class should NOT be instantiated directly.
    Use get_execution_guard() to get the singleton instance.
    """

    _instance: Optional["ClaudeExecutionGuard"] = None
    _lock = threading.Lock()
    _instantiation_allowed = False

    def __new__(cls, config: Optional[GuardConfig] = None) -> "ClaudeExecutionGuard":
        """Create or return singleton instance."""
        if not cls._instantiation_allowed:
            raise GuardViolationError(
                "Direct instantiation not allowed. Use get_execution_guard() instead. "
                "This ensures singleton pattern and proper initialization."
            )

        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, config: Optional[GuardConfig] = None) -> None:
        """Initialize the guard."""
        if getattr(self, "_initialized", False):
            return

        self._config = config or GuardConfig()
        self._execution_history: list[ExecutionRecord] = []
        self._active_executions: dict[str, ExecutionRecord] = {}
        self._execution_lock = threading.Lock()
        self._adapter: Optional[Any] = None  # Lazy loaded
        self._initialized = True

        logger.info("ClaudeExecutionGuard initialized - all executions will be guarded")

    @property
    def config(self) -> GuardConfig:
        """Get current configuration."""
        return self._config

    def _get_adapter(self) -> Any:
        """Get or create the Claude CLI adapter (lazy loading)."""
        if self._adapter is None:
            from .claude_cli_llm import ClaudeCLIAdapter
            self._adapter = ClaudeCLIAdapter()
        return self._adapter

    def _verify_daemon_active(self) -> bool:
        """Verify that daemon pattern is properly configured.

        This checks that the adapter has the required daemon flags.
        Returns True if daemon is active, raises DaemonNotActiveError otherwise.
        """
        adapter = self._get_adapter()

        # Check that _execute_cli exists and has daemon flags
        # This is a sanity check - the flags are hardcoded in the adapter
        import inspect
        source = inspect.getsource(adapter._execute_cli)

        required_flags = [
            "--dangerously-skip-permissions",
            "--print",
        ]

        for flag in required_flags:
            if flag not in source:
                raise DaemonNotActiveError(
                    f"CRITICAL: Daemon flag '{flag}' not found in adapter. "
                    "This is a severe security issue - execution blocked."
                )

        return True

    def _check_execution_limits(self) -> None:
        """Check if we can start a new execution."""
        with self._execution_lock:
            active_count = len(self._active_executions)
            if active_count >= self._config.max_concurrent_executions:
                raise GuardViolationError(
                    f"Max concurrent executions reached ({active_count}). "
                    f"Limit: {self._config.max_concurrent_executions}"
                )

    def _run_pre_hooks(self, prompt: str, context: str) -> bool:
        """Run pre-execution hooks. Return False to block execution."""
        for hook in self._config.pre_execution_hooks:
            try:
                if not hook(prompt, context):
                    return False
            except Exception as e:
                logger.warning(f"Pre-execution hook failed: {e}")
                return False
        return True

    def _run_post_hooks(self, record: ExecutionRecord) -> None:
        """Run post-execution hooks."""
        for hook in self._config.post_execution_hooks:
            try:
                hook(record)
            except Exception as e:
                logger.warning(f"Post-execution hook failed: {e}")

    def _log_execution(self, record: ExecutionRecord) -> None:
        """Log execution to file."""
        if not self._config.log_all_executions:
            return

        try:
            log_line = (
                f"{record.started_at.isoformat()} | "
                f"{record.execution_id} | "
                f"{record.status.value} | "
                f"{record.task_name} | "
                f"{record.caller}\n"
            )

            with open(self._config.log_path, "a") as f:
                f.write(log_line)
        except Exception as e:
            logger.warning(f"Failed to log execution: {e}")

    def execute(
        self,
        prompt: str,
        context: str = "",
        task_name: str = "unnamed_task",
        caller: str = "unknown",
        output_format: str = "text",
        json_schema: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Execute a Claude CLI command through the guard.

        This is the ONLY authorized way to execute Claude commands.

        Args:
            prompt: The prompt to send to Claude.
            context: Additional context/system prompt.
            task_name: Name of the task for tracking.
            caller: Identifier of the calling code.
            output_format: Output format ("text" or "json").
            json_schema: JSON schema for structured output.
            session_id: Optional session ID for context persistence.

        Returns:
            The response from Claude.

        Raises:
            DaemonNotActiveError: If daemon pattern is not active.
            GuardViolationError: If guard rules are violated.
        """
        execution_id = str(uuid4())
        record = ExecutionRecord(
            execution_id=execution_id,
            task_name=task_name,
            caller=caller,
            started_at=datetime.now(timezone.utc),
        )

        try:
            # 1. Verify daemon is active
            self._verify_daemon_active()
            record.daemon_verified = True

            # 2. Check execution limits
            self._check_execution_limits()

            # 3. Run pre-execution hooks
            if not self._run_pre_hooks(prompt, context):
                record.status = ExecutionStatus.BLOCKED
                record.error = "Blocked by pre-execution hook"
                raise GuardViolationError(record.error)

            # 4. Register active execution
            with self._execution_lock:
                self._active_executions[execution_id] = record

            record.status = ExecutionStatus.RUNNING

            # 5. Execute through adapter
            adapter = self._get_adapter()

            if output_format == "json" and json_schema:
                result = adapter._execute_cli(
                    prompt=prompt,
                    system_prompt=context if context else None,
                    output_format="json",
                    json_schema=json_schema,
                    session_id=session_id,
                )
            else:
                result = adapter._execute_cli(
                    prompt=prompt,
                    system_prompt=context if context else None,
                    output_format="text",
                    session_id=session_id,
                )

            # 6. Success
            record.status = ExecutionStatus.SUCCESS
            record.result_summary = result[:200] if result else ""

            return result

        except (DaemonNotActiveError, GuardViolationError):
            raise

        except Exception as e:
            record.status = ExecutionStatus.FAILED
            record.error = str(e)
            raise

        finally:
            # Clean up and log
            record.ended_at = datetime.now(timezone.utc)

            with self._execution_lock:
                self._active_executions.pop(execution_id, None)
                self._execution_history.append(record)

                # Keep only last 1000 records
                if len(self._execution_history) > 1000:
                    self._execution_history = self._execution_history[-1000:]

            self._log_execution(record)
            self._run_post_hooks(record)

    def get_execution_stats(self) -> dict[str, Any]:
        """Get execution statistics."""
        with self._execution_lock:
            total = len(self._execution_history)
            success = sum(1 for r in self._execution_history if r.status == ExecutionStatus.SUCCESS)
            failed = sum(1 for r in self._execution_history if r.status == ExecutionStatus.FAILED)
            blocked = sum(1 for r in self._execution_history if r.status == ExecutionStatus.BLOCKED)

            return {
                "total_executions": total,
                "successful": success,
                "failed": failed,
                "blocked": blocked,
                "active": len(self._active_executions),
                "success_rate": success / total if total > 0 else 0,
            }

    def get_recent_executions(self, limit: int = 10) -> list[ExecutionRecord]:
        """Get recent execution records."""
        with self._execution_lock:
            return list(self._execution_history[-limit:])


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_guard: Optional[ClaudeExecutionGuard] = None
_guard_lock = threading.Lock()


def get_execution_guard(config: Optional[GuardConfig] = None) -> ClaudeExecutionGuard:
    """Get the singleton execution guard instance.

    This is the ONLY way to get a ClaudeExecutionGuard instance.
    Direct instantiation is blocked.

    Args:
        config: Optional configuration (only used on first call).

    Returns:
        The singleton ClaudeExecutionGuard instance.
    """
    global _guard

    with _guard_lock:
        if _guard is None:
            # Temporarily allow instantiation
            ClaudeExecutionGuard._instantiation_allowed = True
            try:
                _guard = ClaudeExecutionGuard(config)
            finally:
                ClaudeExecutionGuard._instantiation_allowed = False

        return _guard


def reset_execution_guard() -> None:
    """Reset the guard instance (for testing only)."""
    global _guard
    with _guard_lock:
        _guard = None
        ClaudeExecutionGuard._instance = None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def execute_with_guard(
    prompt: str,
    context: str = "",
    task_name: str = "unnamed_task",
    caller: str = "unknown",
    output_format: str = "text",
    json_schema: Optional[dict[str, Any]] = None,
) -> str:
    """Convenience function to execute through the guard.

    Args:
        prompt: The prompt to send to Claude.
        context: Additional context/system prompt.
        task_name: Name of the task for tracking.
        caller: Identifier of the calling code.
        output_format: Output format ("text" or "json").
        json_schema: JSON schema for structured output.

    Returns:
        The response from Claude.
    """
    guard = get_execution_guard()
    return guard.execute(
        prompt=prompt,
        context=context,
        task_name=task_name,
        caller=caller,
        output_format=output_format,
        json_schema=json_schema,
    )


@contextmanager
def GuardedExecution(task_name: str, caller: str = "unknown"):
    """Context manager for guarded execution.

    Usage:
        with GuardedExecution("my_task", "my_module") as exec:
            result = exec.run(prompt="...")
    """
    guard = get_execution_guard()

    class ExecutionContext:
        def __init__(self):
            self.task_name = task_name
            self.caller = caller

        def run(
            self,
            prompt: str,
            context: str = "",
            output_format: str = "text",
            json_schema: Optional[dict[str, Any]] = None,
        ) -> str:
            return guard.execute(
                prompt=prompt,
                context=context,
                task_name=self.task_name,
                caller=self.caller,
                output_format=output_format,
                json_schema=json_schema,
            )

    yield ExecutionContext()


# =============================================================================
# VALIDATION ASSERTION (for runtime checks)
# =============================================================================


def assert_daemon_active() -> None:
    """Assert that daemon pattern is active.

    Call this at critical points to verify daemon is working.
    Raises DaemonNotActiveError if not active.
    """
    guard = get_execution_guard()
    guard._verify_daemon_active()


def assert_guard_healthy() -> None:
    """Assert that the execution guard is healthy.

    Raises GuardViolationError if guard is not properly initialized.
    """
    guard = get_execution_guard()

    if not guard._initialized:
        raise GuardViolationError("Guard not properly initialized")

    stats = guard.get_execution_stats()

    # Check for high failure rate
    if stats["total_executions"] > 10 and stats["success_rate"] < 0.5:
        logger.warning(f"Low execution success rate: {stats['success_rate']:.2%}")
