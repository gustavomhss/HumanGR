"""Claude CLI LLM adapter for Pipeline Autonomo v2.0.

This module provides an adapter to use Claude CLI (claude command)
as the LLM backend for CrewAI and Letta frameworks.

Architecture:
    - PAT-028 FIX: Uses --dangerously-skip-permissions for full agent mode
    - Agents have full filesystem access to write code
    - Supports JSON output format for structured responses
    - Manages rate limiting to respect Claude Max limits
    - Thread-safe for concurrent agent usage

Key Features:
    - Zero API keys required (uses Claude Max subscription)
    - Compatible with CrewAI LLM interface
    - Supports streaming and non-streaming modes
    - Automatic retry with backoff

Environment Variables:
    - CLAUDE_CLI_PATH: Path to claude CLI (default: claude)
    - CLAUDE_MODEL: Model to use (default: claude-opus-4-5-20251101)
    - CLAUDE_MAX_TOKENS: Max output tokens (default: 4096)
    - CLAUDE_TIMEOUT: Request timeout in seconds (default: 120)

Author: Pipeline Autonomo Team
Version: 2.0.0 (2025-12-29)
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

from pydantic import BaseModel

import logging

# Security integration for LLM call protection
try:
    from pipeline.security.llm_guard_integration import (
        secure_operation as _async_secure_operation,
        SecurityLevel,
        SecurityBlockedError,
        get_security_orchestrator,
        get_llm_guard_integration,
    )
    SECURITY_INTEGRATION_AVAILABLE = True

    def secure_operation(
        level: str = "standard",
        scan_input: bool = True,
        scan_output: bool = True,
    ):
        """Decorator to secure LLM operations (sync-compatible wrapper).

        This wrapper provides security scanning for synchronous LLM calls by:
        1. Running async security checks in a sync context using asyncio
        2. Scanning input text for prompt injection, PII, toxicity
        3. Scanning output text for sensitive data leakage

        Args:
            level: Security level ("minimal", "standard", "strict", "paranoid")
            scan_input: Whether to scan input for security issues
            scan_output: Whether to scan output for security issues

        Returns:
            Decorated function with security wrapping
        """
        from functools import wraps

        # Map string level to SecurityLevel enum
        level_map = {
            "minimal": SecurityLevel.MINIMAL,
            "standard": SecurityLevel.STANDARD,
            "strict": SecurityLevel.STRICT,
            "high": SecurityLevel.STRICT,  # Map "high" to STRICT
            "paranoid": SecurityLevel.PARANOID,
        }
        security_level = level_map.get(level, SecurityLevel.STANDARD)

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                import asyncio  # Import at function scope to ensure availability
                import concurrent.futures

                # Get input text from args/kwargs
                input_text = ""
                if args:
                    first_arg = args[0] if not hasattr(args[0], '__self__') else (args[1] if len(args) > 1 else "")
                    if isinstance(first_arg, str):
                        input_text = first_arg
                    elif hasattr(first_arg, '__iter__') and not isinstance(first_arg, (str, bytes)):
                        # Handle list of messages
                        for msg in first_arg:
                            if hasattr(msg, 'get'):
                                input_text += msg.get('content', '') + " "
                            elif hasattr(msg, 'content'):
                                input_text += str(msg.content) + " "

                # Run input security check synchronously
                if scan_input and input_text:
                    try:
                        orchestrator = get_security_orchestrator(security_level)

                        # Run async check in sync context
                        try:
                            asyncio.get_running_loop()
                            # Loop is running - use thread to avoid blocking
                            with concurrent.futures.ThreadPoolExecutor() as pool:
                                result = pool.submit(
                                    asyncio.run,
                                    orchestrator.secure_operation(input_text[:5000])
                                ).result(timeout=30)
                        except RuntimeError:
                            # No running loop - safe to use asyncio.run
                            result = asyncio.run(orchestrator.secure_operation(input_text[:5000]))

                        if result.blocked:
                            raise SecurityBlockedError(
                                f"SECURITY: Input blocked - {result.block_reason}",
                                result=result,
                            )

                        logger.debug(f"SECURITY: Input scan passed (risk={result.overall_risk_score:.2f})")
                    except SecurityBlockedError:
                        raise
                    except Exception as e:
                        logger.warning(f"SECURITY: Input scan failed (continuing): {e}")

                # Execute the original function
                output = func(*args, **kwargs)

                # Run output security check
                if scan_output and output and isinstance(output, str):
                    try:
                        orchestrator = get_security_orchestrator(security_level)

                        try:
                            asyncio.get_running_loop()
                            # Loop is running - use thread to avoid blocking
                            with concurrent.futures.ThreadPoolExecutor() as pool:
                                result = pool.submit(
                                    asyncio.run,
                                    orchestrator.secure_operation(
                                        input_text[:2000], output[:5000], operation_type="output_validation"
                                    )
                                ).result(timeout=30)
                        except RuntimeError:
                            # No running loop - safe to use asyncio.run
                            result = asyncio.run(
                                orchestrator.secure_operation(
                                    input_text[:2000], output[:5000], operation_type="output_validation"
                                )
                            )

                        if result.blocked:
                            raise SecurityBlockedError(
                                f"SECURITY: Output blocked - {result.block_reason}",
                                result=result,
                            )

                        logger.debug(f"SECURITY: Output scan passed (risk={result.overall_risk_score:.2f})")
                    except SecurityBlockedError:
                        raise
                    except Exception as e:
                        logger.warning(f"SECURITY: Output scan failed (continuing): {e}")

                return output

            return wrapper
        return decorator

except ImportError:
    SECURITY_INTEGRATION_AVAILABLE = False
    # Provide no-op decorator if security module not available
    def secure_operation(level: str = "standard", scan_input: bool = True, scan_output: bool = True):
        """No-op decorator when security module not available."""
        def decorator(func):
            return func
        return decorator

    class SecurityBlockedError(Exception):
        """Placeholder exception when security not available."""
        pass


# =============================================================================
# GUARDRAILS INTEGRATION (Extracted to claude_cli_guardrails.py)
# =============================================================================
# D-HIGH-049: Extraction to reduce file size and improve modularity
from pipeline.claude_cli_guardrails import (
    # NeMo
    NEMO_GUARDRAILS_AVAILABLE,
    NeMoGuardrailsError,
    check_nemo_guardrails_input as _check_nemo_guardrails_input,
    check_nemo_guardrails_output as _check_nemo_guardrails_output,
    # QuietStar
    QUIETSTAR_REFLEXION_AVAILABLE,
    QuietStarBlockedError,
    get_or_create_quietstar_guardrail as _get_or_create_quietstar_guardrail,
    check_quietstar_pre_generation as _check_quietstar_pre_generation,
    check_quietstar_post_generation as _check_quietstar_post_generation,
)

# Module logger - used throughout for debugging and error reporting
logger = logging.getLogger(__name__)


# Langfuse integration for LLM observability
try:
    from pipeline.langfuse_tracer import (
        track_generation,
        get_current_trace,
        set_current_trace,
    )
    from pipeline.langfuse_client import get_langfuse_client
    LANGFUSE_AVAILABLE = True
    logger.info("Langfuse integration enabled for LLM observability")
except ImportError:
    LANGFUSE_AVAILABLE = False
    logger.debug("Langfuse not available - LLM calls won't be tracked")

    def track_generation(*args, **kwargs):
        return None
    def get_current_trace():
        return None
    def set_current_trace(trace_id):
        pass
    def get_langfuse_client():
        return None

# Grafana metrics integration for Pipeline Control Center dashboard
try:
    from pipeline.grafana_metrics import (
        get_metrics_publisher,
        GrafanaMetricsPublisher,
    )
    GRAFANA_METRICS_AVAILABLE = True
except ImportError:
    GRAFANA_METRICS_AVAILABLE = False
    get_metrics_publisher = None
    GrafanaMetricsPublisher = None

# Type variable for generic return types
T = TypeVar("T")


def _validate_cli_path(path: str) -> str:
    """RED TEAM FIX D-04: Validate CLI path is not a wrapper script.

    Args:
        path: Path to CLI binary.

    Returns:
        Validated path.

    Raises:
        ValueError: If path appears to be a wrapper or unsafe.
    """
    import shutil

    # Resolve to absolute path
    resolved = shutil.which(path)
    if not resolved:
        raise ValueError(f"Claude CLI not found: {path}")

    # Check it's not a shell script (potential wrapper)
    try:
        with open(resolved, "rb") as f:
            header = f.read(128)
            # Check for shebang indicating shell script
            if header.startswith(b"#!") and any(s in header for s in [b"/bin/sh", b"/bin/bash", b"/bin/zsh", b"python"]):
                raise ValueError(
                    f"D-04 VIOLATION: {resolved} appears to be a wrapper script. "
                    "Use the actual Claude CLI binary, not a wrapper."
                )
    except (IOError, PermissionError) as e:
        logger.warning(f"Cannot validate CLI binary {resolved}: {e}")

    return path


@dataclass
class ClaudeCLIConfig:
    """Configuration for Claude CLI adapter."""

    cli_path: str = field(default_factory=lambda: _validate_cli_path(os.getenv("CLAUDE_CLI_PATH", "claude")))
    model: str = field(
        default_factory=lambda: os.getenv("CLAUDE_MODEL", "claude-opus-4-5-20251101")
    )
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv("CLAUDE_MAX_TOKENS", "4096"))
    )
    timeout_seconds: Optional[int] = field(
        default_factory=lambda: None  # No timeout - let Claude run as long as needed
    )  # User preference: no timeout, just status poll
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_per_minute: int = 20  # Claude Max limit
    # P0-04 FIX (2026-01-30): Enable cache by default for performance
    # Rollback: Set CLAUDE_CACHE_ENABLED=false to disable
    cache_enabled: bool = field(
        default_factory=lambda: os.getenv("CLAUDE_CACHE_ENABLED", "true").lower() != "false"
    )
    # P0-04 FIX: Cache size limits to prevent unbounded growth
    MAX_CACHE_SIZE_MB: int = 500
    MAX_CACHE_ENTRIES: int = 10000
    cache_dir: str = field(
        default_factory=lambda: os.getenv(
            "CLAUDE_CACHE_DIR",
            str(Path.home() / ".cache" / "humangr" / "claude_cli"),
        )
    )
    cache_ttl_seconds: int = field(
        default_factory=lambda: int(os.getenv("CLAUDE_CACHE_TTL_SECONDS", "0"))
    )


class ClaudeCLIError(Exception):
    """Raised when Claude CLI execution fails."""

    pass


class ClaudeCLIRateLimitError(ClaudeCLIError):
    """Raised when rate limit is exceeded."""

    pass


class ClaudeCLITimeoutError(ClaudeCLIError):
    """Raised when request times out."""

    pass


class SecurityError(ClaudeCLIError):
    """Raised when a security check fails.

    CRIT-005 FIX: Used to signal security violations during path validation
    and other security-critical operations.
    """

    pass


class Message(BaseModel):
    """A single message in a conversation."""

    role: str  # "user" or "assistant"
    content: str


class CompletionResponse(BaseModel):
    """Response from Claude CLI completion."""

    content: str
    model: str
    stop_reason: Optional[str] = None
    usage: Optional[dict[str, int]] = None
    # 2026-01-10 FIX: Use None default and set in __init__ to get fresh timestamp
    # BEFORE: timestamp: datetime = datetime.now(timezone.utc)
    # BUG (D-HIGH-055): Default was evaluated ONCE at class definition, not per instance
    # Issue: Pydantic field defaults are evaluated at class definition time, not instantiation
    # Tracking: See CONSOLIDATED_GAPS_MASTER.md D-HIGH-055
    timestamp: Optional[datetime] = None
    raw_output: Optional[str] = None

    def __init__(self, **data: Any) -> None:
        """Initialize with fresh timestamp if not provided."""
        if "timestamp" not in data or data["timestamp"] is None:
            data["timestamp"] = datetime.now(timezone.utc)
        super().__init__(**data)


import logging
import re

# Logger for code extraction
_extract_logger = logging.getLogger(__name__)


def _validate_path_security(file_path: str, base: Path) -> tuple[bool, str]:
    """CRIT-005 FIX: Validate file path for path traversal and symlink attacks.

    Args:
        file_path: The relative file path to validate.
        base: The base directory that files must stay within.

    Returns:
        Tuple of (is_safe, error_message). is_safe=True if path is safe.
    """
    # 1. Check for null bytes (common injection technique)
    if '\x00' in file_path:
        return False, "CRIT-005: Null byte detected in path"

    # 2. Normalize and check for path traversal patterns
    # Block explicit traversal patterns
    dangerous_patterns = [
        '..',           # Parent directory traversal
        '~',            # Home directory expansion
        '$',            # Variable expansion
        '${',           # Variable expansion
        '`',            # Command substitution
        '|',            # Pipe
        ';',            # Command separator
        '&',            # Background/AND
        '>',            # Redirect
        '<',            # Redirect
        '\n',           # Newline
        '\r',           # Carriage return
    ]

    for pattern in dangerous_patterns:
        if pattern in file_path:
            return False, f"CRIT-005: Dangerous pattern '{pattern}' detected in path"

    # 3. Resolve the full path and verify it's within base directory
    try:
        # Use resolve() to get canonical path (resolves symlinks)
        full_path = (base / file_path).resolve()
        base_resolved = base.resolve()

        # Check if the resolved path is within the base directory
        try:
            full_path.relative_to(base_resolved)
        except ValueError:
            return False, f"CRIT-005: Path traversal detected - path escapes base directory"

    except Exception as e:
        return False, f"CRIT-005: Path resolution failed: {e}"

    # 4. Check for symlink attacks - ensure parent directories aren't symlinks
    # that could redirect writes outside the base directory
    try:
        current = base
        for part in Path(file_path).parts[:-1]:  # Check all parent directories
            current = current / part
            if current.exists() and current.is_symlink():
                # Verify symlink target is within base
                target = current.resolve()
                try:
                    target.relative_to(base_resolved)
                except ValueError:
                    return False, f"CRIT-005: Symlink escape detected at {current}"
    except Exception as e:
        return False, f"CRIT-005: Symlink validation failed: {e}"

    # 5. Block writes to sensitive system paths even if somehow within base
    sensitive_paths = [
        '/etc/', '/var/', '/usr/', '/bin/', '/sbin/',
        '/root/', '/home/', '/tmp/', '/dev/', '/proc/', '/sys/',
        '.ssh/', '.gnupg/', '.aws/', '.config/',
        '__pycache__/', '.git/hooks/', '.github/workflows/',
    ]

    normalized_path = file_path.lower().replace('\\', '/')
    for sensitive in sensitive_paths:
        if sensitive in normalized_path:
            return False, f"CRIT-005: Blocked write to sensitive path containing '{sensitive}'"

    return True, ""


def extract_and_write_code_blocks(
    output: str,
    base_dir: Optional[str] = None,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """
    PAT-028 FIX: Extract code blocks from Claude output and write to files.

    CRIT-005 FIX (2026-01-22): Added comprehensive path validation to prevent:
    - Path traversal attacks (../)
    - Symlink escape attacks
    - Writes to sensitive system directories
    - Null byte injection
    - Command injection in paths

    This function parses Claude's text output for code blocks with file path
    annotations and writes them to disk. This enables code generation even
    in --print mode where tool calls aren't possible.

    Supported file path patterns:
    - # FILE: path/to/file.py
    - # Path: path/to/file.py
    - # Filename: file.py
    - <!-- FILE: path/to/file.md -->

    Args:
        output: Raw text output from Claude.
        base_dir: Base directory for file paths. Defaults to cwd.
        dry_run: If True, don't write files, just return what would be written.

    Returns:
        List of dicts with {path, content, written, error} for each file.

    Example:
        >>> text = '''
        ... Here's the code:
        ... ```python
        ... # FILE: src/models.py
        ... class User:
        ...     pass
        ... ```
        ... '''
        >>> results = extract_and_write_code_blocks(text, dry_run=True)
        >>> results[0]['path']
        'src/models.py'
    """
    results: list[dict[str, Any]] = []
    base = Path(base_dir) if base_dir else Path.cwd()

    # CRIT-005 FIX: Ensure base directory is absolute and resolved
    base = base.resolve()

    # Pattern to match code blocks with optional language
    code_block_pattern = re.compile(
        r'```(\w+)?\s*\n(.*?)```',
        re.DOTALL
    )

    # Patterns to extract file paths from code block content
    file_path_patterns = [
        re.compile(r'^#\s*FILE:\s*(.+)$', re.MULTILINE | re.IGNORECASE),
        re.compile(r'^#\s*Path:\s*(.+)$', re.MULTILINE | re.IGNORECASE),
        re.compile(r'^#\s*Filename:\s*(.+)$', re.MULTILINE | re.IGNORECASE),
        re.compile(r'<!--\s*FILE:\s*(.+?)\s*-->', re.IGNORECASE),
        # Also match path-like first line comments
        re.compile(r'^#\s*(src/[^\s]+\.py)$', re.MULTILINE),
        re.compile(r'^#\s*(tests/[^\s]+\.py)$', re.MULTILINE),
    ]

    for match in code_block_pattern.finditer(output):
        lang = match.group(1) or ""
        content = match.group(2)

        # Try to extract file path
        file_path = None
        for pattern in file_path_patterns:
            path_match = pattern.search(content)
            if path_match:
                file_path = path_match.group(1).strip()
                # Remove the path comment from content
                content = pattern.sub('', content, count=1).strip()
                break

        if not file_path:
            # Skip code blocks without file paths
            continue

        # CRIT-005 FIX: Normalize path - strip leading slashes but preserve structure
        file_path = file_path.lstrip('/')

        # CRIT-005 FIX: Validate path security BEFORE any file operations
        is_safe, error_msg = _validate_path_security(file_path, base)
        if not is_safe:
            _extract_logger.error(f"SECURITY BLOCKED: {error_msg} for path: {file_path}")
            results.append({
                'path': str(file_path),
                'full_path': 'BLOCKED',
                'content': content,
                'language': lang,
                'written': False,
                'error': error_msg,
                'security_blocked': True,
            })
            continue

        full_path = (base / file_path).resolve()

        result: dict[str, Any] = {
            'path': str(file_path),
            'full_path': str(full_path),
            'content': content,
            'language': lang,
            'written': False,
            'error': None,
            'security_blocked': False,
        }

        if dry_run:
            result['written'] = False
            results.append(result)
            continue

        try:
            # CRIT-005 FIX: Final security check - verify resolved path is still within base
            try:
                full_path.relative_to(base)
            except ValueError:
                raise SecurityError(f"CRIT-005: Final path check failed - {full_path} outside {base}")

            # Create parent directories
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # CRIT-005 FIX: Verify parent is not a symlink after creation
            if full_path.parent.is_symlink():
                parent_target = full_path.parent.resolve()
                try:
                    parent_target.relative_to(base)
                except ValueError:
                    raise SecurityError(f"CRIT-005: Parent symlink escapes base directory")

            # Write file
            full_path.write_text(content + '\n')
            result['written'] = True
            _extract_logger.info(f"PAT-028 FIX: Wrote {file_path} ({len(content)} bytes)")

        except SecurityError as e:
            result['error'] = str(e)
            result['security_blocked'] = True
            _extract_logger.error(f"SECURITY BLOCKED: {e}")
        except Exception as e:
            result['error'] = str(e)
            _extract_logger.error(f"PAT-028 FIX: Failed to write {file_path}: {e}")

        results.append(result)

    if results:
        blocked_count = sum(1 for r in results if r.get('security_blocked', False))
        _extract_logger.info(
            f"PAT-028 FIX: Extracted {len(results)} code blocks, "
            f"{sum(1 for r in results if r['written'])} written, "
            f"{blocked_count} security blocked"
        )

    return results


class ClaudeCLIAdapter:
    """Adapter to use Claude CLI as LLM backend.

    This adapter executes the `claude` CLI command to get completions,
    making it compatible with CrewAI and Letta frameworks.

    Usage:
        adapter = ClaudeCLIAdapter()

        # Simple completion
        response = adapter.complete("What is 2+2?")
        print(response.content)

        # With system prompt
        response = adapter.complete(
            "Summarize this text",
            system_prompt="You are a helpful assistant",
        )

        # Chat completion
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
            Message(role="user", content="How are you?"),
        ]
        response = adapter.chat(messages)
    """

    def __init__(self, config: Optional[ClaudeCLIConfig] = None) -> None:
        """Initialize Claude CLI adapter.

        Args:
            config: Configuration options. Uses defaults from env vars if None.
        """
        self._config = config or ClaudeCLIConfig()
        self._request_times: list[float] = []
        self._lock = threading.Lock()
        self._cli_available: Optional[bool] = None
        self._cache_lock = threading.Lock()

    @property
    def config(self) -> ClaudeCLIConfig:
        """Get current configuration."""
        return self._config

    def is_available(self) -> bool:
        """Check if Claude CLI is available.

        Returns:
            True if claude CLI is installed and accessible.
        """
        if self._cli_available is not None:
            return self._cli_available

        try:
            result = subprocess.run(
                [self._config.cli_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                stdin=subprocess.DEVNULL,  # Prevent terminal pop-ups
                start_new_session=True,  # Detach from controlling terminal
            )
            self._cli_available = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            self._cli_available = False

        return self._cli_available

    def verify_daemon_active(self) -> tuple[bool, str]:
        """RED TEAM FIX D-07: Verify daemon mode is active.

        Checks that Claude CLI is running in daemon mode with
        --dangerously-skip-permissions flag.

        Returns:
            Tuple of (is_active, error_message).
        """
        if not self.is_available():
            return False, "Claude CLI is not available"

        # Check if daemon permissions are available
        try:
            # Try a simple operation that requires daemon mode
            result = subprocess.run(
                [
                    self._config.cli_path,
                    "--dangerously-skip-permissions",
                    "--print",
                    "-p", "test",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                stdin=subprocess.DEVNULL,  # Prevent terminal pop-ups
                start_new_session=True,  # Detach from controlling terminal
            )
            if result.returncode == 0:
                return True, ""
            else:
                return False, f"Daemon mode check failed: {result.stderr[:100]}"
        except subprocess.TimeoutExpired:
            return False, "Daemon check timed out"
        except Exception as e:
            return False, f"Daemon check error: {e}"

    def require_daemon_mode(self) -> None:
        """RED TEAM FIX D-07: Raise if daemon mode is not active.

        Raises:
            ClaudeCLIError: If daemon mode is not available.
        """
        is_active, error = self.verify_daemon_active()
        if not is_active:
            raise ClaudeCLIError(
                f"D-07 VIOLATION: Daemon mode required but not active. "
                f"Run Claude CLI with --dangerously-skip-permissions. Error: {error}"
            )

    def _check_rate_limit(self) -> None:
        """Check and enforce rate limiting.

        Raises:
            ClaudeCLIRateLimitError: If rate limit would be exceeded.
        """
        with self._lock:
            now = time.time()
            # Remove requests older than 1 minute
            self._request_times = [t for t in self._request_times if now - t < 60]

            if len(self._request_times) >= self._config.rate_limit_per_minute:
                wait_time = 60 - (now - self._request_times[0])
                raise ClaudeCLIRateLimitError(
                    f"Rate limit exceeded. Wait {wait_time:.1f}s"
                )

            self._request_times.append(now)

    def _cache_key(
        self,
        prompt: str,
        system_prompt: Optional[str],
        output_format: str,
        json_schema: Optional[dict[str, Any]],
    ) -> str:
        payload = {
            "adapter": "ClaudeCLIAdapter",
            "adapter_version": "2.0.0",
            "cli_path": self._config.cli_path,
            "model": self._config.model,
            "max_tokens": self._config.max_tokens,
            "output_format": output_format,
            "system_prompt": system_prompt or "",
            "prompt": prompt,
            "json_schema": json_schema or None,
        }
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _cache_read(self, key: str) -> Optional[dict[str, Any]]:
        if not self._config.cache_enabled:
            return None

        cache_path = Path(self._config.cache_dir) / f"{key}.json"
        if not cache_path.exists():
            return None

        try:
            entry = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            return None

        created_at = entry.get("created_at_epoch")
        if not isinstance(created_at, (int, float)):
            return None

        ttl = self._config.cache_ttl_seconds
        if ttl and ttl > 0:
            age_seconds = time.time() - float(created_at)
            if age_seconds > ttl:
                return None

        return entry

    def _prune_cache_if_needed(self) -> None:
        """P0-04 FIX: Remove old cache entries if over limits.

        This prevents unbounded cache growth when cache is enabled by default.
        Called before each cache write operation.
        """
        cache_dir = Path(self._config.cache_dir)
        if not cache_dir.exists():
            return

        try:
            entries = sorted(
                cache_dir.glob("*.json"),
                key=lambda p: p.stat().st_mtime
            )

            # Prune by count (keep MAX_CACHE_ENTRIES)
            max_entries = self._config.MAX_CACHE_ENTRIES
            while len(entries) > max_entries:
                oldest = entries.pop(0)
                try:
                    oldest.unlink()
                    logger.debug(f"Cache pruned (count): {oldest.name}")
                except OSError as e:
                    logger.debug(f"CACHE: Cache operation failed: {e}")

            # Prune by size (keep under MAX_CACHE_SIZE_MB)
            max_bytes = self._config.MAX_CACHE_SIZE_MB * 1024 * 1024
            total_size = sum(e.stat().st_size for e in entries if e.exists())
            while total_size > max_bytes and entries:
                oldest = entries.pop(0)
                try:
                    file_size = oldest.stat().st_size
                    oldest.unlink()
                    total_size -= file_size
                    logger.debug(f"Cache pruned (size): {oldest.name}")
                except OSError as e:
                    logger.debug(f"CACHE: Cache operation failed: {e}")
        except Exception as e:
            logger.debug(f"Cache pruning skipped: {e}")

    def _cache_write(self, key: str, entry: dict[str, Any]) -> None:
        if not self._config.cache_enabled:
            return

        cache_dir = Path(self._config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # P0-04 FIX: Prune old entries before writing new one
        self._prune_cache_if_needed()

        cache_path = cache_dir / f"{key}.json"

        tmp_path = cache_path.with_suffix(f".tmp.{os.getpid()}.{int(time.time_ns())}.json")
        tmp_path.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, cache_path)

    # PAT-032: Progress check constants from daemon pattern
    # Claude instances sometimes stop and wait for stimulus (input)
    # Instead of timeout, we send periodic status checks to keep session active
    # 2026-01-27: Reduced from 3600s to 300s for better monitoring visibility
    PROGRESS_CHECK_INTERVAL: int = 300  # 5 minutes between status checks (was 1 hour)
    PROGRESS_CHECK_PROMPT: str = "Como está o progresso da tarefa? Continue trabalhando e reporte o status atual."
    MAX_PROGRESS_CHECKS: int = 288  # Max 24 hours of progress checks (288 * 300s = 24h)

    def _execute_cli(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        output_format: str = "text",
        json_schema: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
        resume: bool = False,
        skip_quietstar: bool = False,
    ) -> str:
        """Execute Claude CLI command with progress check pattern.

        PAT-032 FIX: Uses progress checks instead of hard timeout.
        If Claude doesn't respond within PROGRESS_CHECK_INTERVAL,
        sends a status check prompt to give it stimulus.
        This prevents Claude from getting stuck waiting for input.

        Based on daemon pattern from archive/legacy_pipeline/orchestration/agent_daemon.py

        2026-01-20: Now includes QuietStar pre-generation safety check.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.
            output_format: Output format ("text" or "json").
            json_schema: JSON schema for structured output.
            session_id: Optional session ID for persistent context.
            resume: Whether to resume an existing session.
            skip_quietstar: Skip QuietStar checks (for internal safety LLM calls).

        Returns:
            Raw output from Claude CLI.

        Raises:
            ClaudeCLIError: If execution fails.
            QuietStarBlockedError: If blocked by safety guardrails.
        """
        # 2026-01-20: QuietStar PRE-GENERATION SAFETY CHECK
        # Run BEFORE executing the CLI to catch malicious prompts early
        if not skip_quietstar and QUIETSTAR_REFLEXION_AVAILABLE:
            try:
                # Determine if this is code-related context
                is_code = output_format == "json" or any(
                    kw in prompt.lower()
                    for kw in ["code", "function", "class", "import", "def ", "async "]
                )
                _check_quietstar_pre_generation(
                    prompt,
                    is_code=is_code,
                    is_security_context=False,
                )
            except QuietStarBlockedError:
                raise  # Re-raise to block the request
            except Exception as e:
                # Log but continue on non-blocking errors
                logger.debug(f"QuietStar pre-check skipped: {e}")

        def _build_cmd(prompt_text: str, is_resume: bool) -> list:
            # PAT-028/PAT-030 FIX: Daemon pattern flags
            # SEC-001: Expected behavior for pipeline - log at debug level only
            logger.debug(
                "SEC-001: Using --dangerously-skip-permissions (expected for pipeline). "
                "session_id=%s, prompt_len=%d",
                session_id[:8] if session_id else "none",
                len(prompt_text),
            )

            # SEC-001 FIX: Track security event in Langfuse for observability
            if LANGFUSE_AVAILABLE:
                try:
                    client = get_langfuse_client()
                    if client and client.is_available():
                        lf = client._get_langfuse_sdk()
                        if lf:
                            lf.event(
                                name="security_flag_usage",
                                metadata={
                                    "flag": "--dangerously-skip-permissions",
                                    "session_id": session_id[:8] if session_id else "none",
                                    "cwd": str(Path.cwd()),
                                    "prompt_length": len(prompt_text),
                                    "audit_code": "SEC-001",
                                },
                                tags=["security", "audit", "daemon_mode"],
                            )
                except Exception as e:
                    logger.debug(f"SEC-001: Failed to log security event to Langfuse: {e}")

            cmd = [
                self._config.cli_path,
                "--dangerously-skip-permissions",  # Full access - enables tools
                "--print",  # Non-interactive mode with stdout output
            ]

            # Add output format
            if output_format == "json":
                cmd.extend(["--output-format", "json"])
                if json_schema:
                    cmd.extend(["--json-schema", json.dumps(json_schema)])

            # Add session management
            if session_id:
                if is_resume:
                    cmd.extend(["--resume", session_id])
                else:
                    cmd.extend(["--session-id", session_id])

            # Add system prompt
            if system_prompt:
                cmd.extend(["--system-prompt", system_prompt])

            # Add prompt
            cmd.extend(["-p", prompt_text])
            return cmd

        current_prompt = prompt
        is_resume = resume
        progress_check_count = 0

        # Langfuse: Track timing for observability
        _langfuse_start_time = time.time()

        while True:
            cmd = _build_cmd(current_prompt, is_resume)

            try:
                logger.info(
                    "CLAUDE_INVOKE: session=%s resume=%s progress_check=%d",
                    session_id[:8] if session_id else "none",
                    is_resume,
                    progress_check_count,
                )

                # PAT-032: Use Popen for non-blocking execution with progress check
                # SEC-002 FIX: Explicit shell=False for security (prevents shell injection)
                # 2026-01-24: Added stdin=DEVNULL and start_new_session=True to prevent
                # brief window pop-ups on macOS when running subprocesses
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.DEVNULL,  # Prevent terminal pop-ups
                    text=True,
                    cwd=str(Path.cwd()),
                    shell=False,  # SEC-002: Never use shell=True with untrusted input
                    start_new_session=True,  # Detach from controlling terminal (no pop-ups)
                )

                try:
                    # Wait with progress check interval
                    stdout, stderr = proc.communicate(timeout=self.PROGRESS_CHECK_INTERVAL)
                    returncode = proc.returncode

                    # Process completed within interval
                    if returncode == 0:
                        logger.info("CLAUDE_SUCCESS: exit_code=%d", returncode)

                        # Grafana: Increment LLM call counter for Pipeline Control Center
                        if GRAFANA_METRICS_AVAILABLE and get_metrics_publisher:
                            try:
                                metrics = get_metrics_publisher()
                                # Estimate tokens and cost for dashboard
                                prompt_tokens_est = len(prompt) // 4
                                completion_tokens_est = len(stdout) // 4
                                # Claude Opus 4.5 pricing: $15/1M input, $75/1M output
                                estimated_cost = (prompt_tokens_est * 15 + completion_tokens_est * 75) / 1_000_000
                                # increment_llm_calls also updates estimated_cost cumulatively
                                metrics.increment_llm_calls(cost=estimated_cost)
                                logger.debug(f"GRAFANA: LLM call tracked, est_cost=${estimated_cost:.6f}")
                            except Exception as e:
                                logger.debug(f"GRAFANA: Failed to track LLM call: {e}")

                        # Langfuse: Track generation for LLM observability
                        if LANGFUSE_AVAILABLE:
                            try:
                                latency_ms = int((time.time() - _langfuse_start_time) * 1000)
                                # Estimate tokens (rough: 4 chars per token)
                                prompt_tokens = len(prompt) // 4
                                completion_tokens = len(stdout) // 4

                                # ALWAYS track - create ad-hoc trace if none exists
                                trace_id = get_current_trace()
                                if trace_id is None:
                                    # No active trace context - create ad-hoc trace directly
                                    client = get_langfuse_client()
                                    if client and client.is_available():
                                        lf = client._get_langfuse_sdk()
                                        if lf:
                                            # Create trace and generation directly
                                            trace = lf.trace(
                                                name=f"claude_cli_call",
                                                metadata={
                                                    "session_id": session_id,
                                                    "output_format": output_format,
                                                },
                                                tags=["claude_cli", "llm_call"],
                                            )
                                            trace.generation(
                                                name=f"claude_cli:{session_id[:8] if session_id else 'direct'}",
                                                model=self._config.model,
                                                input=prompt[:2000],
                                                output=stdout[:5000],
                                                usage={
                                                    "input": prompt_tokens,
                                                    "output": completion_tokens,
                                                },
                                                metadata={
                                                    "progress_checks": progress_check_count,
                                                    "latency_ms": latency_ms,
                                                },
                                            )
                                            lf.flush()
                                            logger.debug(f"LANGFUSE: Created ad-hoc trace+generation latency={latency_ms}ms")
                                else:
                                    # Use existing trace context
                                    track_generation(
                                        name=f"claude_cli:{session_id[:8] if session_id else 'direct'}",
                                        model=self._config.model,
                                        prompt=prompt[:2000],
                                        completion=stdout[:5000],
                                        prompt_tokens=prompt_tokens,
                                        completion_tokens=completion_tokens,
                                        latency_ms=latency_ms,
                                        metadata={
                                            "session_id": session_id,
                                            "output_format": output_format,
                                            "progress_checks": progress_check_count,
                                        },
                                    )
                                    logger.debug(f"LANGFUSE: Tracked generation latency={latency_ms}ms tokens={prompt_tokens}+{completion_tokens}")
                            except Exception as e:
                                logger.warning(f"LANGFUSE: Failed to track generation: {e}")
                        return stdout.strip()
                    else:
                        error_msg = stderr.strip() or "Unknown error"
                        raise ClaudeCLIError(f"Claude CLI failed: {error_msg}")

                except subprocess.TimeoutExpired:
                    # Progress check interval expired - send status check prompt
                    progress_check_count += 1

                    if progress_check_count >= self.MAX_PROGRESS_CHECKS:
                        logger.error(
                            "CLAUDE_MAX_PROGRESS_CHECKS: count=%d - terminating",
                            progress_check_count,
                        )
                        proc.terminate()
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
                        raise ClaudeCLIError(
                            f"Claude exceeded {self.MAX_PROGRESS_CHECKS} progress checks "
                            f"({self.MAX_PROGRESS_CHECKS * self.PROGRESS_CHECK_INTERVAL}s total)"
                        )

                    logger.info(
                        "CLAUDE_PROGRESS_CHECK: count=%d - sending stimulus",
                        progress_check_count,
                    )

                    # Terminate current process gracefully
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()

                    # Next iteration will resume with progress check prompt
                    current_prompt = self.PROGRESS_CHECK_PROMPT
                    is_resume = True
                    continue

            except FileNotFoundError:
                logger.error("CLAUDE_CLI_NOT_FOUND")
                raise ClaudeCLIError("Claude CLI not found. Please install claude.")
            except PermissionError as e:
                logger.error("CLAUDE_PERMISSION_DENIED: %s", e)
                raise ClaudeCLIError(f"Permission denied: {e}")
            except OSError as e:
                logger.error("CLAUDE_EXECUTION_FAILED: %s", e)
                raise ClaudeCLIError(f"Execution failed: {e}")

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        skip_quietstar: bool = False,
    ) -> CompletionResponse:
        """Get a completion from Claude.

        2026-01-20: Now includes QuietStar post-generation reflexion check.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.
            temperature: Temperature for sampling (not used in CLI).
            max_tokens: Maximum output tokens (not used in CLI).
            skip_quietstar: Skip QuietStar checks (for internal safety LLM calls).

        Returns:
            CompletionResponse with the generated content.

        Raises:
            ClaudeCLIError: If completion fails.
            QuietStarBlockedError: If output fails safety reflexion.
        """
        cache_key = self._cache_key(
            prompt=prompt,
            system_prompt=system_prompt,
            output_format="text",
            json_schema=None,
        )
        with self._cache_lock:
            cached = self._cache_read(cache_key)
        if cached and isinstance(cached.get("output"), str):
            output = cached["output"]
            return CompletionResponse(
                content=output,
                model=self._config.model,
                stop_reason="end_turn",
                raw_output=output,
            )

        self._check_rate_limit()

        for attempt in range(self._config.max_retries):
            try:
                output = self._execute_cli(
                    prompt,
                    system_prompt=system_prompt,
                    skip_quietstar=skip_quietstar,
                )

                # 2026-01-20: QuietStar POST-GENERATION REFLEXION CHECK
                # Run AFTER receiving response to validate quality and safety
                if not skip_quietstar and QUIETSTAR_REFLEXION_AVAILABLE:
                    try:
                        _check_quietstar_post_generation(
                            prompt,
                            output,
                            min_quality_score=0.6,
                        )
                    except QuietStarBlockedError:
                        raise  # Re-raise to block unsafe output
                    except Exception as e:
                        # Log but continue on non-blocking errors
                        logger.debug(f"QuietStar post-check skipped: {e}")

                with self._cache_lock:
                    self._cache_write(
                        cache_key,
                        {
                            "created_at_epoch": time.time(),
                            "output": output,
                            "output_format": "text",
                            "model": self._config.model,
                        },
                    )

                return CompletionResponse(
                    content=output,
                    model=self._config.model,
                    stop_reason="end_turn",
                    raw_output=output,
                )

            except ClaudeCLIRateLimitError:
                raise
            except QuietStarBlockedError:
                raise  # Don't retry blocked outputs
            except ClaudeCLITimeoutError:
                if attempt == self._config.max_retries - 1:
                    raise
                time.sleep(self._config.retry_delay * (attempt + 1))
            except ClaudeCLIError:
                if attempt == self._config.max_retries - 1:
                    raise
                time.sleep(self._config.retry_delay * (attempt + 1))

        raise ClaudeCLIError("Max retries exceeded")

    def complete_json(
        self,
        prompt: str,
        schema: dict[str, Any],
        system_prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        """Get a structured JSON response from Claude.

        Args:
            prompt: The user prompt.
            schema: JSON schema for the expected response.
            system_prompt: Optional system prompt.

        Returns:
            Parsed JSON response.

        Raises:
            ClaudeCLIError: If completion fails.
            ValueError: If response is not valid JSON.
        """
        cache_key = self._cache_key(
            prompt=prompt,
            system_prompt=system_prompt,
            output_format="json",
            json_schema=schema,
        )
        with self._cache_lock:
            cached = self._cache_read(cache_key)
        if cached and isinstance(cached.get("json"), dict):
            return cached["json"]

        self._check_rate_limit()

        for attempt in range(self._config.max_retries):
            try:
                output = self._execute_cli(
                    prompt,
                    system_prompt=system_prompt,
                    output_format="json",
                    json_schema=schema,
                )

                result: dict[str, Any] = json.loads(output)
                with self._cache_lock:
                    self._cache_write(
                        cache_key,
                        {
                            "created_at_epoch": time.time(),
                            "json": result,
                            "output_format": "json",
                            "model": self._config.model,
                        },
                    )
                return result

            except json.JSONDecodeError as e:
                if attempt == self._config.max_retries - 1:
                    raise ValueError(f"Invalid JSON response: {e}") from e
                time.sleep(self._config.retry_delay * (attempt + 1))
            except ClaudeCLIRateLimitError:
                raise
            except ClaudeCLIError:
                if attempt == self._config.max_retries - 1:
                    raise
                time.sleep(self._config.retry_delay * (attempt + 1))

        raise ClaudeCLIError("Max retries exceeded")

    def chat(
        self,
        messages: list[Message],
        system_prompt: Optional[str] = None,
    ) -> CompletionResponse:
        """Get a chat completion from Claude.

        Converts messages to a single prompt for CLI execution.

        Args:
            messages: List of conversation messages.
            system_prompt: Optional system prompt.

        Returns:
            CompletionResponse with the generated content.
        """
        # Convert messages to a single prompt
        prompt_parts = []
        for msg in messages:
            if msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")

        prompt = "\n\n".join(prompt_parts)
        if messages and messages[-1].role == "user":
            prompt += "\n\nAssistant:"

        return self.complete(prompt, system_prompt=system_prompt)


# =============================================================================
# CrewAI LLM Interface Wrapper
# =============================================================================


try:
    from crewai.llms.base_llm import BaseLLM
    CREWAI_AVAILABLE = True
except ImportError:
    BaseLLM = object  # type: ignore
    CREWAI_AVAILABLE = False


class ClaudeCLIForCrewAI(BaseLLM):  # type: ignore[misc]
    """Claude CLI wrapper compatible with CrewAI LLM interface.

    This class inherits from CrewAI's BaseLLM to be properly recognized
    by CrewAI's LLM instantiation logic.

    Usage with CrewAI:
        from crewai import Agent

        llm = ClaudeCLIForCrewAI()
        agent = Agent(
            role="Developer",
            goal="Write code",
            llm=llm,
        )
    """

    def __init__(self, config: Optional[ClaudeCLIConfig] = None) -> None:
        """Initialize CrewAI-compatible wrapper.

        Args:
            config: Claude CLI configuration.
        """
        self._adapter = ClaudeCLIAdapter(config)
        self._model_name = self._adapter.config.model
        # CrewAI expects _token_usage for cost tracking
        self._token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    @property
    def model(self) -> str:
        """Get model name (CrewAI compatibility)."""
        return self._model_name

    def call(
        self,
        messages: Any,
        tools: Optional[Any] = None,
        callbacks: Optional[list[Callable[..., Any]]] = None,
        available_functions: Optional[dict[str, Any]] = None,
        from_task: Optional[Any] = None,
        from_agent: Optional[Any] = None,
        response_model: Optional[Any] = None,
    ) -> str:
        """Call the LLM with messages (CrewAI BaseLLM interface).

        Security-wrapped via @secure_operation pattern for input/output validation.

        Args:
            messages: String or list of message dicts with 'role' and 'content'.
            tools: Optional tools (not used in CLI mode).
            callbacks: Optional callbacks (not used).
            available_functions: Optional functions (not used).
            from_task: Task context (not used).
            from_agent: Agent context (not used).
            response_model: Response model (not used).

        Returns:
            Generated response text.
        """
        return self._secure_call_impl(messages, tools, callbacks, available_functions,
                                       from_task, from_agent, response_model)

    @secure_operation(level="standard", scan_input=True, scan_output=True)
    def _secure_call_impl(
        self,
        messages: Any,
        tools: Optional[Any] = None,
        callbacks: Optional[list[Callable[..., Any]]] = None,
        available_functions: Optional[dict[str, Any]] = None,
        from_task: Optional[Any] = None,
        from_agent: Optional[Any] = None,
        response_model: Optional[Any] = None,
    ) -> str:
        """Internal secure implementation of call().

        This method is wrapped with @secure_operation to provide:
        - Input scanning for prompt injection (LLM Guard)
        - Output scanning for sensitive data leakage (LLM Guard)
        - NeMo guardrails for jailbreak and content filtering
        - Security event logging via Langfuse
        - DeepEval quality evaluation

        2026-01-20: Added QuietStar/Reflexion as additional security layer
        """
        # Handle string input
        if isinstance(messages, str):
            # 2026-01-16: NeMo guardrails check on input
            _check_nemo_guardrails_input(messages)

            # 2026-01-20: QuietStar PRE-GENERATION SAFETY CHECK
            if QUIETSTAR_REFLEXION_AVAILABLE:
                try:
                    _check_quietstar_pre_generation(messages, is_code=False)
                except QuietStarBlockedError:
                    raise
                except Exception as e:
                    logger.debug(f"QuietStar pre-check skipped: {e}")

            response = self._adapter.complete(messages)

            # 2026-01-16: NeMo guardrails check on output
            output_content = _check_nemo_guardrails_output(response.content)

            # 2026-01-20: QuietStar POST-GENERATION REFLEXION CHECK
            if QUIETSTAR_REFLEXION_AVAILABLE:
                try:
                    _check_quietstar_post_generation(messages, output_content)
                except QuietStarBlockedError:
                    raise
                except Exception as e:
                    logger.debug(f"QuietStar post-check skipped: {e}")

            # 2026-01-16: DeepEval DISABLED - redundante, só loga warning sem ação
            # Causa ~5x mais API calls (370 extras em S00) sem benefício
            # self._evaluate_response(messages, output_content)
            return output_content

        # Handle list of messages
        system_prompt = None
        chat_messages = []
        input_text = ""

        for msg in messages:
            if hasattr(msg, 'get'):
                role = msg.get("role", "user")
                content = msg.get("content", "")
            else:
                # Handle LLMMessage objects
                role = getattr(msg, 'role', 'user')
                content = getattr(msg, 'content', str(msg))

            if role == "system":
                system_prompt = content
            else:
                chat_messages.append(
                    Message(
                        role=role,
                        content=content,
                    )
                )
                input_text += content + " "

        # 2026-01-16: NeMo guardrails check on combined input
        _check_nemo_guardrails_input(input_text.strip())

        # 2026-01-20: QuietStar PRE-GENERATION SAFETY CHECK
        if QUIETSTAR_REFLEXION_AVAILABLE:
            try:
                _check_quietstar_pre_generation(input_text.strip(), is_code=False)
            except QuietStarBlockedError:
                raise
            except Exception as e:
                logger.debug(f"QuietStar pre-check skipped: {e}")

        response = self._adapter.chat(chat_messages, system_prompt=system_prompt)

        # 2026-01-16: NeMo guardrails check on output
        output_content = _check_nemo_guardrails_output(response.content)

        # 2026-01-20: QuietStar POST-GENERATION REFLEXION CHECK
        if QUIETSTAR_REFLEXION_AVAILABLE:
            try:
                _check_quietstar_post_generation(input_text.strip(), output_content)
            except QuietStarBlockedError:
                raise
            except Exception as e:
                logger.debug(f"QuietStar post-check skipped: {e}")

        # 2026-01-16: DeepEval DISABLED - redundante, só loga warning sem ação
        # self._evaluate_response(input_text.strip(), output_content)
        return output_content

    def _evaluate_response(self, input_text: str, output_text: str) -> None:
        """Evaluate LLM response quality with DeepEval.

        This is called EXPLICITLY on every LLM response to ensure
        quality metrics are tracked.
        """
        try:
            from pipeline.langgraph.deepeval_integration import evaluate_llm_output
            result = evaluate_llm_output(input_text[:1000], output_text[:2000])
            if not result.passed:
                logger.warning(
                    f"DeepEval: Low quality response (score={result.overall_score:.2f})"
                )
            else:
                logger.debug(f"DeepEval: Response quality OK (score={result.overall_score:.2f})")
        except ImportError as e:
            logger.debug(f"IMPORT: Module not available: {e}")
        except Exception as e:
            logger.debug(f"DeepEval evaluation skipped: {e}")

    def __call__(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
    ) -> str:
        """Direct call with prompt (CrewAI compatibility).

        Security-wrapped via @secure_operation pattern.

        Args:
            prompt: User prompt.
            stop: Stop sequences (not used in CLI).

        Returns:
            Generated response text.
        """
        return self._secure_direct_call_impl(prompt, stop)

    @secure_operation(level="standard", scan_input=True, scan_output=True)
    def _secure_direct_call_impl(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
    ) -> str:
        """Internal secure implementation of __call__().

        This method is wrapped with @secure_operation to provide:
        - Input scanning for prompt injection
        - Output scanning for sensitive data leakage
        - Security event logging via Langfuse

        2026-01-20: Added QuietStar/Reflexion as additional security layer
        """
        # 2026-01-20: QuietStar PRE-GENERATION SAFETY CHECK
        if QUIETSTAR_REFLEXION_AVAILABLE:
            try:
                _check_quietstar_pre_generation(prompt, is_code=False)
            except QuietStarBlockedError:
                raise
            except Exception as e:
                logger.debug(f"QuietStar pre-check skipped: {e}")

        response = self._adapter.complete(prompt)

        # 2026-01-20: QuietStar POST-GENERATION REFLEXION CHECK
        if QUIETSTAR_REFLEXION_AVAILABLE:
            try:
                _check_quietstar_post_generation(prompt, response.content)
            except QuietStarBlockedError:
                raise
            except Exception as e:
                logger.debug(f"QuietStar post-check skipped: {e}")

        # 2026-01-16: DeepEval DISABLED - redundante, só loga warning sem ação
        # self._evaluate_response(prompt, response.content)
        return response.content


# =============================================================================
# Letta Custom LLM Provider
# =============================================================================


class ClaudeCLIForLetta:
    """Claude CLI provider compatible with Letta (ex-MemGPT).

    Letta allows custom LLM providers. This class provides the
    necessary interface to use Claude CLI as the LLM backend.

    Usage with Letta:
        from letta import Letta

        client = Letta()
        # Configure custom LLM provider
        # Note: Letta's API may require additional configuration
    """

    def __init__(self, config: Optional[ClaudeCLIConfig] = None) -> None:
        """Initialize Letta-compatible wrapper.

        Args:
            config: Claude CLI configuration.
        """
        self._adapter = ClaudeCLIAdapter(config)

    def create_completion(
        self,
        messages: list[dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Create a completion (Letta interface).

        Args:
            messages: List of message dicts.
            model: Model name (ignored, uses CLI config).
            temperature: Temperature (ignored in CLI).
            max_tokens: Max tokens (ignored in CLI).

        Returns:
            Completion response dict compatible with Letta.
        """
        # Extract system prompt
        system_prompt = None
        chat_messages = []

        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content", "")
            else:
                chat_messages.append(
                    Message(
                        role=msg.get("role", "user"),
                        content=msg.get("content", ""),
                    )
                )

        response = self._adapter.chat(chat_messages, system_prompt=system_prompt)

        # Return in OpenAI-compatible format (Letta expects this)
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self._adapter.config.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response.content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,  # CLI doesn't report tokens
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }


# =============================================================================
# Singleton and Factory
# =============================================================================


_adapter: Optional[ClaudeCLIAdapter] = None


def get_claude_adapter(config: Optional[ClaudeCLIConfig] = None) -> ClaudeCLIAdapter:
    """Get the global Claude CLI adapter instance.

    Args:
        config: Optional configuration (only used on first call).

    Returns:
        The Claude CLI adapter singleton.
    """
    global _adapter
    if _adapter is None:
        _adapter = ClaudeCLIAdapter(config)
    return _adapter


def reset_adapter() -> None:
    """Reset the global adapter instance (for testing)."""
    global _adapter
    _adapter = None


# =============================================================================
# PAT-030 FIX: Agent Execution with Tool Instructions (Daemon Pattern)
# =============================================================================

# JSON schema for agent responses (from daemon)
AGENT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string"},
        "summary": {"type": "string"},
        "evidence_paths": {"type": "array", "items": {"type": "string"}},
        "next_steps": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["status", "summary", "evidence_paths", "next_steps"],
}


def build_agent_execution_prompt(
    task: str,
    instruction: str,
    workspace_path: str,
    sprint_id: str = "",
    context_refs: Optional[list[str]] = None,
    expected_outputs: Optional[list[str]] = None,
) -> str:
    """Build execution prompt for agent mode (PAT-030 FIX).

    Based on daemon's _build_execution_prompt. This prompt instructs Claude
    to USE Write/Edit/Bash tools during execution.

    Args:
        task: Task name/identifier.
        instruction: Detailed instruction for the task.
        workspace_path: Directory where files should be created.
        sprint_id: Sprint identifier (e.g., "S00").
        context_refs: List of context file references.
        expected_outputs: List of expected output files.

    Returns:
        Formatted prompt with tool use instructions.
    """
    refs = "\n".join(f"- {x}" for x in (context_refs or [])) or "(nenhum)"
    outs = "\n".join(f"- {x}" for x in (expected_outputs or [])) or "(nenhum)"

    return f"""# EXECUTION MODE: REAL CODE EXECUTION

## MODO DE EXECUÇÃO
Você está em modo **EXECUTE_REAL**. Isso significa que você deve:
- CRIAR arquivos reais usando a tool Write/Edit
- EXECUTAR comandos reais usando a tool Bash
- FAZER commits usando git via Bash
- GERAR evidências de tudo que executar

## FERRAMENTAS DISPONÍVEIS
- **Bash**: Execute comandos shell (build, lint, test, git)
- **Write**: Crie novos arquivos
- **Edit**: Modifique arquivos existentes
- **Read**: Leia arquivos para contexto
- **Glob/Grep**: Busque em código

## WORKSPACE
Diretório de trabalho: {workspace_path}

## TASK ATUAL
SPRINT: {sprint_id or '(não especificada)'}
TASK: {task}

## INSTRUÇÃO
{instruction}

## CONTEXT REFS
{refs}

## EXPECTED OUTPUTS
{outs}

## PROTOCOLO DE EXECUÇÃO
Para cada item do plano:
1. Crie/modifique os arquivos necessários (Write/Edit)
2. Execute validação (Bash: python -m py_compile, ruff check, etc.)
3. Capture o resultado

Ao final:
1. Faça git add dos arquivos criados/modificados
2. Faça git commit com mensagem no formato: <type>({sprint_id}): <descrição>
   - Types válidos: feat, fix, docs, test, chore, refactor
3. Gere evidência dos arquivos criados

## REGRAS IMPORTANTES
- SEMPRE use Write para criar arquivos novos
- SEMPRE use Edit para modificar arquivos existentes
- NUNCA apenas diga que criou um arquivo - CRIE de verdade usando Write!
- Valide cada arquivo após criar

## ESCOPO DE TRABALHO - CRÍTICO!
Você é um agente que trabalha nos PRODUTOS (Veritas e Forekast), NÃO na ferramenta pipeline.

### PERMITIDO - ESCRITA (trabalhe AQUI):
- src/veritas/ - Código do produto Veritas
- src/forekast/ - Código do produto Forekast
- tests/test_veritas/ - Testes do Veritas
- tests/test_forekast/ - Testes do Forekast
- context_packs/ - Documentação

### PERMITIDO - LEITURA (pode consultar para contexto):
- src/pipeline/ - Pode LER para usar ferramentas
- src/pipeline/ - Pode LER para usar ferramentas
- configs/ - Pode LER para contexto

### PROIBIDO - ESCRITA (NUNCA modifique/crie/teste):
- src/pipeline/ - Pipeline é SOMENTE LEITURA
- src/pipeline/ - Pipeline v2 é SOMENTE LEITURA
- tests/test_pipeline/ - Testes do pipeline
- tests/test_pipeline/ - Testes do pipeline
- tests/test_e2e*.py - Testes E2E do pipeline
- configs/pipeline/ - Configs do pipeline

### REORIENTAÇÃO
Se você sentir necessidade de:
- Rodar testes do pipeline → PARE. Rode: pytest tests/test_veritas/ ou pytest tests/test_forekast/
- Modificar código do pipeline → PARE. Isso está FORA do seu escopo.
- Investigar bugs do pipeline → PARE. Foque nos produtos.

Se a tarefa parecer exigir mudanças no pipeline, responda com:
{{"status": "blocked", "summary": "Tarefa requer mudanças no pipeline, fora do escopo do agente", "evidence_paths": [], "next_steps": ["Escalar para operador humano"]}}

## RESPOSTA OBRIGATÓRIA
Você DEVE responder APENAS com um objeto JSON válido. Nenhum texto antes ou depois.
Use exatamente este formato:

{{"status": "success", "summary": "descrição", "evidence_paths": [], "next_steps": []}}

Valores válidos para status: "success", "failed", ou "blocked".
"""


@secure_operation(level="high", scan_input=True, scan_output=True)
def execute_agent_task(
    task: str,
    instruction: str,
    workspace_path: str,
    sprint_id: str = "",
    session_id: Optional[str] = None,
    resume: bool = False,
    context_refs: Optional[list[str]] = None,
    expected_outputs: Optional[list[str]] = None,
    config: Optional[ClaudeCLIConfig] = None,
) -> dict[str, Any]:
    """Execute an agent task with tool use enabled (PAT-030 FIX).

    Security-wrapped with @secure_operation(level="high") for:
    - Input scanning (prompt injection, malicious instructions)
    - Output scanning (sensitive data leakage, secrets)
    - Security event logging via Langfuse

    This function uses the daemon pattern to execute tasks:
    1. Builds prompt with tool use instructions
    2. Uses --dangerously-skip-permissions for tool access
    3. Uses --print with --output-format json for structured output
    4. Uses session management for persistent context

    Args:
        task: Task name/identifier.
        instruction: Detailed instruction for the task.
        workspace_path: Directory where files should be created.
        sprint_id: Sprint identifier.
        session_id: Optional session ID for persistent context.
        resume: Whether to resume existing session.
        context_refs: List of context file references.
        expected_outputs: List of expected output files.
        config: Optional CLI configuration.

    Returns:
        Dict with status, summary, evidence_paths, next_steps.

    Example:
        >>> result = execute_agent_task(
        ...     task="CREATE_SPRINT_SPEC",
        ...     instruction="Create sprint spec for S00",
        ...     workspace_path="/path/to/workspace",
        ...     sprint_id="S00",
        ... )
        >>> print(result["status"])
        'success'
    """
    import uuid

    adapter = ClaudeCLIAdapter(config)

    # Generate session ID if not provided
    if not session_id:
        session_id = str(uuid.uuid4())

    # Build execution prompt with tool instructions
    prompt = build_agent_execution_prompt(
        task=task,
        instruction=instruction,
        workspace_path=workspace_path,
        sprint_id=sprint_id,
        context_refs=context_refs,
        expected_outputs=expected_outputs,
    )

    import time as _time

    try:
        # Track execution time for metrics
        _start_time = _time.time()

        # Execute with daemon pattern
        output = adapter._execute_cli(
            prompt=prompt,
            output_format="json",
            json_schema=AGENT_JSON_SCHEMA,
            session_id=session_id,
            resume=resume,
        )

        # Calculate latency
        _latency_ms = int((_time.time() - _start_time) * 1000)

        # Parse response
        result = json.loads(output)

        # Handle structured_output wrapper (Claude sometimes wraps)
        if isinstance(result.get("structured_output"), dict):
            result = result["structured_output"]
        elif isinstance(result.get("result"), str):
            result = json.loads(result["result"])

        # PAT-032: Validate scope of evidence_paths
        try:
            from pipeline.scope_enforcer import check_scope

            evidence_paths = result.get("evidence_paths", [])
            scope_violations = []
            for path in evidence_paths:
                scope_result = check_scope(action="Write", path=str(path))
                if not scope_result.allowed:
                    scope_violations.append(f"{path}: {scope_result.reason}")

            if scope_violations:
                logger.warning(f"SCOPE VIOLATIONS in evidence_paths: {scope_violations}")
                # Add warning to result but don't block
                result["scope_warnings"] = scope_violations
        except ImportError as e:
            logger.debug(f"IMPORT: Module not available: {e}")
        except Exception as scope_err:
            logger.debug(f"Scope validation error: {scope_err}")

        # Publish LLM call metrics to Grafana dashboard
        try:
            from pipeline.grafana_metrics import get_metrics_publisher
            publisher = get_metrics_publisher()
            publisher.publish_llm_call(
                agent=task,
                model="claude-3-5-sonnet",
                prompt_tokens=len(prompt) // 4,  # Rough estimate
                completion_tokens=len(output) // 4,
                latency_ms=_latency_ms,
            )
        except Exception as pub_err:
            logger.debug(f"Failed to publish LLM call to Grafana: {pub_err}")

        return result

    except json.JSONDecodeError as e:
        return {
            "status": "failed",
            "summary": f"Invalid JSON response: {e}",
            "evidence_paths": [],
            "next_steps": ["Check Claude output format"],
            "raw_output": output[:4000] if 'output' in dir() else "",
        }
    except ClaudeCLIError as e:
        return {
            "status": "failed",
            "summary": f"CLI error: {e}",
            "evidence_paths": [],
            "next_steps": ["Check CLI configuration"],
        }
    except Exception as e:
        return {
            "status": "failed",
            "summary": f"Unexpected error: {e}",
            "evidence_paths": [],
            "next_steps": ["Check logs for details"],
        }
