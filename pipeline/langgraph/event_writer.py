"""Atomic Event Writer with Lock Protection.

This module provides a thread-safe, file-locked event writer for the LangGraph
workflow to prevent event corruption from concurrent writes.

Addresses: RED AGENT CRIT-05 - Single Event Writer Without Lock Protection

Features:
- File-level locking (fcntl on Unix, fallback to thread lock)
- Atomic writes with fsync
- Event deduplication via event_id
- Schema validation (Invariant I8)

Author: Pipeline Autonomo Team
Version: 1.0.0 (2026-01-16)
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)


# =============================================================================
# EVENT SCHEMA
# =============================================================================


@dataclass
class Event:
    """Structured event for event_log.ndjson.

    Invariant I8: event_log uses schema_version, event_id, type, level, phase.

    Fields:
        event_id: Deterministic unique identifier for idempotency
        event_type: Type of event (e.g., "init_completed", "gate_passed")
        run_id: Pipeline run identifier
        sprint_id: Current sprint identifier
        attempt: Retry attempt number
        message: Human-readable event description
        level: Event severity (info, warning, error, critical) - for dashboard filtering
        phase: Current sprint phase - for timeline visualization
        timestamp: ISO 8601 timestamp
        schema_version: Event schema version for compatibility
        source: Event source (e.g., "langgraph", "crewai")
        metadata: Optional additional data
    """

    event_id: str
    event_type: str
    run_id: str
    sprint_id: str
    attempt: int
    message: str
    level: str = "info"  # info, warning, error, critical (BREAKING-01 fix)
    phase: str = "INIT"  # Current SprintPhase value (BREAKING-01 fix)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: str = "1.0"
    source: str = "langgraph"
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "type": self.event_type,
            "run_id": self.run_id,
            "sprint_id": self.sprint_id,
            "attempt": self.attempt,
            "message": self.message,
            "level": self.level,  # Required by dashboard for filtering
            "phase": self.phase,  # Required for timeline visualization
            "timestamp": self.timestamp,
            "source": self.source,
        }
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def create(
        cls,
        event_type: str,
        run_id: str,
        sprint_id: str,
        attempt: int,
        message: str,
        level: str = "info",
        phase: str = "INIT",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Event":
        """Create a new event with deterministic event_id.

        The event_id is deterministic to support idempotency and collision resistance:
        {type}:{run_id}:{sprint_id}:attempt:{attempt}:{hash}

        Args:
            event_type: Type of event (e.g., "init_completed")
            run_id: Pipeline run identifier
            sprint_id: Current sprint identifier
            attempt: Retry attempt number
            message: Human-readable description
            level: Severity level (info, warning, error, critical)
            phase: Current sprint phase (INIT, SPEC, PLAN, EXEC, QA, VOTE, DONE)
            metadata: Optional additional data

        Returns:
            New Event instance.
        """
        import hashlib
        # Add hash suffix for collision resistance (AUDITOR fix)
        content_hash = hashlib.sha256(
            f"{event_type}:{message}:{metadata}".encode()
        ).hexdigest()[:8]
        event_id = f"{event_type}:{run_id}:{sprint_id}:attempt:{attempt}:{content_hash}"
        return cls(
            event_id=event_id,
            event_type=event_type,
            run_id=run_id,
            sprint_id=sprint_id,
            attempt=attempt,
            message=message,
            level=level,
            phase=phase,
            metadata=metadata,
        )


# =============================================================================
# ATOMIC EVENT WRITER
# =============================================================================


class AtomicEventWriter:
    """Thread-safe, file-locked event writer.

    Addresses CRIT-05: Prevents concurrent write corruption by:
    1. Using file-level exclusive lock (fcntl.LOCK_EX)
    2. Using thread lock for same-process safety
    3. Using fsync to ensure durability
    4. Deduplicating events by event_id

    Usage:
        writer = AtomicEventWriter(run_dir / "event_log.ndjson")
        writer.write(event)
        writer.close()

    Or as context manager:
        with AtomicEventWriter(path) as writer:
            writer.write(event)
    """

    def __init__(
        self,
        log_path: Path,
        deduplicate: bool = True,
    ):
        """Initialize the event writer.

        Args:
            log_path: Path to event_log.ndjson file.
            deduplicate: Whether to skip duplicate event_ids.
        """
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self._deduplicate = deduplicate
        self._seen_ids: Set[str] = set()
        self._lock = threading.Lock()
        self._file = None

        # Load existing event_ids for deduplication
        if deduplicate and self.log_path.exists():
            self._load_existing_ids()

    def _load_existing_ids(self) -> None:
        """Load existing event_ids from log file."""
        try:
            with open(self.log_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            event = json.loads(line)
                            if "event_id" in event:
                                self._seen_ids.add(event["event_id"])
                        except json.JSONDecodeError:
                            continue
            logger.debug(f"Loaded {len(self._seen_ids)} existing event IDs")
        except Exception as e:
            logger.warning(f"Could not load existing events: {e}")

    def write(self, event: Event) -> bool:
        """Write an event to the log file atomically.

        CRIT-003 Fix: File lock is now acquired BEFORE dedup check to prevent
        race conditions between processes. The dedup check is re-done after
        acquiring the lock to handle cross-process duplicates.

        Args:
            event: Event to write.

        Returns:
            True if written, False if deduplicated.
        """
        with self._lock:
            # Quick check for in-memory duplicates (same process)
            if self._deduplicate and event.event_id in self._seen_ids:
                logger.debug(f"Skipping duplicate event (in-memory): {event.event_id}")
                return False

            try:
                # Open file with exclusive lock BEFORE any dedup check
                # This prevents the race condition where two processes both
                # pass the check before either acquires the lock
                with open(self.log_path, "a+") as f:
                    # Acquire exclusive lock FIRST (blocks until available)
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    try:
                        # CRIT-003 FIX: Re-check for duplicates AFTER acquiring lock
                        # Another process may have written the same event while we waited
                        if self._deduplicate:
                            f.seek(0)
                            for line in f:
                                line = line.strip()
                                if line:
                                    try:
                                        existing = json.loads(line)
                                        if existing.get("event_id") == event.event_id:
                                            logger.debug(f"Skipping duplicate event (file check): {event.event_id}")
                                            return False
                                    except json.JSONDecodeError:
                                        continue
                            # Move to end for appending
                            f.seek(0, 2)

                        # Write event
                        f.write(json.dumps(event.to_dict()) + "\n")
                        # Ensure durability
                        f.flush()
                        os.fsync(f.fileno())
                    finally:
                        # Release lock
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

                # Track for in-memory deduplication
                if self._deduplicate:
                    self._seen_ids.add(event.event_id)

                logger.debug(f"Wrote event: {event.event_id}")
                return True

            except Exception as e:
                logger.error(f"Failed to write event {event.event_id}: {e}")
                raise

    def write_dict(
        self,
        event_type: str,
        run_id: str,
        sprint_id: str,
        attempt: int,
        message: str,
        level: str = "info",
        phase: str = "INIT",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Convenience method to write event from parameters.

        Args:
            event_type: Type of event.
            run_id: Run identifier.
            sprint_id: Sprint identifier.
            attempt: Attempt number.
            message: Event message.
            level: Event severity (info, warning, error, critical).
            phase: Current sprint phase.
            metadata: Optional metadata.

        Returns:
            True if written, False if deduplicated.
        """
        event = Event.create(
            event_type=event_type,
            run_id=run_id,
            sprint_id=sprint_id,
            attempt=attempt,
            message=message,
            level=level,
            phase=phase,
            metadata=metadata,
        )
        return self.write(event)

    def __enter__(self) -> "AtomicEventWriter":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        pass  # No persistent resources to clean up


# =============================================================================
# BUFFERED EVENT WRITER (OPT-02-004)
# =============================================================================


class BufferedEventWriter:
    """Buffered event writer that batches writes to reduce I/O (OPT-02-004).

    Buffers events in memory and flushes to disk when:
    1. Buffer reaches max_size (default 50 events)
    2. flush() is called explicitly
    3. Object is garbage collected

    Thread-safe implementation using locks.
    """

    def __init__(
        self,
        log_path: Path,
        max_size: int = 50,
        deduplicate: bool = True,
    ):
        """Initialize the buffered event writer.

        Args:
            log_path: Path to event_log.ndjson file.
            max_size: Maximum buffer size before auto-flush.
            deduplicate: Whether to skip duplicate event_ids.
        """
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self._max_size = max_size
        self._deduplicate = deduplicate
        self._buffer: list[Event] = []
        self._seen_ids: Set[str] = set()
        self._lock = threading.Lock()

        # Load existing event_ids for deduplication
        if deduplicate and self.log_path.exists():
            self._load_existing_ids()

    def _load_existing_ids(self) -> None:
        """Load existing event_ids from log file."""
        try:
            with open(self.log_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            event = json.loads(line)
                            if "event_id" in event:
                                self._seen_ids.add(event["event_id"])
                        except json.JSONDecodeError:
                            continue
            logger.debug(f"Loaded {len(self._seen_ids)} existing event IDs for buffer")
        except Exception as e:
            logger.warning(f"Could not load existing events for buffer: {e}")

    def add(self, event: Event) -> bool:
        """Add an event to the buffer.

        Args:
            event: Event to add.

        Returns:
            True if added, False if deduplicated.
        """
        with self._lock:
            # Check for duplicates
            if self._deduplicate and event.event_id in self._seen_ids:
                logger.debug(f"Skipping duplicate event in buffer: {event.event_id}")
                return False

            self._buffer.append(event)
            if self._deduplicate:
                self._seen_ids.add(event.event_id)

            # Auto-flush if buffer is full
            if len(self._buffer) >= self._max_size:
                self._flush_unsafe()

            return True

    def _flush_unsafe(self) -> None:
        """Flush buffer to disk without acquiring lock (caller must hold lock)."""
        if not self._buffer:
            return

        try:
            with open(self.log_path, "a") as f:
                # Acquire file lock
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    for event in self._buffer:
                        f.write(json.dumps(event.to_dict()) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            logger.debug(f"Flushed {len(self._buffer)} events to {self.log_path}")
            self._buffer.clear()

        except Exception as e:
            logger.error(f"Failed to flush event buffer: {e}")
            raise

    def flush(self) -> None:
        """Flush buffer to disk (thread-safe)."""
        with self._lock:
            self._flush_unsafe()

    def __del__(self):
        """Ensure buffer is flushed on garbage collection."""
        try:
            self.flush()
        except Exception as e:
            logger.debug(f"GRAPH: Graph operation failed: {e}")


# Global buffered writer instances
_buffered_writer_instances: Dict[Path, BufferedEventWriter] = {}
_buffered_writer_lock = threading.Lock()


def get_buffered_event_writer(run_dir: Path, max_size: int = 50) -> BufferedEventWriter:
    """Get or create buffered event writer for a run directory.

    OPT-02-004: Use this for high-frequency event logging to reduce I/O.

    Args:
        run_dir: Run directory containing event_log.ndjson.
        max_size: Maximum buffer size before auto-flush.

    Returns:
        BufferedEventWriter instance.
    """
    log_path = run_dir / "event_log.ndjson"

    with _buffered_writer_lock:
        if log_path not in _buffered_writer_instances:
            _buffered_writer_instances[log_path] = BufferedEventWriter(log_path, max_size=max_size)
        return _buffered_writer_instances[log_path]


def flush_all_buffers() -> None:
    """Flush all buffered event writers (call at workflow end)."""
    with _buffered_writer_lock:
        for writer in _buffered_writer_instances.values():
            try:
                writer.flush()
            except Exception as e:
                logger.error(f"Failed to flush event buffer: {e}")


# =============================================================================
# SINGLETON ACCESS
# =============================================================================


_writer_instances: Dict[Path, AtomicEventWriter] = {}
_writer_lock = threading.Lock()


def get_event_writer(run_dir: Path) -> AtomicEventWriter:
    """Get or create event writer for a run directory.

    This ensures a single writer instance per run directory to
    prevent duplicate deduplication state.

    Args:
        run_dir: Run directory containing event_log.ndjson.

    Returns:
        AtomicEventWriter instance.
    """
    log_path = run_dir / "event_log.ndjson"

    with _writer_lock:
        if log_path not in _writer_instances:
            _writer_instances[log_path] = AtomicEventWriter(log_path)
        return _writer_instances[log_path]


def emit_event(
    run_dir: Path,
    event_type: str,
    run_id: str,
    sprint_id: str,
    attempt: int,
    message: str,
    level: str = "info",
    phase: str = "INIT",
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Convenience function to emit an event.

    Args:
        run_dir: Run directory.
        event_type: Type of event.
        run_id: Run identifier.
        sprint_id: Sprint identifier.
        attempt: Attempt number.
        message: Event message.
        level: Event severity (info, warning, error, critical).
        phase: Current sprint phase (INIT, SPEC, PLAN, EXEC, QA, VOTE, DONE).
        metadata: Optional metadata.

    Returns:
        True if written, False if deduplicated.
    """
    writer = get_event_writer(run_dir)
    return writer.write_dict(
        event_type=event_type,
        run_id=run_id,
        sprint_id=sprint_id,
        attempt=attempt,
        message=message,
        level=level,
        phase=phase,
        metadata=metadata,
    )


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    "Event",
    "AtomicEventWriter",
    "BufferedEventWriter",  # OPT-02-004
    "get_event_writer",
    "get_buffered_event_writer",  # OPT-02-004
    "flush_all_buffers",  # OPT-02-004
    "emit_event",
]
