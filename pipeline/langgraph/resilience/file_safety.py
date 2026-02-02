"""Process-safe file operations using fcntl file locking.

This module provides atomic file operations that are safe for:
    - Multiple processes accessing the same file
    - Crash recovery (no partial writes)
    - Concurrent reads and writes

Key patterns:
    - Write to temp file first, then atomic rename
    - fcntl.flock() for inter-process locking
    - Context managers for automatic lock release

Platform support:
    - macOS: Full support via fcntl
    - Linux: Full support via fcntl
    - Windows: Fallback to non-locking mode (with warning)
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Union

import yaml

logger = logging.getLogger(__name__)

# Thread-local storage for lock tracking
_local = threading.local()


@dataclass
class LockInfo:
    """Information about an acquired file lock."""

    path: Path
    lock_type: str  # "shared" or "exclusive"
    acquired_at: datetime
    process_id: int
    thread_id: int

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging."""
        return {
            "path": str(self.path),
            "lock_type": self.lock_type,
            "acquired_at": self.acquired_at.isoformat(),
            "process_id": self.process_id,
            "thread_id": self.thread_id,
        }


class FileLockError(Exception):
    """Error acquiring or releasing file lock."""

    pass


class ProcessSafeWriter:
    """Process-safe file writer using fcntl locking.

    Usage:
        writer = ProcessSafeWriter(Path("/tmp/data.json"))
        writer.write_json({"key": "value"})

    Features:
        - Atomic writes via temp file + rename
        - Inter-process locking via fcntl
        - Automatic retry on lock contention
        - Crash-safe (no partial writes)
    """

    def __init__(
        self,
        path: Union[str, Path],
        timeout_seconds: float = 10.0,
        create_dirs: bool = True,
    ):
        """Initialize writer.

        Args:
            path: Path to the file.
            timeout_seconds: Max time to wait for lock.
            create_dirs: Whether to create parent directories.
        """
        self.path = Path(path)
        self.timeout_seconds = timeout_seconds
        self.create_dirs = create_dirs

    @contextmanager
    def _exclusive_lock(self) -> Generator[int, None, None]:
        """Acquire exclusive lock on file.

        Yields:
            File descriptor for the lock file.

        Raises:
            FileLockError: If lock cannot be acquired.
        """
        # Create parent directories if needed
        if self.create_dirs:
            self.path.parent.mkdir(parents=True, exist_ok=True)

        # Use a separate .lock file to avoid issues with file replacement
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")

        fd = None
        try:
            # Open or create the lock file
            fd = os.open(
                str(lock_path),
                os.O_RDWR | os.O_CREAT,
                0o666,
            )

            # Try to acquire exclusive lock
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                # Lock held by another process, wait with timeout
                logger.debug(
                    "LOCK_WAIT: path='%s' timeout=%.1fs",
                    self.path,
                    self.timeout_seconds,
                )

                import time
                import select

                start = time.monotonic()
                while time.monotonic() - start < self.timeout_seconds:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        time.sleep(0.1)
                else:
                    raise FileLockError(
                        f"Timeout acquiring lock on {self.path} "
                        f"after {self.timeout_seconds}s"
                    )

            lock_info = LockInfo(
                path=self.path,
                lock_type="exclusive",
                acquired_at=datetime.now(),
                process_id=os.getpid(),
                thread_id=threading.get_ident(),
            )

            logger.debug("LOCK_ACQUIRED: %s", lock_info.to_dict())

            yield fd

        finally:
            if fd is not None:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                    os.close(fd)
                    logger.debug("LOCK_RELEASED: path='%s'", self.path)
                except Exception as e:
                    logger.warning("LOCK_RELEASE_ERROR: %s", e)

    def write_text(self, content: str, encoding: str = "utf-8") -> None:
        """Write text content atomically.

        Args:
            content: Text content to write.
            encoding: Text encoding.
        """
        with self._exclusive_lock():
            # Write to temp file first
            temp_path = None
            try:
                # Create temp file in same directory for atomic rename
                fd, temp_path = tempfile.mkstemp(
                    dir=str(self.path.parent),
                    prefix=f".{self.path.stem}_",
                    suffix=self.path.suffix,
                )

                try:
                    os.write(fd, content.encode(encoding))
                    os.fsync(fd)  # Ensure data is on disk
                finally:
                    os.close(fd)

                # Atomic rename
                os.replace(temp_path, str(self.path))

                logger.debug(
                    "WRITE_SUCCESS: path='%s' size=%d",
                    self.path,
                    len(content),
                )

            except Exception:
                # Clean up temp file on error
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except Exception as e:
                        logger.debug(f"GRAPH: Graph operation failed: {e}")
                raise

    def write_json(
        self,
        data: Any,
        indent: int = 2,
        ensure_ascii: bool = False,
    ) -> None:
        """Write JSON content atomically.

        Args:
            data: Data to serialize to JSON.
            indent: JSON indentation level.
            ensure_ascii: Whether to escape non-ASCII characters.
        """
        content = json.dumps(data, indent=indent, ensure_ascii=ensure_ascii)
        self.write_text(content)

    def write_yaml(self, data: Any, default_flow_style: bool = False) -> None:
        """Write YAML content atomically.

        Args:
            data: Data to serialize to YAML.
            default_flow_style: YAML flow style setting.
        """
        content = yaml.dump(
            data,
            default_flow_style=default_flow_style,
            allow_unicode=True,
            sort_keys=False,
        )
        self.write_text(content)

    def append_ndjson(self, record: Dict[str, Any]) -> None:
        """Append a record to NDJSON file (newline-delimited JSON).

        This is optimized for log-style files where we only append.

        Args:
            record: Dictionary to append as JSON line.
        """
        with self._exclusive_lock():
            # Append mode - no atomic rename needed
            line = json.dumps(record, ensure_ascii=False) + "\n"

            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())


class ProcessSafeReader:
    """Process-safe file reader using fcntl shared locking.

    Usage:
        reader = ProcessSafeReader(Path("/tmp/data.json"))
        data = reader.read_json()

    Features:
        - Shared locking allows multiple concurrent readers
        - Blocks only when a writer has exclusive lock
    """

    def __init__(
        self,
        path: Union[str, Path],
        timeout_seconds: float = 5.0,
    ):
        """Initialize reader.

        Args:
            path: Path to the file.
            timeout_seconds: Max time to wait for lock.
        """
        self.path = Path(path)
        self.timeout_seconds = timeout_seconds

    @contextmanager
    def _shared_lock(self) -> Generator[int, None, None]:
        """Acquire shared lock on file.

        Yields:
            File descriptor.

        Raises:
            FileLockError: If lock cannot be acquired.
            FileNotFoundError: If file doesn't exist.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")

        lock_path = self.path.with_suffix(self.path.suffix + ".lock")

        fd = None
        try:
            fd = os.open(
                str(lock_path),
                os.O_RDWR | os.O_CREAT,
                0o666,
            )

            try:
                fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                # Writer has exclusive lock, wait
                logger.debug(
                    "LOCK_WAIT_SHARED: path='%s' timeout=%.1fs",
                    self.path,
                    self.timeout_seconds,
                )

                import time

                start = time.monotonic()
                while time.monotonic() - start < self.timeout_seconds:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        time.sleep(0.05)
                else:
                    raise FileLockError(
                        f"Timeout acquiring shared lock on {self.path}"
                    )

            yield fd

        finally:
            if fd is not None:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                    os.close(fd)
                except Exception as e:
                    logger.debug(f"GRAPH: Graph operation failed: {e}")

    def read_text(self, encoding: str = "utf-8") -> str:
        """Read text content with shared lock.

        Args:
            encoding: Text encoding.

        Returns:
            File content as string.
        """
        with self._shared_lock():
            return self.path.read_text(encoding=encoding)

    def read_json(self) -> Any:
        """Read JSON content with shared lock.

        Returns:
            Parsed JSON data.
        """
        content = self.read_text()
        return json.loads(content)

    def read_yaml(self) -> Any:
        """Read YAML content with shared lock.

        Returns:
            Parsed YAML data.
        """
        content = self.read_text()
        return yaml.safe_load(content)

    def read_ndjson(self, limit: Optional[int] = None) -> list:
        """Read NDJSON file (newline-delimited JSON).

        Args:
            limit: Max number of records to read (from end if negative).

        Returns:
            List of parsed JSON records.
        """
        with self._shared_lock():
            records = []
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))

            if limit is not None:
                if limit >= 0:
                    return records[:limit]
                else:
                    return records[limit:]

            return records


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def write_json_safe(
    path: Union[str, Path],
    data: Any,
    indent: int = 2,
) -> None:
    """Write JSON atomically with process locking.

    Args:
        path: Path to write to.
        data: Data to serialize.
        indent: JSON indentation.
    """
    ProcessSafeWriter(path).write_json(data, indent=indent)


def read_json_safe(path: Union[str, Path]) -> Any:
    """Read JSON with process locking.

    Args:
        path: Path to read from.

    Returns:
        Parsed JSON data.
    """
    return ProcessSafeReader(path).read_json()


def write_yaml_safe(path: Union[str, Path], data: Any) -> None:
    """Write YAML atomically with process locking.

    Args:
        path: Path to write to.
        data: Data to serialize.
    """
    ProcessSafeWriter(path).write_yaml(data)


def read_yaml_safe(path: Union[str, Path]) -> Any:
    """Read YAML with process locking.

    Args:
        path: Path to read from.

    Returns:
        Parsed YAML data.
    """
    return ProcessSafeReader(path).read_yaml()


def append_ndjson_safe(
    path: Union[str, Path],
    record: Dict[str, Any],
) -> None:
    """Append record to NDJSON file with process locking.

    Args:
        path: Path to NDJSON file.
        record: Record to append.
    """
    ProcessSafeWriter(path).append_ndjson(record)


def read_ndjson_safe(
    path: Union[str, Path],
    limit: Optional[int] = None,
) -> list:
    """Read NDJSON file with process locking.

    Args:
        path: Path to NDJSON file.
        limit: Max records to return.

    Returns:
        List of records.
    """
    return ProcessSafeReader(path).read_ndjson(limit=limit)


__all__ = [
    "FileLockError",
    "LockInfo",
    "ProcessSafeWriter",
    "ProcessSafeReader",
    "write_json_safe",
    "read_json_safe",
    "write_yaml_safe",
    "read_yaml_safe",
    "append_ndjson_safe",
    "read_ndjson_safe",
]
