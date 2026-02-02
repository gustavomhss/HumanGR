"""LangGraph Checkpointer - Redis and File-based Persistence.

This module provides checkpoint persistence for LangGraph workflows,
enabling resume from any point after crashes or SAFE_HALT.

Storage Options:
1. Redis: Fast, canonical for active runs
2. File: Durable backup, used for dashboard and long-term storage
3. Hybrid: Redis as primary, file as backup (recommended)

Based on: MIGRATION_V2_TO_LANGGRAPH.md + PIPELINE_V3_MASTER_PLAN.md Section 10.4

Invariants enforced:
- I1: All checkpoints namespaced by run_id
- I2: Idempotent checkpoint operations
- I3: Redis is canonical, file is mirror

Author: Pipeline Autonomo Team
Version: 1.0.0 (2026-01-16)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# =============================================================================
# LAZY CONTEXT PACK SERIALIZATION (P0-05)
# =============================================================================


def _serialize_context_pack(context_pack: Any) -> Optional[Dict[str, Any]]:
    """Serialize context pack for checkpoint storage.

    P0-05: If context_pack is a LazyContextPack, serialize only the reference
    (pack_id, pack_path, sha256) to minimize checkpoint size. On restore,
    the full pack can be loaded lazily.

    Args:
        context_pack: ContextPackState dict or LazyContextPack instance.

    Returns:
        Serialized context pack dict, or None if context_pack is None.
    """
    if context_pack is None:
        return None

    # Check if it's a LazyContextPack
    # Import here to avoid circular imports
    try:
        from pipeline.langgraph.state import LazyContextPack

        if isinstance(context_pack, LazyContextPack):
            # Serialize as reference only
            return context_pack.to_reference()
    except ImportError as e:
        logger.debug(f"CHECKPOINT: Checkpoint operation failed: {e}")

    # Regular dict - check if it has _lazy marker (already serialized reference)
    if isinstance(context_pack, dict):
        if context_pack.get("_lazy") == "true":
            # Already a lazy reference, keep as-is
            return context_pack
        # Regular ContextPackState dict - serialize fully
        return dict(context_pack)

    # Unknown type - attempt to convert to dict
    if hasattr(context_pack, "to_dict"):
        return context_pack.to_dict()

    return None


def _deserialize_context_pack(data: Optional[Dict[str, Any]]) -> Any:
    """Deserialize context pack from checkpoint storage.

    P0-05: If the serialized data is a lazy reference (has _lazy marker),
    create a LazyContextPack that will load the full data on first access.

    Args:
        data: Serialized context pack dict or None.

    Returns:
        LazyContextPack (if lazy reference) or ContextPackState dict.
    """
    if data is None:
        return None

    # Check for lazy marker
    if data.get("_lazy") == "true":
        try:
            from pipeline.langgraph.state import LazyContextPack

            return LazyContextPack(
                pack_id=data.get("pack_id", ""),
                pack_path=data.get("pack_path", ""),
                sha256=data.get("sha256", ""),
            )
        except ImportError:
            # LazyContextPack not available, remove marker and return as-is
            result = dict(data)
            result.pop("_lazy", None)
            return result

    # Regular ContextPackState dict
    return data


def _serialize_state_for_checkpoint(state: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare state dict for checkpoint serialization.

    P0-05: Handles LazyContextPack serialization and other special types.

    Args:
        state: Pipeline state dict.

    Returns:
        State dict ready for JSON serialization.
    """
    result = dict(state)

    # Handle context_pack
    if "context_pack" in result and result["context_pack"] is not None:
        result["context_pack"] = _serialize_context_pack(result["context_pack"])

    return result


def _deserialize_state_from_checkpoint(state: Dict[str, Any]) -> Dict[str, Any]:
    """Restore state dict from checkpoint deserialization.

    P0-05: Handles LazyContextPack deserialization and other special types.

    Args:
        state: Deserialized state dict from checkpoint.

    Returns:
        State dict with restored types.
    """
    result = dict(state)

    # Handle context_pack
    if "context_pack" in result:
        result["context_pack"] = _deserialize_context_pack(result.get("context_pack"))

    return result


# =============================================================================
# CHECKPOINT TYPES
# =============================================================================


@dataclass
class Checkpoint:
    """A single checkpoint snapshot.

    Contains the full state at a point in time, with metadata
    for identification and verification.
    """

    checkpoint_id: str
    run_id: str
    sprint_id: str
    thread_id: str
    ts: str  # ISO timestamp
    state: Dict[str, Any]
    parent_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        P0-05: Uses _serialize_state_for_checkpoint to handle LazyContextPack
        serialization, storing only a reference instead of the full pack data.
        """
        return {
            "checkpoint_id": self.checkpoint_id,
            "run_id": self.run_id,
            "sprint_id": self.sprint_id,
            "thread_id": self.thread_id,
            "ts": self.ts,
            "state": _serialize_state_for_checkpoint(self.state),
            "parent_id": self.parent_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Checkpoint":
        """Create from dictionary.

        P0-05: Uses _deserialize_state_from_checkpoint to restore LazyContextPack
        instances from serialized references.
        """
        return cls(
            checkpoint_id=data["checkpoint_id"],
            run_id=data["run_id"],
            sprint_id=data["sprint_id"],
            thread_id=data["thread_id"],
            ts=data["ts"],
            state=_deserialize_state_from_checkpoint(data["state"]),
            parent_id=data.get("parent_id"),
            metadata=data.get("metadata"),
        )


def generate_checkpoint_id(
    run_id: str,
    sprint_id: str,
    node: str,
    attempt: int,
) -> str:
    """Generate deterministic checkpoint ID.

    Invariant I2: Deterministic keys for idempotency.

    FIX 2026-01-27: Use underscores instead of colons to avoid filesystem issues.
    Colons are invalid in filenames on Windows and some other systems.
    """
    base = f"{run_id}_{sprint_id}_{node}_attempt_{attempt}"
    hash_suffix = hashlib.sha256(base.encode()).hexdigest()[:8]
    return f"ckpt_{base}_{hash_suffix}"


# =============================================================================
# ABSTRACT CHECKPOINTER
# =============================================================================


class BaseCheckpointer(ABC):
    """Abstract base class for checkpointer implementations."""

    @abstractmethod
    async def put(
        self,
        checkpoint: Checkpoint,
    ) -> None:
        """Save a checkpoint.

        Args:
            checkpoint: Checkpoint to save.
        """
        pass

    @abstractmethod
    async def get(
        self,
        checkpoint_id: str,
    ) -> Optional[Checkpoint]:
        """Retrieve a checkpoint by ID.

        Args:
            checkpoint_id: Checkpoint identifier.

        Returns:
            Checkpoint if found, None otherwise.
        """
        pass

    @abstractmethod
    async def get_latest(
        self,
        thread_id: str,
    ) -> Optional[Checkpoint]:
        """Get the latest checkpoint for a thread.

        Args:
            thread_id: Thread identifier (usually run_id).

        Returns:
            Latest checkpoint if any, None otherwise.
        """
        pass

    @abstractmethod
    async def list(
        self,
        thread_id: str,
        limit: int = 10,
    ) -> List[Checkpoint]:
        """List checkpoints for a thread.

        Args:
            thread_id: Thread identifier.
            limit: Maximum number of checkpoints to return.

        Returns:
            List of checkpoints, most recent first.
        """
        pass

    @abstractmethod
    async def delete(
        self,
        checkpoint_id: str,
    ) -> bool:
        """Delete a checkpoint.

        Args:
            checkpoint_id: Checkpoint to delete.

        Returns:
            True if deleted, False if not found.
        """
        pass


# =============================================================================
# FILE-BASED CHECKPOINTER
# =============================================================================


class FileCheckpointer(BaseCheckpointer):
    """File-based checkpoint storage with thread safety and atomic writes.

    Stores checkpoints as JSON files in the run directory.
    Structure:
        {run_dir}/checkpoints/
            {checkpoint_id}.json
            index.json  # Index of all checkpoints

    Thread Safety (SAFE-03):
    - _index_lock protects index reads/writes
    - _write_lock protects checkpoint file writes
    - Atomic writes via temp file + rename

    Durability:
    - Index flushed after each batch of writes
    - Atomic rename prevents partial writes
    """

    def __init__(self, run_dir: Path):
        """Initialize file checkpointer.

        Args:
            run_dir: Directory for run artifacts.
        """
        self.run_dir = run_dir
        self.checkpoints_dir = run_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.checkpoints_dir / "index.json"

        # SAFE-03: Thread safety locks
        self._index_lock = threading.RLock()
        self._write_lock = threading.Lock()

        # OPT-03-004: Index cache with TTL
        self._index_cache: Dict[str, List[str]] = {}
        self._index_cache_timestamp: float = 0.0
        self._index_cache_ttl: float = 2.0  # seconds

        # SAFE-03: Index flush tracking
        self._index_dirty = False
        self._write_count = 0
        self._flush_interval = 10  # Flush index every 10 writes

    def _load_index(self) -> Dict[str, List[str]]:
        """Load checkpoint index with caching (OPT-03-004).

        Thread-safe via _index_lock.
        """
        with self._index_lock:
            now = time.time()

            # Return cached index if still valid (check timestamp > 0 to detect initialized cache)
            if self._index_cache_timestamp > 0 and now - self._index_cache_timestamp < self._index_cache_ttl:
                return self._index_cache

            # Cache miss or expired - load from disk
            if self.index_path.exists():
                try:
                    with open(self.index_path) as f:
                        self._index_cache = json.load(f)
                        self._index_cache_timestamp = now
                        return self._index_cache
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning(f"Failed to load checkpoint index: {e}")

            self._index_cache = {}
            self._index_cache_timestamp = now
            return self._index_cache

    def _invalidate_index_cache(self) -> None:
        """Invalidate index cache after writes (OPT-03-004)."""
        self._index_cache_timestamp = 0.0

    def _save_index(self, index: Dict[str, List[str]]) -> None:
        """Save checkpoint index and invalidate cache (OPT-03-004).

        Non-atomic version - use _save_index_atomic for crash safety.
        """
        try:
            with open(self.index_path, "w") as f:
                json.dump(index, f)  # OPT-03-001: Removed indent for faster persistence
            # OPT-03-004: Update cache with new index
            self._index_cache = index
            self._index_cache_timestamp = time.time()
            self._index_dirty = False
        except Exception as e:
            logger.error(f"Failed to save checkpoint index: {e}")
            self._invalidate_index_cache()  # OPT-03-004: Invalidate on error

    def _save_index_atomic(self) -> None:
        """Save index atomically (temp file + rename).

        SAFE-03: Crash-safe index persistence.
        Must be called with _index_lock held.
        """
        if self._index_cache is None:
            return

        try:
            # Write to temp file in same directory (required for atomic rename)
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self.checkpoints_dir),
                suffix=".index.tmp"
            )
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(self._index_cache, f)

                # Atomic rename (same filesystem required)
                os.replace(tmp_path, str(self.index_path))
                self._index_dirty = False
                logger.debug("Index saved atomically")

            except Exception:
                # Cleanup temp file on error
                if os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError as e:
                        logger.debug(f"CHECKPOINT: Checkpoint operation failed: {e}")
                raise

        except OSError as e:
            logger.error(f"Failed to save index atomically: {e}")
            # Fallback to non-atomic save
            self._save_index(self._index_cache)

    async def put(self, checkpoint: Checkpoint) -> None:
        """Save checkpoint to file with thread safety and atomic writes.

        SAFE-03:
        1. Write checkpoint file atomically
        2. Update in-memory index
        3. Flush index periodically
        """
        # Phase 1: Write checkpoint file atomically
        with self._write_lock:
            checkpoint_path = self.checkpoints_dir / f"{checkpoint.checkpoint_id}.json"

            try:
                # Atomic write via temp file + rename
                fd, tmp_path = tempfile.mkstemp(
                    dir=str(self.checkpoints_dir),
                    suffix=".ckpt.tmp"
                )
                try:
                    with os.fdopen(fd, "w") as f:
                        json.dump(checkpoint.to_dict(), f)

                    os.replace(tmp_path, str(checkpoint_path))

                except Exception:
                    if os.path.exists(tmp_path):
                        try:
                            os.unlink(tmp_path)
                        except OSError as e:
                            logger.debug(f"CHECKPOINT: Checkpoint operation failed: {e}")
                    raise

            except OSError as e:
                logger.error(f"Failed to save checkpoint {checkpoint.checkpoint_id}: {e}")
                raise

        # Phase 2: Update index (separate lock)
        with self._index_lock:
            index = self._load_index()
            thread_id = checkpoint.thread_id

            if thread_id not in index:
                index[thread_id] = []

            if checkpoint.checkpoint_id not in index[thread_id]:
                # Add to front (most recent first)
                index[thread_id].insert(0, checkpoint.checkpoint_id)
                # Keep only last 100 checkpoints per thread
                index[thread_id] = index[thread_id][:100]

            self._index_dirty = True
            self._write_count += 1
            self._index_cache = index
            self._index_cache_timestamp = time.time()

            # Flush periodically
            if self._write_count >= self._flush_interval:
                self._save_index_atomic()
                self._write_count = 0
            else:
                # Always save to maintain backward compatibility
                self._save_index(index)

    async def get(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Retrieve checkpoint from file."""
        checkpoint_path = self.checkpoints_dir / f"{checkpoint_id}.json"
        if not checkpoint_path.exists():
            return None

        try:
            with open(checkpoint_path) as f:
                data = json.load(f)
                return Checkpoint.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            return None

    async def get_latest(self, thread_id: str) -> Optional[Checkpoint]:
        """Get latest checkpoint for thread."""
        index = self._load_index()
        checkpoint_ids = index.get(thread_id, [])

        if not checkpoint_ids:
            return None

        return await self.get(checkpoint_ids[0])

    async def list(self, thread_id: str, limit: int = 10) -> List[Checkpoint]:
        """List checkpoints for thread."""
        index = self._load_index()
        checkpoint_ids = index.get(thread_id, [])[:limit]

        checkpoints = []
        for checkpoint_id in checkpoint_ids:
            checkpoint = await self.get(checkpoint_id)
            if checkpoint:
                checkpoints.append(checkpoint)

        return checkpoints

    async def delete(self, checkpoint_id: str) -> bool:
        """Delete checkpoint file."""
        checkpoint_path = self.checkpoints_dir / f"{checkpoint_id}.json"
        if not checkpoint_path.exists():
            return False

        try:
            checkpoint_path.unlink()

            # Update index
            with self._index_lock:
                index = self._load_index()
                for thread_id in index:
                    if checkpoint_id in index[thread_id]:
                        index[thread_id].remove(checkpoint_id)
                self._save_index(index)

            return True
        except Exception as e:
            logger.error(f"Failed to delete checkpoint {checkpoint_id}: {e}")
            return False

    def flush(self) -> None:
        """Force flush index to disk.

        SAFE-03: Call this to ensure all pending index updates are persisted.
        """
        with self._index_lock:
            if self._index_dirty:
                self._save_index_atomic()
                self._write_count = 0

    def close(self) -> None:
        """Ensure index is flushed on close.

        SAFE-03: Always call this or use context manager to prevent data loss.
        """
        self.flush()

    def __enter__(self) -> "FileCheckpointer":
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit - flush on close."""
        self.close()

    def __del__(self) -> None:
        """Destructor - attempt to flush on garbage collection."""
        try:
            self.flush()
        except Exception as e:
            logger.debug(f"CHECKPOINT: Checkpoint operation failed: {e}")


# =============================================================================
# REDIS-BASED CHECKPOINTER
# =============================================================================


class RedisCheckpointer(BaseCheckpointer):
    """Redis-based checkpoint storage.

    Stores checkpoints in Redis with structure:
        pipeline:checkpoints:{thread_id}:{checkpoint_id} -> JSON checkpoint
        pipeline:checkpoints:{thread_id}:index -> sorted set of checkpoint IDs
    """

    def __init__(self, redis_client: Any, ttl_seconds: int = 86400 * 7):
        """Initialize Redis checkpointer.

        Args:
            redis_client: Redis client instance.
            ttl_seconds: TTL for checkpoints (default: 7 days).
        """
        self.redis = redis_client
        self.ttl = ttl_seconds

    def _key(self, thread_id: str, checkpoint_id: str) -> str:
        """Generate Redis key for checkpoint."""
        return f"pipeline:checkpoints:{thread_id}:{checkpoint_id}"

    def _index_key(self, thread_id: str) -> str:
        """Generate Redis key for checkpoint index."""
        return f"pipeline:checkpoints:{thread_id}:index"

    async def put(self, checkpoint: Checkpoint) -> None:
        """Save checkpoint to Redis."""
        key = self._key(checkpoint.thread_id, checkpoint.checkpoint_id)
        index_key = self._index_key(checkpoint.thread_id)

        try:
            # Save checkpoint
            await self.redis.set(
                key,
                json.dumps(checkpoint.to_dict()),
                ex=self.ttl,
            )

            # Update index (sorted set with timestamp as score)
            # FIX 2026-01-27: Handle various ISO timestamp formats properly
            ts_str = checkpoint.ts
            if ts_str.endswith("Z"):
                ts_str = ts_str[:-1] + "+00:00"
            elif "+" not in ts_str and "-" not in ts_str[-6:]:
                # No timezone info - assume UTC
                ts_str = ts_str + "+00:00"
            try:
                ts = datetime.fromisoformat(ts_str)
            except ValueError:
                # Fallback to current time if parsing fails
                ts = datetime.now(timezone.utc)
            score = ts.timestamp()
            await self.redis.zadd(
                index_key,
                {checkpoint.checkpoint_id: score},
            )

            # Trim index to last 100
            await self.redis.zremrangebyrank(index_key, 0, -101)

        except Exception as e:
            logger.error(f"Failed to save checkpoint to Redis: {e}")
            raise

    async def get(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Retrieve checkpoint from Redis.

        Note: This requires knowing the thread_id, which we extract from checkpoint_id.
        """
        # checkpoint_id format: ckpt_{run_id}_{sprint_id}_{node}_attempt_{n}_{hash}
        # Also support legacy format: lg_{timestamp}_{suffix}_final
        # thread_id is typically run_id
        if checkpoint_id.startswith("ckpt_"):
            # Format: ckpt_{run_id}_{sprint_id}_{node}_attempt_{n}_{hash}
            parts = checkpoint_id.split("_")
            if len(parts) < 3:
                return None
            run_id = parts[1]
        elif checkpoint_id.startswith("lg_"):
            # Format: lg_{timestamp}_{suffix}_final - run_id IS the checkpoint prefix
            run_id = checkpoint_id.replace("_final", "")
        else:
            # Try colon format for backward compatibility
            parts = checkpoint_id.split(":")
            if len(parts) >= 2:
                run_id = parts[1]
            else:
                return None

        key = self._key(run_id, checkpoint_id)

        try:
            data = await self.redis.get(key)
            if not data:
                return None
            return Checkpoint.from_dict(json.loads(data))
        except Exception as e:
            logger.error(f"Failed to get checkpoint from Redis: {e}")
            return None

    async def get_latest(self, thread_id: str) -> Optional[Checkpoint]:
        """Get latest checkpoint from Redis."""
        index_key = self._index_key(thread_id)

        try:
            # Get highest-scored (most recent) checkpoint ID
            result = await self.redis.zrevrange(index_key, 0, 0)
            if not result:
                return None

            checkpoint_id = result[0]
            if isinstance(checkpoint_id, bytes):
                checkpoint_id = checkpoint_id.decode()

            return await self._get_by_thread(thread_id, checkpoint_id)
        except Exception as e:
            logger.error(f"Failed to get latest checkpoint from Redis: {e}")
            return None

    async def _get_by_thread(self, thread_id: str, checkpoint_id: str) -> Optional[Checkpoint]:
        """Get checkpoint knowing the thread_id."""
        key = self._key(thread_id, checkpoint_id)

        try:
            data = await self.redis.get(key)
            if not data:
                return None
            if isinstance(data, bytes):
                data = data.decode()
            return Checkpoint.from_dict(json.loads(data))
        except Exception as e:
            logger.error(f"Failed to get checkpoint from Redis: {e}")
            return None

    async def list(self, thread_id: str, limit: int = 10) -> List[Checkpoint]:
        """List checkpoints from Redis."""
        index_key = self._index_key(thread_id)

        try:
            checkpoint_ids = await self.redis.zrevrange(index_key, 0, limit - 1)

            checkpoints = []
            for checkpoint_id in checkpoint_ids:
                if isinstance(checkpoint_id, bytes):
                    checkpoint_id = checkpoint_id.decode()
                checkpoint = await self._get_by_thread(thread_id, checkpoint_id)
                if checkpoint:
                    checkpoints.append(checkpoint)

            return checkpoints
        except Exception as e:
            logger.error(f"Failed to list checkpoints from Redis: {e}")
            return []

    async def delete(self, checkpoint_id: str) -> bool:
        """Delete checkpoint from Redis."""
        # Extract run_id from checkpoint_id (same logic as get())
        if checkpoint_id.startswith("ckpt_"):
            parts = checkpoint_id.split("_")
            if len(parts) < 3:
                return False
            run_id = parts[1]
        elif checkpoint_id.startswith("lg_"):
            run_id = checkpoint_id.replace("_final", "")
        else:
            parts = checkpoint_id.split(":")
            if len(parts) >= 2:
                run_id = parts[1]
            else:
                return False

        key = self._key(run_id, checkpoint_id)
        index_key = self._index_key(run_id)

        try:
            await self.redis.delete(key)
            await self.redis.zrem(index_key, checkpoint_id)
            return True
        except Exception as e:
            logger.error(f"Failed to delete checkpoint from Redis: {e}")
            return False


# =============================================================================
# HYBRID CHECKPOINTER
# =============================================================================


class HybridCheckpointer(BaseCheckpointer):
    """Hybrid checkpointer using Redis as primary and file as backup.

    Invariant I3: Redis is canonical, file is mirror.

    Write path: Redis -> File (async mirror)
    Read path: Redis -> File (fallback on miss)
    """

    def __init__(
        self,
        redis_checkpointer: RedisCheckpointer,
        file_checkpointer: FileCheckpointer,
        mirror_to_file: bool = True,
    ):
        """Initialize hybrid checkpointer.

        Args:
            redis_checkpointer: Redis checkpointer instance.
            file_checkpointer: File checkpointer instance.
            mirror_to_file: Whether to mirror writes to file.
        """
        self.redis = redis_checkpointer
        self.file = file_checkpointer
        self.mirror_to_file = mirror_to_file

    async def put(self, checkpoint: Checkpoint) -> None:
        """Save checkpoint to Redis, mirror to file."""
        # Primary: Redis
        await self.redis.put(checkpoint)

        # Mirror: File (non-blocking)
        if self.mirror_to_file:
            try:
                await self.file.put(checkpoint)
            except Exception as e:
                # Invariant I11: Sink failures don't block
                logger.warning(f"Failed to mirror checkpoint to file (non-blocking): {e}")

    async def get(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Get checkpoint from Redis, fallback to file."""
        # Try Redis first
        checkpoint = await self.redis.get(checkpoint_id)
        if checkpoint:
            return checkpoint

        # Fallback to file
        checkpoint = await self.file.get(checkpoint_id)
        if checkpoint:
            # Re-warm Redis cache
            try:
                await self.redis.put(checkpoint)
            except Exception as e:
                logger.debug(f"CHECKPOINT: Checkpoint operation failed: {e}")
            return checkpoint

        return None

    async def get_latest(self, thread_id: str) -> Optional[Checkpoint]:
        """Get latest checkpoint from Redis, fallback to file."""
        checkpoint = await self.redis.get_latest(thread_id)
        if checkpoint:
            return checkpoint

        return await self.file.get_latest(thread_id)

    async def list(self, thread_id: str, limit: int = 10) -> List[Checkpoint]:
        """List checkpoints from Redis, fallback to file."""
        checkpoints = await self.redis.list(thread_id, limit)
        if checkpoints:
            return checkpoints

        return await self.file.list(thread_id, limit)

    async def delete(self, checkpoint_id: str) -> bool:
        """Delete checkpoint from both stores."""
        redis_deleted = await self.redis.delete(checkpoint_id)
        file_deleted = await self.file.delete(checkpoint_id)
        return redis_deleted or file_deleted


# =============================================================================
# LANGGRAPH ADAPTER
# =============================================================================


def create_langgraph_checkpointer(
    checkpointer: BaseCheckpointer,
) -> Any:
    """Create a LangGraph-compatible checkpointer adapter.

    This wraps our checkpointer to match LangGraph's expected interface.
    Updated for LangGraph 0.2.x which requires get_tuple/aget_tuple interface.

    Args:
        checkpointer: Our checkpointer implementation.

    Returns:
        LangGraph-compatible checkpointer, or None if LangGraph not available.
    """
    try:
        from langgraph.checkpoint.base import (
            BaseCheckpointSaver,
            CheckpointTuple,
            ChannelVersions,
        )
        from langgraph.checkpoint.base import Checkpoint as LGCheckpoint
        from langgraph.checkpoint.base import CheckpointMetadata
    except ImportError:
        logger.warning("LangGraph not available, cannot create adapter")
        return None

    from typing import Iterator, AsyncIterator

    class LangGraphCheckpointerAdapter(BaseCheckpointSaver):
        """Adapter to make our checkpointer LangGraph-compatible.

        Implements the new LangGraph 0.2.x interface with get_tuple/aget_tuple.
        """

        def __init__(self, inner: BaseCheckpointer):
            super().__init__()
            self.inner = inner

        def _make_checkpoint_tuple(
            self,
            our_checkpoint: Checkpoint,
            config: Dict[str, Any],
        ) -> CheckpointTuple:
            """Convert our Checkpoint to LangGraph CheckpointTuple."""
            thread_id = our_checkpoint.thread_id
            checkpoint_id = our_checkpoint.checkpoint_id

            lg_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_ns": "",
                }
            }

            # Build LangGraph Checkpoint
            # CRITICAL: LangGraph expects 'id' to be a hex string (UUID format without dashes)
            # Convert our checkpoint_id to a deterministic hex string
            lg_checkpoint_id = hashlib.sha256(checkpoint_id.encode()).hexdigest()[:32]

            lg_checkpoint: LGCheckpoint = {
                "v": 1,
                "id": lg_checkpoint_id,  # Must be hex string for binascii.unhexlify
                "ts": our_checkpoint.ts,
                "channel_values": our_checkpoint.state,
                "channel_versions": {},
                "versions_seen": {},
                "pending_sends": [],
            }

            # Build metadata - MUST include 'step' for LangGraph resume to work
            # LangGraph's pregel loop expects: self.step = self.checkpoint_metadata["step"] + 1
            base_metadata = our_checkpoint.metadata or {}
            metadata: CheckpointMetadata = {
                "step": base_metadata.get("step", 0),  # Required for resume
                **base_metadata,
            }

            # Parent config if we have parent_id
            parent_config = None
            if our_checkpoint.parent_id:
                parent_config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_id": our_checkpoint.parent_id,
                        "checkpoint_ns": "",
                    }
                }

            return CheckpointTuple(
                config=lg_config,
                checkpoint=lg_checkpoint,
                metadata=metadata,
                parent_config=parent_config,
                pending_writes=None,
            )

        async def aget_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
            """Async get checkpoint tuple - required by LangGraph 0.2.x."""
            thread_id = config.get("configurable", {}).get("thread_id")
            checkpoint_id = config.get("configurable", {}).get("checkpoint_id")

            if not thread_id:
                return None

            # If specific checkpoint_id requested, get that one
            if checkpoint_id:
                our_checkpoint = await self.inner.get(checkpoint_id)
            else:
                # Get latest
                our_checkpoint = await self.inner.get_latest(thread_id)

            if not our_checkpoint:
                return None

            return self._make_checkpoint_tuple(our_checkpoint, config)

        def get_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
            """Sync get checkpoint tuple."""
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                # Running in async context - use thread to avoid blocking
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return pool.submit(asyncio.run, self.aget_tuple(config)).result()
            except RuntimeError:
                # No running loop - safe to use asyncio.run
                return asyncio.run(self.aget_tuple(config))

        async def aput(
            self,
            config: Dict[str, Any],
            checkpoint: LGCheckpoint,
            metadata: CheckpointMetadata,
            new_versions: ChannelVersions,
        ) -> Dict[str, Any]:
            """Async put checkpoint - new LangGraph 0.2.x interface."""
            thread_id = config.get("configurable", {}).get("thread_id", "unknown")
            parent_checkpoint_id = config.get("configurable", {}).get("checkpoint_id")

            # Extract state from channel_values
            state = checkpoint.get("channel_values", {})

            run_id = state.get("run_id", thread_id)
            sprint_id = state.get("sprint_id", "unknown")
            node = state.get("phase", "unknown")
            attempt = state.get("attempt", 1)

            checkpoint_id = generate_checkpoint_id(run_id, sprint_id, node, attempt)

            # FIX 2026-01-27: Track step counter for proper resume
            # Get step from incoming metadata, default to 0
            step = metadata.get("step", 0) if metadata else 0
            ckpt_metadata = dict(metadata) if metadata else {}
            ckpt_metadata["step"] = step  # Ensure step is always present

            ckpt = Checkpoint(
                checkpoint_id=checkpoint_id,
                run_id=run_id,
                sprint_id=sprint_id,
                thread_id=thread_id,
                ts=datetime.now(timezone.utc).isoformat(),
                state=state,
                parent_id=parent_checkpoint_id,
                metadata=ckpt_metadata,
            )

            await self.inner.put(ckpt)

            return {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_ns": "",
                }
            }

        def put(
            self,
            config: Dict[str, Any],
            checkpoint: LGCheckpoint,
            metadata: CheckpointMetadata,
            new_versions: ChannelVersions,
        ) -> Dict[str, Any]:
            """Sync put checkpoint."""
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return pool.submit(
                        asyncio.run, self.aput(config, checkpoint, metadata, new_versions)
                    ).result()
            except RuntimeError:
                return asyncio.run(self.aput(config, checkpoint, metadata, new_versions))

        async def alist(
            self,
            config: Optional[Dict[str, Any]],
            *,
            filter: Optional[Dict[str, Any]] = None,
            before: Optional[Dict[str, Any]] = None,
            limit: Optional[int] = None,
        ) -> AsyncIterator[CheckpointTuple]:
            """Async list checkpoints."""
            if config is None:
                return

            thread_id = config.get("configurable", {}).get("thread_id")
            if not thread_id:
                return

            checkpoints = await self.inner.list(thread_id, limit=limit or 10)

            for ckpt in checkpoints:
                yield self._make_checkpoint_tuple(ckpt, config)

        def list(
            self,
            config: Optional[Dict[str, Any]],
            *,
            filter: Optional[Dict[str, Any]] = None,
            before: Optional[Dict[str, Any]] = None,
            limit: Optional[int] = None,
        ) -> Iterator[CheckpointTuple]:
            """Sync list checkpoints."""
            if config is None:
                return

            thread_id = config.get("configurable", {}).get("thread_id")
            if not thread_id:
                return

            import asyncio
            try:
                loop = asyncio.get_running_loop()
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    checkpoints = pool.submit(
                        asyncio.run, self.inner.list(thread_id, limit=limit or 10)
                    ).result()
            except RuntimeError:
                checkpoints = asyncio.run(self.inner.list(thread_id, limit=limit or 10))

            for ckpt in checkpoints:
                yield self._make_checkpoint_tuple(ckpt, config)

        async def aput_writes(
            self,
            config: Dict[str, Any],
            writes: Any,  # Sequence[tuple[str, Any]]
            task_id: str,
            task_path: str = "",
        ) -> None:
            """Async store intermediate writes linked to a checkpoint.

            For now, we store writes as part of the checkpoint metadata.
            This is a simplified implementation - writes are not persisted separately.
            """
            # Get current checkpoint
            thread_id = config.get("configurable", {}).get("thread_id")
            if not thread_id:
                return

            checkpoint = await self.inner.get_latest(thread_id)
            if checkpoint:
                # Add writes to metadata
                metadata = checkpoint.metadata or {}
                if "pending_writes" not in metadata:
                    metadata["pending_writes"] = []
                metadata["pending_writes"].append({
                    "task_id": task_id,
                    "task_path": task_path,
                    "writes": list(writes) if writes else [],
                })
                checkpoint.metadata = metadata
                await self.inner.put(checkpoint)

        def put_writes(
            self,
            config: Dict[str, Any],
            writes: Any,  # Sequence[tuple[str, Any]]
            task_id: str,
            task_path: str = "",
        ) -> None:
            """Sync store intermediate writes."""
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    pool.submit(
                        asyncio.run, self.aput_writes(config, writes, task_id, task_path)
                    ).result()
            except RuntimeError:
                asyncio.run(self.aput_writes(config, writes, task_id, task_path))

    return LangGraphCheckpointerAdapter(checkpointer)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


async def create_checkpointer(
    run_dir: Path,
    redis_client: Optional[Any] = None,
    storage: str = "hybrid",
) -> BaseCheckpointer:
    """Create a checkpointer based on configuration.

    Args:
        run_dir: Directory for file-based checkpoints.
        redis_client: Optional Redis client for Redis/hybrid modes.
        storage: Storage mode ("file", "redis", "hybrid").

    Returns:
        Configured checkpointer instance.
    """
    file_checkpointer = FileCheckpointer(run_dir)

    if storage == "file" or redis_client is None:
        return file_checkpointer

    redis_checkpointer = RedisCheckpointer(redis_client)

    if storage == "redis":
        return redis_checkpointer

    # Default: hybrid
    return HybridCheckpointer(redis_checkpointer, file_checkpointer)


# =============================================================================
# INCREMENTAL CHECKPOINT MIXIN (Agent GAMMA)
# =============================================================================


@dataclass
class DeltaCheckpoint:
    """Incremental checkpoint based on delta from previous state.

    Attributes:
        checkpoint_id: Unique checkpoint identifier
        parent_id: ID of parent checkpoint (None for full checkpoints)
        delta: Dictionary of changed fields only
        removed_keys: List of keys removed since parent
        timestamp: ISO timestamp of checkpoint
        compressed: Whether delta is gzip compressed
        original_size: Size of full state in bytes
        delta_size: Size of delta in bytes

    Invariant INV-GAMMA-005: delta_size < original_size (for non-empty deltas)
    """

    checkpoint_id: str
    parent_id: Optional[str]
    delta: Dict[str, Any]
    removed_keys: List[str]
    timestamp: str
    compressed: bool
    original_size: int
    delta_size: int


class IncrementalCheckpointMixin:
    """Mixin to add incremental checkpoint support.

    This mixin can be added to FileCheckpointer to enable delta-based
    checkpointing, reducing storage and I/O for large states.

    Usage:
        class IncrementalFileCheckpointer(IncrementalCheckpointMixin, FileCheckpointer):
            pass

    Invariants enforced:
        INV-GAMMA-005: Delta checkpoints smaller than full checkpoints
        INV-GAMMA-006: Restore from checkpoint must reconstruct identical state
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize mixin state."""
        super().__init__(*args, **kwargs)
        self._last_state: Optional[Dict[str, Any]] = None
        self._checkpoint_chain: List[str] = []
        self._full_checkpoint_interval: int = 10  # Full checkpoint every N deltas

    def _compute_delta(
        self,
        current_state: Dict[str, Any],
        previous_state: Dict[str, Any],
    ) -> tuple[Dict[str, Any], List[str]]:
        """Compute delta between two states.

        Args:
            current_state: Current state dictionary
            previous_state: Previous state dictionary

        Returns:
            Tuple of (delta_dict, removed_keys)
        """
        delta: Dict[str, Any] = {}
        removed: List[str] = []

        # Find new or changed fields
        for key, value in current_state.items():
            if key not in previous_state:
                delta[key] = value
            elif not self._deep_equals(value, previous_state.get(key)):
                delta[key] = value

        # Find removed fields
        for key in previous_state:
            if key not in current_state:
                removed.append(key)

        return delta, removed

    def _deep_equals(self, a: Any, b: Any) -> bool:
        """Deep comparison of two values.

        Args:
            a: First value
            b: Second value

        Returns:
            True if values are deeply equal
        """
        if type(a) != type(b):
            return False

        if isinstance(a, dict):
            if set(a.keys()) != set(b.keys()):
                return False
            return all(self._deep_equals(a[k], b[k]) for k in a)

        if isinstance(a, (list, tuple)):
            if len(a) != len(b):
                return False
            return all(self._deep_equals(x, y) for x, y in zip(a, b))

        return a == b

    def _apply_delta(
        self,
        base_state: Dict[str, Any],
        delta: Dict[str, Any],
        removed_keys: List[str],
    ) -> Dict[str, Any]:
        """Apply delta to reconstruct state.

        Args:
            base_state: Base state to apply delta to
            delta: Delta dictionary
            removed_keys: Keys to remove

        Returns:
            Reconstructed state (INV-GAMMA-006)
        """
        result = base_state.copy()

        # Apply changes
        result.update(delta)

        # Remove keys
        for key in removed_keys:
            result.pop(key, None)

        return result

    def _get_delta_chain(
        self,
        checkpoint_id: str,
        checkpoint_dir: Path,
    ) -> List[DeltaCheckpoint]:
        """Get chain of deltas from full checkpoint to target.

        Args:
            checkpoint_id: Target checkpoint ID
            checkpoint_dir: Directory containing checkpoints

        Returns:
            List of DeltaCheckpoint in chronological order
        """
        import gzip

        chain: List[DeltaCheckpoint] = []
        current_id: Optional[str] = checkpoint_id

        while current_id:
            # Try compressed first
            ckpt_path = checkpoint_dir / f"{current_id}.json.gz"
            if not ckpt_path.exists():
                ckpt_path = checkpoint_dir / f"{current_id}.json"

            if not ckpt_path.exists():
                logger.error(f"Checkpoint not found: {current_id}")
                break

            # Load checkpoint
            try:
                if ckpt_path.suffix == ".gz":
                    with gzip.open(ckpt_path, "rt", encoding="utf-8") as f:
                        data = json.load(f)
                else:
                    data = json.loads(ckpt_path.read_text())
            except Exception as e:
                logger.error(f"Failed to load checkpoint {current_id}: {e}")
                break

            delta_ckpt = DeltaCheckpoint(
                checkpoint_id=data.get("checkpoint_id", current_id),
                parent_id=data.get("parent_id"),
                delta=data.get("delta", data.get("state", {})),
                removed_keys=data.get("removed_keys", []),
                timestamp=data.get("timestamp", data.get("ts", "")),
                compressed=ckpt_path.suffix == ".gz",
                original_size=data.get("original_size", 0),
                delta_size=data.get("delta_size", 0),
            )

            chain.append(delta_ckpt)

            # If no parent, this is a full checkpoint
            if not delta_ckpt.parent_id:
                break

            current_id = delta_ckpt.parent_id

        # Reverse to chronological order
        return list(reversed(chain))

    def _compress_delta(self, delta: Dict[str, Any]) -> bytes:
        """Compress delta using gzip.

        Args:
            delta: Delta dictionary to compress

        Returns:
            Compressed bytes
        """
        import gzip

        json_bytes = json.dumps(
            delta, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        return gzip.compress(json_bytes)

    def _decompress_delta(self, compressed: bytes) -> Dict[str, Any]:
        """Decompress delta.

        Args:
            compressed: Compressed bytes

        Returns:
            Delta dictionary
        """
        import gzip

        json_bytes = gzip.decompress(compressed)
        return json.loads(json_bytes.decode("utf-8"))

    def _should_full_checkpoint(self) -> bool:
        """Decide if a full checkpoint should be saved.

        Returns:
            True if full checkpoint needed
        """
        return len(self._checkpoint_chain) >= self._full_checkpoint_interval

    async def save_incremental(
        self,
        state: Dict[str, Any],
        checkpoint_dir: Path,
        run_id: str,
    ) -> str:
        """Save incremental checkpoint.

        Args:
            state: Current state to checkpoint
            checkpoint_dir: Directory to save checkpoint
            run_id: Run identifier

        Returns:
            Checkpoint ID of saved checkpoint
        """
        import gzip

        checkpoint_id = self._generate_incremental_checkpoint_id(run_id)
        timestamp = datetime.now(timezone.utc).isoformat()
        original_size = len(json.dumps(state))

        # Decide: full or delta
        if self._last_state is None or self._should_full_checkpoint():
            # Full checkpoint
            data = {
                "checkpoint_id": checkpoint_id,
                "parent_id": None,
                "state": state,
                "removed_keys": [],
                "timestamp": timestamp,
                "type": "full",
                "original_size": original_size,
                "delta_size": original_size,
            }
            self._checkpoint_chain = [checkpoint_id]
            logger.info(
                f"Saving FULL checkpoint: {checkpoint_id} ({original_size} bytes)"
            )
        else:
            # Delta checkpoint
            delta, removed = self._compute_delta(state, self._last_state)
            delta_size = len(json.dumps(delta))

            parent_id = self._checkpoint_chain[-1] if self._checkpoint_chain else None

            data = {
                "checkpoint_id": checkpoint_id,
                "parent_id": parent_id,
                "delta": delta,
                "removed_keys": removed,
                "timestamp": timestamp,
                "type": "delta",
                "original_size": original_size,
                "delta_size": delta_size,
            }
            self._checkpoint_chain.append(checkpoint_id)

            # INV-GAMMA-005: Log savings
            if original_size > 0:
                savings = (original_size - delta_size) / original_size * 100
                logger.info(
                    f"Saving DELTA checkpoint: {checkpoint_id} "
                    f"({delta_size} bytes, {savings:.1f}% savings)"
                )

        # Save compressed
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = checkpoint_dir / f"{checkpoint_id}.json.gz"

        with gzip.open(ckpt_path, "wt", encoding="utf-8") as f:
            json.dump(data, f)

        # Update cache
        self._last_state = state.copy()

        return checkpoint_id

    async def restore_incremental(
        self,
        checkpoint_id: str,
        checkpoint_dir: Path,
    ) -> Optional[Dict[str, Any]]:
        """Restore state from incremental checkpoint.

        Args:
            checkpoint_id: Checkpoint to restore
            checkpoint_dir: Directory containing checkpoints

        Returns:
            Reconstructed state or None if failed (INV-GAMMA-006)
        """
        try:
            # Get delta chain
            chain = self._get_delta_chain(checkpoint_id, checkpoint_dir)

            if not chain:
                logger.error(f"No checkpoint chain found for: {checkpoint_id}")
                return None

            # Reconstruct state by applying deltas
            state: Dict[str, Any] = {}
            for ckpt in chain:
                state = self._apply_delta(state, ckpt.delta, ckpt.removed_keys)

            # Update cache
            self._last_state = state.copy()
            self._checkpoint_chain = [c.checkpoint_id for c in chain]

            logger.info(f"Restored state from {len(chain)} checkpoints")
            return state

        except Exception as e:
            logger.error(f"Failed to restore checkpoint: {e}")
            return None

    def _generate_incremental_checkpoint_id(self, run_id: str) -> str:
        """Generate unique checkpoint ID.

        Args:
            run_id: Run identifier

        Returns:
            Unique checkpoint ID
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        hash_suffix = hashlib.md5(f"{run_id}{timestamp}".encode()).hexdigest()[:8]
        return f"ckpt_incr_{run_id}_{timestamp}_{hash_suffix}"


# =============================================================================
# INCREMENTAL FILE CHECKPOINTER
# =============================================================================


class IncrementalFileCheckpointer(IncrementalCheckpointMixin, FileCheckpointer):
    """FileCheckpointer with incremental checkpoint support.

    Combines the file-based persistence of FileCheckpointer with
    the delta-based efficiency of IncrementalCheckpointMixin.

    Usage:
        checkpointer = IncrementalFileCheckpointer(run_dir)
        await checkpointer.save_incremental(state, checkpoint_dir, run_id)
        state = await checkpointer.restore_incremental(checkpoint_id, checkpoint_dir)
    """

    pass


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "Checkpoint",
    "DeltaCheckpoint",
    "BaseCheckpointer",
    "FileCheckpointer",
    "RedisCheckpointer",
    "HybridCheckpointer",
    "IncrementalCheckpointMixin",
    "IncrementalFileCheckpointer",
    "generate_checkpoint_id",
    "create_checkpointer",
    "create_langgraph_checkpointer",
    # P0-05: LazyContextPack serialization helpers
    "_serialize_context_pack",
    "_deserialize_context_pack",
    "_serialize_state_for_checkpoint",
    "_deserialize_state_from_checkpoint",
]
