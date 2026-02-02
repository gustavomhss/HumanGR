"""State Reconciliation for Pipeline Resilience.

Reconciles state across multiple sources (Redis, file, checkpoint) when
inconsistencies are detected. Ensures pipeline can recover from partial
failures where some state sources may be out of sync.

Sources Priority (when timestamps equal):
    1. Redis (canonical write order)
    2. Checkpoint (LangGraph checkpoint)
    3. File (run_state.yml mirror)

When timestamps differ, newest wins.

Usage:
    result = await reconcile_state(run_id, sprint_id, run_dir)
    if result.conflicts_found:
        logger.warning("Conflicts resolved: %s", result.conflicts_found)
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class StateSource(str, Enum):
    """Sources of pipeline state truth."""

    REDIS = "redis"
    """Redis is the primary state store (canonical write order)."""

    FILE = "file"
    """run_state.yml file mirror."""

    CHECKPOINT = "checkpoint"
    """LangGraph checkpoint storage."""

    INFERRED = "inferred"
    """State inferred from existing artifacts (recovery mode)."""


class StateNotFoundError(Exception):
    """Raised when state cannot be found in any source."""

    def __init__(self, message: str, run_id: str = "", sprint_id: str = ""):
        super().__init__(message)
        self.run_id = run_id
        self.sprint_id = sprint_id


class ReconciliationError(Exception):
    """Raised when reconciliation fails."""

    def __init__(
        self,
        message: str,
        sources_checked: List[StateSource] = None,
        partial_state: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.sources_checked = sources_checked or []
        self.partial_state = partial_state


@dataclass
class StateSnapshot:
    """Snapshot of state from a single source."""

    source: StateSource
    run_id: str
    sprint_id: str
    phase: str
    status: str
    updated_at: datetime
    raw_data: Dict[str, Any]
    checksum: str

    @property
    def is_valid(self) -> bool:
        """Check if snapshot has required fields."""
        return bool(self.run_id and self.sprint_id and self.phase)

    def conflicts_with(self, other: "StateSnapshot") -> List[str]:
        """Identify conflicts between this snapshot and another.

        Args:
            other: Another snapshot to compare against.

        Returns:
            List of conflict descriptions. Empty if no conflicts.
        """
        conflicts = []

        if self.run_id != other.run_id:
            conflicts.append(f"run_id: {self.run_id} vs {other.run_id}")

        if self.sprint_id != other.sprint_id:
            conflicts.append(f"sprint_id: {self.sprint_id} vs {other.sprint_id}")

        if self.phase != other.phase:
            conflicts.append(f"phase: {self.phase} vs {other.phase}")

        if self.status != other.status:
            conflicts.append(f"status: {self.status} vs {other.status}")

        # Check significant timestamp difference (> 5 minutes)
        time_diff = abs((self.updated_at - other.updated_at).total_seconds())
        if time_diff > 300:
            conflicts.append(f"updated_at differs by {time_diff:.0f}s")

        return conflicts

    @classmethod
    def from_redis(cls, redis_data: Dict[str, Any]) -> "StateSnapshot":
        """Create snapshot from Redis data.

        Args:
            redis_data: Dictionary loaded from Redis.

        Returns:
            StateSnapshot instance.
        """
        raw = redis_data
        checksum = hashlib.sha256(
            json.dumps(raw, sort_keys=True, default=str).encode()
        ).hexdigest()

        updated_at_str = raw.get("updated_at", "")
        if updated_at_str:
            try:
                updated_at = datetime.fromisoformat(
                    updated_at_str.replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                updated_at = datetime.now(timezone.utc)
        else:
            updated_at = datetime.now(timezone.utc)

        return cls(
            source=StateSource.REDIS,
            run_id=raw.get("run_id", ""),
            sprint_id=raw.get("sprint_id", ""),
            phase=raw.get("phase", ""),
            status=raw.get("status", ""),
            updated_at=updated_at,
            raw_data=raw,
            checksum=checksum,
        )

    @classmethod
    def from_file(cls, file_path: Path) -> "StateSnapshot":
        """Create snapshot from YAML file.

        Args:
            file_path: Path to run_state.yml file.

        Returns:
            StateSnapshot instance.
        """
        import yaml

        with open(file_path, "r") as f:
            raw = yaml.safe_load(f) or {}

        checksum = hashlib.sha256(
            json.dumps(raw, sort_keys=True, default=str).encode()
        ).hexdigest()

        updated_at_str = raw.get("updated_at", "")
        if updated_at_str:
            try:
                updated_at = datetime.fromisoformat(
                    updated_at_str.replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                # Use file modification time as fallback
                updated_at = datetime.fromtimestamp(
                    file_path.stat().st_mtime, tz=timezone.utc
                )
        else:
            # Use file modification time
            updated_at = datetime.fromtimestamp(
                file_path.stat().st_mtime, tz=timezone.utc
            )

        return cls(
            source=StateSource.FILE,
            run_id=raw.get("run_id", ""),
            sprint_id=raw.get("sprint_id", ""),
            phase=raw.get("phase", ""),
            status=raw.get("status", ""),
            updated_at=updated_at,
            raw_data=raw,
            checksum=checksum,
        )

    @classmethod
    def from_checkpoint(
        cls, checkpoint_data: Dict[str, Any], checkpoint_id: str
    ) -> "StateSnapshot":
        """Create snapshot from LangGraph checkpoint.

        Args:
            checkpoint_data: Checkpoint state dictionary.
            checkpoint_id: Checkpoint identifier.

        Returns:
            StateSnapshot instance.
        """
        raw = checkpoint_data
        checksum = hashlib.sha256(
            json.dumps(raw, sort_keys=True, default=str).encode()
        ).hexdigest()

        # Extract timestamp from checkpoint metadata or use current time
        checkpoint_ts = raw.get("_checkpoint_ts")
        if checkpoint_ts:
            try:
                if isinstance(checkpoint_ts, (int, float)):
                    updated_at = datetime.fromtimestamp(checkpoint_ts, tz=timezone.utc)
                else:
                    updated_at = datetime.fromisoformat(str(checkpoint_ts))
            except (ValueError, TypeError):
                updated_at = datetime.now(timezone.utc)
        else:
            updated_at = datetime.now(timezone.utc)

        return cls(
            source=StateSource.CHECKPOINT,
            run_id=raw.get("run_id", ""),
            sprint_id=raw.get("sprint_id", ""),
            phase=raw.get("phase", raw.get("current_phase", "")),
            status=raw.get("status", raw.get("pipeline_status", "")),
            updated_at=updated_at,
            raw_data=raw,
            checksum=checksum,
        )


@dataclass
class ReconciliationResult:
    """Result of state reconciliation."""

    success: bool
    chosen_source: StateSource
    conflicts_found: List[str]
    actions_taken: List[str]
    final_state: Dict[str, Any]
    sources_checked: List[StateSource]
    reconciled_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "success": self.success,
            "chosen_source": self.chosen_source.value,
            "conflicts_found": self.conflicts_found,
            "actions_taken": self.actions_taken,
            "sources_checked": [s.value for s in self.sources_checked],
            "reconciled_at": self.reconciled_at.isoformat(),
        }


# =============================================================================
# State Readers
# =============================================================================


async def read_state_from_redis(
    run_id: str,
    sprint_id: str,
) -> Optional[StateSnapshot]:
    """Read state from Redis.

    Args:
        run_id: Run identifier.
        sprint_id: Sprint identifier.

    Returns:
        StateSnapshot if found, None otherwise.
    """
    try:
        from pipeline.redis_client import get_redis_client

        redis = get_redis_client()
        if redis is None:
            logger.debug("STATE_READ_REDIS: Redis client not available")
            return None

        key = f"state:{run_id}:{sprint_id}"

        data = redis.get(key)
        if not data:
            logger.debug("STATE_READ_REDIS: not found key=%s", key)
            return None

        raw = json.loads(data) if isinstance(data, (str, bytes)) else data
        snapshot = StateSnapshot.from_redis(raw)

        logger.debug(
            "STATE_READ_REDIS: key=%s phase=%s updated=%s",
            key,
            snapshot.phase,
            snapshot.updated_at,
        )

        return snapshot

    except ImportError:
        logger.debug("STATE_READ_REDIS: redis_client not available")
        return None
    except Exception as e:
        logger.warning("STATE_READ_REDIS_ERROR: %s", e)
        return None


def read_state_from_file(file_path: Path) -> Optional[StateSnapshot]:
    """Read state from YAML file.

    Args:
        file_path: Path to run_state.yml file.

    Returns:
        StateSnapshot if file exists, None otherwise.
    """
    try:
        if not file_path.exists():
            logger.debug("STATE_READ_FILE: not found path=%s", file_path)
            return None

        snapshot = StateSnapshot.from_file(file_path)

        logger.debug(
            "STATE_READ_FILE: path=%s phase=%s updated=%s",
            file_path,
            snapshot.phase,
            snapshot.updated_at,
        )

        return snapshot

    except Exception as e:
        logger.warning("STATE_READ_FILE_ERROR: path=%s error=%s", file_path, e)
        return None


def read_state_from_checkpoint(
    run_dir: Path,
    checkpoint_id: Optional[str] = None,
) -> Optional[StateSnapshot]:
    """Read state from LangGraph checkpoint.

    Args:
        run_dir: Run directory path.
        checkpoint_id: Optional specific checkpoint ID (latest if None).

    Returns:
        StateSnapshot if checkpoint found, None otherwise.
    """
    try:
        checkpoint_dir = run_dir / "checkpoints"
        if not checkpoint_dir.exists():
            logger.debug("STATE_READ_CHECKPOINT: no checkpoint dir")
            return None

        # Find latest checkpoint file
        checkpoint_files = list(checkpoint_dir.glob("*.json"))
        if not checkpoint_files:
            logger.debug("STATE_READ_CHECKPOINT: no checkpoint files")
            return None

        if checkpoint_id:
            # Find specific checkpoint
            target_file = None
            for f in checkpoint_files:
                if checkpoint_id in f.stem:
                    target_file = f
                    break
            if not target_file:
                logger.debug(
                    "STATE_READ_CHECKPOINT: checkpoint %s not found", checkpoint_id
                )
                return None
        else:
            # Use latest by modification time
            target_file = max(checkpoint_files, key=lambda f: f.stat().st_mtime)

        with open(target_file, "r") as f:
            data = json.load(f)

        snapshot = StateSnapshot.from_checkpoint(data, target_file.stem)

        logger.debug(
            "STATE_READ_CHECKPOINT: file=%s phase=%s updated=%s",
            target_file.name,
            snapshot.phase,
            snapshot.updated_at,
        )

        return snapshot

    except Exception as e:
        logger.warning("STATE_READ_CHECKPOINT_ERROR: %s", e)
        return None


# =============================================================================
# State Writers
# =============================================================================


async def write_state_to_redis(
    data: Dict[str, Any],
    run_id: str,
    sprint_id: str,
) -> bool:
    """Write state to Redis.

    Args:
        data: State data to write.
        run_id: Run identifier.
        sprint_id: Sprint identifier.

    Returns:
        True if successful, False otherwise.
    """
    try:
        from pipeline.redis_client import get_redis_client

        redis = get_redis_client()
        if redis is None:
            logger.warning("STATE_WRITE_REDIS: Redis client not available")
            return False

        key = f"state:{run_id}:{sprint_id}"

        # Add reconciliation timestamp
        data_with_ts = {
            **data,
            "reconciled_at": datetime.now(timezone.utc).isoformat(),
        }

        redis.set(key, json.dumps(data_with_ts, default=str))

        logger.debug("STATE_WRITE_REDIS: key=%s", key)
        return True

    except Exception as e:
        logger.warning("STATE_WRITE_REDIS_ERROR: %s", e)
        return False


async def write_state_to_file(
    data: Dict[str, Any],
    file_path: Path,
) -> bool:
    """Write state to YAML file.

    Uses ProcessSafeWriter for atomic writes with locking.

    Args:
        data: State data to write.
        file_path: Path to write to.

    Returns:
        True if successful, False otherwise.
    """
    try:
        from .file_safety import write_yaml_safe

        # Add reconciliation timestamp
        data_with_ts = {
            **data,
            "reconciled_at": datetime.now(timezone.utc).isoformat(),
        }

        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        write_yaml_safe(file_path, data_with_ts)

        logger.debug("STATE_WRITE_FILE: path=%s", file_path)
        return True

    except Exception as e:
        logger.warning("STATE_WRITE_FILE_ERROR: path=%s error=%s", file_path, e)
        return False


# =============================================================================
# Reconciliation Engine
# =============================================================================


async def reconcile_state(
    run_id: str,
    sprint_id: str,
    run_dir: Path,
    include_checkpoint: bool = True,
) -> ReconciliationResult:
    """Reconcile state across all sources.

    Priority order (when timestamps equal):
    1. Redis (canonical)
    2. Checkpoint
    3. File

    When timestamps differ, newest wins.

    Args:
        run_id: Run identifier.
        sprint_id: Sprint identifier.
        run_dir: Run directory path.
        include_checkpoint: Whether to include checkpoint source.

    Returns:
        ReconciliationResult with final state.

    Raises:
        StateNotFoundError: If no state found in any source.
        ReconciliationError: If reconciliation fails.
    """
    actions: List[str] = []
    conflicts: List[str] = []
    sources_checked: List[StateSource] = []

    logger.info(
        "STATE_RECONCILE_START: run_id=%s sprint_id=%s",
        run_id,
        sprint_id,
    )

    # Read from all sources
    redis_state = await read_state_from_redis(run_id, sprint_id)
    if redis_state:
        sources_checked.append(StateSource.REDIS)

    file_path = run_dir / "state" / "run_state.yml"
    file_state = read_state_from_file(file_path)
    if file_state:
        sources_checked.append(StateSource.FILE)

    checkpoint_state = None
    if include_checkpoint:
        checkpoint_state = read_state_from_checkpoint(run_dir)
        if checkpoint_state:
            sources_checked.append(StateSource.CHECKPOINT)

    # Collect all valid snapshots
    snapshots: List[StateSnapshot] = [
        s for s in [redis_state, file_state, checkpoint_state] if s is not None
    ]

    # No state found anywhere
    if not snapshots:
        raise StateNotFoundError(
            f"No state found for run={run_id} sprint={sprint_id}",
            run_id=run_id,
            sprint_id=sprint_id,
        )

    # Only one source has data
    if len(snapshots) == 1:
        chosen = snapshots[0]
        actions.append(f"Only {chosen.source.value} has state")

        # Sync to other sources
        if chosen.source != StateSource.REDIS:
            success = await write_state_to_redis(chosen.raw_data, run_id, sprint_id)
            if success:
                actions.append("Synced to Redis")

        if chosen.source != StateSource.FILE:
            success = await write_state_to_file(chosen.raw_data, file_path)
            if success:
                actions.append("Synced to file")

        logger.info(
            "STATE_RECONCILE_SINGLE_SOURCE: source=%s actions=%s",
            chosen.source.value,
            actions,
        )

        return ReconciliationResult(
            success=True,
            chosen_source=chosen.source,
            conflicts_found=[],
            actions_taken=actions,
            final_state=chosen.raw_data,
            sources_checked=sources_checked,
        )

    # Multiple sources - check for conflicts
    # Compare each pair
    for i, s1 in enumerate(snapshots):
        for s2 in snapshots[i + 1 :]:
            pair_conflicts = s1.conflicts_with(s2)
            if pair_conflicts:
                conflicts.extend(
                    [f"{s1.source.value} vs {s2.source.value}: {c}" for c in pair_conflicts]
                )

    # If no conflicts (checksums match), use priority order
    if not conflicts:
        # Priority: Redis > Checkpoint > File
        chosen = redis_state or checkpoint_state or file_state
        if chosen is None:
            # This shouldn't happen, but handle gracefully
            chosen = snapshots[0]

        actions.append("States consistent across sources")

        logger.info(
            "STATE_RECONCILE_CONSISTENT: chosen=%s sources=%d",
            chosen.source.value,
            len(sources_checked),
        )

        return ReconciliationResult(
            success=True,
            chosen_source=chosen.source,
            conflicts_found=[],
            actions_taken=actions,
            final_state=chosen.raw_data,
            sources_checked=sources_checked,
        )

    # Conflicts exist - use newest
    logger.warning(
        "STATE_CONFLICT: run=%s sprint=%s conflicts=%s",
        run_id,
        sprint_id,
        conflicts,
    )

    # Sort by updated_at descending
    snapshots_sorted = sorted(snapshots, key=lambda s: s.updated_at, reverse=True)
    chosen = snapshots_sorted[0]

    actions.append(f"Chose {chosen.source.value} (newest: {chosen.updated_at})")

    # Sync chosen state to other sources
    if chosen.source != StateSource.REDIS:
        success = await write_state_to_redis(chosen.raw_data, run_id, sprint_id)
        if success:
            actions.append("Updated Redis to match")

    if chosen.source != StateSource.FILE:
        success = await write_state_to_file(chosen.raw_data, file_path)
        if success:
            actions.append("Updated file to match")

    logger.info(
        "STATE_RECONCILE_RESOLVED: chosen=%s conflicts=%d actions=%s",
        chosen.source.value,
        len(conflicts),
        actions,
    )

    return ReconciliationResult(
        success=True,
        chosen_source=chosen.source,
        conflicts_found=conflicts,
        actions_taken=actions,
        final_state=chosen.raw_data,
        sources_checked=sources_checked,
    )


async def quick_reconcile(
    run_id: str,
    sprint_id: str,
    run_dir: Path,
) -> Dict[str, Any]:
    """Quick reconciliation returning just the final state.

    Convenience wrapper around reconcile_state.

    Args:
        run_id: Run identifier.
        sprint_id: Sprint identifier.
        run_dir: Run directory path.

    Returns:
        Final reconciled state dictionary.

    Raises:
        StateNotFoundError: If no state found.
    """
    result = await reconcile_state(run_id, sprint_id, run_dir)
    return result.final_state


__all__ = [
    "StateSource",
    "StateSnapshot",
    "ReconciliationResult",
    "StateNotFoundError",
    "ReconciliationError",
    "read_state_from_redis",
    "read_state_from_file",
    "read_state_from_checkpoint",
    "write_state_to_redis",
    "write_state_to_file",
    "reconcile_state",
    "quick_reconcile",
]
