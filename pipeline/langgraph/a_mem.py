"""A-MEM: Agent Memory for Learning Persistence.

This module implements persistent learning storage for agents.
Learnings from violations, errors, and successes are stored
and retrieved for future reference.

The A-MEM system provides:
1. Persistent storage of learnings (Redis + local cache)
2. Context-aware retrieval of relevant learnings
3. Learning lifecycle management (creation, retrieval, decay)
4. Integration with guardrail violation handler and reflexion engine

Author: Pipeline Autonomo Team
Version: 1.0.0 (2026-01-16)
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from pathlib import Path

logger = logging.getLogger(__name__)


class LearningType(str, Enum):
    """Types of learnings that can be stored."""
    ERROR = "error"           # Learning from an error
    SUCCESS = "success"       # Learning from a success
    PATTERN = "pattern"       # Identified pattern
    VIOLATION = "violation"   # Guardrail violation
    REFLEXION = "reflexion"   # Reflexion output
    GATE_FAILURE = "gate_failure"  # Gate failure analysis
    REWORK = "rework"         # Rework attempt learning


class LearningSeverity(str, Enum):
    """Severity levels for learnings."""
    CRITICAL = "critical"     # Must always be considered
    HIGH = "high"             # Important learning
    MEDIUM = "medium"         # Standard learning
    LOW = "low"               # Minor learning
    INFO = "info"             # Informational only


@dataclass
class Learning:
    """A learning entry that persists agent knowledge.

    Attributes:
        learning_id: Unique identifier for this learning.
        learning_type: Type of learning (error, success, pattern, etc.).
        severity: How critical this learning is.
        context: Context in which the learning occurred.
        lesson: The actual lesson learned (human-readable).
        action_taken: What action was taken in response.
        outcome: The result of the action.
        confidence: Confidence score (0.0 to 1.0).
        tags: Tags for categorization and retrieval.
        created_at: When this learning was created.
        agent_id: Which agent generated this learning.
        run_id: Which run this learning belongs to.
        sprint_id: Which sprint this learning belongs to.
        related_learnings: IDs of related learnings.
        retrieval_count: How many times this learning was retrieved.
        last_retrieved_at: When this learning was last retrieved.
    """
    learning_id: str
    learning_type: LearningType
    severity: LearningSeverity
    context: Dict[str, Any]
    lesson: str
    action_taken: str
    outcome: str
    confidence: float
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    agent_id: Optional[str] = None
    run_id: Optional[str] = None
    sprint_id: Optional[str] = None
    related_learnings: List[str] = field(default_factory=list)
    retrieval_count: int = 0
    last_retrieved_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert learning to dictionary."""
        data = asdict(self)
        data["learning_type"] = self.learning_type.value
        data["severity"] = self.severity.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Learning":
        """Create learning from dictionary."""
        # Convert string enums back to enum types
        if isinstance(data.get("learning_type"), str):
            data["learning_type"] = LearningType(data["learning_type"])
        if isinstance(data.get("severity"), str):
            data["severity"] = LearningSeverity(data["severity"])
        return cls(**data)

    def matches_context(self, query_context: Dict[str, Any]) -> float:
        """Calculate how well this learning matches a query context.

        Returns a score from 0.0 to 1.0.
        """
        if not query_context:
            return 0.5  # Neutral match

        score = 0.0
        matches = 0
        total_keys = 0

        for key, value in query_context.items():
            total_keys += 1
            if key in self.context:
                if self.context[key] == value:
                    matches += 1
                    score += 1.0
                elif isinstance(value, str) and isinstance(self.context[key], str):
                    # Partial string match
                    if value.lower() in self.context[key].lower():
                        matches += 0.5
                        score += 0.5

        if total_keys == 0:
            return 0.5

        return score / total_keys


class AMemClient:
    """Agent Memory Client for learning persistence.

    Provides CRUD operations for learnings with:
    - Redis for fast distributed access
    - Local cache for performance
    - File-based backup for durability
    """

    def __init__(
        self,
        redis_client=None,
        backup_dir: Optional[str] = None,
    ):
        """Initialize A-MEM client.

        Args:
            redis_client: Optional Redis client for distributed storage.
            backup_dir: Directory for file-based backup.
        """
        self._redis = redis_client
        self._backup_dir = Path(backup_dir or "out/a_mem")
        self._backup_dir.mkdir(parents=True, exist_ok=True)

        # Local cache for fast access
        self._local_cache: Dict[str, Learning] = {}

        # Index by type for fast retrieval
        self._type_index: Dict[LearningType, Set[str]] = {
            lt: set() for lt in LearningType
        }

        # Index by tags
        self._tag_index: Dict[str, Set[str]] = {}

        # Load from backup on init
        self._load_from_backup()

    def _load_from_backup(self) -> None:
        """Load learnings from file backup."""
        backup_file = self._backup_dir / "learnings.jsonl"
        if not backup_file.exists():
            return

        try:
            with open(backup_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        learning = Learning.from_dict(data)
                        self._index_learning(learning)
                        self._local_cache[learning.learning_id] = learning
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        logger.warning(f"Failed to load learning: {e}")

            logger.info(f"A-MEM: Loaded {len(self._local_cache)} learnings from backup")
        except Exception as e:
            logger.error(f"A-MEM: Failed to load from backup: {e}")

    def _index_learning(self, learning: Learning) -> None:
        """Add learning to indexes."""
        # Type index
        self._type_index[learning.learning_type].add(learning.learning_id)

        # Tag index
        for tag in learning.tags:
            if tag not in self._tag_index:
                self._tag_index[tag] = set()
            self._tag_index[tag].add(learning.learning_id)

    def _remove_from_indexes(self, learning: Learning) -> None:
        """Remove learning from indexes."""
        # Type index
        self._type_index[learning.learning_type].discard(learning.learning_id)

        # Tag index
        for tag in learning.tags:
            if tag in self._tag_index:
                self._tag_index[tag].discard(learning.learning_id)

    async def save_learning(self, learning: Learning) -> bool:
        """Save a learning to persistent storage.

        Args:
            learning: The learning to save.

        Returns:
            True if saved successfully.
        """
        try:
            # Save to Redis if available
            if self._redis:
                try:
                    key = f"amem:learning:{learning.learning_id}"
                    await self._redis.set(key, json.dumps(learning.to_dict()))
                    await self._redis.expire(key, 86400 * 30)  # 30 days TTL

                    # Update type index in Redis
                    await self._redis.sadd(
                        f"amem:index:type:{learning.learning_type.value}",
                        learning.learning_id
                    )

                    # Update tag indexes in Redis
                    for tag in learning.tags:
                        await self._redis.sadd(
                            f"amem:index:tag:{tag}",
                            learning.learning_id
                        )
                except Exception as e:
                    logger.warning(f"A-MEM: Redis save failed (using local): {e}")

            # Save to local cache
            self._local_cache[learning.learning_id] = learning
            self._index_learning(learning)

            # Append to backup file
            self._append_to_backup(learning)

            logger.info(f"A-MEM: Learning saved - {learning.learning_id} ({learning.learning_type.value})")
            return True

        except Exception as e:
            logger.error(f"A-MEM: Failed to save learning: {e}")
            return False

    def save_learning_sync(self, learning: Learning) -> bool:
        """Synchronous version of save_learning."""
        try:
            # Save to local cache
            self._local_cache[learning.learning_id] = learning
            self._index_learning(learning)

            # Append to backup file
            self._append_to_backup(learning)

            logger.info(f"A-MEM: Learning saved (sync) - {learning.learning_id} ({learning.learning_type.value})")
            return True

        except Exception as e:
            logger.error(f"A-MEM: Failed to save learning (sync): {e}")
            return False

    def _append_to_backup(self, learning: Learning) -> None:
        """Append learning to backup file."""
        backup_file = self._backup_dir / "learnings.jsonl"
        with open(backup_file, "a") as f:
            f.write(json.dumps(learning.to_dict()) + "\n")

    async def get_learning(self, learning_id: str) -> Optional[Learning]:
        """Get a specific learning by ID.

        Args:
            learning_id: The learning ID.

        Returns:
            The learning if found, None otherwise.
        """
        # Check local cache first
        if learning_id in self._local_cache:
            learning = self._local_cache[learning_id]
            learning.retrieval_count += 1
            learning.last_retrieved_at = datetime.now(timezone.utc).isoformat()
            return learning

        # Try Redis
        if self._redis:
            try:
                key = f"amem:learning:{learning_id}"
                data = await self._redis.get(key)
                if data:
                    learning = Learning.from_dict(json.loads(data))
                    self._local_cache[learning_id] = learning
                    self._index_learning(learning)
                    learning.retrieval_count += 1
                    learning.last_retrieved_at = datetime.now(timezone.utc).isoformat()
                    return learning
            except Exception as e:
                logger.warning(f"A-MEM: Redis get failed: {e}")

        return None

    async def get_relevant_learnings(
        self,
        context: Dict[str, Any],
        learning_type: Optional[LearningType] = None,
        tags: Optional[List[str]] = None,
        min_confidence: float = 0.5,
        min_severity: Optional[LearningSeverity] = None,
        limit: int = 10,
    ) -> List[Learning]:
        """Retrieve learnings relevant to the given context.

        Args:
            context: Context to match against.
            learning_type: Filter by learning type.
            tags: Filter by tags (any match).
            min_confidence: Minimum confidence score.
            min_severity: Minimum severity level.
            limit: Maximum number of learnings to return.

        Returns:
            List of relevant learnings, sorted by relevance.
        """
        candidates: List[Learning] = []

        # Get candidate IDs
        candidate_ids: Set[str] = set()

        if learning_type:
            candidate_ids = self._type_index.get(learning_type, set()).copy()
        else:
            for ids in self._type_index.values():
                candidate_ids.update(ids)

        # Filter by tags if specified
        if tags:
            tag_matches: Set[str] = set()
            for tag in tags:
                tag_matches.update(self._tag_index.get(tag, set()))
            if candidate_ids:
                candidate_ids &= tag_matches
            else:
                candidate_ids = tag_matches

        # Score and filter candidates
        severity_order = [
            LearningSeverity.CRITICAL,
            LearningSeverity.HIGH,
            LearningSeverity.MEDIUM,
            LearningSeverity.LOW,
            LearningSeverity.INFO,
        ]

        for learning_id in candidate_ids:
            learning = self._local_cache.get(learning_id)
            if not learning:
                continue

            # Filter by confidence
            if learning.confidence < min_confidence:
                continue

            # Filter by severity
            if min_severity:
                if severity_order.index(learning.severity) > severity_order.index(min_severity):
                    continue

            # Calculate relevance score
            context_score = learning.matches_context(context)
            relevance_score = (
                context_score * 0.4 +
                learning.confidence * 0.3 +
                (1.0 - severity_order.index(learning.severity) / len(severity_order)) * 0.3
            )

            candidates.append((relevance_score, learning))

        # Sort by relevance and return top N
        candidates.sort(key=lambda x: x[0], reverse=True)

        result = [learning for _, learning in candidates[:limit]]

        # Update retrieval stats
        for learning in result:
            learning.retrieval_count += 1
            learning.last_retrieved_at = datetime.now(timezone.utc).isoformat()

        return result

    def get_relevant_learnings_sync(
        self,
        context: Dict[str, Any],
        learning_type: Optional[LearningType] = None,
        tags: Optional[List[str]] = None,
        min_confidence: float = 0.5,
        limit: int = 10,
    ) -> List[Learning]:
        """Synchronous version of get_relevant_learnings."""
        candidates: List[Learning] = []
        candidate_ids: Set[str] = set()

        if learning_type:
            candidate_ids = self._type_index.get(learning_type, set()).copy()
        else:
            for ids in self._type_index.values():
                candidate_ids.update(ids)

        if tags:
            tag_matches: Set[str] = set()
            for tag in tags:
                tag_matches.update(self._tag_index.get(tag, set()))
            if candidate_ids:
                candidate_ids &= tag_matches
            else:
                candidate_ids = tag_matches

        for learning_id in candidate_ids:
            learning = self._local_cache.get(learning_id)
            if not learning:
                continue
            if learning.confidence < min_confidence:
                continue
            context_score = learning.matches_context(context)
            candidates.append((context_score * learning.confidence, learning))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [learning for _, learning in candidates[:limit]]

    async def get_learnings_by_type(
        self,
        learning_type: LearningType,
        limit: int = 50,
    ) -> List[Learning]:
        """Get all learnings of a specific type.

        Args:
            learning_type: Type of learning to retrieve.
            limit: Maximum number to return.

        Returns:
            List of learnings.
        """
        learning_ids = list(self._type_index.get(learning_type, set()))[:limit]
        return [
            self._local_cache[lid]
            for lid in learning_ids
            if lid in self._local_cache
        ]

    async def get_learnings_by_tags(
        self,
        tags: List[str],
        match_all: bool = False,
        limit: int = 50,
    ) -> List[Learning]:
        """Get learnings by tags.

        Args:
            tags: Tags to search for.
            match_all: If True, learning must have ALL tags.
            limit: Maximum number to return.

        Returns:
            List of learnings.
        """
        if not tags:
            return []

        if match_all:
            # Intersection of all tag sets
            result_ids = self._tag_index.get(tags[0], set()).copy()
            for tag in tags[1:]:
                result_ids &= self._tag_index.get(tag, set())
        else:
            # Union of all tag sets
            result_ids: Set[str] = set()
            for tag in tags:
                result_ids.update(self._tag_index.get(tag, set()))

        learning_ids = list(result_ids)[:limit]
        return [
            self._local_cache[lid]
            for lid in learning_ids
            if lid in self._local_cache
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored learnings."""
        type_counts = {
            lt.value: len(ids) for lt, ids in self._type_index.items()
        }

        return {
            "total_learnings": len(self._local_cache),
            "by_type": type_counts,
            "unique_tags": len(self._tag_index),
            "backup_dir": str(self._backup_dir),
        }


# =============================================================================
# SINGLETON & FACTORY
# =============================================================================

_amem_client: Optional[AMemClient] = None


def get_amem_client() -> AMemClient:
    """Get singleton A-MEM client."""
    global _amem_client
    if _amem_client is None:
        # Try to get Redis client
        redis = None
        try:
            from pipeline.langgraph.stack_injection import get_redis_client
            redis = get_redis_client()
        except Exception as e:
            # FIX NEW-3: Log Redis connection failure instead of silent pass
            logger.warning(f"A-MEM Redis client unavailable: {e} - using local memory only")

        _amem_client = AMemClient(redis_client=redis)

        # Log A-MEM initialization with stats
        stats = _amem_client.get_stats()
        logger.info(
            f"STACK A-MEM: LOADED (self-learning system) - "
            f"{stats['total_learnings']} learnings, {stats['unique_tags']} tags"
        )

    return _amem_client


def reset_amem_client() -> None:
    """Reset singleton (mainly for testing)."""
    global _amem_client
    _amem_client = None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_learning(
    learning_type: LearningType,
    lesson: str,
    context: Dict[str, Any],
    action_taken: str = "",
    outcome: str = "",
    confidence: float = 0.7,
    severity: LearningSeverity = LearningSeverity.MEDIUM,
    tags: Optional[List[str]] = None,
    agent_id: Optional[str] = None,
    run_id: Optional[str] = None,
    sprint_id: Optional[str] = None,
) -> Learning:
    """Create a new learning instance.

    Args:
        learning_type: Type of learning.
        lesson: The lesson learned.
        context: Context in which it occurred.
        action_taken: Action that was taken.
        outcome: Result of the action.
        confidence: Confidence score (0-1).
        severity: Severity level.
        tags: Tags for categorization.
        agent_id: Agent that generated this.
        run_id: Run ID.
        sprint_id: Sprint ID.

    Returns:
        New Learning instance.
    """
    return Learning(
        learning_id=str(uuid.uuid4()),
        learning_type=learning_type,
        severity=severity,
        context=context,
        lesson=lesson,
        action_taken=action_taken,
        outcome=outcome,
        confidence=confidence,
        tags=tags or [],
        agent_id=agent_id,
        run_id=run_id,
        sprint_id=sprint_id,
    )


async def save_violation_learning(
    violation_type: str,
    message: str,
    context: Dict[str, Any],
    action_taken: str,
    outcome: str,
    run_id: Optional[str] = None,
    sprint_id: Optional[str] = None,
) -> bool:
    """Save a learning from a guardrail violation.

    Args:
        violation_type: Type of violation.
        message: Violation message.
        context: Context of the violation.
        action_taken: What was done about it.
        outcome: Result.
        run_id: Run ID.
        sprint_id: Sprint ID.

    Returns:
        True if saved successfully.
    """
    learning = create_learning(
        learning_type=LearningType.VIOLATION,
        lesson=f"Violation ({violation_type}): {message}",
        context={**context, "violation_type": violation_type},
        action_taken=action_taken,
        outcome=outcome,
        confidence=0.9,  # High confidence for violations
        severity=LearningSeverity.HIGH,
        tags=["violation", violation_type],
        run_id=run_id,
        sprint_id=sprint_id,
    )

    return await get_amem_client().save_learning(learning)


async def save_error_learning(
    error_type: str,
    error_message: str,
    context: Dict[str, Any],
    resolution: str,
    success: bool,
    run_id: Optional[str] = None,
    sprint_id: Optional[str] = None,
) -> bool:
    """Save a learning from an error.

    Args:
        error_type: Type of error.
        error_message: Error message.
        context: Context of the error.
        resolution: How it was resolved.
        success: Whether resolution was successful.
        run_id: Run ID.
        sprint_id: Sprint ID.

    Returns:
        True if saved successfully.
    """
    learning = create_learning(
        learning_type=LearningType.ERROR,
        lesson=f"Error ({error_type}): {error_message}",
        context={**context, "error_type": error_type},
        action_taken=resolution,
        outcome="resolved" if success else "unresolved",
        confidence=0.8 if success else 0.5,
        severity=LearningSeverity.HIGH if not success else LearningSeverity.MEDIUM,
        tags=["error", error_type],
        run_id=run_id,
        sprint_id=sprint_id,
    )

    return await get_amem_client().save_learning(learning)


async def save_success_learning(
    success_type: str,
    description: str,
    context: Dict[str, Any],
    what_worked: str,
    run_id: Optional[str] = None,
    sprint_id: Optional[str] = None,
) -> bool:
    """Save a learning from a success.

    Args:
        success_type: Type of success.
        description: Description of what succeeded.
        context: Context of the success.
        what_worked: What made it work.
        run_id: Run ID.
        sprint_id: Sprint ID.

    Returns:
        True if saved successfully.
    """
    learning = create_learning(
        learning_type=LearningType.SUCCESS,
        lesson=f"Success ({success_type}): {description}",
        context={**context, "success_type": success_type},
        action_taken=what_worked,
        outcome="success",
        confidence=0.9,
        severity=LearningSeverity.MEDIUM,
        tags=["success", success_type],
        run_id=run_id,
        sprint_id=sprint_id,
    )

    return await get_amem_client().save_learning(learning)


async def get_relevant_violation_learnings(
    context: Dict[str, Any],
    limit: int = 5,
) -> List[Learning]:
    """Get relevant learnings from past violations.

    Args:
        context: Current context.
        limit: Maximum number to return.

    Returns:
        List of relevant violation learnings.
    """
    return await get_amem_client().get_relevant_learnings(
        context=context,
        learning_type=LearningType.VIOLATION,
        min_confidence=0.5,
        limit=limit,
    )


async def get_relevant_error_learnings(
    context: Dict[str, Any],
    limit: int = 5,
) -> List[Learning]:
    """Get relevant learnings from past errors.

    Args:
        context: Current context.
        limit: Maximum number to return.

    Returns:
        List of relevant error learnings.
    """
    return await get_amem_client().get_relevant_learnings(
        context=context,
        learning_type=LearningType.ERROR,
        min_confidence=0.5,
        limit=limit,
    )
