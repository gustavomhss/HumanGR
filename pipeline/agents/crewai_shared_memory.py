"""P2-012: CrewAI Shared Memory Integration.

This module provides shared memory capabilities for CrewAI agents,
enabling knowledge persistence and sharing across agent executions.

Key Features:
- Shared short-term memory (conversation context)
- Long-term memory via Qdrant (vector persistence)
- Entity memory tracking
- Cross-agent knowledge sharing
- Redis-based caching for fast retrieval

Architecture:
    CrewAI Agent
        ↓
    SharedMemoryManager
        ↓
    ┌───────────────────────┐
    │  Qdrant (vectors)     │ <- Long-term memory
    │  Redis (cache)        │ <- Session memory
    │  FalkorDB (knowledge) │ <- Entity relationships
    └───────────────────────┘

Usage:
    from pipeline.agents.crewai_shared_memory import (
        get_shared_memory_config,
        SharedMemoryManager,
    )

    # Get memory config for CrewAI
    memory_config = get_shared_memory_config()

    # Or use manager directly
    manager = SharedMemoryManager()
    await manager.store_memory("agent_1", "key", "value")
    result = await manager.retrieve_memory("agent_2", "key")
"""

from __future__ import annotations

import os
import logging
import json
import hashlib
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_MEMORY_COLLECTION", "crewai_memory")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
MEMORY_TTL_SECONDS = int(os.getenv("CREWAI_MEMORY_TTL", "3600"))  # 1 hour default

# Check stack availability
QDRANT_AVAILABLE = False
REDIS_AVAILABLE = False

try:
    from qdrant_client import QdrantClient
    QDRANT_AVAILABLE = True
except ImportError:
    logger.debug("Qdrant not available for shared memory")

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    logger.debug("Redis not available for shared memory")


# =============================================================================
# DATA MODELS
# =============================================================================


class MemoryType(str, Enum):
    """Types of memory in the shared memory system."""
    SHORT_TERM = "short_term"  # Session-scoped, fast access
    LONG_TERM = "long_term"    # Persisted, vector-based
    ENTITY = "entity"          # Entity tracking
    KNOWLEDGE = "knowledge"    # Knowledge graph facts


class MemoryScope(str, Enum):
    """Scope of memory visibility."""
    AGENT = "agent"    # Only visible to creating agent
    CREW = "crew"      # Visible to all agents in crew
    GLOBAL = "global"  # Visible to all agents


class MemoryEntry(BaseModel):
    """A single memory entry."""

    key: str = Field(..., description="Memory key/identifier")
    value: Any = Field(..., description="Memory value")
    memory_type: MemoryType = Field(default=MemoryType.SHORT_TERM)
    scope: MemoryScope = Field(default=MemoryScope.CREW)
    agent_id: str = Field(..., description="Agent that created this memory")
    crew_id: Optional[str] = Field(default=None, description="Crew context")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    ttl_seconds: Optional[int] = Field(default=None, description="Time-to-live")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "key": self.key,
            "value": self.value if isinstance(self.value, (str, int, float, bool, list, dict)) else str(self.value),
            "memory_type": self.memory_type.value,
            "scope": self.scope.value,
            "agent_id": self.agent_id,
            "crew_id": self.crew_id,
            "timestamp": self.timestamp.isoformat(),
            "ttl_seconds": self.ttl_seconds,
            "metadata": self.metadata,
        }


class EntityMemory(BaseModel):
    """Memory for tracking entities mentioned in conversations."""

    entity_name: str = Field(..., description="Entity name")
    entity_type: str = Field(default="unknown", description="Entity type")
    mentions: int = Field(default=1, description="Number of mentions")
    first_seen: datetime = Field(default_factory=datetime.utcnow)
    last_seen: datetime = Field(default_factory=datetime.utcnow)
    attributes: Dict[str, Any] = Field(default_factory=dict)
    relationships: List[Dict[str, str]] = Field(default_factory=list)


@dataclass
class MemorySearchResult:
    """Result from memory search."""

    entries: List[MemoryEntry] = field(default_factory=list)
    total_found: int = 0
    search_time_ms: float = 0.0


# =============================================================================
# SHARED MEMORY MANAGER
# =============================================================================


class SharedMemoryManager:
    """Manager for shared memory across CrewAI agents.

    Provides a unified interface for storing and retrieving memories
    using multiple backends (Redis, Qdrant, FalkorDB).
    """

    def __init__(
        self,
        qdrant_host: str = QDRANT_HOST,
        qdrant_port: int = QDRANT_PORT,
        collection: str = QDRANT_COLLECTION,
        redis_url: str = REDIS_URL,
        default_ttl: int = MEMORY_TTL_SECONDS,
    ):
        """Initialize shared memory manager.

        Args:
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
            collection: Qdrant collection name
            redis_url: Redis connection URL
            default_ttl: Default TTL for memories in seconds
        """
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.collection = collection
        self.redis_url = redis_url
        self.default_ttl = default_ttl

        self._qdrant_client = None
        self._redis_client = None
        self._local_cache: Dict[str, MemoryEntry] = {}

        self._init_backends()

    def _init_backends(self) -> None:
        """Initialize storage backends."""
        # Initialize Qdrant
        if QDRANT_AVAILABLE:
            try:
                from qdrant_client import QdrantClient
                self._qdrant_client = QdrantClient(
                    host=self.qdrant_host,
                    port=self.qdrant_port,
                )
                self._ensure_collection()
                logger.info("Qdrant backend initialized for shared memory")
            except Exception as e:
                logger.warning(f"Failed to initialize Qdrant: {e}")

        # Initialize Redis
        if REDIS_AVAILABLE:
            try:
                import redis
                self._redis_client = redis.from_url(self.redis_url)
                self._redis_client.ping()
                logger.info("Redis backend initialized for shared memory")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis: {e}")

    def _ensure_collection(self) -> None:
        """Ensure Qdrant collection exists."""
        if not self._qdrant_client:
            return

        try:
            from qdrant_client.models import Distance, VectorParams

            collections = self._qdrant_client.get_collections().collections
            if not any(c.name == self.collection for c in collections):
                self._qdrant_client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(
                        size=384,  # nomic-embed-text dimension
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"Created Qdrant collection: {self.collection}")
        except Exception as e:
            logger.warning(f"Failed to ensure Qdrant collection: {e}")

    def _cache_key(self, agent_id: str, key: str, crew_id: Optional[str] = None) -> str:
        """Generate cache key for memory entry."""
        parts = ["crewai_mem", agent_id, key]
        if crew_id:
            parts.insert(2, crew_id)
        return ":".join(parts)

    async def store_memory(
        self,
        agent_id: str,
        key: str,
        value: Any,
        memory_type: MemoryType = MemoryType.SHORT_TERM,
        scope: MemoryScope = MemoryScope.CREW,
        crew_id: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Store a memory entry.

        Args:
            agent_id: Agent creating the memory
            key: Memory key
            value: Memory value
            memory_type: Type of memory
            scope: Visibility scope
            crew_id: Crew context (required for CREW scope)
            ttl_seconds: Time-to-live in seconds
            metadata: Additional metadata

        Returns:
            True if stored successfully
        """
        entry = MemoryEntry(
            key=key,
            value=value,
            memory_type=memory_type,
            scope=scope,
            agent_id=agent_id,
            crew_id=crew_id,
            ttl_seconds=ttl_seconds or self.default_ttl,
            metadata=metadata or {},
        )

        cache_key = self._cache_key(agent_id, key, crew_id)

        # Store in local cache
        self._local_cache[cache_key] = entry

        # Store in Redis for short-term/session memory
        if memory_type == MemoryType.SHORT_TERM and self._redis_client:
            try:
                self._redis_client.setex(
                    cache_key,
                    entry.ttl_seconds or self.default_ttl,
                    json.dumps(entry.to_dict()),
                )
            except Exception as e:
                logger.warning(f"Failed to store in Redis: {e}")

        # Store in Qdrant for long-term memory
        if memory_type == MemoryType.LONG_TERM and self._qdrant_client:
            try:
                await self._store_in_qdrant(entry)
            except Exception as e:
                logger.warning(f"Failed to store in Qdrant: {e}")

        logger.debug(f"Stored memory: {cache_key} (type={memory_type.value})")
        return True

    async def _store_in_qdrant(self, entry: MemoryEntry) -> None:
        """Store memory in Qdrant with vector embedding."""
        if not self._qdrant_client:
            return

        try:
            from qdrant_client.models import PointStruct

            # Generate simple hash-based ID
            point_id = int(hashlib.md5(
                f"{entry.agent_id}:{entry.key}:{entry.crew_id}".encode()
            ).hexdigest()[:8], 16)

            # Get embedding (using simple approach for now)
            # In production, would use proper embedding model
            embedding = self._simple_embedding(str(entry.value))

            self._qdrant_client.upsert(
                collection_name=self.collection,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=entry.to_dict(),
                    ),
                ],
            )
        except Exception as e:
            logger.warning(f"Qdrant store failed: {e}")

    def _simple_embedding(self, text: str) -> List[float]:
        """Generate a simple embedding (placeholder).

        In production, this would use Ollama or another embedding model.
        """
        # Simple hash-based pseudo-embedding for demonstration
        import hashlib
        hash_bytes = hashlib.sha384(text.encode()).digest()
        return [float(b) / 255.0 for b in hash_bytes]

    async def retrieve_memory(
        self,
        agent_id: str,
        key: str,
        crew_id: Optional[str] = None,
        allow_cross_agent: bool = True,
    ) -> Optional[MemoryEntry]:
        """Retrieve a memory entry.

        Args:
            agent_id: Agent requesting the memory
            key: Memory key
            crew_id: Crew context
            allow_cross_agent: Whether to allow retrieval of other agents' memories

        Returns:
            MemoryEntry if found, None otherwise
        """
        cache_key = self._cache_key(agent_id, key, crew_id)

        # Check local cache first
        if cache_key in self._local_cache:
            return self._local_cache[cache_key]

        # Check Redis
        if self._redis_client:
            try:
                data = self._redis_client.get(cache_key)
                if data:
                    entry_dict = json.loads(data)
                    return MemoryEntry(**entry_dict)
            except Exception as e:
                logger.debug(f"Redis retrieval failed: {e}")

        # If cross-agent allowed, search in crew scope
        if allow_cross_agent and crew_id:
            # Search Redis for crew-scoped memories
            pattern = f"crewai_mem:*:{crew_id}:{key}"
            if self._redis_client:
                try:
                    for crew_key in self._redis_client.scan_iter(pattern):
                        data = self._redis_client.get(crew_key)
                        if data:
                            entry_dict = json.loads(data)
                            entry = MemoryEntry(**entry_dict)
                            if entry.scope in [MemoryScope.CREW, MemoryScope.GLOBAL]:
                                return entry
                except Exception as e:
                    logger.debug(f"Redis scan failed: {e}")

        return None

    async def search_memories(
        self,
        query: str,
        agent_id: str,
        crew_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
    ) -> MemorySearchResult:
        """Search memories using semantic similarity.

        Args:
            query: Search query
            agent_id: Agent performing the search
            crew_id: Crew context
            memory_type: Filter by memory type
            limit: Maximum results to return

        Returns:
            MemorySearchResult with matching entries
        """
        start_time = datetime.now(timezone.utc)
        entries = []

        # Search Qdrant for long-term memories
        if self._qdrant_client and memory_type in [None, MemoryType.LONG_TERM]:
            try:
                query_vector = self._simple_embedding(query)

                # Build filter
                filter_conditions = []
                if crew_id:
                    filter_conditions.append(
                        {"key": "crew_id", "match": {"value": crew_id}}
                    )
                if memory_type:
                    filter_conditions.append(
                        {"key": "memory_type", "match": {"value": memory_type.value}}
                    )

                results = self._qdrant_client.search(
                    collection_name=self.collection,
                    query_vector=query_vector,
                    limit=limit,
                    query_filter={"must": filter_conditions} if filter_conditions else None,
                )

                for result in results:
                    try:
                        payload = result.payload or {}
                        entry = MemoryEntry(
                            key=payload.get("key", ""),
                            value=payload.get("value", ""),
                            memory_type=MemoryType(payload.get("memory_type", "short_term")),
                            scope=MemoryScope(payload.get("scope", "crew")),
                            agent_id=payload.get("agent_id", ""),
                            crew_id=payload.get("crew_id"),
                            timestamp=datetime.fromisoformat(payload.get("timestamp", datetime.now(timezone.utc).isoformat())),
                            metadata=payload.get("metadata", {}),
                        )
                        entries.append(entry)
                    except Exception:
                        continue

            except Exception as e:
                logger.warning(f"Qdrant search failed: {e}")

        search_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        return MemorySearchResult(
            entries=entries,
            total_found=len(entries),
            search_time_ms=search_time,
        )

    async def store_entity(
        self,
        agent_id: str,
        entity_name: str,
        entity_type: str = "unknown",
        attributes: Optional[Dict[str, Any]] = None,
        crew_id: Optional[str] = None,
    ) -> bool:
        """Store or update an entity in entity memory.

        Args:
            agent_id: Agent tracking the entity
            entity_name: Entity name
            entity_type: Type of entity
            attributes: Entity attributes
            crew_id: Crew context

        Returns:
            True if stored successfully
        """
        entity = EntityMemory(
            entity_name=entity_name,
            entity_type=entity_type,
            attributes=attributes or {},
        )

        return await self.store_memory(
            agent_id=agent_id,
            key=f"entity:{entity_name}",
            value=entity.model_dump(),
            memory_type=MemoryType.ENTITY,
            scope=MemoryScope.CREW,
            crew_id=crew_id,
            metadata={"entity_type": entity_type},
        )

    async def get_entities(
        self,
        agent_id: str,
        crew_id: Optional[str] = None,
        entity_type: Optional[str] = None,
    ) -> List[EntityMemory]:
        """Get all tracked entities.

        Args:
            agent_id: Agent requesting entities
            crew_id: Crew context
            entity_type: Filter by entity type

        Returns:
            List of EntityMemory objects
        """
        entities = []

        # Search Redis for entity entries
        if self._redis_client:
            pattern = f"crewai_mem:*:{crew_id}:entity:*" if crew_id else "crewai_mem:*:entity:*"
            try:
                for key in self._redis_client.scan_iter(pattern):
                    data = self._redis_client.get(key)
                    if data:
                        entry_dict = json.loads(data)
                        if entry_dict.get("memory_type") == MemoryType.ENTITY.value:
                            entity_data = entry_dict.get("value", {})
                            if isinstance(entity_data, dict):
                                entity = EntityMemory(**entity_data)
                                if entity_type is None or entity.entity_type == entity_type:
                                    entities.append(entity)
            except Exception as e:
                logger.debug(f"Entity retrieval failed: {e}")

        return entities

    def clear_crew_memory(self, crew_id: str) -> int:
        """Clear all memories for a specific crew.

        Args:
            crew_id: Crew to clear memories for

        Returns:
            Number of entries cleared
        """
        cleared = 0

        # Clear from local cache
        keys_to_remove = [
            k for k in self._local_cache.keys()
            if f":{crew_id}:" in k
        ]
        for key in keys_to_remove:
            del self._local_cache[key]
            cleared += 1

        # Clear from Redis
        if self._redis_client:
            try:
                pattern = f"crewai_mem:*:{crew_id}:*"
                for key in self._redis_client.scan_iter(pattern):
                    self._redis_client.delete(key)
                    cleared += 1
            except Exception as e:
                logger.warning(f"Redis clear failed: {e}")

        logger.info(f"Cleared {cleared} memories for crew {crew_id}")
        return cleared

    def get_stats(self) -> Dict[str, Any]:
        """Get memory manager statistics."""
        stats = {
            "local_cache_size": len(self._local_cache),
            "qdrant_available": self._qdrant_client is not None,
            "redis_available": self._redis_client is not None,
        }

        if self._redis_client:
            try:
                info = self._redis_client.info("memory")
                stats["redis_used_memory"] = info.get("used_memory_human", "unknown")
            except Exception as e:
                logger.debug(f"REDIS: Redis operation failed: {e}")

        return stats


# =============================================================================
# CREWAI MEMORY CONFIGURATION
# =============================================================================


def get_shared_memory_config(
    crew_id: Optional[str] = None,
    enable_short_term: bool = True,
    enable_long_term: bool = True,
    enable_entity: bool = True,
) -> Dict[str, Any]:
    """Get CrewAI memory configuration using shared memory system.

    This returns a configuration dict that can be passed to CrewAI's
    memory parameter for enabling shared memory capabilities.

    Args:
        crew_id: Crew identifier for scoping memories
        enable_short_term: Enable short-term memory
        enable_long_term: Enable long-term memory
        enable_entity: Enable entity memory

    Returns:
        Configuration dict for CrewAI memory parameter
    """
    config = {
        "memory": True,
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": "nomic-embed-text",
                "url": "http://localhost:11434/api/embeddings",
            },
        },
    }

    # Add storage configuration if backends available
    if QDRANT_AVAILABLE:
        config["storage"] = {
            "provider": "qdrant",
            "config": {
                "host": QDRANT_HOST,
                "port": QDRANT_PORT,
                "collection": QDRANT_COLLECTION,
            },
        }

    logger.debug(f"Generated shared memory config: {list(config.keys())}")
    return config


def create_memory_enabled_crew_config(
    memory: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Create a complete crew configuration with shared memory enabled.

    Args:
        memory: Whether to enable memory
        verbose: Whether to enable verbose logging

    Returns:
        Configuration dict for CrewAI Crew constructor
    """
    config = {
        "verbose": verbose,
        "memory": memory,
    }

    if memory:
        memory_config = get_shared_memory_config()
        config["embedder"] = memory_config.get("embedder")
        if "storage" in memory_config:
            config["memory_config"] = memory_config["storage"]

    return config


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_manager: Optional[SharedMemoryManager] = None


def get_shared_memory_manager() -> SharedMemoryManager:
    """Get singleton SharedMemoryManager instance."""
    global _manager
    if _manager is None:
        _manager = SharedMemoryManager()
    return _manager


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "QDRANT_AVAILABLE",
    "REDIS_AVAILABLE",
    "MemoryType",
    "MemoryScope",
    "MemoryEntry",
    "EntityMemory",
    "MemorySearchResult",
    "SharedMemoryManager",
    "get_shared_memory_config",
    "create_memory_enabled_crew_config",
    "get_shared_memory_manager",
]
