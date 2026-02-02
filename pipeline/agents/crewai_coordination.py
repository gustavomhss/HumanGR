"""CrewAI Multi-Agent Coordination Module for Pipeline V2.

This module provides enhanced CrewAI integration with multi-agent coordination,
task delegation, and shared memory improvements.

Key Features:
- Multi-agent coordination with role-based delegation
- Task delegation with priority and dependency management
- Shared memory improvements with cross-agent knowledge
- Observability integration with Langfuse callbacks
- Graceful degradation when dependencies unavailable

Architecture:
    CrewCoordinator
        |
        ├─> TaskDelegator (priority-based delegation)
        │       ↓
        │   AgentPool (role-based agent selection)
        │       ↓
        │   TaskExecution (with callbacks)
        │
        └─> SharedKnowledgeBase (cross-agent memory)
                ↓
            ┌───────────────────────┐
            │  Qdrant (vectors)     │
            │  Redis (coordination) │
            │  FalkorDB (relations) │
            └───────────────────────┘

Usage:
    from pipeline.agents.crewai_coordination import (
        get_crew_coordinator,
        create_coordinated_crew,
        delegate_task,
    )

    # Create a coordinated crew
    coordinator = get_crew_coordinator()
    crew = await coordinator.create_crew(
        name="verification_crew",
        agents=["researcher", "analyst", "writer"],
    )

    # Delegate a task
    result = await coordinator.delegate_task(
        task="Verify the claim: The Earth is round",
        agent_role="researcher",
        priority=1,
    )

Author: Pipeline Autonomo Team
Version: 1.0.0 (2026-01-16)
"""

from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional, TypedDict
from datetime import datetime, timezone
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

MAX_CONCURRENT_AGENTS = int(os.getenv("CREWAI_MAX_CONCURRENT", "5"))
TASK_TIMEOUT_SECONDS = float(os.getenv("CREWAI_TASK_TIMEOUT", "300"))
COORDINATION_ENABLED = os.getenv("CREWAI_COORDINATION_ENABLED", "true").lower() == "true"
DEFAULT_MEMORY_ENABLED = os.getenv("CREWAI_MEMORY_ENABLED", "true").lower() == "true"

# Check dependencies
CREWAI_AVAILABLE = False
LANGFUSE_AVAILABLE = False
REDIS_AVAILABLE = False

try:
    from crewai import Agent, Task, Crew, Process
    CREWAI_AVAILABLE = True
except ImportError:
    logger.debug("CrewAI not available")

try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    logger.debug("Langfuse not available for callbacks")

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    logger.debug("Redis not available for coordination")

COORDINATION_AVAILABLE = CREWAI_AVAILABLE

# =============================================================================
# QUIETSTAR GUARDRAILS (2026-01-20)
# =============================================================================
# QuietStar/Reflexion guardrails are integrated at the LLM call layer
# (claude_cli_llm.py and crewai_client.py). This module delegates to those
# layers, so guardrails are applied automatically when tasks are executed.
# =============================================================================

try:
    from pipeline.security.quietstar_reflexion import (
        QuietStarReflexionGuardrail,
        SecurityBlockedError as QuietStarBlockedError,
        QUIETSTAR_AVAILABLE as QUIETSTAR_REFLEXION_AVAILABLE,
    )
except ImportError:
    QUIETSTAR_REFLEXION_AVAILABLE = False

    class QuietStarBlockedError(Exception):
        """Placeholder when QuietStar not available."""
        pass


# =============================================================================
# ENUMS
# =============================================================================


class AgentRole(str, Enum):
    """Predefined agent roles."""
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    WRITER = "writer"
    REVIEWER = "reviewer"
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"


class TaskPriority(str, Enum):
    """Task priority levels."""
    CRITICAL = "critical"  # Must be done immediately
    HIGH = "high"          # High priority
    MEDIUM = "medium"      # Normal priority
    LOW = "low"            # Background task


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class DelegationStrategy(str, Enum):
    """Strategy for task delegation."""
    ROLE_BASED = "role_based"        # Delegate based on agent role
    CAPABILITY_BASED = "capability"   # Delegate based on capabilities
    LOAD_BALANCED = "load_balanced"  # Distribute evenly
    PRIORITY_BASED = "priority"      # Highest priority agent first


# =============================================================================
# RESULT TYPES
# =============================================================================


class AgentCapabilities(TypedDict):
    """Capabilities of an agent."""
    role: str
    skills: List[str]
    tools: List[str]
    max_concurrent_tasks: int
    specializations: List[str]


class TaskDefinition(TypedDict):
    """Definition of a task to delegate."""
    task_id: str
    description: str
    priority: str
    required_role: Optional[str]
    required_capabilities: List[str]
    dependencies: List[str]
    context: Dict[str, Any]
    timeout_seconds: float
    created_at: str


class DelegationResult(TypedDict):
    """Result of task delegation."""
    task_id: str
    assigned_agent: Optional[str]
    delegation_strategy: str
    assignment_reason: str
    estimated_duration_ms: float
    success: bool
    error: Optional[str]


class TaskExecutionResult(TypedDict):
    """Result of task execution."""
    task_id: str
    agent_id: str
    status: str
    output: Any
    artifacts: List[Dict[str, Any]]
    execution_time_ms: float
    tokens_used: int
    memory_updates: List[Dict[str, Any]]
    success: bool
    error: Optional[str]


class CoordinationMetrics(TypedDict):
    """Metrics for crew coordination."""
    total_tasks_delegated: int
    tasks_completed: int
    tasks_failed: int
    average_execution_time_ms: float
    agent_utilization: Dict[str, float]
    memory_entries: int
    active_agents: int


class CrewResult(TypedDict):
    """Result of crew execution."""
    crew_id: str
    tasks_completed: int
    total_execution_time_ms: float
    agent_outputs: Dict[str, Any]
    shared_knowledge: Dict[str, Any]
    success: bool
    error: Optional[str]


# =============================================================================
# TASK DELEGATOR
# =============================================================================


class TaskDelegator:
    """Manages task delegation to agents.

    Implements multiple delegation strategies and handles
    task dependencies and priorities.
    """

    def __init__(
        self,
        default_strategy: DelegationStrategy = DelegationStrategy.ROLE_BASED,
    ):
        """Initialize task delegator.

        Args:
            default_strategy: Default delegation strategy
        """
        self.default_strategy = default_strategy
        self._pending_tasks: Dict[str, TaskDefinition] = {}
        self._task_assignments: Dict[str, str] = {}  # task_id -> agent_id
        self._agent_loads: Dict[str, int] = {}  # agent_id -> current task count

    def create_task(
        self,
        description: str,
        priority: TaskPriority = TaskPriority.MEDIUM,
        required_role: Optional[str] = None,
        required_capabilities: Optional[List[str]] = None,
        dependencies: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        timeout_seconds: float = TASK_TIMEOUT_SECONDS,
    ) -> TaskDefinition:
        """Create a new task definition.

        Args:
            description: Task description
            priority: Task priority
            required_role: Role required to execute
            required_capabilities: Capabilities required
            dependencies: IDs of tasks that must complete first
            context: Additional context
            timeout_seconds: Task timeout

        Returns:
            TaskDefinition for the new task
        """
        task_id = str(uuid.uuid4())[:8]

        task = TaskDefinition(
            task_id=task_id,
            description=description,
            priority=priority.value,
            required_role=required_role,
            required_capabilities=required_capabilities or [],
            dependencies=dependencies or [],
            context=context or {},
            timeout_seconds=timeout_seconds,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        self._pending_tasks[task_id] = task
        return task

    async def delegate(
        self,
        task: TaskDefinition,
        available_agents: Dict[str, AgentCapabilities],
        strategy: Optional[DelegationStrategy] = None,
    ) -> DelegationResult:
        """Delegate a task to an appropriate agent.

        2026-01-20: QuietStar guardrails are applied at the execution layer
        (claude_cli_llm.py and crewai_client.py). Task delegation itself
        doesn't make LLM calls, but we validate the task description here.

        Args:
            task: Task to delegate
            available_agents: Available agents with capabilities
            strategy: Delegation strategy (uses default if None)

        Returns:
            DelegationResult with assignment details

        Raises:
            QuietStarBlockedError: If task description is blocked by guardrails.
        """
        import time
        start_time = time.time()

        strategy = strategy or self.default_strategy

        # 2026-01-20: QuietStar pre-check for task description
        # This validates the task BEFORE delegation to catch malicious tasks early
        if QUIETSTAR_REFLEXION_AVAILABLE:
            try:
                from pipeline.security.quietstar_reflexion import get_guardrail

                guardrail = get_guardrail()
                if guardrail:
                    description = task.get("description", "")
                    if description:
                        result = await guardrail.safety_thinking.analyze(description[:2000])
                        if result.risk_score > 0.7:
                            logger.warning(
                                f"QUIETSTAR DELEGATION BLOCKED: task={task['task_id']}, "
                                f"risk={result.risk_score:.2f}"
                            )
                            return DelegationResult(
                                task_id=task["task_id"],
                                assigned_agent=None,
                                delegation_strategy=strategy.value if strategy else "unknown",
                                assignment_reason=f"BLOCKED by QuietStar: {result.reasoning[:100]}",
                                estimated_duration_ms=0.0,
                                success=False,
                                error=f"Task blocked by security guardrails: {result.reasoning[:200]}",
                            )
            except Exception as e:
                logger.debug(f"QuietStar delegation check skipped: {e}")

        # Check dependencies
        if not self._check_dependencies(task):
            return DelegationResult(
                task_id=task["task_id"],
                assigned_agent=None,
                delegation_strategy=strategy.value,
                assignment_reason="Dependencies not met",
                estimated_duration_ms=0.0,
                success=False,
                error="Task dependencies not completed",
            )

        # Find suitable agent
        selected_agent = None
        assignment_reason = ""

        if strategy == DelegationStrategy.ROLE_BASED:
            selected_agent, assignment_reason = self._delegate_by_role(
                task, available_agents
            )
        elif strategy == DelegationStrategy.CAPABILITY_BASED:
            selected_agent, assignment_reason = self._delegate_by_capability(
                task, available_agents
            )
        elif strategy == DelegationStrategy.LOAD_BALANCED:
            selected_agent, assignment_reason = self._delegate_load_balanced(
                task, available_agents
            )
        elif strategy == DelegationStrategy.PRIORITY_BASED:
            selected_agent, assignment_reason = self._delegate_by_priority(
                task, available_agents
            )

        elapsed_ms = (time.time() - start_time) * 1000

        if selected_agent:
            self._task_assignments[task["task_id"]] = selected_agent
            self._agent_loads[selected_agent] = self._agent_loads.get(selected_agent, 0) + 1

        return DelegationResult(
            task_id=task["task_id"],
            assigned_agent=selected_agent,
            delegation_strategy=strategy.value,
            assignment_reason=assignment_reason,
            estimated_duration_ms=elapsed_ms,
            success=selected_agent is not None,
            error=None if selected_agent else "No suitable agent found",
        )

    def _check_dependencies(self, task: TaskDefinition) -> bool:
        """Check if task dependencies are met."""
        for dep_id in task.get("dependencies", []):
            if dep_id in self._pending_tasks:
                return False  # Dependency still pending
        return True

    def _delegate_by_role(
        self,
        task: TaskDefinition,
        available_agents: Dict[str, AgentCapabilities],
    ) -> tuple[Optional[str], str]:
        """Delegate based on required role."""
        required_role = task.get("required_role")

        for agent_id, caps in available_agents.items():
            if caps["role"] == required_role:
                return agent_id, f"Role match: {required_role}"

        # Fallback to any available agent
        if available_agents:
            agent_id = list(available_agents.keys())[0]
            return agent_id, "Fallback: no role match"

        return None, "No agents available"

    def _delegate_by_capability(
        self,
        task: TaskDefinition,
        available_agents: Dict[str, AgentCapabilities],
    ) -> tuple[Optional[str], str]:
        """Delegate based on capabilities match."""
        required_caps = set(task.get("required_capabilities", []))

        best_match = None
        best_score = 0

        for agent_id, caps in available_agents.items():
            agent_caps = set(caps["skills"] + caps.get("specializations", []))
            match_score = len(required_caps & agent_caps)

            if match_score > best_score:
                best_score = match_score
                best_match = agent_id

        if best_match:
            return best_match, f"Capability match score: {best_score}"

        return None, "No capability match"

    def _delegate_load_balanced(
        self,
        task: TaskDefinition,
        available_agents: Dict[str, AgentCapabilities],
    ) -> tuple[Optional[str], str]:
        """Delegate to agent with lowest load."""
        if not available_agents:
            return None, "No agents available"

        min_load = float("inf")
        selected = None

        for agent_id in available_agents:
            load = self._agent_loads.get(agent_id, 0)
            if load < min_load:
                min_load = load
                selected = agent_id

        return selected, f"Load balanced: current load {min_load}"

    def _delegate_by_priority(
        self,
        task: TaskDefinition,
        available_agents: Dict[str, AgentCapabilities],
    ) -> tuple[Optional[str], str]:
        """Delegate high priority tasks to most capable agents."""
        priority = task.get("priority", "medium")

        if priority in ["critical", "high"]:
            # Find agent with most capabilities
            best_agent = None
            best_score = 0

            for agent_id, caps in available_agents.items():
                score = len(caps["skills"]) + len(caps.get("tools", []))
                if score > best_score:
                    best_score = score
                    best_agent = agent_id

            if best_agent:
                return best_agent, f"Priority match: {priority}"

        # Use load balanced for normal priority
        return self._delegate_load_balanced(task, available_agents)

    def mark_task_complete(self, task_id: str) -> None:
        """Mark a task as completed."""
        if task_id in self._pending_tasks:
            del self._pending_tasks[task_id]

        if task_id in self._task_assignments:
            agent_id = self._task_assignments[task_id]
            if agent_id in self._agent_loads:
                self._agent_loads[agent_id] = max(0, self._agent_loads[agent_id] - 1)
            del self._task_assignments[task_id]


# =============================================================================
# SHARED KNOWLEDGE BASE
# =============================================================================


class SharedKnowledgeBase:
    """Manages shared knowledge across agents.

    Provides a unified interface for agents to share and retrieve
    knowledge using multiple backends.
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
    ):
        """Initialize shared knowledge base.

        Args:
            redis_url: Redis URL for coordination
        """
        self.redis_url = redis_url
        self._local_knowledge: Dict[str, Dict[str, Any]] = {}
        self._redis_client = None

        if REDIS_AVAILABLE and redis_url:
            try:
                self._redis_client = redis.from_url(redis_url)
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")

    async def store(
        self,
        key: str,
        value: Any,
        scope: str = "crew",
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Store knowledge entry.

        Args:
            key: Knowledge key
            value: Knowledge value
            scope: Scope of knowledge ("agent", "crew", "global")
            ttl_seconds: Optional TTL
            metadata: Additional metadata

        Returns:
            True if stored successfully
        """
        entry = {
            "value": value,
            "scope": scope,
            "metadata": metadata or {},
            "stored_at": datetime.now(timezone.utc).isoformat(),
        }

        # Store locally
        self._local_knowledge[key] = entry

        # Store in Redis if available
        if self._redis_client:
            try:
                import json
                redis_key = f"knowledge:{scope}:{key}"
                self._redis_client.setex(
                    redis_key,
                    ttl_seconds or 3600,
                    json.dumps(entry),
                )
            except Exception as e:
                logger.warning(f"Failed to store in Redis: {e}")

        return True

    async def retrieve(
        self,
        key: str,
        scope: str = "crew",
    ) -> Optional[Any]:
        """Retrieve knowledge entry.

        Args:
            key: Knowledge key
            scope: Scope of knowledge

        Returns:
            Knowledge value or None
        """
        # Check local first
        if key in self._local_knowledge:
            return self._local_knowledge[key].get("value")

        # Check Redis
        if self._redis_client:
            try:
                import json
                redis_key = f"knowledge:{scope}:{key}"
                data = self._redis_client.get(redis_key)
                if data:
                    entry = json.loads(data)
                    self._local_knowledge[key] = entry
                    return entry.get("value")
            except Exception as e:
                logger.warning(f"Failed to retrieve from Redis: {e}")

        return None

    async def search(
        self,
        query: str,
        scope: str = "crew",
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search knowledge base.

        Args:
            query: Search query
            scope: Scope to search in
            limit: Maximum results

        Returns:
            List of matching knowledge entries
        """
        results = []
        query_lower = query.lower()

        for key, entry in self._local_knowledge.items():
            if entry.get("scope") == scope:
                # Simple keyword matching
                value_str = str(entry.get("value", "")).lower()
                if query_lower in key.lower() or query_lower in value_str:
                    results.append({
                        "key": key,
                        "value": entry.get("value"),
                        "metadata": entry.get("metadata", {}),
                    })

                if len(results) >= limit:
                    break

        return results

    async def share_with_agents(
        self,
        knowledge_keys: List[str],
        agent_ids: List[str],
    ) -> int:
        """Share specific knowledge with agents.

        Args:
            knowledge_keys: Keys to share
            agent_ids: Agent IDs to share with

        Returns:
            Number of entries shared
        """
        shared = 0

        for key in knowledge_keys:
            if key in self._local_knowledge:
                entry = self._local_knowledge[key]
                for agent_id in agent_ids:
                    # Store with agent scope
                    agent_key = f"{agent_id}:{key}"
                    self._local_knowledge[agent_key] = {
                        **entry,
                        "scope": "agent",
                        "shared_to": agent_id,
                    }
                    shared += 1

        return shared


# =============================================================================
# CREW COORDINATOR
# =============================================================================


class CrewCoordinator:
    """Main coordinator for multi-agent crews.

    Manages agent pools, task delegation, shared knowledge,
    and observability integration.
    """

    def __init__(
        self,
        max_concurrent: int = MAX_CONCURRENT_AGENTS,
        memory_enabled: bool = DEFAULT_MEMORY_ENABLED,
        delegation_strategy: DelegationStrategy = DelegationStrategy.ROLE_BASED,
    ):
        """Initialize crew coordinator.

        Args:
            max_concurrent: Maximum concurrent agents
            memory_enabled: Whether to enable shared memory
            delegation_strategy: Default delegation strategy
        """
        self.max_concurrent = max_concurrent
        self.memory_enabled = memory_enabled

        self.task_delegator = TaskDelegator(delegation_strategy)
        self.knowledge_base = SharedKnowledgeBase()

        self._agent_pool: Dict[str, AgentCapabilities] = {}
        self._active_crews: Dict[str, Dict[str, Any]] = {}
        self._execution_history: List[TaskExecutionResult] = []

        # Langfuse client for observability
        self._langfuse = None
        if LANGFUSE_AVAILABLE:
            try:
                self._langfuse = Langfuse()
            except Exception as e:
                logger.warning(f"Failed to initialize Langfuse: {e}")

    def register_agent(
        self,
        agent_id: str,
        role: AgentRole,
        skills: Optional[List[str]] = None,
        tools: Optional[List[str]] = None,
        specializations: Optional[List[str]] = None,
    ) -> None:
        """Register an agent in the pool.

        Args:
            agent_id: Unique agent identifier
            role: Agent role
            skills: Agent skills
            tools: Available tools
            specializations: Agent specializations
        """
        self._agent_pool[agent_id] = AgentCapabilities(
            role=role.value,
            skills=skills or [],
            tools=tools or [],
            max_concurrent_tasks=3,
            specializations=specializations or [],
        )

    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from the pool.

        Args:
            agent_id: Agent identifier

        Returns:
            True if agent was found and removed
        """
        if agent_id in self._agent_pool:
            del self._agent_pool[agent_id]
            return True
        return False

    async def create_crew(
        self,
        name: str,
        agent_roles: List[AgentRole],
        process_type: str = "sequential",
        memory_enabled: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Create a new coordinated crew.

        Args:
            name: Crew name
            agent_roles: Roles needed for the crew
            process_type: "sequential" or "hierarchical"
            memory_enabled: Whether to enable memory (uses default if None)

        Returns:
            Crew configuration dictionary
        """
        crew_id = str(uuid.uuid4())[:8]

        # Find agents for each role
        assigned_agents = {}
        for role in agent_roles:
            for agent_id, caps in self._agent_pool.items():
                if caps["role"] == role.value and agent_id not in assigned_agents.values():
                    assigned_agents[role.value] = agent_id
                    break

        crew_config = {
            "crew_id": crew_id,
            "name": name,
            "agents": assigned_agents,
            "process_type": process_type,
            "memory_enabled": memory_enabled if memory_enabled is not None else self.memory_enabled,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "ready",
        }

        self._active_crews[crew_id] = crew_config
        return crew_config

    async def delegate_task(
        self,
        task_description: str,
        crew_id: Optional[str] = None,
        agent_role: Optional[AgentRole] = None,
        priority: TaskPriority = TaskPriority.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        strategy: Optional[DelegationStrategy] = None,
    ) -> DelegationResult:
        """Delegate a task to an agent.

        Args:
            task_description: Description of the task
            crew_id: Optional crew to delegate within
            agent_role: Optional specific role to delegate to
            priority: Task priority
            context: Additional context
            strategy: Delegation strategy

        Returns:
            DelegationResult with assignment details
        """
        # Create task definition
        task = self.task_delegator.create_task(
            description=task_description,
            priority=priority,
            required_role=agent_role.value if agent_role else None,
            context=context,
        )

        # Get available agents
        if crew_id and crew_id in self._active_crews:
            crew = self._active_crews[crew_id]
            available = {
                agent_id: self._agent_pool[agent_id]
                for agent_id in crew["agents"].values()
                if agent_id in self._agent_pool
            }
        else:
            available = self._agent_pool

        # Delegate
        result = await self.task_delegator.delegate(task, available, strategy)

        # Log to Langfuse if available
        if self._langfuse and result["success"]:
            try:
                self._langfuse.trace(
                    name="task_delegation",
                    metadata={
                        "task_id": task["task_id"],
                        "agent": result["assigned_agent"],
                        "strategy": result["delegation_strategy"],
                    },
                )
            except Exception as e:
                logger.debug(f"Langfuse logging failed: {e}")

        return result

    async def execute_task(
        self,
        task_id: str,
        agent_id: str,
        task_description: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> TaskExecutionResult:
        """Execute a task with an agent.

        Args:
            task_id: Task identifier
            agent_id: Agent to execute with
            task_description: Task description
            context: Execution context

        Returns:
            TaskExecutionResult with execution details
        """
        import time
        start_time = time.time()

        if agent_id not in self._agent_pool:
            return TaskExecutionResult(
                task_id=task_id,
                agent_id=agent_id,
                status=TaskStatus.FAILED.value,
                output=None,
                artifacts=[],
                execution_time_ms=0.0,
                tokens_used=0,
                memory_updates=[],
                success=False,
                error="Agent not found in pool",
            )

        try:
            # In a real implementation, this would create and run a CrewAI task
            # For now, we simulate execution

            # Retrieve relevant knowledge for context
            knowledge = await self.knowledge_base.search(
                query=task_description[:50],
                scope="crew",
                limit=5,
            )

            # Simulate task execution
            output = f"Task {task_id} completed by {agent_id}"

            # Update shared knowledge
            memory_updates = []
            if self.memory_enabled:
                await self.knowledge_base.store(
                    key=f"task_result:{task_id}",
                    value={
                        "output": output,
                        "agent": agent_id,
                        "context": context,
                    },
                    scope="crew",
                )
                memory_updates.append({
                    "key": f"task_result:{task_id}",
                    "action": "stored",
                })

            # Mark task complete
            self.task_delegator.mark_task_complete(task_id)

            elapsed_ms = (time.time() - start_time) * 1000

            result = TaskExecutionResult(
                task_id=task_id,
                agent_id=agent_id,
                status=TaskStatus.COMPLETED.value,
                output=output,
                artifacts=[],
                execution_time_ms=elapsed_ms,
                tokens_used=0,  # Would track actual token usage
                memory_updates=memory_updates,
                success=True,
                error=None,
            )

            self._execution_history.append(result)
            return result

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            return TaskExecutionResult(
                task_id=task_id,
                agent_id=agent_id,
                status=TaskStatus.FAILED.value,
                output=None,
                artifacts=[],
                execution_time_ms=elapsed_ms,
                tokens_used=0,
                memory_updates=[],
                success=False,
                error=str(e),
            )

    async def run_crew(
        self,
        crew_id: str,
        tasks: List[str],
        context: Optional[Dict[str, Any]] = None,
    ) -> CrewResult:
        """Run a full crew execution.

        Args:
            crew_id: Crew identifier
            tasks: List of task descriptions
            context: Shared context

        Returns:
            CrewResult with execution details
        """
        import time
        start_time = time.time()

        if crew_id not in self._active_crews:
            return CrewResult(
                crew_id=crew_id,
                tasks_completed=0,
                total_execution_time_ms=0.0,
                agent_outputs={},
                shared_knowledge={},
                success=False,
                error="Crew not found",
            )

        crew = self._active_crews[crew_id]
        agent_outputs = {}
        completed = 0

        # Execute tasks sequentially or in parallel based on process type
        for task_desc in tasks:
            # Delegate task
            delegation = await self.delegate_task(
                task_description=task_desc,
                crew_id=crew_id,
                context=context,
            )

            if delegation["success"]:
                # Execute task
                result = await self.execute_task(
                    task_id=delegation["task_id"],
                    agent_id=delegation["assigned_agent"],
                    task_description=task_desc,
                    context=context,
                )

                if result["success"]:
                    completed += 1
                    agent_outputs[delegation["assigned_agent"]] = result["output"]

        elapsed_ms = (time.time() - start_time) * 1000

        # Gather shared knowledge
        shared_knowledge = dict(self.knowledge_base._local_knowledge)

        return CrewResult(
            crew_id=crew_id,
            tasks_completed=completed,
            total_execution_time_ms=elapsed_ms,
            agent_outputs=agent_outputs,
            shared_knowledge=shared_knowledge,
            success=completed == len(tasks),
            error=None if completed == len(tasks) else "Some tasks failed",
        )

    def get_metrics(self) -> CoordinationMetrics:
        """Get coordination metrics.

        Returns:
            CoordinationMetrics with system statistics
        """
        completed = sum(
            1 for r in self._execution_history
            if r["status"] == TaskStatus.COMPLETED.value
        )
        failed = sum(
            1 for r in self._execution_history
            if r["status"] == TaskStatus.FAILED.value
        )

        avg_time = 0.0
        if self._execution_history:
            total_time = sum(r["execution_time_ms"] for r in self._execution_history)
            avg_time = total_time / len(self._execution_history)

        # Calculate agent utilization
        utilization = {}
        for agent_id in self._agent_pool:
            agent_tasks = sum(
                1 for r in self._execution_history
                if r["agent_id"] == agent_id
            )
            utilization[agent_id] = agent_tasks / max(1, len(self._execution_history))

        return CoordinationMetrics(
            total_tasks_delegated=len(self._execution_history),
            tasks_completed=completed,
            tasks_failed=failed,
            average_execution_time_ms=avg_time,
            agent_utilization=utilization,
            memory_entries=len(self.knowledge_base._local_knowledge),
            active_agents=len(self._agent_pool),
        )


# =============================================================================
# SINGLETON
# =============================================================================

_crew_coordinator: Optional[CrewCoordinator] = None


def get_crew_coordinator(
    max_concurrent: Optional[int] = None,
    memory_enabled: bool = DEFAULT_MEMORY_ENABLED,
) -> CrewCoordinator:
    """Get singleton crew coordinator instance.

    Args:
        max_concurrent: Maximum concurrent agents
        memory_enabled: Whether to enable memory

    Returns:
        CrewCoordinator instance
    """
    global _crew_coordinator
    if _crew_coordinator is None:
        _crew_coordinator = CrewCoordinator(
            max_concurrent=max_concurrent or MAX_CONCURRENT_AGENTS,
            memory_enabled=memory_enabled,
        )
    return _crew_coordinator


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


async def create_coordinated_crew(
    name: str,
    agent_roles: List[str],
    process_type: str = "sequential",
) -> Dict[str, Any]:
    """Create a coordinated crew.

    Convenience function for quick crew creation.
    """
    coordinator = get_crew_coordinator()
    roles = [AgentRole(r) for r in agent_roles]
    return await coordinator.create_crew(name, roles, process_type)


async def delegate_task(
    task_description: str,
    agent_role: Optional[str] = None,
    priority: str = "medium",
) -> DelegationResult:
    """Delegate a task.

    Convenience function for quick task delegation.
    """
    coordinator = get_crew_coordinator()
    role = AgentRole(agent_role) if agent_role else None
    pri = TaskPriority(priority)
    return await coordinator.delegate_task(task_description, agent_role=role, priority=pri)


def get_coordination_metrics() -> CoordinationMetrics:
    """Get coordination metrics.

    Convenience function for metrics.
    """
    coordinator = get_crew_coordinator()
    return coordinator.get_metrics()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Classes
    "CrewCoordinator",
    "TaskDelegator",
    "SharedKnowledgeBase",
    # Result types
    "AgentCapabilities",
    "TaskDefinition",
    "DelegationResult",
    "TaskExecutionResult",
    "CoordinationMetrics",
    "CrewResult",
    # Enums
    "AgentRole",
    "TaskPriority",
    "TaskStatus",
    "DelegationStrategy",
    # Functions
    "get_crew_coordinator",
    "create_coordinated_crew",
    "delegate_task",
    "get_coordination_metrics",
    # Constants
    "COORDINATION_AVAILABLE",
    "CREWAI_AVAILABLE",
]
