"""CrewAI client for Pipeline Autonomo v2.0.

This module provides integration with CrewAI framework for orchestrating
hierarchical agent teams. It uses Claude CLI as the LLM backend.

Key Features:
    - Process.hierarchical for manager-worker patterns
    - Integration with hierarchy.py agent structure
    - Claude CLI as LLM provider
    - Task delegation and reporting
    - Stack hooks integration (Langfuse, Redis, Mem0, FalkorDB, Qdrant)

Architecture:
    CrewAI manages crew execution while agents are defined based on
    the pipeline hierarchy. The manager agent delegates to workers
    following the hierarchy rules.

Environment Variables:
    - OTEL_SDK_DISABLED: Disable OpenTelemetry (default: true)
    - CREWAI_TELEMETRY_ENABLED: Enable CrewAI telemetry (default: false)

Author: Pipeline Autonomo Team
Version: 2.1.0 (2026-01-01) - Stack hooks integration
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel

if TYPE_CHECKING:
    from .stack_hooks import StackHooks

logger = logging.getLogger(__name__)

# Security integration for LLM call protection
# Import the sync-compatible wrapper from claude_cli_llm
try:
    from pipeline.claude_cli_llm import (
        secure_operation,
        SECURITY_INTEGRATION_AVAILABLE,
        # 2026-01-20: QuietStar/Reflexion guardrails are integrated in claude_cli_llm
        QUIETSTAR_REFLEXION_AVAILABLE,
        QuietStarBlockedError,
    )
    if SECURITY_INTEGRATION_AVAILABLE:
        from pipeline.security.llm_guard_integration import SecurityBlockedError
    else:
        class SecurityBlockedError(Exception):
            """Placeholder exception when security not available."""
            pass
except ImportError:
    SECURITY_INTEGRATION_AVAILABLE = False
    QUIETSTAR_REFLEXION_AVAILABLE = False
    # Provide no-op decorator if security module not available
    def secure_operation(level: str = "standard", scan_input: bool = True, scan_output: bool = True):
        """No-op decorator when security module not available."""
        def decorator(func):
            return func
        return decorator

    class SecurityBlockedError(Exception):
        """Placeholder exception when security not available."""
        pass

    class QuietStarBlockedError(Exception):
        """Placeholder exception when QuietStar not available."""
        pass

# Disable telemetry before importing crewai
# Note: OTEL_SDK_DISABLED was removed because it also disables Langfuse tracing
# Only disable CrewAI's own telemetry
os.environ.setdefault("CREWAI_TELEMETRY_ENABLED", "false")

# P0-2 FIX: Crew kickoff timeout (30 minutes - VERY GENEROUS)
# This prevents indefinite hangs while allowing complex crews to complete
CREW_KICKOFF_TIMEOUT_SECONDS = 1800  # 30 minutes


# =============================================================================
# Lazy Loaders
# =============================================================================


def _get_stack_hooks() -> Optional["StackHooks"]:
    """Get StackHooks instance (lazy to avoid circular imports).

    Returns:
        StackHooks instance or None if unavailable.
    """
    try:
        from .stack_hooks import get_stack_hooks

        return get_stack_hooks()
    except Exception as e:
        logger.debug(f"StackHooks not available: {e}")
        return None


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class CrewAIConfig:
    """Configuration for CrewAI client.

    F-240 FIX: Memory configuration with explicit documentation.
    """

    model: str = "claude-opus-4-5-20251101"
    temperature: float = 0.7
    max_tokens: int = 4096
    verbose: bool = False

    # F-240 FIX: Memory configuration
    # Memory is disabled by default due to Ollama embeddings timeout issues.
    # When enabled, ensure Ollama is running and responsive:
    #   - ollama serve (must be running)
    #   - ollama pull nomic-embed-text (embeddings model)
    #   - Timeout may need to be increased (see OLLAMA_TIMEOUT env var)
    #
    # To re-enable: Set CREWAI_MEMORY_ENABLED=true in environment
    # or pass memory=True to CrewAIConfig
    memory: bool = False
    memory_timeout_seconds: int = 30  # Timeout for embedding operations

    embedder_model: str = "nomic-embed-text"
    embedder_provider: str = "ollama"

    def __post_init__(self) -> None:
        """F-240: Check environment for memory override."""
        import os
        env_memory = os.getenv("CREWAI_MEMORY_ENABLED", "").lower()
        if env_memory == "true":
            self.memory = True
            logger.info(
                "F-240: CrewAI memory ENABLED via CREWAI_MEMORY_ENABLED=true. "
                "Note: Using native Ollama embedder. Long texts (>4000 chars) may be truncated."
            )


class CrewAIError(Exception):
    """Raised when CrewAI operation fails."""

    pass


class AgentDefinition(BaseModel):
    """Definition for a CrewAI agent."""

    agent_id: str
    role: str
    goal: str
    backstory: str
    tools: list[str] = []
    allow_delegation: bool = True
    verbose: bool = False
    max_iter: int = 10
    max_rpm: int = 10


class TaskDefinition(BaseModel):
    """Definition for a CrewAI task."""

    task_id: str
    description: str
    expected_output: str
    agent_id: str  # ID of the agent assigned to this task
    context: list[str] = []  # IDs of tasks that provide context
    async_execution: bool = False


class CrewResult(BaseModel):
    """Result from crew execution."""

    crew_id: str
    status: str  # "success", "failed", "partial"
    tasks_completed: int
    tasks_failed: int
    output: dict[str, Any] = {}
    errors: list[str] = []
    execution_time_seconds: float = 0.0
    # P1-6.3 FIX: Track degraded features for observability (2026-02-01)
    # Contains list of features that were not available during crew execution
    degraded_features: list[str] = []
    # 2026-01-10 FIX: Use Field with default_factory to get fresh timestamp per instance
    # BEFORE: timestamp: datetime = datetime.now(timezone.utc)
    # BUG (D-HIGH-055): Default was evaluated ONCE at class definition, not per instance
    # Issue: Pydantic field defaults are evaluated at class definition time, not instantiation
    # Tracking: See CONSOLIDATED_GAPS_MASTER.md D-HIGH-055
    timestamp: datetime = None  # type: ignore[assignment]

    def __init__(self, **data: Any) -> None:
        """Initialize with fresh timestamp if not provided."""
        if "timestamp" not in data or data["timestamp"] is None:
            data["timestamp"] = datetime.now(timezone.utc)
        super().__init__(**data)


class CrewAIClient:
    """Client for managing CrewAI crews.

    This client provides a high-level interface for creating and running
    CrewAI crews with hierarchical process. It integrates with the
    pipeline's hierarchy.py for agent structure.

    Usage:
        client = CrewAIClient()

        # Define agents
        agents = [
            AgentDefinition(
                agent_id="qa_master",
                role="QA Master",
                goal="Ensure code quality",
                backstory="Expert QA engineer...",
            ),
            AgentDefinition(
                agent_id="auditor",
                role="Code Auditor",
                goal="Review code for issues",
                backstory="Thorough code reviewer...",
            ),
        ]

        # Define tasks
        tasks = [
            TaskDefinition(
                task_id="review_task",
                description="Review the submitted code",
                expected_output="Review report",
                agent_id="auditor",
            ),
        ]

        # Run crew
        result = client.run_crew(
            crew_id="qa_crew",
            agents=agents,
            tasks=tasks,
            manager_id="qa_master",
        )
    """

    def __init__(self, config: Optional[CrewAIConfig] = None) -> None:
        """Initialize CrewAI client.

        Args:
            config: CrewAI configuration. Uses defaults if None.
        """
        self._config = config or CrewAIConfig()
        self._crews: dict[str, Any] = {}
        self._crewai_available: Optional[bool] = None

    @property
    def config(self) -> CrewAIConfig:
        """Get current configuration."""
        return self._config

    def is_available(self) -> bool:
        """Check if CrewAI is available.

        Returns:
            True if CrewAI is installed and importable.
        """
        if self._crewai_available is not None:
            return self._crewai_available

        try:
            import crewai  # type: ignore[import-not-found]  # noqa: F401

            self._crewai_available = True
        except ImportError:
            self._crewai_available = False

        return self._crewai_available

    def _get_mcp_tools_for_agent(self, agent_id: str) -> tuple[list[Any], list[Any]]:
        """Get MCP tools for an agent.

        PAT-028 FIX: This method creates actual MCP tool objects that can be
        used by CrewAI agents to read/write files.

        Args:
            agent_id: Agent identifier.

        Returns:
            Tuple of (tools list, adapters to stop later).
        """
        tools: list[Any] = []
        adapters: list[Any] = []

        try:
            from crewai_tools import MCPServerAdapter
            from mcp import StdioServerParameters

            from .mcp_tools import get_server_configs_for_agent

            server_configs = get_server_configs_for_agent(agent_id)

            for config in server_configs:
                try:
                    # Create StdioServerParameters from config
                    params = StdioServerParameters(
                        command=config.command,
                        args=config.args,
                        env=config.env,
                    )

                    # Create adapter (starts the MCP server)
                    adapter = MCPServerAdapter(params, connect_timeout=60)
                    adapters.append(adapter)

                    # Get tools from adapter
                    if hasattr(adapter, 'tools') and adapter.tools:
                        tools.extend(adapter.tools)
                        logger.info(
                            f"PAT-028 FIX: Loaded {len(adapter.tools)} MCP tools for {agent_id} "
                            f"from {config.name}"
                        )

                except Exception as e:
                    logger.warning(f"Failed to load MCP tools from {config.name}: {e}")

        except ImportError as e:
            logger.warning(f"MCP tools not available: {e}")

        return tools, adapters

    def _cleanup_mcp_adapter(self, adapter: Any) -> None:
        """Clean up an MCP adapter by trying multiple cleanup methods.

        BUG FIX 2026-01-24: MCPServerAdapter may have different cleanup methods
        depending on the version. Try close(), stop(), __exit__() in order.

        Args:
            adapter: MCP adapter instance to clean up.
        """
        cleanup_methods = ['close', 'stop', '__exit__', 'shutdown', 'disconnect']

        for method_name in cleanup_methods:
            if hasattr(adapter, method_name):
                try:
                    method = getattr(adapter, method_name)
                    if callable(method):
                        # __exit__ needs arguments
                        if method_name == '__exit__':
                            method(None, None, None)
                        else:
                            method()
                        logger.debug(f"MCP adapter cleaned up via {method_name}()")
                        return
                except Exception as e:
                    logger.debug(f"MCP adapter {method_name}() failed: {e}")
                    continue

        # If no cleanup method worked, log warning
        logger.debug(f"MCP adapter has no cleanup method (tried: {cleanup_methods})")

    def _create_crewai_agent(
        self,
        definition: AgentDefinition,
        llm: Any,
        tools: Optional[list[Any]] = None,
    ) -> Any:
        """Create a CrewAI Agent from definition.

        PAT-028 FIX: Now accepts and passes tools to the Agent constructor.

        Args:
            definition: Agent definition.
            llm: LLM instance to use.
            tools: Optional list of tool objects to pass to the agent.

        Returns:
            CrewAI Agent instance.
        """
        if not self.is_available():
            raise CrewAIError("CrewAI is not installed")

        from crewai import Agent

        # PAT-028 FIX: Pass tools to agent so it can actually write files
        return Agent(
            role=definition.role,
            goal=definition.goal,
            backstory=definition.backstory,
            llm=llm,
            tools=tools or [],  # PAT-028: Tools are now passed!
            verbose=definition.verbose or self._config.verbose,
            allow_delegation=definition.allow_delegation,
            max_iter=definition.max_iter,
            max_rpm=definition.max_rpm,
        )

    def _create_crewai_task(
        self,
        definition: TaskDefinition,
        agent: Any,
        context_tasks: Optional[list[Any]] = None,
    ) -> Any:
        """Create a CrewAI Task from definition.

        Args:
            definition: Task definition.
            agent: Agent assigned to the task.
            context_tasks: Tasks that provide context.

        Returns:
            CrewAI Task instance.
        """
        if not self.is_available():
            raise CrewAIError("CrewAI is not installed")

        from crewai import Task

        return Task(
            description=definition.description,
            expected_output=definition.expected_output,
            agent=agent,
            context=context_tasks or [],
            async_execution=definition.async_execution,
        )

    @secure_operation(level="high", scan_input=True, scan_output=True)
    def _secure_kickoff(
        self,
        crew: Any,
        crew_id: str,
        tasks: list[TaskDefinition],
    ) -> Any:
        """Execute crew kickoff with security scanning.

        Security-wrapped with @secure_operation(level="high") for:
        - Input scanning (task descriptions may contain malicious content)
        - Output scanning (crew output may leak sensitive data)
        - Security event logging via Langfuse

        2026-01-20: QuietStar/Reflexion guardrails are also applied via
        ClaudeCLIForCrewAI which integrates with claude_cli_llm. This provides
        an additional layer of pre/post generation safety checks.

        Args:
            crew: The CrewAI Crew instance to execute.
            crew_id: Unique crew identifier for logging.
            tasks: Task definitions for context in security logging.

        Returns:
            Result from crew.kickoff().

        Raises:
            SecurityBlockedError: If security scan fails.
            QuietStarBlockedError: If QuietStar guardrails block the request.
        """
        logger.info(f"SECURITY: Executing secure kickoff for crew {crew_id}")

        # P0-2 FIX: Execute kickoff with timeout to prevent indefinite hangs
        # Uses ThreadPoolExecutor for sync timeout (crew.kickoff is sync)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(crew.kickoff)
            try:
                result = future.result(timeout=CREW_KICKOFF_TIMEOUT_SECONDS)
            except concurrent.futures.TimeoutError:
                logger.error(
                    f"TIMEOUT: Crew {crew_id} kickoff timed out after "
                    f"{CREW_KICKOFF_TIMEOUT_SECONDS} seconds"
                )
                raise TimeoutError(
                    f"Crew {crew_id} kickoff timed out after "
                    f"{CREW_KICKOFF_TIMEOUT_SECONDS} seconds (30 min)"
                )

        logger.info(f"SECURITY: Secure kickoff completed for crew {crew_id}")
        return result

    def run_crew(
        self,
        crew_id: str,
        agents: list[AgentDefinition],
        tasks: list[TaskDefinition],
        manager_id: Optional[str] = None,
        manager_agent: Optional[AgentDefinition] = None,
        process: str = "hierarchical",
        correlation_id: Optional[str] = None,
    ) -> CrewResult:
        """Run a crew with the specified configuration.

        Args:
            crew_id: Unique identifier for this crew run.
            agents: List of agent definitions (workers only for hierarchical).
            tasks: List of task definitions.
            manager_id: ID of the manager agent (for hierarchical process).
            manager_agent: Manager agent definition (for hierarchical process).
            process: Execution process ("hierarchical" or "sequential").
            correlation_id: Correlation ID for tracing (optional).

        Returns:
            CrewResult with execution details.

        Raises:
            CrewAIError: If execution fails.
        """
        start_time = datetime.now(timezone.utc)

        # Get stack hooks for observability
        hooks = _get_stack_hooks()

        if not self.is_available():
            # Return mock result if CrewAI not available
            return CrewResult(
                crew_id=crew_id,
                status="failed",
                tasks_completed=0,
                tasks_failed=len(tasks),
                errors=["CrewAI is not installed"],
                execution_time_seconds=0.0,
            )

        try:
            from crewai import Crew, Process

            from pipeline.claude_cli_llm import ClaudeCLIForCrewAI

            # Create LLM instance
            llm = ClaudeCLIForCrewAI()

            # Notify hooks: task_start for all tasks
            if hooks:
                for task_def in tasks:
                    hooks.on_task_start(
                        agent_id=task_def.agent_id,
                        task_id=task_def.task_id,
                        correlation_id=correlation_id,
                        metadata={"crew_id": crew_id, "description": task_def.description},
                    )

            # PAT-028 FIX: Track all MCP adapters to stop them after kickoff
            all_mcp_adapters: list[Any] = []

            # F-237 FIX: Check worker limits before creating agents
            # Import here to avoid circular dependency
            from pipeline.worker_limits import can_spawn_worker
            from pipeline.hierarchy import get_level

            # Create agents with MCP tools
            crewai_agents: dict[str, Any] = {}
            for agent_def in agents:
                # F-237: Enforce worker limits per INV-012
                agent_level = get_level(agent_def.agent_id) or 5  # Default to L5 worker
                # CQ-002 FIX: Extract squad from agent metadata if available
                agent_squad = getattr(agent_def, 'squad', None) or agent_def.metadata.get('squad') if hasattr(agent_def, 'metadata') else None
                if not can_spawn_worker(
                    agent_type=agent_def.agent_id,
                    level=agent_level,
                    squad=agent_squad,
                    sprint_id=crew_id,
                ):
                    logger.warning(
                        f"F-237: Worker limit reached for {agent_def.agent_id}. "
                        f"Agent creation blocked per resource limits."
                    )
                    raise CrewAIError(
                        f"Worker limit reached - cannot spawn {agent_def.agent_id}. "
                        "Resource limits are enforced per INV-012."
                    )
                # PAT-028 FIX: Get actual MCP tool objects for this agent
                agent_tools, agent_adapters = self._get_mcp_tools_for_agent(
                    agent_def.agent_id
                )
                all_mcp_adapters.extend(agent_adapters)

                crewai_agents[agent_def.agent_id] = self._create_crewai_agent(
                    agent_def, llm, tools=agent_tools  # PAT-028: Pass tools!
                )

            # Create tasks with context
            crewai_tasks: dict[str, Any] = {}
            task_list: list[Any] = []

            for task_def in tasks:
                if task_def.agent_id not in crewai_agents:
                    raise CrewAIError(
                        f"Task {task_def.task_id} references unknown agent {task_def.agent_id}"
                    )

                # Get context tasks
                context_tasks = [
                    crewai_tasks[ctx_id]
                    for ctx_id in task_def.context
                    if ctx_id in crewai_tasks
                ]

                task = self._create_crewai_task(
                    task_def,
                    crewai_agents[task_def.agent_id],
                    context_tasks,
                )
                crewai_tasks[task_def.task_id] = task
                task_list.append(task)

            # Determine process type
            crew_process = (
                Process.hierarchical
                if process == "hierarchical"
                else Process.sequential
            )

            # Get manager agent if hierarchical
            crewai_manager_agent = None
            if process == "hierarchical":
                if manager_agent:
                    # Create manager from provided definition (without tools - managers can't have tools)
                    crewai_manager_agent = self._create_crewai_agent(manager_agent, llm, tools=None)
                    # Also add to crewai_agents dict for task lookups
                    crewai_agents[manager_agent.agent_id] = crewai_manager_agent
                elif manager_id and manager_id in crewai_agents:
                    # Fallback: use existing agent as manager
                    # FIX 2026-01-23: Create a new manager agent WITHOUT tools
                    # In CrewAI hierarchical mode, manager agents cannot have tools
                    existing_agent = crewai_agents[manager_id]
                    # Get the agent definition to recreate without tools
                    manager_def = next(
                        (a for a in agents if a.agent_id == manager_id), None
                    )
                    if manager_def:
                        crewai_manager_agent = self._create_crewai_agent(manager_def, llm, tools=None)
                    else:
                        # Last resort: clear tools on existing agent
                        crewai_manager_agent = existing_agent
                        crewai_manager_agent.tools = []

            # Configure embedder for CrewAI memory
            # 2026-01-24 FIX: Use 'ollama' provider with chunking interceptor.
            #
            # The issue: CrewAI's Pydantic validator uses is_subclass_of which doesn't
            # work correctly with Protocol classes, rejecting our ChunkingOllamaEmbedder
            # even though it IS a valid subclass.
            #
            # Solution: Use 'ollama' provider and enable chunking interceptor that
            # monkey-patches httpx to automatically chunk long texts before they reach Ollama.
            # This is transparent to CrewAI and bypasses the Pydantic validation issue.
            embedder_config = {
                "provider": "ollama",
                "config": {
                    "model_name": self._config.embedder_model,  # "nomic-embed-text"
                    "url": "http://localhost:11434/api/embeddings",
                },
            }
            if self._config.memory:
                # Enable chunking interceptor for long text support
                try:
                    from pipeline.chunking_embedder import enable_ollama_chunking
                    enable_ollama_chunking()
                    logger.info(
                        "CrewAI memory ENABLED with Ollama embedder + chunking interceptor. "
                        "Long texts will be auto-chunked for proper embedding."
                    )
                except ImportError as e:
                    logger.warning(
                        f"CrewAI memory ENABLED but chunking interceptor not available: {e}. "
                        "Long texts (>4000 chars) may fail."
                    )
            else:
                logger.debug(
                    f"CrewAI: Using Ollama embedder with model '{self._config.embedder_model}' "
                    "(memory disabled, no chunking needed)"
                )

            # Create and run crew
            # For hierarchical, exclude manager from agents list
            # FIX 2026-01-23: Also exclude when manager_id is used (not just manager_agent)
            manager_agent_id = manager_agent.agent_id if manager_agent else manager_id
            worker_agents = [
                agent for agent_id, agent in crewai_agents.items()
                if agent_id != manager_agent_id
            ]

            crew = Crew(
                agents=worker_agents,
                tasks=task_list,
                process=crew_process,
                manager_agent=crewai_manager_agent,
                manager_llm=llm,  # Use Claude CLI for manager too
                verbose=self._config.verbose,
                memory=self._config.memory,
                embedder=embedder_config,  # Use Ollama for embeddings
            )

            # CRIT-003 FIX: Wrap kickoff with try/except for proper cleanup
            # SECURITY: Use secure kickoff wrapper for input/output scanning
            try:
                result = self._secure_kickoff(crew, crew_id, tasks)
            except Exception as kickoff_err:
                # CRIT-003 FIX: Cleanup MCP adapters before re-raising
                # BUG FIX 2026-01-24: Try multiple cleanup methods for MCP adapters
                logger.error(f"CRIT-003: CrewAI kickoff failed: {type(kickoff_err).__name__}: {kickoff_err}")
                for adapter in all_mcp_adapters:
                    self._cleanup_mcp_adapter(adapter)
                raise CrewAIError(f"CRIT-003: Crew execution failed: {kickoff_err}") from kickoff_err

            end_time = datetime.now(timezone.utc)
            execution_time = (end_time - start_time).total_seconds()

            # CRIT-004 FIX: Prevent division by zero when tasks empty
            task_count = len(tasks) if tasks else 1  # Prevent division by zero

            # Notify hooks: task_end for all tasks (success)
            if hooks:
                from .stack_hooks import TaskResult

                for task_def in tasks:
                    task_result = TaskResult(
                        task_id=task_def.task_id,
                        agent_id=task_def.agent_id,
                        status="success",
                        decision=f"Completed in crew {crew_id}",
                        output=str(result),
                        duration_ms=execution_time * 1000 / task_count,  # CRIT-004 FIX
                        metadata={"crew_id": crew_id},
                    )
                    hooks.on_task_end(
                        agent_id=task_def.agent_id,
                        task_id=task_def.task_id,
                        result=task_result,
                        correlation_id=correlation_id,
                    )

            # PAT-028 FIX: Close all MCP adapters after crew execution
            # BUG FIX 2026-01-24: Use robust cleanup that tries multiple methods
            for adapter in all_mcp_adapters:
                self._cleanup_mcp_adapter(adapter)

            return CrewResult(
                crew_id=crew_id,
                status="success",
                tasks_completed=len(tasks),
                tasks_failed=0,
                output={"result": str(result)},
                execution_time_seconds=execution_time,
            )

        except Exception as e:
            # PAT-028 FIX: Also cleanup adapters on error
            # BUG FIX 2026-01-24: Use robust cleanup that tries multiple methods
            if 'all_mcp_adapters' in locals():
                for adapter in all_mcp_adapters:
                    self._cleanup_mcp_adapter(adapter)
            end_time = datetime.now(timezone.utc)
            execution_time = (end_time - start_time).total_seconds()

            # Notify hooks: error for all tasks
            if hooks:
                from .stack_hooks import TaskResult

                # CRIT-004 FIX: Prevent division by zero when tasks empty
                task_count = len(tasks) if tasks else 1

                for task_def in tasks:
                    # Record error for each task
                    hooks.on_error(
                        agent_id=task_def.agent_id,
                        error=e,
                        task_id=task_def.task_id,
                        correlation_id=correlation_id,
                    )
                    # Also record task_end with failure status
                    task_result = TaskResult(
                        task_id=task_def.task_id,
                        agent_id=task_def.agent_id,
                        status="failure",
                        error_message=str(e),
                        duration_ms=execution_time * 1000 / task_count,  # CRIT-004 FIX
                        metadata={"crew_id": crew_id},
                    )
                    hooks.on_task_end(
                        agent_id=task_def.agent_id,
                        task_id=task_def.task_id,
                        result=task_result,
                        correlation_id=correlation_id,
                    )

            return CrewResult(
                crew_id=crew_id,
                status="failed",
                tasks_completed=0,
                tasks_failed=len(tasks),
                errors=[str(e)],
                execution_time_seconds=execution_time,
            )

    def run_sequential(
        self,
        crew_id: str,
        agents: list[AgentDefinition],
        tasks: list[TaskDefinition],
    ) -> CrewResult:
        """Run a crew with sequential process.

        Args:
            crew_id: Unique identifier for this crew run.
            agents: List of agent definitions.
            tasks: List of task definitions.

        Returns:
            CrewResult with execution details.
        """
        return self.run_crew(crew_id, agents, tasks, process="sequential")

    def create_correction_task(
        self,
        agent_id: str,
        context: dict[str, Any],
    ) -> Optional[TaskDefinition]:
        """Create a correction task for an agent that failed.

        This method creates a task definition that can be used to correct
        issues identified during sprint execution.

        Args:
            agent_id: ID of the agent that needs to make corrections.
            context: Context about the failure including:
                - error: Error message or reason for correction
                - sprint_id: Sprint where the failure occurred
                - original_result: Original result that needs correction

        Returns:
            TaskDefinition for the correction task, or None if creation failed.
        """
        if not self.is_available():
            logger.warning("CrewAI not available - cannot create correction task")
            return None

        try:
            error = context.get("error", "Unknown error")
            sprint_id = context.get("sprint_id", "unknown")

            task = TaskDefinition(
                task_id=f"correction_{agent_id}_{sprint_id}",
                description=(
                    f"Correction task for {agent_id}.\n"
                    f"Sprint: {sprint_id}\n"
                    f"Issue: {error}\n\n"
                    "Please analyze the issue and provide corrections."
                ),
                expected_output=(
                    "Analysis of the root cause and corrective actions taken. "
                    "Include: 1) What went wrong, 2) How it was fixed, "
                    "3) Verification that the fix works."
                ),
                agent_id=agent_id,
            )

            logger.info(f"Created correction task {task.task_id} for {agent_id}")
            return task

        except Exception as e:
            logger.error(f"Failed to create correction task for {agent_id}: {e}")
            return None


# =============================================================================
# Singleton and Factory
# =============================================================================


_client: Optional[CrewAIClient] = None


def get_crewai_client(config: Optional[CrewAIConfig] = None) -> CrewAIClient:
    """Get the global CrewAI client instance.

    Args:
        config: Optional configuration (only used on first call).

    Returns:
        The CrewAI client singleton.
    """
    global _client
    if _client is None:
        _client = CrewAIClient(config)
    return _client


def reset_crewai_client() -> None:
    """Reset the global client instance (for testing)."""
    global _client
    _client = None
