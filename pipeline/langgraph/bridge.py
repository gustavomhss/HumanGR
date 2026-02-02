"""LangGraph Bridge - Connects V2 Orchestrator with LangGraph Control Plane.

This module provides the integration point between the existing Pipeline V2
orchestrator and the new LangGraph-based control plane.

Key Features:
- Adapts Pydantic PipelineState to TypedDict PipelineState
- Provides run_sprint_langgraph() as drop-in replacement for run_sprint()
- Maintains backward compatibility with existing CLI and APIs
- Enables gradual migration via feature flags

Architecture:
    CLI / API
        |
        v
    LangGraphBridge
        |
        +---> LangGraph Workflow (if enabled)
        |         |
        |         v
        |     WorkflowNodes (INIT -> EXEC -> GATE -> SIGNOFF -> ARTIFACT)
        |         |
        |         v
        |     Checkpointer (Redis/File)
        |
        +---> Original Pipeline.run_sprint() (fallback)

Invariants:
- I1: run_id namespacing preserved
- I2: Idempotency via checkpoint IDs
- I10: LangGraph control plane is the source of truth when enabled

Author: Pipeline Autonomo Team
Version: 1.0.0 (2026-01-16)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

from pydantic import BaseModel

# Local imports
from pipeline.langgraph.state import (
    PipelineState as LangGraphState,
    PipelineStatus,
    create_initial_state,
)
from pipeline.langgraph.workflow import (
    build_workflow,
    LANGGRAPH_AVAILABLE,
)
from pipeline.langgraph.checkpointer import (
    create_checkpointer,
    create_langgraph_checkpointer,
    Checkpoint,
)
from pipeline.langgraph.stack_injection import (
    get_stack_injector,
)

# Metrics v2 integration for Grafana dashboard (Redis direct)
try:
    from pipeline.metrics_v2 import get_metrics_publisher
    METRICS_V2_AVAILABLE = True
except ImportError:
    METRICS_V2_AVAILABLE = False
    get_metrics_publisher = None  # type: ignore

if TYPE_CHECKING:
    from pipeline.orchestrator import Pipeline, PipelineState, SprintResult

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


class LangGraphConfig(BaseModel):
    """Configuration for LangGraph integration."""

    # Feature flags
    enabled: bool = True
    use_checkpointing: bool = True
    checkpointer_mode: str = "hybrid"  # file, redis, hybrid

    # Directories
    run_dir: Path = Path("out/pipeline")
    # FIX 2026-01-27: Use .langgraph/checkpoints to match cockpit server expectations
    checkpoint_dir: Path = Path(".langgraph/checkpoints")

    # Timeouts
    # P0-08: GENEROUS + 30% gordura - NUNCA falhar por timeout
    node_timeout_seconds: int = 2340  # 39 min per node
    workflow_timeout_seconds: int = 18720  # 5h12m per workflow/sprint

    # Fallback
    fallback_on_error: bool = True

    class Config:
        arbitrary_types_allowed = True


def get_langgraph_config() -> LangGraphConfig:
    """Get LangGraph configuration from environment."""
    return LangGraphConfig(
        enabled=os.getenv("LANGGRAPH_ENABLED", "true").lower() == "true",
        use_checkpointing=os.getenv("LANGGRAPH_CHECKPOINTING", "true").lower() == "true",
        checkpointer_mode=os.getenv("LANGGRAPH_CHECKPOINTER_MODE", "hybrid"),
        run_dir=Path(os.getenv("PIPELINE_RUN_DIR", "out/pipeline")),
        checkpoint_dir=Path(os.getenv("LANGGRAPH_CHECKPOINT_DIR", ".langgraph/checkpoints")),
    )


# =============================================================================
# STATE ADAPTERS
# =============================================================================


def pydantic_to_langgraph_state(
    pydantic_state: "PipelineState",
    sprint_id: str,
) -> LangGraphState:
    """Convert Pydantic PipelineState to LangGraph TypedDict state.

    This adapter bridges the existing V2 orchestrator state model with
    the new LangGraph state format.

    Args:
        pydantic_state: V2 orchestrator PipelineState (Pydantic).
        sprint_id: Current sprint identifier.

    Returns:
        LangGraph-compatible TypedDict state.
    """
    # Create base state with required fields
    state = create_initial_state(
        run_id=pydantic_state.run_id,
        sprint_id=sprint_id,
        run_dir=str(pydantic_state.run_dir) if hasattr(pydantic_state, "run_dir") else "out/pipeline",
    )

    # Map status (align with actual PipelineStatus enum values)
    status_map = {
        "initialized": PipelineStatus.INITIALIZED,
        "running": PipelineStatus.RUNNING,
        "completed": PipelineStatus.COMPLETED,
        "failed": PipelineStatus.FAILED,
        "halted": PipelineStatus.HALTED,
        "recovered": PipelineStatus.RECOVERED,
        # Backward compatibility for legacy status values
        "idle": PipelineStatus.INITIALIZED,
        "paused": PipelineStatus.HALTED,
    }
    if hasattr(pydantic_state, "status"):
        state["status"] = status_map.get(pydantic_state.status, PipelineStatus.INITIALIZED).value

    # Map trace information
    if hasattr(pydantic_state, "trace_id") and pydantic_state.trace_id:
        state["trace_id"] = pydantic_state.trace_id

    # Map governance state (flat fields in PipelineState, not nested)
    if hasattr(pydantic_state, "handoffs"):
        state["handoffs"] = pydantic_state.handoffs or {}
    if hasattr(pydantic_state, "approvals"):
        state["approvals"] = pydantic_state.approvals or {}
    if hasattr(pydantic_state, "signoffs"):
        state["signoffs"] = pydantic_state.signoffs or {}

    # Map completed/failed sprints
    if hasattr(pydantic_state, "sprints_completed"):
        state["sprints_completed"] = pydantic_state.sprints_completed or []
    if hasattr(pydantic_state, "sprints_failed"):
        state["sprints_failed"] = pydantic_state.sprints_failed or []

    return state


def langgraph_to_sprint_result(
    lg_state: LangGraphState,
    start_time: datetime,
) -> "SprintResult":
    """Convert LangGraph state to SprintResult for backward compatibility.

    Args:
        lg_state: Final LangGraph state after workflow completion.
        start_time: When the sprint started.

    Returns:
        SprintResult compatible with existing V2 APIs.
    """
    # Import here to avoid circular dependency
    from pipeline.orchestrator import SprintResult

    end_time = datetime.now(timezone.utc)

    # Determine status
    status_map = {
        PipelineStatus.COMPLETED.value: "completed",
        PipelineStatus.FAILED.value: "failed",
        PipelineStatus.HALTED.value: "halted",
        PipelineStatus.RUNNING.value: "running",
        PipelineStatus.INITIALIZED.value: "initialized",
        PipelineStatus.RECOVERED.value: "completed",
    }
    # Handle missing status gracefully (e.g., when gates fail and trigger safe_halt)
    raw_status = lg_state.get("status", PipelineStatus.FAILED.value)
    status = status_map.get(raw_status, "failed")

    # Collect errors
    errors = []
    for error in lg_state.get("errors", []):
        if isinstance(error, dict):
            errors.append(error.get("message", str(error)))
        else:
            errors.append(str(error))

    # Collect gates info
    gate_state = lg_state.get("gate", {})
    gates_passed = []
    gates_failed = []
    if isinstance(gate_state, dict):
        gates_passed = gate_state.get("gates_passed", [])
        gates_failed = gate_state.get("gates_failed", [])

    return SprintResult(
        sprint_id=lg_state.get("sprint_id", "unknown"),
        status=status,
        run_id=lg_state.get("run_id", "unknown"),
        started_at=start_time.isoformat(),
        completed_at=end_time.isoformat(),
        gates_passed=gates_passed,
        gates_failed=gates_failed,
        errors=errors,
        artifacts=lg_state.get("artifacts", {}),
    )


# =============================================================================
# LANGGRAPH BRIDGE
# =============================================================================


class LangGraphBridge:
    """Bridge between V2 Orchestrator and LangGraph control plane.

    This class provides:
    1. run_sprint_langgraph() as alternative to Pipeline.run_sprint()
    2. Checkpoint management for resume
    3. State conversion between Pydantic and TypedDict

    Usage:
        bridge = LangGraphBridge(pipeline)
        result = await bridge.run_sprint_langgraph("S00")
    """

    def __init__(
        self,
        pipeline: Optional["Pipeline"] = None,
        config: Optional[LangGraphConfig] = None,
    ):
        """Initialize the LangGraph bridge.

        Args:
            pipeline: Optional V2 Pipeline instance for fallback.
            config: LangGraph configuration.
        """
        self._pipeline = pipeline
        self._config = config or get_langgraph_config()
        self._checkpointer = None
        self._workflow = None
        self._stack_injector = None

    @property
    def config(self) -> LangGraphConfig:
        """Get current configuration."""
        return self._config

    @property
    def is_available(self) -> bool:
        """Check if LangGraph is available and enabled."""
        return LANGGRAPH_AVAILABLE and self._config.enabled

    async def _ensure_checkpointer(self):
        """Ensure checkpointer is initialized based on configured mode.

        Supports three modes via checkpointer_mode config:
        - "file": File-based only (durable, slower)
        - "redis": Redis-based only (fast, ephemeral)
        - "hybrid": Redis primary + File backup (recommended)
        """
        if self._checkpointer is None and self._config.use_checkpointing:
            checkpoint_dir = self._config.checkpoint_dir
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            mode = self._config.checkpointer_mode
            redis_client = None

            # Get Redis client for redis/hybrid modes
            # GHOST CODE INTEGRATION (2026-01-30): Use redis_enhanced for better
            # connection pooling, health checks, and metrics
            if mode in ("redis", "hybrid"):
                # Try redis_enhanced first (preferred)
                try:
                    from pipeline.infrastructure.redis_enhanced import (
                        RedisConnectionPool,
                        REDIS_ENHANCED_AVAILABLE,
                    )

                    if REDIS_ENHANCED_AVAILABLE:
                        pool = RedisConnectionPool()
                        redis_client = await pool.get_async_client()
                        if redis_client is not None and pool.is_healthy:
                            logger.info(
                                "Using redis_enhanced with connection pooling and health checks"
                            )
                        else:
                            redis_client = None
                            logger.warning("redis_enhanced pool unhealthy, trying fallback")
                except ImportError:
                    pass  # Fall through to legacy client

                # Fallback to legacy redis client
                if redis_client is None:
                    try:
                        from pipeline.redis_client import get_redis_client

                        redis_client = get_redis_client()
                    except ImportError as e:
                        logger.warning(f"Redis client import failed: {e}")

                if redis_client is None:
                    logger.warning(
                        f"Redis client unavailable for {mode} mode, "
                        "falling back to file-only checkpointing"
                    )
                    mode = "file"

            # Create checkpointer using factory function
            self._checkpointer = await create_checkpointer(
                run_dir=checkpoint_dir,
                redis_client=redis_client,
                storage=mode,
            )

            logger.info(f"Initialized {mode} checkpointer for {checkpoint_dir}")

    def _ensure_stack_injector(self):
        """Ensure stack injector is initialized."""
        if self._stack_injector is None:
            self._stack_injector = get_stack_injector()

    async def run_sprint_langgraph(
        self,
        sprint_id: str,
        run_id: Optional[str] = None,
        resume_from: Optional[str] = None,
    ) -> "SprintResult":
        """Run a sprint using LangGraph control plane.

        This is the main entry point for LangGraph-based sprint execution.
        It can be used as a drop-in replacement for Pipeline.run_sprint().

        Args:
            sprint_id: Sprint identifier (e.g., "S00", "S01").
            run_id: Optional run identifier. Generated if not provided.
            resume_from: Optional checkpoint ID to resume from.

        Returns:
            SprintResult compatible with existing APIs.

        Raises:
            ImportError: If LangGraph is not available.
            RuntimeError: If workflow execution fails.
        """
        if not self.is_available:
            # CRIT: DO NOT fallback to self._pipeline.run_sprint() as it creates
            # an infinite recursion loop (Pipeline.run_sprint calls this bridge).
            raise ImportError(
                "LangGraph not available. Install with: pip install langgraph"
            )

        start_time = datetime.now(timezone.utc)

        # Generate run_id if not provided
        if run_id is None:
            run_id = f"lg_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        run_dir = self._config.run_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Add file handler for run.log (allows cockpit to read logs for adopted pipelines)
        run_log_path = run_dir / "run.log"
        file_handler = logging.FileHandler(run_log_path, mode='a')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        # Add handler to root logger so ALL log messages go to file
        logging.getLogger().addHandler(file_handler)
        # Store reference to remove handler later
        self._run_log_handler = file_handler

        logger.info(f"Starting LangGraph sprint execution: {sprint_id} (run: {run_id})")

        # Publish sprint start to Grafana dashboard (metrics v2)
        if METRICS_V2_AVAILABLE and get_metrics_publisher:
            try:
                metrics = get_metrics_publisher()
                metrics.start_sprint(sprint_id)
                metrics.set_phase("INIT")
                logger.debug(f"Published sprint start metrics for {sprint_id}")
            except Exception as e:
                logger.debug(f"Failed to publish sprint start metrics: {e}")

        try:
            # Ensure services are initialized
            await self._ensure_checkpointer()
            self._ensure_stack_injector()

            # Verify stacks before execution
            if self._stack_injector:
                health = self._stack_injector.check_health()
                unhealthy = [
                    name for name, status in health.items()
                    if not status.get("healthy", False)
                ]
                if unhealthy:
                    logger.warning(f"Unhealthy stacks detected: {unhealthy}")

            # Build workflow with checkpointing
            lg_checkpointer = None
            if self._checkpointer:
                lg_checkpointer = create_langgraph_checkpointer(self._checkpointer)

            workflow = build_workflow(
                run_dir=run_dir,
                checkpointer=lg_checkpointer,
            )

            if workflow is None:
                raise RuntimeError("Failed to build LangGraph workflow")

            # Create initial state
            initial_state = create_initial_state(
                run_id=run_id,
                sprint_id=sprint_id,
                run_dir=str(run_dir),
            )

            # Configure workflow
            workflow_config = {
                "configurable": {
                    "thread_id": run_id,
                },
            }

            # Handle resume
            if resume_from:
                logger.info(f"Resuming from checkpoint: {resume_from}")
                checkpoint = None

                # FIX 2026-01-28: Search run-specific checkpoint directory first
                # Checkpoints saved during pipeline runs are in out/runs/{run_id}/checkpoints/
                import re
                run_id_match = re.search(r'lg_\d{8}_\d{6}_[a-f0-9]+', resume_from)
                if run_id_match:
                    extracted_run_id = run_id_match.group(0)
                    run_checkpoint_dir = Path("out/runs") / extracted_run_id / "checkpoints"
                    run_checkpoint_file = run_checkpoint_dir / f"{resume_from}.json"

                    if run_checkpoint_file.exists():
                        logger.info(f"Loading checkpoint from run directory: {run_checkpoint_file}")
                        try:
                            with open(run_checkpoint_file) as f:
                                data = json.load(f)
                                checkpoint = Checkpoint.from_dict(data)
                        except Exception as e:
                            logger.warning(f"Failed to load checkpoint from run dir: {e}")

                # Fall back to checkpointer
                if checkpoint is None and self._checkpointer:
                    checkpoint = await self._checkpointer.get(resume_from)

                if checkpoint and checkpoint.state:
                        # FIX 2026-01-27: MERGE checkpoint state INTO initial_state
                        # Don't replace completely - old checkpoints may have incomplete state
                        # This ensures required fields (run_id, sprint_id, etc.) are always present
                        checkpoint_state = checkpoint.state
                        if isinstance(checkpoint_state, dict):
                            for key, value in checkpoint_state.items():
                                if value is not None:  # Only override with non-None values
                                    initial_state[key] = value
                        logger.info(f"Loaded checkpoint state: phase={initial_state.get('phase')}, run_id={initial_state.get('run_id')}, _skip_exec={initial_state.get('_skip_exec')}, _resume_from_gates={initial_state.get('_resume_from_gates')}")

            # Execute workflow using astream for automatic checkpointing after each node
            # CRITICAL FIX 2026-01-24: ainvoke() does NOT save intermediate checkpoints.
            # Only astream()/stream() triggers checkpoint saves after each node.
            # This is essential for resume functionality to work correctly.
            #
            # CRITICAL FIX 2026-01-27: Use stream_mode="values" to get full merged state.
            # Default astream yields {node_name: output} which is NOT the full state.
            # With stream_mode="values", it yields the complete state after each node.
            logger.info(f"Executing LangGraph workflow for sprint {sprint_id}")
            final_state = initial_state
            async for state_update in workflow.astream(
                initial_state,
                config=workflow_config,
                stream_mode="values",  # FIX: Get full state, not just node output
            ):
                # astream with stream_mode="values" yields full state after each node
                # The checkpointer automatically saves state after each yield
                final_state = state_update
                current_phase = final_state.get("phase", "unknown") if isinstance(final_state, dict) else "unknown"
                logger.debug(f"Checkpoint saved after phase: {current_phase}")

            # Convert to SprintResult
            result = langgraph_to_sprint_result(final_state, start_time)

            # Save final checkpoint
            # FIX 2026-01-27: Use human-readable ID format that won't break filesystem
            # Format: lg_{timestamp}_{run_suffix}_final
            # The sprint_id and run_id are stored as attributes in the Checkpoint dataclass
            if self._checkpointer:
                run_suffix = run_id.split("_")[-1] if "_" in run_id else run_id[:8]
                checkpoint_id = f"lg_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{run_suffix}_final"
                final_checkpoint = Checkpoint(
                    checkpoint_id=checkpoint_id,
                    run_id=run_id,
                    sprint_id=sprint_id,
                    thread_id=run_id,
                    ts=datetime.now(timezone.utc).isoformat(),
                    state=final_state,
                    metadata={"final": True, "step": 0},  # Include step for LangGraph
                )
                await self._checkpointer.put(final_checkpoint)
                logger.info(f"Saved final checkpoint: {checkpoint_id}")

            # Calculate duration from timestamps (datetime already imported at module level)
            duration_str = ""
            if result.started_at and result.completed_at:
                try:
                    start_dt = datetime.fromisoformat(result.started_at.replace("Z", "+00:00"))
                    end_dt = datetime.fromisoformat(result.completed_at.replace("Z", "+00:00"))
                    duration = (end_dt - start_dt).total_seconds()
                    duration_str = f" (duration: {duration:.2f}s)"
                except Exception as e:
                    logger.debug(f"GRAPH: Graph operation failed: {e}")
            logger.info(f"LangGraph sprint {sprint_id} completed: {result.status}{duration_str}")

            # Publish sprint completion to Grafana dashboard (metrics_v2 -> Redis)
            if METRICS_V2_AVAILABLE and get_metrics_publisher:
                try:
                    metrics = get_metrics_publisher()
                    success = result.status == "completed"
                    metrics.end_sprint(sprint_id, success=success)
                    if not success:
                        metrics.set_status("failed")
                    logger.debug(f"Published sprint completion metrics for {sprint_id}")
                except Exception as e:
                    logger.debug(f"Failed to publish sprint completion metrics: {e}")

            return result

        except Exception as e:
            import traceback
            error_msg = str(e) if str(e) else f"{type(e).__name__}: (no message)"
            full_traceback = traceback.format_exc()
            logger.error(f"LangGraph sprint {sprint_id} failed: {error_msg}")
            logger.error(f"Full traceback:\n{full_traceback}")

            # Publish failure to Grafana dashboard (metrics_v2 -> Redis)
            if METRICS_V2_AVAILABLE and get_metrics_publisher:
                try:
                    metrics = get_metrics_publisher()
                    metrics.set_status("failed")
                    metrics.end_sprint(sprint_id, success=False)
                    logger.debug(f"Published sprint failure metrics for {sprint_id}")
                except Exception as e:
                    logger.debug(f"GRAPH: Graph operation failed: {e}")

            # CRIT: DO NOT fallback to self._pipeline.run_sprint() as it creates
            # an infinite recursion loop (Pipeline.run_sprint calls this bridge,
            # which fails, then falls back to Pipeline.run_sprint again).
            # Instead, re-raise the error with full context.
            raise RuntimeError(f"LangGraph workflow failed for {sprint_id}: {error_msg}") from e

        finally:
            # Cleanup file handler for run.log
            if hasattr(self, '_run_log_handler') and self._run_log_handler:
                try:
                    logging.getLogger().removeHandler(self._run_log_handler)
                    self._run_log_handler.close()
                    self._run_log_handler = None
                except Exception as e:
                    logger.debug(f"GRAPH: Graph operation failed: {e}")

    async def get_checkpoint_status(
        self,
        run_id: str,
    ) -> Dict[str, Any]:
        """Get status of checkpoints for a run.

        Args:
            run_id: Run identifier.

        Returns:
            Dictionary with checkpoint information.
        """
        await self._ensure_checkpointer()

        if not self._checkpointer:
            return {"available": False, "reason": "Checkpointing disabled"}

        checkpoints = await self._checkpointer.list(thread_id=run_id)

        # Convert Checkpoint objects to dicts for serialization
        checkpoint_list = []
        for cp in checkpoints:
            checkpoint_list.append({
                "id": cp.checkpoint_id,
                "created_at": cp.ts,
                "phase": cp.state.get("phase", "unknown") if isinstance(cp.state, dict) else "unknown",
            })

        return {
            "available": True,
            "run_id": run_id,
            "checkpoints": checkpoint_list,
            "count": len(checkpoints),
        }

    async def resume_from_checkpoint(
        self,
        checkpoint_id: str,
    ) -> "SprintResult":
        """Resume workflow from a specific checkpoint.

        Args:
            checkpoint_id: Checkpoint identifier.

        Returns:
            SprintResult after resumed execution.
        """
        await self._ensure_checkpointer()

        if not self._checkpointer:
            raise RuntimeError("Checkpointing is disabled")

        # FIX 2026-01-28: Search for checkpoint in run-specific directory first
        # Checkpoints are saved in out/runs/{run_id}/checkpoints/ but the default
        # checkpointer looks in .langgraph/checkpoints/
        checkpoint = None

        # Try to find checkpoint in run-specific directory
        # Extract run_id from checkpoint_id (format: ckpt_{run_id}_{sprint}_...)
        # run_id format: lg_YYYYMMDD_HHMMSS_XXXXXXXX (contains underscores)
        import re
        run_id_match = re.search(r'lg_\d{8}_\d{6}_[a-f0-9]+', checkpoint_id)
        if run_id_match:
            extracted_run_id = run_id_match.group(0)
            run_checkpoint_dir = Path("out/runs") / extracted_run_id / "checkpoints"
            run_checkpoint_file = run_checkpoint_dir / f"{checkpoint_id}.json"

            if run_checkpoint_file.exists():
                logger.info(f"Found checkpoint in run directory: {run_checkpoint_file}")
                try:
                    with open(run_checkpoint_file) as f:
                        data = json.load(f)
                        checkpoint = Checkpoint.from_dict(data)
                except Exception as e:
                    logger.warning(f"Failed to load checkpoint from run dir: {e}")

        # Fall back to default checkpointer
        if checkpoint is None:
            checkpoint = await self._checkpointer.get(checkpoint_id)

        if checkpoint is None:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")

        # FIX 2026-01-27: Checkpoint is a dataclass with sprint_id as direct attribute
        # The original code incorrectly looked for sprint_id inside state dict
        sprint_id = checkpoint.sprint_id
        run_id = checkpoint.run_id

        # Fallback: parse from checkpoint_id if attributes are missing/unknown
        if not sprint_id or sprint_id == "unknown":
            if checkpoint_id.startswith("ckpt_"):
                # Format: ckpt_{run_id}_{sprint_id}_{phase}_attempt_{n}_{hash}
                parts = checkpoint_id.split("_")
                if len(parts) >= 3:
                    sprint_id = parts[2]  # sprint_id is 3rd part
                if len(parts) >= 2 and (not run_id or run_id == "unknown"):
                    run_id = parts[1]
            elif checkpoint_id.startswith("lg_"):
                # Final checkpoint format: lg_{timestamp}_{suffix}_final
                # Get sprint_id from state since it's not in the ID
                state = checkpoint.state
                if isinstance(state, dict):
                    sprint_id = state.get("sprint_id", "S00")
                    if not run_id or run_id == "unknown":
                        run_id = state.get("run_id", checkpoint_id.replace("_final", ""))
                else:
                    sprint_id = "S00"
                    if not run_id or run_id == "unknown":
                        run_id = checkpoint_id.replace("_final", "")
            elif ":" in checkpoint_id:
                # Legacy format: ckpt:{run_id}:{sprint_id}:{phase}:attempt:{n}:{hash}
                parts = checkpoint_id.split(":")
                if len(parts) >= 3:
                    sprint_id = parts[2]
                if len(parts) >= 2 and (not run_id or run_id == "unknown"):
                    run_id = parts[1]
            else:
                # Unknown format - try state
                state = checkpoint.state
                if isinstance(state, dict) and "gate" in state:
                    sprint_id = state["gate"].get("sprint_id", "S00")
                else:
                    sprint_id = "S00"  # Safe fallback
                if not run_id or run_id == "unknown":
                    run_id = checkpoint_id.replace("_final", "")

        logger.info(f"Resuming from checkpoint {checkpoint_id} (sprint: {sprint_id}, run: {run_id})")

        return await self.run_sprint_langgraph(
            sprint_id=sprint_id,
            run_id=run_id,
            resume_from=checkpoint_id,
        )


# =============================================================================
# SINGLETON ACCESS
# =============================================================================


_bridge_instance: Optional[LangGraphBridge] = None
_bridge_lock = threading.Lock()


def get_langgraph_bridge(
    pipeline: Optional["Pipeline"] = None,
    config: Optional[LangGraphConfig] = None,
) -> LangGraphBridge:
    """Get or create the LangGraph bridge singleton.

    Thread-safe implementation using double-checked locking pattern.
    HIGH-001 Fix: Prevents race conditions when multiple threads call
    this function simultaneously.

    Args:
        pipeline: Optional Pipeline instance for fallback.
        config: Optional configuration override.

    Returns:
        LangGraphBridge instance.
    """
    global _bridge_instance

    # Quick check without lock (optimization for common case)
    if _bridge_instance is not None:
        if pipeline is not None:
            with _bridge_lock:
                _bridge_instance._pipeline = pipeline
        return _bridge_instance

    # Thread-safe initialization
    with _bridge_lock:
        # Double-check after acquiring lock
        if _bridge_instance is None:
            _bridge_instance = LangGraphBridge(pipeline=pipeline, config=config)
        elif pipeline is not None:
            _bridge_instance._pipeline = pipeline

    return _bridge_instance


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


async def run_sprint_with_langgraph(
    sprint_id: str,
    run_id: Optional[str] = None,
    resume_from: Optional[str] = None,
    pipeline: Optional["Pipeline"] = None,
) -> "SprintResult":
    """Convenience function to run a sprint with LangGraph.

    This is the recommended entry point for running sprints with LangGraph.

    Args:
        sprint_id: Sprint identifier.
        run_id: Optional run identifier.
        resume_from: Optional checkpoint to resume from.
        pipeline: Optional Pipeline for fallback.

    Returns:
        SprintResult.
    """
    bridge = get_langgraph_bridge(pipeline=pipeline)
    return await bridge.run_sprint_langgraph(
        sprint_id=sprint_id,
        run_id=run_id,
        resume_from=resume_from,
    )


def run_sprint_sync(
    sprint_id: str,
    run_id: Optional[str] = None,
    resume_from: Optional[str] = None,
    pipeline: Optional["Pipeline"] = None,
) -> "SprintResult":
    """Synchronous wrapper for run_sprint_with_langgraph.

    Use this when calling from synchronous code (e.g., CLI).

    Args:
        sprint_id: Sprint identifier.
        run_id: Optional run identifier.
        resume_from: Optional checkpoint to resume from.
        pipeline: Optional Pipeline for fallback.

    Returns:
        SprintResult.
    """
    coro = run_sprint_with_langgraph(
        sprint_id=sprint_id,
        run_id=run_id,
        resume_from=resume_from,
        pipeline=pipeline,
    )

    # Handle case where we're already in an event loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, use asyncio.run()
        return asyncio.run(coro)

    # Already in a loop - use nest_asyncio or run in new thread
    try:
        import nest_asyncio
        nest_asyncio.apply()
        return asyncio.run(coro)
    except ImportError:
        # Fallback: run in a new thread with its own event loop
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Config
    "LangGraphConfig",
    "get_langgraph_config",
    # Adapters
    "pydantic_to_langgraph_state",
    "langgraph_to_sprint_result",
    # Bridge
    "LangGraphBridge",
    "get_langgraph_bridge",
    # Convenience
    "run_sprint_with_langgraph",
    "run_sprint_sync",
]
