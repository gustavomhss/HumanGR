"""Pipeline Orchestrator - DEPRECATED.

This module is DEPRECATED. Use LangGraph workflow instead:

    # Instead of:
    from pipeline.orchestrator import Pipeline
    pipeline = Pipeline()
    result = pipeline.run_sprint(sprint_id)

    # Use:
    from pipeline.langgraph.bridge import run_sprint_sync
    result = run_sprint_sync(sprint_id=sprint_id)

    # Or via CLI:
    PYTHONPATH=src python -m pipeline.cli lg S00

The orchestrator has been replaced by LangGraph StateGraph which provides:
- Native checkpointing (no Temporal required)
- Resume from any checkpoint after crash
- Automatic stack injection
- Complete invariant checking (I1-I11)
- Trust boundary enforcement
- Security guardrails (LLM Guard, QuietStar, NeMo)

For the legacy orchestrator, see: archive/deprecated/orchestrator_legacy.py
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

# Issue deprecation warning on import
warnings.warn(
    "pipeline.orchestrator is DEPRECATED. Use pipeline.langgraph.bridge instead. "
    "See module docstring for migration guide.",
    DeprecationWarning,
    stacklevel=2,
)


@dataclass
class PipelineConfig:
    """Pipeline configuration - DEPRECATED, redirects to LangGraph."""

    run_id: Optional[str] = None
    repo_root: Optional[Path] = None
    output_dir: Optional[Path] = None
    use_temporal: bool = False  # Ignored - LangGraph has native durability
    use_langgraph: bool = True  # Always True now

    def __post_init__(self):
        warnings.warn(
            "PipelineConfig is deprecated. LangGraph uses its own configuration.",
            DeprecationWarning,
            stacklevel=2,
        )


@dataclass
class SprintResult:
    """Result of a sprint execution."""

    sprint_id: str
    status: str  # "completed", "failed", "halted"
    run_id: str
    started_at: str
    completed_at: Optional[str] = None
    gates_passed: List[str] = field(default_factory=list)
    gates_failed: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        """Calculate duration from timestamps."""
        if not self.started_at or not self.completed_at:
            return 0.0
        try:
            start_dt = datetime.fromisoformat(self.started_at.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(self.completed_at.replace("Z", "+00:00"))
            return (end_dt - start_dt).total_seconds()
        except Exception:
            return 0.0


@dataclass
class PipelineState:
    """Pipeline state - DEPRECATED, use pipeline.langgraph.state instead."""

    run_id: str
    sprint_id: str
    status: str = "initialized"
    phase: str = "INIT"


class Pipeline:
    """Pipeline Orchestrator - DEPRECATED.

    This class is a thin wrapper that redirects to LangGraph workflow.
    For new code, use pipeline.langgraph.bridge directly.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize Pipeline (redirects to LangGraph)."""
        warnings.warn(
            "Pipeline class is deprecated. Use LangGraph workflow via bridge.py instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.config = config or PipelineConfig()
        self._initialized = False
        self._state: Optional[PipelineState] = None
        self._active_runs: Dict[str, Dict[str, Any]] = {}

    def init(self, run_id: Optional[str] = None, resume: bool = False) -> PipelineState:
        """Initialize the pipeline state."""
        import uuid
        self._initialized = True
        self._state = PipelineState(
            run_id=run_id or str(uuid.uuid4())[:8],
            sprint_id="",
            status="running" if resume else "initialized",
            phase="INIT",
        )
        return self._state

    def status(self) -> Optional[PipelineState]:
        """Get current pipeline state."""
        return self._state

    def get_active_run_for_sprint(self, sprint_id: str) -> Optional[Dict[str, Any]]:
        """Get active run info for a sprint."""
        return self._active_runs.get(sprint_id)

    def save_active_run_for_sprint(self, sprint_id: str, run_id: str) -> None:
        """Save active run info for a sprint."""
        self._active_runs[sprint_id] = {"run_id": run_id, "sprint_id": sprint_id}

    def clear_active_run_for_sprint(self, sprint_id: str) -> None:
        """Clear active run for a sprint."""
        self._active_runs.pop(sprint_id, None)

    def run_sprint(self, sprint_id: str) -> SprintResult:
        """Run a single sprint using LangGraph.

        This method redirects to LangGraph workflow.
        """
        from pipeline.langgraph.bridge import run_sprint_sync

        try:
            result = run_sprint_sync(sprint_id=sprint_id)

            # Convert LangGraph result to SprintResult
            return SprintResult(
                sprint_id=sprint_id,
                status=result.get("status", "unknown"),
                run_id=result.get("run_id", ""),
                started_at=result.get("started_at", datetime.now(timezone.utc).isoformat()),
                completed_at=result.get("completed_at"),
                gates_passed=result.get("gates_passed", []),
                gates_failed=result.get("gates_failed", []),
                errors=result.get("errors", []),
                artifacts=result.get("artifacts", {}),
            )
        except Exception as e:
            return SprintResult(
                sprint_id=sprint_id,
                status="failed",
                run_id=self.config.run_id or "",
                started_at=datetime.now(timezone.utc).isoformat(),
                errors=[str(e)],
            )

    def run(
        self,
        start_sprint: str = "S00",
        end_sprint: str = "S40",
    ) -> List[SprintResult]:
        """Run multiple sprints using LangGraph.

        This method redirects to LangGraph lg-start functionality.
        """
        from pipeline.langgraph.bridge import LangGraphBridge

        bridge = LangGraphBridge()
        results = []

        # Parse sprint range
        start_num = int(start_sprint.replace("S", "").replace("H", ""))
        end_num = int(end_sprint.replace("S", "").replace("H", ""))

        for i in range(start_num, end_num + 1):
            sprint_id = f"S{i:02d}"
            result = self.run_sprint(sprint_id)
            results.append(result)

            if result.status == "failed":
                break

        return results

    def stack_check(self) -> Dict[str, Any]:
        """Check health of all stacks."""
        try:
            from pipeline.langgraph.validate_all_stacks import check_all_stacks
            return check_all_stacks()
        except ImportError:
            return {"error": "validate_all_stacks not available"}


# Exports for backwards compatibility
__all__ = [
    "Pipeline",
    "PipelineConfig",
    "PipelineState",
    "SprintResult",
]
