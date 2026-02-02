"""
HumanGR Pipeline State

Estado do pipeline SEMPRE com project_id = "HUMANGR".
Nunca pode ser confundido com Veritas.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from .config import get_config


class SprintStatus(str, Enum):
    """Sprint execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class GateStatus(str, Enum):
    """Gate validation status."""
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class GateResult:
    """Result of a gate validation."""
    gate_id: str
    status: GateStatus
    message: str = ""
    details: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SprintResult:
    """Result of a sprint execution."""
    sprint_id: str
    status: SprintStatus
    gates: list[GateResult] = field(default_factory=list)
    deliverables_created: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class PipelineState:
    """
    Pipeline execution state.

    CRITICAL: project_id is ALWAYS "HUMANGR".
    This ensures complete separation from Veritas.
    """

    # Identity - FIXED
    project_id: str = "HUMANGR"  # NEVER changes
    product_name: str = "Human Layer MCP Server"

    # Run identification
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Sprint tracking
    current_sprint: Optional[str] = None
    sprint_range: tuple[str, str] = ("S00", "S40")
    completed_sprints: list[str] = field(default_factory=list)
    failed_sprints: list[str] = field(default_factory=list)

    # Results
    sprint_results: dict[str, SprintResult] = field(default_factory=dict)

    # Paths
    @property
    def run_dir(self) -> Path:
        config = get_config()
        return config.output_dir / "runs" / self.run_id

    @property
    def target_repo(self) -> Path:
        config = get_config()
        return config.target_repo

    # Status
    @property
    def is_complete(self) -> bool:
        """Check if all sprints in range are completed."""
        from .pack_loader import get_sprint_range
        all_sprints = get_sprint_range(self.sprint_range[0], self.sprint_range[1])
        return all(s in self.completed_sprints for s in all_sprints)

    @property
    def progress(self) -> dict[str, Any]:
        """Get progress summary."""
        from .pack_loader import get_sprint_range
        all_sprints = get_sprint_range(self.sprint_range[0], self.sprint_range[1])
        return {
            "total": len(all_sprints),
            "completed": len(self.completed_sprints),
            "failed": len(self.failed_sprints),
            "remaining": len(all_sprints) - len(self.completed_sprints) - len(self.failed_sprints),
            "percentage": round(len(self.completed_sprints) / len(all_sprints) * 100, 1) if all_sprints else 0,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "project_id": self.project_id,
            "product_name": self.product_name,
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(),
            "current_sprint": self.current_sprint,
            "sprint_range": list(self.sprint_range),
            "completed_sprints": self.completed_sprints,
            "failed_sprints": self.failed_sprints,
            "progress": self.progress,
        }


def create_initial_state(
    start_sprint: str = "S00",
    end_sprint: str = "S40",
) -> PipelineState:
    """
    Create initial pipeline state.

    Args:
        start_sprint: First sprint to execute
        end_sprint: Last sprint to execute

    Returns:
        New PipelineState with project_id = "HUMANGR"
    """
    return PipelineState(
        sprint_range=(start_sprint, end_sprint),
    )


def validate_state(state: PipelineState) -> list[str]:
    """
    Validate state for HumanGR compliance.

    Returns:
        List of validation issues (empty if valid)
    """
    issues = []

    # CRITICAL: Must be HumanGR
    if state.project_id != "HUMANGR":
        issues.append(f"CRITICAL: project_id must be 'HUMANGR', got '{state.project_id}'")

    # Check sprint range
    if state.sprint_range[0] > state.sprint_range[1]:
        issues.append(f"Invalid sprint range: {state.sprint_range}")

    return issues
