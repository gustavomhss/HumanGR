"""Cost Controller - Budget and rate limit enforcement.

F-209 FIX: This module implements cost tracking and budget enforcement
based on cost_policy.yml. It prevents runaway API costs by:
- Tracking token usage per sprint and run
- Enforcing budget limits
- Rate limiting requests
- Alerting on threshold breaches

Author: Pipeline Autonomo Team
Version: 1.0.0 (2026-01-08)
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

# Default location for cost policy
DEFAULT_COST_POLICY = Path(__file__).parent.parent.parent / "configs" / "cost_policy.yml"


@dataclass
class CostPolicy:
    """Cost policy configuration.

    Loaded from cost_policy.yml with budget and rate limits.
    """

    max_budget_usd_per_run: float = 10.0
    max_budget_usd_per_sprint: float = 2.0
    rate_limit_tokens_per_minute: int = 100000
    cost_per_prompt_token: float = 0.000003
    cost_per_completion_token: float = 0.000015
    warning_threshold_percent: float = 80.0
    enforce_budget: bool = True
    enforce_rate_limit: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CostPolicy:
        """Create CostPolicy from dictionary."""
        policy = data.get("cost_policy", {})
        return cls(
            max_budget_usd_per_run=policy.get("max_budget_usd_per_run", 10.0),
            max_budget_usd_per_sprint=policy.get("max_budget_usd_per_sprint", 2.0),
            rate_limit_tokens_per_minute=policy.get("rate_limit_tokens_per_minute", 100000),
            cost_per_prompt_token=policy.get("cost_per_prompt_token", 0.000003),
            cost_per_completion_token=policy.get("cost_per_completion_token", 0.000015),
            warning_threshold_percent=policy.get("warning_threshold_percent", 80.0),
            enforce_budget=policy.get("enforce_budget", True),
            enforce_rate_limit=policy.get("enforce_rate_limit", True),
        )


@dataclass
class UsageRecord:
    """Record of a single LLM usage."""

    timestamp: datetime
    agent_id: str
    sprint_id: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    model: str = "claude-sonnet-3.5"


@dataclass
class CostMetrics:
    """Current cost metrics for a run."""

    run_id: str
    total_cost_usd: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    by_sprint: dict[str, float] = field(default_factory=dict)
    by_agent: dict[str, float] = field(default_factory=dict)
    usage_records: list[UsageRecord] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "by_sprint": self.by_sprint,
            "by_agent": self.by_agent,
            "started_at": self.started_at.isoformat(),
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }


class CostLimitExceeded(Exception):
    """Raised when a cost limit is exceeded."""

    def __init__(self, message: str, current: float, limit: float):
        super().__init__(message)
        self.current = current
        self.limit = limit


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str, tokens_used: int, limit: int):
        super().__init__(message)
        self.tokens_used = tokens_used
        self.limit = limit


class CostController:
    """Cost tracking and budget enforcement.

    F-209: Implements budget limits and rate limiting per cost_policy.yml.

    Thread-safe implementation for concurrent access.
    """

    def __init__(
        self,
        run_id: str,
        policy: Optional[CostPolicy] = None,
        policy_path: Optional[Path] = None,
    ):
        """Initialize cost controller.

        Args:
            run_id: Run identifier for tracking.
            policy: Cost policy. Loaded from file if None.
            policy_path: Path to cost policy YAML.
        """
        self.run_id = run_id
        self.policy = policy or self._load_policy(policy_path)
        self.metrics = CostMetrics(run_id=run_id)
        self._lock = threading.RLock()

        # Rate limiting state
        self._minute_window_start: float = time.time()
        self._tokens_this_minute: int = 0

        logger.info(
            f"F-209: CostController initialized for run {run_id}. "
            f"Budget: ${self.policy.max_budget_usd_per_run}/run, "
            f"${self.policy.max_budget_usd_per_sprint}/sprint"
        )

    def _load_policy(self, policy_path: Optional[Path] = None) -> CostPolicy:
        """Load cost policy from YAML file.

        Args:
            policy_path: Path to policy file.

        Returns:
            CostPolicy loaded from file or defaults.
        """
        path = policy_path or DEFAULT_COST_POLICY

        if path.exists():
            try:
                data = yaml.safe_load(path.read_text())
                policy = CostPolicy.from_dict(data)
                logger.info(f"F-209: Loaded cost policy from {path}")
                return policy
            except Exception as e:
                logger.warning(f"F-209: Failed to load cost policy: {e}. Using defaults.")

        return CostPolicy()

    def check_budget(self, sprint_id: Optional[str] = None) -> tuple[bool, str]:
        """Check if budget allows more spending.

        Args:
            sprint_id: Sprint to check budget for.

        Returns:
            Tuple of (allowed, message).
        """
        if not self.policy.enforce_budget:
            return True, "Budget enforcement disabled"

        with self._lock:
            # Check run budget
            if self.metrics.total_cost_usd >= self.policy.max_budget_usd_per_run:
                return False, (
                    f"Run budget exceeded: ${self.metrics.total_cost_usd:.4f} "
                    f">= ${self.policy.max_budget_usd_per_run}"
                )

            # Check sprint budget
            if sprint_id:
                sprint_cost = self.metrics.by_sprint.get(sprint_id, 0.0)
                if sprint_cost >= self.policy.max_budget_usd_per_sprint:
                    return False, (
                        f"Sprint budget exceeded: ${sprint_cost:.4f} "
                        f">= ${self.policy.max_budget_usd_per_sprint}"
                    )

            return True, "Budget OK"

    def check_rate_limit(self, tokens_requested: int) -> tuple[bool, str]:
        """Check if rate limit allows more tokens.

        Args:
            tokens_requested: Number of tokens to request.

        Returns:
            Tuple of (allowed, message).
        """
        if not self.policy.enforce_rate_limit:
            return True, "Rate limiting disabled"

        with self._lock:
            now = time.time()

            # Reset minute window if expired
            if now - self._minute_window_start >= 60:
                self._minute_window_start = now
                self._tokens_this_minute = 0

            # Check if request would exceed limit
            projected = self._tokens_this_minute + tokens_requested
            if projected > self.policy.rate_limit_tokens_per_minute:
                wait_seconds = 60 - (now - self._minute_window_start)
                return False, (
                    f"Rate limit exceeded: {projected} tokens would exceed "
                    f"{self.policy.rate_limit_tokens_per_minute}/min. "
                    f"Wait {wait_seconds:.1f}s"
                )

            return True, "Rate limit OK"

    def record_usage(
        self,
        agent_id: str,
        sprint_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        model: str = "claude-sonnet-3.5",
    ) -> float:
        """Record token usage and calculate cost.

        Args:
            agent_id: Agent that made the request.
            sprint_id: Sprint the request was for.
            prompt_tokens: Number of input tokens.
            completion_tokens: Number of output tokens.
            model: Model used.

        Returns:
            Cost in USD for this usage.

        Raises:
            CostLimitExceeded: If budget is exceeded.
            RateLimitExceeded: If rate limit is exceeded.
        """
        total_tokens = prompt_tokens + completion_tokens

        # Check rate limit first
        allowed, msg = self.check_rate_limit(total_tokens)
        if not allowed:
            raise RateLimitExceeded(msg, self._tokens_this_minute, self.policy.rate_limit_tokens_per_minute)

        with self._lock:
            # Calculate cost
            cost = (
                prompt_tokens * self.policy.cost_per_prompt_token +
                completion_tokens * self.policy.cost_per_completion_token
            )

            # Update rate limiting state
            self._tokens_this_minute += total_tokens

            # Update metrics
            self.metrics.total_cost_usd += cost
            self.metrics.total_prompt_tokens += prompt_tokens
            self.metrics.total_completion_tokens += completion_tokens

            self.metrics.by_sprint[sprint_id] = (
                self.metrics.by_sprint.get(sprint_id, 0.0) + cost
            )
            self.metrics.by_agent[agent_id] = (
                self.metrics.by_agent.get(agent_id, 0.0) + cost
            )

            # Record usage
            self.metrics.usage_records.append(UsageRecord(
                timestamp=datetime.now(timezone.utc),
                agent_id=agent_id,
                sprint_id=sprint_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost,
                model=model,
            ))

            # Check warning threshold
            self._check_warning_threshold()

            # Check if we've exceeded budget AFTER recording
            if self.policy.enforce_budget:
                # Check run budget
                if self.metrics.total_cost_usd > self.policy.max_budget_usd_per_run:
                    raise CostLimitExceeded(
                        f"Run budget exceeded after this request",
                        self.metrics.total_cost_usd,
                        self.policy.max_budget_usd_per_run,
                    )

                # Check sprint budget
                sprint_cost = self.metrics.by_sprint.get(sprint_id, 0.0)
                if sprint_cost > self.policy.max_budget_usd_per_sprint:
                    raise CostLimitExceeded(
                        f"Sprint {sprint_id} budget exceeded",
                        sprint_cost,
                        self.policy.max_budget_usd_per_sprint,
                    )

            logger.debug(
                f"F-209: Recorded usage for {agent_id}/{sprint_id}: "
                f"{prompt_tokens}+{completion_tokens} tokens = ${cost:.6f}"
            )

            return cost

    def _check_warning_threshold(self) -> None:
        """Check if we should warn about approaching budget limit."""
        threshold = self.policy.max_budget_usd_per_run * (self.policy.warning_threshold_percent / 100.0)

        if self.metrics.total_cost_usd >= threshold:
            remaining = self.policy.max_budget_usd_per_run - self.metrics.total_cost_usd
            logger.warning(
                f"F-209 WARNING: Run budget at {self.metrics.total_cost_usd:.4f}/"
                f"{self.policy.max_budget_usd_per_run} USD. "
                f"Only ${remaining:.4f} remaining!"
            )

    def get_metrics(self) -> CostMetrics:
        """Get current cost metrics."""
        with self._lock:
            return self.metrics

    def save_metrics(self, run_dir: Path) -> Optional[Path]:
        """Save metrics to run directory.

        Args:
            run_dir: Pipeline run directory.

        Returns:
            Path to saved metrics file, or None on failure.
        """
        output_dir = run_dir / "state"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / "cost_metrics.json"

        try:
            import json
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(self.metrics.to_dict(), f, indent=2)

            logger.info(f"F-209: Saved cost metrics to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"F-209: Failed to save cost metrics: {e}")
            return None


# Global controller instance
_controller: Optional[CostController] = None
_controller_lock = threading.Lock()


def get_cost_controller(
    run_id: Optional[str] = None,
    policy: Optional[CostPolicy] = None,
) -> CostController:
    """Get or create the global cost controller.

    Args:
        run_id: Run identifier (required on first call).
        policy: Cost policy (uses defaults if None).

    Returns:
        CostController instance.

    Raises:
        ValueError: If run_id not provided on first call.
    """
    global _controller

    with _controller_lock:
        if _controller is None:
            if run_id is None:
                run_id = "default"
            _controller = CostController(run_id, policy)
        return _controller


def reset_cost_controller() -> None:
    """Reset the global cost controller (for testing)."""
    global _controller
    with _controller_lock:
        _controller = None


# Convenience functions
def record_llm_usage(
    agent_id: str,
    sprint_id: str,
    prompt_tokens: int,
    completion_tokens: int,
    model: str = "claude-sonnet-3.5",
) -> float:
    """Record LLM usage and return cost."""
    return get_cost_controller().record_usage(
        agent_id, sprint_id, prompt_tokens, completion_tokens, model
    )


def check_budget_available(sprint_id: Optional[str] = None) -> bool:
    """Check if budget is available."""
    allowed, _ = get_cost_controller().check_budget(sprint_id)
    return allowed
