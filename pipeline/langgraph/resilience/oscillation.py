"""Oscillation Detection for Pipeline Resilience.

Detects and breaks cyclic failure patterns in pipeline execution.
When agents repeatedly reject/rework in patterns (A-B-A-B or A-B-C-A-B-C),
this module detects and breaks the cycle.

Patterns Detected:
    - ABAB: Alternating between 2 states (most common)
    - CYCLE: Repeating cycle of N states (A-B-C-A-B-C)
    - RUNAWAY: Same failure repeated many times (immediate human escalation)

Usage:
    from oscillation import detect_oscillation, break_oscillation

    history = get_rework_history()
    detection = detect_oscillation(history)
    if detection.detected:
        state = break_oscillation(state, detection)
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class OscillationPattern(str, Enum):
    """Types of oscillation patterns."""

    ABAB = "abab"
    """Alternating between 2 states (A-B-A-B)."""

    CYCLE = "cycle"
    """Repeating cycle of N states (A-B-C-A-B-C)."""

    RUNAWAY = "runaway"
    """Same failure repeated many times."""

    NONE = "none"
    """No pattern detected."""


@dataclass
class ReworkHistoryEntry:
    """Entry in rework history for oscillation detection."""

    timestamp: datetime
    rejecting_agent: str
    failed_items: List[str]
    work_type: str
    attempt: int
    reason: str = ""
    tier: str = "L5"

    def signature(self) -> str:
        """Generate signature for pattern matching.

        Returns:
            String signature combining agent, work type, and items.
        """
        items_str = ",".join(sorted(self.failed_items))
        return f"{self.rejecting_agent}:{self.work_type}:{items_str}"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "rejecting_agent": self.rejecting_agent,
            "failed_items": self.failed_items,
            "work_type": self.work_type,
            "attempt": self.attempt,
            "reason": self.reason,
            "tier": self.tier,
            "signature": self.signature(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReworkHistoryEntry":
        """Create from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        elif timestamp is None:
            timestamp = datetime.now(timezone.utc)

        return cls(
            timestamp=timestamp,
            rejecting_agent=data.get("rejecting_agent", ""),
            failed_items=data.get("failed_items", []),
            work_type=data.get("work_type", ""),
            attempt=data.get("attempt", 0),
            reason=data.get("reason", ""),
            tier=data.get("tier", "L5"),
        )


@dataclass
class OscillationDetectionResult:
    """Result of oscillation detection."""

    detected: bool
    pattern: OscillationPattern
    cycle_length: int
    confidence: float  # 0.0 - 1.0
    recommendation: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "detected": self.detected,
            "pattern": self.pattern.value,
            "cycle_length": self.cycle_length,
            "confidence": round(self.confidence, 2),
            "recommendation": self.recommendation,
            "details": self.details,
        }


@dataclass
class OscillationBreakResult:
    """Result of oscillation break attempt."""

    action_taken: str
    state_modified: bool
    requires_human: bool
    human_reason: Optional[str] = None
    modifications: List[str] = field(default_factory=list)


def detect_oscillation(
    history: List[ReworkHistoryEntry],
    min_pattern_length: int = 2,
    min_repetitions: int = 2,
    runaway_threshold: int = 5,
) -> OscillationDetectionResult:
    """Detect oscillation patterns in rework history.

    Detection algorithms:
    1. RUNAWAY: Same failure repeated runaway_threshold times
    2. ABAB: Alternating between 2 states (A-B-A-B)
    3. CYCLE: Repeating cycle of N states (A-B-C-A-B-C)

    Args:
        history: List of rework history entries.
        min_pattern_length: Minimum pattern length to detect.
        min_repetitions: Minimum repetitions to confirm pattern.
        runaway_threshold: Consecutive same-failures to trigger RUNAWAY.

    Returns:
        OscillationDetectionResult with detection info.
    """
    min_required_cycle = min_pattern_length * min_repetitions

    # Check minimum history - runaway needs less than cycle/abab
    min_required = min(min_required_cycle, runaway_threshold)
    if len(history) < min_required:
        return OscillationDetectionResult(
            detected=False,
            pattern=OscillationPattern.NONE,
            cycle_length=0,
            confidence=0.0,
            recommendation="Insufficient history for detection",
            details={"history_length": len(history), "min_required": min_required},
        )

    # Extract signatures
    signatures = [entry.signature() for entry in history]
    recent = signatures[-10:]  # Last 10 entries

    # Check RUNAWAY (same signature repeated) - highest priority
    if len(recent) >= runaway_threshold:
        last_n = recent[-runaway_threshold:]
        if len(set(last_n)) == 1:
            logger.warning(
                "OSCILLATION_DETECTED: RUNAWAY pattern - same failure %d times",
                runaway_threshold,
            )
            return OscillationDetectionResult(
                detected=True,
                pattern=OscillationPattern.RUNAWAY,
                cycle_length=1,
                confidence=0.95,
                recommendation="Same failure repeating - escalate to human immediately",
                details={
                    "repeated_signature": last_n[0],
                    "repetition_count": runaway_threshold,
                },
            )

    # Check ABAB pattern (alternating between 2 states)
    if len(recent) >= 4:
        # Check if last 4 form A-B-A-B
        if (
            recent[-4] == recent[-2]
            and recent[-3] == recent[-1]
            and recent[-4] != recent[-3]
        ):
            # Additional check: count how many alternations
            alt_count = 1
            for i in range(len(recent) - 4, -1, -2):
                if i >= 1:
                    if recent[i] == recent[-4] and recent[i - 1] == recent[-3]:
                        alt_count += 1
                    else:
                        break

            confidence = min(0.9 + (alt_count - 2) * 0.02, 0.99)

            logger.warning(
                "OSCILLATION_DETECTED: ABAB pattern - alternating between 2 states"
            )
            return OscillationDetectionResult(
                detected=True,
                pattern=OscillationPattern.ABAB,
                cycle_length=2,
                confidence=confidence,
                recommendation="Alternating failures - inject noise or escalate",
                details={
                    "state_a": recent[-4],
                    "state_b": recent[-3],
                    "alternation_count": alt_count,
                },
            )

    # Check longer cycles (3-4 length)
    for cycle_len in [3, 4]:
        if len(recent) >= cycle_len * 2:
            cycle1 = recent[-cycle_len * 2 : -cycle_len]
            cycle2 = recent[-cycle_len:]
            if cycle1 == cycle2:
                # Additional check: look for more repetitions
                repetitions = 2
                for i in range(len(recent) - cycle_len * 2, -1, -cycle_len):
                    if i >= cycle_len:
                        prev_cycle = recent[i - cycle_len : i]
                        if prev_cycle == cycle2:
                            repetitions += 1
                        else:
                            break

                confidence = min(0.85 + (repetitions - 2) * 0.03, 0.97)

                logger.warning(
                    "OSCILLATION_DETECTED: CYCLE pattern - length %d, %d repetitions",
                    cycle_len,
                    repetitions,
                )
                return OscillationDetectionResult(
                    detected=True,
                    pattern=OscillationPattern.CYCLE,
                    cycle_length=cycle_len,
                    confidence=confidence,
                    recommendation=f"Cycle of length {cycle_len} detected - escalate tier",
                    details={
                        "cycle_states": cycle2,
                        "repetition_count": repetitions,
                    },
                )

    # No pattern detected
    return OscillationDetectionResult(
        detected=False,
        pattern=OscillationPattern.NONE,
        cycle_length=0,
        confidence=0.0,
        recommendation="No oscillation detected",
        details={"signatures_checked": len(recent)},
    )


def break_oscillation(
    state: Dict[str, Any],
    detection: OscillationDetectionResult,
    noise_magnitude: float = 0.1,
) -> OscillationBreakResult:
    """Attempt to break detected oscillation.

    Strategies by pattern:
    - RUNAWAY: Immediate human escalation
    - ABAB: Inject noise to break symmetry
    - CYCLE: Escalate to higher tier

    Args:
        state: Current pipeline state dictionary.
        detection: Oscillation detection result.
        noise_magnitude: Amount of noise to inject (0-1).

    Returns:
        OscillationBreakResult with action taken.
    """
    if not detection.detected:
        return OscillationBreakResult(
            action_taken="no_action",
            state_modified=False,
            requires_human=False,
            modifications=[],
        )

    # RUNAWAY: Immediate human escalation
    if detection.pattern == OscillationPattern.RUNAWAY:
        logger.warning(
            "OSCILLATION_BREAK: RUNAWAY detected - escalating to human"
        )

        state["_requires_human"] = True
        state["_human_reason"] = f"Runaway rework detected: {detection.recommendation}"
        state["_oscillation_break"] = {
            "pattern": detection.pattern.value,
            "action": "human_escalation",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return OscillationBreakResult(
            action_taken="human_escalation",
            state_modified=True,
            requires_human=True,
            human_reason=detection.recommendation,
            modifications=[
                "Set _requires_human=True",
                "Set _human_reason",
                "Added _oscillation_break metadata",
            ],
        )

    # ABAB: Inject noise
    if detection.pattern == OscillationPattern.ABAB:
        logger.warning("OSCILLATION_BREAK: ABAB pattern - injecting noise")

        rework_context = state.get("_rework_context", {})
        rework_context["noise_factor"] = random.random() * noise_magnitude
        rework_context["oscillation_break_attempt"] = True
        rework_context["break_timestamp"] = datetime.now(timezone.utc).isoformat()
        rework_context["pattern_broken"] = detection.pattern.value

        state["_rework_context"] = rework_context
        state["_oscillation_break"] = {
            "pattern": detection.pattern.value,
            "action": "noise_injection",
            "noise_factor": rework_context["noise_factor"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return OscillationBreakResult(
            action_taken="noise_injection",
            state_modified=True,
            requires_human=False,
            modifications=[
                f"Injected noise_factor={rework_context['noise_factor']:.3f}",
                "Set oscillation_break_attempt=True",
                "Added _oscillation_break metadata",
            ],
        )

    # CYCLE: Escalate tier
    if detection.pattern == OscillationPattern.CYCLE:
        logger.warning("OSCILLATION_BREAK: CYCLE pattern - escalating tier")

        rework_context = state.get("_rework_context", {})
        violation_report = rework_context.get("violation_report", {})
        current_tier = violation_report.get("delegate_tier", "L5")

        # Tier escalation order
        tier_order = ["L6", "L5", "L4", "L3", "L2", "L1", "L0"]
        modifications = []

        try:
            current_idx = tier_order.index(current_tier)
            if current_idx < len(tier_order) - 1:
                new_tier = tier_order[current_idx + 1]
                violation_report["delegate_tier"] = new_tier
                rework_context["violation_report"] = violation_report
                rework_context["escalated_due_to_oscillation"] = True
                state["_rework_context"] = rework_context

                modifications.append(f"Escalated tier from {current_tier} to {new_tier}")
                logger.info(
                    "OSCILLATION_BREAK: Escalated from %s to %s",
                    current_tier,
                    new_tier,
                )
            else:
                # Already at L0, escalate to human
                state["_requires_human"] = True
                state["_human_reason"] = "Oscillation at L0 - human intervention required"
                modifications.append("At L0 - escalated to human")

        except ValueError:
            # Unknown tier, escalate to human
            state["_requires_human"] = True
            state["_human_reason"] = f"Unknown tier '{current_tier}' - human intervention required"
            modifications.append(f"Unknown tier {current_tier} - escalated to human")

        state["_oscillation_break"] = {
            "pattern": detection.pattern.value,
            "action": "tier_escalation",
            "from_tier": current_tier,
            "cycle_length": detection.cycle_length,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        modifications.append("Added _oscillation_break metadata")

        return OscillationBreakResult(
            action_taken="tier_escalation",
            state_modified=True,
            requires_human=state.get("_requires_human", False),
            human_reason=state.get("_human_reason"),
            modifications=modifications,
        )

    # Unknown pattern (shouldn't happen)
    return OscillationBreakResult(
        action_taken="no_action",
        state_modified=False,
        requires_human=False,
        modifications=[],
    )


class OscillationTracker:
    """Tracks rework history and detects oscillations.

    Maintains a sliding window of rework entries and provides
    convenient methods for detection and breaking.

    Usage:
        tracker = OscillationTracker(max_history=20)
        tracker.add_entry(entry)
        if tracker.is_oscillating():
            state = tracker.attempt_break(state)
    """

    def __init__(
        self,
        max_history: int = 20,
        runaway_threshold: int = 5,
    ):
        """Initialize tracker.

        Args:
            max_history: Maximum entries to keep.
            runaway_threshold: Threshold for RUNAWAY detection.
        """
        self._history: List[ReworkHistoryEntry] = []
        self._max_history = max_history
        self._runaway_threshold = runaway_threshold
        self._break_attempts = 0

    @property
    def history(self) -> List[ReworkHistoryEntry]:
        """Get current history."""
        return list(self._history)

    @property
    def break_attempts(self) -> int:
        """Number of break attempts made."""
        return self._break_attempts

    def add_entry(self, entry: ReworkHistoryEntry) -> None:
        """Add entry to history.

        Maintains sliding window by removing oldest entries.
        """
        self._history.append(entry)
        while len(self._history) > self._max_history:
            self._history.pop(0)

    def add_rework(
        self,
        rejecting_agent: str,
        failed_items: List[str],
        work_type: str,
        attempt: int,
        reason: str = "",
        tier: str = "L5",
    ) -> None:
        """Convenience method to add rework entry."""
        entry = ReworkHistoryEntry(
            timestamp=datetime.now(timezone.utc),
            rejecting_agent=rejecting_agent,
            failed_items=failed_items,
            work_type=work_type,
            attempt=attempt,
            reason=reason,
            tier=tier,
        )
        self.add_entry(entry)

    def detect(self) -> OscillationDetectionResult:
        """Detect oscillation in current history."""
        return detect_oscillation(
            self._history,
            runaway_threshold=self._runaway_threshold,
        )

    def is_oscillating(self) -> bool:
        """Check if currently oscillating."""
        return self.detect().detected

    def attempt_break(
        self,
        state: Dict[str, Any],
    ) -> OscillationBreakResult:
        """Attempt to break current oscillation.

        Args:
            state: Pipeline state to modify.

        Returns:
            OscillationBreakResult.
        """
        detection = self.detect()
        if detection.detected:
            self._break_attempts += 1
            return break_oscillation(state, detection)

        return OscillationBreakResult(
            action_taken="no_action",
            state_modified=False,
            requires_human=False,
        )

    def clear(self) -> None:
        """Clear all history."""
        self._history.clear()
        self._break_attempts = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize tracker state."""
        return {
            "history": [e.to_dict() for e in self._history],
            "break_attempts": self._break_attempts,
            "max_history": self._max_history,
            "runaway_threshold": self._runaway_threshold,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OscillationTracker":
        """Create from dictionary."""
        tracker = cls(
            max_history=data.get("max_history", 20),
            runaway_threshold=data.get("runaway_threshold", 5),
        )
        tracker._break_attempts = data.get("break_attempts", 0)
        for entry_data in data.get("history", []):
            tracker.add_entry(ReworkHistoryEntry.from_dict(entry_data))
        return tracker


# Global tracker instance
_global_tracker: Optional[OscillationTracker] = None


def get_oscillation_tracker() -> OscillationTracker:
    """Get or create global oscillation tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = OscillationTracker()
    return _global_tracker


def reset_oscillation_tracker() -> None:
    """Reset global oscillation tracker."""
    global _global_tracker
    _global_tracker = None


__all__ = [
    "OscillationPattern",
    "ReworkHistoryEntry",
    "OscillationDetectionResult",
    "OscillationBreakResult",
    "detect_oscillation",
    "break_oscillation",
    "OscillationTracker",
    "get_oscillation_tracker",
    "reset_oscillation_tracker",
]
