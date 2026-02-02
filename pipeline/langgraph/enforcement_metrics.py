"""Enforcement Metrics for Pipeline V2 Monitoring.

This module provides comprehensive metrics tracking for enforcement operations
using the existing Langfuse integration for observability.

Metrics Tracked:
    - guardrail_violations_total: Counter by violation type
    - invariant_checks_total: Counter by invariant I1-I11
    - trust_denials_total: Counter for trust boundary violations
    - rework_attempts_total: Counter for rework tasks
    - enforcement_latency_seconds: Histogram of enforcement check durations

Architecture:
    EnforcementMetrics
           |
           v
    +------------------+
    | LangfuseClient   |  <- Recording via scores and events
    +------------------+
           |
           v
    +------------------+
    | In-Memory Cache  |  <- Fast aggregation for local queries
    +------------------+

Usage:
    from pipeline.langgraph.enforcement_metrics import (
        record_violation,
        record_invariant_check,
        record_trust_denial,
        record_rework_attempt,
        record_enforcement_latency,
        get_metrics_summary,
    )

    # Record a guardrail violation
    record_violation("security", {"message": "Injection detected"})

    # Record an invariant check
    record_invariant_check("I1", passed=True)

    # Record a trust denial
    record_trust_denial("WORKER", "credentials_vault")

    # Record a rework attempt
    record_rework_attempt("S00", attempt_number=2)

    # Record enforcement latency
    record_enforcement_latency("gate_check", duration_ms=150.5)

Author: Pipeline Autonomo Team
Version: 1.0.0 (2026-01-16)
"""

from __future__ import annotations

import logging
import time
import threading
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, Generator, List, Optional, TypeVar

logger = logging.getLogger(__name__)

# Type variables for generic decorators
F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# ENUMS
# =============================================================================


class ViolationType(str, Enum):
    """Types of guardrail violations for metrics tracking."""

    SECURITY = "security"
    INVARIANT = "invariant"
    TRUST = "trust"
    CONTENT = "content"
    VALIDATION = "validation"
    GATE = "gate"
    TIMEOUT = "timeout"
    RESOURCE = "resource"
    CONSISTENCY = "consistency"


class InvariantCode(str, Enum):
    """Invariant codes I1-I11 for metrics tracking."""

    I1_NAMESPACING = "I1"
    I2_IDEMPOTENCY = "I2"
    I3_PHASE_ORDER = "I3"
    I4_GATES_BEFORE_SIGNOFF = "I4"
    I5_EXECUTIVE_VERIFICATION = "I5"
    I6_TRUTHFULNESS = "I6"
    I7_AUDIT_TRAIL = "I7"
    I8_EVENT_SCHEMA = "I8"
    I9_SAFE_HALT_PRIORITY = "I9"
    I10_REDIS_CANONICAL = "I10"
    I11_RUNAWAY_PROTECTION = "I11"


class CheckType(str, Enum):
    """Types of enforcement checks for latency tracking."""

    GATE_CHECK = "gate_check"
    INVARIANT_CHECK = "invariant_check"
    TRUST_CHECK = "trust_check"
    SECURITY_CHECK = "security_check"
    VALIDATION_CHECK = "validation_check"
    CONTENT_FILTER = "content_filter"
    FULL_ENFORCEMENT = "full_enforcement"


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class MetricEntry:
    """A single metric entry with timestamp and metadata."""

    name: str
    value: float
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LatencyBucket:
    """Histogram bucket for latency metrics."""

    le: float  # Less than or equal to (bucket boundary in ms)
    count: int = 0


@dataclass
class LatencyHistogram:
    """Histogram for tracking latency distributions."""

    # Standard buckets in milliseconds: 10ms, 50ms, 100ms, 250ms, 500ms, 1s, 5s, 10s
    buckets: List[LatencyBucket] = field(
        default_factory=lambda: [
            LatencyBucket(le=10.0),
            LatencyBucket(le=50.0),
            LatencyBucket(le=100.0),
            LatencyBucket(le=250.0),
            LatencyBucket(le=500.0),
            LatencyBucket(le=1000.0),
            LatencyBucket(le=5000.0),
            LatencyBucket(le=10000.0),
            LatencyBucket(le=float("inf")),  # +Inf bucket
        ]
    )
    sum_value: float = 0.0
    count: int = 0

    def observe(self, value_ms: float) -> None:
        """Record an observation in the histogram."""
        self.sum_value += value_ms
        self.count += 1
        for bucket in self.buckets:
            if value_ms <= bucket.le:
                bucket.count += 1

    def get_percentile(self, percentile: float) -> float:
        """Calculate approximate percentile from histogram."""
        if self.count == 0:
            return 0.0

        target_count = self.count * percentile
        prev_count = 0

        for bucket in self.buckets:
            if bucket.count >= target_count:
                # Linear interpolation within bucket
                if bucket.count == prev_count:
                    return bucket.le
                fraction = (target_count - prev_count) / (bucket.count - prev_count)
                prev_le = self.buckets[self.buckets.index(bucket) - 1].le if self.buckets.index(bucket) > 0 else 0
                return prev_le + fraction * (bucket.le - prev_le)
            prev_count = bucket.count

        return self.buckets[-1].le

    def get_average(self) -> float:
        """Get average latency."""
        if self.count == 0:
            return 0.0
        return self.sum_value / self.count


@dataclass
class MetricsSummary:
    """Summary of all enforcement metrics."""

    # Counters
    guardrail_violations_total: Dict[str, int] = field(default_factory=dict)
    invariant_checks_total: Dict[str, Dict[str, int]] = field(default_factory=dict)
    trust_denials_total: int = 0
    rework_attempts_total: int = 0

    # Histograms
    enforcement_latency: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Metadata
    collection_started_at: str = ""
    last_updated_at: str = ""
    total_events: int = 0


# =============================================================================
# ENFORCEMENT METRICS CLASS
# =============================================================================


class EnforcementMetrics:
    """Comprehensive metrics tracking for enforcement operations.

    This class provides:
    - Counter metrics for violations, invariant checks, trust denials, reworks
    - Histogram metrics for latency tracking
    - Integration with Langfuse for observability
    - Thread-safe operations for concurrent access

    Usage:
        metrics = EnforcementMetrics()

        # Record violation
        metrics.record_violation("security", {"message": "Attack detected"})

        # Record invariant check
        metrics.record_invariant_check("I1", passed=True)

        # Get summary
        summary = metrics.get_summary()
    """

    # Bounded cache limits to prevent unbounded memory growth
    MAX_METRIC_ENTRIES = 10000
    MAX_LATENCY_ENTRIES_PER_TYPE = 1000

    def __init__(
        self,
        langfuse_enabled: bool = True,
        run_id: Optional[str] = None,
        sprint_id: Optional[str] = None,
    ):
        """Initialize enforcement metrics.

        Args:
            langfuse_enabled: Whether to send metrics to Langfuse
            run_id: Current run ID for context
            sprint_id: Current sprint ID for context
        """
        self.langfuse_enabled = langfuse_enabled
        self.run_id = run_id
        self.sprint_id = sprint_id

        # Thread safety
        self._lock = threading.RLock()

        # Collection start time
        self._started_at = datetime.now(timezone.utc).isoformat()
        self._last_updated_at = self._started_at

        # Counter metrics (thread-safe with lock)
        self._violation_counts: Dict[str, int] = defaultdict(int)
        self._invariant_counts: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"passed": 0, "failed": 0}
        )
        self._trust_denial_count: int = 0
        self._rework_attempt_count: int = 0

        # Histogram metrics for latency
        self._latency_histograms: Dict[str, LatencyHistogram] = defaultdict(
            LatencyHistogram
        )

        # Recent entries for debugging (bounded)
        self._recent_entries: List[MetricEntry] = []

        # Total event count
        self._total_events: int = 0

        # Langfuse client (lazy loaded)
        self._langfuse_client = None
        self._current_trace_id: Optional[str] = None

        logger.debug(f"EnforcementMetrics initialized (langfuse_enabled={langfuse_enabled})")

    def _get_langfuse_client(self):
        """Lazy load Langfuse client."""
        if self._langfuse_client is None and self.langfuse_enabled:
            try:
                from pipeline.langfuse_client import get_langfuse_client
                self._langfuse_client = get_langfuse_client()
            except ImportError:
                logger.debug("Langfuse client not available")
                self.langfuse_enabled = False
            except Exception as e:
                logger.warning(f"Failed to initialize Langfuse client: {e}")
                self.langfuse_enabled = False
        return self._langfuse_client

    def _prune_entries_if_needed(self) -> None:
        """Prune recent entries if they exceed the limit.

        Called internally - caller must hold the lock.
        """
        if len(self._recent_entries) > self.MAX_METRIC_ENTRIES:
            # Keep the most recent half
            self._recent_entries = self._recent_entries[-(self.MAX_METRIC_ENTRIES // 2):]
            logger.debug(f"Pruned metric entries to {len(self._recent_entries)}")

    def _update_timestamp(self) -> None:
        """Update the last updated timestamp."""
        self._last_updated_at = datetime.now(timezone.utc).isoformat()

    def _send_to_langfuse(
        self,
        metric_name: str,
        value: float,
        labels: Dict[str, str],
        metadata: Dict[str, Any],
    ) -> None:
        """Send metric to Langfuse as a score or event.

        Args:
            metric_name: Name of the metric
            value: Numeric value
            labels: Label key-value pairs
            metadata: Additional metadata
        """
        client = self._get_langfuse_client()
        if not client or not client.is_enabled():
            return

        try:
            # Create a trace for this metric if we don't have one
            if not self._current_trace_id:
                trace = client.create_trace(
                    name="enforcement_metrics",
                    metadata={
                        "run_id": self.run_id,
                        "sprint_id": self.sprint_id,
                    },
                    tags=["metrics", "enforcement"],
                )
                self._current_trace_id = trace.trace_id

            # Create a span for this metric
            span = client.create_span(
                trace_id=self._current_trace_id,
                name=f"metric:{metric_name}",
                input_data={
                    "value": value,
                    "labels": labels,
                },
                metadata=metadata,
            )

            # End the span immediately
            client.end_span(span.span_id)

        except Exception as e:
            logger.debug(f"Failed to send metric to Langfuse: {e}")

    # =========================================================================
    # VIOLATION METRICS
    # =========================================================================

    def record_violation(
        self,
        violation_type: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a guardrail violation.

        Args:
            violation_type: Type of violation (e.g., "security", "invariant")
            details: Additional details about the violation
        """
        details = details or {}

        with self._lock:
            # Increment counter
            self._violation_counts[violation_type] += 1
            self._total_events += 1
            self._update_timestamp()

            # Create entry
            entry = MetricEntry(
                name="guardrail_violations_total",
                value=1.0,
                labels={"type": violation_type},
                metadata={
                    "run_id": self.run_id,
                    "sprint_id": self.sprint_id,
                    "details": details,
                },
            )
            self._recent_entries.append(entry)
            self._prune_entries_if_needed()

        # Send to Langfuse
        self._send_to_langfuse(
            metric_name="guardrail_violations_total",
            value=1.0,
            labels={"type": violation_type},
            metadata={
                "run_id": self.run_id,
                "sprint_id": self.sprint_id,
                **details,
            },
        )

        logger.info(f"Recorded violation: type={violation_type}")

    def get_violation_count(self, violation_type: Optional[str] = None) -> int:
        """Get violation count by type or total.

        Args:
            violation_type: Specific type or None for total

        Returns:
            Violation count
        """
        with self._lock:
            if violation_type:
                return self._violation_counts.get(violation_type, 0)
            return sum(self._violation_counts.values())

    # =========================================================================
    # INVARIANT CHECK METRICS
    # =========================================================================

    def record_invariant_check(
        self,
        invariant_code: str,
        passed: bool,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an invariant check result.

        Args:
            invariant_code: Invariant code (I1-I11)
            passed: Whether the check passed
            details: Additional details
        """
        details = details or {}
        status = "passed" if passed else "failed"

        with self._lock:
            # Increment counter
            self._invariant_counts[invariant_code][status] += 1
            self._total_events += 1
            self._update_timestamp()

            # Create entry
            entry = MetricEntry(
                name="invariant_checks_total",
                value=1.0,
                labels={
                    "invariant": invariant_code,
                    "status": status,
                },
                metadata={
                    "run_id": self.run_id,
                    "sprint_id": self.sprint_id,
                    "details": details,
                },
            )
            self._recent_entries.append(entry)
            self._prune_entries_if_needed()

        # Send to Langfuse
        self._send_to_langfuse(
            metric_name="invariant_checks_total",
            value=1.0 if passed else 0.0,
            labels={
                "invariant": invariant_code,
                "status": status,
            },
            metadata={
                "run_id": self.run_id,
                "sprint_id": self.sprint_id,
                "passed": passed,
                **details,
            },
        )

        logger.debug(f"Recorded invariant check: {invariant_code}={status}")

    def get_invariant_counts(
        self,
        invariant_code: Optional[str] = None,
    ) -> Dict[str, Dict[str, int]]:
        """Get invariant check counts.

        Args:
            invariant_code: Specific code or None for all

        Returns:
            Dict of invariant code -> {passed: N, failed: M}
        """
        with self._lock:
            if invariant_code:
                return {invariant_code: dict(self._invariant_counts[invariant_code])}
            return {k: dict(v) for k, v in self._invariant_counts.items()}

    # =========================================================================
    # TRUST DENIAL METRICS
    # =========================================================================

    def record_trust_denial(
        self,
        agent_role: str,
        resource: str,
        reason: Optional[str] = None,
    ) -> None:
        """Record a trust boundary denial.

        Args:
            agent_role: Role of the agent that was denied
            resource: Resource that was requested
            reason: Reason for denial
        """
        with self._lock:
            # Increment counter
            self._trust_denial_count += 1
            self._total_events += 1
            self._update_timestamp()

            # Create entry
            entry = MetricEntry(
                name="trust_denials_total",
                value=1.0,
                labels={
                    "agent_role": agent_role,
                    "resource": resource,
                },
                metadata={
                    "run_id": self.run_id,
                    "sprint_id": self.sprint_id,
                    "reason": reason,
                },
            )
            self._recent_entries.append(entry)
            self._prune_entries_if_needed()

        # Send to Langfuse
        self._send_to_langfuse(
            metric_name="trust_denials_total",
            value=1.0,
            labels={
                "agent_role": agent_role,
                "resource": resource,
            },
            metadata={
                "run_id": self.run_id,
                "sprint_id": self.sprint_id,
                "reason": reason,
            },
        )

        logger.warning(f"Recorded trust denial: agent={agent_role}, resource={resource}")

    def get_trust_denial_count(self) -> int:
        """Get total trust denial count."""
        with self._lock:
            return self._trust_denial_count

    # =========================================================================
    # REWORK ATTEMPT METRICS
    # =========================================================================

    def record_rework_attempt(
        self,
        sprint_id: str,
        attempt_number: int,
        task_id: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> None:
        """Record a rework attempt.

        Args:
            sprint_id: Sprint ID where rework occurred
            attempt_number: Attempt number (1, 2, 3, ...)
            task_id: Optional task ID
            reason: Reason for rework
        """
        with self._lock:
            # Increment counter
            self._rework_attempt_count += 1
            self._total_events += 1
            self._update_timestamp()

            # Create entry
            entry = MetricEntry(
                name="rework_attempts_total",
                value=float(attempt_number),
                labels={
                    "sprint_id": sprint_id,
                },
                metadata={
                    "run_id": self.run_id,
                    "task_id": task_id,
                    "attempt_number": attempt_number,
                    "reason": reason,
                },
            )
            self._recent_entries.append(entry)
            self._prune_entries_if_needed()

        # Send to Langfuse
        self._send_to_langfuse(
            metric_name="rework_attempts_total",
            value=float(attempt_number),
            labels={
                "sprint_id": sprint_id,
            },
            metadata={
                "run_id": self.run_id,
                "task_id": task_id,
                "attempt_number": attempt_number,
                "reason": reason,
            },
        )

        logger.info(f"Recorded rework attempt: sprint={sprint_id}, attempt={attempt_number}")

    def get_rework_attempt_count(self) -> int:
        """Get total rework attempt count."""
        with self._lock:
            return self._rework_attempt_count

    # =========================================================================
    # LATENCY METRICS
    # =========================================================================

    def record_enforcement_latency(
        self,
        check_type: str,
        duration_ms: float,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record enforcement check latency.

        Args:
            check_type: Type of enforcement check
            duration_ms: Duration in milliseconds
            details: Additional details
        """
        details = details or {}

        with self._lock:
            # Add to histogram
            self._latency_histograms[check_type].observe(duration_ms)
            self._total_events += 1
            self._update_timestamp()

            # Create entry
            entry = MetricEntry(
                name="enforcement_latency_seconds",
                value=duration_ms / 1000.0,  # Convert to seconds for Prometheus compatibility
                labels={
                    "check_type": check_type,
                },
                metadata={
                    "run_id": self.run_id,
                    "sprint_id": self.sprint_id,
                    "duration_ms": duration_ms,
                    **details,
                },
            )
            self._recent_entries.append(entry)
            self._prune_entries_if_needed()

        # Send to Langfuse
        self._send_to_langfuse(
            metric_name="enforcement_latency_seconds",
            value=duration_ms / 1000.0,
            labels={
                "check_type": check_type,
            },
            metadata={
                "run_id": self.run_id,
                "sprint_id": self.sprint_id,
                "duration_ms": duration_ms,
                **details,
            },
        )

        logger.debug(f"Recorded latency: type={check_type}, duration={duration_ms:.2f}ms")

    def get_latency_stats(
        self,
        check_type: Optional[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Get latency statistics.

        Args:
            check_type: Specific type or None for all

        Returns:
            Dict of check_type -> {avg, p50, p95, p99, count}
        """
        with self._lock:
            result = {}

            histograms = (
                {check_type: self._latency_histograms[check_type]}
                if check_type
                else dict(self._latency_histograms)
            )

            for ct, hist in histograms.items():
                result[ct] = {
                    "avg_ms": hist.get_average(),
                    "p50_ms": hist.get_percentile(0.5),
                    "p95_ms": hist.get_percentile(0.95),
                    "p99_ms": hist.get_percentile(0.99),
                    "count": hist.count,
                    "sum_ms": hist.sum_value,
                }

            return result

    # =========================================================================
    # SUMMARY AND EXPORT
    # =========================================================================

    def get_summary(self) -> MetricsSummary:
        """Get a summary of all metrics.

        Returns:
            MetricsSummary with all collected metrics
        """
        with self._lock:
            return MetricsSummary(
                guardrail_violations_total=dict(self._violation_counts),
                invariant_checks_total={
                    k: dict(v) for k, v in self._invariant_counts.items()
                },
                trust_denials_total=self._trust_denial_count,
                rework_attempts_total=self._rework_attempt_count,
                enforcement_latency=self.get_latency_stats(),
                collection_started_at=self._started_at,
                last_updated_at=self._last_updated_at,
                total_events=self._total_events,
            )

    def get_prometheus_format(self) -> str:
        """Export metrics in Prometheus format.

        Returns:
            Prometheus-compatible metrics string
        """
        lines = []

        with self._lock:
            # Violation counters
            lines.append("# HELP guardrail_violations_total Total guardrail violations by type")
            lines.append("# TYPE guardrail_violations_total counter")
            for vtype, count in self._violation_counts.items():
                lines.append(f'guardrail_violations_total{{type="{vtype}"}} {count}')

            # Invariant check counters
            lines.append("")
            lines.append("# HELP invariant_checks_total Total invariant checks by code and status")
            lines.append("# TYPE invariant_checks_total counter")
            for code, counts in self._invariant_counts.items():
                for status, count in counts.items():
                    lines.append(f'invariant_checks_total{{invariant="{code}",status="{status}"}} {count}')

            # Trust denials counter
            lines.append("")
            lines.append("# HELP trust_denials_total Total trust boundary denials")
            lines.append("# TYPE trust_denials_total counter")
            lines.append(f"trust_denials_total {self._trust_denial_count}")

            # Rework attempts counter
            lines.append("")
            lines.append("# HELP rework_attempts_total Total rework attempts")
            lines.append("# TYPE rework_attempts_total counter")
            lines.append(f"rework_attempts_total {self._rework_attempt_count}")

            # Latency histograms
            lines.append("")
            lines.append("# HELP enforcement_latency_seconds Enforcement check latency in seconds")
            lines.append("# TYPE enforcement_latency_seconds histogram")
            for check_type, hist in self._latency_histograms.items():
                for bucket in hist.buckets:
                    le = "+Inf" if bucket.le == float("inf") else f"{bucket.le / 1000.0}"
                    lines.append(
                        f'enforcement_latency_seconds_bucket{{check_type="{check_type}",le="{le}"}} {bucket.count}'
                    )
                lines.append(
                    f'enforcement_latency_seconds_sum{{check_type="{check_type}"}} {hist.sum_value / 1000.0}'
                )
                lines.append(
                    f'enforcement_latency_seconds_count{{check_type="{check_type}"}} {hist.count}'
                )

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all metrics (for testing)."""
        with self._lock:
            self._violation_counts.clear()
            self._invariant_counts.clear()
            self._trust_denial_count = 0
            self._rework_attempt_count = 0
            self._latency_histograms.clear()
            self._recent_entries.clear()
            self._total_events = 0
            self._started_at = datetime.now(timezone.utc).isoformat()
            self._update_timestamp()

        logger.info("Metrics reset")

    def flush_to_langfuse(self) -> None:
        """Flush any pending metrics to Langfuse."""
        client = self._get_langfuse_client()
        if client and client.is_available():
            try:
                from pipeline.langfuse_tracer import flush
                flush()
                logger.debug("Flushed metrics to Langfuse")
            except Exception as e:
                logger.debug(f"Failed to flush to Langfuse: {e}")


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================


_metrics_instance: Optional[EnforcementMetrics] = None
_metrics_lock = threading.Lock()


def get_metrics(
    run_id: Optional[str] = None,
    sprint_id: Optional[str] = None,
    langfuse_enabled: bool = True,
) -> EnforcementMetrics:
    """Get the singleton metrics instance.

    Args:
        run_id: Current run ID (only used on first call)
        sprint_id: Current sprint ID (only used on first call)
        langfuse_enabled: Whether to enable Langfuse (only used on first call)

    Returns:
        EnforcementMetrics singleton instance
    """
    global _metrics_instance

    with _metrics_lock:
        if _metrics_instance is None:
            _metrics_instance = EnforcementMetrics(
                langfuse_enabled=langfuse_enabled,
                run_id=run_id,
                sprint_id=sprint_id,
            )
        return _metrics_instance


def reset_metrics() -> None:
    """Reset the singleton metrics instance (for testing)."""
    global _metrics_instance
    with _metrics_lock:
        _metrics_instance = None


# =============================================================================
# CONVENIENCE FUNCTIONS (MODULE-LEVEL API)
# =============================================================================


def record_violation(
    violation_type: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Record a guardrail violation.

    Args:
        violation_type: Type of violation (e.g., "security", "invariant")
        details: Additional details about the violation
    """
    get_metrics().record_violation(violation_type, details)


def record_invariant_check(
    invariant_code: str,
    passed: bool,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Record an invariant check result.

    Args:
        invariant_code: Invariant code (I1-I11)
        passed: Whether the check passed
        details: Additional details
    """
    get_metrics().record_invariant_check(invariant_code, passed, details)


def record_trust_denial(
    agent_role: str,
    resource: str,
    reason: Optional[str] = None,
) -> None:
    """Record a trust boundary denial.

    Args:
        agent_role: Role of the agent that was denied
        resource: Resource that was requested
        reason: Reason for denial
    """
    get_metrics().record_trust_denial(agent_role, resource, reason)


def record_rework_attempt(
    sprint_id: str,
    attempt_number: int,
    task_id: Optional[str] = None,
    reason: Optional[str] = None,
) -> None:
    """Record a rework attempt.

    Args:
        sprint_id: Sprint ID where rework occurred
        attempt_number: Attempt number (1, 2, 3, ...)
        task_id: Optional task ID
        reason: Reason for rework
    """
    get_metrics().record_rework_attempt(sprint_id, attempt_number, task_id, reason)


def record_enforcement_latency(
    check_type: str,
    duration_ms: float,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Record enforcement check latency.

    Args:
        check_type: Type of enforcement check
        duration_ms: Duration in milliseconds
        details: Additional details
    """
    get_metrics().record_enforcement_latency(check_type, duration_ms, details)


def get_metrics_summary() -> MetricsSummary:
    """Get a summary of all metrics.

    Returns:
        MetricsSummary with all collected metrics
    """
    return get_metrics().get_summary()


def get_prometheus_metrics() -> str:
    """Get metrics in Prometheus format.

    Returns:
        Prometheus-compatible metrics string
    """
    return get_metrics().get_prometheus_format()


# =============================================================================
# DECORATORS FOR AUTOMATIC LATENCY TRACKING
# =============================================================================


def track_enforcement_latency(check_type: str) -> Callable[[F], F]:
    """Decorator to automatically track enforcement latency.

    Args:
        check_type: Type of enforcement check

    Returns:
        Decorated function

    Usage:
        @track_enforcement_latency("gate_check")
        def run_gate_check(state):
            # ... implementation
            return result
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration_ms = (time.perf_counter() - start_time) * 1000
                record_enforcement_latency(check_type, duration_ms)

        return wrapper  # type: ignore

    return decorator


@contextmanager
def measure_enforcement_latency(
    check_type: str,
    details: Optional[Dict[str, Any]] = None,
) -> Generator[Dict[str, Any], None, None]:
    """Context manager for measuring enforcement latency.

    Args:
        check_type: Type of enforcement check
        details: Additional details

    Yields:
        Dict for capturing additional metadata

    Usage:
        with measure_enforcement_latency("gate_check") as ctx:
            # ... do work
            ctx["gate_id"] = "G0"
    """
    ctx: Dict[str, Any] = {"start_time": time.perf_counter()}
    if details:
        ctx.update(details)

    try:
        yield ctx
    finally:
        duration_ms = (time.perf_counter() - ctx["start_time"]) * 1000
        record_enforcement_latency(check_type, duration_ms, ctx)


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Main class
    "EnforcementMetrics",
    # Data classes
    "MetricEntry",
    "LatencyBucket",
    "LatencyHistogram",
    "MetricsSummary",
    # Enums
    "ViolationType",
    "InvariantCode",
    "CheckType",
    # Singleton accessor
    "get_metrics",
    "reset_metrics",
    # Helper functions
    "record_violation",
    "record_invariant_check",
    "record_trust_denial",
    "record_rework_attempt",
    "record_enforcement_latency",
    "get_metrics_summary",
    "get_prometheus_metrics",
    # Decorators
    "track_enforcement_latency",
    "measure_enforcement_latency",
]
