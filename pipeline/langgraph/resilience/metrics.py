"""Metrics and Observability for Pipeline Resilience.

Provides metrics collection for monitoring resilience components:
- Retry attempts, successes, and failures
- Circuit breaker state transitions
- Oscillation detection events
- Run Master escalations
- File operation performance

Metrics are exposed in Prometheus format and can be integrated
with existing observability infrastructure.

Usage:
    from resilience.metrics import (
        record_retry_attempt,
        record_circuit_breaker_state,
        record_oscillation_event,
    )

    # Record a retry attempt
    record_retry_attempt("external_api", attempt=1, success=False)

    # Get metrics summary
    summary = get_metrics_summary()
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of resilience metrics."""

    RETRY_ATTEMPT = "retry_attempt"
    RETRY_SUCCESS = "retry_success"
    RETRY_FAILURE = "retry_failure"
    RETRY_EXHAUSTED = "retry_exhausted"
    CIRCUIT_BREAKER_OPEN = "circuit_breaker_open"
    CIRCUIT_BREAKER_CLOSE = "circuit_breaker_close"
    CIRCUIT_BREAKER_HALF_OPEN = "circuit_breaker_half_open"
    OSCILLATION_DETECTED = "oscillation_detected"
    OSCILLATION_BROKEN = "oscillation_broken"
    ESCALATION_REQUESTED = "escalation_requested"
    ESCALATION_RESOLVED = "escalation_resolved"
    FILE_LOCK_ACQUIRED = "file_lock_acquired"
    FILE_LOCK_TIMEOUT = "file_lock_timeout"
    FILE_OPERATION = "file_operation"
    STATE_RECONCILIATION = "state_reconciliation"


@dataclass
class MetricEntry:
    """Single metric entry."""

    metric_type: MetricType
    timestamp: datetime
    labels: Dict[str, str]
    value: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": self.metric_type.value,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels,
            "value": self.value,
            "metadata": self.metadata,
        }


@dataclass
class CounterMetric:
    """Counter metric with labels."""

    name: str
    description: str
    value: float = 0.0
    labels: Dict[str, str] = field(default_factory=dict)

    def increment(self, amount: float = 1.0) -> None:
        """Increment counter."""
        self.value += amount


@dataclass
class GaugeMetric:
    """Gauge metric with labels."""

    name: str
    description: str
    value: float = 0.0
    labels: Dict[str, str] = field(default_factory=dict)

    def set(self, value: float) -> None:
        """Set gauge value."""
        self.value = value

    def increment(self, amount: float = 1.0) -> None:
        """Increment gauge."""
        self.value += amount

    def decrement(self, amount: float = 1.0) -> None:
        """Decrement gauge."""
        self.value -= amount


@dataclass
class HistogramBucket:
    """Single histogram bucket."""

    le: float  # Less than or equal
    count: int = 0


@dataclass
class HistogramMetric:
    """Histogram metric for measuring distributions."""

    name: str
    description: str
    buckets: List[HistogramBucket] = field(default_factory=list)
    sum: float = 0.0
    count: int = 0
    labels: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.buckets:
            # Default buckets for latency in seconds
            bucket_boundaries = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
            self.buckets = [HistogramBucket(le=b) for b in bucket_boundaries]
            self.buckets.append(HistogramBucket(le=float("inf")))

    def observe(self, value: float) -> None:
        """Record an observation."""
        self.sum += value
        self.count += 1
        for bucket in self.buckets:
            if value <= bucket.le:
                bucket.count += 1


class ResilienceMetricsCollector:
    """Collects and manages resilience metrics.

    Thread-safe singleton that aggregates metrics from all
    resilience components.

    Usage:
        collector = get_metrics_collector()
        collector.record_retry_attempt("api_call", 1, False)
        summary = collector.get_summary()
    """

    def __init__(self):
        """Initialize collector."""
        self._lock = threading.Lock()
        self._entries: List[MetricEntry] = []
        self._max_entries = 10000

        # Counters
        self._retry_attempts: Dict[str, CounterMetric] = {}
        self._retry_successes: Dict[str, CounterMetric] = {}
        self._retry_failures: Dict[str, CounterMetric] = {}
        self._retry_exhausted: Dict[str, CounterMetric] = {}
        self._circuit_breaker_trips: Dict[str, CounterMetric] = {}
        self._oscillation_events: Dict[str, CounterMetric] = {}
        self._escalations: Dict[str, CounterMetric] = {}
        self._file_operations: Dict[str, CounterMetric] = {}

        # Gauges
        self._active_retries: Dict[str, GaugeMetric] = {}
        self._circuit_breaker_states: Dict[str, GaugeMetric] = {}

        # Histograms
        self._retry_durations: Dict[str, HistogramMetric] = {}
        self._file_operation_durations: Dict[str, HistogramMetric] = {}

    def _add_entry(self, entry: MetricEntry) -> None:
        """Add metric entry with sliding window."""
        with self._lock:
            self._entries.append(entry)
            while len(self._entries) > self._max_entries:
                self._entries.pop(0)

    def _get_or_create_counter(
        self,
        storage: Dict[str, CounterMetric],
        name: str,
        description: str,
        labels: Dict[str, str],
    ) -> CounterMetric:
        """Get or create counter metric."""
        key = f"{name}:{sorted(labels.items())}"
        if key not in storage:
            storage[key] = CounterMetric(
                name=name,
                description=description,
                labels=labels,
            )
        return storage[key]

    def _get_or_create_gauge(
        self,
        storage: Dict[str, GaugeMetric],
        name: str,
        description: str,
        labels: Dict[str, str],
    ) -> GaugeMetric:
        """Get or create gauge metric."""
        key = f"{name}:{sorted(labels.items())}"
        if key not in storage:
            storage[key] = GaugeMetric(
                name=name,
                description=description,
                labels=labels,
            )
        return storage[key]

    def _get_or_create_histogram(
        self,
        storage: Dict[str, HistogramMetric],
        name: str,
        description: str,
        labels: Dict[str, str],
    ) -> HistogramMetric:
        """Get or create histogram metric."""
        key = f"{name}:{sorted(labels.items())}"
        if key not in storage:
            storage[key] = HistogramMetric(
                name=name,
                description=description,
                labels=labels,
            )
        return storage[key]

    def record_retry_attempt(
        self,
        operation: str,
        attempt: int,
        success: bool,
        duration_seconds: float = 0.0,
        error_type: Optional[str] = None,
    ) -> None:
        """Record a retry attempt.

        Args:
            operation: Name of the operation being retried.
            attempt: Attempt number (1-indexed).
            success: Whether the attempt succeeded.
            duration_seconds: Time taken for the attempt.
            error_type: Type of error if failed.
        """
        labels = {"operation": operation}

        # Increment attempt counter
        counter = self._get_or_create_counter(
            self._retry_attempts,
            "resilience_retry_attempts_total",
            "Total number of retry attempts",
            labels,
        )
        with self._lock:
            counter.increment()

        # Record success or failure
        if success:
            success_counter = self._get_or_create_counter(
                self._retry_successes,
                "resilience_retry_successes_total",
                "Total number of successful retries",
                labels,
            )
            with self._lock:
                success_counter.increment()
            metric_type = MetricType.RETRY_SUCCESS
        else:
            fail_labels = {**labels, "error_type": error_type or "unknown"}
            fail_counter = self._get_or_create_counter(
                self._retry_failures,
                "resilience_retry_failures_total",
                "Total number of failed retries",
                fail_labels,
            )
            with self._lock:
                fail_counter.increment()
            metric_type = MetricType.RETRY_FAILURE

        # Record duration
        if duration_seconds > 0:
            histogram = self._get_or_create_histogram(
                self._retry_durations,
                "resilience_retry_duration_seconds",
                "Duration of retry attempts",
                labels,
            )
            with self._lock:
                histogram.observe(duration_seconds)

        # Add entry
        self._add_entry(MetricEntry(
            metric_type=metric_type,
            timestamp=datetime.now(timezone.utc),
            labels=labels,
            metadata={
                "attempt": attempt,
                "duration_seconds": duration_seconds,
                "error_type": error_type,
            },
        ))

    def record_retry_exhausted(
        self,
        operation: str,
        total_attempts: int,
        final_error: Optional[str] = None,
    ) -> None:
        """Record when all retries are exhausted.

        Args:
            operation: Name of the operation.
            total_attempts: Total attempts made.
            final_error: Final error message.
        """
        labels = {"operation": operation}

        counter = self._get_or_create_counter(
            self._retry_exhausted,
            "resilience_retry_exhausted_total",
            "Total number of retry exhaustions",
            labels,
        )
        with self._lock:
            counter.increment()

        self._add_entry(MetricEntry(
            metric_type=MetricType.RETRY_EXHAUSTED,
            timestamp=datetime.now(timezone.utc),
            labels=labels,
            metadata={
                "total_attempts": total_attempts,
                "final_error": final_error,
            },
        ))

        logger.warning(
            "RESILIENCE_METRIC: Retry exhausted for '%s' after %d attempts",
            operation,
            total_attempts,
        )

    def record_circuit_breaker_state(
        self,
        name: str,
        state: str,
        failure_count: int = 0,
        success_count: int = 0,
    ) -> None:
        """Record circuit breaker state change.

        Args:
            name: Name of the circuit breaker.
            state: New state (open, closed, half_open).
            failure_count: Current failure count.
            success_count: Current success count.
        """
        labels = {"circuit_breaker": name}

        # Map state to metric type
        state_lower = state.lower()
        if state_lower == "open":
            metric_type = MetricType.CIRCUIT_BREAKER_OPEN
            state_value = 1.0
        elif state_lower == "half_open":
            metric_type = MetricType.CIRCUIT_BREAKER_HALF_OPEN
            state_value = 0.5
        else:  # closed
            metric_type = MetricType.CIRCUIT_BREAKER_CLOSE
            state_value = 0.0

        # Update gauge
        gauge = self._get_or_create_gauge(
            self._circuit_breaker_states,
            "resilience_circuit_breaker_state",
            "Circuit breaker state (0=closed, 0.5=half_open, 1=open)",
            labels,
        )
        with self._lock:
            gauge.set(state_value)

        # Increment trip counter if opening
        if state_lower == "open":
            counter = self._get_or_create_counter(
                self._circuit_breaker_trips,
                "resilience_circuit_breaker_trips_total",
                "Total number of circuit breaker trips",
                labels,
            )
            with self._lock:
                counter.increment()

        self._add_entry(MetricEntry(
            metric_type=metric_type,
            timestamp=datetime.now(timezone.utc),
            labels=labels,
            metadata={
                "state": state,
                "failure_count": failure_count,
                "success_count": success_count,
            },
        ))

    def record_oscillation_event(
        self,
        pattern: str,
        detected: bool,
        action_taken: Optional[str] = None,
        cycle_length: int = 0,
        confidence: float = 0.0,
    ) -> None:
        """Record oscillation detection event.

        Args:
            pattern: Type of pattern (abab, cycle, runaway, none).
            detected: Whether oscillation was detected.
            action_taken: Action taken to break oscillation.
            cycle_length: Length of detected cycle.
            confidence: Detection confidence (0-1).
        """
        labels = {"pattern": pattern}

        if detected:
            metric_type = MetricType.OSCILLATION_DETECTED
            counter = self._get_or_create_counter(
                self._oscillation_events,
                "resilience_oscillation_detected_total",
                "Total oscillation detections",
                labels,
            )
            with self._lock:
                counter.increment()
        else:
            metric_type = MetricType.OSCILLATION_BROKEN

        self._add_entry(MetricEntry(
            metric_type=metric_type,
            timestamp=datetime.now(timezone.utc),
            labels=labels,
            metadata={
                "detected": detected,
                "action_taken": action_taken,
                "cycle_length": cycle_length,
                "confidence": confidence,
            },
        ))

        if detected:
            logger.info(
                "RESILIENCE_METRIC: Oscillation detected - pattern=%s, confidence=%.2f",
                pattern,
                confidence,
            )

    def record_escalation(
        self,
        intervention_type: str,
        run_id: str,
        resolved: bool = False,
        resolution: Optional[str] = None,
        wait_time_seconds: float = 0.0,
    ) -> None:
        """Record Run Master escalation.

        Args:
            intervention_type: Type of intervention requested.
            run_id: Run ID for context.
            resolved: Whether the escalation is resolved.
            resolution: Resolution action if resolved.
            wait_time_seconds: Time waited for resolution.
        """
        labels = {"intervention_type": intervention_type}

        if resolved:
            metric_type = MetricType.ESCALATION_RESOLVED
        else:
            metric_type = MetricType.ESCALATION_REQUESTED
            counter = self._get_or_create_counter(
                self._escalations,
                "resilience_escalations_total",
                "Total Run Master escalations",
                labels,
            )
            with self._lock:
                counter.increment()

        self._add_entry(MetricEntry(
            metric_type=metric_type,
            timestamp=datetime.now(timezone.utc),
            labels={**labels, "run_id": run_id},
            metadata={
                "resolved": resolved,
                "resolution": resolution,
                "wait_time_seconds": wait_time_seconds,
            },
        ))

    def record_file_operation(
        self,
        operation: str,
        path: str,
        success: bool,
        duration_seconds: float = 0.0,
        bytes_processed: int = 0,
        lock_acquired: bool = True,
    ) -> None:
        """Record file operation metrics.

        Args:
            operation: Type of operation (read, write, append).
            path: File path (for labeling, truncated).
            success: Whether operation succeeded.
            duration_seconds: Time taken.
            bytes_processed: Number of bytes read/written.
            lock_acquired: Whether lock was acquired.
        """
        # Truncate path for label
        path_label = path.split("/")[-1] if "/" in path else path
        labels = {"operation": operation, "file": path_label[:50]}

        counter = self._get_or_create_counter(
            self._file_operations,
            "resilience_file_operations_total",
            "Total file operations",
            {**labels, "success": str(success).lower()},
        )
        with self._lock:
            counter.increment()

        if duration_seconds > 0:
            histogram = self._get_or_create_histogram(
                self._file_operation_durations,
                "resilience_file_operation_duration_seconds",
                "Duration of file operations",
                {"operation": operation},
            )
            with self._lock:
                histogram.observe(duration_seconds)

        metric_type = MetricType.FILE_OPERATION
        if not lock_acquired:
            metric_type = MetricType.FILE_LOCK_TIMEOUT

        self._add_entry(MetricEntry(
            metric_type=metric_type,
            timestamp=datetime.now(timezone.utc),
            labels=labels,
            metadata={
                "success": success,
                "duration_seconds": duration_seconds,
                "bytes_processed": bytes_processed,
                "lock_acquired": lock_acquired,
            },
        ))

    def record_state_reconciliation(
        self,
        run_id: str,
        sprint_id: str,
        sources_checked: int,
        conflicts_found: int,
        winner_source: str,
        success: bool,
    ) -> None:
        """Record state reconciliation event.

        Args:
            run_id: Run ID.
            sprint_id: Sprint ID.
            sources_checked: Number of sources checked.
            conflicts_found: Number of conflicts found.
            winner_source: Source chosen as winner.
            success: Whether reconciliation succeeded.
        """
        labels = {
            "winner_source": winner_source,
            "success": str(success).lower(),
        }

        self._add_entry(MetricEntry(
            metric_type=MetricType.STATE_RECONCILIATION,
            timestamp=datetime.now(timezone.utc),
            labels=labels,
            metadata={
                "run_id": run_id,
                "sprint_id": sprint_id,
                "sources_checked": sources_checked,
                "conflicts_found": conflicts_found,
            },
        ))

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary.

        Returns:
            Dictionary with aggregated metrics.
        """
        with self._lock:
            # Count by type
            type_counts = {}
            for entry in self._entries:
                t = entry.metric_type.value
                type_counts[t] = type_counts.get(t, 0) + 1

            # Recent entries
            recent = [e.to_dict() for e in self._entries[-20:]]

            # Counter summaries
            retry_summary = {
                k: {"labels": v.labels, "value": v.value}
                for k, v in self._retry_attempts.items()
            }

            circuit_breaker_summary = {
                k: {"labels": v.labels, "state_value": v.value}
                for k, v in self._circuit_breaker_states.items()
            }

            return {
                "total_entries": len(self._entries),
                "counts_by_type": type_counts,
                "retry_attempts": retry_summary,
                "circuit_breakers": circuit_breaker_summary,
                "recent_entries": recent,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text format.

        Returns:
            Prometheus-formatted metrics string.
        """
        lines = []

        def format_labels(labels: Dict[str, str]) -> str:
            if not labels:
                return ""
            pairs = [f'{k}="{v}"' for k, v in sorted(labels.items())]
            return "{" + ",".join(pairs) + "}"

        with self._lock:
            # Counters
            for storage_name, storage, desc in [
                ("resilience_retry_attempts_total", self._retry_attempts, "Total retry attempts"),
                ("resilience_retry_successes_total", self._retry_successes, "Total retry successes"),
                ("resilience_retry_failures_total", self._retry_failures, "Total retry failures"),
                ("resilience_retry_exhausted_total", self._retry_exhausted, "Total retry exhaustions"),
                ("resilience_circuit_breaker_trips_total", self._circuit_breaker_trips, "Circuit breaker trips"),
                ("resilience_escalations_total", self._escalations, "Run Master escalations"),
                ("resilience_file_operations_total", self._file_operations, "File operations"),
            ]:
                if storage:
                    lines.append(f"# HELP {storage_name} {desc}")
                    lines.append(f"# TYPE {storage_name} counter")
                    for counter in storage.values():
                        lines.append(f"{storage_name}{format_labels(counter.labels)} {counter.value}")

            # Gauges
            if self._circuit_breaker_states:
                lines.append("# HELP resilience_circuit_breaker_state Circuit breaker state")
                lines.append("# TYPE resilience_circuit_breaker_state gauge")
                for gauge in self._circuit_breaker_states.values():
                    lines.append(f"resilience_circuit_breaker_state{format_labels(gauge.labels)} {gauge.value}")

            # Histograms
            for storage_name, storage, desc in [
                ("resilience_retry_duration_seconds", self._retry_durations, "Retry attempt duration"),
                ("resilience_file_operation_duration_seconds", self._file_operation_durations, "File operation duration"),
            ]:
                if storage:
                    lines.append(f"# HELP {storage_name} {desc}")
                    lines.append(f"# TYPE {storage_name} histogram")
                    for histogram in storage.values():
                        for bucket in histogram.buckets:
                            le_str = "+Inf" if bucket.le == float("inf") else str(bucket.le)
                            bucket_labels = {**histogram.labels, "le": le_str}
                            lines.append(f"{storage_name}_bucket{format_labels(bucket_labels)} {bucket.count}")
                        lines.append(f"{storage_name}_sum{format_labels(histogram.labels)} {histogram.sum}")
                        lines.append(f"{storage_name}_count{format_labels(histogram.labels)} {histogram.count}")

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._entries.clear()
            self._retry_attempts.clear()
            self._retry_successes.clear()
            self._retry_failures.clear()
            self._retry_exhausted.clear()
            self._circuit_breaker_trips.clear()
            self._circuit_breaker_states.clear()
            self._oscillation_events.clear()
            self._escalations.clear()
            self._file_operations.clear()
            self._retry_durations.clear()
            self._file_operation_durations.clear()


# Global collector instance
_global_collector: Optional[ResilienceMetricsCollector] = None
_collector_lock = threading.Lock()


def get_metrics_collector() -> ResilienceMetricsCollector:
    """Get or create global metrics collector."""
    global _global_collector
    with _collector_lock:
        if _global_collector is None:
            _global_collector = ResilienceMetricsCollector()
        return _global_collector


def reset_metrics_collector() -> None:
    """Reset global metrics collector."""
    global _global_collector
    with _collector_lock:
        if _global_collector is not None:
            _global_collector.reset()
        _global_collector = None


# Convenience functions
def record_retry_attempt(
    operation: str,
    attempt: int,
    success: bool,
    duration_seconds: float = 0.0,
    error_type: Optional[str] = None,
) -> None:
    """Record retry attempt (convenience function)."""
    get_metrics_collector().record_retry_attempt(
        operation, attempt, success, duration_seconds, error_type
    )


def record_retry_exhausted(
    operation: str,
    total_attempts: int,
    final_error: Optional[str] = None,
) -> None:
    """Record retry exhaustion (convenience function)."""
    get_metrics_collector().record_retry_exhausted(
        operation, total_attempts, final_error
    )


def record_circuit_breaker_state(
    name: str,
    state: str,
    failure_count: int = 0,
    success_count: int = 0,
) -> None:
    """Record circuit breaker state (convenience function)."""
    get_metrics_collector().record_circuit_breaker_state(
        name, state, failure_count, success_count
    )


def record_oscillation_event(
    pattern: str,
    detected: bool,
    action_taken: Optional[str] = None,
    cycle_length: int = 0,
    confidence: float = 0.0,
) -> None:
    """Record oscillation event (convenience function)."""
    get_metrics_collector().record_oscillation_event(
        pattern, detected, action_taken, cycle_length, confidence
    )


def record_escalation(
    intervention_type: str,
    run_id: str,
    resolved: bool = False,
    resolution: Optional[str] = None,
    wait_time_seconds: float = 0.0,
) -> None:
    """Record escalation (convenience function)."""
    get_metrics_collector().record_escalation(
        intervention_type, run_id, resolved, resolution, wait_time_seconds
    )


def record_file_operation(
    operation: str,
    path: str,
    success: bool,
    duration_seconds: float = 0.0,
    bytes_processed: int = 0,
    lock_acquired: bool = True,
) -> None:
    """Record file operation (convenience function)."""
    get_metrics_collector().record_file_operation(
        operation, path, success, duration_seconds, bytes_processed, lock_acquired
    )


def record_state_reconciliation(
    run_id: str,
    sprint_id: str,
    sources_checked: int,
    conflicts_found: int,
    winner_source: str,
    success: bool,
) -> None:
    """Record state reconciliation (convenience function)."""
    get_metrics_collector().record_state_reconciliation(
        run_id, sprint_id, sources_checked, conflicts_found, winner_source, success
    )


def get_metrics_summary() -> Dict[str, Any]:
    """Get metrics summary (convenience function)."""
    return get_metrics_collector().get_summary()


def export_prometheus_metrics() -> str:
    """Export Prometheus metrics (convenience function)."""
    return get_metrics_collector().export_prometheus()


class MetricsTimer:
    """Context manager for timing operations.

    Usage:
        with MetricsTimer() as timer:
            do_something()
        print(f"Took {timer.duration} seconds")
    """

    def __init__(self):
        self.start_time: float = 0.0
        self.end_time: float = 0.0

    @property
    def duration(self) -> float:
        """Get duration in seconds."""
        if self.end_time > 0:
            return self.end_time - self.start_time
        elif self.start_time > 0:
            return time.time() - self.start_time
        return 0.0

    def __enter__(self) -> "MetricsTimer":
        self.start_time = time.time()
        return self

    def __exit__(self, *args) -> None:
        self.end_time = time.time()


__all__ = [
    # Types
    "MetricType",
    "MetricEntry",
    "CounterMetric",
    "GaugeMetric",
    "HistogramBucket",
    "HistogramMetric",
    # Collector
    "ResilienceMetricsCollector",
    "get_metrics_collector",
    "reset_metrics_collector",
    # Convenience functions
    "record_retry_attempt",
    "record_retry_exhausted",
    "record_circuit_breaker_state",
    "record_oscillation_event",
    "record_escalation",
    "record_file_operation",
    "record_state_reconciliation",
    "get_metrics_summary",
    "export_prometheus_metrics",
    # Timer
    "MetricsTimer",
]
