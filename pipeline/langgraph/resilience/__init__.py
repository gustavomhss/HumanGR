"""Pipeline Resilience Module.

This module provides robust retry mechanisms, Run Master escalation,
circuit breakers, oscillation detection, and process-safe file operations
for the Pipeline.

Philosophy:
    - NUNCA block (except critical security)
    - NUNCA continuar silenciosamente
    - SEMPRE: Retry → Escala RM → Aguarda → Continua

Components:
    - retry_config: Immutable retry configuration types
    - retry: Core retry functions with exponential backoff
    - escalation: Run Master escalation handlers
    - circuit_breaker: Circuit breaker pattern implementation
    - file_safety: Process-safe file operations
    - reconciliation: State reconciliation across sources
    - oscillation: Cyclic failure pattern detection and breaking
    - metrics: Observability and Prometheus-compatible metrics
"""

from .retry_config import (
    RetryConfig,
    RETRY_TRANSIENT,
    RETRY_EXTERNAL,
    RETRY_CRITICAL,
    RETRY_NONE,
    RETRY_AGGRESSIVE,
)
from .retry import (
    retry_async,
    retry_async_with_result,
    retry_sync,
    retry_sync_with_result,
    calculate_delay,
    RetryResult,
)
from .file_safety import (
    FileLockError,
    ProcessSafeWriter,
    ProcessSafeReader,
    write_json_safe,
    read_json_safe,
    write_yaml_safe,
    read_yaml_safe,
    append_ndjson_safe,
    read_ndjson_safe,
)
from .escalation import (
    InterventionType,
    InterventionRequest,
    InterventionResolution,
    EscalationError,
    retry_with_run_master,
    retry_and_continue,
)
from .circuit_breaker import (
    CircuitState,
    CircuitBreakerOpen,
    CircuitBreakerStats,
    CircuitBreaker,
    GlobalCircuitBreakerRegistry,
    get_circuit_breaker_registry,
    with_circuit_breaker,
)
from .reconciliation import (
    StateSource,
    StateSnapshot,
    ReconciliationResult,
    StateNotFoundError,
    ReconciliationError,
    read_state_from_redis,
    read_state_from_file,
    read_state_from_checkpoint,
    write_state_to_redis,
    write_state_to_file,
    reconcile_state,
    quick_reconcile,
)
from .oscillation import (
    OscillationPattern,
    ReworkHistoryEntry,
    OscillationDetectionResult,
    OscillationBreakResult,
    detect_oscillation,
    break_oscillation,
    OscillationTracker,
    get_oscillation_tracker,
    reset_oscillation_tracker,
)
from .metrics import (
    MetricType,
    MetricEntry,
    CounterMetric,
    GaugeMetric,
    HistogramMetric,
    ResilienceMetricsCollector,
    get_metrics_collector,
    reset_metrics_collector,
    record_retry_attempt,
    record_retry_exhausted,
    record_circuit_breaker_state,
    record_oscillation_event,
    record_escalation,
    record_file_operation,
    record_state_reconciliation,
    get_metrics_summary,
    export_prometheus_metrics,
    MetricsTimer,
)

__all__ = [
    # Config types
    "RetryConfig",
    # Presets
    "RETRY_TRANSIENT",
    "RETRY_EXTERNAL",
    "RETRY_CRITICAL",
    "RETRY_NONE",
    "RETRY_AGGRESSIVE",
    # Retry functions
    "retry_async",
    "retry_async_with_result",
    "retry_sync",
    "retry_sync_with_result",
    "calculate_delay",
    "RetryResult",
    # File safety
    "FileLockError",
    "ProcessSafeWriter",
    "ProcessSafeReader",
    "write_json_safe",
    "read_json_safe",
    "write_yaml_safe",
    "read_yaml_safe",
    "append_ndjson_safe",
    "read_ndjson_safe",
    # Escalation
    "InterventionType",
    "InterventionRequest",
    "InterventionResolution",
    "EscalationError",
    "retry_with_run_master",
    "retry_and_continue",
    # Circuit Breaker
    "CircuitState",
    "CircuitBreakerOpen",
    "CircuitBreakerStats",
    "CircuitBreaker",
    "GlobalCircuitBreakerRegistry",
    "get_circuit_breaker_registry",
    "with_circuit_breaker",
    # Reconciliation
    "StateSource",
    "StateSnapshot",
    "ReconciliationResult",
    "StateNotFoundError",
    "ReconciliationError",
    "read_state_from_redis",
    "read_state_from_file",
    "read_state_from_checkpoint",
    "write_state_to_redis",
    "write_state_to_file",
    "reconcile_state",
    "quick_reconcile",
    # Oscillation Detection
    "OscillationPattern",
    "ReworkHistoryEntry",
    "OscillationDetectionResult",
    "OscillationBreakResult",
    "detect_oscillation",
    "break_oscillation",
    "OscillationTracker",
    "get_oscillation_tracker",
    "reset_oscillation_tracker",
    # Metrics and Observability
    "MetricType",
    "MetricEntry",
    "CounterMetric",
    "GaugeMetric",
    "HistogramMetric",
    "ResilienceMetricsCollector",
    "get_metrics_collector",
    "reset_metrics_collector",
    "record_retry_attempt",
    "record_retry_exhausted",
    "record_circuit_breaker_state",
    "record_oscillation_event",
    "record_escalation",
    "record_file_operation",
    "record_state_reconciliation",
    "get_metrics_summary",
    "export_prometheus_metrics",
    "MetricsTimer",
]
