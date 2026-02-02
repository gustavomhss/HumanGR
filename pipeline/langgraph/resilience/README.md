# Pipeline Resilience Module

Robust retry mechanisms, circuit breakers, oscillation detection, and process-safe file operations for the Pipeline.

## Philosophy

```
NUNCA block (exceto seguranca critica)
NUNCA continuar silenciosamente
SEMPRE: Retry → Escala RM → Aguarda → Continua
```

## Components

| Component | File | Purpose |
|-----------|------|---------|
| Retry Config | `retry_config.py` | Immutable retry configuration types |
| Retry Engine | `retry.py` | Core retry functions with exponential backoff |
| Escalation | `escalation.py` | Run Master escalation handlers |
| Circuit Breaker | `circuit_breaker.py` | Circuit breaker pattern implementation |
| File Safety | `file_safety.py` | Process-safe file operations with fcntl |
| Reconciliation | `reconciliation.py` | State reconciliation across sources |
| Oscillation | `oscillation.py` | Cyclic failure pattern detection |
| Metrics | `metrics.py` | Prometheus-compatible observability |

## Quick Start

### Retry with Exponential Backoff

```python
from pipeline.langgraph.resilience import (
    RetryConfig,
    RETRY_TRANSIENT,
    retry_async,
)

# Use preset config
result = await retry_async(
    my_async_operation,
    "operation_name",
    RETRY_TRANSIENT,
)

# Or custom config
config = RetryConfig(
    max_retries=5,
    base_delay_seconds=1.0,
    max_delay_seconds=30.0,
    exponential_base=2.0,
    jitter=True,
)
result = await retry_async(my_operation, "my_op", config)
```

### Circuit Breaker

```python
from pipeline.langgraph.resilience import (
    CircuitBreaker,
    CircuitBreakerOpen,
    with_circuit_breaker,
)

# Using convenience function
result = await with_circuit_breaker(
    "redis",
    lambda: redis_client.get("key"),
    failure_threshold=5,
    recovery_timeout_seconds=60.0,
)

# Or direct usage
breaker = CircuitBreaker(
    service_name="external_api",
    failure_threshold=3,
    recovery_timeout_seconds=30.0,
)

try:
    result = await breaker.call(my_async_operation)
except CircuitBreakerOpen as e:
    print(f"Circuit open, retry in {e.time_until_half_open}s")
```

### Oscillation Detection

```python
from pipeline.langgraph.resilience import (
    OscillationTracker,
    OscillationPattern,
    detect_oscillation,
)

tracker = OscillationTracker()

# Add rework entries
tracker.add_rework(
    rejecting_agent="qa_master",
    failed_items=["file.py"],
    work_type="code_review",
    attempt=1,
)

# Detect oscillation
detection = tracker.detect()
if detection.detected:
    print(f"Pattern: {detection.pattern}")  # ABAB, CYCLE, or RUNAWAY

    # Break the oscillation
    result = tracker.attempt_break(state)
    if result.requires_human:
        print("Needs human intervention")
```

### Process-Safe File Operations

```python
from pipeline.langgraph.resilience import (
    write_json_safe,
    read_json_safe,
    write_yaml_safe,
    read_yaml_safe,
    append_ndjson_safe,
    read_ndjson_safe,
)

# JSON operations
write_json_safe(path, {"key": "value"})
data = read_json_safe(path)

# YAML operations
write_yaml_safe(path, config_dict)
config = read_yaml_safe(path)

# NDJSON (append-only event log)
append_ndjson_safe(path, {"event": "gate_passed", "gate": "G0"})
events = read_ndjson_safe(path)
```

### Metrics Collection

```python
from pipeline.langgraph.resilience import (
    record_retry_attempt,
    record_circuit_breaker_state,
    record_oscillation_event,
    get_metrics_summary,
    export_prometheus_metrics,
    MetricsTimer,
)

# Record metrics
with MetricsTimer() as timer:
    result = await my_operation()

record_retry_attempt("my_op", 1, success=True, duration=timer.duration)

# Get summary
summary = get_metrics_summary()
print(f"Total entries: {summary['total_entries']}")

# Export Prometheus format
prometheus_output = export_prometheus_metrics()
```

## Retry Presets

| Preset | Max Retries | Base Delay | Max Delay | Use Case |
|--------|-------------|------------|-----------|----------|
| `RETRY_TRANSIENT` | 3 | 1.0s | 30s | Transient failures |
| `RETRY_EXTERNAL` | 5 | 2.0s | 60s | External API calls |
| `RETRY_CRITICAL` | 10 | 0.5s | 120s | Critical operations |
| `RETRY_AGGRESSIVE` | 7 | 0.1s | 10s | Fast retry scenarios |
| `RETRY_NONE` | 0 | - | - | No retry (fail fast) |

## Circuit Breaker States

```
CLOSED (normal operation)
    │
    │ failure_count >= failure_threshold
    ▼
OPEN (rejecting calls)
    │
    │ recovery_timeout elapsed
    ▼
HALF_OPEN (testing recovery)
    │
    ├─ success × N → CLOSED
    │
    └─ failure → OPEN
```

## Oscillation Patterns

| Pattern | Description | Detection |
|---------|-------------|-----------|
| `ABAB` | Alternating rejections (A→B→A→B) | 2+ cycles of alternation |
| `CYCLE` | Cyclic pattern (A→B→C→A→B→C) | 2+ cycles of same sequence |
| `RUNAWAY` | Same failure repeating | N consecutive identical failures |
| `NONE` | No pattern detected | - |

## Run Master Escalation

```python
from pipeline.langgraph.resilience import (
    retry_with_run_master,
    retry_and_continue,
    InterventionType,
)

# Retry with automatic RM escalation on exhaustion
result = await retry_with_run_master(
    my_operation,
    "critical_task",
    config=RETRY_CRITICAL,
    context={"task_id": "T001"},
)

# Or use retry_and_continue for graceful degradation
result = await retry_and_continue(
    my_operation,
    "optional_task",
    config=RETRY_TRANSIENT,
    default_value=None,
)
```

## Tests

```bash
# Run all resilience tests (357 tests)
PYTHONPATH=src pytest tests/test_resilience*.py -v

# Run specific module tests
PYTHONPATH=src pytest tests/test_resilience_retry.py -v
PYTHONPATH=src pytest tests/test_resilience_circuit_breaker.py -v
PYTHONPATH=src pytest tests/test_resilience_oscillation.py -v
PYTHONPATH=src pytest tests/test_resilience_metrics.py -v
PYTHONPATH=src pytest tests/test_resilience_e2e.py -v
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `RESILIENCE_METRICS_ENABLED` | Enable metrics collection | `true` |
| `RESILIENCE_METRICS_MAX_ENTRIES` | Max entries before rotation | `10000` |
| `CIRCUIT_BREAKER_DEFAULT_THRESHOLD` | Default failure threshold | `5` |
| `CIRCUIT_BREAKER_DEFAULT_TIMEOUT` | Default recovery timeout (s) | `60.0` |

## Integration with Pipeline

The resilience module integrates automatically with:

- **Gate Runner**: Retry on transient failures, escalate on exhaustion
- **LangGraph Workflow**: Circuit breakers for external services
- **State Management**: Process-safe file operations
- **Observability**: Prometheus metrics export

## Key Principles

1. **Never Block**: Operations should never hang indefinitely
2. **Never Silent Fail**: All failures are logged and tracked
3. **Always Escalate**: Exhausted retries escalate to Run Master
4. **Process Safe**: File operations use fcntl locking
5. **Observable**: All operations emit metrics
