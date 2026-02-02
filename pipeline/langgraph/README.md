# LangGraph Pipeline Module

LangGraph-based workflow execution for the Pipeline Autonomo.

## Structure

```
langgraph/
├── __init__.py           # Module exports
├── workflow.py           # Main workflow definition
├── state.py              # State management
├── bridge.py             # Pipeline integration bridge
├── checkpointer.py       # State checkpointing
├── invariants.py         # Workflow invariants
├── stack_injection.py    # Stack context injection
├── trust_boundaries.py   # Trust boundary enforcement
├── cerebro_stack_mapping.py  # Cerebro to node mapping
└── resilience/           # Resilience subsystem
    ├── retry_config.py   # Retry configurations
    ├── retry.py          # Retry engine
    ├── escalation.py     # Run Master escalation
    ├── circuit_breaker.py # Circuit breaker pattern
    ├── file_safety.py    # Process-safe file ops
    ├── reconciliation.py # State reconciliation
    ├── oscillation.py    # Oscillation detection
    └── metrics.py        # Observability metrics
```

## Key Components

### Workflow (`workflow.py`)

Main LangGraph workflow with nodes for each pipeline phase.

### State (`state.py`)

TypedDict-based state management compatible with LangGraph.

### Resilience (`resilience/`)

Complete resilience system with:
- Retry with exponential backoff
- Circuit breaker for cascade failure prevention
- Oscillation detection (ABAB, CYCLE, RUNAWAY patterns)
- Process-safe file operations
- State reconciliation
- Prometheus-compatible metrics

See [resilience/README.md](resilience/README.md) for details.

## Usage

```python
from pipeline.langgraph import (
    create_workflow,
    PipelineState,
)

# Create workflow
workflow = create_workflow()

# Execute with initial state
state = PipelineState(
    sprint_id="S00",
    phase="INIT",
)
result = await workflow.ainvoke(state)
```

## Tests

```bash
# Run all langgraph tests
PYTHONPATH=src pytest tests/test_langgraph*.py tests/test_resilience*.py -v
```
