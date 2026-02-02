# cockpit/resilience_reader.py
"""
Read-only resilience state reader for cockpit.

This module provides functions to read resilience state from the pipeline:
- Circuit breaker states
- Retry metrics
- Oscillation detection state
- Prometheus-format metrics

INVARIANTS:
    INV-P1-001: NEVER import directly from resilience in main cockpit files
    INV-P1-002: ALL functions have try/except and return {} or [] on error
    INV-P1-003: Use lazy import to avoid circular dependencies
    INV-P1-004: Functions are PURE - no global state modification
    INV-P1-005: Logging uses logger.warning, not logger.error (graceful)
    INV-P1-006: Return values must be JSON-serializable (datetimes as ISO strings)
"""
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Lazy Import Helpers (INV-P1-001, INV-P1-003)
# =============================================================================


def _get_metrics_collector() -> Optional[Any]:
    """Get metrics collector via lazy import."""
    try:
        from pipeline.langgraph.resilience import get_metrics_collector
        return get_metrics_collector()
    except ImportError:
        logger.warning("resilience_reader: Could not import get_metrics_collector")
        return None
    except Exception as e:
        logger.warning("resilience_reader: Error getting metrics collector: %s", e)
        return None


def _get_circuit_breaker_registry() -> Optional[Any]:
    """Get circuit breaker registry via lazy import."""
    try:
        from pipeline.langgraph.resilience import get_circuit_breaker_registry
        return get_circuit_breaker_registry()
    except ImportError:
        logger.warning("resilience_reader: Could not import get_circuit_breaker_registry")
        return None
    except Exception as e:
        logger.warning("resilience_reader: Error getting circuit breaker registry: %s", e)
        return None


def _get_oscillation_tracker() -> Optional[Any]:
    """Get oscillation tracker via lazy import."""
    try:
        from pipeline.langgraph.resilience import get_oscillation_tracker
        return get_oscillation_tracker()
    except ImportError:
        logger.warning("resilience_reader: Could not import get_oscillation_tracker")
        return None
    except Exception as e:
        logger.warning("resilience_reader: Error getting oscillation tracker: %s", e)
        return None


# =============================================================================
# Public Functions (INV-P1-002, INV-P1-004, INV-P1-005, INV-P1-006)
# =============================================================================


def read_circuit_breakers() -> Dict[str, Dict[str, Any]]:
    """Read circuit breaker states - READ ONLY."""
    try:
        registry = _get_circuit_breaker_registry()
        if registry is None:
            return {}

        all_stats = registry.get_all_stats()
        result = {}

        for name, stats in all_stats.items():
            try:
                stats_dict = stats.to_dict() if hasattr(stats, 'to_dict') else {}
                result[name] = stats_dict
            except Exception as e:
                logger.warning(
                    "resilience_reader: Error serializing circuit breaker '%s': %s",
                    name, e
                )
                result[name] = {"state": "unknown", "error": str(e)}

        return result

    except Exception as e:
        logger.warning("resilience_reader: Error reading circuit breakers: %s", e)
        return {}


def read_retry_metrics() -> Dict[str, Any]:
    """Read retry metrics - READ ONLY."""
    try:
        collector = _get_metrics_collector()
        if collector is None:
            return {}

        summary = collector.get_summary()
        return _ensure_json_serializable(summary)

    except Exception as e:
        logger.warning("resilience_reader: Error reading retry metrics: %s", e)
        return {}


def read_oscillation_state() -> Dict[str, Any]:
    """Read oscillation tracker state - READ ONLY."""
    try:
        tracker = _get_oscillation_tracker()
        if tracker is None:
            return {}

        state = tracker.to_dict() if hasattr(tracker, 'to_dict') else {}

        try:
            if hasattr(tracker, 'detect'):
                detection = tracker.detect()
                if hasattr(detection, 'to_dict'):
                    state['current_detection'] = detection.to_dict()
                else:
                    state['current_detection'] = {
                        'detected': getattr(detection, 'detected', False),
                        'pattern': str(getattr(detection, 'pattern', 'none')),
                    }
        except Exception as e:
            logger.warning("resilience_reader: Error getting detection: %s", e)
            state['current_detection'] = None

        try:
            if hasattr(tracker, 'is_oscillating'):
                state['is_oscillating'] = tracker.is_oscillating()
            else:
                state['is_oscillating'] = False
        except Exception:
            state['is_oscillating'] = False

        return _ensure_json_serializable(state)

    except Exception as e:
        logger.warning("resilience_reader: Error reading oscillation state: %s", e)
        return {}


def read_prometheus_metrics() -> str:
    """Read metrics in Prometheus format - READ ONLY."""
    try:
        collector = _get_metrics_collector()
        if collector is None:
            return ""

        if hasattr(collector, 'export_prometheus'):
            return collector.export_prometheus()
        elif hasattr(collector, 'export'):
            return collector.export()
        else:
            logger.warning(
                "resilience_reader: Metrics collector has no export method"
            )
            return ""

    except Exception as e:
        logger.warning("resilience_reader: Error reading prometheus metrics: %s", e)
        return ""


def get_resilience_state() -> Dict[str, Any]:
    """Get all resilience state in one call - READ ONLY."""
    try:
        circuit_breakers = read_circuit_breakers()
        retry_metrics = read_retry_metrics()
        oscillation = read_oscillation_state()

        has_open_circuits = any(
            cb.get('state') == 'open'
            for cb in circuit_breakers.values()
        )
        is_oscillating = oscillation.get('is_oscillating', False)

        return {
            'circuit_breakers': circuit_breakers,
            'retry_metrics': retry_metrics,
            'oscillation': oscillation,
            'has_open_circuits': has_open_circuits,
            'is_oscillating': is_oscillating,
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.warning("resilience_reader: Error getting resilience state: %s", e)
        return {
            'circuit_breakers': {},
            'retry_metrics': {},
            'oscillation': {},
            'has_open_circuits': False,
            'is_oscillating': False,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'error': str(e),
        }


# =============================================================================
# Helper Functions
# =============================================================================


def _ensure_json_serializable(obj: Any) -> Any:
    """Ensure object is JSON-serializable."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: _ensure_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_ensure_json_serializable(item) for item in obj]
    elif hasattr(obj, 'to_dict'):
        return _ensure_json_serializable(obj.to_dict())
    elif hasattr(obj, 'value'):
        return obj.value
    else:
        return obj


__all__ = [
    'read_circuit_breakers',
    'read_retry_metrics',
    'read_oscillation_state',
    'read_prometheus_metrics',
    'get_resilience_state',
]
