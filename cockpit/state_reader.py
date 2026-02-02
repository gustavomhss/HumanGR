# cockpit/state_reader.py
"""
READ-ONLY state reader for cockpit.
NEVER writes to any file. Only reads.

This module provides functions to read pipeline state files:
- run_state.yml: Current run state (phase, status, etc.)
- heartbeats/*.json: Agent heartbeats
- event_log.ndjson: Event log (newline-delimited JSON)
- gates/G*_result.json: Gate execution results
- run.log: Python logging output from pipeline
- Redis pipeline:history: Recent pipeline events
"""
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml
import json
import logging

logger = logging.getLogger(__name__)

# Redis connection for reading pipeline history
_redis_client = None


def _get_redis():
    """Get Redis client (lazy initialization)."""
    global _redis_client
    if _redis_client is None:
        try:
            import redis
            _redis_client = redis.Redis(
                host="localhost",
                port=6379,
                decode_responses=True,
                socket_connect_timeout=2,
            )
            # Test connection
            _redis_client.ping()
        except Exception as e:
            logger.debug(f"Redis connection failed: {e}")
            _redis_client = False  # Mark as unavailable
    return _redis_client if _redis_client else None


def read_run_state(run_dir: Path) -> Dict[str, Any]:
    """
    Read run_state.yml - READ ONLY.

    Args:
        run_dir: Path to the run directory

    Returns:
        Dict with run state or empty dict if file doesn't exist
    """
    path = run_dir / "state" / "run_state.yml"
    if not path.exists():
        # Try alternate location
        alt_path = run_dir / "run_state.yml"
        if alt_path.exists():
            path = alt_path
        else:
            return {}

    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"Failed to read run_state: {e}")
        return {}


def read_heartbeats(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Read all agent heartbeats - READ ONLY.

    Args:
        run_dir: Path to the run directory

    Returns:
        Dict mapping agent_id to heartbeat data
    """
    hb_dir = run_dir / "state" / "ipc" / "heartbeats"
    if not hb_dir.exists():
        # Try alternate location
        alt_dir = run_dir / "heartbeats"
        if alt_dir.exists():
            hb_dir = alt_dir
        else:
            return {}

    result = {}
    try:
        for f in hb_dir.glob("*.json"):
            try:
                with open(f, 'r', encoding='utf-8') as fp:
                    result[f.stem] = json.load(fp)
            except Exception as e:
                logger.warning(f"Failed to read heartbeat {f.name}: {e}")
    except Exception as e:
        logger.warning(f"Failed to list heartbeats: {e}")

    return result


def read_events(run_dir: Path, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Read event log - READ ONLY. Returns last N events.

    Args:
        run_dir: Path to the run directory
        limit: Maximum number of events to return (default: 100)

    Returns:
        List of event dicts (most recent last)
    """
    path = run_dir / "state" / "event_log.ndjson"
    if not path.exists():
        # Try alternate locations
        alt_paths = [
            run_dir / "event_log.ndjson",
            run_dir / "events.ndjson",
            run_dir / "state" / "events.ndjson",
        ]
        for alt_path in alt_paths:
            if alt_path.exists():
                path = alt_path
                break
        else:
            return []

    events = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        logger.warning(f"Failed to read events: {e}")
        return []

    return events[-limit:]


def read_gate_results(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Read gate results - READ ONLY.

    Args:
        run_dir: Path to the run directory

    Returns:
        Dict mapping gate_id (e.g., "G0") to gate result data
    """
    gates_dir = run_dir / "state" / "gates"
    if not gates_dir.exists():
        # Try alternate location
        alt_dir = run_dir / "gates"
        if alt_dir.exists():
            gates_dir = alt_dir
        else:
            return {}

    result = {}
    try:
        for f in gates_dir.glob("G*_result.json"):
            try:
                with open(f, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
                    gate_id = data.get("gate_id", f.stem.replace("_result", ""))
                    result[gate_id] = data
            except Exception as e:
                logger.warning(f"Failed to read gate result {f.name}: {e}")
    except Exception as e:
        logger.warning(f"Failed to list gate results: {e}")

    return result


def read_agent_status(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Read agent status from IPC - READ ONLY.

    Args:
        run_dir: Path to the run directory

    Returns:
        Dict mapping agent_id to status info
    """
    status_dir = run_dir / "state" / "ipc" / "agents"
    if not status_dir.exists():
        return {}

    result = {}
    try:
        for f in status_dir.glob("*.json"):
            try:
                with open(f, 'r', encoding='utf-8') as fp:
                    result[f.stem] = json.load(fp)
            except Exception as e:
                logger.warning(f"Failed to read agent status {f.name}: {e}")
    except Exception as e:
        logger.warning(f"Failed to list agent statuses: {e}")

    return result


def read_metrics(run_dir: Path) -> Dict[str, Any]:
    """
    Read pipeline metrics - READ ONLY.

    Args:
        run_dir: Path to the run directory

    Returns:
        Dict with metrics data
    """
    path = run_dir / "state" / "metrics.json"
    if not path.exists():
        alt_path = run_dir / "metrics.json"
        if alt_path.exists():
            path = alt_path
        else:
            return {}

    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to read metrics: {e}")
        return {}


def read_run_log(run_dir: Path, limit: int = 100, offset: int = 0) -> List[str]:
    """
    Read run.log - READ ONLY. Returns last N lines.

    Args:
        run_dir: Path to the run directory
        limit: Maximum number of lines to return (default: 100)
        offset: Number of lines to skip from the end (for pagination)

    Returns:
        List of log lines (most recent last)
    """
    path = run_dir / "run.log"
    if not path.exists():
        return []

    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        # Strip newlines and filter empty lines
        lines = [line.rstrip('\n\r') for line in lines if line.strip()]

        # Apply offset and limit
        if offset > 0:
            lines = lines[:-offset] if offset < len(lines) else []

        return lines[-limit:] if len(lines) > limit else lines

    except Exception as e:
        logger.warning(f"Failed to read run.log: {e}")
        return []


def read_redis_history(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Read pipeline history from Redis - READ ONLY.

    Args:
        limit: Maximum number of events to return

    Returns:
        List of event dicts (most recent first)
    """
    r = _get_redis()
    if not r:
        return []

    try:
        # Get events from list (LRANGE returns newest first based on LPUSH)
        raw_events = r.lrange("pipeline:history", 0, limit - 1)
        events = []
        for raw in raw_events:
            try:
                events.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
        return events
    except Exception as e:
        logger.debug(f"Failed to read Redis history: {e}")
        return []


def read_redis_pipeline_state() -> Dict[str, Any]:
    """
    Read current pipeline state from Redis - READ ONLY.

    Returns:
        Dict with pipeline state from Redis
    """
    r = _get_redis()
    if not r:
        return {}

    try:
        state = {}

        # Get scalar values
        for key in [
            "pipeline:status",
            "pipeline:run_id",
            "pipeline:run_started_at",
            "pipeline:sprint:current",
            "pipeline:sprint:phase",
            "pipeline:agent:current",
            "pipeline:task:current",
        ]:
            val = r.get(key)
            if val:
                # Extract simple key name
                simple_key = key.replace("pipeline:", "").replace(":", "_")
                state[simple_key] = val

        # Get counters
        for key in ["pipeline:gates:passed", "pipeline:gates:failed"]:
            val = r.get(key)
            if val:
                simple_key = key.replace("pipeline:", "").replace(":", "_")
                state[simple_key] = int(val)

        return state
    except Exception as e:
        logger.debug(f"Failed to read Redis state: {e}")
        return {}


def read_current_phase_from_checkpoints(project_root: Path, run_id: str) -> Optional[Dict[str, str]]:
    """
    Read current phase from LangGraph checkpoints - READ ONLY.

    Args:
        project_root: Path to the project root
        run_id: The run ID to look for (can be partial match)

    Returns:
        Dict with 'phase', 'sprint_id', 'run_id' or None
    """
    checkpoints_dir = project_root / ".langgraph" / "checkpoints" / "checkpoints"
    if not checkpoints_dir.exists():
        return None

    try:
        all_checkpoints = []

        # New format (underscores)
        for ckpt in checkpoints_dir.glob(f"ckpt_{run_id}_*.json"):
            all_checkpoints.append(ckpt)

        # Old format (colons)
        for ckpt in checkpoints_dir.glob(f"ckpt:{run_id}:*.json"):
            all_checkpoints.append(ckpt)

        # Partial match
        if not all_checkpoints:
            for ckpt in checkpoints_dir.glob("ckpt_*.json"):
                if run_id in ckpt.name:
                    all_checkpoints.append(ckpt)
            for ckpt in checkpoints_dir.glob("ckpt:*.json"):
                if run_id in ckpt.name:
                    all_checkpoints.append(ckpt)

        if not all_checkpoints:
            return None

        # Sort by modification time
        all_checkpoints = sorted(all_checkpoints, key=lambda p: p.stat().st_mtime, reverse=True)
        latest = all_checkpoints[0]

        name = latest.stem
        result = {}

        if name.startswith("ckpt_"):
            parts = name[5:].split("_")
            if "attempt" in parts:
                attempt_idx = parts.index("attempt")
                if attempt_idx >= 2:
                    result["phase"] = parts[attempt_idx - 1]
                    result["sprint_id"] = parts[attempt_idx - 2]

        elif name.startswith("ckpt:"):
            parts = name[5:].split(":")
            if len(parts) >= 4:
                result["phase"] = parts[2]
                result["sprint_id"] = parts[1]

        # Read checkpoint file for more accurate state
        try:
            with open(latest, 'r', encoding='utf-8') as f:
                ckpt_data = json.load(f)
                state = ckpt_data.get("state", {})
                if state.get("phase"):
                    result["phase"] = state["phase"]
                if state.get("sprint_id"):
                    result["sprint_id"] = state["sprint_id"]
                if state.get("run_id"):
                    result["run_id"] = state["run_id"]
        except Exception:
            pass

        return result if result else None

    except Exception as e:
        logger.warning(f"Failed to read checkpoints: {e}")
        return None


def get_all_state(run_dir: Path, project_root: Optional[Path] = None) -> Dict[str, Any]:
    """
    Get all pipeline state in one call - READ ONLY.

    Args:
        run_dir: Path to the run directory
        project_root: Optional path to project root for checkpoint reading

    Returns:
        Dict with all state data
    """
    run_state = read_run_state(run_dir)
    events = read_events(run_dir, limit=50)

    # Try to get current phase from checkpoints
    if project_root:
        run_id = run_dir.name
        ckpt_info = read_current_phase_from_checkpoints(project_root, run_id)
        if ckpt_info:
            if ckpt_info.get("phase") and not run_state.get("phase"):
                run_state["phase"] = ckpt_info["phase"]
            if ckpt_info.get("sprint_id") and not run_state.get("sprint_id"):
                run_state["sprint_id"] = ckpt_info["sprint_id"]
            if ckpt_info.get("run_id"):
                run_state.setdefault("run_id", ckpt_info["run_id"])

    # Try to get phase from last event
    if not run_state.get("phase") and events:
        last_event = events[-1]
        if last_event.get("phase"):
            run_state["phase"] = last_event["phase"]
        if last_event.get("run_id"):
            run_state["run_id"] = last_event["run_id"]
        if last_event.get("sprint_id"):
            run_state["sprint_id"] = last_event["sprint_id"]

    # Get Redis data
    redis_history = read_redis_history(limit=50)
    redis_state = read_redis_pipeline_state()

    # Merge Redis state
    if redis_state.get("sprint_current"):
        run_state.setdefault("sprint_id", redis_state["sprint_current"])
    if redis_state.get("sprint_phase") and not run_state.get("phase"):
        run_state["phase"] = redis_state["sprint_phase"]
    if redis_state.get("run_id"):
        run_state.setdefault("run_id", redis_state["run_id"])
    if redis_state.get("status"):
        run_state["status"] = redis_state["status"]

    return {
        "run_state": run_state,
        "heartbeats": read_heartbeats(run_dir),
        "events": events,
        "gates": read_gate_results(run_dir),
        "agents": read_agent_status(run_dir),
        "metrics": read_metrics(run_dir),
        "run_log": read_run_log(run_dir, limit=100),
        "redis_history": redis_history,
        "redis_state": redis_state,
    }
