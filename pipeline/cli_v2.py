"""Pipeline v2 CLI.

CLI unificado para o Pipeline v2.

Comandos:
    run      - Executa roadmap unificado com auto-resume (RECOMENDADO)
    roadmap  - Mostra status do roadmap unificado
    init     - Inicializa novo run
    start    - Inicia execução de sprints (legado)
    status   - Mostra estado atual
    sprint   - Executa sprint específico
    health   - Verifica saúde dos serviços
    spec     - Spec Kit v3.0 Ironclad (specs impossíveis de errar)

Uso:
    python -m pipeline.cli run              # Auto-resume do roadmap unificado
    python -m pipeline.cli run -o           # Com dashboards de observabilidade
    python -m pipeline.cli roadmap          # Ver status do roadmap
    python -m pipeline.cli start -s S00 -e S05  # Execução manual (legado)
    python -m pipeline.cli sprint S01       # Sprint específico

    # Spec Kit v3.0 Ironclad
    python -m pipeline.cli spec status                           # Status dos subsistemas
    python -m pipeline.cli spec mastigate -i roadmap.md          # Roadmap -> EARS spec
    python -m pipeline.cli spec validate -s spec.yml             # Validar com 9 gates
    python -m pipeline.cli spec process -i roadmap.md -o out.yml # Pipeline completo

O comando `run` é o RECOMENDADO para uso normal:
- Carrega roadmap unificado (HumanGR)
- Detecta automaticamente de onde parou (checkpoint)
- Continua execução de onde parou
- Inicia Run Master daemon automaticamente (infraestrutura)

IMPORTANTE: O pipeline roda automaticamente em modo DAEMON (background).
Use --foreground para rodar em primeiro plano (debug).

Author: Pipeline Autonomo Team
Version: 2.4.0 (2026-01-26)
"""

from __future__ import annotations

import argparse
import atexit
import logging
import os
import signal
import subprocess
import sys
import uuid
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

# Load .env file for environment variables
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# =============================================================================
# GHOST CODE INTEGRATION (2026-01-30): CrossSprintConsistencyChecker
# =============================================================================

# Lazy import for cross-sprint consistency checking
CONSISTENCY_CHECKER_AVAILABLE = False
CrossSprintConsistencyChecker = None

def _get_consistency_checker():
    """Lazy load CrossSprintConsistencyChecker to avoid import overhead."""
    global CONSISTENCY_CHECKER_AVAILABLE, CrossSprintConsistencyChecker

    if CrossSprintConsistencyChecker is not None:
        return CrossSprintConsistencyChecker

    try:
        from pipeline.spec_kit.consistency_checker import (
            CrossSprintConsistencyChecker as _Checker,
        )
        CrossSprintConsistencyChecker = _Checker
        CONSISTENCY_CHECKER_AVAILABLE = True
        return CrossSprintConsistencyChecker
    except ImportError as e:
        logger.debug(f"CrossSprintConsistencyChecker not available: {e}")
        CONSISTENCY_CHECKER_AVAILABLE = False
        return None


async def check_sprint_consistency(
    start_sprint: str,
    end_sprint: str,
    fail_on_blocking: bool = True,
) -> tuple[bool, dict]:
    """Check cross-sprint consistency before execution.

    This validates that specs across multiple sprints don't have conflicts
    that would cause issues during execution.

    Args:
        start_sprint: Starting sprint ID (e.g., "S00")
        end_sprint: Ending sprint ID (e.g., "S25")
        fail_on_blocking: If True, return False on blocking inconsistencies

    Returns:
        Tuple of (is_consistent, report_dict)
    """
    Checker = _get_consistency_checker()
    if Checker is None:
        logger.debug("Consistency checker not available, skipping check")
        return True, {"skipped": True, "reason": "checker_not_available"}

    try:
        # Parse sprint range
        start_num = int(start_sprint.replace("S", "").replace("H", ""))
        end_num = int(end_sprint.replace("S", "").replace("H", ""))

        if start_num >= end_num:
            # Single sprint or invalid range, no cross-sprint check needed
            return True, {"skipped": True, "reason": "single_sprint"}

        checker = Checker()
        report = await checker.check_pack_consistency(start_sprint, end_sprint)

        result_dict = report.to_dict()

        if report.is_consistent:
            return True, result_dict

        # Check for blocking inconsistencies
        blocking_count = result_dict.get("blocking_count", 0)

        if fail_on_blocking and blocking_count > 0:
            return False, result_dict

        # Warnings only, don't fail
        return True, result_dict

    except Exception as e:
        logger.warning(f"Consistency check failed: {e}")
        return True, {"skipped": True, "reason": str(e)}


def run_consistency_check_sync(
    start_sprint: str,
    end_sprint: str,
    fail_on_blocking: bool = True,
) -> tuple[bool, dict]:
    """Synchronous wrapper for check_sprint_consistency."""
    import asyncio

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're in an async context, create a new loop
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    check_sprint_consistency(start_sprint, end_sprint, fail_on_blocking)
                )
                return future.result(timeout=60)
        else:
            return loop.run_until_complete(
                check_sprint_consistency(start_sprint, end_sprint, fail_on_blocking)
            )
    except Exception as e:
        logger.warning(f"Sync consistency check wrapper failed: {e}")
        return True, {"skipped": True, "reason": str(e)}

load_dotenv()

# =============================================================================
# GHOST CODE INTEGRATION (2026-01-30): Environment Validation
# =============================================================================

# Lazy import for environment validation
ENV_VALIDATION_AVAILABLE = False
_env_validation_result = None

def _validate_environment(strict: bool = False) -> dict:
    """Lazy validate environment variables at startup.

    GAP-CFG-001 FIX: Validates critical environment variables.

    Args:
        strict: If True, fail on warnings too.

    Returns:
        Validation result dict with 'valid', 'missing_critical', 'missing_warning', etc.
    """
    global ENV_VALIDATION_AVAILABLE, _env_validation_result

    if _env_validation_result is not None:
        return _env_validation_result

    try:
        from pipeline.env_validation import (
            validate_env_vars,
            EnvValidationError,
            CRITICAL_ENV_VARS,
        )
        ENV_VALIDATION_AVAILABLE = True

        try:
            _env_validation_result = validate_env_vars(fail_on_warning=strict)
            return _env_validation_result
        except EnvValidationError as e:
            _env_validation_result = {
                "valid": False,
                "error": str(e),
                "missing_critical": e.missing_vars,
                "invalid": e.invalid_vars,
            }
            return _env_validation_result

    except ImportError as e:
        logger.debug(f"Environment validation module not available: {e}")
        ENV_VALIDATION_AVAILABLE = False
        _env_validation_result = {"valid": True, "skipped": True, "reason": "module_not_available"}
        return _env_validation_result


# =============================================================================
# GHOST CODE INTEGRATION (2026-01-30): Streaming Execution
# =============================================================================

# Lazy import for streaming
STREAMING_AVAILABLE = False
_streaming_workflow = None

def _run_pipeline_with_streaming(
    pipeline,
    sprint_ids: list = None,
    start_sprint: str = None,
    end_sprint: str = None,
) -> list:
    """Run pipeline with real-time streaming output.

    GHOST CODE INTEGRATION: Uses pipeline.langgraph.streaming module
    for real-time progress updates during execution.

    Args:
        pipeline: Pipeline instance
        sprint_ids: List of sprint IDs to run (takes precedence)
        start_sprint: Starting sprint (if sprint_ids not provided)
        end_sprint: Ending sprint (if sprint_ids not provided)

    Returns:
        List of sprint results
    """
    global STREAMING_AVAILABLE

    try:
        from pipeline.langgraph.streaming import (
            StreamHandler,
            StreamEventType,
            get_streaming_workflow,
            STREAMING_AVAILABLE as _STREAMING_AVAIL,
        )
        STREAMING_AVAILABLE = _STREAMING_AVAIL

        if not STREAMING_AVAILABLE:
            logger.warning("[Streaming] Streaming module not available, using normal execution")
            if sprint_ids:
                return pipeline.run(sprint_ids=sprint_ids)
            return pipeline.run(start_sprint=start_sprint, end_sprint=end_sprint)

        # Create stream handler for progress display
        handler = StreamHandler()

        def on_node_start(event):
            """Handle node start events."""
            node_id = event.get("node_id", "unknown")
            sprint_id = event.get("metadata", {}).get("sprint_id", "")
            print(f"  📡 [{sprint_id}] Starting: {node_id}")

        def on_node_end(event):
            """Handle node end events."""
            node_id = event.get("node_id", "unknown")
            status = event.get("data", {}).get("status", "completed")
            duration = event.get("data", {}).get("duration_ms", 0)
            icon = "✅" if status == "success" else "⚠️"
            print(f"  {icon} [{event.get('metadata', {}).get('sprint_id', '')}] {node_id}: {status} ({duration}ms)")

        def on_error(event):
            """Handle error events."""
            error = event.get("error", "Unknown error")
            print(f"  ❌ Error: {error}")

        # Register handlers
        handler.on(StreamEventType.NODE_START, on_node_start)
        handler.on(StreamEventType.NODE_END, on_node_end)
        handler.on(StreamEventType.ERROR, on_error)

        print("\n📡 Streaming mode: Real-time progress enabled\n")

        # Execute pipeline with streaming handler injected
        # Note: If the pipeline.run() doesn't natively support streaming,
        # we still run it but show what we can via callbacks
        import asyncio

        async def _run_with_stream():
            results = []
            sprints = sprint_ids if sprint_ids else [
                f"S{i:02d}" for i in range(
                    int(start_sprint[1:]) if start_sprint else 0,
                    int(end_sprint[1:]) + 1 if end_sprint else 63
                )
            ]

            for sprint_id in sprints:
                # Emit start event
                handler.emit(StreamEventType.NODE_START, {
                    "node_id": "sprint",
                    "metadata": {"sprint_id": sprint_id},
                })

                try:
                    # Run single sprint
                    import time
                    start_time = time.time()
                    result = pipeline.run(sprint_ids=[sprint_id])
                    duration_ms = int((time.time() - start_time) * 1000)

                    if result:
                        results.extend(result)
                        # Emit end event
                        handler.emit(StreamEventType.NODE_END, {
                            "node_id": "sprint",
                            "metadata": {"sprint_id": sprint_id},
                            "data": {
                                "status": result[0].status if result else "unknown",
                                "duration_ms": duration_ms,
                            },
                        })
                except Exception as e:
                    handler.emit(StreamEventType.ERROR, {
                        "node_id": "sprint",
                        "metadata": {"sprint_id": sprint_id},
                        "error": str(e),
                    })
                    # Continue to next sprint even on error
                    from dataclasses import dataclass
                    @dataclass
                    class ErrorResult:
                        sprint_id: str = sprint_id
                        status: str = "failed"
                        duration_seconds: float = 0.0
                        errors: list = None
                        def __post_init__(self):
                            self.errors = self.errors or [str(e)]
                    results.append(ErrorResult())

            return results

        # Run async streaming execution
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, _run_with_stream())
                    return future.result(timeout=7200)  # 2 hour timeout
            else:
                return loop.run_until_complete(_run_with_stream())
        except Exception as e:
            logger.warning(f"Async streaming execution failed, using sync: {e}")
            if sprint_ids:
                return pipeline.run(sprint_ids=sprint_ids)
            return pipeline.run(start_sprint=start_sprint, end_sprint=end_sprint)

    except ImportError as e:
        logger.warning(f"[Streaming] Module import failed: {e}")
        if sprint_ids:
            return pipeline.run(sprint_ids=sprint_ids)
        return pipeline.run(start_sprint=start_sprint, end_sprint=end_sprint)


# =============================================================================
# GHOST CODE INTEGRATION (2026-01-30): Health Module Constants
# =============================================================================

# Lazy import for health module constants
HEALTH_MODULE_AVAILABLE = False
_health_required_stacks = None

def _get_required_stacks() -> dict:
    """Get required stacks per operation from health module.

    Returns:
        Dict mapping operation names to required stack lists.
    """
    global HEALTH_MODULE_AVAILABLE, _health_required_stacks

    if _health_required_stacks is not None:
        return _health_required_stacks

    try:
        from pipeline.health import REQUIRED_STACKS
        HEALTH_MODULE_AVAILABLE = True
        _health_required_stacks = REQUIRED_STACKS
        return _health_required_stacks
    except ImportError as e:
        logger.debug(f"Health module not available: {e}")
        HEALTH_MODULE_AVAILABLE = False
        # Fallback to basic requirements
        _health_required_stacks = {
            "sprint_execution": ["redis", "crewai"],
            "gate_execution": ["redis", "crewai"],
        }
        return _health_required_stacks


# Enable Ollama chunking interceptor EARLY (before any CrewAI/ChromaDB imports)
# This patches ollama._client.Client.embed to auto-chunk long texts
def _setup_ollama_chunking():
    """Patch Ollama to auto-chunk long texts."""
    MAX_CHARS = 4000
    CHUNK_SIZE = 3000
    CHUNK_STEP = 1200

    def sliding_chunks(text):
        if len(text) <= CHUNK_SIZE:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            chunks.append(text[start:start + CHUNK_SIZE])
            start += CHUNK_STEP
            if start >= len(text) - 100:
                break
        return chunks

    def avg_embeddings(embeddings):
        import math
        if not embeddings:
            return []
        dim = len(embeddings[0])
        result = [sum(e[i] for e in embeddings) / len(embeddings) for i in range(dim)]
        mag = math.sqrt(sum(x*x for x in result))
        return [x/mag for x in result] if mag > 0 else result

    try:
        from ollama import Client as OllamaClient
        _original_embed = OllamaClient.embed

        def chunking_embed(self, model, input, **kwargs):
            texts = [input] if isinstance(input, str) else list(input)
            results = []
            for text in texts:
                if len(text) > MAX_CHARS:
                    chunks = sliding_chunks(text)
                    chunk_embs = []
                    for chunk in chunks:
                        resp = _original_embed(self, model, chunk, **kwargs)
                        if hasattr(resp, 'embeddings') and resp.embeddings:
                            chunk_embs.append(list(resp.embeddings[0]))
                    if chunk_embs:
                        results.append(avg_embeddings(chunk_embs))
                    else:
                        resp = _original_embed(self, model, text[:MAX_CHARS], **kwargs)
                        if hasattr(resp, 'embeddings') and resp.embeddings:
                            results.append(list(resp.embeddings[0]))
                else:
                    resp = _original_embed(self, model, text, **kwargs)
                    if hasattr(resp, 'embeddings') and resp.embeddings:
                        results.append(list(resp.embeddings[0]))

            class EmbedResponse:
                def __init__(self, embs, m):
                    self.embeddings = embs
                    self.model = m
                    self._data = {"embeddings": embs, "model": m}
                def __getitem__(self, key):
                    return self._data[key]
                def get(self, key, default=None):
                    return self._data.get(key, default)
            return EmbedResponse(results, model)

        OllamaClient.embed = chunking_embed
        print("INFO: Ollama chunking interceptor ENABLED")
    except Exception as e:
        print(f"WARNING: Ollama chunking setup failed: {e}")

# Lazy initialization flag for Ollama chunking
_ollama_chunking_initialized = False

def _ensure_ollama_chunking():
    """Lazy initialize Ollama chunking interceptor."""
    global _ollama_chunking_initialized
    if not _ollama_chunking_initialized:
        _setup_ollama_chunking()
        _ollama_chunking_initialized = True

# Suppress noisy LangSmith warnings (we use Langfuse for observability)
import warnings
warnings.filterwarnings("ignore", message=".*LangSmithUserError.*")
warnings.filterwarnings("ignore", message=".*org-scoped.*workspace.*")
logging.getLogger("langsmith").setLevel(logging.ERROR)
logging.getLogger("langsmith.client").setLevel(logging.ERROR)

# LAZY IMPORT: Only import heavy orchestrator when actually needed
# This makes `--help` fast (<1s instead of 30s)
if TYPE_CHECKING:
    from pipeline.orchestrator import PipelineConfig

# =============================================================================
# AUTO-DAEMONIZE (FIX 2026-01-26: Prevent orphaned/suspended processes)
# =============================================================================

def daemonize(log_file: Path, pid_file: Path, start_sprint: str, end_sprint: str) -> bool:
    """Start pipeline as daemon using subprocess (NOT fork).

    2026-01-27 FIX: Replaced os.fork() with subprocess.Popen to avoid
    "multi-threaded process forked" crash on macOS. Fork doesn't work
    well with Python threads (CrewAI, LangGraph create many threads).

    This function starts a NEW Python process with --foreground flag,
    which runs independently of the parent. The parent then exits.

    Args:
        log_file: Path to redirect stdout/stderr
        pid_file: Path to write PID for tracking
        start_sprint: Start sprint ID (e.g., "S00")
        end_sprint: End sprint ID (e.g., "S25")

    Returns:
        True if this is the parent process (should exit after subprocess starts)
        False should never be returned (subprocess runs independently)
    """
    # Ensure directories exist
    log_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.parent.mkdir(parents=True, exist_ok=True)

    # Find Python executable
    project_root = Path(__file__).parent.parent.parent
    python_path = project_root / ".venv" / "bin" / "python"
    if not python_path.exists():
        python_path = Path(sys.executable)

    # Build command - run same CLI but with --foreground flag
    cmd = [
        str(python_path),
        "-m", "pipeline.cli",
        "start",
        "--start", start_sprint,
        "--end", end_sprint,
        "--foreground",  # Run in foreground (no recursive daemonization)
    ]

    # Environment
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")
    env["PYTHONUNBUFFERED"] = "1"  # Force unbuffered output

    try:
        # Open log file for subprocess output
        log_fd = open(log_file, 'a')

        # Start subprocess - completely independent process
        # Using start_new_session=True to detach from terminal (replaces os.setsid)
        process = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            env=env,
            stdout=log_fd,
            stderr=log_fd,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # Detach from terminal (like os.setsid)
        )

        # Write PID file
        with open(pid_file, "w") as f:
            f.write(f"{process.pid}\n")

        # Give subprocess a moment to start and potentially fail
        import time
        time.sleep(0.5)

        # Check if process started successfully
        if process.poll() is not None:
            # Process already exited - something went wrong
            print(f"❌ Daemon process exited immediately with code {process.returncode}", file=sys.stderr)
            return True  # Parent should exit with error

        # Parent process - daemon started successfully
        return True

    except Exception as e:
        print(f"❌ Failed to start daemon: {e}", file=sys.stderr)
        sys.exit(1)


def get_run_log_path(start_sprint: str, end_sprint: str) -> tuple[Path, Path, str]:
    """Generate paths for run log and PID file.

    Returns:
        Tuple of (log_file, pid_file, run_id)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"lg_{timestamp}_{uuid.uuid4().hex[:8]}"

    run_dir = Path("out/runs") / run_id
    log_file = run_dir / "run.log"
    pid_file = run_dir / "pipeline.pid"

    return log_file, pid_file, run_id


# =============================================================================
# PIPELINE WRAPPER (replaces deprecated orchestrator.py)
# =============================================================================

class PipelineState:
    """Pipeline state container."""
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.status = "initialized"
        self.sprint_id = ""
        self.phase = "INIT"


class PipelineWrapper:
    """Pipeline wrapper that replaces deprecated orchestrator.

    Provides the same interface but uses LangGraph bridge internally.
    """

    # Sentinel to distinguish "not tried" from "failed"
    _UNINITIALIZED = object()

    def __init__(self):
        self._state: PipelineState | None = None
        self._active_runs: dict = {}
        self._initialized = False
        # Lazy-loaded stack references (use sentinel to avoid retry on failure)
        self._redis = self._UNINITIALIZED
        self._crewai = self._UNINITIALIZED
        self._langfuse = self._UNINITIALIZED
        self._letta = self._UNINITIALIZED

    def init(self, run_id: str | None = None, resume: bool = False) -> PipelineState:
        """Initialize pipeline state."""
        self._state = PipelineState(run_id=run_id or str(uuid.uuid4())[:8])
        self._state.status = "running" if resume else "initialized"
        self._initialized = True
        return self._state

    def status(self) -> PipelineState | None:
        """Get current state."""
        return self._state

    def get_active_run_for_sprint(self, sprint_id: str):
        return self._active_runs.get(sprint_id)

    def save_active_run_for_sprint(self, sprint_id: str, run_id: str):
        self._active_runs[sprint_id] = {"run_id": run_id, "sprint_id": sprint_id}

    def clear_active_run_for_sprint(self, sprint_id: str):
        self._active_runs.pop(sprint_id, None)

    def run_sprint(self, sprint_id: str):
        """Run a single sprint using LangGraph bridge."""
        from pipeline.langgraph.bridge import run_sprint_sync
        return run_sprint_sync(sprint_id=sprint_id)

    def run(self, start_sprint: str = "S00", end_sprint: str = "S62", sprint_ids: list | None = None):
        """Run sprints using LangGraph bridge."""
        from pipeline.langgraph.bridge import run_sprint_sync
        results = []
        if sprint_ids:
            for sid in sprint_ids:
                results.append(run_sprint_sync(sprint_id=sid))
        else:
            start_num = int(start_sprint.replace("S", "").replace("H", ""))
            end_num = int(end_sprint.replace("S", "").replace("H", ""))
            for i in range(start_num, end_num + 1):
                results.append(run_sprint_sync(sprint_id=f"S{i:02d}"))
        return results

    @property
    def redis(self):
        if self._redis is self._UNINITIALIZED:
            try:
                from pipeline.redis_client import get_redis_client
                self._redis = get_redis_client()
            except Exception:
                self._redis = None  # Mark as failed, don't retry
        return self._redis

    @property
    def crewai(self):
        if self._crewai is self._UNINITIALIZED:
            try:
                from crewai import Agent, Task, Crew
                # CrewAI is available if import succeeds
                self._crewai = True
            except ImportError:
                self._crewai = False  # Mark as failed, don't retry
        return self._crewai

    @property
    def langfuse(self):
        if self._langfuse is self._UNINITIALIZED:
            try:
                from pipeline.langfuse_client import get_langfuse_client
                self._langfuse = get_langfuse_client()
            except Exception:
                self._langfuse = None  # Mark as failed, don't retry
        return self._langfuse

    @property
    def letta(self):
        if self._letta is self._UNINITIALIZED:
            try:
                from pipeline.letta_client import get_letta_client
                self._letta = get_letta_client()
            except Exception:
                self._letta = None  # Mark as failed, don't retry
        return self._letta


_pipeline_instance: PipelineWrapper | None = None

def get_pipeline(config=None) -> PipelineWrapper:
    """Get singleton pipeline wrapper."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = PipelineWrapper()
    return _pipeline_instance

def get_run_state() -> PipelineWrapper:
    """Alias for get_pipeline() for backwards compatibility."""
    return get_pipeline()


# =============================================================================
# OBSERVABILITY DASHBOARDS
# =============================================================================

OBSERVABILITY_URLS = {
    "langfuse": "http://localhost:3000",
    "qdrant": "http://localhost:6333/dashboard",
    "redisinsight": "http://localhost:5540",  # Redis + FalkorDB visualization
    "neo4j": "http://localhost:7474",  # Neo4j Browser (graph algorithms)
}


# =============================================================================
# RUN MASTER DAEMON INTEGRATION
# =============================================================================

_run_master_thread = None
_run_master_stop_event = None


def start_run_master_daemon() -> bool:
    """Start the Run Master daemon in a background thread.

    The Run Master daemon:
    - Publishes heartbeat every 10 seconds
    - Listens for infrastructure alerts from the pipeline
    - Handles Docker, Redis, disk, memory, network issues
    - NEVER modifies code or makes business decisions

    Returns:
        True if daemon started successfully
    """
    global _run_master_thread, _run_master_stop_event

    if _run_master_thread is not None and _run_master_thread.is_alive():
        logging.getLogger(__name__).debug("Run Master daemon already running")
        return True

    import asyncio
    import threading

    _run_master_stop_event = threading.Event()

    def run_daemon():
        """Run the daemon in a separate thread with its own event loop."""
        try:
            from pipeline.resilience.infra_handler import (
                publish_run_master_heartbeat,
                get_pending_alerts,
                publish_resolution,
            )
            from pipeline.run_master import handle_infra_alert
        except ImportError as e:
            logging.getLogger(__name__).warning(f"Run Master daemon not available: {e}")
            return

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def daemon_loop():
            heartbeat_interval = 10  # seconds
            alert_check_interval = 2  # seconds
            last_heartbeat = 0

            while not _run_master_stop_event.is_set():
                try:
                    import time
                    now = time.time()

                    # Publish heartbeat
                    if now - last_heartbeat >= heartbeat_interval:
                        await publish_run_master_heartbeat()
                        last_heartbeat = now

                    # Check for alerts
                    alerts = await get_pending_alerts()
                    for alert in alerts:
                        resolution = await handle_infra_alert(
                            alert_id=alert.alert_id,
                            category=alert.category,
                            service_name=alert.service_name,
                            error_message=alert.error_message,
                            context=alert.context,
                        )
                        await publish_resolution(resolution)

                    await asyncio.sleep(alert_check_interval)

                except Exception as e:
                    logging.getLogger(__name__).debug(f"Run Master daemon error: {e}")
                    await asyncio.sleep(5)  # Back off on error

        try:
            loop.run_until_complete(daemon_loop())
        finally:
            loop.close()

    _run_master_thread = threading.Thread(target=run_daemon, daemon=True, name="RunMasterDaemon")
    _run_master_thread.start()

    logging.getLogger(__name__).info("🔧 Run Master daemon started (infrastructure monitoring)")
    return True


def stop_run_master_daemon() -> None:
    """Stop the Run Master daemon."""
    global _run_master_thread, _run_master_stop_event

    if _run_master_stop_event is not None:
        _run_master_stop_event.set()

    if _run_master_thread is not None and _run_master_thread.is_alive():
        _run_master_thread.join(timeout=5)
        logging.getLogger(__name__).debug("Run Master daemon stopped")


# =============================================================================
# DOCKER SERVICES AUTO-START
# =============================================================================

def _get_running_containers() -> set[str]:
    """Get set of running container names.

    Consolidates docker ps calls to avoid redundant subprocess invocations.
    """
    import subprocess

    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            return set(result.stdout.strip().split("\n"))
    except Exception as e:
        logger.debug(f"REDIS: Redis operation failed: {e}")
    return set()


def ensure_docker_services() -> bool:
    """Ensure Docker services are running before pipeline execution.

    Checks if critical services (redis, qdrant, langfuse, etc.) are running.
    If not, attempts to start them with docker-compose up -d.

    Returns:
        True if all services are running, False if failed to start.
    """
    import subprocess
    import shutil
    import time

    logger = logging.getLogger(__name__)

    # Check if docker is available
    if not shutil.which("docker"):
        logger.warning("Docker not found in PATH")
        return False

    # Critical services that must be running
    critical_services = ["redis", "qdrant", "langfuse", "grafana"]
    optional_services = ["ollama", "falkordb", "letta", "llm-guard"]

    try:
        # Get running containers (consolidated call)
        running = _get_running_containers()

        # Check critical services
        missing_critical = [s for s in critical_services if s not in running]
        missing_optional = [s for s in optional_services if s not in running]

        if not missing_critical and not missing_optional:
            logger.info("✅ All Docker services already running")
            return True

        # Report what's missing
        if missing_critical:
            print(f"⚠️  Missing critical services: {', '.join(missing_critical)}")
        if missing_optional:
            print(f"ℹ️  Missing optional services: {', '.join(missing_optional)}")

        # Start services with docker-compose
        print("🐳 Starting Docker services...")

        # Find docker-compose.yml
        compose_file = Path("docker-compose.yml")
        if not compose_file.exists():
            compose_file = Path(__file__).parent.parent.parent / "docker-compose.yml"

        if not compose_file.exists():
            logger.error("docker-compose.yml not found")
            return False

        # Run docker-compose up -d
        result = subprocess.run(
            ["docker-compose", "-f", str(compose_file), "up", "-d"],
            capture_output=True,
            text=True,
            timeout=120  # 2 minutes timeout for starting services
        )

        if result.returncode != 0:
            # Try docker compose (newer syntax)
            result = subprocess.run(
                ["docker", "compose", "-f", str(compose_file), "up", "-d"],
                capture_output=True,
                text=True,
                timeout=120
            )

        if result.returncode != 0:
            logger.error(f"Failed to start Docker services: {result.stderr}")
            return False

        print("✅ Docker services started")

        # Poll for services to be ready (instead of fixed sleep)
        print("⏳ Waiting for services to be ready...")
        for attempt in range(10):  # 10 attempts, 1 second apart
            time.sleep(1)
            running = _get_running_containers()
            if all(svc in running for svc in critical_services):
                print("✅ All critical services running")
                return True

        # Final check after polling
        running = _get_running_containers()
        still_missing = [s for s in critical_services if s not in running]
        if still_missing:
            logger.error(f"Critical services failed to start: {still_missing}")
            return False

        print("✅ All critical services running")
        return True

    except subprocess.TimeoutExpired:
        logger.error("Docker command timed out")
        return False
    except FileNotFoundError:
        logger.error("docker-compose not found")
        return False
    except Exception as e:
        logger.error(f"Failed to check/start Docker services: {e}")
        return False


def open_observability_dashboards(dashboards: list[str] | None = None) -> None:
    """Open observability dashboards in browser.

    Args:
        dashboards: List of dashboard names to open. If None, opens all.
    """
    to_open = dashboards or list(OBSERVABILITY_URLS.keys())

    print()
    print("Opening observability dashboards...")
    for name in to_open:
        if name in OBSERVABILITY_URLS:
            url = OBSERVABILITY_URLS[name]
            print(f"  -> {name}: {url}")
            webbrowser.open(url)


def warmup_ollama() -> bool:
    """Warm up Ollama for embedding model before pipeline execution.

    Ollama cold starts can take 60-180 seconds on first request.
    This ensures the model is loaded before RAG operations start.

    Returns:
        True if warmup successful or Ollama not needed, False if critical failure.
    """
    # Initialize Ollama chunking interceptor on first use
    _ensure_ollama_chunking()

    logger = logging.getLogger(__name__)
    print("🔥 Warming up Ollama embedding model...")

    try:
        from pipeline.ollama_client import get_ollama_client

        client = get_ollama_client()

        # Check if Ollama is running
        if not client.is_available():
            print("⚠️  Ollama not available - RAG will use fallback or skip embeddings")
            logger.warning("Ollama not available for warmup")
            return True  # Not critical, continue

        # Perform warmup with longer timeout
        if client.warmup():
            print("✅ Ollama embedding model ready")
            return True
        else:
            print("⚠️  Ollama warmup failed - RAG may be slow on first request")
            logger.warning("Ollama warmup failed, continuing anyway")
            return True  # Not critical, continue

    except Exception as e:
        logger.warning(f"Ollama warmup error: {e}")
        print(f"⚠️  Ollama warmup skipped: {e}")
        return True  # Not critical, continue


def cmd_init(args: argparse.Namespace) -> int:
    """Comando: init - Inicializa novo run ou resume existente."""
    config = PipelineConfig(
        run_dir=Path(args.run_dir) if args.run_dir else Path("out/pipeline"),
        verbose=args.verbose,
    )

    # Validate resume requires run_id
    resume = getattr(args, 'resume', False)
    if resume and not args.run_id:
        print("ERROR: --resume requires --run-id")
        return 1

    pipeline = get_pipeline(config)
    state = pipeline.init(run_id=args.run_id, resume=resume)

    action = "Resumed" if resume else "Initialized"
    print(f"Pipeline {action.lower()}:")
    print(f"  Run ID: {state.run_id}")
    print(f"  Status: {state.status}")
    print(f"  Run Dir: {config.run_dir / state.run_id}")
    if state.sprints_completed:
        print(f"  Sprints completed: {state.sprints_completed}")
    if state.sprints_failed:
        print(f"  Sprints failed: {state.sprints_failed}")

    # Auto-open observability dashboards
    if getattr(args, 'open_dashboards', False):
        open_observability_dashboards()

    return 0


def cmd_start(args: argparse.Namespace) -> int:
    """Comando: start - Inicia execução de sprints com auto-resume.

    Por padrão, roda em modo DAEMON (background) para evitar suspensão
    quando o terminal fecha. Use --foreground para debug.
    """
    import fcntl
    import time

    # GHOST CODE INTEGRATION (2026-01-30): Environment validation
    validate_env = getattr(args, 'validate_env', False)
    strict_env = getattr(args, 'strict_env', False)

    if validate_env:
        print("🔍 Validating environment variables (GAP-CFG-001)...")
        result = _validate_environment(strict=strict_env)

        if not result.get("valid", True):
            print("❌ Environment validation FAILED:")
            if result.get("missing_critical"):
                print(f"   Missing critical vars: {', '.join(result['missing_critical'])}")
            if result.get("invalid"):
                for var, reason in result["invalid"]:
                    print(f"   Invalid: {var} - {reason}")
            print("\n   Fix your .env file and retry.")
            return 1

        if result.get("skipped"):
            print(f"⚠️  Environment validation skipped: {result.get('reason', 'unknown')}")
        else:
            warnings = result.get("missing_warning", [])
            if warnings:
                print(f"⚠️  Missing optional vars: {', '.join(warnings)}")
            print("✅ Environment validation passed")

    # Check for foreground mode
    foreground = getattr(args, 'foreground', False)

    # Auto-daemonize unless --foreground is specified
    if not foreground:
        log_file, pid_file, run_id = get_run_log_path(args.start, args.end)

        print("=" * 60)
        print("🚀 PIPELINE STARTING IN DAEMON MODE")
        print("=" * 60)
        print(f"  Sprints: {args.start} -> {args.end}")
        print(f"  Run ID: {run_id}")
        print(f"  Log: {log_file}")
        print(f"  PID file: {pid_file}")
        print()
        print("  To monitor:")
        print(f"    tail -f {log_file}")
        print()
        print("  To stop:")
        print(f"    kill $(cat {pid_file})")
        print()
        print("  Cockpit dashboard:")
        print("    http://localhost:5001")
        print("=" * 60)

        # Ensure Docker services are running BEFORE starting daemon
        # (so user sees any Docker errors)
        if not ensure_docker_services():
            print("❌ Failed to start Docker services. Cannot proceed.")
            print("   Try manually: docker-compose up -d")
            return 1

        # Start daemon subprocess (2026-01-27: uses subprocess.Popen, not fork)
        daemonize(log_file, pid_file, args.start, args.end)
        # Parent always exits here - daemon runs in separate process
        print(f"\n✅ Pipeline daemon started (PID in {pid_file})")
        return 0

    else:
        # Foreground mode - run normally (for debugging)
        print("⚠️  Running in FOREGROUND mode (--foreground)")
        print("   Pipeline will stop if terminal closes!")
        print()

        # Ensure Docker services are running before starting pipeline
        if not ensure_docker_services():
            print("❌ Failed to start Docker services. Cannot proceed.")
            print("   Try manually: docker-compose up -d")
            return 1

    # Warm up Ollama to avoid cold start timeouts in RAG
    warmup_ollama()

    # GHOST CODE INTEGRATION (2026-01-30): Cross-sprint consistency check
    # Verifies specs don't have conflicts before execution
    skip_consistency = getattr(args, 'skip_consistency_check', False)
    if not skip_consistency:
        start_num = int(args.start.replace("S", "").replace("H", ""))
        end_num = int(args.end.replace("S", "").replace("H", ""))

        if end_num - start_num > 1:  # Only check if >2 sprints
            print("🔍 Running cross-sprint consistency check...")
            is_consistent, report = run_consistency_check_sync(
                args.start, args.end, fail_on_blocking=True
            )

            if report.get("skipped"):
                print(f"  ⏭️  Skipped: {report.get('reason', 'unknown')}")
            elif is_consistent:
                total_specs = report.get("total_specs", 0)
                print(f"  ✅ Consistent ({total_specs} specs verified across sprints)")
            else:
                # Inconsistencies found
                blocking_count = report.get("blocking_count", 0)
                total_inc = report.get("total_inconsistencies", 0)
                print(f"  ⚠️  {total_inc} inconsistencies found ({blocking_count} blocking)")

                # Show first few inconsistencies
                for inc in report.get("inconsistencies", [])[:3]:
                    print(f"      └─ {inc.get('sprint_1')}/{inc.get('spec_id_1')} vs "
                          f"{inc.get('sprint_2')}/{inc.get('spec_id_2')}: {inc.get('description', '')[:50]}")

                if blocking_count > 0:
                    print()
                    print("❌ Blocking inconsistencies detected!")
                    print("   Fix specs or use --skip-consistency-check to bypass.")
                    return 1
            print()

    # LOCK: Prevent concurrent pipeline runs (FIX 2026-01-23)
    lock_file = Path("out/.pipeline.lock")
    lock_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        lock_fd = open(lock_file, 'w')
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Write PID and timestamp to lock file
        lock_fd.write(f"PID: {os.getpid()}\nStarted: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        lock_fd.flush()
    except BlockingIOError:
        # Another instance is running
        try:
            with open(lock_file, 'r') as f:
                lock_info = f.read().strip()
        except Exception:
            lock_info = "unknown"
        print("=" * 60)
        print("ERROR: Another pipeline instance is already running!")
        print("=" * 60)
        print(f"Lock info: {lock_info}")
        print()
        print("Options:")
        print("  1. Wait for the other instance to finish")
        print("  2. Kill the other process if it's stuck")
        print(f"  3. Remove lock file: rm {lock_file}")
        print()
        return 1

    pipeline = get_pipeline()

    # Start Run Master daemon for infrastructure monitoring
    start_run_master_daemon()

    # Start metrics v2 publisher (direct Redis -> Grafana)
    try:
        from pipeline.metrics_v2 import get_metrics_publisher
        metrics = get_metrics_publisher()
        print("📊 Metrics v2 publisher initialized (Redis -> Grafana)")
    except Exception as e:
        print(f"⚠️  Metrics v2 failed to initialize: {e}")
        metrics = None

    # GHOST CODE INTEGRATION (2026-01-30): Check for streaming mode
    stream_enabled = getattr(args, 'stream', False)
    if stream_enabled:
        try:
            from pipeline.langgraph.streaming import (
                get_streaming_workflow,
                LANGGRAPH_AVAILABLE as STREAMING_LANGGRAPH_AVAILABLE,
            )
            if STREAMING_LANGGRAPH_AVAILABLE:
                print("📡 Streaming mode ENABLED (real-time progress)")
            else:
                print("⚠️  Streaming requested but LangGraph not available")
                stream_enabled = False
        except ImportError as e:
            print(f"⚠️  Streaming module not available: {e}")
            stream_enabled = False

    print(f"Starting pipeline execution...")
    print(f"  From: {args.start}")
    print(f"  To: {args.end}")

    # 2026-01-10: Parse --include for additional sprints (hardening, hotfixes, etc)
    include_sprints = []
    if hasattr(args, 'include') and args.include:
        include_sprints = [s.strip() for s in args.include.split(",")]
        print(f"  Include: {', '.join(include_sprints)}")
    print()

    # Check for active runs (auto-resume)
    start_num = int(args.start[1:])
    end_num = int(args.end[1:])

    active_runs = {}
    for sprint_num in range(start_num, end_num + 1):
        sprint_id = f"S{sprint_num:02d}"
        active = pipeline.get_active_run_for_sprint(sprint_id)
        if active:
            active_runs[sprint_id] = active

    if active_runs:
        print("=" * 60)
        print("ACTIVE RUNS DETECTED - AUTO RESUME")
        print("=" * 60)
        for sprint_id, info in active_runs.items():
            print(f"  {sprint_id}: {info['run_id']} (started {info.get('started_at', 'unknown')})")
        print()
        print("Pipeline will resume from existing state.")
        print("To start fresh, run: python -m pipeline.cli clear-sprint <sprint_id>")
        print()

        # Use the first active run's run_id
        first_active = list(active_runs.values())[0]
        resume_run_id = first_active["run_id"]

        # Initialize with the existing run_id
        pipeline.init(run_id=resume_run_id, resume=True)
    else:
        # No active runs - start fresh
        if pipeline.status() is None:
            pipeline.init()

        # Save active run for each sprint we're about to execute
        run_id = pipeline._state.run_id
        for sprint_num in range(start_num, end_num + 1):
            sprint_id = f"S{sprint_num:02d}"
            pipeline.save_active_run_for_sprint(sprint_id, run_id)

    # Auto-open observability dashboards
    if getattr(args, 'open_dashboards', False):
        open_observability_dashboards()

    # FIX 2026-01-25: Initialize metrics v2 with run info
    try:
        if metrics:
            # Build sprint list for metrics
            sprint_list = [f"S{n:02d}" for n in range(start_num, end_num + 1)]
            if include_sprints:
                sprint_list.extend(include_sprints)

            run_id = pipeline._state.run_id if pipeline._state else "unknown"
            metrics.start_run(
                run_id=run_id,
                start_sprint=args.start,
                end_sprint=args.end,
                sprint_list=sprint_list,
            )
            print(f"📊 Metrics: Run {run_id} started ({len(sprint_list)} sprints)")
    except Exception as e:
        print(f"⚠️  Metrics start_run failed: {e}")

    # 2026-01-10: Build sprint list with --include sprints injected at right position
    if include_sprints:
        # Build custom sprint list
        sprint_ids = []
        for sprint_num in range(start_num, end_num + 1):
            sprint_id = f"S{sprint_num:02d}"

            # Check if any include sprints should run BEFORE this sprint
            # Convention: S05H runs after S05, before S06
            for inc in include_sprints[:]:  # Copy to allow removal
                # Extract base sprint number from include (e.g., S05H -> 5)
                try:
                    inc_base = int(''.join(c for c in inc[1:] if c.isdigit()))
                    if inc_base == sprint_num - 1:
                        # This include sprint runs before current sprint
                        sprint_ids.append(inc)
                        include_sprints.remove(inc)
                        print(f"  Injecting {inc} before {sprint_id}")
                except ValueError as e:
                    logger.debug(f"SILENT_FAIL: Operation failed silently: {e}")

            sprint_ids.append(sprint_id)

        # Add any remaining include sprints at the end
        for inc in include_sprints:
            sprint_ids.append(inc)
            print(f"  Appending {inc} at end")

        print(f"\n  Sprint order: {' -> '.join(sprint_ids)}\n")

        # GHOST CODE INTEGRATION (2026-01-30): Use streaming execution if enabled
        if stream_enabled:
            results = _run_pipeline_with_streaming(pipeline, sprint_ids=sprint_ids)
        else:
            results = pipeline.run(sprint_ids=sprint_ids)
    else:
        # GHOST CODE INTEGRATION (2026-01-30): Use streaming execution if enabled
        if stream_enabled:
            results = _run_pipeline_with_streaming(
                pipeline, start_sprint=args.start, end_sprint=args.end
            )
        else:
            results = pipeline.run(start_sprint=args.start, end_sprint=args.end)

    # Sumário - "skipped" means CEO already signed, counts as success
    success = sum(1 for r in results if r.status in ("success", "skipped"))
    failed = sum(1 for r in results if r.status == "failed")

    print()
    print("=" * 60)
    print("EXECUTION SUMMARY")
    print("=" * 60)
    print(f"  Total sprints: {len(results)}")
    print(f"  Success: {success}")
    print(f"  Failed: {failed}")
    print()

    for result in results:
        # "skipped" means CEO already signed - treat as success
        is_success = result.status in ("success", "skipped")
        status_icon = "✅" if is_success else "❌"
        display_status = "completed (CEO signed)" if result.status == "skipped" else result.status
        print(f"  {status_icon} {result.sprint_id}: {display_status} ({result.duration_seconds:.1f}s)")
        if result.errors:
            for error in result.errors[:3]:  # Limitar a 3 erros
                print(f"      └─ {error[:80]}")

        # Clear active run on success (including skipped = already complete)
        if is_success:
            pipeline.clear_active_run_for_sprint(result.sprint_id)

    # Release the lock (FIX 2026-01-23)
    try:
        import fcntl
        lock_file = Path("out/.pipeline.lock")
        if lock_file.exists():
            lock_file.unlink()
    except Exception as e:
        logger.debug(f"PARSE: Failed to parse content: {e}")

    # Stop Run Master daemon
    stop_run_master_daemon()

    return 0 if failed == 0 else 1


def cmd_clear_sprint(args: argparse.Namespace) -> int:
    """Comando: clear-sprint - Limpa estado de sprint para rodar do zero."""
    pipeline = get_pipeline()

    sprint_id = args.sprint_id

    # Check if there's an active run
    active = pipeline.get_active_run_for_sprint(sprint_id)
    if not active:
        print(f"No active run found for {sprint_id}")
        return 0

    print(f"Active run for {sprint_id}:")
    print(f"  Run ID: {active['run_id']}")
    print(f"  Started: {active.get('started_at', 'unknown')}")
    print()

    # Confirm unless --force
    if not getattr(args, 'force', False):
        confirm = input(f"Clear active run for {sprint_id}? This will start fresh. [y/N] ")
        if confirm.lower() != 'y':
            print("Cancelled.")
            return 1

    # Clear the active run file
    pipeline.clear_active_run_for_sprint(sprint_id)
    print(f"Cleared active run for {sprint_id}")

    # Optionally clear Redis state for this sprint
    if getattr(args, 'clear_redis', False) and pipeline.redis:
        run_id = active['run_id']
        # Clear signoffs
        pattern = f"signoff:{run_id}:{sprint_id}:*"
        keys = list(pipeline.redis.client.scan_iter(match=pattern))
        if keys:
            pipeline.redis.client.delete(*keys)
            print(f"Cleared {len(keys)} signoff keys from Redis")

        # Clear handoffs
        pattern = f"handoff:{run_id}:{sprint_id}:*"
        keys = list(pipeline.redis.client.scan_iter(match=pattern))
        if keys:
            pipeline.redis.client.delete(*keys)
            print(f"Cleared {len(keys)} handoff keys from Redis")

    print(f"\n{sprint_id} is now ready to run from scratch.")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Comando: status - Mostra estado atual."""
    pipeline = get_pipeline()
    state = pipeline.status()

    if state is None:
        print("No active pipeline run.")
        print("Use 'pipeline init' to start a new run.")
        return 1

    print("=" * 60)
    print("PIPELINE STATUS")
    print("=" * 60)
    print(f"  Run ID: {state.run_id}")
    print(f"  Status: {state.status}")
    print(f"  Current Sprint: {state.current_sprint or 'None'}")
    print(f"  Current Phase: {state.current_phase or 'None'}")
    print()
    print(f"  Sprints Completed: {len(state.sprints_completed)}")
    if state.sprints_completed:
        print(f"    {', '.join(state.sprints_completed[-5:])}")  # Últimos 5
    print(f"  Sprints Failed: {len(state.sprints_failed)}")
    if state.sprints_failed:
        print(f"    {', '.join(state.sprints_failed)}")
    print()
    print(f"  Started: {state.started_at}")
    print(f"  Completed: {state.completed_at or 'In progress'}")

    if state.error:
        print()
        print(f"  Error: {state.error}")

    return 0


def cmd_sprint(args: argparse.Namespace) -> int:
    """Comando: sprint - Executa sprint específico usando LangGraph.

    NOTA: Este comando agora usa LangGraph como engine principal.
    O orchestrator legado foi descontinuado.
    """
    from pipeline.langgraph.bridge import run_sprint_sync

    print(f"Executing sprint {args.sprint_id} with LangGraph...")
    print()

    result = run_sprint_sync(sprint_id=args.sprint_id)

    # FIX 2026-01-23: SprintResult is a dataclass, not a dict
    # Access attributes directly instead of using .get()
    status = getattr(result, "status", "unknown")
    is_success = status in ("completed", "success")
    status_icon = "✅" if is_success else "❌"

    print()
    print("=" * 60)
    print(f"SPRINT {args.sprint_id} RESULT (LangGraph)")
    print("=" * 60)
    print(f"  {status_icon} Status: {status}")
    print(f"  Run ID: {getattr(result, 'run_id', 'N/A')}")
    print(f"  Duration: {getattr(result, 'duration_seconds', 0):.2f}s")

    gates_passed = getattr(result, "gates_passed", []) or []
    gates_failed = getattr(result, "gates_failed", []) or []

    if gates_passed:
        print(f"  Gates Passed: {', '.join(gates_passed)}")
    if gates_failed:
        print(f"  Gates Failed: {', '.join(gates_failed)}")

    errors = getattr(result, "errors", []) or []
    if errors:
        print()
        print("  Errors:")
        for error in errors[:5]:  # Limit to 5 errors
            error_msg = error.get("error", str(error)) if isinstance(error, dict) else str(error)
            print(f"    - {str(error_msg)[:100]}")

    return 0 if is_success else 1


def cmd_run(args: argparse.Namespace) -> int:
    """Comando: run - Executa roadmap unificado com auto-resume.

    Este é o comando RECOMENDADO para execução do pipeline.
    Ele automaticamente:
    - Carrega o roadmap unificado (HumanGR)
    - Detecta de onde parou (checkpoint)
    - Continua a execução de onde parou
    - Sincroniza progresso com GitHub Projects (se configurado)
    - Usa LangGraph control plane para execução (2026-01-21)

    Uso:
        python -m pipeline.cli run
        python -m pipeline.cli run -o  # com dashboards
    """
    # Ensure Docker services are running before starting pipeline
    if not ensure_docker_services():
        print("❌ Failed to start Docker services. Cannot proceed.")
        print("   Try manually: docker-compose up -d")
        return 1

    from pipeline.roadmap import get_roadmap, get_execution_plan
    from pipeline.gh_projects import get_gh_projects_sync
    from pipeline.langgraph.bridge import run_sprint_sync

    # Start Run Master daemon for infrastructure monitoring
    start_run_master_daemon()

    roadmap = get_roadmap()
    gh_sync = get_gh_projects_sync()

    print("=" * 60)
    print("UNIFIED ROADMAP - AUTO RESUME")
    print("=" * 60)

    # Get execution plan
    plan = get_execution_plan("all")

    print(f"  Total sprints: {plan.total_sprints}")
    print(f"  Completed: {len(plan.completed_sprints)} ({plan.progress_percent:.1f}%)")
    print(f"  Pending: {len(plan.pending_sprints)}")
    print(f"  GitHub Projects: {'Enabled' if gh_sync.enabled else 'Disabled'}")
    print()

    if plan.completed_sprints:
        last_5 = plan.completed_sprints[-5:]
        print(f"  Last completed: {', '.join(last_5)}")

    if not plan.next_sprint:
        print()
        print("🎉 ALL SPRINTS COMPLETE!")
        print("  The entire roadmap has been executed.")
        return 0

    print(f"  Next sprint: {plan.next_sprint}")
    print()

    # Show next 5 sprints to execute
    if plan.pending_sprints:
        next_5 = plan.pending_sprints[:5]
        print(f"  Execution queue: {' -> '.join(next_5)}{'...' if len(plan.pending_sprints) > 5 else ''}")
    print()

    # Auto-open dashboards if requested
    if getattr(args, 'open_dashboards', False):
        open_observability_dashboards()

    # Get pipeline wrapper (replaces deprecated orchestrator)
    pipeline = get_pipeline()
    if pipeline.status() is None:
        pipeline.init()

    # Execute sprints from the unified roadmap
    results = []
    for sprint_id in plan.pending_sprints:
        print(f"\nExecuting {sprint_id}...")

        # Check if this sprint has an active run
        active = pipeline.get_active_run_for_sprint(sprint_id)
        if active:
            print(f"  Resuming from run: {active['run_id']}")
            pipeline.init(run_id=active['run_id'], resume=True)
        else:
            # Save active run for tracking
            pipeline.save_active_run_for_sprint(sprint_id, pipeline._state.run_id)

        # Sync: Sprint starting
        run_id = pipeline._state.run_id if pipeline._state else ""
        gh_sync.on_sprint_start(sprint_id, run_id=run_id)

        try:
            # Use LangGraph control plane for sprint execution
            result = run_sprint_sync(sprint_id=sprint_id, pipeline=pipeline)
            results.append(result)

            # Display result
            is_success = result.status in ("success", "skipped")
            status_icon = "✅" if is_success else "❌"
            display_status = "completed (CEO signed)" if result.status == "skipped" else result.status
            print(f"  {status_icon} {sprint_id}: {display_status} ({result.duration_seconds:.1f}s)")

            # Clear active run on success
            if is_success:
                pipeline.clear_active_run_for_sprint(sprint_id)
                # Save checkpoint
                completed = plan.completed_sprints + [sprint_id]
                roadmap.save_checkpoint("all", completed)

                # Sync: Sprint completed
                gh_sync.on_sprint_complete(
                    sprint_id,
                    duration_seconds=result.duration_seconds,
                    gates_passed=result.gates_passed,
                )
            else:
                # Sync: Sprint failed
                error_msg = result.errors[0] if result.errors else "Unknown error"
                gh_sync.on_sprint_fail(
                    sprint_id,
                    error=error_msg,
                    gates_failed=result.gates_failed,
                    duration_seconds=result.duration_seconds,
                )

                # Stop on failure
                print(f"\n⚠️  Stopping at {sprint_id} due to failure")
                if result.errors:
                    for error in result.errors[:3]:
                        print(f"      └─ {error[:80]}")
                break

        except Exception as e:
            # Sync: Sprint failed with exception
            gh_sync.on_sprint_fail(sprint_id, error=str(e))
            print(f"  ❌ {sprint_id}: Failed - {e}")
            break

    # Summary
    success = sum(1 for r in results if r.status in ("success", "skipped"))
    failed = sum(1 for r in results if r.status == "failed")

    print()
    print("=" * 60)
    print("EXECUTION SUMMARY")
    print("=" * 60)
    print(f"  Sprints executed: {len(results)}")
    print(f"  Success: {success}")
    print(f"  Failed: {failed}")

    if failed == 0 and len(results) == len(plan.pending_sprints):
        print()
        print("🎉 ROADMAP COMPLETE!")

    # Stop Run Master daemon
    stop_run_master_daemon()

    return 0 if failed == 0 else 1


def cmd_gh_projects(args: argparse.Namespace) -> int:
    """Comando: gh-projects - Gerencia integração com GitHub Projects.

    Subcomandos:
        status - Mostra status da integração
        sync   - Sincroniza roadmap inteiro
        test   - Testa sync para um sprint
    """
    from pipeline.gh_projects import GitHubProjectsSync
    from pipeline.roadmap import get_roadmap

    sync = GitHubProjectsSync()
    gh_command = getattr(args, "gh_command", None)

    if gh_command == "status" or gh_command is None:
        print("=" * 60)
        print("GITHUB PROJECTS INTEGRATION STATUS")
        print("=" * 60)
        print()
        print(f"  Enabled: {sync.enabled}")
        print(f"  Owner: {sync._config.owner or '(not configured)'}")
        print(f"  Project Number: {sync._config.project_number or '(not configured)'}")
        print(f"  gh CLI: {'Available' if sync._check_gh_cli() else 'Not available'}")
        print()

        if not sync.enabled:
            print("  To enable:")
            print("  1. Create a GitHub Project (Projects V2)")
            print("  2. Edit configs/pipeline/gh_projects.yaml")
            print("  3. Set enabled: true, owner, and project_number")
            print()

        return 0

    elif gh_command == "sync":
        if not sync.enabled:
            print("GitHub Projects sync is disabled. Configure gh_projects.yaml first.")
            return 1

        roadmap = get_roadmap()
        sprints = [
            {"id": s.id, "name": s.name, "product": s.product}
            for s in roadmap.get_all_sprints()
        ]

        print(f"Syncing {len(sprints)} sprints to GitHub Projects...")
        results = sync.sync_roadmap(sprints)

        success = sum(1 for v in results.values() if v)
        failed = len(results) - success

        print()
        print(f"  Created/Found: {success}")
        print(f"  Failed: {failed}")

        return 0 if failed == 0 else 1

    elif gh_command == "test":
        sprint_id = getattr(args, "sprint_id", "S00")

        if not sync.enabled:
            print("GitHub Projects sync is disabled. Running in dry-run mode...")
            print()

        print(f"Testing sync for {sprint_id}...")
        print()

        print("  1. on_sprint_start()...")
        result1 = sync.on_sprint_start(sprint_id, run_id="test_run_001")
        print(f"     {'OK' if result1 else 'SKIPPED (sync disabled)'}")

        print("  2. on_sprint_complete()...")
        result2 = sync.on_sprint_complete(
            sprint_id,
            duration_seconds=123.4,
            gates_passed=["G0", "G1", "G2"],
        )
        print(f"     {'OK' if result2 else 'SKIPPED (sync disabled)'}")

        print()
        print("Test complete!")
        return 0

    else:
        print(f"Unknown subcommand: {gh_command}")
        print("Use: gh-projects status|sync|test")
        return 1


def cmd_roadmap(args: argparse.Namespace) -> int:
    """Comando: roadmap - Mostra status do roadmap unificado.

    Exibe:
    - Progresso geral do roadmap
    - Sprints completados por produto
    - Próximos sprints a executar
    """
    from pipeline.roadmap import get_roadmap, Product

    roadmap = get_roadmap()

    print("=" * 60)
    print("UNIFIED ROADMAP STATUS")
    print("=" * 60)
    print()

    # Show overall status
    roadmap.print_status("all")

    # Show per-product breakdown
    print("-" * 60)
    print("PER-PRODUCT BREAKDOWN")
    print("-" * 60)
    print()

    for product in [Product.HUMANGR]:
        plan = roadmap.get_execution_plan(product)
        sprints = roadmap.get_sprint_ids(product)

        if sprints:
            print(f"  {product.value.upper()}:")
            print(f"    Total: {plan.total_sprints} sprints")
            print(f"    Completed: {len(plan.completed_sprints)}")
            print(f"    Pending: {len(plan.pending_sprints)}")
            if plan.next_sprint:
                print(f"    Next: {plan.next_sprint}")
            print(f"    Range: {sprints[0]} -> {sprints[-1]}")
            print()

    print("-" * 60)
    print("EXECUTION ORDER (first 20)")
    print("-" * 60)
    all_sprints = roadmap.get_all_sprints()[:20]
    for i, sprint in enumerate(all_sprints, 1):
        print(f"  {i:2d}. {sprint.id:12s} ({sprint.product})")

    if len(roadmap.get_all_sprints()) > 20:
        print(f"  ... and {len(roadmap.get_all_sprints()) - 20} more")

    return 0


def cmd_health(args: argparse.Namespace) -> int:
    """Comando: health - Verifica saúde dos serviços.

    RED TEAM FIX MED-009: Returns non-zero exit code if critical stacks are unhealthy.
    """
    pipeline = get_pipeline()

    print("=" * 60)
    print("HEALTH CHECK")
    print("=" * 60)

    # RED TEAM FIX MED-009: Track failures and return appropriate exit code
    critical_failures = []
    warnings = []

    # Check CrewAI (CRITICAL)
    try:
        crewai = pipeline.crewai
        if crewai:
            print(f"  ✅ CrewAI: Available")
        else:
            print(f"  ❌ CrewAI: Not available")
            critical_failures.append("crewai")
    except Exception as e:
        print(f"  ❌ CrewAI: {e}")
        critical_failures.append("crewai")

    # Check Redis (CRITICAL)
    try:
        redis = pipeline.redis
        if redis and redis.ping():
            print(f"  ✅ Redis: Connected")
        else:
            print(f"  ❌ Redis: Not connected")
            critical_failures.append("redis")
    except Exception as e:
        print(f"  ❌ Redis: {e}")
        critical_failures.append("redis")

    # Check Langfuse (CRITICAL for observability)
    try:
        langfuse = pipeline.langfuse
        if langfuse:
            print(f"  ✅ Langfuse: Available")
        else:
            print(f"  ⚠️  Langfuse: Not configured")
            warnings.append("langfuse")
    except Exception as e:
        print(f"  ❌ Langfuse: {e}")
        warnings.append("langfuse")

    # Check Letta (optional)
    try:
        letta = pipeline.letta
        if letta:
            print(f"  ✅ Letta: Available")
        else:
            print(f"  ⚠️  Letta: Not configured")
            warnings.append("letta")
    except Exception as e:
        print(f"  ❌ Letta: {e}")
        warnings.append("letta")

    # GHOST CODE INTEGRATION (2026-01-30): Show required stacks per operation
    required_stacks = _get_required_stacks()
    if HEALTH_MODULE_AVAILABLE and required_stacks:
        print()
        print("-" * 60)
        print("REQUIRED STACKS PER OPERATION")
        print("-" * 60)
        all_healthy_stacks = {"crewai", "redis", "langfuse", "letta"} - set(critical_failures) - set(warnings)

        for operation, stacks in sorted(required_stacks.items()):
            missing = [s for s in stacks if s not in all_healthy_stacks]
            if missing:
                status = f"❌ Missing: {', '.join(missing)}"
            else:
                status = "✅ Ready"
            print(f"  {operation:25s} {status}")

    # MED-009: Summary with appropriate exit code
    print("=" * 60)
    if critical_failures:
        print(f"❌ UNHEALTHY: Critical failures: {critical_failures}")
        return 1
    elif warnings:
        print(f"⚠️  DEGRADED: Warnings: {warnings}")
        return 0  # Warnings don't block, but are logged
    else:
        print("✅ ALL HEALTHY")
        return 0


# =============================================================================
# LANGGRAPH COMMANDS
# =============================================================================


def cmd_langgraph(args: argparse.Namespace) -> int:
    """Comando: langgraph (lg) - Run sprint with LangGraph control plane.

    This uses the new LangGraph-based control plane with:
    - Automatic checkpointing after each node
    - Resume from any checkpoint after crashes
    - Idempotent node execution

    Usage:
        python -m pipeline.cli langgraph S01
        python -m pipeline.cli lg S01 --resume checkpoint_id
    """
    # Ensure Docker services are running
    if not ensure_docker_services():
        print("❌ Failed to start Docker services. Cannot proceed.")
        return 1

    try:
        from pipeline.langgraph.bridge import (
            get_langgraph_bridge,
            run_sprint_sync,
        )
        from pipeline.langgraph.workflow import LANGGRAPH_AVAILABLE
    except ImportError:
        print("❌ LangGraph not available. Install with: pip install langgraph")
        return 1

    if not LANGGRAPH_AVAILABLE:
        print("❌ LangGraph not available. Install with: pip install langgraph")
        return 1

    sprint_id = args.sprint_id
    run_id = getattr(args, 'run_id', None)
    resume_from = getattr(args, 'resume', None)

    print("=" * 60)
    print("LANGGRAPH SPRINT EXECUTION")
    print("=" * 60)
    print(f"  Sprint: {sprint_id}")
    if run_id:
        print(f"  Run ID: {run_id}")
    if resume_from:
        print(f"  Resume from: {resume_from}")
    print()

    try:
        result = run_sprint_sync(
            sprint_id=sprint_id,
            run_id=run_id,
            resume_from=resume_from,
        )

        status_icon = "✅" if result.status == "success" else "❌"
        print()
        print("=" * 60)
        print(f"SPRINT {sprint_id} RESULT (LangGraph)")
        print("=" * 60)
        print(f"  {status_icon} Status: {result.status}")
        print(f"  Duration: {result.duration_seconds:.1f}s")
        print(f"  Tasks Completed: {result.tasks_completed}")
        print(f"  Tasks Failed: {result.tasks_failed}")

        if result.errors:
            print()
            print("  Errors:")
            for error in result.errors[:5]:
                print(f"    - {error[:100]}")

        return 0 if result.status == "success" else 1

    except Exception as e:
        print(f"❌ LangGraph execution failed: {e}")
        return 1


def cmd_lg_status(args: argparse.Namespace) -> int:
    """Comando: lg-status - Check LangGraph checkpoint status."""
    import asyncio

    try:
        from pipeline.langgraph.bridge import get_langgraph_bridge
    except ImportError:
        print("❌ LangGraph not available. Install with: pip install langgraph")
        return 1

    run_id = args.run_id

    print("=" * 60)
    print("LANGGRAPH CHECKPOINT STATUS")
    print("=" * 60)
    print(f"  Run ID: {run_id}")
    print()

    try:
        bridge = get_langgraph_bridge()
        status = asyncio.run(bridge.get_checkpoint_status(run_id))

        if not status.get("available", False):
            print(f"  ⚠️  Checkpointing: {status.get('reason', 'disabled')}")
            return 1

        checkpoints = status.get("checkpoints", [])
        print(f"  Checkpoints found: {len(checkpoints)}")

        if checkpoints:
            print()
            print("  Available checkpoints:")
            for cp in checkpoints[:10]:  # Show last 10
                cp_id = cp.get("id", "unknown")
                created = cp.get("created_at", "unknown")
                phase = cp.get("phase", "unknown")
                print(f"    - {cp_id} (phase: {phase}, at: {created})")

        return 0

    except Exception as e:
        print(f"❌ Failed to get checkpoint status: {e}")
        return 1


def cmd_lg_resume(args: argparse.Namespace) -> int:
    """Comando: lg-resume - Resume workflow from checkpoint."""
    import asyncio

    try:
        from pipeline.langgraph.bridge import get_langgraph_bridge
    except ImportError:
        print("❌ LangGraph not available. Install with: pip install langgraph")
        return 1

    checkpoint_id = args.checkpoint_id

    print("=" * 60)
    print("LANGGRAPH RESUME FROM CHECKPOINT")
    print("=" * 60)
    print(f"  Checkpoint: {checkpoint_id}")
    print()

    try:
        bridge = get_langgraph_bridge()
        result = asyncio.run(bridge.resume_from_checkpoint(checkpoint_id))

        status_icon = "✅" if result.status == "success" else "❌"
        print()
        print("=" * 60)
        print(f"RESUMED SPRINT RESULT")
        print("=" * 60)
        print(f"  {status_icon} Status: {result.status}")
        print(f"  Sprint: {result.sprint_id}")
        print(f"  Duration: {result.duration_seconds:.1f}s")

        if result.errors:
            print()
            print("  Errors:")
            for error in result.errors[:5]:
                print(f"    - {error[:100]}")

        return 0 if result.status == "success" else 1

    except ValueError as e:
        print(f"❌ Checkpoint not found: {e}")
        return 1
    except Exception as e:
        print(f"❌ Resume failed: {e}")
        return 1


def cmd_lg_start(args: argparse.Namespace) -> int:
    """Comando: lg-start - Run multiple sprints with LangGraph."""
    # Ensure Docker services are running
    if not ensure_docker_services():
        print("❌ Failed to start Docker services. Cannot proceed.")
        return 1

    try:
        from pipeline.langgraph.bridge import run_sprint_sync
    except ImportError:
        print("❌ LangGraph not available. Install with: pip install langgraph")
        return 1

    start_num = int(args.start[1:])
    end_num = int(args.end[1:])

    print("=" * 60)
    print("LANGGRAPH MULTI-SPRINT EXECUTION")
    print("=" * 60)
    print(f"  From: {args.start}")
    print(f"  To: {args.end}")
    print()

    results = []
    for sprint_num in range(start_num, end_num + 1):
        sprint_id = f"S{sprint_num:02d}"
        print(f"Executing {sprint_id}...")

        try:
            result = run_sprint_sync(sprint_id=sprint_id)
            results.append(result)

            status_icon = "✅" if result.status == "success" else "❌"
            print(f"  {status_icon} {sprint_id}: {result.status} ({result.duration_seconds:.1f}s)")

            if result.status != "success":
                print(f"  ⚠️  Stopping at {sprint_id} due to failure")
                break

        except Exception as e:
            print(f"  ❌ {sprint_id}: Failed - {e}")
            break

    # Summary
    success = sum(1 for r in results if r.status == "success")
    failed = sum(1 for r in results if r.status != "success")

    print()
    print("=" * 60)
    print("LANGGRAPH EXECUTION SUMMARY")
    print("=" * 60)
    print(f"  Total sprints: {len(results)}")
    print(f"  Success: {success}")
    print(f"  Failed: {failed}")

    return 0 if failed == 0 else 1


def main() -> int:
    """Entry point do CLI."""
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Pipeline v2 - Stack Modernizado",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--run-dir", help="Diretório base para runs")

    subparsers = parser.add_subparsers(dest="command", help="Comando a executar")

    # init
    init_parser = subparsers.add_parser("init", help="Inicializa novo run")
    init_parser.add_argument("--run-id", help="ID do run (auto-gerado se não fornecido)")
    init_parser.add_argument("--resume", "-r", action="store_true", help="Resume run existente (requer --run-id)")
    init_parser.add_argument(
        "--open-dashboards", "-o",
        action="store_true",
        help="Auto-open observability dashboards (Langfuse, Qdrant, FalkorDB)"
    )

    # start
    start_parser = subparsers.add_parser("start", help="Inicia execução de sprints (auto-resume)")
    start_parser.add_argument("--start", "-s", default="S00", help="Sprint inicial")
    start_parser.add_argument("--end", "-e", default="S40", help="Sprint final")
    start_parser.add_argument(
        "--include", "-i",
        help="Sprints adicionais para incluir (comma-separated, ex: S05H,HOTFIX01)"
    )
    start_parser.add_argument(
        "--open-dashboards", "-o",
        action="store_true",
        help="Auto-open observability dashboards (Langfuse, Qdrant, FalkorDB)"
    )
    start_parser.add_argument(
        "--foreground", "-f",
        action="store_true",
        help="Run in foreground (don't daemonize). Use for debugging."
    )
    start_parser.add_argument(
        "--skip-consistency-check",
        action="store_true",
        help="Skip cross-sprint consistency check before execution"
    )
    # GHOST CODE INTEGRATION (2026-01-30): Streaming support
    start_parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable real-time streaming output (shows node progress in real-time)"
    )
    # GHOST CODE INTEGRATION (2026-01-30): Environment validation
    start_parser.add_argument(
        "--validate-env",
        action="store_true",
        help="Validate critical environment variables before execution (GAP-CFG-001)"
    )
    start_parser.add_argument(
        "--strict-env",
        action="store_true",
        help="Fail on environment variable warnings (not just critical errors)"
    )

    # clear-sprint
    clear_parser = subparsers.add_parser("clear-sprint", help="Limpa estado de sprint para rodar do zero")
    clear_parser.add_argument("sprint_id", help="ID do sprint (ex: S02)")
    clear_parser.add_argument("--force", "-f", action="store_true", help="Não pedir confirmação")
    clear_parser.add_argument("--clear-redis", action="store_true", help="Também limpar dados do Redis")

    # status
    subparsers.add_parser("status", help="Mostra estado atual")

    # sprint
    sprint_parser = subparsers.add_parser("sprint", help="Executa sprint específico")
    sprint_parser.add_argument("sprint_id", help="ID do sprint (ex: S01)")

    # health
    subparsers.add_parser("health", help="Verifica saúde dos serviços")

    # run - Smart auto-resume (NEW!)
    run_parser = subparsers.add_parser(
        "run",
        help="Executa o roadmap com auto-resume (recomendado)"
    )
    run_parser.add_argument(
        "--open-dashboards", "-o",
        action="store_true",
        help="Auto-open observability dashboards"
    )

    # roadmap - Show roadmap status
    subparsers.add_parser("roadmap", help="Mostra status do roadmap unificado")

    # GitHub Projects commands
    gh_parser = subparsers.add_parser(
        "gh-projects",
        help="Gerencia integração com GitHub Projects"
    )
    gh_subparsers = gh_parser.add_subparsers(dest="gh_command", help="Subcomando")
    gh_subparsers.add_parser("status", help="Mostra status da integração")
    gh_subparsers.add_parser("sync", help="Sincroniza roadmap inteiro para GitHub Projects")
    gh_test_parser = gh_subparsers.add_parser("test", help="Testa sync para um sprint")
    gh_test_parser.add_argument("sprint_id", help="Sprint ID para testar (ex: S00)")

    # LangGraph commands
    # langgraph - Run sprint with LangGraph control plane
    lg_parser = subparsers.add_parser(
        "langgraph",
        aliases=["lg"],
        help="Run sprint using LangGraph control plane (checkpoint/resume enabled)"
    )
    lg_parser.add_argument("sprint_id", help="Sprint ID (ex: S01)")
    lg_parser.add_argument("--run-id", help="Run ID (auto-generated if not provided)")
    lg_parser.add_argument("--resume", "-r", help="Resume from checkpoint ID")

    # lg-status - Check LangGraph/checkpoint status
    lg_status_parser = subparsers.add_parser(
        "lg-status",
        help="Check LangGraph checkpoint status for a run"
    )
    lg_status_parser.add_argument("run_id", help="Run ID to check")

    # lg-resume - Resume from checkpoint
    lg_resume_parser = subparsers.add_parser(
        "lg-resume",
        help="Resume workflow from a checkpoint"
    )
    lg_resume_parser.add_argument("checkpoint_id", help="Checkpoint ID to resume from")

    # lg-start - Run multiple sprints with LangGraph
    lg_start_parser = subparsers.add_parser(
        "lg-start",
        help="Run multiple sprints using LangGraph (start to end)"
    )
    lg_start_parser.add_argument("--start", "-s", default="S00", help="Start sprint")
    lg_start_parser.add_argument("--end", "-e", default="S40", help="End sprint")

    # Spec Kit v3.0 Ironclad commands
    from pipeline.spec_kit.cli import setup_spec_subparsers
    setup_spec_subparsers(subparsers)

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.command is None:
        parser.print_help()
        return 1

    # Start Prometheus metrics server for Pipeline Control Center
    try:
        # IMPORTANT: Import metrics module FIRST to register all metrics with REGISTRY
        # before starting the HTTP server
        from pipeline.observability.metrics import (
            PROMETHEUS_AVAILABLE,
            AGENT_MESSAGES_TOTAL,
            PHASE_TRANSITIONS_TOTAL,
            CURRENT_PHASE,
            ACTIVE_AGENTS,
            TASK_DURATION,
            LLM_LATENCY,
        )
        from prometheus_client import start_http_server, REGISTRY
        import os

        if PROMETHEUS_AVAILABLE:
            metrics_port = int(os.getenv("PIPELINE_METRICS_PORT", "8000"))
            start_http_server(metrics_port, registry=REGISTRY)
            # Initialize some metrics so they appear in Grafana
            CURRENT_PHASE.labels(run_id="init", sprint_id="init").set(0)
            ACTIVE_AGENTS.labels(role="system").set(1)
            logging.getLogger(__name__).info(f"📊 Prometheus metrics server started on port {metrics_port}")
        else:
            logging.getLogger(__name__).warning("prometheus_client not available, metrics disabled")
    except ImportError as e:
        logging.getLogger(__name__).debug(f"Metrics setup failed: {e}")
    except OSError as e:
        # Port already in use (another pipeline instance)
        logging.getLogger(__name__).debug(f"Metrics server port in use: {e}")

    # Initialize Grafana Redis metrics publisher for Pipeline Control Center dashboard
    # This publishes real-time metrics to Redis keys that Grafana reads directly
    try:
        from pipeline.grafana_metrics import get_metrics_publisher
        grafana_publisher = get_metrics_publisher()
        # Set initial state - "idle" until a sprint starts
        grafana_publisher.set_status("idle")
        grafana_publisher.set_current_level("L0")
        logging.getLogger(__name__).info("📊 Grafana metrics publisher initialized (Redis)")
    except Exception as e:
        logging.getLogger(__name__).debug(f"Grafana metrics setup failed (Redis not available?): {e}")

    # Import spec command handler
    from pipeline.spec_kit.cli import handle_spec_command

    commands = {
        "init": cmd_init,
        "start": cmd_start,
        "clear-sprint": cmd_clear_sprint,
        "status": cmd_status,
        "sprint": cmd_sprint,
        "health": cmd_health,
        # Unified roadmap commands (RECOMMENDED)
        "run": cmd_run,
        "roadmap": cmd_roadmap,
        # GitHub Projects integration
        "gh-projects": cmd_gh_projects,
        # LangGraph commands
        "langgraph": cmd_langgraph,
        "lg": cmd_langgraph,
        "lg-status": cmd_lg_status,
        "lg-resume": cmd_lg_resume,
        "lg-start": cmd_lg_start,
        # Spec Kit v3.0 Ironclad
        "spec": handle_spec_command,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
