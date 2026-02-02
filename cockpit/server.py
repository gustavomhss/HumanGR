# cockpit/server.py
"""
Standalone Flask server for HumanGR Cockpit.
Runs on port 5002.

Features:
- READ-ONLY state monitoring (never writes to pipeline files)
- Pipeline control (start/stop via subprocess)
- Real-time WebSocket updates
- Resilience state monitoring (circuit breakers, oscillation, retry metrics)

Usage:
    # Run with specific run directory
    PYTHONPATH=src python -m cockpit.server /path/to/run_dir

    # Or import and create app
    from cockpit.server import create_app, socketio
    app = create_app("/path/to/run_dir")
    socketio.run(app, port=5002)
"""
import os
import sys
import time
import logging
import threading
import subprocess
import signal
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass, field

from flask import Flask, render_template, jsonify, request, send_from_directory, Response
from flask_socketio import SocketIO, emit
from flask_cors import CORS

from . import cockpit_transformer
from . import resilience_reader
from . import COCKPIT_DIR

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("cockpit.server")

# =============================================================================
# SOCKETIO
# =============================================================================

socketio = SocketIO(cors_allowed_origins="*", async_mode='threading')

# =============================================================================
# GLOBALS
# =============================================================================

# Store last state for change detection
_last_state_hash = None
_last_resilience_hash: Optional[str] = None
_watcher_running = False
# Timestamp of server startup - only auto-discover runs newer than this
_startup_timestamp: float = time.time()
# Timestamp of last reset - only auto-discover runs newer than this
_reset_timestamp: Optional[float] = None


# =============================================================================
# HASH CALCULATION
# =============================================================================


def _calculate_hash(data: dict) -> str:
    """
    Calculate deterministic hash of data for change detection.

    Uses SHA256 and returns first 16 characters for brevity.
    Keys are sorted to ensure deterministic output.

    Args:
        data: Dictionary to hash

    Returns:
        16-character hex hash string
    """
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]

# =============================================================================
# PIPELINE CONTROLLER
# =============================================================================


class _PseudoProcess:
    """Wraps an existing PID to mimic subprocess.Popen interface."""

    def __init__(self, pid: int):
        self.pid = pid
        self.returncode = None

    def poll(self) -> Optional[int]:
        """Check if process is still running."""
        try:
            os.kill(self.pid, 0)  # Signal 0 = check existence
            return None  # Still running
        except OSError:
            self.returncode = -1
            return -1  # Not running

    def kill(self):
        """Kill the process."""
        try:
            os.kill(self.pid, signal.SIGKILL)
        except OSError:
            pass

    def terminate(self):
        """Terminate the process."""
        try:
            os.kill(self.pid, signal.SIGTERM)
        except OSError:
            pass

    def wait(self, timeout=None):
        """Wait for process to finish."""
        import time
        start = time.time()
        while True:
            if self.poll() is not None:
                return self.returncode
            if timeout and (time.time() - start) > timeout:
                raise subprocess.TimeoutExpired(cmd="", timeout=timeout)
            time.sleep(0.1)


@dataclass
class PipelineProcess:
    """Tracks a running pipeline process."""
    process: subprocess.Popen  # Can also be _PseudoProcess
    start_sprint: str
    end_sprint: str
    started_at: datetime
    status: str = "running"  # running, paused, completed, failed, aborted
    output_lines: list = field(default_factory=list)


class PipelineController:
    """
    Controls pipeline execution via subprocess.
    Supports start, pause, resume, and abort operations.
    """

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.current_process: Optional[PipelineProcess] = None
        self._output_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Try to adopt any running pipeline on startup
        self._adopt_running_pipeline()

    def _adopt_running_pipeline(self):
        """Detect and adopt any running pipeline process."""
        try:
            # Find running pipeline.cli processes
            result = subprocess.run(
                ["pgrep", "-f", "pipeline.cli_v2"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                if pids:
                    pid = int(pids[0])

                    # Get command line to extract sprint info
                    cmd_result = subprocess.run(
                        ["ps", "-p", str(pid), "-o", "args="],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )

                    cmd = cmd_result.stdout.strip()
                    start_sprint = "S00"
                    end_sprint = "S25"

                    # Parse --start and --end from command
                    if "--start" in cmd:
                        parts = cmd.split()
                        for i, p in enumerate(parts):
                            if p == "--start" and i + 1 < len(parts):
                                start_sprint = parts[i + 1]
                            if p == "--end" and i + 1 < len(parts):
                                end_sprint = parts[i + 1]

                    # Create a pseudo-process wrapper
                    self.current_process = PipelineProcess(
                        process=_PseudoProcess(pid),
                        start_sprint=start_sprint,
                        end_sprint=end_sprint,
                        started_at=datetime.now(),  # Unknown, use now
                        status="running",
                    )

                    logger.info(f"Adopted running pipeline: PID {pid} ({start_sprint} -> {end_sprint})")

        except Exception as e:
            logger.debug(f"No running pipeline to adopt: {e}")

    def _check_docker(self) -> Optional[str]:
        """Check if Docker is running. Returns error message if not."""
        # Try up to 3 times with increasing timeouts
        for attempt, timeout in enumerate([10, 15, 20], 1):
            try:
                result = subprocess.run(
                    ["docker", "info"],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                if result.returncode == 0:
                    return None  # Docker is OK
                # Docker returned error - check if daemon is starting
                if "Cannot connect" in (result.stderr or ""):
                    if attempt < 3:
                        logger.debug(f"Docker not ready, retrying ({attempt}/3)...")
                        time.sleep(2)
                        continue
                return "Docker is not running. Please start Docker Desktop."
            except FileNotFoundError:
                return "Docker is not installed. Please install Docker Desktop."
            except subprocess.TimeoutExpired:
                if attempt < 3:
                    logger.debug(f"Docker timeout, retrying ({attempt}/3)...")
                    continue
                return "Docker is not responding. Please restart Docker Desktop."
            except Exception as e:
                return f"Docker check failed: {e}"
        return "Docker check failed after 3 attempts."

    def start(
        self,
        start_sprint: str = "S00",
        end_sprint: str = "S25",
    ) -> Dict[str, Any]:
        """
        Start the pipeline.

        Args:
            start_sprint: Starting sprint (e.g., "S00")
            end_sprint: Ending sprint (e.g., "S25")

        Returns:
            Status dict with success/error info
        """
        with self._lock:
            if self.current_process and self.current_process.status == "running":
                return {
                    "success": False,
                    "error": "Pipeline already running",
                    "status": self.get_status(),
                }

            # Check Docker first
            docker_error = self._check_docker()
            if docker_error:
                return {
                    "success": False,
                    "error": docker_error,
                }

            # Build command
            python_path = self.project_root / ".venv" / "bin" / "python"
            if not python_path.exists():
                python_path = Path(sys.executable)

            # Build pipeline start command
            cmd = [
                str(python_path),
                "-m", "pipeline.cli_v2",
                "start",
                "--start", start_sprint,
                "--end", end_sprint,
            ]

            env = os.environ.copy()
            env["PYTHONPATH"] = str(self.project_root / "src")
            env["PYTHONUNBUFFERED"] = "1"  # Force unbuffered output

            try:
                # Start process
                process = subprocess.Popen(
                    cmd,
                    cwd=str(self.project_root),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    preexec_fn=os.setsid,  # Create new process group for pause/resume
                )

                self.current_process = PipelineProcess(
                    process=process,
                    start_sprint=start_sprint,
                    end_sprint=end_sprint,
                    started_at=datetime.now(),
                    status="running",
                )

                # Start output reader thread
                self._start_output_reader()

                logger.info(f"Pipeline started: {start_sprint} -> {end_sprint} (PID: {process.pid})")

                return {
                    "success": True,
                    "pid": process.pid,
                    "start_sprint": start_sprint,
                    "end_sprint": end_sprint,
                    "status": self.get_status(),
                }

            except Exception as e:
                logger.error(f"Failed to start pipeline: {e}")
                return {
                    "success": False,
                    "error": str(e),
                }

    def pause(self) -> Dict[str, Any]:
        """
        Pause the running pipeline (SIGSTOP).

        Returns:
            Status dict
        """
        with self._lock:
            if not self.current_process or self.current_process.status != "running":
                return {
                    "success": False,
                    "error": "No running pipeline to pause",
                }

            try:
                # Send SIGSTOP to process group
                os.killpg(os.getpgid(self.current_process.process.pid), signal.SIGSTOP)
                self.current_process.status = "paused"

                logger.info(f"Pipeline paused (PID: {self.current_process.process.pid})")

                return {
                    "success": True,
                    "status": self.get_status(),
                }

            except Exception as e:
                logger.error(f"Failed to pause pipeline: {e}")
                return {
                    "success": False,
                    "error": str(e),
                }

    def resume(self) -> Dict[str, Any]:
        """
        Resume a paused pipeline (SIGCONT).

        Returns:
            Status dict
        """
        with self._lock:
            if not self.current_process or self.current_process.status != "paused":
                return {
                    "success": False,
                    "error": "No paused pipeline to resume",
                }

            try:
                # Send SIGCONT to process group
                os.killpg(os.getpgid(self.current_process.process.pid), signal.SIGCONT)
                self.current_process.status = "running"

                logger.info(f"Pipeline resumed (PID: {self.current_process.process.pid})")

                return {
                    "success": True,
                    "status": self.get_status(),
                }

            except Exception as e:
                logger.error(f"Failed to resume pipeline: {e}")
                return {
                    "success": False,
                    "error": str(e),
                }

    def abort(self) -> Dict[str, Any]:
        """
        Abort the running/paused pipeline - kills ALL related processes.

        Returns:
            Status dict
        """
        with self._lock:
            if not self.current_process or self.current_process.status in ["completed", "failed", "aborted"]:
                # Even if no tracked process, try to kill any orphaned pipeline processes
                self._kill_all_pipeline_processes()
                return {
                    "success": True,
                    "error": "No tracked pipeline, but cleaned up any orphaned processes",
                    "status": self.get_status(),
                }

            try:
                pid = self.current_process.process.pid

                # Strategy 1: Kill by process group
                try:
                    pgid = os.getpgid(pid)
                    os.killpg(pgid, signal.SIGTERM)
                    time.sleep(0.5)
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass

                # Strategy 2: Kill all pipeline-related processes by pattern
                self._kill_all_pipeline_processes()

                # Strategy 3: Kill the main process directly
                try:
                    self.current_process.process.kill()
                    self.current_process.process.wait(timeout=2)
                except Exception:
                    pass

                self.current_process.status = "aborted"
                self.current_process.output_lines.append("Pipeline aborted by user")

                # Broadcast abort
                socketio.emit('pipeline_output', {
                    'line': 'Pipeline aborted by user',
                    'timestamp': datetime.now().isoformat(),
                })
                socketio.emit('pipeline_status', self.get_status())

                logger.info(f"Pipeline aborted (PID: {pid})")

                return {
                    "success": True,
                    "status": self.get_status(),
                }

            except Exception as e:
                logger.error(f"Failed to abort pipeline: {e}")
                # Last resort: try pkill anyway
                self._kill_all_pipeline_processes()
                if self.current_process:
                    self.current_process.status = "aborted"
                return {
                    "success": True,
                    "error": f"Force killed: {e}",
                    "status": self.get_status(),
                }

    def _kill_all_pipeline_processes(self):
        """Kill ALL pipeline-related processes using pkill."""
        patterns = [
            "pipeline.cli_v2",
            "pipeline.orchestrator",
            "crewai.*start",
        ]
        for pattern in patterns:
            try:
                subprocess.run(
                    ["pkill", "-9", "-f", pattern],
                    capture_output=True,
                    timeout=5,
                )
            except Exception:
                pass

        logger.info("Killed all pipeline-related processes")

    def resume_from_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """
        Resume pipeline from a LangGraph checkpoint.

        Args:
            checkpoint_id: Checkpoint ID to resume from

        Returns:
            Status dict with success/error info
        """
        with self._lock:
            if self.current_process and self.current_process.status == "running":
                return {
                    "success": False,
                    "error": "Pipeline already running. Stop it first.",
                    "status": self.get_status(),
                }

            # Check Docker first
            docker_error = self._check_docker()
            if docker_error:
                return {
                    "success": False,
                    "error": docker_error,
                }

            # Build command for lg-resume
            python_path = self.project_root / ".venv" / "bin" / "python"
            if not python_path.exists():
                python_path = Path(sys.executable)

            cmd = [
                str(python_path),
                "-m", "pipeline.cli_v2",
                "lg-resume",
                checkpoint_id,
            ]

            env = os.environ.copy()
            env["PYTHONPATH"] = str(self.project_root / "src")
            env["PYTHONUNBUFFERED"] = "1"

            try:
                process = subprocess.Popen(
                    cmd,
                    cwd=str(self.project_root),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    preexec_fn=os.setsid,
                )

                # Extract run_id from checkpoint_id for display
                run_id = checkpoint_id.split("_final")[0] if "_final" in checkpoint_id else checkpoint_id.split(":")[0] if ":" in checkpoint_id else checkpoint_id

                self.current_process = PipelineProcess(
                    process=process,
                    start_sprint="RESUME",  # Special marker for resumed runs
                    end_sprint=checkpoint_id,  # Store checkpoint ID for reference
                    started_at=datetime.now(),
                    status="running",
                )

                self._start_output_reader()

                logger.info(f"Pipeline resumed from checkpoint: {checkpoint_id} (PID: {process.pid})")

                return {
                    "success": True,
                    "pid": process.pid,
                    "checkpoint_id": checkpoint_id,
                    "run_id": run_id,
                    "status": self.get_status(),
                }

            except Exception as e:
                logger.error(f"Failed to resume pipeline from checkpoint: {e}")
                return {
                    "success": False,
                    "error": str(e),
                }

    def get_status(self) -> Dict[str, Any]:
        """
        Get current pipeline status.

        Returns:
            Status dict with all info
        """
        if not self.current_process:
            return {
                "running": False,
                "status": "idle",
            }

        # Check if process has finished
        if self.current_process.status == "running":
            poll_result = self.current_process.process.poll()
            if poll_result is not None:
                if poll_result == 0:
                    self.current_process.status = "completed"
                else:
                    self.current_process.status = "failed"

        # Check if this is an adopted pipeline (no stdout access)
        is_adopted = isinstance(self.current_process.process, _PseudoProcess)

        return {
            "running": self.current_process.status in ["running", "paused"],
            "status": self.current_process.status,
            "pid": self.current_process.process.pid,
            "start_sprint": self.current_process.start_sprint,
            "end_sprint": self.current_process.end_sprint,
            "started_at": self.current_process.started_at.isoformat(),
            "elapsed_seconds": (datetime.now() - self.current_process.started_at).total_seconds(),
            "recent_output": self.current_process.output_lines[-50:],  # Last 50 lines
            "adopted": is_adopted,
        }

    def _start_output_reader(self):
        """Start thread to read process output."""
        def read_output():
            if not self.current_process:
                return

            process = self.current_process.process
            try:
                for line in iter(process.stdout.readline, ''):
                    if not line:
                        break
                    line = line.rstrip()
                    if line:  # Skip empty lines
                        self.current_process.output_lines.append(line)

                        # Broadcast to WebSocket clients
                        socketio.emit('pipeline_output', {
                            'line': line,
                            'timestamp': datetime.now().isoformat(),
                        })

                        # Keep only last 1000 lines in memory
                        if len(self.current_process.output_lines) > 1000:
                            self.current_process.output_lines = self.current_process.output_lines[-500:]

            except Exception as e:
                logger.error(f"Error reading pipeline output: {e}")

            # Wait for process to finish and update status
            try:
                exit_code = process.wait(timeout=5)
                with self._lock:
                    if exit_code == 0:
                        self.current_process.status = "completed"
                        final_msg = "Pipeline completed successfully"
                    else:
                        self.current_process.status = "failed"
                        final_msg = f"Pipeline failed with exit code {exit_code}"

                    self.current_process.output_lines.append(final_msg)

                    # Broadcast final status
                    socketio.emit('pipeline_output', {
                        'line': final_msg,
                        'timestamp': datetime.now().isoformat(),
                    })
                    socketio.emit('pipeline_status', self.get_status())

                    logger.info(f"Pipeline finished: {self.current_process.status} (exit code: {exit_code})")

            except subprocess.TimeoutExpired:
                logger.warning("Process still running after output ended")
            except Exception as e:
                logger.error(f"Error waiting for process: {e}")

        self._output_thread = threading.Thread(
            target=read_output,
            daemon=True,
            name="pipeline-output-reader"
        )
        self._output_thread.start()


# Global pipeline controller (initialized in create_app)
_pipeline_controller: Optional[PipelineController] = None

# =============================================================================
# APP FACTORY
# =============================================================================


def _find_latest_run_dir(project_root: Path) -> Optional[Path]:
    """Find the most recent run directory in out/runs/."""
    runs_dir = project_root / "out" / "runs"
    if not runs_dir.exists():
        return None

    # Get all run directories sorted by modification time (newest first)
    run_dirs = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True
    )

    if run_dirs:
        logger.debug(f"Latest run dir: {run_dirs[0].name}")
        return run_dirs[0]
    return None


def create_app(run_dir: Optional[str] = None) -> Flask:
    """
    Create Flask app for cockpit.

    Args:
        run_dir: Path to the pipeline run directory to monitor.
                 If None, will look for RUN_DIR env var or auto-discover.

    Returns:
        Flask application
    """
    # Calculate project root early for auto-discovery
    project_root = Path(__file__).parent.parent  # cockpit -> project_root

    app = Flask(
        __name__,
        template_folder=str(COCKPIT_DIR / 'templates'),
        static_folder=str(COCKPIT_DIR / 'static'),
    )

    # Configuration
    app.config['SECRET_KEY'] = os.getenv('COCKPIT_SECRET', 'cockpit-secret-dev')

    # Store run_dir in config
    # NOTE: On startup, we don't auto-discover old runs.
    # The watcher will auto-discover NEW runs created after startup.
    if run_dir:
        app.config['RUN_DIR'] = Path(run_dir)
    else:
        env_run_dir = os.getenv('RUN_DIR', os.getenv('PIPELINE_RUN_DIR'))
        if env_run_dir:
            app.config['RUN_DIR'] = Path(env_run_dir)
        else:
            # Start with no run dir - watcher will find NEW runs only
            app.config['RUN_DIR'] = None
            logger.info("Starting in IDLE mode - will auto-discover new runs only")

    # Store project_root for checkpoint reading
    app.config['PROJECT_ROOT'] = project_root

    # CORS for local development
    CORS(app, origins=[
        "http://localhost:*",
        "http://127.0.0.1:*",
        "http://0.0.0.0:*",
    ])

    # Initialize SocketIO
    socketio.init_app(app)

    # Disable caching for static files in development
    @app.after_request
    def add_no_cache_headers(response):
        if 'static' in request.path:
            response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
        return response

    # Initialize pipeline controller
    global _pipeline_controller
    _pipeline_controller = PipelineController(project_root)
    app.config['PIPELINE_CONTROLLER'] = _pipeline_controller

    # Register routes
    _register_routes(app)

    # Register socket handlers
    _register_socket_handlers(app)

    # Always start background watcher to detect new runs
    _start_state_watcher(app)
    if app.config.get('RUN_DIR'):
        logger.info(f"Cockpit monitoring: {app.config['RUN_DIR']}")
    else:
        logger.info("Cockpit started - waiting for pipeline runs")

    return app


# =============================================================================
# ROUTES
# =============================================================================


def _register_routes(app: Flask):
    """Register all HTTP routes."""

    @app.route('/')
    def index():
        """Serve the main cockpit page."""
        return render_template('cockpit.html')

    @app.route('/static/<path:filename>')
    def serve_static(filename):
        """Serve static files."""
        return send_from_directory(app.static_folder, filename)

    @app.route('/api/health')
    def health():
        """Health check endpoint."""
        run_dir = app.config.get('RUN_DIR')
        return jsonify({
            "status": "ok",
            "service": "cockpit",
            "run_dir": str(run_dir) if run_dir else None,
            "run_dir_exists": run_dir.exists() if run_dir else False,
            "timestamp": datetime.now().isoformat(),
        })

    @app.route('/api/cockpit_state')
    def get_cockpit_state():
        """Get current cockpit state (REST fallback)."""
        run_dir = app.config.get('RUN_DIR')
        if not run_dir:
            # Return IDLE state when no run is active
            return jsonify({
                "nodes": {},
                "phase": "IDLE",
                "run_id": "",
                "sprint_id": "",
                "updated_at": datetime.now().isoformat(),
                "event_logs": [],
                "run_log": [],
                "redis_history": [],
                "summary": {
                    "total_nodes": 0,
                    "active_nodes": 0,
                    "complete_nodes": 0,
                    "error_nodes": 0,
                    "gates_passed": 0,
                    "gates_failed": 0,
                },
            }), 200

        try:
            project_root = app.config.get('PROJECT_ROOT')
            state = cockpit_transformer.transform_to_cockpit_state(run_dir, project_root)
            return jsonify(state)
        except Exception as e:
            logger.error(f"Error getting cockpit state: {e}")
            return jsonify({
                "error": str(e),
                "nodes": {},
                "phase": "ERROR",
            }), 200

    @app.route('/api/connections')
    def get_connections():
        """Get node connection definitions for arrows."""
        return jsonify({
            "connections": cockpit_transformer.get_node_connections()
        })

    @app.route('/api/run_dir', methods=['POST'])
    def set_run_dir():
        """Dynamically set the run directory to monitor."""
        data = request.get_json()
        new_run_dir = data.get('run_dir')

        if not new_run_dir:
            return jsonify({"error": "run_dir required"}), 400

        path = Path(new_run_dir)
        if not path.exists():
            return jsonify({"error": f"Path does not exist: {new_run_dir}"}), 400

        app.config['RUN_DIR'] = path
        logger.info(f"Run directory changed to: {path}")

        # Restart watcher
        _start_state_watcher(app)

        return jsonify({
            "status": "ok",
            "run_dir": str(path),
        })

    # =========================================================================
    # PIPELINE CONTROL ROUTES
    # =========================================================================

    @app.route('/api/pipeline/status')
    def pipeline_status():
        """Get current pipeline execution status."""
        controller = app.config.get('PIPELINE_CONTROLLER')
        if not controller:
            return jsonify({"error": "Pipeline controller not initialized"}), 500
        return jsonify(controller.get_status())

    @app.route('/api/pipeline/start', methods=['POST'])
    def pipeline_start():
        """
        Start the pipeline.

        Request body:
        {
            "start_sprint": "S00",  // optional, default "S00"
            "end_sprint": "S25"     // optional, default "S25"
        }
        """
        controller = app.config.get('PIPELINE_CONTROLLER')
        if not controller:
            return jsonify({"error": "Pipeline controller not initialized"}), 500

        data = request.get_json() or {}
        start_sprint = data.get('start_sprint', 'S00')
        end_sprint = data.get('end_sprint', 'S25')

        result = controller.start(start_sprint=start_sprint, end_sprint=end_sprint)

        if result.get('success'):
            # Broadcast status change
            socketio.emit('pipeline_status', result.get('status'))
            return jsonify(result)
        else:
            return jsonify(result), 400

    @app.route('/api/pipeline/pause', methods=['POST'])
    def pipeline_pause():
        """Pause the running pipeline."""
        controller = app.config.get('PIPELINE_CONTROLLER')
        if not controller:
            return jsonify({"error": "Pipeline controller not initialized"}), 500

        result = controller.pause()

        if result.get('success'):
            socketio.emit('pipeline_status', result.get('status'))
            return jsonify(result)
        else:
            return jsonify(result), 400

    @app.route('/api/pipeline/resume', methods=['POST'])
    def pipeline_resume():
        """Resume a paused pipeline."""
        controller = app.config.get('PIPELINE_CONTROLLER')
        if not controller:
            return jsonify({"error": "Pipeline controller not initialized"}), 500

        result = controller.resume()

        if result.get('success'):
            socketio.emit('pipeline_status', result.get('status'))
            return jsonify(result)
        else:
            return jsonify(result), 400

    @app.route('/api/pipeline/abort', methods=['POST'])
    def pipeline_abort():
        """Abort the running/paused pipeline."""
        controller = app.config.get('PIPELINE_CONTROLLER')
        if not controller:
            return jsonify({"error": "Pipeline controller not initialized"}), 500

        result = controller.abort()

        if result.get('success'):
            socketio.emit('pipeline_status', result.get('status'))
            return jsonify(result)
        else:
            return jsonify(result), 400

    @app.route('/api/pipeline/sprints')
    def pipeline_sprints():
        """Get available sprints for selection."""
        # Return available sprint ranges
        return jsonify({
            "sprints": [f"S{i:02d}" for i in range(41)],  # S00-S40
            "packs": [
                {"name": "W0: Foundation", "start": "S00", "end": "S02"},
                {"name": "W1: Core Engine", "start": "S03", "end": "S14"},
                {"name": "W2: OSS Release", "start": "S15", "end": "S24"},
                {"name": "W3: Cloud MVP", "start": "S25", "end": "S35"},
                {"name": "W4: Growth", "start": "S36", "end": "S40"},
            ],
        })

    @app.route('/api/pipeline/checkpoints')
    def pipeline_checkpoints():
        """Get available LangGraph checkpoints for resume.

        Reads checkpoints directly from filesystem for fast response.
        """
        try:
            # Find latest run_id from the runs directory
            runs_dir = app.config['PROJECT_ROOT'] / "out" / "runs"
            if not runs_dir.exists():
                return jsonify({"checkpoints": [], "error": "No runs directory found"})

            # Get most recent LangGraph run
            runs = sorted(
                [r for r in runs_dir.iterdir() if r.name.startswith("lg_")],
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )

            if not runs:
                return jsonify({"checkpoints": [], "error": "No LangGraph runs found"})

            latest_run_id = runs[0].name

            # Read checkpoints directly from filesystem
            checkpoints_dir = app.config['PROJECT_ROOT'] / ".langgraph" / "checkpoints" / "checkpoints"
            if not checkpoints_dir.exists():
                return jsonify({
                    "checkpoints": [],
                    "latest_run_id": latest_run_id,
                    "error": "No checkpoints directory found"
                })

            # Find checkpoints for the latest run
            checkpoints = []
            checkpoint_files = sorted(
                checkpoints_dir.glob(f"*{latest_run_id}*.json"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )

            # Also check for _final checkpoint (different naming)
            final_file = checkpoints_dir / f"{latest_run_id}_final.json"
            if final_file.exists():
                checkpoints.append({
                    "id": f"{latest_run_id}_final",
                    "phase": "final",
                    "run_id": latest_run_id,
                    "mtime": final_file.stat().st_mtime,
                })

            # Parse checkpoint filenames
            # Format: ckpt:lg_xxx:S00:PHASE:attempt:N:hash.json
            for cp_file in checkpoint_files[:10]:  # Limit to 10 most recent
                filename = cp_file.stem  # Remove .json
                parts = filename.split(":")

                if len(parts) >= 4:
                    # Extract phase from filename
                    phase = parts[3] if len(parts) > 3 else "unknown"
                    sprint = parts[2] if len(parts) > 2 else "unknown"

                    checkpoints.append({
                        "id": filename,
                        "phase": phase,
                        "sprint": sprint,
                        "run_id": latest_run_id,
                        "mtime": cp_file.stat().st_mtime,
                    })

            # Remove duplicates and sort by mtime
            seen_ids = set()
            unique_checkpoints = []
            for cp in sorted(checkpoints, key=lambda x: x.get("mtime", 0), reverse=True):
                if cp["id"] not in seen_ids:
                    seen_ids.add(cp["id"])
                    unique_checkpoints.append(cp)

            return jsonify({
                "checkpoints": unique_checkpoints[:10],  # Return top 10
                "latest_run_id": latest_run_id,
            })

        except Exception as e:
            logger.error(f"Failed to get checkpoints: {e}")
            return jsonify({"checkpoints": [], "error": str(e)}), 500

    @app.route('/api/pipeline/lg-resume', methods=['POST'])
    def pipeline_lg_resume():
        """
        Resume pipeline from a LangGraph checkpoint.

        Request body:
        {
            "checkpoint_id": "lg_20260127_052439_b16a4936_final"
        }
        """
        controller = app.config.get('PIPELINE_CONTROLLER')
        if not controller:
            return jsonify({"error": "Pipeline controller not initialized"}), 500

        data = request.get_json() or {}
        checkpoint_id = data.get('checkpoint_id')

        if not checkpoint_id:
            return jsonify({"error": "checkpoint_id is required"}), 400

        result = controller.resume_from_checkpoint(checkpoint_id)

        if result.get('success'):
            # Set RUN_DIR so cockpit can monitor the resumed run
            run_id = result.get('run_id')
            if run_id:
                run_dir = app.config['PROJECT_ROOT'] / "out" / "runs" / run_id
                if run_dir.exists():
                    app.config['RUN_DIR'] = run_dir
                    logger.info(f"Set RUN_DIR for resumed pipeline: {run_dir}")
                    # Emit run_changed event so frontend updates
                    socketio.emit('run_changed', {
                        'run_dir': str(run_dir),
                        'run_id': run_id,
                    })

            socketio.emit('pipeline_status', result.get('status'))
            return jsonify(result)
        else:
            return jsonify(result), 400

    @app.route('/api/pipeline/reset', methods=['POST'])
    def pipeline_reset():
        """Force reset pipeline state and kill any orphaned processes."""
        global _last_state_hash, _reset_timestamp

        controller = app.config.get('PIPELINE_CONTROLLER')
        if not controller:
            return jsonify({"error": "Pipeline controller not initialized"}), 500

        # Kill all pipeline processes
        controller._kill_all_pipeline_processes()

        # Reset controller state
        controller.current_process = None

        # Clear run_dir so cockpit shows fresh state
        app.config['RUN_DIR'] = None
        _last_state_hash = None

        # Set reset timestamp - only auto-discover runs newer than this
        _reset_timestamp = time.time()
        logger.info(f"Reset timestamp set: {_reset_timestamp}")

        # Emit clear state to all clients
        empty_state = {
            "nodes": {},
            "phase": "IDLE",
            "run_id": "",
            "sprint_id": "",
            "updated_at": datetime.now().isoformat(),
            "event_logs": [],
            "run_log": [],
            "redis_history": [],
            "summary": {
                "total_nodes": 0,
                "active_nodes": 0,
                "complete_nodes": 0,
                "error_nodes": 0,
                "gates_passed": 0,
                "gates_failed": 0,
            },
        }
        socketio.emit('cockpit_update', empty_state)
        socketio.emit('pipeline_status', controller.get_status())

        return jsonify({
            "success": True,
            "message": "Pipeline state reset and all processes killed",
            "status": controller.get_status(),
        })

    # =========================================================================
    # RESILIENCE MONITORING ROUTES
    # =========================================================================

    @app.route('/api/resilience/metrics')
    def get_resilience_metrics():
        """Get aggregated resilience state including circuit breakers, retry metrics, and oscillation."""
        try:
            state = resilience_reader.get_resilience_state()
            return jsonify(state)
        except Exception as e:
            app.logger.error(f"Error reading resilience metrics: {e}")
            return jsonify({
                "error": str(e),
                "circuit_breakers": {},
                "retry_metrics": {},
                "oscillation": {},
                "healthy": False,
            }), 500

    @app.route('/api/resilience/circuit-breakers')
    def get_circuit_breakers():
        """Get circuit breaker states with count and healthy flag."""
        try:
            breakers = resilience_reader.read_circuit_breakers()
            return jsonify({
                "circuit_breakers": breakers,
                "count": len(breakers),
                "healthy": all(cb.get("state") == "closed" for cb in breakers.values()),
            })
        except Exception as e:
            app.logger.error(f"Error reading circuit breakers: {e}")
            return jsonify({"error": str(e), "circuit_breakers": {}, "count": 0, "healthy": False}), 500

    @app.route('/api/resilience/oscillation')
    def get_oscillation_state():
        """Get oscillation detection state with alert level."""
        try:
            state = resilience_reader.read_oscillation_state()
            is_osc = state.get("is_oscillating", False)
            if is_osc:
                pattern = state.get("pattern", "none")
                if pattern == "runaway":
                    alert_level = "critical"
                elif pattern == "cycle":
                    alert_level = "warning"
                else:
                    alert_level = "info"
            else:
                alert_level = "none"
            return jsonify({
                **state,
                "alert_level": alert_level,
                "has_oscillation": is_osc,
            })
        except Exception as e:
            app.logger.error(f"Error reading oscillation state: {e}")
            return jsonify({"error": str(e), "active_patterns": [], "alert_level": "unknown", "has_oscillation": False}), 500

    @app.route('/metrics')
    def prometheus_metrics():
        """Return metrics in Prometheus format (text/plain)."""
        try:
            metrics_text = resilience_reader.read_prometheus_metrics()
            return Response(metrics_text, mimetype='text/plain; version=0.0.4; charset=utf-8')
        except Exception as e:
            app.logger.error(f"Error reading Prometheus metrics: {e}")
            error_metric = f'# HELP cockpit_error Error indicator\n# TYPE cockpit_error gauge\ncockpit_error{{reason="{str(e)}"}} 1\n'
            return Response(error_metric, mimetype='text/plain; version=0.0.4; charset=utf-8')


# =============================================================================
# SOCKET HANDLERS
# =============================================================================


def _register_socket_handlers(app: Flask):
    """Register WebSocket event handlers."""

    @socketio.on('connect')
    def handle_connect():
        """Handle client connection."""
        logger.info(f"Client connected: {request.sid}")
        # Send initial state immediately
        _send_current_state(app)

    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection."""
        logger.info(f"Client disconnected: {request.sid}")

    @socketio.on('request_state')
    def handle_request_state():
        """Handle explicit state request from client."""
        _send_current_state(app)

    @socketio.on('request_resilience')
    def handle_request_resilience():
        """
        Handle client request for current resilience state.

        Emits 'resilience_update' event with current resilience state.
        On error, emits a safe default state with available=False.
        """
        try:
            resilience = cockpit_transformer.transform_resilience_state()
            emit('resilience_update', resilience)
        except Exception as e:
            app.logger.error(f"Error handling request_resilience: {e}")
            emit('resilience_update', {
                "error": str(e),
                "available": False,
                "circuit_breakers": [],
                "oscillation": {"alert_level": "unknown", "has_alert": False},
                "retry_metrics": {},
                "overall_health": True,
            })

    @socketio.on('subscribe')
    def handle_subscribe(data):
        """Handle subscription to specific node updates."""
        node_ids = data.get('nodes', [])
        logger.info(f"Client {request.sid} subscribed to: {node_ids}")
        # Could implement per-node subscriptions here


def _send_current_state(app: Flask):
    """Send current state to requesting client."""
    run_dir = app.config.get('RUN_DIR')
    if not run_dir:
        # Send IDLE state when no run is active
        emit('cockpit_update', {
            "nodes": {},
            "phase": "IDLE",
            "run_id": "",
            "sprint_id": "",
            "updated_at": datetime.now().isoformat(),
            "event_logs": [],
            "run_log": [],
            "redis_history": [],
            "summary": {
                "total_nodes": 0,
                "active_nodes": 0,
                "complete_nodes": 0,
                "error_nodes": 0,
                "gates_passed": 0,
                "gates_failed": 0,
            },
        })
        return

    try:
        project_root = app.config.get('PROJECT_ROOT')
        state = cockpit_transformer.transform_to_cockpit_state(run_dir, project_root)
        emit('cockpit_update', state)
    except Exception as e:
        logger.error(f"Error sending state: {e}")
        emit('cockpit_update', {
            "nodes": {},
            "phase": "ERROR",
            "error": str(e),
        })


# =============================================================================
# STATE WATCHER
# =============================================================================


def _start_state_watcher(app: Flask):
    """Start background thread to watch for state changes."""
    global _watcher_running

    if _watcher_running:
        return  # Already running

    def watch_loop():
        global _last_state_hash, _last_resilience_hash, _watcher_running
        _watcher_running = True

        run_dir = app.config.get('RUN_DIR')
        poll_interval = float(os.getenv('COCKPIT_POLL_INTERVAL', '1.0'))
        project_root = app.config.get('PROJECT_ROOT')

        logger.info(f"State watcher started (poll every {poll_interval}s)")

        while _watcher_running:
            try:
                # Auto-discover newer run directories
                # This ensures we switch to new runs when pipeline starts
                latest_run = _find_latest_run_dir(project_root)

                # Only auto-discover if run is newer than startup/reset timestamp
                # This prevents showing old runs on startup or after reset
                should_discover = False
                if latest_run and latest_run != run_dir:
                    # Use the latest of startup or reset timestamp as threshold
                    threshold = max(_startup_timestamp, _reset_timestamp or 0)
                    # Check run.log mtime (constantly updated) instead of dir mtime
                    run_log_path = latest_run / "run.log"
                    if run_log_path.exists():
                        run_mtime = run_log_path.stat().st_mtime
                    else:
                        run_mtime = latest_run.stat().st_mtime
                    if run_mtime > threshold:
                        should_discover = True
                        logger.info(f"New run detected: {latest_run.name} (mtime={run_mtime:.1f}, threshold={threshold:.1f})")
                    else:
                        logger.debug(f"Ignoring old run: {latest_run.name} (mtime={run_mtime:.1f} <= threshold={threshold:.1f})")

                if should_discover:
                    logger.info(f"Detected new run: {latest_run.name}")
                    app.config['RUN_DIR'] = latest_run
                    run_dir = latest_run
                    _last_state_hash = None  # Reset on dir change
                    # Emit immediate update for new run
                    socketio.emit('run_changed', {
                        'run_dir': str(latest_run),
                        'run_id': latest_run.name,
                    })

                # Check if run_dir was manually changed (e.g., reset to None)
                current_run_dir = app.config.get('RUN_DIR')
                if current_run_dir != run_dir:
                    run_dir = current_run_dir
                    _last_state_hash = None  # Reset on dir change

                if not run_dir:
                    # Even when no run_dir, still watch resilience state
                    _watch_resilience_state(app)
                    time.sleep(poll_interval)
                    continue

                # Get current state
                state = cockpit_transformer.transform_to_cockpit_state(run_dir, project_root)

                # Calculate hash for change detection
                # Include event_logs length and last timestamp to detect new events
                event_logs = state.get('event_logs', [])
                run_log = state.get('run_log', [])
                redis_history = state.get('redis_history', [])

                hash_parts = [
                    str(state.get('summary', {})),
                    state.get('phase', ''),
                    str(len(event_logs)),
                    str(len(run_log)),
                    str(len(redis_history)),
                    event_logs[-1].get('timestamp', '') if event_logs else '',
                    run_log[-1] if run_log else '',  # Include last log line content
                ]
                state_hash = hash(''.join(hash_parts))

                # Only broadcast if changed
                if state_hash != _last_state_hash:
                    socketio.emit('cockpit_update', state)
                    _last_state_hash = state_hash
                    logger.info(f"State update broadcast: phase={state.get('phase')}, events={len(event_logs)}, logs={len(run_log)}")

                # Watch resilience state separately (INV-P4-001: only emit on change)
                _watch_resilience_state(app)

                time.sleep(poll_interval)

            except Exception as e:
                logger.error(f"Watcher error: {e}")
                time.sleep(5)  # Back off on error

        _watcher_running = False
        logger.info("State watcher stopped")

    thread = threading.Thread(target=watch_loop, daemon=True, name="cockpit-watcher")
    thread.start()


def _watch_resilience_state(app: Flask) -> None:
    """
    Watch resilience state and emit updates on change.

    This function is called from the watcher loop and handles resilience
    state monitoring separately from pipeline state. It uses hash-based
    change detection to only emit updates when data actually changes.

    INV-P4-001: Only emits when data changes (hash check)
    INV-P4-004: Never crashes server (try/except for all operations)
    INV-P4-006: Uses try/except for ALL operations
    """
    global _last_resilience_hash

    try:
        resilience = cockpit_transformer.transform_resilience_state()
        resilience_hash = _calculate_hash(resilience)

        # Only broadcast if changed (INV-P4-001)
        if resilience_hash != _last_resilience_hash:
            socketio.emit('resilience_update', resilience)
            _last_resilience_hash = resilience_hash
            logger.debug(f"Resilience update broadcast: health={resilience.get('overall_health')}, breakers={len(resilience.get('circuit_breakers', []))}")

    except Exception as e:
        # INV-P4-004: Never crash the server on resilience errors
        app.logger.warning(f"Resilience watch error (non-fatal): {e}")


def stop_watcher():
    """Stop the background watcher."""
    global _watcher_running
    _watcher_running = False


# =============================================================================
# CLI ENTRY POINT
# =============================================================================


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='HumanGR Cockpit Dashboard Server')
    parser.add_argument('run_dir', nargs='?', default=None,
                        help='Path to pipeline run directory')
    parser.add_argument('--port', type=int, default=5002,
                        help='Port to run on (default: 5002)')
    parser.add_argument('--host', default='0.0.0.0',
                        help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')

    args = parser.parse_args()

    # Create app
    app = create_app(args.run_dir)

    # Print startup info
    print(f"""
+============================================================================+
|                       HUMANGR COCKPIT DASHBOARD                            |
+============================================================================+
|  URL:     http://{args.host}:{args.port}                                            |
|  Run Dir: {str(app.config.get('RUN_DIR', 'Not configured'))[:55]:<55} |
|  Debug:   {str(args.debug):<55} |
+============================================================================+
    """)

    # Run server
    socketio.run(app, host=args.host, port=args.port, debug=args.debug, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    main()
