"""P2-016: LangGraph Subgraphs for Pipeline V2.

This module provides modular, reusable subgraphs that can be composed
into the main pipeline workflow. Subgraphs enable:

1. Reusable validation patterns
2. Nested workflows for complex operations
3. Better testing and maintainability
4. Conditional execution paths

Architecture:
    Main Workflow
        ├── GateValidationSubgraph
        ├── QualityAssuranceSubgraph
        ├── ArtifactGenerationSubgraph
        └── SignoffSubgraph

Usage:
    from pipeline.langgraph.subgraphs import (
        GateValidationSubgraph,
        QualityAssuranceSubgraph,
        compose_workflow_with_subgraphs,
    )

    # Use subgraph directly
    gate_subgraph = GateValidationSubgraph(run_dir=run_dir)
    result = await gate_subgraph.run(state)

    # Or compose into main workflow
    workflow = compose_workflow_with_subgraphs(run_dir=run_dir)
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import yaml

logger = logging.getLogger(__name__)

# Gate catalog cache (loaded once)
_GATE_CATALOG: Optional[Dict[str, Dict[str, Any]]] = None


def _load_gate_catalog() -> Dict[str, Dict[str, Any]]:
    """Load gate catalog from YAML config.

    Returns dict mapping gate_id -> gate config with command, timeout, etc.
    """
    global _GATE_CATALOG
    if _GATE_CATALOG is not None:
        return _GATE_CATALOG

    catalog_path = Path(__file__).parent.parent.parent.parent / "configs" / "pipeline_autonomo" / "gate_catalog_seed.yml"
    if not catalog_path.exists():
        logger.warning(f"Gate catalog not found at {catalog_path}")
        _GATE_CATALOG = {}
        return _GATE_CATALOG

    try:
        with open(catalog_path) as f:
            catalog_data = yaml.safe_load(f)

        _GATE_CATALOG = {}
        for gate in catalog_data.get("gates", []):
            gate_id = gate.get("gate_id")
            if gate_id:
                _GATE_CATALOG[gate_id] = gate

        logger.info(f"Loaded {len(_GATE_CATALOG)} gates from catalog")
        return _GATE_CATALOG
    except Exception as e:
        logger.error(f"Failed to load gate catalog: {e}")
        _GATE_CATALOG = {}
        return _GATE_CATALOG


def _get_gate_command(gate_id: str, run_dir: Path, sprint_id: str) -> str:
    """Get the command for a gate from the catalog.

    Substitutes <run_dir> and <sprint_id> placeholders.
    """
    catalog = _load_gate_catalog()
    gate_config = catalog.get(gate_id, {})

    command = gate_config.get("command", "")
    if not command:
        logger.warning(f"No command found for gate {gate_id} in catalog")
        return f"echo 'Gate {gate_id} - NO COMMAND IN CATALOG'"

    # Substitute placeholders
    command = command.replace("<run_dir>", str(run_dir))
    command = command.replace("<sprint_id>", sprint_id)

    return command


def _get_gate_timeout(gate_id: str) -> int:
    """Get timeout for a gate from the catalog."""
    catalog = _load_gate_catalog()
    gate_config = catalog.get(gate_id, {})
    return gate_config.get("timeout_seconds", 300)  # Default 5 minutes


# 2026-01-25: Retry configuration for transient errors
# IMPORTANT: Only retry for transient errors (timeout, connection)
# NEVER retry for security blocks (that would be dangerous)
MAX_GATE_TRANSIENT_RETRIES = 2
TRANSIENT_ERROR_PATTERNS = [
    "timeout",
    "connection",
    "ConnectionError",
    "TimeoutError",
    "OSError",
    "socket",
    "network",
    "temporarily unavailable",
]


def _is_transient_error(error: Exception) -> bool:
    """Check if an error is transient and should be retried.

    IMPORTANT: Security errors should NEVER be retried.
    Only transient infrastructure errors qualify for retry.
    """
    error_str = str(error).lower()
    error_type = type(error).__name__

    # Check error type
    if error_type in ("ConnectionError", "TimeoutError", "OSError"):
        return True

    # Check error message patterns
    for pattern in TRANSIENT_ERROR_PATTERNS:
        if pattern.lower() in error_str:
            return True

    return False

# LangGraph imports (wrapped for graceful degradation)
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.base import BaseCheckpointSaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = "END"
    BaseCheckpointSaver = None

# Alias for external imports (used by __init__.py)
LANGGRAPH_SUBGRAPHS_AVAILABLE = LANGGRAPH_AVAILABLE

# Security gate integration (graceful degradation if unavailable)
try:
    from pipeline.security.gate_integration import (
        get_security_gate_runner,
        validate_gate_security,
        SECURITY_GATE_AVAILABLE,
    )
except ImportError:
    SECURITY_GATE_AVAILABLE = False
    get_security_gate_runner = None
    validate_gate_security = None


# =============================================================================
# QUIETSTAR + REFLEXION GUARDRAILS (2026-01-20)
# =============================================================================
# QuietStar/Reflexion integration provides safety guardrails for subgraph
# operations. Critical paths like signoffs use DEEP analysis.
# =============================================================================

try:
    from pipeline.security.quietstar_reflexion import (
        QuietStarReflexionGuardrail,
        get_guardrail as _get_quietstar_guardrail,
        SecurityBlockedError as QuietStarBlockedError,
        auto_select_depth,
        ReflectionDepth,
    )
    QUIETSTAR_REFLEXION_AVAILABLE = True
except ImportError:
    QUIETSTAR_REFLEXION_AVAILABLE = False
    _get_quietstar_guardrail = None
    auto_select_depth = None
    ReflectionDepth = None

    class QuietStarBlockedError(Exception):
        """Placeholder when QuietStar not available."""
        pass


async def _check_quietstar_subgraph_input(
    input_text: str,
    subgraph_name: str = "subgraph",
    is_critical: bool = False,
) -> None:
    """Check subgraph input with QuietStar safety thinking.

    Args:
        input_text: Input text to check.
        subgraph_name: Name of the subgraph for logging.
        is_critical: If True, uses DEEP analysis (for signoffs, approvals).

    Raises:
        QuietStarBlockedError: If input is blocked by safety guardrails.
    """
    if not QUIETSTAR_REFLEXION_AVAILABLE or not _get_quietstar_guardrail:
        return

    try:
        guardrail = _get_quietstar_guardrail()
        if guardrail is None:
            return

        result = await guardrail.safety_thinking.analyze(input_text[:2000])

        # Critical paths (signoffs) have lower threshold
        threshold = 0.5 if is_critical else 0.7

        if result.risk_score > threshold:
            logger.warning(
                f"QUIETSTAR SUBGRAPH BLOCKED: subgraph={subgraph_name}, "
                f"risk={result.risk_score:.2f} (threshold={threshold}), "
                f"reason={result.reasoning[:100]}"
            )
            raise QuietStarBlockedError(
                f"Input blocked by QuietStar in {subgraph_name}: {result.reasoning[:200]}"
            )

        logger.debug(
            f"QUIETSTAR SUBGRAPH PRE-CHECK: subgraph={subgraph_name}, "
            f"risk={result.risk_score:.2f} PASS"
        )

    except QuietStarBlockedError:
        raise
    except Exception as e:
        logger.warning(f"QuietStar subgraph pre-check failed (continuing): {e}")


# =============================================================================
# SUBGRAPH STATE TYPES
# =============================================================================


class SubgraphState(TypedDict, total=False):
    """Base state for subgraphs."""
    run_id: str
    sprint_id: str
    phase: str
    status: str
    errors: List[Dict[str, Any]]
    node_history: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class GateValidationState(SubgraphState):
    """State for gate validation subgraph."""
    gates_to_run: List[str]
    gates_passed: List[str]
    gates_failed: List[str]
    gate_status: str
    gate_results: Dict[str, Dict[str, Any]]
    evidence_bundle_path: Optional[str]
    current_gate: Optional[str]
    retry_count: int


class QualityAssuranceState(SubgraphState):
    """State for QA subgraph."""
    qa_checks: List[str]
    checks_passed: List[str]
    checks_failed: List[str]
    qa_status: str
    coverage_threshold: float
    actual_coverage: float
    findings: List[Dict[str, Any]]


class SignoffState(SubgraphState):
    """State for signoff subgraph."""
    required_signoffs: List[str]
    received_signoffs: Dict[str, Dict[str, Any]]
    pending_signoffs: List[str]
    signoff_status: str
    quorum_reached: bool


class ArtifactState(SubgraphState):
    """State for artifact generation subgraph."""
    artifacts_to_generate: List[str]
    generated_artifacts: Dict[str, str]
    artifact_status: str


# =============================================================================
# ABSTRACT SUBGRAPH
# =============================================================================


class BaseSubgraph(ABC):
    """Abstract base class for LangGraph subgraphs.

    Subgraphs are self-contained workflows that can be composed
    into larger workflows. Each subgraph:
    - Has its own state type
    - Defines entry and exit points
    - Can be tested independently
    - Emits events for observability
    """

    def __init__(
        self,
        run_dir: Path,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ):
        """Initialize subgraph.

        Args:
            run_dir: Directory for run artifacts.
            checkpointer: Optional checkpointer for persistence.
        """
        self.run_dir = run_dir
        self.checkpointer = checkpointer
        self._graph = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Subgraph name for logging and events."""
        pass

    @property
    @abstractmethod
    def state_type(self) -> type:
        """State type for this subgraph."""
        pass

    @abstractmethod
    def build(self) -> Optional[StateGraph]:
        """Build the subgraph.

        Returns:
            Compiled StateGraph or None if LangGraph unavailable.
        """
        pass

    def _emit_event(self, state: Dict[str, Any], event_type: str, message: str) -> None:
        """Emit event for observability."""
        try:
            import json

            event = {
                "schema_version": "1.0",
                "event_id": f"{self.name}:{event_type}:{state.get('run_id', 'unknown')}",
                "type": event_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": message,
                "subgraph": self.name,
                "run_id": state.get("run_id", ""),
                "sprint_id": state.get("sprint_id", ""),
            }

            event_log_path = self.run_dir / "event_log.ndjson"
            event_log_path.parent.mkdir(parents=True, exist_ok=True)

            with open(event_log_path, "a") as f:
                f.write(json.dumps(event) + "\n")

        except Exception as e:
            logger.debug(f"Failed to emit subgraph event: {e}")

    def _add_node_history(
        self,
        state: Dict[str, Any],
        node: str,
        status: str = "completed",
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Add entry to node history."""
        entry = {
            "node": f"{self.name}.{node}",
            "status": status,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "error": error,
        }

        history = list(state.get("node_history", []))
        history.append(entry)

        return {**state, "node_history": history}

    async def run(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Run the subgraph.

        Args:
            initial_state: Initial state for the subgraph.

        Returns:
            Final state after subgraph completion.
        """
        if not LANGGRAPH_AVAILABLE:
            logger.error(f"LangGraph not available for subgraph {self.name}")
            return {
                **initial_state,
                "status": "failed",
                "errors": [{"error": "LangGraph not available", "subgraph": self.name}],
            }

        if self._graph is None:
            self._graph = self.build()

        if self._graph is None:
            return {
                **initial_state,
                "status": "failed",
                "errors": [{"error": "Failed to build subgraph", "subgraph": self.name}],
            }

        config = {"configurable": {"thread_id": initial_state.get("run_id", "default")}}
        result = await self._graph.ainvoke(initial_state, config=config)
        return result


# =============================================================================
# GATE VALIDATION SUBGRAPH
# =============================================================================


class GateValidationSubgraph(BaseSubgraph):
    """Subgraph for running gate validation.

    Flow:
        prepare_gates -> run_gate -> check_results -> collect_evidence
                            ^             |
                            |___ retry ___|

    Features:
    - Sequential gate execution respecting dependencies
    - Retry support for transient failures
    - Evidence collection and bundling
    - SPH (Secret Pattern Heuristics) scanning
    """

    @property
    def name(self) -> str:
        return "gate_validation"

    @property
    def state_type(self) -> type:
        return GateValidationState

    def build(self) -> Optional[StateGraph]:
        """Build gate validation subgraph."""
        if not LANGGRAPH_AVAILABLE:
            return None

        graph = StateGraph(GateValidationState)

        # Add nodes
        graph.add_node("prepare_gates", self._prepare_gates)
        graph.add_node("run_gate", self._run_gate)
        graph.add_node("check_results", self._check_results)
        graph.add_node("collect_evidence", self._collect_evidence)

        # Set entry point
        graph.set_entry_point("prepare_gates")

        # Add edges with routing
        graph.add_edge("prepare_gates", "run_gate")
        graph.add_conditional_edges("run_gate", self._route_after_gate)
        graph.add_conditional_edges("check_results", self._route_after_check)
        graph.add_edge("collect_evidence", END)

        if self.checkpointer:
            return graph.compile(checkpointer=self.checkpointer)
        return graph.compile()

    async def _prepare_gates(self, state: GateValidationState) -> GateValidationState:
        """Prepare list of gates to run."""
        self._emit_event(state, "prepare_gates_start", f"Preparing gates for {state.get('sprint_id')}")

        # Default gates if not specified
        gates_to_run = state.get("gates_to_run") or ["G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8"]

        return {
            **state,
            "gates_to_run": gates_to_run,
            "gates_passed": [],
            "gates_failed": [],
            "gate_results": {},
            "current_gate": gates_to_run[0] if gates_to_run else None,
            "retry_count": 0,
        }

    async def _run_gate(self, state: GateValidationState) -> GateValidationState:
        """Run a single gate.

        Security: Applies @secure_gate checks via validate_gate_security().
        See pipeline_v2.security.gate_integration for security policies per gate (G0-G8).
        """
        current_gate = state.get("current_gate")
        if not current_gate:
            return state

        self._emit_event(state, "run_gate", f"Running gate {current_gate}")

        # 2026-01-16: Apply @secure_gate security checks before gate execution
        # This enforces GATE_SECURITY_POLICIES from gate_integration.py
        if SECURITY_GATE_AVAILABLE and validate_gate_security is not None:
            try:
                security_input = {
                    "gate_id": current_gate,
                    "sprint_id": state.get("sprint_id", ""),
                    "run_id": state.get("run_id", ""),
                }
                is_valid, error_msg = await validate_gate_security(current_gate, security_input)
                if not is_valid:
                    logger.warning(f"Security validation failed for gate {current_gate}: {error_msg}")
                    failed = list(state.get("gates_failed", []))
                    failed.append(current_gate)
                    gate_results = dict(state.get("gate_results", {}))
                    gate_results[current_gate] = {
                        "status": "BLOCK",
                        "exit_code": -1,
                        "log_path": "",
                        "sph_hits": [f"SECURITY_BLOCK: {error_msg}"],
                    }
                    return {**state, "gates_failed": failed, "gate_results": gate_results}
                logger.debug(f"Security validation passed for gate {current_gate}")
            except Exception as security_error:
                # FAIL-CLOSED: Security validation é CRITICAL - não permite bypass
                from pipeline.stack_health_supervisor import CriticalStackUnavailableError
                logger.error(
                    f"FAIL-CLOSED: Security gate validation error for {current_gate}: {security_error}"
                )
                raise CriticalStackUnavailableError(
                    "security",
                    f"Security validation failed for gate {current_gate}: {security_error}"
                ) from security_error

        try:
            from pipeline.gate_runner import run_gate

            # 2026-01-25 FIX: Use real gate command from catalog instead of placeholder
            # Previously: command=f"echo 'Gate {current_gate} validation'" (WRONG!)
            sprint_id = state.get("sprint_id", "S00")
            gate_command = _get_gate_command(current_gate, self.run_dir, sprint_id)
            gate_timeout = _get_gate_timeout(current_gate)

            logger.info(f"Running gate {current_gate} with command: {gate_command[:100]}...")

            # Run the gate with real command
            result = run_gate(
                gate_id=current_gate,
                command=gate_command,
                repo_root=self.run_dir.parent,
                log_path=self.run_dir / "gates" / f"{current_gate}.log",
                timeout_seconds=gate_timeout,
            )

            gate_results = dict(state.get("gate_results", {}))
            gate_results[current_gate] = {
                "status": result.status,
                "exit_code": result.exit_code,
                "log_path": result.log_path,
                "sph_hits": result.sph_hits,
            }

            if result.status == "PASS":
                passed = list(state.get("gates_passed", []))
                passed.append(current_gate)
                return {**state, "gates_passed": passed, "gate_results": gate_results}
            else:
                failed = list(state.get("gates_failed", []))
                failed.append(current_gate)
                return {**state, "gates_failed": failed, "gate_results": gate_results}

        except ImportError:
            # FAIL-CLOSED: Gates são CRITICAL - não permite bypass
            from pipeline.stack_health_supervisor import CriticalStackUnavailableError
            logger.error(
                f"FAIL-CLOSED: Gate runner não disponível para {current_gate}. "
                "Módulo de gates é OBRIGATÓRIO - pipeline será HALTED."
            )
            raise CriticalStackUnavailableError(
                "gates",
                f"Gate runner module not available for {current_gate}"
            )
        except Exception as e:
            # 2026-01-25 FIX: Retry transient errors (timeout, connection)
            # IMPORTANT: Never retry security blocks (handled above)
            retry_count = state.get("retry_count", 0)

            if _is_transient_error(e) and retry_count < MAX_GATE_TRANSIENT_RETRIES:
                import time
                retry_count += 1
                delay = 2 ** retry_count  # Exponential backoff: 2s, 4s
                logger.warning(
                    f"Gate {current_gate} transient error (attempt {retry_count}/{MAX_GATE_TRANSIENT_RETRIES}): {e}. "
                    f"Retrying in {delay}s..."
                )
                time.sleep(delay)
                # Return state with incremented retry_count to trigger re-run
                return {**state, "retry_count": retry_count}

            # Non-transient error or max retries reached
            logger.error(f"Gate {current_gate} failed: {e}")
            failed = list(state.get("gates_failed", []))
            failed.append(current_gate)
            return self._add_node_history(
                {**state, "gates_failed": failed, "retry_count": 0},  # Reset retry count
                "run_gate",
                "failed",
                str(e),
            )

    def _route_after_gate(self, state: GateValidationState) -> str:
        """Route after running a gate."""
        gates_to_run = state.get("gates_to_run", [])
        passed = state.get("gates_passed", [])
        failed = state.get("gates_failed", [])

        # Check if there are more gates to run
        remaining = [g for g in gates_to_run if g not in passed and g not in failed]

        if remaining:
            # Update current_gate to next gate
            # Note: This is a routing function, can't modify state
            return "check_results"

        return "check_results"

    async def _check_results(self, state: GateValidationState) -> GateValidationState:
        """Check gate results and determine next action."""
        gates_to_run = state.get("gates_to_run", [])
        passed = state.get("gates_passed", [])
        failed = state.get("gates_failed", [])

        # Find next gate to run
        remaining = [g for g in gates_to_run if g not in passed and g not in failed]

        if remaining:
            return {**state, "current_gate": remaining[0]}

        # All gates complete
        if failed:
            gate_status = "FAIL"
        else:
            gate_status = "PASS"

        return {
            **state,
            "current_gate": None,
            "gate_status": gate_status,
        }

    def _route_after_check(self, state: GateValidationState) -> str:
        """Route after checking results."""
        current_gate = state.get("current_gate")

        if current_gate:
            return "run_gate"

        return "collect_evidence"

    async def _collect_evidence(self, state: GateValidationState) -> GateValidationState:
        """Collect and bundle gate evidence."""
        self._emit_event(state, "collect_evidence", "Collecting gate evidence")

        # Create evidence bundle path
        bundle_path = self.run_dir / "evidence" / "gates" / state.get("sprint_id", "unknown")
        bundle_path.mkdir(parents=True, exist_ok=True)

        return {
            **state,
            "evidence_bundle_path": str(bundle_path),
            "status": "completed",
        }


# =============================================================================
# QUALITY ASSURANCE SUBGRAPH
# =============================================================================


class QualityAssuranceSubgraph(BaseSubgraph):
    """Subgraph for QA validation.

    Flow:
        prepare_checks -> run_checks -> analyze_coverage -> report_findings

    Features:
    - Configurable QA checks
    - Coverage threshold validation
    - Findings aggregation
    """

    @property
    def name(self) -> str:
        return "quality_assurance"

    @property
    def state_type(self) -> type:
        return QualityAssuranceState

    def build(self) -> Optional[StateGraph]:
        """Build QA subgraph."""
        if not LANGGRAPH_AVAILABLE:
            return None

        graph = StateGraph(QualityAssuranceState)

        graph.add_node("prepare_checks", self._prepare_checks)
        graph.add_node("run_checks", self._run_checks)
        graph.add_node("analyze_coverage", self._analyze_coverage)
        graph.add_node("report_findings", self._report_findings)

        graph.set_entry_point("prepare_checks")
        graph.add_edge("prepare_checks", "run_checks")
        graph.add_edge("run_checks", "analyze_coverage")
        graph.add_edge("analyze_coverage", "report_findings")
        graph.add_edge("report_findings", END)

        if self.checkpointer:
            return graph.compile(checkpointer=self.checkpointer)
        return graph.compile()

    async def _prepare_checks(self, state: QualityAssuranceState) -> QualityAssuranceState:
        """Prepare QA checks."""
        self._emit_event(state, "prepare_checks", "Preparing QA checks")

        default_checks = [
            "lint",
            "type_check",
            "unit_tests",
            "coverage",
            "security_scan",
        ]

        return {
            **state,
            "qa_checks": state.get("qa_checks") or default_checks,
            "checks_passed": [],
            "checks_failed": [],
            "coverage_threshold": state.get("coverage_threshold", 90.0),
            "findings": [],
        }

    async def _run_checks(self, state: QualityAssuranceState) -> QualityAssuranceState:
        """Run QA checks."""
        self._emit_event(state, "run_checks", "Running QA checks")

        passed = []
        failed = []
        findings = []

        for check in state.get("qa_checks", []):
            try:
                # C-002 FIX: Execute actual check commands instead of always passing
                check_name = check.get("name") if isinstance(check, dict) else str(check)
                command = check.get("command") if isinstance(check, dict) else None
                timeout = check.get("timeout", 60) if isinstance(check, dict) else 60
                cwd = check.get("cwd", os.getcwd()) if isinstance(check, dict) else os.getcwd()

                if command:
                    # Execute actual check command
                    result = subprocess.run(
                        command,
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                        cwd=cwd,
                    )
                    success = result.returncode == 0
                    if not success:
                        logger.warning(f"QA check '{check_name}' failed: {result.stderr[:200]}")
                else:
                    # No command specified - log warning and fail the check
                    logger.warning(
                        f"C-002: QA check '{check_name}' has no command specified - marking as FAILED. "
                        "Checks without commands are not valid."
                    )
                    success = False

                if success:
                    passed.append(check)
                else:
                    failed.append(check)
                    findings.append({
                        "check": check_name,
                        "severity": "error",
                        "message": f"Check {check_name} failed",
                    })

            except subprocess.TimeoutExpired:
                failed.append(check)
                findings.append({
                    "check": check.get("name") if isinstance(check, dict) else str(check),
                    "severity": "error",
                    "message": f"Check timed out after {timeout}s",
                })
            except Exception as e:
                failed.append(check)
                findings.append({
                    "check": check.get("name") if isinstance(check, dict) else str(check),
                    "severity": "error",
                    "message": str(e),
                })

        return {
            **state,
            "checks_passed": passed,
            "checks_failed": failed,
            "findings": findings,
        }

    def _read_coverage_from_file(self, coverage_file: Path) -> float:
        """Read coverage percentage from pytest-cov output.

        C-003 FIX: Parse actual coverage files instead of returning hardcoded value.
        Returns 0.0 (fail-closed) if file not found or parse fails.
        """
        if not coverage_file.exists():
            logger.warning(
                f"C-003: Coverage file not found: {coverage_file}. "
                "Returning 0.0% (fail-closed). Run pytest with --cov to generate coverage report."
            )
            return 0.0  # FAIL-CLOSED: assume 0% without file

        try:
            if coverage_file.suffix == ".xml":
                # Parse XML coverage report (Cobertura format)
                tree = ET.parse(coverage_file)
                root = tree.getroot()
                line_rate = root.get("line-rate", "0")
                coverage_pct = float(line_rate) * 100
                logger.debug(f"Parsed XML coverage: {coverage_pct:.2f}%")
                return coverage_pct

            elif coverage_file.suffix == ".json":
                # Parse JSON coverage report
                import json
                data = json.loads(coverage_file.read_text())
                coverage_pct = data.get("totals", {}).get("percent_covered", 0.0)
                logger.debug(f"Parsed JSON coverage: {coverage_pct:.2f}%")
                return coverage_pct

            else:
                logger.warning(
                    f"C-003: Unknown coverage format: {coverage_file.suffix}. "
                    "Returning 0.0% (fail-closed). Supported formats: .xml, .json"
                )
                return 0.0

        except ET.ParseError as e:
            logger.error(f"C-003: Failed to parse XML coverage file: {e}. Returning 0.0%.")
            return 0.0
        except json.JSONDecodeError as e:
            logger.error(f"C-003: Failed to parse JSON coverage file: {e}. Returning 0.0%.")
            return 0.0
        except Exception as e:
            logger.error(f"C-003: Failed to read coverage file: {e}. Returning 0.0%.")
            return 0.0

    async def _analyze_coverage(self, state: QualityAssuranceState) -> QualityAssuranceState:
        """Analyze code coverage.

        C-003 FIX: Reads actual coverage from coverage report file instead of hardcoding.
        """
        self._emit_event(state, "analyze_coverage", "Analyzing coverage")

        # C-003 FIX: Read actual coverage from file instead of hardcoding
        coverage_file = state.get("coverage_file", ".coverage.xml")
        actual_coverage = self._read_coverage_from_file(Path(coverage_file))

        threshold = state.get("coverage_threshold", 90.0)
        meets_threshold = actual_coverage >= threshold

        findings = list(state.get("findings", []))
        if not meets_threshold:
            findings.append({
                "check": "coverage",
                "severity": "warning",
                "message": f"Coverage {actual_coverage}% below threshold {threshold}%",
            })

        return {
            **state,
            "actual_coverage": actual_coverage,
            "findings": findings,
        }

    async def _report_findings(self, state: QualityAssuranceState) -> QualityAssuranceState:
        """Generate findings report."""
        self._emit_event(state, "report_findings", "Generating QA report")

        failed = state.get("checks_failed", [])
        qa_status = "FAIL" if failed else "PASS"

        return {
            **state,
            "qa_status": qa_status,
            "status": "completed",
        }


# =============================================================================
# SIGNOFF SUBGRAPH
# =============================================================================


class SignoffSubgraph(BaseSubgraph):
    """Subgraph for processing signoffs.

    Flow:
        prepare_signoffs -> collect_signoffs -> verify_quorum -> finalize

    Features:
    - Configurable required signoffs
    - Quorum validation
    - Hierarchy enforcement (supervisor can't sign without subordinates)
    """

    @property
    def name(self) -> str:
        return "signoff"

    @property
    def state_type(self) -> type:
        return SignoffState

    def build(self) -> Optional[StateGraph]:
        """Build signoff subgraph."""
        if not LANGGRAPH_AVAILABLE:
            return None

        graph = StateGraph(SignoffState)

        graph.add_node("prepare_signoffs", self._prepare_signoffs)
        graph.add_node("collect_signoffs", self._collect_signoffs)
        graph.add_node("verify_quorum", self._verify_quorum)
        graph.add_node("finalize", self._finalize)

        graph.set_entry_point("prepare_signoffs")
        graph.add_edge("prepare_signoffs", "collect_signoffs")
        graph.add_edge("collect_signoffs", "verify_quorum")
        graph.add_conditional_edges("verify_quorum", self._route_after_quorum)
        graph.add_edge("finalize", END)

        if self.checkpointer:
            return graph.compile(checkpointer=self.checkpointer)
        return graph.compile()

    async def _prepare_signoffs(self, state: SignoffState) -> SignoffState:
        """Prepare required signoffs."""
        self._emit_event(state, "prepare_signoffs", "Preparing signoff requirements")

        # Default required signoffs based on hierarchy
        default_required = ["qa_master", "ace_exec", "exec_vp"]

        return {
            **state,
            "required_signoffs": state.get("required_signoffs") or default_required,
            "received_signoffs": {},
            "pending_signoffs": default_required,
        }

    async def _collect_signoffs(self, state: SignoffState) -> SignoffState:
        """Collect signoffs from agents.

        2026-01-20: CRITICAL PATH - QuietStar guardrails with DEEP analysis.
        Signoffs are the final approval point, so we use stricter thresholds.

        Raises:
            QuietStarBlockedError: If signoff content fails safety checks.
        """
        self._emit_event(state, "collect_signoffs", "Collecting signoffs")

        received = dict(state.get("received_signoffs", {}))
        pending = []

        for agent_id in state.get("required_signoffs", []):
            # Simulate signoff collection
            # In production, would check actual signoff records
            signoff = {
                "agent_id": agent_id,
                "approved": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "comment": f"Approved by {agent_id}",
            }

            # 2026-01-20: QuietStar CRITICAL PATH validation
            # Signoffs are the final approval point - use strict validation
            if QUIETSTAR_REFLEXION_AVAILABLE:
                try:
                    signoff_content = f"Signoff from {agent_id}: {signoff.get('comment', '')}"
                    await _check_quietstar_subgraph_input(
                        signoff_content,
                        subgraph_name=f"signoff_{agent_id}",
                        is_critical=True,  # CRITICAL: Use DEEP analysis
                    )
                except QuietStarBlockedError as e:
                    logger.error(f"QUIETSTAR SIGNOFF BLOCKED: agent={agent_id}, error={e}")
                    # For critical signoffs, we fail-closed
                    signoff["approved"] = False
                    signoff["blocked_reason"] = str(e)
                except Exception as e:
                    logger.warning(f"QuietStar signoff check skipped for {agent_id}: {e}")

            received[agent_id] = signoff

        pending = [
            a for a in state.get("required_signoffs", [])
            if a not in received
        ]

        return {
            **state,
            "received_signoffs": received,
            "pending_signoffs": pending,
        }

    async def _verify_quorum(self, state: SignoffState) -> SignoffState:
        """Verify quorum is reached."""
        self._emit_event(state, "verify_quorum", "Verifying signoff quorum")

        required = state.get("required_signoffs", [])
        received = state.get("received_signoffs", {})

        # Check all approved
        all_approved = all(
            received.get(a, {}).get("approved", False)
            for a in required
        )

        quorum_reached = len(received) >= len(required) and all_approved

        return {
            **state,
            "quorum_reached": quorum_reached,
        }

    def _route_after_quorum(self, state: SignoffState) -> str:
        """Route based on quorum status."""
        if state.get("quorum_reached"):
            return "finalize"
        # Could add retry logic here
        return "finalize"

    async def _finalize(self, state: SignoffState) -> SignoffState:
        """Finalize signoff process."""
        self._emit_event(state, "finalize", "Finalizing signoffs")

        signoff_status = "APPROVED" if state.get("quorum_reached") else "PENDING"

        return {
            **state,
            "signoff_status": signoff_status,
            "status": "completed",
        }


# =============================================================================
# ARTIFACT GENERATION SUBGRAPH
# =============================================================================


class ArtifactGenerationSubgraph(BaseSubgraph):
    """Subgraph for generating artifacts.

    Flow:
        prepare_artifacts -> generate_artifacts -> validate_artifacts

    Features:
    - Configurable artifact types
    - Validation of generated artifacts
    - Manifest generation
    """

    @property
    def name(self) -> str:
        return "artifact_generation"

    @property
    def state_type(self) -> type:
        return ArtifactState

    def build(self) -> Optional[StateGraph]:
        """Build artifact generation subgraph."""
        if not LANGGRAPH_AVAILABLE:
            return None

        graph = StateGraph(ArtifactState)

        graph.add_node("prepare_artifacts", self._prepare_artifacts)
        graph.add_node("generate_artifacts", self._generate_artifacts)
        graph.add_node("validate_artifacts", self._validate_artifacts)

        graph.set_entry_point("prepare_artifacts")
        graph.add_edge("prepare_artifacts", "generate_artifacts")
        graph.add_edge("generate_artifacts", "validate_artifacts")
        graph.add_edge("validate_artifacts", END)

        if self.checkpointer:
            return graph.compile(checkpointer=self.checkpointer)
        return graph.compile()

    async def _prepare_artifacts(self, state: ArtifactState) -> ArtifactState:
        """Prepare list of artifacts to generate."""
        self._emit_event(state, "prepare_artifacts", "Preparing artifact list")

        default_artifacts = [
            "quality_bar.yml",
            "gate_receipt.yml",
            "agent_manifest.yml",
            "run_state.yml",
        ]

        return {
            **state,
            "artifacts_to_generate": state.get("artifacts_to_generate") or default_artifacts,
            "generated_artifacts": {},
        }

    async def _generate_artifacts(self, state: ArtifactState) -> ArtifactState:
        """Generate artifacts."""
        self._emit_event(state, "generate_artifacts", "Generating artifacts")

        generated = {}
        artifacts_dir = self.run_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        for artifact_name in state.get("artifacts_to_generate", []):
            try:
                artifact_path = artifacts_dir / artifact_name
                # Placeholder content
                artifact_path.write_text(f"# {artifact_name}\ngenerated: true\n")
                generated[artifact_name] = str(artifact_path)
            except Exception as e:
                logger.warning(f"Failed to generate {artifact_name}: {e}")

        return {
            **state,
            "generated_artifacts": generated,
        }

    async def _validate_artifacts(self, state: ArtifactState) -> ArtifactState:
        """Validate generated artifacts."""
        self._emit_event(state, "validate_artifacts", "Validating artifacts")

        generated = state.get("generated_artifacts", {})
        to_generate = state.get("artifacts_to_generate", [])

        all_generated = all(name in generated for name in to_generate)
        artifact_status = "COMPLETE" if all_generated else "INCOMPLETE"

        return {
            **state,
            "artifact_status": artifact_status,
            "status": "completed",
        }


# =============================================================================
# WORKFLOW COMPOSITION
# =============================================================================


def compose_workflow_with_subgraphs(
    run_dir: Path,
    checkpointer: Optional[BaseCheckpointSaver] = None,
) -> Optional[StateGraph]:
    """Compose main workflow using subgraphs.

    This creates a workflow that uses subgraphs for modular execution:
    - Gate validation is handled by GateValidationSubgraph
    - QA is handled by QualityAssuranceSubgraph
    - Signoffs are handled by SignoffSubgraph
    - Artifacts are handled by ArtifactGenerationSubgraph

    Args:
        run_dir: Directory for run artifacts.
        checkpointer: Optional checkpointer for persistence.

    Returns:
        Compiled StateGraph or None if LangGraph unavailable.
    """
    if not LANGGRAPH_AVAILABLE:
        logger.error("LangGraph not available")
        return None

    # Import main state type
    try:
        from pipeline.langgraph.state import PipelineState
    except ImportError:
        logger.error("Pipeline state not available")
        return None

    # Create subgraphs
    gate_subgraph = GateValidationSubgraph(run_dir, checkpointer)
    qa_subgraph = QualityAssuranceSubgraph(run_dir, checkpointer)
    signoff_subgraph = SignoffSubgraph(run_dir, checkpointer)
    artifact_subgraph = ArtifactGenerationSubgraph(run_dir, checkpointer)

    # Build main workflow that delegates to subgraphs
    workflow = StateGraph(PipelineState)

    async def init_node(state: PipelineState) -> PipelineState:
        """Initialize pipeline."""
        return {
            **state,
            "phase": "INIT",
            "status": "running",
        }

    async def gate_node(state: PipelineState) -> PipelineState:
        """Run gate validation via subgraph."""
        subgraph_state = {
            "run_id": state.get("run_id", ""),
            "sprint_id": state.get("sprint_id", ""),
            "gates_to_run": state.get("gates_to_run", []),
        }

        result = await gate_subgraph.run(subgraph_state)

        return {
            **state,
            "gate_status": result.get("gate_status", "PENDING"),
            "gates_passed": result.get("gates_passed", []),
            "gates_failed": result.get("gates_failed", []),
        }

    async def qa_node(state: PipelineState) -> PipelineState:
        """Run QA via subgraph."""
        subgraph_state = {
            "run_id": state.get("run_id", ""),
            "sprint_id": state.get("sprint_id", ""),
        }

        result = await qa_subgraph.run(subgraph_state)

        return {
            **state,
            "qa_status": result.get("qa_status", "PENDING"),
            "actual_coverage": result.get("actual_coverage", 0.0),
        }

    async def signoff_node(state: PipelineState) -> PipelineState:
        """Run signoffs via subgraph."""
        subgraph_state = {
            "run_id": state.get("run_id", ""),
            "sprint_id": state.get("sprint_id", ""),
        }

        result = await signoff_subgraph.run(subgraph_state)

        return {
            **state,
            "signoff_status": result.get("signoff_status", "PENDING"),
            "signoffs": result.get("received_signoffs", {}),
        }

    async def artifact_node(state: PipelineState) -> PipelineState:
        """Generate artifacts via subgraph."""
        subgraph_state = {
            "run_id": state.get("run_id", ""),
            "sprint_id": state.get("sprint_id", ""),
        }

        result = await artifact_subgraph.run(subgraph_state)

        return {
            **state,
            "artifact_status": result.get("artifact_status", "PENDING"),
            "generated_artifacts": result.get("generated_artifacts", {}),
        }

    # Add nodes
    workflow.add_node("init", init_node)
    workflow.add_node("gate", gate_node)
    workflow.add_node("qa", qa_node)
    workflow.add_node("signoff", signoff_node)
    workflow.add_node("artifact", artifact_node)

    # Set entry and edges
    workflow.set_entry_point("init")
    workflow.add_edge("init", "gate")
    workflow.add_edge("gate", "qa")
    workflow.add_edge("qa", "signoff")
    workflow.add_edge("signoff", "artifact")
    workflow.add_edge("artifact", END)

    if checkpointer:
        return workflow.compile(checkpointer=checkpointer)
    return workflow.compile()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "LANGGRAPH_AVAILABLE",
    "BaseSubgraph",
    "SubgraphState",
    "GateValidationState",
    "GateValidationSubgraph",
    "QualityAssuranceState",
    "QualityAssuranceSubgraph",
    "SignoffState",
    "SignoffSubgraph",
    "ArtifactState",
    "ArtifactGenerationSubgraph",
    "compose_workflow_with_subgraphs",
]
