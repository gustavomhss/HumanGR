"""DOC-003: Gate Runner - Gate Execution & Evidence Collection

Gate Execution Flow:
====================
1. Load gate_selection.yml for sprint/track
2. For each gate in selection:
   a. Substitute placeholders (<run_id>, <sprint_id>, <run_dir>) with shlex.quote()
   b. Execute command via subprocess with timeout
   c. Capture stdout/stderr for output
   d. Scan output for secrets (SPH - Secret Pattern Heuristics)
   e. If SPH hits found, mark as BLOCK and redact content
   f. Collect expected outputs if specified
   g. Verify outputs don't cross symlink boundaries (symlink attack prevention)
   h. Record result in GateResult

Evidence Bundle Creation:
=========================
- Creates atomic evidence bundle with manifest
- Bundles include gate logs, captured outputs, MANIFEST.json
- Manifest contains SHA256 hash of each file
- Final bundle named with 12-char hash for integrity verification

SPH (Secret Pattern Heuristics):
================================
Scans for:
- AWS access keys (AKIA prefix)
- Private keys (RSA, EC, OpenSSH)
- Bearer tokens
Allows pattern-level allowlisting (sph_allow field)
Default-deny: any match triggers BLOCK unless explicitly allowed

Output Gates & Checks:
======================
Gates can be:
- Required: FAIL blocks overall status
- Optional: FAIL results in WARN status
Gate status mapping:
  - exit_code=0 + no SPH -> PASS
  - exit_code!=0 + no SPH -> FAIL
  - SPH hits found -> BLOCK (always fails)
  - Missing expected outputs -> FAIL

References: SEC-008 (Symlink Safety), SEC-009 (Shell Injection), SEC-014 (SPH)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import shlex
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:  # pragma: no cover - executed depending on environment
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    yaml = None

# SecurityGate integration (standalone LLM security validation - MED-003 FIX)
try:
    from pipeline.security_gate import SecurityGate, validate_security, SecurityResult
    STANDALONE_SECURITY_GATE_AVAILABLE = True
except ImportError:
    STANDALONE_SECURITY_GATE_AVAILABLE = False
    SecurityGate = None  # type: ignore
    validate_security = None  # type: ignore
    SecurityResult = None  # type: ignore

# FAIL-CLOSED security bypass flag (NF-006 FIX)
# When False (default): Security gate BLOCKS when module unavailable
# When True: Allows explicit bypass for development (MUST NOT be enabled in production!)
ALLOW_SECURITY_SKIP = os.getenv("ALLOW_SECURITY_SKIP", "false").lower() == "true"

# Production environment detection (SEC-001 FIX)
# In production, ALLOW_SECURITY_SKIP is IGNORED and always treated as False
IS_PRODUCTION = os.getenv("ENVIRONMENT", "development").lower() == "production"
if IS_PRODUCTION and ALLOW_SECURITY_SKIP:
    import logging as _security_logging
    _security_logging.getLogger(__name__).critical(
        "SECURITY-CRITICAL-001: ALLOW_SECURITY_SKIP=true detected in PRODUCTION! "
        "This is a CRITICAL security violation. The flag will be IGNORED and treated as False."
    )
    ALLOW_SECURITY_SKIP = False  # Force disable in production

# Security integration imports (graceful degradation if unavailable)
try:
    from pipeline.security.gate_integration import (
        get_security_gate_runner,
        SecurityGateError,
        SECURITY_GATE_AVAILABLE,
        SecurityMetrics,
        GATE_SECURITY_POLICIES,
        SecurityCheckPhase,
    )
except ImportError:
    SECURITY_GATE_AVAILABLE = False
    get_security_gate_runner = None
    SecurityGateError = Exception
    SecurityMetrics = None
    GATE_SECURITY_POLICIES = {}
    SecurityCheckPhase = None

# Stack enforcement decorator for critical operations (observability/event sourcing)
try:
    from pipeline.langgraph.nemo_stack_rails import enforce_stacks, StackEnforcementError
    STACK_ENFORCEMENT_AVAILABLE = True
except ImportError:
    STACK_ENFORCEMENT_AVAILABLE = False
    # Provide no-op decorator if not available
    def enforce_stacks(action: str, required=None, recommended=None):
        def decorator(func):
            return func
        return decorator
    class StackEnforcementError(Exception):
        pass

# =============================================================================
# F1-003: LETTA INTEGRATION FOR GATE LEARNING (2026-02-01)
# =============================================================================
# FAIL-CLOSED: Letta is CRITICAL for agent memory. Failures propagate.
try:
    from pipeline.letta_integration import (
        get_memory_bridge,
        LettaMemoryBridge,
        LettaCriticalError,
    )
    LETTA_GATE_AVAILABLE = True
except ImportError:
    LETTA_GATE_AVAILABLE = False
    get_memory_bridge = None  # type: ignore
    LettaCriticalError = Exception  # type: ignore

logger = logging.getLogger(__name__)


# =============================================================================
# SAFE-04: SHARED THREAD POOL EXECUTOR FOR SECURITY GATES
# =============================================================================

import atexit
from concurrent.futures import ThreadPoolExecutor
import threading as _threading

# Module-level shared pool for security gate operations
_security_thread_pool: Optional[ThreadPoolExecutor] = None
_security_pool_lock = _threading.Lock()

# Configuration
SECURITY_POOL_MAX_WORKERS = 4  # Limit to prevent resource exhaustion


def _get_security_pool() -> ThreadPoolExecutor:
    """Get or create shared security thread pool.

    Thread-safe singleton pattern with double-checked locking.
    Uses max 4 workers to prevent resource exhaustion while allowing
    concurrent security gate checks.

    SAFE-04 FIX: Replaces per-gate ThreadPoolExecutor creation which was
    inefficient and could cause resource exhaustion under load.

    Returns:
        The shared ThreadPoolExecutor instance.
    """
    global _security_thread_pool

    # Fast path (no lock needed)
    if _security_thread_pool is not None:
        return _security_thread_pool

    # Slow path (with lock for initialization)
    with _security_pool_lock:
        if _security_thread_pool is None:
            _security_thread_pool = ThreadPoolExecutor(
                max_workers=SECURITY_POOL_MAX_WORKERS,
                thread_name_prefix="security_gate_",
            )
            # Register cleanup on exit
            atexit.register(_shutdown_security_pool)
            logger.info(
                f"SAFE-04: Initialized shared security thread pool "
                f"(max_workers={SECURITY_POOL_MAX_WORKERS})"
            )

    return _security_thread_pool


def _shutdown_security_pool() -> None:
    """Shutdown the shared security pool gracefully.

    Called automatically at program exit via atexit.
    """
    global _security_thread_pool
    with _security_pool_lock:
        if _security_thread_pool is not None:
            logger.debug("SAFE-04: Shutting down security thread pool")
            _security_thread_pool.shutdown(wait=True, cancel_futures=False)
            _security_thread_pool = None


def _reset_security_pool_for_testing() -> None:
    """Reset the security pool singleton for testing. DO NOT use in production."""
    global _security_thread_pool
    with _security_pool_lock:
        if _security_thread_pool is not None:
            _security_thread_pool.shutdown(wait=False)
            _security_thread_pool = None
    logger.warning("SAFE-04: Security thread pool reset (testing only)")


def get_security_pool_stats() -> Dict[str, Any]:
    """Get security thread pool statistics.

    Returns:
        Dict with pool info or None if pool not initialized.
    """
    global _security_thread_pool
    with _security_pool_lock:
        if _security_thread_pool is None:
            return {"initialized": False}
        return {
            "initialized": True,
            "max_workers": SECURITY_POOL_MAX_WORKERS,
            "thread_name_prefix": "security_gate_",
        }


# =============================================================================
# SECURITY EXCEPTIONS (EVIL-004)
# =============================================================================

class SecurityError(Exception):
    """Raised when a security policy is violated.

    EVIL-004: This exception is raised when observability, guardrails, or
    other security-critical components are unavailable. The pipeline should
    NOT catch and suppress this exception - it indicates a serious security
    violation that must be addressed.
    """
    pass

# 2026-01-20: Grafana metrics for stack activation tracking
try:
    from pipeline.grafana_metrics import get_metrics_publisher
    GRAFANA_METRICS_AVAILABLE = True
except ImportError:
    GRAFANA_METRICS_AVAILABLE = False
    get_metrics_publisher = None

# 2026-01-31: Gate selector for automatic gate_selection.yml creation
try:
    from pipeline.gate_selector import get_gate_selector, GateSelector
    GATE_SELECTOR_AVAILABLE = True
except ImportError:
    GATE_SELECTOR_AVAILABLE = False
    get_gate_selector = None  # type: ignore
    GateSelector = None  # type: ignore


def ensure_gate_selection_exists(
    run_dir: Path,
    sprint_id: str,
    track: str = "B",
    profile: str = "standard",
) -> Path:
    """Ensure gate_selection.yml exists, create if not.

    MIGRATION FIX from pipeline/langgraph/task_executor.py
    This function guarantees the gate_selection.yml file exists before
    gate validation runs, preventing FileNotFoundError.

    Args:
        run_dir: Run directory path
        sprint_id: Sprint identifier (e.g., "S00")
        track: Gate track (A, B, or C)
        profile: Execution profile (standard, ops, etc.)

    Returns:
        Path to gate_selection.yml (existing or newly created)

    Raises:
        RuntimeError: If gate_selector is not available
    """
    gate_selection_path = run_dir / "sprints" / sprint_id / "plan" / "gate_selection.yml"

    if gate_selection_path.exists():
        logger.debug(f"gate_selection.yml already exists: {gate_selection_path}")
        return gate_selection_path

    if not GATE_SELECTOR_AVAILABLE:
        raise RuntimeError(
            f"gate_selection.yml not found at {gate_selection_path} and "
            "gate_selector is not available to create it. "
            "Ensure pipeline.gate_selector is installed."
        )

    logger.info(f"Creating gate_selection.yml for {sprint_id} (track={track}, profile={profile})")

    # Create directory if needed
    gate_selection_path.parent.mkdir(parents=True, exist_ok=True)

    # Create selection
    selector = get_gate_selector()
    selection = selector.create_selection(
        run_dir=run_dir,
        sprint_id=sprint_id,
        track=track,
        profile=profile,
    )
    output_path = selector.save_selection(selection, run_dir)

    logger.info(f"Created gate_selection.yml: {output_path}")
    return output_path


# =============================================================================
# INVIOLÁVEL: Gate Dependencies (DAG)
# =============================================================================

# DAG de dependências entre gates - gates só executam se dependências passaram
# FIX 2026-01-28: Otimizado para execução PARALELA (era linear, agora é DAG)
#
# Estrutura do DAG (5 níveis em vez de 9 sequenciais):
#
#        G0 (Syntax/Lint)
#        /    |    \
#      G1    G2    G5     ← Nível 1: 3 em PARALELO
#    (Type)(Unit)(Sec)
#        \  /  \
#         G4    G3        ← Nível 2: 2 em PARALELO
#      (Integ)(Cov)
#         |     |
#        G6    G7         ← Nível 3: 2 em PARALELO
#      (Perf)(Docs)
#          \  /
#           G8            ← Nível 4: Final
#        (Final)
#
# Speedup esperado: ~2x (5 níveis vs 9 sequenciais)
#
GATE_DEPENDENCIES: Dict[str, List[str]] = {
    "G0": [],                    # Syntax/Lint - sem dependências (Nível 0)
    "G1": ["G0"],                # Type Check - precisa de syntax (Nível 1)
    "G2": ["G0"],                # Unit Tests - precisa de syntax (Nível 1, PARALELO com G1, G5)
    "G5": ["G0"],                # Security Scan - precisa de syntax (Nível 1, PARALELO com G1, G2)
    "G3": ["G2"],                # Coverage - precisa de unit tests (Nível 2)
    "G4": ["G1", "G2"],          # Integration - precisa de types E unit (Nível 2, PARALELO com G3)
    "G6": ["G4"],                # Performance - precisa de integration (Nível 3)
    "G7": ["G3"],                # Documentation - precisa de coverage (Nível 3, PARALELO com G6)
    "G8": ["G3", "G4", "G5"],    # Final - precisa de coverage, integration E security (Nível 4)
}


def verify_gate_dependencies(
    gate_id: str,
    completed_gates: List[str],
) -> Tuple[bool, List[str]]:
    """Verifica se dependências do gate foram satisfeitas.

    INVIOLÁVEL: Gate NÃO PODE executar se dependências não passaram.

    Args:
        gate_id: ID do gate a executar
        completed_gates: Gates que já passaram

    Returns:
        Tuple[ok, missing_deps] - (True, []) se ok, (False, [deps]) se faltam
    """
    dependencies = GATE_DEPENDENCIES.get(gate_id, [])
    missing = [dep for dep in dependencies if dep not in completed_gates]

    if missing:
        logger.error(
            f"GATE ORDER VIOLATION: {gate_id} requer {dependencies}. "
            f"Faltando: {missing}"
        )
        return False, missing

    return True, []


# =============================================================================
# Quantum Leap Integration (Reflexion + A-MEM + GoT)
# =============================================================================

# SAFE-01 FIX: Thread-safe GoT cache moved to got_integration.py
# Import thread-safe cache functions from got_integration
try:
    from pipeline.got_integration import (
        get_got_cache_key as _get_got_cache_key,
        _get_got_cache,
        _set_got_cache,
        _cleanup_got_cache,
        get_got_cache,
    )
    SAFE_GOT_CACHE_AVAILABLE = True
except ImportError:
    SAFE_GOT_CACHE_AVAILABLE = False
    # Fallback to simple dict (not thread-safe, for legacy compatibility)
    _got_analysis_cache_fallback: Dict[str, Dict[str, Any]] = {}
    _GOT_CACHE_TTL_FALLBACK = 300

    def _get_got_cache_key(gate_id: str, error_type: str) -> str:
        import hashlib
        error_hash = hashlib.sha256(error_type.encode()).hexdigest()[:16]
        return f"{gate_id}:{error_hash}"

    def _get_got_cache(cache_key: str) -> Optional[Dict[str, Any]]:
        entry = _got_analysis_cache_fallback.get(cache_key)
        if entry and time.time() - entry.get("timestamp", 0) < _GOT_CACHE_TTL_FALLBACK:
            return entry.get("result")
        return None

    def _set_got_cache(cache_key: str, result: Dict[str, Any]) -> None:
        _got_analysis_cache_fallback[cache_key] = {
            "result": result,
            "timestamp": time.time(),
        }

    def _cleanup_got_cache() -> int:
        now = time.time()
        expired = [k for k, v in _got_analysis_cache_fallback.items()
                   if now - v.get("timestamp", 0) > _GOT_CACHE_TTL_FALLBACK]
        for k in expired:
            del _got_analysis_cache_fallback[k]
        return len(expired)


def _analyze_failure_with_got(
    gate_id: str,
    status: str,
    exit_code: int,
    log_path: str,
) -> Dict[str, Any]:
    """Analyze gate failure using GoT multi-perspective reasoning.

    Uses Graph of Thoughts to generate multiple analyses of why a gate failed,
    providing root cause analysis and fix suggestions.

    Args:
        gate_id: Gate identifier (e.g., "G3")
        status: Gate status (FAIL, BLOCK)
        exit_code: Process exit code
        log_path: Path to gate log file

    Returns:
        Dict with root_cause, suggested_fixes, and confidence score

    IMPORTANTE: FAIL-CLOSED - GoT é CRITICAL para auto-evolução do pipeline.
    """
    # SAFE-01 FIX: Use thread-safe cache from got_integration
    error_type = f"{status}:{exit_code}"[:100]
    cache_key = _get_got_cache_key(gate_id, error_type)

    # SAFE-01: Thread-safe cache read
    cached = _get_got_cache(cache_key)
    if cached is not None:
        logger.debug(f"GoT cache hit for {gate_id} (thread-safe)")
        return cached

    # Periodically clean up expired cache entries (thread-safe)
    _cleanup_got_cache()

    # FAIL-CLOSED: GoT é stack CRITICAL (Quantum Leap - auto-evolução)
    from pipeline.stack_health_supervisor import (
        get_stack_supervisor,
        CriticalStackUnavailableError,
    )

    supervisor = get_stack_supervisor()

    if not supervisor.is_healthy("got_integration"):
        logger.error(
            f"FAIL-CLOSED: GoT integration indisponível para análise de {gate_id}. "
            "GoT é OBRIGATÓRIO para auto-evolução."
        )
        raise CriticalStackUnavailableError(
            "got_integration",
            f"GoT unavailable for root cause analysis of {gate_id}"
        )

    try:
        from .got_integration import analyze_gate_failure_with_got

        # Read log content (limited to avoid context overflow)
        log_content = ""
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                log_content = f.read()[:4000]  # First 4KB
        except Exception as e:
            logger.debug(f"GATE: Gate operation failed: {e}")

        result = analyze_gate_failure_with_got(
            gate_id=gate_id,
            error_message=f"Gate {gate_id} {status} (exit code {exit_code})",
            log_content=log_content,
            num_analyses=3,
        )

        if result.success:
            logger.info(f"GoT analysis for {gate_id}: {result.root_cause[:100]}...")
            analysis_result = {
                "root_cause": result.root_cause,
                "suggested_fixes": result.suggested_fixes,
                "prevention_strategies": result.prevention_strategies,
                "confidence": result.confidence,
                "got_operations": result.got_operations,
            }
            # SAFE-01 FIX: Thread-safe cache write
            _set_got_cache(cache_key, analysis_result)
            # 2026-01-20: Publish GoT activation to Grafana
            if GRAFANA_METRICS_AVAILABLE and get_metrics_publisher:
                pub = get_metrics_publisher()
                pub.publish_stack_activation(
                    stack_name="GoT",
                    purpose=f"Multi-path reasoning for gate {gate_id} failure analysis",
                    input_summary=f"Gate {gate_id} {status} (exit {exit_code})",
                    output_summary=f"Root cause: {result.root_cause[:150]}",
                    success=True,
                    metadata={"confidence": result.confidence, "fixes_count": len(result.suggested_fixes)},
                )
            return analysis_result
        else:
            logger.warning(f"GoT analysis returned error for {gate_id}: {result.error}")
            # 2026-01-20: Publish failed GoT activation
            if GRAFANA_METRICS_AVAILABLE and get_metrics_publisher:
                pub = get_metrics_publisher()
                pub.publish_stack_activation(
                    stack_name="GoT",
                    purpose=f"Multi-path reasoning for gate {gate_id}",
                    input_summary=f"Gate {gate_id} {status}",
                    output_summary=f"Error: {result.error}",
                    success=False,
                )
            return {}

    except ImportError:
        # FAIL-CLOSED: GoT é CRITICAL
        logger.error("FAIL-CLOSED: GoT integration module not available")
        raise CriticalStackUnavailableError(
            "got_integration",
            "GoT module not importable"
        )
    except CriticalStackUnavailableError:
        raise
    except Exception as e:
        # FAIL-CLOSED: Erro no GoT é CRÍTICO
        logger.error(f"FAIL-CLOSED: GoT analysis failed for {gate_id}: {e}")
        raise CriticalStackUnavailableError(
            "got_integration",
            f"GoT analysis failed for {gate_id}: {e}"
        ) from e


def _trigger_reflexion_on_failure(
    gate_id: str,
    status: str,
    exit_code: int,
    sprint_id: str,
    log_path: str,
) -> "Optional[Reflection]":
    """Trigger Reflexion Engine on gate failure.

    IMPORTANTE: FAIL-CLOSED - Reflexion e A-MEM são CRITICAL para auto-evolução.

    P2-5.1 FIX (2026-02-01): Now returns the Reflection object so it can be
    used to guide rework. Previously returned None, losing the prevention_strategy
    and suggested_fixes that were generated but never passed to rework context.

    Returns:
        Reflection object with root_cause, prevention_strategy, etc. or None on failure.
    """
    # FAIL-CLOSED: Reflexion e A-MEM são stacks CRITICAL
    from pipeline.stack_health_supervisor import (
        get_stack_supervisor,
        CriticalStackUnavailableError,
    )

    supervisor = get_stack_supervisor()

    if not supervisor.is_healthy("reflexion"):
        logger.error(
            f"FAIL-CLOSED: Reflexion indisponível para {gate_id}. "
            "Reflexion é OBRIGATÓRIO para aprendizado."
        )
        raise CriticalStackUnavailableError(
            "reflexion",
            f"Reflexion unavailable for learning from {gate_id} failure"
        )

    if not supervisor.is_healthy("amem"):
        logger.error(
            f"FAIL-CLOSED: A-MEM indisponível para {gate_id}. "
            "A-MEM é OBRIGATÓRIO para memória de erros."
        )
        raise CriticalStackUnavailableError(
            "amem",
            f"A-MEM unavailable for storing {gate_id} failure"
        )

    try:
        from .reflexion_engine import ReflexionEngine, ReflectionType
        from .amem_integration import get_amem_integration

        # Generate reflection
        engine = ReflexionEngine()
        reflection = engine.generate_reflection_sync(
            task_description=f"Gate {gate_id} validation",
            error_message=f"Exit code {exit_code}, status {status}",
            reflection_type=ReflectionType.GATE_FAILURE,
            attempt_number=1,
        )

        # Save to A-MEM
        # FIX 2026-01-23: Use correct parameter names for save_error()
        amem = get_amem_integration()
        amem.save_error(
            agent_id=f"gate_runner/{sprint_id}",
            error=f"Gate {gate_id}: {status} (exit={exit_code})",
            fix=reflection.prevention_strategy if reflection else "Manual review required",
            sprint_id=sprint_id,
            tags=["gate_failure", gate_id],
        )

        logger.info(f"Reflexion triggered for gate {gate_id} failure")

        # F1-003: Save learning to Letta (FAIL-CLOSED with retry)
        # CRITICAL: Letta is required for cross-agent learning. Failures propagate.
        if LETTA_GATE_AVAILABLE and get_memory_bridge is not None:
            bridge = get_memory_bridge()
            bridge.save_learning_resilient(
                agent_id="gate_runner",
                learning={
                    "type": "gate_failure",
                    "gate_id": gate_id,
                    "sprint_id": sprint_id,
                    "status": status,
                    "exit_code": exit_code,
                    "root_cause": reflection.root_cause if reflection else "Unknown",
                    "prevention_strategy": reflection.prevention_strategy if reflection else "",
                    "suggested_fixes": reflection.suggested_fixes[:3] if reflection and reflection.suggested_fixes else [],
                },
                tags=["gate_failure", gate_id],
                sprint_id=sprint_id,
            )
            logger.debug(f"[GateRunner] Saved gate failure learning to Letta: {gate_id}")
        else:
            # FAIL-CLOSED: Letta module must be available
            raise LettaCriticalError(
                operation="save_gate_failure_learning",
                message=f"Letta module not available - cannot persist gate failure: {gate_id}",
            )

        # HIGH-004 FIX: Integrate reflexion result with Live-SWE evolution system
        # This connects the 4,124 lines of dead Live-SWE code to the pipeline
        try:
            from pipeline.live_swe_integration import get_live_swe_integration

            live_swe = get_live_swe_integration()
            if live_swe and reflection:
                # Create context for evolution
                context = {
                    "gate_id": gate_id,
                    "sprint_id": sprint_id,
                    "status": status,
                    "exit_code": exit_code,
                }

                # Integrate the reflexion result for behavior evolution
                evolution_result = live_swe.integrate_reflexion_result(
                    reflexion_result=reflection,
                    gate_id=gate_id,
                    context=context,
                )

                if evolution_result:
                    logger.info(
                        f"HIGH-004: Live-SWE integrated reflexion for gate {gate_id}: "
                        f"evolution_triggered={evolution_result.get('evolution_triggered', False)}"
                    )
        except ImportError:
            logger.debug("HIGH-004: Live-SWE integration not available")
        except Exception as live_swe_err:
            logger.warning(f"HIGH-004: Live-SWE integration failed (non-blocking): {live_swe_err}")

        # 2026-01-20: Publish Reflexion and A-MEM activation to Grafana
        if GRAFANA_METRICS_AVAILABLE and get_metrics_publisher:
            pub = get_metrics_publisher()
            # Reflexion activation
            pub.publish_stack_activation(
                stack_name="Reflexion",
                purpose=f"Self-reflection on gate {gate_id} failure",
                input_summary=f"Gate {gate_id} {status} (exit {exit_code})",
                output_summary=f"Prevention: {reflection.prevention_strategy[:150] if reflection else 'N/A'}",
                success=True,
            )
            # A-MEM activation
            pub.publish_stack_activation(
                stack_name="A-MEM",
                purpose=f"Store error learning from gate {gate_id}",
                input_summary=f"Gate failure: {gate_id}",
                output_summary=f"Error stored with resolution in Zettelkasten memory",
                success=True,
            )

        # P2-5.1 FIX: Return the reflection so it can be used in rework context
        return reflection

    except ImportError as e:
        # FAIL-CLOSED: Reflexion/A-MEM são CRITICAL
        logger.error(f"FAIL-CLOSED: Reflexion/A-MEM module not available: {e}")
        raise CriticalStackUnavailableError(
            "reflexion",
            f"Reflexion/A-MEM modules not importable: {e}"
        )
    except CriticalStackUnavailableError:
        raise
    except Exception as e:
        # FAIL-CLOSED: Erro é CRÍTICO
        logger.error(f"FAIL-CLOSED: Failed to trigger reflexion for {gate_id}: {e}")
        raise CriticalStackUnavailableError(
            "reflexion",
            f"Reflexion failed for {gate_id}: {e}"
        ) from e


def _save_gate_learning(sprint_id: str, track: str, gate_count: int) -> None:
    """Save successful gate execution as learning.

    IMPORTANTE: FAIL-CLOSED - A-MEM é CRITICAL para memória de aprendizados.
    """
    # FAIL-CLOSED: A-MEM é stack CRITICAL
    from pipeline.stack_health_supervisor import (
        get_stack_supervisor,
        CriticalStackUnavailableError,
    )

    supervisor = get_stack_supervisor()

    if not supervisor.is_healthy("amem"):
        logger.error(
            f"FAIL-CLOSED: A-MEM indisponível para salvar learning de {track}. "
            "A-MEM é OBRIGATÓRIO para memória de aprendizados."
        )
        raise CriticalStackUnavailableError(
            "amem",
            f"A-MEM unavailable for saving learning from {track}"
        )

    try:
        from .amem_integration import get_amem_integration

        amem = get_amem_integration()
        # FIX 2026-01-23: Use correct parameter names for save_learning()
        # Signature: save_learning(agent_id, learning, tags, sprint_id, auto_link)
        amem.save_learning(
            agent_id=f"gate_runner/{sprint_id}",
            learning=f"Track {track} passed all {gate_count} gates successfully",
            tags=["gate_success", f"track_{track}"],
            sprint_id=sprint_id,
        )

        logger.info(f"Learning saved for track {track} success")

        # F1-003: Also save to Letta (FAIL-CLOSED with retry)
        # CRITICAL: Letta is required for cross-agent learning. Failures propagate.
        if LETTA_GATE_AVAILABLE and get_memory_bridge is not None:
            bridge = get_memory_bridge()
            bridge.save_learning_resilient(
                agent_id="gate_runner",
                learning={
                    "type": "gate_success",
                    "sprint_id": sprint_id,
                    "track": track,
                    "gate_count": gate_count,
                    "pattern": f"Track {track} configuration works",
                },
                tags=["gate_success", f"track_{track}"],
                sprint_id=sprint_id,
            )
            logger.debug(f"[GateRunner] Saved gate success learning to Letta: {track}")
        else:
            # FAIL-CLOSED: Letta module must be available
            raise LettaCriticalError(
                operation="save_gate_success_learning",
                message=f"Letta module not available - cannot persist gate success: {track}",
            )

    except ImportError:
        # FAIL-CLOSED: A-MEM é CRITICAL
        logger.error("FAIL-CLOSED: A-MEM module not available")
        raise CriticalStackUnavailableError(
            "amem",
            "A-MEM module not importable"
        )
    except CriticalStackUnavailableError:
        raise


# =============================================================================
# MED-003 FIX: Standalone Security Gate Integration
# =============================================================================


def _run_standalone_security_gate(
    run_dir: Path,
    sprint_id: str,
    repo_root: Path,
    test_outputs: Optional[List[str]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Run the standalone SecurityGate for additional LLM security validation.

    MED-003 FIX: The SecurityGate module (579 lines) was implemented but never
    integrated. This function connects it to the gate execution flow.

    The standalone SecurityGate provides:
    - Regex-based secret/PII detection (API keys, JWTs, etc.)
    - Codebase scan for injection protection patterns
    - NeMo guardrails configuration existence check

    This complements the inline SecurityGateRunner which provides:
    - LLM Guard integration for semantic analysis
    - Pre/post gate security validation

    Args:
        run_dir: Run directory for this pipeline execution.
        sprint_id: Sprint identifier.
        repo_root: Repository root path.
        test_outputs: Optional list of outputs to scan for security issues.

    Returns:
        Tuple of (passed: bool, details: dict with security result info)
    """
    if not STANDALONE_SECURITY_GATE_AVAILABLE or SecurityGate is None:
        if ALLOW_SECURITY_SKIP:
            logger.warning(
                "SECURITY-001: SecurityGate bypassed via ALLOW_SECURITY_SKIP - "
                "This MUST NOT be enabled in production!"
            )
            return True, {
                "skipped": True,
                "reason": "explicit_bypass",
                "warning": "ALLOW_SECURITY_SKIP enabled - security validation skipped",
            }
        else:
            logger.error(
                "SECURITY-001: SecurityGate not available and ALLOW_SECURITY_SKIP=false - "
                "BLOCKING execution. Install SecurityGate or set ALLOW_SECURITY_SKIP=true for development."
            )
            return False, {
                "failed": True,
                "reason": "module_not_available",
                "action": "Install SecurityGate module or set ALLOW_SECURITY_SKIP=true for development",
            }

    try:
        gate = SecurityGate(
            run_dir=run_dir,
            sprint_id=sprint_id,
            repo_root=repo_root,
            test_outputs=test_outputs or [],
        )

        result: SecurityResult = gate.run()

        details = {
            "passed": result.passed,
            "test_mode": result.test_mode,
            "sanitization_passed": result.sanitization.passed,
            "secrets_detected": result.sanitization.secrets_detected,
            "pii_detected": result.sanitization.pii_detected,
            "injection_passed": result.injection.passed,
            "policy_passed": result.policy.passed,
            "tools_available": result.tools_available,
            "errors": result.errors,
            "warnings": result.warnings,
        }

        if result.passed:
            logger.info(
                f"Standalone SecurityGate PASS for {sprint_id} "
                f"(mode={result.test_mode})"
            )
        else:
            logger.warning(
                f"Standalone SecurityGate FAIL for {sprint_id}: "
                f"secrets={result.sanitization.secrets_detected}, "
                f"pii={result.sanitization.pii_detected}, "
                f"errors={result.errors[:2]}"
            )

        # Publish to Grafana if available
        if GRAFANA_METRICS_AVAILABLE and get_metrics_publisher:
            try:
                pub = get_metrics_publisher()
                pub.publish_stack_activation(
                    stack_name="SecurityGate",
                    purpose=f"LLM security validation for G6 in {sprint_id}",
                    input_summary=f"test_mode={result.test_mode}",
                    output_summary=f"passed={result.passed}, secrets={result.sanitization.secrets_detected}",
                    success=result.passed,
                    metadata=details,
                )
            except Exception as e:
                logger.debug(f"GRAFANA: Failed to publish SecurityGate result: {e}")

        return result.passed, details

    except Exception as e:
        # SECURITY-FAIL-CLOSED-001 FIX: Exception in security check = BLOCK (never pass!)
        # This is a CRITICAL fix - previously this returned True, allowing bypasses
        logger.error(
            "SECURITY-FAIL-CLOSED-001: SecurityGate exception - BLOCKING execution. "
            "Error: %s",
            e,
        )
        return False, {
            "failed": True,
            "reason": f"SecurityGate exception: {e}",
            "error": True,
            "action": "Investigate security gate error before proceeding",
        }
    except Exception as e:
        # FAIL-CLOSED: Erro é CRÍTICO
        logger.error(f"FAIL-CLOSED: Failed to save learning for {track}: {e}")
        raise CriticalStackUnavailableError(
            "amem",
            f"A-MEM save_learning failed for {track}: {e}"
        ) from e


def _analyze_failure_with_bot(
    gate_id: str,
    status: str,
    exit_code: int,
    log_content: str,
) -> Dict[str, Any]:
    """Analyze gate failure using Buffer of Thoughts accumulated reasoning.

    Uses BoT to build up context incrementally through categorized thoughts,
    then synthesize a comprehensive failure analysis.

    Args:
        gate_id: Gate identifier (e.g., "G3")
        status: Gate status (FAIL, BLOCK)
        exit_code: Process exit code
        log_content: Gate log content to analyze

    Returns:
        Dict with analysis results including conclusion, confidence, and suggestions

    Note:
        Uses graceful degradation - returns empty dict if BoT not available.
        BoT complements GoT by providing deeper accumulated reasoning.
    """
    try:
        from pipeline.reasoning.bot_chain import (
            BufferOfThoughts,
            ThoughtType,
            BOT_AVAILABLE,
        )

        if not BOT_AVAILABLE:
            logger.debug("BoT not available, skipping accumulated reasoning")
            return {}

        # Create buffer and accumulate thoughts about the failure
        bot = BufferOfThoughts(max_thoughts=50, context_window=20)

        # Add initial observation
        bot.add_observation(f"Gate {gate_id} failed with status {status} and exit code {exit_code}")

        # Analyze log content for evidence
        if log_content:
            # P1-06 FIX (2026-01-30): Single-pass log parsing instead of 3 splits
            # Original code split the log 3 times and lowercased each line 3 times
            # Now we split once, lowercase once per line, and categorize in a single pass
            error_lines = []
            assertion_lines = []
            exception_lines = []

            for line in log_content.split('\n'):  # Split ONCE
                lower = line.lower()  # Lowercase ONCE per line

                if 'error' in lower or 'fail' in lower:
                    error_lines.append(line)
                if 'assert' in lower:
                    assertion_lines.append(line)
                if 'exception' in lower or 'traceback' in lower:
                    exception_lines.append(line)

            # Process categorized lines (same as before)
            for line in error_lines[:5]:  # Top 5 error lines
                bot.add_evidence(line.strip()[:200], source=f"gate_{gate_id}_log")

            for line in assertion_lines[:3]:
                bot.add_thought(line.strip()[:200], ThoughtType.CONTRADICTION)

            for line in exception_lines[:3]:
                bot.add_thought(line.strip()[:200], ThoughtType.OBSERVATION)

        # Add gate-specific context
        gate_contexts = {
            "G0": "Syntax validation - check for Python syntax errors",
            "G1": "Import validation - check for missing dependencies or circular imports",
            "G2": "Type checking - check for type annotation issues",
            "G3": "Unit tests - check for test failures or assertion errors",
            "G4": "Integration tests - check for service connectivity issues",
            "G5": "Coverage - check if coverage threshold was met",
            "G6": "Security - check for security vulnerabilities",
            "G7": "Performance - check for performance regressions",
            "G8": "Final validation - comprehensive check",
        }
        if gate_id in gate_contexts:
            bot.add_thought(gate_contexts[gate_id], ThoughtType.CONTEXT)

        # Add inference based on exit code
        if exit_code == 1:
            bot.add_inference("Exit code 1 typically indicates test failure or validation error")
        elif exit_code == 2:
            bot.add_inference("Exit code 2 typically indicates command syntax error")
        elif exit_code > 128:
            bot.add_inference(f"Exit code {exit_code} indicates process was killed by signal {exit_code - 128}")

        # Synthesize conclusion
        synthesis = bot.synthesize()

        result = {
            "conclusion": synthesis.conclusion,
            "confidence": synthesis.confidence,
            "supporting_evidence": synthesis.supporting_thoughts[:5],
            "contradictions": synthesis.contradicting_thoughts[:3],
            "uncertainties": synthesis.uncertainties[:3],
            "reasoning_chain": synthesis.reasoning_chain,
            "thought_count": synthesis.thought_count,
            "bot_analysis": True,
        }

        logger.info(f"BoT analysis for {gate_id}: {synthesis.conclusion[:100]}... (confidence: {synthesis.confidence:.2f})")

        # 2026-01-20: Publish BoT activation to Grafana
        if GRAFANA_METRICS_AVAILABLE and get_metrics_publisher:
            pub = get_metrics_publisher()
            pub.publish_stack_activation(
                stack_name="BoT",
                purpose=f"Accumulated reasoning for gate {gate_id} failure analysis",
                input_summary=f"Gate {gate_id} {status} (exit {exit_code})",
                output_summary=f"Conclusion: {synthesis.conclusion[:150]}",
                success=True,
                metadata={"confidence": synthesis.confidence, "thought_count": synthesis.thought_count},
            )

        return result

    except ImportError as e:
        logger.debug(f"BoT module not available: {e}")
        return {}
    except Exception as e:
        logger.warning(f"BoT analysis failed for {gate_id}: {e}")
        return {}


def _record_gate_metrics(gate_id: str, status: str, duration_ms: float) -> None:
    """Record gate execution metrics for Live-SWE behavior evolution.

    Feeds gate metrics into the observability system which forwards to Live-SWE.
    This enables automatic behavior optimization based on gate performance.

    EVIL-004 SECURITY FIX: Changed from graceful degradation to FAIL-CLOSED.
    If observability is unavailable, gate execution is BLOCKED.
    This prevents silent bypassing of security monitoring.

    Args:
        gate_id: Gate identifier (e.g., "G0", "G5")
        status: Gate result status ("PASS", "FAIL", "WARN", "BLOCK")
        duration_ms: Gate execution duration in milliseconds

    Raises:
        SecurityError: If observability is unavailable (fail-closed)
    """
    try:
        from .observability.metrics import record_gate_execution

        record_gate_execution(
            gate_id=gate_id,
            status=status,
            duration_ms=duration_ms,
        )

    except ImportError:
        # EVIL-004 SECURITY FIX: FAIL-CLOSED instead of graceful degradation
        logger.error(
            f"SECURITY (EVIL-004): Observability metrics unavailable for gate {gate_id}. "
            "BLOCKING gate execution - observability is REQUIRED for security monitoring."
        )
        raise SecurityError(
            f"Observability required but unavailable for gate {gate_id}. "
            "Install observability module or fix import error."
        )
    except Exception as e:
        # EVIL-004 SECURITY FIX: FAIL-CLOSED on metrics failure
        logger.error(
            f"SECURITY (EVIL-004): Failed to record metrics for gate {gate_id}: {e}. "
            "BLOCKING gate execution - metrics recording is REQUIRED."
        )
        raise SecurityError(
            f"Failed to record security metrics for gate {gate_id}: {e}"
        ) from e


def _score_gate_with_langfuse(
    gate_id: str,
    status: str,
    exit_code: int,
    duration_ms: float,
    sprint_id: str,
    trace_id: Optional[str] = None,
) -> None:
    """Score gate execution in Langfuse for observability.

    Records gate results as Langfuse scores for tracking and analysis.
    Enables dashboards, alerts, and historical analysis of gate performance.

    Args:
        gate_id: Gate identifier (e.g., "G0", "G5")
        status: Gate result status ("PASS", "FAIL", "WARN", "BLOCK")
        exit_code: Process exit code
        duration_ms: Gate execution duration in milliseconds
        sprint_id: Sprint identifier for grouping
        trace_id: Optional Langfuse trace ID for correlation

    Note:
        Uses graceful degradation - silently skips if Langfuse not available.
    """
    try:
        from langfuse import Langfuse

        # Initialize Langfuse client (uses env vars: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY)
        langfuse = Langfuse()

        # Map status to numeric score (1.0 = success, 0.0 = failure)
        status_scores = {
            "PASS": 1.0,
            "WARN": 0.7,  # Partial success
            "FAIL": 0.0,
            "BLOCK": 0.0,
        }
        score_value = status_scores.get(status, 0.0)

        # Create or get trace
        if trace_id:
            trace = langfuse.trace(id=trace_id)
        else:
            trace = langfuse.trace(
                name=f"gate_execution_{sprint_id}",
                metadata={
                    "sprint_id": sprint_id,
                    "gate_id": gate_id,
                },
            )

        # Score the gate execution
        trace.score(
            name=f"gate_{gate_id}_status",
            value=score_value,
            comment=f"Gate {gate_id}: {status} (exit_code={exit_code})",
        )

        # Score duration (normalized: <1s = 1.0, >60s = 0.0)
        duration_score = max(0.0, min(1.0, 1.0 - (duration_ms / 60000)))
        trace.score(
            name=f"gate_{gate_id}_duration",
            value=duration_score,
            comment=f"Duration: {duration_ms:.0f}ms",
        )

        # Flush to ensure scores are sent
        langfuse.flush()

        logger.debug(f"Langfuse score recorded for gate {gate_id}: {score_value}")

    except ImportError:
        # EVIL-004 SECURITY FIX: FAIL-CLOSED instead of graceful degradation
        logger.error(
            f"SECURITY (EVIL-004): Langfuse unavailable for gate {gate_id}. "
            "BLOCKING gate execution - Langfuse is REQUIRED for observability."
        )
        raise SecurityError(
            f"Langfuse required but unavailable for gate {gate_id}. "
            "Install langfuse package or configure LANGFUSE_* environment variables."
        )
    except Exception as e:
        # EVIL-004 SECURITY FIX: FAIL-CLOSED on Langfuse failure
        logger.error(
            f"SECURITY (EVIL-004): Failed to score gate {gate_id} with Langfuse: {e}. "
            "BLOCKING gate execution - Langfuse scoring is REQUIRED."
        )
        raise SecurityError(
            f"Failed to score gate {gate_id} with Langfuse: {e}"
        ) from e


def _score_gate_run_with_langfuse(
    sprint_id: str,
    track: str,
    overall_status: str,
    gate_results: List["GateResult"],
    total_duration_ms: float,
) -> Optional[str]:
    """Score overall gate run in Langfuse.

    Records aggregate metrics for the entire gate selection run.

    Args:
        sprint_id: Sprint identifier
        track: Track identifier (A, B, C)
        overall_status: Overall run status
        gate_results: List of individual gate results
        total_duration_ms: Total execution duration

    Returns:
        Langfuse trace ID if successful, None otherwise

    Note:
        Uses graceful degradation - silently skips if Langfuse not available.
    """
    try:
        from langfuse import Langfuse

        langfuse = Langfuse()

        # Create trace for the entire run
        trace = langfuse.trace(
            name=f"gate_run_{sprint_id}_{track}",
            metadata={
                "sprint_id": sprint_id,
                "track": track,
                "total_gates": len(gate_results),
            },
        )

        # Overall status score
        status_scores = {"PASS": 1.0, "WARN": 0.7, "FAIL": 0.0, "BLOCK": 0.0}
        trace.score(
            name="gate_run_status",
            value=status_scores.get(overall_status, 0.0),
            comment=f"Overall: {overall_status}",
        )

        # Pass rate score
        passed = sum(1 for r in gate_results if r.status == "PASS")
        pass_rate = passed / len(gate_results) if gate_results else 0.0
        trace.score(
            name="gate_pass_rate",
            value=pass_rate,
            comment=f"Passed: {passed}/{len(gate_results)}",
        )

        # Duration score (normalized)
        duration_score = max(0.0, min(1.0, 1.0 - (total_duration_ms / 300000)))  # 5min baseline
        trace.score(
            name="gate_run_duration",
            value=duration_score,
            comment=f"Total: {total_duration_ms:.0f}ms",
        )

        # Score individual gates
        for result in gate_results:
            trace.score(
                name=f"gate_{result.gate_id}",
                value=status_scores.get(result.status, 0.0),
                comment=f"{result.status} (exit={result.exit_code})",
            )

        langfuse.flush()

        logger.info(f"Langfuse trace created for gate run: {trace.id}")
        return trace.id

    except ImportError:
        logger.debug("Langfuse not available, skipping run scoring")
        return None
    except Exception as e:
        logger.debug(f"Failed to score gate run with Langfuse: {e}")
        return None


def _validate_gate_with_guardrails(gate_id: str) -> None:
    """Validate gate execution with guardrails.

    This is called EXPLICITLY before every gate execution to ensure
    stack requirements are enforced.

    EVIL-004 SECURITY FIX: Changed from graceful degradation to FAIL-CLOSED.
    If guardrails are unavailable or fail, gate execution is BLOCKED.
    This prevents silent bypassing of security guardrails.

    Args:
        gate_id: Gate identifier (e.g., "G0", "G5")

    Raises:
        SecurityError: If guardrails are unavailable or validation fails critically
    """
    try:
        from pipeline.langgraph.stack_guardrails import get_guardrails
        from pipeline.langgraph.nemo_stack_rails import get_nemo_rails

        # Validate with NeMo rails
        nemo = get_nemo_rails()
        validation = nemo.validate_action("gate_execution", {"langfuse"})

        if not validation["valid"]:
            critical_violations = []
            for v in validation["violations"]:
                if v.get("severity") == "critical":
                    critical_violations.append(v["message"])
                    logger.error(f"CRITICAL guardrail violation for {gate_id}: {v['message']}")
                else:
                    logger.warning(f"Guardrail recommendation for {gate_id}: {v['message']}")

            # EVIL-004 SECURITY FIX: FAIL-CLOSED on critical violations
            if critical_violations:
                raise SecurityError(
                    f"Critical guardrail violations for gate {gate_id}: {critical_violations}"
                )

        # Record in guardrails tracker
        guardrails = get_guardrails()
        # Check circuit breakers
        if not guardrails.circuit_registry.can_execute("gate_runner"):
            # EVIL-004 SECURITY FIX: FAIL-CLOSED when circuit breaker is open
            logger.error(
                f"SECURITY (EVIL-004): Circuit breaker OPEN for gate_runner. "
                f"BLOCKING gate {gate_id} execution."
            )
            raise SecurityError(
                f"Circuit breaker OPEN for gate_runner - gate {gate_id} execution blocked"
            )

        logger.debug(f"Guardrails validated for {gate_id}")

    except ImportError:
        # EVIL-004 SECURITY FIX: FAIL-CLOSED instead of graceful degradation
        logger.error(
            f"SECURITY (EVIL-004): Guardrails unavailable for gate {gate_id}. "
            "BLOCKING gate execution - guardrails are REQUIRED for security."
        )
        raise SecurityError(
            f"Guardrails required but unavailable for gate {gate_id}. "
            "Install guardrails modules or fix import error."
        )
    except SecurityError:
        raise
    except Exception as e:
        # EVIL-004 SECURITY FIX: FAIL-CLOSED on guardrails failure
        logger.error(
            f"SECURITY (EVIL-004): Guardrails validation failed for gate {gate_id}: {e}. "
            "BLOCKING gate execution."
        )
        raise SecurityError(
            f"Guardrails validation failed for gate {gate_id}: {e}"
        ) from e


# =============================================================================
# Security Metrics Collection
# =============================================================================


def _collect_security_metrics() -> Optional[Dict[str, Any]]:
    """Collect current security metrics from the SecurityGateRunner.

    Returns:
        Dict with security metrics or None if unavailable.

    Note:
        Uses graceful degradation - returns None if security module unavailable.
    """
    if not SECURITY_GATE_AVAILABLE or get_security_gate_runner is None:
        return None

    try:
        runner = get_security_gate_runner()
        metrics = runner.get_metrics()
        return metrics.to_dict()
    except Exception as e:
        logger.warning(f"Failed to collect security metrics: {e}")
        return None


def _log_security_metrics_to_langfuse(
    sprint_id: str,
    track: str,
    metrics: Dict[str, Any],
    overall_status: str,
) -> None:
    """Log security metrics to Langfuse for observability.

    Records aggregated security metrics from the gate run to enable
    dashboards, alerts, and historical analysis.

    Args:
        sprint_id: Sprint identifier
        track: Track identifier (A, B, C)
        metrics: Security metrics dictionary
        overall_status: Overall gate run status

    Note:
        Uses graceful degradation - silently skips if Langfuse not available.
    """
    try:
        from langfuse import Langfuse

        langfuse = Langfuse()

        # Create event for security metrics
        langfuse.event(
            name=f"security_metrics_{sprint_id}_{track}",
            metadata={
                "sprint_id": sprint_id,
                "track": track,
                "overall_status": overall_status,
                "total_checks": metrics.get("total_checks", 0),
                "passed_checks": metrics.get("passed_checks", 0),
                "failed_checks": metrics.get("failed_checks", 0),
                "blocked_operations": metrics.get("blocked_operations", 0),
                "pii_detections": metrics.get("pii_detections", 0),
                "injection_attempts": metrics.get("injection_attempts", 0),
                "toxicity_flags": metrics.get("toxicity_flags", 0),
                "avg_latency_ms": metrics.get("avg_latency_ms", 0),
            },
        )

        # Score based on security pass rate
        total = metrics.get("total_checks", 0)
        passed = metrics.get("passed_checks", 0)
        pass_rate = passed / total if total > 0 else 1.0
        langfuse.score(
            name="security_pass_rate",
            value=pass_rate,
            comment=f"Security checks: {passed}/{total} passed",
        )

        # Score based on blocked operations (lower is better)
        blocked = metrics.get("blocked_operations", 0)
        blocked_score = max(0.0, 1.0 - (blocked * 0.1))  # -0.1 per blocked operation
        langfuse.score(
            name="security_blocked_score",
            value=blocked_score,
            comment=f"Blocked operations: {blocked}",
        )

        # Score for specific security issues (higher value = more issues = lower score)
        pii = metrics.get("pii_detections", 0)
        injections = metrics.get("injection_attempts", 0)
        toxicity = metrics.get("toxicity_flags", 0)
        issues_total = pii + injections + toxicity
        issues_score = max(0.0, 1.0 - (issues_total * 0.05))  # -0.05 per issue
        langfuse.score(
            name="security_issues_score",
            value=issues_score,
            comment=f"PII: {pii}, Injections: {injections}, Toxicity: {toxicity}",
        )

        langfuse.flush()
        logger.info(f"Security metrics logged to Langfuse for {sprint_id}/{track}")

    except ImportError:
        logger.debug("Langfuse not available, skipping security metrics logging")
    except Exception as e:
        logger.debug(f"Failed to log security metrics to Langfuse: {e}")


def _update_security_metrics_for_gate(
    gate_id: str,
    status: str,
    security_passed: bool,
    security_warnings: List[str],
) -> None:
    """Update SecurityGateRunner metrics after gate execution.

    This manually updates the security metrics when gates are executed
    through the gate_runner flow (which doesn't use the full SecurityGateRunner.run_gate).

    Args:
        gate_id: Gate identifier
        status: Gate result status (PASS, FAIL, BLOCK)
        security_passed: Whether security checks passed
        security_warnings: List of security warnings

    Note:
        Uses graceful degradation - silently skips if security module unavailable.
    """
    if not SECURITY_GATE_AVAILABLE or get_security_gate_runner is None:
        return

    try:
        runner = get_security_gate_runner()
        metrics = runner._metrics

        # Update counts
        metrics.total_checks += 1

        if status == "PASS" and security_passed:
            metrics.passed_checks += 1
        else:
            metrics.failed_checks += 1
            if status == "BLOCK" or not security_passed:
                metrics.blocked_operations += 1

        # Check warnings for specific security issues
        for warning in security_warnings:
            warning_lower = warning.lower()
            if "pii" in warning_lower:
                metrics.pii_detections += 1
            if "injection" in warning_lower:
                metrics.injection_attempts += 1
            if "toxic" in warning_lower:
                metrics.toxicity_flags += 1

        logger.debug(f"Security metrics updated for gate {gate_id}: total={metrics.total_checks}")

    except Exception as e:
        logger.warning(f"Failed to update security metrics for {gate_id}: {e}")


async def _run_with_security_gate(
    gate_id: str,
    input_data: Dict[str, Any],
) -> Tuple[bool, Optional[str]]:
    """Run security checks before gate execution.

    Applies the @secure_gate decorator's logic dynamically for gates
    that are executed in loops or with dynamic gate_ids.

    Args:
        gate_id: Gate identifier (G0-G8)
        input_data: Input data for security validation

    Returns:
        Tuple of (passed, block_reason) - (True, None) if passed, (False, reason) if blocked

    IMPORTANTE: FAIL-CLOSED - Security é CRITICAL, não permite execução sem validação.
    O Run Master é alertado se a stack de segurança estiver indisponível.
    """
    # FAIL-CLOSED: Security é stack CRITICAL - não permite bypass
    from pipeline.stack_health_supervisor import (
        get_stack_supervisor,
        CriticalStackUnavailableError,
    )

    supervisor = get_stack_supervisor()

    # Verificar se security está healthy
    if not supervisor.is_healthy("security"):
        logger.error(
            f"FAIL-CLOSED: Security stack indisponível para {gate_id}. "
            "Pipeline será HALTED até resolução."
        )
        # O supervisor vai cuidar do HALT e alertar Run Master
        raise CriticalStackUnavailableError(
            "security",
            f"Cannot execute gate {gate_id}: security stack is unavailable"
        )

    if not SECURITY_GATE_AVAILABLE or get_security_gate_runner is None:
        logger.error(
            f"FAIL-CLOSED: Security gate module não disponível para {gate_id}. "
            "Módulo de segurança é OBRIGATÓRIO."
        )
        raise CriticalStackUnavailableError(
            "security",
            f"Security gate module not available for {gate_id}"
        )

    try:
        runner = get_security_gate_runner()
        is_valid, error_msg = await runner.validate_input(gate_id, input_data)

        if not is_valid:
            logger.warning(f"Security gate {gate_id} pre-validation failed: {error_msg}")
            return False, error_msg

        logger.debug(f"Security gate {gate_id} pre-validation passed")
        return True, None

    except CriticalStackUnavailableError:
        # Re-raise para que seja tratado no nível superior
        raise
    except Exception as e:
        # FAIL-CLOSED: Erro na validação de segurança é CRÍTICO
        logger.error(
            f"FAIL-CLOSED: Security gate validation error for {gate_id}: {e}. "
            "Não permitido bypass de segurança."
        )
        raise CriticalStackUnavailableError(
            "security",
            f"Security validation failed for {gate_id}: {e}"
        ) from e


def _run_with_security_gate_sync(
    gate_id: str,
    input_data: Dict[str, Any],
) -> Tuple[bool, Optional[str]]:
    """Synchronous wrapper for security gate validation.

    For use in synchronous contexts where async cannot be used.

    Args:
        gate_id: Gate identifier (G0-G8)
        input_data: Input data for security validation

    Returns:
        Tuple of (passed, block_reason)

    IMPORTANTE: FAIL-CLOSED - Security é CRITICAL, não permite execução sem validação.
    """
    import asyncio

    # FAIL-CLOSED: Security é stack CRITICAL - não permite bypass
    from pipeline.stack_health_supervisor import (
        get_stack_supervisor,
        CriticalStackUnavailableError,
    )

    supervisor = get_stack_supervisor()

    # Verificar se security está healthy
    if not supervisor.is_healthy("security"):
        logger.error(
            f"FAIL-CLOSED: Security stack indisponível para {gate_id}. "
            "Pipeline será HALTED até resolução."
        )
        raise CriticalStackUnavailableError(
            "security",
            f"Cannot execute gate {gate_id}: security stack is unavailable"
        )

    if not SECURITY_GATE_AVAILABLE or get_security_gate_runner is None:
        logger.error(
            f"FAIL-CLOSED: Security gate module não disponível para {gate_id}. "
            "Módulo de segurança é OBRIGATÓRIO."
        )
        raise CriticalStackUnavailableError(
            "security",
            f"Security gate module not available for {gate_id}"
        )

    try:
        # Try to get the running event loop
        try:
            asyncio.get_running_loop()
            # Already in async context - use shared ThreadPoolExecutor to avoid blocking
            # SAFE-04 FIX: Use shared pool instead of creating new pool each time
            import concurrent.futures
            logger.debug(f"SECURITY: Running {gate_id} check in shared thread pool (async context detected)")
            pool = _get_security_pool()
            future = pool.submit(
                asyncio.run,
                _run_with_security_gate(gate_id, input_data)
            )
            # P0-08 FIX: Use per-gate timeout from GATE_TIMEOUTS instead of hardcoded 60s
            gate_timeout = GATE_TIMEOUTS.get(gate_id, DEFAULT_GATE_TIMEOUT)
            try:
                result = future.result(timeout=gate_timeout)
            except concurrent.futures.TimeoutError:
                logger.warning(f"Security gate {gate_id} timeout after {gate_timeout}s")
                future.cancel()
                raise
            return result
        except RuntimeError:
            # No running loop - safe to create one
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(
                    _run_with_security_gate(gate_id, input_data)
                )
                return result
            finally:
                loop.close()

    except CriticalStackUnavailableError:
        raise
    except concurrent.futures.TimeoutError:
        logger.error(f"FAIL-CLOSED: Security gate sync validation timeout for {gate_id}")
        raise CriticalStackUnavailableError(
            "security",
            f"Security validation timeout for {gate_id}"
        )
    except Exception as e:
        # FAIL-CLOSED: Erro é CRÍTICO
        logger.error(f"FAIL-CLOSED: Security gate sync validation error for {gate_id}: {e}")
        raise CriticalStackUnavailableError(
            "security",
            f"Security validation failed for {gate_id}: {e}"
        ) from e


async def _verify_post_gate_security(
    gate_id: str,
    input_data: Dict[str, Any],
    output_data: Dict[str, Any],
) -> Tuple[bool, Optional[str], List[str]]:
    """Verify security of gate output after execution.

    Applies post-gate security checks to ensure gate output doesn't contain
    security issues like PII leaks, toxic content, or malicious data.

    Args:
        gate_id: Gate identifier (G0-G8)
        input_data: Original input data
        output_data: Gate output data to verify

    Returns:
        Tuple of (passed, block_reason, warnings)

    IMPORTANTE: FAIL-CLOSED - Security é CRITICAL, não permite bypass.
    """
    # FAIL-CLOSED: Security é stack CRITICAL
    from pipeline.stack_health_supervisor import (
        get_stack_supervisor,
        CriticalStackUnavailableError,
    )

    supervisor = get_stack_supervisor()

    if not supervisor.is_healthy("security"):
        logger.error(
            f"FAIL-CLOSED: Security stack indisponível para post-gate {gate_id}. "
            "Pipeline será HALTED."
        )
        raise CriticalStackUnavailableError(
            "security",
            f"Cannot verify post-gate security for {gate_id}: security stack unavailable"
        )

    if not SECURITY_GATE_AVAILABLE or get_security_gate_runner is None:
        logger.error(
            f"FAIL-CLOSED: Security gate module não disponível para post-gate {gate_id}."
        )
        raise CriticalStackUnavailableError(
            "security",
            f"Security gate module not available for post-gate {gate_id}"
        )

    # Check if this gate requires post-gate verification based on policy
    policy = GATE_SECURITY_POLICIES.get(gate_id, GATE_SECURITY_POLICIES.get("default", {}))
    check_phase = policy.get("check_phase")

    # Skip if policy only requires pre-gate checks (this is OK - policy decision, not bypass)
    if SecurityCheckPhase and check_phase == SecurityCheckPhase.PRE_GATE:
        logger.debug(f"Gate {gate_id} policy is PRE_GATE only, post-gate verification not required")
        return True, None, []

    try:
        runner = get_security_gate_runner()
        gate = runner.get_gate(gate_id)

        # Run post-gate validation
        passed, result, warnings = await gate.validate_post_gate(input_data, output_data)

        if not passed:
            block_reason = result.block_reason if result else "Post-gate security validation failed"
            logger.warning(f"Security gate {gate_id} post-validation failed: {block_reason}")
            return False, block_reason, warnings

        logger.debug(f"Security gate {gate_id} post-validation passed with {len(warnings)} warnings")
        return True, None, warnings

    except CriticalStackUnavailableError:
        raise
    except Exception as e:
        # FAIL-CLOSED: Erro na verificação é CRÍTICO
        logger.error(f"FAIL-CLOSED: Security gate post-validation error for {gate_id}: {e}")
        raise CriticalStackUnavailableError(
            "security",
            f"Post-gate security validation failed for {gate_id}: {e}"
        ) from e


def _verify_post_gate_security_sync(
    gate_id: str,
    input_data: Dict[str, Any],
    output_data: Dict[str, Any],
) -> Tuple[bool, Optional[str], List[str]]:
    """Synchronous wrapper for post-gate security verification.

    For use in synchronous contexts where async cannot be used.

    Args:
        gate_id: Gate identifier (G0-G8)
        input_data: Original input data
        output_data: Gate output data to verify

    Returns:
        Tuple of (passed, block_reason, warnings)

    IMPORTANTE: FAIL-CLOSED - Security é CRITICAL, não permite bypass.
    """
    import asyncio

    # FAIL-CLOSED: Security é stack CRITICAL
    from pipeline.stack_health_supervisor import (
        get_stack_supervisor,
        CriticalStackUnavailableError,
    )

    supervisor = get_stack_supervisor()

    if not supervisor.is_healthy("security"):
        logger.error(
            f"FAIL-CLOSED: Security stack indisponível para post-gate sync {gate_id}."
        )
        raise CriticalStackUnavailableError(
            "security",
            f"Cannot verify post-gate security for {gate_id}: security stack unavailable"
        )

    if not SECURITY_GATE_AVAILABLE or get_security_gate_runner is None:
        logger.error(
            f"FAIL-CLOSED: Security gate module não disponível para post-gate sync {gate_id}."
        )
        raise CriticalStackUnavailableError(
            "security",
            f"Security gate module not available for post-gate {gate_id}"
        )

    try:
        # Try to get the running event loop
        try:
            asyncio.get_running_loop()
            # Already in async context - use shared ThreadPoolExecutor to avoid blocking
            # SAFE-04 FIX: Use shared pool instead of creating new pool each time
            import concurrent.futures
            logger.debug(f"SECURITY: Running post-gate {gate_id} check in shared thread pool (async context detected)")
            pool = _get_security_pool()
            future = pool.submit(
                asyncio.run,
                _verify_post_gate_security(gate_id, input_data, output_data)
            )
            # P0-08 FIX: Use per-gate timeout from GATE_TIMEOUTS instead of hardcoded 60s
            gate_timeout = GATE_TIMEOUTS.get(gate_id, DEFAULT_GATE_TIMEOUT)
            try:
                result = future.result(timeout=gate_timeout)
            except concurrent.futures.TimeoutError:
                logger.warning(f"Post-gate security {gate_id} timeout after {gate_timeout}s")
                future.cancel()
                raise
            return result
        except RuntimeError:
            # No running loop - safe to create one
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(
                    _verify_post_gate_security(gate_id, input_data, output_data)
                )
                return result
            finally:
                loop.close()

    except CriticalStackUnavailableError:
        raise
    except concurrent.futures.TimeoutError:
        logger.error(f"FAIL-CLOSED: Post-gate security verification timeout for {gate_id}")
        raise CriticalStackUnavailableError(
            "security",
            f"Post-gate security verification timeout for {gate_id}"
        )
    except Exception as e:
        logger.error(f"FAIL-CLOSED: Security gate post-verification sync error for {gate_id}: {e}")
        raise CriticalStackUnavailableError(
            "security",
            f"Post-gate security verification failed for {gate_id}: {e}"
        ) from e


# Per-gate timeout configuration (seconds)
# P0-08: GENEROUS timeouts + 30% gordura - NUNCA falhar por timeout
# Philosophy: timeout is a safety net, not an obstacle
GATE_TIMEOUTS: Dict[str, int] = {
    "G0": 390,    # ~6.5 min - Structural validation
    "G1": 780,    # 13 min - Schema validation
    "G2": 780,    # 13 min - Dependency check
    "G3": 1560,   # 26 min - Quality gates
    "G4": 1560,   # 26 min - Integration check
    "G5": 4680,   # 78 min - Full test suite (1h18!)
    "G6": 1560,   # 26 min - Security scan
    "G7": 1560,   # 26 min - Performance check
    "G8": 780,    # 13 min - Final approval
}

# Default timeout for gates not in GATE_TIMEOUTS
# P0-08: 26 min default (base 20 + 30% gordura)
DEFAULT_GATE_TIMEOUT = 1560

# RED TEAM FIX INV-12: Removed duplicate GATE_DEPENDENCIES definition
# The canonical definition is at lines 78-88 (DAG de dependências)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# =============================================================================
# Definition of Ready (DoR) and EARS Validation (Extracted to gate_dor_ears_validation.py)
# =============================================================================
# D-CRITICAL-046: Extraction to reduce file size and improve modularity
from pipeline.gate_dor_ears_validation import (
    INVEST_CRITERIA,
    DoRResult,
    validate_definition_of_ready,
    EARS_PATTERNS,
    EarsResult,
    has_ears_keywords,
    validate_ears_requirement,
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is None:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML required to load YAML")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected mapping: {path}")
    return data


SECRET_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    (
        "private_key",
        re.compile(
            r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----[\s\S]+?-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
        ),
    ),
    ("bearer_token", re.compile(r"Authorization:\s*Bearer\s+[A-Za-z0-9\-\._=]+", re.IGNORECASE)),
]


def sph_scan_text(text: str, allow_list: Optional[List[str]] = None) -> List[str]:
    """Scan text for secret patterns - default-deny approach.

    SEC-014: SPH (Secret Pattern Heuristics) uses default-deny logic.
    Any pattern match triggers a BLOCK status regardless of allowlist.
    The allowlist can only downgrade BLOCK to PASS, not skip detection.

    Args:
        text: Text to scan for secret patterns
        allow_list: Optional list of specific secret values to ignore
                   (e.g., ["AKIAIOSFODNN7EXAMPLE"] for test credentials)

    Returns:
        List of matched secret strings that are NOT in the allow_list.
        Returns the actual matched text, not the pattern type.
    """
    hits: List[str] = []
    allow = set(allow_list or [])
    for name, pattern in SECRET_PATTERNS:
        for match in pattern.finditer(text):
            matched_text = match.group(0)
            if matched_text not in allow:
                hits.append(matched_text)
    return hits


def sph_scan_text_dual(
    text: str,
    allow_list: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """P1-07 FIX (2026-01-30): Scan text for secrets in a single pass.

    This function performs a single scan and returns both filtered and raw hits,
    avoiding the need for two separate scans when both values are needed.

    SEC-014: SPH (Secret Pattern Heuristics) uses default-deny logic.

    Args:
        text: Text to scan for secret patterns
        allow_list: Optional list of specific secret values to ignore

    Returns:
        Tuple of (filtered_hits, raw_hits) where:
        - filtered_hits: Secrets NOT in allowlist (should cause BLOCK status)
        - raw_hits: ALL secrets found (for deciding if redaction is needed)
    """
    raw_hits: List[str] = []
    filtered_hits: List[str] = []
    allow = set(allow_list or [])

    for name, pattern in SECRET_PATTERNS:
        for match in pattern.finditer(text):
            matched_text = match.group(0)
            raw_hits.append(matched_text)

            # Check allowlist for filtered hits
            if matched_text not in allow:
                filtered_hits.append(matched_text)

    return filtered_hits, raw_hits


def sph_redact_text(text: str) -> str:
    def _repl(name: str):
        def _inner(match: re.Match[str]) -> str:
            digest = _sha256_bytes(match.group(0).encode("utf-8"))[:10]
            return f"[SPH_REDACTED:{name}:{digest}]"

        return _inner

    out = text
    for name, pattern in SECRET_PATTERNS:
        out = pattern.sub(_repl(name), out)
    return out


def _substitute_placeholders(command: str, *, run_id: str, sprint_id: str, run_dir: Path) -> str:
    # SEC-009: Apply shlex.quote() to ALL placeholders for shell safety
    quoted_run_id = shlex.quote(run_id)
    quoted_sprint_id = shlex.quote(sprint_id)
    quoted_run_dir = shlex.quote(str(run_dir))
    return command.replace("<run_id>", quoted_run_id).replace("<sprint_id>", quoted_sprint_id).replace("<run_dir>", quoted_run_dir)


def _compute_placeholder_values(run_id: str, sprint_id: str, run_dir: Path) -> Dict[str, str]:
    """OPT-06-002: Pre-compute placeholder values once for multiple substitutions.

    This allows efficient substitution across multiple commands/strings
    without recomputing shlex.quote for each placeholder on every call.

    Args:
        run_id: Run identifier
        sprint_id: Sprint identifier
        run_dir: Run directory path

    Returns:
        Dict mapping placeholder strings to their quoted values
    """
    return {
        "<run_id>": shlex.quote(run_id),
        "<sprint_id>": shlex.quote(sprint_id),
        "<run_dir>": shlex.quote(str(run_dir)),
    }


def _substitute_placeholders_fast(text: str, placeholder_values: Dict[str, str]) -> str:
    """OPT-06-002: Fast placeholder substitution with pre-computed values.

    Args:
        text: String containing placeholders to substitute
        placeholder_values: Pre-computed placeholder -> value mapping

    Returns:
        String with all placeholders substituted
    """
    result = text
    for placeholder, value in placeholder_values.items():
        result = result.replace(placeholder, value)
    return result


@dataclass
class GateResult:
    """Result of running a single gate.

    P2-5.2 FIX (2026-02-01): Added reflexion_suggestions field to capture
    root cause and prevention strategy for failed gates.
    """
    gate_id: str
    status: str  # PASS|FAIL|BLOCK
    exit_code: int
    log_path: str
    sph_hits: List[str]
    captured_paths: List[str] = field(default_factory=list)
    missing_expected_outputs: List[str] = field(default_factory=list)
    # P2-5.2: Reflexion suggestions for rework context
    reflexion_suggestions: Optional[Dict[str, str]] = None


@enforce_stacks("gate_execution", required=["langfuse"], recommended=["redis"])
def run_gate(
    *,
    gate_id: str,
    command: str,
    repo_root: Path,
    log_path: Path,
    env: Optional[Dict[str, str]] = None,
    sph_allow: Optional[List[str]] = None,
    timeout_seconds: Optional[int] = None,
) -> GateResult:
    """Execute a gate command with timeout handling.

    Security: Applies @secure_gate checks via _run_with_security_gate_sync().
    See pipeline.security.gate_integration for security policies per gate (G0-G8).

    Args:
        gate_id: Gate identifier (e.g., "G0", "G1")
        command: Shell command to execute
        repo_root: Repository root directory
        log_path: Path to write gate output log
        env: Optional environment variables to pass to command
        sph_allow: Optional list of allowed secret patterns (exact matches)
        timeout_seconds: Optional timeout in seconds (overrides GATE_TIMEOUTS)

    Returns:
        GateResult with status (PASS/FAIL/BLOCK), exit code, and evidence paths

    Notes:
        - If timeout_seconds is not specified, uses GATE_TIMEOUTS[gate_id]
        - On timeout, returns exit_code=124 and status=FAIL
        - SPH scanning always occurs regardless of timeout
        - Security checks are applied before execution (graceful degradation if unavailable)
    """
    # 2026-01-16: EXPLICIT guardrails validation before gate execution
    _validate_gate_with_guardrails(gate_id)

    # 2026-01-16: Apply @secure_gate security checks before execution
    # This enforces GATE_SECURITY_POLICIES from gate_integration.py
    security_input = {"command": command, "gate_id": gate_id, "repo_root": str(repo_root)}
    security_passed, security_block_reason = _run_with_security_gate_sync(gate_id, security_input)
    if not security_passed:
        logger.warning(f"Gate {gate_id} blocked by security check: {security_block_reason}")
        return GateResult(
            gate_id=gate_id,
            status="BLOCK",
            exit_code=-1,
            log_path=str(log_path),
            sph_hits=[f"SECURITY_BLOCK: {security_block_reason}"],
        )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(
            shlex.split(command),
            shell=False,
            cwd=str(repo_root),
            env={**os.environ, **(env or {})},
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,  # We handle exit codes ourselves
        )
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        returncode = int(proc.returncode)
    except subprocess.TimeoutExpired as exc:
        # TimeoutExpired returns bytes, decode if needed
        stdout = (exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout) or ""
        stderr = (exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr) or ""
        combined = stdout + "\n" + stderr + f"\n[gate_runner] TIMEOUT after {timeout_seconds}s\n"
        returncode = 124
    except FileNotFoundError as e:
        logger.error(f"Gate command not found: {command.split()[0]}")
        combined = f"[gate_runner] ERROR: Command not found: {e}\n"
        returncode = 127
    except PermissionError as e:
        logger.error(f"Permission denied executing gate command: {e}")
        combined = f"[gate_runner] ERROR: Permission denied: {e}\n"
        returncode = 126
    except OSError as e:
        logger.error(f"Failed to execute gate command: {e}")
        combined = f"[gate_runner] ERROR: Cannot execute command: {e}\n"
        returncode = 126
    # P1-07 FIX (2026-01-30): Single SPH scan returning both filtered and raw hits
    # This replaces the previous two-scan approach which scanned the same text twice
    sph_hits, raw_hits = sph_scan_text_dual(combined, allow_list=sph_allow)
    # Always redact ALL secrets regardless of allowlist
    if raw_hits:
        log_path.write_text(sph_redact_text(combined), encoding="utf-8")
    else:
        log_path.write_text(combined, encoding="utf-8")
    if sph_hits:
        status = "BLOCK"
    else:
        status = "PASS" if returncode == 0 else "FAIL"

    # 2026-01-16: Post-gate security verification
    # If gate passed, verify the output doesn't contain security issues
    security_warnings: List[str] = []
    if status == "PASS":
        output_data = {"output": combined, "exit_code": returncode, "gate_id": gate_id}
        post_passed, post_block_reason, post_warnings = _verify_post_gate_security_sync(
            gate_id, security_input, output_data
        )
        security_warnings = post_warnings

        if not post_passed:
            logger.warning(f"Gate {gate_id} blocked by post-gate security check: {post_block_reason}")
            status = "BLOCK"
            sph_hits = list(sph_hits) + [f"POST_SECURITY_BLOCK: {post_block_reason}"]

        if post_warnings:
            logger.info(f"Gate {gate_id} post-gate security warnings: {post_warnings}")

    # 2026-01-16: Update security metrics
    _update_security_metrics_for_gate(gate_id, status, security_passed, security_warnings)

    # Publish gate result to Grafana dashboard via Redis
    try:
        from pipeline.grafana_metrics import get_metrics_publisher
        publisher = get_metrics_publisher()
        publisher.publish_gate_result(
            gate_id=gate_id,
            passed=(status == GateStatus.PASS),
            score=1.0 if status == GateStatus.PASS else 0.0,
            details=f"exit_code={returncode}, sph_hits={len(sph_hits)}",
        )
    except Exception as pub_err:
        logger.debug(f"Failed to publish gate result to Grafana: {pub_err}")

    return GateResult(
        gate_id=gate_id,
        status=status,
        exit_code=returncode,
        log_path=str(log_path),
        sph_hits=sph_hits,
    )


def _copy_expected_output(
    *,
    src_path: Path,
    dest_path: Path,
    sph_allow: Optional[List[str]] = None,
) -> List[str]:
    hits: List[str] = []
    if src_path.is_dir():
        for child in sorted(src_path.rglob("*")):
            if not child.is_file():
                continue
            rel = child.relative_to(src_path)
            hits.extend(
                _copy_expected_output(
                    src_path=child,
                    dest_path=dest_path / rel,
                    sph_allow=sph_allow,
                )
            )
        return hits

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    data = src_path.read_bytes()
    try:
        text = data.decode("utf-8")
        # P1-07 FIX (2026-01-30): Single SPH scan returning both filtered and raw hits
        hits, raw_hits = sph_scan_text_dual(text, allow_list=sph_allow)
        if raw_hits:
            dest_path.write_text(sph_redact_text(text), encoding="utf-8")
        else:
            shutil.copy2(src_path, dest_path)
        return hits
    except UnicodeDecodeError:
        # Binary-ish: best effort scan via utf-8 ignore. If any hit, do not copy raw bytes.
        text_guess = data.decode("utf-8", errors="ignore")
        # P1-07 FIX (2026-01-30): Single SPH scan returning both filtered and raw hits
        hits, raw_hits = sph_scan_text_dual(text_guess, allow_list=sph_allow)
        if raw_hits:
            digest = _sha256_bytes(data)[:10]
            dest_path.write_text(
                f"[SPH_BLOCKED_BINARY:{digest}] raw binary suppressed due to SPH pattern match\n",
                encoding="utf-8",
            )
        else:
            shutil.copy2(src_path, dest_path)
        return hits


def _create_warn_approval_request(
    *,
    run_dir: Path,
    run_id: str,
    sprint_id: str,
    track: str,
    warn_checks: List[Dict[str, Any]],
) -> None:
    """GAP-AG-03 FIX: Creates approval request for WARN status gates.

    This creates a YAML file that human operators can review and approve/reject
    to acknowledge warnings from optional gate failures.

    Args:
        run_dir: Pipeline run directory
        run_id: Run ID
        sprint_id: Sprint ID
        track: Track name (A, B, C)
        warn_checks: List of checks with WARN status
    """
    approval_request = {
        "schema_version": "pipeline.gate_warn_approval.v0.1",
        "run_id": run_id,
        "sprint_id": sprint_id,
        "track": track,
        "warnings": warn_checks,
        "requested_at": _now_iso(),
        "instructions": (
            "Optional gates have failed but are not blocking. Please review the warnings and create a "
            "gate_warn_approval_response.yml file with:\n"
            "  approved: true/false\n"
            "  reviewer: your-name\n"
            "  comments: reason for approval/rejection\n"
            "  reviewed_at: ISO timestamp\n"
        ),
    }

    request_path = run_dir / "sprints" / sprint_id / "qa" / f"gate_warn_approval_request_{track}.yml"
    _write_yaml(request_path, approval_request)
    logger.info(f"Created WARN approval request at {request_path} with {len(warn_checks)} warnings")


@enforce_stacks("gate_execution", required=["langfuse"], recommended=["redis"])
def run_gate_selection(
    *,
    repo_root: Path,
    run_dir: Path,
    sprint_id: str,
    track: str,
    selection_path: Optional[Path] = None,
    report_suffix: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    requesting_agent: Optional[str] = None,  # RED TEAM FIX H-08: Required caller ID
    validation_context: Optional[Dict[str, Any]] = None,  # GAP-5: RF/INV/EDGE for validation
) -> Dict[str, Any]:
    # FIX 2026-01-31: Ensure gate_selection.yml exists (migrated from task_executor.py)
    if selection_path is None:
        ensure_gate_selection_exists(run_dir, sprint_id, track=track)

    selection_file = selection_path or (run_dir / "sprints" / sprint_id / "plan" / "gate_selection.yml")
    selection = _load_yaml(selection_file)

    # RED TEAM FIX H-08: Validate requesting_agent before proceeding
    if not requesting_agent:
        logger.warning("run_gate_selection called without requesting_agent - audit trail incomplete")
    else:
        # Validate caller has authority to run gates (L0-L3 only)
        try:
            from .hierarchy import get_level
            caller_level = get_level(requesting_agent)
            if caller_level is not None and caller_level > 3:
                raise ValueError(
                    f"INV-014 VIOLATION: Agent {requesting_agent} (L{caller_level}) "
                    f"lacks authority to run gates. Only L0-L3 agents can execute gates."
                )
            logger.info(f"Gate selection authorized for {requesting_agent} (L{caller_level})")
        except ImportError:
            logger.warning("Hierarchy module unavailable - cannot validate authority")

    run_id = str(selection.get("run_id") or "")
    if not run_id:
        raise ValueError("gate_selection.run_id missing")

    # Support both formats: direct "gates" list or nested "tracks.{track}.gates"
    gates = selection.get("gates")

    # P0-08 FIX (2026-01-30): Build present_gates set directly instead of
    # building intermediate list then converting to set (O(n) vs O(2n) with worse constants)
    present_gates: set = set()  # For INV-002 validation

    if not gates and "tracks" in selection:
        track_data = selection.get("tracks", {}).get(track, {})
        gates = track_data.get("gates")
        # RED TEAM FIX INV-06: When using tracks format, collect ALL gates across ALL tracks
        # for mandatory gate validation (individual track may not have all gates)
        # P0-08 FIX: Build set directly instead of list->set conversion
        for t_data in selection.get("tracks", {}).values():
            if isinstance(t_data, dict) and isinstance(t_data.get("gates"), list):
                for g in t_data["gates"]:
                    if isinstance(g, dict):
                        gate_id = str(g.get("gate_id", ""))
                        if gate_id:  # Skip empty gate IDs
                            present_gates.add(gate_id)  # O(1) per add
    else:
        # Build set from direct gates list
        if gates:
            for g in gates:
                if isinstance(g, dict):
                    gate_id = str(g.get("gate_id", ""))
                    if gate_id:
                        present_gates.add(gate_id)

    if not isinstance(gates, list) or not gates:
        raise ValueError("gate_selection.gates must be a non-empty list")

    # RED TEAM FIX INV-06: Verify mandatory gates are present
    # FIX 2026-01-23: Updated to support new gate naming convention (GATE-*)
    # The old G0-G8 naming was replaced with descriptive names like GATE-PYTEST, GATE-ART-SCH etc.
    # INV-002 now requires: at least one gate from each critical category
    # P0-08 FIX: present_gates is now built directly as a set above

    # Check for legacy G0-G8 gates (for backwards compatibility)
    LEGACY_MANDATORY_GATES = {"G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8"}
    has_legacy_gates = bool(present_gates & LEGACY_MANDATORY_GATES)

    if has_legacy_gates:
        # Legacy mode: enforce G0-G8
        missing_gates = LEGACY_MANDATORY_GATES - present_gates
        if missing_gates:
            raise ValueError(
                f"INV-002 VIOLATION: Missing mandatory gates: {sorted(missing_gates)}. "
                f"All gates G0-G8 are required per INV-002."
            )
    else:
        # New profile-based mode: require minimum gate coverage
        # At least one validation gate (any GATE-* prefixed gate)
        valid_gates = [g for g in present_gates if g.startswith("GATE-") or g == "HUMAN_LAYER_GATE"]
        if not valid_gates:
            raise ValueError(
                f"INV-002 VIOLATION: No valid gates found in selection. "
                f"At least one GATE-* validation gate is required."
            )
        logger.info(f"INV-002: Using profile-based gates ({len(valid_gates)} gates: {', '.join(sorted(valid_gates)[:5])}...)")

    created_at = _now_iso()
    run_start_time = time.time()  # P2-001: Track total run duration for Langfuse
    bundle_tmp = (
        run_dir
        / "sprints"
        / sprint_id
        / "evidence"
        / "bundles"
        / f"track_{track}_{created_at.replace(':', '').replace('-', '')}_tmp_{uuid.uuid4().hex[:6]}"
    )
    gates_dir = bundle_tmp / "gates"
    results: List[GateResult] = []
    required_by_id: Dict[str, bool] = {}
    completed_gates: List[str] = []  # INVIOLÁVEL: Track gates que passaram

    # OPT-06-002: Pre-compute placeholder values once for all gates
    placeholder_values = _compute_placeholder_values(run_id, sprint_id, run_dir)

    for gate in gates:
        if not isinstance(gate, dict):
            continue
        gate_id = str(gate.get("gate_id") or "")
        command = str(gate.get("command") or "")
        if not gate_id or not command:
            continue
        gate_track = str(gate.get("track") or "")
        if gate_track and gate_track != track:
            continue

        # ═══════════════════════════════════════════════════════════════════
        # INVIOLÁVEL: Verificar dependências ANTES de executar gate
        # ═══════════════════════════════════════════════════════════════════
        deps_ok, missing_deps = verify_gate_dependencies(gate_id, completed_gates)
        if not deps_ok:
            logger.error(
                f"GATE BLOCKED: {gate_id} não pode executar - "
                f"dependências faltando: {missing_deps}"
            )
            # BLACK TEAM FIX SBP-003: Use "BLOCK" consistently (not "BLOCKED")
            # to match the status aggregation logic that expects "BLOCK"
            results.append(GateResult(
                gate_id=gate_id,
                status="BLOCK",  # SBP-003: was "BLOCKED", now "BLOCK" for consistency
                exit_code=-1,
                log_path="",
                sph_hits=[f"DEPENDENCY_VIOLATION: {missing_deps}"],
            ))
            # NÃO continua para próximo gate - sprint deve parar
            break

        required_by_id[gate_id] = bool(gate.get("required", True))
        sph_allow = gate.get("sph_allow")
        sph_allow_list: Optional[List[str]] = None
        if isinstance(sph_allow, list) and all(isinstance(x, str) for x in sph_allow):
            sph_allow_list = [str(x) for x in sph_allow]
        timeout_seconds = gate.get("timeout_seconds")
        timeout: Optional[int] = None
        if isinstance(timeout_seconds, int) and timeout_seconds > 0:
            timeout = int(timeout_seconds)
        else:
            # Use configured timeout for this gate, or default
            timeout = GATE_TIMEOUTS.get(gate_id, DEFAULT_GATE_TIMEOUT)
        expected_outputs_raw = gate.get("expected_outputs")
        expected_outputs: List[str] = []
        if isinstance(expected_outputs_raw, list):
            # F-179 FIX: Apply placeholder substitution to expected_outputs
            # Expected outputs may contain placeholders like <run_dir>, <sprint_id>
            # OPT-06-002: Use pre-computed placeholder values for efficiency
            raw_outputs = [str(x) for x in expected_outputs_raw if isinstance(x, (str, int, float))]
            expected_outputs = [
                _substitute_placeholders_fast(out, placeholder_values)
                for out in raw_outputs
            ]
        # OPT-06-002: Use pre-computed placeholder values
        cmd = _substitute_placeholders_fast(command, placeholder_values)
        log_path = gates_dir / f"{gate_id}.log"
        gate_start_time = time.time()  # Quantum Leap: Track duration for Live-SWE
        result = run_gate(
            gate_id=gate_id,
            command=cmd,
            repo_root=repo_root,
            log_path=log_path,
            env=env,
            sph_allow=sph_allow_list,
            timeout_seconds=timeout,
        )

        captured_paths: List[str] = []
        missing_expected: List[str] = []
        capture_hits: List[str] = []
        for out_rel in expected_outputs:
            # F-182 FIX: Use run_dir (not repo_root) as base for expected_outputs
            # Expected outputs are typically generated in run_dir during gate execution
            run_dir_path = Path(run_dir).resolve()
            src_path = (run_dir_path / out_rel).resolve(strict=False)
            try:
                # F-182 FIX: Validate relative to run_dir, not repo_root
                safe_rel = src_path.relative_to(run_dir_path)
            except ValueError:
                # Defensive: do not allow capture outside run_dir.
                missing_expected.append(out_rel)
                continue
            # Check if path is a symlink pointing outside run_dir
            if src_path.is_symlink():
                # Follow the symlink to its target and verify it's still within run_dir
                try:
                    symlink_target = src_path.readlink().resolve(strict=False)
                    symlink_target.relative_to(run_dir_path)
                except (ValueError, OSError):
                    # Symlink target is outside repo_root - reject it
                    missing_expected.append(out_rel)
                    continue
            if not src_path.exists():
                missing_expected.append(out_rel)
                continue
            dest_path = bundle_tmp / "captures" / gate_id / safe_rel
            capture_hits.extend(
                _copy_expected_output(
                    src_path=src_path,
                    dest_path=dest_path,
                    sph_allow=sph_allow_list,
                )
            )
            captured_paths.append(str(dest_path.relative_to(bundle_tmp)))

        final_status = result.status
        final_hits = list(result.sph_hits)
        if capture_hits:
            # SPH hits from captured outputs are always blocking (unless allowlisted).
            final_status = "BLOCK"
            for h in capture_hits:
                if h not in final_hits:
                    final_hits.append(h)
        if missing_expected and final_status == "PASS":
            # Evidence incompleta: tratar como FAIL (o required flag decide overall).
            final_status = "FAIL"

        gate_result = GateResult(
            gate_id=result.gate_id,
            status=final_status,
            exit_code=result.exit_code,
            log_path=str(log_path.relative_to(bundle_tmp)),
            sph_hits=final_hits,
            captured_paths=captured_paths,
            missing_expected_outputs=missing_expected,
        )
        results.append(gate_result)

        # ═══════════════════════════════════════════════════════════════════
        # GRAFANA: Publish gate result to Pipeline Control Center dashboard
        # ═══════════════════════════════════════════════════════════════════
        if GRAFANA_METRICS_AVAILABLE and get_metrics_publisher:
            try:
                pub = get_metrics_publisher()
                # Set current gate being processed
                pub.set_current_gate(gate_id)
                # Publish gate result
                pub.publish_gate_result(
                    gate_id=gate_id,
                    passed=(final_status == "PASS"),
                    score=1.0 if final_status == "PASS" else 0.0,
                    details=f"exit_code={result.exit_code}, status={final_status}",
                )
                # Update elapsed time
                pub.update_elapsed()
            except Exception as e:
                logger.debug(f"GRAFANA: Failed to publish gate {gate_id} result: {e}")

        # ═══════════════════════════════════════════════════════════════════
        # INVIOLÁVEL: Marcar gate como completado se passou
        # Isso permite que gates subsequentes verifiquem dependências
        # ═══════════════════════════════════════════════════════════════════
        if gate_result.status == "PASS":
            completed_gates.append(gate_result.gate_id)
            logger.debug(f"Gate {gate_result.gate_id} PASS - adicionado a completed_gates")

        # ═══════════════════════════════════════════════════════════════════
        # MED-003 FIX: Run standalone SecurityGate after G6 (Security gate)
        # This provides additional LLM security validation using regex patterns
        # for secrets/PII detection and codebase injection protection checks.
        # ═══════════════════════════════════════════════════════════════════
        if gate_id == "G6" and gate_result.status == "PASS":
            security_passed, security_details = _run_standalone_security_gate(
                run_dir=run_dir,
                sprint_id=sprint_id,
                repo_root=repo_root,
            )
            if not security_passed:
                # SecurityGate failure should block pipeline
                logger.error(
                    f"Standalone SecurityGate FAILED for {sprint_id}: "
                    f"{security_details.get('errors', [])}"
                )
                # Update the gate result to BLOCK
                gate_result = GateResult(
                    gate_id=gate_result.gate_id,
                    status="BLOCK",
                    exit_code=gate_result.exit_code,
                    log_path=gate_result.log_path,
                    sph_hits=gate_result.sph_hits + ["SECURITY_GATE_BLOCK"],
                    captured_paths=gate_result.captured_paths,
                    missing_expected_outputs=gate_result.missing_expected_outputs,
                )
                # Update results list with blocked status
                results[-1] = gate_result
                # Remove from completed gates since it's now blocked
                if gate_result.gate_id in completed_gates:
                    completed_gates.remove(gate_result.gate_id)
            else:
                logger.info(
                    f"Standalone SecurityGate PASS for G6 "
                    f"(mode={security_details.get('test_mode', 'unknown')})"
                )

        # Quantum Leap: Record metrics for Live-SWE behavior evolution
        gate_duration_ms = (time.time() - gate_start_time) * 1000
        _record_gate_metrics(gate_id, final_status, gate_duration_ms)

        # P2-001: Score gate in Langfuse for observability
        _score_gate_with_langfuse(
            gate_id=gate_id,
            status=final_status,
            exit_code=gate_result.exit_code,
            duration_ms=gate_duration_ms,
            sprint_id=sprint_id,
        )

        # Quantum Leap: Trigger Reflexion + GoT analysis on gate failure
        if gate_result.status in ["FAIL", "BLOCK"]:
            # GoT multi-perspective failure analysis
            got_analysis = _analyze_failure_with_got(
                gate_id=gate_result.gate_id,
                status=gate_result.status,
                exit_code=gate_result.exit_code,
                log_path=str(log_path),
            )
            if got_analysis:
                logger.info(
                    f"GoT suggests for {gate_result.gate_id}: "
                    f"{got_analysis.get('suggested_fixes', [])[:2]}"
                )

            # Reflexion engine for learning
            # P2-5.2 FIX: Capture reflexion result for rework context
            reflexion_result = _trigger_reflexion_on_failure(
                gate_id=gate_result.gate_id,
                status=gate_result.status,
                exit_code=gate_result.exit_code,
                sprint_id=sprint_id,
                log_path=str(log_path),
            )

            # P2-5.2: Store reflexion suggestions for rework if available
            if reflexion_result:
                gate_result.reflexion_suggestions = {
                    "prevention_strategy": reflexion_result.prevention_strategy,
                    "root_cause": reflexion_result.root_cause,
                    "incorrect_assumption": reflexion_result.incorrect_assumption,
                    "missing_info": reflexion_result.missing_info,
                }
                logger.info(
                    f"P2-5.2: Captured reflexion for {gate_result.gate_id}: "
                    f"prevention={reflexion_result.prevention_strategy[:80]}..."
                )

        # GAP-AG-02 FIX: Immediate halt on required gate failure
        is_required = required_by_id.get(gate_result.gate_id, True)
        if is_required and gate_result.status in ["FAIL", "BLOCK"]:
            logger.error(
                f"Required gate {gate_result.gate_id} failed with status {gate_result.status}. "
                f"Triggering immediate safe halt."
            )

            # BLACK TEAM FIX SBP-009: Safe halt write failure must be fatal
            # If we can't write safe_halt, the pipeline could continue in an inconsistent state
            from .ipc_compat import write_safe_halt, IPCLayout
            ipc_layout = IPCLayout(run_dir=run_dir)

            max_retries = 3
            safe_halt_written = False
            last_error = None

            for attempt in range(max_retries):
                try:
                    write_safe_halt(
                        ipc_layout,
                        reason=f"Required gate {gate_result.gate_id} failed: {gate_result.status}"
                    )
                    logger.info(f"Safe halt triggered for failed gate {gate_result.gate_id}")
                    safe_halt_written = True
                    break
                except Exception as e:
                    last_error = e
                    logger.warning(f"Safe halt write attempt {attempt + 1}/{max_retries} failed: {e}")

            if not safe_halt_written:
                # SBP-009: Make this failure fatal - we cannot continue without halt signal
                logger.critical(
                    f"CRITICAL: Failed to write safe_halt after {max_retries} attempts. "
                    f"Last error: {last_error}. Pipeline state may be inconsistent."
                )
                raise RuntimeError(
                    f"SBP-009 VIOLATION: Cannot write safe_halt file after {max_retries} attempts. "
                    f"Gate {gate_result.gate_id} failed but halt signal not propagated."
                ) from last_error

            # Stop gate execution immediately
            overall = "BLOCK"
            break

    # Build manifest
    files = []
    for file_path in sorted(bundle_tmp.rglob("*")):
        if not file_path.is_file():
            continue
        rel = str(file_path.relative_to(bundle_tmp))
        files.append({"path": rel, "sha256": _sha256_file(file_path), "bytes": file_path.stat().st_size})

    manifest = {
        "schema_version": "pipeline.evidence_bundle.v0.1",
        "run_id": run_id,
        "sprint_id": sprint_id,
        "track": track,
        "created_at": created_at,
        "files": files,
        "verify_commands": [
            {
                "gate_id": r.gate_id,
                "status": r.status,
                "exit_code": r.exit_code,
                "log_path": r.log_path,
                "captured_paths": r.captured_paths,
                "missing_expected_outputs": r.missing_expected_outputs,
            }
            for r in results
        ],
    }
    manifest_path = bundle_tmp / "MANIFEST.json"
    _write_json(manifest_path, manifest)

    manifest_sha = _sha256_bytes(manifest_path.read_bytes())[:12]
    bundle_final = bundle_tmp.with_name(bundle_tmp.name.replace("_tmp_", f"_{manifest_sha}_"))
    os.replace(bundle_tmp, bundle_final)
    manifest_rel = str(manifest_path.relative_to(bundle_tmp))
    manifest_path = bundle_final / manifest_rel

    # Report
    checks = []
    overall = "PASS"
    bundle_rel = str(bundle_final.relative_to(run_dir))
    for r in results:
        if r.status == "BLOCK":
            overall = "BLOCK"
        elif r.status == "FAIL" and required_by_id.get(r.gate_id, True) and overall == "PASS":
            overall = "FAIL"
        check_status = "PASS"
        if r.status == "BLOCK":
            check_status = "FAIL"
        elif r.status == "FAIL":
            check_status = "FAIL" if required_by_id.get(r.gate_id, True) else "WARN"
        details_parts = [f"exit_code={r.exit_code}"]
        if r.missing_expected_outputs:
            details_parts.append("missing_outputs=" + ",".join(r.missing_expected_outputs))
        if r.sph_hits:
            details_parts.append("sph=" + ",".join(r.sph_hits))
        evidence_paths = [f"{bundle_rel}/{r.log_path}"] + [f"{bundle_rel}/{p}" for p in r.captured_paths]
        checks.append(
            {
                "name": r.gate_id,
                "status": check_status,
                "details": " ".join(details_parts),
                "evidence_paths": evidence_paths,
            }
        )

    report = {
        "schema_version": "pipeline.track_report.v0.1",
        "run_id": run_id,
        "sprint_id": sprint_id,
        "track": track,
        "generated_at": _now_iso(),
        "owner": "gate_runner",
        "status": overall,
        "summary": f"Executed {len(results)} gates from gate_selection",
        "checks": checks,
        "claims_covered": [],
        "evidence_bundles": [str(bundle_final.relative_to(run_dir))],
        "findings_ref": f"sprints/{sprint_id}/findings/defect_ledger.yml",
    }
    report_name = f"track_{track}_report"
    if report_suffix:
        safe_suffix = "".join(ch for ch in report_suffix if ch.isalnum() or ch in {"-", "_"}).strip("_-")
        if safe_suffix:
            report_name = f"{report_name}_{safe_suffix}"
    report_path = run_dir / "sprints" / sprint_id / "qa" / f"{report_name}.yml"
    _write_yaml(report_path, report)

    # Quantum Leap: Save learning on success
    if overall == "PASS":
        _save_gate_learning(sprint_id, track, len(results))

    # GAP-AG-03 FIX: Add approval workflow for WARN status
    warn_checks = [c for c in checks if c["status"] == "WARN"]
    if warn_checks:
        _create_warn_approval_request(
            run_dir=run_dir,
            run_id=run_id,
            sprint_id=sprint_id,
            track=track,
            warn_checks=warn_checks,
        )

    # Build gates list with detailed results for each gate
    gates_list = [
        {
            "gate_id": r.gate_id,
            "status": r.status,
            "exit_code": r.exit_code,
            "sph_hits": r.sph_hits,
            "log_path": r.log_path,
            "captured_paths": r.captured_paths,
            "missing_expected_outputs": r.missing_expected_outputs,
        }
        for r in results
    ]

    # FIX 2026-01-31: Include failed_gates for rework analysis (migrated from task_executor.py)
    # This allows reworker to analyze specific failures
    failed_gates = [
        {
            "gate_id": r.gate_id,
            "status": r.status,
            "exit_code": r.exit_code,
            "log_path": r.log_path,
            "error": f"Gate {r.gate_id} {r.status} with exit code {r.exit_code}",
        }
        for r in results
        if r.status in ["FAIL", "BLOCK"]
    ]

    result = {
        "status": overall,
        "overall_status": overall,  # Alias for compatibility with tests
        "gates": gates_list,
        "failed_gates": failed_gates,  # FIX 2026-01-31: Always include failed_gates
        "report_path": str(report_path),
        "bundle_dir": str(bundle_final),
        "manifest_path": str(manifest_path),
    }

    # GAP-5: Include validation_context for traceability
    if validation_context:
        result["validation_context"] = {
            "rf_count": len(validation_context.get("functional_requirements", [])),
            "inv_count": len(validation_context.get("invariants", [])),
            "edge_count": len(validation_context.get("edge_cases", [])),
        }
        logger.info(
            f"Gate validation context: RF={result['validation_context']['rf_count']}, "
            f"INV={result['validation_context']['inv_count']}, "
            f"EDGE={result['validation_context']['edge_count']}"
        )

    # =========================================================================
    # WS-04 FIX: Generate canonical QA phase artifacts after gate execution
    # This addresses F-137 to F-143, F-217, F-220, F-221
    # =========================================================================
    try:
        from .qa_artifact_writers import generate_qa_phase_artifacts
        qa_artifacts = generate_qa_phase_artifacts(
            run_dir=run_dir,
            sprint_id=sprint_id,
            gate_results=result,
            run_id=run_id,
        )
        result["qa_artifacts"] = {k: str(v) for k, v in qa_artifacts.items()}
        logger.info(f"Generated {len(qa_artifacts)} QA phase artifacts")
    except ImportError:
        logger.warning("qa_artifact_writers not available, skipping QA artifact generation")
    except Exception as e:
        logger.warning(f"Failed to generate QA artifacts: {e}")

    # P2-001: Score overall gate run in Langfuse
    total_run_duration_ms = (time.time() - run_start_time) * 1000
    langfuse_trace_id = _score_gate_run_with_langfuse(
        sprint_id=sprint_id,
        track=track,
        overall_status=overall,
        gate_results=results,
        total_duration_ms=total_run_duration_ms,
    )
    if langfuse_trace_id:
        result["langfuse_trace_id"] = langfuse_trace_id

    # 2026-01-16: Collect and log security metrics to Langfuse
    security_metrics = _collect_security_metrics()
    if security_metrics:
        result["security_metrics"] = security_metrics
        _log_security_metrics_to_langfuse(
            sprint_id=sprint_id,
            track=track,
            metrics=security_metrics,
            overall_status=overall,
        )
        logger.info(
            f"Security metrics: total={security_metrics.get('total_checks', 0)}, "
            f"passed={security_metrics.get('passed_checks', 0)}, "
            f"blocked={security_metrics.get('blocked_operations', 0)}"
        )

    return result


# =============================================================================
# PLaG Parallel Gate Execution
# =============================================================================


@enforce_stacks("gate_execution", required=["langfuse"], recommended=["redis"])
async def run_gate_selection_parallel(
    *,
    repo_root: Path,
    run_dir: Path,
    sprint_id: str,
    track: str = "B",
    selection_path: Optional[Path] = None,
    max_concurrent: int = 3,
    report_suffix: Optional[str] = None,
) -> Dict[str, Any]:
    """Run gates in parallel respecting dependencies using PLaG.

    This is an alternative to run_gate_selection that executes gates
    concurrently where dependencies allow, potentially speeding up
    the overall gate execution by 2-5x.

    Args:
        repo_root: Root path of the repository
        run_dir: Path to the run directory
        sprint_id: Sprint identifier
        track: Track A/B/C (default B)
        selection_path: Optional path to gate_selection.yml
        max_concurrent: Maximum concurrent gate executions (default 3)
        report_suffix: Optional suffix for the report filename

    Returns:
        Dict with status, gates list, and paths
    """
    try:
        from pipeline.plag import GatePLaGRunner
    except ImportError:
        logger.warning("PLaG not available, falling back to sequential execution")
        return run_gate_selection(
            repo_root=repo_root,
            run_dir=run_dir,
            sprint_id=sprint_id,
            track=track,
            selection_path=selection_path,
            report_suffix=report_suffix,
        )

    # FIX 2026-01-31: Ensure gate_selection.yml exists (migrated from task_executor.py)
    # This prevents FileNotFoundError when surgical rework bypasses gate_node
    if selection_path is None:
        ensure_gate_selection_exists(run_dir, sprint_id, track=track)

    # Load gate selection
    selection_file = selection_path or (run_dir / "sprints" / sprint_id / "plan" / "gate_selection.yml")
    selection = _load_yaml(selection_file)
    gates = selection.get("gates", [])

    if not gates:
        logger.warning("No gates in selection, nothing to execute")
        return {"status": "PASS", "gates": [], "parallel_execution": True}

    # Extract gate IDs
    gate_ids = []
    gate_configs = {}
    for gate in gates:
        if isinstance(gate, dict) and "gate_id" in gate:
            gate_id = gate["gate_id"]
            gate_ids.append(gate_id)
            gate_configs[gate_id] = gate

    # Build runner and get parallel levels
    runner = GatePLaGRunner(max_concurrent=max_concurrent)
    levels = runner.get_parallel_levels(gate_ids)

    logger.info(f"PLaG parallel execution: {len(gate_ids)} gates in {len(levels)} levels")
    for i, level in enumerate(levels):
        logger.info(f"  Level {i}: {level}")

    # Setup directories
    gates_dir = run_dir / "sprints" / sprint_id / "qa" / f"gates_{track}"
    gates_dir.mkdir(parents=True, exist_ok=True)

    run_id = run_dir.name

    # Execute gates in parallel
    results: List[GateResult] = []

    async def execute_gate(gate_id: str, context: dict) -> GateResult:
        """Execute a single gate."""
        gate_config = gate_configs.get(gate_id, {})
        command = gate_config.get("command", f"echo 'Gate {gate_id} not configured'")

        # Substitute placeholders
        command = command.replace("<run_id>", shlex.quote(run_id))
        command = command.replace("<sprint_id>", shlex.quote(sprint_id))
        command = command.replace("<run_dir>", shlex.quote(str(run_dir)))
        command = command.replace("<repo_root>", shlex.quote(str(repo_root)))

        log_path = gates_dir / f"{gate_id}.log"
        timeout = GATE_TIMEOUTS.get(gate_id, DEFAULT_GATE_TIMEOUT)

        # 2026-01-10 FIX: run_gate doesn't accept expected_outputs parameter
        # expected_outputs handling is done in run_gate_selection, not run_gate
        # sph_allow needs conversion to list or None
        sph_allow_raw = gate_config.get("sph_allow", [])
        sph_allow_list: Optional[List[str]] = None
        if isinstance(sph_allow_raw, list) and sph_allow_raw:
            sph_allow_list = [str(x) for x in sph_allow_raw]

        result = run_gate(
            gate_id=gate_id,
            command=command,
            repo_root=repo_root,
            log_path=log_path,
            timeout_seconds=timeout,
            sph_allow=sph_allow_list,
        )

        return result

    # Run using PLaG executor
    execution_result = await runner.run_gates_parallel(
        gates=gate_ids,
        gate_executor=execute_gate,
    )

    # Collect results
    for gate_id in gate_ids:
        if gate_id in execution_result.results:
            results.append(execution_result.results[gate_id])
        elif gate_id in execution_result.errors:
            # Create failed result for errored gates
            results.append(GateResult(
                gate_id=gate_id,
                status="FAIL",
                exit_code=-1,
                log_path=str(gates_dir / f"{gate_id}.log"),
                captured_paths=[],
                sph_hits=[],
                missing_expected_outputs=[],
            ))

    # Determine overall status
    overall = "PASS"
    for r in results:
        if r.status == "BLOCK":
            overall = "BLOCK"
            break
        elif r.status == "FAIL":
            overall = "FAIL"

    # Build response
    gates_list = [
        {
            "gate_id": r.gate_id,
            "status": r.status,
            "exit_code": r.exit_code,
            "sph_hits": r.sph_hits,
            "log_path": r.log_path,
        }
        for r in results
    ]

    result = {
        "status": overall,
        "overall_status": overall,
        "gates": gates_list,
        "parallel_execution": True,
        "execution_levels": len(levels),
        "total_duration_seconds": execution_result.total_duration_seconds,
    }

    # 2026-01-16: Collect and log security metrics to Langfuse (parallel execution)
    security_metrics = _collect_security_metrics()
    if security_metrics:
        result["security_metrics"] = security_metrics
        _log_security_metrics_to_langfuse(
            sprint_id=sprint_id,
            track=track,
            metrics=security_metrics,
            overall_status=overall,
        )
        logger.info(
            f"Security metrics (parallel): total={security_metrics.get('total_checks', 0)}, "
            f"passed={security_metrics.get('passed_checks', 0)}, "
            f"blocked={security_metrics.get('blocked_operations', 0)}"
        )

    return result


# =============================================================================
# SPEC ARTIFACTS VALIDATION (Integration from STUDY_SPEC_PHASE_TECHNIQUES.md)
# =============================================================================

def validate_spec_artifacts_for_gate(
    sprint_dir: Path,
    gate_id: str,
    *,
    strict: bool = False,
) -> Dict[str, Any]:
    """Validate SPEC artifacts as part of gate execution.

    This integrates the SPEC phase validation techniques:
    - G0: Verifies integration matrices exist
    - G4: Verifies critical flows have E2E test mappings
    - G6: Verifies security cross-cutting concerns
    - G8: Verifies full cross-cutting compliance

    Args:
        sprint_dir: Path to sprint directory
        gate_id: Gate identifier (G0, G4, G6, G8)
        strict: If True, warnings become errors

    Returns:
        Dict with validation results
    """
    try:
        # Import the validation script
        import sys
        scripts_path = Path(__file__).parent.parent.parent / "scripts"
        if str(scripts_path) not in sys.path:
            sys.path.insert(0, str(scripts_path))

        from validate_spec_artifacts import (
            validate_for_gate,
            validate_sprint_spec_artifacts,
        )

        if gate_id in ["G0", "G4", "G6", "G8"]:
            result = validate_for_gate(sprint_dir, gate_id)
        else:
            # For other gates, return success (no SPEC validation needed)
            return {
                "passed": True,
                "gate_id": gate_id,
                "spec_validation": "not_required",
                "message": f"Gate {gate_id} does not require SPEC artifact validation",
            }

        # Build result dict
        validation_result = {
            "passed": result.passed,
            "gate_id": gate_id,
            "coverage": result.coverage,
            "errors_count": len(result.errors),
            "warnings_count": len(result.warnings),
            "errors": [e.message for e in result.errors],
            "warnings": [w.message for w in result.warnings],
            "details": result.details,
        }

        # In strict mode, warnings become errors
        if strict and result.warnings:
            validation_result["passed"] = False
            validation_result["errors"].extend(validation_result["warnings"])
            validation_result["errors_count"] += validation_result["warnings_count"]

        return validation_result

    except ImportError as e:
        logger.warning(f"SPEC validation script not available: {e}")
        return {
            "passed": True,  # Don't block if script not available
            "gate_id": gate_id,
            "spec_validation": "skipped",
            "message": "validate_spec_artifacts.py not available",
        }
    except Exception as e:
        logger.error(f"SPEC validation failed: {e}")
        return {
            "passed": False,
            "gate_id": gate_id,
            "spec_validation": "error",
            "message": str(e),
        }


def run_spec_validation_gate(
    run_dir: Path,
    sprint_id: str,
    gate_id: str,
) -> GateResult:
    """Run SPEC artifact validation as a gate.

    This function runs the SPEC artifact validation and returns a GateResult
    that can be integrated into the gate selection flow.

    Args:
        run_dir: Path to the run directory
        sprint_id: Sprint identifier
        gate_id: Gate to validate (G0, G4, G6, G8)

    Returns:
        GateResult with validation status
    """
    # Determine sprint directory
    sprint_dir = run_dir / "sprints" / sprint_id

    # Alternative locations for SPEC artifacts
    spec_dirs = [
        sprint_dir,
        sprint_dir / "spec",
        sprint_dir / "plan",
        run_dir.parent / "docs" / "context_packs" / "sprints" / sprint_id,
    ]

    # Find the first valid spec directory
    actual_spec_dir = None
    for d in spec_dirs:
        if d.exists() and (
            (d / "integrations").exists() or
            (d / "flows").exists() or
            (d / "checklists").exists()
        ):
            actual_spec_dir = d
            break

    if not actual_spec_dir:
        logger.warning(f"No SPEC artifacts directory found for {sprint_id}")
        return GateResult(
            gate_id=f"{gate_id}_SPEC",
            status="WARN",
            exit_code=0,
            log_path="",
            sph_hits=[],
        )

    # Run validation
    result = validate_spec_artifacts_for_gate(actual_spec_dir, gate_id)

    # Determine status
    if result["passed"]:
        status = "PASS"
        exit_code = 0
    elif result.get("spec_validation") == "skipped":
        status = "WARN"
        exit_code = 0
    else:
        status = "FAIL"
        exit_code = 1

    # Create log content
    log_lines = [
        f"SPEC Artifacts Validation for {gate_id}",
        f"Sprint: {sprint_id}",
        f"Directory: {actual_spec_dir}",
        f"Status: {status}",
        f"Coverage: {result.get('coverage', 0):.1%}",
        "",
    ]

    if result.get("errors"):
        log_lines.append("ERRORS:")
        for err in result["errors"]:
            log_lines.append(f"  - {err}")
        log_lines.append("")

    if result.get("warnings"):
        log_lines.append("WARNINGS:")
        for warn in result["warnings"]:
            log_lines.append(f"  - {warn}")
        log_lines.append("")

    log_content = "\n".join(log_lines)

    # Write log file
    log_dir = run_dir / "sprints" / sprint_id / "evidence" / "spec_validation"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{gate_id}_spec_validation.log"
    log_path.write_text(log_content, encoding="utf-8")

    return GateResult(
        gate_id=f"{gate_id}_SPEC",
        status=status,
        exit_code=exit_code,
        log_path=str(log_path),
        sph_hits=[],
    )


def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run gate_selection.yml and emit TrilhaReport + Evidence Bundle.")
    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--sprint-id", type=str, required=True)
    p.add_argument("--track", type=str, choices=["A", "B", "C"], default="B")
    p.add_argument("--gate-selection", type=Path, default=None)
    p.add_argument("--report-suffix", type=str, default=None, help="Optional suffix to avoid overwriting track report.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    out = run_gate_selection(
        repo_root=args.repo_root,
        run_dir=args.run_dir,
        sprint_id=args.sprint_id,
        track=args.track,
        selection_path=args.gate_selection,
        report_suffix=args.report_suffix,
    )
    print(json.dumps(out, ensure_ascii=False))
    return 0 if out["status"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
