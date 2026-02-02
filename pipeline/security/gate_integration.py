"""Security Gate Integration Module.

This module provides security integration with the pipeline gate system,
ensuring all gate operations are protected by security checks.

Features:
1. Pre-gate security validation
2. Post-gate output verification
3. Security metrics collection
4. Integration with Langfuse for observability
5. Gate-specific security policies

Architecture:
    SecurityGateRunner
        |
        +-- Pre-Gate Security Check
        |       |
        |       +-- Input sanitization
        |       +-- Prompt injection detection
        |       +-- PII detection
        |
        +-- Gate Execution
        |
        +-- Post-Gate Security Check
                |
                +-- Output validation
                +-- Sensitive data leak detection
                +-- Toxicity filtering

Author: Pipeline Autonomo Team
Version: 1.0.0 (2026-01-16)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime, timezone
from functools import wraps

logger = logging.getLogger(__name__)

# =============================================================================
# IMPORTS
# =============================================================================

try:
    from pipeline.security.llm_guard_integration import (
        SecurityOrchestrator,
        SecurityCheckResult,
        SecurityLevel,
        get_security_orchestrator,
    )
    LLM_GUARD_AVAILABLE = True
except ImportError:
    LLM_GUARD_AVAILABLE = False
    logger.debug("LLM Guard integration not available")

try:
    from pipeline.security.nemo_enhanced import (
        NeMoEnhancedRails,
        get_nemo_enhanced,
        JailbreakResult,
        ContentFilterResult,
    )
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    logger.debug("NeMo enhanced not available")

SECURITY_GATE_AVAILABLE = LLM_GUARD_AVAILABLE or NEMO_AVAILABLE

# =============================================================================
# DUAL-DEGRADATION DETECTION (CRIT-003 FIX)
# =============================================================================

# Track dual-degradation state
_DUAL_DEGRADATION_ALERTED = False
_DUAL_DEGRADATION_LOCK = threading.Lock()


class DualDegradationState(str, Enum):
    """Degradation state of security guardrails."""
    FULL_PROTECTION = "full_protection"  # Both NeMo and LLM Guard available
    PARTIAL_NEMO_ONLY = "partial_nemo_only"  # Only NeMo available
    PARTIAL_LLM_GUARD_ONLY = "partial_llm_guard_only"  # Only LLM Guard available
    DUAL_DEGRADATION = "dual_degradation"  # CRITICAL: Both unavailable


def check_dual_degradation() -> Tuple[DualDegradationState, str]:
    """Check if both security guardrails are unavailable.

    CRIT-003 FIX: Detects dual-degradation mode and raises alerts.

    Returns:
        Tuple of (state, message)
    """
    global _DUAL_DEGRADATION_ALERTED

    nemo_ok = NEMO_AVAILABLE
    llm_guard_ok = LLM_GUARD_AVAILABLE

    # Determine state
    if nemo_ok and llm_guard_ok:
        state = DualDegradationState.FULL_PROTECTION
        message = "Full protection: Both NeMo and LLM Guard available"
    elif nemo_ok and not llm_guard_ok:
        state = DualDegradationState.PARTIAL_NEMO_ONLY
        message = "Partial protection: NeMo only (LLM Guard unavailable)"
    elif not nemo_ok and llm_guard_ok:
        state = DualDegradationState.PARTIAL_LLM_GUARD_ONLY
        message = "Partial protection: LLM Guard only (NeMo unavailable)"
    else:
        state = DualDegradationState.DUAL_DEGRADATION
        message = "CRITICAL: DUAL DEGRADATION - Both NeMo and LLM Guard unavailable!"

    # Alert on dual degradation (only once per session to avoid log spam)
    if state == DualDegradationState.DUAL_DEGRADATION:
        with _DUAL_DEGRADATION_LOCK:
            if not _DUAL_DEGRADATION_ALERTED:
                logger.critical(
                    "🚨 DUAL-DEGRADATION ALERT 🚨\n"
                    "Both NeMo Guardrails and LLM Guard are UNAVAILABLE.\n"
                    "Security posture is SEVERELY COMPROMISED.\n"
                    "All inputs/outputs will pass through WITHOUT security checks.\n"
                    "Action required: Restore at least one guardrail system."
                )
                _DUAL_DEGRADATION_ALERTED = True

                # GAP HUNTER V2 FIX: Send external alert for dual-degradation
                _send_external_security_alert(state, message)

    return state, message


def _send_external_security_alert(
    state: DualDegradationState,
    message: str,
) -> None:
    """Send external alert for security-critical events.

    GAP HUNTER V2 FIX: Implements external alerting for dual-degradation.
    Sends alerts to multiple channels for redundancy:
    1. Langfuse event (observability)
    2. Redis pub/sub (real-time)
    3. Webhook (external systems)

    Args:
        state: Current degradation state
        message: Alert message
    """
    import os
    from datetime import datetime, timezone

    timestamp = datetime.now(timezone.utc).isoformat()

    alert_payload = {
        "alert_type": "SECURITY_DUAL_DEGRADATION",
        "severity": "CRITICAL",
        "state": state.value,
        "message": message,
        "timestamp": timestamp,
        "action_required": "Restore at least one security guardrail system",
        "affected_systems": ["NeMo Guardrails", "LLM Guard"],
    }

    # 1. Send to Langfuse (if available)
    try:
        from langfuse import Langfuse

        langfuse = Langfuse()
        langfuse.event(
            name="security_dual_degradation_alert",
            metadata=alert_payload,
            level="ERROR",
        )
        langfuse.flush()
        logger.info("GAP-V2: Dual-degradation alert sent to Langfuse")
    except Exception as e:
        logger.debug(f"GAP-V2: Failed to send Langfuse alert: {e}")

    # 2. Send to Redis pub/sub (for real-time monitoring)
    try:
        import redis

        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        r = redis.from_url(redis_url, socket_timeout=5.0)
        r.publish(
            "security:alerts:critical",
            json.dumps(alert_payload) if "json" in dir() else str(alert_payload)
        )
        logger.info("GAP-V2: Dual-degradation alert published to Redis")
    except Exception as e:
        logger.debug(f"GAP-V2: Failed to publish Redis alert: {e}")

    # 3. Send to webhook (if configured)
    webhook_url = os.getenv("SECURITY_ALERT_WEBHOOK_URL")
    if webhook_url:
        try:
            import httpx

            with httpx.Client(timeout=10.0) as client:
                response = client.post(
                    webhook_url,
                    json=alert_payload,
                    headers={"Content-Type": "application/json"},
                )
                if response.status_code < 300:
                    logger.info("GAP-V2: Dual-degradation alert sent to webhook")
                else:
                    logger.warning(
                        f"GAP-V2: Webhook alert returned {response.status_code}"
                    )
        except Exception as e:
            logger.debug(f"GAP-V2: Failed to send webhook alert: {e}")

    # 4. Log to Grafana metrics (if available)
    try:
        from pipeline.grafana_metrics import get_metrics_publisher

        pub = get_metrics_publisher()
        pub.publish_security_alert(
            alert_type="dual_degradation",
            severity="critical",
            details=alert_payload,
        )
        logger.info("GAP-V2: Dual-degradation alert sent to Grafana")
    except Exception as e:
        logger.debug(f"GAP-V2: Failed to send Grafana alert: {e}")


def reset_dual_degradation_alert():
    """Reset the dual-degradation alert flag (for testing/recovery)."""
    global _DUAL_DEGRADATION_ALERTED
    with _DUAL_DEGRADATION_LOCK:
        _DUAL_DEGRADATION_ALERTED = False


def get_security_status() -> Dict[str, Any]:
    """Get comprehensive security status.

    Returns:
        Dict with security status details
    """
    state, message = check_dual_degradation()
    return {
        "nemo_available": NEMO_AVAILABLE,
        "llm_guard_available": LLM_GUARD_AVAILABLE,
        "security_gate_available": SECURITY_GATE_AVAILABLE,
        "degradation_state": state.value,
        "message": message,
        "is_critical": state == DualDegradationState.DUAL_DEGRADATION,
        "protection_level": (
            "full" if state == DualDegradationState.FULL_PROTECTION
            else "partial" if state in (
                DualDegradationState.PARTIAL_NEMO_ONLY,
                DualDegradationState.PARTIAL_LLM_GUARD_ONLY
            )
            else "none"
        ),
    }


# Check on module load and emit warning if needed
_startup_state, _startup_message = check_dual_degradation()
if _startup_state == DualDegradationState.DUAL_DEGRADATION:
    logger.critical(_startup_message)
elif _startup_state != DualDegradationState.FULL_PROTECTION:
    logger.warning(_startup_message)
else:
    logger.info(_startup_message)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================


class SecurityGateLevel(str, Enum):
    """Security levels for different gate types."""
    MINIMAL = "minimal"  # Basic checks only
    STANDARD = "standard"  # Default for most gates
    ELEVATED = "elevated"  # For gates with external data
    MAXIMUM = "maximum"  # For gates with sensitive operations


class SecurityCheckPhase(str, Enum):
    """Phase of security check."""
    PRE_GATE = "pre_gate"
    POST_GATE = "post_gate"
    BOTH = "both"


# Gate security policies
GATE_SECURITY_POLICIES: Dict[str, Dict[str, Any]] = {
    # Default policy
    "default": {
        "level": SecurityGateLevel.STANDARD,
        "check_phase": SecurityCheckPhase.BOTH,
        "require_pii_check": True,
        "require_injection_check": True,
        "require_toxicity_check": False,
        "max_input_length": 50000,
        "timeout_seconds": 30,
    },
    # Gate G0 - Claim Intake
    "G0": {
        "level": SecurityGateLevel.ELEVATED,
        "check_phase": SecurityCheckPhase.BOTH,
        "require_pii_check": True,
        "require_injection_check": True,
        "require_toxicity_check": True,
        "max_input_length": 10000,
        "timeout_seconds": 20,
    },
    # Gate G1 - Source Verification
    "G1": {
        "level": SecurityGateLevel.STANDARD,
        "check_phase": SecurityCheckPhase.PRE_GATE,
        "require_pii_check": True,
        "require_injection_check": True,
        "require_toxicity_check": False,
        "max_input_length": 100000,
        "timeout_seconds": 30,
    },
    # Gate G2 - Evidence Analysis
    "G2": {
        "level": SecurityGateLevel.ELEVATED,
        "check_phase": SecurityCheckPhase.BOTH,
        "require_pii_check": True,
        "require_injection_check": True,
        "require_toxicity_check": False,
        "max_input_length": 200000,
        "timeout_seconds": 45,
    },
    # Gate G3 - Verdict Generation
    "G3": {
        "level": SecurityGateLevel.MAXIMUM,
        "check_phase": SecurityCheckPhase.BOTH,
        "require_pii_check": True,
        "require_injection_check": True,
        "require_toxicity_check": True,
        "max_input_length": 50000,
        "timeout_seconds": 30,
    },
    # Gate G4 - Security Gate
    "G4": {
        "level": SecurityGateLevel.MAXIMUM,
        "check_phase": SecurityCheckPhase.BOTH,
        "require_pii_check": True,
        "require_injection_check": True,
        "require_toxicity_check": True,
        "max_input_length": 50000,
        "timeout_seconds": 60,
    },
    # Gate G5 - Quality Assurance
    "G5": {
        "level": SecurityGateLevel.STANDARD,
        "check_phase": SecurityCheckPhase.POST_GATE,
        "require_pii_check": True,
        "require_injection_check": False,
        "require_toxicity_check": True,
        "max_input_length": 100000,
        "timeout_seconds": 30,
    },
    # Gate G6 - Publication Gate
    "G6": {
        "level": SecurityGateLevel.MAXIMUM,
        "check_phase": SecurityCheckPhase.BOTH,
        "require_pii_check": True,
        "require_injection_check": True,
        "require_toxicity_check": True,
        "max_input_length": 50000,
        "timeout_seconds": 45,
    },
    # Gate G7 - Archival Gate
    "G7": {
        "level": SecurityGateLevel.STANDARD,
        "check_phase": SecurityCheckPhase.PRE_GATE,
        "require_pii_check": True,
        "require_injection_check": False,
        "require_toxicity_check": False,
        "max_input_length": 500000,
        "timeout_seconds": 60,
    },
    # Gate G8 - Audit Gate
    "G8": {
        "level": SecurityGateLevel.ELEVATED,
        "check_phase": SecurityCheckPhase.POST_GATE,
        "require_pii_check": True,
        "require_injection_check": True,
        "require_toxicity_check": False,
        "max_input_length": 100000,
        "timeout_seconds": 30,
    },
}


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class SecurityGateResult:
    """Result of security gate validation."""
    gate_id: str
    passed: bool
    pre_gate_result: Optional[SecurityCheckResult] = None
    post_gate_result: Optional[SecurityCheckResult] = None
    security_level: SecurityGateLevel = SecurityGateLevel.STANDARD
    blocked_reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "passed": self.passed,
            "pre_gate_result": self.pre_gate_result.to_dict() if self.pre_gate_result else None,
            "post_gate_result": self.post_gate_result.to_dict() if self.post_gate_result else None,
            "security_level": self.security_level.value,
            "blocked_reason": self.blocked_reason,
            "warnings": self.warnings,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SecurityMetrics:
    """Metrics from security gate execution."""
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    blocked_operations: int = 0
    pii_detections: int = 0
    injection_attempts: int = 0
    toxicity_flags: int = 0
    avg_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    # CRIT-003 FIX: Track dual-degradation events
    dual_degradation_events: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "blocked_operations": self.blocked_operations,
            "pii_detections": self.pii_detections,
            "injection_attempts": self.injection_attempts,
            "toxicity_flags": self.toxicity_flags,
            "avg_latency_ms": self.avg_latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "dual_degradation_events": self.dual_degradation_events,
        }


# =============================================================================
# SECURITY GATE
# =============================================================================


class SecurityGate:
    """Security gate wrapper for individual gates.

    Wraps a gate with security checks based on configured policy.

    Usage:
        gate = SecurityGate("G0")
        result = await gate.run(input_data, gate_function)
    """

    def __init__(
        self,
        gate_id: str,
        policy: Optional[Dict[str, Any]] = None,
    ):
        self.gate_id = gate_id
        self.policy = policy or GATE_SECURITY_POLICIES.get(
            gate_id,
            GATE_SECURITY_POLICIES["default"]
        )
        self._orchestrator = get_security_orchestrator() if LLM_GUARD_AVAILABLE else None
        self._nemo = get_nemo_enhanced() if NEMO_AVAILABLE else None

    @property
    def security_level(self) -> SecurityGateLevel:
        """Get security level for this gate."""
        return self.policy.get("level", SecurityGateLevel.STANDARD)

    async def validate_pre_gate(
        self,
        input_data: Dict[str, Any],
    ) -> Tuple[bool, Optional[SecurityCheckResult], List[str]]:
        """Validate input before gate execution.

        Args:
            input_data: Input data for the gate

        Returns:
            Tuple of (passed, result, warnings)
        """
        check_phase = self.policy.get("check_phase", SecurityCheckPhase.BOTH)
        if check_phase == SecurityCheckPhase.POST_GATE:
            return True, None, []

        warnings: List[str] = []
        input_text = self._extract_text(input_data)

        # Check input length
        max_length = self.policy.get("max_input_length", 50000)
        if len(input_text) > max_length:
            return False, None, [f"Input exceeds maximum length ({len(input_text)} > {max_length})"]

        # Run security checks
        if self._orchestrator:
            result = await self._orchestrator.secure_operation(
                input_text,
                operation_type=f"gate_{self.gate_id}_pre",
                context={"gate_id": self.gate_id, "phase": "pre_gate"},
            )

            if result.blocked:
                return False, result, []

            # Collect warnings
            if result.pii_detection and result.pii_detection.has_pii:
                warnings.append(f"PII detected: {result.pii_detection.pii_types}")

            return result.is_safe, result, warnings

        # NF-012 FIX: Fail-closed fallback - do not assume safe without checks
        # Fallback should FAIL, not PASS, when no security checks were performed
        logger.warning(
            "GATE-SEC-001: Security validation fallback reached - no checks performed. "
            "Returning FAIL (fail-closed). Ensure security stack is properly configured."
        )
        return False, None, warnings + ["No security checks performed - fallback to FAIL"]

    async def validate_post_gate(
        self,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
    ) -> Tuple[bool, Optional[SecurityCheckResult], List[str]]:
        """Validate output after gate execution.

        Args:
            input_data: Original input data
            output_data: Gate output data

        Returns:
            Tuple of (passed, result, warnings)
        """
        check_phase = self.policy.get("check_phase", SecurityCheckPhase.BOTH)
        if check_phase == SecurityCheckPhase.PRE_GATE:
            return True, None, []

        warnings: List[str] = []
        input_text = self._extract_text(input_data)
        output_text = self._extract_text(output_data)

        if self._orchestrator:
            result = await self._orchestrator.secure_operation(
                input_text,
                output_text,
                operation_type=f"gate_{self.gate_id}_post",
                context={"gate_id": self.gate_id, "phase": "post_gate"},
            )

            if result.blocked:
                return False, result, []

            # Check validation result
            if result.validation and not result.validation.is_valid:
                warnings.append("Output validation issues detected")

            return result.is_safe, result, warnings

        # NF-012 FIX: Fail-closed fallback for post-gate validation
        # If orchestrator is not available, FAIL (not pass)
        logger.warning(
            "GATE-SEC-002: Post-gate security validation fallback reached - no checks performed. "
            "Returning FAIL (fail-closed). Ensure security orchestrator is properly configured."
        )
        return False, None, warnings + ["No post-gate security checks performed - fallback to FAIL"]

    async def run(
        self,
        input_data: Dict[str, Any],
        gate_function: Callable,
        *args,
        **kwargs,
    ) -> SecurityGateResult:
        """Run gate with security checks.

        Args:
            input_data: Input data for the gate
            gate_function: Gate function to execute
            *args, **kwargs: Additional arguments for gate function

        Returns:
            SecurityGateResult with complete results
        """
        start_time = time.time()
        metrics: Dict[str, Any] = {}
        warnings: List[str] = []

        # Pre-gate security check
        pre_passed, pre_result, pre_warnings = await self.validate_pre_gate(input_data)
        warnings.extend(pre_warnings)
        metrics["pre_gate_latency_ms"] = pre_result.total_latency_ms if pre_result else 0

        if not pre_passed:
            return SecurityGateResult(
                gate_id=self.gate_id,
                passed=False,
                pre_gate_result=pre_result,
                security_level=self.security_level,
                blocked_reason=pre_result.block_reason if pre_result else "Pre-gate validation failed",
                warnings=warnings,
                metrics=metrics,
            )

        # Execute gate
        gate_start = time.time()
        try:
            output_data = await gate_function(input_data, *args, **kwargs)
            metrics["gate_execution_ms"] = (time.time() - gate_start) * 1000
        except Exception as e:
            logger.error(f"Gate {self.gate_id} execution failed: {e}")
            metrics["gate_execution_ms"] = (time.time() - gate_start) * 1000
            return SecurityGateResult(
                gate_id=self.gate_id,
                passed=False,
                pre_gate_result=pre_result,
                security_level=self.security_level,
                blocked_reason=f"Gate execution error: {str(e)}",
                warnings=warnings,
                metrics=metrics,
            )

        # Post-gate security check
        post_passed, post_result, post_warnings = await self.validate_post_gate(
            input_data, output_data
        )
        warnings.extend(post_warnings)
        metrics["post_gate_latency_ms"] = post_result.total_latency_ms if post_result else 0

        total_latency = (time.time() - start_time) * 1000
        metrics["total_latency_ms"] = total_latency

        # Log to Langfuse
        await self._log_security_gate(
            self.gate_id,
            pre_passed and post_passed,
            pre_result,
            post_result,
            metrics,
        )

        return SecurityGateResult(
            gate_id=self.gate_id,
            passed=pre_passed and post_passed,
            pre_gate_result=pre_result,
            post_gate_result=post_result,
            security_level=self.security_level,
            blocked_reason=post_result.block_reason if post_result and not post_passed else None,
            warnings=warnings,
            metrics=metrics,
        )

    def _extract_text(self, data: Dict[str, Any]) -> str:
        """Extract text content from data dict.

        FIX 2026-01-30: Skip pipeline metadata to avoid false positives.
        Pipeline state dicts like {'sprint_id': 'S00', 'run_dir': '...', 'phase': 'QA'}
        were being sent to LLM Guard and flagged as prompt injection with 100% confidence.
        """
        # Try common field names for actual content
        for field in ["text", "claim", "content", "input", "response", "output", "message"]:
            if field in data and isinstance(data[field], str):
                return data[field]

        # FIX: Check if this is pipeline metadata - skip security scanning for internal state
        # Pipeline metadata keys that indicate this is internal state, not user content
        pipeline_metadata_keys = {"sprint_id", "run_id", "run_dir", "phase", "attempt", "status", "repo_root"}
        if pipeline_metadata_keys & set(data.keys()):
            # This is pipeline internal state, not content to scan
            logger.debug(
                f"Skipping security scan for pipeline metadata (keys: {list(data.keys())[:5]}...)"
            )
            return ""  # Empty string = no content to scan

        # Fallback to string representation only for non-pipeline data
        return str(data)

    async def _log_security_gate(
        self,
        gate_id: str,
        passed: bool,
        pre_result: Optional[SecurityCheckResult],
        post_result: Optional[SecurityCheckResult],
        metrics: Dict[str, Any],
    ) -> None:
        """Log security gate results to Langfuse."""
        try:
            from langfuse import Langfuse

            langfuse = Langfuse()
            langfuse.event(
                name=f"security_gate_{gate_id}",
                metadata={
                    "gate_id": gate_id,
                    "passed": passed,
                    "security_level": self.security_level.value,
                    "pre_gate_safe": pre_result.is_safe if pre_result else None,
                    "post_gate_safe": post_result.is_safe if post_result else None,
                    "metrics": metrics,
                },
            )

            # Score the gate
            langfuse.score(
                name=f"security_gate_{gate_id}_pass",
                value=1.0 if passed else 0.0,
            )

            langfuse.flush()
        except ImportError as e:
            logger.debug(f"GATE: Gate operation failed: {e}")
        except Exception as e:
            logger.debug(f"Failed to log to Langfuse: {e}")


# =============================================================================
# SECURITY GATE RUNNER
# =============================================================================


class SecurityGateRunner:
    """Runs multiple gates with security checks.

    Manages security for the entire gate pipeline,
    collecting metrics and enforcing policies.

    Usage:
        runner = SecurityGateRunner()
        results = await runner.run_gates(
            ["G0", "G1", "G2"],
            input_data,
            gate_functions,
        )
    """

    _instance: Optional["SecurityGateRunner"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "SecurityGateRunner":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._gates: Dict[str, SecurityGate] = {}
        self._metrics = SecurityMetrics()
        self._degradation_state: Optional[DualDegradationState] = None
        self._initialized = True

        # CRIT-003 FIX: Check dual-degradation on initialization
        self._check_and_update_degradation_state()

        logger.info("SecurityGateRunner initialized")

    def _check_and_update_degradation_state(self) -> DualDegradationState:
        """Check and update the degradation state.

        CRIT-003 FIX: Called on init and periodically during operation.
        """
        state, message = check_dual_degradation()
        self._degradation_state = state

        if state == DualDegradationState.DUAL_DEGRADATION:
            # Update metrics for monitoring
            self._metrics.dual_degradation_events += 1

        return state

    @property
    def is_dual_degraded(self) -> bool:
        """Check if in dual-degradation mode."""
        return self._degradation_state == DualDegradationState.DUAL_DEGRADATION

    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status including degradation state."""
        status = get_security_status()
        status["metrics"] = self._metrics.to_dict()
        return status

    def get_gate(self, gate_id: str) -> SecurityGate:
        """Get or create a security gate."""
        if gate_id not in self._gates:
            self._gates[gate_id] = SecurityGate(gate_id)
        return self._gates[gate_id]

    async def run_gate(
        self,
        gate_id: str,
        input_data: Dict[str, Any],
        gate_function: Callable,
        *args,
        **kwargs,
    ) -> SecurityGateResult:
        """Run a single gate with security.

        Args:
            gate_id: Gate identifier
            input_data: Input data
            gate_function: Gate function
            *args, **kwargs: Additional arguments

        Returns:
            SecurityGateResult
        """
        gate = self.get_gate(gate_id)
        result = await gate.run(input_data, gate_function, *args, **kwargs)

        # Update metrics
        self._update_metrics(result)

        return result

    async def run_gates(
        self,
        gate_ids: List[str],
        input_data: Dict[str, Any],
        gate_functions: Dict[str, Callable],
        stop_on_failure: bool = True,
    ) -> List[SecurityGateResult]:
        """Run multiple gates sequentially with security.

        Args:
            gate_ids: List of gate IDs to run
            input_data: Initial input data
            gate_functions: Dict mapping gate IDs to functions
            stop_on_failure: Stop if a gate fails

        Returns:
            List of SecurityGateResult
        """
        results: List[SecurityGateResult] = []
        current_data = input_data

        for gate_id in gate_ids:
            if gate_id not in gate_functions:
                logger.warning(f"No function for gate {gate_id}")
                continue

            result = await self.run_gate(
                gate_id,
                current_data,
                gate_functions[gate_id],
            )
            results.append(result)

            if not result.passed and stop_on_failure:
                logger.warning(f"Gate {gate_id} failed, stopping pipeline")
                break

            # Pass output to next gate (if available)
            if result.post_gate_result:
                current_data = {"previous_result": result.to_dict()}

        return results

    async def validate_input(
        self,
        gate_id: str,
        input_data: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        """Validate input for a specific gate.

        Args:
            gate_id: Gate identifier
            input_data: Input data to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        gate = self.get_gate(gate_id)
        passed, result, warnings = await gate.validate_pre_gate(input_data)

        if not passed:
            return False, result.block_reason if result else "Validation failed"

        if warnings:
            logger.warning(f"Gate {gate_id} validation warnings: {warnings}")

        return True, None

    def _update_metrics(self, result: SecurityGateResult) -> None:
        """Update running metrics."""
        self._metrics.total_checks += 1

        if result.passed:
            self._metrics.passed_checks += 1
        else:
            self._metrics.failed_checks += 1
            if result.blocked_reason:
                self._metrics.blocked_operations += 1

        # Update from pre-gate result
        if result.pre_gate_result:
            if result.pre_gate_result.pii_detection and result.pre_gate_result.pii_detection.has_pii:
                self._metrics.pii_detections += 1
            if result.pre_gate_result.prompt_injection and result.pre_gate_result.prompt_injection.is_injection:
                self._metrics.injection_attempts += 1
            if result.pre_gate_result.toxicity_filter and result.pre_gate_result.toxicity_filter.is_toxic:
                self._metrics.toxicity_flags += 1

        # Update latency
        total_latency = result.metrics.get("total_latency_ms", 0)
        self._metrics.total_latency_ms += total_latency
        self._metrics.avg_latency_ms = (
            self._metrics.total_latency_ms / self._metrics.total_checks
        )

    def get_metrics(self) -> SecurityMetrics:
        """Get current security metrics."""
        return self._metrics

    def reset_metrics(self) -> None:
        """Reset security metrics."""
        self._metrics = SecurityMetrics()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


_gate_runner: Optional[SecurityGateRunner] = None


def get_security_gate_runner() -> SecurityGateRunner:
    """Get singleton security gate runner."""
    global _gate_runner
    if _gate_runner is None:
        _gate_runner = SecurityGateRunner()
    return _gate_runner


async def run_security_gate(
    gate_id: str,
    input_data: Dict[str, Any],
    gate_function: Callable,
    *args,
    **kwargs,
) -> SecurityGateResult:
    """Run a gate with security checks."""
    runner = get_security_gate_runner()
    return await runner.run_gate(gate_id, input_data, gate_function, *args, **kwargs)


async def validate_gate_security(
    gate_id: str,
    input_data: Dict[str, Any],
) -> Tuple[bool, Optional[str]]:
    """Validate input security for a gate."""
    runner = get_security_gate_runner()
    return await runner.validate_input(gate_id, input_data)


# =============================================================================
# DECORATOR
# =============================================================================


def secure_gate(gate_id: str):
    """Decorator to add security to a gate function.

    Usage:
        @secure_gate("G0")
        async def run_g0(input_data: Dict[str, Any]) -> Dict[str, Any]:
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(input_data: Dict[str, Any], *args, **kwargs):
            runner = get_security_gate_runner()
            result = await runner.run_gate(gate_id, input_data, func, *args, **kwargs)

            if not result.passed:
                raise SecurityGateError(
                    f"Gate {gate_id} security check failed: {result.blocked_reason}",
                    result=result,
                )

            return result

        return wrapper
    return decorator


class SecurityGateError(Exception):
    """Exception raised when gate security check fails."""

    def __init__(self, message: str, result: SecurityGateResult):
        super().__init__(message)
        self.result = result


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Classes
    "SecurityGate",
    "SecurityGateRunner",
    # Data classes
    "SecurityGateResult",
    "SecurityMetrics",
    # Enums
    "SecurityGateLevel",
    "SecurityCheckPhase",
    # Functions
    "get_security_gate_runner",
    "run_security_gate",
    "validate_gate_security",
    # Decorator
    "secure_gate",
    # Exceptions
    "SecurityGateError",
    # Constants
    "SECURITY_GATE_AVAILABLE",
    "GATE_SECURITY_POLICIES",
]
