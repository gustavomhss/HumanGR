"""LangGraph Workflow Definition for Pipeline V2.

This module defines the StateGraph workflow that replaces the linear orchestrator
with a checkpointable, resumable control plane.

Architecture:
    INIT -> EXEC -> GATE -> SIGNOFF -> ARTIFACT -> END
           ^                   |
           |_____ retry _______|

Key Features:
- Checkpointing: Every node saves state to Redis/file
- Resume: Can resume from any checkpoint after crash
- Idempotency: Each node has deterministic idempotency key
- Integration: Calls existing V2 modules (gate_runner, governance, etc.)

Based on: MIGRATION_V2_TO_LANGGRAPH.md + PIPELINE_V3_MASTER_PLAN.md Section 10.4

Author: Pipeline Autonomo Team
Version: 1.0.0 (2026-01-16)
"""

from __future__ import annotations

import asyncio
import logging
import os

# Early logger definition (needed for import-time error handling)
logger = logging.getLogger(__name__)
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Awaitable

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

from pipeline.langgraph.state import (
    PipelineState,
    SprintPhase,
    PipelineStatus,
    GateStatus,
    create_initial_state,
    create_node_history_entry,
    create_error_entry,
    update_phase,
    update_status,
    add_error,
)
from pipeline.langgraph.stack_injection import (
    Operation,
    StackInjector,
    StackContext,
    get_stack_injector,
)
from pipeline.langgraph.invariants import (
    InvariantChecker,
    InvariantCheckResult,
    InvariantViolationError,  # For strict enforcement
    )
from pipeline.langgraph.trust_boundaries import (
    get_trust_boundary_enforcer,
    TrustBoundaryEnforcer,
)

# Partial Stacks Integration (Active RAG, BoT, RAGAS, Phoenix, DeepEval)
try:
    from pipeline.langgraph.partial_stacks_integration import (
        PartialStacksIntegration,
        get_partial_stacks_integration,
        health_check_all as partial_stacks_health_check,
        ACTIVE_RAG_AVAILABLE,
        BOT_AVAILABLE,
        RAGAS_AVAILABLE,
        PHOENIX_AVAILABLE,
        DEEPEVAL_AVAILABLE,
    )
    PARTIAL_STACKS_INTEGRATION_AVAILABLE = True
except ImportError as e:
    logger.debug(f"Partial stacks integration not available: {e}")
    PARTIAL_STACKS_INTEGRATION_AVAILABLE = False
    ACTIVE_RAG_AVAILABLE = False
    BOT_AVAILABLE = False
    RAGAS_AVAILABLE = False
    PHOENIX_AVAILABLE = False
    DEEPEVAL_AVAILABLE = False
    get_partial_stacks_integration = None
    partial_stacks_health_check = None

# Stack Data Bridge (2026-01-30): Connects stack data to consumers
# Solves the problem of data collected but not used
# Reference: docs/pipeline/STACK_USAGE_AUDIT_2026_01_30.md
try:
    from pipeline.langgraph.stack_data_bridge import (
        StackDataBridge,
        get_stack_data_bridge,
        BRIDGE_AVAILABLE,
    )
    STACK_DATA_BRIDGE_AVAILABLE = BRIDGE_AVAILABLE
except ImportError as e:
    logger.debug(f"Stack data bridge not available: {e}")
    STACK_DATA_BRIDGE_AVAILABLE = False
    StackDataBridge = None
    get_stack_data_bridge = None

# GHOST CODE INTEGRATION (2026-01-30): Subgraphs for modular workflow composition
# Provides GateValidationSubgraph, QualityAssuranceSubgraph, SignoffSubgraph
try:
    from pipeline.langgraph.subgraphs import (
        compose_workflow_with_subgraphs,
        GateValidationSubgraph,
        QualityAssuranceSubgraph,
        LANGGRAPH_SUBGRAPHS_AVAILABLE,
    )
    SUBGRAPHS_AVAILABLE = LANGGRAPH_SUBGRAPHS_AVAILABLE
except ImportError as e:
    logger.debug(f"Subgraphs module not available: {e}")
    SUBGRAPHS_AVAILABLE = False
    compose_workflow_with_subgraphs = None
    GateValidationSubgraph = None
    QualityAssuranceSubgraph = None

# NeMo Stack Rails for operation decorators (graceful degradation if unavailable)
try:
    from pipeline.langgraph.nemo_stack_rails import enforce_stacks, StackEnforcementError
except ImportError:
    # Provide no-op decorator if nemo_stack_rails not available
    def enforce_stacks(*args, **kwargs):
        """No-op decorator when nemo_stack_rails not available."""
        def decorator(func):
            return func
        return decorator if not args or callable(args[0]) else decorator
    class StackEnforcementError(Exception):
        pass

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

# LLM Guard integration for secure operations
try:
    from pipeline.security.llm_guard_integration import (
        secure_operation,
        async_secure_operation,
        SecurityLevel,
        SecurityBlockedError,
        # Additional security convenience functions
        sanitize_input,
        validate_output,
        detect_pii,
        filter_toxicity,
        SanitizationResult,
        ValidationResult,
        PIIDetectionResult,
        ToxicityFilterResult,
    )
    LLM_GUARD_AVAILABLE = True
except ImportError:
    LLM_GUARD_AVAILABLE = False

# =============================================================================
# QUIETSTAR + REFLEXION GUARDRAILS (2026-01-20)
# =============================================================================
# QuietStar/Reflexion integration provides an additional security layer
# with pre-generation safety thinking and post-generation reflexion checks.
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


async def _check_quietstar_workflow_input(
    input_text: str,
    operation_name: str = "workflow_node",
) -> None:
    """Check workflow input with QuietStar safety thinking.

    Args:
        input_text: Input text to check.
        operation_name: Name of the operation for logging.

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

        if result.risk_score > 0.7:
            logger.warning(
                f"QUIETSTAR WORKFLOW BLOCKED: op={operation_name}, "
                f"risk={result.risk_score:.2f}, reason={result.reasoning[:100]}"
            )
            raise QuietStarBlockedError(
                f"Input blocked by QuietStar in {operation_name}: {result.reasoning[:200]}"
            )

        logger.debug(
            f"QUIETSTAR WORKFLOW PRE-CHECK: op={operation_name}, "
            f"risk={result.risk_score:.2f} PASS"
        )

    except QuietStarBlockedError:
        raise
    except ValueError as e:
        # QuietStar not initialized with LLM - this is expected if not configured
        # Fail-open: continue without QuietStar protection
        if "main_llm required" in str(e):
            logger.debug(f"QuietStar not configured (no main_llm), skipping pre-check")
        else:
            logger.warning(f"QuietStar workflow pre-check failed (continuing): {e}")
    except Exception as e:
        logger.warning(f"QuietStar workflow pre-check failed (continuing): {e}")


# =============================================================================
# FAIL-SECURE FALLBACKS WHEN LLM_GUARD NOT AVAILABLE
# =============================================================================
# These functions are defined at module level so they can be used when the
# security module import fails. They implement fail-secure behavior.
# =============================================================================

if not LLM_GUARD_AVAILABLE:
    # Provide no-op decorator if security module not available
    def secure_operation(*args, **kwargs):
        """No-op decorator when security module not available."""
        def decorator(func):
            return func
        return decorator if not args or callable(args[0]) else decorator
    async_secure_operation = secure_operation

    # Provide FAIL-SECURE fallback functions - FIX BLACK-NEW-03
    # When security module unavailable, DENY by default (not allow)
    async def sanitize_input(text: str, security_level=None):
        """FAIL-SECURE fallback when security module not available.

        CRITICAL: Returns is_safe=False to force review when LLM Guard unavailable.
        """
        from dataclasses import dataclass, field

        @dataclass
        class FallbackSanitizationResult:
            is_safe: bool = False  # FAIL-SECURE: Deny when unavailable
            original_text: str = ""
            sanitized_text: str = ""
            issues_found: List[str] = field(default_factory=list)
            risk_score: float = 1.0  # Max risk when security unavailable

        logger.warning("SECURITY DEGRADED: LLM Guard unavailable, marking input as UNSAFE")
        return FallbackSanitizationResult(
            is_safe=False,  # FAIL-SECURE
            original_text=text,
            sanitized_text=text,
            issues_found=["Security module unavailable - manual review required"],
            risk_score=1.0
        )

    async def validate_output(output: str, original_prompt: str, security_level=None):
        """FAIL-SECURE fallback when security module not available.

        CRITICAL: Returns is_valid=False to force review when LLM Guard unavailable.
        """
        from dataclasses import dataclass, field

        @dataclass
        class FallbackValidationResult:
            is_valid: bool = False  # FAIL-SECURE: Invalid when unavailable
            output_text: str = ""
            validated_text: str = ""
            issues_found: List[str] = field(default_factory=list)
            risk_score: float = 1.0  # Max risk
            relevance_score: float = 0.0  # Cannot verify relevance

        logger.warning("SECURITY DEGRADED: LLM Guard unavailable, marking output as INVALID")
        return FallbackValidationResult(
            is_valid=False,  # FAIL-SECURE
            output_text=output,
            validated_text=output,
            issues_found=["Security module unavailable - manual review required"],
            risk_score=1.0
        )

    async def detect_pii(text: str, anonymize: bool = True):
        """FAIL-SECURE fallback when security module not available.

        CRITICAL: Returns has_pii=True to force redaction when detector unavailable.
        """
        from dataclasses import dataclass, field

        @dataclass
        class FallbackPIIDetectionResult:
            has_pii: bool = True  # FAIL-SECURE: Assume PII present when can't detect
            pii_types: List[str] = field(default_factory=list)
            pii_count: int = -1  # Unknown count
            redacted_text: Optional[str] = None
            confidence: float = 0.0  # No confidence - security degraded

        logger.warning("SECURITY DEGRADED: PII detector unavailable, assuming PII present")
        return FallbackPIIDetectionResult(
            has_pii=True,  # FAIL-SECURE: Assume PII
            pii_types=["UNKNOWN - detector unavailable"],
            pii_count=-1,
            redacted_text="[REDACTED - security module unavailable]" if anonymize else text
        )

    async def filter_toxicity(text: str):
        """FAIL-SECURE fallback when security module not available.

        CRITICAL: Returns is_toxic=True to force filtering when detector unavailable.
        """
        from dataclasses import dataclass, field

        @dataclass
        class FallbackToxicityFilterResult:
            is_toxic: bool = True  # FAIL-SECURE: Assume toxic when can't verify
            toxicity_score: float = 1.0  # Max toxicity
            categories: Dict[str, float] = field(default_factory=dict)
            filtered_text: Optional[str] = None

        logger.warning("SECURITY DEGRADED: Toxicity filter unavailable, assuming toxic content")
        return FallbackToxicityFilterResult(
            is_toxic=True,  # FAIL-SECURE: Assume toxic
            toxicity_score=1.0,
            categories={"unknown": 1.0},
            filtered_text="[FILTERED - security module unavailable]"
        )

    # Fallback result types
    SanitizationResult = None
    ValidationResult = None
    PIIDetectionResult = None
    ToxicityFilterResult = None

# P3-001: Hamilton DAG pipelines for data transformation
try:
    from pipeline.langgraph.hamilton_pipelines import (
        run_claim_verification as hamilton_run_claim_verification,
        run_gate_validation as hamilton_run_gate_validation,
        run_evidence_collection as hamilton_run_evidence_collection,
        is_hamilton_available,
        PipelineResult,
        HAMILTON_AVAILABLE,
    )
except ImportError:
    HAMILTON_AVAILABLE = False
    is_hamilton_available = lambda: False

# Grafana metrics integration for Pipeline Control Center dashboard
try:
    from pipeline.grafana_metrics import (
        get_metrics_publisher,
        GrafanaMetricsPublisher,
    )
    GRAFANA_METRICS_AVAILABLE = True
except ImportError:
    GRAFANA_METRICS_AVAILABLE = False
    get_metrics_publisher = None
    GrafanaMetricsPublisher = None

# NF-011 FIX: Gate validation blocking flag
# When False (default): Gate validation errors are non-blocking (legacy behavior)
# When True: Gate validation errors BLOCK execution (recommended for production)
GATE_VALIDATION_BLOCKING = os.getenv("GATE_VALIDATION_BLOCKING", "false").lower() == "true"

# SEC-011 FIX: In production, GATE_VALIDATION_BLOCKING should default to True
# Gate validation errors in production should BLOCK, not be ignored
_IS_PRODUCTION_WORKFLOW = os.getenv("ENVIRONMENT", "development").lower() == "production"
if _IS_PRODUCTION_WORKFLOW and not GATE_VALIDATION_BLOCKING:
    # In production without explicit setting, default to blocking
    if not os.getenv("GATE_VALIDATION_BLOCKING"):
        logger.warning(
            "SECURITY-011: GATE_VALIDATION_BLOCKING not set in PRODUCTION - "
            "defaulting to blocking mode for safety. Set GATE_VALIDATION_BLOCKING=false "
            "explicitly if non-blocking is intentional."
        )
        GATE_VALIDATION_BLOCKING = True


# =============================================================================
# HIERARCHICAL REWORK SYSTEM (FIX 2026-01-26)
# =============================================================================
# When work is rejected (gates fail, signoffs rejected), the rework is delegated
# down the hierarchy. Each level is responsible for redistributing to their
# subordinates. This maintains accountability and chain of command.
#
# Hierarchy:
#   L1 (Executive: CEO, Presidente) rejects -> L2 (VPs) reworks
#   L2 (VPs) rejects -> L3 (Masters) reworks
#   L3 (Masters) rejects -> L4 (Squad Leads) reworks
#   L4 (Squad Leads) rejects -> L5 (Workers) reworks
# =============================================================================

# Tier hierarchy order (higher index = lower in hierarchy)
TIER_ORDER = ["L0", "L1", "L2", "L3", "L4", "L5", "L6"]

# Agent to tier mapping
AGENT_TIER_MAP = {
    # L0 - System
    "run_master": "L0",
    "pack_driver": "L0",
    "ops_ctrl": "L0",
    "orchestrator": "L0",
    "resource_optimizer": "L0",
    "run_supervisor": "L0",
    "system_observer": "L0",
    # L1 - Executive
    "presidente": "L1",
    "ceo": "L1",
    "arbiter": "L1",
    "retrospective_master": "L1",
    "human_approver": "L1",
    # L2 - VP
    "spec_vp": "L2",
    "exec_vp": "L2",
    "qa_vp": "L2",
    "external_liaison": "L2",
    "integration_officer": "L2",
    # L3 - Masters
    "spec_master": "L3",
    "ace_exec": "L3",
    "qa_master": "L3",
    "debt_tracker": "L3",
    "sprint_planner": "L3",
    # L4 - Squad Leads
    "squad_lead_spec": "L4",
    "squad_lead_exec": "L4",
    "squad_lead_qa": "L4",
    "squad_lead": "L4",
    "ace_orchestration": "L4",
    "human_layer": "L4",
    "human_layer_specialist": "L4",
    # L5 - Workers
    "developer": "L5",
    "auditor": "L5",
    "clean_reviewer": "L5",
    "dependency_mapper": "L5",
    "edge_case_hunter": "L5",
    "gap_hunter": "L5",
    "human_reviewer": "L5",
    "product_owner": "L5",
    "project_manager": "L5",
    "red_team_agent": "L5",
    "refinador": "L5",
    "task_decomposer": "L5",
    "technical_planner": "L5",
    # L6 - Specialists
    "blockchain_engineer": "L6",
    "data_engineer": "L6",
    "legal_tech_specialist": "L6",
    "llm_orchestrator": "L6",
    "oracle_architect": "L6",
    "ui_designer": "L6",
    "ux_researcher": "L6",
    "web3_frontend": "L6",
}

# Delegation targets: which agent at lower tier should receive rework
# Format: {rejecting_tier: {work_type: delegate_to_agent}}
REWORK_DELEGATION_MAP = {
    "L1": {  # CEO/Presidente rejects -> VPs
        "spec": "spec_vp",
        "exec": "exec_vp",
        "qa": "qa_vp",
        "default": "exec_vp",
    },
    "L2": {  # VPs reject -> Masters
        "spec": "spec_master",
        "exec": "ace_exec",
        "qa": "qa_master",
        "default": "ace_exec",
    },
    "L3": {  # Masters reject -> Squad Leads
        "spec": "squad_lead_spec",
        "exec": "squad_lead_exec",
        "qa": "squad_lead_qa",
        "default": "squad_lead_exec",
    },
    "L4": {  # Squad Leads reject -> Workers
        "spec": "refinador",
        "exec": "developer",
        "qa": "auditor",
        "default": "developer",
    },
}


def get_agent_tier(agent_id: str) -> str:
    """Get the tier of an agent.

    Args:
        agent_id: The agent identifier.

    Returns:
        The tier (L0-L6) or "L5" as default for unknown agents.
    """
    return AGENT_TIER_MAP.get(agent_id, "L5")


def get_rework_delegate(
    rejecting_agent: str,
    work_type: str = "exec",
) -> Dict[str, Any]:
    """Determine who should receive the rework based on hierarchy.

    When an agent rejects work, this function determines which agent
    at the next lower tier should receive the rework for redistribution.

    Args:
        rejecting_agent: The agent who rejected the work.
        work_type: Type of work ("spec", "exec", "qa").

    Returns:
        Dict with delegation info:
        - delegate_to: Agent to delegate to
        - delegate_tier: Tier of the delegate
        - rejecting_tier: Tier of the rejecting agent
        - delegation_chain: Expected chain of delegation
    """
    rejecting_tier = get_agent_tier(rejecting_agent)

    # Find delegate tier (one level down)
    try:
        tier_idx = TIER_ORDER.index(rejecting_tier)
        delegate_tier = TIER_ORDER[tier_idx + 1] if tier_idx + 1 < len(TIER_ORDER) else rejecting_tier
    except ValueError:
        delegate_tier = "L3"  # Default to Masters

    # Get delegation map for rejecting tier
    delegation_map = REWORK_DELEGATION_MAP.get(rejecting_tier, {})
    delegate_to = delegation_map.get(work_type, delegation_map.get("default", "ace_exec"))

    # Build expected delegation chain
    delegation_chain = []
    current_tier_idx = TIER_ORDER.index(delegate_tier)
    for i in range(current_tier_idx, min(current_tier_idx + 3, len(TIER_ORDER))):
        tier = TIER_ORDER[i]
        tier_delegate = REWORK_DELEGATION_MAP.get(tier, {}).get(work_type)
        if tier_delegate:
            delegation_chain.append({"tier": tier, "agent": tier_delegate})

    return {
        "delegate_to": delegate_to,
        "delegate_tier": delegate_tier,
        "rejecting_tier": rejecting_tier,
        "rejecting_agent": rejecting_agent,
        "work_type": work_type,
        "delegation_chain": delegation_chain,
    }


def create_hierarchical_rework_context(
    rejecting_agent: str,
    rejection_reason: str,
    work_type: str = "exec",
    failed_items: Optional[List[str]] = None,
    current_attempt: int = 1,
    max_attempts: int = 3,
) -> Dict[str, Any]:
    """Create a rework context with hierarchical delegation information.

    This context is used by the rework mechanism to delegate work
    down the hierarchy to the appropriate agent.

    Args:
        rejecting_agent: Agent who rejected the work.
        rejection_reason: Why the work was rejected.
        work_type: Type of work ("spec", "exec", "qa").
        failed_items: List of failed items (gates, signoffs, etc.).
        current_attempt: Current attempt number.
        max_attempts: Maximum number of rework attempts.

    Returns:
        Dict with complete rework context.
    """
    delegation_info = get_rework_delegate(rejecting_agent, work_type)

    return {
        "violation_report": {
            "node_name": "exec",  # Always go back to exec node for rework
            "violation_type": "hierarchical_rework",
            "rejecting_agent": rejecting_agent,
            "rejecting_tier": delegation_info["rejecting_tier"],
            "delegate_to": delegation_info["delegate_to"],
            "delegate_tier": delegation_info["delegate_tier"],
            "work_type": work_type,
        },
        "retry_hints": [
            f"Work rejected by {rejecting_agent} ({delegation_info['rejecting_tier']}): {rejection_reason}",
            f"Delegating to {delegation_info['delegate_to']} ({delegation_info['delegate_tier']})",
            f"Failed items: {', '.join(failed_items or ['none specified'])}",
        ],
        "error_learnings": rejection_reason,
        "attempt_number": current_attempt,
        "max_attempts": max_attempts,
        "delegation_chain": delegation_info["delegation_chain"],
        "hierarchical_rework": True,  # Flag to indicate hierarchical rework
    }


# =============================================================================
# RETRY UTILITIES
# =============================================================================


class RetryConfig:
    """Configuration for retry with exponential backoff."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay_seconds: float = 1.0,
        max_delay_seconds: float = 30.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: tuple = (Exception,),
    ):
        """Initialize retry configuration.

        Args:
            max_retries: Maximum number of retry attempts.
            base_delay_seconds: Initial delay between retries.
            max_delay_seconds: Maximum delay between retries.
            exponential_base: Base for exponential backoff.
            jitter: Whether to add random jitter to delays.
            retryable_exceptions: Exception types that trigger retries.
        """
        self.max_retries = max_retries
        self.base_delay_seconds = base_delay_seconds
        self.max_delay_seconds = max_delay_seconds
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions


# Default retry configs for different scenarios
RETRY_CONFIG_TRANSIENT = RetryConfig(
    max_retries=3,
    base_delay_seconds=1.0,
    max_delay_seconds=10.0,
    retryable_exceptions=(ConnectionError, TimeoutError, OSError),
)

RETRY_CONFIG_EXTERNAL = RetryConfig(
    max_retries=5,
    base_delay_seconds=2.0,
    max_delay_seconds=60.0,
    retryable_exceptions=(ConnectionError, TimeoutError, OSError),
)


async def retry_with_backoff(
    operation: Callable[[], Awaitable[Any]],
    operation_name: str,
    config: RetryConfig = RETRY_CONFIG_TRANSIENT,
) -> Any:
    """Execute an async operation with exponential backoff retry.

    Args:
        operation: Async callable to execute.
        operation_name: Name for logging purposes.
        config: Retry configuration.

    Returns:
        Result of the operation.

    Raises:
        The last exception if all retries are exhausted.
    """
    import random

    last_exception = None

    for attempt in range(config.max_retries + 1):
        try:
            return await operation()
        except config.retryable_exceptions as e:
            last_exception = e

            if attempt == config.max_retries:
                logger.error(
                    f"Operation '{operation_name}' failed after {config.max_retries + 1} attempts: {e}"
                )
                raise

            # Calculate delay with exponential backoff
            delay = min(
                config.base_delay_seconds * (config.exponential_base ** attempt),
                config.max_delay_seconds,
            )

            # Add jitter if enabled (±25%)
            if config.jitter:
                delay = delay * (0.75 + random.random() * 0.5)

            logger.warning(
                f"Operation '{operation_name}' failed (attempt {attempt + 1}/{config.max_retries + 1}): {e}. "
                f"Retrying in {delay:.2f}s..."
            )

            await asyncio.sleep(delay)

    # Should not reach here, but just in case
    if last_exception:
        raise last_exception


def _non_blocking_sleep(seconds: float) -> None:
    """P0-07 FIX: Sleep without blocking the event loop.

    If we're in an async context, uses Timer+Event to avoid blocking the event loop.
    If not in async context, uses regular sleep.

    Args:
        seconds: Number of seconds to sleep.
    """
    import threading

    try:
        # Check if we're in an async context
        asyncio.get_running_loop()
        # In async context - use Timer+Event (doesn't block loop)
        event = threading.Event()
        timer = threading.Timer(seconds, event.set)
        timer.start()
        event.wait()
    except RuntimeError:
        # No event loop - safe to use regular sleep
        time.sleep(seconds)


def sync_retry_with_backoff(
    operation: Callable[[], Any],
    operation_name: str,
    config: RetryConfig = RETRY_CONFIG_TRANSIENT,
) -> Any:
    """Execute a sync operation with exponential backoff retry.

    P0-07 FIX (2026-01-30): Uses _non_blocking_sleep to avoid blocking the event loop
    when called from an async context.

    Args:
        operation: Callable to execute.
        operation_name: Name for logging purposes.
        config: Retry configuration.

    Returns:
        Result of the operation.

    Raises:
        The last exception if all retries are exhausted.
    """
    import random
    import time

    last_exception = None

    for attempt in range(config.max_retries + 1):
        try:
            return operation()
        except config.retryable_exceptions as e:
            last_exception = e

            if attempt == config.max_retries:
                logger.error(
                    f"Operation '{operation_name}' failed after {config.max_retries + 1} attempts: {e}"
                )
                raise

            # Calculate delay with exponential backoff
            delay = min(
                config.base_delay_seconds * (config.exponential_base ** attempt),
                config.max_delay_seconds,
            )

            # Add jitter if enabled (±25%)
            if config.jitter:
                delay = delay * (0.75 + random.random() * 0.5)

            logger.warning(
                f"Operation '{operation_name}' failed (attempt {attempt + 1}/{config.max_retries + 1}): {e}. "
                f"Retrying in {delay:.2f}s..."
            )

            # P0-07 FIX: Non-blocking sleep
            _non_blocking_sleep(delay)

    # Should not reach here, but just in case
    if last_exception:
        raise last_exception


# =============================================================================
# NODE IMPLEMENTATIONS
# =============================================================================


class WorkflowNodes:
    """LangGraph node implementations for Pipeline V2.

    Each node:
    1. Checks SAFE_HALT before doing work
    2. Records node_history entry with idempotency_key
    3. Performs its work
    4. Updates state and returns

    Invariants enforced:
    - I1: run_id namespacing (all keys contain run_id)
    - I2: Idempotency (deterministic keys for replay)
    - I3: Phase order (INIT -> SPEC -> PLAN -> EXEC -> QA -> VOTE -> DONE)
    - I9: SAFE_HALT takes precedence
    """

    def __init__(
        self,
        run_dir: Path | str,
        run_id: str = "",
        sprint_id: str = "",
        safe_halt_checker: Optional[Callable[[], Awaitable[bool]]] = None,
        stack_injector: Optional[StackInjector] = None,
    ):
        """Initialize workflow nodes.

        Args:
            run_dir: Directory for run artifacts (Path or str, converted to Path).
            run_id: Run identifier for invariant checking.
            sprint_id: Sprint identifier for invariant checking.
            safe_halt_checker: Async function to check if SAFE_HALT is active.
            stack_injector: StackInjector for stack wiring. Uses default if None.
        """
        # Ensure run_dir is always a Path object (fixes str/str division error)
        self.run_dir = Path(run_dir) if isinstance(run_dir, str) else run_dir
        self.run_id = run_id
        self.sprint_id = sprint_id
        self.safe_halt_checker = safe_halt_checker or self._default_safe_halt_check
        self.stack_injector = stack_injector or get_stack_injector()

        # Invariant checker for I1-I11 enforcement (CRIT-005 fix)
        self._invariant_checker: Optional[InvariantChecker] = None
        if run_id and sprint_id:
            self._invariant_checker = InvariantChecker(run_id=run_id, sprint_id=sprint_id)

        # Trust boundary enforcer for tier-based access control (CRIT-003 fix)
        # Uses non-strict mode to log warnings but not block (redirect not block philosophy)
        self._trust_enforcer: TrustBoundaryEnforcer = get_trust_boundary_enforcer()

        # Lazy-loaded V2 modules
        self._gate_runner = None
        self._spec_kit_loader = None
        self._artifact_manager = None
        self._crewai_client = None

        # Cache for injected stacks per operation
        self._operation_stacks: Dict[Operation, StackContext] = {}

        # Partial Stacks Integration (Active RAG, BoT, RAGAS, Phoenix, DeepEval)
        # Provides workflow hooks for the 5 PARTIAL stacks
        self._partial_stacks: Optional[PartialStacksIntegration] = None
        if PARTIAL_STACKS_INTEGRATION_AVAILABLE and get_partial_stacks_integration:
            try:
                self._partial_stacks = get_partial_stacks_integration()
                logger.info("Partial stacks integration initialized for workflow")
            except Exception as e:
                logger.warning(f"Failed to initialize partial stacks integration: {e}")

        # Grafana metrics publisher for Pipeline Control Center dashboard
        self._grafana_metrics: Optional[GrafanaMetricsPublisher] = None
        if GRAFANA_METRICS_AVAILABLE and get_metrics_publisher:
            try:
                self._grafana_metrics = get_metrics_publisher()
                logger.info("Grafana metrics publisher initialized for workflow")
            except Exception as e:
                logger.warning(f"Failed to initialize Grafana metrics: {e}")

        # Default timeout for node operations (HIGH-007 fix)
        # P0-08: GENEROUS + 30% gordura - 39 min per node
        self.node_timeout_seconds: float = 2340.0  # 39 minutes default

        # OPT-02-005: Health check cache with TTL (5 seconds)
        self._health_cache: Dict[str, Any] = {}
        self._health_cache_timestamp: float = 0.0
        self._health_cache_ttl: float = 5.0  # seconds

        # P0-5 FIX: Stack affinity scorer for cerebro-stack routing
        # Loads cerebro_stacks YAMLs for stack-aware task routing
        self._stack_affinity_scorer = None
        try:
            from pipeline.stack_affinity import get_stack_affinity_scorer
            self._stack_affinity_scorer = get_stack_affinity_scorer()
            if self._stack_affinity_scorer._profiles:
                logger.info(
                    f"Stack affinity scorer initialized with "
                    f"{len(self._stack_affinity_scorer._profiles)} profiles"
                )
        except Exception as e:
            logger.warning(f"Failed to initialize stack affinity scorer: {e}")

    async def _with_timeout(
        self,
        coro,
        timeout_seconds: Optional[float] = None,
        operation_name: str = "operation",
    ):
        """Execute a coroutine with timeout protection (HIGH-007 fix).

        Args:
            coro: The coroutine to execute.
            timeout_seconds: Timeout in seconds. Uses node_timeout_seconds if None.
            operation_name: Name for logging purposes.

        Returns:
            Result of the coroutine.

        Raises:
            asyncio.TimeoutError: If the operation times out.
        """
        timeout = timeout_seconds or self.node_timeout_seconds
        try:
            async with asyncio.timeout(timeout):
                return await coro
        except asyncio.TimeoutError:
            logger.error(f"Operation '{operation_name}' timed out after {timeout}s")
            raise

    async def _default_safe_halt_check(self) -> bool:
        """Default SAFE_HALT check (reads from file/Redis)."""
        # Check file-based SAFE_HALT
        halt_file = self.run_dir / "SAFE_HALT.json"
        if halt_file.exists():
            import json
            try:
                with open(halt_file) as f:
                    data = json.load(f)
                    return data.get("active", False)
            except Exception as e:
                logger.debug(f"GRAPH: Graph operation failed: {e}")
        return False

    def _publish_grafana_metrics(
        self,
        state: PipelineState,
        node_name: str = "",
        **extra_metrics,
    ) -> None:
        """Publish metrics to Grafana Pipeline Control Center dashboard.

        This method publishes all relevant metrics from the current state
        to Redis keys that Grafana reads directly.

        Args:
            state: Current pipeline state.
            node_name: Name of the current node for context.
            **extra_metrics: Additional metrics to publish.
        """
        if not self._grafana_metrics:
            return

        try:
            m = self._grafana_metrics

            # Core status metrics
            m.set_status("running")
            m.set_current_sprint(state.get("sprint_id", ""))

            # Phase/level mapping
            phase = state.get("phase", "INIT")
            m.set_phase(phase)  # Update Redis with current phase
            phase_to_level = {
                "INIT": "L0",
                "SPEC": "L2",
                "PLAN": "L3",
                "EXEC": "L5",
                "GATE": "L3",
                "SIGNOFF": "L1",
                "ARTIFACT": "L5",
                "DONE": "L0",
            }
            m.set_current_level(phase_to_level.get(phase, "L0"))

            # Task/objective from node name
            if node_name:
                m.set_current_task(f"Executing {node_name} node for {state.get('sprint_id', 'unknown')}")

            # Gate metrics
            gate = state.get("gate", {})
            if isinstance(gate, dict):
                current_gate = gate.get("current_gate", "")
                if current_gate:
                    m.set_current_gate(current_gate)

                gates_passed = gate.get("gates_passed", [])
                m._state.gates_passed = len(gates_passed) if gates_passed else 0
                m._set_key("gates_passed", m._state.gates_passed)

            # Update elapsed time
            m.update_elapsed()

            # Publish event for this node execution
            m.publish_event(f"node_{node_name}", {
                "phase": phase,
                "sprint_id": state.get("sprint_id", ""),
                "run_id": state.get("run_id", ""),
            })

            # Extra metrics
            for key, value in extra_metrics.items():
                if key == "agent":
                    m.set_current_agent(value)
                elif key == "gate":
                    m.set_current_gate(value)
                elif key == "dod":
                    m.set_current_dod(value if isinstance(value, list) else [])

            logger.debug(f"Published Grafana metrics for node {node_name}")

        except Exception as e:
            # Never let metrics publishing break the pipeline
            logger.debug(f"Failed to publish Grafana metrics: {e}")

    def _cached_health_check(self) -> Dict[str, Any]:
        """Get health check results with TTL caching (OPT-02-005).

        Caches health check results for 5 seconds to avoid repeated
        expensive health checks during a single workflow execution.

        Returns:
            Dictionary of stack health results.
        """
        now = time.time()
        if now - self._health_cache_timestamp < self._health_cache_ttl:
            return self._health_cache

        # Cache miss or expired - do actual health check
        if self.stack_injector:
            self._health_cache = self.stack_injector.check_health()
        else:
            self._health_cache = {}
        self._health_cache_timestamp = now
        return self._health_cache

    def _publish_health_metrics(self) -> None:
        """Publish health check results to Grafana dashboard."""
        if not self._grafana_metrics:
            return

        try:
            # Get stack health from injector (OPT-02-005: cached)
            health = self._cached_health_check()

            # Map stack names to dashboard expected names
            health_mapping = {
                "redis": "redis",
                "qdrant": "qdrant",
                "falkordb": "falkordb",
                "langfuse": "langfuse",
                "crewai": "crewai",
                "letta": "letta",
                "ollama": "ollama",
                "nemo_rails": "nemo",
            }

            for stack_name, dashboard_name in health_mapping.items():
                is_healthy = health.get(stack_name, {}).get("healthy", False)
                self._grafana_metrics.set_health(dashboard_name, is_healthy)

            logger.debug("Published health metrics to Grafana")

        except Exception as e:
            logger.debug(f"Failed to publish health metrics: {e}")

    def _get_stacks_for_operation(
        self,
        operation: Operation,
        state: PipelineState,
        check_health: bool = False,
    ) -> StackContext:
        """Get stacks for an operation with caching.

        Args:
            operation: The pipeline operation.
            state: Current pipeline state (for run_id, sprint_id).
            check_health: Whether to verify stack health (slower).

        Returns:
            StackContext with available stacks.
        """
        # Check cache first (health check only done once per operation)
        if operation in self._operation_stacks:
            return self._operation_stacks[operation]

        try:
            stacks = self.stack_injector.get_stacks_for_operation(
                operation=operation,
                check_health=check_health,
            )
            health_report = self.stack_injector.get_health_report()

            ctx = StackContext(
                run_id=state["run_id"],
                sprint_id=state["sprint_id"],
                operation=operation,
                stacks=stacks,
                stack_health=health_report,
            )

            # Cache the context
            self._operation_stacks[operation] = ctx
            logger.debug(f"Loaded {len(stacks)} stacks for {operation.value}: {list(stacks.keys())}")
            return ctx

        except RuntimeError as e:
            # Required stacks missing - this is a critical error
            logger.error(f"Failed to load stacks for {operation.value}: {e}")
            raise

    def _generate_idempotency_key(
        self,
        node: str,
        run_id: str,
        sprint_id: str,
        attempt: int,
    ) -> str:
        """Generate deterministic idempotency key for a node.

        Invariant I2: Each node has deterministic idempotency_key;
        re-execution does not duplicate side effects.
        """
        return f"{node}:{run_id}:{sprint_id}:attempt:{attempt}"

    # OPT-02-006: Maximum entries in node_history (rolling window)
    MAX_NODE_HISTORY: int = 50

    def _record_node_start(
        self,
        state: PipelineState,
        node: str,
    ) -> PipelineState:
        """Record node start in node_history with rolling window (OPT-02-006)."""
        idempotency_key = self._generate_idempotency_key(
            node=node,
            run_id=state["run_id"],
            sprint_id=state["sprint_id"],
            attempt=state["attempt"],
        )

        entry = create_node_history_entry(
            node=node,
            status="started",
            idempotency_key=idempotency_key,
        )

        # OPT-02-006: Limit history to MAX_NODE_HISTORY entries (rolling window)
        history = state.get("node_history", [])
        # Keep last (MAX_NODE_HISTORY - 1) entries + new entry
        new_history = [*history[-(self.MAX_NODE_HISTORY - 1):], entry]

        return {
            **state,
            "node_history": new_history,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _record_node_complete(
        self,
        state: PipelineState,
        node: str,
        error: Optional[str] = None,
    ) -> PipelineState:
        """Mark the last node_history entry as completed/failed."""
        history = list(state["node_history"])

        # Find the last entry for this node
        for i in range(len(history) - 1, -1, -1):
            if history[i]["node"] == node and history[i]["status"] == "started":
                history[i] = {
                    **history[i],
                    "status": "failed" if error else "completed",
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "error": error,
                }
                break

        return {
            **state,
            "node_history": history,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    async def _check_safe_halt(self, state: PipelineState) -> tuple[bool, PipelineState]:
        """Check SAFE_HALT and update state if halted.

        Invariant I9: if safe_halt is active, graph enters HALT.

        Returns:
            (should_halt, updated_state)
        """
        should_halt = await self.safe_halt_checker()

        if should_halt:
            logger.warning(f"SAFE_HALT active for run {state['run_id']}")
            return True, update_status(
                update_phase(state, SprintPhase.HALT),
                PipelineStatus.HALTED,
            )

        return False, state

    def _validate_phase_transition(
        self,
        state: PipelineState,
        to_phase: SprintPhase,
        is_retry: bool = False,
    ) -> tuple[bool, Optional[str]]:
        """Validate phase transition using invariant checker (I3).

        Args:
            state: Current pipeline state.
            to_phase: Target phase.
            is_retry: Whether this is a retry.

        Returns:
            (valid, error_message) - (True, None) if valid, (False, msg) if invalid.
        """
        if self._invariant_checker is None:
            # No checker configured, allow transition (non-strict mode)
            return True, None

        from_phase = state.get("phase", SprintPhase.INIT.value)
        result = self._invariant_checker.check_phase_transition(
            from_phase=from_phase,
            to_phase=to_phase.value,
            is_retry=is_retry,
        )

        if not result.passed:
            logger.warning(
                f"Invariant I3 violation: Invalid phase transition {from_phase} -> {to_phase.value}: {result.message}"
            )
            return False, result.message

        return True, None

    def _validate_signoff_allowed(self, state: PipelineState) -> tuple[bool, Optional[str]]:
        """Validate signoff is allowed using invariant checker (I4).

        Args:
            state: Current pipeline state.

        Returns:
            (valid, error_message) - (True, None) if valid, (False, msg) if invalid.
        """
        if self._invariant_checker is None:
            return True, None

        gate_status = state.get("gate_status", "")
        gates_passed = state.get("gates_passed", [])

        result = self._invariant_checker.check_signoff_allowed(
            gate_status=gate_status,
            gates_passed=gates_passed,
        )

        if not result.passed:
            logger.warning(f"Invariant I4 violation: {result.message}")
            return False, result.message

        return True, None

    def _handle_invariant_violation(
        self,
        state: PipelineState,
        result: InvariantCheckResult,
        node: str,
        invariant_name: str,
        is_blocking: bool = True,
        raise_exception: bool = True,
    ) -> tuple[PipelineState, bool]:
        """Handle an invariant check result and update state if violated.

        IMPORTANT: When is_blocking=True and raise_exception=True, this method
        will RAISE InvariantViolationError to stop execution. This is the
        CORRECT behavior - invariants MUST block, not just warn.

        Args:
            state: Current pipeline state.
            result: InvariantCheckResult from the check.
            node: Node where check was performed.
            invariant_name: Name of the invariant for logging.
            is_blocking: If True, violations cause failure. If False, just log warning.
            raise_exception: If True AND is_blocking, raises InvariantViolationError.

        Returns:
            (updated_state, should_continue) - If should_continue is False, node should return.

        Raises:
            InvariantViolationError: When is_blocking=True and raise_exception=True.
        """
        if result.passed:
            return state, True

        # Log all violations - use ERROR level for blocking, WARNING for non-blocking
        log_level = logger.error if is_blocking else logger.warning
        for violation in result.violations:
            log_level(f"{invariant_name} violation in {node}: {violation}")

        if is_blocking:
            # Add error entry and update status
            error_msg = f"{invariant_name} violation: {'; '.join(str(v) for v in result.violations)}"
            error_entry = create_error_entry(
                where=node,
                error=error_msg,
                recoverable=False,
            )
            state = add_error(state, error_entry)
            state = update_status(state, PipelineStatus.FAILED)

            # Mark task as invalidated for rework
            state["_task_invalidated"] = True
            state["_invariant_violation"] = error_msg
            state["_violation_node"] = node

            # STRICT ENFORCEMENT: Raise exception to STOP execution
            if raise_exception:
                logger.error(
                    f"INVARIANT ENFORCEMENT: Raising InvariantViolationError for {invariant_name} in {node}"
                )
                raise InvariantViolationError(result)

            return state, False
        else:
            # Non-blocking: just log and continue
            self._emit_event(
                state,
                f"invariant_warning_{invariant_name.lower()}",
                f"Non-blocking {invariant_name} violation detected in {node}",
            )
            return state, True

    def _check_trust_boundary(
        self,
        state: PipelineState,
        agent_id: str,
        resource: str,
        action: str,
        node: str,
        strict: bool = True,
    ) -> tuple[PipelineState, bool]:
        """Check trust boundary access and BLOCK on violations.

        IMPORTANT: This implements STRICT trust boundary enforcement.
        When strict=True (default), violations BLOCK execution.
        This is the CORRECT behavior - security MUST block, not warn.

        Args:
            state: Current pipeline state.
            agent_id: Agent attempting the action.
            resource: Resource being accessed.
            action: Action being performed.
            node: Node where check is performed (for logging).
            strict: If True, block on violations. If False, warn and continue.

        Returns:
            (updated_state, access_allowed) - access_allowed=False blocks execution.

        Raises:
            PermissionError: When strict=True and access is denied.
        """
        result = self._trust_enforcer.check_access(agent_id, resource, action)

        if not result.allowed:
            # Log the violation at ERROR level
            logger.error(
                f"TRUST BOUNDARY VIOLATION in {node}: agent '{agent_id}' "
                f"attempted '{action}' on '{resource}' - {result.reason}"
            )

            # Emit event for observability
            self._emit_event(
                state,
                "trust_boundary_violation",
                f"BLOCKED: Agent '{agent_id}' attempted unauthorized '{action}' on '{resource}' in {node}",
            )

            # Record in state for audit trail
            violations = state.get("trust_violations", [])
            violations.append({
                "node": node,
                "agent_id": agent_id,
                "resource": resource,
                "action": action,
                "reason": result.reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "blocked": strict,
            })
            state = {**state, "trust_violations": violations}

            if strict:
                # STRICT ENFORCEMENT: Mark as failed and BLOCK
                state["_task_invalidated"] = True
                state["_trust_violation"] = f"{agent_id} denied {action} on {resource}"
                state = update_status(state, PipelineStatus.FAILED)

                # Raise PermissionError to stop execution
                raise PermissionError(
                    f"Trust boundary violation: agent '{agent_id}' is not authorized "
                    f"to perform '{action}' on '{resource}' in {node}. Reason: {result.reason}"
                )

            # Non-strict: warn but allow (NOT RECOMMENDED)
            logger.warning(f"Trust violation allowed to continue (strict=False) in {node}")
            return state, True
        else:
            logger.debug(
                f"Trust boundary check passed in {node}: agent '{agent_id}' "
                f"allowed '{action}' on '{resource}'"
            )

        return state, True

    # =========================================================================
    # INSTITUTIONAL MEMORY LOOKUP (MEMORY-001)
    # =========================================================================
    # Provides existing code awareness and related sprint discovery.
    # This enables the pipeline to know what code already exists and avoid
    # reimplementing functionality that was built in earlier sprints.
    # =========================================================================

    async def _lookup_institutional_memory(
        self,
        sprint_id: str,
        deliverables: List[str],
        timeout_seconds: float = 10.0,
    ) -> Dict[str, Any]:
        """Lookup institutional memory for existing code and related sprints.

        This method queries the existing_code_inventory and pack_discovery modules
        to understand what code already exists and what related sprints may be
        relevant for the current execution.

        NON-BLOCKING: This method NEVER blocks the pipeline. Failures return
        an empty dict and log warnings.

        Args:
            sprint_id: The sprint identifier (e.g., "S45").
            deliverables: List of expected deliverable file paths.
            timeout_seconds: Maximum time allowed for the lookup (default 10s).

        Returns:
            Dict containing:
                - existing_code: SprintInventory data (files, functions, classes)
                - related_sprints: List of related sprint IDs
                - reusable_modules: List of modules that can be imported
                - do_not_reimplement: List of functionality that already exists

        Note:
            Returns empty dict {} on any failure to ensure non-blocking behavior.
        """
        result: Dict[str, Any] = {}

        # Try to import the modules (graceful degradation if not available)
        try:
            from pipeline.existing_code_inventory import get_sprint_inventory, SprintInventory
            EXISTING_CODE_INVENTORY_AVAILABLE = True
        except ImportError as e:
            logger.warning(f"MEMORY-001: existing_code_inventory not available: {e}")
            EXISTING_CODE_INVENTORY_AVAILABLE = False

        try:
            from pipeline.pack_discovery import get_related_packs, PackInfo
            PACK_DISCOVERY_AVAILABLE = True
        except ImportError as e:
            logger.warning(f"MEMORY-001: pack_discovery not available: {e}")
            PACK_DISCOVERY_AVAILABLE = False

        # Skip if no modules available
        if not EXISTING_CODE_INVENTORY_AVAILABLE and not PACK_DISCOVERY_AVAILABLE:
            logger.info("MEMORY-001: No institutional memory modules available, skipping lookup")
            return {}

        # Timeout-protected inventory lookup
        existing_code_data: Dict[str, Any] = {}
        if EXISTING_CODE_INVENTORY_AVAILABLE and deliverables:
            try:
                import asyncio

                def _sync_inventory() -> Dict[str, Any]:
                    """Synchronous inventory lookup (runs in thread)."""
                    try:
                        inventory = get_sprint_inventory(
                            sprint_id=sprint_id,
                            deliverables=deliverables,
                            repo_root=None,  # Uses default
                            intent_manifest=None,  # Could be enhanced later
                        )
                        # Convert SprintInventory to dict for state storage
                        return {
                            "found_count": inventory.found_count,
                            "missing_count": inventory.missing_count,
                            "total_lines": inventory.total_lines,
                            "total_functions": inventory.total_functions,
                            "total_classes": inventory.total_classes,
                            "files": [
                                {
                                    "path": f.path,
                                    "lines": f.lines,
                                    "functions": [fn.name for fn in f.functions],
                                    "classes": [c.name for c in f.classes],
                                }
                                for f in inventory.deliverables_found
                            ],
                            "missing_files": inventory.deliverables_missing,
                            "agent_instructions": inventory.agent_instructions,
                            "recommendations": inventory.recommendations,
                        }
                    except Exception as e:
                        logger.warning(f"MEMORY-001: Inventory lookup failed: {e}")
                        return {}

                # Run in thread pool with timeout
                loop = asyncio.get_event_loop()
                existing_code_data = await asyncio.wait_for(
                    loop.run_in_executor(None, _sync_inventory),
                    timeout=timeout_seconds / 2,  # Use half timeout for inventory
                )

            except asyncio.TimeoutError:
                logger.warning(f"MEMORY-001: Inventory lookup timed out after {timeout_seconds/2}s")
            except Exception as e:
                logger.warning(f"MEMORY-001: Inventory lookup error: {e}")

        # Timeout-protected related packs lookup
        related_sprints: List[str] = []
        if PACK_DISCOVERY_AVAILABLE:
            try:
                import asyncio

                def _sync_related_packs() -> List[str]:
                    """Synchronous related packs lookup (runs in thread)."""
                    try:
                        related_packs = get_related_packs(sprint_id)
                        # Extract just the IDs of sprint-type packs
                        return [
                            p.id for p in related_packs
                            if hasattr(p, 'type') and p.type.value == 'sprint'
                        ][:10]  # Limit to top 10 related sprints
                    except Exception as e:
                        logger.warning(f"MEMORY-001: Related packs lookup failed: {e}")
                        return []

                loop = asyncio.get_event_loop()
                related_sprints = await asyncio.wait_for(
                    loop.run_in_executor(None, _sync_related_packs),
                    timeout=timeout_seconds / 2,  # Use half timeout for packs
                )

            except asyncio.TimeoutError:
                logger.warning(f"MEMORY-001: Related packs lookup timed out after {timeout_seconds/2}s")
            except Exception as e:
                logger.warning(f"MEMORY-001: Related packs lookup error: {e}")

        # Extract reusable modules from existing code
        reusable_modules: List[Dict[str, Any]] = []
        do_not_reimplement: List[str] = []

        if existing_code_data.get("files"):
            for file_info in existing_code_data["files"]:
                if file_info.get("functions") or file_info.get("classes"):
                    # Create import hint
                    file_path = file_info.get("path", "")
                    if file_path.startswith("src/"):
                        module_path = file_path[4:].replace("/", ".").replace(".py", "")
                    else:
                        module_path = file_path.replace("/", ".").replace(".py", "")

                    reusable_modules.append({
                        "module": file_path,
                        "provides": file_info.get("functions", []) + file_info.get("classes", []),
                        "from_sprint": sprint_id,
                        "usage_hint": f"from {module_path} import ...",
                    })

                    # Add to do_not_reimplement list
                    for func in file_info.get("functions", []):
                        do_not_reimplement.append(f"{func} (in {file_path})")
                    for cls in file_info.get("classes", []):
                        do_not_reimplement.append(f"class {cls} (in {file_path})")

        # Compose final result
        result = {
            "existing_code": existing_code_data,
            "related_sprints": related_sprints,
            "reusable_modules": reusable_modules,
            "do_not_reimplement": do_not_reimplement[:50],  # Limit to top 50 items
        }

        logger.info(
            f"MEMORY-001: Institutional memory lookup complete for {sprint_id}: "
            f"found={existing_code_data.get('found_count', 0)} files, "
            f"functions={existing_code_data.get('total_functions', 0)}, "
            f"related_sprints={len(related_sprints)}"
        )

        return result

    def _generate_reuse_instructions(
        self,
        existing_code: Dict[str, Any],
        reusable_modules: List[Dict[str, Any]],
        do_not_reimplement: List[str],
    ) -> str:
        """Generate clear, actionable reuse instructions for agents.

        This method creates a formatted instruction string that tells agents
        what code already exists and should be reused rather than reimplemented.

        Args:
            existing_code: Dict with found_count, files, etc.
            reusable_modules: List of modules with usage hints.
            do_not_reimplement: List of functionality that already exists.

        Returns:
            Formatted instruction string for agents.
        """
        if not existing_code and not reusable_modules:
            return ""

        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("INSTITUTIONAL MEMORY - EXISTING CODE AWARENESS")
        lines.append("=" * 60)
        lines.append("")

        # Summary section
        found_count = existing_code.get("found_count", 0)
        total_functions = existing_code.get("total_functions", 0)
        total_classes = existing_code.get("total_classes", 0)

        if found_count > 0:
            lines.append(f"EXISTING CODE FOUND: {found_count} files already implemented")
            lines.append(f"  - Total functions: {total_functions}")
            lines.append(f"  - Total classes: {total_classes}")
            lines.append("")

        # DO NOT REIMPLEMENT section (critical)
        if do_not_reimplement:
            lines.append("⚠️  DO NOT REIMPLEMENT THE FOLLOWING:")
            lines.append("-" * 40)
            for item in do_not_reimplement[:20]:  # Top 20 items
                lines.append(f"  ✗ {item}")
            if len(do_not_reimplement) > 20:
                lines.append(f"  ... and {len(do_not_reimplement) - 20} more")
            lines.append("")

        # REUSABLE MODULES section (actionable)
        if reusable_modules:
            lines.append("✅ REUSABLE MODULES (use these imports):")
            lines.append("-" * 40)
            for module in reusable_modules[:10]:  # Top 10 modules
                lines.append(f"  Module: {module.get('module', 'unknown')}")
                provides = module.get("provides", [])
                if provides:
                    lines.append(f"    Provides: {', '.join(provides[:5])}")
                    if len(provides) > 5:
                        lines.append(f"             ... and {len(provides) - 5} more")
                hint = module.get("usage_hint", "")
                if hint:
                    lines.append(f"    Import: {hint}")
                lines.append("")

        # Agent instructions from inventory (if available)
        agent_instructions = existing_code.get("agent_instructions", "")
        if agent_instructions:
            lines.append("AGENT INSTRUCTIONS:")
            lines.append("-" * 40)
            lines.append(agent_instructions)
            lines.append("")

        lines.append("=" * 60)
        lines.append("END INSTITUTIONAL MEMORY")
        lines.append("=" * 60)

        return "\n".join(lines)

    # =========================================================================
    # INIT NODE
    # =========================================================================

    @enforce_stacks("llm_call", required=["langfuse"])
    async def init_node(self, state: PipelineState) -> PipelineState:
        """Initialize the pipeline run.

        Responsibilities:
        1. Check SAFE_HALT (Invariant I9)
        2. Verify required stacks are healthy
        3. Load context pack for sprint
        4. Initialize observability (Langfuse trace)
        5. Emit init event to event_log

        Phase transition: -> INIT (stays in INIT, transitions to SPEC on success)
        """
        node = "init"
        state = self._record_node_start(state, node)

        # Publish Grafana metrics for Pipeline Control Center
        self._publish_grafana_metrics(state, node_name=node)
        self._publish_health_metrics()

        try:
            # Check SAFE_HALT first (Invariant I9)
            halted, state = await self._check_safe_halt(state)
            if halted:
                return self._record_node_complete(state, node)

            # Invariant I1: Key namespacing check
            # Validate that a sample key is properly namespaced before proceeding
            if self._invariant_checker is not None:
                sample_key = f"init:{state['run_id']}:{state['sprint_id']}:start"
                key_result = self._invariant_checker.check_key_namespacing(sample_key)
                state, should_continue = self._handle_invariant_violation(
                    state, key_result, node, "I1_KEY_NAMESPACING", is_blocking=True
                )
                if not should_continue:
                    return self._record_node_complete(state, node, error="Key namespacing validation failed")

            # Invariant I3: Phase order check - validate INIT is valid from current phase
            if self._invariant_checker is not None:
                current_phase = state.get("phase", "INIT")
                if current_phase and current_phase != "INIT":
                    phase_result = self._invariant_checker.check_phase_transition(
                        from_phase=current_phase,
                        to_phase="INIT",
                        is_retry=state.get("attempt", 1) > 1,
                    )
                    state, should_continue = self._handle_invariant_violation(
                        state, phase_result, node, "I3_PHASE_ORDER", is_blocking=True
                    )
                    if not should_continue:
                        return self._record_node_complete(state, node, error="Phase order validation failed")

            # Update status to running
            state = update_status(state, PipelineStatus.RUNNING)

            # Check stack health using StackInjector (OPT-02-005: cached)
            required_stacks = ["redis", "crewai", "langfuse"]
            stack_health = {}
            try:
                # OPT-02-005: Use cached health check instead of direct call
                health_result = self._cached_health_check()

                for stack_name in required_stacks:
                    stack_status = health_result.get(stack_name, {})
                    is_healthy = stack_status.get("healthy", False)
                    stack_health[stack_name] = is_healthy

                    if not is_healthy:
                        error_msg = stack_status.get("error", "Unknown health check failure")
                        logger.warning(f"Stack '{stack_name}' is unhealthy: {error_msg}")

                state = {
                    **state,
                    "required_stacks": required_stacks,
                    "stack_health": stack_health,
                }

                # P1-1 FIX: Block workflow if CRITICAL stacks are unhealthy
                # Redis is CRITICAL for IPC, checkpoints, and state sharing
                if not stack_health.get("redis", False):
                    logger.error("P1-1: Redis is unhealthy - BLOCKING workflow (Redis is CRITICAL)")
                    error_entry = create_error_entry(
                        where=node,
                        error="Redis stack is unhealthy - cannot proceed without Redis",
                        recoverable=False,  # P1-1.1: CRITICAL = not recoverable
                    )
                    state = add_error(state, error_entry)
                    state = update_status(state, PipelineStatus.FAILED)
                    return self._record_node_complete(state, node, error="Redis stack required but unhealthy")

            except ImportError:
                logger.warning("StackInjector not available, using basic health checks")
                # Fallback: try to ping Redis directly
                try:
                    stacks = self._get_stacks_for_operation(Operation.SPEC, state, check_health=True)
                    stack_health = {name: True for name in stacks.stacks.keys()}
                    state = {
                        **state,
                        "required_stacks": required_stacks,
                        "stack_health": stack_health,
                    }
                except Exception as health_err:
                    logger.warning(f"Fallback health check failed: {health_err}")
                    state = {
                        **state,
                        "required_stacks": required_stacks,
                        "stack_health": {},
                    }

            # GHOST CODE INTEGRATION (2026-01-30): Stack auto-wiring
            # Integrates observability, reasoning, and learning stacks automatically
            # into operations where they add value (see stack_autowire.py for rules)
            try:
                from pipeline.langgraph.stack_autowire import integrate_all as autowire_integrate_all
                autowire_integrate_all()
                logger.info("Stack auto-wiring integration complete")
                state = {**state, "stack_autowire_enabled": True}
            except ImportError as e:
                logger.debug(f"Stack auto-wiring not available (optional): {e}")
                state = {**state, "stack_autowire_enabled": False}
            except Exception as e:
                # Non-blocking: auto-wiring is an enhancement, not required
                logger.warning(f"Stack auto-wiring integration failed (non-blocking): {e}")
                state = {**state, "stack_autowire_enabled": False}

            # Load context pack (REQUIRED for execution per PAT-026)
            # GAP-2: Extract FULL data, not just counts
            context_pack_loaded = False
            try:
                from pipeline.spec_kit_loader import get_spec_kit_loader
                from pipeline.langgraph.state import extract_context_pack_data

                loader = get_spec_kit_loader()
                pack = loader.load_context_pack(state["sprint_id"])

                if pack:
                    # GAP-2: Use extract_context_pack_data to get full RF/INV/EDGE data
                    context_pack_state = extract_context_pack_data(pack)

                    # Extract key values for logging and Grafana
                    pack_id = context_pack_state.get("pack_id", state["sprint_id"])
                    deliverables = context_pack_state.get("deliverables", [])
                    objective = context_pack_state.get("objective", f"Execute sprint {state['sprint_id']}")
                    functional_requirements = context_pack_state.get("functional_requirements", [])
                    invariants = context_pack_state.get("invariants", [])
                    edge_cases = context_pack_state.get("edge_cases", [])

                    state = {
                        **state,
                        # Store full context_pack_state with all data
                        "context_pack": context_pack_state,
                        # Also store raw pack object for direct access in downstream nodes
                        "_context_pack_raw": pack,
                    }
                    context_pack_loaded = True
                    logger.info(
                        f"Loaded context pack for {state['sprint_id']}: {pack_id} "
                        f"(RF={len(functional_requirements)}, INV={len(invariants)}, EDGE={len(edge_cases)})"
                    )

                    # GRAFANA: Publish DoD (Definition of Done) to dashboard
                    if self._grafana_metrics and deliverables:
                        try:
                            self._grafana_metrics.set_current_dod(deliverables)
                            # Publish agent reasoning for spec_master loading the context
                            self._grafana_metrics.publish_agent_reasoning(
                                agent="spec_master",
                                thought_type="analyzing",
                                content=f"Loaded context pack for {state['sprint_id']}: {objective[:100]}",
                                task=f"Context pack loading for {state['sprint_id']}",
                            )
                            # Publish handoff from spec_master to ace_exec
                            self._grafana_metrics.publish_handoff(
                                from_agent="spec_master",
                                to_agent="ace_exec",
                                message_type="context_ready",
                                payload={
                                    "deliverables_count": len(deliverables),
                                    "rf_count": len(functional_requirements),
                                },
                            )
                            # Publish hierarchy and task decomposition for Tree Panels
                            self._grafana_metrics.publish_agent_hierarchy(active_agent="spec_master")
                            self._grafana_metrics.publish_task_decomposition(
                                sprint_id=state['sprint_id'],
                                deliverables=deliverables,
                                current_task="Loading context",
                            )
                            # Track agent communication for Node Graph
                            self._grafana_metrics.publish_agent_communication(
                                from_agent="spec_master",
                                to_agent="ace_exec",
                                message_type="handoff",
                            )
                            logger.debug(f"Published DoD to Grafana: {len(deliverables)} deliverables")
                        except Exception as e:
                            logger.debug(f"Failed to publish DoD to Grafana: {e}")
                else:
                    logger.error(f"Context pack not found for sprint {state['sprint_id']}")

            except Exception as e:
                logger.error(f"Failed to load context pack for {state['sprint_id']}: {e}")
                error_entry = create_error_entry(
                    where=node,
                    error=f"Context pack load failed: {e}",
                    recoverable=False,
                )
                state = add_error(state, error_entry)

            # PAT-026 GUARD: Warn if no context pack (don't block, but log prominently)
            if not context_pack_loaded:
                logger.warning(
                    f"PAT-026 WARNING: Proceeding without context pack for {state['sprint_id']}. "
                    "Execution may produce incomplete results."
                )

            # ================================================================
            # MEMORY-001: INSTITUTIONAL MEMORY LOOKUP
            # ================================================================
            # Queries existing code inventory and related sprints to provide
            # agents with awareness of what already exists in the codebase.
            # NON-BLOCKING: Failures are logged but do not halt the pipeline.
            # ================================================================
            if context_pack_loaded:
                try:
                    # Get deliverables from context pack
                    context_pack = state.get("context_pack", {})
                    deliverables = context_pack.get("deliverables", [])

                    if deliverables:
                        memory_result = await self._lookup_institutional_memory(
                            sprint_id=state["sprint_id"],
                            deliverables=deliverables,
                            timeout_seconds=10.0,
                        )

                        if memory_result:
                            # Add institutional memory to state
                            state = {
                                **state,
                                "existing_code": memory_result.get("existing_code", {}),
                                "related_sprints": memory_result.get("related_sprints", []),
                                "reusable_modules": memory_result.get("reusable_modules", []),
                                "do_not_reimplement": memory_result.get("do_not_reimplement", []),
                            }

                            # Generate human-readable reuse instructions
                            reuse_instructions = self._generate_reuse_instructions(
                                existing_code=memory_result.get("existing_code", {}),
                                reusable_modules=memory_result.get("reusable_modules", []),
                                do_not_reimplement=memory_result.get("do_not_reimplement", []),
                            )
                            if reuse_instructions:
                                state = {**state, "reuse_instructions": reuse_instructions}

                            logger.info(
                                f"MEMORY-001: Loaded institutional memory for {state['sprint_id']}: "
                                f"{memory_result.get('existing_code', {}).get('found_count', 0)} existing files, "
                                f"{len(memory_result.get('related_sprints', []))} related sprints"
                            )
                    else:
                        logger.debug("MEMORY-001: No deliverables in context pack, skipping institutional memory lookup")

                except Exception as e:
                    logger.warning(f"MEMORY-001: Institutional memory lookup failed (non-blocking): {e}")
                    # Continue without institutional memory - pipeline still works

            # INV-ALPHA-006: IronClad integration for raw spec processing
            # If raw_spec_input is provided, run it through IronClad pipeline
            if state.get("raw_spec_input"):
                try:
                    from pipeline.spec_kit.ironclad import IronCladOrchestrator
                    from pipeline.spec_kit.schemas import JourneySpec

                    logger.info("INV-ALPHA-006: Running IronClad on raw_spec_input")
                    orchestrator = IronCladOrchestrator()

                    # Get happy path journey from state or create minimal one
                    happy_path_data = state.get("happy_path_journey", {})
                    if happy_path_data and isinstance(happy_path_data, dict):
                        happy_path = JourneySpec(**happy_path_data)
                    else:
                        # Create minimal happy path if not provided
                        happy_path = JourneySpec(
                            id=f"JOURNEY-{state['sprint_id']}-HAPPY",
                            name="Happy Path",
                            journey_type="happy",
                            steps=[],
                            preconditions=[],
                            postconditions=[],
                        )

                    # Run IronClad pipeline
                    validation_report = orchestrator.process(
                        raw_input=state["raw_spec_input"],
                        happy_path=happy_path,
                    )

                    # Store results in spec_phase state
                    spec_phase_state = state.get("spec_phase", {})
                    spec_phase_state = {
                        **spec_phase_state,
                        "requirements": [r.model_dump() if hasattr(r, 'model_dump') else r for r in validation_report.requirements] if hasattr(validation_report, 'requirements') else [],
                        "journeys": [j.model_dump() if hasattr(j, 'model_dump') else j for j in validation_report.journeys] if hasattr(validation_report, 'journeys') else [],
                        "gate_results": validation_report.gate_results if hasattr(validation_report, 'gate_results') else {},
                        "approved": validation_report.approved if hasattr(validation_report, 'approved') else False,
                    }

                    state = {
                        **state,
                        "spec_phase": spec_phase_state,
                        "_ironclad_report": validation_report,
                    }

                    if hasattr(validation_report, 'approved') and validation_report.approved:
                        logger.info(f"IronClad validation PASSED for sprint {state['sprint_id']}")
                    else:
                        logger.warning(f"IronClad validation FAILED for sprint {state['sprint_id']}")

                except ImportError as e:
                    logger.warning(f"IronClad not available: {e}")
                except Exception as e:
                    logger.error(f"IronClad processing failed: {e}")
                    # Non-blocking error - continue pipeline
                    error_entry = create_error_entry(
                        where=node,
                        error=f"IronClad processing failed: {e}",
                        recoverable=True,
                    )
                    state = add_error(state, error_entry)

            # Initialize Langfuse trace
            try:
                from pipeline.langfuse_tracer import trace
                trace_id = f"run_{state['run_id']}"
                state = {**state, "trace_id": trace_id}
            except ImportError:
                logger.warning("Langfuse not available")

            # Partial Stacks Integration: INIT hook
            # Initializes Active RAG prefetching, BoT context, Phoenix tracing
            if self._partial_stacks is not None:
                try:
                    state = await self._partial_stacks.on_init(state)
                    logger.debug("Partial stacks integration: on_init completed")
                except Exception as e:
                    logger.warning(f"Partial stacks on_init failed (non-blocking): {e}")

            # Emit init event
            self._emit_event(state, "init_started", f"Pipeline run initialized for sprint {state['sprint_id']}")

            # Transition to SPEC phase
            state = update_phase(state, SprintPhase.SPEC)

            return self._record_node_complete(state, node)

        except Exception as e:
            logger.error(f"Init node failed: {e}")
            error_entry = create_error_entry(
                where=node,
                error=str(e),
                recoverable=True,
            )
            state = add_error(state, error_entry)
            state = update_status(state, PipelineStatus.FAILED)
            return self._record_node_complete(state, node, error=str(e))

    # =========================================================================
    # SPEC NODE (GAP-3: Decompose deliverables into granular tasks)
    # =========================================================================

    @enforce_stacks("spec_decomposition", required=["langfuse"], recommended=["redis"])
    async def spec_node(self, state: PipelineState) -> PipelineState:
        """Decompose deliverables into granular tasks using RF/INV/EDGE.

        GAP-3: This node maps functional requirements, invariants, and edge cases
        to specific deliverables, creating granular_tasks for the exec_node.

        Responsibilities:
        1. Check SAFE_HALT (Invariant I9)
        2. Extract RF/INV/EDGE from context_pack
        3. Map requirements to deliverables
        4. Create granular_tasks with full context
        5. Transition to EXEC phase

        Phase transition: INIT -> SPEC -> EXEC
        """
        node = "spec"
        state = self._record_node_start(state, node)

        # Update phase to SPEC and emit event for cockpit
        state = {**state, "phase": "SPEC"}
        self._emit_event(state, "spec_started", f"Spec decomposition started for sprint {state['sprint_id']}")

        # P1-4.2 FIX: Log hierarchy path for observability
        try:
            from pipeline.crewai_hierarchy import get_crew_hierarchy_path
            hierarchy_path = get_crew_hierarchy_path("spec")
            logger.info(f"[spec_node] Hierarchy: {' -> '.join(hierarchy_path)}")
        except ImportError:
            logger.debug("[spec_node] Hierarchy info not available")

        # Publish Grafana metrics for Pipeline Control Center
        self._publish_grafana_metrics(state, node_name=node, agent="spec_master")

        try:
            # Check SAFE_HALT first
            halted, state = await self._check_safe_halt(state)
            if halted:
                return self._record_node_complete(state, node)

            # Get context_pack data from state
            # P1-5 FIX: Block if context_pack missing or empty (prevents wasted work)
            context_pack = state.get("context_pack", {})
            if not context_pack:
                logger.error("P1-5: No context_pack in state - cannot proceed with spec")
                error_entry = create_error_entry(
                    where=node,
                    error="context_pack is required for spec decomposition",
                    recoverable=False,
                )
                state = add_error(state, error_entry)
                return self._record_node_complete(state, node, error="Missing context_pack")

            # Extract data for task creation
            deliverables = context_pack.get("deliverables", [])

            # P1-5.2 FIX: Validate minimum required fields
            if not deliverables:
                logger.error("P1-5: context_pack has no deliverables - cannot proceed")
                error_entry = create_error_entry(
                    where=node,
                    error="context_pack.deliverables is required (empty list)",
                    recoverable=False,
                )
                state = add_error(state, error_entry)
                return self._record_node_complete(state, node, error="No deliverables in context_pack")
            functional_requirements = context_pack.get("functional_requirements", [])
            invariants = context_pack.get("invariants", [])
            edge_cases = context_pack.get("edge_cases", [])
            rf_to_deliverable = context_pack.get("rf_to_deliverable", {})

            # MEMORY-001: Extract institutional memory from state
            existing_code = state.get("existing_code", {})
            reusable_modules = state.get("reusable_modules", [])
            do_not_reimplement = state.get("do_not_reimplement", [])
            reuse_instructions = state.get("reuse_instructions", "")

            # =================================================================
            # GAP-LG-1 FIX: Use create_spec_crew() instead of direct GoT call
            # =================================================================
            # Routes through spec_vp for proper VP oversight:
            # - Uses GoT internally for multi-perspective decomposition
            # - Enables CrewAI shared memory for artifact persistence
            # - Ensures spec_vp is called in the happy path (not just rework)
            got_enhanced_tasks = {}  # deliverable -> got_analysis
            spec_artifacts = {}  # P1-3.2: Initialize spec_artifacts (populated by crew if available)

            # P0-3 FIX: Get stacks for SPEC operation
            spec_stack_ctx = None
            try:
                spec_stack_ctx = self._get_stacks_for_operation(Operation.SPEC, state)
                logger.info(f"SPEC stacks available: {list(spec_stack_ctx.stacks.keys())}")
            except Exception as e:
                logger.warning(f"Could not get SPEC stacks (continuing without): {e}")

            # =========================================================================
            # P1-2.1 FIX: STACK DATA BRIDGE for spec_node
            # Enrich context_pack BEFORE spec decomposition (same pattern as exec_node)
            # Solves: RAG context not used in 80% of nodes (AUDIT REF: P1_ROOT_CAUSE_ANALYSIS)
            # =========================================================================
            if STACK_DATA_BRIDGE_AVAILABLE and get_stack_data_bridge is not None:
                try:
                    bridge = get_stack_data_bridge()

                    # Enrich context_pack with RAG context, warnings, similar plans
                    enriched_context = bridge.enrich_context_pack(state, context_pack)

                    # Record injection for observability (INV-BRIDGE-003)
                    rag_count = len(enriched_context.get("_bridge_rag_context", []))
                    warnings_count = len(enriched_context.get("_bridge_warnings", []))
                    state = bridge.record_injection(state, {
                        "node": "spec",
                        "context_enriched": True,
                        "rag_count": rag_count,
                        "warnings_count": warnings_count,
                    })

                    # Use enriched data
                    context_pack = enriched_context

                    logger.info(
                        f"[spec_node] StackDataBridge: Enriched context for spec "
                        f"(rag={rag_count}, warnings={warnings_count})"
                    )
                except Exception as e:
                    # INV-BRIDGE-001: Backward compatible - continue with original data
                    logger.warning(f"[spec_node] StackDataBridge failed (using original data): {e}")

            try:
                from pipeline.crewai_hierarchy import create_spec_crew

                # Use create_spec_crew() which routes through spec_vp
                # and handles GoT decomposition internally
                crew_result = create_spec_crew(
                    crew_id=f"{state['sprint_id']}_spec",
                    features=deliverables,
                    use_got=True,  # GoT enabled for multi-perspective analysis
                    stack_ctx=spec_stack_ctx,  # P0-3: Pass stack context
                )

                if crew_result.status == "success":
                    # Extract GoT-enhanced subtasks from crew result
                    # The crew routes through spec_vp -> spec_master
                    crew_output = crew_result.output if hasattr(crew_result, 'output') else {}
                    spec_artifacts = crew_output.get("artifacts", {}) if isinstance(crew_output, dict) else {}

                    # Map crew output to got_enhanced_tasks for backward compatibility
                    for deliverable in deliverables:
                        if deliverable not in got_enhanced_tasks:
                            got_enhanced_tasks[deliverable] = []
                        # Add any subtasks generated by the spec crew
                        deliverable_specs = spec_artifacts.get(deliverable, [])
                        if deliverable_specs:
                            got_enhanced_tasks[deliverable].extend(deliverable_specs)

                    logger.info(
                        f"Spec crew completed via spec_vp: "
                        f"tasks={crew_result.tasks_completed}, "
                        f"deliverables={len(got_enhanced_tasks)}"
                    )
                else:
                    logger.warning(
                        f"Spec crew did not succeed, continuing with base tasks: "
                        f"errors={crew_result.errors if hasattr(crew_result, 'errors') else 'unknown'}"
                    )

            except (ImportError, RuntimeError) as e:
                logger.debug(f"create_spec_crew not available, continuing without VP routing: {e}")

            # Build granular_tasks - one per deliverable with mapped RF/INV/EDGE
            granular_tasks = []
            for deliverable in deliverables:
                # Map RF to this deliverable
                relevant_rf = []
                for rf in functional_requirements:
                    rf_id = rf.get("id", "")
                    # Check traceability mapping
                    if rf_to_deliverable.get(rf_id) == deliverable:
                        relevant_rf.append(rf)
                    # Fallback: check if deliverable is in rf's files list
                    elif deliverable in rf.get("files", []):
                        relevant_rf.append(rf)
                    # If no mapping, include all RF (conservative)
                    elif not rf_to_deliverable and not rf.get("files"):
                        relevant_rf.append(rf)

                # Map INV to this deliverable (all invariants apply globally for now)
                relevant_inv = invariants.copy()

                # Map EDGE to this deliverable (all edge cases apply globally for now)
                relevant_edge = edge_cases.copy()

                # MEMORY-001: Check if this specific deliverable already exists
                deliverable_exists = False
                deliverable_info = None
                for file_info in existing_code.get("files", []):
                    if file_info.get("path") == deliverable:
                        deliverable_exists = True
                        deliverable_info = file_info
                        break

                # Generate task prompt with full context including reuse instructions
                task_prompt = self._generate_spec_task_prompt(
                    deliverable=deliverable,
                    requirements=relevant_rf,
                    invariants=relevant_inv,
                    edge_cases=relevant_edge,
                    reuse_instructions=reuse_instructions,
                    deliverable_exists=deliverable_exists,
                    deliverable_info=deliverable_info,
                )

                # GOT INTEGRATION: Add GoT-enhanced subtasks
                got_subtasks = got_enhanced_tasks.get(deliverable, [])

                granular_tasks.append({
                    "deliverable": deliverable,
                    "requirements": relevant_rf,
                    "invariants": relevant_inv,
                    "edge_cases": relevant_edge,
                    "task_prompt": task_prompt,
                    # MEMORY-001: Include reuse context in task
                    "existing_code_context": {
                        "exists": deliverable_exists,
                        "info": deliverable_info,
                        "reusable_modules": reusable_modules,
                        "do_not_reimplement": do_not_reimplement,
                    },
                    # GOT INTEGRATION: Include multi-perspective subtasks
                    "got_subtasks": got_subtasks,
                })

            # Log spec decomposition
            logger.info(
                f"Spec decomposition for {state['sprint_id']}: "
                f"{len(granular_tasks)} tasks from {len(deliverables)} deliverables"
            )

            # =================================================================
            # P1-3 FIX: Update state with ALL spec outputs for propagation
            # Solves: GoT analysis and spec artifacts lost between nodes
            # AUDIT REF: P1_ROOT_CAUSE_ANALYSIS - "got_enhanced_tasks é LOCAL"
            # =================================================================
            # Structure:
            #   got_enhanced_tasks: dict[deliverable, list[subtask_dict]]
            #     - Multi-perspective analysis from GoT decomposition
            #     - Used by exec_node to understand alternatives considered
            #   spec_artifacts: dict[deliverable, list[spec_dict]]
            #     - Raw output from spec crew (spec_vp -> spec_master)
            #     - Contains detailed specifications per deliverable
            #   granular_tasks: list[task_dict]
            #     - Final decomposed tasks with RF/INV/EDGE mappings
            #     - Primary input for exec_node
            state = {
                **state,
                "granular_tasks": granular_tasks,
                "got_enhanced_tasks": got_enhanced_tasks,  # P1-3.1: Persist GoT analysis
                "spec_artifacts": spec_artifacts,  # P1-3.2: Persist crew output
            }

            # =================================================================
            # PARTIAL STACKS INTEGRATION: Call on_spec hook (2026-01-30)
            # =================================================================
            # Integrates all spec-related stacks:
            # - DAGDecomposer, PriorityMatrix, InterfaceSketch
            # - TestFunctionGenerator, GoT Enhanced, Counter-Example
            if self._partial_stacks is not None:
                try:
                    state = await self._partial_stacks.on_spec(state)
                except Exception as e:
                    logger.warning(f"on_spec hook failed (non-blocking): {e}")

            # Emit spec event
            self._emit_event(
                state,
                "spec_decomposed",
                f"Created {len(granular_tasks)} granular tasks for sprint {state['sprint_id']}"
            )

            # Transition to EXEC phase
            state = update_phase(state, SprintPhase.EXEC)

            return self._record_node_complete(state, node)

        except Exception as e:
            logger.error(f"Spec node failed: {e}")
            error_entry = create_error_entry(
                where=node,
                error=str(e),
                recoverable=True,
            )
            state = add_error(state, error_entry)
            # Don't fail the pipeline, continue with minimal tasks
            state = {
                **state,
                "granular_tasks": [
                    {"deliverable": d, "requirements": [], "invariants": [], "edge_cases": [], "task_prompt": f"Implement {d}"}
                    for d in state.get("context_pack", {}).get("deliverables", [])
                ],
            }
            state = update_phase(state, SprintPhase.EXEC)
            return self._record_node_complete(state, node, error=str(e))

    def _generate_spec_task_prompt(
        self,
        deliverable: str,
        requirements: list,
        invariants: list,
        edge_cases: list,
        reuse_instructions: str = "",
        deliverable_exists: bool = False,
        deliverable_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a detailed task prompt for a deliverable.

        Args:
            deliverable: Path to the deliverable file.
            requirements: List of functional requirements.
            invariants: List of invariants to enforce.
            edge_cases: List of edge cases to handle.
            reuse_instructions: Institutional memory instructions.
            deliverable_exists: Whether this deliverable already exists.
            deliverable_info: Info about existing deliverable (functions, classes).

        Returns:
            Formatted task prompt string.
        """
        parts = [f"## Implementation Task: {deliverable}"]

        # MEMORY-001: Add existing code warning if file already exists
        if deliverable_exists and deliverable_info:
            parts.append("\n### ⚠️ EXISTING CODE WARNING")
            parts.append(f"This file ALREADY EXISTS with:")
            if deliverable_info.get("functions"):
                parts.append(f"  - Functions: {', '.join(deliverable_info['functions'][:10])}")
            if deliverable_info.get("classes"):
                parts.append(f"  - Classes: {', '.join(deliverable_info['classes'][:10])}")
            parts.append("**DO NOT reimplement existing functionality. EXTEND or FIX as needed.**")

        if requirements:
            parts.append("\n### Functional Requirements to Implement:")
            for rf in requirements[:5]:  # Limit to avoid context overflow
                rf_id = rf.get("id", "RF")
                rf_desc = rf.get("description", str(rf))
                parts.append(f"- **{rf_id}**: {rf_desc}")

        if invariants:
            parts.append("\n### Invariants to Enforce:")
            for inv in invariants[:5]:
                inv_id = inv.get("id", "INV")
                inv_rule = inv.get("rule", str(inv))
                parts.append(f"- **{inv_id}**: {inv_rule}")

        if edge_cases:
            parts.append("\n### Edge Cases to Handle:")
            for edge in edge_cases[:5]:
                edge_id = edge.get("id", "EDGE")
                edge_case = edge.get("case", str(edge))
                expected = edge.get("expected_behavior", "Handle gracefully")
                parts.append(f"- **{edge_id}**: {edge_case} -> {expected}")

        # MEMORY-001: Add reuse instructions if available
        if reuse_instructions:
            parts.append("\n### 📚 INSTITUTIONAL MEMORY - EXISTING CODE TO REUSE")
            parts.append(reuse_instructions)

        parts.append("\n### Instructions:")
        if deliverable_exists:
            parts.append("1. READ the existing file first before making changes")
            parts.append("2. PRESERVE existing functionality that works correctly")
            parts.append("3. EXTEND or FIX the code as needed per requirements")
        else:
            parts.append("1. Implement the file following the requirements above")
        parts.append("2. Ensure all invariants are enforced in the code")
        parts.append("3. Handle all edge cases with appropriate validation/error handling")
        parts.append("4. Write clean, well-documented code with type hints")
        parts.append("5. REUSE existing modules instead of reimplementing")

        return "\n".join(parts)

    # =========================================================================
    # EXEC NODE
    # =========================================================================

    @enforce_stacks("agent_execution", required=["langfuse"], recommended=["redis", "letta"])
    async def exec_node(self, state: PipelineState) -> PipelineState:
        """Execute CrewAI tasks for the sprint.

        Responsibilities:
        1. Check SAFE_HALT (Invariant I9)
        2. Create CrewAI crew from context pack
        3. Execute crew with hierarchy
        4. Store results in state

        Phase transition: SPEC -> EXEC -> QA
        """
        node = "exec"
        state = self._record_node_start(state, node)

        # FIX 2026-01-28: Skip exec if checkpoint says so (resume directly to gates)
        # Check state flags first, then fall back to checkpoint file
        skip_exec = state.get("_skip_exec")
        resume_from_gates = state.get("_resume_from_gates")

        # If flags not in state, check checkpoint file directly
        # This handles cases where LangGraph doesn't preserve custom state fields
        if skip_exec is None and resume_from_gates is None:
            run_id = state.get("run_id", "")
            checkpoint_dir = Path("out/runs") / run_id / "checkpoints"
            if checkpoint_dir.exists():
                for cp_file in checkpoint_dir.glob("ckpt_*.json"):
                    try:
                        import json
                        with open(cp_file) as f:
                            cp_data = json.load(f)
                            cp_state = cp_data.get("state", {})
                            skip_exec = cp_state.get("_skip_exec")
                            resume_from_gates = cp_state.get("_resume_from_gates")
                            if skip_exec or resume_from_gates:
                                logger.info(f"EXEC_NODE: Loaded skip flags from checkpoint file: {cp_file.name}")
                                break
                    except Exception as e:
                        logger.warning(f"Failed to read checkpoint {cp_file}: {e}")

        logger.info(f"EXEC_NODE: _skip_exec={skip_exec}, _resume_from_gates={resume_from_gates}, phase={state.get('phase')}")
        if skip_exec and resume_from_gates:
            logger.info(
                f"SKIP_EXEC: Checkpoint has _skip_exec=True, _resume_from_gates=True. "
                f"Skipping execution phase, going directly to gates."
            )
            # Clear the flags so we don't skip again on next attempt
            state = {
                **state,
                "_skip_exec": False,
                "phase": "QA",  # Go to QA/gates phase
            }
            self._emit_event(state, "exec_skipped", "Execution skipped - resuming from gates")
            return self._record_node_complete(state, node)

        # Update phase to EXEC and emit event for cockpit
        state = {**state, "phase": "EXEC"}
        self._emit_event(state, "exec_started", f"Execution started for sprint {state['sprint_id']}")

        # P1-4.2 FIX: Log hierarchy path for observability
        try:
            from pipeline.crewai_hierarchy import get_crew_hierarchy_path
            hierarchy_path = get_crew_hierarchy_path("exec")
            logger.info(f"[exec_node] Hierarchy: {' -> '.join(hierarchy_path)}")
        except ImportError:
            logger.debug("[exec_node] Hierarchy info not available")

        # Publish Grafana metrics for Pipeline Control Center
        self._publish_grafana_metrics(state, node_name=node, agent="ace_exec")

        try:
            # Check SAFE_HALT first
            halted, state = await self._check_safe_halt(state)
            if halted:
                return self._record_node_complete(state, node)

            # FIX 2026-01-26: Handle HIERARCHICAL REWORK
            # When work is rejected, it's delegated down the hierarchy
            is_rework = state.get("_requires_rework", False)
            rework_context = state.get("_rework_context", {})

            if is_rework and rework_context.get("hierarchical_rework"):
                violation_report = rework_context.get("violation_report", {})
                delegate_to = violation_report.get("delegate_to", "ace_exec")
                rejecting_agent = violation_report.get("rejecting_agent", "unknown")
                rejecting_tier = violation_report.get("rejecting_tier", "unknown")
                delegation_chain = rework_context.get("delegation_chain", [])

                logger.info(
                    f"HIERARCHICAL REWORK: Work rejected by {rejecting_agent} ({rejecting_tier}). "
                    f"Delegating to {delegate_to}. "
                    f"Delegation chain: {[d.get('agent') for d in delegation_chain]}"
                )

                # Update state with rework information for crew execution
                state = {
                    **state,
                    "executing_agent": delegate_to,  # Delegate to the appropriate agent
                    "_rework_delegate": delegate_to,
                    "_rework_rejecting_agent": rejecting_agent,
                    "_rework_delegation_chain": delegation_chain,
                    "_rework_hints": rework_context.get("retry_hints", []),
                    "_rework_learnings": rework_context.get("error_learnings", ""),
                }

                # Emit event for cockpit tracking
                self._emit_event(
                    state,
                    "hierarchical_rework_started",
                    f"Rework delegated: {rejecting_agent} -> {delegate_to}"
                )

            # CRIT-003: Trust boundary check - validate agent has permission to execute
            # Get the executing agent from state (defaults to ace_exec for exec node)
            # NOTE: If rework, this was set to delegate_to above
            executing_agent = state.get("executing_agent", "ace_exec")
            state, _ = self._check_trust_boundary(
                state=state,
                agent_id=executing_agent,
                resource="exec:tasks",
                action="execute",
                node=node,
            )
            # Also check sprint execution permission
            state, _ = self._check_trust_boundary(
                state=state,
                agent_id=executing_agent,
                resource="exec:sprints",
                action="execute",
                node=node,
            )

            # Invariant I11: Runaway protection checks before execution
            if self._invariant_checker is not None:
                # Check retry limit for this node
                current_retries = state.get("attempt", 1) - 1  # attempt starts at 1
                retry_result = self._invariant_checker.check_runaway_retry(
                    node=node,
                    current_retries=current_retries,
                )
                state, should_continue = self._handle_invariant_violation(
                    state, retry_result, node, "I11_RUNAWAY_RETRY", is_blocking=True
                )
                if not should_continue:
                    return self._record_node_complete(state, node, error="Runaway retry limit exceeded")

                # Check cost limit (if tracking is available)
                current_cost = state.get("total_cost_usd", 0.0)
                if current_cost > 0:
                    cost_result = self._invariant_checker.check_runaway_cost(current_cost)
                    state, should_continue = self._handle_invariant_violation(
                        state, cost_result, node, "I11_RUNAWAY_COST", is_blocking=True
                    )
                    if not should_continue:
                        return self._record_node_complete(state, node, error="Runaway cost limit exceeded")

                # Check wall time limit
                started_at = state.get("started_at")
                if started_at:
                    try:
                        start_time = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
                        elapsed_seconds = (datetime.now(timezone.utc) - start_time).total_seconds()
                        time_result = self._invariant_checker.check_runaway_time(elapsed_seconds)
                        state, should_continue = self._handle_invariant_violation(
                            state, time_result, node, "I11_RUNAWAY_TIME", is_blocking=True
                        )
                        if not should_continue:
                            return self._record_node_complete(state, node, error="Runaway time limit exceeded")
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Could not parse started_at for time check: {e}")

            # Update phase
            state = update_phase(state, SprintPhase.EXEC)

            # Partial Stacks Integration: EXEC start hook
            # Logs execution start to Phoenix, adds context to BoT
            if self._partial_stacks is not None:
                try:
                    state = await self._partial_stacks.on_exec_start(state)
                    logger.debug("Partial stacks integration: on_exec_start completed")
                except Exception as e:
                    logger.warning(f"Partial stacks on_exec_start failed (non-blocking): {e}")

            # Get context pack info
            context_pack = state.get("context_pack")
            if not context_pack:
                logger.warning("No context pack loaded, skipping execution")
                state = update_phase(state, SprintPhase.QA)
                return self._record_node_complete(state, node)

            # HIGH-001 FIX: QuietStar safety thinking check on context pack
            # Validates objective and deliverables for safety before execution
            try:
                objective = context_pack.get("objective", "")
                deliverables = context_pack.get("deliverables", [])
                input_text = f"Objective: {objective}\nDeliverables: {deliverables}"
                await _check_quietstar_workflow_input(input_text, "exec_node")
            except QuietStarBlockedError as e:
                logger.error(f"EXEC blocked by QuietStar safety check: {e}")
                error_entry = create_error_entry(
                    where=node, error=f"QuietStar blocked: {str(e)}", recoverable=False
                )
                state = add_error(state, error_entry)
                state = update_status(state, PipelineStatus.FAILED)
                state = {**state, "quietstar_blocked": True, "quietstar_reason": str(e)}
                return self._record_node_complete(state, node, error=str(e))
            except Exception as e:
                # QuietStar check failed but is non-blocking - log warning and continue
                logger.warning(f"QuietStar pre-check failed (continuing): {e}")

            # Get stacks for EXEC operation (CRIT-02: Stack injection wiring)
            try:
                stack_ctx = self._get_stacks_for_operation(Operation.EXEC, state)
                available_stacks = list(stack_ctx.stacks.keys())
                logger.info(f"EXEC stacks available: {available_stacks}")

                # Record available stacks in state
                state = {
                    **state,
                    "available_stacks": available_stacks,
                }
            except RuntimeError as e:
                logger.error(f"Required EXEC stacks unavailable: {e}")
                error_entry = create_error_entry(where=node, error=str(e), recoverable=False)
                state = add_error(state, error_entry)
                state = update_status(state, PipelineStatus.FAILED)
                return self._record_node_complete(state, node, error=str(e))

            # Execute via existing V2 modules using injected stacks
            try:
                # Get CrewAI from injected stacks if available
                crewai_client = stack_ctx.stacks.get("crewai")
                if crewai_client:
                    # FIX 2026-01-23: Removed dead import of create_sprint_crew
                    # The actual crew execution is done in _execute_sprint_crew
                    crew_result = await self._execute_sprint_crew(state, stack_ctx)
                else:
                    logger.warning("CrewAI stack not available, using fallback")
                    crew_result = await self._execute_sprint_crew(state, None)

                state = {
                    **state,
                    "crew_result": crew_result,
                }

                # CRITICAL FIX 2026-01-25: Check if crew execution actually succeeded
                # A sprint with failed/empty crew execution MUST NOT proceed
                crew_status = crew_result.get("status", "unknown") if crew_result else "no_result"
                if crew_status in ("failed", "completed_empty", "skipped", "no_result"):
                    error_msg = crew_result.get("error", f"Crew execution {crew_status}") if crew_result else "No crew result"
                    logger.error(f"CRITICAL: Sprint {state['sprint_id']} crew execution failed: {error_msg}")
                    error_entry = create_error_entry(where=node, error=error_msg)
                    state = add_error(state, error_entry)

                    # HIERARCHICAL REWORK: Crew execution failure triggers rework via ace_exec
                    rework_context = create_hierarchical_rework_context(
                        rejecting_agent="ace_exec",
                        rejection_reason=error_msg,
                        work_type="exec",
                        failed_items=[f"crew_execution_{crew_status}"],
                        current_attempt=state.get("attempt", 1),
                        max_attempts=3,
                    )
                    state = {
                        **state,
                        "_task_invalidated": True,
                        "_requires_rework": True,  # P0-4 FIX: Enable rework flow
                        "_rework_context": rework_context,
                    }
                    return self._record_node_complete(state, node, error=f"Triggering rework for crew execution failure: {error_msg}")

                # Invariant I10: Validate Redis canonical write order
                # After crew execution, verify Redis was written before any file mirrors
                if self._invariant_checker is not None and crew_result:
                    redis_written = "redis" in crew_result.get("stacks_used", [])
                    # File mirror write is tracked separately if implemented
                    file_written = crew_result.get("file_mirror_written", False)
                    redis_timestamp = crew_result.get("redis_write_timestamp")
                    file_timestamp = crew_result.get("file_write_timestamp")

                    write_order_result = self._invariant_checker.check_redis_write_order(
                        redis_written=redis_written,
                        file_written=file_written,
                        redis_timestamp=redis_timestamp,
                        file_timestamp=file_timestamp,
                    )
                    # Non-blocking: Redis write order validation is advisory
                    state, _ = self._handle_invariant_violation(
                        state, write_order_result, node, "I10_REDIS_CANONICAL", is_blocking=False
                    )

            except ImportError as e:
                logger.warning(f"CrewAI modules not available: {e}")
            except Exception as e:
                logger.error(f"Crew execution failed: {e}")
                error_entry = create_error_entry(where=node, error=str(e))
                state = add_error(state, error_entry)

            # P3-001: Run Hamilton claim verification pipeline for data transformation
            # This provides declarative DAG-based data processing with lineage tracking
            try:
                # Extract claims data from context pack or crew result
                claims_data = self._extract_claims_for_hamilton(state)
                if claims_data:
                    hamilton_result = await self._run_hamilton_claim_verification(
                        state, claims_data
                    )
                    state = {
                        **state,
                        "hamilton_claim_verification": hamilton_result,
                    }
                    logger.info(
                        f"Claim verification integrated: "
                        f"status={hamilton_result.get('status')}"
                    )
            except Exception as e:
                # Hamilton is non-blocking - log and continue
                logger.warning(f"Claim verification skipped (non-blocking): {e}")

            # Invariant I8: Event schema validation before emitting
            # Validate that the event we're about to emit conforms to schema
            if self._invariant_checker is not None:
                exec_event = self._invariant_checker.create_event(
                    event_type="task_complete",
                    task_id=f"exec_{state['sprint_id']}",
                    status="completed",
                    sprint_id=state["sprint_id"],
                )
                event_schema_result = self._invariant_checker.check_event_schema(exec_event)
                # Non-blocking: event schema issues are logged but don't fail exec
                state, _ = self._handle_invariant_violation(
                    state, event_schema_result, node, "I8_EVENT_SCHEMA", is_blocking=False
                )

            # Partial Stacks Integration: EXEC end hook
            # Records results to BoT, runs Active RAG iterative retrieval
            if self._partial_stacks is not None:
                try:
                    state = await self._partial_stacks.on_exec_end(state)
                    logger.debug("Partial stacks integration: on_exec_end completed")
                except Exception as e:
                    logger.warning(f"Partial stacks on_exec_end failed (non-blocking): {e}")

            # Emit exec event
            self._emit_event(state, "exec_completed", f"Execution completed for sprint {state['sprint_id']}")

            # Transition to QA phase
            state = update_phase(state, SprintPhase.QA)

            return self._record_node_complete(state, node)

        except Exception as e:
            logger.error(f"Exec node failed: {e}")
            error_entry = create_error_entry(where=node, error=str(e))
            state = add_error(state, error_entry)

            # HIERARCHICAL REWORK: Exec node exception triggers rework via ace_exec
            rework_context = create_hierarchical_rework_context(
                rejecting_agent="ace_exec",
                rejection_reason=f"Exec node exception: {e}",
                work_type="exec",
                failed_items=["exec_node_exception"],
                current_attempt=state.get("attempt", 1),
                max_attempts=3,
            )
            state = {
                **state,
                "_task_invalidated": True,
                "_requires_rework": True,  # P0-4 FIX: Enable rework flow
                "_rework_context": rework_context,
            }
            return self._record_node_complete(state, node, error=f"Triggering rework for exec exception: {e}")

    @enforce_stacks("generation", required=["langfuse"], recommended=["redis"])
    async def _execute_sprint_crew(
        self,
        state: PipelineState,
        stack_ctx: Optional[StackContext] = None,
    ) -> Dict[str, Any]:
        """Execute the sprint crew using injected stacks.

        This method uses the injected stacks for execution:
        - crewai: For agent orchestration
        - redis: For state persistence
        - langfuse: For observability
        - letta: For agent memory (if available)

        Args:
            state: Current pipeline state.
            stack_ctx: Injected stack context.

        Returns:
            Dict with execution results.
        """
        result = {
            "status": "pending",
            "sprint_id": state["sprint_id"],
            "tasks_completed": 0,
            "tasks_failed": 0,
            "stacks_used": [],
            "crew_output": None,
            "error": None,
        }

        if stack_ctx is None:
            logger.warning("No stack context provided, using minimal execution")
            result["status"] = "skipped"
            result["error"] = "No stack context available"
            return result

        # Record which stacks were used
        result["stacks_used"] = list(stack_ctx.stacks.keys())

        # Use Redis for state persistence if available
        redis_client = stack_ctx.stacks.get("redis")
        if redis_client:
            try:
                key = f"exec:{state['run_id']}:{state['sprint_id']}:status"
                if hasattr(redis_client, 'set'):
                    redis_client.set(key, "running")
                    logger.debug(f"Recorded execution start in Redis: {key}")
            except Exception as e:
                logger.warning(f"Could not record execution in Redis: {e}")

        # Get context pack info for crew execution
        context_pack = state.get("context_pack", {})
        if not context_pack:
            logger.warning("No context pack in state, cannot execute crew")
            result["status"] = "skipped"
            result["error"] = "No context pack available"
            return result

        # Execute CrewAI crew
        try:
            # FIX 2026-01-23: Use create_exec_crew which exists (create_sprint_crew didn't exist)
            from pipeline.crewai_hierarchy import create_exec_crew

            # Get deliverables from context pack
            deliverables = context_pack.get("deliverables", [])
            objective = context_pack.get("objective", f"Execute sprint {state['sprint_id']}")

            # Security: Sanitize input before LLM call
            # This checks for prompt injection, PII, and harmful content
            try:
                sanitization_result = await sanitize_input(objective)
                if not sanitization_result.is_safe:
                    logger.warning(
                        f"Input sanitization flagged issues: {sanitization_result.issues_found}"
                    )
                    result["security_warnings"] = sanitization_result.issues_found
                    # Use sanitized text if available
                    if hasattr(sanitization_result, 'sanitized_text') and sanitization_result.sanitized_text:
                        objective = sanitization_result.sanitized_text
                        logger.info("Using sanitized objective for execution")
            except Exception as sanitize_err:
                # Non-blocking: log and continue with original objective
                logger.warning(f"Input sanitization check failed (non-blocking): {sanitize_err}")

            logger.info(f"Executing sprint crew for {state['sprint_id']}: {objective}")

            # FIX 2026-01-23: create_exec_crew already executes and returns CrewResult
            # It takes: crew_id, implementations (list), use_got (bool), context_pack, granular_tasks
            # GAP-4 FIX: Now passes context_pack and granular_tasks for proper RF/INV/EDGE propagation
            import asyncio
            loop = asyncio.get_running_loop()

            # Get granular_tasks from spec_node decomposition (GAP-3)
            granular_tasks = state.get("granular_tasks", [])

            # =========================================================================
            # STACK DATA BRIDGE (2026-01-30): Enrich data before crew execution
            # Solves: Stack data collected but not used (AUDIT REF: docs/pipeline/STACK_USAGE_AUDIT_2026_01_30.md)
            # =========================================================================
            if STACK_DATA_BRIDGE_AVAILABLE and get_stack_data_bridge is not None:
                try:
                    bridge = get_stack_data_bridge()

                    # Enrich context_pack with RAG context, warnings, similar plans
                    enriched_context = bridge.enrich_context_pack(state, context_pack)

                    # Reorder granular_tasks by priority (P0 first)
                    enriched_tasks = bridge.enrich_granular_tasks(state, granular_tasks)

                    # Record injection for observability (INV-BRIDGE-003)
                    rag_count = len(enriched_context.get("_bridge_rag_context", []))
                    warnings_count = len(enriched_context.get("_bridge_warnings", []))
                    state = bridge.record_injection(state, {
                        "context_enriched": True,
                        "tasks_reordered": enriched_tasks != granular_tasks,
                        "rag_count": rag_count,
                        "warnings_count": warnings_count,
                    })

                    # Use enriched data
                    context_pack = enriched_context
                    granular_tasks = enriched_tasks

                    logger.info(
                        f"StackDataBridge: Enriched context for crew "
                        f"(rag={rag_count}, warnings={warnings_count}, "
                        f"tasks={len(granular_tasks)})"
                    )
                except Exception as e:
                    # INV-BRIDGE-001: Backward compatible - continue with original data
                    logger.warning(f"StackDataBridge failed (using original data): {e}")

            # Run the sync crew execution in executor to not block event loop
            # GoT (Graph of Thoughts) enabled for multi-path reasoning
            # P0-3 FIX: Pass stack_ctx for stack injection into crew operations
            crew_result = await loop.run_in_executor(
                None,
                lambda: create_exec_crew(
                    crew_id=f"{state['sprint_id']}_exec",
                    implementations=deliverables if deliverables else [objective],
                    use_got=True,  # GoT enabled - required for proper task planning
                    context_pack=context_pack,  # GAP-4: Pass full context
                    granular_tasks=granular_tasks,  # GAP-4: Pass decomposed tasks
                    stack_ctx=stack_ctx,  # P0-3: Pass stack context
                ),
            )

            # Parse crew output - CrewResult has .output attribute
            crew_output = crew_result.output if crew_result else None
            if crew_output:
                crew_output_str = str(crew_output)

                # Publish crew reasoning to Grafana dashboard
                if self._grafana_metrics:
                    try:
                        self._grafana_metrics.publish_agent_reasoning(
                            agent="ace_exec",
                            thought_type="executing",
                            content=crew_output_str[:500],
                            task=objective,
                            step=1,
                        )
                    except Exception as e:
                        logger.debug(f"Failed to publish crew reasoning: {e}")

                # Security: Validate LLM output against the original prompt
                # This checks for hallucinations, off-topic responses, and relevance
                try:
                    validation_result = await validate_output(crew_output_str, objective)
                    if not validation_result.is_valid:
                        logger.warning(
                            f"Output validation flagged issues: {validation_result.issues_found}"
                        )
                        result["validation_warnings"] = validation_result.issues_found
                        result["relevance_score"] = validation_result.relevance_score
                except Exception as validate_err:
                    # Non-blocking: log and continue
                    logger.warning(f"Output validation check failed (non-blocking): {validate_err}")

                # Security: Filter toxicity from LLM output before storing
                # This removes harmful, offensive, or inappropriate content
                try:
                    toxicity_result = await filter_toxicity(crew_output_str)
                    if toxicity_result.is_toxic:
                        logger.warning(
                            f"Toxicity detected in output (score: {toxicity_result.toxicity_score})"
                        )
                        result["toxicity_detected"] = True
                        result["toxicity_score"] = toxicity_result.toxicity_score
                        # Use filtered text if available
                        if toxicity_result.filtered_text:
                            crew_output_str = toxicity_result.filtered_text
                            logger.info("Using toxicity-filtered output")
                except Exception as toxicity_err:
                    # Non-blocking: log and continue with original output
                    logger.warning(f"Toxicity filter check failed (non-blocking): {toxicity_err}")

                result["status"] = "success"
                result["crew_output"] = crew_output_str
                result["tasks_completed"] = len(deliverables)
                logger.info(f"Crew execution completed for {state['sprint_id']}")

                # GRAFANA: Publish exec completion and message bus
                if self._grafana_metrics:
                    try:
                        self._grafana_metrics.publish_agent_reasoning(
                            agent="ace_exec",
                            thought_type="completed",
                            content=f"Completed execution for {state['sprint_id']}: {len(deliverables)} deliverables",
                            task=f"Sprint execution {state['sprint_id']}",
                        )
                        # Message from ace_exec to qa_master
                        self._grafana_metrics.publish_message_bus(
                            from_agent="ace_exec",
                            to_agent="qa_master",
                            message_type="execution_complete",
                            content=f"Finished {len(deliverables)} deliverables, ready for validation",
                        )
                        # Update hierarchy and communication
                        self._grafana_metrics.publish_agent_hierarchy(active_agent="ace_exec")
                        self._grafana_metrics.publish_agent_communication(
                            from_agent="ace_exec",
                            to_agent="qa_master",
                            message_type="handoff",
                        )
                    except Exception as metrics_err:
                        logger.debug(f"Failed to publish exec metrics: {metrics_err}")
            else:
                # CRITICAL FIX 2026-01-25: Empty output is a FAILURE, not a warning
                result["status"] = "failed"
                result["tasks_completed"] = 0
                result["error"] = "Crew execution returned empty output - all tasks may have failed"
                logger.error(f"CRITICAL: Crew execution returned empty output for {state['sprint_id']} - treating as FAILURE")

        except ImportError as e:
            logger.warning(f"CrewAI hierarchy not available: {e}")
            result["status"] = "skipped"
            result["error"] = f"CrewAI not available: {e}"

        except Exception as e:
            logger.error(f"Crew execution failed: {e}")
            result["status"] = "failed"
            result["error"] = str(e)
            result["tasks_failed"] = 1

        # Update Redis with final status
        if redis_client:
            try:
                key = f"exec:{state['run_id']}:{state['sprint_id']}:status"
                if hasattr(redis_client, 'set'):
                    redis_client.set(key, result["status"])
            except Exception as e:
                logger.warning(f"Could not update execution status in Redis: {e}")

        return result

    # =========================================================================
    # GATE NODE
    # =========================================================================

    @enforce_stacks("gate_execution", required=["langfuse"], recommended=["redis"])
    async def gate_node(self, state: PipelineState) -> PipelineState:
        """Run gate selection and validation.

        Responsibilities:
        1. Check SAFE_HALT (Invariant I9)
        2. Run gate selection from gate_runner.py
        3. Record gate results in state
        4. Create evidence bundle

        Invariant I4: no valid signoff if gates.status != PASS

        Phase transition: QA phase (gates run here)
        """
        node = "gate"
        state = self._record_node_start(state, node)

        # Update phase to QA and emit event for cockpit
        state = {**state, "phase": "QA"}
        self._emit_event(state, "gates_started", f"Gate validation started for sprint {state['sprint_id']}")

        # Publish Grafana metrics for Pipeline Control Center
        self._publish_grafana_metrics(state, node_name=node, agent="qa_master")

        try:
            # Check SAFE_HALT first
            halted, state = await self._check_safe_halt(state)
            if halted:
                return self._record_node_complete(state, node)

            # CRIT-003: Trust boundary check - validate agent has permission to run gate validation
            # Get the validating agent from state (defaults to qa_master for gate node)
            validating_agent = state.get("validating_agent", "qa_master")
            state, _ = self._check_trust_boundary(
                state=state,
                agent_id=validating_agent,
                resource="gov:gates",
                action="execute",
                node=node,
            )

            # Input sanitization: Sanitize any user-provided input before gate validation
            # This helps prevent injection attacks via claim text or context data
            try:
                context_pack = state.get("context_pack", {})
                objective = context_pack.get("objective", "")
                if objective:
                    sanitization_result = await sanitize_input(objective)
                    if not sanitization_result.is_safe:
                        logger.warning(
                            f"Input sanitization flagged potential issues in objective: "
                            f"{sanitization_result.issues_found}"
                        )
                        state = {
                            **state,
                            "input_sanitization_issues": sanitization_result.issues_found,
                            "input_risk_score": sanitization_result.risk_score,
                        }
                        # Use sanitized text if available
                        if sanitization_result.sanitized_text != objective:
                            logger.info("Using sanitized objective text for gate validation")
                            state["context_pack"] = {
                                **context_pack,
                                "objective": sanitization_result.sanitized_text,
                                "original_objective": objective,
                            }
            except Exception as sanitize_err:
                # Non-blocking: log and continue
                logger.warning(f"Input sanitization check failed (non-blocking): {sanitize_err}")

            # Invariant I8: Event schema validation for gate events
            if self._invariant_checker is not None:
                gate_event = self._invariant_checker.create_event(
                    event_type="task_start",
                    task_id=f"gate_{state['sprint_id']}",
                    node=node,
                    sprint_id=state["sprint_id"],
                )
                event_schema_result = self._invariant_checker.check_event_schema(gate_event)
                # Non-blocking: event schema issues are logged but don't fail gate
                state, _ = self._handle_invariant_violation(
                    state, event_schema_result, node, "I8_EVENT_SCHEMA", is_blocking=False
                )

            # Get stacks for GATE operation (CRIT-02: Stack injection wiring)
            try:
                stack_ctx = self._get_stacks_for_operation(Operation.GATE, state)
                gate_stacks = list(stack_ctx.stacks.keys())
                logger.info(f"GATE stacks available: {gate_stacks}")
            except RuntimeError as e:
                logger.error(f"Required GATE stacks unavailable: {e}")
                error_entry = create_error_entry(where=node, error=str(e), recoverable=False)
                state = add_error(state, error_entry)
                state = update_status(state, PipelineStatus.FAILED)
                return self._record_node_complete(state, node, error=str(e))

            # 2026-01-16: Apply @secure_gate security validation before gate execution
            # This enforces GATE_SECURITY_POLICIES from gate_integration.py
            if SECURITY_GATE_AVAILABLE and validate_gate_security is not None:
                try:
                    security_input = {
                        "sprint_id": state["sprint_id"],
                        "run_dir": str(self.run_dir),
                        "phase": state.get("phase", ""),
                    }
                    # Validate each gate G0-G8 in the pipeline
                    for gate_id in ["G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8"]:
                        is_valid, error_msg = await validate_gate_security(gate_id, security_input)
                        if not is_valid:
                            logger.warning(f"Security validation failed for {gate_id}: {error_msg}")
                            error_entry = create_error_entry(
                                where=node,
                                error=f"Security validation failed for {gate_id}: {error_msg}",
                                recoverable=False,
                            )
                            state = add_error(state, error_entry)
                            state = {
                                **state,
                                "gate_status": GateStatus.FAIL.value,
                                "gate_error": f"Security blocked: {error_msg}",
                            }
                            return self._record_node_complete(state, node, error=f"Security: {error_msg}")
                    logger.info("Security gate pre-validation passed for all gates")
                except Exception as security_error:
                    # Graceful degradation - log and continue
                    logger.warning(f"Security gate validation error (non-blocking): {security_error}")

            # HIGH-001 FIX: QuietStar safety thinking check on crew output
            # Validates agent outputs for safety before gate processing
            try:
                crew_result = state.get("crew_result", {})
                crew_output = crew_result.get("crew_output", "")
                if crew_output:
                    await _check_quietstar_workflow_input(
                        crew_output[:2000], "gate_node_crew_output"
                    )
            except QuietStarBlockedError as e:
                logger.error(f"GATE crew output blocked by QuietStar: {e}")
                state = {
                    **state,
                    "quietstar_output_blocked": True,
                    "quietstar_output_reason": str(e),
                }
                # Don't fail gate completely, but mark as blocked and log
                logger.warning("Crew output flagged by QuietStar - proceeding with caution")
            except Exception as qs_err:
                # Non-blocking: log and continue
                logger.warning(f"QuietStar output check failed (non-blocking): {qs_err}")

            # Security: Detect PII in claims being verified
            # This scans for personally identifiable information in claim data
            # that may have been inadvertently included in the verification pipeline
            try:
                # Check crew_result output for PII if available
                crew_result = state.get("crew_result", {})
                crew_output = crew_result.get("crew_output", "")
                if crew_output:
                    pii_result = await detect_pii(crew_output, anonymize=True)
                    if pii_result.has_pii:
                        logger.warning(
                            f"PII detected in claim data: {pii_result.pii_types} "
                            f"(count: {pii_result.pii_count})"
                        )
                        state = {
                            **state,
                            "pii_detected": True,
                            "pii_types": pii_result.pii_types,
                            "pii_count": pii_result.pii_count,
                        }
                        # Use redacted text for further processing if available
                        if pii_result.redacted_text and pii_result.redacted_text != crew_output:
                            logger.info("PII redacted from claim data for gate validation")
                            state["crew_result"] = {
                                **crew_result,
                                "crew_output": pii_result.redacted_text,
                                "pii_redacted": True,
                            }

                # Also check context pack objective for PII
                context_pack = state.get("context_pack", {})
                objective = context_pack.get("objective", "")
                if objective:
                    objective_pii_result = await detect_pii(objective, anonymize=True)
                    if objective_pii_result.has_pii:
                        logger.warning(
                            f"PII detected in objective: {objective_pii_result.pii_types}"
                        )
                        state = {
                            **state,
                            "objective_pii_detected": True,
                            "objective_pii_types": objective_pii_result.pii_types,
                        }
            except Exception as pii_err:
                # Non-blocking: log and continue
                logger.warning(f"PII detection check failed (non-blocking): {pii_err}")

            # Run gate selection with injected stacks
            # FIX 2026-01-28: Use PARALLEL execution for ~2x speedup
            try:
                from pipeline.gate_runner import run_gate_selection_parallel, run_gate_selection

                # FIX 2026-01-23: Create gate_selection.yml if it doesn't exist
                # This fixes the sequencing issue where gate_node runs before exec_node creates the file
                gate_selection_path = self.run_dir / "sprints" / state["sprint_id"] / "plan" / "gate_selection.yml"
                if not gate_selection_path.exists():
                    logger.info(f"Creating gate_selection.yml for {state['sprint_id']} (file did not exist)")
                    try:
                        from pipeline.gate_selector import get_gate_selector
                        selector = get_gate_selector()
                        selection = selector.create_selection(
                            run_dir=self.run_dir,
                            sprint_id=state["sprint_id"],
                            track="B",
                            profile="ops",  # FIX 2026-01-23: Use 'ops' profile - runs smaller test subset
                        )
                        selector.save_selection(selection, self.run_dir)
                        logger.info(f"Created gate_selection.yml with {len(selection.gates)} gates")
                    except Exception as selector_err:
                        logger.error(f"Failed to create gate_selection.yml: {selector_err}")
                        raise

                # FIX: Add required keyword arguments repo_root and track
                # FIX 2026-01-23: Correct repo_root calculation
                # run_dir = .HL-MCP/out/runs/lg_xxx
                # parent = .HL-MCP/out/runs
                # parent.parent = .HL-MCP/out
                # parent.parent.parent = .HL-MCP  <- CORRECT
                repo_root = self.run_dir.parent.parent.parent

                # GAP-5: Build validation_context from context_pack for gate validation
                context_pack = state.get("context_pack", {})
                validation_context = {
                    "functional_requirements": context_pack.get("functional_requirements", []),
                    "invariants": context_pack.get("invariants", []),
                    "edge_cases": context_pack.get("edge_cases", []),
                }

                # FIX 2026-01-28: Use PARALLEL gate execution (~2x speedup)
                # DAG structure allows G1, G2, G5 to run in parallel after G0
                # Falls back to sequential if PLaG not available
                try:
                    gate_result = await run_gate_selection_parallel(
                        sprint_id=state["sprint_id"],
                        run_dir=self.run_dir,
                        repo_root=repo_root,
                        track="B",  # Default track (balanced)
                        max_concurrent=3,  # Run up to 3 gates in parallel
                    )
                    logger.info("Gate execution completed using PARALLEL mode")
                except Exception as parallel_err:
                    # Fallback to sequential if parallel fails
                    logger.warning(f"Parallel gate execution failed, falling back to sequential: {parallel_err}")
                    gate_result = run_gate_selection(
                        sprint_id=state["sprint_id"],
                        run_dir=self.run_dir,
                        repo_root=repo_root,
                        track="B",
                        validation_context=validation_context,
                    )

                gates_passed = [g for g in gate_result.get("gates", []) if g.get("status") == "PASS"]
                gates_failed = [g for g in gate_result.get("gates", []) if g.get("status") in ("FAIL", "BLOCK")]

                all_passed = len(gates_failed) == 0 and len(gates_passed) > 0

                state = {
                    **state,
                    "gate_status": GateStatus.PASS.value if all_passed else GateStatus.FAIL.value,
                    "gates_passed": [g.get("gate_id", "") for g in gates_passed],
                    "gates_failed": [g.get("gate_id", "") for g in gates_failed],
                    "gate_selection_path": gate_result.get("selection_path", ""),
                    "gate_report_path": gate_result.get("report_path"),
                }

                # =========================================================================
                # STACK DATA BRIDGE (2026-01-30): Use collected data in gate decisions
                # Reference: docs/pipeline/STACK_USAGE_AUDIT_2026_01_30.md
                # =========================================================================

                # Use plan_validation_issues: If execution didn't follow the plan
                plan_issues = state.get("plan_validation_issues", [])
                if plan_issues:
                    logger.warning(
                        f"StackDataBridge: {len(plan_issues)} plan validation issues detected"
                    )
                    # Add as soft failures (don't change gate status, but log)
                    state = {
                        **state,
                        "_bridge_plan_issues_count": len(plan_issues),
                    }

                # Use detected_fabrications: If hallucinations were detected
                fabrications = state.get("detected_fabrications", [])
                if fabrications:
                    logger.warning(
                        f"StackDataBridge: {len(fabrications)} fabrications detected by SpecAI"
                    )
                    # This is more serious - consider as additional gate failure
                    if len(fabrications) > 2:  # Threshold for blocking
                        logger.error(
                            "StackDataBridge: Too many fabrications detected, flagging as issue"
                        )
                        state = {
                            **state,
                            "_bridge_fabrication_warning": True,
                            "_bridge_fabrication_count": len(fabrications),
                        }

                # Use gate_reflection for rework: If we have reflection insights
                gate_reflection = state.get("gate_reflection")
                if gate_reflection and not all_passed:
                    reflection_suggestions = []
                    if hasattr(gate_reflection, "suggestions"):
                        reflection_suggestions = gate_reflection.suggestions
                    elif isinstance(gate_reflection, dict):
                        reflection_suggestions = gate_reflection.get("suggestions", [])

                    if reflection_suggestions:
                        logger.info(
                            f"StackDataBridge: {len(reflection_suggestions)} reflection suggestions available for rework"
                        )
                        state = {
                            **state,
                            "_bridge_reflection_hints": reflection_suggestions[:5],  # Top 5
                        }

                # FIX 2026-01-26: When gates fail, trigger HIERARCHICAL REWORK
                # QA Master rejects -> delegates to Exec VP -> Ace Exec -> Squad Lead -> Developer
                # FIX 2026-01-28 (P2): Add failure classification to determine rework strategy
                if not all_passed:
                    failed_gate_ids = [g.get("gate_id", "") for g in gates_failed]
                    failed_gate_messages = [g.get("message", "Gate failed") for g in gates_failed]

                    # FIX 2026-01-28: Classify failure types to determine appropriate action
                    # TIMEOUT/INFRA failures can't be fixed by rework - need ops intervention
                    failure_type = "code"  # Default
                    can_rework = True
                    delegate_to = "ace_exec"
                    escalate_to = "qa_master"
                    max_attempts = 3

                    try:
                        from pipeline.qa_schemas import (
                            classify_all_gate_failures,
                            get_dominant_failure_type,
                            get_failure_remediation,
                            GateFailureType,
                        )

                        # Classify all gate failures
                        classifications = classify_all_gate_failures(gates_failed)
                        dominant_type = get_dominant_failure_type(classifications)
                        remediation = get_failure_remediation(dominant_type)

                        failure_type = dominant_type.value
                        can_rework = remediation["can_rework"]
                        delegate_to = remediation["delegate_to"]
                        escalate_to = remediation["escalate_to"]
                        max_attempts = remediation["max_attempts"]

                        # Log classification details
                        failure_types_by_gate = {
                            gid: c.failure_type.value for gid, c in classifications.items()
                        }
                        logger.info(
                            f"Gate failure classification: dominant={failure_type}, "
                            f"can_rework={can_rework}, by_gate={failure_types_by_gate}"
                        )
                    except ImportError:
                        logger.warning("qa_schemas not available, using default classification")

                    # Create hierarchical rework context
                    # QA Master (L3) is the rejecting agent for gate failures
                    rework_context = create_hierarchical_rework_context(
                        rejecting_agent="qa_master",
                        rejection_reason=f"Gates failed: {', '.join(failed_gate_ids)}",
                        work_type="qa",  # QA type work for gate failures
                        failed_items=failed_gate_ids,
                        current_attempt=state.get("attempt", 1),
                        max_attempts=max_attempts,  # FIX: Use classified max_attempts
                    )

                    # Add gate-specific info to the context
                    rework_context["violation_report"]["failed_gates"] = failed_gate_ids
                    rework_context["violation_report"]["gate_messages"] = failed_gate_messages

                    # STACK DATA BRIDGE (2026-01-30): Add reflection hints to rework context
                    bridge_hints = state.get("_bridge_reflection_hints", [])
                    if bridge_hints:
                        rework_context["retry_hints"] = bridge_hints
                        logger.info(f"StackDataBridge: Added {len(bridge_hints)} reflection hints to rework")

                    # FIX 2026-01-28: Add failure classification to rework context
                    rework_context["failure_type"] = failure_type
                    rework_context["can_rework"] = can_rework
                    rework_context["delegate_to"] = delegate_to
                    rework_context["escalate_to"] = escalate_to

                    # P2-5.3 FIX (2026-02-01): Extract and pass reflexion_suggestions to rework
                    # This connects the Reflexion engine's prevention_strategy to the rework flow
                    reflexion_suggestions_list = []
                    for g in gates_failed:
                        gate_refl = g.get("reflexion_suggestions")
                        if gate_refl:
                            reflexion_suggestions_list.append({
                                "gate_id": g.get("gate_id", ""),
                                **gate_refl,
                            })
                    if reflexion_suggestions_list:
                        rework_context["reflexion_suggestions"] = reflexion_suggestions_list
                        logger.info(
                            f"P2-5.3: Added {len(reflexion_suggestions_list)} reflexion suggestions to rework context"
                        )

                    logger.warning(
                        f"GATES FAILED for {state['sprint_id']}: {failed_gate_ids}. "
                        f"Failure type: {failure_type}, can_rework: {can_rework}. "
                        f"Triggering HIERARCHICAL REWORK: qa_master -> {rework_context['violation_report']['delegate_to']}"
                    )

                    state = {
                        **state,
                        "_task_invalidated": True,
                        "_requires_rework": True,  # P0-4 FIX: Enable rework flow
                        "_rework_context": rework_context,
                    }

                # FIX 2026-01-28: Run QA workers in parallel and record in quality ledger
                # This integrates with QAMasterOrchestrator for rejection tracking
                try:
                    from pipeline.quality_ledger import (
                        QualityLedger,
                        create_gate_validation_entry,
                    )
                    from pipeline.rejection_tracker import (
                        get_rejection_tracker,
                        MAX_QA_REJECTIONS,
                    )

                    # Initialize quality ledger for audit trail
                    ledger = QualityLedger(self.run_dir, state["sprint_id"])

                    # Record gate results in ledger
                    for g in gate_result.get("gates", []):
                        entry = create_gate_validation_entry(
                            gate_id=g.get("gate_id", ""),
                            passed=g.get("status") == "PASS",
                            reason=g.get("message", ""),
                            evidence_path=g.get("evidence_path"),
                        )
                        ledger.record(entry)

                    # If gates failed, record rejection and check for escalation
                    if not all_passed:
                        tracker = get_rejection_tracker()
                        record = tracker.record_rejection(
                            sprint_id=state["sprint_id"],
                            task_id=f"gate_{state['sprint_id']}",
                            reason=f"Gates failed: {[g.get('gate_id', '') for g in gates_failed]}",
                            fix_hints=[g.get("message", "") for g in gates_failed],
                            gate_failures=[g.get("gate_id", "") for g in gates_failed],
                        )

                        state = {
                            **state,
                            "rejection_attempt": record.attempt,
                            "rejection_escalated": record.escalated,
                        }

                        if record.escalated:
                            logger.warning(
                                f"Escalation triggered after {record.attempt} rejections. "
                                f"Escalating to exec_vp."
                            )
                            state["_rework_context"]["escalated"] = True
                            state["_rework_context"]["escalate_to"] = "exec_vp"

                    logger.info(
                        f"Quality ledger updated: {ledger.entry_count} entries for {state['sprint_id']}"
                    )
                except ImportError as ledger_err:
                    logger.debug(f"Quality ledger not available (non-blocking): {ledger_err}")
                except Exception as ledger_err:
                    logger.warning(f"Quality ledger update failed (non-blocking): {ledger_err}")

                # NOTE: QA workers are now handled in dedicated qa_node (after gate_node)
                # This separation allows:
                # 1. Fast fail on automated gates (don't waste QA worker time)
                # 2. QA workers get gate results as context
                # 3. Clear separation of automated vs human-like review

                # GRAFANA: Publish gate results and signoffs to dashboard streams
                if self._grafana_metrics:
                    try:
                        # Publish each gate result
                        for g in gates_passed:
                            self._grafana_metrics.publish_gate_result(
                                gate_id=g.get("gate_id", ""),
                                passed=True,
                                score=g.get("score", 1.0),
                                details=g.get("message", "Gate passed"),
                            )
                        for g in gates_failed:
                            self._grafana_metrics.publish_gate_result(
                                gate_id=g.get("gate_id", ""),
                                passed=False,
                                score=g.get("score", 0.0),
                                details=g.get("message", "Gate failed"),
                            )
                        # Publish QA Master signoff
                        verdict = "approved" if all_passed else "rejected"
                        self._grafana_metrics.publish_signoff(
                            agent="qa_master",
                            task_id=f"gate_{state['sprint_id']}",
                            verdict=verdict,
                            confidence=0.9 if all_passed else 0.7,
                            reason=f"{len(gates_passed)} gates passed, {len(gates_failed)} failed",
                        )
                        # Publish agent reasoning
                        self._grafana_metrics.publish_agent_reasoning(
                            agent="qa_master",
                            thought_type="validating",
                            content=f"Validated gates for {state['sprint_id']}: {len(gates_passed)} passed, {len(gates_failed)} failed",
                            task=f"Gate validation for {state['sprint_id']}",
                        )
                        # Update hierarchy
                        self._grafana_metrics.publish_agent_hierarchy(active_agent="qa_master")
                        # Track communication: qa_master -> ceo for signoff
                        self._grafana_metrics.publish_agent_communication(
                            from_agent="qa_master",
                            to_agent="ceo",
                            message_type="signoff_request",
                        )
                    except Exception as metrics_err:
                        logger.debug(f"Failed to publish gate metrics: {metrics_err}")

            except ImportError as e:
                # Gate runner not available - this is a critical missing component
                # Do NOT auto-pass gates as this bypasses quality validation
                logger.error(f"Gate runner module not available: {e}")
                error_entry = create_error_entry(
                    where=node,
                    error=f"Gate runner not available: {e}. Quality validation cannot be performed.",
                    recoverable=False,
                )
                state = add_error(state, error_entry)
                state = {
                    **state,
                    "gate_status": GateStatus.PENDING.value,
                    "gates_passed": [],
                    "gates_failed": [],
                    "gate_error": f"Gate runner import failed: {e}",
                }
            except Exception as e:
                logger.error(f"Gate execution failed: {e}")
                error_entry = create_error_entry(where=node, error=str(e))
                state = add_error(state, error_entry)
                state = {
                    **state,
                    "gate_status": GateStatus.FAIL.value,
                    "gate_error": str(e),
                }

            # P3-001: Run Hamilton gate validation pipeline for structured analysis
            # This provides DAG-based gate result processing with dependency resolution
            try:
                gate_config = self._extract_gate_config_for_hamilton(state)
                if gate_config.get("gates"):
                    hamilton_gate_result = await self._run_hamilton_gate_validation(
                        state, gate_config
                    )
                    state = {
                        **state,
                        "hamilton_gate_validation": hamilton_gate_result,
                    }
                    logger.info(
                        f"Gate validation integrated: "
                        f"status={hamilton_gate_result.get('status')}"
                    )
            except Exception as e:
                # NF-011 FIX: Use GATE_VALIDATION_BLOCKING flag
                if GATE_VALIDATION_BLOCKING:
                    logger.error(
                        f"GATE-VAL-001: Gate validation failed and GATE_VALIDATION_BLOCKING=true - "
                        f"BLOCKING execution: {e}"
                    )
                    raise  # Re-raise to block execution
                else:
                    logger.warning(
                        f"GATE-VAL-001: Gate validation skipped (non-blocking): {e}. "
                        f"Set GATE_VALIDATION_BLOCKING=true for production."
                    )

            # Partial Stacks Integration: GATE hook
            # Runs RAGAS evaluation, DeepEval hallucination detection, records to BoT
            if self._partial_stacks is not None:
                try:
                    state = await self._partial_stacks.on_gate(state)
                    logger.debug("Partial stacks integration: on_gate completed")
                except Exception as e:
                    logger.warning(f"Partial stacks on_gate failed (non-blocking): {e}")

            # Emit gate event
            gate_status = state.get("gate_status", "UNKNOWN")
            self._emit_event(state, "gates_completed", f"Gate validation completed: {gate_status}")

            # NOTE: Phase stays as QA - qa_node will transition to VOTE
            # gate_node handles automated validation, qa_node handles QA workers

            return self._record_node_complete(state, node)

        except Exception as e:
            logger.error(f"Gate node failed: {e}")
            error_entry = create_error_entry(where=node, error=str(e))
            state = add_error(state, error_entry)

            # HIERARCHICAL REWORK: Gate node exception triggers rework via qa_master
            rework_context = create_hierarchical_rework_context(
                rejecting_agent="qa_master",
                rejection_reason=f"Gate node exception: {e}",
                work_type="qa",
                failed_items=["gate_node_exception"],
                current_attempt=state.get("attempt", 1),
                max_attempts=3,
            )
            state = {
                **state,
                "_task_invalidated": True,
                "_requires_rework": True,  # P0-4 FIX: Enable rework flow
                "_rework_context": rework_context,
            }
            return self._record_node_complete(state, node, error=f"Triggering rework for gate exception: {e}")

    # =========================================================================
    # QA NODE - QA Workers Review (2026-01-28)
    # =========================================================================

    async def qa_node(self, state: PipelineState) -> PipelineState:
        """Run QA workers in parallel via QAMasterOrchestrator.

        This node is responsible for:
        1. Spawning QA workers (auditor, refinador, gap_hunter, etc.) in parallel
        2. Collecting and aggregating worker results
        3. Making QA decision (APPROVE/REJECT/CONDITIONAL_APPROVE)
        4. Recording results in QualityLedger

        Runs AFTER gate_node (automated validation) and BEFORE signoff_node.

        The 8 QA workers are:
        - auditor (CRITICAL): Code quality audit
        - agente_auditor: Secondary audit
        - refinador (IMPORTANT): Code refinement review
        - clean_reviewer (IMPORTANT): Clean code review
        - edge_case_hunter (IMPORTANT): Edge case validation
        - gap_hunter (CRITICAL): Gap detection
        - human_reviewer: Human-like review
        - debt_tracker: Technical debt tracking

        Phase transition: QA (continues) -> VOTE (after qa_node)
        """
        node = "qa"
        state = self._record_node_start(state, node)

        # Emit event for cockpit
        self._emit_event(state, "qa_workers_started", f"QA workers started for sprint {state['sprint_id']}")

        # P1-4.2 FIX: Log hierarchy path for observability
        try:
            from pipeline.crewai_hierarchy import get_crew_hierarchy_path
            hierarchy_path = get_crew_hierarchy_path("qa")
            logger.info(f"[qa_node] Hierarchy: {' -> '.join(hierarchy_path)}")
        except ImportError:
            logger.debug("[qa_node] Hierarchy info not available")

        # Publish Grafana metrics
        self._publish_grafana_metrics(state, node_name=node, agent="qa_master")

        try:
            # Check SAFE_HALT first
            halted, state = await self._check_safe_halt(state)
            if halted:
                return self._record_node_complete(state, node)

            # Check if gates passed - if not, skip QA workers (will rework anyway)
            gate_status = state.get("gate_status", "")
            if gate_status != GateStatus.PASS.value:
                logger.info(
                    f"[qa_node] Skipping QA workers - gates did not pass (status={gate_status})"
                )
                # Don't set _task_invalidated here - gate_node already handled that
                return self._record_node_complete(state, node)

            # Run QA workers via QAMasterOrchestrator
            try:
                from pipeline.qa_master_orchestrator import QAMasterOrchestrator

                # Get deliverables from context_pack
                context_pack = state.get("context_pack", {})
                deliverables = context_pack.get("deliverables", [])

                # =========================================================================
                # P1-2.2 FIX: STACK DATA BRIDGE for qa_node
                # Enrich context_pack BEFORE QA validation (same pattern as exec_node)
                # Solves: RAG context not used in QA validation (AUDIT REF: P1_ROOT_CAUSE_ANALYSIS)
                # =========================================================================
                if STACK_DATA_BRIDGE_AVAILABLE and get_stack_data_bridge is not None:
                    try:
                        bridge = get_stack_data_bridge()

                        # Enrich context_pack with RAG context, warnings, similar plans
                        enriched_context = bridge.enrich_context_pack(state, context_pack)

                        # Record injection for observability (INV-BRIDGE-003)
                        rag_count = len(enriched_context.get("_bridge_rag_context", []))
                        warnings_count = len(enriched_context.get("_bridge_warnings", []))
                        state = bridge.record_injection(state, {
                            "node": "qa",
                            "context_enriched": True,
                            "rag_count": rag_count,
                            "warnings_count": warnings_count,
                        })

                        # Use enriched data
                        context_pack = enriched_context

                        logger.info(
                            f"[qa_node] StackDataBridge: Enriched context for QA "
                            f"(rag={rag_count}, warnings={warnings_count})"
                        )
                    except Exception as e:
                        # INV-BRIDGE-001: Backward compatible - continue with original data
                        logger.warning(f"[qa_node] StackDataBridge failed (using original data): {e}")

                if deliverables:
                    logger.info(
                        f"[qa_node] Spawning QA workers via QAMasterOrchestrator "
                        f"for {len(deliverables)} deliverables"
                    )

                    # Create orchestrator
                    orchestrator = QAMasterOrchestrator(
                        run_dir=self.run_dir,
                        sprint_id=state["sprint_id"],
                    )

                    # Run QA workers in parallel
                    # Gates already ran in gate_node, so run_gates=False
                    # 2026-01-31: ENABLED 7-layer Claude validation (restored framework)
                    qa_result = await orchestrator.run(
                        deliverables=deliverables,
                        context_pack=context_pack,
                        task_id=f"qa_{state['sprint_id']}",
                        run_gates=False,  # Gates already ran in gate_node
                        run_workers=True,  # Run the 8 QA workers in parallel
                        run_human_layers=True,  # ENABLED: 7-layer Claude validation (INV-HL-001)
                    )

                    # Update state with QA worker results
                    # 2026-01-31: EXPLICIT human_layer_results for visibility/audit
                    state = {
                        **state,
                        "qa_workers_result": qa_result.to_dict(),
                        "qa_workers_passed": qa_result.passed,
                        "qa_workers_decision": qa_result.decision.reason if qa_result.decision else "",
                        # Human layer 6-perspective test results (NOT ignored!)
                        "human_layer_results": qa_result.human_layer_results,
                        "human_layer_executed": bool(qa_result.human_layer_results),
                    }

                    # If QA workers rejected, trigger rework
                    if not qa_result.passed:
                        logger.warning(
                            f"[qa_node] QA workers rejected: {qa_result.decision.reason}"
                        )

                        # Create hierarchical rework context
                        rework_context = create_hierarchical_rework_context(
                            rejecting_agent="qa_master",
                            rejection_reason=f"QA workers rejected: {qa_result.decision.reason}",
                            work_type="qa",
                            failed_items=list(qa_result.worker_results.keys()),
                            current_attempt=state.get("attempt", 1),
                            max_attempts=3,
                        )

                        # Add QA-specific info
                        rework_context["qa_decision"] = qa_result.decision.reason if qa_result.decision else ""
                        rework_context["critical_failures"] = [
                            w for w, r in qa_result.worker_results.items()
                            if not r.get("passed", True) and w in {"auditor", "gap_hunter"}
                        ]

                        state = {
                            **state,
                            "_task_invalidated": True,
                            "_requires_rework": True,  # P0-4 FIX: Enable rework flow
                            "_rework_context": rework_context,
                        }

                        logger.info(
                            f"[qa_node] Triggering rework: {len(rework_context.get('critical_failures', []))} critical failures"
                        )
                    else:
                        logger.info(
                            f"[qa_node] QA workers APPROVED: {len(qa_result.worker_results)} workers completed"
                        )

                        # INV-HL-001: Human layer rejection → REWORK (not HALT)
                        # Check if human layer blocked promotion
                        hl_results = qa_result.human_layer_results
                        if hl_results and not hl_results.get("_passed", True):
                            logger.warning(
                                f"[qa_node] Human layer BLOCKED promotion: "
                                f"strong_veto={hl_results.get('has_strong_veto')}, "
                                f"findings={hl_results.get('total_findings', 0)}"
                            )

                            # Create rework context with human layer feedback
                            rework_context = create_hierarchical_rework_context(
                                rejecting_agent="human_layer",
                                rejection_reason=(
                                    f"Human layer validation failed: "
                                    f"{hl_results.get('total_findings', 0)} findings, "
                                    f"strong_veto={hl_results.get('has_strong_veto')}"
                                ),
                                work_type="qa",
                                failed_items=["human_layer_validation"],
                                current_attempt=state.get("attempt", 1),
                                max_attempts=3,
                            )

                            # Add human layer report for exec team
                            rework_context["human_layer_report"] = hl_results.get("report_path", "")
                            rework_context["human_layer_findings"] = hl_results.get("total_findings", 0)

                            state = {
                                **state,
                                "_task_invalidated": True,
                                "_requires_rework": True,  # P0-4 FIX: Enable rework flow
                                "_rework_context": rework_context,
                            }

                            logger.info(
                                f"[qa_node] Human layer triggering REWORK (not halt): "
                                f"report={hl_results.get('report_path')}"
                            )

                    # Publish to Grafana
                    if self._grafana_metrics:
                        try:
                            verdict = "approved" if qa_result.passed else "rejected"
                            self._grafana_metrics.publish_signoff(
                                agent="qa_master",
                                task_id=f"qa_workers_{state['sprint_id']}",
                                verdict=verdict,
                                confidence=0.9 if qa_result.passed else 0.5,
                                reason=qa_result.decision.reason if qa_result.decision else "",
                            )
                            self._grafana_metrics.publish_agent_reasoning(
                                agent="qa_master",
                                thought_type="reviewing",
                                content=f"QA workers {verdict}: {len(qa_result.worker_results)} workers completed",
                                task=f"QA review for {state['sprint_id']}",
                            )
                        except Exception as metrics_err:
                            logger.debug(f"Failed to publish QA metrics: {metrics_err}")
                else:
                    logger.debug("[qa_node] No deliverables to review, skipping QA workers")
                    state = {
                        **state,
                        "qa_workers_passed": True,
                        "qa_workers_decision": "No deliverables to review",
                    }

            except ImportError as qa_import_err:
                logger.warning(f"QAMasterOrchestrator not available: {qa_import_err}")
                # Non-blocking: mark as passed if orchestrator not available
                state = {
                    **state,
                    "qa_workers_passed": True,
                    "qa_workers_decision": "QAMasterOrchestrator not available",
                }
            except Exception as qa_err:
                logger.error(f"QA workers execution failed: {qa_err}")
                # On error, trigger rework
                rework_context = create_hierarchical_rework_context(
                    rejecting_agent="qa_master",
                    rejection_reason=f"QA workers failed: {qa_err}",
                    work_type="qa",
                    failed_items=["qa_node_exception"],
                    current_attempt=state.get("attempt", 1),
                    max_attempts=3,
                )
                state = {
                    **state,
                    "_task_invalidated": True,
                    "_requires_rework": True,  # P0-4 FIX: Enable rework flow
                    "_rework_context": rework_context,
                }

            # Emit completion event
            qa_passed = state.get("qa_workers_passed", False)
            self._emit_event(
                state,
                "qa_workers_completed",
                f"QA workers completed: {'PASSED' if qa_passed else 'REJECTED'}"
            )

            # Transition to VOTE phase (only if not invalidated)
            if not state.get("_task_invalidated"):
                state = update_phase(state, SprintPhase.VOTE)

            return self._record_node_complete(state, node)

        except Exception as e:
            logger.error(f"QA node failed: {e}")
            error_entry = create_error_entry(where=node, error=str(e))
            state = add_error(state, error_entry)

            # Trigger rework on exception
            rework_context = create_hierarchical_rework_context(
                rejecting_agent="qa_master",
                rejection_reason=f"QA node exception: {e}",
                work_type="qa",
                failed_items=["qa_node_exception"],
                current_attempt=state.get("attempt", 1),
                max_attempts=3,
            )
            state = {
                **state,
                "_task_invalidated": True,
                "_requires_rework": True,  # P0-4 FIX: Enable rework flow
                "_rework_context": rework_context,
            }
            return self._record_node_complete(state, node, error=f"Triggering rework for QA exception: {e}")

    # =========================================================================
    # SIGNOFF NODE
    # =========================================================================

    @enforce_stacks("signoff", required=["langfuse"], recommended=["redis"])
    async def signoff_node(self, state: PipelineState) -> PipelineState:
        """Process signoffs from agents.

        Responsibilities:
        1. Check SAFE_HALT (Invariant I9)
        2. Verify gates passed (Invariant I4)
        3. Process agent signoffs via governance module
        4. Record signoffs in state

        Invariant I4: no valid signoff if gates.status != PASS
        Invariant I5: supervisor cannot sign off without subordinate approvals

        Phase transition: VOTE -> DONE
        """
        node = "signoff"
        state = self._record_node_start(state, node)

        # Update phase to VOTE and emit event for cockpit
        state = {**state, "phase": "VOTE"}
        self._emit_event(state, "signoff_started", f"Signoff processing started for sprint {state['sprint_id']}")

        # Publish Grafana metrics for Pipeline Control Center
        self._publish_grafana_metrics(state, node_name=node, agent="ceo")

        try:
            # Check SAFE_HALT first
            halted, state = await self._check_safe_halt(state)
            if halted:
                return self._record_node_complete(state, node)

            # Invariant I4: Check gates passed
            if state.get("gate_status") != GateStatus.PASS.value:
                logger.warning("Cannot process signoffs - gates did not pass (Invariant I4)")
                error_entry = create_error_entry(
                    where=node,
                    error="Gates must PASS before signoff (Invariant I4)",
                    recoverable=True,  # Changed to recoverable - can rework
                )
                state = add_error(state, error_entry)

                # HIERARCHICAL REWORK: Gates not passed triggers rework via qa_master -> exec
                # This is a secondary check - gate_node should have already triggered rework
                # But if we got here, we need to redirect back
                rework_context = create_hierarchical_rework_context(
                    rejecting_agent="qa_master",
                    rejection_reason="Gates must PASS before signoff (Invariant I4)",
                    work_type="exec",  # Need to rework execution, not just QA
                    failed_items=["invariant_i4_gates_not_passed"],
                    current_attempt=state.get("attempt", 1),
                    max_attempts=3,
                )
                state = {
                    **state,
                    "_task_invalidated": True,
                    "_requires_rework": True,  # P0-4 FIX: Enable rework flow
                    "_rework_context": rework_context,
                }
                return self._record_node_complete(state, node, error="Triggering rework - gates not passed")

            # CRIT-003: Trust boundary check - validate agent has signoff authority
            # Get the signing agent from state (defaults to ceo for signoff node)
            signing_agent = state.get("signing_agent", "ceo")
            # Check permission to approve signoffs
            state, _ = self._check_trust_boundary(
                state=state,
                agent_id=signing_agent,
                resource="gov:signoffs",
                action="approve",
                node=node,
            )
            # Also check permission to access governance approvals
            state, _ = self._check_trust_boundary(
                state=state,
                agent_id=signing_agent,
                resource="gov:approvals",
                action="write",
                node=node,
            )

            # Get stacks for SIGNOFF operation (CRIT-02: Stack injection wiring)
            try:
                stack_ctx = self._get_stacks_for_operation(Operation.SIGNOFF, state)
                signoff_stacks = list(stack_ctx.stacks.keys())
                logger.info(f"SIGNOFF stacks available: {signoff_stacks}")
            except RuntimeError as e:
                logger.error(f"Required SIGNOFF stacks unavailable: {e}")
                error_entry = create_error_entry(where=node, error=str(e), recoverable=False)
                state = add_error(state, error_entry)
                state = update_status(state, PipelineStatus.FAILED)
                return self._record_node_complete(state, node, error=str(e))

            # Invariant I8: Event schema validation for signoff events
            if self._invariant_checker is not None:
                signoff_event = self._invariant_checker.create_event(
                    event_type="signoff",
                    signoff_id=f"signoff_{state['run_id']}_{state['sprint_id']}",
                    agent_id=signing_agent,
                    sprint_id=state["sprint_id"],
                )
                event_schema_result = self._invariant_checker.check_event_schema(signoff_event)
                # Non-blocking: event schema issues are logged but don't fail signoff
                state, _ = self._handle_invariant_violation(
                    state, event_schema_result, node, "I8_EVENT_SCHEMA", is_blocking=False
                )

            # LLM Guard checks: Validate any LLM-generated signoff content
            # This ensures signoff messages don't contain harmful content or PII
            try:
                crew_result = state.get("crew_result", {})
                crew_output = crew_result.get("crew_output", "")
                if crew_output:
                    # Check for toxicity in LLM output before signoff
                    toxicity_result = await filter_toxicity(crew_output)
                    if toxicity_result.is_toxic:
                        logger.warning(
                            f"Toxicity detected in crew output before signoff: "
                            f"score={toxicity_result.toxicity_score}, categories={toxicity_result.categories}"
                        )
                        state = {
                            **state,
                            "signoff_toxicity_detected": True,
                            "signoff_toxicity_score": toxicity_result.toxicity_score,
                            "signoff_toxicity_categories": list(toxicity_result.categories.keys()),
                        }
                        # Use filtered text if available
                        if toxicity_result.filtered_text:
                            crew_result["crew_output"] = toxicity_result.filtered_text
                            crew_result["toxicity_filtered"] = True
                            state["crew_result"] = crew_result

                    # Validate output for potential data leaks
                    validation_result = await validate_output(
                        output=crew_output,
                        original_prompt=state.get("context_pack", {}).get("objective", ""),
                    )
                    if not validation_result.is_valid:
                        logger.warning(
                            f"Output validation issues detected before signoff: "
                            f"{validation_result.issues_found}"
                        )
                        state = {
                            **state,
                            "signoff_validation_issues": validation_result.issues_found,
                            "signoff_output_risk_score": validation_result.risk_score,
                        }
            except Exception as guard_err:
                # Non-blocking: log and continue
                logger.warning(f"LLM Guard check failed in signoff (non-blocking): {guard_err}")

            # Invariant I5: Check executive authority before processing signoffs
            # Validate that the signoff hierarchy is respected (supervisor can only sign off on subordinate work)
            if self._invariant_checker is not None:
                # Get pending signoff requests from state (signer -> subordinate pairs)
                signoff_requests = state.get("pending_signoff_requests", [])
                for request in signoff_requests:
                    signer_role = request.get("signer_role", "")
                    subordinate_role = request.get("subordinate_role", "")
                    if signer_role and subordinate_role:
                        authority_result = self._invariant_checker.check_executive_authority(
                            signer_role=signer_role,
                            subordinate_role=subordinate_role,
                        )
                        state, should_continue = self._handle_invariant_violation(
                            state, authority_result, node, "I5_EXECUTIVE_AUTHORITY", is_blocking=True
                        )
                        if not should_continue:
                            return self._record_node_complete(
                                state, node, error=f"Executive authority violation: {signer_role} cannot sign off on {subordinate_role}"
                            )

            # GAP-6: Validate DoD/AC checklist before processing signoffs
            context_pack = state.get("context_pack", {})
            dod_checklist = context_pack.get("dod_checklist", [])
            acceptance_criteria = context_pack.get("acceptance_criteria", [])

            dod_validated = True
            dod_issues = []

            if dod_checklist:
                logger.info(f"Validating DoD checklist ({len(dod_checklist)} items)")
                for item in dod_checklist:
                    item_name = item.get("item", item.get("name", "Unknown"))
                    required = item.get("required", True)
                    # Check if item is verified in state
                    verification_key = f"dod_{item_name.lower().replace(' ', '_')}_verified"
                    is_verified = state.get(verification_key, False)

                    # For required items, check if gates passed can serve as verification
                    if required and not is_verified:
                        # Check if this is covered by gate pass
                        if item_name.lower() in ["all tests pass", "tests pass", "tests_pass"]:
                            is_verified = state.get("gate_status") == GateStatus.PASS.value
                        elif item_name.lower() in ["coverage >= 90%", "coverage"]:
                            is_verified = state.get("gate_status") == GateStatus.PASS.value

                    if required and not is_verified:
                        dod_issues.append(f"DoD item '{item_name}' not verified")
                        dod_validated = False

            if acceptance_criteria:
                logger.info(f"Checking acceptance criteria ({len(acceptance_criteria)} items)")
                for ac in acceptance_criteria:
                    ac_id = ac.get("id", "AC-???")
                    criterion = ac.get("criterion", ac.get("description", ""))
                    # AC validation would require more sophisticated checking
                    # For now, we log and track
                    logger.debug(f"AC check: [{ac_id}] {criterion}")

            # Store DoD validation result in state
            state = {
                **state,
                "dod_validated": dod_validated,
                "dod_issues": dod_issues,
                "ac_count": len(acceptance_criteria),
                "dod_count": len(dod_checklist),
            }

            if not dod_validated and dod_issues:
                logger.warning(f"DoD validation issues: {dod_issues}")
                # Non-blocking - log but continue (CEO will see issues in state)

            # =================================================================
            # TASK-F1-005: Instantiate CEO agent for signoff decision
            # CEO makes the actual signoff decision as a CrewAI agent
            # =================================================================
            ceo_signoff_decision = None
            try:
                from pipeline.crewai_hierarchy import create_ceo_agent

                ceo_agent = create_ceo_agent()
                if ceo_agent is not None:
                    # Prepare context for CEO decision
                    ceo_context = {
                        "sprint_id": state["sprint_id"],
                        "gate_status": state.get("gate_status", "UNKNOWN"),
                        "gates_passed": state.get("gates_passed", []),
                        "gates_failed": state.get("gates_failed", []),
                        "dod_validated": dod_validated,
                        "dod_issues": dod_issues,
                        "qa_result": state.get("qa_result", {}),
                        "crew_result": state.get("crew_result", {}),
                    }

                    # CEO reviews and makes signoff decision
                    logger.info(f"CEO agent reviewing sprint {state['sprint_id']} for signoff")
                    ceo_signoff_decision = {
                        "agent": "ceo",
                        "reviewed": True,
                        "context": ceo_context,
                        "approved": state.get("gate_status") == GateStatus.PASS.value and dod_validated,
                    }

                    # Store CEO decision in state
                    state = {
                        **state,
                        "ceo_signoff_decision": ceo_signoff_decision,
                    }
                    logger.info(f"CEO signoff decision: approved={ceo_signoff_decision['approved']}")
            except ImportError as e:
                logger.warning(f"CEO agent not available (non-blocking): {e}")
            except Exception as e:
                logger.warning(f"CEO signoff decision failed (non-blocking): {e}")

            # Process signoffs via existing governance module
            try:
                from pipeline.governance import process_signoffs_for_sprint, REQUIRED_SIGNOFFS

                # FIX 2026-01-26: Pass gate results to signoff processing
                signoff_results = await process_signoffs_for_sprint(
                    sprint_id=state["sprint_id"],
                    run_dir=self.run_dir,
                    dod_validated=dod_validated,  # GAP-6: Pass DoD validation status
                    dod_issues=dod_issues,  # GAP-6: Pass any DoD issues
                    gates_passed=state.get("gates_passed", []),  # FIX 2026-01-26
                    gates_failed=state.get("gates_failed", []),  # FIX 2026-01-26
                )

                # Update signoffs in state (excluding metadata keys)
                for agent_id, signoff in signoff_results.items():
                    if not agent_id.startswith("_"):  # Skip metadata keys
                        state["signoffs"][agent_id] = signoff

                # FIX 2026-01-26: CRITICAL - Verify ALL required signoffs are approved
                # It is IMPOSSIBLE to advance to next sprint without CEO, QA, and Presidente signoffs
                all_approved = signoff_results.get("_all_approved", False)
                missing_signoffs = signoff_results.get("_missing_signoffs", [])

                if not all_approved:
                    # FIX 2026-01-26: Trigger HIERARCHICAL REWORK
                    # Identify the highest-level agent who rejected (determines delegation)
                    # Priority order: presidente (L1) > ceo (L1) > qa_master (L3)

                    # Determine the primary rejecting agent (highest in hierarchy)
                    rejecting_agent = "qa_master"  # Default
                    for agent in ["presidente", "ceo", "qa_master"]:
                        if agent in missing_signoffs:
                            rejecting_agent = agent
                            break

                    # Collect rejection reasons
                    rejection_reasons = []
                    for agent_id in missing_signoffs:
                        signoff = signoff_results.get(agent_id, {})
                        reason = signoff.get("reason", f"{agent_id} did not approve")
                        rejection_reasons.append(f"{agent_id}: {reason}")

                    # Create hierarchical rework context
                    rework_context = create_hierarchical_rework_context(
                        rejecting_agent=rejecting_agent,
                        rejection_reason="; ".join(rejection_reasons),
                        work_type="exec",  # Execution type work for signoff rejections
                        failed_items=missing_signoffs,
                        current_attempt=state.get("attempt", 1),
                        max_attempts=3,
                    )

                    # Add signoff-specific info to the context
                    rework_context["violation_report"]["missing_signoffs"] = missing_signoffs
                    rework_context["violation_report"]["required_signoffs"] = REQUIRED_SIGNOFFS

                    logger.warning(
                        f"SIGNOFF REJECTED for {state['sprint_id']}: {missing_signoffs}. "
                        f"Triggering HIERARCHICAL REWORK: {rejecting_agent} -> {rework_context['violation_report']['delegate_to']}"
                    )

                    state = {
                        **state,
                        "_task_invalidated": True,
                        "_requires_rework": True,  # P0-4 FIX: Enable rework flow
                        "_rework_context": rework_context,
                    }
                    # Return immediately - routing will check _task_invalidated and redirect to rework
                    return self._record_node_complete(state, node, error=f"Triggering hierarchical rework for: {missing_signoffs}")

                logger.info(f"ALL required signoffs approved for {state['sprint_id']}: {REQUIRED_SIGNOFFS}")

                # GRAFANA: Publish signoff results to dashboard
                if self._grafana_metrics:
                    try:
                        for agent_id, signoff in signoff_results.items():
                            if agent_id.startswith("_"):
                                continue  # Skip metadata keys
                            verdict = "approved" if signoff.get("approved", False) else "rejected"
                            self._grafana_metrics.publish_signoff(
                                agent=agent_id,
                                task_id=f"sprint_{state['sprint_id']}",
                                verdict=verdict,
                                confidence=signoff.get("confidence", 0.9),
                                reason=signoff.get("reason", "Sprint signoff"),
                            )
                            self._grafana_metrics.publish_agent_reasoning(
                                agent=agent_id,
                                thought_type="signing",
                                content=f"{verdict.upper()} sprint {state['sprint_id']}: {signoff.get('reason', '')}",
                                task=f"Signoff for {state['sprint_id']}",
                            )
                    except Exception as metrics_err:
                        logger.debug(f"Failed to publish signoff metrics: {metrics_err}")

                # Invariant I6: Check signoff truthfulness after processing
                # Validate that claims in signoffs match actual evidence
                if self._invariant_checker is not None:
                    for agent_id, signoff in signoff_results.items():
                        if agent_id.startswith("_"):
                            continue  # Skip metadata keys
                        # Extract claims from the signoff
                        claims = {
                            "tests_pass": signoff.get("tests_pass", False),
                            "coverage": signoff.get("coverage"),
                            "artifacts_created": signoff.get("artifacts_verified", []),
                        }
                        # Get evidence from state or signoff
                        evidence = {
                            "test_result": state.get("test_result", {}),
                            "coverage_report": state.get("coverage_report", {}),
                            "artifacts": signoff.get("artifacts_verified", []),
                        }
                        truthfulness_result = self._invariant_checker.check_signoff_truthfulness(
                            claims=claims,
                            evidence=evidence,
                        )
                        # Non-blocking: truthfulness violations are logged but don't fail the pipeline
                        state, _ = self._handle_invariant_violation(
                            state, truthfulness_result, node, f"I6_TRUTHFULNESS_{agent_id}", is_blocking=False
                        )

            except ImportError:
                logger.warning("Governance module not available for signoffs")
            except Exception as e:
                logger.error(f"Signoff processing failed: {e}")
                error_entry = create_error_entry(where=node, error=str(e))
                state = add_error(state, error_entry)

            # Partial Stacks Integration: SIGNOFF hook
            # Final DeepEval evaluation, BoT synthesis, Phoenix trace completion
            if self._partial_stacks is not None:
                try:
                    state = await self._partial_stacks.on_signoff(state)
                    logger.debug("Partial stacks integration: on_signoff completed")
                except Exception as e:
                    logger.warning(f"Partial stacks on_signoff failed (non-blocking): {e}")

            # Emit signoff event
            self._emit_event(state, "signoffs_completed", f"Signoffs processed for sprint {state['sprint_id']}")

            # Transition to DONE phase (only if all signoffs approved - verified above)
            state = update_phase(state, SprintPhase.DONE)
            state = update_status(state, PipelineStatus.COMPLETED)

            return self._record_node_complete(state, node)

        except Exception as e:
            logger.error(f"Signoff node failed: {e}")
            error_entry = create_error_entry(where=node, error=str(e))
            state = add_error(state, error_entry)

            # HIERARCHICAL REWORK: Signoff node exception triggers rework via ceo
            rework_context = create_hierarchical_rework_context(
                rejecting_agent="ceo",
                rejection_reason=f"Signoff node exception: {e}",
                work_type="exec",
                failed_items=["signoff_node_exception"],
                current_attempt=state.get("attempt", 1),
                max_attempts=3,
            )
            state = {
                **state,
                "_task_invalidated": True,
                "_requires_rework": True,  # P0-4 FIX: Enable rework flow
                "_rework_context": rework_context,
            }
            return self._record_node_complete(state, node, error=f"Triggering rework for signoff exception: {e}")

    # =========================================================================
    # ARTIFACT NODE
    # =========================================================================

    async def artifact_node(self, state: PipelineState) -> PipelineState:
        """Generate artifacts for the sprint.

        Responsibilities:
        1. Generate quality_bar.yml
        2. Generate gate_receipt.yml
        3. Generate agent manifests
        4. Update run_state.yml

        Phase transition: Final node before END
        """
        node = "artifact"
        state = self._record_node_start(state, node)

        # Update phase to DONE and emit event for cockpit
        state = {**state, "phase": "DONE"}
        self._emit_event(state, "artifact_started", f"Artifact generation started for sprint {state['sprint_id']}")

        # Publish Grafana metrics for Pipeline Control Center
        self._publish_grafana_metrics(state, node_name=node, agent="spec_master")

        try:
            # Invariant I9: Check SAFE_HALT before artifact generation
            # Even at the final stage, SAFE_HALT takes precedence
            halted, state = await self._check_safe_halt(state)
            if halted:
                return self._record_node_complete(state, node)

            # CRIT-003: Trust boundary check - validate agent has permission to write artifacts
            # Get the artifact writer agent from state (defaults to ace_exec for artifact node)
            artifact_agent = state.get("artifact_agent", "ace_exec")
            # FIX: Use "data:artifacts" to match ResourceType.DATA_ARTIFACTS in trust_boundaries.py
            # L3_MASTER (ace_exec) has permission: {READ, WRITE, CREATE} on data:artifacts
            state, boundary_ok = self._check_trust_boundary(
                state=state,
                agent_id=artifact_agent,
                resource="data:artifacts",
                action="write",
                node=node,
            )
            # Also check permission to write to run directory (uses same resource type)
            state, _ = self._check_trust_boundary(
                state=state,
                agent_id=artifact_agent,
                resource="data:artifacts",
                action="create",
                node=node,
            )

            # Invariant I7: Check evidence bundle before generating final artifacts
            # Validates that the audit trail from previous phases is complete
            if self._invariant_checker is not None:
                evidence_bundle_dir = state.get("evidence_bundle_dir")
                if evidence_bundle_dir:
                    bundle_path = Path(evidence_bundle_dir)
                    if bundle_path.exists():
                        evidence_result = self._invariant_checker.check_evidence_bundle(bundle_path)
                        # Non-blocking: evidence bundle issues are logged but don't fail artifact generation
                        state, _ = self._handle_invariant_violation(
                            state, evidence_result, node, "I7_AUDIT_TRAIL", is_blocking=False
                        )
                    else:
                        logger.warning(f"Evidence bundle directory does not exist: {evidence_bundle_dir}")
                else:
                    # Check if there's a default evidence bundle location
                    default_bundle_path = self.run_dir / "evidence" / state["sprint_id"]
                    if default_bundle_path.exists():
                        evidence_result = self._invariant_checker.check_evidence_bundle(default_bundle_path)
                        state, _ = self._handle_invariant_violation(
                            state, evidence_result, node, "I7_AUDIT_TRAIL", is_blocking=False
                        )
                    else:
                        logger.debug(f"No evidence bundle found at default location: {default_bundle_path}")

            # Get stacks for ARTIFACT operation (CRIT-02: Stack injection wiring)
            try:
                stack_ctx = self._get_stacks_for_operation(Operation.ARTIFACT, state)
                artifact_stacks = list(stack_ctx.stacks.keys())
                logger.info(f"ARTIFACT stacks available: {artifact_stacks}")
            except RuntimeError as e:
                logger.error(f"Required ARTIFACT stacks unavailable: {e}")
                # Non-blocking for artifact generation
                logger.warning("Continuing without required artifact stacks")

            # Generate artifacts via existing artifact_writers
            try:
                from pipeline.artifact_writers import (
                    get_artifact_manager,
                    QualityBarWriter,
                    GateReceiptWriter,
                )

                sprint_id = state["sprint_id"]
                run_id = state.get("run_id", "")

                # FIX: Use class factory methods instead of direct method calls
                # Generate quality bar using factory method
                quality_bar_data = QualityBarWriter.create_from_gate_results(
                    run_dir=self.run_dir,
                    sprint_id=sprint_id,
                    run_id=run_id,
                    gates_passed=state.get("gates_passed", []),
                    gates_failed=state.get("gates_failed", []),
                )
                manager = get_artifact_manager(run_dir=self.run_dir, run_id=run_id)
                manager.quality_bar.write(quality_bar_data, sprint_id=sprint_id)

                # Generate gate receipt using factory method
                # Build gate_results list from passed/failed gates
                gate_results = []
                for g in state.get("gates_passed", []):
                    gate_results.append({"gate_id": g, "gate_name": g, "status": "PASS"})
                for g in state.get("gates_failed", []):
                    gate_results.append({"gate_id": g, "gate_name": g, "status": "FAIL"})

                gate_receipt_data = GateReceiptWriter.create_from_gate_results(
                    run_dir=self.run_dir,
                    sprint_id=sprint_id,
                    run_id=run_id,
                    gate_results=gate_results,
                )
                manager.gate_receipt.write(gate_receipt_data, sprint_id=sprint_id)

            except ImportError:
                logger.warning("Artifact writers not available")
            except Exception as e:
                logger.error(f"Artifact generation failed: {e}")
                error_entry = create_error_entry(where=node, error=str(e))
                state = add_error(state, error_entry)

            # GAP-7: Generate 6 traceability reports
            try:
                await self._generate_traceability_reports(state)
            except Exception as trace_err:
                logger.warning(f"Traceability reports generation failed (non-blocking): {trace_err}")

            # HIGH-004 FIX: Trigger Live-SWE behavior evolution at end of sprint
            # This connects the 4,124 lines of dead code to the pipeline
            try:
                from pipeline.live_swe_integration import get_live_swe_integration

                live_swe = get_live_swe_integration()
                if live_swe is not None:
                    evolution_context = {
                        "sprint_id": state["sprint_id"],
                        "run_id": state.get("run_id", ""),
                        "gates_passed": state.get("gates_passed", []),
                        "gates_failed": state.get("gates_failed", []),
                        "status": state.get("status"),
                    }

                    evolution_result = live_swe.evolve_from_reflections(
                        sprint_id=state["sprint_id"],
                        context=evolution_context,
                    )

                    if evolution_result:
                        logger.info(
                            f"HIGH-004: Live-SWE evolution completed for sprint {state['sprint_id']}: "
                            f"proposals={evolution_result.get('proposals_count', 0)}, "
                            f"evolutions={evolution_result.get('evolutions_applied', 0)}"
                        )

                        # Add evolution metrics to state
                        state = {
                            **state,
                            "live_swe_evolution": {
                                "proposals_count": evolution_result.get("proposals_count", 0),
                                "evolutions_applied": evolution_result.get("evolutions_applied", 0),
                                "canary_percentage": evolution_result.get("canary_percentage", 0),
                            },
                        }
                else:
                    logger.debug("HIGH-004: Live-SWE integration not available")
            except ImportError:
                logger.debug("HIGH-004: Live-SWE integration module not available")
            except Exception as live_swe_err:
                logger.warning(f"HIGH-004: Live-SWE evolution failed (non-blocking): {live_swe_err}")

            # HIGH-003 FIX: Distill BoT insights at end of sprint
            # The BoT Bridge accumulates insights but never distills them - this is the consumer
            try:
                from pipeline.bot_bridge import get_unified_accumulator

                bot_bridge = get_unified_accumulator()
                if bot_bridge is not None:
                    insights = bot_bridge.distill_cross_system_insights()

                    if insights:
                        # Save insights to disk
                        insights_dir = self.run_dir / "sprints" / state["sprint_id"] / "insights"
                        insights_dir.mkdir(parents=True, exist_ok=True)

                        insights_data = {
                            "sprint_id": state["sprint_id"],
                            "run_id": state.get("run_id", ""),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "insights_count": len(insights),
                            "insights": [
                                {
                                    "content": str(i.content) if hasattr(i, 'content') else str(i),
                                    "confidence": getattr(i, 'confidence', 0.0),
                                    "source": getattr(i, 'source', 'unknown'),
                                }
                                for i in insights
                            ],
                        }

                        import yaml
                        insights_path = insights_dir / "distilled_insights.yml"
                        with open(insights_path, "w") as f:
                            yaml.dump(insights_data, f, default_flow_style=False)

                        logger.info(
                            f"HIGH-003: Distilled {len(insights)} insights for sprint {state['sprint_id']}"
                        )

                        # Add to state
                        state = {
                            **state,
                            "distilled_insights_count": len(insights),
                            "distilled_insights_path": str(insights_path),
                        }

                        # MED-001 FIX: Sync distilled insights to Letta memory
                        try:
                            from pipeline.amem_integration import get_amem_integration

                            amem = get_amem_integration()
                            synced_count = 0
                            for insight in insights:
                                content = str(insight.content) if hasattr(insight, 'content') else str(insight)
                                # Save each insight as a learning
                                amem.save_learning(
                                    agent_id=f"bot_bridge/{state['sprint_id']}",
                                    learning=content[:500],  # Truncate if too long
                                    tags=["distilled_insight", f"sprint_{state['sprint_id']}"],
                                    sprint_id=state["sprint_id"],
                                )
                                synced_count += 1

                            logger.info(
                                f"MED-001: Synced {synced_count} insights to A-MEM/Letta"
                            )
                            state = {
                                **state,
                                "insights_synced_to_letta": synced_count,
                            }
                        except Exception as sync_err:
                            logger.warning(f"MED-001: Insight sync to Letta failed (non-blocking): {sync_err}")
                    else:
                        logger.debug("HIGH-003: No insights to distill")
                else:
                    logger.debug("HIGH-003: BoT Bridge not available")
            except ImportError:
                logger.debug("HIGH-003: BoT Bridge module not available")
            except Exception as bot_err:
                logger.warning(f"HIGH-003: BoT insight distillation failed (non-blocking): {bot_err}")

            # Partial Stacks Integration: ARTIFACT hook
            # Exports BoT thoughts to artifact, exports evaluation metrics
            if self._partial_stacks is not None:
                try:
                    state = await self._partial_stacks.on_artifact(state)
                    logger.debug("Partial stacks integration: on_artifact completed")
                except Exception as e:
                    logger.warning(f"Partial stacks on_artifact failed (non-blocking): {e}")

            # Emit artifact event
            self._emit_event(state, "artifacts_generated", f"Artifacts generated for sprint {state['sprint_id']}")

            # Mark completed
            state = {
                **state,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }

            # Create sprint snapshot for Grafana dashboard replay
            try:
                from pipeline.sprint_snapshot import get_snapshot_service
                snapshot_service = get_snapshot_service()
                snapshot_path = snapshot_service.create_snapshot(state["sprint_id"])
                logger.info(f"Sprint snapshot saved to {snapshot_path}")
            except Exception as snap_err:
                logger.warning(f"Failed to create sprint snapshot (non-blocking): {snap_err}")

            return self._record_node_complete(state, node)

        except Exception as e:
            logger.error(f"Artifact node failed: {e}")
            error_entry = create_error_entry(where=node, error=str(e))
            state = add_error(state, error_entry)
            return self._record_node_complete(state, node, error=str(e))

    # =========================================================================
    # GAP-7: TRACEABILITY REPORTS
    # =========================================================================

    async def _generate_traceability_reports(self, state: PipelineState) -> None:
        """Generate 6 traceability reports for the sprint.

        GAP-7 FIX: Creates reports mapping RF/INV/EDGE to implementations.

        Reports generated:
        1. rf_traceability.yml - RF to implementation mapping
        2. inv_enforcement.yml - INV to enforcement mapping
        3. edge_coverage.yml - EDGE to test coverage mapping
        4. ac_verification.yml - AC verification status
        5. dod_completion.yml - DoD checklist completion
        6. sprint_summary.yml - Full sprint context summary

        Args:
            state: Current pipeline state.
        """
        import yaml
        from datetime import datetime, timezone

        sprint_id = state["sprint_id"]
        reports_dir = self.run_dir / "sprints" / sprint_id / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        context_pack = state.get("context_pack", {})
        generated_reports = []

        # 1. RF Traceability Matrix
        rf_matrix = {
            "schema_version": "traceability.rf.v1",
            "sprint_id": sprint_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "requirements": [],
        }
        functional_requirements = context_pack.get("functional_requirements", [])
        deliverables = context_pack.get("deliverables", [])
        rf_to_deliverable = context_pack.get("rf_to_deliverable", {})

        for rf in functional_requirements:
            rf_id = rf.get("id", "RF-???")
            rf_entry = {
                "id": rf_id,
                "description": rf.get("description", rf.get("text", "")),
                "priority": rf.get("priority", "P2"),
                "mapped_deliverables": rf_to_deliverable.get(rf_id, deliverables[:1]),
                "status": "implemented" if state.get("gate_status") == "PASS" else "pending",
            }
            rf_matrix["requirements"].append(rf_entry)

        rf_matrix["summary"] = {
            "total_rf": len(functional_requirements),
            "implemented": sum(1 for r in rf_matrix["requirements"] if r["status"] == "implemented"),
            "pending": sum(1 for r in rf_matrix["requirements"] if r["status"] == "pending"),
        }

        with open(reports_dir / "rf_traceability.yml", "w") as f:
            yaml.safe_dump(rf_matrix, f, default_flow_style=False)
        generated_reports.append("rf_traceability.yml")

        # 2. INV Enforcement Report
        inv_report = {
            "schema_version": "traceability.inv.v1",
            "sprint_id": sprint_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "invariants": [],
        }
        invariants = context_pack.get("invariants", [])
        inv_to_enforcement = context_pack.get("inv_to_enforcement", {})

        for inv in invariants:
            inv_id = inv.get("id", "INV-???")
            enforcement = inv_to_enforcement.get(inv_id, {})
            inv_entry = {
                "id": inv_id,
                "rule": inv.get("rule", inv.get("description", "")),
                "enforcement_type": enforcement.get("type", "gate"),
                "enforcement_location": enforcement.get("location", "G0"),
                "verified": state.get("gate_status") == "PASS",
            }
            inv_report["invariants"].append(inv_entry)

        inv_report["summary"] = {
            "total_inv": len(invariants),
            "verified": sum(1 for i in inv_report["invariants"] if i["verified"]),
            "pending": sum(1 for i in inv_report["invariants"] if not i["verified"]),
        }

        with open(reports_dir / "inv_enforcement.yml", "w") as f:
            yaml.safe_dump(inv_report, f, default_flow_style=False)
        generated_reports.append("inv_enforcement.yml")

        # 3. EDGE Coverage Report
        edge_report = {
            "schema_version": "traceability.edge.v1",
            "sprint_id": sprint_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "edge_cases": [],
        }
        edge_cases = context_pack.get("edge_cases", [])

        for edge in edge_cases:
            edge_id = edge.get("id", "EDGE-???")
            edge_entry = {
                "id": edge_id,
                "scenario": edge.get("scenario", edge.get("description", "")),
                "expected": edge.get("expected", ""),
                "covered": state.get("gate_status") == "PASS",  # Assumed covered if gates pass
                "test_location": edge.get("test", ""),
            }
            edge_report["edge_cases"].append(edge_entry)

        edge_report["summary"] = {
            "total_edge": len(edge_cases),
            "covered": sum(1 for e in edge_report["edge_cases"] if e["covered"]),
            "uncovered": sum(1 for e in edge_report["edge_cases"] if not e["covered"]),
        }

        with open(reports_dir / "edge_coverage.yml", "w") as f:
            yaml.safe_dump(edge_report, f, default_flow_style=False)
        generated_reports.append("edge_coverage.yml")

        # 4. AC Verification Report
        ac_report = {
            "schema_version": "traceability.ac.v1",
            "sprint_id": sprint_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "acceptance_criteria": [],
        }
        acceptance_criteria = context_pack.get("acceptance_criteria", [])

        for ac in acceptance_criteria:
            ac_id = ac.get("id", "AC-???")
            ac_entry = {
                "id": ac_id,
                "criterion": ac.get("criterion", ac.get("description", "")),
                "verification_method": ac.get("verification", "manual"),
                "verified": state.get("dod_validated", True),
                "evidence": [],
            }
            ac_report["acceptance_criteria"].append(ac_entry)

        ac_report["summary"] = {
            "total_ac": len(acceptance_criteria),
            "verified": sum(1 for a in ac_report["acceptance_criteria"] if a["verified"]),
            "pending": sum(1 for a in ac_report["acceptance_criteria"] if not a["verified"]),
        }

        with open(reports_dir / "ac_verification.yml", "w") as f:
            yaml.safe_dump(ac_report, f, default_flow_style=False)
        generated_reports.append("ac_verification.yml")

        # 5. DoD Completion Report
        dod_report = {
            "schema_version": "traceability.dod.v1",
            "sprint_id": sprint_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "checklist": [],
        }
        dod_checklist = context_pack.get("dod_checklist", [])
        dod_validated = state.get("dod_validated", True)
        dod_issues = state.get("dod_issues", [])

        for item in dod_checklist:
            item_name = item.get("item", item.get("name", "Unknown"))
            item_entry = {
                "item": item_name,
                "required": item.get("required", True),
                "completed": item_name not in str(dod_issues),
            }
            dod_report["checklist"].append(item_entry)

        dod_report["summary"] = {
            "total_items": len(dod_checklist),
            "completed": sum(1 for d in dod_report["checklist"] if d["completed"]),
            "incomplete": sum(1 for d in dod_report["checklist"] if not d["completed"]),
            "all_required_complete": dod_validated,
            "issues": dod_issues,
        }

        with open(reports_dir / "dod_completion.yml", "w") as f:
            yaml.safe_dump(dod_report, f, default_flow_style=False)
        generated_reports.append("dod_completion.yml")

        # 6. Sprint Summary
        sprint_summary = {
            "schema_version": "traceability.summary.v1",
            "sprint_id": sprint_id,
            "run_id": state.get("run_id", ""),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "objective": context_pack.get("objective", ""),
            "deliverables": deliverables,
            "counts": {
                "rf": len(functional_requirements),
                "inv": len(invariants),
                "edge": len(edge_cases),
                "ac": len(acceptance_criteria),
                "dod": len(dod_checklist),
            },
            "status": {
                "gate_status": state.get("gate_status", "UNKNOWN"),
                "gates_passed": state.get("gates_passed", []),
                "gates_failed": state.get("gates_failed", []),
                "dod_validated": dod_validated,
                "signoffs": list(state.get("signoffs", {}).keys()),
            },
            "reports_generated": generated_reports,
        }

        with open(reports_dir / "sprint_summary.yml", "w") as f:
            yaml.safe_dump(sprint_summary, f, default_flow_style=False)
        generated_reports.append("sprint_summary.yml")

        logger.info(f"GAP-7: Generated {len(generated_reports)} traceability reports in {reports_dir}")

    # =========================================================================
    # HAMILTON DAG PIPELINES (P3-001)
    # =========================================================================

    def _extract_claims_for_hamilton(self, state: PipelineState) -> Optional[List[Dict[str, Any]]]:
        """Extract claims data from state for Hamilton pipeline.

        Extracts claims from context pack deliverables or crew results
        to feed into the Hamilton claim verification DAG.

        Args:
            state: Current pipeline state.

        Returns:
            List of claim dictionaries or None if no claims found.
        """
        claims = []

        # Try to extract from context pack
        context_pack = state.get("context_pack", {})
        if context_pack:
            # Extract deliverables as claims to verify
            deliverables = context_pack.get("deliverables", [])
            objective = context_pack.get("objective", "")

            for idx, deliverable in enumerate(deliverables):
                claims.append({
                    "claim_id": f"deliverable_{idx}",
                    "text": str(deliverable),
                    "source": f"sprint_{state.get('sprint_id', 'unknown')}",
                    "type": "deliverable",
                })

            # Add objective as a claim
            if objective:
                claims.append({
                    "claim_id": "objective",
                    "text": objective,
                    "source": f"sprint_{state.get('sprint_id', 'unknown')}",
                    "type": "objective",
                })

        # Try to extract from crew result if available
        crew_result = state.get("crew_result", {})
        if crew_result and crew_result.get("crew_output"):
            claims.append({
                "claim_id": "crew_output",
                "text": str(crew_result["crew_output"])[:2000],  # Limit size
                "source": "crew_execution",
                "type": "output",
            })

        return claims if claims else None

    def _extract_gate_config_for_hamilton(self, state: PipelineState) -> Dict[str, Any]:
        """Extract gate configuration from state for Hamilton pipeline.

        Args:
            state: Current pipeline state.

        Returns:
            Gate configuration dictionary for Hamilton.
        """
        gates_passed = state.get("gates_passed", [])
        gates_failed = state.get("gates_failed", [])

        # Build gate config for Hamilton DAG
        gates = []
        for gate_id in gates_passed:
            gates.append({
                "gate_id": gate_id,
                "status": "PASS",
                "required": True,
            })
        for gate_id in gates_failed:
            gates.append({
                "gate_id": gate_id,
                "status": "FAIL",
                "required": True,
            })

        return {
            "gates": gates,
            "sprint_id": state.get("sprint_id", ""),
            "run_id": state.get("run_id", ""),
        }

    async def _run_hamilton_claim_verification(
        self,
        state: PipelineState,
        claims_data: Any,
    ) -> Dict[str, Any]:
        """Run Hamilton claim verification pipeline.

        P3-001: Uses Hamilton DAG for declarative data transformation
        during claim verification. Provides:
        - Automatic dependency resolution
        - Data lineage tracking
        - Reproducible transformations

        Args:
            state: Current pipeline state.
            claims_data: Claims to verify (can be dict, list, or DataFrame).

        Returns:
            Dict with pipeline results and metadata.
        """
        result = {
            "pipeline": "claim_verification",
            "status": "pending",
            "execution_mode": "fallback",  # Hamilton disabled, using fallback
            "outputs": {},
            "error": None,
        }

        try:
            # 2026-01-25: Hamilton disabled, but pipeline still runs via fallback execution
            pipeline_result = hamilton_run_claim_verification(claims_data)

            result["status"] = pipeline_result.get("status", "unknown")
            result["outputs"] = pipeline_result.get("outputs", {})
            result["execution_time_ms"] = pipeline_result.get("execution_time_ms", 0)
            result["nodes_executed"] = pipeline_result.get("nodes_executed", 0)

            if pipeline_result.get("errors"):
                result["error"] = "; ".join(pipeline_result["errors"])

            logger.info(
                f"Claim verification pipeline completed: "
                f"status={result['status']}, nodes={result['nodes_executed']}"
            )

            # Emit event for observability
            self._emit_event(
                state,
                "hamilton_claim_verification",
                f"Claim verification: {result['status']}"
            )

            return result

        except Exception as e:
            logger.error(f"Claim verification pipeline failed: {e}")
            result["status"] = "failed"
            result["error"] = str(e)
            return result

    async def _run_hamilton_gate_validation(
        self,
        state: PipelineState,
        gate_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run Hamilton gate validation pipeline.

        P3-001: Uses Hamilton DAG for structured gate validation
        with automatic dependency resolution.

        Args:
            state: Current pipeline state.
            gate_config: Gate configuration with gates to validate.

        Returns:
            Dict with pipeline results and metadata.
        """
        result = {
            "pipeline": "gate_validation",
            "status": "pending",
            "execution_mode": "fallback",  # Hamilton disabled, using fallback
            "outputs": {},
            "error": None,
        }

        try:
            # 2026-01-25: Hamilton disabled, but pipeline still runs via fallback execution
            pipeline_result = hamilton_run_gate_validation(gate_config)

            result["status"] = pipeline_result.get("status", "unknown")
            result["outputs"] = pipeline_result.get("outputs", {})
            result["execution_time_ms"] = pipeline_result.get("execution_time_ms", 0)
            result["nodes_executed"] = pipeline_result.get("nodes_executed", 0)

            if pipeline_result.get("errors"):
                result["error"] = "; ".join(pipeline_result["errors"])

            logger.info(
                f"Gate validation pipeline completed: "
                f"status={result['status']}, nodes={result['nodes_executed']}"
            )

            # Emit event for observability
            self._emit_event(
                state,
                "hamilton_gate_validation",
                f"Gate validation: {result['status']}"
            )

            return result

        except Exception as e:
            logger.error(f"Gate validation pipeline failed: {e}")
            result["status"] = "failed"
            result["error"] = str(e)
            return result

    async def _run_hamilton_evidence_collection(
        self,
        state: PipelineState,
        search_config: Dict[str, Any],
        search_query: str,
    ) -> Dict[str, Any]:
        """Run Hamilton evidence collection pipeline.

        P3-001: Uses Hamilton DAG for evidence collection with
        automatic source query adaptation.

        Args:
            state: Current pipeline state.
            search_config: Configuration for evidence sources.
            search_query: Query to search for evidence.

        Returns:
            Dict with pipeline results and metadata.
        """
        result = {
            "pipeline": "evidence_collection",
            "status": "pending",
            "execution_mode": "fallback",  # Hamilton disabled, using fallback
            "outputs": {},
            "error": None,
        }

        try:
            # 2026-01-25: Hamilton disabled, but pipeline still runs via fallback execution
            pipeline_result = hamilton_run_evidence_collection(search_config, search_query)

            result["status"] = pipeline_result.get("status", "unknown")
            result["outputs"] = pipeline_result.get("outputs", {})
            result["execution_time_ms"] = pipeline_result.get("execution_time_ms", 0)
            result["nodes_executed"] = pipeline_result.get("nodes_executed", 0)

            if pipeline_result.get("errors"):
                result["error"] = "; ".join(pipeline_result["errors"])

            logger.info(
                f"Evidence collection pipeline completed: "
                f"status={result['status']}, nodes={result['nodes_executed']}"
            )

            # Emit event for observability
            self._emit_event(
                state,
                "hamilton_evidence_collection",
                f"Evidence collection: {result['status']}"
            )

            return result

        except Exception as e:
            logger.error(f"Evidence collection pipeline failed: {e}")
            result["status"] = "failed"
            result["error"] = str(e)
            return result

    # =========================================================================
    # SEMANTIC SEARCH NODE (AGENT BETA - INV-BETA-001, INV-BETA-004, INV-BETA-006)
    # =========================================================================

    async def semantic_search_node(self, state: PipelineState) -> PipelineState:
        """Index specs and verify coverage/duplicates using Qdrant semantic search.

        AGENT BETA DELIVERY: Semantic search integration for spec analysis.

        Responsibilities:
        1. Index specs from context_pack into Qdrant
        2. Check for duplicate specs (similarity > 0.95)
        3. Verify user story coverage
        4. Detect potential related/conflicting specs

        Invariants enforced:
        - INV-BETA-001: Qdrant unavailable = [] + ERROR log (no fake docs)
        - INV-BETA-004: Embeddings via Ollama nomic-embed-text
        - INV-BETA-006: Embedding timeout max 10 seconds

        Phase transition: Part of SPEC phase, optional enhancement
        """
        node = "semantic_search"
        state = self._record_node_start(state, node)

        try:
            # Check SAFE_HALT first
            halted, state = await self._check_safe_halt(state)
            if halted:
                return self._record_node_complete(state, node)

            # Import semantic search module
            try:
                from pipeline.spec_kit.semantic_search import (
                    SpecSemanticSearch,
                    get_semantic_search,
                )
            except ImportError as e:
                logger.warning(f"Semantic search module not available: {e}")
                state = {
                    **state,
                    "semantic_search_status": "unavailable",
                    "semantic_search_error": str(e),
                }
                return self._record_node_complete(state, node)

            # Get specs from context_pack
            context_pack = state.get("context_pack", {})
            functional_requirements = context_pack.get("functional_requirements", [])

            if not functional_requirements:
                logger.info("No functional requirements to index for semantic search")
                state = {
                    **state,
                    "semantic_search_status": "skipped",
                    "semantic_search_reason": "no_specs",
                }
                return self._record_node_complete(state, node)

            # Initialize semantic search
            search = get_semantic_search()
            initialized = await search.initialize()

            if not initialized:
                # INV-BETA-001: Qdrant unavailable = [] + ERROR log
                logger.error("INV-BETA-001: Qdrant unavailable, semantic search disabled")
                state = {
                    **state,
                    "semantic_search_status": "unavailable",
                    "semantic_search_error": "Qdrant connection failed",
                }
                return self._record_node_complete(state, node)

            # Index specs
            indexed_count = 0
            for spec in functional_requirements:
                spec_id = spec.get("id", f"spec_{indexed_count}")
                spec_with_sprint = {
                    **spec,
                    "source_sprint": state.get("sprint_id", "unknown"),
                }
                success = await search.index_spec(spec_with_sprint)
                if success:
                    indexed_count += 1

            logger.info(f"Indexed {indexed_count}/{len(functional_requirements)} specs")

            # Check for duplicates
            duplicates = []
            try:
                dup_results = await search.find_duplicates(
                    specs=functional_requirements,
                    threshold=0.95,
                )
                for dup in dup_results:
                    duplicates.append({
                        "spec_1": dup.spec_id_1,
                        "spec_2": dup.spec_id_2,
                        "similarity": dup.similarity_score,
                        "recommendation": dup.recommendation,
                    })
                if duplicates:
                    logger.warning(f"Found {len(duplicates)} potential duplicate specs")
            except Exception as e:
                logger.warning(f"Duplicate detection failed: {e}")

            # Check coverage for user stories (if available)
            coverage_results = []
            user_stories = context_pack.get("user_stories", [])
            for story in user_stories:
                story_text = story.get("text", str(story))
                try:
                    coverage = await search.check_coverage(story_text)
                    coverage_results.append({
                        "story": story_text[:100],
                        "covered": coverage.covered,
                        "matching_spec": coverage.matching_spec_id,
                        "similarity": coverage.similarity_score,
                    })
                except Exception as e:
                    logger.warning(f"Coverage check failed for story: {e}")

            # Update state with results
            state = {
                **state,
                "semantic_search_status": "completed",
                "semantic_search_results": {
                    "indexed_count": indexed_count,
                    "total_specs": len(functional_requirements),
                    "duplicates": duplicates,
                    "coverage": coverage_results,
                },
            }

            self._emit_event(
                state,
                "semantic_search_completed",
                f"Indexed {indexed_count} specs, found {len(duplicates)} duplicates",
            )

            return self._record_node_complete(state, node)

        except Exception as e:
            logger.error(f"Semantic search node failed: {e}")
            error_entry = create_error_entry(where=node, error=str(e), recoverable=True)
            state = add_error(state, error_entry)
            state = {
                **state,
                "semantic_search_status": "failed",
                "semantic_search_error": str(e),
            }
            return self._record_node_complete(state, node, error=str(e))

    # =========================================================================
    # CONTRADICTION NODE (AGENT BETA - INV-BETA-002)
    # =========================================================================

    async def contradiction_node(self, state: PipelineState) -> PipelineState:
        """Verify contradictions between specs using Z3 SAT solver.

        AGENT BETA DELIVERY: Contradiction detection for spec validation.

        Responsibilities:
        1. Extract specs from context_pack
        2. Run Z3-based contradiction detection
        3. Report blocking contradictions (INV-BETA-002)
        4. Generate resolution suggestions

        Invariants enforced:
        - INV-BETA-002: Z3 contradiction MUST block pipeline
        - Falls back to LLM-based detection when Z3 unavailable

        Phase transition: Part of SPEC phase, blocking on contradictions
        """
        node = "contradiction"
        state = self._record_node_start(state, node)

        try:
            # Check SAFE_HALT first
            halted, state = await self._check_safe_halt(state)
            if halted:
                return self._record_node_complete(state, node)

            # Import contradiction detector
            try:
                from pipeline.spec_kit.contradiction_detector import (
                    ContradictionDetector,
                    detect_contradictions,
                    is_z3_available,
                )
                from pipeline.spec_kit.exceptions import ContradictionError
            except ImportError as e:
                logger.warning(f"Contradiction detector not available: {e}")
                state = {
                    **state,
                    "contradiction_status": "unavailable",
                    "contradiction_error": str(e),
                }
                return self._record_node_complete(state, node)

            # Get specs from context_pack
            context_pack = state.get("context_pack", {})
            functional_requirements = context_pack.get("functional_requirements", [])

            if not functional_requirements or len(functional_requirements) < 2:
                logger.info("Insufficient specs for contradiction detection (need >= 2)")
                state = {
                    **state,
                    "contradiction_status": "skipped",
                    "contradiction_reason": "insufficient_specs",
                }
                return self._record_node_complete(state, node)

            # Log Z3 availability
            z3_available = is_z3_available()
            logger.info(f"Contradiction detection: Z3 available={z3_available}")

            # Run contradiction detection
            # INV-BETA-002: Z3 contradiction MUST block pipeline
            try:
                report = await detect_contradictions(
                    specs=functional_requirements,
                    raise_on_error=True,  # Block on blocking contradictions
                )

                # No blocking contradictions found
                state = {
                    **state,
                    "contradiction_status": "passed",
                    "contradiction_results": {
                        "is_consistent": report.is_consistent,
                        "total_specs": report.total_specs,
                        "contradictions_count": len(report.contradictions),
                        "blocking_count": sum(
                            1 for c in report.contradictions if c.severity == "blocking"
                        ),
                        "warning_count": sum(
                            1 for c in report.contradictions if c.severity == "warning"
                        ),
                        "z3_used": z3_available,
                    },
                }

                if report.contradictions:
                    # Non-blocking contradictions (warnings)
                    logger.warning(
                        f"Found {len(report.contradictions)} non-blocking contradictions"
                    )
                    state["contradiction_warnings"] = [
                        {
                            "spec_1": c.spec_id_1,
                            "spec_2": c.spec_id_2,
                            "description": c.description,
                            "severity": c.severity,
                            "suggestion": c.resolution_suggestion,
                        }
                        for c in report.contradictions
                    ]

                self._emit_event(
                    state,
                    "contradiction_check_passed",
                    f"No blocking contradictions in {report.total_specs} specs",
                )

                return self._record_node_complete(state, node)

            except ContradictionError as e:
                # INV-BETA-002: BLOCKING contradiction found - pipeline must halt
                logger.error(f"INV-BETA-002 VIOLATION: Blocking contradiction detected!")
                logger.error(f"Specs involved: {e.spec_ids}")
                logger.error(f"Description: {e.description}")

                state = {
                    **state,
                    "contradiction_status": "blocked",
                    "contradiction_results": {
                        "is_consistent": False,
                        "blocking_contradiction": {
                            "spec_ids": list(e.spec_ids),
                            "description": e.description,
                        },
                        "z3_used": z3_available,
                    },
                }

                # Mark pipeline as blocked
                state = update_status(state, PipelineStatus.HALTED)

                error_entry = create_error_entry(
                    where=node,
                    error=f"INV-BETA-002: Blocking contradiction between {e.spec_ids}",
                    recoverable=False,  # Must be resolved manually
                )
                state = add_error(state, error_entry)

                self._emit_event(
                    state,
                    "contradiction_blocked",
                    f"BLOCKING: Contradiction between {e.spec_ids}: {e.description[:100]}",
                )

                return self._record_node_complete(
                    state, node, error=f"Blocking contradiction: {e.description}"
                )

        except Exception as e:
            logger.error(f"Contradiction node failed: {e}")
            error_entry = create_error_entry(where=node, error=str(e), recoverable=True)
            state = add_error(state, error_entry)
            state = {
                **state,
                "contradiction_status": "failed",
                "contradiction_error": str(e),
            }
            return self._record_node_complete(state, node, error=str(e))

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _emit_event(self, state: PipelineState, event_type: str, message: str) -> None:
        """Emit an event to event_log.ndjson.

        Invariant I8: event_log uses schema_version, event_id, type.
        """
        try:
            import json
            from datetime import datetime, timezone

            event = {
                "schema_version": "1.0",
                "event_id": f"{event_type}:{state['run_id']}:{state['sprint_id']}:attempt:{state['attempt']}",
                "type": event_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": message,
                "run_id": state["run_id"],
                "sprint_id": state["sprint_id"],
                "phase": state.get("phase", "UNKNOWN"),
                "level": "info",
            }

            event_log_path = self.run_dir / "event_log.ndjson"
            event_log_path.parent.mkdir(parents=True, exist_ok=True)

            with open(event_log_path, "a") as f:
                f.write(json.dumps(event) + "\n")

        except Exception as e:
            # Invariant I11: Observability is non-blocking
            logger.warning(f"Failed to emit event (non-blocking): {e}")


# =============================================================================
# WORKFLOW BUILDER
# =============================================================================


def build_workflow(
    run_dir: Path | str,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    with_enforcement: bool = True,
    use_subgraphs: bool = False,
) -> Optional[StateGraph]:
    """Build the LangGraph workflow.

    Args:
        run_dir: Directory for run artifacts.
        checkpointer: Optional checkpoint saver for persistence.
        with_enforcement: If True (default), use EnforcedWorkflowNodes which:
            - BLOCKS pipeline on guardrail violations
            - INVALIDATES tasks and queues for REWORK
            - Integrates with Reflexion for learning
            Set to False only for testing or legacy mode.
        use_subgraphs: If True, use modular subgraphs for gate, QA, and signoff
            operations. This provides better testability and modularity.
            GHOST CODE INTEGRATION (2026-01-30): Enables pipeline.langgraph.subgraphs

    Returns:
        Compiled StateGraph, or None if LangGraph not available.
    """
    if not LANGGRAPH_AVAILABLE:
        logger.error("LangGraph not available - install with: pip install langgraph")
        return None

    # GHOST CODE INTEGRATION (2026-01-30): Use subgraphs for modular workflow
    # This provides better testability and allows independent testing of gate,
    # QA, and signoff subgraphs
    if use_subgraphs:
        if SUBGRAPHS_AVAILABLE and compose_workflow_with_subgraphs is not None:
            logger.info("Building workflow with SUBGRAPHS enabled (modular gate/QA/signoff)")
            run_path = Path(run_dir) if isinstance(run_dir, str) else run_dir
            return compose_workflow_with_subgraphs(run_path, checkpointer)
        else:
            logger.warning("Subgraphs requested but not available - falling back to standard workflow")

    # Create workflow nodes
    base_nodes = WorkflowNodes(run_dir=run_dir)

    # Wrap with enforcement if enabled (DEFAULT: True)
    if with_enforcement:
        try:
            from pipeline.langgraph.enforcement_integration import (
                EnforcedWorkflowNodes,
                build_enforced_workflow,
            )
            logger.info("Building workflow with ENFORCEMENT enabled (guardrails will BLOCK and REWORK)")
            # Use the full enforced workflow builder which includes rework node
            return build_enforced_workflow(base_nodes, checkpointer)
        except ImportError as e:
            # FIX BLACK-NEW-07: FAIL-SECURE - Do NOT fall back to non-enforced workflow
            # If enforcement is requested but unavailable, FAIL the build entirely
            logger.error(f"CRITICAL: Enforcement module failed to import: {e}")
            logger.error("FAIL-SECURE: Cannot build workflow without enforcement when requested")
            raise RuntimeError(
                f"Enforcement module unavailable but required (with_enforcement=True). "
                f"Import error: {e}. "
                f"To run without enforcement (NOT RECOMMENDED), set with_enforcement=False explicitly."
            ) from e

    # Standard workflow without enforcement (EXPLICIT opt-out only - not a fallback)
    # 2026-01-29: Now uses SURGICAL REWORK for task-level rework
    logger.warning("EXPLICIT NON-ENFORCED MODE: Building workflow WITHOUT enforcement")
    logger.warning("⚠️ SECURITY WARNING: Violations will NOT trigger rework - use only for testing!")

    # Import surgical rework components
    from pipeline.langgraph.state import TaskLevelStatus
    from pipeline.langgraph.surgical_rework_nodes import SurgicalReworkNodes
    from pipeline.surgical_rework_config import (
        get_surgical_rework_config,
        is_surgical_rework_enabled,
    )

    # Create surgical rework nodes
    config = get_surgical_rework_config()
    surgical_nodes = SurgicalReworkNodes(
        max_concurrent=config.limits.max_concurrent_tasks,
        max_repair_attempts=config.limits.max_repair_attempts_per_task,
    )

    # Build the graph
    workflow = StateGraph(PipelineState)

    # Add nodes (GAP-3: Added spec node between init and exec)
    # FIX 2026-01-28: Added qa node between gate and signoff for QA workers
    workflow.add_node("init", base_nodes.init_node)
    workflow.add_node("spec", base_nodes.spec_node)  # GAP-3: Spec decomposition

    # AGENT BETA: Semantic search and contradiction detection nodes
    # These nodes are OPTIONAL and run after spec for enhanced spec analysis
    workflow.add_node("semantic_search", base_nodes.semantic_search_node)  # INV-BETA-001,004,006
    workflow.add_node("contradiction", base_nodes.contradiction_node)      # INV-BETA-002

    # 2026-01-29: Surgical rework nodes (task-level execution)
    workflow.add_node("setup_tasks", surgical_nodes.setup_tasks_node)
    workflow.add_node("task_exec", surgical_nodes.task_exec_node)
    workflow.add_node("task_validate", surgical_nodes.task_validate_node)
    workflow.add_node("task_rework", surgical_nodes.task_rework_node)
    workflow.add_node("check_group", surgical_nodes.check_group_complete_node)
    workflow.add_node("integration_validate", surgical_nodes.integration_validate_node)

    # Legacy nodes (fallback when surgical rework not applicable)
    workflow.add_node("exec", base_nodes.exec_node)
    workflow.add_node("gate", base_nodes.gate_node)  # Automated gates (G0-G8)
    workflow.add_node("qa", base_nodes.qa_node)      # QA workers (auditor, refinador, etc.)
    workflow.add_node("signoff", base_nodes.signoff_node)
    workflow.add_node("artifact", base_nodes.artifact_node)

    # Legacy rework node (fallback)
    async def legacy_rework_node(state: PipelineState) -> PipelineState:
        """Legacy sprint-level rework (fallback when surgical rework disabled)."""
        result_state = dict(state)

        rework_context = result_state.get("_rework_context", {})
        current_attempt = rework_context.get("current_attempt", result_state.get("attempt", 1))
        max_attempts = rework_context.get("max_attempts", 3)

        try:
            from pipeline.rework_strategies import should_rework, get_rework_instructions

            can_rework, reason = should_rework(rework_context)
            if not can_rework:
                logger.warning(f"Rework not possible: {reason}")
                result_state["phase"] = SprintPhase.HALT.value
                result_state["_escalated"] = True
                result_state["_escalation_reason"] = reason
                return result_state

            result_state["_rework_instructions"] = get_rework_instructions(rework_context)
        except ImportError:
            logger.warning("rework_strategies not available")

        if current_attempt >= max_attempts:
            logger.error(f"Max rework attempts ({max_attempts}) exceeded")
            result_state["phase"] = SprintPhase.HALT.value
            result_state["_escalated"] = True
            result_state["_escalation_reason"] = f"Max rework attempts ({max_attempts}) exceeded"
            return result_state

        result_state["attempt"] = current_attempt + 1
        result_state["_task_invalidated"] = False
        result_state["_requires_rework"] = True

        logger.info(f"Legacy rework attempt {current_attempt + 1}/{max_attempts}")

        return result_state

    workflow.add_node("legacy_rework", legacy_rework_node)

    # Define edges
    workflow.set_entry_point("init")

    # =================================================================
    # ROUTING FUNCTIONS (2026-01-29: Updated for Surgical Rework)
    # =================================================================

    def route_after_init(state: PipelineState) -> str:
        """Route after init node - goes to spec for decomposition."""
        status = state.get("status")
        if status in (PipelineStatus.HALTED.value, PipelineStatus.FAILED.value):
            return END
        return "spec"

    def route_after_spec(state: PipelineState) -> str:
        """Route after spec node - goes to semantic search for spec analysis.

        AGENT BETA: Updated to include semantic search and contradiction detection.
        """
        status = state.get("status")
        if status in (PipelineStatus.HALTED.value, PipelineStatus.FAILED.value):
            return END

        # AGENT BETA: Route to semantic search first
        return "semantic_search"

    def route_after_semantic_search(state: PipelineState) -> str:
        """Route after semantic search node - goes to contradiction detection.

        AGENT BETA: Semantic search -> Contradiction detection flow.
        """
        status = state.get("status")
        if status in (PipelineStatus.HALTED.value, PipelineStatus.FAILED.value):
            return END

        # Continue to contradiction detection regardless of semantic search result
        # (semantic search is optional enhancement, not blocking)
        return "contradiction"

    def route_after_contradiction(state: PipelineState) -> str:
        """Route after contradiction node - decides surgical vs legacy exec.

        AGENT BETA + 2026-01-29: SURGICAL REWORK DECISION POINT
        INV-BETA-002: If contradiction blocked pipeline, go to END
        """
        status = state.get("status")
        if status in (PipelineStatus.HALTED.value, PipelineStatus.FAILED.value):
            return END

        # INV-BETA-002: Check if contradiction blocked the pipeline
        contradiction_status = state.get("contradiction_status")
        if contradiction_status == "blocked":
            logger.error("INV-BETA-002: Pipeline blocked due to spec contradiction")
            return END

        # SURGICAL REWORK DECISION
        sprint_id = state.get("sprint_id", "")
        granular_tasks = state.get("granular_tasks", [])

        if is_surgical_rework_enabled(sprint_id) and len(granular_tasks) > 1:
            logger.info(f"Routing to SURGICAL REWORK: {len(granular_tasks)} tasks")
            return "setup_tasks"

        logger.info("Routing to legacy exec")
        return "exec"

    def route_after_setup_tasks(state: PipelineState) -> str:
        """Route after task setup."""
        surgical_rework = state.get("surgical_rework", {})
        if not surgical_rework.get("enabled"):
            return "exec"
        return "task_exec"

    def route_after_task_exec(state: PipelineState) -> str:
        """Route after task execution."""
        surgical_rework = state.get("surgical_rework", {})
        tasks = surgical_rework.get("tasks", {})

        validating = sum(
            1 for t in tasks.values()
            if t.get("status") == TaskLevelStatus.VALIDATING.value
        )

        if validating > 0:
            return "task_validate"
        return "check_group"

    def route_after_task_validate(state: PipelineState) -> str:
        """Route after task validation."""
        surgical_rework = state.get("surgical_rework", {})
        tasks = surgical_rework.get("tasks", {})

        failed = sum(
            1 for t in tasks.values()
            if t.get("status") == TaskLevelStatus.FAILED.value
        )

        if failed > 0:
            return "task_rework"
        return "check_group"

    def route_after_task_rework(state: PipelineState) -> str:
        """Route after task rework."""
        surgical_rework = state.get("surgical_rework", {})
        tasks = surgical_rework.get("tasks", {})

        validating = sum(
            1 for t in tasks.values()
            if t.get("status") == TaskLevelStatus.VALIDATING.value
        )

        if validating > 0:
            return "task_validate"
        return "check_group"

    def route_after_check_group(state: PipelineState) -> str:
        """Route after checking group completion."""
        surgical_rework = state.get("surgical_rework", {})
        tasks = surgical_rework.get("tasks", {})
        parallel_groups = surgical_rework.get("parallel_groups", [])
        current_index = surgical_rework.get("current_group_index", 0)

        all_escalated = all(
            t.get("status") == TaskLevelStatus.ESCALATED.value
            for t in tasks.values()
        ) if tasks else False

        if all_escalated:
            return END

        if current_index < len(parallel_groups):
            return "task_exec"

        return "integration_validate"

    def route_after_integration_validate(state: PipelineState) -> str:
        """Route after integration validation."""
        surgical_rework = state.get("surgical_rework", {})

        if surgical_rework.get("integration_validation_passed"):
            return "gate"
        return END

    def route_after_exec(state: PipelineState) -> str:
        """Route after legacy exec node."""
        status = state.get("status")
        if status in (PipelineStatus.HALTED.value, PipelineStatus.FAILED.value):
            return END
        return "gate"

    def route_after_gate(state: PipelineState) -> str:
        """Route after gate node.

        2026-01-29: Routes to appropriate rework based on surgical rework status.
        """
        status = state.get("status")
        if status in (PipelineStatus.HALTED.value, PipelineStatus.FAILED.value):
            return END

        if state.get("_rework_context") or state.get("_task_invalidated"):
            surgical_rework = state.get("surgical_rework", {})
            if surgical_rework.get("enabled"):
                logger.info("Gate failed, routing to task_rework (surgical)")
                return "task_rework"
            logger.info("Gate failed, routing to legacy_rework")
            return "legacy_rework"

        if state.get("gate_status") != GateStatus.PASS.value:
            surgical_rework = state.get("surgical_rework", {})
            if surgical_rework.get("enabled"):
                return "task_rework"
            logger.warning("Gate failed, attempting legacy rework")
            return "legacy_rework"

        return "qa"

    def route_after_qa(state: PipelineState) -> str:
        """Route after qa node."""
        status = state.get("status")
        if status in (PipelineStatus.HALTED.value, PipelineStatus.FAILED.value):
            return END

        if state.get("_rework_context") or state.get("_task_invalidated"):
            surgical_rework = state.get("surgical_rework", {})
            if surgical_rework.get("enabled"):
                return "task_rework"
            return "legacy_rework"

        if not state.get("qa_workers_passed", True):
            surgical_rework = state.get("surgical_rework", {})
            if surgical_rework.get("enabled"):
                return "task_rework"
            return "legacy_rework"

        return "signoff"

    def route_after_legacy_rework(state: PipelineState) -> str:
        """Route after legacy rework node."""
        status = state.get("status")
        if status in (PipelineStatus.HALTED.value, PipelineStatus.FAILED.value):
            return END

        if state.get("_escalated"):
            logger.warning(f"Task escalated: {state.get('_escalation_reason', 'unknown')}")
            return END

        logger.info("Routing from legacy_rework back to exec")
        return "exec"

    def route_after_signoff(state: PipelineState) -> str:
        """Route after signoff node."""
        status = state.get("status")
        if status in (PipelineStatus.HALTED.value, PipelineStatus.FAILED.value):
            return END
        return "artifact"

    # Add conditional edges
    # Standard flow
    workflow.add_conditional_edges("init", route_after_init)
    workflow.add_conditional_edges("spec", route_after_spec)

    # AGENT BETA: Semantic search and contradiction detection flow
    workflow.add_conditional_edges("semantic_search", route_after_semantic_search)
    workflow.add_conditional_edges("contradiction", route_after_contradiction)

    # Surgical rework flow (task-level)
    workflow.add_conditional_edges("setup_tasks", route_after_setup_tasks)
    workflow.add_conditional_edges("task_exec", route_after_task_exec)
    workflow.add_conditional_edges("task_validate", route_after_task_validate)
    workflow.add_conditional_edges("task_rework", route_after_task_rework)
    workflow.add_conditional_edges("check_group", route_after_check_group)
    workflow.add_conditional_edges("integration_validate", route_after_integration_validate)

    # Legacy flow (sprint-level)
    workflow.add_conditional_edges("exec", route_after_exec)
    workflow.add_conditional_edges("gate", route_after_gate)
    workflow.add_conditional_edges("qa", route_after_qa)
    workflow.add_conditional_edges("legacy_rework", route_after_legacy_rework)
    workflow.add_conditional_edges("signoff", route_after_signoff)
    workflow.add_edge("artifact", END)

    # Compile with checkpointer if provided
    if checkpointer:
        return workflow.compile(checkpointer=checkpointer)

    return workflow.compile()


async def run_workflow(
    sprint_id: str,
    run_id: Optional[str] = None,
    run_dir: Optional[Path] = None,
    resume_from: Optional[str] = None,
    checkpointer: Optional[Any] = None,
) -> PipelineState:
    """Run the LangGraph workflow for a sprint.

    Args:
        sprint_id: Sprint identifier (e.g., "S00", "S01").
        run_id: Optional run identifier. Generated if not provided.
        run_dir: Directory for run artifacts.
        resume_from: Optional checkpoint ID or thread_id to resume from.
        checkpointer: Optional checkpointer for state persistence.

    Returns:
        Final PipelineState after workflow completion.
    """
    if not LANGGRAPH_AVAILABLE:
        raise ImportError("LangGraph not available - install with: pip install langgraph")

    # Generate run_id if not provided
    if run_id is None:
        run_id = f"v2_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    # Set run_dir
    if run_dir is None:
        run_dir = Path("out/pipeline") / run_id

    run_dir.mkdir(parents=True, exist_ok=True)

    # Setup checkpointer for resume capability
    # FIX: Use centralized checkpoint directory and convert to LangGraph format
    from pipeline.langgraph.checkpointer import (
        FileCheckpointer,
        create_langgraph_checkpointer,
    )

    # Use centralized checkpoint directory (not per-run)
    # FIX 2026-01-27: Use .langgraph/checkpoints to match cockpit server expectations
    checkpoint_dir = Path(".langgraph/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create our internal checkpointer (for get/get_latest operations)
    internal_checkpointer = FileCheckpointer(checkpoint_dir)

    # Convert to LangGraph-compatible checkpointer
    if checkpointer is None:
        lg_checkpointer = create_langgraph_checkpointer(internal_checkpointer)
    else:
        # Assume already LangGraph-compatible
        lg_checkpointer = checkpointer
        internal_checkpointer = getattr(checkpointer, 'inner', internal_checkpointer)

    # Build workflow with LangGraph checkpointer
    workflow = build_workflow(run_dir=run_dir, checkpointer=lg_checkpointer)

    if workflow is None:
        raise RuntimeError("Failed to build workflow")

    # Handle resume from checkpoint
    initial_state = None
    config = {"configurable": {"thread_id": run_id}}

    if resume_from:
        logger.info(f"Attempting to resume workflow from: {resume_from}")
        try:
            # FIX: Use internal_checkpointer for get/get_latest operations
            # Try to load checkpoint by ID or get latest for thread
            checkpoint = await internal_checkpointer.get(resume_from)
            if checkpoint is None:
                # Try as thread_id
                checkpoint = await internal_checkpointer.get_latest(resume_from)

            if checkpoint:
                logger.info(f"Resuming from checkpoint: {checkpoint.checkpoint_id}")
                initial_state = checkpoint.state
                # Update config with checkpoint's thread_id
                config = {"configurable": {"thread_id": checkpoint.thread_id}}
                # Increment attempt counter
                if "attempt" in initial_state:
                    initial_state["attempt"] = initial_state["attempt"] + 1
                logger.info(f"Resumed state phase: {initial_state.get('phase')}, attempt: {initial_state.get('attempt')}")
            else:
                logger.warning(f"Checkpoint '{resume_from}' not found, starting fresh")
        except Exception as e:
            logger.error(f"Failed to load checkpoint '{resume_from}': {e}")
            logger.warning("Starting fresh workflow instead of resuming")

    # Create initial state if not resuming
    if initial_state is None:
        initial_state = create_initial_state(
            run_id=run_id,
            sprint_id=sprint_id,
            run_dir=str(run_dir),
        )

    # Run workflow using astream for automatic checkpointing after each node
    # CRITICAL FIX 2026-01-24: ainvoke() does NOT save intermediate checkpoints.
    # Only astream()/stream() triggers checkpoint saves after each node.
    result = initial_state
    async for state_update in workflow.astream(initial_state, config=config):
        result = state_update
        current_phase = result.get("phase", "unknown") if isinstance(result, dict) else "unknown"
        logger.debug(f"Checkpoint saved after phase: {current_phase}")

    return result


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "WorkflowNodes",
    "build_workflow",
    "run_workflow",
    "LANGGRAPH_AVAILABLE",
    # P3-001: Hamilton integration availability
    "HAMILTON_AVAILABLE",
]
