"""Enforcement Integration for Workflow Nodes.

This module provides the integration layer between the corrective enforcement
system and the LangGraph workflow nodes.

CRITICAL PHILOSOPHY (Updated 2026-01-16):
    Guardrails BLOCK pipeline advancement when violations are detected.
    The task is INVALIDATED entirely and sent for REWORK with:
    - Full violation report
    - Error context for the agent
    - Learnings from Reflexion
    - Rework instructions

    "The guardrail does NOT let the pipeline advance if there's an error.
     It MUST send the agent that caused the error to REDO the work,
     with awareness of the error and memory/learning of errors."

Architecture:
    WorkflowNode
         |
         v
    EnforcementWrapper
         |
    ┌────┴────┐
    v         v
  Input    Guardrail
 Validate   Check
    |         |
    v         v
  [PASS]   [VIOLATION]
    |         |
    v         v
  Execute   INVALIDATE
    Node     TASK
    |         |
    v         v
  Output    Generate
 Validate   Report
    |         |
    v         v
  [PASS]    Queue
    |      REWORK
    v         |
  Return    Agent
  Result   REWORKS
             |
             v
          Retry with
          Error Awareness

Usage:
    from pipeline.langgraph.enforcement_integration import (
        with_enforcement,
        EnforcedWorkflowNodes,
    )

    # Wrap existing workflow nodes
    enforced_nodes = EnforcedWorkflowNodes(existing_nodes)

    # Or use decorator
    @with_enforcement
    async def my_node(state):
        ...

Author: Pipeline Autonomo Team
Version: 2.0.0 (2026-01-16) - BLOCK AND REWORK instead of log and continue
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar

from pipeline.langgraph.corrective_enforcement import (
    CorrectionAction,
    CorrectionContext,
    EnforcementResult,
    EnforcementType,
    EscalationLevel,
    create_correction_context,
    get_enforcement_layer,
)
from pipeline.langgraph.state import (
    PipelineState,
)
from pipeline.langgraph.guardrail_violation_handler import (
    ViolationBuilder,
    ViolationType,
    ViolationSeverity,
    ViolationDetails,
    TaskInvalidation,
    ReworkTask,
    get_violation_handler,
    create_security_violation,
    create_invariant_violation,
    create_validation_violation,
    create_gate_violation,
)

# Targeted Fix imports (2026-01-30)
try:
    from pipeline.langgraph.targeted_fix import (
        can_apply_targeted_fix,
        should_use_targeted_fix,
        GateFailureType,
    )
    TARGETED_FIX_AVAILABLE = True
except ImportError:
    TARGETED_FIX_AVAILABLE = False
    can_apply_targeted_fix = None
    should_use_targeted_fix = None

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# HELPER: CONVERT ENFORCEMENT RESULTS TO VIOLATIONS
# =============================================================================


def _collect_violations_from_results(
    results: List[EnforcementResult],
    context: CorrectionContext,
) -> List[ViolationDetails]:
    """Convert enforcement results into violation details.

    Only results that indicate ESCALATE or RETRY are converted to violations.
    PASS and CORRECTED results are not violations.

    Args:
        results: List of enforcement results
        context: Correction context

    Returns:
        List of ViolationDetails for blocking violations
    """
    violations = []

    for result in results:
        # Only ESCALATE and RETRY actions indicate blocking violations
        if result.action in [CorrectionAction.ESCALATE, CorrectionAction.RETRY]:
            # Map enforcement type to violation type
            # NOTE: EnforcementType enum values are: INVARIANT, SECURITY,
            # TRUST_BOUNDARY, CONTENT_FILTER, STACK_HEALTH, VALIDATION
            violation_type_map = {
                EnforcementType.VALIDATION: ViolationType.VALIDATION,
                EnforcementType.INVARIANT: ViolationType.INVARIANT,
                EnforcementType.TRUST_BOUNDARY: ViolationType.TRUST,
                EnforcementType.SECURITY: ViolationType.SECURITY,
                EnforcementType.CONTENT_FILTER: ViolationType.CONTENT,
                EnforcementType.STACK_HEALTH: ViolationType.VALIDATION,
            }

            # Map escalation level to severity
            severity_map = {
                EscalationLevel.SAFE_MODE: ViolationSeverity.CRITICAL,
                EscalationLevel.REQUIRE_APPROVAL: ViolationSeverity.HIGH,
                EscalationLevel.NOTIFY_HUMAN: ViolationSeverity.HIGH,
                EscalationLevel.LOG_WARNING: ViolationSeverity.MEDIUM,
                EscalationLevel.NONE: ViolationSeverity.LOW,
            }

            # Get suggestion from metadata if available, otherwise use default
            suggestion = (
                result.metadata.get("suggestion", "")
                if result.metadata
                else ""
            ) or "Review the violation and fix the underlying issue"

            violation = ViolationBuilder(
                violation_type=violation_type_map.get(
                    result.enforcement_type, ViolationType.VALIDATION
                ),
                severity=severity_map.get(
                    result.escalation_level, ViolationSeverity.MEDIUM
                ),
                message=result.message,
                guardrail_name=f"{result.enforcement_type.value}_guardrail",
            ).with_context(
                "enforcement_type", result.enforcement_type.value
            ).with_context(
                "escalation_level", result.escalation_level.value
            ).with_context(
                "node", context.node
            ).with_context(
                "attempt", context.attempt
            ).with_explanation(
                what_went_wrong=result.message,
                why_it_matters=f"Guardrail {result.enforcement_type.value} detected an issue that requires rework",
                how_to_fix=suggestion,
            ).build()

            violations.append(violation)

    return violations


async def _check_output_violations(
    results: List[EnforcementResult],
    context: CorrectionContext,
    state: Dict[str, Any],
) -> Optional[tuple[TaskInvalidation, ReworkTask]]:
    """Check output enforcement results for violations.

    If violations are found, triggers task invalidation and rework.

    Args:
        results: List of output enforcement results
        context: Correction context
        state: Current state

    Returns:
        Tuple of (TaskInvalidation, ReworkTask) if violations found, else None
    """
    violations = _collect_violations_from_results(results, context)

    if violations:
        handler = get_violation_handler()
        return await handler.handle_violation(
            violations=violations,
            task_id=context.run_id,
            run_id=context.run_id,
            sprint_id=context.sprint_id,
            node_name=context.node,
            state=state,
            agent_id=context.agent_id,
        )

    return None


# =============================================================================
# NODE WRAPPER
# =============================================================================


def with_enforcement(
    node_name: Optional[str] = None,
    enforce_input: bool = True,
    enforce_output: bool = True,
    enforce_invariants: bool = True,
    enforce_trust: bool = True,
):
    """Decorator to wrap a node with enforcement.

    Args:
        node_name: Name of the node (defaults to function name)
        enforce_input: Whether to enforce input guardrails
        enforce_output: Whether to enforce output guardrails
        enforce_invariants: Whether to check invariants
        enforce_trust: Whether to check trust boundaries

    Example:
        @with_enforcement(node_name="exec")
        async def exec_node(state: PipelineState) -> PipelineState:
            ...
    """
    def decorator(func: Callable[[PipelineState], Awaitable[PipelineState]]):
        @functools.wraps(func)
        async def wrapper(state: PipelineState) -> PipelineState:
            name = node_name or func.__name__

            # Create correction context
            context = create_correction_context(
                run_id=state.get("run_id", "unknown"),
                sprint_id=state.get("sprint_id", "unknown"),
                node=name,
                attempt=state.get("attempt", 1),
                max_attempts=state.get("max_attempts", 3),
                agent_id=state.get("agent_id"),
            )

            layer = get_enforcement_layer()
            corrected_state = dict(state)
            all_results: List[EnforcementResult] = []

            # 1. Enforce invariants on state
            if enforce_invariants:
                corrected_state, inv_results = await layer.enforce_invariants(
                    corrected_state, context
                )
                all_results.extend(inv_results)

            # 2. Enforce input guardrails
            if enforce_input:
                # Check input_data field if present
                input_data = corrected_state.get("input_data")
                if input_data is not None:
                    corrected_input, input_results = await layer.enforce_input(
                        input_data, context
                    )
                    corrected_state["input_data"] = corrected_input
                    all_results.extend(input_results)

                # Check claim field if present
                claim = corrected_state.get("claim")
                if claim is not None and isinstance(claim, str):
                    corrected_claim, claim_results = await layer.enforce_input(
                        claim, context
                    )
                    corrected_state["claim"] = corrected_claim
                    all_results.extend(claim_results)

            # 3. Enforce trust boundaries
            if enforce_trust:
                agent_id = corrected_state.get("agent_id", "pipeline")
                allowed, effective_action, trust_result = await layer.enforce_trust_boundary(
                    agent_id=agent_id,
                    resource=f"node:{name}",
                    action="execute",
                    context=context,
                )
                all_results.append(trust_result)

                # Store effective action (may be downgraded)
                corrected_state["_effective_action"] = effective_action

            # Record enforcement results in state
            enforcement_log = corrected_state.get("_enforcement_log", [])
            enforcement_log.extend([
                {
                    "node": name,
                    "type": r.enforcement_type.value,
                    "action": r.action.value,
                    "message": r.message,
                    "timestamp": r.timestamp,
                }
                for r in all_results
            ])
            corrected_state["_enforcement_log"] = enforcement_log

            # 4. Check for violations BEFORE executing node
            violations = _collect_violations_from_results(all_results, context)

            if violations:
                # BLOCK: Do NOT execute the node if there are violations
                logger.warning(
                    f"Node {name}: {len(violations)} guardrail violations detected. "
                    f"BLOCKING execution and triggering REWORK."
                )

                # Handle violation - invalidate task and queue for rework
                handler = get_violation_handler()
                invalidation, rework = await handler.handle_violation(
                    violations=violations,
                    task_id=context.run_id,
                    run_id=context.run_id,
                    sprint_id=context.sprint_id,
                    node_name=name,
                    state=corrected_state,
                    agent_id=context.agent_id,
                )

                # Mark state as INVALIDATED - pipeline must handle rework
                result_state = dict(corrected_state)
                result_state["_task_invalidated"] = True
                result_state["_invalidation_id"] = invalidation["invalidation_id"]
                result_state["_rework_id"] = rework["rework_id"]
                result_state["_violation_count"] = len(violations)
                result_state["_rework_instructions"] = rework["rework_instructions"]

                # Add rework context for the agent
                result_state["_rework_context"] = {
                    "violation_report": rework["violation_report"],
                    "retry_hints": rework["retry_hints"],
                    "error_learnings": rework["error_learnings"],
                    "attempt_number": rework["attempt_number"],
                    "max_attempts": rework["max_attempts"],
                }

                # Do NOT continue execution - return with invalidation marker
                return result_state

            # 4. Execute the actual node (only if no violations)
            try:
                result_state = await func(corrected_state)
            except Exception as e:
                logger.error(f"Node {name} failed with exception: {e}")

                # Convert exception to violation and trigger rework
                violation = ViolationBuilder(
                    violation_type=ViolationType.VALIDATION,
                    severity=ViolationSeverity.HIGH,
                    message=str(e),
                    guardrail_name="ExecutionGuard",
                ).with_explanation(
                    what_went_wrong=f"Node execution failed: {e}",
                    why_it_matters="Execution errors prevent task completion",
                    how_to_fix="Review the error and fix the root cause",
                ).build()

                handler = get_violation_handler()
                invalidation, rework = await handler.handle_violation(
                    violations=[violation],
                    task_id=context.run_id,
                    run_id=context.run_id,
                    sprint_id=context.sprint_id,
                    node_name=name,
                    state=corrected_state,
                    agent_id=context.agent_id,
                )

                # Mark state for rework
                result_state = dict(corrected_state)
                result_state["_task_invalidated"] = True
                result_state["_invalidation_id"] = invalidation["invalidation_id"]
                result_state["_rework_id"] = rework["rework_id"]
                result_state["_node_error"] = str(e)
                result_state["_rework_instructions"] = rework["rework_instructions"]
                result_state["_rework_context"] = {
                    "violation_report": rework["violation_report"],
                    "retry_hints": rework["retry_hints"],
                    "error_learnings": rework["error_learnings"],
                    "attempt_number": rework["attempt_number"],
                    "max_attempts": rework["max_attempts"],
                }

                # Do NOT continue - return with invalidation
                return result_state

            # 5. Enforce output guardrails
            if enforce_output:
                # Check output_data field if present
                output_data = result_state.get("output_data")
                if output_data is not None:
                    corrected_output, output_results = await layer.enforce_output(
                        output_data, context
                    )
                    result_state["output_data"] = corrected_output
                    all_results.extend(output_results)

                # Check verdict field if present
                verdict = result_state.get("verdict")
                if verdict is not None and isinstance(verdict, str):
                    corrected_verdict, verdict_results = await layer.enforce_output(
                        verdict, context
                    )
                    result_state["verdict"] = corrected_verdict
                    all_results.extend(verdict_results)

            # 6. Check for output violations after execution
            output_violations = _collect_violations_from_results(all_results, context)

            if output_violations:
                # BLOCK: Output violated guardrails - invalidate and rework
                logger.warning(
                    f"Node {name}: {len(output_violations)} OUTPUT violations detected. "
                    f"BLOCKING result and triggering REWORK."
                )

                handler = get_violation_handler()
                invalidation, rework = await handler.handle_violation(
                    violations=output_violations,
                    task_id=context.run_id,
                    run_id=context.run_id,
                    sprint_id=context.sprint_id,
                    node_name=name,
                    state=result_state,
                    agent_id=context.agent_id,
                )

                # Mark result state as invalidated
                result_state["_task_invalidated"] = True
                result_state["_invalidation_id"] = invalidation["invalidation_id"]
                result_state["_rework_id"] = rework["rework_id"]
                result_state["_violation_count"] = len(output_violations)
                result_state["_rework_instructions"] = rework["rework_instructions"]
                result_state["_rework_context"] = {
                    "violation_report": rework["violation_report"],
                    "retry_hints": rework["retry_hints"],
                    "error_learnings": rework["error_learnings"],
                    "attempt_number": rework["attempt_number"],
                    "max_attempts": rework["max_attempts"],
                }

                return result_state

            # Log corrections summary (only for non-blocking results)
            corrections = [r for r in all_results if r.action == CorrectionAction.CORRECTED]
            escalations = [r for r in all_results if r.escalation_level != EscalationLevel.NONE]

            if corrections:
                logger.info(f"Node {name}: Applied {len(corrections)} auto-corrections")
            if escalations:
                logger.warning(f"Node {name}: {len(escalations)} items may need review")

            return result_state

        return wrapper
    return decorator


# =============================================================================
# ENFORCED WORKFLOW NODES
# =============================================================================


class EnforcedWorkflowNodes:
    """Wrapper that adds enforcement to existing workflow nodes.

    This class wraps an existing WorkflowNodes instance and adds
    corrective enforcement to all node executions.
    """

    def __init__(self, workflow_nodes: Any):
        """Initialize with existing workflow nodes.

        Args:
            workflow_nodes: Existing WorkflowNodes instance
        """
        self._nodes = workflow_nodes
        self._enforcement_layer = get_enforcement_layer()
        self._enforcement_stats: Dict[str, int] = {
            "total_executions": 0,
            "corrections_applied": 0,
            "escalations": 0,
        }

    async def init_node(self, state: PipelineState) -> PipelineState:
        """Enforced init node."""
        return await self._execute_with_enforcement(
            "init",
            self._nodes.init_node,
            state,
        )

    async def spec_node(self, state: PipelineState) -> PipelineState:
        """Enforced spec node (GAP-3: spec_kit decomposition)."""
        return await self._execute_with_enforcement(
            "spec",
            self._nodes.spec_node,
            state,
        )

    async def exec_node(self, state: PipelineState) -> PipelineState:
        """Enforced exec node."""
        return await self._execute_with_enforcement(
            "exec",
            self._nodes.exec_node,
            state,
        )

    async def gate_node(self, state: PipelineState) -> PipelineState:
        """Enforced gate node with security integration."""
        # Additional security enforcement for gates
        state = await self._enforce_gate_security(state)

        return await self._execute_with_enforcement(
            "gate",
            self._nodes.gate_node,
            state,
        )

    async def qa_node(self, state: PipelineState) -> PipelineState:
        """Enforced QA node - runs QA workers in parallel via QAMasterOrchestrator.

        FIX 2026-01-28: Dedicated qa_node for QA workers (separated from gate_node).

        The 8 QA workers are:
        - auditor (CRITICAL): Code quality audit
        - agente_auditor: Secondary audit
        - refinador (IMPORTANT): Code refinement review
        - clean_reviewer (IMPORTANT): Clean code review
        - edge_case_hunter (IMPORTANT): Edge case validation
        - gap_hunter (CRITICAL): Gap detection
        - human_reviewer: Human-like review
        - debt_tracker: Technical debt tracking
        """
        return await self._execute_with_enforcement(
            "qa",
            self._nodes.qa_node,
            state,
        )

    async def signoff_node(self, state: PipelineState) -> PipelineState:
        """Enforced signoff node."""
        # Check I4: Gates must pass before signoff
        state = await self._enforce_signoff_requirements(state)

        return await self._execute_with_enforcement(
            "signoff",
            self._nodes.signoff_node,
            state,
        )

    async def artifact_node(self, state: PipelineState) -> PipelineState:
        """Enforced artifact node."""
        return await self._execute_with_enforcement(
            "artifact",
            self._nodes.artifact_node,
            state,
        )

    async def _execute_with_enforcement(
        self,
        node_name: str,
        node_func: Callable[[PipelineState], Awaitable[PipelineState]],
        state: PipelineState,
    ) -> PipelineState:
        """Execute a node with full enforcement.

        CRITICAL: This method BLOCKS on violations and triggers REWORK.
        The pipeline does NOT advance if guardrails detect issues.

        Args:
            node_name: Name of the node
            node_func: The node function to execute
            state: Current pipeline state

        Returns:
            Updated state with rework context if violations detected
        """
        self._enforcement_stats["total_executions"] += 1

        # Create context
        context = create_correction_context(
            run_id=state.get("run_id", "unknown"),
            sprint_id=state.get("sprint_id", "unknown"),
            node=node_name,
            attempt=state.get("attempt", 1),
            max_attempts=state.get("max_attempts", 3),
            agent_id=state.get("agent_id"),
        )

        # Pre-execution enforcement
        corrected_state, pre_results = await self._pre_execution_enforcement(state, context)

        # CHECK FOR PRE-EXECUTION VIOLATIONS - BLOCK IF FOUND
        pre_violations = _collect_violations_from_results(pre_results, context)
        if pre_violations:
            logger.warning(
                f"Node {node_name}: {len(pre_violations)} PRE-EXECUTION violations. "
                f"BLOCKING and triggering REWORK."
            )

            handler = get_violation_handler()
            invalidation, rework = await handler.handle_violation(
                violations=pre_violations,
                task_id=context.run_id,
                run_id=context.run_id,
                sprint_id=context.sprint_id,
                node_name=node_name,
                state=corrected_state,
                agent_id=context.agent_id,
            )

            self._enforcement_stats["escalations"] += 1

            # Return state with rework context
            result_state = dict(corrected_state)
            result_state["_task_invalidated"] = True
            result_state["_invalidation_id"] = invalidation["invalidation_id"]
            result_state["_rework_id"] = rework["rework_id"]
            result_state["_violation_count"] = len(pre_violations)
            result_state["_rework_instructions"] = rework["rework_instructions"]
            result_state["_rework_context"] = {
                "violation_report": rework["violation_report"],
                "retry_hints": rework["retry_hints"],
                "error_learnings": rework["error_learnings"],
                "attempt_number": rework["attempt_number"],
                "max_attempts": rework["max_attempts"],
            }
            return result_state

        # Execute node (only if no pre-violations)
        try:
            result_state = await node_func(corrected_state)
        except Exception as e:
            logger.error(f"Node {node_name} execution failed: {e}")

            # BLOCK: Convert exception to violation and trigger rework
            violation = ViolationBuilder(
                violation_type=ViolationType.VALIDATION,
                severity=ViolationSeverity.HIGH,
                message=str(e),
                guardrail_name="ExecutionGuard",
            ).with_explanation(
                what_went_wrong=f"Node {node_name} failed: {e}",
                why_it_matters="Execution errors prevent task completion",
                how_to_fix="Review error details and fix the root cause",
            ).build()

            handler = get_violation_handler()
            invalidation, rework = await handler.handle_violation(
                violations=[violation],
                task_id=context.run_id,
                run_id=context.run_id,
                sprint_id=context.sprint_id,
                node_name=node_name,
                state=corrected_state,
                agent_id=context.agent_id,
            )

            self._enforcement_stats["escalations"] += 1

            # Return state with rework context
            result_state = dict(corrected_state)
            result_state["_task_invalidated"] = True
            result_state["_invalidation_id"] = invalidation["invalidation_id"]
            result_state["_rework_id"] = rework["rework_id"]
            result_state["_node_error"] = str(e)
            result_state["_rework_instructions"] = rework["rework_instructions"]
            result_state["_rework_context"] = {
                "violation_report": rework["violation_report"],
                "retry_hints": rework["retry_hints"],
                "error_learnings": rework["error_learnings"],
                "attempt_number": rework["attempt_number"],
                "max_attempts": rework["max_attempts"],
            }
            return result_state

        # Post-execution enforcement
        result_state, post_results = await self._post_execution_enforcement(
            result_state, context
        )

        # CHECK FOR POST-EXECUTION VIOLATIONS - BLOCK IF FOUND
        post_violations = _collect_violations_from_results(post_results, context)
        if post_violations:
            logger.warning(
                f"Node {node_name}: {len(post_violations)} POST-EXECUTION violations. "
                f"BLOCKING result and triggering REWORK."
            )

            handler = get_violation_handler()
            invalidation, rework = await handler.handle_violation(
                violations=post_violations,
                task_id=context.run_id,
                run_id=context.run_id,
                sprint_id=context.sprint_id,
                node_name=node_name,
                state=result_state,
                agent_id=context.agent_id,
            )

            self._enforcement_stats["escalations"] += 1

            result_state["_task_invalidated"] = True
            result_state["_invalidation_id"] = invalidation["invalidation_id"]
            result_state["_rework_id"] = rework["rework_id"]
            result_state["_violation_count"] = len(post_violations)
            result_state["_rework_instructions"] = rework["rework_instructions"]
            result_state["_rework_context"] = {
                "violation_report": rework["violation_report"],
                "retry_hints": rework["retry_hints"],
                "error_learnings": rework["error_learnings"],
                "attempt_number": rework["attempt_number"],
                "max_attempts": rework["max_attempts"],
            }
            return result_state

        # Update stats (no violations)
        corrections = len([r for r in pre_results + post_results
                         if r.action == CorrectionAction.CORRECTED])
        self._enforcement_stats["corrections_applied"] += corrections

        return result_state

    async def _pre_execution_enforcement(
        self,
        state: PipelineState,
        context: CorrectionContext,
    ) -> tuple[PipelineState, List[EnforcementResult]]:
        """Enforce guardrails before node execution."""
        all_results: List[EnforcementResult] = []
        corrected_state = dict(state)

        # 1. Invariant checks
        corrected_state, inv_results = await self._enforcement_layer.enforce_invariants(
            corrected_state, context
        )
        all_results.extend(inv_results)

        # 2. Input enforcement
        input_data = corrected_state.get("input_data")
        if input_data is not None:
            corrected_input, input_results = await self._enforcement_layer.enforce_input(
                input_data, context
            )
            corrected_state["input_data"] = corrected_input
            all_results.extend(input_results)

        # 3. Claim text enforcement (security)
        claim = corrected_state.get("claim")
        if claim is not None and isinstance(claim, str):
            corrected_claim, claim_results = await self._enforcement_layer.enforce_input(
                claim, context
            )
            corrected_state["claim"] = corrected_claim
            all_results.extend(claim_results)

        # 4. Trust boundary check
        agent_id = corrected_state.get("agent_id", "pipeline")
        allowed, action, trust_result = await self._enforcement_layer.enforce_trust_boundary(
            agent_id, f"node:{context.node}", "execute", context
        )
        all_results.append(trust_result)

        return corrected_state, all_results

    async def _post_execution_enforcement(
        self,
        state: PipelineState,
        context: CorrectionContext,
    ) -> tuple[PipelineState, List[EnforcementResult]]:
        """Enforce guardrails after node execution."""
        all_results: List[EnforcementResult] = []
        corrected_state = dict(state)

        # 1. Output enforcement
        output_data = corrected_state.get("output_data")
        if output_data is not None:
            corrected_output, output_results = await self._enforcement_layer.enforce_output(
                output_data, context
            )
            corrected_state["output_data"] = corrected_output
            all_results.extend(output_results)

        # 2. Verdict enforcement (check for secrets/PII)
        verdict = corrected_state.get("verdict")
        if verdict is not None and isinstance(verdict, str):
            corrected_verdict, verdict_results = await self._enforcement_layer.enforce_output(
                verdict, context
            )
            corrected_state["verdict"] = corrected_verdict
            all_results.extend(verdict_results)

        # 3. Response text enforcement
        response = corrected_state.get("response")
        if response is not None and isinstance(response, str):
            corrected_response, response_results = await self._enforcement_layer.enforce_output(
                response, context
            )
            corrected_state["response"] = corrected_response
            all_results.extend(response_results)

        return corrected_state, all_results

    async def _enforce_gate_security(
        self,
        state: PipelineState,
    ) -> PipelineState:
        """Additional security enforcement for gate node.

        Uses NeMo and LLM Guard for comprehensive security checks.
        """
        corrected_state = dict(state)

        # Try to use NeMo for jailbreak detection
        try:
            from pipeline.security.nemo_enhanced import get_nemo_enhanced, NEMO_ENHANCED_AVAILABLE

            if NEMO_ENHANCED_AVAILABLE:
                nemo = get_nemo_enhanced()
                claim = corrected_state.get("claim", "")

                if claim:
                    # Check for jailbreak
                    jailbreak_result = await nemo.detect_jailbreak(claim)

                    if jailbreak_result.get("is_jailbreak", False):
                        logger.warning(
                            f"Jailbreak detected in claim: {jailbreak_result.get('jailbreak_type')}"
                        )
                        # Neutralize instead of block
                        neutralized = await nemo.neutralize_jailbreak(claim)
                        corrected_state["claim"] = neutralized.get("neutralized_text", claim)
                        corrected_state["_security_flag"] = "jailbreak_neutralized"

                    # Check content
                    content_result = await nemo.filter_content(claim)

                    if not content_result.get("is_safe", True):
                        logger.warning(
                            f"Content flagged: {content_result.get('categories', [])}"
                        )
                        # Sanitize instead of block
                        corrected_state["claim"] = content_result.get("filtered_text", claim)
                        corrected_state["_security_flag"] = "content_filtered"

        except ImportError:
            logger.debug("NeMo enhanced not available for gate security")
        except Exception as e:
            logger.debug(f"NeMo security check failed: {e}")

        # Try LLM Guard as backup/additional layer
        try:
            from pipeline.security.llm_guard_integration import (
                get_security_orchestrator,
                LLM_GUARD_INTEGRATION_AVAILABLE,
            )

            if LLM_GUARD_INTEGRATION_AVAILABLE:
                orchestrator = get_security_orchestrator()
                claim = corrected_state.get("claim", "")

                if claim:
                    result = await orchestrator.run_security_checks(input_text=claim)

                    if result.get("blocked", False):
                        # Don't block - sanitize
                        corrected_state["claim"] = result.get("sanitized_input", claim)
                        corrected_state["_llm_guard_flag"] = result.get("block_reason", "flagged")

        except ImportError:
            logger.debug("LLM Guard not available for gate security")
        except Exception as e:
            logger.debug(f"LLM Guard check failed: {e}")

        return corrected_state

    async def _enforce_signoff_requirements(
        self,
        state: PipelineState,
    ) -> PipelineState:
        """Enforce I4: Gates must pass before signoff.

        If gates haven't passed, redirect back to gate node instead of blocking.
        """
        corrected_state = dict(state)

        gate_status = corrected_state.get("gate_status", "")
        gates_passed = corrected_state.get("gates_passed", [])

        # FIX 2026-01-23: Support both legacy (G0, G1, G2) and new (GATE-*) naming
        # Check for legacy gates OR new GATE-* prefixed gates
        legacy_gates = ["G0", "G1", "G2"]
        has_legacy_gates = any(g in gates_passed for g in legacy_gates)
        has_new_gates = any(str(g).startswith("GATE-") for g in gates_passed)

        # Normalize gate_status comparison (handle "passed", "PASS", "PASSED", etc.)
        gate_passed = gate_status.lower() in ("pass", "passed") if gate_status else False

        # Signoff is allowed if:
        # 1. gate_status indicates pass, OR
        # 2. At least one gate has passed (legacy or new naming)
        has_any_gates = has_legacy_gates or has_new_gates or len(gates_passed) > 0

        if not gate_passed and not has_any_gates:
            logger.warning(
                f"Signoff attempted without required gates. "
                f"gate_status={gate_status}, gates_passed={gates_passed}. Redirecting to gate node."
            )
            # Set flag for workflow to redirect
            corrected_state["_redirect_to"] = "gate"
            corrected_state["_redirect_reason"] = f"No gates passed: gate_status={gate_status}"

            # Don't block - let workflow handle redirect
            corrected_state["_enforcement_redirect"] = True
        elif gate_passed or has_any_gates:
            # Gates have passed - allow signoff to proceed
            logger.debug(f"Signoff allowed: gate_passed={gate_passed}, gates_passed={gates_passed}")

        return corrected_state

    def get_stats(self) -> Dict[str, int]:
        """Get enforcement statistics."""
        return dict(self._enforcement_stats)


# =============================================================================
# WORKFLOW BUILDER INTEGRATION
# =============================================================================


def build_enforced_workflow(
    workflow_nodes: Any,
    checkpointer: Any = None,
) -> Any:
    """Build a LangGraph workflow with enforcement wrappers.

    UPDATED 2026-01-29: Now uses SURGICAL REWORK system for task-level rework
    instead of sprint-level rework. When tasks fail, only the failed task is
    reworked, not the entire sprint.

    Args:
        workflow_nodes: Existing WorkflowNodes instance
        checkpointer: Optional checkpointer for state persistence

    Returns:
        Compiled workflow with enforcement and surgical rework
    """
    try:
        from langgraph.graph import StateGraph, END

        from pipeline.langgraph.state import (
            PipelineState,
            SprintPhase,
            GateStatus,
            TaskLevelStatus,
        )
        from pipeline.langgraph.surgical_rework_nodes import (
            SurgicalReworkNodes,
            should_use_surgical_rework,
            route_after_task_exec,
            route_after_task_validate,
            route_after_task_rework,
            route_after_check_group,
        )
        from pipeline.surgical_rework_config import (
            get_surgical_rework_config,
            is_surgical_rework_enabled,
        )

        # Create enforced wrapper
        enforced = EnforcedWorkflowNodes(workflow_nodes)

        # Create surgical rework nodes
        config = get_surgical_rework_config()
        surgical_nodes = SurgicalReworkNodes(
            max_concurrent=config.limits.max_concurrent_tasks,
            max_repair_attempts=config.limits.max_repair_attempts_per_task,
        )

        # Build graph
        graph = StateGraph(PipelineState)

        # Add standard nodes
        # FIX 2026-01-28: Added qa node between gate and signoff for QA workers
        graph.add_node("init", enforced.init_node)
        graph.add_node("spec", enforced.spec_node)  # GAP-3: spec_kit decomposition

        # Add surgical rework nodes (REPLACES old exec + rework)
        # 2026-01-29: Task-level execution instead of sprint-level
        graph.add_node("setup_tasks", surgical_nodes.setup_tasks_node)
        graph.add_node("task_exec", surgical_nodes.task_exec_node)
        graph.add_node("task_validate", surgical_nodes.task_validate_node)
        graph.add_node("task_rework", surgical_nodes.task_rework_node)
        graph.add_node("check_group", surgical_nodes.check_group_complete_node)
        graph.add_node("integration_validate", surgical_nodes.integration_validate_node)

        # Keep legacy exec for fallback (when surgical rework not applicable)
        graph.add_node("exec", enforced.exec_node)

        # Standard nodes after exec
        graph.add_node("gate", enforced.gate_node)  # Automated gates (G0-G8)
        graph.add_node("qa", enforced.qa_node)      # QA workers (auditor, refinador, etc.)
        graph.add_node("signoff", enforced.signoff_node)
        graph.add_node("artifact", enforced.artifact_node)

        # Legacy rework node (fallback when surgical rework disabled)
        async def legacy_rework_node(state: PipelineState) -> PipelineState:
            """Legacy sprint-level rework (fallback when surgical rework disabled).

            This node:
            1. Extracts rework context from state
            2. Checks if failure type is reworkable
            3. Increments attempt counter
            4. Routes back to exec for full re-execution

            NOTE: This is the OLD behavior - re-executes entire sprint.
            Prefer surgical rework for task-level rework.
            """
            result_state = dict(state)

            # Get rework context
            rework_context = result_state.get("_rework_context", {})
            attempt = rework_context.get("attempt_number", result_state.get("attempt", 1) + 1)
            max_attempts = rework_context.get("max_attempts", 3)

            # Check if failure type is reworkable
            try:
                from pipeline.rework_strategies import (
                    should_rework,
                    get_rework_instructions,
                )

                can_rework, reason = should_rework(rework_context)
                if not can_rework:
                    logger.warning(f"Legacy rework not possible: {reason}")
                    result_state["phase"] = SprintPhase.HALT.value
                    result_state["_escalated"] = True
                    result_state["_escalation_reason"] = reason
                    result_state["_escalate_to"] = rework_context.get("escalate_to", "ops_ctrl")
                    return result_state

                result_state["_rework_instructions"] = get_rework_instructions(rework_context)
            except ImportError:
                logger.warning("rework_strategies not available")

            # Check max attempts
            if attempt > max_attempts:
                logger.error(f"Legacy rework: max attempts ({max_attempts}) exceeded")
                result_state["phase"] = SprintPhase.HALT.value
                result_state["_escalated"] = True
                result_state["_escalation_reason"] = "Max rework attempts exceeded"
                return result_state

            result_state["attempt"] = attempt
            result_state["_task_invalidated"] = False
            result_state["_requires_rework"] = True

            logger.info(f"Legacy rework attempt {attempt}/{max_attempts}")

            return result_state

        graph.add_node("legacy_rework", legacy_rework_node)

        # =================================================================
        # TARGETED FIX NODE (2026-01-30)
        # =================================================================
        # Correção cirúrgica para erros CODE/PERFORMANCE
        # Complementa o exec_node, NÃO substitui

        async def targeted_fix_node(state: PipelineState) -> PipelineState:
            """
            Aplica fix cirúrgico em erros específicos de gates.

            Pré-condições:
            - Gates falharam (state["gates_failed"] não vazio)
            - Falha é do tipo CODE ou PERFORMANCE
            - Tentativas < max_attempts

            Pós-condições:
            - Fix aplicado via daemon
            - Re-run gates que falharam
            - Atualiza state com resultado
            """
            from pathlib import Path

            result_state = dict(state)
            node = "targeted_fix"

            logger.info(f"[targeted_fix] Starting targeted fix for sprint {state.get('sprint_id')}")

            try:
                # Import targeted_fix module
                from pipeline.langgraph.targeted_fix import (
                    analyze_gate_failure,
                    build_fix_instruction,
                    apply_targeted_fix,
                    can_apply_targeted_fix,
                    get_failure_type_from_context,
                )

                # 1. Get rework context
                rework_context = result_state.get("_rework_context", {})
                failure_type = get_failure_type_from_context(rework_context)

                if failure_type and not can_apply_targeted_fix(failure_type):
                    # Not fixable by targeted fix → delegate to legacy_rework
                    logger.info(f"[targeted_fix] Failure type {failure_type} not fixable, delegating to legacy_rework")
                    result_state["_targeted_fix_skipped"] = True
                    result_state["_targeted_fix_reason"] = f"Failure type {failure_type} not fixable"
                    return result_state

                # 2. Get failed gates info
                gates_failed = result_state.get("gates_failed", [])
                if not gates_failed:
                    logger.warning("[targeted_fix] No failed gates found in state")
                    result_state["_targeted_fix_skipped"] = True
                    result_state["_targeted_fix_reason"] = "No failed gates in state"
                    return result_state

                # Use first failed gate for analysis
                gate_result = gates_failed[0] if isinstance(gates_failed[0], dict) else {"gate_id": gates_failed[0]}
                run_dir = Path(result_state.get("run_dir", "."))

                # 3. Analyze the failure
                import asyncio
                loop = asyncio.get_event_loop()
                failure_analysis = await analyze_gate_failure(gate_result, run_dir)

                logger.info(f"[targeted_fix] Analyzed failure: {failure_analysis.get('error_type')} in {failure_analysis.get('file_path')}")

                # 4. Build fix instruction
                context_pack = result_state.get("context_pack", {})
                instruction = await build_fix_instruction(failure_analysis, context_pack)

                # 5. Apply the fix via daemon
                workspace = Path(result_state.get("repo_root", "."))
                sprint_id = result_state.get("sprint_id", "unknown")
                affected_files = failure_analysis.get("affected_files", [])

                fix_result = await apply_targeted_fix(
                    instruction=instruction,
                    workspace=workspace,
                    sprint_id=sprint_id,
                    affected_files=affected_files,
                )

                # 6. Record result
                result_state["_targeted_fix_applied"] = True
                result_state["_targeted_fix_result"] = fix_result

                if fix_result.get("status") == "success":
                    # Mark for re-run of failed gates
                    result_state["_rerun_failed_gates"] = True
                    result_state["_targeted_fix_success"] = True
                    logger.info(f"[targeted_fix] Fix applied successfully, will re-run failed gates")
                else:
                    # Fix failed
                    result_state["_targeted_fix_success"] = False
                    logger.warning(f"[targeted_fix] Fix failed: {fix_result.get('summary')}")

                return result_state

            except Exception as e:
                logger.error(f"[targeted_fix] Error in targeted_fix_node: {e}")
                result_state["_targeted_fix_error"] = str(e)
                result_state["_targeted_fix_skipped"] = True
                return result_state

        if TARGETED_FIX_AVAILABLE:
            graph.add_node("targeted_fix", targeted_fix_node)
            logger.info("Targeted fix node registered in workflow graph")
        else:
            logger.warning("Targeted fix not available, node not registered")

        # =================================================================
        # ROUTING FUNCTIONS (2026-01-29: Updated for Surgical Rework)
        # (2026-01-30: Added targeted_fix routing)
        # =================================================================

        def route_after_init(state: PipelineState) -> str:
            """Route after init node."""
            if state.get("_task_invalidated"):
                return "legacy_rework"
            if state.get("_redirect_to"):
                return state["_redirect_to"]
            if state.get("phase") == SprintPhase.HALT.value:
                return END
            if state.get("status") == "failed":
                return END
            return "spec"  # GAP-3: Route to spec node for decomposition

        def route_after_spec(state: PipelineState) -> str:
            """Route after spec node - decides surgical vs legacy exec.

            2026-01-29: SURGICAL REWORK DECISION POINT
            If granular tasks exist, use surgical rework (task-level).
            Otherwise, fall back to legacy exec (sprint-level).
            """
            if state.get("_task_invalidated"):
                return "legacy_rework"
            if state.get("_redirect_to"):
                return state["_redirect_to"]
            if state.get("phase") == SprintPhase.HALT.value:
                return END
            if state.get("status") == "failed":
                return END

            # SURGICAL REWORK DECISION: Check if we should use task-level execution
            sprint_id = state.get("sprint_id", "")
            granular_tasks = state.get("granular_tasks", [])

            # Use surgical rework if:
            # 1. Feature flag is enabled for this sprint
            # 2. We have granular tasks from spec decomposition
            if is_surgical_rework_enabled(sprint_id) and len(granular_tasks) > 1:
                logger.info(
                    f"Routing to SURGICAL REWORK: {len(granular_tasks)} tasks "
                    f"will be executed and reworked at task-level"
                )
                return "setup_tasks"

            # Fall back to legacy sprint-level exec
            logger.info("Routing to legacy exec (no granular tasks or surgical rework disabled)")
            return "exec"

        def route_after_setup_tasks(state: PipelineState) -> str:
            """Route after task setup - always goes to task_exec."""
            surgical_rework = state.get("surgical_rework", {})
            if not surgical_rework.get("enabled"):
                return "exec"  # Fallback if setup failed
            return "task_exec"

        def surgical_route_after_task_exec(state: PipelineState) -> str:
            """Route after task execution node."""
            surgical_rework = state.get("surgical_rework", {})
            tasks = surgical_rework.get("tasks", {})

            validating = sum(
                1 for t in tasks.values()
                if t.get("status") == TaskLevelStatus.VALIDATING.value
            )

            if validating > 0:
                return "task_validate"

            return "check_group"

        def surgical_route_after_task_validate(state: PipelineState) -> str:
            """Route after task validation node."""
            surgical_rework = state.get("surgical_rework", {})
            tasks = surgical_rework.get("tasks", {})

            failed = sum(
                1 for t in tasks.values()
                if t.get("status") == TaskLevelStatus.FAILED.value
            )

            if failed > 0:
                logger.info(f"Routing to task_rework: {failed} failed tasks")
                return "task_rework"

            return "check_group"

        def surgical_route_after_task_rework(state: PipelineState) -> str:
            """Route after task rework node."""
            surgical_rework = state.get("surgical_rework", {})
            tasks = surgical_rework.get("tasks", {})

            validating = sum(
                1 for t in tasks.values()
                if t.get("status") == TaskLevelStatus.VALIDATING.value
            )

            if validating > 0:
                return "task_validate"

            return "check_group"

        def surgical_route_after_check_group(state: PipelineState) -> str:
            """Route after checking group completion."""
            surgical_rework = state.get("surgical_rework", {})
            tasks = surgical_rework.get("tasks", {})
            parallel_groups = surgical_rework.get("parallel_groups", [])
            current_index = surgical_rework.get("current_group_index", 0)

            # Check if all tasks escalated
            all_escalated = all(
                t.get("status") == TaskLevelStatus.ESCALATED.value
                for t in tasks.values()
            ) if tasks else False

            if all_escalated:
                logger.warning("All tasks escalated, halting pipeline")
                return END

            # Check if more groups to execute
            if current_index < len(parallel_groups):
                return "task_exec"

            # All groups complete, run integration validation
            return "integration_validate"

        def surgical_route_after_integration_validate(state: PipelineState) -> str:
            """Route after integration validation."""
            surgical_rework = state.get("surgical_rework", {})

            if surgical_rework.get("integration_validation_passed"):
                logger.info("Integration validation PASSED, proceeding to gate")
                return "gate"

            logger.warning("Integration validation FAILED")
            return END

        def route_after_exec(state: PipelineState) -> str:
            """Route after legacy exec node."""
            if state.get("_task_invalidated"):
                return "legacy_rework"
            if state.get("_redirect_to"):
                return state["_redirect_to"]
            if state.get("phase") == SprintPhase.HALT.value:
                return END
            if state.get("status") == "failed":
                return END
            if state.get("_should_retry"):
                return "exec"
            return "gate"

        def route_after_gate(state: PipelineState) -> str:
            """Route after gate node.

            2026-01-30: Added targeted_fix for CODE/PERFORMANCE failures.
            Order of precedence:
            1. If gates passed -> qa
            2. If coming back from successful targeted_fix -> qa
            3. If CODE/PERFORMANCE failure + targeted_fix available -> targeted_fix
            4. If surgical_rework enabled -> task_rework
            5. Otherwise -> legacy_rework
            """
            # Standard early exits
            if state.get("_redirect_to"):
                return state["_redirect_to"]
            if state.get("phase") == SprintPhase.HALT.value:
                return END
            if state.get("status") == "failed":
                logger.warning("route_after_gate: status=failed, going to END")
                return END

            # Handle re-run after targeted_fix
            if state.get("_rerun_failed_gates"):
                # Clear the flag and check if gates passed now
                if state.get("gate_status") == GateStatus.PASS.value:
                    logger.info("route_after_gate: Gates passed after targeted fix -> qa")
                    return "qa"
                else:
                    # Targeted fix didn't work, go to legacy_rework
                    logger.warning("route_after_gate: Gates still failing after targeted fix -> legacy_rework")
                    return "legacy_rework"

            # Gate passed -> QA workers
            if state.get("gate_status") == GateStatus.PASS.value:
                return "qa"

            # === GATES FAILED ===
            # Get failure context
            rework_context = state.get("_rework_context", {})
            failure_type = rework_context.get("failure_type", "code")
            attempt = state.get("attempt", 1)
            max_attempts = state.get("max_attempts", 3)

            # Check if max attempts exceeded
            if attempt >= max_attempts:
                logger.error(f"route_after_gate: Max attempts ({max_attempts}) exceeded -> END")
                return END

            # TARGETED FIX: Try targeted_fix for CODE/PERFORMANCE failures
            # Only if targeted_fix is available and hasn't been tried yet
            if TARGETED_FIX_AVAILABLE and should_use_targeted_fix is not None:
                if not state.get("_targeted_fix_applied"):
                    if should_use_targeted_fix(rework_context, attempt, max_attempts):
                        logger.info(
                            f"route_after_gate: Failure type '{failure_type}' is fixable, "
                            f"routing to targeted_fix (attempt {attempt}/{max_attempts})"
                        )
                        return "targeted_fix"
                else:
                    # Already tried targeted_fix and it failed
                    logger.info("route_after_gate: Targeted fix already attempted, going to legacy_rework")

            # SURGICAL REWORK: Check if surgical rework is active (deprecated, now disabled by default)
            surgical_rework = state.get("surgical_rework", {})
            if surgical_rework.get("enabled"):
                logger.info("route_after_gate: Surgical rework enabled, routing to task_rework")
                return "task_rework"

            # LEGACY REWORK: Default fallback
            if state.get("_task_invalidated"):
                return "legacy_rework"

            # Set invalidation flag and go to legacy rework
            logger.warning(f"route_after_gate: Gate failed (attempt {attempt}/{max_attempts}), triggering legacy_rework")
            return "legacy_rework"

        def route_after_targeted_fix(state: PipelineState) -> str:
            """Route after targeted_fix node.

            2026-01-30: New routing function for targeted_fix.
            - If fix was successful -> re-run gates
            - If fix failed or was skipped -> legacy_rework
            """
            if state.get("_targeted_fix_success"):
                logger.info("route_after_targeted_fix: Fix successful, re-running gates")
                return "gate"

            if state.get("_targeted_fix_skipped"):
                logger.info(f"route_after_targeted_fix: Fix skipped ({state.get('_targeted_fix_reason')}), going to legacy_rework")
                return "legacy_rework"

            if state.get("_targeted_fix_error"):
                logger.warning(f"route_after_targeted_fix: Fix error ({state.get('_targeted_fix_error')}), going to legacy_rework")
                return "legacy_rework"

            # Default: check if rerun is requested
            if state.get("_rerun_failed_gates"):
                return "gate"

            # Fallback to legacy
            logger.warning("route_after_targeted_fix: Unknown state, defaulting to legacy_rework")
            return "legacy_rework"

        def route_after_qa(state: PipelineState) -> str:
            """Route after QA node.

            2026-01-29: When QA fails with surgical rework, route to task_rework.
            """
            if state.get("_task_invalidated"):
                surgical_rework = state.get("surgical_rework", {})
                if surgical_rework.get("enabled"):
                    return "task_rework"
                return "legacy_rework"

            if state.get("_redirect_to"):
                return state["_redirect_to"]
            if state.get("phase") == SprintPhase.HALT.value:
                return END
            if state.get("status") == "failed":
                logger.warning("route_after_qa: status=failed, going to END")
                return END

            if not state.get("qa_workers_passed", True):
                surgical_rework = state.get("surgical_rework", {})
                if surgical_rework.get("enabled"):
                    logger.warning("QA failed with surgical rework active")
                    return "task_rework"
                logger.warning("QA workers found issues, triggering legacy rework")
                return "legacy_rework"

            return "signoff"

        def route_after_signoff(state: PipelineState) -> str:
            """Route after signoff node."""
            if state.get("_task_invalidated"):
                surgical_rework = state.get("surgical_rework", {})
                if surgical_rework.get("enabled"):
                    return "task_rework"
                return "legacy_rework"
            if state.get("_enforcement_redirect"):
                return state.get("_redirect_to", "gate")
            if state.get("phase") == SprintPhase.HALT.value:
                return END
            if state.get("status") == "failed":
                logger.warning("route_after_signoff: status=failed, going to END")
                return END
            return "artifact"

        def route_after_artifact(state: PipelineState) -> str:
            """Route after artifact node."""
            if state.get("_task_invalidated"):
                surgical_rework = state.get("surgical_rework", {})
                if surgical_rework.get("enabled"):
                    return "task_rework"
                return "legacy_rework"
            return END

        def route_after_legacy_rework(state: PipelineState) -> str:
            """Route after legacy rework node (sprint-level re-execution)."""
            if state.get("phase") == SprintPhase.HALT.value:
                return END
            if state.get("status") == "failed":
                return END

            # Route back to the node that failed
            rework_context = state.get("_rework_context", {})
            violation_report = rework_context.get("violation_report", {})
            original_node = violation_report.get("node_name", "exec")

            node_map = {
                "init": "init",
                "spec": "spec",
                "exec": "exec",
                "gate": "gate",
                "qa": "qa",
                "signoff": "signoff",
                "artifact": "artifact",
            }

            target = node_map.get(original_node, "exec")
            logger.info(f"Legacy rework routing back to '{target}'")
            return target

        # =================================================================
        # WIRE UP THE GRAPH
        # =================================================================

        # Set entry point
        graph.set_entry_point("init")

        # Standard routing
        graph.add_conditional_edges("init", route_after_init)
        graph.add_conditional_edges("spec", route_after_spec)

        # Surgical rework routing (task-level)
        graph.add_conditional_edges("setup_tasks", route_after_setup_tasks)
        graph.add_conditional_edges("task_exec", surgical_route_after_task_exec)
        graph.add_conditional_edges("task_validate", surgical_route_after_task_validate)
        graph.add_conditional_edges("task_rework", surgical_route_after_task_rework)
        graph.add_conditional_edges("check_group", surgical_route_after_check_group)
        graph.add_conditional_edges("integration_validate", surgical_route_after_integration_validate)

        # Legacy routing (sprint-level)
        graph.add_conditional_edges("exec", route_after_exec)
        graph.add_conditional_edges("gate", route_after_gate)
        graph.add_conditional_edges("qa", route_after_qa)
        graph.add_conditional_edges("signoff", route_after_signoff)
        graph.add_conditional_edges("artifact", route_after_artifact)
        graph.add_conditional_edges("legacy_rework", route_after_legacy_rework)

        # Targeted fix routing (2026-01-30)
        if TARGETED_FIX_AVAILABLE:
            graph.add_conditional_edges("targeted_fix", route_after_targeted_fix)
            logger.info("Targeted fix routing registered")

        # Compile
        if checkpointer:
            return graph.compile(checkpointer=checkpointer)
        return graph.compile()

    except ImportError as e:
        logger.error(f"Failed to build enforced workflow: {e}")
        raise


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Decorator
    "with_enforcement",
    # Classes
    "EnforcedWorkflowNodes",
    # Functions
    "build_enforced_workflow",
    # Violation handler (re-exported)
    "get_violation_handler",
    "create_security_violation",
    "create_invariant_violation",
    "create_validation_violation",
    "create_gate_violation",
    # Types (re-exported)
    "ViolationType",
    "ViolationSeverity",
    "ViolationDetails",
    "TaskInvalidation",
    "ReworkTask",
]
