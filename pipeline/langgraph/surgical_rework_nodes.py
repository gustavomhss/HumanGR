"""
Surgical Rework Nodes for LangGraph Workflow.

This module provides task-level execution and rework nodes that enable
fine-grained surgical rework when individual tasks fail, rather than
restarting the entire sprint.

Key Nodes:
- task_exec_node: Executes individual tasks via CrewAI (canonical)
- task_validate_node: Validates tasks with gates (gate_runner)
- task_rework_node: Performs surgical rework on failed tasks
- integration_validate_node: Validates task integration after all pass

CANONICAL INTEGRATION (2026-01-31):
- Uses CrewAI for ALL execution (create_exec_crew)
- Uses pipeline.task_context for TaskContext/TaskResult
- Uses pipeline.gate_runner for validation
- Uses pipeline.instruction_builder for rework instructions
- NO direct daemon calls - CrewAI is the orchestrator

Usage:
    from pipeline.langgraph.surgical_rework_nodes import (
        create_surgical_rework_workflow,
        SurgicalReworkNodes,
    )

Created: 2026-01-29
Updated: 2026-01-31 (Canonical CrewAI integration)
Reference: .claude/plans/SURGICAL_REWORK_ARCHITECTURE_STUDY.md
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from langgraph.graph import StateGraph, END

from .state import (
    PipelineState,
    SprintPhase,
    PipelineStatus,
    TaskLevelStatus,
    ReworkPhase,
    FailureCategory,
    SurgicalReworkState,
    DiagnosisEntry,
    PatchEntry,
    create_task_level_entry,
    update_task_level_status,
    add_diagnosis_to_task,
    add_patch_to_task,
    set_parallel_groups,
    advance_to_next_group,
    get_current_group_tasks,
    record_repair_attempt,
    set_integration_validation_result,
)

# CANONICAL IMPORTS (2026-01-31)
# TaskContext and TaskResult from pipeline (source of truth)
from pipeline.task_context import TaskContext, TaskResult

# CrewAI is the official orchestrator for ALL phases
try:
    from pipeline.crewai_hierarchy import create_exec_crew
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False

# Gate runner for validation
try:
    from pipeline.gate_runner import (
        run_gate_selection_parallel,
        ensure_gate_selection_exists,
    )
    GATE_RUNNER_AVAILABLE = True
except ImportError:
    GATE_RUNNER_AVAILABLE = False

# Instruction builder for rework
try:
    from pipeline.instruction_builder import build_rework_instruction
    INSTRUCTION_BUILDER_AVAILABLE = True
except ImportError:
    INSTRUCTION_BUILDER_AVAILABLE = False

# QA schemas for failure classification
try:
    from pipeline.qa_schemas import (
        classify_gate_failure,
        get_failure_remediation,
        GateFailureType,
    )
    QA_SCHEMAS_AVAILABLE = True
except ImportError:
    QA_SCHEMAS_AVAILABLE = False

if TYPE_CHECKING:
    from ..task_models import Task
    from ..task_graph import TaskDependencyGraph

logger = logging.getLogger(__name__)


# =============================================================================
# SURGICAL REWORK NODES
# =============================================================================


class SurgicalReworkNodes:
    """
    Container for surgical rework nodes.

    This class provides all the nodes needed for task-level execution
    and surgical rework within the LangGraph workflow.

    Nodes:
        - task_exec_node: Execute current group of tasks in parallel
        - task_validate_node: Validate completed tasks
        - task_rework_node: Perform surgical rework on failed tasks
        - integration_validate_node: Validate task integration

    Workflow:
        spec -> task_exec -> task_validate
                    ^              |
                    |              v
                    +-- task_rework (if task failed)
                              |
                              v
                    integration_validate (when all tasks pass)
    """

    def __init__(
        self,
        task_executor: Optional["TaskExecutor"] = None,
        task_validator: Optional["TaskValidator"] = None,
        rework_engine: Optional["SurgicalReworkEngine"] = None,
        integration_validator: Optional["IntegrationValidator"] = None,
        max_concurrent: int = 4,
        max_repair_attempts: int = 3,
    ):
        """
        Initialize surgical rework nodes.

        Args:
            task_executor: Executor for individual tasks.
            task_validator: Validator for task outputs.
            rework_engine: Engine for surgical rework.
            integration_validator: Validator for task integration.
            max_concurrent: Maximum concurrent tasks.
            max_repair_attempts: Maximum repair attempts per task.
        """
        self.task_executor = task_executor
        self.task_validator = task_validator
        self.rework_engine = rework_engine
        self.integration_validator = integration_validator
        self.max_concurrent = max_concurrent
        self.max_repair_attempts = max_repair_attempts
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def setup_tasks_node(self, state: PipelineState) -> PipelineState:
        """
        Setup task-level entries from granular tasks.

        This node converts granular_tasks from spec decomposition
        into TaskLevelEntry objects and computes parallel groups.

        Args:
            state: Current pipeline state.

        Returns:
            Updated state with surgical_rework initialized.
        """
        logger.info("Setting up task-level entries for surgical rework")

        result_state = dict(state)
        granular_tasks = result_state.get("granular_tasks", [])
        sprint_id = result_state.get("sprint_id", "")

        if not granular_tasks:
            logger.warning("No granular tasks found, surgical rework disabled")
            return result_state

        # Create task entries
        tasks: Dict[str, Any] = {}
        for i, task_data in enumerate(granular_tasks, start=1):
            task_id = f"{sprint_id}-T{i:02d}"
            entry = create_task_level_entry(
                task_id=task_id,
                sprint_id=sprint_id,
                sequence=i,
                name=task_data.get("deliverable", f"Task {i}"),
                description=task_data.get("task_prompt", ""),
                deliverables=[task_data.get("deliverable", "")],
                depends_on=[],  # Can be enhanced to parse dependencies
                max_attempts=self.max_repair_attempts,
            )
            tasks[task_id] = entry

        # Compute parallel groups using task graph
        try:
            from ..task_graph import TaskDependencyGraph
            from ..task_models import Task

            graph = TaskDependencyGraph()
            for task_id, entry in tasks.items():
                task = Task(
                    id=task_id,
                    sprint_id=entry["sprint_id"],
                    sequence=entry["sequence"],
                    name=entry["name"],
                    description=entry["description"],
                    deliverables=entry["deliverables"],
                    depends_on=entry["depends_on"],
                )
                graph.add_task(task)

            parallel_groups = graph.get_parallel_groups()
        except Exception as e:
            logger.warning(f"Failed to compute parallel groups: {e}")
            # Fallback: sequential execution
            parallel_groups = [[tid] for tid in tasks.keys()]

        # Update surgical rework state
        surgical_rework = SurgicalReworkState(
            enabled=True,
            tasks=tasks,
            parallel_groups=parallel_groups,
            current_group_index=0,
            tasks_pending=len(tasks),
            tasks_executing=0,
            tasks_passed=0,
            tasks_failed=0,
            tasks_escalated=0,
            total_repairs=0,
            total_patches_applied=0,
            integration_validation_passed=False,
            integration_issues=[],
            last_task_completed=None,
            last_task_failed=None,
            started_at=datetime.now(timezone.utc).isoformat(),
            completed_at=None,
        )

        result_state["surgical_rework"] = surgical_rework
        result_state["updated_at"] = datetime.now(timezone.utc).isoformat()

        logger.info(
            f"Setup complete: {len(tasks)} tasks in {len(parallel_groups)} groups"
        )

        return result_state

    async def task_exec_node(self, state: PipelineState) -> PipelineState:
        """
        Execute current group of tasks in parallel.

        This node:
        1. Gets the current parallel group
        2. Executes all tasks in the group concurrently
        3. Updates task statuses
        4. Advances to next group if all passed

        Args:
            state: Current pipeline state.

        Returns:
            Updated state with task execution results.
        """
        logger.info("Executing task group")

        result_state = dict(state)
        surgical_rework = result_state.get("surgical_rework", {})

        if not surgical_rework.get("enabled"):
            logger.info("Surgical rework not enabled, skipping task_exec_node")
            return result_state

        # Get current group tasks
        current_group = get_current_group_tasks(result_state)

        if not current_group:
            logger.info("No more groups to execute")
            return result_state

        logger.info(f"Executing group with {len(current_group)} tasks")

        # Execute tasks in parallel
        tasks = surgical_rework.get("tasks", {})

        async def execute_single_task(task_id: str) -> Dict[str, Any]:
            """Execute a single task with semaphore limiting."""
            async with self._semaphore:
                task_entry = tasks.get(task_id, {})

                # Update status to EXECUTING
                result_state_update = update_task_level_status(
                    result_state,
                    task_id,
                    TaskLevelStatus.EXECUTING,
                )

                try:
                    # CANONICAL: Use CrewAI for ALL execution (2026-01-31)
                    # CrewAI is the official orchestrator per CLAUDE.md
                    if not CREWAI_AVAILABLE:
                        raise ImportError("CrewAI not available - cannot execute tasks")

                    # Get paths from state
                    repo_root = Path(result_state.get("repo_root", "."))
                    run_dir = Path(result_state.get("run_dir", "."))
                    sprint_id = result_state.get("sprint_id", "")
                    context_pack = result_state.get("context_pack", {})

                    # Build task context (canonical structure)
                    task_ctx = TaskContext(
                        sprint_id=sprint_id,
                        task_id=task_id,
                        name=task_entry.get("name", f"Task {task_id}"),
                        description=task_entry.get("description", ""),
                        deliverables=task_entry.get("deliverables", []),
                        depends_on=task_entry.get("depends_on", []),
                        context_pack=context_pack,
                        workspace_path=repo_root,
                        run_dir=run_dir,
                    )

                    # Build granular task for CrewAI
                    granular_task = {
                        "task_id": task_id,
                        "deliverable": task_entry.get("deliverables", [""])[0],
                        "task_prompt": task_entry.get("description", ""),
                        "name": task_entry.get("name", f"Task {task_id}"),
                    }

                    # Execute via CrewAI (canonical orchestrator)
                    logger.info(f"Executing task {task_id} via CrewAI")
                    crew_result = await asyncio.to_thread(
                        create_exec_crew,
                        crew_id=f"exec_{sprint_id}_{task_id}",
                        implementations=task_ctx.deliverables,
                        use_got=True,  # Use Graph of Thoughts for better planning
                        context_pack=context_pack,
                        granular_tasks=[granular_task],
                        stack_ctx=None,  # P0-3: TODO - pass stack_ctx from rework context
                    )

                    # Map CrewAI result to TaskResult format
                    # P0-1 FIX: CrewResult has .status not .success
                    success = crew_result.status == "success" if hasattr(crew_result, 'status') else False
                    code_generated = ""
                    evidence_paths = []

                    if hasattr(crew_result, 'raw'):
                        code_generated = crew_result.raw or ""
                    if hasattr(crew_result, 'evidence_paths'):
                        evidence_paths = crew_result.evidence_paths or []

                    return {
                        "task_id": task_id,
                        "code": code_generated,
                        "evidence_paths": evidence_paths,
                        "success": success,
                        "error": None if success else "CrewAI execution failed",
                    }
                except Exception as e:
                    logger.error(f"Task {task_id} execution failed: {e}")
                    return {
                        "task_id": task_id,
                        "error": str(e),
                        "success": False,
                    }

        # Execute all tasks in current group
        results = await asyncio.gather(
            *[execute_single_task(tid) for tid in current_group],
            return_exceptions=True,
        )

        # Update state with results
        now = datetime.now(timezone.utc).isoformat()
        tasks = dict(surgical_rework.get("tasks", {}))

        for result in results:
            if isinstance(result, Exception):
                continue

            task_id = result.get("task_id")
            if not task_id or task_id not in tasks:
                continue

            task = dict(tasks[task_id])

            if result.get("success"):
                task["code_generated"] = result.get("code")
                task["status"] = TaskLevelStatus.VALIDATING.value
            else:
                task["status"] = TaskLevelStatus.FAILED.value

            tasks[task_id] = task

        # Update surgical rework state
        executing = sum(
            1 for t in tasks.values()
            if t["status"] == TaskLevelStatus.EXECUTING.value
        )
        validating = sum(
            1 for t in tasks.values()
            if t["status"] == TaskLevelStatus.VALIDATING.value
        )

        updated_surgical_rework = dict(surgical_rework)
        updated_surgical_rework["tasks"] = tasks
        updated_surgical_rework["tasks_executing"] = executing

        result_state["surgical_rework"] = updated_surgical_rework
        result_state["updated_at"] = now
        result_state["phase"] = SprintPhase.EXEC.value

        logger.info(f"Task execution complete: {validating} validating")

        return result_state

    async def task_validate_node(self, state: PipelineState) -> PipelineState:
        """
        Validate tasks that have completed execution.

        This node:
        1. Finds all tasks in VALIDATING status
        2. Runs validation (gates + QA workers)
        3. Updates task statuses to PASSED or FAILED

        Args:
            state: Current pipeline state.

        Returns:
            Updated state with validation results.
        """
        logger.info("Validating tasks")

        result_state = dict(state)
        surgical_rework = result_state.get("surgical_rework", {})

        if not surgical_rework.get("enabled"):
            return result_state

        tasks = surgical_rework.get("tasks", {})
        validating_tasks = [
            tid for tid, t in tasks.items()
            if t.get("status") == TaskLevelStatus.VALIDATING.value
        ]

        if not validating_tasks:
            logger.info("No tasks to validate")
            return result_state

        logger.info(f"Validating {len(validating_tasks)} tasks")

        # Validate each task
        tasks = dict(tasks)
        now = datetime.now(timezone.utc).isoformat()

        for task_id in validating_tasks:
            task = dict(tasks[task_id])

            try:
                # CANONICAL: Use gate_runner directly for validation (2026-01-31)
                # gate_runner is the official gate execution layer
                if not GATE_RUNNER_AVAILABLE:
                    raise ImportError("gate_runner not available - cannot validate tasks")

                run_dir = Path(result_state.get("run_dir", "."))
                repo_root = Path(result_state.get("repo_root", "."))
                sprint_id = result_state.get("sprint_id", "")

                # Ensure gate_selection.yml exists (canonical fix)
                ensure_gate_selection_exists(run_dir, sprint_id, track="B", profile="core")

                # Get code path from task
                deliverables = task.get("deliverables", [])
                code_path = deliverables[0] if deliverables else ""

                logger.info(f"Validating task {task_id} with gate_runner")

                # Validate via gate_runner (canonical)
                validation = await run_gate_selection_parallel(
                    repo_root=repo_root,
                    run_dir=run_dir,
                    sprint_id=sprint_id,
                    track="B",  # Use balanced track
                    max_concurrent=3,
                )

                passed = validation.get("overall_status") == "PASS"
                failed_gates = validation.get("failed_gates", [])

                # Store validation result for potential rework
                task["_last_validation"] = {
                    "passed": passed,
                    "confidence": 1.0 if passed else 0.0,
                    "gate_results": validation.get("gate_results", []),
                    "failed_gates": failed_gates,
                }

                if passed:
                    task["status"] = TaskLevelStatus.PASSED.value
                    task["confidence_score"] = 1.0
                    task["completed_at"] = now
                else:
                    task["status"] = TaskLevelStatus.FAILED.value
                    task["last_validation_at"] = now
                    logger.warning(f"Task {task_id} failed {len(failed_gates)} gates")

            except Exception as e:
                logger.error(f"Validation failed for {task_id}: {e}")
                task["status"] = TaskLevelStatus.FAILED.value
                task["last_validation_at"] = now
                task["_last_validation"] = {
                    "passed": False,
                    "confidence": 0.0,
                    "error": str(e),
                    "failed_gates": [],
                    "error_type": "infrastructure",
                }

            tasks[task_id] = task

        # Update counters
        passed = sum(
            1 for t in tasks.values()
            if t["status"] == TaskLevelStatus.PASSED.value
        )
        failed = sum(
            1 for t in tasks.values()
            if t["status"] == TaskLevelStatus.FAILED.value
        )

        updated_surgical_rework = dict(surgical_rework)
        updated_surgical_rework["tasks"] = tasks
        updated_surgical_rework["tasks_passed"] = passed
        updated_surgical_rework["tasks_failed"] = failed
        updated_surgical_rework["last_task_completed"] = (
            validating_tasks[-1] if passed > 0 else None
        )

        result_state["surgical_rework"] = updated_surgical_rework
        result_state["updated_at"] = now

        logger.info(f"Validation complete: {passed} passed, {failed} failed")

        return result_state

    async def task_rework_node(self, state: PipelineState) -> PipelineState:
        """
        Perform surgical rework on failed tasks.

        This node:
        1. Finds all tasks in FAILED status
        2. Generates diagnosis for each failure
        3. Creates and applies surgical patches
        4. Transitions tasks back to VALIDATING

        Args:
            state: Current pipeline state.

        Returns:
            Updated state with rework results.
        """
        logger.info("Performing surgical rework")

        result_state = dict(state)
        surgical_rework = result_state.get("surgical_rework", {})

        if not surgical_rework.get("enabled"):
            return result_state

        tasks = surgical_rework.get("tasks", {})
        failed_tasks = [
            tid for tid, t in tasks.items()
            if t.get("status") == TaskLevelStatus.FAILED.value
        ]

        if not failed_tasks:
            logger.info("No failed tasks to rework")
            return result_state

        logger.info(f"Reworking {len(failed_tasks)} failed tasks")

        tasks = dict(tasks)
        now = datetime.now(timezone.utc).isoformat()
        total_repairs = surgical_rework.get("total_repairs", 0)
        total_patches = surgical_rework.get("total_patches_applied", 0)

        for task_id in failed_tasks:
            task = dict(tasks[task_id])

            # Check max attempts
            attempts = task.get("attempts", 0)
            max_attempts = task.get("max_attempts", self.max_repair_attempts)

            if attempts >= max_attempts:
                logger.warning(f"Task {task_id} exceeded max attempts, escalating")
                task["status"] = TaskLevelStatus.ESCALATED.value
                task["completed_at"] = now
                tasks[task_id] = task
                continue

            # Set rework phase
            task["status"] = TaskLevelStatus.REPAIRING.value
            task["rework_phase"] = ReworkPhase.DIAGNOSING.value

            try:
                # CANONICAL: Use CrewAI + instruction_builder for rework (2026-01-31)
                # build_rework_instruction creates structured prompt for surgical fix
                # create_exec_crew executes the fix via CrewAI orchestration
                if not CREWAI_AVAILABLE:
                    raise ImportError("CrewAI not available - cannot rework tasks")
                if not INSTRUCTION_BUILDER_AVAILABLE:
                    raise ImportError("instruction_builder not available - cannot build rework instructions")

                repo_root = Path(result_state.get("repo_root", "."))
                run_dir = Path(result_state.get("run_dir", "."))
                sprint_id = result_state.get("sprint_id", "")
                context_pack = result_state.get("context_pack", {})

                # Get last validation result
                last_validation = task.get("_last_validation", {})
                failed_gates = last_validation.get("failed_gates", [])

                if not failed_gates:
                    logger.warning(f"No failed gates for task {task_id} - cannot rework")
                    task["status"] = TaskLevelStatus.ESCALATED.value
                    task["completed_at"] = now
                    tasks[task_id] = task
                    continue

                # Build TaskContext for rework instruction
                task_ctx = TaskContext(
                    sprint_id=sprint_id,
                    task_id=task_id,
                    name=task.get("name", f"Task {task_id}"),
                    description=task.get("description", ""),
                    deliverables=task.get("deliverables", []),
                    depends_on=task.get("depends_on", []),
                    context_pack=context_pack,
                    workspace_path=repo_root,
                    run_dir=run_dir,
                )

                # Classify failure type (if qa_schemas available)
                failure_type = "code"  # Default
                remediation = {"strategy": "Fix the identified issues", "specific_fixes": []}
                if QA_SCHEMAS_AVAILABLE and failed_gates:
                    try:
                        failure_type = classify_gate_failure(failed_gates[0]).value
                        remediation = get_failure_remediation(GateFailureType(failure_type))
                    except Exception as classify_err:
                        logger.warning(f"Could not classify failure: {classify_err}")

                # P2-5.4 FIX (2026-02-01): Extract reflexion suggestions from rework context
                rework_context = result_state.get("_rework_context", {})
                reflexion_suggestions = rework_context.get("reflexion_suggestions", [])

                # Build rework instruction (canonical)
                rework_instruction = build_rework_instruction(
                    context=task_ctx,
                    failed_gates=failed_gates,
                    failure_type=failure_type,
                    remediation=remediation,
                    reflexion_suggestions=reflexion_suggestions,  # P2-5.4: Pass reflexion insights
                )

                logger.info(f"Executing rework for task {task_id} via CrewAI")

                # Build granular task for rework
                granular_task = {
                    "task_id": f"{task_id}_rework",
                    "deliverable": task.get("deliverables", [""])[0],
                    "task_prompt": rework_instruction,
                    "name": f"Rework: {task.get('name', task_id)}",
                }

                # Execute rework via CrewAI (canonical orchestrator)
                crew_result = await asyncio.to_thread(
                    create_exec_crew,
                    crew_id=f"rework_{sprint_id}_{task_id}_attempt{attempts+1}",
                    implementations=task_ctx.deliverables,
                    use_got=True,  # Use GoT for better analysis
                    context_pack=context_pack,
                    granular_tasks=[granular_task],
                    stack_ctx=None,  # P0-3: TODO - pass stack_ctx from rework context
                )

                # Evaluate CrewAI result
                # P0-1 FIX: CrewResult has .status not .success
                success = crew_result.status == "success" if hasattr(crew_result, 'status') else False

                if success:
                    code_generated = crew_result.raw if hasattr(crew_result, 'raw') else ""
                    task["code_generated"] = code_generated
                    task["status"] = TaskLevelStatus.VALIDATING.value
                    task["attempts"] = attempts + 1
                    total_repairs += 1
                    total_patches += 1
                    logger.info(f"Rework for {task_id} succeeded, will re-validate")
                else:
                    # Rework failed - escalate if max attempts
                    if attempts + 1 >= max_attempts:
                        task["status"] = TaskLevelStatus.ESCALATED.value
                        task["completed_at"] = now
                    else:
                        task["status"] = TaskLevelStatus.FAILED.value
                    task["attempts"] = attempts + 1
                    logger.warning(f"Rework for {task_id} failed")

                # Record repair attempt
                repair_history = list(task.get("repair_history", []))
                repair_history.append({
                    "attempt": attempts + 1,
                    "timestamp": now,
                    "success": success,
                    "failure_type": failure_type,
                })
                task["repair_history"] = repair_history

            except Exception as e:
                logger.error(f"Rework failed for {task_id}: {e}")
                task["status"] = TaskLevelStatus.ESCALATED.value
                task["completed_at"] = now

            task["rework_phase"] = None
            tasks[task_id] = task

        # Update surgical rework state
        escalated = sum(
            1 for t in tasks.values()
            if t["status"] == TaskLevelStatus.ESCALATED.value
        )

        updated_surgical_rework = dict(surgical_rework)
        updated_surgical_rework["tasks"] = tasks
        updated_surgical_rework["tasks_escalated"] = escalated
        updated_surgical_rework["total_repairs"] = total_repairs
        updated_surgical_rework["total_patches_applied"] = total_patches

        result_state["surgical_rework"] = updated_surgical_rework
        result_state["updated_at"] = now

        logger.info(f"Rework complete: {total_repairs} repairs, {escalated} escalated")

        return result_state

    async def integration_validate_node(
        self, state: PipelineState
    ) -> PipelineState:
        """
        Validate integration between all passed tasks.

        This node:
        1. Collects all passed tasks
        2. Runs integration validation (imports, types, contracts)
        3. Updates integration validation status

        Args:
            state: Current pipeline state.

        Returns:
            Updated state with integration validation result.
        """
        logger.info("Running integration validation")

        result_state = dict(state)
        surgical_rework = result_state.get("surgical_rework", {})

        if not surgical_rework.get("enabled"):
            return result_state

        tasks = surgical_rework.get("tasks", {})
        passed_tasks = {
            tid: t for tid, t in tasks.items()
            if t.get("status") == TaskLevelStatus.PASSED.value
        }

        if not passed_tasks:
            logger.warning("No passed tasks for integration validation")
            result_state = set_integration_validation_result(
                result_state, passed=False, issues=[]
            )
            return result_state

        logger.info(f"Validating integration of {len(passed_tasks)} tasks")

        try:
            if self.integration_validator:
                # Use actual validator
                from ..parallel_runner import TaskRunResult

                task_results = {
                    tid: TaskRunResult(
                        task_id=tid,
                        status=t["status"],
                        code=t.get("code_generated"),
                    )
                    for tid, t in passed_tasks.items()
                }
                result = await self.integration_validator.validate(task_results)
                passed = result.passed
                issues = [
                    {
                        "task_a": i.task_a,
                        "task_b": i.task_b,
                        "type": i.issue_type,
                        "description": i.description,
                    }
                    for i in result.issues
                ]
            else:
                # Mock integration validation
                passed = True
                issues = []

            result_state = set_integration_validation_result(
                result_state, passed=passed, issues=issues
            )

            if passed:
                logger.info("Integration validation PASSED")
                # Mark surgical rework as complete
                now = datetime.now(timezone.utc).isoformat()
                updated_surgical_rework = dict(result_state["surgical_rework"])
                updated_surgical_rework["completed_at"] = now
                result_state["surgical_rework"] = updated_surgical_rework
            else:
                logger.warning(f"Integration validation FAILED: {len(issues)} issues")

        except Exception as e:
            logger.error(f"Integration validation error: {e}")
            result_state = set_integration_validation_result(
                result_state,
                passed=False,
                issues=[{"type": "error", "description": str(e)}],
            )

        return result_state

    async def check_group_complete_node(
        self, state: PipelineState
    ) -> PipelineState:
        """
        Check if current group is complete and advance if so.

        This node:
        1. Checks if all tasks in current group are in terminal state
        2. If complete, advances to next group
        3. If all groups complete, triggers integration validation

        Args:
            state: Current pipeline state.

        Returns:
            Updated state with group advancement.
        """
        result_state = dict(state)
        surgical_rework = result_state.get("surgical_rework", {})

        if not surgical_rework.get("enabled"):
            return result_state

        current_group = get_current_group_tasks(result_state)

        if not current_group:
            # All groups complete
            return result_state

        tasks = surgical_rework.get("tasks", {})

        # Check if all tasks in current group are terminal
        all_complete = all(
            TaskLevelStatus(tasks.get(tid, {}).get("status", "pending")).is_terminal()
            for tid in current_group
        )

        if all_complete:
            logger.info("Current group complete, advancing to next")
            result_state = advance_to_next_group(result_state)

        return result_state


# =============================================================================
# ROUTING FUNCTIONS
# =============================================================================


def route_after_task_exec(state: PipelineState) -> str:
    """Route after task execution node.

    Returns:
        - "task_validate": If there are tasks to validate
        - "check_group": If no tasks need validation
    """
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
    """Route after task validation node.

    Returns:
        - "task_rework": If there are failed tasks to rework
        - "check_group": If all tasks passed or escalated
    """
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
    """Route after task rework node.

    Returns:
        - "task_validate": If there are reworked tasks to validate
        - "check_group": If all failed tasks escalated
    """
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
    """Route after checking group completion.

    Returns:
        - "task_exec": If there are more groups to execute
        - "integration_validate": If all groups complete
        - END: If all tasks escalated
    """
    surgical_rework = state.get("surgical_rework", {})
    tasks = surgical_rework.get("tasks", {})
    parallel_groups = surgical_rework.get("parallel_groups", [])
    current_index = surgical_rework.get("current_group_index", 0)

    # Check if all tasks escalated
    all_escalated = all(
        t.get("status") == TaskLevelStatus.ESCALATED.value
        for t in tasks.values()
    )
    if all_escalated:
        return END

    # Check if more groups to execute
    if current_index < len(parallel_groups):
        return "task_exec"

    # All groups complete, run integration validation
    return "integration_validate"


def route_after_integration_validate(state: PipelineState) -> str:
    """Route after integration validation.

    Returns:
        - "gate": If integration passed, proceed to gates
        - END: If integration failed
    """
    surgical_rework = state.get("surgical_rework", {})

    if surgical_rework.get("integration_validation_passed"):
        return "gate"

    return END


# =============================================================================
# WORKFLOW BUILDER
# =============================================================================


def create_surgical_rework_subgraph(
    nodes: SurgicalReworkNodes,
) -> StateGraph:
    """
    Create a subgraph for surgical rework.

    This subgraph handles task-level execution and rework,
    and can be composed into the main workflow.

    Args:
        nodes: SurgicalReworkNodes instance.

    Returns:
        StateGraph for surgical rework.
    """
    from langgraph.graph import StateGraph

    workflow = StateGraph(PipelineState)

    # Add nodes
    workflow.add_node("setup_tasks", nodes.setup_tasks_node)
    workflow.add_node("task_exec", nodes.task_exec_node)
    workflow.add_node("task_validate", nodes.task_validate_node)
    workflow.add_node("task_rework", nodes.task_rework_node)
    workflow.add_node("check_group", nodes.check_group_complete_node)
    workflow.add_node("integration_validate", nodes.integration_validate_node)

    # Set entry point
    workflow.set_entry_point("setup_tasks")

    # Add edges
    workflow.add_edge("setup_tasks", "task_exec")

    # Conditional routing
    workflow.add_conditional_edges("task_exec", route_after_task_exec)
    workflow.add_conditional_edges("task_validate", route_after_task_validate)
    workflow.add_conditional_edges("task_rework", route_after_task_rework)
    workflow.add_conditional_edges("check_group", route_after_check_group)
    workflow.add_conditional_edges(
        "integration_validate", route_after_integration_validate
    )

    return workflow


def should_use_surgical_rework(state: PipelineState) -> bool:
    """
    Determine if surgical rework should be used for this run.

    Args:
        state: Current pipeline state.

    Returns:
        True if surgical rework is enabled and applicable.
    """
    # Check policy setting
    policy = state.get("policy", {})
    if not policy.get("rework_system_enabled", True):
        return False

    # Check if surgical rework is explicitly enabled
    surgical_rework = state.get("surgical_rework", {})
    if surgical_rework.get("enabled"):
        return True

    # Check if we have granular tasks (from spec decomposition)
    granular_tasks = state.get("granular_tasks", [])
    if len(granular_tasks) > 1:
        return True

    return False
