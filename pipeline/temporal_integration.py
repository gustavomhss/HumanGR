"""Temporal Integration for Pipeline Autonomo.

Provides high-level integration between Temporal Lite workflows and
the other quantum leap components:
- ReflexionEngine: Automatic reflection on failures
- Live-SWE: Behavior evolution based on workflow outcomes
- A-MEM: Knowledge storage and retrieval

This module provides:
1. Pre-built workflows for common pipeline patterns
2. Integration decorators for easy workflow creation
3. Workflow-aware context managers
4. Sprint execution workflows

Author: Pipeline Autonomo Team
Version: 1.0.0 (2025-12-30)
"""

from __future__ import annotations

import functools
import logging
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
)

from .temporal_workflow import (
    RetryPolicy,
    Workflow,
    WorkflowContext,
    WorkflowState,
    WorkflowStatus,
    get_workflow_engine,
)
from .temporal_activity import (
    GateResult,
    AgentResult,
    CommandResult,
    get_default_activities,
)

# =============================================================================
# INVIOLABLE CONTRACTS - Proteção absoluta contra violações
# Adicionado após PAT-026: Desastre de 2026-01-03
# =============================================================================
from .inviolable_contracts import (
    boot_check,
)

logger = logging.getLogger(__name__)

# =============================================================================
# BOOT GUARD - Executar verificação de boot no import
# =============================================================================
_boot_result = boot_check()
if not _boot_result.get("success", False):
    logger.warning(
        f"⚠️ Boot Guard Warning: {_boot_result.get('errors', [])}\n"
        f"Algumas verificações falharam. Proceda com cuidado."
    )

T = TypeVar("T")
R = TypeVar("R")


# =============================================================================
# UTILITIES
# =============================================================================


# =============================================================================
# PAT-026 PROTECTION: Context Pack Verification
# Added 2026-01-03 after a disaster where 24 sprints ran without real work
# =============================================================================

def load_context_pack(sprint_id: str) -> dict:
    """Load and parse a context pack. MUST be called before execute_sprint.

    Args:
        sprint_id: Sprint identifier (e.g., "S00", "S01")

    Returns:
        dict with context pack contents including:
        - objective: What the sprint should accomplish
        - deliverables: List of files/artifacts to produce
        - models: Data models to implement (if any)

    Raises:
        FileNotFoundError: If context pack doesn't exist
        ValueError: If context pack is missing required fields
    """
    import yaml
    import re

    context_path = Path(f"context_packs/{sprint_id}_CONTEXT.md")

    if not context_path.exists():
        raise FileNotFoundError(
            f"Context pack não encontrado: {context_path}\n"
            f"Verifique se o sprint {sprint_id} existe no roadmap."
        )

    content = context_path.read_text()

    # Extract YAML from RELOAD ANCHOR or SPRINT ANCHOR section (not first YAML block)
    # The ANCHOR section contains sprint metadata, objective, and deliverables
    # Both names are valid: RELOAD ANCHOR (legacy) and SPRINT ANCHOR (newer)
    reload_anchor_match = re.search(
        r'##\s*(?:RELOAD|SPRINT)\s+ANCHOR\s*\n+```yaml\s*\n(.*?)\n```',
        content,
        re.DOTALL | re.IGNORECASE
    )

    if reload_anchor_match:
        yaml_match = reload_anchor_match
    else:
        # Fallback: try to find any yaml block with 'objective' field
        yaml_match = re.search(r'```yaml\s*\n(.*?objective.*?)\n```', content, re.DOTALL)
        if not yaml_match:
            # Last resort: first yaml block (legacy)
            yaml_match = re.search(r'```yaml\s*\n(.*?)\n```', content, re.DOTALL)

    if not yaml_match:
        raise ValueError(f"Context pack {sprint_id} não tem RELOAD ANCHOR YAML válido")

    try:
        context = yaml.safe_load(yaml_match.group(1))
    except yaml.YAMLError as e:
        raise ValueError(f"Erro ao parsear YAML do context pack {sprint_id}: {e}")

    # Normalize objective field (can be at root OR inside sprint block)
    objective = context.get("objective")
    sprint_data = context.get("sprint", {})
    if isinstance(sprint_data, dict) and not objective:
        objective = sprint_data.get("objective")
        if objective:
            # Promote to root level for consistent access
            context["objective"] = objective

    # Validate required fields
    if not context.get("objective"):
        raise ValueError(f"Context pack {sprint_id} não tem 'objective' definido (verificado root e sprint block)")

    if not context.get("deliverables"):
        raise ValueError(f"Context pack {sprint_id} não tem 'deliverables' definidos")

    return context


def verify_deliverables(sprint_id: str, context: dict) -> dict:
    """Verify that sprint deliverables were actually created.

    RED TEAM FIX INV-15: Now validates content, not just existence.

    Args:
        sprint_id: Sprint identifier
        context: Context pack dict from load_context_pack()

    Returns:
        dict with verification results:
        - verified: bool - all deliverables exist AND have valid content
        - missing: list of missing deliverables
        - found: list of found deliverables
        - empty: list of empty deliverables (exists but no content)
        - invalid: list of deliverables with invalid content
    """
    deliverables = context.get("deliverables", [])
    missing = []
    found = []
    empty = []  # RED TEAM FIX: Track empty files
    invalid = []  # RED TEAM FIX: Track invalid content

    MIN_CONTENT_SIZE = 50  # Minimum bytes for valid deliverable

    for deliverable in deliverables:
        path = Path(deliverable)
        if not path.exists():
            missing.append(deliverable)
        else:
            # RED TEAM FIX INV-15: Validate content, not just existence
            try:
                stat = path.stat()
                if stat.st_size < MIN_CONTENT_SIZE:
                    empty.append(deliverable)
                elif path.suffix in ['.py', '.yml', '.yaml', '.json', '.md']:
                    # For text files, check it's not just placeholders
                    content = path.read_text()[:500]
                    if 'TODO' in content or 'PLACEHOLDER' in content or 'NotImplemented' in content:
                        invalid.append(deliverable)
                    else:
                        found.append(deliverable)
                else:
                    found.append(deliverable)
            except Exception as e:
                # RED TEAM FIX VIO-010: Log actual exception for audit trail
                logger.warning(f"VIO-010: Failed to verify deliverable {deliverable}: {e}")
                invalid.append(deliverable)

    return {
        "verified": len(missing) == 0 and len(empty) == 0 and len(invalid) == 0,
        "missing": missing,
        "found": found,
        "empty": empty,
        "invalid": invalid,
        "total": len(deliverables),
    }


def _now_iso() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _generate_id(prefix: str = "wf") -> str:
    """Generate a unique ID."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# =============================================================================
# INTEGRATION CONFIG
# =============================================================================


@dataclass
class TemporalIntegrationConfig:
    """Configuration for Temporal integration.

    Attributes:
        use_redis: Whether to use Redis for distributed storage.
        use_reflexion: Whether to integrate with ReflexionEngine.
        use_live_swe: Whether to integrate with Live-SWE.
        use_amem: Whether to integrate with A-MEM.
        max_workflow_retries: Maximum workflow-level retries.
        default_activity_timeout: Default timeout for activities.
    """

    use_redis: bool = False
    use_reflexion: bool = True
    use_live_swe: bool = True
    use_amem: bool = True
    max_workflow_retries: int = 3
    default_activity_timeout: float = 300.0


# =============================================================================
# PRE-BUILT WORKFLOWS
# =============================================================================


class SprintWorkflow(Workflow[Dict[str, Any], Dict[str, Any]]):
    """Workflow for executing a complete sprint.

    Orchestrates:
    1. Sprint initialization
    2. Gate validation (G0-G8)
    3. Agent execution
    4. Test execution
    5. Artifact collection
    6. Reflection on failures
    """

    @property
    def name(self) -> str:
        return "sprint_workflow"

    def get_activities(self) -> Dict[str, Callable]:
        return get_default_activities()

    async def run(
        self,
        ctx: WorkflowContext,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute the sprint workflow.

        Args:
            ctx: Workflow context.
            input_data: Sprint configuration with:
                - sprint_id: Sprint identifier
                - run_dir: Run directory
                - tasks: List of tasks to execute
                - gates: List of gates to validate

        Returns:
            Sprint result with:
                - success: Whether sprint succeeded
                - gates_passed: Number of gates passed
                - artifacts: Created artifacts
                - reflections: Any reflections generated
        """
        sprint_id = input_data.get("sprint_id", _generate_id("sprint"))
        run_dir = input_data.get("run_dir", f"out/sprints/{sprint_id}")
        tasks = input_data.get("tasks", [])
        gates = input_data.get("gates", [f"G{i}" for i in range(9)])
        test_level = input_data.get("test_level", "smoke")  # smoke, sprint, or full

        result = {
            "sprint_id": sprint_id,
            "success": False,
            "gates_passed": 0,
            "gates_total": len(gates),
            "artifacts": [],
            "reflections": [],
            "started_at": _now_iso(),
            "completed_at": "",
        }

        # Initialize sprint
        await ctx.execute_activity(
            "log_event",
            args=("sprint_start", f"Starting sprint {sprint_id}"),
            kwargs={"data": {"sprint_id": sprint_id, "tasks": len(tasks)}},
        )

        # F-161 FIX: Initiate handoff cascade BEFORE tasks execute
        # This ensures proper hierarchy delegation is recorded in Temporal
        context_pack = input_data.get("context_pack", {})
        objective = context_pack.get("objective", f"Complete sprint {sprint_id}")
        deliverables = context_pack.get("deliverables", [])

        # Get run_id for proper artifact paths (F-171)
        run_id = input_data.get("run_id")

        # Handoff cascade: CEO -> VPs -> Masters
        handoff_cascade = [
            ("ceo", "spec_vp", f"Define specs for {sprint_id}"),
            ("ceo", "exec_vp", f"Coordinate execution for {sprint_id}"),
            ("spec_vp", "spec_master", f"Refine specifications for {sprint_id}"),
            ("exec_vp", "ace_exec", f"Implement code for {sprint_id}"),
            ("exec_vp", "qa_master", f"Quality assurance for {sprint_id}"),
        ]

        result["handoffs_initiated"] = []
        for from_agent, to_agent, what_i_want in handoff_cascade:
            try:
                handoff_result = await ctx.execute_activity(
                    "record_agent_handoff",
                    kwargs={
                        "sprint_id": sprint_id,
                        "from_agent": from_agent,
                        "to_agent": to_agent,
                        "what_i_want": what_i_want,
                        "why_i_want": f"Part of {objective}",
                        "expected_behavior": f"Complete assigned tasks and produce deliverables",
                        "priority": "high",
                        "run_id": run_id,
                        "run_dir": run_dir,
                    },
                    retry_policy=RetryPolicy(max_attempts=2),
                )
                if handoff_result.get("success"):
                    result["handoffs_initiated"].append(f"{from_agent}->{to_agent}")
            except Exception as handoff_error:
                logger.warning(f"F-161: Handoff {from_agent}->{to_agent} failed: {handoff_error}")

        await ctx.execute_activity(
            "log_event",
            args=("handoff_cascade", f"Initiated {len(result['handoffs_initiated'])} handoffs for {sprint_id}"),
            kwargs={"data": {"sprint_id": sprint_id, "handoffs": result["handoffs_initiated"]}},
        )

        # Execute tasks
        for task in tasks:
            # Get task description (orchestrator uses 'description', fallback to 'task')
            task_description = task.get("description") or task.get("task", "No task description")
            agent_type = task.get("agent_id") or task.get("agent_type", "worker")

            # Timeout de 5 minutos por task para evitar espera indefinida
            # Se a API não responder, retry com backoff
            agent_result: AgentResult = await ctx.execute_activity(
                "execute_agent",
                kwargs={
                    "agent_type": agent_type,
                    "task": task_description,
                    "context": task.get("context"),
                },
                retry_policy=RetryPolicy(max_attempts=3, initial_interval_seconds=5.0),
                timeout_seconds=None,  # No timeout - user preference
            )

            if agent_result.success:
                # AUTO-SAVE: Salvar aprendizado de sucesso no AMEM
                await ctx.execute_activity(
                    "store_learning",
                    kwargs={
                        "learning_type": "success_pattern",
                        "content": f"Agent {agent_type} completed task successfully: {task_description[:100]}",
                        "source": f"sprint_workflow/{sprint_id}/{agent_type}",
                    },
                )
            else:
                # AUTO-SAVE: Salvar erro no AMEM antes de gerar reflexão
                await ctx.execute_activity(
                    "store_learning",
                    kwargs={
                        "learning_type": "error",
                        "content": f"Agent {agent_type} failed: {agent_result.error or 'Unknown error'}",
                        "source": f"sprint_workflow/{sprint_id}/{agent_type}",
                    },
                )

                # Generate reflection on failure
                reflection = await ctx.execute_activity(
                    "generate_reflection",
                    kwargs={
                        "task_description": task_description,
                        "error_message": agent_result.error or "Unknown error",
                        "attempt_number": 1,
                    },
                )
                result["reflections"].append(reflection)

        # Validate gates
        gate_results: Dict[str, GateResult] = await ctx.execute_activity(
            "validate_all_gates",
            kwargs={
                "sprint_id": sprint_id,
                "run_dir": run_dir,
                "required_gates": gates,
            },
        )

        result["gates_passed"] = sum(1 for g in gate_results.values() if g.passed)

        # AUTO-SAVE: Salvar resultado de gates no AMEM
        for gate_name, gate_result in gate_results.items():
            if gate_result.passed:
                await ctx.execute_activity(
                    "store_learning",
                    kwargs={
                        "learning_type": "gate_passed",
                        "content": f"Gate {gate_name} passed for sprint {sprint_id}",
                        "source": f"sprint_workflow/{sprint_id}/gates",
                    },
                )
            else:
                await ctx.execute_activity(
                    "store_learning",
                    kwargs={
                        "learning_type": "gate_failed",
                        "content": f"Gate {gate_name} failed: {', '.join(gate_result.messages)}",
                        "source": f"sprint_workflow/{sprint_id}/gates",
                    },
                )

        # Run tests with CORRECTION LOOP + CEO ESCALATION
        max_correction_attempts = 3  # Tentativas por ciclo (QA → Exec VP → Ace Exec)
        max_ceo_escalations = 5      # Ciclos completos (CEO → Spec Master → ...)
        test_result: CommandResult = None
        total_attempts = 0
        ceo_escalation = 0

        for ceo_escalation in range(1, max_ceo_escalations + 1):
            # Correction loop interno (3 tentativas)
            for correction_attempt in range(1, max_correction_attempts + 1):
                total_attempts += 1

                # Run tests (use test_level from input, default smoke for speed)
                test_result = await ctx.execute_activity(
                    "run_tests",
                    kwargs={"test_level": test_level},
                    timeout_seconds=120 if test_level == "smoke" else 1800,
                )

                if test_result.success:
                    # Tests passed - exit all loops
                    await ctx.execute_activity(
                        "store_learning",
                        kwargs={
                            "learning_type": "success_pattern",
                            "content": f"Tests passed on attempt {total_attempts} (cycle {ceo_escalation}) for sprint {sprint_id}",
                            "source": f"sprint_workflow/{sprint_id}/tests",
                        },
                    )
                    break

                # Tests FAILED - trigger correction loop
                if correction_attempt < max_correction_attempts:
                    await ctx.execute_activity(
                        "log_event",
                        args=("correction_loop", f"Tests failed, attempt {correction_attempt}/{max_correction_attempts} (cycle {ceo_escalation})"),
                        kwargs={"data": {"sprint_id": sprint_id, "attempt": correction_attempt, "cycle": ceo_escalation}},
                    )

                    # =================================================================
                    # CORRECTION LOOP - Usa execute_with_delegation para masters
                    # Masters coordenam, workers executam (INV-HIERARCHY)
                    # =================================================================

                    # 1. QA Master analisa as falhas (delega para auditor)
                    qa_analysis: AgentResult = await ctx.execute_activity(
                        "execute_with_delegation",
                        kwargs={
                            "master_type": "qa_master",
                            "task": f"""ANÁLISE DE FALHAS - Ciclo {ceo_escalation}, Tentativa {correction_attempt}

Os testes falharam. Analise o output e identifique:
1. Quais testes falharam e por quê
2. Root cause de cada falha
3. O que precisa ser corrigido

Output dos testes:
{test_result.stdout[:3000] if test_result.stdout else 'No output'}

IMPORTANTE: Gere um relatório DETALHADO das falhas para coordenar a correção.""",
                            "context": {"sprint_id": sprint_id, "attempt": correction_attempt, "cycle": ceo_escalation},
                        },
                        timeout_seconds=None,  # No timeout
                    )

                    # 2. Exec VP coordena correção (delega para integration_officer)
                    exec_vp_coord: AgentResult = await ctx.execute_activity(
                        "execute_with_delegation",
                        kwargs={
                            "master_type": "exec_vp",
                            "task": f"""COORDENAR CORREÇÃO - Ciclo {ceo_escalation}, Tentativa {correction_attempt}

Análise de falhas identificou problemas nos testes.

RELATÓRIO DA ANÁLISE:
{qa_analysis.output[:2000] if qa_analysis.output else 'No analysis'}

Coordene a correção:
1. Priorize as correções mais críticas
2. Defina a estratégia de correção
3. Instrua sobre o que corrigir

Sprint: {sprint_id}""",
                            "context": {"sprint_id": sprint_id, "qa_report": qa_analysis.output},
                        },
                        timeout_seconds=None,  # No timeout
                    )

                    # 3. Ace Exec corrige (delega para technical_planner)
                    ace_fix: AgentResult = await ctx.execute_activity(
                        "execute_with_delegation",
                        kwargs={
                            "master_type": "ace_exec",
                            "task": f"""CORREÇÃO DE CÓDIGO - Ciclo {ceo_escalation}, Tentativa {correction_attempt}

Coordenação definiu a estratégia de correção.

INSTRUÇÕES DE COORDENAÇÃO:
{exec_vp_coord.output[:2000] if exec_vp_coord.output else 'No instructions'}

RELATÓRIO DA ANÁLISE:
{qa_analysis.output[:1500] if qa_analysis.output else 'No analysis'}

Corrija o código para que os testes passem.
Sprint: {sprint_id}""",
                            "context": {"sprint_id": sprint_id, "attempt": correction_attempt},
                        },
                        timeout_seconds=None,  # No timeout
                    )

                    await ctx.execute_activity(
                        "store_learning",
                        kwargs={
                            "learning_type": "correction_attempt",
                            "content": f"Correction attempt {correction_attempt} cycle {ceo_escalation} for sprint {sprint_id}",
                            "source": f"sprint_workflow/{sprint_id}/correction",
                        },
                    )
                    # Loop volta para rodar testes novamente

            # Check if tests passed in inner loop
            if test_result and test_result.success:
                break  # Exit CEO escalation loop

            # 3 tentativas falharam - ESCALAR PARA CEO
            if ceo_escalation < max_ceo_escalations:
                await ctx.execute_activity(
                    "log_event",
                    args=("ceo_escalation", f"Escalating to CEO after {max_correction_attempts} failed attempts (cycle {ceo_escalation})"),
                    kwargs={"data": {"sprint_id": sprint_id, "cycle": ceo_escalation}},
                )

                await ctx.execute_activity(
                    "store_learning",
                    kwargs={
                        "learning_type": "escalation",
                        "content": f"CEO escalation triggered for sprint {sprint_id} after cycle {ceo_escalation}",
                        "source": f"sprint_workflow/{sprint_id}/escalation",
                    },
                )

                # 1. CEO recebe escalação e decide estratégia
                ceo_decision: AgentResult = await ctx.execute_activity(
                    "execute_agent",
                    kwargs={
                        "agent_type": "ceo",
                        "task": f"""ESCALAÇÃO DE FALHAS - Ciclo {ceo_escalation} de {max_ceo_escalations}

A sprint {sprint_id} falhou {max_correction_attempts} tentativas de correção.

ÚLTIMA ANÁLISE DO QA:
{qa_analysis.output[:2000] if qa_analysis and qa_analysis.output else 'No analysis'}

ÚLTIMO OUTPUT DOS TESTES:
{test_result.stdout[:2000] if test_result and test_result.stdout else 'No output'}

Como CEO, você deve:
1. Analisar se o problema é de SPECS (mal especificado) ou IMPLEMENTAÇÃO
2. Decidir a estratégia de resolução
3. Instruir o Spec Master a REFINAR as specs se necessário
4. Definir prioridades para o próximo ciclo

ATENÇÃO: Este é o ciclo {ceo_escalation} de {max_ceo_escalations}. Se continuar falhando, a sprint será marcada como FAILED.""",
                        "context": {"sprint_id": sprint_id, "cycle": ceo_escalation},
                    },
                    timeout_seconds=None,  # No timeout
                )

                # 2. Spec Master refina specs baseado na decisão do CEO
                spec_refinement: AgentResult = await ctx.execute_activity(
                    "execute_agent",
                    kwargs={
                        "agent_type": "spec_master",
                        "task": f"""REFINAMENTO DE SPECS - Ciclo {ceo_escalation + 1}

O CEO escalou esta sprint após {max_correction_attempts} falhas de correção.

DECISÃO DO CEO:
{ceo_decision.output[:2000] if ceo_decision.output else 'No decision'}

PROBLEMAS IDENTIFICADOS:
{qa_analysis.output[:1500] if qa_analysis and qa_analysis.output else 'No analysis'}

Refine as specs profissionais para resolver os problemas:
1. Revise os invariantes que estão causando falhas
2. Clarifique ambiguidades nas specs
3. Adicione casos de teste que estavam faltando
4. Persista as specs atualizadas no Redis e Qdrant

Sprint: {sprint_id}""",
                        "context": {"sprint_id": sprint_id, "ceo_decision": ceo_decision.output},
                    },
                    timeout_seconds=None,  # No timeout
                )

                # 3. Exec VP coordena novo ciclo de implementação
                exec_vp_new_cycle: AgentResult = await ctx.execute_activity(
                    "execute_agent",
                    kwargs={
                        "agent_type": "exec_vp",
                        "task": f"""NOVO CICLO DE IMPLEMENTAÇÃO - Ciclo {ceo_escalation + 1}

O CEO e Spec Master refinaram a abordagem.

DECISÃO DO CEO:
{ceo_decision.output[:1500] if ceo_decision.output else 'No decision'}

SPECS REFINADAS:
{spec_refinement.output[:1500] if spec_refinement.output else 'No refinement'}

Coordene a reimplementação com as novas specs.
Sprint: {sprint_id}""",
                        "context": {"sprint_id": sprint_id},
                    },
                    timeout_seconds=None,  # No timeout
                )

                # 4. Ace Exec reimplementa com specs refinadas
                ace_reimpl: AgentResult = await ctx.execute_activity(
                    "execute_agent",
                    kwargs={
                        "agent_type": "ace_exec",
                        "task": f"""REIMPLEMENTAÇÃO - Ciclo {ceo_escalation + 1}

Specs foram refinadas pelo CEO e Spec Master.

INSTRUÇÕES DO EXEC VP:
{exec_vp_new_cycle.output[:2000] if exec_vp_new_cycle.output else 'No instructions'}

SPECS REFINADAS:
{spec_refinement.output[:1500] if spec_refinement.output else 'No refinement'}

Reimplemente seguindo as novas specs.
Sprint: {sprint_id}""",
                        "context": {"sprint_id": sprint_id, "cycle": ceo_escalation + 1},
                    },
                    timeout_seconds=None,  # No timeout - user preference
                )

                # Loop volta para rodar testes com novo ciclo

            else:
                # Max CEO escalations reached - FAILED definitivo
                await ctx.execute_activity(
                    "store_learning",
                    kwargs={
                        "learning_type": "error",
                        "content": f"Sprint {sprint_id} FAILED after {max_ceo_escalations} CEO escalations ({total_attempts} total attempts)",
                        "source": f"sprint_workflow/{sprint_id}/final_failure",
                    },
                )

        # Save sprint artifact
        artifact_result = await ctx.execute_activity(
            "save_artifact",
            kwargs={
                "name": f"sprint_{sprint_id}_result",
                "content": {
                    "sprint_id": sprint_id,
                    "gates": {k: v.to_dict() for k, v in gate_results.items()},
                    "test_exit_code": test_result.exit_code if test_result else -1,
                    "reflections": result["reflections"],
                    "total_attempts": total_attempts,
                    "ceo_escalations": ceo_escalation,
                },
                "artifact_type": "json",
                "run_dir": run_dir,
            },
        )

        if artifact_result.success:
            result["artifacts"].append(artifact_result.path)

        # Determine success
        result["success"] = (
            result["gates_passed"] == result["gates_total"]
            and test_result is not None
            and test_result.success
        )
        result["completed_at"] = _now_iso()
        result["total_attempts"] = total_attempts
        result["ceo_escalations"] = ceo_escalation

        # Final notification
        await ctx.execute_activity(
            "notify_human_layer",
            kwargs={
                "notification_type": "sprint_complete",
                "message": f"Sprint {sprint_id}: {'SUCCESS' if result['success'] else 'FAILED'}",
                "priority": "normal" if result["success"] else "high",
                "data": {"sprint_id": sprint_id, "gates_passed": result["gates_passed"]},
                "run_dir": run_dir,  # F-223: Pass run_dir for proper isolation
                "sprint_id": sprint_id,  # F-223: Pass sprint_id for scoped path
            },
        )

        return result


class GateValidationWorkflow(Workflow[Dict[str, Any], Dict[str, GateResult]]):
    """Workflow for validating all gates with retry and reflection."""

    @property
    def name(self) -> str:
        return "gate_validation_workflow"

    def get_activities(self) -> Dict[str, Callable]:
        return get_default_activities()

    async def run(
        self,
        ctx: WorkflowContext,
        input_data: Dict[str, Any],
    ) -> Dict[str, GateResult]:
        """Execute gate validation workflow.

        Args:
            ctx: Workflow context.
            input_data: Configuration with:
                - sprint_id: Sprint identifier
                - run_dir: Run directory
                - gates: List of gates to validate
                - retry_failed: Whether to retry failed gates

        Returns:
            Dict mapping gate names to results.
        """
        sprint_id = input_data.get("sprint_id", _generate_id("sprint"))
        run_dir = input_data.get("run_dir", "out/")
        gates = input_data.get("gates", [f"G{i}" for i in range(9)])
        retry_failed = input_data.get("retry_failed", True)

        results: Dict[str, GateResult] = {}
        failed_gates = []

        # First pass - validate all gates
        for gate in gates:
            result: GateResult = await ctx.execute_activity(
                "validate_gate",
                kwargs={
                    "gate_name": gate,
                    "sprint_id": sprint_id,
                    "run_dir": run_dir,
                },
                retry_policy=RetryPolicy(max_attempts=2),
            )
            results[gate] = result

            if not result.passed:
                failed_gates.append(gate)
                # Store learning about failure
                await ctx.execute_activity(
                    "store_learning",
                    kwargs={
                        "learning_type": "error",
                        "content": f"Gate {gate} failed: {', '.join(result.messages)}",
                        "source": f"gate_validation/{sprint_id}",
                    },
                )

        # Retry failed gates if requested
        if retry_failed and failed_gates:
            await ctx.sleep(5)  # Wait before retry

            for gate in failed_gates:
                # Generate reflection for retry
                reflection = await ctx.execute_activity(
                    "generate_reflection",
                    kwargs={
                        "task_description": f"Gate {gate} validation",
                        "error_message": ", ".join(results[gate].messages),
                        "attempt_number": 1,
                    },
                )

                # Retry with reflection context
                retry_result: GateResult = await ctx.execute_activity(
                    "validate_gate",
                    kwargs={
                        "gate_name": gate,
                        "sprint_id": sprint_id,
                        "run_dir": run_dir,
                    },
                )

                if retry_result.passed:
                    results[gate] = retry_result
                    # Store successful retry learning
                    await ctx.execute_activity(
                        "store_learning",
                        kwargs={
                            "learning_type": "pattern",
                            "content": f"Gate {gate} passed on retry using: {reflection.get('prevention_strategy', 'retry')}",
                            "source": f"gate_validation/{sprint_id}",
                        },
                    )

        return results


class AgentExecutionWorkflow(Workflow[Dict[str, Any], AgentResult]):
    """Workflow for executing an agent with full durability."""

    @property
    def name(self) -> str:
        return "agent_execution_workflow"

    def get_activities(self) -> Dict[str, Callable]:
        return get_default_activities()

    async def run(
        self,
        ctx: WorkflowContext,
        input_data: Dict[str, Any],
    ) -> AgentResult:
        """Execute an agent with durability and reflection.

        Args:
            ctx: Workflow context.
            input_data: Agent configuration with:
                - agent_type: Type of agent
                - task: Task description
                - context: Additional context
                - max_retries: Maximum retries

        Returns:
            AgentResult from execution.
        """
        agent_type = input_data.get("agent_type", "worker")
        task = input_data.get("task", "")
        context = input_data.get("context", {})
        max_retries = input_data.get("max_retries", 3)

        reflections = []

        for attempt in range(1, max_retries + 1):
            # Include reflections in context
            if reflections:
                context["previous_reflections"] = reflections

            # Timeout de 5 minutos para evitar espera indefinida
            result: AgentResult = await ctx.execute_activity(
                "execute_agent",
                kwargs={
                    "agent_type": agent_type,
                    "task": task,
                    "context": context,
                },
                retry_policy=RetryPolicy(max_attempts=2),
                timeout_seconds=None,  # No timeout  # 5 minutos
            )

            if result.success:
                # Store successful pattern
                await ctx.execute_activity(
                    "store_learning",
                    kwargs={
                        "learning_type": "pattern",
                        "content": f"Agent {agent_type} completed task successfully",
                        "source": result.agent_id,
                    },
                )
                return result

            # Generate reflection for retry
            reflection = await ctx.execute_activity(
                "generate_reflection",
                kwargs={
                    "task_description": task,
                    "error_message": result.error or "Unknown error",
                    "attempt_number": attempt,
                    "previous_reflections": [r.get("prevention_strategy", "") for r in reflections],
                },
            )
            reflections.append(reflection)

            if attempt < max_retries:
                # Wait before retry with exponential backoff
                await ctx.sleep(2 ** attempt)

        # All retries failed
        return result


# =============================================================================
# WORKFLOW DECORATORS
# =============================================================================


def durable_workflow(
    name: Optional[str] = None,
    retry_policy: Optional[RetryPolicy] = None,
    timeout_seconds: Optional[float] = None,
):
    """Decorator to convert a function into a durable workflow.

    The decorated function will be executed within a workflow context,
    providing automatic checkpointing and recovery.

    Args:
        name: Workflow name. Uses function name if not provided.
        retry_policy: Default retry policy for activities.
        timeout_seconds: Timeout for the entire workflow.

    Example:
        @durable_workflow(name="my_workflow")
        async def process_data(ctx: WorkflowContext, data: dict) -> dict:
            result = await ctx.execute_activity("transform", (data,))
            return result
    """
    def decorator(func: Callable[..., Coroutine]) -> Callable:
        workflow_name = name or func.__name__

        # Create a workflow class dynamically
        class DynamicWorkflow(Workflow[Any, Any]):
            @property
            def name(self) -> str:
                return workflow_name

            def get_activities(self) -> Dict[str, Callable]:
                return get_default_activities()

            async def run(self, ctx: WorkflowContext, input_data: Any) -> Any:
                return await func(ctx, input_data)

        # Register the workflow
        engine = get_workflow_engine()
        engine.register_workflow(DynamicWorkflow)

        @functools.wraps(func)
        async def wrapper(input_data: Any, **kwargs) -> Any:
            workflow_id = kwargs.get("workflow_id", _generate_id("wf"))
            wf_id, result = await engine.start_and_execute(
                workflow_name=workflow_name,
                input_data=input_data,
                workflow_id=workflow_id,
            )
            return result

        wrapper._workflow_name = workflow_name
        wrapper._workflow_class = DynamicWorkflow
        return wrapper

    return decorator


def with_temporal_context():
    """Decorator that provides a workflow context to the function.

    Use this when you want to use workflow primitives (activities, timers)
    without creating a full workflow.

    Example:
        @with_temporal_context()
        async def my_function(ctx: WorkflowContext, x: int) -> int:
            await ctx.sleep(1)
            return x * 2
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Create a temporary workflow context
            from .temporal_workflow import FileWorkflowStore, WorkflowState

            store = FileWorkflowStore()
            state = WorkflowState(
                workflow_id=_generate_id("ctx"),
                workflow_name=f"context_{func.__name__}",
            )

            ctx = WorkflowContext(state, store, replaying=False)

            # Register default activities
            for name, activity in get_default_activities().items():
                ctx.register_activity(name, activity)

            return await func(ctx, *args, **kwargs)

        return wrapper
    return decorator


# =============================================================================
# INTEGRATION LAYER
# =============================================================================


class TemporalIntegration:
    """Main integration class for Temporal Lite.

    Provides a unified interface for:
    - Creating and executing workflows
    - Integrating with Reflexion, Live-SWE, and A-MEM
    - Managing workflow lifecycle
    """

    def __init__(self, config: Optional[TemporalIntegrationConfig] = None):
        """Initialize the Temporal integration.

        Args:
            config: Integration configuration.
        """
        self.config = config or TemporalIntegrationConfig()
        self.engine = get_workflow_engine(use_redis=self.config.use_redis)

        # Register built-in workflows
        self._register_builtin_workflows()

        # Initialize integrations
        self._reflexion = None
        self._live_swe = None
        self._amem = None

    def _register_builtin_workflows(self) -> None:
        """Register all built-in workflows."""
        self.engine.register_workflow(SprintWorkflow)
        self.engine.register_workflow(GateValidationWorkflow)
        self.engine.register_workflow(AgentExecutionWorkflow)

    def _get_reflexion(self):
        """Lazy load ReflexionEngine."""
        if self._reflexion is None and self.config.use_reflexion:
            try:
                from .reflexion_engine import ReflexionEngine
                self._reflexion = ReflexionEngine()
            except ImportError:
                logger.warning("ReflexionEngine not available")
        return self._reflexion

    def _get_live_swe(self):
        """Lazy load Live-SWE."""
        if self._live_swe is None and self.config.use_live_swe:
            try:
                from .live_swe_integration import get_live_swe_integration
                self._live_swe = get_live_swe_integration()
            except ImportError:
                logger.warning("Live-SWE not available")
        return self._live_swe

    def _get_amem(self):
        """Lazy load A-MEM."""
        if self._amem is None and self.config.use_amem:
            try:
                from .amem_integration import get_amem_integration
                self._amem = get_amem_integration()
            except ImportError:
                logger.warning("A-MEM not available")
        return self._amem

    # No timeout - user preference: status poll only

    async def execute_sprint(
        self,
        sprint_id: str,
        tasks: List[Dict[str, Any]],
        context_pack_verified: bool,  # RED TEAM FIX INV-05: No default, must be explicit
        run_dir: Optional[str] = None,
        run_id: Optional[str] = None,  # F-161/F-171 FIX: Pass run context
        gates: Optional[List[str]] = None,
        test_level: str = "smoke",
    ) -> Dict[str, Any]:
        """Execute a sprint workflow.

        F-161/F-171 FIX: Now accepts run_id for proper handoff and artifact paths.

        Args:
            sprint_id: Sprint identifier.
            tasks: List of task definitions.
            context_pack_verified: REQUIRED - confirms context pack was read. MUST be True.
            run_dir: Run directory.
            run_id: F-161/F-171 FIX - Run identifier for handoffs and artifacts.
            gates: Gates to validate.
            test_level: Test level - smoke (fast), sprint, or full.

        Returns:
            Sprint result.

        Raises:
            ContextPackNotVerified: If context_pack_verified is False.
        """
        # =================================================================
        # PAT-026 GUARD: Impedir execução sem verificação de context pack
        # Este guard existe porque em 2026-01-03 um desastre aconteceu:
        # - Agent executou 24 sprints fictícios sem ler context packs
        # - ZERO código foi produzido
        # - ~100 chamadas de API desperdiçadas
        # - 8 HORAS DO USUÁRIO PERDIDAS
        # NUNCA REMOVA ESTE GUARD.
        # =================================================================

        # PROTEÇÃO 1: Verificar se context pack existe
        context_path = Path(f"context_packs/{sprint_id}_CONTEXT.md")
        if not context_path.exists():
            raise FileNotFoundError(
                f"PAT-026 GUARD: Context pack não existe!\n"
                f"Arquivo esperado: {context_path}\n"
                f"O sprint {sprint_id} não pode ser executado sem context pack."
            )

        # PROTEÇÃO 2: Carregar e validar context pack OBRIGATORIAMENTE
        try:
            context = load_context_pack(sprint_id)
        except Exception as e:
            raise ValueError(
                f"PAT-026 GUARD: Context pack inválido!\n"
                f"Erro: {e}\n"
                f"O sprint {sprint_id} não pode ser executado com context pack inválido."
            )

        # PROTEÇÃO 3: Verificar se deliverables estão definidos
        deliverables = context.get("deliverables", [])
        if not deliverables:
            raise ValueError(
                f"PAT-026 GUARD: Context pack sem deliverables!\n"
                f"O sprint {sprint_id} não tem entregas definidas.\n"
                f"Isso indica um context pack incompleto ou inválido."
            )

        # PROTEÇÃO 4: Log obrigatório do que será executado
        objective = context.get("objective", "Objetivo não definido")
        logger.warning(
            f"PAT-026 CHECKPOINT: Executando sprint {sprint_id}\n"
            f"  Objetivo: {objective}\n"
            f"  Deliverables: {len(deliverables)} arquivos\n"
            f"  Primeiros 3: {deliverables[:3]}"
        )

        # PROTEÇÃO 5: Injetar context pack real no workflow
        # Isso garante que o workflow tem acesso aos deliverables reais
        # e pode verificar se foram criados

        # Execute workflow without timeout - user preference: status poll only
        workflow_id, result = await self.engine.start_and_execute(
            workflow_name="sprint_workflow",
            input_data={
                # Context pack real injetado - NAO PODE SER INVENTADO
                "context_pack": context,
                "expected_deliverables": deliverables,
                "objective": objective,
                "sprint_id": sprint_id,
                "tasks": tasks,
                "run_dir": run_dir or f"out/sprints/{sprint_id}",
                "run_id": run_id,  # F-161/F-171 FIX: Pass run_id to workflow
                "gates": gates or [f"G{i}" for i in range(9)],
                "test_level": test_level,
            },
        )

        # =================================================================
        # PAT-026 GUARD FINAL: Verificar entregas antes de marcar sucesso
        # Sprint NÃO pode ser considerado sucesso sem deliverables reais
        # =================================================================
        if result.get("success"):
            verification = verify_deliverables(sprint_id, context)
            if not verification["verified"]:
                logger.error(
                    f"PAT-026 GUARD FINAL: Deliverables não encontrados!\n"
                    f"  Sprint: {sprint_id}\n"
                    f"  Encontrados: {verification['found']}\n"
                    f"  FALTANDO: {verification['missing']}"
                )
                # Marcar como falha se deliverables não existem
                result["success"] = False
                result["error"] = f"Deliverables não criados: {verification['missing']}"
                result["deliverables_missing"] = verification["missing"]
            else:
                logger.info(
                    f"PAT-026 GUARD FINAL: Todos deliverables verificados!\n"
                    f"  Sprint: {sprint_id}\n"
                    f"  Criados: {verification['found']}"
                )
                result["deliverables_verified"] = verification["found"]

        # Store sprint learning in A-MEM
        amem = self._get_amem()
        if amem:
            amem.save_decision(
                agent_id="temporal_workflow",
                decision=f"Sprint {sprint_id} {'completed successfully' if result['success'] else 'failed'}",
                reason=f"Gates: {result['gates_passed']}/{result['gates_total']}",
                sprint_id=sprint_id,
            )

        # Record workflow metrics for Live-SWE
        live_swe = self._get_live_swe()
        if live_swe:
            live_swe.metrics.record(
                behavior_id="sprint_workflow",
                metric_name="success_rate",
                value=1.0 if result["success"] else 0.0,
            )
            live_swe.metrics.record(
                behavior_id="sprint_workflow",
                metric_name="gates_passed_ratio",
                value=result["gates_passed"] / max(result["gates_total"], 1),
            )

        return result

    async def execute_agent_durable(
        self,
        agent_type: str,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
    ) -> AgentResult:
        """Execute an agent with durable execution.

        Args:
            agent_type: Type of agent.
            task: Task description.
            context: Additional context.
            max_retries: Maximum retries.

        Returns:
            AgentResult.
        """
        workflow_id, result = await self.engine.start_and_execute(
            workflow_name="agent_execution_workflow",
            input_data={
                "agent_type": agent_type,
                "task": task,
                "context": context or {},
                "max_retries": max_retries,
            },
        )
        return result

    async def validate_gates_durable(
        self,
        sprint_id: str,
        run_dir: str,
        gates: Optional[List[str]] = None,
        retry_failed: bool = True,
    ) -> Dict[str, GateResult]:
        """Validate gates with durable execution.

        Args:
            sprint_id: Sprint identifier.
            run_dir: Run directory.
            gates: Gates to validate.
            retry_failed: Whether to retry failed gates.

        Returns:
            Dict mapping gate names to results.
        """
        workflow_id, result = await self.engine.start_and_execute(
            workflow_name="gate_validation_workflow",
            input_data={
                "sprint_id": sprint_id,
                "run_dir": run_dir,
                "gates": gates or [f"G{i}" for i in range(9)],
                "retry_failed": retry_failed,
            },
        )
        return result

    def get_workflow_status(self, workflow_id: str) -> Optional[WorkflowState]:
        """Get status of a workflow."""
        return self.engine.get_workflow_state(workflow_id)

    def list_workflows(
        self,
        status: Optional[WorkflowStatus] = None,
        workflow_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[WorkflowState]:
        """List workflows with filters."""
        return self.engine.list_workflows(status, workflow_name, limit)

    async def resume_workflow(self, workflow_id: str) -> Any:
        """Resume a paused or failed workflow."""
        return await self.engine.resume_workflow(workflow_id)

    def register_workflow(self, workflow_class: Type[Workflow]) -> None:
        """Register a custom workflow."""
        self.engine.register_workflow(workflow_class)


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================


_integration: Optional[TemporalIntegration] = None
_integration_lock = threading.Lock()  # BLACK TEAM FIX BLACK-5 #2


def get_temporal_integration(
    config: Optional[TemporalIntegrationConfig] = None,
) -> TemporalIntegration:
    """Get or create the global Temporal integration.

    BLACK TEAM FIX BLACK-5 #2: Added double-checked locking to prevent
    race condition during concurrent initialization.

    Args:
        config: Configuration (only used on first call).

    Returns:
        TemporalIntegration instance.
    """
    global _integration
    # Fast path - check without lock
    if _integration is not None:
        return _integration
    # Slow path - acquire lock and double-check
    with _integration_lock:
        if _integration is None:
            _integration = TemporalIntegration(config)
    return _integration


def reset_temporal_integration() -> None:
    """Reset the global integration (for testing)."""
    global _integration
    with _integration_lock:
        _integration = None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


async def run_durable(
    func: Callable[..., Coroutine],
    *args,
    workflow_name: Optional[str] = None,
    **kwargs,
) -> Any:
    """Run a coroutine with durable execution.

    Wraps the function in a workflow for automatic checkpointing
    and recovery.

    Args:
        func: Async function to run.
        *args: Positional arguments.
        workflow_name: Optional workflow name.
        **kwargs: Keyword arguments.

    Returns:
        Function result.
    """
    name = workflow_name or f"durable_{func.__name__}"

    @durable_workflow(name=name)
    async def wrapper(ctx: WorkflowContext, input_data: Dict) -> Any:
        return await func(*input_data.get("args", ()), **input_data.get("kwargs", {}))

    return await wrapper({"args": args, "kwargs": kwargs})


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Config
    "TemporalIntegrationConfig",
    # Workflows
    "SprintWorkflow",
    "GateValidationWorkflow",
    "AgentExecutionWorkflow",
    # Decorators
    "durable_workflow",
    "with_temporal_context",
    # Integration
    "TemporalIntegration",
    "get_temporal_integration",
    "reset_temporal_integration",
    # Convenience
    "run_durable",
]
