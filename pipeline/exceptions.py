"""Pipeline v2 Exceptions - INVIOLABLE GUARDRAILS.

Este módulo define exceções que DEVEM ser tratadas. Não há como ignorá-las.
São as "leis da física" do pipeline - violação é estruturalmente impossível.

═══════════════════════════════════════════════════════════════════════════════
HANDOFF vs SIGNOFF - FLUXO BIDIRECIONAL (ZERO AMBIGUIDADE)
═══════════════════════════════════════════════════════════════════════════════

HANDOFF (Delegação - PARA BAIXO na hierarquia):
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  CEO → VPs → Masters → Squad Leads → Workers                            │
    │                                                                          │
    │  Superior diz ao subordinado O QUE FAZER:                               │
    │    - what_i_want: "O QUE EU QUERO que você faça"                        │
    │    - why_i_want: "PRA QUE EU QUERO isso / contexto"                     │
    │    - expected_behavior: "COMO EU ESPERO que você se comporte"           │
    └─────────────────────────────────────────────────────────────────────────┘

SIGNOFF (Aprovação - PARA CIMA na hierarquia):
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  Workers → Squad Leads → Masters → VPs → CEO                            │
    │                                                                          │
    │  Subordinado reporta ao superior O QUE FEZ:                             │
    │    - what_i_did: "O QUE EU FIZ"                                         │
    │    - why_this_way: "POR QUE FIZ ASSIM"                                  │
    │    - how_it_works: "COMO FUNCIONA"                                      │
    │                                                                          │
    │  Superior APROVA ou REJEITA o trabalho do subordinado:                  │
    │    - approved: True/False                                                │
    │    - justification: "Por que aprovei/rejeitei"                          │
    │    - subordinates_approved: ["worker1", "worker2"]                      │
    └─────────────────────────────────────────────────────────────────────────┘

FLUXO COMPLETO DE UM AGENTE (ex: Ace Exec):
    1. RECEBE handoff do Exec VP (seu superior)
    2. PASSA handoff para Squad Leads (seus subordinados)
    3. ESPERA signoffs dos Squad Leads
    4. APROVA/REJEITA trabalho de cada Squad Lead
    5. FAZ seu próprio signoff para o Exec VP
    6. Exec VP então aprova/rejeita o trabalho do Ace Exec

REGRA INVIOLÁVEL:
    - Agente NÃO PODE fazer signoff se subordinados não fizeram signoff
    - Agente NÃO PODE fazer signoff se não APROVOU trabalho dos subordinados
    - Superior NÃO PODE aprovar se subordinado não explicou o trabalho

═══════════════════════════════════════════════════════════════════════════════

PRINCÍPIO FUNDAMENTAL:
- Erros NÃO matam o pipeline inteiro
- Erros REDIRECIONAM para correção localizada
- Agente que errou é NOTIFICADO e CORRIGE
- Se não corrigir → ESCALA para superior
- SEMPRE: Agente APRENDE com o erro (A-MEM)

ESCALAÇÃO GRADUAL:
    Nível 1: Próprio agente corrige (maioria dos casos)
    Nível 2: Superior orienta e agente refaz
    Nível 3: Run Master intervém (drift, padrão repetido)
    Nível 4: Humano (último recurso)

Author: Pipeline Autonomo Team
Version: 2.0.0 (2026-01-04)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# =============================================================================
# BASE EXCEPTION - Todas as exceções do pipeline herdam desta
# =============================================================================


class PipelineError(Exception):
    """Base exception para todos os erros do pipeline.

    INVIOLÁVEL: Esta exceção e suas filhas DEVEM ser tratadas.
    Ignorar resulta em crash do pipeline (comportamento correto).
    """

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        self.message = message
        self.context = context or {}
        self.timestamp = datetime.now(timezone.utc).isoformat()
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        ctx = ", ".join(f"{k}={v}" for k, v in self.context.items())
        if ctx:
            return f"{self.message} [{ctx}]"
        return self.message


# =============================================================================
# GUARDRAIL EXCEPTIONS - Violação de regras invioláveis
# =============================================================================


class GuardrailViolation(PipelineError):
    """Violação de guardrail - LEI DA FÍSICA do pipeline.

    Esta exceção indica que uma regra INVIOLÁVEL foi violada.
    O pipeline DEVE parar. Não há recovery possível sem intervenção.
    """

    def __init__(
        self,
        guardrail_id: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.guardrail_id = guardrail_id
        ctx = context or {}
        ctx["guardrail_id"] = guardrail_id
        super().__init__(f"GUARDRAIL VIOLATION [{guardrail_id}]: {message}", ctx)


# =============================================================================
# SIGNOFF EXCEPTIONS - Hierarquia é lei
# =============================================================================


class SignoffError(PipelineError):
    """Base para erros de signoff."""
    pass


class HierarchyViolation(SignoffError):
    """Tentativa de signoff sem subordinados terem assinado.

    LEI: CEO só assina se VPs assinaram.
         VPs só assinam se Masters assinaram.
         E assim por diante.

    Esta não é uma sugestão - é uma LEI DA FÍSICA do pipeline.
    """

    def __init__(
        self,
        agent_id: str,
        sprint_id: str,
        missing_subordinates: List[str],
    ):
        self.agent_id = agent_id
        self.sprint_id = sprint_id
        self.missing_subordinates = missing_subordinates
        super().__init__(
            f"{agent_id} não pode assinar {sprint_id}: "
            f"subordinados pendentes: {missing_subordinates}",
            {
                "agent_id": agent_id,
                "sprint_id": sprint_id,
                "missing": missing_subordinates,
            },
        )


class MilestoneIncomplete(SignoffError):
    """Tentativa de signoff sem completar milestones.

    LEI: Agente só assina se completou seus milestones.
         Milestones são âncoras de sanidade.
    """

    def __init__(
        self,
        agent_id: str,
        sprint_id: str,
        missing_milestones: List[str],
    ):
        self.agent_id = agent_id
        self.sprint_id = sprint_id
        self.missing_milestones = missing_milestones
        super().__init__(
            f"{agent_id} não pode assinar {sprint_id}: "
            f"milestones pendentes: {missing_milestones}",
            {
                "agent_id": agent_id,
                "sprint_id": sprint_id,
                "missing": missing_milestones,
            },
        )


class QualityCheckFailed(SignoffError):
    """Artefatos não passam na verificação de qualidade.

    LEI: Código com placeholders, arquivos vazios, ou qualidade
         abaixo do threshold não pode ser assinado.
    """

    def __init__(
        self,
        agent_id: str,
        sprint_id: str,
        quality_problems: Dict[str, List[str]],
    ):
        self.agent_id = agent_id
        self.sprint_id = sprint_id
        self.quality_problems = quality_problems
        problem_count = sum(len(v) for v in quality_problems.values())
        super().__init__(
            f"{agent_id} não pode assinar {sprint_id}: "
            f"{problem_count} problemas de qualidade detectados",
            {
                "agent_id": agent_id,
                "sprint_id": sprint_id,
                "problems": quality_problems,
            },
        )


class ExplanationMissing(SignoffError):
    """Trabalho não explicado adequadamente.

    LEI: Agente DEVE explicar O QUE FEZ, POR QUE, e COMO FUNCIONA.
         Sem explicação, não há como verificar se o trabalho está correto.
    """

    def __init__(
        self,
        agent_id: str,
        sprint_id: str,
        missing_explanations: List[str],
    ):
        self.agent_id = agent_id
        self.sprint_id = sprint_id
        self.missing_explanations = missing_explanations
        super().__init__(
            f"{agent_id} não pode assinar {sprint_id}: "
            f"explicações faltando: {missing_explanations}",
            {
                "agent_id": agent_id,
                "sprint_id": sprint_id,
                "missing": missing_explanations,
            },
        )


class SubordinatesNotApproved(SignoffError):
    """Superior tentou fazer signoff sem aprovar subordinados.

    LEI: Superior DEVE aprovar trabalho de TODOS os subordinados
         ANTES de fazer seu próprio signoff.
         Não se pode "pular" a aprovação.
    """

    def __init__(
        self,
        agent_id: str,
        sprint_id: str,
        unapproved_subordinates: List[str],
    ):
        self.agent_id = agent_id
        self.sprint_id = sprint_id
        self.unapproved_subordinates = unapproved_subordinates
        super().__init__(
            f"{agent_id} não pode assinar {sprint_id}: "
            f"subordinados não aprovados: {unapproved_subordinates}",
            {
                "agent_id": agent_id,
                "sprint_id": sprint_id,
                "unapproved": unapproved_subordinates,
            },
        )


class ApprovalRejected(SignoffError):
    """Superior rejeitou o trabalho do subordinado.

    LEI: Trabalho rejeitado NÃO pode avançar.
         Subordinado DEVE corrigir e resubmeter.
    """

    def __init__(
        self,
        approver_id: str,
        subordinate_id: str,
        sprint_id: str,
        rejection_reason: str,
    ):
        self.approver_id = approver_id
        self.subordinate_id = subordinate_id
        self.sprint_id = sprint_id
        self.rejection_reason = rejection_reason
        super().__init__(
            f"{approver_id} REJEITOU trabalho de {subordinate_id} em {sprint_id}: "
            f"{rejection_reason}",
            {
                "approver": approver_id,
                "subordinate": subordinate_id,
                "sprint_id": sprint_id,
                "reason": rejection_reason,
            },
        )


class RedisUnavailable(SignoffError):
    """Redis não disponível para registrar signoff.

    LEI: Signoffs DEVEM ser persistidos. Sem Redis, não há signoff.
    """

    def __init__(self, operation: str):
        self.operation = operation
        super().__init__(
            f"Redis indisponível para operação: {operation}",
            {"operation": operation},
        )


# =============================================================================
# SPRINT EXCEPTIONS
# =============================================================================


class SprintError(PipelineError):
    """Base para erros de sprint."""
    pass


class SprintNotSignedOff(SprintError):
    """Sprint marcado como success mas sem signoff do CEO.

    LEI: Sprint só é success SE E SOMENTE SE tem signoff do CEO.
         Esta é a definição de success. Não há outra.
    """

    def __init__(self, sprint_id: str):
        self.sprint_id = sprint_id
        super().__init__(
            f"Sprint {sprint_id} não pode ser success sem CEO signoff. "
            f"Isso é uma LEI, não uma sugestão.",
            {"sprint_id": sprint_id},
        )


class CircuitBreakerOpen(SprintError):
    """Circuit breaker aberto - pipeline deve parar.

    LEI: Quando circuit breaker abre, pipeline PARA.
         Continuar seria perpetuar falhas.
    """

    def __init__(self, circuit_id: str, failures: int):
        self.circuit_id = circuit_id
        self.failures = failures
        super().__init__(
            f"Circuit breaker '{circuit_id}' aberto após {failures} falhas",
            {"circuit_id": circuit_id, "failures": failures},
        )


# =============================================================================
# HANDOFF EXCEPTIONS
# =============================================================================


class HandoffError(PipelineError):
    """Base para erros de handoff."""
    pass


class InvalidHandoff(HandoffError):
    """Handoff inválido - não segue as regras.

    LEI: Handoff DEVE ter explicação clara do que foi pedido,
         contexto, e critérios de aceitação.
    """

    def __init__(
        self,
        from_agent: str,
        to_agent: str,
        sprint_id: str,
        validation_errors: List[str],
    ):
        self.from_agent = from_agent
        self.to_agent = to_agent
        self.sprint_id = sprint_id
        self.validation_errors = validation_errors
        super().__init__(
            f"Handoff inválido de {from_agent} para {to_agent} em {sprint_id}: "
            f"{len(validation_errors)} erros",
            {
                "from_agent": from_agent,
                "to_agent": to_agent,
                "sprint_id": sprint_id,
                "errors": validation_errors,
            },
        )


# =============================================================================
# ZAP EXCEPTIONS - Zero Ambiguity Protocol é lei
# =============================================================================


class ZAPError(PipelineError):
    """Base para erros de ZAP (Zero Ambiguity Protocol)."""
    pass


class ZAPViolation(ZAPError):
    """Violação do Zero Ambiguity Protocol.

    LEI: Toda comunicação DEVE ser não-ambígua.
         Ambiguidade causa erros. Erros causam desperdício.
    """

    def __init__(
        self,
        source: str,
        ambiguity_type: str,
        details: str,
    ):
        self.source = source
        self.ambiguity_type = ambiguity_type
        self.details = details
        super().__init__(
            f"ZAP VIOLATION em {source}: {ambiguity_type} - {details}",
            {"source": source, "type": ambiguity_type, "details": details},
        )


class ZAPMissingContext(ZAPError):
    """Contexto obrigatório faltando.

    LEI: Toda operação DEVE incluir contexto completo.
         Sem contexto, agentes cometem erros.
    """

    def __init__(self, operation: str, missing_fields: List[str]):
        self.operation = operation
        self.missing_fields = missing_fields
        super().__init__(
            f"ZAP: Operação '{operation}' sem contexto: {missing_fields}",
            {"operation": operation, "missing": missing_fields},
        )


# =============================================================================
# SPEC KIT EXCEPTIONS - Uso do Spec Kit é obrigatório
# =============================================================================


class SpecKitError(PipelineError):
    """Base para erros de Spec Kit."""
    pass


class SpecKitNotLoaded(SpecKitError):
    """Tentativa de executar sprint sem carregar Spec Kit.

    LEI: Context pack DEVE ser carregado ANTES de execução.
         Executar sem spec é garantia de falha (PAT-026).
    """

    def __init__(self, sprint_id: str):
        self.sprint_id = sprint_id
        super().__init__(
            f"PAT-026 GUARD: Sprint {sprint_id} não pode executar sem carregar context pack!",
            {"sprint_id": sprint_id},
        )


class SpecKitInvalid(SpecKitError):
    """Context pack inválido ou corrompido.

    LEI: Specs DEVEM estar bem formados.
         Spec corrompido = sprint abortado.
    """

    def __init__(self, sprint_id: str, validation_errors: List[str]):
        self.sprint_id = sprint_id
        self.validation_errors = validation_errors
        super().__init__(
            f"Spec Kit inválido para {sprint_id}: {len(validation_errors)} erros",
            {"sprint_id": sprint_id, "errors": validation_errors},
        )


# =============================================================================
# STACK EXCEPTIONS - Uso de stacks é obrigatório
# =============================================================================


class StackError(PipelineError):
    """Base para erros de Stack."""
    pass


class StackNotHealthy(StackError):
    """Stack não está healthy e é obrigatório.

    LEI: Stacks críticas DEVEM estar healthy para execução.
         Redis, Langfuse, etc. são obrigatórios.
    """

    def __init__(self, stack_name: str, health_details: Dict[str, Any]):
        self.stack_name = stack_name
        self.health_details = health_details
        super().__init__(
            f"Stack '{stack_name}' não está healthy e é OBRIGATÓRIA",
            {"stack": stack_name, "health": health_details},
        )


class StackNotConsulted(StackError):
    """Operação executada sem consultar stack obrigatória.

    LEI: Certas operações EXIGEM consulta a stacks específicas.
         Pular consulta = violar protocolo.
    """

    def __init__(self, operation: str, required_stack: str):
        self.operation = operation
        self.required_stack = required_stack
        super().__init__(
            f"Operação '{operation}' executada sem consultar {required_stack}",
            {"operation": operation, "required_stack": required_stack},
        )


# =============================================================================
# MILESTONE EXCEPTIONS - Milestones são âncoras de sanidade
# =============================================================================


class MilestoneError(PipelineError):
    """Base para erros de Milestone."""
    pass


class MilestoneNotRecorded(MilestoneError):
    """Milestone não foi registrado.

    LEI: Milestones DEVEM ser registrados quando atingidos.
         Sem registro, não há prova de progresso.
    """

    def __init__(self, agent_id: str, milestone_id: str):
        self.agent_id = agent_id
        self.milestone_id = milestone_id
        super().__init__(
            f"Milestone '{milestone_id}' de {agent_id} não foi registrado",
            {"agent_id": agent_id, "milestone": milestone_id},
        )


class MilestoneSkipped(MilestoneError):
    """Tentativa de pular milestone obrigatório.

    LEI: Milestones não podem ser pulados.
         Cada milestone é um checkpoint de sanidade.
    """

    def __init__(self, agent_id: str, skipped_milestone: str, reason: str):
        self.agent_id = agent_id
        self.skipped_milestone = skipped_milestone
        self.reason = reason
        super().__init__(
            f"{agent_id} tentou pular milestone '{skipped_milestone}': {reason}",
            {"agent_id": agent_id, "milestone": skipped_milestone, "reason": reason},
        )


# =============================================================================
# GATE EXECUTION EXCEPTIONS - Ordem de gates é lei
# =============================================================================


class GateError(PipelineError):
    """Base para erros de Gate."""
    pass


class GateOrderViolation(GateError):
    """Tentativa de executar gate fora de ordem.

    LEI: Gates DEVEM ser executados na ordem do DAG.
         G0 antes de G1, G1 antes de G2, etc.
         Dependências DEVEM ser respeitadas.
    """

    def __init__(
        self,
        gate_id: str,
        missing_dependencies: List[str],
    ):
        self.gate_id = gate_id
        self.missing_dependencies = missing_dependencies
        super().__init__(
            f"Gate '{gate_id}' não pode executar: "
            f"dependências não completadas: {missing_dependencies}",
            {"gate_id": gate_id, "missing": missing_dependencies},
        )


class GatePreconditionFailed(GateError):
    """Pré-condições do gate não satisfeitas.

    LEI: Cada gate tem pré-condições que DEVEM ser satisfeitas.
         Sem pré-condições = sem execução.
    """

    def __init__(
        self,
        gate_id: str,
        failed_preconditions: List[str],
    ):
        self.gate_id = gate_id
        self.failed_preconditions = failed_preconditions
        super().__init__(
            f"Gate '{gate_id}' não pode executar: "
            f"pré-condições falharam: {failed_preconditions}",
            {"gate_id": gate_id, "failed": failed_preconditions},
        )


# =============================================================================
# AGENT RESPONSIBILITY EXCEPTIONS - Cada agente tem seu papel
# =============================================================================


class AgentResponsibilityError(PipelineError):
    """Base para erros de responsabilidade de agente."""
    pass


class AgentNotAuthorized(AgentResponsibilityError):
    """Agente tentou fazer algo fora de sua responsabilidade.

    LEI: Cada agente tem responsabilidades ESPECÍFICAS.
         spec_master cuida de specs, ace_exec cuida de código.
         Cruzar responsabilidades = caos.
    """

    def __init__(
        self,
        agent_id: str,
        attempted_action: str,
        authorized_agents: List[str],
    ):
        self.agent_id = agent_id
        self.attempted_action = attempted_action
        self.authorized_agents = authorized_agents
        super().__init__(
            f"Agente '{agent_id}' não autorizado para '{attempted_action}'. "
            f"Agentes autorizados: {authorized_agents}",
            {
                "agent_id": agent_id,
                "action": attempted_action,
                "authorized": authorized_agents,
            },
        )


class AgentLevelViolation(AgentResponsibilityError):
    """Agente tentou comandar alguém acima de seu nível.

    LEI: Hierarquia é lei.
         L5 não comanda L3.
         L4 não comanda L2.
         Cada um cuida do seu nível.
    """

    def __init__(
        self,
        agent_id: str,
        agent_level: int,
        target_id: str,
        target_level: int,
    ):
        self.agent_id = agent_id
        self.agent_level = agent_level
        self.target_id = target_id
        self.target_level = target_level
        super().__init__(
            f"Agente '{agent_id}' (L{agent_level}) não pode comandar "
            f"'{target_id}' (L{target_level}). Hierarquia violada.",
            {
                "agent_id": agent_id,
                "agent_level": agent_level,
                "target_id": target_id,
                "target_level": target_level,
            },
        )


class ResponsibilityNotFulfilled(AgentResponsibilityError):
    """Agente não cumpriu suas responsabilidades obrigatórias.

    LEI: Cada agente TEM responsabilidades que DEVE cumprir.
         Não cumprir = não pode fazer signoff.
    """

    def __init__(
        self,
        agent_id: str,
        missing_responsibilities: List[str],
    ):
        self.agent_id = agent_id
        self.missing_responsibilities = missing_responsibilities
        super().__init__(
            f"Agente '{agent_id}' não cumpriu responsabilidades: "
            f"{missing_responsibilities}",
            {"agent_id": agent_id, "missing": missing_responsibilities},
        )


# =============================================================================
# SELF-IMPROVEMENT EXCEPTIONS - Auto-evolução é obrigatória
# =============================================================================


class SelfImprovementError(PipelineError):
    """Base para erros de Self-Improvement."""
    pass


class ReflexionNotExecuted(SelfImprovementError):
    """Reflexão não foi executada após falha.

    LEI: Toda falha DEVE gerar reflexão.
         Sem reflexão, erros se repetem.
    """

    def __init__(self, gate_id: str, failure_type: str):
        self.gate_id = gate_id
        self.failure_type = failure_type
        super().__init__(
            f"Gate '{gate_id}' falhou ({failure_type}) mas reflexão não foi executada",
            {"gate_id": gate_id, "failure_type": failure_type},
        )


class LearningNotSaved(SelfImprovementError):
    """Aprendizado não foi salvo no A-MEM.

    LEI: Todo aprendizado DEVE ser persistido.
         Sem persistência, conhecimento é perdido.
    """

    def __init__(self, learning_type: str, context: str):
        self.learning_type = learning_type
        self.context = context
        super().__init__(
            f"Aprendizado '{learning_type}' não foi salvo: {context}",
            {"type": learning_type, "context": context},
        )


class PatternNotCataloged(SelfImprovementError):
    """Padrão detectado não foi catalogado.

    LEI: Padrões (erros ou sucessos) DEVEM ser catalogados.
         Sem catálogo, história se repete.
    """

    def __init__(self, pattern_id: str, pattern_type: str):
        self.pattern_id = pattern_id
        self.pattern_type = pattern_type
        super().__init__(
            f"Padrão '{pattern_id}' ({pattern_type}) não foi catalogado",
            {"pattern_id": pattern_id, "type": pattern_type},
        )


# =============================================================================
# RESULT TYPE - Para operações que DEVEM ser verificadas
# =============================================================================


@dataclass(frozen=True)
class SignoffResult:
    """Resultado de operação de signoff - DEVE ser verificado.

    Este tipo força o chamador a verificar o resultado.
    Não é possível ignorá-lo sem erro de tipo/lógica.

    Usage:
        result = record_signoff(...)
        if not result.success:
            raise result.error  # DEVE tratar o erro
        # Só aqui pode prosseguir
    """

    success: bool
    sprint_id: str
    agent_id: str
    error: Optional[SignoffError] = None
    signoff_data: Optional[Dict[str, Any]] = None

    def unwrap(self) -> Dict[str, Any]:
        """Retorna dados do signoff ou levanta exceção.

        Similar ao unwrap() do Rust - force o tratamento do erro.
        """
        if not self.success:
            if self.error:
                raise self.error
            raise SignoffError(f"Signoff falhou para {self.agent_id} em {self.sprint_id}")
        return self.signoff_data or {}

    def expect(self, message: str) -> Dict[str, Any]:
        """Retorna dados ou levanta exceção com mensagem customizada."""
        if not self.success:
            if self.error:
                raise type(self.error)(f"{message}: {self.error.message}")
            raise SignoffError(f"{message}: signoff falhou")
        return self.signoff_data or {}


@dataclass(frozen=True)
class HandoffResult:
    """Resultado de handoff - DEVE ser verificado."""

    success: bool
    from_agent: str
    to_agent: str
    sprint_id: str
    error: Optional[HandoffError] = None
    handoff_data: Optional[Dict[str, Any]] = None

    def unwrap(self) -> Dict[str, Any]:
        if not self.success:
            if self.error:
                raise self.error
            raise HandoffError(f"Handoff falhou de {self.from_agent} para {self.to_agent}")
        return self.handoff_data or {}


@dataclass(frozen=True)
class ZAPResult:
    """Resultado de verificação ZAP - DEVE ser verificado."""

    success: bool
    source: str
    error: Optional[ZAPError] = None
    validated_data: Optional[Dict[str, Any]] = None

    def unwrap(self) -> Dict[str, Any]:
        if not self.success:
            if self.error:
                raise self.error
            raise ZAPError(f"ZAP validation falhou em {self.source}")
        return self.validated_data or {}


@dataclass(frozen=True)
class SpecKitResult:
    """Resultado de carregamento de Spec Kit - DEVE ser verificado."""

    success: bool
    sprint_id: str
    error: Optional[SpecKitError] = None
    spec_data: Optional[Dict[str, Any]] = None

    def unwrap(self) -> Dict[str, Any]:
        if not self.success:
            if self.error:
                raise self.error
            raise SpecKitError(f"Spec Kit não carregado para {self.sprint_id}")
        return self.spec_data or {}


@dataclass(frozen=True)
class StackResult:
    """Resultado de verificação de Stack - DEVE ser verificado."""

    success: bool
    stack_name: str
    error: Optional[StackError] = None
    health_data: Optional[Dict[str, Any]] = None

    def unwrap(self) -> Dict[str, Any]:
        if not self.success:
            if self.error:
                raise self.error
            raise StackError(f"Stack '{self.stack_name}' não está healthy")
        return self.health_data or {}


@dataclass(frozen=True)
class MilestoneResult:
    """Resultado de operação de Milestone - DEVE ser verificado."""

    success: bool
    agent_id: str
    milestone_id: str
    error: Optional[MilestoneError] = None
    milestone_data: Optional[Dict[str, Any]] = None

    def unwrap(self) -> Dict[str, Any]:
        if not self.success:
            if self.error:
                raise self.error
            raise MilestoneError(f"Milestone '{self.milestone_id}' falhou para {self.agent_id}")
        return self.milestone_data or {}


@dataclass(frozen=True)
class ReflexionResult:
    """Resultado de reflexão - DEVE ser verificado."""

    success: bool
    gate_id: str
    error: Optional[SelfImprovementError] = None
    reflexion_data: Optional[Dict[str, Any]] = None

    def unwrap(self) -> Dict[str, Any]:
        if not self.success:
            if self.error:
                raise self.error
            raise SelfImprovementError(f"Reflexão não executada para {self.gate_id}")
        return self.reflexion_data or {}


@dataclass(frozen=True)
class ApprovalResult:
    """Resultado de aprovação de subordinado - DEVE ser verificado.

    Superior usa isso para aprovar/rejeitar trabalho do subordinado.
    """

    success: bool
    approver_id: str
    subordinate_id: str
    sprint_id: str
    approved: bool  # True = aprovado, False = rejeitado
    error: Optional[SignoffError] = None
    approval_data: Optional[Dict[str, Any]] = None

    def unwrap(self) -> Dict[str, Any]:
        if not self.success:
            if self.error:
                raise self.error
            raise SignoffError(
                f"Aprovação falhou: {self.approver_id} → {self.subordinate_id}"
            )
        return self.approval_data or {}

    def is_approved(self) -> bool:
        """Verifica se foi aprovado (não apenas se operação teve sucesso)."""
        return self.success and self.approved


@dataclass(frozen=True)
class GateResult:
    """Resultado de execução de Gate - DEVE ser verificado."""

    success: bool
    gate_id: str
    sprint_id: str
    error: Optional[GateError] = None
    gate_data: Optional[Dict[str, Any]] = None
    dependencies_met: bool = True

    def unwrap(self) -> Dict[str, Any]:
        if not self.success:
            if self.error:
                raise self.error
            raise GateError(f"Gate '{self.gate_id}' falhou em {self.sprint_id}")
        return self.gate_data or {}


@dataclass(frozen=True)
class AgentActionResult:
    """Resultado de ação de agente - DEVE ser verificado.

    Verifica se agente está autorizado a fazer a ação.
    """

    success: bool
    agent_id: str
    action: str
    error: Optional[AgentResponsibilityError] = None
    action_data: Optional[Dict[str, Any]] = None

    def unwrap(self) -> Dict[str, Any]:
        if not self.success:
            if self.error:
                raise self.error
            raise AgentResponsibilityError(
                f"Agente '{self.agent_id}' não autorizado para '{self.action}'"
            )
        return self.action_data or {}


@dataclass(frozen=True)
class StackHealthResult:
    """Resultado de verificação de saúde de TODAS as stacks."""

    success: bool
    all_healthy: bool
    unhealthy_stacks: List[str] = field(default_factory=list)
    stack_details: Optional[Dict[str, Any]] = None
    error: Optional[StackError] = None

    def unwrap(self) -> Dict[str, Any]:
        if not self.success or not self.all_healthy:
            if self.error:
                raise self.error
            raise StackNotHealthy(
                f"Stacks não healthy: {self.unhealthy_stacks}",
                {"unhealthy": self.unhealthy_stacks},
            )
        return self.stack_details or {}


# =============================================================================
# CORRECTION SYSTEM - Redireciona para correção em vez de matar pipeline
# =============================================================================


class EscalationLevel:
    """Níveis de escalação."""
    SELF = 1       # Próprio agente corrige
    SUPERIOR = 2   # Superior orienta
    RUN_MASTER = 3 # Run Master intervém
    HUMAN = 4      # Humano (último recurso)


@dataclass
class CorrectionRequest:
    """Pedido de correção para um agente.

    Em vez de matar o pipeline, redireciona para o agente corrigir.
    Se não corrigir, escala gradualmente.

    Usage:
        result = do_something()
        if result.needs_correction:
            # Notifica agente, não mata pipeline
            correction = result.correction
            notify_agent(correction.agent_id, correction.message)
            # Agente corrige e resubmete
    """

    agent_id: str
    sprint_id: str
    error_type: str
    message: str
    what_went_wrong: str
    how_to_fix: str
    evidence: Dict[str, Any]
    escalation_level: int = EscalationLevel.SELF
    attempts: int = 0
    max_attempts: int = 3
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def escalate(self) -> "CorrectionRequest":
        """Escala para próximo nível se não corrigiu."""
        return CorrectionRequest(
            agent_id=self.agent_id,
            sprint_id=self.sprint_id,
            error_type=self.error_type,
            message=self.message,
            what_went_wrong=self.what_went_wrong,
            how_to_fix=self.how_to_fix,
            evidence=self.evidence,
            escalation_level=min(self.escalation_level + 1, EscalationLevel.HUMAN),
            attempts=self.attempts + 1,
            max_attempts=self.max_attempts,
        )

    def should_escalate(self) -> bool:
        """Verifica se deve escalar (tentativas esgotadas neste nível)."""
        return self.attempts >= self.max_attempts

    def to_learning(self) -> Dict[str, Any]:
        """Converte para formato de aprendizado (A-MEM)."""
        return {
            "type": "error_correction",
            "agent_id": self.agent_id,
            "error_type": self.error_type,
            "what_went_wrong": self.what_went_wrong,
            "how_to_fix": self.how_to_fix,
            "escalation_level": self.escalation_level,
            "attempts": self.attempts,
            "timestamp": self.created_at,
        }


@dataclass
class CorrectionResult:
    """Resultado que pode precisar de correção (não mata pipeline).

    Se success=True: tudo ok, continua
    Se success=False: precisa correção, mas NÃO explode
    """

    success: bool
    agent_id: str
    sprint_id: str
    correction: Optional[CorrectionRequest] = None
    data: Optional[Dict[str, Any]] = None

    @property
    def needs_correction(self) -> bool:
        return not self.success and self.correction is not None

    def unwrap_or_correct(self) -> Dict[str, Any]:
        """Retorna dados OU retorna a correção necessária (não explode)."""
        if self.success:
            return self.data or {}
        return {"needs_correction": True, "correction": self.correction}

    @staticmethod
    def ok(agent_id: str, sprint_id: str, data: Dict[str, Any]) -> "CorrectionResult":
        """Factory para resultado ok."""
        return CorrectionResult(
            success=True,
            agent_id=agent_id,
            sprint_id=sprint_id,
            data=data,
        )

    @staticmethod
    def needs_fix(
        agent_id: str,
        sprint_id: str,
        error_type: str,
        what_went_wrong: str,
        how_to_fix: str,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> "CorrectionResult":
        """Factory para resultado que precisa correção."""
        return CorrectionResult(
            success=False,
            agent_id=agent_id,
            sprint_id=sprint_id,
            correction=CorrectionRequest(
                agent_id=agent_id,
                sprint_id=sprint_id,
                error_type=error_type,
                message=f"{error_type}: {what_went_wrong}",
                what_went_wrong=what_went_wrong,
                how_to_fix=how_to_fix,
                evidence=evidence or {},
            ),
        )


# =============================================================================
# CONTEXT MANAGER - Para operações atômicas
# =============================================================================


class AtomicSignoff:
    """Context manager para signoff atômico.

    Garante que ou TODOS os signoffs são registrados,
    ou NENHUM é. Não há estado intermediário.

    Usage:
        with AtomicSignoff(pipeline, sprint_id) as signoff:
            signoff.add("spec_master", justification="...")
            signoff.add("ace_exec", justification="...")
            signoff.add("qa_master", justification="...")
        # Só aqui todos foram registrados
    """

    def __init__(self, pipeline: Any, sprint_id: str):
        self.pipeline = pipeline
        self.sprint_id = sprint_id
        self.pending_signoffs: List[Dict[str, Any]] = []
        self.committed = False

    def __enter__(self) -> "AtomicSignoff":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None:
            # Rollback - não commita nada
            self.pending_signoffs.clear()
            return False

        # Commit all signoffs
        for signoff in self.pending_signoffs:
            result = self.pipeline.record_signoff(**signoff)
            result.unwrap()  # Levanta exceção se falhar

        self.committed = True
        return False

    def add(
        self,
        agent_id: str,
        approved: bool = True,
        justification: str = "",
        **kwargs,
    ) -> None:
        """Adiciona signoff ao batch (não persiste ainda)."""
        self.pending_signoffs.append({
            "sprint_id": self.sprint_id,
            "agent_id": agent_id,
            "approved": approved,
            "justification": justification,
            **kwargs,
        })
