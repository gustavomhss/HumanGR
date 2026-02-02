"""
Task-level models for surgical rework architecture.

This module defines the core data structures for task-level
pipeline execution with surgical rework capabilities.

Key classes:
- Task: Atomic unit of work
- TaskStatus: State machine for task lifecycle
- Diagnosis: Precise failure diagnosis
- Patch: Surgical code correction
- ValidationResult: Validation outcome

Created: 2026-01-29
Reference: .claude/plans/SURGICAL_REWORK_ARCHITECTURE_STUDY.md
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass  # Para imports circulares futuros


# ===========================================================================
# EXCEPTIONS
# ===========================================================================


class InvalidStateTransition(Exception):
    """Raised when an invalid state transition is attempted."""
    pass


class PatchApplicationError(Exception):
    """Raised when a patch cannot be applied."""
    pass


class CyclicDependencyError(Exception):
    """Raised when a cyclic dependency is detected."""
    pass


# ===========================================================================
# ENUMS
# ===========================================================================


class TaskStatus(Enum):
    """
    Estados possíveis de uma task no ciclo de vida.

    State Machine:
        PENDING ──────► BLOCKED (se tem dependências não resolvidas)
            │
            ▼
        EXECUTING ────► VALIDATING
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
            PASSED                   FAILED
                                        │
                                        ▼
                                    REPAIRING
                                        │
                                        ▼
                                    VALIDATING (loop)
                                        │
                                        ▼
                                    ESCALATED (se max_attempts)
    """

    PENDING = "pending"
    BLOCKED = "blocked"
    EXECUTING = "executing"
    VALIDATING = "validating"
    PASSED = "passed"
    FAILED = "failed"
    REPAIRING = "repairing"
    ESCALATED = "escalated"

    def is_terminal(self) -> bool:
        """Retorna True se é um estado final (não pode transicionar)."""
        return self in (TaskStatus.PASSED, TaskStatus.ESCALATED)

    def can_transition_to(self, target: "TaskStatus") -> bool:
        """
        Valida se transição para o estado alvo é permitida.

        Args:
            target: Estado alvo da transição

        Returns:
            True se a transição é válida, False caso contrário
        """
        valid_transitions: Dict[TaskStatus, set] = {
            TaskStatus.PENDING: {TaskStatus.BLOCKED, TaskStatus.EXECUTING},
            TaskStatus.BLOCKED: {TaskStatus.PENDING, TaskStatus.EXECUTING},
            TaskStatus.EXECUTING: {TaskStatus.VALIDATING},
            TaskStatus.VALIDATING: {TaskStatus.PASSED, TaskStatus.FAILED},
            TaskStatus.FAILED: {TaskStatus.REPAIRING, TaskStatus.ESCALATED},
            TaskStatus.REPAIRING: {TaskStatus.VALIDATING},
            TaskStatus.PASSED: set(),  # Terminal - sem transições
            TaskStatus.ESCALATED: set(),  # Terminal - sem transições
        }
        return target in valid_transitions.get(self, set())


class FailureType(Enum):
    """
    Tipos de falha para classificação e roteamento de rework.

    A classificação determina a estratégia de rework:
    - CODE: Agentes podem corrigir via patch cirúrgico
    - TIMEOUT: Problema de infraestrutura - aumentar timeout
    - INFRASTRUCTURE: Requer intervenção de ops
    - SECURITY: Requer revisão humana obrigatória
    - INTEGRATION: Conflito entre tasks
    - PERFORMANCE: Agentes podem otimizar
    """

    CODE = "code"
    TIMEOUT = "timeout"
    INFRASTRUCTURE = "infrastructure"
    SECURITY = "security"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"

    def can_auto_repair(self) -> bool:
        """Retorna True se pode ser reparado automaticamente por agentes."""
        return self in (FailureType.CODE, FailureType.PERFORMANCE)

    def requires_human(self) -> bool:
        """Retorna True se requer intervenção humana obrigatória."""
        return self == FailureType.SECURITY

    def requires_ops(self) -> bool:
        """Retorna True se requer intervenção de operações."""
        return self in (FailureType.TIMEOUT, FailureType.INFRASTRUCTURE)


class ChangeType(Enum):
    """
    Tipo de mudança em um patch cirúrgico.

    - ADD: Adicionar linhas novas (inserção)
    - REMOVE: Remover linhas existentes (deleção)
    - REPLACE: Substituir linhas (modificação)
    """

    ADD = "add"
    REMOVE = "remove"
    REPLACE = "replace"


# ===========================================================================
# DATACLASSES
# ===========================================================================


@dataclass
class Task:
    """
    Unidade atômica de trabalho no pipeline.

    Uma Task representa uma única unidade de geração de código
    que pode ser executada, validada e reparada independentemente.

    Lifecycle:
        1. Criada com status PENDING
        2. Transiciona para EXECUTING quando inicia
        3. Transiciona para VALIDATING após execução
        4. Se passar: PASSED (terminal)
        5. Se falhar: FAILED -> REPAIRING -> VALIDATING (loop)
        6. Se exceder max_attempts: ESCALATED (terminal)

    Attributes:
        id: Identificador único no formato "S00-T01"
        sprint_id: ID da sprint pai (ex: "S00")
        sequence: Ordem de execução dentro da sprint (1, 2, 3...)
        name: Nome descritivo da task
        description: Descrição detalhada do que fazer
        deliverables: Lista de arquivos a serem gerados
        depends_on: IDs de tasks das quais esta depende
        status: Estado atual no ciclo de vida
        attempts: Número de tentativas de execução
        max_attempts: Limite máximo de tentativas (default: 3)
    """

    # === IDENTIFICAÇÃO ===
    id: str
    sprint_id: str
    sequence: int

    # === DEFINIÇÃO ===
    name: str
    description: str
    deliverables: List[str] = field(default_factory=list)
    context_packs: List[str] = field(default_factory=list)

    # === DEPENDÊNCIAS ===
    depends_on: List[str] = field(default_factory=list)

    # === ESTADO ===
    status: TaskStatus = TaskStatus.PENDING
    attempts: int = 0
    max_attempts: int = 3

    # === EXECUÇÃO ===
    code_generated: Optional[str] = None
    validation_result: Optional["ValidationResult"] = None
    diagnosis: Optional["Diagnosis"] = None

    # === MÉTRICAS ===
    confidence_score: float = 0.0
    execution_time_ms: int = 0
    tokens_used: int = 0

    # === HISTÓRICO ===
    repair_history: List[Dict[str, Any]] = field(default_factory=list)
    checkpoint_path: Optional[str] = None

    # === TIMESTAMPS ===
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def transition_to(self, new_status: TaskStatus) -> None:
        """
        Transiciona para novo estado com validação.

        Args:
            new_status: Estado alvo da transição

        Raises:
            InvalidStateTransition: Se a transição não é permitida

        Side Effects:
            - Atualiza started_at quando transiciona para EXECUTING
            - Incrementa attempts quando transiciona para EXECUTING
            - Atualiza completed_at quando atinge estado terminal
        """
        if not self.status.can_transition_to(new_status):
            raise InvalidStateTransition(
                f"Cannot transition Task {self.id} from {self.status.value} "
                f"to {new_status.value}"
            )

        self.status = new_status

        # Side effects
        if new_status == TaskStatus.EXECUTING:
            self.started_at = datetime.now(timezone.utc)
            self.attempts += 1

        if new_status.is_terminal():
            self.completed_at = datetime.now(timezone.utc)

    def calculate_confidence(self) -> float:
        """
        Calcula score de confiança baseado no histórico de rework.

        Fórmula:
            - Passou de primeira (0 reworks): 1.0
            - Passou após 1 rework: 0.7
            - Passou após 2 reworks: 0.4
            - Passou após 3+ reworks: 0.2
            - Escalado ou não passou: 0.0

        Returns:
            Score de confiança entre 0.0 e 1.0
        """
        if self.status == TaskStatus.ESCALATED:
            return 0.0

        if self.status != TaskStatus.PASSED:
            return 0.0

        rework_count = len(self.repair_history)

        if rework_count == 0:
            return 1.0
        elif rework_count == 1:
            return 0.7
        elif rework_count == 2:
            return 0.4
        else:
            return 0.2

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializa a task para dicionário.

        Útil para:
            - Salvar em checkpoint
            - Enviar para state do LangGraph
            - Logging e debugging

        Returns:
            Dicionário com todos os campos serializáveis
        """
        return {
            "id": self.id,
            "sprint_id": self.sprint_id,
            "sequence": self.sequence,
            "name": self.name,
            "description": self.description,
            "deliverables": self.deliverables,
            "depends_on": self.depends_on,
            "status": self.status.value,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "confidence_score": self.confidence_score,
            "execution_time_ms": self.execution_time_ms,
            "tokens_used": self.tokens_used,
            "repair_history_count": len(self.repair_history),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """
        Cria Task a partir de dicionário.

        Args:
            data: Dicionário com campos da task

        Returns:
            Nova instância de Task

        Raises:
            KeyError: Se campos obrigatórios faltam
        """
        return cls(
            id=data["id"],
            sprint_id=data["sprint_id"],
            sequence=data["sequence"],
            name=data["name"],
            description=data["description"],
            deliverables=data.get("deliverables", []),
            depends_on=data.get("depends_on", []),
            status=TaskStatus(data.get("status", "pending")),
            attempts=data.get("attempts", 0),
            max_attempts=data.get("max_attempts", 3),
            confidence_score=data.get("confidence_score", 0.0),
            execution_time_ms=data.get("execution_time_ms", 0),
            tokens_used=data.get("tokens_used", 0),
            repair_history=data.get("repair_history", []),
            checkpoint_path=data.get("checkpoint_path"),
        )


@dataclass
class Diagnosis:
    """
    Diagnóstico preciso de uma falha.

    Um diagnóstico contém toda informação necessária para
    gerar um patch cirúrgico que corrija o problema.

    Níveis de Precisão (do mais genérico ao mais específico):
        1. task_id: Qual task falhou
        2. file_path: Qual arquivo tem o problema
        3. function_name: Qual função/classe
        4. line_number: Qual linha específica
        5. root_cause: Por que falhou (análise)
        6. suggested_fix: Como corrigir (sugestão)

    O objetivo é sempre chegar no nível mais específico possível
    para permitir correção cirúrgica.

    Attributes:
        id: UUID único do diagnóstico
        task_id: ID da task que falhou
        file_path: Caminho do arquivo problemático
        line_number: Linha do problema (se identificada)
        function_name: Nome da função problemática
        failure_type: Classificação do tipo de falha
        error_message: Mensagem de erro original
        root_cause: Análise da causa raiz
        suggested_fix: Sugestão de correção
        fix_confidence: Confiança na sugestão (0.0-1.0)
    """

    # === IDENTIFICAÇÃO ===
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""

    # === LOCALIZAÇÃO ===
    file_path: str = ""
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    function_name: Optional[str] = None
    class_name: Optional[str] = None

    # === CLASSIFICAÇÃO ===
    failure_type: FailureType = FailureType.CODE
    severity: str = "medium"  # low, medium, high, critical

    # === DESCRIÇÃO ===
    error_message: str = ""
    root_cause: str = ""
    expected_behavior: str = ""
    actual_behavior: str = ""

    # === CONTEXTO ===
    code_snippet: str = ""
    stack_trace: Optional[str] = None
    related_files: List[str] = field(default_factory=list)

    # === SUGESTÃO ===
    suggested_fix: str = ""
    fix_confidence: float = 0.0
    alternative_fixes: List[str] = field(default_factory=list)

    # === METADADOS ===
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    diagnosis_time_ms: int = 0

    def to_prompt_context(self) -> str:
        """
        Formata diagnóstico para uso em prompt de LLM.

        Returns:
            String formatada em Markdown
        """
        return f"""## Diagnóstico de Falha

**Task**: {self.task_id}
**Arquivo**: {self.file_path}
**Linha**: {self.line_number or 'N/A'}
**Função**: {self.function_name or 'N/A'}
**Tipo de Falha**: {self.failure_type.value}
**Severidade**: {self.severity}

### Código Problemático
```python
{self.code_snippet or 'N/A'}
```

### Erro
{self.error_message or 'N/A'}

### Causa Raiz
{self.root_cause or 'N/A'}

### Comportamento Esperado
{self.expected_behavior or 'N/A'}

### Comportamento Atual
{self.actual_behavior or 'N/A'}

### Sugestão de Correção (Confiança: {self.fix_confidence:.0%})
{self.suggested_fix or 'N/A'}
"""


@dataclass
class Patch:
    """
    Correção cirúrgica a ser aplicada em um arquivo.

    Um Patch representa a MENOR mudança possível que
    corrige um problema específico identificado no Diagnosis.

    Princípios:
        1. Afeta o MÍNIMO de linhas possível
        2. PRESERVA código funcional adjacente
        3. É REVERSÍVEL (pode desfazer)
        4. É VERIFICÁVEL (pode validar aplicação)

    Tipos de Mudança:
        - ADD: Inserir linhas novas entre linhas existentes
        - REMOVE: Deletar linhas existentes
        - REPLACE: Substituir linhas existentes por novas

    Attributes:
        id: UUID único do patch
        task_id: ID da task sendo corrigida
        diagnosis_id: ID do diagnóstico que originou o patch
        file_path: Caminho do arquivo a modificar
        line_start: Primeira linha afetada (1-indexed)
        line_end: Última linha afetada (1-indexed)
        change_type: Tipo de mudança (ADD, REMOVE, REPLACE)
        old_content: Conteúdo original (para REPLACE/REMOVE)
        new_content: Novo conteúdo
        explanation: Explicação da mudança
        confidence: Confiança de que o patch está correto
    """

    # === IDENTIFICAÇÃO ===
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    diagnosis_id: str = ""

    # === LOCALIZAÇÃO ===
    file_path: str = ""
    line_start: int = 0
    line_end: int = 0

    # === MUDANÇA ===
    change_type: ChangeType = ChangeType.REPLACE
    old_content: Optional[str] = None
    new_content: str = ""

    # === CONTEXTO ===
    context_before: str = ""  # 3 linhas antes para referência
    context_after: str = ""   # 3 linhas depois para referência

    # === EXPLICAÇÃO ===
    explanation: str = ""
    confidence: float = 0.0

    # === METADADOS ===
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    applied_at: Optional[datetime] = None
    verified_at: Optional[datetime] = None

    def apply_to(self, content: str) -> str:
        """
        Aplica o patch ao conteúdo de um arquivo.

        Args:
            content: Conteúdo original do arquivo

        Returns:
            Conteúdo com o patch aplicado

        Raises:
            PatchApplicationError: Se o patch não pode ser aplicado
        """
        lines = content.split('\n')

        # Validar limites
        if self.line_start < 1 or self.line_end > len(lines):
            raise PatchApplicationError(
                f"Line range {self.line_start}-{self.line_end} out of bounds "
                f"(file has {len(lines)} lines)"
            )

        # Converter para 0-indexed
        start_idx = self.line_start - 1
        end_idx = self.line_end  # Exclusive

        if self.change_type == ChangeType.ADD:
            # Inserir novas linhas após line_start
            new_lines = self.new_content.split('\n') if self.new_content else []
            result = lines[:self.line_start] + new_lines + lines[self.line_start:]

        elif self.change_type == ChangeType.REMOVE:
            # Remover linhas de line_start até line_end (inclusive)
            result = lines[:start_idx] + lines[end_idx:]

        elif self.change_type == ChangeType.REPLACE:
            # Substituir linhas de line_start até line_end
            new_lines = self.new_content.split('\n') if self.new_content else []
            result = lines[:start_idx] + new_lines + lines[end_idx:]

        else:
            raise PatchApplicationError(f"Unknown change type: {self.change_type}")

        # Marcar como aplicado
        self.applied_at = datetime.now(timezone.utc)

        return '\n'.join(result)

    def to_unified_diff(self) -> str:
        """
        Gera representação do patch no formato unified diff.

        Returns:
            String no formato unified diff
        """
        old_lines = self.old_content.split('\n') if self.old_content else []
        new_lines = self.new_content.split('\n') if self.new_content else []

        diff_lines = [
            f"--- a/{self.file_path}",
            f"+++ b/{self.file_path}",
            f"@@ -{self.line_start},{len(old_lines)} +{self.line_start},{len(new_lines)} @@",
        ]

        for line in old_lines:
            diff_lines.append(f"-{line}")

        for line in new_lines:
            diff_lines.append(f"+{line}")

        return '\n'.join(diff_lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serializa o patch para dicionário."""
        return {
            "id": self.id,
            "task_id": self.task_id,
            "diagnosis_id": self.diagnosis_id,
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "change_type": self.change_type.value,
            "old_content": self.old_content,
            "new_content": self.new_content,
            "explanation": self.explanation,
            "confidence": self.confidence,
        }


@dataclass
class GateResult:
    """
    Resultado de execução de um gate individual.

    Attributes:
        gate_id: ID do gate executado
        passed: Se passou no gate
        exit_code: Código de saída (0 = sucesso)
        output: Saída padrão do gate
        error: Saída de erro do gate
        execution_time_ms: Tempo de execução em ms
    """

    gate_id: str = ""
    passed: bool = False
    exit_code: int = 0
    output: str = ""
    error: str = ""
    execution_time_ms: int = 0


@dataclass
class QAWorkerResult:
    """
    Resultado de verificação de um QA worker.

    Attributes:
        worker_id: ID do worker que executou a verificação
        passed: Se passou na verificação
        findings: Lista de problemas encontrados
        suggestions: Lista de sugestões de melhoria
    """

    worker_id: str = ""
    passed: bool = False
    findings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """
    Resultado de validação de uma task.

    Contém os resultados de todos os gates executados
    na validação da task.

    Attributes:
        task_id: ID da task validada
        passed: Se passou em todos os gates obrigatórios
        gate_results: Resultados por gate
        qa_results: Resultados por QA worker
        gates_executed: Lista de gates que foram executados
        gates_passed: Lista de gates que passaram
        gates_failed: Lista de gates que falharam
        error_details: Detalhes dos erros encontrados
        execution_time_ms: Tempo de validação em ms
    """

    # === IDENTIFICAÇÃO ===
    task_id: str = ""
    validation_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # === RESULTADO ===
    passed: bool = False
    gate_results: Dict[str, GateResult] = field(default_factory=dict)
    qa_results: Dict[str, QAWorkerResult] = field(default_factory=dict)
    gates_executed: List[str] = field(default_factory=list)
    gates_passed: List[str] = field(default_factory=list)
    gates_failed: List[str] = field(default_factory=list)

    # === DETALHES ===
    error_details: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    # === MÉTRICAS ===
    execution_time_ms: int = 0
    coverage_percentage: Optional[float] = None
    test_count: int = 0
    test_passed: int = 0
    test_failed: int = 0

    # === METADADOS ===
    validated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Serializa o resultado para dicionário."""
        return {
            "task_id": self.task_id,
            "validation_id": self.validation_id,
            "passed": self.passed,
            "gates_executed": self.gates_executed,
            "gates_passed": self.gates_passed,
            "gates_failed": self.gates_failed,
            "error_details": self.error_details,
            "warnings": self.warnings,
            "execution_time_ms": self.execution_time_ms,
            "coverage_percentage": self.coverage_percentage,
            "test_count": self.test_count,
            "test_passed": self.test_passed,
            "test_failed": self.test_failed,
            "validated_at": self.validated_at.isoformat(),
        }

    def get_failure_summary(self) -> str:
        """Gera resumo das falhas para diagnóstico."""
        if self.passed:
            return "No failures - all gates passed."

        lines = [f"Validation failed for task {self.task_id}"]
        lines.append(f"Gates failed: {', '.join(self.gates_failed)}")

        for gate, error in self.error_details.items():
            lines.append(f"\n{gate}:")
            lines.append(f"  {error}")

        return '\n'.join(lines)
