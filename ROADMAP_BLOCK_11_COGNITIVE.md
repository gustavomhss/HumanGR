# ROADMAP Block 11: Cognitive Modules

> **Bloco**: 11 de 12
> **Tema**: Módulos Cognitivos (Budget, Trust, Triage, Feedback)
> **Tokens Estimados**: ~25,000
> **Dependências**: Block 08-10 (Engine, Layers, Perspectives)

---

## Visão Geral do Bloco

Os módulos cognitivos adicionam inteligência ao Human Layer:

| Módulo | Função | Descrição |
|--------|--------|-----------|
| Budget Manager | Controle de custo | Gerencia tokens/$ por execução |
| Trust Scorer | Avaliação de confiança | Score de confiabilidade dos resultados |
| Triage Engine | Priorização | Ordena issues por severidade/impacto |
| Feedback Loop | Aprendizado | Aprende com resultados anteriores |
| Confidence Calculator | Cálculo de confidence | Combina sinais em score final |

---

## Módulo HLS-COG-001: Budget Manager

```
ID: HLS-COG-001
Nome: BudgetManager
Caminho: src/hl_mcp/cognitive/budget.py
Dependências: Nenhuma
Exports: BudgetManager, BudgetConfig, BudgetStatus
Linhas: ~250
```

### Código

```python
"""
HLS-COG-001: Budget Manager
===========================

Gerenciamento de orçamento de tokens/custo.

Responsabilidades:
- Rastrear uso de tokens por layer/perspectiva
- Aplicar limites de custo
- Sugerir otimizações
- Alocar budget entre componentes

Exemplo:
    >>> budget = BudgetManager(max_tokens=100000, max_cost_usd=1.0)
    >>> budget.allocate("HL-2", 20000)  # 20k para security
    >>> budget.record_usage("HL-2", 15000, 0.15)
    >>> print(budget.remaining_tokens)  # 85000

Changelog:
    - v1.0.0 (2026-02-01): Implementação inicial
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class BudgetComponent(str, Enum):
    """Componentes que consomem budget."""

    LAYER_HL1 = "HL-1"
    LAYER_HL2 = "HL-2"
    LAYER_HL3 = "HL-3"
    LAYER_HL4 = "HL-4"
    LAYER_HL5 = "HL-5"
    LAYER_HL6 = "HL-6"
    LAYER_HL7 = "HL-7"
    PERSPECTIVE = "perspective"
    REDUNDANCY = "redundancy"
    CONSENSUS = "consensus"
    OTHER = "other"


@dataclass
class BudgetConfig:
    """
    Configuração de budget.

    Attributes:
        max_tokens: Limite total de tokens
        max_cost_usd: Limite de custo em USD
        warn_at_percent: Avisar quando atingir %
        reserve_percent: Reservar % para emergências
        cost_per_1k_tokens: Custo por 1000 tokens
    """

    max_tokens: int = 100000
    max_cost_usd: float = 1.0
    warn_at_percent: float = 0.8
    reserve_percent: float = 0.1
    cost_per_1k_tokens: float = 0.01  # Estimativa média


@dataclass
class UsageRecord:
    """Registro de uso de budget."""

    component: BudgetComponent
    tokens_used: int
    cost_usd: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class BudgetStatus:
    """Status atual do budget."""

    total_tokens_used: int
    total_cost_usd: float
    remaining_tokens: int
    remaining_cost_usd: float
    percent_used: float
    is_exceeded: bool
    is_warning: bool
    by_component: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total_tokens_used": self.total_tokens_used,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "remaining_tokens": self.remaining_tokens,
            "remaining_cost_usd": round(self.remaining_cost_usd, 4),
            "percent_used": round(self.percent_used * 100, 1),
            "is_exceeded": self.is_exceeded,
            "is_warning": self.is_warning,
            "by_component": self.by_component,
        }


class BudgetManager:
    """
    Gerenciador de budget de tokens/custo.

    Rastreia uso e aplica limites para evitar
    gastos excessivos com LLM.

    Example:
        >>> config = BudgetConfig(max_tokens=50000, max_cost_usd=0.5)
        >>> budget = BudgetManager(config)
        >>>
        >>> # Antes de chamar LLM
        >>> if budget.can_spend(10000):
        ...     response = await llm.complete(prompt)
        ...     budget.record_usage(BudgetComponent.LAYER_HL2, 8500, 0.085)
        >>>
        >>> # Verificar status
        >>> status = budget.get_status()
        >>> print(f"Usado: {status.percent_used:.0%}")
    """

    def __init__(self, config: Optional[BudgetConfig] = None):
        """
        Inicializa o budget manager.

        Args:
            config: Configuração de budget
        """
        self.config = config or BudgetConfig()
        self._usage_history: list[UsageRecord] = []
        self._allocations: dict[BudgetComponent, int] = {}

    def allocate(self, component: BudgetComponent, tokens: int) -> bool:
        """
        Aloca tokens para um componente.

        Args:
            component: Componente que vai usar
            tokens: Quantidade de tokens

        Returns:
            True se alocação foi aceita
        """
        # Verifica se cabe no budget
        total_allocated = sum(self._allocations.values())
        available = self.config.max_tokens - self.total_tokens_used

        if total_allocated + tokens > available:
            logger.warning(f"Alocação negada: {component.value} pediu {tokens}")
            return False

        self._allocations[component] = tokens
        logger.debug(f"Alocado {tokens} tokens para {component.value}")
        return True

    def record_usage(
        self,
        component: BudgetComponent,
        tokens: int,
        cost_usd: Optional[float] = None,
    ) -> None:
        """
        Registra uso de tokens.

        Args:
            component: Componente que usou
            tokens: Tokens consumidos
            cost_usd: Custo em USD (calculado se None)
        """
        if cost_usd is None:
            cost_usd = (tokens / 1000) * self.config.cost_per_1k_tokens

        record = UsageRecord(
            component=component,
            tokens_used=tokens,
            cost_usd=cost_usd,
        )
        self._usage_history.append(record)

        logger.debug(
            f"Registrado: {component.value} usou {tokens} tokens (${cost_usd:.4f})"
        )

        # Verifica limites
        status = self.get_status()
        if status.is_exceeded:
            logger.error("BUDGET EXCEDIDO!")
        elif status.is_warning:
            logger.warning(f"Budget em {status.percent_used:.0%}")

    def can_spend(self, tokens: int) -> bool:
        """
        Verifica se pode gastar tokens.

        Args:
            tokens: Tokens a gastar

        Returns:
            True se dentro do budget
        """
        remaining = self.remaining_tokens
        reserve = int(self.config.max_tokens * self.config.reserve_percent)

        return (remaining - tokens) >= reserve

    @property
    def total_tokens_used(self) -> int:
        """Total de tokens usados."""
        return sum(r.tokens_used for r in self._usage_history)

    @property
    def total_cost_usd(self) -> float:
        """Custo total em USD."""
        return sum(r.cost_usd for r in self._usage_history)

    @property
    def remaining_tokens(self) -> int:
        """Tokens restantes."""
        return self.config.max_tokens - self.total_tokens_used

    @property
    def remaining_cost_usd(self) -> float:
        """Budget restante em USD."""
        return self.config.max_cost_usd - self.total_cost_usd

    def get_status(self) -> BudgetStatus:
        """Retorna status atual do budget."""
        used = self.total_tokens_used
        cost = self.total_cost_usd
        percent = used / self.config.max_tokens if self.config.max_tokens > 0 else 0

        # Agrupa por componente
        by_component = {}
        for record in self._usage_history:
            key = record.component.value
            by_component[key] = by_component.get(key, 0) + record.tokens_used

        return BudgetStatus(
            total_tokens_used=used,
            total_cost_usd=cost,
            remaining_tokens=self.remaining_tokens,
            remaining_cost_usd=self.remaining_cost_usd,
            percent_used=percent,
            is_exceeded=percent >= 1.0 or cost >= self.config.max_cost_usd,
            is_warning=percent >= self.config.warn_at_percent,
            by_component=by_component,
        )

    def get_usage_by_component(self) -> dict[str, dict]:
        """Retorna uso detalhado por componente."""
        result = {}

        for record in self._usage_history:
            key = record.component.value
            if key not in result:
                result[key] = {"tokens": 0, "cost": 0.0, "calls": 0}
            result[key]["tokens"] += record.tokens_used
            result[key]["cost"] += record.cost_usd
            result[key]["calls"] += 1

        return result

    def suggest_optimization(self) -> list[str]:
        """Sugere otimizações baseado no uso."""
        suggestions = []
        usage = self.get_usage_by_component()

        # Encontra componentes mais caros
        sorted_usage = sorted(
            usage.items(),
            key=lambda x: x[1]["tokens"],
            reverse=True,
        )

        if sorted_usage:
            top = sorted_usage[0]
            if top[1]["tokens"] > self.total_tokens_used * 0.5:
                suggestions.append(
                    f"Considere otimizar {top[0]} - usando {top[1]['tokens']/self.total_tokens_used:.0%} do budget"
                )

        # Verifica se redundância está muito cara
        redundancy = usage.get("redundancy", {})
        if redundancy.get("tokens", 0) > self.total_tokens_used * 0.3:
            suggestions.append(
                "Redundância tripla consumindo muito - considere reduzir para layers críticas"
            )

        return suggestions

    def reset(self) -> None:
        """Reseta o budget (nova sessão)."""
        self._usage_history.clear()
        self._allocations.clear()
        logger.info("Budget resetado")


__all__ = [
    "BudgetComponent",
    "BudgetConfig",
    "UsageRecord",
    "BudgetStatus",
    "BudgetManager",
]
```

---

## Módulo HLS-COG-002: Trust Scorer

```
ID: HLS-COG-002
Nome: TrustScorer
Caminho: src/hl_mcp/cognitive/trust.py
Dependências: HLS-MDL-001
Exports: TrustScorer, TrustScore, TrustFactors
Linhas: ~220
```

### Código

```python
"""
HLS-COG-002: Trust Scorer
=========================

Avaliação de confiabilidade de resultados.

Responsabilidades:
- Calcular score de confiança
- Identificar fatores de risco
- Avaliar qualidade de findings
- Sugerir verificações adicionais

Exemplo:
    >>> scorer = TrustScorer()
    >>> score = scorer.evaluate(findings, consensus_level, layer_id)
    >>> print(f"Confiança: {score.value:.0%}")
    >>> if score.needs_review:
    ...     print(f"Revisar: {score.review_reasons}")

Changelog:
    - v1.0.0 (2026-02-01): Implementação inicial
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class TrustLevel(str, Enum):
    """Níveis de confiança."""

    VERY_HIGH = "very_high"  # >= 0.9
    HIGH = "high"            # >= 0.75
    MEDIUM = "medium"        # >= 0.5
    LOW = "low"              # >= 0.25
    VERY_LOW = "very_low"    # < 0.25


@dataclass
class TrustFactors:
    """
    Fatores que influenciam a confiança.

    Positive factors aumentam confiança.
    Negative factors diminuem confiança.
    """

    # Fatores positivos
    redundancy_agreement: float = 0.0  # 3/3 concordaram
    multiple_perspectives: bool = False
    has_evidence: bool = False
    consistent_severity: bool = False

    # Fatores negativos
    low_confidence_findings: int = 0
    contradictions: int = 0
    parsing_errors: int = 0
    missing_context: bool = False

    def positive_score(self) -> float:
        """Score dos fatores positivos (0-1)."""
        score = 0.0
        score += self.redundancy_agreement * 0.4
        score += 0.2 if self.multiple_perspectives else 0
        score += 0.2 if self.has_evidence else 0
        score += 0.2 if self.consistent_severity else 0
        return min(1.0, score)

    def negative_score(self) -> float:
        """Score dos fatores negativos (0-1)."""
        score = 0.0
        score += min(0.3, self.low_confidence_findings * 0.1)
        score += min(0.3, self.contradictions * 0.15)
        score += min(0.2, self.parsing_errors * 0.1)
        score += 0.2 if self.missing_context else 0
        return min(1.0, score)


@dataclass
class TrustScore:
    """
    Score de confiança calculado.

    Attributes:
        value: Score de 0 a 1
        level: Nível de confiança
        factors: Fatores usados no cálculo
        needs_review: Se precisa revisão humana
        review_reasons: Razões para revisão
    """

    value: float
    level: TrustLevel
    factors: TrustFactors
    needs_review: bool = False
    review_reasons: list[str] = field(default_factory=list)

    @classmethod
    def from_value(cls, value: float, factors: TrustFactors) -> "TrustScore":
        """Cria TrustScore a partir de valor."""
        # Determina nível
        if value >= 0.9:
            level = TrustLevel.VERY_HIGH
        elif value >= 0.75:
            level = TrustLevel.HIGH
        elif value >= 0.5:
            level = TrustLevel.MEDIUM
        elif value >= 0.25:
            level = TrustLevel.LOW
        else:
            level = TrustLevel.VERY_LOW

        # Determina se precisa revisão
        needs_review = value < 0.6
        reasons = []

        if factors.contradictions > 0:
            reasons.append(f"{factors.contradictions} contradições encontradas")
        if factors.low_confidence_findings > 2:
            reasons.append(f"{factors.low_confidence_findings} findings com baixa confiança")
        if factors.missing_context:
            reasons.append("Contexto insuficiente para análise completa")

        return cls(
            value=round(value, 3),
            level=level,
            factors=factors,
            needs_review=needs_review,
            review_reasons=reasons,
        )

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "level": self.level.value,
            "needs_review": self.needs_review,
            "review_reasons": self.review_reasons,
        }


class TrustScorer:
    """
    Calculador de score de confiança.

    Avalia a confiabilidade dos resultados do Human Layer
    baseado em múltiplos fatores.

    Example:
        >>> scorer = TrustScorer()
        >>>
        >>> factors = TrustFactors(
        ...     redundancy_agreement=1.0,  # 3/3 concordaram
        ...     has_evidence=True,
        ... )
        >>>
        >>> score = scorer.calculate(factors)
        >>> print(f"Trust: {score.level.value}")  # "high"
    """

    def __init__(
        self,
        base_weight: float = 0.5,
        positive_weight: float = 0.35,
        negative_weight: float = 0.15,
    ):
        """
        Inicializa o scorer.

        Args:
            base_weight: Peso do score base
            positive_weight: Peso dos fatores positivos
            negative_weight: Peso dos fatores negativos
        """
        self.base_weight = base_weight
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight

    def calculate(self, factors: TrustFactors) -> TrustScore:
        """
        Calcula score de confiança.

        Args:
            factors: Fatores a considerar

        Returns:
            TrustScore calculado
        """
        # Score base (assume confiança média)
        base = 0.6

        # Ajusta por fatores positivos
        positive = factors.positive_score()

        # Ajusta por fatores negativos
        negative = factors.negative_score()

        # Calcula score final
        value = (
            base * self.base_weight +
            positive * self.positive_weight -
            negative * self.negative_weight
        )

        # Clamp entre 0 e 1
        value = max(0.0, min(1.0, value))

        return TrustScore.from_value(value, factors)

    def evaluate_findings(
        self,
        findings: list[dict],
        consensus_level: Optional[str] = None,
    ) -> TrustScore:
        """
        Avalia confiança baseado em findings.

        Args:
            findings: Lista de findings
            consensus_level: Nível de consenso (unanimous, majority, split)

        Returns:
            TrustScore
        """
        # Constrói fatores a partir dos findings
        factors = TrustFactors()

        # Redundancy agreement
        if consensus_level == "unanimous":
            factors.redundancy_agreement = 1.0
        elif consensus_level == "majority":
            factors.redundancy_agreement = 0.67
        elif consensus_level == "split":
            factors.redundancy_agreement = 0.33

        # Verifica evidências
        factors.has_evidence = any(
            f.get("evidence") or f.get("code_snippet")
            for f in findings
        )

        # Conta findings com baixa confiança
        for finding in findings:
            confidence = finding.get("confidence", 0.7)
            if confidence < 0.5:
                factors.low_confidence_findings += 1

        # Verifica consistência de severidade
        severities = [f.get("severity", "medium") for f in findings]
        factors.consistent_severity = len(set(severities)) <= 2

        return self.calculate(factors)


__all__ = [
    "TrustLevel",
    "TrustFactors",
    "TrustScore",
    "TrustScorer",
]
```

---

## Módulo HLS-COG-003: Triage Engine

```
ID: HLS-COG-003
Nome: TriageEngine
Caminho: src/hl_mcp/cognitive/triage.py
Dependências: HLS-MDL-001
Exports: TriageEngine, TriagedFinding, TriageResult
Linhas: ~200
```

### Código

```python
"""
HLS-COG-003: Triage Engine
==========================

Priorização e ordenação de issues.

Responsabilidades:
- Ordenar findings por severidade/impacto
- Agrupar por categoria
- Identificar quick wins
- Sugerir ordem de resolução

Exemplo:
    >>> triage = TriageEngine()
    >>> result = triage.prioritize(findings)
    >>> for finding in result.critical_first:
    ...     print(f"[{finding.priority}] {finding.title}")

Changelog:
    - v1.0.0 (2026-02-01): Implementação inicial
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class Priority(str, Enum):
    """Níveis de prioridade."""

    P0 = "P0"  # Crítico - resolver imediatamente
    P1 = "P1"  # Alto - resolver em 24h
    P2 = "P2"  # Médio - resolver esta sprint
    P3 = "P3"  # Baixo - backlog
    P4 = "P4"  # Info - opcional


@dataclass
class TriagedFinding:
    """Finding com prioridade atribuída."""

    finding: dict
    priority: Priority
    score: float
    effort_estimate: str  # low, medium, high
    is_quick_win: bool
    related_findings: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.finding.get("id", "unknown")

    @property
    def title(self) -> str:
        return self.finding.get("title", "No title")

    @property
    def severity(self) -> str:
        return self.finding.get("severity", "medium")


@dataclass
class TriageResult:
    """Resultado da triagem."""

    triaged_findings: list[TriagedFinding]
    by_priority: dict[str, list[TriagedFinding]] = field(default_factory=dict)
    quick_wins: list[TriagedFinding] = field(default_factory=list)
    blocked_by: dict[str, list[str]] = field(default_factory=dict)

    @property
    def critical_first(self) -> list[TriagedFinding]:
        """Retorna findings ordenados por prioridade."""
        return sorted(
            self.triaged_findings,
            key=lambda f: (f.priority.value, -f.score),
        )

    @property
    def p0_count(self) -> int:
        return len(self.by_priority.get("P0", []))

    def to_dict(self) -> dict:
        return {
            "total": len(self.triaged_findings),
            "by_priority": {
                k: len(v) for k, v in self.by_priority.items()
            },
            "quick_wins": len(self.quick_wins),
        }


class TriageEngine:
    """
    Engine de triagem e priorização.

    Analisa findings e atribui prioridades baseado em:
    - Severidade
    - Impacto potencial
    - Esforço de correção
    - Dependências

    Example:
        >>> engine = TriageEngine()
        >>> findings = [
        ...     {"severity": "critical", "title": "SQL Injection"},
        ...     {"severity": "low", "title": "Typo in label"},
        ... ]
        >>> result = engine.prioritize(findings)
        >>> print(f"P0s: {result.p0_count}")
    """

    # Mapeamento severidade -> prioridade base
    SEVERITY_PRIORITY = {
        "critical": Priority.P0,
        "high": Priority.P1,
        "medium": Priority.P2,
        "low": Priority.P3,
        "info": Priority.P4,
    }

    # Categorias que aumentam prioridade
    HIGH_PRIORITY_CATEGORIES = {
        "security",
        "vulnerability",
        "injection",
        "auth",
        "data_exposure",
    }

    def __init__(self):
        pass

    def prioritize(self, findings: list[dict]) -> TriageResult:
        """
        Prioriza lista de findings.

        Args:
            findings: Lista de findings para priorizar

        Returns:
            TriageResult com findings priorizados
        """
        triaged = []
        by_priority: dict[str, list[TriagedFinding]] = {}
        quick_wins = []

        for finding in findings:
            tf = self._triage_finding(finding)
            triaged.append(tf)

            # Agrupa por prioridade
            priority_key = tf.priority.value
            if priority_key not in by_priority:
                by_priority[priority_key] = []
            by_priority[priority_key].append(tf)

            # Identifica quick wins
            if tf.is_quick_win:
                quick_wins.append(tf)

        logger.info(
            f"Triagem: {len(triaged)} findings, "
            f"{len(by_priority.get('P0', []))} P0s, "
            f"{len(quick_wins)} quick wins"
        )

        return TriageResult(
            triaged_findings=triaged,
            by_priority=by_priority,
            quick_wins=quick_wins,
        )

    def _triage_finding(self, finding: dict) -> TriagedFinding:
        """Faz triagem de um finding individual."""
        severity = finding.get("severity", "medium").lower()
        category = finding.get("category", "").lower()

        # Prioridade base pela severidade
        priority = self.SEVERITY_PRIORITY.get(severity, Priority.P2)

        # Aumenta prioridade para categorias críticas
        for high_cat in self.HIGH_PRIORITY_CATEGORIES:
            if high_cat in category:
                if priority == Priority.P1:
                    priority = Priority.P0
                elif priority == Priority.P2:
                    priority = Priority.P1
                break

        # Calcula score (0-100)
        score = self._calculate_score(finding, priority)

        # Estima esforço
        effort = self._estimate_effort(finding)

        # Identifica quick wins (alto impacto, baixo esforço)
        is_quick_win = (
            priority.value <= "P2" and
            effort == "low" and
            score >= 50
        )

        return TriagedFinding(
            finding=finding,
            priority=priority,
            score=score,
            effort_estimate=effort,
            is_quick_win=is_quick_win,
        )

    def _calculate_score(self, finding: dict, priority: Priority) -> float:
        """Calcula score de importância (0-100)."""
        base_scores = {
            Priority.P0: 90,
            Priority.P1: 70,
            Priority.P2: 50,
            Priority.P3: 30,
            Priority.P4: 10,
        }

        score = base_scores.get(priority, 50)

        # Bônus por ter evidência
        if finding.get("evidence") or finding.get("code_snippet"):
            score += 5

        # Bônus por ter remediação
        if finding.get("remediation"):
            score += 5

        return min(100, score)

    def _estimate_effort(self, finding: dict) -> str:
        """Estima esforço de correção."""
        # Heurística simples baseada em categoria e descrição
        category = finding.get("category", "").lower()
        description = finding.get("description", "")

        # Alto esforço: refactoring, arquitetura
        if any(word in description.lower() for word in ["refactor", "redesign", "architecture"]):
            return "high"

        # Médio esforço: mudanças moderadas
        if any(word in category for word in ["integration", "performance"]):
            return "medium"

        # Baixo esforço: fixes simples
        return "low"


__all__ = [
    "Priority",
    "TriagedFinding",
    "TriageResult",
    "TriageEngine",
]
```

---

## Módulo HLS-COG-004: Feedback Loop

```
ID: HLS-COG-004
Nome: FeedbackLoop
Caminho: src/hl_mcp/cognitive/feedback.py
Dependências: Nenhuma
Exports: FeedbackLoop, FeedbackEntry, LearningInsight
Linhas: ~180
```

### Código

```python
"""
HLS-COG-004: Feedback Loop
==========================

Sistema de aprendizado com resultados anteriores.

Responsabilidades:
- Registrar resultados de execuções
- Aprender padrões de sucesso/falha
- Ajustar comportamento futuro
- Identificar tendências

Exemplo:
    >>> feedback = FeedbackLoop()
    >>> feedback.record_execution(result, user_feedback="helpful")
    >>> insights = feedback.get_insights()
    >>> for insight in insights:
    ...     print(f"{insight.type}: {insight.message}")

Changelog:
    - v1.0.0 (2026-02-01): Implementação inicial
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class FeedbackType(str, Enum):
    """Tipos de feedback."""

    HELPFUL = "helpful"
    NOT_HELPFUL = "not_helpful"
    FALSE_POSITIVE = "false_positive"
    MISSED_ISSUE = "missed_issue"
    ACCURATE = "accurate"


@dataclass
class FeedbackEntry:
    """Entrada de feedback."""

    execution_id: str
    layer_id: Optional[str]
    finding_id: Optional[str]
    feedback_type: FeedbackType
    user_comment: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class LearningInsight:
    """Insight aprendido do feedback."""

    type: str
    message: str
    confidence: float
    based_on_samples: int
    recommendation: Optional[str] = None


class FeedbackLoop:
    """
    Sistema de feedback e aprendizado.

    Coleta feedback de execuções anteriores e gera
    insights para melhorar execuções futuras.

    Example:
        >>> feedback = FeedbackLoop()
        >>>
        >>> # Registrar feedback
        >>> feedback.record(
        ...     execution_id="exec_123",
        ...     layer_id="HL-2",
        ...     feedback_type=FeedbackType.ACCURATE,
        ... )
        >>>
        >>> # Obter insights
        >>> insights = feedback.get_insights()
    """

    def __init__(self, max_history: int = 1000):
        """
        Inicializa o feedback loop.

        Args:
            max_history: Máximo de entradas a manter
        """
        self.max_history = max_history
        self._entries: list[FeedbackEntry] = []
        self._layer_stats: dict[str, dict] = {}

    def record(
        self,
        execution_id: str,
        layer_id: Optional[str] = None,
        finding_id: Optional[str] = None,
        feedback_type: FeedbackType = FeedbackType.HELPFUL,
        user_comment: Optional[str] = None,
    ) -> None:
        """Registra feedback."""
        entry = FeedbackEntry(
            execution_id=execution_id,
            layer_id=layer_id,
            finding_id=finding_id,
            feedback_type=feedback_type,
            user_comment=user_comment,
        )

        self._entries.append(entry)

        # Mantém limite de histórico
        if len(self._entries) > self.max_history:
            self._entries = self._entries[-self.max_history:]

        # Atualiza stats por layer
        if layer_id:
            self._update_layer_stats(layer_id, feedback_type)

        logger.debug(f"Feedback registrado: {feedback_type.value}")

    def _update_layer_stats(self, layer_id: str, feedback_type: FeedbackType) -> None:
        """Atualiza estatísticas por layer."""
        if layer_id not in self._layer_stats:
            self._layer_stats[layer_id] = {
                "total": 0,
                "helpful": 0,
                "accurate": 0,
                "false_positive": 0,
            }

        stats = self._layer_stats[layer_id]
        stats["total"] += 1

        if feedback_type == FeedbackType.HELPFUL:
            stats["helpful"] += 1
        elif feedback_type == FeedbackType.ACCURATE:
            stats["accurate"] += 1
        elif feedback_type == FeedbackType.FALSE_POSITIVE:
            stats["false_positive"] += 1

    def get_layer_accuracy(self, layer_id: str) -> float:
        """Retorna taxa de acurácia de uma layer."""
        stats = self._layer_stats.get(layer_id, {})
        total = stats.get("total", 0)

        if total == 0:
            return 0.5  # Sem dados, assume 50%

        accurate = stats.get("accurate", 0) + stats.get("helpful", 0)
        false_pos = stats.get("false_positive", 0)

        if accurate + false_pos == 0:
            return 0.5

        return accurate / (accurate + false_pos)

    def get_insights(self) -> list[LearningInsight]:
        """Gera insights a partir do histórico."""
        insights = []

        for layer_id, stats in self._layer_stats.items():
            accuracy = self.get_layer_accuracy(layer_id)

            if accuracy < 0.6 and stats["total"] >= 10:
                insights.append(LearningInsight(
                    type="low_accuracy",
                    message=f"{layer_id} tem baixa acurácia ({accuracy:.0%})",
                    confidence=0.8,
                    based_on_samples=stats["total"],
                    recommendation="Considere revisar prompts ou aumentar threshold",
                ))

            if stats.get("false_positive", 0) > stats["total"] * 0.3:
                insights.append(LearningInsight(
                    type="high_false_positive",
                    message=f"{layer_id} tem muitos falsos positivos",
                    confidence=0.75,
                    based_on_samples=stats["total"],
                    recommendation="Aumentar threshold de confiança",
                ))

        return insights

    def get_summary(self) -> dict:
        """Retorna resumo do feedback."""
        total = len(self._entries)

        by_type = {}
        for entry in self._entries:
            key = entry.feedback_type.value
            by_type[key] = by_type.get(key, 0) + 1

        return {
            "total_entries": total,
            "by_type": by_type,
            "layers_tracked": list(self._layer_stats.keys()),
        }


__all__ = [
    "FeedbackType",
    "FeedbackEntry",
    "LearningInsight",
    "FeedbackLoop",
]
```

---

## Módulo HLS-COG-005: Confidence Calculator

```
ID: HLS-COG-005
Nome: ConfidenceCalculator
Caminho: src/hl_mcp/cognitive/confidence.py
Dependências: HLS-COG-002, HLS-MDL-002
Exports: ConfidenceCalculator, ConfidenceBreakdown
Linhas: ~150
```

### Código

```python
"""
HLS-COG-005: Confidence Calculator
==================================

Cálculo de confiança final combinando múltiplos sinais.

Responsabilidades:
- Combinar sinais de confiança
- Pesar por fonte
- Fornecer breakdown detalhado
- Normalizar scores

Exemplo:
    >>> calc = ConfidenceCalculator()
    >>> breakdown = calc.calculate(
    ...     layer_confidence=0.85,
    ...     consensus_level="majority",
    ...     trust_score=0.72,
    ... )
    >>> print(f"Final: {breakdown.final_score:.0%}")

Changelog:
    - v1.0.0 (2026-02-01): Implementação inicial
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceBreakdown:
    """
    Breakdown detalhado do cálculo de confiança.

    Attributes:
        final_score: Score final (0-1)
        components: Scores por componente
        weights: Pesos usados
        adjustments: Ajustes aplicados
    """

    final_score: float
    components: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    adjustments: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "final_score": round(self.final_score, 3),
            "components": {
                k: round(v, 3) for k, v in self.components.items()
            },
            "adjustments": self.adjustments,
        }


class ConfidenceCalculator:
    """
    Calculador de confiança combinada.

    Combina múltiplos sinais de confiança em um score final.

    Example:
        >>> calc = ConfidenceCalculator()
        >>> result = calc.calculate(
        ...     layer_confidence=0.9,
        ...     consensus_level="unanimous",
        ... )
        >>> print(result.final_score)  # ~0.92
    """

    # Pesos padrão
    DEFAULT_WEIGHTS = {
        "layer": 0.3,
        "consensus": 0.25,
        "trust": 0.2,
        "redundancy": 0.15,
        "feedback": 0.1,
    }

    # Mapeamento de consensus level para score
    CONSENSUS_SCORES = {
        "unanimous": 1.0,
        "majority": 0.75,
        "split": 0.4,
        "failed": 0.1,
    }

    def __init__(
        self,
        weights: Optional[dict[str, float]] = None,
    ):
        """
        Inicializa o calculador.

        Args:
            weights: Pesos customizados
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()

        # Normaliza pesos
        total = sum(self.weights.values())
        if total != 1.0:
            self.weights = {
                k: v / total for k, v in self.weights.items()
            }

    def calculate(
        self,
        layer_confidence: float = 0.5,
        consensus_level: Optional[str] = None,
        trust_score: Optional[float] = None,
        redundancy_agreement: Optional[float] = None,
        feedback_accuracy: Optional[float] = None,
    ) -> ConfidenceBreakdown:
        """
        Calcula confiança combinada.

        Args:
            layer_confidence: Confiança da layer (0-1)
            consensus_level: Nível de consenso
            trust_score: Score de trust (0-1)
            redundancy_agreement: Concordância de redundância (0-1)
            feedback_accuracy: Acurácia de feedback histórico (0-1)

        Returns:
            ConfidenceBreakdown com score final
        """
        components = {}
        adjustments = []

        # Layer confidence
        components["layer"] = layer_confidence

        # Consensus
        if consensus_level:
            components["consensus"] = self.CONSENSUS_SCORES.get(
                consensus_level, 0.5
            )
            if consensus_level == "unanimous":
                adjustments.append("Bônus: consenso unânime")

        # Trust score
        if trust_score is not None:
            components["trust"] = trust_score

        # Redundancy
        if redundancy_agreement is not None:
            components["redundancy"] = redundancy_agreement

        # Feedback
        if feedback_accuracy is not None:
            components["feedback"] = feedback_accuracy

        # Calcula média ponderada
        weighted_sum = 0.0
        total_weight = 0.0

        for key, value in components.items():
            weight = self.weights.get(key, 0)
            weighted_sum += value * weight
            total_weight += weight

        if total_weight > 0:
            final_score = weighted_sum / total_weight
        else:
            final_score = 0.5

        # Aplica ajustes
        if len(components) < 3:
            final_score *= 0.95
            adjustments.append("Penalidade: poucos sinais")

        if layer_confidence < 0.3:
            final_score *= 0.9
            adjustments.append("Penalidade: layer com baixa confiança")

        # Clamp
        final_score = max(0.0, min(1.0, final_score))

        return ConfidenceBreakdown(
            final_score=final_score,
            components=components,
            weights=self.weights,
            adjustments=adjustments,
        )


__all__ = [
    "ConfidenceBreakdown",
    "ConfidenceCalculator",
]
```

---

## Cognitive __init__.py

```python
"""
HLS-COG: Cognitive Modules
==========================

Módulos de inteligência do Human Layer.

Exports:
    - BudgetManager, BudgetConfig, BudgetStatus
    - TrustScorer, TrustScore, TrustFactors
    - TriageEngine, TriagedFinding, TriageResult
    - FeedbackLoop, FeedbackEntry, LearningInsight
    - ConfidenceCalculator, ConfidenceBreakdown
"""

from .budget import (
    BudgetComponent,
    BudgetConfig,
    UsageRecord,
    BudgetStatus,
    BudgetManager,
)

from .trust import (
    TrustLevel,
    TrustFactors,
    TrustScore,
    TrustScorer,
)

from .triage import (
    Priority,
    TriagedFinding,
    TriageResult,
    TriageEngine,
)

from .feedback import (
    FeedbackType,
    FeedbackEntry,
    LearningInsight,
    FeedbackLoop,
)

from .confidence import (
    ConfidenceBreakdown,
    ConfidenceCalculator,
)


__all__ = [
    # Budget
    "BudgetComponent",
    "BudgetConfig",
    "UsageRecord",
    "BudgetStatus",
    "BudgetManager",
    # Trust
    "TrustLevel",
    "TrustFactors",
    "TrustScore",
    "TrustScorer",
    # Triage
    "Priority",
    "TriagedFinding",
    "TriageResult",
    "TriageEngine",
    # Feedback
    "FeedbackType",
    "FeedbackEntry",
    "LearningInsight",
    "FeedbackLoop",
    # Confidence
    "ConfidenceBreakdown",
    "ConfidenceCalculator",
]
```

---

## Resumo do Block 11

| Módulo | ID | Função | Linhas |
|--------|----|--------|--------|
| BudgetManager | HLS-COG-001 | Controle de tokens/custo | ~250 |
| TrustScorer | HLS-COG-002 | Score de confiança | ~220 |
| TriageEngine | HLS-COG-003 | Priorização de issues | ~200 |
| FeedbackLoop | HLS-COG-004 | Aprendizado | ~180 |
| ConfidenceCalculator | HLS-COG-005 | Cálculo final | ~150 |
| **TOTAL** | | | **~1,000** |

---

## Próximo: Block 12 - MCP Server Final

O Block 12 finaliza o MCP Server:
1. **HLS-MCP-001**: MCPServer (servidor principal)
2. **HLS-MCP-002**: ToolHandlers (handlers de tools)
3. **HLS-MCP-003**: ResourceHandlers (handlers de resources)
4. **HLS-MCP-004**: Configuration (config e setup)

Quer que eu continue com o Block 12 (MCP Server Final)?
