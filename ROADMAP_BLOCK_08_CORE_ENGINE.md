# ROADMAP Block 08: Core Engine

> **Bloco**: 08 de 12
> **Tema**: Core Engine (Runner, Orchestrator, Gates)
> **Tokens Estimados**: ~35,000
> **Dependências**: Block 04-07 (Models, LLM, Browser)

---

## Visão Geral do Bloco

Este bloco implementa o núcleo do Human Layer - a engine que:

1. **HumanLayerRunner** - Executa uma layer individual
2. **TripleRedundancy** - Executa 3x e aplica consenso 2/3
3. **LayerOrchestrator** - Orquestra as 7 layers em sequência
4. **VetoGate** - Aplica lógica de veto (WEAK/MEDIUM/STRONG)
5. **ConsensusEngine** - Consolida resultados de múltiplas execuções

---

## Módulo HLS-ENG-001: HumanLayerRunner

```
ID: HLS-ENG-001
Nome: HumanLayerRunner
Caminho: src/hl_mcp/engine/runner.py
Dependências: HLS-LLM-001, HLS-MDL-001, HLS-MDL-002
Exports: HumanLayerRunner, RunContext, RunResult
Linhas: ~350
```

### Código

```python
"""
HLS-ENG-001: HumanLayerRunner
=============================

Executor de uma Human Layer individual.

Responsabilidades:
- Preparar contexto para a layer
- Invocar LLM com prompt apropriado
- Parsear resposta em Findings
- Retornar LayerResult estruturado

Cada layer (HL-1 a HL-7) é executada por um Runner
com seu prompt e perspectiva específicos.

Exemplo:
    >>> runner = HumanLayerRunner(llm_client)
    >>> context = RunContext(
    ...     layer_id="HL-1",
    ...     target_description="Login form",
    ...     code_snippet="def login(user, pwd): ...",
    ... )
    >>> result = await runner.run(context)
    >>> print(f"Findings: {len(result.findings)}")

Changelog:
    - v1.0.0 (2026-02-01): Implementação inicial
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Protocol
import logging
import uuid

logger = logging.getLogger(__name__)


class LayerID(str, Enum):
    """Identificadores das 7 Human Layers."""

    HL_1 = "HL-1"  # UI/UX Review
    HL_2 = "HL-2"  # Security Scan
    HL_3 = "HL-3"  # Edge Cases
    HL_4 = "HL-4"  # Accessibility
    HL_5 = "HL-5"  # Performance
    HL_6 = "HL-6"  # Integration
    HL_7 = "HL-7"  # Final Human Check


@dataclass
class RunContext:
    """
    Contexto para execução de uma layer.

    Attributes:
        layer_id: ID da layer (HL-1 a HL-7)
        target_description: Descrição do que está sendo testado
        code_snippet: Código relevante (se aplicável)
        screenshot_path: Path do screenshot (se disponível)
        journey_result: Resultado de jornada executada
        previous_findings: Findings de layers anteriores
        perspective: Perspectiva a usar (opcional)
        extra_context: Contexto adicional livre
        run_id: ID único desta execução
    """

    layer_id: LayerID
    target_description: str
    code_snippet: Optional[str] = None
    screenshot_path: Optional[str] = None
    journey_result: Optional[dict] = None
    previous_findings: list[dict] = field(default_factory=list)
    perspective: Optional[str] = None
    extra_context: dict = field(default_factory=dict)
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def to_dict(self) -> dict:
        """Converte para dicionário."""
        return {
            "layer_id": self.layer_id.value,
            "target_description": self.target_description,
            "code_snippet": self.code_snippet,
            "screenshot_path": self.screenshot_path,
            "perspective": self.perspective,
            "run_id": self.run_id,
        }


@dataclass
class RunResult:
    """
    Resultado de uma execução de layer.

    Attributes:
        context: Contexto usado
        success: Se a execução foi bem sucedida
        findings: Findings encontrados
        raw_response: Resposta bruta do LLM
        veto_level: Nível de veto sugerido
        confidence: Confiança na análise (0-1)
        duration_ms: Duração em ms
        error: Erro (se falhou)
        timestamp: Momento da execução
    """

    context: RunContext
    success: bool
    findings: list[dict] = field(default_factory=list)
    raw_response: str = ""
    veto_level: Optional[str] = None
    confidence: float = 0.0
    duration_ms: float = 0
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def finding_count(self) -> int:
        """Número de findings."""
        return len(self.findings)

    @property
    def critical_count(self) -> int:
        """Número de findings críticos."""
        return sum(
            1 for f in self.findings
            if f.get("severity") in ["critical", "high"]
        )

    def to_dict(self) -> dict:
        """Converte para dicionário."""
        return {
            "run_id": self.context.run_id,
            "layer_id": self.context.layer_id.value,
            "success": self.success,
            "finding_count": self.finding_count,
            "critical_count": self.critical_count,
            "veto_level": self.veto_level,
            "confidence": self.confidence,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
        }


class LLMClientProtocol(Protocol):
    """Protocolo para cliente LLM."""

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Executa completion."""
        ...


class ResponseParserProtocol(Protocol):
    """Protocolo para parser de respostas."""

    def parse(self, response: str) -> dict:
        """Parseia resposta do LLM."""
        ...


class HumanLayerRunner:
    """
    Executor de uma Human Layer individual.

    Responsável por:
    1. Montar prompt com contexto
    2. Invocar LLM
    3. Parsear resposta
    4. Retornar RunResult estruturado

    Example:
        >>> from hl_mcp.llm import get_llm_client
        >>> from hl_mcp.llm.parser import ResponseParser
        >>>
        >>> llm = get_llm_client("claude")
        >>> parser = ResponseParser()
        >>> runner = HumanLayerRunner(llm, parser)
        >>>
        >>> context = RunContext(
        ...     layer_id=LayerID.HL_2,
        ...     target_description="Authentication endpoint",
        ...     code_snippet="def auth(token): ...",
        ... )
        >>> result = await runner.run(context)
    """

    def __init__(
        self,
        llm_client: LLMClientProtocol,
        response_parser: ResponseParserProtocol,
        prompt_templates: Optional[dict[str, str]] = None,
        default_timeout: int = 120000,
    ):
        """
        Inicializa o runner.

        Args:
            llm_client: Cliente LLM para completions
            response_parser: Parser de respostas
            prompt_templates: Templates por layer (opcional)
            default_timeout: Timeout padrão em ms
        """
        self.llm_client = llm_client
        self.response_parser = response_parser
        self.prompt_templates = prompt_templates or {}
        self.default_timeout = default_timeout

    def _get_system_prompt(self, layer_id: LayerID) -> str:
        """Obtém system prompt para a layer."""
        base_prompts = {
            LayerID.HL_1: (
                "You are a UI/UX expert reviewing user interfaces. "
                "Focus on usability, clarity, and user experience. "
                "Identify confusing elements, poor labeling, or bad flows."
            ),
            LayerID.HL_2: (
                "You are a security expert performing code review. "
                "Focus on vulnerabilities: injection, XSS, auth bypass, "
                "sensitive data exposure. Use OWASP Top 10 as reference."
            ),
            LayerID.HL_3: (
                "You are a QA expert finding edge cases. "
                "Think about boundary conditions, null values, "
                "race conditions, and unexpected inputs."
            ),
            LayerID.HL_4: (
                "You are an accessibility expert. "
                "Focus on WCAG compliance, screen reader compatibility, "
                "keyboard navigation, and color contrast."
            ),
            LayerID.HL_5: (
                "You are a performance expert. "
                "Look for N+1 queries, missing indexes, memory leaks, "
                "unnecessary computations, and scalability issues."
            ),
            LayerID.HL_6: (
                "You are an integration expert. "
                "Focus on API contracts, data consistency, "
                "error propagation, and system boundaries."
            ),
            LayerID.HL_7: (
                "You are performing final human review. "
                "Look for anything that previous reviews might have missed. "
                "Consider the full context and overall quality."
            ),
        }
        return base_prompts.get(layer_id, base_prompts[LayerID.HL_7])

    def _build_prompt(self, context: RunContext) -> str:
        """Constrói prompt a partir do contexto."""
        parts = []

        # Descrição do alvo
        parts.append(f"## Target\n{context.target_description}")

        # Código se disponível
        if context.code_snippet:
            parts.append(f"## Code\n```\n{context.code_snippet}\n```")

        # Perspectiva se especificada
        if context.perspective:
            parts.append(
                f"## Perspective\n"
                f"Analyze from the perspective of: {context.perspective}"
            )

        # Findings anteriores
        if context.previous_findings:
            findings_text = "\n".join(
                f"- [{f.get('severity', 'medium')}] {f.get('title', 'No title')}"
                for f in context.previous_findings
            )
            parts.append(f"## Previous Findings\n{findings_text}")

        # Contexto extra
        if context.extra_context:
            extra = "\n".join(
                f"- {k}: {v}"
                for k, v in context.extra_context.items()
            )
            parts.append(f"## Additional Context\n{extra}")

        # Instruções de output
        parts.append(
            "## Instructions\n"
            "Analyze the target and report findings in JSON format:\n"
            "```json\n"
            "{\n"
            '  "findings": [\n'
            '    {\n'
            '      "id": "unique-id",\n'
            '      "category": "category",\n'
            '      "severity": "critical|high|medium|low|info",\n'
            '      "title": "Short title",\n'
            '      "description": "Detailed description",\n'
            '      "evidence": "Code or element causing issue",\n'
            '      "remediation": "How to fix"\n'
            '    }\n'
            '  ],\n'
            '  "veto_level": "NONE|WEAK|MEDIUM|STRONG",\n'
            '  "confidence": 0.0-1.0,\n'
            '  "summary": "Brief summary"\n'
            "}\n"
            "```"
        )

        return "\n\n".join(parts)

    async def run(
        self,
        context: RunContext,
        timeout: Optional[int] = None,
    ) -> RunResult:
        """
        Executa uma layer.

        Args:
            context: Contexto de execução
            timeout: Timeout em ms (opcional)

        Returns:
            RunResult com findings e métricas
        """
        start = datetime.utcnow()

        logger.info(
            f"Executando {context.layer_id.value} "
            f"(run_id={context.run_id})"
        )

        try:
            # Monta prompts
            system_prompt = self._get_system_prompt(context.layer_id)
            user_prompt = self._build_prompt(context)

            # Chama LLM
            raw_response = await self.llm_client.complete(
                prompt=user_prompt,
                system_prompt=system_prompt,
            )

            # Parseia resposta
            parsed = self.response_parser.parse(raw_response)

            duration = (datetime.utcnow() - start).total_seconds() * 1000

            result = RunResult(
                context=context,
                success=True,
                findings=parsed.get("findings", []),
                raw_response=raw_response,
                veto_level=parsed.get("veto_level"),
                confidence=parsed.get("confidence", 0.5),
                duration_ms=duration,
            )

            logger.info(
                f"{context.layer_id.value} concluído: "
                f"{result.finding_count} findings, "
                f"{duration:.0f}ms"
            )

            return result

        except Exception as e:
            duration = (datetime.utcnow() - start).total_seconds() * 1000

            logger.error(f"Erro em {context.layer_id.value}: {e}")

            return RunResult(
                context=context,
                success=False,
                error=str(e),
                duration_ms=duration,
            )

    async def run_with_screenshot(
        self,
        context: RunContext,
        page: Any,  # Playwright Page
    ) -> RunResult:
        """
        Executa layer com captura de screenshot.

        Args:
            context: Contexto de execução
            page: Página Playwright para screenshot

        Returns:
            RunResult com screenshot anexado
        """
        from pathlib import Path
        import tempfile

        # Captura screenshot
        with tempfile.NamedTemporaryFile(
            suffix=".png",
            delete=False,
        ) as f:
            screenshot_path = f.name

        await page.screenshot(path=screenshot_path)

        # Atualiza contexto
        context.screenshot_path = screenshot_path

        # Executa layer
        return await self.run(context)


# Factory function
def create_runner(
    llm_client: LLMClientProtocol,
    response_parser: ResponseParserProtocol,
) -> HumanLayerRunner:
    """
    Factory para criar HumanLayerRunner.

    Args:
        llm_client: Cliente LLM
        response_parser: Parser de respostas

    Returns:
        HumanLayerRunner configurado
    """
    return HumanLayerRunner(
        llm_client=llm_client,
        response_parser=response_parser,
    )


# Exports
__all__ = [
    "LayerID",
    "RunContext",
    "RunResult",
    "LLMClientProtocol",
    "ResponseParserProtocol",
    "HumanLayerRunner",
    "create_runner",
]
```

---

## Módulo HLS-ENG-002: TripleRedundancy

```
ID: HLS-ENG-002
Nome: TripleRedundancy
Caminho: src/hl_mcp/engine/redundancy.py
Dependências: HLS-ENG-001
Exports: TripleRedundancy, RedundancyResult, ConsensusLevel
Linhas: ~280
```

### Código

```python
"""
HLS-ENG-002: TripleRedundancy
=============================

Execução tripla com consenso 2/3.

Responsabilidades:
- Executar mesma layer 3 vezes
- Aplicar consenso 2/3 nos findings
- Detectar discrepâncias
- Calcular confiança baseada em acordo

O padrão de redundância tripla garante que:
- Findings aparecem em pelo menos 2 de 3 runs
- Vetos são confirmados por maioria
- Discrepâncias são sinalizadas para review

Exemplo:
    >>> redundancy = TripleRedundancy(runner)
    >>> result = await redundancy.execute(context)
    >>> print(f"Consenso: {result.consensus_level}")
    >>> print(f"Findings confirmados: {len(result.confirmed_findings)}")

Changelog:
    - v1.0.0 (2026-02-01): Implementação inicial
"""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import logging

from .runner import HumanLayerRunner, RunContext, RunResult

logger = logging.getLogger(__name__)


class ConsensusLevel(str, Enum):
    """Níveis de consenso."""

    UNANIMOUS = "unanimous"  # 3/3 concordam
    MAJORITY = "majority"    # 2/3 concordam
    SPLIT = "split"          # Sem consenso claro
    FAILED = "failed"        # Execuções falharam


@dataclass
class RedundancyResult:
    """
    Resultado de execução com redundância tripla.

    Attributes:
        context: Contexto base usado
        consensus_level: Nível de consenso alcançado
        confirmed_findings: Findings com 2/3 consenso
        disputed_findings: Findings sem consenso
        run_results: Resultados das 3 execuções
        final_veto: Veto por maioria
        confidence: Confiança final
        duration_ms: Duração total
    """

    context: RunContext
    consensus_level: ConsensusLevel
    confirmed_findings: list[dict] = field(default_factory=list)
    disputed_findings: list[dict] = field(default_factory=list)
    run_results: list[RunResult] = field(default_factory=list)
    final_veto: Optional[str] = None
    confidence: float = 0.0
    duration_ms: float = 0

    @property
    def unanimous_findings(self) -> list[dict]:
        """Findings presentes em todas as 3 execuções."""
        return [
            f for f in self.confirmed_findings
            if f.get("_consensus_count", 0) == 3
        ]

    @property
    def success_rate(self) -> float:
        """Taxa de sucesso das execuções."""
        if not self.run_results:
            return 0.0
        successful = sum(1 for r in self.run_results if r.success)
        return successful / len(self.run_results)

    def to_dict(self) -> dict:
        """Converte para dicionário."""
        return {
            "layer_id": self.context.layer_id.value,
            "consensus_level": self.consensus_level.value,
            "confirmed_count": len(self.confirmed_findings),
            "disputed_count": len(self.disputed_findings),
            "unanimous_count": len(self.unanimous_findings),
            "final_veto": self.final_veto,
            "confidence": self.confidence,
            "success_rate": self.success_rate,
            "duration_ms": self.duration_ms,
        }


class TripleRedundancy:
    """
    Executor com redundância tripla.

    Executa a mesma layer 3 vezes e aplica consenso 2/3
    para confirmar findings e vetos.

    Example:
        >>> runner = HumanLayerRunner(llm, parser)
        >>> redundancy = TripleRedundancy(runner)
        >>> result = await redundancy.execute(context)
        >>> if result.consensus_level == ConsensusLevel.UNANIMOUS:
        ...     print("Alta confiança nos resultados")
    """

    def __init__(
        self,
        runner: HumanLayerRunner,
        parallel: bool = True,
        min_successful_runs: int = 2,
    ):
        """
        Inicializa o executor.

        Args:
            runner: HumanLayerRunner para executar layers
            parallel: Executar runs em paralelo
            min_successful_runs: Mínimo de runs bem-sucedidos
        """
        self.runner = runner
        self.parallel = parallel
        self.min_successful_runs = min_successful_runs

    async def execute(
        self,
        context: RunContext,
    ) -> RedundancyResult:
        """
        Executa layer 3 vezes com consenso.

        Args:
            context: Contexto de execução

        Returns:
            RedundancyResult com findings confirmados
        """
        start = datetime.utcnow()

        logger.info(
            f"Iniciando redundância tripla para {context.layer_id.value}"
        )

        # Executa 3 runs
        if self.parallel:
            run_results = await self._execute_parallel(context)
        else:
            run_results = await self._execute_sequential(context)

        duration = (datetime.utcnow() - start).total_seconds() * 1000

        # Verifica se temos runs suficientes
        successful_runs = [r for r in run_results if r.success]
        if len(successful_runs) < self.min_successful_runs:
            logger.warning(
                f"Apenas {len(successful_runs)}/{len(run_results)} "
                f"runs bem-sucedidos"
            )
            return RedundancyResult(
                context=context,
                consensus_level=ConsensusLevel.FAILED,
                run_results=run_results,
                duration_ms=duration,
            )

        # Aplica consenso
        confirmed, disputed = self._apply_consensus(successful_runs)

        # Determina veto por maioria
        final_veto = self._get_majority_veto(successful_runs)

        # Determina nível de consenso
        consensus_level = self._determine_consensus_level(
            confirmed, disputed, successful_runs
        )

        # Calcula confiança
        confidence = self._calculate_confidence(
            consensus_level, confirmed, disputed
        )

        logger.info(
            f"Redundância concluída: {consensus_level.value}, "
            f"{len(confirmed)} confirmados, {len(disputed)} disputados"
        )

        return RedundancyResult(
            context=context,
            consensus_level=consensus_level,
            confirmed_findings=confirmed,
            disputed_findings=disputed,
            run_results=run_results,
            final_veto=final_veto,
            confidence=confidence,
            duration_ms=duration,
        )

    async def _execute_parallel(
        self,
        context: RunContext,
    ) -> list[RunResult]:
        """Executa 3 runs em paralelo."""
        tasks = [
            self.runner.run(
                RunContext(
                    layer_id=context.layer_id,
                    target_description=context.target_description,
                    code_snippet=context.code_snippet,
                    screenshot_path=context.screenshot_path,
                    journey_result=context.journey_result,
                    previous_findings=context.previous_findings,
                    perspective=context.perspective,
                    extra_context=context.extra_context,
                )
            )
            for _ in range(3)
        ]

        return await asyncio.gather(*tasks)

    async def _execute_sequential(
        self,
        context: RunContext,
    ) -> list[RunResult]:
        """Executa 3 runs sequencialmente."""
        results = []
        for i in range(3):
            result = await self.runner.run(context)
            results.append(result)
        return results

    def _apply_consensus(
        self,
        results: list[RunResult],
    ) -> tuple[list[dict], list[dict]]:
        """
        Aplica consenso 2/3 nos findings.

        Returns:
            Tuple (confirmed_findings, disputed_findings)
        """
        # Coleta todos os findings
        all_findings: list[dict] = []
        for result in results:
            all_findings.extend(result.findings)

        # Agrupa por similaridade (usando título + categoria)
        finding_groups: dict[str, list[dict]] = {}
        for finding in all_findings:
            key = self._finding_key(finding)
            if key not in finding_groups:
                finding_groups[key] = []
            finding_groups[key].append(finding)

        # Separa confirmados (2+) de disputados (1)
        confirmed = []
        disputed = []

        for key, findings in finding_groups.items():
            count = len(findings)
            # Usa o finding mais completo do grupo
            best_finding = max(
                findings,
                key=lambda f: len(f.get("description", ""))
            )
            best_finding["_consensus_count"] = count

            if count >= 2:
                confirmed.append(best_finding)
            else:
                disputed.append(best_finding)

        return confirmed, disputed

    def _finding_key(self, finding: dict) -> str:
        """Gera chave única para agrupar findings similares."""
        title = finding.get("title", "").lower().strip()
        category = finding.get("category", "").lower().strip()
        severity = finding.get("severity", "medium")
        return f"{category}::{severity}::{title[:50]}"

    def _get_majority_veto(self, results: list[RunResult]) -> Optional[str]:
        """Obtém veto por maioria."""
        vetos = [r.veto_level for r in results if r.veto_level]

        if not vetos:
            return None

        counter = Counter(vetos)
        most_common = counter.most_common(1)

        if most_common and most_common[0][1] >= 2:
            return most_common[0][0]

        return None

    def _determine_consensus_level(
        self,
        confirmed: list[dict],
        disputed: list[dict],
        results: list[RunResult],
    ) -> ConsensusLevel:
        """Determina nível de consenso."""
        if not confirmed and not disputed:
            # Sem findings = consenso unânime de "OK"
            return ConsensusLevel.UNANIMOUS

        # Verifica se todos os findings confirmados são unânimes
        all_unanimous = all(
            f.get("_consensus_count", 0) == 3
            for f in confirmed
        )

        if all_unanimous and not disputed:
            return ConsensusLevel.UNANIMOUS

        if confirmed:
            return ConsensusLevel.MAJORITY

        return ConsensusLevel.SPLIT

    def _calculate_confidence(
        self,
        consensus_level: ConsensusLevel,
        confirmed: list[dict],
        disputed: list[dict],
    ) -> float:
        """Calcula confiança baseada no consenso."""
        base_confidence = {
            ConsensusLevel.UNANIMOUS: 0.95,
            ConsensusLevel.MAJORITY: 0.75,
            ConsensusLevel.SPLIT: 0.50,
            ConsensusLevel.FAILED: 0.10,
        }

        confidence = base_confidence.get(consensus_level, 0.5)

        # Penaliza por disputas
        if disputed:
            dispute_ratio = len(disputed) / (len(confirmed) + len(disputed) + 1)
            confidence *= (1 - dispute_ratio * 0.3)

        # Bonifica por unanimidade
        unanimous_count = sum(
            1 for f in confirmed if f.get("_consensus_count") == 3
        )
        if confirmed and unanimous_count == len(confirmed):
            confidence = min(confidence * 1.1, 0.99)

        return round(confidence, 3)


# Exports
__all__ = [
    "ConsensusLevel",
    "RedundancyResult",
    "TripleRedundancy",
]
```

---

## Módulo HLS-ENG-003: LayerOrchestrator

```
ID: HLS-ENG-003
Nome: LayerOrchestrator
Caminho: src/hl_mcp/engine/orchestrator.py
Dependências: HLS-ENG-001, HLS-ENG-002, HLS-ENG-004
Exports: LayerOrchestrator, OrchestratorConfig, OrchestrationResult
Linhas: ~400
```

### Código

```python
"""
HLS-ENG-003: LayerOrchestrator
==============================

Orquestra execução das 7 Human Layers.

Responsabilidades:
- Executar layers em sequência
- Aplicar vetos entre layers
- Parar se veto STRONG
- Propagar findings entre layers
- Gerar relatório final

O Orchestrator coordena todo o fluxo de validação:
HL-1 → HL-2 → ... → HL-7, parando em vetos STRONG.

Exemplo:
    >>> config = OrchestratorConfig(
    ...     layers=[LayerID.HL_1, LayerID.HL_2, LayerID.HL_3],
    ...     use_redundancy=True,
    ... )
    >>> orchestrator = LayerOrchestrator(runner, config)
    >>> result = await orchestrator.run(target_description="Login form")
    >>> print(f"Final verdict: {result.final_verdict}")

Changelog:
    - v1.0.0 (2026-02-01): Implementação inicial
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional
import logging

from .runner import HumanLayerRunner, LayerID, RunContext, RunResult
from .redundancy import TripleRedundancy, RedundancyResult, ConsensusLevel

logger = logging.getLogger(__name__)


class FinalVerdict(str, Enum):
    """Veredicto final da orquestração."""

    APPROVED = "approved"       # Nenhum veto forte
    REJECTED = "rejected"       # Veto STRONG
    NEEDS_REVIEW = "needs_review"  # Vetos MEDIUM
    INCOMPLETE = "incomplete"   # Execução não completou


class VetoLevel(str, Enum):
    """Níveis de veto."""

    NONE = "NONE"
    WEAK = "WEAK"       # Aviso, continua
    MEDIUM = "MEDIUM"   # Atenção, continua
    STRONG = "STRONG"   # Bloqueia, para execução


@dataclass
class OrchestratorConfig:
    """
    Configuração do orquestrador.

    Attributes:
        layers: Lista de layers a executar
        use_redundancy: Usar redundância tripla
        stop_on_strong_veto: Parar em veto STRONG
        propagate_findings: Passar findings para próxima layer
        parallel_independent: Executar layers independentes em paralelo
        max_duration_ms: Tempo máximo de execução
    """

    layers: list[LayerID] = field(
        default_factory=lambda: list(LayerID)
    )
    use_redundancy: bool = True
    stop_on_strong_veto: bool = True
    propagate_findings: bool = True
    parallel_independent: bool = False
    max_duration_ms: int = 600000  # 10 min

    @classmethod
    def quick_check(cls) -> "OrchestratorConfig":
        """Config para verificação rápida (HL-1, HL-2, HL-3)."""
        return cls(
            layers=[LayerID.HL_1, LayerID.HL_2, LayerID.HL_3],
            use_redundancy=False,
        )

    @classmethod
    def full_check(cls) -> "OrchestratorConfig":
        """Config para verificação completa (todas as layers)."""
        return cls(
            layers=list(LayerID),
            use_redundancy=True,
        )

    @classmethod
    def security_focus(cls) -> "OrchestratorConfig":
        """Config focada em segurança."""
        return cls(
            layers=[LayerID.HL_2, LayerID.HL_3, LayerID.HL_6],
            use_redundancy=True,
        )


@dataclass
class LayerExecutionResult:
    """Resultado de execução de uma layer."""

    layer_id: LayerID
    success: bool
    findings: list[dict] = field(default_factory=list)
    veto_level: VetoLevel = VetoLevel.NONE
    confidence: float = 0.0
    duration_ms: float = 0
    redundancy_result: Optional[RedundancyResult] = None
    run_result: Optional[RunResult] = None
    stopped_execution: bool = False


@dataclass
class OrchestrationResult:
    """
    Resultado completo da orquestração.

    Attributes:
        final_verdict: Veredicto final
        all_findings: Todos os findings coletados
        layer_results: Resultados por layer
        total_duration_ms: Duração total
        layers_executed: Número de layers executadas
        stopped_at: Layer onde parou (se parou)
        veto_summary: Resumo de vetos por layer
    """

    final_verdict: FinalVerdict
    all_findings: list[dict] = field(default_factory=list)
    layer_results: list[LayerExecutionResult] = field(default_factory=list)
    total_duration_ms: float = 0
    layers_executed: int = 0
    stopped_at: Optional[LayerID] = None
    veto_summary: dict = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None

    @property
    def critical_findings(self) -> list[dict]:
        """Findings críticos."""
        return [
            f for f in self.all_findings
            if f.get("severity") in ["critical", "high"]
        ]

    @property
    def has_blocking_issues(self) -> bool:
        """Se há issues que bloqueiam."""
        return self.final_verdict == FinalVerdict.REJECTED

    def to_dict(self) -> dict:
        """Converte para dicionário."""
        return {
            "final_verdict": self.final_verdict.value,
            "total_findings": len(self.all_findings),
            "critical_findings": len(self.critical_findings),
            "layers_executed": self.layers_executed,
            "stopped_at": self.stopped_at.value if self.stopped_at else None,
            "total_duration_ms": self.total_duration_ms,
            "veto_summary": self.veto_summary,
        }


class LayerOrchestrator:
    """
    Orquestrador das 7 Human Layers.

    Executa layers em sequência, aplica vetos, e gera
    relatório final consolidado.

    Example:
        >>> orchestrator = LayerOrchestrator(runner)
        >>> result = await orchestrator.run("Login form validation")
        >>> if result.final_verdict == FinalVerdict.APPROVED:
        ...     print("Ready for deployment!")
    """

    def __init__(
        self,
        runner: HumanLayerRunner,
        config: Optional[OrchestratorConfig] = None,
    ):
        """
        Inicializa o orquestrador.

        Args:
            runner: HumanLayerRunner para execução
            config: Configuração do orquestrador
        """
        self.runner = runner
        self.config = config or OrchestratorConfig()
        self._redundancy = TripleRedundancy(runner) if config and config.use_redundancy else None

    async def run(
        self,
        target_description: str,
        code_snippet: Optional[str] = None,
        extra_context: Optional[dict] = None,
        on_layer_complete: Optional[Callable[[LayerExecutionResult], None]] = None,
    ) -> OrchestrationResult:
        """
        Executa orquestração completa.

        Args:
            target_description: Descrição do alvo
            code_snippet: Código a analisar
            extra_context: Contexto adicional
            on_layer_complete: Callback após cada layer

        Returns:
            OrchestrationResult com todos os resultados
        """
        start_time = datetime.utcnow()

        logger.info(
            f"Iniciando orquestração: {len(self.config.layers)} layers, "
            f"redundancy={self.config.use_redundancy}"
        )

        layer_results: list[LayerExecutionResult] = []
        all_findings: list[dict] = []
        veto_summary: dict[str, str] = {}
        stopped_at: Optional[LayerID] = None
        previous_findings: list[dict] = []

        for layer_id in self.config.layers:
            # Verifica timeout
            elapsed = (datetime.utcnow() - start_time).total_seconds() * 1000
            if elapsed > self.config.max_duration_ms:
                logger.warning("Timeout atingido")
                stopped_at = layer_id
                break

            # Monta contexto
            context = RunContext(
                layer_id=layer_id,
                target_description=target_description,
                code_snippet=code_snippet,
                previous_findings=previous_findings if self.config.propagate_findings else [],
                extra_context=extra_context or {},
            )

            # Executa layer
            layer_result = await self._execute_layer(context)
            layer_results.append(layer_result)

            # Callback
            if on_layer_complete:
                on_layer_complete(layer_result)

            # Coleta findings
            all_findings.extend(layer_result.findings)
            previous_findings = all_findings.copy()

            # Registra veto
            veto_summary[layer_id.value] = layer_result.veto_level.value

            # Verifica se deve parar
            if (
                self.config.stop_on_strong_veto and
                layer_result.veto_level == VetoLevel.STRONG
            ):
                logger.warning(
                    f"STRONG veto em {layer_id.value}, parando execução"
                )
                layer_result.stopped_execution = True
                stopped_at = layer_id
                break

        end_time = datetime.utcnow()
        total_duration = (end_time - start_time).total_seconds() * 1000

        # Determina veredicto final
        final_verdict = self._determine_verdict(layer_results, stopped_at)

        logger.info(
            f"Orquestração concluída: {final_verdict.value}, "
            f"{len(all_findings)} findings, {total_duration:.0f}ms"
        )

        return OrchestrationResult(
            final_verdict=final_verdict,
            all_findings=all_findings,
            layer_results=layer_results,
            total_duration_ms=total_duration,
            layers_executed=len(layer_results),
            stopped_at=stopped_at,
            veto_summary=veto_summary,
            start_time=start_time,
            end_time=end_time,
        )

    async def _execute_layer(
        self,
        context: RunContext,
    ) -> LayerExecutionResult:
        """Executa uma layer individual."""
        start = datetime.utcnow()

        logger.info(f"Executando {context.layer_id.value}...")

        try:
            if self.config.use_redundancy and self._redundancy:
                # Execução com redundância
                redundancy_result = await self._redundancy.execute(context)

                duration = (datetime.utcnow() - start).total_seconds() * 1000

                return LayerExecutionResult(
                    layer_id=context.layer_id,
                    success=redundancy_result.consensus_level != ConsensusLevel.FAILED,
                    findings=redundancy_result.confirmed_findings,
                    veto_level=VetoLevel(redundancy_result.final_veto or "NONE"),
                    confidence=redundancy_result.confidence,
                    duration_ms=duration,
                    redundancy_result=redundancy_result,
                )
            else:
                # Execução simples
                run_result = await self.runner.run(context)

                duration = (datetime.utcnow() - start).total_seconds() * 1000

                return LayerExecutionResult(
                    layer_id=context.layer_id,
                    success=run_result.success,
                    findings=run_result.findings,
                    veto_level=VetoLevel(run_result.veto_level or "NONE"),
                    confidence=run_result.confidence,
                    duration_ms=duration,
                    run_result=run_result,
                )

        except Exception as e:
            duration = (datetime.utcnow() - start).total_seconds() * 1000
            logger.error(f"Erro em {context.layer_id.value}: {e}")

            return LayerExecutionResult(
                layer_id=context.layer_id,
                success=False,
                duration_ms=duration,
            )

    def _determine_verdict(
        self,
        results: list[LayerExecutionResult],
        stopped_at: Optional[LayerID],
    ) -> FinalVerdict:
        """Determina veredicto final."""
        if stopped_at:
            # Parou por veto STRONG
            return FinalVerdict.REJECTED

        if not results:
            return FinalVerdict.INCOMPLETE

        # Verifica se alguma layer falhou
        failed = [r for r in results if not r.success]
        if len(failed) > len(results) / 2:
            return FinalVerdict.INCOMPLETE

        # Conta vetos
        medium_vetos = sum(
            1 for r in results if r.veto_level == VetoLevel.MEDIUM
        )
        strong_vetos = sum(
            1 for r in results if r.veto_level == VetoLevel.STRONG
        )

        if strong_vetos > 0:
            return FinalVerdict.REJECTED

        if medium_vetos >= 2:
            return FinalVerdict.NEEDS_REVIEW

        return FinalVerdict.APPROVED

    async def run_quick_check(
        self,
        target_description: str,
        code_snippet: Optional[str] = None,
    ) -> OrchestrationResult:
        """Atalho para verificação rápida."""
        original_config = self.config
        self.config = OrchestratorConfig.quick_check()

        try:
            return await self.run(target_description, code_snippet)
        finally:
            self.config = original_config


# Exports
__all__ = [
    "FinalVerdict",
    "VetoLevel",
    "OrchestratorConfig",
    "LayerExecutionResult",
    "OrchestrationResult",
    "LayerOrchestrator",
]
```

---

## Módulo HLS-ENG-004: VetoGate

```
ID: HLS-ENG-004
Nome: VetoGate
Caminho: src/hl_mcp/engine/veto.py
Dependências: HLS-MDL-001, HLS-MDL-002
Exports: VetoGate, VetoDecision, VetoPolicy
Linhas: ~220
```

### Código

```python
"""
HLS-ENG-004: VetoGate
=====================

Lógica de aplicação de vetos.

Responsabilidades:
- Avaliar findings e determinar nível de veto
- Aplicar políticas de veto configuráveis
- Suportar override de vetos
- Registrar decisões para auditoria

O VetoGate é o gatekeeper que decide se um resultado
deve bloquear, alertar, ou passar.

Exemplo:
    >>> gate = VetoGate(VetoPolicy.STRICT)
    >>> decision = gate.evaluate(findings)
    >>> if decision.should_block:
    ...     print(f"Bloqueado: {decision.reason}")

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


class VetoLevel(str, Enum):
    """Níveis de veto."""

    NONE = "NONE"
    WEAK = "WEAK"       # Log/warning
    MEDIUM = "MEDIUM"   # Atenção humana
    STRONG = "STRONG"   # Bloqueio


class VetoPolicy(str, Enum):
    """Políticas de veto pré-definidas."""

    PERMISSIVE = "permissive"  # Só bloqueia critical
    STANDARD = "standard"      # Bloqueia critical/high
    STRICT = "strict"          # Bloqueia qualquer issue
    SECURITY = "security"      # Foco em security findings


@dataclass
class VetoDecision:
    """
    Decisão de veto.

    Attributes:
        level: Nível de veto aplicado
        reason: Razão do veto
        blocking_findings: Findings que causaram bloqueio
        warning_findings: Findings que geraram avisos
        can_override: Se pode ser sobrescrito
        override_requires: O que é necessário para override
        timestamp: Momento da decisão
    """

    level: VetoLevel
    reason: str = ""
    blocking_findings: list[dict] = field(default_factory=list)
    warning_findings: list[dict] = field(default_factory=list)
    can_override: bool = True
    override_requires: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def should_block(self) -> bool:
        """Se deve bloquear a ação."""
        return self.level == VetoLevel.STRONG

    @property
    def needs_attention(self) -> bool:
        """Se precisa de atenção humana."""
        return self.level in [VetoLevel.MEDIUM, VetoLevel.STRONG]

    def to_dict(self) -> dict:
        """Converte para dicionário."""
        return {
            "level": self.level.value,
            "should_block": self.should_block,
            "needs_attention": self.needs_attention,
            "reason": self.reason,
            "blocking_count": len(self.blocking_findings),
            "warning_count": len(self.warning_findings),
            "can_override": self.can_override,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PolicyConfig:
    """Configuração de política de veto."""

    block_on_critical: bool = True
    block_on_high: bool = True
    block_on_security: bool = True
    medium_on_warning: bool = True
    min_confidence_for_block: float = 0.7
    max_warnings_before_block: int = 5


class VetoGate:
    """
    Gate de aplicação de vetos.

    Avalia findings e aplica política de veto configurada.

    Example:
        >>> gate = VetoGate(policy=VetoPolicy.STRICT)
        >>> findings = [
        ...     {"severity": "critical", "title": "SQL Injection"},
        ...     {"severity": "low", "title": "Typo in label"},
        ... ]
        >>> decision = gate.evaluate(findings)
        >>> print(decision.level)  # VetoLevel.STRONG
    """

    # Configurações por política
    POLICY_CONFIGS = {
        VetoPolicy.PERMISSIVE: PolicyConfig(
            block_on_critical=True,
            block_on_high=False,
            block_on_security=True,
            medium_on_warning=False,
        ),
        VetoPolicy.STANDARD: PolicyConfig(
            block_on_critical=True,
            block_on_high=True,
            block_on_security=True,
            medium_on_warning=True,
        ),
        VetoPolicy.STRICT: PolicyConfig(
            block_on_critical=True,
            block_on_high=True,
            block_on_security=True,
            medium_on_warning=True,
            max_warnings_before_block=3,
        ),
        VetoPolicy.SECURITY: PolicyConfig(
            block_on_critical=True,
            block_on_high=True,
            block_on_security=True,
            medium_on_warning=True,
            min_confidence_for_block=0.5,
        ),
    }

    def __init__(
        self,
        policy: VetoPolicy = VetoPolicy.STANDARD,
        custom_config: Optional[PolicyConfig] = None,
    ):
        """
        Inicializa o gate.

        Args:
            policy: Política de veto
            custom_config: Configuração customizada (sobrescreve policy)
        """
        self.policy = policy
        self.config = custom_config or self.POLICY_CONFIGS.get(
            policy, PolicyConfig()
        )

    def evaluate(
        self,
        findings: list[dict],
        confidence: float = 1.0,
    ) -> VetoDecision:
        """
        Avalia findings e retorna decisão de veto.

        Args:
            findings: Lista de findings a avaliar
            confidence: Confiança nos findings (0-1)

        Returns:
            VetoDecision com nível e detalhes
        """
        if not findings:
            return VetoDecision(
                level=VetoLevel.NONE,
                reason="No findings",
            )

        blocking = []
        warnings = []

        for finding in findings:
            severity = finding.get("severity", "medium").lower()
            category = finding.get("category", "").lower()
            is_security = category in ["security", "vulnerability", "injection"]

            # Verifica se bloqueia
            should_block = False

            if severity == "critical" and self.config.block_on_critical:
                should_block = True
            elif severity == "high" and self.config.block_on_high:
                should_block = True
            elif is_security and self.config.block_on_security:
                should_block = True

            # Aplica threshold de confiança
            if should_block and confidence < self.config.min_confidence_for_block:
                should_block = False

            if should_block:
                blocking.append(finding)
            else:
                warnings.append(finding)

        # Determina nível
        if blocking:
            return VetoDecision(
                level=VetoLevel.STRONG,
                reason=f"{len(blocking)} blocking issue(s) found",
                blocking_findings=blocking,
                warning_findings=warnings,
                can_override=False,
            )

        if len(warnings) > self.config.max_warnings_before_block:
            return VetoDecision(
                level=VetoLevel.STRONG,
                reason=f"Too many warnings ({len(warnings)})",
                blocking_findings=[],
                warning_findings=warnings,
                can_override=True,
                override_requires="Senior approval",
            )

        if warnings and self.config.medium_on_warning:
            return VetoDecision(
                level=VetoLevel.MEDIUM,
                reason=f"{len(warnings)} warning(s) need attention",
                blocking_findings=[],
                warning_findings=warnings,
            )

        if warnings:
            return VetoDecision(
                level=VetoLevel.WEAK,
                reason=f"{len(warnings)} minor issue(s)",
                warning_findings=warnings,
            )

        return VetoDecision(
            level=VetoLevel.NONE,
            reason="All findings are acceptable",
        )

    def can_proceed(
        self,
        decision: VetoDecision,
        has_override: bool = False,
    ) -> bool:
        """
        Verifica se pode prosseguir dado a decisão.

        Args:
            decision: Decisão de veto
            has_override: Se tem override autorizado

        Returns:
            True se pode prosseguir
        """
        if decision.level == VetoLevel.STRONG:
            return has_override and decision.can_override

        return True


# Factory functions
def create_permissive_gate() -> VetoGate:
    """Cria gate permissivo."""
    return VetoGate(VetoPolicy.PERMISSIVE)


def create_strict_gate() -> VetoGate:
    """Cria gate estrito."""
    return VetoGate(VetoPolicy.STRICT)


def create_security_gate() -> VetoGate:
    """Cria gate focado em segurança."""
    return VetoGate(VetoPolicy.SECURITY)


# Exports
__all__ = [
    "VetoLevel",
    "VetoPolicy",
    "VetoDecision",
    "PolicyConfig",
    "VetoGate",
    "create_permissive_gate",
    "create_strict_gate",
    "create_security_gate",
]
```

---

## Módulo HLS-ENG-005: ConsensusEngine

```
ID: HLS-ENG-005
Nome: ConsensusEngine
Caminho: src/hl_mcp/engine/consensus.py
Dependências: HLS-MDL-001
Exports: ConsensusEngine, ConsensusResult, MergeStrategy
Linhas: ~250
```

### Código

```python
"""
HLS-ENG-005: ConsensusEngine
============================

Engine de consenso para múltiplas execuções.

Responsabilidades:
- Agrupar findings similares de múltiplas fontes
- Aplicar estratégias de merge
- Calcular confiança agregada
- Detectar contradições

Usado para consolidar resultados de redundância tripla
e de múltiplas perspectivas.

Exemplo:
    >>> engine = ConsensusEngine()
    >>> results = [run1_findings, run2_findings, run3_findings]
    >>> consensus = engine.consolidate(results)
    >>> print(f"Confirmados: {len(consensus.confirmed)}")

Changelog:
    - v1.0.0 (2026-02-01): Implementação inicial
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional
import logging
import hashlib

logger = logging.getLogger(__name__)


class MergeStrategy(str, Enum):
    """Estratégias de merge de findings."""

    MAJORITY = "majority"      # 2/3 para confirmar
    UNANIMOUS = "unanimous"    # Todos devem concordar
    ANY = "any"                # Qualquer ocorrência confirma
    WEIGHTED = "weighted"      # Peso por confiança


@dataclass
class ConsolidatedFinding:
    """
    Finding consolidado de múltiplas fontes.

    Attributes:
        finding: O finding consolidado
        sources: Número de fontes que reportaram
        total_sources: Total de fontes
        confidence: Confiança calculada
        variations: Variações encontradas
    """

    finding: dict
    sources: int
    total_sources: int
    confidence: float = 0.0
    variations: list[dict] = field(default_factory=list)

    @property
    def is_unanimous(self) -> bool:
        """Se todas as fontes reportaram."""
        return self.sources == self.total_sources

    @property
    def agreement_ratio(self) -> float:
        """Taxa de concordância."""
        if self.total_sources == 0:
            return 0.0
        return self.sources / self.total_sources


@dataclass
class ConsensusResult:
    """
    Resultado de consolidação de consenso.

    Attributes:
        confirmed: Findings confirmados
        disputed: Findings sem consenso
        rejected: Findings rejeitados
        contradictions: Contradições encontradas
        overall_confidence: Confiança geral
    """

    confirmed: list[ConsolidatedFinding] = field(default_factory=list)
    disputed: list[ConsolidatedFinding] = field(default_factory=list)
    rejected: list[ConsolidatedFinding] = field(default_factory=list)
    contradictions: list[dict] = field(default_factory=list)
    overall_confidence: float = 0.0

    @property
    def confirmed_findings(self) -> list[dict]:
        """Retorna apenas os findings confirmados."""
        return [cf.finding for cf in self.confirmed]

    def to_dict(self) -> dict:
        """Converte para dicionário."""
        return {
            "confirmed_count": len(self.confirmed),
            "disputed_count": len(self.disputed),
            "rejected_count": len(self.rejected),
            "contradiction_count": len(self.contradictions),
            "overall_confidence": self.overall_confidence,
        }


class ConsensusEngine:
    """
    Engine de consenso para consolidar findings.

    Agrupa findings similares de múltiplas execuções
    e determina quais são confirmados baseado na
    estratégia de merge.

    Example:
        >>> engine = ConsensusEngine(
        ...     strategy=MergeStrategy.MAJORITY,
        ...     min_threshold=2,
        ... )
        >>> results = [run1, run2, run3]  # Cada um é list[dict]
        >>> consensus = engine.consolidate(results)
    """

    def __init__(
        self,
        strategy: MergeStrategy = MergeStrategy.MAJORITY,
        min_threshold: int = 2,
        similarity_threshold: float = 0.8,
        custom_similarity_fn: Optional[Callable[[dict, dict], float]] = None,
    ):
        """
        Inicializa a engine.

        Args:
            strategy: Estratégia de merge
            min_threshold: Mínimo de fontes para confirmar
            similarity_threshold: Threshold de similaridade (0-1)
            custom_similarity_fn: Função customizada de similaridade
        """
        self.strategy = strategy
        self.min_threshold = min_threshold
        self.similarity_threshold = similarity_threshold
        self.similarity_fn = custom_similarity_fn or self._default_similarity

    def consolidate(
        self,
        finding_sets: list[list[dict]],
        weights: Optional[list[float]] = None,
    ) -> ConsensusResult:
        """
        Consolida findings de múltiplas fontes.

        Args:
            finding_sets: Lista de listas de findings
            weights: Pesos por fonte (para WEIGHTED strategy)

        Returns:
            ConsensusResult com findings consolidados
        """
        if not finding_sets:
            return ConsensusResult()

        total_sources = len(finding_sets)
        weights = weights or [1.0] * total_sources

        # Agrupa findings similares
        groups = self._group_similar_findings(finding_sets)

        # Classifica grupos
        confirmed = []
        disputed = []
        rejected = []

        for canonical, variants in groups.items():
            sources = len(variants)

            # Calcula confiança
            if self.strategy == MergeStrategy.WEIGHTED:
                confidence = self._weighted_confidence(variants, weights)
            else:
                confidence = sources / total_sources

            consolidated = ConsolidatedFinding(
                finding=self._merge_findings(variants),
                sources=sources,
                total_sources=total_sources,
                confidence=confidence,
                variations=variants,
            )

            # Classifica baseado na estratégia
            if self._is_confirmed(sources, total_sources, confidence):
                confirmed.append(consolidated)
            elif sources == 1:
                rejected.append(consolidated)
            else:
                disputed.append(consolidated)

        # Detecta contradições
        contradictions = self._detect_contradictions(finding_sets)

        # Calcula confiança geral
        overall_confidence = self._calculate_overall_confidence(
            confirmed, disputed, rejected
        )

        logger.info(
            f"Consenso: {len(confirmed)} confirmados, "
            f"{len(disputed)} disputados, {len(rejected)} rejeitados"
        )

        return ConsensusResult(
            confirmed=confirmed,
            disputed=disputed,
            rejected=rejected,
            contradictions=contradictions,
            overall_confidence=overall_confidence,
        )

    def _group_similar_findings(
        self,
        finding_sets: list[list[dict]],
    ) -> dict[str, list[dict]]:
        """Agrupa findings similares."""
        groups: dict[str, list[dict]] = defaultdict(list)

        for findings in finding_sets:
            for finding in findings:
                # Tenta encontrar grupo existente
                matched = False
                for canonical_key, group_findings in list(groups.items()):
                    if self._is_similar(finding, group_findings[0]):
                        groups[canonical_key].append(finding)
                        matched = True
                        break

                if not matched:
                    # Cria novo grupo
                    key = self._finding_key(finding)
                    groups[key].append(finding)

        return dict(groups)

    def _finding_key(self, finding: dict) -> str:
        """Gera chave canônica para finding."""
        title = finding.get("title", "").lower().strip()
        category = finding.get("category", "").lower()
        severity = finding.get("severity", "medium")

        raw = f"{category}|{severity}|{title[:50]}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def _default_similarity(self, f1: dict, f2: dict) -> float:
        """Calcula similaridade default entre dois findings."""
        score = 0.0

        # Categoria exata
        if f1.get("category") == f2.get("category"):
            score += 0.3

        # Severidade exata ou próxima
        s1 = f1.get("severity", "medium")
        s2 = f2.get("severity", "medium")
        if s1 == s2:
            score += 0.2

        # Título similar
        t1 = f1.get("title", "").lower()
        t2 = f2.get("title", "").lower()
        if t1 and t2:
            # Jaccard simplificado
            words1 = set(t1.split())
            words2 = set(t2.split())
            if words1 or words2:
                jaccard = len(words1 & words2) / len(words1 | words2)
                score += jaccard * 0.5

        return score

    def _is_similar(self, f1: dict, f2: dict) -> bool:
        """Verifica se dois findings são similares."""
        similarity = self.similarity_fn(f1, f2)
        return similarity >= self.similarity_threshold

    def _merge_findings(self, findings: list[dict]) -> dict:
        """Merge múltiplos findings similares em um."""
        if len(findings) == 1:
            return findings[0].copy()

        # Usa o finding mais completo como base
        base = max(findings, key=lambda f: len(str(f)))
        merged = base.copy()

        # Adiciona metadados de merge
        merged["_merged_from"] = len(findings)

        return merged

    def _is_confirmed(
        self,
        sources: int,
        total: int,
        confidence: float,
    ) -> bool:
        """Determina se um finding está confirmado."""
        if self.strategy == MergeStrategy.UNANIMOUS:
            return sources == total

        if self.strategy == MergeStrategy.ANY:
            return sources >= 1

        if self.strategy == MergeStrategy.WEIGHTED:
            return confidence >= 0.6

        # MAJORITY
        return sources >= self.min_threshold

    def _weighted_confidence(
        self,
        variants: list[dict],
        weights: list[float],
    ) -> float:
        """Calcula confiança ponderada."""
        # Simplificado - considera peso igual para todas as variantes
        return sum(weights[:len(variants)]) / sum(weights)

    def _detect_contradictions(
        self,
        finding_sets: list[list[dict]],
    ) -> list[dict]:
        """Detecta contradições entre findings."""
        contradictions = []

        # Procura findings que dizem coisas opostas
        # (ex: um diz "seguro", outro diz "vulnerável")
        # Por simplicidade, retorna vazio na implementação inicial

        return contradictions

    def _calculate_overall_confidence(
        self,
        confirmed: list[ConsolidatedFinding],
        disputed: list[ConsolidatedFinding],
        rejected: list[ConsolidatedFinding],
    ) -> float:
        """Calcula confiança geral do consenso."""
        if not (confirmed or disputed or rejected):
            return 1.0  # Sem findings = alta confiança

        total = len(confirmed) + len(disputed) + len(rejected)
        confirmed_weight = len(confirmed) / total

        # Penaliza por disputas
        dispute_penalty = len(disputed) * 0.1

        confidence = confirmed_weight - dispute_penalty
        return max(0.0, min(1.0, confidence))


# Exports
__all__ = [
    "MergeStrategy",
    "ConsolidatedFinding",
    "ConsensusResult",
    "ConsensusEngine",
]
```

---

## Engine __init__.py

```python
"""
HLS-ENG: Core Engine Module
===========================

Módulos Lego do core engine do Human Layer.

Exports:
    - HumanLayerRunner, RunContext, RunResult, LayerID
    - TripleRedundancy, RedundancyResult, ConsensusLevel
    - LayerOrchestrator, OrchestratorConfig, OrchestrationResult
    - VetoGate, VetoDecision, VetoPolicy
    - ConsensusEngine, ConsensusResult, MergeStrategy
"""

# HLS-ENG-001: Runner
from .runner import (
    LayerID,
    RunContext,
    RunResult,
    LLMClientProtocol,
    ResponseParserProtocol,
    HumanLayerRunner,
    create_runner,
)

# HLS-ENG-002: Redundancy
from .redundancy import (
    ConsensusLevel,
    RedundancyResult,
    TripleRedundancy,
)

# HLS-ENG-003: Orchestrator
from .orchestrator import (
    FinalVerdict,
    VetoLevel as OrchestratorVetoLevel,
    OrchestratorConfig,
    LayerExecutionResult,
    OrchestrationResult,
    LayerOrchestrator,
)

# HLS-ENG-004: Veto
from .veto import (
    VetoLevel,
    VetoPolicy,
    VetoDecision,
    PolicyConfig,
    VetoGate,
    create_permissive_gate,
    create_strict_gate,
    create_security_gate,
)

# HLS-ENG-005: Consensus
from .consensus import (
    MergeStrategy,
    ConsolidatedFinding,
    ConsensusResult,
    ConsensusEngine,
)


__all__ = [
    # Runner
    "LayerID",
    "RunContext",
    "RunResult",
    "LLMClientProtocol",
    "ResponseParserProtocol",
    "HumanLayerRunner",
    "create_runner",
    # Redundancy
    "ConsensusLevel",
    "RedundancyResult",
    "TripleRedundancy",
    # Orchestrator
    "FinalVerdict",
    "OrchestratorVetoLevel",
    "OrchestratorConfig",
    "LayerExecutionResult",
    "OrchestrationResult",
    "LayerOrchestrator",
    # Veto
    "VetoLevel",
    "VetoPolicy",
    "VetoDecision",
    "PolicyConfig",
    "VetoGate",
    "create_permissive_gate",
    "create_strict_gate",
    "create_security_gate",
    # Consensus
    "MergeStrategy",
    "ConsolidatedFinding",
    "ConsensusResult",
    "ConsensusEngine",
]
```

---

## Testes - tests/test_engine.py

```python
"""
Testes para módulos HLS-ENG (Core Engine).
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock


# ============================================================
# HLS-ENG-001: HumanLayerRunner Tests
# ============================================================

class TestLayerID:
    """Testes para LayerID enum."""

    def test_all_layers_exist(self):
        """Todas as 7 layers existem."""
        from hl_mcp.engine.runner import LayerID

        assert len(LayerID) == 7
        assert LayerID.HL_1 == "HL-1"
        assert LayerID.HL_7 == "HL-7"


class TestRunContext:
    """Testes para RunContext."""

    def test_run_context_creation(self):
        """RunContext pode ser criado."""
        from hl_mcp.engine.runner import RunContext, LayerID

        context = RunContext(
            layer_id=LayerID.HL_2,
            target_description="Login form",
        )

        assert context.layer_id == LayerID.HL_2
        assert context.target_description == "Login form"
        assert context.run_id  # Auto-gerado

    def test_run_context_to_dict(self):
        """to_dict() serializa corretamente."""
        from hl_mcp.engine.runner import RunContext, LayerID

        context = RunContext(
            layer_id=LayerID.HL_1,
            target_description="Test",
            perspective="tired_user",
        )

        d = context.to_dict()

        assert d["layer_id"] == "HL-1"
        assert d["perspective"] == "tired_user"


class TestRunResult:
    """Testes para RunResult."""

    def test_finding_count(self):
        """finding_count retorna número correto."""
        from hl_mcp.engine.runner import RunResult, RunContext, LayerID

        context = RunContext(layer_id=LayerID.HL_1, target_description="Test")
        result = RunResult(
            context=context,
            success=True,
            findings=[
                {"severity": "high", "title": "Issue 1"},
                {"severity": "low", "title": "Issue 2"},
            ],
        )

        assert result.finding_count == 2

    def test_critical_count(self):
        """critical_count conta apenas critical/high."""
        from hl_mcp.engine.runner import RunResult, RunContext, LayerID

        context = RunContext(layer_id=LayerID.HL_1, target_description="Test")
        result = RunResult(
            context=context,
            success=True,
            findings=[
                {"severity": "critical", "title": "Critical"},
                {"severity": "high", "title": "High"},
                {"severity": "medium", "title": "Medium"},
            ],
        )

        assert result.critical_count == 2


# ============================================================
# HLS-ENG-002: TripleRedundancy Tests
# ============================================================

class TestConsensusLevel:
    """Testes para ConsensusLevel."""

    def test_consensus_levels(self):
        """Todos os níveis de consenso existem."""
        from hl_mcp.engine.redundancy import ConsensusLevel

        assert ConsensusLevel.UNANIMOUS == "unanimous"
        assert ConsensusLevel.MAJORITY == "majority"
        assert ConsensusLevel.SPLIT == "split"
        assert ConsensusLevel.FAILED == "failed"


class TestRedundancyResult:
    """Testes para RedundancyResult."""

    def test_unanimous_findings(self):
        """unanimous_findings retorna apenas findings 3/3."""
        from hl_mcp.engine.redundancy import RedundancyResult, ConsensusLevel
        from hl_mcp.engine.runner import RunContext, LayerID

        context = RunContext(layer_id=LayerID.HL_1, target_description="Test")
        result = RedundancyResult(
            context=context,
            consensus_level=ConsensusLevel.MAJORITY,
            confirmed_findings=[
                {"title": "All 3", "_consensus_count": 3},
                {"title": "Only 2", "_consensus_count": 2},
            ],
        )

        unanimous = result.unanimous_findings
        assert len(unanimous) == 1
        assert unanimous[0]["title"] == "All 3"


# ============================================================
# HLS-ENG-003: LayerOrchestrator Tests
# ============================================================

class TestFinalVerdict:
    """Testes para FinalVerdict."""

    def test_verdicts(self):
        """Todos os veredictos existem."""
        from hl_mcp.engine.orchestrator import FinalVerdict

        assert FinalVerdict.APPROVED == "approved"
        assert FinalVerdict.REJECTED == "rejected"
        assert FinalVerdict.NEEDS_REVIEW == "needs_review"


class TestOrchestratorConfig:
    """Testes para OrchestratorConfig."""

    def test_default_config(self):
        """Config padrão inclui todas as layers."""
        from hl_mcp.engine.orchestrator import OrchestratorConfig
        from hl_mcp.engine.runner import LayerID

        config = OrchestratorConfig()

        assert len(config.layers) == 7
        assert config.use_redundancy is True

    def test_quick_check_config(self):
        """quick_check() retorna config com 3 layers."""
        from hl_mcp.engine.orchestrator import OrchestratorConfig

        config = OrchestratorConfig.quick_check()

        assert len(config.layers) == 3
        assert config.use_redundancy is False


class TestOrchestrationResult:
    """Testes para OrchestrationResult."""

    def test_critical_findings(self):
        """critical_findings filtra corretamente."""
        from hl_mcp.engine.orchestrator import (
            OrchestrationResult,
            FinalVerdict,
        )

        result = OrchestrationResult(
            final_verdict=FinalVerdict.NEEDS_REVIEW,
            all_findings=[
                {"severity": "critical", "title": "SQL Injection"},
                {"severity": "low", "title": "Typo"},
            ],
        )

        assert len(result.critical_findings) == 1
        assert result.critical_findings[0]["title"] == "SQL Injection"


# ============================================================
# HLS-ENG-004: VetoGate Tests
# ============================================================

class TestVetoLevel:
    """Testes para VetoLevel."""

    def test_veto_levels(self):
        """Todos os níveis de veto existem."""
        from hl_mcp.engine.veto import VetoLevel

        assert VetoLevel.NONE == "NONE"
        assert VetoLevel.WEAK == "WEAK"
        assert VetoLevel.MEDIUM == "MEDIUM"
        assert VetoLevel.STRONG == "STRONG"


class TestVetoDecision:
    """Testes para VetoDecision."""

    def test_should_block(self):
        """should_block é True apenas para STRONG."""
        from hl_mcp.engine.veto import VetoDecision, VetoLevel

        strong = VetoDecision(level=VetoLevel.STRONG)
        medium = VetoDecision(level=VetoLevel.MEDIUM)

        assert strong.should_block is True
        assert medium.should_block is False

    def test_needs_attention(self):
        """needs_attention é True para MEDIUM e STRONG."""
        from hl_mcp.engine.veto import VetoDecision, VetoLevel

        strong = VetoDecision(level=VetoLevel.STRONG)
        medium = VetoDecision(level=VetoLevel.MEDIUM)
        weak = VetoDecision(level=VetoLevel.WEAK)

        assert strong.needs_attention is True
        assert medium.needs_attention is True
        assert weak.needs_attention is False


class TestVetoGate:
    """Testes para VetoGate."""

    def test_no_findings_returns_none(self):
        """Sem findings retorna NONE."""
        from hl_mcp.engine.veto import VetoGate, VetoLevel

        gate = VetoGate()
        decision = gate.evaluate([])

        assert decision.level == VetoLevel.NONE

    def test_critical_blocks(self):
        """Finding critical retorna STRONG."""
        from hl_mcp.engine.veto import VetoGate, VetoLevel

        gate = VetoGate()
        findings = [{"severity": "critical", "title": "SQL Injection"}]
        decision = gate.evaluate(findings)

        assert decision.level == VetoLevel.STRONG
        assert len(decision.blocking_findings) == 1

    def test_low_severity_weak(self):
        """Finding low retorna WEAK ou MEDIUM."""
        from hl_mcp.engine.veto import VetoGate, VetoLevel

        gate = VetoGate()
        findings = [{"severity": "low", "title": "Typo"}]
        decision = gate.evaluate(findings)

        assert decision.level in [VetoLevel.WEAK, VetoLevel.MEDIUM]


# ============================================================
# HLS-ENG-005: ConsensusEngine Tests
# ============================================================

class TestMergeStrategy:
    """Testes para MergeStrategy."""

    def test_strategies(self):
        """Todas as estratégias existem."""
        from hl_mcp.engine.consensus import MergeStrategy

        assert MergeStrategy.MAJORITY == "majority"
        assert MergeStrategy.UNANIMOUS == "unanimous"
        assert MergeStrategy.ANY == "any"


class TestConsensusEngine:
    """Testes para ConsensusEngine."""

    def test_empty_sets(self):
        """Lista vazia retorna resultado vazio."""
        from hl_mcp.engine.consensus import ConsensusEngine

        engine = ConsensusEngine()
        result = engine.consolidate([])

        assert len(result.confirmed) == 0

    def test_majority_consensus(self):
        """2/3 confirma finding."""
        from hl_mcp.engine.consensus import ConsensusEngine, MergeStrategy

        engine = ConsensusEngine(strategy=MergeStrategy.MAJORITY)

        findings_sets = [
            [{"title": "Issue A", "severity": "high"}],
            [{"title": "Issue A", "severity": "high"}],
            [],  # Terceiro run não encontrou
        ]

        result = engine.consolidate(findings_sets)

        assert len(result.confirmed) == 1

    def test_unanimous_requires_all(self):
        """UNANIMOUS exige todos concordando."""
        from hl_mcp.engine.consensus import ConsensusEngine, MergeStrategy

        engine = ConsensusEngine(strategy=MergeStrategy.UNANIMOUS)

        findings_sets = [
            [{"title": "Issue A", "severity": "high"}],
            [{"title": "Issue A", "severity": "high"}],
            [],
        ]

        result = engine.consolidate(findings_sets)

        # Não é unânime, então vai para disputed
        assert len(result.confirmed) == 0
        assert len(result.disputed) == 1
```

---

## Atualização do LEGO_INDEX.yaml

```yaml
# ============================================================
# CORE ENGINE (HLS-ENG-001 a HLS-ENG-005)
# ============================================================

HLS-ENG-001:
  name: HumanLayerRunner
  path: src/hl_mcp/engine/runner.py
  category: engine
  description: Executor de uma Human Layer individual
  exports:
    - LayerID
    - RunContext
    - RunResult
    - HumanLayerRunner
    - create_runner
  dependencies:
    - HLS-LLM-001
    - HLS-MDL-001
    - HLS-MDL-002
  search_hints:
    - runner
    - layer
    - execute
    - run
    - HL-1
    - HL-7

HLS-ENG-002:
  name: TripleRedundancy
  path: src/hl_mcp/engine/redundancy.py
  category: engine
  description: Execução tripla com consenso 2/3
  exports:
    - ConsensusLevel
    - RedundancyResult
    - TripleRedundancy
  dependencies:
    - HLS-ENG-001
  search_hints:
    - redundancy
    - triple
    - consensus
    - 2/3
    - majority

HLS-ENG-003:
  name: LayerOrchestrator
  path: src/hl_mcp/engine/orchestrator.py
  category: engine
  description: Orquestra execução das 7 Human Layers
  exports:
    - FinalVerdict
    - VetoLevel
    - OrchestratorConfig
    - OrchestrationResult
    - LayerOrchestrator
  dependencies:
    - HLS-ENG-001
    - HLS-ENG-002
    - HLS-ENG-004
  search_hints:
    - orchestrator
    - pipeline
    - sequence
    - layers
    - verdict

HLS-ENG-004:
  name: VetoGate
  path: src/hl_mcp/engine/veto.py
  category: engine
  description: Lógica de aplicação de vetos
  exports:
    - VetoLevel
    - VetoPolicy
    - VetoDecision
    - VetoGate
  dependencies:
    - HLS-MDL-001
    - HLS-MDL-002
  search_hints:
    - veto
    - block
    - gate
    - policy
    - strong
    - medium
    - weak

HLS-ENG-005:
  name: ConsensusEngine
  path: src/hl_mcp/engine/consensus.py
  category: engine
  description: Engine de consenso para múltiplas execuções
  exports:
    - MergeStrategy
    - ConsolidatedFinding
    - ConsensusResult
    - ConsensusEngine
  dependencies:
    - HLS-MDL-001
  search_hints:
    - consensus
    - merge
    - consolidate
    - group
    - similarity
```

---

## Resumo do Block 08

| Módulo | ID | Linhas | Exports |
|--------|-----|--------|---------|
| HumanLayerRunner | HLS-ENG-001 | ~350 | 6 |
| TripleRedundancy | HLS-ENG-002 | ~280 | 3 |
| LayerOrchestrator | HLS-ENG-003 | ~400 | 6 |
| VetoGate | HLS-ENG-004 | ~220 | 7 |
| ConsensusEngine | HLS-ENG-005 | ~250 | 4 |
| **TOTAL** | | **~1,500** | **26** |

---

## Próximo: Block 09 - 7 Human Layers

O Block 09 vai implementar as 7 Human Layers:
1. **HLS-LAY-001**: HL-1 UI/UX Review
2. **HLS-LAY-002**: HL-2 Security Scan
3. **HLS-LAY-003**: HL-3 Edge Cases
4. **HLS-LAY-004**: HL-4 Accessibility
5. **HLS-LAY-005**: HL-5 Performance
6. **HLS-LAY-006**: HL-6 Integration
7. **HLS-LAY-007**: HL-7 Final Human Check

Quer que eu continue com o Block 09 (7 Human Layers)?
