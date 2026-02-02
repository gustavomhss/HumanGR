# ROADMAP Block 10: 6 Perspectives Implementation

> **Bloco**: 10 de 12
> **Tema**: As 6 Perspectivas de Teste
> **Tokens Estimados**: ~22,000
> **Dependências**: Block 08-09 (Engine, Layers)

---

## Visão Geral do Bloco

As 6 perspectivas simulam diferentes tipos de usuários testando o sistema:

| Perspective | Nome | Foco | Mindset |
|-------------|------|------|---------|
| PRS-001 | tired_user | Erros por distração | "Estou cansado, quero acabar logo" |
| PRS-002 | malicious_insider | Explorar falhas | "Como posso abusar disso?" |
| PRS-003 | confused_newbie | Primeira vez | "O que isso faz?" |
| PRS-004 | power_user | Atalhos, edge cases | "E se eu fizer isso..." |
| PRS-005 | auditor | Compliance, logs | "Isso está documentado?" |
| PRS-006 | 3am_operator | Stress, fadiga | "Preciso resolver rápido" |

---

## Base: Perspective Protocol

```python
"""
HLS-PRS-000: Perspective Base
=============================

Protocolo base para todas as perspectivas.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Protocol
import logging

logger = logging.getLogger(__name__)


class PerspectiveID(str, Enum):
    """IDs das 6 perspectivas."""

    TIRED_USER = "tired_user"
    MALICIOUS_INSIDER = "malicious_insider"
    CONFUSED_NEWBIE = "confused_newbie"
    POWER_USER = "power_user"
    AUDITOR = "auditor"
    OPERATOR_3AM = "3am_operator"


@dataclass
class PerspectiveContext:
    """
    Contexto para execução de uma perspectiva.

    Attributes:
        perspective_id: ID da perspectiva
        target: O que está sendo testado
        code: Código relevante
        ui_elements: Elementos de UI
        journey: Jornada a simular
        time_pressure: Nível de pressão de tempo (0-1)
        expertise_level: Nível de expertise (0-1)
    """

    perspective_id: PerspectiveID
    target: str
    code: Optional[str] = None
    ui_elements: list[dict] = field(default_factory=list)
    journey: Optional[dict] = None
    time_pressure: float = 0.5
    expertise_level: float = 0.5
    extra: dict = field(default_factory=dict)


@dataclass
class PerspectiveResult:
    """
    Resultado de análise sob uma perspectiva.

    Attributes:
        perspective_id: ID da perspectiva
        findings: Issues encontrados
        behaviors: Comportamentos simulados
        pain_points: Pontos de dor identificados
        confidence: Confiança na análise
    """

    perspective_id: PerspectiveID
    success: bool
    findings: list[dict] = field(default_factory=list)
    behaviors: list[str] = field(default_factory=list)
    pain_points: list[str] = field(default_factory=list)
    confidence: float = 0.0
    duration_ms: float = 0

    @property
    def finding_count(self) -> int:
        return len(self.findings)

    def to_dict(self) -> dict:
        return {
            "perspective": self.perspective_id.value,
            "success": self.success,
            "finding_count": self.finding_count,
            "behaviors": self.behaviors,
            "pain_points": self.pain_points,
            "confidence": self.confidence,
            "findings": self.findings,
        }


class PerspectiveBase(ABC):
    """
    Classe base para perspectivas.

    Cada perspectiva deve:
    1. Definir sua persona (mindset, comportamentos típicos)
    2. Implementar analyze() para simular o usuário
    3. Gerar findings específicos do ponto de vista
    """

    PERSPECTIVE_ID: PerspectiveID
    NAME: str
    DESCRIPTION: str

    @abstractmethod
    async def analyze(
        self,
        context: PerspectiveContext,
    ) -> PerspectiveResult:
        """Executa análise sob esta perspectiva."""
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Retorna system prompt específico da perspectiva."""
        pass

    @abstractmethod
    def get_persona_description(self) -> str:
        """Retorna descrição da persona."""
        pass
```

---

## Módulo HLS-PRS-001: Tired User

```
ID: HLS-PRS-001
Nome: TiredUserPerspective
Caminho: src/hl_mcp/perspectives/tired_user.py
Dependências: HLS-PRS-000
Exports: TiredUserPerspective
Linhas: ~200
```

### Código

```python
"""
HLS-PRS-001: Tired User Perspective
===================================

Simula usuário cansado, distraído, com pressa.

Mindset:
- "Quero acabar logo com isso"
- "Não vou ler tudo, só clicar"
- "Se der erro, vou tentar de novo"

Comportamentos típicos:
- Pula instruções
- Clica rápido demais
- Ignora avisos
- Comete erros de digitação
- Fica frustrado facilmente

Foco de teste:
- Fluxos muito longos
- Mensagens de erro confusas
- Falta de feedback
- Ações destrutivas sem confirmação

Changelog:
    - v1.0.0 (2026-02-01): Implementação inicial
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Protocol
import logging

from .base import (
    PerspectiveBase,
    PerspectiveContext,
    PerspectiveResult,
    PerspectiveID,
)

logger = logging.getLogger(__name__)


class LLMClientProtocol(Protocol):
    async def complete(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        ...


TIRED_USER_SYSTEM_PROMPT = """You are simulating a TIRED, DISTRACTED user.

YOUR MINDSET:
- You want to finish this task as fast as possible
- You're not going to read every instruction carefully
- You'll click through quickly and may miss things
- If something doesn't work, you'll get frustrated
- You might make typos or click the wrong button

HOW YOU BEHAVE:
1. Skip reading long text - just scan for buttons
2. Click the first thing that looks right
3. Don't notice small warnings or hints
4. Get annoyed if things take too long
5. Try to take shortcuts
6. May accidentally double-click or rapid-click

WHAT YOU LOOK FOR (issues that would affect you):
- Long forms that can't be saved halfway
- Confusing button labels ("Submit" vs "Continue" vs "Next")
- Actions that can't be undone
- Small text or easy-to-miss warnings
- Too many steps to complete a task
- Slow loading without feedback

For each issue found, explain:
- What you were trying to do
- What went wrong
- How it made you feel (frustrated, confused, etc.)
- What would have helped

Respond in JSON:
{
    "behaviors": ["Skipped reading the terms", "Clicked Submit twice"],
    "pain_points": ["Had to re-enter all data after error"],
    "findings": [
        {
            "category": "ux",
            "severity": "medium",
            "title": "No save draft option",
            "description": "Lost all progress when I accidentally closed the tab",
            "user_impact": "Frustration, wasted time",
            "recommendation": "Add auto-save or save draft"
        }
    ],
    "confidence": 0.8
}
"""


class TiredUserPerspective(PerspectiveBase):
    """
    Perspectiva do usuário cansado.

    Simula comportamento de quem está com pressa,
    distraído, e propenso a erros.

    Example:
        >>> perspective = TiredUserPerspective(llm_client)
        >>> result = await perspective.analyze(context)
        >>> print(result.pain_points)
    """

    PERSPECTIVE_ID = PerspectiveID.TIRED_USER
    NAME = "Tired User"
    DESCRIPTION = "Distracted user who wants to finish quickly"

    def __init__(self, llm_client: LLMClientProtocol):
        self.llm = llm_client

    def get_system_prompt(self) -> str:
        return TIRED_USER_SYSTEM_PROMPT

    def get_persona_description(self) -> str:
        return (
            "A tired office worker at 5pm trying to complete a task "
            "before leaving. Has been working all day, is distracted, "
            "and just wants to click through as fast as possible."
        )

    async def analyze(
        self,
        context: PerspectiveContext,
    ) -> PerspectiveResult:
        """Analisa sob perspectiva de usuário cansado."""
        start = datetime.utcnow()

        logger.info(f"Analisando como {self.NAME}: {context.target}")

        prompt = self._build_prompt(context)

        try:
            response = await self.llm.complete(
                prompt=prompt,
                system_prompt=self.get_system_prompt(),
            )

            parsed = self._parse_response(response)
            duration = (datetime.utcnow() - start).total_seconds() * 1000

            return PerspectiveResult(
                perspective_id=self.PERSPECTIVE_ID,
                success=True,
                findings=parsed.get("findings", []),
                behaviors=parsed.get("behaviors", []),
                pain_points=parsed.get("pain_points", []),
                confidence=parsed.get("confidence", 0.7),
                duration_ms=duration,
            )

        except Exception as e:
            duration = (datetime.utcnow() - start).total_seconds() * 1000
            logger.error(f"Erro em {self.NAME}: {e}")
            return PerspectiveResult(
                perspective_id=self.PERSPECTIVE_ID,
                success=False,
                duration_ms=duration,
            )

    def _build_prompt(self, context: PerspectiveContext) -> str:
        parts = [
            f"## Scenario\n{context.target}",
            f"## Your State\nYou are tired (fatigue level: {context.time_pressure:.0%})",
        ]

        if context.ui_elements:
            elements = "\n".join(f"- {e}" for e in context.ui_elements[:10])
            parts.append(f"## UI Elements\n{elements}")

        if context.journey:
            steps = context.journey.get("steps", [])
            steps_str = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps[:10]))
            parts.append(f"## Journey Steps\n{steps_str}")

        parts.append(
            "## Task\n"
            "Simulate going through this as a tired user. "
            "What problems would you encounter? "
            "What would frustrate you?"
        )

        return "\n\n".join(parts)

    def _parse_response(self, response: str) -> dict:
        import json
        try:
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                return json.loads(response[start:end].strip())
            elif "{" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                return json.loads(response[start:end])
        except:
            pass
        return {"findings": [], "behaviors": [], "pain_points": []}


__all__ = ["TiredUserPerspective"]
```

---

## Módulo HLS-PRS-002: Malicious Insider

```
ID: HLS-PRS-002
Nome: MaliciousInsiderPerspective
Caminho: src/hl_mcp/perspectives/malicious_insider.py
Dependências: HLS-PRS-000
Exports: MaliciousInsiderPerspective
Linhas: ~220
```

### Código

```python
"""
HLS-PRS-002: Malicious Insider Perspective
==========================================

Simula funcionário malicioso tentando explorar o sistema.

Mindset:
- "Como posso abusar das minhas permissões?"
- "Que dados posso acessar que não deveria?"
- "Como escondo minhas ações?"

Comportamentos típicos:
- Tenta acessar recursos de outros usuários
- Busca escalação de privilégios
- Tenta exfiltrar dados
- Procura logs e como evitá-los
- Explora funcionalidades edge

Foco de teste:
- Broken access control
- IDOR vulnerabilities
- Audit log bypasses
- Data exfiltration paths
- Privilege escalation

Changelog:
    - v1.0.0 (2026-02-01): Implementação inicial
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Protocol
import logging

from .base import (
    PerspectiveBase,
    PerspectiveContext,
    PerspectiveResult,
    PerspectiveID,
)

logger = logging.getLogger(__name__)


class LLMClientProtocol(Protocol):
    async def complete(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        ...


MALICIOUS_INSIDER_SYSTEM_PROMPT = """You are simulating a MALICIOUS INSIDER - an employee trying to exploit the system.

YOUR MINDSET:
- You have legitimate access but want to abuse it
- You're looking for ways to access data you shouldn't
- You want to do harmful things without getting caught
- You know how the system works from the inside

ATTACK VECTORS YOU TRY:
1. IDOR (Insecure Direct Object Reference)
   - Change IDs in URLs to access other users' data
   - Modify request parameters to see unauthorized info

2. PRIVILEGE ESCALATION
   - Try to access admin functions
   - Modify your own role/permissions
   - Access API endpoints beyond your scope

3. DATA EXFILTRATION
   - Export more data than needed
   - Copy sensitive information
   - Access bulk download features

4. AUDIT EVASION
   - Look for actions not logged
   - Try to delete or modify audit trails
   - Use features that bypass normal logging

5. SOCIAL ENGINEERING
   - Abuse password reset for other users
   - Access shared resources inappropriately
   - Exploit trust relationships

For each vulnerability found:
- What you tried
- Whether it worked
- The impact if exploited
- How to fix it

Respond in JSON:
{
    "behaviors": ["Changed user_id parameter", "Tried to access /admin"],
    "findings": [
        {
            "category": "security/access_control",
            "severity": "critical",
            "title": "IDOR allows access to other users' data",
            "description": "Changing user_id=123 to user_id=456 shows other user's data",
            "attack_vector": "Parameter tampering",
            "impact": "Full data breach possible",
            "remediation": "Implement proper authorization checks"
        }
    ],
    "confidence": 0.9
}
"""


class MaliciousInsiderPerspective(PerspectiveBase):
    """
    Perspectiva do insider malicioso.

    Simula funcionário tentando abusar de acesso legítimo.

    Example:
        >>> perspective = MaliciousInsiderPerspective(llm_client)
        >>> result = await perspective.analyze(context)
        >>> for f in result.findings:
        ...     if f["severity"] == "critical":
        ...         print(f"CRITICAL: {f['title']}")
    """

    PERSPECTIVE_ID = PerspectiveID.MALICIOUS_INSIDER
    NAME = "Malicious Insider"
    DESCRIPTION = "Employee trying to exploit their access"

    def __init__(self, llm_client: LLMClientProtocol):
        self.llm = llm_client

    def get_system_prompt(self) -> str:
        return MALICIOUS_INSIDER_SYSTEM_PROMPT

    def get_persona_description(self) -> str:
        return (
            "A disgruntled employee with legitimate system access "
            "looking for ways to steal data, escalate privileges, "
            "or cause damage without being detected."
        )

    async def analyze(
        self,
        context: PerspectiveContext,
    ) -> PerspectiveResult:
        """Analisa sob perspectiva de insider malicioso."""
        start = datetime.utcnow()

        logger.info(f"Analisando como {self.NAME}: {context.target}")

        prompt = self._build_prompt(context)

        try:
            response = await self.llm.complete(
                prompt=prompt,
                system_prompt=self.get_system_prompt(),
            )

            parsed = self._parse_response(response)
            duration = (datetime.utcnow() - start).total_seconds() * 1000

            # Findings de insider são geralmente mais severos
            findings = self._enhance_findings(parsed.get("findings", []))

            return PerspectiveResult(
                perspective_id=self.PERSPECTIVE_ID,
                success=True,
                findings=findings,
                behaviors=parsed.get("behaviors", []),
                pain_points=[],  # Insider não tem "pain points"
                confidence=parsed.get("confidence", 0.8),
                duration_ms=duration,
            )

        except Exception as e:
            duration = (datetime.utcnow() - start).total_seconds() * 1000
            logger.error(f"Erro em {self.NAME}: {e}")
            return PerspectiveResult(
                perspective_id=self.PERSPECTIVE_ID,
                success=False,
                duration_ms=duration,
            )

    def _build_prompt(self, context: PerspectiveContext) -> str:
        parts = [
            f"## Target System\n{context.target}",
            "## Your Role\nYou are an employee with standard user access.",
        ]

        if context.code:
            parts.append(f"## Code (for analysis)\n```\n{context.code[:2000]}\n```")

        if context.ui_elements:
            elements = "\n".join(f"- {e}" for e in context.ui_elements[:15])
            parts.append(f"## Available UI Elements\n{elements}")

        parts.append(
            "## Task\n"
            "As a malicious insider, analyze this system for:\n"
            "- Access control weaknesses\n"
            "- Data exfiltration opportunities\n"
            "- Privilege escalation paths\n"
            "- Audit bypass methods"
        )

        return "\n\n".join(parts)

    def _enhance_findings(self, findings: list[dict]) -> list[dict]:
        """Adiciona metadados de segurança aos findings."""
        for finding in findings:
            # Adiciona attack vector se não tiver
            if "attack_vector" not in finding:
                finding["attack_vector"] = "insider_abuse"

            # Categoriza como security se não tiver categoria
            if not finding.get("category", "").startswith("security"):
                finding["category"] = f"security/{finding.get('category', 'access_control')}"

        return findings

    def _parse_response(self, response: str) -> dict:
        import json
        try:
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                return json.loads(response[start:end].strip())
            elif "{" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                return json.loads(response[start:end])
        except:
            pass
        return {"findings": [], "behaviors": []}


__all__ = ["MaliciousInsiderPerspective"]
```

---

## Módulos HLS-PRS-003 a HLS-PRS-006 (Resumidos)

### HLS-PRS-003: Confused Newbie

```python
"""
HLS-PRS-003: Confused Newbie Perspective

Simula usuário usando o sistema pela primeira vez.

Mindset:
- "O que isso faz?"
- "Por onde eu começo?"
- "Isso está certo?"

Foco:
- Onboarding confuso
- Falta de documentação
- Jargão técnico
- Fluxos não intuitivos
"""

# ~180 linhas
```

### HLS-PRS-004: Power User

```python
"""
HLS-PRS-004: Power User Perspective

Simula usuário avançado explorando limites.

Mindset:
- "E se eu fizer isso diferente?"
- "Deve ter um atalho para isso"
- "O sistema aguenta isso?"

Foco:
- Edge cases avançados
- Limites do sistema
- Bulk operations
- API abuse
"""

# ~180 linhas
```

### HLS-PRS-005: Auditor

```python
"""
HLS-PRS-005: Auditor Perspective

Simula auditor externo verificando compliance.

Mindset:
- "Isso está documentado?"
- "Onde está o log disso?"
- "Quem aprovou isso?"

Foco:
- Audit trails
- Compliance
- Data retention
- Access logs
"""

# ~180 linhas
```

### HLS-PRS-006: 3AM Operator

```python
"""
HLS-PRS-006: 3AM Operator Perspective

Simula operador de madrugada sob pressão.

Mindset:
- "O sistema caiu, preciso resolver AGORA"
- "Não tenho tempo para ler documentação"
- "Vou fazer o que for necessário"

Foco:
- Error messages claras
- Recovery procedures
- Emergency access
- Logging sob stress
"""

# ~180 linhas
```

---

## Perspectives __init__.py

```python
"""
HLS-PRS: Perspectives Module
============================

As 6 perspectivas de teste do Human Layer.

Exports:
    - TiredUserPerspective
    - MaliciousInsiderPerspective
    - ConfusedNewbiePerspective
    - PowerUserPerspective
    - AuditorPerspective
    - Operator3AMPerspective
    - PerspectiveRunner (executa múltiplas perspectivas)
"""

from .base import (
    PerspectiveID,
    PerspectiveContext,
    PerspectiveResult,
    PerspectiveBase,
)

from .tired_user import TiredUserPerspective
from .malicious_insider import MaliciousInsiderPerspective
# from .confused_newbie import ConfusedNewbiePerspective
# from .power_user import PowerUserPerspective
# from .auditor import AuditorPerspective
# from .operator_3am import Operator3AMPerspective


# Registry de perspectivas
PERSPECTIVE_REGISTRY = {
    PerspectiveID.TIRED_USER: TiredUserPerspective,
    PerspectiveID.MALICIOUS_INSIDER: MaliciousInsiderPerspective,
    # PerspectiveID.CONFUSED_NEWBIE: ConfusedNewbiePerspective,
    # PerspectiveID.POWER_USER: PowerUserPerspective,
    # PerspectiveID.AUDITOR: AuditorPerspective,
    # PerspectiveID.OPERATOR_3AM: Operator3AMPerspective,
}


def get_perspective(perspective_id: PerspectiveID, llm_client):
    """Obtém instância de perspectiva por ID."""
    cls = PERSPECTIVE_REGISTRY.get(perspective_id)
    if cls:
        return cls(llm_client)
    return None


def get_all_perspectives(llm_client) -> list[PerspectiveBase]:
    """Retorna todas as perspectivas instanciadas."""
    return [cls(llm_client) for cls in PERSPECTIVE_REGISTRY.values()]


__all__ = [
    # Base
    "PerspectiveID",
    "PerspectiveContext",
    "PerspectiveResult",
    "PerspectiveBase",
    # Perspectives
    "TiredUserPerspective",
    "MaliciousInsiderPerspective",
    # Registry
    "PERSPECTIVE_REGISTRY",
    "get_perspective",
    "get_all_perspectives",
]
```

---

## PerspectiveRunner

```python
"""
HLS-PRS-RUN: Perspective Runner
===============================

Executa múltiplas perspectivas e consolida resultados.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import logging

from .base import (
    PerspectiveID,
    PerspectiveContext,
    PerspectiveResult,
    PerspectiveBase,
)
from . import get_all_perspectives, get_perspective

logger = logging.getLogger(__name__)


@dataclass
class MultiPerspectiveResult:
    """
    Resultado de análise com múltiplas perspectivas.

    Attributes:
        results: Resultados por perspectiva
        consolidated_findings: Findings consolidados
        unique_findings: Findings únicos (dedup)
        coverage: Cobertura de perspectivas
    """

    results: dict[PerspectiveID, PerspectiveResult] = field(default_factory=dict)
    consolidated_findings: list[dict] = field(default_factory=list)
    unique_findings: list[dict] = field(default_factory=list)
    total_duration_ms: float = 0

    @property
    def successful_perspectives(self) -> int:
        return sum(1 for r in self.results.values() if r.success)

    @property
    def all_pain_points(self) -> list[str]:
        points = []
        for r in self.results.values():
            points.extend(r.pain_points)
        return list(set(points))

    def to_dict(self) -> dict:
        return {
            "perspectives_run": len(self.results),
            "successful": self.successful_perspectives,
            "total_findings": len(self.consolidated_findings),
            "unique_findings": len(self.unique_findings),
            "pain_points": self.all_pain_points,
            "duration_ms": self.total_duration_ms,
        }


class PerspectiveRunner:
    """
    Executor de múltiplas perspectivas.

    Pode executar todas as perspectivas ou um subset,
    em paralelo ou sequencialmente.

    Example:
        >>> runner = PerspectiveRunner(llm_client)
        >>> result = await runner.run_all(context)
        >>> print(f"Findings: {len(result.unique_findings)}")
    """

    def __init__(
        self,
        llm_client,
        perspectives: Optional[list[PerspectiveID]] = None,
    ):
        """
        Inicializa o runner.

        Args:
            llm_client: Cliente LLM
            perspectives: Lista de perspectivas a usar (None = todas)
        """
        self.llm = llm_client
        self.perspective_ids = perspectives or list(PerspectiveID)

    async def run_all(
        self,
        context: PerspectiveContext,
        parallel: bool = True,
    ) -> MultiPerspectiveResult:
        """
        Executa todas as perspectivas.

        Args:
            context: Contexto de análise
            parallel: Executar em paralelo

        Returns:
            MultiPerspectiveResult consolidado
        """
        start = datetime.utcnow()

        logger.info(f"Executando {len(self.perspective_ids)} perspectivas")

        if parallel:
            results = await self._run_parallel(context)
        else:
            results = await self._run_sequential(context)

        duration = (datetime.utcnow() - start).total_seconds() * 1000

        # Consolida findings
        all_findings = []
        for result in results.values():
            all_findings.extend(result.findings)

        # Deduplica
        unique = self._deduplicate_findings(all_findings)

        return MultiPerspectiveResult(
            results=results,
            consolidated_findings=all_findings,
            unique_findings=unique,
            total_duration_ms=duration,
        )

    async def _run_parallel(
        self,
        context: PerspectiveContext,
    ) -> dict[PerspectiveID, PerspectiveResult]:
        """Executa perspectivas em paralelo."""
        tasks = {}

        for pid in self.perspective_ids:
            perspective = get_perspective(pid, self.llm)
            if perspective:
                # Cria contexto específico
                ctx = PerspectiveContext(
                    perspective_id=pid,
                    target=context.target,
                    code=context.code,
                    ui_elements=context.ui_elements,
                    journey=context.journey,
                    time_pressure=context.time_pressure,
                    expertise_level=context.expertise_level,
                    extra=context.extra,
                )
                tasks[pid] = perspective.analyze(ctx)

        results_list = await asyncio.gather(*tasks.values())

        return dict(zip(tasks.keys(), results_list))

    async def _run_sequential(
        self,
        context: PerspectiveContext,
    ) -> dict[PerspectiveID, PerspectiveResult]:
        """Executa perspectivas sequencialmente."""
        results = {}

        for pid in self.perspective_ids:
            perspective = get_perspective(pid, self.llm)
            if perspective:
                ctx = PerspectiveContext(
                    perspective_id=pid,
                    target=context.target,
                    code=context.code,
                    ui_elements=context.ui_elements,
                    journey=context.journey,
                    time_pressure=context.time_pressure,
                    expertise_level=context.expertise_level,
                    extra=context.extra,
                )
                results[pid] = await perspective.analyze(ctx)

        return results

    def _deduplicate_findings(self, findings: list[dict]) -> list[dict]:
        """Remove findings duplicados."""
        seen = set()
        unique = []

        for finding in findings:
            # Chave baseada em título e categoria
            key = (
                finding.get("title", "").lower(),
                finding.get("category", ""),
            )
            if key not in seen:
                seen.add(key)
                unique.append(finding)

        return unique


__all__ = ["MultiPerspectiveResult", "PerspectiveRunner"]
```

---

## Testes - tests/test_perspectives.py

```python
"""
Testes para módulos HLS-PRS (Perspectives).
"""

import pytest
from unittest.mock import AsyncMock


class TestPerspectiveID:
    """Testes para PerspectiveID enum."""

    def test_all_perspectives_exist(self):
        """Todas as 6 perspectivas existem."""
        from hl_mcp.perspectives.base import PerspectiveID

        assert len(PerspectiveID) == 6
        assert PerspectiveID.TIRED_USER == "tired_user"
        assert PerspectiveID.MALICIOUS_INSIDER == "malicious_insider"


class TestTiredUserPerspective:
    """Testes para TiredUserPerspective."""

    def test_metadata(self):
        """Metadados estão corretos."""
        from hl_mcp.perspectives.tired_user import TiredUserPerspective

        assert TiredUserPerspective.NAME == "Tired User"

    @pytest.mark.asyncio
    async def test_analyze_returns_result(self):
        """analyze() retorna PerspectiveResult."""
        from hl_mcp.perspectives.tired_user import TiredUserPerspective
        from hl_mcp.perspectives.base import (
            PerspectiveContext,
            PerspectiveResult,
            PerspectiveID,
        )

        mock_llm = AsyncMock()
        mock_llm.complete.return_value = '{"findings": [], "behaviors": [], "pain_points": []}'

        perspective = TiredUserPerspective(mock_llm)
        context = PerspectiveContext(
            perspective_id=PerspectiveID.TIRED_USER,
            target="Login form",
        )

        result = await perspective.analyze(context)

        assert isinstance(result, PerspectiveResult)
        assert result.perspective_id == PerspectiveID.TIRED_USER


class TestMaliciousInsiderPerspective:
    """Testes para MaliciousInsiderPerspective."""

    def test_metadata(self):
        """Metadados estão corretos."""
        from hl_mcp.perspectives.malicious_insider import MaliciousInsiderPerspective

        assert MaliciousInsiderPerspective.NAME == "Malicious Insider"


class TestPerspectiveRunner:
    """Testes para PerspectiveRunner."""

    @pytest.mark.asyncio
    async def test_run_all(self):
        """run_all() executa múltiplas perspectivas."""
        from hl_mcp.perspectives.runner import PerspectiveRunner
        from hl_mcp.perspectives.base import PerspectiveContext, PerspectiveID

        mock_llm = AsyncMock()
        mock_llm.complete.return_value = '{"findings": [], "behaviors": []}'

        runner = PerspectiveRunner(
            mock_llm,
            perspectives=[PerspectiveID.TIRED_USER],
        )

        context = PerspectiveContext(
            perspective_id=PerspectiveID.TIRED_USER,
            target="Test",
        )

        result = await runner.run_all(context)

        assert result.successful_perspectives >= 0
```

---

## Atualização do LEGO_INDEX.yaml

```yaml
# ============================================================
# PERSPECTIVES (HLS-PRS-001 a HLS-PRS-006)
# ============================================================

HLS-PRS-001:
  name: TiredUserPerspective
  path: src/hl_mcp/perspectives/tired_user.py
  category: perspectives
  description: Usuário cansado, distraído, com pressa
  exports:
    - TiredUserPerspective
  search_hints:
    - tired
    - distracted
    - rush
    - frustrated

HLS-PRS-002:
  name: MaliciousInsiderPerspective
  path: src/hl_mcp/perspectives/malicious_insider.py
  category: perspectives
  description: Insider malicioso tentando explorar sistema
  exports:
    - MaliciousInsiderPerspective
  search_hints:
    - malicious
    - insider
    - exploit
    - abuse
    - idor

HLS-PRS-003:
  name: ConfusedNewbiePerspective
  path: src/hl_mcp/perspectives/confused_newbie.py
  category: perspectives
  description: Usuário novato confuso
  search_hints:
    - newbie
    - beginner
    - confused
    - first time

HLS-PRS-004:
  name: PowerUserPerspective
  path: src/hl_mcp/perspectives/power_user.py
  category: perspectives
  description: Usuário avançado explorando limites
  search_hints:
    - power user
    - advanced
    - shortcuts
    - edge case

HLS-PRS-005:
  name: AuditorPerspective
  path: src/hl_mcp/perspectives/auditor.py
  category: perspectives
  description: Auditor externo verificando compliance
  search_hints:
    - auditor
    - compliance
    - audit
    - logs

HLS-PRS-006:
  name: Operator3AMPerspective
  path: src/hl_mcp/perspectives/operator_3am.py
  category: perspectives
  description: Operador de madrugada sob pressão
  search_hints:
    - operator
    - 3am
    - stress
    - emergency

HLS-PRS-RUN:
  name: PerspectiveRunner
  path: src/hl_mcp/perspectives/runner.py
  category: perspectives
  description: Executor de múltiplas perspectivas
  exports:
    - PerspectiveRunner
    - MultiPerspectiveResult
  search_hints:
    - runner
    - multiple
    - all perspectives
```

---

## Resumo do Block 10

| Módulo | ID | Perspectiva | Linhas |
|--------|----|-------------|--------|
| TiredUserPerspective | HLS-PRS-001 | tired_user | ~200 |
| MaliciousInsiderPerspective | HLS-PRS-002 | malicious_insider | ~220 |
| ConfusedNewbiePerspective | HLS-PRS-003 | confused_newbie | ~180 |
| PowerUserPerspective | HLS-PRS-004 | power_user | ~180 |
| AuditorPerspective | HLS-PRS-005 | auditor | ~180 |
| Operator3AMPerspective | HLS-PRS-006 | 3am_operator | ~180 |
| PerspectiveRunner | HLS-PRS-RUN | executor | ~200 |
| **TOTAL** | | | **~1,340** |

---

## Próximo: Block 11 - Cognitive Modules

O Block 11 vai implementar os módulos cognitivos:
1. **HLS-COG-001**: Budget Manager (gerenciamento de tokens)
2. **HLS-COG-002**: Trust Scorer (scoring de confiança)
3. **HLS-COG-003**: Triage Engine (priorização de issues)
4. **HLS-COG-004**: Feedback Loop (aprendizado)
5. **HLS-COG-005**: Confidence Calculator (cálculo de confidence)

Quer que eu continue com o Block 11 (Cognitive Modules)?
