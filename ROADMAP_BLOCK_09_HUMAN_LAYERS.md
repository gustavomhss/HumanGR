# ROADMAP Block 09: 7 Human Layers Implementation

> **Bloco**: 09 de 12
> **Tema**: As 7 Human Layers (HL-1 a HL-7)
> **Tokens Estimados**: ~30,000
> **Dependências**: Block 08 (Core Engine)

---

## Visão Geral do Bloco

Este bloco implementa as 7 Human Layers - cada uma especializada em um tipo de validação:

| Layer | Nome | Foco | Veto Power |
|-------|------|------|------------|
| HL-1 | UI/UX Review | Usabilidade, clareza | WEAK |
| HL-2 | Security Scan | Vulnerabilidades, OWASP | STRONG |
| HL-3 | Edge Cases | Casos limite, erros | MEDIUM |
| HL-4 | Accessibility | WCAG, a11y | MEDIUM |
| HL-5 | Performance | N+1, memory, escala | MEDIUM |
| HL-6 | Integration | APIs, contratos | STRONG |
| HL-7 | Final Check | Revisão humana final | STRONG |

---

## Módulo HLS-LAY-001: HL-1 UI/UX Review

```
ID: HLS-LAY-001
Nome: UIUXLayer
Caminho: src/hl_mcp/layers/hl1_uiux.py
Dependências: HLS-ENG-001, HLS-LLM-004
Exports: UIUXLayer, UIUXConfig, UIUXFindings
Linhas: ~280
```

### Código

```python
"""
HLS-LAY-001: HL-1 UI/UX Review Layer
====================================

Validação de interface de usuário e experiência.

Responsabilidades:
- Avaliar clareza de labels e mensagens
- Verificar fluxo de navegação
- Identificar elementos confusos
- Validar feedback ao usuário
- Checar consistência visual

Veto Power: WEAK
Pode sugerir melhorias mas não bloqueia deploy.

Exemplo:
    >>> layer = UIUXLayer(llm_client)
    >>> result = await layer.validate(
    ...     target="Login form",
    ...     screenshot_path="./login.png",
    ... )
    >>> for finding in result.findings:
    ...     print(f"[{finding.severity}] {finding.title}")

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

logger = logging.getLogger(__name__)


class UIUXCategory(str, Enum):
    """Categorias de findings UI/UX."""

    CLARITY = "clarity"           # Clareza de texto/labels
    NAVIGATION = "navigation"     # Fluxo de navegação
    FEEDBACK = "feedback"         # Feedback ao usuário
    CONSISTENCY = "consistency"   # Consistência visual
    AFFORDANCE = "affordance"     # Indicações de interação
    ERROR_HANDLING = "error_handling"  # Tratamento de erros UI


@dataclass
class UIUXConfig:
    """
    Configuração da layer HL-1.

    Attributes:
        check_labels: Verificar clareza de labels
        check_navigation: Verificar fluxo de navegação
        check_feedback: Verificar feedback ao usuário
        check_consistency: Verificar consistência visual
        check_mobile: Considerar experiência mobile
        min_contrast_ratio: Ratio mínimo de contraste
        max_label_length: Tamanho máximo de labels
    """

    check_labels: bool = True
    check_navigation: bool = True
    check_feedback: bool = True
    check_consistency: bool = True
    check_mobile: bool = True
    min_contrast_ratio: float = 4.5
    max_label_length: int = 50


@dataclass
class UIUXFinding:
    """Finding específico de UI/UX."""

    category: UIUXCategory
    severity: str
    title: str
    description: str
    element: Optional[str] = None
    selector: Optional[str] = None
    recommendation: str = ""
    screenshot_region: Optional[dict] = None

    def to_dict(self) -> dict:
        """Converte para dicionário."""
        return {
            "id": f"uiux-{self.category.value}-{hash(self.title) % 10000:04d}",
            "category": f"uiux/{self.category.value}",
            "severity": self.severity,
            "title": self.title,
            "description": self.description,
            "element": self.element,
            "selector": self.selector,
            "remediation": self.recommendation,
        }


@dataclass
class UIUXResult:
    """Resultado da validação UI/UX."""

    success: bool
    findings: list[UIUXFinding] = field(default_factory=list)
    veto_level: str = "NONE"
    confidence: float = 0.0
    duration_ms: float = 0
    suggestions: list[str] = field(default_factory=list)

    @property
    def has_critical(self) -> bool:
        """Se há issues críticos."""
        return any(f.severity == "high" for f in self.findings)

    def to_dict(self) -> dict:
        """Converte para dicionário."""
        return {
            "layer": "HL-1",
            "layer_name": "UI/UX Review",
            "success": self.success,
            "finding_count": len(self.findings),
            "veto_level": self.veto_level,
            "confidence": self.confidence,
            "findings": [f.to_dict() for f in self.findings],
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


UIUX_SYSTEM_PROMPT = """You are a UI/UX expert conducting a user interface review.

Your focus areas:
1. CLARITY: Are labels, messages, and instructions clear and understandable?
2. NAVIGATION: Is the flow logical and intuitive?
3. FEEDBACK: Does the UI provide appropriate feedback for user actions?
4. CONSISTENCY: Are visual elements and patterns consistent?
5. ERROR HANDLING: Are error states handled gracefully?

For each issue found, provide:
- Category (clarity, navigation, feedback, consistency, error_handling)
- Severity (critical, high, medium, low, info)
- Title (brief description)
- Description (detailed explanation)
- Recommendation (how to fix)

Respond in JSON format:
{
    "findings": [
        {
            "category": "clarity",
            "severity": "medium",
            "title": "Unclear button label",
            "description": "The 'Submit' button doesn't indicate what will happen",
            "element": "Submit button",
            "recommendation": "Change to 'Create Account' or 'Sign Up'"
        }
    ],
    "suggestions": ["Consider adding a progress indicator"],
    "overall_score": 7.5,
    "veto_level": "NONE"
}
"""


class UIUXLayer:
    """
    HL-1: UI/UX Review Layer.

    Analisa interface de usuário para problemas de
    usabilidade, clareza e experiência.

    Veto Power: WEAK
    Pode identificar problemas mas raramente bloqueia.

    Example:
        >>> layer = UIUXLayer(llm_client)
        >>> result = await layer.validate(
        ...     target="Checkout page",
        ...     html_snippet="<form>...</form>",
        ... )
    """

    LAYER_ID = "HL-1"
    LAYER_NAME = "UI/UX Review"
    VETO_POWER = "WEAK"

    def __init__(
        self,
        llm_client: LLMClientProtocol,
        config: Optional[UIUXConfig] = None,
    ):
        """
        Inicializa a layer.

        Args:
            llm_client: Cliente LLM para análise
            config: Configuração da layer
        """
        self.llm = llm_client
        self.config = config or UIUXConfig()

    async def validate(
        self,
        target: str,
        html_snippet: Optional[str] = None,
        screenshot_path: Optional[str] = None,
        context: Optional[dict] = None,
    ) -> UIUXResult:
        """
        Executa validação UI/UX.

        Args:
            target: Descrição do que está sendo validado
            html_snippet: HTML do componente (se disponível)
            screenshot_path: Path do screenshot
            context: Contexto adicional

        Returns:
            UIUXResult com findings
        """
        start = datetime.utcnow()

        logger.info(f"HL-1 validando: {target}")

        # Monta prompt
        prompt = self._build_prompt(target, html_snippet, context)

        try:
            # Chama LLM
            response = await self.llm.complete(
                prompt=prompt,
                system_prompt=UIUX_SYSTEM_PROMPT,
            )

            # Parseia resposta
            parsed = self._parse_response(response)

            duration = (datetime.utcnow() - start).total_seconds() * 1000

            # Converte para UIUXFindings
            findings = [
                UIUXFinding(
                    category=UIUXCategory(f.get("category", "clarity")),
                    severity=f.get("severity", "medium"),
                    title=f.get("title", ""),
                    description=f.get("description", ""),
                    element=f.get("element"),
                    selector=f.get("selector"),
                    recommendation=f.get("recommendation", ""),
                )
                for f in parsed.get("findings", [])
            ]

            # Determina veto (HL-1 quase nunca veta)
            veto = self._determine_veto(findings)

            return UIUXResult(
                success=True,
                findings=findings,
                veto_level=veto,
                confidence=parsed.get("overall_score", 0.7) / 10,
                duration_ms=duration,
                suggestions=parsed.get("suggestions", []),
            )

        except Exception as e:
            duration = (datetime.utcnow() - start).total_seconds() * 1000
            logger.error(f"HL-1 erro: {e}")

            return UIUXResult(
                success=False,
                duration_ms=duration,
            )

    def _build_prompt(
        self,
        target: str,
        html: Optional[str],
        context: Optional[dict],
    ) -> str:
        """Constrói prompt para análise."""
        parts = [f"## Target\n{target}"]

        if html:
            parts.append(f"## HTML\n```html\n{html[:2000]}\n```")

        if context:
            ctx_str = "\n".join(f"- {k}: {v}" for k, v in context.items())
            parts.append(f"## Context\n{ctx_str}")

        parts.append(
            "## Task\n"
            "Analyze the UI/UX of this component. "
            "Focus on clarity, usability, and user experience."
        )

        return "\n\n".join(parts)

    def _parse_response(self, response: str) -> dict:
        """Parseia resposta do LLM."""
        import json

        # Tenta extrair JSON
        try:
            # Procura bloco JSON
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
            elif "{" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end]
            else:
                return {"findings": []}

            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("Falha ao parsear resposta JSON")
            return {"findings": []}

    def _determine_veto(self, findings: list[UIUXFinding]) -> str:
        """Determina nível de veto baseado nos findings."""
        # HL-1 tem veto power WEAK
        # Só veta em casos extremos de UX terrível

        critical_count = sum(
            1 for f in findings
            if f.severity in ["critical", "high"]
        )

        if critical_count >= 5:
            return "WEAK"  # Máximo que HL-1 pode dar

        return "NONE"


# Exports
__all__ = [
    "UIUXCategory",
    "UIUXConfig",
    "UIUXFinding",
    "UIUXResult",
    "UIUXLayer",
    "UIUX_SYSTEM_PROMPT",
]
```

---

## Módulo HLS-LAY-002: HL-2 Security Scan

```
ID: HLS-LAY-002
Nome: SecurityLayer
Caminho: src/hl_mcp/layers/hl2_security.py
Dependências: HLS-ENG-001, HLS-LLM-004
Exports: SecurityLayer, SecurityConfig, SecurityFinding
Linhas: ~350
```

### Código

```python
"""
HLS-LAY-002: HL-2 Security Scan Layer
=====================================

Validação de segurança usando análise de código.

Responsabilidades:
- Detectar vulnerabilidades OWASP Top 10
- Identificar injection (SQL, XSS, Command)
- Verificar autenticação e autorização
- Detectar exposição de dados sensíveis
- Validar configurações de segurança

Veto Power: STRONG
Pode e deve bloquear deploy em vulnerabilidades.

Exemplo:
    >>> layer = SecurityLayer(llm_client)
    >>> result = await layer.validate(
    ...     code="def login(user, pwd): db.query(f'SELECT * FROM users WHERE...')",
    ...     language="python",
    ... )
    >>> if result.veto_level == "STRONG":
    ...     print("BLOQUEADO: Vulnerabilidade crítica!")

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

logger = logging.getLogger(__name__)


class VulnerabilityType(str, Enum):
    """Tipos de vulnerabilidade (OWASP-aligned)."""

    INJECTION = "injection"           # A03:2021
    BROKEN_AUTH = "broken_auth"       # A07:2021
    SENSITIVE_DATA = "sensitive_data" # A02:2021
    XXE = "xxe"                        # XML External Entities
    BROKEN_ACCESS = "broken_access"   # A01:2021
    MISCONFIG = "misconfig"           # A05:2021
    XSS = "xss"                        # A03:2021
    INSECURE_DESER = "insecure_deser" # A08:2021
    COMPONENTS = "vulnerable_components"  # A06:2021
    LOGGING = "logging_monitoring"    # A09:2021
    SSRF = "ssrf"                      # A10:2021


class CWEReference(str, Enum):
    """Referências CWE comuns."""

    CWE_89 = "CWE-89"    # SQL Injection
    CWE_79 = "CWE-79"    # XSS
    CWE_78 = "CWE-78"    # OS Command Injection
    CWE_22 = "CWE-22"    # Path Traversal
    CWE_352 = "CWE-352"  # CSRF
    CWE_311 = "CWE-311"  # Missing Encryption
    CWE_798 = "CWE-798"  # Hardcoded Credentials
    CWE_502 = "CWE-502"  # Deserialization
    CWE_918 = "CWE-918"  # SSRF


@dataclass
class SecurityConfig:
    """
    Configuração da layer HL-2.

    Attributes:
        check_injection: Verificar SQL/Command injection
        check_xss: Verificar Cross-Site Scripting
        check_auth: Verificar autenticação
        check_secrets: Verificar secrets hardcoded
        check_crypto: Verificar uso de criptografia
        owasp_version: Versão OWASP a usar
        severity_threshold: Threshold para bloquear
    """

    check_injection: bool = True
    check_xss: bool = True
    check_auth: bool = True
    check_secrets: bool = True
    check_crypto: bool = True
    owasp_version: str = "2021"
    severity_threshold: str = "high"


@dataclass
class SecurityFinding:
    """Finding específico de segurança."""

    vuln_type: VulnerabilityType
    severity: str
    title: str
    description: str
    cwe: Optional[str] = None
    owasp: Optional[str] = None
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    remediation: str = ""
    references: list[str] = field(default_factory=list)

    @property
    def is_critical(self) -> bool:
        """Se é uma vulnerabilidade crítica."""
        return self.severity in ["critical", "high"]

    def to_dict(self) -> dict:
        """Converte para dicionário."""
        return {
            "id": f"sec-{self.vuln_type.value}-{hash(self.title) % 10000:04d}",
            "category": f"security/{self.vuln_type.value}",
            "severity": self.severity,
            "title": self.title,
            "description": self.description,
            "cwe": self.cwe,
            "owasp": self.owasp,
            "line": self.line_number,
            "evidence": self.code_snippet,
            "remediation": self.remediation,
            "references": self.references,
        }


@dataclass
class SecurityResult:
    """Resultado da validação de segurança."""

    success: bool
    findings: list[SecurityFinding] = field(default_factory=list)
    veto_level: str = "NONE"
    confidence: float = 0.0
    duration_ms: float = 0
    vulnerabilities_by_severity: dict = field(default_factory=dict)

    @property
    def critical_count(self) -> int:
        """Número de vulnerabilidades críticas."""
        return sum(1 for f in self.findings if f.is_critical)

    @property
    def should_block(self) -> bool:
        """Se deve bloquear o deploy."""
        return self.veto_level == "STRONG"

    def to_dict(self) -> dict:
        """Converte para dicionário."""
        return {
            "layer": "HL-2",
            "layer_name": "Security Scan",
            "success": self.success,
            "finding_count": len(self.findings),
            "critical_count": self.critical_count,
            "veto_level": self.veto_level,
            "should_block": self.should_block,
            "confidence": self.confidence,
            "findings": [f.to_dict() for f in self.findings],
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


SECURITY_SYSTEM_PROMPT = """You are a security expert performing code review for vulnerabilities.

FOCUS ON OWASP TOP 10 (2021):
A01: Broken Access Control
A02: Cryptographic Failures
A03: Injection (SQL, XSS, Command)
A04: Insecure Design
A05: Security Misconfiguration
A06: Vulnerable Components
A07: Authentication Failures
A08: Software/Data Integrity Failures
A09: Security Logging Failures
A10: Server-Side Request Forgery

FOR EACH VULNERABILITY FOUND:
- vuln_type: Type from OWASP categories
- severity: critical, high, medium, low
- title: Brief description
- description: Detailed explanation of the risk
- cwe: CWE identifier (e.g., CWE-89 for SQL injection)
- owasp: OWASP category (e.g., A03:2021)
- line_number: Line where issue exists (if known)
- code_snippet: The vulnerable code
- remediation: How to fix it

BE STRICT. Security vulnerabilities are serious.
If you find SQL injection, XSS, or auth bypass, mark as CRITICAL.

Respond in JSON:
{
    "findings": [...],
    "veto_level": "NONE|WEAK|MEDIUM|STRONG",
    "confidence": 0.0-1.0,
    "summary": "Brief security assessment"
}
"""


class SecurityLayer:
    """
    HL-2: Security Scan Layer.

    Analisa código para vulnerabilidades de segurança
    usando OWASP Top 10 como referência.

    Veto Power: STRONG
    DEVE bloquear em vulnerabilidades críticas.

    Example:
        >>> layer = SecurityLayer(llm_client)
        >>> result = await layer.validate(
        ...     code="db.query(f'SELECT * FROM users WHERE id={user_id}')",
        ...     language="python",
        ... )
        >>> print(result.veto_level)  # "STRONG"
    """

    LAYER_ID = "HL-2"
    LAYER_NAME = "Security Scan"
    VETO_POWER = "STRONG"

    def __init__(
        self,
        llm_client: LLMClientProtocol,
        config: Optional[SecurityConfig] = None,
    ):
        """
        Inicializa a layer.

        Args:
            llm_client: Cliente LLM para análise
            config: Configuração da layer
        """
        self.llm = llm_client
        self.config = config or SecurityConfig()

    async def validate(
        self,
        code: str,
        language: str = "python",
        context: Optional[dict] = None,
        file_path: Optional[str] = None,
    ) -> SecurityResult:
        """
        Executa scan de segurança.

        Args:
            code: Código a analisar
            language: Linguagem do código
            context: Contexto adicional
            file_path: Path do arquivo

        Returns:
            SecurityResult com vulnerabilidades
        """
        start = datetime.utcnow()

        logger.info(f"HL-2 escaneando: {language} ({len(code)} chars)")

        # Monta prompt
        prompt = self._build_prompt(code, language, context, file_path)

        try:
            # Chama LLM
            response = await self.llm.complete(
                prompt=prompt,
                system_prompt=SECURITY_SYSTEM_PROMPT,
            )

            # Parseia resposta
            parsed = self._parse_response(response)

            duration = (datetime.utcnow() - start).total_seconds() * 1000

            # Converte para SecurityFindings
            findings = self._convert_findings(parsed.get("findings", []))

            # Determina veto
            veto = self._determine_veto(findings)

            # Agrupa por severidade
            by_severity = {}
            for f in findings:
                by_severity.setdefault(f.severity, 0)
                by_severity[f.severity] += 1

            return SecurityResult(
                success=True,
                findings=findings,
                veto_level=veto,
                confidence=parsed.get("confidence", 0.8),
                duration_ms=duration,
                vulnerabilities_by_severity=by_severity,
            )

        except Exception as e:
            duration = (datetime.utcnow() - start).total_seconds() * 1000
            logger.error(f"HL-2 erro: {e}")

            return SecurityResult(
                success=False,
                duration_ms=duration,
            )

    def _build_prompt(
        self,
        code: str,
        language: str,
        context: Optional[dict],
        file_path: Optional[str],
    ) -> str:
        """Constrói prompt para análise."""
        parts = []

        if file_path:
            parts.append(f"## File: {file_path}")

        parts.append(f"## Language: {language}")
        parts.append(f"## Code\n```{language}\n{code}\n```")

        if context:
            ctx_str = "\n".join(f"- {k}: {v}" for k, v in context.items())
            parts.append(f"## Context\n{ctx_str}")

        parts.append(
            "## Task\n"
            "Perform a security review of this code. "
            "Look for OWASP Top 10 vulnerabilities. "
            "Be thorough and flag any potential security issues."
        )

        return "\n\n".join(parts)

    def _parse_response(self, response: str) -> dict:
        """Parseia resposta do LLM."""
        import json

        try:
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
            elif "{" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end]
            else:
                return {"findings": []}

            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("Falha ao parsear resposta JSON")
            return {"findings": []}

    def _convert_findings(self, raw_findings: list[dict]) -> list[SecurityFinding]:
        """Converte findings raw para SecurityFinding."""
        findings = []

        for f in raw_findings:
            # Mapeia vuln_type
            vuln_type_str = f.get("vuln_type", "misconfig")
            try:
                vuln_type = VulnerabilityType(vuln_type_str)
            except ValueError:
                vuln_type = VulnerabilityType.MISCONFIG

            findings.append(SecurityFinding(
                vuln_type=vuln_type,
                severity=f.get("severity", "medium"),
                title=f.get("title", "Security Issue"),
                description=f.get("description", ""),
                cwe=f.get("cwe"),
                owasp=f.get("owasp"),
                line_number=f.get("line_number"),
                code_snippet=f.get("code_snippet"),
                remediation=f.get("remediation", ""),
                references=f.get("references", []),
            ))

        return findings

    def _determine_veto(self, findings: list[SecurityFinding]) -> str:
        """Determina nível de veto baseado nos findings."""
        # HL-2 tem veto power STRONG

        critical_types = {
            VulnerabilityType.INJECTION,
            VulnerabilityType.XSS,
            VulnerabilityType.BROKEN_AUTH,
            VulnerabilityType.SSRF,
        }

        for f in findings:
            if f.severity == "critical":
                return "STRONG"
            if f.severity == "high" and f.vuln_type in critical_types:
                return "STRONG"

        if any(f.severity == "high" for f in findings):
            return "MEDIUM"

        if findings:
            return "WEAK"

        return "NONE"


# Exports
__all__ = [
    "VulnerabilityType",
    "CWEReference",
    "SecurityConfig",
    "SecurityFinding",
    "SecurityResult",
    "SecurityLayer",
    "SECURITY_SYSTEM_PROMPT",
]
```

---

## Módulo HLS-LAY-003: HL-3 Edge Cases

```
ID: HLS-LAY-003
Nome: EdgeCasesLayer
Caminho: src/hl_mcp/layers/hl3_edge_cases.py
Dependências: HLS-ENG-001, HLS-LLM-004
Exports: EdgeCasesLayer, EdgeCaseConfig, EdgeCaseFinding
Linhas: ~250
```

### Código

```python
"""
HLS-LAY-003: HL-3 Edge Cases Layer
==================================

Validação de casos limite e condições de erro.

Responsabilidades:
- Identificar boundary conditions
- Testar valores null/undefined
- Verificar race conditions
- Validar tratamento de erros
- Detectar overflow/underflow

Veto Power: MEDIUM
Pode bloquear se edge cases críticos não forem tratados.

Exemplo:
    >>> layer = EdgeCasesLayer(llm_client)
    >>> result = await layer.validate(
    ...     code="def divide(a, b): return a / b",
    ...     function_signature="divide(a: int, b: int) -> float",
    ... )

Changelog:
    - v1.0.0 (2026-02-01): Implementação inicial
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Protocol
import logging

logger = logging.getLogger(__name__)


class EdgeCaseCategory(str, Enum):
    """Categorias de edge cases."""

    BOUNDARY = "boundary"          # Limites de valores
    NULL_UNDEFINED = "null"        # Valores nulos
    EMPTY = "empty"                # Coleções vazias
    RACE_CONDITION = "race"        # Condições de corrida
    OVERFLOW = "overflow"          # Overflow numérico
    TYPE_COERCION = "type"         # Coerção de tipos
    TIMEOUT = "timeout"            # Timeouts
    CONCURRENCY = "concurrency"    # Problemas de concorrência


@dataclass
class EdgeCaseConfig:
    """Configuração da layer HL-3."""

    check_boundaries: bool = True
    check_nulls: bool = True
    check_empty: bool = True
    check_race: bool = True
    check_overflow: bool = True
    generate_test_cases: bool = True


@dataclass
class EdgeCaseFinding:
    """Finding específico de edge case."""

    category: EdgeCaseCategory
    severity: str
    title: str
    description: str
    input_example: Optional[str] = None
    expected_behavior: Optional[str] = None
    actual_behavior: Optional[str] = None
    test_case: Optional[str] = None

    def to_dict(self) -> dict:
        """Converte para dicionário."""
        return {
            "id": f"edge-{self.category.value}-{hash(self.title) % 10000:04d}",
            "category": f"edge_case/{self.category.value}",
            "severity": self.severity,
            "title": self.title,
            "description": self.description,
            "input_example": self.input_example,
            "expected_behavior": self.expected_behavior,
            "test_case": self.test_case,
        }


@dataclass
class EdgeCasesResult:
    """Resultado da validação de edge cases."""

    success: bool
    findings: list[EdgeCaseFinding] = field(default_factory=list)
    veto_level: str = "NONE"
    confidence: float = 0.0
    duration_ms: float = 0
    generated_tests: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Converte para dicionário."""
        return {
            "layer": "HL-3",
            "layer_name": "Edge Cases",
            "success": self.success,
            "finding_count": len(self.findings),
            "veto_level": self.veto_level,
            "confidence": self.confidence,
            "generated_tests_count": len(self.generated_tests),
            "findings": [f.to_dict() for f in self.findings],
        }


class LLMClientProtocol(Protocol):
    """Protocolo para cliente LLM."""

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        ...


EDGE_CASES_SYSTEM_PROMPT = """You are a QA expert specialized in finding edge cases and boundary conditions.

LOOK FOR:
1. BOUNDARY CONDITIONS: Min/max values, off-by-one errors
2. NULL/UNDEFINED: Missing or null inputs
3. EMPTY COLLECTIONS: Empty arrays, strings, maps
4. RACE CONDITIONS: Concurrent access issues
5. OVERFLOW/UNDERFLOW: Numeric limits
6. TYPE ISSUES: Type coercion problems
7. TIMEOUT SCENARIOS: Long-running operations

For each edge case found:
- category: Type of edge case
- severity: How bad if not handled (critical, high, medium, low)
- title: Brief description
- description: Detailed explanation
- input_example: Example input that triggers the issue
- expected_behavior: What should happen
- test_case: Python test code to verify

Respond in JSON:
{
    "findings": [...],
    "generated_tests": ["def test_divide_by_zero(): ..."],
    "veto_level": "NONE|WEAK|MEDIUM|STRONG",
    "confidence": 0.0-1.0
}
"""


class EdgeCasesLayer:
    """
    HL-3: Edge Cases Layer.

    Identifica casos limite que podem causar falhas.

    Veto Power: MEDIUM
    Bloqueia se edge cases críticos não forem tratados.

    Example:
        >>> layer = EdgeCasesLayer(llm_client)
        >>> result = await layer.validate(
        ...     code="def process(items): return items[0]",
        ... )
        >>> # Vai encontrar: empty list edge case
    """

    LAYER_ID = "HL-3"
    LAYER_NAME = "Edge Cases"
    VETO_POWER = "MEDIUM"

    def __init__(
        self,
        llm_client: LLMClientProtocol,
        config: Optional[EdgeCaseConfig] = None,
    ):
        self.llm = llm_client
        self.config = config or EdgeCaseConfig()

    async def validate(
        self,
        code: str,
        function_signature: Optional[str] = None,
        context: Optional[dict] = None,
    ) -> EdgeCasesResult:
        """Executa validação de edge cases."""
        start = datetime.utcnow()

        logger.info(f"HL-3 analisando edge cases")

        prompt = self._build_prompt(code, function_signature, context)

        try:
            response = await self.llm.complete(
                prompt=prompt,
                system_prompt=EDGE_CASES_SYSTEM_PROMPT,
            )

            parsed = self._parse_response(response)
            duration = (datetime.utcnow() - start).total_seconds() * 1000

            findings = [
                EdgeCaseFinding(
                    category=EdgeCaseCategory(f.get("category", "boundary")),
                    severity=f.get("severity", "medium"),
                    title=f.get("title", ""),
                    description=f.get("description", ""),
                    input_example=f.get("input_example"),
                    expected_behavior=f.get("expected_behavior"),
                    test_case=f.get("test_case"),
                )
                for f in parsed.get("findings", [])
            ]

            veto = self._determine_veto(findings)

            return EdgeCasesResult(
                success=True,
                findings=findings,
                veto_level=veto,
                confidence=parsed.get("confidence", 0.7),
                duration_ms=duration,
                generated_tests=parsed.get("generated_tests", []),
            )

        except Exception as e:
            duration = (datetime.utcnow() - start).total_seconds() * 1000
            logger.error(f"HL-3 erro: {e}")
            return EdgeCasesResult(success=False, duration_ms=duration)

    def _build_prompt(
        self,
        code: str,
        signature: Optional[str],
        context: Optional[dict],
    ) -> str:
        parts = ["## Code\n```\n" + code + "\n```"]

        if signature:
            parts.append(f"## Function Signature\n`{signature}`")

        if context:
            ctx = "\n".join(f"- {k}: {v}" for k, v in context.items())
            parts.append(f"## Context\n{ctx}")

        parts.append(
            "## Task\n"
            "Analyze this code for edge cases. "
            "Think about what inputs could break it."
        )

        return "\n\n".join(parts)

    def _parse_response(self, response: str) -> dict:
        import json
        try:
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
            elif "{" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end]
            else:
                return {"findings": []}
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {"findings": []}

    def _determine_veto(self, findings: list[EdgeCaseFinding]) -> str:
        critical_categories = {
            EdgeCaseCategory.RACE_CONDITION,
            EdgeCaseCategory.OVERFLOW,
        }

        for f in findings:
            if f.severity == "critical":
                return "MEDIUM"  # Max para HL-3
            if f.severity == "high" and f.category in critical_categories:
                return "MEDIUM"

        if any(f.severity == "high" for f in findings):
            return "WEAK"

        return "NONE"


__all__ = [
    "EdgeCaseCategory",
    "EdgeCaseConfig",
    "EdgeCaseFinding",
    "EdgeCasesResult",
    "EdgeCasesLayer",
]
```

---

## Módulo HLS-LAY-004: HL-4 Accessibility

```
ID: HLS-LAY-004
Nome: AccessibilityLayer
Caminho: src/hl_mcp/layers/hl4_accessibility.py
Dependências: HLS-ENG-001, HLS-BRW-005
Exports: AccessibilityLayer, A11yConfig, A11yFinding
Linhas: ~220
```

### Código

```python
"""
HLS-LAY-004: HL-4 Accessibility Layer
=====================================

Validação de acessibilidade (WCAG compliance).

Responsabilidades:
- Verificar conformidade WCAG 2.1
- Validar uso de ARIA
- Checar navegação por teclado
- Verificar contraste de cores
- Validar screen reader compatibility

Veto Power: MEDIUM
Pode bloquear se acessibilidade crítica falhar.

Changelog:
    - v1.0.0 (2026-02-01): Implementação inicial
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Protocol, Any
import logging

logger = logging.getLogger(__name__)


class WCAGLevel(str, Enum):
    """Níveis WCAG."""
    A = "A"
    AA = "AA"
    AAA = "AAA"


class A11yCategory(str, Enum):
    """Categorias de acessibilidade."""

    PERCEIVABLE = "perceivable"       # 1.x
    OPERABLE = "operable"             # 2.x
    UNDERSTANDABLE = "understandable" # 3.x
    ROBUST = "robust"                 # 4.x


@dataclass
class A11yConfig:
    """Configuração da layer HL-4."""

    wcag_level: WCAGLevel = WCAGLevel.AA
    check_contrast: bool = True
    check_keyboard: bool = True
    check_aria: bool = True
    check_alt_text: bool = True
    use_axe_core: bool = True


@dataclass
class A11yFinding:
    """Finding de acessibilidade."""

    category: A11yCategory
    severity: str
    title: str
    description: str
    wcag_criterion: Optional[str] = None
    element: Optional[str] = None
    selector: Optional[str] = None
    remediation: str = ""

    def to_dict(self) -> dict:
        return {
            "id": f"a11y-{self.category.value}-{hash(self.title) % 10000:04d}",
            "category": f"accessibility/{self.category.value}",
            "severity": self.severity,
            "title": self.title,
            "description": self.description,
            "wcag": self.wcag_criterion,
            "selector": self.selector,
            "remediation": self.remediation,
        }


@dataclass
class A11yResult:
    """Resultado da validação de acessibilidade."""

    success: bool
    findings: list[A11yFinding] = field(default_factory=list)
    veto_level: str = "NONE"
    confidence: float = 0.0
    duration_ms: float = 0
    wcag_level_tested: WCAGLevel = WCAGLevel.AA
    axe_violations: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "layer": "HL-4",
            "layer_name": "Accessibility",
            "success": self.success,
            "finding_count": len(self.findings),
            "veto_level": self.veto_level,
            "wcag_level": self.wcag_level_tested.value,
            "axe_violation_count": len(self.axe_violations),
            "findings": [f.to_dict() for f in self.findings],
        }


class LLMClientProtocol(Protocol):
    async def complete(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        ...


A11Y_SYSTEM_PROMPT = """You are an accessibility expert reviewing for WCAG 2.1 compliance.

WCAG PRINCIPLES:
1. PERCEIVABLE: Info must be presentable in ways users can perceive
2. OPERABLE: UI components must be operable
3. UNDERSTANDABLE: Info and UI operation must be understandable
4. ROBUST: Content must be robust for various user agents

CHECK FOR:
- Missing alt text on images
- Poor color contrast
- Missing form labels
- Keyboard navigation issues
- Missing ARIA attributes
- Focus management problems
- Screen reader compatibility

Respond in JSON:
{
    "findings": [
        {
            "category": "perceivable",
            "severity": "high",
            "title": "Missing alt text",
            "description": "Image lacks alternative text",
            "wcag_criterion": "1.1.1",
            "element": "<img src='logo.png'>",
            "remediation": "Add alt='Company Logo'"
        }
    ],
    "veto_level": "NONE|WEAK|MEDIUM|STRONG",
    "confidence": 0.0-1.0
}
"""


class AccessibilityLayer:
    """
    HL-4: Accessibility Layer.

    Valida conformidade WCAG e acessibilidade.

    Veto Power: MEDIUM
    """

    LAYER_ID = "HL-4"
    LAYER_NAME = "Accessibility"
    VETO_POWER = "MEDIUM"

    def __init__(
        self,
        llm_client: LLMClientProtocol,
        config: Optional[A11yConfig] = None,
    ):
        self.llm = llm_client
        self.config = config or A11yConfig()

    async def validate(
        self,
        html: Optional[str] = None,
        page: Optional[Any] = None,  # Playwright Page
        url: Optional[str] = None,
    ) -> A11yResult:
        """Executa validação de acessibilidade."""
        start = datetime.utcnow()

        logger.info(f"HL-4 validando acessibilidade")

        axe_violations = []

        # Se tiver page e axe-core habilitado, usa axe
        if page and self.config.use_axe_core:
            try:
                from hl_mcp.browser.accessibility import AccessibilityChecker
                checker = AccessibilityChecker(page, self.config.wcag_level)
                axe_result = await checker.check()
                axe_violations = [v.to_finding() for v in axe_result.violations]
            except Exception as e:
                logger.warning(f"axe-core falhou: {e}")

        # Análise via LLM
        if html:
            prompt = self._build_prompt(html)
            try:
                response = await self.llm.complete(
                    prompt=prompt,
                    system_prompt=A11Y_SYSTEM_PROMPT,
                )
                parsed = self._parse_response(response)
            except Exception as e:
                logger.error(f"HL-4 LLM erro: {e}")
                parsed = {"findings": []}
        else:
            parsed = {"findings": []}

        duration = (datetime.utcnow() - start).total_seconds() * 1000

        # Combina findings de axe + LLM
        findings = []

        for f in parsed.get("findings", []):
            findings.append(A11yFinding(
                category=A11yCategory(f.get("category", "perceivable")),
                severity=f.get("severity", "medium"),
                title=f.get("title", ""),
                description=f.get("description", ""),
                wcag_criterion=f.get("wcag_criterion"),
                element=f.get("element"),
                selector=f.get("selector"),
                remediation=f.get("remediation", ""),
            ))

        veto = self._determine_veto(findings, axe_violations)

        return A11yResult(
            success=True,
            findings=findings,
            veto_level=veto,
            confidence=parsed.get("confidence", 0.75),
            duration_ms=duration,
            wcag_level_tested=self.config.wcag_level,
            axe_violations=axe_violations,
        )

    def _build_prompt(self, html: str) -> str:
        return f"""## HTML
```html
{html[:3000]}
```

## Task
Review this HTML for WCAG {self.config.wcag_level.value} accessibility issues.
"""

    def _parse_response(self, response: str) -> dict:
        import json
        try:
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
            else:
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end]
            return json.loads(json_str)
        except:
            return {"findings": []}

    def _determine_veto(
        self,
        findings: list[A11yFinding],
        axe_violations: list[dict],
    ) -> str:
        # Conta issues críticos
        critical_count = sum(1 for f in findings if f.severity in ["critical", "high"])
        critical_count += sum(
            1 for v in axe_violations
            if v.get("severity") in ["critical", "high"]
        )

        if critical_count >= 3:
            return "MEDIUM"
        if critical_count >= 1:
            return "WEAK"
        return "NONE"


__all__ = [
    "WCAGLevel",
    "A11yCategory",
    "A11yConfig",
    "A11yFinding",
    "A11yResult",
    "AccessibilityLayer",
]
```

---

## Módulos HL-5, HL-6, HL-7 (Resumidos)

Para manter o documento conciso, os módulos restantes seguem a mesma estrutura:

### HLS-LAY-005: HL-5 Performance Layer

```python
"""
HLS-LAY-005: HL-5 Performance Layer
===================================

Validação de performance e escalabilidade.

Foco:
- N+1 queries
- Missing indexes
- Memory leaks
- Unnecessary computations
- Scalability issues

Veto Power: MEDIUM
"""

# Estrutura similar às anteriores
# ~220 linhas
```

### HLS-LAY-006: HL-6 Integration Layer

```python
"""
HLS-LAY-006: HL-6 Integration Layer
===================================

Validação de integração e contratos de API.

Foco:
- API contracts
- Data consistency
- Error propagation
- System boundaries
- Backwards compatibility

Veto Power: STRONG
"""

# Estrutura similar às anteriores
# ~220 linhas
```

### HLS-LAY-007: HL-7 Final Human Check

```python
"""
HLS-LAY-007: HL-7 Final Human Check Layer
=========================================

Revisão humana final - última linha de defesa.

Foco:
- Anything missed by previous layers
- Overall quality assessment
- Business logic validation
- User experience holistic view

Veto Power: STRONG
"""

# Estrutura similar às anteriores
# ~200 linhas
```

---

## Layers __init__.py

```python
"""
HLS-LAY: Human Layers Module
============================

As 7 Human Layers de validação.

Exports:
    - UIUXLayer (HL-1)
    - SecurityLayer (HL-2)
    - EdgeCasesLayer (HL-3)
    - AccessibilityLayer (HL-4)
    - PerformanceLayer (HL-5)
    - IntegrationLayer (HL-6)
    - FinalCheckLayer (HL-7)
"""

from .hl1_uiux import (
    UIUXCategory,
    UIUXConfig,
    UIUXFinding,
    UIUXResult,
    UIUXLayer,
)

from .hl2_security import (
    VulnerabilityType,
    CWEReference,
    SecurityConfig,
    SecurityFinding,
    SecurityResult,
    SecurityLayer,
)

from .hl3_edge_cases import (
    EdgeCaseCategory,
    EdgeCaseConfig,
    EdgeCaseFinding,
    EdgeCasesResult,
    EdgeCasesLayer,
)

from .hl4_accessibility import (
    WCAGLevel,
    A11yCategory,
    A11yConfig,
    A11yFinding,
    A11yResult,
    AccessibilityLayer,
)

# HL-5, HL-6, HL-7 seguem o mesmo padrão

# Layer registry para fácil acesso
LAYER_REGISTRY = {
    "HL-1": UIUXLayer,
    "HL-2": SecurityLayer,
    "HL-3": EdgeCasesLayer,
    "HL-4": AccessibilityLayer,
    # "HL-5": PerformanceLayer,
    # "HL-6": IntegrationLayer,
    # "HL-7": FinalCheckLayer,
}


def get_layer(layer_id: str):
    """Obtém classe de layer por ID."""
    return LAYER_REGISTRY.get(layer_id)


__all__ = [
    # HL-1
    "UIUXCategory",
    "UIUXConfig",
    "UIUXFinding",
    "UIUXResult",
    "UIUXLayer",
    # HL-2
    "VulnerabilityType",
    "CWEReference",
    "SecurityConfig",
    "SecurityFinding",
    "SecurityResult",
    "SecurityLayer",
    # HL-3
    "EdgeCaseCategory",
    "EdgeCaseConfig",
    "EdgeCaseFinding",
    "EdgeCasesResult",
    "EdgeCasesLayer",
    # HL-4
    "WCAGLevel",
    "A11yCategory",
    "A11yConfig",
    "A11yFinding",
    "A11yResult",
    "AccessibilityLayer",
    # Registry
    "LAYER_REGISTRY",
    "get_layer",
]
```

---

## Testes - tests/test_layers.py

```python
"""
Testes para módulos HLS-LAY (Human Layers).
"""

import pytest
from unittest.mock import AsyncMock


class TestUIUXLayer:
    """Testes para HL-1 UIUXLayer."""

    def test_layer_metadata(self):
        """Metadados da layer estão corretos."""
        from hl_mcp.layers.hl1_uiux import UIUXLayer

        assert UIUXLayer.LAYER_ID == "HL-1"
        assert UIUXLayer.VETO_POWER == "WEAK"

    @pytest.mark.asyncio
    async def test_validate_returns_result(self):
        """validate() retorna UIUXResult."""
        from hl_mcp.layers.hl1_uiux import UIUXLayer, UIUXResult

        mock_llm = AsyncMock()
        mock_llm.complete.return_value = '{"findings": [], "veto_level": "NONE"}'

        layer = UIUXLayer(mock_llm)
        result = await layer.validate("Test target")

        assert isinstance(result, UIUXResult)
        assert result.success is True


class TestSecurityLayer:
    """Testes para HL-2 SecurityLayer."""

    def test_layer_metadata(self):
        """Metadados da layer estão corretos."""
        from hl_mcp.layers.hl2_security import SecurityLayer

        assert SecurityLayer.LAYER_ID == "HL-2"
        assert SecurityLayer.VETO_POWER == "STRONG"

    def test_security_finding_is_critical(self):
        """is_critical funciona corretamente."""
        from hl_mcp.layers.hl2_security import (
            SecurityFinding,
            VulnerabilityType,
        )

        critical = SecurityFinding(
            vuln_type=VulnerabilityType.INJECTION,
            severity="critical",
            title="SQL Injection",
            description="...",
        )

        low = SecurityFinding(
            vuln_type=VulnerabilityType.MISCONFIG,
            severity="low",
            title="Minor config issue",
            description="...",
        )

        assert critical.is_critical is True
        assert low.is_critical is False


class TestEdgeCasesLayer:
    """Testes para HL-3 EdgeCasesLayer."""

    def test_layer_metadata(self):
        """Metadados da layer estão corretos."""
        from hl_mcp.layers.hl3_edge_cases import EdgeCasesLayer

        assert EdgeCasesLayer.LAYER_ID == "HL-3"
        assert EdgeCasesLayer.VETO_POWER == "MEDIUM"


class TestAccessibilityLayer:
    """Testes para HL-4 AccessibilityLayer."""

    def test_layer_metadata(self):
        """Metadados da layer estão corretos."""
        from hl_mcp.layers.hl4_accessibility import AccessibilityLayer

        assert AccessibilityLayer.LAYER_ID == "HL-4"
        assert AccessibilityLayer.VETO_POWER == "MEDIUM"


class TestLayerRegistry:
    """Testes para o registry de layers."""

    def test_get_layer(self):
        """get_layer retorna classe correta."""
        from hl_mcp.layers import get_layer, UIUXLayer, SecurityLayer

        assert get_layer("HL-1") == UIUXLayer
        assert get_layer("HL-2") == SecurityLayer

    def test_get_unknown_layer(self):
        """get_layer retorna None para layer desconhecida."""
        from hl_mcp.layers import get_layer

        assert get_layer("HL-99") is None
```

---

## Atualização do LEGO_INDEX.yaml

```yaml
# ============================================================
# HUMAN LAYERS (HLS-LAY-001 a HLS-LAY-007)
# ============================================================

HLS-LAY-001:
  name: UIUXLayer
  path: src/hl_mcp/layers/hl1_uiux.py
  category: layers
  description: HL-1 UI/UX Review - Usabilidade e clareza
  veto_power: WEAK
  exports:
    - UIUXCategory
    - UIUXConfig
    - UIUXFinding
    - UIUXResult
    - UIUXLayer
  search_hints:
    - uiux
    - usability
    - ui
    - ux
    - clarity
    - navigation

HLS-LAY-002:
  name: SecurityLayer
  path: src/hl_mcp/layers/hl2_security.py
  category: layers
  description: HL-2 Security Scan - Vulnerabilidades OWASP
  veto_power: STRONG
  exports:
    - VulnerabilityType
    - SecurityConfig
    - SecurityFinding
    - SecurityResult
    - SecurityLayer
  search_hints:
    - security
    - owasp
    - injection
    - xss
    - vulnerability
    - cwe

HLS-LAY-003:
  name: EdgeCasesLayer
  path: src/hl_mcp/layers/hl3_edge_cases.py
  category: layers
  description: HL-3 Edge Cases - Casos limite e erros
  veto_power: MEDIUM
  exports:
    - EdgeCaseCategory
    - EdgeCaseConfig
    - EdgeCaseFinding
    - EdgeCasesResult
    - EdgeCasesLayer
  search_hints:
    - edge case
    - boundary
    - null
    - empty
    - race condition

HLS-LAY-004:
  name: AccessibilityLayer
  path: src/hl_mcp/layers/hl4_accessibility.py
  category: layers
  description: HL-4 Accessibility - WCAG compliance
  veto_power: MEDIUM
  exports:
    - WCAGLevel
    - A11yCategory
    - A11yConfig
    - A11yFinding
    - A11yResult
    - AccessibilityLayer
  search_hints:
    - accessibility
    - a11y
    - wcag
    - aria
    - screen reader

HLS-LAY-005:
  name: PerformanceLayer
  path: src/hl_mcp/layers/hl5_performance.py
  category: layers
  description: HL-5 Performance - N+1, memory, escala
  veto_power: MEDIUM
  search_hints:
    - performance
    - n+1
    - memory
    - scalability

HLS-LAY-006:
  name: IntegrationLayer
  path: src/hl_mcp/layers/hl6_integration.py
  category: layers
  description: HL-6 Integration - APIs e contratos
  veto_power: STRONG
  search_hints:
    - integration
    - api
    - contract
    - boundary

HLS-LAY-007:
  name: FinalCheckLayer
  path: src/hl_mcp/layers/hl7_final_check.py
  category: layers
  description: HL-7 Final Human Check - Revisão final
  veto_power: STRONG
  search_hints:
    - final
    - human
    - review
    - last check
```

---

## Resumo do Block 09

| Módulo | ID | Layer | Veto Power | Linhas |
|--------|----|-------|------------|--------|
| UIUXLayer | HLS-LAY-001 | HL-1 | WEAK | ~280 |
| SecurityLayer | HLS-LAY-002 | HL-2 | STRONG | ~350 |
| EdgeCasesLayer | HLS-LAY-003 | HL-3 | MEDIUM | ~250 |
| AccessibilityLayer | HLS-LAY-004 | HL-4 | MEDIUM | ~220 |
| PerformanceLayer | HLS-LAY-005 | HL-5 | MEDIUM | ~220 |
| IntegrationLayer | HLS-LAY-006 | HL-6 | STRONG | ~220 |
| FinalCheckLayer | HLS-LAY-007 | HL-7 | STRONG | ~200 |
| **TOTAL** | | | | **~1,740** |

---

## Próximo: Block 10 - 6 Perspectives

O Block 10 vai implementar as 6 perspectivas de teste:
1. **HLS-PRS-001**: tired_user - Usuário cansado/distraído
2. **HLS-PRS-002**: malicious_insider - Insider malicioso
3. **HLS-PRS-003**: confused_newbie - Usuário novato confuso
4. **HLS-PRS-004**: power_user - Usuário avançado
5. **HLS-PRS-005**: auditor - Auditor externo
6. **HLS-PRS-006**: 3am_operator - Operador de madrugada

Quer que eu continue com o Block 10 (6 Perspectives)?
