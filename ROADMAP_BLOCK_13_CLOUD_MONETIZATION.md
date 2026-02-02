# ROADMAP Block 13: Cloud & Monetization

> **Bloco**: 13 (Adicional)
> **Tema**: Cloud Features + Modelo de Negócio
> **Tokens Estimados**: ~25,000
> **Dependências**: Blocks 01-12

---

## Filosofia Core

```
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│   "USA O LLM QUE VOCÊ JÁ PAGA. SE ACABAR, PROBLEMA SEU."       │
│                                                                │
│   • Claude Max ($20/mês) → Funciona                            │
│   • GPT Plus/Pro ($20/mês) → Funciona                          │
│   • Gemini Pro → Funciona                                      │
│   • Qualquer LLM → Funciona                                    │
│                                                                │
│   Token acabou?                                                │
│   → A gente AVISA                                              │
│   → Você DECIDE o que fazer                                    │
│                                                                │
│   Simples assim.                                               │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## Modelo de Negócio

### Open Source (Self-Hosted)

```
100% FUNCIONAL, GRÁTIS FOREVER

O que você tem:
✓ Todas as 7 Human Layers
✓ Todas as 6 Perspectivas
✓ Redundância tripla
✓ Core engine completo
✓ Browser automation
✓ Todo o código

O que você faz:
• Conecta seu LLM (Claude/GPT/Gemini)
• Configura Playwright local
• Configura stacks extras se quiser
• Roda onde quiser

Custo: $0
```

### Cloud (Managed)

```
CONVENIÊNCIA + EXTRAS

O que você ganha:
✓ Zero config - funciona em 30 segundos
✓ Playwright gerenciado (não instala nada)
✓ Dashboard visual
✓ Histórico de validações
✓ Team collaboration
✓ CI/CD integrations prontas
✓ Suporte

O que você ainda faz:
• Conecta SEU LLM (usa o plano que já paga)

Custo: $X/mês pela plataforma
LLM: Você usa o seu (não cobramos LLM)
```

---

## Modos de Uso

### Modo Interativo (Principal)

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   Desenvolvedor usando manualmente                          │
│                                                             │
│   Claude Desktop ←──→ Human Layer MCP                       │
│        │                                                    │
│        └── Usa tokens do Claude Max dele                    │
│                                                             │
│   OU                                                        │
│                                                             │
│   ChatGPT / Gemini ←──→ Human Layer                         │
│        │                                                    │
│        └── Usa tokens do plano dele                         │
│                                                             │
│   Custo LLM extra: $0                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Modo Automatizado (CI/CD)

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   CI/CD Pipeline (headless, sem usuário)                    │
│                                                             │
│   GitHub Action ──→ Human Layer ──→ LLM via API             │
│                                          │                  │
│                                          └── BYOK           │
│                                              (sua API key)  │
│                                                             │
│   Custo LLM: Pay-per-use (você paga direto pro provider)    │
│                                                             │
│   OU                                                        │
│                                                             │
│   GitHub Action ──→ Human Layer ──→ Pass-through            │
│                                          │                  │
│                                          └── Custo + 15%    │
│                                              (conveniência) │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Comparativo Final

| Aspecto | OSS | Cloud |
|---------|-----|-------|
| **Código** | 100% | 100% |
| **7 Layers** | ✅ | ✅ |
| **6 Perspectivas** | ✅ | ✅ |
| **Redundância** | ✅ | ✅ |
| **LLM** | Seu (Max/Plus/Pro) | Seu (Max/Plus/Pro) |
| **Playwright** | Instala você | Gerenciado |
| **Config** | Você faz | Zero config |
| **Dashboard** | ❌ | ✅ |
| **Histórico** | Local | Cloud (90+ dias) |
| **Team features** | ❌ | ✅ |
| **CI/CD integrations** | Manual | 1-click |
| **Suporte** | Community | Dedicado |
| **Preço** | **Grátis** | **$12-149/mês** |

---

## Pricing Cloud

| Tier | Preço | Para quem |
|------|-------|-----------|
| **Free** | $0/mês | Experimentar Cloud |
| **Solo** | $12/mês | Dev individual |
| **Pro** | $39/mês | Dev que quer CI/CD |
| **Team** | $99/mês | Equipe pequena |
| **Business** | $249/mês | Equipe grande |
| **Enterprise** | Custom | Empresa |

### O que cada tier inclui

#### Free
- 20 validações/mês
- Dashboard básico
- Histórico 7 dias
- 1 usuário
- Community support

#### Solo ($12/mês)
- 200 validações/mês
- Dashboard completo
- Histórico 30 dias
- 1 usuário
- Email support

#### Pro ($39/mês)
- 1000 validações/mês
- Tudo do Solo
- CI/CD integrations
- Webhooks
- Histórico 90 dias
- 3 usuários
- Priority support

#### Team ($99/mês)
- 3000 validações/mês
- Tudo do Pro
- 10 usuários
- Team dashboard
- Role management
- Histórico 180 dias

#### Business ($249/mês)
- 10000 validações/mês
- Tudo do Team
- 30 usuários
- SSO (Google/GitHub)
- API access
- Histórico 1 ano
- Dedicated support

#### Enterprise (Custom)
- Validações ilimitadas
- Usuários ilimitados
- SSO/SAML
- SLA 99.9%
- Self-hosted option
- Dedicated success manager
- Custom integrations

---

## Módulos Cloud-Only

### HLS-CLD-001: Multi-LLM Connector

```
ID: HLS-CLD-001
Nome: MultiLLMConnector
Caminho: src/hl_mcp/cloud/llm_connector.py
Exports: MultiLLMConnector, LLMConnection, ConnectionStatus
Linhas: ~250
```

```python
"""
HLS-CLD-001: Multi-LLM Connector
================================

Conecta e gerencia múltiplos LLMs.

Suporta:
- Claude (via Claude Desktop MCP ou API)
- GPT (via ChatGPT ou API)
- Gemini (via Gemini ou API)
- Qualquer LLM compatível com OpenAI API format

Exemplo:
    >>> connector = MultiLLMConnector()
    >>> connector.connect_claude_desktop()  # Usa Claude Max
    >>> connector.connect_openai(api_key)   # Ou usa API
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Protocol
import logging

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Provedores de LLM suportados."""

    CLAUDE = "claude"
    OPENAI = "openai"
    GEMINI = "gemini"
    CUSTOM = "custom"


class ConnectionMode(str, Enum):
    """Modo de conexão."""

    DESKTOP_APP = "desktop_app"  # Claude Desktop, ChatGPT app
    API_KEY = "api_key"          # API direta
    PASS_THROUGH = "pass_through"  # Via Human Layer


@dataclass
class LLMConnection:
    """Representa uma conexão com LLM."""

    provider: LLMProvider
    mode: ConnectionMode
    is_connected: bool = False
    last_used: Optional[datetime] = None
    tokens_used_session: int = 0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "provider": self.provider.value,
            "mode": self.mode.value,
            "is_connected": self.is_connected,
            "tokens_used_session": self.tokens_used_session,
        }


class MultiLLMConnector:
    """
    Gerenciador de conexões com múltiplos LLMs.

    Permite alternar entre LLMs e modos de conexão.

    Example:
        >>> connector = MultiLLMConnector()
        >>>
        >>> # Conectar via app de chat (usa plano do usuário)
        >>> await connector.connect_desktop(LLMProvider.CLAUDE)
        >>>
        >>> # Ou via API
        >>> await connector.connect_api(
        ...     LLMProvider.OPENAI,
        ...     api_key="sk-..."
        ... )
    """

    def __init__(self):
        self._connections: dict[LLMProvider, LLMConnection] = {}
        self._active_provider: Optional[LLMProvider] = None

    async def connect_desktop(
        self,
        provider: LLMProvider,
    ) -> LLMConnection:
        """
        Conecta via app desktop (usa plano do usuário).

        Args:
            provider: Claude, OpenAI, ou Gemini

        Returns:
            LLMConnection estabelecida
        """
        logger.info(f"Conectando via desktop: {provider.value}")

        connection = LLMConnection(
            provider=provider,
            mode=ConnectionMode.DESKTOP_APP,
            is_connected=True,
        )

        self._connections[provider] = connection
        self._active_provider = provider

        return connection

    async def connect_api(
        self,
        provider: LLMProvider,
        api_key: str,
    ) -> LLMConnection:
        """
        Conecta via API key (BYOK).

        Args:
            provider: Provedor
            api_key: API key do usuário

        Returns:
            LLMConnection estabelecida
        """
        logger.info(f"Conectando via API: {provider.value}")

        # Valida API key
        valid = await self._validate_api_key(provider, api_key)

        connection = LLMConnection(
            provider=provider,
            mode=ConnectionMode.API_KEY,
            is_connected=valid,
            error=None if valid else "Invalid API key",
        )

        if valid:
            self._connections[provider] = connection
            self._active_provider = provider

        return connection

    async def connect_pass_through(
        self,
        provider: LLMProvider,
    ) -> LLMConnection:
        """
        Conecta via pass-through (Human Layer paga, repassa + 15%).

        Args:
            provider: Provedor preferido

        Returns:
            LLMConnection estabelecida
        """
        logger.info(f"Conectando via pass-through: {provider.value}")

        connection = LLMConnection(
            provider=provider,
            mode=ConnectionMode.PASS_THROUGH,
            is_connected=True,
        )

        self._connections[provider] = connection
        self._active_provider = provider

        return connection

    async def _validate_api_key(
        self,
        provider: LLMProvider,
        api_key: str,
    ) -> bool:
        """Valida API key fazendo chamada de teste."""
        # Implementação simplificada
        return api_key and len(api_key) > 10

    def get_active_connection(self) -> Optional[LLMConnection]:
        """Retorna conexão ativa."""
        if self._active_provider:
            return self._connections.get(self._active_provider)
        return None

    def list_connections(self) -> list[LLMConnection]:
        """Lista todas as conexões."""
        return list(self._connections.values())

    async def switch_provider(self, provider: LLMProvider) -> bool:
        """Troca para outro provider conectado."""
        if provider in self._connections:
            self._active_provider = provider
            return True
        return False


__all__ = [
    "LLMProvider",
    "ConnectionMode",
    "LLMConnection",
    "MultiLLMConnector",
]
```

---

### HLS-CLD-002: Token Monitor

```
ID: HLS-CLD-002
Nome: TokenMonitor
Caminho: src/hl_mcp/cloud/token_monitor.py
Exports: TokenMonitor, TokenStatus, TokenAlert
Linhas: ~200
```

```python
"""
HLS-CLD-002: Token Monitor
==========================

Monitora uso de tokens e avisa quando está acabando.

Responsabilidades:
- Rastrear tokens usados por sessão
- Estimar tokens restantes (quando possível)
- Alertar usuário quando baixo
- Sugerir alternativas

"A gente AVISA, você DECIDE."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Optional
import logging

logger = logging.getLogger(__name__)


class AlertLevel(str, Enum):
    """Níveis de alerta."""

    OK = "ok"              # Tudo bem
    WARNING = "warning"    # Ficando baixo
    CRITICAL = "critical"  # Quase acabando
    EXHAUSTED = "exhausted"  # Acabou


@dataclass
class TokenStatus:
    """Status atual de tokens."""

    provider: str
    tokens_used_session: int
    tokens_used_today: int
    estimated_remaining: Optional[int]  # None se não souber
    alert_level: AlertLevel
    reset_time: Optional[datetime]  # Quando reseta o limite

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "tokens_used_session": self.tokens_used_session,
            "tokens_used_today": self.tokens_used_today,
            "estimated_remaining": self.estimated_remaining,
            "alert_level": self.alert_level.value,
            "reset_time": self.reset_time.isoformat() if self.reset_time else None,
        }


@dataclass
class TokenAlert:
    """Alerta de tokens."""

    level: AlertLevel
    message: str
    suggestions: list[str] = field(default_factory=list)

    @classmethod
    def exhausted(cls, provider: str) -> "TokenAlert":
        return cls(
            level=AlertLevel.EXHAUSTED,
            message=f"Limite de tokens do {provider} atingido.",
            suggestions=[
                "Aguardar reset do limite",
                "Trocar para outro LLM",
                "Usar API pay-per-use",
            ],
        )

    @classmethod
    def warning(cls, provider: str, percent_used: int) -> "TokenAlert":
        return cls(
            level=AlertLevel.WARNING,
            message=f"{provider}: ~{percent_used}% do limite usado.",
            suggestions=[
                "Considere reduzir validações",
                "Prepare fallback para outro LLM",
            ],
        )


class TokenMonitor:
    """
    Monitor de tokens.

    Rastreia uso e alerta quando necessário.

    Example:
        >>> monitor = TokenMonitor()
        >>> monitor.on_alert(lambda alert: print(alert.message))
        >>>
        >>> # Registrar uso
        >>> monitor.record_usage("claude", 5000)
        >>>
        >>> # Verificar status
        >>> status = monitor.get_status("claude")
    """

    # Limites estimados por provider (tokens/dia para planos de chat)
    ESTIMATED_LIMITS = {
        "claude": 100000,   # Claude Max ~100k/dia estimado
        "openai": 80000,    # GPT Plus ~80k/dia estimado
        "gemini": 60000,    # Gemini Pro ~60k/dia estimado
    }

    def __init__(self):
        self._usage: dict[str, dict] = {}
        self._alert_callbacks: list[Callable[[TokenAlert], None]] = []

    def record_usage(
        self,
        provider: str,
        tokens: int,
    ) -> TokenStatus:
        """
        Registra uso de tokens.

        Args:
            provider: Nome do provider
            tokens: Tokens usados

        Returns:
            TokenStatus atualizado
        """
        if provider not in self._usage:
            self._usage[provider] = {
                "session": 0,
                "today": 0,
                "last_reset": datetime.utcnow().date(),
            }

        # Reset diário
        today = datetime.utcnow().date()
        if self._usage[provider]["last_reset"] != today:
            self._usage[provider]["today"] = 0
            self._usage[provider]["last_reset"] = today

        self._usage[provider]["session"] += tokens
        self._usage[provider]["today"] += tokens

        status = self._calculate_status(provider)

        # Dispara alertas se necessário
        self._check_alerts(provider, status)

        return status

    def _calculate_status(self, provider: str) -> TokenStatus:
        """Calcula status atual."""
        usage = self._usage.get(provider, {"session": 0, "today": 0})
        limit = self.ESTIMATED_LIMITS.get(provider.lower())

        if limit:
            remaining = max(0, limit - usage["today"])
            percent_used = (usage["today"] / limit) * 100

            if percent_used >= 100:
                alert_level = AlertLevel.EXHAUSTED
            elif percent_used >= 80:
                alert_level = AlertLevel.CRITICAL
            elif percent_used >= 60:
                alert_level = AlertLevel.WARNING
            else:
                alert_level = AlertLevel.OK
        else:
            remaining = None
            alert_level = AlertLevel.OK

        return TokenStatus(
            provider=provider,
            tokens_used_session=usage["session"],
            tokens_used_today=usage["today"],
            estimated_remaining=remaining,
            alert_level=alert_level,
            reset_time=None,  # Depende do provider
        )

    def _check_alerts(self, provider: str, status: TokenStatus) -> None:
        """Verifica e dispara alertas."""
        if status.alert_level == AlertLevel.EXHAUSTED:
            alert = TokenAlert.exhausted(provider)
            self._fire_alert(alert)
        elif status.alert_level in [AlertLevel.WARNING, AlertLevel.CRITICAL]:
            percent = int((status.tokens_used_today / self.ESTIMATED_LIMITS.get(provider.lower(), 1)) * 100)
            alert = TokenAlert.warning(provider, percent)
            self._fire_alert(alert)

    def _fire_alert(self, alert: TokenAlert) -> None:
        """Dispara alert para callbacks."""
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Erro em alert callback: {e}")

    def on_alert(self, callback: Callable[[TokenAlert], None]) -> None:
        """Registra callback para alertas."""
        self._alert_callbacks.append(callback)

    def get_status(self, provider: str) -> TokenStatus:
        """Retorna status atual de um provider."""
        return self._calculate_status(provider)

    def get_all_status(self) -> dict[str, TokenStatus]:
        """Retorna status de todos os providers."""
        return {
            provider: self._calculate_status(provider)
            for provider in self._usage.keys()
        }


__all__ = [
    "AlertLevel",
    "TokenStatus",
    "TokenAlert",
    "TokenMonitor",
]
```

---

### HLS-CLD-003: Fallback Manager

```
ID: HLS-CLD-003
Nome: FallbackManager
Caminho: src/hl_mcp/cloud/fallback.py
Exports: FallbackManager, FallbackChain
Linhas: ~150
```

```python
"""
HLS-CLD-003: Fallback Manager
=============================

Gerencia fallback entre LLMs quando um atinge limite.

Quando Claude Max acaba:
1. Avisa o usuário
2. Oferece opções:
   - Trocar para GPT
   - Trocar para Gemini
   - Usar API pay-per-use
   - Aguardar reset
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import logging

from .llm_connector import LLMProvider, MultiLLMConnector

logger = logging.getLogger(__name__)


@dataclass
class FallbackOption:
    """Uma opção de fallback."""

    provider: LLMProvider
    mode: str  # "desktop", "api", "pass_through"
    available: bool
    reason: str  # Por que está/não está disponível

    def to_dict(self) -> dict:
        return {
            "provider": self.provider.value,
            "mode": self.mode,
            "available": self.available,
            "reason": self.reason,
        }


@dataclass
class FallbackChain:
    """Cadeia de fallback configurada."""

    primary: LLMProvider
    fallbacks: list[LLMProvider] = field(default_factory=list)
    auto_switch: bool = False  # Trocar automaticamente ou perguntar

    @classmethod
    def default(cls) -> "FallbackChain":
        return cls(
            primary=LLMProvider.CLAUDE,
            fallbacks=[LLMProvider.OPENAI, LLMProvider.GEMINI],
            auto_switch=False,
        )


class FallbackManager:
    """
    Gerenciador de fallback entre LLMs.

    Example:
        >>> manager = FallbackManager(connector)
        >>>
        >>> # Quando token acaba
        >>> options = manager.get_fallback_options()
        >>> for opt in options:
        ...     print(f"{opt.provider}: {opt.reason}")
        >>>
        >>> # Trocar para fallback
        >>> await manager.switch_to_fallback(LLMProvider.OPENAI)
    """

    def __init__(
        self,
        connector: MultiLLMConnector,
        chain: Optional[FallbackChain] = None,
    ):
        self.connector = connector
        self.chain = chain or FallbackChain.default()

    def get_fallback_options(self) -> list[FallbackOption]:
        """
        Retorna opções de fallback disponíveis.

        Returns:
            Lista de opções ordenadas por preferência
        """
        options = []

        for provider in self.chain.fallbacks:
            # Verifica se já está conectado
            connections = self.connector.list_connections()
            connected = any(
                c.provider == provider and c.is_connected
                for c in connections
            )

            if connected:
                options.append(FallbackOption(
                    provider=provider,
                    mode="desktop",
                    available=True,
                    reason="Já conectado, pronto para usar",
                ))
            else:
                options.append(FallbackOption(
                    provider=provider,
                    mode="desktop",
                    available=True,
                    reason="Disponível, precisa conectar",
                ))

        # Sempre oferece API como última opção
        options.append(FallbackOption(
            provider=self.chain.primary,
            mode="api",
            available=True,
            reason="Pay-per-use via API (sem limite)",
        ))

        # Opção de aguardar
        options.append(FallbackOption(
            provider=self.chain.primary,
            mode="wait",
            available=True,
            reason="Aguardar reset do limite",
        ))

        return options

    async def switch_to_fallback(
        self,
        provider: LLMProvider,
        mode: str = "desktop",
    ) -> bool:
        """
        Troca para um fallback.

        Args:
            provider: Provider de destino
            mode: Modo de conexão

        Returns:
            True se troca foi bem sucedida
        """
        logger.info(f"Trocando para fallback: {provider.value} ({mode})")

        if mode == "desktop":
            connection = await self.connector.connect_desktop(provider)
            return connection.is_connected
        elif mode == "api":
            # Precisa de API key - retorna False para pedir ao usuário
            return False
        elif mode == "wait":
            # Não faz nada, usuário escolheu aguardar
            return True

        return False


__all__ = [
    "FallbackOption",
    "FallbackChain",
    "FallbackManager",
]
```

---

### HLS-CLD-004: Usage Dashboard

```
ID: HLS-CLD-004
Nome: UsageDashboard
Caminho: src/hl_mcp/cloud/dashboard.py
Exports: DashboardData, DashboardService
Linhas: ~180
```

```python
"""
HLS-CLD-004: Usage Dashboard
============================

Dados para dashboard de uso.

Cloud-only feature - visualização de:
- Validações realizadas
- Findings por categoria
- Tendências de qualidade
- Uso por layer/perspectiva
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationSummary:
    """Resumo de validações."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    needs_review: int = 0

    @property
    def pass_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.passed / self.total


@dataclass
class FindingsSummary:
    """Resumo de findings."""

    total: int = 0
    by_severity: dict[str, int] = field(default_factory=dict)
    by_category: dict[str, int] = field(default_factory=dict)
    by_layer: dict[str, int] = field(default_factory=dict)


@dataclass
class DashboardData:
    """Dados do dashboard."""

    period_start: datetime
    period_end: datetime
    validations: ValidationSummary
    findings: FindingsSummary
    tokens_used: int = 0
    cost_estimate_usd: float = 0.0

    def to_dict(self) -> dict:
        return {
            "period": {
                "start": self.period_start.isoformat(),
                "end": self.period_end.isoformat(),
            },
            "validations": {
                "total": self.validations.total,
                "passed": self.validations.passed,
                "failed": self.validations.failed,
                "pass_rate": round(self.validations.pass_rate * 100, 1),
            },
            "findings": {
                "total": self.findings.total,
                "by_severity": self.findings.by_severity,
                "by_category": self.findings.by_category,
            },
            "usage": {
                "tokens": self.tokens_used,
                "cost_estimate": round(self.cost_estimate_usd, 2),
            },
        }


class DashboardService:
    """
    Serviço de dashboard.

    Agrega dados para visualização.

    Example:
        >>> service = DashboardService(storage)
        >>> data = await service.get_dashboard(
        ...     period="last_7_days"
        ... )
        >>> print(f"Pass rate: {data.validations.pass_rate:.0%}")
    """

    def __init__(self, storage):
        self.storage = storage

    async def get_dashboard(
        self,
        period: str = "last_7_days",
        user_id: Optional[str] = None,
        team_id: Optional[str] = None,
    ) -> DashboardData:
        """
        Gera dados do dashboard.

        Args:
            period: Período ("today", "last_7_days", "last_30_days")
            user_id: Filtrar por usuário
            team_id: Filtrar por team

        Returns:
            DashboardData agregado
        """
        # Calcula período
        now = datetime.utcnow()
        if period == "today":
            start = now.replace(hour=0, minute=0, second=0)
        elif period == "last_7_days":
            start = now - timedelta(days=7)
        elif period == "last_30_days":
            start = now - timedelta(days=30)
        else:
            start = now - timedelta(days=7)

        # Busca dados (mock - real implementation queries storage)
        validations = ValidationSummary(
            total=150,
            passed=120,
            failed=20,
            needs_review=10,
        )

        findings = FindingsSummary(
            total=89,
            by_severity={"critical": 5, "high": 15, "medium": 40, "low": 29},
            by_category={"security": 25, "uiux": 30, "edge_case": 20, "a11y": 14},
            by_layer={"HL-1": 30, "HL-2": 25, "HL-3": 20, "HL-4": 14},
        )

        return DashboardData(
            period_start=start,
            period_end=now,
            validations=validations,
            findings=findings,
            tokens_used=450000,
            cost_estimate_usd=0.0,  # $0 se usa plano de chat
        )


__all__ = [
    "ValidationSummary",
    "FindingsSummary",
    "DashboardData",
    "DashboardService",
]
```

---

### HLS-CLD-005: History Storage

```
ID: HLS-CLD-005
Nome: HistoryStorage
Caminho: src/hl_mcp/cloud/history.py
Exports: HistoryStorage, ValidationRecord
Linhas: ~150
```

```python
"""
HLS-CLD-005: History Storage
============================

Armazenamento de histórico de validações.

Cloud-only feature - persiste:
- Todas as validações
- Findings encontrados
- Métricas de qualidade
- Por período configurável (7-365 dias por tier)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationRecord:
    """Registro de uma validação."""

    id: str
    timestamp: datetime
    target: str
    layers_run: list[str]
    verdict: str
    findings_count: int
    critical_count: int
    duration_ms: float
    user_id: Optional[str] = None
    team_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "target": self.target,
            "layers_run": self.layers_run,
            "verdict": self.verdict,
            "findings_count": self.findings_count,
            "critical_count": self.critical_count,
            "duration_ms": self.duration_ms,
        }


class HistoryStorage:
    """
    Storage de histórico.

    Example:
        >>> storage = HistoryStorage(retention_days=90)
        >>>
        >>> # Salvar validação
        >>> await storage.save(record)
        >>>
        >>> # Buscar histórico
        >>> records = await storage.query(
        ...     user_id="user_123",
        ...     limit=50,
        ... )
    """

    def __init__(self, retention_days: int = 90):
        self.retention_days = retention_days
        self._records: list[ValidationRecord] = []  # In-memory para demo

    async def save(self, record: ValidationRecord) -> None:
        """Salva registro de validação."""
        self._records.append(record)
        logger.debug(f"Saved validation: {record.id}")

    async def query(
        self,
        user_id: Optional[str] = None,
        team_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        verdict: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ValidationRecord]:
        """
        Busca registros no histórico.

        Args:
            user_id: Filtrar por usuário
            team_id: Filtrar por team
            start_date: Data inicial
            end_date: Data final
            verdict: Filtrar por veredicto
            limit: Máximo de registros
            offset: Offset para paginação

        Returns:
            Lista de ValidationRecord
        """
        results = self._records.copy()

        # Aplica filtros
        if user_id:
            results = [r for r in results if r.user_id == user_id]
        if team_id:
            results = [r for r in results if r.team_id == team_id]
        if start_date:
            results = [r for r in results if r.timestamp >= start_date]
        if end_date:
            results = [r for r in results if r.timestamp <= end_date]
        if verdict:
            results = [r for r in results if r.verdict == verdict]

        # Ordena por timestamp desc
        results.sort(key=lambda r: r.timestamp, reverse=True)

        # Aplica paginação
        return results[offset:offset + limit]

    async def get_by_id(self, record_id: str) -> Optional[ValidationRecord]:
        """Busca registro por ID."""
        for record in self._records:
            if record.id == record_id:
                return record
        return None

    async def cleanup_old(self) -> int:
        """Remove registros mais antigos que retention_days."""
        from datetime import timedelta

        cutoff = datetime.utcnow() - timedelta(days=self.retention_days)
        before = len(self._records)
        self._records = [r for r in self._records if r.timestamp >= cutoff]
        removed = before - len(self._records)

        if removed > 0:
            logger.info(f"Removed {removed} old records")

        return removed


__all__ = [
    "ValidationRecord",
    "HistoryStorage",
]
```

---

### HLS-CLD-006: Team Collaboration

```
ID: HLS-CLD-006
Nome: TeamCollaboration
Caminho: src/hl_mcp/cloud/team.py
Exports: Team, TeamMember, TeamService
Linhas: ~180
```

```python
"""
HLS-CLD-006: Team Collaboration
===============================

Features de colaboração em equipe.

Cloud-only feature:
- Múltiplos usuários
- Dashboard compartilhado
- Role management
- Activity feed
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class TeamRole(str, Enum):
    """Roles de time."""

    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


@dataclass
class TeamMember:
    """Membro de um time."""

    user_id: str
    email: str
    name: str
    role: TeamRole
    joined_at: datetime = field(default_factory=datetime.utcnow)
    last_active: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "email": self.email,
            "name": self.name,
            "role": self.role.value,
            "joined_at": self.joined_at.isoformat(),
        }


@dataclass
class Team:
    """Time."""

    id: str
    name: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    members: list[TeamMember] = field(default_factory=list)
    settings: dict = field(default_factory=dict)

    @property
    def member_count(self) -> int:
        return len(self.members)

    def get_member(self, user_id: str) -> Optional[TeamMember]:
        for member in self.members:
            if member.user_id == user_id:
                return member
        return None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "member_count": self.member_count,
            "members": [m.to_dict() for m in self.members],
        }


class TeamService:
    """
    Serviço de gerenciamento de times.

    Example:
        >>> service = TeamService()
        >>>
        >>> # Criar time
        >>> team = await service.create_team("My Team", owner_id)
        >>>
        >>> # Adicionar membro
        >>> await service.add_member(team.id, "user@email.com", TeamRole.MEMBER)
    """

    def __init__(self):
        self._teams: dict[str, Team] = {}

    async def create_team(
        self,
        name: str,
        owner_id: str,
        owner_email: str,
        owner_name: str,
    ) -> Team:
        """Cria novo time."""
        import uuid

        team_id = str(uuid.uuid4())[:8]

        owner = TeamMember(
            user_id=owner_id,
            email=owner_email,
            name=owner_name,
            role=TeamRole.OWNER,
        )

        team = Team(
            id=team_id,
            name=name,
            members=[owner],
        )

        self._teams[team_id] = team
        logger.info(f"Created team: {team_id}")

        return team

    async def add_member(
        self,
        team_id: str,
        user_id: str,
        email: str,
        name: str,
        role: TeamRole = TeamRole.MEMBER,
    ) -> TeamMember:
        """Adiciona membro ao time."""
        team = self._teams.get(team_id)
        if not team:
            raise ValueError(f"Team not found: {team_id}")

        member = TeamMember(
            user_id=user_id,
            email=email,
            name=name,
            role=role,
        )

        team.members.append(member)
        logger.info(f"Added member {email} to team {team_id}")

        return member

    async def remove_member(
        self,
        team_id: str,
        user_id: str,
    ) -> bool:
        """Remove membro do time."""
        team = self._teams.get(team_id)
        if not team:
            return False

        team.members = [m for m in team.members if m.user_id != user_id]
        return True

    async def get_team(self, team_id: str) -> Optional[Team]:
        """Busca time por ID."""
        return self._teams.get(team_id)


__all__ = [
    "TeamRole",
    "TeamMember",
    "Team",
    "TeamService",
]
```

---

### HLS-CLD-007: CI/CD Integrations

```
ID: HLS-CLD-007
Nome: CICDIntegrations
Caminho: src/hl_mcp/cloud/cicd.py
Exports: GitHubIntegration, GitLabIntegration, WebhookConfig
Linhas: ~200
```

```python
"""
HLS-CLD-007: CI/CD Integrations
===============================

Integrações com CI/CD.

Cloud-only feature - integração 1-click com:
- GitHub Actions
- GitLab CI
- Webhooks genéricos

NOTA: Para CI/CD, precisa de API key (BYOK) porque
é headless (sem Claude Desktop).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class CICDProvider(str, Enum):
    """Providers de CI/CD."""

    GITHUB = "github"
    GITLAB = "gitlab"
    WEBHOOK = "webhook"


@dataclass
class WebhookConfig:
    """Configuração de webhook."""

    url: str
    secret: str
    events: list[str] = field(default_factory=lambda: ["validation.completed"])
    enabled: bool = True


@dataclass
class CICDConfig:
    """Configuração de integração CI/CD."""

    provider: CICDProvider
    enabled: bool = True
    llm_api_key_id: Optional[str] = None  # Referência à API key armazenada
    settings: dict = field(default_factory=dict)
    webhook: Optional[WebhookConfig] = None

    def to_dict(self) -> dict:
        return {
            "provider": self.provider.value,
            "enabled": self.enabled,
            "has_api_key": self.llm_api_key_id is not None,
        }


class GitHubIntegration:
    """
    Integração com GitHub.

    Permite:
    - Comentar em PRs com resultados
    - Bloquear merge se veto STRONG
    - Status checks

    Example:
        >>> gh = GitHubIntegration(token="ghp_...")
        >>> await gh.comment_on_pr(
        ...     repo="org/repo",
        ...     pr_number=123,
        ...     validation_result=result,
        ... )
    """

    def __init__(self, token: str):
        self.token = token

    async def comment_on_pr(
        self,
        repo: str,
        pr_number: int,
        validation_result: dict,
    ) -> bool:
        """Comenta resultado em PR."""
        # Implementação real usaria GitHub API
        logger.info(f"Commenting on {repo}#{pr_number}")
        return True

    async def set_status(
        self,
        repo: str,
        sha: str,
        state: str,  # "success", "failure", "pending"
        description: str,
    ) -> bool:
        """Define status check."""
        logger.info(f"Setting status on {repo}@{sha}: {state}")
        return True

    def generate_workflow_yaml(self) -> str:
        """Gera YAML para GitHub Actions."""
        return """
name: Human Layer Validation

on:
  pull_request:
    branches: [main]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run Human Layer
        uses: human-layer/action@v1
        with:
          api-key: ${{ secrets.HL_API_KEY }}
          llm-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
          layers: "HL-1,HL-2,HL-3"
"""


class GitLabIntegration:
    """Integração com GitLab."""

    def __init__(self, token: str):
        self.token = token

    async def comment_on_mr(
        self,
        project: str,
        mr_iid: int,
        validation_result: dict,
    ) -> bool:
        """Comenta resultado em MR."""
        logger.info(f"Commenting on {project}!{mr_iid}")
        return True

    def generate_gitlab_ci(self) -> str:
        """Gera .gitlab-ci.yml."""
        return """
human-layer:
  stage: test
  image: human-layer/cli:latest
  script:
    - hl-mcp validate --layers HL-1,HL-2,HL-3
  variables:
    HL_API_KEY: $HL_API_KEY
    ANTHROPIC_API_KEY: $ANTHROPIC_API_KEY
"""


class CICDService:
    """
    Serviço de integrações CI/CD.

    Example:
        >>> service = CICDService()
        >>>
        >>> # Configurar GitHub
        >>> config = await service.setup_github(
        ...     user_id="user_123",
        ...     github_token="ghp_...",
        ...     llm_api_key="sk-ant-...",
        ... )
    """

    def __init__(self):
        self._configs: dict[str, CICDConfig] = {}

    async def setup_github(
        self,
        user_id: str,
        github_token: str,
        llm_api_key: str,
    ) -> CICDConfig:
        """Configura integração GitHub."""
        config = CICDConfig(
            provider=CICDProvider.GITHUB,
            llm_api_key_id=f"key_{user_id}_github",
            settings={"token": github_token},
        )

        self._configs[f"{user_id}_github"] = config
        return config

    async def setup_webhook(
        self,
        user_id: str,
        webhook_url: str,
        secret: str,
    ) -> CICDConfig:
        """Configura webhook genérico."""
        config = CICDConfig(
            provider=CICDProvider.WEBHOOK,
            webhook=WebhookConfig(url=webhook_url, secret=secret),
        )

        self._configs[f"{user_id}_webhook"] = config
        return config


__all__ = [
    "CICDProvider",
    "WebhookConfig",
    "CICDConfig",
    "GitHubIntegration",
    "GitLabIntegration",
    "CICDService",
]
```

---

### HLS-CLD-008: Billing

```
ID: HLS-CLD-008
Nome: Billing
Caminho: src/hl_mcp/cloud/billing.py
Exports: BillingService, Subscription, Invoice
Linhas: ~200
```

```python
"""
HLS-CLD-008: Billing
====================

Sistema de billing para Cloud.

Gerencia:
- Assinaturas (tiers)
- Uso de validações
- Faturas
- Integração Stripe
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class SubscriptionTier(str, Enum):
    """Tiers de assinatura."""

    FREE = "free"
    SOLO = "solo"
    PRO = "pro"
    TEAM = "team"
    BUSINESS = "business"
    ENTERPRISE = "enterprise"


@dataclass
class TierLimits:
    """Limites por tier."""

    validations_per_month: int
    users: int
    history_days: int
    cicd: bool
    team_features: bool
    priority_support: bool


TIER_LIMITS = {
    SubscriptionTier.FREE: TierLimits(20, 1, 7, False, False, False),
    SubscriptionTier.SOLO: TierLimits(200, 1, 30, False, False, False),
    SubscriptionTier.PRO: TierLimits(1000, 3, 90, True, False, True),
    SubscriptionTier.TEAM: TierLimits(3000, 10, 180, True, True, True),
    SubscriptionTier.BUSINESS: TierLimits(10000, 30, 365, True, True, True),
    SubscriptionTier.ENTERPRISE: TierLimits(999999, 999, 365, True, True, True),
}


@dataclass
class Subscription:
    """Assinatura de um usuário/time."""

    id: str
    user_id: str
    team_id: Optional[str]
    tier: SubscriptionTier
    status: str  # "active", "canceled", "past_due"
    current_period_start: datetime
    current_period_end: datetime
    validations_used: int = 0

    @property
    def limits(self) -> TierLimits:
        return TIER_LIMITS[self.tier]

    @property
    def validations_remaining(self) -> int:
        return max(0, self.limits.validations_per_month - self.validations_used)

    @property
    def is_active(self) -> bool:
        return self.status == "active"

    def to_dict(self) -> dict:
        return {
            "tier": self.tier.value,
            "status": self.status,
            "validations_used": self.validations_used,
            "validations_remaining": self.validations_remaining,
            "limits": {
                "validations": self.limits.validations_per_month,
                "users": self.limits.users,
                "history_days": self.limits.history_days,
            },
        }


@dataclass
class Invoice:
    """Fatura."""

    id: str
    subscription_id: str
    amount_cents: int
    currency: str = "usd"
    status: str = "pending"  # "pending", "paid", "failed"
    created_at: datetime = field(default_factory=datetime.utcnow)
    paid_at: Optional[datetime] = None


class BillingService:
    """
    Serviço de billing.

    Example:
        >>> billing = BillingService()
        >>>
        >>> # Criar assinatura
        >>> sub = await billing.create_subscription(
        ...     user_id="user_123",
        ...     tier=SubscriptionTier.PRO,
        ... )
        >>>
        >>> # Verificar limites
        >>> if sub.validations_remaining > 0:
        ...     # OK para validar
        ...     await billing.increment_usage(sub.id)
    """

    def __init__(self, stripe_key: Optional[str] = None):
        self.stripe_key = stripe_key
        self._subscriptions: dict[str, Subscription] = {}

    async def create_subscription(
        self,
        user_id: str,
        tier: SubscriptionTier,
        team_id: Optional[str] = None,
    ) -> Subscription:
        """Cria nova assinatura."""
        import uuid
        from datetime import timedelta

        now = datetime.utcnow()

        sub = Subscription(
            id=str(uuid.uuid4())[:8],
            user_id=user_id,
            team_id=team_id,
            tier=tier,
            status="active",
            current_period_start=now,
            current_period_end=now + timedelta(days=30),
        )

        self._subscriptions[sub.id] = sub
        logger.info(f"Created subscription: {sub.id} ({tier.value})")

        return sub

    async def get_subscription(
        self,
        user_id: str,
    ) -> Optional[Subscription]:
        """Busca assinatura do usuário."""
        for sub in self._subscriptions.values():
            if sub.user_id == user_id and sub.is_active:
                return sub
        return None

    async def increment_usage(
        self,
        subscription_id: str,
        count: int = 1,
    ) -> bool:
        """Incrementa uso de validações."""
        sub = self._subscriptions.get(subscription_id)
        if not sub:
            return False

        sub.validations_used += count
        return True

    async def check_limit(
        self,
        subscription_id: str,
    ) -> tuple[bool, int]:
        """
        Verifica se está dentro do limite.

        Returns:
            (can_validate, remaining)
        """
        sub = self._subscriptions.get(subscription_id)
        if not sub:
            return False, 0

        remaining = sub.validations_remaining
        return remaining > 0, remaining

    async def upgrade_tier(
        self,
        subscription_id: str,
        new_tier: SubscriptionTier,
    ) -> bool:
        """Faz upgrade de tier."""
        sub = self._subscriptions.get(subscription_id)
        if not sub:
            return False

        sub.tier = new_tier
        logger.info(f"Upgraded {subscription_id} to {new_tier.value}")
        return True


__all__ = [
    "SubscriptionTier",
    "TierLimits",
    "TIER_LIMITS",
    "Subscription",
    "Invoice",
    "BillingService",
]
```

---

## Cloud __init__.py

```python
"""
HLS-CLD: Cloud Features
=======================

Módulos cloud-only do Human Layer.

Exports:
    - MultiLLMConnector (conecta qualquer LLM)
    - TokenMonitor (avisa quando token acabando)
    - FallbackManager (troca de LLM)
    - DashboardService (dashboard de uso)
    - HistoryStorage (histórico de validações)
    - TeamService (colaboração em equipe)
    - CICDService (integrações CI/CD)
    - BillingService (assinaturas)
"""

from .llm_connector import (
    LLMProvider,
    ConnectionMode,
    LLMConnection,
    MultiLLMConnector,
)

from .token_monitor import (
    AlertLevel,
    TokenStatus,
    TokenAlert,
    TokenMonitor,
)

from .fallback import (
    FallbackOption,
    FallbackChain,
    FallbackManager,
)

from .dashboard import (
    ValidationSummary,
    FindingsSummary,
    DashboardData,
    DashboardService,
)

from .history import (
    ValidationRecord,
    HistoryStorage,
)

from .team import (
    TeamRole,
    TeamMember,
    Team,
    TeamService,
)

from .cicd import (
    CICDProvider,
    WebhookConfig,
    CICDConfig,
    GitHubIntegration,
    GitLabIntegration,
    CICDService,
)

from .billing import (
    SubscriptionTier,
    TierLimits,
    TIER_LIMITS,
    Subscription,
    Invoice,
    BillingService,
)


__all__ = [
    # LLM Connector
    "LLMProvider",
    "ConnectionMode",
    "LLMConnection",
    "MultiLLMConnector",
    # Token Monitor
    "AlertLevel",
    "TokenStatus",
    "TokenAlert",
    "TokenMonitor",
    # Fallback
    "FallbackOption",
    "FallbackChain",
    "FallbackManager",
    # Dashboard
    "ValidationSummary",
    "FindingsSummary",
    "DashboardData",
    "DashboardService",
    # History
    "ValidationRecord",
    "HistoryStorage",
    # Team
    "TeamRole",
    "TeamMember",
    "Team",
    "TeamService",
    # CI/CD
    "CICDProvider",
    "WebhookConfig",
    "CICDConfig",
    "GitHubIntegration",
    "GitLabIntegration",
    "CICDService",
    # Billing
    "SubscriptionTier",
    "TierLimits",
    "TIER_LIMITS",
    "Subscription",
    "Invoice",
    "BillingService",
]
```

---

## Resumo Block 13

| Módulo | ID | Função | Linhas |
|--------|----|--------|--------|
| MultiLLMConnector | HLS-CLD-001 | Conecta qualquer LLM | ~250 |
| TokenMonitor | HLS-CLD-002 | Avisa token acabando | ~200 |
| FallbackManager | HLS-CLD-003 | Troca entre LLMs | ~150 |
| DashboardService | HLS-CLD-004 | Dashboard de uso | ~180 |
| HistoryStorage | HLS-CLD-005 | Histórico de validações | ~150 |
| TeamService | HLS-CLD-006 | Colaboração em equipe | ~180 |
| CICDService | HLS-CLD-007 | Integrações CI/CD | ~200 |
| BillingService | HLS-CLD-008 | Assinaturas e billing | ~200 |
| **TOTAL** | | | **~1,510** |

---

## Arquitetura Final Atualizada

```
src/hl_mcp/
├── models/          # HLS-MDL (7 módulos)
├── llm/             # HLS-LLM (5 módulos)
├── browser/         # HLS-BRW (6 módulos)
├── engine/          # HLS-ENG (5 módulos)
├── layers/          # HLS-LAY (7 módulos)
├── perspectives/    # HLS-PRS (7 módulos)
├── cognitive/       # HLS-COG (5 módulos)
├── server/          # HLS-MCP (4 módulos)
└── cloud/           # HLS-CLD (8 módulos) ← NOVO
```

**Total: 54 módulos OSS + 8 módulos Cloud = 62 módulos**

---

## Fluxo Completo

```
┌─────────────────────────────────────────────────────────────────┐
│                         USUÁRIO                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Tem Claude Max / GPT Plus / Gemini Pro?                        │
│                                                                 │
│  SIM ───────────────────────────────┐                           │
│                                     ▼                           │
│                          Usa plano dele                         │
│                          Custo LLM: $0                          │
│                                                                 │
│  NÃO ───────────────────────────────┐                           │
│                                     ▼                           │
│                          BYOK (API key própria)                 │
│                          ou                                     │
│                          Pass-through (+15%)                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Quer só usar? (OSS)              Quer extras? (Cloud)          │
│                                                                 │
│  • Instala local                  • Zero config                 │
│  • Configura tudo                 • Dashboard                   │
│  • Custo: $0                      • Histórico                   │
│                                   • Team                        │
│                                   • CI/CD                       │
│                                   • Custo: $12-249/mês          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Token do plano acabou?                                         │
│                                                                 │
│  → A GENTE AVISA                                                │
│  → Você decide:                                                 │
│    • Aguardar reset                                             │
│    • Trocar LLM                                                 │
│    • Usar API pay-per-use                                       │
│                                                                 │
│  PROBLEMA SEU. Simples assim.                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

**Block 13 Completo!**

O modelo de negócio está definido:
- **OSS**: Grátis, funciona com qualquer LLM
- **Cloud**: Paga pela plataforma, não pelo LLM
- **LLM**: Usuário usa o plano que já paga
- **Token acabou**: Problema do usuário, a gente avisa
