# ROADMAP Block 12: MCP Server Final

> **Bloco**: 12 de 12 (FINAL)
> **Tema**: MCP Server Implementation
> **Tokens Estimados**: ~18,000
> **Dependências**: Todos os blocos anteriores

---

## Visão Geral do Bloco

Este bloco finaliza o Human Layer MCP Server, integrando todos os componentes:

| Módulo | Função |
|--------|--------|
| MCPServer | Servidor principal MCP |
| ToolHandlers | Handlers de tools |
| ResourceHandlers | Handlers de resources |
| Configuration | Setup e config |

---

## Módulo HLS-MCP-001: MCPServer

```
ID: HLS-MCP-001
Nome: MCPServer
Caminho: src/hl_mcp/server/main.py
Dependências: mcp, todas as layers
Exports: HumanLayerMCPServer, create_server
Linhas: ~300
```

### Código

```python
"""
HLS-MCP-001: Human Layer MCP Server
===================================

Servidor MCP principal do Human Layer.

Responsabilidades:
- Inicializar servidor MCP
- Registrar tools e resources
- Gerenciar ciclo de vida
- Coordenar componentes

Exemplo de uso:
    $ python -m hl_mcp.server

Ou programaticamente:
    >>> server = create_server()
    >>> await server.run()

Changelog:
    - v1.0.0 (2026-02-01): Implementação inicial
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, Resource
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    Server = None

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """
    Configuração do servidor MCP.

    Attributes:
        name: Nome do servidor
        version: Versão do servidor
        max_concurrent_runs: Máximo de runs simultâneos
        default_timeout_ms: Timeout padrão
        enable_browser: Habilitar automação de browser
        enable_redundancy: Habilitar redundância tripla
        log_level: Nível de log
    """

    name: str = "human-layer"
    version: str = "1.0.0"
    max_concurrent_runs: int = 3
    default_timeout_ms: int = 120000
    enable_browser: bool = True
    enable_redundancy: bool = True
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Cria config a partir de variáveis de ambiente."""
        import os

        return cls(
            name=os.getenv("HL_MCP_NAME", "human-layer"),
            version=os.getenv("HL_MCP_VERSION", "1.0.0"),
            max_concurrent_runs=int(os.getenv("HL_MCP_MAX_CONCURRENT", "3")),
            enable_browser=os.getenv("HL_MCP_BROWSER", "true").lower() == "true",
            enable_redundancy=os.getenv("HL_MCP_REDUNDANCY", "true").lower() == "true",
            log_level=os.getenv("HL_MCP_LOG_LEVEL", "INFO"),
        )


class HumanLayerMCPServer:
    """
    Servidor MCP do Human Layer.

    Expõe as funcionalidades do Human Layer como tools e resources
    para uso por assistentes de IA.

    Example:
        >>> config = ServerConfig(enable_redundancy=True)
        >>> server = HumanLayerMCPServer(config)
        >>> await server.run()
    """

    def __init__(self, config: Optional[ServerConfig] = None):
        """
        Inicializa o servidor.

        Args:
            config: Configuração do servidor
        """
        if not MCP_AVAILABLE:
            raise RuntimeError(
                "MCP SDK não instalado. Execute: pip install mcp"
            )

        self.config = config or ServerConfig()
        self._server: Optional[Server] = None
        self._running = False

        # Componentes (lazy loaded)
        self._llm_client = None
        self._layer_orchestrator = None
        self._perspective_runner = None
        self._budget_manager = None

        # Setup logging
        logging.basicConfig(level=getattr(logging, self.config.log_level))

    async def initialize(self) -> None:
        """Inicializa componentes do servidor."""
        logger.info(f"Inicializando {self.config.name} v{self.config.version}")

        # Cria servidor MCP
        self._server = Server(self.config.name)

        # Registra handlers
        await self._register_tools()
        await self._register_resources()

        # Inicializa componentes
        await self._init_components()

        logger.info("Servidor inicializado")

    async def _init_components(self) -> None:
        """Inicializa componentes internos."""
        from hl_mcp.llm import get_llm_client
        from hl_mcp.engine import LayerOrchestrator, HumanLayerRunner
        from hl_mcp.cognitive import BudgetManager, BudgetConfig

        # LLM Client
        self._llm_client = get_llm_client("claude")

        # Budget Manager
        self._budget_manager = BudgetManager(BudgetConfig())

        # Layer Orchestrator
        from hl_mcp.llm.parser import ResponseParser
        parser = ResponseParser()
        runner = HumanLayerRunner(self._llm_client, parser)
        self._layer_orchestrator = LayerOrchestrator(runner)

        logger.debug("Componentes inicializados")

    async def _register_tools(self) -> None:
        """Registra tools MCP."""
        from .tools import register_all_tools
        await register_all_tools(self._server, self)

    async def _register_resources(self) -> None:
        """Registra resources MCP."""
        from .resources import register_all_resources
        await register_all_resources(self._server, self)

    async def run(self) -> None:
        """Executa o servidor."""
        if not self._server:
            await self.initialize()

        self._running = True
        logger.info(f"Servidor rodando: {self.config.name}")

        try:
            async with stdio_server() as (read_stream, write_stream):
                await self._server.run(
                    read_stream,
                    write_stream,
                    self._server.create_initialization_options(),
                )
        except asyncio.CancelledError:
            logger.info("Servidor cancelado")
        finally:
            self._running = False
            await self.shutdown()

    async def shutdown(self) -> None:
        """Desliga o servidor gracefully."""
        logger.info("Desligando servidor...")
        self._running = False

    @property
    def is_running(self) -> bool:
        """Retorna True se servidor está rodando."""
        return self._running

    @property
    def llm_client(self):
        """Retorna LLM client."""
        return self._llm_client

    @property
    def orchestrator(self):
        """Retorna Layer Orchestrator."""
        return self._layer_orchestrator

    @property
    def budget(self):
        """Retorna Budget Manager."""
        return self._budget_manager


def create_server(config: Optional[ServerConfig] = None) -> HumanLayerMCPServer:
    """
    Factory function para criar servidor.

    Args:
        config: Configuração opcional

    Returns:
        HumanLayerMCPServer configurado
    """
    return HumanLayerMCPServer(config or ServerConfig.from_env())


async def main():
    """Entry point do servidor."""
    server = create_server()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())


__all__ = [
    "ServerConfig",
    "HumanLayerMCPServer",
    "create_server",
    "main",
]
```

---

## Módulo HLS-MCP-002: ToolHandlers

```
ID: HLS-MCP-002
Nome: ToolHandlers
Caminho: src/hl_mcp/server/tools.py
Dependências: HLS-MCP-001
Exports: register_all_tools
Linhas: ~350
```

### Código

```python
"""
HLS-MCP-002: Tool Handlers
==========================

Handlers de tools MCP do Human Layer.

Tools disponíveis:
- run_full_validation: Executa todas as 7 layers
- run_quick_check: Verificação rápida (3 layers)
- run_security_scan: Foco em segurança (HL-2)
- run_perspective: Executa perspectiva específica
- validate_accessibility: Verifica acessibilidade
- generate_tests: Gera testes para edge cases

Changelog:
    - v1.0.0 (2026-02-01): Implementação inicial
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server import Server
    from .main import HumanLayerMCPServer

logger = logging.getLogger(__name__)


async def register_all_tools(server: "Server", hl_server: "HumanLayerMCPServer") -> None:
    """Registra todos os tools no servidor MCP."""

    # =========================================
    # Tool: run_full_validation
    # =========================================
    @server.tool()
    async def run_full_validation(
        target: str,
        code: str | None = None,
        use_redundancy: bool = True,
    ) -> dict:
        """
        Executa validação completa com todas as 7 Human Layers.

        Args:
            target: Descrição do que está sendo validado
            code: Código fonte (opcional)
            use_redundancy: Usar redundância tripla (padrão: True)

        Returns:
            Resultado completo com findings e vetos
        """
        logger.info(f"run_full_validation: {target}")

        from hl_mcp.engine import OrchestratorConfig

        config = OrchestratorConfig(
            use_redundancy=use_redundancy,
        )

        result = await hl_server.orchestrator.run(
            target_description=target,
            code_snippet=code,
        )

        return result.to_dict()

    # =========================================
    # Tool: run_quick_check
    # =========================================
    @server.tool()
    async def run_quick_check(
        target: str,
        code: str | None = None,
    ) -> dict:
        """
        Executa verificação rápida com 3 layers (HL-1, HL-2, HL-3).

        Mais rápido que validação completa, bom para iteração.

        Args:
            target: Descrição do que está sendo validado
            code: Código fonte (opcional)

        Returns:
            Resultado com findings principais
        """
        logger.info(f"run_quick_check: {target}")

        result = await hl_server.orchestrator.run_quick_check(
            target_description=target,
            code_snippet=code,
        )

        return result.to_dict()

    # =========================================
    # Tool: run_security_scan
    # =========================================
    @server.tool()
    async def run_security_scan(
        code: str,
        language: str = "python",
        context: dict | None = None,
    ) -> dict:
        """
        Executa scan de segurança focado (HL-2).

        Detecta vulnerabilidades OWASP Top 10.

        Args:
            code: Código a analisar
            language: Linguagem do código
            context: Contexto adicional

        Returns:
            Findings de segurança com CWE/OWASP
        """
        logger.info(f"run_security_scan: {language}")

        from hl_mcp.layers import SecurityLayer

        layer = SecurityLayer(hl_server.llm_client)
        result = await layer.validate(
            code=code,
            language=language,
            context=context,
        )

        return result.to_dict()

    # =========================================
    # Tool: run_perspective
    # =========================================
    @server.tool()
    async def run_perspective(
        perspective_id: str,
        target: str,
        ui_elements: list[str] | None = None,
    ) -> dict:
        """
        Executa análise de uma perspectiva específica.

        Perspectivas disponíveis:
        - tired_user
        - malicious_insider
        - confused_newbie
        - power_user
        - auditor
        - 3am_operator

        Args:
            perspective_id: ID da perspectiva
            target: Descrição do alvo
            ui_elements: Elementos de UI visíveis

        Returns:
            Findings e pain points da perspectiva
        """
        logger.info(f"run_perspective: {perspective_id}")

        from hl_mcp.perspectives import (
            get_perspective,
            PerspectiveID,
            PerspectiveContext,
        )

        pid = PerspectiveID(perspective_id)
        perspective = get_perspective(pid, hl_server.llm_client)

        if not perspective:
            return {"error": f"Perspectiva não encontrada: {perspective_id}"}

        context = PerspectiveContext(
            perspective_id=pid,
            target=target,
            ui_elements=ui_elements or [],
        )

        result = await perspective.analyze(context)
        return result.to_dict()

    # =========================================
    # Tool: run_all_perspectives
    # =========================================
    @server.tool()
    async def run_all_perspectives(
        target: str,
        parallel: bool = True,
    ) -> dict:
        """
        Executa análise com todas as 6 perspectivas.

        Args:
            target: Descrição do alvo
            parallel: Executar em paralelo

        Returns:
            Findings consolidados de todas perspectivas
        """
        logger.info(f"run_all_perspectives: {target}")

        from hl_mcp.perspectives import (
            PerspectiveRunner,
            PerspectiveContext,
            PerspectiveID,
        )

        runner = PerspectiveRunner(hl_server.llm_client)
        context = PerspectiveContext(
            perspective_id=PerspectiveID.TIRED_USER,
            target=target,
        )

        result = await runner.run_all(context, parallel=parallel)
        return result.to_dict()

    # =========================================
    # Tool: validate_accessibility
    # =========================================
    @server.tool()
    async def validate_accessibility(
        html: str,
        wcag_level: str = "AA",
    ) -> dict:
        """
        Valida acessibilidade de HTML (WCAG).

        Args:
            html: HTML a validar
            wcag_level: Nível WCAG (A, AA, AAA)

        Returns:
            Violações de acessibilidade
        """
        logger.info(f"validate_accessibility: WCAG {wcag_level}")

        from hl_mcp.layers import AccessibilityLayer, A11yConfig, WCAGLevel

        config = A11yConfig(
            wcag_level=WCAGLevel(wcag_level),
            use_axe_core=False,  # Sem browser
        )

        layer = AccessibilityLayer(hl_server.llm_client, config)
        result = await layer.validate(html=html)

        return result.to_dict()

    # =========================================
    # Tool: analyze_edge_cases
    # =========================================
    @server.tool()
    async def analyze_edge_cases(
        code: str,
        function_signature: str | None = None,
    ) -> dict:
        """
        Analisa código para edge cases (HL-3).

        Args:
            code: Código a analisar
            function_signature: Assinatura da função

        Returns:
            Edge cases e testes sugeridos
        """
        logger.info("analyze_edge_cases")

        from hl_mcp.layers import EdgeCasesLayer

        layer = EdgeCasesLayer(hl_server.llm_client)
        result = await layer.validate(
            code=code,
            function_signature=function_signature,
        )

        return result.to_dict()

    # =========================================
    # Tool: prioritize_findings
    # =========================================
    @server.tool()
    async def prioritize_findings(
        findings: list[dict],
    ) -> dict:
        """
        Prioriza lista de findings por severidade/impacto.

        Args:
            findings: Lista de findings a priorizar

        Returns:
            Findings ordenados com prioridades
        """
        logger.info(f"prioritize_findings: {len(findings)} findings")

        from hl_mcp.cognitive import TriageEngine

        engine = TriageEngine()
        result = engine.prioritize(findings)

        return {
            "total": len(result.triaged_findings),
            "by_priority": result.to_dict()["by_priority"],
            "quick_wins": len(result.quick_wins),
            "critical_first": [
                {
                    "id": f.id,
                    "title": f.title,
                    "priority": f.priority.value,
                    "effort": f.effort_estimate,
                }
                for f in result.critical_first[:10]
            ],
        }

    # =========================================
    # Tool: get_budget_status
    # =========================================
    @server.tool()
    async def get_budget_status() -> dict:
        """
        Retorna status atual do budget de tokens.

        Returns:
            Uso de tokens e custo
        """
        return hl_server.budget.get_status().to_dict()

    logger.info("Tools registrados: 8 tools")


__all__ = ["register_all_tools"]
```

---

## Módulo HLS-MCP-003: ResourceHandlers

```
ID: HLS-MCP-003
Nome: ResourceHandlers
Caminho: src/hl_mcp/server/resources.py
Dependências: HLS-MCP-001
Exports: register_all_resources
Linhas: ~200
```

### Código

```python
"""
HLS-MCP-003: Resource Handlers
==============================

Handlers de resources MCP do Human Layer.

Resources disponíveis:
- human_layer://layers - Lista de layers disponíveis
- human_layer://perspectives - Lista de perspectivas
- human_layer://config - Configuração atual
- human_layer://report/{id} - Relatório de execução

Changelog:
    - v1.0.0 (2026-02-01): Implementação inicial
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server import Server
    from .main import HumanLayerMCPServer

logger = logging.getLogger(__name__)


async def register_all_resources(
    server: "Server",
    hl_server: "HumanLayerMCPServer",
) -> None:
    """Registra todos os resources no servidor MCP."""

    # =========================================
    # Resource: human_layer://layers
    # =========================================
    @server.resource("human_layer://layers")
    async def get_layers() -> dict:
        """
        Lista todas as Human Layers disponíveis.

        Returns:
            Informações sobre cada layer
        """
        return {
            "layers": [
                {
                    "id": "HL-1",
                    "name": "UI/UX Review",
                    "description": "Analisa usabilidade e clareza de interface",
                    "veto_power": "WEAK",
                },
                {
                    "id": "HL-2",
                    "name": "Security Scan",
                    "description": "Detecta vulnerabilidades OWASP Top 10",
                    "veto_power": "STRONG",
                },
                {
                    "id": "HL-3",
                    "name": "Edge Cases",
                    "description": "Identifica casos limite e condições de erro",
                    "veto_power": "MEDIUM",
                },
                {
                    "id": "HL-4",
                    "name": "Accessibility",
                    "description": "Valida conformidade WCAG",
                    "veto_power": "MEDIUM",
                },
                {
                    "id": "HL-5",
                    "name": "Performance",
                    "description": "Analisa performance e escalabilidade",
                    "veto_power": "MEDIUM",
                },
                {
                    "id": "HL-6",
                    "name": "Integration",
                    "description": "Valida contratos de API e integrações",
                    "veto_power": "STRONG",
                },
                {
                    "id": "HL-7",
                    "name": "Final Check",
                    "description": "Revisão humana final",
                    "veto_power": "STRONG",
                },
            ],
        }

    # =========================================
    # Resource: human_layer://perspectives
    # =========================================
    @server.resource("human_layer://perspectives")
    async def get_perspectives() -> dict:
        """
        Lista todas as perspectivas de teste disponíveis.

        Returns:
            Informações sobre cada perspectiva
        """
        return {
            "perspectives": [
                {
                    "id": "tired_user",
                    "name": "Tired User",
                    "description": "Usuário cansado, distraído, com pressa",
                    "focus": ["erros de distração", "fluxos longos", "falta de feedback"],
                },
                {
                    "id": "malicious_insider",
                    "name": "Malicious Insider",
                    "description": "Funcionário tentando explorar o sistema",
                    "focus": ["IDOR", "escalação de privilégio", "exfiltração"],
                },
                {
                    "id": "confused_newbie",
                    "name": "Confused Newbie",
                    "description": "Usuário usando pela primeira vez",
                    "focus": ["onboarding", "documentação", "fluxos intuitivos"],
                },
                {
                    "id": "power_user",
                    "name": "Power User",
                    "description": "Usuário avançado explorando limites",
                    "focus": ["edge cases", "bulk operations", "atalhos"],
                },
                {
                    "id": "auditor",
                    "name": "Auditor",
                    "description": "Auditor externo verificando compliance",
                    "focus": ["audit trails", "logs", "documentação"],
                },
                {
                    "id": "3am_operator",
                    "name": "3AM Operator",
                    "description": "Operador de madrugada sob pressão",
                    "focus": ["mensagens claras", "recovery", "emergency access"],
                },
            ],
        }

    # =========================================
    # Resource: human_layer://config
    # =========================================
    @server.resource("human_layer://config")
    async def get_config() -> dict:
        """
        Retorna configuração atual do servidor.

        Returns:
            Configurações do Human Layer
        """
        return {
            "server": {
                "name": hl_server.config.name,
                "version": hl_server.config.version,
            },
            "features": {
                "browser_automation": hl_server.config.enable_browser,
                "triple_redundancy": hl_server.config.enable_redundancy,
                "max_concurrent_runs": hl_server.config.max_concurrent_runs,
            },
            "budget": hl_server.budget.get_status().to_dict(),
        }

    # =========================================
    # Resource: human_layer://veto-levels
    # =========================================
    @server.resource("human_layer://veto-levels")
    async def get_veto_levels() -> dict:
        """
        Explica os níveis de veto.

        Returns:
            Descrição de cada nível de veto
        """
        return {
            "veto_levels": {
                "NONE": {
                    "description": "Nenhum problema encontrado",
                    "action": "Pode prosseguir",
                },
                "WEAK": {
                    "description": "Sugestões de melhoria",
                    "action": "Considerar mudanças, não bloqueia",
                },
                "MEDIUM": {
                    "description": "Issues que precisam atenção",
                    "action": "Revisar antes de deploy, não bloqueia",
                },
                "STRONG": {
                    "description": "Issues críticos encontrados",
                    "action": "BLOQUEIA deploy, resolver obrigatório",
                },
            },
        }

    # =========================================
    # Resource: human_layer://context-pack-templates
    # =========================================
    @server.resource("human_layer://context-pack-templates")
    async def get_context_pack_templates() -> dict:
        """
        Templates de context pack para diferentes fluxos.

        Returns:
            Templates disponíveis
        """
        return {
            "templates": [
                {
                    "id": "login_flow",
                    "name": "Login Flow",
                    "description": "Validação de fluxo de autenticação",
                    "layers": ["HL-1", "HL-2", "HL-4"],
                    "perspectives": ["tired_user", "malicious_insider"],
                },
                {
                    "id": "checkout_flow",
                    "name": "Checkout Flow",
                    "description": "Validação de fluxo de compra",
                    "layers": ["HL-1", "HL-2", "HL-3", "HL-4"],
                    "perspectives": ["tired_user", "confused_newbie"],
                },
                {
                    "id": "api_endpoint",
                    "name": "API Endpoint",
                    "description": "Validação de endpoint de API",
                    "layers": ["HL-2", "HL-3", "HL-6"],
                    "perspectives": ["malicious_insider", "power_user"],
                },
                {
                    "id": "admin_panel",
                    "name": "Admin Panel",
                    "description": "Validação de painel administrativo",
                    "layers": ["HL-2", "HL-4", "HL-6"],
                    "perspectives": ["malicious_insider", "auditor", "3am_operator"],
                },
            ],
        }

    logger.info("Resources registrados: 5 resources")


__all__ = ["register_all_resources"]
```

---

## Módulo HLS-MCP-004: Configuration

```
ID: HLS-MCP-004
Nome: Configuration
Caminho: src/hl_mcp/server/config.py
Dependências: Nenhuma
Exports: load_config, save_config, ConfigManager
Linhas: ~120
```

### Código

```python
"""
HLS-MCP-004: Configuration
==========================

Gerenciamento de configuração do servidor.

Changelog:
    - v1.0.0 (2026-02-01): Implementação inicial
"""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Optional
import logging

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from .main import ServerConfig

logger = logging.getLogger(__name__)


DEFAULT_CONFIG_PATHS = [
    Path.home() / ".hl-mcp" / "config.yaml",
    Path.cwd() / "hl-mcp.yaml",
    Path.cwd() / ".hl-mcp.yaml",
]


def find_config_file() -> Optional[Path]:
    """Procura arquivo de configuração."""
    for path in DEFAULT_CONFIG_PATHS:
        if path.exists():
            return path
    return None


def load_config(path: Optional[Path] = None) -> ServerConfig:
    """
    Carrega configuração de arquivo ou ambiente.

    Args:
        path: Caminho do arquivo (opcional)

    Returns:
        ServerConfig carregado
    """
    # Tenta encontrar arquivo
    if path is None:
        path = find_config_file()

    if path and path.exists() and YAML_AVAILABLE:
        logger.info(f"Carregando config de {path}")
        with open(path) as f:
            data = yaml.safe_load(f)

        return ServerConfig(
            name=data.get("name", "human-layer"),
            version=data.get("version", "1.0.0"),
            max_concurrent_runs=data.get("max_concurrent_runs", 3),
            enable_browser=data.get("enable_browser", True),
            enable_redundancy=data.get("enable_redundancy", True),
            log_level=data.get("log_level", "INFO"),
        )

    # Fallback para variáveis de ambiente
    logger.info("Carregando config de variáveis de ambiente")
    return ServerConfig.from_env()


def save_config(config: ServerConfig, path: Path) -> None:
    """
    Salva configuração em arquivo.

    Args:
        config: Configuração a salvar
        path: Caminho de destino
    """
    if not YAML_AVAILABLE:
        raise RuntimeError("PyYAML não disponível")

    path.parent.mkdir(parents=True, exist_ok=True)

    data = asdict(config)

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    logger.info(f"Config salva em {path}")


class ConfigManager:
    """
    Gerenciador de configuração.

    Example:
        >>> manager = ConfigManager()
        >>> config = manager.load()
        >>> manager.save(config)
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or (
            Path.home() / ".hl-mcp" / "config.yaml"
        )

    def load(self) -> ServerConfig:
        """Carrega configuração."""
        return load_config(self.config_path)

    def save(self, config: ServerConfig) -> None:
        """Salva configuração."""
        save_config(config, self.config_path)

    def reset(self) -> ServerConfig:
        """Reseta para configuração padrão."""
        config = ServerConfig()
        self.save(config)
        return config


__all__ = [
    "find_config_file",
    "load_config",
    "save_config",
    "ConfigManager",
]
```

---

## Server __init__.py

```python
"""
HLS-MCP: Human Layer MCP Server
===============================

Servidor MCP completo do Human Layer.

Usage:
    $ python -m hl_mcp.server

Or programmatically:
    >>> from hl_mcp.server import create_server
    >>> server = create_server()
    >>> await server.run()
"""

from .main import (
    ServerConfig,
    HumanLayerMCPServer,
    create_server,
    main,
)

from .config import (
    load_config,
    save_config,
    ConfigManager,
)

__all__ = [
    "ServerConfig",
    "HumanLayerMCPServer",
    "create_server",
    "main",
    "load_config",
    "save_config",
    "ConfigManager",
]
```

---

## Package __init__.py (Root)

```python
"""
Human Layer MCP Server
======================

MCP server que implementa as 7 Human Layers para validação
de código e interfaces.

Quick Start:
    $ python -m hl_mcp.server

Or use the CLI:
    $ hl-mcp start

Example usage via MCP:
    >>> # Run full validation
    >>> result = await run_full_validation(
    ...     target="Login form",
    ...     code="def login(user, pwd): ...",
    ... )
    >>> print(result["final_verdict"])

Modules:
    - models: Data models (Finding, LayerResult, etc.)
    - llm: LLM integration (Claude, OpenAI)
    - browser: Browser automation (Playwright)
    - engine: Core engine (Runner, Orchestrator)
    - layers: 7 Human Layers
    - perspectives: 6 Testing Perspectives
    - cognitive: Budget, Trust, Triage, Feedback
    - server: MCP Server

Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Human Layer Team"

from .server import create_server, HumanLayerMCPServer

__all__ = [
    "__version__",
    "create_server",
    "HumanLayerMCPServer",
]
```

---

## pyproject.toml

```toml
[project]
name = "hl-mcp"
version = "1.0.0"
description = "Human Layer MCP Server - 7 Layers of Validation"
readme = "README.md"
requires-python = ">=3.11"
license = {text = "MIT"}

dependencies = [
    "mcp>=1.0.0",
    "anthropic>=0.18.0",
    "openai>=1.10.0",
    "pydantic>=2.0.0",
    "httpx>=0.26.0",
    "PyYAML>=6.0.0",
]

[project.optional-dependencies]
browser = [
    "playwright>=1.40.0",
]
all = [
    "hl-mcp[browser]",
]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=4.1.0",
    "ruff>=0.2.0",
    "mypy>=1.8.0",
]

[project.scripts]
hl-mcp = "hl_mcp.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/hl_mcp"]

[tool.ruff]
line-length = 88
target-version = "py311"

[tool.mypy]
python_version = "3.11"
strict = true

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

---

## Resumo do Block 12

| Módulo | ID | Função | Linhas |
|--------|----|--------|--------|
| MCPServer | HLS-MCP-001 | Servidor principal | ~300 |
| ToolHandlers | HLS-MCP-002 | Handlers de tools | ~350 |
| ResourceHandlers | HLS-MCP-003 | Handlers de resources | ~200 |
| Configuration | HLS-MCP-004 | Gerenciamento de config | ~120 |
| **TOTAL** | | | **~970** |

---

## Resumo Final do Roadmap Completo

| Block | Tema | Módulos | Linhas |
|-------|------|---------|--------|
| 01 | Dissection | Inventário | ~15,000 |
| 02 | Architecture | 67 Legos | ~18,000 |
| 03 | Resources & Builder | MCP Resources | ~20,000 |
| 04 | Data Models (Part 1) | 7 modules | ~22,000 |
| 05 | Data Models (Part 2) | + models | ~28,000 |
| 06 | LLM Integration | 5 modules | ~32,000 |
| 07 | Browser Automation | 6 modules | ~28,000 |
| 08 | Core Engine | 5 modules | ~35,000 |
| 09 | 7 Human Layers | 7 modules | ~30,000 |
| 10 | 6 Perspectives | 7 modules | ~22,000 |
| 11 | Cognitive Modules | 5 modules | ~25,000 |
| 12 | MCP Server Final | 4 modules | ~18,000 |
| **TOTAL** | | **67 módulos** | **~293,000** |

---

## Arquitetura Final

```
src/hl_mcp/
├── __init__.py              # Package root
├── cli.py                   # CLI entry point
│
├── models/                  # HLS-MDL-001 to HLS-MDL-007
│   ├── __init__.py
│   ├── enums.py
│   ├── finding.py
│   ├── layer_result.py
│   ├── report.py
│   ├── journey.py
│   ├── test_suite.py
│   └── review.py
│
├── llm/                     # HLS-LLM-001 to HLS-LLM-005
│   ├── __init__.py
│   ├── client.py
│   ├── claude.py
│   ├── openai.py
│   ├── templates.py
│   └── parser.py
│
├── browser/                 # HLS-BRW-001 to HLS-BRW-006
│   ├── __init__.py
│   ├── driver.py
│   ├── actions.py
│   ├── screenshots.py
│   ├── video.py
│   ├── accessibility.py
│   └── journey.py
│
├── engine/                  # HLS-ENG-001 to HLS-ENG-005
│   ├── __init__.py
│   ├── runner.py
│   ├── redundancy.py
│   ├── orchestrator.py
│   ├── veto.py
│   └── consensus.py
│
├── layers/                  # HLS-LAY-001 to HLS-LAY-007
│   ├── __init__.py
│   ├── hl1_uiux.py
│   ├── hl2_security.py
│   ├── hl3_edge_cases.py
│   ├── hl4_accessibility.py
│   ├── hl5_performance.py
│   ├── hl6_integration.py
│   └── hl7_final_check.py
│
├── perspectives/            # HLS-PRS-001 to HLS-PRS-006
│   ├── __init__.py
│   ├── base.py
│   ├── tired_user.py
│   ├── malicious_insider.py
│   ├── confused_newbie.py
│   ├── power_user.py
│   ├── auditor.py
│   ├── operator_3am.py
│   └── runner.py
│
├── cognitive/               # HLS-COG-001 to HLS-COG-005
│   ├── __init__.py
│   ├── budget.py
│   ├── trust.py
│   ├── triage.py
│   ├── feedback.py
│   └── confidence.py
│
└── server/                  # HLS-MCP-001 to HLS-MCP-004
    ├── __init__.py
    ├── main.py
    ├── tools.py
    ├── resources.py
    └── config.py
```

---

## Como Usar

```bash
# Instalar
pip install hl-mcp

# Instalar com browser automation
pip install hl-mcp[browser]
playwright install

# Rodar servidor MCP
python -m hl_mcp.server

# Ou via CLI
hl-mcp start
```

---

## LEGO_INDEX.yaml Final

O LEGO_INDEX.yaml completo contém todos os 67 módulos indexados:

- 7 módulos de Models (HLS-MDL-*)
- 5 módulos de LLM (HLS-LLM-*)
- 6 módulos de Browser (HLS-BRW-*)
- 5 módulos de Engine (HLS-ENG-*)
- 7 módulos de Layers (HLS-LAY-*)
- 7 módulos de Perspectives (HLS-PRS-*)
- 5 módulos de Cognitive (HLS-COG-*)
- 4 módulos de Server (HLS-MCP-*)

Cada módulo pode ser encontrado com Ctrl+F no LEGO_INDEX.yaml usando seu ID único.

---

**ROADMAP COMPLETO!**

O Human Layer MCP Server está pronto para implementação, com:
- 67 módulos Lego documentados
- ~15,000 linhas de código especificado
- Arquitetura hiper-modularizada
- Índice navegável para manutenção fácil
