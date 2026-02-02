# S15 - mcp-server-base | Context Pack v1.0

---

## PRODUCT REFERENCE

```yaml
product_reference:
  product_id: "HUMANGR"
  product_name: "Human Layer MCP Server"
  wave: "W2-OSSRelease"
  product_pack: "./PRODUCT_PACK.md"
  sprint_index: "./SPRINT_INDEX.yaml"
```

---

## RELOAD ANCHOR

```yaml
sprint:
  id: S15
  name: mcp-server-base
  title: "MCP Server Base"
  wave: W2-OSSRelease
  priority: P0-CRITICAL
  type: implementation

objective: "Base do MCP Server usando Model Context Protocol"

source_block: "../MASTER_REQUIREMENTS_MAP.md#PARTE-6-MCP"

dependencies:
  - S14  # Veto Gate

deliverables:
  - src/hl_mcp/server/__init__.py
  - src/hl_mcp/server/server.py
  - src/hl_mcp/server/config.py
  - tests/test_server/test_server.py
```

---

## MCP PROTOCOL OVERVIEW

```yaml
mcp_protocol:
  description: "Model Context Protocol - standard for LLM integrations"

  components:
    tools: "Actions the server can perform"
    resources: "Data the server exposes (read-only)"
    prompts: "Templates for common operations"

  supported_clients:
    - "Claude Desktop"
    - "Cline"
    - "Continue.dev"
    - "Custom MCP clients"

  transport:
    - "stdio (stdin/stdout)"
    - "HTTP/SSE (Server-Sent Events)"
```

---

## IMPLEMENTATION SPEC

### Server (server/server.py)

```python
"""Human Layer MCP Server."""
import asyncio
from typing import Optional
from mcp.server import Server
from mcp.server.stdio import stdio_server

from .config import ServerConfig
from ..core import DecisionEngine
from ..llm import create_llm_client


class HumanLayerServer:
    """MCP Server for Human Layer validation.

    This server exposes Human Layer functionality through the
    Model Context Protocol, allowing integration with Claude Desktop,
    Cline, and other MCP-compatible clients.
    """

    def __init__(self, config: Optional[ServerConfig] = None):
        self.config = config or ServerConfig()
        self._server = Server("human-layer")
        self._decision_engine = DecisionEngine()
        self._setup_handlers()

    def _setup_handlers(self):
        """Setup MCP request handlers."""
        # Tools will be registered in S16
        # Resources will be registered in S17
        pass

    async def run_stdio(self):
        """Run server using stdio transport."""
        async with stdio_server() as (read_stream, write_stream):
            await self._server.run(
                read_stream,
                write_stream,
                self._server.create_initialization_options()
            )

    async def run_http(self, host: str = "127.0.0.1", port: int = 8080):
        """Run server using HTTP/SSE transport."""
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.routing import Route

        sse = SseServerTransport("/sse")

        async def handle_sse(request):
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                await self._server.run(
                    streams[0], streams[1],
                    self._server.create_initialization_options()
                )

        app = Starlette(routes=[
            Route("/sse", endpoint=handle_sse),
        ])

        import uvicorn
        await uvicorn.Server(
            uvicorn.Config(app, host=host, port=port)
        ).serve()


def create_server(config: Optional[ServerConfig] = None) -> HumanLayerServer:
    """Factory function to create MCP server."""
    return HumanLayerServer(config)


async def main():
    """Entry point for MCP server."""
    server = create_server()
    await server.run_stdio()


if __name__ == "__main__":
    asyncio.run(main())
```

### Config (server/config.py)

```python
"""Server configuration."""
from typing import Optional, List
from pydantic import BaseModel, Field


class LayerConfig(BaseModel):
    """Configuration for individual layers."""
    enabled: bool = True
    custom_prompt: Optional[str] = None


class ServerConfig(BaseModel):
    """Configuration for Human Layer MCP Server."""

    # Server settings
    name: str = "human-layer"
    version: str = "1.0.0"

    # LLM settings (BYOK)
    llm_provider: str = "claude"
    llm_model: Optional[str] = None

    # Layer settings
    layers: dict[str, LayerConfig] = Field(default_factory=dict)
    triple_redundancy: bool = True

    # Logging
    log_level: str = "INFO"
    log_findings: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "ServerConfig":
        """Load config from YAML file."""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Load config from environment variables."""
        import os
        return cls(
            llm_provider=os.getenv("HL_LLM_PROVIDER", "claude"),
            llm_model=os.getenv("HL_LLM_MODEL"),
            log_level=os.getenv("HL_LOG_LEVEL", "INFO"),
        )
```

---

## INTENT MANIFEST

```yaml
INTENT_MANIFEST:
  RF:
    - RF-001: "MCP Server base funcional"
    - RF-002: "Suporta stdio e HTTP/SSE transport"
    - RF-003: "Configuração via YAML ou env vars"
    - RF-004: "Factory function para criar server"
    - RF-005: "Integra DecisionEngine"

  INV:
    - INV-001: "Server name sempre é 'human-layer'"
    - INV-002: "BYOK - LLM key vem do client"
    - INV-003: "Triple redundancy habilitado por default"
    - INV-004: "Sem credenciais hardcoded"

  EDGE:
    - EDGE-001: "Config não encontrado → defaults"
    - EDGE-002: "LLM não configurado → erro claro"
    - EDGE-003: "Porta ocupada → erro HTTP"
```

---

## GATES

```yaml
gates:
  G0_FILES_EXIST:
    description: "Arquivos existem"
    validation: |
      ls src/hl_mcp/server/server.py
      ls src/hl_mcp/server/config.py

  G1_IMPORTS_WORK:
    description: "Imports funcionam"
    validation: |
      python -c "from hl_mcp.server import HumanLayerServer, ServerConfig"

  G2_TESTS_PASS:
    description: "Testes passam"
    validation: "pytest tests/test_server/ -v"

  G3_MCP_COMPATIBLE:
    description: "MCP protocol compatible"
    validation: |
      python -c "
      from hl_mcp.server import create_server
      server = create_server()
      assert hasattr(server, 'run_stdio')
      assert hasattr(server, 'run_http')
      "
```

---

## REFERÊNCIA

- `./S14_CONTEXT.md` - Decision Engine
- `./S16_CONTEXT.md` - MCP Tools
- `./S17_CONTEXT.md` - MCP Resources
