# S16 - mcp-tools | Context Pack v1.0

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
  id: S16
  name: mcp-tools
  title: "MCP Tools"
  wave: W2-OSSRelease
  priority: P0-CRITICAL
  type: implementation

objective: "Tools do MCP (validate, test, report)"

source_block: "../MASTER_REQUIREMENTS_MAP.md#PARTE-6-MCP"

dependencies:
  - S15  # MCP Server Base

deliverables:
  - src/hl_mcp/server/tools.py
  - tests/test_server/test_tools.py
```

---

## MCP TOOLS SPECIFICATION

```yaml
tools:
  validate:
    description: "Run Human Layer validation on an action"
    input:
      agent_id: "ID of the agent performing the action"
      action_type: "Type of action (file_write, api_call, etc)"
      action_description: "What the action does"
      code_diff: "Optional: code changes"
      affected_files: "Optional: list of files"
    output:
      decision: "approved | rejected | needs_review"
      findings: "List of findings from all layers"
      veto_level: "NONE | WEAK | MEDIUM | STRONG"

  test_action:
    description: "Quick test without full validation"
    input:
      action_description: "Quick description"
      layers: "Optional: specific layers to run"
    output:
      quick_result: "pass | warn | fail"

  get_report:
    description: "Get detailed report for a validation"
    input:
      report_id: "ID from previous validation"
    output:
      full_report: "Complete HumanLayerReport"

  list_layers:
    description: "List all available layers"
    output:
      layers: "List of layer info"
```

---

## IMPLEMENTATION SPEC

### Tools (server/tools.py)

```python
"""MCP Tools for Human Layer."""
from typing import Optional, List
import uuid
from mcp.server import Server
from mcp.types import Tool, TextContent

from ..core import DecisionEngine
from ..layers.base import ActionContext
from ..layers import get_registry
from ..llm import create_llm_client


def register_tools(server: Server, decision_engine: DecisionEngine):
    """Register all Human Layer tools with MCP server."""

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """Return list of available tools."""
        return [
            Tool(
                name="validate",
                description="Run full Human Layer validation on an action",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "agent_id": {
                            "type": "string",
                            "description": "ID of the agent"
                        },
                        "action_type": {
                            "type": "string",
                            "description": "Type of action"
                        },
                        "action_description": {
                            "type": "string",
                            "description": "What the action does"
                        },
                        "code_diff": {
                            "type": "string",
                            "description": "Code changes (optional)"
                        },
                        "affected_files": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Files affected (optional)"
                        },
                    },
                    "required": ["agent_id", "action_type", "action_description"]
                }
            ),
            Tool(
                name="test_action",
                description="Quick test without full validation",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "action_description": {
                            "type": "string",
                            "description": "Quick description"
                        },
                        "layers": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific layers to run"
                        }
                    },
                    "required": ["action_description"]
                }
            ),
            Tool(
                name="get_report",
                description="Get detailed report by ID",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "report_id": {
                            "type": "string",
                            "description": "Report ID"
                        }
                    },
                    "required": ["report_id"]
                }
            ),
            Tool(
                name="list_layers",
                description="List all Human Layers",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        """Handle tool calls."""
        if name == "validate":
            return await _handle_validate(arguments, decision_engine)
        elif name == "test_action":
            return await _handle_test_action(arguments)
        elif name == "get_report":
            return await _handle_get_report(arguments)
        elif name == "list_layers":
            return await _handle_list_layers()
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def _handle_validate(args: dict, engine: DecisionEngine) -> list[TextContent]:
    """Handle validate tool call."""
    action = ActionContext(
        agent_id=args["agent_id"],
        action_type=args["action_type"],
        action_description=args["action_description"],
        code_diff=args.get("code_diff"),
        affected_files=args.get("affected_files", []),
    )

    report_id = str(uuid.uuid4())

    # Run all layers with triple redundancy
    # (This would be expanded to actually run layers)

    result = {
        "report_id": report_id,
        "decision": "approved",  # Placeholder
        "veto_level": "NONE",
        "findings_count": 0,
    }

    import json
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def _handle_test_action(args: dict) -> list[TextContent]:
    """Handle test_action tool call."""
    result = {
        "quick_result": "pass",
        "message": "Quick test passed",
    }
    import json
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def _handle_get_report(args: dict) -> list[TextContent]:
    """Handle get_report tool call."""
    result = {
        "report_id": args["report_id"],
        "status": "not_found",
        "message": "Report storage not yet implemented",
    }
    import json
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def _handle_list_layers() -> list[TextContent]:
    """Handle list_layers tool call."""
    layers = [
        {"id": "HL1", "name": "UX & Usability", "veto_power": "WEAK"},
        {"id": "HL2", "name": "Functionality", "veto_power": "MEDIUM"},
        {"id": "HL3", "name": "Edge Cases", "veto_power": "MEDIUM"},
        {"id": "HL4", "name": "Security", "veto_power": "STRONG"},
        {"id": "HL5", "name": "Performance", "veto_power": "MEDIUM"},
        {"id": "HL6", "name": "Compliance", "veto_power": "STRONG"},
        {"id": "HL7", "name": "Final Review", "veto_power": "STRONG"},
    ]
    import json
    return [TextContent(type="text", text=json.dumps({"layers": layers}, indent=2))]
```

---

## INTENT MANIFEST

```yaml
INTENT_MANIFEST:
  RF:
    - RF-001: "validate tool roda validação completa"
    - RF-002: "test_action para testes rápidos"
    - RF-003: "get_report retorna relatório completo"
    - RF-004: "list_layers mostra todas as 7 layers"
    - RF-005: "Tools seguem MCP schema"

  INV:
    - INV-001: "validate sempre retorna decision"
    - INV-002: "report_id é UUID único"
    - INV-003: "JSON output sempre válido"
    - INV-004: "list_layers sempre retorna 7 layers"

  EDGE:
    - EDGE-001: "Tool desconhecido → erro"
    - EDGE-002: "Campos obrigatórios faltando → erro"
    - EDGE-003: "Report não encontrado → status not_found"
```

---

## GATES

```yaml
gates:
  G0_FILES_EXIST:
    description: "Arquivo existe"
    validation: "ls src/hl_mcp/server/tools.py"

  G1_IMPORTS_WORK:
    description: "Import funciona"
    validation: |
      python -c "from hl_mcp.server.tools import register_tools"

  G2_TESTS_PASS:
    description: "Testes passam"
    validation: "pytest tests/test_server/test_tools.py -v"
```

---

## REFERÊNCIA

- `./S15_CONTEXT.md` - MCP Server Base
- `./S17_CONTEXT.md` - MCP Resources
