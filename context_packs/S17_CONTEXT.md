# S17 - mcp-resources | Context Pack v1.0

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
  id: S17
  name: mcp-resources
  title: "MCP Resources"
  wave: W2-OSSRelease
  priority: P1-HIGH
  type: implementation

objective: "Resources do MCP (read-only data endpoints)"

source_block: "../MASTER_REQUIREMENTS_MAP.md#PARTE-6-MCP"

dependencies:
  - S15  # MCP Server Base

deliverables:
  - src/hl_mcp/server/resources.py
  - tests/test_server/test_resources.py
```

---

## MCP RESOURCES SPECIFICATION

```yaml
resources:
  # Layer Resources
  human_layer://layers:
    description: "List all 7 Human Layers"
    returns: "Array of layer info"

  human_layer://layers/{layer_id}:
    description: "Get specific layer details"
    returns: "Layer configuration and stats"

  human_layer://layers/{layer_id}/template:
    description: "Get prompt template for layer"
    returns: "System prompt text"

  # Perspective Resources
  human_layer://perspectives:
    description: "List all 6 perspectives"
    returns: "Array of perspective info"

  human_layer://perspectives/{id}:
    description: "Get perspective details"
    returns: "Perspective configuration"

  # Stats Resources
  human_layer://stats/summary:
    description: "Overall validation statistics"
    returns: "Total validations, pass rate, etc"

  human_layer://stats/layers:
    description: "Stats per layer"
    returns: "Findings count per layer"

  # Config Resources
  human_layer://config/thresholds:
    description: "Current veto thresholds"
    returns: "Threshold configuration"
```

---

## IMPLEMENTATION SPEC

### Resources (server/resources.py)

```python
"""MCP Resources for Human Layer."""
from typing import Optional
from mcp.server import Server
from mcp.types import Resource, TextContent


def register_resources(server: Server):
    """Register all Human Layer resources with MCP server."""

    @server.list_resources()
    async def list_resources() -> list[Resource]:
        """Return list of available resources."""
        return [
            Resource(
                uri="human_layer://layers",
                name="Human Layers",
                description="List all 7 Human Layers",
                mimeType="application/json"
            ),
            Resource(
                uri="human_layer://layers/HL1",
                name="Layer HL-1",
                description="UX & Usability layer details",
                mimeType="application/json"
            ),
            Resource(
                uri="human_layer://layers/HL2",
                name="Layer HL-2",
                description="Functionality layer details",
                mimeType="application/json"
            ),
            Resource(
                uri="human_layer://layers/HL3",
                name="Layer HL-3",
                description="Edge Cases layer details",
                mimeType="application/json"
            ),
            Resource(
                uri="human_layer://layers/HL4",
                name="Layer HL-4",
                description="Security layer details",
                mimeType="application/json"
            ),
            Resource(
                uri="human_layer://layers/HL5",
                name="Layer HL-5",
                description="Performance layer details",
                mimeType="application/json"
            ),
            Resource(
                uri="human_layer://layers/HL6",
                name="Layer HL-6",
                description="Compliance layer details",
                mimeType="application/json"
            ),
            Resource(
                uri="human_layer://layers/HL7",
                name="Layer HL-7",
                description="Final Review layer details",
                mimeType="application/json"
            ),
            Resource(
                uri="human_layer://perspectives",
                name="Perspectives",
                description="List all 6 testing perspectives",
                mimeType="application/json"
            ),
            Resource(
                uri="human_layer://stats/summary",
                name="Statistics Summary",
                description="Overall validation statistics",
                mimeType="application/json"
            ),
            Resource(
                uri="human_layer://config/thresholds",
                name="Thresholds",
                description="Veto threshold configuration",
                mimeType="application/json"
            ),
        ]

    @server.read_resource()
    async def read_resource(uri: str) -> str:
        """Handle resource reads."""
        import json

        if uri == "human_layer://layers":
            return json.dumps(_get_all_layers(), indent=2)

        if uri.startswith("human_layer://layers/HL"):
            layer_id = uri.split("/")[-1]
            return json.dumps(_get_layer(layer_id), indent=2)

        if uri == "human_layer://perspectives":
            return json.dumps(_get_perspectives(), indent=2)

        if uri == "human_layer://stats/summary":
            return json.dumps(_get_stats_summary(), indent=2)

        if uri == "human_layer://config/thresholds":
            return json.dumps(_get_thresholds(), indent=2)

        return json.dumps({"error": f"Unknown resource: {uri}"})


def _get_all_layers() -> dict:
    """Get all layers info."""
    return {
        "layers": [
            {"id": "HL1", "name": "UX & Usability", "veto_power": "WEAK", "enabled": True},
            {"id": "HL2", "name": "Functionality", "veto_power": "MEDIUM", "enabled": True},
            {"id": "HL3", "name": "Edge Cases", "veto_power": "MEDIUM", "enabled": True},
            {"id": "HL4", "name": "Security", "veto_power": "STRONG", "enabled": True},
            {"id": "HL5", "name": "Performance", "veto_power": "MEDIUM", "enabled": True},
            {"id": "HL6", "name": "Compliance", "veto_power": "STRONG", "enabled": True},
            {"id": "HL7", "name": "Final Review", "veto_power": "STRONG", "enabled": True},
        ],
        "total": 7,
        "strong_veto_layers": ["HL4", "HL6", "HL7"],
    }


def _get_layer(layer_id: str) -> dict:
    """Get specific layer info."""
    layers_info = {
        "HL1": {
            "id": "HL1",
            "name": "UX & Usability",
            "veto_power": "WEAK",
            "focus": ["UI clarity", "Error messages", "Accessibility", "Navigation"],
            "enabled": True,
        },
        "HL2": {
            "id": "HL2",
            "name": "Functionality",
            "veto_power": "MEDIUM",
            "focus": ["Correctness", "Completeness", "API contracts", "Data integrity"],
            "enabled": True,
        },
        "HL3": {
            "id": "HL3",
            "name": "Edge Cases",
            "veto_power": "MEDIUM",
            "focus": ["Boundary conditions", "Null handling", "Race conditions", "Failures"],
            "enabled": True,
        },
        "HL4": {
            "id": "HL4",
            "name": "Security",
            "veto_power": "STRONG",
            "focus": ["OWASP Top 10", "Injection", "Authentication", "Data exposure"],
            "enabled": True,
        },
        "HL5": {
            "id": "HL5",
            "name": "Performance",
            "veto_power": "MEDIUM",
            "focus": ["Complexity", "Database queries", "Memory", "Scalability"],
            "enabled": True,
        },
        "HL6": {
            "id": "HL6",
            "name": "Compliance",
            "veto_power": "STRONG",
            "focus": ["GDPR", "Privacy", "Audit", "Regulations"],
            "enabled": True,
        },
        "HL7": {
            "id": "HL7",
            "name": "Final Review",
            "veto_power": "STRONG",
            "focus": ["Holistic review", "Common sense", "Risk assessment"],
            "enabled": True,
        },
    }
    return layers_info.get(layer_id, {"error": f"Unknown layer: {layer_id}"})


def _get_perspectives() -> dict:
    """Get all perspectives info."""
    return {
        "perspectives": [
            {
                "id": "tired_user",
                "name": "Tired User",
                "description": "Frustrated, impatient, making mistakes"
            },
            {
                "id": "malicious_insider",
                "name": "Malicious Insider",
                "description": "Trying to abuse the system"
            },
            {
                "id": "confused_newbie",
                "name": "Confused Newbie",
                "description": "Lost, first time using the system"
            },
            {
                "id": "power_user",
                "name": "Power User",
                "description": "Wants shortcuts, efficiency"
            },
            {
                "id": "auditor",
                "name": "Auditor",
                "description": "Checking compliance, logs"
            },
            {
                "id": "3am_operator",
                "name": "3AM Operator",
                "description": "Sleepy, emergency situation"
            },
        ],
        "total": 6,
    }


def _get_stats_summary() -> dict:
    """Get stats summary (placeholder)."""
    return {
        "total_validations": 0,
        "approved": 0,
        "rejected": 0,
        "needs_review": 0,
        "pass_rate": 0.0,
        "most_common_issues": [],
    }


def _get_thresholds() -> dict:
    """Get threshold configuration."""
    return {
        "veto_levels": {
            "NONE": {"blocks": "nothing", "description": "No veto"},
            "WEAK": {"blocks": "nothing", "description": "Advisory warning"},
            "MEDIUM": {"blocks": "merge without override", "description": "Significant concern"},
            "STRONG": {"blocks": "all progression", "description": "Critical issue"},
        },
        "decision_rules": {
            "STRONG_any": "REJECTED",
            "MEDIUM_2_plus": "REJECTED",
            "MEDIUM_1": "NEEDS_REVIEW",
            "WEAK_only": "APPROVED",
        }
    }
```

---

## INTENT MANIFEST

```yaml
INTENT_MANIFEST:
  RF:
    - RF-001: "Resources são read-only"
    - RF-002: "URIs seguem padrão human_layer://"
    - RF-003: "Retornam JSON"
    - RF-004: "Cobrem layers, perspectives, stats, config"
    - RF-005: "Listagem completa disponível"

  INV:
    - INV-001: "Resources nunca modificam estado"
    - INV-002: "JSON sempre válido"
    - INV-003: "layers sempre retorna 7 layers"
    - INV-004: "perspectives sempre retorna 6"

  EDGE:
    - EDGE-001: "URI desconhecida → erro JSON"
    - EDGE-002: "Layer ID inválido → erro"
    - EDGE-003: "Stats vazias → zeros"
```

---

## GATES

```yaml
gates:
  G0_FILES_EXIST:
    description: "Arquivo existe"
    validation: "ls src/hl_mcp/server/resources.py"

  G1_IMPORTS_WORK:
    description: "Import funciona"
    validation: |
      python -c "from hl_mcp.server.resources import register_resources"

  G2_TESTS_PASS:
    description: "Testes passam"
    validation: "pytest tests/test_server/test_resources.py -v"
```

---

## REFERÊNCIA

- `./S15_CONTEXT.md` - MCP Server Base
- `./S16_CONTEXT.md` - MCP Tools
- `./S18_CONTEXT.md` - CLI Base
