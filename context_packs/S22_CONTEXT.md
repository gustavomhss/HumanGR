# S22 - docs-api | Context Pack v1.0

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
  id: S22
  name: docs-api
  title: "Docs: API Reference"
  wave: W2-OSSRelease
  priority: P1-HIGH
  type: documentation

objective: "Referência de API completa"

source_block: "../MASTER_REQUIREMENTS_MAP.md#PARTE-8-DOCS"

dependencies:
  - S20  # Docs Getting Started

deliverables:
  - docs/api/index.md
  - docs/api/models.md
  - docs/api/layers.md
  - docs/api/llm.md
  - docs/api/server.md
  - docs/api/cli.md
```

---

## API DOCS STRUCTURE

```yaml
api_docs:
  index.md: "API overview and module map"
  models.md: "Data models (Finding, LayerResult, Report)"
  layers.md: "Layer classes and prompts"
  llm.md: "LLM client interfaces"
  server.md: "MCP server and tools"
  cli.md: "CLI commands reference"
```

---

## CONTENT SPECS

### api/index.md

```markdown
# API Reference

Complete API documentation for Human Layer.

## Module Map

```
hl_mcp/
├── models/         # Data models
│   ├── enums.py    # VetoLevel, Severity, LayerStatus
│   ├── findings.py # Finding
│   ├── layers.py   # LayerResult
│   └── report.py   # HumanLayerReport
│
├── layers/         # Human Layers
│   ├── base.py     # BaseHumanLayer
│   ├── prompts.py  # LayerPromptManager
│   ├── hl1_ux.py   # HL-1 UX Layer
│   └── ...         # HL-2 through HL-7
│
├── llm/            # LLM Clients
│   ├── base.py     # BaseLLMClient
│   ├── claude.py   # ClaudeClient
│   ├── openai.py   # OpenAIClient
│   └── factory.py  # create_llm_client
│
├── core/           # Core Engine
│   ├── consensus.py # ConsensusEngine
│   ├── veto.py      # VetoGate
│   └── decision.py  # DecisionEngine
│
├── server/         # MCP Server
│   ├── server.py   # HumanLayerServer
│   ├── tools.py    # MCP Tools
│   └── resources.py # MCP Resources
│
└── cli/            # CLI
    ├── main.py     # Entry point
    └── commands/   # Command implementations
```

## Quick Links

- [Models](models.md) - Data structures
- [Layers](layers.md) - Validation layers
- [LLM](llm.md) - LLM integration
- [Server](server.md) - MCP server
- [CLI](cli.md) - Command line
```

### api/models.md (Abbreviated)

```markdown
# Models API

Data models for Human Layer.

## VetoLevel

```python
from hl_mcp.models import VetoLevel

class VetoLevel(Enum):
    NONE = "none"      # No veto
    WEAK = "weak"      # Advisory
    MEDIUM = "medium"  # Can block merge
    STRONG = "strong"  # Blocks everything
```

## Finding

```python
from hl_mcp.models import Finding

class Finding(BaseModel):
    id: str                    # Unique ID
    layer_id: LayerID          # Which layer
    severity: Severity         # CRITICAL-INFO
    title: str                 # Short title
    description: str           # Details
    suggestion: Optional[str]  # How to fix
```

## LayerResult

```python
from hl_mcp.models import LayerResult

class LayerResult(BaseModel):
    layer_id: LayerID
    status: LayerStatus
    veto_level: VetoLevel
    findings: List[Finding]
    execution_time_ms: int
```

[Continue for all models...]
```

---

## INTENT MANIFEST

```yaml
INTENT_MANIFEST:
  RF:
    - RF-001: "Documenta todos os módulos públicos"
    - RF-002: "Inclui type hints"
    - RF-003: "Exemplos de uso"
    - RF-004: "Module map claro"
    - RF-005: "Auto-gerado onde possível"

  INV:
    - INV-001: "Código de exemplo funciona"
    - INV-002: "Types corretos"
    - INV-003: "Links válidos"
```

---

## GATES

```yaml
gates:
  G0_FILES_EXIST:
    description: "API docs existem"
    validation: |
      ls docs/api/index.md
      ls docs/api/models.md
      ls docs/api/layers.md

  G1_CODE_VALID:
    description: "Código de exemplo é válido"
    validation: "python -c 'from hl_mcp.models import VetoLevel, Finding, LayerResult'"
```

---

## REFERÊNCIA

- `./S20_CONTEXT.md` - Docs Getting Started
- `./S23_CONTEXT.md` - GitHub OSS Setup
