# S05 - layer-base | Context Pack v1.0

---

## PRODUCT REFERENCE

```yaml
product_reference:
  product_id: "HUMANGR"
  product_name: "Human Layer MCP Server"
  wave: "W1-CoreEngine"
  product_pack: "./PRODUCT_PACK.md"
  sprint_index: "./SPRINT_INDEX.yaml"
```

---

## RELOAD ANCHOR

```yaml
sprint:
  id: S05
  name: layer-base
  title: "Layer Base Class"
  wave: W1-CoreEngine
  priority: P0-CRITICAL
  type: implementation

objective: "Classe base para Human Layers e sistema de prompts"

source_block: "../MASTER_REQUIREMENTS_MAP.md#PARTE-4-LAYERS"

dependencies:
  - S03  # LLM Client

deliverables:
  - src/hl_mcp/layers/__init__.py
  - src/hl_mcp/layers/base.py
  - src/hl_mcp/layers/prompts.py
  - src/hl_mcp/layers/registry.py
  - tests/test_layers/test_base.py
  - tests/test_layers/test_prompts.py
```

---

## LAYER ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────┐
│                        LAYER SYSTEM                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  BaseHumanLayer (ABC)                                           │
│  ├── layer_id: LayerID                                          │
│  ├── veto_power: VetoLevel                                      │
│  ├── llm_client: BaseLLMClient                                  │
│  └── execute(action) -> LayerResult                             │
│                                                                  │
│  LayerPromptManager                                              │
│  ├── get_system_prompt(layer_id) -> str                         │
│  ├── get_analysis_prompt(action) -> str                         │
│  └── parse_llm_response(text) -> List[Finding]                  │
│                                                                  │
│  LayerRegistry                                                   │
│  ├── register(layer)                                             │
│  ├── get(layer_id) -> BaseHumanLayer                            │
│  └── all() -> List[BaseHumanLayer]                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## IMPLEMENTATION SPECS

### Base Layer (layers/base.py)

```python
"""Base class for Human Layers."""
from abc import ABC, abstractmethod
from typing import Optional, List
from datetime import datetime

from ..models import Finding, LayerResult, LayerID, LayerStatus, VetoLevel, Severity
from ..llm import BaseLLMClient

class ActionContext:
    """Context for the action being validated."""

    def __init__(
        self,
        agent_id: str,
        action_type: str,
        action_description: str,
        code_diff: Optional[str] = None,
        affected_files: Optional[List[str]] = None,
        environment: Optional[dict] = None,
    ):
        self.agent_id = agent_id
        self.action_type = action_type
        self.action_description = action_description
        self.code_diff = code_diff
        self.affected_files = affected_files or []
        self.environment = environment or {}


class BaseHumanLayer(ABC):
    """Abstract base class for all Human Layers."""

    def __init__(
        self,
        layer_id: LayerID,
        veto_power: VetoLevel,
        llm_client: BaseLLMClient,
    ):
        self.layer_id = layer_id
        self.veto_power = veto_power
        self.llm_client = llm_client

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable layer name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """What this layer validates."""
        pass

    @abstractmethod
    async def analyze(self, action: ActionContext) -> List[Finding]:
        """Analyze action and return findings."""
        pass

    async def execute(self, action: ActionContext, run_number: int = 1) -> LayerResult:
        """Execute layer validation and return result.

        This method handles the standard execution flow:
        1. Mark as running
        2. Call analyze() (implemented by subclass)
        3. Determine veto level based on findings
        4. Return structured result
        """
        start_time = datetime.utcnow()
        try:
            findings = await self.analyze(action)
            status = self._determine_status(findings)
            veto = self._determine_veto(findings)

            return LayerResult(
                layer_id=self.layer_id,
                status=status,
                veto_level=veto,
                findings=findings,
                execution_time_ms=self._elapsed_ms(start_time),
                run_number=run_number,
            )
        except Exception as e:
            return LayerResult(
                layer_id=self.layer_id,
                status=LayerStatus.ERROR,
                veto_level=VetoLevel.NONE,
                findings=[],
                execution_time_ms=self._elapsed_ms(start_time),
                run_number=run_number,
                error=str(e),
            )

    def _determine_status(self, findings: List[Finding]) -> LayerStatus:
        """Determine status based on findings."""
        if not findings:
            return LayerStatus.PASS

        severities = [f.severity for f in findings]
        if Severity.CRITICAL in severities:
            return LayerStatus.FAIL
        if Severity.HIGH in severities:
            return LayerStatus.FAIL
        if Severity.MEDIUM in severities:
            return LayerStatus.WARN
        return LayerStatus.PASS

    def _determine_veto(self, findings: List[Finding]) -> VetoLevel:
        """Determine veto level based on findings and layer's veto power."""
        if not findings:
            return VetoLevel.NONE

        has_critical = any(f.severity == Severity.CRITICAL for f in findings)
        has_high = any(f.severity == Severity.HIGH for f in findings)

        if has_critical:
            return self.veto_power  # Use layer's max veto power
        if has_high and self.veto_power >= VetoLevel.MEDIUM:
            return VetoLevel.MEDIUM
        if has_high:
            return VetoLevel.WEAK
        return VetoLevel.NONE

    def _elapsed_ms(self, start: datetime) -> int:
        """Calculate elapsed milliseconds."""
        return int((datetime.utcnow() - start).total_seconds() * 1000)
```

### Prompt Manager (layers/prompts.py)

```python
"""Prompt templates for Human Layers."""
from typing import List, Dict, Any
from jinja2 import Template

from ..models import Finding, LayerID, Severity
from .base import ActionContext

# System prompts by layer
SYSTEM_PROMPTS: Dict[LayerID, str] = {
    LayerID.HL1: """You are a UI/UX expert reviewing AI agent actions.
Focus on: user experience, usability, accessibility, clarity.
Be critical but fair. Identify issues that would frustrate users.""",

    LayerID.HL2: """You are a functionality expert reviewing AI agent actions.
Focus on: correctness, completeness, edge cases, error handling.
Verify the action does what it claims to do.""",

    LayerID.HL3: """You are an edge case specialist reviewing AI agent actions.
Focus on: boundary conditions, unusual inputs, race conditions, failure modes.
Think adversarially - what could go wrong?""",

    LayerID.HL4: """You are a security expert reviewing AI agent actions.
Focus on: vulnerabilities, injection attacks, data exposure, authentication.
This layer has STRONG veto power - flag anything concerning.""",

    LayerID.HL5: """You are a performance expert reviewing AI agent actions.
Focus on: efficiency, resource usage, scalability, response times.
Identify potential bottlenecks and performance issues.""",

    LayerID.HL6: """You are a compliance expert reviewing AI agent actions.
Focus on: regulations, policies, standards, legal requirements.
This layer has STRONG veto power - compliance is non-negotiable.""",

    LayerID.HL7: """You are the final human reviewer for AI agent actions.
This is the last check before execution. Be thorough.
This layer has STRONG veto power - if in doubt, veto.""",
}

ANALYSIS_TEMPLATE = Template("""
Analyze this AI agent action:

**Agent**: {{ action.agent_id }}
**Action Type**: {{ action.action_type }}
**Description**: {{ action.action_description }}

{% if action.code_diff %}
**Code Changes**:
```
{{ action.code_diff }}
```
{% endif %}

{% if action.affected_files %}
**Affected Files**:
{% for file in action.affected_files %}
- {{ file }}
{% endfor %}
{% endif %}

Respond with a JSON array of findings. Each finding must have:
- severity: "critical", "high", "medium", "low", or "info"
- title: Short title (max 200 chars)
- description: Detailed description
- suggestion: How to fix (optional)

Example response:
```json
[
  {
    "severity": "high",
    "title": "SQL Injection Risk",
    "description": "User input is directly concatenated into SQL query",
    "suggestion": "Use parameterized queries instead"
  }
]
```

If no issues found, respond with: []
""")


class LayerPromptManager:
    """Manages prompts for layer analysis."""

    def get_system_prompt(self, layer_id: LayerID) -> str:
        """Get system prompt for a layer."""
        return SYSTEM_PROMPTS.get(layer_id, "You are reviewing AI agent actions.")

    def get_analysis_prompt(self, action: ActionContext) -> str:
        """Generate analysis prompt for an action."""
        return ANALYSIS_TEMPLATE.render(action=action)

    def parse_findings(self, response_text: str, layer_id: LayerID) -> List[Finding]:
        """Parse LLM response into findings."""
        import json
        import uuid

        # Extract JSON from response
        text = response_text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        try:
            data = json.loads(text)
            if not isinstance(data, list):
                data = [data]

            findings = []
            for item in data:
                findings.append(Finding(
                    id=str(uuid.uuid4()),
                    layer_id=layer_id,
                    severity=Severity(item["severity"]),
                    title=item["title"][:200],
                    description=item["description"],
                    suggestion=item.get("suggestion"),
                ))
            return findings
        except (json.JSONDecodeError, KeyError, ValueError):
            return []  # No valid findings if parsing fails
```

### Registry (layers/registry.py)

```python
"""Layer registry for managing Human Layers."""
from typing import Dict, List, Optional

from ..models import LayerID
from .base import BaseHumanLayer

class LayerRegistry:
    """Registry for Human Layer instances."""

    def __init__(self):
        self._layers: Dict[LayerID, BaseHumanLayer] = {}

    def register(self, layer: BaseHumanLayer) -> None:
        """Register a layer."""
        self._layers[layer.layer_id] = layer

    def get(self, layer_id: LayerID) -> Optional[BaseHumanLayer]:
        """Get a layer by ID."""
        return self._layers.get(layer_id)

    def all(self) -> List[BaseHumanLayer]:
        """Get all registered layers in order."""
        return [
            self._layers[lid]
            for lid in LayerID
            if lid in self._layers
        ]

    def count(self) -> int:
        """Number of registered layers."""
        return len(self._layers)

    def clear(self) -> None:
        """Remove all registered layers."""
        self._layers.clear()


# Global registry instance
_registry = LayerRegistry()


def get_registry() -> LayerRegistry:
    """Get the global layer registry."""
    return _registry
```

---

## INTENT MANIFEST

```yaml
INTENT_MANIFEST:
  RF:
    - RF-001: "BaseHumanLayer define interface para todas as 7 layers"
    - RF-002: "execute() retorna LayerResult padronizado"
    - RF-003: "analyze() é implementado por cada layer específica"
    - RF-004: "LayerPromptManager gerencia prompts por layer"
    - RF-005: "LayerRegistry permite registrar e buscar layers"

  INV:
    - INV-001: "execute() nunca levanta exceção - erros vão em LayerResult.error"
    - INV-002: "Veto level nunca excede veto_power da layer"
    - INV-003: "Finding.severity determina status (CRITICAL/HIGH = FAIL)"
    - INV-004: "System prompts são read-only"

  EDGE:
    - EDGE-001: "analyze() levanta exceção → LayerResult com status=ERROR"
    - EDGE-002: "LLM retorna JSON inválido → findings=[]"
    - EDGE-003: "Layer não registrada → registry.get() retorna None"
```

---

## GATES

```yaml
gates:
  G0_FILES_EXIST:
    description: "Arquivos de layers existem"
    validation: |
      ls src/hl_mcp/layers/base.py
      ls src/hl_mcp/layers/prompts.py
      ls src/hl_mcp/layers/registry.py

  G1_IMPORTS_WORK:
    description: "Imports funcionam"
    validation: |
      python -c "from hl_mcp.layers import BaseHumanLayer, LayerPromptManager, get_registry"

  G2_TESTS_PASS:
    description: "Testes passam"
    validation: "pytest tests/test_layers/ -v"

  G3_COVERAGE:
    description: "Coverage >= 90%"
    validation: "pytest tests/test_layers/ --cov=src/hl_mcp/layers --cov-fail-under=90"

  G4_VETO_INVARIANT:
    description: "Veto nunca excede layer power"
    validation: |
      python -c "
      from hl_mcp.layers.base import BaseHumanLayer
      from hl_mcp.models import VetoLevel, Finding, Severity, LayerID
      # Verified by _determine_veto implementation
      "
```

---

## DECISION TREE

```
START S05
│
├─> Criar ActionContext (layers/base.py)
│   └─> agent_id, action_type, action_description, code_diff, etc.
│
├─> Criar BaseHumanLayer (layers/base.py)
│   ├─> layer_id, veto_power, llm_client
│   ├─> execute() → LayerResult
│   └─> analyze() (abstract)
│
├─> Criar LayerPromptManager (layers/prompts.py)
│   ├─> System prompts para 7 layers
│   ├─> Analysis template
│   └─> parse_findings()
│
├─> Criar LayerRegistry (layers/registry.py)
│   ├─> register(), get(), all()
│   └─> Global instance
│
└─> VALIDAR GATES
    ├─> G0: Arquivos existem
    ├─> G1: Imports funcionam
    ├─> G2: Testes passam
    ├─> G3: Coverage >= 90%
    └─> G4: Veto invariant
```

---

## REFERÊNCIA

Para detalhes completos, consulte:
- `./S03_CONTEXT.md` - LLM Client interface
- `./S06_CONTEXT.md` - HL-1 UX Layer
- `../MASTER_REQUIREMENTS_MAP.md` - PARTE 4: Layers
