# S06 - layer-hl1-ux | Context Pack v1.0

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
  id: S06
  name: layer-hl1-ux
  title: "Layer HL-1: UX & Usability"
  wave: W1-CoreEngine
  priority: P0-CRITICAL
  type: implementation

objective: "Implementar Human Layer 1 - UX & Usability"

source_block: "../MASTER_REQUIREMENTS_MAP.md#PARTE-4-LAYERS"

dependencies:
  - S05  # Layer Base

deliverables:
  - src/hl_mcp/layers/hl1_ux.py
  - tests/test_layers/test_hl1_ux.py
```

---

## LAYER HL-1 SPECIFICATION

```yaml
layer:
  id: HL1
  name: "UX & Usability"
  veto_power: WEAK
  focus_areas:
    - "User interface clarity"
    - "Navigation and flow"
    - "Error messages quality"
    - "Accessibility basics"
    - "Response time perception"
    - "Cognitive load"

  what_it_catches:
    - "Confusing UI elements"
    - "Poor error messages"
    - "Accessibility issues (WCAG)"
    - "Inconsistent terminology"
    - "Missing confirmations for destructive actions"

  veto_scenarios:
    WEAK:
      - "Minor UX issues that don't block functionality"
      - "Aesthetic concerns"
      - "Optional improvements"
    NONE:
      - "No UX issues found"
```

---

## IMPLEMENTATION SPEC

### HL-1 UX Layer (layers/hl1_ux.py)

```python
"""Human Layer 1: UX & Usability."""
from typing import List

from ..models import LayerID, VetoLevel, Finding
from ..llm import BaseLLMClient
from .base import BaseHumanLayer, ActionContext
from .prompts import LayerPromptManager

class HL1UXLayer(BaseHumanLayer):
    """Human Layer 1: UX & Usability Review.

    Focus: User experience, usability, accessibility, clarity.
    Veto Power: WEAK (can add warnings, rarely blocks)
    """

    def __init__(self, llm_client: BaseLLMClient):
        super().__init__(
            layer_id=LayerID.HL1,
            veto_power=VetoLevel.WEAK,
            llm_client=llm_client,
        )
        self._prompt_manager = LayerPromptManager()

    @property
    def name(self) -> str:
        return "UX & Usability"

    @property
    def description(self) -> str:
        return "Reviews user experience, usability, and accessibility"

    async def analyze(self, action: ActionContext) -> List[Finding]:
        """Analyze action for UX issues.

        Checks for:
        - UI clarity and consistency
        - Error message quality
        - Accessibility (WCAG basics)
        - Navigation flow
        - Cognitive load
        """
        system_prompt = self._prompt_manager.get_system_prompt(self.layer_id)
        analysis_prompt = self._build_ux_prompt(action)

        response = await self.llm_client.generate(
            prompt=analysis_prompt,
            system_prompt=system_prompt,
        )

        return self._prompt_manager.parse_findings(
            response.content,
            self.layer_id
        )

    def _build_ux_prompt(self, action: ActionContext) -> str:
        """Build UX-specific analysis prompt."""
        base_prompt = self._prompt_manager.get_analysis_prompt(action)

        ux_specifics = """

**UX Review Checklist** (check all that apply):

1. **Clarity**
   - Is the purpose of this action clear?
   - Are labels and messages unambiguous?

2. **Error Handling**
   - Are error messages helpful?
   - Do errors provide recovery guidance?

3. **Accessibility (WCAG)**
   - Color contrast sufficient?
   - Alt text for images?
   - Keyboard navigation possible?

4. **Flow**
   - Is the action sequence logical?
   - Are there unnecessary steps?

5. **Feedback**
   - Does user get confirmation of actions?
   - Are loading states shown?

6. **Cognitive Load**
   - Is information overwhelming?
   - Are complex tasks broken into steps?

Focus on findings severity:
- **critical**: UX issue makes feature unusable
- **high**: Significant usability problem
- **medium**: Notable UX concern
- **low**: Minor improvement opportunity
- **info**: Suggestion for enhancement
"""
        return base_prompt + ux_specifics


def create_hl1_layer(llm_client: BaseLLMClient) -> HL1UXLayer:
    """Factory function to create HL-1 layer."""
    return HL1UXLayer(llm_client)
```

---

## INTENT MANIFEST

```yaml
INTENT_MANIFEST:
  RF:
    - RF-001: "HL1 analisa UX e usabilidade"
    - RF-002: "Veto power é WEAK (warnings, não bloqueia)"
    - RF-003: "Checklist inclui clarity, errors, a11y, flow, feedback"
    - RF-004: "Findings incluem suggestion para correção"
    - RF-005: "Factory function para criação padronizada"

  INV:
    - INV-001: "layer_id sempre é LayerID.HL1"
    - INV-002: "veto_power sempre é VetoLevel.WEAK"
    - INV-003: "analyze() nunca bloqueia indefinidamente"
    - INV-004: "WCAG checks são básicos (AA level)"

  EDGE:
    - EDGE-001: "Action sem UI → poucos/nenhum finding"
    - EDGE-002: "LLM timeout → LayerResult com error"
    - EDGE-003: "Action só com backend → checklist parcial"
```

---

## TEST SCENARIOS

```yaml
test_scenarios:
  test_clean_action:
    description: "Action com boa UX retorna 0 findings"
    input:
      action_type: "display_message"
      action_description: "Shows clear success message with dismiss button"
    expected:
      status: PASS
      findings: []

  test_bad_error_message:
    description: "Error message ruim é detectado"
    input:
      action_type: "show_error"
      action_description: "Shows 'Error 500' to user"
    expected:
      status: WARN
      findings:
        - severity: medium
          title: "Unhelpful error message"

  test_no_confirmation:
    description: "Ação destrutiva sem confirmação"
    input:
      action_type: "delete_file"
      action_description: "Deletes file immediately without confirmation"
    expected:
      status: WARN
      findings:
        - severity: high
          title: "Missing confirmation for destructive action"

  test_accessibility_issue:
    description: "Issue de acessibilidade básico"
    input:
      action_type: "add_image"
      action_description: "Adds image without alt text"
    expected:
      status: WARN
      findings:
        - severity: medium
          title: "Missing alt text"
```

---

## GATES

```yaml
gates:
  G0_FILES_EXIST:
    description: "Arquivo de HL-1 existe"
    validation: "ls src/hl_mcp/layers/hl1_ux.py"

  G1_IMPORTS_WORK:
    description: "Import funciona"
    validation: |
      python -c "from hl_mcp.layers import HL1UXLayer, create_hl1_layer"

  G2_TESTS_PASS:
    description: "Testes passam"
    validation: "pytest tests/test_layers/test_hl1_ux.py -v"

  G3_COVERAGE:
    description: "Coverage >= 90%"
    validation: "pytest tests/test_layers/test_hl1_ux.py --cov=src/hl_mcp/layers/hl1_ux --cov-fail-under=90"

  G4_LAYER_INVARIANTS:
    description: "Layer invariants"
    validation: |
      python -c "
      from hl_mcp.layers.hl1_ux import HL1UXLayer
      from hl_mcp.models import LayerID, VetoLevel
      from unittest.mock import MagicMock

      layer = HL1UXLayer(MagicMock())
      assert layer.layer_id == LayerID.HL1
      assert layer.veto_power == VetoLevel.WEAK
      assert layer.name == 'UX & Usability'
      "
```

---

## DECISION TREE

```
START S06
│
├─> Criar HL1UXLayer (layers/hl1_ux.py)
│   ├─> Herda de BaseHumanLayer
│   ├─> layer_id = LayerID.HL1
│   ├─> veto_power = VetoLevel.WEAK
│   └─> analyze() com UX checklist
│
├─> Implementar _build_ux_prompt()
│   └─> Checklist: clarity, errors, a11y, flow, feedback
│
├─> Criar factory create_hl1_layer()
│
├─> Criar testes
│   ├─> test_clean_action
│   ├─> test_bad_error_message
│   ├─> test_no_confirmation
│   └─> test_accessibility_issue
│
└─> VALIDAR GATES
    ├─> G0: Arquivo existe
    ├─> G1: Import funciona
    ├─> G2: Testes passam
    ├─> G3: Coverage >= 90%
    └─> G4: Invariants
```

---

## REFERÊNCIA

Para detalhes completos, consulte:
- `./S05_CONTEXT.md` - Layer Base
- `./S07_CONTEXT.md` - HL-2 Functionality
- `../MASTER_REQUIREMENTS_MAP.md` - PARTE 4: Layers
