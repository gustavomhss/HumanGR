# S12 - layer-hl7-final-review | Context Pack v1.0

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
  id: S12
  name: layer-hl7-final-review
  title: "Layer HL-7: Final Human Review"
  wave: W1-CoreEngine
  priority: P0-CRITICAL
  type: implementation

objective: "Implementar Human Layer 7 - Final Human Review (STRONG veto)"

source_block: "../MASTER_REQUIREMENTS_MAP.md#PARTE-4-LAYERS"

dependencies:
  - S05  # Layer Base

deliverables:
  - src/hl_mcp/layers/hl7_final_review.py
  - tests/test_layers/test_hl7_final_review.py
```

---

## LAYER HL-7 SPECIFICATION

```yaml
layer:
  id: HL7
  name: "Final Human Review"
  veto_power: STRONG  # CRITICAL - last line of defense
  focus_areas:
    - "Holistic review"
    - "Cross-layer concerns"
    - "Common sense check"
    - "Risk assessment"
    - "Final approval gate"

  what_it_catches:
    - "Issues that slipped through other layers"
    - "Combinations of minor issues that become major"
    - "Context-specific concerns"
    - "Gut-feel red flags"
    - "Things that 'just don't seem right'"

  philosophy: |
    This is the final check before execution.
    If in doubt, VETO.
    It's better to be cautious than to let a problem through.
```

---

## IMPLEMENTATION SPEC

### HL-7 Final Review Layer (layers/hl7_final_review.py)

```python
"""Human Layer 7: Final Human Review.

STRONG VETO POWER - Last line of defense.
When in doubt, VETO.
"""
from typing import List

from ..models import LayerID, VetoLevel, Finding
from ..llm import BaseLLMClient
from .base import BaseHumanLayer, ActionContext
from .prompts import LayerPromptManager


class HL7FinalReviewLayer(BaseHumanLayer):
    """Human Layer 7: Final Human Review.

    Focus: Holistic review, cross-layer concerns, common sense.
    Veto Power: STRONG (last line of defense)

    This is the final layer before execution.
    When in doubt, VETO.
    """

    def __init__(self, llm_client: BaseLLMClient):
        super().__init__(
            layer_id=LayerID.HL7,
            veto_power=VetoLevel.STRONG,
            llm_client=llm_client,
        )
        self._prompt_manager = LayerPromptManager()

    @property
    def name(self) -> str:
        return "Final Human Review"

    @property
    def description(self) -> str:
        return "Final holistic review - last line of defense"

    async def analyze(self, action: ActionContext) -> List[Finding]:
        """Final holistic review of the action."""
        system_prompt = self._prompt_manager.get_system_prompt(self.layer_id)
        analysis_prompt = self._build_final_review_prompt(action)

        response = await self.llm_client.generate(
            prompt=analysis_prompt,
            system_prompt=system_prompt,
        )

        return self._prompt_manager.parse_findings(
            response.content,
            self.layer_id
        )

    def _build_final_review_prompt(self, action: ActionContext) -> str:
        """Build final-review-specific analysis prompt."""
        base_prompt = self._prompt_manager.get_analysis_prompt(action)

        final_review_specifics = """

**FINAL HUMAN REVIEW - STRONG VETO POWER**

This is the LAST CHECK before execution.
When in doubt, VETO.

**Holistic Review Checklist**:

1. **Overall Assessment**
   - Does this action make sense?
   - Is the purpose clear and legitimate?
   - Would you be comfortable explaining this to a user?

2. **Risk Assessment**
   - What's the worst that could happen?
   - Is this action reversible?
   - What's the blast radius if something goes wrong?

3. **Common Sense Check**
   - Does anything feel "off"?
   - Are there any red flags?
   - Would you let this run at 3 AM unattended?

4. **Cross-Cutting Concerns**
   - Are there combinations of minor issues that become major?
   - Does this action conflict with other parts of the system?
   - Are there hidden dependencies?

5. **Context Check**
   - Is this action appropriate for the current context?
   - Is the timing appropriate?
   - Are all prerequisites met?

6. **Human Judgment**
   - Would a senior engineer approve this?
   - Would you sign off on this yourself?
   - What would an auditor think?

**Final Decision Guidance**:

- If everything looks good: Return empty array []
- If minor concerns: Return with severity "low" or "info"
- If notable concerns: Return with severity "medium"
- If significant concerns: Return with severity "high"
- If this should NOT proceed: Return with severity "critical"

**Golden Rule**:
If you have ANY doubt, flag it.
It's better to be cautious than to let a problem through.
This is the last line of defense.

WHEN IN DOUBT, VETO.
"""
        return base_prompt + final_review_specifics


def create_hl7_layer(llm_client: BaseLLMClient) -> HL7FinalReviewLayer:
    """Factory function to create HL-7 layer."""
    return HL7FinalReviewLayer(llm_client)
```

---

## INTENT MANIFEST

```yaml
INTENT_MANIFEST:
  RF:
    - RF-001: "HL7 é a última linha de defesa"
    - RF-002: "Veto power é STRONG"
    - RF-003: "Faz review holístico cross-layer"
    - RF-004: "Aplica common sense check"
    - RF-005: "Filosofia: When in doubt, VETO"

  INV:
    - INV-001: "layer_id sempre é LayerID.HL7"
    - INV-002: "veto_power sempre é VetoLevel.STRONG"
    - INV-003: "É executada por último"
    - INV-004: "Pode vetar por 'gut feeling'"

  EDGE:
    - EDGE-001: "Todas as outras layers passaram → still can veto"
    - EDGE-002: "Minor issues em múltiplas layers → can escalate"
    - EDGE-003: "Context suggests risk → can veto"
```

---

## GATES

```yaml
gates:
  G0_FILES_EXIST:
    description: "Arquivo de HL-7 existe"
    validation: "ls src/hl_mcp/layers/hl7_final_review.py"

  G1_IMPORTS_WORK:
    description: "Import funciona"
    validation: |
      python -c "from hl_mcp.layers import HL7FinalReviewLayer, create_hl7_layer"

  G2_TESTS_PASS:
    description: "Testes passam"
    validation: "pytest tests/test_layers/test_hl7_final_review.py -v"

  G3_STRONG_VETO:
    description: "STRONG veto power verificado"
    validation: |
      python -c "
      from hl_mcp.layers.hl7_final_review import HL7FinalReviewLayer
      from hl_mcp.models import VetoLevel
      from unittest.mock import MagicMock

      layer = HL7FinalReviewLayer(MagicMock())
      assert layer.veto_power == VetoLevel.STRONG, 'HL7 must have STRONG veto'
      "
```

---

## LAYER SUMMARY (ALL 7 LAYERS)

```
┌───────┬─────────────────────┬───────────┬─────────────────────────┐
│ Layer │ Name                │ Veto      │ Focus                   │
├───────┼─────────────────────┼───────────┼─────────────────────────┤
│ HL-1  │ UX & Usability      │ WEAK      │ User experience         │
│ HL-2  │ Functionality       │ MEDIUM    │ Correctness             │
│ HL-3  │ Edge Cases          │ MEDIUM    │ Boundaries              │
│ HL-4  │ Security            │ STRONG    │ Vulnerabilities         │
│ HL-5  │ Performance         │ MEDIUM    │ Efficiency              │
│ HL-6  │ Compliance          │ STRONG    │ Regulations             │
│ HL-7  │ Final Review        │ STRONG    │ Last line of defense    │
└───────┴─────────────────────┴───────────┴─────────────────────────┘
```

---

## REFERÊNCIA

- `./S05_CONTEXT.md` - Layer Base
- `./S11_CONTEXT.md` - HL-6 Compliance
- `./S13_CONTEXT.md` - Consensus Engine
