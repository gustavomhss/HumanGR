# S07 - layer-hl2-functionality | Context Pack v1.0

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
  id: S07
  name: layer-hl2-functionality
  title: "Layer HL-2: Functionality"
  wave: W1-CoreEngine
  priority: P0-CRITICAL
  type: implementation

objective: "Implementar Human Layer 2 - Functionality Review"

source_block: "../MASTER_REQUIREMENTS_MAP.md#PARTE-4-LAYERS"

dependencies:
  - S05  # Layer Base

deliverables:
  - src/hl_mcp/layers/hl2_functionality.py
  - tests/test_layers/test_hl2_functionality.py
```

---

## LAYER HL-2 SPECIFICATION

```yaml
layer:
  id: HL2
  name: "Functionality"
  veto_power: MEDIUM
  focus_areas:
    - "Correctness of implementation"
    - "Feature completeness"
    - "API contract adherence"
    - "Data integrity"
    - "Business logic accuracy"
    - "State management"

  what_it_catches:
    - "Features that don't work as specified"
    - "Missing functionality"
    - "Incorrect data transformations"
    - "API mismatches"
    - "Logic errors"
    - "State inconsistencies"

  veto_scenarios:
    MEDIUM:
      - "Feature doesn't work as specified"
      - "Missing critical functionality"
      - "Data corruption possible"
    WEAK:
      - "Minor functional issues"
      - "Edge case not handled"
    NONE:
      - "All functionality correct"
```

---

## IMPLEMENTATION SPEC

### HL-2 Functionality Layer (layers/hl2_functionality.py)

```python
"""Human Layer 2: Functionality Review."""
from typing import List

from ..models import LayerID, VetoLevel, Finding
from ..llm import BaseLLMClient
from .base import BaseHumanLayer, ActionContext
from .prompts import LayerPromptManager


class HL2FunctionalityLayer(BaseHumanLayer):
    """Human Layer 2: Functionality Review.

    Focus: Correctness, completeness, API contracts, data integrity.
    Veto Power: MEDIUM (can block merge for significant issues)
    """

    def __init__(self, llm_client: BaseLLMClient):
        super().__init__(
            layer_id=LayerID.HL2,
            veto_power=VetoLevel.MEDIUM,
            llm_client=llm_client,
        )
        self._prompt_manager = LayerPromptManager()

    @property
    def name(self) -> str:
        return "Functionality"

    @property
    def description(self) -> str:
        return "Reviews correctness, completeness, and API contracts"

    async def analyze(self, action: ActionContext) -> List[Finding]:
        """Analyze action for functionality issues.

        Checks for:
        - Implementation correctness
        - Feature completeness
        - API contract adherence
        - Data integrity
        - Business logic accuracy
        """
        system_prompt = self._prompt_manager.get_system_prompt(self.layer_id)
        analysis_prompt = self._build_functionality_prompt(action)

        response = await self.llm_client.generate(
            prompt=analysis_prompt,
            system_prompt=system_prompt,
        )

        return self._prompt_manager.parse_findings(
            response.content,
            self.layer_id
        )

    def _build_functionality_prompt(self, action: ActionContext) -> str:
        """Build functionality-specific analysis prompt."""
        base_prompt = self._prompt_manager.get_analysis_prompt(action)

        functionality_specifics = """

**Functionality Review Checklist**:

1. **Correctness**
   - Does the code do what it claims?
   - Are calculations/transformations correct?
   - Does it handle all specified requirements?

2. **Completeness**
   - Are all features implemented?
   - Any TODO comments that should be addressed?
   - Missing validation or error handling?

3. **API Contracts**
   - Do inputs match expected types/formats?
   - Are outputs in the correct format?
   - Are error responses appropriate?

4. **Data Integrity**
   - Can data be corrupted by this action?
   - Are transactions atomic where needed?
   - Is data validated before persistence?

5. **Business Logic**
   - Does the logic match business requirements?
   - Are edge cases in business rules handled?
   - Are calculations/formulas correct?

6. **State Management**
   - Is state updated correctly?
   - Are race conditions possible?
   - Is cleanup performed on failure?

Focus on findings severity:
- **critical**: Core functionality broken
- **high**: Significant feature doesn't work
- **medium**: Notable functional issue
- **low**: Minor functional concern
- **info**: Suggestion for improvement
"""
        return base_prompt + functionality_specifics


def create_hl2_layer(llm_client: BaseLLMClient) -> HL2FunctionalityLayer:
    """Factory function to create HL-2 layer."""
    return HL2FunctionalityLayer(llm_client)
```

---

## INTENT MANIFEST

```yaml
INTENT_MANIFEST:
  RF:
    - RF-001: "HL2 analisa funcionalidade e correção"
    - RF-002: "Veto power é MEDIUM (pode bloquear merge)"
    - RF-003: "Checklist inclui correctness, completeness, API, data, logic"
    - RF-004: "Verifica se features funcionam como especificado"
    - RF-005: "Factory function para criação padronizada"

  INV:
    - INV-001: "layer_id sempre é LayerID.HL2"
    - INV-002: "veto_power sempre é VetoLevel.MEDIUM"
    - INV-003: "Findings HIGH podem gerar MEDIUM veto"
    - INV-004: "Findings CRITICAL geram MEDIUM veto"

  EDGE:
    - EDGE-001: "Action sem código → análise textual apenas"
    - EDGE-002: "Code diff muito grande → análise parcial"
    - EDGE-003: "Action só de configuração → checklist adaptado"
```

---

## GATES

```yaml
gates:
  G0_FILES_EXIST:
    description: "Arquivo de HL-2 existe"
    validation: "ls src/hl_mcp/layers/hl2_functionality.py"

  G1_IMPORTS_WORK:
    description: "Import funciona"
    validation: |
      python -c "from hl_mcp.layers import HL2FunctionalityLayer, create_hl2_layer"

  G2_TESTS_PASS:
    description: "Testes passam"
    validation: "pytest tests/test_layers/test_hl2_functionality.py -v"

  G3_COVERAGE:
    description: "Coverage >= 90%"
    validation: "pytest tests/test_layers/test_hl2_functionality.py --cov=src/hl_mcp/layers/hl2_functionality --cov-fail-under=90"

  G4_LAYER_INVARIANTS:
    description: "Layer invariants"
    validation: |
      python -c "
      from hl_mcp.layers.hl2_functionality import HL2FunctionalityLayer
      from hl_mcp.models import LayerID, VetoLevel
      from unittest.mock import MagicMock

      layer = HL2FunctionalityLayer(MagicMock())
      assert layer.layer_id == LayerID.HL2
      assert layer.veto_power == VetoLevel.MEDIUM
      "
```

---

## REFERÊNCIA

- `./S05_CONTEXT.md` - Layer Base
- `./S06_CONTEXT.md` - HL-1 UX
- `./S08_CONTEXT.md` - HL-3 Edge Cases
