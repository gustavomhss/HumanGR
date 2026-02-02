# S08 - layer-hl3-edge-cases | Context Pack v1.0

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
  id: S08
  name: layer-hl3-edge-cases
  title: "Layer HL-3: Edge Cases"
  wave: W1-CoreEngine
  priority: P0-CRITICAL
  type: implementation

objective: "Implementar Human Layer 3 - Edge Cases Analysis"

source_block: "../MASTER_REQUIREMENTS_MAP.md#PARTE-4-LAYERS"

dependencies:
  - S05  # Layer Base

deliverables:
  - src/hl_mcp/layers/hl3_edge_cases.py
  - tests/test_layers/test_hl3_edge_cases.py
```

---

## LAYER HL-3 SPECIFICATION

```yaml
layer:
  id: HL3
  name: "Edge Cases"
  veto_power: MEDIUM
  focus_areas:
    - "Boundary conditions"
    - "Unusual inputs"
    - "Race conditions"
    - "Failure modes"
    - "Empty/null handling"
    - "Overflow/underflow"

  what_it_catches:
    - "Off-by-one errors"
    - "Null pointer issues"
    - "Empty collection handling"
    - "Concurrent access problems"
    - "Resource exhaustion"
    - "Timeout scenarios"

  adversarial_thinking:
    - "What if the input is empty?"
    - "What if the input is extremely large?"
    - "What if two users do this simultaneously?"
    - "What if the network fails mid-operation?"
    - "What if the disk is full?"
```

---

## IMPLEMENTATION SPEC

### HL-3 Edge Cases Layer (layers/hl3_edge_cases.py)

```python
"""Human Layer 3: Edge Cases Analysis."""
from typing import List

from ..models import LayerID, VetoLevel, Finding
from ..llm import BaseLLMClient
from .base import BaseHumanLayer, ActionContext
from .prompts import LayerPromptManager


class HL3EdgeCasesLayer(BaseHumanLayer):
    """Human Layer 3: Edge Cases Analysis.

    Focus: Boundary conditions, unusual inputs, failure modes.
    Veto Power: MEDIUM (can block for unhandled edge cases)
    """

    def __init__(self, llm_client: BaseLLMClient):
        super().__init__(
            layer_id=LayerID.HL3,
            veto_power=VetoLevel.MEDIUM,
            llm_client=llm_client,
        )
        self._prompt_manager = LayerPromptManager()

    @property
    def name(self) -> str:
        return "Edge Cases"

    @property
    def description(self) -> str:
        return "Analyzes boundary conditions and unusual scenarios"

    async def analyze(self, action: ActionContext) -> List[Finding]:
        """Analyze action for edge case issues.

        Thinks adversarially about:
        - Boundary conditions
        - Unusual inputs
        - Race conditions
        - Failure modes
        """
        system_prompt = self._prompt_manager.get_system_prompt(self.layer_id)
        analysis_prompt = self._build_edge_cases_prompt(action)

        response = await self.llm_client.generate(
            prompt=analysis_prompt,
            system_prompt=system_prompt,
        )

        return self._prompt_manager.parse_findings(
            response.content,
            self.layer_id
        )

    def _build_edge_cases_prompt(self, action: ActionContext) -> str:
        """Build edge-cases-specific analysis prompt."""
        base_prompt = self._prompt_manager.get_analysis_prompt(action)

        edge_cases_specifics = """

**Edge Cases Review - Think Adversarially**:

For each aspect, ask "What if...?":

1. **Empty/Null Inputs**
   - What if the input is empty?
   - What if the input is null/undefined?
   - What if a required field is missing?

2. **Boundary Values**
   - What if the value is 0? -1? MAX_INT?
   - What if the string is empty? 1 char? 10MB?
   - What if the list has 0 items? 1 item? 1 million?

3. **Concurrent Access**
   - What if two users do this at the same time?
   - What if this runs while another operation is in progress?
   - Are there race conditions?

4. **Resource Exhaustion**
   - What if memory runs out?
   - What if disk is full?
   - What if the connection pool is exhausted?

5. **Network Failures**
   - What if the network fails mid-operation?
   - What if a timeout occurs?
   - What if the response is corrupted?

6. **Time-Related Issues**
   - What happens at midnight?
   - What about daylight saving time?
   - What if the operation takes hours?

7. **Unicode and Encoding**
   - What if input contains emojis?
   - What about right-to-left text?
   - What if encoding is wrong?

Focus on findings severity:
- **critical**: Crash or data loss on edge case
- **high**: Significant failure on edge case
- **medium**: Unexpected behavior on edge case
- **low**: Minor edge case not handled
- **info**: Edge case worth documenting
"""
        return base_prompt + edge_cases_specifics


def create_hl3_layer(llm_client: BaseLLMClient) -> HL3EdgeCasesLayer:
    """Factory function to create HL-3 layer."""
    return HL3EdgeCasesLayer(llm_client)
```

---

## INTENT MANIFEST

```yaml
INTENT_MANIFEST:
  RF:
    - RF-001: "HL3 analisa edge cases e boundary conditions"
    - RF-002: "Veto power é MEDIUM"
    - RF-003: "Pensa adversarialmente (What if...?)"
    - RF-004: "Cobre null, boundary, concurrent, resources, network, time"
    - RF-005: "Factory function para criação padronizada"

  INV:
    - INV-001: "layer_id sempre é LayerID.HL3"
    - INV-002: "veto_power sempre é VetoLevel.MEDIUM"
    - INV-003: "Prompt inclui adversarial thinking"
    - INV-004: "Cobre pelo menos 7 categorias de edge cases"

  EDGE:
    - EDGE-001: "Action sem código → edge cases conceituais"
    - EDGE-002: "Action read-only → menos edge cases"
    - EDGE-003: "Action de configuração → edge cases de valores"
```

---

## GATES

```yaml
gates:
  G0_FILES_EXIST:
    description: "Arquivo de HL-3 existe"
    validation: "ls src/hl_mcp/layers/hl3_edge_cases.py"

  G1_IMPORTS_WORK:
    description: "Import funciona"
    validation: |
      python -c "from hl_mcp.layers import HL3EdgeCasesLayer, create_hl3_layer"

  G2_TESTS_PASS:
    description: "Testes passam"
    validation: "pytest tests/test_layers/test_hl3_edge_cases.py -v"

  G3_COVERAGE:
    description: "Coverage >= 90%"
    validation: "pytest tests/test_layers/test_hl3_edge_cases.py --cov=src/hl_mcp/layers/hl3_edge_cases --cov-fail-under=90"
```

---

## REFERÊNCIA

- `./S05_CONTEXT.md` - Layer Base
- `./S07_CONTEXT.md` - HL-2 Functionality
- `./S09_CONTEXT.md` - HL-4 Security
