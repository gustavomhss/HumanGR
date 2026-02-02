# S10 - layer-hl5-performance | Context Pack v1.0

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
  id: S10
  name: layer-hl5-performance
  title: "Layer HL-5: Performance"
  wave: W1-CoreEngine
  priority: P0-CRITICAL
  type: implementation

objective: "Implementar Human Layer 5 - Performance Review"

source_block: "../MASTER_REQUIREMENTS_MAP.md#PARTE-4-LAYERS"

dependencies:
  - S05  # Layer Base

deliverables:
  - src/hl_mcp/layers/hl5_performance.py
  - tests/test_layers/test_hl5_performance.py
```

---

## LAYER HL-5 SPECIFICATION

```yaml
layer:
  id: HL5
  name: "Performance"
  veto_power: MEDIUM
  focus_areas:
    - "Time complexity"
    - "Space complexity"
    - "Database queries (N+1)"
    - "Memory leaks"
    - "Caching opportunities"
    - "Resource usage"

  what_it_catches:
    - "O(n²) where O(n) is possible"
    - "N+1 query problems"
    - "Missing indexes"
    - "Unbounded memory growth"
    - "Missing pagination"
    - "Synchronous blocking operations"
```

---

## IMPLEMENTATION SPEC

### HL-5 Performance Layer (layers/hl5_performance.py)

```python
"""Human Layer 5: Performance Review."""
from typing import List

from ..models import LayerID, VetoLevel, Finding
from ..llm import BaseLLMClient
from .base import BaseHumanLayer, ActionContext
from .prompts import LayerPromptManager


class HL5PerformanceLayer(BaseHumanLayer):
    """Human Layer 5: Performance Review.

    Focus: Efficiency, resource usage, scalability.
    Veto Power: MEDIUM (can block for significant performance issues)
    """

    def __init__(self, llm_client: BaseLLMClient):
        super().__init__(
            layer_id=LayerID.HL5,
            veto_power=VetoLevel.MEDIUM,
            llm_client=llm_client,
        )
        self._prompt_manager = LayerPromptManager()

    @property
    def name(self) -> str:
        return "Performance"

    @property
    def description(self) -> str:
        return "Reviews efficiency, scalability, and resource usage"

    async def analyze(self, action: ActionContext) -> List[Finding]:
        """Analyze action for performance issues."""
        system_prompt = self._prompt_manager.get_system_prompt(self.layer_id)
        analysis_prompt = self._build_performance_prompt(action)

        response = await self.llm_client.generate(
            prompt=analysis_prompt,
            system_prompt=system_prompt,
        )

        return self._prompt_manager.parse_findings(
            response.content,
            self.layer_id
        )

    def _build_performance_prompt(self, action: ActionContext) -> str:
        """Build performance-specific analysis prompt."""
        base_prompt = self._prompt_manager.get_analysis_prompt(action)

        performance_specifics = """

**Performance Review Checklist**:

1. **Time Complexity**
   - Is the algorithm efficient?
   - Can O(n²) be reduced to O(n log n) or O(n)?
   - Are there unnecessary nested loops?

2. **Space Complexity**
   - Is memory usage proportional to input?
   - Are large objects copied unnecessarily?
   - Is streaming possible instead of loading all at once?

3. **Database Performance**
   - N+1 query problems?
   - Missing indexes on filtered/sorted columns?
   - Unbounded queries (missing LIMIT)?
   - SELECT * instead of specific columns?

4. **Caching**
   - Can results be cached?
   - Is cache invalidation handled?
   - Are expensive computations memoized?

5. **I/O Operations**
   - Are I/O operations async where possible?
   - Is batching used for multiple operations?
   - Are connections pooled?

6. **Memory Management**
   - Potential memory leaks?
   - Large objects held longer than necessary?
   - Circular references?

7. **Scalability**
   - Does it scale with more users?
   - Does it scale with more data?
   - Is pagination implemented?

8. **Response Time**
   - Are slow operations async/background?
   - Is progress feedback provided for long operations?
   - Are timeouts configured?

Severity guidelines:
- **critical**: Performance makes feature unusable at scale
- **high**: Significant performance degradation
- **medium**: Notable performance concern
- **low**: Minor optimization opportunity
- **info**: Performance best practice
"""
        return base_prompt + performance_specifics


def create_hl5_layer(llm_client: BaseLLMClient) -> HL5PerformanceLayer:
    """Factory function to create HL-5 layer."""
    return HL5PerformanceLayer(llm_client)
```

---

## INTENT MANIFEST

```yaml
INTENT_MANIFEST:
  RF:
    - RF-001: "HL5 analisa eficiência e performance"
    - RF-002: "Veto power é MEDIUM"
    - RF-003: "Cobre time/space complexity, DB, caching, I/O"
    - RF-004: "Detecta N+1, memory leaks, missing indexes"
    - RF-005: "Factory function para criação padronizada"

  INV:
    - INV-001: "layer_id sempre é LayerID.HL5"
    - INV-002: "veto_power sempre é VetoLevel.MEDIUM"
    - INV-003: "Foco em scalability issues"
    - INV-004: "Não bloqueia por micro-optimizations"

  EDGE:
    - EDGE-001: "Action sem código → análise conceitual"
    - EDGE-002: "Action read-only → foco em queries"
    - EDGE-003: "Action de batch → foco em scalability"
```

---

## GATES

```yaml
gates:
  G0_FILES_EXIST:
    description: "Arquivo de HL-5 existe"
    validation: "ls src/hl_mcp/layers/hl5_performance.py"

  G1_IMPORTS_WORK:
    description: "Import funciona"
    validation: |
      python -c "from hl_mcp.layers import HL5PerformanceLayer, create_hl5_layer"

  G2_TESTS_PASS:
    description: "Testes passam"
    validation: "pytest tests/test_layers/test_hl5_performance.py -v"

  G3_COVERAGE:
    description: "Coverage >= 90%"
    validation: "pytest tests/test_layers/test_hl5_performance.py --cov=src/hl_mcp/layers/hl5_performance --cov-fail-under=90"
```

---

## REFERÊNCIA

- `./S05_CONTEXT.md` - Layer Base
- `./S09_CONTEXT.md` - HL-4 Security
- `./S11_CONTEXT.md` - HL-6 Compliance
