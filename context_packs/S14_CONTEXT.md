# S14 - veto-gate | Context Pack v1.0

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
  id: S14
  name: veto-gate
  title: "Veto Gate"
  wave: W1-CoreEngine
  priority: P0-CRITICAL
  type: implementation

objective: "Implementar gate de veto final e engine de decisão"

source_block: "../MASTER_REQUIREMENTS_MAP.md#PARTE-5-CONSENSUS"

dependencies:
  - S13  # Consensus Engine

deliverables:
  - src/hl_mcp/core/veto.py
  - src/hl_mcp/core/decision.py
  - src/hl_mcp/core/runner.py
  - tests/test_core/test_veto.py
  - tests/test_core/test_decision.py
```

---

## VETO SYSTEM

```yaml
veto_levels:
  NONE:
    description: "No veto - action can proceed"
    blocks: nothing

  WEAK:
    description: "Advisory warning"
    blocks: nothing
    action: "Log warning, continue"

  MEDIUM:
    description: "Significant concern"
    blocks: "merge/deploy without override"
    action: "Require explicit approval"

  STRONG:
    description: "Critical issue"
    blocks: "all progression"
    action: "MUST be resolved"

final_decision_rules:
  - "STRONG veto from ANY layer → REJECTED"
  - "MEDIUM veto from 2+ layers → REJECTED"
  - "Only WEAK/NONE vetos → APPROVED"
  - "Single MEDIUM veto → NEEDS_REVIEW"
```

---

## IMPLEMENTATION SPEC

### Veto Gate (core/veto.py)

```python
"""Veto Gate - Final veto decision logic."""
from typing import List
from enum import Enum

from ..models import VetoLevel
from .consensus import ConsensusResult


class FinalDecision(str, Enum):
    """Final decision for an action."""
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs_review"


class VetoGate:
    """Gate that makes final veto decision based on all layer results.

    Decision Rules:
    - STRONG veto from ANY layer → REJECTED
    - MEDIUM veto from 2+ layers → REJECTED
    - Only WEAK/NONE vetos → APPROVED
    - Single MEDIUM veto → NEEDS_REVIEW
    """

    def evaluate(
        self,
        layer_results: List[ConsensusResult],
    ) -> FinalDecision:
        """Evaluate all layer results and make final decision.

        Args:
            layer_results: Consensus results from all 7 layers

        Returns:
            FinalDecision (APPROVED, REJECTED, or NEEDS_REVIEW)
        """
        veto_levels = [r.veto_level for r in layer_results]

        # Rule 1: STRONG veto from ANY layer → REJECTED
        if VetoLevel.STRONG in veto_levels:
            return FinalDecision.REJECTED

        # Rule 2: MEDIUM veto from 2+ layers → REJECTED
        medium_count = sum(1 for v in veto_levels if v == VetoLevel.MEDIUM)
        if medium_count >= 2:
            return FinalDecision.REJECTED

        # Rule 3: Single MEDIUM veto → NEEDS_REVIEW
        if medium_count == 1:
            return FinalDecision.NEEDS_REVIEW

        # Rule 4: Only WEAK/NONE → APPROVED
        return FinalDecision.APPROVED

    def get_blocking_reasons(
        self,
        layer_results: List[ConsensusResult],
    ) -> List[str]:
        """Get list of reasons for rejection/review.

        Args:
            layer_results: Consensus results from all layers

        Returns:
            List of human-readable blocking reasons
        """
        reasons = []
        for result in layer_results:
            if result.veto_level == VetoLevel.STRONG:
                reasons.append(
                    f"STRONG veto from layer {result.individual_results[0].layer_id.value}"
                )
            elif result.veto_level == VetoLevel.MEDIUM:
                reasons.append(
                    f"MEDIUM veto from layer {result.individual_results[0].layer_id.value}"
                )
        return reasons


def create_veto_gate() -> VetoGate:
    """Factory function to create veto gate."""
    return VetoGate()
```

### Decision Engine (core/decision.py)

```python
"""Decision Engine - Aggregates layer results into final decision."""
from typing import List, Optional
from datetime import datetime

from ..models import HumanLayerReport, ReportMetadata, LayerResult, VetoLevel
from .consensus import ConsensusEngine, ConsensusResult
from .veto import VetoGate, FinalDecision
from ..layers.base import ActionContext


class DecisionEngine:
    """Engine that orchestrates layers, consensus, and veto.

    This is the main entry point for Human Layer validation.
    """

    def __init__(self):
        self._consensus_engine = ConsensusEngine()
        self._veto_gate = VetoGate()

    def make_decision(
        self,
        layer_results_by_layer: dict[str, List[LayerResult]],
        action: ActionContext,
        report_id: str,
    ) -> HumanLayerReport:
        """Make final decision from all layer results.

        Args:
            layer_results_by_layer: Dict mapping layer_id to list of 3 runs
            action: The action being validated
            report_id: Unique ID for this report

        Returns:
            Complete HumanLayerReport with decision
        """
        start_time = datetime.utcnow()

        # Calculate consensus for each layer
        consensus_results: List[ConsensusResult] = []
        for layer_id, results in layer_results_by_layer.items():
            consensus = self._consensus_engine.calculate_consensus(results)
            consensus_results.append(consensus)

        # Make veto decision
        decision = self._veto_gate.evaluate(consensus_results)
        final_veto = self._calculate_final_veto(consensus_results)

        # Build report
        return HumanLayerReport(
            id=report_id,
            metadata=ReportMetadata(
                agent_id=action.agent_id,
                action_type=action.action_type,
                action_description=action.action_description,
            ),
            layer_results=[
                self._consensus_to_layer_result(c) for c in consensus_results
            ],
            final_decision=decision.value,
            final_veto_level=final_veto,
            created_at=start_time,
            completed_at=datetime.utcnow(),
            total_execution_time_ms=self._elapsed_ms(start_time),
        )

    def _calculate_final_veto(
        self,
        consensus_results: List[ConsensusResult],
    ) -> VetoLevel:
        """Calculate the final (highest) veto level."""
        veto_levels = [r.veto_level for r in consensus_results]
        if not veto_levels:
            return VetoLevel.NONE
        return max(veto_levels, key=lambda v: list(VetoLevel).index(v))

    def _consensus_to_layer_result(
        self,
        consensus: ConsensusResult,
    ) -> LayerResult:
        """Convert consensus result to single layer result for report."""
        # Use first result as base, override with consensus values
        base = consensus.individual_results[0]
        return LayerResult(
            layer_id=base.layer_id,
            status=consensus.status,
            veto_level=consensus.veto_level,
            findings=consensus.merged_findings,
            execution_time_ms=sum(r.execution_time_ms for r in consensus.individual_results),
            run_number=0,  # 0 indicates merged result
        )

    def _elapsed_ms(self, start: datetime) -> int:
        """Calculate elapsed milliseconds."""
        return int((datetime.utcnow() - start).total_seconds() * 1000)


def create_decision_engine() -> DecisionEngine:
    """Factory function to create decision engine."""
    return DecisionEngine()
```

---

## INTENT MANIFEST

```yaml
INTENT_MANIFEST:
  RF:
    - RF-001: "VetoGate aplica regras de decisão final"
    - RF-002: "STRONG veto de qualquer layer → REJECTED"
    - RF-003: "MEDIUM de 2+ layers → REJECTED"
    - RF-004: "DecisionEngine orquestra consensus + veto"
    - RF-005: "Produz HumanLayerReport completo"

  INV:
    - INV-001: "STRONG veto SEMPRE bloqueia"
    - INV-002: "Decisão é determinística (mesmos inputs → mesmo output)"
    - INV-003: "Blocking reasons são human-readable"
    - INV-004: "Report inclui todos os layer results"

  EDGE:
    - EDGE-001: "0 layers → APPROVED (nothing to block)"
    - EDGE-002: "Todos NONE → APPROVED"
    - EDGE-003: "7 STRONG vetos → REJECTED (one is enough)"
```

---

## GATES

```yaml
gates:
  G0_FILES_EXIST:
    description: "Arquivos existem"
    validation: |
      ls src/hl_mcp/core/veto.py
      ls src/hl_mcp/core/decision.py

  G1_IMPORTS_WORK:
    description: "Imports funcionam"
    validation: |
      python -c "from hl_mcp.core import VetoGate, DecisionEngine, FinalDecision"

  G2_TESTS_PASS:
    description: "Testes passam"
    validation: "pytest tests/test_core/ -v"

  G3_STRONG_BLOCKS:
    description: "STRONG veto always blocks"
    validation: |
      python -c "
      from hl_mcp.core.veto import VetoGate, FinalDecision
      from hl_mcp.core.consensus import ConsensusResult
      from hl_mcp.models import VetoLevel, LayerStatus, LayerResult, LayerID

      gate = VetoGate()

      # Create mock results with STRONG veto
      mock_layer_result = LayerResult(layer_id=LayerID.HL4, status=LayerStatus.FAIL)
      result_with_strong = ConsensusResult(
          status=LayerStatus.FAIL,
          veto_level=VetoLevel.STRONG,
          merged_findings=[],
          agreement_ratio=1.0,
          individual_results=[mock_layer_result, mock_layer_result, mock_layer_result]
      )

      decision = gate.evaluate([result_with_strong])
      assert decision == FinalDecision.REJECTED, 'STRONG veto must reject'
      "
```

---

## REFERÊNCIA

- `./S13_CONTEXT.md` - Consensus Engine
- `./S15_CONTEXT.md` - MCP Server Base
- `../MASTER_REQUIREMENTS_MAP.md` - PARTE 5: Consensus
