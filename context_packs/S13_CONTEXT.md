# S13 - consensus-engine | Context Pack v1.0

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
  id: S13
  name: consensus-engine
  title: "Consensus Engine"
  wave: W1-CoreEngine
  priority: P0-CRITICAL
  type: implementation

objective: "Implementar motor de consenso 2/3 (Triple Redundancy)"

source_block: "../MASTER_REQUIREMENTS_MAP.md#PARTE-5-CONSENSUS"

dependencies:
  - S06  # HL-1
  - S07  # HL-2
  - S08  # HL-3
  - S09  # HL-4
  - S10  # HL-5
  - S11  # HL-6
  - S12  # HL-7

deliverables:
  - src/hl_mcp/core/__init__.py
  - src/hl_mcp/core/consensus.py
  - tests/test_core/test_consensus.py
```

---

## TRIPLE REDUNDANCY CONCEPT

```yaml
triple_redundancy:
  principle: "3 runs per layer, 2/3 consensus required"

  why_triple:
    - "Single run may have LLM hallucination"
    - "Two runs can tie (disagreement)"
    - "Three runs allow majority consensus"

  consensus_rules:
    - "2/3 agree on PASS → layer PASSES"
    - "2/3 agree on FAIL → layer FAILS"
    - "2/3 agree on veto level → that level applies"
    - "Disagreement on veto → use HIGHER level"

  example:
    run_1: "PASS, VETO=NONE"
    run_2: "FAIL, VETO=MEDIUM"
    run_3: "FAIL, VETO=MEDIUM"
    result: "FAIL, VETO=MEDIUM (2/3 consensus)"
```

---

## IMPLEMENTATION SPEC

### Consensus Engine (core/consensus.py)

```python
"""Consensus Engine - Triple Redundancy with 2/3 consensus."""
from typing import List, Tuple
from collections import Counter

from ..models import LayerResult, LayerStatus, VetoLevel, Finding


class ConsensusResult:
    """Result of consensus calculation."""

    def __init__(
        self,
        status: LayerStatus,
        veto_level: VetoLevel,
        merged_findings: List[Finding],
        agreement_ratio: float,
        individual_results: List[LayerResult],
    ):
        self.status = status
        self.veto_level = veto_level
        self.merged_findings = merged_findings
        self.agreement_ratio = agreement_ratio
        self.individual_results = individual_results

    @property
    def has_consensus(self) -> bool:
        """Check if consensus was reached (2/3 or better)."""
        return self.agreement_ratio >= 0.67


class ConsensusEngine:
    """Engine for calculating consensus from triple redundancy runs.

    Implements the 2/3 consensus rule:
    - 3 runs per layer
    - 2/3 agreement required for decision
    - Disagreements on veto → use higher level
    """

    def calculate_consensus(
        self,
        results: List[LayerResult],
    ) -> ConsensusResult:
        """Calculate consensus from multiple runs.

        Args:
            results: List of LayerResult from triple runs (expects 3)

        Returns:
            ConsensusResult with merged decision

        Raises:
            ValueError: If not exactly 3 results
        """
        if len(results) != 3:
            raise ValueError(f"Triple redundancy requires 3 runs, got {len(results)}")

        status = self._calculate_status_consensus(results)
        veto_level = self._calculate_veto_consensus(results)
        merged_findings = self._merge_findings(results)
        agreement = self._calculate_agreement(results)

        return ConsensusResult(
            status=status,
            veto_level=veto_level,
            merged_findings=merged_findings,
            agreement_ratio=agreement,
            individual_results=results,
        )

    def _calculate_status_consensus(
        self,
        results: List[LayerResult],
    ) -> LayerStatus:
        """Calculate status by 2/3 majority."""
        statuses = [r.status for r in results]
        status_counts = Counter(statuses)

        # Check for 2/3 majority
        for status, count in status_counts.most_common():
            if count >= 2:
                return status

        # Fallback: if all different, use most severe
        return self._most_severe_status(statuses)

    def _calculate_veto_consensus(
        self,
        results: List[LayerResult],
    ) -> VetoLevel:
        """Calculate veto level - use highest on disagreement."""
        veto_levels = [r.veto_level for r in results]
        veto_counts = Counter(veto_levels)

        # Check for 2/3 majority
        for veto, count in veto_counts.most_common():
            if count >= 2:
                return veto

        # Disagreement: use highest (most restrictive)
        return max(veto_levels, key=lambda v: list(VetoLevel).index(v))

    def _merge_findings(
        self,
        results: List[LayerResult],
    ) -> List[Finding]:
        """Merge findings from all runs, deduplicating by title."""
        seen_titles = set()
        merged = []

        for result in results:
            for finding in result.findings:
                if finding.title not in seen_titles:
                    seen_titles.add(finding.title)
                    merged.append(finding)

        return merged

    def _calculate_agreement(
        self,
        results: List[LayerResult],
    ) -> float:
        """Calculate agreement ratio (0.0 - 1.0)."""
        statuses = [r.status for r in results]
        status_counts = Counter(statuses)
        max_count = status_counts.most_common(1)[0][1]
        return max_count / len(results)

    def _most_severe_status(
        self,
        statuses: List[LayerStatus],
    ) -> LayerStatus:
        """Get most severe status from list."""
        severity_order = [
            LayerStatus.ERROR,
            LayerStatus.FAIL,
            LayerStatus.WARN,
            LayerStatus.PASS,
            LayerStatus.SKIP,
            LayerStatus.PENDING,
        ]
        for status in severity_order:
            if status in statuses:
                return status
        return LayerStatus.PENDING


def create_consensus_engine() -> ConsensusEngine:
    """Factory function to create consensus engine."""
    return ConsensusEngine()
```

---

## INTENT MANIFEST

```yaml
INTENT_MANIFEST:
  RF:
    - RF-001: "Triple redundancy = 3 runs por layer"
    - RF-002: "2/3 consensus para decisão"
    - RF-003: "Merge findings deduplicados"
    - RF-004: "Veto disagreement → use highest"
    - RF-005: "Calculate agreement ratio"

  INV:
    - INV-001: "Sempre exatamente 3 runs"
    - INV-002: "Consensus >= 2/3 para decisão"
    - INV-003: "Veto nunca reduz (só aumenta em disagreement)"
    - INV-004: "Findings são deduplicados por título"

  EDGE:
    - EDGE-001: "Todos os 3 runs diferem → use most severe"
    - EDGE-002: "2 runs ERROR, 1 PASS → ERROR"
    - EDGE-003: "0 findings em todos os runs → merged = []"
```

---

## TEST SCENARIOS

```yaml
test_scenarios:
  test_unanimous_pass:
    description: "3/3 concordam em PASS"
    input:
      - status: PASS, veto: NONE
      - status: PASS, veto: NONE
      - status: PASS, veto: NONE
    expected:
      status: PASS
      veto_level: NONE
      agreement_ratio: 1.0

  test_majority_fail:
    description: "2/3 concordam em FAIL"
    input:
      - status: FAIL, veto: MEDIUM
      - status: FAIL, veto: MEDIUM
      - status: PASS, veto: NONE
    expected:
      status: FAIL
      veto_level: MEDIUM
      agreement_ratio: 0.67

  test_veto_disagreement:
    description: "Disagreement em veto → usa highest"
    input:
      - status: FAIL, veto: WEAK
      - status: FAIL, veto: MEDIUM
      - status: FAIL, veto: STRONG
    expected:
      status: FAIL
      veto_level: STRONG  # highest
      agreement_ratio: 1.0  # status agrees

  test_all_different:
    description: "3 status diferentes → most severe"
    input:
      - status: PASS, veto: NONE
      - status: WARN, veto: WEAK
      - status: FAIL, veto: MEDIUM
    expected:
      status: FAIL  # most severe
      veto_level: MEDIUM  # highest
      agreement_ratio: 0.33
```

---

## GATES

```yaml
gates:
  G0_FILES_EXIST:
    description: "Arquivo de consensus existe"
    validation: "ls src/hl_mcp/core/consensus.py"

  G1_IMPORTS_WORK:
    description: "Import funciona"
    validation: |
      python -c "from hl_mcp.core import ConsensusEngine, ConsensusResult"

  G2_TESTS_PASS:
    description: "Testes passam"
    validation: "pytest tests/test_core/test_consensus.py -v"

  G3_TRIPLE_ENFORCED:
    description: "Triple redundancy enforced"
    validation: |
      python -c "
      from hl_mcp.core.consensus import ConsensusEngine
      engine = ConsensusEngine()
      try:
          engine.calculate_consensus([])  # Should fail
          assert False
      except ValueError as e:
          assert '3 runs' in str(e)
      "
```

---

## REFERÊNCIA

- `./S06_CONTEXT.md` - `./S12_CONTEXT.md` - All 7 Layers
- `./S14_CONTEXT.md` - Veto Gate
- `../MASTER_REQUIREMENTS_MAP.md` - PARTE 5: Consensus
