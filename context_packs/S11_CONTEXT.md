# S11 - layer-hl6-compliance | Context Pack v1.0

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
  id: S11
  name: layer-hl6-compliance
  title: "Layer HL-6: Compliance"
  wave: W1-CoreEngine
  priority: P0-CRITICAL
  type: implementation

objective: "Implementar Human Layer 6 - Compliance Review (STRONG veto)"

source_block: "../MASTER_REQUIREMENTS_MAP.md#PARTE-4-LAYERS"

dependencies:
  - S05  # Layer Base

deliverables:
  - src/hl_mcp/layers/hl6_compliance.py
  - tests/test_layers/test_hl6_compliance.py
```

---

## LAYER HL-6 SPECIFICATION

```yaml
layer:
  id: HL6
  name: "Compliance"
  veto_power: STRONG  # CRITICAL - compliance is non-negotiable
  focus_areas:
    - "GDPR compliance"
    - "Data privacy"
    - "Audit logging"
    - "Regulatory requirements"
    - "Terms of service"
    - "License compliance"

  what_it_catches:
    - "PII handling violations"
    - "Missing consent mechanisms"
    - "Inadequate data retention policies"
    - "Audit trail gaps"
    - "License violations"
    - "Export control issues"
```

---

## IMPLEMENTATION SPEC

### HL-6 Compliance Layer (layers/hl6_compliance.py)

```python
"""Human Layer 6: Compliance Review.

STRONG VETO POWER - Compliance is non-negotiable.
"""
from typing import List

from ..models import LayerID, VetoLevel, Finding
from ..llm import BaseLLMClient
from .base import BaseHumanLayer, ActionContext
from .prompts import LayerPromptManager


class HL6ComplianceLayer(BaseHumanLayer):
    """Human Layer 6: Compliance Review.

    Focus: GDPR, privacy, audit, regulatory requirements.
    Veto Power: STRONG (compliance is non-negotiable)

    This is one of the three layers with STRONG veto power.
    Any compliance concern must be addressed before proceeding.
    """

    def __init__(self, llm_client: BaseLLMClient):
        super().__init__(
            layer_id=LayerID.HL6,
            veto_power=VetoLevel.STRONG,
            llm_client=llm_client,
        )
        self._prompt_manager = LayerPromptManager()

    @property
    def name(self) -> str:
        return "Compliance"

    @property
    def description(self) -> str:
        return "Reviews regulatory compliance, privacy, and audit requirements"

    async def analyze(self, action: ActionContext) -> List[Finding]:
        """Analyze action for compliance issues."""
        system_prompt = self._prompt_manager.get_system_prompt(self.layer_id)
        analysis_prompt = self._build_compliance_prompt(action)

        response = await self.llm_client.generate(
            prompt=analysis_prompt,
            system_prompt=system_prompt,
        )

        return self._prompt_manager.parse_findings(
            response.content,
            self.layer_id
        )

    def _build_compliance_prompt(self, action: ActionContext) -> str:
        """Build compliance-specific analysis prompt."""
        base_prompt = self._prompt_manager.get_analysis_prompt(action)

        compliance_specifics = """

**COMPLIANCE REVIEW - STRONG VETO POWER**

This layer has STRONG veto power. Compliance is NON-NEGOTIABLE.

**GDPR Compliance Checklist**:

1. **Lawful Basis**
   - Is there a lawful basis for processing?
   - Is consent obtained where required?
   - Is consent granular and withdrawable?

2. **Data Minimization**
   - Is only necessary data collected?
   - Is data retained only as long as needed?
   - Can data be anonymized?

3. **Rights of Data Subjects**
   - Can users access their data?
   - Can users request deletion?
   - Can users export their data (portability)?

4. **Data Protection**
   - Is PII encrypted?
   - Is access to PII logged?
   - Are there data processing agreements with third parties?

**Audit & Logging**:

5. **Audit Trail**
   - Are actions logged?
   - Are logs immutable?
   - Can logs be used for forensics?

6. **Access Logging**
   - Who accessed what data?
   - When was data accessed?
   - Is access auditable?

**Regulatory Compliance**:

7. **Industry Regulations**
   - HIPAA (healthcare)?
   - PCI-DSS (payments)?
   - SOX (financial)?
   - COPPA (children)?

8. **Geographic Compliance**
   - Data residency requirements?
   - Cross-border transfer restrictions?
   - Local regulations?

**License & IP**:

9. **License Compliance**
   - Are third-party licenses respected?
   - Are open source obligations met?
   - Are there export control issues?

10. **Terms of Service**
    - Does action comply with ToS?
    - Are users notified of changes?

Severity guidelines:
- **critical**: Direct violation of regulation
- **high**: Significant compliance gap
- **medium**: Compliance concern
- **low**: Minor compliance improvement
- **info**: Compliance best practice

COMPLIANCE VIOLATIONS CANNOT BE IGNORED.
"""
        return base_prompt + compliance_specifics


def create_hl6_layer(llm_client: BaseLLMClient) -> HL6ComplianceLayer:
    """Factory function to create HL-6 layer."""
    return HL6ComplianceLayer(llm_client)
```

---

## INTENT MANIFEST

```yaml
INTENT_MANIFEST:
  RF:
    - RF-001: "HL6 analisa compliance e regulamentação"
    - RF-002: "Veto power é STRONG (compliance é non-negotiable)"
    - RF-003: "Cobre GDPR, audit, regulatory, license"
    - RF-004: "Detecta PII violations, consent issues"
    - RF-005: "Factory function para criação padronizada"

  INV:
    - INV-001: "layer_id sempre é LayerID.HL6"
    - INV-002: "veto_power sempre é VetoLevel.STRONG"
    - INV-003: "Qualquer finding CRITICAL gera STRONG veto"
    - INV-004: "Compliance violations não podem ser ignorados"

  EDGE:
    - EDGE-001: "Action sem dados pessoais → menos checks"
    - EDGE-002: "Action de admin → GDPR check obrigatório"
    - EDGE-003: "Action com third-party → license check"
```

---

## GATES

```yaml
gates:
  G0_FILES_EXIST:
    description: "Arquivo de HL-6 existe"
    validation: "ls src/hl_mcp/layers/hl6_compliance.py"

  G1_IMPORTS_WORK:
    description: "Import funciona"
    validation: |
      python -c "from hl_mcp.layers import HL6ComplianceLayer, create_hl6_layer"

  G2_TESTS_PASS:
    description: "Testes passam"
    validation: "pytest tests/test_layers/test_hl6_compliance.py -v"

  G3_STRONG_VETO:
    description: "STRONG veto power verificado"
    validation: |
      python -c "
      from hl_mcp.layers.hl6_compliance import HL6ComplianceLayer
      from hl_mcp.models import VetoLevel
      from unittest.mock import MagicMock

      layer = HL6ComplianceLayer(MagicMock())
      assert layer.veto_power == VetoLevel.STRONG, 'HL6 must have STRONG veto'
      "
```

---

## REFERÊNCIA

- `./S05_CONTEXT.md` - Layer Base
- `./S10_CONTEXT.md` - HL-5 Performance
- `./S12_CONTEXT.md` - HL-7 Final Review
