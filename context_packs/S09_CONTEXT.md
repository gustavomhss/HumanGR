# S09 - layer-hl4-security | Context Pack v1.0

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
  id: S09
  name: layer-hl4-security
  title: "Layer HL-4: Security"
  wave: W1-CoreEngine
  priority: P0-CRITICAL
  type: implementation

objective: "Implementar Human Layer 4 - Security Review (STRONG veto)"

source_block: "../MASTER_REQUIREMENTS_MAP.md#PARTE-4-LAYERS"

dependencies:
  - S05  # Layer Base

deliverables:
  - src/hl_mcp/layers/hl4_security.py
  - tests/test_layers/test_hl4_security.py
```

---

## LAYER HL-4 SPECIFICATION

```yaml
layer:
  id: HL4
  name: "Security"
  veto_power: STRONG  # CRITICAL - can block everything
  focus_areas:
    - "OWASP Top 10"
    - "Injection vulnerabilities"
    - "Authentication/Authorization"
    - "Data exposure"
    - "Cryptographic failures"
    - "Security misconfigurations"

  what_it_catches:
    - "SQL/NoSQL injection"
    - "XSS (Cross-Site Scripting)"
    - "Command injection"
    - "Path traversal"
    - "Insecure direct object references"
    - "Broken authentication"
    - "Sensitive data exposure"
    - "Hardcoded credentials"

  veto_scenarios:
    STRONG:
      - "Any injection vulnerability"
      - "Authentication bypass"
      - "Sensitive data exposure"
      - "Cryptographic weakness"
    MEDIUM:
      - "Missing input validation"
      - "Verbose error messages"
    WEAK:
      - "Minor security headers missing"
```

---

## IMPLEMENTATION SPEC

### HL-4 Security Layer (layers/hl4_security.py)

```python
"""Human Layer 4: Security Review.

STRONG VETO POWER - Security is non-negotiable.
"""
from typing import List

from ..models import LayerID, VetoLevel, Finding
from ..llm import BaseLLMClient
from .base import BaseHumanLayer, ActionContext
from .prompts import LayerPromptManager


class HL4SecurityLayer(BaseHumanLayer):
    """Human Layer 4: Security Review.

    Focus: OWASP Top 10, injection, auth, data exposure.
    Veto Power: STRONG (can block everything)

    This is one of the three layers with STRONG veto power.
    Any security concern must be addressed before proceeding.
    """

    def __init__(self, llm_client: BaseLLMClient):
        super().__init__(
            layer_id=LayerID.HL4,
            veto_power=VetoLevel.STRONG,
            llm_client=llm_client,
        )
        self._prompt_manager = LayerPromptManager()

    @property
    def name(self) -> str:
        return "Security"

    @property
    def description(self) -> str:
        return "Reviews security vulnerabilities and OWASP Top 10"

    async def analyze(self, action: ActionContext) -> List[Finding]:
        """Analyze action for security vulnerabilities.

        Checks for:
        - OWASP Top 10 vulnerabilities
        - Injection attacks
        - Authentication issues
        - Data exposure
        """
        system_prompt = self._prompt_manager.get_system_prompt(self.layer_id)
        analysis_prompt = self._build_security_prompt(action)

        response = await self.llm_client.generate(
            prompt=analysis_prompt,
            system_prompt=system_prompt,
        )

        return self._prompt_manager.parse_findings(
            response.content,
            self.layer_id
        )

    def _build_security_prompt(self, action: ActionContext) -> str:
        """Build security-specific analysis prompt."""
        base_prompt = self._prompt_manager.get_analysis_prompt(action)

        security_specifics = """

**SECURITY REVIEW - STRONG VETO POWER**

This layer has STRONG veto power. Flag ALL security concerns.

**OWASP Top 10 Checklist**:

1. **A01 - Broken Access Control**
   - Can users access resources they shouldn't?
   - Are authorization checks in place?
   - Is there IDOR (Insecure Direct Object Reference)?

2. **A02 - Cryptographic Failures**
   - Is sensitive data encrypted in transit (HTTPS)?
   - Is sensitive data encrypted at rest?
   - Are passwords properly hashed (bcrypt/argon2)?

3. **A03 - Injection**
   - SQL injection possible?
   - NoSQL injection possible?
   - Command injection possible?
   - LDAP/XPath injection possible?

4. **A04 - Insecure Design**
   - Is the design inherently insecure?
   - Are there missing security controls?

5. **A05 - Security Misconfiguration**
   - Are default credentials in use?
   - Are unnecessary features enabled?
   - Are error messages too verbose?

6. **A06 - Vulnerable Components**
   - Are there known vulnerable dependencies?
   - Are dependencies up to date?

7. **A07 - Authentication Failures**
   - Can authentication be bypassed?
   - Are sessions managed securely?
   - Is MFA available for sensitive operations?

8. **A08 - Data Integrity Failures**
   - Are updates verified for integrity?
   - Is CI/CD pipeline secure?

9. **A09 - Logging Failures**
   - Are security events logged?
   - Are logs protected from tampering?

10. **A10 - SSRF**
    - Can the server be tricked into making requests?
    - Are URLs validated?

**Additional Checks**:
- Hardcoded credentials or API keys?
- Sensitive data in logs?
- XSS vulnerabilities?
- Path traversal possible?
- Rate limiting in place?

Severity guidelines:
- **critical**: Exploitable vulnerability (injection, auth bypass)
- **high**: Significant security flaw
- **medium**: Security weakness
- **low**: Minor security concern
- **info**: Security best practice suggestion

BE PARANOID. When in doubt, flag it.
"""
        return base_prompt + security_specifics


def create_hl4_layer(llm_client: BaseLLMClient) -> HL4SecurityLayer:
    """Factory function to create HL-4 layer."""
    return HL4SecurityLayer(llm_client)
```

---

## INTENT MANIFEST

```yaml
INTENT_MANIFEST:
  RF:
    - RF-001: "HL4 analisa vulnerabilidades de segurança"
    - RF-002: "Veto power é STRONG (pode bloquear tudo)"
    - RF-003: "Cobre OWASP Top 10 completo"
    - RF-004: "Detecta injection, auth, data exposure"
    - RF-005: "Prompt instrui 'BE PARANOID'"

  INV:
    - INV-001: "layer_id sempre é LayerID.HL4"
    - INV-002: "veto_power sempre é VetoLevel.STRONG"
    - INV-003: "Qualquer finding CRITICAL gera STRONG veto"
    - INV-004: "Nunca ignora vulnerabilidades conhecidas"

  EDGE:
    - EDGE-001: "Action sem código → análise conceitual"
    - EDGE-002: "Action read-only → foco em data exposure"
    - EDGE-003: "Action de admin → extra scrutiny"
```

---

## GATES

```yaml
gates:
  G0_FILES_EXIST:
    description: "Arquivo de HL-4 existe"
    validation: "ls src/hl_mcp/layers/hl4_security.py"

  G1_IMPORTS_WORK:
    description: "Import funciona"
    validation: |
      python -c "from hl_mcp.layers import HL4SecurityLayer, create_hl4_layer"

  G2_TESTS_PASS:
    description: "Testes passam"
    validation: "pytest tests/test_layers/test_hl4_security.py -v"

  G3_STRONG_VETO:
    description: "STRONG veto power verificado"
    validation: |
      python -c "
      from hl_mcp.layers.hl4_security import HL4SecurityLayer
      from hl_mcp.models import VetoLevel
      from unittest.mock import MagicMock

      layer = HL4SecurityLayer(MagicMock())
      assert layer.veto_power == VetoLevel.STRONG, 'HL4 must have STRONG veto'
      "
```

---

## REFERÊNCIA

- `./S05_CONTEXT.md` - Layer Base
- `./S08_CONTEXT.md` - HL-3 Edge Cases
- `./S10_CONTEXT.md` - HL-5 Performance
