# S36 - perspectives-implementation | Context Pack v1.0

---

## PRODUCT REFERENCE

```yaml
product_reference:
  product_id: "HUMANGR"
  product_name: "Human Layer MCP Server"
  wave: "W4-Growth"
  product_pack: "./PRODUCT_PACK.md"
  sprint_index: "./SPRINT_INDEX.yaml"
```

---

## RELOAD ANCHOR

```yaml
sprint:
  id: S36
  name: perspectives-implementation
  title: "6 Perspectives"
  wave: W4-Growth
  priority: P2-MEDIUM
  type: implementation

objective: "Implementar 6 perspectivas de teste"

dependencies:
  - S24  # OSS Launch

deliverables:
  - src/hl_mcp/perspectives/__init__.py
  - src/hl_mcp/perspectives/base.py
  - src/hl_mcp/perspectives/tired_user.py
  - src/hl_mcp/perspectives/malicious_insider.py
  - src/hl_mcp/perspectives/confused_newbie.py
  - src/hl_mcp/perspectives/power_user.py
  - src/hl_mcp/perspectives/auditor.py
  - src/hl_mcp/perspectives/3am_operator.py
  - tests/test_perspectives/
```

---

## 6 PERSPECTIVES

```yaml
perspectives:
  tired_user:
    description: "Frustrated, impatient, making mistakes"
    behaviors:
      - "Clicks wrong button"
      - "Ignores warnings"
      - "Rushes through forms"
      - "Expects quick results"
    prompt_modifier: "Think like an exhausted user who just wants to get this done..."

  malicious_insider:
    description: "Trying to abuse the system"
    behaviors:
      - "Looks for backdoors"
      - "Tests privilege escalation"
      - "Tries to access others' data"
      - "Exploits edge cases"
    prompt_modifier: "Think like someone trying to abuse this feature..."

  confused_newbie:
    description: "Lost, first time using the system"
    behaviors:
      - "Doesn't understand jargon"
      - "Needs hand-holding"
      - "Clicks everything"
      - "Expects magic"
    prompt_modifier: "Think like someone who has never used this before..."

  power_user:
    description: "Wants shortcuts, efficiency"
    behaviors:
      - "Uses keyboard shortcuts"
      - "Wants batch operations"
      - "Hates unnecessary clicks"
      - "Knows the system"
    prompt_modifier: "Think like an expert user who wants maximum efficiency..."

  auditor:
    description: "Checking compliance, logs"
    behaviors:
      - "Wants audit trail"
      - "Checks permissions"
      - "Verifies data integrity"
      - "Needs reports"
    prompt_modifier: "Think like a compliance auditor examining this action..."

  3am_operator:
    description: "Sleepy, emergency situation"
    behaviors:
      - "Making mistakes"
      - "Stressed"
      - "Needs clear guidance"
      - "Can't think straight"
    prompt_modifier: "Think like someone handling an emergency at 3 AM..."
```

---

## IMPLEMENTATION SPEC (Abbreviated)

```python
# perspectives/base.py
from abc import ABC, abstractmethod
from typing import List
from ..models import Finding
from ..layers.base import ActionContext


class BasePerspective(ABC):
    """Base class for testing perspectives."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    def get_prompt_modifier(self) -> str:
        """Get perspective-specific prompt addition."""
        pass

    @abstractmethod
    async def analyze(self, action: ActionContext) -> List[Finding]:
        """Analyze action from this perspective."""
        pass
```

```python
# perspectives/tired_user.py
class TiredUserPerspective(BasePerspective):
    @property
    def name(self) -> str:
        return "Tired User"

    @property
    def description(self) -> str:
        return "Frustrated, impatient, making mistakes"

    def get_prompt_modifier(self) -> str:
        return """
        Think like an exhausted user who:
        - Just wants to get this done quickly
        - Might click the wrong button
        - Will ignore warnings if they can
        - Gets frustrated with extra steps
        """
```

---

## GATES

```yaml
gates:
  G0_FILES_EXIST:
    validation: |
      ls src/hl_mcp/perspectives/base.py
      ls src/hl_mcp/perspectives/tired_user.py
      ls src/hl_mcp/perspectives/malicious_insider.py

  G1_ALL_6_PERSPECTIVES:
    validation: |
      python -c "
      from hl_mcp.perspectives import (
          TiredUserPerspective,
          MaliciousInsiderPerspective,
          ConfusedNewbiePerspective,
          PowerUserPerspective,
          AuditorPerspective,
          ThreeAMOperatorPerspective,
      )
      assert len([TiredUserPerspective, MaliciousInsiderPerspective,
                  ConfusedNewbiePerspective, PowerUserPerspective,
                  AuditorPerspective, ThreeAMOperatorPerspective]) == 6
      "
```

---

## REFERÊNCIA

- `./S24_CONTEXT.md` - OSS Launch
- `./S37_CONTEXT.md` - CI/CD Integrations
