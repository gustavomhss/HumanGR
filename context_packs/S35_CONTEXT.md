# S35 - billing-tiers | Context Pack v1.0

---

## RELOAD ANCHOR

```yaml
sprint:
  id: S35
  name: billing-tiers
  title: "Billing Tiers"
  wave: W3-CloudMVP
  priority: P1-HIGH
  type: implementation

objective: "Implementar 7 tiers de pricing"

dependencies:
  - S34  # Billing Integration

deliverables:
  - src/hl_mcp/cloud/billing/tiers.py
  - src/hl_mcp/cloud/billing/limits.py
```

---

## PRICING TIERS

```yaml
tiers:
  free:
    price: "$0/forever"
    validations: "Unlimited (self-hosted)"
    features: ["All 7 layers", "Triple redundancy", "CLI", "MCP"]

  starter:
    price: "$12/month"
    validations: "1,000/month"
    features: ["Dashboard", "30-day history", "Email support"]

  pro:
    price: "$49/month"
    validations: "10,000/month"
    features: ["90-day history", "CI/CD", "Priority support", "5 users"]

  business:
    price: "$249/month"
    validations: "100,000/month"
    features: ["Unlimited history", "SSO", "SLA 99.9%", "Unlimited users"]

  enterprise:
    price: "Custom"
    validations: "Custom"
    features: ["Dedicated support", "Custom SLA", "On-prem option"]
```

---

## IMPLEMENTATION SPEC (Abbreviated)

```python
# billing/tiers.py
from enum import Enum
from pydantic import BaseModel


class TierID(str, Enum):
    FREE = "free"
    STARTER = "starter"
    PRO = "pro"
    BUSINESS = "business"
    ENTERPRISE = "enterprise"


class TierLimits(BaseModel):
    validations_per_month: int
    history_days: int
    users: int
    ci_cd: bool
    sso: bool


TIER_LIMITS = {
    TierID.FREE: TierLimits(
        validations_per_month=999999,  # Unlimited (self-hosted)
        history_days=0,
        users=1,
        ci_cd=False,
        sso=False,
    ),
    TierID.STARTER: TierLimits(
        validations_per_month=1000,
        history_days=30,
        users=1,
        ci_cd=False,
        sso=False,
    ),
    TierID.PRO: TierLimits(
        validations_per_month=10000,
        history_days=90,
        users=5,
        ci_cd=True,
        sso=False,
    ),
    TierID.BUSINESS: TierLimits(
        validations_per_month=100000,
        history_days=365,
        users=999,
        ci_cd=True,
        sso=True,
    ),
}


def check_tier_limit(tier: TierID, validations_this_month: int) -> bool:
    """Check if user is within tier limits."""
    limits = TIER_LIMITS.get(tier)
    if not limits:
        return False
    return validations_this_month < limits.validations_per_month
```

---

## 🎉 MILESTONE: CLOUD MVP

```
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║                    🚀 CLOUD MVP COMPLETE 🚀                        ║
║                                                                    ║
║    Wave 3: Cloud MVP ✅                                            ║
║                                                                    ║
║    Features:                                                       ║
║    ✅ Cloud Infrastructure (Docker, K8s)                           ║
║    ✅ Database (PostgreSQL, Alembic)                               ║
║    ✅ Authentication (Clerk)                                       ║
║    ✅ REST API (FastAPI)                                           ║
║    ✅ Dashboard (Next.js)                                          ║
║    ✅ Billing (Stripe)                                             ║
║                                                                    ║
║    Next: Wave 4 - Growth Features                                  ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
```

---

## GATES

```yaml
gates:
  G0_FILES_EXIST:
    validation: |
      ls src/hl_mcp/cloud/billing/tiers.py
      ls src/hl_mcp/cloud/billing/limits.py

  G1_TIERS_DEFINED:
    validation: |
      python -c "
      from hl_mcp.cloud.billing.tiers import TierID, TIER_LIMITS
      assert len(TierID) >= 4
      assert TierID.STARTER in TIER_LIMITS
      "
```

---

## REFERÊNCIA

- `./S34_CONTEXT.md` - Billing Integration
- `./S36_CONTEXT.md` - Perspectives (Wave 4)
