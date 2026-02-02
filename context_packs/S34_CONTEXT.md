# S34 - billing-integration | Context Pack v1.0

---

## RELOAD ANCHOR

```yaml
sprint:
  id: S34
  name: billing-integration
  title: "Billing Integration"
  wave: W3-CloudMVP
  priority: P1-HIGH
  type: implementation

objective: "Integração Stripe para billing"

dependencies:
  - S28  # Cloud API Base

deliverables:
  - src/hl_mcp/cloud/billing/__init__.py
  - src/hl_mcp/cloud/billing/stripe_client.py
  - src/hl_mcp/cloud/billing/webhooks.py
  - src/hl_mcp/cloud/api/routes/billing.py
```

---

## BILLING OVERVIEW

```yaml
billing:
  provider: Stripe
  features:
    - "Subscription management"
    - "Usage-based billing (future)"
    - "Invoice generation"
    - "Payment methods"

  webhooks:
    - customer.subscription.created
    - customer.subscription.updated
    - customer.subscription.deleted
    - invoice.paid
    - invoice.payment_failed
```

---

## IMPLEMENTATION SPEC (Abbreviated)

```python
# billing/stripe_client.py
import os
import stripe

stripe.api_key = os.getenv("STRIPE_SECRET_KEY")


class StripeClient:
    """Stripe billing client."""

    async def create_checkout_session(
        self,
        user_id: str,
        price_id: str,
        success_url: str,
        cancel_url: str,
    ) -> str:
        """Create checkout session for subscription."""
        session = stripe.checkout.Session.create(
            customer_email=None,  # Get from user
            mode="subscription",
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=success_url,
            cancel_url=cancel_url,
            metadata={"user_id": user_id},
        )
        return session.url

    async def create_portal_session(
        self,
        customer_id: str,
        return_url: str,
    ) -> str:
        """Create billing portal session."""
        session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=return_url,
        )
        return session.url


stripe_client = StripeClient()
```

---

## GATES

```yaml
gates:
  G0_FILES_EXIST:
    validation: |
      ls src/hl_mcp/cloud/billing/stripe_client.py
      ls src/hl_mcp/cloud/billing/webhooks.py
```

---

## REFERÊNCIA

- `./S28_CONTEXT.md` - Cloud API
- `./S35_CONTEXT.md` - Billing Tiers
