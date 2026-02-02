# S39 - analytics | Context Pack v1.0

---

## RELOAD ANCHOR

```yaml
sprint:
  id: S39
  name: analytics
  title: "Analytics & Trends"
  wave: W4-Growth
  priority: P2-MEDIUM
  type: implementation

objective: "Analytics e métricas de validações"

dependencies:
  - S29  # Cloud API Endpoints

deliverables:
  - src/hl_mcp/cloud/analytics/__init__.py
  - src/hl_mcp/cloud/analytics/metrics.py
  - src/hl_mcp/cloud/analytics/trends.py
  - src/hl_mcp/cloud/api/routes/analytics.py
  - dashboard/app/dashboard/analytics/page.tsx
```

---

## ANALYTICS OVERVIEW

```yaml
analytics:
  metrics:
    - total_validations
    - approval_rate
    - rejection_rate
    - needs_review_rate
    - avg_validation_time
    - findings_by_severity
    - findings_by_layer
    - veto_distribution

  trends:
    - validations_over_time
    - approval_rate_trend
    - common_issues
    - layer_performance

  time_ranges:
    - last_7_days
    - last_30_days
    - last_90_days
    - custom_range
```

---

## IMPLEMENTATION SPEC (Abbreviated)

```python
# analytics/metrics.py
from datetime import datetime, timedelta
from typing import Dict, List
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..db.models import Validation, Finding


class AnalyticsService:
    """Analytics and metrics service."""

    def __init__(self, db: Session):
        self.db = db

    def get_summary(
        self,
        org_id: str,
        days: int = 30,
    ) -> Dict:
        """Get summary metrics."""
        since = datetime.utcnow() - timedelta(days=days)

        validations = self.db.query(Validation).filter(
            Validation.org_id == org_id,
            Validation.created_at >= since,
        ).all()

        total = len(validations)
        approved = len([v for v in validations if v.decision == "approved"])
        rejected = len([v for v in validations if v.decision == "rejected"])

        return {
            "total_validations": total,
            "approved": approved,
            "rejected": rejected,
            "approval_rate": approved / total if total > 0 else 0,
            "period_days": days,
        }

    def get_findings_by_layer(
        self,
        org_id: str,
        days: int = 30,
    ) -> Dict[str, int]:
        """Get finding counts by layer."""
        # Implementation
        pass

    def get_trends(
        self,
        org_id: str,
        days: int = 30,
        granularity: str = "day",
    ) -> List[Dict]:
        """Get validation trends over time."""
        # Implementation
        pass
```

---

## GATES

```yaml
gates:
  G0_FILES_EXIST:
    validation: |
      ls src/hl_mcp/cloud/analytics/metrics.py
      ls src/hl_mcp/cloud/api/routes/analytics.py

  G1_METRICS_WORK:
    validation: |
      python -c "from hl_mcp.cloud.analytics import AnalyticsService"
```

---

## REFERÊNCIA

- `./S29_CONTEXT.md` - Cloud API Endpoints
- `./S40_CONTEXT.md` - Team Features
