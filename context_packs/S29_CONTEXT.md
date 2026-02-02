# S29 - cloud-api-endpoints | Context Pack v1.0

---

## RELOAD ANCHOR

```yaml
sprint:
  id: S29
  name: cloud-api-endpoints
  title: "Cloud API Endpoints"
  wave: W3-CloudMVP
  priority: P1-HIGH
  type: implementation

objective: "Endpoints REST para validations, reports, etc"

dependencies:
  - S28  # Cloud API Base

deliverables:
  - src/hl_mcp/cloud/api/routes/__init__.py
  - src/hl_mcp/cloud/api/routes/validations.py
  - src/hl_mcp/cloud/api/routes/reports.py
  - src/hl_mcp/cloud/api/routes/users.py
```

---

## API ENDPOINTS

```yaml
endpoints:
  validations:
    POST /api/v1/validations: "Run new validation"
    GET /api/v1/validations: "List validations"
    GET /api/v1/validations/{id}: "Get validation"

  reports:
    GET /api/v1/reports/{id}: "Get detailed report"
    GET /api/v1/reports/{id}/pdf: "Export PDF"

  users:
    GET /api/v1/users/me: "Current user"
    PATCH /api/v1/users/me: "Update profile"
```

---

## IMPLEMENTATION SPEC (Abbreviated)

```python
# routes/validations.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List

from ...auth import get_current_user, ClerkUser

router = APIRouter(prefix="/validations", tags=["validations"])


class ValidationRequest(BaseModel):
    agent_id: str
    action_type: str
    action_description: str


class ValidationResponse(BaseModel):
    id: str
    decision: str
    veto_level: str
    findings_count: int


@router.post("/", response_model=ValidationResponse)
async def create_validation(
    request: ValidationRequest,
    user: ClerkUser = Depends(get_current_user),
):
    """Run a new validation."""
    # Implementation
    pass


@router.get("/", response_model=List[ValidationResponse])
async def list_validations(
    user: ClerkUser = Depends(get_current_user),
    limit: int = 50,
    offset: int = 0,
):
    """List user's validations."""
    pass
```

---

## GATES

```yaml
gates:
  G0_FILES_EXIST:
    validation: "ls src/hl_mcp/cloud/api/routes/validations.py"

  G1_ROUTES_REGISTERED:
    validation: |
      python -c "
      from hl_mcp.cloud.api import app
      routes = [r.path for r in app.routes]
      assert '/api/v1/validations' in str(routes)
      "
```

---

## REFERÊNCIA

- `./S28_CONTEXT.md` - Cloud API Base
- `./S30_CONTEXT.md` - WebSocket Real-time
