# S27 - auth-setup | Context Pack v1.0

---

## PRODUCT REFERENCE

```yaml
product_reference:
  product_id: "HUMANGR"
  product_name: "Human Layer MCP Server"
  wave: "W3-CloudMVP"
  product_pack: "./PRODUCT_PACK.md"
  sprint_index: "./SPRINT_INDEX.yaml"
```

---

## RELOAD ANCHOR

```yaml
sprint:
  id: S27
  name: auth-setup
  title: "Authentication"
  wave: W3-CloudMVP
  priority: P1-HIGH
  type: implementation

objective: "Authentication com Clerk"

dependencies:
  - S25  # Cloud Infrastructure

deliverables:
  - src/hl_mcp/cloud/auth/__init__.py
  - src/hl_mcp/cloud/auth/clerk.py
  - src/hl_mcp/cloud/auth/middleware.py
  - src/hl_mcp/cloud/auth/dependencies.py
```

---

## AUTH OVERVIEW

```yaml
authentication:
  provider: Clerk
  features:
    - "Social login (Google, GitHub)"
    - "Email/password"
    - "Magic links"
    - "MFA support"
    - "Session management"

  why_clerk:
    - "Fast implementation"
    - "Good DX"
    - "Handles security complexity"
    - "SOC 2 compliant"
```

---

## IMPLEMENTATION SPEC

### Clerk Integration (auth/clerk.py)

```python
"""Clerk authentication integration."""
import os
from typing import Optional
import httpx
from pydantic import BaseModel


class ClerkUser(BaseModel):
    """User from Clerk."""
    id: str
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None


class ClerkAuth:
    """Clerk authentication client."""

    def __init__(self):
        self.secret_key = os.getenv("CLERK_SECRET_KEY")
        self.publishable_key = os.getenv("CLERK_PUBLISHABLE_KEY")
        self._client = httpx.AsyncClient(
            base_url="https://api.clerk.com/v1",
            headers={"Authorization": f"Bearer {self.secret_key}"}
        )

    async def verify_session(self, session_token: str) -> Optional[ClerkUser]:
        """Verify session token and return user."""
        try:
            response = await self._client.get(
                "/sessions/verify",
                params={"token": session_token}
            )
            if response.status_code == 200:
                data = response.json()
                return ClerkUser(
                    id=data["user_id"],
                    email=data["user"]["email_addresses"][0]["email"],
                    first_name=data["user"].get("first_name"),
                    last_name=data["user"].get("last_name"),
                )
        except Exception:
            pass
        return None

    async def get_user(self, user_id: str) -> Optional[ClerkUser]:
        """Get user by ID."""
        try:
            response = await self._client.get(f"/users/{user_id}")
            if response.status_code == 200:
                data = response.json()
                return ClerkUser(
                    id=data["id"],
                    email=data["email_addresses"][0]["email_address"],
                    first_name=data.get("first_name"),
                    last_name=data.get("last_name"),
                )
        except Exception:
            pass
        return None


clerk_auth = ClerkAuth()
```

### Auth Dependencies (auth/dependencies.py)

```python
"""FastAPI auth dependencies."""
from typing import Optional
from fastapi import Depends, HTTPException, Header

from .clerk import clerk_auth, ClerkUser


async def get_current_user(
    authorization: Optional[str] = Header(None),
) -> ClerkUser:
    """Get current authenticated user."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Not authenticated")

    token = authorization.replace("Bearer ", "")
    user = await clerk_auth.verify_session(token)

    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")

    return user


async def get_optional_user(
    authorization: Optional[str] = Header(None),
) -> Optional[ClerkUser]:
    """Get user if authenticated, None otherwise."""
    if not authorization:
        return None

    token = authorization.replace("Bearer ", "")
    return await clerk_auth.verify_session(token)
```

---

## INTENT MANIFEST

```yaml
INTENT_MANIFEST:
  RF:
    - RF-001: "Clerk integration"
    - RF-002: "Session verification"
    - RF-003: "FastAPI dependencies"
    - RF-004: "Optional auth support"

  INV:
    - INV-001: "Keys via env vars"
    - INV-002: "Nunca logar tokens"
    - INV-003: "401 para unauthorized"
```

---

## GATES

```yaml
gates:
  G0_FILES_EXIST:
    validation: |
      ls src/hl_mcp/cloud/auth/clerk.py
      ls src/hl_mcp/cloud/auth/dependencies.py

  G1_IMPORTS_WORK:
    validation: |
      python -c "from hl_mcp.cloud.auth import get_current_user"
```

---

## REFERÊNCIA

- `./S25_CONTEXT.md` - Cloud Infrastructure
- `./S28_CONTEXT.md` - Cloud API Base
