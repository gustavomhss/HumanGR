# S28 - cloud-api-base | Context Pack v1.0

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
  id: S28
  name: cloud-api-base
  title: "Cloud API Base"
  wave: W3-CloudMVP
  priority: P1-HIGH
  type: implementation

objective: "FastAPI base para Cloud dashboard"

dependencies:
  - S27  # Authentication

deliverables:
  - src/hl_mcp/cloud/api/__init__.py
  - src/hl_mcp/cloud/api/app.py
  - src/hl_mcp/cloud/api/middleware.py
  - src/hl_mcp/cloud/api/health.py
```

---

## API OVERVIEW

```yaml
api:
  framework: FastAPI
  docs: "/docs (Swagger), /redoc"
  versioning: "/api/v1"

  middleware:
    - CORS
    - Request ID
    - Logging
    - Rate limiting
```

---

## IMPLEMENTATION SPEC

### FastAPI App (api/app.py)

```python
"""Human Layer Cloud API."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .middleware import RequestIDMiddleware, LoggingMiddleware
from .health import router as health_router


def create_app() -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="Human Layer Cloud",
        description="7 Layers of Human Judgment for AI Agents",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(LoggingMiddleware)

    # Routes
    app.include_router(health_router, prefix="/api/v1")

    return app


app = create_app()
```

### Health Check (api/health.py)

```python
"""Health check endpoints."""
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ..db.session import get_async_db

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    version: str
    database: str
    redis: str


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        database="connected",
        redis="connected",
    )


@router.get("/ready")
async def readiness_check():
    """Readiness check for Kubernetes."""
    return {"ready": True}
```

---

## INTENT MANIFEST

```yaml
INTENT_MANIFEST:
  RF:
    - RF-001: "FastAPI application"
    - RF-002: "Health/ready endpoints"
    - RF-003: "CORS middleware"
    - RF-004: "API versioning /api/v1"

  INV:
    - INV-001: "Health sempre responde"
    - INV-002: "CORS configurável"
```

---

## GATES

```yaml
gates:
  G0_FILES_EXIST:
    validation: |
      ls src/hl_mcp/cloud/api/app.py
      ls src/hl_mcp/cloud/api/health.py

  G1_APP_STARTS:
    validation: |
      python -c "from hl_mcp.cloud.api import app; print(app.title)"
```

---

## REFERÊNCIA

- `./S27_CONTEXT.md` - Authentication
- `./S29_CONTEXT.md` - Cloud API Endpoints
