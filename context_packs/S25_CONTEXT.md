# S25 - cloud-infrastructure | Context Pack v1.0

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
  id: S25
  name: cloud-infrastructure
  title: "Cloud Infrastructure"
  wave: W3-CloudMVP
  priority: P1-HIGH
  type: infrastructure

objective: "Infraestrutura cloud (Docker, containers)"

dependencies:
  - S24  # OSS Launch

deliverables:
  - docker-compose.yml
  - Dockerfile
  - docker/
  - deploy/kubernetes/
```

---

## INFRASTRUCTURE OVERVIEW

```yaml
infrastructure:
  containerization: Docker
  orchestration: "Docker Compose (dev), Kubernetes (prod)"

  services:
    api: "Human Layer API (FastAPI)"
    worker: "Background workers"
    postgres: "Database"
    redis: "Cache/queue"

  cloud_providers:
    primary: "AWS (ECS/EKS)"
    alternative: "GCP, Azure"
```

---

## IMPLEMENTATION SPEC

### Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11-slim AS base

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir build && \
    pip install --no-cache-dir .

# Copy source
COPY src/ src/

# Production stage
FROM base AS production
CMD ["python", "-m", "hl_mcp.cli", "serve", "--transport", "http"]

# Development stage
FROM base AS development
RUN pip install --no-cache-dir ".[dev]"
CMD ["pytest", "tests/", "-v"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  api:
    build:
      context: .
      target: production
    ports:
      - "8080:8080"
    environment:
      - DATABASE_URL=postgresql://hl:hl@postgres/humanlayer
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: hl
      POSTGRES_PASSWORD: hl
      POSTGRES_DB: humanlayer
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U hl"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  postgres_data:
  redis_data:
```

---

## INTENT MANIFEST

```yaml
INTENT_MANIFEST:
  RF:
    - RF-001: "Dockerfile multi-stage"
    - RF-002: "docker-compose para dev"
    - RF-003: "PostgreSQL para persistência"
    - RF-004: "Redis para cache"
    - RF-005: "Health checks configurados"

  INV:
    - INV-001: "Sem credenciais em Dockerfile"
    - INV-002: "Volumes para dados persistentes"
    - INV-003: "Health checks obrigatórios"
```

---

## GATES

```yaml
gates:
  G0_FILES_EXIST:
    validation: |
      ls Dockerfile
      ls docker-compose.yml

  G1_BUILD_WORKS:
    validation: "docker build -t humanlayer:test ."

  G2_COMPOSE_UP:
    validation: "docker compose up -d && docker compose ps"
```

---

## REFERÊNCIA

- `./S24_CONTEXT.md` - OSS Launch
- `./S26_CONTEXT.md` - Database Setup
