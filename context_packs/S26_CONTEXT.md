# S26 - database-setup | Context Pack v1.0

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
  id: S26
  name: database-setup
  title: "Database Setup"
  wave: W3-CloudMVP
  priority: P1-HIGH
  type: infrastructure

objective: "PostgreSQL + Alembic migrations"

dependencies:
  - S25  # Cloud Infrastructure

deliverables:
  - src/hl_mcp/cloud/db/__init__.py
  - src/hl_mcp/cloud/db/models.py
  - src/hl_mcp/cloud/db/session.py
  - alembic/
  - alembic.ini
```

---

## DATABASE SCHEMA

```yaml
tables:
  users:
    - id: UUID PK
    - email: VARCHAR UNIQUE
    - created_at: TIMESTAMP
    - tier: VARCHAR (free, starter, pro, business)

  organizations:
    - id: UUID PK
    - name: VARCHAR
    - owner_id: FK users
    - tier: VARCHAR

  validations:
    - id: UUID PK
    - org_id: FK organizations
    - agent_id: VARCHAR
    - action_type: VARCHAR
    - action_description: TEXT
    - decision: VARCHAR
    - veto_level: VARCHAR
    - created_at: TIMESTAMP

  findings:
    - id: UUID PK
    - validation_id: FK validations
    - layer_id: VARCHAR
    - severity: VARCHAR
    - title: VARCHAR
    - description: TEXT
```

---

## IMPLEMENTATION SPEC

### Database Models (db/models.py)

```python
"""SQLAlchemy models for Human Layer Cloud."""
from datetime import datetime
from uuid import uuid4
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Enum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    email = Column(String(255), unique=True, nullable=False)
    tier = Column(String(50), default="free")
    created_at = Column(DateTime, default=datetime.utcnow)

    organizations = relationship("Organization", back_populates="owner")


class Organization(Base):
    __tablename__ = "organizations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False)
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    tier = Column(String(50), default="free")
    created_at = Column(DateTime, default=datetime.utcnow)

    owner = relationship("User", back_populates="organizations")
    validations = relationship("Validation", back_populates="organization")


class Validation(Base):
    __tablename__ = "validations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    org_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"))
    agent_id = Column(String(255), nullable=False)
    action_type = Column(String(100), nullable=False)
    action_description = Column(Text, nullable=False)
    decision = Column(String(50), nullable=False)
    veto_level = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    organization = relationship("Organization", back_populates="validations")
    findings = relationship("Finding", back_populates="validation")


class Finding(Base):
    __tablename__ = "findings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    validation_id = Column(UUID(as_uuid=True), ForeignKey("validations.id"))
    layer_id = Column(String(10), nullable=False)
    severity = Column(String(20), nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)

    validation = relationship("Validation", back_populates="findings")
```

### Session Management (db/session.py)

```python
"""Database session management."""
import os
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://hl:hl@localhost/humanlayer")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@contextmanager
def get_db() -> Session:
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_async_db():
    """Async database dependency for FastAPI."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

---

## INTENT MANIFEST

```yaml
INTENT_MANIFEST:
  RF:
    - RF-001: "SQLAlchemy 2.0 models"
    - RF-002: "Alembic migrations"
    - RF-003: "UUID primary keys"
    - RF-004: "Relationships configuradas"

  INV:
    - INV-001: "DATABASE_URL via env var"
    - INV-002: "Session cleanup garantido"
    - INV-003: "Migrations reversíveis"
```

---

## GATES

```yaml
gates:
  G0_FILES_EXIST:
    validation: |
      ls src/hl_mcp/cloud/db/models.py
      ls alembic.ini

  G1_MIGRATIONS_WORK:
    validation: "alembic upgrade head"
```

---

## REFERÊNCIA

- `./S25_CONTEXT.md` - Cloud Infrastructure
- `./S27_CONTEXT.md` - Authentication
