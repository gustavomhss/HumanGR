# Human Layer MCP Server - Roadmap Block 03
# MCP RESOURCES + CONTEXT PACK BUILDER

> **Objetivo**: Definir Resources do MCP + Wizard para Context Packs
> **Versão**: 1.0.0 | Data: 2026-02-01

---

## 1. MCP RESOURCES

Resources são dados que o MCP expõe para leitura. Diferente de Tools (ações), Resources são **informações estáticas ou dinâmicas** que o cliente pode consultar.

### 1.1 Resource URIs

```yaml
# Formato: human_layer://{category}/{resource}[/{id}]

resources:

  # ===========================================================================
  # LAYER RESOURCES - Informações sobre os 7 layers
  # ===========================================================================
  human_layer://layers:
    description: "List all 7 Human Layers"
    returns: "Array of layer summaries"

  human_layer://layers/{layer_id}:
    description: "Get details of specific layer"
    example: "human_layer://layers/HL-1"
    returns: "Layer config, questions, red_flags"

  human_layer://layers/{layer_id}/template:
    description: "Get prompt template for layer"
    returns: "Jinja2 template for layer execution"

  # ===========================================================================
  # PERSPECTIVE RESOURCES - Informações sobre as 6 perspectivas
  # ===========================================================================
  human_layer://perspectives:
    description: "List all 6 perspectives"
    returns: "Array of perspective summaries"

  human_layer://perspectives/{perspective_id}:
    description: "Get perspective details"
    example: "human_layer://perspectives/tired_user"
    returns: "Persona, focus areas, test patterns, weight"

  human_layer://perspectives/intuition-templates:
    description: "Get all intuition templates"
    returns: "Templates for auth, forms, payment, etc."

  # ===========================================================================
  # STATISTICS RESOURCES - Métricas e estatísticas
  # ===========================================================================
  human_layer://stats/summary:
    description: "Get overall statistics"
    returns: "Total runs, pass rate, avg time, etc."

  human_layer://stats/layers:
    description: "Statistics per layer"
    returns: "Pass rate, veto rate, common findings per layer"

  human_layer://stats/perspectives:
    description: "Statistics per perspective"
    returns: "Tests generated, coverage, consensus rates"

  human_layer://stats/findings:
    description: "Finding distribution"
    returns: "By severity, by layer, by category"

  # ===========================================================================
  # TRUST/BUDGET RESOURCES - Estado cognitivo
  # ===========================================================================
  human_layer://trust/{agent_id}:
    description: "Trust score for specific agent"
    returns: "Current score, history, calibration status"

  human_layer://trust/leaderboard:
    description: "Trust scores ranked"
    returns: "Top/bottom agents by trust"

  human_layer://budget/status:
    description: "Current cognitive budget status"
    returns: "Remaining units, by category, fatigue level"

  human_layer://budget/history:
    description: "Budget usage history"
    returns: "Daily usage, trends, predictions"

  # ===========================================================================
  # CONFIGURATION RESOURCES - Configurações atuais
  # ===========================================================================
  human_layer://config/thresholds:
    description: "Current threshold configurations"
    returns: "Confidence, triage, consensus thresholds"

  human_layer://config/templates:
    description: "Available templates"
    returns: "Layer packs, journey templates, etc."

  # ===========================================================================
  # SESSION RESOURCES - Estado da sessão atual
  # ===========================================================================
  human_layer://session/current:
    description: "Current session state"
    returns: "Active validations, queue, history"

  human_layer://session/{session_id}/report:
    description: "Full report of a session"
    returns: "Complete HumanLayerReport"

  human_layer://session/{session_id}/artifacts:
    description: "Artifacts from session"
    returns: "Screenshots, videos, logs"
```

### 1.2 Implementação de Resources

```python
# src/hl_mcp/server/resources.py

from mcp.server import Server
from mcp.types import Resource, ResourceContents, TextResourceContents

async def register_resources(server: Server) -> None:
    """Register all Human Layer resources."""

    @server.list_resources()
    async def list_resources() -> list[Resource]:
        return [
            Resource(
                uri="human_layer://layers",
                name="Human Layers",
                description="List of all 7 Human Layers",
                mimeType="application/json",
            ),
            Resource(
                uri="human_layer://perspectives",
                name="Perspectives",
                description="List of all 6 testing perspectives",
                mimeType="application/json",
            ),
            Resource(
                uri="human_layer://stats/summary",
                name="Statistics Summary",
                description="Overall Human Layer statistics",
                mimeType="application/json",
            ),
            Resource(
                uri="human_layer://budget/status",
                name="Budget Status",
                description="Current cognitive budget status",
                mimeType="application/json",
            ),
            Resource(
                uri="human_layer://config/thresholds",
                name="Thresholds",
                description="Current threshold configurations",
                mimeType="application/json",
            ),
        ]

    @server.read_resource()
    async def read_resource(uri: str) -> ResourceContents:
        # Parse URI
        parts = uri.replace("human_layer://", "").split("/")

        if parts[0] == "layers":
            return await _read_layers_resource(parts[1:])
        elif parts[0] == "perspectives":
            return await _read_perspectives_resource(parts[1:])
        elif parts[0] == "stats":
            return await _read_stats_resource(parts[1:])
        elif parts[0] == "trust":
            return await _read_trust_resource(parts[1:])
        elif parts[0] == "budget":
            return await _read_budget_resource(parts[1:])
        elif parts[0] == "config":
            return await _read_config_resource(parts[1:])
        elif parts[0] == "session":
            return await _read_session_resource(parts[1:])

        raise ValueError(f"Unknown resource: {uri}")


async def _read_layers_resource(parts: list[str]) -> TextResourceContents:
    """Read layer resources."""
    from hl_mcp.config.layers import get_all_layers, get_layer

    if not parts:
        # List all layers
        layers = get_all_layers()
        return TextResourceContents(
            uri="human_layer://layers",
            mimeType="application/json",
            text=json.dumps([l.to_summary() for l in layers]),
        )

    layer_id = parts[0]
    layer = get_layer(layer_id)

    if len(parts) == 1:
        # Layer details
        return TextResourceContents(
            uri=f"human_layer://layers/{layer_id}",
            mimeType="application/json",
            text=json.dumps(layer.to_dict()),
        )

    if parts[1] == "template":
        # Layer prompt template
        return TextResourceContents(
            uri=f"human_layer://layers/{layer_id}/template",
            mimeType="text/plain",
            text=layer.get_prompt_template(),
        )

    raise ValueError(f"Unknown layer resource: {'/'.join(parts)}")
```

---

## 2. CONTEXT PACK BUILDER

O Context Pack Builder é um **wizard interativo** que ajuda usuários a configurar o contexto para validação do Human Layer.

### 2.1 O que é um Context Pack?

```yaml
# Um Context Pack contém tudo que o Human Layer precisa saber sobre o que vai testar

context_pack:
  # Identificação
  id: "cp_20260201_login_flow"
  name: "Login Flow Validation"
  created_at: "2026-02-01T10:30:00Z"

  # O que está sendo testado
  target:
    type: "user_journey"  # code, spec, config, doc, user_journey, api
    name: "User Login Flow"
    description: "Complete login flow from landing page to dashboard"

  # Contexto do produto/sistema
  system_context:
    product_name: "MyApp"
    domain: "fintech"
    user_base: "B2B"
    criticality: "high"
    compliance: ["SOC2", "GDPR"]

  # URLs e endpoints (para browser testing)
  endpoints:
    base_url: "https://app.mycompany.com"
    login_page: "/login"
    dashboard: "/dashboard"

  # Credenciais de teste (seguras)
  test_credentials:
    type: "test_account"  # Nunca prod!
    username_env: "TEST_USER"
    password_env: "TEST_PASS"

  # Quais layers executar
  layer_selection:
    pack: "FULL"  # FULL, SECURITY, USABILITY, MINIMAL, CUSTOM
    custom_layers: []  # Se pack=CUSTOM, lista de HL-1 a HL-7

  # Quais perspectivas usar
  perspective_selection:
    all: true
    custom: []  # Se all=false, lista de perspectivas

  # Journey (se type=user_journey)
  journey:
    happy_path:
      - step: "Navigate to login"
        action: "goto"
        target: "/login"
      - step: "Enter username"
        action: "fill"
        selector: "#username"
        value: "${TEST_USER}"
      - step: "Enter password"
        action: "fill"
        selector: "#password"
        value: "${TEST_PASS}"
      - step: "Click login"
        action: "click"
        selector: "#login-btn"
      - step: "Verify dashboard"
        action: "wait"
        selector: ".dashboard-container"

    generate_paths:
      errors: true
      edges: true
      personas: true

  # Opções de execução
  execution:
    record_video: true
    take_screenshots: true
    check_accessibility: true
    redundancy: 3
    timeout_per_layer: 120

  # Thresholds customizados (opcional)
  thresholds:
    confidence_auto_approve: 0.90
    confidence_human_required: 0.60
```

### 2.2 Context Pack Builder - Wizard Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      CONTEXT PACK BUILDER WIZARD                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STEP 1: What are you testing?                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ ○ Code (function, class, module)                                    │   │
│  │ ○ Specification (requirements, design doc)                          │   │
│  │ ○ Configuration (YAML, JSON, env vars)                              │   │
│  │ ● User Journey (UI flow, browser test)    ← Selected                │   │
│  │ ○ API Endpoint (REST, GraphQL)                                      │   │
│  │ ○ Documentation (README, guides)                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  STEP 2: Tell us about your system                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Product Name: [MyApp                    ]                           │   │
│  │ Domain:       [Fintech            ▼]                                │   │
│  │ User Base:    [B2B                ▼]                                │   │
│  │ Criticality:  ○ Low  ○ Medium  ● High                               │   │
│  │ Compliance:   ☑ SOC2  ☑ GDPR  ☐ HIPAA  ☐ PCI-DSS                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  STEP 3: Configure your journey                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Base URL: [https://app.mycompany.com    ]                           │   │
│  │                                                                      │   │
│  │ Happy Path Steps:                                                   │   │
│  │ ┌───────────────────────────────────────────────────────────────┐   │   │
│  │ │ 1. [goto    ▼] [/login            ] [              ]          │   │   │
│  │ │ 2. [fill    ▼] [#username         ] [${TEST_USER}  ]          │   │   │
│  │ │ 3. [fill    ▼] [#password         ] [${TEST_PASS}  ]          │   │   │
│  │ │ 4. [click   ▼] [#login-btn        ] [              ]          │   │   │
│  │ │ 5. [wait    ▼] [.dashboard        ] [              ]          │   │   │
│  │ │ [+ Add Step]                                                   │   │   │
│  │ └───────────────────────────────────────────────────────────────┘   │   │
│  │                                                                      │   │
│  │ Generate Additional Paths:                                          │   │
│  │ ☑ Error paths (network failure, validation errors)                  │   │
│  │ ☑ Edge cases (empty inputs, special chars)                          │   │
│  │ ☑ Persona paths (impatient, cautious, mobile user)                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  STEP 4: Select validation layers                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Layer Pack:                                                         │   │
│  │ ● FULL (all 7 layers) - Recommended for critical flows              │   │
│  │ ○ SECURITY (HL-4, HL-5, HL-6) - Security-focused                    │   │
│  │ ○ USABILITY (HL-1, HL-7) - UX-focused                               │   │
│  │ ○ MINIMAL (HL-4 only) - Quick strategic check                       │   │
│  │ ○ CUSTOM - Select specific layers                                   │   │
│  │                                                                      │   │
│  │ Selected Layers:                                                    │   │
│  │ ☑ HL-1 Usuario (WEAK)      ☑ HL-5 Seguranca (STRONG)               │   │
│  │ ☑ HL-2 Operador (MEDIUM)   ☑ HL-6 Hacker (STRONG)                  │   │
│  │ ☑ HL-3 Mantenedor (MEDIUM) ☑ HL-7 Simplificador (WEAK)             │   │
│  │ ☑ HL-4 Decisor (STRONG)                                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  STEP 5: Execution options                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ ☑ Record video of browser session                                   │   │
│  │ ☑ Take screenshots at each step                                     │   │
│  │ ☑ Check accessibility (a11y)                                        │   │
│  │                                                                      │   │
│  │ Redundancy: [3 ▼] runs per layer (for consensus)                    │   │
│  │ Timeout:    [120] seconds per layer                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  STEP 6: Review & Generate                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                      │   │
│  │  ✓ Testing: User Journey - Login Flow                               │   │
│  │  ✓ System: MyApp (Fintech, B2B, High criticality)                   │   │
│  │  ✓ Compliance: SOC2, GDPR                                           │   │
│  │  ✓ Journey: 5 steps + auto-generated paths                          │   │
│  │  ✓ Layers: FULL (7 layers)                                          │   │
│  │  ✓ Recording: Video + Screenshots + A11y                            │   │
│  │                                                                      │   │
│  │  Estimated time: 15-20 minutes                                      │   │
│  │  Estimated cost: ~$0.50 (LLM tokens)                                │   │
│  │                                                                      │   │
│  │  [Generate Context Pack]  [Save as Template]  [Cancel]              │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Context Pack Builder - MCP Implementation

```python
# src/hl_mcp/builder/context_pack_builder.py
# HLS-BLD-001

"""
Module: HLS-BLD-001 - ContextPackBuilder
========================================

Interactive wizard for building Context Packs.

Provides step-by-step guidance to help users create
complete context configurations for Human Layer validation.

Example:
    >>> builder = ContextPackBuilder()
    >>> pack = await builder.build_interactive()

Dependencies:
    - HLS-MDL-004: Journey models
    - HLS-CFG-002: Layer configs

Author: Human Layer Team
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime
import uuid


class TargetType(Enum):
    """Types of validation targets."""
    CODE = "code"
    SPEC = "spec"
    CONFIG = "config"
    USER_JOURNEY = "user_journey"
    API = "api"
    DOCUMENTATION = "documentation"


class LayerPack(Enum):
    """Pre-defined layer packs."""
    FULL = "FULL"           # All 7 layers
    SECURITY = "SECURITY"   # HL-4, HL-5, HL-6
    USABILITY = "USABILITY" # HL-1, HL-7
    MINIMAL = "MINIMAL"     # HL-4 only
    CUSTOM = "CUSTOM"       # User-selected


class Domain(Enum):
    """Business domains with specific concerns."""
    FINTECH = "fintech"
    HEALTHCARE = "healthcare"
    ECOMMERCE = "ecommerce"
    SAAS = "saas"
    ENTERPRISE = "enterprise"
    CONSUMER = "consumer"
    OTHER = "other"


class Compliance(Enum):
    """Compliance frameworks."""
    SOC2 = "SOC2"
    GDPR = "GDPR"
    HIPAA = "HIPAA"
    PCI_DSS = "PCI-DSS"
    ISO27001 = "ISO27001"
    NONE = "none"


@dataclass
class SystemContext:
    """Context about the system being tested."""
    product_name: str
    domain: Domain
    user_base: str  # B2B, B2C, Internal
    criticality: str  # low, medium, high
    compliance: List[Compliance] = field(default_factory=list)


@dataclass
class JourneyStep:
    """A single step in a user journey."""
    step: str           # Description
    action: str         # goto, click, fill, wait, screenshot
    target: str         # URL, selector, or value
    value: str = ""     # Value for fill actions


@dataclass
class JourneyConfig:
    """Configuration for journey-based testing."""
    base_url: str
    happy_path: List[JourneyStep]
    generate_error_paths: bool = True
    generate_edge_paths: bool = True
    generate_persona_paths: bool = True


@dataclass
class ExecutionConfig:
    """Execution options for validation."""
    record_video: bool = True
    take_screenshots: bool = True
    check_accessibility: bool = True
    redundancy: int = 3
    timeout_per_layer: int = 120


@dataclass
class ContextPack:
    """Complete context pack for Human Layer validation."""
    id: str
    name: str
    created_at: datetime

    target_type: TargetType
    target_name: str
    target_description: str

    system_context: SystemContext
    journey: Optional[JourneyConfig]

    layer_pack: LayerPack
    custom_layers: List[str]  # If layer_pack == CUSTOM

    perspectives_all: bool
    custom_perspectives: List[str]  # If perspectives_all == False

    execution: ExecutionConfig

    # Optional custom thresholds
    custom_thresholds: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "target": {
                "type": self.target_type.value,
                "name": self.target_name,
                "description": self.target_description,
            },
            "system_context": {
                "product_name": self.system_context.product_name,
                "domain": self.system_context.domain.value,
                "user_base": self.system_context.user_base,
                "criticality": self.system_context.criticality,
                "compliance": [c.value for c in self.system_context.compliance],
            },
            "journey": self._journey_to_dict() if self.journey else None,
            "layers": {
                "pack": self.layer_pack.value,
                "custom": self.custom_layers,
            },
            "perspectives": {
                "all": self.perspectives_all,
                "custom": self.custom_perspectives,
            },
            "execution": {
                "record_video": self.execution.record_video,
                "take_screenshots": self.execution.take_screenshots,
                "check_accessibility": self.execution.check_accessibility,
                "redundancy": self.execution.redundancy,
                "timeout_per_layer": self.execution.timeout_per_layer,
            },
            "custom_thresholds": self.custom_thresholds,
        }

    def _journey_to_dict(self) -> Dict[str, Any]:
        return {
            "base_url": self.journey.base_url,
            "happy_path": [
                {
                    "step": s.step,
                    "action": s.action,
                    "target": s.target,
                    "value": s.value,
                }
                for s in self.journey.happy_path
            ],
            "generate_paths": {
                "errors": self.journey.generate_error_paths,
                "edges": self.journey.generate_edge_paths,
                "personas": self.journey.generate_persona_paths,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextPack":
        """Deserialize from dictionary."""
        system_ctx = SystemContext(
            product_name=data["system_context"]["product_name"],
            domain=Domain(data["system_context"]["domain"]),
            user_base=data["system_context"]["user_base"],
            criticality=data["system_context"]["criticality"],
            compliance=[Compliance(c) for c in data["system_context"]["compliance"]],
        )

        journey = None
        if data.get("journey"):
            j = data["journey"]
            journey = JourneyConfig(
                base_url=j["base_url"],
                happy_path=[
                    JourneyStep(**step) for step in j["happy_path"]
                ],
                generate_error_paths=j["generate_paths"]["errors"],
                generate_edge_paths=j["generate_paths"]["edges"],
                generate_persona_paths=j["generate_paths"]["personas"],
            )

        execution = ExecutionConfig(**data["execution"])

        return cls(
            id=data["id"],
            name=data["name"],
            created_at=datetime.fromisoformat(data["created_at"]),
            target_type=TargetType(data["target"]["type"]),
            target_name=data["target"]["name"],
            target_description=data["target"]["description"],
            system_context=system_ctx,
            journey=journey,
            layer_pack=LayerPack(data["layers"]["pack"]),
            custom_layers=data["layers"]["custom"],
            perspectives_all=data["perspectives"]["all"],
            custom_perspectives=data["perspectives"]["custom"],
            execution=execution,
            custom_thresholds=data.get("custom_thresholds"),
        )


class ContextPackBuilder:
    """Interactive builder for Context Packs.

    Guides users through creating complete context configurations
    for Human Layer validation.

    Example:
        >>> builder = ContextPackBuilder()
        >>> # Step by step
        >>> builder.set_target(TargetType.USER_JOURNEY, "Login Flow")
        >>> builder.set_system_context(product_name="MyApp", domain=Domain.FINTECH)
        >>> builder.add_journey_step("goto", "/login")
        >>> pack = builder.build()
    """

    def __init__(self) -> None:
        """Initialize builder with defaults."""
        self._id = f"cp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self._name: str = ""
        self._target_type: Optional[TargetType] = None
        self._target_name: str = ""
        self._target_description: str = ""
        self._system_context: Optional[SystemContext] = None
        self._journey_base_url: str = ""
        self._journey_steps: List[JourneyStep] = []
        self._generate_error_paths: bool = True
        self._generate_edge_paths: bool = True
        self._generate_persona_paths: bool = True
        self._layer_pack: LayerPack = LayerPack.FULL
        self._custom_layers: List[str] = []
        self._perspectives_all: bool = True
        self._custom_perspectives: List[str] = []
        self._execution = ExecutionConfig()
        self._custom_thresholds: Optional[Dict[str, float]] = None

    def set_target(
        self,
        target_type: TargetType,
        name: str,
        description: str = "",
    ) -> "ContextPackBuilder":
        """Set what is being tested."""
        self._target_type = target_type
        self._target_name = name
        self._target_description = description
        self._name = name
        return self

    def set_system_context(
        self,
        product_name: str,
        domain: Domain,
        user_base: str = "B2C",
        criticality: str = "medium",
        compliance: Optional[List[Compliance]] = None,
    ) -> "ContextPackBuilder":
        """Set system context."""
        self._system_context = SystemContext(
            product_name=product_name,
            domain=domain,
            user_base=user_base,
            criticality=criticality,
            compliance=compliance or [],
        )
        return self

    def set_journey_base_url(self, url: str) -> "ContextPackBuilder":
        """Set base URL for journey."""
        self._journey_base_url = url
        return self

    def add_journey_step(
        self,
        action: str,
        target: str,
        value: str = "",
        description: str = "",
    ) -> "ContextPackBuilder":
        """Add a step to the journey."""
        step_num = len(self._journey_steps) + 1
        step_desc = description or f"Step {step_num}: {action} {target}"
        self._journey_steps.append(JourneyStep(
            step=step_desc,
            action=action,
            target=target,
            value=value,
        ))
        return self

    def set_path_generation(
        self,
        errors: bool = True,
        edges: bool = True,
        personas: bool = True,
    ) -> "ContextPackBuilder":
        """Configure path generation."""
        self._generate_error_paths = errors
        self._generate_edge_paths = edges
        self._generate_persona_paths = personas
        return self

    def set_layer_pack(
        self,
        pack: LayerPack,
        custom_layers: Optional[List[str]] = None,
    ) -> "ContextPackBuilder":
        """Set layer pack to use."""
        self._layer_pack = pack
        self._custom_layers = custom_layers or []
        return self

    def set_perspectives(
        self,
        all_perspectives: bool = True,
        custom: Optional[List[str]] = None,
    ) -> "ContextPackBuilder":
        """Configure perspectives."""
        self._perspectives_all = all_perspectives
        self._custom_perspectives = custom or []
        return self

    def set_execution(
        self,
        record_video: bool = True,
        take_screenshots: bool = True,
        check_accessibility: bool = True,
        redundancy: int = 3,
        timeout: int = 120,
    ) -> "ContextPackBuilder":
        """Configure execution options."""
        self._execution = ExecutionConfig(
            record_video=record_video,
            take_screenshots=take_screenshots,
            check_accessibility=check_accessibility,
            redundancy=redundancy,
            timeout_per_layer=timeout,
        )
        return self

    def set_custom_thresholds(
        self,
        thresholds: Dict[str, float],
    ) -> "ContextPackBuilder":
        """Set custom thresholds."""
        self._custom_thresholds = thresholds
        return self

    def validate(self) -> List[str]:
        """Validate the builder state. Returns list of errors."""
        errors = []

        if not self._target_type:
            errors.append("Target type is required")
        if not self._target_name:
            errors.append("Target name is required")
        if not self._system_context:
            errors.append("System context is required")

        if self._target_type == TargetType.USER_JOURNEY:
            if not self._journey_base_url:
                errors.append("Journey base URL is required for user journeys")
            if not self._journey_steps:
                errors.append("At least one journey step is required")

        if self._layer_pack == LayerPack.CUSTOM and not self._custom_layers:
            errors.append("Custom layers required when pack is CUSTOM")

        return errors

    def build(self) -> ContextPack:
        """Build the context pack."""
        errors = self.validate()
        if errors:
            raise ValueError(f"Invalid context pack: {', '.join(errors)}")

        journey = None
        if self._target_type == TargetType.USER_JOURNEY and self._journey_steps:
            journey = JourneyConfig(
                base_url=self._journey_base_url,
                happy_path=self._journey_steps,
                generate_error_paths=self._generate_error_paths,
                generate_edge_paths=self._generate_edge_paths,
                generate_persona_paths=self._generate_persona_paths,
            )

        return ContextPack(
            id=self._id,
            name=self._name,
            created_at=datetime.utcnow(),
            target_type=self._target_type,
            target_name=self._target_name,
            target_description=self._target_description,
            system_context=self._system_context,
            journey=journey,
            layer_pack=self._layer_pack,
            custom_layers=self._custom_layers,
            perspectives_all=self._perspectives_all,
            custom_perspectives=self._custom_perspectives,
            execution=self._execution,
            custom_thresholds=self._custom_thresholds,
        )

    def get_estimate(self) -> Dict[str, Any]:
        """Get time and cost estimate for this context pack."""
        # Base times per component (in seconds)
        layer_time = 30  # Per layer
        perspective_time = 10  # Per perspective
        journey_step_time = 5  # Per step

        # Count layers
        if self._layer_pack == LayerPack.FULL:
            num_layers = 7
        elif self._layer_pack == LayerPack.SECURITY:
            num_layers = 3
        elif self._layer_pack == LayerPack.USABILITY:
            num_layers = 2
        elif self._layer_pack == LayerPack.MINIMAL:
            num_layers = 1
        else:
            num_layers = len(self._custom_layers)

        # Count perspectives
        num_perspectives = 6 if self._perspectives_all else len(self._custom_perspectives)

        # Count journey steps
        num_steps = len(self._journey_steps)
        if self._generate_error_paths:
            num_steps *= 2
        if self._generate_edge_paths:
            num_steps *= 1.5
        if self._generate_persona_paths:
            num_steps *= 1.3

        # Calculate totals
        total_seconds = (
            (num_layers * layer_time * self._execution.redundancy) +
            (num_perspectives * perspective_time) +
            (num_steps * journey_step_time)
        )

        # Estimate cost (rough LLM token estimate)
        tokens_per_layer = 2000
        cost_per_1k_tokens = 0.01  # Approximate
        total_tokens = num_layers * tokens_per_layer * self._execution.redundancy
        total_cost = (total_tokens / 1000) * cost_per_1k_tokens

        return {
            "estimated_time_seconds": int(total_seconds),
            "estimated_time_human": self._format_time(total_seconds),
            "estimated_cost_usd": round(total_cost, 2),
            "breakdown": {
                "layers": num_layers,
                "perspectives": num_perspectives,
                "journey_paths": int(num_steps),
                "redundancy": self._execution.redundancy,
            },
        }

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds into human-readable string."""
        if seconds < 60:
            return f"{int(seconds)} seconds"
        elif seconds < 3600:
            mins = int(seconds / 60)
            return f"{mins}-{mins + 5} minutes"
        else:
            hours = int(seconds / 3600)
            return f"{hours}-{hours + 1} hours"


# Convenience function
def create_quick_pack(
    target_type: TargetType,
    name: str,
    product: str,
    domain: Domain,
    base_url: str = "",
    steps: Optional[List[Dict[str, str]]] = None,
) -> ContextPack:
    """Create a context pack quickly with minimal input.

    Args:
        target_type: What's being tested
        name: Name of the target
        product: Product name
        domain: Business domain
        base_url: Base URL (for journeys)
        steps: Journey steps as dicts

    Returns:
        Complete ContextPack
    """
    builder = ContextPackBuilder()
    builder.set_target(target_type, name)
    builder.set_system_context(product, domain)

    if target_type == TargetType.USER_JOURNEY and base_url:
        builder.set_journey_base_url(base_url)
        for step in (steps or []):
            builder.add_journey_step(
                action=step.get("action", "goto"),
                target=step.get("target", ""),
                value=step.get("value", ""),
            )

    return builder.build()


# Export
__all__ = [
    "ContextPackBuilder",
    "ContextPack",
    "TargetType",
    "LayerPack",
    "Domain",
    "Compliance",
    "SystemContext",
    "JourneyStep",
    "JourneyConfig",
    "ExecutionConfig",
    "create_quick_pack",
]
```

### 2.4 MCP Tools para Context Pack Builder

```python
# Adicionar em src/hl_mcp/server/tools.py

# Context Pack Builder Tools
Tool(
    name="human_layer.builder.start",
    description="Start building a new Context Pack",
    inputSchema={
        "type": "object",
        "properties": {
            "target_type": {
                "type": "string",
                "enum": ["code", "spec", "config", "user_journey", "api", "documentation"],
            },
            "name": {"type": "string"},
            "description": {"type": "string"},
        },
        "required": ["target_type", "name"],
    },
),

Tool(
    name="human_layer.builder.set_system",
    description="Set system context for the pack",
    inputSchema={
        "type": "object",
        "properties": {
            "product_name": {"type": "string"},
            "domain": {
                "type": "string",
                "enum": ["fintech", "healthcare", "ecommerce", "saas", "enterprise", "consumer", "other"],
            },
            "user_base": {"type": "string", "enum": ["B2B", "B2C", "Internal"]},
            "criticality": {"type": "string", "enum": ["low", "medium", "high"]},
            "compliance": {
                "type": "array",
                "items": {"type": "string", "enum": ["SOC2", "GDPR", "HIPAA", "PCI-DSS", "ISO27001"]},
            },
        },
        "required": ["product_name", "domain"],
    },
),

Tool(
    name="human_layer.builder.add_journey_step",
    description="Add a step to the journey",
    inputSchema={
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["goto", "click", "fill", "wait", "screenshot"]},
            "target": {"type": "string", "description": "URL, selector, or element"},
            "value": {"type": "string", "description": "Value for fill actions"},
            "description": {"type": "string"},
        },
        "required": ["action", "target"],
    },
),

Tool(
    name="human_layer.builder.set_layers",
    description="Configure which layers to use",
    inputSchema={
        "type": "object",
        "properties": {
            "pack": {"type": "string", "enum": ["FULL", "SECURITY", "USABILITY", "MINIMAL", "CUSTOM"]},
            "custom_layers": {
                "type": "array",
                "items": {"type": "string", "enum": ["HL-1", "HL-2", "HL-3", "HL-4", "HL-5", "HL-6", "HL-7"]},
            },
        },
        "required": ["pack"],
    },
),

Tool(
    name="human_layer.builder.get_estimate",
    description="Get time and cost estimate for current pack",
    inputSchema={"type": "object", "properties": {}},
),

Tool(
    name="human_layer.builder.build",
    description="Build and return the complete Context Pack",
    inputSchema={"type": "object", "properties": {}},
),

Tool(
    name="human_layer.builder.from_template",
    description="Create Context Pack from a template",
    inputSchema={
        "type": "object",
        "properties": {
            "template": {
                "type": "string",
                "enum": ["login_flow", "checkout_flow", "signup_flow", "api_crud", "config_change"],
            },
            "customizations": {"type": "object"},
        },
        "required": ["template"],
    },
),
```

---

## 3. TEMPLATES PRÉ-DEFINIDOS

### 3.1 Template: Login Flow

```yaml
# configs/templates/login_flow.yaml

template:
  id: "login_flow"
  name: "Login Flow Validation"
  description: "Complete user authentication flow"

  defaults:
    target_type: "user_journey"
    layer_pack: "FULL"
    criticality: "high"
    compliance: ["SOC2"]

  journey_template:
    steps:
      - action: "goto"
        target: "${base_url}/login"
        description: "Navigate to login page"

      - action: "wait"
        target: "#login-form"
        description: "Wait for login form"

      - action: "fill"
        target: "#username, #email, input[name='username'], input[name='email']"
        value: "${test_username}"
        description: "Enter username/email"

      - action: "fill"
        target: "#password, input[name='password'], input[type='password']"
        value: "${test_password}"
        description: "Enter password"

      - action: "click"
        target: "#login-btn, button[type='submit'], .login-button"
        description: "Click login button"

      - action: "wait"
        target: ".dashboard, .home, .main-content"
        description: "Verify successful login"

  required_inputs:
    - name: "base_url"
      description: "Base URL of your application"
      example: "https://app.example.com"

    - name: "test_username"
      description: "Test account username or email"
      example: "test@example.com"

    - name: "test_password"
      description: "Test account password"
      example: "${TEST_PASSWORD}"
      sensitive: true

  suggested_error_paths:
    - "Invalid credentials"
    - "Account locked"
    - "2FA required"
    - "Session expired"
    - "Network timeout"

  suggested_edge_cases:
    - "Empty username"
    - "Empty password"
    - "SQL injection attempt"
    - "XSS in username"
    - "Very long password"
    - "Unicode characters"
```

### 3.2 Template: Checkout Flow

```yaml
# configs/templates/checkout_flow.yaml

template:
  id: "checkout_flow"
  name: "E-commerce Checkout Flow"
  description: "Complete purchase flow from cart to confirmation"

  defaults:
    target_type: "user_journey"
    layer_pack: "FULL"
    criticality: "high"
    compliance: ["PCI-DSS", "GDPR"]

  journey_template:
    steps:
      - action: "goto"
        target: "${base_url}/cart"
        description: "View shopping cart"

      - action: "click"
        target: ".checkout-btn, #checkout"
        description: "Proceed to checkout"

      - action: "fill"
        target: "#shipping-address"
        value: "${test_address}"
        description: "Enter shipping address"

      - action: "fill"
        target: "#card-number"
        value: "${test_card}"
        description: "Enter payment card"

      - action: "click"
        target: "#place-order"
        description: "Place order"

      - action: "wait"
        target: ".order-confirmation"
        description: "Verify order confirmation"

  required_inputs:
    - name: "base_url"
    - name: "test_address"
    - name: "test_card"
      sensitive: true

  suggested_error_paths:
    - "Payment declined"
    - "Out of stock"
    - "Invalid address"
    - "Session timeout during payment"
```

### 3.3 Template: API CRUD

```yaml
# configs/templates/api_crud.yaml

template:
  id: "api_crud"
  name: "API CRUD Operations"
  description: "Test Create, Read, Update, Delete API operations"

  defaults:
    target_type: "api"
    layer_pack: "SECURITY"  # Focus on security for APIs
    criticality: "medium"

  api_template:
    endpoints:
      create:
        method: "POST"
        path: "${base_path}"
        body: "${create_payload}"
        expected_status: 201

      read:
        method: "GET"
        path: "${base_path}/${id}"
        expected_status: 200

      update:
        method: "PUT"
        path: "${base_path}/${id}"
        body: "${update_payload}"
        expected_status: 200

      delete:
        method: "DELETE"
        path: "${base_path}/${id}"
        expected_status: 204

  required_inputs:
    - name: "base_path"
      example: "/api/v1/users"
    - name: "create_payload"
    - name: "update_payload"

  suggested_edge_cases:
    - "Invalid JSON body"
    - "Missing required fields"
    - "Invalid field types"
    - "Unauthorized access"
    - "Non-existent resource"
    - "Concurrent modifications"
```

---

## 4. MÓDULOS IMPLEMENTADOS NESTE BLOCK

### 4.1 Arquivos Criados

| ID | Arquivo | Status |
|----|---------|--------|
| HLS-BLD-001 | `src/hl_mcp/builder/context_pack_builder.py` | Definido |
| HLS-BLD-002 | `src/hl_mcp/builder/templates.py` | Definido |
| HLS-SRV-003 | `src/hl_mcp/server/resources.py` | Definido |

### 4.2 MCP Tools Adicionados

- `human_layer.builder.start`
- `human_layer.builder.set_system`
- `human_layer.builder.add_journey_step`
- `human_layer.builder.set_layers`
- `human_layer.builder.get_estimate`
- `human_layer.builder.build`
- `human_layer.builder.from_template`

### 4.3 MCP Resources Definidos

- `human_layer://layers`
- `human_layer://layers/{id}`
- `human_layer://perspectives`
- `human_layer://stats/*`
- `human_layer://trust/*`
- `human_layer://budget/*`
- `human_layer://config/*`
- `human_layer://session/*`

---

## PRÓXIMO BLOCK

O Block 04 vai cobrir:
1. Implementação do primeiro módulo real (HLS-MDL-007: Enums)
2. Implementação dos Data Models (HLS-MDL-001 a HLS-MDL-006)
3. Testes unitários para os models
4. Schema validation

---

*ROADMAP_BLOCK_03_RESOURCES_AND_BUILDER.md - v1.0.0 - 2026-02-01*
