# Human Layer MCP Server - Roadmap Block 02
# ARQUITETURA HIPER MODULARIZADA

> **Princípio**: Peças como Legos. Cada peça independente, documentada, indexada.
> **Objetivo**: Manutenção FACILÍSSIMA. Encontrar qualquer coisa em segundos.
> **Versão**: 1.0.0 | Data: 2026-02-01

---

## FILOSOFIA: LEGO ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LEGO ARCHITECTURE PRINCIPLES                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. SINGLE RESPONSIBILITY                                                   │
│     Cada arquivo faz UMA coisa. Máximo 200-300 linhas.                     │
│                                                                             │
│  2. EXPLICIT INTERFACES                                                     │
│     Toda peça expõe interface clara via __init__.py                        │
│                                                                             │
│  3. ZERO HIDDEN DEPENDENCIES                                                │
│     Dependências declaradas no topo do arquivo                             │
│                                                                             │
│  4. SELF-DOCUMENTING                                                        │
│     Docstring + type hints + examples em todo lugar                        │
│                                                                             │
│  5. INDEXED CATALOG                                                         │
│     LEGO_INDEX.yaml com mapa de todos os módulos                          │
│                                                                             │
│  6. PLUG-AND-PLAY                                                           │
│     Trocar implementação sem quebrar nada                                  │
│                                                                             │
│  7. TESTABLE IN ISOLATION                                                   │
│     Cada peça tem seus próprios testes                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## ESTRUTURA DE PASTAS

```
hl-mcp/
│
├── LEGO_INDEX.yaml              # ÍNDICE MESTRE (encontre qualquer coisa)
├── README.md                    # Getting started
├── pyproject.toml               # Dependencies
│
├── src/
│   └── hl_mcp/
│       │
│       ├── __init__.py          # Public API exports
│       │
│       ├── server/              # MCP Server Layer
│       │   ├── __init__.py
│       │   ├── mcp_server.py    # Main MCP server (HLS-SRV-001)
│       │   ├── tools.py         # Tool definitions (HLS-SRV-002)
│       │   ├── resources.py     # Resource definitions (HLS-SRV-003)
│       │   └── prompts.py       # Prompt templates (HLS-SRV-004)
│       │
│       ├── core/                # Core Engine
│       │   ├── __init__.py
│       │   ├── runner.py        # HumanLayerRunner (HLS-CORE-001)
│       │   ├── orchestrator.py  # CognitiveOrchestrator (HLS-CORE-002)
│       │   ├── gate.py          # HumanLayerGate (HLS-CORE-003)
│       │   └── session.py       # Session management (HLS-CORE-004)
│       │
│       ├── layers/              # 7 Human Layers (cada um = 1 arquivo)
│       │   ├── __init__.py      # Exports all layers
│       │   ├── base.py          # BaseLayer interface (HLS-LYR-000)
│       │   ├── hl1_usuario.py   # HL-1 Usuario (HLS-LYR-001)
│       │   ├── hl2_operador.py  # HL-2 Operador (HLS-LYR-002)
│       │   ├── hl3_mantenedor.py # HL-3 Mantenedor (HLS-LYR-003)
│       │   ├── hl4_decisor.py   # HL-4 Decisor (HLS-LYR-004)
│       │   ├── hl5_seguranca.py # HL-5 Seguranca (HLS-LYR-005)
│       │   ├── hl6_hacker.py    # HL-6 Hacker (HLS-LYR-006)
│       │   └── hl7_simplificador.py # HL-7 Simplificador (HLS-LYR-007)
│       │
│       ├── perspectives/        # 6 Perspective Tests
│       │   ├── __init__.py
│       │   ├── base.py          # BasePerspective (HLS-PERSP-000)
│       │   ├── tired_user.py    # (HLS-PERSP-001)
│       │   ├── malicious_insider.py # (HLS-PERSP-002)
│       │   ├── confused_newbie.py # (HLS-PERSP-003)
│       │   ├── power_user.py    # (HLS-PERSP-004)
│       │   ├── auditor.py       # (HLS-PERSP-005)
│       │   ├── 3am_operator.py  # (HLS-PERSP-006)
│       │   ├── generator.py     # TestGenerator (HLS-PERSP-010)
│       │   └── consensus.py     # ConsensusEngine (HLS-PERSP-011)
│       │
│       ├── cognitive/           # Cognitive Modules
│       │   ├── __init__.py
│       │   ├── budget.py        # CognitiveBudgetManager (HLS-COG-001)
│       │   ├── trust.py         # TrustCalibration (HLS-COG-002)
│       │   ├── triage.py        # PredictiveTriage (HLS-COG-003)
│       │   ├── feedback.py      # FeedbackLearner (HLS-COG-004)
│       │   ├── confidence.py    # ConfidenceScorer (HLS-COG-005)
│       │   └── humanizer.py     # HumanizedErrorGenerator (HLS-COG-006)
│       │
│       ├── browser/             # Browser Automation
│       │   ├── __init__.py
│       │   ├── driver.py        # BrowserDriver (HLS-BRW-001)
│       │   ├── actions.py       # BrowserActions (HLS-BRW-002)
│       │   ├── screenshot.py    # ScreenshotManager (HLS-BRW-003)
│       │   ├── video.py         # VideoRecorder (HLS-BRW-004)
│       │   ├── accessibility.py # AccessibilityChecker (HLS-BRW-005)
│       │   └── journey.py       # JourneyExecutor (HLS-BRW-006)
│       │
│       ├── journeys/            # Journey Generation
│       │   ├── __init__.py
│       │   ├── completer.py     # JourneyCompleter (HLS-JRN-001)
│       │   ├── error_paths.py   # ErrorPathGenerator (HLS-JRN-002)
│       │   ├── edge_paths.py    # EdgePathGenerator (HLS-JRN-003)
│       │   ├── persona_paths.py # PersonaPathGenerator (HLS-JRN-004)
│       │   └── templates.py     # JourneyTemplates (HLS-JRN-005)
│       │
│       ├── models/              # Data Models (Contracts)
│       │   ├── __init__.py
│       │   ├── finding.py       # Finding, SecurityFinding (HLS-MDL-001)
│       │   ├── layer_result.py  # LayerResult (HLS-MDL-002)
│       │   ├── report.py        # HumanLayerReport (HLS-MDL-003)
│       │   ├── journey.py       # Journey, JourneyResult (HLS-MDL-004)
│       │   ├── test.py          # GeneratedTest, TestConsensus (HLS-MDL-005)
│       │   ├── review.py        # ReviewItem, ReviewSession (HLS-MDL-006)
│       │   └── enums.py         # All enums (HLS-MDL-007)
│       │
│       ├── llm/                 # LLM Integration
│       │   ├── __init__.py
│       │   ├── client.py        # LLMClient interface (HLS-LLM-001)
│       │   ├── claude.py        # Claude implementation (HLS-LLM-002)
│       │   ├── openai.py        # OpenAI implementation (HLS-LLM-003)
│       │   ├── prompts.py       # PromptTemplates (HLS-LLM-004)
│       │   └── parser.py        # ResponseParser (HLS-LLM-005)
│       │
│       ├── config/              # Configuration
│       │   ├── __init__.py
│       │   ├── settings.py      # Global settings (HLS-CFG-001)
│       │   ├── layers.py        # Layer configs (HLS-CFG-002)
│       │   ├── perspectives.py  # Perspective configs (HLS-CFG-003)
│       │   ├── thresholds.py    # Threshold configs (HLS-CFG-004)
│       │   └── templates.py     # Template configs (HLS-CFG-005)
│       │
│       ├── storage/             # State Storage
│       │   ├── __init__.py
│       │   ├── interface.py     # StorageInterface (HLS-STR-001)
│       │   ├── memory.py        # InMemoryStorage (HLS-STR-002)
│       │   ├── sqlite.py        # SQLiteStorage (HLS-STR-003)
│       │   └── redis.py         # RedisStorage (HLS-STR-004)
│       │
│       └── utils/               # Utilities
│           ├── __init__.py
│           ├── hashing.py       # Hash utilities (HLS-UTL-001)
│           ├── timing.py        # Timing utilities (HLS-UTL-002)
│           ├── logging.py       # Logging setup (HLS-UTL-003)
│           └── validation.py    # Validation helpers (HLS-UTL-004)
│
├── tests/                       # Tests (mirror src structure)
│   ├── unit/
│   │   ├── core/
│   │   ├── layers/
│   │   ├── perspectives/
│   │   ├── cognitive/
│   │   ├── browser/
│   │   ├── journeys/
│   │   └── models/
│   ├── integration/
│   └── e2e/
│
├── configs/                     # Configuration files
│   ├── default.yaml
│   ├── layers/
│   │   ├── hl1_usuario.yaml
│   │   ├── hl2_operador.yaml
│   │   └── ...
│   └── perspectives/
│       ├── tired_user.yaml
│       └── ...
│
├── templates/                   # Prompt templates
│   ├── layers/
│   └── perspectives/
│
└── docs/                        # Documentation
    ├── LEGO_CATALOG.md          # Full module catalog
    ├── api/                     # API docs
    ├── guides/                  # User guides
    └── examples/                # Usage examples
```

---

## LEGO_INDEX.yaml - ÍNDICE MESTRE

```yaml
# LEGO_INDEX.yaml
# Master index of all Human Layer MCP modules
# Use: Ctrl+F para encontrar qualquer módulo em segundos

version: "1.0.0"
updated: "2026-02-01"

# ============================================================================
# NOMENCLATURA: HLS-{AREA}-{NUMBER}
# ============================================================================
# HLS = Human Layer Server
# AREA = SRV (server), CORE, LYR (layer), PERSP (perspective),
#        COG (cognitive), BRW (browser), JRN (journey), MDL (model),
#        LLM, CFG (config), STR (storage), UTL (utility)

modules:

  # ==========================================================================
  # SERVER LAYER
  # ==========================================================================
  HLS-SRV-001:
    name: "MCPServer"
    file: "src/hl_mcp/server/mcp_server.py"
    description: "Main MCP server entry point"
    exports: ["HumanLayerMCPServer", "run_server"]
    depends_on: ["HLS-SRV-002", "HLS-SRV-003", "HLS-CORE-001"]

  HLS-SRV-002:
    name: "ToolDefinitions"
    file: "src/hl_mcp/server/tools.py"
    description: "MCP tool definitions"
    exports: ["TOOLS", "register_tools"]
    depends_on: []

  HLS-SRV-003:
    name: "ResourceDefinitions"
    file: "src/hl_mcp/server/resources.py"
    description: "MCP resource definitions"
    exports: ["RESOURCES", "register_resources"]
    depends_on: []

  HLS-SRV-004:
    name: "PromptDefinitions"
    file: "src/hl_mcp/server/prompts.py"
    description: "MCP prompt templates"
    exports: ["PROMPTS", "register_prompts"]
    depends_on: []

  # ==========================================================================
  # CORE ENGINE
  # ==========================================================================
  HLS-CORE-001:
    name: "HumanLayerRunner"
    file: "src/hl_mcp/core/runner.py"
    description: "Main execution engine for 7 layers"
    exports: ["HumanLayerRunner", "run_all_layers", "run_single_layer"]
    depends_on: ["HLS-LYR-*", "HLS-MDL-002", "HLS-MDL-003"]

  HLS-CORE-002:
    name: "CognitiveOrchestrator"
    file: "src/hl_mcp/core/orchestrator.py"
    description: "Central coordinator for reviews"
    exports: ["CognitiveOrchestrator", "schedule_reviews"]
    depends_on: ["HLS-COG-001", "HLS-COG-002", "HLS-COG-003"]

  HLS-CORE-003:
    name: "HumanLayerGate"
    file: "src/hl_mcp/core/gate.py"
    description: "Validation gate with veto logic"
    exports: ["HumanLayerGate", "run_gate", "HumanLayerResult"]
    depends_on: ["HLS-CORE-001", "HLS-MDL-001"]

  HLS-CORE-004:
    name: "SessionManager"
    file: "src/hl_mcp/core/session.py"
    description: "Session state management"
    exports: ["SessionManager", "Session"]
    depends_on: ["HLS-STR-001"]

  # ==========================================================================
  # 7 HUMAN LAYERS
  # ==========================================================================
  HLS-LYR-000:
    name: "BaseLayer"
    file: "src/hl_mcp/layers/base.py"
    description: "Base interface for all layers"
    exports: ["BaseLayer", "LayerConfig", "LayerContext"]
    depends_on: ["HLS-MDL-002", "HLS-LLM-001"]

  HLS-LYR-001:
    name: "HL1Usuario"
    file: "src/hl_mcp/layers/hl1_usuario.py"
    description: "Usability validation layer"
    exports: ["HL1Usuario"]
    veto_power: "WEAK"
    focus: ["usability", "UX", "friction"]
    depends_on: ["HLS-LYR-000"]

  HLS-LYR-002:
    name: "HL2Operador"
    file: "src/hl_mcp/layers/hl2_operador.py"
    description: "Operability validation layer"
    exports: ["HL2Operador"]
    veto_power: "MEDIUM"
    focus: ["diagnostics", "runbooks", "rollback"]
    depends_on: ["HLS-LYR-000"]

  HLS-LYR-003:
    name: "HL3Mantenedor"
    file: "src/hl_mcp/layers/hl3_mantenedor.py"
    description: "Maintainability validation layer"
    exports: ["HL3Mantenedor"]
    veto_power: "MEDIUM"
    focus: ["code_quality", "tests", "docs"]
    depends_on: ["HLS-LYR-000"]

  HLS-LYR-004:
    name: "HL4Decisor"
    file: "src/hl_mcp/layers/hl4_decisor.py"
    description: "Strategic alignment layer"
    exports: ["HL4Decisor"]
    veto_power: "STRONG"
    focus: ["strategy", "ROI", "trust"]
    depends_on: ["HLS-LYR-000"]

  HLS-LYR-005:
    name: "HL5Seguranca"
    file: "src/hl_mcp/layers/hl5_seguranca.py"
    description: "Accidental security layer"
    exports: ["HL5Seguranca"]
    veto_power: "STRONG"
    focus: ["safety", "data_protection", "fail_safe"]
    depends_on: ["HLS-LYR-000"]

  HLS-LYR-006:
    name: "HL6Hacker"
    file: "src/hl_mcp/layers/hl6_hacker.py"
    description: "Adversarial thinking layer"
    exports: ["HL6Hacker"]
    veto_power: "STRONG"
    focus: ["exploits", "attack_vectors", "abuse"]
    depends_on: ["HLS-LYR-000"]

  HLS-LYR-007:
    name: "HL7Simplificador"
    file: "src/hl_mcp/layers/hl7_simplificador.py"
    description: "Simplification layer"
    exports: ["HL7Simplificador"]
    veto_power: "WEAK"
    focus: ["YAGNI", "complexity", "minimalism"]
    depends_on: ["HLS-LYR-000"]

  # ==========================================================================
  # 6 PERSPECTIVES
  # ==========================================================================
  HLS-PERSP-000:
    name: "BasePerspective"
    file: "src/hl_mcp/perspectives/base.py"
    description: "Base interface for perspectives"
    exports: ["BasePerspective", "PerspectiveConfig"]
    depends_on: ["HLS-MDL-005"]

  HLS-PERSP-001:
    name: "TiredUser"
    file: "src/hl_mcp/perspectives/tired_user.py"
    description: "User with low patience perspective"
    exports: ["TiredUserPerspective"]
    weight: 1.0
    depends_on: ["HLS-PERSP-000"]

  HLS-PERSP-002:
    name: "MaliciousInsider"
    file: "src/hl_mcp/perspectives/malicious_insider.py"
    description: "Employee with bad intentions"
    exports: ["MaliciousInsiderPerspective"]
    weight: 1.5  # Security = higher weight
    depends_on: ["HLS-PERSP-000"]

  HLS-PERSP-003:
    name: "ConfusedNewbie"
    file: "src/hl_mcp/perspectives/confused_newbie.py"
    description: "First-time user perspective"
    exports: ["ConfusedNewbiePerspective"]
    weight: 1.0
    depends_on: ["HLS-PERSP-000"]

  HLS-PERSP-004:
    name: "PowerUser"
    file: "src/hl_mcp/perspectives/power_user.py"
    description: "Expert user perspective"
    exports: ["PowerUserPerspective"]
    weight: 0.8
    depends_on: ["HLS-PERSP-000"]

  HLS-PERSP-005:
    name: "Auditor"
    file: "src/hl_mcp/perspectives/auditor.py"
    description: "Compliance officer perspective"
    exports: ["AuditorPerspective"]
    weight: 1.2
    depends_on: ["HLS-PERSP-000"]

  HLS-PERSP-006:
    name: "3AMOperator"
    file: "src/hl_mcp/perspectives/3am_operator.py"
    description: "On-call engineer at 3am"
    exports: ["ThreeAMOperatorPerspective"]
    weight: 1.0
    depends_on: ["HLS-PERSP-000"]

  HLS-PERSP-010:
    name: "TestGenerator"
    file: "src/hl_mcp/perspectives/generator.py"
    description: "Multi-perspective test generator"
    exports: ["PerspectiveTestGenerator", "generate_tests"]
    depends_on: ["HLS-PERSP-001:006", "HLS-MDL-005"]

  HLS-PERSP-011:
    name: "ConsensusEngine"
    file: "src/hl_mcp/perspectives/consensus.py"
    description: "Multi-perspective consensus"
    exports: ["ConsensusEngine", "calculate_consensus"]
    depends_on: ["HLS-PERSP-010"]

  # ==========================================================================
  # COGNITIVE MODULES
  # ==========================================================================
  HLS-COG-001:
    name: "CognitiveBudgetManager"
    file: "src/hl_mcp/cognitive/budget.py"
    description: "Review budget management"
    exports: ["CognitiveBudgetManager", "AttentionUnit", "BudgetStatus"]
    depends_on: ["HLS-CFG-004"]

  HLS-COG-002:
    name: "TrustCalibration"
    file: "src/hl_mcp/cognitive/trust.py"
    description: "Dynamic trust scoring"
    exports: ["TrustCalibrationSystem", "TrustScore", "FeedbackType"]
    depends_on: ["HLS-STR-001"]

  HLS-COG-003:
    name: "PredictiveTriage"
    file: "src/hl_mcp/cognitive/triage.py"
    description: "ML-based review triage"
    exports: ["PredictiveTriage", "TriageDecision", "HITLDecision"]
    depends_on: ["HLS-COG-002"]

  HLS-COG-004:
    name: "FeedbackLearner"
    file: "src/hl_mcp/cognitive/feedback.py"
    description: "Bi-directional learning"
    exports: ["FeedbackLearner", "HumanFeedback", "LearnedPattern"]
    depends_on: ["HLS-STR-001"]

  HLS-COG-005:
    name: "ConfidenceScorer"
    file: "src/hl_mcp/cognitive/confidence.py"
    description: "Multi-dimensional confidence scoring"
    exports: ["ConfidenceScorer", "ScoredResult", "ReviewTrigger"]
    depends_on: ["HLS-CFG-004"]

  HLS-COG-006:
    name: "HumanizedErrorGenerator"
    file: "src/hl_mcp/cognitive/humanizer.py"
    description: "Technical to human error conversion"
    exports: ["HumanizedErrorGenerator", "HumanizedError", "RecipientType"]
    depends_on: []

  # ==========================================================================
  # BROWSER AUTOMATION
  # ==========================================================================
  HLS-BRW-001:
    name: "BrowserDriver"
    file: "src/hl_mcp/browser/driver.py"
    description: "Playwright browser management"
    exports: ["BrowserDriver", "BrowserContext"]
    depends_on: []  # External: playwright

  HLS-BRW-002:
    name: "BrowserActions"
    file: "src/hl_mcp/browser/actions.py"
    description: "Browser action primitives"
    exports: ["BrowserActions", "click", "fill", "goto", "wait"]
    depends_on: ["HLS-BRW-001"]

  HLS-BRW-003:
    name: "ScreenshotManager"
    file: "src/hl_mcp/browser/screenshot.py"
    description: "Screenshot capture and management"
    exports: ["ScreenshotManager", "take_screenshot"]
    depends_on: ["HLS-BRW-001"]

  HLS-BRW-004:
    name: "VideoRecorder"
    file: "src/hl_mcp/browser/video.py"
    description: "Session video recording"
    exports: ["VideoRecorder", "start_recording", "stop_recording"]
    depends_on: ["HLS-BRW-001"]

  HLS-BRW-005:
    name: "AccessibilityChecker"
    file: "src/hl_mcp/browser/accessibility.py"
    description: "a11y validation"
    exports: ["AccessibilityChecker", "AccessibilityResult"]
    depends_on: ["HLS-BRW-001"]

  HLS-BRW-006:
    name: "JourneyExecutor"
    file: "src/hl_mcp/browser/journey.py"
    description: "Journey execution engine"
    exports: ["JourneyExecutor", "execute_journey"]
    depends_on: ["HLS-BRW-001:005", "HLS-MDL-004"]

  # ==========================================================================
  # JOURNEY GENERATION
  # ==========================================================================
  HLS-JRN-001:
    name: "JourneyCompleter"
    file: "src/hl_mcp/journeys/completer.py"
    description: "Complete journeys beyond happy path"
    exports: ["JourneyCompleter", "complete_journey"]
    depends_on: ["HLS-JRN-002:004"]

  HLS-JRN-002:
    name: "ErrorPathGenerator"
    file: "src/hl_mcp/journeys/error_paths.py"
    description: "Generate error path variations"
    exports: ["ErrorPathGenerator", "generate_error_paths"]
    depends_on: ["HLS-JRN-005"]

  HLS-JRN-003:
    name: "EdgePathGenerator"
    file: "src/hl_mcp/journeys/edge_paths.py"
    description: "Generate edge case paths"
    exports: ["EdgePathGenerator", "generate_edge_paths"]
    depends_on: ["HLS-JRN-005"]

  HLS-JRN-004:
    name: "PersonaPathGenerator"
    file: "src/hl_mcp/journeys/persona_paths.py"
    description: "Generate persona-specific paths"
    exports: ["PersonaPathGenerator", "generate_persona_paths"]
    depends_on: ["HLS-JRN-005", "HLS-PERSP-*"]

  HLS-JRN-005:
    name: "JourneyTemplates"
    file: "src/hl_mcp/journeys/templates.py"
    description: "Journey template definitions"
    exports: ["TEMPLATES", "get_template"]
    depends_on: []

  # ==========================================================================
  # DATA MODELS
  # ==========================================================================
  HLS-MDL-001:
    name: "Finding"
    file: "src/hl_mcp/models/finding.py"
    description: "Finding and SecurityFinding models"
    exports: ["Finding", "SecurityFinding", "Severity"]
    depends_on: ["HLS-MDL-007"]

  HLS-MDL-002:
    name: "LayerResult"
    file: "src/hl_mcp/models/layer_result.py"
    description: "Layer execution result"
    exports: ["LayerResult", "LayerStatus"]
    depends_on: ["HLS-MDL-001", "HLS-MDL-007"]

  HLS-MDL-003:
    name: "HumanLayerReport"
    file: "src/hl_mcp/models/report.py"
    description: "Full validation report"
    exports: ["HumanLayerReport"]
    depends_on: ["HLS-MDL-002"]

  HLS-MDL-004:
    name: "Journey"
    file: "src/hl_mcp/models/journey.py"
    description: "Journey and JourneyResult models"
    exports: ["Journey", "JourneyStep", "JourneyResult"]
    depends_on: ["HLS-MDL-007"]

  HLS-MDL-005:
    name: "Test"
    file: "src/hl_mcp/models/test.py"
    description: "Generated test models"
    exports: ["GeneratedTest", "TestConsensus"]
    depends_on: ["HLS-MDL-007"]

  HLS-MDL-006:
    name: "Review"
    file: "src/hl_mcp/models/review.py"
    description: "Review session models"
    exports: ["ReviewItem", "ReviewSession", "ScheduledReview"]
    depends_on: ["HLS-MDL-007"]

  HLS-MDL-007:
    name: "Enums"
    file: "src/hl_mcp/models/enums.py"
    description: "All enumerations"
    exports: ["VetoLevel", "LayerStatus", "Severity", "HITLDecision"]
    depends_on: []

  # ==========================================================================
  # LLM INTEGRATION
  # ==========================================================================
  HLS-LLM-001:
    name: "LLMClient"
    file: "src/hl_mcp/llm/client.py"
    description: "LLM client interface"
    exports: ["LLMClient", "LLMResponse"]
    depends_on: []

  HLS-LLM-002:
    name: "ClaudeClient"
    file: "src/hl_mcp/llm/claude.py"
    description: "Claude/Anthropic implementation"
    exports: ["ClaudeClient"]
    depends_on: ["HLS-LLM-001"]

  HLS-LLM-003:
    name: "OpenAIClient"
    file: "src/hl_mcp/llm/openai.py"
    description: "OpenAI implementation"
    exports: ["OpenAIClient"]
    depends_on: ["HLS-LLM-001"]

  HLS-LLM-004:
    name: "PromptTemplates"
    file: "src/hl_mcp/llm/prompts.py"
    description: "Prompt template management"
    exports: ["PromptTemplates", "get_prompt", "render_prompt"]
    depends_on: []

  HLS-LLM-005:
    name: "ResponseParser"
    file: "src/hl_mcp/llm/parser.py"
    description: "LLM response parsing"
    exports: ["ResponseParser", "parse_layer_response"]
    depends_on: ["HLS-MDL-001", "HLS-MDL-002"]

  # ==========================================================================
  # CONFIGURATION
  # ==========================================================================
  HLS-CFG-001:
    name: "Settings"
    file: "src/hl_mcp/config/settings.py"
    description: "Global settings management"
    exports: ["Settings", "get_settings"]
    depends_on: []

  HLS-CFG-002:
    name: "LayerConfigs"
    file: "src/hl_mcp/config/layers.py"
    description: "Layer configuration loader"
    exports: ["LayerConfigs", "load_layer_config"]
    depends_on: ["HLS-CFG-001"]

  HLS-CFG-003:
    name: "PerspectiveConfigs"
    file: "src/hl_mcp/config/perspectives.py"
    description: "Perspective configuration loader"
    exports: ["PerspectiveConfigs", "load_perspective_config"]
    depends_on: ["HLS-CFG-001"]

  HLS-CFG-004:
    name: "Thresholds"
    file: "src/hl_mcp/config/thresholds.py"
    description: "Threshold configurations"
    exports: ["Thresholds", "ConfidenceThresholds", "TriageThresholds"]
    depends_on: []

  HLS-CFG-005:
    name: "TemplateConfigs"
    file: "src/hl_mcp/config/templates.py"
    description: "Template configurations"
    exports: ["TemplateConfigs", "load_templates"]
    depends_on: []

  # ==========================================================================
  # STORAGE
  # ==========================================================================
  HLS-STR-001:
    name: "StorageInterface"
    file: "src/hl_mcp/storage/interface.py"
    description: "Storage interface definition"
    exports: ["StorageInterface"]
    depends_on: []

  HLS-STR-002:
    name: "InMemoryStorage"
    file: "src/hl_mcp/storage/memory.py"
    description: "In-memory storage implementation"
    exports: ["InMemoryStorage"]
    depends_on: ["HLS-STR-001"]

  HLS-STR-003:
    name: "SQLiteStorage"
    file: "src/hl_mcp/storage/sqlite.py"
    description: "SQLite storage implementation"
    exports: ["SQLiteStorage"]
    depends_on: ["HLS-STR-001"]

  HLS-STR-004:
    name: "RedisStorage"
    file: "src/hl_mcp/storage/redis.py"
    description: "Redis storage implementation"
    exports: ["RedisStorage"]
    depends_on: ["HLS-STR-001"]

  # ==========================================================================
  # UTILITIES
  # ==========================================================================
  HLS-UTL-001:
    name: "Hashing"
    file: "src/hl_mcp/utils/hashing.py"
    description: "Hash utilities"
    exports: ["hash_content", "generate_id"]
    depends_on: []

  HLS-UTL-002:
    name: "Timing"
    file: "src/hl_mcp/utils/timing.py"
    description: "Timing utilities"
    exports: ["Timer", "measure_time", "timeout"]
    depends_on: []

  HLS-UTL-003:
    name: "Logging"
    file: "src/hl_mcp/utils/logging.py"
    description: "Logging setup"
    exports: ["setup_logging", "get_logger"]
    depends_on: []

  HLS-UTL-004:
    name: "Validation"
    file: "src/hl_mcp/utils/validation.py"
    description: "Validation helpers"
    exports: ["validate_config", "validate_input"]
    depends_on: []

# ============================================================================
# QUICK LOOKUP TABLES
# ============================================================================

by_area:
  server: ["HLS-SRV-001", "HLS-SRV-002", "HLS-SRV-003", "HLS-SRV-004"]
  core: ["HLS-CORE-001", "HLS-CORE-002", "HLS-CORE-003", "HLS-CORE-004"]
  layers: ["HLS-LYR-000", "HLS-LYR-001", "HLS-LYR-002", "HLS-LYR-003", "HLS-LYR-004", "HLS-LYR-005", "HLS-LYR-006", "HLS-LYR-007"]
  perspectives: ["HLS-PERSP-000", "HLS-PERSP-001", "HLS-PERSP-002", "HLS-PERSP-003", "HLS-PERSP-004", "HLS-PERSP-005", "HLS-PERSP-006", "HLS-PERSP-010", "HLS-PERSP-011"]
  cognitive: ["HLS-COG-001", "HLS-COG-002", "HLS-COG-003", "HLS-COG-004", "HLS-COG-005", "HLS-COG-006"]
  browser: ["HLS-BRW-001", "HLS-BRW-002", "HLS-BRW-003", "HLS-BRW-004", "HLS-BRW-005", "HLS-BRW-006"]
  journeys: ["HLS-JRN-001", "HLS-JRN-002", "HLS-JRN-003", "HLS-JRN-004", "HLS-JRN-005"]
  models: ["HLS-MDL-001", "HLS-MDL-002", "HLS-MDL-003", "HLS-MDL-004", "HLS-MDL-005", "HLS-MDL-006", "HLS-MDL-007"]
  llm: ["HLS-LLM-001", "HLS-LLM-002", "HLS-LLM-003", "HLS-LLM-004", "HLS-LLM-005"]
  config: ["HLS-CFG-001", "HLS-CFG-002", "HLS-CFG-003", "HLS-CFG-004", "HLS-CFG-005"]
  storage: ["HLS-STR-001", "HLS-STR-002", "HLS-STR-003", "HLS-STR-004"]
  utils: ["HLS-UTL-001", "HLS-UTL-002", "HLS-UTL-003", "HLS-UTL-004"]

total_modules: 60
```

---

## INTERFACES ENTRE MÓDULOS

### Regra de Ouro: Dependency Inversion

```python
# ERRADO - Dependência direta
from hl_mcp.llm.claude import ClaudeClient

class HL1Usuario:
    def __init__(self):
        self.llm = ClaudeClient()  # Acoplamento forte!

# CERTO - Dependência por interface
from hl_mcp.llm.client import LLMClient

class HL1Usuario:
    def __init__(self, llm: LLMClient):
        self.llm = llm  # Injetado!
```

### Diagrama de Dependências (Simplificado)

```
                           ┌─────────────┐
                           │ MCP Server  │
                           │ (HLS-SRV-*) │
                           └──────┬──────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
              ┌─────▼─────┐ ┌─────▼─────┐ ┌─────▼─────┐
              │  Runner   │ │Orchestrator│ │   Gate    │
              │(CORE-001) │ │ (CORE-002) │ │(CORE-003) │
              └─────┬─────┘ └─────┬─────┘ └─────┬─────┘
                    │             │             │
         ┌──────────┴──────────┐  │             │
         │                     │  │             │
   ┌─────▼─────┐         ┌─────▼──▼──┐    ┌─────▼─────┐
   │  Layers   │         │ Cognitive │    │  Browser  │
   │ (LYR-*)   │         │  (COG-*)  │    │ (BRW-*)   │
   └─────┬─────┘         └─────┬─────┘    └─────┬─────┘
         │                     │                │
   ┌─────▼─────┐         ┌─────▼─────┐    ┌─────▼─────┐
   │   LLM     │         │  Storage  │    │ Journeys  │
   │ (LLM-*)   │         │ (STR-*)   │    │ (JRN-*)   │
   └───────────┘         └───────────┘    └───────────┘
         │                     │                │
         └─────────────────────┼────────────────┘
                               │
                         ┌─────▼─────┐
                         │  Models   │
                         │ (MDL-*)   │
                         └───────────┘
```

---

## TEMPLATE DE MÓDULO

Cada módulo DEVE seguir este template:

```python
"""
Module: HLS-XXX-NNN - ModuleName
================================

Brief one-line description.

Longer description if needed. Explain the purpose,
when to use, and any important concepts.

Example:
    >>> from hl_mcp.area.module import Something
    >>> result = Something().do_thing()

Dependencies:
    - HLS-YYY-MMM: Why this dependency

See Also:
    - HLS-ZZZ-PPP: Related module

Author: Human Layer Team
Version: 1.0.0
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Standard library imports
import logging

# Type-only imports (avoid circular deps)
if TYPE_CHECKING:
    from hl_mcp.other import OtherType

# Local imports (explicit)
from hl_mcp.models.enums import Severity

# Constants
DEFAULT_TIMEOUT = 30

# Logger
logger = logging.getLogger(__name__)


class ModuleName:
    """Brief class description.

    Attributes:
        attr1: Description of attr1
        attr2: Description of attr2

    Example:
        >>> obj = ModuleName(config)
        >>> obj.do_something()
    """

    def __init__(self, config: ConfigType) -> None:
        """Initialize ModuleName.

        Args:
            config: Configuration object
        """
        self.config = config

    def public_method(self, param: str) -> ResultType:
        """Do something useful.

        Args:
            param: Description of param

        Returns:
            Description of return value

        Raises:
            ValueError: When param is invalid
        """
        pass


# Public exports
__all__ = ["ModuleName", "CONSTANT"]
```

---

## MCP TOOLS DEFINITION

Cada funcionalidade exposta como MCP Tool:

```python
# src/hl_mcp/server/tools.py

from mcp.types import Tool

TOOLS = [
    # =========================================================================
    # CORE TOOLS
    # =========================================================================
    Tool(
        name="human_layer.run",
        description="Execute Human Layer validation with 7 layers",
        inputSchema={
            "type": "object",
            "properties": {
                "artifact": {"type": "string", "description": "Content to validate"},
                "artifact_type": {"type": "string", "enum": ["code", "spec", "config", "doc"]},
                "layer_pack": {"type": "string", "enum": ["FULL", "SECURITY", "USABILITY", "MINIMAL"], "default": "FULL"},
                "redundancy": {"type": "integer", "default": 3, "minimum": 1, "maximum": 5},
            },
            "required": ["artifact", "artifact_type"],
        },
    ),

    Tool(
        name="human_layer.run_layer",
        description="Execute a single Human Layer",
        inputSchema={
            "type": "object",
            "properties": {
                "artifact": {"type": "string"},
                "layer": {"type": "string", "enum": ["HL-1", "HL-2", "HL-3", "HL-4", "HL-5", "HL-6", "HL-7"]},
            },
            "required": ["artifact", "layer"],
        },
    ),

    # =========================================================================
    # PERSPECTIVE TOOLS
    # =========================================================================
    Tool(
        name="human_layer.generate_tests",
        description="Generate tests from 6 human perspectives",
        inputSchema={
            "type": "object",
            "properties": {
                "specification": {"type": "string"},
                "component": {"type": "string"},
                "max_per_perspective": {"type": "integer", "default": 5},
            },
            "required": ["specification", "component"],
        },
    ),

    Tool(
        name="human_layer.generate_consensus_tests",
        description="Generate tests with multi-perspective consensus",
        inputSchema={
            "type": "object",
            "properties": {
                "specification": {"type": "string"},
                "component": {"type": "string"},
                "min_agreement": {"type": "number", "default": 0.6},
            },
            "required": ["specification", "component"],
        },
    ),

    # =========================================================================
    # BROWSER TOOLS
    # =========================================================================
    Tool(
        name="human_layer.browser.execute_journey",
        description="Execute a browser journey with recording",
        inputSchema={
            "type": "object",
            "properties": {
                "journey": {"type": "object"},
                "record_video": {"type": "boolean", "default": True},
                "check_accessibility": {"type": "boolean", "default": True},
            },
            "required": ["journey"],
        },
    ),

    Tool(
        name="human_layer.browser.screenshot",
        description="Take a screenshot of current page",
        inputSchema={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "full_page": {"type": "boolean", "default": False},
            },
            "required": ["url"],
        },
    ),

    # =========================================================================
    # JOURNEY TOOLS
    # =========================================================================
    Tool(
        name="human_layer.journey.complete",
        description="Complete a journey with error/edge/persona paths",
        inputSchema={
            "type": "object",
            "properties": {
                "happy_path": {"type": "object"},
                "include_errors": {"type": "boolean", "default": True},
                "include_edges": {"type": "boolean", "default": True},
                "include_personas": {"type": "boolean", "default": True},
            },
            "required": ["happy_path"],
        },
    ),

    # =========================================================================
    # COGNITIVE TOOLS
    # =========================================================================
    Tool(
        name="human_layer.score",
        description="Score confidence of a result",
        inputSchema={
            "type": "object",
            "properties": {
                "result": {"type": "object"},
                "context": {"type": "object"},
            },
            "required": ["result", "context"],
        },
    ),

    Tool(
        name="human_layer.humanize_error",
        description="Transform technical error to human-friendly",
        inputSchema={
            "type": "object",
            "properties": {
                "error_code": {"type": "string"},
                "error_message": {"type": "string"},
                "recipient": {"type": "string", "enum": ["developer", "operator", "manager", "user", "auditor"]},
            },
            "required": ["error_code", "error_message"],
        },
    ),

    Tool(
        name="human_layer.triage",
        description="Decide if item needs human review",
        inputSchema={
            "type": "object",
            "properties": {
                "confidence": {"type": "number"},
                "novelty": {"type": "number"},
                "risk": {"type": "number"},
                "complexity": {"type": "number"},
            },
            "required": ["confidence", "risk"],
        },
    ),

    # =========================================================================
    # BUDGET/TRUST TOOLS
    # =========================================================================
    Tool(
        name="human_layer.budget.status",
        description="Get current cognitive budget status",
        inputSchema={"type": "object", "properties": {}},
    ),

    Tool(
        name="human_layer.trust.get",
        description="Get trust score for an agent",
        inputSchema={
            "type": "object",
            "properties": {
                "agent_id": {"type": "string"},
            },
            "required": ["agent_id"],
        },
    ),

    Tool(
        name="human_layer.feedback.record",
        description="Record human feedback on AI decision",
        inputSchema={
            "type": "object",
            "properties": {
                "decision_id": {"type": "string"},
                "human_decision": {"type": "string"},
                "reasoning": {"type": "string"},
            },
            "required": ["decision_id", "human_decision"],
        },
    ),
]
```

---

## ESTATÍSTICAS DA ARQUITETURA

| Área | Módulos | Linhas Est. | Complexidade |
|------|---------|-------------|--------------|
| Server | 4 | 400 | Baixa |
| Core | 4 | 1200 | Alta |
| Layers | 8 | 1600 | Média |
| Perspectives | 9 | 1200 | Média |
| Cognitive | 6 | 1500 | Alta |
| Browser | 6 | 1000 | Alta |
| Journeys | 5 | 600 | Média |
| Models | 7 | 800 | Baixa |
| LLM | 5 | 600 | Média |
| Config | 5 | 400 | Baixa |
| Storage | 4 | 500 | Média |
| Utils | 4 | 300 | Baixa |
| **TOTAL** | **67** | **~10,100** | - |

---

## PRÓXIMO BLOCK

O Block 03 vai cobrir:
1. MCP Resources (exposição de dados)
2. Context Pack Builder (wizard para usuários)
3. Sistema de documentação automática
4. Implementação do primeiro módulo (HLS-MDL-007: Enums)

---

*ROADMAP_BLOCK_02_ARCHITECTURE.md - v1.0.0 - 2026-02-01*
