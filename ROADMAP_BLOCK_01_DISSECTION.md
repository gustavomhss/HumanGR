# Human Layer MCP Server - Roadmap Block 01
# COMPLETE DISSECTION OF CURRENT HUMAN LAYER

> **Objetivo**: Documentar TUDO que o Human Layer atual oferece antes de adicionar novas features.
> **Arquitetura Alvo**: HIPER MODULARIZADA - Peças como Legos, cada peça documentada e indexada.
> **Versão**: 1.0.0 | Data: 2026-02-01

---

## INDICE DE MODULOS (LEGO INDEX)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      HUMAN LAYER - LEGO ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  CAMADA 1: CORE ENGINE (Orquestração)                                       │
│  ├── HL-CORE-001: HumanLayerRunner      [Executor principal]                │
│  ├── HL-CORE-002: CognitiveOrchestrator [Coordenador central]               │
│  └── HL-CORE-003: HumanLayerGate        [Gate de validação]                 │
│                                                                             │
│  CAMADA 2: 7 HUMAN PERSONAS (Validação Humana)                              │
│  ├── HL-PERS-001: HL-1 Usuario          [Usabilidade]                       │
│  ├── HL-PERS-002: HL-2 Operador         [Operabilidade]                     │
│  ├── HL-PERS-003: HL-3 Mantenedor       [Manutenibilidade]                  │
│  ├── HL-PERS-004: HL-4 Decisor          [Estratégia/Trust]                  │
│  ├── HL-PERS-005: HL-5 Seguranca        [Segurança acidental]               │
│  ├── HL-PERS-006: HL-6 Hacker           [Segurança adversarial]             │
│  └── HL-PERS-007: HL-7 Simplificador    [Simplificação]                     │
│                                                                             │
│  CAMADA 3: 6 PERSPECTIVE TESTS (QA Automatizado)                            │
│  ├── HL-PERSP-001: tired_user           [Usuário cansado]                   │
│  ├── HL-PERSP-002: malicious_insider    [Insider malicioso]                 │
│  ├── HL-PERSP-003: confused_newbie      [Novato confuso]                    │
│  ├── HL-PERSP-004: power_user           [Power user]                        │
│  ├── HL-PERSP-005: auditor              [Auditor compliance]                │
│  └── HL-PERSP-006: 3am_operator         [Operador 3AM]                      │
│                                                                             │
│  CAMADA 4: COGNITIVE MODULES (Inteligência)                                 │
│  ├── HL-COG-001: CognitiveBudgetManager [Orçamento cognitivo]               │
│  ├── HL-COG-002: TrustCalibration       [Calibração de trust]               │
│  ├── HL-COG-003: PredictiveTriage       [Triagem ML-based]                  │
│  ├── HL-COG-004: FeedbackLearner        [Aprendizado feedback]              │
│  ├── HL-COG-005: ConfidenceScorer       [Scoring de confiança]              │
│  └── HL-COG-006: HumanizedErrors        [Erros humanizados]                 │
│                                                                             │
│  CAMADA 5: BROWSER AUTOMATION (UI Testing)                                  │
│  ├── HL-UI-001: UIHum                   [Playwright wrapper]                │
│  ├── HL-UI-002: JourneyRunner           [Executor de jornadas]              │
│  ├── HL-UI-003: AccessibilityValidator  [Validador a11y]                    │
│  └── HL-UI-004: VideoRecorder           [Gravador de sessões]               │
│                                                                             │
│  CAMADA 6: JOURNEY GENERATION (Path Testing)                                │
│  ├── HL-JRN-001: JourneyCompleter       [Gerador de jornadas]               │
│  ├── HL-JRN-002: ErrorPathGenerator     [Caminhos de erro]                  │
│  ├── HL-JRN-003: EdgePathGenerator      [Edge cases]                        │
│  └── HL-JRN-004: PersonaPathGenerator   [Caminhos por persona]              │
│                                                                             │
│  CAMADA 7: DATA MODELS (Contratos)                                          │
│  ├── HL-DATA-001: Finding               [Finding de validação]              │
│  ├── HL-DATA-002: SecurityFinding       [Finding de segurança]              │
│  ├── HL-DATA-003: LayerResult           [Resultado de layer]                │
│  ├── HL-DATA-004: HumanLayerReport      [Relatório completo]                │
│  ├── HL-DATA-005: JourneyResult         [Resultado de jornada]              │
│  └── HL-DATA-006: GeneratedTest         [Teste gerado]                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. CAMADA CORE ENGINE - Orquestração

### 1.1 HL-CORE-001: HumanLayerRunner

**Arquivo**: `pipeline/human_layer_runner.py`
**Linhas**: ~800+
**Dependências**: Claude LLM, Pydantic, asyncio

```python
# INTERFACE PRINCIPAL
class HumanLayerRunner:
    """Executa validação com 7 Human Layers."""

    async def run_all_layers(
        artifact: str,
        artifact_type: str,
        layer_pack: str = "FULL",  # FULL, SECURITY, USABILITY, MINIMAL
        redundancy: int = 3,       # Triple redundancy
        consensus: float = 0.67,   # 2/3 consensus
    ) -> HumanLayerReport

    async def run_single_layer(
        artifact: str,
        layer: HumanLayerSpec,
    ) -> LayerResult
```

**Funcionalidades**:
- Execute 7 layers em sequência ou paralelo
- Triple redundancy (3 execuções por layer)
- Consenso 2/3 para decisão final
- Sistema de veto WEAK/MEDIUM/STRONG
- Integração com Claude via `create_claude_executor()`

**Invariantes**:
```
INV-HLX-001: Triple redundancy obrigatória para todos os layers
INV-HLX-002: STRONG veto bloqueia tudo
INV-HLX-003: MEDIUM veto bloqueia merge
INV-HLX-004: Todos findings devem ter fix_hint
INV-HLX-005: Security findings sempre prioridade máxima
```

**Configuração Exportável (para MCP)**:
```yaml
human_layer_runner:
  redundancy: 3
  consensus_threshold: 0.67
  timeout_per_layer: 120  # segundos
  parallel_execution: true
  fail_fast_on_strong_veto: true
```

---

### 1.2 HL-CORE-002: CognitiveOrchestrator

**Arquivo**: `pipeline/human_layer/cognitive_orchestrator.py`
**Linhas**: ~400
**Dependências**: CognitiveBudgetManager, TrustCalibration, PredictiveTriage

```python
class CognitiveOrchestrator:
    """Coordenador central do Human Layer."""

    def __init__(
        budget_manager: CognitiveBudgetManager,
        trust_system: TrustCalibrationSystem,
        triage: PredictiveTriage,
    )

    async def schedule_reviews(
        items: List[ReviewItem],
        time_window: timedelta = timedelta(hours=4),
    ) -> List[ScheduledReview]

    async def process_review(
        item: ReviewItem,
        human_decision: Optional[str] = None,
    ) -> ReviewSession
```

**Funcionalidades**:
- Agenda reviews por prioridade
- Respeita budget cognitivo
- Aplica predictive triage
- Roteia para layer apropriado
- Batch reviews para eficiência

**Estados do Review**:
```
PENDING → IN_PROGRESS → COMPLETED
                     → ESCALATED
                     → SKIPPED (auto-approve)
```

---

### 1.3 HL-CORE-003: HumanLayerGate

**Arquivo**: `pipeline/human_layer_gate.py`
**Linhas**: 424
**Dependências**: HumanLayerRunner, Finding, SecurityFinding

```python
@dataclass
class HumanLayerResult:
    """Resultado do gate de Human Layer."""
    passed: bool
    can_promote: bool     # Pode ir para prod?
    can_merge: bool       # Pode fazer merge?
    layers_run: int
    layers_passed: int
    findings: List[Finding]
    security_findings: List[SecurityFinding]
    veto_summary: Dict[str, int]  # {STRONG: 0, MEDIUM: 2, WEAK: 5}

async def run_human_layer_gate(
    artifact_path: str,
    layer_pack: str = "FULL",
) -> HumanLayerResult
```

**Lógica de Decisão**:
```
can_promote = (STRONG_VETOS == 0) AND (critical_findings == 0)
can_merge = can_promote AND (MEDIUM_VETOS <= 1)
```

---

## 2. CAMADA 7 HUMAN PERSONAS

### Definição de cada Layer

| ID | Nome | Pergunta Central | Veto Power | Focus Areas |
|----|------|------------------|------------|-------------|
| HL-1 | Humano-Usuario | É usável? | WEAK | UX, friction, clarity |
| HL-2 | Humano-Operador | Consigo operar? | MEDIUM | Diagnostics, runbooks, rollback |
| HL-3 | Humano-Mantenedor | Consigo manter? | MEDIUM | Code quality, docs, tech debt |
| HL-4 | Humano-Decisor | Faz sentido estratégico? | STRONG | ROI, alignment, risk/reward |
| HL-5 | Humano-Seguranca | É perigoso por acidente? | STRONG | Safety, misuse, unintended |
| HL-6 | Humano-Hacker | Como eu abusaria isso? | STRONG | Exploits, attack vectors |
| HL-7 | Humano-Simplificador | Dá pra simplificar? | WEAK | Complexity, YAGNI, minimalism |

### 2.1 HL-PERS-001: HL-1 Usuario

**Configuração** (de `human_layer_config.yaml`):
```yaml
HL-1:
  id: "HL-1_Humano_Usuario"
  name: "Humano Usuario"
  focus: "Usabilidade e experiencia do usuario"
  veto_power: WEAK

  core_questions:
    - "O usuario entende o que esta acontecendo?"
    - "A jornada e intuitiva?"
    - "As mensagens de erro ajudam ou confundem?"
    - "O usuario consegue completar a tarefa sem ajuda?"

  red_flags:
    - "Jargao tecnico exposto ao usuario"
    - "Fluxo com mais de 5 passos"
    - "Erro generico sem orientacao"
    - "Acao irreversivel sem confirmacao"

  output_format:
    usability_score: float  # 0.0-1.0
    friction_points: List[str]
    recommendations: List[str]
```

### 2.2 HL-PERS-002: HL-2 Operador

```yaml
HL-2:
  id: "HL-2_Humano_Operador"
  name: "Humano Operador"
  focus: "Operabilidade e observabilidade"
  veto_power: MEDIUM

  core_questions:
    - "Como sei que esta funcionando?"
    - "Como diagnostico problemas?"
    - "Como faco rollback se der errado?"
    - "Tem runbook para incidentes?"

  red_flags:
    - "Sem logs estruturados"
    - "Sem metricas de health"
    - "Rollback manual complexo"
    - "Dependencia sem circuit breaker"
```

### 2.3 HL-PERS-003: HL-3 Mantenedor

```yaml
HL-3:
  id: "HL-3_Humano_Mantenedor"
  name: "Humano Mantenedor"
  focus: "Manutenibilidade do codigo"
  veto_power: MEDIUM

  core_questions:
    - "Outro dev entende este codigo?"
    - "Tem testes adequados?"
    - "A arquitetura escala?"
    - "Tech debt e gerenciavel?"

  red_flags:
    - "Funcao com mais de 50 linhas"
    - "Acoplamento alto entre modulos"
    - "Testes flaky ou ausentes"
    - "Documentacao desatualizada"
```

### 2.4 HL-PERS-004: HL-4 Decisor

```yaml
HL-4:
  id: "HL-4_Humano_Decisor"
  name: "Humano Decisor"
  focus: "Alinhamento estrategico e confianca"
  veto_power: STRONG  # Pode vetar tudo

  core_questions:
    - "Faz sentido para o negocio?"
    - "O risco justifica o beneficio?"
    - "Esta alinhado com a estrategia?"
    - "Consigo explicar para stakeholders?"

  red_flags:
    - "ROI negativo ou incerto"
    - "Conflito com prioridades"
    - "Risco reputacional"
    - "Decisao irreversivel sem plano B"
```

### 2.5 HL-PERS-005: HL-5 Seguranca

```yaml
HL-5:
  id: "HL-5_Humano_Seguranca"
  name: "Humano Seguranca"
  focus: "Seguranca e riscos acidentais"
  veto_power: STRONG

  core_questions:
    - "Pode causar dano acidentalmente?"
    - "Dados sensiveis estao protegidos?"
    - "Falha gracefully?"
    - "Blast radius e limitado?"

  red_flags:
    - "Dados em texto plano"
    - "SQL sem parametrizacao"
    - "Segredo hardcoded"
    - "Permissao excessiva"

  owasp_checks:
    - "A01:2021 – Broken Access Control"
    - "A02:2021 – Cryptographic Failures"
    - "A03:2021 – Injection"
```

### 2.6 HL-PERS-006: HL-6 Hacker

```yaml
HL-6:
  id: "HL-6_Humano_Hacker"
  name: "Humano Hacker"
  focus: "Pensamento adversarial"
  veto_power: STRONG

  core_questions:
    - "Como EU abusaria isso?"
    - "Que input quebra isso?"
    - "Posso escalar privilegios?"
    - "Posso exfiltrar dados?"

  attack_vectors:
    - "Injection (SQL, XSS, Command)"
    - "Authentication bypass"
    - "Authorization escalation"
    - "Data exfiltration"
    - "Denial of service"

  red_flags:
    - "Input nao sanitizado"
    - "Rate limit ausente"
    - "Session fixation possivel"
    - "IDOR vulneravel"
```

### 2.7 HL-PERS-007: HL-7 Simplificador

```yaml
HL-7:
  id: "HL-7_Humano_Simplificador"
  name: "Humano Simplificador"
  focus: "Simplicidade e YAGNI"
  veto_power: WEAK

  core_questions:
    - "Precisa ser tao complexo?"
    - "O que pode ser removido?"
    - "Existe forma mais simples?"
    - "Esta over-engineered?"

  simplification_targets:
    - "Abstrações desnecessárias"
    - "Configurações demais"
    - "Features não usadas"
    - "Código duplicado"
```

---

## 3. CAMADA 6 PERSPECTIVE TESTS

**Arquivo**: `pipeline/human_layer/perspective_tests.py`
**Linhas**: 753

### 3.1 Definição das 6 Perspectivas

```python
PERSPECTIVES = {
    "tired_user": Perspective(
        id="tired_user",
        name="Tired User",
        persona="User at the end of a long day, low patience",
        focus=["minimal_friction_paths", "error_recovery", "clear_messages"],
        weight=1.0,
        test_patterns=[
            "What if user clicks multiple times?",
            "What if user abandons mid-flow?",
            "What if error message is confusing?",
        ],
    ),
    "malicious_insider": Perspective(
        id="malicious_insider",
        name="Malicious Insider",
        persona="Employee with system access, bad intentions",
        focus=["privilege_escalation", "data_exfiltration", "audit_bypass"],
        weight=1.5,  # SECURITY = HIGHER WEIGHT
    ),
    "confused_newbie": Perspective(
        id="confused_newbie",
        name="Confused Newbie",
        persona="First-time user, no domain knowledge",
        focus=["onboarding_clarity", "terminology", "help_availability"],
        weight=1.0,
    ),
    "power_user": Perspective(
        id="power_user",
        name="Power User",
        persona="Expert user, wants efficiency",
        focus=["keyboard_shortcuts", "bulk_operations", "api_access"],
        weight=0.8,
    ),
    "auditor": Perspective(
        id="auditor",
        name="Auditor",
        persona="Compliance officer, needs evidence",
        focus=["audit_trails", "data_provenance", "compliance_controls"],
        weight=1.2,
    ),
    "3am_operator": Perspective(
        id="3am_operator",
        name="3AM Operator",
        persona="On-call engineer at 3am",
        focus=["diagnostics", "rollback", "minimal_context", "runbooks"],
        weight=1.0,
    ),
}
```

### 3.2 PerspectiveTestGenerator

```python
class PerspectiveTestGenerator:
    """Gera testes de 6 perspectivas humanas."""

    def generate_tests(
        specification: str,
        component: str,
        max_tests_per_perspective: int = 5,
    ) -> List[GeneratedTest]

    def generate_consensus_tests(
        specification: str,
        component: str,
        min_agreement: float = 0.6,
    ) -> List[TestConsensus]
```

### 3.3 Intuition Templates

Templates de "human intuition" para cenários comuns:

```python
INTUITION_TEMPLATES = {
    "authentication_flow": {
        "trigger_keywords": ["auth", "login", "password", "token"],
        "human_thoughts": [
            "What if password manager autofill is slow?",
            "What if 2FA device is unavailable?",
            "What if session expires mid-action?",
        ],
    },
    "data_input_form": {
        "trigger_keywords": ["form", "input", "field", "submit"],
        "human_thoughts": [
            "What if paste includes invisible characters?",
            "What if copy from Excel has trailing tabs?",
            "What if emoji in name field?",
        ],
    },
    "payment_flow": {
        "trigger_keywords": ["payment", "checkout", "cart", "price"],
        "human_thoughts": [
            "What if currency conversion fails?",
            "What if payment times out?",
            "What if user refreshes during processing?",
        ],
    },
    "file_upload": {
        "trigger_keywords": ["upload", "file", "attachment"],
        "human_thoughts": [
            "What if file is corrupted?",
            "What if file extension doesn't match content?",
            "What if user uploads malicious file?",
        ],
    },
    "api_endpoint": {
        "trigger_keywords": ["api", "endpoint", "request", "response"],
        "human_thoughts": [
            "What if rate limit is exceeded?",
            "What if API version is deprecated?",
            "What if concurrent requests race?",
        ],
    },
}
```

---

## 4. CAMADA COGNITIVE MODULES

### 4.1 HL-COG-001: CognitiveBudgetManager

**Arquivo**: `pipeline/human_layer/cognitive_budget.py`
**Conceito**: Gerencia "attention units" para evitar review fatigue

```python
@dataclass
class CognitiveBudgetManager:
    """Gerencia orçamento cognitivo de revisores."""

    total_daily_units: int = 100

    category_allocations: Dict[str, int] = {
        "security": 20,      # Alta prioridade
        "decision": 25,      # Decisões estratégicas
        "code_quality": 15,  # Qualidade de código
        "documentation": 10, # Documentação
        "general": 30,       # Revisões gerais
    }

    def can_review(item: ReviewItem) -> bool
    def allocate(item: ReviewItem) -> AttentionUnit
    def release(unit: AttentionUnit) -> None
    def get_remaining_budget() -> Dict[str, int]
```

**Configuração**:
```yaml
cognitive_budget:
  total_daily_units: 100
  fatigue_threshold: 0.3      # 30% restante = fadiga
  recovery_rate: 10           # unidades/hora de descanso
  batch_discount: 0.8         # 20% desconto em batch
  high_priority_reserve: 0.2  # 20% reservado para urgentes
```

### 4.2 HL-COG-002: TrustCalibrationSystem

**Arquivo**: `pipeline/human_layer/trust_calibration.py`
**Conceito**: Trust score dinâmico que aprende com feedback

```python
@dataclass
class TrustScore:
    """Score de confiança com bounds e história."""
    value: float              # 0.0-1.0
    min_bound: float = 0.50   # Nunca abaixo disso
    max_bound: float = 0.98   # Nunca acima disso
    baseline: float = 0.70    # Baseline para decay

class TrustCalibrationSystem:
    """Sistema de calibração de trust."""

    adjustments = {
        "AGREE": +0.02,       # Humano concordou
        "OVERRIDE": -0.05,    # Humano discordou fortemente
        "ESCALATE": -0.03,    # Precisou escalar
        "PARTIAL": +0.01,     # Concordância parcial
    }

    def update_trust(agent_id: str, feedback: FeedbackType) -> TrustScore
    def get_trust(agent_id: str) -> TrustScore
    def decay_towards_baseline(hours_since_last: float) -> None
```

**Lógica de Decay**:
```
Se trust > baseline: decay = -0.01/hora
Se trust < baseline: decay = +0.005/hora
Objetivo: Voltar gradualmente ao baseline
```

### 4.3 HL-COG-003: PredictiveTriage

**Arquivo**: `pipeline/human_layer/predictive_triage.py`
**Conceito**: ML-based prediction de o que precisa human review

```python
class HITLDecision(Enum):
    """Human-in-the-loop decision."""
    SKIP = "skip"           # Auto-approve
    ASYNC_QUEUE = "async"   # Queue for later
    SYNC_REQUIRED = "sync"  # Needs immediate review

@dataclass
class TriageFactors:
    """Fatores para decisão de triage."""
    confidence: float       # Confiança do modelo
    novelty: float          # Quão novo é o padrão
    risk: float             # Risco estimado
    precedent: float        # Tem precedente similar?
    budget_available: float # Budget cognitivo disponível
    complexity: float       # Complexidade do item
    urgency: float          # Urgência do review

class PredictiveTriage:
    """Triagem preditiva para HITL."""

    def decide(factors: TriageFactors) -> TriageDecision
    def learn_from_outcome(decision: TriageDecision, was_correct: bool) -> None
```

**Thresholds**:
```yaml
triage_thresholds:
  auto_skip:
    min_confidence: 0.95
    max_risk: 0.1
    max_novelty: 0.2

  sync_required:
    min_risk: 0.7
    max_confidence: 0.6

  async_queue:  # Everything else
```

### 4.4 HL-COG-004: FeedbackLearner

**Arquivo**: `pipeline/human_layer/feedback_learning.py`
**Conceito**: Aprendizado bi-direcional AI <-> Human

```python
@dataclass
class HumanFeedback:
    """Feedback de humano sobre decisão AI."""
    decision_id: str
    original_decision: str
    human_decision: str
    reasoning: str
    confidence: float
    timestamp: datetime

@dataclass
class LearnedPattern:
    """Padrão aprendido do feedback."""
    pattern_id: str
    trigger_conditions: Dict[str, Any]
    learned_action: str
    confidence: float
    source_feedback_count: int

class FeedbackLearner:
    """Aprende padrões do feedback humano."""

    def record_feedback(feedback: HumanFeedback) -> None
    def extract_patterns() -> List[LearnedPattern]
    def apply_learned_patterns(item: ReviewItem) -> Optional[str]
    def get_pattern_statistics() -> Dict[str, Any]
```

### 4.5 HL-COG-005: HumanLayerConfidenceScorer

**Arquivo**: `pipeline/human_layer/confidence_scorer.py`
**Conceito**: Scoring multi-dimensional de confiança

```python
@dataclass
class ConfidenceThresholds:
    auto_approve: float = 0.90    # Acima disso, auto-aprova
    human_optional: float = 0.75  # Review opcional
    human_required: float = 0.60  # Review obrigatório
    escalation: float = 0.40      # Abaixo disso, escala
    critical_modifier: float = 1.15  # Multiplicador para críticos

class ReviewTrigger(Enum):
    LOW_CONFIDENCE = "low_confidence"
    NOVELTY_DETECTED = "novelty_detected"
    HIGH_RISK = "high_risk"
    CRITICAL_PATH = "critical_path"
    SECURITY_CONCERN = "security_concern"
    PRECEDENT_MISMATCH = "precedent_mismatch"
    CONFLICTING_EVIDENCE = "conflicting_evidence"

@dataclass
class ScoredResult:
    result: Any
    dimensions: List[ConfidenceDimension]
    triggers: List[ReviewTrigger]

    @property
    def overall_score(self) -> float

    def needs_human_review() -> bool
    def get_review_recommendation() -> Dict[str, Any]
```

**Dimensões de Confiança**:
```python
dimensions = {
    "evidence_quality": 0.30,   # Qualidade da evidência
    "test_coverage": 0.25,      # Cobertura de testes
    "pattern_match": 0.20,      # Match com padrões conhecidos
    "reversibility": 0.15,      # Quão reversível é
    "consensus": 0.10,          # Consenso entre revisores
}
```

### 4.6 HL-COG-006: HumanizedErrorGenerator

**Arquivo**: `pipeline/human_layer/humanized_errors.py`
**Conceito**: Transforma erros técnicos em WHY + HOW TO FIX

```python
class RecipientType(Enum):
    DEVELOPER = "developer"   # Técnico com fix suggestions
    OPERATOR = "operator"     # Operacional com diagnostic steps
    MANAGER = "manager"       # Business impact com timeline
    USER = "user"             # Simples com next steps
    AUDITOR = "auditor"       # Compliance-focused com evidence

@dataclass
class HumanizedError:
    what_happened: str
    why_happened: str
    how_to_fix: List[str]
    estimated_effort: str
    prevention_tip: str
    severity: Severity
    business_impact: str
    technical_details: str
    diagnostic_steps: List[str]
    compliance_notes: str

    def format_for_recipient(recipient: RecipientType) -> str

class HumanizedErrorGenerator:
    """Gera erros humanizados."""

    def generate_error(
        context: ErrorContext,
        recipient: RecipientType = RecipientType.DEVELOPER,
    ) -> HumanizedError
```

**Error Templates**:
```python
ERROR_TEMPLATES = {
    "SECURITY-001": {
        "pattern": r"input.*(validation|sanitization).*missing",
        "what": "The '{field}' field accepts input without sanitization.",
        "why": "Without validation, vulnerable to SQL injection, XSS...",
        "fix": [
            "Add input validation using Pydantic schema",
            "Apply appropriate sanitization",
            "Add validation test coverage",
        ],
        "effort": "15-30 min",
        "severity": Severity.HIGH,
    },
    "GATE_FAILURE": {...},
    "COVERAGE_LOW": {...},
    "MUTATION_LOW": {...},
    "TYPE_ERROR": {...},
    "IMPORT_ERROR": {...},
    "TIMEOUT": {...},
}
```

---

## 5. CAMADA BROWSER AUTOMATION

### 5.1 HL-UI-001: UIHum (Playwright Wrapper)

**Arquivo**: `pipeline/ui_hum.py`
**Linhas**: 559
**Dependências**: Playwright

```python
class UIHum:
    """Browser automation para human-like testing."""

    # Ações suportadas
    ACTIONS = {
        "goto": navegação,
        "wait": espera,
        "click": clique,
        "fill": preencher,
        "screenshot": captura,
    }

    async def execute_journey(
        journey: Journey,
        record_video: bool = True,
        validate_accessibility: bool = True,
    ) -> UIHumResult

    async def take_screenshot(
        name: str,
        full_page: bool = False,
    ) -> str  # Returns path
```

**Configuração de Recording**:
```python
browser_context = await browser.new_context(
    record_video_dir="./recordings/",
    viewport={"width": 1280, "height": 720},
)
```

### 5.2 Accessibility Validation

```python
@dataclass
class AccessibilityResult:
    """Resultado de validação de acessibilidade."""
    passed: bool
    issues: List[AccessibilityIssue]
    warnings: List[str]

    # Checks incluídos:
    # - aria-* attributes
    # - role= attributes
    # - alt= on images
    # - keyboard navigation
    # - color contrast
```

### 5.3 JourneyResult

```python
@dataclass
class JourneyResult:
    """Resultado de execução de jornada."""
    journey_id: str
    success: bool
    steps_completed: int
    steps_total: int
    screenshots: List[str]
    video_path: Optional[str]
    accessibility: AccessibilityResult
    duration_ms: int
    errors: List[str]
```

---

## 6. CAMADA JOURNEY GENERATION

### 6.1 HL-JRN-001: JourneyCompleter

**Arquivo**: `pipeline/spec_kit/journey_completer.py`
**Conceito**: Gera jornadas além do happy path

```python
class JourneyCompleter:
    """Completa jornadas com error/edge/persona paths."""

    templates = {
        "error_paths": [...],
        "alternative_paths": [...],
        "edge_paths": [...],
        "personas": [...],
    }

    def complete_journey(
        happy_path: Journey,
        include_errors: bool = True,
        include_alternatives: bool = True,
        include_edges: bool = True,
        include_personas: bool = True,
    ) -> List[Journey]
```

### 6.2 Path Types

**Error Paths**:
```yaml
error_path_templates:
  - network_failure_mid_step
  - validation_error_on_submit
  - session_timeout
  - server_error_500
  - rate_limit_exceeded
```

**Edge Paths**:
```yaml
edge_path_templates:
  - empty_input
  - max_length_input
  - special_characters
  - unicode_emoji
  - concurrent_submission
```

**Persona Paths**:
```yaml
persona_templates:
  - impatient_user: "clicks rapidly, abandons if slow"
  - cautious_user: "reads everything, hesitates"
  - mobile_user: "small screen, touch errors"
  - accessibility_user: "keyboard only, screen reader"
```

---

## 7. CAMADA DATA MODELS

### 7.1 HL-DATA-001: Finding

```python
@dataclass
class Finding:
    """Finding de validação do Human Layer."""
    id: str
    layer: str                    # HL-1 a HL-7
    severity: Severity            # critical, high, medium, low
    title: str
    description: str
    location: Optional[str]       # Arquivo/linha
    fix_hint: str                 # OBRIGATÓRIO
    evidence: List[str]
    tags: List[str]

    # Schema validation
    def validate(self) -> bool
```

### 7.2 HL-DATA-002: SecurityFinding

```python
@dataclass
class SecurityFinding(Finding):
    """Finding de segurança com campos extras."""
    cwe_id: Optional[str]         # CWE-79, CWE-89, etc
    owasp_category: Optional[str] # A01, A02, etc
    attack_vector: str
    exploitability: float         # 0.0-1.0
    remediation_priority: str     # immediate, soon, scheduled
```

### 7.3 HL-DATA-003: LayerResult

```python
@dataclass
class LayerResult:
    """Resultado de execução de um layer."""
    layer_id: str
    status: LayerStatus           # PASS, WARN, FAIL, SKIP, ERROR
    veto: VetoLevel              # NONE, WEAK, MEDIUM, STRONG
    findings: List[Finding]
    security_findings: List[SecurityFinding]
    execution_time_ms: int
    consensus_score: float        # 0.0-1.0 (2/3 = 0.67)
    runs: int                     # Quantas execuções (triple redundancy)
```

### 7.4 HL-DATA-004: HumanLayerReport

```python
@dataclass
class HumanLayerReport:
    """Relatório completo de Human Layer."""
    artifact_id: str
    artifact_type: str
    layer_pack: str

    layers: List[LayerResult]

    summary: Dict[str, Any]
    # {
    #   "total_layers": 7,
    #   "layers_passed": 5,
    #   "layers_warned": 1,
    #   "layers_failed": 1,
    #   "strong_vetos": 0,
    #   "medium_vetos": 1,
    #   "weak_vetos": 2,
    #   "total_findings": 15,
    #   "critical_findings": 0,
    # }

    can_promote: bool
    can_merge: bool

    execution_time_ms: int
    timestamp: datetime
```

---

## MAPEAMENTO PARA MCP SERVER

### Módulos que viram MCP Tools

| Current Module | MCP Tool Name | Description |
|----------------|---------------|-------------|
| HumanLayerRunner | `human_layer.run` | Execute all 7 layers |
| PerspectiveTestGenerator | `human_layer.generate_tests` | Generate perspective tests |
| UIHum | `human_layer.browser.execute` | Execute browser journey |
| JourneyCompleter | `human_layer.journey.complete` | Complete journey paths |
| ConfidenceScorer | `human_layer.score` | Score confidence |
| HumanizedErrorGenerator | `human_layer.humanize_error` | Humanize errors |
| CognitiveBudgetManager | `human_layer.budget.*` | Manage review budget |
| TrustCalibration | `human_layer.trust.*` | Manage trust scores |
| PredictiveTriage | `human_layer.triage` | Decide HITL need |
| FeedbackLearner | `human_layer.feedback.*` | Record/apply feedback |

### Resources que MCP expõe

```yaml
mcp_resources:
  - human_layer://perspectives
  - human_layer://layers
  - human_layer://templates
  - human_layer://statistics
  - human_layer://trust/{agent_id}
  - human_layer://budget/status
```

---

## ESTATÍSTICAS DO CÓDIGO ATUAL

| Módulo | Linhas | Classes | Funções | Cobertura |
|--------|--------|---------|---------|-----------|
| human_layer_runner.py | 800+ | 5 | 25+ | - |
| cognitive_orchestrator.py | 400 | 3 | 15 | - |
| human_layer_gate.py | 424 | 2 | 8 | - |
| perspective_tests.py | 753 | 4 | 20 | - |
| confidence_scorer.py | 656 | 5 | 18 | - |
| humanized_errors.py | 668 | 4 | 15 | - |
| cognitive_budget.py | ~300 | 3 | 12 | - |
| trust_calibration.py | ~350 | 3 | 10 | - |
| predictive_triage.py | ~400 | 4 | 12 | - |
| feedback_learning.py | ~300 | 4 | 10 | - |
| ui_hum.py | 559 | 3 | 15 | - |
| journey_completer.py | ~250 | 2 | 8 | - |
| **TOTAL** | **~5,800** | **42** | **168** | - |

---

## PRÓXIMO BLOCK

O Block 02 vai cobrir:
1. Arquitetura HIPER MODULARIZADA do MCP Server
2. Definição de cada "Lego piece"
3. Sistema de indexação e documentação
4. Estrutura de pastas
5. Interfaces entre módulos

---

*ROADMAP_BLOCK_01_DISSECTION.md - v1.0.0 - 2026-02-01*
