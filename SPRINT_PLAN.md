# HUMAN LAYER - SPRINT PLAN

> **Gerado de**: MASTER_REQUIREMENTS_MAP.md (18 PARTEs, ~2000 items)
> **Metodologia**: MoSCoW + Dependency Mapping + Wave Planning
> **Data**: 2026-02-01

---

## PRIORIZAÇÃO MoSCoW

### Critérios de Priorização

```yaml
MUST_HAVE (M):
  - Sem isso o produto não funciona
  - Bloqueador de lançamento
  - Core value proposition

SHOULD_HAVE (S):
  - Importante mas não bloqueador
  - Diferencial competitivo
  - Esperado pelos usuários

COULD_HAVE (C):
  - Nice to have
  - Melhoria de UX
  - Features avançadas

WONT_HAVE_NOW (W):
  - Futuro (pós-launch)
  - Baixo ROI no momento
  - Dependência externa forte
```

---

## WAVE PLANNING (5 Waves)

```
┌─────────────────────────────────────────────────────────────────────┐
│                        WAVE ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  W0: Foundation     → Arquitetura, Models, Infraestrutura básica   │
│  (2-3 semanas)         MUST HAVE - Sem isso nada funciona          │
│                                                                     │
│  W1: Core Engine    → 7 Layers, Consensus, Veto, MCP Server        │
│  (4-6 semanas)         MUST HAVE - Core product                    │
│                                                                     │
│  W2: OSS Release    → CLI, Docs, GitHub, PyPI                      │
│  (2-3 semanas)         MUST HAVE - OSS launch                      │
│                                                                     │
│  W3: Cloud MVP      → Auth, Dashboard, API, Basic Billing          │
│  (4-6 semanas)         SHOULD HAVE - Revenue enabler               │
│                                                                     │
│  W4: Growth         → Advanced features, Integrations, Marketing   │
│  (ongoing)             COULD HAVE - Scale & differentiation        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## DEPENDENCY GRAPH (Simplificado)

```
                    ┌──────────────┐
                    │   S00-S02    │
                    │  Foundation  │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ S03-S05  │ │ S06-S08  │ │ S09-S11  │
        │  Models  │ │  Layers  │ │   LLM    │
        └────┬─────┘ └────┬─────┘ └────┬─────┘
             │            │            │
             └────────────┼────────────┘
                          │
                          ▼
                    ┌──────────────┐
                    │   S12-S14    │
                    │ Consensus +  │
                    │    Veto      │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ S15-S17  │ │ S18-S20  │ │ S21-S23  │
        │   MCP    │ │   CLI    │ │   Docs   │
        │  Server  │ │          │ │          │
        └────┬─────┘ └────┬─────┘ └────┬─────┘
             │            │            │
             └────────────┼────────────┘
                          │
                          ▼
                   ┌──────────────┐
                   │    S24       │
                   │  OSS LAUNCH  │
                   └──────┬───────┘
                          │
                          ▼
                   ┌──────────────┐
                   │   S25-S35    │
                   │    CLOUD     │
                   └──────────────┘
```

---

# WAVE 0: FOUNDATION (S00-S02)

> **Objetivo**: Criar a base arquitetural do projeto
> **Duração estimada**: 2-3 semanas
> **Prioridade**: MUST HAVE

---

## S00 - Project Setup & Architecture

```yaml
sprint:
  id: S00
  name: project-setup
  title: "Project Setup & Architecture"
  wave: W0-Foundation
  priority: P0-CRITICAL
  type: setup

objective: "Criar estrutura do projeto, configurações, e arquitetura base"

source_parts:
  - "PARTE 1: Fundação"
  - "PARTE 5: Arquitetura Técnica"

deliverables:
  # Project Structure
  - human-layer/
  - human-layer/pyproject.toml
  - human-layer/README.md
  - human-layer/LICENSE
  - human-layer/.gitignore
  - human-layer/.env.example

  # Source Structure
  - src/hl_mcp/__init__.py
  - src/hl_mcp/py.typed

  # Config
  - human-layer.yaml.example

  # CI/CD
  - .github/workflows/ci.yml
  - .github/workflows/release.yml
  - .github/ISSUE_TEMPLATE/bug_report.yml
  - .github/ISSUE_TEMPLATE/feature_request.yml
  - .github/PULL_REQUEST_TEMPLATE.md

dependencies: []

gates:
  G0_FILES_EXIST:
    validation: "ls pyproject.toml src/hl_mcp/__init__.py"

  G1_INSTALL_WORKS:
    validation: "pip install -e . && python -c 'import hl_mcp'"

  G2_CI_PASSES:
    validation: "pytest && ruff check ."

acceptance_criteria:
  - Projeto instalável via pip install -e .
  - Import básico funciona
  - CI rodando no GitHub Actions
  - Estrutura de pastas conforme arquitetura definida
```

---

## S01 - Enums & Constants

```yaml
sprint:
  id: S01
  name: enums-constants
  title: "Core Enums & Constants"
  wave: W0-Foundation
  priority: P0-CRITICAL
  type: implementation

objective: "Implementar enums fundamentais do sistema"

source_parts:
  - "PARTE 4: Features (Enums)"

deliverables:
  - src/hl_mcp/models/__init__.py
  - src/hl_mcp/models/enums.py
  - tests/test_models/test_enums.py

dependencies:
  - S00

gates:
  G0_FILES_EXIST:
    validation: "ls src/hl_mcp/models/enums.py"

  G1_IMPORTS_WORK:
    validation: |
      python -c "
      from hl_mcp.models import VetoLevel, Severity, LayerStatus, LayerID
      print('All enums imported successfully')
      "

  G2_TESTS_PASS:
    validation: "pytest tests/test_models/test_enums.py -v"

  G3_COVERAGE:
    validation: "pytest tests/test_models/test_enums.py --cov=src/hl_mcp/models/enums --cov-fail-under=95"

implementation_spec: |
  ```python
  # src/hl_mcp/models/enums.py
  from enum import Enum
  from typing import List

  class VetoLevel(Enum):
      """Veto power levels for layers."""
      NONE = "none"      # Informational only
      WEAK = "weak"      # Can add warnings
      MEDIUM = "medium"  # Can block merge
      STRONG = "strong"  # Can block everything

      @property
      def can_block_merge(self) -> bool:
          return self in (VetoLevel.MEDIUM, VetoLevel.STRONG)

      @property
      def can_block_deploy(self) -> bool:
          return self == VetoLevel.STRONG

      @classmethod
      def from_string(cls, value: str) -> "VetoLevel":
          try:
              return cls(value.lower())
          except ValueError:
              raise ValueError(f"Invalid VetoLevel: {value}")

  class Severity(Enum):
      """Finding severity levels."""
      CRITICAL = "critical"
      HIGH = "high"
      MEDIUM = "medium"
      LOW = "low"
      INFO = "info"

      @property
      def numeric_value(self) -> int:
          return {
              Severity.CRITICAL: 5,
              Severity.HIGH: 4,
              Severity.MEDIUM: 3,
              Severity.LOW: 2,
              Severity.INFO: 1,
          }[self]

      def __lt__(self, other: "Severity") -> bool:
          return self.numeric_value < other.numeric_value

  class LayerStatus(Enum):
      """Layer execution status."""
      PENDING = "pending"
      RUNNING = "running"
      PASS = "pass"
      WARN = "warn"
      FAIL = "fail"
      ERROR = "error"
      SKIP = "skip"

      @property
      def is_terminal(self) -> bool:
          return self in (LayerStatus.PASS, LayerStatus.WARN,
                          LayerStatus.FAIL, LayerStatus.ERROR, LayerStatus.SKIP)

      @property
      def is_success(self) -> bool:
          return self in (LayerStatus.PASS, LayerStatus.WARN)

  class LayerID(Enum):
      """The 7 Human Layers."""
      HL1_UX = 1
      HL2_FUNCTIONALITY = 2
      HL3_EDGE_CASES = 3
      HL4_SECURITY = 4
      HL5_PERFORMANCE = 5
      HL6_COMPLIANCE = 6
      HL7_FINAL_REVIEW = 7

      @property
      def default_veto_level(self) -> VetoLevel:
          strong = [LayerID.HL4_SECURITY, LayerID.HL6_COMPLIANCE, LayerID.HL7_FINAL_REVIEW]
          medium = [LayerID.HL2_FUNCTIONALITY, LayerID.HL3_EDGE_CASES]
          if self in strong:
              return VetoLevel.STRONG
          elif self in medium:
              return VetoLevel.MEDIUM
          return VetoLevel.WEAK

  class PerspectiveID(Enum):
      """The 6 Testing Perspectives."""
      TIRED_USER = "tired_user"
      MALICIOUS_INSIDER = "malicious_insider"
      CONFUSED_NEWBIE = "confused_newbie"
      POWER_USER = "power_user"
      AUDITOR = "auditor"
      THREE_AM_OPERATOR = "3am_operator"
  ```
```

---

## S02 - Core Data Models

```yaml
sprint:
  id: S02
  name: data-models
  title: "Core Data Models (Finding, LayerResult, Report)"
  wave: W0-Foundation
  priority: P0-CRITICAL
  type: implementation

objective: "Implementar models Pydantic para Finding, LayerResult, ValidationReport"

source_parts:
  - "PARTE 4: Features (Data Models)"

deliverables:
  - src/hl_mcp/models/findings.py
  - src/hl_mcp/models/layers.py
  - src/hl_mcp/models/report.py
  - src/hl_mcp/models/targets.py
  - tests/test_models/test_findings.py
  - tests/test_models/test_layers.py
  - tests/test_models/test_report.py

dependencies:
  - S01

gates:
  G0_FILES_EXIST:
    validation: "ls src/hl_mcp/models/findings.py src/hl_mcp/models/layers.py"

  G1_IMPORTS_WORK:
    validation: |
      python -c "
      from hl_mcp.models import Finding, LayerResult, ValidationReport, ValidationTarget
      print('All models imported successfully')
      "

  G2_TESTS_PASS:
    validation: "pytest tests/test_models/ -v"

  G3_COVERAGE:
    validation: "pytest tests/test_models/ --cov=src/hl_mcp/models --cov-fail-under=90"

  G4_SERIALIZATION:
    validation: |
      python -c "
      from hl_mcp.models import Finding, Severity
      f = Finding(severity=Severity.HIGH, description='Test', layer_id=1)
      json_str = f.model_dump_json()
      f2 = Finding.model_validate_json(json_str)
      assert f == f2
      print('Serialization works')
      "

implementation_spec: |
  ```python
  # src/hl_mcp/models/findings.py
  from pydantic import BaseModel, Field
  from typing import Optional, List
  from datetime import datetime
  from .enums import Severity, LayerID

  class Finding(BaseModel):
      """A single finding from a validation layer."""
      id: str = Field(default_factory=lambda: str(uuid4()))
      severity: Severity
      description: str = Field(..., min_length=1)
      layer_id: LayerID
      suggestion: Optional[str] = None
      code_location: Optional[str] = None  # file:line
      evidence: Optional[str] = None
      created_at: datetime = Field(default_factory=datetime.utcnow)

      class Config:
          frozen = True  # Immutable

  # src/hl_mcp/models/layers.py
  class LayerResult(BaseModel):
      """Result from a single layer execution."""
      layer_id: LayerID
      status: LayerStatus
      findings: List[Finding] = Field(default_factory=list)
      duration_ms: int
      tokens_used: int = 0
      run_number: int = 1  # For triple redundancy
      confidence: float = Field(ge=0.0, le=1.0, default=1.0)

      @property
      def has_blocking_findings(self) -> bool:
          return any(f.severity in (Severity.CRITICAL, Severity.HIGH)
                     for f in self.findings)

  # src/hl_mcp/models/report.py
  class ValidationReport(BaseModel):
      """Complete validation report."""
      id: str = Field(default_factory=lambda: str(uuid4()))
      target: ValidationTarget
      layer_results: List[LayerResult]
      consensus_results: List[ConsensusResult]
      final_decision: FinalDecision
      total_duration_ms: int
      total_tokens: int
      created_at: datetime = Field(default_factory=datetime.utcnow)

      @property
      def approved(self) -> bool:
          return self.final_decision.approved

      @property
      def all_findings(self) -> List[Finding]:
          return [f for lr in self.layer_results for f in lr.findings]
  ```
```

---

# WAVE 1: CORE ENGINE (S03-S14)

> **Objetivo**: Implementar o motor de validação completo
> **Duração estimada**: 4-6 semanas
> **Prioridade**: MUST HAVE

---

## S03 - LLM Client Base

```yaml
sprint:
  id: S03
  name: llm-client-base
  title: "LLM Client Base & Interface"
  wave: W1-Core
  priority: P0-CRITICAL
  type: implementation

objective: "Criar interface base para LLM clients e implementação para Claude"

deliverables:
  - src/hl_mcp/llm/__init__.py
  - src/hl_mcp/llm/base.py
  - src/hl_mcp/llm/claude.py
  - src/hl_mcp/llm/config.py
  - tests/test_llm/test_claude.py

dependencies:
  - S02

implementation_spec: |
  ```python
  # src/hl_mcp/llm/base.py
  from abc import ABC, abstractmethod
  from typing import List, Optional
  from pydantic import BaseModel

  class LLMMessage(BaseModel):
      role: str  # "user", "assistant", "system"
      content: str

  class LLMResponse(BaseModel):
      content: str
      tokens_used: int
      model: str
      latency_ms: int

  class BaseLLMClient(ABC):
      """Base class for LLM clients."""

      @abstractmethod
      async def complete(
          self,
          messages: List[LLMMessage],
          temperature: float = 0.7,
          max_tokens: int = 4096,
      ) -> LLMResponse:
          pass

      @abstractmethod
      async def health_check(self) -> bool:
          pass
  ```
```

---

## S04 - LLM Providers (OpenAI, Gemini, Ollama)

```yaml
sprint:
  id: S04
  name: llm-providers
  title: "Additional LLM Providers"
  wave: W1-Core
  priority: P1-HIGH
  type: implementation

objective: "Implementar clients para OpenAI, Gemini e Ollama"

deliverables:
  - src/hl_mcp/llm/openai.py
  - src/hl_mcp/llm/gemini.py
  - src/hl_mcp/llm/ollama.py
  - src/hl_mcp/llm/factory.py
  - tests/test_llm/test_openai.py
  - tests/test_llm/test_gemini.py
  - tests/test_llm/test_ollama.py

dependencies:
  - S03
```

---

## S05 - Layer Base Implementation

```yaml
sprint:
  id: S05
  name: layer-base
  title: "Layer Base Class & Interface"
  wave: W1-Core
  priority: P0-CRITICAL
  type: implementation

objective: "Criar classe base para os 7 Human Layers"

deliverables:
  - src/hl_mcp/layers/__init__.py
  - src/hl_mcp/layers/base.py
  - src/hl_mcp/layers/prompts.py
  - tests/test_layers/test_base.py

dependencies:
  - S03

implementation_spec: |
  ```python
  # src/hl_mcp/layers/base.py
  from abc import ABC, abstractmethod
  from typing import List
  from ..models import LayerResult, Finding, ValidationTarget, LayerID, VetoLevel
  from ..llm import BaseLLMClient

  class BaseLayer(ABC):
      """Base class for Human Layers."""

      layer_id: LayerID
      name: str
      description: str
      default_veto_level: VetoLevel

      def __init__(self, llm_client: BaseLLMClient, veto_level: VetoLevel = None):
          self.llm_client = llm_client
          self.veto_level = veto_level or self.default_veto_level

      @abstractmethod
      async def validate(self, target: ValidationTarget) -> LayerResult:
          """Execute validation for this layer."""
          pass

      @abstractmethod
      def get_prompt(self, target: ValidationTarget) -> str:
          """Generate the prompt for this layer."""
          pass

      def parse_findings(self, llm_response: str) -> List[Finding]:
          """Parse LLM response into structured findings."""
          # Implementation with structured output parsing
          pass
  ```
```

---

## S06-S12 - Individual Layers (HL-1 to HL-7)

```yaml
sprints:
  - id: S06
    name: layer-hl1-ux
    title: "Layer HL-1: UX & Usability"
    deliverables:
      - src/hl_mcp/layers/hl1_ux.py
      - tests/test_layers/test_hl1_ux.py
    dependencies: [S05]

  - id: S07
    name: layer-hl2-functionality
    title: "Layer HL-2: Functionality"
    deliverables:
      - src/hl_mcp/layers/hl2_functionality.py
      - tests/test_layers/test_hl2_functionality.py
    dependencies: [S05]

  - id: S08
    name: layer-hl3-edge-cases
    title: "Layer HL-3: Edge Cases"
    deliverables:
      - src/hl_mcp/layers/hl3_edge_cases.py
      - tests/test_layers/test_hl3_edge_cases.py
    dependencies: [S05]

  - id: S09
    name: layer-hl4-security
    title: "Layer HL-4: Security"
    deliverables:
      - src/hl_mcp/layers/hl4_security.py
      - tests/test_layers/test_hl4_security.py
    dependencies: [S05]

  - id: S10
    name: layer-hl5-performance
    title: "Layer HL-5: Performance"
    deliverables:
      - src/hl_mcp/layers/hl5_performance.py
      - tests/test_layers/test_hl5_performance.py
    dependencies: [S05]

  - id: S11
    name: layer-hl6-compliance
    title: "Layer HL-6: Compliance"
    deliverables:
      - src/hl_mcp/layers/hl6_compliance.py
      - tests/test_layers/test_hl6_compliance.py
    dependencies: [S05]

  - id: S12
    name: layer-hl7-final-review
    title: "Layer HL-7: Final Human Review"
    deliverables:
      - src/hl_mcp/layers/hl7_final_review.py
      - tests/test_layers/test_hl7_final_review.py
    dependencies: [S05]
```

---

## S13 - Consensus Engine

```yaml
sprint:
  id: S13
  name: consensus-engine
  title: "Consensus Engine (Triple Redundancy)"
  wave: W1-Core
  priority: P0-CRITICAL
  type: implementation

objective: "Implementar motor de consenso com 2/3 agreement"

deliverables:
  - src/hl_mcp/core/__init__.py
  - src/hl_mcp/core/consensus.py
  - tests/test_core/test_consensus.py

dependencies:
  - S06
  - S07
  - S08
  - S09
  - S10
  - S11
  - S12

implementation_spec: |
  ```python
  # src/hl_mcp/core/consensus.py
  from typing import List
  from ..models import LayerResult, ConsensusResult, LayerID, LayerStatus

  class ConsensusEngine:
      """Calculates consensus from multiple runs."""

      def __init__(self, threshold: float = 0.67):
          self.threshold = threshold  # 2/3 = 0.67

      def calculate(self, runs: List[LayerResult]) -> ConsensusResult:
          """Calculate consensus from multiple runs of the same layer."""
          if not runs:
              raise ValueError("No runs provided")

          layer_id = runs[0].layer_id
          pass_count = sum(1 for r in runs if r.status.is_success)
          total = len(runs)

          consensus_ratio = pass_count / total
          reached_consensus = consensus_ratio >= self.threshold

          # Aggregate findings from all runs
          all_findings = [f for r in runs for f in r.findings]

          return ConsensusResult(
              layer_id=layer_id,
              runs=runs,
              pass_count=pass_count,
              total_runs=total,
              consensus_ratio=consensus_ratio,
              reached_consensus=reached_consensus,
              final_status=LayerStatus.PASS if reached_consensus else LayerStatus.FAIL,
              aggregated_findings=self._deduplicate_findings(all_findings),
          )

      def _deduplicate_findings(self, findings: List[Finding]) -> List[Finding]:
          # Deduplicate based on description similarity
          pass
  ```
```

---

## S14 - Veto Gate

```yaml
sprint:
  id: S14
  name: veto-gate
  title: "Veto Gate (Final Decision)"
  wave: W1-Core
  priority: P0-CRITICAL
  type: implementation

objective: "Implementar gate de veto que toma decisão final"

deliverables:
  - src/hl_mcp/core/veto.py
  - src/hl_mcp/core/decision.py
  - tests/test_core/test_veto.py
  - tests/test_core/test_decision.py

dependencies:
  - S13

implementation_spec: |
  ```python
  # src/hl_mcp/core/veto.py
  from typing import List
  from ..models import ConsensusResult, FinalDecision, VetoLevel

  class VetoGate:
      """Makes final approval decision based on veto levels."""

      def evaluate(self, consensus_results: List[ConsensusResult]) -> FinalDecision:
          """Evaluate all consensus results and make final decision."""

          blocking_layers = []
          warning_layers = []

          for cr in consensus_results:
              if not cr.reached_consensus:
                  veto_level = cr.layer_id.default_veto_level

                  if veto_level == VetoLevel.STRONG:
                      blocking_layers.append(cr)
                  elif veto_level == VetoLevel.MEDIUM:
                      blocking_layers.append(cr)
                  elif veto_level == VetoLevel.WEAK:
                      warning_layers.append(cr)

          approved = len(blocking_layers) == 0

          return FinalDecision(
              approved=approved,
              blocking_layers=[bl.layer_id for bl in blocking_layers],
              warning_layers=[wl.layer_id for wl in warning_layers],
              reason=self._generate_reason(blocking_layers, warning_layers),
          )
  ```
```

---

# WAVE 2: OSS RELEASE (S15-S24)

> **Objetivo**: Preparar e lançar versão open source
> **Duração estimada**: 2-3 semanas
> **Prioridade**: MUST HAVE

---

## S15-S17 - MCP Server

```yaml
sprints:
  - id: S15
    name: mcp-server-base
    title: "MCP Server Base"
    deliverables:
      - src/hl_mcp/server/__init__.py
      - src/hl_mcp/server/server.py
      - src/hl_mcp/server/config.py
    dependencies: [S14]

  - id: S16
    name: mcp-tools
    title: "MCP Tools (validate, report, etc.)"
    deliverables:
      - src/hl_mcp/server/tools.py
    dependencies: [S15]

  - id: S17
    name: mcp-resources
    title: "MCP Resources"
    deliverables:
      - src/hl_mcp/server/resources.py
    dependencies: [S15]
```

---

## S18-S19 - CLI

```yaml
sprints:
  - id: S18
    name: cli-base
    title: "CLI Base (Typer)"
    deliverables:
      - src/hl_mcp/cli/__init__.py
      - src/hl_mcp/cli/main.py
      - src/hl_mcp/cli/commands/
    dependencies: [S14]

  - id: S19
    name: cli-commands
    title: "CLI Commands (init, validate, serve)"
    deliverables:
      - src/hl_mcp/cli/commands/init.py
      - src/hl_mcp/cli/commands/validate.py
      - src/hl_mcp/cli/commands/serve.py
      - src/hl_mcp/cli/commands/config.py
    dependencies: [S18]
```

---

## S20-S22 - Documentation

```yaml
sprints:
  - id: S20
    name: docs-getting-started
    title: "Docs: Getting Started"
    deliverables:
      - docs/index.md
      - docs/getting-started.md
      - docs/installation.md
      - docs/quickstart.md
    dependencies: [S19]

  - id: S21
    name: docs-guides
    title: "Docs: Guides"
    deliverables:
      - docs/guides/layers.md
      - docs/guides/perspectives.md
      - docs/guides/configuration.md
      - docs/guides/cicd.md
    dependencies: [S20]

  - id: S22
    name: docs-api
    title: "Docs: API Reference"
    deliverables:
      - docs/api/models.md
      - docs/api/layers.md
      - docs/api/cli.md
      - docs/api/mcp.md
    dependencies: [S20]
```

---

## S23 - GitHub Profile & OSS Setup

```yaml
sprint:
  id: S23
  name: github-oss-setup
  title: "GitHub Profile & OSS Setup"
  wave: W2-OSS
  priority: P0-CRITICAL
  type: setup

objective: "Configurar GitHub profile, README, Contributing, etc."

deliverables:
  - README.md (completo, premium)
  - CONTRIBUTING.md
  - CODE_OF_CONDUCT.md
  - SECURITY.md
  - .github/profile/README.md
  - .github/FUNDING.yml

dependencies:
  - S22

source_parts:
  - "PARTE 18: GitHub Profile"
```

---

## S24 - OSS Launch Milestone

```yaml
sprint:
  id: S24
  name: oss-launch
  title: "OSS Launch Milestone"
  wave: W2-OSS
  priority: P0-CRITICAL
  type: milestone

objective: "Lançar versão OSS no PyPI e GitHub"

deliverables:
  - Release v1.0.0 no GitHub
  - Package no PyPI
  - Announcement blog post draft

dependencies:
  - S23

gates:
  G0_ALL_TESTS_PASS:
    validation: "pytest --cov=src --cov-fail-under=85"

  G1_DOCS_COMPLETE:
    validation: "mkdocs build"

  G2_README_COMPLETE:
    validation: "ls README.md CONTRIBUTING.md LICENSE"

  G3_PYPI_READY:
    validation: "python -m build && twine check dist/*"

acceptance_criteria:
  - Todos os testes passando (>85% coverage)
  - Documentação completa
  - README profissional
  - Package no PyPI instalável
  - GitHub Actions funcionando
```

---

# WAVE 3: CLOUD MVP (S25-S35)

> **Objetivo**: Lançar versão cloud com features pagas
> **Duração estimada**: 4-6 semanas
> **Prioridade**: SHOULD HAVE

---

## S25-S27 - Cloud Infrastructure

```yaml
sprints:
  - id: S25
    name: cloud-infrastructure
    title: "Cloud Infrastructure Setup"
    deliverables:
      - docker-compose.yml
      - Dockerfile
      - kubernetes/ (optional)
      - terraform/ (optional)
    dependencies: [S24]

  - id: S26
    name: database-setup
    title: "Database Setup (PostgreSQL)"
    deliverables:
      - src/hl_mcp/cloud/db/
      - alembic/ migrations
    dependencies: [S25]

  - id: S27
    name: auth-setup
    title: "Authentication (Clerk)"
    deliverables:
      - src/hl_mcp/cloud/auth/
    dependencies: [S25]
```

---

## S28-S30 - Cloud API

```yaml
sprints:
  - id: S28
    name: cloud-api-base
    title: "Cloud API Base (FastAPI)"
    deliverables:
      - src/hl_mcp/cloud/api/
    dependencies: [S27]

  - id: S29
    name: cloud-api-endpoints
    title: "Cloud API Endpoints"
    deliverables:
      - src/hl_mcp/cloud/api/routes/
    dependencies: [S28]

  - id: S30
    name: cloud-websocket
    title: "WebSocket for Real-time"
    deliverables:
      - src/hl_mcp/cloud/ws/
    dependencies: [S28]
```

---

## S31-S33 - Dashboard

```yaml
sprints:
  - id: S31
    name: dashboard-setup
    title: "Dashboard Setup (Next.js)"
    deliverables:
      - dashboard/
      - dashboard/package.json
    dependencies: [S29]

  - id: S32
    name: dashboard-pages
    title: "Dashboard Pages"
    deliverables:
      - dashboard/app/
    dependencies: [S31]

  - id: S33
    name: dashboard-components
    title: "Dashboard Components"
    deliverables:
      - dashboard/components/
    dependencies: [S31]
```

---

## S34-S35 - Billing

```yaml
sprints:
  - id: S34
    name: billing-integration
    title: "Billing Integration (Stripe)"
    deliverables:
      - src/hl_mcp/cloud/billing/
    dependencies: [S28]

  - id: S35
    name: billing-tiers
    title: "Billing Tiers & Limits"
    deliverables:
      - src/hl_mcp/cloud/billing/tiers.py
      - src/hl_mcp/cloud/billing/limits.py
    dependencies: [S34]
```

---

# WAVE 4: GROWTH (S36+)

> **Objetivo**: Features avançadas e scale
> **Duração**: Ongoing
> **Prioridade**: COULD HAVE

---

## S36-S40 - Advanced Features

```yaml
sprints:
  - id: S36
    name: perspectives-implementation
    title: "6 Perspectives Implementation"
    deliverables:
      - src/hl_mcp/perspectives/
    dependencies: [S24]

  - id: S37
    name: cicd-integrations
    title: "CI/CD Integrations (GitHub Action, GitLab)"
    deliverables:
      - human-layer-action/
    dependencies: [S24]

  - id: S38
    name: cockpit-visual
    title: "Cockpit Visual Dashboard"
    deliverables:
      - dashboard/app/cockpit/
    dependencies: [S33]

  - id: S39
    name: analytics
    title: "Analytics & Trends"
    deliverables:
      - src/hl_mcp/cloud/analytics/
    dependencies: [S29]

  - id: S40
    name: team-features
    title: "Team Features"
    deliverables:
      - src/hl_mcp/cloud/teams/
    dependencies: [S27]
```

---

# SPRINT INDEX SUMMARY

```yaml
total_sprints: 40
waves:
  W0_Foundation:
    sprints: [S00, S01, S02]
    total: 3
    priority: MUST_HAVE

  W1_Core:
    sprints: [S03, S04, S05, S06, S07, S08, S09, S10, S11, S12, S13, S14]
    total: 12
    priority: MUST_HAVE

  W2_OSS:
    sprints: [S15, S16, S17, S18, S19, S20, S21, S22, S23, S24]
    total: 10
    priority: MUST_HAVE

  W3_Cloud:
    sprints: [S25, S26, S27, S28, S29, S30, S31, S32, S33, S34, S35]
    total: 11
    priority: SHOULD_HAVE

  W4_Growth:
    sprints: [S36, S37, S38, S39, S40]
    total: 5
    priority: COULD_HAVE

critical_path:
  - S00 → S01 → S02 → S03 → S05 → S06-S12 → S13 → S14 → S15-S17 → S18-S19 → S20-S22 → S23 → S24

milestones:
  M1_OSS_LAUNCH:
    sprint: S24
    target: "W0 + W1 + W2 complete"

  M2_CLOUD_MVP:
    sprint: S35
    target: "W3 complete"

  M3_GROWTH:
    sprint: S40
    target: "W4 complete"
```

---

# NEXT: Gerar context_packs/

Após aprovação deste plano, o próximo passo é converter cada sprint em um context_pack compatível com o pipeline, no formato:

```
context_packs/
├── S00_CONTEXT.md
├── S01_CONTEXT.md
├── ...
└── SPRINT_INDEX.yaml
```
