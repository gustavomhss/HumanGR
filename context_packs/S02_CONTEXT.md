# S02 - data-models | Context Pack v1.0

---

## PRODUCT REFERENCE

```yaml
product_reference:
  product_id: "HUMANGR"
  product_name: "Human Layer MCP Server"
  wave: "W0-Foundation"
  product_pack: "./PRODUCT_PACK.md"
  sprint_index: "./SPRINT_INDEX.yaml"
```

---

## RELOAD ANCHOR

```yaml
sprint:
  id: S02
  name: data-models
  title: "Core Data Models"
  wave: W0-Foundation
  priority: P0-CRITICAL
  type: implementation

objective: "Implementar Models Pydantic (Finding, LayerResult, Report)"

source_block: "../MASTER_REQUIREMENTS_MAP.md#PARTE-2-MODELOS"

dependencies:
  - S01  # Enums & Constants

deliverables:
  - src/hl_mcp/models/findings.py
  - src/hl_mcp/models/layers.py
  - src/hl_mcp/models/report.py
  - tests/test_models/test_findings.py
  - tests/test_models/test_layers.py
  - tests/test_models/test_report.py
```

---

## HIERARQUIA DE MODELS

```
Report (topo)
├── LayerResult[] (7 layers)
│   └── Finding[] (múltiplos por layer)
└── metadata
```

---

## IMPLEMENTATION SPECS

### Finding (models/findings.py)

```python
"""Finding model - single finding from a layer."""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

from .enums import Severity, LayerID

class Finding(BaseModel):
    """A single finding from a Human Layer validation."""

    id: str = Field(..., description="Unique finding ID")
    layer_id: LayerID = Field(..., description="Which layer produced this")
    severity: Severity = Field(..., description="Finding severity")
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1)
    suggestion: Optional[str] = Field(None, description="How to fix")
    evidence: Optional[dict] = Field(None, description="Supporting data")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        frozen = True  # Immutable after creation
```

### LayerResult (models/layers.py)

```python
"""LayerResult model - result from a single layer execution."""
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, computed_field

from .enums import LayerID, LayerStatus, VetoLevel, Severity
from .findings import Finding

class LayerResult(BaseModel):
    """Result from executing a single Human Layer."""

    layer_id: LayerID
    status: LayerStatus = Field(default=LayerStatus.PENDING)
    veto_level: VetoLevel = Field(default=VetoLevel.NONE)
    findings: List[Finding] = Field(default_factory=list)
    execution_time_ms: int = Field(default=0, ge=0)
    run_number: int = Field(default=1, ge=1, le=3)  # Triple redundancy
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    error: Optional[str] = Field(None)

    @computed_field
    @property
    def has_critical(self) -> bool:
        """Check if any finding is critical."""
        return any(f.severity == Severity.CRITICAL for f in self.findings)

    @computed_field
    @property
    def can_block(self) -> bool:
        """Check if this result can block progression."""
        return self.veto_level in (VetoLevel.MEDIUM, VetoLevel.STRONG)

    @computed_field
    @property
    def finding_count(self) -> int:
        """Total number of findings."""
        return len(self.findings)
```

### Report (models/report.py)

```python
"""HumanLayerReport - aggregated report from all layers."""
from datetime import datetime
from typing import List, Dict, Optional
from pydantic import BaseModel, Field, computed_field

from .enums import VetoLevel, Severity, LayerID
from .layers import LayerResult

class ReportMetadata(BaseModel):
    """Metadata for a validation report."""
    agent_id: str
    action_type: str
    action_description: str
    context: Optional[dict] = None
    tags: List[str] = Field(default_factory=list)

class HumanLayerReport(BaseModel):
    """Complete report from Human Layer validation."""

    id: str = Field(..., description="Report ID")
    metadata: ReportMetadata
    layer_results: List[LayerResult] = Field(default_factory=list)
    final_decision: str = Field(default="pending")  # approved/rejected/pending
    final_veto_level: VetoLevel = Field(default=VetoLevel.NONE)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    total_execution_time_ms: int = Field(default=0, ge=0)

    @computed_field
    @property
    def is_approved(self) -> bool:
        """Check if the action was approved."""
        return self.final_decision == "approved"

    @computed_field
    @property
    def has_strong_veto(self) -> bool:
        """Check if any layer has STRONG veto."""
        return any(lr.veto_level == VetoLevel.STRONG for lr in self.layer_results)

    @computed_field
    @property
    def total_findings(self) -> int:
        """Total findings across all layers."""
        return sum(lr.finding_count for lr in self.layer_results)

    @computed_field
    @property
    def findings_by_severity(self) -> Dict[str, int]:
        """Count findings by severity."""
        counts = {s.value: 0 for s in Severity}
        for lr in self.layer_results:
            for f in lr.findings:
                counts[f.severity.value] += 1
        return counts

    def get_layer_result(self, layer_id: LayerID) -> Optional[LayerResult]:
        """Get result for a specific layer."""
        for lr in self.layer_results:
            if lr.layer_id == layer_id:
                return lr
        return None
```

---

## INTENT MANIFEST

```yaml
INTENT_MANIFEST:
  RF:
    - RF-001: "Finding deve ter id, layer_id, severity, title, description"
    - RF-002: "LayerResult deve agregar múltiplos Findings"
    - RF-003: "Report deve agregar todos os LayerResults"
    - RF-004: "Todos os models devem ser Pydantic BaseModel"
    - RF-005: "Computed fields para métricas derivadas"

  INV:
    - INV-001: "Finding.severity nunca pode ser None"
    - INV-002: "Finding.title deve ter 1-200 caracteres"
    - INV-003: "Finding é imutável após criação (frozen=True)"
    - INV-004: "LayerResult.run_number deve ser 1, 2 ou 3"
    - INV-005: "Report.final_decision só pode ser approved/rejected/pending"

  EDGE:
    - EDGE-001: "LayerResult com 0 findings (layer passed clean)"
    - EDGE-002: "Report com 0 layer_results (not started)"
    - EDGE-003: "Finding com suggestion=None"
```

---

## GATES

```yaml
gates:
  G0_FILES_EXIST:
    description: "Arquivos de models existem"
    validation: |
      ls src/hl_mcp/models/findings.py
      ls src/hl_mcp/models/layers.py
      ls src/hl_mcp/models/report.py

  G1_IMPORTS_WORK:
    description: "Imports funcionam"
    validation: |
      python -c "from hl_mcp.models import Finding, LayerResult, HumanLayerReport"

  G2_TESTS_PASS:
    description: "Testes passam"
    validation: "pytest tests/test_models/ -v"

  G3_COVERAGE:
    description: "Coverage >= 90%"
    validation: "pytest tests/test_models/ --cov=src/hl_mcp/models --cov-fail-under=90"

  G4_INVARIANTS:
    description: "Invariantes validadas"
    validation: |
      python -c "
      from hl_mcp.models import Finding, LayerResult
      from hl_mcp.models.enums import Severity, LayerID

      # INV-001: severity required
      try:
          Finding(id='f1', layer_id=LayerID.HL1, title='t', description='d')
          assert False, 'Should require severity'
      except:
          pass

      # INV-004: run_number 1-3
      lr = LayerResult(layer_id=LayerID.HL1, run_number=2)
      assert lr.run_number == 2
      "
```

---

## DECISION TREE

```
START S02
│
├─> Criar Finding (models/findings.py)
│   ├─> id, layer_id, severity, title, description
│   ├─> Optional: suggestion, evidence
│   └─> frozen=True (imutável)
│
├─> Criar LayerResult (models/layers.py)
│   ├─> layer_id, status, veto_level, findings[]
│   ├─> execution_time_ms, run_number, timestamp
│   └─> Computed: has_critical, can_block, finding_count
│
├─> Criar HumanLayerReport (models/report.py)
│   ├─> id, metadata, layer_results[]
│   ├─> final_decision, final_veto_level
│   └─> Computed: is_approved, has_strong_veto, total_findings
│
├─> Atualizar models/__init__.py
│   └─> Exportar todas as classes
│
└─> VALIDAR GATES
    ├─> G0: Arquivos existem
    ├─> G1: Imports funcionam
    ├─> G2: Testes passam
    ├─> G3: Coverage >= 90%
    └─> G4: Invariantes
```

---

## REFERÊNCIA

Para detalhes completos, consulte:
- `./S01_CONTEXT.md` - Enums utilizados
- `../MASTER_REQUIREMENTS_MAP.md` - PARTE 2: Modelos
