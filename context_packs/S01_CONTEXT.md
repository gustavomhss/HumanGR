# S01 - enums-constants | Context Pack v1.0

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
  id: S01
  name: enums-constants
  title: "Core Enums & Constants"
  wave: W0-Foundation
  priority: P0-CRITICAL
  type: implementation

objective: "Implementar enums fundamentais: VetoLevel, Severity, LayerStatus, LayerID, PerspectiveID"

source_block: "../MASTER_REQUIREMENTS_MAP.md#PARTE-4-FEATURES"

dependencies:
  - S00  # Project Setup

deliverables:
  - src/hl_mcp/models/__init__.py     # Module exports
  - src/hl_mcp/models/enums.py        # All enums
  - tests/test_models/__init__.py
  - tests/test_models/test_enums.py   # 100% coverage
```

---

## HIERARQUIA DE DEPENDÊNCIAS

```
Todos os Módulos do Sistema
           │
           ▼
        Models
           │
           ▼
        Enums  ← ESTE SPRINT (S01)
```

**IMPORTANTE**: Este é o módulo fundacional. TODOS os outros módulos dependem dele.

---

## INTENT MANIFEST

```yaml
INTENT_MANIFEST:
  RF:
    - RF-001: "VetoLevel tem 4 valores: NONE, WEAK, MEDIUM, STRONG"
    - RF-002: "VetoLevel.can_block_merge() retorna True para MEDIUM e STRONG"
    - RF-003: "VetoLevel.can_block_deploy() retorna True apenas para STRONG"
    - RF-004: "VetoLevel.from_string() converte string para enum"
    - RF-005: "Severity tem 5 valores: CRITICAL, HIGH, MEDIUM, LOW, INFO"
    - RF-006: "Severity é comparável (CRITICAL > HIGH > MEDIUM > LOW > INFO)"
    - RF-007: "LayerStatus tem 7 valores: PENDING, RUNNING, PASS, WARN, FAIL, ERROR, SKIP"
    - RF-008: "LayerStatus.is_terminal() identifica estados finais"
    - RF-009: "LayerStatus.is_success() retorna True para PASS e WARN"
    - RF-010: "LayerID tem 7 valores (HL1 a HL7)"
    - RF-011: "LayerID.default_veto_level() retorna o veto padrão de cada layer"
    - RF-012: "PerspectiveID tem 6 valores para as 6 perspectivas"

  INV:
    - INV-001: "Todos os enums são imutáveis (frozen)"
    - INV-002: "Todos os valores de enum são strings lowercase"
    - INV-003: "VetoLevel.STRONG sempre pode bloquear tudo"
    - INV-004: "HL-4 (Security), HL-6 (Compliance), HL-7 (Final) têm STRONG veto"
    - INV-005: "Severity nunca pode ser None em um Finding"

  EDGE:
    - EDGE-001: "VetoLevel.from_string('INVALID') deve lançar ValueError"
    - EDGE-002: "Comparação de Severity com tipo diferente deve falhar gracefully"
    - EDGE-003: "LayerID(99) deve lançar ValueError"
```

---

## GATES

```yaml
gates:
  G0_FILES_EXIST:
    description: "Arquivos de enums existem"
    validation: |
      ls src/hl_mcp/models/__init__.py
      ls src/hl_mcp/models/enums.py
      ls tests/test_models/test_enums.py

  G1_IMPORTS_WORK:
    description: "Imports funcionam corretamente"
    validation: |
      python -c "
      from hl_mcp.models import VetoLevel, Severity, LayerStatus, LayerID, PerspectiveID
      print('✓ All enums imported successfully')
      "

  G2_VETO_LOGIC:
    description: "Lógica de veto funciona"
    validation: |
      python -c "
      from hl_mcp.models import VetoLevel
      assert VetoLevel.STRONG.can_block_merge == True
      assert VetoLevel.STRONG.can_block_deploy == True
      assert VetoLevel.MEDIUM.can_block_merge == True
      assert VetoLevel.MEDIUM.can_block_deploy == False
      assert VetoLevel.WEAK.can_block_merge == False
      assert VetoLevel.NONE.can_block_merge == False
      print('✓ Veto logic correct')
      "

  G3_SEVERITY_COMPARISON:
    description: "Severity é comparável"
    validation: |
      python -c "
      from hl_mcp.models import Severity
      assert Severity.CRITICAL > Severity.HIGH
      assert Severity.HIGH > Severity.MEDIUM
      assert Severity.MEDIUM > Severity.LOW
      assert Severity.LOW > Severity.INFO
      print('✓ Severity comparison works')
      "

  G4_LAYER_VETO_DEFAULTS:
    description: "Layers têm veto correto"
    validation: |
      python -c "
      from hl_mcp.models import LayerID, VetoLevel
      assert LayerID.HL4_SECURITY.default_veto_level == VetoLevel.STRONG
      assert LayerID.HL6_COMPLIANCE.default_veto_level == VetoLevel.STRONG
      assert LayerID.HL7_FINAL_REVIEW.default_veto_level == VetoLevel.STRONG
      assert LayerID.HL1_UX.default_veto_level == VetoLevel.WEAK
      print('✓ Layer veto defaults correct')
      "

  G5_TESTS_PASS:
    description: "Todos os testes passam"
    validation: "pytest tests/test_models/test_enums.py -v"

  G6_COVERAGE:
    description: "Coverage >= 95%"
    validation: "pytest tests/test_models/test_enums.py --cov=src/hl_mcp/models/enums --cov-fail-under=95"
```

---

## IMPLEMENTATION SPEC

### src/hl_mcp/models/enums.py

```python
"""Core enums for Human Layer validation system."""

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class VetoLevel(Enum):
    """Veto power levels for validation layers.

    Determines what actions a layer can block when validation fails.

    Levels:
        NONE: Informational only, no blocking power
        WEAK: Can add warnings but cannot block
        MEDIUM: Can block merge/PR
        STRONG: Can block everything (merge, deploy, promote)
    """

    NONE = "none"
    WEAK = "weak"
    MEDIUM = "medium"
    STRONG = "strong"

    @property
    def can_block_merge(self) -> bool:
        """Whether this veto level can block merges/PRs."""
        return self in (VetoLevel.MEDIUM, VetoLevel.STRONG)

    @property
    def can_block_deploy(self) -> bool:
        """Whether this veto level can block deployments."""
        return self == VetoLevel.STRONG

    @property
    def can_block_promote(self) -> bool:
        """Whether this veto level can block promotions."""
        return self == VetoLevel.STRONG

    @classmethod
    def from_string(cls, value: str) -> "VetoLevel":
        """Convert string to VetoLevel.

        Args:
            value: String value (case-insensitive)

        Returns:
            Corresponding VetoLevel enum

        Raises:
            ValueError: If value is not a valid VetoLevel
        """
        try:
            return cls(value.lower())
        except ValueError:
            valid = [v.value for v in cls]
            raise ValueError(
                f"Invalid VetoLevel: '{value}'. Valid values: {valid}"
            ) from None


class Severity(Enum):
    """Finding severity levels.

    Used to classify the importance of validation findings.
    Severities are comparable (CRITICAL > HIGH > MEDIUM > LOW > INFO).
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

    @property
    def numeric_value(self) -> int:
        """Numeric value for comparison (higher = more severe)."""
        mapping = {
            Severity.CRITICAL: 5,
            Severity.HIGH: 4,
            Severity.MEDIUM: 3,
            Severity.LOW: 2,
            Severity.INFO: 1,
        }
        return mapping[self]

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self.numeric_value < other.numeric_value

    def __le__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self.numeric_value <= other.numeric_value

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self.numeric_value > other.numeric_value

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self.numeric_value >= other.numeric_value

    @classmethod
    def from_string(cls, value: str) -> "Severity":
        """Convert string to Severity."""
        try:
            return cls(value.lower())
        except ValueError:
            valid = [v.value for v in cls]
            raise ValueError(
                f"Invalid Severity: '{value}'. Valid values: {valid}"
            ) from None


class LayerStatus(Enum):
    """Layer execution status.

    Tracks the current state of a validation layer execution.
    """

    PENDING = "pending"    # Not yet started
    RUNNING = "running"    # Currently executing
    PASS = "pass"          # Completed successfully, no issues
    WARN = "warn"          # Completed with warnings
    FAIL = "fail"          # Completed with failures
    ERROR = "error"        # Execution error (not validation failure)
    SKIP = "skip"          # Skipped (e.g., not applicable)

    @property
    def is_terminal(self) -> bool:
        """Whether this is a terminal (final) state."""
        return self in (
            LayerStatus.PASS,
            LayerStatus.WARN,
            LayerStatus.FAIL,
            LayerStatus.ERROR,
            LayerStatus.SKIP,
        )

    @property
    def is_success(self) -> bool:
        """Whether this status represents a successful outcome."""
        return self in (LayerStatus.PASS, LayerStatus.WARN)

    @property
    def is_failure(self) -> bool:
        """Whether this status represents a failure."""
        return self in (LayerStatus.FAIL, LayerStatus.ERROR)


class LayerID(Enum):
    """The 7 Human Layers for validation.

    Each layer focuses on a specific aspect of validation:
        HL-1: UX & Usability
        HL-2: Functionality
        HL-3: Edge Cases
        HL-4: Security (STRONG veto)
        HL-5: Performance
        HL-6: Compliance (STRONG veto)
        HL-7: Final Human Review (STRONG veto)
    """

    HL1_UX = 1
    HL2_FUNCTIONALITY = 2
    HL3_EDGE_CASES = 3
    HL4_SECURITY = 4
    HL5_PERFORMANCE = 5
    HL6_COMPLIANCE = 6
    HL7_FINAL_REVIEW = 7

    @property
    def name_display(self) -> str:
        """Human-readable name for the layer."""
        names = {
            LayerID.HL1_UX: "UX & Usability",
            LayerID.HL2_FUNCTIONALITY: "Functionality",
            LayerID.HL3_EDGE_CASES: "Edge Cases",
            LayerID.HL4_SECURITY: "Security",
            LayerID.HL5_PERFORMANCE: "Performance",
            LayerID.HL6_COMPLIANCE: "Compliance",
            LayerID.HL7_FINAL_REVIEW: "Final Human Review",
        }
        return names[self]

    @property
    def default_veto_level(self) -> VetoLevel:
        """Default veto level for this layer."""
        strong_veto = (
            LayerID.HL4_SECURITY,
            LayerID.HL6_COMPLIANCE,
            LayerID.HL7_FINAL_REVIEW,
        )
        medium_veto = (
            LayerID.HL2_FUNCTIONALITY,
            LayerID.HL3_EDGE_CASES,
        )

        if self in strong_veto:
            return VetoLevel.STRONG
        elif self in medium_veto:
            return VetoLevel.MEDIUM
        else:
            return VetoLevel.WEAK


class PerspectiveID(Enum):
    """The 6 Testing Perspectives.

    Each perspective simulates a different user persona for multi-angle validation.
    """

    TIRED_USER = "tired_user"
    MALICIOUS_INSIDER = "malicious_insider"
    CONFUSED_NEWBIE = "confused_newbie"
    POWER_USER = "power_user"
    AUDITOR = "auditor"
    THREE_AM_OPERATOR = "3am_operator"

    @property
    def description(self) -> str:
        """Description of this perspective."""
        descriptions = {
            PerspectiveID.TIRED_USER: "User who is tired, distracted, or in a hurry",
            PerspectiveID.MALICIOUS_INSIDER: "User attempting to exploit or misuse the system",
            PerspectiveID.CONFUSED_NEWBIE: "First-time user with no prior knowledge",
            PerspectiveID.POWER_USER: "Expert user pushing the system to its limits",
            PerspectiveID.AUDITOR: "Compliance/security auditor reviewing the system",
            PerspectiveID.THREE_AM_OPERATOR: "On-call operator dealing with production issues",
        }
        return descriptions[self]
```

### src/hl_mcp/models/__init__.py

```python
"""Data models for Human Layer validation system."""

from .enums import (
    LayerID,
    LayerStatus,
    PerspectiveID,
    Severity,
    VetoLevel,
)

__all__ = [
    "VetoLevel",
    "Severity",
    "LayerStatus",
    "LayerID",
    "PerspectiveID",
]
```

### tests/test_models/test_enums.py

```python
"""Tests for core enums."""

import pytest
from hl_mcp.models import VetoLevel, Severity, LayerStatus, LayerID, PerspectiveID


class TestVetoLevel:
    """Tests for VetoLevel enum."""

    def test_values(self):
        """Test all VetoLevel values exist."""
        assert VetoLevel.NONE.value == "none"
        assert VetoLevel.WEAK.value == "weak"
        assert VetoLevel.MEDIUM.value == "medium"
        assert VetoLevel.STRONG.value == "strong"

    def test_can_block_merge(self):
        """Test can_block_merge property."""
        assert VetoLevel.NONE.can_block_merge is False
        assert VetoLevel.WEAK.can_block_merge is False
        assert VetoLevel.MEDIUM.can_block_merge is True
        assert VetoLevel.STRONG.can_block_merge is True

    def test_can_block_deploy(self):
        """Test can_block_deploy property."""
        assert VetoLevel.NONE.can_block_deploy is False
        assert VetoLevel.WEAK.can_block_deploy is False
        assert VetoLevel.MEDIUM.can_block_deploy is False
        assert VetoLevel.STRONG.can_block_deploy is True

    def test_from_string_valid(self):
        """Test from_string with valid values."""
        assert VetoLevel.from_string("none") == VetoLevel.NONE
        assert VetoLevel.from_string("STRONG") == VetoLevel.STRONG
        assert VetoLevel.from_string("Medium") == VetoLevel.MEDIUM

    def test_from_string_invalid(self):
        """Test from_string with invalid value raises ValueError."""
        with pytest.raises(ValueError, match="Invalid VetoLevel"):
            VetoLevel.from_string("invalid")


class TestSeverity:
    """Tests for Severity enum."""

    def test_values(self):
        """Test all Severity values exist."""
        assert len(Severity) == 5
        assert Severity.CRITICAL.value == "critical"
        assert Severity.INFO.value == "info"

    def test_comparison(self):
        """Test severity comparison."""
        assert Severity.CRITICAL > Severity.HIGH
        assert Severity.HIGH > Severity.MEDIUM
        assert Severity.MEDIUM > Severity.LOW
        assert Severity.LOW > Severity.INFO

    def test_comparison_operators(self):
        """Test all comparison operators."""
        assert Severity.CRITICAL >= Severity.HIGH
        assert Severity.HIGH <= Severity.CRITICAL
        assert not (Severity.INFO > Severity.LOW)

    def test_comparison_with_invalid_type(self):
        """Test comparison with invalid type returns NotImplemented."""
        result = Severity.HIGH.__lt__("not_a_severity")
        assert result is NotImplemented


class TestLayerStatus:
    """Tests for LayerStatus enum."""

    def test_is_terminal(self):
        """Test is_terminal property."""
        assert LayerStatus.PENDING.is_terminal is False
        assert LayerStatus.RUNNING.is_terminal is False
        assert LayerStatus.PASS.is_terminal is True
        assert LayerStatus.FAIL.is_terminal is True
        assert LayerStatus.ERROR.is_terminal is True

    def test_is_success(self):
        """Test is_success property."""
        assert LayerStatus.PASS.is_success is True
        assert LayerStatus.WARN.is_success is True
        assert LayerStatus.FAIL.is_success is False
        assert LayerStatus.ERROR.is_success is False


class TestLayerID:
    """Tests for LayerID enum."""

    def test_seven_layers(self):
        """Test that there are exactly 7 layers."""
        assert len(LayerID) == 7

    def test_default_veto_levels(self):
        """Test default veto levels for each layer."""
        # STRONG veto layers
        assert LayerID.HL4_SECURITY.default_veto_level == VetoLevel.STRONG
        assert LayerID.HL6_COMPLIANCE.default_veto_level == VetoLevel.STRONG
        assert LayerID.HL7_FINAL_REVIEW.default_veto_level == VetoLevel.STRONG

        # MEDIUM veto layers
        assert LayerID.HL2_FUNCTIONALITY.default_veto_level == VetoLevel.MEDIUM
        assert LayerID.HL3_EDGE_CASES.default_veto_level == VetoLevel.MEDIUM

        # WEAK veto layers
        assert LayerID.HL1_UX.default_veto_level == VetoLevel.WEAK
        assert LayerID.HL5_PERFORMANCE.default_veto_level == VetoLevel.WEAK

    def test_name_display(self):
        """Test human-readable names."""
        assert LayerID.HL1_UX.name_display == "UX & Usability"
        assert LayerID.HL4_SECURITY.name_display == "Security"


class TestPerspectiveID:
    """Tests for PerspectiveID enum."""

    def test_six_perspectives(self):
        """Test that there are exactly 6 perspectives."""
        assert len(PerspectiveID) == 6

    def test_descriptions(self):
        """Test that all perspectives have descriptions."""
        for perspective in PerspectiveID:
            assert perspective.description
            assert len(perspective.description) > 10
```

---

## DECISION TREE

```
START S01
│
├─> Criar src/hl_mcp/models/enums.py
│   ├─> VetoLevel (4 valores + métodos)
│   ├─> Severity (5 valores + comparação)
│   ├─> LayerStatus (7 valores + propriedades)
│   ├─> LayerID (7 valores + veto default)
│   └─> PerspectiveID (6 valores + descriptions)
│
├─> Criar src/hl_mcp/models/__init__.py
│   └─> Exportar todos os enums
│
├─> Criar tests/test_models/test_enums.py
│   ├─> TestVetoLevel
│   ├─> TestSeverity
│   ├─> TestLayerStatus
│   ├─> TestLayerID
│   └─> TestPerspectiveID
│
└─> VALIDAR GATES
    ├─> G0: Arquivos existem
    ├─> G1: Imports funcionam
    ├─> G2: Lógica de veto
    ├─> G3: Comparação de severity
    ├─> G4: Veto defaults dos layers
    ├─> G5: Testes passam
    └─> G6: Coverage >= 95%
```

---

## REFERÊNCIA

Para detalhes completos, consulte:
- `../MASTER_REQUIREMENTS_MAP.md` - PARTE 4: Features
- `../SPRINT_PLAN.md` - Sprint S01 details
