# Human Layer MCP Server - Roadmap Block 04
# DATA MODELS IMPLEMENTATION

> **Objetivo**: Implementar os 7 módulos de Data Models (HLS-MDL-*)
> **Versão**: 1.0.0 | Data: 2026-02-01

---

## IMPLEMENTAÇÃO: CAMADA DE MODELS

A camada de Models é a **fundação** de todo o sistema. Todos os outros módulos dependem dela.

```
Dependência: Todos os módulos → Models → Enums
```

---

## 1. HLS-MDL-007: Enums (PRIMEIRO MÓDULO)

```python
# src/hl_mcp/models/enums.py
"""
Module: HLS-MDL-007 - Enums
===========================

All enumerations used across Human Layer MCP.

This is the foundational module - all other models depend on it.
No dependencies on other hl_mcp modules.

Example:
    >>> from hl_mcp.models.enums import VetoLevel, Severity
    >>> veto = VetoLevel.STRONG
    >>> severity = Severity.CRITICAL

Dependencies:
    - None (foundation module)

Author: Human Layer Team
Version: 1.0.0
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Dict, List


class VetoLevel(Enum):
    """Veto power levels for Human Layers.

    Veto levels determine what a layer can block:
    - NONE: No veto power (informational only)
    - WEAK: Can add warnings but not block
    - MEDIUM: Can block merge but not promotion
    - STRONG: Can block everything (promotion and merge)

    Example:
        >>> layer_veto = VetoLevel.STRONG
        >>> if layer_veto == VetoLevel.STRONG:
        ...     print("This layer can block everything")
    """

    NONE = "none"
    WEAK = "weak"
    MEDIUM = "medium"
    STRONG = "strong"

    @property
    def can_block_promotion(self) -> bool:
        """Check if this veto level can block promotion to production."""
        return self == VetoLevel.STRONG

    @property
    def can_block_merge(self) -> bool:
        """Check if this veto level can block merge."""
        return self in (VetoLevel.STRONG, VetoLevel.MEDIUM)

    @property
    def display_name(self) -> str:
        """Human-readable display name."""
        return self.value.upper()

    @classmethod
    def from_string(cls, value: str) -> "VetoLevel":
        """Create from string value (case-insensitive)."""
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(f"Invalid VetoLevel: {value}. Valid: {[v.value for v in cls]}")


class LayerStatus(Enum):
    """Execution status of a Human Layer.

    Status flow: PENDING → RUNNING → PASS/WARN/FAIL/ERROR/SKIP

    Example:
        >>> status = LayerStatus.PASS
        >>> if status.is_success:
        ...     print("Layer passed!")
    """

    PENDING = "pending"
    RUNNING = "running"
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    ERROR = "error"
    SKIP = "skip"
    PRECONDITION_NOT_MET = "precondition_not_met"

    @property
    def is_success(self) -> bool:
        """Check if status indicates success."""
        return self in (LayerStatus.PASS, LayerStatus.WARN, LayerStatus.SKIP)

    @property
    def is_failure(self) -> bool:
        """Check if status indicates failure."""
        return self in (LayerStatus.FAIL, LayerStatus.ERROR)

    @property
    def is_terminal(self) -> bool:
        """Check if status is terminal (no more processing)."""
        return self not in (LayerStatus.PENDING, LayerStatus.RUNNING)

    @property
    def emoji(self) -> str:
        """Emoji representation for display."""
        return {
            LayerStatus.PENDING: "⏳",
            LayerStatus.RUNNING: "🔄",
            LayerStatus.PASS: "✅",
            LayerStatus.WARN: "⚠️",
            LayerStatus.FAIL: "❌",
            LayerStatus.ERROR: "💥",
            LayerStatus.SKIP: "⏭️",
            LayerStatus.PRECONDITION_NOT_MET: "🚫",
        }[self]


class Severity(Enum):
    """Severity levels for findings.

    Ordered from most severe to least severe:
    CRITICAL > HIGH > MEDIUM > LOW > INFO

    Example:
        >>> finding_severity = Severity.HIGH
        >>> if finding_severity >= Severity.HIGH:
        ...     print("Needs immediate attention")
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

    @property
    def priority_order(self) -> int:
        """Numeric priority (lower = more severe)."""
        return {
            Severity.CRITICAL: 0,
            Severity.HIGH: 1,
            Severity.MEDIUM: 2,
            Severity.LOW: 3,
            Severity.INFO: 4,
        }[self]

    def __lt__(self, other: "Severity") -> bool:
        """Compare severities (CRITICAL < HIGH means CRITICAL is more severe)."""
        if not isinstance(other, Severity):
            return NotImplemented
        return self.priority_order < other.priority_order

    def __le__(self, other: "Severity") -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self.priority_order <= other.priority_order

    def __gt__(self, other: "Severity") -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self.priority_order > other.priority_order

    def __ge__(self, other: "Severity") -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self.priority_order >= other.priority_order

    @property
    def color(self) -> str:
        """ANSI color code for terminal output."""
        return {
            Severity.CRITICAL: "\033[91m",  # Red
            Severity.HIGH: "\033[93m",      # Yellow
            Severity.MEDIUM: "\033[94m",    # Blue
            Severity.LOW: "\033[96m",       # Cyan
            Severity.INFO: "\033[90m",      # Gray
        }[self]

    @property
    def emoji(self) -> str:
        """Emoji representation."""
        return {
            Severity.CRITICAL: "🔴",
            Severity.HIGH: "🟠",
            Severity.MEDIUM: "🟡",
            Severity.LOW: "🟢",
            Severity.INFO: "🔵",
        }[self]


class HITLDecision(Enum):
    """Human-in-the-Loop decision from triage.

    Determines whether an item needs human review:
    - SKIP: Auto-approve, no human needed
    - ASYNC_QUEUE: Queue for later human review
    - SYNC_REQUIRED: Needs immediate human review

    Example:
        >>> decision = HITLDecision.SYNC_REQUIRED
        >>> if decision == HITLDecision.SYNC_REQUIRED:
        ...     await notify_human_reviewer()
    """

    SKIP = "skip"
    ASYNC_QUEUE = "async_queue"
    SYNC_REQUIRED = "sync_required"

    @property
    def needs_human(self) -> bool:
        """Check if human involvement is needed."""
        return self != HITLDecision.SKIP

    @property
    def is_urgent(self) -> bool:
        """Check if immediate human review is needed."""
        return self == HITLDecision.SYNC_REQUIRED


class ReviewTrigger(Enum):
    """Triggers that invoke human review.

    Example:
        >>> triggers = [ReviewTrigger.LOW_CONFIDENCE, ReviewTrigger.HIGH_RISK]
        >>> if ReviewTrigger.SECURITY_CONCERN in triggers:
        ...     escalate_to_security_team()
    """

    LOW_CONFIDENCE = "low_confidence"
    NOVELTY_DETECTED = "novelty_detected"
    HIGH_RISK = "high_risk"
    CRITICAL_PATH = "critical_path"
    SECURITY_CONCERN = "security_concern"
    PRECEDENT_MISMATCH = "precedent_mismatch"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    DOMAIN_BOUNDARY = "domain_boundary"
    BUDGET_EXCEEDED = "budget_exceeded"
    TRUST_THRESHOLD = "trust_threshold"

    @property
    def is_security_related(self) -> bool:
        """Check if trigger is security-related."""
        return self in (
            ReviewTrigger.SECURITY_CONCERN,
            ReviewTrigger.HIGH_RISK,
        )


class FeedbackType(Enum):
    """Types of human feedback on AI decisions.

    Example:
        >>> feedback = FeedbackType.OVERRIDE
        >>> trust_adjustment = feedback.trust_delta
    """

    AGREE = "agree"           # Human agreed with AI
    PARTIAL = "partial"       # Partial agreement
    OVERRIDE = "override"     # Human overrode AI decision
    ESCALATE = "escalate"     # Human escalated to higher level

    @property
    def trust_delta(self) -> float:
        """Trust score adjustment for this feedback type."""
        return {
            FeedbackType.AGREE: +0.02,
            FeedbackType.PARTIAL: +0.01,
            FeedbackType.OVERRIDE: -0.05,
            FeedbackType.ESCALATE: -0.03,
        }[self]


class BrowserAction(Enum):
    """Browser automation actions.

    Example:
        >>> action = BrowserAction.CLICK
        >>> await browser.execute(action, selector="#btn")
    """

    GOTO = "goto"
    CLICK = "click"
    FILL = "fill"
    WAIT = "wait"
    SCREENSHOT = "screenshot"
    SCROLL = "scroll"
    HOVER = "hover"
    SELECT = "select"
    CHECK = "check"
    UNCHECK = "uncheck"

    @property
    def requires_value(self) -> bool:
        """Check if action requires a value parameter."""
        return self in (BrowserAction.FILL, BrowserAction.SELECT)

    @property
    def requires_selector(self) -> bool:
        """Check if action requires a selector."""
        return self not in (BrowserAction.GOTO, BrowserAction.SCREENSHOT)


class LayerID(Enum):
    """Human Layer identifiers.

    Example:
        >>> layer = LayerID.HL_5_SEGURANCA
        >>> print(f"Running {layer.display_name}")
    """

    HL_1_USUARIO = "HL-1"
    HL_2_OPERADOR = "HL-2"
    HL_3_MANTENEDOR = "HL-3"
    HL_4_DECISOR = "HL-4"
    HL_5_SEGURANCA = "HL-5"
    HL_6_HACKER = "HL-6"
    HL_7_SIMPLIFICADOR = "HL-7"

    @property
    def display_name(self) -> str:
        """Human-readable name."""
        names = {
            LayerID.HL_1_USUARIO: "Humano Usuario",
            LayerID.HL_2_OPERADOR: "Humano Operador",
            LayerID.HL_3_MANTENEDOR: "Humano Mantenedor",
            LayerID.HL_4_DECISOR: "Humano Decisor",
            LayerID.HL_5_SEGURANCA: "Humano Seguranca",
            LayerID.HL_6_HACKER: "Humano Hacker",
            LayerID.HL_7_SIMPLIFICADOR: "Humano Simplificador",
        }
        return names[self]

    @property
    def veto_level(self) -> VetoLevel:
        """Default veto level for this layer."""
        levels = {
            LayerID.HL_1_USUARIO: VetoLevel.WEAK,
            LayerID.HL_2_OPERADOR: VetoLevel.MEDIUM,
            LayerID.HL_3_MANTENEDOR: VetoLevel.MEDIUM,
            LayerID.HL_4_DECISOR: VetoLevel.STRONG,
            LayerID.HL_5_SEGURANCA: VetoLevel.STRONG,
            LayerID.HL_6_HACKER: VetoLevel.STRONG,
            LayerID.HL_7_SIMPLIFICADOR: VetoLevel.WEAK,
        }
        return levels[self]

    @property
    def focus_areas(self) -> List[str]:
        """Primary focus areas for this layer."""
        areas = {
            LayerID.HL_1_USUARIO: ["usability", "UX", "friction", "clarity"],
            LayerID.HL_2_OPERADOR: ["diagnostics", "runbooks", "rollback", "observability"],
            LayerID.HL_3_MANTENEDOR: ["code_quality", "tests", "docs", "tech_debt"],
            LayerID.HL_4_DECISOR: ["strategy", "ROI", "trust", "alignment"],
            LayerID.HL_5_SEGURANCA: ["safety", "data_protection", "fail_safe", "blast_radius"],
            LayerID.HL_6_HACKER: ["exploits", "attack_vectors", "abuse", "injection"],
            LayerID.HL_7_SIMPLIFICADOR: ["YAGNI", "complexity", "minimalism", "cleanup"],
        }
        return areas[self]

    @classmethod
    def security_layers(cls) -> List["LayerID"]:
        """Get layers focused on security."""
        return [cls.HL_4_DECISOR, cls.HL_5_SEGURANCA, cls.HL_6_HACKER]

    @classmethod
    def usability_layers(cls) -> List["LayerID"]:
        """Get layers focused on usability."""
        return [cls.HL_1_USUARIO, cls.HL_7_SIMPLIFICADOR]


class PerspectiveID(Enum):
    """Perspective identifiers for 6-perspective testing.

    Example:
        >>> perspective = PerspectiveID.MALICIOUS_INSIDER
        >>> if perspective.weight > 1.0:
        ...     print("High priority perspective")
    """

    TIRED_USER = "tired_user"
    MALICIOUS_INSIDER = "malicious_insider"
    CONFUSED_NEWBIE = "confused_newbie"
    POWER_USER = "power_user"
    AUDITOR = "auditor"
    THREE_AM_OPERATOR = "3am_operator"

    @property
    def weight(self) -> float:
        """Weight in consensus scoring."""
        weights = {
            PerspectiveID.TIRED_USER: 1.0,
            PerspectiveID.MALICIOUS_INSIDER: 1.5,  # Security = higher
            PerspectiveID.CONFUSED_NEWBIE: 1.0,
            PerspectiveID.POWER_USER: 0.8,
            PerspectiveID.AUDITOR: 1.2,
            PerspectiveID.THREE_AM_OPERATOR: 1.0,
        }
        return weights[self]

    @property
    def display_name(self) -> str:
        """Human-readable name."""
        names = {
            PerspectiveID.TIRED_USER: "Tired User",
            PerspectiveID.MALICIOUS_INSIDER: "Malicious Insider",
            PerspectiveID.CONFUSED_NEWBIE: "Confused Newbie",
            PerspectiveID.POWER_USER: "Power User",
            PerspectiveID.AUDITOR: "Auditor",
            PerspectiveID.THREE_AM_OPERATOR: "3AM Operator",
        }
        return names[self]


class RecipientType(Enum):
    """Types of error message recipients.

    Example:
        >>> recipient = RecipientType.DEVELOPER
        >>> error.format_for(recipient)
    """

    DEVELOPER = "developer"
    OPERATOR = "operator"
    MANAGER = "manager"
    USER = "user"
    AUDITOR = "auditor"


class TargetType(Enum):
    """Types of validation targets.

    Example:
        >>> target = TargetType.USER_JOURNEY
        >>> if target == TargetType.USER_JOURNEY:
        ...     use_browser_automation()
    """

    CODE = "code"
    SPEC = "spec"
    CONFIG = "config"
    USER_JOURNEY = "user_journey"
    API = "api"
    DOCUMENTATION = "documentation"


class LayerPack(Enum):
    """Pre-defined layer packs.

    Example:
        >>> pack = LayerPack.SECURITY
        >>> layers = pack.get_layers()
    """

    FULL = "FULL"
    SECURITY = "SECURITY"
    USABILITY = "USABILITY"
    MINIMAL = "MINIMAL"
    CUSTOM = "CUSTOM"

    def get_layers(self) -> List[LayerID]:
        """Get layers included in this pack."""
        packs = {
            LayerPack.FULL: list(LayerID),
            LayerPack.SECURITY: [LayerID.HL_4_DECISOR, LayerID.HL_5_SEGURANCA, LayerID.HL_6_HACKER],
            LayerPack.USABILITY: [LayerID.HL_1_USUARIO, LayerID.HL_7_SIMPLIFICADOR],
            LayerPack.MINIMAL: [LayerID.HL_4_DECISOR],
            LayerPack.CUSTOM: [],  # User-defined
        }
        return packs[self]


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Veto and Status
    "VetoLevel",
    "LayerStatus",
    "Severity",
    # HITL
    "HITLDecision",
    "ReviewTrigger",
    "FeedbackType",
    # Browser
    "BrowserAction",
    # Identifiers
    "LayerID",
    "PerspectiveID",
    "RecipientType",
    "TargetType",
    "LayerPack",
]
```

---

## 2. HLS-MDL-001: Finding

```python
# src/hl_mcp/models/finding.py
"""
Module: HLS-MDL-001 - Finding
=============================

Finding and SecurityFinding models for validation results.

A Finding represents an issue discovered during Human Layer validation.
SecurityFinding extends Finding with security-specific metadata.

Example:
    >>> finding = Finding(
    ...     layer=LayerID.HL_5_SEGURANCA,
    ...     severity=Severity.HIGH,
    ...     title="SQL Injection Vulnerability",
    ...     description="User input not sanitized",
    ...     fix_hint="Use parameterized queries",
    ... )

Dependencies:
    - HLS-MDL-007: Enums (Severity, LayerID)

Author: Human Layer Team
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import hashlib
import uuid

from .enums import Severity, LayerID


@dataclass
class Finding:
    """A finding from Human Layer validation.

    Represents an issue, concern, or observation discovered
    during layer execution.

    Attributes:
        id: Unique identifier
        layer: Which layer found this
        severity: Issue severity
        title: Short title
        description: Detailed description
        location: File/line if applicable
        fix_hint: How to fix (REQUIRED)
        evidence: Supporting evidence
        tags: Categorization tags

    Invariant:
        fix_hint must always be provided (INV-HLX-004)

    Example:
        >>> f = Finding.create(
        ...     layer=LayerID.HL_1_USUARIO,
        ...     severity=Severity.MEDIUM,
        ...     title="Confusing Error Message",
        ...     description="Error shows stack trace to user",
        ...     fix_hint="Replace with user-friendly message",
        ... )
    """

    # Required fields
    layer: LayerID
    severity: Severity
    title: str
    description: str
    fix_hint: str  # REQUIRED per INV-HLX-004

    # Optional fields
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    location: Optional[str] = None
    evidence: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate finding after initialization."""
        # INV-HLX-004: fix_hint must be provided
        if not self.fix_hint or not self.fix_hint.strip():
            raise ValueError("Finding must have a fix_hint (INV-HLX-004)")

        # Validate severity
        if not isinstance(self.severity, Severity):
            self.severity = Severity(self.severity)

        # Validate layer
        if not isinstance(self.layer, LayerID):
            self.layer = LayerID(self.layer)

    @classmethod
    def create(
        cls,
        layer: LayerID,
        severity: Severity,
        title: str,
        description: str,
        fix_hint: str,
        **kwargs: Any,
    ) -> "Finding":
        """Factory method to create a finding.

        Args:
            layer: Source layer
            severity: Issue severity
            title: Short title
            description: Full description
            fix_hint: How to fix
            **kwargs: Additional fields

        Returns:
            New Finding instance
        """
        return cls(
            layer=layer,
            severity=severity,
            title=title,
            description=description,
            fix_hint=fix_hint,
            **kwargs,
        )

    @property
    def is_blocking(self) -> bool:
        """Check if this finding should block progress."""
        return self.severity in (Severity.CRITICAL, Severity.HIGH)

    @property
    def content_hash(self) -> str:
        """Hash of finding content for deduplication."""
        content = f"{self.layer.value}:{self.title}:{self.description}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "layer": self.layer.value,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "fix_hint": self.fix_hint,
            "location": self.location,
            "evidence": self.evidence,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Finding":
        """Deserialize from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())[:12]),
            layer=LayerID(data["layer"]),
            severity=Severity(data["severity"]),
            title=data["title"],
            description=data["description"],
            fix_hint=data["fix_hint"],
            location=data.get("location"),
            evidence=data.get("evidence", []),
            tags=data.get("tags", []),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            metadata=data.get("metadata", {}),
        )

    def __str__(self) -> str:
        """String representation."""
        return f"[{self.severity.emoji} {self.severity.value.upper()}] {self.title}"


@dataclass
class SecurityFinding(Finding):
    """Security-specific finding with OWASP/CWE metadata.

    Extends Finding with security-specific fields for
    vulnerability tracking and compliance.

    Attributes:
        cwe_id: CWE identifier (e.g., "CWE-79")
        owasp_category: OWASP Top 10 category (e.g., "A03")
        attack_vector: Description of attack vector
        exploitability: How easy to exploit (0.0-1.0)
        remediation_priority: When to fix

    Example:
        >>> sf = SecurityFinding.create(
        ...     layer=LayerID.HL_6_HACKER,
        ...     severity=Severity.CRITICAL,
        ...     title="XSS in Comment Field",
        ...     description="User input reflected without encoding",
        ...     fix_hint="Use HTML encoding on output",
        ...     cwe_id="CWE-79",
        ...     owasp_category="A03",
        ...     attack_vector="Inject script via comment",
        ... )
    """

    # Security-specific fields
    cwe_id: Optional[str] = None  # CWE-79, CWE-89, etc
    owasp_category: Optional[str] = None  # A01, A02, etc
    attack_vector: str = ""
    exploitability: float = 0.5  # 0.0 = hard, 1.0 = trivial
    remediation_priority: str = "soon"  # immediate, soon, scheduled

    def __post_init__(self) -> None:
        """Validate security finding."""
        super().__post_init__()

        # Security findings from HL-5 or HL-6 should have high priority
        if self.layer in (LayerID.HL_5_SEGURANCA, LayerID.HL_6_HACKER):
            if self.severity >= Severity.HIGH and "security" not in self.tags:
                self.tags.append("security")

        # Validate exploitability range
        if not 0.0 <= self.exploitability <= 1.0:
            raise ValueError(f"exploitability must be 0.0-1.0, got {self.exploitability}")

    @classmethod
    def create(
        cls,
        layer: LayerID,
        severity: Severity,
        title: str,
        description: str,
        fix_hint: str,
        attack_vector: str = "",
        cwe_id: Optional[str] = None,
        owasp_category: Optional[str] = None,
        **kwargs: Any,
    ) -> "SecurityFinding":
        """Factory method for security findings."""
        return cls(
            layer=layer,
            severity=severity,
            title=title,
            description=description,
            fix_hint=fix_hint,
            attack_vector=attack_vector,
            cwe_id=cwe_id,
            owasp_category=owasp_category,
            **kwargs,
        )

    @property
    def cvss_estimate(self) -> float:
        """Rough CVSS score estimate based on severity and exploitability."""
        base_scores = {
            Severity.CRITICAL: 9.0,
            Severity.HIGH: 7.0,
            Severity.MEDIUM: 5.0,
            Severity.LOW: 3.0,
            Severity.INFO: 1.0,
        }
        base = base_scores[self.severity]
        # Adjust by exploitability
        adjusted = base * (0.7 + 0.3 * self.exploitability)
        return min(10.0, round(adjusted, 1))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize including security fields."""
        data = super().to_dict()
        data.update({
            "cwe_id": self.cwe_id,
            "owasp_category": self.owasp_category,
            "attack_vector": self.attack_vector,
            "exploitability": self.exploitability,
            "remediation_priority": self.remediation_priority,
            "cvss_estimate": self.cvss_estimate,
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SecurityFinding":
        """Deserialize from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())[:12]),
            layer=LayerID(data["layer"]),
            severity=Severity(data["severity"]),
            title=data["title"],
            description=data["description"],
            fix_hint=data["fix_hint"],
            location=data.get("location"),
            evidence=data.get("evidence", []),
            tags=data.get("tags", []),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            metadata=data.get("metadata", {}),
            cwe_id=data.get("cwe_id"),
            owasp_category=data.get("owasp_category"),
            attack_vector=data.get("attack_vector", ""),
            exploitability=data.get("exploitability", 0.5),
            remediation_priority=data.get("remediation_priority", "soon"),
        )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def findings_by_severity(findings: List[Finding]) -> Dict[Severity, List[Finding]]:
    """Group findings by severity level."""
    result: Dict[Severity, List[Finding]] = {s: [] for s in Severity}
    for f in findings:
        result[f.severity].append(f)
    return result


def count_blocking_findings(findings: List[Finding]) -> int:
    """Count findings that block progress."""
    return sum(1 for f in findings if f.is_blocking)


def deduplicate_findings(findings: List[Finding]) -> List[Finding]:
    """Remove duplicate findings based on content hash."""
    seen: Dict[str, Finding] = {}
    for f in findings:
        h = f.content_hash
        if h not in seen:
            seen[h] = f
        elif f.severity < seen[h].severity:  # Keep more severe
            seen[h] = f
    return list(seen.values())


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "Finding",
    "SecurityFinding",
    "findings_by_severity",
    "count_blocking_findings",
    "deduplicate_findings",
]
```

---

## 3. HLS-MDL-002: LayerResult

```python
# src/hl_mcp/models/layer_result.py
"""
Module: HLS-MDL-002 - LayerResult
=================================

Result model for Human Layer execution.

Captures the complete outcome of running a single layer,
including status, veto, findings, and timing.

Example:
    >>> result = LayerResult(
    ...     layer_id=LayerID.HL_4_DECISOR,
    ...     status=LayerStatus.WARN,
    ...     veto=VetoLevel.NONE,
    ...     findings=[...],
    ... )

Dependencies:
    - HLS-MDL-001: Finding
    - HLS-MDL-007: Enums

Author: Human Layer Team
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .enums import LayerID, LayerStatus, VetoLevel, Severity
from .finding import Finding, SecurityFinding


@dataclass
class LayerResult:
    """Result of executing a single Human Layer.

    Attributes:
        layer_id: Which layer was executed
        status: Execution status
        veto: Veto level exercised
        findings: Issues found
        security_findings: Security-specific issues
        execution_time_ms: How long it took
        consensus_score: Agreement across redundant runs
        runs: Number of executions (triple redundancy)

    Example:
        >>> result = LayerResult.passed(LayerID.HL_1_USUARIO)
        >>> result.has_findings
        False
    """

    layer_id: LayerID
    status: LayerStatus
    veto: VetoLevel = VetoLevel.NONE

    findings: List[Finding] = field(default_factory=list)
    security_findings: List[SecurityFinding] = field(default_factory=list)

    execution_time_ms: int = 0
    consensus_score: float = 1.0  # 0.0-1.0
    runs: int = 1

    started_at: Optional[datetime] = None
    completed_at: datetime = field(default_factory=datetime.utcnow)

    raw_output: Optional[str] = None  # LLM raw output
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate result."""
        if not isinstance(self.layer_id, LayerID):
            self.layer_id = LayerID(self.layer_id)
        if not isinstance(self.status, LayerStatus):
            self.status = LayerStatus(self.status)
        if not isinstance(self.veto, VetoLevel):
            self.veto = VetoLevel(self.veto)

    # =========================================================================
    # FACTORY METHODS
    # =========================================================================

    @classmethod
    def passed(
        cls,
        layer_id: LayerID,
        execution_time_ms: int = 0,
        **kwargs: Any,
    ) -> "LayerResult":
        """Create a passed result."""
        return cls(
            layer_id=layer_id,
            status=LayerStatus.PASS,
            veto=VetoLevel.NONE,
            execution_time_ms=execution_time_ms,
            **kwargs,
        )

    @classmethod
    def failed(
        cls,
        layer_id: LayerID,
        findings: List[Finding],
        veto: VetoLevel = VetoLevel.MEDIUM,
        **kwargs: Any,
    ) -> "LayerResult":
        """Create a failed result with findings."""
        return cls(
            layer_id=layer_id,
            status=LayerStatus.FAIL,
            veto=veto,
            findings=findings,
            **kwargs,
        )

    @classmethod
    def warned(
        cls,
        layer_id: LayerID,
        findings: List[Finding],
        **kwargs: Any,
    ) -> "LayerResult":
        """Create a warned result."""
        return cls(
            layer_id=layer_id,
            status=LayerStatus.WARN,
            veto=VetoLevel.WEAK,
            findings=findings,
            **kwargs,
        )

    @classmethod
    def error(
        cls,
        layer_id: LayerID,
        error_message: str,
        **kwargs: Any,
    ) -> "LayerResult":
        """Create an error result."""
        return cls(
            layer_id=layer_id,
            status=LayerStatus.ERROR,
            veto=VetoLevel.NONE,
            metadata={"error": error_message},
            **kwargs,
        )

    @classmethod
    def skipped(
        cls,
        layer_id: LayerID,
        reason: str,
        **kwargs: Any,
    ) -> "LayerResult":
        """Create a skipped result."""
        return cls(
            layer_id=layer_id,
            status=LayerStatus.SKIP,
            veto=VetoLevel.NONE,
            metadata={"skip_reason": reason},
            **kwargs,
        )

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def has_findings(self) -> bool:
        """Check if there are any findings."""
        return bool(self.findings or self.security_findings)

    @property
    def all_findings(self) -> List[Finding]:
        """Get all findings (regular + security)."""
        return self.findings + self.security_findings

    @property
    def critical_count(self) -> int:
        """Count critical findings."""
        return sum(1 for f in self.all_findings if f.severity == Severity.CRITICAL)

    @property
    def high_count(self) -> int:
        """Count high severity findings."""
        return sum(1 for f in self.all_findings if f.severity == Severity.HIGH)

    @property
    def blocks_promotion(self) -> bool:
        """Check if this result blocks promotion to production."""
        return self.veto.can_block_promotion or self.critical_count > 0

    @property
    def blocks_merge(self) -> bool:
        """Check if this result blocks merge."""
        return self.veto.can_block_merge or self.critical_count > 0 or self.high_count > 0

    @property
    def summary(self) -> str:
        """One-line summary of result."""
        findings_str = ""
        if self.has_findings:
            findings_str = f" ({len(self.all_findings)} findings)"
        return f"{self.layer_id.display_name}: {self.status.emoji} {self.status.value.upper()}{findings_str}"

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "layer_id": self.layer_id.value,
            "layer_name": self.layer_id.display_name,
            "status": self.status.value,
            "veto": self.veto.value,
            "findings": [f.to_dict() for f in self.findings],
            "security_findings": [f.to_dict() for f in self.security_findings],
            "execution_time_ms": self.execution_time_ms,
            "consensus_score": self.consensus_score,
            "runs": self.runs,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat(),
            "metadata": self.metadata,
            # Computed
            "blocks_promotion": self.blocks_promotion,
            "blocks_merge": self.blocks_merge,
            "critical_count": self.critical_count,
            "high_count": self.high_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LayerResult":
        """Deserialize from dictionary."""
        return cls(
            layer_id=LayerID(data["layer_id"]),
            status=LayerStatus(data["status"]),
            veto=VetoLevel(data["veto"]),
            findings=[Finding.from_dict(f) for f in data.get("findings", [])],
            security_findings=[SecurityFinding.from_dict(f) for f in data.get("security_findings", [])],
            execution_time_ms=data.get("execution_time_ms", 0),
            consensus_score=data.get("consensus_score", 1.0),
            runs=data.get("runs", 1),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else datetime.utcnow(),
            metadata=data.get("metadata", {}),
        )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = ["LayerResult"]
```

---

## 4. HLS-MDL-003: HumanLayerReport

```python
# src/hl_mcp/models/report.py
"""
Module: HLS-MDL-003 - HumanLayerReport
======================================

Complete validation report from Human Layer execution.

Aggregates results from all layers into a final decision.

Example:
    >>> report = HumanLayerReport.from_results(
    ...     artifact_id="code_123",
    ...     artifact_type="code",
    ...     results=[result1, result2, ...],
    ... )
    >>> report.can_promote
    True

Dependencies:
    - HLS-MDL-002: LayerResult
    - HLS-MDL-001: Finding
    - HLS-MDL-007: Enums

Author: Human Layer Team
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .enums import LayerStatus, VetoLevel, Severity, LayerPack
from .finding import Finding, SecurityFinding, findings_by_severity
from .layer_result import LayerResult


@dataclass
class HumanLayerReport:
    """Complete Human Layer validation report.

    Aggregates all layer results and provides final decisions
    on promotion and merge eligibility.

    Attributes:
        artifact_id: What was validated
        artifact_type: Type of artifact
        layer_pack: Which layer pack was used
        results: Results from each layer
        summary: Aggregated statistics
        can_promote: Can go to production?
        can_merge: Can merge to main?

    Example:
        >>> report = HumanLayerReport.from_results(...)
        >>> if report.can_promote:
        ...     deploy_to_production()
    """

    artifact_id: str
    artifact_type: str
    layer_pack: LayerPack

    results: List[LayerResult] = field(default_factory=list)

    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    execution_time_ms: int = 0

    metadata: Dict[str, Any] = field(default_factory=dict)

    # =========================================================================
    # FACTORY
    # =========================================================================

    @classmethod
    def from_results(
        cls,
        artifact_id: str,
        artifact_type: str,
        results: List[LayerResult],
        layer_pack: LayerPack = LayerPack.FULL,
        started_at: Optional[datetime] = None,
        **kwargs: Any,
    ) -> "HumanLayerReport":
        """Create report from layer results."""
        completed_at = datetime.utcnow()
        started = started_at or (
            min((r.started_at for r in results if r.started_at), default=completed_at)
        )
        total_time = sum(r.execution_time_ms for r in results)

        return cls(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            layer_pack=layer_pack,
            results=results,
            started_at=started,
            completed_at=completed_at,
            execution_time_ms=total_time,
            **kwargs,
        )

    # =========================================================================
    # COMPUTED PROPERTIES
    # =========================================================================

    @property
    def layers_run(self) -> int:
        """Number of layers executed."""
        return len(self.results)

    @property
    def layers_passed(self) -> int:
        """Number of layers that passed."""
        return sum(1 for r in self.results if r.status.is_success)

    @property
    def layers_failed(self) -> int:
        """Number of layers that failed."""
        return sum(1 for r in self.results if r.status.is_failure)

    @property
    def all_findings(self) -> List[Finding]:
        """All findings from all layers."""
        findings: List[Finding] = []
        for r in self.results:
            findings.extend(r.all_findings)
        return findings

    @property
    def all_security_findings(self) -> List[SecurityFinding]:
        """All security findings."""
        findings: List[SecurityFinding] = []
        for r in self.results:
            findings.extend(r.security_findings)
        return findings

    @property
    def veto_summary(self) -> Dict[str, int]:
        """Count of vetos by level."""
        summary = {v.value: 0 for v in VetoLevel}
        for r in self.results:
            summary[r.veto.value] += 1
        return summary

    @property
    def strong_vetos(self) -> int:
        """Count of STRONG vetos."""
        return self.veto_summary[VetoLevel.STRONG.value]

    @property
    def medium_vetos(self) -> int:
        """Count of MEDIUM vetos."""
        return self.veto_summary[VetoLevel.MEDIUM.value]

    @property
    def can_promote(self) -> bool:
        """Check if artifact can be promoted to production.

        Blocked if:
        - Any STRONG veto
        - Any CRITICAL findings
        """
        if self.strong_vetos > 0:
            return False

        critical = sum(1 for f in self.all_findings if f.severity == Severity.CRITICAL)
        return critical == 0

    @property
    def can_merge(self) -> bool:
        """Check if artifact can be merged to main.

        Blocked if:
        - Cannot promote (strong veto or critical)
        - More than 1 MEDIUM veto
        - Too many HIGH findings
        """
        if not self.can_promote:
            return False

        if self.medium_vetos > 1:
            return False

        high_count = sum(1 for f in self.all_findings if f.severity == Severity.HIGH)
        return high_count <= 3  # Configurable threshold

    @property
    def summary(self) -> Dict[str, Any]:
        """Summary statistics."""
        findings = self.all_findings
        by_severity = findings_by_severity(findings)

        return {
            "total_layers": self.layers_run,
            "layers_passed": self.layers_passed,
            "layers_failed": self.layers_failed,
            "strong_vetos": self.strong_vetos,
            "medium_vetos": self.medium_vetos,
            "total_findings": len(findings),
            "critical_findings": len(by_severity[Severity.CRITICAL]),
            "high_findings": len(by_severity[Severity.HIGH]),
            "medium_findings": len(by_severity[Severity.MEDIUM]),
            "low_findings": len(by_severity[Severity.LOW]),
            "security_findings": len(self.all_security_findings),
            "can_promote": self.can_promote,
            "can_merge": self.can_merge,
            "execution_time_ms": self.execution_time_ms,
        }

    @property
    def pass_rate(self) -> float:
        """Layer pass rate."""
        if self.layers_run == 0:
            return 0.0
        return self.layers_passed / self.layers_run

    # =========================================================================
    # METHODS
    # =========================================================================

    def get_result_for_layer(self, layer_id: str) -> Optional[LayerResult]:
        """Get result for specific layer."""
        for r in self.results:
            if r.layer_id.value == layer_id:
                return r
        return None

    def get_blocking_findings(self) -> List[Finding]:
        """Get findings that are blocking promotion/merge."""
        return [
            f for f in self.all_findings
            if f.severity in (Severity.CRITICAL, Severity.HIGH)
        ]

    def format_summary(self) -> str:
        """Format human-readable summary."""
        s = self.summary
        lines = [
            f"Human Layer Report: {self.artifact_id}",
            f"{'=' * 50}",
            f"",
            f"Layers: {s['layers_passed']}/{s['total_layers']} passed",
            f"Findings: {s['total_findings']} ({s['critical_findings']} critical, {s['high_findings']} high)",
            f"Security: {s['security_findings']} security findings",
            f"",
            f"Vetos: {s['strong_vetos']} STRONG, {s['medium_vetos']} MEDIUM",
            f"",
            f"Decision:",
            f"  Can Promote: {'✅ YES' if s['can_promote'] else '❌ NO'}",
            f"  Can Merge:   {'✅ YES' if s['can_merge'] else '❌ NO'}",
            f"",
            f"Time: {s['execution_time_ms']}ms",
        ]
        return "\n".join(lines)

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "layer_pack": self.layer_pack.value,
            "results": [r.to_dict() for r in self.results],
            "summary": self.summary,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HumanLayerReport":
        """Deserialize from dictionary."""
        return cls(
            artifact_id=data["artifact_id"],
            artifact_type=data["artifact_type"],
            layer_pack=LayerPack(data["layer_pack"]),
            results=[LayerResult.from_dict(r) for r in data.get("results", [])],
            started_at=datetime.fromisoformat(data["started_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            execution_time_ms=data.get("execution_time_ms", 0),
            metadata=data.get("metadata", {}),
        )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = ["HumanLayerReport"]
```

---

## 5. TESTES UNITÁRIOS

```python
# tests/unit/models/test_enums.py
"""Tests for HLS-MDL-007: Enums module."""

import pytest
from hl_mcp.models.enums import (
    VetoLevel,
    LayerStatus,
    Severity,
    HITLDecision,
    LayerID,
    PerspectiveID,
    LayerPack,
)


class TestVetoLevel:
    """Tests for VetoLevel enum."""

    def test_can_block_promotion(self):
        assert VetoLevel.STRONG.can_block_promotion is True
        assert VetoLevel.MEDIUM.can_block_promotion is False
        assert VetoLevel.WEAK.can_block_promotion is False
        assert VetoLevel.NONE.can_block_promotion is False

    def test_can_block_merge(self):
        assert VetoLevel.STRONG.can_block_merge is True
        assert VetoLevel.MEDIUM.can_block_merge is True
        assert VetoLevel.WEAK.can_block_merge is False
        assert VetoLevel.NONE.can_block_merge is False

    def test_from_string(self):
        assert VetoLevel.from_string("strong") == VetoLevel.STRONG
        assert VetoLevel.from_string("STRONG") == VetoLevel.STRONG
        assert VetoLevel.from_string("Strong") == VetoLevel.STRONG

    def test_from_string_invalid(self):
        with pytest.raises(ValueError):
            VetoLevel.from_string("invalid")


class TestSeverity:
    """Tests for Severity enum."""

    def test_comparison(self):
        assert Severity.CRITICAL < Severity.HIGH
        assert Severity.HIGH < Severity.MEDIUM
        assert Severity.MEDIUM < Severity.LOW
        assert Severity.LOW < Severity.INFO

    def test_priority_order(self):
        assert Severity.CRITICAL.priority_order == 0
        assert Severity.INFO.priority_order == 4

    def test_has_emoji(self):
        for severity in Severity:
            assert severity.emoji is not None


class TestLayerStatus:
    """Tests for LayerStatus enum."""

    def test_is_success(self):
        assert LayerStatus.PASS.is_success is True
        assert LayerStatus.WARN.is_success is True
        assert LayerStatus.SKIP.is_success is True
        assert LayerStatus.FAIL.is_success is False

    def test_is_failure(self):
        assert LayerStatus.FAIL.is_failure is True
        assert LayerStatus.ERROR.is_failure is True
        assert LayerStatus.PASS.is_failure is False

    def test_is_terminal(self):
        assert LayerStatus.PENDING.is_terminal is False
        assert LayerStatus.RUNNING.is_terminal is False
        assert LayerStatus.PASS.is_terminal is True


class TestLayerID:
    """Tests for LayerID enum."""

    def test_display_name(self):
        assert LayerID.HL_1_USUARIO.display_name == "Humano Usuario"
        assert LayerID.HL_6_HACKER.display_name == "Humano Hacker"

    def test_veto_level(self):
        assert LayerID.HL_1_USUARIO.veto_level == VetoLevel.WEAK
        assert LayerID.HL_4_DECISOR.veto_level == VetoLevel.STRONG

    def test_security_layers(self):
        security = LayerID.security_layers()
        assert LayerID.HL_5_SEGURANCA in security
        assert LayerID.HL_6_HACKER in security
        assert LayerID.HL_1_USUARIO not in security


class TestLayerPack:
    """Tests for LayerPack enum."""

    def test_full_has_all_layers(self):
        layers = LayerPack.FULL.get_layers()
        assert len(layers) == 7

    def test_security_pack(self):
        layers = LayerPack.SECURITY.get_layers()
        assert LayerID.HL_5_SEGURANCA in layers
        assert LayerID.HL_1_USUARIO not in layers

    def test_minimal_pack(self):
        layers = LayerPack.MINIMAL.get_layers()
        assert len(layers) == 1
        assert LayerID.HL_4_DECISOR in layers


# tests/unit/models/test_finding.py
"""Tests for HLS-MDL-001: Finding module."""

import pytest
from hl_mcp.models.finding import (
    Finding,
    SecurityFinding,
    findings_by_severity,
    count_blocking_findings,
    deduplicate_findings,
)
from hl_mcp.models.enums import Severity, LayerID


class TestFinding:
    """Tests for Finding dataclass."""

    def test_create_basic(self):
        f = Finding(
            layer=LayerID.HL_1_USUARIO,
            severity=Severity.MEDIUM,
            title="Test Finding",
            description="A test finding",
            fix_hint="Fix this",
        )
        assert f.title == "Test Finding"
        assert f.severity == Severity.MEDIUM

    def test_fix_hint_required(self):
        """INV-HLX-004: fix_hint must be provided."""
        with pytest.raises(ValueError, match="fix_hint"):
            Finding(
                layer=LayerID.HL_1_USUARIO,
                severity=Severity.LOW,
                title="Test",
                description="Test",
                fix_hint="",  # Empty = invalid
            )

    def test_is_blocking(self):
        critical = Finding.create(
            layer=LayerID.HL_5_SEGURANCA,
            severity=Severity.CRITICAL,
            title="Critical",
            description="Critical finding",
            fix_hint="Fix now",
        )
        assert critical.is_blocking is True

        low = Finding.create(
            layer=LayerID.HL_1_USUARIO,
            severity=Severity.LOW,
            title="Low",
            description="Low finding",
            fix_hint="Fix later",
        )
        assert low.is_blocking is False

    def test_serialization(self):
        f = Finding.create(
            layer=LayerID.HL_3_MANTENEDOR,
            severity=Severity.MEDIUM,
            title="Tech Debt",
            description="Code needs refactoring",
            fix_hint="Refactor the class",
        )

        data = f.to_dict()
        restored = Finding.from_dict(data)

        assert restored.title == f.title
        assert restored.severity == f.severity
        assert restored.layer == f.layer


class TestSecurityFinding:
    """Tests for SecurityFinding dataclass."""

    def test_create_with_owasp(self):
        sf = SecurityFinding.create(
            layer=LayerID.HL_6_HACKER,
            severity=Severity.CRITICAL,
            title="SQL Injection",
            description="Input not sanitized",
            fix_hint="Use parameterized queries",
            cwe_id="CWE-89",
            owasp_category="A03",
            attack_vector="Inject via search field",
        )

        assert sf.cwe_id == "CWE-89"
        assert sf.owasp_category == "A03"
        assert "security" in sf.tags

    def test_cvss_estimate(self):
        sf = SecurityFinding.create(
            layer=LayerID.HL_5_SEGURANCA,
            severity=Severity.CRITICAL,
            title="Test",
            description="Test",
            fix_hint="Fix",
            exploitability=1.0,  # Very easy to exploit
        )
        assert sf.cvss_estimate >= 9.0

    def test_exploitability_bounds(self):
        with pytest.raises(ValueError):
            SecurityFinding.create(
                layer=LayerID.HL_6_HACKER,
                severity=Severity.HIGH,
                title="Test",
                description="Test",
                fix_hint="Fix",
                exploitability=1.5,  # Invalid
            )


class TestFindingUtilities:
    """Tests for finding utility functions."""

    def test_findings_by_severity(self):
        findings = [
            Finding.create(LayerID.HL_1_USUARIO, Severity.CRITICAL, "C1", "d", "f"),
            Finding.create(LayerID.HL_1_USUARIO, Severity.HIGH, "H1", "d", "f"),
            Finding.create(LayerID.HL_1_USUARIO, Severity.HIGH, "H2", "d", "f"),
            Finding.create(LayerID.HL_1_USUARIO, Severity.LOW, "L1", "d", "f"),
        ]

        by_sev = findings_by_severity(findings)

        assert len(by_sev[Severity.CRITICAL]) == 1
        assert len(by_sev[Severity.HIGH]) == 2
        assert len(by_sev[Severity.LOW]) == 1

    def test_count_blocking_findings(self):
        findings = [
            Finding.create(LayerID.HL_1_USUARIO, Severity.CRITICAL, "C", "d", "f"),
            Finding.create(LayerID.HL_1_USUARIO, Severity.HIGH, "H", "d", "f"),
            Finding.create(LayerID.HL_1_USUARIO, Severity.LOW, "L", "d", "f"),
        ]

        assert count_blocking_findings(findings) == 2

    def test_deduplicate_findings(self):
        f1 = Finding.create(LayerID.HL_1_USUARIO, Severity.LOW, "Dup", "Same desc", "fix")
        f2 = Finding.create(LayerID.HL_1_USUARIO, Severity.HIGH, "Dup", "Same desc", "fix")

        deduped = deduplicate_findings([f1, f2])

        assert len(deduped) == 1
        assert deduped[0].severity == Severity.HIGH  # Kept more severe
```

---

## RESUMO DO BLOCK 04

| Módulo | Arquivo | Status | Linhas |
|--------|---------|--------|--------|
| HLS-MDL-007 | `models/enums.py` | ✅ Implementado | ~350 |
| HLS-MDL-001 | `models/finding.py` | ✅ Implementado | ~250 |
| HLS-MDL-002 | `models/layer_result.py` | ✅ Implementado | ~220 |
| HLS-MDL-003 | `models/report.py` | ✅ Implementado | ~280 |
| Tests | `tests/unit/models/` | ✅ Implementado | ~200 |

**Total**: ~1,300 linhas de código implementado

---

## PRÓXIMO BLOCK

O Block 05 vai cobrir:
1. HLS-MDL-004: Journey (models para browser testing)
2. HLS-MDL-005: Test (models para generated tests)
3. HLS-MDL-006: Review (models para review sessions)
4. Models `__init__.py` com exports consolidados

---

*ROADMAP_BLOCK_04_DATA_MODELS.md - v1.0.0 - 2026-02-01*
