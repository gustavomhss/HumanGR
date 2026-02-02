# Human Layer MCP Server - Roadmap Block 05
# MODELS COMPLETE + INIT EXPORTS

> **Objetivo**: Finalizar todos os 7 módulos de Models
> **Versão**: 1.0.0 | Data: 2026-02-01

---

## 1. HLS-MDL-004: Journey

```python
# src/hl_mcp/models/journey.py
"""
Module: HLS-MDL-004 - Journey
=============================

Models for browser journey testing.

Journey represents a sequence of browser actions to test,
JourneyResult captures the execution outcome.

Example:
    >>> journey = Journey(
    ...     name="Login Flow",
    ...     base_url="https://app.example.com",
    ...     steps=[
    ...         JourneyStep(action=BrowserAction.GOTO, target="/login"),
    ...         JourneyStep(action=BrowserAction.FILL, target="#email", value="test@test.com"),
    ...         JourneyStep(action=BrowserAction.CLICK, target="#submit"),
    ...     ],
    ... )

Dependencies:
    - HLS-MDL-007: Enums (BrowserAction)

Author: Human Layer Team
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid

from .enums import BrowserAction


class JourneyType(Enum):
    """Types of journeys."""
    HAPPY_PATH = "happy_path"
    ERROR_PATH = "error_path"
    EDGE_CASE = "edge_case"
    PERSONA = "persona"
    ALTERNATIVE = "alternative"
    REGRESSION = "regression"


class StepStatus(Enum):
    """Status of a journey step execution."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


@dataclass
class JourneyStep:
    """A single step in a user journey.

    Attributes:
        action: Browser action to perform
        target: URL, selector, or element identifier
        value: Value for fill/select actions
        description: Human-readable description
        timeout_ms: Max wait time for this step
        optional: If True, failure doesn't stop journey

    Example:
        >>> step = JourneyStep(
        ...     action=BrowserAction.FILL,
        ...     target="#username",
        ...     value="test@example.com",
        ...     description="Enter username",
        ... )
    """

    action: BrowserAction
    target: str
    value: str = ""
    description: str = ""
    timeout_ms: int = 30000
    optional: bool = False

    # Execution state (filled after execution)
    status: StepStatus = StepStatus.PENDING
    execution_time_ms: int = 0
    error_message: Optional[str] = None
    screenshot_path: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate step configuration."""
        if not isinstance(self.action, BrowserAction):
            self.action = BrowserAction(self.action)

        # Validate that required fields are present
        if self.action.requires_selector and not self.target:
            raise ValueError(f"Action {self.action.value} requires a target selector")

        if self.action.requires_value and not self.value:
            raise ValueError(f"Action {self.action.value} requires a value")

        # Auto-generate description if not provided
        if not self.description:
            self.description = self._auto_description()

    def _auto_description(self) -> str:
        """Generate description from action and target."""
        descriptions = {
            BrowserAction.GOTO: f"Navigate to {self.target}",
            BrowserAction.CLICK: f"Click {self.target}",
            BrowserAction.FILL: f"Fill {self.target} with value",
            BrowserAction.WAIT: f"Wait for {self.target}",
            BrowserAction.SCREENSHOT: "Take screenshot",
            BrowserAction.SCROLL: f"Scroll to {self.target}",
            BrowserAction.HOVER: f"Hover over {self.target}",
            BrowserAction.SELECT: f"Select {self.value} in {self.target}",
            BrowserAction.CHECK: f"Check {self.target}",
            BrowserAction.UNCHECK: f"Uncheck {self.target}",
        }
        return descriptions.get(self.action, f"{self.action.value} {self.target}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "action": self.action.value,
            "target": self.target,
            "value": self.value,
            "description": self.description,
            "timeout_ms": self.timeout_ms,
            "optional": self.optional,
            "status": self.status.value,
            "execution_time_ms": self.execution_time_ms,
            "error_message": self.error_message,
            "screenshot_path": self.screenshot_path,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JourneyStep":
        """Deserialize from dictionary."""
        return cls(
            action=BrowserAction(data["action"]),
            target=data["target"],
            value=data.get("value", ""),
            description=data.get("description", ""),
            timeout_ms=data.get("timeout_ms", 30000),
            optional=data.get("optional", False),
            status=StepStatus(data.get("status", "pending")),
            execution_time_ms=data.get("execution_time_ms", 0),
            error_message=data.get("error_message"),
            screenshot_path=data.get("screenshot_path"),
        )

    # Convenience factory methods
    @classmethod
    def goto(cls, url: str, **kwargs: Any) -> "JourneyStep":
        """Create a navigation step."""
        return cls(action=BrowserAction.GOTO, target=url, **kwargs)

    @classmethod
    def click(cls, selector: str, **kwargs: Any) -> "JourneyStep":
        """Create a click step."""
        return cls(action=BrowserAction.CLICK, target=selector, **kwargs)

    @classmethod
    def fill(cls, selector: str, value: str, **kwargs: Any) -> "JourneyStep":
        """Create a fill step."""
        return cls(action=BrowserAction.FILL, target=selector, value=value, **kwargs)

    @classmethod
    def wait(cls, selector: str, timeout_ms: int = 30000, **kwargs: Any) -> "JourneyStep":
        """Create a wait step."""
        return cls(action=BrowserAction.WAIT, target=selector, timeout_ms=timeout_ms, **kwargs)

    @classmethod
    def screenshot(cls, name: str = "screenshot", **kwargs: Any) -> "JourneyStep":
        """Create a screenshot step."""
        return cls(action=BrowserAction.SCREENSHOT, target=name, **kwargs)


@dataclass
class AccessibilityIssue:
    """An accessibility issue found during testing.

    Attributes:
        rule_id: WCAG rule identifier
        severity: Issue severity
        message: Description of the issue
        selector: Element with the issue
        help_url: Link to more information
    """

    rule_id: str
    severity: str  # critical, serious, moderate, minor
    message: str
    selector: str
    help_url: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity,
            "message": self.message,
            "selector": self.selector,
            "help_url": self.help_url,
        }


@dataclass
class AccessibilityResult:
    """Result of accessibility validation.

    Attributes:
        passed: Overall pass/fail
        issues: List of issues found
        violations_count: Number of violations by severity
        tested_at: When test was run
    """

    passed: bool
    issues: List[AccessibilityIssue] = field(default_factory=list)
    tested_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def violations_count(self) -> Dict[str, int]:
        """Count violations by severity."""
        counts = {"critical": 0, "serious": 0, "moderate": 0, "minor": 0}
        for issue in self.issues:
            if issue.severity in counts:
                counts[issue.severity] += 1
        return counts

    @property
    def has_critical(self) -> bool:
        """Check if there are critical issues."""
        return self.violations_count["critical"] > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "issues": [i.to_dict() for i in self.issues],
            "violations_count": self.violations_count,
            "tested_at": self.tested_at.isoformat(),
        }


@dataclass
class Journey:
    """A complete user journey for testing.

    Attributes:
        id: Unique identifier
        name: Journey name
        description: What this journey tests
        journey_type: Type of journey
        base_url: Base URL for the journey
        steps: List of steps to execute
        tags: Categorization tags
        persona: If persona journey, which persona

    Example:
        >>> journey = Journey(
        ...     name="Complete Checkout",
        ...     base_url="https://shop.example.com",
        ...     steps=[
        ...         JourneyStep.goto("/cart"),
        ...         JourneyStep.click("#checkout-btn"),
        ...         JourneyStep.fill("#card-number", "4111111111111111"),
        ...         JourneyStep.click("#submit"),
        ...         JourneyStep.wait(".confirmation"),
        ...     ],
        ... )
    """

    name: str
    base_url: str
    steps: List[JourneyStep] = field(default_factory=list)

    id: str = field(default_factory=lambda: f"jrn_{uuid.uuid4().hex[:12]}")
    description: str = ""
    journey_type: JourneyType = JourneyType.HAPPY_PATH
    tags: List[str] = field(default_factory=list)
    persona: Optional[str] = None  # e.g., "tired_user", "power_user"

    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate journey."""
        if not self.steps:
            raise ValueError("Journey must have at least one step")

        if not isinstance(self.journey_type, JourneyType):
            self.journey_type = JourneyType(self.journey_type)

    @property
    def step_count(self) -> int:
        """Number of steps in journey."""
        return len(self.steps)

    @property
    def estimated_duration_ms(self) -> int:
        """Estimated duration based on step timeouts."""
        # Rough estimate: 2 seconds per step + timeouts for waits
        base = 2000 * self.step_count
        waits = sum(s.timeout_ms for s in self.steps if s.action == BrowserAction.WAIT)
        return base + waits

    def add_step(self, step: JourneyStep) -> "Journey":
        """Add a step to the journey (builder pattern)."""
        self.steps.append(step)
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "journey_type": self.journey_type.value,
            "base_url": self.base_url,
            "steps": [s.to_dict() for s in self.steps],
            "tags": self.tags,
            "persona": self.persona,
            "step_count": self.step_count,
            "estimated_duration_ms": self.estimated_duration_ms,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Journey":
        """Deserialize from dictionary."""
        return cls(
            id=data.get("id", f"jrn_{uuid.uuid4().hex[:12]}"),
            name=data["name"],
            description=data.get("description", ""),
            journey_type=JourneyType(data.get("journey_type", "happy_path")),
            base_url=data["base_url"],
            steps=[JourneyStep.from_dict(s) for s in data["steps"]],
            tags=data.get("tags", []),
            persona=data.get("persona"),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            metadata=data.get("metadata", {}),
        )


@dataclass
class JourneyResult:
    """Result of executing a journey.

    Attributes:
        journey: The journey that was executed
        success: Overall success/failure
        steps_completed: How many steps completed
        screenshots: Paths to screenshots taken
        video_path: Path to video recording
        accessibility: Accessibility test result
        duration_ms: Total execution time
        errors: List of errors encountered

    Example:
        >>> result = await executor.run(journey)
        >>> if result.success:
        ...     print(f"Journey passed in {result.duration_ms}ms")
        >>> else:
        ...     print(f"Failed at step {result.steps_completed}")
    """

    journey: Journey
    success: bool

    steps_completed: int = 0
    screenshots: List[str] = field(default_factory=list)
    video_path: Optional[str] = None
    accessibility: Optional[AccessibilityResult] = None

    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    duration_ms: int = 0

    errors: List[str] = field(default_factory=list)
    failed_step_index: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def steps_total(self) -> int:
        """Total steps in journey."""
        return self.journey.step_count

    @property
    def completion_rate(self) -> float:
        """Percentage of steps completed."""
        if self.steps_total == 0:
            return 0.0
        return self.steps_completed / self.steps_total

    @property
    def failed_step(self) -> Optional[JourneyStep]:
        """Get the step that failed, if any."""
        if self.failed_step_index is not None and self.failed_step_index < len(self.journey.steps):
            return self.journey.steps[self.failed_step_index]
        return None

    @property
    def has_accessibility_issues(self) -> bool:
        """Check if there are accessibility issues."""
        return self.accessibility is not None and not self.accessibility.passed

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "journey_id": self.journey.id,
            "journey_name": self.journey.name,
            "success": self.success,
            "steps_completed": self.steps_completed,
            "steps_total": self.steps_total,
            "completion_rate": self.completion_rate,
            "screenshots": self.screenshots,
            "video_path": self.video_path,
            "accessibility": self.accessibility.to_dict() if self.accessibility else None,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "errors": self.errors,
            "failed_step_index": self.failed_step_index,
            "metadata": self.metadata,
        }

    # Factory methods
    @classmethod
    def success_result(
        cls,
        journey: Journey,
        duration_ms: int,
        screenshots: Optional[List[str]] = None,
        video_path: Optional[str] = None,
        **kwargs: Any,
    ) -> "JourneyResult":
        """Create a successful result."""
        return cls(
            journey=journey,
            success=True,
            steps_completed=journey.step_count,
            duration_ms=duration_ms,
            screenshots=screenshots or [],
            video_path=video_path,
            completed_at=datetime.utcnow(),
            **kwargs,
        )

    @classmethod
    def failure_result(
        cls,
        journey: Journey,
        failed_step_index: int,
        error: str,
        duration_ms: int,
        **kwargs: Any,
    ) -> "JourneyResult":
        """Create a failure result."""
        return cls(
            journey=journey,
            success=False,
            steps_completed=failed_step_index,
            failed_step_index=failed_step_index,
            errors=[error],
            duration_ms=duration_ms,
            completed_at=datetime.utcnow(),
            **kwargs,
        )


# ============================================================================
# JOURNEY BUILDERS
# ============================================================================

class JourneyBuilder:
    """Fluent builder for creating journeys.

    Example:
        >>> journey = (
        ...     JourneyBuilder("Login Test", "https://app.example.com")
        ...     .goto("/login")
        ...     .fill("#email", "test@test.com")
        ...     .fill("#password", "secret123")
        ...     .click("#submit")
        ...     .wait(".dashboard")
        ...     .screenshot("after_login")
        ...     .build()
        ... )
    """

    def __init__(self, name: str, base_url: str) -> None:
        self._name = name
        self._base_url = base_url
        self._steps: List[JourneyStep] = []
        self._description = ""
        self._journey_type = JourneyType.HAPPY_PATH
        self._tags: List[str] = []
        self._persona: Optional[str] = None

    def description(self, desc: str) -> "JourneyBuilder":
        self._description = desc
        return self

    def journey_type(self, jtype: JourneyType) -> "JourneyBuilder":
        self._journey_type = jtype
        return self

    def tag(self, *tags: str) -> "JourneyBuilder":
        self._tags.extend(tags)
        return self

    def persona(self, persona: str) -> "JourneyBuilder":
        self._persona = persona
        return self

    def goto(self, url: str, **kwargs: Any) -> "JourneyBuilder":
        self._steps.append(JourneyStep.goto(url, **kwargs))
        return self

    def click(self, selector: str, **kwargs: Any) -> "JourneyBuilder":
        self._steps.append(JourneyStep.click(selector, **kwargs))
        return self

    def fill(self, selector: str, value: str, **kwargs: Any) -> "JourneyBuilder":
        self._steps.append(JourneyStep.fill(selector, value, **kwargs))
        return self

    def wait(self, selector: str, timeout_ms: int = 30000, **kwargs: Any) -> "JourneyBuilder":
        self._steps.append(JourneyStep.wait(selector, timeout_ms, **kwargs))
        return self

    def screenshot(self, name: str = "screenshot", **kwargs: Any) -> "JourneyBuilder":
        self._steps.append(JourneyStep.screenshot(name, **kwargs))
        return self

    def step(self, step: JourneyStep) -> "JourneyBuilder":
        self._steps.append(step)
        return self

    def build(self) -> Journey:
        return Journey(
            name=self._name,
            base_url=self._base_url,
            steps=self._steps,
            description=self._description,
            journey_type=self._journey_type,
            tags=self._tags,
            persona=self._persona,
        )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "JourneyType",
    "StepStatus",
    # Core models
    "JourneyStep",
    "Journey",
    "JourneyResult",
    # Accessibility
    "AccessibilityIssue",
    "AccessibilityResult",
    # Builder
    "JourneyBuilder",
]
```

---

## 2. HLS-MDL-005: Test

```python
# src/hl_mcp/models/test.py
"""
Module: HLS-MDL-005 - Test
==========================

Models for generated tests from perspective analysis.

GeneratedTest represents a test case generated from a human perspective.
TestConsensus captures agreement across multiple perspectives.

Example:
    >>> test = GeneratedTest(
    ...     perspective_id=PerspectiveID.TIRED_USER,
    ...     name="test_multiple_clicks_handled",
    ...     description="System handles rapid multiple clicks",
    ...     assertions=["Operation is idempotent"],
    ... )

Dependencies:
    - HLS-MDL-007: Enums (PerspectiveID)

Author: Human Layer Team
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import hashlib

from .enums import PerspectiveID


class TestPriority(Enum):
    """Priority levels for generated tests."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @property
    def order(self) -> int:
        """Numeric order for sorting."""
        return {"critical": 0, "high": 1, "medium": 2, "low": 3}[self.value]


class TestStatus(Enum):
    """Status of a generated test."""
    GENERATED = "generated"      # Just generated, not yet implemented
    IMPLEMENTED = "implemented"  # Test code written
    PASSING = "passing"          # Test runs and passes
    FAILING = "failing"          # Test runs but fails
    SKIPPED = "skipped"          # Test skipped
    FLAKY = "flaky"              # Intermittent failures


@dataclass
class GeneratedTest:
    """A test generated from perspective analysis.

    Represents a test case that emerged from thinking
    about the system from a specific human perspective.

    Attributes:
        test_id: Unique identifier
        perspective_id: Which perspective generated this
        name: Test function name (snake_case)
        description: What the test checks
        test_code: Generated pytest code
        assertions: List of assertions to verify
        priority: Test priority
        tags: Categorization tags
        edge_case_type: If edge case, what type

    Example:
        >>> test = GeneratedTest.create(
        ...     perspective=PerspectiveID.MALICIOUS_INSIDER,
        ...     name="test_cannot_access_other_users_data",
        ...     description="Verify users cannot access data belonging to others",
        ...     assertions=["Access denied when accessing other user's data"],
        ...     priority=TestPriority.CRITICAL,
        ... )
    """

    perspective_id: PerspectiveID
    name: str
    description: str
    assertions: List[str] = field(default_factory=list)

    test_id: str = ""
    test_code: str = ""
    priority: TestPriority = TestPriority.MEDIUM
    status: TestStatus = TestStatus.GENERATED
    tags: List[str] = field(default_factory=list)
    edge_case_type: Optional[str] = None

    component: str = ""  # What component this tests
    source_pattern: str = ""  # The perspective pattern that generated this

    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and auto-generate fields."""
        if not isinstance(self.perspective_id, PerspectiveID):
            self.perspective_id = PerspectiveID(self.perspective_id)

        if not isinstance(self.priority, TestPriority):
            self.priority = TestPriority(self.priority)

        # Auto-generate test_id if not provided
        if not self.test_id:
            self.test_id = self._generate_id()

        # Ensure name is valid Python identifier
        self.name = self._sanitize_name(self.name)

        # Add perspective tag
        if self.perspective_id.value not in self.tags:
            self.tags.append(self.perspective_id.value)

    def _generate_id(self) -> str:
        """Generate unique test ID."""
        content = f"{self.perspective_id.value}:{self.name}:{self.description}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def _sanitize_name(self, name: str) -> str:
        """Ensure name is valid Python test function name."""
        # Ensure starts with test_
        if not name.startswith("test_"):
            name = f"test_{name}"

        # Replace invalid characters
        name = name.lower()
        name = name.replace(" ", "_")
        name = name.replace("-", "_")
        name = "".join(c if c.isalnum() or c == "_" else "" for c in name)

        # Ensure not too long
        return name[:80]

    @classmethod
    def create(
        cls,
        perspective: PerspectiveID,
        name: str,
        description: str,
        assertions: Optional[List[str]] = None,
        priority: TestPriority = TestPriority.MEDIUM,
        **kwargs: Any,
    ) -> "GeneratedTest":
        """Factory method to create a test."""
        return cls(
            perspective_id=perspective,
            name=name,
            description=description,
            assertions=assertions or [],
            priority=priority,
            **kwargs,
        )

    def generate_code(self) -> str:
        """Generate pytest code for this test.

        Returns skeleton code that can be filled in.
        """
        assertions_doc = "\n".join(f"    - {a}" for a in self.assertions)
        assertions_code = "\n    ".join(
            f"# {a}" for a in self.assertions
        )

        code = f'''def {self.name}():
    """Test generated from {self.perspective_id.display_name} perspective.

    {self.description}

    Assertions:
{assertions_doc}

    Generated by Human Layer MCP - do not edit header.
    """
    # Arrange - Set up test data and mocks
    test_data = {{}}  # TODO: Add test data

    # Act - Execute the action being tested
    result = None  # TODO: Call function under test

    # Assert - Verify expectations
    {assertions_code}
    assert result is not None, "Expected a result"
'''
        self.test_code = code
        return code

    @property
    def is_security_test(self) -> bool:
        """Check if this is a security-related test."""
        security_perspectives = {PerspectiveID.MALICIOUS_INSIDER, PerspectiveID.AUDITOR}
        return self.perspective_id in security_perspectives or "security" in self.tags

    @property
    def content_hash(self) -> str:
        """Hash of test content for deduplication."""
        content = f"{self.name}:{self.description}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "test_id": self.test_id,
            "perspective_id": self.perspective_id.value,
            "perspective_name": self.perspective_id.display_name,
            "name": self.name,
            "description": self.description,
            "test_code": self.test_code,
            "assertions": self.assertions,
            "priority": self.priority.value,
            "status": self.status.value,
            "tags": self.tags,
            "edge_case_type": self.edge_case_type,
            "component": self.component,
            "source_pattern": self.source_pattern,
            "is_security_test": self.is_security_test,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeneratedTest":
        """Deserialize from dictionary."""
        return cls(
            test_id=data.get("test_id", ""),
            perspective_id=PerspectiveID(data["perspective_id"]),
            name=data["name"],
            description=data["description"],
            test_code=data.get("test_code", ""),
            assertions=data.get("assertions", []),
            priority=TestPriority(data.get("priority", "medium")),
            status=TestStatus(data.get("status", "generated")),
            tags=data.get("tags", []),
            edge_case_type=data.get("edge_case_type"),
            component=data.get("component", ""),
            source_pattern=data.get("source_pattern", ""),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            metadata=data.get("metadata", {}),
        )


@dataclass
class TestConsensus:
    """Consensus result from multi-perspective test generation.

    When multiple perspectives agree a test is important,
    it gets higher confidence.

    Attributes:
        test: The generated test
        agreement_score: Weighted agreement (0.0-1.0)
        contributing_perspectives: Which perspectives agreed
        confidence: Confidence in test value

    Example:
        >>> # Both MALICIOUS_INSIDER and AUDITOR think this test is important
        >>> consensus.agreement_score
        0.85
        >>> consensus.contributing_perspectives
        ['malicious_insider', 'auditor']
    """

    test: GeneratedTest
    agreement_score: float
    contributing_perspectives: List[str]
    confidence: float

    def __post_init__(self) -> None:
        """Validate scores."""
        if not 0.0 <= self.agreement_score <= 1.0:
            raise ValueError(f"agreement_score must be 0.0-1.0, got {self.agreement_score}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be 0.0-1.0, got {self.confidence}")

    @property
    def is_high_agreement(self) -> bool:
        """Check if there's high agreement (>= 0.7)."""
        return self.agreement_score >= 0.7

    @property
    def perspective_count(self) -> int:
        """Number of contributing perspectives."""
        return len(self.contributing_perspectives)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "test": self.test.to_dict(),
            "agreement_score": round(self.agreement_score, 3),
            "contributing_perspectives": self.contributing_perspectives,
            "confidence": round(self.confidence, 3),
            "is_high_agreement": self.is_high_agreement,
            "perspective_count": self.perspective_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestConsensus":
        """Deserialize from dictionary."""
        return cls(
            test=GeneratedTest.from_dict(data["test"]),
            agreement_score=data["agreement_score"],
            contributing_perspectives=data["contributing_perspectives"],
            confidence=data["confidence"],
        )


@dataclass
class TestSuite:
    """A collection of generated tests.

    Attributes:
        name: Suite name
        description: What this suite tests
        tests: List of generated tests
        consensus_tests: Tests with multi-perspective consensus
    """

    name: str
    tests: List[GeneratedTest] = field(default_factory=list)
    consensus_tests: List[TestConsensus] = field(default_factory=list)

    id: str = field(default_factory=lambda: f"suite_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
    description: str = ""
    component: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def test_count(self) -> int:
        return len(self.tests)

    @property
    def high_priority_count(self) -> int:
        return sum(1 for t in self.tests if t.priority in (TestPriority.CRITICAL, TestPriority.HIGH))

    @property
    def security_test_count(self) -> int:
        return sum(1 for t in self.tests if t.is_security_test)

    @property
    def by_perspective(self) -> Dict[str, List[GeneratedTest]]:
        """Group tests by perspective."""
        result: Dict[str, List[GeneratedTest]] = {}
        for test in self.tests:
            key = test.perspective_id.value
            if key not in result:
                result[key] = []
            result[key].append(test)
        return result

    @property
    def by_priority(self) -> Dict[str, List[GeneratedTest]]:
        """Group tests by priority."""
        result: Dict[str, List[GeneratedTest]] = {}
        for test in self.tests:
            key = test.priority.value
            if key not in result:
                result[key] = []
            result[key].append(test)
        return result

    def add_test(self, test: GeneratedTest) -> None:
        """Add a test to the suite."""
        self.tests.append(test)

    def add_consensus(self, consensus: TestConsensus) -> None:
        """Add a consensus test."""
        self.consensus_tests.append(consensus)

    def get_prioritized_tests(self) -> List[GeneratedTest]:
        """Get tests sorted by priority."""
        return sorted(self.tests, key=lambda t: t.priority.order)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "component": self.component,
            "test_count": self.test_count,
            "high_priority_count": self.high_priority_count,
            "security_test_count": self.security_test_count,
            "tests": [t.to_dict() for t in self.tests],
            "consensus_tests": [c.to_dict() for c in self.consensus_tests],
            "created_at": self.created_at.isoformat(),
        }


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "TestPriority",
    "TestStatus",
    "GeneratedTest",
    "TestConsensus",
    "TestSuite",
]
```

---

## 3. HLS-MDL-006: Review

```python
# src/hl_mcp/models/review.py
"""
Module: HLS-MDL-006 - Review
============================

Models for review sessions and human-in-the-loop decisions.

ReviewItem represents something that needs human review.
ReviewSession captures the review process and outcome.

Example:
    >>> item = ReviewItem(
    ...     item_id="finding_123",
    ...     item_type="finding",
    ...     content=finding.to_dict(),
    ...     urgency="high",
    ... )
    >>> session = await orchestrator.schedule_review(item)

Dependencies:
    - HLS-MDL-007: Enums (HITLDecision, ReviewTrigger)

Author: Human Layer Team
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid

from .enums import HITLDecision, ReviewTrigger, FeedbackType


class ReviewUrgency(Enum):
    """Urgency levels for review items."""
    IMMEDIATE = "immediate"  # Drop everything
    HIGH = "high"            # Today
    MEDIUM = "medium"        # This week
    LOW = "low"              # When convenient
    DEFERRED = "deferred"    # Backlog


class ReviewOutcome(Enum):
    """Outcome of a review session."""
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"
    ESCALATED = "escalated"
    DEFERRED = "deferred"
    EXPIRED = "expired"


class ReviewCategory(Enum):
    """Categories of review items."""
    SECURITY = "security"
    DECISION = "decision"
    CODE_QUALITY = "code_quality"
    DOCUMENTATION = "documentation"
    GENERAL = "general"


@dataclass
class ReviewItem:
    """An item that needs human review.

    Attributes:
        item_id: Unique identifier
        item_type: Type of item (finding, test, journey, etc.)
        content: The actual content to review
        urgency: How urgent is this review
        category: Review category
        triggers: What triggered this review
        context: Additional context for reviewer
        attention_cost: Estimated attention units needed

    Example:
        >>> item = ReviewItem.from_finding(finding)
        >>> item.urgency
        ReviewUrgency.HIGH
    """

    item_type: str
    content: Dict[str, Any]
    urgency: ReviewUrgency = ReviewUrgency.MEDIUM
    category: ReviewCategory = ReviewCategory.GENERAL

    item_id: str = field(default_factory=lambda: f"rev_{uuid.uuid4().hex[:12]}")
    triggers: List[ReviewTrigger] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    attention_cost: int = 5  # Default attention units

    source_layer: Optional[str] = None  # Which layer generated this
    source_perspective: Optional[str] = None  # Which perspective

    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate item."""
        if not isinstance(self.urgency, ReviewUrgency):
            self.urgency = ReviewUrgency(self.urgency)
        if not isinstance(self.category, ReviewCategory):
            self.category = ReviewCategory(self.category)

    @property
    def is_expired(self) -> bool:
        """Check if review has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    @property
    def is_security_related(self) -> bool:
        """Check if security-related."""
        return (
            self.category == ReviewCategory.SECURITY or
            ReviewTrigger.SECURITY_CONCERN in self.triggers or
            ReviewTrigger.HIGH_RISK in self.triggers
        )

    @property
    def summary(self) -> str:
        """One-line summary."""
        return f"[{self.urgency.value.upper()}] {self.item_type}: {self.item_id}"

    # Factory methods
    @classmethod
    def from_finding(cls, finding: Any, **kwargs: Any) -> "ReviewItem":
        """Create from a Finding object."""
        from .finding import Finding

        urgency = ReviewUrgency.HIGH if finding.is_blocking else ReviewUrgency.MEDIUM
        category = (
            ReviewCategory.SECURITY
            if finding.layer.value in ("HL-5", "HL-6")
            else ReviewCategory.GENERAL
        )

        return cls(
            item_type="finding",
            content=finding.to_dict() if hasattr(finding, "to_dict") else dict(finding),
            urgency=urgency,
            category=category,
            source_layer=finding.layer.value if hasattr(finding, "layer") else None,
            attention_cost=10 if finding.is_blocking else 5,
            **kwargs,
        )

    @classmethod
    def from_triage(
        cls,
        decision: HITLDecision,
        content: Dict[str, Any],
        triggers: List[ReviewTrigger],
        **kwargs: Any,
    ) -> "ReviewItem":
        """Create from triage decision."""
        urgency = (
            ReviewUrgency.IMMEDIATE if decision == HITLDecision.SYNC_REQUIRED
            else ReviewUrgency.MEDIUM
        )

        return cls(
            item_type="triage",
            content=content,
            urgency=urgency,
            triggers=triggers,
            **kwargs,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "item_id": self.item_id,
            "item_type": self.item_type,
            "content": self.content,
            "urgency": self.urgency.value,
            "category": self.category.value,
            "triggers": [t.value for t in self.triggers],
            "context": self.context,
            "attention_cost": self.attention_cost,
            "source_layer": self.source_layer,
            "source_perspective": self.source_perspective,
            "is_expired": self.is_expired,
            "is_security_related": self.is_security_related,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReviewItem":
        """Deserialize from dictionary."""
        return cls(
            item_id=data.get("item_id", f"rev_{uuid.uuid4().hex[:12]}"),
            item_type=data["item_type"],
            content=data["content"],
            urgency=ReviewUrgency(data.get("urgency", "medium")),
            category=ReviewCategory(data.get("category", "general")),
            triggers=[ReviewTrigger(t) for t in data.get("triggers", [])],
            context=data.get("context", {}),
            attention_cost=data.get("attention_cost", 5),
            source_layer=data.get("source_layer"),
            source_perspective=data.get("source_perspective"),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow(),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            metadata=data.get("metadata", {}),
        )


@dataclass
class ReviewSession:
    """A human review session.

    Captures the process and outcome of reviewing an item.

    Attributes:
        item: The item being reviewed
        outcome: Review outcome
        human_decision: What the human decided
        reasoning: Why they decided this
        feedback_type: Type of feedback (agree, override, etc.)
        modifications: Any modifications made
        duration_ms: How long the review took

    Example:
        >>> session = ReviewSession.create(item)
        >>> session.start()
        >>> # ... human reviews ...
        >>> session.complete(
        ...     outcome=ReviewOutcome.APPROVED,
        ...     reasoning="Looks good, edge cases covered",
        ... )
    """

    item: ReviewItem
    outcome: Optional[ReviewOutcome] = None

    session_id: str = field(default_factory=lambda: f"sess_{uuid.uuid4().hex[:12]}")
    human_decision: str = ""
    reasoning: str = ""
    feedback_type: Optional[FeedbackType] = None

    modifications: Dict[str, Any] = field(default_factory=dict)
    reviewer_id: Optional[str] = None

    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: int = 0

    attention_spent: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        """Check if review is complete."""
        return self.outcome is not None

    @property
    def is_in_progress(self) -> bool:
        """Check if review is in progress."""
        return self.started_at is not None and not self.is_complete

    @property
    def was_approved(self) -> bool:
        """Check if item was approved."""
        return self.outcome in (ReviewOutcome.APPROVED, ReviewOutcome.MODIFIED)

    # Lifecycle methods
    @classmethod
    def create(cls, item: ReviewItem, **kwargs: Any) -> "ReviewSession":
        """Create a new review session."""
        return cls(item=item, **kwargs)

    def start(self, reviewer_id: Optional[str] = None) -> "ReviewSession":
        """Start the review session."""
        self.started_at = datetime.utcnow()
        self.reviewer_id = reviewer_id
        return self

    def complete(
        self,
        outcome: ReviewOutcome,
        reasoning: str = "",
        human_decision: str = "",
        feedback_type: Optional[FeedbackType] = None,
        modifications: Optional[Dict[str, Any]] = None,
    ) -> "ReviewSession":
        """Complete the review session."""
        self.completed_at = datetime.utcnow()
        self.outcome = outcome
        self.reasoning = reasoning
        self.human_decision = human_decision
        self.feedback_type = feedback_type
        self.modifications = modifications or {}

        # Calculate duration
        if self.started_at:
            delta = self.completed_at - self.started_at
            self.duration_ms = int(delta.total_seconds() * 1000)

        # Set attention spent
        self.attention_spent = self.item.attention_cost

        return self

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "session_id": self.session_id,
            "item": self.item.to_dict(),
            "outcome": self.outcome.value if self.outcome else None,
            "human_decision": self.human_decision,
            "reasoning": self.reasoning,
            "feedback_type": self.feedback_type.value if self.feedback_type else None,
            "modifications": self.modifications,
            "reviewer_id": self.reviewer_id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "attention_spent": self.attention_spent,
            "is_complete": self.is_complete,
            "was_approved": self.was_approved,
            "metadata": self.metadata,
        }


@dataclass
class ScheduledReview:
    """A review scheduled for a specific time slot.

    Used by CognitiveOrchestrator to batch and schedule reviews.

    Attributes:
        item: Item to review
        scheduled_for: When to review
        slot_duration_ms: How long the slot is
        batch_id: If part of a batch
    """

    item: ReviewItem
    scheduled_for: datetime
    slot_duration_ms: int = 300000  # 5 minutes default

    schedule_id: str = field(default_factory=lambda: f"sched_{uuid.uuid4().hex[:12]}")
    batch_id: Optional[str] = None
    priority_order: int = 0

    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_due(self) -> bool:
        """Check if review is due."""
        return datetime.utcnow() >= self.scheduled_for

    @property
    def is_overdue(self) -> bool:
        """Check if review is overdue by more than slot duration."""
        if not self.is_due:
            return False
        overdue_by = datetime.utcnow() - self.scheduled_for
        return overdue_by.total_seconds() * 1000 > self.slot_duration_ms

    @property
    def time_until(self) -> timedelta:
        """Time until scheduled review."""
        return self.scheduled_for - datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schedule_id": self.schedule_id,
            "item": self.item.to_dict(),
            "scheduled_for": self.scheduled_for.isoformat(),
            "slot_duration_ms": self.slot_duration_ms,
            "batch_id": self.batch_id,
            "priority_order": self.priority_order,
            "is_due": self.is_due,
            "is_overdue": self.is_overdue,
            "created_at": self.created_at.isoformat(),
        }


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "ReviewUrgency",
    "ReviewOutcome",
    "ReviewCategory",
    "ReviewItem",
    "ReviewSession",
    "ScheduledReview",
]
```

---

## 4. Models __init__.py

```python
# src/hl_mcp/models/__init__.py
"""
Human Layer MCP - Data Models
=============================

All data models used across Human Layer MCP.

Modules:
    - enums: All enumerations (HLS-MDL-007)
    - finding: Finding and SecurityFinding (HLS-MDL-001)
    - layer_result: LayerResult (HLS-MDL-002)
    - report: HumanLayerReport (HLS-MDL-003)
    - journey: Journey, JourneyStep, JourneyResult (HLS-MDL-004)
    - test: GeneratedTest, TestConsensus (HLS-MDL-005)
    - review: ReviewItem, ReviewSession (HLS-MDL-006)

Example:
    >>> from hl_mcp.models import (
    ...     Finding,
    ...     SecurityFinding,
    ...     LayerResult,
    ...     HumanLayerReport,
    ...     Journey,
    ...     JourneyStep,
    ...     GeneratedTest,
    ...     ReviewItem,
    ...     VetoLevel,
    ...     Severity,
    ... )

Author: Human Layer Team
Version: 1.0.0
"""

# ============================================================================
# ENUMS (HLS-MDL-007)
# ============================================================================
from .enums import (
    # Veto and Status
    VetoLevel,
    LayerStatus,
    Severity,
    # HITL
    HITLDecision,
    ReviewTrigger,
    FeedbackType,
    # Browser
    BrowserAction,
    # Identifiers
    LayerID,
    PerspectiveID,
    RecipientType,
    TargetType,
    LayerPack,
)

# ============================================================================
# FINDING (HLS-MDL-001)
# ============================================================================
from .finding import (
    Finding,
    SecurityFinding,
    findings_by_severity,
    count_blocking_findings,
    deduplicate_findings,
)

# ============================================================================
# LAYER RESULT (HLS-MDL-002)
# ============================================================================
from .layer_result import LayerResult

# ============================================================================
# REPORT (HLS-MDL-003)
# ============================================================================
from .report import HumanLayerReport

# ============================================================================
# JOURNEY (HLS-MDL-004)
# ============================================================================
from .journey import (
    JourneyType,
    StepStatus,
    JourneyStep,
    Journey,
    JourneyResult,
    JourneyBuilder,
    AccessibilityIssue,
    AccessibilityResult,
)

# ============================================================================
# TEST (HLS-MDL-005)
# ============================================================================
from .test import (
    TestPriority,
    TestStatus,
    GeneratedTest,
    TestConsensus,
    TestSuite,
)

# ============================================================================
# REVIEW (HLS-MDL-006)
# ============================================================================
from .review import (
    ReviewUrgency,
    ReviewOutcome,
    ReviewCategory,
    ReviewItem,
    ReviewSession,
    ScheduledReview,
)

# ============================================================================
# EXPORTS
# ============================================================================
__all__ = [
    # === Enums ===
    "VetoLevel",
    "LayerStatus",
    "Severity",
    "HITLDecision",
    "ReviewTrigger",
    "FeedbackType",
    "BrowserAction",
    "LayerID",
    "PerspectiveID",
    "RecipientType",
    "TargetType",
    "LayerPack",
    # === Finding ===
    "Finding",
    "SecurityFinding",
    "findings_by_severity",
    "count_blocking_findings",
    "deduplicate_findings",
    # === Layer Result ===
    "LayerResult",
    # === Report ===
    "HumanLayerReport",
    # === Journey ===
    "JourneyType",
    "StepStatus",
    "JourneyStep",
    "Journey",
    "JourneyResult",
    "JourneyBuilder",
    "AccessibilityIssue",
    "AccessibilityResult",
    # === Test ===
    "TestPriority",
    "TestStatus",
    "GeneratedTest",
    "TestConsensus",
    "TestSuite",
    # === Review ===
    "ReviewUrgency",
    "ReviewOutcome",
    "ReviewCategory",
    "ReviewItem",
    "ReviewSession",
    "ScheduledReview",
]

__version__ = "1.0.0"
```

---

## 5. TESTES ADICIONAIS

```python
# tests/unit/models/test_journey.py
"""Tests for HLS-MDL-004: Journey module."""

import pytest
from hl_mcp.models.journey import (
    JourneyStep,
    Journey,
    JourneyResult,
    JourneyBuilder,
    JourneyType,
    StepStatus,
)
from hl_mcp.models.enums import BrowserAction


class TestJourneyStep:
    """Tests for JourneyStep."""

    def test_create_basic(self):
        step = JourneyStep(
            action=BrowserAction.CLICK,
            target="#button",
        )
        assert step.action == BrowserAction.CLICK
        assert step.target == "#button"

    def test_auto_description(self):
        step = JourneyStep(action=BrowserAction.GOTO, target="/login")
        assert "Navigate" in step.description

    def test_factory_methods(self):
        step1 = JourneyStep.goto("/home")
        assert step1.action == BrowserAction.GOTO

        step2 = JourneyStep.click("#btn")
        assert step2.action == BrowserAction.CLICK

        step3 = JourneyStep.fill("#input", "value")
        assert step3.action == BrowserAction.FILL
        assert step3.value == "value"

    def test_requires_value_validation(self):
        with pytest.raises(ValueError):
            JourneyStep(action=BrowserAction.FILL, target="#input", value="")

    def test_serialization(self):
        step = JourneyStep.click("#btn", description="Click button")
        data = step.to_dict()
        restored = JourneyStep.from_dict(data)
        assert restored.action == step.action
        assert restored.target == step.target


class TestJourney:
    """Tests for Journey."""

    def test_create_basic(self):
        journey = Journey(
            name="Test Journey",
            base_url="https://example.com",
            steps=[JourneyStep.goto("/")],
        )
        assert journey.step_count == 1

    def test_must_have_steps(self):
        with pytest.raises(ValueError):
            Journey(name="Empty", base_url="https://example.com", steps=[])

    def test_estimated_duration(self):
        journey = Journey(
            name="Test",
            base_url="https://example.com",
            steps=[
                JourneyStep.goto("/"),
                JourneyStep.click("#btn"),
                JourneyStep.wait("#result", timeout_ms=5000),
            ],
        )
        # Should include wait timeout
        assert journey.estimated_duration_ms >= 5000

    def test_serialization(self):
        journey = Journey(
            name="Test",
            base_url="https://example.com",
            steps=[JourneyStep.goto("/"), JourneyStep.click("#btn")],
            tags=["login", "smoke"],
        )
        data = journey.to_dict()
        restored = Journey.from_dict(data)
        assert restored.name == journey.name
        assert restored.step_count == journey.step_count


class TestJourneyBuilder:
    """Tests for JourneyBuilder."""

    def test_fluent_building(self):
        journey = (
            JourneyBuilder("Login", "https://app.example.com")
            .goto("/login")
            .fill("#email", "test@test.com")
            .fill("#password", "secret")
            .click("#submit")
            .wait(".dashboard")
            .build()
        )

        assert journey.name == "Login"
        assert journey.step_count == 5

    def test_with_metadata(self):
        journey = (
            JourneyBuilder("Test", "https://example.com")
            .description("A test journey")
            .journey_type(JourneyType.ERROR_PATH)
            .tag("smoke", "critical")
            .goto("/")
            .build()
        )

        assert journey.description == "A test journey"
        assert journey.journey_type == JourneyType.ERROR_PATH
        assert "smoke" in journey.tags


class TestJourneyResult:
    """Tests for JourneyResult."""

    def test_success_result(self):
        journey = Journey(
            name="Test",
            base_url="https://example.com",
            steps=[JourneyStep.goto("/")],
        )
        result = JourneyResult.success_result(journey, duration_ms=1000)

        assert result.success is True
        assert result.steps_completed == 1
        assert result.completion_rate == 1.0

    def test_failure_result(self):
        journey = Journey(
            name="Test",
            base_url="https://example.com",
            steps=[
                JourneyStep.goto("/"),
                JourneyStep.click("#missing"),
            ],
        )
        result = JourneyResult.failure_result(
            journey,
            failed_step_index=1,
            error="Element not found",
            duration_ms=500,
        )

        assert result.success is False
        assert result.steps_completed == 1
        assert result.failed_step_index == 1
        assert "Element not found" in result.errors


# tests/unit/models/test_test.py
"""Tests for HLS-MDL-005: Test module."""

import pytest
from hl_mcp.models.test import (
    GeneratedTest,
    TestConsensus,
    TestSuite,
    TestPriority,
    TestStatus,
)
from hl_mcp.models.enums import PerspectiveID


class TestGeneratedTest:
    """Tests for GeneratedTest."""

    def test_create_basic(self):
        test = GeneratedTest(
            perspective_id=PerspectiveID.TIRED_USER,
            name="test_handles_multiple_clicks",
            description="System handles rapid clicks",
            assertions=["Operation is idempotent"],
        )
        assert test.name.startswith("test_")
        assert test.perspective_id == PerspectiveID.TIRED_USER

    def test_auto_generates_id(self):
        test = GeneratedTest.create(
            perspective=PerspectiveID.AUDITOR,
            name="test_audit_trail",
            description="Audit trail exists",
        )
        assert test.test_id != ""
        assert len(test.test_id) == 12

    def test_sanitizes_name(self):
        test = GeneratedTest.create(
            perspective=PerspectiveID.POWER_USER,
            name="What if user clicks twice?",  # Not valid Python
            description="Test",
        )
        # Should be converted to valid function name
        assert test.name.startswith("test_")
        assert " " not in test.name
        assert "?" not in test.name

    def test_adds_perspective_tag(self):
        test = GeneratedTest.create(
            perspective=PerspectiveID.MALICIOUS_INSIDER,
            name="test_access_control",
            description="Test",
        )
        assert "malicious_insider" in test.tags

    def test_is_security_test(self):
        security_test = GeneratedTest.create(
            perspective=PerspectiveID.MALICIOUS_INSIDER,
            name="test_sql_injection",
            description="Test",
        )
        assert security_test.is_security_test is True

        non_security = GeneratedTest.create(
            perspective=PerspectiveID.TIRED_USER,
            name="test_loading",
            description="Test",
        )
        assert non_security.is_security_test is False

    def test_generate_code(self):
        test = GeneratedTest.create(
            perspective=PerspectiveID.AUDITOR,
            name="test_audit_exists",
            description="Verify audit trail",
            assertions=["Audit log exists", "Contains user ID"],
        )
        code = test.generate_code()

        assert "def test_audit_exists" in code
        assert "Audit log exists" in code
        assert "pytest" not in code.lower() or "import" not in code


class TestTestConsensus:
    """Tests for TestConsensus."""

    def test_create_basic(self):
        test = GeneratedTest.create(
            perspective=PerspectiveID.MALICIOUS_INSIDER,
            name="test_access",
            description="Test",
        )
        consensus = TestConsensus(
            test=test,
            agreement_score=0.85,
            contributing_perspectives=["malicious_insider", "auditor"],
            confidence=0.9,
        )

        assert consensus.is_high_agreement is True
        assert consensus.perspective_count == 2

    def test_validates_scores(self):
        test = GeneratedTest.create(
            perspective=PerspectiveID.TIRED_USER,
            name="test",
            description="Test",
        )
        with pytest.raises(ValueError):
            TestConsensus(
                test=test,
                agreement_score=1.5,  # Invalid
                contributing_perspectives=[],
                confidence=0.5,
            )


class TestTestSuite:
    """Tests for TestSuite."""

    def test_add_tests(self):
        suite = TestSuite(name="Login Tests")

        suite.add_test(GeneratedTest.create(
            perspective=PerspectiveID.TIRED_USER,
            name="test_1",
            description="Test 1",
            priority=TestPriority.HIGH,
        ))
        suite.add_test(GeneratedTest.create(
            perspective=PerspectiveID.MALICIOUS_INSIDER,
            name="test_2",
            description="Test 2",
            priority=TestPriority.CRITICAL,
        ))

        assert suite.test_count == 2
        assert suite.high_priority_count == 2
        assert suite.security_test_count == 1

    def test_by_perspective(self):
        suite = TestSuite(name="Tests")
        suite.add_test(GeneratedTest.create(PerspectiveID.TIRED_USER, "t1", "d"))
        suite.add_test(GeneratedTest.create(PerspectiveID.TIRED_USER, "t2", "d"))
        suite.add_test(GeneratedTest.create(PerspectiveID.AUDITOR, "t3", "d"))

        by_persp = suite.by_perspective
        assert len(by_persp["tired_user"]) == 2
        assert len(by_persp["auditor"]) == 1


# tests/unit/models/test_review.py
"""Tests for HLS-MDL-006: Review module."""

import pytest
from datetime import datetime, timedelta
from hl_mcp.models.review import (
    ReviewItem,
    ReviewSession,
    ScheduledReview,
    ReviewUrgency,
    ReviewOutcome,
    ReviewCategory,
)
from hl_mcp.models.enums import ReviewTrigger, FeedbackType


class TestReviewItem:
    """Tests for ReviewItem."""

    def test_create_basic(self):
        item = ReviewItem(
            item_type="finding",
            content={"title": "Test Finding"},
            urgency=ReviewUrgency.HIGH,
        )
        assert item.item_type == "finding"
        assert item.urgency == ReviewUrgency.HIGH

    def test_is_security_related(self):
        security_item = ReviewItem(
            item_type="finding",
            content={},
            category=ReviewCategory.SECURITY,
        )
        assert security_item.is_security_related is True

        with_trigger = ReviewItem(
            item_type="finding",
            content={},
            triggers=[ReviewTrigger.SECURITY_CONCERN],
        )
        assert with_trigger.is_security_related is True

    def test_expiration(self):
        # Not expired
        item1 = ReviewItem(
            item_type="test",
            content={},
            expires_at=datetime.utcnow() + timedelta(hours=1),
        )
        assert item1.is_expired is False

        # Expired
        item2 = ReviewItem(
            item_type="test",
            content={},
            expires_at=datetime.utcnow() - timedelta(hours=1),
        )
        assert item2.is_expired is True

    def test_serialization(self):
        item = ReviewItem(
            item_type="finding",
            content={"key": "value"},
            urgency=ReviewUrgency.MEDIUM,
            triggers=[ReviewTrigger.LOW_CONFIDENCE],
        )
        data = item.to_dict()
        restored = ReviewItem.from_dict(data)

        assert restored.item_type == item.item_type
        assert restored.urgency == item.urgency


class TestReviewSession:
    """Tests for ReviewSession."""

    def test_lifecycle(self):
        item = ReviewItem(item_type="test", content={})
        session = ReviewSession.create(item)

        assert session.is_complete is False
        assert session.is_in_progress is False

        session.start(reviewer_id="user_123")
        assert session.is_in_progress is True

        session.complete(
            outcome=ReviewOutcome.APPROVED,
            reasoning="Looks good",
            feedback_type=FeedbackType.AGREE,
        )
        assert session.is_complete is True
        assert session.was_approved is True
        assert session.duration_ms > 0

    def test_rejection(self):
        item = ReviewItem(item_type="test", content={})
        session = ReviewSession.create(item)
        session.start()
        session.complete(
            outcome=ReviewOutcome.REJECTED,
            reasoning="Needs more work",
            feedback_type=FeedbackType.OVERRIDE,
        )

        assert session.was_approved is False
        assert session.feedback_type == FeedbackType.OVERRIDE


class TestScheduledReview:
    """Tests for ScheduledReview."""

    def test_is_due(self):
        item = ReviewItem(item_type="test", content={})

        # Due now
        scheduled1 = ScheduledReview(
            item=item,
            scheduled_for=datetime.utcnow() - timedelta(minutes=5),
        )
        assert scheduled1.is_due is True

        # Not due yet
        scheduled2 = ScheduledReview(
            item=item,
            scheduled_for=datetime.utcnow() + timedelta(hours=1),
        )
        assert scheduled2.is_due is False

    def test_is_overdue(self):
        item = ReviewItem(item_type="test", content={})

        # Overdue by more than slot duration
        scheduled = ScheduledReview(
            item=item,
            scheduled_for=datetime.utcnow() - timedelta(minutes=10),
            slot_duration_ms=300000,  # 5 minutes
        )
        assert scheduled.is_overdue is True
```

---

## RESUMO DO BLOCK 05

| Módulo | Arquivo | Linhas | Classes/Funções |
|--------|---------|--------|-----------------|
| HLS-MDL-004 | `models/journey.py` | ~450 | 8 classes |
| HLS-MDL-005 | `models/test.py` | ~320 | 5 classes |
| HLS-MDL-006 | `models/review.py` | ~350 | 6 classes |
| Init | `models/__init__.py` | ~120 | 40+ exports |
| Tests | `tests/unit/models/` | ~400 | 25+ tests |

**Total Block 05**: ~1,640 linhas

**Total Models Layer**: ~2,940 linhas (Block 04 + Block 05)

---

## PRÓXIMO BLOCK

O Block 06 vai cobrir:
1. HLS-LLM-001: LLMClient interface
2. HLS-LLM-002: ClaudeClient implementation
3. HLS-LLM-003: OpenAIClient implementation
4. HLS-LLM-004: PromptTemplates
5. HLS-LLM-005: ResponseParser

---

*ROADMAP_BLOCK_05_MODELS_COMPLETE.md - v1.0.0 - 2026-02-01*
