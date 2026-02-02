"""DOC-003: Definition of Ready (DoR) and EARS Validation Module

Extracted from gate_runner.py for better separation of concerns.

This module provides:
- Definition of Ready (DoR) validation using INVEST criteria
- EARS (Easy Approach to Requirements Syntax) pattern validation

References:
- INVEST: Independent, Negotiable, Valuable, Estimable, Small, Testable
- EARS: Ubiquitous, Event-Driven, State-Driven, Optional, Unwanted patterns
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Pattern, Tuple, Union
import logging

logger = logging.getLogger(__name__)



# =============================================================================
# Definition of Ready (DoR) Validation
# =============================================================================

# INVEST criteria for Definition of Ready
# Each criterion maps to a validation check
INVEST_CRITERIA: Dict[str, str] = {
    "independent": "The story/task can be completed without external dependencies",
    "negotiable": "Requirements are flexible enough for implementation choices",
    "valuable": "Delivers value to stakeholders",
    "estimable": "Size/effort can be reasonably estimated",
    "small": "Can be completed within a sprint",
    "testable": "Clear acceptance criteria exist",
}


@dataclass
class DoRResult:
    """Result of Definition of Ready validation."""
    is_ready: bool
    passed_criteria: List[str]
    failed_criteria: List[str]
    warnings: List[str]
    details: Dict[str, Any]


def validate_definition_of_ready(
    spec: Dict[str, Any],
    *,
    strict: bool = False,
) -> DoRResult:
    """Validate that a specification meets Definition of Ready criteria.

    This function checks the INVEST criteria:
    - Independent: No blocking external dependencies
    - Negotiable: Implementation approach is flexible
    - Valuable: Clear value proposition stated
    - Estimable: Effort estimate provided
    - Small: Scope is bounded and fits in a sprint
    - Testable: Acceptance criteria defined

    Args:
        spec: The specification dictionary to validate.
              Expected structure:
              {
                  "title": str,
                  "description": str,
                  "acceptance_criteria": list[str],
                  "dependencies": list[str] (optional),
                  "estimate_hours": int (optional),
                  "value_statement": str (optional),
                  "scope_boundary": str (optional),
              }
        strict: If True, all 6 INVEST criteria must pass.
                If False (default), only 4 mandatory criteria required.

    Returns:
        DoRResult with validation status and details.

    Example:
        >>> spec = {
        ...     "title": "Add login button",
        ...     "description": "Add a login button to the header",
        ...     "acceptance_criteria": ["Button visible", "Redirects to /login"],
        ...     "estimate_hours": 4,
        ...     "value_statement": "Users can access their accounts",
        ... }
        >>> result = validate_definition_of_ready(spec)
        >>> result.is_ready
        True
    """
    passed: List[str] = []
    failed: List[str] = []
    warnings: List[str] = []
    details: Dict[str, Any] = {"spec_keys": list(spec.keys())}

    # 1. INDEPENDENT: Check for blocking dependencies
    dependencies = spec.get("dependencies", [])
    blocking_deps = [d for d in dependencies if isinstance(d, str) and d.startswith("BLOCKING:")]
    if not blocking_deps:
        passed.append("independent")
        details["independent"] = {"blocking_deps": 0}
    else:
        failed.append("independent")
        details["independent"] = {
            "blocking_deps": len(blocking_deps),
            "blockers": blocking_deps,
        }

    # 2. NEGOTIABLE: Check that description doesn't over-specify implementation
    description = spec.get("description", "")
    # Over-specified indicators: mentions specific file paths, line numbers, or exact code
    over_specified_patterns = [
        r"line \d+",
        r"file:.+\.\w+",
        r"```\w+\n.+```",  # Code blocks in description
    ]
    is_over_specified = any(
        re.search(pattern, description, re.IGNORECASE | re.DOTALL)
        for pattern in over_specified_patterns
    )
    if not is_over_specified and description:
        passed.append("negotiable")
        details["negotiable"] = {"over_specified": False}
    elif not description:
        failed.append("negotiable")
        details["negotiable"] = {"error": "No description provided"}
    else:
        # Over-specified is a warning, not a failure
        passed.append("negotiable")
        warnings.append("Description may be over-specified")
        details["negotiable"] = {"over_specified": True, "warning": True}

    # 3. VALUABLE: Check for value statement or user story format
    value_statement = spec.get("value_statement", "")
    title = spec.get("title", "")
    # User story format: "As a <role>, I want <action>, so that <benefit>"
    has_user_story = bool(
        re.search(r"as an?\s+\w+.*i want.*so that", title + " " + description, re.IGNORECASE)
    )
    if value_statement or has_user_story:
        passed.append("valuable")
        details["valuable"] = {
            "has_value_statement": bool(value_statement),
            "has_user_story": has_user_story,
        }
    else:
        failed.append("valuable")
        details["valuable"] = {
            "error": "No value statement or user story format found",
        }

    # 4. ESTIMABLE: Check for effort estimate
    estimate = spec.get("estimate_hours") or spec.get("estimate") or spec.get("story_points")
    if estimate is not None:
        try:
            estimate_val = float(estimate)
            if estimate_val > 0:
                passed.append("estimable")
                details["estimable"] = {"estimate": estimate_val}
            else:
                failed.append("estimable")
                details["estimable"] = {"error": "Estimate must be positive"}
        except (TypeError, ValueError):
            failed.append("estimable")
            details["estimable"] = {"error": f"Invalid estimate format: {estimate}"}
    else:
        failed.append("estimable")
        details["estimable"] = {"error": "No estimate provided"}

    # 5. SMALL: Check scope boundary or estimate reasonability
    scope_boundary = spec.get("scope_boundary", "")
    # Small = has scope boundary OR estimate <= 40 hours (1 week)
    estimate_small = False
    if estimate is not None:
        try:
            estimate_small = float(estimate) <= 40
        except (TypeError, ValueError):
            logger.debug(f"GATE: Gate operation failed: {e}")

    if scope_boundary or estimate_small:
        passed.append("small")
        details["small"] = {
            "has_scope_boundary": bool(scope_boundary),
            "estimate_reasonable": estimate_small,
        }
    else:
        failed.append("small")
        details["small"] = {
            "error": "No scope boundary and estimate exceeds 40 hours",
        }

    # 6. TESTABLE: Check for acceptance criteria
    acceptance_criteria = spec.get("acceptance_criteria", [])
    if isinstance(acceptance_criteria, list) and len(acceptance_criteria) >= 1:
        passed.append("testable")
        details["testable"] = {"criteria_count": len(acceptance_criteria)}
    else:
        failed.append("testable")
        details["testable"] = {
            "error": "No acceptance criteria defined",
            "criteria_count": 0,
        }

    # Determine overall readiness
    # Mandatory criteria: independent, valuable, testable (at minimum)
    mandatory_criteria = {"independent", "testable"}
    if strict:
        mandatory_criteria = set(INVEST_CRITERIA.keys())

    mandatory_passed = mandatory_criteria.issubset(set(passed))

    return DoRResult(
        is_ready=mandatory_passed,
        passed_criteria=passed,
        failed_criteria=failed,
        warnings=warnings,
        details=details,
    )


# =============================================================================
# EARS (Easy Approach to Requirements Syntax) Validation
# =============================================================================

# EARS pattern types and their keywords
# OPT-07-001: Pre-compiled regex patterns for better performance
EARS_PATTERNS: Dict[str, Dict[str, Any]] = {
    "ubiquitous": {
        "pattern": r"\bthe\s+\w+\s+shall\b",
        "compiled": re.compile(r"\bthe\s+\w+\s+shall\b", re.IGNORECASE),
        "description": "The <system> shall <action>",
        "example": "The system shall validate user input",
    },
    "event_driven": {
        "pattern": r"\bwhen\s+.+,\s*the\s+\w+\s+shall\b",
        "compiled": re.compile(r"\bwhen\s+.+,\s*the\s+\w+\s+shall\b", re.IGNORECASE),
        "description": "When <trigger>, the <system> shall <action>",
        "example": "When the user clicks login, the system shall authenticate",
    },
    "state_driven": {
        "pattern": r"\bwhile\s+.+,\s*the\s+\w+\s+shall\b",
        "compiled": re.compile(r"\bwhile\s+.+,\s*the\s+\w+\s+shall\b", re.IGNORECASE),
        "description": "While <state>, the <system> shall <action>",
        "example": "While offline, the system shall cache data locally",
    },
    "optional": {
        "pattern": r"\bwhere\s+.+,\s*the\s+\w+\s+shall\b",
        "compiled": re.compile(r"\bwhere\s+.+,\s*the\s+\w+\s+shall\b", re.IGNORECASE),
        "description": "Where <feature>, the <system> shall <action>",
        "example": "Where dark mode is enabled, the system shall use dark theme",
    },
    "unwanted": {
        "pattern": r"\bif\s+.+,\s*the\s+\w+\s+shall\s+not\b",
        "compiled": re.compile(r"\bif\s+.+,\s*the\s+\w+\s+shall\s+not\b", re.IGNORECASE),
        "description": "If <condition>, the <system> shall not <action>",
        "example": "If session expires, the system shall not allow access",
    },
}


@dataclass
class EarsResult:
    """Result of EARS pattern validation."""
    has_ears_format: bool
    detected_patterns: List[str]
    pattern_details: Dict[str, List[str]]
    suggestions: List[str]


def has_ears_keywords(text: str) -> EarsResult:
    """Check if text follows EARS (Easy Approach to Requirements Syntax) format.

    EARS provides 5 patterns for writing requirements:
    - UBIQUITOUS: "The <system> shall <action>"
    - EVENT-DRIVEN: "When <trigger>, the <system> shall <action>"
    - STATE-DRIVEN: "While <state>, the <system> shall <action>"
    - OPTIONAL: "Where <feature>, the <system> shall <action>"
    - UNWANTED: "If <condition>, the <system> shall not <action>"

    Args:
        text: The requirement text to analyze.

    Returns:
        EarsResult with detection status and matched patterns.

    Example:
        >>> result = has_ears_keywords("The system shall validate user input")
        >>> result.has_ears_format
        True
        >>> result.detected_patterns
        ['ubiquitous']
    """
    detected: List[str] = []
    pattern_details: Dict[str, List[str]] = {}
    suggestions: List[str] = []

    # Normalize text for matching
    normalized = text.strip()  # OPT-07-001: No need to lowercase, compiled pattern has IGNORECASE

    for pattern_name, pattern_info in EARS_PATTERNS.items():
        # OPT-07-001: Use pre-compiled pattern for better performance
        compiled_pattern = pattern_info["compiled"]
        matches = compiled_pattern.findall(normalized)
        if matches:
            detected.append(pattern_name)
            pattern_details[pattern_name] = [m.strip() for m in matches]

    # Generate suggestions if no EARS patterns found
    if not detected:
        # Check for common non-EARS patterns and suggest improvements
        if "should" in normalized and "shall" not in normalized:
            suggestions.append("Replace 'should' with 'shall' for EARS compliance")
        if "must" in normalized and "shall" not in normalized:
            suggestions.append("Replace 'must' with 'shall' for EARS compliance")
        if "will" in normalized and "shall" not in normalized:
            suggestions.append("Replace 'will' with 'shall' for EARS compliance")
        if not suggestions:
            suggestions.append(
                "Consider using EARS format: 'The <system> shall <action>'"
            )

    return EarsResult(
        has_ears_format=len(detected) > 0,
        detected_patterns=detected,
        pattern_details=pattern_details,
        suggestions=suggestions,
    )


def validate_ears_requirement(
    requirement: str,
    *,
    strict: bool = False,
) -> Tuple[bool, List[str]]:
    """Validate a single requirement against EARS format.

    Args:
        requirement: The requirement text to validate.
        strict: If True, requires "shall" keyword. If False, allows "should"/"must".

    Returns:
        Tuple of (is_valid, list_of_issues).

    Example:
        >>> valid, issues = validate_ears_requirement("The system shall log errors")
        >>> valid
        True
        >>> issues
        []
    """
    issues: List[str] = []

    # Check for "shall" keyword (EARS requirement)
    has_shall = "shall" in requirement.lower()

    if strict and not has_shall:
        issues.append("Missing 'shall' keyword (EARS requires 'shall')")
        return False, issues

    if not strict and not has_shall:
        # Check for acceptable alternatives
        has_alternative = any(
            word in requirement.lower() for word in ["should", "must", "will"]
        )
        if not has_alternative:
            issues.append("Missing requirement keyword ('shall', 'should', 'must')")
            return False, issues

    # Check for subject-verb pattern
    ears_result = has_ears_keywords(requirement)
    if not ears_result.has_ears_format and strict:
        issues.append("Does not match any EARS pattern")
        issues.extend(ears_result.suggestions)
        return False, issues

    # LIE-003 FIX: Return False if there are any issues, not True
    # This implements fail-closed pattern - any issues should fail validation
    if issues:
        return False, issues

    return True, issues


# =============================================================================
# Convenience exports
# =============================================================================

__all__ = [
    # DoR
    "INVEST_CRITERIA",
    "DoRResult",
    "validate_definition_of_ready",
    # EARS
    "EARS_PATTERNS",
    "EarsResult",
    "has_ears_keywords",
    "validate_ears_requirement",
]
