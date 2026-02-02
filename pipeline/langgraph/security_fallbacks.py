"""Security Fallbacks for Colang Actions.

This module provides pure Python implementations of critical security actions
that normally require NeMo Guardrails. When NeMo is not available, these
fallbacks ensure reasonable security protections remain active.

The fallbacks integrate with existing security modules:
- trust_boundaries: For permission validation
- stack_guardrails: For circuit breaker checks

Critical Actions Implemented:
- scan_for_injection: Prompt injection detection
- scan_for_sensitive_data: PII/secret detection
- redact_sensitive_data: Data redaction
- validate_permission: Access control validation
- check_circuit_breakers: Resilience monitoring
- validate_stack_health: Health monitoring

Author: Pipeline Autonomo Team
Version: 1.0.0 (2026-01-16)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Pattern, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# INJECTION DETECTION PATTERNS
# =============================================================================

# Common prompt injection patterns
INJECTION_PATTERNS: List[Tuple[str, Pattern]] = [
    # SQL Injection (GAP-SEC-004 FIX: Enhanced regex patterns)
    ("sql_injection", re.compile(
        r"(?i)("
        # Basic SQL injection with quote bypass
        r"';?\s*(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|TRUNCATE|EXEC|EXECUTE)\s|"
        # UNION-based injection
        r"UNION\s+(ALL\s+)?SELECT|"
        # Comment-based termination
        r"--\s*$|/\*.*\*/|;\s*--|#\s*$|"
        # Boolean-based blind injection
        r"'\s*(OR|AND)\s+('?1'?\s*=\s*'?1'?|'?true'?)|"
        # Time-based blind injection
        r"WAITFOR\s+DELAY|SLEEP\s*\(|BENCHMARK\s*\(|"
        # Stacked queries
        r";\s*(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC)|"
        # Error-based injection
        r"EXTRACTVALUE\s*\(|UPDATEXML\s*\(|"
        # NoSQL injection patterns
        r"\$where\s*:|{\s*\$gt\s*:|{\s*\$ne\s*:|{\s*\$regex\s*:"
        r")",
        re.IGNORECASE
    )),

    # XSS Patterns
    ("xss", re.compile(
        r"(?i)(<script[^>]*>|</script>|javascript:|on\w+\s*=|<img[^>]+onerror)",
        re.IGNORECASE
    )),

    # Command Injection
    ("command_injection", re.compile(
        r"(?i)(\|\||&&|;|\$\(|`|\||>\s*[/\w]|<\s*[/\w]|"
        r"rm\s+-rf|chmod\s+[0-7]{3}|wget\s+|curl\s+.*\s+-o|"
        r"python\s+-c|bash\s+-c|eval\s*\()",
        re.IGNORECASE
    )),

    # Prompt Injection Attempts
    # FIX: Removed overly broad patterns that cause false positives:
    #   - "system\s*:\s*" matches pipeline state dicts with "system:" keys
    #   - "\[system\]" and "<\|system\|>" are too broad for pipeline context
    # Kept only high-confidence injection patterns
    ("prompt_injection", re.compile(
        r"(?i)(ignore\s+(previous|all|above)\s+instructions?|"
        r"disregard\s+.*\s+instructions?|"
        r"forget\s+.*\s+(instructions?|rules?)|"
        r"you\s+are\s+now\s+|"
        r"pretend\s+you\s+are\s+|"
        r"act\s+as\s+if\s+|"
        r"new\s+instructions?:|"
        r"###\s*instruction|"
        r"jailbreak|"
        r"DAN\s+mode|"
        r"do\s+anything\s+now)",
        re.IGNORECASE
    )),

    # Path Traversal
    ("path_traversal", re.compile(
        r"(?i)(\.\./|\.\.\\|%2e%2e%2f|%2e%2e/|\.%2e/|%2e\./)",
        re.IGNORECASE
    )),

    # LDAP Injection
    ("ldap_injection", re.compile(
        r"(?i)(\)\s*\(|\(\s*\||\(\s*&|\*\s*\)|\\[0-9a-f]{2})",
        re.IGNORECASE
    )),
]

# High-confidence injection indicators (more specific patterns)
HIGH_CONFIDENCE_INJECTION_PATTERNS: List[Tuple[str, Pattern]] = [
    ("definite_prompt_injection", re.compile(
        r"(?i)(ignore\s+all\s+previous\s+instructions|"
        r"you\s+must\s+now\s+ignore|"
        r"new\s+system\s+prompt\s*:|"
        r"override\s+safety|"
        r"bypass\s+restrictions)",
        re.IGNORECASE
    )),
]


# =============================================================================
# SENSITIVE DATA PATTERNS
# =============================================================================

# PII and sensitive data patterns
SENSITIVE_DATA_PATTERNS: Dict[str, Pattern] = {
    # Credentials
    "api_key": re.compile(
        r"(?i)(api[_-]?key|apikey|api[_-]?secret)\s*[=:]\s*['\"]?([a-zA-Z0-9_\-]{16,})['\"]?",
        re.IGNORECASE
    ),
    "password": re.compile(
        r"(?i)(password|passwd|pwd|secret)\s*[=:]\s*['\"]?([^\s'\"]{4,})['\"]?",
        re.IGNORECASE
    ),
    "bearer_token": re.compile(
        r"(?i)(bearer|authorization)\s*[=:]\s*['\"]?([a-zA-Z0-9_\-\.]+)['\"]?",
        re.IGNORECASE
    ),
    "jwt_token": re.compile(
        r"eyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+",
        re.IGNORECASE
    ),

    # AWS Credentials
    "aws_access_key": re.compile(
        r"(?:A3T[A-Z0-9]|AKIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ASIA)[A-Z0-9]{16}",
    ),
    "aws_secret_key": re.compile(
        r"(?i)(aws[_-]?secret[_-]?access[_-]?key)\s*[=:]\s*['\"]?([a-zA-Z0-9/+]{40})['\"]?",
    ),

    # Generic secrets
    "generic_secret": re.compile(
        r"(?i)(secret|token|private[_-]?key|encryption[_-]?key)\s*[=:]\s*['\"]?([a-zA-Z0-9_\-]{16,})['\"]?",
        re.IGNORECASE
    ),

    # Connection strings
    "connection_string": re.compile(
        r"(?i)(mongodb|postgres|mysql|redis|amqp)://[^\s\"']+",
        re.IGNORECASE
    ),

    # PII
    "email": re.compile(
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    ),
    "phone_number": re.compile(
        r"(?:\+?1[-.\s]?)?(?:\([0-9]{3}\)|[0-9]{3})[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}",
    ),
    "ssn": re.compile(
        r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b",
    ),
    "credit_card": re.compile(
        r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b",
    ),

    # IP Addresses (internal)
    "internal_ip": re.compile(
        r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2[0-9]|3[01])\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3})\b",
    ),
}

# Critical patterns that MUST be redacted
CRITICAL_REDACTION_PATTERNS: List[Tuple[str, Pattern, str]] = [
    ("api_key", re.compile(r"(?i)(api[_-]?key|apikey)\s*[=:]\s*['\"]?([a-zA-Z0-9_\-]{16,})['\"]?"), r"\1=[REDACTED]"),
    ("password", re.compile(r"(?i)(password|passwd|pwd)\s*[=:]\s*['\"]?[^\s'\"]+['\"]?"), r"\1=[REDACTED]"),
    ("bearer_token", re.compile(r"(?i)(bearer)\s+[a-zA-Z0-9_\-\.]+"), r"\1 [REDACTED]"),
    ("jwt", re.compile(r"eyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+"), "[REDACTED_JWT]"),
    ("aws_key", re.compile(r"(?:A3T[A-Z0-9]|AKIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ASIA)[A-Z0-9]{16}"), "[REDACTED_AWS_KEY]"),
    ("connection_string", re.compile(r"(?i)(mongodb|postgres|mysql|redis|amqp)://[^\s\"']+"), r"\1://[REDACTED]"),
    ("secret_value", re.compile(r"(?i)(secret|token|private[_-]?key)\s*[=:]\s*['\"]?([a-zA-Z0-9_\-]{16,})['\"]?"), r"\1=[REDACTED]"),
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class InjectionScanResult:
    """Result of injection scan."""

    injection_detected: bool
    input_safe: bool
    detected_types: List[str]
    high_confidence: bool
    risk_score: float  # 0.0 to 1.0
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Colang compatibility."""
        return {
            "injection_detected": self.injection_detected,
            "input_safe": self.input_safe,
            "detected_types": self.detected_types,
            "high_confidence": self.high_confidence,
            "risk_score": self.risk_score,
            "details": self.details,
        }


@dataclass
class SensitiveDataScanResult:
    """Result of sensitive data scan."""

    sensitive_data_found: bool
    sensitive_types: List[str]
    occurrences: Dict[str, int]
    risk_level: str  # "low", "medium", "high", "critical"
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Colang compatibility."""
        return {
            "sensitive_data_found": self.sensitive_data_found,
            "sensitive_types": self.sensitive_types,
            "occurrences": self.occurrences,
            "risk_level": self.risk_level,
            "details": self.details,
        }


@dataclass
class RedactionResult:
    """Result of data redaction."""

    redacted_output: str
    original_output: str
    redaction_count: int
    redacted_types: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Colang compatibility."""
        return {
            "redacted_output": self.redacted_output,
            "original_output": self.original_output,
            "redaction_count": self.redaction_count,
            "redacted_types": self.redacted_types,
        }


# =============================================================================
# INJECTION SCANNER
# =============================================================================

class InjectionScanner:
    """Scans input for potential injection attacks.

    Uses multi-layered detection:
    1. Regex patterns for known injection types
    2. High-confidence patterns for definite threats
    3. Risk scoring based on pattern matches

    Usage:
        scanner = InjectionScanner()
        result = scanner.scan("user input here")
        if result.injection_detected:
            handle_threat(result)
    """

    def __init__(
        self,
        additional_patterns: Optional[List[Tuple[str, Pattern]]] = None,
        sensitivity: str = "normal",  # "low", "normal", "high", "paranoid"
    ):
        """Initialize the scanner.

        Args:
            additional_patterns: Extra patterns to check.
            sensitivity: Detection sensitivity level.
        """
        self.patterns = list(INJECTION_PATTERNS)
        self.high_confidence_patterns = list(HIGH_CONFIDENCE_INJECTION_PATTERNS)

        if additional_patterns:
            self.patterns.extend(additional_patterns)

        self.sensitivity = sensitivity

        # Sensitivity thresholds
        self._thresholds = {
            "low": {"pattern_score": 0.3, "high_conf_score": 0.5},
            "normal": {"pattern_score": 0.2, "high_conf_score": 0.4},
            "high": {"pattern_score": 0.1, "high_conf_score": 0.3},
            "paranoid": {"pattern_score": 0.05, "high_conf_score": 0.2},
        }

    def scan(self, input_text: str) -> InjectionScanResult:
        """Scan input text for injection attempts.

        Args:
            input_text: Text to scan.

        Returns:
            InjectionScanResult with detection details.
        """
        if not input_text:
            return InjectionScanResult(
                injection_detected=False,
                input_safe=True,
                detected_types=[],
                high_confidence=False,
                risk_score=0.0,
                details={"message": "Empty input"},
            )

        detected_types: List[str] = []
        matches: Dict[str, List[str]] = {}

        # Check standard patterns
        for name, pattern in self.patterns:
            found = pattern.findall(input_text)
            if found:
                detected_types.append(name)
                matches[name] = [str(m) if isinstance(m, str) else str(m[0]) for m in found[:3]]

        # Check high-confidence patterns
        high_confidence = False
        for name, pattern in self.high_confidence_patterns:
            if pattern.search(input_text):
                high_confidence = True
                if name not in detected_types:
                    detected_types.append(name)
                matches[name] = ["high_confidence_match"]

        # Calculate risk score
        risk_score = self._calculate_risk_score(detected_types, high_confidence)

        # Determine if injection is detected based on sensitivity
        threshold = self._thresholds.get(self.sensitivity, self._thresholds["normal"])
        injection_detected = (
            high_confidence or
            risk_score >= threshold["pattern_score"] or
            len(detected_types) >= 2
        )

        return InjectionScanResult(
            injection_detected=injection_detected,
            input_safe=not injection_detected,
            detected_types=detected_types,
            high_confidence=high_confidence,
            risk_score=risk_score,
            details={
                "matches": matches,
                "sensitivity": self.sensitivity,
                "threshold_used": threshold["pattern_score"],
            },
        )

    def _calculate_risk_score(
        self,
        detected_types: List[str],
        high_confidence: bool,
    ) -> float:
        """Calculate risk score based on detections.

        Args:
            detected_types: Types of patterns detected.
            high_confidence: Whether high-confidence patterns matched.

        Returns:
            Risk score between 0.0 and 1.0.
        """
        if not detected_types:
            return 0.0

        # Base score from number of pattern types
        base_score = min(len(detected_types) * 0.15, 0.6)

        # Bonus for high-confidence
        if high_confidence:
            base_score = max(base_score, 0.7)

        # Type-specific weights
        type_weights = {
            "prompt_injection": 0.3,
            "definite_prompt_injection": 0.5,
            "sql_injection": 0.25,
            "command_injection": 0.3,
            "xss": 0.2,
            "path_traversal": 0.15,
            "ldap_injection": 0.15,
        }

        for dtype in detected_types:
            base_score += type_weights.get(dtype, 0.1)

        return min(base_score, 1.0)


# =============================================================================
# SENSITIVE DATA SCANNER
# =============================================================================

class SensitiveDataScanner:
    """Scans output for sensitive data.

    Detects:
    - API keys and tokens
    - Passwords and secrets
    - PII (email, phone, SSN, credit cards)
    - Internal IP addresses
    - Connection strings

    Usage:
        scanner = SensitiveDataScanner()
        result = scanner.scan("output text here")
        if result.sensitive_data_found:
            redact_or_block(result)
    """

    def __init__(
        self,
        additional_patterns: Optional[Dict[str, Pattern]] = None,
        ignore_types: Optional[Set[str]] = None,
    ):
        """Initialize the scanner.

        Args:
            additional_patterns: Extra patterns to check.
            ignore_types: Pattern types to ignore.
        """
        self.patterns = dict(SENSITIVE_DATA_PATTERNS)

        if additional_patterns:
            self.patterns.update(additional_patterns)

        self.ignore_types = ignore_types or set()

    def scan(self, output_text: str) -> SensitiveDataScanResult:
        """Scan output text for sensitive data.

        Args:
            output_text: Text to scan.

        Returns:
            SensitiveDataScanResult with detection details.
        """
        if not output_text:
            return SensitiveDataScanResult(
                sensitive_data_found=False,
                sensitive_types=[],
                occurrences={},
                risk_level="low",
                details={"message": "Empty output"},
            )

        sensitive_types: List[str] = []
        occurrences: Dict[str, int] = {}
        sample_matches: Dict[str, List[str]] = {}

        for name, pattern in self.patterns.items():
            if name in self.ignore_types:
                continue

            found = pattern.findall(output_text)
            if found:
                sensitive_types.append(name)
                occurrences[name] = len(found)
                # Store masked samples (first 3 chars only)
                sample_matches[name] = [
                    self._mask_value(str(m) if isinstance(m, str) else str(m[0]))
                    for m in found[:3]
                ]

        # Determine risk level
        risk_level = self._calculate_risk_level(sensitive_types, occurrences)

        return SensitiveDataScanResult(
            sensitive_data_found=len(sensitive_types) > 0,
            sensitive_types=sensitive_types,
            occurrences=occurrences,
            risk_level=risk_level,
            details={
                "sample_matches": sample_matches,
                "total_occurrences": sum(occurrences.values()),
            },
        )

    def _mask_value(self, value: str, visible_chars: int = 3) -> str:
        """Mask a sensitive value for logging.

        Args:
            value: Value to mask.
            visible_chars: Number of visible characters at start.

        Returns:
            Masked value.
        """
        if len(value) <= visible_chars:
            return "*" * len(value)
        return value[:visible_chars] + "*" * (len(value) - visible_chars)

    def _calculate_risk_level(
        self,
        sensitive_types: List[str],
        occurrences: Dict[str, int],
    ) -> str:
        """Calculate risk level based on findings.

        Args:
            sensitive_types: Types of sensitive data found.
            occurrences: Count per type.

        Returns:
            Risk level string.
        """
        if not sensitive_types:
            return "low"

        # Critical types
        critical_types = {
            "password", "api_key", "aws_access_key", "aws_secret_key",
            "bearer_token", "jwt_token", "generic_secret",
        }

        # High risk types
        high_types = {
            "connection_string", "ssn", "credit_card",
        }

        # Medium risk types
        medium_types = {
            "email", "phone_number", "internal_ip",
        }

        # Check for critical
        if any(t in critical_types for t in sensitive_types):
            return "critical"

        # Check for high
        if any(t in high_types for t in sensitive_types):
            return "high"

        # Check for medium
        if any(t in medium_types for t in sensitive_types):
            return "medium"

        # Multiple low-risk findings
        total = sum(occurrences.values())
        if total > 5:
            return "medium"

        return "low"


# =============================================================================
# DATA REDACTOR
# =============================================================================

class DataRedactor:
    """Redacts sensitive data from output.

    Uses pattern matching to replace sensitive data with redaction markers.

    Usage:
        redactor = DataRedactor()
        result = redactor.redact("text with password=secret123")
        print(result.redacted_output)  # "text with password=[REDACTED]"
    """

    def __init__(
        self,
        additional_patterns: Optional[List[Tuple[str, Pattern, str]]] = None,
    ):
        """Initialize the redactor.

        Args:
            additional_patterns: Extra (name, pattern, replacement) tuples.
        """
        self.patterns = list(CRITICAL_REDACTION_PATTERNS)

        if additional_patterns:
            self.patterns.extend(additional_patterns)

    def redact(self, output_text: str) -> RedactionResult:
        """Redact sensitive data from output.

        Args:
            output_text: Text to redact.

        Returns:
            RedactionResult with redacted text and metadata.
        """
        if not output_text:
            return RedactionResult(
                redacted_output="",
                original_output="",
                redaction_count=0,
                redacted_types=[],
            )

        redacted = output_text
        redaction_count = 0
        redacted_types: List[str] = []

        for name, pattern, replacement in self.patterns:
            new_text, count = pattern.subn(replacement, redacted)
            if count > 0:
                redacted = new_text
                redaction_count += count
                if name not in redacted_types:
                    redacted_types.append(name)

        # Additional: redact long base64-like strings
        long_token_pattern = re.compile(r"\b[a-zA-Z0-9_-]{40,}\b")
        new_text, count = long_token_pattern.subn("[REDACTED_TOKEN]", redacted)
        if count > 0:
            redacted = new_text
            redaction_count += count
            if "long_token" not in redacted_types:
                redacted_types.append("long_token")

        return RedactionResult(
            redacted_output=redacted,
            original_output=output_text,
            redaction_count=redaction_count,
            redacted_types=redacted_types,
        )


# =============================================================================
# FALLBACK ACTION IMPLEMENTATIONS
# =============================================================================

# Singleton instances
_injection_scanner: Optional[InjectionScanner] = None
_sensitive_scanner: Optional[SensitiveDataScanner] = None
_redactor: Optional[DataRedactor] = None


def _get_injection_scanner() -> InjectionScanner:
    """Get singleton injection scanner."""
    global _injection_scanner
    if _injection_scanner is None:
        _injection_scanner = InjectionScanner(sensitivity="normal")
    return _injection_scanner


def _get_sensitive_scanner() -> SensitiveDataScanner:
    """Get singleton sensitive data scanner."""
    global _sensitive_scanner
    if _sensitive_scanner is None:
        _sensitive_scanner = SensitiveDataScanner()
    return _sensitive_scanner


def _get_redactor() -> DataRedactor:
    """Get singleton data redactor."""
    global _redactor
    if _redactor is None:
        _redactor = DataRedactor()
    return _redactor


async def scan_for_injection_fallback(input_text: str = "") -> Dict[str, Any]:
    """Fallback implementation of scan_for_injection.

    Scans input for potential injection attacks using regex patterns.

    Args:
        input_text: Text to scan for injection attempts.

    Returns:
        Dictionary with:
        - injection_detected: bool
        - input_safe: bool
        - detected_types: List of injection types found
        - risk_score: Float between 0.0 and 1.0
    """
    scanner = _get_injection_scanner()
    result = scanner.scan(input_text)

    logger.debug(
        f"Injection scan complete: detected={result.injection_detected}, "
        f"types={result.detected_types}, score={result.risk_score:.2f}"
    )

    return result.to_dict()


async def scan_for_sensitive_data_fallback(output: str = "") -> Dict[str, Any]:
    """Fallback implementation of scan_for_sensitive_data.

    Scans output for sensitive data like PII, credentials, and secrets.

    Args:
        output: Text to scan for sensitive data.

    Returns:
        Dictionary with:
        - sensitive_data_found: bool
        - sensitive_types: List of sensitive data types found
        - risk_level: "low", "medium", "high", or "critical"
    """
    scanner = _get_sensitive_scanner()
    result = scanner.scan(output)

    if result.sensitive_data_found:
        logger.warning(
            f"Sensitive data detected: types={result.sensitive_types}, "
            f"risk_level={result.risk_level}"
        )

    return result.to_dict()


async def redact_sensitive_data_fallback(output: str = "") -> Dict[str, Any]:
    """Fallback implementation of redact_sensitive_data.

    Redacts sensitive data from output using pattern matching.

    Args:
        output: Text to redact.

    Returns:
        Dictionary with:
        - redacted_output: Redacted text
        - original_output: Original text
        - redaction_count: Number of redactions made
        - redacted_types: Types of data redacted
    """
    redactor = _get_redactor()
    result = redactor.redact(output)

    if result.redaction_count > 0:
        logger.info(
            f"Redacted {result.redaction_count} sensitive items: "
            f"types={result.redacted_types}"
        )

    return result.to_dict()


async def validate_permission_fallback(
    action: str = "",
    resource: str = "",
    agent_id: str = "worker",
) -> Dict[str, Any]:
    """Fallback implementation of validate_permission.

    Uses TrustBoundaryEnforcer for permission validation.

    Args:
        action: Action being performed.
        resource: Resource being accessed.
        agent_id: Agent requesting permission.

    Returns:
        Dictionary with:
        - permission_granted: bool
        - action: Action checked
        - resource: Resource checked
        - reason: Reason for decision
    """
    try:
        from pipeline.langgraph.trust_boundaries import check_access

        result = check_access(agent_id, resource, action)

        if not result.allowed:
            logger.warning(
                f"Permission denied: agent={agent_id}, action={action}, "
                f"resource={resource}, reason={result.reason}"
            )

        return {
            "permission_granted": result.allowed,
            "action": action,
            "resource": resource,
            "agent_id": agent_id,
            "reason": result.reason,
        }

    except ImportError as e:
        logger.warning(f"TrustBoundaryEnforcer not available: {e}")
        # Default to deny in strict mode
        return {
            "permission_granted": False,
            "action": action,
            "resource": resource,
            "agent_id": agent_id,
            "reason": "Trust boundary module not available",
        }

    except Exception as e:
        logger.error(f"Permission validation failed: {e}")
        return {
            "permission_granted": False,
            "action": action,
            "resource": resource,
            "error": str(e),
        }


async def check_circuit_breakers_fallback() -> Dict[str, Any]:
    """Fallback implementation of check_circuit_breakers.

    Uses CircuitBreakerRegistry for resilience monitoring.

    Returns:
        Dictionary with:
        - circuit_open: First open circuit name (or None)
        - open_circuits: List of open circuits
        - half_open_circuits: List of half-open circuits
        - all_closed: bool
    """
    try:
        from pipeline.langgraph.stack_guardrails import (
            CircuitBreakerRegistry,
            CircuitState,
        )

        registry = CircuitBreakerRegistry()
        open_circuits: List[str] = []
        half_open_circuits: List[str] = []

        for name, breaker in registry.breakers.items():
            if breaker.state == CircuitState.OPEN:
                open_circuits.append(name)
            elif breaker.state == CircuitState.HALF_OPEN:
                half_open_circuits.append(name)

        if open_circuits:
            logger.warning(f"Open circuits detected: {open_circuits}")

        return {
            "circuit_open": open_circuits[0] if open_circuits else None,
            "open_circuits": open_circuits,
            "half_open_circuits": half_open_circuits,
            "all_closed": len(open_circuits) == 0,
        }

    except ImportError:
        logger.debug("CircuitBreakerRegistry not available")
        return {
            "circuit_open": None,
            "open_circuits": [],
            "half_open_circuits": [],
            "all_closed": True,
        }

    except Exception as e:
        logger.error(f"Circuit breaker check failed: {e}")
        return {
            "circuit_open": None,
            "all_closed": True,
            "error": str(e),
        }


async def validate_stack_health_fallback() -> Dict[str, Any]:
    """Fallback implementation of validate_stack_health.

    Uses StackInjector for health monitoring.

    Returns:
        Dictionary with:
        - stacks_valid: bool (all healthy)
        - missing_stacks: List of unhealthy stacks
        - unhealthy_stacks: List of unhealthy stacks
        - healthy_count: Number of healthy stacks
        - total_count: Total number of stacks
    """
    try:
        from pipeline.langgraph.stack_injection import get_stack_injector

        injector = get_stack_injector()
        health = injector.check_health()

        unhealthy = [
            name for name, data in health.items()
            if not data.get("healthy", False)
        ]

        if unhealthy:
            logger.warning(f"Unhealthy stacks detected: {unhealthy}")

        return {
            "stacks_valid": len(unhealthy) == 0,
            "missing_stacks": unhealthy,
            "unhealthy_stacks": unhealthy,
            "healthy_count": len(health) - len(unhealthy),
            "total_count": len(health),
        }

    except ImportError:
        logger.debug("StackInjector not available")
        return {
            "stacks_valid": True,
            "missing_stacks": [],
            "unhealthy_stacks": [],
            "healthy_count": 0,
            "total_count": 0,
            "error": "StackInjector not available",
        }

    except Exception as e:
        logger.error(f"Stack health check failed: {e}")
        return {
            "stacks_valid": False,
            "error": str(e),
        }


# =============================================================================
# FALLBACK REGISTRY
# =============================================================================

# Mapping of action names to their fallback implementations
SECURITY_FALLBACKS: Dict[str, Any] = {
    "scan_for_injection": scan_for_injection_fallback,
    "scan_for_sensitive_data": scan_for_sensitive_data_fallback,
    "redact_sensitive_data": redact_sensitive_data_fallback,
    "validate_permission": validate_permission_fallback,
    "check_circuit_breakers": check_circuit_breakers_fallback,
    "validate_stack_health": validate_stack_health_fallback,
}


def get_fallback_action(action_name: str) -> Optional[Any]:
    """Get the fallback implementation for an action.

    Args:
        action_name: Name of the action.

    Returns:
        Fallback function or None if not available.
    """
    return SECURITY_FALLBACKS.get(action_name)


def has_fallback(action_name: str) -> bool:
    """Check if a fallback exists for an action.

    Args:
        action_name: Name of the action.

    Returns:
        True if fallback exists.
    """
    return action_name in SECURITY_FALLBACKS


def list_fallback_actions() -> List[str]:
    """List all actions with fallbacks.

    Returns:
        List of action names.
    """
    return list(SECURITY_FALLBACKS.keys())


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Data classes
    "InjectionScanResult",
    "SensitiveDataScanResult",
    "RedactionResult",
    # Scanner classes
    "InjectionScanner",
    "SensitiveDataScanner",
    "DataRedactor",
    # Fallback functions
    "scan_for_injection_fallback",
    "scan_for_sensitive_data_fallback",
    "redact_sensitive_data_fallback",
    "validate_permission_fallback",
    "check_circuit_breakers_fallback",
    "validate_stack_health_fallback",
    # Registry
    "SECURITY_FALLBACKS",
    "get_fallback_action",
    "has_fallback",
    "list_fallback_actions",
    # Constants
    "INJECTION_PATTERNS",
    "SENSITIVE_DATA_PATTERNS",
]
