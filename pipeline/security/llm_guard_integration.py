"""LLM Guard Integration Module.

This module provides comprehensive integration with LLM Guard for security scanning,
including input sanitization, output validation, PII detection, prompt injection
detection, toxicity filtering, and integration with pipeline gates.

Architecture:
    SecurityOrchestrator
        |
        +-- LLMGuardIntegration (Main coordinator)
        |       |
        |       +-- InputSanitizer
        |       +-- OutputValidator
        |       +-- PIIDetector
        |       +-- ToxicityFilter
        |       +-- PromptInjectionDetector
        |
        +-- Langfuse Integration (Observability)

Features:
1. Defense in depth - multiple security layers
2. Graceful degradation when LLM Guard service unavailable
3. Async operations for all security checks
4. Comprehensive logging to Langfuse
5. Integration with pipeline gates

Author: Pipeline Autonomo Team
Version: 1.0.0 (2026-01-16)
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
import time
import unicodedata  # BLUE-016 FIX: Moved from function to module level
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime, timezone  # BLUE-018 FIX: Added timezone import
from functools import wraps

logger = logging.getLogger(__name__)

# Environment variable to allow fallback (dev mode only)
PIPELINE_ALLOW_FALLBACK = os.getenv("PIPELINE_ALLOW_FALLBACK", "false").lower() == "true"

# RED-009 FIX: Configurable timeout for fallback operations
# Default: 5.0 seconds. Can be overridden via environment variable.
# This allows operators to tune fallback timeout based on their infrastructure.
FALLBACK_TIMEOUT_SECONDS = float(os.getenv("LLM_GUARD_FALLBACK_TIMEOUT", "5.0"))

# RED-009: Validate timeout is reasonable (between 0.1 and 60 seconds)
if FALLBACK_TIMEOUT_SECONDS < 0.1:
    logger.warning(f"RED-009: FALLBACK_TIMEOUT_SECONDS={FALLBACK_TIMEOUT_SECONDS} too low, using 0.1")
    FALLBACK_TIMEOUT_SECONDS = 0.1
elif FALLBACK_TIMEOUT_SECONDS > 60.0:
    logger.warning(f"RED-009: FALLBACK_TIMEOUT_SECONDS={FALLBACK_TIMEOUT_SECONDS} too high, using 60.0")
    FALLBACK_TIMEOUT_SECONDS = 60.0

# Health check configuration
LLM_GUARD_HEALTH_CHECK_TIMEOUT = float(os.getenv("LLM_GUARD_HEALTH_CHECK_TIMEOUT", "5.0"))
LLM_GUARD_SERVICE_URL = os.getenv("LLM_GUARD_SERVICE_URL", "http://localhost:50053")
LLM_GUARD_DOCKER_CONTAINER = os.getenv("LLM_GUARD_DOCKER_CONTAINER", "llm-guard")

# =============================================================================
# AVAILABILITY CHECK
# =============================================================================
#
# DEPRECATION NOTICE (AUDITOR-2):
# LLM Guard service should be deployed and available in production.
# The fallback pattern-based mode is for development only and provides
# reduced security coverage. To ensure full security:
#   1. Deploy LLM Guard service (docker-compose up llm-guard)
#   2. Set LLM_GUARD_SERVICE_URL environment variable
#   3. Do NOT set PIPELINE_ALLOW_FALLBACK in production
#
# See: docs/pipeline/STACK_SETUP_GUIDE.md for deployment instructions.
# =============================================================================

try:
    from pipeline.services.llm_guard_client import (
        LLMGuardClient,
        ScanResult,
        ScannerResult,
        PromptInjectionResult,
        ToxicityResult,
        PIIResult,
        RiskLevel,
        get_llm_guard_client,
        is_llm_guard_available,
    )
    LLM_GUARD_CLIENT_AVAILABLE = True
except ImportError:
    LLM_GUARD_CLIENT_AVAILABLE = False
    if not PIPELINE_ALLOW_FALLBACK:
        logger.warning(
            "LLM Guard client not available and PIPELINE_ALLOW_FALLBACK is not set. "
            "Security scanning will operate in DEGRADED MODE with pattern-based fallbacks. "
            "Set PIPELINE_ALLOW_FALLBACK=true for dev mode."
        )
    else:
        logger.info("LLM Guard client not available - FALLBACK MODE (dev only)")

LLM_GUARD_INTEGRATION_AVAILABLE = LLM_GUARD_CLIENT_AVAILABLE

# BLUE-019 FIX: Import shared PII patterns from nemo_enhanced
# This ensures consistency across security modules
try:
    from pipeline.security.nemo_enhanced import PII_PATTERNS
except ImportError:
    # Fallback if nemo_enhanced is not available (shouldn't happen in normal operation)
    PII_PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b",
        "ssn": r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b",
        "credit_card": r"\b(?:\d{4}[-.\s]?){3}\d{4}\b",
        "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    }

# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================


class SecurityLevel(str, Enum):
    """Security check strictness levels."""
    MINIMAL = "minimal"  # Basic checks only
    STANDARD = "standard"  # Default security level
    STRICT = "strict"  # All checks enabled
    PARANOID = "paranoid"  # Maximum security, may have false positives


class SanitizationStrategy(str, Enum):
    """Strategies for sanitizing content."""
    BLOCK = "block"  # Block unsafe content entirely
    REDACT = "redact"  # Redact sensitive parts
    REPLACE = "replace"  # Replace with safe alternatives
    WARN = "warn"  # Allow but warn


class SecurityCheckType(str, Enum):
    """Types of security checks."""
    PROMPT_INJECTION = "prompt_injection"
    TOXICITY = "toxicity"
    PII = "pii"
    SECRETS = "secrets"
    JAILBREAK = "jailbreak"
    CONTENT_FILTER = "content_filter"
    OUTPUT_RELEVANCE = "output_relevance"


# Risk thresholds for each security level
RISK_THRESHOLDS = {
    SecurityLevel.MINIMAL: 0.8,
    SecurityLevel.STANDARD: 0.6,
    SecurityLevel.STRICT: 0.4,
    SecurityLevel.PARANOID: 0.2,
}

# Default scanners for each level
DEFAULT_SCANNERS = {
    SecurityLevel.MINIMAL: ["PromptInjection"],
    SecurityLevel.STANDARD: ["PromptInjection", "Toxicity", "Secrets"],
    SecurityLevel.STRICT: ["PromptInjection", "Toxicity", "Secrets", "Anonymize", "BanSubstrings"],
    SecurityLevel.PARANOID: ["PromptInjection", "Toxicity", "Secrets", "Anonymize", "BanSubstrings", "TokenLimit", "InvisibleText"],
}


# =============================================================================
# RED-010 FIX: EXCEPTION SANITIZATION FOR LOGGING
# =============================================================================

def _sanitize_exception_for_logging(exc: Exception, max_length: int = 200) -> str:
    """RED-010 FIX: Sanitize exception messages before logging.

    This prevents sensitive text (user prompts, PII, secrets) from being
    leaked into log files via exception messages. Attackers could craft
    inputs that cause exceptions with the input text in the message.

    Args:
        exc: The exception to sanitize
        max_length: Maximum length of the sanitized message

    Returns:
        Sanitized exception message safe for logging
    """
    exc_type = type(exc).__name__
    exc_msg = str(exc)

    # Remove potential sensitive content patterns
    # 1. Truncate long messages (may contain user content)
    if len(exc_msg) > max_length:
        exc_msg = exc_msg[:max_length] + "...[TRUNCATED]"

    # 2. Redact anything that looks like a prompt or user content
    # These patterns indicate user content may be in the exception
    sensitive_indicators = [
        (r'(?i)(prompt|input|text|content|message)[:\s]*["\'].*?["\']', '[REDACTED_CONTENT]'),
        (r'(?i)(api[_-]?key|secret|password|token)[:\s]*\S+', '[REDACTED_CREDENTIAL]'),
        (r'[\'"]{3}.*?[\'"]{3}', '[REDACTED_BLOCK]'),  # Triple-quoted strings
        (r'data:.*?base64,\S+', '[REDACTED_DATA_URI]'),  # Data URIs
    ]

    for pattern, replacement in sensitive_indicators:
        exc_msg = re.sub(pattern, replacement, exc_msg, flags=re.DOTALL)

    return f"{exc_type}: {exc_msg}"


# =============================================================================
# RED-011 FIX: BASIC RATE LIMITING STRUCTURE
# =============================================================================

class RateLimiter:
    """RED-011 FIX: Basic rate limiter for security operations.

    This provides a simple in-memory rate limiter that can be expanded later.
    Prevents DoS attacks against security scanning endpoints.

    Features:
    - Token bucket algorithm
    - Per-client rate limiting (by client_id or IP)
    - Configurable via environment variables
    - Thread-safe

    Usage:
        limiter = RateLimiter()
        if not limiter.allow("client_123"):
            raise RateLimitExceeded("Too many requests")
    """

    # Configuration via environment
    DEFAULT_RATE_LIMIT = int(os.getenv("LLM_GUARD_RATE_LIMIT", "100"))  # requests per window
    DEFAULT_WINDOW_SECONDS = int(os.getenv("LLM_GUARD_RATE_WINDOW", "60"))  # window size

    def __init__(
        self,
        rate_limit: Optional[int] = None,
        window_seconds: Optional[int] = None,
    ):
        """Initialize rate limiter.

        Args:
            rate_limit: Max requests per window (default from env or 100)
            window_seconds: Window size in seconds (default from env or 60)
        """
        self.rate_limit = rate_limit or self.DEFAULT_RATE_LIMIT
        self.window_seconds = window_seconds or self.DEFAULT_WINDOW_SECONDS
        self._buckets: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

        logger.debug(
            f"RED-011: RateLimiter initialized with limit={self.rate_limit}/window={self.window_seconds}s"
        )

    def allow(self, client_id: str) -> bool:
        """Check if request is allowed for client.

        Args:
            client_id: Unique identifier for the client (IP, user_id, etc.)

        Returns:
            True if request is allowed, False if rate limited
        """
        current_time = time.time()
        window_start = current_time - self.window_seconds

        with self._lock:
            # Get or create bucket for client
            if client_id not in self._buckets:
                self._buckets[client_id] = []

            bucket = self._buckets[client_id]

            # Remove old entries outside the window
            bucket[:] = [t for t in bucket if t > window_start]

            # Check if we're at the limit
            if len(bucket) >= self.rate_limit:
                logger.warning(
                    f"RED-011: Rate limit exceeded for client {client_id[:8]}...: "
                    f"{len(bucket)}/{self.rate_limit} requests in {self.window_seconds}s"
                )
                return False

            # Add current request
            bucket.append(current_time)
            return True

    def get_remaining(self, client_id: str) -> int:
        """Get remaining requests in current window for client.

        Args:
            client_id: Unique identifier for the client

        Returns:
            Number of remaining requests allowed
        """
        current_time = time.time()
        window_start = current_time - self.window_seconds

        with self._lock:
            bucket = self._buckets.get(client_id, [])
            current_count = len([t for t in bucket if t > window_start])
            return max(0, self.rate_limit - current_count)

    def reset(self, client_id: Optional[str] = None) -> None:
        """Reset rate limiter for a client or all clients.

        Args:
            client_id: If provided, reset only this client. Otherwise reset all.
        """
        with self._lock:
            if client_id:
                self._buckets.pop(client_id, None)
            else:
                self._buckets.clear()
        logger.debug(f"RED-011: Rate limiter reset for {'all clients' if not client_id else client_id}")

    def cleanup_stale(self) -> int:
        """Remove stale entries to prevent memory bloat.

        Should be called periodically by a background task.

        Returns:
            Number of stale client entries removed
        """
        current_time = time.time()
        window_start = current_time - self.window_seconds
        removed = 0

        with self._lock:
            stale_clients = [
                client_id
                for client_id, bucket in self._buckets.items()
                if not any(t > window_start for t in bucket)
            ]
            for client_id in stale_clients:
                del self._buckets[client_id]
                removed += 1

        if removed > 0:
            logger.debug(f"RED-011: Cleaned up {removed} stale rate limiter entries")

        return removed


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""

    def __init__(self, message: str = "Rate limit exceeded", retry_after: Optional[int] = None):
        self.message = message
        self.retry_after = retry_after
        super().__init__(message)


# Global rate limiter instance
_global_rate_limiter: Optional[RateLimiter] = None
_rate_limiter_lock = threading.Lock()


def get_rate_limiter() -> RateLimiter:
    """Get or create global rate limiter instance.

    Returns:
        Singleton RateLimiter instance
    """
    global _global_rate_limiter
    if _global_rate_limiter is None:
        with _rate_limiter_lock:
            if _global_rate_limiter is None:
                _global_rate_limiter = RateLimiter()
    return _global_rate_limiter


# =============================================================================
# UNICODE NORMALIZATION FOR BYPASS PREVENTION (AUDIT FIX HIGH-001)
# =============================================================================

# Zero-width characters that can be used to bypass pattern matching
# AUDIT FIX RED-002/BLUE-001: Synced with nemo_enhanced.py (33 chars total)
ZERO_WIDTH_CHARS: set[str] = {
    '\u200b',  # Zero Width Space
    '\u200c',  # Zero Width Non-Joiner
    '\u200d',  # Zero Width Joiner
    '\u200e',  # Left-to-Right Mark
    '\u200f',  # Right-to-Left Mark
    '\u2060',  # Word Joiner
    '\u2061',  # Function Application
    '\u2062',  # Invisible Times
    '\u2063',  # Invisible Separator
    '\u2064',  # Invisible Plus
    '\ufeff',  # Zero Width No-Break Space (BOM)
    '\u00ad',  # Soft Hyphen
    '\u034f',  # Combining Grapheme Joiner
    '\u061c',  # Arabic Letter Mark
    '\u115f',  # Hangul Choseong Filler
    '\u1160',  # Hangul Jungseong Filler
    '\u17b4',  # Khmer Vowel Inherent Aq
    '\u17b5',  # Khmer Vowel Inherent Aa
    '\u180b',  # Mongolian Free Variation Selector One
    '\u180c',  # Mongolian Free Variation Selector Two
    '\u180d',  # Mongolian Free Variation Selector Three
    '\u180e',  # Mongolian Vowel Separator
    '\u3164',  # Hangul Filler
    '\uffa0',  # Halfwidth Hangul Filler
    # Bidi control characters (CRITICAL for RTL Override attacks) - RED-002
    '\u202a',  # Left-to-Right Embedding
    '\u202b',  # Right-to-Left Embedding
    '\u202c',  # Pop Directional Formatting
    '\u202d',  # Left-to-Right Override
    '\u202e',  # Right-to-Left Override (MOST DANGEROUS)
    '\u2066',  # Left-to-Right Isolate
    '\u2067',  # Right-to-Left Isolate
    '\u2068',  # First Strong Isolate
    '\u2069',  # Pop Directional Isolate
}

# Common homoglyph mappings (Cyrillic, Greek, etc. to ASCII)
# AUDIT FIX RED-003: Added critical Cyrillic/Greek confusables - synced with nemo_enhanced.py
#
# BLUE-008 FIX: Descriptive comments explaining the homoglyph attack vector
#
# SECURITY PURPOSE:
# Homoglyph attacks use visually similar characters from different Unicode scripts
# to bypass text-based security filters. For example:
# - "admin" spelled with Cyrillic 'а' (\u0430) looks identical but bypasses ASCII filters
# - "secret" with Greek 'ε' (\u03b5) appears the same but has different codepoints
#
# ATTACK VECTORS MITIGATED:
# 1. Keyword filter bypass: Using "рrompt" (Cyrillic 'р') to bypass "prompt" detection
# 2. Phishing URLs: Using "gοοgle.com" (Greek omicrons) to look like "google.com"
# 3. Identity spoofing: Using Cyrillic lookalikes in usernames/identifiers
# 4. Prompt injection: Bypassing blocklists with visually identical characters
#
# MAINTENANCE NOTE:
# This map should be kept in sync with nemo_enhanced.py HOMOGLYPH_MAP.
# When adding new entries, document:
# - The Unicode codepoint
# - The script it belongs to (Cyrillic, Greek, etc.)
# - The ASCII character it mimics
#
HOMOGLYPH_MAP: dict[str, str] = {
    # === CYRILLIC LOWERCASE (U+0430-U+045F) ===
    # Cyrillic has many characters visually identical to Latin
    '\u0430': 'a',  # Cyrillic Small A - looks identical to Latin 'a'
    '\u0435': 'e',  # Cyrillic Small Ie - looks identical to Latin 'e'
    '\u043e': 'o',  # Cyrillic Small O - looks identical to Latin 'o'
    '\u0440': 'p',  # Cyrillic Small Er - looks identical to Latin 'p'
    '\u0441': 'c',  # Cyrillic Small Es - looks identical to Latin 'c'
    '\u0443': 'y',  # Cyrillic Small U - looks similar to Latin 'y'
    '\u0445': 'x',  # Cyrillic Small Ha - looks identical to Latin 'x'
    '\u0456': 'i',  # Cyrillic Small Byelorussian-Ukrainian I - RED-003 CRITICAL
    '\u0432': 'b',  # Cyrillic Small Ve - looks similar to Latin 'b'
    '\u043d': 'h',  # Cyrillic Small En - looks similar to Latin 'h'
    '\u0455': 's',  # Cyrillic Small Dze - looks identical to Latin 's'
    '\u0458': 'j',  # Cyrillic Small Je - looks identical to Latin 'j'

    # === CYRILLIC UPPERCASE (U+0410-U+042F) ===
    '\u0410': 'A',  # Cyrillic Capital A - identical to Latin 'A'
    '\u0412': 'B',  # Cyrillic Capital Ve - identical to Latin 'B'
    '\u0415': 'E',  # Cyrillic Capital Ie - identical to Latin 'E'
    '\u041a': 'K',  # Cyrillic Capital Ka - identical to Latin 'K'
    '\u041c': 'M',  # Cyrillic Capital Em - identical to Latin 'M'
    '\u041d': 'H',  # Cyrillic Capital En - identical to Latin 'H'
    '\u041e': 'O',  # Cyrillic Capital O - identical to Latin 'O'
    '\u0420': 'P',  # Cyrillic Capital Er - identical to Latin 'P'
    '\u0421': 'C',  # Cyrillic Capital Es - identical to Latin 'C'
    '\u0422': 'T',  # Cyrillic Capital Te - identical to Latin 'T'
    '\u0425': 'X',  # Cyrillic Capital Ha - identical to Latin 'X'
    '\u0406': 'I',  # Cyrillic Capital Byelorussian-Ukrainian I - RED-003 CRITICAL

    # === GREEK LOWERCASE (U+03B0-U+03FF) ===
    # Greek is another common source of confusable characters
    '\u03b1': 'a',  # Greek Small Alpha - similar to Latin 'a'
    '\u03b5': 'e',  # Greek Small Epsilon - similar to Latin 'e'
    '\u03bf': 'o',  # Greek Small Omicron - identical to Latin 'o' (phishing favorite!)
    '\u03c1': 'p',  # Greek Small Rho - similar to Latin 'p'
    '\u03b9': 'i',  # Greek Small Iota - RED-003 CRITICAL: used in i-based attacks
    '\u03c5': 'u',  # Greek Small Upsilon - similar to Latin 'u'

    # === GREEK UPPERCASE (U+0391-U+03A9) ===
    '\u0391': 'A',  # Greek Capital Alpha - identical to Latin 'A'
    '\u0392': 'B',  # Greek Capital Beta - identical to Latin 'B'
    '\u0395': 'E',  # Greek Capital Epsilon - identical to Latin 'E'
    '\u0397': 'H',  # Greek Capital Eta - identical to Latin 'H'
    '\u039a': 'K',  # Greek Capital Kappa - identical to Latin 'K'
    '\u039c': 'M',  # Greek Capital Mu - identical to Latin 'M'
    '\u039d': 'N',  # Greek Capital Nu - identical to Latin 'N'
    '\u039f': 'O',  # Greek Capital Omicron - identical to Latin 'O'
    '\u03a1': 'P',  # Greek Capital Rho - identical to Latin 'P'
    '\u03a4': 'T',  # Greek Capital Tau - identical to Latin 'T'
    '\u03a7': 'X',  # Greek Capital Chi - identical to Latin 'X'

    # === OTHER CONFUSABLE CHARACTERS ===
    # From various Unicode blocks that can be used for homoglyph attacks
    '\u0131': 'i',  # Latin Small Dotless I (Turkish) - used in case-folding attacks
    '\u0237': 'j',  # Latin Small Dotless J - similar to 'j'
    '\u1d00': 'a',  # Latin Letter Small Capital A (phonetic)
    '\u0261': 'g',  # Latin Small Script G (IPA)
    '\u0251': 'a',  # Latin Small Alpha (IPA)
    '\u0252': 'a',  # Latin Small Turned Alpha (IPA)
    '\u1e9a': 'a',  # Latin Small A with Right Half Ring

    # === FULLWIDTH FORMS (U+FF00-U+FFEF) ===
    # Fullwidth forms look identical but have different codepoints
    '\uff21': 'A',  # Fullwidth Latin Capital A
    '\uff22': 'B',  # Fullwidth Latin Capital B
    '\uff41': 'a',  # Fullwidth Latin Small A
    '\uff42': 'b',  # Fullwidth Latin Small B
}


def normalize_text_for_security(text: str) -> str:
    """Normalize text to prevent Unicode-based bypass attacks.

    AUDIT FIX HIGH-001: This function must be called before ALL fallback
    pattern matching to prevent attackers from bypassing detection using
    zero-width characters or homoglyphs.

    Steps:
    1. Remove zero-width characters
    2. Replace homoglyphs with ASCII equivalents
    3. Apply NFKC normalization

    Args:
        text: Input text to normalize

    Returns:
        Normalized text safe for pattern matching
    """
    # BLUE-016 FIX: unicodedata is now imported at module level
    if not text:
        return ""

    # Step 1: Remove zero-width characters
    result = ''.join(c for c in text if c not in ZERO_WIDTH_CHARS)

    # Step 2: Replace homoglyphs with ASCII equivalents
    result = ''.join(HOMOGLYPH_MAP.get(c, c) for c in result)

    # Step 3: Apply NFKC normalization (compatibility decomposition + canonical composition)
    result = unicodedata.normalize('NFKC', result)

    return result


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class SanitizationResult:
    """Result of input sanitization."""
    is_safe: bool
    original_text: str
    sanitized_text: str
    issues_found: List[str] = field(default_factory=list)
    risk_score: float = 0.0
    strategy_applied: SanitizationStrategy = SanitizationStrategy.WARN
    checks_performed: List[SecurityCheckType] = field(default_factory=list)
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))  # BLUE-018 FIX

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_safe": self.is_safe,
            "original_text": self.original_text[:100] + "..." if len(self.original_text) > 100 else self.original_text,
            "sanitized_text": self.sanitized_text[:100] + "..." if len(self.sanitized_text) > 100 else self.sanitized_text,
            "issues_found": self.issues_found,
            "risk_score": self.risk_score,
            "strategy_applied": self.strategy_applied.value,
            "checks_performed": [c.value for c in self.checks_performed],
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ValidationResult:
    """Result of output validation."""
    is_valid: bool
    output_text: str
    validated_text: str
    issues_found: List[str] = field(default_factory=list)
    risk_score: float = 0.0
    relevance_score: float = 1.0
    checks_performed: List[SecurityCheckType] = field(default_factory=list)
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))  # BLUE-018 FIX

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "output_text": self.output_text[:100] + "..." if len(self.output_text) > 100 else self.output_text,
            "validated_text": self.validated_text[:100] + "..." if len(self.validated_text) > 100 else self.validated_text,
            "issues_found": self.issues_found,
            "risk_score": self.risk_score,
            "relevance_score": self.relevance_score,
            "checks_performed": [c.value for c in self.checks_performed],
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PIIDetectionResult:
    """Result of PII detection."""
    has_pii: bool
    pii_types: List[str] = field(default_factory=list)
    pii_count: int = 0
    redacted_text: Optional[str] = None
    confidence: float = 0.0
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))  # BLUE-018 FIX

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_pii": self.has_pii,
            "pii_types": self.pii_types,
            "pii_count": self.pii_count,
            "redacted_text": self.redacted_text[:100] + "..." if self.redacted_text and len(self.redacted_text) > 100 else self.redacted_text,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ToxicityFilterResult:
    """Result of toxicity filtering."""
    is_toxic: bool
    toxicity_score: float = 0.0
    categories: Dict[str, float] = field(default_factory=dict)
    filtered_text: Optional[str] = None
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))  # BLUE-018 FIX

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_toxic": self.is_toxic,
            "toxicity_score": self.toxicity_score,
            "categories": self.categories,
            "filtered_text": self.filtered_text[:100] + "..." if self.filtered_text and len(self.filtered_text) > 100 else self.filtered_text,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PromptInjectionDetectionResult:
    """Result of prompt injection detection."""
    is_injection: bool
    injection_type: Optional[str] = None
    confidence: float = 0.0
    indicators: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))  # BLUE-018 FIX

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_injection": self.is_injection,
            "injection_type": self.injection_type,
            "confidence": self.confidence,
            "indicators": self.indicators,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SecurityCheckResult:
    """Comprehensive result from all security checks."""
    is_safe: bool
    overall_risk_score: float
    sanitization: Optional[SanitizationResult] = None
    validation: Optional[ValidationResult] = None
    pii_detection: Optional[PIIDetectionResult] = None
    toxicity_filter: Optional[ToxicityFilterResult] = None
    prompt_injection: Optional[PromptInjectionDetectionResult] = None
    blocked: bool = False
    block_reason: Optional[str] = None
    total_latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))  # BLUE-018 FIX

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_safe": self.is_safe,
            "overall_risk_score": self.overall_risk_score,
            "sanitization": self.sanitization.to_dict() if self.sanitization else None,
            "validation": self.validation.to_dict() if self.validation else None,
            "pii_detection": self.pii_detection.to_dict() if self.pii_detection else None,
            "toxicity_filter": self.toxicity_filter.to_dict() if self.toxicity_filter else None,
            "prompt_injection": self.prompt_injection.to_dict() if self.prompt_injection else None,
            "blocked": self.blocked,
            "block_reason": self.block_reason,
            "total_latency_ms": self.total_latency_ms,
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# INPUT SANITIZER
# =============================================================================


class InputSanitizer:
    """Sanitizes user inputs for security.

    Performs:
    1. Prompt injection detection
    2. PII detection and optional redaction
    3. Harmful content filtering
    4. Input normalization

    Usage:
        sanitizer = InputSanitizer()
        result = await sanitizer.sanitize("user input text")
        if not result.is_safe:
            print(f"Issues: {result.issues_found}")
    """

    def __init__(
        self,
        security_level: SecurityLevel = SecurityLevel.STANDARD,
        strategy: SanitizationStrategy = SanitizationStrategy.REDACT,
    ):
        self.security_level = security_level
        self.strategy = strategy
        self.risk_threshold = RISK_THRESHOLDS[security_level]

        # Fallback patterns when LLM Guard unavailable
        self._injection_patterns = [
            r"(?i)ignore\s+(previous|prior|all)\s+instructions",
            r"(?i)system\s*:\s*",
            r"(?i)you\s+are\s+now\s+",
            r"(?i)forget\s+everything",
            r"(?i)new\s+instructions",
        ]

        # BLUE-019 FIX: Use shared PII_PATTERNS constant instead of inline patterns
        self._pii_patterns = PII_PATTERNS

    async def sanitize(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> SanitizationResult:
        """Sanitize input text.

        Args:
            text: Input text to sanitize
            context: Optional context for scanning

        Returns:
            SanitizationResult with sanitization details
        """
        start_time = time.time()
        issues: List[str] = []
        checks_performed: List[SecurityCheckType] = []
        risk_scores: List[float] = []
        sanitized_text = text

        # RED-004 FIX: Explicit flag to track if LLM Guard completed successfully
        llm_guard_succeeded = False

        # Try LLM Guard service first
        if LLM_GUARD_CLIENT_AVAILABLE:
            try:
                client = get_llm_guard_client()
                if await client.is_ready():
                    scan_result = await client.scan_input(
                        text,
                        scanners=DEFAULT_SCANNERS.get(self.security_level),
                        sanitize=self.strategy == SanitizationStrategy.REDACT,
                    )

                    for scanner_result in scan_result.scanner_results:
                        if not scanner_result.is_valid:
                            issues.append(f"{scanner_result.scanner_name}: {scanner_result.details}")
                            risk_scores.append(scanner_result.risk_score)

                    if scan_result.sanitized_text:
                        sanitized_text = scan_result.sanitized_text

                    checks_performed.append(SecurityCheckType.CONTENT_FILTER)
                    # RED-004 FIX: Only mark success AFTER all LLM Guard logic completes
                    llm_guard_succeeded = True

            except Exception as e:
                logger.warning(f"LLM Guard scan failed, using fallback: {_sanitize_exception_for_logging(e)}")

        # RED-004 FIX: Run fallback for injection detection if LLM Guard did not complete
        if not llm_guard_succeeded:
            # AUDIT FIX HIGH-001: Normalize text before pattern matching
            normalized_text = normalize_text_for_security(text)
            # AUDIT FIX RED-002/BLUE-001: Use normalized text for redaction too
            sanitized_text = normalized_text

            # Check for injection patterns
            for pattern in self._injection_patterns:
                if re.search(pattern, normalized_text):
                    issues.append(f"Potential injection pattern detected: {pattern[:30]}...")
                    risk_scores.append(0.8)

            checks_performed.append(SecurityCheckType.PROMPT_INJECTION)
        else:
            # Even if LLM Guard succeeded, normalize for PII check
            normalized_text = normalize_text_for_security(text)

        # AUDIT FIX 2026-01-20: ALWAYS run PII detection as defense-in-depth layer
        # LLM Guard service may not have Anonymize scanner, so local patterns are critical
        # Check for PII - both detection and redaction use normalized text
        for pii_type, pattern in self._pii_patterns.items():
            if re.search(pattern, normalized_text):
                issues.append(f"PII detected: {pii_type}")
                risk_scores.append(0.6)
                # Redact on normalized text (AUDIT FIX RED-002/BLUE-001)
                if self.strategy == SanitizationStrategy.REDACT:
                    sanitized_text = re.sub(pattern, f"[{pii_type.upper()}_REDACTED]", sanitized_text)

        if SecurityCheckType.PII not in checks_performed:
            checks_performed.append(SecurityCheckType.PII)

        # Calculate overall risk
        overall_risk = max(risk_scores) if risk_scores else 0.0
        is_safe = overall_risk < self.risk_threshold

        # Determine final strategy action
        if not is_safe:
            if self.strategy == SanitizationStrategy.BLOCK:
                sanitized_text = "[BLOCKED_CONTENT]"
            elif self.strategy == SanitizationStrategy.REPLACE:
                sanitized_text = "[SANITIZED_CONTENT]"

        latency = (time.time() - start_time) * 1000

        return SanitizationResult(
            is_safe=is_safe,
            original_text=text,
            sanitized_text=sanitized_text,
            issues_found=issues,
            risk_score=overall_risk,
            strategy_applied=self.strategy,
            checks_performed=checks_performed,
            latency_ms=latency,
        )


# =============================================================================
# OUTPUT VALIDATOR
# =============================================================================


class OutputValidator:
    """Validates LLM outputs for security and quality.

    Performs:
    1. Sensitive data leak detection
    2. Harmful content filtering
    3. Relevance checking
    4. Factuality markers verification

    Usage:
        validator = OutputValidator()
        result = await validator.validate("llm output", "original prompt")
        if not result.is_valid:
            print(f"Issues: {result.issues_found}")
    """

    def __init__(
        self,
        security_level: SecurityLevel = SecurityLevel.STANDARD,
    ):
        self.security_level = security_level
        self.risk_threshold = RISK_THRESHOLDS[security_level]

        # Patterns for sensitive data leaks
        self._secret_patterns = {
            "api_key": r"(?i)(api[_-]?key|apikey)\s*[:=]\s*['\"]?([a-zA-Z0-9_-]{20,})['\"]?",
            "password": r"(?i)(password|passwd|pwd)\s*[:=]\s*['\"]?([^\s'\"]{8,})['\"]?",
            "bearer_token": r"(?i)bearer\s+[a-zA-Z0-9._-]{20,}",
            "connection_string": r"(?i)(mongodb|postgresql|mysql|redis)://[^\s]+",
        }

        # Harmful output patterns
        self._harmful_patterns = [
            r"(?i)here('s| is) (how to|instructions for) (hack|attack|exploit)",
            r"(?i)step\s*1.*step\s*2.*(illegal|harmful)",
        ]

    async def validate(
        self,
        output: str,
        original_prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """Validate LLM output.

        Args:
            output: LLM output to validate
            original_prompt: Original input prompt
            context: Optional context

        Returns:
            ValidationResult with validation details
        """
        start_time = time.time()
        issues: List[str] = []
        checks_performed: List[SecurityCheckType] = []
        risk_scores: List[float] = []
        validated_text = output

        # Try LLM Guard service
        if LLM_GUARD_CLIENT_AVAILABLE:
            try:
                client = get_llm_guard_client()
                if await client.is_ready():
                    scan_result = await client.scan_output(
                        original_prompt,
                        output,
                        scanners=["Deanonymize", "NoRefusal", "Relevance", "Secrets"],
                    )

                    for scanner_result in scan_result.scanner_results:
                        if not scanner_result.is_valid:
                            issues.append(f"{scanner_result.scanner_name}: {scanner_result.details}")
                            risk_scores.append(scanner_result.risk_score)

                    if scan_result.sanitized_text:
                        validated_text = scan_result.sanitized_text

                    checks_performed.append(SecurityCheckType.OUTPUT_RELEVANCE)

            except Exception as e:
                logger.warning(f"LLM Guard output scan failed: {_sanitize_exception_for_logging(e)}")

        # Fallback validation
        if not checks_performed:
            # AUDIT FIX HIGH-001: Normalize text before pattern matching
            normalized_output = normalize_text_for_security(output)
            # AUDIT FIX RED-002/BLUE-001: Use normalized text for redaction too
            validated_text = normalized_output

            # Check for secret leaks - both detection and redaction use normalized output
            for secret_type, pattern in self._secret_patterns.items():
                if re.search(pattern, normalized_output):
                    issues.append(f"Potential {secret_type} leak detected")
                    risk_scores.append(0.9)
                    # Redact on normalized text (AUDIT FIX RED-002/BLUE-001)
                    validated_text = re.sub(pattern, f"[{secret_type.upper()}_REDACTED]", validated_text)

            checks_performed.append(SecurityCheckType.SECRETS)

            # Check for harmful content
            for pattern in self._harmful_patterns:
                if re.search(pattern, normalized_output):
                    issues.append("Potentially harmful instructions detected")
                    risk_scores.append(0.8)

            checks_performed.append(SecurityCheckType.CONTENT_FILTER)

        # Calculate risk
        overall_risk = max(risk_scores) if risk_scores else 0.0
        is_valid = overall_risk < self.risk_threshold

        latency = (time.time() - start_time) * 1000

        return ValidationResult(
            is_valid=is_valid,
            output_text=output,
            validated_text=validated_text,
            issues_found=issues,
            risk_score=overall_risk,
            checks_performed=checks_performed,
            latency_ms=latency,
        )


# =============================================================================
# PII DETECTOR
# =============================================================================


class PIIDetector:
    """Detects and optionally redacts PII from text.

    Supports detection of:
    - Email addresses
    - Phone numbers
    - Social Security Numbers
    - Credit card numbers
    - IP addresses
    - Physical addresses
    - Names (when context available)

    Usage:
        detector = PIIDetector()
        result = await detector.detect("text with email@example.com")
        if result.has_pii:
            print(f"Found: {result.pii_types}")
    """

    # BLUE-019 FIX: Redaction placeholders mapping for PII types
    # These are used with the shared PII_PATTERNS constant
    _REDACTION_PLACEHOLDERS = {
        "email": "[EMAIL_REDACTED]",
        "phone": "[PHONE_REDACTED]",
        "ssn": "[SSN_REDACTED]",
        "credit_card": "[CC_REDACTED]",
        "ip_address": "[IP_REDACTED]",
        "phone_intl": "[PHONE_REDACTED]",
        "date_of_birth": "[DOB_REDACTED]",
    }

    def __init__(self, redact: bool = True):
        self.redact = redact

        # BLUE-019 FIX: Build patterns from shared PII_PATTERNS constant
        # Additional patterns are added for extended detection
        self._patterns = {}
        for pii_type, pattern in PII_PATTERNS.items():
            redaction = self._REDACTION_PLACEHOLDERS.get(pii_type, f"[{pii_type.upper()}_REDACTED]")
            self._patterns[pii_type] = (pattern, redaction)

        # Additional patterns not in shared constant
        self._patterns["phone_intl"] = (
            r"\+[0-9]{1,4}[-.\s]?[0-9]{1,4}[-.\s]?[0-9]{4,10}",
            "[PHONE_REDACTED]"
        )
        self._patterns["date_of_birth"] = (
            r"\b(?:0[1-9]|1[0-2])[/\-.](?:0[1-9]|[12]\d|3[01])[/\-.](?:19|20)\d{2}\b",
            "[DOB_REDACTED]"
        )

    async def detect(
        self,
        text: str,
        anonymize: bool = None,
    ) -> PIIDetectionResult:
        """Detect PII in text.

        Args:
            text: Text to scan
            anonymize: Override default redaction setting

        Returns:
            PIIDetectionResult with detection details
        """
        start_time = time.time()
        should_redact = anonymize if anonymize is not None else self.redact
        pii_types: List[str] = []
        pii_count = 0
        redacted_text = text

        # Try LLM Guard service
        if LLM_GUARD_CLIENT_AVAILABLE:
            try:
                client = get_llm_guard_client()
                if await client.is_ready():
                    result = await client.detect_pii(text, anonymize=should_redact)
                    latency = (time.time() - start_time) * 1000

                    return PIIDetectionResult(
                        has_pii=result.has_pii,
                        pii_types=result.pii_types_found,
                        pii_count=len(result.pii_types_found),
                        redacted_text=result.anonymized_text,
                        confidence=0.9,  # High confidence from service
                        latency_ms=latency,
                    )

            except Exception as e:
                logger.warning(f"LLM Guard PII detection failed: {_sanitize_exception_for_logging(e)}")

        # Fallback detection
        # AUDIT FIX HIGH-001: Normalize text before pattern matching
        normalized_text = normalize_text_for_security(text)
        # AUDIT FIX BLUE-008: Use normalized text for redaction too
        redacted_text = normalized_text
        for pii_type, (pattern, replacement) in self._patterns.items():
            matches = re.findall(pattern, normalized_text)
            if matches:
                pii_types.append(pii_type)
                pii_count += len(matches)
                if should_redact:
                    # Redact on normalized text (AUDIT FIX BLUE-008)
                    redacted_text = re.sub(pattern, replacement, redacted_text)

        latency = (time.time() - start_time) * 1000

        return PIIDetectionResult(
            has_pii=len(pii_types) > 0,
            pii_types=pii_types,
            pii_count=pii_count,
            redacted_text=redacted_text if should_redact and pii_types else None,
            confidence=0.7,  # Lower confidence for pattern matching
            latency_ms=latency,
        )


# =============================================================================
# TOXICITY FILTER
# =============================================================================


class ToxicityFilter:
    """Filters toxic content from text.

    Detects and optionally filters:
    - Hate speech
    - Harassment
    - Self-harm content
    - Sexual content
    - Violence
    - Profanity

    Usage:
        filter = ToxicityFilter()
        result = await filter.filter("text to check")
        if result.is_toxic:
            print(f"Toxicity score: {result.toxicity_score}")
    """

    def __init__(
        self,
        toxicity_threshold: float = 0.7,
        filter_content: bool = True,
    ):
        self.toxicity_threshold = toxicity_threshold
        self.filter_content = filter_content

        # Basic toxicity patterns (fallback)
        self._toxic_patterns = {
            "hate_speech": r"(?i)\b(hate|despise|loathe)\s+(all\s+)?(blacks?|whites?|jews?|muslims?|gays?)\b",
            "harassment": r"(?i)\b(kill\s+yourself|kys|die|worthless|pathetic)\b",
            "profanity": r"(?i)\b(f+u+c+k+|s+h+i+t+|a+s+s+h+o+l+e+)\b",
        }

    async def filter(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ToxicityFilterResult:
        """Filter text for toxicity.

        Args:
            text: Text to filter
            context: Optional context

        Returns:
            ToxicityFilterResult with filtering details
        """
        start_time = time.time()
        categories: Dict[str, float] = {}
        filtered_text = text

        # Try LLM Guard service
        if LLM_GUARD_CLIENT_AVAILABLE:
            try:
                client = get_llm_guard_client()
                if await client.is_ready():
                    result = await client.detect_toxicity(text)
                    latency = (time.time() - start_time) * 1000

                    return ToxicityFilterResult(
                        is_toxic=result.is_toxic,
                        toxicity_score=result.toxicity_score,
                        categories=result.categories,
                        filtered_text=None if not result.is_toxic else "[TOXIC_CONTENT_FILTERED]",
                        latency_ms=latency,
                    )

            except Exception as e:
                logger.warning(f"LLM Guard toxicity detection failed: {_sanitize_exception_for_logging(e)}")

        # Fallback detection
        # AUDIT FIX HIGH-001: Normalize text before pattern matching
        normalized_text = normalize_text_for_security(text)
        # AUDIT FIX BLUE-009: Use normalized text for filtering too
        filtered_text = normalized_text
        toxicity_scores: List[float] = []
        for category, pattern in self._toxic_patterns.items():
            if re.search(pattern, normalized_text):
                categories[category] = 0.8
                toxicity_scores.append(0.8)

        overall_toxicity = max(toxicity_scores) if toxicity_scores else 0.0
        is_toxic = overall_toxicity >= self.toxicity_threshold

        if is_toxic and self.filter_content:
            for pattern in self._toxic_patterns.values():
                # Filter on normalized text (AUDIT FIX BLUE-009)
                filtered_text = re.sub(pattern, "[FILTERED]", filtered_text)

        latency = (time.time() - start_time) * 1000

        return ToxicityFilterResult(
            is_toxic=is_toxic,
            toxicity_score=overall_toxicity,
            categories=categories,
            filtered_text=filtered_text if is_toxic else None,
            latency_ms=latency,
        )


# =============================================================================
# PROMPT INJECTION DETECTOR
# =============================================================================


class PromptInjectionDetector:
    """Detects prompt injection attacks.

    Detects:
    - Direct instruction injection
    - Indirect injection via context
    - Encoding-based injection
    - Multi-turn injection
    - Context manipulation

    Usage:
        detector = PromptInjectionDetector()
        result = await detector.detect("ignore previous instructions...")
        if result.is_injection:
            print(f"Injection type: {result.injection_type}")
    """

    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold

        self._injection_indicators = {
            "direct_injection": [
                r"(?i)ignore\s+(previous|prior|all|your)\s+instructions",
                r"(?i)disregard\s+(previous|prior|all|your)\s+(instructions|rules)",
                r"(?i)forget\s+(everything|all|your)\s+(instructions|rules)",
                r"(?i)new\s+instructions\s*:",
                r"(?i)system\s*:\s*you\s+are",
            ],
            "role_play": [
                r"(?i)pretend\s+(you\s+are|to\s+be|you're)",
                r"(?i)act\s+(as|like)\s+(a|an)",
                r"(?i)you\s+are\s+now\s+",
                r"(?i)from\s+now\s+on\s+you\s+are",
            ],
            "context_manipulation": [
                r"(?i)(end|close)\s+(of\s+)?(system|assistant|user)\s+(message|response)",
                r"(?i)<\|im_start\|>",
                r"(?i)<\|im_end\|>",
                r"(?i)---\s*(system|assistant|user)\s*---",
            ],
            "encoding": [
                r"(?i)base64\s*:",
                r"(?i)decode\s+this",
                r"[A-Za-z0-9+/]{50,}={0,2}",
            ],
        }

    async def detect(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> PromptInjectionDetectionResult:
        """Detect prompt injection in text.

        Args:
            text: Text to analyze
            context: Optional context

        Returns:
            PromptInjectionDetectionResult with detection details
        """
        start_time = time.time()
        indicators_found: List[str] = []
        injection_type: Optional[str] = None
        confidence_scores: List[float] = []

        # Try LLM Guard service
        if LLM_GUARD_CLIENT_AVAILABLE:
            try:
                client = get_llm_guard_client()
                if await client.is_ready():
                    result = await client.detect_prompt_injection(text)
                    latency = (time.time() - start_time) * 1000

                    return PromptInjectionDetectionResult(
                        is_injection=result.is_injection,
                        injection_type=result.injection_type,
                        confidence=result.confidence,
                        indicators=[result.details] if result.details else [],
                        latency_ms=latency,
                    )

            except Exception as e:
                logger.warning(f"LLM Guard injection detection failed: {_sanitize_exception_for_logging(e)}")

        # Fallback detection
        # AUDIT FIX HIGH-001: Normalize text before pattern matching
        normalized_text = normalize_text_for_security(text)
        for inj_type, patterns in self._injection_indicators.items():
            for pattern in patterns:
                if re.search(pattern, normalized_text):
                    indicators_found.append(f"{inj_type}: {pattern[:30]}...")
                    confidence_scores.append(0.7)
                    if not injection_type:
                        injection_type = inj_type

        # Boost confidence for multiple indicators
        if len(indicators_found) > 1:
            confidence_scores = [min(c + 0.1, 1.0) for c in confidence_scores]

        overall_confidence = max(confidence_scores) if confidence_scores else 0.0
        is_injection = overall_confidence >= self.confidence_threshold

        latency = (time.time() - start_time) * 1000

        return PromptInjectionDetectionResult(
            is_injection=is_injection,
            injection_type=injection_type,
            confidence=overall_confidence,
            indicators=indicators_found,
            latency_ms=latency,
        )


# =============================================================================
# LLM GUARD INTEGRATION
# =============================================================================


class LLMGuardIntegration:
    """Main integration class for LLM Guard security features.

    Coordinates all security components:
    - InputSanitizer
    - OutputValidator
    - PIIDetector
    - ToxicityFilter
    - PromptInjectionDetector

    Usage:
        integration = LLMGuardIntegration()
        result = await integration.run_full_check(input_text, output_text)
    """

    _instance: Optional["LLMGuardIntegration"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(
        cls,
        security_level: SecurityLevel = SecurityLevel.STANDARD,
    ) -> "LLMGuardIntegration":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        security_level: SecurityLevel = SecurityLevel.STANDARD,
    ):
        if self._initialized:
            return

        self.security_level = security_level
        self.input_sanitizer = InputSanitizer(security_level)
        self.output_validator = OutputValidator(security_level)
        self.pii_detector = PIIDetector()
        self.toxicity_filter = ToxicityFilter()
        self.injection_detector = PromptInjectionDetector()
        self._initialized = True

        logger.info(f"LLMGuardIntegration initialized with level: {security_level}")

    @property
    def available(self) -> bool:
        """Check if LLM Guard integration is available.

        FIX GAP-NEW-03: Actually check if LLM Guard library is installed.
        """
        return LLM_GUARD_CLIENT_AVAILABLE

    async def run_full_check(
        self,
        input_text: str,
        output_text: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> SecurityCheckResult:
        """Run comprehensive security check.

        Args:
            input_text: User input to check
            output_text: Optional LLM output to validate
            context: Optional context

        Returns:
            SecurityCheckResult with all check results
        """
        # RED-008 FIX: Early return for empty text to avoid unnecessary processing
        # Empty/whitespace-only input is inherently safe but should be logged
        if not input_text or not input_text.strip():
            logger.debug("RED-008: Early return for empty/whitespace input text")
            return SecurityCheckResult(
                is_safe=True,
                overall_risk_score=0.0,
                sanitization=None,
                validation=None,
                pii_detection=None,
                toxicity_filter=None,
                prompt_injection=None,
                blocked=False,
                block_reason=None,
                total_latency_ms=0.0,
            )

        start_time = time.time()
        risk_scores: List[float] = []
        blocked = False
        block_reason = None

        # Run checks in parallel
        tasks = [
            self.input_sanitizer.sanitize(input_text, context),
            self.pii_detector.detect(input_text),
            self.toxicity_filter.filter(input_text, context),
            self.injection_detector.detect(input_text, context),
        ]

        if output_text:
            tasks.append(self.output_validator.validate(output_text, input_text, context))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        sanitization = results[0] if not isinstance(results[0], Exception) else None
        pii_detection = results[1] if not isinstance(results[1], Exception) else None
        toxicity_result = results[2] if not isinstance(results[2], Exception) else None
        injection_result = results[3] if not isinstance(results[3], Exception) else None
        validation = results[4] if len(results) > 4 and not isinstance(results[4], Exception) else None

        # Collect risk scores
        if sanitization:
            risk_scores.append(sanitization.risk_score)
            if not sanitization.is_safe:
                blocked = True
                block_reason = "Input sanitization failed"

        if pii_detection and pii_detection.has_pii:
            risk_scores.append(0.5)

        if toxicity_result and toxicity_result.is_toxic:
            risk_scores.append(toxicity_result.toxicity_score)
            blocked = True
            block_reason = "Toxic content detected"

        if injection_result and injection_result.is_injection:
            risk_scores.append(injection_result.confidence)
            blocked = True
            block_reason = f"Prompt injection detected: {injection_result.injection_type}"

        if validation:
            risk_scores.append(validation.risk_score)
            if not validation.is_valid:
                blocked = True
                block_reason = block_reason or "Output validation failed"

        # Calculate overall
        overall_risk = max(risk_scores) if risk_scores else 0.0
        is_safe = not blocked and overall_risk < self.input_sanitizer.risk_threshold

        # Log to Langfuse
        await self._log_security_check(
            is_safe, overall_risk, blocked, block_reason, context
        )

        total_latency = (time.time() - start_time) * 1000

        return SecurityCheckResult(
            is_safe=is_safe,
            overall_risk_score=overall_risk,
            sanitization=sanitization,
            validation=validation,
            pii_detection=pii_detection,
            toxicity_filter=toxicity_result,
            prompt_injection=injection_result,
            blocked=blocked,
            block_reason=block_reason,
            total_latency_ms=total_latency,
        )

    async def _log_security_check(
        self,
        is_safe: bool,
        risk_score: float,
        blocked: bool,
        block_reason: Optional[str],
        context: Optional[Dict[str, Any]],
    ) -> None:
        """Log security check to Langfuse."""
        try:
            from langfuse import Langfuse

            langfuse = Langfuse()
            langfuse.event(
                name="llm_guard_security_check",
                metadata={
                    "is_safe": is_safe,
                    "risk_score": risk_score,
                    "blocked": blocked,
                    "block_reason": block_reason,
                    "security_level": self.security_level.value,
                    "context": context,
                },
            )
            langfuse.flush()
        except ImportError as e:
            logger.debug(f"LANGFUSE: Langfuse operation failed: {e}")
        except Exception as e:
            logger.debug(f"Failed to log to Langfuse: {e}")


# =============================================================================
# SECURITY ORCHESTRATOR
# =============================================================================


class SecurityOrchestrator:
    """Orchestrates all security checks for the pipeline.

    Integrates:
    - NeMo Enhanced Rails
    - LLM Guard Integration
    - Langfuse logging

    Provides a unified interface for all security operations.

    Usage:
        orchestrator = SecurityOrchestrator()
        result = await orchestrator.secure_operation(
            input_text="user input",
            operation_type="claim_verification"
        )
    """

    _instance: Optional["SecurityOrchestrator"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(
        cls,
        security_level: SecurityLevel = SecurityLevel.STANDARD,
    ) -> "SecurityOrchestrator":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        security_level: SecurityLevel = SecurityLevel.STANDARD,
    ):
        if self._initialized:
            return

        self.security_level = security_level
        self.llm_guard = LLMGuardIntegration(security_level)

        # Try to load NeMo enhanced
        self._nemo_enhanced = None
        try:
            from pipeline.security.nemo_enhanced import get_nemo_enhanced
            self._nemo_enhanced = get_nemo_enhanced()
        except ImportError:
            logger.debug("NeMo enhanced not available")

        self._initialized = True
        logger.info("SecurityOrchestrator initialized")

    @property
    def available(self) -> bool:
        """Check if orchestrator is available.

        FIX GAP-NEW-03: Actually check if underlying security is available.
        """
        return LLM_GUARD_CLIENT_AVAILABLE or (self._nemo_enhanced is not None)

    async def secure_operation(
        self,
        input_text: str,
        output_text: Optional[str] = None,
        operation_type: str = "general",
        context: Optional[Dict[str, Any]] = None,
    ) -> SecurityCheckResult:
        """Secure a pipeline operation.

        Args:
            input_text: User input
            output_text: Optional LLM output
            operation_type: Type of operation
            context: Optional context

        Returns:
            SecurityCheckResult with comprehensive results
        """
        # FIX 2026-01-30: Skip scanning for empty text or pipeline metadata
        # Pipeline metadata dicts converted to string look like "{'sprint_id': ..., 'run_dir': ...}"
        # and were causing false positive prompt injection alerts with 100% confidence
        if not input_text or input_text.strip() == "":
            logger.debug("Skipping security scan - empty input text")
            return SecurityCheckResult(
                is_safe=True,
                overall_risk_score=0.0,
                sanitization=None,
                validation=None,
                pii_detection=None,
                toxicity_filter=None,
                prompt_injection=None,
                blocked=False,
                block_reason=None,
                total_latency_ms=0.0,
            )

        # FIX: Detect pipeline metadata converted to string (false positive source)
        # These patterns indicate internal state was accidentally passed as text
        pipeline_metadata_indicators = ["'sprint_id':", "'run_dir':", "'run_id':", "'phase':"]
        if any(indicator in input_text for indicator in pipeline_metadata_indicators):
            logger.debug(
                f"Skipping security scan - detected pipeline metadata in input (operation: {operation_type})"
            )
            return SecurityCheckResult(
                is_safe=True,
                overall_risk_score=0.0,
                sanitization=None,
                validation=None,
                pii_detection=None,
                toxicity_filter=None,
                prompt_injection=None,
                blocked=False,
                block_reason=None,
                total_latency_ms=0.0,
            )

        # Run NeMo check first if available
        nemo_blocked = False
        nemo_issues: List[str] = []

        if self._nemo_enhanced:
            nemo_result = await self._nemo_enhanced.check_input(input_text)
            if not nemo_result["is_safe"]:
                nemo_blocked = True
                for issue in nemo_result.get("issues", []):
                    nemo_issues.append(str(issue))

        # Run LLM Guard check
        guard_result = await self.llm_guard.run_full_check(
            input_text, output_text, context
        )

        # Combine results
        if nemo_blocked and not guard_result.blocked:
            guard_result.blocked = True
            guard_result.block_reason = f"NeMo: {', '.join(nemo_issues)}"
            guard_result.is_safe = False

        return guard_result

    async def validate_for_gate(
        self,
        gate_id: str,
        input_data: Dict[str, Any],
        output_data: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Validate data for a pipeline gate.

        Args:
            gate_id: Gate identifier
            input_data: Input data to validate
            output_data: Optional output data

        Returns:
            Tuple of (is_valid, error_message)
        """
        input_text = str(input_data.get("text", input_data.get("claim", "")))
        output_text = str(output_data.get("response", "")) if output_data else None

        result = await self.secure_operation(
            input_text,
            output_text,
            operation_type=f"gate_{gate_id}",
            context={"gate_id": gate_id},
        )

        if result.blocked:
            return False, result.block_reason

        return result.is_safe, None


# =============================================================================
# CONVENIENCE FUNCTIONS (BLUE-021 FIX: Thread-safe globals)
# =============================================================================


# BLUE-021 FIX: Use threading.local for thread-safe global state
_thread_local = threading.local()


def get_llm_guard_integration(
    security_level: SecurityLevel = SecurityLevel.STANDARD,
) -> LLMGuardIntegration:
    """Get thread-local LLM Guard integration.

    BLUE-021 FIX: Uses threading.local to avoid race conditions.
    Each thread gets its own instance.
    """
    if not hasattr(_thread_local, 'llm_guard_integration'):
        _thread_local.llm_guard_integration = LLMGuardIntegration(security_level)
    return _thread_local.llm_guard_integration


def get_security_orchestrator(
    security_level: SecurityLevel = SecurityLevel.STANDARD,
) -> SecurityOrchestrator:
    """Get thread-local security orchestrator.

    BLUE-021 FIX: Uses threading.local to avoid race conditions.
    Each thread gets its own instance.
    """
    if not hasattr(_thread_local, 'security_orchestrator'):
        _thread_local.security_orchestrator = SecurityOrchestrator(security_level)
    return _thread_local.security_orchestrator


async def sanitize_input(
    text: str,
    security_level: SecurityLevel = SecurityLevel.STANDARD,
) -> SanitizationResult:
    """Sanitize input text."""
    integration = get_llm_guard_integration(security_level)
    return await integration.input_sanitizer.sanitize(text)


async def validate_output(
    output: str,
    original_prompt: str,
    security_level: SecurityLevel = SecurityLevel.STANDARD,
) -> ValidationResult:
    """Validate LLM output."""
    integration = get_llm_guard_integration(security_level)
    return await integration.output_validator.validate(output, original_prompt)


async def detect_pii(
    text: str,
    anonymize: bool = True,
) -> PIIDetectionResult:
    """Detect PII in text."""
    integration = get_llm_guard_integration()
    return await integration.pii_detector.detect(text, anonymize)


async def filter_toxicity(
    text: str,
) -> ToxicityFilterResult:
    """Filter toxic content."""
    integration = get_llm_guard_integration()
    return await integration.toxicity_filter.filter(text)


async def run_security_checks(
    input_text: str,
    output_text: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> SecurityCheckResult:
    """Run comprehensive security checks."""
    orchestrator = get_security_orchestrator()
    return await orchestrator.secure_operation(input_text, output_text, context=context)


def is_llm_guard_integration_available() -> bool:
    """Check if LLM Guard integration is available.

    FIX GAP-NEW-03: Actually check availability instead of always True.
    """
    return LLM_GUARD_CLIENT_AVAILABLE


# =============================================================================
# DECORATOR
# =============================================================================


def secure_operation(
    security_level: SecurityLevel = SecurityLevel.STANDARD,
    check_output: bool = False,
):
    """Decorator to secure async operations.

    Usage:
        @secure_operation(security_level=SecurityLevel.STRICT)
        async def my_llm_call(prompt: str) -> str:
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get input from args/kwargs
            input_text = kwargs.get("prompt") or kwargs.get("text") or (args[0] if args else "")

            # Check input
            orchestrator = get_security_orchestrator(security_level)
            check_result = await orchestrator.secure_operation(str(input_text))

            if check_result.blocked:
                raise SecurityBlockedError(
                    f"Operation blocked: {check_result.block_reason}",
                    result=check_result,
                )

            # Execute function
            result = await func(*args, **kwargs)

            # Check output if requested
            if check_output and result:
                output_check = await orchestrator.secure_operation(
                    str(input_text),
                    str(result),
                    operation_type="output_validation",
                )
                if output_check.blocked:
                    raise SecurityBlockedError(
                        f"Output blocked: {output_check.block_reason}",
                        result=output_check,
                    )

            return result

        return wrapper
    return decorator


# Alias for async operations (secure_operation already handles async)
async_secure_operation = secure_operation


class SecurityBlockedError(Exception):
    """Exception raised when operation is blocked by security."""

    def __init__(self, message: str, result: SecurityCheckResult):
        super().__init__(message)
        self.result = result


# =============================================================================
# HEALTH CHECK FUNCTIONALITY
# =============================================================================


class LLMGuardHealthChecker:
    """Health checker for LLM Guard service.

    Provides comprehensive health checking including:
    - HTTP endpoint availability (/health)
    - Docker container status verification
    - Timeout handling

    Usage:
        checker = LLMGuardHealthChecker()
        status = await checker.check_health()
        if status["healthy"]:
            print("LLM Guard is operational")
    """

    def __init__(
        self,
        service_url: str = LLM_GUARD_SERVICE_URL,
        timeout: float = LLM_GUARD_HEALTH_CHECK_TIMEOUT,
        container_name: str = LLM_GUARD_DOCKER_CONTAINER,
    ):
        self.service_url = service_url.rstrip("/")
        self.timeout = timeout
        self.container_name = container_name

    async def check_health_endpoint(self) -> Dict[str, Any]:
        """Check LLM Guard /health endpoint.

        Returns:
            Dict with health status from the endpoint.
        """
        import httpx

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.service_url}/health")

                if response.status_code == 200:
                    return {
                        "endpoint_healthy": True,
                        "status_code": response.status_code,
                        "response": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text,
                    }
                else:
                    return {
                        "endpoint_healthy": False,
                        "status_code": response.status_code,
                        "error": f"Unhealthy status code: {response.status_code}",
                    }

        except httpx.TimeoutException:
            return {
                "endpoint_healthy": False,
                "error": f"Health check timed out after {self.timeout}s",
            }
        except httpx.ConnectError:
            return {
                "endpoint_healthy": False,
                "error": f"Could not connect to {self.service_url}",
            }
        except Exception as e:
            return {
                "endpoint_healthy": False,
                "error": f"Health check failed: {_sanitize_exception_for_logging(e)}",
            }

    def check_docker_status(self) -> Dict[str, Any]:
        """Check Docker container status for LLM Guard.

        Returns:
            Dict with Docker container status.
        """
        import subprocess

        try:
            # Check if container exists and is running
            result = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Status}}", self.container_name],
                capture_output=True,
                text=True,
                timeout=5.0,
            )

            if result.returncode == 0:
                status = result.stdout.strip()
                is_running = status == "running"

                return {
                    "docker_available": True,
                    "container_exists": True,
                    "container_status": status,
                    "container_running": is_running,
                }
            else:
                # Container doesn't exist
                return {
                    "docker_available": True,
                    "container_exists": False,
                    "container_running": False,
                    "error": f"Container '{self.container_name}' not found",
                }

        except subprocess.TimeoutExpired:
            return {
                "docker_available": False,
                "error": "Docker command timed out",
            }
        except FileNotFoundError:
            return {
                "docker_available": False,
                "error": "Docker not installed or not in PATH",
            }
        except Exception as e:
            return {
                "docker_available": False,
                "error": f"Docker check failed: {e}",
            }

    async def check_health(self) -> Dict[str, Any]:
        """Perform comprehensive health check.

        Checks both the HTTP endpoint and Docker container status.

        Returns:
            Dict with combined health status.

        Example:
            checker = LLMGuardHealthChecker()
            status = await checker.check_health()
            print(f"Healthy: {status['healthy']}")
            print(f"Endpoint: {status['endpoint']}")
            print(f"Docker: {status['docker']}")
        """
        # Check endpoint (async)
        endpoint_status = await self.check_health_endpoint()

        # Check Docker (sync, run in executor)
        loop = asyncio.get_event_loop()
        docker_status = await loop.run_in_executor(None, self.check_docker_status)

        # Determine overall health
        endpoint_ok = endpoint_status.get("endpoint_healthy", False)
        docker_ok = docker_status.get("container_running", False)

        # Service is healthy if endpoint responds OR (docker running and endpoint not required)
        # Prefer endpoint check when available
        is_healthy = endpoint_ok or docker_ok

        return {
            "healthy": is_healthy,
            "endpoint": endpoint_status,
            "docker": docker_status,
            "service_url": self.service_url,
            "library_available": LLM_GUARD_CLIENT_AVAILABLE,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def is_available(self) -> bool:
        """Simple availability check.

        Returns:
            True if service is available (endpoint responds).
        """
        try:
            status = await self.check_health_endpoint()
            return status.get("endpoint_healthy", False)
        except Exception:
            return False


# Singleton health checker
_health_checker: Optional[LLMGuardHealthChecker] = None
_health_checker_lock = threading.Lock()


def get_llm_guard_health_checker() -> LLMGuardHealthChecker:
    """Get singleton LLM Guard health checker.

    Returns:
        LLMGuardHealthChecker instance.
    """
    global _health_checker
    if _health_checker is None:
        with _health_checker_lock:
            if _health_checker is None:
                _health_checker = LLMGuardHealthChecker()
    return _health_checker


async def check_llm_guard_health() -> Dict[str, Any]:
    """Convenience function to check LLM Guard health.

    Returns:
        Dict with health status.

    Example:
        status = await check_llm_guard_health()
        if status["healthy"]:
            print("LLM Guard operational")
    """
    checker = get_llm_guard_health_checker()
    return await checker.check_health()


async def is_llm_guard_service_available() -> bool:
    """Check if LLM Guard service is available.

    Returns:
        True if service endpoint responds.
    """
    checker = get_llm_guard_health_checker()
    return await checker.is_available()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Classes
    "LLMGuardIntegration",
    "InputSanitizer",
    "OutputValidator",
    "PIIDetector",
    "ToxicityFilter",
    "PromptInjectionDetector",
    "SecurityOrchestrator",
    "LLMGuardHealthChecker",
    # Data classes
    "SanitizationResult",
    "ValidationResult",
    "PIIDetectionResult",
    "ToxicityFilterResult",
    "PromptInjectionDetectionResult",
    "SecurityCheckResult",
    # Enums
    "SecurityLevel",
    "SanitizationStrategy",
    "SecurityCheckType",
    # Functions
    "get_llm_guard_integration",
    "get_security_orchestrator",
    "get_llm_guard_health_checker",
    "sanitize_input",
    "validate_output",
    "detect_pii",
    "filter_toxicity",
    "run_security_checks",
    "is_llm_guard_integration_available",
    "check_llm_guard_health",
    "is_llm_guard_service_available",
    # Decorator
    "secure_operation",
    # Exceptions
    "SecurityBlockedError",
    "RateLimitExceeded",
    # Rate Limiter
    "RateLimiter",
    "get_rate_limiter",
    # Constants
    "LLM_GUARD_INTEGRATION_AVAILABLE",
    "RISK_THRESHOLDS",
    "DEFAULT_SCANNERS",
]
