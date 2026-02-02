"""Output Sanitizer for Credential Protection.

THREAT-I-001 FIX: Automatic redaction of credentials in logs and outputs.

This module provides functions to sanitize text before logging or outputting
to prevent accidental credential exposure.

Key Features:
    - Automatic API key redaction
    - Password/secret detection and masking
    - JWT/Bearer token obfuscation
    - Environment variable value protection
    - Custom pattern support

Usage:
    from pipeline.security.output_sanitizer import (
        sanitize_for_logging,
        get_sanitized_logger,
        redact_credentials,
    )

    # Sanitize any text
    safe_text = sanitize_for_logging(dangerous_text)

    # Use sanitized logger
    logger = get_sanitized_logger(__name__)
    logger.info(text_with_possible_secrets)  # Automatically sanitized

Author: Pipeline Autonomo Team
Version: 1.0.0 (2026-01-21)
SECURITY FIX: THREAT-I-001
"""

from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from typing import Any, Dict, Optional, Pattern, Set

# =============================================================================
# CONFIGURATION
# =============================================================================

# Environment variable to control sanitization strictness
SANITIZE_STRICT_MODE = os.getenv("SANITIZE_STRICT_MODE", "true").lower() == "true"

# Minimum length for potential secrets (shorter strings are less likely to be secrets)
MIN_SECRET_LENGTH = 8

# Redaction placeholder
REDACTION_PLACEHOLDER = "[REDACTED]"

# =============================================================================
# CREDENTIAL PATTERNS
# =============================================================================

# Patterns for various credential types with groups for the actual secret
CREDENTIAL_PATTERNS: Dict[str, str] = {
    # API Keys
    "api_key_assignment": r'(?i)(api[_-]?key|apikey)\s*[:=]\s*["\']?([a-zA-Z0-9_-]{20,})["\']?',
    "api_key_header": r'(?i)(x-api-key|api-key)\s*[:=]\s*["\']?([a-zA-Z0-9_-]{20,})["\']?',

    # Passwords
    "password_assignment": r'(?i)(password|passwd|pwd)\s*[:=]\s*["\']?([^\s"\']{8,})["\']?',
    "password_url": r'(?i):([^:@\s]{8,})@[a-zA-Z0-9.-]+',  # user:password@host

    # Secrets/Tokens
    "secret_assignment": r'(?i)(secret|token|auth)\s*[:=]\s*["\']?([a-zA-Z0-9_.-]{20,})["\']?',

    # Bearer Tokens
    "bearer_token": r'(?i)bearer\s+([a-zA-Z0-9_.-]{20,})',

    # JWT
    "jwt_token": r'eyJ[a-zA-Z0-9_-]{10,}\.eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}',

    # AWS
    "aws_access_key": r'(?i)(aws[_-]?access[_-]?key[_-]?id)\s*[:=]\s*["\']?(AKIA[A-Z0-9]{16})["\']?',
    "aws_secret_key": r'(?i)(aws[_-]?secret[_-]?access[_-]?key)\s*[:=]\s*["\']?([a-zA-Z0-9/+=]{40})["\']?',

    # GitHub
    "github_token": r'(?i)(github[_-]?token|gh[_-]?token)\s*[:=]\s*["\']?(ghp_[a-zA-Z0-9]{36}|gho_[a-zA-Z0-9]{36})["\']?',

    # Stripe
    "stripe_key": r'(?i)(stripe[_-]?key|sk_live|pk_live)\s*[:=]\s*["\']?([a-zA-Z0-9_]{20,})["\']?',

    # Generic secrets (catch-all for common patterns)
    "generic_key_value": r'(?i)(private[_-]?key|access[_-]?token|client[_-]?secret)\s*[:=]\s*["\']?([a-zA-Z0-9_.-]{20,})["\']?',

    # GAP-SEC-005 FIX: Additional credential patterns for comprehensive redaction

    # OpenAI/Anthropic API keys
    "openai_key": r'(?i)(openai[_-]?api[_-]?key|sk-[a-zA-Z0-9]{20,})\s*[:=]?\s*["\']?([a-zA-Z0-9_-]{20,})["\']?',
    "anthropic_key": r'(?i)(anthropic[_-]?api[_-]?key|sk-ant-[a-zA-Z0-9_-]{20,})',

    # Database connection strings
    "database_url": r'(?i)(postgres|mysql|mongodb|redis)://[^:]+:([^@\s]{8,})@',

    # SSH private keys
    "ssh_key": r'-----BEGIN (RSA |DSA |EC |OPENSSH )?PRIVATE KEY-----',

    # Google/GCP
    "gcp_key": r'(?i)(google[_-]?api[_-]?key|gcp[_-]?api[_-]?key)\s*[:=]\s*["\']?([a-zA-Z0-9_-]{35,})["\']?',

    # Slack tokens
    "slack_token": r'(?i)(xox[pboa]-[0-9]+-[0-9]+-[a-zA-Z0-9]+)',

    # NPM tokens
    "npm_token": r'(?i)(npm_[a-zA-Z0-9]{36})',

    # Twilio
    "twilio_key": r'(?i)(twilio[_-]?(auth[_-]?token|api[_-]?key|sid))\s*[:=]\s*["\']?([a-zA-Z0-9]{32})["\']?',

    # Sendgrid
    "sendgrid_key": r'(?i)(sendgrid[_-]?api[_-]?key|SG\.[a-zA-Z0-9_-]{22}\.[a-zA-Z0-9_-]{43})',

    # Mailgun
    "mailgun_key": r'(?i)(mailgun[_-]?api[_-]?key|key-[a-zA-Z0-9]{32})',

    # Generic base64 secrets (long base64 strings often contain encoded secrets)
    "base64_secret": r'(?i)(secret|key|token|password|credential)\s*[:=]\s*["\']?([A-Za-z0-9+/]{40,}={0,2})["\']?',
}

# Environment variable names that typically contain secrets
# GAP-SEC-005 FIX: Extended list of sensitive environment variables
SENSITIVE_ENV_VARS: Set[str] = {
    # API Keys
    "API_KEY", "APIKEY", "API_SECRET", "API_TOKEN",
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "CLAUDE_API_KEY",
    "GOOGLE_API_KEY", "GCP_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS",
    "AZURE_API_KEY", "AZURE_OPENAI_KEY",
    "HUGGINGFACE_TOKEN", "HF_TOKEN",

    # Authentication
    "PASSWORD", "PASSWD", "PWD",
    "SECRET", "SECRET_KEY", "PRIVATE_KEY",
    "ACCESS_TOKEN", "AUTH_TOKEN", "BEARER_TOKEN",
    "JWT_SECRET", "JWT_KEY", "SESSION_SECRET",
    "CLERK_SECRET_KEY", "CLERK_API_KEY",

    # Cloud providers
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
    "AZURE_CLIENT_SECRET", "AZURE_TENANT_ID",

    # Databases
    "DATABASE_URL", "DATABASE_PASSWORD", "DB_PASSWORD",
    "POSTGRES_PASSWORD", "MYSQL_PASSWORD", "MONGO_PASSWORD",
    "REDIS_PASSWORD", "REDIS_URL",
    "QDRANT_API_KEY", "NEO4J_PASSWORD", "FALKORDB_PASSWORD",

    # Version Control
    "GITHUB_TOKEN", "GH_TOKEN", "GITLAB_TOKEN", "BITBUCKET_TOKEN",

    # Payment/Financial
    "STRIPE_SECRET_KEY", "STRIPE_KEY", "STRIPE_API_KEY",
    "PAYPAL_SECRET", "SQUARE_ACCESS_TOKEN",

    # Communication
    "SLACK_TOKEN", "SLACK_WEBHOOK_URL",
    "DISCORD_TOKEN", "DISCORD_WEBHOOK",
    "TWILIO_AUTH_TOKEN", "SENDGRID_API_KEY", "MAILGUN_API_KEY",

    # Monitoring
    "SENTRY_DSN", "DATADOG_API_KEY", "NEW_RELIC_LICENSE_KEY",
    "LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY",

    # Misc
    "ENCRYPTION_KEY", "SIGNING_KEY", "WEBHOOK_SECRET",
    "SSH_PRIVATE_KEY", "PGP_PRIVATE_KEY",
}

# =============================================================================
# COMPILED PATTERNS (for performance)
# =============================================================================

@lru_cache(maxsize=1)
def _get_compiled_patterns() -> Dict[str, Pattern]:
    """Get compiled regex patterns (cached for performance)."""
    return {name: re.compile(pattern) for name, pattern in CREDENTIAL_PATTERNS.items()}


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def redact_credentials(text: str) -> str:
    """THREAT-I-001 FIX: Redact credentials from text.

    This function detects and masks various types of credentials:
    - API keys
    - Passwords
    - Secret tokens
    - Bearer tokens
    - JWT tokens
    - AWS credentials
    - etc.

    Args:
        text: Text potentially containing credentials.

    Returns:
        Text with credentials replaced by [REDACTED].

    Example:
        >>> redact_credentials("api_key=sk_test_EXAMPLE_KEY_HERE")
        'api_key=[REDACTED]'
    """
    if not text:
        return text

    result = text
    patterns = _get_compiled_patterns()

    for pattern_name, pattern in patterns.items():
        # For patterns with groups, replace the captured group
        def replace_match(m):
            if m.lastindex and m.lastindex >= 1:
                # Replace the last group (usually the secret)
                full_match = m.group(0)
                secret = m.group(m.lastindex)
                return full_match.replace(secret, REDACTION_PLACEHOLDER)
            return REDACTION_PLACEHOLDER

        result = pattern.sub(replace_match, result)

    return result


def sanitize_for_logging(text: str, strict: bool = None) -> str:
    """THREAT-I-001 FIX: Sanitize text for safe logging.

    This function applies credential redaction and additional
    sanitization to make text safe for logging.

    Args:
        text: Text to sanitize.
        strict: Override for strict mode (defaults to SANITIZE_STRICT_MODE).

    Returns:
        Sanitized text safe for logging.
    """
    if not text:
        return text

    strict = strict if strict is not None else SANITIZE_STRICT_MODE

    # Apply credential redaction
    result = redact_credentials(text)

    if strict:
        # Additional strict mode redactions
        # Redact anything that looks like a base64-encoded secret
        result = re.sub(
            r'[A-Za-z0-9+/]{40,}={0,2}',
            REDACTION_PLACEHOLDER,
            result
        )

        # Redact long hex strings (potential hashes/keys)
        result = re.sub(
            r'\b[0-9a-fA-F]{32,}\b',
            REDACTION_PLACEHOLDER,
            result
        )

    return result


def sanitize_dict(data: Dict[str, Any], sensitive_keys: Optional[Set[str]] = None) -> Dict[str, Any]:
    """THREAT-I-001 FIX: Sanitize a dictionary for safe logging.

    Recursively sanitizes dictionary values, with special handling
    for known sensitive keys.

    Args:
        data: Dictionary to sanitize.
        sensitive_keys: Additional keys to treat as sensitive.

    Returns:
        Sanitized dictionary (deep copy).
    """
    if not data:
        return data

    keys_to_redact = SENSITIVE_ENV_VARS.copy()
    if sensitive_keys:
        keys_to_redact.update(sensitive_keys)

    result = {}
    for key, value in data.items():
        key_upper = key.upper()

        # Check if key name suggests sensitive data
        is_sensitive_key = (
            key_upper in keys_to_redact or
            any(s in key_upper for s in ("KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL"))
        )

        if is_sensitive_key:
            result[key] = REDACTION_PLACEHOLDER
        elif isinstance(value, dict):
            result[key] = sanitize_dict(value, sensitive_keys)
        elif isinstance(value, str):
            result[key] = sanitize_for_logging(value)
        elif isinstance(value, list):
            result[key] = [
                sanitize_dict(item, sensitive_keys) if isinstance(item, dict)
                else sanitize_for_logging(item) if isinstance(item, str)
                else item
                for item in value
            ]
        else:
            result[key] = value

    return result


# =============================================================================
# SANITIZED LOGGER
# =============================================================================

class SanitizedLoggerAdapter(logging.LoggerAdapter):
    """THREAT-I-001 FIX: Logger adapter that automatically sanitizes log messages."""

    def process(self, msg, kwargs):
        """Sanitize the log message before passing to the underlying logger."""
        if isinstance(msg, str):
            msg = sanitize_for_logging(msg)
        return msg, kwargs


def get_sanitized_logger(name: str) -> SanitizedLoggerAdapter:
    """THREAT-I-001 FIX: Get a logger that automatically sanitizes output.

    Args:
        name: Logger name (typically __name__).

    Returns:
        SanitizedLoggerAdapter wrapping the standard logger.

    Usage:
        logger = get_sanitized_logger(__name__)
        logger.info("Processing with api_key=sk_live_secret123")  # Key is redacted
    """
    return SanitizedLoggerAdapter(logging.getLogger(name), {})


# =============================================================================
# ENVIRONMENT VARIABLE PROTECTION
# =============================================================================

def get_safe_env(var_name: str, default: str = "") -> str:
    """THREAT-I-001 FIX: Get environment variable with logging protection.

    Returns the environment variable value but logs a safe placeholder
    if the variable is sensitive.

    Args:
        var_name: Environment variable name.
        default: Default value if not set.

    Returns:
        Environment variable value.
    """
    value = os.getenv(var_name, default)

    # Check if this is a sensitive variable
    if var_name.upper() in SENSITIVE_ENV_VARS or any(
        s in var_name.upper() for s in ("KEY", "SECRET", "TOKEN", "PASSWORD")
    ):
        # Log that we accessed it, but not the value
        logging.getLogger(__name__).debug(
            f"Accessed sensitive env var: {var_name} (value redacted)"
        )

    return value


def log_env_summary() -> Dict[str, str]:
    """THREAT-I-001 FIX: Log a safe summary of environment configuration.

    Returns a dictionary with sensitive values redacted, suitable for logging
    the current environment configuration.

    Returns:
        Dictionary with environment variable names and safe values.
    """
    result = {}
    for key in os.environ:
        if key.upper() in SENSITIVE_ENV_VARS or any(
            s in key.upper() for s in ("KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL")
        ):
            result[key] = REDACTION_PLACEHOLDER
        else:
            value = os.environ[key]
            # Still sanitize in case the value contains sensitive data
            result[key] = sanitize_for_logging(value[:100])  # Truncate for safety

    return result


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "redact_credentials",
    "sanitize_for_logging",
    "sanitize_dict",
    "get_sanitized_logger",
    "get_safe_env",
    "log_env_summary",
    "REDACTION_PLACEHOLDER",
    "SanitizedLoggerAdapter",
]
