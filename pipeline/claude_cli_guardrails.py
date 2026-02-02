"""Claude CLI Guardrails Module.

Extracted from claude_cli_llm.py for better separation of concerns (D-HIGH-049).

This module provides security guardrails for LLM operations:
- NeMo Guardrails: Jailbreak detection and content filtering
- QuietStar/Reflexion: Pre-generation safety thinking and post-generation assessment

These guardrails provide multiple layers of security:
1. Input validation (jailbreak, prompt injection, malicious content)
2. Output validation (content policy violations)
3. Safety thinking before generation
4. Quality assessment after generation

AUDIT FIX (2026-01-30): Made fail-closed when guardrails unavailable.
Production environment forces guardrails to be required.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)


# =============================================================================
# FAIL-CLOSED CONFIGURATION
# =============================================================================

# Production check - bypass flags are FORCED to False in production
_IS_PRODUCTION = os.getenv("ENVIRONMENT", "development").lower() == "production"

# Explicit bypass flag - allows skipping guardrails in dev/test ONLY
# In production, this is ALWAYS False regardless of env var
_ALLOW_GUARDRAILS_BYPASS_RAW = os.getenv("ALLOW_GUARDRAILS_BYPASS", "false").lower() == "true"
ALLOW_GUARDRAILS_BYPASS = False if _IS_PRODUCTION else _ALLOW_GUARDRAILS_BYPASS_RAW

if _IS_PRODUCTION and _ALLOW_GUARDRAILS_BYPASS_RAW:
    logger.warning(
        "GUARDRAILS-PROD-001: ALLOW_GUARDRAILS_BYPASS=true ignored in production. "
        "Guardrails are REQUIRED in production environments."
    )


# =============================================================================
# NEMO GUARDRAILS INTEGRATION
# =============================================================================

# NeMo Guardrails integration for jailbreak and content filtering
try:
    from pipeline.security.nemo_enhanced import (
        get_nemo_enhanced,
        NEMO_ENHANCED_AVAILABLE as _NEMO_AVAILABLE,
    )
    NEMO_GUARDRAILS_AVAILABLE = _NEMO_AVAILABLE
except ImportError:
    NEMO_GUARDRAILS_AVAILABLE = False

    def get_nemo_enhanced():  # type: ignore[misc]
        return None


class NeMoGuardrailsError(Exception):
    """Raised when NeMo guardrails block a request."""

    def __init__(self, message: str, guardrail_type: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.guardrail_type = guardrail_type
        self.details = details or {}


def check_nemo_guardrails_input(text: str) -> None:
    """Check input text against NeMo guardrails.

    This function checks for:
    - Jailbreak attempts (DAN, role-play, etc.)
    - Prompt injection
    - Malicious content

    Args:
        text: Input text to check.

    Raises:
        NeMoGuardrailsError: If guardrails detect a violation or are unavailable
                            (fail-closed behavior unless explicitly bypassed).
    """
    # Empty text is valid (nothing to check)
    if not text:
        return

    # AUDIT FIX: Fail-closed when guardrails unavailable
    if not NEMO_GUARDRAILS_AVAILABLE:
        if ALLOW_GUARDRAILS_BYPASS:
            logger.warning(
                "GUARDRAILS-BYPASS-001: NeMo guardrails not available but "
                "ALLOW_GUARDRAILS_BYPASS=true - skipping input check. "
                "This MUST NOT be used in production!"
            )
            return
        else:
            raise NeMoGuardrailsError(
                "GUARDRAILS-UNAVAILABLE-001: NeMo guardrails not available and "
                "ALLOW_GUARDRAILS_BYPASS=false. Input check blocked (fail-closed). "
                "Install NeMo Guardrails or set ALLOW_GUARDRAILS_BYPASS=true for dev.",
                guardrail_type="unavailable",
                details={"reason": "nemo_not_available", "bypass_enabled": False},
            )

    nemo = get_nemo_enhanced()
    if nemo is None:
        if ALLOW_GUARDRAILS_BYPASS:
            logger.warning(
                "GUARDRAILS-BYPASS-002: NeMo instance is None but "
                "ALLOW_GUARDRAILS_BYPASS=true - skipping input check."
            )
            return
        else:
            raise NeMoGuardrailsError(
                "GUARDRAILS-UNAVAILABLE-002: NeMo instance unavailable and "
                "ALLOW_GUARDRAILS_BYPASS=false. Input check blocked (fail-closed).",
                guardrail_type="unavailable",
                details={"reason": "nemo_instance_none", "bypass_enabled": False},
            )

    try:
        # Check for jailbreak
        jailbreak_result = nemo.detect_jailbreak_sync(text[:5000])
        if jailbreak_result.get("is_jailbreak", False):
            raise NeMoGuardrailsError(
                f"NEMO GUARDRAILS: Jailbreak attempt detected - {jailbreak_result.get('jailbreak_type', 'unknown')}",
                guardrail_type="jailbreak",
                details=jailbreak_result,
            )

        # Check content filter
        content_result = nemo.filter_content_sync(text[:5000])
        if not content_result.get("is_safe", True):
            categories = content_result.get("violated_categories", [])
            raise NeMoGuardrailsError(
                f"NEMO GUARDRAILS: Content policy violation - {categories}",
                guardrail_type="content",
                details=content_result,
            )

    except NeMoGuardrailsError:
        raise
    except Exception as e:
        # AUDIT FIX: Fail-closed on errors unless bypass enabled
        if ALLOW_GUARDRAILS_BYPASS:
            logger.warning(
                f"GUARDRAILS-ERROR-001: NeMo check failed but ALLOW_GUARDRAILS_BYPASS=true - "
                f"allowing request to proceed: {e}"
            )
        else:
            raise NeMoGuardrailsError(
                f"GUARDRAILS-ERROR-001: NeMo guardrails check failed and "
                f"ALLOW_GUARDRAILS_BYPASS=false. Request blocked (fail-closed): {e}",
                guardrail_type="error",
                details={"error": str(e), "bypass_enabled": False},
            )


def check_nemo_guardrails_output(text: str) -> str:
    """Check and potentially filter output text.

    Args:
        text: Output text to check.

    Returns:
        Original text or filtered version.

    Raises:
        NeMoGuardrailsError: If output contains critical violations or guardrails
                            are unavailable (fail-closed unless explicitly bypassed).
    """
    # Empty text is valid
    if not text:
        return text

    # AUDIT FIX: Fail-closed when guardrails unavailable
    if not NEMO_GUARDRAILS_AVAILABLE:
        if ALLOW_GUARDRAILS_BYPASS:
            logger.warning(
                "GUARDRAILS-BYPASS-003: NeMo guardrails not available but "
                "ALLOW_GUARDRAILS_BYPASS=true - skipping output check."
            )
            return text
        else:
            raise NeMoGuardrailsError(
                "GUARDRAILS-UNAVAILABLE-003: NeMo guardrails not available and "
                "ALLOW_GUARDRAILS_BYPASS=false. Output check blocked (fail-closed).",
                guardrail_type="unavailable",
                details={"reason": "nemo_not_available", "bypass_enabled": False},
            )

    nemo = get_nemo_enhanced()
    if nemo is None:
        if ALLOW_GUARDRAILS_BYPASS:
            logger.warning(
                "GUARDRAILS-BYPASS-004: NeMo instance is None but "
                "ALLOW_GUARDRAILS_BYPASS=true - skipping output check."
            )
            return text
        else:
            raise NeMoGuardrailsError(
                "GUARDRAILS-UNAVAILABLE-004: NeMo instance unavailable and "
                "ALLOW_GUARDRAILS_BYPASS=false. Output check blocked (fail-closed).",
                guardrail_type="unavailable",
                details={"reason": "nemo_instance_none", "bypass_enabled": False},
            )

    try:
        content_result = nemo.filter_content_sync(text[:10000])
        if not content_result.get("is_safe", True):
            categories = content_result.get("violated_categories", [])
            # For output, we can try to get filtered version
            filtered = content_result.get("filtered_text")
            if filtered:
                logger.warning(f"NeMo: Output filtered due to {categories}")
                return filtered
            # If no filtered version and critical, raise error
            if "harmful" in categories or "illegal" in categories:
                raise NeMoGuardrailsError(
                    f"NEMO GUARDRAILS: Output blocked - {categories}",
                    guardrail_type="output_content",
                    details=content_result,
                )
        return text
    except NeMoGuardrailsError:
        raise
    except Exception as e:
        # AUDIT FIX: Fail-closed on errors unless bypass enabled
        if ALLOW_GUARDRAILS_BYPASS:
            logger.warning(
                f"GUARDRAILS-ERROR-002: NeMo output check failed but "
                f"ALLOW_GUARDRAILS_BYPASS=true - returning original text: {e}"
            )
            return text
        else:
            raise NeMoGuardrailsError(
                f"GUARDRAILS-ERROR-002: NeMo output check failed and "
                f"ALLOW_GUARDRAILS_BYPASS=false. Output blocked (fail-closed): {e}",
                guardrail_type="error",
                details={"error": str(e), "bypass_enabled": False},
            )


# =============================================================================
# QUIETSTAR + REFLEXION GUARDRAILS
# =============================================================================

# QuietStar/Reflexion integration for safety thinking
try:
    from pipeline.security.quietstar_reflexion import (
        QuietStarReflexionGuardrail,
        QuietStarConfig,
        ReflectionDepth,
        ReflexionMode,
        ThinkingResult,
        ReflexionResult,
        GuardrailResult,
        auto_select_depth,
        get_guardrail as _get_quietstar_guardrail,
        SecurityBlockedError as QuietStarBlockedError,
    )
    QUIETSTAR_REFLEXION_AVAILABLE = True
except ImportError:
    QUIETSTAR_REFLEXION_AVAILABLE = False
    QuietStarReflexionGuardrail = None  # type: ignore[assignment, misc]
    QuietStarConfig = None  # type: ignore[assignment, misc]
    ReflectionDepth = None  # type: ignore[assignment, misc]
    ReflexionMode = None  # type: ignore[assignment, misc]
    ThinkingResult = None  # type: ignore[assignment, misc]
    ReflexionResult = None  # type: ignore[assignment, misc]
    GuardrailResult = None  # type: ignore[assignment, misc]
    auto_select_depth = None  # type: ignore[assignment, misc]
    _get_quietstar_guardrail = None  # type: ignore[assignment, misc]

    class QuietStarBlockedError(Exception):  # type: ignore[no-redef]
        """Placeholder when QuietStar not available."""
        pass


# Singleton for QuietStar guardrail (lazy initialization)
_quietstar_guardrail: Optional[Any] = None


def get_or_create_quietstar_guardrail(
    llm_factory: Optional[Callable[[], Any]] = None,
) -> Optional[Any]:
    """Get or create the QuietStar guardrail singleton.

    Args:
        llm_factory: Optional factory function to create an LLM adapter.
                    If None, uses a default ClaudeCLIAdapter.

    Returns:
        QuietStarReflexionGuardrail instance or None if unavailable.
    """
    global _quietstar_guardrail

    if not QUIETSTAR_REFLEXION_AVAILABLE:
        return None

    if _quietstar_guardrail is None:
        try:
            # Create a lightweight LLM callable for safety analysis
            def _safety_llm(prompt: str, max_tokens: int = 200) -> str:
                """Lightweight LLM for safety analysis (sync).

                IMPORTANT: Uses skip_quietstar=True to prevent infinite recursion
                when the guardrail itself needs to call the LLM.
                """
                # Use a local import to avoid circular dependency
                from pipeline.claude_cli_llm import ClaudeCLIAdapter
                adapter = ClaudeCLIAdapter()
                # CRITICAL: skip_quietstar=True to avoid recursion
                response = adapter.complete(prompt, skip_quietstar=True)
                return response.content[:max_tokens * 4]  # Rough token estimate

            # Configure for standard mode (not paranoid to avoid latency)
            config = QuietStarConfig(
                pre_reflection_enabled=True,
                safety_thinking_enabled=True,
                reflexion_mode=ReflexionMode.QUALITY,
                max_reflexion_retries=1,  # Single retry to avoid latency
                risk_threshold=0.7,
                timeout_seconds=30.0,
            )

            _quietstar_guardrail = QuietStarReflexionGuardrail(
                main_llm=_safety_llm,
                safety_llm=_safety_llm,
                config=config,
            )
            logger.info("QuietStar/Reflexion guardrail initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize QuietStar guardrail: {e}")
            return None

    return _quietstar_guardrail


def check_quietstar_pre_generation(
    input_text: str,
    is_code: bool = False,
    is_security_context: bool = False,
) -> Optional[Any]:
    """Run QuietStar pre-generation safety check.

    Args:
        input_text: Input text to check.
        is_code: Whether this is a code generation context.
        is_security_context: Whether this is security-sensitive.

    Returns:
        ThinkingResult if check was performed, None if unavailable.

    Raises:
        QuietStarBlockedError: If input is blocked by safety thinking.
    """
    if not QUIETSTAR_REFLEXION_AVAILABLE:
        return None

    guardrail = get_or_create_quietstar_guardrail()
    if guardrail is None:
        return None

    try:
        import asyncio

        # Select appropriate depth based on context
        depth = auto_select_depth(
            input_text,
            is_code=is_code,
            is_security_context=is_security_context,
        )

        # Run async check in sync context
        try:
            asyncio.get_running_loop()
            # Loop is running - use thread to avoid blocking
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                result = pool.submit(
                    asyncio.run,
                    guardrail.pre_generation_check(input_text, depth)
                ).result(timeout=30)
        except RuntimeError:
            # No running loop - safe to use asyncio.run
            result = asyncio.run(guardrail.pre_generation_check(input_text, depth))

        # Check if blocked
        if hasattr(result, 'blocked') and result.blocked:
            raise QuietStarBlockedError(
                f"QuietStar: Input blocked - {getattr(result, 'block_reason', 'safety concern')}"
            )

        logger.debug(
            f"QuietStar pre-check passed (depth={depth}, "
            f"risk={getattr(result, 'risk_score', 0.0):.2f})"
        )
        return result

    except QuietStarBlockedError:
        raise
    except Exception as e:
        logger.warning(f"QuietStar pre-check failed (continuing): {e}")
        return None


def check_quietstar_post_generation(
    input_text: str,
    output_text: str,
    is_code: bool = False,
) -> Optional[Any]:
    """Run QuietStar post-generation quality assessment.

    Args:
        input_text: Original input text.
        output_text: Generated output to assess.
        is_code: Whether this is code output.

    Returns:
        ReflexionResult if assessment was performed, None if unavailable.
    """
    if not QUIETSTAR_REFLEXION_AVAILABLE:
        return None

    guardrail = get_or_create_quietstar_guardrail()
    if guardrail is None:
        return None

    try:
        import asyncio

        # Run async check in sync context
        try:
            asyncio.get_running_loop()
            # Loop is running - use thread to avoid blocking
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                result = pool.submit(
                    asyncio.run,
                    guardrail.post_generation_check(input_text, output_text)
                ).result(timeout=30)
        except RuntimeError:
            # No running loop - safe to use asyncio.run
            result = asyncio.run(
                guardrail.post_generation_check(input_text, output_text)
            )

        logger.debug(
            f"QuietStar post-check complete (quality={getattr(result, 'quality_score', 0.0):.2f})"
        )
        return result

    except Exception as e:
        logger.warning(f"QuietStar post-check failed (continuing): {e}")
        return None


# =============================================================================
# Convenience exports
# =============================================================================

__all__ = [
    # NeMo
    "NEMO_GUARDRAILS_AVAILABLE",
    "NeMoGuardrailsError",
    "check_nemo_guardrails_input",
    "check_nemo_guardrails_output",
    # QuietStar
    "QUIETSTAR_REFLEXION_AVAILABLE",
    "QuietStarBlockedError",
    "get_or_create_quietstar_guardrail",
    "check_quietstar_pre_generation",
    "check_quietstar_post_generation",
]
