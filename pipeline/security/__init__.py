"""Pipeline V2 Security Module.

This module provides comprehensive security integrations for the Pipeline Autonomo project,
including NeMo Guardrails enhancements and LLM Guard integration.

Components:
1. NeMo Guardrails Enhancements:
   - Additional Colang flows for edge cases
   - Custom action implementations
   - Dialog management
   - Jailbreak detection patterns
   - Content filtering rules

2. LLM Guard Integration:
   - Input sanitization
   - Output validation
   - PII detection and redaction
   - Prompt injection detection
   - Toxicity filtering
   - Integration with pipeline gates

Architecture:
    SecurityOrchestrator
        |
        +-- NeMoEnhancedRails (Colang-based guardrails)
        |       |
        |       +-- JailbreakDetector
        |       +-- ContentFilter
        |       +-- DialogManager
        |
        +-- LLMGuardIntegration (Security scanning)
                |
                +-- InputSanitizer
                +-- OutputValidator
                +-- PIIDetector
                +-- ToxicityFilter

Design Principles:
- Defense in depth: Multiple layers of security
- Graceful degradation: Falls back when services unavailable
- Async-first: All operations are async
- Observable: Comprehensive logging via Langfuse
- Configurable: Policies are configurable

Author: Pipeline Autonomo Team
Version: 1.0.0 (2026-01-16)
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# =============================================================================
# NEMO ENHANCEMENTS
# =============================================================================

try:
    from pipeline.security.nemo_enhanced import (
        # Classes
        NeMoEnhancedRails,
        JailbreakDetector,
        ContentFilter,
        DialogManager,
        # Functions
        get_nemo_enhanced,
        detect_jailbreak,
        filter_content,
        manage_dialog,
        is_nemo_enhanced_available,
        # Data classes
        JailbreakResult,
        ContentFilterResult,
        DialogState,
        # Constants
        NEMO_ENHANCED_AVAILABLE,
    )
    __all__ = [
        "NeMoEnhancedRails",
        "JailbreakDetector",
        "ContentFilter",
        "DialogManager",
        "get_nemo_enhanced",
        "detect_jailbreak",
        "filter_content",
        "manage_dialog",
        "is_nemo_enhanced_available",
        "JailbreakResult",
        "ContentFilterResult",
        "DialogState",
        "NEMO_ENHANCED_AVAILABLE",
    ]
except ImportError as e:
    logger.debug(f"NeMo enhancements not available: {e}")
    NEMO_ENHANCED_AVAILABLE = False
    __all__ = ["NEMO_ENHANCED_AVAILABLE"]

# =============================================================================
# LLM GUARD INTEGRATION
# =============================================================================

try:
    from pipeline.security.llm_guard_integration import (
        # Classes
        LLMGuardIntegration,
        InputSanitizer,
        OutputValidator,
        PIIDetector,
        ToxicityFilter,
        SecurityOrchestrator,
        # Functions
        get_llm_guard_integration,
        get_security_orchestrator,
        sanitize_input,
        validate_output,
        detect_pii,
        filter_toxicity,
        run_security_checks,
        is_llm_guard_integration_available,
        # Data classes
        SanitizationResult,
        ValidationResult,
        PIIDetectionResult,
        ToxicityFilterResult,
        SecurityCheckResult,
        # Constants
        LLM_GUARD_INTEGRATION_AVAILABLE,
    )
    __all__.extend([
        "LLMGuardIntegration",
        "InputSanitizer",
        "OutputValidator",
        "PIIDetector",
        "ToxicityFilter",
        "SecurityOrchestrator",
        "get_llm_guard_integration",
        "get_security_orchestrator",
        "sanitize_input",
        "validate_output",
        "detect_pii",
        "filter_toxicity",
        "run_security_checks",
        "is_llm_guard_integration_available",
        "SanitizationResult",
        "ValidationResult",
        "PIIDetectionResult",
        "ToxicityFilterResult",
        "SecurityCheckResult",
        "LLM_GUARD_INTEGRATION_AVAILABLE",
    ])
except ImportError as e:
    logger.debug(f"LLM Guard integration not available: {e}")
    LLM_GUARD_INTEGRATION_AVAILABLE = False
    if "LLM_GUARD_INTEGRATION_AVAILABLE" not in __all__:
        __all__.append("LLM_GUARD_INTEGRATION_AVAILABLE")

# =============================================================================
# SECURITY GATE INTEGRATION
# =============================================================================

try:
    from pipeline.security.gate_integration import (
        # Classes
        SecurityGate,
        SecurityGateRunner,
        # Functions
        get_security_gate_runner,
        run_security_gate,
        validate_gate_security,
        # Data classes
        SecurityGateResult,
        # Constants
        SECURITY_GATE_AVAILABLE,
    )
    __all__.extend([
        "SecurityGate",
        "SecurityGateRunner",
        "get_security_gate_runner",
        "run_security_gate",
        "validate_gate_security",
        "SecurityGateResult",
        "SECURITY_GATE_AVAILABLE",
    ])
except ImportError as e:
    logger.debug(f"Security gate integration not available: {e}")
    SECURITY_GATE_AVAILABLE = False
    if "SECURITY_GATE_AVAILABLE" not in __all__:
        __all__.append("SECURITY_GATE_AVAILABLE")

# =============================================================================
# PYRIT RED TEAMING
# =============================================================================

try:
    from pipeline.security.pyrit_client import (
        PyRITClient,
        AttackStrategy,
        AttackPrompt,
        AttackResult,
        VulnerabilityReport,
        RedTeamAssessment,
        SeverityLevel,
        get_pyrit_client,
        PYRIT_AVAILABLE,
    )
    __all__.extend([
        "PyRITClient",
        "AttackStrategy",
        "AttackPrompt",
        "AttackResult",
        "VulnerabilityReport",
        "RedTeamAssessment",
        "SeverityLevel",
        "get_pyrit_client",
        "PYRIT_AVAILABLE",
    ])
except ImportError as e:
    logger.debug(f"PyRIT client not available: {e}")
    PYRIT_AVAILABLE = False
    if "PYRIT_AVAILABLE" not in __all__:
        __all__.append("PYRIT_AVAILABLE")
