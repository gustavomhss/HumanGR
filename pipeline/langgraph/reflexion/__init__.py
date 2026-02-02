"""Reflexion Integration Module for Pipeline V2.

This module provides enhanced Reflexion capabilities for self-improvement
through self-reflection loops, error correction, and learning from feedback.

Key Features:
- Self-reflection loops with iterative improvement
- Error correction mechanisms with pattern detection
- Learning from feedback with persistent memory
- Integration with A-MEM for learning persistence
- Graceful degradation when dependencies unavailable

Architecture:
    Gate Failure
        |
        v
    ReflexionEngine
        |
        ├─> SelfReflectionLoop
        │       ↓
        │   ReflectionResult
        │       ↓
        │   ActionPlan
        │
        ├─> ErrorCorrector
        │       ↓
        │   ErrorAnalysis
        │       ↓
        │   Correction
        │
        └─> FeedbackLearner
                ↓
            LearningEntry
                ↓
            A-MEM (persistent)

Usage:
    from pipeline.langgraph.reflexion import (
        get_reflexion_engine,
        run_self_reflection,
        analyze_and_correct_error,
        learn_from_feedback,
    )

    # Run self-reflection on failure
    engine = get_reflexion_engine()
    result = await engine.reflect(
        failure_context={"gate_id": "G1", "error": "validation failed"},
        max_iterations=3,
    )

    # Learn from feedback
    await engine.learn_from_feedback(
        action="retry_with_different_approach",
        outcome="success",
        context={"gate_id": "G1"},
    )

Author: Pipeline Autonomo Team
Version: 1.0.0 (2026-01-16)
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Import core components
try:
    from pipeline.langgraph.reflexion.engine import (
        # Classes
        ReflexionEngine,
        SelfReflectionLoop,
        ErrorCorrector,
        FeedbackLearner,
        # Result types
        ReflectionResult,
        ReflectionIteration,
        ActionPlan,
        ErrorAnalysis,
        CorrectionResult,
        LearningEntry,
        FeedbackRecord,
        ReflexionMetrics,
        # Enums
        ReflectionQuality,
        ErrorSeverity,
        LearningType,
        # Functions
        get_reflexion_engine,
        run_self_reflection,
        analyze_and_correct_error,
        learn_from_feedback,
        get_reflexion_metrics,
        # Constants
        REFLEXION_AVAILABLE,
    )
    _REFLEXION_IMPORT_SUCCESS = True
except ImportError as e:
    logger.debug(f"Reflexion engine not available: {e}")
    _REFLEXION_IMPORT_SUCCESS = False
    REFLEXION_AVAILABLE = False

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = []

if _REFLEXION_IMPORT_SUCCESS:
    __all__.extend([
        # Classes
        "ReflexionEngine",
        "SelfReflectionLoop",
        "ErrorCorrector",
        "FeedbackLearner",
        # Result types
        "ReflectionResult",
        "ReflectionIteration",
        "ActionPlan",
        "ErrorAnalysis",
        "CorrectionResult",
        "LearningEntry",
        "FeedbackRecord",
        "ReflexionMetrics",
        # Enums
        "ReflectionQuality",
        "ErrorSeverity",
        "LearningType",
        # Functions
        "get_reflexion_engine",
        "run_self_reflection",
        "analyze_and_correct_error",
        "learn_from_feedback",
        "get_reflexion_metrics",
        # Constants
        "REFLEXION_AVAILABLE",
    ])
else:
    __all__.append("REFLEXION_AVAILABLE")
