"""Reflexion Engine for Pipeline V2.

This module implements the core Reflexion engine with self-reflection loops,
error correction mechanisms, and learning from feedback.

Based on the Reflexion paper: "Reflexion: Language Agents with Verbal Reinforcement Learning"
(Shinn et al., 2023)

Key Components:
- SelfReflectionLoop: Iterative self-reflection with quality scoring
- ErrorCorrector: Pattern-based error detection and correction
- FeedbackLearner: Learning from outcomes with A-MEM persistence

Author: Pipeline Autonomo Team
Version: 1.0.0 (2026-01-16)
"""

from __future__ import annotations

import os
import logging
import hashlib
import json
from typing import Any, Dict, List, Optional, TypedDict, Callable, Awaitable
from datetime import datetime, timedelta, timezone
from enum import Enum

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

MAX_REFLECTION_ITERATIONS = int(os.getenv("REFLEXION_MAX_ITERATIONS", "5"))
IMPROVEMENT_THRESHOLD = float(os.getenv("REFLEXION_IMPROVEMENT_THRESHOLD", "0.1"))
MIN_QUALITY_SCORE = float(os.getenv("REFLEXION_MIN_QUALITY", "0.3"))
LEARNING_PERSISTENCE_ENABLED = os.getenv("REFLEXION_LEARNING_ENABLED", "true").lower() == "true"

# Check dependencies
REDIS_AVAILABLE = False
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    logger.debug("Redis not available for Reflexion")

REFLEXION_AVAILABLE = True  # Core functionality always available


# =============================================================================
# ENUMS
# =============================================================================


class ReflectionQuality(str, Enum):
    """Quality level of a reflection."""
    EXCELLENT = "excellent"    # Score >= 0.9
    GOOD = "good"              # Score >= 0.7
    ADEQUATE = "adequate"      # Score >= 0.5
    POOR = "poor"              # Score >= 0.3
    INSUFFICIENT = "insufficient"  # Score < 0.3


class ErrorSeverity(str, Enum):
    """Severity of an error."""
    CRITICAL = "critical"      # Requires immediate intervention
    HIGH = "high"              # Significant impact
    MEDIUM = "medium"          # Moderate impact
    LOW = "low"                # Minor impact
    INFO = "info"              # Informational only


class LearningType(str, Enum):
    """Type of learning entry."""
    ERROR_PATTERN = "error_pattern"      # Learned error pattern
    SUCCESS_PATTERN = "success_pattern"  # Learned success pattern
    STRATEGY = "strategy"                # Learned strategy
    HEURISTIC = "heuristic"              # Learned heuristic
    CONSTRAINT = "constraint"            # Learned constraint


# =============================================================================
# RESULT TYPES
# =============================================================================


class ReflectionIteration(TypedDict):
    """A single iteration of self-reflection."""
    iteration_number: int
    thought: str
    analysis: str
    identified_issues: List[str]
    proposed_improvements: List[str]
    quality_score: float
    timestamp: str


class ActionPlan(TypedDict):
    """Action plan derived from reflection."""
    actions: List[Dict[str, Any]]
    priority_order: List[int]
    estimated_impact: float
    confidence: float
    rationale: str


class ReflectionResult(TypedDict):
    """Complete result of self-reflection."""
    reflection_id: str
    context: Dict[str, Any]
    iterations: List[ReflectionIteration]
    final_analysis: str
    action_plan: ActionPlan
    total_iterations: int
    converged: bool
    final_quality: float
    quality_level: str
    total_time_ms: float
    success: bool
    error: Optional[str]


class ErrorAnalysis(TypedDict):
    """Analysis of an error."""
    error_id: str
    error_type: str
    error_message: str
    severity: str
    root_cause: Optional[str]
    contributing_factors: List[str]
    similar_errors: List[Dict[str, Any]]
    pattern_id: Optional[str]
    context: Dict[str, Any]


class CorrectionResult(TypedDict):
    """Result of error correction."""
    error_id: str
    correction_applied: bool
    correction_type: str
    original_state: Dict[str, Any]
    corrected_state: Dict[str, Any]
    confidence: float
    verification_passed: bool
    side_effects: List[str]
    success: bool
    error: Optional[str]


class LearningEntry(TypedDict):
    """An entry in the learning store."""
    learning_id: str
    learning_type: str
    pattern: Dict[str, Any]
    context: Dict[str, Any]
    outcome: str
    confidence: float
    occurrences: int
    first_seen: str
    last_seen: str
    metadata: Dict[str, Any]


class FeedbackRecord(TypedDict):
    """Record of feedback received."""
    feedback_id: str
    action_taken: str
    outcome: str
    context: Dict[str, Any]
    learning_applied: List[str]
    timestamp: str


class ReflexionMetrics(TypedDict):
    """Metrics for Reflexion system."""
    total_reflections: int
    successful_reflections: int
    average_iterations: float
    average_quality: float
    total_corrections: int
    successful_corrections: int
    total_learnings: int
    learning_applications: int
    uptime_seconds: float


# =============================================================================
# SELF-REFLECTION LOOP
# =============================================================================


class SelfReflectionLoop:
    """Implements iterative self-reflection with quality scoring.

    The loop continues until:
    1. Quality score converges (improvement < threshold)
    2. Maximum iterations reached
    3. Quality exceeds target threshold
    """

    def __init__(
        self,
        max_iterations: int = MAX_REFLECTION_ITERATIONS,
        improvement_threshold: float = IMPROVEMENT_THRESHOLD,
        min_quality: float = MIN_QUALITY_SCORE,
        quality_evaluator: Optional[Callable[[str], Awaitable[float]]] = None,
    ):
        """Initialize self-reflection loop.

        Args:
            max_iterations: Maximum reflection iterations
            improvement_threshold: Minimum improvement to continue
            min_quality: Minimum quality score required
            quality_evaluator: Optional async function to evaluate quality
        """
        self.max_iterations = max_iterations
        self.improvement_threshold = improvement_threshold
        self.min_quality = min_quality
        self._quality_evaluator = quality_evaluator
        self._iterations: List[ReflectionIteration] = []

    async def reflect(
        self,
        context: Dict[str, Any],
        initial_analysis: Optional[str] = None,
    ) -> ReflectionResult:
        """Run self-reflection loop.

        Args:
            context: Context for reflection (e.g., failure details)
            initial_analysis: Optional initial analysis to start from

        Returns:
            ReflectionResult with all iterations and final analysis
        """
        import time
        import uuid
        start_time = time.time()

        reflection_id = str(uuid.uuid4())[:8]
        self._iterations = []
        current_quality = 0.0
        converged = False

        try:
            # Initial reflection
            current_thought = initial_analysis or self._generate_initial_thought(context)

            for i in range(self.max_iterations):
                # Generate reflection iteration
                iteration = await self._generate_iteration(
                    iteration_number=i + 1,
                    previous_thought=current_thought,
                    context=context,
                )
                self._iterations.append(iteration)

                # Check for convergence
                improvement = iteration["quality_score"] - current_quality
                current_quality = iteration["quality_score"]

                if improvement < self.improvement_threshold and i > 0:
                    converged = True
                    break

                if current_quality >= 0.9:
                    converged = True
                    break

                current_thought = iteration["analysis"]

            # Generate final analysis and action plan
            final_analysis = self._synthesize_final_analysis()
            action_plan = self._create_action_plan()
            quality_level = self._get_quality_level(current_quality)

            elapsed_ms = (time.time() - start_time) * 1000

            return ReflectionResult(
                reflection_id=reflection_id,
                context=context,
                iterations=self._iterations,
                final_analysis=final_analysis,
                action_plan=action_plan,
                total_iterations=len(self._iterations),
                converged=converged,
                final_quality=current_quality,
                quality_level=quality_level.value,
                total_time_ms=elapsed_ms,
                success=True,
                error=None,
            )

        except Exception as e:
            logger.error(f"Self-reflection failed: {e}")
            elapsed_ms = (time.time() - start_time) * 1000
            return ReflectionResult(
                reflection_id=reflection_id,
                context=context,
                iterations=self._iterations,
                final_analysis="",
                action_plan=ActionPlan(
                    actions=[],
                    priority_order=[],
                    estimated_impact=0.0,
                    confidence=0.0,
                    rationale="Reflection failed",
                ),
                total_iterations=len(self._iterations),
                converged=False,
                final_quality=0.0,
                quality_level=ReflectionQuality.INSUFFICIENT.value,
                total_time_ms=elapsed_ms,
                success=False,
                error=str(e),
            )

    def _generate_initial_thought(self, context: Dict[str, Any]) -> str:
        """Generate initial thought from context."""
        gate_id = context.get("gate_id", "unknown")
        error = context.get("error", "unknown error")
        return f"Analyzing failure in {gate_id}: {error}"

    async def _generate_iteration(
        self,
        iteration_number: int,
        previous_thought: str,
        context: Dict[str, Any],
    ) -> ReflectionIteration:
        """Generate a single reflection iteration."""
        # Analyze previous thought
        issues = self._identify_issues(previous_thought, context)
        improvements = self._propose_improvements(issues, context)

        # Generate new analysis
        analysis = self._generate_analysis(
            previous_thought,
            issues,
            improvements,
            context,
        )

        # Evaluate quality
        quality_score = await self._evaluate_quality(analysis)

        return ReflectionIteration(
            iteration_number=iteration_number,
            thought=previous_thought,
            analysis=analysis,
            identified_issues=issues,
            proposed_improvements=improvements,
            quality_score=quality_score,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _identify_issues(
        self,
        thought: str,
        context: Dict[str, Any],
    ) -> List[str]:
        """Identify issues in the current thought/approach."""
        issues = []

        # Check for common patterns
        error_type = context.get("error_type", "")
        error_message = context.get("error", "")

        if "validation" in error_message.lower():
            issues.append("Input validation failure detected")

        if "timeout" in error_message.lower():
            issues.append("Timeout issue - possible resource contention")

        if "not found" in error_message.lower():
            issues.append("Missing resource or dependency")

        if context.get("retry_count", 0) > 0:
            issues.append("Recurring failure - pattern may exist")

        if not issues:
            issues.append("Unknown issue type - requires deeper analysis")

        return issues

    def _propose_improvements(
        self,
        issues: List[str],
        context: Dict[str, Any],
    ) -> List[str]:
        """Propose improvements for identified issues."""
        improvements = []

        for issue in issues:
            if "validation" in issue.lower():
                improvements.append("Add input sanitization before validation")
                improvements.append("Review validation rules for edge cases")

            elif "timeout" in issue.lower():
                improvements.append("Increase timeout threshold")
                improvements.append("Add circuit breaker pattern")

            elif "not found" in issue.lower():
                improvements.append("Add existence check before access")
                improvements.append("Implement fallback mechanism")

            elif "recurring" in issue.lower():
                improvements.append("Investigate root cause pattern")
                improvements.append("Consider alternative approach")

        if not improvements:
            improvements.append("Collect more diagnostic information")

        return improvements

    def _generate_analysis(
        self,
        previous_thought: str,
        issues: List[str],
        improvements: List[str],
        context: Dict[str, Any],
    ) -> str:
        """Generate analysis combining thought, issues, and improvements."""
        gate_id = context.get("gate_id", "unknown")

        analysis_parts = [
            f"Reflection on {gate_id}:",
            f"Previous approach: {previous_thought}",
            f"Identified issues ({len(issues)}): {', '.join(issues)}",
            f"Proposed improvements ({len(improvements)}): {', '.join(improvements[:3])}",
        ]

        return " | ".join(analysis_parts)

    async def _evaluate_quality(self, analysis: str) -> float:
        """Evaluate quality of analysis."""
        if self._quality_evaluator:
            try:
                return await self._quality_evaluator(analysis)
            except Exception as e:
                logger.warning(f"Quality evaluator failed: {e}")

        # Heuristic quality scoring
        score = 0.3  # Base score

        # Length bonus (more detail = better)
        if len(analysis) > 100:
            score += 0.1
        if len(analysis) > 200:
            score += 0.1

        # Keyword bonuses
        keywords = ["root cause", "solution", "improvement", "action", "fix"]
        for keyword in keywords:
            if keyword in analysis.lower():
                score += 0.05

        # Structure bonus
        if "|" in analysis:  # Has sections
            score += 0.1

        return min(1.0, score)

    def _synthesize_final_analysis(self) -> str:
        """Synthesize final analysis from all iterations."""
        if not self._iterations:
            return "No iterations completed"

        # Combine best insights from iterations
        all_issues = set()
        all_improvements = set()

        for iteration in self._iterations:
            all_issues.update(iteration["identified_issues"])
            all_improvements.update(iteration["proposed_improvements"])

        return (
            f"Final analysis after {len(self._iterations)} iterations: "
            f"Issues: {list(all_issues)[:5]}. "
            f"Recommended actions: {list(all_improvements)[:5]}."
        )

    def _create_action_plan(self) -> ActionPlan:
        """Create action plan from reflection."""
        if not self._iterations:
            return ActionPlan(
                actions=[],
                priority_order=[],
                estimated_impact=0.0,
                confidence=0.0,
                rationale="No reflection data",
            )

        # Extract unique improvements
        all_improvements = set()
        for iteration in self._iterations:
            all_improvements.update(iteration["proposed_improvements"])

        actions = [
            {"action": imp, "type": "improvement", "effort": "medium"}
            for imp in list(all_improvements)[:5]
        ]

        return ActionPlan(
            actions=actions,
            priority_order=list(range(len(actions))),
            estimated_impact=self._iterations[-1]["quality_score"],
            confidence=self._iterations[-1]["quality_score"] * 0.8,
            rationale=f"Based on {len(self._iterations)} reflection iterations",
        )

    def _get_quality_level(self, score: float) -> ReflectionQuality:
        """Get quality level from score."""
        if score >= 0.9:
            return ReflectionQuality.EXCELLENT
        elif score >= 0.7:
            return ReflectionQuality.GOOD
        elif score >= 0.5:
            return ReflectionQuality.ADEQUATE
        elif score >= 0.3:
            return ReflectionQuality.POOR
        else:
            return ReflectionQuality.INSUFFICIENT


# =============================================================================
# ERROR CORRECTOR
# =============================================================================


class ErrorCorrector:
    """Pattern-based error detection and correction.

    Maintains a registry of known error patterns and their corrections.
    Learns new patterns from successful corrections.
    """

    def __init__(self):
        """Initialize error corrector."""
        self._error_patterns: Dict[str, Dict[str, Any]] = {}
        self._corrections_applied: int = 0
        self._successful_corrections: int = 0

    def register_pattern(
        self,
        pattern_id: str,
        error_type: str,
        matcher: Callable[[Dict[str, Any]], bool],
        corrector: Callable[[Dict[str, Any]], Dict[str, Any]],
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    ) -> None:
        """Register an error pattern with its correction.

        Args:
            pattern_id: Unique pattern identifier
            error_type: Type of error this pattern matches
            matcher: Function to check if error matches pattern
            corrector: Function to apply correction
            severity: Severity of errors matching this pattern
        """
        self._error_patterns[pattern_id] = {
            "error_type": error_type,
            "matcher": matcher,
            "corrector": corrector,
            "severity": severity,
            "match_count": 0,
            "success_count": 0,
        }

    async def analyze_error(
        self,
        error_context: Dict[str, Any],
    ) -> ErrorAnalysis:
        """Analyze an error and identify patterns.

        Args:
            error_context: Context of the error

        Returns:
            ErrorAnalysis with detailed error information
        """
        import uuid

        error_id = str(uuid.uuid4())[:8]
        error_type = error_context.get("error_type", "unknown")
        error_message = str(error_context.get("error", "Unknown error"))

        # Determine severity
        severity = self._determine_severity(error_context)

        # Find matching patterns
        matching_pattern = None
        similar_errors = []

        for pattern_id, pattern in self._error_patterns.items():
            try:
                if pattern["matcher"](error_context):
                    matching_pattern = pattern_id
                    break
            except Exception:
                continue

        # Extract contributing factors
        factors = self._extract_contributing_factors(error_context)

        return ErrorAnalysis(
            error_id=error_id,
            error_type=error_type,
            error_message=error_message,
            severity=severity.value,
            root_cause=self._infer_root_cause(error_context),
            contributing_factors=factors,
            similar_errors=similar_errors,
            pattern_id=matching_pattern,
            context=error_context,
        )

    async def correct_error(
        self,
        error_analysis: ErrorAnalysis,
        state: Dict[str, Any],
    ) -> CorrectionResult:
        """Attempt to correct an error.

        Args:
            error_analysis: Analysis of the error
            state: Current state to correct

        Returns:
            CorrectionResult with correction details
        """
        self._corrections_applied += 1

        pattern_id = error_analysis["pattern_id"]

        if not pattern_id or pattern_id not in self._error_patterns:
            return CorrectionResult(
                error_id=error_analysis["error_id"],
                correction_applied=False,
                correction_type="none",
                original_state=state,
                corrected_state=state,
                confidence=0.0,
                verification_passed=False,
                side_effects=[],
                success=False,
                error="No matching correction pattern found",
            )

        pattern = self._error_patterns[pattern_id]

        try:
            # Apply correction
            original_state = state.copy()
            corrected_state = pattern["corrector"](state)

            # Verify correction
            verification_passed = self._verify_correction(
                original_state,
                corrected_state,
                error_analysis,
            )

            if verification_passed:
                self._successful_corrections += 1
                pattern["success_count"] += 1

            return CorrectionResult(
                error_id=error_analysis["error_id"],
                correction_applied=True,
                correction_type=pattern["error_type"],
                original_state=original_state,
                corrected_state=corrected_state,
                confidence=pattern["success_count"] / max(1, pattern["match_count"]),
                verification_passed=verification_passed,
                side_effects=[],
                success=verification_passed,
                error=None if verification_passed else "Verification failed",
            )

        except Exception as e:
            return CorrectionResult(
                error_id=error_analysis["error_id"],
                correction_applied=False,
                correction_type=pattern["error_type"],
                original_state=state,
                corrected_state=state,
                confidence=0.0,
                verification_passed=False,
                side_effects=[],
                success=False,
                error=str(e),
            )

    def _determine_severity(self, error_context: Dict[str, Any]) -> ErrorSeverity:
        """Determine error severity from context."""
        error_message = str(error_context.get("error", "")).lower()

        if any(word in error_message for word in ["critical", "fatal", "crash"]):
            return ErrorSeverity.CRITICAL

        if any(word in error_message for word in ["fail", "error", "exception"]):
            return ErrorSeverity.HIGH

        if any(word in error_message for word in ["warning", "timeout"]):
            return ErrorSeverity.MEDIUM

        if any(word in error_message for word in ["info", "debug"]):
            return ErrorSeverity.INFO

        return ErrorSeverity.LOW

    def _extract_contributing_factors(
        self,
        error_context: Dict[str, Any],
    ) -> List[str]:
        """Extract contributing factors from error context."""
        factors = []

        if error_context.get("retry_count", 0) > 0:
            factors.append(f"Previous retries: {error_context['retry_count']}")

        if error_context.get("timeout", False):
            factors.append("Timeout occurred")

        if error_context.get("external_dependency", False):
            factors.append("External dependency involved")

        if error_context.get("resource_constraint", False):
            factors.append("Resource constraint detected")

        return factors

    def _infer_root_cause(self, error_context: Dict[str, Any]) -> Optional[str]:
        """Infer root cause from error context."""
        error_message = str(error_context.get("error", "")).lower()

        if "connection" in error_message:
            return "Network connectivity issue"
        if "timeout" in error_message:
            return "Operation timeout - possible resource contention"
        if "validation" in error_message:
            return "Input validation failure"
        if "permission" in error_message:
            return "Permission or access denied"
        if "not found" in error_message:
            return "Missing resource or dependency"

        return None

    def _verify_correction(
        self,
        original_state: Dict[str, Any],
        corrected_state: Dict[str, Any],
        error_analysis: ErrorAnalysis,
    ) -> bool:
        """Verify that correction was effective."""
        # Basic verification - check state changed
        if original_state == corrected_state:
            return False

        # Check that error condition no longer exists
        # (This would need actual verification logic in production)
        return True


# =============================================================================
# FEEDBACK LEARNER
# =============================================================================


class FeedbackLearner:
    """Learns from feedback and persists to A-MEM.

    Tracks patterns of success and failure, building a knowledge
    base of effective strategies.
    """

    def __init__(
        self,
        persistence_enabled: bool = LEARNING_PERSISTENCE_ENABLED,
        redis_url: Optional[str] = None,
    ):
        """Initialize feedback learner.

        Args:
            persistence_enabled: Whether to persist learnings
            redis_url: Redis URL for persistence
        """
        self.persistence_enabled = persistence_enabled
        self.redis_url = redis_url

        self._learnings: Dict[str, LearningEntry] = {}
        self._feedback_records: List[FeedbackRecord] = []
        self._redis_client = None

        if persistence_enabled and REDIS_AVAILABLE and redis_url:
            try:
                self._redis_client = redis.from_url(redis_url)
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")

    async def learn_from_feedback(
        self,
        action: str,
        outcome: str,
        context: Dict[str, Any],
        confidence: float = 0.7,
    ) -> LearningEntry:
        """Learn from feedback about an action.

        Args:
            action: Action that was taken
            outcome: Outcome of the action ("success", "failure", "partial")
            context: Context in which action was taken
            confidence: Confidence in the learning

        Returns:
            LearningEntry with the new learning
        """
        import uuid

        # Generate pattern hash for deduplication
        pattern_hash = self._generate_pattern_hash(action, context)

        if pattern_hash in self._learnings:
            # Update existing learning
            learning = self._learnings[pattern_hash]
            learning["occurrences"] += 1
            learning["last_seen"] = datetime.now(timezone.utc).isoformat()

            # Update confidence based on outcome
            if outcome == "success":
                learning["confidence"] = min(1.0, learning["confidence"] + 0.05)
            elif outcome == "failure":
                learning["confidence"] = max(0.0, learning["confidence"] - 0.1)

        else:
            # Create new learning
            learning_type = (
                LearningType.SUCCESS_PATTERN if outcome == "success"
                else LearningType.ERROR_PATTERN
            )

            learning = LearningEntry(
                learning_id=str(uuid.uuid4())[:8],
                learning_type=learning_type.value,
                pattern={
                    "action": action,
                    "context_keys": list(context.keys()),
                    "context_signature": self._get_context_signature(context),
                },
                context=context,
                outcome=outcome,
                confidence=confidence,
                occurrences=1,
                first_seen=datetime.now(timezone.utc).isoformat(),
                last_seen=datetime.now(timezone.utc).isoformat(),
                metadata={},
            )

            self._learnings[pattern_hash] = learning

        # Record feedback
        feedback_record = FeedbackRecord(
            feedback_id=str(uuid.uuid4())[:8],
            action_taken=action,
            outcome=outcome,
            context=context,
            learning_applied=[pattern_hash],
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._feedback_records.append(feedback_record)

        # Persist if enabled
        if self.persistence_enabled:
            await self._persist_learning(learning)

        return learning

    async def get_relevant_learnings(
        self,
        context: Dict[str, Any],
        min_confidence: float = 0.5,
        limit: int = 10,
    ) -> List[LearningEntry]:
        """Get learnings relevant to a context.

        Args:
            context: Current context
            min_confidence: Minimum confidence threshold
            limit: Maximum learnings to return

        Returns:
            List of relevant LearningEntry
        """
        context_signature = self._get_context_signature(context)
        relevant = []

        for learning in self._learnings.values():
            if learning["confidence"] < min_confidence:
                continue

            # Check context similarity
            learning_signature = learning["pattern"].get("context_signature", "")
            similarity = self._calculate_similarity(
                context_signature,
                learning_signature,
            )

            if similarity > 0.3:
                relevant.append((learning, similarity))

        # Sort by similarity and confidence
        relevant.sort(key=lambda x: x[1] * x[0]["confidence"], reverse=True)

        return [l[0] for l in relevant[:limit]]

    async def get_success_strategies(
        self,
        context: Dict[str, Any],
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Get successful strategies for a context.

        Args:
            context: Current context
            limit: Maximum strategies to return

        Returns:
            List of strategy dictionaries
        """
        learnings = await self.get_relevant_learnings(context, min_confidence=0.6)

        strategies = []
        for learning in learnings:
            if learning["outcome"] == "success":
                strategies.append({
                    "action": learning["pattern"]["action"],
                    "confidence": learning["confidence"],
                    "occurrences": learning["occurrences"],
                    "context_match": self._get_context_signature(context),
                })

        return strategies[:limit]

    def _generate_pattern_hash(
        self,
        action: str,
        context: Dict[str, Any],
    ) -> str:
        """Generate hash for pattern deduplication.

        GAP HUNTER V2 FIX: Use SHA256 instead of MD5.
        """
        signature = f"{action}:{self._get_context_signature(context)}"
        return hashlib.sha256(signature.encode()).hexdigest()[:12]

    def _get_context_signature(self, context: Dict[str, Any]) -> str:
        """Generate context signature for matching."""
        # Use key features of context
        features = []

        if "gate_id" in context:
            features.append(f"gate:{context['gate_id']}")

        if "error_type" in context:
            features.append(f"error:{context['error_type']}")

        if "phase" in context:
            features.append(f"phase:{context['phase']}")

        return "|".join(sorted(features))

    def _calculate_similarity(self, sig1: str, sig2: str) -> float:
        """Calculate similarity between two signatures."""
        if not sig1 or not sig2:
            return 0.0

        parts1 = set(sig1.split("|"))
        parts2 = set(sig2.split("|"))

        intersection = len(parts1 & parts2)
        union = len(parts1 | parts2)

        return intersection / union if union > 0 else 0.0

    async def _persist_learning(self, learning: LearningEntry) -> None:
        """Persist learning to storage."""
        if not self._redis_client:
            return

        try:
            key = f"reflexion:learning:{learning['learning_id']}"
            self._redis_client.setex(
                key,
                timedelta(days=30),  # 30 day TTL
                json.dumps(learning),
            )
        except Exception as e:
            logger.warning(f"Failed to persist learning: {e}")


# =============================================================================
# REFLEXION ENGINE
# =============================================================================


class ReflexionEngine:
    """Main Reflexion engine combining all components.

    Provides a unified interface for self-reflection, error correction,
    and learning from feedback.
    """

    def __init__(
        self,
        max_iterations: int = MAX_REFLECTION_ITERATIONS,
        improvement_threshold: float = IMPROVEMENT_THRESHOLD,
        learning_enabled: bool = LEARNING_PERSISTENCE_ENABLED,
    ):
        """Initialize Reflexion engine.

        Args:
            max_iterations: Maximum reflection iterations
            improvement_threshold: Minimum improvement to continue
            learning_enabled: Whether to enable learning persistence
        """
        self.reflection_loop = SelfReflectionLoop(
            max_iterations=max_iterations,
            improvement_threshold=improvement_threshold,
        )
        self.error_corrector = ErrorCorrector()
        self.feedback_learner = FeedbackLearner(
            persistence_enabled=learning_enabled,
        )

        self._start_time = datetime.now(timezone.utc)
        self._total_reflections = 0
        self._successful_reflections = 0

        # Register default error patterns
        self._register_default_patterns()

    def _register_default_patterns(self) -> None:
        """Register default error patterns."""
        # Validation error pattern
        self.error_corrector.register_pattern(
            pattern_id="validation_error",
            error_type="validation",
            matcher=lambda ctx: "validation" in str(ctx.get("error", "")).lower(),
            corrector=lambda state: {**state, "needs_revalidation": True},
            severity=ErrorSeverity.MEDIUM,
        )

        # Timeout pattern
        self.error_corrector.register_pattern(
            pattern_id="timeout_error",
            error_type="timeout",
            matcher=lambda ctx: "timeout" in str(ctx.get("error", "")).lower(),
            corrector=lambda state: {**state, "timeout_extended": True},
            severity=ErrorSeverity.HIGH,
        )

    async def reflect(
        self,
        failure_context: Dict[str, Any],
        max_iterations: Optional[int] = None,
    ) -> ReflectionResult:
        """Run full reflection cycle on a failure.

        Args:
            failure_context: Context of the failure
            max_iterations: Optional override for max iterations

        Returns:
            ReflectionResult with complete reflection analysis
        """
        self._total_reflections += 1

        # Check for relevant learnings first
        learnings = await self.feedback_learner.get_relevant_learnings(
            failure_context,
            min_confidence=0.5,
        )

        # Add learnings to context
        if learnings:
            failure_context["prior_learnings"] = [
                {"action": l["pattern"]["action"], "outcome": l["outcome"]}
                for l in learnings[:3]
            ]

        # Run reflection loop
        result = await self.reflection_loop.reflect(failure_context)

        if result["success"]:
            self._successful_reflections += 1

        return result

    async def analyze_and_correct(
        self,
        error_context: Dict[str, Any],
        state: Dict[str, Any],
    ) -> CorrectionResult:
        """Analyze error and attempt correction.

        Args:
            error_context: Context of the error
            state: Current state to correct

        Returns:
            CorrectionResult with correction details
        """
        # Analyze error
        analysis = await self.error_corrector.analyze_error(error_context)

        # Attempt correction
        result = await self.error_corrector.correct_error(analysis, state)

        # Learn from result
        await self.feedback_learner.learn_from_feedback(
            action="error_correction",
            outcome="success" if result["success"] else "failure",
            context=error_context,
        )

        return result

    async def learn(
        self,
        action: str,
        outcome: str,
        context: Dict[str, Any],
    ) -> LearningEntry:
        """Record learning from an action outcome.

        Args:
            action: Action that was taken
            outcome: Outcome of the action
            context: Context in which action was taken

        Returns:
            LearningEntry with the recorded learning
        """
        return await self.feedback_learner.learn_from_feedback(
            action=action,
            outcome=outcome,
            context=context,
        )

    def get_metrics(self) -> ReflexionMetrics:
        """Get Reflexion system metrics.

        Returns:
            ReflexionMetrics with system statistics
        """
        uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds()

        return ReflexionMetrics(
            total_reflections=self._total_reflections,
            successful_reflections=self._successful_reflections,
            average_iterations=0.0,  # Would need to track
            average_quality=0.0,  # Would need to track
            total_corrections=self.error_corrector._corrections_applied,
            successful_corrections=self.error_corrector._successful_corrections,
            total_learnings=len(self.feedback_learner._learnings),
            learning_applications=len(self.feedback_learner._feedback_records),
            uptime_seconds=uptime,
        )


# =============================================================================
# SINGLETON
# =============================================================================

_reflexion_engine: Optional[ReflexionEngine] = None


def get_reflexion_engine(
    max_iterations: Optional[int] = None,
    learning_enabled: bool = LEARNING_PERSISTENCE_ENABLED,
) -> ReflexionEngine:
    """Get singleton Reflexion engine instance.

    Args:
        max_iterations: Optional override for max iterations
        learning_enabled: Whether to enable learning

    Returns:
        ReflexionEngine instance
    """
    global _reflexion_engine
    if _reflexion_engine is None:
        _reflexion_engine = ReflexionEngine(
            max_iterations=max_iterations or MAX_REFLECTION_ITERATIONS,
            learning_enabled=learning_enabled,
        )
    return _reflexion_engine


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


async def run_self_reflection(
    failure_context: Dict[str, Any],
    max_iterations: int = MAX_REFLECTION_ITERATIONS,
) -> ReflectionResult:
    """Run self-reflection on a failure.

    Convenience function for quick reflection.
    """
    engine = get_reflexion_engine()
    return await engine.reflect(failure_context, max_iterations)


async def analyze_and_correct_error(
    error_context: Dict[str, Any],
    state: Dict[str, Any],
) -> CorrectionResult:
    """Analyze and correct an error.

    Convenience function for quick error correction.
    """
    engine = get_reflexion_engine()
    return await engine.analyze_and_correct(error_context, state)


async def learn_from_feedback(
    action: str,
    outcome: str,
    context: Dict[str, Any],
) -> LearningEntry:
    """Learn from feedback.

    Convenience function for quick learning.
    """
    engine = get_reflexion_engine()
    return await engine.learn(action, outcome, context)


def get_reflexion_metrics() -> ReflexionMetrics:
    """Get Reflexion system metrics.

    Convenience function for metrics.
    """
    engine = get_reflexion_engine()
    return engine.get_metrics()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
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
]
