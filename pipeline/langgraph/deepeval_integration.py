"""DeepEval Integration for LLM Output Evaluation.

This module provides REAL integration with DeepEval to evaluate LLM outputs
for quality, relevance, faithfulness, and toxicity.

Every LLM output SHOULD be evaluated. This is not optional.

Key Metrics:
1. ANSWER_RELEVANCY - Is the output relevant to the input?
2. FAITHFULNESS - Does the output stay true to provided context?
3. CONTEXTUAL_RELEVANCY - Is the retrieved context relevant?
4. HALLUCINATION - Does the output contain fabricated information?
5. TOXICITY - Is the output toxic or harmful?

Usage:
    from pipeline.langgraph.deepeval_integration import (
        evaluate_llm_output,
        evaluate_with_context,
        get_evaluation_metrics,
    )

    # Simple evaluation
    result = evaluate_llm_output(
        input_text="What is Python?",
        output_text="Python is a programming language.",
    )
    print(f"Score: {result.overall_score}")

    # RAG evaluation with context
    result = evaluate_with_context(
        input_text="What is Python?",
        output_text="Python is a programming language.",
        retrieval_context=["Python is a high-level programming language."],
    )

Author: Pipeline Autonomo Team
Version: 1.0.0 (2026-01-16)
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURABLE THRESHOLDS (Bloco 4 FIX)
# =============================================================================

# Default thresholds - can be overridden via environment variables
DEEPEVAL_RELEVANCY_THRESHOLD = float(os.getenv("DEEPEVAL_RELEVANCY_THRESHOLD", "0.5"))
DEEPEVAL_FAITHFULNESS_THRESHOLD = float(os.getenv("DEEPEVAL_FAITHFULNESS_THRESHOLD", "0.5"))
DEEPEVAL_HALLUCINATION_THRESHOLD = float(os.getenv("DEEPEVAL_HALLUCINATION_THRESHOLD", "0.5"))
DEEPEVAL_TOXICITY_THRESHOLD = float(os.getenv("DEEPEVAL_TOXICITY_THRESHOLD", "0.3"))
DEEPEVAL_BIAS_THRESHOLD = float(os.getenv("DEEPEVAL_BIAS_THRESHOLD", "0.5"))


# =============================================================================
# EVALUATION METRICS
# =============================================================================


class EvaluationMetric(Enum):
    """Available evaluation metrics."""
    ANSWER_RELEVANCY = "answer_relevancy"
    FAITHFULNESS = "faithfulness"
    CONTEXTUAL_RELEVANCY = "contextual_relevancy"
    HALLUCINATION = "hallucination"
    TOXICITY = "toxicity"
    BIAS = "bias"
    SUMMARIZATION = "summarization"


@dataclass
class MetricResult:
    """Result for a single metric."""
    metric: EvaluationMetric
    score: float  # 0.0 to 1.0
    passed: bool
    reason: str = ""
    threshold: float = 0.5


@dataclass
class EvaluationResult:
    """Complete evaluation result for an LLM output."""

    input_text: str
    output_text: str
    retrieval_context: List[str] = field(default_factory=list)

    # Scores
    overall_score: float = 0.0
    metrics: List[MetricResult] = field(default_factory=list)

    # Status
    passed: bool = False
    evaluation_error: Optional[str] = None

    # Metadata
    model_used: str = ""
    evaluation_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "input_text": self.input_text[:500],  # Truncate for logging
            "output_text": self.output_text[:500],
            "overall_score": self.overall_score,
            "passed": self.passed,
            "metrics": [
                {
                    "metric": m.metric.value,
                    "score": m.score,
                    "passed": m.passed,
                    "reason": m.reason[:200],
                }
                for m in self.metrics
            ],
            "evaluation_error": self.evaluation_error,
            "model_used": self.model_used,
            "evaluation_time_ms": self.evaluation_time_ms,
        }


# =============================================================================
# DEEPEVAL WRAPPER
# =============================================================================


class DeepEvalWrapper:
    """Wrapper around DeepEval for LLM output evaluation.

    Uses singleton pattern to avoid re-initializing DeepEval.
    """

    _instance: Optional['DeepEvalWrapper'] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> 'DeepEvalWrapper':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._deepeval = None
        self._metrics_cache: Dict[str, Any] = {}
        self._initialized = True

        # Try to import DeepEval
        self._init_deepeval()

    def _init_deepeval(self):
        """Initialize DeepEval metrics with Claude as the evaluation model."""
        try:
            import deepeval
            from deepeval.metrics import (
                AnswerRelevancyMetric,
                FaithfulnessMetric,
                ContextualRelevancyMetric,
                HallucinationMetric,
                ToxicityMetric,
            )
            from deepeval.test_case import LLMTestCase
            from deepeval.models import DeepEvalBaseLLM

            self._deepeval = deepeval
            self._LLMTestCase = LLMTestCase

            # Create custom Claude model for DeepEval
            claude_model = self._create_claude_model(DeepEvalBaseLLM)

            # If no model available, disable DeepEval
            if claude_model is None:
                logger.info("DeepEval disabled: No API key available")
                self._deepeval = None
                return

            # Pre-create metric instances using Claude
            # Bloco 4 FIX: Use configurable thresholds from environment
            self._metrics_cache = {
                EvaluationMetric.ANSWER_RELEVANCY: AnswerRelevancyMetric(
                    threshold=DEEPEVAL_RELEVANCY_THRESHOLD,
                    model=claude_model,  # Use Claude instead of GPT
                    include_reason=True,
                ),
                EvaluationMetric.FAITHFULNESS: FaithfulnessMetric(
                    threshold=DEEPEVAL_FAITHFULNESS_THRESHOLD,
                    model=claude_model,
                    include_reason=True,
                ),
                EvaluationMetric.CONTEXTUAL_RELEVANCY: ContextualRelevancyMetric(
                    threshold=DEEPEVAL_RELEVANCY_THRESHOLD,
                    model=claude_model,
                    include_reason=True,
                ),
                EvaluationMetric.HALLUCINATION: HallucinationMetric(
                    threshold=DEEPEVAL_HALLUCINATION_THRESHOLD,
                    model=claude_model,
                    include_reason=True,
                ),
                EvaluationMetric.TOXICITY: ToxicityMetric(
                    threshold=DEEPEVAL_TOXICITY_THRESHOLD,
                    model=claude_model,
                    include_reason=True,
                ),
            }

            logger.info("DeepEval initialized with 5 metrics using Claude")

        except ImportError as e:
            logger.warning(f"DeepEval not available: {e}")
            self._deepeval = None
        except Exception as e:
            logger.error(f"Failed to initialize DeepEval: {e}")
            self._deepeval = None

    def _create_claude_model(self, base_class):
        """Create a custom Claude model for DeepEval.

        Uses Anthropic's Claude with instructor for structured outputs.
        """
        try:
            from anthropic import Anthropic
            import instructor
            from pydantic import BaseModel
            import os
        except ImportError as e:
            logger.warning(f"Cannot create Claude model for DeepEval: {e}")
            return None  # No fallback - return None to disable evaluation

        # Check if ANTHROPIC_API_KEY is available
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.info("DeepEval: ANTHROPIC_API_KEY not set, evaluation metrics disabled")
            return None

        class CustomClaudeHaiku(base_class):
            """Custom Claude model for DeepEval evaluation.

            Uses Claude 3.5 Haiku for fast, cost-effective evaluation.
            """
            def __init__(self):
                self.model_name = "claude-3-5-haiku-20241022"
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if api_key:
                    self._client = Anthropic(api_key=api_key)
                else:
                    self._client = Anthropic()  # Will use env var

            def load_model(self):
                return self._client

            def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
                """Generate structured output using Claude with instructor."""
                client = self.load_model()
                instructor_client = instructor.from_anthropic(client)
                resp = instructor_client.messages.create(
                    model=self.model_name,
                    max_tokens=1024,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    response_model=schema,
                )
                return resp

            async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
                """Async generate - uses sync for simplicity."""
                return self.generate(prompt, schema)

            def get_model_name(self):
                return "Claude-3.5-Haiku"

        try:
            return CustomClaudeHaiku()
        except Exception as e:
            logger.warning(f"Failed to create Claude model: {e}, evaluation disabled")
            return None

    @property
    def available(self) -> bool:
        """Check if DeepEval is available."""
        return self._deepeval is not None

    def evaluate(
        self,
        input_text: str,
        output_text: str,
        retrieval_context: Optional[List[str]] = None,
        metrics: Optional[List[EvaluationMetric]] = None,
    ) -> EvaluationResult:
        """Evaluate an LLM output.

        Args:
            input_text: The input/prompt sent to the LLM.
            output_text: The output received from the LLM.
            retrieval_context: Optional list of context chunks (for RAG).
            metrics: Specific metrics to evaluate. If None, uses defaults.

        Returns:
            EvaluationResult with scores and pass/fail status.
        """
        import time
        start_time = time.time()

        result = EvaluationResult(
            input_text=input_text,
            output_text=output_text,
            retrieval_context=retrieval_context or [],
        )

        if not self.available:
            # Fallback evaluation without DeepEval
            result = self._evaluate_fallback(input_text, output_text, retrieval_context)
            result.evaluation_time_ms = (time.time() - start_time) * 1000
            return result

        try:
            # Create test case
            test_case = self._LLMTestCase(
                input=input_text,
                actual_output=output_text,
                retrieval_context=retrieval_context or [],
            )

            # Select metrics to evaluate
            metrics_to_run = metrics or [
                EvaluationMetric.ANSWER_RELEVANCY,
                EvaluationMetric.TOXICITY,
            ]

            # Add RAG metrics if context is provided
            if retrieval_context:
                metrics_to_run.extend([
                    EvaluationMetric.FAITHFULNESS,
                    EvaluationMetric.HALLUCINATION,
                ])

            # Run evaluations
            metric_results = []
            total_score = 0.0
            all_passed = True

            for metric_type in metrics_to_run:
                if metric_type not in self._metrics_cache:
                    continue

                metric = self._metrics_cache[metric_type]
                try:
                    metric.measure(test_case)

                    metric_result = MetricResult(
                        metric=metric_type,
                        score=metric.score,
                        passed=metric.is_successful(),
                        reason=metric.reason or "",
                        threshold=metric.threshold,
                    )
                    metric_results.append(metric_result)
                    total_score += metric.score

                    if not metric.is_successful():
                        all_passed = False

                except Exception as e:
                    logger.warning(f"Metric {metric_type.value} failed: {e}")
                    metric_results.append(MetricResult(
                        metric=metric_type,
                        score=0.0,
                        passed=False,
                        reason=f"Evaluation failed: {e}",
                    ))
                    all_passed = False

            # Calculate overall score
            if metric_results:
                result.overall_score = total_score / len(metric_results)

            result.metrics = metric_results
            result.passed = all_passed
            result.model_used = "deepeval"

        except Exception as e:
            logger.error(f"DeepEval evaluation failed: {e}")
            result.evaluation_error = str(e)
            result.passed = False

        result.evaluation_time_ms = (time.time() - start_time) * 1000
        return result

    def _evaluate_fallback(
        self,
        input_text: str,
        output_text: str,
        retrieval_context: Optional[List[str]],
    ) -> EvaluationResult:
        """Fallback evaluation when DeepEval is not available.

        Uses simple heuristics:
        - Output length check
        - Input/output overlap check
        - Basic toxicity patterns
        """
        result = EvaluationResult(
            input_text=input_text,
            output_text=output_text,
            retrieval_context=retrieval_context or [],
        )

        metrics = []

        # 1. Basic relevancy check (word overlap)
        input_words = set(input_text.lower().split())
        output_words = set(output_text.lower().split())
        overlap = len(input_words & output_words)
        relevancy_score = min(overlap / max(len(input_words), 1), 1.0)

        # Bloco 4 FIX: Use configurable thresholds
        metrics.append(MetricResult(
            metric=EvaluationMetric.ANSWER_RELEVANCY,
            score=relevancy_score,
            passed=relevancy_score > DEEPEVAL_RELEVANCY_THRESHOLD,
            reason=f"Word overlap: {overlap} words",
            threshold=DEEPEVAL_RELEVANCY_THRESHOLD,
        ))

        # 2. Output length check
        length_score = min(len(output_text) / 100, 1.0)  # Expect at least 100 chars
        metrics.append(MetricResult(
            metric=EvaluationMetric.FAITHFULNESS,
            score=length_score,
            passed=length_score > DEEPEVAL_FAITHFULNESS_THRESHOLD,
            reason=f"Output length: {len(output_text)} chars",
            threshold=DEEPEVAL_FAITHFULNESS_THRESHOLD,
        ))

        # 3. Basic toxicity check (simple patterns)
        toxic_patterns = ["kill", "hate", "stupid", "idiot", "die"]
        toxicity_score = 1.0
        for pattern in toxic_patterns:
            if pattern in output_text.lower():
                toxicity_score -= 0.2
        toxicity_score = max(toxicity_score, 0.0)

        # Bloco 4 FIX: Use configurable threshold
        metrics.append(MetricResult(
            metric=EvaluationMetric.TOXICITY,
            score=toxicity_score,
            passed=toxicity_score > DEEPEVAL_TOXICITY_THRESHOLD,
            reason="Basic toxicity check",
            threshold=DEEPEVAL_TOXICITY_THRESHOLD,
        ))

        # Calculate overall
        result.metrics = metrics
        result.overall_score = sum(m.score for m in metrics) / len(metrics)
        result.passed = all(m.passed for m in metrics)
        result.model_used = "fallback_heuristics"

        return result

    def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]],
        metrics: Optional[List[EvaluationMetric]] = None,
    ) -> List[EvaluationResult]:
        """Evaluate multiple LLM outputs.

        Args:
            test_cases: List of dicts with input_text, output_text, retrieval_context.
            metrics: Metrics to evaluate.

        Returns:
            List of EvaluationResult objects.
        """
        results = []
        for case in test_cases:
            result = self.evaluate(
                input_text=case.get("input_text", ""),
                output_text=case.get("output_text", ""),
                retrieval_context=case.get("retrieval_context"),
                metrics=metrics,
            )
            results.append(result)
        return results


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


_deepeval_instance: Optional[DeepEvalWrapper] = None


def get_deepeval() -> DeepEvalWrapper:
    """Get the singleton DeepEval wrapper instance."""
    global _deepeval_instance
    if _deepeval_instance is None:
        _deepeval_instance = DeepEvalWrapper()
    return _deepeval_instance


def evaluate_llm_output(
    input_text: str,
    output_text: str,
    metrics: Optional[List[EvaluationMetric]] = None,
) -> EvaluationResult:
    """Evaluate an LLM output for quality.

    This is the primary function to use for evaluating LLM outputs.

    Args:
        input_text: The input/prompt sent to the LLM.
        output_text: The output received from the LLM.
        metrics: Specific metrics to evaluate.

    Returns:
        EvaluationResult with scores and status.

    Example:
        result = evaluate_llm_output(
            input_text="What is Python?",
            output_text="Python is a programming language.",
        )
        if result.passed:
            print("Output quality is acceptable")
        else:
            print(f"Output quality issues: {result.overall_score}")
    """
    return get_deepeval().evaluate(
        input_text=input_text,
        output_text=output_text,
        metrics=metrics,
    )


def evaluate_with_context(
    input_text: str,
    output_text: str,
    retrieval_context: List[str],
    metrics: Optional[List[EvaluationMetric]] = None,
) -> EvaluationResult:
    """Evaluate an LLM output in a RAG context.

    Use this for RAG applications where context is provided to the LLM.

    Args:
        input_text: The input/prompt sent to the LLM.
        output_text: The output received from the LLM.
        retrieval_context: The context chunks provided to the LLM.
        metrics: Specific metrics to evaluate.

    Returns:
        EvaluationResult with RAG-specific scores.
    """
    return get_deepeval().evaluate(
        input_text=input_text,
        output_text=output_text,
        retrieval_context=retrieval_context,
        metrics=metrics or [
            EvaluationMetric.ANSWER_RELEVANCY,
            EvaluationMetric.FAITHFULNESS,
            EvaluationMetric.CONTEXTUAL_RELEVANCY,
            EvaluationMetric.HALLUCINATION,
        ],
    )


def get_evaluation_metrics() -> List[str]:
    """Get list of available evaluation metrics."""
    return [m.value for m in EvaluationMetric]


def is_deepeval_available() -> bool:
    """Check if DeepEval is available."""
    return get_deepeval().available


# =============================================================================
# DECORATOR FOR AUTOMATIC EVALUATION
# =============================================================================


def evaluate_output(
    min_score: float = 0.5,
    metrics: Optional[List[EvaluationMetric]] = None,
    fail_on_low_score: bool = False,
):
    """Decorator to automatically evaluate LLM function outputs.

    Usage:
        @evaluate_output(min_score=0.6)
        def my_llm_function(prompt: str) -> str:
            # Call LLM
            return response

    The decorator will:
    1. Call the function
    2. Evaluate the output
    3. Log the evaluation result
    4. Optionally raise an error if score is too low

    Args:
        min_score: Minimum acceptable score (0.0 to 1.0).
        metrics: Metrics to evaluate.
        fail_on_low_score: If True, raise exception on low scores.
    """
    import functools

    def decorator(func):
        @functools.wraps(func)
        def wrapper(prompt: str, *args, **kwargs):
            # Call the function
            output = func(prompt, *args, **kwargs)

            # Evaluate
            result = evaluate_llm_output(
                input_text=prompt,
                output_text=output if isinstance(output, str) else str(output),
                metrics=metrics,
            )

            # Log result
            logger.info(
                f"LLM output evaluation: score={result.overall_score:.2f}, "
                f"passed={result.passed}, metrics={len(result.metrics)}"
            )

            # Optionally fail
            if fail_on_low_score and result.overall_score < min_score:
                raise ValueError(
                    f"LLM output score ({result.overall_score:.2f}) "
                    f"below minimum ({min_score})"
                )

            return output

        return wrapper
    return decorator


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Core classes
    "DeepEvalWrapper",
    "EvaluationResult",
    "MetricResult",
    "EvaluationMetric",

    # Functions
    "get_deepeval",
    "evaluate_llm_output",
    "evaluate_with_context",
    "get_evaluation_metrics",
    "is_deepeval_available",

    # Decorator
    "evaluate_output",
]
