"""Stack Auto-Wiring - Automatic Stack Integration System.

This module ensures that stacks are ACTUALLY USED, not just available.
It auto-wires stacks into operations where they SHOULD be used.

The problem: Stacks can be available but not used.
The solution: Auto-wire stacks into every operation where they add value.

Auto-Wiring Rules:
- EVERY LLM call -> langfuse.trace()
- EVERY failure -> got.analyze() + reflexion.reflect() + letta.save()
- EVERY search -> qdrant.search()
- EVERY output -> deepeval.evaluate()
- EVERY decision -> got.analyze()
- EVERY learning -> letta.save()

Author: Pipeline Autonomo Team
Version: 1.0.0 (2026-01-16)
"""

from __future__ import annotations

import logging
import functools
import threading
import time
from typing import Any, Callable, Dict, List, Optional, TypeVar
from dataclasses import dataclass
from contextlib import contextmanager

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# AUTO-WIRING RULES
# =============================================================================

@dataclass
class AutoWireRule:
    """Rule for automatic stack wiring."""

    trigger: str  # Operation type that triggers this rule
    stack: str  # Stack to auto-wire
    method: str  # Method to call on the stack
    when: str  # "before", "after", "on_error", "always"
    priority: int = 0  # Higher priority runs first
    required: bool = False  # If True, block if stack unavailable
    extract_args: Optional[Callable] = None  # Function to extract args from context


# Define all auto-wiring rules
AUTO_WIRE_RULES: List[AutoWireRule] = [
    # ==========================================================================
    # OBSERVABILITY - Langfuse on EVERYTHING
    # ==========================================================================
    AutoWireRule(
        trigger="llm_call",
        stack="langfuse",
        method="trace_generation",
        when="always",
        priority=100,
        required=False,  # Graceful degradation if stack not available
    ),
    AutoWireRule(
        trigger="agent_execution",
        stack="langfuse",
        method="trace_agent",
        when="always",
        priority=100,
        required=False,  # Graceful degradation if stack not available
    ),
    AutoWireRule(
        trigger="gate_execution",
        stack="langfuse",
        method="trace_gate",
        when="always",
        priority=100,
        required=False,  # Graceful degradation if stack not available
    ),

    # ==========================================================================
    # REASONING - GoT for analysis
    # ==========================================================================
    AutoWireRule(
        trigger="gate_failure",
        stack="got",
        method="analyze_failure",
        when="on_error",
        priority=90,
        required=False,
    ),
    AutoWireRule(
        trigger="complex_decision",
        stack="got",
        method="multi_perspective_analysis",
        when="before",
        priority=80,
        required=False,
    ),
    AutoWireRule(
        trigger="root_cause_analysis",
        stack="got",
        method="trace_root_cause",
        when="before",
        priority=80,
        required=False,
    ),

    # ==========================================================================
    # LEARNING - Reflexion on failures
    # ==========================================================================
    AutoWireRule(
        trigger="gate_failure",
        stack="reflexion",
        method="reflect_on_failure",
        when="on_error",
        priority=85,
        required=False,
    ),
    AutoWireRule(
        trigger="task_failure",
        stack="reflexion",
        method="reflect_on_failure",
        when="on_error",
        priority=85,
        required=False,
    ),
    AutoWireRule(
        trigger="retry",
        stack="reflexion",
        method="retry_with_learning",
        when="before",
        priority=85,
        required=False,
    ),

    # ==========================================================================
    # MEMORY - Letta for persistence
    # ==========================================================================
    AutoWireRule(
        trigger="learning",
        stack="letta",
        method="save_learning",
        when="after",
        priority=70,
        required=False,
    ),
    AutoWireRule(
        trigger="gate_failure",
        stack="letta",
        method="save_failure_context",
        when="after",
        priority=70,
        required=False,
    ),
    AutoWireRule(
        trigger="decision",
        stack="letta",
        method="save_decision",
        when="after",
        priority=70,
        required=False,
    ),
    AutoWireRule(
        trigger="agent_start",
        stack="letta",
        method="recall_context",
        when="before",
        priority=95,
        required=False,
    ),

    # ==========================================================================
    # EVALUATION - DeepEval for quality
    # ==========================================================================
    AutoWireRule(
        trigger="llm_output",
        stack="deepeval",
        method="evaluate_response",
        when="after",
        priority=60,
        required=False,
    ),
    AutoWireRule(
        trigger="generation",
        stack="deepeval",
        method="score_generation",
        when="after",
        priority=60,
        required=False,
    ),

    # ==========================================================================
    # SEARCH - Qdrant for semantic
    # ==========================================================================
    AutoWireRule(
        trigger="context_retrieval",
        stack="qdrant",
        method="semantic_search",
        when="before",
        priority=75,
        required=False,
    ),
    AutoWireRule(
        trigger="similar_search",
        stack="qdrant",
        method="search",
        when="before",
        priority=75,
        required=False,
    ),

    # ==========================================================================
    # KNOWLEDGE GRAPH - FalkorDB for relations
    # ==========================================================================
    AutoWireRule(
        trigger="dependency_analysis",
        stack="falkordb",
        method="query_dependencies",
        when="before",
        priority=70,
        required=False,
    ),
    AutoWireRule(
        trigger="impact_analysis",
        stack="falkordb",
        method="query_impact",
        when="before",
        priority=70,
        required=False,
    ),

    # ==========================================================================
    # SECURITY - NeMo for guardrails
    # ==========================================================================
    AutoWireRule(
        trigger="llm_call",
        stack="nemo",
        method="apply_rails",
        when="before",
        priority=99,
        required=False,
    ),
    AutoWireRule(
        trigger="user_input",
        stack="nemo",
        method="validate_input",
        when="before",
        priority=99,
        required=False,
    ),

    # ==========================================================================
    # EVALUATION - DeepEval for quality scoring
    # ==========================================================================
    AutoWireRule(
        trigger="llm_output",
        stack="deepeval",
        method="evaluate_llm_output",
        when="after",
        priority=50,
        required=False,
    ),
    AutoWireRule(
        trigger="generation",
        stack="deepeval",
        method="evaluate_llm_output",
        when="after",
        priority=50,
        required=False,
    ),
    AutoWireRule(
        trigger="rag_response",
        stack="deepeval",
        method="evaluate_with_context",
        when="after",
        priority=50,
        required=False,
    ),

    # ==========================================================================
    # REASONING - Quiet Star for silent thinking
    # 2026-01-20: STACK AUDIT - Added missing integration
    # ==========================================================================
    AutoWireRule(
        trigger="complex_reasoning",
        stack="quiet_star",
        method="generate_with_thinking",
        when="before",
        priority=85,
        required=False,
    ),
    AutoWireRule(
        trigger="answer_generation",
        stack="quiet_star",
        method="generate_with_thinking",
        when="before",
        priority=80,
        required=False,
    ),

    # ==========================================================================
    # EVALUATION - RAGAS for RAG quality
    # 2026-01-20: STACK AUDIT - Added missing integration
    # ==========================================================================
    AutoWireRule(
        trigger="rag_response",
        stack="ragas",
        method="evaluate_rag_response",
        when="after",
        priority=55,
        required=False,
    ),
    AutoWireRule(
        trigger="context_retrieval",
        stack="ragas",
        method="evaluate_context_relevance",
        when="after",
        priority=55,
        required=False,
    ),

    # ==========================================================================
    # TRACING - Phoenix for LLM observability
    # 2026-01-20: STACK AUDIT - Added missing integration
    # ==========================================================================
    AutoWireRule(
        trigger="llm_call",
        stack="phoenix",
        method="start_trace",
        when="before",
        priority=95,
        required=False,
    ),
    AutoWireRule(
        trigger="embedding_generation",
        stack="phoenix",
        method="detect_embedding_drift",
        when="after",
        priority=50,
        required=False,
    ),

    # ==========================================================================
    # GUARDRAILS - Stack enforcement and circuit breakers
    # ==========================================================================
    AutoWireRule(
        trigger="gate_execution",
        stack="guardrails",
        method="enforce_mandatory",  # FIX CONSISTENCY-03: was enforce_mandatory_stacks
        when="before",
        priority=98,
        required=False,
        extract_args=lambda ctx: {
            "operation": ctx.get("gate_id", ctx.get("operation", "gate_execution")),
            "used_stacks": set(ctx.get("used_stacks", ctx.get("active_stacks", []))),
        },
    ),
    AutoWireRule(
        trigger="llm_call",
        stack="guardrails",
        method="protected_stack_call",
        when="before",
        priority=98,
        required=False,
    ),
    AutoWireRule(
        trigger="stack_call",
        stack="guardrails",
        method="protected_stack_call",
        when="before",
        priority=98,
        required=False,
    ),
]


# =============================================================================
# AUTO-WIRE EXECUTOR
# =============================================================================


class StackAutoWire:
    """Automatic stack wiring system.

    This class ensures stacks are USED, not just available.
    It intercepts operations and applies the right stacks automatically.

    Usage:
        autowire = StackAutoWire()

        # Wrap a function to auto-wire stacks
        @autowire.wire("llm_call")
        def my_llm_function():
            ...

        # Or use context manager
        with autowire.operation("gate_failure", context={"gate": "G3"}):
            handle_failure()
    """

    _instance: Optional['StackAutoWire'] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> 'StackAutoWire':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._stack_injector = None
        self._rules = sorted(AUTO_WIRE_RULES, key=lambda r: -r.priority)
        self._stack_cache: Dict[str, Any] = {}
        self._execution_stats: Dict[str, Dict[str, int]] = {}
        self._initialized = True

        logger.info(f"StackAutoWire initialized with {len(self._rules)} rules")

    def _get_stack_injector(self):
        """Lazy load stack injector."""
        if self._stack_injector is None:
            try:
                from pipeline.langgraph.stack_injection import get_stack_injector
                self._stack_injector = get_stack_injector()
            except ImportError as e:
                logger.error(f"Could not load StackInjector: {e}")
        return self._stack_injector

    def _get_stack(self, stack_name: str) -> Optional[Any]:
        """Get a stack client, with caching.

        Handles both external stacks (via stack_injector) and local modules.
        """
        if stack_name not in self._stack_cache:
            # First try local modules (guardrails, deepeval, nemo_rails)
            local_stack = self._get_local_stack(stack_name)
            if local_stack is not None:
                self._stack_cache[stack_name] = local_stack
            else:
                # Fall back to stack injector for external stacks
                injector = self._get_stack_injector()
                if injector:
                    try:
                        self._stack_cache[stack_name] = injector.get_stack(stack_name)
                    except Exception as e:
                        logger.debug(f"Could not get stack '{stack_name}': {e}")
                        self._stack_cache[stack_name] = None
                else:
                    self._stack_cache[stack_name] = None
        return self._stack_cache[stack_name]

    def _get_local_stack(self, stack_name: str) -> Optional[Any]:
        """Get a local module as a stack.

        These are modules in pipeline/langgraph that provide stack-like functionality.
        """
        if stack_name == "guardrails":
            try:
                from pipeline.langgraph.stack_guardrails import get_guardrails
                return get_guardrails()
            except ImportError:
                logger.debug("stack_guardrails module not available")
                return None

        if stack_name == "deepeval":
            try:
                from pipeline.langgraph.deepeval_integration import get_deepeval
                return get_deepeval()
            except ImportError:
                logger.debug("deepeval_integration module not available")
                return None

        if stack_name == "nemo_rails":
            try:
                from pipeline.langgraph.nemo_stack_rails import get_nemo_rails
                return get_nemo_rails()
            except ImportError:
                logger.debug("nemo_stack_rails module not available")
                return None

        # 2026-01-20: STACK AUDIT - Added missing stack loaders
        if stack_name == "quiet_star":
            try:
                from pipeline.reasoning.quiet_star import get_quiet_star
                return get_quiet_star()
            except ImportError:
                logger.debug("quiet_star module not available")
                return None

        if stack_name == "ragas":
            try:
                from pipeline.evaluation.ragas_eval import RAGASEvaluator
                return RAGASEvaluator()
            except ImportError:
                logger.debug("ragas_eval module not available")
                return None

        if stack_name == "phoenix":
            try:
                from pipeline.evaluation.phoenix_traces import PhoenixTracer
                return PhoenixTracer()
            except ImportError:
                logger.debug("phoenix_traces module not available")
                return None

        # 2026-01-20: FIX - Added GoT loader (was importing from wrong path)
        if stack_name == "got":
            try:
                # GoT is in pipeline_autonomo, NOT pipeline.langgraph
                from pipeline.got_integration import (
                    decompose_spec_with_got,
                    validate_with_got,
                    analyze_gate_failure_with_got,
                )
                # Return a dict-based stack with the main functions
                return {
                    "analyze_failure": analyze_gate_failure_with_got,
                    "multi_perspective_analysis": validate_with_got,
                    "decompose": decompose_spec_with_got,
                    "analyze": validate_with_got,
                }
            except ImportError as e:
                logger.debug(f"got_integration module not available: {e}")
                return None

        # 2026-01-20: FIX - Added Reflexion loader (was importing from wrong path)
        if stack_name == "reflexion":
            try:
                # Reflexion is in pipeline.langgraph.reflexion, NOT reflexion_integration
                from pipeline.langgraph.reflexion.engine import (
                    REFLEXION_AVAILABLE,
                    run_self_reflection,
                    analyze_and_correct_error,
                    learn_from_feedback,
                )
                if REFLEXION_AVAILABLE:
                    # Use module-level async functions wrapped for sync context
                    import asyncio

                    def sync_reflect(**kwargs):
                        """Sync wrapper for async reflect."""
                        try:
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                # Already in async context, just log
                                logger.debug("Reflexion called in async context")
                                return {"status": "deferred", "context": kwargs}
                            return loop.run_until_complete(
                                run_self_reflection(kwargs.get("context", {}))
                            )
                        except Exception as e:
                            logger.warning(f"Reflexion sync wrapper failed: {e}")
                            return {"status": "error", "error": str(e)}

                    return {
                        "reflect_on_failure": sync_reflect,
                        "reflect": sync_reflect,
                        "retry_with_learning": sync_reflect,
                    }
            except ImportError as e:
                logger.debug(f"reflexion.engine module not available: {e}")

            # Fallback: return a simple dict that logs reflexion triggers
            def noop_reflect(**kwargs):
                logger.info(f"Reflexion triggered (no-op): {kwargs.get('context', {}).get('_trigger', 'unknown')}")
                return {"status": "logged", "context": kwargs}

            return {
                "reflect_on_failure": noop_reflect,
                "reflect": noop_reflect,
                "retry_with_learning": noop_reflect,
            }

        return None

    def _get_rules_for_trigger(self, trigger: str, when: str) -> List[AutoWireRule]:
        """Get applicable rules for a trigger and timing."""
        return [
            r for r in self._rules
            if r.trigger == trigger and r.when in (when, "always")
        ]

    def _execute_rule(
        self,
        rule: AutoWireRule,
        context: Dict[str, Any],
        result: Any = None,
        error: Optional[Exception] = None,
    ) -> Optional[Any]:
        """Execute a single auto-wire rule."""
        stack = self._get_stack(rule.stack)

        if stack is None:
            if rule.required:
                raise StackAutoWireError(
                    f"Required stack '{rule.stack}' not available for rule '{rule.trigger}'"
                )
            logger.debug(f"Stack '{rule.stack}' not available, skipping rule")
            return None

        # Build arguments for the stack method
        args = {
            "context": context,
            "result": result,
            "error": error,
        }

        if rule.extract_args:
            args.update(rule.extract_args(context))

        # Execute the stack method
        try:
            method = self._get_stack_method(stack, rule.method, rule.stack)
            if method:
                # 2026-01-20 FIX: Filter args to only pass what the method accepts
                import inspect
                try:
                    sig = inspect.signature(method)
                    accepted_params = set(sig.parameters.keys())
                    # Filter to only accepted params (allow **kwargs methods to accept all)
                    if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                        args = {k: v for k, v in args.items() if k in accepted_params}
                except (ValueError, TypeError):
                    logger.debug(f"GRAPH: Graph operation failed: {e}")
                rule_result = method(**args)

                # Track execution
                self._track_execution(rule.stack, rule.method)

                logger.debug(
                    f"Auto-wired {rule.stack}.{rule.method} for '{rule.trigger}'"
                )
                return rule_result

        except Exception as e:
            logger.warning(
                f"Auto-wire rule failed: {rule.stack}.{rule.method}: {e}"
            )
            if rule.required:
                raise

        return None

    def _get_stack_method(
        self,
        stack: Any,
        method_name: str,
        stack_name: str,
    ) -> Optional[Callable]:
        """Get the appropriate method from a stack."""
        # Direct method access
        if hasattr(stack, method_name):
            return getattr(stack, method_name)

        # For dict-based stacks (like our custom implementations)
        if isinstance(stack, dict):
            if method_name in stack:
                return stack[method_name]
            # Try to find a callable
            for key, value in stack.items():
                if callable(value) and method_name in key.lower():
                    return value

        # Stack-specific method mapping
        method_mapping = self._get_method_mapping(stack_name)
        if method_name in method_mapping:
            actual_method = method_mapping[method_name]
            if hasattr(stack, actual_method):
                return getattr(stack, actual_method)

        return None

    def _get_method_mapping(self, stack_name: str) -> Dict[str, str]:
        """Get method name mapping for a stack."""
        mappings = {
            "langfuse": {
                "trace_generation": "generation",
                "trace_agent": "trace",
                "trace_gate": "trace",
            },
            "got": {
                "analyze_failure": "analyze",
                "multi_perspective_analysis": "analyze",
                "trace_root_cause": "trace_root_cause",
            },
            "reflexion": {
                "reflect_on_failure": "reflect",
                "retry_with_learning": "retry_with_learning",
            },
            "letta": {
                "save_learning": "add_to_archival",
                "save_failure_context": "add_to_archival",
                "save_decision": "add_to_archival",
                "recall_context": "search_archival",
            },
            "deepeval": {
                "evaluate_response": "evaluate",
                "score_generation": "evaluate",
                "evaluate_llm_output": "evaluate_llm_output",
                "evaluate_with_context": "evaluate_with_context",
            },
            "guardrails": {
                "enforce_mandatory": "enforce_mandatory",  # FIX CONSISTENCY-03: correct method name
                "protected_stack_call": "protected_stack_call",
                "validate_action": "validate_action",
            },
            "qdrant": {
                "semantic_search": "search",
                "search": "search",
            },
            "falkordb": {
                "query_dependencies": "execute",
                "query_impact": "execute",
            },
            "nemo": {
                "apply_rails": "generate",
                "validate_input": "generate",
            },
        }
        return mappings.get(stack_name, {})

    def _track_execution(self, stack: str, method: str):
        """Track stack method execution for metrics."""
        if stack not in self._execution_stats:
            self._execution_stats[stack] = {}
        if method not in self._execution_stats[stack]:
            self._execution_stats[stack][method] = 0
        self._execution_stats[stack][method] += 1

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    @contextmanager
    def operation(
        self,
        trigger: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Context manager for auto-wired operations.

        Usage:
            with autowire.operation("llm_call", {"prompt": "..."}) as ctx:
                result = call_llm()
                ctx["result"] = result

        All applicable stacks will be auto-wired before, after, and on error.
        """
        ctx = context or {}
        ctx["_trigger"] = trigger
        ctx["_start_time"] = time.time()

        # Execute "before" rules
        before_rules = self._get_rules_for_trigger(trigger, "before")
        before_results = {}
        for rule in before_rules:
            try:
                before_results[rule.stack] = self._execute_rule(rule, ctx)
            except Exception as e:
                if rule.required:
                    raise
                logger.warning(f"Before rule failed: {rule.stack}: {e}")

        ctx["_before_results"] = before_results

        error = None
        result = None

        try:
            yield ctx
            result = ctx.get("result")

        except Exception as e:
            error = e
            ctx["_error"] = e

            # Execute "on_error" rules
            error_rules = self._get_rules_for_trigger(trigger, "on_error")
            for rule in error_rules:
                try:
                    self._execute_rule(rule, ctx, error=e)
                except Exception as rule_error:
                    logger.warning(f"Error rule failed: {rule.stack}: {rule_error}")

            raise

        finally:
            ctx["_end_time"] = time.time()
            ctx["_duration_ms"] = (ctx["_end_time"] - ctx["_start_time"]) * 1000

            # Execute "after" rules
            after_rules = self._get_rules_for_trigger(trigger, "after")
            for rule in after_rules:
                try:
                    self._execute_rule(rule, ctx, result=result, error=error)
                except Exception as e:
                    logger.warning(f"After rule failed: {rule.stack}: {e}")

    def wire(self, trigger: str) -> Callable[[F], F]:
        """Decorator to auto-wire stacks for a function.

        Usage:
            @autowire.wire("llm_call")
            def my_llm_function(prompt):
                return call_llm(prompt)
        """
        def decorator(func: F) -> F:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                context = {
                    "function": func.__name__,
                    "args": args,
                    "kwargs": kwargs,
                }

                with self.operation(trigger, context) as ctx:
                    result = func(*args, **kwargs)
                    ctx["result"] = result
                    return result

            return wrapper  # type: ignore
        return decorator

    def wire_class(self, triggers: Dict[str, str]):
        """Decorator to auto-wire stacks for class methods.

        Usage:
            @autowire.wire_class({
                "run": "agent_execution",
                "analyze": "complex_decision",
            })
            class MyAgent:
                def run(self): ...
                def analyze(self): ...
        """
        def decorator(cls):
            for method_name, trigger in triggers.items():
                if hasattr(cls, method_name):
                    original_method = getattr(cls, method_name)
                    wired_method = self.wire(trigger)(original_method)
                    setattr(cls, method_name, wired_method)
            return cls
        return decorator

    def get_stats(self) -> Dict[str, Any]:
        """Get auto-wire execution statistics."""
        total_executions = sum(
            sum(methods.values())
            for methods in self._execution_stats.values()
        )
        return {
            "total_executions": total_executions,
            "by_stack": self._execution_stats,
            "rules_count": len(self._rules),
        }

    def get_rules_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all auto-wire rules."""
        return [
            {
                "trigger": r.trigger,
                "stack": r.stack,
                "method": r.method,
                "when": r.when,
                "priority": r.priority,
                "required": r.required,
            }
            for r in self._rules
        ]


# =============================================================================
# EXCEPTIONS
# =============================================================================


class StackAutoWireError(Exception):
    """Exception raised when auto-wiring fails."""
    pass


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


_autowire_instance: Optional[StackAutoWire] = None


def get_autowire() -> StackAutoWire:
    """Get the singleton StackAutoWire instance."""
    global _autowire_instance
    if _autowire_instance is None:
        _autowire_instance = StackAutoWire()
    return _autowire_instance


def autowire_operation(trigger: str, context: Dict = None):
    """Context manager for auto-wired operations."""
    return get_autowire().operation(trigger, context)


def autowire(trigger: str):
    """Decorator to auto-wire stacks for a function."""
    return get_autowire().wire(trigger)


def autowire_class(triggers: Dict[str, str]):
    """Decorator to auto-wire stacks for class methods."""
    return get_autowire().wire_class(triggers)


def get_autowire_stats() -> Dict[str, Any]:
    """Get auto-wire execution statistics."""
    return get_autowire().get_stats()


# =============================================================================
# INTEGRATION WITH EXISTING COMPONENTS
# =============================================================================


def integrate_with_gate_runner():
    """Integrate auto-wiring with the gate runner.

    This patches the gate runner to auto-wire stacks on gate execution.
    """
    try:
        from pipeline import gate_runner

        original_run_gate = gate_runner.run_gate

        @functools.wraps(original_run_gate)
        def wired_run_gate(*args, **kwargs):
            context = {"gate_args": args, "gate_kwargs": kwargs}
            with autowire_operation("gate_execution", context) as ctx:
                try:
                    result = original_run_gate(*args, **kwargs)
                    ctx["result"] = result
                    return result
                except Exception as e:
                    ctx["error"] = e
                    # Trigger gate_failure rules
                    with autowire_operation("gate_failure", ctx):
                        pass
                    raise

        gate_runner.run_gate = wired_run_gate
        logger.info("Auto-wiring integrated with gate_runner")

    except ImportError:
        logger.debug("gate_runner not available for integration")


def integrate_with_crewai():
    """Integrate auto-wiring with CrewAI agents.

    This patches CrewAI to auto-wire stacks on agent execution.
    """
    try:
        from crewai import Agent

        original_execute = Agent.execute_task

        @functools.wraps(original_execute)
        def wired_execute(self, *args, **kwargs):
            context = {
                "agent_role": self.role,
                "agent_goal": self.goal,
                "task_args": args,
            }
            with autowire_operation("agent_execution", context) as ctx:
                result = original_execute(self, *args, **kwargs)
                ctx["result"] = result
                return result

        Agent.execute_task = wired_execute
        logger.info("Auto-wiring integrated with CrewAI")

    except ImportError:
        logger.debug("CrewAI not available for integration")


def integrate_all():
    """Integrate auto-wiring with all available components."""
    integrate_with_gate_runner()
    integrate_with_crewai()
    logger.info("Auto-wiring integration complete")


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Classes
    "StackAutoWire",
    "AutoWireRule",
    "StackAutoWireError",
    # Functions
    "get_autowire",
    "autowire_operation",
    "autowire",
    "autowire_class",
    "get_autowire_stats",
    "integrate_all",
    # Constants
    "AUTO_WIRE_RULES",
]
