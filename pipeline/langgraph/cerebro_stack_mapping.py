"""Cerebro-to-Stack Mapping for LangGraph Pipeline.

This module maps each cerebro (agent) to the stacks it requires,
ensuring proper resource allocation and stack injection per agent role.

Based on: MIGRATION_V2_TO_LANGGRAPH.md Section "CEREBROS → STACKS"
and docs/Agents/ hierarchy (L0-L6).

Stack Categories:
- Primary: Required for agent operation
- Reasoning: Optional reasoning augmentation (GoT, BoT, DSPy)
- Memory: Optional memory systems (Letta, Graphiti)
- Eval: Optional evaluation tools (TruLens, RAGAS)
- Security: Optional security guardrails (NeMo, LLM Guard)
- Observability: Optional tracing (Langfuse, Phoenix)

Author: Pipeline Autonomo Team
Version: 1.0.0 (2026-01-16)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set


# =============================================================================
# STACK CATEGORIES
# =============================================================================


class StackCategory(str, Enum):
    """Categories of stacks."""

    PRIMARY = "primary"
    REASONING = "reasoning"
    MEMORY = "memory"
    EVAL = "eval"
    SECURITY = "security"
    OBSERVABILITY = "observability"


@dataclass
class CerebroStackConfig:
    """Configuration for a cerebro's stack requirements."""

    cerebro_id: str
    tier: str  # L0, L1, L2, L3, L4, L5, L6
    description: str

    # Required stacks (operation fails without these)
    primary_stacks: List[str] = field(default_factory=list)

    # Optional stacks by category
    reasoning_stacks: List[str] = field(default_factory=list)
    memory_stacks: List[str] = field(default_factory=list)
    eval_stacks: List[str] = field(default_factory=list)
    security_stacks: List[str] = field(default_factory=list)
    observability_stacks: List[str] = field(default_factory=list)

    def get_all_stacks(self) -> Set[str]:
        """Get all stacks (required + optional)."""
        all_stacks = set(self.primary_stacks)
        all_stacks.update(self.reasoning_stacks)
        all_stacks.update(self.memory_stacks)
        all_stacks.update(self.eval_stacks)
        all_stacks.update(self.security_stacks)
        all_stacks.update(self.observability_stacks)
        return all_stacks

    def get_required_stacks(self) -> List[str]:
        """Get only required stacks."""
        return self.primary_stacks.copy()


# =============================================================================
# CEREBRO STACK MAPPINGS
# =============================================================================


CEREBRO_STACK_MAPPINGS: Dict[str, CerebroStackConfig] = {
    # =========================================================================
    # L0 - SYSTEM TIER
    # =========================================================================
    "run_master": CerebroStackConfig(
        cerebro_id="run_master",
        tier="L0",
        description="Pipeline orchestrator and zelador",
        primary_stacks=["langgraph", "redis", "crewai"],
        reasoning_stacks=["got"],
        memory_stacks=["letta", "graphiti"],
        security_stacks=["nemo"],
        observability_stacks=["langfuse", "phoenix"],
    ),
    "pack_driver": CerebroStackConfig(
        cerebro_id="pack_driver",
        tier="L0",
        description="Context pack loading and management",
        primary_stacks=["redis"],
        memory_stacks=["qdrant"],
        observability_stacks=["langfuse"],
    ),
    "ops_ctrl": CerebroStackConfig(
        cerebro_id="ops_ctrl",
        tier="L0",
        description="Operational control and monitoring",
        primary_stacks=["redis", "langgraph"],
        observability_stacks=["langfuse", "phoenix"],
    ),

    # =========================================================================
    # L1 - EXECUTIVE TIER
    # =========================================================================
    "presidente": CerebroStackConfig(
        cerebro_id="presidente",
        tier="L1",
        description="Strategic oversight and final decisions",
        primary_stacks=["redis", "crewai"],
        reasoning_stacks=["got"],
        memory_stacks=["graphiti"],
        observability_stacks=["langfuse"],
    ),
    "ceo": CerebroStackConfig(
        cerebro_id="ceo",
        tier="L1",
        description="Executive decisions and approvals",
        primary_stacks=["redis", "crewai"],
        reasoning_stacks=["got", "reflexion"],
        memory_stacks=["letta", "graphiti"],
        observability_stacks=["langfuse"],
    ),
    "arbiter": CerebroStackConfig(
        cerebro_id="arbiter",
        tier="L1",
        description="Conflict resolution between agents",
        primary_stacks=["redis"],
        reasoning_stacks=["got"],
        memory_stacks=["graphiti"],
        observability_stacks=["langfuse"],
    ),

    # =========================================================================
    # L2 - VP TIER
    # =========================================================================
    "spec_vp": CerebroStackConfig(
        cerebro_id="spec_vp",
        tier="L2",
        description="Specification oversight",
        primary_stacks=["redis", "crewai"],
        reasoning_stacks=["got", "dspy"],
        memory_stacks=["qdrant", "graphiti"],
        observability_stacks=["langfuse"],
    ),
    "exec_vp": CerebroStackConfig(
        cerebro_id="exec_vp",
        tier="L2",
        description="Execution oversight",
        primary_stacks=["redis", "crewai", "temporal"],
        reasoning_stacks=["got"],
        memory_stacks=["letta"],
        observability_stacks=["langfuse"],
    ),
    "qa_vp": CerebroStackConfig(
        cerebro_id="qa_vp",
        tier="L2",
        description="Quality assurance oversight",
        primary_stacks=["redis"],
        reasoning_stacks=["got"],
        eval_stacks=["trulens", "ragas"],
        observability_stacks=["langfuse"],
    ),

    # =========================================================================
    # L3 - MASTER TIER
    # =========================================================================
    "spec_master": CerebroStackConfig(
        cerebro_id="spec_master",
        tier="L3",
        description="Specification creation and refinement",
        primary_stacks=["redis", "crewai"],
        reasoning_stacks=["got", "bot", "dspy"],
        memory_stacks=["qdrant", "active_rag"],
        security_stacks=["nemo"],
        observability_stacks=["langfuse"],
    ),
    "ace_exec": CerebroStackConfig(
        cerebro_id="ace_exec",
        tier="L3",
        description="Code execution master",
        primary_stacks=["redis", "crewai", "temporal"],
        reasoning_stacks=["got"],
        memory_stacks=["letta"],
        security_stacks=["llm_guard"],
        observability_stacks=["langfuse"],
    ),
    "qa_master": CerebroStackConfig(
        cerebro_id="qa_master",
        tier="L3",
        description="Quality assurance execution",
        primary_stacks=["redis"],
        reasoning_stacks=["got", "reflexion"],
        eval_stacks=["trulens", "ragas", "deepeval", "cleanlab"],
        security_stacks=["z3"],
        observability_stacks=["langfuse"],
    ),

    # =========================================================================
    # L4 - SQUAD LEAD TIER
    # =========================================================================
    "squad_lead_spec": CerebroStackConfig(
        cerebro_id="squad_lead_spec",
        tier="L4",
        description="Spec squad coordination",
        primary_stacks=["redis", "crewai"],
        reasoning_stacks=["got"],
        memory_stacks=["qdrant"],
        observability_stacks=["langfuse"],
    ),
    "squad_lead_exec": CerebroStackConfig(
        cerebro_id="squad_lead_exec",
        tier="L4",
        description="Execution squad coordination",
        primary_stacks=["redis", "crewai"],
        reasoning_stacks=["got"],
        memory_stacks=["letta"],
        observability_stacks=["langfuse"],
    ),
    "squad_lead_qa": CerebroStackConfig(
        cerebro_id="squad_lead_qa",
        tier="L4",
        description="QA squad coordination",
        primary_stacks=["redis"],
        reasoning_stacks=["got"],
        eval_stacks=["trulens"],
        observability_stacks=["langfuse"],
    ),

    # =========================================================================
    # L5 - WORKER TIER
    # =========================================================================
    "developer": CerebroStackConfig(
        cerebro_id="developer",
        tier="L5",
        description="Code implementation",
        primary_stacks=["redis"],
        reasoning_stacks=["bot"],
        memory_stacks=["letta"],
        security_stacks=["llm_guard"],
        observability_stacks=["langfuse"],
    ),
    "auditor": CerebroStackConfig(
        cerebro_id="auditor",
        tier="L5",
        description="Code and spec auditing",
        primary_stacks=["redis"],
        reasoning_stacks=["got"],
        eval_stacks=["cleanlab"],
        observability_stacks=["langfuse"],
    ),
    "refinador": CerebroStackConfig(
        cerebro_id="refinador",
        tier="L5",
        description="Spec refinement",
        primary_stacks=["redis"],
        reasoning_stacks=["got", "dspy"],
        memory_stacks=["qdrant"],
        observability_stacks=["langfuse"],
    ),
    "tester": CerebroStackConfig(
        cerebro_id="tester",
        tier="L5",
        description="Test implementation and execution",
        primary_stacks=["redis"],
        eval_stacks=["deepeval", "ragas"],
        observability_stacks=["langfuse"],
    ),
    "tech_writer": CerebroStackConfig(
        cerebro_id="tech_writer",
        tier="L5",
        description="Documentation creation",
        primary_stacks=["redis"],
        memory_stacks=["qdrant"],
        observability_stacks=["langfuse"],
    ),
    "reviewer": CerebroStackConfig(
        cerebro_id="reviewer",
        tier="L5",
        description="Code and PR review",
        primary_stacks=["redis"],
        reasoning_stacks=["got"],
        eval_stacks=["cleanlab"],
        observability_stacks=["langfuse"],
    ),
    "integrator": CerebroStackConfig(
        cerebro_id="integrator",
        tier="L5",
        description="System integration",
        primary_stacks=["redis"],
        memory_stacks=["graphiti"],
        observability_stacks=["langfuse"],
    ),
    "debugger": CerebroStackConfig(
        cerebro_id="debugger",
        tier="L5",
        description="Bug investigation and fixing",
        primary_stacks=["redis"],
        reasoning_stacks=["got", "reflexion"],
        memory_stacks=["letta"],
        observability_stacks=["langfuse"],
    ),
    "optimizer": CerebroStackConfig(
        cerebro_id="optimizer",
        tier="L5",
        description="Performance optimization",
        primary_stacks=["redis"],
        reasoning_stacks=["bot"],
        observability_stacks=["langfuse", "phoenix"],
    ),

    # =========================================================================
    # L6 - SPECIALIST TIER
    # =========================================================================
    "oracle_architect": CerebroStackConfig(
        cerebro_id="oracle_architect",
        tier="L6",
        description="Oracle system design",
        primary_stacks=["redis"],
        reasoning_stacks=["got", "bot"],
        memory_stacks=["graphiti", "qdrant"],
        observability_stacks=["langfuse"],
    ),
    "security_specialist": CerebroStackConfig(
        cerebro_id="security_specialist",
        tier="L6",
        description="Security analysis and hardening",
        primary_stacks=["redis"],
        reasoning_stacks=["got"],
        security_stacks=["nemo", "llm_guard", "z3"],
        observability_stacks=["langfuse"],
    ),
    "integration_officer": CerebroStackConfig(
        cerebro_id="integration_officer",
        tier="L6",
        description="External system integration",
        primary_stacks=["redis"],
        memory_stacks=["graphiti"],
        observability_stacks=["langfuse"],
    ),
    "data_scientist": CerebroStackConfig(
        cerebro_id="data_scientist",
        tier="L6",
        description="Data analysis and ML",
        primary_stacks=["redis"],
        reasoning_stacks=["dspy"],
        memory_stacks=["qdrant"],
        eval_stacks=["ragas", "trulens"],
        observability_stacks=["langfuse"],
    ),
    "performance_analyst": CerebroStackConfig(
        cerebro_id="performance_analyst",
        tier="L6",
        description="System performance analysis",
        primary_stacks=["redis"],
        observability_stacks=["langfuse", "phoenix"],
    ),
    "ux_specialist": CerebroStackConfig(
        cerebro_id="ux_specialist",
        tier="L6",
        description="User experience design",
        primary_stacks=["redis"],
        memory_stacks=["qdrant"],
        observability_stacks=["langfuse"],
    ),
    "devops_engineer": CerebroStackConfig(
        cerebro_id="devops_engineer",
        tier="L6",
        description="Infrastructure and deployment",
        primary_stacks=["redis", "temporal"],
        observability_stacks=["langfuse", "phoenix"],
    ),
    "ml_engineer": CerebroStackConfig(
        cerebro_id="ml_engineer",
        tier="L6",
        description="ML model development",
        primary_stacks=["redis"],
        reasoning_stacks=["dspy", "bot"],
        memory_stacks=["qdrant"],
        eval_stacks=["trulens"],
        observability_stacks=["langfuse"],
    ),
    "knowledge_engineer": CerebroStackConfig(
        cerebro_id="knowledge_engineer",
        tier="L6",
        description="Knowledge graph management",
        primary_stacks=["redis", "falkordb"],
        memory_stacks=["graphiti", "qdrant", "graphrag"],
        observability_stacks=["langfuse"],
    ),
    "llm_orchestrator": CerebroStackConfig(
        cerebro_id="llm_orchestrator",
        tier="L6",
        description="LLM orchestration, model cascading, and prompt optimization",
        primary_stacks=["redis", "crewai"],
        reasoning_stacks=["dspy", "bot", "medprompt", "quiet_star", "got"],
        memory_stacks=["letta"],
        eval_stacks=["trulens", "ragas", "deepeval"],
        security_stacks=["nemo", "llm_guard"],
        observability_stacks=["langfuse", "phoenix"],
    ),
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_cerebro_config(cerebro_id: str) -> Optional[CerebroStackConfig]:
    """Get stack configuration for a cerebro.

    Args:
        cerebro_id: Cerebro identifier.

    Returns:
        CerebroStackConfig or None if unknown.
    """
    return CEREBRO_STACK_MAPPINGS.get(cerebro_id.lower())


def get_cerebros_by_tier(tier: str) -> List[CerebroStackConfig]:
    """Get all cerebros in a tier.

    Args:
        tier: Tier identifier (L0, L1, L2, L3, L4, L5, L6).

    Returns:
        List of CerebroStackConfig for that tier.
    """
    return [
        config for config in CEREBRO_STACK_MAPPINGS.values()
        if config.tier == tier
    ]


def get_cerebros_using_stack(stack_name: str) -> List[str]:
    """Get all cerebros that use a specific stack.

    Args:
        stack_name: Stack name (e.g., "got", "redis").

    Returns:
        List of cerebro IDs.
    """
    result = []
    for cerebro_id, config in CEREBRO_STACK_MAPPINGS.items():
        if stack_name in config.get_all_stacks():
            result.append(cerebro_id)
    return result


def get_stack_usage_summary() -> Dict[str, int]:
    """Get usage count for each stack across all cerebros.

    Returns:
        Dict of stack_name -> usage_count.
    """
    usage: Dict[str, int] = {}
    for config in CEREBRO_STACK_MAPPINGS.values():
        for stack in config.get_all_stacks():
            usage[stack] = usage.get(stack, 0) + 1
    return dict(sorted(usage.items(), key=lambda x: -x[1]))


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "StackCategory",
    "CerebroStackConfig",
    "CEREBRO_STACK_MAPPINGS",
    "get_cerebro_config",
    "get_cerebros_by_tier",
    "get_cerebros_using_stack",
    "get_stack_usage_summary",
]
