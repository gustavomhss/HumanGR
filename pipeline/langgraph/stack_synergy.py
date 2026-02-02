"""Stack Synergy Manager for Pipeline V2.

This module provides unified stack coordination, discovery, and context injection
for all pipeline components and agents. It ensures that:

1. All components know what stacks are available
2. Agents receive proper stack context in their prompts
3. Graceful degradation is handled uniformly
4. Stack usage is optimized and tracked

Architecture:
    StackSynergyManager (Coordinator)
        |
        +-- StackInjector (Access Control)
        +-- CerebroStackMapping (Per-Agent Config)
        +-- StackContextBuilder (Prompt Injection)
        |
        v
    Agents/Pipeline Nodes (Consumers)

Author: Pipeline Autonomo Team
Version: 1.0.0 (2026-01-16)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)

# =============================================================================
# DATA STRUCTURES
# =============================================================================


class StackStatus(str, Enum):
    """Stack availability status."""

    AVAILABLE = "available"        # Stack is healthy and ready
    DEGRADED = "degraded"          # Stack has issues but functional
    UNAVAILABLE = "unavailable"    # Stack is not accessible
    NOT_CONFIGURED = "not_configured"  # Stack not configured for this agent


@dataclass
class StackCapability:
    """Describes what a stack can do."""

    name: str
    category: str  # primary, reasoning, memory, eval, security, observability
    description: str
    operations: List[str]  # What operations it supports
    usage_example: str
    fallback: Optional[str] = None  # Alternative stack if unavailable
    priority: int = 2  # 0=critical, 1=important, 2=optional


@dataclass
class AgentStackContext:
    """Context about stacks for agent injection.

    This class generates STATE-OF-THE-ART context for agents,
    teaching them exactly how to extract MAXIMUM VALUE from each stack.
    """

    agent_id: str
    tier: str
    available_stacks: Dict[str, StackStatus]
    capabilities: Dict[str, StackCapability]
    recommendations: List[str]
    fallbacks: Dict[str, str]  # stack -> fallback_stack

    def to_prompt_section(self, detail_level: str = "full") -> str:
        """Generate comprehensive prompt section for agent injection.

        This generates a STATE-OF-THE-ART context that teaches agents:
        1. WHAT each stack does (core purpose)
        2. WHEN to use it (triggers and scenarios)
        3. HOW to use it (API patterns and examples)
        4. COMBINE with other stacks (synergies)
        5. AVOID common mistakes (pitfalls)

        Args:
            detail_level: "full" for complete guide, "compact" for summary

        Returns:
            Comprehensive prompt section with stack mastery guide
        """
        try:
            from pipeline.langgraph.stack_mastery import (
                STACK_MASTERY,
                generate_mastery_prompt,
            )
            use_mastery = True
        except ImportError:
            use_mastery = False

        # Get available stack names
        available_names = [
            name for name, status in self.available_stacks.items()
            if status == StackStatus.AVAILABLE
        ]

        if use_mastery and detail_level == "full":
            return self._generate_full_mastery_prompt(available_names)
        else:
            return self._generate_compact_prompt(available_names)

    def _generate_full_mastery_prompt(self, available_names: List[str]) -> str:
        """Generate full mastery prompt with complete API patterns."""
        from pipeline.langgraph.stack_mastery import STACK_MASTERY

        lines = [
            "=" * 70,
            "   GUIA COMPLETO DE STACKS - EXTRAIA O MAXIMO DE CADA UMA",
            "=" * 70,
            "",
            f"**Agent:** {self.agent_id} | **Tier:** {self.tier}",
            "",
            "IMPORTANTE: Voce tem acesso a stacks PODEROSAS. Este guia ensina",
            "exatamente COMO usar cada uma para MAXIMIZAR sua eficiencia.",
            "NAO ignore este guia - ele contem patterns de codigo prontos.",
            "",
            "-" * 70,
            "",
        ]

        # Group by category with priority order
        categories = {
            "reasoning": ("REASONING - Pense Melhor", []),
            "memory": ("MEMORY - Lembre de Tudo", []),
            "eval": ("EVAL - Avalie Qualidade", []),
            "security": ("SECURITY - Proteja", []),
            "observability": ("OBSERVABILITY - Monitore", []),
            "primary": ("INFRASTRUCTURE - Base", []),
        }

        for name in available_names:
            mastery = STACK_MASTERY.get(name)
            if mastery and mastery.category in categories:
                categories[mastery.category][1].append(mastery)

        for cat_key, (cat_title, masteries) in categories.items():
            if masteries:
                lines.append(f"## {cat_title}")
                lines.append("")

                for m in masteries:
                    lines.append(f"### {m.name.upper()}: {m.tagline}")
                    lines.append("")

                    # Core purpose
                    lines.append("**O QUE FAZ:**")
                    lines.append(m.core_purpose.strip())
                    lines.append("")

                    # Key features
                    lines.append("**CAPACIDADES:**")
                    for feat in m.key_features:
                        lines.append(f"  - {feat}")
                    lines.append("")

                    # When to use - with triggers
                    lines.append("**QUANDO USAR:**")
                    for scenario in m.ideal_scenarios:
                        lines.append(f"  - {scenario}")
                    lines.append("")
                    lines.append(f"  **Triggers:** {', '.join(m.triggers[:7])}")
                    lines.append("")

                    # How to use - API patterns
                    lines.append("**COMO USAR (codigo pronto):**")
                    for op_name, pattern in list(m.api_patterns.items())[:2]:
                        lines.append(f"")
                        lines.append(f"*{op_name}:*")
                        lines.append("```python")
                        lines.append(pattern.strip())
                        lines.append("```")
                    lines.append("")

                    # Combinations
                    if m.combines_with:
                        lines.append("**COMBINA COM:**")
                        for other, how in m.combines_with.items():
                            lines.append(f"  - **{other}**: {how}")
                        lines.append("")

                    # Pitfalls
                    if m.pitfalls:
                        lines.append("**CUIDADO:**")
                        for pitfall in m.pitfalls:
                            lines.append(f"  - {pitfall}")
                        lines.append("")

                    lines.append("-" * 40)
                    lines.append("")

        # Decision matrix
        lines.append("=" * 70)
        lines.append("   MATRIZ DE DECISAO - QUAL STACK USAR?")
        lines.append("=" * 70)
        lines.append("")
        lines.append("| SITUACAO | STACK | POR QUE |")
        lines.append("|----------|-------|---------|")
        lines.append("| Analisar falha de gate | `got` + `reflexion` | Multi-perspectiva + aprendizado |")
        lines.append("| Buscar codigo/docs similar | `qdrant` | Busca semantica por significado |")
        lines.append("| Lembrar algo para futuro | `letta` | Memoria persistente entre sessoes |")
        lines.append("| Modelar dependencias | `falkordb` | Grafo de relacoes |")
        lines.append("| Avaliar output de LLM | `deepeval` | Metricas especializadas |")
        lines.append("| Filtrar input/output | `nemo` | Guardrails de seguranca |")
        lines.append("| Rastrear execucao | `langfuse` | **OBRIGATORIO** em toda chamada LLM |")
        lines.append("| Workflow que nao pode falhar | `temporal` | Durabilidade e checkpoint |")
        lines.append("| Extrair dados estruturados | `instructor` | Pydantic models do LLM |")
        lines.append("| Orquestrar agents | `crewai` | Hierarquia e delegacao |")
        lines.append("")

        # Synergy patterns
        lines.append("=" * 70)
        lines.append("   PADROES DE SINERGIA - COMBINE PARA POTENCIA MAXIMA")
        lines.append("=" * 70)
        lines.append("")
        lines.append("**Pattern 1: Analise de Falhas (GOLD STANDARD)**")
        lines.append("```")
        lines.append("gate_falha -> got.analyze(multi-perspectiva)")
        lines.append("          -> reflexion.reflect(aprendizado)")
        lines.append("          -> letta.save(persistir licao)")
        lines.append("          -> langfuse.trace(observabilidade)")
        lines.append("```")
        lines.append("")
        lines.append("**Pattern 2: RAG Turbinado**")
        lines.append("```")
        lines.append("query -> qdrant.search(semantico)")
        lines.append("      -> falkordb.expand(relacoes)")
        lines.append("      -> graphrag.combine(hibrido)")
        lines.append("      -> deepeval.score(qualidade)")
        lines.append("```")
        lines.append("")
        lines.append("**Pattern 3: Implementacao Segura**")
        lines.append("```")
        lines.append("input -> nemo.validate(guardrails)")
        lines.append("      -> llm.generate(com langfuse.trace)")
        lines.append("      -> instructor.extract(estruturado)")
        lines.append("      -> deepeval.evaluate(qualidade)")
        lines.append("```")
        lines.append("")

        # Unavailable stacks with alternatives
        unavailable = [
            (name, status) for name, status in self.available_stacks.items()
            if status == StackStatus.UNAVAILABLE
        ]
        if unavailable:
            lines.append("=" * 70)
            lines.append("   STACKS INDISPONIVEIS - USE ALTERNATIVAS")
            lines.append("=" * 70)
            lines.append("")
            for name, _ in unavailable:
                fallback = self.fallbacks.get(name)
                if fallback:
                    lines.append(f"  - ~~{name}~~ -> use **{fallback}**")
                else:
                    lines.append(f"  - ~~{name}~~ (sem alternativa)")
            lines.append("")

        # Final recommendations
        if self.recommendations:
            lines.append("=" * 70)
            lines.append("   RECOMENDACOES ESPECIFICAS PARA VOCE")
            lines.append("=" * 70)
            lines.append("")
            for rec in self.recommendations:
                lines.append(f"  - {rec}")
            lines.append("")

        # Mandatory reminder
        lines.append("=" * 70)
        lines.append("   LEMBRETES OBRIGATORIOS")
        lines.append("=" * 70)
        lines.append("")
        lines.append("1. TODA chamada LLM deve ter langfuse.trace()")
        lines.append("2. TODA falha deve ter reflexion.reflect()")
        lines.append("3. TODO aprendizado deve ser salvo em letta")
        lines.append("4. TODA decisao importante deve usar got.analyze()")
        lines.append("")

        return "\n".join(lines)

    def _generate_compact_prompt(self, available_names: List[str]) -> str:
        """Generate compact prompt for limited context scenarios."""
        lines = [
            "## STACKS DISPONIVEIS",
            "",
            f"**Agent:** {self.agent_id} | **Tier:** {self.tier}",
            "",
        ]

        for name in sorted(available_names):
            cap = self.capabilities.get(name)
            if cap:
                lines.append(f"- **{name}**: {cap.description}")

        lines.append("")
        lines.append("**Decisao Rapida:**")
        lines.append("- Analise falhas: got + reflexion")
        lines.append("- Busca semantica: qdrant")
        lines.append("- Memoria: letta")
        lines.append("- Tracing: langfuse (OBRIGATORIO)")
        lines.append("")

        return "\n".join(lines)


@dataclass
class SynergyReport:
    """Report on stack synergy status."""

    total_stacks: int
    available_count: int
    degraded_count: int
    unavailable_count: int
    coverage_pct: float
    gaps: List[str]
    recommendations: List[str]
    stack_health: Dict[str, Dict[str, Any]]


# =============================================================================
# STACK CAPABILITIES REGISTRY
# =============================================================================


STACK_CAPABILITIES: Dict[str, StackCapability] = {
    # Infrastructure (Critical)
    "redis": StackCapability(
        name="redis",
        category="primary",
        description="Event bus, cache, IPC between agents",
        operations=["pub/sub", "cache", "heartbeat", "safe_halt"],
        usage_example="redis.publish('channel', message); redis.get('key')",
        priority=0,
    ),
    "langgraph": StackCapability(
        name="langgraph",
        category="primary",
        description="State machine control plane for workflows",
        operations=["state_management", "checkpointing", "node_execution"],
        usage_example="graph.add_node('name', node_fn); compiled.invoke(state)",
        priority=0,
    ),
    "crewai": StackCapability(
        name="crewai",
        category="primary",
        description="Agent orchestration and hierarchy",
        operations=["crew_execution", "task_delegation", "agent_coordination"],
        usage_example="crew.kickoff(); agent.execute_task(task)",
        priority=0,
    ),

    # Memory Stacks
    "qdrant": StackCapability(
        name="qdrant",
        category="memory",
        description="Vector database for semantic search and embeddings",
        operations=["semantic_search", "embedding_storage", "similarity"],
        usage_example="qdrant.search(collection, vector, limit=10)",
        fallback="redis",  # Can use Redis for basic caching if Qdrant unavailable
        priority=1,
    ),
    "letta": StackCapability(
        name="letta",
        category="memory",
        description="Persistent agent state and memory",
        operations=["state_persistence", "conversation_memory", "long_term_memory"],
        usage_example="letta.save_state(agent_id, state); letta.recall(agent_id)",
        priority=1,
    ),
    "graphiti": StackCapability(
        name="graphiti",
        category="memory",
        description="Semantic knowledge graph with temporal awareness",
        operations=["kg_query", "relationship_storage", "temporal_facts"],
        usage_example="graphiti.add_fact(subject, predicate, object); graphiti.query(cypher)",
        fallback="qdrant",  # Semantic search fallback
        priority=1,
    ),
    "falkordb": StackCapability(
        name="falkordb",
        category="memory",
        description="Graph database for knowledge storage",
        operations=["graph_storage", "cypher_queries", "pattern_matching"],
        usage_example="falkordb.execute('MATCH (n) RETURN n')",
        priority=1,
    ),
    "active_rag": StackCapability(
        name="active_rag",
        category="memory",
        description="Forward-looking RAG with proactive retrieval",
        operations=["predictive_retrieval", "context_augmentation"],
        usage_example="active_rag.retrieve(query, anticipate_next=True)",
        fallback="qdrant",
        priority=2,
    ),
    "graphrag": StackCapability(
        name="graphrag",
        category="memory",
        description="Graph-enhanced RAG combining KG and vector search",
        operations=["hybrid_search", "graph_augmented_retrieval"],
        usage_example="graphrag.query(question, use_graph=True)",
        fallback="qdrant",
        priority=2,
    ),

    # Reasoning Stacks
    "got": StackCapability(
        name="got",
        category="reasoning",
        description="Graph of Thoughts for multi-path reasoning",
        operations=["multi_perspective", "root_cause_analysis", "consensus"],
        usage_example="got.analyze(problem, perspectives=3); got.reach_consensus(options)",
        priority=1,
    ),
    "reflexion": StackCapability(
        name="reflexion",
        category="reasoning",
        description="Self-reflection and verbal reinforcement learning",
        operations=["failure_analysis", "learning_loop", "retry_with_context"],
        usage_example="reflexion.reflect(failure); reflexion.retry_with_learning(task)",
        fallback="got",
        priority=1,
    ),
    "bot": StackCapability(
        name="bot",
        category="reasoning",
        description="Buffer of Thoughts for accumulated reasoning",
        operations=["thought_buffering", "accumulated_context", "chain_reasoning"],
        usage_example="bot.add_thought(thought); bot.synthesize()",
        fallback="got",
        priority=2,
    ),
    "dspy": StackCapability(
        name="dspy",
        category="reasoning",
        description="Prompt programming and optimization",
        operations=["prompt_optimization", "module_composition", "few_shot"],
        usage_example="dspy.ChainOfThought(signature); module(input)",
        priority=2,
    ),

    # Evaluation Stacks
    "trulens": StackCapability(
        name="trulens",
        category="eval",
        description="LLM evaluation and feedback functions",
        operations=["groundedness", "relevance", "coherence_scoring"],
        usage_example="trulens.evaluate(response, context); trulens.score()",
        priority=2,
    ),
    "ragas": StackCapability(
        name="ragas",
        category="eval",
        description="RAG quality metrics",
        operations=["faithfulness", "answer_relevancy", "context_precision"],
        usage_example="ragas.evaluate(dataset); ragas.score_response(response, context)",
        fallback="deepeval",
        priority=2,
    ),
    "deepeval": StackCapability(
        name="deepeval",
        category="eval",
        description="Deep evaluation for LLM outputs",
        operations=["task_specific_eval", "custom_metrics", "comparison"],
        usage_example="deepeval.test_case(input, output, expected)",
        fallback="trulens",
        priority=2,
    ),
    "cleanlab": StackCapability(
        name="cleanlab",
        category="eval",
        description="Hallucination detection and data quality",
        operations=["hallucination_detection", "label_quality", "data_cleaning"],
        usage_example="cleanlab.find_issues(data); cleanlab.detect_hallucination(text)",
        priority=2,
    ),

    # Security Stacks
    "nemo": StackCapability(
        name="nemo",
        category="security",
        description="NeMo Guardrails for content moderation",
        operations=["input_validation", "output_filtering", "policy_enforcement"],
        usage_example="rails.generate(prompt, policies=['no_pii', 'factual'])",
        fallback="llm_guard",
        priority=1,
    ),
    "llm_guard": StackCapability(
        name="llm_guard",
        category="security",
        description="LLM security scanning",
        operations=["injection_detection", "pii_scanning", "toxicity_check"],
        usage_example="llm_guard.scan_input(text); llm_guard.scan_output(response)",
        fallback="nemo",
        priority=1,
    ),
    "z3": StackCapability(
        name="z3",
        category="security",
        description="Formal verification and constraint solving",
        operations=["formal_verification", "constraint_satisfaction", "proof_checking"],
        usage_example="z3.verify(constraints); z3.solve(formula)",
        priority=2,
    ),

    # Observability Stacks
    "langfuse": StackCapability(
        name="langfuse",
        category="observability",
        description="LLM observability and tracing (MANDATORY)",
        operations=["tracing", "cost_tracking", "debugging", "analytics"],
        usage_example="langfuse.trace(run_id); langfuse.log_generation(prompt, response)",
        priority=0,  # Mandatory
    ),
    "phoenix": StackCapability(
        name="phoenix",
        category="observability",
        description="ML observability and experiment tracking",
        operations=["experiment_tracking", "model_monitoring", "drift_detection"],
        usage_example="phoenix.log_evaluation(metrics); phoenix.track_experiment(run)",
        fallback="langfuse",
        priority=2,
    ),

    # Workflow Stacks
    "temporal": StackCapability(
        name="temporal",
        category="primary",
        description="Durable workflow execution with checkpoint/resume",
        operations=["checkpointing", "retry_policies", "workflow_durability"],
        usage_example="temporal.checkpoint(); temporal.resume_from(checkpoint_id)",
        priority=1,
    ),

    # Output Stacks
    "instructor": StackCapability(
        name="instructor",
        category="primary",
        description="Structured output extraction from LLMs",
        operations=["structured_extraction", "validation", "retry_parsing"],
        usage_example="instructor.extract(Model, response); instructor.patch(client)",
        fallback="pydantic_ai",
        priority=2,
    ),
    "pydantic_ai": StackCapability(
        name="pydantic_ai",
        category="primary",
        description="Type-safe AI with Pydantic models",
        operations=["type_safe_generation", "schema_validation", "model_output"],
        usage_example="pydantic_ai.generate(MyModel, prompt)",
        fallback="instructor",
        priority=2,
    ),
    "hamilton": StackCapability(
        name="hamilton",
        category="primary",
        description="DAG-based data transformations",
        operations=["dag_execution", "data_pipeline", "transformation"],
        usage_example="hamilton.execute(dag, inputs)",
        priority=2,
    ),
    "neo4j": StackCapability(
        name="neo4j",
        category="memory",
        description="Enterprise graph database",
        operations=["graph_storage", "cypher_queries", "graph_algorithms"],
        usage_example="neo4j.execute('MATCH (n) RETURN n')",
        fallback="falkordb",
        priority=2,
    ),
}


# =============================================================================
# STACK SYNERGY MANAGER
# =============================================================================


class StackSynergyManager:
    """Manages stack synergy across pipeline and agents.

    This class provides:
    1. Unified stack discovery and availability checking
    2. Agent context generation for prompt injection
    3. Graceful degradation recommendations
    4. Synergy metrics and reporting
    """

    _instance: Optional['StackSynergyManager'] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> 'StackSynergyManager':
        """Singleton pattern with double-check locking for thread safety."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the synergy manager."""
        if self._initialized:
            return

        self._stack_injector = None
        self._cerebro_mapping = None
        self._health_cache: Dict[str, Dict[str, Any]] = {}
        self._initialized = True

        logger.info("StackSynergyManager initialized")

    def _ensure_injector(self):
        """Lazy load stack injector."""
        if self._stack_injector is None:
            try:
                from pipeline.langgraph.stack_injection import get_stack_injector
                self._stack_injector = get_stack_injector()
            except ImportError as e:
                logger.warning(f"Could not load StackInjector: {e}")

    def _ensure_cerebro_mapping(self):
        """Lazy load cerebro mapping."""
        if self._cerebro_mapping is None:
            try:
                from pipeline.langgraph.cerebro_stack_mapping import (
                    CEREBRO_STACK_MAPPINGS,
                    get_cerebro_config,
                )
                self._cerebro_mapping = CEREBRO_STACK_MAPPINGS
                self._get_cerebro_config = get_cerebro_config
            except ImportError as e:
                logger.warning(f"Could not load CerebroStackMapping: {e}")
                self._cerebro_mapping = {}
                self._get_cerebro_config = lambda x: None

    def refresh_health(self) -> Dict[str, Dict[str, Any]]:
        """Refresh stack health cache."""
        self._ensure_injector()
        if self._stack_injector:
            self._health_cache = self._stack_injector.check_health()
        return self._health_cache

    def get_stack_status(self, stack_name: str) -> StackStatus:
        """Get status of a specific stack."""
        if not self._health_cache:
            self.refresh_health()

        health = self._health_cache.get(stack_name, {})

        if health.get("healthy"):
            latency = health.get("latency_ms", 0)
            if latency > 1000:  # > 1 second is degraded
                return StackStatus.DEGRADED
            return StackStatus.AVAILABLE
        elif "error" in health:
            return StackStatus.UNAVAILABLE
        else:
            return StackStatus.NOT_CONFIGURED

    def get_available_stacks(self) -> List[str]:
        """Get list of all available stacks."""
        if not self._health_cache:
            self.refresh_health()

        return [
            name for name, health in self._health_cache.items()
            if health.get("healthy")
        ]

    def get_agent_context(
        self,
        agent_id: str,
        operation: Optional[str] = None,
    ) -> AgentStackContext:
        """Generate stack context for an agent.

        Args:
            agent_id: Cerebro identifier
            operation: Optional operation type (SPEC, EXEC, GATE, etc.)

        Returns:
            AgentStackContext with all relevant stack info
        """
        self._ensure_cerebro_mapping()
        if not self._health_cache:
            self.refresh_health()

        # Get cerebro config
        config = self._get_cerebro_config(agent_id) if self._get_cerebro_config else None
        tier = config.tier if config else "Unknown"

        # Get configured stacks for this agent
        if config:
            configured_stacks = config.get_all_stacks()
        else:
            # Default to all known stacks if no config
            configured_stacks = set(STACK_CAPABILITIES.keys())

        # Build status map
        available_stacks: Dict[str, StackStatus] = {}
        fallbacks: Dict[str, str] = {}

        for stack in configured_stacks:
            status = self.get_stack_status(stack)
            available_stacks[stack] = status

            # Add fallback if unavailable
            if status == StackStatus.UNAVAILABLE:
                cap = STACK_CAPABILITIES.get(stack)
                if cap and cap.fallback:
                    fallbacks[stack] = cap.fallback

        # Get capabilities for configured stacks
        capabilities = {
            stack: STACK_CAPABILITIES[stack]
            for stack in configured_stacks
            if stack in STACK_CAPABILITIES
        }

        # Generate recommendations
        recommendations = self._generate_recommendations(
            available_stacks, capabilities, tier
        )

        return AgentStackContext(
            agent_id=agent_id,
            tier=tier,
            available_stacks=available_stacks,
            capabilities=capabilities,
            recommendations=recommendations,
            fallbacks=fallbacks,
        )

    def _generate_recommendations(
        self,
        available: Dict[str, StackStatus],
        capabilities: Dict[str, StackCapability],
        tier: str,
    ) -> List[str]:
        """Generate usage recommendations."""
        recs = []

        # Count available vs needed
        available_count = sum(1 for s in available.values() if s == StackStatus.AVAILABLE)
        total = len(available)

        if available_count < total * 0.5:
            recs.append("Mais de 50% das stacks indisponiveis - considere modo degradado")

        # Check critical stacks
        critical = ["redis", "langfuse"]
        for stack in critical:
            if stack in available and available[stack] != StackStatus.AVAILABLE:
                recs.append(f"CRITICO: {stack} indisponivel - funcionalidade limitada")

        # Tier-specific recommendations
        if tier in ["L0", "L1", "L2"]:
            if "got" in available and available["got"] != StackStatus.AVAILABLE:
                recs.append("GoT indisponivel - reasoning multi-path limitado")

        if tier in ["L3", "L4", "L5"]:
            if "letta" in available and available["letta"] != StackStatus.AVAILABLE:
                recs.append("Letta indisponivel - memoria persistente limitada")

        # Eval stacks for QA tiers
        if "qa" in tier.lower() or tier in ["L3", "L4"]:
            eval_stacks = ["trulens", "ragas", "deepeval"]
            eval_available = sum(1 for s in eval_stacks if available.get(s) == StackStatus.AVAILABLE)
            if eval_available == 0:
                recs.append("Nenhuma stack de eval disponivel - validacao limitada")

        return recs

    def get_synergy_report(self) -> SynergyReport:
        """Generate comprehensive synergy report."""
        self._ensure_cerebro_mapping()
        if not self._health_cache:
            self.refresh_health()

        # Count statuses
        available_count = 0
        degraded_count = 0
        unavailable_count = 0

        for health in self._health_cache.values():
            if health.get("healthy"):
                if health.get("latency_ms", 0) > 1000:
                    degraded_count += 1
                else:
                    available_count += 1
            else:
                unavailable_count += 1

        total = len(self._health_cache)
        coverage = (available_count + degraded_count) / total * 100 if total > 0 else 0

        # Find gaps
        gaps = []
        recommendations = []

        # Check each cerebro's requirements
        if self._cerebro_mapping:
            for cerebro_id, config in self._cerebro_mapping.items():
                for stack in config.primary_stacks:
                    status = self.get_stack_status(stack)
                    if status == StackStatus.UNAVAILABLE:
                        gaps.append(f"{cerebro_id} requires {stack} (PRIMARY) but unavailable")

        # General recommendations
        if unavailable_count > total * 0.3:
            recommendations.append("More than 30% stacks unavailable - check Docker and dependencies")

        if "redis" in self._health_cache and not self._health_cache["redis"].get("healthy"):
            recommendations.append("Redis is DOWN - start Docker: docker-compose up -d redis")

        if "falkordb" in self._health_cache and not self._health_cache["falkordb"].get("healthy"):
            recommendations.append("FalkorDB is DOWN - start Docker: docker-compose up -d falkordb")

        return SynergyReport(
            total_stacks=total,
            available_count=available_count,
            degraded_count=degraded_count,
            unavailable_count=unavailable_count,
            coverage_pct=coverage,
            gaps=gaps,
            recommendations=recommendations,
            stack_health=self._health_cache,
        )

    def inject_context_into_prompt(
        self,
        agent_id: str,
        base_prompt: str,
        operation: Optional[str] = None,
    ) -> str:
        """Inject stack context into agent prompt.

        Args:
            agent_id: Cerebro identifier
            base_prompt: Original prompt text
            operation: Optional operation context

        Returns:
            Enhanced prompt with stack context
        """
        context = self.get_agent_context(agent_id, operation)
        stack_section = context.to_prompt_section()

        # Inject at the beginning of the prompt
        return f"{stack_section}\n\n---\n\n{base_prompt}"

    def get_best_stack_for_task(
        self,
        task_type: str,
        agent_id: Optional[str] = None,
    ) -> Optional[str]:
        """Recommend the best available stack for a task type.

        Args:
            task_type: Type of task (reasoning, memory, eval, security)
            agent_id: Optional agent to filter by configured stacks

        Returns:
            Best available stack name or None
        """
        # Get agent's configured stacks
        configured = None
        if agent_id:
            self._ensure_cerebro_mapping()
            config = self._get_cerebro_config(agent_id) if self._get_cerebro_config else None
            if config:
                configured = config.get_all_stacks()

        # Find stacks by category
        candidates = []
        for name, cap in STACK_CAPABILITIES.items():
            if cap.category == task_type or task_type in cap.operations:
                if configured is None or name in configured:
                    status = self.get_stack_status(name)
                    if status in [StackStatus.AVAILABLE, StackStatus.DEGRADED]:
                        candidates.append((name, cap.priority, status))

        if not candidates:
            return None

        # Sort by priority (lower is better), prefer AVAILABLE over DEGRADED
        candidates.sort(key=lambda x: (x[1], x[2] != StackStatus.AVAILABLE))

        return candidates[0][0]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


_synergy_manager: Optional[StackSynergyManager] = None


def get_synergy_manager() -> StackSynergyManager:
    """Get singleton synergy manager."""
    global _synergy_manager
    if _synergy_manager is None:
        _synergy_manager = StackSynergyManager()
    return _synergy_manager


def get_agent_stack_context(agent_id: str) -> AgentStackContext:
    """Get stack context for an agent."""
    return get_synergy_manager().get_agent_context(agent_id)


def inject_stacks_into_prompt(agent_id: str, prompt: str) -> str:
    """Inject stack context into a prompt."""
    return get_synergy_manager().inject_context_into_prompt(agent_id, prompt)


def get_synergy_report() -> SynergyReport:
    """Get synergy report."""
    return get_synergy_manager().get_synergy_report()


def get_best_stack(task_type: str, agent_id: Optional[str] = None) -> Optional[str]:
    """Get best available stack for a task type."""
    return get_synergy_manager().get_best_stack_for_task(task_type, agent_id)


# =============================================================================
# EXPORTS
# =============================================================================


# Alias for backwards compatibility
StackSynergy = StackSynergyManager

__all__ = [
    # Data structures
    "StackStatus",
    "StackCapability",
    "AgentStackContext",
    "SynergyReport",

    # Registry
    "STACK_CAPABILITIES",

    # Main class
    "StackSynergyManager",
    "StackSynergy",  # Alias

    # Convenience functions
    "get_synergy_manager",
    "get_agent_stack_context",
    "inject_stacks_into_prompt",
    "get_synergy_report",
    "get_best_stack",
]
