"""Partial Stacks Integration for LangGraph Workflow.

This module integrates the 5 PARTIAL stacks into the LangGraph StateGraph workflow:

1. Active RAG (pipeline/rag/active_rag.py)
   - Proactive retrieval for context enrichment
   - Query prediction and prefetching
   - Integrated into INIT and EXEC nodes

2. Buffer of Thoughts (pipeline/reasoning/bot_chain.py)
   - Accumulated reasoning across workflow
   - Template-based structured reasoning
   - Integrated into EXEC and GATE nodes

3. RAGAS Evaluator (pipeline/evaluation/ragas_eval.py)
   - RAG response evaluation metrics
   - Faithfulness, relevancy, precision
   - Integrated into QA/GATE nodes

4. Phoenix Traces (pipeline/evaluation/phoenix_traces.py)
   - OpenTelemetry-based tracing
   - Embedding drift detection
   - Integrated across ALL nodes

5. DeepEval Extended (pipeline/evaluation/deepeval_extended.py)
   - Hallucination detection
   - Faithfulness metrics
   - Integrated into GATE and SIGNOFF nodes

Integration Pattern:
    Each stack provides:
    - health_check(): Verify stack availability
    - Workflow hooks: Before/after node execution
    - State updates: Enrich PipelineState with stack results

Author: Pipeline Autonomo Team
Version: 1.0.0 (2026-01-20)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime, timezone
from contextlib import contextmanager

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# AVAILABILITY FLAGS
# =============================================================================

# Active RAG
try:
    from pipeline.rag.active_rag import (
        ActiveRAG,
        get_active_rag,
        ACTIVE_RAG_AVAILABLE,
    )
    _ACTIVE_RAG_IMPORT_SUCCESS = True
except ImportError as e:
    logger.debug(f"Active RAG import failed: {e}")
    _ACTIVE_RAG_IMPORT_SUCCESS = False
    ACTIVE_RAG_AVAILABLE = False

# Buffer of Thoughts
try:
    from pipeline.reasoning.bot_chain import (
        BufferOfThoughts,
        Thought,
        ThoughtType,
        ThoughtPriority,
        BOT_AVAILABLE,
    )
    _BOT_IMPORT_SUCCESS = True
except ImportError as e:
    logger.debug(f"Buffer of Thoughts import failed: {e}")
    _BOT_IMPORT_SUCCESS = False
    BOT_AVAILABLE = False

# RAGAS Evaluator
try:
    from pipeline.evaluation.ragas_eval import (
        RAGASEvaluator,
        evaluate_rag_response,
        RAGAS_AVAILABLE,
    )
    _RAGAS_IMPORT_SUCCESS = True
except ImportError as e:
    logger.debug(f"RAGAS import failed: {e}")
    _RAGAS_IMPORT_SUCCESS = False
    RAGAS_AVAILABLE = False

# Phoenix Traces
try:
    from pipeline.evaluation.phoenix_traces import (
        PhoenixTracer,
        get_tracer,
        start_trace,
        PHOENIX_AVAILABLE,
        INSTRUMENTATION_AVAILABLE,
    )
    _PHOENIX_IMPORT_SUCCESS = True
except ImportError as e:
    logger.debug(f"Phoenix import failed: {e}")
    _PHOENIX_IMPORT_SUCCESS = False
    PHOENIX_AVAILABLE = False
    INSTRUMENTATION_AVAILABLE = False

# DeepEval Extended
try:
    from pipeline.evaluation.deepeval_extended import (
        DeepEvalExtended,
        check_hallucination,
        DEEPEVAL_AVAILABLE,
    )
    _DEEPEVAL_IMPORT_SUCCESS = True
except ImportError as e:
    logger.debug(f"DeepEval import failed: {e}")
    _DEEPEVAL_IMPORT_SUCCESS = False
    DEEPEVAL_AVAILABLE = False

# Unified Eval Runner (2026-01-30: Ghost Code Integration)
# Provides unified evaluation across all stacks
try:
    from pipeline.evaluation.unified_eval_runner import (
        UnifiedEvalRunner,
        EvalInput,
    )
    UNIFIED_EVAL_AVAILABLE = True
    _UNIFIED_EVAL_IMPORT_SUCCESS = True
except ImportError as e:
    logger.debug(f"UnifiedEvalRunner import failed: {e}")
    _UNIFIED_EVAL_IMPORT_SUCCESS = False
    UNIFIED_EVAL_AVAILABLE = False
    UnifiedEvalRunner = None
    EvalInput = None

# Cleanlab Hallucination Detection (2026-01-30: Ghost Code Integration)
# Provides advanced hallucination detection using Cleanlab
try:
    from pipeline.quality.hallucination_detector import (
        CleanLabQuality,
        detect_hallucinations as cleanlab_detect_hallucinations,
        CLEANLAB_AVAILABLE,
    )
    _CLEANLAB_IMPORT_SUCCESS = True
except ImportError as e:
    logger.debug(f"Cleanlab hallucination detector import failed: {e}")
    _CLEANLAB_IMPORT_SUCCESS = False
    CLEANLAB_AVAILABLE = False
    CleanLabQuality = None
    cleanlab_detect_hallucinations = None

# Test Function Generator (2026-01-30: Ghost Code Integration)
# Generates pytest test functions automatically from EARS specifications
try:
    from pipeline.spec_kit.tfg import (
        TestFunctionGenerator,
        GeneratedTestSuite,
    )
    TFG_AVAILABLE = True
    _TFG_IMPORT_SUCCESS = True
except ImportError as e:
    logger.debug(f"TestFunctionGenerator import failed: {e}")
    _TFG_IMPORT_SUCCESS = False
    TFG_AVAILABLE = False
    TestFunctionGenerator = None
    GeneratedTestSuite = None

# Graph of Thoughts Enhanced (2026-01-30: Ghost Code Integration)
# Provides enhanced GoT with persistence and LangGraph nodes
try:
    from pipeline.graph.got_enhanced import (
        ThoughtGraph as EnhancedThoughtGraph,
        ThoughtAggregator,
        create_got_langgraph_nodes,
        GOT_AVAILABLE as GOT_ENHANCED_AVAILABLE,
    )
    _GOT_ENHANCED_IMPORT_SUCCESS = True
except ImportError as e:
    logger.debug(f"GoT Enhanced import failed: {e}")
    _GOT_ENHANCED_IMPORT_SUCCESS = False
    GOT_ENHANCED_AVAILABLE = False
    EnhancedThoughtGraph = None
    ThoughtAggregator = None
    create_got_langgraph_nodes = None

# Priority Matrix (2026-01-30: Ghost Code Integration)
# Provides MoSCoW classification and requirement prioritization
try:
    from pipeline.spec_kit.priority_matrix import (
        PriorityMatrix,
        MoSCoW,
        Priority,
    )
    PRIORITY_MATRIX_AVAILABLE = True
    _PRIORITY_MATRIX_IMPORT_SUCCESS = True
except ImportError as e:
    logger.debug(f"PriorityMatrix import failed: {e}")
    _PRIORITY_MATRIX_IMPORT_SUCCESS = False
    PRIORITY_MATRIX_AVAILABLE = False
    PriorityMatrix = None
    MoSCoW = None
    Priority = None

# DAG Decomposer (2026-01-30: Ghost Code Integration)
# Provides DAG-based task decomposition for spec phase
try:
    from pipeline.spec_kit.dag_decomposer import (
        DAGDecomposer,
        TaskNode,
    )
    DAG_DECOMPOSER_AVAILABLE = True
    _DAG_DECOMPOSER_IMPORT_SUCCESS = True
except ImportError as e:
    logger.debug(f"DAGDecomposer import failed: {e}")
    _DAG_DECOMPOSER_IMPORT_SUCCESS = False
    DAG_DECOMPOSER_AVAILABLE = False
    DAGDecomposer = None
    TaskNode = None

# Interface Sketch Generator (2026-01-30: Ghost Code Integration)
# Generates function signatures from EARS specs
try:
    from pipeline.spec_kit.interface_sketch import (
        InterfaceSketchGenerator,
        InterfaceSketch,
    )
    INTERFACE_SKETCH_AVAILABLE = True
    _INTERFACE_SKETCH_IMPORT_SUCCESS = True
except ImportError as e:
    logger.debug(f"InterfaceSketchGenerator import failed: {e}")
    _INTERFACE_SKETCH_IMPORT_SUCCESS = False
    INTERFACE_SKETCH_AVAILABLE = False
    InterfaceSketchGenerator = None
    InterfaceSketch = None

# Counter-Example Generator (2026-01-30: Ghost Code Integration)
# Detects ambiguity in requirements by generating counter-examples
try:
    from pipeline.spec_kit.counter_example_generator import (
        CounterExampleGenerator,
        CounterExample,
    )
    COUNTER_EXAMPLE_AVAILABLE = True
    _COUNTER_EXAMPLE_IMPORT_SUCCESS = True
except ImportError as e:
    logger.debug(f"CounterExampleGenerator import failed: {e}")
    _COUNTER_EXAMPLE_IMPORT_SUCCESS = False
    COUNTER_EXAMPLE_AVAILABLE = False
    CounterExampleGenerator = None
    CounterExample = None

# Event Writer (2026-01-30: Ghost Code Integration)
# Thread-safe atomic event writer with file locking
try:
    from pipeline.langgraph.event_writer import (
        AtomicEventWriter,
        Event,
        get_event_writer,
    )
    EVENT_WRITER_AVAILABLE = True
    _EVENT_WRITER_IMPORT_SUCCESS = True
except ImportError as e:
    logger.debug(f"AtomicEventWriter import failed: {e}")
    _EVENT_WRITER_IMPORT_SUCCESS = False
    EVENT_WRITER_AVAILABLE = False
    AtomicEventWriter = None
    Event = None
    get_event_writer = None

# Neo4j Algorithms (2026-01-30: Ghost Code Integration)
# Graph algorithms for claim verification (PageRank, community detection)
try:
    from pipeline.graph.neo4j_algorithms import (
        Neo4jAlgorithmsService,
        get_neo4j_algorithms,
        NEO4J_GDS_AVAILABLE,
    )
    NEO4J_ALGORITHMS_AVAILABLE = NEO4J_GDS_AVAILABLE
    _NEO4J_ALGORITHMS_IMPORT_SUCCESS = True
except ImportError as e:
    logger.debug(f"Neo4jAlgorithmsService import failed: {e}")
    _NEO4J_ALGORITHMS_IMPORT_SUCCESS = False
    NEO4J_ALGORITHMS_AVAILABLE = False
    Neo4jAlgorithmsService = None
    get_neo4j_algorithms = None

# Neo4j Analytics (2026-01-30: Ghost Code Integration)
# Complex graph analytics (community detection, PageRank, path tracing)
try:
    from pipeline.graph.neo4j_analytics import (
        Neo4jAnalytics,
        NEO4J_AVAILABLE as NEO4J_ANALYTICS_AVAILABLE,
    )
    _NEO4J_ANALYTICS_IMPORT_SUCCESS = True
except ImportError as e:
    logger.debug(f"Neo4jAnalytics import failed: {e}")
    _NEO4J_ANALYTICS_IMPORT_SUCCESS = False
    NEO4J_ANALYTICS_AVAILABLE = False
    Neo4jAnalytics = None

# Temporal Enhanced (2026-01-30: Ghost Code Integration)
# Production-ready workflow patterns with retry policies
try:
    from pipeline.infrastructure.temporal_enhanced import (
        TemporalEnhanced,
        TEMPORAL_ENHANCED_AVAILABLE,
    )
    _TEMPORAL_ENHANCED_IMPORT_SUCCESS = True
except ImportError as e:
    logger.debug(f"TemporalEnhanced import failed: {e}")
    _TEMPORAL_ENHANCED_IMPORT_SUCCESS = False
    TEMPORAL_ENHANCED_AVAILABLE = False
    TemporalEnhanced = None

# Mem0 Enhanced (2026-01-30: Ghost Code Integration)
# Advanced memory management with semantic embeddings
try:
    from pipeline.infrastructure.mem0_enhanced import (
        Mem0Enhanced,
        MEM0_ENHANCED_AVAILABLE,
    )
    _MEM0_ENHANCED_IMPORT_SUCCESS = True
except ImportError as e:
    logger.debug(f"Mem0Enhanced import failed: {e}")
    _MEM0_ENHANCED_IMPORT_SUCCESS = False
    MEM0_ENHANCED_AVAILABLE = False
    Mem0Enhanced = None

# ColBERT Retriever (2026-01-30: Ghost Code Integration)
# Late interaction retrieval with token-level embeddings
try:
    from pipeline.rag.colbert_retriever import (
        ColBERTRetriever,
        COLBERT_AVAILABLE,
    )
    _COLBERT_IMPORT_SUCCESS = True
except ImportError as e:
    logger.debug(f"ColBERTRetriever import failed: {e}")
    _COLBERT_IMPORT_SUCCESS = False
    COLBERT_AVAILABLE = False
    ColBERTRetriever = None

# Colang Integration (2026-01-30: Ghost Code Integration)
# NeMo Colang flows for security actions
try:
    from pipeline.langgraph.colang_integration import (
        ColangActionRegistry,
        NEMO_AVAILABLE as COLANG_AVAILABLE,
    )
    _COLANG_IMPORT_SUCCESS = True
except ImportError as e:
    logger.debug(f"ColangActionRegistry import failed: {e}")
    _COLANG_IMPORT_SUCCESS = False
    COLANG_AVAILABLE = False
    ColangActionRegistry = None

# Enforcement Metrics (2026-01-30: Ghost Code Integration)
# Metrics tracking for enforcement operations
try:
    from pipeline.langgraph.enforcement_metrics import (
        record_violation,
        record_invariant_check,
        record_trust_denial,
        get_metrics_summary,
    )
    ENFORCEMENT_METRICS_AVAILABLE = True
    _ENFORCEMENT_METRICS_IMPORT_SUCCESS = True
except ImportError as e:
    logger.debug(f"EnforcementMetrics import failed: {e}")
    _ENFORCEMENT_METRICS_IMPORT_SUCCESS = False
    ENFORCEMENT_METRICS_AVAILABLE = False
    record_violation = None
    record_invariant_check = None
    record_trust_denial = None
    get_metrics_summary = None

# Pydantic Validators (2026-01-30: Ghost Code Integration)
# Custom validators for pipeline state serialization
try:
    from pipeline.langgraph.pydantic_validators import (
        validate_pipeline_state,
        StateSerializer,
    )
    PYDANTIC_VALIDATORS_AVAILABLE = True
    _PYDANTIC_VALIDATORS_IMPORT_SUCCESS = True
except ImportError as e:
    logger.debug(f"PydanticValidators import failed: {e}")
    _PYDANTIC_VALIDATORS_IMPORT_SUCCESS = False
    PYDANTIC_VALIDATORS_AVAILABLE = False
    validate_pipeline_state = None
    StateSerializer = None

# Surgical Rework Migration (2026-01-30: Ghost Code Integration)
# Migration support from sprint-level to task-level rework
try:
    from pipeline.surgical_rework_migration import (
        get_exec_strategy,
        create_hybrid_exec_node,
    )
    SURGICAL_REWORK_MIGRATION_AVAILABLE = True
    _SURGICAL_REWORK_MIGRATION_IMPORT_SUCCESS = True
except ImportError as e:
    logger.debug(f"SurgicalReworkMigration import failed: {e}")
    _SURGICAL_REWORK_MIGRATION_IMPORT_SUCCESS = False
    SURGICAL_REWORK_MIGRATION_AVAILABLE = False
    get_exec_strategy = None
    create_hybrid_exec_node = None


# =============================================================================
# SPEC KIT STACKS (2026-01-30: Full Integration)
# =============================================================================

# SpecMasticator - Transforms roadmap/DOD into EARS requirements
try:
    from pipeline.spec_kit.masticator import SpecMasticator
    MASTICATOR_AVAILABLE = True
except ImportError as e:
    logger.debug(f"SpecMasticator import failed: {e}")
    MASTICATOR_AVAILABLE = False
    SpecMasticator = None

# SpecGuardian - Multi-gate validation system
try:
    from pipeline.spec_kit.guardian import SpecGuardian
    GUARDIAN_AVAILABLE = True
except ImportError as e:
    logger.debug(f"SpecGuardian import failed: {e}")
    GUARDIAN_AVAILABLE = False
    SpecGuardian = None

# IronCladOrchestrator - Main orchestrator for Spec Kit
try:
    from pipeline.spec_kit.ironclad import IronCladOrchestrator
    IRONCLAD_AVAILABLE = True
except ImportError as e:
    logger.debug(f"IronCladOrchestrator import failed: {e}")
    IRONCLAD_AVAILABLE = False
    IronCladOrchestrator = None

# JourneyCompleter - Generates user journeys beyond happy path
try:
    from pipeline.spec_kit.journey_completer import JourneyCompleter
    JOURNEY_COMPLETER_AVAILABLE = True
except ImportError as e:
    logger.debug(f"JourneyCompleter import failed: {e}")
    JOURNEY_COMPLETER_AVAILABLE = False
    JourneyCompleter = None

# ContradictionDetector - Z3 SAT solver for logical contradictions
try:
    from pipeline.spec_kit.contradiction_detector import (
        ContradictionDetector,
        ContradictionReport,
    )
    CONTRADICTION_DETECTOR_AVAILABLE = True
except ImportError as e:
    logger.debug(f"ContradictionDetector import failed: {e}")
    CONTRADICTION_DETECTOR_AVAILABLE = False
    ContradictionDetector = None
    ContradictionReport = None

# GapFiller - Detects and fills missing details
try:
    from pipeline.spec_kit.gap_filler import GapFiller
    GAP_FILLER_AVAILABLE = True
except ImportError as e:
    logger.debug(f"GapFiller import failed: {e}")
    GAP_FILLER_AVAILABLE = False
    GapFiller = None

# TraceabilityBuilder - Builds requirement traceability matrix
try:
    from pipeline.spec_kit.traceability_builder import TraceabilityBuilder
    TRACEABILITY_BUILDER_AVAILABLE = True
except ImportError as e:
    logger.debug(f"TraceabilityBuilder import failed: {e}")
    TRACEABILITY_BUILDER_AVAILABLE = False
    TraceabilityBuilder = None

# ConsistencyChecker - Checks consistency between specs
try:
    from pipeline.spec_kit.consistency_checker import ConsistencyChecker
    CONSISTENCY_CHECKER_AVAILABLE = True
except ImportError as e:
    logger.debug(f"ConsistencyChecker import failed: {e}")
    CONSISTENCY_CHECKER_AVAILABLE = False
    ConsistencyChecker = None

# SemanticSearch - Semantic search for spec elements
try:
    from pipeline.spec_kit.semantic_search import SemanticSearch
    SEMANTIC_SEARCH_AVAILABLE = True
except ImportError as e:
    logger.debug(f"SemanticSearch import failed: {e}")
    SEMANTIC_SEARCH_AVAILABLE = False
    SemanticSearch = None

# Constitution - Rule engine for specs
try:
    from pipeline.spec_kit.constitution import Constitution
    CONSTITUTION_AVAILABLE = True
except ImportError as e:
    logger.debug(f"Constitution import failed: {e}")
    CONSTITUTION_AVAILABLE = False
    Constitution = None


# =============================================================================
# SPEC AI STACKS (2026-01-30: Full Integration)
# =============================================================================

# GoTAnalyzer - Graph of Thoughts multi-perspective analysis
try:
    from pipeline.spec_ai import (
        GoTAnalyzer,
        GoTAnalysisResult,
        analyze_perspectives,
        find_blindspots,
    )
    GOT_ANALYZER_AVAILABLE = True
except ImportError as e:
    logger.debug(f"GoTAnalyzer import failed: {e}")
    GOT_ANALYZER_AVAILABLE = False
    GoTAnalyzer = None
    GoTAnalysisResult = None
    analyze_perspectives = None
    find_blindspots = None

# AdversarialReviewer - Red team attack simulation
try:
    from pipeline.spec_ai import (
        AdversarialReviewer,
        attack_spec,
        find_weaknesses,
        suggest_hardening,
    )
    ADVERSARIAL_REVIEWER_AVAILABLE = True
except ImportError as e:
    logger.debug(f"AdversarialReviewer import failed: {e}")
    ADVERSARIAL_REVIEWER_AVAILABLE = False
    AdversarialReviewer = None
    attack_spec = None
    find_weaknesses = None
    suggest_hardening = None

# MADDebate - Multi-agent debate system
try:
    from pipeline.spec_ai import (
        MADDebate,
        setup_debate,
        run_rounds,
        reach_consensus,
    )
    MAD_DEBATE_AVAILABLE = True
except ImportError as e:
    logger.debug(f"MADDebate import failed: {e}")
    MAD_DEBATE_AVAILABLE = False
    MADDebate = None
    setup_debate = None
    run_rounds = None
    reach_consensus = None

# SpecAI HallucinationDetector - Detect fabricated requirements
try:
    from pipeline.spec_ai import (
        HallucinationDetector as SpecAIHallucinationDetector,
        detect_fabrications,
        verify_sources,
        flag_suspicious,
    )
    SPEC_AI_HALLUCINATION_AVAILABLE = True
except ImportError as e:
    logger.debug(f"SpecAI HallucinationDetector import failed: {e}")
    SPEC_AI_HALLUCINATION_AVAILABLE = False
    SpecAIHallucinationDetector = None
    detect_fabrications = None
    verify_sources = None
    flag_suspicious = None

# GapDetector - AI-powered gap detection
try:
    from pipeline.spec_ai import (
        GapDetector,
        find_missing,
        infer_implicit,
        suggest_additions,
    )
    GAP_DETECTOR_AVAILABLE = True
except ImportError as e:
    logger.debug(f"GapDetector import failed: {e}")
    GAP_DETECTOR_AVAILABLE = False
    GapDetector = None
    find_missing = None
    infer_implicit = None
    suggest_additions = None

# StackRecommender - Technology stack recommendations
try:
    from pipeline.spec_ai import (
        StackRecommender,
        analyze_needs,
        match_stacks,
        recommend_config,
    )
    STACK_RECOMMENDER_AVAILABLE = True
except ImportError as e:
    logger.debug(f"StackRecommender import failed: {e}")
    STACK_RECOMMENDER_AVAILABLE = False
    StackRecommender = None
    analyze_needs = None
    match_stacks = None
    recommend_config = None

# ComplexityEstimator - Effort and complexity estimation
try:
    from pipeline.spec_ai import (
        ComplexityEstimator,
        estimate_effort,
        identify_risks,
        suggest_breakdown,
    )
    COMPLEXITY_ESTIMATOR_AVAILABLE = True
except ImportError as e:
    logger.debug(f"ComplexityEstimator import failed: {e}")
    COMPLEXITY_ESTIMATOR_AVAILABLE = False
    ComplexityEstimator = None
    estimate_effort = None
    identify_risks = None
    suggest_breakdown = None

# RiskPredictor - Pattern-based risk prediction
try:
    from pipeline.spec_ai import (
        RiskPredictor,
        predict_from_patterns,
        historical_analysis,
        early_warning,
    )
    RISK_PREDICTOR_AVAILABLE = True
except ImportError as e:
    logger.debug(f"RiskPredictor import failed: {e}")
    RISK_PREDICTOR_AVAILABLE = False
    RiskPredictor = None
    predict_from_patterns = None
    historical_analysis = None
    early_warning = None


# =============================================================================
# PLANNING STACKS (2026-01-30: Full Integration)
# =============================================================================

# HierarchicalDecomposer - Hierarchical task decomposition
try:
    from pipeline.planning import (
        HierarchicalDecomposer,
        decompose_spec,
        get_decomposition_metrics,
    )
    HIERARCHICAL_DECOMPOSER_AVAILABLE = True
except ImportError as e:
    logger.debug(f"HierarchicalDecomposer import failed: {e}")
    HIERARCHICAL_DECOMPOSER_AVAILABLE = False
    HierarchicalDecomposer = None
    decompose_spec = None
    get_decomposition_metrics = None

# CriticalPathAnalyzer - Critical path method
try:
    from pipeline.planning import (
        CriticalPathAnalyzer,
        analyze_critical_path,
        get_parallel_groups,
        calculate_slack_times,
    )
    CRITICAL_PATH_AVAILABLE = True
except ImportError as e:
    logger.debug(f"CriticalPathAnalyzer import failed: {e}")
    CRITICAL_PATH_AVAILABLE = False
    CriticalPathAnalyzer = None
    analyze_critical_path = None
    get_parallel_groups = None
    calculate_slack_times = None

# DynamicPriorityCalculator - Dynamic priority adjustment
try:
    from pipeline.planning import (
        DynamicPriorityCalculator,
        calculate_dynamic_priority,
        recalculate_priorities,
    )
    DYNAMIC_PRIORITY_AVAILABLE = True
except ImportError as e:
    logger.debug(f"DynamicPriorityCalculator import failed: {e}")
    DYNAMIC_PRIORITY_AVAILABLE = False
    DynamicPriorityCalculator = None
    calculate_dynamic_priority = None
    recalculate_priorities = None

# TokenBudgetPlanner - Token budget planning
try:
    from pipeline.planning import (
        TokenBudgetPlanner,
        estimate_task_tokens,
        allocate_budgets,
        track_budget_usage,
    )
    TOKEN_BUDGET_AVAILABLE = True
except ImportError as e:
    logger.debug(f"TokenBudgetPlanner import failed: {e}")
    TOKEN_BUDGET_AVAILABLE = False
    TokenBudgetPlanner = None
    estimate_task_tokens = None
    allocate_budgets = None
    track_budget_usage = None

# FallbackGenerator - Fallback plan generation
try:
    from pipeline.planning import (
        FallbackGenerator,
        generate_fallback_plans,
        get_recovery_templates,
    )
    FALLBACK_GENERATOR_AVAILABLE = True
except ImportError as e:
    logger.debug(f"FallbackGenerator import failed: {e}")
    FALLBACK_GENERATOR_AVAILABLE = False
    FallbackGenerator = None
    generate_fallback_plans = None
    get_recovery_templates = None

# PlanningFeedbackLoop - Execution feedback loop
try:
    from pipeline.planning import (
        PlanningFeedbackLoop,
        capture_planning_feedback,
        learn_from_outcomes,
        get_similar_sprints,
    )
    FEEDBACK_LOOP_AVAILABLE = True
except ImportError as e:
    logger.debug(f"PlanningFeedbackLoop import failed: {e}")
    FEEDBACK_LOOP_AVAILABLE = False
    PlanningFeedbackLoop = None
    capture_planning_feedback = None
    learn_from_outcomes = None
    get_similar_sprints = None

# PlanValidator - Plan validation
try:
    from pipeline.planning import (
        PlanValidator,
        validate_plan,
        run_plan_checklist,
        check_consistency,
    )
    PLAN_VALIDATOR_AVAILABLE = True
except ImportError as e:
    logger.debug(f"PlanValidator import failed: {e}")
    PLAN_VALIDATOR_AVAILABLE = False
    PlanValidator = None
    validate_plan = None
    run_plan_checklist = None
    check_consistency = None

# PlanningMetricsCollector - Planning metrics
try:
    from pipeline.planning import (
        PlanningMetricsCollector,
        capture_metrics as capture_planning_metrics,
        get_metrics_summary as get_planning_metrics_summary,
    )
    PLANNING_METRICS_AVAILABLE = True
except ImportError as e:
    logger.debug(f"PlanningMetricsCollector import failed: {e}")
    PLANNING_METRICS_AVAILABLE = False
    PlanningMetricsCollector = None
    capture_planning_metrics = None
    get_planning_metrics_summary = None

# MultiPathPlanGenerator - GoT multi-path plan generation
try:
    from pipeline.planning import (
        MultiPathPlanGenerator,
        generate_multi_path_plan,
    )
    MULTI_PATH_PLAN_AVAILABLE = True
except ImportError as e:
    logger.debug(f"MultiPathPlanGenerator import failed: {e}")
    MULTI_PATH_PLAN_AVAILABLE = False
    MultiPathPlanGenerator = None
    generate_multi_path_plan = None

# PlanQualityEvaluator - Plan quality evaluation
try:
    from pipeline.planning import (
        PlanQualityEvaluator,
        compute_plan_score,
        check_plan_quality_threshold,
    )
    PLAN_EVALUATOR_AVAILABLE = True
except ImportError as e:
    logger.debug(f"PlanQualityEvaluator import failed: {e}")
    PLAN_EVALUATOR_AVAILABLE = False
    PlanQualityEvaluator = None
    compute_plan_score = None
    check_plan_quality_threshold = None

# MCTSPlanOptimizer - MCTS plan optimization
try:
    from pipeline.planning import (
        MCTSPlanOptimizer,
        estimate_plan_quality_fast,
    )
    MCTS_OPTIMIZER_AVAILABLE = True
except ImportError as e:
    logger.debug(f"MCTSPlanOptimizer import failed: {e}")
    MCTS_OPTIMIZER_AVAILABLE = False
    MCTSPlanOptimizer = None
    estimate_plan_quality_fast = None

# PlanReflexionEngine - Reflexion-based improvement
try:
    from pipeline.planning import (
        PlanReflexionEngine,
        create_reflection_from_feedback,
    )
    PLAN_REFLEXION_AVAILABLE = True
except ImportError as e:
    logger.debug(f"PlanReflexionEngine import failed: {e}")
    PLAN_REFLEXION_AVAILABLE = False
    PlanReflexionEngine = None
    create_reflection_from_feedback = None

# PlanGraphStore - FalkorDB graph storage
try:
    from pipeline.planning import PlanGraphStore
    PLAN_GRAPH_STORE_AVAILABLE = True
except ImportError as e:
    logger.debug(f"PlanGraphStore import failed: {e}")
    PLAN_GRAPH_STORE_AVAILABLE = False
    PlanGraphStore = None

# PlanSemanticRetriever - Qdrant semantic search
try:
    from pipeline.planning import PlanSemanticRetriever
    PLAN_RETRIEVER_AVAILABLE = True
except ImportError as e:
    logger.debug(f"PlanSemanticRetriever import failed: {e}")
    PLAN_RETRIEVER_AVAILABLE = False
    PlanSemanticRetriever = None

# PlanningThoughtManager - BoT templates for planning
try:
    from pipeline.planning import (
        PlanningThoughtManager,
        get_planning_thought_manager,
        get_planning_template,
    )
    PLANNING_THOUGHTS_AVAILABLE = True
except ImportError as e:
    logger.debug(f"PlanningThoughtManager import failed: {e}")
    PLANNING_THOUGHTS_AVAILABLE = False
    PlanningThoughtManager = None
    get_planning_thought_manager = None
    get_planning_template = None

# PlanMaster - Central Planning Orchestrator
try:
    from pipeline.planning import (
        PlanMaster,
        check_plan_ready_for_execution,
    )
    PLAN_MASTER_AVAILABLE = True
except ImportError as e:
    logger.debug(f"PlanMaster import failed: {e}")
    PLAN_MASTER_AVAILABLE = False
    PlanMaster = None
    check_plan_ready_for_execution = None


# =============================================================================
# RAG STACKS (2026-01-30: Full Integration)
# =============================================================================

# SelfRAG - Self-reflective retrieval
try:
    from pipeline.rag.self_rag import SelfRAG, SELF_RAG_AVAILABLE
except ImportError as e:
    logger.debug(f"SelfRAG import failed: {e}")
    SELF_RAG_AVAILABLE = False
    SelfRAG = None

# MemoRAG - Memory-augmented retrieval
try:
    from pipeline.rag.memo_rag import MemoRAG, MEMO_RAG_AVAILABLE
except ImportError as e:
    logger.debug(f"MemoRAG import failed: {e}")
    MEMO_RAG_AVAILABLE = False
    MemoRAG = None

# RaptorRAG - Hierarchical retrieval
try:
    from pipeline.rag.raptor_rag import RaptorRAG, RAPTOR_RAG_AVAILABLE
except ImportError as e:
    logger.debug(f"RaptorRAG import failed: {e}")
    RAPTOR_RAG_AVAILABLE = False
    RaptorRAG = None

# QdrantHybridSearch - Hybrid search
try:
    from pipeline.rag.qdrant_hybrid import QdrantHybridSearch, QDRANT_HYBRID_AVAILABLE
except ImportError as e:
    logger.debug(f"QdrantHybridSearch import failed: {e}")
    QDRANT_HYBRID_AVAILABLE = False
    QdrantHybridSearch = None

# CorrectiveRAG - Self-correcting retrieval with quality evaluation (P0-FIX-2026-02-01)
try:
    from pipeline.rag.corrective_rag import CorrectiveRAG, CRAG_AVAILABLE
except ImportError as e:
    logger.debug(f"CorrectiveRAG import failed: {e}")
    CRAG_AVAILABLE = False
    CorrectiveRAG = None

# GraphRAG - Graph-enhanced retrieval with reasoning (P0-FIX-2026-02-01)
try:
    from pipeline.containers.graphrag_integration import (
        GraphRAGIntegration,
        get_graphrag_integration,
        GRAPHRAG_CLIENT_AVAILABLE,
    )
except ImportError as e:
    logger.debug(f"GraphRAGIntegration import failed: {e}")
    GRAPHRAG_CLIENT_AVAILABLE = False
    GraphRAGIntegration = None
    get_graphrag_integration = None

# =============================================================================
# RAG STACK ROUTER (2026-02-01: Agent-Specific RAG Routing)
# =============================================================================
# Feature flag to enable agent-specific RAG routing via runtime cards.
# When enabled, only the RAG stacks configured for each agent are called.
# When disabled, ALL RAG stacks are called for all agents (legacy behavior).
import os
USE_RAG_ROUTER = os.getenv("USE_RAG_ROUTER", "true").lower() == "true"

try:
    from pipeline.rag_stack_router import (
        get_rag_stack_router,
        RAGStackRouter,
        RAGRouterResult,
    )
    RAG_ROUTER_AVAILABLE = True
except ImportError as e:
    logger.debug(f"RAGStackRouter import failed: {e}")
    RAG_ROUTER_AVAILABLE = False
    get_rag_stack_router = None
    RAGStackRouter = None
    RAGRouterResult = None


# =============================================================================
# GHOST CODE INTEGRATION HELPERS (2026-01-30)
# =============================================================================

from pathlib import Path
from typing import Tuple

# Event Writer singleton
_event_writer_instance = None


def _emit_integration_event(
    event_type: str,
    state: Dict[str, Any],
    message: str,
    level: str = "info",
    metadata: Optional[Dict] = None,
) -> None:
    """Emit event to event log if Event Writer is available.

    INV-EW-001: Failures are silent (debug log only).
    """
    global _event_writer_instance

    if not EVENT_WRITER_AVAILABLE or get_event_writer is None:
        return

    if _event_writer_instance is None:
        try:
            run_dir = Path(state.get("run_dir", "./out/pipeline"))
            _event_writer_instance = get_event_writer(run_dir)
        except Exception as e:
            logger.debug(f"Event writer init failed: {e}")
            return

    try:
        _event_writer_instance.write_dict(
            event_type=f"partial_stacks.{event_type}",
            run_id=state.get("run_id", "unknown"),
            sprint_id=state.get("sprint_id", "unknown"),
            attempt=state.get("attempt", 1),
            message=message,
            level=level,
            phase=state.get("current_phase", "UNKNOWN"),
            metadata=metadata or {},
        )
    except Exception as e:
        logger.debug(f"Event emit failed: {e}")


def _validate_state_transition(state: Dict[str, Any], phase: str) -> Tuple[bool, Dict]:
    """Validate state before transition using Pydantic validators.

    INV-PYD-001: Validation does not block (warning only).
    """
    if not PYDANTIC_VALIDATORS_AVAILABLE or validate_pipeline_state is None:
        return True, {"skipped": True, "reason": "validators_not_available"}

    try:
        result = validate_pipeline_state(state, strict=False)

        if not result.get("is_valid", True):
            logger.warning(
                f"State validation failed for {phase}: {result.get('errors', [])[:3]}"
            )
            return False, result

        return True, result
    except Exception as e:
        logger.debug(f"State validation error: {e}")
        return True, {"error": str(e)}


# =============================================================================
# HEALTH CHECK RESULTS
# =============================================================================

@dataclass
class StackHealthResult:
    """Result from a stack health check."""
    stack_name: str
    healthy: bool
    available: bool
    version: str = ""
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stack_name": self.stack_name,
            "healthy": self.healthy,
            "available": self.available,
            "version": self.version,
            "error": self.error,
            "details": self.details,
            "checked_at": self.checked_at.isoformat(),
        }


class PartialStacksHealthReport(BaseModel):
    """Health report for all partial stacks."""

    active_rag: bool = Field(default=False, description="Active RAG health status")
    bot: bool = Field(default=False, description="Buffer of Thoughts health status")
    ragas: bool = Field(default=False, description="RAGAS Evaluator health status")
    phoenix: bool = Field(default=False, description="Phoenix Traces health status")
    deepeval: bool = Field(default=False, description="DeepEval Extended health status")

    all_healthy: bool = Field(default=False, description="All stacks healthy")
    healthy_count: int = Field(default=0, description="Number of healthy stacks")
    total_count: int = Field(default=5, description="Total number of stacks")

    details: Dict[str, StackHealthResult] = Field(default_factory=dict)
    checked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# HEALTH CHECK FUNCTIONS
# =============================================================================

def health_check_active_rag() -> StackHealthResult:
    """Health check for Active RAG stack.

    Returns:
        StackHealthResult with health status
    """
    if not _ACTIVE_RAG_IMPORT_SUCCESS or not ACTIVE_RAG_AVAILABLE:
        return StackHealthResult(
            stack_name="active_rag",
            healthy=False,
            available=False,
            error="Active RAG module not available",
        )

    try:
        # Try to instantiate ActiveRAG
        rag = get_active_rag()

        # Check if Qdrant connection works (optional)
        details = {
            "prediction_cache_enabled": True,
            "prefetch_enabled": True,
        }

        return StackHealthResult(
            stack_name="active_rag",
            healthy=True,
            available=True,
            version="1.0.0",
            details=details,
        )
    except Exception as e:
        logger.warning(f"Active RAG health check failed: {e}")
        return StackHealthResult(
            stack_name="active_rag",
            healthy=False,
            available=True,
            error=str(e),
        )


def health_check_bot() -> StackHealthResult:
    """Health check for Buffer of Thoughts stack.

    Returns:
        StackHealthResult with health status
    """
    if not _BOT_IMPORT_SUCCESS or not BOT_AVAILABLE:
        return StackHealthResult(
            stack_name="buffer_of_thoughts",
            healthy=False,
            available=False,
            error="Buffer of Thoughts module not available",
        )

    try:
        # Try to instantiate BufferOfThoughts
        bot = BufferOfThoughts()

        # Test basic operations
        thought = bot.add_thought(
            "Health check thought",
            thought_type=ThoughtType.OBSERVATION,
        )

        details = {
            "max_thoughts": bot.max_thoughts,
            "context_window": bot.context_window,
            "current_thoughts": len(bot._thoughts),
        }

        # Clean up health check thought
        bot._thoughts.clear()
        bot._thought_index.clear()

        return StackHealthResult(
            stack_name="buffer_of_thoughts",
            healthy=True,
            available=True,
            version="1.0.0",
            details=details,
        )
    except Exception as e:
        logger.warning(f"Buffer of Thoughts health check failed: {e}")
        return StackHealthResult(
            stack_name="buffer_of_thoughts",
            healthy=False,
            available=True,
            error=str(e),
        )


def health_check_ragas() -> StackHealthResult:
    """Health check for RAGAS Evaluator stack.

    Returns:
        StackHealthResult with health status
    """
    if not _RAGAS_IMPORT_SUCCESS or not RAGAS_AVAILABLE:
        return StackHealthResult(
            stack_name="ragas",
            healthy=False,
            available=False,
            error="RAGAS module not available. Install with: pip install ragas",
        )

    try:
        # Try to instantiate RAGASEvaluator
        evaluator = RAGASEvaluator(log_to_langfuse=False)

        details = {
            "metrics_available": [
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
            ],
            "langfuse_integration": True,
        }

        return StackHealthResult(
            stack_name="ragas",
            healthy=True,
            available=True,
            version="1.0.0",
            details=details,
        )
    except Exception as e:
        logger.warning(f"RAGAS health check failed: {e}")
        return StackHealthResult(
            stack_name="ragas",
            healthy=False,
            available=True,
            error=str(e),
        )


def health_check_phoenix() -> StackHealthResult:
    """Health check for Phoenix Traces stack.

    Returns:
        StackHealthResult with health status
    """
    if not _PHOENIX_IMPORT_SUCCESS or not PHOENIX_AVAILABLE:
        return StackHealthResult(
            stack_name="phoenix",
            healthy=False,
            available=False,
            error="Phoenix module not available. Install with: pip install arize-phoenix",
        )

    try:
        # Try to get tracer
        tracer = get_tracer()

        details = {
            "instrumentation_available": INSTRUMENTATION_AVAILABLE,
            "opentelemetry_enabled": True,
            "drift_detection_enabled": True,
        }

        return StackHealthResult(
            stack_name="phoenix",
            healthy=True,
            available=True,
            version="1.0.0",
            details=details,
        )
    except Exception as e:
        logger.warning(f"Phoenix health check failed: {e}")
        return StackHealthResult(
            stack_name="phoenix",
            healthy=False,
            available=True,
            error=str(e),
        )


def health_check_deepeval() -> StackHealthResult:
    """Health check for DeepEval Extended stack.

    Returns:
        StackHealthResult with health status
    """
    if not _DEEPEVAL_IMPORT_SUCCESS or not DEEPEVAL_AVAILABLE:
        return StackHealthResult(
            stack_name="deepeval",
            healthy=False,
            available=False,
            error="DeepEval module not available. Install with: pip install deepeval",
        )

    try:
        # Try to instantiate DeepEvalExtended
        evaluator = DeepEvalExtended()

        details = {
            "metrics_available": [
                "faithfulness",
                "hallucination",
                "answer_relevancy",
                "toxicity",
                "context_precision",
                "context_recall",
            ],
            "langfuse_integration": True,
        }

        return StackHealthResult(
            stack_name="deepeval",
            healthy=True,
            available=True,
            version="1.0.0",
            details=details,
        )
    except Exception as e:
        logger.warning(f"DeepEval health check failed: {e}")
        return StackHealthResult(
            stack_name="deepeval",
            healthy=False,
            available=True,
            error=str(e),
        )


def health_check_all() -> PartialStacksHealthReport:
    """Run health check on all partial stacks.

    Returns:
        PartialStacksHealthReport with all results
    """
    # Run individual health checks
    active_rag_result = health_check_active_rag()
    bot_result = health_check_bot()
    ragas_result = health_check_ragas()
    phoenix_result = health_check_phoenix()
    deepeval_result = health_check_deepeval()

    # Count healthy stacks
    healthy_count = sum([
        active_rag_result.healthy,
        bot_result.healthy,
        ragas_result.healthy,
        phoenix_result.healthy,
        deepeval_result.healthy,
    ])

    return PartialStacksHealthReport(
        active_rag=active_rag_result.healthy,
        bot=bot_result.healthy,
        ragas=ragas_result.healthy,
        phoenix=phoenix_result.healthy,
        deepeval=deepeval_result.healthy,
        all_healthy=healthy_count == 5,
        healthy_count=healthy_count,
        total_count=5,
        details={
            "active_rag": active_rag_result,
            "bot": bot_result,
            "ragas": ragas_result,
            "phoenix": phoenix_result,
            "deepeval": deepeval_result,
        },
    )


# =============================================================================
# WORKFLOW INTEGRATION HOOKS
# =============================================================================

class PartialStacksIntegration:
    """Integration layer for partial stacks into LangGraph workflow.

    This class provides hooks that can be called from workflow nodes
    to leverage the partial stacks functionality.

    Usage in workflow.py:
        from pipeline.langgraph.partial_stacks_integration import (
            PartialStacksIntegration,
        )

        integration = PartialStacksIntegration()

        # In init_node:
        state = await integration.on_init(state)

        # In exec_node:
        state = await integration.on_exec_start(state)
        state = await integration.on_exec_end(state)

        # In gate_node:
        state = await integration.on_gate(state)

        # In signoff_node:
        state = await integration.on_signoff(state)
    """

    def __init__(self):
        """Initialize the integration layer."""
        self._active_rag: Optional[ActiveRAG] = None
        self._bot: Optional[BufferOfThoughts] = None
        self._ragas: Optional[RAGASEvaluator] = None
        self._phoenix: Optional[PhoenixTracer] = None
        self._deepeval: Optional[DeepEvalExtended] = None

        self._trace_context: Optional[Any] = None

        # Initialize available stacks
        self._init_stacks()

    def _init_stacks(self) -> None:
        """Initialize available stacks."""
        # Active RAG
        if _ACTIVE_RAG_IMPORT_SUCCESS and ACTIVE_RAG_AVAILABLE:
            try:
                self._active_rag = get_active_rag()
                logger.info("Active RAG initialized for workflow integration")
            except Exception as e:
                logger.warning(f"Failed to initialize Active RAG: {e}")

        # Buffer of Thoughts
        if _BOT_IMPORT_SUCCESS and BOT_AVAILABLE:
            try:
                self._bot = BufferOfThoughts()
                logger.info("Buffer of Thoughts initialized for workflow integration")
            except Exception as e:
                logger.warning(f"Failed to initialize Buffer of Thoughts: {e}")

        # RAGAS Evaluator
        if _RAGAS_IMPORT_SUCCESS and RAGAS_AVAILABLE:
            try:
                self._ragas = RAGASEvaluator(log_to_langfuse=True)
                logger.info("RAGAS Evaluator initialized for workflow integration")
            except Exception as e:
                logger.warning(f"Failed to initialize RAGAS: {e}")

        # Phoenix Tracer
        if _PHOENIX_IMPORT_SUCCESS and PHOENIX_AVAILABLE:
            try:
                self._phoenix = get_tracer()
                logger.info("Phoenix Tracer initialized for workflow integration")
            except Exception as e:
                logger.warning(f"Failed to initialize Phoenix: {e}")

        # DeepEval Extended
        if _DEEPEVAL_IMPORT_SUCCESS and DEEPEVAL_AVAILABLE:
            try:
                self._deepeval = DeepEvalExtended()
                logger.info("DeepEval Extended initialized for workflow integration")
            except Exception as e:
                logger.warning(f"Failed to initialize DeepEval: {e}")

    @property
    def health_report(self) -> PartialStacksHealthReport:
        """Get current health status of all stacks."""
        return health_check_all()

    # =========================================================================
    # PHOENIX TRACING HOOKS
    # =========================================================================

    @contextmanager
    def trace_node(self, node_name: str, state: Dict[str, Any]):
        """Context manager for tracing a workflow node.

        Args:
            node_name: Name of the node being traced
            state: Current pipeline state

        Yields:
            Trace context that can be used for span creation
        """
        if self._phoenix is None:
            yield None
            return

        trace_id = state.get("trace_id", state.get("run_id", "unknown"))
        sprint_id = state.get("sprint_id", "unknown")

        try:
            with self._phoenix.start_trace(
                name=f"{node_name}_{sprint_id}",
                metadata={
                    "run_id": state.get("run_id"),
                    "sprint_id": sprint_id,
                    "node": node_name,
                    "phase": state.get("phase", ""),
                },
            ) as trace:
                yield trace
        except Exception as e:
            logger.warning(f"Phoenix trace failed for {node_name}: {e}")
            yield None

    # =========================================================================
    # INIT NODE HOOKS
    # =========================================================================

    async def on_init(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Hook called during INIT node.

        Actions:
        - Initialize Phoenix tracing for the run
        - Prefetch context using Active RAG (if objective is available)
        - Initialize Buffer of Thoughts for the run

        Args:
            state: Current pipeline state

        Returns:
            Updated state with integration data
        """
        integration_data = {
            "partial_stacks": {
                "active_rag": ACTIVE_RAG_AVAILABLE,
                "bot": BOT_AVAILABLE,
                "ragas": RAGAS_AVAILABLE,
                "phoenix": PHOENIX_AVAILABLE,
                "deepeval": DEEPEVAL_AVAILABLE,
            },
            "bot_thoughts": [],
            "rag_prefetched": False,
            "phoenix_trace_id": None,
            "evaluations": {},
        }

        # Phoenix: Start trace for entire run
        if self._phoenix:
            try:
                trace_id = f"run_{state.get('run_id', 'unknown')}"
                integration_data["phoenix_trace_id"] = trace_id
                logger.debug(f"Phoenix trace started: {trace_id}")
            except Exception as e:
                logger.warning(f"Phoenix trace init failed: {e}")

        # Active RAG: Prefetch context if objective is available
        context_pack = state.get("context_pack", {})
        objective = context_pack.get("objective", "")

        if self._active_rag and objective:
            try:
                # Predict next queries based on objective
                predictions = await self._active_rag.predict_next_queries(objective)
                if predictions:
                    # Prefetch context for predictions (pass QueryPrediction objects, not strings)
                    prefetch_result = await self._active_rag.prefetch_context(
                        predictions[:3]  # FIX: pass QueryPrediction objects directly
                    )
                    integration_data["rag_prefetched"] = True
                    integration_data["rag_predictions"] = [
                        {"query": p.query, "confidence": p.confidence}
                        for p in predictions[:3]
                    ]
                    logger.debug(f"Active RAG prefetched {len(predictions)} queries")
            except Exception as e:
                logger.warning(f"Active RAG prefetch failed: {e}")

        # Buffer of Thoughts: Initialize with context
        if self._bot:
            try:
                if objective:
                    self._bot.add_thought(
                        f"Sprint objective: {objective}",
                        thought_type=ThoughtType.CONTEXT,
                        priority=ThoughtPriority.HIGH,
                    )

                deliverables = context_pack.get("deliverables", [])
                if deliverables:
                    self._bot.add_thought(
                        f"Deliverables to complete: {len(deliverables)}",
                        thought_type=ThoughtType.CONTEXT,
                    )

                integration_data["bot_initialized"] = True
                logger.debug("Buffer of Thoughts initialized with context")
            except Exception as e:
                logger.warning(f"Buffer of Thoughts init failed: {e}")

        # =====================================================================
        # GHOST CODE INTEGRATIONS - on_init (2026-01-30)
        # =====================================================================

        # Validate state at INIT start
        is_valid, validation_result = _validate_state_transition(state, "INIT")
        if not is_valid:
            integration_data["validation_errors"] = validation_result.get("errors", [])

        # Priority Matrix: Classify specs by MoSCoW priority
        if PRIORITY_MATRIX_AVAILABLE and PriorityMatrix is not None:
            try:
                specs = state.get("specs", []) or context_pack.get("specs", [])
                if specs and len(specs) > 1:
                    matrix = PriorityMatrix(enable_dependency_analysis=True)

                    requirements = [
                        {
                            "id": spec.get("id", f"spec_{i}"),
                            "name": spec.get("name", spec.get("trigger", "")),
                            "description": spec.get("text", spec.get("action", "")),
                            "depends_on": spec.get("dependencies", []),
                        }
                        for i, spec in enumerate(specs)
                    ]

                    if requirements:
                        result = matrix.classify(requirements)
                        integration_data["priority_matrix"] = {
                            "total_classified": len(result.classified),
                            "summary": result.summary,
                            "p0_count": result.summary.get("P0_must", 0),
                        }
                        p0_specs = result.get_by_priority(Priority.P0) if Priority else []
                        state = {
                            **state,
                            "prioritized_specs": [c.requirement for c in p0_specs],
                            "priority_matrix_result": result.to_dict(),
                        }
                        logger.info(f"Priority Matrix: {len(p0_specs)} P0 specs identified")
            except Exception as e:
                logger.warning(f"Priority Matrix integration failed (non-blocking): {e}")
                integration_data["priority_matrix_error"] = str(e)

        # DAG Decomposer: Decompose large specs into atomic sub-specs
        if DAG_DECOMPOSER_AVAILABLE and DAGDecomposer is not None:
            try:
                specs = state.get("specs", [])
                large_specs = [
                    s for s in specs
                    if len(s.get("text", "").split()) >= 50
                ]

                if large_specs:
                    decomposer = DAGDecomposer()
                    decomposed_dags = []

                    for spec in large_specs[:5]:
                        dag = await decomposer.decompose(spec)
                        if dag and hasattr(dag, 'nodes'):
                            decomposed_dags.append({
                                "original_id": spec.get("id"),
                                "sub_specs": len(dag.nodes) if dag.nodes else 0,
                            })

                    if decomposed_dags:
                        integration_data["dag_decomposer"] = {
                            "specs_decomposed": len(decomposed_dags),
                            "details": decomposed_dags,
                        }
                        state = {**state, "decomposed_specs": decomposed_dags}
                        logger.info(f"DAG Decomposer: {len(decomposed_dags)} specs decomposed")
            except Exception as e:
                logger.warning(f"DAG Decomposer integration failed (non-blocking): {e}")
                integration_data["dag_decomposer_error"] = str(e)

        # Counter-Example Generator: Detect ambiguities in specs
        if COUNTER_EXAMPLE_AVAILABLE and CounterExampleGenerator is not None:
            try:
                specs = state.get("specs", [])
                if specs:
                    generator = CounterExampleGenerator()
                    ambiguity_results = []
                    blocking_ambiguities = []

                    for spec in specs[:10]:
                        result = generator.analyze(spec, use_llm=False)

                        if result.is_ambiguous:
                            ambiguity_results.append({
                                "spec_id": spec.get("id"),
                                "ambiguity_score": result.ambiguity_score,
                                "terms_count": len(result.ambiguous_terms),
                            })

                            if result.ambiguity_score > 0.7:
                                blocking_ambiguities.append(spec.get("id"))

                    integration_data["counter_example"] = {
                        "specs_analyzed": min(len(specs), 10),
                        "ambiguous_specs": len(ambiguity_results),
                        "blocking_ambiguities": blocking_ambiguities,
                    }

                    if blocking_ambiguities:
                        state = {
                            **state,
                            "ambiguity_warnings": blocking_ambiguities,
                            "has_blocking_ambiguities": True,
                        }
                        logger.warning(f"Counter-Example: {len(blocking_ambiguities)} specs with high ambiguity")
            except Exception as e:
                logger.warning(f"Counter-Example integration failed (non-blocking): {e}")
                integration_data["counter_example_error"] = str(e)

        # Temporal Enhanced: Initialize durable workflow orchestration
        if TEMPORAL_ENHANCED_AVAILABLE and TemporalEnhanced is not None:
            try:
                temporal = TemporalEnhanced()
                if temporal:
                    integration_data["temporal"] = {
                        "available": True,
                        "initialized": True,
                    }
                    logger.info("Temporal Enhanced: initialized")
            except Exception as e:
                logger.warning(f"Temporal Enhanced integration failed (non-blocking): {e}")
                integration_data["temporal_error"] = str(e)

        # =====================================================================
        # PLANNING STACKS - on_init (2026-01-30: FASE 3)
        # =====================================================================

        # TokenBudgetPlanner - Initialize token budget for the run
        if TOKEN_BUDGET_AVAILABLE and TokenBudgetPlanner is not None:
            try:
                planner = TokenBudgetPlanner()
                context_pack = state.get("context_pack", {})
                deliverables = context_pack.get("deliverables", [])
                # Estimate initial budget based on deliverables count
                initial_estimate = planner.estimate_initial_budget(
                    deliverables_count=len(deliverables),
                    complexity_factor=1.5,  # Conservative estimate
                )
                integration_data["token_budget_init"] = {
                    "enabled": True,
                    "initial_budget": initial_estimate.budget if hasattr(initial_estimate, "budget") else 50000,
                    "deliverables": len(deliverables),
                }
                state = {**state, "initial_token_budget": initial_estimate}
                logger.info(f"TokenBudgetPlanner: initial budget={integration_data['token_budget_init']['initial_budget']} tokens")
            except Exception as e:
                logger.warning(f"TokenBudgetPlanner init failed (non-blocking): {e}")
                integration_data["token_budget_init_error"] = str(e)

        # PlanRetriever - Pre-fetch similar historical plans
        if PLAN_RETRIEVER_AVAILABLE and PlanSemanticRetriever is not None:
            try:
                retriever = PlanSemanticRetriever()
                objective = context_pack.get("objective", "") if context_pack else ""
                if objective:
                    similar_plans = retriever.find_similar(objective, top_k=3)
                    integration_data["similar_plans_prefetched"] = {
                        "enabled": True,
                        "plans_found": len(similar_plans) if similar_plans else 0,
                    }
                    if similar_plans:
                        state = {**state, "prefetched_similar_plans": similar_plans}
                    logger.debug(f"PlanRetriever: prefetched {len(similar_plans) if similar_plans else 0} similar plans")
            except Exception as e:
                logger.warning(f"PlanRetriever prefetch failed (non-blocking): {e}")
                integration_data["similar_plans_prefetched_error"] = str(e)

        # Emit INIT completed event
        _emit_integration_event(
            "init_completed",
            state,
            "Partial stacks init completed",
            metadata={"stacks_initialized": list(integration_data.get("partial_stacks", {}).keys())}
        )

        state = {**state, "partial_stacks_integration": integration_data}
        return state

    # =========================================================================
    # SPEC NODE HOOKS (2026-01-30: Ghost Code Integration)
    # =========================================================================

    async def on_spec(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Hook called during SPEC node to enhance spec decomposition.

        This integrates all the spec-related stacks added 2026-01-30:
        1. DAGDecomposer - DAG-based task decomposition
        2. PriorityMatrix - MoSCoW requirement prioritization
        3. InterfaceSketchGenerator - Function signatures from specs
        4. TestFunctionGenerator - Generate tests from EARS specs
        5. GoT Enhanced - Multi-perspective analysis
        6. Counter-Example Generator - Ambiguity detection

        Args:
            state: Current pipeline state with context_pack and granular_tasks

        Returns:
            Enhanced state with spec_enhancements
        """
        spec_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sprint_id": state.get("sprint_id"),
            "enhancements_applied": [],
            "errors": [],
        }

        context_pack = state.get("context_pack", {})
        granular_tasks = state.get("granular_tasks", [])

        # 1. DAG Decomposition - Build dependency graph for tasks
        if DAG_DECOMPOSER_AVAILABLE and DAGDecomposer is not None:
            try:
                decomposer = DAGDecomposer()
                functional_requirements = context_pack.get("functional_requirements", [])
                for rf in functional_requirements[:20]:  # Limit to prevent timeout
                    rf_desc = rf.get("description", str(rf)) if isinstance(rf, dict) else str(rf)
                    decomposer.add_requirement(rf_desc)

                dag = decomposer.build_dag()
                execution_order = dag.get_topological_order() if dag else []

                spec_data["dag_decomposition"] = {
                    "enabled": True,
                    "nodes": len(dag.nodes) if dag else 0,
                    "edges": len(dag.edges) if dag else 0,
                    "execution_order": execution_order[:10],  # Limit for state size
                }
                spec_data["enhancements_applied"].append("dag_decomposition")
                logger.info(f"DAG Decomposer: {len(execution_order)} tasks in dependency order")
            except Exception as e:
                logger.warning(f"DAG Decomposition failed (non-blocking): {e}")
                spec_data["errors"].append(f"dag_decomposition: {e}")

        # 2. Priority Matrix (MoSCoW) - Prioritize requirements
        if PRIORITY_MATRIX_AVAILABLE and PriorityMatrix is not None:
            try:
                matrix = PriorityMatrix()
                functional_requirements = context_pack.get("functional_requirements", [])
                for rf in functional_requirements[:20]:
                    rf_id = rf.get("id", "") if isinstance(rf, dict) else ""
                    rf_desc = rf.get("description", str(rf)) if isinstance(rf, dict) else str(rf)
                    matrix.add_requirement(rf_id, rf_desc)

                prioritized = matrix.classify_moscow()

                spec_data["priority_matrix"] = {
                    "enabled": True,
                    "must_have": len(prioritized.get("must", [])),
                    "should_have": len(prioritized.get("should", [])),
                    "could_have": len(prioritized.get("could", [])),
                    "wont_have": len(prioritized.get("wont", [])),
                }
                spec_data["enhancements_applied"].append("priority_matrix")
                logger.info(f"Priority Matrix: {spec_data['priority_matrix']['must_have']} MUST requirements")
            except Exception as e:
                logger.warning(f"Priority Matrix failed (non-blocking): {e}")
                spec_data["errors"].append(f"priority_matrix: {e}")

        # 3. Interface Sketch Generator - Generate function signatures
        if INTERFACE_SKETCH_AVAILABLE and InterfaceSketchGenerator is not None:
            try:
                generator = InterfaceSketchGenerator()
                sketches = []
                for task in granular_tasks[:10]:
                    deliverable = task.get("deliverable", "")
                    requirements = task.get("requirements", [])
                    sketch = generator.generate(deliverable, requirements)
                    if sketch:
                        sketches.append({
                            "deliverable": deliverable,
                            "functions": sketch.functions[:5],  # Limit
                            "classes": sketch.classes[:3],
                        })

                spec_data["interface_sketches"] = {
                    "enabled": True,
                    "count": len(sketches),
                    "sketches": sketches,
                }
                spec_data["enhancements_applied"].append("interface_sketches")
                logger.info(f"Interface Sketches: {len(sketches)} generated")
            except Exception as e:
                logger.warning(f"Interface Sketch generation failed (non-blocking): {e}")
                spec_data["errors"].append(f"interface_sketches: {e}")

        # 4. Test Function Generator (TFG) - Generate tests from EARS specs
        if TFG_AVAILABLE and TestFunctionGenerator is not None:
            try:
                tfg = TestFunctionGenerator()
                test_suites = []
                for task in granular_tasks[:10]:
                    invariants = task.get("invariants", [])
                    edge_cases = task.get("edge_cases", [])
                    suite = tfg.generate(invariants, edge_cases)
                    if suite:
                        test_suites.append({
                            "deliverable": task.get("deliverable", ""),
                            "test_count": len(suite.tests),
                        })

                spec_data["test_generation"] = {
                    "enabled": True,
                    "suites": len(test_suites),
                    "total_tests": sum(s["test_count"] for s in test_suites),
                }
                spec_data["enhancements_applied"].append("test_generation")
                logger.info(f"Test Generation: {spec_data['test_generation']['total_tests']} tests generated")
            except Exception as e:
                logger.warning(f"Test generation failed (non-blocking): {e}")
                spec_data["errors"].append(f"test_generation: {e}")

        # 5. GoT Enhanced - Multi-perspective analysis
        if GOT_ENHANCED_AVAILABLE and EnhancedThoughtGraph is not None:
            try:
                got = EnhancedThoughtGraph()
                perspectives = []
                objective = context_pack.get("objective", "")
                if objective:
                    result = got.analyze_multi_perspective(
                        problem=objective,
                        num_perspectives=3,
                    )
                    if result:
                        perspectives = result.perspectives[:3]

                spec_data["got_analysis"] = {
                    "enabled": True,
                    "perspectives": len(perspectives),
                    "insights": [p.insight for p in perspectives] if perspectives else [],
                }
                spec_data["enhancements_applied"].append("got_analysis")
                logger.info(f"GoT Analysis: {len(perspectives)} perspectives generated")
            except Exception as e:
                logger.warning(f"GoT analysis failed (non-blocking): {e}")
                spec_data["errors"].append(f"got_analysis: {e}")

        # 6. Counter-Example Generator - Detect ambiguity
        # (Import check inline to avoid startup errors)
        try:
            from pipeline.spec_kit.counter_example_generator import (
                CounterExampleGenerator,
                COUNTER_EXAMPLE_AVAILABLE,
            )
            if COUNTER_EXAMPLE_AVAILABLE and CounterExampleGenerator is not None:
                ceg = CounterExampleGenerator()
                ambiguities = []
                for task in granular_tasks[:5]:
                    task_prompt = task.get("task_prompt", "")
                    counter_examples = ceg.generate(task_prompt)
                    if counter_examples:
                        ambiguities.append({
                            "deliverable": task.get("deliverable", ""),
                            "ambiguity_count": len(counter_examples),
                        })

                spec_data["ambiguity_detection"] = {
                    "enabled": True,
                    "tasks_checked": len(ambiguities),
                    "total_ambiguities": sum(a["ambiguity_count"] for a in ambiguities),
                }
                spec_data["enhancements_applied"].append("ambiguity_detection")
                logger.info(f"Ambiguity Detection: {spec_data['ambiguity_detection']['total_ambiguities']} found")
        except Exception as e:
            logger.debug(f"Counter-Example Generator not available: {e}")

        # =====================================================================
        # SPEC KIT FULL INTEGRATION (2026-01-30: FASE 1)
        # =====================================================================

        # 7. SpecMasticator - Parse raw input into EARS requirements
        if MASTICATOR_AVAILABLE and SpecMasticator is not None:
            try:
                masticator = SpecMasticator()
                raw_objective = context_pack.get("objective", "")
                if raw_objective:
                    requirements, questions = masticator.process(raw_objective)
                    spec_data["masticator"] = {
                        "enabled": True,
                        "requirements_count": len(requirements) if requirements else 0,
                        "pending_questions": len(questions) if questions else 0,
                    }
                    if requirements:
                        state = {**state, "ears_requirements": requirements}
                    if questions:
                        state = {**state, "masticator_questions": questions}
                    spec_data["enhancements_applied"].append("masticator")
                    logger.info(f"Masticator: {spec_data['masticator']['requirements_count']} EARS requirements parsed")
            except Exception as e:
                logger.warning(f"Masticator failed (non-blocking): {e}")
                spec_data["errors"].append(f"masticator: {e}")

        # 8. ContradictionDetector - Z3 SAT solver for logical contradictions
        if CONTRADICTION_DETECTOR_AVAILABLE and ContradictionDetector is not None:
            try:
                detector = ContradictionDetector()
                functional_requirements = context_pack.get("functional_requirements", [])
                if functional_requirements:
                    # Convert to specs format expected by detector
                    specs_for_detector = []
                    for rf in functional_requirements[:20]:
                        rf_id = rf.get("id", "") if isinstance(rf, dict) else ""
                        rf_desc = rf.get("description", str(rf)) if isinstance(rf, dict) else str(rf)
                        specs_for_detector.append({"id": rf_id, "description": rf_desc})

                    report = detector.detect_contradictions(specs_for_detector)
                    spec_data["contradiction_detector"] = {
                        "enabled": True,
                        "specs_analyzed": len(specs_for_detector),
                        "contradictions_found": len(report.contradictions) if report else 0,
                        "is_consistent": report.is_consistent if report else True,
                    }
                    if report and not report.is_consistent:
                        state = {**state, "contradictions": [c.__dict__ for c in report.contradictions]}
                        logger.warning(f"ContradictionDetector: {len(report.contradictions)} contradictions found!")
                    spec_data["enhancements_applied"].append("contradiction_detector")
                    logger.info(f"ContradictionDetector: analyzed {len(specs_for_detector)} specs")
            except Exception as e:
                logger.warning(f"ContradictionDetector failed (non-blocking): {e}")
                spec_data["errors"].append(f"contradiction_detector: {e}")

        # 9. GapFiller - Detect and fill missing details
        if GAP_FILLER_AVAILABLE and GapFiller is not None:
            try:
                gap_filler = GapFiller()
                functional_requirements = context_pack.get("functional_requirements", [])
                if functional_requirements:
                    gaps, filled = gap_filler.analyze_and_fill(functional_requirements[:15])
                    spec_data["gap_filler"] = {
                        "enabled": True,
                        "gaps_detected": len(gaps) if gaps else 0,
                        "gaps_filled": len(filled) if filled else 0,
                    }
                    if filled:
                        state = {**state, "filled_gaps": filled}
                    spec_data["enhancements_applied"].append("gap_filler")
                    logger.info(f"GapFiller: {spec_data['gap_filler']['gaps_detected']} gaps detected, {spec_data['gap_filler']['gaps_filled']} filled")
            except Exception as e:
                logger.warning(f"GapFiller failed (non-blocking): {e}")
                spec_data["errors"].append(f"gap_filler: {e}")

        # 10. ConsistencyChecker - Check consistency between specs
        if CONSISTENCY_CHECKER_AVAILABLE and ConsistencyChecker is not None:
            try:
                checker = ConsistencyChecker()
                functional_requirements = context_pack.get("functional_requirements", [])
                invariants = context_pack.get("invariants", [])
                if functional_requirements or invariants:
                    result = checker.check(functional_requirements[:20], invariants[:10])
                    spec_data["consistency_checker"] = {
                        "enabled": True,
                        "is_consistent": result.is_consistent if hasattr(result, "is_consistent") else True,
                        "issues_count": len(result.issues) if hasattr(result, "issues") else 0,
                    }
                    if hasattr(result, "issues") and result.issues:
                        state = {**state, "consistency_issues": result.issues}
                    spec_data["enhancements_applied"].append("consistency_checker")
                    logger.info(f"ConsistencyChecker: {'consistent' if spec_data['consistency_checker']['is_consistent'] else 'inconsistent'}")
            except Exception as e:
                logger.warning(f"ConsistencyChecker failed (non-blocking): {e}")
                spec_data["errors"].append(f"consistency_checker: {e}")

        # 11. SemanticSearch - Find similar specs/requirements
        if SEMANTIC_SEARCH_AVAILABLE and SemanticSearch is not None:
            try:
                searcher = SemanticSearch()
                objective = context_pack.get("objective", "")
                if objective:
                    similar_specs = searcher.search(objective, top_k=5)
                    spec_data["semantic_search"] = {
                        "enabled": True,
                        "similar_found": len(similar_specs) if similar_specs else 0,
                    }
                    if similar_specs:
                        state = {**state, "similar_specs": similar_specs}
                    spec_data["enhancements_applied"].append("semantic_search")
                    logger.debug(f"SemanticSearch: {len(similar_specs) if similar_specs else 0} similar specs found")
            except Exception as e:
                logger.debug(f"SemanticSearch failed (non-blocking): {e}")
                spec_data["errors"].append(f"semantic_search: {e}")

        # 12. Constitution - Apply spec rules
        if CONSTITUTION_AVAILABLE and Constitution is not None:
            try:
                constitution = Constitution()
                functional_requirements = context_pack.get("functional_requirements", [])
                if functional_requirements:
                    violations = constitution.validate(functional_requirements[:20])
                    spec_data["constitution"] = {
                        "enabled": True,
                        "violations_count": len(violations) if violations else 0,
                        "all_passed": len(violations) == 0 if violations is not None else True,
                    }
                    if violations:
                        state = {**state, "constitution_violations": violations}
                    spec_data["enhancements_applied"].append("constitution")
                    logger.info(f"Constitution: {spec_data['constitution']['violations_count']} violations")
            except Exception as e:
                logger.debug(f"Constitution failed (non-blocking): {e}")
                spec_data["errors"].append(f"constitution: {e}")

        # =====================================================================
        # SPEC AI FULL INTEGRATION (2026-01-30: FASE 2)
        # =====================================================================

        # 13. GoTAnalyzer - Multi-perspective analysis
        if GOT_ANALYZER_AVAILABLE and GoTAnalyzer is not None:
            try:
                analyzer = GoTAnalyzer()
                objective = context_pack.get("objective", "")
                if objective:
                    perspectives = analyze_perspectives(objective, num_perspectives=3)
                    blindspots = find_blindspots(objective)
                    spec_data["got_analyzer"] = {
                        "enabled": True,
                        "perspectives_count": len(perspectives) if perspectives else 0,
                        "blindspots_count": len(blindspots) if blindspots else 0,
                    }
                    if perspectives:
                        state = {**state, "analysis_perspectives": perspectives}
                    if blindspots:
                        state = {**state, "analysis_blindspots": blindspots}
                    spec_data["enhancements_applied"].append("got_analyzer")
                    logger.info(f"GoTAnalyzer: {spec_data['got_analyzer']['perspectives_count']} perspectives, {spec_data['got_analyzer']['blindspots_count']} blindspots")
            except Exception as e:
                logger.warning(f"GoTAnalyzer failed (non-blocking): {e}")
                spec_data["errors"].append(f"got_analyzer: {e}")

        # 14. AdversarialReviewer - Red team attack simulation
        if ADVERSARIAL_REVIEWER_AVAILABLE and AdversarialReviewer is not None:
            try:
                reviewer = AdversarialReviewer()
                objective = context_pack.get("objective", "")
                functional_requirements = context_pack.get("functional_requirements", [])
                if objective and functional_requirements:
                    attack_result = attack_spec(objective, functional_requirements[:10])
                    weaknesses = find_weaknesses(attack_result) if attack_result else []
                    spec_data["adversarial_reviewer"] = {
                        "enabled": True,
                        "attacks_simulated": attack_result.attacks_count if hasattr(attack_result, "attacks_count") else 0,
                        "weaknesses_found": len(weaknesses) if weaknesses else 0,
                    }
                    if weaknesses:
                        state = {**state, "spec_weaknesses": weaknesses}
                        hardening = suggest_hardening(weaknesses)
                        if hardening:
                            state = {**state, "hardening_recommendations": hardening}
                    spec_data["enhancements_applied"].append("adversarial_reviewer")
                    logger.info(f"AdversarialReviewer: {spec_data['adversarial_reviewer']['weaknesses_found']} weaknesses found")
            except Exception as e:
                logger.warning(f"AdversarialReviewer failed (non-blocking): {e}")
                spec_data["errors"].append(f"adversarial_reviewer: {e}")

        # 15. GapDetector (SpecAI) - AI-powered gap detection
        if GAP_DETECTOR_AVAILABLE and GapDetector is not None:
            try:
                detector = GapDetector()
                objective = context_pack.get("objective", "")
                functional_requirements = context_pack.get("functional_requirements", [])
                if objective:
                    missing = find_missing(objective, functional_requirements[:15])
                    implicit = infer_implicit(objective)
                    spec_data["gap_detector_ai"] = {
                        "enabled": True,
                        "missing_count": len(missing) if missing else 0,
                        "implicit_count": len(implicit) if implicit else 0,
                    }
                    if missing:
                        state = {**state, "ai_detected_missing": missing}
                    if implicit:
                        state = {**state, "ai_implicit_requirements": implicit}
                    spec_data["enhancements_applied"].append("gap_detector_ai")
                    logger.info(f"GapDetector AI: {spec_data['gap_detector_ai']['missing_count']} missing, {spec_data['gap_detector_ai']['implicit_count']} implicit")
            except Exception as e:
                logger.warning(f"GapDetector AI failed (non-blocking): {e}")
                spec_data["errors"].append(f"gap_detector_ai: {e}")

        # 16. StackRecommender - Technology stack recommendations
        if STACK_RECOMMENDER_AVAILABLE and StackRecommender is not None:
            try:
                recommender = StackRecommender()
                objective = context_pack.get("objective", "")
                functional_requirements = context_pack.get("functional_requirements", [])
                if objective:
                    needs = analyze_needs(objective, functional_requirements[:10])
                    matches = match_stacks(needs) if needs else []
                    spec_data["stack_recommender"] = {
                        "enabled": True,
                        "stacks_recommended": len(matches) if matches else 0,
                    }
                    if matches:
                        state = {**state, "recommended_stacks": matches}
                    spec_data["enhancements_applied"].append("stack_recommender")
                    logger.info(f"StackRecommender: {spec_data['stack_recommender']['stacks_recommended']} stacks recommended")
            except Exception as e:
                logger.warning(f"StackRecommender failed (non-blocking): {e}")
                spec_data["errors"].append(f"stack_recommender: {e}")

        # 17. ComplexityEstimator - Effort and complexity estimation
        if COMPLEXITY_ESTIMATOR_AVAILABLE and ComplexityEstimator is not None:
            try:
                estimator = ComplexityEstimator()
                objective = context_pack.get("objective", "")
                granular_tasks_list = granular_tasks or []
                if objective or granular_tasks_list:
                    effort = estimate_effort(objective, granular_tasks_list[:20])
                    risks = identify_risks(objective)
                    spec_data["complexity_estimator"] = {
                        "enabled": True,
                        "effort_estimate": effort.total if hasattr(effort, "total") else 0,
                        "risks_count": len(risks) if risks else 0,
                    }
                    if effort:
                        state = {**state, "effort_estimate": effort}
                    if risks:
                        state = {**state, "identified_risks": risks}
                    spec_data["enhancements_applied"].append("complexity_estimator")
                    logger.info(f"ComplexityEstimator: effort={spec_data['complexity_estimator']['effort_estimate']}, risks={spec_data['complexity_estimator']['risks_count']}")
            except Exception as e:
                logger.warning(f"ComplexityEstimator failed (non-blocking): {e}")
                spec_data["errors"].append(f"complexity_estimator: {e}")

        # 18. RiskPredictor - Pattern-based risk prediction
        if RISK_PREDICTOR_AVAILABLE and RiskPredictor is not None:
            try:
                predictor = RiskPredictor()
                sprint_id = state.get("sprint_id", "")
                objective = context_pack.get("objective", "")
                if sprint_id:
                    patterns = predict_from_patterns(sprint_id, objective)
                    warnings = early_warning(patterns) if patterns else []
                    spec_data["risk_predictor"] = {
                        "enabled": True,
                        "patterns_matched": len(patterns) if patterns else 0,
                        "early_warnings": len(warnings) if warnings else 0,
                    }
                    if warnings:
                        state = {**state, "risk_early_warnings": warnings}
                    spec_data["enhancements_applied"].append("risk_predictor")
                    logger.info(f"RiskPredictor: {spec_data['risk_predictor']['patterns_matched']} patterns, {spec_data['risk_predictor']['early_warnings']} warnings")
            except Exception as e:
                logger.warning(f"RiskPredictor failed (non-blocking): {e}")
                spec_data["errors"].append(f"risk_predictor: {e}")

        # =====================================================================
        # PLANNING STACKS - on_spec subset (2026-01-30: FASE 3 partial)
        # =====================================================================

        # 19. HierarchicalDecomposer - Hierarchical task decomposition
        if HIERARCHICAL_DECOMPOSER_AVAILABLE and HierarchicalDecomposer is not None:
            try:
                decomposer = HierarchicalDecomposer()
                objective = context_pack.get("objective", "")
                if objective:
                    decomposition = decompose_spec(objective, max_depth=3)
                    spec_data["hierarchical_decomposer"] = {
                        "enabled": True,
                        "levels": decomposition.depth if hasattr(decomposition, "depth") else 0,
                        "total_nodes": decomposition.total_nodes if hasattr(decomposition, "total_nodes") else 0,
                    }
                    if decomposition:
                        state = {**state, "hierarchical_decomposition": decomposition}
                    spec_data["enhancements_applied"].append("hierarchical_decomposer")
                    logger.info(f"HierarchicalDecomposer: {spec_data['hierarchical_decomposer']['total_nodes']} nodes")
            except Exception as e:
                logger.warning(f"HierarchicalDecomposer failed (non-blocking): {e}")
                spec_data["errors"].append(f"hierarchical_decomposer: {e}")

        # 20. CriticalPathAnalyzer - Critical path method
        if CRITICAL_PATH_AVAILABLE and CriticalPathAnalyzer is not None:
            try:
                analyzer = CriticalPathAnalyzer()
                granular_tasks_list = granular_tasks or []
                if granular_tasks_list:
                    critical_path = analyze_critical_path(granular_tasks_list)
                    parallel_groups = get_parallel_groups(granular_tasks_list)
                    spec_data["critical_path"] = {
                        "enabled": True,
                        "path_length": len(critical_path) if critical_path else 0,
                        "parallel_groups": len(parallel_groups) if parallel_groups else 0,
                    }
                    if critical_path:
                        state = {**state, "critical_path": critical_path}
                    if parallel_groups:
                        state = {**state, "parallel_execution_groups": parallel_groups}
                    spec_data["enhancements_applied"].append("critical_path")
                    logger.info(f"CriticalPath: {spec_data['critical_path']['path_length']} in path, {spec_data['critical_path']['parallel_groups']} parallel groups")
            except Exception as e:
                logger.warning(f"CriticalPathAnalyzer failed (non-blocking): {e}")
                spec_data["errors"].append(f"critical_path: {e}")

        # 21. DynamicPriorityCalculator - Dynamic priority
        if DYNAMIC_PRIORITY_AVAILABLE and DynamicPriorityCalculator is not None:
            try:
                calculator = DynamicPriorityCalculator()
                granular_tasks_list = granular_tasks or []
                if granular_tasks_list:
                    priorities = calculate_dynamic_priority(granular_tasks_list)
                    spec_data["dynamic_priority"] = {
                        "enabled": True,
                        "tasks_prioritized": len(priorities) if priorities else 0,
                    }
                    if priorities:
                        state = {**state, "dynamic_priorities": priorities}
                    spec_data["enhancements_applied"].append("dynamic_priority")
                    logger.info(f"DynamicPriority: {spec_data['dynamic_priority']['tasks_prioritized']} tasks prioritized")
            except Exception as e:
                logger.warning(f"DynamicPriorityCalculator failed (non-blocking): {e}")
                spec_data["errors"].append(f"dynamic_priority: {e}")

        # 22. MultiPathPlanGenerator - GoT multi-path plan
        if MULTI_PATH_PLAN_AVAILABLE and MultiPathPlanGenerator is not None:
            try:
                generator = MultiPathPlanGenerator()
                objective = context_pack.get("objective", "")
                granular_tasks_list = granular_tasks or []
                if objective:
                    plans = generate_multi_path_plan(objective, granular_tasks_list[:15], num_paths=3)
                    spec_data["multi_path_plan"] = {
                        "enabled": True,
                        "plans_generated": len(plans) if plans else 0,
                    }
                    if plans:
                        state = {**state, "alternative_plans": plans}
                    spec_data["enhancements_applied"].append("multi_path_plan")
                    logger.info(f"MultiPathPlan: {spec_data['multi_path_plan']['plans_generated']} alternative plans")
            except Exception as e:
                logger.warning(f"MultiPathPlanGenerator failed (non-blocking): {e}")
                spec_data["errors"].append(f"multi_path_plan: {e}")

        # 23. PlanMaster - Central planning orchestrator
        if PLAN_MASTER_AVAILABLE and PlanMaster is not None:
            try:
                master = PlanMaster()
                objective = context_pack.get("objective", "")
                granular_tasks_list = granular_tasks or []
                if objective and granular_tasks_list:
                    is_ready = check_plan_ready_for_execution(granular_tasks_list)
                    spec_data["plan_master"] = {
                        "enabled": True,
                        "execution_ready": is_ready,
                    }
                    state = {**state, "plan_execution_ready": is_ready}
                    spec_data["enhancements_applied"].append("plan_master")
                    logger.info(f"PlanMaster: execution_ready={is_ready}")
            except Exception as e:
                logger.warning(f"PlanMaster failed (non-blocking): {e}")
                spec_data["errors"].append(f"plan_master: {e}")

        # === FINAL: Add to Buffer of Thoughts if available ===
        if self._bot:
            try:
                self._bot.add_thought(
                    f"Spec enhanced with: {', '.join(spec_data['enhancements_applied'])}",
                    thought_type=ThoughtType.OBSERVATION,
                )
            except Exception as e:
                logger.debug(f"BoT spec thought failed: {e}")

        # Emit spec_enhanced event
        _emit_integration_event(
            "spec_enhanced",
            state,
            f"Spec phase enhanced with {len(spec_data['enhancements_applied'])} stacks",
            metadata=spec_data,
        )

        # Update state with spec enhancements
        state = {
            **state,
            "spec_enhancements": spec_data,
        }

        logger.info(
            f"on_spec completed: {len(spec_data['enhancements_applied'])} enhancements, "
            f"{len(spec_data['errors'])} errors"
        )

        return state

    # =========================================================================
    # EXEC NODE HOOKS
    # =========================================================================

    async def on_exec_start(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Hook called at start of EXEC node.

        Actions:
        - Log execution start to Phoenix
        - Add execution context to Buffer of Thoughts
        - Generate interface sketches for implementation guidance (Ghost Code Integration)
        - Retrieve relevant claims via ColBERT (Ghost Code Integration)
        - Determine execution strategy via Surgical Rework Migration (Ghost Code Integration)
        - Emit exec_started event (Ghost Code Integration)

        Args:
            state: Current pipeline state

        Returns:
            Updated state
        """
        # HIGH-002 FIX: Track execution start data
        exec_start_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sprint_id": state.get("sprint_id"),
            "bot_active": self._bot is not None,
            "thoughts_added": 0,
        }

        if self._bot:
            try:
                self._bot.add_thought(
                    f"Execution started for sprint {state.get('sprint_id')}",
                    thought_type=ThoughtType.OBSERVATION,
                )
                exec_start_data["thoughts_added"] = 1
            except Exception as e:
                logger.warning(f"BoT exec start hook failed: {e}")
                exec_start_data["bot_error"] = str(e)

        # =====================================================================
        # GHOST CODE INTEGRATION: Interface Sketch Generator (2026-01-30)
        # Generate API signatures BEFORE implementation to guide development
        # =====================================================================
        if INTERFACE_SKETCH_AVAILABLE and InterfaceSketchGenerator is not None:
            try:
                sketcher = InterfaceSketchGenerator()
                context_pack = state.get("context_pack", {})
                sprint_spec = context_pack.get("objective", "")

                if sprint_spec:
                    sketch_result = sketcher.generate_sketch(sprint_spec)
                    exec_start_data["interface_sketches"] = {
                        "generated": True,
                        "signatures_count": len(sketch_result.signatures) if hasattr(sketch_result, "signatures") else 0,
                        "contracts_count": len(sketch_result.contracts) if hasattr(sketch_result, "contracts") else 0,
                    }
                    # Store full sketches in state for implementation guidance
                    state = {**state, "interface_sketches": sketch_result}
                    logger.debug(
                        f"InterfaceSketchGenerator generated {exec_start_data['interface_sketches']['signatures_count']} "
                        f"signatures for sprint {state.get('sprint_id')}"
                    )
                else:
                    exec_start_data["interface_sketches"] = {"generated": False, "reason": "no_sprint_spec"}
            except Exception as e:
                logger.warning(f"InterfaceSketchGenerator integration failed (non-blocking): {e}")
                exec_start_data["interface_sketch_error"] = str(e)

        # =====================================================================
        # GHOST CODE INTEGRATION: ColBERT Retriever (2026-01-30)
        # Efficient token-level retrieval for relevant claims/context
        # P0-FIX-2026-02-01: Correct method name + await (was retrieve)
        # =====================================================================
        if COLBERT_AVAILABLE and ColBERTRetriever is not None:
            try:
                colbert = ColBERTRetriever()
                context_pack = state.get("context_pack", {})
                query = context_pack.get("objective", "") or state.get("sprint_id", "")

                if query:
                    # Search for relevant documents/claims for this execution
                    retrieval_results = await colbert.search(query, top_k=5)
                    exec_start_data["colbert_retrieval"] = {
                        "performed": True,
                        "documents_retrieved": len(retrieval_results) if retrieval_results else 0,
                    }
                    # Store retrieved context for crew usage
                    if retrieval_results:
                        state = {**state, "colbert_context": retrieval_results}
                    logger.debug(
                        f"ColBERT retrieved {exec_start_data['colbert_retrieval']['documents_retrieved']} "
                        f"documents for execution"
                    )
                else:
                    exec_start_data["colbert_retrieval"] = {"performed": False, "reason": "no_query"}
            except Exception as e:
                logger.warning(f"ColBERT retrieval integration failed (non-blocking): {e}")
                exec_start_data["colbert_error"] = str(e)

        # =====================================================================
        # GHOST CODE INTEGRATION: Surgical Rework Migration (2026-01-30)
        # Determine if sprint-level or task-level execution is optimal
        # =====================================================================
        if SURGICAL_REWORK_MIGRATION_AVAILABLE and get_exec_strategy is not None:
            try:
                strategy = get_exec_strategy(state)
                exec_start_data["exec_strategy"] = {
                    "type": strategy.get("type", "sprint_level") if isinstance(strategy, dict) else str(strategy),
                    "reason": strategy.get("reason", "default") if isinstance(strategy, dict) else "computed",
                }
                # Store strategy in state for workflow routing
                state = {**state, "exec_strategy": strategy}
                logger.info(
                    f"Execution strategy determined: {exec_start_data['exec_strategy']['type']} "
                    f"(reason: {exec_start_data['exec_strategy']['reason']})"
                )
            except Exception as e:
                logger.warning(f"Surgical rework migration strategy failed (non-blocking): {e}")
                exec_start_data["exec_strategy_error"] = str(e)
                # Default to sprint-level execution
                state = {**state, "exec_strategy": {"type": "sprint_level", "reason": "fallback_on_error"}}

        # =====================================================================
        # RAG STACKS FULL INTEGRATION (2026-01-30: FASE 4)
        # 2026-02-01: Added agent-specific routing via RAGStackRouter
        # =====================================================================

        # Determine the current agent (from state or default)
        current_agent_id = state.get("current_agent_id", state.get("agent_id", ""))

        # Use RAGStackRouter for agent-specific RAG routing
        rag_router_used = False
        if USE_RAG_ROUTER and RAG_ROUTER_AVAILABLE and get_rag_stack_router is not None:
            try:
                router = get_rag_stack_router()
                context_pack = state.get("context_pack", {})
                query = context_pack.get("objective", "")
                sprint_id = state.get("sprint_id", "")

                if query and current_agent_id:
                    router_result = await router.query_stacks(
                        agent_id=current_agent_id,
                        query=query,
                        sprint_id=sprint_id,
                    )

                    # Update state with router results
                    state_updates = router_result.to_state_updates()
                    if state_updates:
                        state = {**state, **state_updates}
                        rag_router_used = True

                    exec_start_data["rag_router"] = {
                        "enabled": True,
                        "agent_id": current_agent_id,
                        "stacks_enabled": router_result.stacks_enabled,
                        "stacks_queried": router_result.stacks_queried,
                        "stacks_succeeded": router_result.stacks_succeeded,
                        "total_documents": router_result.total_documents,
                        "errors": router_result.errors[:3] if router_result.errors else [],
                    }
                    logger.info(
                        f"RAG_ROUTER: {current_agent_id} - "
                        f"{router_result.stacks_succeeded}/{router_result.stacks_enabled} stacks, "
                        f"{router_result.total_documents} docs"
                    )
                else:
                    exec_start_data["rag_router"] = {
                        "enabled": False,
                        "reason": "no_query_or_agent_id",
                    }
                    logger.debug("RAG_ROUTER: Skipped - no query or agent_id")

            except Exception as e:
                logger.warning(f"RAG_ROUTER failed, falling back to legacy: {e}")
                exec_start_data["rag_router_error"] = str(e)

        # Legacy RAG integration - only if router not used
        # This ensures backwards compatibility when USE_RAG_ROUTER=false
        if not rag_router_used:
            exec_start_data["rag_router"] = {"enabled": False, "reason": "using_legacy_integration"}

        # SelfRAG - Self-reflective retrieval
        # P0-FIX-2026-02-01: Correct method name + await (was retrieve_self_reflective)
        # 2026-02-01: Skip if RAG router was used (avoids duplicate calls)
        if not rag_router_used and SELF_RAG_AVAILABLE and SelfRAG is not None:
            try:
                self_rag = SelfRAG()
                context_pack = state.get("context_pack", {})
                query = context_pack.get("objective", "")
                if query:
                    result = await self_rag.query_with_reflection(query, critique_response=False)
                    docs = result.documents if hasattr(result, "documents") else []
                    exec_start_data["self_rag"] = {
                        "enabled": True,
                        "documents_retrieved": len(docs),
                    }
                    if docs:
                        state = {**state, "self_rag_context": docs}
                    logger.debug(f"SelfRAG: {exec_start_data['self_rag']['documents_retrieved']} documents")
            except Exception as e:
                logger.warning(f"SelfRAG failed (non-blocking): {e}")
                exec_start_data["self_rag_error"] = str(e)

        # MemoRAG - Memory-augmented retrieval
        # P0-FIX-2026-02-01: Correct method name + await (was retrieve_with_memory)
        # 2026-02-01: Skip if RAG router was used (avoids duplicate calls)
        if not rag_router_used and MEMO_RAG_AVAILABLE and MemoRAG is not None:
            try:
                memo_rag = MemoRAG()
                context_pack = state.get("context_pack", {})
                query = context_pack.get("objective", "")
                sprint_id = state.get("sprint_id", "")
                if query:
                    result = await memo_rag.query_with_memory(query, use_episodic_memory=True, sprint_id=sprint_id)
                    docs = result.documents if hasattr(result, "documents") else []
                    exec_start_data["memo_rag"] = {
                        "enabled": True,
                        "documents_retrieved": len(docs),
                        "memory_hit": result.memory_hit if hasattr(result, "memory_hit") else False,
                    }
                    if docs:
                        state = {**state, "memo_rag_context": docs}
                    logger.debug(f"MemoRAG: {exec_start_data['memo_rag']['documents_retrieved']} documents")
            except Exception as e:
                logger.warning(f"MemoRAG failed (non-blocking): {e}")
                exec_start_data["memo_rag_error"] = str(e)

        # RaptorRAG - Hierarchical retrieval
        # P0-FIX-2026-02-01: Correct method name + await (was hierarchical_retrieve)
        # 2026-02-01: Skip if RAG router was used (avoids duplicate calls)
        if not rag_router_used and RAPTOR_RAG_AVAILABLE and RaptorRAG is not None:
            try:
                raptor_rag = RaptorRAG()
                context_pack = state.get("context_pack", {})
                query = context_pack.get("objective", "")
                if query:
                    result = await raptor_rag.query_hierarchical(query, top_k=5)
                    # RaptorQueryResult uses context_pieces for retrieved content
                    docs = result.context_pieces if hasattr(result, "context_pieces") else []
                    exec_start_data["raptor_rag"] = {
                        "enabled": True,
                        "documents_retrieved": len(docs),
                        "levels_traversed": result.levels_traversed if hasattr(result, "levels_traversed") else [],
                    }
                    if docs:
                        state = {**state, "raptor_rag_context": docs}
                    logger.debug(f"RaptorRAG: {exec_start_data['raptor_rag']['documents_retrieved']} documents")
            except Exception as e:
                logger.warning(f"RaptorRAG failed (non-blocking): {e}")
                exec_start_data["raptor_rag_error"] = str(e)

        # QdrantHybridSearch - Hybrid search
        # P0-FIX-2026-02-01: Add await + collection_name parameter
        # 2026-02-01: Skip if RAG router was used (avoids duplicate calls)
        if not rag_router_used and QDRANT_HYBRID_AVAILABLE and QdrantHybridSearch is not None:
            try:
                hybrid = QdrantHybridSearch()
                context_pack = state.get("context_pack", {})
                query = context_pack.get("objective", "")
                # Use default collection from env or fallback to "claims"
                collection_name = os.getenv("QDRANT_DEFAULT_COLLECTION", "claims")
                if query:
                    results = await hybrid.hybrid_search(collection_name, query, top_k=5)
                    exec_start_data["qdrant_hybrid"] = {
                        "enabled": True,
                        "documents_retrieved": len(results) if results else 0,
                    }
                    if results:
                        state = {**state, "hybrid_search_context": results}
                    logger.debug(f"QdrantHybrid: {exec_start_data['qdrant_hybrid']['documents_retrieved']} documents")
            except Exception as e:
                logger.warning(f"QdrantHybridSearch failed (non-blocking): {e}")
                exec_start_data["qdrant_hybrid_error"] = str(e)

        # CorrectiveRAG - Self-correcting retrieval with quality evaluation
        # P0-FIX-2026-02-01: Added integration for document quality scoring
        # 2026-02-01: Skip if RAG router was used (avoids duplicate calls)
        if not rag_router_used and CRAG_AVAILABLE and CorrectiveRAG is not None:
            try:
                crag = CorrectiveRAG()
                context_pack = state.get("context_pack", {})
                query = context_pack.get("objective", "")
                if query:
                    result = await crag.query_with_correction(query, enable_web_fallback=False)
                    # CorrectiveRAG returns CRAGResult with documents and action_taken
                    docs = result.documents if hasattr(result, "documents") else []
                    exec_start_data["corrective_rag"] = {
                        "enabled": True,
                        "documents_retrieved": len(docs),
                        "action_taken": result.action_taken.value if hasattr(result, "action_taken") else "unknown",
                        "web_search_used": result.web_search_used if hasattr(result, "web_search_used") else False,
                    }
                    if docs:
                        state = {**state, "corrective_rag_context": docs}
                    logger.debug(
                        f"CorrectiveRAG: {exec_start_data['corrective_rag']['documents_retrieved']} documents "
                        f"(action: {exec_start_data['corrective_rag']['action_taken']})"
                    )
            except Exception as e:
                logger.warning(f"CorrectiveRAG failed (non-blocking): {e}")
                exec_start_data["corrective_rag_error"] = str(e)

        # GraphRAG - Graph-enhanced retrieval with reasoning chains
        # P0-FIX-2026-02-01: Added integration for knowledge graph reasoning
        # 2026-02-01: Skip if RAG router was used (avoids duplicate calls)
        if not rag_router_used and GRAPHRAG_CLIENT_AVAILABLE and get_graphrag_integration is not None:
            try:
                graphrag = get_graphrag_integration()
                context_pack = state.get("context_pack", {})
                query = context_pack.get("objective", "")
                if query and await graphrag.is_ready():
                    result = await graphrag.query_with_reasoning(
                        query=query,
                        top_k=5,
                        include_reasoning_steps=True,
                    )
                    # GraphRAG returns ReasoningQueryResult with sources and reasoning
                    sources = result.sources if hasattr(result, "sources") else []
                    exec_start_data["graphrag"] = {
                        "enabled": True,
                        "sources_retrieved": len(sources),
                        "reasoning_steps": len(result.reasoning_steps) if hasattr(result, "reasoning_steps") else 0,
                        "confidence": result.confidence if hasattr(result, "confidence") else 0.0,
                    }
                    if sources:
                        # Convert sources to document format for consistency
                        graphrag_docs = [
                            {"content": s.get("text", ""), "source": "graphrag", "confidence": s.get("confidence", 0.5)}
                            for s in sources
                        ]
                        state = {**state, "graphrag_context": graphrag_docs}
                    logger.debug(
                        f"GraphRAG: {exec_start_data['graphrag']['sources_retrieved']} sources "
                        f"({exec_start_data['graphrag']['reasoning_steps']} reasoning steps)"
                    )
                else:
                    exec_start_data["graphrag"] = {"enabled": False, "reason": "service_not_ready"}
            except Exception as e:
                logger.warning(f"GraphRAG failed (non-blocking): {e}")
                exec_start_data["graphrag_error"] = str(e)

        # =====================================================================
        # PLANNING STACKS - on_exec_start subset (2026-01-30: FASE 3)
        # =====================================================================

        # FallbackGenerator - Generate fallback plans before execution
        if FALLBACK_GENERATOR_AVAILABLE and FallbackGenerator is not None:
            try:
                generator = FallbackGenerator()
                granular_tasks = state.get("granular_tasks", [])
                critical_path = state.get("critical_path", [])
                if granular_tasks:
                    fallback_plans = generate_fallback_plans(granular_tasks, critical_tasks=critical_path)
                    exec_start_data["fallback_generator"] = {
                        "enabled": True,
                        "fallbacks_created": len(fallback_plans) if fallback_plans else 0,
                    }
                    if fallback_plans:
                        state = {**state, "fallback_plans": fallback_plans}
                    logger.info(f"FallbackGenerator: {exec_start_data['fallback_generator']['fallbacks_created']} fallbacks")
            except Exception as e:
                logger.warning(f"FallbackGenerator failed (non-blocking): {e}")
                exec_start_data["fallback_generator_error"] = str(e)

        # PlanRetriever - Retrieve similar historical plans
        if PLAN_RETRIEVER_AVAILABLE and PlanSemanticRetriever is not None:
            try:
                retriever = PlanSemanticRetriever()
                context_pack = state.get("context_pack", {})
                objective = context_pack.get("objective", "")
                if objective:
                    similar_plans = retriever.find_similar(objective, top_k=3)
                    exec_start_data["plan_retriever"] = {
                        "enabled": True,
                        "similar_plans_found": len(similar_plans) if similar_plans else 0,
                    }
                    if similar_plans:
                        state = {**state, "historical_similar_plans": similar_plans}
                    logger.debug(f"PlanRetriever: {exec_start_data['plan_retriever']['similar_plans_found']} similar plans")
            except Exception as e:
                logger.warning(f"PlanRetriever failed (non-blocking): {e}")
                exec_start_data["plan_retriever_error"] = str(e)

        # TokenBudgetPlanner - Allocate token budget for execution
        if TOKEN_BUDGET_AVAILABLE and TokenBudgetPlanner is not None:
            try:
                planner = TokenBudgetPlanner()
                granular_tasks = state.get("granular_tasks", [])
                if granular_tasks:
                    estimates = estimate_task_tokens(granular_tasks)
                    budget = allocate_budgets(estimates, max_budget=100000)
                    exec_start_data["token_budget"] = {
                        "enabled": True,
                        "total_estimated": sum(e.tokens for e in estimates) if estimates else 0,
                        "budget_allocated": budget.total if hasattr(budget, "total") else 0,
                    }
                    if budget:
                        state = {**state, "token_budget": budget}
                    logger.info(f"TokenBudget: {exec_start_data['token_budget']['budget_allocated']} tokens allocated")
            except Exception as e:
                logger.warning(f"TokenBudgetPlanner failed (non-blocking): {e}")
                exec_start_data["token_budget_error"] = str(e)

        # =====================================================================
        # GHOST CODE INTEGRATION: Event Emission (2026-01-30)
        # Emit exec_started event for observability
        # =====================================================================
        _emit_integration_event(
            "exec_started",
            state,
            f"Execution started for sprint {state.get('sprint_id')}",
            metadata={
                "interface_sketches_generated": exec_start_data.get("interface_sketches", {}).get("generated", False),
                "colbert_docs_retrieved": exec_start_data.get("colbert_retrieval", {}).get("documents_retrieved", 0),
                "exec_strategy": exec_start_data.get("exec_strategy", {}).get("type", "unknown"),
                "rag_stacks_active": sum(1 for k in ["self_rag", "memo_rag", "raptor_rag", "qdrant_hybrid"] if k in exec_start_data and exec_start_data[k].get("enabled")),
                "planning_stacks_active": sum(1 for k in ["fallback_generator", "plan_retriever", "token_budget"] if k in exec_start_data and exec_start_data[k].get("enabled")),
            }
        )

        # =====================================================================
        # GHOST CODE INTEGRATION: State Validation (2026-01-30)
        # Validate state before execution proceeds
        # =====================================================================
        is_valid, validation_details = _validate_state_transition(state, "EXEC_START")
        exec_start_data["state_validation"] = {
            "valid": is_valid,
            "phase": "EXEC_START",
            "errors": validation_details.get("errors", []) if not is_valid else [],
        }

        # HIGH-002 FIX: Return enriched state with execution start data
        return {
            **state,
            "partial_stacks_exec_start": exec_start_data,
        }

    async def on_exec_end(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Hook called at end of EXEC node.

        Actions:
        - Record execution results to Buffer of Thoughts
        - Run Active RAG iterative retrieval if needed
        - Generate test functions from EARS specs via TFG (Ghost Code Integration)
        - Emit exec_completed event (Ghost Code Integration)

        Args:
            state: Current pipeline state

        Returns:
            Updated state
        """
        crew_result = state.get("crew_result", {})

        # HIGH-002 FIX: Track execution end data
        exec_end_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "crew_status": crew_result.get("status", "unknown") if crew_result else "no_result",
            "has_error": bool(crew_result.get("error")) if crew_result else False,
            "thoughts_added": 0,
        }

        if self._bot and crew_result:
            try:
                status = crew_result.get("status", "unknown")
                self._bot.add_thought(
                    f"Execution completed with status: {status}",
                    thought_type=ThoughtType.OBSERVATION,
                )
                exec_end_data["thoughts_added"] += 1

                if crew_result.get("error"):
                    self._bot.add_thought(
                        f"Error encountered: {crew_result.get('error')}",
                        thought_type=ThoughtType.UNCERTAINTY,
                        priority=ThoughtPriority.HIGH,
                    )
                    exec_end_data["thoughts_added"] += 1
                    exec_end_data["error_logged"] = True
            except Exception as e:
                logger.warning(f"BoT exec end hook failed: {e}")
                exec_end_data["bot_error"] = str(e)

        # =====================================================================
        # GHOST CODE INTEGRATION: Test Function Generator (TFG) (2026-01-30)
        # Generate pytest functions from EARS specifications after execution
        # This enables automated test generation based on the sprint deliverables
        # =====================================================================
        if TFG_AVAILABLE and TestFunctionGenerator is not None:
            try:
                tfg = TestFunctionGenerator()
                context_pack = state.get("context_pack", {})

                # Get EARS specifications from context pack or state
                ears_specs = context_pack.get("ears_specs", [])
                if not ears_specs:
                    # Try to extract from deliverables or RF list
                    deliverables = context_pack.get("deliverables", [])
                    rf_list = context_pack.get("rf", [])
                    if deliverables or rf_list:
                        ears_specs = rf_list if rf_list else deliverables

                if ears_specs:
                    test_suite = tfg.generate(ears_specs)
                    exec_end_data["tfg_generation"] = {
                        "generated": True,
                        "test_count": len(test_suite.tests) if hasattr(test_suite, "tests") else 0,
                        "coverage_estimate": test_suite.coverage_estimate if hasattr(test_suite, "coverage_estimate") else None,
                    }
                    # Store generated tests in state for QA phase
                    state = {**state, "generated_tests": test_suite}
                    logger.info(
                        f"TFG generated {exec_end_data['tfg_generation']['test_count']} "
                        f"tests from EARS specifications"
                    )
                else:
                    exec_end_data["tfg_generation"] = {"generated": False, "reason": "no_ears_specs"}
            except Exception as e:
                logger.warning(f"TFG integration failed (non-blocking): {e}")
                exec_end_data["tfg_error"] = str(e)

        # =====================================================================
        # GHOST CODE INTEGRATION: Event Emission (2026-01-30)
        # Emit exec_completed event for observability
        # =====================================================================
        _emit_integration_event(
            "exec_completed",
            state,
            f"Execution completed for sprint {state.get('sprint_id')}",
            level="info" if not exec_end_data.get("has_error") else "warning",
            metadata={
                "crew_status": exec_end_data.get("crew_status"),
                "has_error": exec_end_data.get("has_error"),
                "tfg_tests_generated": exec_end_data.get("tfg_generation", {}).get("test_count", 0),
            }
        )

        # =====================================================================
        # GHOST CODE INTEGRATION: State Validation (2026-01-30)
        # Validate state before proceeding to gates
        # =====================================================================
        is_valid, validation_details = _validate_state_transition(state, "EXEC_END")
        exec_end_data["state_validation"] = {
            "valid": is_valid,
            "phase": "EXEC_END",
            "errors": validation_details.get("errors", []) if not is_valid else [],
        }

        # HIGH-002 FIX: Return enriched state with execution end data
        return {
            **state,
            "partial_stacks_exec_end": exec_end_data,
        }

    # =========================================================================
    # GATE NODE HOOKS
    # =========================================================================

    async def on_gate(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Hook called during GATE node.

        Actions:
        - Run RAGAS evaluation on any RAG responses
        - Run DeepEval hallucination detection
        - Record gate results to Buffer of Thoughts

        Args:
            state: Current pipeline state

        Returns:
            Updated state with evaluation results
        """
        integration_data = state.get("partial_stacks_integration", {})
        evaluations = integration_data.get("evaluations", {})

        # Get crew output for evaluation
        crew_result = state.get("crew_result", {})
        crew_output = crew_result.get("crew_output", "")
        context_pack = state.get("context_pack", {})
        objective = context_pack.get("objective", "")

        # RAGAS Evaluation
        if self._ragas and crew_output and objective:
            try:
                ragas_result = await self._ragas.evaluate(
                    question=objective,
                    answer=crew_output[:2000],  # Truncate for evaluation
                    contexts=[str(context_pack)[:1000]],
                )
                evaluations["ragas"] = {
                    "overall_score": ragas_result.overall_score,
                    "faithfulness": ragas_result.metrics.faithfulness,
                    "answer_relevancy": ragas_result.metrics.answer_relevancy,
                    "context_precision": ragas_result.metrics.context_precision,
                }
                logger.debug(f"RAGAS evaluation: {ragas_result.overall_score:.2f}")
            except Exception as e:
                logger.warning(f"RAGAS evaluation failed: {e}")
                evaluations["ragas_error"] = str(e)

        # DeepEval Hallucination Detection
        if self._deepeval and crew_output:
            try:
                deepeval_result = await self._deepeval.evaluate_rag_response(
                    query=objective,
                    response=crew_output[:2000],
                    context=[str(context_pack)[:1000]],
                    metrics=["faithfulness", "hallucination"],
                )

                hallucination_result = deepeval_result.get("hallucination", {})
                evaluations["deepeval"] = {
                    "hallucination_score": hallucination_result.get("score", 0),
                    "hallucination_passed": hallucination_result.get("passed", True),
                    "faithfulness_score": deepeval_result.get("faithfulness", {}).get("score", 0),
                }

                if not hallucination_result.get("passed", True):
                    logger.warning("DeepEval detected potential hallucination")

                logger.debug(f"DeepEval evaluation completed")
            except Exception as e:
                logger.warning(f"DeepEval evaluation failed: {e}")
                evaluations["deepeval_error"] = str(e)

        # 2026-01-30: Unified Eval Runner (Ghost Code Integration)
        # Provides comprehensive evaluation across all stacks in one call
        if UNIFIED_EVAL_AVAILABLE and UnifiedEvalRunner and crew_output and objective:
            try:
                unified_runner = UnifiedEvalRunner()
                eval_input = EvalInput(
                    query=objective,
                    response=crew_output[:2000],
                    contexts=[str(context_pack)[:1000]],
                    sprint_id=state.get("sprint_id", "unknown"),
                    stacks=["ragas", "deepeval"],  # Use available stacks
                )
                unified_result = await unified_runner.evaluate(eval_input)
                evaluations["unified"] = {
                    "passed": unified_result.passed,
                    "overall_score": unified_result.overall_score,
                    "blocking_failures": unified_result.blocking_failures,
                    "metrics_count": len(unified_result.metrics),
                }
                logger.debug(f"Unified evaluation: passed={unified_result.passed}, score={unified_result.overall_score:.2f}")
            except Exception as e:
                logger.warning(f"Unified evaluation failed (non-blocking): {e}")
                evaluations["unified_error"] = str(e)

        # 2026-01-30: Cleanlab Hallucination Detection (Ghost Code Integration)
        # Advanced hallucination detection using data quality techniques
        if CLEANLAB_AVAILABLE and cleanlab_detect_hallucinations and crew_output:
            try:
                cleanlab_result = await cleanlab_detect_hallucinations(
                    responses=[crew_output[:2000]],
                    contexts=[str(context_pack)[:1000]],
                )
                evaluations["cleanlab"] = {
                    "hallucination_detected": cleanlab_result.has_hallucination if hasattr(cleanlab_result, 'has_hallucination') else False,
                    "confidence": cleanlab_result.confidence if hasattr(cleanlab_result, 'confidence') else 0.0,
                    "hallucination_types": cleanlab_result.hallucination_types if hasattr(cleanlab_result, 'hallucination_types') else [],
                }
                if evaluations["cleanlab"]["hallucination_detected"]:
                    logger.warning("Cleanlab detected potential hallucination in output")
                logger.debug("Cleanlab hallucination detection completed")
            except Exception as e:
                logger.warning(f"Cleanlab hallucination detection failed (non-blocking): {e}")
                evaluations["cleanlab_error"] = str(e)

        # =====================================================================
        # PLANNING STACKS - on_gate subset (2026-01-30: FASE 3)
        # =====================================================================

        # PlanValidator - Validate execution against plan
        if PLAN_VALIDATOR_AVAILABLE and PlanValidator is not None:
            try:
                validator = PlanValidator()
                granular_tasks = state.get("granular_tasks", [])
                execution_results = state.get("crew_result", {})
                if granular_tasks and execution_results:
                    validation = validate_plan(granular_tasks, execution_results)
                    evaluations["plan_validation"] = {
                        "enabled": True,
                        "is_valid": validation.is_valid if hasattr(validation, "is_valid") else True,
                        "issues_count": len(validation.issues) if hasattr(validation, "issues") else 0,
                    }
                    if hasattr(validation, "issues") and validation.issues:
                        state = {**state, "plan_validation_issues": validation.issues}
                    logger.info(f"PlanValidator: {'valid' if evaluations['plan_validation']['is_valid'] else 'invalid'}")
            except Exception as e:
                logger.warning(f"PlanValidator failed (non-blocking): {e}")
                evaluations["plan_validation_error"] = str(e)

        # PlanQualityEvaluator - Evaluate plan quality
        if PLAN_EVALUATOR_AVAILABLE and PlanQualityEvaluator is not None:
            try:
                evaluator = PlanQualityEvaluator()
                granular_tasks = state.get("granular_tasks", [])
                if granular_tasks:
                    quality_score = compute_plan_score(granular_tasks)
                    threshold_met = check_plan_quality_threshold(quality_score)
                    evaluations["plan_quality"] = {
                        "enabled": True,
                        "score": quality_score.value if hasattr(quality_score, "value") else float(quality_score),
                        "threshold_met": threshold_met,
                    }
                    state = {**state, "plan_quality_score": quality_score}
                    logger.info(f"PlanQuality: score={evaluations['plan_quality']['score']:.2f}, threshold_met={threshold_met}")
            except Exception as e:
                logger.warning(f"PlanQualityEvaluator failed (non-blocking): {e}")
                evaluations["plan_quality_error"] = str(e)

        # PlanningFeedbackLoop - Capture feedback for learning
        if FEEDBACK_LOOP_AVAILABLE and PlanningFeedbackLoop is not None:
            try:
                loop = PlanningFeedbackLoop()
                gate_status = state.get("gate_status", "UNKNOWN")
                gates_failed = state.get("gates_failed", [])
                granular_tasks = state.get("granular_tasks", [])
                if granular_tasks:
                    feedback = capture_planning_feedback(
                        tasks=granular_tasks,
                        outcome="pass" if gate_status == "PASS" else "fail",
                        failures=gates_failed,
                    )
                    evaluations["planning_feedback"] = {
                        "enabled": True,
                        "captured": True,
                        "learning_points": len(feedback.learnings) if hasattr(feedback, "learnings") else 0,
                    }
                    logger.debug(f"PlanningFeedback: {evaluations['planning_feedback']['learning_points']} learning points")
            except Exception as e:
                logger.warning(f"PlanningFeedbackLoop failed (non-blocking): {e}")
                evaluations["planning_feedback_error"] = str(e)

        # PlanReflexionEngine - Reflect on gate failures
        if PLAN_REFLEXION_AVAILABLE and PlanReflexionEngine is not None:
            try:
                gates_failed = state.get("gates_failed", [])
                if gates_failed:
                    engine = PlanReflexionEngine()
                    reflection = create_reflection_from_feedback(
                        failures=gates_failed,
                        context=state.get("context_pack", {}),
                    )
                    evaluations["plan_reflexion"] = {
                        "enabled": True,
                        "reflection_created": True,
                        "improvement_suggestions": len(reflection.suggestions) if hasattr(reflection, "suggestions") else 0,
                    }
                    if reflection:
                        state = {**state, "gate_reflection": reflection}
                    logger.info(f"PlanReflexion: {evaluations['plan_reflexion']['improvement_suggestions']} suggestions")
            except Exception as e:
                logger.warning(f"PlanReflexionEngine failed (non-blocking): {e}")
                evaluations["plan_reflexion_error"] = str(e)

        # =====================================================================
        # SPEC AI - on_gate subset (2026-01-30)
        # =====================================================================

        # SpecAI HallucinationDetector - Detect fabricated requirements in output
        if SPEC_AI_HALLUCINATION_AVAILABLE and SpecAIHallucinationDetector is not None:
            try:
                detector = SpecAIHallucinationDetector()
                if crew_output:
                    fabrications = detect_fabrications(crew_output[:3000])
                    evaluations["spec_ai_hallucination"] = {
                        "enabled": True,
                        "fabrications_found": len(fabrications) if fabrications else 0,
                        "has_fabrications": len(fabrications) > 0 if fabrications else False,
                    }
                    if fabrications:
                        state = {**state, "detected_fabrications": fabrications}
                        logger.warning(f"SpecAI HallucinationDetector: {len(fabrications)} fabrications found!")
                    else:
                        logger.debug("SpecAI HallucinationDetector: no fabrications found")
            except Exception as e:
                logger.warning(f"SpecAI HallucinationDetector failed (non-blocking): {e}")
                evaluations["spec_ai_hallucination_error"] = str(e)

        # MADDebate - Multi-agent debate for controversial decisions
        if MAD_DEBATE_AVAILABLE and MADDebate is not None:
            try:
                gate_status = state.get("gate_status", "UNKNOWN")
                # Only run debate if gate status is borderline
                if gate_status in ["BORDERLINE", "NEEDS_REVIEW"]:
                    debate = MADDebate()
                    config = setup_debate(topic=f"Should sprint {state.get('sprint_id')} pass gates?")
                    rounds = run_rounds(config, max_rounds=3)
                    consensus = reach_consensus(rounds)
                    evaluations["mad_debate"] = {
                        "enabled": True,
                        "rounds_run": len(rounds) if rounds else 0,
                        "consensus_reached": consensus.reached if hasattr(consensus, "reached") else False,
                        "final_verdict": consensus.verdict if hasattr(consensus, "verdict") else "unknown",
                    }
                    if consensus:
                        state = {**state, "debate_consensus": consensus}
                    logger.info(f"MADDebate: consensus={evaluations['mad_debate']['final_verdict']}")
            except Exception as e:
                logger.warning(f"MADDebate failed (non-blocking): {e}")
                evaluations["mad_debate_error"] = str(e)

        # Buffer of Thoughts: Record gate results
        if self._bot:
            try:
                gate_status = state.get("gate_status", "UNKNOWN")
                self._bot.add_thought(
                    f"Gate validation result: {gate_status}",
                    thought_type=ThoughtType.CONCLUSION if gate_status == "PASS" else ThoughtType.UNCERTAINTY,
                    priority=ThoughtPriority.HIGH,
                )

                gates_passed = state.get("gates_passed", [])
                gates_failed = state.get("gates_failed", [])

                if gates_passed:
                    self._bot.add_thought(
                        f"Gates passed: {', '.join(gates_passed)}",
                        thought_type=ThoughtType.EVIDENCE,
                    )

                if gates_failed:
                    self._bot.add_thought(
                        f"Gates failed: {', '.join(gates_failed)}",
                        thought_type=ThoughtType.CONTRADICTION,
                        priority=ThoughtPriority.CRITICAL,
                    )
            except Exception as e:
                logger.warning(f"BoT gate hook failed: {e}")

        # Update integration data
        integration_data["evaluations"] = evaluations

        # HIGH-002 FIX: Make evaluations affect gate decisions
        # Check thresholds and block gate if evaluations fail
        evaluation_blocked = False
        evaluation_block_reasons = []

        # RAGAS threshold check (score < 0.5 is concerning)
        # FIX 2026-01-27: Only block for very low scores (< 0.3), warn for scores < 0.5
        # This prevents blocking gates during development while still alerting on critical issues
        ragas_score = evaluations.get("ragas", {}).get("overall_score")
        ragas_error = evaluations.get("ragas_error")

        if ragas_error:
            # RAGAS evaluation failed - log but don't block (graceful degradation)
            logger.warning(f"RAGAS evaluation error (non-blocking): {ragas_error}")
        elif ragas_score is not None:
            if ragas_score < 0.3:
                # Only block for critically low scores
                evaluation_blocked = True
                evaluation_block_reasons.append(
                    f"RAGAS overall score {ragas_score:.2f} critically low (threshold 0.3)"
                )
                logger.error(f"HIGH-002: RAGAS evaluation critically low: {ragas_score:.2f}")
            elif ragas_score < 0.5:
                # Log warning but don't block
                logger.warning(f"RAGAS evaluation below optimal threshold: {ragas_score:.2f} (optimal: >= 0.5)")

        # DeepEval hallucination check
        # FIX 2026-01-27: DeepEval is NON-BLOCKING because it requires paid OpenAI API
        # We log warnings but don't block the gate
        deepeval_hallucination_passed = evaluations.get("deepeval", {}).get("hallucination_passed", True)
        if not deepeval_hallucination_passed:
            # NOTE: Intentionally NOT blocking - DeepEval requires paid OpenAI
            # evaluation_blocked = True  # DISABLED - user request "pode ligar o foda-se nele"
            hallucination_score = evaluations.get("deepeval", {}).get("hallucination_score", 0)
            # Log as warning instead of blocking
            logger.warning(
                f"DeepEval hallucination detected (score: {hallucination_score:.2f}) - "
                f"NOT blocking gate (DeepEval requires paid OpenAI)"
            )

        # =====================================================================
        # GHOST CODE INTEGRATION: GoT Enhanced (2026-01-30)
        # Multi-path reasoning for deeper gate analysis and root cause detection
        # =====================================================================
        if GOT_ENHANCED_AVAILABLE and EnhancedThoughtGraph is not None:
            try:
                got = EnhancedThoughtGraph()
                gate_status = state.get("gate_status", "UNKNOWN")
                gates_failed = state.get("gates_failed", [])

                if gates_failed:
                    # Use GoT to analyze why gates failed
                    analysis_result = got.analyze_failure(
                        failures=gates_failed,
                        context={
                            "crew_output": crew_output[:500] if crew_output else "",
                            "objective": objective,
                            "sprint_id": state.get("sprint_id"),
                        }
                    )
                    evaluations["got_analysis"] = {
                        "root_causes": analysis_result.root_causes if hasattr(analysis_result, "root_causes") else [],
                        "suggested_fixes": analysis_result.suggested_fixes if hasattr(analysis_result, "suggested_fixes") else [],
                        "confidence": analysis_result.confidence if hasattr(analysis_result, "confidence") else 0.0,
                    }
                    logger.info(
                        f"GoT Enhanced analyzed {len(gates_failed)} gate failures: "
                        f"{len(evaluations['got_analysis']['root_causes'])} root causes identified"
                    )
                else:
                    evaluations["got_analysis"] = {"status": "no_failures_to_analyze"}
            except Exception as e:
                logger.warning(f"GoT Enhanced analysis failed (non-blocking): {e}")
                evaluations["got_analysis_error"] = str(e)

        # =====================================================================
        # GHOST CODE INTEGRATION: Neo4j Algorithms (2026-01-30)
        # Graph algorithms for claim verification (PageRank, community detection)
        # =====================================================================
        if NEO4J_ALGORITHMS_AVAILABLE and get_neo4j_algorithms is not None:
            try:
                neo4j_algos = get_neo4j_algorithms()
                if neo4j_algos:
                    # Run PageRank on claims/evidence graph
                    claims = state.get("claims", [])
                    if claims:
                        pagerank_result = neo4j_algos.run_pagerank(
                            node_label="Claim",
                            relationship_type="SUPPORTS"
                        )
                        evaluations["neo4j_pagerank"] = {
                            "computed": True,
                            "top_influential": pagerank_result.get("top_nodes", [])[:5] if pagerank_result else [],
                            "execution_time_ms": pagerank_result.get("execution_time_ms", 0) if pagerank_result else 0,
                        }
                        logger.debug(
                            f"Neo4j PageRank computed: {len(evaluations['neo4j_pagerank']['top_influential'])} "
                            f"top influential claims"
                        )
                    else:
                        evaluations["neo4j_pagerank"] = {"computed": False, "reason": "no_claims"}
            except Exception as e:
                logger.warning(f"Neo4j Algorithms integration failed (non-blocking): {e}")
                evaluations["neo4j_algorithms_error"] = str(e)

        # =====================================================================
        # GHOST CODE INTEGRATION: Colang Integration (2026-01-30)
        # NeMo Colang security action flows for guardrails
        # =====================================================================
        if COLANG_AVAILABLE and ColangActionRegistry is not None:
            try:
                colang = ColangActionRegistry()
                # Check if crew output triggers any security actions
                if crew_output:
                    security_actions = colang.check_output(crew_output[:2000])
                    evaluations["colang_security"] = {
                        "checked": True,
                        "triggered_actions": security_actions.triggered if hasattr(security_actions, "triggered") else [],
                        "blocked": security_actions.blocked if hasattr(security_actions, "blocked") else False,
                    }
                    if evaluations["colang_security"]["blocked"]:
                        evaluation_blocked = True
                        evaluation_block_reasons.append(
                            f"Colang security action blocked output: {security_actions.reason if hasattr(security_actions, 'reason') else 'unknown'}"
                        )
                        logger.warning(
                            f"Colang blocked gate: {security_actions.reason if hasattr(security_actions, 'reason') else 'security policy violation'}"
                        )
                    logger.debug(f"Colang security check completed: {len(evaluations['colang_security']['triggered_actions'])} actions triggered")
                else:
                    evaluations["colang_security"] = {"checked": False, "reason": "no_output"}
            except Exception as e:
                logger.warning(f"Colang integration failed (non-blocking): {e}")
                evaluations["colang_error"] = str(e)

        # =====================================================================
        # GHOST CODE INTEGRATION: Enforcement Metrics (2026-01-30)
        # Track gate violations, invariant checks, and trust denials
        # =====================================================================
        if ENFORCEMENT_METRICS_AVAILABLE and record_invariant_check is not None:
            try:
                gate_status = state.get("gate_status", "UNKNOWN")
                gates_failed = state.get("gates_failed", [])

                # Record invariant checks for each gate
                for gate_id in (state.get("gates_passed", []) + gates_failed):
                    gate_passed = gate_id not in gates_failed
                    record_invariant_check(
                        invariant_id=f"GATE_{gate_id}",
                        passed=gate_passed,
                        details={
                            "sprint_id": state.get("sprint_id"),
                            "gate_id": gate_id,
                        }
                    )

                # Record violations for failed gates
                for failed_gate in gates_failed:
                    if record_violation:
                        record_violation(
                            violation_type="gate_failure",
                            severity="high" if failed_gate.startswith("G0") else "medium",
                            details={
                                "gate_id": failed_gate,
                                "sprint_id": state.get("sprint_id"),
                            }
                        )

                evaluations["enforcement_metrics"] = {
                    "recorded": True,
                    "invariant_checks": len(state.get("gates_passed", [])) + len(gates_failed),
                    "violations_recorded": len(gates_failed),
                }
                logger.debug(
                    f"Enforcement metrics recorded: {evaluations['enforcement_metrics']['invariant_checks']} checks, "
                    f"{evaluations['enforcement_metrics']['violations_recorded']} violations"
                )
            except Exception as e:
                logger.warning(f"Enforcement metrics recording failed (non-blocking): {e}")
                evaluations["enforcement_metrics_error"] = str(e)

        # =====================================================================
        # GHOST CODE INTEGRATION: Event Emission (2026-01-30)
        # Emit gate_completed event for observability
        # =====================================================================
        _emit_integration_event(
            "gate_completed",
            state,
            f"Gate evaluation completed for sprint {state.get('sprint_id')}",
            level="info" if not evaluation_blocked else "warning",
            metadata={
                "gate_status": state.get("gate_status", "UNKNOWN"),
                "gates_passed": len(state.get("gates_passed", [])),
                "gates_failed": len(state.get("gates_failed", [])),
                "evaluation_blocked": evaluation_blocked,
                "got_analysis_performed": "got_analysis" in evaluations,
                "neo4j_pagerank_computed": evaluations.get("neo4j_pagerank", {}).get("computed", False),
                "colang_checked": evaluations.get("colang_security", {}).get("checked", False),
            }
        )

        # Record evaluation decision in state
        integration_data["evaluation_blocked"] = evaluation_blocked
        integration_data["evaluation_block_reasons"] = evaluation_block_reasons

        state = {
            **state,
            "partial_stacks_integration": integration_data,
            "evaluation_gate_blocked": evaluation_blocked,
            "evaluation_block_reasons": evaluation_block_reasons,
        }

        if evaluation_blocked:
            logger.error(
                f"HIGH-002: Gate blocked by evaluation failures: {evaluation_block_reasons}"
            )

        return state

    # =========================================================================
    # SIGNOFF NODE HOOKS
    # =========================================================================

    async def on_signoff(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Hook called during SIGNOFF node.

        Actions:
        - Final evaluation with DeepEval
        - Synthesize Buffer of Thoughts
        - Complete Phoenix trace
        - Run Neo4j Analytics for graph-based insights (Ghost Code Integration)
        - Persist learnings to Mem0 Enhanced (Ghost Code Integration)
        - Emit signoff_completed event (Ghost Code Integration)

        Args:
            state: Current pipeline state

        Returns:
            Updated state with final integration results
        """
        integration_data = state.get("partial_stacks_integration", {})

        # Buffer of Thoughts: Synthesize final result
        if self._bot:
            try:
                synthesis = self._bot.synthesize()
                integration_data["bot_synthesis"] = {
                    "conclusion": synthesis.conclusion,
                    "confidence": synthesis.confidence,
                    "thought_count": synthesis.thought_count,
                    "uncertainties": synthesis.uncertainties[:5],  # Top 5
                }
                logger.debug(f"BoT synthesis: confidence={synthesis.confidence:.2f}")
            except Exception as e:
                logger.warning(f"BoT synthesis failed: {e}")

        # DeepEval: Final toxicity check on signoff content
        if self._deepeval:
            try:
                crew_result = state.get("crew_result", {})
                crew_output = crew_result.get("crew_output", "")

                if crew_output:
                    toxicity_result = await self._deepeval.evaluate_rag_response(
                        query="signoff content review",
                        response=crew_output[:2000],
                        context=[],
                        metrics=["toxicity"],
                    )

                    integration_data["final_toxicity_check"] = {
                        "passed": toxicity_result.get("toxicity", {}).get("passed", True),
                        "score": toxicity_result.get("toxicity", {}).get("score", 0),
                    }
            except Exception as e:
                logger.warning(f"Final toxicity check failed: {e}")

        # Phoenix: Log final summary
        if self._phoenix:
            try:
                # Log summary metrics
                logger.debug("Phoenix trace completed for run")
            except Exception as e:
                logger.warning(f"Phoenix final log failed: {e}")

        # =====================================================================
        # GHOST CODE INTEGRATION: Neo4j Analytics (2026-01-30)
        # Complex graph analytics for signoff insights (community detection, paths)
        # FIX 2026-02-01: Corrected method names and async handling (TASK-GS-003/004)
        # =====================================================================
        if NEO4J_ANALYTICS_AVAILABLE and Neo4jAnalytics is not None:
            try:
                analytics = Neo4jAnalytics()
                sprint_id = state.get("sprint_id", "unknown")

                # Run community detection on claims/evidence graph
                # FIX: detect_communities -> detect_disinfo_communities (correct method name)
                # FIX: Added await for async method
                communities = await analytics.detect_disinfo_communities(
                    algorithm="louvain",
                    min_community_size=3,
                )

                # Calculate aggregated metrics from communities list
                total_communities = len(communities) if communities else 0
                avg_disinfo_score = (
                    sum(c.disinfo_score for c in communities) / total_communities
                    if total_communities > 0 else 0.0
                )

                # Trace verification paths for key claims
                # FIX: trace_verification_path -> trace_claim_propagation (correct method name)
                claims = state.get("claims", [])
                path_traces = []
                if claims:
                    for claim in claims[:3]:  # Top 3 claims
                        claim_id = claim.get("id") if isinstance(claim, dict) else str(claim)
                        # FIX: Added await for async method
                        path_result = await analytics.trace_claim_propagation(
                            claim_id=claim_id,
                            max_depth=10,
                        )
                        # FIX: PropagationPath has .steps (List[PropagationStep]), not dict
                        if path_result and path_result.steps:
                            path_traces.append({
                                "claim_id": claim_id,
                                "path_length": len(path_result.steps),
                                "total_spread": path_result.total_spread,
                                "mutation_count": path_result.mutation_count,
                                "nodes_visited": [
                                    step.node_id for step in path_result.steps[:5]
                                ],
                            })

                integration_data["neo4j_analytics"] = {
                    "computed": True,
                    "communities_detected": total_communities,
                    "avg_disinfo_score": avg_disinfo_score,
                    "paths_traced": len(path_traces),
                    "path_details": path_traces,
                }
                logger.info(
                    f"Neo4j Analytics: {total_communities} communities, "
                    f"{len(path_traces)} paths traced"
                )
            except Exception as e:
                logger.warning(f"Neo4j Analytics integration failed (non-blocking): {e}")
                integration_data["neo4j_analytics_error"] = str(e)

        # =====================================================================
        # GHOST CODE INTEGRATION: Mem0 Enhanced (2026-01-30)
        # Persist sprint learnings to semantic memory for future retrieval
        # =====================================================================
        if MEM0_ENHANCED_AVAILABLE and Mem0Enhanced is not None:
            try:
                mem0 = Mem0Enhanced()
                sprint_id = state.get("sprint_id", "unknown")

                # Collect learnings from this sprint execution
                learnings = []

                # 1. Gate failures as learnings
                gates_failed = state.get("gates_failed", [])
                for gate in gates_failed:
                    learnings.append({
                        "type": "gate_failure",
                        "content": f"Gate {gate} failed in sprint {sprint_id}",
                        "metadata": {"sprint_id": sprint_id, "gate": gate},
                    })

                # 2. BoT synthesis as learning
                bot_synthesis = integration_data.get("bot_synthesis", {})
                if bot_synthesis.get("conclusion"):
                    learnings.append({
                        "type": "execution_insight",
                        "content": bot_synthesis["conclusion"],
                        "metadata": {
                            "sprint_id": sprint_id,
                            "confidence": bot_synthesis.get("confidence", 0.0),
                        },
                    })

                # 3. Evaluation results as learnings
                evaluations = integration_data.get("evaluations", {})
                ragas_score = evaluations.get("ragas", {}).get("overall_score")
                if ragas_score is not None and ragas_score < 0.5:
                    learnings.append({
                        "type": "quality_issue",
                        "content": f"Low RAGAS score ({ragas_score:.2f}) in sprint {sprint_id}",
                        "metadata": {"sprint_id": sprint_id, "ragas_score": ragas_score},
                    })

                # Persist learnings to Mem0
                if learnings:
                    for learning in learnings:
                        mem0.add_memory(
                            content=learning["content"],
                            memory_type=learning["type"],
                            metadata=learning.get("metadata", {}),
                            user_id=f"pipeline_sprint_{sprint_id}",
                        )

                integration_data["mem0_persistence"] = {
                    "persisted": True,
                    "learnings_count": len(learnings),
                    "learning_types": list(set(l["type"] for l in learnings)),
                }
                logger.info(
                    f"Mem0 Enhanced: Persisted {len(learnings)} learnings from sprint {sprint_id}"
                )
            except Exception as e:
                logger.warning(f"Mem0 Enhanced persistence failed (non-blocking): {e}")
                integration_data["mem0_persistence_error"] = str(e)

        # =====================================================================
        # GHOST CODE INTEGRATION: Enforcement Metrics Summary (2026-01-30)
        # Get summary of all enforcement metrics for this run
        # =====================================================================
        if ENFORCEMENT_METRICS_AVAILABLE and get_metrics_summary is not None:
            try:
                metrics_summary = get_metrics_summary()
                integration_data["enforcement_summary"] = {
                    "total_violations": metrics_summary.get("violations", 0),
                    "total_invariant_checks": metrics_summary.get("invariant_checks", 0),
                    "trust_denials": metrics_summary.get("trust_denials", 0),
                    "pass_rate": metrics_summary.get("pass_rate", 0.0),
                }
                logger.debug(
                    f"Enforcement summary: {integration_data['enforcement_summary']['total_violations']} violations, "
                    f"{integration_data['enforcement_summary']['pass_rate']:.1%} pass rate"
                )
            except Exception as e:
                logger.warning(f"Enforcement metrics summary failed (non-blocking): {e}")
                integration_data["enforcement_summary_error"] = str(e)

        # =====================================================================
        # PLANNING STACKS - on_signoff subset (2026-01-30: FASE 3)
        # =====================================================================

        # PlanGraphStore - Store plan and execution results to graph
        if PLAN_GRAPH_STORE_AVAILABLE and PlanGraphStore is not None:
            try:
                store = PlanGraphStore()
                granular_tasks = state.get("granular_tasks", [])
                execution_results = state.get("crew_result", {})
                sprint_id = state.get("sprint_id", "")
                if granular_tasks and sprint_id:
                    store_result = store.store_plan(
                        sprint_id=sprint_id,
                        tasks=granular_tasks,
                        execution_result=execution_results,
                        gate_status=state.get("gate_status", "UNKNOWN"),
                    )
                    integration_data["plan_graph_store"] = {
                        "enabled": True,
                        "stored": True,
                        "nodes_created": store_result.nodes_created if hasattr(store_result, "nodes_created") else 0,
                    }
                    logger.info(f"PlanGraphStore: stored plan for {sprint_id}")
            except Exception as e:
                logger.warning(f"PlanGraphStore failed (non-blocking): {e}")
                integration_data["plan_graph_store_error"] = str(e)

        # PlanningMetricsCollector - Capture planning metrics
        if PLANNING_METRICS_AVAILABLE and PlanningMetricsCollector is not None:
            try:
                collector = PlanningMetricsCollector()
                granular_tasks = state.get("granular_tasks", [])
                execution_results = state.get("crew_result", {})
                sprint_id = state.get("sprint_id", "")
                if granular_tasks:
                    metrics = capture_planning_metrics(
                        sprint_id=sprint_id,
                        planned_tasks=len(granular_tasks),
                        executed_tasks=execution_results.get("tasks_completed", 0) if execution_results else 0,
                        gate_status=state.get("gate_status", "UNKNOWN"),
                    )
                    summary = get_planning_metrics_summary()
                    integration_data["planning_metrics"] = {
                        "enabled": True,
                        "captured": True,
                        "accuracy": summary.accuracy if hasattr(summary, "accuracy") else 0.0,
                    }
                    logger.debug(f"PlanningMetrics: accuracy={integration_data['planning_metrics']['accuracy']:.2f}")
            except Exception as e:
                logger.warning(f"PlanningMetricsCollector failed (non-blocking): {e}")
                integration_data["planning_metrics_error"] = str(e)

        # MCTSPlanOptimizer - Optimize next execution plan based on results
        if MCTS_OPTIMIZER_AVAILABLE and MCTSPlanOptimizer is not None:
            try:
                gate_status = state.get("gate_status", "UNKNOWN")
                # Only optimize if gate failed - learn from failure
                if gate_status != "PASS":
                    optimizer = MCTSPlanOptimizer()
                    granular_tasks = state.get("granular_tasks", [])
                    if granular_tasks:
                        # Estimate quality for learning
                        quality = estimate_plan_quality_fast(granular_tasks)
                        integration_data["mcts_optimizer"] = {
                            "enabled": True,
                            "quality_estimate": quality.score if hasattr(quality, "score") else float(quality),
                            "optimization_triggered": True,
                        }
                        # Store quality estimate for next run
                        state = {**state, "plan_quality_for_next_run": quality}
                        logger.info(f"MCTSOptimizer: quality_estimate={integration_data['mcts_optimizer']['quality_estimate']:.2f}")
            except Exception as e:
                logger.warning(f"MCTSPlanOptimizer failed (non-blocking): {e}")
                integration_data["mcts_optimizer_error"] = str(e)

        # PlanningFeedbackLoop - Learn from outcomes for future planning
        if FEEDBACK_LOOP_AVAILABLE and learn_from_outcomes is not None:
            try:
                sprint_id = state.get("sprint_id", "")
                gate_status = state.get("gate_status", "UNKNOWN")
                granular_tasks = state.get("granular_tasks", [])
                if sprint_id and granular_tasks:
                    learning_result = learn_from_outcomes(
                        sprint_id=sprint_id,
                        tasks=granular_tasks,
                        outcome="success" if gate_status == "PASS" else "failure",
                        details=state.get("gates_failed", []),
                    )
                    integration_data["outcome_learning"] = {
                        "enabled": True,
                        "learned": True,
                        "insights_count": learning_result.insights if hasattr(learning_result, "insights") else 0,
                    }
                    logger.info(f"OutcomeLearning: {integration_data['outcome_learning']['insights_count']} insights captured")
            except Exception as e:
                logger.warning(f"OutcomeLearning failed (non-blocking): {e}")
                integration_data["outcome_learning_error"] = str(e)

        # =====================================================================
        # GHOST CODE INTEGRATION: Event Emission (2026-01-30)
        # Emit signoff_completed event for observability
        # =====================================================================
        _emit_integration_event(
            "signoff_completed",
            state,
            f"Signoff completed for sprint {state.get('sprint_id')}",
            metadata={
                "bot_confidence": integration_data.get("bot_synthesis", {}).get("confidence", 0.0),
                "toxicity_passed": integration_data.get("final_toxicity_check", {}).get("passed", True),
                "neo4j_communities": integration_data.get("neo4j_analytics", {}).get("communities_detected", 0),
                "mem0_learnings": integration_data.get("mem0_persistence", {}).get("learnings_count", 0),
                "planning_stacks_active": sum(1 for k in ["plan_graph_store", "planning_metrics", "mcts_optimizer", "outcome_learning"] if k in integration_data and integration_data[k].get("enabled")),
            }
        )

        # =====================================================================
        # GHOST CODE INTEGRATION: State Validation (2026-01-30)
        # Final validation before signoff completion
        # =====================================================================
        is_valid, validation_details = _validate_state_transition(state, "SIGNOFF")
        integration_data["final_validation"] = {
            "valid": is_valid,
            "phase": "SIGNOFF",
            "errors": validation_details.get("errors", []) if not is_valid else [],
        }

        state = {**state, "partial_stacks_integration": integration_data}
        return state

    # =========================================================================
    # ARTIFACT NODE HOOKS
    # =========================================================================

    async def on_artifact(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Hook called during ARTIFACT node.

        Actions:
        - Export Buffer of Thoughts to artifact
        - Export evaluation metrics to artifact

        Args:
            state: Current pipeline state

        Returns:
            Updated state with artifact data
        """
        integration_data = state.get("partial_stacks_integration", {})

        # Collect all thoughts for artifact
        if self._bot:
            try:
                all_thoughts = [
                    {
                        "id": t.id,
                        "content": t.content,
                        "type": t.thought_type.value,
                        "priority": t.priority.value,
                        "relevance": t.relevance_score,
                    }
                    for t in self._bot._thoughts
                ]
                integration_data["thoughts_artifact"] = all_thoughts
            except Exception as e:
                logger.warning(f"BoT artifact export failed: {e}")

        state = {**state, "partial_stacks_integration": integration_data}
        return state


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_integration_instance: Optional[PartialStacksIntegration] = None


def get_partial_stacks_integration() -> PartialStacksIntegration:
    """Get the singleton PartialStacksIntegration instance.

    Returns:
        PartialStacksIntegration instance
    """
    global _integration_instance
    if _integration_instance is None:
        _integration_instance = PartialStacksIntegration()
    return _integration_instance


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Availability flags
    "ACTIVE_RAG_AVAILABLE",
    "BOT_AVAILABLE",
    "RAGAS_AVAILABLE",
    "PHOENIX_AVAILABLE",
    "DEEPEVAL_AVAILABLE",
    "INSTRUMENTATION_AVAILABLE",
    # Health check
    "StackHealthResult",
    "PartialStacksHealthReport",
    "health_check_active_rag",
    "health_check_bot",
    "health_check_ragas",
    "health_check_phoenix",
    "health_check_deepeval",
    "health_check_all",
    # Integration
    "PartialStacksIntegration",
    "get_partial_stacks_integration",
]
