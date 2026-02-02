"""LangGraph Pipeline V2 Module.

This module provides the LangGraph-based control plane for Pipeline V2,
enabling stateful workflows with checkpointing, retry, and resume capabilities.

Architecture:
    LangGraph (Control Plane) -> CrewAI (Data Plane)
                |
                v
    PipelineState (Checkpointed)
                |
                v
    Nodes: INIT -> EXEC -> GATE -> SIGNOFF -> ARTIFACT

Key Components:
    - state.py: PipelineState TypedDict with all required fields
    - workflow.py: StateGraph definition and compilation
    - stack_injection.py: Stack wiring and injection
    - checkpointer.py: Redis/file-based checkpointing

Author: Pipeline Autonomo Team
Version: 1.0.0 (2026-01-16)
"""

import logging

logger = logging.getLogger(__name__)

from pipeline.langgraph.state import (
    # Core State
    PipelineState,
    SprintPhase,
    PipelineStatus,
    GateStatus,

    # Sub-States
    IdentityState,
    PointerState,
    PolicySnapshotState,
    StackHealthEntry,
    StackHealthState,
    ContextPackState,
    HandoffEntry,
    ApprovalEntry,
    SignoffEntry,
    MilestoneEntry,
    GovernanceState,
    TaskEntry,
    TemporalState,
    ExecutionState,
    GateState,
    MemoryState,
    NodeHistoryEntry,
    EventResumeState,
    TimestampState,
    ErrorEntry,

    # New Stacks State (BoT, Active RAG, Security)
    ThoughtTemplate,
    ThoughtBufferState,
    RAGContextState,
    SecurityContextState,

    # P0-05: LazyContextPack
    LazyContextPack,

    # Factory Functions
    create_initial_state,
    create_policy_snapshot,
    create_stack_health_entry,
)

__all__ = [
    # Core State
    "PipelineState",
    "SprintPhase",
    "PipelineStatus",
    "GateStatus",

    # Sub-States
    "IdentityState",
    "PointerState",
    "PolicySnapshotState",
    "StackHealthEntry",
    "StackHealthState",
    "ContextPackState",
    "HandoffEntry",
    "ApprovalEntry",
    "SignoffEntry",
    "MilestoneEntry",
    "GovernanceState",
    "TaskEntry",
    "TemporalState",
    "ExecutionState",
    "GateState",
    "MemoryState",
    "NodeHistoryEntry",
    "EventResumeState",
    "TimestampState",
    "ErrorEntry",

    # New Stacks State
    "ThoughtTemplate",
    "ThoughtBufferState",
    "RAGContextState",
    "SecurityContextState",

    # P0-05: LazyContextPack
    "LazyContextPack",

    # Factory Functions
    "create_initial_state",
    "create_policy_snapshot",
    "create_stack_health_entry",
]

# Import workflow module
try:
    from pipeline.langgraph.workflow import (
        WorkflowNodes,
        build_workflow,
        run_workflow,
        LANGGRAPH_AVAILABLE,
    )
    __all__.extend([
        "WorkflowNodes",
        "build_workflow",
        "run_workflow",
        "LANGGRAPH_AVAILABLE",
    ])
except ImportError as e:
    logger.warning(f"workflow module import failed: {e}")
    LANGGRAPH_AVAILABLE = False

# Import stack injection module
try:
    from pipeline.langgraph.stack_injection import (
        Operation,
        StackPolicy,
        StackHealth,
        StackContext,
        StackInjector,
        get_stack_injector,
        get_stacks,
    )
    __all__.extend([
        "Operation",
        "StackPolicy",
        "StackHealth",
        "StackContext",
        "StackInjector",
        "get_stack_injector",
        "get_stacks",
    ])
except ImportError as e:
    logger.warning(f"stack_injection module import failed: {e}")

# Import checkpointer module
try:
    from pipeline.langgraph.checkpointer import (
        Checkpoint,
        BaseCheckpointer,
        FileCheckpointer,
        RedisCheckpointer,
        HybridCheckpointer,
        create_checkpointer,
        create_langgraph_checkpointer,
        generate_checkpoint_id,
    )
    __all__.extend([
        "Checkpoint",
        "BaseCheckpointer",
        "FileCheckpointer",
        "RedisCheckpointer",
        "HybridCheckpointer",
        "create_checkpointer",
        "create_langgraph_checkpointer",
        "generate_checkpoint_id",
    ])
except ImportError as e:
    logger.warning(f"checkpointer module import failed: {e}")

# Import bridge module
try:
    from pipeline.langgraph.bridge import (
        LangGraphConfig,
        LangGraphBridge,
        get_langgraph_config,
        get_langgraph_bridge,
        run_sprint_with_langgraph,
        run_sprint_sync,
        pydantic_to_langgraph_state,
        langgraph_to_sprint_result,
    )
    __all__.extend([
        "LangGraphConfig",
        "LangGraphBridge",
        "get_langgraph_config",
        "get_langgraph_bridge",
        "run_sprint_with_langgraph",
        "run_sprint_sync",
        "pydantic_to_langgraph_state",
        "langgraph_to_sprint_result",
    ])
except ImportError as e:
    logger.warning(f"bridge module import failed: {e}")

# Import event writer module (CRIT-05 fix)
try:
    from pipeline.langgraph.event_writer import (
        Event,
        AtomicEventWriter,
        get_event_writer,
        emit_event,
    )
    __all__.extend([
        "Event",
        "AtomicEventWriter",
        "get_event_writer",
        "emit_event",
    ])
except ImportError as e:
    logger.warning(f"event_writer module import failed: {e}")

# Import invariants module (CRIT-01, CRIT-04 fix, I1-I11)
try:
    from pipeline.langgraph.invariants import (
        # Codes
        InvariantCode,
        InvariantViolation,
        InvariantCheckResult,
        # Enforcers I1-I4
        NamespacingEnforcer,
        IdempotencyEnforcer,
        PhaseOrderEnforcer,
        GatesBeforeSignoffEnforcer,
        # Enforcers I5-I11
        ExecutiveVerificationEnforcer,
        TruthfulnessEnforcer,
        AuditTrailEnforcer,
        SafeHaltEnforcer,
        RedisCanonicalEnforcer,
        RunawayProtectionEnforcer,
        # Combined
        InvariantChecker,
        get_invariant_checker,
    )
    __all__.extend([
        "InvariantCode",
        "InvariantViolation",
        "InvariantCheckResult",
        "NamespacingEnforcer",
        "IdempotencyEnforcer",
        "PhaseOrderEnforcer",
        "GatesBeforeSignoffEnforcer",
        "ExecutiveVerificationEnforcer",
        "TruthfulnessEnforcer",
        "AuditTrailEnforcer",
        "SafeHaltEnforcer",
        "RedisCanonicalEnforcer",
        "RunawayProtectionEnforcer",
        "InvariantChecker",
        "get_invariant_checker",
    ])
except ImportError as e:
    logger.warning(f"invariants module import failed: {e}")

# Import trust boundaries module (CRIT-03 fix)
try:
    from pipeline.langgraph.trust_boundaries import (
        AgentTier,
        ResourceType,
        Action,
        TrustBoundaryViolation,
        AccessCheckResult,
        TrustBoundaryEnforcer,
        get_trust_boundary_enforcer,
        check_access,
        require_access,
        AGENT_TIERS,
        DEFAULT_ACCESS_POLICIES,
    )
    __all__.extend([
        "AgentTier",
        "ResourceType",
        "Action",
        "TrustBoundaryViolation",
        "AccessCheckResult",
        "TrustBoundaryEnforcer",
        "get_trust_boundary_enforcer",
        "check_access",
        "require_access",
        "AGENT_TIERS",
        "DEFAULT_ACCESS_POLICIES",
    ])
except ImportError as e:
    logger.warning(f"trust_boundaries module import failed: {e}")

# Import cerebro stack mapping
try:
    from pipeline.langgraph.cerebro_stack_mapping import (
        StackCategory,
        CerebroStackConfig,
        CEREBRO_STACK_MAPPINGS,
        get_cerebro_config,
        get_cerebros_by_tier,
        get_cerebros_using_stack,
        get_stack_usage_summary,
    )
    __all__.extend([
        "StackCategory",
        "CerebroStackConfig",
        "CEREBRO_STACK_MAPPINGS",
        "get_cerebro_config",
        "get_cerebros_by_tier",
        "get_cerebros_using_stack",
        "get_stack_usage_summary",
    ])
except ImportError as e:
    logger.warning(f"cerebro_stack_mapping module import failed: {e}")

# Import stack synergy module
try:
    from pipeline.langgraph.stack_synergy import (
        StackStatus,
        StackCapability,
        AgentStackContext,
        SynergyReport,
        STACK_CAPABILITIES,
        StackSynergyManager,
        get_synergy_manager,
        get_agent_stack_context,
        inject_stacks_into_prompt,
        get_synergy_report,
        get_best_stack,
    )
    __all__.extend([
        "StackStatus",
        "StackCapability",
        "AgentStackContext",
        "SynergyReport",
        "STACK_CAPABILITIES",
        "StackSynergyManager",
        "get_synergy_manager",
        "get_agent_stack_context",
        "inject_stacks_into_prompt",
        "get_synergy_report",
        "get_best_stack",
    ])
except ImportError as e:
    logger.warning(f"stack_synergy module import failed: {e}")

# Import stack mastery module (comprehensive stack usage guide)
try:
    from pipeline.langgraph.stack_mastery import (
        StackMastery,
        STACK_MASTERY,
        get_stack_mastery,
        get_all_masteries,
        generate_mastery_prompt,
    )
    __all__.extend([
        "StackMastery",
        "STACK_MASTERY",
        "get_stack_mastery",
        "get_all_masteries",
        "generate_mastery_prompt",
    ])
except ImportError as e:
    logger.warning(f"stack_mastery module import failed: {e}")

# Import stack guardrails module (enforcement and circuit breakers)
try:
    from pipeline.langgraph.stack_guardrails import (
        StackGuardrails,
        CircuitBreaker,
        CircuitState,
        ViolationTracker,
        get_guardrails,
        validate_stacks,
        protected_stack_call,
        require_stack,
        with_circuit_breaker,
        get_guardrails_status,
    )
    __all__.extend([
        "StackGuardrails",
        "CircuitBreaker",
        "CircuitState",
        "ViolationTracker",
        "get_guardrails",
        "validate_stacks",
        "protected_stack_call",
        "require_stack",
        "with_circuit_breaker",
        "get_guardrails_status",
    ])
except ImportError as e:
    logger.warning(f"stack_guardrails module import failed: {e}")

# Import NeMo stack rails module (policy enforcement)
try:
    from pipeline.langgraph.nemo_stack_rails import (
        NemoStackRails,
        get_nemo_rails,
        validate_stack_usage,
        generate_with_rails,
        enforce_stacks,
        StackEnforcementError,
    )
    __all__.extend([
        "NemoStackRails",
        "get_nemo_rails",
        "validate_stack_usage",
        "generate_with_rails",
        "enforce_stacks",
        "StackEnforcementError",
    ])
except ImportError as e:
    logger.warning(f"nemo_stack_rails module import failed: {e}")

# Import stack auto-wiring module (automatic stack integration)
try:
    from pipeline.langgraph.stack_autowire import (
        StackAutoWire,
        AutoWireRule,
        StackAutoWireError,
        get_autowire,
        autowire_operation,
        autowire,
        autowire_class,
        get_autowire_stats,
        integrate_all,
        AUTO_WIRE_RULES,
    )
    __all__.extend([
        "StackAutoWire",
        "AutoWireRule",
        "StackAutoWireError",
        "get_autowire",
        "autowire_operation",
        "autowire",
        "autowire_class",
        "get_autowire_stats",
        "integrate_all",
        "AUTO_WIRE_RULES",
    ])

    # Auto-wire stacks on module load
    # This ensures stacks are automatically used in operations where they add value
    try:
        integrate_all()
        logger.info("Stack auto-wiring activated on langgraph module load")
    except Exception as e:
        logger.debug(f"Stack auto-wiring deferred (non-fatal): {e}")

except ImportError as e:
    logger.warning(f"stack_autowire module import failed: {e}")

# Import deepeval integration module (LLM output evaluation)
try:
    from pipeline.langgraph.deepeval_integration import (
        DeepEvalWrapper,
        EvaluationResult,
        MetricResult,
        EvaluationMetric,
        get_deepeval,
        evaluate_llm_output,
        evaluate_with_context,
        get_evaluation_metrics,
        is_deepeval_available,
        evaluate_output,
    )
    __all__.extend([
        "DeepEvalWrapper",
        "EvaluationResult",
        "MetricResult",
        "EvaluationMetric",
        "get_deepeval",
        "evaluate_llm_output",
        "evaluate_with_context",
        "get_evaluation_metrics",
        "is_deepeval_available",
        "evaluate_output",
    ])
except ImportError as e:
    logger.warning(f"deepeval_integration module import failed: {e}")

# P2-016: Import subgraphs module (modular workflow components)
try:
    from pipeline.langgraph.subgraphs import (
        # Base class
        BaseSubgraph,
        SubgraphState,
        # Concrete subgraphs
        GateValidationSubgraph,
        QualityAssuranceSubgraph,
        SignoffSubgraph,
        ArtifactGenerationSubgraph,
        # Composer function
        compose_workflow_with_subgraphs,
        # Availability flag
        LANGGRAPH_SUBGRAPHS_AVAILABLE,
    )
    __all__.extend([
        "BaseSubgraph",
        "SubgraphState",
        "GateValidationSubgraph",
        "QualityAssuranceSubgraph",
        "SignoffSubgraph",
        "ArtifactGenerationSubgraph",
        "compose_workflow_with_subgraphs",
        "LANGGRAPH_SUBGRAPHS_AVAILABLE",
    ])
except ImportError as e:
    logger.warning(f"subgraphs module import failed: {e}")
    LANGGRAPH_SUBGRAPHS_AVAILABLE = False

# P2-022: Import Colang integration module (NeMo Colang flows)
try:
    from pipeline.langgraph.colang_integration import (
        # Classes
        ColangActionRegistry,
        ColangIntegration,
        # Decorators
        colang_action,
        # Functions
        get_colang_integration,
        is_colang_available,
        generate_with_colang,
        # Constants
        COLANG_INTEGRATION_AVAILABLE,
        NEMO_CONFIG_DIR,
    )
    __all__.extend([
        "ColangActionRegistry",
        "ColangIntegration",
        "colang_action",
        "get_colang_integration",
        "is_colang_available",
        "generate_with_colang",
        "COLANG_INTEGRATION_AVAILABLE",
        "NEMO_CONFIG_DIR",
    ])
except ImportError as e:
    logger.warning(f"colang_integration module import failed: {e}")
    COLANG_INTEGRATION_AVAILABLE = False

# P2-026: Import Z3 formal verification module
try:
    from pipeline.langgraph.z3_verification import (
        # Classes
        Z3Context,
        Z3Verifier,
        InvariantSpec,
        VerificationResult,
        # Enums
        VerificationStatus,
        # Registry
        INVARIANT_SPECS,
        # Functions
        get_z3_verifier,
        verify_invariant,
        verify_gate_invariants,
        verify_critical_invariants,
        is_z3_available,
        get_invariant_specs,
        # Constants
        Z3_AVAILABLE,
    )
    __all__.extend([
        "Z3Context",
        "Z3Verifier",
        "InvariantSpec",
        "VerificationResult",
        "VerificationStatus",
        "INVARIANT_SPECS",
        "get_z3_verifier",
        "verify_invariant",
        "verify_gate_invariants",
        "verify_critical_invariants",
        "is_z3_available",
        "get_invariant_specs",
        "Z3_AVAILABLE",
    ])
except ImportError as e:
    logger.warning(f"z3_verification module import failed: {e}")
    Z3_AVAILABLE = False

# P3-001: Import Hamilton pipelines module (DAG-based transformations)
try:
    from pipeline.langgraph.hamilton_pipelines import (
        # Classes
        HamiltonPipeline,
        HamiltonPipelineManager,
        PipelineNode,
        PipelineConfig,
        # Enums
        PipelineStatus,
        TransformationType,
        # Registry
        PIPELINE_MODULES,
        # Functions
        get_pipeline_manager,
        run_pipeline,
        list_pipelines,
        is_hamilton_available,
        run_claim_verification,
        run_gate_validation,
        run_evidence_collection,
        # Constants
        HAMILTON_AVAILABLE,
    )
    __all__.extend([
        "HamiltonPipeline",
        "HamiltonPipelineManager",
        "PipelineNode",
        "PipelineConfig",
        "PipelineStatus",
        "TransformationType",
        "PIPELINE_MODULES",
        "get_pipeline_manager",
        "run_pipeline",
        "list_pipelines",
        "is_hamilton_available",
        "run_claim_verification",
        "run_gate_validation",
        "run_evidence_collection",
        "HAMILTON_AVAILABLE",
    ])
except ImportError as e:
    logger.warning(f"hamilton_pipelines module import failed: {e}")
    HAMILTON_AVAILABLE = False

# Security module imports (Batch 6 - Security Stacks)
try:
    from pipeline.security import (
        # NeMo Enhanced
        NeMoEnhancedRails,
        JailbreakDetector,
        ContentFilter,
        DialogManager,
        get_nemo_enhanced,
        detect_jailbreak,
        filter_content,
        manage_dialog,
        is_nemo_enhanced_available,
        JailbreakResult,
        ContentFilterResult,
        DialogState,
        NEMO_ENHANCED_AVAILABLE,
    )
    __all__.extend([
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
    ])
except ImportError as e:
    logger.warning(f"NeMo enhanced security module import failed: {e}")
    NEMO_ENHANCED_AVAILABLE = False

try:
    from pipeline.security import (
        # LLM Guard Integration
        LLMGuardIntegration,
        InputSanitizer,
        OutputValidator,
        PIIDetector,
        ToxicityFilter,
        SecurityOrchestrator,
        get_llm_guard_integration,
        get_security_orchestrator,
        sanitize_input,
        validate_output,
        detect_pii,
        filter_toxicity,
        run_security_checks,
        is_llm_guard_integration_available,
        SanitizationResult,
        ValidationResult,
        PIIDetectionResult,
        ToxicityFilterResult,
        SecurityCheckResult,
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
    logger.warning(f"LLM Guard integration module import failed: {e}")
    LLM_GUARD_INTEGRATION_AVAILABLE = False

try:
    from pipeline.security import (
        # Security Gate Integration
        SecurityGate,
        SecurityGateRunner,
        get_security_gate_runner,
        run_security_gate,
        validate_gate_security,
        SecurityGateResult,
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
    logger.warning(f"Security gate integration module import failed: {e}")
    SECURITY_GATE_AVAILABLE = False

# P2-B7/B8: Import Reflexion module (self-reflection loops)
try:
    from pipeline.langgraph.reflexion import (
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
    __all__.extend([
        "ReflexionEngine",
        "SelfReflectionLoop",
        "ErrorCorrector",
        "FeedbackLearner",
        "ReflectionResult",
        "ReflectionIteration",
        "ActionPlan",
        "ErrorAnalysis",
        "CorrectionResult",
        "LearningEntry",
        "FeedbackRecord",
        "ReflexionMetrics",
        "ReflectionQuality",
        "ErrorSeverity",
        "LearningType",
        "get_reflexion_engine",
        "run_self_reflection",
        "analyze_and_correct_error",
        "learn_from_feedback",
        "get_reflexion_metrics",
        "REFLEXION_AVAILABLE",
    ])
except ImportError as e:
    logger.warning(f"Reflexion module import failed: {e}")
    REFLEXION_AVAILABLE = False

# P2-B8: Import streaming and human-in-the-loop module
try:
    from pipeline.langgraph.streaming import (
        # Classes
        StreamingWorkflow,
        StreamHandler,
        EnhancedCheckpointer,
        HumanLoop,
        BreakpointManager,
        # Result types
        StreamEvent,
        CheckpointVersion,
        ApprovalRequest,
        HumanDecision,
        BreakpointConfig,
        StreamingResult,
        # Enums
        StreamEventType,
        ApprovalStatus,
        BreakpointType,
        CheckpointStrategy,
        # Functions
        get_streaming_workflow,
        stream_workflow_execution,
        create_approval_request,
        resume_after_approval,
        # Constants
        STREAMING_AVAILABLE,
    )
    __all__.extend([
        "StreamingWorkflow",
        "StreamHandler",
        "EnhancedCheckpointer",
        "HumanLoop",
        "BreakpointManager",
        "StreamEvent",
        "CheckpointVersion",
        "ApprovalRequest",
        "HumanDecision",
        "BreakpointConfig",
        "StreamingResult",
        "StreamEventType",
        "ApprovalStatus",
        "BreakpointType",
        "CheckpointStrategy",
        "get_streaming_workflow",
        "stream_workflow_execution",
        "create_approval_request",
        "resume_after_approval",
        "STREAMING_AVAILABLE",
    ])
except ImportError as e:
    logger.warning(f"Streaming module import failed: {e}")
    STREAMING_AVAILABLE = False

# P2-B8: Import Pydantic validators module
try:
    from pipeline.langgraph.pydantic_validators import (
        # Classes
        StateValidator,
        StateSerializer,
        # Result types
        ValidationResult,
        ValidationErrorDetail,
        SerializationResult,
        DeserializationResult,
        # Validator functions
        validate_sprint_id,
        validate_gate_id,
        validate_phase,
        validate_confidence_score,
        validate_timestamp,
        validate_url,
        validate_claim_verdict,
        validate_error_severity,
        # Convenience functions
        validate_pipeline_state,
        serialize_state,
        deserialize_state,
        # Constants
        PYDANTIC_VALIDATORS_AVAILABLE,
        PYDANTIC_V2_AVAILABLE,
    )
    __all__.extend([
        "StateValidator",
        "StateSerializer",
        "ValidationResult",
        "ValidationErrorDetail",
        "SerializationResult",
        "DeserializationResult",
        "validate_sprint_id",
        "validate_gate_id",
        "validate_phase",
        "validate_confidence_score",
        "validate_timestamp",
        "validate_url",
        "validate_claim_verdict",
        "validate_error_severity",
        "validate_pipeline_state",
        "serialize_state",
        "deserialize_state",
        "PYDANTIC_VALIDATORS_AVAILABLE",
        "PYDANTIC_V2_AVAILABLE",
    ])
except ImportError as e:
    logger.warning(f"Pydantic validators module import failed: {e}")
    PYDANTIC_VALIDATORS_AVAILABLE = False

# P2-B9: Import Corrective Enforcement module
try:
    from pipeline.langgraph.corrective_enforcement import (
        # Enums
        CorrectionAction,
        EnforcementType,
        EscalationLevel,
        # Data classes
        EnforcementResult,
        CorrectionContext,
        CorrectionStrategy,
        # Classes
        CorrectionEngine,
        EnforcementLayer,
        # Accessors
        get_enforcement_layer,
        get_correction_engine,
        # Convenience
        enforce_and_correct,
        create_correction_context,
    )
    CORRECTIVE_ENFORCEMENT_AVAILABLE = True
    __all__.extend([
        "CorrectionAction",
        "EnforcementType",
        "EscalationLevel",
        "EnforcementResult",
        "CorrectionContext",
        "CorrectionStrategy",
        "CorrectionEngine",
        "EnforcementLayer",
        "get_enforcement_layer",
        "get_correction_engine",
        "enforce_and_correct",
        "create_correction_context",
        "CORRECTIVE_ENFORCEMENT_AVAILABLE",
    ])
except ImportError as e:
    logger.warning(f"Corrective enforcement module import failed: {e}")
    CORRECTIVE_ENFORCEMENT_AVAILABLE = False

# P2-B9: Import Enforcement Integration module
try:
    from pipeline.langgraph.enforcement_integration import (
        # Decorator
        with_enforcement,
        # Classes
        EnforcedWorkflowNodes,
        # Functions
        build_enforced_workflow,
    )
    ENFORCEMENT_INTEGRATION_AVAILABLE = True
    __all__.extend([
        "with_enforcement",
        "EnforcedWorkflowNodes",
        "build_enforced_workflow",
        "ENFORCEMENT_INTEGRATION_AVAILABLE",
    ])
except ImportError as e:
    logger.warning(f"Enforcement integration module import failed: {e}")
    ENFORCEMENT_INTEGRATION_AVAILABLE = False

# P2-B9: Import Guardrail Violation Handler module (BLOCK AND REWORK)
try:
    from pipeline.langgraph.guardrail_violation_handler import (
        # Classes
        GuardrailViolationHandler,
        ViolationBuilder,
        # Enums
        ViolationType,
        ViolationSeverity,
        TaskInvalidationReason,
        ReworkPriority,
        # Types
        ViolationDetails,
        ViolationReport,
        TaskInvalidation,
        ReworkTask,
        # Functions
        get_violation_handler,
        create_security_violation,
        create_invariant_violation,
        create_validation_violation,
        create_gate_violation,
    )
    GUARDRAIL_VIOLATION_HANDLER_AVAILABLE = True
    __all__.extend([
        "GuardrailViolationHandler",
        "ViolationBuilder",
        "ViolationType",
        "ViolationSeverity",
        "TaskInvalidationReason",
        "ReworkPriority",
        "ViolationDetails",
        "ViolationReport",
        "TaskInvalidation",
        "ReworkTask",
        "get_violation_handler",
        "create_security_violation",
        "create_invariant_violation",
        "create_validation_violation",
        "create_gate_violation",
        "GUARDRAIL_VIOLATION_HANDLER_AVAILABLE",
    ])
except ImportError as e:
    logger.warning(f"Guardrail violation handler module import failed: {e}")
    GUARDRAIL_VIOLATION_HANDLER_AVAILABLE = False

# P2-B9: Import Enforcement Metrics module (monitoring and observability)
try:
    from pipeline.langgraph.enforcement_metrics import (
        # Classes
        EnforcementMetrics,
        MetricEntry,
        LatencyBucket,
        LatencyHistogram,
        MetricsSummary,
        # Enums
        ViolationType as MetricsViolationType,
        InvariantCode as MetricsInvariantCode,
        CheckType,
        # Singleton accessor
        get_metrics,
        reset_metrics,
        # Helper functions
        record_violation,
        record_invariant_check,
        record_trust_denial,
        record_rework_attempt,
        record_enforcement_latency,
        get_metrics_summary,
        get_prometheus_metrics,
        # Decorators
        track_enforcement_latency,
        measure_enforcement_latency,
    )
    ENFORCEMENT_METRICS_AVAILABLE = True
    __all__.extend([
        "EnforcementMetrics",
        "MetricEntry",
        "LatencyBucket",
        "LatencyHistogram",
        "MetricsSummary",
        "MetricsViolationType",
        "MetricsInvariantCode",
        "CheckType",
        "get_metrics",
        "reset_metrics",
        "record_violation",
        "record_invariant_check",
        "record_trust_denial",
        "record_rework_attempt",
        "record_enforcement_latency",
        "get_metrics_summary",
        "get_prometheus_metrics",
        "track_enforcement_latency",
        "measure_enforcement_latency",
        "ENFORCEMENT_METRICS_AVAILABLE",
    ])
except ImportError as e:
    logger.warning(f"Enforcement metrics module import failed: {e}")
    ENFORCEMENT_METRICS_AVAILABLE = False

# Partial Stacks Integration (Active RAG, BoT, RAGAS, Phoenix, DeepEval)
try:
    from pipeline.langgraph.partial_stacks_integration import (
        # Availability flags
        ACTIVE_RAG_AVAILABLE,
        BOT_AVAILABLE,
        RAGAS_AVAILABLE,
        PHOENIX_AVAILABLE,
        DEEPEVAL_AVAILABLE,
        INSTRUMENTATION_AVAILABLE,
        # Health check
        StackHealthResult,
        PartialStacksHealthReport,
        health_check_active_rag,
        health_check_bot,
        health_check_ragas,
        health_check_phoenix,
        health_check_deepeval,
        health_check_all,
        # Integration
        PartialStacksIntegration,
        get_partial_stacks_integration,
    )
    PARTIAL_STACKS_INTEGRATION_AVAILABLE = True
    __all__.extend([
        "ACTIVE_RAG_AVAILABLE",
        "BOT_AVAILABLE",
        "RAGAS_AVAILABLE",
        "PHOENIX_AVAILABLE",
        "DEEPEVAL_AVAILABLE",
        "INSTRUMENTATION_AVAILABLE",
        "StackHealthResult",
        "PartialStacksHealthReport",
        "health_check_active_rag",
        "health_check_bot",
        "health_check_ragas",
        "health_check_phoenix",
        "health_check_deepeval",
        "health_check_all",
        "PartialStacksIntegration",
        "get_partial_stacks_integration",
        "PARTIAL_STACKS_INTEGRATION_AVAILABLE",
    ])
except ImportError as e:
    logger.warning(f"Partial stacks integration module import failed: {e}")
    PARTIAL_STACKS_INTEGRATION_AVAILABLE = False
    __all__.append("PARTIAL_STACKS_INTEGRATION_AVAILABLE")