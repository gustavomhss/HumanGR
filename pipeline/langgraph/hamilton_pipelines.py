"""Hamilton DAG Pipelines for Pipeline V2.

P3-001: Hamilton-based data transformation pipelines.

Hamilton provides a declarative way to define data pipelines as
directed acyclic graphs (DAGs). Each function represents a node,
and dependencies are inferred from function signatures.

Key Features:
1. Function-based DAG definition
2. Automatic dependency resolution
3. Data lineage tracking
4. Reproducible transformations
5. Type safety with decorators

Author: Pipeline Autonomo Team
Version: 1.0.0 (2026-01-16)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypedDict

import pandas as pd

logger = logging.getLogger(__name__)

# =============================================================================
# HAMILTON AVAILABILITY CHECK
# =============================================================================

# 2026-01-25: Hamilton DISABLED permanently.
#
# Reason: Hamilton's rigid type validation causes more problems than it solves
# for our use case (simple sequential pipelines). The fallback execution does
# exactly the same thing without the type mismatch errors.
#
# The fallback:
# - Executes functions in topological order (same as Hamilton)
# - Uses Python's duck typing (no rigid type validation)
# - Works reliably without configuration issues
#
# Hamilton would be valuable for complex data science pipelines with parallel
# branches and caching needs, but that's not our use case.
HAMILTON_AVAILABLE = False


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

class PipelineStatus(str, Enum):
    """Status of pipeline execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TransformationType(str, Enum):
    """Types of data transformations."""
    EXTRACT = "extract"
    TRANSFORM = "transform"
    LOAD = "load"
    VALIDATE = "validate"
    AGGREGATE = "aggregate"
    ENRICH = "enrich"
    FILTER = "filter"


class PipelineResult(TypedDict):
    """Result of pipeline execution."""
    pipeline_name: str
    status: str
    outputs: Dict[str, Any]
    execution_time_ms: float
    nodes_executed: int
    errors: List[str]
    executed_at: str


@dataclass
class PipelineNode:
    """Definition of a pipeline node."""
    name: str
    transform_type: TransformationType
    function: Callable
    dependencies: List[str] = field(default_factory=list)
    description: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution."""
    name: str
    description: str = ""
    timeout_seconds: int = 300
    enable_caching: bool = True
    enable_validation: bool = True
    parallel_execution: bool = True


# =============================================================================
# CLAIM VERIFICATION PIPELINE NODES
# =============================================================================

# Note: These functions would be in separate modules for real Hamilton usage
# Here we define them as a reference implementation


def raw_claims(claims_data: Any) -> pd.DataFrame:
    """Extract raw claims from input data.

    This is the entry point for claim verification pipeline.
    Accepts Dict, List[Dict], or DataFrame.

    2026-01-25 FIX: Changed type hint from Dict[str, Any] to Any to allow
    Hamilton to accept list inputs (which are common in sprint workflows).
    """
    if isinstance(claims_data, pd.DataFrame):
        return claims_data
    elif isinstance(claims_data, list):
        return pd.DataFrame(claims_data)
    elif isinstance(claims_data, dict):
        return pd.DataFrame([claims_data])
    else:
        raise ValueError(f"Unsupported claims_data type: {type(claims_data)}")


def normalized_claims(raw_claims: pd.DataFrame) -> pd.DataFrame:
    """Normalize claims for processing.

    Applies text normalization, removes duplicates, validates schema.
    """
    df = raw_claims.copy()

    # Ensure required columns
    required_cols = ["claim_id", "text", "source"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    # Normalize text
    if "text" in df.columns:
        df["text"] = df["text"].str.strip().str.lower()

    # Remove duplicates
    df = df.drop_duplicates(subset=["text"], keep="first")

    return df


def enriched_claims(normalized_claims: pd.DataFrame) -> pd.DataFrame:
    """Enrich claims with additional metadata.

    Adds entity extraction, sentiment, complexity scores.
    """
    df = normalized_claims.copy()

    # Add enrichment columns
    df["entity_count"] = df["text"].str.count(r"\b[A-Z][a-z]+\b") if "text" in df.columns else 0
    df["word_count"] = df["text"].str.split().str.len() if "text" in df.columns else 0
    df["complexity_score"] = df["word_count"] / 10  # Simple heuristic

    return df


def verified_claims(enriched_claims: pd.DataFrame) -> pd.DataFrame:
    """Verify claims against knowledge sources.

    This is a placeholder - real implementation would query KG.
    """
    df = enriched_claims.copy()

    # Add verification placeholders
    df["verification_status"] = "pending"
    df["confidence_score"] = 0.0
    df["evidence_count"] = 0
    df["verified_at"] = None

    return df


def claim_report(verified_claims: pd.DataFrame) -> Dict[str, Any]:
    """Generate claim verification report.

    Aggregates results into summary statistics.
    """
    total = len(verified_claims)
    verified = len(verified_claims[verified_claims["verification_status"] == "verified"])
    pending = len(verified_claims[verified_claims["verification_status"] == "pending"])
    failed = len(verified_claims[verified_claims["verification_status"] == "failed"])

    return {
        "total_claims": total,
        "verified_claims": verified,
        "pending_claims": pending,
        "failed_claims": failed,
        "verification_rate": verified / total if total > 0 else 0,
        "average_confidence": verified_claims["confidence_score"].mean() if total > 0 else 0,
    }


# =============================================================================
# GATE VALIDATION PIPELINE NODES
# =============================================================================

def gate_inputs(gate_config: Dict[str, Any]) -> pd.DataFrame:
    """Extract gate validation inputs.

    Converts gate configuration to structured format.
    """
    gates = gate_config.get("gates", [])
    return pd.DataFrame(gates)


def gate_requirements(gate_inputs: pd.DataFrame) -> pd.DataFrame:
    """Map gate requirements.

    Adds threshold and criteria columns.
    """
    df = gate_inputs.copy()

    # Default thresholds by gate type
    thresholds = {
        "G0": 1.0,  # Spec compliance
        "G1": 0.9,  # Quality
        "G2": 1.0,  # Security
        "G3": 0.95, # Integration
        "G4": 1.0,  # Blindagem
        "G5": 0.9,  # Performance
        "G6": 0.95, # Observability
        "G7": 1.0,  # Deliverable
        "G8": 0.7,  # Mutation
    }

    if "gate_id" in df.columns:
        df["threshold"] = df["gate_id"].map(thresholds).fillna(0.9)
    else:
        df["threshold"] = 0.9

    return df


def gate_results(gate_requirements: pd.DataFrame) -> pd.DataFrame:
    """Execute gate validations.

    Placeholder - real implementation would run actual validations.
    """
    df = gate_requirements.copy()

    # Add result columns
    df["score"] = 0.0
    df["status"] = "pending"
    df["evidence"] = None
    df["validated_at"] = None

    return df


def gate_summary(gate_results: pd.DataFrame) -> Dict[str, Any]:
    """Generate gate validation summary."""
    total = len(gate_results)
    passed = len(gate_results[gate_results["status"] == "passed"])
    failed = len(gate_results[gate_results["status"] == "failed"])

    return {
        "total_gates": total,
        "passed_gates": passed,
        "failed_gates": failed,
        "pass_rate": passed / total if total > 0 else 0,
        "all_passed": passed == total,
    }


# =============================================================================
# EVIDENCE COLLECTION PIPELINE NODES
# =============================================================================

def evidence_sources(search_config: Dict[str, Any]) -> pd.DataFrame:
    """Identify evidence sources to search.

    Returns list of sources to query.
    """
    sources = search_config.get("sources", [
        {"name": "knowledge_graph", "type": "kg"},
        {"name": "qdrant", "type": "vector"},
        {"name": "web", "type": "search"},
    ])
    return pd.DataFrame(sources)


def source_queries(
    evidence_sources: pd.DataFrame,
    search_query: str
) -> pd.DataFrame:
    """Generate queries for each source.

    Adapts query format for each source type.
    """
    df = evidence_sources.copy()
    df["query"] = search_query
    df["adapted_query"] = df.apply(
        lambda row: adapt_query(row["type"], search_query),
        axis=1
    )
    return df


def adapt_query(source_type: str, query: str) -> str:
    """Adapt query for source type."""
    if source_type == "kg":
        # Cypher-like format
        return f"MATCH (n) WHERE n.text CONTAINS '{query}' RETURN n"
    elif source_type == "vector":
        # Vector search format
        return query  # Will be embedded
    else:
        return query


def evidence_results(source_queries: pd.DataFrame) -> pd.DataFrame:
    """Execute evidence queries.

    Placeholder - real implementation would query actual sources.
    """
    df = source_queries.copy()
    df["results"] = None
    df["result_count"] = 0
    df["relevance_score"] = 0.0
    df["queried_at"] = datetime.now(timezone.utc).isoformat()
    return df


def evidence_report(evidence_results: pd.DataFrame) -> Dict[str, Any]:
    """Generate evidence collection report."""
    total_results = evidence_results["result_count"].sum()
    avg_relevance = evidence_results["relevance_score"].mean()

    return {
        "sources_queried": len(evidence_results),
        "total_evidence": total_results,
        "average_relevance": avg_relevance,
        "results_by_source": evidence_results.set_index("name")["result_count"].to_dict(),
    }


# =============================================================================
# PIPELINE REGISTRY
# =============================================================================

PIPELINE_MODULES = {
    "claim_verification": {
        "nodes": [
            PipelineNode("raw_claims", TransformationType.EXTRACT, raw_claims),
            PipelineNode("normalized_claims", TransformationType.TRANSFORM, normalized_claims, ["raw_claims"]),
            PipelineNode("enriched_claims", TransformationType.ENRICH, enriched_claims, ["normalized_claims"]),
            PipelineNode("verified_claims", TransformationType.VALIDATE, verified_claims, ["enriched_claims"]),
            PipelineNode("claim_report", TransformationType.AGGREGATE, claim_report, ["verified_claims"]),
        ],
        "config": PipelineConfig(
            name="claim_verification",
            description="Verify claims through knowledge sources",
        ),
    },
    "gate_validation": {
        "nodes": [
            PipelineNode("gate_inputs", TransformationType.EXTRACT, gate_inputs),
            PipelineNode("gate_requirements", TransformationType.TRANSFORM, gate_requirements, ["gate_inputs"]),
            PipelineNode("gate_results", TransformationType.VALIDATE, gate_results, ["gate_requirements"]),
            PipelineNode("gate_summary", TransformationType.AGGREGATE, gate_summary, ["gate_results"]),
        ],
        "config": PipelineConfig(
            name="gate_validation",
            description="Validate gates for sprint completion",
        ),
    },
    "evidence_collection": {
        "nodes": [
            PipelineNode("evidence_sources", TransformationType.EXTRACT, evidence_sources),
            PipelineNode("source_queries", TransformationType.TRANSFORM, source_queries, ["evidence_sources"]),
            PipelineNode("evidence_results", TransformationType.VALIDATE, evidence_results, ["source_queries"]),
            PipelineNode("evidence_report", TransformationType.AGGREGATE, evidence_report, ["evidence_results"]),
        ],
        "config": PipelineConfig(
            name="evidence_collection",
            description="Collect evidence from multiple sources",
        ),
    },
}


# =============================================================================
# HAMILTON DRIVER WRAPPER
# =============================================================================

class HamiltonPipeline:
    """Wrapper for Hamilton DAG execution.

    Provides a high-level interface for defining and executing
    Hamilton-based data transformation pipelines.

    Usage:
        pipeline = HamiltonPipeline("claim_verification")
        result = pipeline.execute({"claims_data": claims})
    """

    def __init__(self, pipeline_name: str):
        """Initialize pipeline.

        Args:
            pipeline_name: Name of pipeline from registry
        """
        if pipeline_name not in PIPELINE_MODULES:
            raise ValueError(f"Unknown pipeline: {pipeline_name}")

        self.name = pipeline_name
        self._module = PIPELINE_MODULES[pipeline_name]
        self._config = self._module["config"]
        self._nodes = self._module["nodes"]
        self._driver: Optional[Any] = None

        if HAMILTON_AVAILABLE:
            self._init_driver()

    def _init_driver(self) -> None:
        """Initialize Hamilton driver (DISABLED - using fallback instead)."""
        # Hamilton disabled - see comment at top of file
        self._driver = None

    @property
    def available(self) -> bool:
        """Check if pipeline is available (always True - fallback works)."""
        return True  # Fallback execution always available

    def execute(
        self,
        inputs: Dict[str, Any],
        final_vars: Optional[List[str]] = None
    ) -> PipelineResult:
        """Execute the pipeline.

        Args:
            inputs: Input data for the pipeline
            final_vars: Specific outputs to compute (None = all)

        Returns:
            PipelineResult with outputs and metadata
        """
        import time
        start_time = time.time()
        errors: List[str] = []

        if not self.available:
            # Fallback execution without Hamilton
            return self._execute_fallback(inputs, final_vars, start_time)

        # Hamilton disabled - always use fallback (simpler, works)
        return self._execute_fallback(inputs, final_vars, start_time)

    def _execute_fallback(
        self,
        inputs: Dict[str, Any],
        final_vars: Optional[List[str]],
        start_time: float
    ) -> PipelineResult:
        """Execute pipeline without Hamilton (fallback mode).

        Manually executes nodes in dependency order.
        """
        import time

        errors: List[str] = []
        outputs: Dict[str, Any] = {}
        nodes_executed = 0

        try:
            # Build execution order (topological sort)
            execution_order = self._get_execution_order()

            # Execute each node
            context = inputs.copy()
            for node in execution_order:
                try:
                    # Get node inputs from context
                    node_inputs = self._get_node_inputs(node, context)

                    # Execute node
                    result = node.function(**node_inputs)

                    # Store result
                    context[node.name] = result
                    outputs[node.name] = result
                    nodes_executed += 1

                except Exception as e:
                    logger.error(f"Node {node.name} failed: {e}")
                    errors.append(f"{node.name}: {str(e)}")
                    break

            # Filter to final vars if specified
            if final_vars:
                outputs = {k: v for k, v in outputs.items() if k in final_vars}

            execution_time_ms = (time.time() - start_time) * 1000
            status = PipelineStatus.COMPLETED if not errors else PipelineStatus.FAILED

            return PipelineResult(
                pipeline_name=self.name,
                status=status.value,
                outputs=outputs,
                execution_time_ms=execution_time_ms,
                nodes_executed=nodes_executed,
                errors=errors,
                executed_at=datetime.now(timezone.utc).isoformat(),
            )

        except Exception as e:
            logger.error(f"Fallback execution failed: {e}")
            execution_time_ms = (time.time() - start_time) * 1000
            return PipelineResult(
                pipeline_name=self.name,
                status=PipelineStatus.FAILED.value,
                outputs={},
                execution_time_ms=execution_time_ms,
                nodes_executed=nodes_executed,
                errors=[str(e)],
                executed_at=datetime.now(timezone.utc).isoformat(),
            )

    def _get_execution_order(self) -> List[PipelineNode]:
        """Get topological sort of nodes."""
        # Simple topological sort
        visited = set()
        order = []

        def visit(node: PipelineNode):
            if node.name in visited:
                return
            visited.add(node.name)

            # Visit dependencies first
            for dep_name in node.dependencies:
                dep_node = next((n for n in self._nodes if n.name == dep_name), None)
                if dep_node:
                    visit(dep_node)

            order.append(node)

        for node in self._nodes:
            visit(node)

        return order

    def _get_node_inputs(
        self,
        node: PipelineNode,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get inputs for a node from context."""
        import inspect
        sig = inspect.signature(node.function)

        inputs = {}
        for param_name in sig.parameters:
            if param_name in context:
                inputs[param_name] = context[param_name]

        return inputs

    def visualize(self) -> str:
        """Generate ASCII visualization of the pipeline DAG."""
        lines = [f"Pipeline: {self.name}", "=" * 40]

        for node in self._nodes:
            deps = " <- " + ", ".join(node.dependencies) if node.dependencies else ""
            lines.append(f"  [{node.transform_type.value}] {node.name}{deps}")

        return "\n".join(lines)

    def get_info(self) -> Dict[str, Any]:
        """Get pipeline information."""
        return {
            "name": self.name,
            "description": self._config.description,
            "nodes": [n.name for n in self._nodes],
            "node_count": len(self._nodes),
            "execution_mode": "fallback",  # Hamilton disabled
        }


# =============================================================================
# PIPELINE MANAGER
# =============================================================================

class HamiltonPipelineManager:
    """Manager for Hamilton pipelines.

    Provides centralized access to pipeline creation and execution.

    Usage:
        manager = HamiltonPipelineManager()
        result = manager.run_pipeline("claim_verification", inputs)
    """

    _instance: Optional["HamiltonPipelineManager"] = None
    _lock: threading.Lock = threading.Lock()
    _pipelines: Dict[str, HamiltonPipeline] = {}

    def __new__(cls) -> "HamiltonPipelineManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def get_pipeline(self, name: str) -> HamiltonPipeline:
        """Get or create a pipeline by name."""
        if name not in self._pipelines:
            self._pipelines[name] = HamiltonPipeline(name)
        return self._pipelines[name]

    def run_pipeline(
        self,
        name: str,
        inputs: Dict[str, Any],
        final_vars: Optional[List[str]] = None
    ) -> PipelineResult:
        """Run a pipeline by name.

        Args:
            name: Pipeline name
            inputs: Input data
            final_vars: Specific outputs to compute

        Returns:
            PipelineResult
        """
        pipeline = self.get_pipeline(name)
        return pipeline.execute(inputs, final_vars)

    def list_pipelines(self) -> List[str]:
        """List available pipelines."""
        return list(PIPELINE_MODULES.keys())

    def get_pipeline_info(self, name: str) -> Dict[str, Any]:
        """Get information about a pipeline."""
        return self.get_pipeline(name).get_info()

    def visualize_pipeline(self, name: str) -> str:
        """Get ASCII visualization of a pipeline."""
        return self.get_pipeline(name).visualize()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_manager: Optional[HamiltonPipelineManager] = None


def get_pipeline_manager() -> HamiltonPipelineManager:
    """Get singleton pipeline manager."""
    global _manager
    if _manager is None:
        _manager = HamiltonPipelineManager()
    return _manager


def run_pipeline(
    name: str,
    inputs: Dict[str, Any],
    final_vars: Optional[List[str]] = None
) -> PipelineResult:
    """Run a pipeline by name."""
    return get_pipeline_manager().run_pipeline(name, inputs, final_vars)


def list_pipelines() -> List[str]:
    """List available pipelines."""
    return get_pipeline_manager().list_pipelines()


def is_hamilton_available() -> bool:
    """Check if Hamilton is available (always False - using fallback)."""
    return False  # Hamilton disabled, using simpler fallback execution


def run_claim_verification(claims_data: Any) -> PipelineResult:
    """Run claim verification pipeline."""
    return run_pipeline("claim_verification", {"claims_data": claims_data})


def run_gate_validation(gate_config: Dict[str, Any]) -> PipelineResult:
    """Run gate validation pipeline."""
    return run_pipeline("gate_validation", {"gate_config": gate_config})


def run_evidence_collection(
    search_config: Dict[str, Any],
    search_query: str
) -> PipelineResult:
    """Run evidence collection pipeline."""
    return run_pipeline(
        "evidence_collection",
        {"search_config": search_config, "search_query": search_query}
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Classes
    "HamiltonPipeline",
    "HamiltonPipelineManager",
    "PipelineNode",
    "PipelineConfig",
    # Enums
    "PipelineStatus",
    "TransformationType",
    # TypedDicts
    "PipelineResult",
    # Registry
    "PIPELINE_MODULES",
    # Functions
    "get_pipeline_manager",
    "run_pipeline",
    "list_pipelines",
    "is_hamilton_available",
    "run_claim_verification",
    "run_gate_validation",
    "run_evidence_collection",
    # Constants
    "HAMILTON_AVAILABLE",
]
