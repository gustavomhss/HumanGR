"""Spec Kit Loader - Carrega Context Packs, Cerebros e Constitution.

Este módulo fornece funcionalidades para carregar e parsear os artefatos
do Spec Kit que definem como o Pipeline deve operar.

Componentes:
    - Context Packs: Definições de sprints (S00-S62)
    - Cerebros: Identidades e comportamentos dos agentes (YAML modular ou Markdown)
    - Constitution: Artigos invioláveis do pipeline

O loader agora prioriza o formato YAML modular (6 arquivos por Cerebro)
via CerebroYAMLLoader, com fallback para o formato Markdown legado.

Author: Pipeline Autonomo Team
Version: 2.0.0 (2026-01-11) - INTENT MANIFEST v2.0 support
"""

from __future__ import annotations

import logging
import threading
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import yaml

if TYPE_CHECKING:
    from pipeline.cerebro_yaml_loader import CerebroYAMLLoader

logger = logging.getLogger(__name__)

# OPT-12-006: Pre-compiled regex patterns for YAML block extraction
# Compiled at module load time for reuse across all SpecKitLoader instances
_YAML_BLOCK_PATTERN = re.compile(r"```ya?ml\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_SECTION_PATTERN_CACHE: dict[str, re.Pattern[str]] = {}


# =============================================================================
# GAP-002 FIX: PERFORMANCE METRICS
# =============================================================================

from dataclasses import dataclass as metrics_dataclass


@metrics_dataclass
class SpecKitMetrics:
    """Performance metrics for SpecKitLoader operations.

    GAP-002 FIX: Added metrics tracking for observability.
    AUDIT FIX: Added thread safety for concurrent access.
    """
    # Cache metrics
    context_pack_cache_hits: int = 0
    context_pack_cache_misses: int = 0
    cerebro_cache_hits: int = 0
    cerebro_cache_misses: int = 0
    constitution_cache_hits: int = 0
    constitution_cache_misses: int = 0

    # Timing metrics (in milliseconds)
    total_load_time_ms: float = 0.0
    context_pack_load_times: list = None
    cerebro_load_times: list = None

    # Error metrics
    parse_errors: int = 0
    file_not_found_errors: int = 0

    # Thread safety lock (not included in dataclass fields)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def __post_init__(self):
        if self.context_pack_load_times is None:
            self.context_pack_load_times = []
        if self.cerebro_load_times is None:
            self.cerebro_load_times = []

    # Thread-safe increment methods
    def increment_context_pack_cache_hit(self) -> None:
        """Thread-safe increment for context pack cache hits."""
        with self._lock:
            self.context_pack_cache_hits += 1

    def increment_context_pack_cache_miss(self) -> None:
        """Thread-safe increment for context pack cache misses."""
        with self._lock:
            self.context_pack_cache_misses += 1

    def increment_cerebro_cache_hit(self) -> None:
        """Thread-safe increment for cerebro cache hits."""
        with self._lock:
            self.cerebro_cache_hits += 1

    def increment_cerebro_cache_miss(self) -> None:
        """Thread-safe increment for cerebro cache misses."""
        with self._lock:
            self.cerebro_cache_misses += 1

    def increment_constitution_cache_hit(self) -> None:
        """Thread-safe increment for constitution cache hits."""
        with self._lock:
            self.constitution_cache_hits += 1

    def increment_constitution_cache_miss(self) -> None:
        """Thread-safe increment for constitution cache misses."""
        with self._lock:
            self.constitution_cache_misses += 1

    def increment_parse_errors(self) -> None:
        """Thread-safe increment for parse errors."""
        with self._lock:
            self.parse_errors += 1

    def increment_file_not_found_errors(self) -> None:
        """Thread-safe increment for file not found errors."""
        with self._lock:
            self.file_not_found_errors += 1

    def add_context_pack_load_time(self, time_ms: float) -> None:
        """Thread-safe append for context pack load times."""
        with self._lock:
            self.context_pack_load_times.append(time_ms)

    def add_cerebro_load_time(self, time_ms: float) -> None:
        """Thread-safe append for cerebro load times."""
        with self._lock:
            self.cerebro_load_times.append(time_ms)

    def add_total_load_time(self, time_ms: float) -> None:
        """Thread-safe add for total load time."""
        with self._lock:
            self.total_load_time_ms += time_ms

    @property
    def context_pack_cache_hit_ratio(self) -> float:
        """Calculate cache hit ratio for context packs."""
        total = self.context_pack_cache_hits + self.context_pack_cache_misses
        return self.context_pack_cache_hits / total if total > 0 else 0.0

    @property
    def cerebro_cache_hit_ratio(self) -> float:
        """Calculate cache hit ratio for cerebros."""
        total = self.cerebro_cache_hits + self.cerebro_cache_misses
        return self.cerebro_cache_hits / total if total > 0 else 0.0

    @property
    def avg_context_pack_load_time_ms(self) -> float:
        """Average context pack load time in milliseconds."""
        if not self.context_pack_load_times:
            return 0.0
        return sum(self.context_pack_load_times) / len(self.context_pack_load_times)

    @property
    def avg_cerebro_load_time_ms(self) -> float:
        """Average cerebro load time in milliseconds."""
        if not self.cerebro_load_times:
            return 0.0
        return sum(self.cerebro_load_times) / len(self.cerebro_load_times)

    def to_dict(self) -> dict:
        """Convert metrics to dictionary for reporting."""
        return {
            "cache": {
                "context_pack_hit_ratio": round(self.context_pack_cache_hit_ratio, 3),
                "cerebro_hit_ratio": round(self.cerebro_cache_hit_ratio, 3),
                "context_pack_hits": self.context_pack_cache_hits,
                "context_pack_misses": self.context_pack_cache_misses,
                "cerebro_hits": self.cerebro_cache_hits,
                "cerebro_misses": self.cerebro_cache_misses,
            },
            "timing": {
                "total_load_time_ms": round(self.total_load_time_ms, 2),
                "avg_context_pack_load_ms": round(self.avg_context_pack_load_time_ms, 2),
                "avg_cerebro_load_ms": round(self.avg_cerebro_load_time_ms, 2),
            },
            "errors": {
                "parse_errors": self.parse_errors,
                "file_not_found": self.file_not_found_errors,
            },
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self.context_pack_cache_hits = 0
        self.context_pack_cache_misses = 0
        self.cerebro_cache_hits = 0
        self.cerebro_cache_misses = 0
        self.constitution_cache_hits = 0
        self.constitution_cache_misses = 0
        self.total_load_time_ms = 0.0
        self.context_pack_load_times = []
        self.cerebro_load_times = []
        self.parse_errors = 0
        self.file_not_found_errors = 0


# Global metrics instance
_spec_kit_metrics = SpecKitMetrics()


def get_spec_kit_metrics() -> SpecKitMetrics:
    """Get the global SpecKitMetrics instance.

    GAP-002 FIX: Returns metrics for observability.

    Returns:
        SpecKitMetrics instance with current metrics.
    """
    return _spec_kit_metrics


def reset_spec_kit_metrics() -> None:
    """Reset all SpecKit metrics.

    GAP-002 FIX: Use this in tests or after reporting.
    """
    _spec_kit_metrics.reset()


# =============================================================================
# SPEC-KIT GAP-001 FIX: LANGFUSE TRACING
# =============================================================================


def _get_langfuse_client():
    """Get Langfuse client for tracing (lazy load).

    GAP-001 FIX: Added Langfuse tracing to SpecKitLoader operations.

    Returns:
        Langfuse client or None if unavailable.
    """
    try:
        from pipeline.langfuse_client import get_langfuse_client
        return get_langfuse_client()
    except ImportError:
        logger.debug("GAP-001: Langfuse client not available for spec_kit tracing")
        return None
    except Exception as e:
        logger.debug(f"GAP-001: Failed to get Langfuse client: {e}")
        return None


def _trace_spec_kit_operation(
    operation_name: str,
    sprint_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    metadata: Optional[dict] = None,
):
    """Context manager for tracing spec kit operations.

    GAP-001 FIX: Adds Langfuse tracing to SpecKitLoader operations.

    Args:
        operation_name: Name of the operation (e.g., 'load_context_pack')
        sprint_id: Sprint ID if applicable
        agent_id: Agent ID if applicable
        metadata: Additional metadata

    Yields:
        Dict for recording operation results.
    """
    from contextlib import contextmanager

    @contextmanager
    def trace_context():
        start_time = time.time()
        langfuse = _get_langfuse_client()
        trace = None
        result_info = {"success": False, "error": None, "duration_ms": 0}

        try:
            # Start trace if Langfuse available
            if langfuse is not None:
                trace_metadata = {
                    "component": "spec_kit_loader",
                    "operation": operation_name,
                }
                if sprint_id:
                    trace_metadata["sprint_id"] = sprint_id
                if agent_id:
                    trace_metadata["agent_id"] = agent_id
                if metadata:
                    trace_metadata.update(metadata)

                trace = langfuse.create_trace(
                    name=f"spec_kit.{operation_name}",
                    metadata=trace_metadata,
                    tags=["spec_kit", operation_name],
                )

            yield result_info

            result_info["success"] = True

        except Exception as e:
            result_info["error"] = str(e)
            raise

        finally:
            result_info["duration_ms"] = int((time.time() - start_time) * 1000)

            # End trace with result
            if trace is not None and langfuse is not None:
                try:
                    langfuse.end_trace(
                        trace.trace_id if hasattr(trace, 'trace_id') else str(trace),
                        status="success" if result_info["success"] else "error",
                        output_data={
                            "duration_ms": result_info["duration_ms"],
                            "error": result_info["error"],
                        },
                    )
                except Exception as e:
                    logger.debug(f"GAP-001: Failed to end trace: {e}")

            # Log operation
            if result_info["success"]:
                logger.debug(
                    f"GAP-001: spec_kit.{operation_name} completed in "
                    f"{result_info['duration_ms']}ms"
                )
            else:
                logger.warning(
                    f"GAP-001: spec_kit.{operation_name} failed: {result_info['error']}"
                )

    return trace_context()


# =============================================================================
# LAZY LOADERS
# =============================================================================


def _get_cerebro_yaml_loader() -> Optional["CerebroYAMLLoader"]:
    """Get CerebroYAMLLoader instance (lazy to avoid circular imports).

    Returns:
        CerebroYAMLLoader instance or None if unavailable.
    """
    try:
        from pipeline.cerebro_yaml_loader import get_cerebro_yaml_loader

        return get_cerebro_yaml_loader()
    except Exception as e:
        logger.debug(f"CerebroYAMLLoader not available: {e}")
        return None


# =============================================================================
# DATA CLASSES
# =============================================================================


# =============================================================================
# INTENT MANIFEST v2.0 DATA CLASSES
# =============================================================================


@dataclass
class Precondition:
    """Precondição DbC - o que DEVE ser verdade ANTES."""

    id: str
    condition: str
    on_violation: str = ""
    applies_to: list[str] = field(default_factory=list)


@dataclass
class Postcondition:
    """Poscondição DbC - o que DEVE ser verdade DEPOIS."""

    id: str
    condition: str
    verification: str = ""
    applies_to: list[str] = field(default_factory=list)


@dataclass
class Invariant:
    """Invariante DbC - o que NUNCA pode ser violado."""

    id: str
    rule: str
    enforcement: str = ""
    violation_severity: str = "HIGH"  # CRITICAL, HIGH, MEDIUM, LOW


@dataclass
class ContractsData:
    """Design by Contract - preconditions, postconditions, invariants."""

    preconditions: list[Precondition] = field(default_factory=list)
    postconditions: list[Postcondition] = field(default_factory=list)
    invariants: list[Invariant] = field(default_factory=list)
    class_invariants: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class EdgeCase:
    """Edge case com comportamento esperado."""

    id: str
    case: str
    expected_behavior: str
    test_id: str = ""


@dataclass
class FailureMode:
    """Modo de falha FMEA Light."""

    id: str
    mode: str
    cause: str
    effect: str
    severity: str = "HIGH"
    detection: str = ""
    prevention: str = ""


@dataclass
class BoundariesData:
    """Limites e restrições - MUST, MUST_NOT, edge cases, failure modes."""

    must_do: list[dict[str, Any]] = field(default_factory=list)
    must_not_do: list[dict[str, Any]] = field(default_factory=list)
    edge_cases: list[EdgeCase] = field(default_factory=list)
    failure_modes: list[FailureMode] = field(default_factory=list)


@dataclass
class AcceptanceCriterion:
    """Critério de aceitação mensurável."""

    id: str
    criteria: str
    measurement: str = ""
    threshold: str = ""


@dataclass
class AntiPattern:
    """Anti-pattern a evitar."""

    id: str
    name: str
    code_smell: str
    correct: str
    detection: str = ""


@dataclass
class QualityData:
    """Critérios de qualidade - DoD, AC, anti-patterns."""

    definition_of_done: dict[str, list[str]] = field(default_factory=dict)
    acceptance_criteria: list[AcceptanceCriterion] = field(default_factory=list)
    anti_patterns: list[AntiPattern] = field(default_factory=list)


@dataclass
class GherkinScenario:
    """Cenário Gherkin."""

    name: str
    type: str = "happy_path"  # happy_path, error_case, edge_case
    given: list[str] = field(default_factory=list)
    when: list[str] = field(default_factory=list)
    then: list[str] = field(default_factory=list)
    test_id: str = ""


@dataclass
class Feature:
    """Feature com scenarios."""

    name: str
    scenarios: list[GherkinScenario] = field(default_factory=list)
    file: str = ""


@dataclass
class BehaviorsData:
    """Comportamentos - features Gherkin e state machines."""

    features: list[Feature] = field(default_factory=list)
    state_machine: dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceabilityData:
    """Rastreabilidade RF -> Deliverable -> Test -> Gate."""

    rf_to_deliverable: dict[str, str] = field(default_factory=dict)
    rf_to_test: dict[str, list[str]] = field(default_factory=dict)
    inv_to_enforcement: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass
class IntentManifestV2:
    """Intent Manifest v2.0 - 7 camadas de documentação."""

    version: str = "2.0"

    # Layer 1: Identity
    identity: dict[str, Any] = field(default_factory=dict)

    # Layer 2: Desire (SMART + Requirements)
    desire: dict[str, Any] = field(default_factory=dict)
    functional_requirements: list[dict[str, Any]] = field(default_factory=list)
    non_functional_requirements: list[dict[str, Any]] = field(default_factory=list)

    # Layer 3: Contracts (DbC)
    contracts: ContractsData = field(default_factory=ContractsData)

    # Layer 4: Behaviors (Gherkin)
    behaviors: BehaviorsData = field(default_factory=BehaviorsData)

    # Layer 5: Boundaries
    boundaries: BoundariesData = field(default_factory=BoundariesData)

    # Layer 6: Quality
    quality: QualityData = field(default_factory=QualityData)

    # Layer 7: Context
    context: dict[str, Any] = field(default_factory=dict)

    # Traceability
    traceability: TraceabilityData = field(default_factory=TraceabilityData)


# =============================================================================
# CONTEXT PACK DATA CLASS
# =============================================================================


@dataclass
class ContextPack:
    """Representa um Context Pack de sprint."""

    sprint_id: str
    module: str = ""
    name: str = ""
    wave: str = ""
    priority: str = ""
    program: str = ""

    objective: str = ""
    deliverables: list[str] = field(default_factory=list)
    models: dict[str, Any] = field(default_factory=dict)
    enums: dict[str, Any] = field(default_factory=dict)
    invariants: dict[str, str] = field(default_factory=dict)
    gates: list[str] = field(default_factory=list)
    dependencies: dict[str, Any] = field(default_factory=dict)

    # Testing requirements
    tests: dict[str, Any] = field(default_factory=dict)
    testing_gaps: dict[str, Any] = field(default_factory=dict)

    # Additional sections (v1.0 legacy)
    decision_tree: str = ""
    contracts: str = ""
    anti_patterns: str = ""
    validation_checklist: str = ""
    protocol_references: dict[str, Any] = field(default_factory=dict)

    # INTENT MANIFEST v2.0
    intent_manifest: Optional[IntentManifestV2] = None
    manifest_version: str = "1.0"  # "1.0" or "2.0"

    # Raw content for reference
    raw_yaml: dict[str, Any] = field(default_factory=dict)
    raw_content: str = ""

    def has_v2_manifest(self) -> bool:
        """Check if this pack has a v2.0 INTENT MANIFEST."""
        return self.intent_manifest is not None and self.manifest_version == "2.0"

    def get_all_invariants(self) -> list[Invariant]:
        """Get all invariants from both v1 and v2 formats."""
        result = []
        # v1 invariants (dict)
        for inv_id, rule in self.invariants.items():
            result.append(Invariant(id=inv_id, rule=rule))
        # v2 invariants
        if self.intent_manifest:
            result.extend(self.intent_manifest.contracts.invariants)
        return result

    def get_all_edge_cases(self) -> list[EdgeCase]:
        """Get all edge cases from v2 manifest."""
        if self.intent_manifest:
            return self.intent_manifest.boundaries.edge_cases
        return []

    def get_definition_of_done(self) -> dict[str, list[str]]:
        """Get Definition of Done from v2 manifest."""
        if self.intent_manifest:
            return self.intent_manifest.quality.definition_of_done
        return {}


@dataclass
class CerebroIdentity:
    """Identidade de um Cerebro."""

    role: str = ""
    alias: str = ""
    version: str = ""
    level: int = 0
    type: str = "CLAUDE"
    reports_to: str = ""
    responds_to: str = ""
    mode: str = ""
    mission: str = ""
    subordinados_diretos: list[str] = field(default_factory=list)


@dataclass
class CerebroSpecKit:
    """Integração Spec Kit do Cerebro."""

    version: str = ""
    reference: str = ""
    my_role: dict[str, Any] = field(default_factory=dict)
    comandos_disponiveis: list[str] = field(default_factory=list)
    responsabilidades: list[str] = field(default_factory=list)
    traceability: dict[str, Any] = field(default_factory=dict)
    constitution_compliance: dict[str, Any] = field(default_factory=dict)


@dataclass
class CerebroHandoff:
    """Configuração de Handoff do Cerebro."""

    version: str = ""
    meu_nivel: str = ""
    receives_from: list[dict[str, Any]] = field(default_factory=list)
    sends_to: list[dict[str, Any]] = field(default_factory=list)
    zap_enforcement: dict[str, Any] = field(default_factory=dict)


@dataclass
class CerebroData:
    """Dados completos de um Cerebro."""

    agent_id: str
    identity: CerebroIdentity = field(default_factory=CerebroIdentity)
    spec_kit: CerebroSpecKit = field(default_factory=CerebroSpecKit)
    handoff: CerebroHandoff = field(default_factory=CerebroHandoff)
    tools_available: dict[str, Any] = field(default_factory=dict)
    contratos: list[dict[str, Any]] = field(default_factory=list)
    mantras: list[str] = field(default_factory=list)

    # Vision anchor (backstory for CrewAI)
    vision_anchor: str = ""
    raw_content: str = ""


@dataclass
class ConstitutionArticle:
    """Artigo constitucional."""

    article_id: str
    nome: str = ""
    status: str = "VIGENTE"
    regra: str = ""
    violacao: dict[str, Any] = field(default_factory=dict)
    severidade: str = ""
    gate: str = ""
    threshold: int = 0
    referencias: list[str] = field(default_factory=list)
    raw_yaml: dict[str, Any] = field(default_factory=dict)


@dataclass
class Constitution:
    """Constitution completa do pipeline."""

    version: str = "1.0.0"
    articles: dict[str, ConstitutionArticle] = field(default_factory=dict)
    raw_content: str = ""


# =============================================================================
# SPEC KIT LOADER
# =============================================================================


class SpecKitLoader:
    """Carrega e parseia artefatos do Spec Kit.

    O Spec Kit define:
    - Context Packs: O que cada sprint deve entregar
    - Cerebros: Como cada agente deve se comportar
    - Constitution: Regras invioláveis do sistema

    Uso:
        loader = SpecKitLoader()

        # Carregar context pack
        pack = loader.load_context_pack("S00")

        # Carregar cerebro
        cerebro = loader.load_cerebro("ace_exec")

        # Carregar constitution
        constitution = loader.load_constitution()

        # Validar dependencias
        ok = loader.validate_dependencies("S05")
    """

    def __init__(
        self,
        docs_dir: Optional[Path] = None,
        project_root: Optional[Path] = None,
    ) -> None:
        """Inicializa o loader.

        Args:
            docs_dir: Diretório de documentação.
            project_root: Raiz do projeto.
        """
        if project_root is None:
            # Auto-detect project root
            project_root = Path(__file__).parent.parent.parent
        self._project_root = project_root

        if docs_dir is None:
            docs_dir = project_root / "docs"
        self._docs_dir = docs_dir

        # Paths específicos
        self._context_packs_dir = docs_dir / "context_packs" / "context_packs"
        self._cerebros_dir = docs_dir / "Agents"
        self._constitution_path = docs_dir / "pipeline" / "CONSTITUTION.md"

        # Cache para evitar re-parsing
        self._context_pack_cache: dict[str, ContextPack] = {}
        self._cerebro_cache: dict[str, CerebroData] = {}
        self._constitution_cache: Optional[Constitution] = None

        # Mapping de agent_id para arquivo cerebro
        self._cerebro_file_map = self._build_cerebro_file_map()

    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------

    def load_context_pack(self, sprint_id: str) -> Optional[ContextPack]:
        """Carrega um Context Pack.

        GAP-001 FIX: Now includes Langfuse tracing for observability.

        Args:
            sprint_id: ID do sprint (ex: "S00", "S01").

        Returns:
            ContextPack ou None se não encontrado.
        """
        # GAP-001 FIX: Add Langfuse tracing
        # GAP-002 FIX: Add metrics tracking
        start_time = time.time()
        with _trace_spec_kit_operation("load_context_pack", sprint_id=sprint_id) as trace_info:
            # Check cache
            if sprint_id in self._context_pack_cache:
                trace_info["cache_hit"] = True
                _spec_kit_metrics.increment_context_pack_cache_hit()
                return self._context_pack_cache[sprint_id]

            _spec_kit_metrics.increment_context_pack_cache_miss()

            # Normalizar sprint_id
            if not sprint_id.startswith("S"):
                sprint_id = f"S{sprint_id.zfill(2)}"

            # Encontrar arquivo
            file_path = self._context_packs_dir / f"{sprint_id}_CONTEXT.md"
            if not file_path.exists():
                logger.warning(f"Context pack not found: {file_path}")
                _spec_kit_metrics.increment_file_not_found_errors()
                return None

            # Parse
            try:
                content = file_path.read_text(encoding="utf-8")
                pack = self._parse_context_pack(sprint_id, content)
                self._context_pack_cache[sprint_id] = pack
                # GAP-002: Record load time (thread-safe)
                load_time_ms = (time.time() - start_time) * 1000
                _spec_kit_metrics.add_context_pack_load_time(load_time_ms)
                _spec_kit_metrics.add_total_load_time(load_time_ms)
                logger.info(f"Loaded context pack: {sprint_id}")
                return pack
            except Exception as e:
                logger.error(f"Failed to parse context pack {sprint_id}: {e}")
                _spec_kit_metrics.increment_parse_errors()
                return None

    def get_context_pack_path(self, sprint_id: str) -> Path:
        """Get the file path for a context pack.

        F-119: Returns the path to the context pack file for receipt generation.

        Args:
            sprint_id: ID do sprint (ex: "S00", "S01").

        Returns:
            Path to the context pack file.
        """
        # Normalizar sprint_id
        if not sprint_id.startswith("S"):
            sprint_id = f"S{sprint_id.zfill(2)}"

        return self._context_packs_dir / f"{sprint_id}_CONTEXT.md"

    def load_cerebro(self, agent_id: str) -> Optional[CerebroData]:
        """Carrega um Cerebro.

        Prioriza o formato YAML modular (6 arquivos via CerebroYAMLLoader)
        e faz fallback para o formato Markdown legado se necessário.

        GAP-001 FIX: Now includes Langfuse tracing for observability.

        Args:
            agent_id: ID do agente (ex: "ace_exec", "spec_master").

        Returns:
            CerebroData ou None se não encontrado.
        """
        # GAP-001 FIX: Add Langfuse tracing
        # GAP-002 FIX: Add metrics tracking
        start_time = time.time()
        with _trace_spec_kit_operation("load_cerebro", agent_id=agent_id) as trace_info:
            # Check cache
            if agent_id in self._cerebro_cache:
                trace_info["cache_hit"] = True
                _spec_kit_metrics.increment_cerebro_cache_hit()
                return self._cerebro_cache[agent_id]

            _spec_kit_metrics.increment_cerebro_cache_miss()

            # PRIORITY 1: Try CerebroYAMLLoader (6 YAML files per agent)
            yaml_loader = _get_cerebro_yaml_loader()
            if yaml_loader is not None:
                try:
                    cerebro_config = yaml_loader.load(agent_id)
                    # Convert CerebroConfig to CerebroData for compatibility
                    cerebro = self._convert_cerebro_config_to_data(agent_id, cerebro_config)
                    self._cerebro_cache[agent_id] = cerebro
                    trace_info["source"] = "yaml"
                    # GAP-002: Record load time (thread-safe)
                    load_time_ms = (time.time() - start_time) * 1000
                    _spec_kit_metrics.add_cerebro_load_time(load_time_ms)
                    _spec_kit_metrics.add_total_load_time(load_time_ms)
                    logger.info(f"Loaded cerebro from YAML: {agent_id}")
                    return cerebro
                except FileNotFoundError:
                    logger.debug(f"Cerebro YAML not found for {agent_id}, trying Markdown")
                except Exception as e:
                    logger.warning(f"Failed to load cerebro YAML {agent_id}: {e}")
                    _spec_kit_metrics.increment_parse_errors()

            # FALLBACK: Markdown format (legacy)
            file_path = self._cerebro_file_map.get(agent_id)
            if file_path is None or not file_path.exists():
                logger.warning(f"Cerebro not found for agent: {agent_id}")
                _spec_kit_metrics.increment_file_not_found_errors()
                return None

            # Parse Markdown
            try:
                content = file_path.read_text(encoding="utf-8")
                cerebro = self._parse_cerebro(agent_id, content)
                self._cerebro_cache[agent_id] = cerebro
                trace_info["source"] = "markdown"
                # GAP-002: Record load time (thread-safe)
                load_time_ms = (time.time() - start_time) * 1000
                _spec_kit_metrics.add_cerebro_load_time(load_time_ms)
                _spec_kit_metrics.add_total_load_time(load_time_ms)
                logger.info(f"Loaded cerebro from Markdown: {agent_id}")
                return cerebro
            except Exception as e:
                logger.error(f"Failed to parse cerebro {agent_id}: {e}")
                _spec_kit_metrics.increment_parse_errors()
                return None

    def _convert_cerebro_config_to_data(
        self, agent_id: str, config: Any
    ) -> CerebroData:
        """Converte CerebroConfig (YAML) para CerebroData (compatível).

        Args:
            agent_id: ID do agente.
            config: CerebroConfig do CerebroYAMLLoader.

        Returns:
            CerebroData compatível.
        """
        # Extract identity from core.yml
        identity_data = config.identity
        identity = CerebroIdentity(
            role=identity_data.get("role", agent_id),
            alias=identity_data.get("alias", ""),
            version=identity_data.get("version", "1.0.0"),
            level=identity_data.get("level", 0),
            type=identity_data.get("type", "CLAUDE"),
            reports_to=config.reports_to,
            responds_to="",
            mode=identity_data.get("mode", ""),
            mission=identity_data.get("mission", ""),
            subordinados_diretos=config.core.get("hierarchy", {}).get("subordinates", []),
        )

        # Extract spec_kit integration
        spec_kit = CerebroSpecKit(
            version="2.0.0",
            reference=f"docs/Agents/.../{agent_id}/",
            my_role={"agent_id": agent_id},
            comandos_disponiveis=[],
            responsabilidades=[],
            traceability={},
            constitution_compliance={},
        )

        # Extract handoff engine
        handoff_data = config.handoff_engine
        handoff = CerebroHandoff(
            version=handoff_data.get("version", "3.0"),
            meu_nivel=f"L{config.level}",
            receives_from=handoff_data.get("receives_from", []),
            sends_to=handoff_data.get("sends_to", []),
            zap_enforcement=handoff_data.get("zap_enforcement", {}),
        )

        # Extract contracts
        contracts = config.contract_list

        # Extract mantras from contracts (look for MANTRA in IDs)
        mantras = []
        for contract in contracts:
            if "MANTRA" in contract.get("id", "").upper():
                mantras.append(contract.get("texto", contract.get("rule", "")))

        # Vision anchor fallback: vision_anchor -> mission
        vision_anchor = config.vision_anchor
        if not vision_anchor and identity.mission:
            vision_anchor = identity.mission

        return CerebroData(
            agent_id=agent_id,
            identity=identity,
            spec_kit=spec_kit,
            handoff=handoff,
            tools_available=config.core.get("tools", {}),
            contratos=contracts,
            mantras=mantras,
            vision_anchor=vision_anchor,
            raw_content="",  # YAML format doesn't have single raw content
        )

    def load_constitution(self) -> Optional[Constitution]:
        """Carrega a Constitution.

        GAP-001 FIX: Now includes Langfuse tracing for observability.

        Returns:
            Constitution ou None se não encontrada.
        """
        # GAP-001 FIX: Add Langfuse tracing
        with _trace_spec_kit_operation("load_constitution") as trace_info:
            # Check cache
            if self._constitution_cache is not None:
                trace_info["cache_hit"] = True
                return self._constitution_cache

            # Verificar arquivo
            if not self._constitution_path.exists():
                logger.warning(f"Constitution not found: {self._constitution_path}")
                return None

            # Parse
            try:
                content = self._constitution_path.read_text(encoding="utf-8")
                constitution = self._parse_constitution(content)
                self._constitution_cache = constitution
                trace_info["article_count"] = len(constitution.articles)
                logger.info(f"Loaded constitution with {len(constitution.articles)} articles")
                return constitution
            except Exception as e:
                logger.error(f"Failed to parse constitution: {e}")
                return None

    def validate_dependencies(self, sprint_id: str) -> bool:
        """Valida se dependências do sprint estão satisfeitas.

        Args:
            sprint_id: ID do sprint.

        Returns:
            True se todas as dependências estão satisfeitas.
        """
        pack = self.load_context_pack(sprint_id)
        if pack is None:
            return False

        # Verificar dependências upstream
        upstream = pack.dependencies.get("upstream")
        if upstream is None:
            return True

        # Para cada dependência upstream, verificar se existe
        if isinstance(upstream, list):
            for dep in upstream:
                dep_pack = self.load_context_pack(dep)
                if dep_pack is None:
                    logger.warning(f"Missing upstream dependency: {dep}")
                    return False
        elif isinstance(upstream, str):
            dep_pack = self.load_context_pack(upstream)
            if dep_pack is None:
                logger.warning(f"Missing upstream dependency: {upstream}")
                return False

        return True

    def get_sprint_gates(self, sprint_id: str) -> list[str]:
        """Retorna gates requeridas para um sprint.

        Args:
            sprint_id: ID do sprint.

        Returns:
            Lista de gate IDs.
        """
        pack = self.load_context_pack(sprint_id)
        if pack is None:
            return []
        return pack.gates

    def get_sprint_deliverables(self, sprint_id: str) -> list[str]:
        """Retorna deliverables de um sprint.

        Args:
            sprint_id: ID do sprint.

        Returns:
            Lista de caminhos de arquivos.
        """
        pack = self.load_context_pack(sprint_id)
        if pack is None:
            return []
        return pack.deliverables

    def get_article(self, article_id: str) -> Optional[ConstitutionArticle]:
        """Retorna um artigo específico da Constitution.

        Args:
            article_id: ID do artigo (ex: "III_COVERAGE_GATE").

        Returns:
            ConstitutionArticle ou None.
        """
        constitution = self.load_constitution()
        if constitution is None:
            return None
        return constitution.articles.get(article_id)

    def list_context_packs(self) -> list[str]:
        """Lista todos os Context Packs disponíveis.

        Returns:
            Lista de sprint IDs.
        """
        packs = []
        if self._context_packs_dir.exists():
            for f in self._context_packs_dir.glob("S*_CONTEXT.md"):
                sprint_id = f.stem.replace("_CONTEXT", "")
                packs.append(sprint_id)
        return sorted(packs)

    def list_cerebros(self) -> list[str]:
        """Lista todos os Cerebros disponíveis.

        Returns:
            Lista de agent IDs.
        """
        return sorted(self._cerebro_file_map.keys())

    def clear_cache(self) -> None:
        """Limpa o cache de artefatos.

        Note:
            This requires manual invocation. For automatic cache invalidation
            in development mode, consider using a file watcher (e.g., watchdog).
            See: SPEC_KIT_AUDIT.md GAP-004 for implementation recommendations.
        """
        self._context_pack_cache.clear()
        self._cerebro_cache.clear()
        self._constitution_cache = None
        logger.debug("Spec Kit cache cleared")

    def health_check(self) -> dict[str, Any]:
        """Perform health check on the Spec Kit.

        GAP-002 FIX: Added health check for observability.

        Validates:
            - Docs directory exists
            - Context packs directory exists
            - Cerebros directory exists
            - At least one context pack can be loaded
            - At least one cerebro can be loaded

        Returns:
            Dictionary with health status and details.
        """
        health = {
            "healthy": True,
            "status": "OK",
            "checks": {},
            "metrics": get_spec_kit_metrics().to_dict(),
        }

        # Check 1: Docs directory exists
        docs_exists = self._docs_dir.exists()
        health["checks"]["docs_directory"] = {
            "status": "PASS" if docs_exists else "FAIL",
            "path": str(self._docs_dir),
        }
        if not docs_exists:
            health["healthy"] = False
            health["status"] = "UNHEALTHY"

        # Check 2: Context packs directory exists
        cp_dir_exists = self._context_packs_dir.exists()
        health["checks"]["context_packs_directory"] = {
            "status": "PASS" if cp_dir_exists else "FAIL",
            "path": str(self._context_packs_dir),
        }
        if not cp_dir_exists:
            health["healthy"] = False
            health["status"] = "UNHEALTHY"

        # Check 3: Cerebros directory exists
        cerebros_dir_exists = self._cerebros_dir.exists()
        health["checks"]["cerebros_directory"] = {
            "status": "PASS" if cerebros_dir_exists else "FAIL",
            "path": str(self._cerebros_dir),
        }
        if not cerebros_dir_exists:
            health["healthy"] = False
            health["status"] = "UNHEALTHY"

        # Check 4: Can load at least one context pack
        if cp_dir_exists:
            try:
                pack = self.load_context_pack("S00")
                cp_loadable = pack is not None
            except Exception:
                cp_loadable = False
            health["checks"]["context_pack_loadable"] = {
                "status": "PASS" if cp_loadable else "WARN",
                "tested": "S00",
            }
            if not cp_loadable:
                health["status"] = "DEGRADED" if health["healthy"] else health["status"]

        # Check 5: Can load at least one cerebro
        if cerebros_dir_exists:
            try:
                cerebro = self.load_cerebro("ace_exec")
                cerebro_loadable = cerebro is not None
            except Exception:
                cerebro_loadable = False
            health["checks"]["cerebro_loadable"] = {
                "status": "PASS" if cerebro_loadable else "WARN",
                "tested": "ace_exec",
            }
            if not cerebro_loadable:
                health["status"] = "DEGRADED" if health["healthy"] else health["status"]

        # Check 6: Count available packs
        try:
            context_packs = self.list_context_packs()
            cerebros = self.list_cerebros()
            health["checks"]["availability"] = {
                "status": "PASS",
                "context_packs_count": len(context_packs),
                "cerebros_count": len(cerebros),
            }
        except Exception as e:
            health["checks"]["availability"] = {
                "status": "FAIL",
                "error": str(e),
            }

        return health

    # -------------------------------------------------------------------------
    # Private Methods - Parsing
    # -------------------------------------------------------------------------

    def _parse_context_pack(self, sprint_id: str, content: str) -> ContextPack:
        """Parseia conteúdo de um Context Pack.

        Args:
            sprint_id: ID do sprint.
            content: Conteúdo do arquivo markdown.

        Returns:
            ContextPack parseado.
        """
        pack = ContextPack(sprint_id=sprint_id, raw_content=content)

        # Extrair bloco YAML principal do RELOAD ANCHOR
        yaml_blocks = self._extract_yaml_blocks(content)

        # Encontrar o bloco RELOAD ANCHOR (contém 'sprint' ou 'objective')
        # NÃO é mais o primeiro bloco - pode ter product_reference antes
        main_yaml = None
        for block in yaml_blocks:
            # RELOAD ANCHOR tem 'sprint' dict ou 'objective' string
            if "sprint" in block or "objective" in block:
                main_yaml = block
                break

        # Fallback: primeiro bloco se nenhum tem sprint/objective (legacy)
        if not main_yaml and yaml_blocks:
            main_yaml = yaml_blocks[0]
            logger.warning(
                f"Context pack {sprint_id}: usando primeiro bloco YAML como fallback. "
                f"Considere adicionar seção RELOAD ANCHOR com 'sprint' e 'objective'."
            )

        if main_yaml:
            pack.raw_yaml = main_yaml

            # Sprint metadata
            sprint_data = main_yaml.get("sprint", {})
            if isinstance(sprint_data, dict):
                pack.module = sprint_data.get("module", "")
                pack.name = sprint_data.get("name", "")
                pack.wave = sprint_data.get("wave", "")
                pack.priority = sprint_data.get("priority", "")
                pack.program = sprint_data.get("program", "")

            # Objective - can be at root level OR inside sprint block
            pack.objective = main_yaml.get("objective", "")
            if not pack.objective and isinstance(sprint_data, dict):
                pack.objective = sprint_data.get("objective", "")

            # Deliverables
            deliverables = main_yaml.get("deliverables", [])
            if isinstance(deliverables, list):
                # Limpar comentários dos deliverables
                pack.deliverables = [
                    d.split("#")[0].strip() if isinstance(d, str) else str(d)
                    for d in deliverables
                ]

            # Models
            pack.models = main_yaml.get("models", {})

            # Enums
            pack.enums = main_yaml.get("enums", {})

            # Invariants
            pack.invariants = main_yaml.get("invariants", {})

            # Gates
            gates = main_yaml.get("gates", [])
            if isinstance(gates, list):
                pack.gates = gates
            elif isinstance(gates, str):
                pack.gates = [g.strip() for g in gates.split(",")]

            # Dependencies
            pack.dependencies = main_yaml.get("dependencies", {})

            # Tests
            pack.tests = main_yaml.get("tests", {})
            pack.testing_gaps = main_yaml.get("testing_gaps", {})

        # Extrair seções de texto (v1.0 legacy)
        pack.decision_tree = self._extract_section(content, "DECISION TREE")
        pack.contracts = self._extract_section(content, "CONTRACTS")
        pack.anti_patterns = self._extract_section(content, "ANTI-PATTERNS")
        pack.validation_checklist = self._extract_section(content, "VALIDATION CHECKLIST")

        # Protocol references (último bloco YAML geralmente)
        if len(yaml_blocks) > 1:
            for block in yaml_blocks[1:]:
                if "protocols_applicable" in block:
                    pack.protocol_references = block

        # Parse INTENT MANIFEST v2.0 if present
        pack.intent_manifest = self._parse_intent_manifest_v2(content, yaml_blocks)
        if pack.intent_manifest:
            pack.manifest_version = "2.0"
            logger.debug(f"Parsed INTENT MANIFEST v2.0 for {sprint_id}")

            # 2026-01-11: Extract deliverables from rf_to_deliverable if deliverables list is empty
            # This handles newer packs (S26+) that have deliverables only in traceability
            if not pack.deliverables and pack.intent_manifest.traceability:
                rf_to_del = pack.intent_manifest.traceability.rf_to_deliverable
                if rf_to_del:
                    # Extract unique file paths (format: "src/path/file.py:Function")
                    seen_files: set[str] = set()
                    for rf_path in rf_to_del.values():
                        # Split "src/veritas/module.py:ClassName.method" -> "src/veritas/module.py"
                        file_path = rf_path.split(":")[0] if ":" in rf_path else rf_path
                        if file_path and file_path.startswith("src/"):
                            seen_files.add(file_path)
                    pack.deliverables = sorted(seen_files)
                    if pack.deliverables:
                        logger.debug(
                            f"Extracted {len(pack.deliverables)} deliverables from "
                            f"rf_to_deliverable for {sprint_id}"
                        )

        return pack

    def _parse_intent_manifest_v2(
        self, content: str, yaml_blocks: list[dict[str, Any]]
    ) -> Optional[IntentManifestV2]:
        """Parse INTENT MANIFEST sections (v1.0 or v2.0 format).

        This method parses INTENT MANIFEST sections from context packs.
        It supports both v1.0 (partial) and v2.0 (full) formats.

        v1.0 format has: desire, requisitos_funcionais, requisitos_nao_funcionais, gherkin
        v2.0 format adds: structured contracts, boundaries, quality, traceability

        Args:
            content: Full markdown content.
            yaml_blocks: All YAML blocks from the document.

        Returns:
            IntentManifestV2 or None if no INTENT MANIFEST found.
        """
        # Check if INTENT MANIFEST section exists
        if "## INTENT MANIFEST" not in content:
            return None

        manifest = IntentManifestV2()

        # Find YAML blocks within INTENT MANIFEST section
        # Handle both "## INTENT MANIFEST" and "## INTENT MANIFEST - ..."
        intent_section = ""
        pattern = r"## INTENT MANIFEST[^\n]*\n(.*?)(?=\n## [A-Z]|\Z)"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            intent_section = match.group(0)

        if not intent_section:
            return None

        # Extract YAML blocks from intent section
        intent_yaml_blocks = self._extract_yaml_blocks(intent_section)

        for block in intent_yaml_blocks:
            # Parse desire (Layer 2)
            if "desire" in block:
                manifest.desire = block.get("desire", {})

            # Parse functional requirements
            if "requisitos_funcionais" in block:
                rf_data = block.get("requisitos_funcionais", {})
                for rf_id, rf_info in rf_data.items():
                    if isinstance(rf_info, dict):
                        manifest.functional_requirements.append({
                            "id": rf_id,
                            **rf_info
                        })

            # Parse non-functional requirements
            if "requisitos_nao_funcionais" in block:
                rnf_data = block.get("requisitos_nao_funcionais", {})
                for rnf_id, rnf_info in rnf_data.items():
                    if isinstance(rnf_info, dict):
                        manifest.non_functional_requirements.append({
                            "id": rnf_id,
                            **rnf_info
                        })

            # Parse contracts (Layer 3)
            if "contracts" in block:
                contracts_data = block.get("contracts", {})
                manifest.contracts = self._parse_contracts_data(contracts_data)

            # Parse boundaries (Layer 5)
            if "boundaries" in block:
                boundaries_data = block.get("boundaries", {})
                manifest.boundaries = self._parse_boundaries_data(boundaries_data)

            # Parse quality (Layer 6)
            if "quality" in block or "definition_of_done" in block:
                quality_data = block.get("quality", block)
                manifest.quality = self._parse_quality_data(quality_data)

            # Parse context (Layer 7)
            if "context" in block or "contexto" in block:
                manifest.context = block.get("context", block.get("contexto", {}))

            # Parse traceability
            if "traceability" in block:
                trace_data = block.get("traceability", {})
                manifest.traceability = self._parse_traceability_data(trace_data)

            # Parse identity (Layer 1)
            if "identity" in block:
                manifest.identity = block.get("identity", {})

        # Extract Gherkin scenarios from content (Layer 4)
        manifest.behaviors = self._parse_behaviors_from_content(intent_section)

        return manifest

    def _parse_contracts_data(self, data: dict[str, Any]) -> ContractsData:
        """Parse contracts (DbC) data."""
        contracts = ContractsData()

        # Preconditions
        for pre in data.get("preconditions", []):
            if isinstance(pre, dict):
                contracts.preconditions.append(Precondition(
                    id=pre.get("id", ""),
                    condition=pre.get("condition", ""),
                    on_violation=pre.get("on_violation", ""),
                    applies_to=pre.get("applies_to", []),
                ))

        # Postconditions
        for post in data.get("postconditions", []):
            if isinstance(post, dict):
                contracts.postconditions.append(Postcondition(
                    id=post.get("id", ""),
                    condition=post.get("condition", ""),
                    verification=post.get("verification", ""),
                    applies_to=post.get("applies_to", []),
                ))

        # Invariants
        for inv in data.get("invariants", []):
            if isinstance(inv, dict):
                contracts.invariants.append(Invariant(
                    id=inv.get("id", ""),
                    rule=inv.get("rule", ""),
                    enforcement=inv.get("enforcement", ""),
                    violation_severity=inv.get("violation_severity", "HIGH"),
                ))

        # Class invariants
        contracts.class_invariants = data.get("class_invariants", {})

        return contracts

    def _parse_boundaries_data(self, data: dict[str, Any]) -> BoundariesData:
        """Parse boundaries data."""
        boundaries = BoundariesData()

        boundaries.must_do = data.get("must_do", [])
        boundaries.must_not_do = data.get("must_not_do", [])

        # Edge cases
        for ec in data.get("edge_cases", []):
            if isinstance(ec, dict):
                boundaries.edge_cases.append(EdgeCase(
                    id=ec.get("id", ""),
                    case=ec.get("case", ec.get("caso", "")),
                    expected_behavior=ec.get("expected_behavior", ec.get("comportamento_esperado", "")),
                    test_id=ec.get("test_id", ""),
                ))

        # Failure modes
        for fm in data.get("failure_modes", []):
            if isinstance(fm, dict):
                boundaries.failure_modes.append(FailureMode(
                    id=fm.get("id", ""),
                    mode=fm.get("mode", fm.get("modo", "")),
                    cause=fm.get("cause", fm.get("causa", "")),
                    effect=fm.get("effect", fm.get("efeito", "")),
                    severity=fm.get("severity", "HIGH"),
                    detection=fm.get("detection", fm.get("deteccao", "")),
                    prevention=fm.get("prevention", fm.get("prevencao", "")),
                ))

        return boundaries

    def _parse_quality_data(self, data: dict[str, Any]) -> QualityData:
        """Parse quality data."""
        quality = QualityData()

        # Definition of Done
        dod = data.get("definition_of_done", {})
        if isinstance(dod, dict):
            quality.definition_of_done = dod
        elif isinstance(dod, list):
            quality.definition_of_done = {"items": dod}

        # Acceptance criteria
        for ac in data.get("acceptance_criteria", []):
            if isinstance(ac, dict):
                quality.acceptance_criteria.append(AcceptanceCriterion(
                    id=ac.get("id", ""),
                    criteria=ac.get("criteria", ac.get("criterio", "")),
                    measurement=ac.get("measurement", ac.get("medicao", "")),
                    threshold=ac.get("threshold", ""),
                ))

        # Anti-patterns
        for ap in data.get("anti_patterns", []):
            if isinstance(ap, dict):
                quality.anti_patterns.append(AntiPattern(
                    id=ap.get("id", ""),
                    name=ap.get("name", ap.get("nome", "")),
                    code_smell=ap.get("code_smell", ap.get("codigo_errado", "")),
                    correct=ap.get("correct", ap.get("codigo_certo", "")),
                    detection=ap.get("detection", ap.get("como_detectar", "")),
                ))

        return quality

    def _parse_traceability_data(self, data: dict[str, Any]) -> TraceabilityData:
        """Parse traceability data."""
        return TraceabilityData(
            rf_to_deliverable=data.get("rf_to_deliverable", {}),
            rf_to_test=data.get("rf_to_test", {}),
            inv_to_enforcement=data.get("inv_to_enforcement", {}),
        )

    def _parse_behaviors_from_content(self, content: str) -> BehaviorsData:
        """Parse Gherkin behaviors from content."""
        behaviors = BehaviorsData()

        # Find Gherkin blocks
        gherkin_pattern = r"```gherkin\s*\n(.*?)```"
        gherkin_matches = re.findall(gherkin_pattern, content, re.DOTALL)

        for gherkin_content in gherkin_matches:
            # Extract feature name
            feature_match = re.search(r"Feature:\s*(.+?)$", gherkin_content, re.MULTILINE)
            feature_name = feature_match.group(1).strip() if feature_match else "Unnamed Feature"

            feature = Feature(name=feature_name)

            # Extract scenarios
            scenario_pattern = r"Scenario(?:\s+Outline)?:\s*(.+?)$\s*(.*?)(?=Scenario|$)"
            scenario_matches = re.findall(scenario_pattern, gherkin_content, re.MULTILINE | re.DOTALL)

            for scenario_name, scenario_body in scenario_matches:
                scenario = GherkinScenario(name=scenario_name.strip())

                # Extract Given/When/Then
                given_matches = re.findall(r"Given\s+(.+?)$", scenario_body, re.MULTILINE)
                when_matches = re.findall(r"When\s+(.+?)$", scenario_body, re.MULTILINE)
                then_matches = re.findall(r"Then\s+(.+?)$", scenario_body, re.MULTILINE)
                and_matches = re.findall(r"And\s+(.+?)$", scenario_body, re.MULTILINE)

                scenario.given = given_matches
                scenario.when = when_matches
                scenario.then = then_matches + and_matches

                # Determine type
                if "erro" in scenario_name.lower() or "rejeitar" in scenario_name.lower():
                    scenario.type = "error_case"
                elif "borda" in scenario_name.lower() or "edge" in scenario_name.lower():
                    scenario.type = "edge_case"
                else:
                    scenario.type = "happy_path"

                feature.scenarios.append(scenario)

            if feature.scenarios:
                behaviors.features.append(feature)

        return behaviors

    def _parse_cerebro(self, agent_id: str, content: str) -> CerebroData:
        """Parseia conteúdo de um Cerebro.

        Args:
            agent_id: ID do agente.
            content: Conteúdo do arquivo markdown.

        Returns:
            CerebroData parseado.
        """
        cerebro = CerebroData(agent_id=agent_id, raw_content=content)

        # Extrair todos os blocos YAML
        yaml_blocks = self._extract_yaml_blocks(content)

        for block in yaml_blocks:
            # Identity block
            if "identity" in block:
                id_data = block["identity"]
                cerebro.identity = CerebroIdentity(
                    role=id_data.get("role", ""),
                    alias=id_data.get("alias", ""),
                    version=id_data.get("version", ""),
                    level=id_data.get("level", 0),
                    type=id_data.get("type", "CLAUDE"),
                    reports_to=id_data.get("reports_to", ""),
                    responds_to=id_data.get("responds_to", ""),
                    mode=id_data.get("mode", ""),
                    mission=id_data.get("mission", ""),
                    subordinados_diretos=id_data.get("subordinados_diretos", []),
                )

            # Spec Kit integration
            if "spec_kit" in block:
                sk_data = block["spec_kit"]
                cerebro.spec_kit = CerebroSpecKit(
                    version=sk_data.get("version", ""),
                    reference=sk_data.get("reference", ""),
                    my_role=sk_data.get("my_role", {}),
                    comandos_disponiveis=sk_data.get("comandos_disponiveis", []),
                    responsabilidades=sk_data.get("responsabilidades", []),
                    traceability=sk_data.get("traceability", {}),
                    constitution_compliance=sk_data.get("constitution_compliance", {}),
                )

            # Handoff engine
            if "handoff_engine" in block:
                hf_data = block["handoff_engine"]
                cerebro.handoff = CerebroHandoff(
                    version=hf_data.get("version", ""),
                    meu_nivel=hf_data.get("meu_nivel", ""),
                    receives_from=hf_data.get("receives_from", []),
                    sends_to=hf_data.get("sends_to", []),
                    zap_enforcement=hf_data.get("zap_enforcement", {}),
                )

            # Tools available
            if "tools_available" in block:
                cerebro.tools_available = block["tools_available"]

            # Contratos
            if "contratos" in block:
                cerebro.contratos = block["contratos"]

        # Extrair mantras
        mantras_section = self._extract_section(content, "MANTRAS")
        if mantras_section:
            # Parse numbered mantras
            cerebro.mantras = re.findall(r'\d+\.\s*"([^"]+)"', mantras_section)

        # Vision anchor - usar mission ou primeira seção descritiva
        if cerebro.identity.mission:
            cerebro.vision_anchor = cerebro.identity.mission
        else:
            # Tentar extrair do primeiro parágrafo após o título
            match = re.search(r"^>\s*\*\*[^*]+\*\*:?\s*(.+?)$", content, re.MULTILINE)
            if match:
                cerebro.vision_anchor = match.group(1).strip()

        return cerebro

    def _parse_constitution(self, content: str) -> Constitution:
        """Parseia a Constitution.

        Args:
            content: Conteúdo do arquivo markdown.

        Returns:
            Constitution parseada.
        """
        constitution = Constitution(raw_content=content)

        # Extrair blocos YAML de cada artigo
        yaml_blocks = self._extract_yaml_blocks(content)

        for block in yaml_blocks:
            # Verificar se é um artigo
            if "artigo" in block:
                article_id = block["artigo"]
                article = ConstitutionArticle(
                    article_id=article_id,
                    nome=block.get("nome", ""),
                    status=block.get("status", "VIGENTE"),
                    regra=block.get("regra", ""),
                    violacao=block.get("violacao", {}),
                    severidade=block.get("violacao", {}).get("severidade", ""),
                    gate=block.get("gate", ""),
                    threshold=block.get("threshold", 0),
                    referencias=block.get("referencias", []),
                    raw_yaml=block,
                )
                constitution.articles[article_id] = article

        return constitution

    # -------------------------------------------------------------------------
    # Private Methods - Helpers
    # -------------------------------------------------------------------------

    def _extract_yaml_blocks(self, content: str) -> list[dict[str, Any]]:
        """Extrai blocos YAML de conteúdo markdown.

        Args:
            content: Conteúdo markdown.

        Returns:
            Lista de dicts parseados do YAML.
        """
        blocks = []

        # OPT-12-006: Use pre-compiled pattern for better performance
        matches = _YAML_BLOCK_PATTERN.findall(content)

        for match in matches:
            try:
                parsed = yaml.safe_load(match)
                if parsed and isinstance(parsed, dict):
                    blocks.append(parsed)
            except yaml.YAMLError as e:
                logger.debug(f"Failed to parse YAML block: {e}")

        return blocks

    def _extract_section(self, content: str, section_name: str) -> str:
        """Extrai uma seção de texto do markdown.

        Args:
            content: Conteúdo markdown.
            section_name: Nome da seção (sem ##).

        Returns:
            Conteúdo da seção.
        """
        # Pattern: ## SECTION_NAME até próximo ## ou fim
        pattern = rf"##\s*{re.escape(section_name)}\s*\n(.*?)(?=\n##\s|\Z)"
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    def _build_cerebro_file_map(self) -> dict[str, Path]:
        """Constrói mapeamento de agent_id para arquivo cerebro.

        Returns:
            Dict de agent_id -> Path.
        """
        file_map = {}

        if not self._cerebros_dir.exists():
            return file_map

        # Mapeamento de nomes de arquivo para agent_id
        name_to_id = {
            "Cerebro Ace Exec": "ace_exec",
            "Cerebro ACE Exec": "ace_exec",
            "Cerebro Spec Master": "spec_master",
            "Cerebro QA Master": "qa_master",
            "Cerebro CEO": "ceo",
            "Cerebro Presidente": "presidente",
            "Cerebro Arbiter": "arbiter",
            "Cerebro Retrospective Master": "retrospective_master",
            "Cerebro Spec VP": "spec_vp",
            "Cerebro Exec VP": "exec_vp",
            "Cerebro Integration Officer": "integration_officer",
            "Cerebro External Liaison": "external_liaison",
            "Cerebro Sprint Planner": "sprint_planner",
            "Cerebro Debt Tracker": "debt_tracker",
            "Cerebro Squad Lead": "squad_lead",
            "Cerebro Human Layer": "human_layer",
            "Cerebro Human Layer Specialist": "human_layer_specialist",
            "Cerebro ACE Orchestration": "ace_orchestration",
            "Cerebro Technical Planner": "technical_planner",
            "Cerebro Product Owner": "product_owner",
            "Cerebro Project Manager": "project_manager",
            "Cerebro Auditor": "auditor",
            "Cerebro Refinador": "refinador",
            "Cerebro Clean Reviewer": "clean_reviewer",
            "Cerebro Edge Case Hunter": "edge_case_hunter",
            "Cerebro Gap Hunter": "gap_hunter",
            "Cerebro Dependency Mapper": "dependency_mapper",
            "Cerebro Task Decomposer": "task_decomposer",
            "Cerebro Red Team Agent": "red_team_agent",
            "Cerebro OPS Ctrl": "ops_ctrl",
            "Cerebro Resource Optimizer": "resource_optimizer",
            "Cerebro System Observer": "system_observer",
            "Cerebro Orchestrator": "orchestrator",
            "Cerebro Run Master": "run_master",
            "Cerebro Run Supervisor": "run_supervisor",
            "Cerebro Blockchain Engineer": "blockchain_engineer",
            "Cerebro Data Engineer": "data_engineer",
            "Cerebro LLM Orchestrator": "llm_orchestrator",
            "Cerebro Legal Tech Specialist": "legal_tech_specialist",
            "Cerebro Oracle Architect": "oracle_architect",
            "Cerebro UI Designer": "ui_designer",
            "Cerebro UX Researcher": "ux_researcher",
            "Cerebro Web3 Frontend": "web3_frontend",
        }

        # Buscar em todos os subdiretórios
        for level_dir in self._cerebros_dir.iterdir():
            if level_dir.is_dir() and level_dir.name.startswith("L"):
                for cerebro_file in level_dir.glob("Cerebro *.md"):
                    name = cerebro_file.stem
                    agent_id = name_to_id.get(name)
                    if agent_id:
                        file_map[agent_id] = cerebro_file
                    else:
                        # Fallback: converter nome para snake_case
                        fallback_id = name.replace("Cerebro ", "").lower().replace(" ", "_")
                        file_map[fallback_id] = cerebro_file

        return file_map


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


# DRIFT-002 FIX: Singleton with proper invalidation and TTL
import time as _time

_loader_instance: Optional[SpecKitLoader] = None
_loader_lock: threading.Lock = threading.Lock()
_loader_created_at: float = 0.0
_loader_docs_dir: Optional[Path] = None
_loader_project_root: Optional[Path] = None

# Default TTL for loader instance: 5 minutes (300 seconds)
# After this time, loader will be recreated on next access
SPEC_KIT_LOADER_TTL_SECONDS: float = 300.0


def get_spec_kit_loader(
    docs_dir: Optional[Path] = None,
    project_root: Optional[Path] = None,
    force_reload: bool = False,
    ttl_seconds: Optional[float] = None,
) -> SpecKitLoader:
    """Obtém instância singleton do SpecKitLoader.

    DRIFT-002 FIX: Now supports force_reload, TTL checking, and parameter validation.

    Args:
        docs_dir: Diretório de documentação.
        project_root: Raiz do projeto.
        force_reload: If True, recreate the loader even if one exists.
        ttl_seconds: Optional custom TTL. If None, uses SPEC_KIT_LOADER_TTL_SECONDS.

    Returns:
        Instância do SpecKitLoader.

    Notes:
        - If docs_dir or project_root differ from the existing loader, a warning
          is logged but the existing loader is returned (unless force_reload=True).
        - After TTL expires, loader is automatically recreated on next access.
    """
    global _loader_instance, _loader_created_at, _loader_docs_dir, _loader_project_root

    ttl = ttl_seconds if ttl_seconds is not None else SPEC_KIT_LOADER_TTL_SECONDS
    now = _time.time()

    # Check if we need to invalidate due to TTL
    ttl_expired = (
        _loader_instance is not None
        and _loader_created_at > 0
        and (now - _loader_created_at) > ttl
    )

    # Check if parameters changed (warning only, doesn't auto-invalidate)
    params_changed = False
    if _loader_instance is not None:
        if docs_dir is not None and _loader_docs_dir != docs_dir:
            params_changed = True
            logger.warning(
                f"DRIFT-002: get_spec_kit_loader called with different docs_dir "
                f"(current: {_loader_docs_dir}, requested: {docs_dir}). "
                f"Use force_reload=True to update."
            )
        if project_root is not None and _loader_project_root != project_root:
            params_changed = True
            logger.warning(
                f"DRIFT-002: get_spec_kit_loader called with different project_root "
                f"(current: {_loader_project_root}, requested: {project_root}). "
                f"Use force_reload=True to update."
            )

    # Create new loader if needed (with double-check locking)
    if _loader_instance is None or force_reload or ttl_expired:
        with _loader_lock:
            # Re-check conditions inside lock
            ttl_expired_inner = (
                _loader_instance is not None
                and _loader_created_at > 0
                and (now - _loader_created_at) > ttl
            )
            if _loader_instance is None or force_reload or ttl_expired_inner:
                if ttl_expired_inner:
                    logger.debug(f"DRIFT-002: SpecKitLoader TTL expired, recreating")
                _loader_instance = SpecKitLoader(docs_dir=docs_dir, project_root=project_root)
                _loader_created_at = now
                _loader_docs_dir = docs_dir
                _loader_project_root = project_root

    return _loader_instance


def reset_spec_kit_loader() -> None:
    """Reset a instância do loader (para testes).

    DRIFT-002 FIX: Also resets tracking variables.
    """
    global _loader_instance, _loader_created_at, _loader_docs_dir, _loader_project_root
    with _loader_lock:
        _loader_instance = None
        _loader_created_at = 0.0
        _loader_docs_dir = None
        _loader_project_root = None


def invalidate_spec_kit_cache() -> None:
    """Invalidate the SpecKitLoader cache without resetting the instance.

    DRIFT-002 FIX: Use this when you know files have changed but don't want
    to recreate the entire loader. This clears internal caches while keeping
    the loader instance.
    """
    global _loader_instance
    if _loader_instance is not None:
        _loader_instance.clear_cache()
        logger.debug("DRIFT-002: SpecKitLoader cache invalidated")


def get_spec_kit_loader_age() -> Optional[float]:
    """Get the age of the current loader instance in seconds.

    DRIFT-002 FIX: Useful for monitoring and debugging cache behavior.

    Returns:
        Age in seconds, or None if no loader exists.
    """
    global _loader_instance, _loader_created_at
    if _loader_instance is None or _loader_created_at == 0:
        return None
    return _time.time() - _loader_created_at


# =============================================================================
# GAP-004 FIX: FILE WATCHER FOR AUTOMATIC CACHE INVALIDATION
# =============================================================================


_file_watcher_thread: Optional[threading.Thread] = None
_file_watcher_stop_event: Optional[threading.Event] = None
_file_watcher_enabled: bool = False


class SpecKitFileEventHandler:
    """File event handler for spec kit cache invalidation.

    GAP-004 FIX: Watches for file changes in docs/ directory and
    automatically invalidates the cache when changes are detected.
    """

    def __init__(self, debounce_seconds: float = 1.0):
        """Initialize the event handler.

        Args:
            debounce_seconds: Minimum time between cache invalidations.
        """
        self._debounce_seconds = debounce_seconds
        self._last_invalidation = 0.0
        self._lock = threading.Lock()

    def _should_invalidate(self, path: str) -> bool:
        """Check if file change should trigger cache invalidation.

        Args:
            path: Path to the changed file.

        Returns:
            True if cache should be invalidated.
        """
        # Only watch for YAML and Markdown files
        if not path.endswith(('.yml', '.yaml', '.md')):
            return False

        # Ignore hidden files and directories
        if '/.' in path or path.startswith('.'):
            return False

        return True

    def _handle_event(self, event_type: str, src_path: str) -> None:
        """Handle a file system event.

        Args:
            event_type: Type of event (created, modified, deleted, moved).
            src_path: Path to the affected file.
        """
        if not self._should_invalidate(src_path):
            return

        with self._lock:
            now = time.time()
            if now - self._last_invalidation < self._debounce_seconds:
                return

            self._last_invalidation = now
            logger.info(
                f"GAP-004: File {event_type}: {src_path}, invalidating cache"
            )
            invalidate_spec_kit_cache()

    def on_created(self, src_path: str) -> None:
        """Handle file created event."""
        self._handle_event("created", src_path)

    def on_modified(self, src_path: str) -> None:
        """Handle file modified event."""
        self._handle_event("modified", src_path)

    def on_deleted(self, src_path: str) -> None:
        """Handle file deleted event."""
        self._handle_event("deleted", src_path)

    def on_moved(self, src_path: str, dest_path: str) -> None:
        """Handle file moved event."""
        self._handle_event("moved", src_path)


def _run_file_watcher(
    watch_path: Path,
    handler: SpecKitFileEventHandler,
    stop_event: threading.Event,
) -> None:
    """Run the file watcher in a background thread.

    GAP-004 FIX: Uses watchdog library for efficient file system monitoring.

    Args:
        watch_path: Directory to watch.
        handler: Event handler for file changes.
        stop_event: Event to signal when to stop watching.
    """
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler

        class WatchdogHandler(FileSystemEventHandler):
            """Adapter for watchdog events."""

            def __init__(self, spec_kit_handler: SpecKitFileEventHandler):
                self._handler = spec_kit_handler

            def on_created(self, event):
                if not event.is_directory:
                    self._handler.on_created(event.src_path)

            def on_modified(self, event):
                if not event.is_directory:
                    self._handler.on_modified(event.src_path)

            def on_deleted(self, event):
                if not event.is_directory:
                    self._handler.on_deleted(event.src_path)

            def on_moved(self, event):
                if not event.is_directory:
                    self._handler.on_moved(event.src_path, event.dest_path)

        observer = Observer()
        watchdog_handler = WatchdogHandler(handler)
        observer.schedule(watchdog_handler, str(watch_path), recursive=True)
        observer.start()

        logger.info(f"GAP-004: File watcher started for {watch_path}")

        # Wait for stop signal
        while not stop_event.wait(timeout=1.0):
            pass

        observer.stop()
        observer.join()
        logger.info("GAP-004: File watcher stopped")

    except ImportError:
        logger.warning(
            "GAP-004: watchdog library not installed. "
            "Install with: pip install watchdog"
        )
        # Fallback: simple polling-based watcher
        _run_polling_watcher(watch_path, handler, stop_event)


def _run_polling_watcher(
    watch_path: Path,
    handler: SpecKitFileEventHandler,
    stop_event: threading.Event,
    poll_interval: float = 5.0,
) -> None:
    """Fallback polling-based file watcher.

    GAP-004 FIX: Used when watchdog is not available.

    Args:
        watch_path: Directory to watch.
        handler: Event handler for file changes.
        stop_event: Event to signal when to stop watching.
        poll_interval: Seconds between polls.
    """
    logger.info(f"GAP-004: Using polling watcher for {watch_path} (interval: {poll_interval}s)")

    # Track file modification times
    file_mtimes: dict[str, float] = {}

    def scan_files() -> dict[str, float]:
        """Scan all relevant files and their modification times."""
        mtimes = {}
        for pattern in ["**/*.yml", "**/*.yaml", "**/*.md"]:
            for file_path in watch_path.glob(pattern):
                if file_path.is_file():
                    try:
                        mtimes[str(file_path)] = file_path.stat().st_mtime
                    except (OSError, FileNotFoundError):
                        logger.debug(f"FILE: File operation failed: {e}")
        return mtimes

    # Initial scan
    file_mtimes = scan_files()

    while not stop_event.wait(timeout=poll_interval):
        try:
            current_mtimes = scan_files()

            # Check for new files
            for path, mtime in current_mtimes.items():
                if path not in file_mtimes:
                    handler.on_created(path)
                elif mtime > file_mtimes[path]:
                    handler.on_modified(path)

            # Check for deleted files
            for path in file_mtimes:
                if path not in current_mtimes:
                    handler.on_deleted(path)

            file_mtimes = current_mtimes

        except Exception as e:
            logger.debug(f"GAP-004: Polling error: {e}")

    logger.info("GAP-004: Polling watcher stopped")


def start_spec_kit_file_watcher(
    watch_path: Optional[Path] = None,
    debounce_seconds: float = 1.0,
) -> bool:
    """Start the file watcher for automatic cache invalidation.

    GAP-004 FIX: Starts a background thread that watches for file changes
    in the docs/ directory and automatically invalidates the cache.

    Only starts in dev mode (SPEC_KIT_FILE_WATCHER=true environment variable).

    Args:
        watch_path: Directory to watch. Defaults to docs/ directory.
        debounce_seconds: Minimum time between cache invalidations.

    Returns:
        True if watcher was started, False otherwise.
    """
    global _file_watcher_thread, _file_watcher_stop_event, _file_watcher_enabled

    import os

    # Check if file watcher is enabled via environment variable
    env_enabled = os.environ.get("SPEC_KIT_FILE_WATCHER", "").lower()
    if env_enabled not in ("true", "1", "yes", "on"):
        logger.debug("GAP-004: File watcher disabled (set SPEC_KIT_FILE_WATCHER=true to enable)")
        return False

    # Don't start if already running
    if _file_watcher_thread is not None and _file_watcher_thread.is_alive():
        logger.debug("GAP-004: File watcher already running")
        return True

    # Determine watch path
    if watch_path is None:
        loader = get_spec_kit_loader()
        watch_path = loader._docs_dir if loader else None

    if watch_path is None or not watch_path.exists():
        logger.warning(f"GAP-004: Watch path does not exist: {watch_path}")
        return False

    # Create handler and stop event
    handler = SpecKitFileEventHandler(debounce_seconds=debounce_seconds)
    _file_watcher_stop_event = threading.Event()

    # Start watcher thread
    _file_watcher_thread = threading.Thread(
        target=_run_file_watcher,
        args=(watch_path, handler, _file_watcher_stop_event),
        daemon=True,
        name="SpecKitFileWatcher",
    )
    _file_watcher_thread.start()
    _file_watcher_enabled = True

    return True


def stop_spec_kit_file_watcher() -> None:
    """Stop the file watcher.

    GAP-004 FIX: Stops the background file watcher thread.
    """
    global _file_watcher_thread, _file_watcher_stop_event, _file_watcher_enabled

    if _file_watcher_stop_event is not None:
        _file_watcher_stop_event.set()

    if _file_watcher_thread is not None:
        _file_watcher_thread.join(timeout=5.0)
        _file_watcher_thread = None

    _file_watcher_stop_event = None
    _file_watcher_enabled = False
    logger.debug("GAP-004: File watcher stopped")


def is_file_watcher_running() -> bool:
    """Check if the file watcher is running.

    GAP-004 FIX: Returns True if the file watcher is active.

    Returns:
        True if file watcher is running.
    """
    global _file_watcher_thread
    return _file_watcher_thread is not None and _file_watcher_thread.is_alive()
