#!/usr/bin/env python3
"""Pack Discovery System - API para agentes descobrirem e carregarem packs.

Este modulo fornece uma interface simples e unificada para:
- Descobrir packs por busca semantica ou tags
- Carregar conteudo de packs
- Navegar entre packs relacionados
- Listar packs por tipo, produto, sprint, etc.

USO RAPIDO:
    from pipeline.pack_discovery import (
        discover_packs,
        load_pack,
        get_related_packs,
        list_packs,
    )

    # Busca semantica
    packs = discover_packs("como implementar FSM")
    # -> [PackInfo(id="S01", type="sprint", ...), PackInfo(id="CONSTITUTION", ...)]

    # Carregar pack
    content = load_pack("S01")
    # -> Conteudo do arquivo S01_CONTEXT.md

    # Packs relacionados
    related = get_related_packs("S01")
    # -> [PackInfo(id="FOUNDATION"), PackInfo(id="CONSTITUTION"), ...]

    # Listar por tipo
    sprints = list_packs(pack_type="sprint", product="FOUNDATION")
    # -> [PackInfo(id="S00"), PackInfo(id="S01"), ...]

Author: Pipeline Autonomo
Date: 2026-01-11
Version: 1.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import yaml

# PERF-001 FIX: Added lru_cache import for caching expensive operations

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

# Base path do projeto (assumindo que estamos em src/pipeline/)
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_CONTEXT_PACKS = _PROJECT_ROOT / "docs" / "context_packs"
_UNIFIED_INDEX_PATH = _CONTEXT_PACKS / "UNIFIED_PACK_INDEX.yaml"
_STACK_INDEX_PATH = _CONTEXT_PACKS / "STACK_MASTER_INDEX.yaml"
_BINDING_INDEX_PATH = _CONTEXT_PACKS / "PACK_BINDING_INDEX.yaml"

# NEW: Paths for documentation packs
_FEATURE_PACKS_DIR = _CONTEXT_PACKS / "feature_packs"
_USER_JOURNEYS_DIR = _CONTEXT_PACKS / "user_journeys"
_SPECS_DIR = _CONTEXT_PACKS / "specs"

# Cache do indice
_unified_index_cache: dict | None = None
_stack_index_cache: dict | None = None
_binding_index_cache: dict | None = None


class PackType(str, Enum):
    """Tipos de packs disponiveis."""

    SPRINT = "sprint"
    PRODUCT = "product"
    STACK = "stack"
    THEORY = "theory"
    GATE = "gate"
    PROTOCOL = "protocol"
    QUANTUM_LEAP = "quantum_leap"
    # NEW: Documentation packs for development workflow
    FEATURE = "feature"
    USER_JOURNEY = "user_journey"
    SPECIFICATION = "specification"
    BUSINESS_RULES = "business_rules"
    ALIGNMENT = "alignment"
    LEGACY_FOREKAST_SPRINT = "forekast_sprint"
    LEGACY_VISIONARY_SPRINT = "visionary_sprint"


@dataclass
class PackInfo:
    """Informacoes sobre um pack."""

    id: str
    type: PackType
    name: str
    file: str
    path: Path
    tags: list[str] = field(default_factory=list)
    summary: str = ""
    product: str | None = None
    related_sprints: list[str] = field(default_factory=list)
    related_stacks: list[str] = field(default_factory=list)
    related_packs: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"PackInfo(id={self.id!r}, type={self.type.value!r})"


@dataclass
class SearchResult:
    """Resultado de uma busca de packs."""

    query: str
    packs: list[PackInfo]
    total: int
    search_type: str  # "semantic", "tag", "hint"

    def __repr__(self) -> str:
        return f"SearchResult(query={self.query!r}, total={self.total})"


# ============================================================================
# INDEX LOADING
# ============================================================================


def _load_unified_index() -> dict:
    """Carrega o indice unificado (com cache)."""
    global _unified_index_cache

    if _unified_index_cache is not None:
        return _unified_index_cache

    if not _UNIFIED_INDEX_PATH.exists():
        logger.warning(f"Unified index not found: {_UNIFIED_INDEX_PATH}")
        return {}

    with open(_UNIFIED_INDEX_PATH, encoding="utf-8") as f:
        _unified_index_cache = yaml.safe_load(f)

    logger.info(f"Loaded unified index from {_UNIFIED_INDEX_PATH}")
    return _unified_index_cache or {}


def _load_stack_index() -> dict:
    """Carrega o indice de stacks (com cache)."""
    global _stack_index_cache

    if _stack_index_cache is not None:
        return _stack_index_cache

    if not _STACK_INDEX_PATH.exists():
        logger.warning(f"Stack index not found: {_STACK_INDEX_PATH}")
        return {}

    with open(_STACK_INDEX_PATH, encoding="utf-8") as f:
        _stack_index_cache = yaml.safe_load(f)

    logger.info(f"Loaded stack index from {_STACK_INDEX_PATH}")
    return _stack_index_cache or {}


def _load_binding_index() -> dict:
    """Carrega o indice de bindings sprint-to-docs (com cache)."""
    global _binding_index_cache

    if _binding_index_cache is not None:
        return _binding_index_cache

    if not _BINDING_INDEX_PATH.exists():
        logger.warning(f"Binding index not found: {_BINDING_INDEX_PATH}")
        return {}

    with open(_BINDING_INDEX_PATH, encoding="utf-8") as f:
        _binding_index_cache = yaml.safe_load(f)

    logger.info(f"Loaded binding index from {_BINDING_INDEX_PATH}")
    return _binding_index_cache or {}


def clear_cache() -> None:
    """Limpa o cache dos indices."""
    global _unified_index_cache, _stack_index_cache, _binding_index_cache
    _unified_index_cache = None
    _stack_index_cache = None
    _binding_index_cache = None
    logger.info("Pack index cache cleared")


# ============================================================================
# SPRINT DOCUMENTATION BINDING - CRITICAL FOR DEVELOPMENT WORKFLOW
# ============================================================================


@dataclass
class SprintDocumentation:
    """All documentation relevant to a sprint."""

    sprint_id: str
    name: str
    feature_packs: list[str]
    user_journeys: list[str]
    stack_packs: list[str]
    specs: list[str]
    business_rules: list[str]

    def get_all_pack_ids(self) -> list[str]:
        """Get all pack IDs for loading."""
        return (
            self.feature_packs +
            self.user_journeys +
            self.stack_packs +
            self.specs
        )


def get_sprint_documentation(sprint_id: str) -> SprintDocumentation | None:
    """Get all documentation bound to a sprint.

    This is the CRITICAL function for the development workflow.
    When executing a sprint, the pipeline should call this to get
    ALL relevant documentation.

    Args:
        sprint_id: Sprint ID (e.g., "S22", "FK20", "VIS-A01")

    Returns:
        SprintDocumentation with all bound packs, or None if not found

    Example:
        >>> docs = get_sprint_documentation("S22")
        >>> print(docs.feature_packs)
        ["FEAT-OPP-001"]
        >>> print(docs.user_journeys)
        ["UJ-NEW-001", "UJ-NEW-027"]
    """
    binding_index = _load_binding_index()

    # Check HumanGR sprints
    if sprint_id.startswith("S") and sprint_id[1:].isdigit():
        bindings = binding_index.get("humangr_sprints", {}).get(sprint_id)
        if bindings:
            return SprintDocumentation(
                sprint_id=sprint_id,
                name=bindings.get("name", sprint_id),
                feature_packs=bindings.get("feature_packs", []),
                user_journeys=bindings.get("user_journeys", []),
                stack_packs=bindings.get("stack_packs", []),
                specs=bindings.get("specs", []),
                business_rules=bindings.get("business_rules", []),
            )

    # Check LEGACY_FOREKAST sprints
    if sprint_id.startswith("FK"):
        bindings = binding_index.get("legacy_forekast_sprints", {}).get(sprint_id)
        if bindings:
            return SprintDocumentation(
                sprint_id=sprint_id,
                name=bindings.get("name", sprint_id),
                feature_packs=bindings.get("feature_packs", []),
                user_journeys=bindings.get("user_journeys", []),
                stack_packs=bindings.get("stack_packs", []),
                specs=bindings.get("specs", []),
                business_rules=bindings.get("business_rules", []),
            )

    # Check LEGACY_VISIONARY sprints
    if sprint_id.startswith("VIS-"):
        bindings = binding_index.get("visionary_sprints", {}).get(sprint_id)
        if bindings:
            return SprintDocumentation(
                sprint_id=sprint_id,
                name=bindings.get("name", sprint_id),
                feature_packs=bindings.get("feature_packs", []),
                user_journeys=bindings.get("user_journeys", []),
                stack_packs=bindings.get("stack_packs", []),
                specs=bindings.get("specs", []),
                business_rules=bindings.get("business_rules", []),
            )

    return None


def load_feature_pack(feature_id: str) -> str | None:
    """Load a feature pack by ID.

    Args:
        feature_id: Feature pack ID (e.g., "FEAT-OPP-001", "FEAT-MOAT-006")

    Returns:
        Content of the feature pack YAML file, or None if not found
    """
    # Try with different naming patterns
    patterns = [
        f"{feature_id}.yaml",
        f"{feature_id}_*.yaml",
    ]

    for pattern in patterns:
        files = list(_FEATURE_PACKS_DIR.glob(pattern))
        if files:
            try:
                return files[0].read_text(encoding="utf-8")
            except Exception as e:
                logger.error(f"Error reading feature pack {feature_id}: {e}")
                return None

    logger.warning(f"Feature pack not found: {feature_id}")
    return None


def load_user_journey(journey_id: str) -> str | None:
    """Load a user journey by ID.

    Args:
        journey_id: Journey ID (e.g., "UJ-NEW-001")

    Returns:
        Content of the user journey YAML file, or None if not found
    """
    file_path = _USER_JOURNEYS_DIR / f"{journey_id}.yaml"
    if file_path.exists():
        try:
            return file_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"Error reading user journey {journey_id}: {e}")
            return None

    logger.warning(f"User journey not found: {journey_id}")
    return None


def load_specification(spec_id: str) -> str | None:
    """Load a specification by ID.

    Args:
        spec_id: Specification ID (e.g., "MQV_SPECIFICATION")

    Returns:
        Content of the specification YAML file, or None if not found
    """
    file_path = _SPECS_DIR / f"{spec_id}.yaml"
    if file_path.exists():
        try:
            return file_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"Error reading specification {spec_id}: {e}")
            return None

    logger.warning(f"Specification not found: {spec_id}")
    return None


def load_all_sprint_documentation(sprint_id: str) -> dict[str, str]:
    """Load ALL documentation for a sprint.

    This is the MAIN function for pipeline integration.
    It returns a dictionary with all loaded documentation.

    Args:
        sprint_id: Sprint ID

    Returns:
        Dictionary mapping pack ID to content

    Example:
        >>> docs = load_all_sprint_documentation("S22")
        >>> print(list(docs.keys()))
        ["FEAT-OPP-001", "UJ-NEW-001", "UJ-NEW-027", "MAC_CONTEXT", ...]
    """
    sprint_docs = get_sprint_documentation(sprint_id)
    if sprint_docs is None:
        logger.warning(f"No bindings found for sprint {sprint_id}")
        return {}

    loaded: dict[str, str] = {}

    # Load feature packs
    for fp_id in sprint_docs.feature_packs:
        content = load_feature_pack(fp_id)
        if content:
            loaded[fp_id] = content

    # Load user journeys
    for uj_id in sprint_docs.user_journeys:
        content = load_user_journey(uj_id)
        if content:
            loaded[uj_id] = content

    # Load stack packs
    for stack_id in sprint_docs.stack_packs:
        content = load_pack(stack_id)
        if content:
            loaded[stack_id] = content

    # Load specifications
    for spec_id in sprint_docs.specs:
        content = load_specification(spec_id)
        if content:
            loaded[spec_id] = content

    logger.info(f"Loaded {len(loaded)} documentation packs for sprint {sprint_id}")
    return loaded


def get_module_documentation(module_id: str) -> dict[str, str]:
    """Load all documentation for a HumanGR module.

    Args:
        module_id: Module ID (e.g., "MQV", "MAC", "TDS", "CGM")

    Returns:
        Dictionary mapping pack ID to content
    """
    binding_index = _load_binding_index()
    module_bindings = binding_index.get("module_bindings", {}).get(module_id)

    if not module_bindings:
        logger.warning(f"No bindings found for module {module_id}")
        return {}

    loaded: dict[str, str] = {}

    # Load all bound documentation
    for fp_id in module_bindings.get("feature_packs", []):
        content = load_feature_pack(fp_id)
        if content:
            loaded[fp_id] = content

    for uj_id in module_bindings.get("user_journeys", []):
        content = load_user_journey(uj_id)
        if content:
            loaded[uj_id] = content

    for stack_id in module_bindings.get("stack_packs", []):
        content = load_pack(stack_id)
        if content:
            loaded[stack_id] = content

    for spec_id in module_bindings.get("specs", []):
        content = load_specification(spec_id)
        if content:
            loaded[spec_id] = content

    return loaded


# ============================================================================
# PACK RESOLUTION
# ============================================================================


def _resolve_pack_path(pack_type: PackType, filename: str) -> Path:
    """Resolve o path completo de um pack."""
    index = _load_unified_index()
    base_paths = index.get("metadata", {}).get("base_paths", {})

    type_to_key = {
        PackType.SPRINT: "sprints",
        PackType.PRODUCT: "products",
        PackType.STACK: "stacks",
        PackType.THEORY: "theory",
        PackType.GATE: "gates",
        PackType.PROTOCOL: "protocols",
        PackType.QUANTUM_LEAP: "quantum_leap",
    }

    key = type_to_key.get(pack_type, "sprints")
    base_path = base_paths.get(key, "context_packs/")

    return _PROJECT_ROOT / base_path / filename


def _build_pack_info(pack_data: dict, pack_type: PackType) -> PackInfo:
    """Constroi um PackInfo a partir de dados do indice."""
    pack_id = pack_data.get("id", "")
    filename = pack_data.get("file", "")
    path = _resolve_pack_path(pack_type, filename)

    return PackInfo(
        id=pack_id,
        type=pack_type,
        name=pack_data.get("name", pack_id),
        file=filename,
        path=path,
        tags=pack_data.get("tags", []),
        summary=pack_data.get("summary", ""),
        product=pack_data.get("product"),
        related_sprints=pack_data.get("related_sprints", []),
        related_stacks=pack_data.get("related_stacks", []),
        related_packs=pack_data.get("related_packs", []),
    )


# ============================================================================
# DISCOVERY FUNCTIONS
# ============================================================================


def discover_packs(
    query: str,
    pack_types: list[PackType] | None = None,
    max_results: int = 10,
) -> SearchResult:
    """Descobre packs relevantes para uma query.

    Usa uma combinacao de:
    1. Search hints pre-definidos (mais rapido, mais preciso)
    2. Busca por tags
    3. Busca semantica via Qdrant (se disponivel)

    Args:
        query: Texto de busca (ex: "como implementar FSM", "agents", "qdrant")
        pack_types: Filtrar por tipos de pack (opcional)
        max_results: Numero maximo de resultados

    Returns:
        SearchResult com lista de PackInfo ordenada por relevancia

    Example:
        >>> result = discover_packs("FSM")
        >>> print(result.packs[0].id)
        "S01"
    """
    index = _load_unified_index()
    query_lower = query.lower().strip()
    found_packs: list[PackInfo] = []
    search_type = "hint"

    # 1. Tentar search hints primeiro (mais preciso)
    hints = index.get("search_hints", {}).get("hints", {})
    for hint_key, hint_data in hints.items():
        if hint_key in query_lower or query_lower in hint_key:
            pack_ids = hint_data.get("packs", [])
            for pack_id in pack_ids:
                pack_info = get_pack_info(pack_id)
                if pack_info and pack_info not in found_packs:
                    if pack_types is None or pack_info.type in pack_types:
                        found_packs.append(pack_info)

    # 2. Se nao encontrou por hints, buscar por tags
    if not found_packs:
        search_type = "tag"
        found_packs = _search_by_tags(query_lower, pack_types)

    # 3. Se ainda nao encontrou, tentar busca semantica
    if not found_packs:
        search_type = "semantic"
        found_packs = _search_semantic(query, pack_types, max_results)

    # Limitar resultados
    found_packs = found_packs[:max_results]

    return SearchResult(
        query=query,
        packs=found_packs,
        total=len(found_packs),
        search_type=search_type,
    )


def _search_by_tags(query: str, pack_types: list[PackType] | None) -> list[PackInfo]:
    """Busca packs por tags."""
    index = _load_unified_index()
    found: list[PackInfo] = []
    query_words = set(query.lower().split())

    # Buscar em sprint_packs
    if pack_types is None or PackType.SPRINT in pack_types:
        for pack_data in index.get("sprint_packs", {}).get("packs", []):
            tags = [t.lower() for t in pack_data.get("tags", [])]
            if query_words & set(tags):
                found.append(_build_pack_info(pack_data, PackType.SPRINT))

    # Buscar em product_packs
    if pack_types is None or PackType.PRODUCT in pack_types:
        for pack_data in index.get("product_packs", {}).get("packs", []):
            tags = [t.lower() for t in pack_data.get("tags", [])]
            if query_words & set(tags):
                found.append(_build_pack_info(pack_data, PackType.PRODUCT))

    # Buscar em theory_packs
    if pack_types is None or PackType.THEORY in pack_types:
        for pack_data in index.get("theory_packs", {}).get("packs", []):
            tags = [t.lower() for t in pack_data.get("tags", [])]
            if query_words & set(tags):
                found.append(_build_pack_info(pack_data, PackType.THEORY))

    # Buscar em gate_packs
    if pack_types is None or PackType.GATE in pack_types:
        for pack_data in index.get("gate_packs", {}).get("packs", []):
            tags = [t.lower() for t in pack_data.get("tags", [])]
            if query_words & set(tags):
                found.append(_build_pack_info(pack_data, PackType.GATE))

    # Buscar em quantum_leap_packs
    if pack_types is None or PackType.QUANTUM_LEAP in pack_types:
        for pack_data in index.get("quantum_leap_packs", {}).get("packs", []):
            tags = [t.lower() for t in pack_data.get("tags", [])]
            if query_words & set(tags):
                found.append(_build_pack_info(pack_data, PackType.QUANTUM_LEAP))

    return found


def _search_semantic(
    query: str,
    pack_types: list[PackType] | None,
    max_results: int,
) -> list[PackInfo]:
    """Busca semantica via Qdrant.

    Usa Ollama para gerar embeddings e Qdrant para buscar.
    A collection stack_packs contem os 75 stack packs indexados.
    """
    try:
        from pipeline.qdrant_client import get_qdrant_client
        from pipeline.ollama_client import get_ollama_client

        qdrant = get_qdrant_client()
        ollama = get_ollama_client()

        if not qdrant.is_available():
            logger.debug("Qdrant not available for semantic search")
            return []

        if not ollama.is_available():
            logger.debug("Ollama not available for semantic search")
            return []

        # Gerar embedding da query
        try:
            query_vector = ollama.embed(query)
        except Exception as e:
            logger.debug(f"Failed to embed query: {e}")
            return []

        # Mapear tipos para collections (usando nomes sem prefixo - o client adiciona)
        collections_to_search = []
        if pack_types is None:
            # Buscar em todas as collections disponiveis - INCLUDING NEW TYPES
            collections_to_search = [
                ("stack_packs", PackType.STACK),
                ("context_packs", PackType.SPRINT),
                ("feature_packs", PackType.FEATURE),
                ("user_journeys", PackType.USER_JOURNEY),
                ("specs", PackType.SPECIFICATION),
            ]
        else:
            type_to_collection = {
                PackType.STACK: "stack_packs",
                PackType.SPRINT: "context_packs",
                PackType.PRODUCT: "product_packs",
                PackType.THEORY: "theory_packs",
                PackType.GATE: "gate_packs",
                PackType.QUANTUM_LEAP: "quantum_leap_packs",
                # NEW: Documentation pack collections
                PackType.FEATURE: "feature_packs",
                PackType.USER_JOURNEY: "user_journeys",
                PackType.SPECIFICATION: "specs",
                PackType.BUSINESS_RULES: "business_rules",
                PackType.ALIGNMENT: "alignment",
                PackType.LEGACY_FOREKAST_SPRINT: "legacy_forekast_roadmap",
                PackType.LEGACY_VISIONARY_SPRINT: "legacy_visionary_roadmap",
            }
            for pt in pack_types:
                if pt in type_to_collection:
                    collections_to_search.append((type_to_collection[pt], pt))

        found: list[PackInfo] = []

        for collection_name, pack_type in collections_to_search:
            try:
                # Verificar se collection existe
                if not qdrant.collection_exists(collection_name):
                    logger.debug(f"Collection {collection_name} does not exist")
                    continue

                # Buscar por similaridade
                results = qdrant.search_similar(
                    collection=collection_name,
                    query_vector=query_vector,
                    top_k=max_results,
                )

                for result in results:
                    # Extrair pack_id do resultado
                    pack_id = result.metadata.get("stack_id") or result.metadata.get("original_id", result.id)
                    # Limpar prefixo "stack_" se presente
                    if pack_id.startswith("stack_"):
                        pack_id = pack_id[6:]

                    pack_info = get_pack_info(pack_id)
                    if pack_info and pack_info not in found:
                        found.append(pack_info)

            except Exception as e:
                logger.debug(f"Error searching {collection_name}: {e}")
                continue

        return found[:max_results]

    except ImportError as e:
        logger.debug(f"Import error for semantic search: {e}")
        return []
    except Exception as e:
        logger.debug(f"Semantic search failed: {e}")
        return []


def get_pack_info(pack_id: str) -> PackInfo | None:
    """Obtem informacoes de um pack pelo ID.

    Args:
        pack_id: ID do pack (ex: "S01", "FOUNDATION", "GOT", "G0")

    Returns:
        PackInfo ou None se nao encontrado

    Example:
        >>> info = get_pack_info("S01")
        >>> print(info.name)
        "states-fsm"
    """
    index = _load_unified_index()
    pack_id_upper = pack_id.upper()

    # Verificar sprint_packs
    for pack_data in index.get("sprint_packs", {}).get("packs", []):
        if pack_data.get("id", "").upper() == pack_id_upper:
            return _build_pack_info(pack_data, PackType.SPRINT)

    # Verificar product_packs
    for pack_data in index.get("product_packs", {}).get("packs", []):
        if pack_data.get("id", "").upper() == pack_id_upper:
            return _build_pack_info(pack_data, PackType.PRODUCT)

    # Verificar theory_packs
    for pack_data in index.get("theory_packs", {}).get("packs", []):
        if pack_data.get("id", "").upper() == pack_id_upper:
            return _build_pack_info(pack_data, PackType.THEORY)

    # Verificar gate_packs
    for pack_data in index.get("gate_packs", {}).get("packs", []):
        if pack_data.get("id", "").upper() == pack_id_upper:
            return _build_pack_info(pack_data, PackType.GATE)

    # Verificar quantum_leap_packs
    for pack_data in index.get("quantum_leap_packs", {}).get("packs", []):
        if pack_data.get("id", "").upper() == pack_id_upper:
            return _build_pack_info(pack_data, PackType.QUANTUM_LEAP)

    # Verificar stack_packs no STACK_MASTER_INDEX
    stack_index = _load_stack_index()
    for category in stack_index.values():
        if isinstance(category, dict) and "stacks" in category:
            for stack_data in category["stacks"]:
                if stack_data.get("id", "").upper() == pack_id_upper:
                    return PackInfo(
                        id=stack_data["id"],
                        type=PackType.STACK,
                        name=stack_data.get("name", stack_data["id"]),
                        file=f"{stack_data['id']}_CONTEXT.md",
                        path=_CONTEXT_PACKS / "stack_packs" / f"{stack_data['id']}_CONTEXT.md",
                        tags=stack_data.get("usage", []),
                        summary=stack_data.get("usage", [""])[0] if stack_data.get("usage") else "",
                        related_sprints=stack_data.get("sprints", []),
                        product=stack_data.get("products", [""])[0] if stack_data.get("products") else None,
                    )

    return None


def load_pack(pack_id: str) -> str | None:
    """Carrega o conteudo de um pack.

    Args:
        pack_id: ID do pack (ex: "S01", "FOUNDATION")

    Returns:
        Conteudo do arquivo ou None se nao encontrado

    Example:
        >>> content = load_pack("S01")
        >>> print(content[:100])
        "# S01_CONTEXT.md..."
    """
    pack_info = get_pack_info(pack_id)
    if pack_info is None:
        logger.warning(f"Pack not found: {pack_id}")
        return None

    if not pack_info.path.exists():
        logger.warning(f"Pack file not found: {pack_info.path}")
        return None

    try:
        return pack_info.path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error(f"Error reading pack {pack_id}: {e}")
        return None


def get_related_packs(pack_id: str) -> list[PackInfo]:
    """Obtem packs relacionados a um pack.

    Args:
        pack_id: ID do pack

    Returns:
        Lista de PackInfo relacionados

    Example:
        >>> related = get_related_packs("S01")
        >>> [p.id for p in related]
        ["FOUNDATION", "CONSTITUTION", "MQV"]
    """
    pack_info = get_pack_info(pack_id)
    if pack_info is None:
        return []

    related: list[PackInfo] = []

    # Adicionar produto pai (se for sprint)
    if pack_info.product:
        product_info = get_pack_info(pack_info.product)
        if product_info:
            related.append(product_info)

    # Adicionar stacks relacionadas
    for stack_id in pack_info.related_stacks:
        stack_info = get_pack_info(stack_id)
        if stack_info:
            related.append(stack_info)

    # Adicionar sprints relacionados
    for sprint_id in pack_info.related_sprints:
        sprint_info = get_pack_info(sprint_id)
        if sprint_info and sprint_info.id != pack_id:
            related.append(sprint_info)

    # Adicionar packs relacionados genericos
    for related_id in pack_info.related_packs:
        related_info = get_pack_info(related_id)
        if related_info and related_info not in related:
            related.append(related_info)

    return related


def list_packs(
    pack_type: PackType | None = None,
    product: str | None = None,
    tags: list[str] | None = None,
) -> list[PackInfo]:
    """Lista packs com filtros opcionais.

    Args:
        pack_type: Filtrar por tipo
        product: Filtrar por produto (ex: "FOUNDATION")
        tags: Filtrar por tags

    Returns:
        Lista de PackInfo

    Example:
        >>> sprints = list_packs(pack_type=PackType.SPRINT, product="FOUNDATION")
        >>> len(sprints)
        8
    """
    index = _load_unified_index()
    packs: list[PackInfo] = []

    # Coletar todos os packs
    if pack_type is None or pack_type == PackType.SPRINT:
        for pack_data in index.get("sprint_packs", {}).get("packs", []):
            packs.append(_build_pack_info(pack_data, PackType.SPRINT))

    if pack_type is None or pack_type == PackType.PRODUCT:
        for pack_data in index.get("product_packs", {}).get("packs", []):
            packs.append(_build_pack_info(pack_data, PackType.PRODUCT))

    if pack_type is None or pack_type == PackType.THEORY:
        for pack_data in index.get("theory_packs", {}).get("packs", []):
            packs.append(_build_pack_info(pack_data, PackType.THEORY))

    if pack_type is None or pack_type == PackType.GATE:
        for pack_data in index.get("gate_packs", {}).get("packs", []):
            packs.append(_build_pack_info(pack_data, PackType.GATE))

    if pack_type is None or pack_type == PackType.QUANTUM_LEAP:
        for pack_data in index.get("quantum_leap_packs", {}).get("packs", []):
            packs.append(_build_pack_info(pack_data, PackType.QUANTUM_LEAP))

    # Protocol packs do UNIFIED_PACK_INDEX.yaml
    if pack_type is None or pack_type == PackType.PROTOCOL:
        for pack_data in index.get("protocol_packs", {}).get("packs", []):
            packs.append(PackInfo(
                id=pack_data.get("id", ""),
                type=PackType.PROTOCOL,
                name=pack_data.get("id", ""),
                file=pack_data.get("file", ""),
                path=_CONTEXT_PACKS / "protocols" / pack_data.get("file", ""),
                tags=pack_data.get("tags", []),
                summary=pack_data.get("summary", ""),
            ))

    # Stack packs do STACK_MASTER_INDEX.yaml
    if pack_type is None or pack_type == PackType.STACK:
        stack_index = _load_stack_index()
        for category_name, category_data in stack_index.items():
            if isinstance(category_data, dict) and "stacks" in category_data:
                for stack_data in category_data["stacks"]:
                    stack_id = stack_data.get("id", "")
                    # Skip deprecated stacks
                    if stack_data.get("status") == "DEPRECATED":
                        continue
                    packs.append(PackInfo(
                        id=stack_id,
                        type=PackType.STACK,
                        name=stack_data.get("name", stack_id),
                        file=f"{stack_id}_CONTEXT.md",
                        path=_CONTEXT_PACKS / "stack_packs" / f"{stack_id}_CONTEXT.md",
                        tags=stack_data.get("usage", []),
                        summary=stack_data.get("usage", [""])[0] if stack_data.get("usage") else "",
                        related_sprints=stack_data.get("sprints", []),
                        product=stack_data.get("products", [""])[0] if stack_data.get("products") else None,
                    ))

    # Feature packs - scan directory directly
    if pack_type is None or pack_type == PackType.FEATURE:
        if _FEATURE_PACKS_DIR.exists():
            for fp_file in _FEATURE_PACKS_DIR.glob("FEAT-*.yaml"):
                feat_id = fp_file.stem
                try:
                    with open(fp_file, encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                    packs.append(PackInfo(
                        id=data.get("id", feat_id),
                        type=PackType.FEATURE,
                        name=data.get("name", feat_id),
                        file=fp_file.name,
                        path=fp_file,
                        tags=data.get("tags", []),
                        summary=data.get("summary", ""),
                        product=data.get("category", None),
                        related_sprints=data.get("implementing_sprints", []),
                    ))
                except Exception as e:
                    logger.debug(f"Error loading feature pack {feat_id}: {e}")

    # User journey packs - scan directory directly
    if pack_type is None or pack_type == PackType.USER_JOURNEY:
        if _USER_JOURNEYS_DIR.exists():
            for uj_file in _USER_JOURNEYS_DIR.glob("UJ-*.yaml"):
                uj_id = uj_file.stem
                try:
                    with open(uj_file, encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                    packs.append(PackInfo(
                        id=data.get("id", uj_id),
                        type=PackType.USER_JOURNEY,
                        name=data.get("name", uj_id),
                        file=uj_file.name,
                        path=uj_file,
                        tags=data.get("tags", []),
                        summary=data.get("summary", ""),
                        related_sprints=data.get("implementing_sprints", []),
                    ))
                except Exception as e:
                    logger.debug(f"Error loading user journey {uj_id}: {e}")

    # Legacy Forekast sprints (FK-series) - filter from sprint_packs
    if pack_type == PackType.LEGACY_FOREKAST_SPRINT:
        for pack_data in index.get("sprint_packs", {}).get("packs", []):
            pack_id = pack_data.get("id", "")
            if pack_id.startswith("FK") and not pack_id.startswith("FK-ADMIN"):
                packs.append(_build_pack_info(pack_data, PackType.LEGACY_FOREKAST_SPRINT))

    # Legacy Visionary sprints (VIS-series) - from visionary_packs section
    if pack_type is None or pack_type == PackType.LEGACY_VISIONARY_SPRINT:
        for pack_data in index.get("visionary_packs", {}).get("packs", []):
            pack_id = pack_data.get("id", "")
            if pack_id.startswith("VIS-"):
                packs.append(_build_pack_info(pack_data, PackType.LEGACY_VISIONARY_SPRINT))

    # Aplicar filtros
    if product:
        product_upper = product.upper()
        packs = [p for p in packs if p.product and p.product.upper() == product_upper]

    if tags:
        tags_lower = [t.lower() for t in tags]
        packs = [p for p in packs if any(t.lower() in tags_lower for t in p.tags)]

    return packs


def get_sprint_packs_for_product(product_id: str) -> list[PackInfo]:
    """Obtem todos os sprint packs de um produto.

    Args:
        product_id: ID do produto (ex: "FOUNDATION")

    Returns:
        Lista de PackInfo ordenada por sprint

    Example:
        >>> sprints = get_sprint_packs_for_product("FOUNDATION")
        >>> [s.id for s in sprints]
        ["S00", "S01", "S02", "S03", "S04", "S05", "S05H", "S06", "S07"]
    """
    return list_packs(pack_type=PackType.SPRINT, product=product_id)


def get_packs_for_stack(stack_id: str) -> list[PackInfo]:
    """Obtem packs relacionados a uma stack.

    Args:
        stack_id: ID da stack (ex: "QDRANT", "GOT")

    Returns:
        Lista de PackInfo que usam essa stack

    Example:
        >>> packs = get_packs_for_stack("GOT")
        >>> [p.id for p in packs]
        ["S18", "S19", "S20"]
    """
    stack_id_upper = stack_id.upper()
    index = _load_unified_index()
    found: list[PackInfo] = []

    for pack_data in index.get("sprint_packs", {}).get("packs", []):
        related_stacks = [s.upper() for s in pack_data.get("related_stacks", [])]
        if stack_id_upper in related_stacks:
            found.append(_build_pack_info(pack_data, PackType.SPRINT))

    return found


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def quick_search(query: str) -> list[str]:
    """Busca rapida - retorna apenas IDs dos packs.

    Args:
        query: Texto de busca

    Returns:
        Lista de IDs de packs

    Example:
        >>> quick_search("FSM")
        ["S01", "CONSTITUTION", "MQV"]
    """
    result = discover_packs(query)
    return [p.id for p in result.packs]


def get_pack_content_summary(pack_id: str, max_lines: int = 50) -> str:
    """Obtem um resumo do conteudo de um pack.

    Args:
        pack_id: ID do pack
        max_lines: Numero maximo de linhas

    Returns:
        Primeiras N linhas do pack ou string vazia

    Example:
        >>> summary = get_pack_content_summary("S01", 10)
        >>> print(summary)
    """
    content = load_pack(pack_id)
    if content is None:
        return ""

    lines = content.split("\n")[:max_lines]
    return "\n".join(lines)


def print_pack_tree(product_id: str | None = None) -> None:
    """Imprime arvore de packs (util para debug).

    Args:
        product_id: Filtrar por produto (opcional)
    """
    index = _load_unified_index()

    if product_id:
        print(f"\n=== Packs for {product_id} ===")
        sprints = get_sprint_packs_for_product(product_id)
        for s in sprints:
            print(f"  {s.id}: {s.name}")
            if s.related_stacks:
                print(f"    Stacks: {', '.join(s.related_stacks)}")
    else:
        print("\n=== Pack Tree ===")
        for pack_data in index.get("product_packs", {}).get("packs", []):
            product = pack_data["id"]
            print(f"\n{product} ({pack_data.get('name', '')})")
            sprints = pack_data.get("sprints", [])
            for sprint_id in sprints:
                sprint_info = get_pack_info(sprint_id)
                if sprint_info:
                    print(f"  ├── {sprint_id}: {sprint_info.name}")


# ============================================================================
# CLI INTERFACE
# ============================================================================


def main() -> None:
    """CLI para testar pack discovery."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pack_discovery.py <command> [args]")
        print("")
        print("Commands:")
        print("  search <query>     - Search for packs")
        print("  info <pack_id>     - Get pack info")
        print("  load <pack_id>     - Load pack content")
        print("  related <pack_id>  - Get related packs")
        print("  list [type]        - List packs (sprint/product/stack/theory/gate)")
        print("  tree [product]     - Print pack tree")
        return

    command = sys.argv[1]

    if command == "search" and len(sys.argv) > 2:
        query = " ".join(sys.argv[2:])
        result = discover_packs(query)
        print(f"Query: {result.query}")
        print(f"Search type: {result.search_type}")
        print(f"Found: {result.total} packs")
        for pack in result.packs:
            print(f"  - {pack.id} ({pack.type.value}): {pack.summary[:60]}...")

    elif command == "info" and len(sys.argv) > 2:
        pack_id = sys.argv[2]
        info = get_pack_info(pack_id)
        if info:
            print(f"ID: {info.id}")
            print(f"Type: {info.type.value}")
            print(f"Name: {info.name}")
            print(f"File: {info.file}")
            print(f"Path: {info.path}")
            print(f"Tags: {', '.join(info.tags)}")
            print(f"Summary: {info.summary}")
            print(f"Product: {info.product}")
            print(f"Related stacks: {', '.join(info.related_stacks)}")
        else:
            print(f"Pack not found: {pack_id}")

    elif command == "load" and len(sys.argv) > 2:
        pack_id = sys.argv[2]
        content = load_pack(pack_id)
        if content:
            print(content[:2000])
            if len(content) > 2000:
                print(f"\n... ({len(content)} chars total)")
        else:
            print(f"Pack not found: {pack_id}")

    elif command == "related" and len(sys.argv) > 2:
        pack_id = sys.argv[2]
        related = get_related_packs(pack_id)
        print(f"Related to {pack_id}:")
        for pack in related:
            print(f"  - {pack.id} ({pack.type.value})")

    elif command == "list":
        pack_type = None
        if len(sys.argv) > 2:
            type_map = {
                "sprint": PackType.SPRINT,
                "product": PackType.PRODUCT,
                "stack": PackType.STACK,
                "theory": PackType.THEORY,
                "gate": PackType.GATE,
            }
            pack_type = type_map.get(sys.argv[2].lower())

        packs = list_packs(pack_type=pack_type)
        print(f"Found {len(packs)} packs:")
        for pack in packs:
            print(f"  - {pack.id} ({pack.type.value})")

    elif command == "tree":
        product = sys.argv[2] if len(sys.argv) > 2 else None
        print_pack_tree(product)

    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
