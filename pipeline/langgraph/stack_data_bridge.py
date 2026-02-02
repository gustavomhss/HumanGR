"""Stack Data Bridge - Conecta dados das stacks aos consumidores.

Este módulo resolve o problema de dados coletados mas não usados.
As stacks de integração (partial_stacks_integration.py) coletam dados valiosos
mas esses dados não chegavam aos consumidores (daemon, crew, gates).

O StackDataBridge:
1. Agrega RAG context de múltiplas fontes
2. Reordena tasks por prioridade
3. Constrói warnings consolidados
4. Formata planos históricos como exemplos
5. Injeta tudo nos consumidores corretos

Invariantes:
- INV-BRIDGE-001: Backward compatible - sistema funciona sem bridge
- INV-BRIDGE-002: Data integrity - não modifica dados originais
- INV-BRIDGE-003: Observability - toda injeção é rastreada
- INV-BRIDGE-004: No silent failures - warnings logados

Author: Pipeline Team
Version: 1.0.0 (2026-01-30)
Audit Reference: docs/pipeline/STACK_USAGE_AUDIT_2026_01_30.md
"""

from __future__ import annotations

import logging
import hashlib
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class EnrichedContext:
    """Contexto enriquecido com dados das stacks.

    Contém todos os dados agregados que serão injetados nos consumidores.
    """
    rag_documents: List[Dict[str, Any]] = field(default_factory=list)
    priority_order: List[str] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    similar_plans: List[Dict[str, Any]] = field(default_factory=list)
    fallback_hints: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            "rag_documents": self.rag_documents,
            "priority_order": self.priority_order,
            "warnings": self.warnings,
            "similar_plans": self.similar_plans,
            "fallback_hints": self.fallback_hints,
        }

    def is_empty(self) -> bool:
        """Verifica se não há dados enriquecidos."""
        return (
            not self.rag_documents
            and not self.priority_order
            and not self.warnings
            and not self.similar_plans
            and not self.fallback_hints
        )


@dataclass
class BridgeInjectionResult:
    """Resultado de uma injeção do bridge.

    Usado para tracking e observabilidade (INV-BRIDGE-003).
    """
    success: bool
    injected_keys: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    rag_documents_count: int = 0
    warnings_count: int = 0
    tasks_reordered: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            "success": self.success,
            "injected_keys": self.injected_keys,
            "errors": self.errors,
            "timestamp": self.timestamp,
            "rag_documents_count": self.rag_documents_count,
            "warnings_count": self.warnings_count,
            "tasks_reordered": self.tasks_reordered,
        }


# =============================================================================
# CONSTANTS
# =============================================================================

# State keys que contêm RAG context
# P0-FIX-2026-02-01: Added corrective_rag_context and graphrag_context
RAG_CONTEXT_KEYS = [
    "self_rag_context",
    "memo_rag_context",
    "raptor_rag_context",
    "hybrid_search_context",
    "colbert_context",
    "corrective_rag_context",  # P0-FIX: Added Corrective RAG
    "graphrag_context",        # P0-FIX: Added GraphRAG
]

# Prioridade das fontes RAG (maior = melhor)
# P0-FIX-2026-02-01: Added corrective_rag and graphrag priority
RAG_SOURCE_PRIORITY = {
    "graphrag": 6,         # Graph-enhanced with reasoning chains (highest)
    "self_rag": 5,         # Self-reflective, alta qualidade
    "corrective_rag": 4.5, # Self-correcting, document quality scoring
    "qdrant_hybrid": 4,    # Hybrid search
    "colbert": 3,          # Token-level
    "memo_rag": 2,         # Memory-augmented
    "raptor_rag": 1,       # Hierarchical
}

# State keys que contêm warnings
WARNING_KEYS = [
    "contradictions",
    "ambiguity_warnings",
    "consistency_issues",
    "detected_fabrications",
]

# Severidade de warnings
WARNING_SEVERITY = {
    "contradictions": "BLOCKING",
    "detected_fabrications": "BLOCKING",
    "ambiguity_warnings": "WARNING",
    "consistency_issues": "WARNING",
}

# Prefixo para keys injetadas pelo bridge
BRIDGE_PREFIX = "_bridge_"


# =============================================================================
# STACK DATA BRIDGE CLASS
# =============================================================================

class StackDataBridge:
    """Bridge que conecta dados das stacks aos consumidores.

    Este é o componente central que resolve o problema de dados coletados
    mas não usados. Ele agrega, formata e injeta dados nos lugares certos.

    Usage:
        bridge = StackDataBridge()

        # Enriquecer context_pack antes de passar para crew
        enriched = bridge.enrich_context_pack(state, context_pack)

        # Reordenar tasks por prioridade
        ordered_tasks = bridge.enrich_granular_tasks(state, tasks)

        # Construir instruction enriquecida para daemon
        enriched_instruction = bridge.build_enriched_instruction(
            base_instruction, state
        )
    """

    def __init__(self, top_k_rag: int = 5, max_similar_plans: int = 3):
        """Inicializa o bridge.

        Args:
            top_k_rag: Número máximo de documentos RAG a incluir
            max_similar_plans: Número máximo de planos históricos a incluir
        """
        self.top_k_rag = top_k_rag
        self.max_similar_plans = max_similar_plans
        self._injection_history: List[BridgeInjectionResult] = []

    # =========================================================================
    # RAG AGGREGATION
    # =========================================================================

    def aggregate_rag_context(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Agrega RAG context de múltiplas fontes.

        Coleta documentos de todas as stacks RAG, deduplica e rankeia.

        Args:
            state: Pipeline state com dados das stacks

        Returns:
            Lista de documentos agregados e rankeados
        """
        all_docs: List[Dict[str, Any]] = []
        sources_found: List[str] = []

        for key in RAG_CONTEXT_KEYS:
            docs = state.get(key, [])
            if docs:
                # Identificar fonte
                source = key.replace("_context", "").replace("_", "")
                sources_found.append(key)

                # Adicionar source tag se não existir
                for doc in docs:
                    if isinstance(doc, dict):
                        doc_copy = dict(doc)
                        if "_source" not in doc_copy:
                            doc_copy["_source"] = source
                        all_docs.append(doc_copy)
                    elif isinstance(doc, str):
                        # Documento é string simples
                        all_docs.append({
                            "content": doc,
                            "_source": source,
                        })

        if not all_docs:
            # INV-BRIDGE-004: Log warning, não silenciar
            logger.debug("StackDataBridge: No RAG context found in state")
            return []

        logger.debug(f"StackDataBridge: Aggregated {len(all_docs)} RAG docs from {sources_found}")

        # Deduplicate
        deduped = self.deduplicate_documents(all_docs)

        # Rank and limit
        ranked = self.rank_and_limit(deduped, self.top_k_rag)

        return ranked

    def deduplicate_documents(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove documentos duplicados.

        Usa hash do conteúdo para identificar duplicatas.
        Mantém documento com maior confidence se duplicata encontrada.

        Args:
            docs: Lista de documentos

        Returns:
            Lista sem duplicatas
        """
        if not docs:
            return []

        seen_hashes: Dict[str, Dict[str, Any]] = {}

        for doc in docs:
            # Extrair conteúdo para hash
            content = doc.get("content", "") or doc.get("text", "") or str(doc)
            content_hash = hashlib.md5(content.encode()).hexdigest()[:16]

            if content_hash in seen_hashes:
                # Duplicata encontrada - manter com maior confidence
                existing = seen_hashes[content_hash]
                existing_conf = existing.get("confidence", 0) or existing.get("score", 0)
                new_conf = doc.get("confidence", 0) or doc.get("score", 0)

                if new_conf > existing_conf:
                    seen_hashes[content_hash] = doc
            else:
                seen_hashes[content_hash] = doc

        result = list(seen_hashes.values())

        if len(result) < len(docs):
            logger.debug(
                f"StackDataBridge: Deduplicated {len(docs)} -> {len(result)} documents"
            )

        return result

    def rank_and_limit(
        self,
        docs: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Rankeia documentos e limita a top-k.

        Ranking por:
        1. Confidence/score (maior = melhor)
        2. Source priority (self_rag > qdrant > outros)

        Args:
            docs: Lista de documentos
            top_k: Número máximo de documentos

        Returns:
            Top-k documentos rankeados
        """
        if not docs:
            return []

        def get_score(doc: Dict[str, Any]) -> Tuple[float, int]:
            """Calcula score para ranking."""
            # Confidence ou score do documento
            conf = doc.get("confidence", 0) or doc.get("score", 0) or 0.5

            # Prioridade da fonte
            source = doc.get("_source", "unknown")
            source_priority = RAG_SOURCE_PRIORITY.get(source, 0)

            return (conf, source_priority)

        # Ordenar por score (maior primeiro)
        sorted_docs = sorted(docs, key=get_score, reverse=True)

        return sorted_docs[:top_k]

    # =========================================================================
    # PRIORITY MERGER
    # =========================================================================

    def reorder_by_priority(
        self,
        tasks: List[Dict[str, Any]],
        priorities: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Reordena tasks conforme priorização MoSCoW.

        P0 (MUST) vem primeiro, depois P1 (SHOULD), P2 (COULD), P3 (WONT).

        Args:
            tasks: Lista de tasks original
            priorities: Dict com priorização (de prioritized_specs ou priority_matrix_result)

        Returns:
            Lista reordenada (CÓPIA - original não modificada)

        INV-BRIDGE-002: Não modifica lista original
        """
        if not tasks:
            return []

        if not priorities:
            # Sem prioridades - retorna cópia sem reordenar
            return deepcopy(tasks)

        # Criar cópia para não modificar original
        tasks_copy = deepcopy(tasks)

        # Extrair mapping de prioridade
        priority_map: Dict[str, int] = {}

        # Tentar diferentes formatos de priorities
        if isinstance(priorities, dict):
            # Formato: {"P0": ["task1", "task2"], "P1": [...]}
            for priority_level, items in priorities.items():
                if isinstance(priority_level, str) and priority_level.startswith("P"):
                    try:
                        level = int(priority_level[1:])
                        if isinstance(items, list):
                            for item in items:
                                if isinstance(item, dict):
                                    task_id = item.get("id", "") or item.get("deliverable", "")
                                elif isinstance(item, str):
                                    task_id = item
                                else:
                                    continue
                                priority_map[task_id] = level
                    except (ValueError, IndexError):
                        pass

            # Formato alternativo: lista de specs priorizados
            prioritized_list = priorities.get("prioritized_specs", [])
            if prioritized_list and isinstance(prioritized_list, list):
                for idx, spec in enumerate(prioritized_list):
                    spec_id = spec.get("id", "") if isinstance(spec, dict) else str(spec)
                    if spec_id:
                        # Assume P0 para os priorizados, ordem define sub-prioridade
                        priority_map[spec_id] = 0  # P0

        def get_priority(task: Dict[str, Any]) -> Tuple[int, int]:
            """Retorna (prioridade, índice original) para sorting."""
            task_id = task.get("id", "") or task.get("deliverable", "") or ""

            # Buscar prioridade
            priority = priority_map.get(task_id, 2)  # Default P2

            # Índice original como tiebreaker
            original_idx = tasks_copy.index(task) if task in tasks_copy else 999

            return (priority, original_idx)

        # Ordenar por prioridade
        sorted_tasks = sorted(tasks_copy, key=get_priority)

        # Verificar se houve reordenação
        if sorted_tasks != tasks_copy:
            logger.info(
                f"StackDataBridge: Reordered {len(tasks)} tasks by priority"
            )

        return sorted_tasks

    def preserve_dependencies(
        self,
        tasks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Garante que dependências são respeitadas após reordenação.

        Se task B depende de A, A deve vir antes de B mesmo que B tenha
        prioridade maior.

        Args:
            tasks: Lista de tasks (já reordenada por prioridade)

        Returns:
            Lista com dependências respeitadas
        """
        if not tasks:
            return []

        # Criar mapa de task_id -> task
        task_map: Dict[str, Dict[str, Any]] = {}
        for task in tasks:
            task_id = task.get("id", "") or task.get("deliverable", "")
            if task_id:
                task_map[task_id] = task

        # Identificar dependências
        dependencies: Dict[str, List[str]] = {}
        for task in tasks:
            task_id = task.get("id", "") or task.get("deliverable", "")
            deps = task.get("dependencies", []) or task.get("depends_on", [])
            if task_id and deps:
                dependencies[task_id] = deps

        if not dependencies:
            # Sem dependências - retorna como está
            return tasks

        # Topological sort simplificado
        result: List[Dict[str, Any]] = []
        added: Set[str] = set()

        def add_with_deps(task_id: str):
            """Adiciona task e suas dependências recursivamente."""
            if task_id in added:
                return

            # Adicionar dependências primeiro
            for dep_id in dependencies.get(task_id, []):
                if dep_id not in added and dep_id in task_map:
                    add_with_deps(dep_id)

            # Adicionar task
            if task_id in task_map:
                result.append(task_map[task_id])
                added.add(task_id)

        # Processar todas as tasks na ordem atual
        for task in tasks:
            task_id = task.get("id", "") or task.get("deliverable", "")
            if task_id:
                add_with_deps(task_id)
            else:
                # Task sem ID - adicionar no final
                result.append(task)

        return result

    # =========================================================================
    # WARNING BUILDER
    # =========================================================================

    def build_warnings(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Constrói lista consolidada de warnings.

        Coleta warnings de todas as fontes e classifica por severidade.

        Args:
            state: Pipeline state

        Returns:
            Lista de warnings ordenada por severidade (BLOCKING primeiro)
        """
        all_warnings: List[Dict[str, Any]] = []

        for key in WARNING_KEYS:
            items = state.get(key, [])
            if not items:
                continue

            severity = WARNING_SEVERITY.get(key, "INFO")

            if isinstance(items, list):
                for item in items:
                    warning = {
                        "source": key,
                        "severity": severity,
                        "message": str(item) if not isinstance(item, dict) else item.get("message", str(item)),
                        "details": item if isinstance(item, dict) else None,
                    }
                    all_warnings.append(warning)
            elif isinstance(items, dict):
                warning = {
                    "source": key,
                    "severity": severity,
                    "message": items.get("message", str(items)),
                    "details": items,
                }
                all_warnings.append(warning)

        if not all_warnings:
            return []

        # Ordenar por severidade (BLOCKING > WARNING > INFO)
        severity_order = {"BLOCKING": 0, "WARNING": 1, "INFO": 2}
        sorted_warnings = sorted(
            all_warnings,
            key=lambda w: severity_order.get(w["severity"], 99)
        )

        logger.debug(
            f"StackDataBridge: Built {len(sorted_warnings)} warnings "
            f"({sum(1 for w in sorted_warnings if w['severity'] == 'BLOCKING')} blocking)"
        )

        return sorted_warnings

    def format_warnings_for_prompt(self, warnings: List[Dict[str, Any]]) -> str:
        """Formata warnings como texto para inclusão em prompt.

        Args:
            warnings: Lista de warnings

        Returns:
            String formatada ou vazia se sem warnings
        """
        if not warnings:
            return ""

        lines = ["", "⚠️ ATENÇÃO - Issues Detectados:", ""]

        for warning in warnings:
            severity = warning.get("severity", "INFO")
            message = warning.get("message", "Unknown issue")
            source = warning.get("source", "unknown")

            if severity == "BLOCKING":
                emoji = "🚫"
            elif severity == "WARNING":
                emoji = "⚠️"
            else:
                emoji = "ℹ️"

            lines.append(f"{emoji} [{severity}] ({source}): {message}")

        lines.append("")

        return "\n".join(lines)

    # =========================================================================
    # SIMILAR PLANS FORMATTER
    # =========================================================================

    def format_similar_plans(
        self,
        state: Dict[str, Any],
        max_plans: int = 3
    ) -> str:
        """Formata planos históricos similares como exemplos.

        Args:
            state: Pipeline state
            max_plans: Número máximo de planos a incluir

        Returns:
            String formatada ou vazia se sem planos
        """
        # Tentar ambas as keys possíveis
        plans = (
            state.get("historical_similar_plans", [])
            or state.get("prefetched_similar_plans", [])
        )

        if not plans:
            return ""

        # Limitar quantidade
        plans = plans[:max_plans]

        lines = ["", "📚 REFERÊNCIAS HISTÓRICAS:", ""]

        for plan in plans:
            if isinstance(plan, dict):
                sprint_id = plan.get("sprint_id", plan.get("id", "Unknown"))
                objective = plan.get("objective", "N/A")
                result = plan.get("result", plan.get("outcome", "N/A"))
                similarity = plan.get("similarity", plan.get("score", 0))

                lines.append(f"Sprint {sprint_id} (similarity: {similarity:.2f}):")
                lines.append(f"  Objetivo: {objective[:100]}...")
                lines.append(f"  Resultado: {result}")
                lines.append("")
            elif isinstance(plan, str):
                lines.append(f"- {plan}")

        return "\n".join(lines)

    # =========================================================================
    # MAIN BRIDGE METHODS
    # =========================================================================

    def enrich_context_pack(
        self,
        state: Dict[str, Any],
        context_pack: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enriquece context_pack com dados das stacks.

        Este é o método principal para enriquecer o context que vai para o crew.
        Cria uma CÓPIA do context_pack e adiciona dados agregados.

        Args:
            state: Pipeline state com dados das stacks
            context_pack: Context pack original

        Returns:
            Context pack enriquecido (CÓPIA)

        INV-BRIDGE-002: Não modifica original
        """
        # Criar cópia profunda
        enriched = deepcopy(context_pack)

        # Agregar RAG context
        rag_docs = self.aggregate_rag_context(state)
        if rag_docs:
            enriched[f"{BRIDGE_PREFIX}rag_context"] = rag_docs
            enriched[f"{BRIDGE_PREFIX}rag_count"] = len(rag_docs)

        # Construir warnings
        warnings = self.build_warnings(state)
        if warnings:
            enriched[f"{BRIDGE_PREFIX}warnings"] = warnings
            enriched[f"{BRIDGE_PREFIX}warnings_text"] = self.format_warnings_for_prompt(warnings)

        # Formatar planos similares
        similar_plans_text = self.format_similar_plans(state)
        if similar_plans_text:
            enriched[f"{BRIDGE_PREFIX}similar_plans"] = similar_plans_text

        # Adicionar fallback hints se disponível
        fallback_plans = state.get("fallback_plans", [])
        if fallback_plans:
            enriched[f"{BRIDGE_PREFIX}fallback_available"] = True
            enriched[f"{BRIDGE_PREFIX}fallback_count"] = len(fallback_plans)

        logger.info(
            f"StackDataBridge: Enriched context_pack with "
            f"rag={len(rag_docs)}, warnings={len(warnings)}"
        )

        return enriched

    def enrich_granular_tasks(
        self,
        state: Dict[str, Any],
        tasks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Reordena granular_tasks por prioridade.

        Usa prioritized_specs ou priority_matrix_result do state para
        reordenar tasks. Preserva dependências.

        Args:
            state: Pipeline state
            tasks: Lista de tasks original

        Returns:
            Lista reordenada (CÓPIA)

        INV-BRIDGE-002: Não modifica original
        """
        # Buscar prioridades
        priorities = (
            state.get("priority_matrix_result")
            or state.get("prioritized_specs")
            or {}
        )

        # Reordenar por prioridade
        reordered = self.reorder_by_priority(tasks, priorities)

        # Preservar dependências
        final = self.preserve_dependencies(reordered)

        return final

    def build_enriched_instruction(
        self,
        base_instruction: str,
        state: Dict[str, Any]
    ) -> str:
        """Constrói instruction enriquecida para o daemon.

        Adiciona contexto RAG, warnings e exemplos históricos ao prompt.

        Args:
            base_instruction: Instruction original
            state: Pipeline state

        Returns:
            Instruction enriquecida ou original se nada para adicionar
        """
        additions: List[str] = []

        # RAG context
        rag_docs = self.aggregate_rag_context(state)
        if rag_docs:
            additions.append("\n📄 CONTEXTO RELEVANTE:")
            for doc in rag_docs[:3]:  # Limitar para não sobrecarregar
                content = doc.get("content", "") or doc.get("text", "")
                if content:
                    additions.append(f"- {content[:200]}...")
            additions.append("")

        # Warnings
        warnings = self.build_warnings(state)
        warnings_text = self.format_warnings_for_prompt(warnings)
        if warnings_text:
            additions.append(warnings_text)

        # Planos similares
        similar_text = self.format_similar_plans(state)
        if similar_text:
            additions.append(similar_text)

        if not additions:
            return base_instruction

        enriched = base_instruction + "\n" + "\n".join(additions)

        logger.debug(
            f"StackDataBridge: Enriched instruction "
            f"(original: {len(base_instruction)}, enriched: {len(enriched)})"
        )

        return enriched

    def record_injection(
        self,
        state: Dict[str, Any],
        injections: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Registra o que foi injetado para observabilidade.

        Adiciona tracking ao state para que possamos auditar o que o bridge fez.

        Args:
            state: Pipeline state
            injections: Dict descrevendo o que foi injetado

        Returns:
            State atualizado com registro de injeção

        INV-BRIDGE-003: Toda injeção rastreável
        """
        result = BridgeInjectionResult(
            success=True,
            injected_keys=list(injections.keys()),
            rag_documents_count=injections.get("rag_count", 0),
            warnings_count=injections.get("warnings_count", 0),
            tasks_reordered=injections.get("tasks_reordered", False),
        )

        self._injection_history.append(result)

        # Adicionar ao state
        existing = state.get(f"{BRIDGE_PREFIX}injections", [])
        if not isinstance(existing, list):
            existing = []

        updated_state = {
            **state,
            f"{BRIDGE_PREFIX}injections": existing + [result.to_dict()],
            f"{BRIDGE_PREFIX}last_injection": result.to_dict(),
        }

        return updated_state

    def get_fallback_for_task(
        self,
        state: Dict[str, Any],
        task_id: str
    ) -> Optional[Dict[str, Any]]:
        """Busca fallback plan para uma task específica.

        Usado quando uma task falha e precisamos de plano B.

        Args:
            state: Pipeline state
            task_id: ID da task que falhou

        Returns:
            Fallback plan ou None se não encontrado
        """
        fallback_plans = state.get("fallback_plans", [])

        if not fallback_plans:
            return None

        for plan in fallback_plans:
            if isinstance(plan, dict):
                plan_task_id = plan.get("task_id", "") or plan.get("for_task", "")
                if plan_task_id == task_id:
                    logger.info(f"StackDataBridge: Found fallback for task {task_id}")
                    return plan

        return None


# =============================================================================
# MODULE-LEVEL AVAILABILITY FLAG
# =============================================================================

# Flag para verificar se bridge está disponível (INV-BRIDGE-001)
BRIDGE_AVAILABLE = True

def get_stack_data_bridge() -> StackDataBridge:
    """Factory function para criar bridge.

    Returns:
        Nova instância de StackDataBridge
    """
    return StackDataBridge()
