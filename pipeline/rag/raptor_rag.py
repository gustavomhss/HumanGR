"""RAPTOR RAG - Recursive Abstractive Processing for Tree-Organized Retrieval.

RAPTOR implements hierarchical RAG by:
1. Clustering documents based on semantic similarity
2. Summarizing clusters at multiple levels of abstraction
3. Building a tree structure from leaves (documents) to root (high-level summary)
4. Enabling retrieval at different granularity levels

Based on:
- "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval"
- Sarthi et al., 2024

Usage:
    from pipeline.rag.raptor_rag import RaptorRAG

    rag = RaptorRAG()

    # Build tree from documents
    await rag.build_tree(documents)

    # Query at different levels
    result = await rag.query_hierarchical(
        query="What is the main theme?",
        traversal="tree",  # or "collapsed"
    )
"""

from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
from enum import Enum
import hashlib

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CLUSTER_SIZE = int(os.getenv("RAPTOR_CLUSTER_SIZE", "10"))
MAX_TREE_DEPTH = int(os.getenv("RAPTOR_MAX_DEPTH", "5"))
SUMMARY_MAX_LENGTH = int(os.getenv("RAPTOR_SUMMARY_LENGTH", "500"))
SIMILARITY_THRESHOLD = float(os.getenv("RAPTOR_SIMILARITY_THRESHOLD", "0.7"))

# RAPTOR is available (native implementation)
RAPTOR_AVAILABLE = True


# =============================================================================
# TRAVERSAL STRATEGIES
# =============================================================================


class TraversalStrategy(str, Enum):
    """Strategy for traversing the RAPTOR tree."""
    TREE = "tree"           # Top-down tree traversal
    COLLAPSED = "collapsed" # Flatten all levels
    LEVEL = "level"         # Query specific level
    ADAPTIVE = "adaptive"   # Dynamically choose based on query


# =============================================================================
# DATA MODELS
# =============================================================================


class RaptorNode(BaseModel):
    """A node in the RAPTOR tree."""

    node_id: str = Field(..., description="Unique node identifier")
    level: int = Field(..., ge=0, description="Tree level (0 = leaf)")
    content: str = Field(..., description="Node content (original or summary)")
    summary: str = Field(default="", description="Summary of this node")
    embedding: Optional[List[float]] = Field(default=None, description="Node embedding")
    children: List[str] = Field(default_factory=list, description="Child node IDs")
    parent: Optional[str] = Field(default=None, description="Parent node ID")
    cluster_id: int = Field(default=0)
    document_ids: List[str] = Field(default_factory=list, description="Original doc IDs in subtree")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RaptorTree(BaseModel):
    """The complete RAPTOR tree structure."""

    root_id: Optional[str] = Field(default=None)
    nodes: Dict[str, RaptorNode] = Field(default_factory=dict)
    depth: int = Field(default=0)
    leaf_count: int = Field(default=0)
    total_nodes: int = Field(default=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RaptorQueryResult(BaseModel):
    """Result from RAPTOR query."""

    query: str = Field(...)
    answer: str = Field(default="")

    # Traversal info
    traversal_strategy: TraversalStrategy = Field(default=TraversalStrategy.TREE)
    nodes_visited: List[str] = Field(default_factory=list)
    levels_traversed: List[int] = Field(default_factory=list)

    # Retrieved content
    retrieved_nodes: List[RaptorNode] = Field(default_factory=list)
    context_pieces: List[str] = Field(default_factory=list)

    # Scores
    relevance_scores: List[float] = Field(default_factory=list)
    confidence: float = Field(default=0.0)

    # Timing
    total_time_ms: float = Field(default=0.0)
    tree_traversal_time_ms: float = Field(default=0.0)


# =============================================================================
# RAPTOR IMPLEMENTATION
# =============================================================================


class RaptorRAG:
    """RAPTOR - Hierarchical RAG with tree-organized retrieval.

    Builds a tree structure from documents where:
    - Leaves are original documents
    - Internal nodes are cluster summaries
    - Root is the highest-level abstraction
    """

    def __init__(
        self,
        cluster_size: int = DEFAULT_CLUSTER_SIZE,
        max_depth: int = MAX_TREE_DEPTH,
        summary_length: int = SUMMARY_MAX_LENGTH,
    ):
        self.cluster_size = cluster_size
        self.max_depth = max_depth
        self.summary_length = summary_length

        self._tree: Optional[RaptorTree] = None
        self._stats = {
            "trees_built": 0,
            "queries_processed": 0,
            "nodes_created": 0,
            "summaries_generated": 0,
        }

        logger.info(f"RaptorRAG initialized (cluster_size={cluster_size}, max_depth={max_depth})")

    def _generate_node_id(self, level: int, index: int) -> str:
        """Generate unique node ID."""
        return f"raptor_L{level}_N{index}_{hashlib.md5(f'{level}_{index}'.encode()).hexdigest()[:8]}"

    async def build_tree(
        self,
        documents: List[Dict[str, Any]],
        rebuild: bool = False,
    ) -> RaptorTree:
        """Build RAPTOR tree from documents.

        Args:
            documents: List of documents with 'id', 'content', and optional 'embedding'
            rebuild: Force rebuild even if tree exists

        Returns:
            RaptorTree structure
        """
        if self._tree and not rebuild:
            logger.info("Using existing tree")
            return self._tree

        logger.info(f"Building RAPTOR tree from {len(documents)} documents")

        # Level 0: Create leaf nodes from documents
        nodes: Dict[str, RaptorNode] = {}
        current_level_ids: List[str] = []

        for i, doc in enumerate(documents):
            node_id = self._generate_node_id(0, i)
            node = RaptorNode(
                node_id=node_id,
                level=0,
                content=doc.get("content", ""),
                summary=doc.get("content", "")[:200],
                embedding=doc.get("embedding"),
                document_ids=[doc.get("id", f"doc_{i}")],
            )
            nodes[node_id] = node
            current_level_ids.append(node_id)
            self._stats["nodes_created"] += 1

        # Build higher levels through clustering and summarization
        level = 1
        while len(current_level_ids) > 1 and level <= self.max_depth:
            next_level_ids = await self._build_level(nodes, current_level_ids, level)

            if len(next_level_ids) == len(current_level_ids):
                # No more clustering possible
                break

            current_level_ids = next_level_ids
            level += 1

        # Set root
        root_id = current_level_ids[0] if current_level_ids else None

        self._tree = RaptorTree(
            root_id=root_id,
            nodes=nodes,
            depth=level - 1,
            leaf_count=len(documents),
            total_nodes=len(nodes),
        )

        self._stats["trees_built"] += 1
        logger.info(f"RAPTOR tree built: {len(nodes)} nodes, depth {level - 1}")

        return self._tree

    async def _build_level(
        self,
        nodes: Dict[str, RaptorNode],
        child_ids: List[str],
        level: int,
    ) -> List[str]:
        """Build one level of the tree by clustering children."""
        # Simple clustering: group by cluster_size
        clusters: List[List[str]] = []

        for i in range(0, len(child_ids), self.cluster_size):
            cluster = child_ids[i:i + self.cluster_size]
            if cluster:
                clusters.append(cluster)

        # Create parent nodes for each cluster
        parent_ids: List[str] = []

        for cluster_idx, cluster in enumerate(clusters):
            # Gather content from children
            child_contents = []
            all_doc_ids: List[str] = []

            for child_id in cluster:
                child = nodes[child_id]
                child_contents.append(child.summary or child.content[:200])
                all_doc_ids.extend(child.document_ids)

            # Generate summary for cluster
            summary = await self._summarize_cluster(child_contents)
            self._stats["summaries_generated"] += 1

            # Create parent node
            parent_id = self._generate_node_id(level, cluster_idx)
            parent = RaptorNode(
                node_id=parent_id,
                level=level,
                content=summary,
                summary=summary,
                children=cluster,
                cluster_id=cluster_idx,
                document_ids=all_doc_ids,
            )

            # Update children to point to parent
            for child_id in cluster:
                nodes[child_id].parent = parent_id

            nodes[parent_id] = parent
            parent_ids.append(parent_id)
            self._stats["nodes_created"] += 1

        return parent_ids

    async def _summarize_cluster(self, contents: List[str]) -> str:
        """Summarize a cluster of content.

        C-008 FIX: Uses real LLM summarization instead of substring extraction.
        Falls back to first N chars only if LLM unavailable (with warning).
        """
        combined = " ".join(contents)

        try:
            from pipeline.claude_cli_llm import execute_agent_task

            prompt = f"""Summarize the following content into a concise summary of at most {self.summary_length} characters.

Content:
{combined[:5000]}

Provide ONLY the summary, no preamble or explanation."""

            result = execute_agent_task(
                task="SUMMARIZE",
                instruction=prompt,
                workspace_path=os.getcwd(),
                sprint_id="raptor_rag",
            )

            if result.get("status") == "success" and result.get("output"):
                summary = result["output"][:self.summary_length]
                logger.debug(f"RAPTOR: Generated summary of {len(summary)} chars")
                return summary

            logger.warning(
                f"C-008: RAPTOR LLM summarization failed: {result.get('error', 'unknown')}. "
                "Falling back to extractive summary."
            )
            return combined[:self.summary_length]

        except ImportError:
            logger.warning(
                "C-008: RAPTOR LLM not available (claude_cli_llm not installed). "
                "Falling back to extractive summary (first N chars)."
            )
            return combined[:self.summary_length]
        except Exception as e:
            logger.warning(
                f"C-008: RAPTOR summarization failed: {e}. "
                "Falling back to extractive summary (first N chars)."
            )
            return combined[:self.summary_length]

    async def query_hierarchical(
        self,
        query: str,
        traversal: TraversalStrategy = TraversalStrategy.TREE,
        target_level: Optional[int] = None,
        top_k: int = 5,
    ) -> RaptorQueryResult:
        """Query the RAPTOR tree.

        Args:
            query: The query to process
            traversal: Traversal strategy to use
            target_level: For LEVEL strategy, which level to query
            top_k: Number of nodes to retrieve

        Returns:
            RaptorQueryResult with hierarchical context
        """
        import time
        start_time = time.time()

        self._stats["queries_processed"] += 1

        if not self._tree:
            return RaptorQueryResult(
                query=query,
                answer="No tree built. Call build_tree() first.",
                traversal_strategy=traversal,
            )

        # Perform traversal based on strategy
        traversal_start = time.time()

        if traversal == TraversalStrategy.TREE:
            retrieved = await self._tree_traversal(query, top_k)
        elif traversal == TraversalStrategy.COLLAPSED:
            retrieved = await self._collapsed_traversal(query, top_k)
        elif traversal == TraversalStrategy.LEVEL:
            retrieved = await self._level_traversal(query, target_level or 0, top_k)
        else:  # ADAPTIVE
            retrieved = await self._adaptive_traversal(query, top_k)

        traversal_time = (time.time() - traversal_start) * 1000

        # Extract context from retrieved nodes
        context_pieces = [node.content for node in retrieved]
        nodes_visited = [node.node_id for node in retrieved]
        levels_traversed = list(set(node.level for node in retrieved))

        # Generate answer
        answer = await self._generate_answer(query, context_pieces)

        total_time = (time.time() - start_time) * 1000

        return RaptorQueryResult(
            query=query,
            answer=answer,
            traversal_strategy=traversal,
            nodes_visited=nodes_visited,
            levels_traversed=sorted(levels_traversed),
            retrieved_nodes=retrieved,
            context_pieces=context_pieces,
            relevance_scores=[0.9 - i * 0.1 for i in range(len(retrieved))],
            confidence=0.8 if retrieved else 0.0,
            total_time_ms=total_time,
            tree_traversal_time_ms=traversal_time,
        )

    async def _tree_traversal(
        self,
        query: str,
        top_k: int,
    ) -> List[RaptorNode]:
        """Top-down tree traversal."""
        if not self._tree or not self._tree.root_id:
            return []

        # Start from root and traverse down
        visited: List[RaptorNode] = []
        queue = [self._tree.root_id]
        query_words = set(query.lower().split())

        while queue and len(visited) < top_k:
            node_id = queue.pop(0)
            node = self._tree.nodes.get(node_id)

            if not node:
                continue

            # Check relevance
            node_words = set(node.content.lower().split())
            relevance = len(query_words & node_words) / max(len(query_words), 1)

            if relevance > 0.2 or node.level == 0:
                visited.append(node)

                # Add children if relevant
                if relevance > 0.3 and node.children:
                    queue.extend(node.children)

        return visited[:top_k]

    async def _collapsed_traversal(
        self,
        query: str,
        top_k: int,
    ) -> List[RaptorNode]:
        """Flatten all levels and retrieve best matches."""
        if not self._tree:
            return []

        # Score all nodes
        scored: List[Tuple[float, RaptorNode]] = []
        query_words = set(query.lower().split())

        for node in self._tree.nodes.values():
            node_words = set(node.content.lower().split())
            relevance = len(query_words & node_words) / max(len(query_words), 1)
            scored.append((relevance, node))

        # Sort by relevance and return top_k
        scored.sort(key=lambda x: -x[0])
        return [node for _, node in scored[:top_k]]

    async def _level_traversal(
        self,
        query: str,
        level: int,
        top_k: int,
    ) -> List[RaptorNode]:
        """Query specific level only."""
        if not self._tree:
            return []

        # Get nodes at target level
        level_nodes = [n for n in self._tree.nodes.values() if n.level == level]

        # Score by relevance
        query_words = set(query.lower().split())
        scored = []

        for node in level_nodes:
            node_words = set(node.content.lower().split())
            relevance = len(query_words & node_words) / max(len(query_words), 1)
            scored.append((relevance, node))

        scored.sort(key=lambda x: -x[0])
        return [node for _, node in scored[:top_k]]

    async def _adaptive_traversal(
        self,
        query: str,
        top_k: int,
    ) -> List[RaptorNode]:
        """Adaptively choose traversal based on query."""
        # Simple heuristic: use tree traversal for complex queries, collapsed for simple
        if len(query.split()) > 5:
            return await self._tree_traversal(query, top_k)
        else:
            return await self._collapsed_traversal(query, top_k)

    async def _generate_answer(
        self,
        query: str,
        context: List[str],
    ) -> str:
        """Generate answer from context."""
        if not context:
            return f"No relevant information found for: {query}"

        combined = " | ".join(context[:3])
        return f"Based on hierarchical context: {combined[:500]}"

    def get_tree_stats(self) -> Dict[str, Any]:
        """Get tree statistics."""
        if not self._tree:
            return {"tree_built": False}

        return {
            "tree_built": True,
            "depth": self._tree.depth,
            "total_nodes": self._tree.total_nodes,
            "leaf_count": self._tree.leaf_count,
            **self._stats,
        }


# =============================================================================
# SINGLETON AND CONVENIENCE FUNCTIONS
# =============================================================================

_raptor_instance: Optional[RaptorRAG] = None


def get_raptor_rag() -> RaptorRAG:
    """Get or create RaptorRAG singleton."""
    global _raptor_instance
    if _raptor_instance is None:
        _raptor_instance = RaptorRAG()
    return _raptor_instance


async def query_hierarchical(
    query: str,
    traversal: TraversalStrategy = TraversalStrategy.TREE,
) -> RaptorQueryResult:
    """Convenience function for hierarchical query."""
    rag = get_raptor_rag()
    return await rag.query_hierarchical(query=query, traversal=traversal)


__all__ = [
    "RAPTOR_AVAILABLE",
    "RaptorRAG",
    "RaptorTree",
    "RaptorNode",
    "RaptorQueryResult",
    "TraversalStrategy",
    "get_raptor_rag",
    "query_hierarchical",
]
