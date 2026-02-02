"""Runtime Card Loader - Lightweight persona loading for agents.

P2.3: Runtime Cards are ~50-line YAML files that provide:
- Essential persona information (role, goal, backstory)
- Key skills and constraints
- Much faster loading than full 1500-line Cerebro documents

Loading hierarchy:
1. Runtime Card (~50 lines, fast)
2. Full Cerebro (~1500 lines, complete)
3. PERSONA_TEMPLATES (fallback)

CRIT-001 FIX (2026-01-22): Added persona sanitization to detect and block
prompt injection patterns in backstory/goal/role fields. This prevents
malicious YAML content from manipulating LLM behavior.

Author: Pipeline Autonomo Team
Version: 1.1.0 (2026-01-22) - Security hardened
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger("pipeline_autonomo.runtime_card_loader")


# =============================================================================
# CRIT-001 FIX: Prompt Injection Detection and Sanitization
# =============================================================================

# Patterns that indicate potential prompt injection attacks
PROMPT_INJECTION_PATTERNS = [
    # Direct instruction override attempts
    r"ignore\s+(previous|all|above|prior)\s+(instructions?|prompts?|rules?)",
    r"disregard\s+(previous|all|above|prior)\s+(instructions?|prompts?)",
    r"forget\s+(everything|all|what)\s+(you|I)\s+(told|said)",
    r"new\s+instructions?:",
    r"system\s*prompt:",
    r"you\s+are\s+now\s+a",
    r"act\s+as\s+if\s+you\s+are",
    r"pretend\s+(you|to\s+be)",
    r"roleplay\s+as",

    # Instruction delimiters that could override context
    r"\[SYSTEM\]",
    r"\[INST\]",
    r"<\|im_start\|>",
    r"<\|im_end\|>",
    r"###\s*Instruction",
    r"###\s*System",
    r"###\s*Human",
    r"###\s*Assistant",

    # Jailbreak patterns
    r"DAN\s*mode",
    r"developer\s*mode",
    r"do\s+anything\s+now",
    r"no\s+restrictions",
    r"bypass\s+(safety|filters?|guardrails?)",
    r"disable\s+(safety|filters?|guardrails?)",

    # Output manipulation
    r"print\s*\(\s*['\"]",
    r"execute\s*\(",
    r"eval\s*\(",
    r"exec\s*\(",
    r"__import__",
    r"subprocess",
    r"os\.system",

    # Data exfiltration attempts
    r"send\s+(this|the|all)\s+(to|data)",
    r"exfiltrate",
    r"extract\s+(and\s+)?send",

    # Credential/secret extraction
    r"(api|secret|private)[\s_-]?key",
    r"password",
    r"token",
    r"credential",
]

# Compiled patterns for efficiency
_INJECTION_PATTERNS_COMPILED = [
    re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    for pattern in PROMPT_INJECTION_PATTERNS
]


def _normalize_unicode(text: str) -> str:
    """Normalize unicode to catch homoglyph attacks.

    CRIT-001 FIX: Attackers may use unicode lookalikes to bypass pattern
    detection. Normalize to ASCII-compatible form first.
    """
    # NFKC normalization converts lookalikes to standard forms
    normalized = unicodedata.normalize('NFKC', text)

    # Remove zero-width characters that could hide malicious content
    zero_width_chars = [
        '\u200b',  # Zero Width Space
        '\u200c',  # Zero Width Non-Joiner
        '\u200d',  # Zero Width Joiner
        '\u2060',  # Word Joiner
        '\ufeff',  # Zero Width No-Break Space
    ]
    for char in zero_width_chars:
        normalized = normalized.replace(char, '')

    return normalized


def _detect_prompt_injection(text: str, field_name: str) -> tuple[bool, list[str]]:
    """Detect potential prompt injection in text.

    Args:
        text: Text to scan for injection patterns.
        field_name: Name of the field being checked (for logging).

    Returns:
        Tuple of (is_safe, list_of_matched_patterns).
        is_safe=True if no injection detected.
    """
    if not text:
        return True, []

    # Normalize unicode first
    normalized = _normalize_unicode(text)

    matched_patterns = []
    for pattern in _INJECTION_PATTERNS_COMPILED:
        if pattern.search(normalized):
            matched_patterns.append(pattern.pattern)

    if matched_patterns:
        logger.warning(
            f"CRIT-001: Prompt injection detected in {field_name}. "
            f"Matched patterns: {matched_patterns[:3]}..."  # Log first 3
        )
        return False, matched_patterns

    return True, []


def sanitize_persona_field(text: str, field_name: str, max_length: int = 5000) -> str:
    """Sanitize a persona field to prevent prompt injection.

    CRIT-001 FIX: This function:
    1. Normalizes unicode to catch homoglyph attacks
    2. Detects prompt injection patterns
    3. Truncates overly long content
    4. Removes dangerous control characters

    Args:
        text: Text to sanitize.
        field_name: Name of the field (for logging).
        max_length: Maximum allowed length (default 5000 chars).

    Returns:
        Sanitized text, or raises ValueError if malicious content detected.

    Raises:
        ValueError: If prompt injection is detected and cannot be safely removed.
    """
    if not text:
        return text

    # 1. Normalize unicode
    sanitized = _normalize_unicode(text)

    # 2. Remove control characters except newlines and tabs
    sanitized = ''.join(
        char for char in sanitized
        if char == '\n' or char == '\t' or not unicodedata.category(char).startswith('C')
    )

    # 3. Check for prompt injection
    is_safe, matched_patterns = _detect_prompt_injection(sanitized, field_name)
    if not is_safe:
        # FAIL-CLOSED: Do not allow potentially malicious content
        raise ValueError(
            f"CRIT-001: Prompt injection detected in {field_name}. "
            f"Content blocked for security. Matched: {matched_patterns[:3]}"
        )

    # 4. Truncate if too long (prevents context overflow attacks)
    if len(sanitized) > max_length:
        logger.warning(
            f"CRIT-001: Truncating {field_name} from {len(sanitized)} to {max_length} chars"
        )
        sanitized = sanitized[:max_length] + "... [TRUNCATED FOR SECURITY]"

    return sanitized

# Default location for runtime cards
DEFAULT_RUNTIME_CARDS_DIR = Path(__file__).parent.parent.parent / "configs" / "pipeline_autonomo" / "runtime_cards"


@dataclass
class SquadMemoryConfig:
    """F4-001: Squad memory configuration for an agent.

    Defines how an agent participates in squad-based knowledge sharing.
    Workers submit findings to squad lead; lead broadcasts approved content.
    """

    enabled: bool = False
    squad_id: str = ""
    role: str = "worker"  # "worker" or "lead"
    squad_lead: str = ""
    readable_segments: list[str] = field(default_factory=list)
    submittable_findings: list[str] = field(default_factory=list)
    managed_segments: list[str] = field(default_factory=list)  # For leads
    workers: list[str] = field(default_factory=list)  # For leads
    awareness_prompt: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SquadMemoryConfig":
        """Create SquadMemoryConfig from dictionary."""
        if not data:
            return cls()

        return cls(
            enabled=data.get("enabled", False),
            squad_id=data.get("squad_id", ""),
            role=data.get("role", "worker"),
            squad_lead=data.get("squad_lead", ""),
            readable_segments=data.get("readable_segments", []),
            submittable_findings=data.get("submittable_findings", []),
            managed_segments=data.get("managed_segments", []),
            workers=data.get("workers", []),
            awareness_prompt=data.get("awareness_prompt", ""),
        )


@dataclass
class GraphStacksConfig:
    """Graph stacks configuration for an agent.

    Defines which graph stacks an agent can use and what operations
    are allowed. Used by GraphStackOrchestrator for FAIL-CLOSED access control.

    Added: 2026-02-01 (TASK-GS-019)
    """

    # FalkorDB - Knowledge graph for claims/evidence
    falkordb_enabled: bool = False
    falkordb_operations: list[str] = field(default_factory=list)

    # Neo4j Enhanced - Vector search + GDS algorithms
    neo4j_enabled: bool = False
    neo4j_operations: list[str] = field(default_factory=list)

    # Neo4j Algorithms - PageRank, community detection
    neo4j_algorithms_enabled: bool = False
    neo4j_algorithms_operations: list[str] = field(default_factory=list)

    # Neo4j Analytics - Advanced claim/source analysis
    neo4j_analytics_enabled: bool = False
    neo4j_analytics_operations: list[str] = field(default_factory=list)

    # GoT Enhanced - Multi-path reasoning
    got_enabled: bool = False
    got_operations: list[str] = field(default_factory=list)

    # Graphiti - Temporal knowledge graph
    graphiti_enabled: bool = False
    graphiti_operations: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GraphStacksConfig":
        """Create GraphStacksConfig from dictionary."""
        if not data:
            return cls()

        return cls(
            falkordb_enabled=data.get("falkordb_enabled", False),
            falkordb_operations=data.get("falkordb_operations", []),
            neo4j_enabled=data.get("neo4j_enabled", False),
            neo4j_operations=data.get("neo4j_operations", []),
            neo4j_algorithms_enabled=data.get("neo4j_algorithms_enabled", False),
            neo4j_algorithms_operations=data.get("neo4j_algorithms_operations", []),
            neo4j_analytics_enabled=data.get("neo4j_analytics_enabled", False),
            neo4j_analytics_operations=data.get("neo4j_analytics_operations", []),
            got_enabled=data.get("got_enabled", False),
            got_operations=data.get("got_operations", []),
            graphiti_enabled=data.get("graphiti_enabled", False),
            graphiti_operations=data.get("graphiti_operations", []),
        )

    def has_any_enabled(self) -> bool:
        """Check if any stack is enabled."""
        return any([
            self.falkordb_enabled,
            self.neo4j_enabled,
            self.neo4j_algorithms_enabled,
            self.neo4j_analytics_enabled,
            self.got_enabled,
            self.graphiti_enabled,
        ])


@dataclass
class RagStackConfig:
    """Configuration for a single RAG stack.

    Defines whether a RAG stack is enabled for an agent and its priority.
    Lower priority number = higher priority (called first).

    Added: 2026-02-01 (RAG Stack Router Implementation)
    """

    enabled: bool = False
    priority: int = 99  # Lower = higher priority (1 is highest)
    use_cases: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    reason: str = ""  # If disabled, why

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RagStackConfig":
        """Create RagStackConfig from dictionary."""
        if not data:
            return cls()

        return cls(
            enabled=data.get("enabled", False),
            priority=data.get("priority", 99),
            use_cases=data.get("use_cases", []),
            parameters=data.get("parameters", {}),
            reason=data.get("reason", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {"enabled": self.enabled}
        if self.enabled:
            result["priority"] = self.priority
            if self.use_cases:
                result["use_cases"] = self.use_cases
            if self.parameters:
                result["parameters"] = self.parameters
        else:
            if self.reason:
                result["reason"] = self.reason
        return result


@dataclass
class RagStacksConfig:
    """RAG stacks configuration for an agent.

    Defines which RAG stacks an agent can use, their priorities,
    and stack-specific parameters. Used by RAGStackRouter for
    agent-specific RAG query routing.

    Stacks available:
    - self_rag: Self-reflective RAG with critique loop
    - corrective_rag: CRAG with quality validation
    - graphrag: Graph-based RAG for relationships
    - memo_rag: Memory-augmented RAG (episodic)
    - raptor_rag: Hierarchical document RAG
    - colbert: Token-level dense retrieval
    - qdrant_hybrid: Hybrid sparse+dense search

    Added: 2026-02-01 (RAG Stack Router Implementation)
    """

    self_rag: RagStackConfig = field(default_factory=RagStackConfig)
    corrective_rag: RagStackConfig = field(default_factory=RagStackConfig)
    graphrag: RagStackConfig = field(default_factory=RagStackConfig)
    memo_rag: RagStackConfig = field(default_factory=RagStackConfig)
    raptor_rag: RagStackConfig = field(default_factory=RagStackConfig)
    colbert: RagStackConfig = field(default_factory=RagStackConfig)
    qdrant_hybrid: RagStackConfig = field(default_factory=RagStackConfig)
    awareness_prompt: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RagStacksConfig":
        """Create RagStacksConfig from dictionary."""
        if not data:
            return cls()

        return cls(
            self_rag=RagStackConfig.from_dict(data.get("self_rag", {})),
            corrective_rag=RagStackConfig.from_dict(data.get("corrective_rag", {})),
            graphrag=RagStackConfig.from_dict(data.get("graphrag", {})),
            memo_rag=RagStackConfig.from_dict(data.get("memo_rag", {})),
            raptor_rag=RagStackConfig.from_dict(data.get("raptor_rag", {})),
            colbert=RagStackConfig.from_dict(data.get("colbert", {})),
            qdrant_hybrid=RagStackConfig.from_dict(data.get("qdrant_hybrid", {})),
            awareness_prompt=data.get("awareness_prompt", ""),
        )

    def get_enabled_stacks(self) -> list[tuple[str, int]]:
        """Return [(stack_name, priority)] for enabled stacks, sorted by priority.

        Returns:
            List of (stack_name, priority) tuples, sorted by priority (ascending).
            Lower priority number = higher priority (called first).
        """
        enabled = []
        stack_configs = [
            ("self_rag", self.self_rag),
            ("corrective_rag", self.corrective_rag),
            ("graphrag", self.graphrag),
            ("memo_rag", self.memo_rag),
            ("raptor_rag", self.raptor_rag),
            ("colbert", self.colbert),
            ("qdrant_hybrid", self.qdrant_hybrid),
        ]
        for name, config in stack_configs:
            if config.enabled:
                enabled.append((name, config.priority))
        # Sort by priority (ascending - lower number = higher priority)
        return sorted(enabled, key=lambda x: x[1])

    def has_any_enabled(self) -> bool:
        """Check if any RAG stack is enabled."""
        return any([
            self.self_rag.enabled,
            self.corrective_rag.enabled,
            self.graphrag.enabled,
            self.memo_rag.enabled,
            self.raptor_rag.enabled,
            self.colbert.enabled,
            self.qdrant_hybrid.enabled,
        ])

    def get_stack_config(self, stack_name: str) -> RagStackConfig | None:
        """Get configuration for a specific stack by name.

        Args:
            stack_name: Name of the stack (e.g., "self_rag", "graphrag")

        Returns:
            RagStackConfig for the stack, or None if not found.
        """
        stack_map = {
            "self_rag": self.self_rag,
            "corrective_rag": self.corrective_rag,
            "graphrag": self.graphrag,
            "memo_rag": self.memo_rag,
            "raptor_rag": self.raptor_rag,
            "colbert": self.colbert,
            "qdrant_hybrid": self.qdrant_hybrid,
        }
        return stack_map.get(stack_name)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {}
        if self.self_rag.enabled or self.self_rag.reason:
            result["self_rag"] = self.self_rag.to_dict()
        if self.corrective_rag.enabled or self.corrective_rag.reason:
            result["corrective_rag"] = self.corrective_rag.to_dict()
        if self.graphrag.enabled or self.graphrag.reason:
            result["graphrag"] = self.graphrag.to_dict()
        if self.memo_rag.enabled or self.memo_rag.reason:
            result["memo_rag"] = self.memo_rag.to_dict()
        if self.raptor_rag.enabled or self.raptor_rag.reason:
            result["raptor_rag"] = self.raptor_rag.to_dict()
        if self.colbert.enabled or self.colbert.reason:
            result["colbert"] = self.colbert.to_dict()
        if self.qdrant_hybrid.enabled or self.qdrant_hybrid.reason:
            result["qdrant_hybrid"] = self.qdrant_hybrid.to_dict()
        if self.awareness_prompt:
            result["awareness_prompt"] = self.awareness_prompt
        return result


@dataclass
class RuntimeCard:
    """Lightweight persona definition for an agent.

    A Runtime Card contains only the essential information needed
    at runtime, extracted from the full Cerebro document.

    Fields:
        agent_id: Unique agent identifier (e.g., "qa_master")
        role: Agent's role description
        goal: Primary mission/goal
        backstory: Brief backstory (max 500 chars for efficiency)
        key_skills: List of primary capabilities
        key_constraints: List of important constraints/rules
        reports_to: Supervisor agent ID
        level: Hierarchy level (0-6)
        version: Runtime Card version for cache invalidation
        squad_memory: F4-001 Squad memory configuration
        graph_stacks: Graph stacks configuration (2026-02-01)
        rag_stacks: RAG stacks configuration (2026-02-01)
    """

    agent_id: str
    role: str
    goal: str
    backstory: str
    reports_to: Optional[str] = None
    level: int = 0
    key_skills: list[str] = field(default_factory=list)
    key_constraints: list[str] = field(default_factory=list)
    version: str = "1.0.0"
    squad_memory: Optional[SquadMemoryConfig] = None
    graph_stacks: Optional[GraphStacksConfig] = None
    rag_stacks: Optional[RagStacksConfig] = None

    def to_persona(self) -> dict[str, str]:
        """Convert to persona dict for CrewAI.

        F4-001: Includes squad memory awareness prompt if configured.
        2026-02-01: Includes RAG stacks awareness prompt if configured.
        """
        backstory = self.backstory

        # Inject squad memory awareness if enabled
        if self.squad_memory and self.squad_memory.enabled:
            if self.squad_memory.awareness_prompt:
                backstory = f"{backstory}\n\n{self.squad_memory.awareness_prompt}"

        # Inject RAG stacks awareness if enabled
        if self.rag_stacks and self.rag_stacks.has_any_enabled():
            if self.rag_stacks.awareness_prompt:
                backstory = f"{backstory}\n\n{self.rag_stacks.awareness_prompt}"

        return {
            "role": self.role,
            "goal": self.goal,
            "backstory": backstory,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "agent_id": self.agent_id,
            "role": self.role,
            "goal": self.goal,
            "backstory": self.backstory,
            "reports_to": self.reports_to,
            "level": self.level,
            "key_skills": self.key_skills,
            "key_constraints": self.key_constraints,
            "version": self.version,
        }
        if self.squad_memory:
            result["squad_memory"] = {
                "enabled": self.squad_memory.enabled,
                "squad_id": self.squad_memory.squad_id,
                "role": self.squad_memory.role,
                "squad_lead": self.squad_memory.squad_lead,
                "readable_segments": self.squad_memory.readable_segments,
                "submittable_findings": self.squad_memory.submittable_findings,
            }
        if self.graph_stacks:
            result["graph_stacks"] = {
                "falkordb_enabled": self.graph_stacks.falkordb_enabled,
                "falkordb_operations": self.graph_stacks.falkordb_operations,
                "neo4j_enabled": self.graph_stacks.neo4j_enabled,
                "neo4j_operations": self.graph_stacks.neo4j_operations,
                "neo4j_algorithms_enabled": self.graph_stacks.neo4j_algorithms_enabled,
                "neo4j_algorithms_operations": self.graph_stacks.neo4j_algorithms_operations,
                "neo4j_analytics_enabled": self.graph_stacks.neo4j_analytics_enabled,
                "neo4j_analytics_operations": self.graph_stacks.neo4j_analytics_operations,
                "got_enabled": self.graph_stacks.got_enabled,
                "got_operations": self.graph_stacks.got_operations,
                "graphiti_enabled": self.graph_stacks.graphiti_enabled,
                "graphiti_operations": self.graph_stacks.graphiti_operations,
            }
        if self.rag_stacks:
            result["rag_stacks"] = self.rag_stacks.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RuntimeCard:
        """Create RuntimeCard from dictionary.

        CRIT-001 FIX (2026-01-22): Added sanitization of persona fields
        to prevent prompt injection attacks via malicious YAML content.
        """
        agent_id = data.get("agent_id", "")

        # CRIT-001 FIX: Sanitize all text fields that will be passed to LLM
        try:
            role = sanitize_persona_field(
                data.get("role", ""),
                f"{agent_id}.role",
                max_length=500
            )
            goal = sanitize_persona_field(
                data.get("goal", ""),
                f"{agent_id}.goal",
                max_length=1000
            )
            backstory = sanitize_persona_field(
                data.get("backstory", ""),
                f"{agent_id}.backstory",
                max_length=5000
            )

            # Also sanitize skills and constraints as they may be injected
            key_skills = [
                sanitize_persona_field(skill, f"{agent_id}.skill", max_length=200)
                for skill in data.get("key_skills", [])
                if isinstance(skill, str)
            ]
            key_constraints = [
                sanitize_persona_field(constraint, f"{agent_id}.constraint", max_length=200)
                for constraint in data.get("key_constraints", [])
                if isinstance(constraint, str)
            ]

        except ValueError as e:
            # CRIT-001: Prompt injection detected - log and fail safely
            logger.error(f"CRIT-001 SECURITY: Blocked malicious runtime card for {agent_id}: {e}")
            raise ValueError(
                f"CRIT-001: Runtime card for {agent_id} contains malicious content and was blocked"
            ) from e

        # F4-001: Load squad_memory configuration
        squad_memory = None
        if "squad_memory" in data:
            squad_memory = SquadMemoryConfig.from_dict(data["squad_memory"])

        # 2026-02-01: Load graph_stacks configuration
        graph_stacks = None
        if "graph_stacks" in data:
            graph_stacks = GraphStacksConfig.from_dict(data["graph_stacks"])

        # 2026-02-01: Load rag_stacks configuration
        rag_stacks = None
        if "rag_stacks" in data:
            rag_stacks = RagStacksConfig.from_dict(data["rag_stacks"])

        return cls(
            agent_id=agent_id,
            role=role,
            goal=goal,
            backstory=backstory,
            reports_to=data.get("reports_to"),
            level=data.get("level", 0),
            key_skills=key_skills,
            key_constraints=key_constraints,
            version=data.get("version", "1.0.0"),
            squad_memory=squad_memory,
            graph_stacks=graph_stacks,
            rag_stacks=rag_stacks,
        )


class RuntimeCardLoader:
    """Loader for Runtime Cards with caching and fallback support.

    P2.3: Implements the loading hierarchy:
    1. Runtime Card (fast, ~50 lines)
    2. Full Cerebro via Spec Kit (fallback, ~1500 lines)
    3. Default template (emergency fallback)
    """

    def __init__(self, cards_dir: Optional[Path] = None):
        """Initialize loader.

        Args:
            cards_dir: Directory containing runtime card YAML files.
                      Defaults to configs/pipeline_autonomo/runtime_cards/
        """
        self.cards_dir = Path(cards_dir) if cards_dir else DEFAULT_RUNTIME_CARDS_DIR
        self._cache: dict[str, RuntimeCard] = {}
        self._load_failed: set[str] = set()  # Track failed loads to avoid retry

    def load(self, agent_id: str) -> Optional[RuntimeCard]:
        """Load a Runtime Card for an agent.

        Args:
            agent_id: Agent identifier (e.g., "qa_master")

        Returns:
            RuntimeCard if found, None otherwise.
        """
        agent_id = agent_id.lower().strip()

        # Check cache first
        if agent_id in self._cache:
            return self._cache[agent_id]

        # Skip if previously failed
        if agent_id in self._load_failed:
            return None

        # Try to load from YAML file
        card = self._load_from_file(agent_id)
        if card is not None:
            self._cache[agent_id] = card
            logger.debug("RUNTIME_CARD_LOADED: %s", agent_id)
            return card

        # Mark as failed to avoid retrying
        self._load_failed.add(agent_id)
        return None

    def _load_from_file(self, agent_id: str) -> Optional[RuntimeCard]:
        """Load Runtime Card from YAML file.

        Args:
            agent_id: Agent identifier

        Returns:
            RuntimeCard if file exists and is valid, None otherwise.
        """
        # Try different filename patterns
        filenames = [
            f"{agent_id}.yml",
            f"{agent_id}.yaml",
            f"runtime_card_{agent_id}.yml",
            f"runtime_card_{agent_id}.yaml",
        ]

        for filename in filenames:
            file_path = self.cards_dir / filename
            if file_path.exists():
                try:
                    with open(file_path, encoding="utf-8") as f:
                        data = yaml.safe_load(f)

                    if data is None:
                        logger.warning("RUNTIME_CARD_EMPTY: %s", file_path)
                        continue

                    # Ensure agent_id is set
                    if "agent_id" not in data:
                        data["agent_id"] = agent_id

                    return RuntimeCard.from_dict(data)

                except yaml.YAMLError as e:
                    logger.error("RUNTIME_CARD_PARSE_ERROR: %s - %s", file_path, e)
                except Exception as e:
                    logger.error("RUNTIME_CARD_LOAD_ERROR: %s - %s", file_path, e)

        return None

    def clear_cache(self) -> None:
        """Clear the runtime card cache."""
        self._cache.clear()
        self._load_failed.clear()
        logger.info("RUNTIME_CARD_CACHE_CLEARED")

    def get_all_loaded(self) -> list[str]:
        """Get list of all loaded agent IDs."""
        return list(self._cache.keys())


# Global singleton instance
_loader: Optional[RuntimeCardLoader] = None


def get_runtime_card_loader(cards_dir: Optional[Path] = None) -> RuntimeCardLoader:
    """Get the global RuntimeCardLoader instance.

    Args:
        cards_dir: Optional directory for runtime cards.

    Returns:
        RuntimeCardLoader singleton instance.
    """
    global _loader
    if _loader is None:
        _loader = RuntimeCardLoader(cards_dir)
    return _loader


def load_runtime_card(agent_id: str) -> Optional[RuntimeCard]:
    """Convenience function to load a runtime card.

    Args:
        agent_id: Agent identifier.

    Returns:
        RuntimeCard if found, None otherwise.
    """
    return get_runtime_card_loader().load(agent_id)


def get_persona_with_runtime_card(
    agent_id: str,
    include_memory: bool = True,
    include_skills: bool = True,
) -> dict[str, str]:
    """Get persona for an agent, preferring Runtime Card.

    P2.3: Loading hierarchy:
    1. Runtime Card (fast, ~50 lines)
    2. Full Cerebro via Spec Kit (fallback, ~1500 lines)
    3. Default template (emergency fallback)

    F-229 FIX: Now includes skills from skills_loader.

    Args:
        agent_id: Agent identifier.
        include_memory: Whether to include Letta persistent memory.
        include_skills: Whether to include skills from skills_loader.

    Returns:
        Dictionary with role, goal, and backstory.
    """
    agent_id = agent_id.lower().strip()

    # Get Letta memory context if requested
    memory_context = ""
    if include_memory:
        try:
            from pipeline.letta_integration import get_memory_bridge
            bridge = get_memory_bridge()
            memory_context = bridge.get_prompt_context(agent_id, max_items=5)
            if memory_context:
                memory_context = f"\n\n---\n## Your Persistent Memory\n{memory_context}\n---\n"
        except Exception as e:
            # RED TEAM FIX CRIT-002: Log memory fetch failures
            logger.warning(
                f"CRIT-002: Letta memory fetch failed for {agent_id}: {e}. "
                "Agent operating WITHOUT persistent memory context."
            )

    # F-229 FIX: Get skills context for the agent
    skills_context = ""
    if include_skills:
        try:
            from pipeline.skills_loader import get_skills_loader
            from pipeline.hierarchy import get_level
            level = get_level(agent_id)
            skills_loader = get_skills_loader()
            skills_prompt = skills_loader.get_prompt_for_agent(agent_id, level)
            if skills_prompt:
                skills_context = f"\n\n{skills_prompt}"
                logger.debug("F-229: Loaded skills for agent %s", agent_id)
        except Exception as e:
            logger.debug(f"F-229: Skills loading failed for {agent_id}: {e}")

    # 1. Try Runtime Card first (fast path)
    card = load_runtime_card(agent_id)
    if card is not None:
        persona = card.to_persona()
        persona["backstory"] = persona["backstory"] + memory_context + skills_context
        logger.debug("PERSONA_FROM_RUNTIME_CARD: %s", agent_id)
        return persona

    # 2. Fall back to full get_persona (loads Cerebro or templates)
    # Import here to avoid circular import
    from pipeline.crewai_hierarchy import get_persona
    logger.debug("PERSONA_FALLBACK_TO_CEREBRO: %s", agent_id)
    persona = get_persona(agent_id, include_memory=include_memory)

    # ENTERPRISE-CULTURE: Validate persona has required keys before returning
    required_keys = {"role", "goal", "backstory"}
    missing_keys = required_keys - set(persona.keys())
    if missing_keys:
        logger.error(f"INVALID_PERSONA: {agent_id} missing required keys: {missing_keys}")
        raise ValueError(f"Persona for {agent_id} is incomplete. Missing: {missing_keys}")

    # F-229: Also add skills to fallback path
    if skills_context:
        persona["backstory"] = persona["backstory"] + skills_context
    return persona
