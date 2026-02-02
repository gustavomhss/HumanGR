"""
HumanGR Pipeline Configuration

TODAS as configurações são específicas para HumanGR.
Nenhuma referência a Veritas, pipeline_, ou brains.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class HumanGRConfig:
    """
    Configuração do pipeline HumanGR.

    SEPARAÇÃO GARANTIDA:
    - product_id: sempre "HUMANGR"
    - qdrant_prefix: sempre "humangr_"
    - paths: sempre dentro de HL-MCP
    """

    # Identity - NUNCA MUDA
    product_id: str = "HUMANGR"
    product_name: str = "Human Layer MCP Server"

    # Paths - ISOLADOS
    project_root: Path = field(
        default_factory=lambda: Path(__file__).parent.parent
    )

    @property
    def context_packs_dir(self) -> Path:
        return self.project_root / "context_packs"

    @property
    def target_repo(self) -> Path:
        return self.project_root / "target" / "human-layer"

    @property
    def output_dir(self) -> Path:
        return self.project_root / "out"

    # Qdrant - PREFIXO SEPARADO
    qdrant_host: str = field(
        default_factory=lambda: os.getenv("QDRANT_HOST", "localhost")
    )
    qdrant_port: int = field(
        default_factory=lambda: int(os.getenv("QDRANT_PORT", "6333"))
    )
    qdrant_collection_prefix: str = "humangr_"  # NUNCA "pipeline_"

    @property
    def qdrant_context_collection(self) -> str:
        return f"{self.qdrant_collection_prefix}context_packs"

    # Sprint Range
    sprint_start: str = "S00"
    sprint_end: str = "S40"

    # LLM
    llm_provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "anthropic")
    )

    def validate(self) -> list[str]:
        """Validate configuration and return list of issues."""
        issues = []

        if not self.context_packs_dir.exists():
            issues.append(f"Context packs dir not found: {self.context_packs_dir}")

        if self.qdrant_collection_prefix != "humangr_":
            issues.append(f"CRITICAL: qdrant prefix must be 'humangr_', got '{self.qdrant_collection_prefix}'")

        if self.product_id != "HUMANGR":
            issues.append(f"CRITICAL: product_id must be 'HUMANGR', got '{self.product_id}'")

        return issues


# Singleton
_config: Optional[HumanGRConfig] = None


def get_config() -> HumanGRConfig:
    """Get or create the singleton config."""
    global _config
    if _config is None:
        _config = HumanGRConfig()
        issues = _config.validate()
        if issues:
            for issue in issues:
                print(f"WARNING: {issue}")
    return _config


def reset_config() -> None:
    """Reset config (for testing)."""
    global _config
    _config = None
