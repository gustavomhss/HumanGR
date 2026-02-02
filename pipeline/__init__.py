"""
HumanGR Pipeline - Fork isolado do Brains Pipeline

Este é um pipeline COMPLETAMENTE SEPARADO do brains.
Nunca importa nada de pipeline.

SEPARAÇÃO ABSOLUTA:
- Qdrant collection: humangr_* (NUNCA pipeline_*)
- Context packs: HL-MCP/context_packs/ (NUNCA brains/)
- Target repo: HL-MCP/target/human-layer/ (NUNCA brains/)
"""

__version__ = "1.0.0"
__product_id__ = "HUMANGR"

from .config import HumanGRConfig, get_config
from .pack_loader import load_context_pack, list_sprints
from .state import PipelineState, create_initial_state

__all__ = [
    "HumanGRConfig",
    "get_config",
    "load_context_pack",
    "list_sprints",
    "PipelineState",
    "create_initial_state",
]
