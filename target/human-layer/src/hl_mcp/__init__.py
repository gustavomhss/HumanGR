"""
Human Layer MCP Server

7 Layers of Human Judgment for AI Agent Validation.

Example:
    >>> from hl_mcp import validate
    >>> result = validate("Delete all user data from production")
    >>> print(result.decision)  # REJECTED
    >>> print(result.veto_layer)  # HL-4 Security (STRONG veto)
"""

__version__ = "0.1.0"
__author__ = "HumanGR"
__license__ = "Apache-2.0"

# Public API will be exported here as modules are implemented
# from .core import validate, ValidationResult
# from .layers import Layer, LayerResult
# from .consensus import ConsensusEngine

__all__ = [
    "__version__",
    "__author__",
    "__license__",
]
