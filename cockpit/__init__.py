# cockpit/__init__.py
"""
HumanGR Cockpit - Pipeline dashboard for Human Layer MCP Server.

READ-ONLY observer that monitors pipeline state.
NEVER writes to pipeline files.

Standalone Flask server on port 5002.
"""
from pathlib import Path

COCKPIT_DIR = Path(__file__).parent

__version__ = "1.0.0"
__all__ = ["COCKPIT_DIR"]
