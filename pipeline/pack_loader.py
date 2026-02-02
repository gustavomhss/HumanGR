"""
HumanGR Context Pack Loader

Carrega context packs APENAS do diretório HL-MCP/context_packs/.
NUNCA acessa context_packs/ ou qualquer path do brains.
"""

import re
import yaml
from pathlib import Path
from typing import Any, Optional

from .config import get_config


def list_sprints() -> list[str]:
    """
    List all available sprint IDs.

    Returns:
        List of sprint IDs like ["S00", "S01", ..., "S40"]
    """
    config = get_config()
    packs_dir = config.context_packs_dir

    if not packs_dir.exists():
        raise FileNotFoundError(f"Context packs directory not found: {packs_dir}")

    sprints = []
    for f in sorted(packs_dir.glob("S*_CONTEXT.md")):
        # Extract sprint ID from filename (S00_CONTEXT.md -> S00)
        match = re.match(r"(S\d+)_CONTEXT\.md", f.name)
        if match:
            sprints.append(match.group(1))

    return sprints


def load_context_pack(sprint_id: str) -> dict[str, Any]:
    """
    Load a context pack by sprint ID.

    Args:
        sprint_id: Sprint identifier (e.g., "S00", "S09")

    Returns:
        Dictionary with parsed context pack data

    Raises:
        FileNotFoundError: If context pack doesn't exist
    """
    config = get_config()
    context_path = config.context_packs_dir / f"{sprint_id}_CONTEXT.md"

    if not context_path.exists():
        raise FileNotFoundError(
            f"Context pack not found: {context_path}\n"
            f"Available sprints: {list_sprints()}"
        )

    content = context_path.read_text()

    # Parse the context pack
    pack = {
        "sprint_id": sprint_id,
        "raw_content": content,
        "product_id": "HUMANGR",  # Always HumanGR
    }

    # Extract YAML blocks
    pack["reload_anchor"] = _extract_yaml_block(content, "RELOAD ANCHOR")
    pack["product_reference"] = _extract_yaml_block(content, "PRODUCT REFERENCE")
    pack["intent_manifest"] = _extract_yaml_block(content, "INTENT MANIFEST")

    # Extract key fields from reload anchor
    if pack["reload_anchor"]:
        anchor = pack["reload_anchor"]
        if "sprint" in anchor:
            pack["title"] = anchor["sprint"].get("title", "")
            pack["wave"] = anchor["sprint"].get("wave", "")
            pack["priority"] = anchor["sprint"].get("priority", "")
        pack["objective"] = anchor.get("objective", "")
        pack["deliverables"] = anchor.get("deliverables", [])
        pack["dependencies"] = anchor.get("dependencies", [])

    return pack


def _extract_yaml_block(content: str, section_name: str) -> Optional[dict]:
    """Extract a YAML block from markdown content."""
    # Find section header
    pattern = rf"##\s+{re.escape(section_name)}\s*\n"
    match = re.search(pattern, content, re.IGNORECASE)
    if not match:
        return None

    # Find the next ```yaml block
    yaml_start = content.find("```yaml", match.end())
    if yaml_start == -1:
        return None

    yaml_end = content.find("```", yaml_start + 7)
    if yaml_end == -1:
        return None

    yaml_content = content[yaml_start + 7:yaml_end].strip()

    try:
        return yaml.safe_load(yaml_content)
    except yaml.YAMLError:
        return None


def get_sprint_info(sprint_id: str) -> dict[str, Any]:
    """
    Get summary info about a sprint without loading full content.

    Args:
        sprint_id: Sprint identifier

    Returns:
        Dictionary with sprint summary
    """
    pack = load_context_pack(sprint_id)
    return {
        "sprint_id": sprint_id,
        "title": pack.get("title", ""),
        "wave": pack.get("wave", ""),
        "priority": pack.get("priority", ""),
        "objective": pack.get("objective", ""),
        "deliverables_count": len(pack.get("deliverables", [])),
        "dependencies": pack.get("dependencies", []),
    }


def get_sprint_range(start: str, end: str) -> list[str]:
    """
    Get list of sprints in a range.

    Args:
        start: Start sprint ID (e.g., "S00")
        end: End sprint ID (e.g., "S10")

    Returns:
        List of sprint IDs in range
    """
    all_sprints = list_sprints()

    try:
        start_idx = all_sprints.index(start)
    except ValueError:
        raise ValueError(f"Start sprint {start} not found. Available: {all_sprints}")

    try:
        end_idx = all_sprints.index(end)
    except ValueError:
        raise ValueError(f"End sprint {end} not found. Available: {all_sprints}")

    if start_idx > end_idx:
        raise ValueError(f"Start {start} comes after end {end}")

    return all_sprints[start_idx:end_idx + 1]


def verify_deliverables(sprint_id: str) -> dict[str, Any]:
    """
    Check which deliverables exist for a sprint.

    Args:
        sprint_id: Sprint identifier

    Returns:
        Dictionary with verification results
    """
    config = get_config()
    pack = load_context_pack(sprint_id)
    deliverables = pack.get("deliverables", [])

    results = {
        "sprint_id": sprint_id,
        "total": len(deliverables),
        "found": 0,
        "missing": [],
        "existing": [],
    }

    for deliverable in deliverables:
        path = config.target_repo / deliverable
        if path.exists():
            results["found"] += 1
            results["existing"].append(deliverable)
        else:
            results["missing"].append(deliverable)

    results["complete"] = results["found"] == results["total"]
    results["percentage"] = (
        round(results["found"] / results["total"] * 100, 1)
        if results["total"] > 0
        else 0
    )

    return results
