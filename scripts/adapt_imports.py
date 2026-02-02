#!/usr/bin/env python3
"""
Adapt imports from brains pipeline to HumanGR pipeline.

This script patches all Python files to:
1. Change imports from pipeline_autonomo -> pipeline
2. Change imports from pipeline_v2 -> pipeline
3. Change paths from docs/veritas_library -> context_packs
4. Change collection prefix from pipeline_ -> humangr_
5. Add HUMANGR product_id where needed
"""

import os
import re
from pathlib import Path

# Directories to process
PIPELINE_DIR = Path(__file__).parent.parent / "pipeline"

# Replacement rules
REPLACEMENTS = [
    # Import replacements
    (r"from pipeline_autonomo\.", "from pipeline."),
    (r"from pipeline_autonomo import", "from pipeline import"),
    (r"import pipeline_autonomo\.", "import pipeline."),
    (r"from pipeline_v2\.", "from pipeline."),
    (r"from pipeline_v2 import", "from pipeline import"),
    (r"import pipeline_v2\.", "import pipeline."),

    # Path replacements
    (r'docs/veritas_library/context_packs', 'context_packs'),
    (r'docs/veritas_library', 'context_packs'),
    (r'"veritas_library"', '"context_packs"'),
    (r"'veritas_library'", "'context_packs'"),

    # Collection prefix
    (r'collection_prefix:\s*str\s*=\s*"pipeline_"', 'collection_prefix: str = "humangr_"'),
    (r"collection_prefix:\s*str\s*=\s*'pipeline_'", "collection_prefix: str = 'humangr_'"),
    (r'"pipeline_context_packs"', '"humangr_context_packs"'),
    (r"'pipeline_context_packs'", "'humangr_context_packs'"),

    # Product ID
    (r'product_id\s*=\s*"VERITAS"', 'product_id = "HUMANGR"'),
    (r"product_id\s*=\s*'VERITAS'", "product_id = 'HUMANGR'"),
]

def patch_file(filepath: Path) -> tuple[int, list[str]]:
    """Patch a single file with replacements."""
    try:
        content = filepath.read_text()
    except Exception as e:
        return 0, [f"Error reading {filepath}: {e}"]

    original = content
    changes = []

    for pattern, replacement in REPLACEMENTS:
        new_content = re.sub(pattern, replacement, content)
        if new_content != content:
            # Count occurrences
            count = len(re.findall(pattern, content))
            changes.append(f"  {pattern[:40]}... -> {replacement[:30]}... ({count}x)")
            content = new_content

    if content != original:
        filepath.write_text(content)
        return len(changes), changes

    return 0, []

def main():
    print("=" * 60)
    print("ADAPTING IMPORTS FOR HUMANGR PIPELINE")
    print("=" * 60)
    print()

    total_files = 0
    total_changes = 0

    # Process all Python files
    for py_file in PIPELINE_DIR.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue

        num_changes, changes = patch_file(py_file)

        if num_changes > 0:
            total_files += 1
            total_changes += num_changes
            rel_path = py_file.relative_to(PIPELINE_DIR)
            print(f"{rel_path}: {num_changes} changes")
            for change in changes[:3]:  # Show first 3 changes
                print(change)
            if len(changes) > 3:
                print(f"  ... and {len(changes) - 3} more")
            print()

    print("=" * 60)
    print(f"SUMMARY: {total_changes} changes in {total_files} files")
    print("=" * 60)

if __name__ == "__main__":
    main()
