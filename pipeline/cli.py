#!/usr/bin/env python3
"""
HumanGR Pipeline CLI

CLI para executar o pipeline HumanGR.
Completamente isolado do pipeline Veritas.

Uso:
    python -m pipeline.cli status
    python -m pipeline.cli list
    python -m pipeline.cli info S00
    python -m pipeline.cli check S00
    python -m pipeline.cli start --start S00 --end S05
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import get_config
from pipeline.pack_loader import (
    list_sprints,
    load_context_pack,
    get_sprint_info,
    get_sprint_range,
    verify_deliverables,
)
from pipeline.state import create_initial_state, validate_state


def cmd_status(args):
    """Show pipeline status and configuration."""
    config = get_config()
    issues = config.validate()

    print("=" * 60)
    print("HUMANGR PIPELINE STATUS")
    print("=" * 60)
    print()
    print(f"Product ID:      {config.product_id}")
    print(f"Product Name:    {config.product_name}")
    print(f"Project Root:    {config.project_root}")
    print(f"Context Packs:   {config.context_packs_dir}")
    print(f"Target Repo:     {config.target_repo}")
    print(f"Output Dir:      {config.output_dir}")
    print()
    print(f"Qdrant Host:     {config.qdrant_host}:{config.qdrant_port}")
    print(f"Qdrant Collection: {config.qdrant_context_collection}")
    print()
    print(f"Sprint Range:    {config.sprint_start} - {config.sprint_end}")
    print()

    # List sprints
    sprints = list_sprints()
    print(f"Available Sprints: {len(sprints)}")
    print(f"  {', '.join(sprints[:10])}{'...' if len(sprints) > 10 else ''}")
    print()

    # Validation
    if issues:
        print("ISSUES:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Status: OK")

    print("=" * 60)


def cmd_list(args):
    """List all available sprints."""
    sprints = list_sprints()

    if args.format == "json":
        print(json.dumps(sprints))
    else:
        print(f"Available Sprints ({len(sprints)}):")
        print("-" * 40)
        for sprint_id in sprints:
            info = get_sprint_info(sprint_id)
            print(f"  {sprint_id}: {info['title'][:50]}")


def cmd_info(args):
    """Show detailed info about a sprint."""
    sprint_id = args.sprint_id.upper()

    try:
        pack = load_context_pack(sprint_id)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print("=" * 60)
    print(f"SPRINT: {sprint_id}")
    print("=" * 60)
    print()
    print(f"Title:      {pack.get('title', 'N/A')}")
    print(f"Wave:       {pack.get('wave', 'N/A')}")
    print(f"Priority:   {pack.get('priority', 'N/A')}")
    print(f"Product:    {pack.get('product_id', 'N/A')}")
    print()
    print(f"Objective:")
    print(f"  {pack.get('objective', 'N/A')}")
    print()
    print(f"Dependencies: {pack.get('dependencies', [])}")
    print()
    print(f"Deliverables ({len(pack.get('deliverables', []))}):")
    for d in pack.get("deliverables", [])[:10]:
        print(f"  - {d}")
    if len(pack.get("deliverables", [])) > 10:
        print(f"  ... and {len(pack.get('deliverables', [])) - 10} more")
    print()
    print("=" * 60)


def cmd_check(args):
    """Check deliverables status for a sprint."""
    sprint_id = args.sprint_id.upper()

    try:
        result = verify_deliverables(sprint_id)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print("=" * 60)
    print(f"DELIVERABLES CHECK: {sprint_id}")
    print("=" * 60)
    print()
    print(f"Total:    {result['total']}")
    print(f"Found:    {result['found']}")
    print(f"Missing:  {len(result['missing'])}")
    print(f"Progress: {result['percentage']}%")
    print()

    if result["existing"]:
        print("Existing:")
        for d in result["existing"][:5]:
            print(f"  {d}")
        if len(result["existing"]) > 5:
            print(f"  ... and {len(result['existing']) - 5} more")
        print()

    if result["missing"]:
        print("Missing:")
        for d in result["missing"][:10]:
            print(f"  {d}")
        if len(result["missing"]) > 10:
            print(f"  ... and {len(result['missing']) - 10} more")

    print()
    print("=" * 60)


def cmd_start(args):
    """Start pipeline execution (placeholder)."""
    start = args.start.upper()
    end = args.end.upper()

    print("=" * 60)
    print("HUMANGR PIPELINE - START")
    print("=" * 60)
    print()

    # Create state
    state = create_initial_state(start, end)
    issues = validate_state(state)

    if issues:
        print("STATE VALIDATION FAILED:")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)

    print(f"Run ID:       {state.run_id}")
    print(f"Project ID:   {state.project_id}")
    print(f"Sprint Range: {start} - {end}")
    print()

    # Get sprint range
    try:
        sprints = get_sprint_range(start, end)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"Sprints to execute: {len(sprints)}")
    for s in sprints:
        info = get_sprint_info(s)
        print(f"  {s}: {info['title'][:40]}")

    print()
    print("-" * 60)
    print("NOTE: Full execution not yet implemented.")
    print("This is a preview of what would run.")
    print("-" * 60)
    print()
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="HumanGR Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # status
    status_parser = subparsers.add_parser("status", help="Show pipeline status")

    # list
    list_parser = subparsers.add_parser("list", help="List available sprints")
    list_parser.add_argument("--format", choices=["text", "json"], default="text")

    # info
    info_parser = subparsers.add_parser("info", help="Show sprint details")
    info_parser.add_argument("sprint_id", help="Sprint ID (e.g., S00)")

    # check
    check_parser = subparsers.add_parser("check", help="Check sprint deliverables")
    check_parser.add_argument("sprint_id", help="Sprint ID (e.g., S00)")

    # start
    start_parser = subparsers.add_parser("start", help="Start pipeline execution")
    start_parser.add_argument("--start", "-s", default="S00", help="Start sprint")
    start_parser.add_argument("--end", "-e", default="S40", help="End sprint")

    args = parser.parse_args()

    if args.command == "status":
        cmd_status(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "info":
        cmd_info(args)
    elif args.command == "check":
        cmd_check(args)
    elif args.command == "start":
        cmd_start(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
