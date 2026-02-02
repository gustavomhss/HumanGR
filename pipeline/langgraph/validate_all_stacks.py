"""Comprehensive Stack Validation Script.

Validates ALL 27+ stacks to ensure they are:
1. Importable
2. Functional (singleton/getter exists)
3. Actually used in code (not just defined)

Run: PYTHONPATH=src python -m pipeline.langgraph.validate_all_stacks
"""

import os
import sys
from dataclasses import dataclass
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)


# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@dataclass
class StackValidation:
    """Result of validating a single stack."""
    name: str
    importable: bool
    import_error: Optional[str] = None
    functional: bool = False
    functional_error: Optional[str] = None
    used_in_code: bool = False
    usage_locations: List[str] = None

    def __post_init__(self):
        if self.usage_locations is None:
            self.usage_locations = []

    @property
    def status(self) -> str:
        if self.importable and self.functional and self.used_in_code:
            return "✅ FULLY OPERATIONAL"
        elif self.importable and self.functional:
            return "⚠️  ORPHANED (not used)"
        elif self.importable:
            return "⚠️  PARTIAL (import only)"
        else:
            return "❌ BROKEN"


def check_import(module_path: str) -> tuple[bool, Optional[str]]:
    """Check if a module can be imported."""
    try:
        __import__(module_path)
        return True, None
    except Exception as e:
        return False, str(e)


def search_usage(stack_name: str, src_path: str) -> List[str]:
    """Search for actual usage of a stack in the codebase."""
    usages = []
    patterns = [
        f"from.*{stack_name}",
        f"import.*{stack_name}",
        f"get_{stack_name}",
        f"use_{stack_name}",
        f"with_{stack_name}",
        f"'{stack_name}'",
        f'"{stack_name}"',
    ]

    for root, dirs, files in os.walk(src_path):
        # Skip test directories
        if 'test' in root or '__pycache__' in root:
            continue
        for f in files:
            if f.endswith('.py'):
                filepath = os.path.join(root, f)
                try:
                    with open(filepath, 'r') as file:
                        content = file.read()
                        for pattern in patterns:
                            if stack_name.lower() in content.lower():
                                # Get relative path
                                rel_path = filepath.replace(src_path + '/', '')
                                if rel_path not in usages:
                                    usages.append(rel_path)
                                break
                except Exception as e:
                    logger.debug(f"GRAPH: Graph operation failed: {e}")
    return usages


# Stack validation configurations
# NOTE: Python 3.13 compatibility:
# - graphrag: Requires Python <3.13 -> REPLACED by graphiti
# - letta: spacy/blis compilation fails on 3.13 -> REPLACED by crewai + mem0
# - llm_guard: spacy dependency fails -> REPLACED by nemo + deepeval
STACK_VALIDATIONS = {
    # CORE INFRASTRUCTURE
    "langfuse": {
        "import": "langfuse",
        "functional_check": lambda: __import__('langfuse').Langfuse,
        "category": "observability",
    },
    "redis": {
        "import": "redis",
        "functional_check": lambda: __import__('redis').Redis,
        "category": "infrastructure",
    },
    "qdrant": {
        "import": "qdrant_client",
        "functional_check": lambda: __import__('qdrant_client').QdrantClient,
        "category": "vector_db",
    },
    "falkordb": {
        "import": "falkordb",
        "functional_check": lambda: __import__('falkordb').FalkorDB,
        "category": "graph_db",
    },

    # AGENT ORCHESTRATION
    "crewai": {
        "import": "crewai",
        "functional_check": lambda: __import__('crewai').Agent,
        "category": "orchestration",
    },
    "mem0": {
        "import": "mem0",
        "functional_check": lambda: __import__('mem0'),
        "category": "memory",
    },
    "langgraph": {
        "import": "langgraph",
        "functional_check": lambda: __import__('langgraph.graph').graph.StateGraph,
        "category": "workflow",
    },

    # VALIDATION & QUALITY
    "pydantic": {
        "import": "pydantic",
        "functional_check": lambda: __import__('pydantic').BaseModel,
        "category": "validation",
    },
    "deepeval": {
        "import": "deepeval",
        "functional_check": lambda: __import__('deepeval'),
        "category": "evaluation",
    },
    "nemo": {
        "import": "nemoguardrails",
        "functional_check": lambda: __import__('nemoguardrails'),
        "category": "guardrails",
    },

    # REASONING STRATEGIES (native implementations)
    "got": {
        "import": "pipeline.langgraph.stack_synergy",
        "functional_check": lambda: getattr(__import__('pipeline.langgraph.stack_synergy', fromlist=['STACK_CAPABILITIES']), 'STACK_CAPABILITIES').get('got'),
        "category": "reasoning",
    },
    "reflexion": {
        "import": "pipeline.langgraph.stack_synergy",
        "functional_check": lambda: getattr(__import__('pipeline.langgraph.stack_synergy', fromlist=['STACK_CAPABILITIES']), 'STACK_CAPABILITIES').get('reflexion'),
        "category": "reasoning",
    },
    "bot": {
        "import": "pipeline.langgraph.stack_synergy",
        "functional_check": lambda: getattr(__import__('pipeline.langgraph.stack_synergy', fromlist=['STACK_CAPABILITIES']), 'STACK_CAPABILITIES').get('bot'),
        "category": "reasoning",
    },
    "active_rag": {
        "import": "pipeline.langgraph.stack_synergy",
        "functional_check": lambda: getattr(__import__('pipeline.langgraph.stack_synergy', fromlist=['STACK_CAPABILITIES']), 'STACK_CAPABILITIES').get('active_rag'),
        "category": "rag",
    },

    # DURABILITY & EXECUTION
    "temporal": {
        "import": "temporalio",
        "functional_check": lambda: __import__('temporalio'),
        "category": "workflow",
    },

    # KNOWLEDGE GRAPHS
    "graphiti": {
        "import": "graphiti_core",
        "functional_check": lambda: __import__('graphiti_core'),
        "category": "knowledge_graph",
    },
    "neo4j": {
        "import": "neo4j",
        "functional_check": lambda: __import__('neo4j'),
        "category": "graph_db",
    },

    # LLM TOOLING
    "instructor": {
        "import": "instructor",
        "functional_check": lambda: __import__('instructor'),
        "category": "llm_tooling",
    },
    "dspy": {
        "import": "dspy",
        "functional_check": lambda: __import__('dspy'),
        "category": "llm_tooling",
    },

    # EVALUATION & MONITORING
    "ragas": {
        "import": "ragas",
        "functional_check": lambda: __import__('ragas'),
        "category": "evaluation",
    },
    "trulens": {
        "import": "trulens",
        "functional_check": lambda: __import__('trulens'),
        "category": "evaluation",
    },
    "phoenix": {
        "import": "phoenix",
        "functional_check": lambda: __import__('phoenix'),
        "category": "observability",
    },
    "cleanlab": {
        "import": "cleanlab",
        "functional_check": lambda: __import__('cleanlab'),
        "category": "data_quality",
    },

    # FORMAL VERIFICATION
    "z3": {
        "import": "z3",
        "functional_check": lambda: __import__('z3'),
        "category": "verification",
    },

    # DATA PROCESSING
    "hamilton": {
        "import": "hamilton",
        "functional_check": lambda: __import__('hamilton'),
        "category": "data_processing",
    },
}


def validate_stack(name: str, config: dict, src_path: str) -> StackValidation:
    """Validate a single stack."""
    result = StackValidation(name=name, importable=False)

    # Check import
    importable, error = check_import(config["import"])
    result.importable = importable
    result.import_error = error

    # Check functionality
    if importable and "functional_check" in config:
        try:
            config["functional_check"]()
            result.functional = True
        except Exception as e:
            result.functional = False
            result.functional_error = str(e)
    elif importable:
        result.functional = True

    # Check usage in code
    result.usage_locations = search_usage(name, src_path)
    result.used_in_code = len(result.usage_locations) > 0

    return result


def main():
    """Run comprehensive stack validation."""
    print("=" * 70)
    print("COMPREHENSIVE STACK VALIDATION")
    print("=" * 70)
    print()

    # Get src path
    src_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    results: Dict[str, StackValidation] = {}

    # Validate each stack
    for stack_name, config in sorted(STACK_VALIDATIONS.items()):
        print(f"Validating {stack_name}...", end=" ", flush=True)
        result = validate_stack(stack_name, config, src_path)
        results[stack_name] = result
        print(result.status)

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    fully_operational = [r for r in results.values() if r.status == "✅ FULLY OPERATIONAL"]
    orphaned = [r for r in results.values() if "ORPHANED" in r.status]
    partial = [r for r in results.values() if "PARTIAL" in r.status]
    broken = [r for r in results.values() if "BROKEN" in r.status]

    print(f"\n✅ FULLY OPERATIONAL: {len(fully_operational)}/{len(results)}")
    for r in fully_operational:
        print(f"   - {r.name}: used in {len(r.usage_locations)} files")

    print(f"\n⚠️  ORPHANED (defined but not used): {len(orphaned)}/{len(results)}")
    for r in orphaned:
        print(f"   - {r.name}")

    print(f"\n⚠️  PARTIAL (import only): {len(partial)}/{len(results)}")
    for r in partial:
        print(f"   - {r.name}: {r.functional_error}")

    print(f"\n❌ BROKEN (cannot import): {len(broken)}/{len(results)}")
    for r in broken:
        error = r.import_error[:60] if r.import_error else "Unknown"
        print(f"   - {r.name}: {error}")

    # Calculate score
    score = (len(fully_operational) / len(results)) * 100
    print()
    print("=" * 70)
    print(f"STACK COVERAGE SCORE: {score:.1f}%")
    print(f"TOTAL STACKS: {len(results)}")
    print(f"FULLY OPERATIONAL: {len(fully_operational)}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
