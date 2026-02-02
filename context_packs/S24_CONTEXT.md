# S24 - oss-launch | Context Pack v1.0

---

## PRODUCT REFERENCE

```yaml
product_reference:
  product_id: "HUMANGR"
  product_name: "Human Layer MCP Server"
  wave: "W2-OSSRelease"
  product_pack: "./PRODUCT_PACK.md"
  sprint_index: "./SPRINT_INDEX.yaml"
```

---

## RELOAD ANCHOR

```yaml
sprint:
  id: S24
  name: oss-launch
  title: "OSS Launch Milestone"
  wave: W2-OSSRelease
  priority: P0-CRITICAL
  type: milestone

objective: "🚀 Lançar Human Layer OSS no PyPI"

source_block: "../MASTER_REQUIREMENTS_MAP.md#PARTE-8-DOCS"

dependencies:
  - S23  # GitHub OSS Setup

deliverables:
  - Release v1.0.0
  - PyPI package published
  - GitHub release created
  - Announcement ready
```

---

## MILESTONE CHECKLIST

```yaml
milestone_checklist:
  code_complete:
    - "All W0 sprints complete (S00-S02)"
    - "All W1 sprints complete (S03-S14)"
    - "All W2 sprints complete (S15-S23)"
    - "Tests passing (>85% coverage)"
    - "No critical bugs"

  documentation:
    - "README.md polished"
    - "Getting started guide"
    - "API reference"
    - "CI/CD guides"

  legal:
    - "Apache 2.0 license"
    - "CONTRIBUTING.md"
    - "CODE_OF_CONDUCT.md"
    - "No license conflicts in dependencies"

  infrastructure:
    - "PyPI account ready"
    - "GitHub repo public"
    - "CI/CD working"
    - "Release workflow tested"
```

---

## RELEASE PROCESS

### 1. Pre-release Checks

```bash
# Run full test suite
pytest tests/ -v --cov=src --cov-fail-under=85

# Check types
mypy src/

# Check linting
ruff check src/

# Check all imports work
python -c "import hl_mcp; print(hl_mcp.__version__)"

# Test CLI
human-layer version
human-layer validate "test action"
```

### 2. Version Bump

```python
# Update src/hl_mcp/__init__.py
__version__ = "1.0.0"

# Update pyproject.toml
version = "1.0.0"
```

### 3. Create Release

```bash
# Create git tag
git tag -a v1.0.0 -m "Release v1.0.0 - Human Layer OSS Launch"
git push origin v1.0.0

# GitHub release will trigger via workflow
```

### 4. PyPI Publish

The `.github/workflows/release.yml` handles this automatically on tag push:

```yaml
name: Release
on:
  push:
    tags:
      - 'v*'

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install build tools
        run: pip install build twine

      - name: Build package
        run: python -m build

      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_TOKEN }}
        run: twine upload dist/*
```

### 5. Verify Release

```bash
# Install from PyPI
pip install human-layer

# Verify version
python -c "import hl_mcp; print(hl_mcp.__version__)"
# Should print: 1.0.0

# Test functionality
human-layer version
human-layer validate "test"
```

---

## ANNOUNCEMENT TEMPLATE

```markdown
# 🚀 Human Layer v1.0.0 - OSS Launch!

We're excited to announce Human Layer - an MCP server that provides
7 layers of human judgment for AI agent validation.

## Highlights

- **7 Human Layers** with WEAK/MEDIUM/STRONG veto powers
- **Triple Redundancy** - 3 runs per layer, 2/3 consensus
- **BYOK** - Use your existing LLM subscription
- **MCP Native** - Works with Claude Desktop, Cline, Continue.dev

## Install Now

```bash
pip install human-layer
human-layer init
human-layer validate "your action here"
```

## Links

- 📦 PyPI: https://pypi.org/project/human-layer/
- 📖 Docs: https://docs.humangr.ai
- 🐙 GitHub: https://github.com/humangr/human-layer

## What's Next

- Cloud version with dashboard
- More LLM providers
- Team collaboration features

Thank you to everyone who contributed!
```

---

## INTENT MANIFEST

```yaml
INTENT_MANIFEST:
  RF:
    - RF-001: "Release v1.0.0 no PyPI"
    - RF-002: "GitHub release com changelog"
    - RF-003: "Tests >85% coverage"
    - RF-004: "Docs completa"
    - RF-005: "Announcement ready"

  INV:
    - INV-001: "Versão consistente em todos os arquivos"
    - INV-002: "No critical bugs"
    - INV-003: "Apache 2.0 license"
    - INV-004: "PyPI token configurado"

  EDGE:
    - EDGE-001: "PyPI upload fail → retry"
    - EDGE-002: "Version já existe → increment"
```

---

## GATES

```yaml
gates:
  G0_CODE_COMPLETE:
    description: "Código completo"
    validation: "pytest tests/ --cov=src --cov-fail-under=85"

  G1_DOCS_COMPLETE:
    description: "Docs completa"
    validation: |
      ls docs/index.md
      ls docs/getting-started.md
      ls README.md

  G2_VERSION_CONSISTENT:
    description: "Versão consistente"
    validation: |
      python -c "
      from hl_mcp import __version__
      import tomllib
      with open('pyproject.toml', 'rb') as f:
          toml = tomllib.load(f)
      assert __version__ == toml['project']['version']
      "

  G3_PYPI_READY:
    description: "PyPI pronto"
    validation: |
      python -m build
      ls dist/*.whl
      ls dist/*.tar.gz
```

---

## 🎉 MILESTONE: OSS LAUNCH

```
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║                    🚀 HUMAN LAYER v1.0.0 🚀                        ║
║                                                                    ║
║         7 Layers of Human Judgment for AI Agent Validation         ║
║                                                                    ║
║    pip install human-layer                                         ║
║                                                                    ║
║    Wave 0: Foundation ✅                                           ║
║    Wave 1: Core Engine ✅                                          ║
║    Wave 2: OSS Release ✅                                          ║
║                                                                    ║
║    Next: Wave 3 - Cloud MVP                                        ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
```

---

## REFERÊNCIA

- `./S23_CONTEXT.md` - GitHub OSS Setup
- `./S25_CONTEXT.md` - Cloud Infrastructure (Wave 3)
