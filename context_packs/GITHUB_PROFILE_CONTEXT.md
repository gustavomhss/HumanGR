# GITHUB PROFILE | Context Pack v1.0

> **Fonte**: MASTER_REQUIREMENTS_MAP.md PARTE 18
> **Objetivo**: Perfil GitHub premium para a organização HumanGR

---

## PRODUCT REFERENCE

```yaml
product_reference:
  product_id: "HUMANGR"
  product_name: "Human Layer MCP Server"
  type: "github"
  source: "PARTE 18 - GitHub Profile (OSS Presentation)"
```

---

## ORGANIZATION PROFILE

### HumanGR Organization

```yaml
organization:
  name: "HumanGR"
  display_name: "HumanGR"
  bio: "7 Layers of Human Judgment for AI Agent Validation"
  url: "https://humangr.ai"
  twitter: "@humangr"
  location: "Global (Remote-First)"

  profile_readme: |
    # HumanGR 🛡️

    Building the **Human Layer** - an MCP server that validates AI agent
    actions through 7 specialized review layers.

    ## Our Mission

    Make AI agents safer by adding structured human judgment
    before they execute potentially harmful actions.

    ## Featured Project

    ### [Human Layer](https://github.com/humangr/human-layer)
    MCP Server for AI Agent Validation

    - 7 Human Layers (UX → Security → Compliance → Final Review)
    - Triple Redundancy (3 runs, 2/3 consensus)
    - BYOK (Bring Your Own Key)
    - 100% OSS Core (Apache 2.0)

    ```bash
    pip install human-layer
    human-layer validate "Your action here"
    ```

    ## Connect

    - 🌐 [Website](https://humangr.ai)
    - 📖 [Documentation](https://docs.humangr.ai)
    - 💬 [Discord](https://discord.gg/humangr)
    - 🐦 [Twitter](https://twitter.com/humangr)
```

---

## REPOSITORY README

### Main README.md

```markdown
# Human Layer 🛡️

[![PyPI](https://img.shields.io/pypi/v/human-layer)](https://pypi.org/project/human-layer/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Tests](https://github.com/humangr/human-layer/actions/workflows/ci.yml/badge.svg)](https://github.com/humangr/human-layer/actions)
[![Discord](https://img.shields.io/discord/XXXXX)](https://discord.gg/humangr)
[![Downloads](https://pepy.tech/badge/human-layer)](https://pepy.tech/project/human-layer)

> 7 Layers of Human Judgment for AI Agent Validation

Human Layer is an MCP server that validates AI agent actions through
7 specialized review layers, ensuring safety before execution.

<p align="center">
  <img src="docs/assets/demo.gif" alt="Human Layer Demo" width="600">
</p>

## 🚀 Quick Start

```bash
# Install
pip install human-layer

# Initialize
human-layer init

# Validate an action
human-layer validate "Delete all user data from production"
# DECISION: REJECTED (HL-4 Security STRONG veto)
```

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔒 **7 Human Layers** | UX, Functionality, Edge Cases, Security, Performance, Compliance, Final Review |
| 🔄 **Triple Redundancy** | 3 runs per layer, 2/3 consensus required |
| 🎯 **6 Perspectives** | tired_user, malicious_insider, confused_newbie, power_user, auditor, 3am_operator |
| 🔑 **BYOK** | Bring Your Own Key - use your existing LLM subscription |
| 🔌 **MCP Native** | Works with Claude Desktop, Cline, Continue.dev |
| 📊 **Dashboard** | Visual validation tracking (Cloud) |

## 🏗️ The 7 Layers

| Layer | Name | Veto Power | Focus |
|-------|------|------------|-------|
| HL-1 | UX & Usability | WEAK | User experience |
| HL-2 | Functionality | MEDIUM | Correctness |
| HL-3 | Edge Cases | MEDIUM | Boundaries |
| HL-4 | Security | **STRONG** | OWASP Top 10 |
| HL-5 | Performance | MEDIUM | Efficiency |
| HL-6 | Compliance | **STRONG** | GDPR, regulations |
| HL-7 | Final Review | **STRONG** | Last check |

## 🔧 Installation

### pip (Recommended)

```bash
pip install human-layer
```

### With specific LLM support

```bash
pip install human-layer[anthropic]  # Claude
pip install human-layer[openai]     # OpenAI
pip install human-layer[all]        # All providers
```

### From source

```bash
git clone https://github.com/humangr/human-layer
cd human-layer
pip install -e ".[dev]"
```

## 📖 Documentation

- [Getting Started](docs/getting-started.md)
- [Quickstart](docs/quickstart.md)
- [API Reference](docs/api/index.md)
- [CI/CD Integration](docs/guides/cicd.md)

## 💡 Use Cases

- **Code Review**: Validate AI-generated code before merge
- **API Actions**: Check agent API calls before execution
- **Data Operations**: Ensure data safety before modifications
- **Compliance**: Automated compliance checking

## 🔌 Integrations

### Claude Desktop

```json
{
  "mcpServers": {
    "human-layer": {
      "command": "human-layer",
      "args": ["serve"]
    }
  }
}
```

### GitHub Actions

```yaml
- uses: humangr/human-layer-action@v1
  with:
    api-key: ${{ secrets.ANTHROPIC_API_KEY }}
```

## 🏢 Pricing

| Tier | Price | Best For |
|------|-------|----------|
| OSS | Free | Self-hosted |
| Starter | $12/mo | Individuals |
| Pro | $49/mo | Teams |
| Business | $249/mo | Enterprise |

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Setup development environment
git clone https://github.com/humangr/human-layer
cd human-layer
pip install -e ".[dev]"
pre-commit install

# Run tests
pytest tests/ -v
```

## 📜 License

Apache 2.0 - See [LICENSE](LICENSE)

## 🔗 Links

- [Website](https://humangr.ai)
- [Documentation](https://docs.humangr.ai)
- [Discord](https://discord.gg/humangr)
- [Twitter](https://twitter.com/humangr)
- [Blog](https://humangr.ai/blog)

---

<p align="center">
  Built with ❤️ by <a href="https://humangr.ai">HumanGR</a>
</p>
```

---

## CONTRIBUTING.md

```markdown
# Contributing to Human Layer

Thank you for your interest in contributing! 🎉

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Pull Request Process](#pull-request-process)
- [Style Guide](#style-guide)

## Code of Conduct

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing.

## Getting Started

### Good First Issues

Look for issues labeled `good first issue` - these are great for newcomers!

### Types of Contributions

- 🐛 Bug fixes
- ✨ New features
- 📖 Documentation
- 🧪 Tests
- 🎨 UI/UX improvements

## Development Setup

```bash
# Clone the repo
git clone https://github.com/humangr/human-layer
cd human-layer

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest tests/ -v
```

## Making Changes

### Branch Naming

```
feature/add-new-layer
fix/security-validation-bug
docs/update-api-reference
test/add-consensus-tests
```

### Commit Messages

Follow [Conventional Commits](https://conventionalcommits.org/):

```
feat: add custom perspective support
fix: resolve race condition in consensus engine
docs: update quickstart guide
test: add edge case coverage for HL-4
```

## Pull Request Process

1. **Fork** the repository
2. **Create** your feature branch
3. **Make** your changes
4. **Run** tests: `pytest tests/ -v`
5. **Run** linting: `ruff check src/`
6. **Submit** pull request

### PR Template

```markdown
## Description
[What does this PR do?]

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Tests

## Testing
- [ ] Tests pass locally
- [ ] Added new tests

## Checklist
- [ ] Code follows style guide
- [ ] Self-reviewed
- [ ] Documented changes
```

## Style Guide

### Python

- **Formatter**: Black
- **Linter**: Ruff
- **Type hints**: Required
- **Docstrings**: Google style

```python
def validate_action(
    action: ActionContext,
    layers: list[LayerID] | None = None,
) -> ValidationResult:
    """Validate an action through Human Layers.

    Args:
        action: The action context to validate.
        layers: Optional list of specific layers to run.

    Returns:
        ValidationResult with decision and findings.

    Raises:
        ValidationError: If validation fails unexpectedly.
    """
    ...
```

### Tests

- Use pytest
- Aim for >90% coverage
- Test edge cases
- Mock external services

```python
def test_security_layer_detects_sql_injection():
    """HL-4 should detect SQL injection vulnerabilities."""
    action = ActionContext(
        action_type="database_query",
        description="SELECT * FROM users WHERE id = " + user_input,
    )
    result = hl4_layer.analyze(action)
    assert any(f.severity == Severity.CRITICAL for f in result.findings)
```

## Questions?

- 💬 [Discord](https://discord.gg/humangr)
- 📧 contributors@humangr.ai
```

---

## ISSUE TEMPLATES

### Bug Report (bug_report.yml)

```yaml
name: 🐛 Bug Report
description: Report a bug in Human Layer
labels: ["bug", "triage"]
body:
  - type: markdown
    attributes:
      value: |
        Thanks for reporting a bug! Please fill out the form below.

  - type: input
    id: version
    attributes:
      label: Version
      description: What version of human-layer are you using?
      placeholder: "1.0.0"
    validations:
      required: true

  - type: textarea
    id: description
    attributes:
      label: Bug Description
      description: What happened?
    validations:
      required: true

  - type: textarea
    id: expected
    attributes:
      label: Expected Behavior
      description: What should have happened?
    validations:
      required: true

  - type: textarea
    id: reproduce
    attributes:
      label: Steps to Reproduce
      description: How can we reproduce this?
      placeholder: |
        1. Run `human-layer validate "..."`
        2. See error
    validations:
      required: true

  - type: textarea
    id: logs
    attributes:
      label: Logs
      description: Relevant log output
      render: shell
```

### Feature Request (feature_request.yml)

```yaml
name: ✨ Feature Request
description: Suggest a new feature
labels: ["enhancement"]
body:
  - type: textarea
    id: problem
    attributes:
      label: Problem
      description: What problem does this solve?
    validations:
      required: true

  - type: textarea
    id: solution
    attributes:
      label: Proposed Solution
      description: How would you solve it?
    validations:
      required: true

  - type: textarea
    id: alternatives
    attributes:
      label: Alternatives Considered
      description: Other approaches you've considered

  - type: checkboxes
    id: contribution
    attributes:
      label: Contribution
      options:
        - label: I'm willing to implement this feature
```

---

## CI/CD WORKFLOWS

### ci.yml

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"

      - name: Lint
        run: ruff check src/

      - name: Type check
        run: mypy src/

      - name: Test
        run: pytest tests/ --cov=src --cov-report=xml -v

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        if: matrix.python-version == '3.11'
```

### release.yml

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

      - name: Build
        run: |
          pip install build
          python -m build

      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_TOKEN }}
        run: |
          pip install twine
          twine upload dist/*
```

---

## DELIVERABLES CHECKLIST

```yaml
github_deliverables:
  organization:
    - "[ ] Organization profile README"
    - "[ ] Organization settings configured"
    - "[ ] Teams created (core, community)"

  repository:
    - "[ ] README.md (comprehensive)"
    - "[ ] CONTRIBUTING.md"
    - "[ ] CODE_OF_CONDUCT.md"
    - "[ ] SECURITY.md"
    - "[ ] LICENSE (Apache 2.0)"
    - "[ ] CHANGELOG.md"

  templates:
    - "[ ] Bug report template"
    - "[ ] Feature request template"
    - "[ ] Pull request template"

  workflows:
    - "[ ] CI workflow (test, lint, type check)"
    - "[ ] Release workflow (PyPI publish)"
    - "[ ] Dependabot configuration"

  settings:
    - "[ ] Branch protection (main)"
    - "[ ] Required reviews"
    - "[ ] Required CI checks"
    - "[ ] Discussions enabled"
    - "[ ] Wiki disabled"
```

---

## REFERÊNCIA

Para detalhes completos: `../MASTER_REQUIREMENTS_MAP.md#PARTE-18`
