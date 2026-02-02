# S23 - github-oss-setup | Context Pack v1.0

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
  id: S23
  name: github-oss-setup
  title: "GitHub & OSS Setup"
  wave: W2-OSSRelease
  priority: P0-CRITICAL
  type: setup

objective: "Setup GitHub profile e arquivos OSS"

source_block: "../MASTER_REQUIREMENTS_MAP.md#PARTE-8-DOCS"

dependencies:
  - S22  # Docs API

deliverables:
  - README.md
  - CONTRIBUTING.md
  - CODE_OF_CONDUCT.md
  - SECURITY.md
  - .github/ISSUE_TEMPLATE/bug_report.yml
  - .github/ISSUE_TEMPLATE/feature_request.yml
  - .github/PULL_REQUEST_TEMPLATE.md
```

---

## OSS FILES CONTENT

### README.md (Summary)

```markdown
# Human Layer 🛡️

> 7 Layers of Human Judgment for AI Agent Validation

[![PyPI](https://img.shields.io/pypi/v/human-layer)](https://pypi.org/project/human-layer/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Tests](https://github.com/humangr/human-layer/actions/workflows/ci.yml/badge.svg)](https://github.com/humangr/human-layer/actions)

Human Layer is an MCP server that validates AI agent actions through 7 specialized
review layers before they execute. Think of it as a "human review committee"
that never sleeps.

## Features

- 🔒 **7 Human Layers** with WEAK/MEDIUM/STRONG veto powers
- 🔄 **Triple Redundancy** - 3 runs per layer, 2/3 consensus
- 🎯 **6 Perspectives** - tired_user, malicious_insider, confused_newbie...
- 🔑 **BYOK** - Bring Your Own Key (use your existing LLM subscription)
- 🔌 **MCP Native** - Works with Claude Desktop, Cline, Continue.dev

## Quickstart

```bash
pip install human-layer
human-layer init
human-layer validate "Delete all production data"
# DECISION: REJECTED (HL-4 Security STRONG veto)
```

## The 7 Layers

| Layer | Name | Veto | Focus |
|-------|------|------|-------|
| HL-1 | UX & Usability | WEAK | User experience |
| HL-2 | Functionality | MEDIUM | Correctness |
| HL-3 | Edge Cases | MEDIUM | Boundaries |
| HL-4 | Security | **STRONG** | OWASP Top 10 |
| HL-5 | Performance | MEDIUM | Efficiency |
| HL-6 | Compliance | **STRONG** | GDPR, regulations |
| HL-7 | Final Review | **STRONG** | Last check |

## Documentation

- [Getting Started](docs/getting-started.md)
- [Quickstart](docs/quickstart.md)
- [API Reference](docs/api/index.md)
- [CI/CD Integration](docs/guides/cicd.md)

## License

Apache 2.0 - See [LICENSE](LICENSE)
```

### CONTRIBUTING.md (Summary)

```markdown
# Contributing to Human Layer

Thank you for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/humangr/human-layer
cd human-layer
pip install -e ".[dev]"
pre-commit install
```

## Running Tests

```bash
pytest tests/ -v
```

## Code Style

- Use ruff for linting
- Type hints required
- Docstrings on public functions

## Pull Request Process

1. Fork the repo
2. Create feature branch
3. Add tests for new features
4. Run `pytest` and `ruff`
5. Submit PR

## Code of Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
```

### SECURITY.md

```markdown
# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.x     | ✅        |
| < 1.0   | ❌        |

## Reporting a Vulnerability

Please report security vulnerabilities to: security@humangr.ai

Do NOT open public issues for security vulnerabilities.

We will respond within 48 hours and work with you to fix the issue.
```

### CODE_OF_CONDUCT.md

```markdown
# Code of Conduct

## Our Pledge

We pledge to make participation in our community a harassment-free experience
for everyone, regardless of age, body size, disability, ethnicity, sex
characteristics, gender identity and expression, level of experience,
education, socio-economic status, nationality, personal appearance, race,
religion, or sexual identity and orientation.

## Our Standards

Examples of behavior that contributes to a positive environment:
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Accepting constructive criticism gracefully
- Focusing on what is best for the community

## Enforcement

Instances of abusive behavior may be reported to: conduct@humangr.ai
```

---

## INTENT MANIFEST

```yaml
INTENT_MANIFEST:
  RF:
    - RF-001: "README.md com badges e quickstart"
    - RF-002: "CONTRIBUTING.md com setup e guidelines"
    - RF-003: "SECURITY.md com reporting process"
    - RF-004: "CODE_OF_CONDUCT.md inclusivo"
    - RF-005: "Issue templates estruturados"

  INV:
    - INV-001: "Badges apontam para repos corretos"
    - INV-002: "Links funcionam"
    - INV-003: "Emails são válidos"
```

---

## GATES

```yaml
gates:
  G0_FILES_EXIST:
    description: "OSS files existem"
    validation: |
      ls README.md
      ls CONTRIBUTING.md
      ls CODE_OF_CONDUCT.md
      ls SECURITY.md

  G1_BADGES_VALID:
    description: "Badges válidos"
    validation: |
      grep -q "pypi.org/project/human-layer" README.md
```

---

## REFERÊNCIA

- `./S22_CONTEXT.md` - Docs API
- `./S24_CONTEXT.md` - OSS Launch Milestone
