# S21 - docs-guides | Context Pack v1.0

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
  id: S21
  name: docs-guides
  title: "Docs: Guides"
  wave: W2-OSSRelease
  priority: P1-HIGH
  type: documentation

objective: "Guias de uso detalhados"

source_block: "../MASTER_REQUIREMENTS_MAP.md#PARTE-8-DOCS"

dependencies:
  - S20  # Docs Getting Started

deliverables:
  - docs/guides/layers.md
  - docs/guides/perspectives.md
  - docs/guides/cicd.md
  - docs/guides/configuration.md
```

---

## GUIDES OVERVIEW

```yaml
guides:
  layers.md: "Deep dive into the 7 layers"
  perspectives.md: "The 6 testing perspectives"
  cicd.md: "CI/CD integration"
  configuration.md: "Advanced configuration"
```

---

## CONTENT SPECS

### layers.md (Abbreviated)

```markdown
# The 7 Human Layers

Deep dive into each validation layer.

## Layer Philosophy

Each layer examines the action from a different angle:
- Layers 1-3: Quality (UX, Functionality, Edge Cases)
- Layer 4: Security (STRONG veto)
- Layer 5: Efficiency (Performance)
- Layers 6-7: Governance (Compliance, Final Review - STRONG veto)

## HL-1: UX & Usability

**Veto Power:** WEAK

Examines user experience aspects:
- UI clarity and consistency
- Error message quality
- Accessibility (WCAG basics)
- Navigation flow
- Cognitive load

**Example findings:**
- "Error message 'Error 500' is not helpful"
- "Missing confirmation for destructive action"
- "Color contrast insufficient for accessibility"

## HL-4: Security

**Veto Power:** STRONG

Examines security vulnerabilities:
- OWASP Top 10
- Injection attacks
- Authentication issues
- Data exposure
- Cryptographic failures

**Example findings:**
- "SQL injection vulnerability in user input"
- "Hardcoded API key in source code"
- "Missing rate limiting on authentication endpoint"

[Continue for all 7 layers...]
```

### cicd.md (Abbreviated)

```markdown
# CI/CD Integration

Integrate Human Layer into your pipeline.

## GitHub Actions

```yaml
# .github/workflows/human-layer.yml
name: Human Layer Validation

on:
  pull_request:
    branches: [main]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install Human Layer
        run: pip install human-layer

      - name: Validate Changes
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          human-layer validate --diff
```

## GitLab CI

```yaml
# .gitlab-ci.yml
human-layer:
  stage: test
  script:
    - pip install human-layer
    - human-layer validate --diff
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
```

## Pre-commit Hook

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: human-layer
        name: Human Layer Validation
        entry: human-layer validate --diff
        language: system
        pass_filenames: false
```
```

---

## INTENT MANIFEST

```yaml
INTENT_MANIFEST:
  RF:
    - RF-001: "layers.md explica todas as 7 layers"
    - RF-002: "perspectives.md explica 6 perspectives"
    - RF-003: "cicd.md tem GitHub, GitLab, pre-commit"
    - RF-004: "configuration.md = opções avançadas"
    - RF-005: "Exemplos práticos em todos"

  INV:
    - INV-001: "Exemplos de código funcionam"
    - INV-002: "YAML válido"
    - INV-003: "Links funcionam"
```

---

## GATES

```yaml
gates:
  G0_FILES_EXIST:
    description: "Guides existem"
    validation: |
      ls docs/guides/layers.md
      ls docs/guides/perspectives.md
      ls docs/guides/cicd.md

  G1_YAML_VALID:
    description: "YAML em exemplos é válido"
    validation: |
      python -c "
      import yaml
      from pathlib import Path
      for guide in Path('docs/guides').glob('*.md'):
          content = guide.read_text()
          # Extract YAML blocks and validate
      "
```

---

## REFERÊNCIA

- `./S20_CONTEXT.md` - Docs Getting Started
- `./S22_CONTEXT.md` - Docs API Reference
