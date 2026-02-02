# S20 - docs-getting-started | Context Pack v1.0

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
  id: S20
  name: docs-getting-started
  title: "Docs: Getting Started"
  wave: W2-OSSRelease
  priority: P0-CRITICAL
  type: documentation

objective: "Documentação inicial para OSS users"

source_block: "../MASTER_REQUIREMENTS_MAP.md#PARTE-8-DOCS"

dependencies:
  - S19  # CLI Commands

deliverables:
  - docs/index.md
  - docs/getting-started.md
  - docs/quickstart.md
  - docs/installation.md
```

---

## DOCUMENTATION STRUCTURE

```yaml
docs_structure:
  index.md: "Landing page, overview"
  getting-started.md: "Full setup guide"
  quickstart.md: "5-minute setup"
  installation.md: "Installation options"
```

---

## CONTENT SPECS

### index.md

```markdown
# Human Layer

> 7 Layers of Human Judgment for AI Agent Validation

Human Layer is an MCP server that validates AI agent actions through
7 specialized review layers, ensuring safety before execution.

## Key Features

- **7 Human Layers** - UX, Functionality, Edge Cases, Security, Performance, Compliance, Final Review
- **Triple Redundancy** - 3 runs per layer, 2/3 consensus required
- **STRONG Veto Power** - Security, Compliance, and Final Review can block everything
- **BYOK** - Bring Your Own Key (use your existing LLM subscription)
- **MCP Native** - Works with Claude Desktop, Cline, Continue.dev

## Quick Example

```bash
# Install
pip install human-layer

# Initialize
human-layer init

# Validate an action
human-layer validate "Delete all user data from production"
# DECISION: REJECTED (HL-4 Security STRONG veto)
```

## The 7 Layers

| Layer | Name | Veto Power | Focus |
|-------|------|------------|-------|
| HL-1 | UX & Usability | WEAK | User experience |
| HL-2 | Functionality | MEDIUM | Correctness |
| HL-3 | Edge Cases | MEDIUM | Boundaries |
| HL-4 | Security | STRONG | Vulnerabilities |
| HL-5 | Performance | MEDIUM | Efficiency |
| HL-6 | Compliance | STRONG | Regulations |
| HL-7 | Final Review | STRONG | Last check |

## Next Steps

- [Quickstart](quickstart.md) - Get running in 5 minutes
- [Getting Started](getting-started.md) - Full setup guide
- [Installation](installation.md) - Installation options
```

### quickstart.md

```markdown
# Quickstart

Get Human Layer running in 5 minutes.

## Prerequisites

- Python 3.9+
- An LLM API key (Claude, OpenAI, Gemini, or Ollama)

## Installation

```bash
pip install human-layer
```

## Setup

```bash
# Initialize configuration
human-layer init

# Edit .env with your API key
cp .env.example .env
# Add: ANTHROPIC_API_KEY=sk-ant-...
```

## Try It

```bash
# Validate a safe action
human-layer validate "Add logging to user service"
# DECISION: APPROVED

# Validate a risky action
human-layer validate "Disable authentication for testing"
# DECISION: REJECTED (HL-4 Security STRONG veto)
```

## Use with Claude Desktop

Add to your Claude Desktop config (`claude_desktop_config.json`):

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

Now Claude will validate actions through Human Layer!

## Next Steps

- [Full Getting Started Guide](getting-started.md)
- [Configure Layers](guides/layers.md)
- [CI/CD Integration](guides/cicd.md)
```

### getting-started.md

```markdown
# Getting Started

Complete guide to setting up Human Layer.

## Installation Options

### pip (Recommended)

```bash
pip install human-layer
```

### With specific LLM support

```bash
pip install human-layer[anthropic]  # Claude support
pip install human-layer[openai]     # OpenAI support
pip install human-layer[all]        # All providers
```

### From source

```bash
git clone https://github.com/humangr/human-layer
cd human-layer
pip install -e ".[dev]"
```

## Configuration

### Initialize

```bash
human-layer init
```

This creates:
- `human-layer.yaml` - Main configuration
- `.env.example` - Environment template

### Configure LLM (BYOK)

Copy and edit `.env`:

```bash
cp .env.example .env
```

Add your API key:

```bash
# For Claude (recommended)
ANTHROPIC_API_KEY=sk-ant-...

# For OpenAI
OPENAI_API_KEY=sk-...

# For Gemini
GOOGLE_API_KEY=...

# For Ollama (no key needed)
# Just have Ollama running locally
```

### Customize Layers

Edit `human-layer.yaml`:

```yaml
layers:
  HL1:
    enabled: true
  HL4:
    enabled: true
    # Security is always recommended
  HL6:
    enabled: false
    # Disable if no compliance requirements
```

## Usage

### CLI Validation

```bash
# Simple validation
human-layer validate "Create new user endpoint"

# Validate a file
human-layer validate --file src/api/users.py

# Validate git changes
human-layer validate --diff
```

### MCP Server

```bash
# Start for Claude Desktop
human-layer serve

# Start HTTP server
human-layer serve --transport http --port 8080
```

### Integration

See [CI/CD Guide](guides/cicd.md) for:
- GitHub Actions
- GitLab CI
- Pre-commit hooks

## Understanding Results

### Decision Types

- **APPROVED** - Action is safe to proceed
- **REJECTED** - Action blocked (STRONG veto or 2+ MEDIUM)
- **NEEDS_REVIEW** - Single MEDIUM veto, needs human review

### Veto Levels

- **NONE** - No concerns
- **WEAK** - Advisory (HL-1 only)
- **MEDIUM** - Significant concern
- **STRONG** - Critical issue, blocks everything

## Troubleshooting

### "API key required"

Make sure your `.env` file has the correct API key set.

### "Layer timeout"

Increase timeout in `human-layer.yaml`:

```yaml
llm:
  timeout_seconds: 120
```

### "Rate limit"

You've hit your LLM provider's rate limit. Wait or upgrade your plan.
```

### installation.md

```markdown
# Installation

## Requirements

- Python 3.9 or higher
- pip or uv package manager

## Install from PyPI

```bash
# Basic installation
pip install human-layer

# With specific provider support
pip install human-layer[anthropic]
pip install human-layer[openai]
pip install human-layer[all]

# With development dependencies
pip install human-layer[dev]
```

## Install from Source

```bash
git clone https://github.com/humangr/human-layer
cd human-layer
pip install -e .

# Or with uv (faster)
uv pip install -e .
```

## Verify Installation

```bash
human-layer version
# Human Layer v1.0.0
```

## Platform-Specific Notes

### macOS

No additional requirements.

### Linux

No additional requirements.

### Windows

Requires Python 3.9+ from python.org (not Windows Store).

## Docker

```bash
docker run -it humangr/human-layer validate "your action"
```

## Next Steps

- [Quickstart](quickstart.md)
- [Configuration](getting-started.md#configuration)
```

---

## INTENT MANIFEST

```yaml
INTENT_MANIFEST:
  RF:
    - RF-001: "index.md é landing page"
    - RF-002: "quickstart.md = 5 minutos"
    - RF-003: "getting-started.md = guia completo"
    - RF-004: "installation.md = todas as opções"
    - RF-005: "Exemplos práticos em todos"

  INV:
    - INV-001: "Código em exemplos funciona"
    - INV-002: "Links internos válidos"
    - INV-003: "Sem credenciais de exemplo reais"
    - INV-004: "Versão correta mencionada"

  EDGE:
    - EDGE-001: "Troubleshooting para erros comuns"
    - EDGE-002: "Platform-specific notes"
    - EDGE-003: "Docker option"
```

---

## GATES

```yaml
gates:
  G0_FILES_EXIST:
    description: "Docs existem"
    validation: |
      ls docs/index.md
      ls docs/getting-started.md
      ls docs/quickstart.md
      ls docs/installation.md

  G1_LINKS_VALID:
    description: "Links internos válidos"
    validation: |
      python -c "
      import re
      from pathlib import Path
      docs = Path('docs')
      for md in docs.glob('*.md'):
          content = md.read_text()
          links = re.findall(r'\[.*?\]\((.*?\.md)\)', content)
          for link in links:
              if not (docs / link).exists():
                  print(f'Broken link in {md}: {link}')
      "
```

---

## REFERÊNCIA

- `./S19_CONTEXT.md` - CLI Commands
- `./S21_CONTEXT.md` - Docs Guides
