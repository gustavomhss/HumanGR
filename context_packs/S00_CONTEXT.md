# S00 - project-setup | Context Pack v1.0

---

## PRODUCT REFERENCE

```yaml
product_reference:
  product_id: "HUMANGR"
  product_name: "Human Layer MCP Server"
  wave: "W0-Foundation"
  product_pack: "./PRODUCT_PACK.md"
  sprint_index: "./SPRINT_INDEX.yaml"
```

---

## RELOAD ANCHOR

```yaml
sprint:
  id: S00
  name: project-setup
  title: "Project Setup & Architecture"
  wave: W0-Foundation
  priority: P0-CRITICAL
  type: setup

objective: "Criar estrutura do projeto, configurações base, e CI/CD"

source_block: "../MASTER_REQUIREMENTS_MAP.md#PARTE-1-FUNDAÇÃO"

dependencies: []

deliverables:
  # Root files
  - pyproject.toml
  - README.md
  - LICENSE
  - .gitignore
  - .env.example
  - human-layer.yaml.example

  # Source structure
  - src/hl_mcp/__init__.py
  - src/hl_mcp/py.typed

  # CI/CD
  - .github/workflows/ci.yml
  - .github/workflows/release.yml
  - .github/ISSUE_TEMPLATE/bug_report.yml
  - .github/ISSUE_TEMPLATE/feature_request.yml
  - .github/PULL_REQUEST_TEMPLATE.md
```

---

## ESTRUTURA DO PROJETO

```
human-layer/
├── pyproject.toml           # Package config (PEP 517)
├── README.md                # Basic README (será expandido em S23)
├── LICENSE                  # Apache 2.0
├── .gitignore
├── .env.example             # Template de environment vars
├── human-layer.yaml.example # Configuração exemplo
│
├── src/
│   └── hl_mcp/
│       ├── __init__.py      # Public API exports
│       ├── py.typed         # PEP 561 marker
│       ├── models/          # (S01-S02)
│       ├── llm/             # (S03-S04)
│       ├── layers/          # (S05-S12)
│       ├── core/            # (S13-S14)
│       ├── server/          # (S15-S17)
│       └── cli/             # (S18-S19)
│
├── tests/
│   └── __init__.py
│
├── docs/                    # (S20-S22)
│
└── .github/
    ├── workflows/
    │   ├── ci.yml           # Tests + lint on PR
    │   └── release.yml      # PyPI publish on tag
    ├── ISSUE_TEMPLATE/
    │   ├── bug_report.yml
    │   └── feature_request.yml
    └── PULL_REQUEST_TEMPLATE.md
```

---

## INTENT MANIFEST

```yaml
INTENT_MANIFEST:
  RF:
    - RF-001: "Projeto instalável via pip install -e ."
    - RF-002: "Import básico funciona: import hl_mcp"
    - RF-003: "CI roda tests e linting em PRs"
    - RF-004: "Release workflow publica no PyPI em tags"
    - RF-005: "Estrutura de pastas segue arquitetura definida"

  INV:
    - INV-001: "Python >= 3.9 required"
    - INV-002: "Type hints obrigatórios (py.typed)"
    - INV-003: "Apache 2.0 license"
    - INV-004: "Sem secrets no repositório (.env.example apenas)"

  EDGE:
    - EDGE-001: "pip install falha sem dependências"
    - EDGE-002: "Import circular entre módulos"
```

---

## GATES

```yaml
gates:
  G0_FILES_EXIST:
    description: "Arquivos essenciais existem"
    validation: |
      ls pyproject.toml
      ls src/hl_mcp/__init__.py
      ls .github/workflows/ci.yml

  G1_INSTALL_WORKS:
    description: "Instalação funciona"
    validation: |
      pip install -e .
      python -c "import hl_mcp; print(hl_mcp.__version__)"

  G2_CI_SYNTAX:
    description: "CI YAML é válido"
    validation: |
      python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"

  G3_TESTS_STRUCTURE:
    description: "Estrutura de tests existe"
    validation: |
      ls tests/__init__.py
```

---

## IMPLEMENTATION SPECS

### pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "human-layer"
version = "0.1.0"
description = "7 Layers of Human Judgment for AI Agent Validation"
readme = "README.md"
license = "Apache-2.0"
requires-python = ">=3.9"
authors = [
    { name = "HumanGR", email = "hello@humangr.ai" }
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Software Development :: Quality Assurance",
    "Topic :: Software Development :: Testing",
]
keywords = ["ai", "validation", "mcp", "llm", "testing", "human-layer"]

dependencies = [
    "pydantic>=2.0",
    "httpx>=0.25",
    "typer>=0.9",
    "rich>=13.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
anthropic = ["anthropic>=0.18"]
openai = ["openai>=1.0"]
all = ["human-layer[anthropic,openai]"]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "pytest-asyncio>=0.21",
    "ruff>=0.1",
    "mypy>=1.0",
    "pre-commit>=3.0",
]

[project.scripts]
human-layer = "hl_mcp.cli:main"

[project.urls]
Homepage = "https://humangr.ai"
Documentation = "https://docs.humangr.ai"
Repository = "https://github.com/humangr/human-layer"
Issues = "https://github.com/humangr/human-layer/issues"

[tool.hatch.build.targets.wheel]
packages = ["src/hl_mcp"]

[tool.ruff]
target-version = "py39"
line-length = 100
select = ["E", "F", "I", "N", "W", "UP", "B", "C4", "SIM"]

[tool.mypy]
python_version = "3.9"
strict = true
warn_return_any = true
warn_unused_ignores = true

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

### src/hl_mcp/__init__.py

```python
"""Human Layer - 7 Layers of Human Judgment for AI Agent Validation."""

__version__ = "0.1.0"
__author__ = "HumanGR"

# Public API will be exported here as modules are implemented
# from .models import Finding, LayerResult, ValidationReport  # S02
# from .core import HumanLayerRunner  # S14
# from .server import MCPServer  # S15

__all__ = [
    "__version__",
    "__author__",
]
```

### .github/workflows/ci.yml

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

      - name: Lint with ruff
        run: ruff check .

      - name: Type check with mypy
        run: mypy src/

      - name: Test with pytest
        run: pytest --cov=src --cov-report=xml -v

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        if: matrix.python-version == '3.11'
        with:
          file: ./coverage.xml
```

---

## DECISION TREE

```
START S00
│
├─> Criar pyproject.toml
│   ├─> Definir dependencies base
│   ├─> Configurar build system (hatch)
│   └─> Adicionar scripts entry point
│
├─> Criar src/hl_mcp/__init__.py
│   └─> Exportar __version__
│
├─> Criar estrutura básica
│   ├─> README.md (básico)
│   ├─> LICENSE (Apache 2.0)
│   ├─> .gitignore
│   └─> .env.example
│
├─> Configurar CI/CD
│   ├─> ci.yml (tests + lint)
│   └─> release.yml (PyPI)
│
├─> Criar templates GitHub
│   ├─> bug_report.yml
│   ├─> feature_request.yml
│   └─> PULL_REQUEST_TEMPLATE.md
│
└─> VALIDAR GATES
    ├─> G0: Arquivos existem
    ├─> G1: pip install funciona
    ├─> G2: CI YAML válido
    └─> G3: Estrutura tests existe
```

---

## REFERÊNCIA

Para detalhes completos, consulte:
- `../MASTER_REQUIREMENTS_MAP.md` - PARTE 1: Fundação
- `../MASTER_REQUIREMENTS_MAP.md` - PARTE 5: Arquitetura Técnica
- `../SPRINT_PLAN.md` - Sprint S00 details
