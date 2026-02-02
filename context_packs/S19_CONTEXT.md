# S19 - cli-commands | Context Pack v1.0

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
  id: S19
  name: cli-commands
  title: "CLI Commands"
  wave: W2-OSSRelease
  priority: P0-CRITICAL
  type: implementation

objective: "Comandos CLI adicionais (init, validate, serve)"

source_block: "../MASTER_REQUIREMENTS_MAP.md#PARTE-7-CLI"

dependencies:
  - S18  # CLI Base

deliverables:
  - src/hl_mcp/cli/commands/__init__.py
  - src/hl_mcp/cli/commands/init.py
  - src/hl_mcp/cli/commands/validate.py
  - src/hl_mcp/cli/commands/serve.py
  - tests/test_cli/test_commands.py
```

---

## COMMANDS DETAIL

```yaml
commands:
  init:
    description: "Initialize Human Layer in a project"
    options:
      --config: "Config file path"
      --force: "Overwrite existing"
      --template: "Config template"
    creates:
      - human-layer.yaml
      - .env.example

  validate:
    description: "Run validation on an action or file"
    options:
      --file: "File to validate"
      --diff: "Git diff to validate"
      --agent: "Agent ID"
      --layers: "Specific layers"
      --format: "Output format"
    outputs:
      - decision
      - findings
      - report

  serve:
    description: "Start MCP server"
    options:
      --transport: "stdio or http"
      --host: "HTTP host"
      --port: "HTTP port"
      --config: "Config file"
```

---

## IMPLEMENTATION SPEC

### Init Command (commands/init.py)

```python
"""Init command - Initialize Human Layer configuration."""
import typer
from pathlib import Path
from rich.console import Console

app = typer.Typer()
console = Console()

DEFAULT_CONFIG = '''# Human Layer Configuration
# https://github.com/humangr/human-layer

llm:
  provider: claude  # claude, openai, gemini, ollama
  # api_key: Set via ANTHROPIC_API_KEY env var

layers:
  HL1: {enabled: true}  # UX & Usability (WEAK veto)
  HL2: {enabled: true}  # Functionality (MEDIUM veto)
  HL3: {enabled: true}  # Edge Cases (MEDIUM veto)
  HL4: {enabled: true}  # Security (STRONG veto)
  HL5: {enabled: true}  # Performance (MEDIUM veto)
  HL6: {enabled: true}  # Compliance (STRONG veto)
  HL7: {enabled: true}  # Final Review (STRONG veto)

triple_redundancy: true
log_level: INFO
'''

ENV_EXAMPLE = '''# Human Layer Environment Variables
# Copy to .env and fill in your keys

# LLM API Keys (BYOK - use your own subscription)
ANTHROPIC_API_KEY=  # For Claude
OPENAI_API_KEY=     # For OpenAI
GOOGLE_API_KEY=     # For Gemini
# Ollama runs locally, no key needed

# Optional
HL_LOG_LEVEL=INFO
'''


@app.command()
def init(
    config_path: str = typer.Option("human-layer.yaml", "--config", "-c"),
    force: bool = typer.Option(False, "--force", "-f"),
    with_env: bool = typer.Option(True, "--env/--no-env"),
):
    """Initialize Human Layer configuration files."""
    config_file = Path(config_path)

    if config_file.exists() and not force:
        console.print(f"[yellow]Config exists: {config_path}[/yellow]")
        console.print("Use --force to overwrite")
        raise typer.Exit(1)

    config_file.write_text(DEFAULT_CONFIG)
    console.print(f"[green]Created: {config_path}[/green]")

    if with_env:
        env_file = Path(".env.example")
        if not env_file.exists() or force:
            env_file.write_text(ENV_EXAMPLE)
            console.print("[green]Created: .env.example[/green]")

    console.print()
    console.print("[bold]Next steps:[/bold]")
    console.print("1. Copy .env.example to .env")
    console.print("2. Add your LLM API key")
    console.print("3. Run: human-layer validate 'your action'")
```

### Validate Command (commands/validate.py)

```python
"""Validate command - Run Human Layer validation."""
import typer
from pathlib import Path
from rich.console import Console
from rich.table import Table
from typing import Optional, List
import json

app = typer.Typer()
console = Console()


@app.command()
def validate(
    description: str = typer.Argument(None, help="Action description"),
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="File to validate"),
    diff: bool = typer.Option(False, "--diff", "-d", help="Validate git diff"),
    agent: str = typer.Option("cli-user", "--agent", "-a"),
    layers: Optional[List[str]] = typer.Option(None, "--layer", "-l"),
    output_format: str = typer.Option("rich", "--format", "-o"),
):
    """Run Human Layer validation."""
    from ...core import DecisionEngine
    from ...layers.base import ActionContext

    # Determine what to validate
    if file:
        code_content = file.read_text()
        action_desc = f"Validate file: {file}"
    elif diff:
        import subprocess
        result = subprocess.run(["git", "diff"], capture_output=True, text=True)
        code_content = result.stdout
        action_desc = "Validate git diff"
    elif description:
        code_content = None
        action_desc = description
    else:
        console.print("[red]Provide description, --file, or --diff[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Validating:[/bold] {action_desc}")
    console.print()

    # Run validation (placeholder)
    results = _run_validation(action_desc, code_content, layers)

    # Output
    if output_format == "json":
        console.print(json.dumps(results, indent=2))
    else:
        _display_rich_results(results)


def _run_validation(description: str, code: Optional[str], layers: Optional[List[str]]) -> dict:
    """Run validation and return results."""
    # Placeholder - would run actual validation
    return {
        "decision": "approved",
        "veto_level": "NONE",
        "layer_results": [
            {"layer": "HL1", "status": "PASS", "veto": "NONE", "findings": []},
            {"layer": "HL2", "status": "PASS", "veto": "NONE", "findings": []},
            {"layer": "HL3", "status": "PASS", "veto": "NONE", "findings": []},
            {"layer": "HL4", "status": "PASS", "veto": "NONE", "findings": []},
            {"layer": "HL5", "status": "PASS", "veto": "NONE", "findings": []},
            {"layer": "HL6", "status": "PASS", "veto": "NONE", "findings": []},
            {"layer": "HL7", "status": "PASS", "veto": "NONE", "findings": []},
        ],
        "total_findings": 0,
    }


def _display_rich_results(results: dict):
    """Display results with rich formatting."""
    table = Table(title="Validation Results")
    table.add_column("Layer", style="cyan")
    table.add_column("Status")
    table.add_column("Veto")
    table.add_column("Findings")

    for lr in results["layer_results"]:
        status_style = "green" if lr["status"] == "PASS" else "red"
        table.add_row(
            lr["layer"],
            f"[{status_style}]{lr['status']}[/{status_style}]",
            lr["veto"],
            str(len(lr["findings"]))
        )

    console.print(table)
    console.print()

    decision = results["decision"]
    if decision == "approved":
        console.print("[bold green]DECISION: APPROVED[/bold green]")
    elif decision == "rejected":
        console.print("[bold red]DECISION: REJECTED[/bold red]")
    else:
        console.print("[bold yellow]DECISION: NEEDS REVIEW[/bold yellow]")
```

### Serve Command (commands/serve.py)

```python
"""Serve command - Start MCP server."""
import typer
from pathlib import Path
from rich.console import Console
from typing import Optional

app = typer.Typer()
console = Console()


@app.command()
def serve(
    transport: str = typer.Option("stdio", "--transport", "-t"),
    host: str = typer.Option("127.0.0.1", "--host", "-h"),
    port: int = typer.Option(8080, "--port", "-p"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
):
    """Start Human Layer MCP server."""
    import asyncio
    from ...server import create_server, ServerConfig

    console.print("[bold]Human Layer MCP Server[/bold]")
    console.print(f"Transport: {transport}")

    server_config = None
    if config and config.exists():
        server_config = ServerConfig.from_yaml(str(config))
        console.print(f"Config: {config}")

    server = create_server(server_config)

    if transport == "stdio":
        console.print("Mode: stdio (for Claude Desktop, Cline, etc.)")
        console.print("Server running... (Press Ctrl+C to stop)")
        try:
            asyncio.run(server.run_stdio())
        except KeyboardInterrupt:
            console.print("\n[yellow]Server stopped[/yellow]")

    elif transport == "http":
        console.print(f"Mode: HTTP/SSE")
        console.print(f"URL: http://{host}:{port}")
        console.print("Server running... (Press Ctrl+C to stop)")
        try:
            asyncio.run(server.run_http(host, port))
        except KeyboardInterrupt:
            console.print("\n[yellow]Server stopped[/yellow]")

    else:
        console.print(f"[red]Unknown transport: {transport}[/red]")
        console.print("Valid options: stdio, http")
        raise typer.Exit(1)
```

---

## INTENT MANIFEST

```yaml
INTENT_MANIFEST:
  RF:
    - RF-001: "init cria human-layer.yaml e .env.example"
    - RF-002: "validate suporta description, file, diff"
    - RF-003: "serve suporta stdio e http"
    - RF-004: "Output em rich ou json"
    - RF-005: "Comandos modulares em commands/"

  INV:
    - INV-001: "init não sobrescreve sem --force"
    - INV-002: "validate requer input (desc, file, ou diff)"
    - INV-003: "serve roda até Ctrl+C"
    - INV-004: "Exit codes corretos (0=ok, 1=erro)"

  EDGE:
    - EDGE-001: "Arquivo não existe em --file → erro"
    - EDGE-002: "git diff vazio → validate anyway"
    - EDGE-003: "Porta ocupada em http → erro"
```

---

## GATES

```yaml
gates:
  G0_FILES_EXIST:
    description: "Arquivos existem"
    validation: |
      ls src/hl_mcp/cli/commands/init.py
      ls src/hl_mcp/cli/commands/validate.py
      ls src/hl_mcp/cli/commands/serve.py

  G1_IMPORTS_WORK:
    description: "Imports funcionam"
    validation: |
      python -c "from hl_mcp.cli.commands import init, validate, serve"

  G2_TESTS_PASS:
    description: "Testes passam"
    validation: "pytest tests/test_cli/test_commands.py -v"
```

---

## REFERÊNCIA

- `./S18_CONTEXT.md` - CLI Base
- `./S20_CONTEXT.md` - Docs Getting Started
