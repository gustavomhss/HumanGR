# Human Layer

> 7 Layers of Human Judgment for AI Agent Validation

Human Layer is an MCP server that validates AI agent actions through 7 specialized review layers, ensuring safety before execution.

## Quick Start

```bash
pip install human-layer
human-layer init
human-layer validate "Delete all user data from production"
# DECISION: REJECTED (HL-4 Security STRONG veto)
```

## Features

- **7 Human Layers**: UX, Functionality, Edge Cases, Security, Performance, Compliance, Final Review
- **Triple Redundancy**: 3 runs per layer, 2/3 consensus required
- **Veto Powers**: WEAK, MEDIUM, STRONG (STRONG blocks everything)
- **BYOK**: Bring Your Own Key - use your existing LLM subscription
- **MCP Native**: Works with Claude Desktop, Cline, Continue.dev

## Status

🚧 **Under Development** - This package is being built by the HumanGR Pipeline.

## License

Apache 2.0
