# HumanGR Pipeline

Pipeline autônomo para construir o **Human Layer MCP Server**.

## Separação Absoluta

Este pipeline é **COMPLETAMENTE ISOLADO** do pipeline Veritas/Brains:

| Aspecto | HumanGR | Veritas |
|---------|---------|---------|
| Product ID | `HUMANGR` | `VERITAS` |
| Qdrant Prefix | `humangr_*` | `pipeline_*` |
| Context Packs | `HL-MCP/context_packs/` | `brains/docs/veritas_library/` |
| Target Repo | `HL-MCP/target/human-layer/` | `brains/src/veritas/` |

**Nunca há cruzamento de dados entre os dois pipelines.**

## Quick Start

```bash
# 1. Entrar no diretório
cd /Users/gustavoschneiter/Documents/HL-MCP

# 2. Verificar status
python -m pipeline.cli status

# 3. Listar sprints disponíveis
python -m pipeline.cli list

# 4. Ver detalhes de uma sprint
python -m pipeline.cli info S00

# 5. Checar deliverables
python -m pipeline.cli check S00

# 6. Preview de execução
python -m pipeline.cli start --start S00 --end S05
```

## Estrutura

```
HL-MCP/
├── context_packs/           # 45 context packs (S00-S40 + extras)
│   ├── S00_CONTEXT.md
│   ├── ...
│   └── S40_CONTEXT.md
│
├── pipeline/                # Pipeline fork (isolado do brains)
│   ├── config.py           # Configurações HumanGR
│   ├── pack_loader.py      # Carrega context packs
│   ├── state.py            # Estado do pipeline
│   ├── qdrant_client.py    # Busca semântica
│   └── cli.py              # Interface CLI
│
├── scripts/
│   └── upload_to_qdrant.py # Upload para Qdrant
│
├── target/                  # Onde o código será gerado
│   └── human-layer/
│       ├── src/hl_mcp/
│       └── tests/
│
└── out/                     # Logs e reports
```

## Sprints

| Wave | Sprints | Objetivo |
|------|---------|----------|
| W0-Foundation | S00-S02 | Setup projeto, modelos base |
| W1-CoreEngine | S03-S14 | 7 Layers, Consensus, Veto |
| W2-OSSRelease | S15-S24 | MCP Server, CLI, Docs |
| W3-CloudMVP | S25-S35 | Cloud, Dashboard, Billing |
| W4-Growth | S36-S40 | Perspectives, CI/CD, Analytics |

## Qdrant

O pipeline usa Qdrant para busca semântica de context packs.

```bash
# Verificar conexão
python -c "from pipeline.qdrant_client import get_qdrant_client; print(get_qdrant_client().health_check())"

# Buscar context packs
python -c "
from pipeline.qdrant_client import get_qdrant_client
client = get_qdrant_client()
for r in client.search('security validation', limit=3):
    print(f'{r.sprint_id}: {r.title}')
"
```

## Status

- [x] Context Packs (41 sprints)
- [x] Pipeline Core (config, loader, state, qdrant)
- [x] CLI funcional
- [x] Qdrant com 45 documentos indexados
- [ ] Orchestrator (execução de sprints)
- [ ] Daemon (integração Claude CLI)
- [ ] Gate Runner (validação de gates)
