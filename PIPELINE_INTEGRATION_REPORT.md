# PIPELINE MULTI-PROJECT INTEGRATION REPORT

> **Data**: 2026-02-01
> **Objetivo**: Habilitar pipeline para trabalhar com HumanGR SEM perder Veritas
> **Requisito Crítico**: SEPARAÇÃO ABSOLUTA - nunca misturar projetos

---

## PRINCÍPIO FUNDAMENTAL

```
╔══════════════════════════════════════════════════════════════════════════╗
║                    SEPARAÇÃO ABSOLUTA - INVIOLÁVEL                       ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  O pipeline deve suportar MÚLTIPLOS PROJETOS simultaneamente:            ║
║                                                                          ║
║  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                  ║
║  │   VERITAS   │    │  HUMANGR    │    │  FOREKAST   │                  ║
║  │  (existente)│    │   (novo)    │    │  (futuro)   │                  ║
║  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘                  ║
║         │                  │                  │                          ║
║         ▼                  ▼                  ▼                          ║
║  ┌────────────────────────────────────────────────────────────┐         ║
║  │              PIPELINE ORCHESTRATOR (ÚNICO)                  │         ║
║  │                                                             │         ║
║  │  project_id → determina TUDO:                              │         ║
║  │  - Qual Qdrant collection usar                             │         ║
║  │  - Qual diretório de context packs                         │         ║
║  │  - Qual repositório alvo                                   │         ║
║  │  - Qual formato de sprint (S00 vs HR00 vs FK00)           │         ║
║  └────────────────────────────────────────────────────────────┘         ║
║                                                                          ║
║  REGRA: Um run do pipeline = UM projeto. Nunca dois ao mesmo tempo.     ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## 1. REFERÊNCIAS HARDCODED ENCONTRADAS

### 1.1 Paths para veritas_library

| Arquivo | Linha | Código Hardcoded | Impacto |
|---------|-------|------------------|---------|
| `src/pipeline_autonomo/pack_discovery.py` | 56-71 | `_VERITAS_LIBRARY = _PROJECT_ROOT / "docs" / "veritas_library"` | CRÍTICO |
| `src/pipeline_autonomo/temporal_integration.py` | 88-159 | `context_path = Path(f"docs/veritas_library/context_packs/{sprint_id}_CONTEXT.md")` | CRÍTICO |
| `src/pipeline_v2/spec_kit_loader.py` | * | Paths para veritas specs | MÉDIO |
| `src/pipeline_autonomo/schema_gate.py` | * | Schema validation paths | MÉDIO |

### 1.2 Qdrant Collection Prefix

| Arquivo | Linha | Código Hardcoded | Impacto |
|---------|-------|------------------|---------|
| `src/pipeline_autonomo/qdrant_client.py` | 75-82 | `collection_prefix: str = "pipeline_"` | CRÍTICO |

**Problema**: Collections criadas:
- `pipeline_context_packs` (Veritas)
- `pipeline_stack_packs` (Veritas)

**Necessário para HumanGR**:
- `humangr_context_packs` (JÁ EXISTE - criamos hoje)

### 1.3 Sprint Format Defaults

| Arquivo | Linha | Código Hardcoded | Impacto |
|---------|-------|------------------|---------|
| `src/pipeline_v2/cli.py` | 2196-2197 | `default="S00"`, `default="S62"` | MÉDIO |

### 1.4 State sem Project ID

| Arquivo | Problema |
|---------|----------|
| `src/pipeline_v2/langgraph/state.py` | `PipelineState` não tem `project_id` |
| `src/pipeline_v2/langgraph/bridge.py` | State só passa `sprint_id`, não projeto |

---

## 2. INFRAESTRUTURA MULTI-PROJETO JÁ EXISTENTE

O pipeline **já tem framework parcial** para múltiplos projetos:

### 2.1 Pack Discovery (pack_discovery.py:244-287)

```python
# JÁ EXISTE detecção por prefixo:
if sprint_id.startswith("S"):     # → veritas_sprints
if sprint_id.startswith("FK"):    # → forekast_sprints
if sprint_id.startswith("VIS-"):  # → visionary_sprints
```

**Falta adicionar**: `HR` ou `HG` para HumanGR

### 2.2 YAML Bindings

O arquivo `PACK_BINDING_INDEX.yaml` já suporta múltiplos projetos:

```yaml
veritas_sprints:
  S00: { name: "...", feature_packs: [...] }

forekast_sprints:
  FK00: { name: "...", feature_packs: [...] }

# FALTA ADICIONAR:
humangr_sprints:
  HR00: { name: "...", feature_packs: [...] }
```

---

## 3. O QUE PRECISA SER FEITO

### 3.1 Configuração de Projeto (NOVA)

**Criar arquivo**: `configs/pipeline_autonomo/projects.yaml`

```yaml
projects:
  veritas:
    product_id: "VERITAS"
    documentation_root: "docs/veritas_library"
    context_packs_dir: "docs/veritas_library/context_packs"
    qdrant_collection_prefix: "pipeline_"
    sprint_prefix: "S"
    sprint_range: ["S00", "S62"]
    target_repository: "/Users/gustavoschneiter/Documents/brains"

  humangr:
    product_id: "HUMANGR"
    documentation_root: "/Users/gustavoschneiter/Documents/HL-MCP"
    context_packs_dir: "/Users/gustavoschneiter/Documents/HL-MCP/context_packs"
    qdrant_collection_prefix: "humangr_"
    sprint_prefix: "HR"  # ou "S" se quiser manter S00-S40
    sprint_range: ["S00", "S40"]
    target_repository: "TBD"  # DEFINIR REPOSITÓRIO

  forekast:
    product_id: "FOREKAST"
    documentation_root: "docs/forekast_library"
    context_packs_dir: "docs/forekast_library/context_packs"
    qdrant_collection_prefix: "forekast_"
    sprint_prefix: "FK"
    sprint_range: ["FK00", "FK20"]
    target_repository: "TBD"
```

### 3.2 Environment Variables (Adicionar)

```bash
# .env
PROJECT_ID=veritas  # ou "humangr", "forekast"
# OU passar via CLI: --project humangr
```

### 3.3 Modificações no Código

#### Prioridade 1: Project Config Loader (NOVO)

```python
# src/pipeline_autonomo/project_config.py

@dataclass
class ProjectConfig:
    project_id: str
    product_id: str
    documentation_root: Path
    context_packs_dir: Path
    qdrant_collection_prefix: str
    sprint_prefix: str
    sprint_range: tuple[str, str]
    target_repository: Path

def load_project_config(project_id: str) -> ProjectConfig:
    """Load project configuration from YAML."""
    ...

def get_current_project() -> ProjectConfig:
    """Get project from env var or raise error."""
    project_id = os.getenv("PROJECT_ID")
    if not project_id:
        raise ValueError("PROJECT_ID não definido!")
    return load_project_config(project_id)
```

#### Prioridade 2: Refatorar pack_discovery.py

```python
# ANTES (hardcoded):
_VERITAS_LIBRARY = _PROJECT_ROOT / "docs" / "veritas_library"

# DEPOIS (dinâmico):
def get_library_path(project: ProjectConfig) -> Path:
    return Path(project.documentation_root)
```

#### Prioridade 3: Refatorar temporal_integration.py

```python
# ANTES:
context_path = Path(f"docs/veritas_library/context_packs/{sprint_id}_CONTEXT.md")

# DEPOIS:
def load_context_pack(sprint_id: str, project: ProjectConfig) -> dict:
    context_path = project.context_packs_dir / f"{sprint_id}_CONTEXT.md"
```

#### Prioridade 4: Refatorar qdrant_client.py

```python
# ANTES:
collection_prefix: str = "pipeline_"

# DEPOIS:
collection_prefix: str = field(
    default_factory=lambda: get_current_project().qdrant_collection_prefix
)
```

#### Prioridade 5: Adicionar project_id ao State

```python
# src/pipeline_v2/langgraph/state.py
class PipelineState(TypedDict):
    project_id: str  # NOVO - OBRIGATÓRIO
    sprint_id: str
    run_id: str
    ...
```

#### Prioridade 6: CLI --project flag

```bash
# Uso futuro:
python -m pipeline_v2.cli start --project humangr --start S00 --end S40
python -m pipeline_v2.cli start --project veritas --start S00 --end S62
```

---

## 4. FORMATO DAS SPRINTS HUMANGR

### 4.1 Comparação de Formatos

| Aspecto | Veritas | HumanGR | Compatível? |
|---------|---------|---------|-------------|
| Prefixo Sprint | S | S (ou HR) | ✅ Sim |
| Range | S00-S62 | S00-S40 | ✅ Sim |
| Arquivo | `S00_CONTEXT.md` | `S00_CONTEXT.md` | ✅ Sim |
| Estrutura | RELOAD ANCHOR, SPECS, GATES | RELOAD ANCHOR, SPECS, GATES | ✅ Sim |

**Conclusão**: O formato já está compatível. Só precisa do roteamento correto.

### 4.2 Verificação do Formato HumanGR

```bash
# Verificar estrutura de um context pack HumanGR
head -50 /Users/gustavoschneiter/Documents/HL-MCP/context_packs/S00_CONTEXT.md
```

Campos obrigatórios no context pack:
- [ ] `sprint_id`
- [ ] `title`
- [ ] `wave`
- [ ] `objective`
- [ ] `deliverables`
- [ ] `INTENT MANIFEST` (RF, INV, EDGE)
- [ ] `GATES`

---

## 5. REPOSITÓRIO ALVO (A DEFINIR)

### Opções:

1. **Criar novo repositório**: `github.com/humangr/human-layer`
   - Pros: Separação total
   - Cons: Precisa criar e configurar

2. **Diretório dentro de brains**: `src/humangr/`
   - Pros: Rápido
   - Cons: Mistura códigos

3. **Repositório separado local**: `/Users/gustavoschneiter/Documents/human-layer/`
   - Pros: Separação, pode virar GitHub depois
   - Cons: Precisa criar estrutura

**Recomendação**: Opção 3 - criar repositório local separado

```bash
# Estrutura sugerida:
/Users/gustavoschneiter/Documents/human-layer/
├── src/
│   └── human_layer/
│       ├── __init__.py
│       ├── core/
│       │   ├── models.py      # S02
│       │   ├── layers/        # S06-S12
│       │   └── consensus.py   # S13
│       ├── mcp/
│       │   ├── server.py      # S15
│       │   ├── tools.py       # S16
│       │   └── resources.py   # S17
│       ├── cli/
│       │   └── main.py        # S18-S19
│       └── cloud/
│           └── ...            # S25-S35
├── tests/
├── docs/
├── pyproject.toml
└── README.md
```

---

## 6. CHECKLIST DE SEPARAÇÃO

```
╔══════════════════════════════════════════════════════════════════════════╗
║                    GARANTIAS DE SEPARAÇÃO                                ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  [ ] Qdrant collections isoladas:                                        ║
║      - pipeline_* → Veritas APENAS                                       ║
║      - humangr_* → HumanGR APENAS                                        ║
║                                                                          ║
║  [ ] Context packs em diretórios separados:                              ║
║      - docs/veritas_library/ → Veritas                                   ║
║      - /Documents/HL-MCP/context_packs/ → HumanGR                        ║
║                                                                          ║
║  [ ] Repositórios de código separados:                                   ║
║      - /Documents/brains/src/veritas/ → Veritas                          ║
║      - /Documents/human-layer/src/ → HumanGR                             ║
║                                                                          ║
║  [ ] State sempre com project_id:                                        ║
║      - Toda operação sabe qual projeto está rodando                      ║
║      - Nunca assume default                                              ║
║                                                                          ║
║  [ ] Logs e outputs separados:                                           ║
║      - out/veritas/ ou out/pipeline/                                     ║
║      - out/humangr/                                                      ║
║                                                                          ║
║  [ ] NUNCA compartilhar:                                                 ║
║      - Collections Qdrant                                                ║
║      - Diretórios de código                                              ║
║      - State entre runs                                                  ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## 7. PRÓXIMOS PASSOS RECOMENDADOS

### Fase 1: Preparação (Esta Semana)

1. [ ] Definir repositório alvo para HumanGR
2. [ ] Criar `projects.yaml` com configurações
3. [ ] Criar `project_config.py` loader

### Fase 2: Refatoração Core (Semana 2)

4. [ ] Refatorar `pack_discovery.py` para aceitar project
5. [ ] Refatorar `temporal_integration.py`
6. [ ] Refatorar `qdrant_client.py`

### Fase 3: State & CLI (Semana 3)

7. [ ] Adicionar `project_id` ao PipelineState
8. [ ] Adicionar `--project` flag ao CLI
9. [ ] Atualizar bridge.py para propagar project

### Fase 4: Teste E2E (Semana 4)

10. [ ] Rodar sprint S00 do HumanGR
11. [ ] Verificar que Veritas continua funcionando
12. [ ] Validar separação total

---

## 8. QUESTÕES A RESOLVER

1. **Repositório HumanGR**: Qual será o path?
2. **Sprint Prefix**: Manter `S00` ou usar `HR00`?
3. **Cockpit**: O cockpit precisa de selector de projeto?
4. **Agents/Cerebros**: Usar os mesmos ou criar específicos?

---

## RESUMO

O pipeline tem infraestrutura parcial para multi-projeto, mas está hardcoded para Veritas. As mudanças necessárias são:

| Componente | Esforço | Risco |
|------------|---------|-------|
| project_config.py (novo) | Baixo | Baixo |
| pack_discovery.py | Médio | Médio |
| temporal_integration.py | Médio | Médio |
| qdrant_client.py | Baixo | Baixo |
| state.py + bridge.py | Alto | Alto |
| CLI | Baixo | Baixo |

**Total estimado**: 3-4 semanas para implementação completa com testes.

**Separação já garantida no Qdrant**: Collection `humangr_context_packs` já existe com 45 documentos, completamente isolada de `pipeline_*`.
