# PRODUCT_PACK - HumanGR Human Layer MCP Server

> **Empresa**: HumanGR
> **Produto**: Human Layer MCP Server
> **Versão**: 1.0.0
> **Data**: 2026-02-01

---

## VISÃO DO PRODUTO

```yaml
product:
  id: HUMANGR
  name: "Human Layer MCP Server"
  tagline: "Guardrails Humanos para AI Agents"

vision: |
  Permitir que AI agents operem com segurança através de
  validação humana automatizada em 7 camadas, garantindo
  que ações críticas passem por escrutínio antes de execução.

value_proposition:
  - 7 Human Layers com poder de veto (WEAK/MEDIUM/STRONG)
  - 6 Perspectives simulando diferentes usuários
  - Triple Redundancy (3 runs, consenso 2/3)
  - Integração MCP (Claude Desktop, GPT, Gemini)
  - Open Core (100% funcional OSS + Cloud premium)
```

---

## ARQUITETURA LEGO

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      HUMAN LAYER MCP - LEGO ARCHITECTURE                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  CAMADA 1: MCP SERVER                                                       │
│  ├── Tools (validate, test, report)                                         │
│  ├── Resources (layers, perspectives, stats)                                │
│  └── Prompts (templates)                                                    │
│                                                                             │
│  CAMADA 2: CORE ENGINE                                                      │
│  ├── HumanLayerRunner (executa layers)                                      │
│  ├── TripleRedundancy (3x com consenso)                                     │
│  ├── LayerOrchestrator (coordena 7 layers)                                  │
│  └── VetoGate (aplica vetos)                                                │
│                                                                             │
│  CAMADA 3: 7 HUMAN LAYERS                                                   │
│  ├── HL-1 UI/UX Review                                                      │
│  ├── HL-2 Security Scan                                                     │
│  ├── HL-3 Edge Cases                                                        │
│  ├── HL-4 Accessibility                                                     │
│  ├── HL-5 Performance                                                       │
│  ├── HL-6 Integration                                                       │
│  └── HL-7 Final Human Check                                                 │
│                                                                             │
│  CAMADA 4: 6 PERSPECTIVES                                                   │
│  ├── tired_user (cansado, impaciente)                                       │
│  ├── malicious_insider (tentando abusar)                                    │
│  ├── confused_newbie (perdido, novato)                                      │
│  ├── power_user (quer atalhos, eficiência)                                  │
│  ├── auditor (compliance, logs)                                             │
│  └── 3am_operator (sonolento, emergência)                                   │
│                                                                             │
│  CAMADA 5: COGNITIVE MODULES                                                │
│  ├── BudgetManager (tokens/recursos)                                        │
│  ├── TrustScorer (confiança em agents)                                      │
│  ├── TriageEngine (priorização inteligente)                                 │
│  └── FeedbackLoop (aprendizado)                                             │
│                                                                             │
│  CAMADA 6: BROWSER AUTOMATION                                               │
│  ├── BrowserDriver (Playwright)                                             │
│  ├── ScreenshotManager                                                      │
│  ├── VideoRecorder                                                          │
│  └── AccessibilityChecker                                                   │
│                                                                             │
│  CAMADA 7: DATA MODELS                                                      │
│  ├── Finding, LayerResult                                                   │
│  ├── Journey, JourneyResult                                                 │
│  ├── HumanLayerReport                                                       │
│  └── Enums (VetoLevel, Severity, etc)                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## WAVES DE EXECUÇÃO

| Wave | Nome | Sprints | Objetivo |
|------|------|---------|----------|
| W0 | Reference | S00-S01 | Documentar e arquitetar |
| W1 | Foundation | S02-S05 | Models, Resources, LLM |
| W2 | Core | S06-S08 | Browser, Engine, Layers |
| W3 | Intelligence | S09-S10 | Perspectives, Cognitive |
| W4 | Integration | S11-S12 | MCP Server, Cloud |

---

## SPRINTS

| Sprint | Nome | Wave | Prioridade | Módulos |
|--------|------|------|------------|---------|
| S00 | Dissection | W0 | P0-CRITICAL | Referência |
| S01 | Architecture | W0 | P0-CRITICAL | 2 |
| S02 | Resources & Builder | W1 | P1-HIGH | 4 |
| S03 | Data Models Base | W1 | P0-CRITICAL | 5 |
| S04 | Models Complete | W1 | P0-CRITICAL | 4 |
| S05 | LLM Integration | W1 | P0-CRITICAL | 5 |
| S06 | Browser Automation | W2 | P1-HIGH | 6 |
| S07 | Core Engine | W2 | P0-CRITICAL | 5 |
| S08 | 7 Human Layers | W2 | P0-CRITICAL | 8 |
| S09 | 6 Perspectives | W3 | P1-HIGH | 8 |
| S10 | Cognitive Modules | W3 | P1-HIGH | 5 |
| S11 | MCP Server | W4 | P0-CRITICAL | 4 |
| S12 | Cloud & Monetization | W4 | P2-MEDIUM | 8 |

**Total: 13 sprints, ~67 módulos**

---

## MODELO DE NEGÓCIO

```yaml
business_model: "Open Core"

tiers:
  free:
    name: "OSS Self-Hosted"
    price: "$0/forever"
    features:
      - 100% funcionalidade
      - Self-hosted
      - User configura tudo
      - Community support
    llm: "User usa próprio plano (Claude Max, GPT Plus, etc)"

  starter:
    name: "Cloud Starter"
    price: "$12/mês"
    features:
      - Dashboard básico
      - 30 dias histórico
      - 1 usuário
      - Email support
    llm: "BYOK ou pass-through (+15%)"

  pro:
    name: "Cloud Pro"
    price: "$49/mês"
    features:
      - Dashboard completo
      - 90 dias histórico
      - 5 usuários
      - CI/CD integration
      - Priority support
    llm: "BYOK ou pass-through (+15%)"

  business:
    name: "Cloud Business"
    price: "$249/mês"
    features:
      - Tudo do Pro
      - Histórico ilimitado
      - Usuários ilimitados
      - SSO/SAML
      - SLA 99.9%
      - Dedicated support
    llm: "BYOK ou pass-through (+10%)"
```

---

## INVARIANTES DO PRODUTO

```yaml
INV-001: "7 Human Layers sempre executam (não pular)"
INV-002: "Triple Redundancy = 3 runs, consenso 2/3"
INV-003: "Veto STRONG bloqueia tudo"
INV-004: "User usa próprio LLM (não forçar API)"
INV-005: "OSS = 100% funcional, sempre"
INV-006: "Token acabou = warn user, não bloquear"
INV-007: "Cada módulo max 200-300 linhas"
INV-008: "Tudo indexado em LEGO_INDEX.yaml"
```

---

## DEPENDÊNCIAS TÉCNICAS

```yaml
core:
  - python: ">=3.11"
  - pydantic: ">=2.0"
  - mcp: "latest"

llm:
  - anthropic: "Claude API"
  - openai: "OpenAI API"
  - ollama: "Local models"

browser:
  - playwright: "Browser automation"

optional_cloud:
  - fastapi: "Dashboard API"
  - postgresql: "History storage"
  - redis: "Caching"
  - stripe: "Billing"
```

---

## LINKS

- **Sprints**: `./S{XX}_CONTEXT.md`
- **Índice**: `./SPRINT_INDEX.yaml`
- **Roadmap Original**: `../ROADMAP_BLOCK_*.md`
