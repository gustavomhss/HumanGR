# MASTER REQUIREMENTS MAP - HumanGR

> **Empresa/Projeto**: HumanGR
> **Produto Principal**: Human Layer (MCP Server)
> **Objetivo**: Mapear TUDO que precisa existir, E2E, sem exceção.
> **Status**: MAPEAMENTO EM PROGRESSO
> **Data**: 2026-02-01

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   HumanGR (Empresa)                                         │
│   └── Human Layer (Produto)                                 │
│       ├── OSS Self-Hosted (gratuito)                        │
│       └── Cloud Managed (pago)                              │
│           ├── Dashboard / Cockpit Visual                    │
│           ├── History Storage                               │
│           ├── Team Features                                 │
│           └── CI/CD Integrations                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## COMO USAR ESTE DOCUMENTO

```
1. CHECKLIST: [ ] = não mapeado, [~] = parcial, [x] = completo
2. CADA ITEM: será expandido em detalhes antes de virar sprint
3. NENHUM ITEM pode ficar sem detalhar antes de implementar
```

---

# PARTE 1: FUNDAÇÃO DO PRODUTO

## 1.1 VISÃO & ESTRATÉGIA

### [x] Vision Statement
> **"Um mundo onde todo código gerado por AI passa por validação humana estruturada antes de afetar usuários reais."**

A visão é que Human Layer se torne o padrão da indústria para validação de outputs de AI, assim como linters e testes automatizados se tornaram padrão para qualidade de código.

### [x] Mission Statement
> **"Capacitar desenvolvedores a confiar em AI sem abrir mão do julgamento humano, através de validação estruturada, redundante e auditável."**

### [x] Value Proposition (por persona)

| Persona | Value Proposition |
|---------|-------------------|
| **Dev Solo** | "Valide código AI em 5 minutos, sem setup complexo. Gratuito para sempre no OSS." |
| **Tech Lead** | "Garanta que sua equipe não ship bugs de AI. Dashboard unificado, histórico completo." |
| **Engineering Manager** | "Reduza risco de AI em produção com métricas claras e audit trail completo." |
| **DevOps** | "Integre validação AI no CI/CD em 10 minutos. GitHub Actions, GitLab, Jenkins." |
| **QA Engineer** | "6 perspectivas automatizadas que você não consegue testar manualmente." |
| **CTO/VP Eng** | "Compliance-ready. SOC2 roadmap. SSO. On-prem option. Enterprise SLA." |

### [x] Problema que resolve

**Problema Central**: AI gera código cada vez melhor, mas ainda comete erros que humanos pegariam facilmente. Revisão manual não escala. Testes automatizados não pegam problemas de UX, segurança sutil, ou edge cases bizarros.

**Dores Específicas**:
1. **Confiança cega em AI** → Bugs em produção que "pareciam certos"
2. **Revisão manual inconsistente** → Depende de quem revisa, quando, humor
3. **Sem audit trail** → "Quem aprovou isso?" sem resposta
4. **Perspectivas limitadas** → Dev não pensa como usuário cansado às 3am
5. **Escala impossível** → 100 PRs/dia com AI, 2 reviewers humanos

### [x] Diferencial competitivo

| Diferencial | Human Layer | Competidores |
|-------------|-------------|--------------|
| **7 Camadas Especializadas** | Cada layer com foco específico (UX, Security, Edge Cases, A11y, Perf, Integration, Final) | Validação genérica única |
| **6 Perspectivas Simuladas** | tired_user, malicious_insider, confused_newbie, power_user, auditor, 3am_operator | Sem simulação de personas |
| **Triple Redundancy** | 3 execuções por layer, consenso 2/3 | Single-shot |
| **Veto Powers** | WEAK/MEDIUM/STRONG com semântica clara | Pass/Fail binário |
| **BYOK (Bring Your Own Key)** | Usuário usa SEU plano LLM (Claude Max, GPT Plus) | Vendor lock-in com API própria |
| **Open Source Core** | Core completo OSS, Apache 2.0 | Closed source ou OSS limitado |
| **MCP Native** | Funciona direto no Claude Desktop, Cursor, etc. | Integração manual |

### [x] Posicionamento de mercado

```
                    High Automation
                         │
                         │
    ┌────────────────────┼────────────────────┐
    │                    │                    │
    │   Linters/SAST     │    Human Layer     │ ← NOSSO QUADRANTE
    │   (rápido, raso)   │  (profundo, smart) │
    │                    │                    │
Low ─┼────────────────────┼────────────────────┼─ High
Trust│                    │                    │  Trust
    │                    │                    │
    │   Nada             │   Code Review      │
    │   (YOLO)           │   (lento, manual)  │
    │                    │                    │
    └────────────────────┼────────────────────┘
                         │
                    Low Automation
```

**Posição**: Alta automação + Alta confiança. Não substituímos code review humano, complementamos com validação estruturada que escala.

### [x] Análise competitiva

| Competidor | Tipo | Força | Fraqueza | Nossa Vantagem |
|------------|------|-------|----------|----------------|
| **Human Layer (YC F24)** | Human-in-the-loop para AI agents | Funding, YC network | Foco em aprovação, não validação profunda | Validação multi-camada, não só aprovação |
| **Codium/Qodo** | AI test generation | Boa geração de testes | Não valida output de AI, só gera testes | Validação de qualquer output, não só testes |
| **Cursor/Copilot** | AI coding assistants | Mainstream adoption | Zero validação built-in | Complementar, não competidor |
| **SonarQube/Snyk** | SAST/DAST | Estabelecido, enterprise | Regras estáticas, não entende contexto | AI-powered, entende intenção |
| **Manual Code Review** | Humanos | Alta confiança | Não escala, inconsistente | Escala infinita, consistente |

### [x] Roadmap de longo prazo

**Ano 1 (2026)**:
- Q1: OSS launch, MCP server funcionando
- Q2: Cloud beta, primeiros paying customers
- Q3: CI/CD integrations completas
- Q4: Team features, 1000 users

**Ano 2 (2027)**:
- Q1: Enterprise features (SSO, SAML)
- Q2: SOC2 Type 1
- Q3: On-premise option
- Q4: Series A (se fizer sentido)

**Ano 3 (2028)**:
- Expansão internacional
- Marketplace de layers customizados
- AI model fine-tuning para validação
- IPO track ou acquisition target

---

## 1.2 MODELO DE NEGÓCIO

### [x] Open Core philosophy detalhada

**Princípio Central**: O core de validação é 100% OSS, para sempre. Monetizamos conveniência, escala e features enterprise.

```
┌─────────────────────────────────────────────────────────────┐
│                        OPEN SOURCE                          │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ • 7 Human Layers (todas)                              │  │
│  │ • 6 Perspectives (todas)                              │  │
│  │ • Triple Redundancy                                   │  │
│  │ • Consensus Engine                                    │  │
│  │ • Veto Gate                                           │  │
│  │ • MCP Server                                          │  │
│  │ • CLI                                                 │  │
│  │ • Local storage                                       │  │
│  │ • All LLM integrations                                │  │
│  └───────────────────────────────────────────────────────┘  │
│                       Apache 2.0 License                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                         CLOUD                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ • Dashboard visual                                    │  │
│  │ • History storage (cloud)                             │  │
│  │ • Team management                                     │  │
│  │ • CI/CD integrations managed                          │  │
│  │ • Webhooks                                            │  │
│  │ • API access                                          │  │
│  │ • SSO/SAML                                            │  │
│  │ • Priority support                                    │  │
│  │ • SLA guarantees                                      │  │
│  └───────────────────────────────────────────────────────┘  │
│                      Proprietary License                    │
└─────────────────────────────────────────────────────────────┘
```

**Regra de Ouro**: Se um dev solo consegue fazer localmente, é OSS. Se precisa de infra/equipe/compliance, é Cloud.

### [x] O que é OSS vs Cloud (linha clara)

| Feature | OSS | Cloud |
|---------|-----|-------|
| 7 Human Layers | ✅ | ✅ |
| 6 Perspectives | ✅ | ✅ |
| Triple Redundancy | ✅ | ✅ |
| MCP Server | ✅ | ✅ |
| CLI | ✅ | ✅ |
| Local history | ✅ | ✅ |
| Cloud history | ❌ | ✅ |
| Dashboard visual | ❌ | ✅ |
| Team management | ❌ | ✅ |
| CI/CD managed | ❌ | ✅ |
| Webhooks | ❌ | ✅ |
| API access | ❌ | ✅ (Business+) |
| SSO/SAML | ❌ | ✅ (Business+) |
| SLA | ❌ | ✅ (Enterprise) |
| Support | Community | Paid tiers |

### [x] Pricing strategy (why these prices)

**Filosofia**: Preços acessíveis para indie devs, escaláveis para enterprise. Baseado em valor entregue, não custo.

| Tier | Preço | Justificativa |
|------|-------|---------------|
| **OSS** | $0 | Adoção máxima, comunidade forte, feedback loop |
| **Free Cloud** | $0 | Funil de conversão, experimentação sem risco |
| **Solo** | $12/mo | Preço de "coffee money", impulse buy para dev solo |
| **Pro** | $39/mo | Preço de ferramenta profissional (similar Postman, Notion Pro) |
| **Team** | $99/mo | Preço de SaaS B2B entry-level, aprovação sem procurement |
| **Business** | $249/mo | Preço sério mas ainda auto-serve, inclui SSO |
| **Enterprise** | Custom | Negociação, SLA, on-prem, compliance docs |

**Comparação de Mercado**:
- GitHub Copilot: $19/mo individual, $39/mo business
- Postman: $14/mo, $29/mo team
- Linear: $8/mo, $12/mo
- Notion: $10/mo, $18/mo

### [x] Unit economics (CAC, LTV, margins)

**Assumptions (Year 1 targets)**:

```
CAC (Customer Acquisition Cost):
- OSS → Free Cloud: $0 (organic)
- Free → Solo: $5 (email nurture)
- Solo → Pro: $20 (in-app + email)
- Pro → Team: $50 (sales touch)
- Team → Business: $200 (light sales)
- Business → Enterprise: $2000 (full sales)

LTV (Lifetime Value):
- Solo: $12 × 12 months avg = $144
- Pro: $39 × 18 months avg = $702
- Team: $99 × 24 months avg = $2,376
- Business: $249 × 30 months avg = $7,470
- Enterprise: $2000/mo × 36 months = $72,000

LTV:CAC Ratios:
- Solo: 144/5 = 28.8x ✅
- Pro: 702/20 = 35.1x ✅
- Team: 2376/50 = 47.5x ✅
- Business: 7470/200 = 37.4x ✅
- Enterprise: 72000/2000 = 36x ✅

Gross Margin:
- Infra cost per user: ~$2-5/mo (cloud, storage, compute)
- Solo margin: ($12-$3)/$12 = 75%
- Pro margin: ($39-$5)/$39 = 87%
- Team margin: ($99-$10)/$99 = 90%
- Business margin: ($249-$20)/$249 = 92%
```

### [x] Revenue projections

**Conservative Scenario (Year 1)**:

| Quarter | Free Users | Paid Users | MRR | ARR |
|---------|------------|------------|-----|-----|
| Q1 | 500 | 10 | $500 | $6K |
| Q2 | 2,000 | 50 | $2,500 | $30K |
| Q3 | 5,000 | 150 | $7,500 | $90K |
| Q4 | 10,000 | 400 | $20,000 | $240K |

**Assumptions**:
- 5% free → paid conversion
- Average paid tier: $50/mo blended
- 10% monthly growth in free users after launch

### [x] Break-even analysis

**Fixed Costs (Monthly)**:
- Founder salary: $0 (bootstrapped initially)
- Cloud infra base: $500/mo
- Tools/services: $200/mo
- Total fixed: $700/mo

**Variable Costs (per user)**:
- Cloud per paid user: $3/mo avg
- Support per paid user: $1/mo avg
- Total variable: $4/mo per paid user

**Break-even**:
- Fixed costs / (ARPU - Variable cost) = Users needed
- $700 / ($50 - $4) = 15 paid users
- **Break-even: ~15 paid users** ✅ (achievable Q1)

### [x] Funding requirements (if any)

**Bootstrap Path (Preferred)**:
- Year 1: Self-funded, reach $20K MRR
- Year 2: Profitable, reinvest
- Year 3: Consider strategic funding for acceleration

**If Funding Needed**:
- Seed: $500K-1M for 18 months runway
- Use: 1-2 hires, marketing, enterprise sales
- Target: $100K MRR before Series A conversation

**Current Status**: Bootstrapping. No external funding required for MVP and initial traction.

---

## 1.3 TIERS & LIMITES (Detalhado)

```
Tier         | Price   | Validations | History | Users | Support    | Features
-------------|---------|-------------|---------|-------|------------|----------
OSS          | $0      | Unlimited   | Local   | N/A   | Community  | Core
Free Cloud   | $0      | 20/mo       | 7d      | 1     | Community  | Core + Dashboard
Solo         | $12/mo  | 200/mo      | 30d     | 1     | Email      | + Exports
Pro          | $39/mo  | 1000/mo     | 90d     | 3     | Priority   | + CI/CD, Webhooks
Team         | $99/mo  | 3000/mo     | 180d    | 10    | Priority   | + Roles, Team Dashboard
Business     | $249/mo | 10000/mo    | 1yr     | 30    | Dedicated  | + SSO, API
Enterprise   | Custom  | Custom      | Custom  | ∞     | Dedicated  | + SLA, On-prem option
```

### [x] Detalhes de cada feature por tier

**OSS (Free Forever)**:
- 7 Human Layers completos
- 6 Perspectives completas
- Triple Redundancy
- MCP Server
- CLI com todas opções
- Configuração via YAML/JSON
- Todos LLM providers (Claude, GPT, Gemini, Ollama)
- Output em JSON/Markdown
- Local file storage
- Community support via GitHub Issues

**Free Cloud**:
- Tudo do OSS +
- Dashboard web básico
- 20 validações/mês
- 7 dias de histórico
- 1 usuário
- Exportação manual (copy/paste)

**Solo ($12/mo)**:
- Tudo do Free Cloud +
- 200 validações/mês
- 30 dias de histórico
- Exportação PDF/JSON/CSV
- Email support (48h SLA)
- Prioridade em bug fixes

**Pro ($39/mo)**:
- Tudo do Solo +
- 1000 validações/mês
- 90 dias de histórico
- 3 usuários (billing owner + 2)
- CI/CD integrations (GitHub Actions, GitLab CI)
- Webhooks (até 5 endpoints)
- Priority support (24h SLA)
- Custom layer configs

**Team ($99/mo)**:
- Tudo do Pro +
- 3000 validações/mês
- 180 dias de histórico
- 10 usuários
- Team dashboard
- Role management (Admin, Member, Viewer)
- Webhooks ilimitados
- Team-wide settings
- Audit log básico

**Business ($249/mo)**:
- Tudo do Team +
- 10000 validações/mês
- 1 ano de histórico
- 30 usuários
- SSO (Google, GitHub, Microsoft)
- REST API access
- Dedicated support (8h SLA)
- Advanced audit log
- Custom integrations support
- Quarterly business review

**Enterprise (Custom)**:
- Tudo do Business +
- Validações custom/ilimitadas
- Histórico custom (até 7 anos)
- Usuários ilimitados
- SAML SSO
- On-premise deployment option
- SLA garantido (99.9%)
- Dedicated success manager
- Custom training
- Compliance docs (SOC2, GDPR DPA)
- Priority feature requests

### [x] O que acontece ao atingir limite

**Validações**:
1. 80% do limite: Banner amarelo no dashboard + email notification
2. 100% do limite: Banner vermelho, botão upgrade proeminente
3. Acima do limite: Validações bloqueadas, mostra "Upgrade to continue"
4. Não há overage automático - upgrade ou esperar próximo mês

**Histórico**:
- Dados além do período são deletados automaticamente
- 7 dias antes: Email warning "Your data will be deleted in 7 days"
- Opção de exportar antes de deletar
- Upgrade preserva dados existentes se dentro do novo limite

**Usuários**:
- Não pode adicionar além do limite
- Pode remover e adicionar outro (swap)
- Upgrade imediato libera slots

### [x] Grace period policies

**Validações**:
- Sem grace period. Limite é limite.
- Razão: Evitar abuse, manter previsibilidade de custos

**Pagamento falho**:
- Day 0: Pagamento falha, retry automático
- Day 3: Email "Payment failed, please update"
- Day 7: Segundo email, banner no dashboard
- Day 14: Downgrade para Free, dados preservados por 30 dias
- Day 44: Dados deletados se não reativar

**Downgrade voluntário**:
- Efetivo no fim do período pago
- Dados preservados dentro do novo limite
- Dados excedentes marcados para deleção (30 dias grace)

### [x] Overage pricing (if any)

**Decisão: SEM OVERAGE PRICING**

Razões:
1. Simplicidade > Receita incremental
2. Evita surpresas no billing
3. Força upgrade consciente
4. Mais fácil de comunicar

Alternativa oferecida: Upgrade mid-cycle com proration

### [x] Upgrade incentives

**Annual Discount**:
- 2 meses grátis no plano anual (16.7% off)
- Solo: $120/ano (vs $144 mensal)
- Pro: $390/ano (vs $468 mensal)
- Team: $990/ano (vs $1188 mensal)
- Business: $2490/ano (vs $2988 mensal)

**First Upgrade**:
- 14 dias grátis no primeiro upgrade para plano pago
- Aplicável uma vez por conta

**Referral**:
- $20 crédito por referral que converte para pago
- Referido ganha 1 mês grátis

### [x] Downgrade policies

**Self-serve downgrade**:
- Disponível a qualquer momento
- Efetivo no fim do período pago atual
- Sem reembolso parcial (mas sem cobrança adicional)

**Dados no downgrade**:
- Histórico: Mantém dados mais recentes dentro do novo limite
- Usuários: Owner mantém acesso, outros viram "inactive"
- Configurações: Preservadas, features bloqueadas ficam read-only
- Webhooks: Desativados se acima do limite

**Reativação**:
- Upgrade restaura acesso a dados preservados
- Dados deletados após grace period não são recuperáveis

### [x] Refund policies

**Política Geral**: Satisfação garantida nos primeiros 30 dias de qualquer plano pago.

**Elegível para refund**:
- Primeiro pagamento de qualquer tier
- Dentro de 30 dias do pagamento
- Sem abuse (ex: usar 900 validações e pedir refund)

**Não elegível**:
- Renovações (já conhecia o produto)
- Após 30 dias
- Enterprise (termos custom)
- Se usou >50% do limite mensal

**Processo**:
1. Email para billing@humangr.ai
2. Resposta em 24h
3. Refund processado em 5-7 dias úteis
4. Conta revertida para Free

---

# PARTE 2: PERSONAS & SEGMENTOS

> **Metodologia**: Personas baseadas em Jobs-to-be-Done (JTBD), com empathy maps completos.
> **Validação**: Cada persona deve ser validável com 5+ entrevistas reais antes de GA.

## 2.1 PERSONAS PRIMÁRIAS

---

### P1: Dev Solo (OSS) — "Alex, the Indie Hacker"

**[x] Demographics**
```yaml
name: "Alex"
age: 28
location: "Remote (US/EU timezone)"
title: "Full-stack Developer / Indie Hacker"
company_size: "Solo or 1-3 person startup"
income: "$60K-120K (or revenue from side projects)"
education: "CS degree or self-taught bootcamp"
experience: "3-7 years coding"
tech_stack: "TypeScript, Python, React, Node.js"
ai_usage: "Heavy - Cursor, Copilot, Claude daily"
```

**[x] Goals & Motivations**
- **Primary JTBD**: "When I'm shipping fast with AI, I want to catch bugs before users do, so I don't lose credibility and users."
- **Secondary JTBD**: "When I'm working alone, I want a second pair of eyes, so I feel confident about my code."

| Goal | Intensity | Frequency |
|------|-----------|-----------|
| Ship fast without breaking things | 🔥🔥🔥🔥🔥 | Daily |
| Learn and improve skills | 🔥🔥🔥 | Weekly |
| Build reputation (OSS, Twitter) | 🔥🔥🔥🔥 | Weekly |
| Minimize costs (bootstrapped) | 🔥🔥🔥🔥🔥 | Always |
| Stay current with AI tools | 🔥🔥🔥🔥 | Weekly |

**[x] Pain Points**
1. **"AI makes me fast but sloppy"** — Ships AI code without proper review, finds bugs in prod
2. **"I can't afford a team"** — No code reviewer, no QA, no second opinion
3. **"I miss obvious stuff"** — Tunnel vision after hours of coding
4. **"Testing is boring"** — Writes minimal tests, hates manual QA
5. **"I don't think like a user"** — Dev mindset misses UX issues
6. **"Security scares me"** — Knows AI might introduce vulnerabilities, doesn't know how to check

**[x] Technical proficiency**
```
Frontend:      ████████░░ 8/10
Backend:       ███████░░░ 7/10
DevOps:        █████░░░░░ 5/10
Security:      ████░░░░░░ 4/10
Testing:       █████░░░░░ 5/10
AI/ML:         ██████░░░░ 6/10
CLI comfort:   █████████░ 9/10
```

**[x] Typical workflow**
```
06:00 - Wake up, check GitHub notifications
07:00 - Coffee, Twitter, catch up on AI news
08:00 - Start coding with Cursor/Claude
12:00 - Lunch, ship to staging
13:00 - Manual testing (quick)
14:00 - More features with AI
18:00 - Push to prod (YOLO or minimal CI)
20:00 - Handle user feedback/bugs
22:00 - Sleep (maybe)
```

**Where Human Layer fits**: Between 13:00 and 14:00 (validation before more features) and before 18:00 (validation before prod).

**[x] Decision factors**
| Factor | Weight | Notes |
|--------|--------|-------|
| **Free/OSS** | 🔥🔥🔥🔥🔥 | Non-negotiable for adoption |
| **Setup time** | 🔥🔥🔥🔥🔥 | Must be <5 minutes |
| **CLI-first** | 🔥🔥🔥🔥 | Hates GUI-only tools |
| **Works with my LLM** | 🔥🔥🔥🔥🔥 | Already paying for Claude Max |
| **No vendor lock-in** | 🔥🔥🔥🔥 | Can switch anytime |
| **Documentation quality** | 🔥🔥🔥 | Good README is enough |
| **Community active** | 🔥🔥🔥 | GitHub stars, Discord |

**[x] Objections & Concerns**
| Objection | Response |
|-----------|----------|
| "Another tool to learn" | 5-minute setup, uses your existing LLM |
| "Will slow me down" | Runs in parallel, doesn't block workflow |
| "AI checking AI is circular" | 7 specialized layers, triple redundancy, consensus |
| "I can review my own code" | 6 perspectives you physically can't simulate |
| "Free = abandoned later" | Apache 2.0, fork-friendly, community-driven |

**[x] Success metrics (for them)**
- **Time to first validation**: < 5 minutes
- **Bugs caught before prod**: 2+ per week
- **Confidence shipping**: "I feel safer" (qualitative)
- **Integration friction**: Zero change to existing workflow
- **Cost**: $0 (uses own LLM subscription)

**Acquisition channels**: GitHub trending, Hacker News, Twitter/X, Reddit r/programming, Dev.to

**Conversion trigger**: Ships a bug that Human Layer would have caught. "Never again."

---

### P2: Dev Solo (Cloud) — "Jordan, the Convenience Seeker"

**[x] Demographics**
```yaml
name: "Jordan"
age: 32
location: "SF Bay Area / NYC / London"
title: "Senior Developer / Tech Lead (small startup)"
company_size: "2-10 employees"
income: "$100K-180K"
education: "CS degree from good school"
experience: "5-10 years"
tech_stack: "Modern stack, whatever's trending"
ai_usage: "Power user - multiple AI tools"
```

**[x] Goals & Motivations**
- **Primary JTBD**: "When I'm juggling multiple responsibilities, I want validation without setup hassle, so I can focus on building product."
- **Secondary JTBD**: "When I need to show investors/stakeholders our quality process, I want a dashboard, so I look professional."

| Goal | Intensity | Frequency |
|------|-----------|-----------|
| Maximize output with minimal overhead | 🔥🔥🔥🔥🔥 | Daily |
| Look professional to stakeholders | 🔥🔥🔥🔥 | Weekly |
| Track quality over time | 🔥🔥🔥 | Monthly |
| Delegate without micromanaging | 🔥🔥🔥🔥 | Weekly |

**[x] Pain Points**
1. **"Self-hosting is a distraction"** — Every minute on infra is a minute not building
2. **"I need history and reports"** — Can't show investors a CLI output
3. **"Setup costs me time"** — Willing to pay to skip config
4. **"I forget to validate"** — Needs automation/reminders
5. **"Local storage is messy"** — Wants cloud-accessible history

**[x] Why Cloud over OSS**
| OSS Friction | Cloud Solution |
|--------------|----------------|
| Local storage management | Cloud history |
| No dashboard | Visual dashboard |
| Manual setup each project | Auto-configured |
| No team sharing | Shareable reports |
| No historical trends | Analytics built-in |

**[x] Typical workflow**
```
07:00 - Coffee, emails, Slack
08:00 - Sprint planning / standups
09:00 - Code with AI assistance
12:00 - Lunch, quick deployment
13:00 - Check Human Layer dashboard
14:00 - Address findings, more features
17:00 - Push to staging, trigger CI/CD
18:00 - Wrap up, check notifications
```

**[x] Decision factors**
| Factor | Weight | Notes |
|--------|--------|-------|
| **Zero setup** | 🔥🔥🔥🔥🔥 | Signup and go |
| **Dashboard** | 🔥🔥🔥🔥🔥 | Visual is non-negotiable |
| **Price** | 🔥🔥🔥 | $12-39/mo is nothing |
| **History/trends** | 🔥🔥🔥🔥 | Need to track over time |
| **Team sharing** | 🔥🔥🔥 | Even if small team |
| **Support available** | 🔥🔥🔥🔥 | Wants email support |

**[x] Objections & Concerns**
| Objection | Response |
|-----------|----------|
| "Why pay when OSS is free?" | Time > money. Dashboard alone worth $12/mo |
| "My data in your cloud?" | BYOK - we never see your code or LLM responses |
| "20 validations/mo is nothing" | Free to try, Solo tier is $12 for 200/mo |

**[x] Success metrics (for them)**
- **Time to first validation**: < 2 minutes (no setup)
- **Dashboard usefulness**: Checks daily
- **Reports generated**: 1+ per week
- **ROI feeling**: "Worth every penny"

**Acquisition channels**: OSS → Cloud upgrade, Google Ads "AI code validation", LinkedIn

**Conversion trigger**: Hits Free tier limit. Upgrades without thinking.

---

### P3: Tech Lead / Team Lead — "Morgan, the Quality Gatekeeper"

**[x] Demographics**
```yaml
name: "Morgan"
age: 35
location: "Global tech hub"
title: "Tech Lead / Team Lead"
company_size: "20-100 employees"
team_size: "3-8 direct reports"
income: "$120K-200K"
experience: "8-12 years"
reports_to: "Engineering Manager or VP Eng"
```

**[x] Goals & Motivations**
- **Primary JTBD**: "When my team uses AI to code, I want to ensure quality without reviewing everything myself, so I can focus on architecture and mentoring."
- **Secondary JTBD**: "When we ship bugs, I want to understand why, so I can improve our process."

| Goal | Intensity | Frequency |
|------|-----------|-----------|
| Maintain code quality | 🔥🔥🔥🔥🔥 | Daily |
| Scale review without burnout | 🔥🔥🔥🔥🔥 | Daily |
| Mentor team effectively | 🔥🔥🔥🔥 | Weekly |
| Meet deadlines | 🔥🔥🔥🔥🔥 | Sprint |
| Look good to management | 🔥🔥🔥 | Quarterly |

**[x] Pain Points**
1. **"I'm the bottleneck"** — Every PR needs my review, queue is infinite
2. **"AI made my team faster but sloppier"** — More PRs, same quality issues
3. **"Junior devs trust AI too much"** — Don't question AI output
4. **"I can't review for security"** — Not my expertise, still responsible
5. **"No visibility into AI usage"** — Team uses AI, I don't know how much

**[x] Team size typical**
- Direct reports: 4-6 developers
- Mix: 1-2 senior, 2-3 mid, 1-2 junior
- AI usage: 80%+ of team uses AI daily

**[x] Decision authority**
```
Can approve without manager:     Tools < $100/mo
Can recommend with justification: Tools < $500/mo
Needs VP/EM approval:            Tools > $500/mo
```

Human Layer Team tier ($99/mo) fits in "recommend with justification" sweet spot.

**[x] Typical workflow**
```
08:00 - Check overnight PRs
09:00 - Standup
09:30 - Architecture/planning
11:00 - PR reviews (queue of 5-10)
12:00 - Lunch
13:00 - More reviews, 1:1s
15:00 - Own coding work
17:00 - Final review pass
18:00 - Plan tomorrow, go home
```

**Where Human Layer fits**: Replace 50% of the 11:00 and 13:00 review time. Focus on AI-generated PRs.

**[x] Objections & Concerns**
| Objection | Response |
|-----------|----------|
| "My team won't use it" | Integrates with CI/CD, automatic |
| "I should review code myself" | You still do - this catches what you miss |
| "Another tool to manage" | 10-minute setup, runs automatically |
| "What if it misses something?" | Triple redundancy, better than single reviewer |
| "My manager won't approve" | $99/mo ROI: saves 10+ hours/week of review |

**[x] Success metrics (for them)**
- **Review time reduction**: 50%+ (10h → 5h/week)
- **Bugs in prod from AI code**: 80% reduction
- **Team satisfaction**: "Finally, help with reviews"
- **Manager perception**: "Team ships faster, fewer incidents"

**Acquisition channels**: Team member already using OSS, word of mouth, dev conferences

**Conversion trigger**: Big bug ships because review was rushed. "Never again."

---

### P4: Engineering Manager — "Taylor, the Process Optimizer"

**[x] Demographics**
```yaml
name: "Taylor"
age: 38
location: "Major tech hub"
title: "Engineering Manager"
company_size: "50-500 employees"
teams_managed: "2-4 teams (10-25 ICs)"
income: "$150K-250K"
experience: "10-15 years (5+ in management)"
reports_to: "VP Engineering or CTO"
```

**[x] Goals & Motivations**
- **Primary JTBD**: "When AI increases my team's output, I want to ensure quality scales too, so velocity doesn't come at the cost of reliability."
- **Secondary JTBD**: "When executives ask about AI risks, I want data and processes, so I can demonstrate we're managing it responsibly."

| Goal | Intensity | Frequency |
|------|-----------|-----------|
| Hit delivery targets | 🔥🔥🔥🔥🔥 | Sprint |
| Maintain quality metrics | 🔥🔥🔥🔥🔥 | Sprint |
| Manage AI adoption risk | 🔥🔥🔥🔥 | Quarterly |
| Keep team happy/retained | 🔥🔥🔥🔥 | Always |
| Report to leadership | 🔥🔥🔥 | Monthly |

**[x] Pain Points**
1. **"AI is a black box"** — Team uses it, I have no visibility
2. **"Quality metrics are slipping"** — Bug rate up since AI adoption
3. **"No process for AI code"** — Same review for AI and human code
4. **"Executives asking hard questions"** — "What's our AI governance?"
5. **"Tool fatigue on team"** — Another tool is another overhead

**[x] Budget authority**
```
Direct approval:        < $500/mo per team
With justification:     < $2000/mo
VP/Finance approval:    > $2000/mo
```

Human Layer Business tier ($249/mo) fits in "direct approval" for most EMs.

**[x] Decision process**
1. Tech Lead recommends
2. EM evaluates ROI (time saved × hourly rate vs cost)
3. Trial period (2-4 weeks)
4. Review metrics
5. Roll out or reject
6. Report to VP if successful

**[x] Objections & Concerns**
| Objection | Response |
|-----------|----------|
| "We have code review" | This augments, doesn't replace |
| "Can't justify another tool" | ROI: 1 prevented incident = 10x cost |
| "Team won't adopt" | CI/CD integration = automatic |
| "Security concerns" | BYOK, no code leaves your infra |
| "How do I measure success?" | Dashboard with metrics, trends, reports |

**[x] Success metrics (for them)**
- **Bug escape rate**: 50% reduction
- **Review cycle time**: 30% faster
- **Team velocity**: Maintained or improved
- **Exec confidence**: "We have AI governance"
- **ROI**: 5x+ (saves 40+ hours/month across teams)

**Acquisition channels**: VP/CTO referral down, Tech Lead referral up, industry reports, conferences

**Conversion trigger**: Board asks "What's your AI risk mitigation strategy?" Panic → Google → Human Layer.

---

### P5: DevOps / Platform Engineer — "Casey, the Automation Architect"

**[x] Demographics**
```yaml
name: "Casey"
age: 33
location: "Remote-first"
title: "Senior DevOps Engineer / Platform Engineer"
company_size: "50-500 employees"
income: "$130K-200K"
experience: "7-12 years"
focus: "CI/CD, infrastructure, developer experience"
```

**[x] Goals & Motivations**
- **Primary JTBD**: "When teams ship faster with AI, I want automated quality gates, so I'm not the bottleneck or the cleanup crew."
- **Secondary JTBD**: "When I integrate new tools, I want them to be CI/CD native, so they don't break my pipelines."

| Goal | Intensity | Frequency |
|------|-----------|-----------|
| Automate everything | 🔥🔥🔥🔥🔥 | Always |
| Reliable pipelines | 🔥🔥🔥🔥🔥 | Daily |
| Developer experience | 🔥🔥🔥🔥 | Sprint |
| Reduce toil | 🔥🔥🔥🔥🔥 | Always |
| Stay current | 🔥🔥🔥 | Weekly |

**[x] Pain Points**
1. **"AI PRs flood the pipeline"** — 3x more PRs, same CI resources
2. **"Flaky AI code breaks builds"** — AI introduces subtle issues
3. **"No validation step for AI"** — Linters catch syntax, not logic
4. **"Devs bypass quality gates"** — "It works on my machine"
5. **"Incident fatigue"** — Oncall for AI-induced bugs

**[x] CI/CD expertise**
```
GitHub Actions:    ████████░░ 8/10
GitLab CI:         ███████░░░ 7/10
Jenkins:           █████████░ 9/10
CircleCI:          ██████░░░░ 6/10
ArgoCD/GitOps:     ████████░░ 8/10
Terraform:         █████████░ 9/10
Kubernetes:        ████████░░ 8/10
```

**[x] Typical workflow**
```
07:00 - Check overnight alerts
08:00 - Review pipeline failures
09:00 - Standup, support requests
10:00 - Pipeline optimization
12:00 - Lunch
13:00 - Tool integrations
15:00 - Documentation
16:00 - PR reviews (infra)
17:00 - Plan automation projects
```

**[x] Objections & Concerns**
| Objection | Response |
|-----------|----------|
| "Another step in pipeline" | Parallel execution, doesn't block |
| "Will it break my builds?" | Configurable: warn vs block |
| "Maintenance burden" | Managed cloud, zero maintenance |
| "Integration complexity" | Native GitHub Actions, GitLab CI templates |
| "Performance impact" | Async validation, no pipeline delay |

**[x] Success metrics (for them)**
- **Pipeline reliability**: No Human Layer-caused failures
- **Integration time**: < 30 minutes
- **Maintenance time**: < 1 hour/month
- **Developer adoption**: 80%+ PRs validated
- **Incident reduction**: Fewer AI-induced prod issues

**Acquisition channels**: DevOps community (Slack, Discord), HashiCorp/CNCF adjacent, Platform Eng blogs

**Conversion trigger**: AI-generated code takes down prod at 3am. "We need a gate."

---

### P6: QA Engineer — "Riley, the Quality Champion"

**[x] Demographics**
```yaml
name: "Riley"
age: 30
location: "Global"
title: "QA Engineer / SDET"
company_size: "50-500 employees"
income: "$80K-140K"
experience: "5-10 years"
focus: "Test automation, quality processes"
```

**[x] Goals & Motivations**
- **Primary JTBD**: "When AI generates code faster than I can test, I want automated validation, so I can focus on complex test scenarios."
- **Secondary JTBD**: "When developers skip testing, I want a safety net, so bugs don't reach users."

| Goal | Intensity | Frequency |
|------|-----------|-----------|
| Catch bugs before prod | 🔥🔥🔥🔥🔥 | Daily |
| Expand test coverage | 🔥🔥🔥🔥 | Sprint |
| Reduce manual testing | 🔥🔥🔥🔥🔥 | Always |
| Influence dev quality | 🔥🔥🔥 | Weekly |
| Career growth | 🔥🔥🔥 | Quarterly |

**[x] Pain Points**
1. **"AI code is undertested"** — Devs assume AI code is correct
2. **"Testing backlog grows"** — AI increases output, QA capacity fixed
3. **"Edge cases missed"** — AI doesn't think about edge cases
4. **"Accessibility overlooked"** — Devs don't think about a11y
5. **"Security is not my job but..."** — QA catches security issues too late

**[x] Testing expertise**
```
Manual testing:       █████████░ 9/10
Test automation:      ████████░░ 8/10
API testing:          ████████░░ 8/10
Performance testing:  ██████░░░░ 6/10
Security testing:     █████░░░░░ 5/10
Accessibility testing:███████░░░ 7/10
```

**[x] Typical workflow**
```
08:00 - Review overnight test results
09:00 - Standup, bug triage
10:00 - Exploratory testing
12:00 - Lunch
13:00 - Test automation
15:00 - Review PRs (test perspective)
16:00 - Write test cases
17:00 - Documentation
```

**[x] Objections & Concerns**
| Objection | Response |
|-----------|----------|
| "Overlaps with my job" | Augments, frees you for complex scenarios |
| "AI can't test like humans" | 6 perspectives, triple redundancy |
| "Dev tools, not QA tools" | QA-friendly dashboard, detailed findings |
| "Another tool to learn" | Simple reports, no learning curve |

**[x] Success metrics (for them)**
- **Bug escape rate**: 60% reduction
- **Test coverage feeling**: "More confident"
- **Time on complex testing**: 50% more
- **Recognition**: "QA improved since Human Layer"
- **Collaboration with devs**: Better quality conversations

**Acquisition channels**: QA community (Ministry of Testing), SDET blogs, Test automation forums

**Conversion trigger**: Prod bug that QA "should have caught" but was overwhelmed.

---

### P7: CTO / VP Engineering — "Drew, the Strategic Leader"

**[x] Demographics**
```yaml
name: "Drew"
age: 42
location: "Major tech hub"
title: "CTO / VP Engineering"
company_size: "100-1000 employees"
teams: "5-15 teams (50-150 engineers)"
income: "$200K-400K + equity"
experience: "15-20 years (7+ in leadership)"
reports_to: "CEO / Board"
```

**[x] Goals & Motivations**
- **Primary JTBD**: "When AI transforms how we build software, I want governance and quality assurance, so we capture benefits without creating liability."
- **Secondary JTBD**: "When the board asks about AI risks, I want demonstrable processes, so we maintain trust and avoid regulatory issues."

| Goal | Intensity | Frequency |
|------|-----------|-----------|
| Ship faster than competitors | 🔥🔥🔥🔥🔥 | Quarterly |
| Manage AI risk | 🔥🔥🔥🔥🔥 | Monthly |
| Retain top talent | 🔥🔥🔥🔥 | Always |
| Control costs | 🔥🔥🔥🔥 | Quarterly |
| Board confidence | 🔥🔥🔥🔥🔥 | Quarterly |

**[x] Pain Points**
1. **"AI is everywhere, ungoverned"** — No visibility into AI code quality
2. **"Liability concerns"** — AI-generated bugs could be costly
3. **"Compliance questions"** — Auditors asking about AI processes
4. **"Security exposure"** — AI might introduce vulnerabilities at scale
5. **"Cultural resistance"** — Some team leads resist AI, others embrace recklessly

**[x] Budget authority**
```
Direct approval:        < $50K/year
With CEO:               < $200K/year
Board involvement:      > $200K/year
```

Human Layer Enterprise is well within direct approval range.

**[x] Decision process**
1. Problem identified (incident, audit, board question)
2. VP Eng/EM proposes solution
3. CTO evaluates strategic fit
4. Security/Legal review
5. Pilot with 1-2 teams
6. Metrics review (30-60 days)
7. Org-wide rollout or rejection
8. Quarterly business review

**[x] Compliance concerns**
| Concern | Human Layer Answer |
|---------|-------------------|
| Data privacy | BYOK - code never leaves your infra |
| Audit trail | Full history, exportable logs |
| SOC2 | On roadmap (2027 Q2) |
| GDPR | Compliant by design |
| AI governance | Built-in policy enforcement |
| Vendor risk | OSS core, can self-host |

**[x] Success metrics (for them)**
- **Incident reduction**: 50%+ (AI-related)
- **Audit readiness**: "We have documented AI governance"
- **Developer productivity**: Maintained or improved
- **Cost efficiency**: ROI > 10x
- **Board confidence**: "Risk managed"

**Acquisition channels**: Board/peer referral, Gartner/Forrester mentions, Enterprise sales outreach

**Conversion trigger**: AI incident makes news. Board asks "Could this happen to us?"

---

## 2.2 PERSONAS SECUNDÁRIAS

---

### P8: Open Source Contributor — "Sam, the Community Builder"

**[x] Motivations**
- Build reputation in AI/DevTools space
- Learn by contributing to real project
- Give back to community
- Network with other contributors
- Potential job opportunities

**[x] Contribution types**
| Type | Effort | Impact | Recognition |
|------|--------|--------|-------------|
| Bug reports | Low | Medium | Issue credit |
| Documentation | Low-Medium | High | Contributors list |
| Bug fixes | Medium | Medium | PR credit |
| New features | High | High | Changelog mention |
| Integrations | High | Very High | Partner status |
| Translations | Medium | High | i18n credits |

**[x] Recognition needs**
- Name in CONTRIBUTORS.md
- Changelog mentions
- Discord/Community role
- Swag (stickers, t-shirts)
- LinkedIn endorsement
- Reference letters

**[x] Community engagement**
- GitHub Discussions participation
- Discord community membership
- Issue triage volunteer
- Documentation reviewer
- Beta tester

---

### P9: Integration Partner — "Acme CI/CD Inc."

**[x] Partner types (CI/CD vendors, etc.)**
| Type | Examples | Value Exchange |
|------|----------|----------------|
| CI/CD platforms | GitHub, GitLab, CircleCI | Native integration |
| IDE vendors | Cursor, VS Code | Extension |
| AI assistants | Claude, GPT apps | MCP native |
| LLM providers | Anthropic, OpenAI | Recommended tool |
| Security vendors | Snyk, SonarQube | Complementary positioning |
| Observability | Datadog, Sentry | Error correlation |

**[x] Partnership models**
- **Technology partner**: Free integration, co-marketing
- **Marketplace listing**: Listing in their marketplace/directory
- **Co-development**: Joint features, shared roadmap
- **Reseller**: Partner sells our Cloud tier
- **OEM**: White-label for their enterprise customers

**[x] Technical requirements**
- Clean API/SDK for integration
- Webhook support
- OAuth/SSO capability
- Sandbox environment for testing
- Technical documentation
- Developer support contact

**[x] Business terms**
- No revenue share for basic integration
- Co-marketing commitment
- Joint case study
- Quarterly sync calls
- Escalation path

---

### P10: Evaluator / POC User — "The Proof-of-Concept Champion"

**[x] Evaluation criteria**
| Criteria | Weight | Evaluation Method |
|----------|--------|-------------------|
| Ease of setup | 25% | Time to first validation |
| Feature completeness | 20% | Checklist |
| Integration capability | 20% | CI/CD test |
| Security/compliance | 15% | Questionnaire |
| Cost/ROI | 10% | Calculation |
| Support quality | 10% | Response time test |

**[x] Timeline typical**
```
Week 1: Discovery, initial evaluation
Week 2: Technical POC (1-2 projects)
Week 3: Team trial (3-5 users)
Week 4: Metrics review, decision
```

**[x] Decision process**
1. Champion identifies need
2. Shortlist 2-3 tools
3. Quick evaluation (free tier)
4. Deeper POC (1 week)
5. Stakeholder demo
6. Budget approval
7. Contract/signup
8. Rollout plan

**[x] Success criteria for POC**
- Setup < 30 minutes
- First validation successful
- At least 2 bugs caught that would have been missed
- Team feedback positive (7+/10)
- No security red flags
- ROI math works (saves > costs)

---

# PARTE 3: USER JOURNEYS (E2E)

> **Metodologia**: Customer Journey Mapping com touchpoints, emotions, e pain points por estágio.
> **Formato**: Cada jornada inclui fluxo detalhado, métricas de sucesso, e pontos de intervenção.

---

## 3.1 DISCOVERY & AWARENESS

---

### J1: Descoberta Orgânica — "Encontrando Human Layer via Google"

**[x] Search keywords (Target)**

| Keyword | Intent | Volume Est. | Difficulty | Priority |
|---------|--------|-------------|------------|----------|
| "AI code validation" | High intent | Medium | Medium | P0 |
| "validate AI generated code" | High intent | Low-Medium | Low | P0 |
| "AI code review tool" | High intent | Medium | High | P1 |
| "human in the loop AI" | Research | High | High | P2 |
| "LLM output validation" | Technical | Low | Low | P0 |
| "Claude code validation" | High intent | Low | Low | P0 |
| "GPT code checker" | High intent | Medium | Medium | P1 |
| "AI code quality assurance" | High intent | Low | Low | P0 |
| "triple redundancy AI" | Unique | Very Low | Very Low | P1 |
| "MCP validation server" | Technical | Very Low | Very Low | P0 |

**SEO Content Strategy**:
- Landing page optimized for "AI code validation"
- Blog posts for long-tail keywords
- Documentation pages for technical queries
- GitHub README for "MCP" queries

**[x] Landing page experience**

```
┌─────────────────────────────────────────────────────────────────────────┐
│ SCROLL 0 (Above the fold - 5 seconds to hook)                          │
│                                                                         │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ HEADLINE: "Validate AI Code Before It Breaks Production"            │ │
│ │ SUBHEAD: "7 Human Layers × Triple Redundancy × Your LLM"           │ │
│ │                                                                     │ │
│ │ [Get Started Free] [View Docs] [GitHub ★ 1.2k]                     │ │
│ │                                                                     │ │
│ │ ┌─────────────────────────────────────────────────────────────┐     │ │
│ │ │          [Hero Visual: Validation Flow Animation]            │     │ │
│ │ └─────────────────────────────────────────────────────────────┘     │ │
│ │                                                                     │ │
│ │ "Used by 500+ developers" | "OSS Forever" | "Works with Claude"    │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
│
│ SCROLL 1 (Problem Statement)
│ "AI generates code 10x faster. But who validates 10x faster?"
│ - 73% of AI-generated code ships without proper review
│ - Manual review doesn't scale
│ - Linters catch syntax, not logic
│
│ SCROLL 2 (Solution - 7 Layers)
│ [Interactive layer selector with examples]
│
│ SCROLL 3 (How It Works - 3 Steps)
│ pip install → Configure → Validate
│
│ SCROLL 4 (Social Proof)
│ Testimonials, GitHub stars, logos
│
│ SCROLL 5 (Pricing Preview)
│ OSS: Free | Cloud: From $12/mo
│
│ SCROLL 6 (Final CTA)
│ "Start validating in 5 minutes"
```

**Load Performance Targets**:
- LCP < 2.0s
- FID < 50ms
- CLS < 0.05
- Mobile: 95+ Lighthouse score

**[x] Value proposition clarity**

**Primary VP (5 words)**: "Validate AI code, ship confidently."

**Secondary VPs by section**:
| Section | VP |
|---------|-----|
| Hero | "7 layers of human-like review for AI code" |
| Problem | "Your AI codes fast. Who reviews that fast?" |
| Solution | "Structured validation that scales with your AI" |
| Features | "Security, UX, edge cases - all covered" |
| Social Proof | "Join 500+ developers who ship with confidence" |

**[x] Call to action (CTA hierarchy)**

| CTA | Location | Style | Goal |
|-----|----------|-------|------|
| "Get Started Free" | Hero, sticky header | Primary (blue) | Signup/Install |
| "View Docs" | Hero | Secondary (outline) | Documentation |
| "Star on GitHub" | Hero, footer | Tertiary (ghost) | Social proof |
| "Try Demo" | Features section | Primary | Experience |
| "Contact Sales" | Pricing, Enterprise | Secondary | Enterprise lead |

**[x] Competitive comparison**

**Comparison Table on Site**:

| Feature | Human Layer | Manual Review | Linters | AI Assistants |
|---------|-------------|---------------|---------|---------------|
| Understands context | ✅ | ✅ | ❌ | ⚠️ |
| Scales infinitely | ✅ | ❌ | ✅ | ✅ |
| Multiple perspectives | ✅ (6) | ❌ (1) | ❌ | ❌ |
| Triple redundancy | ✅ | ❌ | ❌ | ❌ |
| Security focus | ✅ | ⚠️ | ⚠️ | ❌ |
| Audit trail | ✅ | ⚠️ | ❌ | ❌ |
| Works with AI code | ✅ | ✅ | ⚠️ | ⚠️ |
| Free/OSS option | ✅ | ✅ | ✅ | ⚠️ |

**[x] Trust signals**

| Signal | Location | Implementation |
|--------|----------|----------------|
| GitHub stars | Header, hero | Live counter via API |
| OSS badge | Hero | "Apache 2.0 License" badge |
| Security | Footer | "SOC2 Roadmap" / "Your data stays yours" |
| BYOK | Features | "Uses YOUR LLM subscription" |
| Companies | Social proof | Logos (when available) |
| Testimonials | Social proof | Quotes with photos/names |

**[x] Social proof**

**Launch Strategy (Before Users)**:
1. GitHub stars from beta testers
2. Testimonials from private beta
3. "Featured on" badges (HN, ProductHunt)
4. Tweet embeds from developers
5. "Join X developers on waitlist"

**Post-Launch**:
1. Company logos (with permission)
2. Case studies
3. Usage statistics ("X validations run")
4. Integration partner logos
5. Media mentions

---

### J2: Descoberta via Referral — "Friend Recommends Human Layer"

**[x] Referral sources**

| Source | Typical Message | Conversion Rate Est. |
|--------|-----------------|---------------------|
| Colleague at work | "We use this, you should try" | 40% |
| Twitter/X mention | "Just found this tool 🔥" | 15% |
| Discord/Slack share | Link in dev community | 20% |
| GitHub star notification | "X starred humangr/human-layer" | 5% |
| Blog mention | "Tools I use for AI coding" | 10% |
| Conference talk | Speaker mentions in slides | 25% |

**[x] First impression (Referral landing)**

**Referral Link Format**: `humangr.ai/ref/{referrer_id}`

**Referral Landing Variant**:
```
"Alex thinks you'll love Human Layer"
[Same hero but personalized]
+ "Alex has saved 40 hours using Human Layer"
[Both Alex and you get 1 month free when you sign up]
```

**[x] Expectation setting**

**Referrer provides context**:
- "It's like having 7 reviewers for your AI code"
- "Free OSS, works with Claude"
- "5 minute setup"

**Landing confirms**:
- Same messaging as referrer said
- No bait and switch
- Clear OSS vs Cloud distinction

**[x] Referral incentives**

| Tier | Referrer Gets | Referred Gets |
|------|---------------|---------------|
| Free → Free | $0 | $0 |
| Free → Paid | $20 credit | 1 month free |
| Paid → Paid | 1 month free | 1 month free |
| Enterprise | $500 Amazon card | 2 months free |

**Referral Mechanics**:
- Unique referral link per user
- Dashboard shows referral stats
- Credit applied automatically
- Email notification on conversion

---

### J3: Descoberta via Content — "Found via Blog/Video"

**[x] Blog posts (Content Strategy)**

| Category | Topics | Frequency | Goal |
|----------|--------|-----------|------|
| Product | Release notes, features | 2/month | Retention |
| Engineering | How we built X | 1/month | Technical credibility |
| Tutorials | How to use with X | 2/month | SEO, onboarding |
| Thought Leadership | AI code quality trends | 1/month | Authority |
| Case Studies | Customer stories | 1/quarter | Social proof |

**Launch Content Plan**:
1. "Why We Built Human Layer"
2. "The 7 Layers Explained"
3. "How Triple Redundancy Works"
4. "Setting Up Human Layer in 5 Minutes"
5. "Human Layer vs Manual Code Review"

**[x] Tutorials**

| Tutorial | Format | Platform | Goal |
|----------|--------|----------|------|
| "5-Minute Setup" | Video + Blog | YouTube, Blog | Onboarding |
| "CI/CD Integration" | Blog + Code | Blog, GitHub | Adoption |
| "Custom Layer Config" | Blog | Blog | Power users |
| "Enterprise Setup" | Docs | Docs site | Enterprise |

**[x] Videos**

| Video Type | Length | Platform | Frequency |
|------------|--------|----------|-----------|
| Product demo | 2-3 min | YouTube, Landing | Launch |
| Tutorial | 5-10 min | YouTube | Monthly |
| Release highlights | 1-2 min | Twitter, YouTube | Per release |
| Conference talk | 20-40 min | YouTube | Quarterly |
| Livestream | 30-60 min | YouTube, Twitch | Monthly |

**[x] Podcasts**

**Guest Strategy**:
- Target dev-focused podcasts (Changelog, Syntax, etc.)
- Pitch angle: "AI code quality crisis"
- Provide host with demo account
- Follow up with exclusive content

**Own Podcast (Future)**:
- "The Human Layer Podcast"
- AI code quality topics
- Guest developers

**[x] Conference talks**

| Conference Type | Talk Angle | Goal |
|-----------------|------------|------|
| AI/ML conferences | "Validating LLM outputs at scale" | Thought leadership |
| DevOps/Platform | "Quality gates for AI code" | DevOps personas |
| Testing/QA | "Automated validation beyond tests" | QA personas |
| Security | "Security implications of AI code" | Security angle |
| General dev (PyCon, JSConf) | "Shipping AI code safely" | Broad awareness |

---

## 3.2 EVALUATION & TRIAL

---

### J4: Avaliação OSS — "Trying Self-Hosted Version"

**[x] README first impression**

```markdown
# Human Layer 🛡️

> Validate AI-generated code with 7 human-like layers and triple redundancy.

[![GitHub Stars](badge)][License: Apache 2.0](badge)[![PyPI](badge)]

## 🚀 Quick Start (5 minutes)

​```bash
pip install human-layer
human-layer init
human-layer validate ./my-project
​```

## 🎯 What It Does

- **7 Specialized Layers**: UI/UX, Security, Edge Cases, A11y, Performance, Integration, Final Review
- **6 Perspectives**: Simulates tired user, malicious insider, confused newbie, power user, auditor, 3am operator
- **Triple Redundancy**: 3 runs per layer, 2/3 consensus required
- **Works with YOUR LLM**: Claude, GPT, Gemini, Ollama - your subscription, your data

## 📖 Documentation

[Full docs](link) | [Getting Started](link) | [Configuration](link)

## 🌟 Why Human Layer?

AI generates code 10x faster. But who reviews 10x faster?

Human Layer provides structured, scalable validation that catches what linters miss.

## ☁️ Cloud Version

Want dashboard, history, and team features? Try [Human Layer Cloud](link) - free tier available.

## 🤝 Contributing

We love contributions! See [CONTRIBUTING.md](link).

## 📜 License

Apache 2.0 - Free forever, fork-friendly.
```

**README Metrics to Track**:
- Time to first star after view
- Clone rate
- Issue creation rate
- PR submission rate

**[x] Installation steps**

**Primary (pip)**:
```bash
# Python 3.9+
pip install human-layer
```

**Alternative (Docker)**:
```bash
docker run -v $(pwd):/app ghcr.io/humangr/human-layer validate /app
```

**Alternative (From Source)**:
```bash
git clone https://github.com/humangr/human-layer.git
cd human-layer
pip install -e .
```

**Verification**:
```bash
human-layer --version
# human-layer 1.0.0

human-layer doctor
# ✅ Python 3.11
# ✅ LLM provider configured
# ✅ Ready to validate
```

**[x] First run experience**

**Zero-Config First Run**:
```bash
cd my-project
human-layer validate .

# Output:
# 🔍 Scanning project...
# 📁 Found 12 files to validate
# 🧠 Using Claude (from ANTHROPIC_API_KEY)
#
# Running Layer 1/7: UI/UX Review...
# Running Layer 2/7: Security Scan...
# ...
#
# ═══════════════════════════════════════════
# VALIDATION COMPLETE
# ═══════════════════════════════════════════
#
# 🟢 PASS: 5 layers
# 🟡 WARN: 2 layers (3 findings)
# 🔴 FAIL: 0 layers
#
# FINDINGS:
#
# [MEDIUM] HL-2 Security: SQL injection risk in query.py:42
#          → Use parameterized queries
#
# [LOW] HL-1 UX: Button label unclear in form.tsx:18
#       → Consider "Submit Order" instead of "Go"
#
# [LOW] HL-3 Edge Case: Empty array not handled in utils.py:77
#       → Add check for empty input
#
# Full report: ./human-layer-report.json
```

**Success Moment**: First finding that user agrees with = "Aha! This works!"

**[x] Configuration complexity**

**Level 1 - Zero Config**:
```bash
# Just set LLM API key
export ANTHROPIC_API_KEY=sk-...
human-layer validate .
```

**Level 2 - Basic Config (human-layer.yaml)**:
```yaml
llm:
  provider: claude
  model: claude-3-sonnet

layers:
  enabled: all  # or [security, ux, edge-cases]

redundancy: 3  # 1-5

output:
  format: json
  file: ./report.json
```

**Level 3 - Advanced Config**:
```yaml
llm:
  provider: claude
  model: claude-3-opus
  fallback:
    provider: openai
    model: gpt-4

layers:
  security:
    veto_power: strong
    custom_rules:
      - "Check for hardcoded credentials"
      - "Verify input sanitization"

  ux:
    veto_power: medium
    perspectives: [tired_user, confused_newbie]

perspectives:
  weights:
    tired_user: 1.5
    malicious_insider: 2.0

thresholds:
  consensus: 0.67  # 2/3
  confidence_min: 0.8

ignore:
  paths: [node_modules, .git, __pycache__]
  patterns: ["*.test.js", "*.spec.py"]
```

**[x] First success moment**

**Definition**: User sees a finding they agree with and would have missed.

**Optimization for Success Moment**:
1. Example project included with known issues
2. First validation finds real issues
3. Clear, actionable suggestions
4. User thinks "This would have been a bug"

**[x] Documentation quality**

**Docs Structure**:
```
docs.humangr.ai/
├── Getting Started
│   ├── Installation
│   ├── Quick Start
│   ├── First Validation
│   └── Configuration
├── Concepts
│   ├── 7 Human Layers
│   ├── 6 Perspectives
│   ├── Triple Redundancy
│   ├── Consensus & Veto
│   └── BYOK (Bring Your Own Key)
├── Guides
│   ├── CI/CD Integration
│   ├── Team Setup
│   ├── Custom Layers
│   └── Troubleshooting
├── Reference
│   ├── CLI Commands
│   ├── Configuration Options
│   ├── MCP Tools
│   ├── API Reference
│   └── Error Codes
└── Community
    ├── Contributing
    ├── Changelog
    └── FAQ
```

**Docs Quality Standards**:
- Every page has "Edit on GitHub"
- Code examples are copy-paste ready
- All examples tested in CI
- Search works well (Algolia DocSearch)
- Mobile-friendly

**[x] Community support availability**

| Channel | Response Time | Type |
|---------|---------------|------|
| GitHub Issues | <24h | Bug reports, features |
| GitHub Discussions | <48h | Questions, ideas |
| Discord | <4h (community) | Chat, help |
| Stack Overflow | Community | Q&A |

**[x] Decision to adopt or abandon**

**Abandon Triggers**:
- Installation fails → Need better error messages
- No findings on first run → Need example project
- Findings seem wrong → Need better prompts
- Too slow → Need performance optimization
- Too expensive (LLM) → Need cost estimates upfront

**Adopt Triggers**:
- Finds real issue in first 5 minutes
- Setup was easy
- Documentation was helpful
- Community was responsive
- Integrates with existing workflow

---

### J5: Avaliação Cloud Free — "Trying Cloud Dashboard"

**[x] Signup flow**

```
Step 1: Landing → Click "Get Started Free"
        ↓
Step 2: Signup Page
        ┌─────────────────────────────────┐
        │ Create your account             │
        │                                 │
        │ [Continue with Google]          │
        │ [Continue with GitHub]          │
        │                                 │
        │ ─────── or ───────              │
        │                                 │
        │ Email: [________________]       │
        │ Password: [________________]    │
        │                                 │
        │ ☐ I agree to Terms & Privacy   │
        │                                 │
        │ [Create Account]                │
        │                                 │
        │ Already have account? Sign in   │
        └─────────────────────────────────┘
        ↓
Step 3: Email Verification (if email signup)
        "Check your email for verification link"
        ↓
Step 4: Welcome Screen
        "Welcome to Human Layer!"
        [Start Setup →]
```

**Signup Metrics**:
- Signup page → Account created: Target 60%+
- OAuth vs Email: Track ratio
- Time to complete: Target <60 seconds

**[x] Onboarding wizard**

```
ONBOARDING WIZARD (4 Steps)

Step 1/4: Connect Your LLM ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
┌─────────────────────────────────────────────────────────────┐
│ Human Layer uses YOUR LLM subscription.                     │
│ Your code and data never leave your control.                │
│                                                             │
│ Select provider:                                            │
│ ○ Claude (Anthropic) - Recommended                         │
│ ○ GPT-4 (OpenAI)                                           │
│ ○ Gemini (Google)                                          │
│ ○ Ollama (Local)                                           │
│                                                             │
│ API Key: [sk-ant-api03-...]                                │
│          🔒 Encrypted, never stored in plain text          │
│                                                             │
│ [Test Connection]  ✅ Connected!                           │
│                                                             │
│ [← Back]                               [Continue →]         │
└─────────────────────────────────────────────────────────────┘

Step 2/4: First Validation ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
┌─────────────────────────────────────────────────────────────┐
│ Let's validate some code!                                   │
│                                                             │
│ ○ Use sample project (recommended for first time)          │
│ ○ Connect GitHub repo                                      │
│ ○ Paste code directly                                      │
│                                                             │
│ [Run Validation]                                           │
│                                                             │
│ ┌─────────────────────────────────────────────────────┐     │
│ │ 🔍 Running validation...                            │     │
│ │ Layer 2/7: Security Scan ████████░░░░ 65%          │     │
│ └─────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘

Step 3/4: Review Results ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
┌─────────────────────────────────────────────────────────────┐
│ ✅ Validation Complete!                                     │
│                                                             │
│ Found 3 issues:                                            │
│ 🔴 [HIGH] SQL Injection in query.py                        │
│ 🟡 [MED] Missing error handling in api.js                  │
│ 🟢 [LOW] Unclear button label in form.tsx                  │
│                                                             │
│ 💡 See how Human Layer found issues you might have missed? │
│                                                             │
│ [View Full Report]                    [Continue →]          │
└─────────────────────────────────────────────────────────────┘

Step 4/4: You're Ready! ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
┌─────────────────────────────────────────────────────────────┐
│ 🎉 Welcome to Human Layer!                                  │
│                                                             │
│ What's next?                                               │
│ • Explore your Dashboard                                   │
│ • Set up CI/CD integration (Pro)                          │
│ • Invite team members (Team)                              │
│ • Read the docs                                           │
│                                                             │
│ Free tier: 20 validations/month, 7 days history           │
│ Need more? [View Plans]                                    │
│                                                             │
│ [Go to Dashboard →]                                        │
└─────────────────────────────────────────────────────────────┘
```

**[x] First validation**
- Guided with sample project
- Shows progress in real-time
- Explains each layer as it runs
- Celebrates findings (not failures)

**[x] Dashboard first use**
- Tooltip tour (optional, skippable)
- Recent validation prominent
- Empty states helpful ("Run your first validation")
- Quick actions visible

**[x] Feature discovery**
- Progressive disclosure
- Tooltips on hover
- "Pro feature" badges (not blocking)
- "Learn more" links

**[x] Limit awareness**
```
┌─────────────────────────────────────────────────────────────┐
│ Usage This Month                                           │
│ ████████████████░░░░░░░░░░░░░░░░ 16/20 validations        │
│                                                             │
│ 4 validations remaining • Resets in 12 days               │
│ [Upgrade for more →]                                       │
└─────────────────────────────────────────────────────────────┘
```

**[x] Upgrade prompts (Non-intrusive)**

| Trigger | Prompt Style | Frequency |
|---------|--------------|-----------|
| 80% limit | Yellow banner | Once |
| 100% limit | Block + modal | When hitting |
| CI/CD attempt | Inline tooltip | Once |
| Team invite | Inline tooltip | Once |
| 7th day | Email | Once |
| Day 25 | Email | Once |

**[x] Decision to upgrade or churn**

**Upgrade Triggers**:
- Hit limit and need more
- Want CI/CD integration
- Want team features
- Happy with product

**Churn Triggers**:
- Didn't find value in free tier
- Too expensive for perceived value
- Found alternative
- Project ended

---

### J6: POC / Enterprise Evaluation — "Formal Enterprise Trial"

**[x] Contact sales flow**

```
/enterprise → "Contact Sales" button
        ↓
Contact Form:
- Company name
- Your name
- Email (work)
- Team size (dropdown)
- Current AI usage (dropdown)
- Anything else?
        ↓
Confirmation: "We'll reach out within 24 hours"
        ↓
Sales rep emails to schedule call
```

**[x] POC setup**

**POC Package**:
- 30-day full Business tier access
- Dedicated Slack channel
- Weekly check-in calls
- Success criteria agreed upfront
- 2-3 teams piloting

**[x] Success criteria definition**

| Metric | Target | How Measured |
|--------|--------|--------------|
| Setup time | <1 hour | Time tracking |
| Bugs caught | 5+ per team | Dashboard |
| False positive rate | <20% | User feedback |
| Team adoption | 80%+ using | Dashboard |
| User satisfaction | 7+/10 | Survey |

**[x] Technical evaluation**

**Security Questionnaire Topics**:
- Data handling (BYOK - we never see code)
- Authentication methods
- Encryption standards
- Compliance certifications
- Incident response
- Vendor risk

**[x] Security review**

**Security Documentation Provided**:
- Architecture diagram
- Data flow diagram
- Encryption details
- SOC2 roadmap
- Penetration test results (when available)
- Security questionnaire (pre-filled)

**[x] Compliance check**

| Compliance | Status | Documentation |
|------------|--------|---------------|
| GDPR | Compliant | DPA available |
| CCPA | Compliant | Privacy policy |
| SOC2 Type 1 | Roadmap 2027 Q2 | Timeline doc |
| SOC2 Type 2 | Roadmap 2027 Q4 | Timeline doc |
| HIPAA | Not applicable | N/A |
| FedRAMP | Not planned | N/A |

**[x] Pricing negotiation**

**Enterprise Pricing Variables**:
- Number of users
- Validation volume
- History retention
- Support level
- SLA requirements
- Contract length
- Payment terms

**Typical Enterprise Deal**:
- $2,000-10,000/month
- Annual contract
- Net 30 payment
- Custom SLA

**[x] Contract signing**

**Contract Includes**:
- Master Service Agreement (MSA)
- Data Processing Agreement (DPA)
- Service Level Agreement (SLA)
- Statement of Work (SOW)
- Order Form

**Timeline**: 1-4 weeks depending on legal complexity

---

## 3.3 ONBOARDING (Detalhado)

---

### J7: Onboarding OSS — "Self-Hosted Setup"

**[x] Step 1: Prerequisites check**

```bash
human-layer doctor

# Output:
# ┌─────────────────────────────────────────────────────────┐
# │ Human Layer System Check                                │
# ├─────────────────────────────────────────────────────────┤
# │ ✅ Python 3.11.4 (3.9+ required)                       │
# │ ✅ pip 23.2.1                                          │
# │ ✅ Node.js 18.17.0 (for Playwright, optional)          │
# │ ✅ Git 2.41.0                                          │
# │ ✅ ANTHROPIC_API_KEY set                               │
# │ ⚠️  OPENAI_API_KEY not set (optional fallback)         │
# │ ✅ Disk space: 2.1GB free                              │
# │ ✅ Network: Can reach api.anthropic.com                │
# ├─────────────────────────────────────────────────────────┤
# │ Ready to validate! 🚀                                  │
# └─────────────────────────────────────────────────────────┘
```

**[x] Step 2: Installation**

```bash
# Option A: pip (recommended)
pip install human-layer

# Option B: pipx (isolated)
pipx install human-layer

# Option C: Docker
docker pull ghcr.io/humangr/human-layer:latest

# Option D: From source
git clone https://github.com/humangr/human-layer.git
cd human-layer
pip install -e ".[dev]"
```

**Verification**:
```bash
human-layer --version
# human-layer v1.0.0

human-layer --help
# Available commands: validate, init, config, doctor
```

**[x] Step 3: Configuration**

```bash
# Initialize config
human-layer init

# Creates human-layer.yaml with defaults
# Prompts for LLM API key if not set
```

**[x] Step 4: First Run**

```bash
# Option A: Validate current directory
human-layer validate .

# Option B: Try with example project
human-layer example
# Downloads small example project with known issues
# Runs validation
# Shows expected findings

# Option C: Validate specific files
human-layer validate ./src/api.py ./src/utils.py
```

**[x] Step 5: Customization**

```yaml
# human-layer.yaml

# Customize which layers run
layers:
  security: true
  ux: true
  edge_cases: true
  accessibility: false  # Disable if not relevant
  performance: true
  integration: true
  final_review: true

# Customize perspectives
perspectives:
  enabled: [tired_user, malicious_insider, power_user]

# Adjust thresholds
validation:
  redundancy: 3  # 1-5, default 3
  consensus_threshold: 0.67  # 2/3 majority

# Ignore patterns
ignore:
  paths: [node_modules, .git, vendor, __pycache__]
  patterns: ["*.test.*", "*.spec.*", "*_test.py"]
```

---

### J8: Onboarding Cloud — "Dashboard Setup"

*[Wizard flow detailed above in J5]*

**Key Metrics**:
- Time from signup to first validation: <5 minutes
- Onboarding completion rate: >80%
- Drop-off by step: Track and optimize

---

### J9: Onboarding CI/CD — "Pipeline Integration"

**[x] Step 1: Platform Selection**

```
Supported Platforms:
├── GitHub Actions (recommended) - native action
├── GitLab CI - YAML template
├── CircleCI - orb available
├── Jenkins - plugin
├── Azure DevOps - extension
├── Bitbucket Pipelines - pipe
└── Generic - CLI in any CI
```

**[x] Step 2: Configuration (GitHub Actions example)**

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

      - uses: humangr/human-layer-action@v1
        with:
          api-key: ${{ secrets.HUMANLAYER_API_KEY }}
          llm-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
          layers: 'security,ux,edge-cases'
          fail-on: 'high,critical'

      - name: Comment on PR
        uses: actions/github-script@v6
        with:
          script: |
            const report = require('./human-layer-report.json');
            // Post summary as PR comment
```

**[x] Step 3: First Automated Run**

- Push PR to trigger
- Watch action run
- See results in PR comment
- Review findings

**[x] Step 4: Optimization**

```yaml
# Optimizations:
- cache: true  # Cache LLM responses for unchanged files
- changed-only: true  # Only validate changed files
- parallel: true  # Run layers in parallel
- timeout: 600  # 10 min max
```

---

### J10: Onboarding Team — "Team Setup"

**[x] Step 1: Team Creation**

```
Settings → Teams → Create Team
- Team name: "Acme Engineering"
- Team URL: humangr.ai/teams/acme-eng
- Billing: [Add payment method]
```

**[x] Step 2: Member Invitation**

```
Team Settings → Members → Invite
- Email addresses (one per line)
- Role: Admin / Member / Viewer
- Personal message (optional)
- [Send Invites]

Invite email:
"You've been invited to join Acme Engineering on Human Layer"
[Accept Invitation]
```

**[x] Step 3: Member Acceptance**

- Click link in email
- Sign up (if new) or sign in
- Automatically added to team
- See team dashboard

**[x] Step 4: Collaboration Setup**

- Shared validation history
- Team-wide notification settings
- Shared configuration templates
- Role-based access

---

## 3.4 DAILY USAGE

---

### J11: Validação Interativa (Manual) — "Ad-hoc Validation"

**[x] Complete Flow**

```
Developer: "I just finished this feature with Claude. Let me validate it."

1. Open Claude Desktop
2. Human Layer MCP is loaded (pre-configured)
3. Use /validate command or natural language

   User: "Validate the code in src/checkout/ for security and UX issues"

   Human Layer:
   "🔍 Starting validation of src/checkout/

    Running 7 layers with triple redundancy...

    Layer 1/7: UI/UX Review ████████████ Complete
    Layer 2/7: Security Scan ████████████ Complete
    ...

    ═══════════════════════════════════════
    VALIDATION SUMMARY
    ═══════════════════════════════════════

    🔴 CRITICAL (1):
    • [HL-2] Payment data logged to console (payment.js:47)
      Risk: PCI compliance violation
      Fix: Remove console.log, use secure logging

    🟡 MEDIUM (2):
    • [HL-1] Unclear error message for declined cards
    • [HL-3] Edge case: Empty cart not handled

    🟢 LOW (1):
    • [HL-5] Could benefit from lazy loading images

    Confidence: 94% (consensus 3/3 on critical findings)

    Would you like me to:
    • Explain any finding in detail?
    • Generate a fix for a specific issue?
    • Re-run with different layers?"

4. Developer reviews, addresses critical
5. Re-validates (optional)
6. Marks resolved, continues work
```

---

### J12: Validação Automatizada (CI/CD) — "PR Validation"

**[x] Complete Flow**

```
1. Developer pushes PR
   └─> GitHub Action triggers

2. Human Layer runs
   └─> Validates changed files
   └─> Posts results to PR

3. PR Comment:
   ┌─────────────────────────────────────────────────────────┐
   │ 🛡️ Human Layer Validation                               │
   ├─────────────────────────────────────────────────────────┤
   │ Status: ⚠️ Warnings Found                               │
   │ Files: 4 validated                                      │
   │ Time: 2m 34s                                            │
   ├─────────────────────────────────────────────────────────┤
   │ Findings:                                               │
   │                                                         │
   │ 🔴 **CRITICAL** (blocks merge)                         │
   │ None ✅                                                 │
   │                                                         │
   │ 🟡 **MEDIUM** (review recommended)                      │
   │ • auth.py:42 - Session token not invalidated on logout │
   │                                                         │
   │ 🟢 **LOW** (informational)                             │
   │ • form.tsx:18 - Consider aria-label for icon button    │
   │                                                         │
   │ [View Full Report](link) | [Re-run](link)              │
   └─────────────────────────────────────────────────────────┘

4. Developer reviews comment
   └─> Addresses medium finding
   └─> Pushes fix

5. Re-validation runs automatically
   └─> Status: ✅ All Clear

6. PR approved and merged
```

---

### J13: Dashboard Usage — "Daily Check-in"

**[x] Activities**

```
Morning routine (Tech Lead):

1. Open dashboard (humangr.ai/dashboard)
2. Quick glance at stats
   - 12 validations yesterday
   - 3 critical findings (all resolved)
   - Team velocity: Normal

3. Filter: Last 24h, Status: Open
   - See 2 open medium findings
   - Assign to team members

4. Check trends
   - Security findings down 20% this week 📉
   - Edge case findings stable

5. Generate weekly report
   - Export PDF for standup
```

---

### J14: Report Review — "Sprint Retrospective"

**[x] Flow**

```
End of sprint report review:

1. Access Reports → This Sprint
2. Summary:
   - 47 validations
   - 156 findings
   - 89% resolution rate

3. Breakdown by layer:
   - Security: 23 findings (14 critical) ← Focus area
   - UX: 41 findings
   - Edge cases: 38 findings
   - ...

4. Top recurring issues:
   - Input validation (18 occurrences)
   - Error handling (12 occurrences)
   - Accessibility labels (15 occurrences)

5. Action items:
   - Create input validation utility
   - Training on error handling patterns
   - Update component library for a11y
```

---

## 3.5 UPGRADE & CONVERSION

*[Detailed upgrade flows - abbreviated for length. Each upgrade path includes:]*

- **Trigger**: What causes user to consider upgrade
- **UI**: Upgrade prompt design (non-intrusive)
- **Flow**: Step-by-step upgrade process
- **Confirmation**: Welcome to new tier
- **Success criteria**: User uses new features within 7 days

---

## 3.6 DOWNGRADE & CHURN

### J20-J22: Downgrade, Cancellation, Deletion

**Key Principles**:
1. Make it easy (no dark patterns)
2. Understand why (exit survey)
3. Offer alternatives (pause, downgrade vs cancel)
4. Preserve data option (export before delete)
5. Leave door open (win-back email in 30 days)

**Exit Survey Questions**:
- Primary reason for leaving?
- What could we have done better?
- Would you recommend Human Layer?
- Can we contact you in the future?

---

## 3.7 SUPPORT & HELP

### J23: Self-Service Support

**Documentation hierarchy**:
1. In-app help (tooltips, inline docs)
2. Docs site (comprehensive)
3. FAQ (common questions)
4. Community (Discord, GitHub)

### J24: Assisted Support

**SLA by tier**:
| Tier | First Response | Resolution |
|------|----------------|------------|
| Free | Community | Community |
| Solo | 48h email | 5 days |
| Pro | 24h email | 3 days |
| Team | 24h email | 2 days |
| Business | 8h email | 1 day |
| Enterprise | 4h + phone | 4h critical |

### J25: Escalation Path

```
L1: Support team → Basic questions, known issues
 ↓ (if unresolved 24h)
L2: Technical support → Complex technical issues
 ↓ (if unresolved 48h or critical)
L3: Engineering → Bugs, edge cases
 ↓ (if business impact or VIP)
Executive: Founder/CEO → Enterprise escalation
```

---

# PARTE 4: FEATURES COMPLETAS

> **Formato**: Especificações técnicas completas com interfaces, comportamentos, e invariantes.
> **Padrão**: Cada feature deve ser implementável sem ambiguidade a partir desta spec.

---

## 4.1 CORE ENGINE

---

### F1: HumanLayerRunner — "Orquestrador Principal"

**[x] Interface**

```python
class HumanLayerRunner:
    """
    Orquestrador principal que coordena validação através das 7 camadas.

    Invariantes:
    - INV-001: Sempre executa layers na ordem definida (1→7)
    - INV-002: Triple redundancy é default (configurável 1-5)
    - INV-003: Timeout global default: 10 minutos
    - INV-004: Progress reporting via callback ou async generator
    """

    def __init__(
        self,
        llm_client: LLMClient,
        config: HumanLayerConfig,
        progress_callback: Optional[Callable[[Progress], None]] = None,
    ) -> None: ...

    async def validate(
        self,
        target: ValidationTarget,
        layers: Optional[List[LayerID]] = None,  # None = all
        perspectives: Optional[List[PerspectiveID]] = None,  # None = all
        redundancy: int = 3,  # 1-5
        timeout: float = 600.0,  # seconds
    ) -> ValidationReport: ...

    async def validate_stream(
        self,
        target: ValidationTarget,
        **kwargs,
    ) -> AsyncGenerator[ValidationEvent, None]: ...

    def cancel(self) -> None: ...
```

**[x] Input handling**

```python
@dataclass
class ValidationTarget:
    """O que será validado."""

    # Exatamente um deve ser fornecido
    code: Optional[str] = None              # Código direto
    files: Optional[List[Path]] = None       # Lista de arquivos
    directory: Optional[Path] = None         # Diretório (recursivo)
    git_diff: Optional[str] = None           # Git diff
    url: Optional[str] = None                # URL para validar UI

    # Metadados
    context: Optional[str] = None            # Contexto adicional
    language: Optional[str] = None           # Linguagem (auto-detect se None)
    framework: Optional[str] = None          # Framework (auto-detect se None)

    def validate(self) -> None:
        """Raises ValueError se input inválido."""
        sources = [self.code, self.files, self.directory, self.git_diff, self.url]
        if sum(1 for s in sources if s is not None) != 1:
            raise ValueError("Exactly one input source required")
```

**[x] Layer execution**

```python
async def _execute_layer(
    self,
    layer: Layer,
    target: ValidationTarget,
    perspectives: List[Perspective],
    redundancy: int,
) -> LayerResult:
    """
    Executa uma layer com triple redundancy.

    Flow:
    1. Prepara prompt com target + perspectives
    2. Executa N vezes (redundancy)
    3. Coleta resultados
    4. Calcula consenso
    5. Retorna LayerResult consolidado
    """
    results: List[SingleRunResult] = []

    for run_idx in range(redundancy):
        prompt = self._build_layer_prompt(layer, target, perspectives)

        try:
            response = await self.llm_client.complete(prompt)
            parsed = self._parse_layer_response(response)
            results.append(SingleRunResult(
                run_index=run_idx,
                findings=parsed.findings,
                confidence=parsed.confidence,
                raw_response=response,
            ))
        except LLMError as e:
            results.append(SingleRunResult(
                run_index=run_idx,
                error=e,
            ))

    return self._consolidate_results(layer, results)
```

**[x] Result aggregation**

```python
def _consolidate_results(
    self,
    layer: Layer,
    results: List[SingleRunResult],
) -> LayerResult:
    """
    Consolida N runs em resultado único via consenso.

    Algoritmo:
    1. Agrupa findings similares (fuzzy match)
    2. Finding presente em ≥2/3 runs → incluído
    3. Confidence = média das confidences concordantes
    4. Status = mais severo dos findings consensuais
    """
    # Agrupa findings por similaridade
    finding_groups = self._group_similar_findings(
        [f for r in results for f in r.findings]
    )

    consensus_findings = []
    for group in finding_groups:
        occurrence_rate = len(group.occurrences) / len(results)
        if occurrence_rate >= self.config.consensus_threshold:  # default 0.67
            consensus_findings.append(Finding(
                description=group.canonical_description,
                severity=group.max_severity,
                confidence=group.avg_confidence * occurrence_rate,
                layer_id=layer.id,
                suggestion=group.best_suggestion,
                location=group.location,
                occurrences=len(group.occurrences),
            ))

    return LayerResult(
        layer_id=layer.id,
        status=self._calculate_status(consensus_findings, layer.veto_power),
        findings=consensus_findings,
        run_count=len(results),
        successful_runs=len([r for r in results if not r.error]),
    )
```

**[x] Error handling**

| Error Type | Handling | Retry | User Message |
|------------|----------|-------|--------------|
| LLM rate limit | Exponential backoff | 3x | "Rate limited, retrying..." |
| LLM timeout | Increase timeout, retry | 2x | "LLM slow, retrying..." |
| LLM API error | Log, fail run | 1x | "LLM error, partial results" |
| Parse error | Log, skip finding | 0x | Warning in report |
| Network error | Retry with backoff | 3x | "Network issue, retrying..." |
| Config error | Fail fast | 0x | "Invalid configuration: {details}" |

**[x] Timeout handling**

```python
TIMEOUT_CONFIG = {
    "layer_timeout": 120.0,      # Per layer
    "run_timeout": 60.0,         # Per individual run
    "total_timeout": 600.0,      # Total validation
    "llm_timeout": 30.0,         # Per LLM call
}

async def _execute_with_timeout(self, coro, timeout, name):
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"{name} timed out after {timeout}s")
        raise LayerTimeoutError(name, timeout)
```

**[x] Retry logic**

```python
RETRY_CONFIG = {
    "max_retries": 3,
    "base_delay": 1.0,           # seconds
    "max_delay": 30.0,           # seconds
    "exponential_base": 2,
    "jitter": True,
    "retryable_errors": [
        RateLimitError,
        TemporaryNetworkError,
        LLMOverloadedError,
    ],
}
```

**[x] Cancellation**

```python
def cancel(self) -> None:
    """
    Cancela validação em andamento.

    Comportamento:
    - Define flag _cancelled = True
    - Layers em execução terminam o run atual
    - Resultados parciais são retornados
    - Status final = CANCELLED
    """
    self._cancelled = True
    for task in self._running_tasks:
        task.cancel()
```

**[x] Progress reporting**

```python
@dataclass
class Progress:
    phase: Literal["preparing", "running", "consolidating", "complete"]
    current_layer: Optional[LayerID]
    current_run: Optional[int]
    total_runs: int
    layers_complete: int
    total_layers: int
    findings_so_far: int
    elapsed_seconds: float
    estimated_remaining: Optional[float]

    @property
    def percent(self) -> float:
        return (self.layers_complete / self.total_layers) * 100
```

---

### F2: TripleRedundancy — "Sistema de Consenso"

**[x] Especificação**

```python
class TripleRedundancy:
    """
    Executa cada layer N vezes e consolida via consenso.

    Invariantes:
    - INV-010: Default N=3 (configurável 1-5)
    - INV-011: Consenso requer ≥67% concordância (configurável)
    - INV-012: Findings sem consenso são descartados
    - INV-013: Confidence final = confidence média × taxa de concordância
    """

    def __init__(
        self,
        redundancy: int = 3,
        consensus_threshold: float = 0.67,
        similarity_threshold: float = 0.8,  # Para agrupar findings
    ): ...
```

**[x] 3x execution**

```python
async def execute(
    self,
    layer: Layer,
    prompt: str,
    llm_client: LLMClient,
) -> List[SingleRunResult]:
    """
    Executa prompt N vezes em paralelo.

    Paralelização:
    - Runs são independentes
    - Falha de um run não afeta outros
    - Resultados parciais são válidos
    """
    tasks = [
        self._single_run(layer, prompt, llm_client, run_idx)
        for run_idx in range(self.redundancy)
    ]
    return await asyncio.gather(*tasks, return_exceptions=True)
```

**[x] Consensus calculation (2/3)**

```python
def calculate_consensus(
    self,
    runs: List[SingleRunResult],
) -> ConsensusResult:
    """
    Algoritmo de consenso:

    1. Extrai todos findings de todos runs
    2. Agrupa findings similares (embedding similarity > 0.8)
    3. Para cada grupo:
       - Se presente em ≥67% dos runs → CONSENSO
       - Severity = max severity do grupo
       - Confidence = avg confidence × occurrence rate
    4. Findings sem consenso → descartados (não reportados)
    """
    all_findings = [f for r in runs for f in r.findings if not r.error]
    groups = self._group_by_similarity(all_findings)

    consensus_findings = []
    for group in groups:
        rate = group.occurrence_count / len([r for r in runs if not r.error])
        if rate >= self.consensus_threshold:
            consensus_findings.append(self._merge_group(group, rate))

    return ConsensusResult(
        findings=consensus_findings,
        total_runs=len(runs),
        successful_runs=len([r for r in runs if not r.error]),
        consensus_rate=len(consensus_findings) / max(len(groups), 1),
    )
```

**[x] Tiebreaker logic**

```python
def _resolve_tie(self, findings: List[Finding]) -> Finding:
    """
    Quando findings são similares mas diferem em detalhes:

    1. Severity: maior vence (CRITICAL > HIGH > MEDIUM > LOW > INFO)
    2. Description: mais detalhada vence (mais palavras)
    3. Suggestion: mais específica vence
    4. Confidence: média ponderada
    """
    return Finding(
        severity=max(f.severity for f in findings),
        description=max(findings, key=lambda f: len(f.description)).description,
        suggestion=max(findings, key=lambda f: len(f.suggestion or "")).suggestion,
        confidence=sum(f.confidence for f in findings) / len(findings),
    )
```

**[x] Partial failure handling**

```python
PARTIAL_FAILURE_POLICY = {
    "min_successful_runs": 2,       # Mínimo para resultado válido
    "fallback_on_total_failure": "error_result",  # ou "retry_sync"
}

def handle_partial_failure(self, runs: List[SingleRunResult]) -> ConsensusResult:
    successful = [r for r in runs if not r.error]

    if len(successful) >= self.min_successful_runs:
        # Continua com runs bem-sucedidos
        return self.calculate_consensus(successful)
    elif len(successful) > 0:
        # Warning mas continua
        logger.warning(f"Only {len(successful)} runs succeeded")
        return self.calculate_consensus(successful)
    else:
        # Falha total
        raise AllRunsFailedError(runs)
```

**[x] Cost optimization**

```python
COST_OPTIMIZATION_STRATEGIES = {
    "cache_identical_inputs": True,      # Cache por hash do input
    "parallel_execution": True,          # Runs em paralelo
    "early_termination": True,           # Para se 3/3 concordam em PASS
    "reduced_redundancy_for_low_risk": True,  # 1x para arquivos triviais
}

def should_reduce_redundancy(self, target: ValidationTarget) -> int:
    """
    Reduz redundancy para inputs de baixo risco:
    - Arquivos pequenos (<50 linhas): redundancy=1
    - Apenas documentação: redundancy=1
    - Arquivos de config: redundancy=1
    """
    if target.is_documentation_only:
        return 1
    if target.total_lines < 50:
        return 1
    if target.is_config_only:
        return 1
    return self.default_redundancy
```

---

### F3: LayerOrchestrator — "Sequenciador de Layers"

**[x] Layer sequencing**

```python
LAYER_SEQUENCE = [
    LayerID.HL1_UX,          # 1. UI/UX primeiro (mais visível)
    LayerID.HL2_SECURITY,    # 2. Security (mais crítico)
    LayerID.HL3_EDGE_CASES,  # 3. Edge cases
    LayerID.HL4_A11Y,        # 4. Accessibility
    LayerID.HL5_PERFORMANCE, # 5. Performance
    LayerID.HL6_INTEGRATION, # 6. Integration
    LayerID.HL7_FINAL,       # 7. Final review (holístico)
]

# Ordem é fixa por design:
# - Security cedo para short-circuit em falhas críticas
# - Final review por último (precisa de contexto das outras)
```

**[x] Parallel execution option**

```python
PARALLEL_CONFIG = {
    "enabled": True,           # Layers 1-6 podem rodar em paralelo
    "max_concurrent": 3,       # Max 3 layers simultâneas
    "layer_7_always_last": True,  # Final review sempre espera outras
}

async def execute_parallel(self, layers: List[Layer]) -> List[LayerResult]:
    # Layers 1-6 em paralelo (máx 3 concurrent)
    parallel_layers = [l for l in layers if l.id != LayerID.HL7_FINAL]
    semaphore = asyncio.Semaphore(self.max_concurrent)

    async def run_with_semaphore(layer):
        async with semaphore:
            return await self._execute_layer(layer)

    results = await asyncio.gather(*[
        run_with_semaphore(l) for l in parallel_layers
    ])

    # Layer 7 por último
    if LayerID.HL7_FINAL in [l.id for l in layers]:
        final_result = await self._execute_layer(
            self.layers[LayerID.HL7_FINAL],
            context=results,  # Passa resultados anteriores
        )
        results.append(final_result)

    return results
```

**[x] Dependency handling**

```python
LAYER_DEPENDENCIES = {
    LayerID.HL7_FINAL: [LayerID.HL1_UX, LayerID.HL2_SECURITY],  # Precisa contexto
}

def _resolve_dependencies(self, layers: List[LayerID]) -> List[List[LayerID]]:
    """Retorna layers agrupadas por nível de dependência."""
    # Topological sort
    ...
```

**[x] Short-circuit on critical failure**

```python
SHORT_CIRCUIT_CONFIG = {
    "enabled": True,
    "trigger_on": [VetoLevel.STRONG],  # STRONG veto = para tudo
    "layers_that_can_short_circuit": [LayerID.HL2_SECURITY],
}

async def _check_short_circuit(self, result: LayerResult) -> bool:
    """
    Retorna True se deve parar execução.

    Trigger:
    - Layer com veto STRONG encontra finding CRITICAL
    - Security (HL-2) encontra vulnerabilidade crítica
    """
    if result.layer_id == LayerID.HL2_SECURITY:
        critical_findings = [f for f in result.findings if f.severity == Severity.CRITICAL]
        if critical_findings and result.layer.veto_power == VetoLevel.STRONG:
            logger.warning("Short-circuiting due to critical security finding")
            return True
    return False
```

**[x] Result merging**

```python
def merge_results(self, layer_results: List[LayerResult]) -> ValidationReport:
    """
    Merge todos LayerResults em ValidationReport final.

    Algoritmo:
    1. Coleta todos findings
    2. Deduplica findings idênticos cross-layer
    3. Ordena por severity (CRITICAL first)
    4. Calcula status final (worst status wins)
    5. Gera métricas agregadas
    """
    all_findings = []
    for lr in layer_results:
        for finding in lr.findings:
            finding.layer_id = lr.layer_id  # Ensure layer attribution
            all_findings.append(finding)

    # Dedupe
    unique_findings = self._deduplicate_findings(all_findings)

    # Sort
    unique_findings.sort(key=lambda f: f.severity.value, reverse=True)

    return ValidationReport(
        status=self._calculate_final_status(layer_results),
        findings=unique_findings,
        layer_results=layer_results,
        summary=self._generate_summary(unique_findings),
        metadata=self._collect_metadata(layer_results),
    )
```

---

### F4: VetoGate — "Sistema de Veto"

**[x] Veto levels**

```python
class VetoLevel(Enum):
    """
    Níveis de veto com semântica clara.

    NONE: Informational only, nunca bloqueia
    WEAK: Pode adicionar warnings, nunca bloqueia
    MEDIUM: Pode bloquear merge, não bloqueia deploy
    STRONG: Pode bloquear tudo (merge, deploy, release)
    """
    NONE = "none"
    WEAK = "weak"
    MEDIUM = "medium"
    STRONG = "strong"

    @property
    def can_block_merge(self) -> bool:
        return self in [VetoLevel.MEDIUM, VetoLevel.STRONG]

    @property
    def can_block_deploy(self) -> bool:
        return self == VetoLevel.STRONG

    @property
    def can_block_release(self) -> bool:
        return self == VetoLevel.STRONG
```

**[x] Layer veto powers**

| Layer | Veto Power | Rationale |
|-------|------------|-----------|
| HL-1 UI/UX | MEDIUM | UX issues should block merge, not deploy |
| HL-2 Security | STRONG | Security issues block everything |
| HL-3 Edge Cases | MEDIUM | Edge cases block merge |
| HL-4 A11y | MEDIUM | A11y issues block merge |
| HL-5 Performance | WEAK | Performance is warning only |
| HL-6 Integration | MEDIUM | Integration issues block merge |
| HL-7 Final | STRONG | Final review can block everything |

**[x] Override mechanism**

```python
@dataclass
class VetoOverride:
    """
    Override de veto requer:
    1. Usuário autenticado
    2. Reason obrigatório
    3. Audit log criado
    4. Expira em 24h (re-validation required)
    """
    finding_id: str
    user_id: str
    reason: str
    created_at: datetime
    expires_at: datetime  # +24h from created_at

    def is_valid(self) -> bool:
        return datetime.utcnow() < self.expires_at

def override_veto(
    self,
    finding: Finding,
    user: User,
    reason: str,
) -> VetoOverride:
    if len(reason) < 20:
        raise ValueError("Override reason must be at least 20 characters")

    override = VetoOverride(
        finding_id=finding.id,
        user_id=user.id,
        reason=reason,
        created_at=datetime.utcnow(),
        expires_at=datetime.utcnow() + timedelta(hours=24),
    )

    # Audit log (always)
    self.audit_log.record(AuditEvent(
        type="veto_override",
        user_id=user.id,
        finding_id=finding.id,
        reason=reason,
        severity=finding.severity,
    ))

    return override
```

---

### F5: ConsensusEngine — "Motor de Consenso"

**[x] Especificação completa**

```python
class ConsensusEngine:
    """
    Calcula consenso entre múltiplas execuções.

    Invariantes:
    - INV-020: Threshold default = 67% (2/3)
    - INV-021: Weighted voting opcional (default: equal weights)
    - INV-022: Confidence nunca > 1.0
    - INV-023: Disagreement sempre logado para análise
    """

    def __init__(
        self,
        threshold: float = 0.67,
        similarity_threshold: float = 0.8,
        use_weighted_voting: bool = False,
    ): ...

    def calculate(
        self,
        runs: List[SingleRunResult],
        weights: Optional[List[float]] = None,
    ) -> ConsensusResult: ...
```

**[x] Confidence scoring**

```python
def calculate_confidence(
    self,
    finding_group: FindingGroup,
    total_runs: int,
) -> float:
    """
    Confidence = base_confidence × occurrence_rate × agreement_bonus

    Onde:
    - base_confidence: média das confidences individuais
    - occurrence_rate: runs com finding / total runs
    - agreement_bonus: 1.0 se unânime, 0.9 se 2/3, 0.8 se marginal
    """
    base = sum(f.confidence for f in finding_group.findings) / len(finding_group.findings)
    rate = finding_group.occurrence_count / total_runs

    if rate >= 1.0:
        bonus = 1.0  # Unânime
    elif rate >= 0.8:
        bonus = 0.95
    elif rate >= 0.67:
        bonus = 0.9
    else:
        bonus = 0.8

    return min(1.0, base * rate * bonus)
```

---

## 4.2 SEVEN HUMAN LAYERS

---

### F6: HL-1 UI/UX Review

**[x] Especificação**

```yaml
layer:
  id: HL-1
  name: "UI/UX Review"
  veto_power: MEDIUM
  focus: "User interface, user experience, usability"

system_prompt: |
  You are a senior UX designer reviewing code for user interface and
  user experience issues. Focus on:

  1. USABILITY
     - Is the interface intuitive?
     - Are actions reversible?
     - Is feedback immediate and clear?
     - Are error messages helpful?

  2. VISUAL DESIGN
     - Is layout consistent?
     - Is typography readable?
     - Are colors accessible (contrast)?
     - Is spacing consistent?

  3. INTERACTION DESIGN
     - Are click targets large enough (44px mobile)?
     - Is loading state indicated?
     - Are transitions smooth?
     - Is focus management correct?

  4. INFORMATION ARCHITECTURE
     - Is navigation clear?
     - Is content hierarchy logical?
     - Are labels descriptive?

  For each issue found, provide:
  - Severity (critical/high/medium/low/info)
  - Location (file:line or component name)
  - Description (what's wrong)
  - Suggestion (how to fix)

  Output as JSON array of findings.

checklist:
  - "Form labels are associated with inputs"
  - "Buttons have descriptive text (not just 'Click here')"
  - "Loading states exist for async operations"
  - "Error messages are user-friendly"
  - "Success feedback is provided"
  - "Destructive actions have confirmation"
  - "Navigation is consistent"
  - "Mobile touch targets >= 44px"

red_flags:
  - "alert() or confirm() used"
  - "No loading indicator for API calls"
  - "Cryptic error codes shown to user"
  - "No way to undo destructive action"
  - "Form submits on enter unexpectedly"

example_findings:
  - severity: MEDIUM
    location: "checkout.tsx:47"
    description: "Delete button has no confirmation dialog"
    suggestion: "Add confirmation modal before deleting cart items"

  - severity: LOW
    location: "form.tsx:23"
    description: "Submit button says 'Go' - unclear action"
    suggestion: "Change to 'Submit Order' or 'Complete Purchase'"
```

---

### F7: HL-2 Security Scan

**[x] Especificação**

```yaml
layer:
  id: HL-2
  name: "Security Scan"
  veto_power: STRONG  # Can block everything
  focus: "Security vulnerabilities, OWASP Top 10, data protection"

system_prompt: |
  You are a senior security engineer performing a security review.
  This is a CRITICAL layer - security issues can block deployment.

  Check for:

  1. INJECTION (OWASP A03)
     - SQL injection
     - Command injection
     - LDAP injection
     - XPath injection
     - Template injection

  2. BROKEN AUTHENTICATION (OWASP A07)
     - Weak passwords allowed
     - Session fixation
     - Missing MFA for sensitive ops
     - Credential exposure in logs

  3. SENSITIVE DATA EXPOSURE (OWASP A02)
     - Hardcoded secrets/keys
     - PII in logs
     - Unencrypted sensitive data
     - Excessive data exposure

  4. XSS (OWASP A03)
     - Reflected XSS
     - Stored XSS
     - DOM-based XSS
     - Missing output encoding

  5. BROKEN ACCESS CONTROL (OWASP A01)
     - Missing authorization checks
     - IDOR vulnerabilities
     - Privilege escalation
     - CORS misconfiguration

  6. SECURITY MISCONFIGURATION (OWASP A05)
     - Debug mode in production
     - Default credentials
     - Unnecessary features enabled
     - Missing security headers

  7. INSECURE DEPENDENCIES
     - Known vulnerable packages
     - Outdated dependencies
     - Typosquatting risk

  SEVERITY MAPPING:
  - CRITICAL: Exploitable vulnerability (RCE, SQL injection, auth bypass)
  - HIGH: Significant risk (XSS, IDOR, sensitive data exposure)
  - MEDIUM: Moderate risk (missing headers, weak validation)
  - LOW: Minor issues (informational disclosure, best practices)

checklist:
  - "No hardcoded credentials or API keys"
  - "User input is sanitized/validated"
  - "Parameterized queries used for SQL"
  - "Output encoding for HTML/JS contexts"
  - "Authentication on all protected routes"
  - "Authorization checks before actions"
  - "HTTPS enforced"
  - "Security headers present (CSP, HSTS, etc.)"
  - "Dependencies up to date"
  - "No sensitive data in logs"

red_flags:
  - "exec(), eval(), or similar with user input"
  - "String concatenation in SQL queries"
  - "innerHTML with user data"
  - "API keys or passwords in source"
  - "console.log with sensitive data"
  - "HTTP used for sensitive data"
  - "Missing CSRF protection"
  - "'admin/admin' or similar defaults"

example_findings:
  - severity: CRITICAL
    location: "api/users.py:42"
    description: "SQL injection via unsanitized user_id parameter"
    suggestion: "Use parameterized query: cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))"
    cwe: "CWE-89"

  - severity: HIGH
    location: "auth/login.js:15"
    description: "API key hardcoded in source"
    suggestion: "Move to environment variable: process.env.API_KEY"
    cwe: "CWE-798"
```

---

### F8-F12: Layers 3-6 (Condensed)

```yaml
# HL-3: Edge Cases
layer:
  id: HL-3
  name: "Edge Cases"
  veto_power: MEDIUM
  focus: "Boundary conditions, null checks, empty states, race conditions"

checklist:
  - "Null/undefined handling"
  - "Empty array/string handling"
  - "Boundary values (0, -1, MAX_INT)"
  - "Concurrent access handling"
  - "Timeout/retry logic"

---

# HL-4: Accessibility
layer:
  id: HL-4
  name: "Accessibility"
  veto_power: MEDIUM
  focus: "WCAG 2.1 AA compliance, screen readers, keyboard navigation"

checklist:
  - "All images have alt text"
  - "Color contrast >= 4.5:1"
  - "Keyboard navigation works"
  - "Focus indicators visible"
  - "ARIA labels on interactive elements"
  - "Form labels associated"
  - "Heading hierarchy correct"

---

# HL-5: Performance
layer:
  id: HL-5
  name: "Performance"
  veto_power: WEAK  # Warning only
  focus: "Performance anti-patterns, N+1 queries, memory leaks"

checklist:
  - "No N+1 queries"
  - "Large lists are paginated"
  - "Images are optimized"
  - "Memoization where appropriate"
  - "No memory leaks (event listeners)"
  - "Lazy loading for heavy components"

---

# HL-6: Integration
layer:
  id: HL-6
  name: "Integration"
  veto_power: MEDIUM
  focus: "API contracts, breaking changes, backwards compatibility"

checklist:
  - "API changes are backwards compatible"
  - "Error responses follow standard"
  - "Timeouts configured for external calls"
  - "Retries with backoff for transient failures"
  - "Circuit breaker for critical dependencies"
```

---

### F12: HL-7 Final Human Check

**[x] Especificação**

```yaml
layer:
  id: HL-7
  name: "Final Human Check"
  veto_power: STRONG
  focus: "Holistic review, overall quality, production readiness"

system_prompt: |
  You are a senior engineering lead doing final review before production.
  You have access to findings from all previous layers.

  Your job is to:

  1. HOLISTIC ASSESSMENT
     - Does the code achieve its intended purpose?
     - Is the implementation appropriate for the problem?
     - Are there any "code smells" that weren't caught?

  2. PRODUCTION READINESS
     - Is this code ready for real users?
     - Are edge cases handled gracefully?
     - Will this scale?
     - Is monitoring/logging adequate?

  3. SYNTHESIS
     - Considering all layer findings, what's the overall verdict?
     - Are any findings more/less important in context?
     - Any patterns across findings?

  4. SUBJECTIVE QUALITY
     - Would you be comfortable shipping this?
     - Would you be proud of this code?
     - Any "gut feeling" concerns?

  Previous layer findings:
  {previous_layer_findings}

  Provide:
  - Overall verdict (PASS/WARN/FAIL)
  - Summary of key concerns
  - Any additional findings not caught by other layers
  - Recommendation (ship/fix/redesign)

context_required: true  # Needs findings from other layers
runs_last: true        # Always runs after all other layers
```

---

## 4.3 SIX PERSPECTIVES

---

**[x] All Perspectives Specified**

```yaml
perspectives:

  tired_user:
    name: "Tired User"
    description: "End of a long day, low patience, prone to mistakes"
    weight: 1.2  # Slightly higher - catches frustrating UX
    behavioral_patterns:
      - "Clicks rapidly when nothing happens"
      - "Misreads instructions"
      - "Skips help text"
      - "Types wrong format without noticing"
      - "Gives up quickly on confusing interfaces"
    test_focus:
      - "Error message clarity"
      - "Recovery from mistakes"
      - "Visual feedback for actions"
      - "Minimal steps to complete tasks"
    example_scenario: |
      User tries to checkout at 11pm, enters card number with spaces,
      form rejects with "Invalid format" - no indication of what's wrong.
      Frustrated, abandons cart.

  malicious_insider:
    name: "Malicious Insider"
    description: "Disgruntled employee with legitimate access trying to cause harm"
    weight: 1.5  # Higher - security critical
    behavioral_patterns:
      - "Tests authorization boundaries"
      - "Attempts privilege escalation"
      - "Looks for data exfiltration paths"
      - "Tests rate limits"
      - "Probes for hidden APIs"
    test_focus:
      - "Authorization on every action"
      - "Audit logging completeness"
      - "Data access controls"
      - "Rate limiting"
    example_scenario: |
      Employee with viewer access tries to access admin API endpoint
      directly. Should be denied and logged.

  confused_newbie:
    name: "Confused Newbie"
    description: "First-time user, unfamiliar with domain, needs guidance"
    weight: 1.0
    behavioral_patterns:
      - "Doesn't read documentation first"
      - "Expects defaults to work"
      - "Confused by technical jargon"
      - "Tries obvious paths first"
      - "Clicks everything to see what happens"
    test_focus:
      - "Onboarding clarity"
      - "Default behaviors"
      - "Help text quality"
      - "Error prevention"
    example_scenario: |
      New user sees "Configure your LLM provider" without knowing
      what LLM means or why it's needed. Gives up.

  power_user:
    name: "Power User"
    description: "Expert user who wants efficiency, shortcuts, advanced features"
    weight: 1.0
    behavioral_patterns:
      - "Uses keyboard shortcuts"
      - "Wants batch operations"
      - "Needs API access"
      - "Customizes everything"
      - "Pushes limits of system"
    test_focus:
      - "Keyboard accessibility"
      - "Bulk operations"
      - "Advanced configuration"
      - "Performance at scale"
    example_scenario: |
      Power user wants to validate 100 files at once.
      Is there a batch mode? Is there an API?

  auditor:
    name: "Auditor"
    description: "Compliance officer reviewing for regulatory requirements"
    weight: 1.3  # Higher - compliance critical
    behavioral_patterns:
      - "Asks 'where is this logged?'"
      - "Checks data retention policies"
      - "Verifies access controls"
      - "Reviews audit trails"
      - "Tests data export/deletion"
    test_focus:
      - "Audit logging"
      - "Data retention"
      - "Access controls documentation"
      - "GDPR/CCPA compliance"
    example_scenario: |
      Auditor asks: "Show me all access to customer PII in the last 30 days."
      Is this query possible?

  3am_operator:
    name: "3AM Operator"
    description: "On-call engineer debugging production issue at 3am"
    weight: 1.1
    behavioral_patterns:
      - "Needs quick answers"
      - "Can't read walls of text"
      - "Wants clear error messages"
      - "Needs rollback options"
      - "Under pressure, makes mistakes"
    test_focus:
      - "Error message actionability"
      - "Logging clarity"
      - "Rollback mechanisms"
      - "Monitoring dashboards"
    example_scenario: |
      Production alert: "Validation service down."
      Is the error message actionable? Can they restart/rollback quickly?
```

---

## 4.4-4.9 (Condensed for Length)

**Modules specified with interfaces, invariants, and behaviors:**

### F19-F23: Cognitive Modules

```python
# BudgetManager - Tracks token usage and costs
class BudgetManager:
    def track_usage(self, tokens: int, model: str) -> None: ...
    def estimate_cost(self, validation: ValidationTarget) -> CostEstimate: ...
    def check_budget(self, user: User) -> BudgetStatus: ...

# TrustScorer - Tracks user/code trust over time
class TrustScorer:
    def calculate_trust(self, target: ValidationTarget) -> float: ...
    def update_trust(self, outcome: ValidationOutcome) -> None: ...

# TriageEngine - Prioritizes what to validate
class TriageEngine:
    def triage(self, targets: List[ValidationTarget]) -> List[TriagedTarget]: ...

# FeedbackLoop - Learns from user feedback
class FeedbackLoop:
    def record_feedback(self, finding: Finding, helpful: bool) -> None: ...
    def improve_prompts(self) -> None: ...  # Periodic

# ConfidenceCalculator
class ConfidenceCalculator:
    def calculate(self, runs: List[SingleRunResult]) -> float: ...
```

### F24-F29: Browser Automation

```python
# Playwright-based automation for UI validation
class BrowserDriver:
    async def navigate(self, url: str) -> None: ...
    async def screenshot(self, selector: Optional[str] = None) -> bytes: ...
    async def check_accessibility(self) -> List[A11yIssue]: ...

class JourneyExecutor:
    async def execute_journey(self, journey: Journey) -> JourneyResult: ...
```

### F30-F37: LLM Integration

```python
# Abstract interface all LLM clients implement
class LLMClient(ABC):
    @abstractmethod
    async def complete(self, prompt: str, **kwargs) -> str: ...

    @abstractmethod
    def count_tokens(self, text: str) -> int: ...

# Concrete implementations
class ClaudeClient(LLMClient): ...
class OpenAIClient(LLMClient): ...
class GeminiClient(LLMClient): ...
class OllamaClient(LLMClient): ...

# Multi-LLM with fallback
class MultiLLMManager:
    def __init__(self, clients: List[LLMClient], fallback_order: List[str]): ...
    async def complete(self, prompt: str) -> str:
        """Tries clients in order until one succeeds."""
```

### F38-F41: MCP Server

```yaml
mcp_tools:
  - name: validate_code
    description: "Validate code through Human Layer"
    parameters:
      code: string (required)
      layers: array of LayerID (optional)
      perspectives: array of PerspectiveID (optional)
    returns: ValidationReport

  - name: validate_ui
    description: "Validate UI at URL"
    parameters:
      url: string (required)
    returns: ValidationReport

  - name: get_report
    description: "Get validation report by ID"
    parameters:
      report_id: string (required)
    returns: ValidationReport

  - name: list_findings
    description: "List findings with filters"
    parameters:
      severity: array of Severity (optional)
      layer: array of LayerID (optional)
      limit: integer (default 100)
    returns: array of Finding

mcp_resources:
  - uri: human_layer://layers
    description: "List all 7 Human Layers"

  - uri: human_layer://layers/{layer_id}
    description: "Get specific layer details"

  - uri: human_layer://perspectives
    description: "List all 6 perspectives"

  - uri: human_layer://stats
    description: "Get validation statistics"
```

### F42-F46: Data Models

```python
@dataclass
class Finding:
    id: str                          # UUID
    severity: Severity               # CRITICAL|HIGH|MEDIUM|LOW|INFO
    layer_id: LayerID               # Which layer found it
    description: str                 # What's wrong
    suggestion: Optional[str]        # How to fix
    location: Optional[Location]     # File:line
    confidence: float               # 0.0-1.0
    occurrences: int                # How many runs found it
    cwe: Optional[str]              # CWE ID for security
    created_at: datetime

@dataclass
class LayerResult:
    layer_id: LayerID
    status: LayerStatus             # PASS|WARN|FAIL|ERROR|SKIP
    findings: List[Finding]
    run_count: int
    successful_runs: int
    duration_seconds: float
    tokens_used: int

@dataclass
class ValidationReport:
    id: str
    status: ValidationStatus        # PASS|WARN|FAIL|ERROR
    findings: List[Finding]
    layer_results: List[LayerResult]
    summary: Summary
    metadata: Metadata
    created_at: datetime

class VetoLevel(Enum):
    NONE = "none"
    WEAK = "weak"
    MEDIUM = "medium"
    STRONG = "strong"

class Severity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class LayerStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    ERROR = "error"
    SKIP = "skip"
```

### F47-F56: Cloud Features

```yaml
# Cloud-only features specification

user_management:
  registration: "Email or OAuth (Google, GitHub)"
  login: "Email+password or OAuth"
  mfa: "TOTP (Google Authenticator compatible)"
  password_requirements: "8+ chars, 1 uppercase, 1 number"

team_management:
  roles:
    owner: "Full access, billing, can delete team"
    admin: "Full access except billing"
    member: "Can run validations, view history"
    viewer: "Read-only access"

subscription:
  provider: "Stripe"
  webhook_events: ["checkout.session.completed", "invoice.paid", "customer.subscription.updated"]

dashboard:
  realtime: "WebSocket for live updates"
  charts: "Recharts or similar"
  export: "PDF, JSON, CSV"

ci_cd_integrations:
  github_actions:
    type: "GitHub Action"
    repo: "humangr/human-layer-action"
  gitlab_ci:
    type: "Docker image"
  generic:
    type: "CLI in any CI"

webhooks:
  events: ["validation.completed", "finding.critical", "tier.limit.reached"]
  retry: "3x with exponential backoff"
  signature: "HMAC-SHA256"
```

---

# PARTE 5: ARQUITETURA TÉCNICA

> **Decisões de arquitetura são FINAIS**. Mudanças requerem ADR (Architecture Decision Record).

---

## 5.1 OSS ARCHITECTURE

### [x] A1: Package Structure

```
human-layer/
├── src/
│   └── hl_mcp/                      # Main package
│       ├── __init__.py              # Public API exports
│       ├── core/                    # Core engine
│       │   ├── runner.py            # HumanLayerRunner
│       │   ├── orchestrator.py      # LayerOrchestrator
│       │   ├── consensus.py         # ConsensusEngine
│       │   └── veto.py              # VetoGate
│       ├── layers/                  # 7 Human Layers
│       │   ├── base.py              # Layer base class
│       │   ├── hl1_ux.py
│       │   ├── hl2_security.py
│       │   ├── hl3_edge_cases.py
│       │   ├── hl4_accessibility.py
│       │   ├── hl5_performance.py
│       │   ├── hl6_integration.py
│       │   └── hl7_final.py
│       ├── perspectives/            # 6 Perspectives
│       │   ├── base.py
│       │   ├── tired_user.py
│       │   ├── malicious_insider.py
│       │   ├── confused_newbie.py
│       │   ├── power_user.py
│       │   ├── auditor.py
│       │   └── operator_3am.py
│       ├── llm/                     # LLM clients
│       │   ├── base.py              # LLMClient ABC
│       │   ├── claude.py
│       │   ├── openai.py
│       │   ├── gemini.py
│       │   └── ollama.py
│       ├── server/                  # MCP Server
│       │   ├── server.py            # Main server
│       │   ├── tools.py             # MCP Tools
│       │   └── resources.py         # MCP Resources
│       ├── models/                  # Data models
│       │   ├── enums.py             # VetoLevel, Severity, etc.
│       │   ├── findings.py          # Finding
│       │   ├── layers.py            # LayerResult
│       │   └── report.py            # ValidationReport
│       ├── browser/                 # Playwright automation
│       │   ├── driver.py
│       │   └── actions.py
│       ├── cognitive/               # Cognitive modules
│       │   ├── budget.py
│       │   ├── trust.py
│       │   └── triage.py
│       └── cli.py                   # CLI entry point
├── tests/
├── docs/
├── pyproject.toml
├── LICENSE                          # Apache 2.0
└── README.md
```

**Public API (`hl_mcp/__init__.py`)**:
```python
# Core
from hl_mcp.core.runner import HumanLayerRunner
from hl_mcp.core.orchestrator import LayerOrchestrator

# Models
from hl_mcp.models.findings import Finding
from hl_mcp.models.layers import LayerResult
from hl_mcp.models.report import ValidationReport
from hl_mcp.models.enums import VetoLevel, Severity, LayerID, PerspectiveID

# Server
from hl_mcp.server.server import HumanLayerMCPServer

# LLM
from hl_mcp.llm.base import LLMClient

__version__ = "1.0.0"
```

### [x] A2: Configuration

**Config file (`human-layer.yaml`)**:
```yaml
# LLM Configuration (REQUIRED)
llm:
  provider: claude              # claude|openai|gemini|ollama
  model: claude-3-sonnet-20240229
  api_key_env: ANTHROPIC_API_KEY  # Env var name
  timeout: 30                   # seconds
  max_retries: 3

# Validation settings
validation:
  redundancy: 3                 # 1-5, default 3
  consensus_threshold: 0.67    # 0.5-1.0
  timeout: 600                 # total timeout (seconds)
  parallel_layers: true        # Run layers 1-6 in parallel

# Layer configuration
layers:
  enabled: all                  # or list: [security, ux, edge_cases]
  hl2_security:
    veto_power: strong         # Override default
  hl5_performance:
    veto_power: none           # Disable veto

# Perspective configuration
perspectives:
  enabled: all
  weights:
    malicious_insider: 1.5     # Higher weight for security
    auditor: 1.3

# Output
output:
  format: json                 # json|markdown|html
  file: null                   # null = stdout
  verbosity: normal            # quiet|normal|verbose

# Ignore patterns
ignore:
  paths:
    - node_modules
    - .git
    - __pycache__
    - vendor
  patterns:
    - "*.test.*"
    - "*.spec.*"
    - "*_test.py"
```

**Environment variables**:
```bash
# LLM API Keys (choose one)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
OLLAMA_HOST=http://localhost:11434

# Human Layer Cloud (optional)
HUMANLAYER_API_KEY=hl_...
HUMANLAYER_CLOUD_URL=https://api.humangr.ai

# Debug
HUMANLAYER_DEBUG=false
HUMANLAYER_LOG_LEVEL=INFO
```

### [x] A3: Installation

```bash
# Primary: pip
pip install human-layer

# With extras
pip install human-layer[browser]  # Include Playwright
pip install human-layer[all]      # All optional deps

# Docker
docker pull ghcr.io/humangr/human-layer:latest
docker run -v $(pwd):/app ghcr.io/humangr/human-layer validate /app

# Development
git clone https://github.com/humangr/human-layer
cd human-layer
pip install -e ".[dev]"
```

**Dependencies**:
```toml
[project]
dependencies = [
    "pydantic>=2.0",
    "httpx>=0.24",
    "click>=8.0",
    "rich>=13.0",
    "pyyaml>=6.0",
    "jinja2>=3.0",
]

[project.optional-dependencies]
browser = ["playwright>=1.40"]
claude = ["anthropic>=0.25"]
openai = ["openai>=1.0"]
gemini = ["google-generativeai>=0.3"]
all = ["human-layer[browser,claude,openai,gemini]"]
```

### [x] A4: CLI

```bash
# Commands
human-layer --help
human-layer validate <path>       # Validate code
human-layer init                  # Create config file
human-layer doctor                # Check prerequisites
human-layer config                # Show current config
human-layer serve                 # Start MCP server

# Options
human-layer validate . \
  --layers security,ux \          # Specific layers
  --redundancy 3 \                # Override redundancy
  --output report.json \          # Output file
  --format json \                 # Output format
  --timeout 300 \                 # Timeout
  --verbose                       # Verbose output

# Exit codes
0 = PASS (no issues)
1 = WARN (warnings only)
2 = FAIL (issues found)
3 = ERROR (execution error)
10 = CONFIG_ERROR
11 = LLM_ERROR
```

---

## 5.2 CLOUD ARCHITECTURE

### [x] Tech Stack Decision

```yaml
cloud_stack:
  runtime: "Node.js 20 (API) + Python 3.11 (validation workers)"
  framework: "Fastify (API) + FastAPI (internal)"
  database: "PostgreSQL 15 (Supabase)"
  cache: "Redis 7 (Upstash)"
  queue: "BullMQ (Redis-based)"
  auth: "Clerk"
  billing: "Stripe"
  hosting: "Vercel (API) + Fly.io (workers)"
  cdn: "Cloudflare"
  monitoring: "Grafana Cloud"
  error_tracking: "Sentry"
```

### [x] Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              CLOUDFLARE CDN                              │
└─────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           VERCEL EDGE (API)                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   /api/v1   │  │  /webhook   │  │    /auth    │  │  /billing   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────────────────────┘
         │                   │                │                │
         ▼                   ▼                ▼                ▼
┌─────────────────┐  ┌─────────────────┐  ┌───────┐  ┌────────────────┐
│   REDIS QUEUE   │  │   POSTGRESQL    │  │ CLERK │  │     STRIPE     │
│    (BullMQ)     │  │   (Supabase)    │  │ (SSO) │  │   (Billing)    │
└─────────────────┘  └─────────────────┘  └───────┘  └────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        FLY.IO WORKERS (Python)                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │   Validation Worker Pool (auto-scaling 2-20 instances)          │    │
│  │                                                                  │    │
│  │   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │    │
│  │   │  Worker 1   │ │  Worker 2   │ │  Worker N   │               │    │
│  │   │ (HumanLayer)│ │ (HumanLayer)│ │ (HumanLayer)│               │    │
│  │   └─────────────┘ └─────────────┘ └─────────────┘               │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │   Playwright Farm (for UI validation)                           │    │
│  │   - Browserless.io or self-hosted Playwright containers         │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

### [x] Database Schema (Core Tables)

```sql
-- Users (via Clerk, minimal local storage)
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    clerk_id VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) NOT NULL,
    name VARCHAR(255),
    tier VARCHAR(50) DEFAULT 'free',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Teams
CREATE TABLE teams (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    owner_id UUID REFERENCES users(id),
    tier VARCHAR(50) DEFAULT 'team',
    created_at TIMESTAMP DEFAULT NOW()
);

-- Team memberships
CREATE TABLE team_members (
    team_id UUID REFERENCES teams(id),
    user_id UUID REFERENCES users(id),
    role VARCHAR(50) DEFAULT 'member',
    PRIMARY KEY (team_id, user_id)
);

-- Validations
CREATE TABLE validations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    team_id UUID REFERENCES teams(id),
    status VARCHAR(50) NOT NULL,
    target_type VARCHAR(50) NOT NULL,
    target_hash VARCHAR(64),
    config JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP
);

-- Findings
CREATE TABLE findings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    validation_id UUID REFERENCES validations(id),
    severity VARCHAR(50) NOT NULL,
    layer_id VARCHAR(50) NOT NULL,
    description TEXT NOT NULL,
    suggestion TEXT,
    location JSONB,
    confidence FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- API Keys
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    team_id UUID REFERENCES teams(id),
    key_hash VARCHAR(64) NOT NULL,
    name VARCHAR(255),
    last_used TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Usage tracking
CREATE TABLE usage (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    team_id UUID REFERENCES teams(id),
    period_start DATE NOT NULL,
    validations_count INT DEFAULT 0,
    tokens_used BIGINT DEFAULT 0
);

-- Indexes
CREATE INDEX idx_validations_user ON validations(user_id);
CREATE INDEX idx_validations_team ON validations(team_id);
CREATE INDEX idx_validations_created ON validations(created_at);
CREATE INDEX idx_findings_validation ON findings(validation_id);
CREATE INDEX idx_findings_severity ON findings(severity);
CREATE INDEX idx_usage_period ON usage(user_id, period_start);
```

---

## 5.3 SECURITY ARCHITECTURE

### [x] Data Flow Security

```
USER CODE → Never leaves user's environment
         ↓
         ↓ (only code hash sent if dedup enabled)
         ↓
LLM API  → Uses user's own API key (BYOK)
         ↓
         ↓ (LLM provider sees code temporarily)
         ↓
RESPONSE → Processed locally or in isolated worker
         ↓
         ↓ (only findings stored, not original code)
         ↓
STORAGE  → Findings only, encrypted at rest
```

### [x] Security Controls

| Control | Implementation |
|---------|----------------|
| **Authentication** | Clerk (OAuth2/OIDC) |
| **API Keys** | SHA-256 hashed, prefix visible (`hl_...abc`) |
| **Encryption at rest** | AES-256 (Supabase default) |
| **Encryption in transit** | TLS 1.3 everywhere |
| **Secrets** | Environment variables, Vercel/Fly.io encrypted |
| **RBAC** | Team roles: owner, admin, member, viewer |
| **Audit log** | All sensitive actions logged |
| **Rate limiting** | Per-user, per-tier limits |

---

## 5.4 OSS vs CLOUD SEPARATION

### [x] Code Separation Strategy

```python
# human-layer (OSS repo)
# Contains: All core functionality

# human-layer-cloud (Private repo)
# Contains: Cloud-specific features

# Import pattern in cloud:
from hl_mcp.core import HumanLayerRunner  # From OSS
from hl_cloud.api import create_app       # Cloud-only

# Feature flags for tier enforcement
TIER_LIMITS = {
    "free": {"validations_per_month": 20, "history_days": 7},
    "solo": {"validations_per_month": 200, "history_days": 30},
    # ...
}
```

---

# PARTE 6: INTEGRAÇÕES

> **Todas as integrações seguem padrão**: Setup guide + Code example + Error handling

---

## 6.1 LLM PROVIDERS

### [x] Provider Matrix

| Provider | Models | Recommended | Token Limit | Cost/1K tokens |
|----------|--------|-------------|-------------|----------------|
| **Claude** | claude-3-opus, claude-3-sonnet, claude-3-haiku | sonnet (balance) | 200K | $0.003-$0.015 |
| **OpenAI** | gpt-4-turbo, gpt-4o, gpt-3.5-turbo | gpt-4o | 128K | $0.005-$0.030 |
| **Gemini** | gemini-1.5-pro, gemini-1.5-flash | 1.5-pro | 1M | $0.001-$0.007 |
| **Ollama** | llama3, mistral, codellama | llama3:70b | 8K-32K | $0 (local) |

### [x] Integration Code (Claude Example)

```python
from hl_mcp.llm import ClaudeClient

client = ClaudeClient(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    model="claude-3-sonnet-20240229",
    timeout=30,
    max_retries=3,
)

# In config:
llm:
  provider: claude
  model: claude-3-sonnet-20240229
  api_key_env: ANTHROPIC_API_KEY
```

---

## 6.2 CI/CD PLATFORMS

### [x] GitHub Actions

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

      - uses: humangr/human-layer-action@v1
        with:
          api-key: ${{ secrets.HUMANLAYER_API_KEY }}
          llm-provider: claude
          llm-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
          layers: security,ux,edge-cases
          fail-on: critical,high
```

### [x] GitLab CI

```yaml
# .gitlab-ci.yml
human-layer:
  image: ghcr.io/humangr/human-layer:latest
  script:
    - human-layer validate . --output report.json
  artifacts:
    reports:
      junit: report.xml
```

---

## 6.3 NOTIFICATIONS

### [x] Slack Integration

```yaml
# Webhook payload
{
  "event": "validation.completed",
  "validation_id": "abc123",
  "status": "warn",
  "findings": {
    "critical": 0,
    "high": 2,
    "medium": 3,
    "low": 5
  },
  "url": "https://app.humangr.ai/validations/abc123"
}

# Slack message format
{
  "blocks": [
    {
      "type": "section",
      "text": {
        "type": "mrkdwn",
        "text": "⚠️ *Human Layer Validation Complete*\n2 high, 3 medium issues found"
      }
    }
  ]
}
```

---

# PARTE 7: OPERAÇÕES

---

## 7.1 DEPLOYMENT

### [x] Release Process

```bash
# OSS Release
1. Update version in pyproject.toml
2. Update CHANGELOG.md
3. Create git tag: git tag v1.2.3
4. Push tag: git push origin v1.2.3
5. GitHub Actions publishes to PyPI and ghcr.io

# Cloud Deployment
1. Push to main → Deploy to staging (Vercel preview)
2. Run E2E tests against staging
3. Merge to production branch
4. Blue/green deployment on Fly.io
5. Health checks pass → traffic switch
6. Monitor for 15 minutes
7. Rollback if issues
```

---

## 7.2 MONITORING

### [x] Key Metrics

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| API latency (p99) | <500ms | >1000ms |
| Validation time (p99) | <5min | >10min |
| Error rate | <0.1% | >1% |
| Uptime | 99.9% | <99.5% |
| Worker queue depth | <100 | >500 |

### [x] Dashboards

```
Grafana Dashboard:
├── Overview (traffic, errors, latency)
├── Validations (count, duration, status)
├── Workers (queue depth, processing time)
├── Business (signups, upgrades, revenue)
└── Alerts (active, acknowledged, resolved)
```

---

# PARTE 8: SUPORTE & DOCUMENTAÇÃO

---

## 8.1 DOCUMENTATION STRUCTURE

### [x] Docs Site (docs.humangr.ai)

```
├── Getting Started
│   ├── Quick Start (5 min)
│   ├── Installation
│   ├── First Validation
│   └── Configuration
├── Concepts
│   ├── 7 Human Layers
│   ├── 6 Perspectives
│   ├── Triple Redundancy
│   ├── Consensus & Veto
│   └── BYOK Model
├── Guides
│   ├── CI/CD Integration
│   ├── Team Setup
│   ├── Custom Configuration
│   └── Troubleshooting
├── Reference
│   ├── CLI Reference
│   ├── Configuration Options
│   ├── MCP Tools & Resources
│   ├── REST API (Cloud)
│   └── Error Codes
├── Cloud
│   ├── Dashboard
│   ├── Team Management
│   ├── Billing
│   └── API Access
└── Community
    ├── Contributing
    ├── Code of Conduct
    └── Changelog
```

---

## 8.2 SUPPORT TIERS

| Tier | Channel | Response SLA | Resolution SLA |
|------|---------|--------------|----------------|
| OSS/Free | GitHub Issues, Discord | Community | Community |
| Solo | Email | 48h | 5 days |
| Pro | Email | 24h | 3 days |
| Team | Email | 24h | 2 days |
| Business | Email + Slack | 8h | 1 day |
| Enterprise | Dedicated + Phone | 4h | 4h (critical) |

---

# PARTE 9: MÉTRICAS & KPIs

---

## 9.1 PRODUCT METRICS

### [x] North Star Metric

**"Validations that caught a real issue"** = Validations with ≥1 finding marked as "helpful" by user

### [x] Key Metrics Dashboard

| Metric | Definition | Target (Y1) |
|--------|------------|-------------|
| **DAU** | Users who ran ≥1 validation | 500 |
| **WAU** | Weekly active users | 1,500 |
| **Activation rate** | Signup → First validation | 60% |
| **Retention D7** | Return after 7 days | 40% |
| **Retention D30** | Return after 30 days | 25% |
| **NPS** | Net Promoter Score | 50+ |

---

## 9.2 BUSINESS METRICS

### [x] Revenue Metrics

| Metric | Definition | Target (Y1 Q4) |
|--------|------------|----------------|
| **MRR** | Monthly Recurring Revenue | $20,000 |
| **ARR** | Annual Recurring Revenue | $240,000 |
| **ARPU** | Avg Revenue Per User | $50 |
| **LTV** | Lifetime Value | $600 |
| **CAC** | Customer Acquisition Cost | $20 |
| **LTV:CAC** | Ratio | 30:1 |

---

# PARTE 10: COMPLIANCE & LEGAL

---

## 10.1 PRIVACY

### [x] GDPR Compliance

| Requirement | Implementation |
|-------------|----------------|
| Lawful basis | Consent (signup) + Contract (service) |
| Data minimization | Only findings stored, not code |
| Right to access | Dashboard export |
| Right to deletion | Account deletion flow |
| Data portability | JSON export |
| DPA | Available for Business+ |

### [x] Data Handling

```
USER CODE:
- OSS: Never leaves user machine
- Cloud: Sent to user's LLM provider, not stored by us

FINDINGS:
- Stored encrypted (AES-256)
- Retention per tier (7d-1yr)
- Deletable on request

PII:
- Email, name only
- No tracking pixels in emails
- No third-party analytics (self-hosted)
```

---

## 10.2 LEGAL DOCUMENTS

| Document | Location | Last Updated |
|----------|----------|--------------|
| Terms of Service | humangr.ai/legal/terms | 2026-02-01 |
| Privacy Policy | humangr.ai/legal/privacy | 2026-02-01 |
| DPA | humangr.ai/legal/dpa | 2026-02-01 |
| OSS License | github.com/.../LICENSE | Apache 2.0 |
| SLA | humangr.ai/legal/sla | Business+ only |

---

# PARTE 11: GO-TO-MARKET

---

## 11.1 LAUNCH TIMELINE

```
2026 Q1:
├── Week 1-2: Private alpha (10 users)
├── Week 3-4: Fix critical bugs
├── Week 5-6: Public OSS launch (GitHub)
├── Week 7-8: Hacker News, Reddit
├── Week 9-10: Cloud beta (invite-only)
├── Week 11-12: Cloud GA

2026 Q2:
├── Product Hunt launch
├── First paid customers
├── CI/CD integrations
├── Team features

2026 Q3-Q4:
├── Enterprise features
├── SOC2 Type 1 prep
├── Series A conversations (optional)
```

---

## 11.2 MARKETING CHANNELS

| Channel | Effort | Expected Impact | Priority |
|---------|--------|-----------------|----------|
| GitHub + OSS | High | High (trust, adoption) | P0 |
| Hacker News | Medium | High (dev audience) | P0 |
| Twitter/X | Medium | Medium (ongoing) | P1 |
| Dev blogs/tutorials | High | High (SEO, education) | P1 |
| Product Hunt | Low | Medium (spike) | P1 |
| Conferences | High | Medium (brand) | P2 |
| Paid ads | Low | Low (dev audience resistant) | P3 |

---

## 11.3 LAUNCH CHECKLIST

### [x] OSS Launch

```
Pre-launch:
- [ ] README polished
- [ ] Documentation complete
- [ ] 10+ GitHub stars (from alpha users)
- [ ] Example project with known issues
- [ ] Contributing guide
- [ ] Code of conduct
- [ ] License file (Apache 2.0)

Launch day:
- [ ] HN post prepared ("Show HN: Human Layer - Validate AI code with 7 layers")
- [ ] Reddit posts (r/programming, r/Python)
- [ ] Twitter thread
- [ ] Discord server ready

Post-launch:
- [ ] Respond to all comments within 4h
- [ ] Fix critical bugs within 24h
- [ ] Thank early adopters
- [ ] Collect feedback
```

### [x] Cloud GA Launch

```
Pre-launch:
- [ ] Pricing page live
- [ ] Stripe integration tested
- [ ] Onboarding wizard polished
- [ ] Support email monitored
- [ ] Status page live

Launch day:
- [ ] Email to beta users
- [ ] Blog post announcement
- [ ] Product Hunt ship
- [ ] Twitter announcement

Post-launch:
- [ ] Monitor conversion funnel
- [ ] Support queue manageable
- [ ] Iterate based on feedback
```

### G5: Content
- [ ] Blog posts
- [ ] Tutorials
- [ ] Case studies
- [ ] Videos

### G6: Community
- [ ] Discord/Slack
- [ ] GitHub Discussions
- [ ] Meetups
- [ ] Conferences

## 11.3 SALES (Enterprise)

### G7: Sales Process
- [ ] Lead qualification
- [ ] Discovery
- [ ] Demo
- [ ] Proposal
- [ ] Close

### G8: Sales Materials
- [ ] Pitch deck
- [ ] One-pager
- [ ] ROI calculator
- [ ] Security questionnaire

---

# PARTE 12: FRONTEND & UI

> **Tech Stack**: Next.js 14 (App Router) + Tailwind CSS + shadcn/ui + Framer Motion
> **Design System**: Custom design tokens exported to Figma + Code
> **Status**: ✅ ESPECIFICADO

---

## 12.1 DESIGN SYSTEM

### [x] UI1: Brand Identity

```yaml
logo:
  primary: "HumanGR" wordmark + shield icon
  variations:
    - Full (wordmark + icon)
    - Icon only (shield with "HL" letters)
    - Dark background variant
    - Light background variant
  formats: [SVG, PNG @1x @2x @3x]

colors:
  primary: "#3B82F6"      # Blue - trust, technology
  secondary: "#10B981"    # Green - success, safety
  accent: "#8B5CF6"       # Purple - AI, intelligence

  semantic:
    success: "#22C55E"
    warning: "#F59E0B"
    error: "#EF4444"
    info: "#3B82F6"

  severity:
    critical: "#DC2626"
    high: "#EA580C"
    medium: "#CA8A04"
    low: "#16A34A"
    info: "#0EA5E9"

typography:
  font_sans: "Inter" (Google Fonts)
  font_mono: "JetBrains Mono" (code)
  scale: [12, 14, 16, 18, 20, 24, 30, 36, 48, 60]

icons: "Lucide Icons" (open source, consistent)

voice_tone:
  - Professional but approachable
  - Technical without jargon
  - Confident but not arrogant
  - Helpful, not condescending
```

### [x] UI2: Design Tokens

```css
/* CSS Variables (via Tailwind config) */
:root {
  /* Colors */
  --color-primary: 59 130 246;
  --color-secondary: 16 185 129;

  /* Spacing (4px base) */
  --spacing-1: 0.25rem;  /* 4px */
  --spacing-2: 0.5rem;   /* 8px */
  --spacing-3: 0.75rem;  /* 12px */
  --spacing-4: 1rem;     /* 16px */
  --spacing-6: 1.5rem;   /* 24px */
  --spacing-8: 2rem;     /* 32px */

  /* Border radius */
  --radius-sm: 0.25rem;
  --radius-md: 0.5rem;
  --radius-lg: 0.75rem;
  --radius-full: 9999px;

  /* Shadows */
  --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
  --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1);
  --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1);

  /* Transitions */
  --transition-fast: 150ms ease;
  --transition-normal: 200ms ease;
  --transition-slow: 300ms ease;
}

/* Dark mode */
.dark {
  --color-bg: 10 10 15;
  --color-text: 240 240 245;
  --color-border: 42 42 58;
}

/* Breakpoints */
--breakpoint-sm: 640px;
--breakpoint-md: 768px;
--breakpoint-lg: 1024px;
--breakpoint-xl: 1280px;
```

### [x] UI3: Component Library

**Base**: shadcn/ui (Radix primitives + Tailwind)

| Component | Source | Customization |
|-----------|--------|---------------|
| Button | shadcn/ui | Custom variants: primary, secondary, ghost, destructive |
| Input | shadcn/ui | + validation states |
| Select | shadcn/ui | Standard |
| Card | shadcn/ui | + severity variants for findings |
| Modal/Dialog | shadcn/ui | Standard |
| Toast | shadcn/ui (Sonner) | Positioned bottom-right |
| Table | shadcn/ui | + sorting, filtering |
| Tabs | shadcn/ui | Standard |
| Tooltip | shadcn/ui | Standard |
| Badge | shadcn/ui | + severity colors |
| Progress | shadcn/ui | + animated |
| Skeleton | shadcn/ui | Standard |
| **Custom: FindingCard** | Custom | Severity, layer, expandable |
| **Custom: LayerNode** | Custom | For cockpit visualization |
| **Custom: ValidationTimeline** | Custom | Steps with status |

### [x] UI4: Layout System

```tsx
// Tailwind grid + custom layouts
const layouts = {
  marketing: "max-w-7xl mx-auto px-4 sm:px-6 lg:px-8",
  dashboard: "flex min-h-screen",
  sidebar: "w-64 shrink-0",
  main: "flex-1 p-6",
};

// Breakpoints (Tailwind default)
// sm: 640px, md: 768px, lg: 1024px, xl: 1280px, 2xl: 1536px
```

## 12.2 LANDING PAGE & MARKETING

### UI5: Homepage
- [ ] Hero section (headline, subhead, CTA)
- [ ] Problem statement
- [ ] Solution overview
- [ ] Features showcase (7 layers visual)
- [ ] How it works (3-step)
- [ ] Use cases / personas
- [ ] Social proof (logos, testimonials)
- [ ] Pricing preview
- [ ] FAQ
- [ ] CTA section
- [ ] Footer

### UI6: Pricing Page
- [ ] Plan comparison table
- [ ] Feature matrix
- [ ] Toggle annual/monthly
- [ ] FAQ
- [ ] Enterprise CTA
- [ ] Money-back guarantee
- [ ] Trust badges

### UI7: About Page
- [ ] Company story
- [ ] Mission/Vision
- [ ] Team (optional)
- [ ] Values

### UI8: Blog
- [ ] Blog listing
- [ ] Blog post template
- [ ] Categories/Tags
- [ ] Search
- [ ] Related posts
- [ ] Newsletter signup

### UI9: Changelog
- [ ] Version history
- [ ] Release notes template
- [ ] Subscribe to updates

## 12.3 DOCS SITE

### UI10: Docs Structure
- [ ] Navigation (sidebar)
- [ ] Search (full-text)
- [ ] Version selector
- [ ] Language selector (i18n)
- [ ] Breadcrumbs
- [ ] Table of contents (in-page)
- [ ] Previous/Next navigation
- [ ] Edit on GitHub link

### UI11: Docs Content Types
- [ ] Getting started guide
- [ ] Conceptual docs
- [ ] How-to guides
- [ ] API reference
- [ ] Tutorials
- [ ] Examples/Recipes
- [ ] Troubleshooting
- [ ] Glossary

### UI12: Docs Components
- [ ] Code blocks (syntax highlight, copy)
- [ ] Tabs (for OS/language variants)
- [ ] Callouts (info, warning, danger)
- [ ] Expandable sections
- [ ] Embedded videos
- [ ] Interactive examples
- [ ] API playground

## 12.4 AUTH FLOWS

### UI13: Signup Flow
- [ ] Signup form (email)
- [ ] OAuth buttons (Google, GitHub)
- [ ] Password requirements indicator
- [ ] Terms/Privacy checkbox
- [ ] Email verification screen
- [ ] Verification success
- [ ] Error states

### UI14: Login Flow
- [ ] Login form
- [ ] OAuth buttons
- [ ] Remember me
- [ ] Forgot password link
- [ ] Error states (invalid credentials, locked, etc.)
- [ ] MFA screen (if enabled)

### UI15: Password Reset
- [ ] Request reset form
- [ ] Email sent confirmation
- [ ] Reset form (new password)
- [ ] Success confirmation
- [ ] Link expired state

### UI16: OAuth Flows
- [ ] Provider selection
- [ ] Redirect handling
- [ ] Account linking
- [ ] Error handling

## 12.5 DASHBOARD

### UI17: Dashboard Overview
- [ ] Summary cards (validations, findings, trends)
- [ ] Recent validations list
- [ ] Quick actions
- [ ] Alerts/Notifications
- [ ] Usage meter (tier limits)

### UI18: Validations List
- [ ] List/Grid view toggle
- [ ] Filters (status, date, severity)
- [ ] Search
- [ ] Sorting
- [ ] Pagination
- [ ] Bulk actions
- [ ] Quick preview

### UI19: Validation Detail
- [ ] Header (title, status, date)
- [ ] Summary section
- [ ] Findings list (by layer)
- [ ] Finding detail (severity, description, suggestion)
- [ ] Code snippets
- [ ] Screenshots (if UI validation)
- [ ] Actions (re-run, export, share)
- [ ] History/Timeline

### UI20: Reports
- [ ] Report list
- [ ] Report detail view
- [ ] Export options (PDF, JSON, CSV)
- [ ] Share link generation
- [ ] Print-friendly view

### UI21: Analytics/Trends
- [ ] Time series charts
- [ ] Findings by severity
- [ ] Findings by layer
- [ ] Findings by perspective
- [ ] Comparison (week over week)
- [ ] Custom date range

## 12.6 SETTINGS

### UI22: Account Settings
- [ ] Profile (name, email, avatar)
- [ ] Password change
- [ ] Email change (with verification)
- [ ] MFA setup
- [ ] Connected accounts (OAuth)
- [ ] Delete account

### UI23: Team Settings (Team+ tiers)
- [ ] Team profile (name, logo)
- [ ] Members list
- [ ] Invite members
- [ ] Pending invites
- [ ] Role management
- [ ] Remove members
- [ ] Transfer ownership
- [ ] Delete team

### UI24: Billing Settings
- [ ] Current plan
- [ ] Usage this period
- [ ] Upgrade/Downgrade buttons
- [ ] Payment method
- [ ] Billing history
- [ ] Invoices download
- [ ] Cancel subscription

### UI25: Integration Settings
- [ ] LLM configuration
- [ ] CI/CD connections
- [ ] Webhooks management
- [ ] API keys management
- [ ] SSO configuration (Business+)

### UI26: Notification Settings
- [ ] Email preferences
- [ ] In-app preferences
- [ ] Slack integration
- [ ] Per-event toggles

### UI27: Appearance Settings
- [ ] Theme (light/dark/system)
- [ ] Compact mode
- [ ] Language (i18n)
- [ ] Timezone

## 12.7 ONBOARDING UI

### UI28: Welcome Wizard
- [ ] Welcome screen
- [ ] Step indicator
- [ ] LLM setup step
- [ ] First validation step
- [ ] Dashboard tour
- [ ] Completion celebration
- [ ] Skip option

### UI29: Feature Tours
- [ ] Tooltip tours
- [ ] Highlight animations
- [ ] Progress tracking
- [ ] Dismiss/Complete

### UI30: Contextual Help
- [ ] Inline help icons
- [ ] Help panel/drawer
- [ ] Video embeds
- [ ] Link to docs

## 12.8 ERROR & EMPTY STATES

### UI31: Error Pages
- [ ] 404 Not Found
- [ ] 500 Server Error
- [ ] 403 Forbidden
- [ ] 429 Rate Limited
- [ ] Maintenance mode
- [ ] Network error

### UI32: Empty States
- [ ] No validations yet
- [ ] No findings (success!)
- [ ] No team members
- [ ] No webhooks
- [ ] Search no results
- [ ] Filter no results

### UI33: Loading States
- [ ] Page loading
- [ ] Component loading (skeletons)
- [ ] Button loading
- [ ] Table loading
- [ ] Infinite scroll loading

### UI34: Form Errors
- [ ] Field-level errors
- [ ] Form-level errors
- [ ] Server errors
- [ ] Validation feedback (real-time)

## 12.9 NOTIFICATIONS UI

### UI35: Toast Notifications
- [ ] Success toast
- [ ] Error toast
- [ ] Warning toast
- [ ] Info toast
- [ ] Action toast (with button)
- [ ] Persistent vs auto-dismiss
- [ ] Stacking behavior

### UI36: In-App Notifications
- [ ] Notification bell/icon
- [ ] Notification dropdown
- [ ] Notification center (full page)
- [ ] Mark as read
- [ ] Mark all as read
- [ ] Notification preferences link

### UI37: Banners
- [ ] Info banner
- [ ] Warning banner
- [ ] Upgrade prompt banner
- [ ] Maintenance banner
- [ ] Dismissible vs persistent

## 12.10 CLI UI

### UI38: Terminal Output
- [ ] Color scheme
- [ ] Progress bars
- [ ] Spinners
- [ ] Tables (ASCII)
- [ ] Tree views
- [ ] Diff output
- [ ] Syntax highlighting

### UI39: CLI Interactive
- [ ] Prompts
- [ ] Confirmations
- [ ] Multi-select
- [ ] Autocomplete
- [ ] Help formatting

## 12.11 EMAIL TEMPLATES

### UI40: Transactional Emails
- [ ] Welcome email
- [ ] Email verification
- [ ] Password reset
- [ ] Team invitation
- [ ] Validation complete
- [ ] Weekly summary
- [ ] Limit warning
- [ ] Payment confirmation
- [ ] Payment failed
- [ ] Subscription changed
- [ ] Account deletion

### UI41: Marketing Emails
- [ ] Newsletter template
- [ ] Product update
- [ ] Feature announcement
- [ ] Re-engagement

### UI42: Email Design System
- [ ] Header/Footer
- [ ] Button styles
- [ ] Typography
- [ ] Mobile responsiveness
- [ ] Dark mode support

## 12.12 MOBILE & RESPONSIVE

### UI43: Responsive Behavior
- [ ] Mobile navigation (hamburger)
- [ ] Touch targets (min 44px)
- [ ] Swipe gestures
- [ ] Collapsible sections
- [ ] Stacked layouts
- [ ] Font scaling

### UI44: Mobile-Specific
- [ ] App-like feel
- [ ] Pull to refresh
- [ ] Bottom navigation option
- [ ] Reduced data views

## 12.13 ACCESSIBILITY (A11Y)

### UI45: WCAG Compliance
- [ ] Color contrast (AA minimum)
- [ ] Keyboard navigation
- [ ] Focus indicators
- [ ] Screen reader support
- [ ] ARIA labels
- [ ] Alt text for images
- [ ] Form labels
- [ ] Error announcements
- [ ] Skip links
- [ ] Heading hierarchy

### UI46: Accessibility Testing
- [ ] Automated testing (axe)
- [ ] Manual testing checklist
- [ ] Screen reader testing
- [ ] Keyboard-only testing

## 12.14 ANIMATIONS & MICRO-INTERACTIONS

### UI47: Transitions
- [ ] Page transitions
- [ ] Component enter/exit
- [ ] Loading transitions
- [ ] Hover effects
- [ ] Focus effects

### UI48: Micro-interactions
- [ ] Button feedback
- [ ] Form validation feedback
- [ ] Success celebrations
- [ ] Error shakes
- [ ] Pull to refresh
- [ ] Skeleton to content

### UI49: Performance
- [ ] Animation performance (GPU)
- [ ] Reduced motion support
- [ ] Lazy loading
- [ ] Optimistic UI

## 12.15 STATE MANAGEMENT (State of the Art)

### UI50: State Architecture
```typescript
// Modern state management stack
stack:
  global: "Zustand" (lightweight, TypeScript-first)
  server: "TanStack Query v5" (server state, caching)
  forms: "React Hook Form + Zod" (validation)
  url: "nuqs" (URL state sync)

patterns:
  - Colocation: State próximo de onde é usado
  - Derived state: Computed ao invés de duplicado
  - Optimistic updates: UI imediata, rollback em erro
  - Stale-while-revalidate: Cache + background refresh
```

### UI51: State Patterns
- [ ] **Optimistic UI**: Ações imediatas, rollback em falha
- [ ] **Pessimistic UI**: Loading states para ações destrutivas
- [ ] **Background sync**: Offline queue + sync quando online
- [ ] **Real-time sync**: WebSocket state subscription
- [ ] **URL as state**: Compartilhável, bookmarkable
- [ ] **Form state isolation**: Não polui estado global

### UI52: Caching Strategy
```yaml
cache_strategy:
  api_responses:
    default_stale_time: 60s
    default_cache_time: 5m
    invalidation: "smart" (mutation-based)

  static_assets:
    strategy: "cache-first"
    versioning: "content-hash"

  user_preferences:
    storage: "localStorage"
    sync: "on-mount"
```

## 12.16 TESTING UI (Comprehensive)

### UI53: Testing Stack
```yaml
testing_stack:
  unit:
    tool: "Vitest"
    coverage: ">= 80%"

  component:
    tool: "Testing Library"
    approach: "user-centric"

  integration:
    tool: "Playwright"
    coverage: "critical paths"

  visual:
    tool: "Chromatic (Storybook)"
    baseline: "per-branch"

  accessibility:
    tool: "axe-core + pa11y"
    ci: "blocking"
```

### UI54: Component Testing Strategy
- [ ] **Isolation tests**: Cada componente testável sozinho
- [ ] **Interaction tests**: user-event para simular usuário
- [ ] **Snapshot tests**: Apenas para regressão visual
- [ ] **A11y tests**: axe checks em cada componente
- [ ] **Responsive tests**: Viewports críticos

### UI55: E2E Testing
- [ ] **Critical paths**: Login → Dashboard → Validation → Report
- [ ] **Error paths**: Network failures, 500s, timeouts
- [ ] **Edge cases**: Empty states, limits, permissions
- [ ] **Cross-browser**: Chrome, Firefox, Safari
- [ ] **Mobile**: iOS Safari, Android Chrome

## 12.17 DESIGN SYSTEM AUTOMATION

### UI56: Design-to-Code Pipeline
```yaml
design_to_code:
  source: "Figma"

  sync:
    tokens:
      tool: "Figma Tokens → Style Dictionary"
      output: [CSS, Tailwind, TypeScript]
      ci: "auto-PR on token change"

    components:
      tool: "Figma → Storybook comparison"
      drift_detection: "weekly"

    icons:
      tool: "Figma → SVG → React components"
      optimization: "SVGO"
```

### UI57: Component Documentation
- [ ] **Storybook**: Documentação viva de componentes
- [ ] **Props table**: Auto-gerada de TypeScript
- [ ] **Usage examples**: Código copiável
- [ ] **Do/Don't**: Guidelines visuais
- [ ] **Changelog**: Versão por componente

### UI58: Design System Metrics
- [ ] Adoption rate (% de uso vs custom)
- [ ] Component coverage
- [ ] Consistency score
- [ ] A11y compliance %
- [ ] Performance budget per component

## 12.18 AI-ENHANCED UI

### UI59: AI-Powered Features
```yaml
ai_features:
  smart_search:
    description: "Busca semântica em validations/findings"
    tech: "Embeddings + vector search"

  auto_categorization:
    description: "Categoriza findings automaticamente"
    tech: "Classification model"

  trend_detection:
    description: "Detecta padrões em findings"
    tech: "Time-series analysis"

  natural_language_queries:
    description: "Perguntas em linguagem natural"
    example: "Show me all critical findings from last week"
    tech: "NL2SQL or RAG"
```

### UI60: Smart Assistance
- [ ] **Autocomplete**: Sugestões contextuais
- [ ] **Smart defaults**: Baseados em histórico
- [ ] **Error prediction**: "This might fail because..."
- [ ] **Recommendation engine**: "Teams like yours usually..."
- [ ] **Anomaly detection**: Alertas proativos

## 12.19 PROGRESSIVE ENHANCEMENT

### UI61: Progressive Enhancement Strategy
```yaml
progressive_enhancement:
  baseline:
    - HTML semântico
    - CSS básico (sem JS)
    - Formulários funcionais

  enhanced:
    - JavaScript interatividade
    - Animações
    - Real-time updates

  premium:
    - WebGL visualizations
    - Service worker (offline)
    - Push notifications

  graceful_degradation:
    no_js: "Core functionality works"
    slow_connection: "Skeleton → Content"
    old_browser: "Warning banner + basic mode"
```

### UI62: Offline Capabilities
- [ ] Service worker registration
- [ ] Asset caching (app shell)
- [ ] API response caching
- [ ] Background sync queue
- [ ] Offline indicator UI
- [ ] Conflict resolution

---

# PARTE 14: COCKPIT VISUAL (Real-Time Dashboard)

> **Referência**: Template do Pipeline HumanGR (`cockpit/templates/cockpit.html`)
> **Tema Piloto**: SPOVEST (mesmo tema do pipeline original)

## 14.1 ARQUITETURA DO COCKPIT

### CK1: Estrutura Geral
- [ ] Header com logo, status, métricas principais
- [ ] Hero metrics row (sprint, progress, agents, tokens, tasks, warnings)
- [ ] Flow panel principal (nodes conectados)
- [ ] Node popup overlay (detalhes + logs)
- [ ] Summary panel (totais)
- [ ] Pipeline output panel (logs gerais)

### CK2: Conexão Real-Time
- [ ] WebSocket (Socket.IO)
- [ ] Connection status indicator
- [ ] Reconnection logic
- [ ] Heartbeat/ping
- [ ] Event-driven updates

## 14.2 FLOW VISUALIZATION (Nodes)

### CK3: Node Structure
- [ ] Node icon (SVG)
- [ ] Node label
- [ ] Node status indicator (glow effect)
- [ ] Click handler para popup
- [ ] Hover effects
- [ ] Pulse animation quando ativo

### CK4: Node States
```css
/* Estados visuais dos nodes */
- pending:  cinza, sem brilho
- active:   azul pulsando, brilho forte
- complete: verde, brilho suave
- warning:  amarelo, brilho intermitente
- error:    vermelho, brilho forte
- skipped:  cinza com opacity reduzida
```

### CK5: Node Types (Human Layer Specific)
- [ ] **Input Node**: Artefato recebido para validação
- [ ] **Layer Nodes**: HL-1 a HL-7 (7 nodes)
- [ ] **Perspective Nodes**: 6 perspectivas
- [ ] **Redundancy Nodes**: 3 execuções por layer
- [ ] **Consensus Node**: Consolidação 2/3
- [ ] **Veto Gate Node**: Decisão final
- [ ] **Output Node**: Resultado/Report

### CK6: Connections (Edges)
- [ ] SVG paths entre nodes
- [ ] Animated flow direction
- [ ] Color based on status
- [ ] Thickness based on importance
- [ ] Glow effect quando dados fluindo

## 14.3 NODE POPUP (Detail View)

### CK7: Popup Header
- [ ] Node title
- [ ] Status indicator
- [ ] Close button
- [ ] Expand/collapse toggle

### CK8: Popup Content - Metrics
- [ ] Duration
- [ ] Tokens used
- [ ] Findings count
- [ ] Confidence score
- [ ] Custom metrics per node type

### CK9: Popup Content - Logs
- [ ] Real-time log stream
- [ ] Log levels (info, warn, error)
- [ ] Timestamps
- [ ] Auto-scroll
- [ ] Search/filter
- [ ] Copy to clipboard
- [ ] Max lines (virtualized)

### CK10: Popup Content - Details
- [ ] Layer-specific: Questions asked, red flags found
- [ ] Perspective-specific: Persona info, focus areas
- [ ] Findings list with severity badges
- [ ] Suggestions/recommendations

## 14.4 VIEW MODES

### CK11: Flow View (Default)
- [ ] Horizontal flow layout
- [ ] Phase headers (INPUT → LAYERS → PERSPECTIVES → CONSENSUS → OUTPUT)
- [ ] Absolute positioned nodes
- [ ] SVG connections
- [ ] Zoom/pan controls
- [ ] Fullscreen toggle

### CK12: Grid View (Alternative)
- [ ] Column-based layout (por fase ou por tipo)
- [ ] Compact cards
- [ ] Sortable
- [ ] Filterable by status

### CK13: Timeline View (Future)
- [ ] Temporal view
- [ ] Gantt-like
- [ ] Duration bars
- [ ] Parallel execution visualization

## 14.5 METRICS & STATS

### CK14: Hero Metrics
- [ ] Current validation ID
- [ ] Overall progress %
- [ ] Active layers count
- [ ] Findings count (by severity)
- [ ] Elapsed time
- [ ] Tokens used
- [ ] Warnings count

### CK15: Completion Donut
- [ ] Done % (green)
- [ ] Active % (blue)
- [ ] Pending % (gray)
- [ ] Error % (red)
- [ ] Animated transitions

### CK16: Summary Stats
- [ ] Total nodes
- [ ] Complete count
- [ ] Active count
- [ ] Error count
- [ ] Skipped count

## 14.6 CONTROLS

### CK17: Validation Controls
- [ ] Start validation button
- [ ] Pause/Resume button
- [ ] Abort button
- [ ] Re-run button
- [ ] Configuration quick access

### CK18: Layer Selection
- [ ] Layer pack selector (FULL, SECURITY, USABILITY, MINIMAL)
- [ ] Individual layer toggles
- [ ] Perspective toggles
- [ ] Redundancy level slider (1-5)

### CK19: Display Controls
- [ ] View toggle (Flow/Grid/Timeline)
- [ ] Fullscreen toggle
- [ ] Theme toggle (dark/light)
- [ ] Animation speed
- [ ] Auto-refresh toggle

## 14.7 REAL-TIME EVENTS

### CK20: Event Types
```javascript
// WebSocket events
'validation:started'     // Validação iniciou
'validation:complete'    // Validação terminou
'layer:started'          // Layer começou
'layer:complete'         // Layer terminou
'layer:finding'          // Finding descoberto
'perspective:started'    // Perspectiva começou
'perspective:complete'   // Perspectiva terminou
'consensus:started'      // Consenso começou
'consensus:vote'         // Voto recebido
'consensus:complete'     // Consenso finalizado
'veto:triggered'         // Veto acionado
'node:log'              // Log de um node
'node:metrics'          // Métricas de um node
'error:occurred'        // Erro ocorreu
```

### CK21: Event Handlers
- [ ] Update node status
- [ ] Update metrics
- [ ] Append log
- [ ] Show toast notification
- [ ] Play sound (optional)
- [ ] Browser notification (optional)

## 14.8 THEMING

### CK22: Color Palette (Dark Theme - Default)
```css
--bg-primary: #0a0a0f       /* Fundo principal */
--bg-secondary: #12121a     /* Fundo secundário */
--bg-muted: #1a1a24         /* Fundo muted */
--text-primary: #f0f0f5     /* Texto principal */
--text-muted: #6b7280       /* Texto muted */
--accent: #3b82f6           /* Azul accent */
--success: #22c55e          /* Verde sucesso */
--warning: #f59e0b          /* Amarelo warning */
--error: #ef4444            /* Vermelho erro */
--border-subtle: #2a2a3a    /* Borda sutil */

/* Layer colors (L0-L6 do pipeline) */
--l0: #ef4444               /* System - vermelho */
--l1: #f59e0b               /* Exec - laranja */
--l2: #eab308               /* VPs - amarelo */
--l3: #22c55e               /* Masters - verde */
--l4: #06b6d4               /* Leads - cyan */
--l5: #3b82f6               /* Workers - azul */
--l6: #8b5cf6               /* AI - roxo */
```

### CK23: Light Theme
- [ ] Inverted colors
- [ ] Adjusted contrasts
- [ ] Subtle shadows

### CK24: Human Layer Specific Colors
```css
/* Severity colors */
--severity-critical: #dc2626
--severity-high: #ea580c
--severity-medium: #ca8a04
--severity-low: #16a34a
--severity-info: #0ea5e9

/* Veto colors */
--veto-strong: #dc2626
--veto-medium: #f59e0b
--veto-weak: #3b82f6
--veto-none: #6b7280
```

## 14.9 ANIMATIONS

### CK25: Node Animations
- [ ] Pulse when active (CSS keyframes)
- [ ] Glow intensity based on activity
- [ ] Fade in/out on state change
- [ ] Shake on error
- [ ] Bounce on complete

### CK26: Connection Animations
- [ ] Flowing dots along path
- [ ] Color transition
- [ ] Thickness pulse
- [ ] Dashed to solid transition

### CK27: Transition Animations
- [ ] View mode transitions
- [ ] Popup slide in/out
- [ ] Metric value counting up
- [ ] Progress bar smooth fill

## 14.10 RESPONSIVE & MOBILE

### CK28: Responsive Behavior
- [ ] Collapsible sidebar
- [ ] Stacked metrics on mobile
- [ ] Simplified flow view on small screens
- [ ] Touch-friendly node tap
- [ ] Swipe to switch views

### CK29: Mobile Considerations
- [ ] PWA support (optional)
- [ ] Offline indicator
- [ ] Reduced animations on low-end devices

## 14.11 ACCESSIBILITY

### CK30: Cockpit A11y
- [ ] Keyboard navigation between nodes
- [ ] Focus indicators
- [ ] ARIA labels for nodes
- [ ] Screen reader announcements for events
- [ ] Color-blind friendly palette option
- [ ] Reduced motion option

## 14.12 PERFORMANCE

### CK31: Performance Optimizations
- [ ] Virtualized log rendering
- [ ] Debounced updates
- [ ] RAF for animations
- [ ] Lazy load popup content
- [ ] SVG path caching
- [ ] WebSocket batching

## 14.13 ADVANCED VISUALIZATIONS (State of the Art)

### CK32: Graph Visualization Engine
```yaml
graph_engine:
  library: "D3.js + React Flow"

  features:
    force_layout:
      enabled: true
      physics: "d3-force"
      collision_detection: true

    hierarchical_layout:
      enabled: true
      direction: "LR"  # Left to Right
      spacing:
        horizontal: 200
        vertical: 100

    minimap:
      enabled: true
      position: "bottom-right"
      interactive: true

    controls:
      zoom: { min: 0.5, max: 2, step: 0.1 }
      pan: { enabled: true, smooth: true }
      fit: { padding: 50, duration: 500 }
```

### CK33: Node Visualization Details
```typescript
interface NodeVisualization {
  // Forma e tamanho adaptáveis
  shape: "circle" | "hexagon" | "rounded-rect";
  size: {
    base: 80;
    scale_by: "importance" | "activity";
    min: 60;
    max: 120;
  };

  // Visual state machine
  states: {
    idle: { glow: 0, pulse: false, border: "1px solid gray" };
    pending: { glow: 0.2, pulse: false, border: "1px dashed gray" };
    active: { glow: 0.8, pulse: true, border: "2px solid blue" };
    processing: { glow: 0.6, pulse: "slow", border: "2px solid cyan" };
    success: { glow: 0.4, pulse: false, border: "2px solid green" };
    warning: { glow: 0.6, pulse: "fast", border: "2px solid yellow" };
    error: { glow: 1.0, pulse: "fast", border: "3px solid red" };
  };

  // Inner content
  content: {
    icon: "SVG based on node type";
    label: "truncated at 12 chars";
    badge: "severity or count";
    progress_ring: "optional circular progress";
  };
}
```

### CK34: Edge Visualization
```yaml
edge_visualization:
  types:
    data_flow:
      style: "solid"
      animated: true
      particles: 3
      color: "based on status"

    dependency:
      style: "dashed"
      animated: false
      color: "gray"

    error_flow:
      style: "solid"
      animated: true
      particles: 5
      color: "red"
      glow: true

  animations:
    particle_speed: 2s
    path_morph: 300ms
    color_transition: 200ms
```

### CK35: 3D Visualization Mode (Premium)
```yaml
3d_mode:
  enabled: true
  library: "Three.js + react-three-fiber"

  features:
    - Perspective camera with orbit controls
    - Depth-based layering (7 layers = 7 z-levels)
    - Particle system for data flow
    - Bloom effect for active nodes
    - Environment mapping for premium feel

  performance:
    max_nodes: 500
    lod_levels: 3  # Level of detail
    frustum_culling: true
    instanced_meshes: true
```

## 14.14 REAL-TIME INTELLIGENCE

### CK36: Predictive Visualization
```yaml
predictions:
  eta_calculation:
    method: "historical + current rate"
    display: "ETA: ~2m 30s"
    confidence_interval: true

  bottleneck_detection:
    enabled: true
    visual: "pulsing red outline on slow nodes"
    threshold: "2x average duration"

  failure_prediction:
    enabled: true
    model: "pattern matching from history"
    visual: "warning glow before failure"
```

### CK37: Smart Alerts
- [ ] **Proactive alerts**: "Layer 4 is taking longer than usual"
- [ ] **Anomaly detection**: "Unusual pattern detected in findings"
- [ ] **Threshold alerts**: "Token budget at 80%"
- [ ] **Recommendation alerts**: "Consider running HL-3 for better coverage"
- [ ] **Completion prediction**: "Estimated completion in 3 minutes"

### CK38: Historical Comparison
```yaml
historical_comparison:
  overlays:
    - Previous run ghost (semi-transparent)
    - Average timeline
    - Best run timeline

  metrics:
    - Delta vs previous
    - Trend arrows
    - Percentile ranking

  timeline_scrubber:
    enabled: true
    playback_speed: [0.5x, 1x, 2x, 4x]
    jump_to_events: true
```

## 14.15 COLLABORATION FEATURES

### CK39: Multi-User View
```yaml
collaboration:
  presence:
    show_cursors: true
    show_selections: true
    colors: "unique per user"

  annotations:
    enabled: true
    types: [comment, highlight, flag]
    persistence: "per-validation"

  sharing:
    live_link: "real-time view sharing"
    snapshot: "point-in-time capture"
    embed: "iframe for external dashboards"
```

### CK40: Export & Reporting
- [ ] **Screenshot**: High-res PNG/SVG
- [ ] **Video recording**: Full session replay
- [ ] **PDF report**: With cockpit snapshot
- [ ] **JSON export**: Full state dump
- [ ] **Embed code**: For status pages
- [ ] **Slack/Teams integration**: Rich preview cards

## 14.16 KEYBOARD SHORTCUTS

### CK41: Keyboard Navigation
```yaml
shortcuts:
  navigation:
    "1-7": "Jump to Layer 1-7"
    "Tab": "Next node"
    "Shift+Tab": "Previous node"
    "Enter": "Open popup"
    "Escape": "Close popup"
    "f": "Toggle fullscreen"

  view:
    "+/-": "Zoom in/out"
    "0": "Fit to screen"
    "g": "Toggle grid view"
    "t": "Toggle timeline view"
    "m": "Toggle minimap"

  actions:
    "r": "Re-run validation"
    "p": "Pause/Resume"
    "x": "Abort"
    "e": "Export"

  help:
    "?": "Show shortcuts"
```

## 14.17 CUSTOMIZATION

### CK42: User Preferences
```yaml
user_preferences:
  layout:
    default_view: "flow" | "grid" | "timeline"
    sidebar_position: "left" | "right"
    metrics_position: "top" | "bottom"

  visual:
    theme: "dark" | "light" | "system"
    animation_speed: 0.5 | 1 | 2
    particle_density: "low" | "medium" | "high"

  behavior:
    auto_follow_active: true
    sound_notifications: false
    auto_expand_errors: true

  persistence: "localStorage + cloud sync"
```

### CK43: Saved Views
- [ ] Save current layout as preset
- [ ] Name and describe presets
- [ ] Share presets with team
- [ ] Quick switch between presets
- [ ] Default preset per project

---

# PARTE 13: INTERNATIONALIZATION (i18n)

## 13.1 LANGUAGE SUPPORT

### i18n1: Supported Languages
- [ ] English (en) - Primary
- [ ] Portuguese Brazilian (pt-BR)
- [ ] Spanish (es)
- [ ] French (fr)
- [ ] German (de)
- [ ] Japanese (ja)
- [ ] Chinese Simplified (zh-CN)
- [ ] Korean (ko)
- [ ] Language roadmap (priority order)
- [ ] Community contribution process

### i18n2: Language Detection
- [ ] Browser language detection
- [ ] User preference storage
- [ ] URL-based (subdomain vs path)
- [ ] Fallback chain (pt-BR → pt → en)
- [ ] Cookie/localStorage persistence

## 13.2 CONTENT TRANSLATION

### i18n3: UI Strings
- [ ] Button labels
- [ ] Form labels
- [ ] Navigation items
- [ ] Error messages
- [ ] Success messages
- [ ] Tooltips
- [ ] Placeholders
- [ ] Validation messages
- [ ] Empty states
- [ ] Loading messages

### i18n4: Dynamic Content
- [ ] Finding descriptions
- [ ] Layer explanations
- [ ] Perspective descriptions
- [ ] Report summaries
- [ ] LLM prompt localization
- [ ] LLM response handling (multilingual)

### i18n5: Long-form Content
- [ ] Documentation
- [ ] Blog posts
- [ ] Landing page copy
- [ ] Legal pages (Terms, Privacy)
- [ ] Email templates
- [ ] Help articles
- [ ] Changelog

### i18n6: Marketing Content
- [ ] Landing page
- [ ] Pricing page
- [ ] Feature descriptions
- [ ] Testimonials
- [ ] Case studies
- [ ] SEO meta tags per language

## 13.3 TECHNICAL IMPLEMENTATION

### i18n7: Translation System
- [ ] Translation file format (JSON, YAML, PO)
- [ ] Namespacing strategy
- [ ] Key naming convention
- [ ] Pluralization rules
- [ ] Interpolation (variables in strings)
- [ ] Context hints for translators
- [ ] Fallback handling

### i18n8: Translation Management
- [ ] Translation platform (Crowdin, Lokalise, etc.)
- [ ] Translator workflow
- [ ] Review process
- [ ] Version control integration
- [ ] CI/CD integration
- [ ] Missing translation detection
- [ ] Translation coverage reports

### i18n9: Frontend Implementation
- [ ] i18n library (react-i18next, vue-i18n, etc.)
- [ ] Lazy loading translations
- [ ] SSR considerations
- [ ] SEO (hreflang tags)
- [ ] URL structure
- [ ] Language switcher component

### i18n10: Backend Implementation
- [ ] API response localization
- [ ] Error message localization
- [ ] Email localization
- [ ] Notification localization
- [ ] Report generation localization

### i18n11: CLI Implementation
- [ ] CLI output localization
- [ ] Help text localization
- [ ] Error messages
- [ ] Environment variable (LANG)

## 13.4 LOCALIZATION (L10n)

### i18n12: Date & Time
- [ ] Date formats per locale
- [ ] Time formats (12h vs 24h)
- [ ] Timezone display
- [ ] Relative time ("2 hours ago")
- [ ] Calendar start day (Sun vs Mon)

### i18n13: Numbers & Currency
- [ ] Number formats (1,000 vs 1.000)
- [ ] Decimal separators
- [ ] Currency display
- [ ] Currency conversion (display only)
- [ ] Percentage formats

### i18n14: Text Direction
- [ ] RTL support (Arabic, Hebrew)
- [ ] Bidirectional text
- [ ] CSS logical properties
- [ ] Icon mirroring

### i18n15: Cultural Considerations
- [ ] Color meanings
- [ ] Icon appropriateness
- [ ] Image localization
- [ ] Name formats
- [ ] Address formats
- [ ] Phone formats

## 13.5 QUALITY ASSURANCE

### i18n16: Testing
- [ ] Pseudo-localization testing
- [ ] String expansion testing (German ~30% longer)
- [ ] RTL testing
- [ ] Screenshot comparison per locale
- [ ] Missing translation CI check
- [ ] Hardcoded string detection

### i18n17: Review Process
- [ ] Native speaker review
- [ ] Context validation
- [ ] Consistency check
- [ ] Brand voice check
- [ ] Technical accuracy

## 13.6 OSS vs CLOUD i18n

### i18n18: OSS Localization
- [ ] Community translation contribution
- [ ] Translation PR process
- [ ] Credit/attribution
- [ ] Partial translation handling

### i18n19: Cloud Localization
- [ ] Dashboard language setting
- [ ] Per-user preference
- [ ] Team default language
- [ ] Report language selection
- [ ] Email language preference

## 13.7 AI-POWERED TRANSLATION (State of the Art)

### i18n20: AI Translation Pipeline
```yaml
ai_translation:
  primary_model: "Claude" (Anthropic)
  fallback_model: "GPT-4" (OpenAI)

  workflow:
    1_machine_translation:
      tool: "AI model with context"
      preserves: ["variables", "HTML tags", "markdown"]

    2_terminology_check:
      glossary: "product-specific terms"
      consistency: "across all strings"

    3_quality_scoring:
      metrics: ["fluency", "accuracy", "consistency"]
      threshold: 0.85

    4_human_review:
      required_for: ["marketing", "legal", "UI critical"]
      optional_for: ["tooltips", "help text"]

  context_injection:
    - Component name
    - Screenshot of usage
    - Related strings
    - Product glossary
```

### i18n21: Smart Translation Features
- [ ] **Context-aware**: Entende onde a string aparece
- [ ] **Consistent terminology**: Mesmos termos em todo produto
- [ ] **Tone matching**: Mantém voz da marca
- [ ] **Length adaptation**: Ajusta para espaço disponível
- [ ] **Pluralization auto**: Gera todas as formas plurais
- [ ] **Gender handling**: Para idiomas com gênero gramatical
- [ ] **Formality levels**: Formal vs informal por cultura

### i18n22: Translation Memory + TM
```yaml
translation_memory:
  storage: "Vector database (Qdrant)"
  matching:
    exact: "100% match → reuse"
    fuzzy: ">85% → suggest with diff"
    semantic: "Similar meaning → flag for review"

  benefits:
    - Consistency across project
    - Faster translation
    - Cost reduction
    - Quality improvement
```

### i18n23: Continuous Localization
```yaml
continuous_localization:
  trigger: "PR merged to main"

  pipeline:
    1. Extract new/changed strings
    2. Send to translation queue
    3. AI translation (immediate draft)
    4. Human review queue (async)
    5. PR with translations
    6. Deploy to staging
    7. Visual QA
    8. Promote to production

  metrics:
    time_to_translate: "< 24h for AI draft"
    time_to_production: "< 1 week with review"
    coverage_target: ">= 95% per release"
```

---

# PARTE 15: WEBSITE HUMANGR (Site Institucional Completo)

> **Empresa**: HumanGR
> **Objetivo**: Site de empresa de AI de primeiro mundo, premium, com design xique
> **Referência**: Sites de empresas como Anthropic, OpenAI, Vercel, Linear, Stripe

## 15.1 ARQUITETURA DO SITE

### W1: Estrutura de Páginas
```
humangr.ai/
├── /                           # Homepage (Landing Page)
├── /product                    # Human Layer Product Page
├── /pricing                    # Pricing Page
├── /docs                       # Documentation (separate subdomain?)
├── /blog                       # Blog
├── /changelog                  # Changelog
├── /about                      # About Us
├── /careers                    # Careers (future)
├── /contact                    # Contact
├── /security                   # Security & Compliance
├── /legal/
│   ├── /terms                  # Terms of Service
│   ├── /privacy                # Privacy Policy
│   ├── /license                # Open Source License
│   └── /dpa                    # Data Processing Agreement
├── /enterprise                 # Enterprise page
├── /partners                   # Partner program
├── /community                  # Community hub
├── /login                      # Login redirect
└── /signup                     # Signup redirect
```

### W2: Tech Stack (Site)
- [ ] Framework (Next.js, Astro, Remix)
- [ ] Hosting (Vercel, Cloudflare Pages)
- [ ] CMS for blog (Contentlayer, Sanity, MDX)
- [ ] Analytics (Plausible, Posthog)
- [ ] Forms (formspree, custom)
- [ ] Search (Algolia DocSearch for docs)
- [ ] Image optimization (next/image, Cloudflare Images)
- [ ] CDN configuration

### W3: SEO & Meta
- [ ] Meta tags per page
- [ ] Open Graph images
- [ ] Twitter cards
- [ ] JSON-LD structured data
- [ ] Sitemap.xml
- [ ] robots.txt
- [ ] Canonical URLs
- [ ] hreflang for i18n

## 15.2 HOMEPAGE (LANDING PAGE)

### W4: Hero Section
- [ ] Headline principal (impactante, conciso)
- [ ] Subheadline explicativo
- [ ] CTA primário ("Get Started Free")
- [ ] CTA secundário ("View Docs" / "Watch Demo")
- [ ] Visual hero (mockup, ilustração, ou animação)
- [ ] Trust badges (GitHub stars, "OSS", "YC-style")
- [ ] Social proof snippet ("Used by X developers")

### W5: Problem Statement
- [ ] Headline do problema
- [ ] Bullets com dor points (3-4)
- [ ] Visual ou ícone de cada dor
- [ ] Transição para solução

### W6: Solution Overview
- [ ] "How Human Layer Solves This"
- [ ] Visual do fluxo de validação
- [ ] 7 Layers preview (mini icons)
- [ ] Triple Redundancy visual
- [ ] Consensus visualization

### W7: Features Showcase
- [ ] Feature grid (6-9 features)
- [ ] Icon + Título + Descrição curta
- [ ] Hover effects interativos
- [ ] Links para detalhes

### W8: 7 Human Layers Section
```
Interactive visualization:
- [ ] Layer selector (click ou hover)
- [ ] Detail panel que muda
- [ ] Animação entre layers
- [ ] Veto power indicator por layer
- [ ] Example finding por layer
```

### W9: How It Works (3-Step)
```
Step 1: Install
- [ ] Code snippet visual (pip install)
- [ ] "5 minutes to setup"

Step 2: Configure
- [ ] Config file preview
- [ ] "Your LLM, your control"

Step 3: Validate
- [ ] Result preview
- [ ] "Triple redundancy, 2/3 consensus"
```

### W10: Use Cases Grid
- [ ] Use case cards (4-6)
- [ ] Icon + Title + Description
- [ ] Target persona each
- [ ] Link to detailed page

### W11: Comparison Section
- [ ] "Why Human Layer vs Manual Review"
- [ ] vs "Automated Tests alone"
- [ ] Feature comparison table
- [ ] Speed comparison
- [ ] Cost comparison

### W12: Social Proof Section
- [ ] Customer logos (se houver)
- [ ] GitHub stars counter (live)
- [ ] NPM/PyPI downloads (se houver)
- [ ] Testimonials carousel
- [ ] Quote cards
- [ ] Company name + role

### W13: Pricing Preview
- [ ] 3 tiers highlighted (Free, Pro, Business)
- [ ] Key differentiator per tier
- [ ] "See all plans" CTA
- [ ] "OSS forever free" badge

### W14: Open Source Section
- [ ] "Open Source at Heart"
- [ ] GitHub preview embed
- [ ] Contributor count
- [ ] Star count
- [ ] "Star us on GitHub" CTA
- [ ] License info (Apache 2.0)

### W15: CTA Section (Final)
- [ ] Compelling headline
- [ ] "Start validating in 5 minutes"
- [ ] Get Started button (prominent)
- [ ] "No credit card required"
- [ ] Secondary: Schedule Demo

### W16: Footer
- [ ] Logo
- [ ] Product links
- [ ] Company links
- [ ] Resources links
- [ ] Legal links
- [ ] Social links (GitHub, Twitter/X, Discord)
- [ ] Newsletter signup
- [ ] Language selector
- [ ] "Made with ❤️" or similar
- [ ] Copyright

## 15.3 PRODUCT PAGE (/product)

### W17: Product Hero
- [ ] Product name + tagline
- [ ] Key value proposition
- [ ] Product screenshot/demo
- [ ] CTAs (Try Free, View Docs)

### W18: Deep Feature Sections
```
For each major feature:
- [ ] Feature headline
- [ ] Detailed description
- [ ] Visual (screenshot, animation, illustration)
- [ ] Code example (if applicable)
- [ ] Benefits list
- [ ] Use case example
```

### W19: Technical Architecture
- [ ] Architecture diagram
- [ ] MCP protocol explanation
- [ ] Component breakdown
- [ ] Integration points

### W20: Integrations Section
- [ ] LLM providers (Claude, GPT, Gemini, Ollama)
- [ ] CI/CD platforms (GitHub, GitLab, etc.)
- [ ] "Works with your stack"

### W21: Live Demo (Optional)
- [ ] Interactive demo
- [ ] Sample validation
- [ ] Result visualization

## 15.4 PRICING PAGE (/pricing)

### W22: Pricing Header
- [ ] Headline ("Simple, transparent pricing")
- [ ] Billing toggle (Monthly/Annual with discount)
- [ ] "OSS is always free" badge

### W23: Pricing Cards
```
For each tier (7 tiers):
├── OSS (Free forever)
├── Free Cloud
├── Solo ($12/mo)
├── Pro ($39/mo)
├── Team ($99/mo)
├── Business ($249/mo)
└── Enterprise (Custom)

Per card:
- [ ] Tier name
- [ ] Price (monthly & annual)
- [ ] Key differentiator tagline
- [ ] Feature list (bulleted)
- [ ] CTA button
- [ ] "Popular" badge (Pro or Team)
- [ ] "Contact Sales" (Enterprise)
```

### W24: Feature Comparison Table
- [ ] Full feature matrix
- [ ] Expandable rows
- [ ] Tooltips for complex features
- [ ] Sticky header
- [ ] Mobile-friendly (collapsible)

### W25: Pricing FAQ
- [ ] "Can I use my own LLM?"
- [ ] "What counts as a validation?"
- [ ] "Can I switch plans?"
- [ ] "What happens if I exceed limits?"
- [ ] "Is there a free trial?"
- [ ] "Do you offer discounts?"
- [ ] "What payment methods?"

### W26: Enterprise Section
- [ ] "Need more?" headline
- [ ] Enterprise features highlight
- [ ] Custom pricing
- [ ] Contact sales CTA
- [ ] "Let's talk" form

### W27: Money Back / Guarantee
- [ ] Satisfaction guarantee
- [ ] Refund policy
- [ ] Trust badges (secure payment)

## 15.5 ABOUT PAGE (/about)

### W28: Company Story
- [ ] Origin story
- [ ] Mission statement
- [ ] Vision statement
- [ ] "Why we built Human Layer"

### W29: Team Section (Optional)
- [ ] Team photos
- [ ] Names + roles
- [ ] Brief bios
- [ ] Social links

### W30: Values
- [ ] Core values (3-5)
- [ ] Visual representation
- [ ] How values guide product

### W31: Timeline / Milestones
- [ ] Company timeline
- [ ] Key milestones
- [ ] Future roadmap preview

### W32: Press / Media
- [ ] Press kit link
- [ ] Media mentions
- [ ] Logos for press use

## 15.6 SECURITY PAGE (/security)

### W33: Security Overview
- [ ] Security philosophy
- [ ] "Your data stays yours"
- [ ] BYOK emphasis
- [ ] No data training policy

### W34: Security Features
- [ ] Encryption (at rest, in transit)
- [ ] Authentication methods
- [ ] Access controls
- [ ] Audit logging
- [ ] Data retention

### W35: Compliance
- [ ] GDPR compliance
- [ ] CCPA compliance
- [ ] SOC2 (if applicable, or roadmap)
- [ ] Certifications

### W36: Security Practices
- [ ] Vulnerability disclosure
- [ ] Penetration testing
- [ ] Code review practices
- [ ] Dependency management

### W37: Trust Center
- [ ] Status page link
- [ ] Incident history
- [ ] Contact security team

## 15.7 LEGAL PAGES

### W38: Terms of Service (/legal/terms)
- [ ] Plain language summary
- [ ] Full legal text
- [ ] Table of contents
- [ ] Last updated date
- [ ] Version history

### W39: Privacy Policy (/legal/privacy)
- [ ] Plain language summary
- [ ] What we collect
- [ ] How we use data
- [ ] Third parties
- [ ] Your rights
- [ ] Contact info

### W40: Open Source License (/legal/license)
- [ ] License type (Apache 2.0 recommended)
- [ ] What you can do
- [ ] Attribution requirements
- [ ] Liability disclaimer
- [ ] Link to GitHub LICENSE file

### W41: DPA (/legal/dpa)
- [ ] Data Processing Agreement
- [ ] For enterprise customers
- [ ] GDPR compliance terms
- [ ] Sub-processors list

### W42: Acceptable Use Policy
- [ ] Prohibited uses
- [ ] Fair use guidelines
- [ ] Enforcement

## 15.8 ENTERPRISE PAGE (/enterprise)

### W43: Enterprise Hero
- [ ] "Built for scale"
- [ ] Enterprise-specific value prop
- [ ] Request demo CTA

### W44: Enterprise Features
- [ ] SSO / SAML
- [ ] Custom SLA
- [ ] Dedicated support
- [ ] On-premise option
- [ ] Custom integrations
- [ ] Volume pricing
- [ ] Priority roadmap influence

### W45: Enterprise Security
- [ ] SOC2 (or roadmap)
- [ ] Custom data residency
- [ ] Audit capabilities
- [ ] Compliance support

### W46: Case Studies (Enterprise)
- [ ] Enterprise customer stories
- [ ] Results/ROI
- [ ] Quote from decision maker

### W47: Contact Sales
- [ ] Contact form
- [ ] Schedule demo
- [ ] Request pricing

## 15.9 COMMUNITY PAGE (/community)

### W48: Community Hub
- [ ] "Join the Community"
- [ ] Discord/Slack link
- [ ] GitHub Discussions link
- [ ] Stack Overflow tag

### W49: Contribute
- [ ] "How to contribute"
- [ ] Good first issues
- [ ] Contributor recognition
- [ ] Code of conduct

### W50: Showcase
- [ ] Community projects
- [ ] Integrations built by community
- [ ] Featured contributors

## 15.10 CONTACT PAGE (/contact)

### W51: Contact Form
- [ ] Name field
- [ ] Email field
- [ ] Company (optional)
- [ ] Subject/Category dropdown
- [ ] Message field
- [ ] Submit + confirmation

### W52: Contact Options
- [ ] Support email
- [ ] Sales email
- [ ] Security email
- [ ] Social links
- [ ] Office address (if applicable)

## 15.11 BLOG (/blog)

### W53: Blog Listing
- [ ] Featured post
- [ ] Post cards (title, excerpt, date, author)
- [ ] Categories/Tags filter
- [ ] Search
- [ ] Pagination
- [ ] Newsletter signup

### W54: Blog Post Template
- [ ] Hero image
- [ ] Title
- [ ] Author + date
- [ ] Reading time
- [ ] Table of contents
- [ ] Content (MDX with components)
- [ ] Related posts
- [ ] Share buttons
- [ ] Author bio
- [ ] Newsletter CTA

### W55: Blog Categories
- [ ] Product updates
- [ ] Engineering
- [ ] Tutorials
- [ ] Case studies
- [ ] Company news
- [ ] Industry insights

## 15.12 CHANGELOG (/changelog)

### W56: Changelog Listing
- [ ] Version history
- [ ] Date per version
- [ ] Category badges (feature, fix, breaking)
- [ ] Search/filter
- [ ] Subscribe to updates

### W57: Changelog Entry Template
- [ ] Version number
- [ ] Release date
- [ ] Summary
- [ ] New features
- [ ] Improvements
- [ ] Bug fixes
- [ ] Breaking changes
- [ ] Migration guide links

## 15.13 INTERACTIVE ELEMENTS (Xique)

### W58: Hero Animations
- [ ] Gradient mesh background
- [ ] Floating particles/shapes
- [ ] Parallax on scroll
- [ ] Smooth scroll behavior
- [ ] Typing effect for tagline

### W59: Interactive Diagrams
- [ ] SVG animations
- [ ] Click/hover to reveal
- [ ] Animated flow diagrams
- [ ] Interactive layer selector
- [ ] Consensus visualization

### W60: Code Snippets
- [ ] Syntax highlighting
- [ ] Copy button
- [ ] Language tabs
- [ ] Running example (optional)
- [ ] Terminal mockup style

### W61: Scroll Animations
- [ ] Fade in on scroll
- [ ] Slide in from sides
- [ ] Staggered reveals
- [ ] Counter animations (stats)
- [ ] Progress indicator

### W62: Micro-interactions
- [ ] Button hover effects
- [ ] Card hover lift
- [ ] Link underline animations
- [ ] Toggle switches
- [ ] Checkbox animations
- [ ] Form focus states

### W63: Loading States
- [ ] Page transitions
- [ ] Skeleton screens
- [ ] Blur-up images
- [ ] Progressive loading

### W64: Easter Eggs (Optional)
- [ ] Konami code
- [ ] Hidden features
- [ ] Developer console message
- [ ] Fun 404 page

## 15.14 DESIGN PREMIUM (First-World)

### W65: Visual Design Principles
- [ ] Clean, minimal aesthetic
- [ ] Generous whitespace
- [ ] Consistent spacing grid
- [ ] Limited color palette
- [ ] High-quality imagery
- [ ] Custom illustrations

### W66: Typography
- [ ] Premium font pairing
- [ ] Clear hierarchy
- [ ] Optimal line lengths
- [ ] Proper letter spacing
- [ ] Responsive scaling

### W67: Imagery
- [ ] Custom illustrations
- [ ] High-quality screenshots
- [ ] Product mockups
- [ ] Abstract backgrounds
- [ ] Icon consistency

### W68: Dark Mode
- [ ] Full dark mode support
- [ ] System preference detection
- [ ] Smooth transition
- [ ] Proper contrast in dark

### W69: Premium Details
- [ ] Subtle shadows
- [ ] Glass morphism (selective)
- [ ] Gradient accents
- [ ] Smooth corners
- [ ] Micro-animations everywhere

## 15.15 PERFORMANCE & TECH

### W70: Performance Targets
- [ ] Lighthouse 95+ score
- [ ] LCP < 2.5s
- [ ] FID < 100ms
- [ ] CLS < 0.1
- [ ] TTI < 3.5s

### W71: Technical Optimizations
- [ ] Static generation (SSG)
- [ ] Image optimization
- [ ] Font optimization
- [ ] Code splitting
- [ ] Lazy loading
- [ ] Prefetching
- [ ] CDN caching
- [ ] Compression (Brotli)

### W72: Accessibility
- [ ] WCAG 2.1 AA compliance
- [ ] Keyboard navigation
- [ ] Screen reader tested
- [ ] Color contrast
- [ ] Alt texts
- [ ] Focus indicators

### W73: Analytics & Tracking
- [ ] Privacy-friendly analytics (Plausible/Posthog)
- [ ] Conversion tracking
- [ ] Event tracking
- [ ] Heatmaps (optional)
- [ ] A/B testing capability

## 15.16 RESPONSIVE & MOBILE

### W74: Responsive Design
- [ ] Mobile-first approach
- [ ] Breakpoints (mobile, tablet, desktop, wide)
- [ ] Touch-friendly targets
- [ ] No horizontal scroll
- [ ] Readable font sizes

### W75: Mobile Navigation
- [ ] Hamburger menu
- [ ] Slide-out drawer
- [ ] Sticky header option
- [ ] Bottom CTA bar (mobile)

### W76: Mobile Optimizations
- [ ] Reduced animations
- [ ] Optimized images
- [ ] Touch gestures
- [ ] Phone number click-to-call
- [ ] App-like feel

## 15.17 LOCALIZATION

### W77: Multi-language Support
- [ ] Language switcher
- [ ] URL structure (/pt, /es, etc.)
- [ ] Localized SEO
- [ ] RTL support (if needed)
- [ ] Currency localization (pricing)

## 15.18 CONVERSION OPTIMIZATION

### W78: CTAs Throughout
- [ ] Clear primary CTA per page
- [ ] Secondary CTAs
- [ ] Exit intent popup (careful)
- [ ] Sticky CTA bar
- [ ] Form optimization

### W79: Trust Signals
- [ ] Security badges
- [ ] Customer logos
- [ ] GitHub stars
- [ ] Reviews/ratings
- [ ] Certifications

### W80: Lead Capture
- [ ] Newsletter signup
- [ ] Demo request form
- [ ] Contact form
- [ ] Gated content (optional)

---

# PARTE 16: MARKETING MATERIALS & PITCH ASSETS

> **Objetivo**: Materiais de marketing state-of-the-art para investidores, clientes e comunidade
> **Formato**: Apresentações, decks, one-pagers, benchmarks

---

## 16.1 PITCH DECK (Investor-Ready)

### MK1: Deck Structure (12-15 slides)
```yaml
pitch_deck:
  format: "16:9, high-resolution"
  style: "Clean, data-driven, premium"
  branding: "HumanGR visual identity"

  slides:
    1_title:
      content: "HumanGR - Human Layer for AI Agents"
      elements: [Logo, Tagline, Contact info]

    2_problem:
      headline: "AI Agents Are Breaking Things"
      points:
        - "90% of AI agent failures are preventable"
        - "No standardized validation layer"
        - "Hallucinations cost $X billion/year"
      visual: "Before/After comparison"

    3_solution:
      headline: "Human Layer: 7 Layers of Validation"
      points:
        - "Structured validation protocol"
        - "Triple redundancy (2/3 consensus)"
        - "Veto power system"
      visual: "Layer diagram"

    4_product:
      headline: "MCP Server + Cloud Dashboard"
      points:
        - "OSS core (Apache 2.0)"
        - "Cloud premium features"
        - "BYOK (Bring Your Own Key)"
      visual: "Product screenshots"

    5_how_it_works:
      headline: "Validate → Score → Act"
      visual: "3-step flow diagram"

    6_market:
      headline: "The AI Agent Market"
      data:
        - TAM: "$50B by 2028"
        - SAM: "$15B (enterprise AI tools)"
        - SOM: "$500M (validation tools)"
      source: "Gartner, McKinsey"

    7_business_model:
      headline: "Open Core + Cloud"
      tiers:
        - OSS: "Free forever"
        - Solo: "$12/mo"
        - Pro: "$39/mo"
        - Team: "$99/mo"
        - Business: "$249/mo"
        - Enterprise: "Custom"
      visual: "Tier pyramid"

    8_traction:
      headline: "Early Traction"
      metrics:
        - GitHub stars: "target: 1K+ in 6 months"
        - Downloads: "target: 10K+ in 6 months"
        - Cloud users: "target: 500+ in 6 months"
        - Revenue: "target: $10K MRR in 12 months"

    9_competition:
      headline: "Competitive Landscape"
      visual: "2x2 matrix"
      differentiator: "Only structured 7-layer validation"

    10_team:
      headline: "The Team"
      members: [Founder info, advisors]

    11_roadmap:
      headline: "Product Roadmap"
      phases:
        Q1: "Launch OSS + Free Cloud"
        Q2: "Paid tiers + Enterprise features"
        Q3: "Advanced analytics + Integrations"
        Q4: "Platform expansion"

    12_financials:
      headline: "Financial Projections"
      projections:
        Year1: "$120K ARR"
        Year2: "$600K ARR"
        Year3: "$2.4M ARR"
      visual: "Growth chart"

    13_ask:
      headline: "The Ask"
      raise: "$500K - $1M Seed"
      use_of_funds:
        - "40% Engineering"
        - "30% Go-to-market"
        - "20% Operations"
        - "10% Reserve"

    14_contact:
      headline: "Let's Build Together"
      elements: [Email, LinkedIn, Website, QR code]
```

### MK2: Deck Versions
- [ ] **Full Deck** (15 slides): Para reuniões presenciais
- [ ] **Email Deck** (10 slides): Para envio frio
- [ ] **Demo Deck** (8 slides): Para demos de produto
- [ ] **One-Pager** (1 página): Executive summary

## 16.2 ONE-PAGER (Executive Summary)

### MK3: One-Pager Structure
```yaml
one_pager:
  format: "A4 or Letter, single page"
  sections:
    header:
      - Logo
      - Tagline
      - Website

    problem_solution:
      - 2 sentences on problem
      - 2 sentences on solution

    product:
      - Key features (bullets)
      - Screenshot/visual

    market:
      - TAM/SAM/SOM one-liner
      - Growth rate

    business_model:
      - Pricing summary
      - Key metrics

    traction:
      - Top 3 metrics

    team:
      - Founder one-liner

    contact:
      - Email
      - QR code to deck
```

## 16.3 SALES PRESENTATIONS

### MK4: Sales Deck (Customer-Facing)
```yaml
sales_deck:
  audience: "Technical decision makers"
  focus: "ROI, implementation, support"

  slides:
    1_intro:
      headline: "Stop AI Agent Failures"

    2_problem:
      headline: "The Cost of Unvalidated AI"
      data: "Case studies, failure costs"

    3_solution:
      headline: "Human Layer Validation"

    4_demo:
      headline: "See It In Action"
      content: "Live demo or video"

    5_features:
      headline: "Key Capabilities"

    6_integration:
      headline: "Works With Your Stack"
      visual: "Integration logos"

    7_pricing:
      headline: "Simple, Transparent Pricing"

    8_support:
      headline: "World-Class Support"

    9_case_study:
      headline: "Customer Success"

    10_next_steps:
      headline: "Get Started Today"
      CTA: "Free trial, demo, contact"
```

### MK5: Technical Deep-Dive Deck
- [ ] Architecture overview (20+ slides)
- [ ] Security & compliance focus
- [ ] Integration guide
- [ ] API documentation highlights

### MK6: Industry-Specific Decks
- [ ] **FinTech**: Compliance, fraud detection
- [ ] **HealthTech**: HIPAA, patient safety
- [ ] **E-commerce**: Customer experience, checkout
- [ ] **SaaS**: API validation, integration testing

## 16.4 KPIS & METRICS DASHBOARD

### MK7: Business KPIs
```yaml
business_kpis:
  acquisition:
    - MRR (Monthly Recurring Revenue)
    - ARR (Annual Recurring Revenue)
    - New MRR
    - Expansion MRR
    - Churned MRR
    - Net Revenue Retention (NRR)

  growth:
    - MoM Growth Rate
    - Customer Acquisition Cost (CAC)
    - Lifetime Value (LTV)
    - LTV/CAC Ratio (target: >3)
    - Payback Period (target: <12 months)

  engagement:
    - Daily Active Users (DAU)
    - Weekly Active Users (WAU)
    - Monthly Active Users (MAU)
    - DAU/MAU Ratio (stickiness)
    - Feature adoption rate

  conversion:
    - Website → Signup rate
    - Signup → Activation rate
    - Trial → Paid conversion
    - Free → Paid conversion
    - Upsell rate

  retention:
    - Logo churn rate
    - Revenue churn rate
    - Net dollar retention
    - Customer satisfaction (NPS, CSAT)
```

### MK8: Product KPIs
```yaml
product_kpis:
  usage:
    - Total validations run
    - Validations per user
    - Layers activated
    - Findings generated
    - Consensus rate

  quality:
    - False positive rate
    - False negative rate
    - Accuracy score
    - User satisfaction per validation

  performance:
    - Avg validation time
    - P95 validation time
    - API response time
    - Uptime %

  adoption:
    - OSS downloads
    - GitHub stars
    - Contributors
    - Documentation views
    - Community members
```

### MK9: Marketing KPIs
```yaml
marketing_kpis:
  awareness:
    - Website traffic
    - Organic search rankings
    - Social followers
    - Brand mentions
    - Share of voice

  engagement:
    - Blog views
    - Newsletter subscribers
    - Content downloads
    - Webinar attendees
    - Community activity

  lead_generation:
    - MQLs (Marketing Qualified Leads)
    - SQLs (Sales Qualified Leads)
    - Lead velocity
    - Cost per lead
    - Lead-to-opportunity rate
```

## 16.5 BENCHMARKS & COMPARISONS

### MK10: Industry Benchmarks
```yaml
industry_benchmarks:
  saas_metrics:
    reference: "OpenView, Bessemer, ChartMogul"

    growth:
      early_stage: "15-20% MoM"
      growth_stage: "10-15% MoM"
      mature: "5-10% MoM"

    retention:
      good_ndr: ">100%"
      great_ndr: ">120%"
      logo_churn: "<5% annually"

    efficiency:
      good_ltv_cac: ">3"
      great_ltv_cac: ">5"
      payback: "<12 months"

    conversion:
      website_to_signup: "2-5%"
      trial_to_paid: "15-25%"
      freemium_to_paid: "2-5%"

  developer_tools:
    reference: "Similar companies"

    github:
      good_stars: "1K+"
      great_stars: "5K+"
      exceptional_stars: "10K+"

    npm_downloads:
      good: "10K/month"
      great: "100K/month"

    community:
      discord_members: "1K+"
      contributors: "50+"
```

### MK11: Competitive Comparison Matrix
```yaml
competitive_matrix:
  dimensions:
    - Feature completeness
    - Ease of use
    - Pricing
    - OSS availability
    - Enterprise readiness
    - Integration ecosystem
    - Support quality
    - Documentation

  competitors:
    human_layer:
      positioning: "Leader in AI agent validation"
      strengths: ["7-layer depth", "OSS", "BYOK"]
      weaknesses: ["New entrant"]

    competitor_a:
      name: "Generic AI testing tool"
      strengths: ["Established"]
      weaknesses: ["Not agent-specific"]

    competitor_b:
      name: "In-house solutions"
      strengths: ["Customized"]
      weaknesses: ["Maintenance burden"]

  visual: "Radar chart or feature matrix"
```

### MK12: ROI Calculator
```yaml
roi_calculator:
  inputs:
    - Number of AI agents
    - Average cost per failure
    - Current failure rate
    - Team size
    - Current testing time

  calculations:
    failures_prevented: "agents * failure_rate * prevention_rate"
    cost_savings: "failures_prevented * cost_per_failure"
    time_savings: "team_size * testing_time * efficiency_gain"
    total_roi: "(cost_savings + time_savings) / subscription_cost"

  outputs:
    - Monthly savings
    - Annual savings
    - ROI percentage
    - Payback period
    - 5-year TCO comparison

  format: "Interactive web calculator + PDF export"
```

## 16.6 CASE STUDIES & SUCCESS STORIES

### MK13: Case Study Template
```yaml
case_study_template:
  sections:
    executive_summary:
      - Company name (or anonymized)
      - Industry
      - Key result (headline number)

    challenge:
      - Business context
      - Specific pain points
      - Previous solutions tried

    solution:
      - How Human Layer was implemented
      - Layers/features used
      - Integration approach

    results:
      - Quantitative metrics
      - Before/After comparison
      - Time to value

    testimonial:
      - Quote from stakeholder
      - Name, title, company (if public)

    key_takeaways:
      - 3-4 bullet points
      - Applicable insights

  format: "2-page PDF + web page"
```

### MK14: Target Case Studies
- [ ] **FinTech startup**: Reduced fraud false positives by 60%
- [ ] **E-commerce platform**: Cut checkout failures by 40%
- [ ] **SaaS company**: Improved API reliability to 99.9%
- [ ] **Enterprise**: Compliance automation saved 100+ hours/month
- [ ] **Open source project**: Community adoption story

## 16.7 WEBINARS & DEMOS

### MK15: Webinar Topics
```yaml
webinar_series:
  intro_series:
    - "Introduction to Human Layer"
    - "Getting Started with AI Agent Validation"
    - "OSS vs Cloud: Choosing Your Path"

  technical_series:
    - "Deep Dive: 7 Human Layers Explained"
    - "Integration Workshop: CI/CD + Human Layer"
    - "Advanced: Custom Perspectives"

  business_series:
    - "ROI of AI Validation"
    - "Case Study Showcase"
    - "Enterprise Deployment Best Practices"

  community_series:
    - "Office Hours (Q&A)"
    - "Contributor Spotlight"
    - "Roadmap Review"

  format:
    duration: "30-45 minutes"
    recording: "Always recorded, available on-demand"
    interactivity: "Q&A, polls, live demo"
```

### MK16: Demo Scripts
- [ ] **5-min Quick Demo**: Elevator pitch + one validation
- [ ] **15-min Standard Demo**: Full flow, key features
- [ ] **30-min Deep Dive**: Technical integration, customization
- [ ] **Custom Demo**: Tailored to prospect's use case

---

# PARTE 17: SYSTEM DIAGRAMS (HTML Human-Friendly)

> **Objetivo**: Diagramas de arquitetura em múltiplos níveis, todos em HTML interativo
> **Formato**: Standalone HTML files, mobile-friendly, dark mode

---

## 17.1 DIAGRAM 1: HIGH-LEVEL OVERVIEW

### D1: System Overview Diagram
```html
<!-- Arquivo: diagrams/01_system_overview.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Human Layer - System Overview</title>
    <style>
        :root {
            --bg: #0a0a0f;
            --text: #f0f0f5;
            --primary: #3b82f6;
            --secondary: #10b981;
            --accent: #8b5cf6;
            --border: #2a2a3a;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', system-ui, sans-serif;
            background: var(--bg);
            color: var(--text);
            padding: 2rem;
            min-height: 100vh;
        }
        .diagram-container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            margin-bottom: 2rem;
            font-size: 2rem;
            background: linear-gradient(90deg, var(--primary), var(--accent));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .flow {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            flex-wrap: wrap;
        }
        .box {
            background: linear-gradient(135deg, #1a1a24, #12121a);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            min-width: 200px;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        .box:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 30px rgba(59, 130, 246, 0.2);
            border-color: var(--primary);
        }
        .box h3 { color: var(--primary); margin-bottom: 0.5rem; }
        .box p { font-size: 0.9rem; opacity: 0.8; }
        .arrow {
            font-size: 2rem;
            color: var(--primary);
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .legend {
            margin-top: 2rem;
            padding: 1rem;
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
        }
        .legend h4 { margin-bottom: 0.5rem; }
        .legend-item { display: flex; align-items: center; gap: 0.5rem; margin: 0.25rem 0; }
        .dot { width: 12px; height: 12px; border-radius: 50%; }
        .dot.input { background: var(--secondary); }
        .dot.process { background: var(--primary); }
        .dot.output { background: var(--accent); }
    </style>
</head>
<body>
    <div class="diagram-container">
        <h1>🛡️ Human Layer - System Overview</h1>

        <div class="flow">
            <div class="box" style="border-color: var(--secondary);">
                <h3>📥 Input</h3>
                <p>AI Agent Output</p>
                <small>(Code, Text, Decision)</small>
            </div>

            <span class="arrow">→</span>

            <div class="box">
                <h3>🔄 MCP Server</h3>
                <p>Human Layer Engine</p>
                <small>Orchestration</small>
            </div>

            <span class="arrow">→</span>

            <div class="box">
                <h3>📊 7 Layers</h3>
                <p>Validation Pipeline</p>
                <small>HL-1 to HL-7</small>
            </div>

            <span class="arrow">→</span>

            <div class="box">
                <h3>🔍 6 Perspectives</h3>
                <p>Multi-Angle Review</p>
                <small>Personas</small>
            </div>

            <span class="arrow">→</span>

            <div class="box">
                <h3>🤝 Consensus</h3>
                <p>2/3 Agreement</p>
                <small>Triple Redundancy</small>
            </div>

            <span class="arrow">→</span>

            <div class="box" style="border-color: var(--accent);">
                <h3>📋 Output</h3>
                <p>Validation Report</p>
                <small>PASS / WARN / FAIL</small>
            </div>
        </div>

        <div class="legend">
            <h4>Legend</h4>
            <div class="legend-item"><span class="dot input"></span> Input/Output</div>
            <div class="legend-item"><span class="dot process"></span> Processing</div>
            <div class="legend-item"><span class="dot output"></span> Decision</div>
        </div>
    </div>
</body>
</html>
```

## 17.2 DIAGRAM 2: 7 HUMAN LAYERS

### D2: Layer Breakdown Diagram
```html
<!-- Arquivo: diagrams/02_seven_layers.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Human Layer - 7 Layers Breakdown</title>
    <style>
        :root {
            --bg: #0a0a0f;
            --text: #f0f0f5;
            --l1: #ef4444; /* UX - Red */
            --l2: #f97316; /* Functionality - Orange */
            --l3: #eab308; /* Edge Cases - Yellow */
            --l4: #22c55e; /* Security - Green */
            --l5: #06b6d4; /* Performance - Cyan */
            --l6: #3b82f6; /* Compliance - Blue */
            --l7: #8b5cf6; /* Final Review - Purple */
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', system-ui, sans-serif;
            background: var(--bg);
            color: var(--text);
            padding: 2rem;
        }
        .container { max-width: 1000px; margin: 0 auto; }
        h1 {
            text-align: center;
            margin-bottom: 2rem;
            font-size: 1.8rem;
        }
        .layers {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        .layer {
            display: grid;
            grid-template-columns: 80px 1fr 120px;
            align-items: center;
            padding: 1rem 1.5rem;
            border-radius: 12px;
            background: linear-gradient(90deg, rgba(255,255,255,0.05), transparent);
            border-left: 4px solid;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        .layer:hover {
            transform: translateX(8px);
            background: linear-gradient(90deg, rgba(255,255,255,0.1), transparent);
        }
        .layer-num {
            font-size: 1.5rem;
            font-weight: bold;
        }
        .layer-info h3 { margin-bottom: 0.25rem; }
        .layer-info p { font-size: 0.9rem; opacity: 0.7; }
        .veto {
            text-align: right;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 500;
        }
        .veto-strong { background: rgba(239, 68, 68, 0.2); color: #ef4444; }
        .veto-medium { background: rgba(249, 115, 22, 0.2); color: #f97316; }
        .veto-weak { background: rgba(59, 130, 246, 0.2); color: #3b82f6; }

        .l1 { border-color: var(--l1); }
        .l2 { border-color: var(--l2); }
        .l3 { border-color: var(--l3); }
        .l4 { border-color: var(--l4); }
        .l5 { border-color: var(--l5); }
        .l6 { border-color: var(--l6); }
        .l7 { border-color: var(--l7); }

        .l1 .layer-num { color: var(--l1); }
        .l2 .layer-num { color: var(--l2); }
        .l3 .layer-num { color: var(--l3); }
        .l4 .layer-num { color: var(--l4); }
        .l5 .layer-num { color: var(--l5); }
        .l6 .layer-num { color: var(--l6); }
        .l7 .layer-num { color: var(--l7); }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 7 Human Layers</h1>

        <div class="layers">
            <div class="layer l1">
                <span class="layer-num">HL-1</span>
                <div class="layer-info">
                    <h3>UX & Usability</h3>
                    <p>User experience, accessibility, intuitive design</p>
                </div>
                <span class="veto veto-weak">WEAK Veto</span>
            </div>

            <div class="layer l2">
                <span class="layer-num">HL-2</span>
                <div class="layer-info">
                    <h3>Functionality</h3>
                    <p>Features work as expected, business logic correct</p>
                </div>
                <span class="veto veto-medium">MEDIUM Veto</span>
            </div>

            <div class="layer l3">
                <span class="layer-num">HL-3</span>
                <div class="layer-info">
                    <h3>Edge Cases</h3>
                    <p>Unusual inputs, boundary conditions, error handling</p>
                </div>
                <span class="veto veto-medium">MEDIUM Veto</span>
            </div>

            <div class="layer l4">
                <span class="layer-num">HL-4</span>
                <div class="layer-info">
                    <h3>Security</h3>
                    <p>Vulnerabilities, injection, authentication</p>
                </div>
                <span class="veto veto-strong">STRONG Veto</span>
            </div>

            <div class="layer l5">
                <span class="layer-num">HL-5</span>
                <div class="layer-info">
                    <h3>Performance</h3>
                    <p>Speed, memory, scalability, efficiency</p>
                </div>
                <span class="veto veto-weak">WEAK Veto</span>
            </div>

            <div class="layer l6">
                <span class="layer-num">HL-6</span>
                <div class="layer-info">
                    <h3>Compliance</h3>
                    <p>Regulations, policies, standards adherence</p>
                </div>
                <span class="veto veto-strong">STRONG Veto</span>
            </div>

            <div class="layer l7">
                <span class="layer-num">HL-7</span>
                <div class="layer-info">
                    <h3>Final Human Review</h3>
                    <p>Holistic assessment, common sense check</p>
                </div>
                <span class="veto veto-strong">STRONG Veto</span>
            </div>
        </div>
    </div>
</body>
</html>
```

## 17.3 DIAGRAM 3: CONSENSUS MECHANISM

### D3: Triple Redundancy & Consensus
```html
<!-- Arquivo: diagrams/03_consensus_mechanism.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Human Layer - Consensus Mechanism</title>
    <style>
        :root {
            --bg: #0a0a0f;
            --text: #f0f0f5;
            --primary: #3b82f6;
            --success: #22c55e;
            --warning: #f59e0b;
            --error: #ef4444;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', system-ui, sans-serif;
            background: var(--bg);
            color: var(--text);
            padding: 2rem;
        }
        .container { max-width: 900px; margin: 0 auto; }
        h1 { text-align: center; margin-bottom: 0.5rem; }
        .subtitle { text-align: center; opacity: 0.7; margin-bottom: 2rem; }

        .consensus-flow {
            display: flex;
            flex-direction: column;
            gap: 2rem;
        }
        .stage {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 16px;
            padding: 1.5rem;
        }
        .stage h3 {
            color: var(--primary);
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .runs {
            display: flex;
            justify-content: center;
            gap: 2rem;
            margin: 1rem 0;
        }
        .run {
            width: 100px;
            height: 100px;
            border-radius: 50%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            border: 3px solid;
            transition: all 0.3s ease;
        }
        .run:hover { transform: scale(1.1); }
        .run.pass { border-color: var(--success); background: rgba(34, 197, 94, 0.1); }
        .run.fail { border-color: var(--error); background: rgba(239, 68, 68, 0.1); }
        .run span { font-size: 1.5rem; }
        .run small { font-size: 0.8rem; opacity: 0.7; }

        .result {
            text-align: center;
            padding: 1rem;
            border-radius: 12px;
            font-size: 1.2rem;
            font-weight: 600;
        }
        .result.pass { background: rgba(34, 197, 94, 0.2); color: var(--success); }
        .result.fail { background: rgba(239, 68, 68, 0.2); color: var(--error); }

        .formula {
            background: rgba(59, 130, 246, 0.1);
            border: 1px solid var(--primary);
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
            font-family: monospace;
            font-size: 1.1rem;
        }
        .examples {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }
        .example {
            padding: 1rem;
            border-radius: 8px;
            background: rgba(255,255,255,0.03);
        }
        .example h4 { margin-bottom: 0.5rem; font-size: 0.9rem; }
        .example-votes { display: flex; gap: 0.5rem; margin-bottom: 0.5rem; }
        .vote { padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.8rem; }
        .vote.p { background: rgba(34, 197, 94, 0.2); color: var(--success); }
        .vote.f { background: rgba(239, 68, 68, 0.2); color: var(--error); }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤝 Consensus Mechanism</h1>
        <p class="subtitle">Triple Redundancy with 2/3 Agreement Required</p>

        <div class="consensus-flow">
            <div class="stage">
                <h3>📌 Step 1: Triple Execution</h3>
                <p>Each layer runs 3 times independently</p>
                <div class="runs">
                    <div class="run pass">
                        <span>Run 1</span>
                        <small>Independent</small>
                    </div>
                    <div class="run pass">
                        <span>Run 2</span>
                        <small>Independent</small>
                    </div>
                    <div class="run fail">
                        <span>Run 3</span>
                        <small>Independent</small>
                    </div>
                </div>
            </div>

            <div class="stage">
                <h3>🧮 Step 2: Consensus Calculation</h3>
                <div class="formula">
                    PASS if (pass_count / total_runs) ≥ 0.67 (2/3)
                </div>
                <div class="examples">
                    <div class="example">
                        <h4>✅ Consensus: PASS</h4>
                        <div class="example-votes">
                            <span class="vote p">PASS</span>
                            <span class="vote p">PASS</span>
                            <span class="vote f">FAIL</span>
                        </div>
                        <small>2/3 = 67% ≥ 67% → PASS</small>
                    </div>
                    <div class="example">
                        <h4>✅ Consensus: PASS</h4>
                        <div class="example-votes">
                            <span class="vote p">PASS</span>
                            <span class="vote p">PASS</span>
                            <span class="vote p">PASS</span>
                        </div>
                        <small>3/3 = 100% ≥ 67% → PASS</small>
                    </div>
                    <div class="example">
                        <h4>❌ Consensus: FAIL</h4>
                        <div class="example-votes">
                            <span class="vote p">PASS</span>
                            <span class="vote f">FAIL</span>
                            <span class="vote f">FAIL</span>
                        </div>
                        <small>1/3 = 33% < 67% → FAIL</small>
                    </div>
                    <div class="example">
                        <h4>❌ Consensus: FAIL</h4>
                        <div class="example-votes">
                            <span class="vote f">FAIL</span>
                            <span class="vote f">FAIL</span>
                            <span class="vote f">FAIL</span>
                        </div>
                        <small>0/3 = 0% < 67% → FAIL</small>
                    </div>
                </div>
            </div>

            <div class="stage">
                <h3>📊 Step 3: Final Result</h3>
                <div class="result pass">
                    ✅ LAYER PASSED (2/3 Consensus Achieved)
                </div>
            </div>
        </div>
    </div>
</body>
</html>
```

## 17.4 DIAGRAM 4: VETO SYSTEM

### D4: Veto Power Flow
```html
<!-- Arquivo: diagrams/04_veto_system.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Human Layer - Veto System</title>
    <style>
        :root {
            --bg: #0a0a0f;
            --text: #f0f0f5;
            --strong: #ef4444;
            --medium: #f59e0b;
            --weak: #3b82f6;
            --none: #6b7280;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', system-ui, sans-serif;
            background: var(--bg);
            color: var(--text);
            padding: 2rem;
        }
        .container { max-width: 1000px; margin: 0 auto; }
        h1 { text-align: center; margin-bottom: 2rem; }

        .veto-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        .veto-card {
            padding: 1.5rem;
            border-radius: 16px;
            border: 2px solid;
            text-align: center;
        }
        .veto-card.strong {
            border-color: var(--strong);
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), transparent);
        }
        .veto-card.medium {
            border-color: var(--medium);
            background: linear-gradient(135deg, rgba(249, 115, 22, 0.1), transparent);
        }
        .veto-card.weak {
            border-color: var(--weak);
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), transparent);
        }
        .veto-card.none {
            border-color: var(--none);
            background: linear-gradient(135deg, rgba(107, 114, 128, 0.1), transparent);
        }
        .veto-icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
        .veto-name { font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem; }
        .veto-desc { font-size: 0.9rem; opacity: 0.7; margin-bottom: 1rem; }
        .veto-power { font-size: 0.8rem; padding: 0.5rem; background: rgba(0,0,0,0.3); border-radius: 8px; }

        .strong .veto-name { color: var(--strong); }
        .medium .veto-name { color: var(--medium); }
        .weak .veto-name { color: var(--weak); }
        .none .veto-name { color: var(--none); }

        .flow-section {
            background: rgba(255,255,255,0.03);
            border-radius: 16px;
            padding: 1.5rem;
        }
        .flow-section h3 { margin-bottom: 1rem; }
        .decision-flow {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        .decision {
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 1rem;
            border-radius: 8px;
            background: rgba(255,255,255,0.05);
        }
        .decision-condition { flex: 1; }
        .decision-arrow { font-size: 1.5rem; }
        .decision-result {
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-weight: 500;
        }
        .decision-result.block { background: rgba(239, 68, 68, 0.2); color: var(--strong); }
        .decision-result.warn { background: rgba(249, 115, 22, 0.2); color: var(--medium); }
        .decision-result.pass { background: rgba(34, 197, 94, 0.2); color: #22c55e; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚫 Veto Power System</h1>

        <div class="veto-grid">
            <div class="veto-card strong">
                <div class="veto-icon">🛑</div>
                <div class="veto-name">STRONG</div>
                <div class="veto-desc">Blocks everything</div>
                <div class="veto-power">
                    Can block: Merge, Deploy, Promotion
                </div>
            </div>

            <div class="veto-card medium">
                <div class="veto-icon">⚠️</div>
                <div class="veto-name">MEDIUM</div>
                <div class="veto-desc">Blocks merge</div>
                <div class="veto-power">
                    Can block: Merge<br>
                    Allows: Local testing
                </div>
            </div>

            <div class="veto-card weak">
                <div class="veto-icon">💬</div>
                <div class="veto-name">WEAK</div>
                <div class="veto-desc">Adds warnings</div>
                <div class="veto-power">
                    Can add: Warnings<br>
                    Allows: Proceed with caution
                </div>
            </div>

            <div class="veto-card none">
                <div class="veto-icon">ℹ️</div>
                <div class="veto-name">NONE</div>
                <div class="veto-desc">Informational only</div>
                <div class="veto-power">
                    Output: Informational<br>
                    No blocking power
                </div>
            </div>
        </div>

        <div class="flow-section">
            <h3>Decision Flow</h3>
            <div class="decision-flow">
                <div class="decision">
                    <div class="decision-condition">
                        <strong>IF</strong> any layer with STRONG veto fails
                    </div>
                    <span class="decision-arrow">→</span>
                    <div class="decision-result block">🛑 BLOCKED</div>
                </div>
                <div class="decision">
                    <div class="decision-condition">
                        <strong>ELSE IF</strong> any layer with MEDIUM veto fails
                    </div>
                    <span class="decision-arrow">→</span>
                    <div class="decision-result warn">⚠️ MERGE BLOCKED</div>
                </div>
                <div class="decision">
                    <div class="decision-condition">
                        <strong>ELSE IF</strong> any layer with WEAK veto fails
                    </div>
                    <span class="decision-arrow">→</span>
                    <div class="decision-result warn">⚠️ WARN + PROCEED</div>
                </div>
                <div class="decision">
                    <div class="decision-condition">
                        <strong>ELSE</strong> all layers pass or only INFO findings
                    </div>
                    <span class="decision-arrow">→</span>
                    <div class="decision-result pass">✅ APPROVED</div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
```

## 17.5 DIAGRAM 5: ARCHITECTURE LAYERS

### D5: Technical Architecture
```html
<!-- Arquivo: diagrams/05_technical_architecture.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Human Layer - Technical Architecture</title>
    <style>
        :root {
            --bg: #0a0a0f;
            --text: #f0f0f5;
            --primary: #3b82f6;
            --secondary: #10b981;
            --accent: #8b5cf6;
            --border: #2a2a3a;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', system-ui, sans-serif;
            background: var(--bg);
            color: var(--text);
            padding: 2rem;
        }
        .container { max-width: 1100px; margin: 0 auto; }
        h1 { text-align: center; margin-bottom: 0.5rem; }
        .subtitle { text-align: center; opacity: 0.7; margin-bottom: 2rem; }

        .architecture {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        .tier {
            display: grid;
            grid-template-columns: 120px 1fr;
            gap: 1rem;
            align-items: center;
        }
        .tier-label {
            writing-mode: vertical-rl;
            text-orientation: mixed;
            transform: rotate(180deg);
            font-weight: 600;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 2px;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
        }
        .tier-content {
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
        }
        .component {
            padding: 1rem 1.5rem;
            border-radius: 12px;
            border: 1px solid var(--border);
            background: linear-gradient(135deg, rgba(255,255,255,0.05), transparent);
            min-width: 150px;
            transition: all 0.3s ease;
        }
        .component:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        }
        .component h4 { font-size: 0.9rem; margin-bottom: 0.25rem; }
        .component p { font-size: 0.75rem; opacity: 0.7; }
        .component .tech { font-size: 0.7rem; color: var(--primary); margin-top: 0.5rem; }

        .t-client .tier-label { background: rgba(139, 92, 246, 0.2); color: var(--accent); }
        .t-api .tier-label { background: rgba(59, 130, 246, 0.2); color: var(--primary); }
        .t-core .tier-label { background: rgba(16, 185, 129, 0.2); color: var(--secondary); }
        .t-infra .tier-label { background: rgba(107, 114, 128, 0.2); color: #9ca3af; }

        .connections {
            margin-top: 2rem;
            padding: 1rem;
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
        }
        .connections h3 { margin-bottom: 1rem; font-size: 1rem; }
        .conn-list { display: flex; flex-wrap: wrap; gap: 1rem; }
        .conn {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.85rem;
            padding: 0.5rem 1rem;
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
        }
        .conn-arrow { color: var(--primary); }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏗️ Technical Architecture</h1>
        <p class="subtitle">Layered architecture for Human Layer MCP Server</p>

        <div class="architecture">
            <div class="tier t-client">
                <div class="tier-label">Clients</div>
                <div class="tier-content">
                    <div class="component">
                        <h4>CLI</h4>
                        <p>Command line interface</p>
                        <div class="tech">Python (Typer)</div>
                    </div>
                    <div class="component">
                        <h4>SDK</h4>
                        <p>Python/JS libraries</p>
                        <div class="tech">PyPI, npm</div>
                    </div>
                    <div class="component">
                        <h4>Dashboard</h4>
                        <p>Web UI (Cloud)</p>
                        <div class="tech">Next.js 14</div>
                    </div>
                    <div class="component">
                        <h4>CI/CD Plugins</h4>
                        <p>GitHub, GitLab, etc.</p>
                        <div class="tech">Actions, Webhooks</div>
                    </div>
                </div>
            </div>

            <div class="tier t-api">
                <div class="tier-label">API Layer</div>
                <div class="tier-content">
                    <div class="component">
                        <h4>MCP Server</h4>
                        <p>Protocol implementation</p>
                        <div class="tech">Python (mcp-sdk)</div>
                    </div>
                    <div class="component">
                        <h4>REST API</h4>
                        <p>Cloud API endpoints</p>
                        <div class="tech">FastAPI</div>
                    </div>
                    <div class="component">
                        <h4>WebSocket</h4>
                        <p>Real-time updates</p>
                        <div class="tech">Socket.IO</div>
                    </div>
                </div>
            </div>

            <div class="tier t-core">
                <div class="tier-label">Core Engine</div>
                <div class="tier-content">
                    <div class="component">
                        <h4>Orchestrator</h4>
                        <p>Layer coordination</p>
                        <div class="tech">HumanLayerRunner</div>
                    </div>
                    <div class="component">
                        <h4>7 Layers</h4>
                        <p>Validation logic</p>
                        <div class="tech">HL-1 to HL-7</div>
                    </div>
                    <div class="component">
                        <h4>6 Perspectives</h4>
                        <p>Multi-angle review</p>
                        <div class="tech">Persona classes</div>
                    </div>
                    <div class="component">
                        <h4>Consensus</h4>
                        <p>2/3 agreement</p>
                        <div class="tech">ConsensusEngine</div>
                    </div>
                    <div class="component">
                        <h4>Veto Gate</h4>
                        <p>Final decision</p>
                        <div class="tech">VetoGate</div>
                    </div>
                </div>
            </div>

            <div class="tier t-infra">
                <div class="tier-label">Infrastructure</div>
                <div class="tier-content">
                    <div class="component">
                        <h4>LLM Clients</h4>
                        <p>Claude, GPT, etc.</p>
                        <div class="tech">BYOK</div>
                    </div>
                    <div class="component">
                        <h4>Database</h4>
                        <p>Persistence</p>
                        <div class="tech">PostgreSQL</div>
                    </div>
                    <div class="component">
                        <h4>Cache</h4>
                        <p>Performance</p>
                        <div class="tech">Redis</div>
                    </div>
                    <div class="component">
                        <h4>Queue</h4>
                        <p>Async processing</p>
                        <div class="tech">Redis Streams</div>
                    </div>
                </div>
            </div>
        </div>

        <div class="connections">
            <h3>Data Flow</h3>
            <div class="conn-list">
                <div class="conn">Client <span class="conn-arrow">→</span> MCP Server</div>
                <div class="conn">MCP Server <span class="conn-arrow">→</span> Orchestrator</div>
                <div class="conn">Orchestrator <span class="conn-arrow">→</span> Layers</div>
                <div class="conn">Layers <span class="conn-arrow">→</span> LLM</div>
                <div class="conn">Layers <span class="conn-arrow">→</span> Consensus</div>
                <div class="conn">Consensus <span class="conn-arrow">→</span> Veto Gate</div>
                <div class="conn">Veto Gate <span class="conn-arrow">→</span> Report</div>
            </div>
        </div>
    </div>
</body>
</html>
```

## 17.6 DIAGRAM 6: INTEGRATION FLOW

### D6: CI/CD Integration
```html
<!-- Arquivo: diagrams/06_cicd_integration.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Human Layer - CI/CD Integration</title>
    <style>
        :root {
            --bg: #0a0a0f;
            --text: #f0f0f5;
            --github: #238636;
            --gitlab: #fc6d26;
            --human: #3b82f6;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', system-ui, sans-serif;
            background: var(--bg);
            color: var(--text);
            padding: 2rem;
        }
        .container { max-width: 900px; margin: 0 auto; }
        h1 { text-align: center; margin-bottom: 2rem; }

        .pipeline {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }
        .step {
            display: grid;
            grid-template-columns: 40px 1fr 150px;
            align-items: center;
            gap: 1rem;
            padding: 1rem;
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
            border-left: 4px solid var(--github);
        }
        .step.human { border-color: var(--human); }
        .step-num {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            background: rgba(255,255,255,0.1);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
        }
        .step-content h4 { margin-bottom: 0.25rem; }
        .step-content p { font-size: 0.85rem; opacity: 0.7; }
        .step-badge {
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.75rem;
            text-align: center;
        }
        .step-badge.ci { background: rgba(35, 134, 54, 0.2); color: var(--github); }
        .step-badge.hl { background: rgba(59, 130, 246, 0.2); color: var(--human); }

        .connector {
            height: 24px;
            width: 2px;
            background: rgba(255,255,255,0.1);
            margin-left: 55px;
        }

        .code-example {
            margin-top: 2rem;
            padding: 1rem;
            background: #1a1a24;
            border-radius: 12px;
            overflow-x: auto;
        }
        .code-example h3 { margin-bottom: 1rem; font-size: 0.9rem; }
        pre {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            line-height: 1.5;
        }
        .comment { color: #6b7280; }
        .key { color: #3b82f6; }
        .string { color: #10b981; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔄 CI/CD Integration Flow</h1>

        <div class="pipeline">
            <div class="step">
                <div class="step-num">1</div>
                <div class="step-content">
                    <h4>Push / PR</h4>
                    <p>Developer pushes code or opens PR</p>
                </div>
                <span class="step-badge ci">GitHub</span>
            </div>
            <div class="connector"></div>

            <div class="step">
                <div class="step-num">2</div>
                <div class="step-content">
                    <h4>CI Triggered</h4>
                    <p>GitHub Actions workflow starts</p>
                </div>
                <span class="step-badge ci">GitHub</span>
            </div>
            <div class="connector"></div>

            <div class="step">
                <div class="step-num">3</div>
                <div class="step-content">
                    <h4>Standard Tests</h4>
                    <p>Unit tests, linting, build</p>
                </div>
                <span class="step-badge ci">GitHub</span>
            </div>
            <div class="connector"></div>

            <div class="step human">
                <div class="step-num">4</div>
                <div class="step-content">
                    <h4>Human Layer Validation</h4>
                    <p>7 layers + 6 perspectives + consensus</p>
                </div>
                <span class="step-badge hl">Human Layer</span>
            </div>
            <div class="connector"></div>

            <div class="step human">
                <div class="step-num">5</div>
                <div class="step-content">
                    <h4>Veto Check</h4>
                    <p>PASS / WARN / FAIL decision</p>
                </div>
                <span class="step-badge hl">Human Layer</span>
            </div>
            <div class="connector"></div>

            <div class="step">
                <div class="step-num">6</div>
                <div class="step-content">
                    <h4>PR Status Update</h4>
                    <p>Check status + comment with report</p>
                </div>
                <span class="step-badge ci">GitHub</span>
            </div>
            <div class="connector"></div>

            <div class="step">
                <div class="step-num">7</div>
                <div class="step-content">
                    <h4>Merge / Block</h4>
                    <p>Based on veto level</p>
                </div>
                <span class="step-badge ci">GitHub</span>
            </div>
        </div>

        <div class="code-example">
            <h3>📄 .github/workflows/human-layer.yml</h3>
            <pre>
<span class="key">name:</span> <span class="string">Human Layer Validation</span>
<span class="key">on:</span> [push, pull_request]

<span class="key">jobs:</span>
  <span class="key">validate:</span>
    <span class="key">runs-on:</span> <span class="string">ubuntu-latest</span>
    <span class="key">steps:</span>
      - <span class="key">uses:</span> <span class="string">actions/checkout@v4</span>

      - <span class="key">name:</span> <span class="string">Run Human Layer</span>
        <span class="key">uses:</span> <span class="string">humangr/human-layer-action@v1</span>
        <span class="key">with:</span>
          <span class="key">layers:</span> <span class="string">"1,2,3,4,5,6,7"</span>
          <span class="key">perspectives:</span> <span class="string">"all"</span>
          <span class="key">fail-on:</span> <span class="string">"STRONG"</span>
        <span class="key">env:</span>
          <span class="key">ANTHROPIC_API_KEY:</span> <span class="string">${{ secrets.ANTHROPIC_API_KEY }}</span>
</pre>
        </div>
    </div>
</body>
</html>
```

### D7: Additional Diagrams
- [ ] **07_data_flow.html**: Complete data flow from input to output
- [ ] **08_deployment_options.html**: OSS vs Cloud deployment
- [ ] **09_security_model.html**: Security architecture
- [ ] **10_scaling_architecture.html**: Horizontal scaling diagram

---

# PARTE 18: GITHUB PROFILE HUMANGR (OSS Presentation)

> **Objetivo**: Perfil GitHub premium para a organização HumanGR
> **Público**: Desenvolvedores OSS, contribuidores, avaliadores

---

## 18.1 ORGANIZATION PROFILE

### GH1: Organization README
```markdown
<!-- .github/profile/README.md -->

<div align="center">
  <img src="./assets/humangr-logo.svg" alt="HumanGR Logo" width="200" />

  # HumanGR

  **Bringing Human Judgment to AI Agent Validation**

  [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
  [![Discord](https://img.shields.io/discord/1234567890?color=7289da&label=Discord&logo=discord)](https://discord.gg/humangr)
  [![Twitter](https://img.shields.io/twitter/follow/humangr?style=social)](https://twitter.com/humangr)

  [Website](https://humangr.ai) •
  [Documentation](https://docs.humangr.ai) •
  [Blog](https://humangr.ai/blog) •
  [Discord](https://discord.gg/humangr)

</div>

---

## 🛡️ Our Mission

We're building the missing validation layer for AI agents. As AI becomes more autonomous,
the need for structured human oversight grows. Human Layer provides a standardized,
comprehensive validation protocol that brings human judgment to AI decisions.

## 🚀 Our Products

### Human Layer MCP Server

The core open-source project. A Model Context Protocol (MCP) server that validates
AI agent outputs through 7 human-like validation layers.

- **7 Human Layers**: UX, Functionality, Edge Cases, Security, Performance, Compliance, Final Review
- **6 Perspectives**: Simulated user personas for multi-angle validation
- **Triple Redundancy**: 2/3 consensus required for decisions
- **BYOK**: Bring Your Own Key - use your own LLM subscription

```bash
# Quick start
pip install human-layer
human-layer init
human-layer validate ./my-agent-output
```

### Human Layer Cloud (Coming Soon)

Premium cloud features for teams and enterprises.

- Real-time dashboard
- Team collaboration
- Advanced analytics
- CI/CD integrations
- Priority support

## 📊 Project Stats

| Repository | Stars | Downloads | Contributors |
|------------|-------|-----------|--------------|
| [human-layer](https://github.com/humangr/human-layer) | ![Stars](https://img.shields.io/github/stars/humangr/human-layer) | ![Downloads](https://img.shields.io/pypi/dm/human-layer) | ![Contributors](https://img.shields.io/github/contributors/humangr/human-layer) |

## 🤝 Contributing

We love contributions! Check out our [Contributing Guide](CONTRIBUTING.md) to get started.

### Good First Issues

Looking to contribute? Check out issues labeled `good-first-issue` in our repositories.

### Community

- **Discord**: Join our [Discord server](https://discord.gg/humangr) for discussions
- **GitHub Discussions**: For longer-form conversations
- **Twitter**: Follow [@humangr](https://twitter.com/humangr) for updates

## 📜 License

Our core products are open source under the [Apache 2.0 License](LICENSE).

---

<div align="center">
  <sub>Built with ❤️ by the HumanGR team</sub>
</div>
```

## 18.2 MAIN REPOSITORY PROFILE

### GH2: human-layer Repository README
```markdown
<!-- human-layer/README.md -->

<div align="center">
  <img src="./docs/assets/human-layer-banner.svg" alt="Human Layer" width="600" />

  # Human Layer

  **7 Layers of Human Judgment for AI Agent Validation**

  [![PyPI](https://img.shields.io/pypi/v/human-layer)](https://pypi.org/project/human-layer/)
  [![Python](https://img.shields.io/pypi/pyversions/human-layer)](https://pypi.org/project/human-layer/)
  [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
  [![CI](https://github.com/humangr/human-layer/actions/workflows/ci.yml/badge.svg)](https://github.com/humangr/human-layer/actions)
  [![Coverage](https://img.shields.io/codecov/c/github/humangr/human-layer)](https://codecov.io/gh/humangr/human-layer)
  [![Discord](https://img.shields.io/discord/1234567890?color=7289da&label=Discord)](https://discord.gg/humangr)

  [Quick Start](#-quick-start) •
  [Documentation](https://docs.humangr.ai) •
  [Examples](./examples) •
  [Contributing](CONTRIBUTING.md)

</div>

---

## 🤔 What is Human Layer?

Human Layer is an MCP (Model Context Protocol) server that validates AI agent outputs
through a structured 7-layer validation process, bringing human-like judgment to AI decisions.

### The Problem

AI agents are increasingly autonomous, but they can:
- Produce harmful or incorrect outputs
- Miss edge cases and security vulnerabilities
- Lack the common sense of human review

### The Solution

Human Layer provides:

| Layer | Focus | Veto Power |
|-------|-------|------------|
| HL-1 | UX & Usability | Weak |
| HL-2 | Functionality | Medium |
| HL-3 | Edge Cases | Medium |
| HL-4 | Security | **Strong** |
| HL-5 | Performance | Weak |
| HL-6 | Compliance | **Strong** |
| HL-7 | Final Review | **Strong** |

Plus **6 Perspectives** (simulated user personas) and **Triple Redundancy** (2/3 consensus).

---

## ✨ Features

- 🔍 **7 Human Layers** - Comprehensive validation coverage
- 👥 **6 Perspectives** - Multi-angle review (tired user, malicious insider, etc.)
- 🔄 **Triple Redundancy** - 2/3 consensus for reliable decisions
- 🚫 **Veto System** - Configurable blocking power per layer
- 🔑 **BYOK** - Bring Your Own Key (use your LLM subscription)
- 📊 **Detailed Reports** - Actionable findings with suggestions
- 🔌 **MCP Protocol** - Standard integration with AI tools
- 🛠️ **CI/CD Ready** - GitHub Actions, GitLab CI, etc.

---

## 🚀 Quick Start

### Installation

```bash
pip install human-layer
```

### Basic Usage

```python
from human_layer import HumanLayerRunner

# Initialize with your LLM key
runner = HumanLayerRunner(
    llm_provider="anthropic",  # or "openai", "gemini", "ollama"
    api_key="your-api-key"     # or use ANTHROPIC_API_KEY env var
)

# Validate AI agent output
result = await runner.validate(
    target="./agent_output.txt",
    layers=[1, 2, 3, 4, 5, 6, 7],  # All 7 layers
    perspectives="all"
)

# Check result
if result.approved:
    print("✅ Validation passed!")
else:
    print(f"❌ Blocked by: {result.blocking_layers}")
    for finding in result.findings:
        print(f"  - [{finding.severity}] {finding.description}")
```

### CLI Usage

```bash
# Initialize configuration
human-layer init

# Validate a file
human-layer validate ./my-output.txt

# Validate with specific layers
human-layer validate ./code.py --layers 1,4,7

# Output JSON report
human-layer validate ./code.py --output report.json
```

### MCP Server

```bash
# Start the MCP server
human-layer serve

# Or with Docker
docker run -p 8080:8080 humangr/human-layer
```

---

## 📖 Documentation

- [Getting Started](https://docs.humangr.ai/getting-started)
- [Configuration](https://docs.humangr.ai/configuration)
- [7 Layers Explained](https://docs.humangr.ai/layers)
- [6 Perspectives](https://docs.humangr.ai/perspectives)
- [CI/CD Integration](https://docs.humangr.ai/cicd)
- [API Reference](https://docs.humangr.ai/api)

---

## 🔧 Configuration

Create `human-layer.yaml`:

```yaml
# LLM Configuration
llm:
  provider: anthropic
  model: claude-3-sonnet

# Layer Configuration
layers:
  enabled: [1, 2, 3, 4, 5, 6, 7]
  redundancy: 3  # Triple redundancy
  consensus_threshold: 0.67  # 2/3 agreement

# Veto Configuration
veto:
  strong_blocks: [merge, deploy, promote]
  medium_blocks: [merge]
  weak_blocks: []  # Warnings only

# Perspectives
perspectives:
  enabled: all  # or specific: [tired_user, malicious_insider]
```

---

## 🔌 Integrations

### GitHub Actions

```yaml
- uses: humangr/human-layer-action@v1
  with:
    layers: "1,2,3,4,5,6,7"
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

### GitLab CI

```yaml
human-layer:
  image: humangr/human-layer:latest
  script:
    - human-layer validate ./src
```

### Pre-commit Hook

```yaml
repos:
  - repo: https://github.com/humangr/human-layer
    rev: v1.0.0
    hooks:
      - id: human-layer
```

---

## 📊 Example Output

```
🛡️ Human Layer Validation Report
================================

Target: ./agent_code.py
Layers: 7/7 completed
Perspectives: 6/6 completed
Consensus: 2/3 required

Results:
├── HL-1 (UX): ✅ PASS
├── HL-2 (Functionality): ✅ PASS
├── HL-3 (Edge Cases): ⚠️ WARN (2 findings)
├── HL-4 (Security): ❌ FAIL (1 critical)
├── HL-5 (Performance): ✅ PASS
├── HL-6 (Compliance): ✅ PASS
└── HL-7 (Final Review): ⚠️ WARN

Findings:
┌────────────┬──────────┬─────────────────────────────────────┐
│ Severity   │ Layer    │ Description                         │
├────────────┼──────────┼─────────────────────────────────────┤
│ 🔴 CRITICAL│ HL-4     │ SQL injection vulnerability in      │
│            │          │ user input handling (line 42)       │
│ 🟡 MEDIUM  │ HL-3     │ Missing null check for optional     │
│            │          │ parameter 'config'                  │
│ 🟡 MEDIUM  │ HL-3     │ Integer overflow possible with      │
│            │          │ large input values                  │
│ 🔵 LOW     │ HL-7     │ Consider adding logging for         │
│            │          │ debugging purposes                  │
└────────────┴──────────┴─────────────────────────────────────┘

Decision: ❌ BLOCKED (Strong veto from HL-4)
```

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md).

### Development Setup

```bash
# Clone the repo
git clone https://github.com/humangr/human-layer.git
cd human-layer

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linter
ruff check .
```

### Good First Issues

Check out issues labeled [`good-first-issue`](https://github.com/humangr/human-layer/labels/good-first-issue).

---

## 🗺️ Roadmap

- [x] Core 7 layers implementation
- [x] 6 perspectives
- [x] CLI tool
- [x] MCP server
- [ ] GitHub Action (v1.1)
- [ ] VS Code extension (v1.2)
- [ ] Web dashboard (v2.0)
- [ ] Team features (v2.1)

See our [public roadmap](https://github.com/orgs/humangr/projects/1).

---

## 📜 License

Apache 2.0 - see [LICENSE](LICENSE) for details.

---

## 💬 Community

- [Discord](https://discord.gg/humangr) - Chat with the community
- [GitHub Discussions](https://github.com/humangr/human-layer/discussions) - Q&A
- [Twitter](https://twitter.com/humangr) - Updates

---

<div align="center">
  <sub>Made with ❤️ by <a href="https://humangr.ai">HumanGR</a></sub>
</div>
```

## 18.3 CONTRIBUTING GUIDE

### GH3: CONTRIBUTING.md
```markdown
# Contributing to Human Layer

First off, thank you for considering contributing to Human Layer! 🎉

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)
- [Recognition](#recognition)

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code.

## How Can I Contribute?

### 🐛 Reporting Bugs

1. Check if the bug is already reported in [Issues](https://github.com/humangr/human-layer/issues)
2. If not, create a new issue using the bug report template
3. Include:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (Python version, OS, etc.)

### 💡 Suggesting Features

1. Check [Discussions](https://github.com/humangr/human-layer/discussions) for existing ideas
2. Open a new discussion in the "Ideas" category
3. Describe the use case and proposed solution

### 🔧 Code Contributions

1. Look for issues labeled `good-first-issue` or `help-wanted`
2. Comment on the issue to claim it
3. Fork, code, and submit a PR

## Development Setup

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/human-layer.git
cd human-layer

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run linter
ruff check .

# Run type checker
mypy src/
```

## Pull Request Process

1. **Branch**: Create a branch from `main` with a descriptive name
   - `feature/add-new-layer`
   - `fix/consensus-calculation`
   - `docs/improve-readme`

2. **Develop**: Make your changes
   - Write tests for new functionality
   - Update documentation if needed
   - Follow style guidelines

3. **Test**: Ensure all tests pass
   ```bash
   pytest
   ruff check .
   mypy src/
   ```

4. **Commit**: Use conventional commits
   - `feat: add new perspective for QA engineers`
   - `fix: correct consensus threshold calculation`
   - `docs: update installation instructions`

5. **PR**: Open a pull request
   - Fill out the PR template
   - Link related issues
   - Request review

6. **Review**: Address feedback
   - Make requested changes
   - Re-request review when ready

## Style Guidelines

### Python

- Follow [PEP 8](https://pep8.org/)
- Use [Ruff](https://github.com/astral-sh/ruff) for linting
- Use type hints everywhere
- Docstrings in [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)

```python
def validate_layer(
    self,
    target: ValidationTarget,
    layer_id: LayerID,
) -> LayerResult:
    """Validate a target through a specific layer.

    Args:
        target: The validation target (code, text, etc.)
        layer_id: Which layer to use (1-7)

    Returns:
        LayerResult with findings and status

    Raises:
        ValidationError: If target format is invalid
    """
```

### Commits

- Use [Conventional Commits](https://www.conventionalcommits.org/)
- Present tense ("add feature" not "added feature")
- Imperative mood ("fix bug" not "fixes bug")

### Documentation

- Update docs for user-facing changes
- Include code examples
- Keep language clear and concise

## Recognition

Contributors are recognized in:
- [CONTRIBUTORS.md](CONTRIBUTORS.md)
- Release notes
- Our website's contributors page

Thank you for contributing! 🙏
```

## 18.4 ISSUE & PR TEMPLATES

### GH4: Issue Templates
```yaml
# .github/ISSUE_TEMPLATE/bug_report.yml
name: Bug Report
description: Report a bug in Human Layer
labels: ["bug", "triage"]
body:
  - type: markdown
    attributes:
      value: |
        Thanks for reporting a bug! Please fill out the sections below.

  - type: textarea
    id: description
    attributes:
      label: Bug Description
      description: A clear description of the bug
      placeholder: What happened?
    validations:
      required: true

  - type: textarea
    id: steps
    attributes:
      label: Steps to Reproduce
      description: How can we reproduce this?
      placeholder: |
        1. Install human-layer
        2. Run `human-layer validate ...`
        3. See error
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
    id: actual
    attributes:
      label: Actual Behavior
      description: What actually happened?
    validations:
      required: true

  - type: input
    id: version
    attributes:
      label: Human Layer Version
      placeholder: "1.0.0"
    validations:
      required: true

  - type: dropdown
    id: python
    attributes:
      label: Python Version
      options:
        - "3.9"
        - "3.10"
        - "3.11"
        - "3.12"
    validations:
      required: true

  - type: dropdown
    id: os
    attributes:
      label: Operating System
      options:
        - macOS
        - Linux
        - Windows
    validations:
      required: true

  - type: textarea
    id: logs
    attributes:
      label: Relevant Logs
      description: Paste any error messages or logs
      render: shell
```

### GH5: PR Template
```markdown
<!-- .github/PULL_REQUEST_TEMPLATE.md -->

## Description

<!-- What does this PR do? Why is this change needed? -->

## Related Issues

<!-- Link related issues: Fixes #123, Closes #456 -->

## Type of Change

- [ ] 🐛 Bug fix (non-breaking change that fixes an issue)
- [ ] ✨ New feature (non-breaking change that adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to change)
- [ ] 📝 Documentation update
- [ ] 🧹 Code refactoring (no functional changes)
- [ ] 🧪 Test update

## Checklist

- [ ] I have read the [Contributing Guide](CONTRIBUTING.md)
- [ ] My code follows the project's style guidelines
- [ ] I have added tests for my changes
- [ ] All new and existing tests pass
- [ ] I have updated documentation (if applicable)
- [ ] My commits follow the conventional commit format

## Screenshots (if applicable)

<!-- Add screenshots for UI changes -->

## Additional Notes

<!-- Any additional information reviewers should know -->
```

## 18.5 COMMUNITY FILES

### GH6: CODE_OF_CONDUCT.md
- [ ] Contributor Covenant v2.1
- [ ] Enforcement section
- [ ] Contact information

### GH7: SECURITY.md
```markdown
# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

Please report security vulnerabilities to: security@humangr.ai

Do NOT open a public issue for security vulnerabilities.

We will respond within 48 hours and work with you to understand and address the issue.

## Security Best Practices

When using Human Layer:
- Keep your API keys secure (use environment variables)
- Update to the latest version regularly
- Review validation outputs before acting on them
```

### GH8: FUNDING.yml
```yaml
# .github/FUNDING.yml
github: humangr
open_collective: humangr
custom: ["https://humangr.ai/sponsor"]
```

## 18.6 GITHUB ACTIONS

### GH9: CI Workflow
```yaml
# .github/workflows/ci.yml
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
          pip install -e ".[dev]"

      - name: Lint
        run: ruff check .

      - name: Type check
        run: mypy src/

      - name: Test
        run: pytest --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml

  publish:
    needs: test
    if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/')
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - name: Publish to PyPI
        run: |
          pip install build twine
          python -m build
          twine upload dist/*
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_TOKEN }}
```

### GH10: Release Notes Automation
- [ ] Auto-generate from conventional commits
- [ ] Changelog generation
- [ ] Version bumping

---

# ESTATÍSTICAS DO MAPEAMENTO

```
Total de Seções: 18
Total de Categorias: ~150
Total de Items: ~2000+
Total de Linhas: ~9300+

Por Seção:
├── PARTE 1:  Fundação (~20 items)         ✅ COMPLETA (State of the Art)
├── PARTE 2:  Personas (~100 items)        ✅ COMPLETA (JTBD, Empathy Maps)
├── PARTE 3:  User Journeys (~200 items)   ✅ COMPLETA (E2E Flows)
├── PARTE 4:  Features (~300 items)        ✅ COMPLETA (Technical Specs)
├── PARTE 5:  Arquitetura (~100 items)     ✅ COMPLETA (Architecture Decisions)
├── PARTE 6:  Integrações (~50 items)      ✅ COMPLETA (Provider Matrix)
├── PARTE 7:  Operações (~50 items)        ✅ COMPLETA (Deployment, Monitoring)
├── PARTE 8:  Suporte & Docs (~30 items)   ✅ COMPLETA (Docs Structure, SLAs)
├── PARTE 9:  Métricas (~30 items)         ✅ COMPLETA (KPIs, Dashboards)
├── PARTE 10: Compliance (~20 items)       ✅ COMPLETA (GDPR, Legal)
├── PARTE 11: Go-to-Market (~30 items)     ✅ COMPLETA (Launch Plan)
├── PARTE 12: Frontend & UI (~250 items)   ✅ COMPLETA (Design System, 62 componentes, AI-Enhanced)
├── PARTE 13: Internationalization (~80 items)  ✅ COMPLETA (i18n, L10n, AI Translation, 23 seções)
├── PARTE 14: Cockpit Visual (~200 items)  ✅ COMPLETA (Real-time Dashboard, 43 seções, 3D Mode)
├── PARTE 15: Website HumanGR (~200 items) ✅ COMPLETA (80 widgets, Premium Design)
├── PARTE 16: Marketing & Pitch (~150 items) ✅ COMPLETA (Decks, KPIs, Benchmarks, ROI Calculator)
├── PARTE 17: System Diagrams (~50 items)  ✅ COMPLETA (6 HTML Diagrams, Interactive)
└── PARTE 18: GitHub Profile (~100 items)  ✅ COMPLETA (README, Contributing, Templates, CI/CD)

Status:
[x] Completo: 18 partes (~2000 items)
[~] Parcial: 0 partes
[ ] Não iniciado: 0 partes

Progresso: 100% 🎉
```

---

# PRÓXIMOS PASSOS

## Fase 1: Mapeamento ✅ COMPLETA
1. [x] Mapear estrutura (18 partes, ~2000 items) ✅
2. [x] Preencher PARTE 1: Fundação ✅
3. [x] Preencher PARTE 2: Personas ✅
4. [x] Preencher PARTE 3: User Journeys ✅
5. [x] Preencher PARTE 4: Features ✅
6. [x] Preencher PARTE 5-11: Arquitetura → GTM ✅
7. [x] Preencher PARTE 12: Frontend & UI (+ State Management, Testing, AI-Enhanced) ✅
8. [x] Preencher PARTE 13: i18n (+ AI Translation Pipeline) ✅
9. [x] Preencher PARTE 14: Cockpit Visual (+ 3D Mode, Graph Engine, Collaboration) ✅
10. [x] Preencher PARTE 15: Website HumanGR ✅
11. [x] Preencher PARTE 16: Marketing Materials (Pitch Deck, KPIs, Benchmarks, ROI) ✅
12. [x] Preencher PARTE 17: System Diagrams (6 HTML Interativos) ✅
13. [x] Preencher PARTE 18: GitHub Profile (README, Contributing, CI/CD) ✅

## Fase 2: Refinamento (PRÓXIMA)
14. [ ] Revisar com stakeholders
15. [ ] Priorizar por importância (MoSCoW ou RICE)
16. [ ] Identificar dependências entre items
17. [ ] Agrupar em sprints lógicos

## Fase 3: Execução
18. [ ] Converter para formato pipeline-compatible (context_packs)
19. [ ] Gerar SPRINT_INDEX.yaml
20. [ ] Criar LEGO_INDEX.yaml com módulos
21. [ ] Iniciar execução do pipeline

