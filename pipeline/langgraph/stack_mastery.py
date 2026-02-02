"""Stack Mastery Guide for Pipeline Agents.

This module provides comprehensive guidance for agents to extract
MAXIMUM VALUE from every available stack. Each stack includes:

1. WHAT: Core purpose and capabilities
2. WHEN: Specific scenarios to use it
3. HOW: API patterns and code examples
4. COMBINE: Integration with other stacks
5. AVOID: Common mistakes and pitfalls

Author: Pipeline Autonomo Team
Version: 1.0.0 (2026-01-16)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class StackMastery:
    """Complete mastery guide for a stack."""

    name: str
    category: str
    tagline: str  # One-line value proposition

    # Core capabilities
    core_purpose: str
    key_features: List[str]

    # When to use
    ideal_scenarios: List[str]
    triggers: List[str]  # Keywords/situations that should trigger this stack

    # How to use
    api_patterns: Dict[str, str]  # operation_name -> code pattern
    step_by_step: List[str] = field(default_factory=list)  # Sequential usage guide

    # Integration
    combines_with: Dict[str, str] = field(default_factory=dict)  # stack_name -> integration pattern
    replaces: Optional[str] = None  # What it can replace

    # Warnings
    pitfalls: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    # Priority
    priority: int = 2  # 0=critical, 1=important, 2=optional


# =============================================================================
# COMPLETE STACK MASTERY REGISTRY
# =============================================================================

STACK_MASTERY: Dict[str, StackMastery] = {
    # =========================================================================
    # REASONING STACKS - Para pensar melhor
    # =========================================================================
    "got": StackMastery(
        name="got",
        category="reasoning",
        tagline="Pense em MULTIPLAS perspectivas antes de decidir",
        core_purpose="""
Graph of Thoughts (GoT) permite raciocinio em ARVORE, explorando multiplos
caminhos de pensamento simultaneamente. Em vez de pensar linearmente,
voce explora varias hipoteses, compara resultados e chega a uma conclusao
mais robusta.
""",
        key_features=[
            "Multi-path reasoning - explore 3-5 perspectivas diferentes",
            "Consensus building - compare e sintetize resultados",
            "Root cause analysis - trace problemas ate a origem",
            "Decision trees - avalie opcoes sistematicamente",
        ],
        ideal_scenarios=[
            "Analise de falhas de gate - entender POR QUE falhou",
            "Decisoes arquiteturais - avaliar trade-offs",
            "Problemas complexos sem solucao obvia",
            "Quando uma unica perspectiva nao e suficiente",
            "Validacao de specs antes de implementar",
        ],
        triggers=[
            "falhou", "erro", "bug", "porque", "analise", "decisao",
            "arquitetura", "trade-off", "comparar", "avaliar", "complexo",
        ],
        api_patterns={
            "multi_perspective": """
from pipeline.got import GoTController, ClaudeCLILanguageModel

lm = ClaudeCLILanguageModel()
got = GoTController(lm=lm)

# Analise com 3 perspectivas diferentes
result = got.analyze(
    problem="Por que o gate G3 falhou?",
    perspectives=[
        "Perspectiva tecnica: codigo e implementacao",
        "Perspectiva de requirements: specs e criterios",
        "Perspectiva de processo: workflow e sequencia",
    ]
)

# Sintetize as conclusoes
consensus = got.reach_consensus(result.perspectives)
""",
            "root_cause": """
# Trace ate a causa raiz
root_cause = got.trace_root_cause(
    symptom="Testes de integracao falhando",
    max_depth=5,  # Pergunte "por que" ate 5 vezes
)
print(f"Causa raiz: {root_cause.root}")
print(f"Cadeia causal: {' -> '.join(root_cause.chain)}")
""",
            "decision_tree": """
# Avalie opcoes sistematicamente
decision = got.evaluate_options(
    question="Qual banco de dados usar para knowledge graph?",
    options=["neo4j", "falkordb", "dgraph"],
    criteria=["performance", "custo", "facilidade", "escalabilidade"],
)
print(f"Melhor opcao: {decision.best}")
print(f"Justificativa: {decision.reasoning}")
""",
        },
        step_by_step=[
            "1. Identifique que precisa de analise multi-perspectiva",
            "2. Defina 3-5 perspectivas relevantes ao problema",
            "3. Execute GoT.analyze() com as perspectivas",
            "4. Compare os resultados de cada perspectiva",
            "5. Use reach_consensus() para sintetizar",
            "6. Documente a decisao e o raciocinio",
        ],
        combines_with={
            "reflexion": "GoT analisa -> Reflexion aprende com resultado",
            "langfuse": "Trace toda a arvore de pensamento para debug",
            "letta": "Salve insights importantes na memoria",
        },
        pitfalls=[
            "NAO use para problemas simples - overhead desnecessario",
            "NAO confie em uma unica perspectiva mesmo usando GoT",
            "SEMPRE documente o raciocinio para auditoria",
        ],
        limitations=[
            "Mais lento que raciocinio linear",
            "Consome mais tokens",
        ],
        priority=1,
    ),

    "reflexion": StackMastery(
        name="reflexion",
        category="reasoning",
        tagline="Aprenda com FALHAS e melhore iterativamente",
        core_purpose="""
Reflexion implementa verbal reinforcement learning - quando algo falha,
o sistema reflete sobre O QUE deu errado, POR QUE, e COMO evitar no futuro.
Essencial para melhoria continua do pipeline.
""",
        key_features=[
            "Failure reflection - analise estruturada de falhas",
            "Learning loop - melhoria iterativa",
            "Retry with context - retenta com conhecimento da falha",
            "Pattern detection - identifica erros recorrentes",
        ],
        ideal_scenarios=[
            "Gate falhou - preciso entender e evitar recorrencia",
            "Task falhou apos multiplas tentativas",
            "Padrao de erro se repetindo",
            "Preciso melhorar um processo que nao esta funcionando",
        ],
        triggers=[
            "falhou", "erro", "tentar novamente", "retry", "melhorar",
            "aprender", "evitar", "recorrente", "padrao de erro",
        ],
        api_patterns={
            "reflect_on_failure": """
from pipeline.reflexion_engine import ReflexionEngine

engine = ReflexionEngine()

# Quando um gate falha
reflection = engine.reflect(
    failure={
        "gate": "G3",
        "error": "Coverage abaixo de 90%",
        "context": "Sprint S15, modulo auth",
    }
)

print(f"O que deu errado: {reflection.what_went_wrong}")
print(f"Por que: {reflection.why}")
print(f"Como evitar: {reflection.how_to_avoid}")
print(f"Acao corretiva: {reflection.corrective_action}")
""",
            "retry_with_learning": """
# Retenta uma task com o aprendizado da falha anterior
result = engine.retry_with_learning(
    task="Implementar testes para modulo auth",
    previous_failure=reflection,
    max_retries=3,
)
""",
            "pattern_detection": """
# Detecta padroes em falhas recorrentes
patterns = engine.detect_patterns(
    failures=last_10_failures,
    min_occurrences=3,
)
for pattern in patterns:
    print(f"Padrao detectado: {pattern.description}")
    print(f"Frequencia: {pattern.frequency}")
    print(f"Sugestao: {pattern.suggestion}")
""",
        },
        step_by_step=[
            "1. Capture o erro/falha com contexto completo",
            "2. Execute reflect() para analise estruturada",
            "3. Salve a reflexao no A-MEM (memoria de aprendizado)",
            "4. Se for retentar, use retry_with_learning()",
            "5. Monitore se o padrao se repete",
            "6. Atualize o PATTERN_CATALOG se for recorrente",
        ],
        combines_with={
            "got": "GoT analisa causa raiz -> Reflexion aprende",
            "langfuse": "Trace todo o processo de reflexao",
            "letta": "Persista aprendizados na memoria de longo prazo",
        },
        pitfalls=[
            "NAO ignore reflexoes - elas sao valiosas",
            "NAO retente sem incorporar o aprendizado",
            "SEMPRE salve reflexoes no A-MEM",
        ],
        priority=1,
    ),

    "bot": StackMastery(
        name="bot",
        category="reasoning",
        tagline="Acumule pensamentos ao longo de uma sessao",
        core_purpose="""
Buffer of Thoughts (BoT) mantem um buffer de raciocinio que acumula
insights ao longo de uma sessao. Util para tarefas longas onde
contexto anterior e importante.
""",
        key_features=[
            "Thought accumulation - acumule insights",
            "Context preservation - mantenha contexto entre passos",
            "Synthesis - sintetize pensamentos acumulados",
        ],
        ideal_scenarios=[
            "Tarefas longas com muitos passos",
            "Analises que precisam de contexto acumulado",
            "Quando insights anteriores sao relevantes para decisoes futuras",
        ],
        triggers=[
            "contexto", "acumular", "lembrar", "anterior", "sessao",
        ],
        api_patterns={
            "accumulate": """
from pipeline.langgraph.stack_injection import get_stack

bot = get_stack("bot")

# Acumule pensamentos durante uma analise longa
bot.add_thought("O modulo auth tem alta complexidade ciclomatica")
bot.add_thought("Detectei 3 code smells no arquivo principal")
bot.add_thought("Cobertura de testes e apenas 45%")

# Sintetize ao final
synthesis = bot.synthesize()
print(f"Conclusao: {synthesis}")
""",
        },
        step_by_step=[
            "1. Inicie uma sessao de BoT para tarefas longas",
            "2. Adicione pensamentos relevantes durante a execucao",
            "3. Sintetize ao final ou em pontos de decisao",
            "4. Use a sintese para informar proximos passos",
        ],
        combines_with={
            "got": "BoT acumula -> GoT sintetiza multiplas perspectivas",
            "letta": "Persista sinteses importantes",
        },
        priority=2,
    ),

    "dspy": StackMastery(
        name="dspy",
        category="reasoning",
        tagline="Otimize prompts automaticamente",
        core_purpose="""
DSPy permite programar prompts de forma modular e otimiza-los
automaticamente. Util para criar pipelines de LLM robustos.
""",
        key_features=[
            "Modular prompting - compose modulos de prompt",
            "Auto-optimization - otimize prompts com exemplos",
            "Signatures - defina inputs/outputs esperados",
        ],
        ideal_scenarios=[
            "Criar pipelines de prompt reutilizaveis",
            "Otimizar qualidade de respostas com exemplos",
            "Estruturar prompts complexos",
        ],
        triggers=[
            "prompt", "otimizar", "modular", "pipeline", "few-shot",
        ],
        api_patterns={
            "chain_of_thought": """
import dspy

# Defina uma signature
class QA(dspy.Signature):
    question = dspy.InputField()
    answer = dspy.OutputField()

# Use Chain of Thought
cot = dspy.ChainOfThought(QA)
result = cot(question="O que e Human Layer?")
""",
        },
        step_by_step=[
            "1. Defina Signatures para seus casos de uso",
            "2. Compose modulos (ChainOfThought, ReAct, etc)",
            "3. Otimize com exemplos de treino",
            "4. Deploy o pipeline otimizado",
        ],
        combines_with={
            "instructor": "DSPy estrutura -> Instructor valida output",
        },
        priority=2,
    ),

    # =========================================================================
    # MEMORY STACKS - Para lembrar
    # =========================================================================
    "qdrant": StackMastery(
        name="qdrant",
        category="memory",
        tagline="Busca SEMANTICA - encontre por significado, nao palavras",
        core_purpose="""
Qdrant e um banco de vetores que permite busca semantica.
Voce converte texto em embeddings e busca por SIGNIFICADO similar,
nao apenas palavras exatas.
""",
        key_features=[
            "Semantic search - busque por significado",
            "Vector storage - armazene embeddings",
            "Similarity scoring - quantifique similaridade",
            "Filtering - combine semantica com filtros",
        ],
        ideal_scenarios=[
            "Encontrar documentos/codigo semanticamente similar",
            "RAG - recuperar contexto relevante para prompts",
            "Detectar duplicatas semanticas",
            "Buscar exemplos similares a um caso",
        ],
        triggers=[
            "buscar", "encontrar", "similar", "relacionado", "contexto",
            "rag", "embedding", "semantico",
        ],
        api_patterns={
            "semantic_search": """
from pipeline.qdrant_client import get_qdrant_client

qdrant = get_qdrant_client()

# Busque documentos semanticamente similares
results = qdrant.search(
    collection_name="humangr_docs",
    query_text="como implementar FSM",  # Vai encontrar mesmo sem "FSM" exato
    limit=5,
)

for result in results:
    print(f"Score: {result.score:.2f} - {result.payload['title']}")
""",
            "store_embeddings": """
# Armazene documentos com embeddings
qdrant.upsert(
    collection_name="code_snippets",
    documents=[
        {"id": "1", "text": "def validate_claim()...", "metadata": {"file": "models.py"}},
        {"id": "2", "text": "class ClaimValidator...", "metadata": {"file": "validators.py"}},
    ]
)
""",
            "filtered_search": """
# Busca semantica com filtros
results = qdrant.search(
    collection_name="code_snippets",
    query_text="validacao de entrada",
    filter={
        "must": [
            {"key": "file", "match": {"value": "validators.py"}}
        ]
    },
    limit=10,
)
""",
        },
        step_by_step=[
            "1. Identifique que precisa de busca semantica (nao exata)",
            "2. Determine a collection apropriada",
            "3. Execute search() com o texto de query",
            "4. Use os resultados como contexto para sua task",
            "5. Opcionalmente, adicione filtros para refinar",
        ],
        combines_with={
            "graphrag": "Qdrant (vetores) + Graph (relacoes) = busca poderosa",
            "active_rag": "Qdrant backend para FLARE retrieval",
            "letta": "Busque memorias semanticamente",
        },
        pitfalls=[
            "NAO use para busca exata - use grep/search normal",
            "SEMPRE verifique a collection correta",
            "Embeddings tem custo - nao abuse",
        ],
        priority=1,
    ),

    "letta": StackMastery(
        name="letta",
        category="memory",
        tagline="Memoria PERSISTENTE - lembre de tudo, sempre",
        core_purpose="""
Letta fornece memoria persistente para agents. O que voce salvar
estara disponivel em sessoes futuras. Essencial para aprendizado
continuo e contexto de longo prazo.
""",
        key_features=[
            "Persistent memory - sobrevive entre sessoes",
            "Structured recall - recupere por criterios",
            "Memory types - core (identidade), archival (fatos), recall (recente)",
            "Agent state - persista estado completo do agent",
        ],
        ideal_scenarios=[
            "Salvar aprendizados para sessoes futuras",
            "Manter contexto de usuario/projeto",
            "Persistir decisoes arquiteturais e razoes",
            "Lembrar de erros passados para nao repetir",
        ],
        triggers=[
            "lembrar", "salvar", "persistir", "memoria", "futuro",
            "proximo run", "nao esquecer", "importante",
        ],
        api_patterns={
            "save_learning": """
from pipeline.letta_client import get_letta_client

letta = get_letta_client()

# Salve um aprendizado importante
letta.add_to_archival(
    agent_id="spec_master",
    memory={
        "type": "learning",
        "content": "Gate G3 falha frequentemente quando coverage < 85%",
        "context": "Sprint S15",
        "action": "Sempre verificar coverage antes de submeter",
    }
)
""",
            "recall_relevant": """
# Recupere memorias relevantes antes de uma task
memories = letta.search_archival(
    agent_id="spec_master",
    query="erros comuns em gates",
    limit=5,
)

for mem in memories:
    print(f"Aprendizado: {mem['content']}")
""",
            "save_state": """
# Persista estado completo do agent
letta.save_state(
    agent_id="qa_master",
    state={
        "current_sprint": "S15",
        "pending_reviews": ["PR-123", "PR-456"],
        "quality_threshold": 0.85,
    }
)
""",
        },
        step_by_step=[
            "1. Sempre recupere memorias relevantes ANTES de iniciar uma task",
            "2. Durante a execucao, identifique insights importantes",
            "3. SALVE aprendizados, decisoes e erros no Letta",
            "4. Use tags para organizar (learning, error, decision, fact)",
            "5. Revise memorias periodicamente para consolidar",
        ],
        combines_with={
            "reflexion": "Reflexion gera insight -> Letta persiste",
            "qdrant": "Letta armazena -> Qdrant busca semanticamente",
            "got": "GoT decide -> Letta lembra a decisao",
        },
        replaces="mem0",
        pitfalls=[
            "NAO salve tudo - seja seletivo com o que e importante",
            "SEMPRE adicione contexto/tags para facilitar busca",
            "NAO confie apenas na memoria - verifique fatos",
        ],
        priority=1,
    ),

    "falkordb": StackMastery(
        name="falkordb",
        category="memory",
        tagline="GRAFO de conhecimento - relacione tudo",
        core_purpose="""
FalkorDB e um banco de grafos que armazena entidades e RELACOES.
Ideal para modelar conhecimento onde relacoes sao importantes.
""",
        key_features=[
            "Graph storage - nos e arestas",
            "Cypher queries - linguagem de consulta poderosa",
            "Pattern matching - encontre padroes de relacao",
            "Path finding - encontre caminhos entre entidades",
        ],
        ideal_scenarios=[
            "Modelar dependencias entre modulos/componentes",
            "Rastrear relacoes claim -> evidencia -> fonte",
            "Encontrar caminhos de impacto (quem depende de quem)",
            "Armazenar knowledge graph do dominio",
        ],
        triggers=[
            "relacionamento", "dependencia", "grafo", "conectado",
            "impacto", "quem depende", "caminho entre",
        ],
        api_patterns={
            "store_relationship": """
from pipeline.falkordb_client import get_falkordb_client

fdb = get_falkordb_client()

# Armazene uma relacao
fdb.execute('''
    CREATE (c:Claim {id: 'C001', text: 'Python e popular'})
    CREATE (e:Evidence {id: 'E001', source: 'stackoverflow.com'})
    CREATE (c)-[:SUPPORTED_BY {confidence: 0.95}]->(e)
''')
""",
            "find_dependencies": """
# Encontre todas as dependencias de um modulo
result = fdb.execute('''
    MATCH (m:Module {name: 'auth'})-[:DEPENDS_ON*1..3]->(dep:Module)
    RETURN dep.name, COUNT(*) as depth
    ORDER BY depth
''')

for row in result:
    print(f"Depende de: {row['dep.name']} (profundidade: {row['depth']})")
""",
            "impact_analysis": """
# Analise de impacto: quem sera afetado se eu mudar X?
affected = fdb.execute('''
    MATCH (changed:Module {name: 'core'})<-[:DEPENDS_ON*]-(dependent:Module)
    RETURN DISTINCT dependent.name
''')
""",
        },
        step_by_step=[
            "1. Identifique que precisa modelar relacoes",
            "2. Defina o schema (tipos de nos e relacoes)",
            "3. Popule o grafo com CREATE/MERGE",
            "4. Consulte com MATCH para encontrar padroes",
            "5. Use para decisoes (impacto, dependencias, etc)",
        ],
        combines_with={
            "graphiti": "FalkorDB armazena -> Graphiti adiciona semantica",
            "qdrant": "FalkorDB (relacoes) + Qdrant (semantica) = GraphRAG",
        },
        priority=1,
    ),

    "graphiti": StackMastery(
        name="graphiti",
        category="memory",
        tagline="Knowledge graph SEMANTICO com tempo",
        core_purpose="""
Graphiti adiciona semantica e consciencia temporal ao knowledge graph.
Fatos podem ter validade temporal e relacoes sao inferidas semanticamente.
""",
        key_features=[
            "Temporal facts - fatos com validade no tempo",
            "Semantic relationships - relacoes inferidas",
            "Fact extraction - extraia fatos de texto",
        ],
        ideal_scenarios=[
            "Armazenar fatos que mudam com o tempo",
            "Extrair conhecimento de documentos",
            "Manter historico de evolucao do conhecimento",
        ],
        triggers=[
            "fato", "temporal", "validade", "historico", "evolucao",
        ],
        api_patterns={
            "add_temporal_fact": """
from pipeline.langgraph.stack_injection import get_stack

graphiti = get_stack("graphiti")

# Adicione um fato com validade temporal
graphiti.add_fact(
    subject="Python",
    predicate="version",
    object="3.13",
    valid_from="2024-10-01",
    source="python.org",
)
""",
        },
        combines_with={
            "falkordb": "Graphiti semantica + FalkorDB armazenamento",
        },
        priority=2,
    ),

    "graphrag": StackMastery(
        name="graphrag",
        category="memory",
        tagline="RAG turbinado com GRAFOS",
        core_purpose="""
GraphRAG combina busca semantica (vetores) com conhecimento estruturado
(grafo) para retrieval mais poderoso.
""",
        key_features=[
            "Hybrid search - vetores + grafo",
            "Context enrichment - expanda contexto via relacoes",
            "Entity-aware RAG - entenda entidades, nao so texto",
        ],
        ideal_scenarios=[
            "RAG onde relacoes entre entidades sao importantes",
            "Perguntas que requerem raciocinio sobre relacoes",
            "Contexto que precisa ser enriquecido com conhecimento",
        ],
        triggers=[
            "rag", "contexto", "relacionado", "entidade", "enriquecer",
        ],
        api_patterns={
            "hybrid_query": """
from pipeline.langgraph.stack_injection import get_stack

graphrag = get_stack("graphrag")

# Query hibrida: semantica + grafo
results = graphrag.query(
    question="Quais claims sobre Python sao suportados por evidencias de alta confianca?",
    use_graph=True,
    use_vectors=True,
)
""",
        },
        combines_with={
            "qdrant": "Backend vetorial",
            "falkordb": "Backend de grafo",
        },
        priority=2,
    ),

    "active_rag": StackMastery(
        name="active_rag",
        category="memory",
        tagline="RAG PROATIVO - antecipe o que voce vai precisar",
        core_purpose="""
Active RAG (FLARE) nao espera voce pedir - ele ANTECIPA que informacao
voce vai precisar e ja busca proativamente.
""",
        key_features=[
            "Predictive retrieval - antecipe necessidades",
            "Forward-looking - olhe adiante no raciocinio",
            "Context pre-loading - carregue contexto antes de precisar",
        ],
        ideal_scenarios=[
            "Tarefas complexas onde contexto futuro e previsivel",
            "Quando latencia de retrieval e critica",
            "Analises que seguem um padrao previsivel",
        ],
        triggers=[
            "proativo", "antecipar", "pre-carregar", "flare",
        ],
        api_patterns={
            "anticipate": """
from pipeline.langgraph.stack_injection import get_stack

active_rag = get_stack("active_rag")

# Busca proativa
results = active_rag.retrieve(
    current_query="Implementar validacao de claims",
    anticipate_next=True,  # Ja busca contexto para proximos passos
)
""",
        },
        combines_with={
            "qdrant": "Backend de busca",
        },
        priority=2,
    ),

    "neo4j": StackMastery(
        name="neo4j",
        category="memory",
        tagline="Grafo ENTERPRISE - escalabilidade e algoritmos",
        core_purpose="""
Neo4j e o banco de grafos mais maduro, com algoritmos de grafo
nativos e escalabilidade enterprise.
""",
        key_features=[
            "Graph algorithms - PageRank, community detection, etc",
            "APOC procedures - biblioteca de funcoes uteis",
            "Enterprise scale - bilhoes de nos",
        ],
        ideal_scenarios=[
            "Quando precisar de algoritmos de grafo (centralidade, comunidades)",
            "Scale enterprise",
            "Analises complexas de rede",
        ],
        triggers=[
            "algoritmo", "centralidade", "pagerank", "comunidade",
        ],
        api_patterns={
            "graph_algorithm": """
from pipeline.langgraph.stack_injection import get_stack

neo4j = get_stack("neo4j")

# Execute algoritmo de PageRank
result = neo4j.execute('''
    CALL gds.pageRank.stream('myGraph')
    YIELD nodeId, score
    RETURN gds.util.asNode(nodeId).name AS name, score
    ORDER BY score DESC
    LIMIT 10
''')
""",
        },
        combines_with={
            "falkordb": "Alternativa mais leve",
        },
        priority=2,
    ),

    # =========================================================================
    # EVALUATION STACKS - Para avaliar qualidade
    # =========================================================================
    "deepeval": StackMastery(
        name="deepeval",
        category="eval",
        tagline="AVALIE outputs de LLM com metricas profundas",
        core_purpose="""
DeepEval fornece metricas especializadas para avaliar qualidade
de outputs de LLM, incluindo faithfulness, relevance, toxicity, etc.
""",
        key_features=[
            "LLM-as-judge metrics - use LLM para avaliar LLM",
            "Built-in metrics - faithfulness, answer relevancy, etc",
            "Custom metrics - crie suas proprias metricas",
            "Test framework - integra com pytest",
        ],
        ideal_scenarios=[
            "Avaliar qualidade de respostas geradas",
            "Validar que output e fiel ao contexto (faithfulness)",
            "Detectar toxicidade ou conteudo inapropriado",
            "Comparar qualidade entre diferentes prompts/modelos",
        ],
        triggers=[
            "avaliar", "qualidade", "metricas", "faithfulness",
            "relevancia", "testar output", "validar resposta",
        ],
        api_patterns={
            "evaluate_response": """
from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

# Crie um test case
test_case = LLMTestCase(
    input="O que e Human Layer?",
    actual_output="Human Layer e um sistema de validacao com 7 camadas...",
    retrieval_context=["Human Layer: sistema de verificacao..."],
)

# Avalie com multiplas metricas
faithfulness = FaithfulnessMetric(threshold=0.7)
relevancy = AnswerRelevancyMetric(threshold=0.7)

results = evaluate([test_case], [faithfulness, relevancy])
print(f"Faithfulness: {faithfulness.score}")
print(f"Relevancy: {relevancy.score}")
""",
            "custom_metric": """
from deepeval.metrics import BaseMetric

class TechnicalAccuracyMetric(BaseMetric):
    def measure(self, test_case):
        # Sua logica de avaliacao
        self.score = 0.85
        self.reason = "Resposta tecnicamente precisa"
        return self.score
""",
        },
        step_by_step=[
            "1. Defina o que quer avaliar (faithfulness, relevancy, etc)",
            "2. Crie LLMTestCase com input, output e contexto",
            "3. Escolha metricas apropriadas",
            "4. Execute evaluate()",
            "5. Analise scores e tome decisoes",
        ],
        combines_with={
            "langfuse": "DeepEval avalia -> Langfuse registra scores",
            "reflexion": "Score baixo -> trigger reflexion",
        },
        priority=2,
    ),

    "ragas": StackMastery(
        name="ragas",
        category="eval",
        tagline="Metricas especificas para RAG",
        core_purpose="""
RAGAS fornece metricas especializadas para avaliar sistemas RAG:
faithfulness, context precision, answer relevancy, etc.
""",
        key_features=[
            "RAG-specific metrics - otimizadas para RAG",
            "Component-level eval - avalie retriever e generator separadamente",
            "Benchmark datasets - compare com baselines",
        ],
        ideal_scenarios=[
            "Avaliar qualidade do seu sistema RAG",
            "Comparar diferentes configuracoes de RAG",
            "Identificar se problema esta no retrieval ou generation",
        ],
        triggers=[
            "rag", "retrieval", "contexto", "precision", "recall",
        ],
        api_patterns={
            "evaluate_rag": """
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, answer_relevancy

# Prepare seu dataset
dataset = {
    "question": ["O que e Human Layer?"],
    "answer": ["Sistema de verificacao..."],
    "contexts": [["Human Layer..."]],
    "ground_truth": ["Human Layer e um sistema..."],
}

# Avalie
results = evaluate(dataset, metrics=[
    faithfulness,
    context_precision,
    answer_relevancy,
])

print(f"Faithfulness: {results['faithfulness']}")
print(f"Context Precision: {results['context_precision']}")
""",
        },
        combines_with={
            "qdrant": "Avalie qualidade do retrieval do Qdrant",
            "deepeval": "Combine metricas de ambos",
        },
        priority=2,
    ),

    "trulens": StackMastery(
        name="trulens",
        category="eval",
        tagline="Feedback functions para LLM apps",
        core_purpose="""
TruLens fornece feedback functions e observabilidade para
aplicacoes LLM, ajudando a identificar problemas e melhorar.
""",
        key_features=[
            "Feedback functions - groundedness, relevance, etc",
            "App instrumentation - wrappe sua app para monitorar",
            "Leaderboard - compare versoes",
        ],
        ideal_scenarios=[
            "Monitorar qualidade de LLM app em producao",
            "A/B testing de diferentes configuracoes",
            "Debug de problemas de qualidade",
        ],
        triggers=[
            "monitorar", "producao", "feedback", "groundedness",
        ],
        api_patterns={
            "instrument_app": """
from trulens.core import Tru

tru = Tru()

# Registre sua aplicacao
from trulens.apps.custom import TruCustomApp

tru_app = TruCustomApp(
    app=my_llm_app,
    feedbacks=[groundedness, relevance],
)

# Use normalmente
with tru_app as recording:
    response = my_llm_app.query("pergunta")
""",
        },
        combines_with={
            "langfuse": "TruLens feedback + Langfuse tracing",
        },
        priority=2,
    ),

    "cleanlab": StackMastery(
        name="cleanlab",
        category="eval",
        tagline="Detecte ALUCINACOES e problemas de dados",
        core_purpose="""
Cleanlab detecta problemas de qualidade em dados e outputs,
incluindo alucinacoes, labels incorretos, e dados ruidosos.
""",
        key_features=[
            "Hallucination detection - detecte fabricacoes",
            "Label quality - encontre labels errados",
            "Data quality - identifique outliers e ruido",
        ],
        ideal_scenarios=[
            "Validar que output nao contem alucinacoes",
            "Limpar datasets de treino",
            "Identificar exemplos problematicos",
        ],
        triggers=[
            "alucinacao", "fabricacao", "qualidade de dados", "label",
        ],
        api_patterns={
            "detect_hallucination": """
from cleanlab import Datalab

# Analise qualidade de um dataset
lab = Datalab(data=my_dataset, label_name="label")
lab.find_issues()

# Veja problemas encontrados
print(lab.get_issues())
""",
        },
        combines_with={
            "deepeval": "Cleanlab dados + DeepEval outputs",
        },
        priority=2,
    ),

    # =========================================================================
    # SECURITY STACKS - Para proteger
    # =========================================================================
    "nemo": StackMastery(
        name="nemo",
        category="security",
        tagline="GUARDRAILS - controle o que entra e sai do LLM",
        core_purpose="""
NeMo Guardrails permite definir regras (rails) que controlam
o comportamento do LLM, prevenindo outputs indesejados.
""",
        key_features=[
            "Input rails - filtre/transforme inputs",
            "Output rails - filtre/transforme outputs",
            "Dialog rails - controle fluxo de conversa",
            "Topical rails - mantenha conversa on-topic",
        ],
        ideal_scenarios=[
            "Prevenir prompt injection",
            "Filtrar PII de inputs/outputs",
            "Manter LLM focado no dominio",
            "Bloquear topicos off-limits",
        ],
        triggers=[
            "seguranca", "filtrar", "bloquear", "pii", "injection",
            "guardrail", "policy", "regra",
        ],
        api_patterns={
            "apply_guardrails": """
from nemoguardrails import RailsConfig, LLMRails

# Configure rails
config = RailsConfig.from_path("./config")
rails = LLMRails(config)

# Gere com guardrails aplicados
response = rails.generate(
    messages=[{"role": "user", "content": user_input}]
)
# Input e output foram filtrados automaticamente
""",
            "define_rails": """
# config/rails.co
define user express greeting
  "hello"
  "hi"

define bot express greeting
  "Hello! How can I help you with Human Layer today?"

define flow greeting
  user express greeting
  bot express greeting
""",
        },
        step_by_step=[
            "1. Defina suas politicas de seguranca",
            "2. Crie rails (input, output, dialog) em config/",
            "3. Carregue config com RailsConfig",
            "4. Wrappe todas as chamadas LLM com LLMRails",
            "5. Monitore violacoes de rails",
        ],
        combines_with={
            "langfuse": "Log violacoes de rails no Langfuse",
        },
        priority=1,
    ),

    "z3": StackMastery(
        name="z3",
        category="security",
        tagline="VERIFICACAO FORMAL - prove que esta correto",
        core_purpose="""
Z3 e um SMT solver que permite verificar formalmente
propriedades logicas e resolver constraints.
""",
        key_features=[
            "Constraint solving - resolva sistemas de constraints",
            "Formal verification - prove propriedades",
            "Model checking - verifique invariantes",
        ],
        ideal_scenarios=[
            "Verificar invariantes de sistema formalmente",
            "Resolver problemas de scheduling/alocacao",
            "Provar corretude de algoritmos criticos",
        ],
        triggers=[
            "provar", "verificar", "formal", "constraint", "invariante",
        ],
        api_patterns={
            "verify_invariant": """
from z3 import *

# Defina variaveis
x = Int('x')
y = Int('y')

# Defina constraints
solver = Solver()
solver.add(x > 0)
solver.add(y > 0)
solver.add(x + y == 10)

# Verifique se e satisfazivel
if solver.check() == sat:
    model = solver.model()
    print(f"x = {model[x]}, y = {model[y]}")
""",
        },
        combines_with={
            "got": "GoT gera hipoteses -> Z3 verifica formalmente",
        },
        priority=2,
    ),

    # =========================================================================
    # OBSERVABILITY STACKS - Para monitorar
    # =========================================================================
    "langfuse": StackMastery(
        name="langfuse",
        category="observability",
        tagline="TRACE TUDO - visibilidade total do LLM (OBRIGATORIO)",
        core_purpose="""
Langfuse fornece observabilidade completa para aplicacoes LLM:
traces, costs, latency, qualidade. E OBRIGATORIO usar em toda
chamada LLM para debug e monitoramento.
""",
        key_features=[
            "Tracing - veja o fluxo completo de execucao",
            "Cost tracking - monitore gastos com LLM",
            "Latency metrics - identifique gargalos",
            "Quality scores - registre avaliacoes",
        ],
        ideal_scenarios=[
            "TODA chamada LLM deve ser trackeada",
            "Debug de problemas em producao",
            "Analise de custos e otimizacao",
            "Correlacionar qualidade com metricas",
        ],
        triggers=[
            "trace", "debug", "custo", "latencia", "monitorar",
            "observabilidade", "log",
        ],
        api_patterns={
            "trace_generation": """
from langfuse import Langfuse

langfuse = Langfuse()

# Crie um trace para a operacao
trace = langfuse.trace(
    name="claim_verification",
    metadata={"sprint": "S15", "agent": "spec_master"},
)

# Log a geracao
generation = trace.generation(
    name="verify_claim",
    model="claude-3",
    input=prompt,
    output=response,
    usage={"prompt_tokens": 100, "completion_tokens": 50},
)

# Adicione score de qualidade
trace.score(
    name="faithfulness",
    value=0.92,
)
""",
            "span_context": """
# Para operacoes aninhadas
with langfuse.span(name="complex_analysis") as span:
    # Suboperacoes serao aninhadas
    result1 = span.generation(name="step1", ...)
    result2 = span.generation(name="step2", ...)
""",
        },
        step_by_step=[
            "1. SEMPRE crie um trace no inicio de uma operacao",
            "2. Log todas as geracoes LLM com generation()",
            "3. Adicione metadata relevante (sprint, agent, etc)",
            "4. Registre scores de qualidade quando disponivel",
            "5. Use spans para operacoes aninhadas",
        ],
        combines_with={
            "deepeval": "DeepEval scores -> Langfuse registra",
            "reflexion": "Reflexion insights -> Langfuse metadata",
        },
        pitfalls=[
            "NUNCA faca chamada LLM sem trace",
            "SEMPRE adicione metadata para filtrar depois",
            "NAO esqueca de registrar custos (usage)",
        ],
        priority=0,  # MANDATORY
    ),

    "phoenix": StackMastery(
        name="phoenix",
        category="observability",
        tagline="ML observability e experiment tracking",
        core_purpose="""
Phoenix fornece observabilidade para ML/LLM com foco em
experimentos, drift detection e debugging visual.
""",
        key_features=[
            "Experiment tracking - compare runs",
            "Drift detection - detecte mudancas de distribuicao",
            "Embedding visualization - visualize vetores",
        ],
        ideal_scenarios=[
            "Comparar experimentos de prompt",
            "Detectar drift em embeddings",
            "Debug visual de problemas",
        ],
        triggers=[
            "experimento", "drift", "visualizar", "comparar",
        ],
        api_patterns={
            "track_experiment": """
import phoenix as px

# Inicie sessao
session = px.launch_app()

# Log metricas
px.log_evaluation(
    name="prompt_v2",
    metrics={"accuracy": 0.85, "latency": 1.2},
)
""",
        },
        combines_with={
            "langfuse": "Langfuse traces + Phoenix experiments",
        },
        priority=2,
    ),

    # =========================================================================
    # INFRASTRUCTURE STACKS - Base do sistema
    # =========================================================================
    "redis": StackMastery(
        name="redis",
        category="primary",
        tagline="EVENT BUS e CACHE - comunicacao entre agents",
        core_purpose="""
Redis e o barramento de eventos do pipeline. Agents se comunicam
via pub/sub, compartilham cache, e coordenam via locks.
""",
        key_features=[
            "Pub/Sub - comunicacao assincrona entre agents",
            "Cache - armazene dados temporarios",
            "Locks - coordene acesso a recursos",
            "Streams - event sourcing",
        ],
        ideal_scenarios=[
            "Comunicar eventos entre agents",
            "Cache de resultados computacionalmente caros",
            "Coordenar acesso a recursos compartilhados",
            "Heartbeats e health checks",
        ],
        triggers=[
            "evento", "notificar", "cache", "lock", "coordenar",
            "compartilhar", "broadcast",
        ],
        api_patterns={
            "publish_event": """
from pipeline.redis_client import get_redis_client

redis = get_redis_client()

# Publique um evento
redis.publish("gate_completed", {
    "gate": "G3",
    "status": "passed",
    "sprint": "S15",
})
""",
            "cache_result": """
# Cache um resultado caro
redis.setex(
    "expensive_computation:S15:auth",
    3600,  # TTL em segundos
    result,
)

# Recupere do cache
cached = redis.get("expensive_computation:S15:auth")
""",
        },
        step_by_step=[
            "1. Use pub/sub para eventos (nao polling)",
            "2. Cache resultados que serao reusados",
            "3. Use locks para recursos compartilhados",
            "4. Sempre defina TTL no cache",
        ],
        combines_with={
            "temporal": "Redis eventos + Temporal durabilidade",
        },
        priority=0,
    ),

    "crewai": StackMastery(
        name="crewai",
        category="primary",
        tagline="ORQUESTRACAO de agents - hierarquia e delegacao",
        core_purpose="""
CrewAI orquestra multiplos agents em hierarquia, delegando tasks
e coordenando execucao. Cada agent tem role, goal e backstory.
""",
        key_features=[
            "Agent hierarchy - organize agents em niveis",
            "Task delegation - delegue tasks automaticamente",
            "Crew execution - execute squads de agents",
            "Tool integration - integre tools externos",
        ],
        ideal_scenarios=[
            "Executar tasks que requerem multiplos agents",
            "Delegar subtasks para agents especializados",
            "Coordenar squads de trabalho",
        ],
        triggers=[
            "equipe", "squad", "delegar", "agents", "coordenar",
        ],
        api_patterns={
            "create_crew": """
from crewai import Agent, Task, Crew

# Defina agents
spec_agent = Agent(
    role="Spec Master",
    goal="Definir especificacoes precisas",
    backstory="Especialista em requirements...",
)

qa_agent = Agent(
    role="QA Master",
    goal="Garantir qualidade",
    backstory="Especialista em testes...",
)

# Defina tasks
spec_task = Task(
    description="Especificar modulo auth",
    agent=spec_agent,
)

# Crie e execute crew
crew = Crew(agents=[spec_agent, qa_agent], tasks=[spec_task])
result = crew.kickoff()
""",
        },
        combines_with={
            "langgraph": "CrewAI agents + LangGraph workflows",
            "letta": "Persista estado dos agents",
        },
        priority=0,
    ),

    "langgraph": StackMastery(
        name="langgraph",
        category="primary",
        tagline="STATE MACHINE - controle de fluxo do pipeline",
        core_purpose="""
LangGraph implementa maquinas de estado para workflows LLM,
com checkpointing automatico e controle de fluxo.
""",
        key_features=[
            "State management - gerencie estado do workflow",
            "Checkpointing - recupere de falhas",
            "Conditional edges - fluxo condicional",
            "Parallel execution - execute nos em paralelo",
        ],
        ideal_scenarios=[
            "Definir workflows complexos com estado",
            "Implementar retry automatico",
            "Fluxos condicionais baseados em resultado",
        ],
        triggers=[
            "workflow", "estado", "fluxo", "checkpoint", "fsm",
        ],
        api_patterns={
            "define_workflow": """
from langgraph.graph import StateGraph

# Defina o grafo
graph = StateGraph(MyState)

# Adicione nos
graph.add_node("spec", spec_node)
graph.add_node("impl", impl_node)
graph.add_node("gate", gate_node)

# Adicione edges
graph.add_edge("spec", "impl")
graph.add_conditional_edges(
    "gate",
    lambda s: "pass" if s["passed"] else "fail",
    {"pass": END, "fail": "spec"},
)

# Compile e execute
compiled = graph.compile(checkpointer=MemorySaver())
result = compiled.invoke(initial_state)
""",
        },
        combines_with={
            "temporal": "LangGraph estado + Temporal durabilidade",
            "redis": "LangGraph fluxo + Redis eventos",
        },
        priority=0,
    ),

    "temporal": StackMastery(
        name="temporal",
        category="primary",
        tagline="DURABILIDADE - nunca perca progresso",
        core_purpose="""
Temporal garante que workflows sobrevivam a falhas. Se o processo
morrer, o workflow resume de onde parou.
""",
        key_features=[
            "Workflow durability - sobreviva a crashes",
            "Activity retry - retry automatico com backoff",
            "Checkpointing - salve estado automaticamente",
            "Long-running - workflows de dias/semanas",
        ],
        ideal_scenarios=[
            "Workflows que nao podem falhar",
            "Operacoes longas (sprints, packs)",
            "Retry automatico de atividades",
        ],
        triggers=[
            "duravel", "nao pode falhar", "recuperar", "longo",
        ],
        api_patterns={
            "durable_workflow": """
from pipeline.temporal_integration import get_temporal_integration

temporal = get_temporal_integration()

# Execute workflow duravel
result = await temporal.execute_sprint(
    sprint_id="S15",
    tasks=tasks,
    context_pack_verified=True,
)

# Se falhar, resume automaticamente
""",
        },
        combines_with={
            "langgraph": "Temporal durabilidade + LangGraph estado",
        },
        priority=1,
    ),

    "instructor": StackMastery(
        name="instructor",
        category="primary",
        tagline="STRUCTURED OUTPUT - extraia dados tipados do LLM",
        core_purpose="""
Instructor extrai outputs estruturados (Pydantic models) de LLMs,
com validacao automatica e retry.
""",
        key_features=[
            "Pydantic extraction - extraia models tipados",
            "Automatic retry - retenta se validacao falhar",
            "Streaming - extraia incrementalmente",
        ],
        ideal_scenarios=[
            "Extrair dados estruturados de texto",
            "Garantir que output segue um schema",
            "Parsear respostas LLM de forma robusta",
        ],
        triggers=[
            "extrair", "estruturado", "pydantic", "schema", "json",
        ],
        api_patterns={
            "extract_model": """
import instructor
from pydantic import BaseModel

class ClaimAnalysis(BaseModel):
    claim: str
    confidence: float
    evidence: list[str]

# Patche o client
client = instructor.patch(openai_client)

# Extraia structured output
result = client.chat.completions.create(
    model="claude-3",
    response_model=ClaimAnalysis,
    messages=[{"role": "user", "content": "Analise: Python e popular"}],
)

# result e um ClaimAnalysis validado
""",
        },
        combines_with={
            "pydantic_ai": "Alternativas para structured output",
        },
        priority=2,
    ),

    "pydantic_ai": StackMastery(
        name="pydantic_ai",
        category="primary",
        tagline="Type-safe AI agents com Pydantic",
        core_purpose="""
Pydantic AI permite criar agents type-safe com validacao
automatica de inputs e outputs.
""",
        key_features=[
            "Type-safe agents - validacao em tempo de execucao",
            "Structured tools - tools com schemas Pydantic",
            "Multi-model - funciona com varios providers",
        ],
        ideal_scenarios=[
            "Agents que precisam de type safety",
            "Integracao com ecossistema Pydantic",
        ],
        triggers=[
            "type-safe", "pydantic", "validacao",
        ],
        api_patterns={
            "type_safe_agent": """
from pydantic_ai import Agent
from pydantic import BaseModel

class Output(BaseModel):
    answer: str
    confidence: float

agent = Agent(
    model="claude-3",
    result_type=Output,
)

result = agent.run_sync("Pergunta aqui")
# result e Output validado
""",
        },
        combines_with={
            "instructor": "Alternativas para structured output",
        },
        priority=2,
    ),

    "hamilton": StackMastery(
        name="hamilton",
        category="primary",
        tagline="DAG de transformacoes de dados",
        core_purpose="""
Hamilton permite definir pipelines de dados como DAGs,
com rastreabilidade e reproducibilidade.
""",
        key_features=[
            "DAG execution - execute transformacoes como grafo",
            "Lineage tracking - rastreie origem dos dados",
            "Parallelization - execute em paralelo automaticamente",
        ],
        ideal_scenarios=[
            "Pipelines de processamento de dados",
            "Transformacoes com dependencias",
            "Reproducibilidade de experimentos",
        ],
        triggers=[
            "pipeline", "dag", "transformacao", "dados",
        ],
        api_patterns={
            "define_dag": """
import hamilton
from hamilton import driver

# Defina funcoes (nos do DAG)
def raw_claims(data: pd.DataFrame) -> pd.DataFrame:
    return data

def validated_claims(raw_claims: pd.DataFrame) -> pd.DataFrame:
    return raw_claims[raw_claims['valid']]

# Execute
dr = driver.Driver(config, module)
result = dr.execute(['validated_claims'], inputs={'data': df})
""",
        },
        priority=2,
    ),
}


def get_stack_mastery(stack_name: str) -> Optional[StackMastery]:
    """Get mastery guide for a specific stack."""
    return STACK_MASTERY.get(stack_name)


def get_all_masteries() -> Dict[str, StackMastery]:
    """Get all stack masteries."""
    return STACK_MASTERY


def generate_mastery_prompt(available_stacks: list[str]) -> str:
    """Generate comprehensive mastery prompt for available stacks.

    This generates a DETAILED guide that teaches agents exactly
    how to use each stack to its maximum potential.
    """
    lines = [
        "# GUIA COMPLETO DE STACKS - USE AO MAXIMO",
        "",
        "Voce tem acesso as seguintes stacks. Este guia ensina COMO usar",
        "cada uma para extrair MAXIMO VALOR. Leia com atencao.",
        "",
        "=" * 70,
        "",
    ]

    # Group by category
    categories = {}
    for stack_name in available_stacks:
        mastery = STACK_MASTERY.get(stack_name)
        if mastery:
            cat = mastery.category
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(mastery)

    cat_order = [
        ("reasoning", "REASONING - Para pensar melhor"),
        ("memory", "MEMORY - Para lembrar"),
        ("eval", "EVAL - Para avaliar qualidade"),
        ("security", "SECURITY - Para proteger"),
        ("observability", "OBSERVABILITY - Para monitorar"),
        ("primary", "INFRASTRUCTURE - Base do sistema"),
    ]

    for cat_key, cat_title in cat_order:
        if cat_key in categories:
            lines.append(f"## {cat_title}")
            lines.append("")

            for mastery in categories[cat_key]:
                lines.append(f"### {mastery.name.upper()}: {mastery.tagline}")
                lines.append("")
                lines.append(f"**O QUE FAZ:** {mastery.core_purpose.strip()}")
                lines.append("")
                lines.append("**QUANDO USAR:**")
                for scenario in mastery.ideal_scenarios[:3]:
                    lines.append(f"  - {scenario}")
                lines.append("")
                lines.append("**COMO USAR:**")
                # Show first API pattern
                first_pattern = list(mastery.api_patterns.values())[0] if mastery.api_patterns else ""
                if first_pattern:
                    lines.append("```python")
                    lines.append(first_pattern.strip())
                    lines.append("```")
                lines.append("")

                if mastery.combines_with:
                    lines.append("**COMBINA COM:**")
                    for other, how in list(mastery.combines_with.items())[:2]:
                        lines.append(f"  - {other}: {how}")
                    lines.append("")

                if mastery.pitfalls:
                    lines.append("**CUIDADO:**")
                    for pitfall in mastery.pitfalls[:2]:
                        lines.append(f"  - {pitfall}")
                    lines.append("")

                lines.append("-" * 40)
                lines.append("")

    # Add decision table
    lines.append("## TABELA DE DECISAO RAPIDA")
    lines.append("")
    lines.append("| Situacao | Stack | Razao |")
    lines.append("|----------|-------|-------|")
    lines.append("| Preciso analisar uma falha | got + reflexion | Multi-perspectiva + aprendizado |")
    lines.append("| Preciso buscar por significado | qdrant | Busca semantica |")
    lines.append("| Preciso lembrar algo para depois | letta | Memoria persistente |")
    lines.append("| Preciso modelar relacoes | falkordb | Knowledge graph |")
    lines.append("| Preciso avaliar qualidade | deepeval | Metricas de LLM |")
    lines.append("| Preciso proteger input/output | nemo | Guardrails |")
    lines.append("| Preciso rastrear execucao | langfuse | Tracing OBRIGATORIO |")
    lines.append("| Preciso workflow duravel | temporal | Checkpoint/resume |")
    lines.append("")

    return "\n".join(lines)


# Aliases for backward compatibility
STACK_CAPABILITIES = {
    name: mastery
    for name, mastery in STACK_MASTERY.items()
}

__all__ = [
    "StackMastery",
    "STACK_MASTERY",
    "get_stack_mastery",
    "get_all_masteries",
    "generate_mastery_prompt",
    "STACK_CAPABILITIES",
]
