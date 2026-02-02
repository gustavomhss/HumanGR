"""
Targeted Fix - Correção cirúrgica de erros específicos.

Quando um gate falha com erro fixável (CODE, PERFORMANCE),
este módulo:
1. Analisa o erro específico (arquivo, linha, mensagem)
2. Gera instrução de fix direcionada
3. Chama daemon para aplicar o fix
4. Retorna resultado para re-validação

NÃO substitui exec_node - complementa após falhas.

Invariantes:
    INV-TF-001: targeted_fix NUNCA substitui exec_node
    INV-TF-002: targeted_fix SÓ ativa para CODE ou PERFORMANCE
    INV-TF-003: Máximo 3 tentativas de targeted_fix por sprint
    INV-TF-004: targeted_fix usa daemon real (execute_agent_task)
    INV-TF-005: Após targeted_fix bem sucedido → re-run APENAS gates que falharam

Created: 2026-01-30
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .state import PipelineState

logger = logging.getLogger(__name__)


# =============================================================================
# FAILURE TYPE CLASSIFICATION
# =============================================================================

# Import GateFailureType from qa_schemas
try:
    from pipeline.qa_schemas import (
        GateFailureType,
        classify_gate_failure,
        get_failure_remediation,
    )
    QA_SCHEMAS_AVAILABLE = True
except ImportError:
    logger.warning("qa_schemas not available, using fallback classification")
    QA_SCHEMAS_AVAILABLE = False

    # Fallback enum
    from enum import Enum

    class GateFailureType(Enum):
        CODE = "code"
        TIMEOUT = "timeout"
        INFRASTRUCTURE = "infra"
        SECURITY = "security"
        PERFORMANCE = "performance"


def can_apply_targeted_fix(failure_type: GateFailureType) -> bool:
    """
    Verifica se o tipo de falha permite fix cirúrgico.

    Somente falhas do tipo CODE e PERFORMANCE podem ser corrigidas
    cirurgicamente por agentes. Outros tipos requerem intervenção
    de ops (TIMEOUT, INFRASTRUCTURE) ou humanos (SECURITY).

    Args:
        failure_type: Tipo de falha classificada

    Returns:
        True se o tipo permite fix cirúrgico, False caso contrário

    Examples:
        >>> can_apply_targeted_fix(GateFailureType.CODE)
        True
        >>> can_apply_targeted_fix(GateFailureType.TIMEOUT)
        False
    """
    fixable_types = (GateFailureType.CODE, GateFailureType.PERFORMANCE)
    return failure_type in fixable_types


# =============================================================================
# FAILURE ANALYSIS
# =============================================================================

async def analyze_gate_failure(
    gate_result: Dict[str, Any],
    run_dir: Path,
) -> Dict[str, Any]:
    """
    Analisa falha de gate e extrai informações específicas do erro.

    Lê o log do gate e usa regex patterns para extrair:
    - Tipo de erro (assertion, syntax, type, import, etc.)
    - Arquivo afetado
    - Número da linha
    - Mensagem de erro
    - Arquivos potencialmente afetados

    Args:
        gate_result: Resultado do gate com exit_code, log_path, etc.
        run_dir: Diretório do run para resolver paths relativos

    Returns:
        Dict com análise estruturada:
        {
            "error_type": str,
            "file_path": Optional[str],
            "line_number": Optional[int],
            "error_message": str,
            "affected_files": List[str],
            "full_traceback": Optional[str],
            "gate_id": str,
            "exit_code": int,
        }

    Raises:
        FileNotFoundError: Se log_path não existir
    """
    analysis = {
        "error_type": "unknown",
        "file_path": None,
        "line_number": None,
        "error_message": "",
        "affected_files": [],
        "full_traceback": None,
        "gate_id": gate_result.get("gate_id", "unknown"),
        "exit_code": gate_result.get("exit_code", 1),
    }

    # Get log path
    log_path = gate_result.get("log_path")
    if not log_path:
        logger.warning(f"No log_path in gate_result for {analysis['gate_id']}")
        analysis["error_message"] = "No log available for analysis"
        return analysis

    log_file = Path(log_path)
    if not log_file.is_absolute():
        log_file = run_dir / log_path

    if not log_file.exists():
        logger.warning(f"Log file not found: {log_file}")
        analysis["error_message"] = f"Log file not found: {log_path}"
        return analysis

    # Read log content
    try:
        log_content = log_file.read_text(encoding="utf-8", errors="replace")
        analysis["full_traceback"] = log_content[-5000:]  # Last 5000 chars
    except Exception as e:
        logger.error(f"Failed to read log file: {e}")
        analysis["error_message"] = f"Failed to read log: {e}"
        return analysis

    # Extract error information using patterns
    analysis = _extract_error_info(log_content, analysis)

    return analysis


def _extract_error_info(log_content: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extrai informações de erro do conteúdo do log usando regex patterns.

    Patterns suportados:
    - Python tracebacks (File "...", line N)
    - AssertionError, TypeError, ValueError, etc.
    - SyntaxError
    - ImportError/ModuleNotFoundError
    - pytest output
    """
    # Pattern: Python traceback - File "path", line N
    file_line_pattern = r'File "([^"]+)", line (\d+)'
    matches = re.findall(file_line_pattern, log_content)

    if matches:
        # Get the last match (usually the actual error location)
        last_file, last_line = matches[-1]
        analysis["file_path"] = last_file
        analysis["line_number"] = int(last_line)

        # Collect all unique files mentioned
        affected = list(set(m[0] for m in matches))
        # Filter to only project files (not stdlib)
        analysis["affected_files"] = [
            f for f in affected
            if "site-packages" not in f
            and "/lib/python" not in f
            and "src/" in f or "tests/" in f
        ]

    # Pattern: Error type and message
    error_patterns = [
        # AssertionError: message
        (r'AssertionError:\s*(.+?)(?:\n|$)', "assertion_error"),
        # TypeError: message
        (r'TypeError:\s*(.+?)(?:\n|$)', "type_error"),
        # ValueError: message
        (r'ValueError:\s*(.+?)(?:\n|$)', "value_error"),
        # NameError: message
        (r'NameError:\s*(.+?)(?:\n|$)', "name_error"),
        # AttributeError: message
        (r'AttributeError:\s*(.+?)(?:\n|$)', "attribute_error"),
        # ImportError: message
        (r'(?:Import|ModuleNotFound)Error:\s*(.+?)(?:\n|$)', "import_error"),
        # SyntaxError: message
        (r'SyntaxError:\s*(.+?)(?:\n|$)', "syntax_error"),
        # IndentationError: message
        (r'IndentationError:\s*(.+?)(?:\n|$)', "indentation_error"),
        # KeyError: message
        (r'KeyError:\s*(.+?)(?:\n|$)', "key_error"),
        # Generic Exception: message
        (r'Exception:\s*(.+?)(?:\n|$)', "generic_exception"),
        # pytest FAILED marker
        (r'FAILED\s+([^\s]+)\s+-\s+(.+?)(?:\n|$)', "test_failure"),
    ]

    for pattern, error_type in error_patterns:
        match = re.search(pattern, log_content, re.IGNORECASE)
        if match:
            analysis["error_type"] = error_type
            if error_type == "test_failure":
                analysis["error_message"] = f"Test {match.group(1)} failed: {match.group(2)}"
            else:
                analysis["error_message"] = match.group(1).strip()
            break

    # If no specific error found, try to extract last error-like line
    if analysis["error_type"] == "unknown":
        error_lines = re.findall(r'^E\s+(.+)$', log_content, re.MULTILINE)
        if error_lines:
            analysis["error_message"] = error_lines[-1]
            analysis["error_type"] = "pytest_error"

    return analysis


# =============================================================================
# FIX INSTRUCTION BUILDING
# =============================================================================

async def build_fix_instruction(
    failure_analysis: Dict[str, Any],
    context_pack: Dict[str, Any],
) -> str:
    """
    Constrói instrução específica para o daemon fixar o erro.

    Gera um prompt detalhado que inclui:
    - Descrição do erro
    - Arquivo e linha afetados
    - Contexto do sprint (requirements, deliverables)
    - Instruções específicas baseadas no tipo de erro

    Args:
        failure_analysis: Análise do erro de analyze_gate_failure()
        context_pack: Context pack do sprint com requirements

    Returns:
        String com instrução completa para o daemon
    """
    error_type = failure_analysis.get("error_type", "unknown")
    file_path = failure_analysis.get("file_path")
    line_number = failure_analysis.get("line_number")
    error_message = failure_analysis.get("error_message", "Unknown error")
    affected_files = failure_analysis.get("affected_files", [])
    traceback = failure_analysis.get("full_traceback", "")
    gate_id = failure_analysis.get("gate_id", "unknown")

    # Get sprint context
    sprint_objective = context_pack.get("objective", "")
    deliverables = context_pack.get("deliverables", [])

    # Build specific instructions based on error type
    type_specific_instructions = _get_type_specific_instructions(error_type)

    # Build file context section
    file_context = ""
    if file_path:
        file_context = f"""
## LOCALIZAÇÃO DO ERRO
- **Arquivo:** `{file_path}`
- **Linha:** {line_number if line_number else "Não identificada"}
"""

    # Build affected files section
    affected_section = ""
    if affected_files:
        affected_section = f"""
## ARQUIVOS AFETADOS
{chr(10).join(f"- `{f}`" for f in affected_files[:5])}
"""

    # Build traceback section (truncated)
    traceback_section = ""
    if traceback:
        # Get last 1500 chars of traceback
        truncated = traceback[-1500:] if len(traceback) > 1500 else traceback
        traceback_section = f"""
## TRACEBACK
```
{truncated}
```
"""

    instruction = f"""# TARGETED FIX - Correção Cirúrgica

## CONTEXTO
- **Gate:** {gate_id}
- **Tipo de erro:** {error_type}
- **Sprint objetivo:** {sprint_objective[:200] if sprint_objective else "N/A"}

## ERRO DETECTADO
```
{error_message}
```
{file_context}
{affected_section}
{traceback_section}

## INSTRUÇÕES ESPECÍFICAS
{type_specific_instructions}

## AÇÕES OBRIGATÓRIAS
1. **Leia** o arquivo que contém o erro usando Read
2. **Analise** o contexto ao redor da linha problemática
3. **Corrija** o erro usando Edit (prefira Edit a Write para arquivos existentes)
4. **NÃO** faça mudanças além do necessário para corrigir o erro
5. **NÃO** adicione features ou refatore código não relacionado

## VALIDAÇÃO
Após o fix, o gate {gate_id} será re-executado automaticamente.
Seu fix será considerado bem sucedido se o gate passar.

## RESPOSTA OBRIGATÓRIA
Responda em JSON:
{{
    "status": "success" | "failed" | "blocked",
    "summary": "Descrição do que foi corrigido",
    "evidence_paths": ["path/do/arquivo/corrigido.py"],
    "next_steps": ["Próximos passos se houver"]
}}
"""

    return instruction


def _get_type_specific_instructions(error_type: str) -> str:
    """Retorna instruções específicas baseadas no tipo de erro."""

    instructions = {
        "assertion_error": """
Este é um erro de ASSERÇÃO em teste.
- Verifique se o valor retornado está correto
- Pode ser um bug na lógica do código
- OU pode ser que a asserção esteja incorreta
- Prefira corrigir o código, não o teste (a menos que o teste esteja claramente errado)
""",
        "type_error": """
Este é um erro de TIPO.
- Verifique se os tipos dos argumentos estão corretos
- Pode faltar uma conversão de tipo
- Verifique se métodos estão sendo chamados no tipo correto
""",
        "value_error": """
Este é um erro de VALOR.
- Verifique se o valor passado está dentro do range esperado
- Pode faltar validação de entrada
- Verifique constraints de domínio
""",
        "name_error": """
Este é um erro de NOME não definido.
- Verifique se a variável/função foi importada
- Pode ser um typo no nome
- Verifique o escopo da variável
""",
        "attribute_error": """
Este é um erro de ATRIBUTO não encontrado.
- Verifique se o objeto tem o atributo esperado
- Pode ser que o objeto seja None
- Verifique a inicialização do objeto
""",
        "import_error": """
Este é um erro de IMPORTAÇÃO.
- Verifique se o módulo existe
- Pode faltar instalar uma dependência
- Verifique o caminho do import (relativo vs absoluto)
""",
        "syntax_error": """
Este é um erro de SINTAXE.
- Verifique parênteses, colchetes e chaves
- Pode faltar dois-pontos após if/for/def/class
- Verifique indentação
""",
        "indentation_error": """
Este é um erro de INDENTAÇÃO.
- Use espaços consistentes (4 espaços é o padrão Python)
- Não misture tabs e espaços
- Verifique blocos de código
""",
        "key_error": """
Este é um erro de CHAVE não encontrada em dict.
- Verifique se a chave existe antes de acessar
- Use .get() com valor default
- Pode ser um typo no nome da chave
""",
        "test_failure": """
Este é um TESTE falhando.
- Analise a mensagem de erro do pytest
- Verifique se o comportamento esperado está correto
- Corrija o código, não desabilite o teste
""",
        "pytest_error": """
Este é um erro detectado pelo PYTEST.
- Leia a mensagem de erro cuidadosamente
- Verifique assertions e expects
- Pode ser setup/teardown incorreto
""",
    }

    return instructions.get(error_type, """
Tipo de erro não reconhecido especificamente.
- Leia o traceback cuidadosamente
- Identifique a causa raiz
- Faça a correção mínima necessária
""")


# =============================================================================
# FIX APPLICATION
# =============================================================================

async def apply_targeted_fix(
    instruction: str,
    workspace: Path,
    sprint_id: str,
    affected_files: List[str],
) -> Dict[str, Any]:
    """
    Chama daemon para aplicar fix cirúrgico.

    Usa execute_agent_task do claude_cli_llm para executar o fix
    com as flags corretas do daemon pattern:
    - --dangerously-skip-permissions
    - --print
    - --output-format json

    Args:
        instruction: Instrução completa do fix
        workspace: Diretório raiz do projeto
        sprint_id: ID do sprint para contexto
        affected_files: Lista de arquivos que podem ser modificados

    Returns:
        Dict com resultado do daemon:
        {
            "status": "success" | "failed" | "blocked",
            "summary": str,
            "evidence_paths": List[str],
            "next_steps": List[str],
        }
    """
    try:
        from pipeline.claude_cli_llm import execute_agent_task
    except ImportError as e:
        logger.error(f"Failed to import execute_agent_task: {e}")
        return {
            "status": "failed",
            "summary": f"Daemon not available: {e}",
            "evidence_paths": [],
            "next_steps": ["Install claude_cli_llm module"],
        }

    logger.info(f"Applying targeted fix for sprint {sprint_id}")
    logger.debug(f"Affected files: {affected_files}")

    try:
        # Call daemon via execute_agent_task
        result = await asyncio.to_thread(
            execute_agent_task,
            task="TARGETED_FIX",
            instruction=instruction,
            workspace_path=str(workspace),
            sprint_id=sprint_id,
            context_refs=affected_files,
            expected_outputs=affected_files,
        )

        logger.info(f"Targeted fix result: {result.get('status')}")
        return result

    except Exception as e:
        logger.error(f"Targeted fix execution failed: {e}")
        return {
            "status": "failed",
            "summary": f"Execution error: {e}",
            "evidence_paths": [],
            "next_steps": ["Check daemon availability", "Review error logs"],
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_failure_type_from_context(rework_context: Dict[str, Any]) -> Optional[GateFailureType]:
    """
    Extrai GateFailureType do rework_context.

    Args:
        rework_context: Contexto de rework do state

    Returns:
        GateFailureType ou None se não encontrado
    """
    failure_type_str = rework_context.get("failure_type")
    if not failure_type_str:
        return None

    try:
        return GateFailureType(failure_type_str)
    except ValueError:
        logger.warning(f"Unknown failure type: {failure_type_str}")
        return None


def should_use_targeted_fix(
    rework_context: Dict[str, Any],
    current_attempt: int,
    max_attempts: int = 3,
) -> bool:
    """
    Decide se deve usar targeted fix ou legacy rework.

    Args:
        rework_context: Contexto de rework
        current_attempt: Tentativa atual (1-based)
        max_attempts: Máximo de tentativas

    Returns:
        True se deve usar targeted fix
    """
    # Check if already exceeded attempts
    if current_attempt >= max_attempts:
        logger.info(f"Max attempts ({max_attempts}) reached, not using targeted fix")
        return False

    # Check failure type
    failure_type = get_failure_type_from_context(rework_context)
    if failure_type is None:
        logger.warning("No failure type in context, defaulting to targeted fix")
        return True  # Default to trying targeted fix

    can_fix = can_apply_targeted_fix(failure_type)
    logger.info(f"Failure type {failure_type.value}: can_fix={can_fix}")

    return can_fix
