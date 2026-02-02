"""
Task Executor for isolated task execution.

This module provides isolated execution of individual tasks,
enabling parallel execution without interference between tasks.

Key Features:
- Isolated execution context per task
- Checkpoint saving after each task
- Timeout handling
- Error isolation

Usage:
    executor = TaskExecutor(llm, workspace, checkpoint_dir)
    result = await executor.execute(task, sprint_context)

Created: 2026-01-29
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from .task_models import Task

logger = logging.getLogger(__name__)


# ===========================================================================
# PROTOCOLS AND CONFIG
# ===========================================================================


class LLMClient(Protocol):
    """Protocol for LLM client."""

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 4000,
    ) -> str:
        """Generate text from prompt."""
        ...


@dataclass
class ExecutorConfig:
    """Configuration for task executor."""

    timeout_seconds: int = 300  # 5 minutes per task
    max_retries: int = 2
    checkpoint_enabled: bool = True


# ===========================================================================
# TASK EXECUTOR
# ===========================================================================


class TaskExecutor:
    """
    Executa tasks individuais com isolamento.

    Cada task é executada em contexto isolado:
    - Não afeta outras tasks
    - Resultado salvo em checkpoint
    - Timeout independente
    - Falhas não propagam

    Responsabilidades:
        1. Carregar contexto mínimo necessário
        2. Construir prompt de execução
        3. Chamar LLM para gerar código
        4. Salvar checkpoint
        5. Retornar resultado isolado
    """

    def __init__(
        self,
        llm: LLMClient,
        workspace: Path,
        checkpoint_dir: Path,
        config: Optional[ExecutorConfig] = None,
    ):
        """
        Inicializa o executor.

        Args:
            llm: Cliente LLM para geração de código
            workspace: Diretório raiz do projeto
            checkpoint_dir: Diretório para salvar checkpoints
            config: Configurações opcionais
        """
        self.llm = llm
        self.workspace = Path(workspace)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.config = config or ExecutorConfig()

        # Criar diretório de checkpoints
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    async def execute(
        self,
        task: "Task",
        sprint_context: Dict[str, Any],
    ) -> "Task":
        """
        Executa uma única task.

        Args:
            task: Task a ser executada
            sprint_context: Contexto da sprint (read-only)

        Returns:
            Task atualizada com código gerado

        Raises:
            asyncio.TimeoutError: Se exceder timeout
            Exception: Outros erros de execução
        """
        from .task_models import TaskStatus

        logger.info(f"Executing task {task.id}: {task.name}")

        # Transicionar estado
        task.transition_to(TaskStatus.EXECUTING)
        start_time = datetime.now(timezone.utc)

        try:
            # 1. Preparar contexto
            task_context = self._prepare_context(task, sprint_context)

            # 2. Construir prompt
            prompt = self._build_execution_prompt(task, task_context)

            # 3. Executar com timeout
            response = await asyncio.wait_for(
                self.llm.generate(
                    prompt=prompt,
                    temperature=0.2,
                    max_tokens=4000,
                ),
                timeout=self.config.timeout_seconds,
            )

            # 4. Extrair código
            code = self._extract_code(response)

            # 5. Atualizar task
            task.code_generated = code
            task.execution_time_ms = int(
                (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )

            # 6. Salvar checkpoint
            if self.config.checkpoint_enabled:
                await self._save_checkpoint(task)

            logger.info(
                f"Task {task.id} executed successfully "
                f"({task.execution_time_ms}ms, {len(code)} chars)"
            )

            return task

        except asyncio.TimeoutError:
            logger.error(f"Task {task.id} timed out after {self.config.timeout_seconds}s")
            task.code_generated = None
            raise

        except Exception as e:
            logger.error(f"Task {task.id} execution failed: {e}")
            task.code_generated = None
            raise

    def _prepare_context(
        self,
        task: "Task",
        sprint_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Prepara contexto mínimo para a task.

        Args:
            task: Task a executar
            sprint_context: Contexto geral da sprint

        Returns:
            Dicionário com contexto necessário
        """
        context = {
            "sprint_id": task.sprint_id,
            "task_id": task.id,
            "task_name": task.name,
            "task_description": task.description,
            "deliverables": task.deliverables,
        }

        # Carregar código de dependências (se disponível)
        deps_code = {}
        for dep_id in task.depends_on:
            checkpoint_path = self.checkpoint_dir / f"{dep_id}.json"
            if checkpoint_path.exists():
                try:
                    with open(checkpoint_path) as f:
                        dep_data = json.load(f)
                        if dep_data.get("code_generated"):
                            deps_code[dep_id] = dep_data["code_generated"]
                except Exception as e:
                    logger.warning(f"Failed to load dependency {dep_id}: {e}")

        context["dependencies_code"] = deps_code

        return context

    def _build_execution_prompt(
        self,
        task: "Task",
        context: Dict[str, Any],
    ) -> str:
        """
        Constrói prompt de execução para o LLM.

        Args:
            task: Task a executar
            context: Contexto preparado

        Returns:
            Prompt formatado
        """
        # Formatar dependências
        deps_str = "Nenhuma dependência."
        if context.get("dependencies_code"):
            deps_lines = []
            for dep_id, code in context["dependencies_code"].items():
                # Mostrar apenas primeiras 50 linhas
                code_preview = "\n".join(code.split("\n")[:50])
                deps_lines.append(f"### {dep_id}\n```python\n{code_preview}\n```")
            deps_str = "\n\n".join(deps_lines)

        # Formatar deliverables
        deliverables_str = "\n".join(f"- {d}" for d in task.deliverables)

        return f"""# Task de Geração de Código

## Identificação
- **Sprint**: {task.sprint_id}
- **Task ID**: {task.id}
- **Task**: {task.name}

## O Que Você Deve Fazer
{task.description}

## Arquivos a Gerar
{deliverables_str}

## Código de Dependências (para referência)
{deps_str}

## Regras Obrigatórias
1. Gere código **COMPLETO e FUNCIONAL** - não use placeholders
2. Inclua **type hints** em todas as funções
3. Inclua **docstrings** em todas as funções públicas
4. Siga os padrões PEP 8
5. Use imports absolutos (não relativos)
6. **NÃO** gere código que já existe nas dependências
7. **NÃO** use `pass` ou `...` como implementação

## Formato de Resposta
Responda com o código Python completo dentro de um bloco de código:

```python
# {task.deliverables[0] if task.deliverables else 'output.py'}

[SEU CÓDIGO COMPLETO AQUI]
```
"""

    def _extract_code(self, response: str) -> str:
        """
        Extrai código Python da resposta do LLM.

        Args:
            response: Resposta completa do LLM

        Returns:
            Código extraído

        Strategy:
            1. Procurar bloco ```python ... ```
            2. Se não encontrar, procurar ``` ... ```
            3. Se não encontrar, retornar resposta limpa
        """
        # Tentar extrair bloco python
        python_pattern = r"```python\n(.*?)```"
        matches = re.findall(python_pattern, response, re.DOTALL)
        if matches:
            # Retornar o maior bloco (provavelmente o código principal)
            return max(matches, key=len).strip()

        # Tentar extrair bloco genérico
        generic_pattern = r"```\n(.*?)```"
        matches = re.findall(generic_pattern, response, re.DOTALL)
        if matches:
            return max(matches, key=len).strip()

        # Fallback: limpar e retornar
        # Remover linhas que parecem ser explicação
        lines = response.split("\n")
        code_lines = [
            line for line in lines
            if not line.startswith("#") or line.startswith("# ")
        ]
        return "\n".join(code_lines).strip()

    async def _save_checkpoint(self, task: "Task") -> None:
        """
        Salva checkpoint da task.

        Args:
            task: Task a salvar

        Side Effects:
            - Cria arquivo JSON no checkpoint_dir
            - Atualiza task.checkpoint_path
        """
        checkpoint_path = self.checkpoint_dir / f"{task.id}.json"

        checkpoint_data = {
            "task_id": task.id,
            "sprint_id": task.sprint_id,
            "name": task.name,
            "status": task.status.value,
            "code_generated": task.code_generated,
            "execution_time_ms": task.execution_time_ms,
            "attempts": task.attempts,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        try:
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint_data, f, indent=2)

            task.checkpoint_path = str(checkpoint_path)
            logger.debug(f"Checkpoint saved: {checkpoint_path}")

        except Exception as e:
            logger.warning(f"Failed to save checkpoint for {task.id}: {e}")

    def load_from_checkpoint(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Carrega task de checkpoint se existir.

        Args:
            task_id: ID da task

        Returns:
            Dados do checkpoint ou None
        """
        checkpoint_path = self.checkpoint_dir / f"{task_id}.json"

        if not checkpoint_path.exists():
            return None

        try:
            with open(checkpoint_path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load checkpoint {task_id}: {e}")
            return None

    def clear_checkpoints(self) -> int:
        """
        Remove todos os checkpoints.

        Returns:
            Número de checkpoints removidos
        """
        count = 0
        for checkpoint_file in self.checkpoint_dir.glob("*.json"):
            try:
                checkpoint_file.unlink()
                count += 1
            except Exception as e:
                logger.warning(f"Failed to remove {checkpoint_file}: {e}")

        logger.info(f"Cleared {count} checkpoints")
        return count
