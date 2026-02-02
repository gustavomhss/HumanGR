# S03 - llm-client-base | Context Pack v1.0

---

## PRODUCT REFERENCE

```yaml
product_reference:
  product_id: "HUMANGR"
  product_name: "Human Layer MCP Server"
  wave: "W1-CoreEngine"
  product_pack: "./PRODUCT_PACK.md"
  sprint_index: "./SPRINT_INDEX.yaml"
```

---

## RELOAD ANCHOR

```yaml
sprint:
  id: S03
  name: llm-client-base
  title: "LLM Client Base & Claude"
  wave: W1-CoreEngine
  priority: P0-CRITICAL
  type: implementation

objective: "Interface base LLM e client Claude (BYOK - Bring Your Own Key)"

source_block: "../MASTER_REQUIREMENTS_MAP.md#PARTE-3-LLM"

dependencies:
  - S02  # Data Models

deliverables:
  - src/hl_mcp/llm/__init__.py
  - src/hl_mcp/llm/base.py
  - src/hl_mcp/llm/claude.py
  - src/hl_mcp/llm/exceptions.py
  - tests/test_llm/test_base.py
  - tests/test_llm/test_claude.py
```

---

## BYOK - BRING YOUR OWN KEY

```yaml
philosophy:
  core_principle: "User brings their own LLM subscription"
  supported_subscriptions:
    - "Claude Max ($20/month)"
    - "GPT Plus ($20/month)"
    - "Gemini Advanced ($20/month)"
    - "Ollama (free, local)"

  cost_to_user: "$0 for Human Layer"
  cost_for_llm: "User's existing subscription"

  why_byok:
    - "Users already have LLM subscriptions"
    - "No need to resell API at markup"
    - "User keeps full control of their keys"
    - "No vendor lock-in"
```

---

## IMPLEMENTATION SPECS

### Base Interface (llm/base.py)

```python
"""Abstract base for LLM clients."""
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

class LLMConfig(BaseModel):
    """Configuration for LLM client."""
    api_key: Optional[str] = None  # BYOK
    model: str = "claude-3-opus-20240229"
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout_seconds: int = 60

class LLMResponse(BaseModel):
    """Standardized LLM response."""
    content: str
    model: str
    tokens_used: int
    finish_reason: str
    raw_response: Optional[Dict[str, Any]] = None

class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate completion from prompt."""
        pass

    @abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        response_model: type,
        system_prompt: Optional[str] = None,
    ) -> Any:
        """Generate structured output matching Pydantic model."""
        pass

    @abstractmethod
    def validate_config(self) -> bool:
        """Validate the configuration is usable."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return provider name (claude, openai, etc)."""
        pass
```

### Claude Client (llm/claude.py)

```python
"""Claude LLM client using Anthropic SDK."""
import anthropic
from typing import Optional, Any, Type
from pydantic import BaseModel

from .base import BaseLLMClient, LLMConfig, LLMResponse
from .exceptions import LLMError, APIKeyError, RateLimitError

class ClaudeClient(BaseLLMClient):
    """Claude client using user's API key (BYOK)."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        if not config.api_key:
            raise APIKeyError("Claude API key required (BYOK)")
        self._client = anthropic.AsyncAnthropic(api_key=config.api_key)

    @property
    def provider_name(self) -> str:
        return "claude"

    def validate_config(self) -> bool:
        """Check if API key is valid."""
        return bool(self.config.api_key)

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate completion using Claude."""
        try:
            message = await self._client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}],
            )
            return LLMResponse(
                content=message.content[0].text,
                model=message.model,
                tokens_used=message.usage.input_tokens + message.usage.output_tokens,
                finish_reason=message.stop_reason,
                raw_response=message.model_dump(),
            )
        except anthropic.RateLimitError as e:
            raise RateLimitError(f"Claude rate limit: {e}")
        except Exception as e:
            raise LLMError(f"Claude error: {e}")

    async def generate_structured(
        self,
        prompt: str,
        response_model: Type[BaseModel],
        system_prompt: Optional[str] = None,
    ) -> Any:
        """Generate structured output matching Pydantic model."""
        schema_prompt = f"""
        Respond ONLY with valid JSON matching this schema:
        {response_model.model_json_schema()}

        {prompt}
        """
        response = await self.generate(schema_prompt, system_prompt)
        return response_model.model_validate_json(response.content)
```

### Exceptions (llm/exceptions.py)

```python
"""LLM-related exceptions."""

class LLMError(Exception):
    """Base LLM error."""
    pass

class APIKeyError(LLMError):
    """API key missing or invalid."""
    pass

class RateLimitError(LLMError):
    """Rate limit exceeded."""
    pass

class TokenLimitError(LLMError):
    """Token limit exceeded."""
    pass

class ModelNotFoundError(LLMError):
    """Requested model not available."""
    pass
```

---

## INTENT MANIFEST

```yaml
INTENT_MANIFEST:
  RF:
    - RF-001: "BaseLLMClient define interface para todos os providers"
    - RF-002: "ClaudeClient implementa integração com Anthropic"
    - RF-003: "generate() retorna LLMResponse padronizado"
    - RF-004: "generate_structured() retorna Pydantic model"
    - RF-005: "BYOK - API key vem do usuário"

  INV:
    - INV-001: "Nunca armazenar API keys em código"
    - INV-002: "Sempre usar async para chamadas de API"
    - INV-003: "Rate limit errors devem ser propagados"
    - INV-004: "Token usage sempre rastreado"
    - INV-005: "Config validation antes de usar"

  EDGE:
    - EDGE-001: "API key inválida → APIKeyError"
    - EDGE-002: "Rate limit → RateLimitError"
    - EDGE-003: "Network timeout → LLMError"
    - EDGE-004: "Invalid JSON response → validation error"
```

---

## GATES

```yaml
gates:
  G0_FILES_EXIST:
    description: "Arquivos de LLM existem"
    validation: |
      ls src/hl_mcp/llm/base.py
      ls src/hl_mcp/llm/claude.py
      ls src/hl_mcp/llm/exceptions.py

  G1_IMPORTS_WORK:
    description: "Imports funcionam"
    validation: |
      python -c "from hl_mcp.llm import BaseLLMClient, ClaudeClient, LLMConfig"

  G2_TESTS_PASS:
    description: "Testes passam"
    validation: "pytest tests/test_llm/ -v"

  G3_COVERAGE:
    description: "Coverage >= 90%"
    validation: "pytest tests/test_llm/ --cov=src/hl_mcp/llm --cov-fail-under=90"

  G4_BYOK_ENFORCED:
    description: "BYOK enforced - no hardcoded keys"
    validation: |
      ! grep -r "sk-ant-" src/hl_mcp/llm/
      ! grep -r "ANTHROPIC_API_KEY" src/hl_mcp/llm/*.py
```

---

## DECISION TREE

```
START S03
│
├─> Criar exceptions (llm/exceptions.py)
│   └─> LLMError, APIKeyError, RateLimitError, etc.
│
├─> Criar base interface (llm/base.py)
│   ├─> LLMConfig (api_key, model, max_tokens, etc.)
│   ├─> LLMResponse (content, tokens_used, etc.)
│   └─> BaseLLMClient (ABC)
│
├─> Criar Claude client (llm/claude.py)
│   ├─> ClaudeClient extends BaseLLMClient
│   ├─> generate() → LLMResponse
│   └─> generate_structured() → Pydantic model
│
├─> Criar llm/__init__.py
│   └─> Exportar classes públicas
│
└─> VALIDAR GATES
    ├─> G0: Arquivos existem
    ├─> G1: Imports funcionam
    ├─> G2: Testes passam
    ├─> G3: Coverage >= 90%
    └─> G4: BYOK enforced
```

---

## REFERÊNCIA

Para detalhes completos, consulte:
- `./S02_CONTEXT.md` - Models utilizados
- `./S04_CONTEXT.md` - Providers adicionais
- `../MASTER_REQUIREMENTS_MAP.md` - PARTE 3: LLM
