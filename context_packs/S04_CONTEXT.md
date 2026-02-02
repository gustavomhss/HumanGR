# S04 - llm-providers | Context Pack v1.0

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
  id: S04
  name: llm-providers
  title: "Additional LLM Providers"
  wave: W1-CoreEngine
  priority: P1-HIGH
  type: implementation

objective: "Implementar clients OpenAI, Gemini, Ollama e factory"

source_block: "../MASTER_REQUIREMENTS_MAP.md#PARTE-3-LLM"

dependencies:
  - S03  # LLM Base

deliverables:
  - src/hl_mcp/llm/openai.py
  - src/hl_mcp/llm/gemini.py
  - src/hl_mcp/llm/ollama.py
  - src/hl_mcp/llm/factory.py
  - tests/test_llm/test_openai.py
  - tests/test_llm/test_gemini.py
  - tests/test_llm/test_ollama.py
  - tests/test_llm/test_factory.py
```

---

## BYOK PROVIDERS

```yaml
providers:
  claude:
    subscription: "Claude Max $20/month"
    api_key_env: "ANTHROPIC_API_KEY"
    models: ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"]
    status: "S03 - implemented"

  openai:
    subscription: "GPT Plus $20/month"
    api_key_env: "OPENAI_API_KEY"
    models: ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
    status: "S04 - this sprint"

  gemini:
    subscription: "Gemini Advanced $20/month"
    api_key_env: "GOOGLE_API_KEY"
    models: ["gemini-1.5-pro", "gemini-1.5-flash"]
    status: "S04 - this sprint"

  ollama:
    subscription: "Free (local)"
    api_key_env: null
    models: ["llama3.1", "mistral", "codellama"]
    status: "S04 - this sprint"
```

---

## IMPLEMENTATION SPECS

### OpenAI Client (llm/openai.py)

```python
"""OpenAI LLM client."""
from openai import AsyncOpenAI
from typing import Optional, Type, Any
from pydantic import BaseModel

from .base import BaseLLMClient, LLMConfig, LLMResponse
from .exceptions import LLMError, APIKeyError, RateLimitError

class OpenAIClient(BaseLLMClient):
    """OpenAI client using user's API key (BYOK)."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        if not config.api_key:
            raise APIKeyError("OpenAI API key required (BYOK)")
        self._client = AsyncOpenAI(api_key=config.api_key)

    @property
    def provider_name(self) -> str:
        return "openai"

    def validate_config(self) -> bool:
        return bool(self.config.api_key)

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate completion using OpenAI."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await self._client.chat.completions.create(
                model=self.config.model or "gpt-4o",
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                tokens_used=response.usage.total_tokens,
                finish_reason=response.choices[0].finish_reason,
                raw_response=response.model_dump(),
            )
        except Exception as e:
            if "rate_limit" in str(e).lower():
                raise RateLimitError(f"OpenAI rate limit: {e}")
            raise LLMError(f"OpenAI error: {e}")

    async def generate_structured(
        self,
        prompt: str,
        response_model: Type[BaseModel],
        system_prompt: Optional[str] = None,
    ) -> Any:
        """Generate structured output using JSON mode."""
        schema_prompt = f"""
        Respond ONLY with valid JSON matching this schema:
        {response_model.model_json_schema()}

        {prompt}
        """
        response = await self.generate(schema_prompt, system_prompt)
        return response_model.model_validate_json(response.content)
```

### Gemini Client (llm/gemini.py)

```python
"""Google Gemini LLM client."""
import google.generativeai as genai
from typing import Optional, Type, Any
from pydantic import BaseModel

from .base import BaseLLMClient, LLMConfig, LLMResponse
from .exceptions import LLMError, APIKeyError

class GeminiClient(BaseLLMClient):
    """Gemini client using user's API key (BYOK)."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        if not config.api_key:
            raise APIKeyError("Google API key required (BYOK)")
        genai.configure(api_key=config.api_key)
        self._model = genai.GenerativeModel(config.model or "gemini-1.5-pro")

    @property
    def provider_name(self) -> str:
        return "gemini"

    def validate_config(self) -> bool:
        return bool(self.config.api_key)

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate completion using Gemini."""
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        try:
            response = await self._model.generate_content_async(full_prompt)
            return LLMResponse(
                content=response.text,
                model=self.config.model,
                tokens_used=response.usage_metadata.total_token_count,
                finish_reason="stop",
                raw_response={"text": response.text},
            )
        except Exception as e:
            raise LLMError(f"Gemini error: {e}")

    async def generate_structured(
        self,
        prompt: str,
        response_model: Type[BaseModel],
        system_prompt: Optional[str] = None,
    ) -> Any:
        """Generate structured output."""
        schema_prompt = f"""
        Respond ONLY with valid JSON matching this schema:
        {response_model.model_json_schema()}

        {prompt}
        """
        response = await self.generate(schema_prompt, system_prompt)
        return response_model.model_validate_json(response.content)
```

### Ollama Client (llm/ollama.py)

```python
"""Ollama LLM client (local, free)."""
import httpx
from typing import Optional, Type, Any
from pydantic import BaseModel

from .base import BaseLLMClient, LLMConfig, LLMResponse
from .exceptions import LLMError

class OllamaClient(BaseLLMClient):
    """Ollama client for local models (free)."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._base_url = config.api_key or "http://localhost:11434"
        self._client = httpx.AsyncClient(base_url=self._base_url)

    @property
    def provider_name(self) -> str:
        return "ollama"

    def validate_config(self) -> bool:
        return True  # Ollama doesn't need API key

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate completion using Ollama."""
        try:
            response = await self._client.post(
                "/api/generate",
                json={
                    "model": self.config.model or "llama3.1",
                    "prompt": prompt,
                    "system": system_prompt or "",
                    "stream": False,
                },
            )
            data = response.json()
            return LLMResponse(
                content=data["response"],
                model=data.get("model", self.config.model),
                tokens_used=data.get("eval_count", 0),
                finish_reason="stop",
                raw_response=data,
            )
        except Exception as e:
            raise LLMError(f"Ollama error: {e}")

    async def generate_structured(
        self,
        prompt: str,
        response_model: Type[BaseModel],
        system_prompt: Optional[str] = None,
    ) -> Any:
        """Generate structured output."""
        schema_prompt = f"""
        Respond ONLY with valid JSON matching this schema:
        {response_model.model_json_schema()}

        {prompt}
        """
        response = await self.generate(schema_prompt, system_prompt)
        return response_model.model_validate_json(response.content)
```

### Factory (llm/factory.py)

```python
"""LLM client factory."""
from typing import Optional
from .base import BaseLLMClient, LLMConfig
from .claude import ClaudeClient
from .openai import OpenAIClient
from .gemini import GeminiClient
from .ollama import OllamaClient
from .exceptions import ModelNotFoundError

PROVIDERS = {
    "claude": ClaudeClient,
    "anthropic": ClaudeClient,
    "openai": OpenAIClient,
    "gpt": OpenAIClient,
    "gemini": GeminiClient,
    "google": GeminiClient,
    "ollama": OllamaClient,
    "local": OllamaClient,
}

def create_llm_client(
    provider: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> BaseLLMClient:
    """Create LLM client by provider name.

    Args:
        provider: Provider name (claude, openai, gemini, ollama)
        api_key: User's API key (BYOK)
        model: Model name (optional, uses provider default)
        **kwargs: Additional config options

    Returns:
        Configured LLM client

    Raises:
        ModelNotFoundError: If provider not supported
    """
    provider_lower = provider.lower()
    if provider_lower not in PROVIDERS:
        raise ModelNotFoundError(
            f"Unknown provider: {provider}. "
            f"Supported: {list(PROVIDERS.keys())}"
        )

    config = LLMConfig(
        api_key=api_key,
        model=model,
        **kwargs
    )
    return PROVIDERS[provider_lower](config)


def get_available_providers() -> list[str]:
    """Get list of unique provider names."""
    return ["claude", "openai", "gemini", "ollama"]
```

---

## INTENT MANIFEST

```yaml
INTENT_MANIFEST:
  RF:
    - RF-001: "OpenAIClient suporta GPT-4o, GPT-4-turbo"
    - RF-002: "GeminiClient suporta Gemini 1.5 Pro/Flash"
    - RF-003: "OllamaClient suporta modelos locais"
    - RF-004: "Factory cria client por nome do provider"
    - RF-005: "Todos os clients seguem BaseLLMClient interface"

  INV:
    - INV-001: "BYOK enforced - keys vêm do usuário"
    - INV-002: "Ollama não requer API key"
    - INV-003: "Factory não expõe keys em erros"
    - INV-004: "Todos os providers são async"

  EDGE:
    - EDGE-001: "Provider desconhecido → ModelNotFoundError"
    - EDGE-002: "Ollama offline → LLMError"
    - EDGE-003: "API key inválida → APIKeyError"
```

---

## GATES

```yaml
gates:
  G0_FILES_EXIST:
    description: "Arquivos de providers existem"
    validation: |
      ls src/hl_mcp/llm/openai.py
      ls src/hl_mcp/llm/gemini.py
      ls src/hl_mcp/llm/ollama.py
      ls src/hl_mcp/llm/factory.py

  G1_IMPORTS_WORK:
    description: "Imports funcionam"
    validation: |
      python -c "from hl_mcp.llm import create_llm_client, get_available_providers"
      python -c "from hl_mcp.llm import OpenAIClient, GeminiClient, OllamaClient"

  G2_TESTS_PASS:
    description: "Testes passam"
    validation: "pytest tests/test_llm/ -v"

  G3_COVERAGE:
    description: "Coverage >= 90%"
    validation: "pytest tests/test_llm/ --cov=src/hl_mcp/llm --cov-fail-under=90"

  G4_FACTORY_WORKS:
    description: "Factory cria clients corretamente"
    validation: |
      python -c "
      from hl_mcp.llm import create_llm_client, get_available_providers
      assert set(get_available_providers()) == {'claude', 'openai', 'gemini', 'ollama'}
      "
```

---

## DECISION TREE

```
START S04
│
├─> Criar OpenAI client (llm/openai.py)
│   └─> GPT-4o, GPT-4-turbo support
│
├─> Criar Gemini client (llm/gemini.py)
│   └─> Gemini 1.5 Pro/Flash support
│
├─> Criar Ollama client (llm/ollama.py)
│   └─> Local models (llama3.1, mistral)
│
├─> Criar factory (llm/factory.py)
│   ├─> create_llm_client(provider, api_key, model)
│   └─> get_available_providers()
│
├─> Atualizar llm/__init__.py
│   └─> Exportar todas as classes e factory
│
└─> VALIDAR GATES
    ├─> G0: Arquivos existem
    ├─> G1: Imports funcionam
    ├─> G2: Testes passam
    ├─> G3: Coverage >= 90%
    └─> G4: Factory funciona
```

---

## REFERÊNCIA

Para detalhes completos, consulte:
- `./S03_CONTEXT.md` - LLM Base interface
- `./S05_CONTEXT.md` - Layer Base
- `../MASTER_REQUIREMENTS_MAP.md` - PARTE 3: LLM
