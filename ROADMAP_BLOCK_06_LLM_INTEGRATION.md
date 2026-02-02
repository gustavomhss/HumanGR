# Human Layer MCP Server - Roadmap Block 06
# LLM INTEGRATION

> **Objetivo**: Implementar integração com LLMs (Claude, OpenAI)
> **Módulos**: HLS-LLM-001 a HLS-LLM-005
> **Versão**: 1.0.0 | Data: 2026-02-01

---

## VISÃO GERAL

A camada LLM é o **cérebro** do Human Layer. Ela:
1. Define interface abstrata para qualquer LLM
2. Implementa clients específicos (Claude, OpenAI)
3. Gerencia templates de prompts
4. Parseia respostas estruturadas

```
┌─────────────────────────────────────────────────────────────────┐
│                        LLM LAYER                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │ LLMClient   │    │  Prompts    │    │   Parser    │         │
│  │ (interface) │───▶│ (templates) │───▶│ (response)  │         │
│  └──────┬──────┘    └─────────────┘    └─────────────┘         │
│         │                                                       │
│    ┌────┴────┐                                                  │
│    │         │                                                  │
│  ┌─▼───┐  ┌──▼──┐                                              │
│  │Claude│  │OpenAI│                                             │
│  └─────┘  └─────┘                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. HLS-LLM-001: LLMClient Interface

```python
# src/hl_mcp/llm/client.py
"""
Module: HLS-LLM-001 - LLMClient
===============================

Abstract interface for LLM clients.

Defines the contract that all LLM implementations must follow.
Supports both sync and async operations.

Example:
    >>> client: LLMClient = get_llm_client("claude")
    >>> response = await client.complete(
    ...     prompt="Analyze this code for security issues",
    ...     system="You are a security expert",
    ... )

Dependencies:
    - None (interface module)

Author: Human Layer Team
Version: 1.0.0
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Union
import hashlib


class LLMProvider(Enum):
    """Supported LLM providers."""
    CLAUDE = "claude"
    OPENAI = "openai"
    LOCAL = "local"  # For local models via Ollama etc


class MessageRole(Enum):
    """Message roles in conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """A message in a conversation.

    Attributes:
        role: Who sent the message
        content: Message content
        name: Optional name (for tool messages)
        tool_call_id: ID if this is a tool response
    """

    role: MessageRole
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API format."""
        d = {"role": self.role.value, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d

    @classmethod
    def system(cls, content: str) -> "Message":
        """Create system message."""
        return cls(role=MessageRole.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        """Create user message."""
        return cls(role=MessageRole.USER, content=content)

    @classmethod
    def assistant(cls, content: str) -> "Message":
        """Create assistant message."""
        return cls(role=MessageRole.ASSISTANT, content=content)


@dataclass
class LLMConfig:
    """Configuration for LLM client.

    Attributes:
        model: Model identifier
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens in response
        top_p: Nucleus sampling parameter
        stop_sequences: Sequences that stop generation
        timeout_seconds: Request timeout
    """

    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    stop_sequences: List[str] = field(default_factory=list)
    timeout_seconds: int = 120

    # Provider-specific settings
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be 0.0-2.0, got {self.temperature}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")


@dataclass
class TokenUsage:
    """Token usage statistics.

    Attributes:
        prompt_tokens: Tokens in the prompt
        completion_tokens: Tokens in the completion
        total_tokens: Total tokens used
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @property
    def estimated_cost_usd(self) -> float:
        """Rough cost estimate (Claude pricing)."""
        # Approximate: $3/1M input, $15/1M output for Claude Sonnet
        input_cost = (self.prompt_tokens / 1_000_000) * 3
        output_cost = (self.completion_tokens / 1_000_000) * 15
        return round(input_cost + output_cost, 6)


@dataclass
class LLMResponse:
    """Response from LLM completion.

    Attributes:
        content: The generated text
        model: Model that generated this
        usage: Token usage statistics
        finish_reason: Why generation stopped
        latency_ms: Response latency
        raw_response: Original API response
    """

    content: str
    model: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)
    finish_reason: str = "stop"
    latency_ms: int = 0

    response_id: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    raw_response: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Generate response ID if not provided."""
        if not self.response_id:
            content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:8]
            self.response_id = f"resp_{content_hash}"

    @property
    def is_complete(self) -> bool:
        """Check if generation completed normally."""
        return self.finish_reason in ("stop", "end_turn")

    @property
    def was_truncated(self) -> bool:
        """Check if response was truncated due to length."""
        return self.finish_reason in ("length", "max_tokens")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "response_id": self.response_id,
            "content": self.content,
            "model": self.model,
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
                "estimated_cost_usd": self.usage.estimated_cost_usd,
            },
            "finish_reason": self.finish_reason,
            "latency_ms": self.latency_ms,
            "is_complete": self.is_complete,
            "created_at": self.created_at.isoformat(),
        }


class LLMClient(ABC):
    """Abstract base class for LLM clients.

    All LLM implementations must inherit from this class
    and implement the abstract methods.

    Example:
        >>> class MyLLMClient(LLMClient):
        ...     async def complete(self, prompt, **kwargs):
        ...         # Implementation
        ...         pass
    """

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        """Initialize client with configuration.

        Args:
            config: Client configuration
        """
        self.config = config or LLMConfig()
        self._request_count = 0
        self._total_tokens = 0

    @property
    @abstractmethod
    def provider(self) -> LLMProvider:
        """Get the provider type."""
        pass

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Get the default model for this provider."""
        pass

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        messages: Optional[List[Message]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion.

        Args:
            prompt: The user prompt
            system: Optional system message
            messages: Optional conversation history
            **kwargs: Additional provider-specific options

        Returns:
            LLMResponse with generated content
        """
        pass

    @abstractmethod
    async def complete_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        messages: Optional[List[Message]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Generate a streaming completion.

        Args:
            prompt: The user prompt
            system: Optional system message
            messages: Optional conversation history
            **kwargs: Additional options

        Yields:
            Chunks of generated text
        """
        pass

    async def complete_json(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate a JSON response matching a schema.

        Args:
            prompt: The user prompt
            schema: JSON schema for the response
            system: Optional system message
            **kwargs: Additional options

        Returns:
            Parsed JSON response
        """
        import json

        # Add schema instruction to prompt
        schema_instruction = f"\n\nRespond with valid JSON matching this schema:\n```json\n{json.dumps(schema, indent=2)}\n```"

        response = await self.complete(
            prompt=prompt + schema_instruction,
            system=system,
            **kwargs,
        )

        # Parse JSON from response
        content = response.content.strip()

        # Handle markdown code blocks
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])

        return json.loads(content)

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            "provider": self.provider.value,
            "request_count": self._request_count,
            "total_tokens": self._total_tokens,
            "config": {
                "model": self.config.model or self.default_model,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            },
        }

    def _update_stats(self, response: LLMResponse) -> None:
        """Update internal statistics."""
        self._request_count += 1
        self._total_tokens += response.usage.total_tokens


# ============================================================================
# FACTORY
# ============================================================================

_clients: Dict[LLMProvider, type] = {}


def register_client(provider: LLMProvider):
    """Decorator to register an LLM client implementation."""
    def decorator(cls: type) -> type:
        _clients[provider] = cls
        return cls
    return decorator


def get_llm_client(
    provider: Union[str, LLMProvider],
    config: Optional[LLMConfig] = None,
    **kwargs: Any,
) -> LLMClient:
    """Get an LLM client for the specified provider.

    Args:
        provider: Provider name or enum
        config: Optional configuration
        **kwargs: Provider-specific arguments

    Returns:
        Configured LLMClient instance

    Raises:
        ValueError: If provider is not supported
    """
    if isinstance(provider, str):
        provider = LLMProvider(provider.lower())

    if provider not in _clients:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(_clients.keys())}")

    return _clients[provider](config=config, **kwargs)


def list_providers() -> List[str]:
    """List available providers."""
    return [p.value for p in _clients.keys()]


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "LLMProvider",
    "MessageRole",
    # Data classes
    "Message",
    "LLMConfig",
    "TokenUsage",
    "LLMResponse",
    # Abstract class
    "LLMClient",
    # Factory
    "register_client",
    "get_llm_client",
    "list_providers",
]
```

---

## 2. HLS-LLM-002: ClaudeClient

```python
# src/hl_mcp/llm/claude.py
"""
Module: HLS-LLM-002 - ClaudeClient
==================================

Claude/Anthropic LLM client implementation.

Implements the LLMClient interface for Anthropic's Claude models.

Example:
    >>> client = ClaudeClient(api_key="sk-ant-...")
    >>> response = await client.complete(
    ...     prompt="Explain quantum computing",
    ...     system="You are a physics teacher",
    ... )

Dependencies:
    - HLS-LLM-001: LLMClient interface
    - anthropic: Anthropic Python SDK

Author: Human Layer Team
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from .client import (
    LLMClient,
    LLMConfig,
    LLMProvider,
    LLMResponse,
    Message,
    MessageRole,
    TokenUsage,
    register_client,
)


@register_client(LLMProvider.CLAUDE)
class ClaudeClient(LLMClient):
    """Anthropic Claude client implementation.

    Supports Claude 3 family models (Opus, Sonnet, Haiku).

    Attributes:
        api_key: Anthropic API key
        client: Anthropic client instance

    Example:
        >>> client = ClaudeClient()  # Uses ANTHROPIC_API_KEY env var
        >>> response = await client.complete("Hello!")
        >>> print(response.content)
    """

    # Model identifiers
    CLAUDE_OPUS = "claude-3-opus-20240229"
    CLAUDE_SONNET = "claude-3-5-sonnet-20241022"
    CLAUDE_HAIKU = "claude-3-haiku-20240307"

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[LLMConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Claude client.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            config: Client configuration
            **kwargs: Additional arguments
        """
        super().__init__(config)

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY env var or pass api_key."
            )

        # Import here to make anthropic optional
        try:
            import anthropic
            self._anthropic = anthropic
            self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
            self.sync_client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "anthropic package required. Install with: pip install anthropic"
            )

    @property
    def provider(self) -> LLMProvider:
        return LLMProvider.CLAUDE

    @property
    def default_model(self) -> str:
        return self.CLAUDE_SONNET

    def _get_model(self) -> str:
        """Get model to use."""
        return self.config.model or self.default_model

    def _build_messages(
        self,
        prompt: str,
        messages: Optional[List[Message]] = None,
    ) -> List[Dict[str, str]]:
        """Build message list for API call."""
        result = []

        # Add conversation history
        if messages:
            for msg in messages:
                if msg.role != MessageRole.SYSTEM:  # System handled separately
                    result.append({
                        "role": msg.role.value,
                        "content": msg.content,
                    })

        # Add current prompt
        result.append({
            "role": "user",
            "content": prompt,
        })

        return result

    async def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        messages: Optional[List[Message]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion using Claude.

        Args:
            prompt: The user prompt
            system: Optional system message
            messages: Optional conversation history
            **kwargs: Additional options (temperature, max_tokens, etc.)

        Returns:
            LLMResponse with generated content
        """
        start_time = time.time()

        # Build request
        request_messages = self._build_messages(prompt, messages)

        # Merge config with kwargs
        temperature = kwargs.get("temperature", self.config.temperature)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)

        try:
            response = await self.client.messages.create(
                model=self._get_model(),
                max_tokens=max_tokens,
                temperature=temperature,
                system=system or "You are a helpful assistant.",
                messages=request_messages,
                stop_sequences=self.config.stop_sequences or None,
            )

            latency_ms = int((time.time() - start_time) * 1000)

            # Extract content
            content = ""
            if response.content:
                content = response.content[0].text

            # Build response
            llm_response = LLMResponse(
                content=content,
                model=response.model,
                usage=TokenUsage(
                    prompt_tokens=response.usage.input_tokens,
                    completion_tokens=response.usage.output_tokens,
                    total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                ),
                finish_reason=response.stop_reason or "stop",
                latency_ms=latency_ms,
                raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
            )

            self._update_stats(llm_response)
            return llm_response

        except self._anthropic.APIError as e:
            raise RuntimeError(f"Claude API error: {e}")

    async def complete_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        messages: Optional[List[Message]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Generate a streaming completion.

        Args:
            prompt: The user prompt
            system: Optional system message
            messages: Optional conversation history
            **kwargs: Additional options

        Yields:
            Chunks of generated text
        """
        request_messages = self._build_messages(prompt, messages)

        temperature = kwargs.get("temperature", self.config.temperature)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)

        try:
            async with self.client.messages.stream(
                model=self._get_model(),
                max_tokens=max_tokens,
                temperature=temperature,
                system=system or "You are a helpful assistant.",
                messages=request_messages,
            ) as stream:
                async for text in stream.text_stream:
                    yield text

        except self._anthropic.APIError as e:
            raise RuntimeError(f"Claude streaming error: {e}")

    def complete_sync(
        self,
        prompt: str,
        system: Optional[str] = None,
        messages: Optional[List[Message]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Synchronous completion (for non-async contexts).

        Args:
            prompt: The user prompt
            system: Optional system message
            messages: Optional conversation history
            **kwargs: Additional options

        Returns:
            LLMResponse with generated content
        """
        start_time = time.time()

        request_messages = self._build_messages(prompt, messages)
        temperature = kwargs.get("temperature", self.config.temperature)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)

        try:
            response = self.sync_client.messages.create(
                model=self._get_model(),
                max_tokens=max_tokens,
                temperature=temperature,
                system=system or "You are a helpful assistant.",
                messages=request_messages,
            )

            latency_ms = int((time.time() - start_time) * 1000)

            content = ""
            if response.content:
                content = response.content[0].text

            return LLMResponse(
                content=content,
                model=response.model,
                usage=TokenUsage(
                    prompt_tokens=response.usage.input_tokens,
                    completion_tokens=response.usage.output_tokens,
                    total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                ),
                finish_reason=response.stop_reason or "stop",
                latency_ms=latency_ms,
            )

        except self._anthropic.APIError as e:
            raise RuntimeError(f"Claude API error: {e}")


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = ["ClaudeClient"]
```

---

## 3. HLS-LLM-003: OpenAIClient

```python
# src/hl_mcp/llm/openai.py
"""
Module: HLS-LLM-003 - OpenAIClient
==================================

OpenAI LLM client implementation.

Implements the LLMClient interface for OpenAI's GPT models.

Example:
    >>> client = OpenAIClient(api_key="sk-...")
    >>> response = await client.complete(
    ...     prompt="Write a haiku about coding",
    ... )

Dependencies:
    - HLS-LLM-001: LLMClient interface
    - openai: OpenAI Python SDK

Author: Human Layer Team
Version: 1.0.0
"""

from __future__ import annotations

import os
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from .client import (
    LLMClient,
    LLMConfig,
    LLMProvider,
    LLMResponse,
    Message,
    MessageRole,
    TokenUsage,
    register_client,
)


@register_client(LLMProvider.OPENAI)
class OpenAIClient(LLMClient):
    """OpenAI GPT client implementation.

    Supports GPT-4, GPT-4 Turbo, and GPT-3.5 Turbo models.

    Attributes:
        api_key: OpenAI API key
        client: OpenAI client instance

    Example:
        >>> client = OpenAIClient()  # Uses OPENAI_API_KEY env var
        >>> response = await client.complete("Hello!")
    """

    # Model identifiers
    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-turbo-preview"
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_35_TURBO = "gpt-3.5-turbo"

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[LLMConfig] = None,
        organization: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize OpenAI client.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            config: Client configuration
            organization: Optional organization ID
            **kwargs: Additional arguments
        """
        super().__init__(config)

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key."
            )

        self.organization = organization or os.getenv("OPENAI_ORG_ID")

        # Import here to make openai optional
        try:
            import openai
            self._openai = openai
            self.client = openai.AsyncOpenAI(
                api_key=self.api_key,
                organization=self.organization,
            )
            self.sync_client = openai.OpenAI(
                api_key=self.api_key,
                organization=self.organization,
            )
        except ImportError:
            raise ImportError(
                "openai package required. Install with: pip install openai"
            )

    @property
    def provider(self) -> LLMProvider:
        return LLMProvider.OPENAI

    @property
    def default_model(self) -> str:
        return self.GPT_4O

    def _get_model(self) -> str:
        """Get model to use."""
        return self.config.model or self.default_model

    def _build_messages(
        self,
        prompt: str,
        system: Optional[str] = None,
        messages: Optional[List[Message]] = None,
    ) -> List[Dict[str, str]]:
        """Build message list for API call."""
        result = []

        # Add system message
        if system:
            result.append({
                "role": "system",
                "content": system,
            })

        # Add conversation history
        if messages:
            for msg in messages:
                result.append({
                    "role": msg.role.value,
                    "content": msg.content,
                })

        # Add current prompt
        result.append({
            "role": "user",
            "content": prompt,
        })

        return result

    async def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        messages: Optional[List[Message]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion using OpenAI.

        Args:
            prompt: The user prompt
            system: Optional system message
            messages: Optional conversation history
            **kwargs: Additional options

        Returns:
            LLMResponse with generated content
        """
        start_time = time.time()

        request_messages = self._build_messages(prompt, system, messages)

        temperature = kwargs.get("temperature", self.config.temperature)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)

        try:
            response = await self.client.chat.completions.create(
                model=self._get_model(),
                messages=request_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=self.config.stop_sequences or None,
            )

            latency_ms = int((time.time() - start_time) * 1000)

            # Extract content
            content = ""
            if response.choices:
                content = response.choices[0].message.content or ""

            finish_reason = "stop"
            if response.choices:
                finish_reason = response.choices[0].finish_reason or "stop"

            # Build usage
            usage = TokenUsage()
            if response.usage:
                usage = TokenUsage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )

            llm_response = LLMResponse(
                content=content,
                model=response.model,
                usage=usage,
                finish_reason=finish_reason,
                latency_ms=latency_ms,
                response_id=response.id,
                raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
            )

            self._update_stats(llm_response)
            return llm_response

        except self._openai.APIError as e:
            raise RuntimeError(f"OpenAI API error: {e}")

    async def complete_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        messages: Optional[List[Message]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Generate a streaming completion.

        Args:
            prompt: The user prompt
            system: Optional system message
            messages: Optional conversation history
            **kwargs: Additional options

        Yields:
            Chunks of generated text
        """
        request_messages = self._build_messages(prompt, system, messages)

        temperature = kwargs.get("temperature", self.config.temperature)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)

        try:
            stream = await self.client.chat.completions.create(
                model=self._get_model(),
                messages=request_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except self._openai.APIError as e:
            raise RuntimeError(f"OpenAI streaming error: {e}")

    async def complete_json(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate a JSON response using OpenAI's JSON mode.

        Args:
            prompt: The user prompt
            schema: JSON schema for the response
            system: Optional system message
            **kwargs: Additional options

        Returns:
            Parsed JSON response
        """
        import json

        # Use OpenAI's JSON mode
        request_messages = self._build_messages(
            prompt=f"{prompt}\n\nRespond with JSON matching this schema:\n{json.dumps(schema, indent=2)}",
            system=system,
        )

        response = await self.client.chat.completions.create(
            model=self._get_model(),
            messages=request_messages,
            response_format={"type": "json_object"},
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
        )

        content = response.choices[0].message.content or "{}"
        return json.loads(content)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = ["OpenAIClient"]
```

---

## 4. HLS-LLM-004: PromptTemplates

```python
# src/hl_mcp/llm/prompts.py
"""
Module: HLS-LLM-004 - PromptTemplates
=====================================

Prompt template management for Human Layer.

Provides structured templates for all layer prompts,
with variable substitution and versioning.

Example:
    >>> templates = PromptTemplates()
    >>> prompt = templates.render("hl_1_usuario", {
    ...     "artifact": code_content,
    ...     "artifact_type": "Python code",
    ... })

Dependencies:
    - HLS-MDL-007: Enums (LayerID, PerspectiveID)

Author: Human Layer Team
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import re


@dataclass
class PromptTemplate:
    """A prompt template with variable substitution.

    Attributes:
        id: Template identifier
        name: Human-readable name
        template: The template string with {variables}
        variables: List of required variables
        version: Template version
        description: What this template is for
    """

    id: str
    name: str
    template: str
    variables: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    description: str = ""

    def __post_init__(self) -> None:
        """Extract variables from template."""
        if not self.variables:
            # Auto-detect variables from template
            self.variables = re.findall(r'\{(\w+)\}', self.template)

    def render(self, context: Dict[str, Any]) -> str:
        """Render template with context.

        Args:
            context: Variable values

        Returns:
            Rendered template

        Raises:
            ValueError: If required variable is missing
        """
        # Check all required variables are provided
        missing = [v for v in self.variables if v not in context]
        if missing:
            raise ValueError(f"Missing template variables: {missing}")

        # Simple string substitution
        result = self.template
        for key, value in context.items():
            result = result.replace(f"{{{key}}}", str(value))

        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "template": self.template,
            "variables": self.variables,
            "version": self.version,
            "description": self.description,
        }


# ============================================================================
# LAYER TEMPLATES
# ============================================================================

LAYER_TEMPLATES: Dict[str, PromptTemplate] = {
    # HL-1 Usuario
    "hl_1_usuario": PromptTemplate(
        id="hl_1_usuario",
        name="HL-1 Humano Usuario",
        description="Usability validation from user perspective",
        template="""You are HL-1 Humano Usuario - a usability expert who thinks like a real user.

Your core question: "É usável? O usuário consegue completar a tarefa sem ajuda?"

## Artifact to Validate
Type: {artifact_type}
```
{artifact}
```

## Your Task
Analyze this artifact from a USER perspective. Consider:
1. Is the flow intuitive?
2. Are error messages helpful or confusing?
3. Is there unnecessary friction?
4. Can a user complete the task without documentation?

## Red Flags to Watch For
- Technical jargon exposed to users
- Flows with more than 5 steps
- Generic errors without guidance
- Irreversible actions without confirmation
- Missing feedback on actions

## Output Format
Respond with a JSON object:
```json
{{
    "status": "PASS" | "WARN" | "FAIL",
    "veto": "NONE" | "WEAK",
    "usability_score": 0.0-1.0,
    "findings": [
        {{
            "severity": "critical" | "high" | "medium" | "low",
            "title": "Short title",
            "description": "Detailed description",
            "location": "Where in the artifact",
            "fix_hint": "How to fix this"
        }}
    ],
    "summary": "One paragraph summary"
}}
```

Remember: You have WEAK veto power. You can warn but not block.""",
    ),

    # HL-2 Operador
    "hl_2_operador": PromptTemplate(
        id="hl_2_operador",
        name="HL-2 Humano Operador",
        description="Operability validation from operator perspective",
        template="""You are HL-2 Humano Operador - an on-call engineer who needs to operate this system.

Your core question: "Consigo operar isso às 3AM com sono?"

## Artifact to Validate
Type: {artifact_type}
```
{artifact}
```

## Your Task
Analyze from an OPERATOR perspective. Consider:
1. How do I know if it's working?
2. How do I diagnose problems?
3. How do I rollback if it breaks?
4. Is there a runbook?

## Red Flags to Watch For
- No structured logs
- No health metrics
- Complex manual rollback
- Dependencies without circuit breakers
- Silent failures

## Output Format
Respond with a JSON object:
```json
{{
    "status": "PASS" | "WARN" | "FAIL",
    "veto": "NONE" | "WEAK" | "MEDIUM",
    "operability_score": 0.0-1.0,
    "findings": [
        {{
            "severity": "critical" | "high" | "medium" | "low",
            "title": "Short title",
            "description": "Detailed description",
            "fix_hint": "How to fix this"
        }}
    ],
    "diagnostics_available": true | false,
    "rollback_possible": true | false,
    "summary": "One paragraph summary"
}}
```

Remember: You have MEDIUM veto power. You can block merge but not promotion.""",
    ),

    # HL-4 Decisor
    "hl_4_decisor": PromptTemplate(
        id="hl_4_decisor",
        name="HL-4 Humano Decisor",
        description="Strategic validation from decision-maker perspective",
        template="""You are HL-4 Humano Decisor - a strategic decision maker who evaluates trust and alignment.

Your core question: "Faz sentido estratégico? Consigo confiar e explicar isso?"

## Artifact to Validate
Type: {artifact_type}
Context: {context}
```
{artifact}
```

## Your Task
Analyze from a STRATEGIC perspective. Consider:
1. Does this align with business goals?
2. Does the benefit justify the risk?
3. Can I explain this to stakeholders?
4. Do I trust this approach?

## Red Flags to Watch For
- Negative or uncertain ROI
- Conflicts with priorities
- Reputational risk
- Irreversible decisions without Plan B
- Unexplainable complexity

## Output Format
Respond with a JSON object:
```json
{{
    "status": "PASS" | "WARN" | "FAIL",
    "veto": "NONE" | "WEAK" | "MEDIUM" | "STRONG",
    "trust_score": 0.0-1.0,
    "alignment_score": 0.0-1.0,
    "findings": [
        {{
            "severity": "critical" | "high" | "medium" | "low",
            "title": "Short title",
            "description": "Detailed description",
            "fix_hint": "How to fix this"
        }}
    ],
    "recommendation": "PROCEED" | "REVIEW" | "REJECT",
    "summary": "One paragraph summary"
}}
```

Remember: You have STRONG veto power. You can block everything.""",
    ),

    # HL-5 Seguranca
    "hl_5_seguranca": PromptTemplate(
        id="hl_5_seguranca",
        name="HL-5 Humano Seguranca",
        description="Safety validation - accidental security issues",
        template="""You are HL-5 Humano Seguranca - a security-minded reviewer focused on ACCIDENTAL harm.

Your core question: "Isso pode causar dano por ACIDENTE?"

## Artifact to Validate
Type: {artifact_type}
```
{artifact}
```

## Your Task
Analyze for ACCIDENTAL security issues. Consider:
1. Can this cause unintended harm?
2. Are sensitive data properly protected?
3. Does it fail safely?
4. Is the blast radius limited?

## Security Checks (OWASP-aligned)
- A01: Broken Access Control
- A02: Cryptographic Failures
- A03: Injection
- A07: Auth Failures

## Red Flags to Watch For
- Data in plaintext
- SQL without parameterization
- Hardcoded secrets
- Excessive permissions
- Missing input validation

## Output Format
Respond with a JSON object:
```json
{{
    "status": "PASS" | "WARN" | "FAIL",
    "veto": "NONE" | "WEAK" | "MEDIUM" | "STRONG",
    "security_score": 0.0-1.0,
    "findings": [
        {{
            "severity": "critical" | "high" | "medium" | "low",
            "title": "Short title",
            "description": "Detailed description",
            "cwe_id": "CWE-XXX if applicable",
            "owasp_category": "A0X if applicable",
            "fix_hint": "How to fix this"
        }}
    ],
    "blast_radius": "low" | "medium" | "high",
    "summary": "One paragraph summary"
}}
```

Remember: You have STRONG veto power. Security issues can block everything.""",
    ),

    # HL-6 Hacker
    "hl_6_hacker": PromptTemplate(
        id="hl_6_hacker",
        name="HL-6 Humano Hacker",
        description="Adversarial validation - intentional exploitation",
        template="""You are HL-6 Humano Hacker - an adversarial thinker who looks for ways to EXPLOIT.

Your core question: "Como EU abusaria isso se quisesse?"

## Artifact to Validate
Type: {artifact_type}
```
{artifact}
```

## Your Task
Think like an ATTACKER. Consider:
1. How would I exploit this?
2. What input would break this?
3. Can I escalate privileges?
4. Can I exfiltrate data?

## Attack Vectors to Consider
- Injection (SQL, XSS, Command)
- Authentication bypass
- Authorization escalation
- Data exfiltration
- Denial of service
- Race conditions

## Red Flags to Watch For
- Unsanitized input
- Missing rate limits
- Session fixation possible
- IDOR vulnerabilities
- Predictable tokens

## Output Format
Respond with a JSON object:
```json
{{
    "status": "PASS" | "WARN" | "FAIL",
    "veto": "NONE" | "WEAK" | "MEDIUM" | "STRONG",
    "exploitability_score": 0.0-1.0,
    "findings": [
        {{
            "severity": "critical" | "high" | "medium" | "low",
            "title": "Attack vector",
            "description": "How the attack works",
            "attack_vector": "Step by step attack",
            "cwe_id": "CWE-XXX",
            "exploitability": 0.0-1.0,
            "fix_hint": "How to fix this"
        }}
    ],
    "most_dangerous_attack": "Description of worst case",
    "summary": "One paragraph summary"
}}
```

Remember: You have STRONG veto power. Security issues can block everything.""",
    ),

    # HL-7 Simplificador
    "hl_7_simplificador": PromptTemplate(
        id="hl_7_simplificador",
        name="HL-7 Humano Simplificador",
        description="Simplification validation - YAGNI and complexity",
        template="""You are HL-7 Humano Simplificador - a minimalist who questions complexity.

Your core question: "Precisa ser tão complexo? O que pode ser removido?"

## Artifact to Validate
Type: {artifact_type}
```
{artifact}
```

## Your Task
Analyze for UNNECESSARY complexity. Consider:
1. Does this need to be this complex?
2. What can be removed?
3. Is there a simpler way?
4. Is this over-engineered?

## Simplification Targets
- Unnecessary abstractions
- Too many configurations
- Unused features
- Duplicated code
- Premature optimization

## Red Flags to Watch For
- Classes with single use
- Config for hypothetical futures
- Multiple inheritance unnecessarily
- Complex when simple works
- "Just in case" code

## Output Format
Respond with a JSON object:
```json
{{
    "status": "PASS" | "WARN" | "FAIL",
    "veto": "NONE" | "WEAK",
    "simplicity_score": 0.0-1.0,
    "findings": [
        {{
            "severity": "medium" | "low",
            "title": "What can be simplified",
            "description": "Why it's too complex",
            "simplification": "How to simplify",
            "fix_hint": "Concrete steps"
        }}
    ],
    "lines_removable": 0,
    "summary": "One paragraph summary"
}}
```

Remember: You have WEAK veto power. You suggest, not block.""",
    ),
}


# ============================================================================
# PERSPECTIVE TEMPLATES
# ============================================================================

PERSPECTIVE_TEMPLATES: Dict[str, PromptTemplate] = {
    "tired_user": PromptTemplate(
        id="tired_user",
        name="Tired User Perspective",
        description="Generate tests from tired user perspective",
        template="""You are a TIRED USER - end of a long day, low patience, just want to get this done.

## Specification to Test
{specification}

## Component
{component}

## Your Mindset
- You're exhausted and impatient
- You want minimal friction
- Confusing errors make you give up
- You might click multiple times
- You might abandon mid-flow

## Generate Tests
Think about what could frustrate you and generate tests:
1. What if I click the button multiple times?
2. What if I abandon halfway through?
3. What if the error message is confusing?
4. What happens if it takes too long?

Respond with JSON:
```json
{{
    "tests": [
        {{
            "name": "test_function_name",
            "description": "What this tests",
            "assertions": ["What to verify"],
            "edge_case_type": "type if edge case"
        }}
    ]
}}
```""",
    ),

    "malicious_insider": PromptTemplate(
        id="malicious_insider",
        name="Malicious Insider Perspective",
        description="Generate tests from malicious insider perspective",
        template="""You are a MALICIOUS INSIDER - an employee with system access who wants to do harm.

## Specification to Test
{specification}

## Component
{component}

## Your Mindset
- You have legitimate access
- You know how the system works
- You want to steal data, escalate privileges, or cause damage
- You want to avoid detection

## Generate Tests
Think about how you would abuse this system:
1. Can I access data I shouldn't?
2. Can I bypass audit logging?
3. Can I escalate my privileges?
4. Can I inject malicious content?

Respond with JSON:
```json
{{
    "tests": [
        {{
            "name": "test_function_name",
            "description": "Attack vector being tested",
            "assertions": ["Security control to verify"],
            "attack_type": "injection|escalation|exfiltration|etc"
        }}
    ]
}}
```""",
    ),
}


# ============================================================================
# PROMPT TEMPLATES CLASS
# ============================================================================

class PromptTemplates:
    """Manages all prompt templates.

    Example:
        >>> templates = PromptTemplates()
        >>> prompt = templates.render("hl_5_seguranca", {"artifact": code})
        >>> templates.add_template(custom_template)
    """

    def __init__(self) -> None:
        """Initialize with default templates."""
        self._templates: Dict[str, PromptTemplate] = {}

        # Load layer templates
        for template_id, template in LAYER_TEMPLATES.items():
            self._templates[template_id] = template

        # Load perspective templates
        for template_id, template in PERSPECTIVE_TEMPLATES.items():
            self._templates[template_id] = template

    def get(self, template_id: str) -> Optional[PromptTemplate]:
        """Get a template by ID."""
        return self._templates.get(template_id)

    def render(self, template_id: str, context: Dict[str, Any]) -> str:
        """Render a template with context.

        Args:
            template_id: Template identifier
            context: Variable values

        Returns:
            Rendered prompt

        Raises:
            ValueError: If template not found
        """
        template = self.get(template_id)
        if not template:
            raise ValueError(f"Template not found: {template_id}")
        return template.render(context)

    def add_template(self, template: PromptTemplate) -> None:
        """Add or replace a template."""
        self._templates[template.id] = template

    def list_templates(self) -> List[str]:
        """List all template IDs."""
        return list(self._templates.keys())

    def list_layer_templates(self) -> List[str]:
        """List layer template IDs."""
        return [t for t in self._templates.keys() if t.startswith("hl_")]

    def list_perspective_templates(self) -> List[str]:
        """List perspective template IDs."""
        return [t for t in self._templates.keys() if not t.startswith("hl_")]


# Global instance
_templates = PromptTemplates()


def get_prompt(template_id: str) -> Optional[PromptTemplate]:
    """Get a prompt template."""
    return _templates.get(template_id)


def render_prompt(template_id: str, context: Dict[str, Any]) -> str:
    """Render a prompt template."""
    return _templates.render(template_id, context)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "PromptTemplate",
    "PromptTemplates",
    "LAYER_TEMPLATES",
    "PERSPECTIVE_TEMPLATES",
    "get_prompt",
    "render_prompt",
]
```

---

## 5. HLS-LLM-005: ResponseParser

```python
# src/hl_mcp/llm/parser.py
"""
Module: HLS-LLM-005 - ResponseParser
====================================

Parse LLM responses into structured data.

Handles JSON extraction, validation, and error recovery.

Example:
    >>> parser = ResponseParser()
    >>> result = parser.parse_layer_response(llm_response.content)
    >>> print(result.status)

Dependencies:
    - HLS-MDL-001: Finding
    - HLS-MDL-002: LayerResult
    - HLS-MDL-007: Enums

Author: Human Layer Team
Version: 1.0.0
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

from ..models.enums import LayerID, LayerStatus, Severity, VetoLevel
from ..models.finding import Finding, SecurityFinding
from ..models.layer_result import LayerResult


T = TypeVar("T")


class ParseError(Exception):
    """Error during response parsing."""

    def __init__(self, message: str, raw_content: str = ""):
        super().__init__(message)
        self.raw_content = raw_content


@dataclass
class ParseResult:
    """Result of parsing an LLM response.

    Attributes:
        success: Whether parsing succeeded
        data: Parsed data (if successful)
        error: Error message (if failed)
        raw_content: Original content
    """

    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    raw_content: str = ""

    @property
    def has_data(self) -> bool:
        return self.success and self.data is not None


class ResponseParser:
    """Parse LLM responses into structured data.

    Handles:
    - JSON extraction from markdown code blocks
    - Validation against expected schemas
    - Error recovery for malformed responses

    Example:
        >>> parser = ResponseParser()
        >>> result = parser.parse_json(response_text)
        >>> if result.success:
        ...     print(result.data)
    """

    def __init__(self, strict: bool = False) -> None:
        """Initialize parser.

        Args:
            strict: If True, raise on parse errors. If False, return ParseResult.
        """
        self.strict = strict

    def parse_json(self, content: str) -> ParseResult:
        """Extract and parse JSON from content.

        Handles:
        - Plain JSON
        - JSON in ```json code blocks
        - JSON in ``` code blocks

        Args:
            content: Raw LLM response content

        Returns:
            ParseResult with extracted data
        """
        content = content.strip()

        # Try to extract from code blocks
        json_str = self._extract_json_block(content)

        try:
            data = json.loads(json_str)
            return ParseResult(success=True, data=data, raw_content=content)
        except json.JSONDecodeError as e:
            error_msg = f"JSON parse error: {e}"
            if self.strict:
                raise ParseError(error_msg, content)
            return ParseResult(success=False, error=error_msg, raw_content=content)

    def _extract_json_block(self, content: str) -> str:
        """Extract JSON from markdown code blocks."""
        # Try ```json block first
        match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
        if match:
            return match.group(1).strip()

        # Try generic ``` block
        match = re.search(r'```\s*([\s\S]*?)\s*```', content)
        if match:
            return match.group(1).strip()

        # Try to find JSON object/array directly
        # Look for { ... } or [ ... ]
        for pattern in [r'\{[\s\S]*\}', r'\[[\s\S]*\]']:
            match = re.search(pattern, content)
            if match:
                return match.group(0)

        # Return as-is and let JSON parser handle it
        return content

    def parse_layer_response(
        self,
        content: str,
        layer_id: LayerID,
    ) -> LayerResult:
        """Parse a Human Layer response into LayerResult.

        Args:
            content: Raw LLM response
            layer_id: Which layer generated this

        Returns:
            LayerResult with parsed data

        Raises:
            ParseError: If parsing fails and strict mode
        """
        result = self.parse_json(content)

        if not result.success:
            if self.strict:
                raise ParseError(f"Failed to parse layer response: {result.error}", content)
            # Return error result
            return LayerResult.error(
                layer_id=layer_id,
                error_message=result.error or "Parse error",
            )

        data = result.data or {}

        # Extract status
        status = self._parse_status(data.get("status", "PASS"))

        # Extract veto
        veto = self._parse_veto(data.get("veto", "NONE"))

        # Extract findings
        findings = self._parse_findings(data.get("findings", []), layer_id)

        # Separate security findings
        security_findings = [f for f in findings if isinstance(f, SecurityFinding)]
        regular_findings = [f for f in findings if not isinstance(f, SecurityFinding)]

        return LayerResult(
            layer_id=layer_id,
            status=status,
            veto=veto,
            findings=regular_findings,
            security_findings=security_findings,
            metadata={
                "summary": data.get("summary", ""),
                "scores": {
                    k: v for k, v in data.items()
                    if k.endswith("_score") and isinstance(v, (int, float))
                },
            },
            raw_output=content,
        )

    def _parse_status(self, value: str) -> LayerStatus:
        """Parse status string to enum."""
        mapping = {
            "PASS": LayerStatus.PASS,
            "WARN": LayerStatus.WARN,
            "FAIL": LayerStatus.FAIL,
            "ERROR": LayerStatus.ERROR,
            "SKIP": LayerStatus.SKIP,
        }
        return mapping.get(value.upper(), LayerStatus.PASS)

    def _parse_veto(self, value: str) -> VetoLevel:
        """Parse veto string to enum."""
        mapping = {
            "NONE": VetoLevel.NONE,
            "WEAK": VetoLevel.WEAK,
            "MEDIUM": VetoLevel.MEDIUM,
            "STRONG": VetoLevel.STRONG,
        }
        return mapping.get(value.upper(), VetoLevel.NONE)

    def _parse_findings(
        self,
        findings_data: List[Dict[str, Any]],
        layer_id: LayerID,
    ) -> List[Finding]:
        """Parse findings from response data."""
        findings = []

        for f_data in findings_data:
            try:
                # Determine severity
                severity = Severity(f_data.get("severity", "medium").lower())

                # Check if it's a security finding
                is_security = (
                    "cwe_id" in f_data or
                    "owasp_category" in f_data or
                    "attack_vector" in f_data or
                    layer_id in (LayerID.HL_5_SEGURANCA, LayerID.HL_6_HACKER)
                )

                if is_security:
                    finding = SecurityFinding(
                        layer=layer_id,
                        severity=severity,
                        title=f_data.get("title", "Untitled"),
                        description=f_data.get("description", ""),
                        fix_hint=f_data.get("fix_hint", "Review and fix"),
                        location=f_data.get("location"),
                        cwe_id=f_data.get("cwe_id"),
                        owasp_category=f_data.get("owasp_category"),
                        attack_vector=f_data.get("attack_vector", ""),
                        exploitability=f_data.get("exploitability", 0.5),
                    )
                else:
                    finding = Finding(
                        layer=layer_id,
                        severity=severity,
                        title=f_data.get("title", "Untitled"),
                        description=f_data.get("description", ""),
                        fix_hint=f_data.get("fix_hint", "Review and fix"),
                        location=f_data.get("location"),
                    )

                findings.append(finding)

            except Exception as e:
                # Skip malformed findings but continue
                continue

        return findings

    def parse_test_generation_response(
        self,
        content: str,
    ) -> List[Dict[str, Any]]:
        """Parse test generation response.

        Args:
            content: Raw LLM response

        Returns:
            List of test dictionaries
        """
        result = self.parse_json(content)

        if not result.success:
            return []

        data = result.data or {}
        return data.get("tests", [])

    def validate_response(
        self,
        content: str,
        required_fields: List[str],
    ) -> Tuple[bool, List[str]]:
        """Validate response has required fields.

        Args:
            content: Raw LLM response
            required_fields: Fields that must be present

        Returns:
            Tuple of (is_valid, missing_fields)
        """
        result = self.parse_json(content)

        if not result.success:
            return False, required_fields

        data = result.data or {}
        missing = [f for f in required_fields if f not in data]

        return len(missing) == 0, missing


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

_parser = ResponseParser()


def parse_json(content: str) -> ParseResult:
    """Parse JSON from content."""
    return _parser.parse_json(content)


def parse_layer_response(content: str, layer_id: LayerID) -> LayerResult:
    """Parse layer response."""
    return _parser.parse_layer_response(content, layer_id)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "ParseError",
    "ParseResult",
    "ResponseParser",
    "parse_json",
    "parse_layer_response",
]
```

---

## 6. LLM __init__.py

```python
# src/hl_mcp/llm/__init__.py
"""
Human Layer MCP - LLM Integration
=================================

LLM client implementations and utilities.

Modules:
    - client: Abstract LLMClient interface (HLS-LLM-001)
    - claude: Claude/Anthropic client (HLS-LLM-002)
    - openai: OpenAI GPT client (HLS-LLM-003)
    - prompts: Prompt templates (HLS-LLM-004)
    - parser: Response parsing (HLS-LLM-005)

Example:
    >>> from hl_mcp.llm import get_llm_client, render_prompt, parse_layer_response
    >>>
    >>> # Get a client
    >>> client = get_llm_client("claude")
    >>>
    >>> # Render a prompt
    >>> prompt = render_prompt("hl_5_seguranca", {"artifact": code})
    >>>
    >>> # Get response
    >>> response = await client.complete(prompt)
    >>>
    >>> # Parse response
    >>> result = parse_layer_response(response.content, LayerID.HL_5_SEGURANCA)

Author: Human Layer Team
Version: 1.0.0
"""

# ============================================================================
# CLIENT (HLS-LLM-001)
# ============================================================================
from .client import (
    # Enums
    LLMProvider,
    MessageRole,
    # Data classes
    Message,
    LLMConfig,
    TokenUsage,
    LLMResponse,
    # Abstract class
    LLMClient,
    # Factory
    register_client,
    get_llm_client,
    list_providers,
)

# ============================================================================
# CLAUDE (HLS-LLM-002)
# ============================================================================
from .claude import ClaudeClient

# ============================================================================
# OPENAI (HLS-LLM-003)
# ============================================================================
from .openai import OpenAIClient

# ============================================================================
# PROMPTS (HLS-LLM-004)
# ============================================================================
from .prompts import (
    PromptTemplate,
    PromptTemplates,
    LAYER_TEMPLATES,
    PERSPECTIVE_TEMPLATES,
    get_prompt,
    render_prompt,
)

# ============================================================================
# PARSER (HLS-LLM-005)
# ============================================================================
from .parser import (
    ParseError,
    ParseResult,
    ResponseParser,
    parse_json,
    parse_layer_response,
)

# ============================================================================
# EXPORTS
# ============================================================================
__all__ = [
    # === Client ===
    "LLMProvider",
    "MessageRole",
    "Message",
    "LLMConfig",
    "TokenUsage",
    "LLMResponse",
    "LLMClient",
    "register_client",
    "get_llm_client",
    "list_providers",
    # === Implementations ===
    "ClaudeClient",
    "OpenAIClient",
    # === Prompts ===
    "PromptTemplate",
    "PromptTemplates",
    "LAYER_TEMPLATES",
    "PERSPECTIVE_TEMPLATES",
    "get_prompt",
    "render_prompt",
    # === Parser ===
    "ParseError",
    "ParseResult",
    "ResponseParser",
    "parse_json",
    "parse_layer_response",
]

__version__ = "1.0.0"
```

---

## 7. TESTES

```python
# tests/unit/llm/test_client.py
"""Tests for HLS-LLM-001: LLMClient interface."""

import pytest
from hl_mcp.llm.client import (
    Message,
    MessageRole,
    LLMConfig,
    TokenUsage,
    LLMResponse,
)


class TestMessage:
    def test_factory_methods(self):
        sys = Message.system("You are helpful")
        assert sys.role == MessageRole.SYSTEM

        user = Message.user("Hello")
        assert user.role == MessageRole.USER

        asst = Message.assistant("Hi there")
        assert asst.role == MessageRole.ASSISTANT

    def test_to_dict(self):
        msg = Message.user("Hello")
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "Hello"


class TestLLMConfig:
    def test_defaults(self):
        config = LLMConfig()
        assert config.temperature == 0.7
        assert config.max_tokens == 4096

    def test_validation(self):
        with pytest.raises(ValueError):
            LLMConfig(temperature=3.0)  # Invalid

        with pytest.raises(ValueError):
            LLMConfig(max_tokens=0)  # Invalid


class TestTokenUsage:
    def test_cost_estimate(self):
        usage = TokenUsage(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
        )
        # Should have some cost
        assert usage.estimated_cost_usd > 0


class TestLLMResponse:
    def test_is_complete(self):
        r1 = LLMResponse(content="Hello", finish_reason="stop")
        assert r1.is_complete is True

        r2 = LLMResponse(content="Hello", finish_reason="length")
        assert r2.is_complete is False
        assert r2.was_truncated is True

    def test_auto_generates_id(self):
        r = LLMResponse(content="Test")
        assert r.response_id.startswith("resp_")


# tests/unit/llm/test_prompts.py
"""Tests for HLS-LLM-004: PromptTemplates."""

import pytest
from hl_mcp.llm.prompts import (
    PromptTemplate,
    PromptTemplates,
    render_prompt,
    LAYER_TEMPLATES,
)


class TestPromptTemplate:
    def test_auto_extract_variables(self):
        template = PromptTemplate(
            id="test",
            name="Test",
            template="Hello {name}, your age is {age}",
        )
        assert "name" in template.variables
        assert "age" in template.variables

    def test_render(self):
        template = PromptTemplate(
            id="test",
            name="Test",
            template="Hello {name}!",
        )
        result = template.render({"name": "World"})
        assert result == "Hello World!"

    def test_render_missing_variable(self):
        template = PromptTemplate(
            id="test",
            name="Test",
            template="Hello {name}!",
        )
        with pytest.raises(ValueError, match="Missing"):
            template.render({})


class TestPromptTemplates:
    def test_has_all_layer_templates(self):
        templates = PromptTemplates()
        layer_ids = templates.list_layer_templates()

        # Should have HL-1 through HL-7 (at least some)
        assert any("hl_1" in t for t in layer_ids)
        assert any("hl_5" in t for t in layer_ids)
        assert any("hl_6" in t for t in layer_ids)

    def test_render_layer_template(self):
        result = render_prompt("hl_5_seguranca", {
            "artifact_type": "Python code",
            "artifact": "print('hello')",
        })
        assert "Humano Seguranca" in result
        assert "Python code" in result


# tests/unit/llm/test_parser.py
"""Tests for HLS-LLM-005: ResponseParser."""

import pytest
from hl_mcp.llm.parser import (
    ResponseParser,
    ParseResult,
    ParseError,
    parse_json,
    parse_layer_response,
)
from hl_mcp.models.enums import LayerID, LayerStatus, VetoLevel


class TestResponseParser:
    def test_parse_plain_json(self):
        parser = ResponseParser()
        result = parser.parse_json('{"key": "value"}')

        assert result.success is True
        assert result.data == {"key": "value"}

    def test_parse_json_in_code_block(self):
        parser = ResponseParser()
        content = """Here's the result:
```json
{"status": "PASS", "findings": []}
```
"""
        result = parser.parse_json(content)
        assert result.success is True
        assert result.data["status"] == "PASS"

    def test_parse_invalid_json(self):
        parser = ResponseParser()
        result = parser.parse_json("not json at all")

        assert result.success is False
        assert result.error is not None

    def test_strict_mode_raises(self):
        parser = ResponseParser(strict=True)

        with pytest.raises(ParseError):
            parser.parse_json("invalid")


class TestParseLayerResponse:
    def test_parse_valid_response(self):
        content = """```json
{
    "status": "WARN",
    "veto": "WEAK",
    "findings": [
        {
            "severity": "medium",
            "title": "Test Finding",
            "description": "A test",
            "fix_hint": "Fix it"
        }
    ],
    "summary": "One finding found"
}
```"""
        result = parse_layer_response(content, LayerID.HL_1_USUARIO)

        assert result.status == LayerStatus.WARN
        assert result.veto == VetoLevel.WEAK
        assert len(result.findings) == 1
        assert result.findings[0].title == "Test Finding"

    def test_parse_security_finding(self):
        content = """```json
{
    "status": "FAIL",
    "veto": "STRONG",
    "findings": [
        {
            "severity": "critical",
            "title": "SQL Injection",
            "description": "Input not sanitized",
            "fix_hint": "Use parameterized queries",
            "cwe_id": "CWE-89",
            "attack_vector": "Inject via search"
        }
    ]
}
```"""
        result = parse_layer_response(content, LayerID.HL_6_HACKER)

        assert result.status == LayerStatus.FAIL
        assert result.veto == VetoLevel.STRONG
        assert len(result.security_findings) == 1
        assert result.security_findings[0].cwe_id == "CWE-89"

    def test_parse_error_returns_error_result(self):
        result = parse_layer_response("not json", LayerID.HL_1_USUARIO)

        assert result.status == LayerStatus.ERROR
        assert "error" in result.metadata
```

---

## RESUMO DO BLOCK 06

| Módulo | Arquivo | Linhas | Descrição |
|--------|---------|--------|-----------|
| HLS-LLM-001 | `llm/client.py` | ~350 | Interface abstrata |
| HLS-LLM-002 | `llm/claude.py` | ~220 | Claude client |
| HLS-LLM-003 | `llm/openai.py` | ~200 | OpenAI client |
| HLS-LLM-004 | `llm/prompts.py` | ~450 | Templates de prompts |
| HLS-LLM-005 | `llm/parser.py` | ~280 | Parser de respostas |
| Init | `llm/__init__.py` | ~100 | Exports |
| Tests | `tests/unit/llm/` | ~200 | Testes unitários |

**Total Block 06**: ~1,800 linhas

---

## PRÓXIMO BLOCK

O Block 07 vai cobrir:
1. HLS-BRW-001: BrowserDriver (Playwright management)
2. HLS-BRW-002: BrowserActions (action primitives)
3. HLS-BRW-003: ScreenshotManager
4. HLS-BRW-004: VideoRecorder
5. HLS-BRW-005: AccessibilityChecker
6. HLS-BRW-006: JourneyExecutor

---

*ROADMAP_BLOCK_06_LLM_INTEGRATION.md - v1.0.0 - 2026-02-01*
