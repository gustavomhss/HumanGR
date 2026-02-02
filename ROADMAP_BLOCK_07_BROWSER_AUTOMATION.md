# ROADMAP Block 07: Browser Automation Layer

> **Bloco**: 07 de 12
> **Tema**: Browser Automation (Playwright)
> **Tokens Estimados**: ~28,000
> **Dependências**: Block 04-05 (Models), Block 06 (LLM)

---

## Visão Geral do Bloco

Este bloco implementa a camada de automação de browser usando Playwright. São 6 módulos Lego que fornecem:

1. **BrowserDriver** - Gerenciamento do ciclo de vida do browser
2. **BrowserActions** - Primitivas de ação (click, type, navigate)
3. **ScreenshotManager** - Captura e organização de screenshots
4. **VideoRecorder** - Gravação de sessões
5. **AccessibilityChecker** - Validação de acessibilidade (WCAG)
6. **JourneyExecutor** - Execução de jornadas completas

---

## Módulo HLS-BRW-001: BrowserDriver

```
ID: HLS-BRW-001
Nome: BrowserDriver
Caminho: src/hl_mcp/browser/driver.py
Dependências: playwright
Exports: BrowserDriver, BrowserConfig, BrowserType
Linhas: ~280
```

### Código

```python
"""
HLS-BRW-001: BrowserDriver
==========================

Gerenciamento do ciclo de vida do browser Playwright.

Responsabilidades:
- Inicialização/fechamento do browser
- Gerenciamento de contextos e páginas
- Configuração de viewport, headers, cookies
- Pool de browsers para execução paralela

Exemplo:
    >>> async with BrowserDriver() as driver:
    ...     page = await driver.new_page()
    ...     await page.goto("https://example.com")
    ...     await driver.close_page(page)

Changelog:
    - v1.0.0 (2026-02-01): Implementação inicial
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Optional
from pathlib import Path
import logging

try:
    from playwright.async_api import (
        async_playwright,
        Browser,
        BrowserContext,
        Page,
        Playwright,
        Error as PlaywrightError,
    )
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    Browser = None
    BrowserContext = None
    Page = None
    Playwright = None
    PlaywrightError = Exception

logger = logging.getLogger(__name__)


class BrowserType(str, Enum):
    """Tipos de browser suportados."""

    CHROMIUM = "chromium"
    FIREFOX = "firefox"
    WEBKIT = "webkit"


@dataclass
class BrowserConfig:
    """
    Configuração do browser.

    Attributes:
        browser_type: Tipo de browser (chromium, firefox, webkit)
        headless: Executar sem interface gráfica
        viewport_width: Largura do viewport
        viewport_height: Altura do viewport
        user_agent: User agent customizado
        locale: Locale do browser (pt-BR, en-US)
        timezone: Timezone (America/Sao_Paulo)
        slow_mo: Delay entre ações (ms) para debug
        downloads_path: Diretório para downloads
        record_video: Gravar vídeo das sessões
        record_har: Gravar HAR (HTTP Archive)
        ignore_https_errors: Ignorar erros de certificado
        extra_http_headers: Headers HTTP adicionais
        proxy: Configuração de proxy
        storage_state: Estado de storage (cookies, localStorage)
    """

    browser_type: BrowserType = BrowserType.CHROMIUM
    headless: bool = True
    viewport_width: int = 1280
    viewport_height: int = 720
    user_agent: Optional[str] = None
    locale: str = "en-US"
    timezone: str = "America/Sao_Paulo"
    slow_mo: int = 0
    downloads_path: Optional[Path] = None
    record_video: bool = False
    record_har: bool = False
    ignore_https_errors: bool = False
    extra_http_headers: dict[str, str] = field(default_factory=dict)
    proxy: Optional[dict[str, str]] = None
    storage_state: Optional[Path] = None

    def to_browser_args(self) -> dict:
        """Converte para argumentos do Playwright browser.launch()."""
        args = {
            "headless": self.headless,
            "slow_mo": self.slow_mo,
        }
        if self.proxy:
            args["proxy"] = self.proxy
        if self.downloads_path:
            args["downloads_path"] = str(self.downloads_path)
        return args

    def to_context_args(self) -> dict:
        """Converte para argumentos do browser.new_context()."""
        args = {
            "viewport": {
                "width": self.viewport_width,
                "height": self.viewport_height,
            },
            "locale": self.locale,
            "timezone_id": self.timezone,
            "ignore_https_errors": self.ignore_https_errors,
        }
        if self.user_agent:
            args["user_agent"] = self.user_agent
        if self.extra_http_headers:
            args["extra_http_headers"] = self.extra_http_headers
        if self.storage_state:
            args["storage_state"] = str(self.storage_state)
        if self.record_video:
            args["record_video_dir"] = str(
                self.downloads_path or Path("./videos")
            )
        if self.record_har:
            args["record_har_path"] = str(
                (self.downloads_path or Path(".")) / "trace.har"
            )
        return args


class BrowserDriver:
    """
    Gerenciador do ciclo de vida do browser Playwright.

    Suporta uso como context manager assíncrono para garantir
    cleanup adequado dos recursos.

    Example:
        >>> config = BrowserConfig(headless=True)
        >>> async with BrowserDriver(config) as driver:
        ...     page = await driver.new_page()
        ...     await page.goto("https://example.com")
    """

    def __init__(self, config: Optional[BrowserConfig] = None):
        """
        Inicializa o driver.

        Args:
            config: Configuração do browser. Se None, usa defaults.
        """
        if not PLAYWRIGHT_AVAILABLE:
            raise RuntimeError(
                "Playwright não instalado. "
                "Execute: pip install playwright && playwright install"
            )

        self.config = config or BrowserConfig()
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._contexts: list[BrowserContext] = []
        self._pages: list[Page] = []
        self._is_started = False

    async def start(self) -> None:
        """
        Inicia o Playwright e o browser.

        Raises:
            RuntimeError: Se já estiver iniciado
            PlaywrightError: Se falhar ao iniciar browser
        """
        if self._is_started:
            raise RuntimeError("BrowserDriver já iniciado")

        logger.info(
            f"Iniciando browser {self.config.browser_type.value} "
            f"(headless={self.config.headless})"
        )

        self._playwright = await async_playwright().start()

        # Seleciona o tipo de browser
        browser_launcher = getattr(
            self._playwright,
            self.config.browser_type.value
        )

        self._browser = await browser_launcher.launch(
            **self.config.to_browser_args()
        )

        self._is_started = True
        logger.info("Browser iniciado com sucesso")

    async def stop(self) -> None:
        """
        Para o browser e libera recursos.

        Fecha todas as páginas e contextos antes de fechar o browser.
        """
        if not self._is_started:
            return

        logger.info("Parando browser...")

        # Fecha páginas
        for page in self._pages[:]:
            try:
                await page.close()
            except Exception as e:
                logger.warning(f"Erro ao fechar página: {e}")
        self._pages.clear()

        # Fecha contextos
        for context in self._contexts[:]:
            try:
                await context.close()
            except Exception as e:
                logger.warning(f"Erro ao fechar contexto: {e}")
        self._contexts.clear()

        # Fecha browser
        if self._browser:
            await self._browser.close()
            self._browser = None

        # Para Playwright
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

        self._is_started = False
        logger.info("Browser parado")

    async def new_context(self) -> BrowserContext:
        """
        Cria novo contexto de browser (sessão isolada).

        Um contexto é como uma janela anônima - cookies, localStorage
        e cache são isolados de outros contextos.

        Returns:
            Novo BrowserContext

        Raises:
            RuntimeError: Se browser não estiver iniciado
        """
        if not self._is_started or not self._browser:
            raise RuntimeError("Browser não iniciado. Chame start() primeiro.")

        context = await self._browser.new_context(
            **self.config.to_context_args()
        )
        self._contexts.append(context)

        logger.debug(f"Novo contexto criado. Total: {len(self._contexts)}")
        return context

    async def new_page(
        self,
        context: Optional[BrowserContext] = None
    ) -> Page:
        """
        Cria nova página (aba).

        Args:
            context: Contexto para criar a página. Se None, cria novo contexto.

        Returns:
            Nova Page
        """
        if context is None:
            context = await self.new_context()

        page = await context.new_page()
        self._pages.append(page)

        logger.debug(f"Nova página criada. Total: {len(self._pages)}")
        return page

    async def close_page(self, page: Page) -> None:
        """
        Fecha uma página específica.

        Args:
            page: Página a fechar
        """
        if page in self._pages:
            self._pages.remove(page)
        await page.close()
        logger.debug(f"Página fechada. Restantes: {len(self._pages)}")

    async def close_context(self, context: BrowserContext) -> None:
        """
        Fecha um contexto específico.

        Também fecha todas as páginas do contexto.

        Args:
            context: Contexto a fechar
        """
        # Remove páginas do contexto
        for page in context.pages:
            if page in self._pages:
                self._pages.remove(page)

        if context in self._contexts:
            self._contexts.remove(context)

        await context.close()
        logger.debug(f"Contexto fechado. Restantes: {len(self._contexts)}")

    @property
    def is_started(self) -> bool:
        """Retorna True se o browser está iniciado."""
        return self._is_started

    @property
    def page_count(self) -> int:
        """Número de páginas abertas."""
        return len(self._pages)

    @property
    def context_count(self) -> int:
        """Número de contextos abertos."""
        return len(self._contexts)

    async def __aenter__(self) -> "BrowserDriver":
        """Suporte a async context manager."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Cleanup ao sair do context manager."""
        await self.stop()


@asynccontextmanager
async def create_browser(
    config: Optional[BrowserConfig] = None
) -> AsyncIterator[BrowserDriver]:
    """
    Factory function para criar BrowserDriver como context manager.

    Args:
        config: Configuração do browser

    Yields:
        BrowserDriver iniciado

    Example:
        >>> async with create_browser() as driver:
        ...     page = await driver.new_page()
        ...     await page.goto("https://example.com")
    """
    driver = BrowserDriver(config)
    try:
        await driver.start()
        yield driver
    finally:
        await driver.stop()


# Exports
__all__ = [
    "BrowserType",
    "BrowserConfig",
    "BrowserDriver",
    "create_browser",
    "PLAYWRIGHT_AVAILABLE",
]
```

---

## Módulo HLS-BRW-002: BrowserActions

```
ID: HLS-BRW-002
Nome: BrowserActions
Caminho: src/hl_mcp/browser/actions.py
Dependências: HLS-BRW-001
Exports: BrowserActions, ActionResult, WaitStrategy
Linhas: ~350
```

### Código

```python
"""
HLS-BRW-002: BrowserActions
===========================

Primitivas de ação para automação de browser.

Responsabilidades:
- Navegação (goto, back, forward, reload)
- Interação (click, type, select, hover)
- Espera (wait for selector, wait for navigation)
- Extração (get text, get attribute, get all)

Cada ação retorna ActionResult com sucesso/falha e detalhes.

Exemplo:
    >>> actions = BrowserActions(page)
    >>> result = await actions.click("#submit-button")
    >>> if result.success:
    ...     print(f"Clicou em {result.selector}")

Changelog:
    - v1.0.0 (2026-02-01): Implementação inicial
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Union
import logging

try:
    from playwright.async_api import Page, Locator, TimeoutError as PlaywrightTimeout
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    Page = Any
    Locator = Any
    PlaywrightTimeout = TimeoutError

logger = logging.getLogger(__name__)


class WaitStrategy(str, Enum):
    """Estratégia de espera após ação."""

    NONE = "none"  # Não espera
    LOAD = "load"  # Espera load event
    DOMCONTENTLOADED = "domcontentloaded"  # Espera DOMContentLoaded
    NETWORKIDLE = "networkidle"  # Espera network idle
    COMMIT = "commit"  # Espera resposta inicial


class ActionType(str, Enum):
    """Tipos de ação suportados."""

    NAVIGATE = "navigate"
    CLICK = "click"
    TYPE = "type"
    SELECT = "select"
    HOVER = "hover"
    FOCUS = "focus"
    CLEAR = "clear"
    CHECK = "check"
    UNCHECK = "uncheck"
    UPLOAD = "upload"
    SCROLL = "scroll"
    WAIT = "wait"
    EXTRACT = "extract"


@dataclass
class ActionResult:
    """
    Resultado de uma ação de browser.

    Attributes:
        success: Se a ação foi bem sucedida
        action_type: Tipo da ação executada
        selector: Seletor usado (se aplicável)
        value: Valor usado/extraído
        duration_ms: Duração da ação em ms
        error: Mensagem de erro (se falhou)
        screenshot_path: Path do screenshot (se capturado)
        timestamp: Momento da execução
    """

    success: bool
    action_type: ActionType
    selector: Optional[str] = None
    value: Optional[Any] = None
    duration_ms: float = 0
    error: Optional[str] = None
    screenshot_path: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        """Converte para dicionário."""
        return {
            "success": self.success,
            "action_type": self.action_type.value,
            "selector": self.selector,
            "value": self.value,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "screenshot_path": self.screenshot_path,
            "timestamp": self.timestamp.isoformat(),
        }


class BrowserActions:
    """
    Primitivas de ação para automação de browser.

    Encapsula ações comuns do Playwright com:
    - Tratamento de erros consistente
    - Logging automático
    - Métricas de duração
    - Screenshots em falha (opcional)

    Example:
        >>> actions = BrowserActions(page, screenshot_on_failure=True)
        >>> await actions.navigate("https://example.com")
        >>> await actions.click("#login-button")
        >>> await actions.type("#username", "test@test.com")
    """

    def __init__(
        self,
        page: Page,
        default_timeout: int = 30000,
        screenshot_on_failure: bool = False,
        screenshot_dir: Optional[str] = None,
    ):
        """
        Inicializa BrowserActions.

        Args:
            page: Página Playwright
            default_timeout: Timeout padrão em ms
            screenshot_on_failure: Capturar screenshot em falhas
            screenshot_dir: Diretório para screenshots
        """
        self.page = page
        self.default_timeout = default_timeout
        self.screenshot_on_failure = screenshot_on_failure
        self.screenshot_dir = screenshot_dir or "./screenshots"

    async def _capture_failure_screenshot(
        self,
        action_type: ActionType
    ) -> Optional[str]:
        """Captura screenshot em caso de falha."""
        if not self.screenshot_on_failure:
            return None

        try:
            from pathlib import Path
            Path(self.screenshot_dir).mkdir(parents=True, exist_ok=True)

            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            path = f"{self.screenshot_dir}/failure_{action_type.value}_{timestamp}.png"
            await self.page.screenshot(path=path)
            return path
        except Exception as e:
            logger.warning(f"Falha ao capturar screenshot: {e}")
            return None

    async def navigate(
        self,
        url: str,
        wait_until: WaitStrategy = WaitStrategy.LOAD,
        timeout: Optional[int] = None,
    ) -> ActionResult:
        """
        Navega para uma URL.

        Args:
            url: URL de destino
            wait_until: Estratégia de espera
            timeout: Timeout em ms

        Returns:
            ActionResult com sucesso/falha
        """
        start = datetime.utcnow()

        try:
            await self.page.goto(
                url,
                wait_until=wait_until.value if wait_until != WaitStrategy.NONE else None,
                timeout=timeout or self.default_timeout,
            )

            duration = (datetime.utcnow() - start).total_seconds() * 1000
            logger.debug(f"Navegou para {url} em {duration:.0f}ms")

            return ActionResult(
                success=True,
                action_type=ActionType.NAVIGATE,
                value=url,
                duration_ms=duration,
            )
        except Exception as e:
            duration = (datetime.utcnow() - start).total_seconds() * 1000
            screenshot = await self._capture_failure_screenshot(ActionType.NAVIGATE)

            logger.error(f"Falha ao navegar para {url}: {e}")

            return ActionResult(
                success=False,
                action_type=ActionType.NAVIGATE,
                value=url,
                duration_ms=duration,
                error=str(e),
                screenshot_path=screenshot,
            )

    async def click(
        self,
        selector: str,
        timeout: Optional[int] = None,
        force: bool = False,
        button: str = "left",
        click_count: int = 1,
    ) -> ActionResult:
        """
        Clica em um elemento.

        Args:
            selector: Seletor CSS/XPath do elemento
            timeout: Timeout em ms
            force: Forçar clique mesmo se elemento não visível
            button: Botão do mouse (left, right, middle)
            click_count: Número de cliques (2 para double-click)

        Returns:
            ActionResult com sucesso/falha
        """
        start = datetime.utcnow()

        try:
            await self.page.click(
                selector,
                timeout=timeout or self.default_timeout,
                force=force,
                button=button,
                click_count=click_count,
            )

            duration = (datetime.utcnow() - start).total_seconds() * 1000
            logger.debug(f"Clicou em {selector} em {duration:.0f}ms")

            return ActionResult(
                success=True,
                action_type=ActionType.CLICK,
                selector=selector,
                duration_ms=duration,
            )
        except Exception as e:
            duration = (datetime.utcnow() - start).total_seconds() * 1000
            screenshot = await self._capture_failure_screenshot(ActionType.CLICK)

            logger.error(f"Falha ao clicar em {selector}: {e}")

            return ActionResult(
                success=False,
                action_type=ActionType.CLICK,
                selector=selector,
                duration_ms=duration,
                error=str(e),
                screenshot_path=screenshot,
            )

    async def type_text(
        self,
        selector: str,
        text: str,
        delay: int = 0,
        clear_first: bool = True,
        timeout: Optional[int] = None,
    ) -> ActionResult:
        """
        Digita texto em um elemento.

        Args:
            selector: Seletor do campo
            text: Texto a digitar
            delay: Delay entre caracteres em ms
            clear_first: Limpar campo antes de digitar
            timeout: Timeout em ms

        Returns:
            ActionResult com sucesso/falha
        """
        start = datetime.utcnow()

        try:
            if clear_first:
                await self.page.fill(
                    selector,
                    "",
                    timeout=timeout or self.default_timeout,
                )

            await self.page.type(
                selector,
                text,
                delay=delay,
                timeout=timeout or self.default_timeout,
            )

            duration = (datetime.utcnow() - start).total_seconds() * 1000
            logger.debug(f"Digitou em {selector} em {duration:.0f}ms")

            return ActionResult(
                success=True,
                action_type=ActionType.TYPE,
                selector=selector,
                value=text,
                duration_ms=duration,
            )
        except Exception as e:
            duration = (datetime.utcnow() - start).total_seconds() * 1000
            screenshot = await self._capture_failure_screenshot(ActionType.TYPE)

            logger.error(f"Falha ao digitar em {selector}: {e}")

            return ActionResult(
                success=False,
                action_type=ActionType.TYPE,
                selector=selector,
                value=text,
                duration_ms=duration,
                error=str(e),
                screenshot_path=screenshot,
            )

    async def fill(
        self,
        selector: str,
        value: str,
        timeout: Optional[int] = None,
    ) -> ActionResult:
        """
        Preenche um campo (mais rápido que type).

        Diferente de type(), fill() substitui todo o conteúdo
        de uma vez, sem simular digitação caractere por caractere.

        Args:
            selector: Seletor do campo
            value: Valor a preencher
            timeout: Timeout em ms
        """
        start = datetime.utcnow()

        try:
            await self.page.fill(
                selector,
                value,
                timeout=timeout or self.default_timeout,
            )

            duration = (datetime.utcnow() - start).total_seconds() * 1000

            return ActionResult(
                success=True,
                action_type=ActionType.TYPE,
                selector=selector,
                value=value,
                duration_ms=duration,
            )
        except Exception as e:
            duration = (datetime.utcnow() - start).total_seconds() * 1000
            screenshot = await self._capture_failure_screenshot(ActionType.TYPE)

            return ActionResult(
                success=False,
                action_type=ActionType.TYPE,
                selector=selector,
                value=value,
                duration_ms=duration,
                error=str(e),
                screenshot_path=screenshot,
            )

    async def select_option(
        self,
        selector: str,
        value: Optional[str] = None,
        label: Optional[str] = None,
        index: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> ActionResult:
        """
        Seleciona opção em um <select>.

        Args:
            selector: Seletor do select
            value: Valor da opção
            label: Texto visível da opção
            index: Índice da opção (0-based)
            timeout: Timeout em ms
        """
        start = datetime.utcnow()

        try:
            options = []
            if value is not None:
                options = [value]
            elif label is not None:
                options = [{"label": label}]
            elif index is not None:
                options = [{"index": index}]

            selected = await self.page.select_option(
                selector,
                options,
                timeout=timeout or self.default_timeout,
            )

            duration = (datetime.utcnow() - start).total_seconds() * 1000

            return ActionResult(
                success=True,
                action_type=ActionType.SELECT,
                selector=selector,
                value=selected,
                duration_ms=duration,
            )
        except Exception as e:
            duration = (datetime.utcnow() - start).total_seconds() * 1000
            screenshot = await self._capture_failure_screenshot(ActionType.SELECT)

            return ActionResult(
                success=False,
                action_type=ActionType.SELECT,
                selector=selector,
                duration_ms=duration,
                error=str(e),
                screenshot_path=screenshot,
            )

    async def hover(
        self,
        selector: str,
        timeout: Optional[int] = None,
    ) -> ActionResult:
        """Move o mouse sobre um elemento."""
        start = datetime.utcnow()

        try:
            await self.page.hover(
                selector,
                timeout=timeout or self.default_timeout,
            )

            duration = (datetime.utcnow() - start).total_seconds() * 1000

            return ActionResult(
                success=True,
                action_type=ActionType.HOVER,
                selector=selector,
                duration_ms=duration,
            )
        except Exception as e:
            duration = (datetime.utcnow() - start).total_seconds() * 1000

            return ActionResult(
                success=False,
                action_type=ActionType.HOVER,
                selector=selector,
                duration_ms=duration,
                error=str(e),
            )

    async def check(
        self,
        selector: str,
        timeout: Optional[int] = None,
    ) -> ActionResult:
        """Marca um checkbox."""
        start = datetime.utcnow()

        try:
            await self.page.check(
                selector,
                timeout=timeout or self.default_timeout,
            )

            duration = (datetime.utcnow() - start).total_seconds() * 1000

            return ActionResult(
                success=True,
                action_type=ActionType.CHECK,
                selector=selector,
                duration_ms=duration,
            )
        except Exception as e:
            duration = (datetime.utcnow() - start).total_seconds() * 1000

            return ActionResult(
                success=False,
                action_type=ActionType.CHECK,
                selector=selector,
                duration_ms=duration,
                error=str(e),
            )

    async def uncheck(
        self,
        selector: str,
        timeout: Optional[int] = None,
    ) -> ActionResult:
        """Desmarca um checkbox."""
        start = datetime.utcnow()

        try:
            await self.page.uncheck(
                selector,
                timeout=timeout or self.default_timeout,
            )

            duration = (datetime.utcnow() - start).total_seconds() * 1000

            return ActionResult(
                success=True,
                action_type=ActionType.UNCHECK,
                selector=selector,
                duration_ms=duration,
            )
        except Exception as e:
            duration = (datetime.utcnow() - start).total_seconds() * 1000

            return ActionResult(
                success=False,
                action_type=ActionType.UNCHECK,
                selector=selector,
                duration_ms=duration,
                error=str(e),
            )

    async def wait_for_selector(
        self,
        selector: str,
        state: str = "visible",
        timeout: Optional[int] = None,
    ) -> ActionResult:
        """
        Espera por um elemento.

        Args:
            selector: Seletor do elemento
            state: Estado esperado (visible, hidden, attached, detached)
            timeout: Timeout em ms
        """
        start = datetime.utcnow()

        try:
            await self.page.wait_for_selector(
                selector,
                state=state,
                timeout=timeout or self.default_timeout,
            )

            duration = (datetime.utcnow() - start).total_seconds() * 1000

            return ActionResult(
                success=True,
                action_type=ActionType.WAIT,
                selector=selector,
                value=state,
                duration_ms=duration,
            )
        except Exception as e:
            duration = (datetime.utcnow() - start).total_seconds() * 1000

            return ActionResult(
                success=False,
                action_type=ActionType.WAIT,
                selector=selector,
                value=state,
                duration_ms=duration,
                error=str(e),
            )

    async def get_text(
        self,
        selector: str,
        timeout: Optional[int] = None,
    ) -> ActionResult:
        """Extrai texto de um elemento."""
        start = datetime.utcnow()

        try:
            text = await self.page.text_content(
                selector,
                timeout=timeout or self.default_timeout,
            )

            duration = (datetime.utcnow() - start).total_seconds() * 1000

            return ActionResult(
                success=True,
                action_type=ActionType.EXTRACT,
                selector=selector,
                value=text,
                duration_ms=duration,
            )
        except Exception as e:
            duration = (datetime.utcnow() - start).total_seconds() * 1000

            return ActionResult(
                success=False,
                action_type=ActionType.EXTRACT,
                selector=selector,
                duration_ms=duration,
                error=str(e),
            )

    async def get_attribute(
        self,
        selector: str,
        attribute: str,
        timeout: Optional[int] = None,
    ) -> ActionResult:
        """Extrai atributo de um elemento."""
        start = datetime.utcnow()

        try:
            value = await self.page.get_attribute(
                selector,
                attribute,
                timeout=timeout or self.default_timeout,
            )

            duration = (datetime.utcnow() - start).total_seconds() * 1000

            return ActionResult(
                success=True,
                action_type=ActionType.EXTRACT,
                selector=selector,
                value={attribute: value},
                duration_ms=duration,
            )
        except Exception as e:
            duration = (datetime.utcnow() - start).total_seconds() * 1000

            return ActionResult(
                success=False,
                action_type=ActionType.EXTRACT,
                selector=selector,
                duration_ms=duration,
                error=str(e),
            )

    async def scroll_to(
        self,
        selector: Optional[str] = None,
        x: int = 0,
        y: int = 0,
    ) -> ActionResult:
        """
        Scroll na página.

        Args:
            selector: Se fornecido, scroll até o elemento
            x: Posição horizontal (se selector=None)
            y: Posição vertical (se selector=None)
        """
        start = datetime.utcnow()

        try:
            if selector:
                await self.page.locator(selector).scroll_into_view_if_needed()
            else:
                await self.page.evaluate(f"window.scrollTo({x}, {y})")

            duration = (datetime.utcnow() - start).total_seconds() * 1000

            return ActionResult(
                success=True,
                action_type=ActionType.SCROLL,
                selector=selector,
                value={"x": x, "y": y},
                duration_ms=duration,
            )
        except Exception as e:
            duration = (datetime.utcnow() - start).total_seconds() * 1000

            return ActionResult(
                success=False,
                action_type=ActionType.SCROLL,
                selector=selector,
                duration_ms=duration,
                error=str(e),
            )


# Exports
__all__ = [
    "WaitStrategy",
    "ActionType",
    "ActionResult",
    "BrowserActions",
]
```

---

## Módulo HLS-BRW-003: ScreenshotManager

```
ID: HLS-BRW-003
Nome: ScreenshotManager
Caminho: src/hl_mcp/browser/screenshots.py
Dependências: HLS-BRW-001
Exports: ScreenshotManager, ScreenshotConfig, Screenshot
Linhas: ~200
```

### Código

```python
"""
HLS-BRW-003: ScreenshotManager
==============================

Captura e gerenciamento de screenshots.

Responsabilidades:
- Captura de screenshots (full page, viewport, elemento)
- Organização por sessão/timestamp
- Comparação visual (diff)
- Cleanup automático

Exemplo:
    >>> manager = ScreenshotManager(page, output_dir="./screenshots")
    >>> shot = await manager.capture_full_page("login_page")
    >>> print(shot.path)
    ./screenshots/2026-02-01/session_123/login_page.png

Changelog:
    - v1.0.0 (2026-02-01): Implementação inicial
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import logging
import hashlib

try:
    from playwright.async_api import Page
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    Page = Any

logger = logging.getLogger(__name__)


@dataclass
class ScreenshotConfig:
    """
    Configuração para captura de screenshots.

    Attributes:
        output_dir: Diretório base para screenshots
        format: Formato da imagem (png, jpeg)
        quality: Qualidade JPEG (0-100, ignorado para PNG)
        full_page: Capturar página inteira vs viewport
        organize_by_date: Criar subdiretórios por data
        organize_by_session: Criar subdiretórios por sessão
        max_age_days: Limpar screenshots mais antigos que X dias
    """

    output_dir: Path = field(default_factory=lambda: Path("./screenshots"))
    format: str = "png"
    quality: int = 80
    full_page: bool = False
    organize_by_date: bool = True
    organize_by_session: bool = True
    max_age_days: int = 7

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)


@dataclass
class Screenshot:
    """
    Representa um screenshot capturado.

    Attributes:
        path: Caminho completo do arquivo
        name: Nome do screenshot
        width: Largura em pixels
        height: Altura em pixels
        size_bytes: Tamanho do arquivo
        timestamp: Momento da captura
        hash: Hash MD5 da imagem (para comparação)
        metadata: Metadados adicionais
    """

    path: Path
    name: str
    width: int = 0
    height: int = 0
    size_bytes: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    hash: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Converte para dicionário."""
        return {
            "path": str(self.path),
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "size_bytes": self.size_bytes,
            "timestamp": self.timestamp.isoformat(),
            "hash": self.hash,
            "metadata": self.metadata,
        }


class ScreenshotManager:
    """
    Gerenciador de screenshots para sessões de browser.

    Organiza screenshots por data e sessão, suporta captura
    de página inteira, viewport, ou elementos específicos.

    Example:
        >>> manager = ScreenshotManager(page, session_id="test_123")
        >>> await manager.ensure_output_dir()
        >>> shot = await manager.capture("login_page")
        >>> print(f"Saved: {shot.path}")
    """

    def __init__(
        self,
        page: Page,
        config: Optional[ScreenshotConfig] = None,
        session_id: Optional[str] = None,
    ):
        """
        Inicializa o manager.

        Args:
            page: Página Playwright
            config: Configuração de screenshots
            session_id: ID da sessão para organização
        """
        self.page = page
        self.config = config or ScreenshotConfig()
        self.session_id = session_id or datetime.utcnow().strftime("%H%M%S")
        self._screenshots: list[Screenshot] = []

    def _get_output_path(self, name: str) -> Path:
        """
        Gera path de saída baseado em configuração.

        Args:
            name: Nome do screenshot (sem extensão)

        Returns:
            Path completo para o arquivo
        """
        path = self.config.output_dir

        if self.config.organize_by_date:
            date_str = datetime.utcnow().strftime("%Y-%m-%d")
            path = path / date_str

        if self.config.organize_by_session:
            path = path / f"session_{self.session_id}"

        extension = self.config.format.lower()
        filename = f"{name}.{extension}"

        return path / filename

    async def ensure_output_dir(self) -> None:
        """Garante que o diretório de saída existe."""
        path = self._get_output_path("dummy").parent
        path.mkdir(parents=True, exist_ok=True)

    async def capture(
        self,
        name: str,
        full_page: Optional[bool] = None,
        selector: Optional[str] = None,
        clip: Optional[dict] = None,
    ) -> Screenshot:
        """
        Captura screenshot.

        Args:
            name: Nome do screenshot (sem extensão)
            full_page: Página inteira (None = usar config)
            selector: Capturar apenas este elemento
            clip: Região específica {"x", "y", "width", "height"}

        Returns:
            Screenshot com path e metadados
        """
        await self.ensure_output_dir()

        path = self._get_output_path(name)
        full = full_page if full_page is not None else self.config.full_page

        screenshot_args = {
            "path": str(path),
            "full_page": full and selector is None,
        }

        if self.config.format.lower() == "jpeg":
            screenshot_args["type"] = "jpeg"
            screenshot_args["quality"] = self.config.quality

        if clip:
            screenshot_args["clip"] = clip

        try:
            if selector:
                # Captura de elemento específico
                element = self.page.locator(selector)
                await element.screenshot(path=str(path))
            else:
                # Captura de página/viewport
                await self.page.screenshot(**screenshot_args)

            # Calcula hash e tamanho
            content = path.read_bytes()
            file_hash = hashlib.md5(content).hexdigest()

            # Obtém dimensões do viewport
            viewport = self.page.viewport_size or {"width": 0, "height": 0}

            screenshot = Screenshot(
                path=path,
                name=name,
                width=viewport.get("width", 0),
                height=viewport.get("height", 0),
                size_bytes=len(content),
                hash=file_hash,
                metadata={
                    "full_page": full,
                    "selector": selector,
                    "url": self.page.url,
                },
            )

            self._screenshots.append(screenshot)
            logger.info(f"Screenshot capturado: {path}")

            return screenshot

        except Exception as e:
            logger.error(f"Falha ao capturar screenshot {name}: {e}")
            raise

    async def capture_full_page(self, name: str) -> Screenshot:
        """Atalho para captura de página inteira."""
        return await self.capture(name, full_page=True)

    async def capture_element(self, name: str, selector: str) -> Screenshot:
        """Atalho para captura de elemento."""
        return await self.capture(name, selector=selector)

    async def capture_viewport(self, name: str) -> Screenshot:
        """Atalho para captura do viewport atual."""
        return await self.capture(name, full_page=False)

    def get_all_screenshots(self) -> list[Screenshot]:
        """Retorna todos os screenshots da sessão."""
        return self._screenshots.copy()

    @staticmethod
    def compare_screenshots(
        screenshot1: Screenshot,
        screenshot2: Screenshot,
    ) -> dict:
        """
        Compara dois screenshots.

        Returns:
            dict com:
            - identical: bool se são idênticos (hash)
            - size_diff: diferença de tamanho em bytes
            - dimension_match: se dimensões são iguais
        """
        return {
            "identical": screenshot1.hash == screenshot2.hash,
            "size_diff": abs(screenshot1.size_bytes - screenshot2.size_bytes),
            "dimension_match": (
                screenshot1.width == screenshot2.width and
                screenshot1.height == screenshot2.height
            ),
        }

    async def cleanup_old(self) -> int:
        """
        Remove screenshots mais antigos que max_age_days.

        Returns:
            Número de arquivos removidos
        """
        from datetime import timedelta
        import os

        cutoff = datetime.utcnow() - timedelta(days=self.config.max_age_days)
        removed = 0

        for path in self.config.output_dir.rglob(f"*.{self.config.format}"):
            try:
                mtime = datetime.fromtimestamp(path.stat().st_mtime)
                if mtime < cutoff:
                    path.unlink()
                    removed += 1
            except Exception as e:
                logger.warning(f"Erro ao remover {path}: {e}")

        logger.info(f"Cleanup: {removed} screenshots removidos")
        return removed


# Exports
__all__ = [
    "ScreenshotConfig",
    "Screenshot",
    "ScreenshotManager",
]
```

---

## Módulo HLS-BRW-004: VideoRecorder

```
ID: HLS-BRW-004
Nome: VideoRecorder
Caminho: src/hl_mcp/browser/video.py
Dependências: HLS-BRW-001
Exports: VideoRecorder, VideoConfig, Recording
Linhas: ~180
```

### Código

```python
"""
HLS-BRW-004: VideoRecorder
==========================

Gravação de vídeo das sessões de browser.

Responsabilidades:
- Configurar gravação de contexto Playwright
- Gerenciar arquivos de vídeo
- Fornecer metadados da gravação

Nota: Playwright grava vídeo por contexto, não por página.
A gravação é configurada ao criar o contexto.

Exemplo:
    >>> config = VideoConfig(output_dir="./videos", size={"width": 1280, "height": 720})
    >>> recorder = VideoRecorder(config)
    >>> context = await recorder.create_recording_context(browser)
    >>> page = await context.new_page()
    >>> # ... interações ...
    >>> recording = await recorder.stop_recording(context)
    >>> print(recording.path)

Changelog:
    - v1.0.0 (2026-02-01): Implementação inicial
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import logging

try:
    from playwright.async_api import Browser, BrowserContext
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    Browser = Any
    BrowserContext = Any

logger = logging.getLogger(__name__)


@dataclass
class VideoConfig:
    """
    Configuração de gravação de vídeo.

    Attributes:
        output_dir: Diretório para vídeos
        size: Dimensões do vídeo {"width": int, "height": int}
        organize_by_date: Criar subdiretórios por data
        max_age_days: Limpar vídeos mais antigos
    """

    output_dir: Path = field(default_factory=lambda: Path("./videos"))
    size: dict = field(default_factory=lambda: {"width": 1280, "height": 720})
    organize_by_date: bool = True
    max_age_days: int = 3

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)


@dataclass
class Recording:
    """
    Representa uma gravação de vídeo.

    Attributes:
        path: Caminho do arquivo de vídeo
        session_id: ID da sessão
        start_time: Início da gravação
        end_time: Fim da gravação
        duration_seconds: Duração em segundos
        size_bytes: Tamanho do arquivo
        width: Largura do vídeo
        height: Altura do vídeo
    """

    path: Path
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0
    size_bytes: int = 0
    width: int = 0
    height: int = 0

    def to_dict(self) -> dict:
        """Converte para dicionário."""
        return {
            "path": str(self.path),
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "size_bytes": self.size_bytes,
            "width": self.width,
            "height": self.height,
        }


class VideoRecorder:
    """
    Gerenciador de gravação de vídeo para sessões de browser.

    O Playwright grava vídeo por contexto de browser. Este manager
    facilita a criação de contextos com gravação e coleta de metadados.

    Example:
        >>> recorder = VideoRecorder()
        >>> context = await recorder.create_recording_context(browser)
        >>> page = await context.new_page()
        >>> await page.goto("https://example.com")
        >>> # ... interações ...
        >>> recording = await recorder.stop_recording(context)
    """

    def __init__(self, config: Optional[VideoConfig] = None):
        """
        Inicializa o recorder.

        Args:
            config: Configuração de vídeo
        """
        self.config = config or VideoConfig()
        self._active_recordings: dict[BrowserContext, dict] = {}

    def _get_output_dir(self, session_id: str) -> Path:
        """Gera diretório de saída para a sessão."""
        path = self.config.output_dir

        if self.config.organize_by_date:
            date_str = datetime.utcnow().strftime("%Y-%m-%d")
            path = path / date_str

        path = path / f"session_{session_id}"
        return path

    async def create_recording_context(
        self,
        browser: Browser,
        session_id: Optional[str] = None,
        **context_kwargs,
    ) -> BrowserContext:
        """
        Cria contexto de browser com gravação habilitada.

        Args:
            browser: Browser Playwright
            session_id: ID da sessão (gerado se não fornecido)
            **context_kwargs: Args adicionais para new_context()

        Returns:
            BrowserContext com gravação habilitada
        """
        session_id = session_id or datetime.utcnow().strftime("%H%M%S%f")
        output_dir = self._get_output_dir(session_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Merge configurações
        context_args = {
            "record_video_dir": str(output_dir),
            "record_video_size": self.config.size,
            **context_kwargs,
        }

        context = await browser.new_context(**context_args)

        # Registra início da gravação
        self._active_recordings[context] = {
            "session_id": session_id,
            "start_time": datetime.utcnow(),
            "output_dir": output_dir,
        }

        logger.info(f"Gravação iniciada: session={session_id}")
        return context

    async def stop_recording(self, context: BrowserContext) -> Optional[Recording]:
        """
        Para gravação e retorna metadados.

        Args:
            context: Contexto com gravação ativa

        Returns:
            Recording com path e metadados, ou None se não estava gravando
        """
        if context not in self._active_recordings:
            logger.warning("Contexto não tem gravação ativa")
            return None

        record_info = self._active_recordings.pop(context)

        # Obtém path do vídeo de cada página
        video_paths = []
        for page in context.pages:
            video = page.video
            if video:
                path = await video.path()
                video_paths.append(Path(path))

        # Fecha contexto para finalizar gravação
        await context.close()

        end_time = datetime.utcnow()
        duration = (end_time - record_info["start_time"]).total_seconds()

        if not video_paths:
            logger.warning("Nenhum vídeo gerado")
            return None

        # Usa o primeiro vídeo (geralmente só tem um)
        video_path = video_paths[0]

        # Aguarda arquivo ser escrito completamente
        await asyncio.sleep(0.5)

        try:
            size_bytes = video_path.stat().st_size if video_path.exists() else 0
        except Exception:
            size_bytes = 0

        recording = Recording(
            path=video_path,
            session_id=record_info["session_id"],
            start_time=record_info["start_time"],
            end_time=end_time,
            duration_seconds=duration,
            size_bytes=size_bytes,
            width=self.config.size.get("width", 0),
            height=self.config.size.get("height", 0),
        )

        logger.info(
            f"Gravação finalizada: {video_path} "
            f"({duration:.1f}s, {size_bytes/1024:.1f}KB)"
        )

        return recording

    def is_recording(self, context: BrowserContext) -> bool:
        """Verifica se contexto está gravando."""
        return context in self._active_recordings

    async def cleanup_old(self) -> int:
        """
        Remove vídeos mais antigos que max_age_days.

        Returns:
            Número de arquivos removidos
        """
        from datetime import timedelta

        cutoff = datetime.utcnow() - timedelta(days=self.config.max_age_days)
        removed = 0

        for path in self.config.output_dir.rglob("*.webm"):
            try:
                mtime = datetime.fromtimestamp(path.stat().st_mtime)
                if mtime < cutoff:
                    path.unlink()
                    removed += 1
            except Exception as e:
                logger.warning(f"Erro ao remover {path}: {e}")

        logger.info(f"Cleanup: {removed} vídeos removidos")
        return removed


# Exports
__all__ = [
    "VideoConfig",
    "Recording",
    "VideoRecorder",
]
```

---

## Módulo HLS-BRW-005: AccessibilityChecker

```
ID: HLS-BRW-005
Nome: AccessibilityChecker
Caminho: src/hl_mcp/browser/accessibility.py
Dependências: HLS-BRW-001, HLS-MDL-001 (Finding)
Exports: AccessibilityChecker, AccessibilityViolation, WCAGLevel
Linhas: ~280
```

### Código

```python
"""
HLS-BRW-005: AccessibilityChecker
=================================

Validação de acessibilidade usando axe-core.

Responsabilidades:
- Injetar e executar axe-core na página
- Converter violações para Findings
- Suportar níveis WCAG (A, AA, AAA)
- Fornecer detalhes de correção

Exemplo:
    >>> checker = AccessibilityChecker(page)
    >>> result = await checker.check()
    >>> print(f"Violações: {len(result.violations)}")
    >>> for v in result.violations:
    ...     print(f"  - {v.id}: {v.description}")

Changelog:
    - v1.0.0 (2026-02-01): Implementação inicial
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
import logging
import json

try:
    from playwright.async_api import Page
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    Page = Any

logger = logging.getLogger(__name__)


class WCAGLevel(str, Enum):
    """Níveis de conformidade WCAG."""

    A = "A"          # Nível básico
    AA = "AA"        # Nível recomendado
    AAA = "AAA"      # Nível mais alto


class ImpactLevel(str, Enum):
    """Níveis de impacto de violação."""

    MINOR = "minor"
    MODERATE = "moderate"
    SERIOUS = "serious"
    CRITICAL = "critical"


@dataclass
class AccessibilityViolation:
    """
    Representa uma violação de acessibilidade.

    Attributes:
        id: ID da regra (e.g., "color-contrast")
        description: Descrição da violação
        help: Texto de ajuda
        help_url: URL com mais informações
        impact: Nível de impacto
        wcag_tags: Tags WCAG relacionadas
        nodes: Elementos afetados
        html: HTML do elemento violador
        selector: Seletor CSS do elemento
    """

    id: str
    description: str
    help: str
    help_url: str
    impact: ImpactLevel
    wcag_tags: list[str] = field(default_factory=list)
    nodes: list[dict] = field(default_factory=list)
    html: str = ""
    selector: str = ""

    @classmethod
    def from_axe_result(cls, violation: dict) -> list["AccessibilityViolation"]:
        """
        Cria violações a partir de resultado axe-core.

        Uma violação axe pode ter múltiplos nodes afetados.
        Retorna uma AccessibilityViolation por node.
        """
        results = []

        for node in violation.get("nodes", []):
            results.append(cls(
                id=violation.get("id", "unknown"),
                description=violation.get("description", ""),
                help=violation.get("help", ""),
                help_url=violation.get("helpUrl", ""),
                impact=ImpactLevel(violation.get("impact", "minor")),
                wcag_tags=[
                    tag for tag in violation.get("tags", [])
                    if tag.startswith("wcag")
                ],
                nodes=[node],
                html=node.get("html", ""),
                selector=", ".join(node.get("target", [])),
            ))

        return results

    def to_finding(self) -> dict:
        """Converte para formato Finding do HLS-MDL-001."""
        return {
            "id": f"a11y-{self.id}",
            "category": "accessibility",
            "severity": self._impact_to_severity(),
            "title": self.help,
            "description": self.description,
            "evidence": self.html,
            "selector": self.selector,
            "remediation": f"See: {self.help_url}",
            "wcag_tags": self.wcag_tags,
        }

    def _impact_to_severity(self) -> str:
        """Mapeia impact para severity."""
        mapping = {
            ImpactLevel.MINOR: "low",
            ImpactLevel.MODERATE: "medium",
            ImpactLevel.SERIOUS: "high",
            ImpactLevel.CRITICAL: "critical",
        }
        return mapping.get(self.impact, "medium")


@dataclass
class AccessibilityResult:
    """
    Resultado de uma verificação de acessibilidade.

    Attributes:
        url: URL verificada
        timestamp: Momento da verificação
        violations: Lista de violações encontradas
        passes: Número de regras passando
        incomplete: Número de verificações incompletas
        inapplicable: Número de regras não aplicáveis
        wcag_level: Nível WCAG testado
    """

    url: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    violations: list[AccessibilityViolation] = field(default_factory=list)
    passes: int = 0
    incomplete: int = 0
    inapplicable: int = 0
    wcag_level: WCAGLevel = WCAGLevel.AA

    @property
    def is_compliant(self) -> bool:
        """Retorna True se não há violações."""
        return len(self.violations) == 0

    @property
    def critical_count(self) -> int:
        """Número de violações críticas."""
        return sum(
            1 for v in self.violations
            if v.impact == ImpactLevel.CRITICAL
        )

    @property
    def serious_count(self) -> int:
        """Número de violações sérias."""
        return sum(
            1 for v in self.violations
            if v.impact == ImpactLevel.SERIOUS
        )

    def to_dict(self) -> dict:
        """Converte para dicionário."""
        return {
            "url": self.url,
            "timestamp": self.timestamp.isoformat(),
            "is_compliant": self.is_compliant,
            "wcag_level": self.wcag_level.value,
            "summary": {
                "violations": len(self.violations),
                "critical": self.critical_count,
                "serious": self.serious_count,
                "passes": self.passes,
                "incomplete": self.incomplete,
                "inapplicable": self.inapplicable,
            },
            "violations": [
                {
                    "id": v.id,
                    "impact": v.impact.value,
                    "description": v.description,
                    "selector": v.selector,
                }
                for v in self.violations
            ],
        }


# axe-core CDN URL
AXE_CORE_CDN = "https://cdnjs.cloudflare.com/ajax/libs/axe-core/4.8.4/axe.min.js"


class AccessibilityChecker:
    """
    Verificador de acessibilidade usando axe-core.

    Injeta axe-core na página e executa verificações WCAG.

    Example:
        >>> checker = AccessibilityChecker(page, level=WCAGLevel.AA)
        >>> result = await checker.check()
        >>> if not result.is_compliant:
        ...     for v in result.violations:
        ...         print(f"{v.impact.value}: {v.description}")
    """

    def __init__(
        self,
        page: Page,
        level: WCAGLevel = WCAGLevel.AA,
        exclude_selectors: Optional[list[str]] = None,
    ):
        """
        Inicializa o checker.

        Args:
            page: Página Playwright
            level: Nível WCAG a verificar
            exclude_selectors: Seletores a ignorar na verificação
        """
        self.page = page
        self.level = level
        self.exclude_selectors = exclude_selectors or []
        self._axe_injected = False

    async def _inject_axe(self) -> None:
        """Injeta axe-core na página se necessário."""
        if self._axe_injected:
            return

        # Verifica se já está injetado
        has_axe = await self.page.evaluate("typeof axe !== 'undefined'")
        if has_axe:
            self._axe_injected = True
            return

        # Injeta via CDN
        await self.page.add_script_tag(url=AXE_CORE_CDN)

        # Aguarda carregamento
        await self.page.wait_for_function("typeof axe !== 'undefined'")

        self._axe_injected = True
        logger.debug("axe-core injetado com sucesso")

    def _get_run_options(self) -> dict:
        """Gera opções de execução do axe."""
        # Tags WCAG por nível
        wcag_tags = {
            WCAGLevel.A: ["wcag2a", "wcag21a"],
            WCAGLevel.AA: ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"],
            WCAGLevel.AAA: ["wcag2a", "wcag2aa", "wcag2aaa", "wcag21a", "wcag21aa", "wcag21aaa"],
        }

        options = {
            "runOnly": {
                "type": "tag",
                "values": wcag_tags.get(self.level, wcag_tags[WCAGLevel.AA]),
            },
        }

        if self.exclude_selectors:
            options["exclude"] = [[sel] for sel in self.exclude_selectors]

        return options

    async def check(
        self,
        selector: Optional[str] = None,
    ) -> AccessibilityResult:
        """
        Executa verificação de acessibilidade.

        Args:
            selector: Se fornecido, verifica apenas este elemento

        Returns:
            AccessibilityResult com violações encontradas
        """
        await self._inject_axe()

        options = self._get_run_options()

        # Monta comando axe.run()
        if selector:
            script = f"axe.run('{selector}', {json.dumps(options)})"
        else:
            script = f"axe.run({json.dumps(options)})"

        try:
            axe_result = await self.page.evaluate(script)
        except Exception as e:
            logger.error(f"Falha ao executar axe-core: {e}")
            return AccessibilityResult(
                url=self.page.url,
                wcag_level=self.level,
            )

        # Processa violações
        violations = []
        for violation_data in axe_result.get("violations", []):
            violations.extend(
                AccessibilityViolation.from_axe_result(violation_data)
            )

        result = AccessibilityResult(
            url=self.page.url,
            violations=violations,
            passes=len(axe_result.get("passes", [])),
            incomplete=len(axe_result.get("incomplete", [])),
            inapplicable=len(axe_result.get("inapplicable", [])),
            wcag_level=self.level,
        )

        logger.info(
            f"Acessibilidade verificada: {len(violations)} violações, "
            f"{result.passes} passes ({self.level.value})"
        )

        return result

    async def check_element(self, selector: str) -> AccessibilityResult:
        """Atalho para verificar elemento específico."""
        return await self.check(selector=selector)

    async def get_violations_as_findings(self) -> list[dict]:
        """
        Executa verificação e retorna violações como Findings.

        Returns:
            Lista de dicts no formato Finding
        """
        result = await self.check()
        return [v.to_finding() for v in result.violations]


# Exports
__all__ = [
    "WCAGLevel",
    "ImpactLevel",
    "AccessibilityViolation",
    "AccessibilityResult",
    "AccessibilityChecker",
]
```

---

## Módulo HLS-BRW-006: JourneyExecutor

```
ID: HLS-BRW-006
Nome: JourneyExecutor
Caminho: src/hl_mcp/browser/journey.py
Dependências: HLS-BRW-001, HLS-BRW-002, HLS-BRW-003, HLS-MDL-004
Exports: JourneyExecutor, JourneyExecutionResult
Linhas: ~300
```

### Código

```python
"""
HLS-BRW-006: JourneyExecutor
============================

Execução de jornadas de usuário completas.

Responsabilidades:
- Executar sequência de steps de jornada
- Capturar screenshots em cada step
- Registrar tempos de execução
- Coletar métricas de performance
- Integrar com accessibility checker

Uma jornada é uma sequência de ações que simula
um fluxo completo de usuário (login, checkout, etc).

Exemplo:
    >>> journey = Journey(
    ...     id="login_flow",
    ...     steps=[
    ...         JourneyStep(action="navigate", target="https://app.com/login"),
    ...         JourneyStep(action="type", target="#email", value="test@test.com"),
    ...         JourneyStep(action="type", target="#password", value="secret"),
    ...         JourneyStep(action="click", target="#submit"),
    ...         JourneyStep(action="wait", target="#dashboard", value="visible"),
    ...     ]
    ... )
    >>> executor = JourneyExecutor(driver)
    >>> result = await executor.execute(journey)
    >>> print(f"Sucesso: {result.success}, Steps: {len(result.step_results)}")

Changelog:
    - v1.0.0 (2026-02-01): Implementação inicial
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
import logging

from .driver import BrowserDriver, BrowserConfig
from .actions import BrowserActions, ActionResult, WaitStrategy
from .screenshots import ScreenshotManager, Screenshot
from .accessibility import AccessibilityChecker, AccessibilityResult

logger = logging.getLogger(__name__)


@dataclass
class JourneyStep:
    """
    Um passo de uma jornada.

    Attributes:
        action: Tipo de ação (navigate, click, type, wait, etc)
        target: Alvo da ação (URL ou seletor)
        value: Valor para a ação (texto, estado de espera)
        screenshot: Capturar screenshot após step
        accessibility_check: Verificar acessibilidade após step
        timeout: Timeout específico para este step
        description: Descrição legível do step
    """

    action: str
    target: str
    value: Optional[str] = None
    screenshot: bool = False
    accessibility_check: bool = False
    timeout: Optional[int] = None
    description: Optional[str] = None

    def to_dict(self) -> dict:
        """Converte para dicionário."""
        return {
            "action": self.action,
            "target": self.target,
            "value": self.value,
            "screenshot": self.screenshot,
            "accessibility_check": self.accessibility_check,
            "description": self.description,
        }


@dataclass
class StepResult:
    """
    Resultado de execução de um step.

    Attributes:
        step: Step executado
        success: Se foi bem sucedido
        action_result: Resultado da ação
        screenshot: Screenshot capturado (se solicitado)
        accessibility: Resultado de acessibilidade (se verificado)
        error: Erro (se falhou)
        duration_ms: Duração em ms
    """

    step: JourneyStep
    success: bool
    action_result: Optional[ActionResult] = None
    screenshot: Optional[Screenshot] = None
    accessibility: Optional[AccessibilityResult] = None
    error: Optional[str] = None
    duration_ms: float = 0

    def to_dict(self) -> dict:
        """Converte para dicionário."""
        return {
            "step": self.step.to_dict(),
            "success": self.success,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "screenshot_path": str(self.screenshot.path) if self.screenshot else None,
            "accessibility_compliant": (
                self.accessibility.is_compliant if self.accessibility else None
            ),
        }


@dataclass
class Journey:
    """
    Definição de uma jornada de usuário.

    Attributes:
        id: Identificador único
        name: Nome legível
        description: Descrição da jornada
        steps: Lista de steps
        tags: Tags para categorização
        setup_steps: Steps de preparação (executados antes)
        teardown_steps: Steps de limpeza (executados depois)
    """

    id: str
    name: str = ""
    description: str = ""
    steps: list[JourneyStep] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    setup_steps: list[JourneyStep] = field(default_factory=list)
    teardown_steps: list[JourneyStep] = field(default_factory=list)

    def __post_init__(self):
        if not self.name:
            self.name = self.id


@dataclass
class JourneyExecutionResult:
    """
    Resultado completo de execução de jornada.

    Attributes:
        journey: Jornada executada
        success: Se todos os steps passaram
        step_results: Resultados de cada step
        total_duration_ms: Duração total em ms
        screenshots: Todos os screenshots capturados
        accessibility_results: Todos os resultados de acessibilidade
        start_time: Início da execução
        end_time: Fim da execução
    """

    journey: Journey
    success: bool
    step_results: list[StepResult] = field(default_factory=list)
    total_duration_ms: float = 0
    screenshots: list[Screenshot] = field(default_factory=list)
    accessibility_results: list[AccessibilityResult] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None

    @property
    def failed_steps(self) -> list[StepResult]:
        """Retorna steps que falharam."""
        return [r for r in self.step_results if not r.success]

    @property
    def passed_steps(self) -> int:
        """Número de steps que passaram."""
        return sum(1 for r in self.step_results if r.success)

    def to_dict(self) -> dict:
        """Converte para dicionário."""
        return {
            "journey_id": self.journey.id,
            "journey_name": self.journey.name,
            "success": self.success,
            "total_steps": len(self.step_results),
            "passed_steps": self.passed_steps,
            "failed_steps": len(self.failed_steps),
            "total_duration_ms": self.total_duration_ms,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "step_results": [r.to_dict() for r in self.step_results],
        }


class JourneyExecutor:
    """
    Executor de jornadas de usuário.

    Executa sequência de steps, captura evidências,
    e coleta métricas de cada interação.

    Example:
        >>> async with BrowserDriver() as driver:
        ...     executor = JourneyExecutor(driver)
        ...     result = await executor.execute(journey)
        ...     print(f"Passou: {result.success}")
    """

    def __init__(
        self,
        driver: BrowserDriver,
        screenshot_dir: str = "./journey_screenshots",
        default_timeout: int = 30000,
        stop_on_failure: bool = True,
    ):
        """
        Inicializa o executor.

        Args:
            driver: BrowserDriver iniciado
            screenshot_dir: Diretório para screenshots
            default_timeout: Timeout padrão em ms
            stop_on_failure: Parar na primeira falha
        """
        self.driver = driver
        self.screenshot_dir = screenshot_dir
        self.default_timeout = default_timeout
        self.stop_on_failure = stop_on_failure

    async def execute(
        self,
        journey: Journey,
        context_options: Optional[dict] = None,
    ) -> JourneyExecutionResult:
        """
        Executa uma jornada completa.

        Args:
            journey: Jornada a executar
            context_options: Opções para o contexto de browser

        Returns:
            JourneyExecutionResult com todos os resultados
        """
        start_time = datetime.utcnow()

        # Cria página para a jornada
        page = await self.driver.new_page()
        actions = BrowserActions(
            page,
            default_timeout=self.default_timeout,
            screenshot_on_failure=True,
            screenshot_dir=self.screenshot_dir,
        )
        screenshot_manager = ScreenshotManager(
            page,
            session_id=journey.id,
        )
        accessibility_checker = AccessibilityChecker(page)

        step_results: list[StepResult] = []
        screenshots: list[Screenshot] = []
        accessibility_results: list[AccessibilityResult] = []
        overall_success = True

        # Combina todos os steps
        all_steps = (
            journey.setup_steps +
            journey.steps +
            journey.teardown_steps
        )

        for i, step in enumerate(all_steps):
            step_start = datetime.utcnow()

            logger.info(
                f"Executando step {i+1}/{len(all_steps)}: "
                f"{step.action} -> {step.target}"
            )

            try:
                # Executa ação
                action_result = await self._execute_step(actions, step)

                step_result = StepResult(
                    step=step,
                    success=action_result.success,
                    action_result=action_result,
                    error=action_result.error,
                    duration_ms=action_result.duration_ms,
                )

                # Captura screenshot se solicitado
                if step.screenshot and action_result.success:
                    try:
                        shot = await screenshot_manager.capture(
                            f"step_{i+1}_{step.action}"
                        )
                        step_result.screenshot = shot
                        screenshots.append(shot)
                    except Exception as e:
                        logger.warning(f"Falha ao capturar screenshot: {e}")

                # Verifica acessibilidade se solicitado
                if step.accessibility_check and action_result.success:
                    try:
                        a11y = await accessibility_checker.check()
                        step_result.accessibility = a11y
                        accessibility_results.append(a11y)
                    except Exception as e:
                        logger.warning(f"Falha ao verificar acessibilidade: {e}")

                step_results.append(step_result)

                if not action_result.success:
                    overall_success = False
                    if self.stop_on_failure:
                        logger.warning(f"Step falhou, parando execução")
                        break

            except Exception as e:
                overall_success = False
                step_result = StepResult(
                    step=step,
                    success=False,
                    error=str(e),
                    duration_ms=(datetime.utcnow() - step_start).total_seconds() * 1000,
                )
                step_results.append(step_result)

                if self.stop_on_failure:
                    break

        # Fecha página
        await self.driver.close_page(page)

        end_time = datetime.utcnow()
        total_duration = (end_time - start_time).total_seconds() * 1000

        return JourneyExecutionResult(
            journey=journey,
            success=overall_success,
            step_results=step_results,
            total_duration_ms=total_duration,
            screenshots=screenshots,
            accessibility_results=accessibility_results,
            start_time=start_time,
            end_time=end_time,
        )

    async def _execute_step(
        self,
        actions: BrowserActions,
        step: JourneyStep,
    ) -> ActionResult:
        """
        Executa um step individual.

        Args:
            actions: BrowserActions instance
            step: Step a executar

        Returns:
            ActionResult da ação
        """
        action = step.action.lower()
        timeout = step.timeout or self.default_timeout

        if action == "navigate" or action == "goto":
            return await actions.navigate(
                step.target,
                timeout=timeout,
            )

        elif action == "click":
            return await actions.click(
                step.target,
                timeout=timeout,
            )

        elif action == "type":
            return await actions.type_text(
                step.target,
                step.value or "",
                timeout=timeout,
            )

        elif action == "fill":
            return await actions.fill(
                step.target,
                step.value or "",
                timeout=timeout,
            )

        elif action == "select":
            return await actions.select_option(
                step.target,
                value=step.value,
                timeout=timeout,
            )

        elif action == "hover":
            return await actions.hover(
                step.target,
                timeout=timeout,
            )

        elif action == "check":
            return await actions.check(
                step.target,
                timeout=timeout,
            )

        elif action == "uncheck":
            return await actions.uncheck(
                step.target,
                timeout=timeout,
            )

        elif action == "wait":
            return await actions.wait_for_selector(
                step.target,
                state=step.value or "visible",
                timeout=timeout,
            )

        elif action == "scroll":
            return await actions.scroll_to(
                selector=step.target if step.target != "page" else None,
            )

        elif action == "extract" or action == "get_text":
            return await actions.get_text(
                step.target,
                timeout=timeout,
            )

        else:
            from .actions import ActionResult, ActionType
            return ActionResult(
                success=False,
                action_type=ActionType.WAIT,
                error=f"Ação desconhecida: {action}",
            )


# Exports
__all__ = [
    "JourneyStep",
    "StepResult",
    "Journey",
    "JourneyExecutionResult",
    "JourneyExecutor",
]
```

---

## Browser __init__.py

```python
"""
HLS-BRW: Browser Automation Module
==================================

Módulos Lego para automação de browser com Playwright.

Exports:
    - BrowserDriver, BrowserConfig, BrowserType
    - BrowserActions, ActionResult, WaitStrategy
    - ScreenshotManager, Screenshot, ScreenshotConfig
    - VideoRecorder, Recording, VideoConfig
    - AccessibilityChecker, AccessibilityResult, WCAGLevel
    - JourneyExecutor, Journey, JourneyStep
"""

# HLS-BRW-001: Driver
from .driver import (
    BrowserType,
    BrowserConfig,
    BrowserDriver,
    create_browser,
    PLAYWRIGHT_AVAILABLE,
)

# HLS-BRW-002: Actions
from .actions import (
    WaitStrategy,
    ActionType,
    ActionResult,
    BrowserActions,
)

# HLS-BRW-003: Screenshots
from .screenshots import (
    ScreenshotConfig,
    Screenshot,
    ScreenshotManager,
)

# HLS-BRW-004: Video
from .video import (
    VideoConfig,
    Recording,
    VideoRecorder,
)

# HLS-BRW-005: Accessibility
from .accessibility import (
    WCAGLevel,
    ImpactLevel,
    AccessibilityViolation,
    AccessibilityResult,
    AccessibilityChecker,
)

# HLS-BRW-006: Journey
from .journey import (
    JourneyStep,
    StepResult,
    Journey,
    JourneyExecutionResult,
    JourneyExecutor,
)


__all__ = [
    # Driver
    "BrowserType",
    "BrowserConfig",
    "BrowserDriver",
    "create_browser",
    "PLAYWRIGHT_AVAILABLE",
    # Actions
    "WaitStrategy",
    "ActionType",
    "ActionResult",
    "BrowserActions",
    # Screenshots
    "ScreenshotConfig",
    "Screenshot",
    "ScreenshotManager",
    # Video
    "VideoConfig",
    "Recording",
    "VideoRecorder",
    # Accessibility
    "WCAGLevel",
    "ImpactLevel",
    "AccessibilityViolation",
    "AccessibilityResult",
    "AccessibilityChecker",
    # Journey
    "JourneyStep",
    "StepResult",
    "Journey",
    "JourneyExecutionResult",
    "JourneyExecutor",
]
```

---

## Testes - tests/test_browser.py

```python
"""
Testes para módulos HLS-BRW (Browser Automation).
"""

import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


# ============================================================
# HLS-BRW-001: BrowserDriver Tests
# ============================================================

class TestBrowserConfig:
    """Testes para BrowserConfig."""

    def test_default_config(self):
        """Config padrão tem valores sensatos."""
        from hl_mcp.browser.driver import BrowserConfig, BrowserType

        config = BrowserConfig()

        assert config.browser_type == BrowserType.CHROMIUM
        assert config.headless is True
        assert config.viewport_width == 1280
        assert config.viewport_height == 720
        assert config.locale == "en-US"

    def test_to_browser_args(self):
        """to_browser_args() gera dict correto."""
        from hl_mcp.browser.driver import BrowserConfig

        config = BrowserConfig(headless=False, slow_mo=100)
        args = config.to_browser_args()

        assert args["headless"] is False
        assert args["slow_mo"] == 100

    def test_to_context_args(self):
        """to_context_args() inclui viewport e locale."""
        from hl_mcp.browser.driver import BrowserConfig

        config = BrowserConfig(
            viewport_width=1920,
            viewport_height=1080,
            locale="pt-BR",
        )
        args = config.to_context_args()

        assert args["viewport"]["width"] == 1920
        assert args["viewport"]["height"] == 1080
        assert args["locale"] == "pt-BR"


# ============================================================
# HLS-BRW-002: BrowserActions Tests
# ============================================================

class TestActionResult:
    """Testes para ActionResult."""

    def test_action_result_creation(self):
        """ActionResult pode ser criado com valores básicos."""
        from hl_mcp.browser.actions import ActionResult, ActionType

        result = ActionResult(
            success=True,
            action_type=ActionType.CLICK,
            selector="#button",
        )

        assert result.success is True
        assert result.action_type == ActionType.CLICK
        assert result.selector == "#button"

    def test_action_result_to_dict(self):
        """to_dict() serializa corretamente."""
        from hl_mcp.browser.actions import ActionResult, ActionType

        result = ActionResult(
            success=False,
            action_type=ActionType.NAVIGATE,
            value="https://example.com",
            error="Timeout",
        )

        d = result.to_dict()

        assert d["success"] is False
        assert d["action_type"] == "navigate"
        assert d["error"] == "Timeout"


# ============================================================
# HLS-BRW-003: ScreenshotManager Tests
# ============================================================

class TestScreenshot:
    """Testes para Screenshot."""

    def test_screenshot_creation(self):
        """Screenshot pode ser criado."""
        from hl_mcp.browser.screenshots import Screenshot

        shot = Screenshot(
            path=Path("/tmp/test.png"),
            name="test",
            width=1280,
            height=720,
        )

        assert shot.name == "test"
        assert shot.width == 1280

    def test_screenshot_to_dict(self):
        """to_dict() serializa path como string."""
        from hl_mcp.browser.screenshots import Screenshot

        shot = Screenshot(
            path=Path("/tmp/test.png"),
            name="test",
        )

        d = shot.to_dict()

        assert d["path"] == "/tmp/test.png"
        assert isinstance(d["timestamp"], str)


class TestScreenshotConfig:
    """Testes para ScreenshotConfig."""

    def test_default_config(self):
        """Config padrão é PNG, organizado por data."""
        from hl_mcp.browser.screenshots import ScreenshotConfig

        config = ScreenshotConfig()

        assert config.format == "png"
        assert config.organize_by_date is True
        assert config.max_age_days == 7


# ============================================================
# HLS-BRW-004: VideoRecorder Tests
# ============================================================

class TestRecording:
    """Testes para Recording."""

    def test_recording_creation(self):
        """Recording pode ser criado."""
        from hl_mcp.browser.video import Recording

        rec = Recording(
            path=Path("/tmp/video.webm"),
            session_id="test123",
            start_time=datetime.utcnow(),
        )

        assert rec.session_id == "test123"
        assert rec.duration_seconds == 0

    def test_recording_to_dict(self):
        """to_dict() serializa corretamente."""
        from hl_mcp.browser.video import Recording

        rec = Recording(
            path=Path("/tmp/video.webm"),
            session_id="test",
            start_time=datetime.utcnow(),
            duration_seconds=30.5,
        )

        d = rec.to_dict()

        assert d["duration_seconds"] == 30.5
        assert "session_id" in d


# ============================================================
# HLS-BRW-005: AccessibilityChecker Tests
# ============================================================

class TestAccessibilityViolation:
    """Testes para AccessibilityViolation."""

    def test_violation_creation(self):
        """Violation pode ser criada."""
        from hl_mcp.browser.accessibility import (
            AccessibilityViolation,
            ImpactLevel,
        )

        v = AccessibilityViolation(
            id="color-contrast",
            description="Insufficient color contrast",
            help="Fix color contrast",
            help_url="https://example.com",
            impact=ImpactLevel.SERIOUS,
        )

        assert v.id == "color-contrast"
        assert v.impact == ImpactLevel.SERIOUS

    def test_from_axe_result(self):
        """from_axe_result() parseia resultado axe-core."""
        from hl_mcp.browser.accessibility import AccessibilityViolation

        axe_violation = {
            "id": "button-name",
            "description": "Buttons must have discernible text",
            "help": "Button has no accessible name",
            "helpUrl": "https://example.com",
            "impact": "critical",
            "tags": ["wcag2a", "wcag412"],
            "nodes": [
                {
                    "html": "<button></button>",
                    "target": ["#submit"],
                },
            ],
        }

        violations = AccessibilityViolation.from_axe_result(axe_violation)

        assert len(violations) == 1
        assert violations[0].id == "button-name"
        assert violations[0].selector == "#submit"

    def test_to_finding(self):
        """to_finding() converte para formato Finding."""
        from hl_mcp.browser.accessibility import (
            AccessibilityViolation,
            ImpactLevel,
        )

        v = AccessibilityViolation(
            id="test-rule",
            description="Test description",
            help="Test help",
            help_url="https://example.com",
            impact=ImpactLevel.CRITICAL,
            wcag_tags=["wcag2a"],
        )

        finding = v.to_finding()

        assert finding["id"] == "a11y-test-rule"
        assert finding["category"] == "accessibility"
        assert finding["severity"] == "critical"


class TestAccessibilityResult:
    """Testes para AccessibilityResult."""

    def test_is_compliant(self):
        """is_compliant é True quando não há violações."""
        from hl_mcp.browser.accessibility import AccessibilityResult

        result = AccessibilityResult(url="https://example.com")

        assert result.is_compliant is True

    def test_critical_count(self):
        """critical_count conta violações críticas."""
        from hl_mcp.browser.accessibility import (
            AccessibilityResult,
            AccessibilityViolation,
            ImpactLevel,
        )

        result = AccessibilityResult(
            url="https://example.com",
            violations=[
                AccessibilityViolation(
                    id="v1", description="", help="", help_url="",
                    impact=ImpactLevel.CRITICAL,
                ),
                AccessibilityViolation(
                    id="v2", description="", help="", help_url="",
                    impact=ImpactLevel.SERIOUS,
                ),
                AccessibilityViolation(
                    id="v3", description="", help="", help_url="",
                    impact=ImpactLevel.CRITICAL,
                ),
            ],
        )

        assert result.critical_count == 2
        assert result.serious_count == 1


# ============================================================
# HLS-BRW-006: JourneyExecutor Tests
# ============================================================

class TestJourneyStep:
    """Testes para JourneyStep."""

    def test_step_creation(self):
        """JourneyStep pode ser criado."""
        from hl_mcp.browser.journey import JourneyStep

        step = JourneyStep(
            action="click",
            target="#button",
            description="Click submit button",
        )

        assert step.action == "click"
        assert step.target == "#button"

    def test_step_to_dict(self):
        """to_dict() serializa corretamente."""
        from hl_mcp.browser.journey import JourneyStep

        step = JourneyStep(
            action="type",
            target="#email",
            value="test@test.com",
            screenshot=True,
        )

        d = step.to_dict()

        assert d["action"] == "type"
        assert d["value"] == "test@test.com"
        assert d["screenshot"] is True


class TestJourney:
    """Testes para Journey."""

    def test_journey_creation(self):
        """Journey pode ser criada com steps."""
        from hl_mcp.browser.journey import Journey, JourneyStep

        journey = Journey(
            id="login_test",
            name="Login Flow Test",
            steps=[
                JourneyStep(action="navigate", target="https://example.com"),
                JourneyStep(action="click", target="#login"),
            ],
        )

        assert journey.id == "login_test"
        assert len(journey.steps) == 2

    def test_journey_default_name(self):
        """Se name não for fornecido, usa id."""
        from hl_mcp.browser.journey import Journey

        journey = Journey(id="test_journey")

        assert journey.name == "test_journey"


class TestJourneyExecutionResult:
    """Testes para JourneyExecutionResult."""

    def test_failed_steps(self):
        """failed_steps retorna apenas steps que falharam."""
        from hl_mcp.browser.journey import (
            Journey,
            JourneyStep,
            JourneyExecutionResult,
            StepResult,
        )

        journey = Journey(id="test")
        step1 = JourneyStep(action="click", target="#a")
        step2 = JourneyStep(action="click", target="#b")

        result = JourneyExecutionResult(
            journey=journey,
            success=False,
            step_results=[
                StepResult(step=step1, success=True),
                StepResult(step=step2, success=False, error="Not found"),
            ],
        )

        assert len(result.failed_steps) == 1
        assert result.failed_steps[0].step.target == "#b"
        assert result.passed_steps == 1

    def test_to_dict(self):
        """to_dict() inclui resumo correto."""
        from hl_mcp.browser.journey import (
            Journey,
            JourneyExecutionResult,
        )

        result = JourneyExecutionResult(
            journey=Journey(id="test", name="Test Journey"),
            success=True,
            total_duration_ms=1500,
        )

        d = result.to_dict()

        assert d["journey_id"] == "test"
        assert d["journey_name"] == "Test Journey"
        assert d["success"] is True
        assert d["total_duration_ms"] == 1500
```

---

## Atualização do LEGO_INDEX.yaml

Adicionar ao índice existente:

```yaml
# ============================================================
# BROWSER AUTOMATION (HLS-BRW-001 a HLS-BRW-006)
# ============================================================

HLS-BRW-001:
  name: BrowserDriver
  path: src/hl_mcp/browser/driver.py
  category: browser
  description: Gerenciamento do ciclo de vida do browser Playwright
  exports:
    - BrowserType
    - BrowserConfig
    - BrowserDriver
    - create_browser
    - PLAYWRIGHT_AVAILABLE
  dependencies:
    - playwright
  search_hints:
    - browser
    - playwright
    - driver
    - headless
    - chromium
    - firefox
    - webkit

HLS-BRW-002:
  name: BrowserActions
  path: src/hl_mcp/browser/actions.py
  category: browser
  description: Primitivas de ação para automação de browser
  exports:
    - WaitStrategy
    - ActionType
    - ActionResult
    - BrowserActions
  dependencies:
    - HLS-BRW-001
  search_hints:
    - click
    - type
    - navigate
    - action
    - wait
    - selector
    - fill

HLS-BRW-003:
  name: ScreenshotManager
  path: src/hl_mcp/browser/screenshots.py
  category: browser
  description: Captura e gerenciamento de screenshots
  exports:
    - ScreenshotConfig
    - Screenshot
    - ScreenshotManager
  dependencies:
    - HLS-BRW-001
  search_hints:
    - screenshot
    - capture
    - image
    - png
    - jpeg
    - visual

HLS-BRW-004:
  name: VideoRecorder
  path: src/hl_mcp/browser/video.py
  category: browser
  description: Gravação de vídeo das sessões de browser
  exports:
    - VideoConfig
    - Recording
    - VideoRecorder
  dependencies:
    - HLS-BRW-001
  search_hints:
    - video
    - recording
    - session
    - webm
    - gravacao

HLS-BRW-005:
  name: AccessibilityChecker
  path: src/hl_mcp/browser/accessibility.py
  category: browser
  description: Validação de acessibilidade usando axe-core
  exports:
    - WCAGLevel
    - ImpactLevel
    - AccessibilityViolation
    - AccessibilityResult
    - AccessibilityChecker
  dependencies:
    - HLS-BRW-001
    - HLS-MDL-001
  search_hints:
    - accessibility
    - a11y
    - wcag
    - axe-core
    - violation
    - screen reader

HLS-BRW-006:
  name: JourneyExecutor
  path: src/hl_mcp/browser/journey.py
  category: browser
  description: Execução de jornadas de usuário completas
  exports:
    - JourneyStep
    - StepResult
    - Journey
    - JourneyExecutionResult
    - JourneyExecutor
  dependencies:
    - HLS-BRW-001
    - HLS-BRW-002
    - HLS-BRW-003
    - HLS-MDL-004
  search_hints:
    - journey
    - user journey
    - flow
    - test flow
    - sequence
    - steps
```

---

## Resumo do Block 07

| Módulo | ID | Linhas | Exports |
|--------|-----|--------|---------|
| BrowserDriver | HLS-BRW-001 | ~280 | 5 |
| BrowserActions | HLS-BRW-002 | ~350 | 4 |
| ScreenshotManager | HLS-BRW-003 | ~200 | 3 |
| VideoRecorder | HLS-BRW-004 | ~180 | 3 |
| AccessibilityChecker | HLS-BRW-005 | ~280 | 5 |
| JourneyExecutor | HLS-BRW-006 | ~300 | 5 |
| **TOTAL** | | **~1,590** | **25** |

---

## Próximo: Block 08 - Core Engine

O Block 08 vai cobrir:
1. **HLS-ENG-001**: HumanLayerRunner (executa uma layer)
2. **HLS-ENG-002**: TripleRedundancy (3 runs, 2/3 consensus)
3. **HLS-ENG-003**: LayerOrchestrator (orquestra 7 layers)
4. **HLS-ENG-004**: VetoGate (aplica vetos WEAK/MEDIUM/STRONG)
5. **HLS-ENG-005**: ConsensusEngine (consolida resultados)

Quer que eu continue com o Block 08 (Core Engine)?
