from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path

from .config import Settings
from .models import ChatMessage, CompletionResult


logger = logging.getLogger("uvicorn.error")


class Backend(ABC):
    def __init__(self, settings: Settings):
        self.settings = settings

    @abstractmethod
    async def complete(self, messages: list[ChatMessage], model: str | None = None, new_session: bool = False, level: str | None = None) -> CompletionResult:
        raise NotImplementedError

    async def health(self) -> dict:
        return {"ok": True, "backend": self.settings.backend}


class MockBackend(Backend):
    async def complete(self, messages: list[ChatMessage], model: str | None = None, new_session: bool = False, level: str | None = None) -> CompletionResult:
        last_user = next((m.text() for m in reversed(messages) if m.role == "user"), "")
        text = f"[mock:{self.settings.model_id}] {last_user}"
        return CompletionResult(text=text, model=model or self.settings.model_id, level=level, prompt_tokens=sum(len(m.text().split()) for m in messages), completion_tokens=len(text.split()))


class BrowserBackend(Backend):
    """Browser-backed ChatGPT.com worker.

    This intentionally keeps the first implementation conservative. It launches a
    persistent Chromium profile and drives ChatGPT through Playwright. Real
    production use should pin selectors with browser-level regression tests.
    """

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self._lock = asyncio.Lock()
        self._playwright = None
        self._context = None
        self._page = None

    async def _ensure_page(self):
        if self._page:
            return self._page
        try:
            from playwright.async_api import async_playwright
        except Exception as exc:  # pragma: no cover - depends on optional browser env
            raise RuntimeError("playwright is not installed") from exc

        Path(self.settings.profile_dir).mkdir(parents=True, exist_ok=True)
        self._playwright = await async_playwright().start()
        self._context = await self._playwright.chromium.launch_persistent_context(
            self.settings.profile_dir,
            headless=self.settings.headless,
            channel=self.settings.browser_channel,
            ignore_default_args=["--disable-extensions"] if self.settings.enable_extensions else None,
            viewport={"width": 1360, "height": 820},
            args=["--disable-blink-features=AutomationControlled"],
        )
        self._page = self._context.pages[0] if self._context.pages else await self._context.new_page()
        await self._page.goto("https://chatgpt.com/", wait_until="domcontentloaded", timeout=60_000)
        return self._page

    async def complete(self, messages: list[ChatMessage], model: str | None = None, new_session: bool = False, level: str | None = None) -> CompletionResult:
        async with self._lock:
            total_started = time.perf_counter()
            selected_model = model or self.settings.model_id
            selected_level = level or "-"
            message_count = len(messages)
            prompt = self._render_prompt(messages)
            prompt_chars = len(prompt)
            phase = "ensure_page"

            logger.info(
                "browser_request_start model=%s level=%s messages=%d prompt_chars=%d new_session=%s",
                selected_model,
                selected_level,
                message_count,
                prompt_chars,
                str(new_session).lower(),
            )

            try:
                phase_started = time.perf_counter()
                page = await self._ensure_page()
                ensure_page_ms = (time.perf_counter() - phase_started) * 1000

                phase = "auth_check"
                if "log in" in (await page.title()).lower():
                    raise RuntimeError(
                        "ChatGPT browser profile is not logged in; "
                        "run setup with a visible browser first"
                    )

                new_session_ms = 0.0
                if new_session:
                    phase = "new_session"
                    phase_started = time.perf_counter()
                    await self._start_new_session(page)
                    new_session_ms = (time.perf_counter() - phase_started) * 1000

                phase = "preferences"
                phase_started = time.perf_counter()
                await self._apply_preferences(page, selected_model, level)
                preferences_ms = (time.perf_counter() - phase_started) * 1000

                phase = "composer_wait"
                composer = page.locator(
                    "#prompt-textarea, div[contenteditable='true']"
                ).last
                phase_started = time.perf_counter()
                await composer.wait_for(timeout=30_000)
                composer_wait_ms = (time.perf_counter() - phase_started) * 1000

                phase = "fill"
                phase_started = time.perf_counter()
                try:
                    await composer.fill(prompt)
                except Exception as exc:
                    # Playwright includes the fill() value in its call log.
                    # Convert the exception here so prompts never reach journald.
                    raise RuntimeError(
                        f"ChatGPT composer fill failed ({type(exc).__name__})"
                    ) from None
                fill_ms = (time.perf_counter() - phase_started) * 1000

                phase = "submit"
                phase_started = time.perf_counter()
                await self._submit_prompt(page, composer)
                submit_ms = (time.perf_counter() - phase_started) * 1000

                phase = "generation"
                phase_started = time.perf_counter()
                await self._wait_until_idle(page)
                generation_ms = (time.perf_counter() - phase_started) * 1000

                phase = "extraction"
                phase_started = time.perf_counter()
                text = await self._extract_last_answer(page)
                extraction_ms = (time.perf_counter() - phase_started) * 1000

                total_ms = (time.perf_counter() - total_started) * 1000

                logger.info(
                    "browser_request_complete model=%s level=%s "
                    "messages=%d prompt_chars=%d new_session=%s "
                    "ensure_page_ms=%.1f new_session_ms=%.1f "
                    "preferences_ms=%.1f composer_wait_ms=%.1f "
                    "fill_ms=%.1f submit_ms=%.1f generation_ms=%.1f "
                    "extraction_ms=%.1f total_ms=%.1f",
                    selected_model,
                    selected_level,
                    message_count,
                    prompt_chars,
                    str(new_session).lower(),
                    ensure_page_ms,
                    new_session_ms,
                    preferences_ms,
                    composer_wait_ms,
                    fill_ms,
                    submit_ms,
                    generation_ms,
                    extraction_ms,
                    total_ms,
                )

                return CompletionResult(
                    text=text,
                    model=selected_model,
                    level=level,
                )

            except Exception as exc:
                total_ms = (time.perf_counter() - total_started) * 1000
                logger.error(
                    "browser_request_failed model=%s level=%s "
                    "messages=%d prompt_chars=%d new_session=%s "
                    "stage=%s error_type=%s total_ms=%.1f",
                    selected_model,
                    selected_level,
                    message_count,
                    prompt_chars,
                    str(new_session).lower(),
                    phase,
                    type(exc).__name__,
                    total_ms,
                )
                raise

    async def health(self) -> dict:
        try:
            page = await self._ensure_page()
            title = await page.title()
            return {"ok": True, "backend": "browser", "title": title, "logged_in_hint": "log in" not in title.lower()}
        except Exception as exc:
            return {"ok": False, "backend": "browser", "error": str(exc)}

    async def _apply_preferences(self, page, model: str, level: str | None) -> None:  # pragma: no cover - browser integration
        """Best-effort model / reasoning-level selection in the ChatGPT UI.

        ChatGPT UI changes often. Failure to locate a control is non-fatal: the
        request still runs with the currently selected browser model, while the
        API response records the requested model/level.
        """
        await self._try_select_label(page, self.settings.model_label(model))
        if level:
            await self._try_select_label(page, self.settings.level_label(level))

    @staticmethod
    async def _try_select_label(page, label: str) -> bool:  # pragma: no cover - browser integration
        """Select an exact model/reasoning option from the ChatGPT UI.

        Exact matching is required so labels such as "High" cannot match
        "Extra High". Opening a menu is not considered success; an exact
        option must actually be clicked.
        """
        if not label:
            return False

        exact_options = (
            f"[role='menuitem']:text-is('{label}')",
            f"[role='menuitemradio']:text-is('{label}')",
            f"[role='option']:text-is('{label}')",
            f"[data-radix-collection-item]:text-is('{label}')",
        )

        async def click_exact_option() -> bool:
            for selector in exact_options:
                try:
                    options = page.locator(selector)
                    count = await options.count()

                    for index in range(count):
                        option = options.nth(index)
                        if await option.is_visible(timeout=500):
                            await option.click(timeout=3000)
                            await page.wait_for_timeout(500)
                            return True
                except Exception:
                    continue

            return False

        # Handle the case where a menu is already open.
        if await click_exact_option():
            return True

        opener_selectors = (
            "[data-testid='model-switcher-dropdown-button']",
            "button[aria-haspopup='menu']",
            "button:has-text('ChatGPT')",
            "button:has-text('GPT')",
        )

        # Walk every candidate opener. Using only `.first` can repeatedly
        # open the model menu while never reaching the reasoning menu.
        for opener_selector in opener_selectors:
            try:
                openers = page.locator(opener_selector)
                opener_count = min(await openers.count(), 12)
            except Exception:
                continue

            for index in range(opener_count):
                try:
                    opener = openers.nth(index)

                    if not await opener.is_visible(timeout=500):
                        continue

                    await opener.click(timeout=3000)
                    await page.wait_for_timeout(500)

                    if await click_exact_option():
                        return True

                    # Do not leave the wrong menu open while trying another.
                    await page.keyboard.press("Escape")
                    await page.wait_for_timeout(200)

                except Exception:
                    try:
                        await page.keyboard.press("Escape")
                    except Exception:
                        pass
                    continue

        return False

    @staticmethod
    def _render_prompt(messages: list[ChatMessage]) -> str:
        return "\n\n".join(f"{m.role.upper()}: {m.text()}" for m in messages)

    @staticmethod
    async def _submit_prompt(page, composer) -> None:  # pragma: no cover - browser integration
        """Wait for the ChatGPT Send button, click it, and verify submission.

        Large fills can leave the composer populated before ChatGPT has made
        Send actionable. Waiting here prevents a race where an Enter keypress
        is ignored and the request appears to hang.
        """
        selectors = (
            "button[data-testid='send-button']",
            "button[aria-label='Send prompt']",
            "button[aria-label='Send message']",
            "button[aria-label='Send']",
        )

        deadline = time.monotonic() + 30.0
        send_button = None

        while time.monotonic() < deadline:
            for selector in selectors:
                try:
                    buttons = page.locator(selector)
                    count = min(await buttons.count(), 4)

                    for index in range(count):
                        button = buttons.nth(index)

                        if not await button.is_visible(timeout=250):
                            continue

                        if await button.is_enabled(timeout=250):
                            send_button = button
                            break

                    if send_button is not None:
                        break

                except Exception:
                    continue

            if send_button is not None:
                break

            await page.wait_for_timeout(250)

        if send_button is None:
            raise RuntimeError(
                "ChatGPT Send button did not become enabled within 30 seconds"
            )

        await send_button.click(timeout=3000)

        # Confirm that the click actually submitted the prompt.
        stop = page.locator(
            "button[aria-label*='Stop'], button[data-testid*='stop']"
        )

        confirm_deadline = time.monotonic() + 5.0

        while time.monotonic() < confirm_deadline:
            try:
                stop_count = await stop.count()
                for index in range(stop_count):
                    if await stop.nth(index).is_visible(timeout=200):
                        return
            except Exception:
                pass

            try:
                remaining = (
                    await composer.inner_text(timeout=500)
                ).strip()

                if not remaining:
                    return
            except Exception:
                # Composer replacement/removal also indicates submission.
                return

            await page.wait_for_timeout(200)

        raise RuntimeError(
            "ChatGPT Send button was clicked but submission was not confirmed"
        )

    @staticmethod
    async def _wait_until_idle(page) -> None:  # pragma: no cover - browser integration
        stop = page.locator("button[aria-label*='Stop'], button[data-testid*='stop']")
        for _ in range(240):
            try:
                if await stop.count() == 0:
                    await page.wait_for_timeout(1500)
                    return
            except Exception:
                return
            await page.wait_for_timeout(1000)

    @staticmethod
    async def _start_new_session(page) -> None:  # pragma: no cover - browser integration
        """Move ChatGPT to a fresh conversation before sending the prompt."""
        await page.goto("https://chatgpt.com/", wait_until="domcontentloaded", timeout=60_000)
        await page.wait_for_timeout(1500)
        if await page.locator("#prompt-textarea, div[contenteditable='true']").count() > 0:
            return
        for selector in (
            "[data-testid='create-new-chat-button']",
            "button[aria-label*='New chat']",
            "a[aria-label*='New chat']",
            "a[href='/']",
        ):
            candidate = page.locator(selector).first
            try:
                if await candidate.count() > 0:
                    await candidate.click(timeout=5_000)
                    await page.wait_for_timeout(1500)
                    return
            except Exception:
                continue

    @staticmethod
    async def _extract_last_answer(page) -> str:  # pragma: no cover - browser integration
        candidates = page.locator("[data-message-author-role='assistant']")
        count = await candidates.count()
        if count == 0:
            body = await page.locator("body").inner_text(timeout=5000)
            return body[-4000:]
        return (await candidates.nth(count - 1).inner_text()).strip()


def build_backend(settings: Settings) -> Backend:
    if settings.backend == "browser":
        return BrowserBackend(settings)
    return MockBackend(settings)
