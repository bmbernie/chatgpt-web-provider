from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import Settings
from .models import ChatMessage, CompletionResult
from .tool_bridge import (
    parse_tool_calls,
    render_tool_catalog,
    tool_catalog_fingerprint,
)


logger = logging.getLogger("uvicorn.error")


class ChatGPTUIRateLimitError(RuntimeError):
    """ChatGPT web UI is temporarily blocking interaction."""

    def __init__(
        self,
        *,
        phase: str,
    ):
        self.phase = phase

        super().__init__(
            "ChatGPT UI is temporarily rate limited"
        )


@dataclass(slots=True)
class _BrowserSession:
    session_id: str
    model: str
    level: str
    conversation_policy: str = "regular"
    page: Any | None = None
    state: str = "initializing"
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # Logical OpenAI-side conversation represented by this browser
    # conversation. Affinity requests use this to avoid replaying
    # history that ChatGPT already has in the page.
    logical_transcript: tuple[str, ...] = ()
    last_request: tuple[str, ...] | None = None
    last_result: CompletionResult | None = None
    tool_catalog_fingerprint: str | None = None


class Backend(ABC):
    def __init__(self, settings: Settings):
        self.settings = settings

    @abstractmethod
    async def complete(self, messages: list[ChatMessage], model: str | None = None, new_session: bool = False, level: str | None = None) -> CompletionResult:
        raise NotImplementedError

    async def health(self) -> dict:
        return {"ok": True, "backend": self.settings.backend}


    async def create_session(
        self,
        session_id: str,
        model: str,
        level: str,
        conversation_policy: str = "regular",
    ) -> dict:
        raise NotImplementedError("backend does not support provider sessions")

    async def complete_affinity_session(
        self,
        session_id: str,
        messages: list[ChatMessage],
        *,
        tools: list[dict] | None = None,
        tool_choice=None,
        parallel_tool_calls: bool = True,
    ) -> CompletionResult:
        return await self.complete_session(
            session_id,
            messages,
        )

    async def list_sessions(self) -> list[dict]:
        raise NotImplementedError("backend does not support provider sessions")

    async def get_session(self, session_id: str) -> dict:
        raise NotImplementedError("backend does not support provider sessions")

    async def delete_session(self, session_id: str) -> None:
        raise NotImplementedError("backend does not support provider sessions")

    async def complete_session(
        self,
        session_id: str,
        messages: list[ChatMessage],
    ) -> CompletionResult:
        raise NotImplementedError("backend does not support provider sessions")


class MockBackend(Backend):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self._sessions: dict[str, dict] = {}

    async def create_session(
        self,
        session_id: str,
        model: str,
        level: str,
        conversation_policy: str = "regular",
    ) -> dict:
        if session_id in self._sessions:
            raise ValueError("session already exists")

        record = {
            "session_id": session_id,
            "model": model,
            "level": level,
            "conversation_policy": conversation_policy,
            "state": "ready",
        }
        self._sessions[session_id] = record
        return dict(record)

    async def list_sessions(self) -> list[dict]:
        return [
            dict(self._sessions[key])
            for key in sorted(self._sessions)
        ]

    async def get_session(self, session_id: str) -> dict:
        try:
            return dict(self._sessions[session_id])
        except KeyError:
            raise KeyError(session_id) from None

    async def delete_session(self, session_id: str) -> None:
        if session_id not in self._sessions:
            raise KeyError(session_id)
        del self._sessions[session_id]

    async def complete_session(
        self,
        session_id: str,
        messages: list[ChatMessage],
    ) -> CompletionResult:
        session = await self.get_session(session_id)

        return await self.complete(
            messages,
            model=session["model"],
            level=session["level"],
            new_session=False,
        )

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

        # Legacy /v1/chat/completions page.
        self._lock = asyncio.Lock()

        # Protect one-time browser/context initialization.
        self._context_lock = asyncio.Lock()

        # Protect session registry mutations only.
        self._sessions_lock = asyncio.Lock()
        self._sessions: dict[str, _BrowserSession] = {}

        self._playwright = None
        self._context = None
        self._page = None

    @staticmethod
    async def _raise_if_ui_rate_limited(
        page,
        *,
        phase: str,
    ) -> None:
        try:
            modal = page.locator(
                '[data-testid="modal-conversation-history-rate-limit"]'
            )
            visible = await modal.is_visible()
        except Exception:
            # Detection is advisory. If the page object or DOM shape
            # cannot be inspected, preserve the original browser
            # operation and let it report its own failure.
            return

        if not visible:
            return

        logger.warning(
            "chatgpt_ui_rate_limited phase=%s",
            phase,
        )

        raise ChatGPTUIRateLimitError(
            phase=phase,
        )

    @staticmethod
    def _browser_host_bridge_context(
        messages: list[ChatMessage],
    ) -> str:
        """Disambiguate Hermes host context from the ChatGPT web backend."""

        if not any(
            message.role == "developer"
            for message in messages
        ):
            return ""

        return (
            "SYSTEM:\n"
            "HOST BRIDGE CONTEXT:\n"
            "You are the reasoning backend for a Hermes Agent host. "
            "The DEVELOPER message below contains policy and runtime "
            "context for the Hermes host agent you are driving. "
            "References there to 'you', Hermes Agent, runtime tools, "
            "or tool availability describe the host agent, not the "
            "native ChatGPT web UI.\n"
            "The external tool catalog supplied with this request is "
            "authoritative for host-tool availability. A listed "
            "external tool is available even though it is not a native "
            "ChatGPT UI tool. When the user's request requires a listed "
            "tool, use the external tool-call protocol supplied by the "
            "provider instead of reporting that the tool is unavailable."
        )

    @staticmethod
    def _browser_tool_reminder(
        tools: list[dict] | None,
    ) -> str:
        if not tools:
            return ""

        available_names = {
            function.get("name")
            for tool in tools
            if isinstance(tool, dict)
            and isinstance(
                function := tool.get("function"),
                dict,
            )
            and isinstance(function.get("name"), str)
        }

        gateway_names = {
            "tool_search",
            "tool_describe",
            "tool_call",
        }

        if gateway_names.issubset(available_names):
            return (
                "EXTERNAL TOOL REMINDER:\n"
                "Hermes host tools remain available. "
                "When a requested capability is not directly visible, "
                "use tool_search, tool_describe, and tool_call to "
                "discover and invoke it. Do not report a tool or "
                "server as unavailable before attempting that path."
            )

        return (
            "EXTERNAL TOOL REMINDER:\n"
            "The external tools listed above are available through "
            "the Hermes host. When the user explicitly asks for an "
            "operation provided by one of them, call the matching "
            "tool instead of answering that it is unavailable. "
            "Return normal prose only after the required tool work "
            "has completed or an actual tool call reports failure."
        )

    @staticmethod
    async def _write_composer_text(
        page,
        composer,
        text: str,
    ) -> str:
        """Write prompt text without making large contenteditable fills stall."""

        # locator.fill() is simple and reliable for ordinary prompts.
        if len(text) < 16_384:
            await composer.fill(text)
            return "fill"

        # Large Hermes prompts can cause Playwright's contenteditable fill()
        # action to exceed its action timeout even though Chrome eventually
        # applies the text. Use the browser editing pipeline directly.
        # Large single clipboard pastes can be converted by ChatGPT
        # into a special "Pasted text" attachment. Split the payload
        # into small paste events so the content remains inline in the
        # composer while retaining clipboard-paste performance.
        try:
            chunk_size = 4_096
            chunks = [
                text[offset:offset + chunk_size]
                for offset in range(0, len(text), chunk_size)
            ]

            await page.context.grant_permissions(
                [
                    "clipboard-read",
                    "clipboard-write",
                ],
                origin="https://chatgpt.com",
            )

            phase_started = time.perf_counter()
            await composer.focus()
            await page.keyboard.press("Control+A")
            await page.keyboard.press("Backspace")

            logger.info(
                "browser_composer_chunked_paste_start "
                "chars=%d chunks=%d chunk_size=%d",
                len(text),
                len(chunks),
                chunk_size,
            )

            for index, chunk in enumerate(chunks, start=1):
                chunk_started = time.perf_counter()

                await page.evaluate(
                    """async (value) => {
                        await navigator.clipboard.writeText(value);
                    }""",
                    chunk,
                )

                await page.keyboard.press("Control+V")

                logger.info(
                    "browser_composer_chunk_paste_complete "
                    "chunk=%d/%d chars=%d elapsed_ms=%.1f",
                    index,
                    len(chunks),
                    len(chunk),
                    (time.perf_counter() - chunk_started) * 1000,
                )

            logger.info(
                "browser_composer_chunked_paste_complete "
                "chars=%d chunks=%d elapsed_ms=%.1f",
                len(text),
                len(chunks),
                (time.perf_counter() - phase_started) * 1000,
            )

            return "clipboard_chunked_paste"

        except Exception as exc:
            logger.warning(
                "browser_composer_clipboard_failed "
                "error_type=%s fallback=keyboard_insert_text",
                type(exc).__name__,
            )

            # Preserve the known-working path if clipboard access is
            # unavailable in this Chromium/session configuration.
            await composer.focus()
            await page.keyboard.press("Control+A")
            await page.keyboard.press("Backspace")

            phase_started = time.perf_counter()
            await page.keyboard.insert_text(text)

            logger.info(
                "browser_composer_insert_text_complete "
                "chars=%d elapsed_ms=%.1f fallback=true",
                len(text),
                (time.perf_counter() - phase_started) * 1000,
            )

            return "keyboard_insert_text_fallback"

    async def _ensure_page(self):
        if self._page:
            return self._page

        async with self._context_lock:
            if self._page:
                return self._page

            try:
                from playwright.async_api import async_playwright
            except Exception as exc:  # pragma: no cover - depends on optional browser env
                raise RuntimeError("playwright is not installed") from exc

            Path(self.settings.profile_dir).mkdir(
                parents=True,
                exist_ok=True,
            )

            self._playwright = await async_playwright().start()
            self._context = (
                await self._playwright.chromium.launch_persistent_context(
                    self.settings.profile_dir,
                    headless=self.settings.headless,
                    channel=self.settings.browser_channel,
                    ignore_default_args=(
                        ["--disable-extensions"]
                        if self.settings.enable_extensions
                        else None
                    ),
                    viewport={"width": 1360, "height": 820},
                    args=[
                        "--disable-blink-features=AutomationControlled"
                    ],
                )
            )

            self._page = (
                self._context.pages[0]
                if self._context.pages
                else await self._context.new_page()
            )

            await self._page.goto(
                "https://chatgpt.com/",
                wait_until="domcontentloaded",
                timeout=60_000,
            )

            return self._page

    @staticmethod
    def _public_session(session: _BrowserSession) -> dict:
        return {
            "session_id": session.session_id,
            "model": session.model,
            "level": session.level,
            "conversation_policy": session.conversation_policy,
            "state": session.state,
        }

    async def _get_browser_session(
        self,
        session_id: str,
    ) -> _BrowserSession:
        async with self._sessions_lock:
            try:
                return self._sessions[session_id]
            except KeyError:
                raise KeyError(session_id) from None

    async def create_session(
        self,
        session_id: str,
        model: str,
        level: str,
        conversation_policy: str = "regular",
    ) -> dict:
        session = _BrowserSession(
            session_id=session_id,
            model=model,
            level=level,
            conversation_policy=conversation_policy,
        )

        # Reserve identity before slow browser initialization.
        async with self._sessions_lock:
            if session_id in self._sessions:
                raise ValueError("session already exists")
            self._sessions[session_id] = session

        logger.info(
            "browser_session_create_start session_id=%s "
            "model=%s level=%s",
            session_id,
            model,
            level,
        )

        started = time.perf_counter()

        try:
            session.page = await self._create_pinned_session_page(
                model,
                level,
                conversation_policy,
            )
            session.state = "ready"

            logger.info(
                "browser_session_create_complete session_id=%s "
                "model=%s level=%s total_ms=%.1f",
                session_id,
                model,
                level,
                (time.perf_counter() - started) * 1000,
            )

            return self._public_session(session)

        except Exception:
            async with self._sessions_lock:
                self._sessions.pop(session_id, None)

            logger.exception(
                "browser_session_create_failed session_id=%s "
                "model=%s level=%s",
                session_id,
                model,
                level,
            )
            raise

    async def list_sessions(self) -> list[dict]:
        async with self._sessions_lock:
            return [
                self._public_session(self._sessions[key])
                for key in sorted(self._sessions)
            ]

    async def get_session(self, session_id: str) -> dict:
        session = await self._get_browser_session(session_id)
        return self._public_session(session)

    async def delete_session(self, session_id: str) -> None:
        session = await self._get_browser_session(session_id)

        async with session.lock:
            session.state = "closing"

            if session.page is not None:
                await self._close_pinned_session_page(
                    session.page
                )

            async with self._sessions_lock:
                current = self._sessions.get(session_id)
                if current is session:
                    del self._sessions[session_id]

        logger.info(
            "browser_session_deleted session_id=%s",
            session_id,
        )

    async def complete_session(
        self,
        session_id: str,
        messages: list[ChatMessage],
    ) -> CompletionResult:
        session = await self._get_browser_session(session_id)

        # Serialize only within this conversation.
        async with session.lock:
            if session.state != "ready" or session.page is None:
                raise RuntimeError(
                    f"browser session is not ready: {session_id}"
                )

            session.state = "busy"

            try:
                return await self._complete_pinned_session(
                    session,
                    messages,
                )
            finally:
                if session.state == "busy":
                    session.state = "ready"

    async def complete_affinity_session(
        self,
        session_id: str,
        messages: list[ChatMessage],
        *,
        tools: list[dict] | None = None,
        tool_choice=None,
        parallel_tool_calls: bool = True,
    ) -> CompletionResult:
        """Complete an OpenAI-style full-history request on a pinned page.

        The client supplies its complete logical transcript on every call,
        while the browser page already retains prior ChatGPT history.
        Submit only the new suffix when the histories agree.

        If the client history is rewritten or compacted, start a fresh
        ChatGPT conversation and seed it with the new transcript.
        """
        session = await self._get_browser_session(
            session_id
        )

        async with session.lock:
            if (
                session.state != "ready"
                or session.page is None
            ):
                raise RuntimeError(
                    f"browser session is not ready: "
                    f"{session_id}"
                )

            incoming = tuple(
                self._message_signature(message)
                for message in messages
            )

            requested_tool_fingerprint = (
                tool_catalog_fingerprint(
                    tools,
                    tool_choice=tool_choice,
                    parallel_tool_calls=parallel_tool_calls,
                )
                if tools
                else None
            )

            # A transport retry of the exact same OpenAI request must
            # not submit the user message to ChatGPT a second time.
            if (
                session.last_request == incoming
                and session.last_result is not None
                and session.tool_catalog_fingerprint
                == requested_tool_fingerprint
            ):
                logger.info(
                    "browser_affinity_replay "
                    "session_id=%s messages=%d",
                    session_id,
                    len(messages),
                )
                return session.last_result

            recorded = session.logical_transcript

            extends_recorded = (
                len(incoming) >= len(recorded)
                and incoming[:len(recorded)] == recorded
            )

            reset = False

            if extends_recorded:
                delta = messages[len(recorded):]

            else:
                # Hermes may compact or otherwise rewrite its logical
                # transcript. The old browser conversation can no
                # longer represent that context exactly, so establish
                # a fresh ChatGPT conversation and seed it once with
                # the incoming transcript.
                reset = True

                await self._start_policy_session(
                    session.page,
                    session.conversation_policy,
                )

                await self._apply_preferences(
                    session.page,
                    session.model,
                    session.level,
                )

                session.logical_transcript = ()
                session.last_request = None
                session.last_result = None
                session.tool_catalog_fingerprint = None

                delta = messages

                logger.info(
                    "browser_affinity_reset "
                    "session_id=%s reason=transcript_discontinuity "
                    "incoming_messages=%d",
                    session_id,
                    len(messages),
                )

            tool_catalog_prompt = None

            if (
                tools
                and session.tool_catalog_fingerprint
                != requested_tool_fingerprint
            ):
                catalog_tools = tools

                logger.info(
                    "browser_tool_catalog "
                    "available_tools=%d presented_tools=%d",
                    len(tools),
                    len(catalog_tools),
                )

                tool_catalog_prompt = render_tool_catalog(
                    catalog_tools,
                    tool_choice=tool_choice,
                    parallel_tool_calls=parallel_tool_calls,
                )

            if not delta and tool_catalog_prompt is None:
                raise RuntimeError(
                    "affinity request contained no new messages"
                )

            logger.info(
                "browser_affinity_request "
                "session_id=%s incoming_messages=%d "
                "delta_messages=%d reset=%s",
                session_id,
                len(messages),
                len(delta),
                str(reset).lower(),
            )

            session.state = "busy"

            try:
                if tools:
                    result = await self._complete_pinned_session(
                        session,
                        delta,
                        tools=tools,
                        tool_choice=tool_choice,
                        parallel_tool_calls=parallel_tool_calls,
                        tool_catalog_prompt=tool_catalog_prompt,
                    )
                else:
                    # Preserve compatibility with existing backend
                    # test doubles that implement the original
                    # two-argument helper.
                    result = await self._complete_pinned_session(
                        session,
                        delta,
                    )

                assistant_message = ChatMessage(
                    role="assistant",
                    content=(
                        None
                        if result.tool_calls
                        else result.text
                    ),
                    tool_calls=(
                        result.tool_calls
                        if result.tool_calls
                        else None
                    ),
                )

                session.logical_transcript = (
                    incoming
                    + (
                        self._message_signature(
                            assistant_message
                        ),
                    )
                )

                session.last_request = incoming
                session.last_result = result

                if tools:
                    session.tool_catalog_fingerprint = (
                        requested_tool_fingerprint
                    )
                else:
                    session.tool_catalog_fingerprint = None

                return result

            finally:
                if session.state == "busy":
                    session.state = "ready"

    async def _start_policy_session(
        self,
        page,
        conversation_policy: str,
    ) -> None:
        """Start a fresh ChatGPT conversation under a pinned policy."""
        await self._start_new_session(page)

        await self._raise_if_ui_rate_limited(
            page,
            phase="session_start",
        )

        if conversation_policy == "regular":
            return

        if conversation_policy not in {
            "temporary_personalized",
            "temporary_unpersonalized",
        }:
            raise RuntimeError(
                "unsupported conversation policy: "
                f"{conversation_policy}"
            )

        await self._enable_temporary_chat(page)

        desired_personalization = (
            "Personalized"
            if conversation_policy
            == "temporary_personalized"
            else "Unpersonalized"
        )

        await self._set_temporary_personalization(
            page,
            desired_personalization,
        )

    @staticmethod
    async def _enable_temporary_chat(
        page,
    ) -> None:
        """Enable Temporary Chat and verify the UI entered that mode."""
        enabled = page.locator(
            'button[aria-label="Turn off temporary chat"]'
        )

        if await enabled.is_visible():
            return

        toggle = page.locator(
            'button[aria-label="Temporary chat"]'
        )

        await toggle.wait_for(
            state="visible",
            timeout=30_000,
        )
        await BrowserBackend._raise_if_ui_rate_limited(
            page,
            phase="temporary_chat_toggle",
        )

        try:
            await toggle.click()
        except Exception:
            await BrowserBackend._raise_if_ui_rate_limited(
                page,
                phase="temporary_chat_toggle",
            )
            raise

        await enabled.wait_for(
            state="visible",
            timeout=10_000,
        )

    @staticmethod
    async def _set_temporary_personalization(
        page,
        desired: str,
    ) -> None:
        """Pin Temporary Chat to Personalized or Unpersonalized."""
        desired_button = page.locator(
            f'button[aria-label="{desired}"]'
        )

        if await desired_button.is_visible():
            return

        other = (
            "Unpersonalized"
            if desired == "Personalized"
            else "Personalized"
        )

        launcher = page.locator(
            f'button[aria-label="{other}"]'
        )

        await launcher.wait_for(
            state="visible",
            timeout=10_000,
        )
        await BrowserBackend._raise_if_ui_rate_limited(
            page,
            phase="personalization_menu",
        )

        try:
            await launcher.click()
        except Exception:
            await BrowserBackend._raise_if_ui_rate_limited(
                page,
                phase="personalization_menu",
            )
            raise

        option = page.locator(
            '[role="menuitemradio"]'
        ).filter(
            has=page.get_by_text(
                desired,
                exact=True,
            )
        )

        await option.wait_for(
            state="visible",
            timeout=10_000,
        )
        await option.click()

        await desired_button.wait_for(
            state="visible",
            timeout=10_000,
        )

    async def _create_pinned_session_page(
        self,
        model: str,
        level: str,
        conversation_policy: str = "regular",
    ):
        # Ensure the persistent authenticated browser exists.
        await self._ensure_page()

        if self._context is None:
            raise RuntimeError("browser context is unavailable")

        page = await self._context.new_page()

        try:
            await page.goto(
                "https://chatgpt.com/",
                wait_until="domcontentloaded",
                timeout=60_000,
            )

            if "log in" in (await page.title()).lower():
                raise RuntimeError(
                    "ChatGPT browser profile is not logged in"
                )

            # This page/conversation now belongs to this provider session.
            await self._start_policy_session(
                page,
                conversation_policy,
            )

            # Wait for the hydrated reasoning UI and pin this
            # session's configured reasoning tier.
            await self._apply_preferences(
                page,
                model,
                level,
            )

            try:
                await page.keyboard.press("Escape")
            except Exception:
                pass

            return page

        except Exception:
            try:
                await page.close()
            except Exception:
                pass
            raise

    @staticmethod
    async def _close_pinned_session_page(page) -> None:
        await page.close()

    async def _complete_pinned_session(
        self,
        session: _BrowserSession,
        messages: list[ChatMessage],
        *,
        tools: list[dict] | None = None,
        tool_choice=None,
        parallel_tool_calls: bool = True,
        tool_catalog_prompt: str | None = None,
    ) -> CompletionResult:
        page = session.page

        if page is None:
            raise RuntimeError(
                f"browser session has no page: "
                f"{session.session_id}"
            )

        tool_catalog_text = (
            "SYSTEM:\n" + tool_catalog_prompt
            if tool_catalog_prompt
            else ""
        )

        host_bridge_text = (
            self._browser_host_bridge_context(messages)
        )

        message_text = (
            self._render_prompt(messages)
            if messages
            else ""
        )

        tool_reminder_text = (
            self._browser_tool_reminder(tools)
        )

        # Keep the current transcript first. Put the lightweight
        # reminder next and the complete external-tool catalog last,
        # so its exact tool-call serialization protocol is closest
        # to generation on catalog-bearing turns.
        prompt_parts = [
            part
            for part in (
                host_bridge_text,
                message_text,
                tool_reminder_text,
                tool_catalog_text,
            )
            if part
        ]

        prompt = "\n\n".join(prompt_parts)

        logger.info(
            "browser_prompt_components "
            "host_bridge_chars=%d "
            "tool_catalog_chars=%d "
            "message_chars=%d "
            "tool_reminder_chars=%d "
            "total_chars=%d",
            len(host_bridge_text),
            len(tool_catalog_text),
            len(message_text),
            len(tool_reminder_text),
            len(prompt),
        )
        started = time.perf_counter()

        logger.info(
            "browser_session_request_start session_id=%s "
            "model=%s level=%s messages=%d prompt_chars=%d",
            session.session_id,
            session.model,
            session.level,
            len(messages),
            len(prompt),
        )

        if "log in" in (await page.title()).lower():
            raise RuntimeError(
                "ChatGPT browser profile is not logged in"
            )

        await self._raise_if_ui_rate_limited(
            page,
            phase="composer",
        )

        composer = page.locator(
            "#prompt-textarea, div[contenteditable='true']"
        ).last

        await composer.wait_for(timeout=30_000)

        await self._raise_if_ui_rate_limited(
            page,
            phase="composer_fill",
        )

        input_method = (
            "clipboard_chunked_paste"
            if len(prompt) >= 16_384
            else "fill"
        )

        logger.info(
            "browser_composer_input_start "
            "method=%s prompt_chars=%d",
            input_method,
            len(prompt),
        )

        try:
            input_method = await self._write_composer_text(
                page,
                composer,
                prompt,
            )
        except Exception as exc:
            await self._raise_if_ui_rate_limited(
                page,
                phase="composer_input",
            )

            raise RuntimeError(
                f"ChatGPT composer input failed "
                f"({type(exc).__name__})"
            ) from None

        logger.info(
            "browser_composer_input_complete "
            "method=%s prompt_chars=%d",
            input_method,
            len(prompt),
        )

        await self._submit_prompt(page, composer)
        await self._wait_until_idle(page)
        response_text = await self._extract_last_answer(page)

        logger.info(
            "browser_session_request_complete session_id=%s "
            "model=%s level=%s messages=%d prompt_chars=%d "
            "total_ms=%.1f",
            session.session_id,
            session.model,
            session.level,
            len(messages),
            len(prompt),
            (time.perf_counter() - started) * 1000,
        )

        tool_calls = (
            parse_tool_calls(
                response_text,
                tools,
                parallel_tool_calls=parallel_tool_calls,
            )
            if tools
            else None
        )

        logger.info(
            "browser_external_tool_result "
            "session_id=%s tool_calls=%d names=%s",
            session.session_id,
            len(tool_calls or []),
            ",".join(
                call.function.name
                for call in (tool_calls or [])
            ) or "-",
        )

        return CompletionResult(
            text=(
                ""
                if tool_calls
                else response_text
            ),
            model=session.model,
            level=session.level,
            tool_calls=tool_calls or [],
        )

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

    async def _apply_preferences(
        self,
        page,
        model: str,
        level: str | None,
    ) -> None:  # pragma: no cover - browser integration
        """Pin model and reasoning level through the composer picker."""

        if model != self.settings.model_id:
            raise RuntimeError(
                "browser pinned sessions currently support only the "
                f"configured model: requested={model} "
                f"configured={self.settings.model_id}"
            )

        await self._select_model(page, model)

        if level:
            await self._select_reasoning_level(page, level)

    @staticmethod
    def _normalize_preference_text(value: str | None) -> str:
        return " ".join((value or "").split()).casefold()

    async def _wait_for_reasoning_control(
        self,
        page,
        timeout_seconds: float = 45.0,
    ):
        """Wait for the reasoning control inside the composer shell."""
        known_labels = {
            self._normalize_preference_text(
                self.settings.level_label(level)
            )
            for level in self.settings.available_levels
        }

        composer = page.locator(
            "#prompt-textarea, div[contenteditable='true']"
        ).last

        await composer.wait_for(timeout=30_000)

        deadline = time.monotonic() + timeout_seconds
        started = time.monotonic()

        while time.monotonic() < deadline:
            container = composer

            for depth in range(1, 9):
                try:
                    container = container.locator("xpath=..")
                    controls = container.locator(
                        "[aria-haspopup='menu']"
                    )
                    count = min(await controls.count(), 20)
                except Exception:
                    break

                for index in range(count):
                    control = controls.nth(index)

                    try:
                        if not await control.is_visible(timeout=100):
                            continue

                        raw_text = (
                            await control.inner_text(timeout=200)
                        ).strip()
                    except Exception:
                        continue

                    normalized = self._normalize_preference_text(
                        raw_text
                    )

                    if normalized not in known_labels:
                        continue

                    logger.info(
                        "reasoning_control_ready current=%r "
                        "composer_depth=%d wait_ms=%.1f",
                        raw_text,
                        depth,
                        (time.monotonic() - started) * 1000,
                    )

                    return control, raw_text

            await page.wait_for_timeout(200)

        raise RuntimeError(
            "ChatGPT reasoning picker did not become ready "
            f"within {timeout_seconds:.0f} seconds"
        )

    async def _open_intelligence_picker(self, page):
        control, current_level = (
            await self._wait_for_reasoning_control(page)
        )

        await control.click(timeout=3000)

        content = page.locator(
            "[data-testid='composer-intelligence-picker-content']"
        ).last

        await content.wait_for(
            state="visible",
            timeout=5000,
        )

        return control, current_level, content

    async def _find_model_option(
        self,
        content,
        target_label: str,
    ):
        advanced = content.locator(
            "[data-testid='composer-model-picker-slider-advanced-view']"
        ).last

        await advanced.wait_for(
            state="visible",
            timeout=5000,
        )

        options = advanced.locator(
            "[role='menuitemradio']"
        )

        count = min(await options.count(), 20)
        target = self._normalize_preference_text(
            target_label
        )

        for index in range(count):
            option = options.nth(index)

            try:
                if not await option.is_visible(timeout=150):
                    continue

                raw_text = (
                    await option.inner_text(timeout=250)
                ).strip()
            except Exception:
                continue

            whole = self._normalize_preference_text(
                raw_text
            )

            first_line = ""
            for line in raw_text.splitlines():
                if line.strip():
                    first_line = (
                        self._normalize_preference_text(line)
                    )
                    break

            if whole == target or first_line == target:
                return option

        return None

    async def _activate_model_view(
        self,
        page,
        content,
    ):
        """Switch the intelligence picker from reasoning to model view."""
        simple = content.locator(
            "[data-testid='composer-model-picker-slider-simple-view']"
        ).last

        advanced = content.locator(
            "[data-testid='composer-model-picker-slider-advanced-view']"
        ).last

        # The advanced panel exists in the DOM even when the simple panel
        # overlays it. data-active tells us which view actually owns input.
        try:
            if await advanced.get_attribute("data-active") == "true":
                return advanced
        except Exception:
            pass

        select_model = content.locator(
            "[role='menuitem'][aria-label='Select model']"
        ).last

        await select_model.wait_for(
            state="visible",
            timeout=5000,
        )

        logger.info("model_view_activate_start")

        await select_model.click(timeout=3000)

        deadline = time.monotonic() + 5.0

        while time.monotonic() < deadline:
            try:
                advanced_active = (
                    await advanced.get_attribute("data-active")
                )
                simple_active = (
                    await simple.get_attribute("data-active")
                )

                if (
                    advanced_active == "true"
                    or simple_active != "true"
                ):
                    # Give the transition a short period to settle before
                    # asking Playwright to perform a pointer action.
                    await page.wait_for_timeout(200)

                    logger.info(
                        "model_view_activate_complete "
                        "advanced_active=%r simple_active=%r",
                        advanced_active,
                        simple_active,
                    )

                    return advanced

            except Exception:
                pass

            await page.wait_for_timeout(100)

        raise RuntimeError(
            "ChatGPT model picker did not switch to advanced view"
        )

    async def _select_model(
        self,
        page,
        model: str,
    ) -> None:
        """Select the model and leave the combined picker on simple view."""
        target_label = self.settings.model_label(model)

        _, _, content = await self._open_intelligence_picker(
            page
        )

        await self._activate_model_view(
            page,
            content,
        )

        option = await self._find_model_option(
            content,
            target_label,
        )

        if option is None:
            raise RuntimeError(
                f"could not find model option: {target_label}"
            )

        async def option_checked() -> bool:
            try:
                return (
                    await option.get_attribute("aria-checked")
                    == "true"
                    or await option.get_attribute("data-state")
                    == "checked"
                )
            except Exception:
                return False

        if await option_checked():
            logger.info(
                "model_already_selected model=%s label=%r",
                model,
                target_label,
            )
        else:
            logger.info(
                "model_select_start model=%s label=%r",
                model,
                target_label,
            )

            await option.click(timeout=5000)

            deadline = time.monotonic() + 5.0

            while time.monotonic() < deadline:
                if await option_checked():
                    logger.info(
                        "model_select_complete model=%s label=%r",
                        model,
                        target_label,
                    )
                    break

                await page.wait_for_timeout(100)
            else:
                raise RuntimeError(
                    f"model selection was not verified: "
                    f"{target_label}"
                )

        simple = content.locator(
            "[data-testid='composer-model-picker-slider-simple-view']"
        ).last

        # Selecting a model often transitions back to simple view itself.
        # If it did not, one Escape moves advanced -> simple. Do NOT close
        # the picker; reasoning configuration follows immediately.
        deadline = time.monotonic() + 5.0
        escape_sent = False

        while time.monotonic() < deadline:
            try:
                if (
                    await simple.get_attribute("data-active")
                    == "true"
                ):
                    logger.info(
                        "model_to_reasoning_handoff_complete"
                    )
                    return
            except Exception:
                pass

            if not escape_sent:
                try:
                    await page.keyboard.press("Escape")
                    escape_sent = True
                except Exception:
                    pass

            await page.wait_for_timeout(100)

        raise RuntimeError(
            "model selection succeeded but reasoning view "
            "did not become active"
        )

    async def _select_reasoning_level(
        self,
        page,
        level: str,
    ) -> None:
        """Adjust thinking effort in the already-open combined picker."""
        target_label = self.settings.level_label(level)

        picker = page.locator(
            "[data-testid='composer-intelligence-picker-content']"
        ).last

        # Normal path: _select_model() deliberately left this open.
        try:
            picker_open = await picker.is_visible(timeout=500)
        except Exception:
            picker_open = False

        # Defensive fallback for callers that invoke reasoning directly.
        if not picker_open:
            _, _, picker = await self._open_intelligence_picker(
                page
            )

        simple = picker.locator(
            "[data-testid='composer-model-picker-slider-simple-view']"
        ).last

        await simple.wait_for(
            state="visible",
            timeout=5000,
        )

        # Ensure the simple reasoning panel, not advanced model panel,
        # actually owns interaction.
        if await simple.get_attribute("data-active") != "true":
            try:
                await page.keyboard.press("Escape")
            except Exception:
                pass

            deadline = time.monotonic() + 3.0

            while time.monotonic() < deadline:
                if (
                    await simple.get_attribute("data-active")
                    == "true"
                ):
                    break

                await page.wait_for_timeout(100)
            else:
                raise RuntimeError(
                    "reasoning simple view did not become active"
                )

        def level_from_text(value: str):
            normalized = self._normalize_preference_text(
                value
            )

            # Longest first: Extra High must win over High.
            candidates = sorted(
                (
                    (
                        candidate_level,
                        self.settings.level_label(
                            candidate_level
                        ),
                    )
                    for candidate_level
                    in self.settings.available_levels
                ),
                key=lambda item: len(item[1]),
                reverse=True,
            )

            for candidate_level, candidate_label in candidates:
                label_norm = (
                    self._normalize_preference_text(
                        candidate_label
                    )
                )

                if (
                    normalized == label_norm
                    or normalized.startswith(
                        label_norm + ","
                    )
                    or normalized.startswith(
                        label_norm + " "
                    )
                ):
                    return (
                        candidate_level,
                        candidate_label,
                    )

            return None

        simple_text = (
            await simple.inner_text(timeout=1000)
        ).strip()

        current = level_from_text(simple_text)

        if current is None:
            raise RuntimeError(
                "could not determine current reasoning level: "
                f"{simple_text!r}"
            )

        current_level, current_label = current

        logger.info(
            "reasoning_level_observed "
            "level=%s label=%r text=%r",
            current_level,
            current_label,
            simple_text,
        )

        if current_level == level:
            logger.info(
                "reasoning_level_already_selected "
                "level=%s label=%r",
                level,
                current_label,
            )

            await page.keyboard.press("Escape")
            return

        if current_level == "xhigh" and level == "high":
            key = "ArrowLeft"
        elif current_level == "high" and level == "xhigh":
            key = "ArrowRight"
        else:
            raise RuntimeError(
                "unsupported reasoning transition: "
                f"current={current_level!r} "
                f"target={level!r}"
            )

        logger.info(
            "reasoning_level_select_start "
            "current=%r target=%r key=%s",
            current_label,
            target_label,
            key,
        )

        # Prefer the element that exposes actual slider semantics.
        slider = None
        slider_selector = None

        for selector in (
            "[role='slider']",
            "[aria-valuenow]",
            "[aria-valuetext]",
            "[tabindex='0']:not([aria-label='Select model'])",
        ):
            try:
                candidates = simple.locator(selector)
                count = min(await candidates.count(), 20)

                for index in range(count):
                    candidate = candidates.nth(index)

                    if await candidate.is_visible(timeout=100):
                        slider = candidate
                        slider_selector = selector
                        break
            except Exception:
                continue

            if slider is not None:
                break

        sent = False

        if slider is not None:
            try:
                logger.info(
                    "reasoning_slider_target selector=%r "
                    "role=%r tabindex=%r now=%r "
                    "valuetext=%r",
                    slider_selector,
                    await slider.get_attribute("role"),
                    await slider.get_attribute("tabindex"),
                    await slider.get_attribute(
                        "aria-valuenow"
                    ),
                    await slider.get_attribute(
                        "aria-valuetext"
                    ),
                )

                await slider.focus()
                await slider.press(key)
                sent = True
            except Exception as exc:
                logger.info(
                    "reasoning_slider_direct_key_failed "
                    "error_type=%s",
                    type(exc).__name__,
                )

        if not sent:
            # Fallback: focus the visible thinking-effort panel itself.
            box = await simple.bounding_box()

            if box:
                await simple.click(
                    position={
                        "x": box["width"] / 2,
                        "y": box["height"] / 2,
                    },
                    timeout=3000,
                )

            await page.keyboard.press(key)

            logger.info(
                "reasoning_slider_key_sent_fallback "
                "key=%s",
                key,
            )

        deadline = time.monotonic() + 5.0
        last_text = simple_text

        while time.monotonic() < deadline:
            try:
                last_text = (
                    await simple.inner_text(timeout=500)
                ).strip()

                observed = level_from_text(last_text)

                if observed and observed[0] == level:
                    logger.info(
                        "reasoning_level_select_complete "
                        "level=%s label=%r text=%r",
                        level,
                        observed[1],
                        last_text,
                    )

                    try:
                        await page.keyboard.press("Escape")
                    except Exception:
                        pass

                    return
            except Exception:
                pass

            await page.wait_for_timeout(100)

        raise RuntimeError(
            "reasoning-level slider did not reach target: "
            f"expected={target_label!r} "
            f"observed_text={last_text!r}"
        )

    @staticmethod
    def _message_signature(
        message: ChatMessage,
    ) -> str:
        return json.dumps(
            message.model_dump(
                mode="json",
                exclude_none=True,
            ),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    @staticmethod
    def _render_prompt(
        messages: list[ChatMessage],
    ) -> str:
        rendered = []

        for message in messages:
            if (
                message.role == "assistant"
                and message.tool_calls
            ):
                calls = [
                    call.model_dump(
                        mode="json",
                        exclude_none=True,
                    )
                    for call in message.tool_calls
                ]

                rendered.append(
                    "ASSISTANT EXTERNAL TOOL CALLS:\n"
                    + json.dumps(
                        calls,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    )
                )

                continue

            if message.role == "tool":
                metadata = []

                if message.tool_call_id:
                    metadata.append(
                        f"id={message.tool_call_id}"
                    )

                if message.name:
                    metadata.append(
                        f"name={message.name}"
                    )

                suffix = (
                    " " + " ".join(metadata)
                    if metadata
                    else ""
                )

                rendered.append(
                    "EXTERNAL TOOL RESULT"
                    + suffix
                    + ":\n"
                    + message.text()
                )

                continue

            rendered.append(
                f"{message.role.upper()}: "
                f"{message.text()}"
            )

        return "\n\n".join(rendered)

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
