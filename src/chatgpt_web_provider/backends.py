from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .browser_prompts import (
    TOOL_CATALOG_SYSTEM_PREFIX,
    render_host_bridge_context,
    render_tool_reminder,
    render_transcript,
)
from .config import Settings
from .chatgpt_ui import (
    ASSISTANT_MESSAGES,
    BODY,
    COMPOSER,
    CONTROL_ANCESTOR_MAX_DEPTH,
    CONTROL_TEXT_PROBE_MS,
    CONTROL_VISIBLE_PROBE_MS,
    INTELLIGENCE_PICKER_CONTENT,
    MODEL_OPTIONS,
    MODEL_PICKER_ADVANCED_VIEW,
    MODEL_PICKER_SIMPLE_VIEW,
    NEW_CHAT_SELECTORS,
    OPTION_TEXT_PROBE_MS,
    OPTION_VISIBLE_PROBE_MS,
    PARENT_XPATH,
    PERSONALIZATION_OPTION,
    PICKER_OPEN_PROBE_MS,
    PICKER_STATE_TEXT_TIMEOUT_MS,
    PICKER_TEXT_TIMEOUT_MS,
    RATE_LIMIT_MODAL,
    REASONING_CONTROLS,
    REASONING_SLIDER_SELECTORS,
    SELECT_MODEL_MENUITEM,
    SEND_BUTTON_CANDIDATE_LIMIT,
    SEND_BUTTON_PROBE_MS,
    SEND_BUTTON_SELECTORS,
    STATE_POLL_MS,
    STOP_BUTTON,
    SUBMIT_COMPOSER_PROBE_MS,
    SUBMIT_POLL_MS,
    SUBMIT_STOP_PROBE_MS,
    TEMPORARY_CHAT_ENABLED,
    TEMPORARY_CHAT_TOGGLE,
    TRANSITION_SETTLE_MS,
    UI_CANDIDATE_LIMIT,
    aria_label_button,
)
from .models import ChatMessage, CompletionResult
from .tool_bridge import (
    parse_tool_calls,
    render_tool_catalog,
    tool_catalog_fingerprint,
)


logger = logging.getLogger("uvicorn.error")

RESPONSE_BODY_FALLBACK_CHARS = 4_000


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


class ChatGPTBrowserLoginRequiredError(RuntimeError):
    """The persistent ChatGPT browser profile requires authentication."""

    def __init__(
        self,
        *,
        phase: str,
    ):
        self.phase = phase

        super().__init__(
            "ChatGPT browser profile is not logged in"
        )


class ChatGPTBrowserOperationError(RuntimeError):
    """A required ChatGPT Web UI operation could not be completed."""

    def __init__(
        self,
        *,
        phase: str,
        operation: str,
    ):
        self.phase = phase
        self.operation = operation

        super().__init__(
            "ChatGPT browser operation failed"
        )


class ChatGPTBrowserStateError(RuntimeError):
    """Browser/UI state could not be inspected reliably."""

    def __init__(
        self,
        *,
        phase: str,
        operation: str,
    ):
        self.phase = phase
        self.operation = operation

        super().__init__(
            "ChatGPT browser state could not be inspected reliably"
        )


class ChatGPTGenerationTimeoutError(RuntimeError):
    """ChatGPT generation exceeded the configured completion deadline."""

    def __init__(
        self,
        *,
        timeout_seconds: int,
    ):
        self.timeout_seconds = timeout_seconds

        super().__init__(
            "ChatGPT generation did not finish before "
            "the configured timeout"
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


@dataclass(frozen=True, slots=True)
class _AffinityRequestPlan:
    """Deterministic decision for one affinity completion request."""

    incoming: tuple[str, ...]
    requested_tool_fingerprint: str | None
    replay: bool
    reset: bool
    delta_start: int
    inject_tool_catalog: bool


class Backend(ABC):
    def __init__(self, settings: Settings):
        self.settings = settings

    @abstractmethod
    async def complete(self, messages: list[ChatMessage], model: str | None = None, new_session: bool = False, level: str | None = None) -> CompletionResult:
        raise NotImplementedError

    async def health(self) -> dict:
        return {"ok": True, "backend": self.settings.backend}

    async def complete_with_tools(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        new_session: bool = False,
        level: str | None = None,
        tools: list[dict] | None = None,
        tool_choice=None,
        parallel_tool_calls: bool = True,
    ) -> CompletionResult:
        """Complete a non-affinity request with external tools.

        Backends that support browser-side tool bridging should override
        this method. Keep complete() unchanged for compatibility with
        existing backend implementations and test doubles.
        """
        if not tools:
            return await self.complete(
                messages,
                model=model,
                new_session=new_session,
                level=level,
            )

        raise NotImplementedError(
            "backend does not support non-affinity external tools"
        )


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


@dataclass(slots=True)
class _BrowserTurnResult:
    text: str
    input_method: str
    composer_wait_ms: float
    input_ms: float
    submit_ms: float
    generation_ms: float
    extraction_ms: float


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
    async def _raise_if_browser_login_required(
        page,
        *,
        phase: str,
    ) -> None:
        title = await page.title()

        if "log in" not in title.lower():
            return

        logger.warning(
            "chatgpt_browser_login_required phase=%s",
            phase,
        )

        raise ChatGPTBrowserLoginRequiredError(
            phase=phase,
        )

    @staticmethod
    async def _raise_if_ui_rate_limited(
        page,
        *,
        phase: str,
    ) -> None:
        try:
            modal = page.locator(
                RATE_LIMIT_MODAL
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
    async def _write_composer_text(
        page,
        composer,
        text: str,
        *,
        inline_fill_max_chars: int,
        clipboard_chunk_size: int,
        clipboard_origin: str,
    ) -> str:
        """Write prompt text without making large contenteditable fills stall."""

        # locator.fill() is simple and reliable for ordinary prompts.
        if len(text) < inline_fill_max_chars:
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
            chunks = [
                text[offset:offset + clipboard_chunk_size]
                for offset in range(
                    0,
                    len(text),
                    clipboard_chunk_size,
                )
            ]

            await page.context.grant_permissions(
                [
                    "clipboard-read",
                    "clipboard-write",
                ],
                origin=clipboard_origin,
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
                clipboard_chunk_size,
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
                    viewport={
                        "width": self.settings.viewport_width,
                        "height": self.settings.viewport_height,
                    },
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
                self.settings.chatgpt_base_url,
                wait_until="domcontentloaded",
                timeout=self.settings.navigation_timeout_ms,
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

    def _plan_affinity_request(
        self,
        session: _BrowserSession,
        messages: list[ChatMessage],
        *,
        tools: list[dict] | None,
        tool_choice,
        parallel_tool_calls: bool,
    ) -> _AffinityRequestPlan:
        """Decide affinity behavior without mutating session or browser state."""
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

        replay = (
            session.last_request == incoming
            and session.last_result is not None
            and session.tool_catalog_fingerprint
            == requested_tool_fingerprint
        )

        recorded = session.logical_transcript

        extends_recorded = (
            len(incoming) >= len(recorded)
            and incoming[:len(recorded)] == recorded
        )

        reset = not extends_recorded

        delta_start = (
            len(recorded)
            if extends_recorded
            else 0
        )

        # A reset establishes a fresh browser conversation, so an
        # available tool catalog must be seeded again even when its
        # fingerprint matches the prior conversation.
        inject_tool_catalog = bool(tools) and (
            reset
            or session.tool_catalog_fingerprint
            != requested_tool_fingerprint
        )

        return _AffinityRequestPlan(
            incoming=incoming,
            requested_tool_fingerprint=(
                requested_tool_fingerprint
            ),
            replay=replay,
            reset=reset,
            delta_start=delta_start,
            inject_tool_catalog=inject_tool_catalog,
        )

    async def _reset_affinity_session(
        self,
        session: _BrowserSession,
        *,
        incoming_message_count: int,
    ) -> None:
        """Reset a pinned browser conversation after transcript discontinuity."""
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

        logger.info(
            "browser_affinity_reset "
            "session_id=%s reason=transcript_discontinuity "
            "incoming_messages=%d",
            session.session_id,
            incoming_message_count,
        )

    @staticmethod
    def _prepare_affinity_tool_catalog(
        plan: _AffinityRequestPlan,
        *,
        tools: list[dict] | None,
        tool_choice,
        parallel_tool_calls: bool,
    ) -> str | None:
        """Render a tool catalog only when this browser conversation needs it."""
        if not plan.inject_tool_catalog:
            return None

        assert tools

        logger.info(
            "browser_tool_catalog "
            "available_tools=%d presented_tools=%d",
            len(tools),
            len(tools),
        )

        return render_tool_catalog(
            tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )

    def _commit_affinity_result(
        self,
        session: _BrowserSession,
        plan: _AffinityRequestPlan,
        result: CompletionResult,
    ) -> None:
        """Commit affinity state only after successful browser execution."""
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
            plan.incoming
            + (
                self._message_signature(
                    assistant_message
                ),
            )
        )

        session.last_request = plan.incoming
        session.last_result = result
        session.tool_catalog_fingerprint = (
            plan.requested_tool_fingerprint
        )

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

            plan = self._plan_affinity_request(
                session,
                messages,
                tools=tools,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
            )

            if plan.replay:
                logger.info(
                    "browser_affinity_replay "
                    "session_id=%s messages=%d",
                    session_id,
                    len(messages),
                )

                assert session.last_result is not None
                return session.last_result

            if plan.reset:
                await self._reset_affinity_session(
                    session,
                    incoming_message_count=len(messages),
                )

            delta = messages[plan.delta_start:]

            tool_catalog_prompt = (
                self._prepare_affinity_tool_catalog(
                    plan,
                    tools=tools,
                    tool_choice=tool_choice,
                    parallel_tool_calls=parallel_tool_calls,
                )
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
                str(plan.reset).lower(),
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

                self._commit_affinity_result(
                    session,
                    plan,
                    result,
                )

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

        await self._enable_temporary_chat(
            page,
            launch_timeout_ms=(
                self.settings.policy_launch_timeout_ms
            ),
            ready_timeout_ms=(
                self.settings.policy_ready_timeout_ms
            ),
        )

        desired_personalization = (
            "Personalized"
            if conversation_policy
            == "temporary_personalized"
            else "Unpersonalized"
        )

        await self._set_temporary_personalization(
            page,
            desired_personalization,
            ready_timeout_ms=(
                self.settings.policy_ready_timeout_ms
            ),
        )

    @staticmethod
    async def _enable_temporary_chat(
        page,
        *,
        launch_timeout_ms: int,
        ready_timeout_ms: int,
    ) -> None:
        """Enable Temporary Chat and verify the UI entered that mode."""
        enabled = page.locator(
            TEMPORARY_CHAT_ENABLED
        )

        if await enabled.is_visible():
            return

        toggle = page.locator(
            TEMPORARY_CHAT_TOGGLE
        )

        await toggle.wait_for(
            state="visible",
            timeout=launch_timeout_ms,
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
            timeout=ready_timeout_ms,
        )

    @staticmethod
    async def _set_temporary_personalization(
        page,
        desired: str,
        *,
        ready_timeout_ms: int,
    ) -> None:
        """Pin Temporary Chat to Personalized or Unpersonalized."""
        desired_button = page.locator(
            aria_label_button(desired)
        )

        if await desired_button.is_visible():
            return

        other = (
            "Unpersonalized"
            if desired == "Personalized"
            else "Personalized"
        )

        launcher = page.locator(
            aria_label_button(other)
        )

        await launcher.wait_for(
            state="visible",
            timeout=ready_timeout_ms,
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
            PERSONALIZATION_OPTION
        ).filter(
            has=page.get_by_text(
                desired,
                exact=True,
            )
        )

        await option.wait_for(
            state="visible",
            timeout=ready_timeout_ms,
        )
        await option.click()

        await desired_button.wait_for(
            state="visible",
            timeout=ready_timeout_ms,
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
            raise ChatGPTBrowserStateError(
                phase='session_create',
                operation='browser_context_available',
            )

        page = await self._context.new_page()

        try:
            await page.goto(
                self.settings.chatgpt_base_url,
                wait_until="domcontentloaded",
                timeout=self.settings.navigation_timeout_ms,
            )

            await self._raise_if_browser_login_required(
                page,
                phase="session_create",
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

    def _build_browser_prompt(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict] | None = None,
        tool_catalog_prompt: str | None = None,
    ) -> str:
        """Build the browser prompt including host/tool bridge context."""
        tool_catalog_text = (
            TOOL_CATALOG_SYSTEM_PREFIX + tool_catalog_prompt
            if tool_catalog_prompt
            else ""
        )

        host_bridge_text = (
            render_host_bridge_context(messages)
        )

        message_text = (
            render_transcript(messages)
            if messages
            else ""
        )

        tool_reminder_text = (
            render_tool_reminder(tools)
        )

        prompt_parts = [
            part
            for part in (
                message_text,
                host_bridge_text,
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

        return prompt

    async def _execute_browser_turn(
        self,
        page,
        prompt: str,
    ) -> _BrowserTurnResult:
        """Execute one prompt/response exchange on an existing browser page."""
        phase = "composer"

        try:
            await self._raise_if_ui_rate_limited(
                page,
                phase="composer",
            )

            phase = "composer_wait"
            composer = page.locator(COMPOSER).last

            phase_started = time.perf_counter()

            await composer.wait_for(
                timeout=self.settings.composer_ready_timeout_ms
            )

            composer_wait_ms = (
                time.perf_counter()
                - phase_started
            ) * 1000

            phase = "composer_fill"

            await self._raise_if_ui_rate_limited(
                page,
                phase="composer_fill",
            )

            input_method = (
                "clipboard_chunked_paste"
                if len(prompt)
                >= self.settings.inline_fill_max_chars
                else "fill"
            )

            logger.info(
                "browser_composer_input_start "
                "method=%s prompt_chars=%d",
                input_method,
                len(prompt),
            )

            phase = "composer_input"
            phase_started = time.perf_counter()

            try:
                input_method = await self._write_composer_text(
                    page,
                    composer,
                    prompt,
                    inline_fill_max_chars=(
                        self.settings.inline_fill_max_chars
                    ),
                    clipboard_chunk_size=(
                        self.settings.clipboard_chunk_size
                    ),
                    clipboard_origin=self.settings.chatgpt_origin,
                )

            except Exception as exc:
                await self._raise_if_ui_rate_limited(
                    page,
                    phase="composer_input",
                )

                raise ChatGPTBrowserOperationError(
                    phase='composer_input',
                    operation='composer_write',
                ) from None

            input_ms = (
                time.perf_counter()
                - phase_started
            ) * 1000

            logger.info(
                "browser_composer_input_complete "
                "method=%s prompt_chars=%d",
                input_method,
                len(prompt),
            )

            phase = "submit"
            phase_started = time.perf_counter()

            await self._submit_prompt(
                page,
                composer,
            )

            submit_ms = (
                time.perf_counter()
                - phase_started
            ) * 1000

            phase = "generation"
            phase_started = time.perf_counter()

            await self._wait_until_idle(page)

            generation_ms = (
                time.perf_counter()
                - phase_started
            ) * 1000

            phase = "extraction"
            phase_started = time.perf_counter()

            response_text = await self._extract_last_answer(
                page
            )

            extraction_ms = (
                time.perf_counter()
                - phase_started
            ) * 1000

            return _BrowserTurnResult(
                text=response_text,
                input_method=input_method,
                composer_wait_ms=composer_wait_ms,
                input_ms=input_ms,
                submit_ms=submit_ms,
                generation_ms=generation_ms,
                extraction_ms=extraction_ms,
            )

        except Exception as exc:
            logger.error(
                "browser_turn_failed "
                "stage=%s error_type=%s prompt_chars=%d",
                phase,
                type(exc).__name__,
                len(prompt),
            )
            raise

    @staticmethod
    def _browser_completion_result(
        response_text: str,
        *,
        model: str,
        level: str | None,
        tools: list[dict] | None,
        parallel_tool_calls: bool,
    ) -> CompletionResult:
        """Convert one browser response into provider completion semantics."""
        tool_calls = (
            parse_tool_calls(
                response_text,
                tools,
                parallel_tool_calls=parallel_tool_calls,
            )
            if tools
            else None
        )

        return CompletionResult(
            text=(
                ""
                if tool_calls
                else response_text
            ),
            model=model,
            level=level,
            tool_calls=tool_calls or [],
        )

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

        prompt = self._build_browser_prompt(
            messages,
            tools=tools,
            tool_catalog_prompt=tool_catalog_prompt,
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

        await self._raise_if_browser_login_required(
            page,
            phase="pinned_completion",
        )

        turn = await self._execute_browser_turn(
            page,
            prompt,
        )

        response_text = turn.text

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

        result = self._browser_completion_result(
            response_text,
            model=session.model,
            level=session.level,
            tools=tools,
            parallel_tool_calls=parallel_tool_calls,
        )

        logger.info(
            "browser_external_tool_result "
            "session_id=%s tool_calls=%d names=%s",
            session.session_id,
            len(result.tool_calls),
            ",".join(
                call.function.name
                for call in result.tool_calls
            ) or "-",
        )

        return result

    async def complete(
        self,
        messages: list[ChatMessage],
        model: str | None = None,
        new_session: bool = False,
        level: str | None = None,
    ) -> CompletionResult:
        return await self._complete_non_affinity(
            messages,
            model=model,
            new_session=new_session,
            level=level,
        )

    async def complete_with_tools(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        new_session: bool = False,
        level: str | None = None,
        tools: list[dict] | None = None,
        tool_choice=None,
        parallel_tool_calls: bool = True,
    ) -> CompletionResult:
        return await self._complete_non_affinity(
            messages,
            model=model,
            new_session=new_session,
            level=level,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )

    async def _complete_non_affinity(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        new_session: bool = False,
        level: str | None = None,
        tools: list[dict] | None = None,
        tool_choice=None,
        parallel_tool_calls: bool = True,
    ) -> CompletionResult:
        async with self._lock:
            total_started = time.perf_counter()
            selected_model = model or self.settings.model_id
            selected_level = level or "-"
            message_count = len(messages)
            if tools:
                tool_catalog_prompt = render_tool_catalog(
                    tools,
                    tool_choice=tool_choice,
                    parallel_tool_calls=parallel_tool_calls,
                )

                prompt = self._build_browser_prompt(
                    messages,
                    tools=tools,
                    tool_catalog_prompt=tool_catalog_prompt,
                )
            else:
                prompt = render_transcript(messages)

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
                await self._raise_if_browser_login_required(
                    page,
                    phase="non_affinity_completion",
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

                phase = "browser_turn"

                turn = await self._execute_browser_turn(
                    page,
                    prompt,
                )

                composer_wait_ms = turn.composer_wait_ms
                fill_ms = turn.input_ms
                submit_ms = turn.submit_ms
                generation_ms = turn.generation_ms
                extraction_ms = turn.extraction_ms
                text = turn.text

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

                result = self._browser_completion_result(
                    text,
                    model=selected_model,
                    level=level,
                    tools=tools,
                    parallel_tool_calls=parallel_tool_calls,
                )

                if tools:
                    logger.info(
                        "browser_external_tool_result "
                        "session_id=- tool_calls=%d names=%s",
                        len(result.tool_calls),
                        ",".join(
                            call.function.name
                            for call in result.tool_calls
                        ) or "-",
                    )

                return result

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

            return {
                "ok": True,
                "backend": "browser",
                "logged_in_hint": (
                    "log in" not in title.lower()
                ),
            }

        except Exception as exc:
            logger.warning(
                "browser_health_check_failed "
                "error_type=%s",
                type(exc).__name__,
            )

            return {
                "ok": False,
                "backend": "browser",
                "error": "browser backend unavailable",
            }

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
        timeout_seconds: float | None = None,
    ):
        """Wait for the reasoning control inside the composer shell."""
        if timeout_seconds is None:
            timeout_seconds = (
                self.settings.reasoning_control_timeout_ms
                / 1000
            )
        known_labels = {
            self._normalize_preference_text(
                self.settings.level_label(level)
            )
            for level in self.settings.available_levels
        }

        composer = page.locator(
            COMPOSER
        ).last

        await composer.wait_for(
            timeout=self.settings.composer_ready_timeout_ms
        )

        deadline = time.monotonic() + timeout_seconds
        started = time.monotonic()

        while time.monotonic() < deadline:
            container = composer

            for depth in range(1, CONTROL_ANCESTOR_MAX_DEPTH + 1):
                try:
                    container = container.locator(PARENT_XPATH)
                    controls = container.locator(
                        REASONING_CONTROLS
                    )
                    count = min(await controls.count(), UI_CANDIDATE_LIMIT)
                except Exception:
                    break

                for index in range(count):
                    control = controls.nth(index)

                    try:
                        if not await control.is_visible(timeout=CONTROL_VISIBLE_PROBE_MS):
                            continue

                        raw_text = (
                            await control.inner_text(timeout=CONTROL_TEXT_PROBE_MS)
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

            await page.wait_for_timeout(TRANSITION_SETTLE_MS)

        raise ChatGPTBrowserOperationError(
            phase='preferences',
            operation='reasoning_picker_ready',
        )

    async def _open_intelligence_picker(self, page):
        control, current_level = (
            await self._wait_for_reasoning_control(page)
        )

        await control.click(
            timeout=self.settings.ui_action_timeout_ms
        )

        content = page.locator(
            INTELLIGENCE_PICKER_CONTENT
        ).last

        await content.wait_for(
            state="visible",
            timeout=self.settings.picker_ready_timeout_ms,
        )

        return control, current_level, content

    async def _find_model_option(
        self,
        content,
        target_label: str,
    ):
        advanced = content.locator(
            MODEL_PICKER_ADVANCED_VIEW
        ).last

        await advanced.wait_for(
            state="visible",
            timeout=self.settings.picker_ready_timeout_ms,
        )

        options = advanced.locator(
            MODEL_OPTIONS
        )

        count = min(await options.count(), UI_CANDIDATE_LIMIT)
        target = self._normalize_preference_text(
            target_label
        )

        for index in range(count):
            option = options.nth(index)

            try:
                if not await option.is_visible(timeout=OPTION_VISIBLE_PROBE_MS):
                    continue

                raw_text = (
                    await option.inner_text(timeout=OPTION_TEXT_PROBE_MS)
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
            MODEL_PICKER_SIMPLE_VIEW
        ).last

        advanced = content.locator(
            MODEL_PICKER_ADVANCED_VIEW
        ).last

        # The advanced panel exists in the DOM even when the simple panel
        # overlays it. data-active tells us which view actually owns input.
        try:
            if await advanced.get_attribute("data-active") == "true":
                return advanced
        except Exception:
            pass

        select_model = content.locator(
            SELECT_MODEL_MENUITEM
        ).last

        await select_model.wait_for(
            state="visible",
            timeout=self.settings.picker_ready_timeout_ms,
        )

        logger.info("model_view_activate_start")

        await select_model.click(
            timeout=self.settings.ui_action_timeout_ms
        )

        deadline = (
            time.monotonic()
            + self.settings.picker_transition_timeout_ms / 1000
        )

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
                    await page.wait_for_timeout(TRANSITION_SETTLE_MS)

                    logger.info(
                        "model_view_activate_complete "
                        "advanced_active=%r simple_active=%r",
                        advanced_active,
                        simple_active,
                    )

                    return advanced

            except Exception:
                pass

            await page.wait_for_timeout(STATE_POLL_MS)

        raise ChatGPTBrowserOperationError(
            phase='preferences',
            operation='model_picker_advanced_view',
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
            raise ChatGPTBrowserOperationError(
                phase='preferences',
                operation='model_option_lookup',
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

            await option.click(
                timeout=self.settings.picker_transition_timeout_ms
            )

            deadline = (
                time.monotonic()
                + self.settings.picker_transition_timeout_ms / 1000
            )

            while time.monotonic() < deadline:
                if await option_checked():
                    logger.info(
                        "model_select_complete model=%s label=%r",
                        model,
                        target_label,
                    )
                    break

                await page.wait_for_timeout(STATE_POLL_MS)
            else:
                raise ChatGPTBrowserOperationError(
                    phase='preferences',
                    operation='model_selection_verify',
                )

        simple = content.locator(
            MODEL_PICKER_SIMPLE_VIEW
        ).last

        # Selecting a model often transitions back to simple view itself.
        # If it did not, one Escape moves advanced -> simple. Do NOT close
        # the picker; reasoning configuration follows immediately.
        deadline = (
            time.monotonic()
            + self.settings.picker_transition_timeout_ms / 1000
        )
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

            await page.wait_for_timeout(STATE_POLL_MS)

        raise ChatGPTBrowserOperationError(
            phase='preferences',
            operation='reasoning_view_activate',
        )

    async def _select_reasoning_level(
        self,
        page,
        level: str,
    ) -> None:
        """Adjust thinking effort in the already-open combined picker."""
        target_label = self.settings.level_label(level)

        picker = page.locator(
            INTELLIGENCE_PICKER_CONTENT
        ).last

        # Normal path: _select_model() deliberately left this open.
        try:
            picker_open = await picker.is_visible(timeout=PICKER_OPEN_PROBE_MS)
        except Exception:
            picker_open = False

        # Defensive fallback for callers that invoke reasoning directly.
        if not picker_open:
            _, _, picker = await self._open_intelligence_picker(
                page
            )

        simple = picker.locator(
            MODEL_PICKER_SIMPLE_VIEW
        ).last

        await simple.wait_for(
            state="visible",
            timeout=self.settings.picker_ready_timeout_ms,
        )

        # Ensure the simple reasoning panel, not advanced model panel,
        # actually owns interaction.
        if await simple.get_attribute("data-active") != "true":
            try:
                await page.keyboard.press("Escape")
            except Exception:
                pass

            deadline = (
                time.monotonic()
                + self.settings.ui_action_timeout_ms / 1000
            )

            while time.monotonic() < deadline:
                if (
                    await simple.get_attribute("data-active")
                    == "true"
                ):
                    break

                await page.wait_for_timeout(STATE_POLL_MS)
            else:
                raise ChatGPTBrowserOperationError(
                    phase='preferences',
                    operation='reasoning_simple_view_activate',
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
            await simple.inner_text(timeout=PICKER_TEXT_TIMEOUT_MS)
        ).strip()

        current = level_from_text(simple_text)

        if current is None:
            raise ChatGPTBrowserOperationError(
                phase='preferences',
                operation='reasoning_level_detect',
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

        for selector in REASONING_SLIDER_SELECTORS:
            try:
                candidates = simple.locator(selector)
                count = min(await candidates.count(), UI_CANDIDATE_LIMIT)

                for index in range(count):
                    candidate = candidates.nth(index)

                    if await candidate.is_visible(timeout=CONTROL_VISIBLE_PROBE_MS):
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
                    timeout=self.settings.ui_action_timeout_ms,
                )

            await page.keyboard.press(key)

            logger.info(
                "reasoning_slider_key_sent_fallback "
                "key=%s",
                key,
            )

        deadline = (
            time.monotonic()
            + self.settings.picker_transition_timeout_ms / 1000
        )
        last_text = simple_text

        while time.monotonic() < deadline:
            try:
                last_text = (
                    await simple.inner_text(timeout=PICKER_STATE_TEXT_TIMEOUT_MS)
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

            await page.wait_for_timeout(STATE_POLL_MS)

        raise ChatGPTBrowserOperationError(
            phase='preferences',
            operation='reasoning_level_select',
        )

    @staticmethod
    def _message_signature(
        message: ChatMessage,
    ) -> str:
        payload = message.model_dump(
            mode="json",
            exclude_none=True,
        )

        # OpenAI-compatible clients may round-trip an assistant
        # tool-call message with content omitted, null, or "".
        # Those forms are semantically equivalent when the message
        # contains tool calls, so keep affinity matching stable.
        if (
            message.role == "assistant"
            and message.tool_calls
            and message.content in (None, "")
        ):
            payload.pop("content", None)

        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    async def _recover_submit_composer_state(
        self,
        page,
    ) -> tuple[Any | None, bool]:
        """Recover submission state after the original composer becomes unusable.

        A live page with no composer indicates that the submitted composer was
        removed. A replacement composer is inspected directly. Browser/page
        failures are not treated as successful submission.
        """
        try:
            if page.is_closed():
                raise ChatGPTBrowserStateError(
                    phase="submit_confirm",
                    operation="composer_state_inspection",
                )

            composers = page.locator(COMPOSER)
            count = await composers.count()

            if count == 0:
                return None, True

            replacement = composers.last

            remaining = (
                await replacement.inner_text(
                    timeout=SUBMIT_COMPOSER_PROBE_MS
                )
            ).strip()

            return replacement, not remaining

        except ChatGPTBrowserStateError:
            raise

        except Exception as exc:
            logger.error(
                "browser_submit_state_inspection_failed "
                "operation=composer_state_inspection "
                "error_type=%s",
                type(exc).__name__,
            )

            raise ChatGPTBrowserStateError(
                phase="submit_confirm",
                operation="composer_state_inspection",
            ) from exc

    async def _submit_prompt(self, page, composer) -> None:  # pragma: no cover - browser integration
        """Wait for the ChatGPT Send button, click it, and verify submission.

        Large fills can leave the composer populated before ChatGPT has made
        Send actionable. Waiting here prevents a race where an Enter keypress
        is ignored and the request appears to hang.
        """
        selectors = SEND_BUTTON_SELECTORS

        submit_ready_seconds = (
            self.settings.submit_ready_timeout_ms
            / 1000
        )

        deadline = (
            time.monotonic()
            + submit_ready_seconds
        )

        send_button = None

        while time.monotonic() < deadline:
            for selector in selectors:
                try:
                    buttons = page.locator(selector)
                    count = min(await buttons.count(), SEND_BUTTON_CANDIDATE_LIMIT)

                    for index in range(count):
                        button = buttons.nth(index)

                        if not await button.is_visible(timeout=SEND_BUTTON_PROBE_MS):
                            continue

                        if await button.is_enabled(timeout=SEND_BUTTON_PROBE_MS):
                            send_button = button
                            break

                    if send_button is not None:
                        break

                except Exception:
                    continue

            if send_button is not None:
                break

            await page.wait_for_timeout(SEND_BUTTON_PROBE_MS)

        if send_button is None:
            raise ChatGPTBrowserOperationError(
                phase='submit',
                operation='send_button_ready',
            )

        await send_button.click(
            timeout=self.settings.ui_action_timeout_ms
        )

        # Confirm that the click actually submitted the prompt.
        stop = page.locator(
            STOP_BUTTON
        )

        confirm_deadline = (
            time.monotonic()
            + self.settings.submit_confirm_timeout_ms / 1000
        )

        while time.monotonic() < confirm_deadline:
            try:
                stop_count = await stop.count()
                for index in range(stop_count):
                    if await stop.nth(index).is_visible(timeout=SUBMIT_STOP_PROBE_MS):
                        return
            except Exception:
                pass

            try:
                remaining = (
                    await composer.inner_text(timeout=SUBMIT_COMPOSER_PROBE_MS)
                ).strip()

                if not remaining:
                    return
            except Exception as exc:
                logger.info(
                    "browser_submit_composer_replaced "
                    "error_type=%s",
                    type(exc).__name__,
                )

                composer, submitted = (
                    await self._recover_submit_composer_state(
                        page
                    )
                )

                if submitted:
                    return

            await page.wait_for_timeout(SUBMIT_POLL_MS)

        raise ChatGPTBrowserOperationError(
            phase='submit_confirm',
            operation='submission_confirm',
        )

    async def _wait_until_idle(self, page) -> None:  # pragma: no cover - browser integration
        stop = page.locator(
            STOP_BUTTON
        )

        deadline = (
            time.monotonic()
            + self.settings.generation_timeout_seconds
        )

        while time.monotonic() < deadline:
            try:
                if await stop.count() == 0:
                    await page.wait_for_timeout(
                        self.settings.generation_settle_ms
                    )
                    return
            except Exception as exc:
                logger.error(
                    "browser_generation_state_inspection_failed "
                    "operation=stop_control_count "
                    "error_type=%s",
                    type(exc).__name__,
                )

                raise ChatGPTBrowserStateError(
                    phase="generation_wait",
                    operation="stop_control_count",
                ) from exc

            await page.wait_for_timeout(
                self.settings.generation_poll_ms
            )

        logger.warning(
            "browser_generation_wait_timeout "
            "timeout_seconds=%d",
            self.settings.generation_timeout_seconds,
        )

        raise ChatGPTGenerationTimeoutError(
            timeout_seconds=(
                self.settings.generation_timeout_seconds
            ),
        )

    async def _start_new_session(self, page) -> None:  # pragma: no cover - browser integration
        """Move ChatGPT to a fresh conversation before sending the prompt."""
        await page.goto(
            self.settings.chatgpt_base_url,
            wait_until="domcontentloaded",
            timeout=self.settings.navigation_timeout_ms,
        )

        await page.wait_for_timeout(
            self.settings.new_session_settle_ms
        )

        if (
            await page.locator(
                COMPOSER
            ).count()
            > 0
        ):
            return

        for selector in NEW_CHAT_SELECTORS:
            candidate = page.locator(selector).first

            try:
                if await candidate.count() == 0:
                    continue

                await candidate.click(
                    timeout=self.settings.ui_action_timeout_ms
                )

                await page.wait_for_timeout(
                    self.settings.new_session_settle_ms
                )

                return

            except Exception:
                continue

    async def _extract_last_answer(self, page) -> str:  # pragma: no cover - browser integration
        candidates = page.locator(
            ASSISTANT_MESSAGES
        )

        count = await candidates.count()

        if count == 0:
            body = await page.locator(BODY).inner_text(
                timeout=self.settings.extraction_timeout_ms
            )

            return body[
                -RESPONSE_BODY_FALLBACK_CHARS:
            ]

        return (
            await candidates.nth(
                count - 1
            ).inner_text()
        ).strip()



def build_backend(settings: Settings) -> Backend:
    if settings.backend == "browser":
        return BrowserBackend(settings)
    return MockBackend(settings)
