import asyncio

from chatgpt_web_provider.backends import (
    BrowserBackend,
    _BrowserSession,
)
from chatgpt_web_provider.config import Settings
from chatgpt_web_provider.models import ChatMessage, CompletionResult
from chatgpt_web_provider.tool_bridge import (
    tool_catalog_fingerprint,
)


class FakeBrowserSessionBackend(BrowserBackend):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.created = []
        self.closed = []

        self.active_global = 0
        self.max_active_global = 0

        self.active_by_session = {}
        self.max_active_by_session = {}

    async def _create_pinned_session_page(
        self,
        model: str,
        level: str,
        conversation_policy: str = "regular",
    ):
        page = object()
        self.created.append(
            (model, level, page, conversation_policy)
        )
        return page

    async def _close_pinned_session_page(self, page) -> None:
        self.closed.append(page)

    async def _complete_pinned_session(self, session, messages):
        session_id = session.session_id

        self.active_global += 1
        self.max_active_global = max(
            self.max_active_global,
            self.active_global,
        )

        current = self.active_by_session.get(session_id, 0) + 1
        self.active_by_session[session_id] = current
        self.max_active_by_session[session_id] = max(
            self.max_active_by_session.get(session_id, 0),
            current,
        )

        try:
            await asyncio.sleep(0.05)

            last_user = next(
                (
                    m.text()
                    for m in reversed(messages)
                    if m.role == "user"
                ),
                "",
            )

            return CompletionResult(
                text=f"{session_id}:{last_user}",
                model=session.model,
                level=session.level,
            )
        finally:
            self.active_global -= 1
            self.active_by_session[session_id] -= 1


def settings():
    return Settings(
        api_keys=["test-token"],
        backend="browser",
        model_id="gpt-a",
        available_models=["gpt-a"],
        available_levels=["high", "xhigh"],
    )


def test_browser_session_creation_pins_model_and_level():
    async def run():
        backend = FakeBrowserSessionBackend(settings())

        record = await backend.create_session(
            "re-high",
            "gpt-a",
            "high",
        )

        assert record == {
            "session_id": "re-high",
            "model": "gpt-a",
            "level": "high",
            "conversation_policy": "regular",
            "state": "ready",
        }

        assert len(backend.created) == 1
        assert backend.created[0][:2] == ("gpt-a", "high")

        result = await backend.complete_session(
            "re-high",
            [ChatMessage(role="user", content="hello")],
        )

        assert result.text == "re-high:hello"
        assert result.model == "gpt-a"
        assert result.level == "high"

        after = await backend.get_session("re-high")
        assert after["state"] == "ready"

    asyncio.run(run())


def test_browser_sessions_run_concurrently_but_each_session_serializes():
    async def run():
        backend = FakeBrowserSessionBackend(settings())

        await backend.create_session("re-high", "gpt-a", "high")
        await backend.create_session("review-xhigh", "gpt-a", "xhigh")

        message = [ChatMessage(role="user", content="work")]

        # Different sessions should execute concurrently.
        await asyncio.gather(
            backend.complete_session("re-high", message),
            backend.complete_session("review-xhigh", message),
        )

        assert backend.max_active_global == 2

        # Reset only the global observation.
        backend.active_global = 0
        backend.max_active_global = 0

        # Two operations against the same conversation must serialize.
        await asyncio.gather(
            backend.complete_session("re-high", message),
            backend.complete_session("re-high", message),
        )

        assert backend.max_active_by_session["re-high"] == 1

    asyncio.run(run())


def test_browser_session_delete_closes_page_and_removes_identity():
    async def run():
        backend = FakeBrowserSessionBackend(settings())

        await backend.create_session(
            "review-xhigh",
            "gpt-a",
            "xhigh",
        )

        page = backend.created[0][2]

        await backend.delete_session("review-xhigh")

        assert backend.closed == [page]

        try:
            await backend.get_session("review-xhigh")
        except KeyError:
            pass
        else:
            raise AssertionError("deleted session still exists")

    asyncio.run(run())


class FakeKeyboard:
    def __init__(self):
        self.pressed = []

    async def press(self, key):
        self.pressed.append(key)


class FakePinnedPage:
    def __init__(self):
        self.goto_calls = []
        self.keyboard = FakeKeyboard()
        self.closed = False

    async def goto(self, url, **kwargs):
        self.goto_calls.append((url, kwargs))

    async def title(self):
        return "ChatGPT"

    async def close(self):
        self.closed = True


class FakePinnedContext:
    def __init__(self, page):
        self.page = page
        self.new_page_calls = 0

    async def new_page(self):
        self.new_page_calls += 1
        return self.page


class PinnedPageInitializationBackend(BrowserBackend):
    def __init__(self, settings):
        super().__init__(settings)
        self.fake_page = FakePinnedPage()
        self.fake_context = FakePinnedContext(self.fake_page)
        self.started_sessions = []
        self.applied_preferences = []

    async def _ensure_page(self):
        self._context = self.fake_context
        return object()

    async def _start_new_session(self, page):
        self.started_sessions.append(page)

    async def _apply_preferences(self, page, model, level):
        self.applied_preferences.append(
            (page, model, level)
        )


def test_create_pinned_session_page_uses_current_preference_path():
    async def run():
        backend = PinnedPageInitializationBackend(settings())

        page = await backend._create_pinned_session_page(
            "gpt-a",
            "high",
        )

        assert page is backend.fake_page
        assert backend.fake_context.new_page_calls == 1

        assert backend.started_sessions == [
            backend.fake_page
        ]

        assert backend.applied_preferences == [
            (
                backend.fake_page,
                "gpt-a",
                "high",
            )
        ]

        assert backend.fake_page.goto_calls
        assert (
            backend.fake_page.goto_calls[0][0]
            == "https://chatgpt.com/"
        )

    asyncio.run(run())


def test_create_pinned_session_page_closes_page_on_preference_failure():
    class FailingBackend(PinnedPageInitializationBackend):
        async def _apply_preferences(
            self,
            page,
            model,
            level,
        ):
            raise RuntimeError("preference failure")

    async def run():
        backend = FailingBackend(settings())

        try:
            await backend._create_pinned_session_page(
                "gpt-a",
                "high",
            )
        except RuntimeError as exc:
            assert str(exc) == "preference failure"
        else:
            raise AssertionError("expected preference failure")

        assert backend.fake_page.closed is True

    asyncio.run(run())


def test_affinity_session_submits_only_new_transcript_delta():
    class AffinityBackend(FakeBrowserSessionBackend):
        def __init__(self, settings):
            super().__init__(settings)
            self.submitted_batches = []
            self.reset_pages = []

        async def _complete_pinned_session(
            self,
            session,
            messages,
        ):
            self.submitted_batches.append(
                [
                    (message.role, message.text())
                    for message in messages
                ]
            )

            return await super()._complete_pinned_session(
                session,
                messages,
            )

        async def _start_new_session(self, page):
            self.reset_pages.append(page)

        async def _apply_preferences(
            self,
            page,
            model,
            level,
        ):
            return None

    async def run():
        backend = AffinityBackend(settings())

        await backend.create_session(
            "passthrough-xhigh",
            "gpt-a",
            "xhigh",
        )

        first_messages = [
            ChatMessage(
                role="system",
                content="system prompt",
            ),
            ChatMessage(
                role="user",
                content="alpha",
            ),
        ]

        first = await backend.complete_affinity_session(
            "passthrough-xhigh",
            first_messages,
        )

        assert backend.submitted_batches == [
            [
                ("system", "system prompt"),
                ("user", "alpha"),
            ]
        ]

        second_messages = [
            ChatMessage(
                role="system",
                content="system prompt",
            ),
            ChatMessage(
                role="user",
                content="alpha",
            ),
            ChatMessage(
                role="assistant",
                content=first.text,
            ),
            ChatMessage(
                role="user",
                content="beta",
            ),
        ]

        second = await backend.complete_affinity_session(
            "passthrough-xhigh",
            second_messages,
        )

        # Browser conversation already contains the first exchange.
        # Only the newly appended Hermes message is submitted.
        assert backend.submitted_batches[1] == [
            ("user", "beta"),
        ]

        assert backend.reset_pages == []

        # Exact retry must not submit to ChatGPT again.
        repeated = await backend.complete_affinity_session(
            "passthrough-xhigh",
            second_messages,
        )

        assert repeated.text == second.text
        assert len(backend.submitted_batches) == 2

        # Simulate Hermes context compression / history rewrite.
        compacted = [
            ChatMessage(
                role="system",
                content="compressed context",
            ),
            ChatMessage(
                role="user",
                content="gamma",
            ),
        ]

        await backend.complete_affinity_session(
            "passthrough-xhigh",
            compacted,
        )

        assert len(backend.reset_pages) == 1

        # After discontinuity, the complete new logical transcript
        # seeds a fresh ChatGPT conversation.
        assert backend.submitted_batches[2] == [
            ("system", "compressed context"),
            ("user", "gamma"),
        ]

    asyncio.run(run())


def test_policy_session_dispatches_conversation_modes():
    class PolicyBackend(FakeBrowserSessionBackend):
        def __init__(self, settings):
            super().__init__(settings)
            self.started = []
            self.temporary_enabled = []
            self.personalization = []

        async def _start_new_session(self, page):
            self.started.append(page)

        async def _enable_temporary_chat(
            self,
            page,
            *,
            launch_timeout_ms,
            ready_timeout_ms,
        ):
            assert launch_timeout_ms == (
                self.settings.policy_launch_timeout_ms
            )
            assert ready_timeout_ms == (
                self.settings.policy_ready_timeout_ms
            )
            self.temporary_enabled.append(page)

        async def _set_temporary_personalization(
            self,
            page,
            desired,
            *,
            ready_timeout_ms,
        ):
            assert ready_timeout_ms == (
                self.settings.policy_ready_timeout_ms
            )
            self.personalization.append(
                (page, desired)
            )

    async def run():
        backend = PolicyBackend(settings())
        page = object()

        await backend._start_policy_session(
            page,
            "regular",
        )

        assert backend.started == [page]
        assert backend.temporary_enabled == []
        assert backend.personalization == []

        await backend._start_policy_session(
            page,
            "temporary_personalized",
        )

        assert backend.started == [page, page]
        assert backend.temporary_enabled == [page]
        assert backend.personalization == [
            (page, "Personalized"),
        ]

        await backend._start_policy_session(
            page,
            "temporary_unpersonalized",
        )

        assert backend.started == [
            page,
            page,
            page,
        ]
        assert backend.temporary_enabled == [
            page,
            page,
        ]
        assert backend.personalization == [
            (page, "Personalized"),
            (page, "Unpersonalized"),
        ]

    asyncio.run(run())


def test_temporary_personalization_uses_exact_menu_text():
    class FakeLocator:
        def __init__(self, visible=True):
            self.visible = visible
            self.clicks = 0
            self.filter_kwargs = None

        async def is_visible(self):
            return self.visible

        async def wait_for(self, **kwargs):
            return None

        async def click(self):
            self.clicks += 1

        def filter(self, **kwargs):
            self.filter_kwargs = kwargs
            return self

    class FakePage:
        def __init__(self):
            self.desired = FakeLocator(visible=False)
            self.other = FakeLocator()
            self.menu = FakeLocator()
            self.text_token = object()
            self.get_by_text_calls = []

        def locator(self, selector):
            if selector == 'button[aria-label="Personalized"]':
                return self.desired

            if selector == 'button[aria-label="Unpersonalized"]':
                return self.other

            if selector == '[role="menuitemradio"]':
                return self.menu

            raise AssertionError(
                f"unexpected selector: {selector}"
            )

        def get_by_text(self, text, exact=False):
            self.get_by_text_calls.append(
                (text, exact)
            )
            return self.text_token

    async def run():
        page = FakePage()

        await BrowserBackend._set_temporary_personalization(
            page,
            "Personalized",
            ready_timeout_ms=10_000,
        )

        assert page.get_by_text_calls == [
            ("Personalized", True),
        ]

        assert page.menu.filter_kwargs == {
            "has": page.text_token,
        }

        assert page.other.clicks == 1
        assert page.menu.clicks == 1

    asyncio.run(run())


def test_affinity_tool_call_and_tool_result_preserve_delta_history():
    from chatgpt_web_provider.models import (
        ToolCall,
        ToolFunctionCall,
    )

    class ToolAffinityBackend(FakeBrowserSessionBackend):
        def __init__(self, settings):
            super().__init__(settings)
            self.submitted = []
            self.catalog_prompts = []
            self.calls = 0

        async def _complete_pinned_session(
            self,
            session,
            messages,
            *,
            tools=None,
            tool_choice=None,
            parallel_tool_calls=True,
            tool_catalog_prompt=None,
        ):
            self.submitted.append(
                [
                    message.model_dump(
                        exclude_none=True
                    )
                    for message in messages
                ]
            )

            self.catalog_prompts.append(
                tool_catalog_prompt
            )

            self.calls += 1

            if self.calls == 1:
                return CompletionResult(
                    model=session.model,
                    level=session.level,
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            function=ToolFunctionCall(
                                name="worker_list",
                                arguments="{}",
                            ),
                        )
                    ],
                )

            return CompletionResult(
                text="five workers",
                model=session.model,
                level=session.level,
            )

    async def run():
        backend = ToolAffinityBackend(settings())

        await backend.create_session(
            "ctf-parent",
            "gpt-a",
            "xhigh",
        )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "worker_list",
                    "description": "List workers.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            }
        ]

        first_messages = [
            ChatMessage(
                role="user",
                content="List workers.",
            ),
        ]

        first = await backend.complete_affinity_session(
            "ctf-parent",
            first_messages,
            tools=tools,
            tool_choice="auto",
            parallel_tool_calls=True,
        )

        assert len(first.tool_calls) == 1
        assert first.tool_calls[0].id == "call-1"
        assert backend.catalog_prompts[0] is not None

        second_messages = [
            ChatMessage(
                role="user",
                content="List workers.",
            ),
            ChatMessage(
                role="assistant",
                content=None,
                tool_calls=first.tool_calls,
            ),
            ChatMessage(
                role="tool",
                content='{"workers":["re-high"]}',
                tool_call_id="call-1",
            ),
        ]

        second = await backend.complete_affinity_session(
            "ctf-parent",
            second_messages,
            tools=tools,
            tool_choice="auto",
            parallel_tool_calls=True,
        )

        assert second.text == "five workers"

        # The browser already contains the user turn and its
        # external-tool request. Only the tool result is new.
        assert len(backend.submitted[1]) == 1
        assert backend.submitted[1][0]["role"] == "tool"
        assert (
            backend.submitted[1][0]["tool_call_id"]
            == "call-1"
        )

        # Same catalog is not re-injected every turn.
        assert backend.catalog_prompts[1] is None

        session = await backend._get_browser_session(
            "ctf-parent"
        )

        assert session.logical_transcript
        assert backend.calls == 2

    asyncio.run(run())


def test_chatgpt_rate_limit_modal_is_detected():
    from chatgpt_web_provider.backends import (
        ChatGPTUIRateLimitError,
    )

    class FakeModal:
        async def is_visible(self):
            return True

    class FakePage:
        def __init__(self):
            self.selectors = []

        def locator(self, selector):
            self.selectors.append(selector)
            return FakeModal()

    async def run():
        page = FakePage()

        try:
            await BrowserBackend._raise_if_ui_rate_limited(
                page,
                phase="test",
            )
        except ChatGPTUIRateLimitError as exc:
            assert exc.phase == "test"
        else:
            raise AssertionError(
                "visible rate-limit modal was ignored"
            )

        assert page.selectors == [
            '[data-testid="modal-conversation-history-rate-limit"]'
        ]

    asyncio.run(run())


def test_composer_writer_uses_fast_path_for_large_prompts():
    class FakeContext:
        def __init__(self, events):
            self.events = events

        async def grant_permissions(
            self,
            permissions,
            origin=None,
        ):
            self.events.append(
                (
                    "grant_permissions",
                    tuple(permissions),
                    origin,
                )
            )

    class FakeKeyboard:
        def __init__(self, events):
            self.events = events

        async def press(self, key):
            self.events.append(
                ("keyboard_press", key)
            )

        async def insert_text(self, value):
            self.events.append(
                ("insert_text", len(value))
            )

    class FakePage:
        def __init__(self, events):
            self.events = events
            self.context = FakeContext(events)
            self.keyboard = FakeKeyboard(events)

        async def evaluate(self, script, value):
            self.events.append(
                ("clipboard_write", len(value))
            )

    class FakeComposer:
        def __init__(self, events):
            self.events = events

        async def fill(self, value):
            self.events.append(
                ("fill", len(value))
            )

        async def focus(self):
            self.events.append(("focus",))

    async def run():
        small_events = []

        method = await BrowserBackend._write_composer_text(
            FakePage(small_events),
            FakeComposer(small_events),
            "hello",
            inline_fill_max_chars=16_384,
            clipboard_chunk_size=4_096,
            clipboard_origin="https://chatgpt.com",
        )

        assert method == "fill"
        assert small_events == [
            ("fill", 5),
        ]

        large_events = []
        large_text = "x" * 70_000

        method = await BrowserBackend._write_composer_text(
            FakePage(large_events),
            FakeComposer(large_events),
            large_text,
            inline_fill_max_chars=16_384,
            clipboard_chunk_size=4_096,
            clipboard_origin="https://chatgpt.com",
        )

        assert method == "clipboard_chunked_paste"

        chunks = [
            large_text[offset:offset + 4_096]
            for offset in range(0, len(large_text), 4_096)
        ]

        expected = [
            (
                "grant_permissions",
                (
                    "clipboard-read",
                    "clipboard-write",
                ),
                "https://chatgpt.com",
            ),
            ("focus",),
            ("keyboard_press", "Control+A"),
            ("keyboard_press", "Backspace"),
        ]

        for chunk in chunks:
            expected.extend(
                [
                    ("clipboard_write", len(chunk)),
                    ("keyboard_press", "Control+V"),
                ]
            )

        assert len(chunks) == 18
        assert sum(len(chunk) for chunk in chunks) == 70_000
        assert large_events == expected

    asyncio.run(run())


def test_browser_tool_reminder_keeps_tools_salient():
    from chatgpt_web_provider.browser_prompts import (
        render_tool_reminder,
    )
    def tool(name):
        return {
            "type": "function",
            "function": {
                "name": name,
                "parameters": {"type": "object"},
            },
        }

    gateway_tools = [
        tool("tool_search"),
        tool("tool_describe"),
        tool("tool_call"),
    ]

    reminder = render_tool_reminder(
        gateway_tools
    )

    assert "tool_search" in reminder
    assert "tool_describe" in reminder
    assert "tool_call" in reminder

    direct = render_tool_reminder(
        [tool("mcp__worker_broker__worker_list")]
    )

    assert "external tools listed above" in direct
    assert "call the matching tool" in direct
    assert "actual tool call reports failure" in direct

    assert render_tool_reminder(None) == ""
    assert render_tool_reminder([]) == ""


def test_browser_host_bridge_context_for_developer_messages():
    from chatgpt_web_provider.browser_prompts import (
        render_host_bridge_context,
    )
    developer_messages = [
        ChatMessage(
            role="developer",
            content="You are Hermes Agent.",
        ),
        ChatMessage(
            role="user",
            content="list workers",
        ),
    ]

    bridge = render_host_bridge_context(
        developer_messages
    )

    assert "HOST BRIDGE CONTEXT" in bridge
    assert "reasoning backend for a Hermes Agent host" in bridge
    assert "not the native ChatGPT web UI" in bridge
    assert "external tool catalog" in bridge
    assert "authoritative tool set for this request" in bridge

    user_only = [
        ChatMessage(
            role="user",
            content="hello",
        ),
    ]

    assert (
        render_host_bridge_context(user_only)
        == ""
    )

    assert (
        render_host_bridge_context([])
        == ""
    )


def test_browser_complete_uses_shared_turn_executor():
    from types import SimpleNamespace

    class FakePage:
        async def title(self):
            return "ChatGPT"

    class SharedTurnBackend(BrowserBackend):
        def __init__(self, provider_settings):
            super().__init__(provider_settings)
            self.page = FakePage()
            self.reset_pages = []
            self.preferences = []
            self.turns = []

        async def _ensure_page(self):
            return self.page

        async def _start_new_session(self, page):
            self.reset_pages.append(page)

        async def _apply_preferences(
            self,
            page,
            model,
            level,
        ):
            self.preferences.append(
                (page, model, level)
            )

        async def _execute_browser_turn(
            self,
            page,
            prompt,
        ):
            self.turns.append(
                (page, prompt)
            )

            return SimpleNamespace(
                text="shared turn result",
                input_method="fill",
                composer_wait_ms=1.0,
                input_ms=2.0,
                submit_ms=3.0,
                generation_ms=4.0,
                extraction_ms=5.0,
            )

    async def run():
        backend = SharedTurnBackend(settings())

        result = await backend.complete(
            [
                ChatMessage(
                    role="user",
                    content="shared turn probe",
                )
            ],
            model="gpt-a",
            new_session=True,
            level="high",
        )

        assert backend.reset_pages == [
            backend.page,
        ]

        assert backend.preferences == [
            (
                backend.page,
                "gpt-a",
                "high",
            )
        ]

        assert len(backend.turns) == 1

        page, prompt = backend.turns[0]

        assert page is backend.page
        assert "shared turn probe" in prompt

        assert result.text == "shared turn result"
        assert result.model == "gpt-a"
        assert result.level == "high"

    asyncio.run(run())


def test_generation_wait_timeout_raises_typed_error(
    monkeypatch,
):
    import chatgpt_web_provider.backends as backends_module

    from chatgpt_web_provider.backends import (
        ChatGPTGenerationTimeoutError,
    )

    class FakeStop:
        async def count(self):
            return 1

    class FakePage:
        def __init__(self):
            self.waits = []

        def locator(self, selector):
            return FakeStop()

        async def wait_for_timeout(self, milliseconds):
            self.waits.append(milliseconds)

    async def run():
        provider_settings = settings()
        provider_settings.generation_timeout_seconds = 2
        provider_settings.generation_poll_ms = 17

        backend = BrowserBackend(provider_settings)
        page = FakePage()

        # deadline creation, first while check, second while check
        clock = iter(
            [
                100.0,
                100.0,
                103.0,
            ]
        )

        class FakeTime:
            def monotonic(self):
                return next(clock)

        # Replace the module-local time reference rather than modifying
        # time.monotonic globally; asyncio itself uses the real monotonic
        # clock during event-loop shutdown.
        monkeypatch.setattr(
            backends_module,
            "time",
            FakeTime(),
        )

        try:
            await backend._wait_until_idle(page)
        except ChatGPTGenerationTimeoutError as exc:
            assert exc.timeout_seconds == 2
        else:
            raise AssertionError(
                "generation timeout was silently ignored"
            )

        assert page.waits == [17]

    asyncio.run(run())


def test_browser_non_affinity_tools_use_external_tool_bridge():
    from types import SimpleNamespace

    class FakePage:
        async def title(self):
            return "ChatGPT"

    class ToolBackend(BrowserBackend):
        def __init__(self, provider_settings):
            super().__init__(provider_settings)
            self.page = FakePage()
            self.prompts = []

        async def _ensure_page(self):
            return self.page

        async def _apply_preferences(
            self,
            page,
            model,
            level,
        ):
            return None

        async def _execute_browser_turn(
            self,
            page,
            prompt,
        ):
            self.prompts.append(prompt)

            return SimpleNamespace(
                text=(
                    "<<<HERMES_EXTERNAL_TOOL_CALLS>>>\n"
                    '{"calls":[{"name":"worker_list",'
                    '"arguments":{}}]}\n'
                    "<<<END_HERMES_EXTERNAL_TOOL_CALLS>>>"
                ),
                input_method="fill",
                composer_wait_ms=1.0,
                input_ms=2.0,
                submit_ms=3.0,
                generation_ms=4.0,
                extraction_ms=5.0,
            )

    async def run():
        backend = ToolBackend(settings())

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "worker_list",
                    "description": "List workers.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            }
        ]

        result = await backend.complete_with_tools(
            [
                ChatMessage(
                    role="user",
                    content="List workers.",
                )
            ],
            model="gpt-a",
            level="high",
            tools=tools,
            tool_choice="auto",
            parallel_tool_calls=True,
        )

        assert len(backend.prompts) == 1
        assert "EXTERNAL TOOL" in backend.prompts[0]
        assert "worker_list" in backend.prompts[0]

        assert result.text == ""
        assert len(result.tool_calls) == 1
        assert (
            result.tool_calls[0].function.name
            == "worker_list"
        )
        assert (
            result.tool_calls[0].function.arguments
            == "{}"
        )

    asyncio.run(run())


def test_generation_state_inspection_failure_is_not_success():
    from chatgpt_web_provider.backends import (
        ChatGPTBrowserStateError,
    )

    class FailingStop:
        async def count(self):
            raise RuntimeError("page closed")

    class FakePage:
        def locator(self, selector):
            return FailingStop()

        async def wait_for_timeout(self, milliseconds):
            raise AssertionError(
                "poll wait should not occur after inspection failure"
            )

    async def run():
        backend = BrowserBackend(settings())

        try:
            await backend._wait_until_idle(
                FakePage()
            )

        except ChatGPTBrowserStateError as exc:
            assert exc.phase == "generation_wait"
            assert exc.operation == "stop_control_count"

        else:
            raise AssertionError(
                "browser-state inspection failure "
                "was treated as successful completion"
            )

    asyncio.run(run())


def test_submit_prompt_recovers_replaced_composer():
    from chatgpt_web_provider.chatgpt_ui import (
        COMPOSER,
        STOP_BUTTON,
    )

    class FakeButton:
        def __init__(self):
            self.clicks = 0

        async def is_visible(self, timeout=None):
            return True

        async def is_enabled(self, timeout=None):
            return True

        async def click(self, timeout=None):
            self.clicks += 1

    class FakeButtons:
        def __init__(self, button):
            self.button = button

        async def count(self):
            return 1

        def nth(self, index):
            assert index == 0
            return self.button

    class FakeStop:
        async def count(self):
            return 0

    class BrokenComposer:
        async def inner_text(self, timeout=None):
            raise RuntimeError("composer detached")

    class ReplacementComposer:
        async def inner_text(self, timeout=None):
            return ""

    class FakeComposers:
        def __init__(self, replacement):
            self.replacement = replacement

        async def count(self):
            return 1

        @property
        def last(self):
            return self.replacement

    class FakePage:
        def __init__(self):
            self.button = FakeButton()
            self.replacement = ReplacementComposer()

        def is_closed(self):
            return False

        def locator(self, selector):
            if selector == STOP_BUTTON:
                return FakeStop()

            if selector == COMPOSER:
                return FakeComposers(
                    self.replacement
                )

            return FakeButtons(self.button)

        async def wait_for_timeout(self, milliseconds):
            return None

    async def run():
        backend = BrowserBackend(settings())
        page = FakePage()

        await backend._submit_prompt(
            page,
            BrokenComposer(),
        )

        assert page.button.clicks == 1

    asyncio.run(run())


def test_submit_prompt_removed_composer_confirms_submission():
    from chatgpt_web_provider.chatgpt_ui import (
        COMPOSER,
        STOP_BUTTON,
    )

    class FakeButton:
        async def is_visible(self, timeout=None):
            return True

        async def is_enabled(self, timeout=None):
            return True

        async def click(self, timeout=None):
            return None

    class FakeButtons:
        async def count(self):
            return 1

        def nth(self, index):
            return FakeButton()

    class FakeStop:
        async def count(self):
            return 0

    class BrokenComposer:
        async def inner_text(self, timeout=None):
            raise RuntimeError("composer removed")

    class NoComposers:
        async def count(self):
            return 0

        @property
        def last(self):
            raise AssertionError(
                "last should not be read when count is zero"
            )

    class FakePage:
        def is_closed(self):
            return False

        def locator(self, selector):
            if selector == STOP_BUTTON:
                return FakeStop()

            if selector == COMPOSER:
                return NoComposers()

            return FakeButtons()

        async def wait_for_timeout(self, milliseconds):
            return None

    async def run():
        backend = BrowserBackend(settings())

        await backend._submit_prompt(
            FakePage(),
            BrokenComposer(),
        )

    asyncio.run(run())


def test_submit_prompt_closed_page_is_browser_state_error():
    from chatgpt_web_provider.backends import (
        ChatGPTBrowserStateError,
    )
    from chatgpt_web_provider.chatgpt_ui import (
        STOP_BUTTON,
    )

    class FakeButton:
        async def is_visible(self, timeout=None):
            return True

        async def is_enabled(self, timeout=None):
            return True

        async def click(self, timeout=None):
            return None

    class FakeButtons:
        async def count(self):
            return 1

        def nth(self, index):
            return FakeButton()

    class FakeStop:
        async def count(self):
            return 0

    class BrokenComposer:
        async def inner_text(self, timeout=None):
            raise RuntimeError("page closed")

    class FakePage:
        def is_closed(self):
            return True

        def locator(self, selector):
            if selector == STOP_BUTTON:
                return FakeStop()

            return FakeButtons()

        async def wait_for_timeout(self, milliseconds):
            return None

    async def run():
        backend = BrowserBackend(settings())

        try:
            await backend._submit_prompt(
                FakePage(),
                BrokenComposer(),
            )

        except ChatGPTBrowserStateError as exc:
            assert exc.phase == "submit_confirm"
            assert (
                exc.operation
                == "composer_state_inspection"
            )

        else:
            raise AssertionError(
                "closed browser page was treated as "
                "successful submission"
            )

    asyncio.run(run())


def test_affinity_request_plan_selects_transcript_delta():
    backend = BrowserBackend(settings())

    first = ChatMessage(
        role="user",
        content="alpha",
    )

    assistant = ChatMessage(
        role="assistant",
        content="first response",
    )

    second = ChatMessage(
        role="user",
        content="beta",
    )

    session = _BrowserSession(
        session_id="plan-delta",
        model="gpt-a",
        level="high",
        state="ready",
    )

    session.logical_transcript = (
        backend._message_signature(first),
        backend._message_signature(assistant),
    )

    plan = backend._plan_affinity_request(
        session,
        [
            first,
            assistant,
            second,
        ],
        tools=None,
        tool_choice=None,
        parallel_tool_calls=True,
    )

    assert plan.replay is False
    assert plan.reset is False
    assert plan.delta_start == 2
    assert plan.inject_tool_catalog is False


def test_affinity_request_plan_detects_replay():
    backend = BrowserBackend(settings())

    message = ChatMessage(
        role="user",
        content="alpha",
    )

    incoming = (
        backend._message_signature(message),
    )

    session = _BrowserSession(
        session_id="plan-replay",
        model="gpt-a",
        level="high",
        state="ready",
    )

    session.last_request = incoming
    session.last_result = CompletionResult(
        text="cached",
        model="gpt-a",
        level="high",
    )

    plan = backend._plan_affinity_request(
        session,
        [message],
        tools=None,
        tool_choice=None,
        parallel_tool_calls=True,
    )

    assert plan.replay is True


def test_affinity_request_plan_resets_on_discontinuity():
    backend = BrowserBackend(settings())

    old = ChatMessage(
        role="user",
        content="old history",
    )

    new = ChatMessage(
        role="user",
        content="rewritten history",
    )

    session = _BrowserSession(
        session_id="plan-reset",
        model="gpt-a",
        level="high",
        state="ready",
    )

    session.logical_transcript = (
        backend._message_signature(old),
    )

    plan = backend._plan_affinity_request(
        session,
        [new],
        tools=None,
        tool_choice=None,
        parallel_tool_calls=True,
    )

    assert plan.reset is True
    assert plan.delta_start == 0


def test_affinity_request_plan_reinjects_tools_after_reset():
    backend = BrowserBackend(settings())

    tools = [
        {
            "type": "function",
            "function": {
                "name": "worker_list",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        }
    ]

    fingerprint = tool_catalog_fingerprint(
        tools,
        tool_choice="auto",
        parallel_tool_calls=True,
    )

    old = ChatMessage(
        role="user",
        content="old history",
    )

    new = ChatMessage(
        role="user",
        content="rewritten history",
    )

    session = _BrowserSession(
        session_id="plan-tool-reset",
        model="gpt-a",
        level="high",
        state="ready",
    )

    session.logical_transcript = (
        backend._message_signature(old),
    )

    # Same catalog existed in the old browser conversation.
    session.tool_catalog_fingerprint = fingerprint

    plan = backend._plan_affinity_request(
        session,
        [new],
        tools=tools,
        tool_choice="auto",
        parallel_tool_calls=True,
    )

    assert plan.reset is True
    assert (
        plan.requested_tool_fingerprint
        == fingerprint
    )
    assert plan.inject_tool_catalog is True


def test_browser_health_does_not_expose_page_title():
    class FakePage:
        async def title(self):
            return "Sensitive Conversation Title"

    class HealthBackend(BrowserBackend):
        async def _ensure_page(self):
            return FakePage()

    async def run():
        backend = HealthBackend(settings())

        result = await backend.health()

        assert result == {
            "ok": True,
            "backend": "browser",
            "logged_in_hint": True,
        }

        assert "title" not in result

    asyncio.run(run())


def test_browser_health_does_not_expose_exception_text():
    class HealthBackend(BrowserBackend):
        async def _ensure_page(self):
            raise RuntimeError(
                "profile=/home/example/private "
                "token=secret-value"
            )

    async def run():
        backend = HealthBackend(settings())

        result = await backend.health()

        assert result == {
            "ok": False,
            "backend": "browser",
            "error": "browser backend unavailable",
        }

        rendered = str(result)

        assert "/home/example/private" not in rendered
        assert "secret-value" not in rendered

    asyncio.run(run())


def test_submit_ready_timeout_is_browser_operation_error():
    from chatgpt_web_provider.backends import (
        ChatGPTBrowserOperationError,
    )

    async def run():
        provider_settings = settings()

        # Force the readiness loop to expire before probing the page.
        provider_settings.submit_ready_timeout_ms = 0

        backend = BrowserBackend(
            provider_settings
        )

        try:
            await backend._submit_prompt(
                object(),
                object(),
            )

        except ChatGPTBrowserOperationError as exc:
            assert exc.phase == "submit"
            assert exc.operation == "send_button_ready"

        else:
            raise AssertionError(
                "send-button readiness failure "
                "was not typed as a browser operation"
            )

    asyncio.run(run())


def test_browser_login_state_check_detects_logged_out_profile():
    from chatgpt_web_provider.backends import (
        ChatGPTBrowserLoginRequiredError,
    )

    class LoggedOutPage:
        async def title(self):
            return "Log in - ChatGPT"

    class LoggedInPage:
        async def title(self):
            return "ChatGPT"

    async def run():
        await BrowserBackend._raise_if_browser_login_required(
            LoggedInPage(),
            phase="test",
        )

        try:
            await BrowserBackend._raise_if_browser_login_required(
                LoggedOutPage(),
                phase="test",
            )

        except ChatGPTBrowserLoginRequiredError as exc:
            assert exc.phase == "test"

        else:
            raise AssertionError(
                "logged-out browser profile was not detected"
            )

    asyncio.run(run())


def test_browser_prompt_places_host_bridge_after_developer_context():
    backend = BrowserBackend(settings())

    messages = [
        ChatMessage(
            role="developer",
            content="You are Hermes Agent.",
        ),
        ChatMessage(
            role="user",
            content="Call live_probe.",
        ),
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "live_probe",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        }
    ]

    prompt = backend._build_browser_prompt(
        messages,
        tools=tools,
        tool_catalog_prompt="LIVE TOOL CATALOG",
    )

    developer_pos = prompt.index(
        "DEVELOPER: You are Hermes Agent."
    )

    bridge_pos = prompt.index(
        "HOST BRIDGE CONTEXT:"
    )

    reminder_pos = prompt.index(
        "EXTERNAL TOOL REMINDER:"
    )

    catalog_pos = prompt.index(
        "LIVE TOOL CATALOG"
    )

    assert (
        developer_pos
        < bridge_pos
        < reminder_pos
        < catalog_pos
    )


def test_affinity_tool_call_signature_normalizes_empty_content():
    from chatgpt_web_provider.models import (
        ToolCall,
        ToolFunctionCall,
    )

    backend = BrowserBackend(settings())

    call = ToolCall(
        id="call-roundtrip",
        function=ToolFunctionCall(
            name="worker_list",
            arguments="{}",
        ),
    )

    null_content = ChatMessage(
        role="assistant",
        content=None,
        tool_calls=[call],
    )

    empty_content = ChatMessage(
        role="assistant",
        content="",
        tool_calls=[call],
    )

    omitted_content = ChatMessage.model_validate(
        {
            "role": "assistant",
            "tool_calls": [
                call.model_dump(mode="json"),
            ],
        }
    )

    assert omitted_content.content == ""

    expected = backend._message_signature(
        null_content
    )

    assert (
        backend._message_signature(empty_content)
        == expected
    )
    assert (
        backend._message_signature(omitted_content)
        == expected
    )
