import asyncio

from chatgpt_web_provider.backends import BrowserBackend
from chatgpt_web_provider.config import Settings
from chatgpt_web_provider.models import ChatMessage, CompletionResult


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

        async def _enable_temporary_chat(self, page):
            self.temporary_enabled.append(page)

        async def _set_temporary_personalization(
            self,
            page,
            desired,
        ):
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
