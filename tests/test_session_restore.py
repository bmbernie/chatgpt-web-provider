import asyncio

from chatgpt_web_provider.backends import BrowserBackend
from chatgpt_web_provider.config import Settings
from chatgpt_web_provider.session_store import (
    PersistedSession,
    SessionStore,
)


def test_load_persisted_sessions_restores_only_regular(tmp_path):
    async def run():
        backend = BrowserBackend(
            Settings(
                api_keys=["test-token"],
                backend="browser",
                model_id="gpt-a",
                available_models=["gpt-a"],
                available_levels=["high"],
                profile_dir=str(
                    tmp_path / "chrome-profile"
                ),
            )
        )

        store = SessionStore(
            tmp_path / "sessions.json"
        )

        regular = PersistedSession(
            session_id="restored-high",
            model="gpt-a",
            level="high",
            conversation_policy="regular",
            conversation_url=(
                "https://chatgpt.com/c/restored-high"
            ),
        )

        temporary = PersistedSession(
            session_id="temporary",
            model="gpt-a",
            level="high",
            conversation_policy="temporary_personalized",
            conversation_url=(
                "https://chatgpt.com/c/temporary"
            ),
        )

        store.save(
            {
                regular.session_id: regular,
                temporary.session_id: temporary,
            }
        )

        await backend._load_persisted_sessions(
            store
        )

        session = await backend._get_browser_session(
            "restored-high"
        )

        assert session.page is None
        assert session.state == "dormant"
        assert (
            session.conversation_url
            == regular.conversation_url
        )
        assert session.model == "gpt-a"
        assert session.level == "high"

        try:
            await backend._get_browser_session(
                "temporary"
            )
        except KeyError:
            pass
        else:
            raise AssertionError(
                "temporary session was restored"
            )

    asyncio.run(run())


from chatgpt_web_provider.backends import _BrowserSession
from chatgpt_web_provider.models import (
    ChatMessage,
    CompletionResult,
)


class RestoreKeyboard:
    def __init__(self):
        self.pressed = []

    async def press(self, key):
        self.pressed.append(key)


class RestorePage:
    def __init__(self):
        self.goto_calls = []
        self.keyboard = RestoreKeyboard()
        self.closed = False

    async def goto(self, url, **kwargs):
        self.goto_calls.append(
            (url, kwargs)
        )

    async def title(self):
        return "ChatGPT"

    async def close(self):
        self.closed = True


class RestoreContext:
    def __init__(self, page):
        self.page = page
        self.new_page_calls = 0

    async def new_page(self):
        self.new_page_calls += 1
        return self.page


class RestoringBrowserBackend(BrowserBackend):
    def __init__(self, settings):
        super().__init__(settings)

        self.restore_page = RestorePage()
        self.restore_context = RestoreContext(
            self.restore_page
        )
        self.applied_preferences = []
        self.completed_pages = []

    async def _ensure_page(self):
        self._context = self.restore_context
        return object()

    async def _apply_preferences(
        self,
        page,
        model,
        level,
    ):
        self.applied_preferences.append(
            (page, model, level)
        )

    async def _complete_pinned_session(
        self,
        session,
        messages,
    ):
        self.completed_pages.append(
            session.page
        )

        last_user = next(
            (
                message.text()
                for message in reversed(messages)
                if message.role == "user"
            ),
            "",
        )

        return CompletionResult(
            text="answer:" + last_user,
            model=session.model,
            level=session.level,
        )


def test_restorable_conversation_url_validation(
    tmp_path,
):
    backend = BrowserBackend(
        Settings(
            backend="browser",
            model_id="gpt-a",
            available_models=["gpt-a"],
            available_levels=["high"],
            profile_dir=str(
                tmp_path / "chrome-profile"
            ),
        )
    )

    backend._validate_restorable_conversation_url(
        "https://chatgpt.com/c/abc123"
    )

    invalid = [
        "https://example.com/c/abc123",
        "https://chatgpt.com/share/abc123",
        "https://chatgpt.com/",
    ]

    for url in invalid:
        try:
            backend._validate_restorable_conversation_url(
                url
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                "accepted invalid restore URL: "
                + url
            )


def test_dormant_session_restores_once_and_reuses_page(
    tmp_path,
):
    async def run():
        backend = RestoringBrowserBackend(
            Settings(
                backend="browser",
                model_id="gpt-a",
                available_models=["gpt-a"],
                available_levels=["high"],
                profile_dir=str(
                    tmp_path / "chrome-profile"
                ),
            )
        )

        session = _BrowserSession(
            session_id="restored-high",
            model="gpt-a",
            level="high",
            conversation_policy="regular",
            page=None,
            state="dormant",
            conversation_url=(
                "https://chatgpt.com/c/abc123"
            ),
        )

        backend._sessions[
            session.session_id
        ] = session

        first = await backend.complete_session(
            session.session_id,
            [
                ChatMessage(
                    role="user",
                    content="first",
                )
            ],
        )

        second = await backend.complete_session(
            session.session_id,
            [
                ChatMessage(
                    role="user",
                    content="second",
                )
            ],
        )

        assert first.text == "answer:first"
        assert second.text == "answer:second"

        assert (
            backend.restore_context.new_page_calls
            == 1
        )

        assert backend.restore_page.goto_calls[0][0] == (
            "https://chatgpt.com/c/abc123"
        )

        assert backend.applied_preferences == [
            (
                backend.restore_page,
                "gpt-a",
                "high",
            )
        ]

        assert backend.completed_pages == [
            backend.restore_page,
            backend.restore_page,
        ]

        assert session.page is backend.restore_page
        assert session.state == "ready"

    asyncio.run(run())


from chatgpt_web_provider.backends import _BrowserSession
from chatgpt_web_provider.models import (
    ChatMessage,
    CompletionResult,
)


class RestoreKeyboard:
    def __init__(self):
        self.pressed = []

    async def press(self, key):
        self.pressed.append(key)


class RestorePage:
    def __init__(self):
        self.goto_calls = []
        self.keyboard = RestoreKeyboard()
        self.closed = False

    async def goto(self, url, **kwargs):
        self.goto_calls.append(
            (url, kwargs)
        )

    async def title(self):
        return "ChatGPT"

    async def close(self):
        self.closed = True


class RestoreContext:
    def __init__(self, page):
        self.page = page
        self.new_page_calls = 0

    async def new_page(self):
        self.new_page_calls += 1
        return self.page


class RestoringBrowserBackend(BrowserBackend):
    def __init__(self, settings):
        super().__init__(settings)

        self.restore_page = RestorePage()
        self.restore_context = RestoreContext(
            self.restore_page
        )
        self.applied_preferences = []
        self.completed_pages = []

    async def _ensure_page(self):
        self._context = self.restore_context
        return object()

    async def _apply_preferences(
        self,
        page,
        model,
        level,
    ):
        self.applied_preferences.append(
            (page, model, level)
        )

    async def _complete_pinned_session(
        self,
        session,
        messages,
    ):
        self.completed_pages.append(
            session.page
        )

        last_user = next(
            (
                message.text()
                for message in reversed(messages)
                if message.role == "user"
            ),
            "",
        )

        return CompletionResult(
            text="answer:" + last_user,
            model=session.model,
            level=session.level,
        )


def test_restorable_conversation_url_validation(
    tmp_path,
):
    backend = BrowserBackend(
        Settings(
            backend="browser",
            model_id="gpt-a",
            available_models=["gpt-a"],
            available_levels=["high"],
            profile_dir=str(
                tmp_path / "chrome-profile"
            ),
        )
    )

    backend._validate_restorable_conversation_url(
        "https://chatgpt.com/c/abc123"
    )

    invalid = [
        "https://example.com/c/abc123",
        "https://chatgpt.com/share/abc123",
        "https://chatgpt.com/",
    ]

    for url in invalid:
        try:
            backend._validate_restorable_conversation_url(
                url
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                "accepted invalid restore URL: "
                + url
            )


def test_dormant_session_restores_once_and_reuses_page(
    tmp_path,
):
    async def run():
        backend = RestoringBrowserBackend(
            Settings(
                backend="browser",
                model_id="gpt-a",
                available_models=["gpt-a"],
                available_levels=["high"],
                profile_dir=str(
                    tmp_path / "chrome-profile"
                ),
            )
        )

        session = _BrowserSession(
            session_id="restored-high",
            model="gpt-a",
            level="high",
            conversation_policy="regular",
            page=None,
            state="dormant",
            conversation_url=(
                "https://chatgpt.com/c/abc123"
            ),
        )

        backend._sessions[
            session.session_id
        ] = session

        first = await backend.complete_session(
            session.session_id,
            [
                ChatMessage(
                    role="user",
                    content="first",
                )
            ],
        )

        second = await backend.complete_session(
            session.session_id,
            [
                ChatMessage(
                    role="user",
                    content="second",
                )
            ],
        )

        assert first.text == "answer:first"
        assert second.text == "answer:second"

        assert (
            backend.restore_context.new_page_calls
            == 1
        )

        assert backend.restore_page.goto_calls[0][0] == (
            "https://chatgpt.com/c/abc123"
        )

        assert backend.applied_preferences == [
            (
                backend.restore_page,
                "gpt-a",
                "high",
            )
        ]

        assert backend.completed_pages == [
            backend.restore_page,
            backend.restore_page,
        ]

        assert session.page is backend.restore_page
        assert session.state == "ready"

    asyncio.run(run())
