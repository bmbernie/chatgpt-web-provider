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
