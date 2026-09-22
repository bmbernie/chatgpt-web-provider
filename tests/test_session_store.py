import json
import stat

import pytest

from chatgpt_web_provider.session_store import (
    PersistedSession,
    SessionStore,
)


def record(
    session_id="worker-1",
):
    return PersistedSession(
        session_id=session_id,
        model="chatgpt-5.6-sol-web",
        level="high",
        conversation_policy="regular",
        conversation_url=(
            "https://chatgpt.com/c/"
            + session_id
        ),
    )


def test_session_store_round_trip(
    tmp_path,
):
    path = (
        tmp_path
        / "provider"
        / "sessions.json"
    )
    store = SessionStore(path)

    assert store.load() == {}

    store.upsert(
        record()
    )

    assert store.load() == {
        "worker-1": record()
    }

    mode = stat.S_IMODE(
        path.stat().st_mode
    )

    assert mode == 0o600


def test_session_store_preserves_multiple_sessions_and_delete(
    tmp_path,
):
    path = tmp_path / "sessions.json"
    store = SessionStore(path)

    first = record("first")
    second = record("second")

    store.upsert(first)
    store.upsert(second)

    assert store.load() == {
        "first": first,
        "second": second,
    }

    store.delete("first")

    assert store.load() == {
        "second": second,
    }


def test_session_store_rejects_version_mismatch(
    tmp_path,
):
    path = tmp_path / "sessions.json"

    path.write_text(
        json.dumps(
            {
                "version": 999,
                "sessions": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=(
            "unsupported session-state "
            "version"
        ),
    ):
        SessionStore(path).load()
