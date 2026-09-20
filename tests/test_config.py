from chatgpt_web_provider.config import DEFAULT_MODEL_LABELS, DEFAULT_MODELS, Settings


def test_default_model_catalog_is_current(monkeypatch):
    for name in ("CHATGPT_WEB_MODEL", "CHATGPT_WEB_MODELS", "CHATGPT_WEB_MODEL_LABELS"):
        monkeypatch.delenv(name, raising=False)

    settings = Settings.from_env()

    assert settings.model_id == "chatgpt-5.6-sol-web"
    assert settings.available_models == [
        "chatgpt-5.6-sol-web",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.5",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.3",
        "gpt-5.3-codex-spark",
        "o3",
    ]
    assert settings.available_models == list(DEFAULT_MODELS)
    assert settings.model_labels == DEFAULT_MODEL_LABELS


def test_session_policy_defaults_to_regular(monkeypatch):
    monkeypatch.delenv(
        "CHATGPT_WEB_SESSION_POLICY",
        raising=False,
    )

    settings = Settings.from_env()

    assert settings.session_policy == "regular"


def test_session_policy_from_environment(monkeypatch):
    monkeypatch.setenv(
        "CHATGPT_WEB_SESSION_POLICY",
        "temporary_unpersonalized",
    )

    settings = Settings.from_env()

    assert (
        settings.session_policy
        == "temporary_unpersonalized"
    )


def test_invalid_session_policy_rejected(monkeypatch):
    monkeypatch.setenv(
        "CHATGPT_WEB_SESSION_POLICY",
        "mystery",
    )

    try:
        Settings.from_env()
    except ValueError as exc:
        assert "CHATGPT_WEB_SESSION_POLICY" in str(exc)
    else:
        raise AssertionError(
            "invalid session policy was accepted"
        )
