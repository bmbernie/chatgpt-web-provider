import os

import pytest

from chatgpt_web_provider.config import Settings


def clear_provider_env(monkeypatch):
    for name in list(os.environ):
        if name.startswith("CHATGPT_WEB_"):
            monkeypatch.delenv(
                name,
                raising=False,
            )


def test_toml_configuration_loads(tmp_path, monkeypatch):
    clear_provider_env(monkeypatch)

    config = tmp_path / "config.toml"
    profile = tmp_path / "browser-profile"

    config.write_text(
        f'''
[provider]
backend = "browser"
api_keys = ["must-not-load"]

[chatgpt]
model = "gpt-test"
models = ["gpt-test", "gpt-other"]
model_labels = {{ "gpt-test" = "Test Model" }}
levels = ["high", "xhigh"]
level_labels = {{ high = "High", xhigh = "Extra High" }}
session_policy = "temporary_unpersonalized"

[server]
host = "0.0.0.0"
port = 9001
public_base_url = "http://example.test:9001"

[browser]
profile_dir = "{profile}"
headless = false
channel = "chrome"
enable_extensions = true

[timeouts]
request_seconds = 111
queue_seconds = 222

[queue]
max_concurrent_requests = 3

[rate_limit]
retry_after_seconds = 333
'''
    )

    settings = Settings.from_env(config)

    assert settings.api_keys == []
    assert settings.backend == "browser"
    assert settings.model_id == "gpt-test"
    assert settings.available_models == [
        "gpt-test",
        "gpt-other",
    ]
    assert settings.model_labels["gpt-test"] == "Test Model"
    assert settings.available_levels == [
        "high",
        "xhigh",
    ]
    assert settings.level_labels["xhigh"] == "Extra High"
    assert settings.session_policy == "temporary_unpersonalized"

    assert settings.host == "0.0.0.0"
    assert settings.port == 9001
    assert settings.public_base_url == "http://example.test:9001"

    assert settings.profile_dir == str(profile)
    assert settings.headless is False
    assert settings.browser_channel == "chrome"
    assert settings.enable_extensions is True

    assert settings.request_timeout_seconds == 111
    assert settings.queue_timeout_seconds == 222
    assert settings.max_concurrent_requests == 3
    assert settings.ui_rate_limit_retry_after_seconds == 333


def test_environment_overrides_toml(tmp_path, monkeypatch):
    clear_provider_env(monkeypatch)

    config = tmp_path / "config.toml"
    config.write_text(
        '''
[chatgpt]
model = "from-config"

[server]
host = "10.0.0.1"
port = 9001

[rate_limit]
retry_after_seconds = 333
'''
    )

    monkeypatch.setenv(
        "CHATGPT_WEB_MODEL",
        "from-env",
    )
    monkeypatch.setenv(
        "CHATGPT_WEB_HOST",
        "127.9.9.9",
    )
    monkeypatch.setenv(
        "CHATGPT_WEB_PORT",
        "9999",
    )
    monkeypatch.setenv(
        "CHATGPT_WEB_UI_RATE_LIMIT_RETRY_AFTER_SECONDS",
        "",
    )

    settings = Settings.from_env(config)

    assert settings.model_id == "from-env"
    assert settings.host == "127.9.9.9"
    assert settings.port == 9999
    assert settings.ui_rate_limit_retry_after_seconds is None


def test_config_path_can_come_from_environment(
    tmp_path,
    monkeypatch,
):
    clear_provider_env(monkeypatch)

    config = tmp_path / "provider.toml"
    config.write_text(
        '''
[server]
port = 9123
'''
    )

    monkeypatch.setenv(
        "CHATGPT_WEB_CONFIG",
        str(config),
    )

    settings = Settings.from_env()

    assert settings.port == 9123


def test_explicit_missing_config_fails(
    tmp_path,
    monkeypatch,
):
    clear_provider_env(monkeypatch)

    with pytest.raises(FileNotFoundError):
        Settings.from_env(
            tmp_path / "missing.toml"
        )


def test_missing_default_config_is_optional(
    tmp_path,
    monkeypatch,
):
    clear_provider_env(monkeypatch)

    monkeypatch.setenv(
        "XDG_CONFIG_HOME",
        str(tmp_path),
    )

    settings = Settings.from_env()

    assert settings.host == "127.0.0.1"
    assert settings.port == 8791
