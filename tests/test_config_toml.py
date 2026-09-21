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
base_url = "https://chatgpt.example.test/app/"
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
viewport_width = 1440
viewport_height = 900

[transport]
inline_fill_max_chars = 12000
clipboard_chunk_size = 2048

[timeouts]
request_seconds = 111
queue_seconds = 222
navigation_ms = 44444
composer_ready_ms = 55555
policy_launch_ms = 31000
policy_ready_ms = 11000
reasoning_control_ms = 46000
generation_seconds = 241
generation_poll_ms = 1100
generation_settle_ms = 1600
new_session_settle_ms = 1700
picker_ready_ms = 5100
picker_transition_ms = 5200
ui_action_ms = 3100
submit_ready_ms = 33000
submit_confirm_ms = 5300

[queue]
max_concurrent_requests = 3

[rate_limit]
retry_after_seconds = 333
'''
    )

    settings = Settings.from_env(config)

    assert settings.api_keys == []
    assert settings.backend == "browser"
    assert settings.chatgpt_base_url == (
        "https://chatgpt.example.test/app/"
    )
    assert settings.chatgpt_origin == (
        "https://chatgpt.example.test"
    )
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
    assert settings.viewport_width == 1440
    assert settings.viewport_height == 900

    assert settings.inline_fill_max_chars == 12000
    assert settings.clipboard_chunk_size == 2048
    assert settings.navigation_timeout_ms == 44444
    assert settings.composer_ready_timeout_ms == 55555
    assert settings.policy_launch_timeout_ms == 31000
    assert settings.policy_ready_timeout_ms == 11000
    assert settings.reasoning_control_timeout_ms == 46000
    assert settings.generation_timeout_seconds == 241
    assert settings.generation_poll_ms == 1100
    assert settings.generation_settle_ms == 1600
    assert settings.new_session_settle_ms == 1700
    assert settings.picker_ready_timeout_ms == 5100
    assert settings.picker_transition_timeout_ms == 5200
    assert settings.ui_action_timeout_ms == 3100
    assert settings.submit_ready_timeout_ms == 33000
    assert settings.submit_confirm_timeout_ms == 5300

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


def test_browser_environment_overrides_toml(
    tmp_path,
    monkeypatch,
):
    clear_provider_env(monkeypatch)

    config = tmp_path / "config.toml"
    config.write_text(
        '''
[chatgpt]
base_url = "https://from-config.example/"

[browser]
viewport_width = 1000
viewport_height = 700

[timeouts]
navigation_ms = 40000
composer_ready_ms = 20000
'''
    )

    monkeypatch.setenv(
        "CHATGPT_WEB_CHATGPT_BASE_URL",
        "https://from-env.example/path/",
    )
    monkeypatch.setenv(
        "CHATGPT_WEB_VIEWPORT_WIDTH",
        "1600",
    )
    monkeypatch.setenv(
        "CHATGPT_WEB_VIEWPORT_HEIGHT",
        "1000",
    )
    monkeypatch.setenv(
        "CHATGPT_WEB_NAVIGATION_TIMEOUT_MS",
        "65000",
    )
    monkeypatch.setenv(
        "CHATGPT_WEB_COMPOSER_READY_TIMEOUT_MS",
        "35000",
    )
    monkeypatch.setenv(
        "CHATGPT_WEB_POLICY_LAUNCH_TIMEOUT_MS",
        "32000",
    )
    monkeypatch.setenv(
        "CHATGPT_WEB_POLICY_READY_TIMEOUT_MS",
        "12000",
    )
    monkeypatch.setenv(
        "CHATGPT_WEB_REASONING_CONTROL_TIMEOUT_MS",
        "47000",
    )
    monkeypatch.setenv(
        "CHATGPT_WEB_GENERATION_TIMEOUT_SECONDS",
        "242",
    )
    monkeypatch.setenv(
        "CHATGPT_WEB_GENERATION_POLL_MS",
        "1200",
    )
    monkeypatch.setenv(
        "CHATGPT_WEB_GENERATION_SETTLE_MS",
        "1800",
    )
    monkeypatch.setenv(
        "CHATGPT_WEB_NEW_SESSION_SETTLE_MS",
        "1900",
    )
    monkeypatch.setenv(
        "CHATGPT_WEB_PICKER_READY_TIMEOUT_MS",
        "6100",
    )
    monkeypatch.setenv(
        "CHATGPT_WEB_PICKER_TRANSITION_TIMEOUT_MS",
        "6200",
    )
    monkeypatch.setenv(
        "CHATGPT_WEB_UI_ACTION_TIMEOUT_MS",
        "4100",
    )
    monkeypatch.setenv(
        "CHATGPT_WEB_SUBMIT_READY_TIMEOUT_MS",
        "34000",
    )
    monkeypatch.setenv(
        "CHATGPT_WEB_SUBMIT_CONFIRM_TIMEOUT_MS",
        "6300",
    )

    settings = Settings.from_env(config)

    assert settings.chatgpt_base_url == (
        "https://from-env.example/path/"
    )
    assert settings.chatgpt_origin == (
        "https://from-env.example"
    )
    assert settings.viewport_width == 1600
    assert settings.viewport_height == 1000
    assert settings.navigation_timeout_ms == 65000
    assert settings.composer_ready_timeout_ms == 35000
    assert settings.policy_launch_timeout_ms == 32000
    assert settings.policy_ready_timeout_ms == 12000
    assert settings.reasoning_control_timeout_ms == 47000
    assert settings.generation_timeout_seconds == 242
    assert settings.generation_poll_ms == 1200
    assert settings.generation_settle_ms == 1800
    assert settings.new_session_settle_ms == 1900
    assert settings.picker_ready_timeout_ms == 6100
    assert settings.picker_transition_timeout_ms == 6200
    assert settings.ui_action_timeout_ms == 4100
    assert settings.submit_ready_timeout_ms == 34000
    assert settings.submit_confirm_timeout_ms == 6300
