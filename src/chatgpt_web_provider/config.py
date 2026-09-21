from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit


DEFAULT_MODEL = "chatgpt-5.6-sol-web"
DEFAULT_MODELS = (
    "chatgpt-5.6-sol-web",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.3",
    "gpt-5.3-codex-spark",
    "o3",
)
SESSION_POLICIES = (
    "regular",
    "temporary_personalized",
    "temporary_unpersonalized",
)

DEFAULT_SESSION_POLICY = "regular"


DEFAULT_MODEL_LABELS = {
    "chatgpt-5.6-sol-web": "GPT-5.6 Sol",
    "gpt-5.6-terra": "5.6 Terra",
    "gpt-5.6-luna": "5.6 Luna",
    "gpt-5.5": "GPT-5.5",
    "gpt-5.4": "GPT-5.4",
    "gpt-5.4-mini": "5.4 Mini",
    "gpt-5.3": "GPT-5.3",
    "gpt-5.3-codex-spark": "5.3 Codex Spark",
    "o3": "o3",
}


def _csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _kv_csv(value: str | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for part in _csv(value):
        if "=" in part:
            key, val = part.split("=", 1)
            key = key.strip()
            val = val.strip()
            if key and val:
                result[key] = val
    return result


def _default_config_path() -> Path:
    xdg_config_home = os.getenv("XDG_CONFIG_HOME")

    base = (
        Path(xdg_config_home).expanduser()
        if xdg_config_home
        else Path.home() / ".config"
    )

    return (
        base
        / "chatgpt-web-provider"
        / "config.toml"
    )


def _load_toml_config(
    config_path: str | Path | None = None,
) -> dict:
    explicit = config_path is not None

    if config_path is None:
        env_path = os.getenv(
            "CHATGPT_WEB_CONFIG",
            "",
        ).strip()

        if env_path:
            path = Path(env_path).expanduser()
            explicit = True
        else:
            path = _default_config_path()
    else:
        path = Path(config_path).expanduser()

    if not path.exists():
        if explicit:
            raise FileNotFoundError(
                f"ChatGPT Web Provider config not found: {path}"
            )

        return {}

    with path.open("rb") as handle:
        data = tomllib.load(handle)

    if not isinstance(data, dict):
        raise ValueError(
            f"invalid TOML configuration root: {path}"
        )

    return data


def _config_section(
    config: dict,
    name: str,
) -> dict:
    value = config.get(name)

    if value is None:
        return {}

    if not isinstance(value, dict):
        raise ValueError(
            f"config section [{name}] must be a table"
        )

    return value


def _toml_str_list(
    value,
    default,
) -> list[str]:
    if value is None:
        return list(default)

    if not isinstance(value, list) or not all(
        isinstance(item, str)
        for item in value
    ):
        raise ValueError(
            "expected a TOML array of strings"
        )

    return [
        item.strip()
        for item in value
        if item.strip()
    ]


def _toml_str_map(
    value,
    default,
) -> dict[str, str]:
    if value is None:
        return dict(default)

    if not isinstance(value, dict) or not all(
        isinstance(key, str)
        and isinstance(item, str)
        for key, item in value.items()
    ):
        raise ValueError(
            "expected a TOML table of string values"
        )

    return {
        key: item
        for key, item in value.items()
    }


def _toml_bool(
    value,
    default: bool,
    *,
    name: str,
) -> bool:
    if value is None:
        return default

    if not isinstance(value, bool):
        raise ValueError(
            f"{name} must be true or false"
        )

    return value


def _env_bool(
    name: str,
    default: bool,
) -> bool:
    raw = os.getenv(name)

    if raw is None:
        return default

    return raw.lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@dataclass(slots=True)
class Settings:
    api_keys: list[str] = field(default_factory=list)
    backend: str = "mock"
    model_id: str = DEFAULT_MODEL
    available_models: list[str] = field(default_factory=lambda: list(DEFAULT_MODELS))
    model_labels: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_MODEL_LABELS))
    available_levels: list[str] = field(default_factory=list)
    level_labels: dict[str, str] = field(default_factory=dict)
    chatgpt_base_url: str = "https://chatgpt.com/"
    host: str = "127.0.0.1"
    port: int = 8791
    public_base_url: str = "http://127.0.0.1:8791"
    profile_dir: str = str(Path.home() / ".local/share/chatgpt-web-provider/chrome-profile")
    headless: bool = True
    browser_channel: str | None = None
    enable_extensions: bool = False
    viewport_width: int = 1360
    viewport_height: int = 820
    inline_fill_max_chars: int = 16_384
    clipboard_chunk_size: int = 4_096
    navigation_timeout_ms: int = 60_000
    composer_ready_timeout_ms: int = 30_000
    policy_launch_timeout_ms: int = 30_000
    policy_ready_timeout_ms: int = 10_000
    reasoning_control_timeout_ms: int = 45_000
    generation_timeout_seconds: int = 240
    generation_poll_ms: int = 1_000
    generation_settle_ms: int = 1_500
    new_session_settle_ms: int = 1_500
    picker_ready_timeout_ms: int = 5_000
    picker_transition_timeout_ms: int = 5_000
    ui_action_timeout_ms: int = 3_000
    submit_ready_timeout_ms: int = 30_000
    submit_confirm_timeout_ms: int = 5_000
    extraction_timeout_ms: int = 5_000
    request_timeout_seconds: int = 300
    max_concurrent_requests: int = 1
    queue_timeout_seconds: int = 600
    ui_rate_limit_retry_after_seconds: int | None = None
    session_policy: str = DEFAULT_SESSION_POLICY

    def __post_init__(self) -> None:
        if not self.available_models:
            self.available_models = [self.model_id]
        elif self.model_id not in self.available_models:
            self.available_models.insert(0, self.model_id)
        if not self.available_levels:
            self.available_levels = ["auto", "fast", "standard", "high"]

        if self.session_policy not in SESSION_POLICIES:
            raise ValueError(
                "CHATGPT_WEB_SESSION_POLICY must be one of: "
                + ", ".join(SESSION_POLICIES)
            )

    @property
    def chatgpt_origin(self) -> str:
        """Origin used for browser permissions such as clipboard access."""
        parsed = urlsplit(self.chatgpt_base_url)

        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
        ):
            raise ValueError(
                "CHATGPT_WEB_CHATGPT_BASE_URL / "
                "chatgpt.base_url must be an absolute HTTP(S) URL"
            )

        return f"{parsed.scheme}://{parsed.netloc}"

    @classmethod
    def from_env(
        cls,
        config_path: str | Path | None = None,
    ) -> "Settings":
        config = _load_toml_config(config_path)

        provider = _config_section(
            config,
            "provider",
        )
        chatgpt = _config_section(
            config,
            "chatgpt",
        )
        server = _config_section(
            config,
            "server",
        )
        browser = _config_section(
            config,
            "browser",
        )
        transport = _config_section(
            config,
            "transport",
        )
        timeouts = _config_section(
            config,
            "timeouts",
        )
        queue = _config_section(
            config,
            "queue",
        )
        rate_limit = _config_section(
            config,
            "rate_limit",
        )

        model_env = os.getenv(
            "CHATGPT_WEB_MODEL"
        )
        model_id = (
            model_env
            if model_env is not None
            else str(
                chatgpt.get(
                    "model",
                    DEFAULT_MODEL,
                )
            )
        )

        models_env = os.getenv(
            "CHATGPT_WEB_MODELS"
        )

        if models_env is not None:
            available_models = (
                _csv(models_env)
                or list(DEFAULT_MODELS)
            )
        else:
            available_models = _toml_str_list(
                chatgpt.get("models"),
                DEFAULT_MODELS,
            )

        if model_id not in available_models:
            available_models.insert(
                0,
                model_id,
            )

        model_labels_env = os.getenv(
            "CHATGPT_WEB_MODEL_LABELS"
        )

        if model_labels_env is not None:
            model_labels = (
                _kv_csv(model_labels_env)
                or dict(DEFAULT_MODEL_LABELS)
            )
        else:
            model_labels = _toml_str_map(
                chatgpt.get("model_labels"),
                DEFAULT_MODEL_LABELS,
            )

        levels_default = [
            "auto",
            "fast",
            "standard",
            "high",
        ]

        levels_env = os.getenv(
            "CHATGPT_WEB_LEVELS"
        )

        if levels_env is not None:
            available_levels = (
                _csv(levels_env)
                or levels_default
            )
        else:
            available_levels = _toml_str_list(
                chatgpt.get("levels"),
                levels_default,
            )

        level_labels_env = os.getenv(
            "CHATGPT_WEB_LEVEL_LABELS"
        )

        if level_labels_env is not None:
            level_labels = _kv_csv(
                level_labels_env
            )
        else:
            level_labels = _toml_str_map(
                chatgpt.get("level_labels"),
                {},
            )

        retry_after_env = os.getenv(
            "CHATGPT_WEB_UI_RATE_LIMIT_RETRY_AFTER_SECONDS"
        )

        if retry_after_env is not None:
            ui_rate_limit_retry_after_seconds = (
                int(retry_after_env)
                if retry_after_env.strip()
                else None
            )
        else:
            retry_after_config = rate_limit.get(
                "retry_after_seconds"
            )

            ui_rate_limit_retry_after_seconds = (
                int(retry_after_config)
                if retry_after_config is not None
                else None
            )

        profile_dir_default = browser.get(
            "profile_dir",
            str(
                Path.home()
                / ".local/share/"
                "chatgpt-web-provider/"
                "chrome-profile"
            ),
        )

        profile_dir = str(
            Path(
                os.getenv(
                    "CHATGPT_WEB_PROFILE_DIR",
                    str(profile_dir_default),
                )
            ).expanduser()
        )

        channel_env = os.getenv(
            "CHATGPT_WEB_BROWSER_CHANNEL"
        )

        channel_value = (
            channel_env
            if channel_env is not None
            else browser.get("channel")
        )

        browser_channel = (
            str(channel_value).strip()
            if channel_value is not None
            else ""
        ) or None

        headless_default = _toml_bool(
            browser.get("headless"),
            True,
            name="browser.headless",
        )

        extensions_default = _toml_bool(
            browser.get(
                "enable_extensions"
            ),
            False,
            name="browser.enable_extensions",
        )

        backend_value = os.getenv(
            "CHATGPT_WEB_BACKEND"
        )

        if backend_value is None:
            backend_value = provider.get(
                "backend",
                "mock",
            )

        session_policy_value = os.getenv(
            "CHATGPT_WEB_SESSION_POLICY"
        )

        if session_policy_value is None:
            session_policy_value = (
                chatgpt.get(
                    "session_policy",
                    DEFAULT_SESSION_POLICY,
                )
            )

        # API keys intentionally remain environment-only.
        return cls(
            api_keys=_csv(
                os.getenv(
                    "CHATGPT_WEB_API_KEYS"
                )
            ),
            backend=(
                str(backend_value)
                .strip()
                .lower()
                or "mock"
            ),
            model_id=model_id,
            available_models=available_models,
            model_labels=model_labels,
            available_levels=available_levels,
            level_labels=level_labels,
            chatgpt_base_url=os.getenv(
                "CHATGPT_WEB_CHATGPT_BASE_URL",
                str(
                    chatgpt.get(
                        "base_url",
                        "https://chatgpt.com/",
                    )
                ),
            ),
            host=os.getenv(
                "CHATGPT_WEB_HOST",
                str(
                    server.get(
                        "host",
                        "127.0.0.1",
                    )
                ),
            ),
            port=int(
                os.getenv(
                    "CHATGPT_WEB_PORT",
                    str(
                        server.get(
                            "port",
                            8791,
                        )
                    ),
                )
            ),
            public_base_url=os.getenv(
                "CHATGPT_WEB_PUBLIC_BASE_URL",
                str(
                    server.get(
                        "public_base_url",
                        "http://127.0.0.1:8791",
                    )
                ),
            ),
            profile_dir=profile_dir,
            headless=_env_bool(
                "CHATGPT_WEB_HEADLESS",
                headless_default,
            ),
            browser_channel=browser_channel,
            enable_extensions=_env_bool(
                "CHATGPT_WEB_ENABLE_EXTENSIONS",
                extensions_default,
            ),
            viewport_width=int(
                os.getenv(
                    "CHATGPT_WEB_VIEWPORT_WIDTH",
                    str(
                        browser.get(
                            "viewport_width",
                            1360,
                        )
                    ),
                )
            ),
            viewport_height=int(
                os.getenv(
                    "CHATGPT_WEB_VIEWPORT_HEIGHT",
                    str(
                        browser.get(
                            "viewport_height",
                            820,
                        )
                    ),
                )
            ),
            inline_fill_max_chars=int(
                os.getenv(
                    "CHATGPT_WEB_INLINE_FILL_MAX_CHARS",
                    str(
                        transport.get(
                            "inline_fill_max_chars",
                            16_384,
                        )
                    ),
                )
            ),
            clipboard_chunk_size=int(
                os.getenv(
                    "CHATGPT_WEB_CLIPBOARD_CHUNK_SIZE",
                    str(
                        transport.get(
                            "clipboard_chunk_size",
                            4_096,
                        )
                    ),
                )
            ),
            navigation_timeout_ms=int(
                os.getenv(
                    "CHATGPT_WEB_NAVIGATION_TIMEOUT_MS",
                    str(
                        timeouts.get(
                            "navigation_ms",
                            60_000,
                        )
                    ),
                )
            ),
            composer_ready_timeout_ms=int(
                os.getenv(
                    "CHATGPT_WEB_COMPOSER_READY_TIMEOUT_MS",
                    str(
                        timeouts.get(
                            "composer_ready_ms",
                            30_000,
                        )
                    ),
                )
            ),
            policy_launch_timeout_ms=int(
                os.getenv(
                    "CHATGPT_WEB_POLICY_LAUNCH_TIMEOUT_MS",
                    str(
                        timeouts.get(
                            "policy_launch_ms",
                            30_000,
                        )
                    ),
                )
            ),
            policy_ready_timeout_ms=int(
                os.getenv(
                    "CHATGPT_WEB_POLICY_READY_TIMEOUT_MS",
                    str(
                        timeouts.get(
                            "policy_ready_ms",
                            10_000,
                        )
                    ),
                )
            ),
            reasoning_control_timeout_ms=int(
                os.getenv(
                    "CHATGPT_WEB_REASONING_CONTROL_TIMEOUT_MS",
                    str(
                        timeouts.get(
                            "reasoning_control_ms",
                            45_000,
                        )
                    ),
                )
            ),
            generation_timeout_seconds=int(
                os.getenv(
                    "CHATGPT_WEB_GENERATION_TIMEOUT_SECONDS",
                    str(
                        timeouts.get(
                            "generation_seconds",
                            240,
                        )
                    ),
                )
            ),
            generation_poll_ms=int(
                os.getenv(
                    "CHATGPT_WEB_GENERATION_POLL_MS",
                    str(
                        timeouts.get(
                            "generation_poll_ms",
                            1_000,
                        )
                    ),
                )
            ),
            generation_settle_ms=int(
                os.getenv(
                    "CHATGPT_WEB_GENERATION_SETTLE_MS",
                    str(
                        timeouts.get(
                            "generation_settle_ms",
                            1_500,
                        )
                    ),
                )
            ),
            new_session_settle_ms=int(
                os.getenv(
                    "CHATGPT_WEB_NEW_SESSION_SETTLE_MS",
                    str(
                        timeouts.get(
                            "new_session_settle_ms",
                            1_500,
                        )
                    ),
                )
            ),
            picker_ready_timeout_ms=int(
                os.getenv(
                    "CHATGPT_WEB_PICKER_READY_TIMEOUT_MS",
                    str(
                        timeouts.get(
                            "picker_ready_ms",
                            5_000,
                        )
                    ),
                )
            ),
            picker_transition_timeout_ms=int(
                os.getenv(
                    "CHATGPT_WEB_PICKER_TRANSITION_TIMEOUT_MS",
                    str(
                        timeouts.get(
                            "picker_transition_ms",
                            5_000,
                        )
                    ),
                )
            ),
            ui_action_timeout_ms=int(
                os.getenv(
                    "CHATGPT_WEB_UI_ACTION_TIMEOUT_MS",
                    str(
                        timeouts.get(
                            "ui_action_ms",
                            3_000,
                        )
                    ),
                )
            ),
            submit_ready_timeout_ms=int(
                os.getenv(
                    "CHATGPT_WEB_SUBMIT_READY_TIMEOUT_MS",
                    str(
                        timeouts.get(
                            "submit_ready_ms",
                            30_000,
                        )
                    ),
                )
            ),
            submit_confirm_timeout_ms=int(
                os.getenv(
                    "CHATGPT_WEB_SUBMIT_CONFIRM_TIMEOUT_MS",
                    str(
                        timeouts.get(
                            "submit_confirm_ms",
                            5_000,
                        )
                    ),
                )
            ),
            extraction_timeout_ms=int(
                os.getenv(
                    "CHATGPT_WEB_EXTRACTION_TIMEOUT_MS",
                    str(
                        timeouts.get(
                            "extraction_ms",
                            5_000,
                        )
                    ),
                )
            ),
            request_timeout_seconds=int(
                os.getenv(
                    "CHATGPT_WEB_REQUEST_TIMEOUT_SECONDS",
                    str(
                        timeouts.get(
                            "request_seconds",
                            300,
                        )
                    ),
                )
            ),
            max_concurrent_requests=int(
                os.getenv(
                    "CHATGPT_WEB_MAX_CONCURRENT_REQUESTS",
                    str(
                        queue.get(
                            "max_concurrent_requests",
                            1,
                        )
                    ),
                )
            ),
            queue_timeout_seconds=int(
                os.getenv(
                    "CHATGPT_WEB_QUEUE_TIMEOUT_SECONDS",
                    str(
                        timeouts.get(
                            "queue_seconds",
                            600,
                        )
                    ),
                )
            ),
            ui_rate_limit_retry_after_seconds=(
                ui_rate_limit_retry_after_seconds
            ),
            session_policy=(
                str(session_policy_value)
                .strip()
                .lower()
                or DEFAULT_SESSION_POLICY
            ),
        )

    def validate_for_runtime(self) -> None:
        # Validate the configured browser target and browser-level limits.
        _ = self.chatgpt_origin

        if self.viewport_width < 1:
            raise ValueError(
                "CHATGPT_WEB_VIEWPORT_WIDTH must be >= 1"
            )
        if self.viewport_height < 1:
            raise ValueError(
                "CHATGPT_WEB_VIEWPORT_HEIGHT must be >= 1"
            )
        if self.navigation_timeout_ms < 1:
            raise ValueError(
                "CHATGPT_WEB_NAVIGATION_TIMEOUT_MS must be >= 1"
            )
        if self.composer_ready_timeout_ms < 1:
            raise ValueError(
                "CHATGPT_WEB_COMPOSER_READY_TIMEOUT_MS must be >= 1"
            )
        if self.policy_launch_timeout_ms < 1:
            raise ValueError(
                "CHATGPT_WEB_POLICY_LAUNCH_TIMEOUT_MS must be >= 1"
            )
        if self.policy_ready_timeout_ms < 1:
            raise ValueError(
                "CHATGPT_WEB_POLICY_READY_TIMEOUT_MS must be >= 1"
            )
        if self.reasoning_control_timeout_ms < 1:
            raise ValueError(
                "CHATGPT_WEB_REASONING_CONTROL_TIMEOUT_MS must be >= 1"
            )
        if self.generation_timeout_seconds < 1:
            raise ValueError(
                "CHATGPT_WEB_GENERATION_TIMEOUT_SECONDS must be >= 1"
            )
        if self.generation_poll_ms < 1:
            raise ValueError(
                "CHATGPT_WEB_GENERATION_POLL_MS must be >= 1"
            )
        if self.generation_settle_ms < 1:
            raise ValueError(
                "CHATGPT_WEB_GENERATION_SETTLE_MS must be >= 1"
            )
        if self.new_session_settle_ms < 1:
            raise ValueError(
                "CHATGPT_WEB_NEW_SESSION_SETTLE_MS must be >= 1"
            )
        if self.picker_ready_timeout_ms < 1:
            raise ValueError(
                "CHATGPT_WEB_PICKER_READY_TIMEOUT_MS must be >= 1"
            )
        if self.picker_transition_timeout_ms < 1:
            raise ValueError(
                "CHATGPT_WEB_PICKER_TRANSITION_TIMEOUT_MS must be >= 1"
            )
        if self.ui_action_timeout_ms < 1:
            raise ValueError(
                "CHATGPT_WEB_UI_ACTION_TIMEOUT_MS must be >= 1"
            )
        if self.submit_ready_timeout_ms < 1:
            raise ValueError(
                "CHATGPT_WEB_SUBMIT_READY_TIMEOUT_MS must be >= 1"
            )
        if self.submit_confirm_timeout_ms < 1:
            raise ValueError(
                "CHATGPT_WEB_SUBMIT_CONFIRM_TIMEOUT_MS must be >= 1"
            )
        if self.extraction_timeout_ms < 1:
            raise ValueError(
                "CHATGPT_WEB_EXTRACTION_TIMEOUT_MS must be >= 1"
            )
        if not self.api_keys:
            raise ValueError("CHATGPT_WEB_API_KEYS must contain at least one API key")
        if self.backend not in {"mock", "browser"}:
            raise ValueError("CHATGPT_WEB_BACKEND must be 'mock' or 'browser'")
        if not self.available_models:
            raise ValueError("CHATGPT_WEB_MODELS must contain at least one model")
        if self.model_id not in self.available_models:
            raise ValueError("CHATGPT_WEB_MODEL must be included in CHATGPT_WEB_MODELS")
        if not self.available_levels:
            raise ValueError("CHATGPT_WEB_LEVELS must contain at least one level")
        if self.max_concurrent_requests < 1:
            raise ValueError("CHATGPT_WEB_MAX_CONCURRENT_REQUESTS must be >= 1")
        if self.queue_timeout_seconds < 1:
            raise ValueError("CHATGPT_WEB_QUEUE_TIMEOUT_SECONDS must be >= 1")

        if (
            self.ui_rate_limit_retry_after_seconds is not None
            and self.ui_rate_limit_retry_after_seconds < 1
        ):
            raise ValueError(
                "CHATGPT_WEB_UI_RATE_LIMIT_RETRY_AFTER_SECONDS "
                "must be >= 1 when configured"
            )

    def model_label(self, model: str) -> str:
        return self.model_labels.get(model, model)

    def level_label(self, level: str) -> str:
        return self.level_labels.get(level, level)
