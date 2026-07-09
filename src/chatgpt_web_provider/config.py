from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_MODEL = "chatgpt-5.6-sol-high-web"
DEFAULT_MODELS = (
    "chatgpt-5.6-sol-high-web",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.3",
    "gpt-5.3-codex-spark",
    "o3",
)
DEFAULT_MODEL_LABELS = {
    "chatgpt-5.6-sol-high-web": "GPT-5.6 Sol",
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


@dataclass(slots=True)
class Settings:
    api_keys: list[str] = field(default_factory=list)
    backend: str = "mock"
    model_id: str = DEFAULT_MODEL
    available_models: list[str] = field(default_factory=lambda: list(DEFAULT_MODELS))
    model_labels: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_MODEL_LABELS))
    available_levels: list[str] = field(default_factory=list)
    level_labels: dict[str, str] = field(default_factory=dict)
    host: str = "127.0.0.1"
    port: int = 8791
    public_base_url: str = "http://127.0.0.1:8791"
    profile_dir: str = str(Path.home() / ".local/share/chatgpt-web-provider/chrome-profile")
    headless: bool = True
    request_timeout_seconds: int = 300
    max_concurrent_requests: int = 1
    queue_timeout_seconds: int = 600

    def __post_init__(self) -> None:
        if not self.available_models:
            self.available_models = [self.model_id]
        elif self.model_id not in self.available_models:
            self.available_models.insert(0, self.model_id)
        if not self.available_levels:
            self.available_levels = ["auto", "fast", "standard", "high"]

    @classmethod
    def from_env(cls) -> "Settings":
        model_id = os.getenv("CHATGPT_WEB_MODEL", DEFAULT_MODEL)
        available_models = _csv(os.getenv("CHATGPT_WEB_MODELS")) or list(DEFAULT_MODELS)
        if model_id not in available_models:
            available_models.insert(0, model_id)
        available_levels = _csv(os.getenv("CHATGPT_WEB_LEVELS")) or ["auto", "fast", "standard", "high"]
        return cls(
            api_keys=_csv(os.getenv("CHATGPT_WEB_API_KEYS")),
            backend=os.getenv("CHATGPT_WEB_BACKEND", "mock").strip().lower() or "mock",
            model_id=model_id,
            available_models=available_models,
            model_labels=_kv_csv(os.getenv("CHATGPT_WEB_MODEL_LABELS")) or dict(DEFAULT_MODEL_LABELS),
            available_levels=available_levels,
            level_labels=_kv_csv(os.getenv("CHATGPT_WEB_LEVEL_LABELS")),
            host=os.getenv("CHATGPT_WEB_HOST", "127.0.0.1"),
            port=int(os.getenv("CHATGPT_WEB_PORT", "8791")),
            public_base_url=os.getenv("CHATGPT_WEB_PUBLIC_BASE_URL", "http://127.0.0.1:8791"),
            profile_dir=os.getenv("CHATGPT_WEB_PROFILE_DIR", str(Path.home() / ".local/share/chatgpt-web-provider/chrome-profile")),
            headless=os.getenv("CHATGPT_WEB_HEADLESS", "true").lower() in {"1", "true", "yes", "on"},
            request_timeout_seconds=int(os.getenv("CHATGPT_WEB_REQUEST_TIMEOUT_SECONDS", "300")),
            max_concurrent_requests=int(os.getenv("CHATGPT_WEB_MAX_CONCURRENT_REQUESTS", "1")),
            queue_timeout_seconds=int(os.getenv("CHATGPT_WEB_QUEUE_TIMEOUT_SECONDS", "600")),
        )

    def validate_for_runtime(self) -> None:
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

    def model_label(self, model: str) -> str:
        return self.model_labels.get(model, model)

    def level_label(self, level: str) -> str:
        return self.level_labels.get(level, level)
