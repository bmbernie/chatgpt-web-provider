import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from chatgpt_web_provider.app import create_app
from chatgpt_web_provider.backends import Backend
from chatgpt_web_provider.config import Settings
from chatgpt_web_provider.models import ChatMessage, CompletionResult


@pytest.fixture()
def client():
    app = create_app(Settings(api_keys=["test-token"], backend="mock", public_base_url="https://codex.guber.dev"))
    return TestClient(app)


def test_health_is_public(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert r.json()["backend"] == "mock"


def test_root_redirects_to_docs(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 308
    assert r.headers["location"] == "/docs"


def test_models_requires_bearer_token(client):
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers={"Authorization": "Bearer wrong"}).status_code == 403


def test_models_returns_openai_compatible_shape(client):
    r = client.get("/v1/models", headers={"Authorization": "Bearer test-token"})
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "list"
    assert data["data"][0]["id"] == "chatgpt-5.6-sol-web"


def test_models_accepts_x_api_key_header(client):
    r = client.get("/v1/models", headers={"X-API-Key": "test-token"})
    assert r.status_code == 200
    assert r.json()["data"][0]["id"] == "chatgpt-5.6-sol-web"


def test_auth_uses_valid_bearer_when_x_api_key_placeholder_is_unresolved(client):
    r = client.get(
        "/v1/models",
        headers={"X-API-Key": "{{api_key}}", "Authorization": "Bearer test-token"},
    )
    assert r.status_code == 200
    assert r.json()["data"][0]["id"] == "chatgpt-5.6-sol-web"


def test_provider_status_requires_auth_and_reports_queue(client):
    assert client.get("/v1/provider/status").status_code == 401
    r = client.get("/v1/provider/status", headers={"Authorization": "Bearer test-token"})
    assert r.status_code == 200
    data = r.json()
    assert data["backend"] == "mock"
    assert data["model"] == "chatgpt-5.6-sol-web"
    assert data["queue"]["max_concurrent_requests"] == 1


def test_chat_completions_non_stream(client):
    payload = {
        "model": "chatgpt-5.6-sol-web",
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Say pong"},
        ],
    }
    r = client.post("/v1/chat/completions", json=payload, headers={"Authorization": "Bearer test-token"})
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "chatgpt-5.6-sol-web"
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert "Say pong" in data["choices"][0]["message"]["content"]
    assert data["choices"][0]["finish_reason"] == "stop"


def test_responses_endpoint_returns_openai_responses_like_shape(client):
    r = client.post(
        "/v1/responses",
        json={"model": "chatgpt-5.6-sol-web", "input": "hello"},
        headers={"Authorization": "Bearer test-token"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "response"
    assert data["model"] == "chatgpt-5.6-sol-web"
    assert data["output"][0]["content"][0]["type"] == "output_text"
    assert "hello" in data["output_text"]


def test_chat_completions_stream_returns_openai_sse_chunks(client):
    r = client.post(
        "/v1/chat/completions",
        json={"model": "chatgpt-5.6-sol-web", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        headers={"Authorization": "Bearer test-token"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    body = r.text
    assert "chat.completion.chunk" in body
    assert "[mock:chatgpt-5.6-sol-web] hi" in body
    assert "data: [DONE]" in body


class SlowBackend(Backend):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.active = 0
        self.max_active = 0

    async def complete(self, messages: list[ChatMessage], model: str | None = None, new_session: bool = False, level: str | None = None) -> CompletionResult:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.05)
        self.active -= 1
        return CompletionResult(text="ok", model=model or self.settings.model_id)


def test_requests_are_queued_by_default():
    async def run():
        settings = Settings(api_keys=["test-token"], backend="mock", max_concurrent_requests=1)
        backend = SlowBackend(settings)
        app = create_app(settings, backend=backend)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            payload = {"model": "chatgpt-5.6-sol-web", "messages": [{"role": "user", "content": "hi"}]}
            responses = await asyncio.gather(
                ac.post("/v1/chat/completions", json=payload, headers={"Authorization": "Bearer test-token"}),
                ac.post("/v1/chat/completions", json=payload, headers={"Authorization": "Bearer test-token"}),
            )
        assert [r.status_code for r in responses] == [200, 200]
        assert backend.max_active == 1

    asyncio.run(run())


class SessionRecordingBackend(Backend):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.new_session_values: list[bool] = []

    async def complete(self, messages: list[ChatMessage], model: str | None = None, new_session: bool = False, level: str | None = None) -> CompletionResult:
        self.new_session_values.append(new_session)
        return CompletionResult(text="ok", model=model or self.settings.model_id)


def test_chat_completions_can_request_new_session_via_body_or_header():
    settings = Settings(api_keys=["test-token"], backend="mock")
    backend = SessionRecordingBackend(settings)
    app = create_app(settings, backend=backend)
    client = TestClient(app)
    payload = {"model": "chatgpt-5.6-sol-web", "messages": [{"role": "user", "content": "hi"}]}

    assert client.post("/v1/chat/completions", json={**payload, "new_session": True}, headers={"Authorization": "Bearer test-token"}).status_code == 200
    assert client.post("/v1/chat/completions", json=payload, headers={"Authorization": "Bearer test-token", "X-New-Session": "true"}).status_code == 200

    assert backend.new_session_values == [True, True]


def test_responses_can_request_new_session_via_body_or_header():
    settings = Settings(api_keys=["test-token"], backend="mock")
    backend = SessionRecordingBackend(settings)
    app = create_app(settings, backend=backend)
    client = TestClient(app)

    assert client.post("/v1/responses", json={"model": "chatgpt-5.6-sol-web", "input": "hi", "new_session": True}, headers={"Authorization": "Bearer test-token"}).status_code == 200
    assert client.post("/v1/responses", json={"model": "chatgpt-5.6-sol-web", "input": "hi"}, headers={"Authorization": "Bearer test-token", "X-New-Session": "1"}).status_code == 200

    assert backend.new_session_values == [True, True]

def test_models_lists_all_configured_models_and_capabilities_lists_levels():
    settings = Settings(
        api_keys=["test-token"],
        backend="mock",
        model_id="gpt-a",
        available_models=["gpt-a", "gpt-b"],
        model_labels={"gpt-a": "GPT A", "gpt-b": "GPT B"},
        available_levels=["auto", "high"],
        level_labels={"auto": "Auto", "high": "High"},
    )
    client = TestClient(create_app(settings))
    headers = {"Authorization": "Bearer test-token"}

    models = client.get("/v1/models", headers=headers).json()["data"]
    assert [m["id"] for m in models] == ["gpt-a", "gpt-b"]
    assert models[0]["display_name"] == "GPT A"
    assert models[0]["default"] is True

    caps = client.get("/v1/provider/capabilities", headers=headers).json()
    assert [m["id"] for m in caps["models"]] == ["gpt-a", "gpt-b"]
    assert [l["id"] for l in caps["levels"]] == ["auto", "high"]
    assert caps["new_session"]["header"] == "X-New-Session"


def test_chat_completions_validates_model_and_level():
    settings = Settings(api_keys=["test-token"], backend="mock", model_id="gpt-a", available_models=["gpt-a"], available_levels=["auto", "high"])
    client = TestClient(create_app(settings))
    headers = {"Authorization": "Bearer test-token"}

    bad_model = client.post("/v1/chat/completions", json={"model":"missing", "messages":[{"role":"user","content":"hi"}]}, headers=headers)
    assert bad_model.status_code == 400
    assert bad_model.json()["detail"]["error"] == "unsupported_model"

    bad_level = client.post("/v1/chat/completions", json={"model":"gpt-a", "level":"ultra", "messages":[{"role":"user","content":"hi"}]}, headers=headers)
    assert bad_level.status_code == 400
    assert bad_level.json()["detail"]["error"] == "unsupported_level"


def test_chat_completions_passes_requested_model_and_reasoning_level():
    settings = Settings(api_keys=["test-token"], backend="mock", model_id="gpt-a", available_models=["gpt-a", "gpt-b"], available_levels=["auto", "high"])
    client = TestClient(create_app(settings))
    r = client.post(
        "/v1/chat/completions",
        json={"model":"gpt-b", "reasoning_effort":"high", "messages":[{"role":"user","content":"hi"}]},
        headers={"Authorization":"Bearer test-token"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["model"] == "gpt-b"
    assert data["level"] == "high"


def test_responses_passes_requested_model_and_level():
    settings = Settings(api_keys=["test-token"], backend="mock", model_id="gpt-a", available_models=["gpt-a", "gpt-b"], available_levels=["auto", "high"])
    client = TestClient(create_app(settings))
    r = client.post(
        "/v1/responses",
        json={"model":"gpt-b", "level":"high", "input":"hi"},
        headers={"Authorization":"Bearer test-token"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["model"] == "gpt-b"
    assert data["level"] == "high"


# --- provider session API contract ---

def test_session_api_lifecycle_and_pinned_configuration():
    settings = Settings(
        api_keys=["test-token"],
        backend="mock",
        model_id="gpt-a",
        available_models=["gpt-a", "gpt-b"],
        available_levels=["high", "xhigh"],
    )
    client = TestClient(create_app(settings))
    headers = {"Authorization": "Bearer test-token"}

    created = client.post(
        "/v1/sessions",
        json={
            "session_id": "re-high",
            "model": "gpt-a",
            "reasoning_effort": "high",
        },
        headers=headers,
    )
    assert created.status_code == 201
    assert created.json() == {
        "session_id": "re-high",
        "model": "gpt-a",
        "level": "high",
        "conversation_policy": "regular",
        "state": "ready",
    }

    fetched = client.get("/v1/sessions/re-high", headers=headers)
    assert fetched.status_code == 200
    assert fetched.json()["session_id"] == "re-high"
    assert fetched.json()["model"] == "gpt-a"
    assert fetched.json()["level"] == "high"

    listed = client.get("/v1/sessions", headers=headers)
    assert listed.status_code == 200
    assert listed.json()["object"] == "list"
    assert [s["session_id"] for s in listed.json()["data"]] == ["re-high"]

    completed = client.post(
        "/v1/sessions/re-high/completions",
        json={"messages": [{"role": "user", "content": "Say pong"}]},
        headers=headers,
    )
    assert completed.status_code == 200
    body = completed.json()
    assert body["session_id"] == "re-high"
    assert body["model"] == "gpt-a"
    assert body["level"] == "high"
    assert "Say pong" in body["choices"][0]["message"]["content"]

    deleted = client.delete("/v1/sessions/re-high", headers=headers)
    assert deleted.status_code == 204

    assert client.get("/v1/sessions/re-high", headers=headers).status_code == 404


def test_session_api_rejects_duplicate_and_invalid_configuration():
    settings = Settings(
        api_keys=["test-token"],
        backend="mock",
        model_id="gpt-a",
        available_models=["gpt-a"],
        available_levels=["high", "xhigh"],
    )
    client = TestClient(create_app(settings))
    headers = {"Authorization": "Bearer test-token"}

    payload = {
        "session_id": "review-xhigh",
        "model": "gpt-a",
        "reasoning_effort": "xhigh",
    }

    assert client.post("/v1/sessions", json=payload, headers=headers).status_code == 201
    assert client.post("/v1/sessions", json=payload, headers=headers).status_code == 409

    bad_level = client.post(
        "/v1/sessions",
        json={
            "session_id": "bad",
            "model": "gpt-a",
            "reasoning_effort": "ultra",
        },
        headers=headers,
    )
    assert bad_level.status_code == 400
    assert bad_level.json()["detail"]["error"] == "unsupported_level"


def test_session_completion_cannot_override_pinned_model_or_reasoning():
    settings = Settings(
        api_keys=["test-token"],
        backend="mock",
        model_id="gpt-a",
        available_models=["gpt-a", "gpt-b"],
        available_levels=["high", "xhigh"],
    )
    client = TestClient(create_app(settings))
    headers = {"Authorization": "Bearer test-token"}

    assert client.post(
        "/v1/sessions",
        json={
            "session_id": "crypto-high",
            "model": "gpt-a",
            "reasoning_effort": "high",
        },
        headers=headers,
    ).status_code == 201

    attempt = client.post(
        "/v1/sessions/crypto-high/completions",
        json={
            "model": "gpt-b",
            "reasoning_effort": "xhigh",
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers=headers,
    )

    assert attempt.status_code == 422


def test_chat_completions_affinity_lazily_creates_and_pins_session():
    settings = Settings(
        api_keys=["test-token"],
        backend="mock",
        model_id="gpt-a",
        available_models=["gpt-a", "gpt-b"],
        available_levels=["high", "xhigh"],
    )

    client = TestClient(create_app(settings))

    headers = {
        "Authorization": "Bearer test-token",
        "X-ChatGPT-Session": "passthrough-xhigh",
    }

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-a",
            "reasoning_effort": "xhigh",
            "messages": [
                {
                    "role": "user",
                    "content": "first",
                }
            ],
        },
        headers=headers,
    )

    assert first.status_code == 200

    session = client.get(
        "/v1/sessions/passthrough-xhigh",
        headers={
            "Authorization": "Bearer test-token",
        },
    )

    assert session.status_code == 200
    assert session.json()["model"] == "gpt-a"
    assert session.json()["level"] == "xhigh"

    # Affinity identity pins both model and reasoning tier.
    conflict = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-a",
            "reasoning_effort": "high",
            "messages": [
                {
                    "role": "user",
                    "content": "different tier",
                }
            ],
        },
        headers=headers,
    )

    assert conflict.status_code == 409
    assert (
        conflict.json()["detail"]["error"]
        == "session_configuration_conflict"
    )

    # Explicit reset retains the same pinned configuration.
    reset = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-a",
            "reasoning_effort": "xhigh",
            "new_session": True,
            "messages": [
                {
                    "role": "user",
                    "content": "fresh conversation",
                }
            ],
        },
        headers=headers,
    )

    assert reset.status_code == 200

    session = client.get(
        "/v1/sessions/passthrough-xhigh",
        headers={
            "Authorization": "Bearer test-token",
        },
    )

    assert session.status_code == 200
    assert session.json()["model"] == "gpt-a"
    assert session.json()["level"] == "xhigh"


def test_session_api_accepts_conversation_policy():
    settings = Settings(
        api_keys=["test-token"],
        backend="mock",
        model_id="gpt-a",
        available_models=["gpt-a"],
        available_levels=["high", "xhigh"],
    )

    client = TestClient(create_app(settings))
    headers = {
        "Authorization": "Bearer test-token",
    }

    created = client.post(
        "/v1/sessions",
        json={
            "session_id": "isolated-worker",
            "model": "gpt-a",
            "reasoning_effort": "high",
            "conversation_policy":
                "temporary_unpersonalized",
        },
        headers=headers,
    )

    assert created.status_code == 201
    assert (
        created.json()["conversation_policy"]
        == "temporary_unpersonalized"
    )

    fetched = client.get(
        "/v1/sessions/isolated-worker",
        headers=headers,
    )

    assert (
        fetched.json()["conversation_policy"]
        == "temporary_unpersonalized"
    )


def test_affinity_header_can_select_conversation_policy():
    settings = Settings(
        api_keys=["test-token"],
        backend="mock",
        model_id="gpt-a",
        available_models=["gpt-a"],
        available_levels=["high", "xhigh"],
    )

    client = TestClient(create_app(settings))

    headers = {
        "Authorization": "Bearer test-token",
        "X-ChatGPT-Session": "affinity-isolated",
        "X-ChatGPT-Conversation-Policy":
            "temporary_unpersonalized",
    }

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-a",
            "reasoning_effort": "xhigh",
            "messages": [
                {
                    "role": "user",
                    "content": "hello",
                }
            ],
        },
        headers=headers,
    )

    assert response.status_code == 200

    session = client.get(
        "/v1/sessions/affinity-isolated",
        headers={
            "Authorization": "Bearer test-token",
        },
    )

    assert (
        session.json()["conversation_policy"]
        == "temporary_unpersonalized"
    )


def test_affinity_policy_is_pinned():
    settings = Settings(
        api_keys=["test-token"],
        backend="mock",
        model_id="gpt-a",
        available_models=["gpt-a"],
        available_levels=["high", "xhigh"],
    )

    client = TestClient(create_app(settings))

    first_headers = {
        "Authorization": "Bearer test-token",
        "X-ChatGPT-Session": "affinity-policy",
        "X-ChatGPT-Conversation-Policy":
            "temporary_unpersonalized",
    }

    first = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-a",
            "reasoning_effort": "xhigh",
            "messages": [
                {
                    "role": "user",
                    "content": "first",
                }
            ],
        },
        headers=first_headers,
    )

    assert first.status_code == 200

    conflicting_headers = {
        "Authorization": "Bearer test-token",
        "X-ChatGPT-Session": "affinity-policy",
        "X-ChatGPT-Conversation-Policy":
            "regular",
    }

    conflict = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-a",
            "reasoning_effort": "xhigh",
            "messages": [
                {
                    "role": "user",
                    "content": "second",
                }
            ],
        },
        headers=conflicting_headers,
    )

    assert conflict.status_code == 409
    assert (
        conflict.json()["detail"]["error"]
        == "session_configuration_conflict"
    )


def test_affinity_chat_completion_returns_openai_tool_calls():
    from chatgpt_web_provider.backends import MockBackend
    from chatgpt_web_provider.models import (
        ToolCall,
        ToolFunctionCall,
    )

    class ToolAwareBackend(MockBackend):
        def __init__(self, settings):
            super().__init__(settings)
            self.received_tools = None
            self.received_tool_choice = None
            self.received_parallel = None

        async def complete_affinity_session(
            self,
            session_id,
            messages,
            *,
            tools=None,
            tool_choice=None,
            parallel_tool_calls=True,
        ):
            self.received_tools = tools
            self.received_tool_choice = tool_choice
            self.received_parallel = parallel_tool_calls

            return CompletionResult(
                model="gpt-a",
                level="xhigh",
                tool_calls=[
                    ToolCall(
                        id="call-test",
                        function=ToolFunctionCall(
                            name="worker_list",
                            arguments="{}",
                        ),
                    )
                ],
            )

    settings = Settings(
        api_keys=["test-token"],
        backend="mock",
        model_id="gpt-a",
        available_models=["gpt-a"],
        available_levels=["xhigh"],
    )

    backend = ToolAwareBackend(settings)
    client = TestClient(
        create_app(settings, backend=backend)
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

    response = client.post(
        "/v1/chat/completions",
        headers={
            "Authorization": "Bearer test-token",
            "X-ChatGPT-Session": "tool-affinity",
        },
        json={
            "model": "gpt-a",
            "reasoning_effort": "xhigh",
            "messages": [
                {
                    "role": "user",
                    "content": "List workers.",
                }
            ],
            "tools": tools,
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        },
    )

    assert response.status_code == 200

    choice = response.json()["choices"][0]

    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None

    assert choice["message"]["tool_calls"] == [
        {
            "id": "call-test",
            "type": "function",
            "function": {
                "name": "worker_list",
                "arguments": "{}",
            },
        }
    ]

    assert backend.received_tools == tools
    assert backend.received_tool_choice == "auto"
    assert backend.received_parallel is True


def test_streaming_affinity_tool_call_uses_tool_calls_finish_reason():
    from chatgpt_web_provider.backends import MockBackend
    from chatgpt_web_provider.models import (
        ToolCall,
        ToolFunctionCall,
    )

    class ToolAwareBackend(MockBackend):
        async def complete_affinity_session(
            self,
            session_id,
            messages,
            *,
            tools=None,
            tool_choice=None,
            parallel_tool_calls=True,
        ):
            return CompletionResult(
                model="gpt-a",
                level="xhigh",
                tool_calls=[
                    ToolCall(
                        id="call-stream",
                        function=ToolFunctionCall(
                            name="worker_list",
                            arguments="{}",
                        ),
                    )
                ],
            )

    settings = Settings(
        api_keys=["test-token"],
        backend="mock",
        model_id="gpt-a",
        available_models=["gpt-a"],
        available_levels=["xhigh"],
    )

    client = TestClient(
        create_app(
            settings,
            backend=ToolAwareBackend(settings),
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers={
            "Authorization": "Bearer test-token",
            "X-ChatGPT-Session": "tool-stream",
        },
        json={
            "model": "gpt-a",
            "reasoning_effort": "xhigh",
            "stream": True,
            "messages": [
                {
                    "role": "user",
                    "content": "List workers.",
                }
            ],
            "tools": [
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
            ],
        },
    )

    assert response.status_code == 200
    assert '"tool_calls"' in response.text
    assert '"call-stream"' in response.text
    assert '"finish_reason": "tool_calls"' in response.text


def test_chatgpt_ui_rate_limit_maps_to_http_429():
    from chatgpt_web_provider.backends import (
        ChatGPTUIRateLimitError,
        MockBackend,
    )

    class RateLimitedBackend(MockBackend):
        async def complete(
            self,
            messages,
            model=None,
            new_session=False,
            level=None,
        ):
            raise ChatGPTUIRateLimitError(
                phase="composer",
                retry_after_seconds=60,
            )

    settings = Settings(
        api_keys=["test-token"],
        backend="mock",
    )

    client = TestClient(
        create_app(
            settings,
            backend=RateLimitedBackend(settings),
        )
    )

    response = client.post(
        "/v1/chat/completions",
        headers={
            "Authorization": "Bearer test-token",
        },
        json={
            "model": settings.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": "hello",
                }
            ],
        },
    )

    assert response.status_code == 429
    assert response.headers["retry-after"] == "60"

    assert (
        response.json()["error"]["code"]
        == "chatgpt_ui_rate_limited"
    )
