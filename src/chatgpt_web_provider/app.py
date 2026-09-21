from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Optional

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse

from .backends import (
    Backend,
    ChatGPTGenerationTimeoutError,
    ChatGPTUIRateLimitError,
    build_backend,
)
from .config import SESSION_POLICIES, Settings
from .models import (
    ChatCompletionRequest,
    ChatMessage,
    ResponsesRequest,
    SessionCompletionRequest,
    SessionCreateRequest,
)
from .security import redact_secret


logger = logging.getLogger("uvicorn.error")


def _require_auth(settings: Settings):
    async def dep(authorization: Optional[str] = Header(default=None), x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> str:
        candidates: list[str] = []
        if x_api_key and x_api_key.strip() and not x_api_key.strip().startswith("{{"):
            candidates.append(x_api_key.strip())
        if authorization and authorization.lower().startswith("bearer "):
            candidates.append(authorization.split(" ", 1)[1].strip())

        if not candidates:
            raise HTTPException(status_code=401, detail="missing bearer token")
        for token in candidates:
            if token in settings.api_keys:
                return token
        raise HTTPException(status_code=403, detail="invalid bearer token")
    return dep


def _truthy_header(value: str | None) -> bool:
    return bool(value and value.strip().lower() in {"1", "true", "yes", "on"})


def _request_level(req) -> str | None:
    return getattr(req, "level", None) or getattr(req, "reasoning_effort", None)


def _validate_conversation_policy(
    policy: str,
) -> None:
    if policy not in SESSION_POLICIES:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "unsupported_conversation_policy",
                "conversation_policy": policy,
                "available_policies": list(SESSION_POLICIES),
            },
        )


def _validate_requested_options(settings: Settings, model: str, level: str | None) -> None:
    if model not in settings.available_models:
        raise HTTPException(
            status_code=400,
            detail={"error": "unsupported_model", "model": model, "available_models": settings.available_models},
        )
    if level and level not in settings.available_levels:
        raise HTTPException(
            status_code=400,
            detail={"error": "unsupported_level", "level": level, "available_levels": settings.available_levels},
        )


async def _run_with_queue(
    settings: Settings,
    queue_sem: asyncio.Semaphore,
    operation,
    *,
    context: str = "provider",
):
    queued_at = time.perf_counter()

    try:
        async with asyncio.timeout(settings.queue_timeout_seconds):
            await queue_sem.acquire()
            acquired_at = time.perf_counter()

            try:
                result = await operation()
            except Exception as exc:
                finished_at = time.perf_counter()
                logger.error(
                    "provider_queue_failed %s queue_wait_ms=%.1f "
                    "operation_ms=%.1f total_ms=%.1f error_type=%s",
                    context,
                    (acquired_at - queued_at) * 1000,
                    (finished_at - acquired_at) * 1000,
                    (finished_at - queued_at) * 1000,
                    type(exc).__name__,
                )
                raise
            finally:
                queue_sem.release()

            finished_at = time.perf_counter()
            logger.info(
                "provider_queue_complete %s queue_wait_ms=%.1f "
                "operation_ms=%.1f total_ms=%.1f",
                context,
                (acquired_at - queued_at) * 1000,
                (finished_at - acquired_at) * 1000,
                (finished_at - queued_at) * 1000,
            )
            return result

    except TimeoutError as exc:
        finished_at = time.perf_counter()
        logger.error(
            "provider_queue_timeout %s total_ms=%.1f",
            context,
            (finished_at - queued_at) * 1000,
        )
        raise HTTPException(
            status_code=504,
            detail="request timed out while waiting for provider queue",
        ) from exc



async def _chat_completion_stream(
    response_id: str,
    created: int,
    model: str,
    text: str,
    usage: dict,
    tool_calls=None,
):
    if tool_calls:
        delta_tool_calls = []

        for index, call in enumerate(tool_calls):
            delta_tool_calls.append(
                {
                    "index": index,
                    "id": call.id,
                    "type": call.type,
                    "function": {
                        "name": call.function.name,
                        "arguments":
                            call.function.arguments,
                    },
                }
            )

        first_delta = {
            "role": "assistant",
            "tool_calls": delta_tool_calls,
        }

        finish_reason = "tool_calls"

    else:
        first_delta = {
            "role": "assistant",
            "content": text,
        }

        finish_reason = "stop"

    first = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": first_delta,
                "finish_reason": None,
            }
        ],
    }

    final = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage,
    }

    yield (
        "data: "
        + json.dumps(
            first,
            ensure_ascii=False,
        )
        + "\n\n"
    )

    yield (
        "data: "
        + json.dumps(
            final,
            ensure_ascii=False,
        )
        + "\n\n"
    )

    yield "data: [DONE]\n\n"


def create_app(settings: Settings | None = None, backend: Backend | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.validate_for_runtime()
    backend = backend or build_backend(settings)
    auth = _require_auth(settings)
    queue_sem = asyncio.Semaphore(settings.max_concurrent_requests)

    app = FastAPI(
        title="chatgpt-web-provider",
        version="0.1.0",
        description="OpenAI-compatible API facade for a browser-backed ChatGPT.com worker.",
    )
    app.state.settings = settings
    app.state.backend = backend

    @app.exception_handler(ChatGPTGenerationTimeoutError)
    async def chatgpt_generation_timeout(
        _: Request,
        exc: ChatGPTGenerationTimeoutError,
    ):
        return JSONResponse(
            status_code=504,
            content={
                "error": {
                    "message": (
                        "ChatGPT generation did not finish before "
                        "the configured timeout"
                    ),
                    "type": "generation_timeout_error",
                    "code": "chatgpt_generation_timeout",
                    "timeout_seconds": exc.timeout_seconds,
                }
            },
        )

    @app.exception_handler(ChatGPTUIRateLimitError)
    async def chatgpt_ui_rate_limit(
        _: Request,
        exc: ChatGPTUIRateLimitError,
    ):
        headers = {}

        if (
            settings.ui_rate_limit_retry_after_seconds
            is not None
        ):
            headers["Retry-After"] = str(
                settings.ui_rate_limit_retry_after_seconds
            )

        return JSONResponse(
            status_code=429,
            headers=headers,
            content={
                "error": {
                    "message":
                        "ChatGPT UI is temporarily rate limited",
                    "type": "rate_limit_error",
                    "code": "chatgpt_ui_rate_limited",
                }
            },
        )

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception):
        return JSONResponse(status_code=500, content={"error": {"message": redact_secret(str(exc)), "type": exc.__class__.__name__}})

    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse("/docs", status_code=308)

    @app.get("/health")
    async def health():
        data = await backend.health()
        return {"ok": bool(data.get("ok")), "backend": settings.backend, "model": settings.model_id, **data}

    @app.get("/v1/models")
    async def models(_token: str = Depends(auth)):
        return {
            "object": "list",
            "data": [
                {
                    "id": model,
                    "object": "model",
                    "created": 0,
                    "owned_by": "chatgpt-web-provider",
                    "display_name": settings.model_label(model),
                    "default": model == settings.model_id,
                }
                for model in settings.available_models
            ],
        }

    @app.get("/v1/provider/capabilities")
    async def provider_capabilities(_token: str = Depends(auth)):
        return {
            "backend": settings.backend,
            "default_model": settings.model_id,
            "models": [
                {"id": model, "display_name": settings.model_label(model), "default": model == settings.model_id}
                for model in settings.available_models
            ],
            "levels": [
                {"id": level, "display_name": settings.level_label(level), "default": level == settings.available_levels[0]}
                for level in settings.available_levels
            ],
            "new_session": {"body_field": "new_session", "header": "X-New-Session"},
        }

    @app.get("/v1/provider/status")
    async def provider_status(_token: str = Depends(auth)):
        health_data = await backend.health()
        waiters = max(0, settings.max_concurrent_requests - getattr(queue_sem, "_value", settings.max_concurrent_requests))
        return {
            "backend": settings.backend,
            "model": settings.model_id,
            "models": settings.available_models,
            "levels": settings.available_levels,
            "health": health_data,
            "queue": {
                "max_concurrent_requests": settings.max_concurrent_requests,
                "queue_timeout_seconds": settings.queue_timeout_seconds,
                "in_flight_estimate": waiters,
            },
        }

    async def _session_backend_call(operation):
        try:
            return await operation
        except NotImplementedError as exc:
            raise HTTPException(
                status_code=501,
                detail=str(exc),
            ) from exc
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail="session not found",
            ) from exc

    @app.post("/v1/sessions", status_code=201)
    async def create_session(
        req: SessionCreateRequest,
        _token: str = Depends(auth),
    ):
        selected_policy = (
            req.conversation_policy
            or settings.session_policy
        )

        _validate_requested_options(
            settings,
            req.model,
            req.reasoning_effort,
        )
        _validate_conversation_policy(
            selected_policy
        )

        try:
            return await backend.create_session(
                req.session_id,
                req.model,
                req.reasoning_effort,
                selected_policy,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=409,
                detail=str(exc),
            ) from exc
        except NotImplementedError as exc:
            raise HTTPException(
                status_code=501,
                detail=str(exc),
            ) from exc

    @app.get("/v1/sessions")
    async def list_sessions(
        _token: str = Depends(auth),
    ):
        sessions = await _session_backend_call(
            backend.list_sessions()
        )
        return {
            "object": "list",
            "data": sessions,
        }

    @app.get("/v1/sessions/{session_id}")
    async def get_session(
        session_id: str,
        _token: str = Depends(auth),
    ):
        return await _session_backend_call(
            backend.get_session(session_id)
        )

    @app.delete(
        "/v1/sessions/{session_id}",
        status_code=204,
    )
    async def delete_session(
        session_id: str,
        _token: str = Depends(auth),
    ):
        await _session_backend_call(
            backend.delete_session(session_id)
        )
        return Response(status_code=204)

    @app.post("/v1/sessions/{session_id}/completions")
    async def session_completion(
        session_id: str,
        req: SessionCompletionRequest,
        _token: str = Depends(auth),
    ):
        result = await _session_backend_call(
            backend.complete_session(
                session_id,
                req.messages,
            )
        )

        response_id = f"chatcmpl-{uuid.uuid4().hex}"

        usage = {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": (
                result.prompt_tokens
                + result.completion_tokens
            ),
        }

        return {
            "id": response_id,
            "object": "chat.completion",
            "session_id": session_id,
            "model": result.model,
            "level": result.level,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result.text,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(
        req: ChatCompletionRequest,
        x_new_session: Optional[str] = Header(
            default=None,
            alias="X-New-Session",
        ),
        x_chatgpt_session: Optional[str] = Header(
            default=None,
            alias="X-ChatGPT-Session",
        ),
        x_chatgpt_conversation_policy: Optional[str] = Header(
            default=None,
            alias="X-ChatGPT-Conversation-Policy",
        ),
        _token: str = Depends(auth),
    ):
        started = int(time.time())
        selected_model = req.model or settings.model_id
        selected_level = _request_level(req)

        _validate_requested_options(
            settings,
            selected_model,
            selected_level,
        )

        new_session = (
            req.new_session
            or _truthy_header(x_new_session)
        )

        affinity_session_id = (
            x_chatgpt_session.strip()
            if x_chatgpt_session
            else ""
        )

        requested_policy = (
            x_chatgpt_conversation_policy.strip().lower()
            if x_chatgpt_conversation_policy
            else None
        )

        if requested_policy is not None:
            _validate_conversation_policy(
                requested_policy
            )

        tool_names = [
            str(tool.get("function", {}).get("name", "?"))
            for tool in (req.tools or [])
            if isinstance(tool, dict)
        ]

        logger.info(
            "chat_completion_capabilities "
            "session_id=%s tools=%d "
            "tool_choice=%s parallel_tool_calls=%s "
            "tool_names=%s",
            affinity_session_id or "-",
            len(req.tools or []),
            (
                str(req.tool_choice)
                if req.tool_choice is not None
                else "-"
            ),
            str(req.parallel_tool_calls).lower(),
            ",".join(tool_names),
        )

        if affinity_session_id:
            valid_session_id = (
                1 <= len(affinity_session_id) <= 64
                and affinity_session_id[0].isascii()
                and affinity_session_id[0].isalnum()
                and all(
                    (
                        char.isascii()
                        and (
                            char.isalnum()
                            or char in "._-"
                        )
                    )
                    for char in affinity_session_id
                )
            )

            if not valid_session_id:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "invalid_session_id",
                        "session_id": affinity_session_id,
                    },
                )

            try:
                existing = await backend.get_session(
                    affinity_session_id
                )
            except KeyError:
                existing = None
            except NotImplementedError as exc:
                raise HTTPException(
                    status_code=501,
                    detail={
                        "error": "session_affinity_unsupported",
                    },
                ) from exc

            effective_level = selected_level
            effective_policy = requested_policy

            if existing is not None:
                if effective_level is None:
                    effective_level = existing["level"]

                if effective_policy is None:
                    effective_policy = existing[
                        "conversation_policy"
                    ]

                if (
                    existing["model"] != selected_model
                    or existing["level"] != effective_level
                    or existing["conversation_policy"]
                    != effective_policy
                ):
                    raise HTTPException(
                        status_code=409,
                        detail={
                            "error":
                                "session_configuration_conflict",
                            "session_id":
                                affinity_session_id,
                            "existing_model":
                                existing["model"],
                            "existing_level":
                                existing["level"],
                            "existing_conversation_policy":
                                existing["conversation_policy"],
                            "requested_model":
                                selected_model,
                            "requested_level":
                                effective_level,
                            "requested_conversation_policy":
                                effective_policy,
                        },
                    )

                if new_session:
                    await backend.delete_session(
                        affinity_session_id
                    )
                    existing = None

            if existing is None:
                if effective_level is None:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error":
                                "session_affinity_requires_level",
                            "session_id":
                                affinity_session_id,
                        },
                    )

                if effective_policy is None:
                    effective_policy = settings.session_policy

                _validate_conversation_policy(
                    effective_policy
                )

                try:
                    await backend.create_session(
                        affinity_session_id,
                        selected_model,
                        effective_level,
                        effective_policy,
                    )

                except ValueError:
                    # Another request may have won a concurrent
                    # first-use race. Adopt it only when all pinned
                    # configuration matches.
                    existing = await backend.get_session(
                        affinity_session_id
                    )

                    if (
                        existing["model"] != selected_model
                        or existing["level"] != effective_level
                        or existing["conversation_policy"]
                        != effective_policy
                    ):
                        raise HTTPException(
                            status_code=409,
                            detail={
                                "error":
                                    "session_configuration_conflict",
                                "session_id":
                                    affinity_session_id,
                                "existing_model":
                                    existing["model"],
                                "existing_level":
                                    existing["level"],
                                "existing_conversation_policy":
                                    existing[
                                        "conversation_policy"
                                    ],
                                "requested_model":
                                    selected_model,
                                "requested_level":
                                    effective_level,
                                "requested_conversation_policy":
                                    effective_policy,
                            },
                        )

            affinity_complete = getattr(
                backend,
                "complete_affinity_session",
                backend.complete_session,
            )

            result = await _run_with_queue(
                settings,
                queue_sem,
                lambda: affinity_complete(
                    affinity_session_id,
                    req.messages,
                    tools=req.tools,
                    tool_choice=req.tool_choice,
                    parallel_tool_calls=(
                        req.parallel_tool_calls
                    ),
                ),
                context=(
                    "endpoint=chat_completions "
                    f"session_id={affinity_session_id} "
                    f"model={selected_model} "
                    f"level={effective_level} "
                    f"messages={len(req.messages)} "
                    "affinity=true "
                    f"new_session={str(new_session).lower()}"
                ),
            )

        else:
            try:
                if req.tools:
                    result = await _run_with_queue(
                        settings,
                        queue_sem,
                        lambda: backend.complete_with_tools(
                            req.messages,
                            model=selected_model,
                            new_session=new_session,
                            level=selected_level,
                            tools=req.tools,
                            tool_choice=req.tool_choice,
                            parallel_tool_calls=(
                                req.parallel_tool_calls
                            ),
                        ),
                        context=(
                            "endpoint=chat_completions "
                            f"model={selected_model} "
                            f"level={selected_level or '-'} "
                            f"messages={len(req.messages)} "
                            "affinity=false tools=true "
                            f"new_session={str(new_session).lower()}"
                        ),
                    )
                else:
                    result = await _run_with_queue(
                        settings,
                        queue_sem,
                        lambda: backend.complete(
                            req.messages,
                            model=selected_model,
                            new_session=new_session,
                            level=selected_level,
                        ),
                        context=(
                            "endpoint=chat_completions "
                            f"model={selected_model} "
                            f"level={selected_level or '-'} "
                            f"messages={len(req.messages)} "
                            f"new_session={str(new_session).lower()}"
                        ),
                    )

            except NotImplementedError as exc:
                raise HTTPException(
                    status_code=501,
                    detail={
                        "error":
                            "non_affinity_tools_unsupported",
                    },
                ) from exc

        response_id = (
            f"chatcmpl-{uuid.uuid4().hex}"
        )

        usage = {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens":
                result.completion_tokens,
            "total_tokens":
                result.prompt_tokens
                + result.completion_tokens,
        }

        if req.stream:
            return StreamingResponse(
                _chat_completion_stream(
                    response_id,
                    started,
                    result.model,
                    result.text,
                    usage,
                    result.tool_calls,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        if result.tool_calls:
            response_message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    call.model_dump(
                        mode="json",
                        exclude_none=True,
                    )
                    for call in result.tool_calls
                ],
            }

            finish_reason = "tool_calls"

        else:
            response_message = {
                "role": "assistant",
                "content": result.text,
            }

            finish_reason = "stop"

        return {
            "id": response_id,
            "object": "chat.completion",
            "created": started,
            "model": result.model,
            "level": result.level,
            "choices": [
                {
                    "index": 0,
                    "message": response_message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage,
        }

    @app.post("/v1/responses")
    async def responses(req: ResponsesRequest, x_new_session: Optional[str] = Header(default=None, alias="X-New-Session"), _token: str = Depends(auth)):
        if req.stream:
            raise HTTPException(status_code=501, detail="streaming is not implemented yet")
        messages = _responses_input_to_messages(req.input)
        started = int(time.time())
        selected_model = req.model or settings.model_id
        selected_level = _request_level(req)
        _validate_requested_options(settings, selected_model, selected_level)
        new_session = req.new_session or _truthy_header(x_new_session)
        result = await _run_with_queue(
            settings,
            queue_sem,
            lambda: backend.complete(
                messages,
                model=selected_model,
                new_session=new_session,
                level=selected_level,
            ),
            context=(
                f"endpoint=responses model={selected_model} "
                f"level={selected_level or '-'} messages={len(messages)} "
                f"new_session={str(new_session).lower()}"
            ),
        )
        response_id = f"resp_{uuid.uuid4().hex}"
        return {
            "id": response_id,
            "object": "response",
            "created_at": started,
            "status": "completed",
            "model": result.model,
            "level": result.level,
            "output": [
                {
                    "id": f"msg_{uuid.uuid4().hex}",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": result.text}],
                }
            ],
            "output_text": result.text,
            "usage": {
                "input_tokens": result.prompt_tokens,
                "output_tokens": result.completion_tokens,
                "total_tokens": result.prompt_tokens + result.completion_tokens,
            },
        }

    return app


def _responses_input_to_messages(value: str | list | dict) -> list[ChatMessage]:
    if isinstance(value, str):
        return [ChatMessage(role="user", content=value)]
    if isinstance(value, list):
        messages: list[ChatMessage] = []
        for item in value:
            if isinstance(item, dict) and "role" in item:
                content = item.get("content", "")
                if isinstance(content, list):
                    content = "\n".join(str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in content)
                messages.append(ChatMessage(role=str(item.get("role", "user")), content=content))
        return messages or [ChatMessage(role="user", content=str(value))]
    return [ChatMessage(role="user", content=str(value))]


def main() -> None:
    settings = Settings.from_env()
    settings.validate_for_runtime()
    uvicorn.run("chatgpt_web_provider.app:create_app", factory=True, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
