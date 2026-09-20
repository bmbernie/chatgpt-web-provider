from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    role: str
    content: str | list | dict | None = ""

    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return str(self.content or "")


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    new_session: bool = False
    level: str | None = None
    reasoning_effort: str | None = None


class ResponsesRequest(BaseModel):
    model: str | None = None
    input: str | list | dict
    temperature: float | None = None
    max_output_tokens: int | None = None
    stream: bool = False
    new_session: bool = False
    level: str | None = None
    reasoning_effort: str | None = None


class CompletionResult(BaseModel):
    text: str
    model: str
    level: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


class SessionCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$",
    )
    model: str
    reasoning_effort: str


class SessionCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage]


class ErrorBody(BaseModel):
    error: dict[str, str] = Field(default_factory=dict)
