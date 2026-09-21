"""Browser-facing prompt and transcript resources.

These strings and renderers define how provider context is presented to
the ChatGPT Web reasoning backend. They are adapter/protocol resources,
not operator configuration.
"""

from __future__ import annotations

import json

from .models import ChatMessage


TOOL_CATALOG_SYSTEM_PREFIX = "SYSTEM:\n"

HOST_BRIDGE_CONTEXT = (
    "SYSTEM:\n"
    "HOST BRIDGE CONTEXT:\n"
    "You are the reasoning backend for a Hermes Agent host. "
    "The DEVELOPER message below contains policy and runtime "
    "context for the Hermes host agent you are driving. "
    "References there to 'you', Hermes Agent, runtime tools, "
    "or tool availability describe the host agent, not the "
    "native ChatGPT web UI.\n"
    "The external tool catalog supplied with this request is "
    "authoritative for host-tool availability. A listed "
    "external tool is available even though it is not a native "
    "ChatGPT UI tool. When the user's request requires a listed "
    "tool, use the external tool-call protocol supplied by the "
    "provider instead of reporting that the tool is unavailable."
)

GATEWAY_TOOL_REMINDER = (
    "EXTERNAL TOOL REMINDER:\n"
    "Hermes host tools remain available. "
    "When a requested capability is not directly visible, "
    "use tool_search, tool_describe, and tool_call to "
    "discover and invoke it. Do not report a tool or "
    "server as unavailable before attempting that path."
)

DIRECT_TOOL_REMINDER = (
    "EXTERNAL TOOL REMINDER:\n"
    "The external tools listed above are available through "
    "the Hermes host. When the user explicitly asks for an "
    "operation provided by one of them, call the matching "
    "tool instead of answering that it is unavailable. "
    "Return normal prose only after the required tool work "
    "has completed or an actual tool call reports failure."
)

ASSISTANT_EXTERNAL_TOOL_CALLS_HEADING = (
    "ASSISTANT EXTERNAL TOOL CALLS:\n"
)

EXTERNAL_TOOL_RESULT_HEADING = "EXTERNAL TOOL RESULT"

GATEWAY_TOOL_NAMES = frozenset(
    {
        "tool_search",
        "tool_describe",
        "tool_call",
    }
)


def render_host_bridge_context(
    messages: list[ChatMessage],
) -> str:
    """Render host/backend disambiguation when developer context exists."""
    if not any(
        message.role == "developer"
        for message in messages
    ):
        return ""

    return HOST_BRIDGE_CONTEXT


def render_tool_reminder(
    tools: list[dict] | None,
) -> str:
    """Keep browser-backed external tools salient to the reasoning backend."""
    if not tools:
        return ""

    available_names = {
        function.get("name")
        for tool in tools
        if isinstance(tool, dict)
        and isinstance(
            function := tool.get("function"),
            dict,
        )
        and isinstance(
            function.get("name"),
            str,
        )
    }

    if GATEWAY_TOOL_NAMES.issubset(
        available_names
    ):
        return GATEWAY_TOOL_REMINDER

    return DIRECT_TOOL_REMINDER


def render_transcript(
    messages: list[ChatMessage],
) -> str:
    """Render OpenAI-style messages into the browser prompt transcript."""
    rendered = []

    for message in messages:
        if (
            message.role == "assistant"
            and message.tool_calls
        ):
            calls = [
                call.model_dump(
                    mode="json",
                    exclude_none=True,
                )
                for call in message.tool_calls
            ]

            rendered.append(
                ASSISTANT_EXTERNAL_TOOL_CALLS_HEADING
                + json.dumps(
                    calls,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
            )

            continue

        if message.role == "tool":
            metadata = []

            if message.tool_call_id:
                metadata.append(
                    f"id={message.tool_call_id}"
                )

            if message.name:
                metadata.append(
                    f"name={message.name}"
                )

            suffix = (
                " " + " ".join(metadata)
                if metadata
                else ""
            )

            rendered.append(
                EXTERNAL_TOOL_RESULT_HEADING
                + suffix
                + ":\n"
                + message.text()
            )

            continue

        rendered.append(
            f"{message.role.upper()}: "
            f"{message.text()}"
        )

    return "\n\n".join(rendered)
