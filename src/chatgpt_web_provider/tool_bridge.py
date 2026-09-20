from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable

from .models import ToolCall, ToolFunctionCall


START = "<<<HERMES_EXTERNAL_TOOL_CALLS>>>"
END = "<<<END_HERMES_EXTERNAL_TOOL_CALLS>>>"


class ToolBridgeError(RuntimeError):
    pass


def _canonical(value) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _function_tools(
    tools: list[dict] | None,
) -> dict[str, dict]:
    result: dict[str, dict] = {}

    for tool in tools or []:
        if tool.get("type") != "function":
            continue

        function = tool.get("function")
        if not isinstance(function, dict):
            continue

        name = function.get("name")

        if isinstance(name, str) and name:
            result[name] = function

    return result


def tool_catalog_fingerprint(
    tools: list[dict] | None,
    *,
    tool_choice=None,
    parallel_tool_calls: bool = True,
) -> str:
    payload = {
        "tools": tools or [],
        "tool_choice": tool_choice,
        "parallel_tool_calls": parallel_tool_calls,
    }

    return hashlib.sha256(
        _canonical(payload).encode("utf-8")
    ).hexdigest()


def render_tool_catalog(
    tools: list[dict] | None,
    *,
    tool_choice=None,
    parallel_tool_calls: bool = True,
) -> str:
    functions = _function_tools(tools)

    catalog = [
        {
            "name": name,
            "description": function.get(
                "description",
                "",
            ),
            "parameters": function.get(
                "parameters",
                {"type": "object"},
            ),
        }
        for name, function in functions.items()
    ]

    protocol = {
        "parallel_tool_calls": parallel_tool_calls,
        "tool_choice": tool_choice,
        "tools": catalog,
    }

    return (
        "External tools are available through the Hermes host.\n"
        "Do not claim these tools are unavailable merely because "
        "they are not native ChatGPT UI tools.\n"
        "When an external tool is required, your ENTIRE response "
        "must be exactly this envelope with valid JSON between "
        "the markers:\n"
        f"{START}\n"
        '{"calls":[{"name":"tool_name","arguments":{}}]}\n'
        f"{END}\n"
        "Do not put prose, Markdown, or code fences outside the "
        "envelope when making a tool call.\n"
        "Only call tools appearing in this catalog.\n"
        "Otherwise answer normally without either marker.\n\n"
        "External tool catalog:\n"
        + _canonical(protocol)
    )


def parse_tool_calls(
    text: str,
    tools: list[dict] | None,
    *,
    parallel_tool_calls: bool = True,
    id_factory: Callable[[], str] | None = None,
) -> list[ToolCall] | None:
    stripped = text.strip()

    prefix = START + "\n"
    suffix = "\n" + END

    if not (
        stripped.startswith(prefix)
        and stripped.endswith(suffix)
    ):
        return None

    payload_text = stripped[
        len(prefix):len(stripped) - len(suffix)
    ]

    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise ToolBridgeError(
            "invalid external tool-call JSON"
        ) from exc

    if not isinstance(payload, dict):
        raise ToolBridgeError(
            "external tool-call envelope must be an object"
        )

    calls = payload.get("calls")

    if not isinstance(calls, list) or not calls:
        raise ToolBridgeError(
            "external tool-call envelope requires calls"
        )

    if not parallel_tool_calls and len(calls) > 1:
        raise ToolBridgeError(
            "parallel external tool calls are disabled"
        )

    available = _function_tools(tools)

    make_id = id_factory or (
        lambda: f"call_{uuid.uuid4().hex}"
    )

    result: list[ToolCall] = []

    for item in calls:
        if not isinstance(item, dict):
            raise ToolBridgeError(
                "external tool call must be an object"
            )

        name = item.get("name")
        arguments = item.get("arguments", {})

        if name not in available:
            raise ToolBridgeError(
                f"unknown external tool: {name!r}"
            )

        if not isinstance(arguments, dict):
            raise ToolBridgeError(
                "external tool arguments must be an object"
            )

        result.append(
            ToolCall(
                id=make_id(),
                type="function",
                function=ToolFunctionCall(
                    name=name,
                    arguments=_canonical(arguments),
                ),
            )
        )

    return result
