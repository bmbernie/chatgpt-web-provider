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


def _browser_tool_name_maps(
    tools: list[dict] | None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Map browser-safe aliases to original host tool names."""

    functions = _function_tools(tools)

    alias_to_original: dict[str, str] = {}
    original_to_alias: dict[str, str] = {}

    # Non-MCP names keep their original identity and reserve those
    # names against possible MCP alias collisions.
    reserved = {
        name
        for name in functions
        if not name.startswith("mcp__")
    }

    for original in functions:
        if not original.startswith("mcp__"):
            alias = original
        else:
            parts = [
                part
                for part in original.split("__")[1:]
                if part
            ]

            alias = "_".join(parts) or "external_tool"

            if (
                alias in reserved
                or alias in alias_to_original
            ):
                digest = hashlib.sha256(
                    original.encode("utf-8")
                ).hexdigest()[:8]

                alias = f"{alias}_{digest}"

        alias_to_original[alias] = original
        original_to_alias[original] = alias

    return alias_to_original, original_to_alias


def _browser_tool_choice(
    tool_choice,
    original_to_alias: dict[str, str],
):
    """Translate a forced host tool name into its browser alias."""

    if not isinstance(tool_choice, dict):
        return tool_choice

    translated = dict(tool_choice)
    function = translated.get("function")

    if not isinstance(function, dict):
        return translated

    translated_function = dict(function)
    name = translated_function.get("name")

    if isinstance(name, str):
        translated_function["name"] = (
            original_to_alias.get(name, name)
        )

    translated["function"] = translated_function
    return translated


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

    (
        alias_to_original,
        original_to_alias,
    ) = _browser_tool_name_maps(tools)

    catalog = [
        {
            "name": original_to_alias[name],
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
        "tool_choice": _browser_tool_choice(
            tool_choice,
            original_to_alias,
        ),
        "tools": catalog,
    }

    tool_names = [
        original_to_alias[name]
        for name in functions
    ]

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
        + "\n\n"
        + "Available external tool names:\n"
        + ", ".join(tool_names)
        + "\n\n"
        + "IMPORTANT — EXTERNAL TOOL CALL PROTOCOL:\n"
        + "If the user's request requires a listed external tool, "
        + "call it instead of claiming it is unavailable.\n"
        + "Your ENTIRE response for a tool call must be exactly:\n"
        + f"{START}\n"
        + '{"calls":[{"name":"EXACT_TOOL_NAME","arguments":{}}]}\n'
        + f"{END}\n"
        + "Replace EXACT_TOOL_NAME with the exact listed tool name "
        + "and provide arguments matching that tool's schema.\n"
        + "Do not include prose, Markdown, explanation, or code "
        + "fences outside the markers when calling a tool."
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

    (
        alias_to_original,
        original_to_alias,
    ) = _browser_tool_name_maps(tools)

    make_id = id_factory or (
        lambda: f"call_{uuid.uuid4().hex}"
    )

    result: list[ToolCall] = []

    for item in calls:
        if not isinstance(item, dict):
            raise ToolBridgeError(
                "external tool call must be an object"
            )

        browser_name = item.get("name")
        arguments = item.get("arguments", {})

        original_name = (
            alias_to_original.get(browser_name)
            if isinstance(browser_name, str)
            else None
        )

        # Accept an original host name as a compatibility fallback,
        # but browser prompts normally expose only aliases.
        if (
            original_name is None
            and browser_name in available
        ):
            original_name = browser_name

        if original_name is None:
            raise ToolBridgeError(
                f"unknown external tool: {browser_name!r}"
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
                    name=original_name,
                    arguments=_canonical(arguments),
                ),
            )
        )

    return result
