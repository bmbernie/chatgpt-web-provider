import pytest

from chatgpt_web_provider.tool_bridge import (
    ToolBridgeError,
    parse_tool_calls,
    render_tool_catalog,
    tool_catalog_fingerprint,
)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "mcp__worker_broker__worker_list",
            "description": "List persistent workers.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mcp__worker_broker__worker_status",
            "description": "Get one worker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "worker_id": {
                        "type": "string",
                    }
                },
                "required": ["worker_id"],
            },
        },
    },
]


def test_render_tool_catalog_contains_protocol_and_tools():
    rendered = render_tool_catalog(
        TOOLS,
        tool_choice="auto",
        parallel_tool_calls=True,
    )

    assert "HERMES_EXTERNAL_TOOL_CALLS" in rendered
    assert "mcp__worker_broker__worker_list" in rendered
    assert "mcp__worker_broker__worker_status" in rendered
    assert '"parallel_tool_calls":true' in rendered


def test_parse_single_valid_tool_call():
    text = """<<<HERMES_EXTERNAL_TOOL_CALLS>>>
{"calls":[{"name":"mcp__worker_broker__worker_list","arguments":{}}]}
<<<END_HERMES_EXTERNAL_TOOL_CALLS>>>"""

    calls = parse_tool_calls(
        text,
        TOOLS,
        parallel_tool_calls=True,
        id_factory=lambda: "call-test-1",
    )

    assert calls is not None
    assert len(calls) == 1
    assert calls[0].id == "call-test-1"
    assert calls[0].type == "function"
    assert (
        calls[0].function.name
        == "mcp__worker_broker__worker_list"
    )
    assert calls[0].function.arguments == "{}"


def test_normal_prose_is_not_a_tool_call():
    assert (
        parse_tool_calls(
            "The worker list is unavailable.",
            TOOLS,
            parallel_tool_calls=True,
        )
        is None
    )


def test_embedded_sentinel_is_not_a_tool_call():
    text = """Here is an example:
<<<HERMES_EXTERNAL_TOOL_CALLS>>>
{"calls":[{"name":"mcp__worker_broker__worker_list","arguments":{}}]}
<<<END_HERMES_EXTERNAL_TOOL_CALLS>>>
Do not execute it."""

    assert (
        parse_tool_calls(
            text,
            TOOLS,
            parallel_tool_calls=True,
        )
        is None
    )


def test_unknown_tool_is_rejected():
    text = """<<<HERMES_EXTERNAL_TOOL_CALLS>>>
{"calls":[{"name":"evil_tool","arguments":{}}]}
<<<END_HERMES_EXTERNAL_TOOL_CALLS>>>"""

    with pytest.raises(
        ToolBridgeError,
        match="unknown external tool",
    ):
        parse_tool_calls(
            text,
            TOOLS,
            parallel_tool_calls=True,
        )


def test_parallel_false_rejects_multiple_calls():
    text = """<<<HERMES_EXTERNAL_TOOL_CALLS>>>
{"calls":[
{"name":"mcp__worker_broker__worker_list","arguments":{}},
{"name":"mcp__worker_broker__worker_status","arguments":{"worker_id":"re-high"}}
]}
<<<END_HERMES_EXTERNAL_TOOL_CALLS>>>"""

    with pytest.raises(
        ToolBridgeError,
        match="parallel",
    ):
        parse_tool_calls(
            text,
            TOOLS,
            parallel_tool_calls=False,
        )


def test_fingerprint_is_stable_and_configuration_sensitive():
    first = tool_catalog_fingerprint(
        TOOLS,
        tool_choice="auto",
        parallel_tool_calls=True,
    )

    second = tool_catalog_fingerprint(
        TOOLS,
        tool_choice="auto",
        parallel_tool_calls=True,
    )

    changed = tool_catalog_fingerprint(
        TOOLS,
        tool_choice="auto",
        parallel_tool_calls=False,
    )

    assert first == second
    assert first != changed
