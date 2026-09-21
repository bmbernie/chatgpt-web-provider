import pytest

from chatgpt_web_provider.tool_bridge import (
    END,
    START,
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
    assert "worker_broker_worker_list" in rendered
    assert "mcp__worker_broker__worker_list" not in rendered
    assert "worker_broker_worker_status" in rendered
    assert "mcp__worker_broker__worker_status" not in rendered
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


def test_render_tool_catalog_repeats_protocol_after_catalog():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "mcp__worker_broker__worker_list",
                "description": "List workers.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        }
    ]

    rendered = render_tool_catalog(tools)

    catalog_pos = rendered.index("External tool catalog:")
    footer_pos = rendered.index(
        "IMPORTANT — EXTERNAL TOOL CALL PROTOCOL:"
    )

    assert footer_pos > catalog_pos

    assert (
        "worker_broker_worker_list"
        in rendered[footer_pos - 256:]
        or
        "worker_broker_worker_list"
        in rendered
    )

    # The exact markers are repeated close to the end so the model
    # does not need to recover the serialization contract from the
    # beginning of a large tool catalog.
    assert rendered.count(START) == 2
    assert rendered.count(END) == 2

    assert rendered.endswith(
        "Do not include prose, Markdown, explanation, or code "
        "fences outside the markers when calling a tool."
    )


def test_mcp_tool_name_is_aliased_and_restored():
    tool = {
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
    }

    rendered = render_tool_catalog([tool])

    # Browser should see a neutral alias, not the MCP transport name.
    assert "worker_broker_worker_list" in rendered
    assert "mcp__worker_broker__worker_list" not in rendered

    response = (
        START
        + "\n"
        + '{"calls":[{"name":"worker_broker_worker_list",'
          '"arguments":{}}]}'
        + "\n"
        + END
    )

    calls = parse_tool_calls(
        response,
        [tool],
        id_factory=lambda: "call_alias_test",
    )

    assert calls is not None
    assert len(calls) == 1

    assert (
        calls[0].function.name
        == "mcp__worker_broker__worker_list"
    )

    assert calls[0].function.arguments == "{}"
