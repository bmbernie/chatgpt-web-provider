from chatgpt_web_provider.browser_prompts import (
    ASSISTANT_EXTERNAL_TOOL_CALLS_HEADING,
    EXTERNAL_TOOL_RESULT_HEADING,
    render_transcript,
)
from chatgpt_web_provider.models import (
    ChatMessage,
    ToolCall,
    ToolFunctionCall,
)


def test_render_transcript_preserves_external_tool_labels():
    messages = [
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="call-1",
                    function=ToolFunctionCall(
                        name="worker_list",
                        arguments="{}",
                    ),
                )
            ],
        ),
        ChatMessage(
            role="tool",
            content="five workers",
            tool_call_id="call-1",
            name="worker_list",
        ),
    ]

    rendered = render_transcript(messages)

    assert (
        ASSISTANT_EXTERNAL_TOOL_CALLS_HEADING
        in rendered
    )

    assert '"name":"worker_list"' in rendered

    assert (
        EXTERNAL_TOOL_RESULT_HEADING
        + " id=call-1 name=worker_list:\n"
        + "five workers"
        in rendered
    )


def test_host_bridge_disambiguates_external_from_native_tools():
    from chatgpt_web_provider.browser_prompts import (
        render_host_bridge_context,
    )

    messages = [
        ChatMessage(
            role="developer",
            content="You are Hermes Agent.",
        ),
    ]

    bridge = render_host_bridge_context(messages)

    assert "NOT native ChatGPT tools" in bridge
    assert "not expected to appear in the native ChatGPT tool set" in bridge
    assert "Do not use the native ChatGPT tool set" in bridge
    assert "authoritative tool set for this request" in bridge
    assert "emit the exact external tool-call envelope" in bridge
    assert "provider will intercept that envelope" in bridge
