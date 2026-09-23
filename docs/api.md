# API

## API base URL

Local development:

```text
http://127.0.0.1:8791
```

Example hosted endpoint:

```text
https://codex.example.com
```

OpenAI-compatible base URL for clients that ask for one:

```text
https://codex.example.com/v1
```

## Chat Completions API

Use this for normal chat clients:

```http
POST /v1/chat/completions
```

Non-streaming request:

```bash
curl --location 'https://codex.example.com/v1/chat/completions' \
  --header 'Content-Type: application/json' \
  --header 'User-Agent: PostmanRuntime/7.45.0' \
  --header "Authorization: Bearer $CHATGPT_WEB_API_KEY" \
  --data '{
    "model": "chatgpt-5.6-sol-web",
    "level": "high",
    "messages": [
      {"role": "system", "content": "Reply concisely."},
      {"role": "user", "content": "Say exactly: pong"}
    ],
    "stream": false
  }'
```

Response shape:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1782610607,
  "model": "chatgpt-5.6-sol-web",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "pong"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  }
}
```

Token usage is exact for the mock backend and currently best-effort/zero for the
browser backend unless token accounting is added separately.

### Streaming chat completions

Set `stream: true`:

```bash
curl --no-buffer --location 'https://codex.example.com/v1/chat/completions' \
  --header 'Content-Type: application/json' \
  --header 'User-Agent: PostmanRuntime/7.45.0' \
  --header "Authorization: Bearer $CHATGPT_WEB_API_KEY" \
  --data '{
    "model": "chatgpt-5.6-sol-web",
    "messages": [{"role": "user", "content": "Say pong"}],
    "stream": true
  }'
```

The server returns OpenAI-style SSE chunks:

```text
data: {"id":"chatcmpl-...","object":"chat.completion.chunk",...}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk",..."finish_reason":"stop"...}

data: [DONE]
```

The current streaming shim waits for the browser completion and then emits the
answer as SSE. It is protocol-compatible, not true token-by-token browser
streaming yet.

## Responses API

The project also exposes an OpenAI Responses-like endpoint:

```http
POST /v1/responses
```

Example:

```bash
curl --location 'https://codex.example.com/v1/responses' \
  --header 'Content-Type: application/json' \
  --header 'User-Agent: PostmanRuntime/7.45.0' \
  --header "Authorization: Bearer $CHATGPT_WEB_API_KEY" \
  --data '{
    "model": "chatgpt-5.6-sol-web",
    "input": "Say exactly: pong",
    "stream": false
  }'
```

Response includes `output_text`:

```json
{
  "id": "resp_...",
  "object": "response",
  "status": "completed",
  "model": "chatgpt-5.6-sol-web",
  "output_text": "pong"
}
```

Streaming for `/v1/responses` is not implemented yet.

## Model and level selection

The provider can advertise more than one ChatGPT model and more than one effort/level.

Read the configured capabilities:

```http
GET /v1/provider/capabilities
```

Example response shape:

```json
{
  "backend": "browser",
  "default_model": "chatgpt-5.6-sol-web",
  "models": [
    {"id": "chatgpt-5.6-sol-web", "display_name": "GPT-5.6 Sol", "default": true},
    {"id": "gpt-5.6-terra", "display_name": "5.6 Terra", "default": false},
    {"id": "gpt-5.6-luna", "display_name": "5.6 Luna", "default": false},
    {"id": "gpt-5.5", "display_name": "GPT-5.5", "default": false},
    {"id": "gpt-5.4", "display_name": "GPT-5.4", "default": false},
    {"id": "gpt-5.4-mini", "display_name": "5.4 Mini", "default": false},
    {"id": "gpt-5.3", "display_name": "GPT-5.3", "default": false},
    {"id": "gpt-5.3-codex-spark", "display_name": "5.3 Codex Spark", "default": false},
    {"id": "o3", "display_name": "o3", "default": false}
  ],
  "levels": [
    {"id": "auto", "display_name": "Auto", "default": true},
    {"id": "fast", "display_name": "Fast", "default": false},
    {"id": "standard", "display_name": "Standard", "default": false},
    {"id": "high", "display_name": "High", "default": false}
  ],
  "new_session": {"body_field": "new_session", "header": "X-New-Session"}
}
```

Choose a model per request with `model`.

Choose an effort/level with either:

- `level`
- `reasoning_effort`

Both map to the same provider-side field.

If the requested model or level is not in the configured allowlist, the API returns `400`.


By default, the browser backend continues whatever ChatGPT conversation the
browser worker is currently on.

To force a fresh ChatGPT conversation before the prompt, use either request body:

```json
{
  "model": "chatgpt-5.6-sol-web",
  "messages": [{"role": "user", "content": "Start fresh"}],
  "new_session": true
}
```

or header:

```http
X-New-Session: true
```

Full curl with body flag:

```bash
curl --location 'https://codex.example.com/v1/chat/completions' \
  --header 'Content-Type: application/json' \
  --header 'User-Agent: PostmanRuntime/7.45.0' \
  --header "Authorization: Bearer $CHATGPT_WEB_API_KEY" \
  --data '{
    "model": "chatgpt-5.6-sol-web",
    "messages": [{"role":"user","content":"Say exactly: fresh-session-pong"}],
    "stream": false,
    "new_session": true
  }'
```

Full curl with header:

```bash
curl --location 'https://codex.example.com/v1/chat/completions' \
  --header 'Content-Type: application/json' \
  --header 'User-Agent: PostmanRuntime/7.45.0' \
  --header "Authorization: Bearer $CHATGPT_WEB_API_KEY" \
  --header 'X-New-Session: true' \
  --data '{
    "model": "chatgpt-5.6-sol-web",
    "messages": [{"role":"user","content":"Say exactly: fresh-session-pong"}],
    "stream": false
  }'
```

`new_session` is supported on both `/v1/chat/completions` and `/v1/responses`.
