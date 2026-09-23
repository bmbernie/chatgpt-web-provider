# Features

- `GET /health`
- `GET /v1/models`
- `GET /v1/provider/capabilities`
- `GET /v1/provider/status`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- non-stream and SSE streaming support for `/v1/chat/completions`
- Bearer-token auth and `X-API-Key` support
- ignores unresolved Postman placeholders like `X-API-Key: $CHATGPT_WEB_API_KEY` when a valid bearer token is present
- `new_session` option for forcing a fresh ChatGPT conversation
- mock backend for tests/local smoke checks
- browser backend using Playwright persistent Chromium profile
- simple in-process request queue with provider status endpoint
- secret redaction helpers
- example systemd/Caddy/Postman assets
