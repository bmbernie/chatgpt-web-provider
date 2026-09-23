# Postman

## Postman project

Import files from `postman/`:

1. `chatgpt-web-provider.postman_collection.json`
2. `chatgpt-web-provider.template.postman_environment.json`

Set environment variable:

```text
api_key = <api-key>
```

The collection includes:

- Health
- Models
- Provider Status
- Chat Completions - non-stream
- Chat Completions - new session
- Chat Completions - stream SSE
- Responses API

The collection sets:

```http
User-Agent: {{user_agent}}
```

with default:

```text
PostmanRuntime/7.45.0
```

This matters because Cloudflare may block generic clients such as Python's
default `urllib` user agent while allowing Postman/curl-style user agents.
