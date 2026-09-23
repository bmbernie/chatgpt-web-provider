# Provider Runtime

## Provider status and queue

`GET /v1/provider/status` shows backend health and queue settings:

```bash
curl -s \
  -H "Authorization: Bearer $CHATGPT_WEB_API_KEY" \
  -H 'User-Agent: PostmanRuntime/7.45.0' \
  https://codex.guber.dev/v1/provider/status
```

Example shape:

```json
{
  "backend": "browser",
  "model": "chatgpt-5.6-sol-web",
  "health": {"ok": true, "backend": "browser", "title": "ChatGPT"},
  "queue": {
    "max_concurrent_requests": 1,
    "queue_timeout_seconds": 600,
    "in_flight_estimate": 0
  }
}
```

The browser backend is serialized by default. Keep concurrency at `1` unless you
add a real browser pool.
