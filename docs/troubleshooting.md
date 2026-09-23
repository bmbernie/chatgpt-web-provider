# Troubleshooting

### Response starts with `[mock:...]`

You are on the mock backend. Set:

```bash
CHATGPT_WEB_BACKEND=browser
```

and restart the service.

### `/health` title is `Just a moment...`

The browser is stuck at Cloudflare. Use headed mode with a real desktop session:

```bash
CHATGPT_WEB_HEADLESS=false
export DISPLAY=:0
export XAUTHORITY=$HOME/.Xauthority
```

Then restart and log in/pass checks in the dedicated profile if needed.

### Public endpoint returns Cloudflare 502/530/1033

Separate origin from edge:

```bash
curl http://127.0.0.1:8791/health
pgrep -af cloudflared
```

If local health works but public fails, restart/check the Cloudflare Tunnel. If
local health fails, fix the app first.

### Auth returns 401/403

Use one of:

```http
Authorization: Bearer $CHATGPT_WEB_API_KEY
X-API-Key: $CHATGPT_WEB_API_KEY
```

If using Postman, make sure `api_key` is set in the active environment. If both
headers are present, a valid bearer token still works even if `X-API-Key` is an
unresolved `{{api_key}}` placeholder.

### Postman/curl works but Python urllib gets blocked

Set a non-default user agent, for example:

```http
User-Agent: PostmanRuntime/7.45.0
```
