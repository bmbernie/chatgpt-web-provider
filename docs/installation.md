# Installation

## Quick start

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
cp .env.example .env
python -m pytest
CHATGPT_WEB_API_KEYS=dev-token CHATGPT_WEB_BACKEND=mock chatgpt-web-provider
```

Smoke test with the mock backend:

```bash
curl -s http://127.0.0.1:8791/health

curl -s \
  -H "X-API-Key: $CHATGPT_WEB_API_KEY" \
  http://127.0.0.1:8791/v1/models

curl -s \
  -H "X-API-Key: $CHATGPT_WEB_API_KEY" \
  -H 'Content-Type: application/json' \
  http://127.0.0.1:8791/v1/chat/completions \
  -d '{"model":"chatgpt-5.6-sol-web","messages":[{"role":"user","content":"Say pong"}]}'

curl -s \
  -H "X-API-Key: $CHATGPT_WEB_API_KEY" \
  http://127.0.0.1:8791/v1/provider/status
```
