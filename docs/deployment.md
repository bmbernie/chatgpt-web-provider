# Deployment

## Deployment notes from the development host

The development deployment used:

- app bound to `127.0.0.1:8791`
- Cloudflare Tunnel for `codex.guber.dev`
- API auth at the app layer
- dedicated ChatGPT profile under `~/.local/share/chatgpt-web-provider/chrome-profile`
- headed browser worker with `DISPLAY=:0` and `XAUTHORITY=$HOME/.Xauthority`
- cron/user launch scripts to keep app and tunnel running

These are operational notes, not requirements. You can run the same API behind
Caddy, nginx, Cloudflare Tunnel, Tailscale, or localhost only.
