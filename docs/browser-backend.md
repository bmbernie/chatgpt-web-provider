# Browser Backend

## Browser backend setup

The browser backend uses a dedicated persistent Chromium profile. Do not reuse
your normal browser profile.

Environment:

```bash
CHATGPT_WEB_BACKEND=browser
CHATGPT_WEB_PROFILE_DIR=$HOME/.local/share/chatgpt-web-provider/chrome-profile
CHATGPT_WEB_HEADLESS=false
```

Open a visible setup browser and log in:

```bash
cd ~/github/chatgpt-web-provider
. .venv/bin/activate
set -a
. ~/.config/chatgpt-web-provider/env
set +a
CHATGPT_WEB_HEADLESS=false chatgpt-web-provider-browser-setup
```

In the opened browser:

1. log in to `chatgpt.com`
2. pass any Cloudflare/browser checks
3. select the desired model, for example GPT-5.5 High
4. press Enter in the terminal running `chatgpt-web-provider-browser-setup` to close and save the profile

Then start the API service with the same profile.

### Headless vs headed mode

In practice, ChatGPT often blocks headless Chromium with a Cloudflare page like:

```text
Just a moment...
```

If `GET /health` reports title `Just a moment...`, run the browser backend in
headed mode on the desktop session instead of headless.

For a user-launched service on Linux/X11, the launcher can export:

```bash
export DISPLAY=:0
export XAUTHORITY=$HOME/.Xauthority
CHATGPT_WEB_HEADLESS=false
```

The development deployment uses this pattern because headless mode reached
Cloudflare but headed mode reached normal ChatGPT and produced real responses.

### Browser backend health

`GET /health` returns browser hints:

```json
{
  "ok": true,
  "backend": "browser",
  "model": "chatgpt-5.6-sol-web",
  "title": "ChatGPT",
  "logged_in_hint": true
}
```

If `backend` is `mock`, you are not using the real browser worker. If responses
start with `[mock:...]`, you are still on the mock backend.
