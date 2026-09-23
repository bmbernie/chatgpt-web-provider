# Security

## Security model

Default intended deployment:

- bind app to `127.0.0.1`
- expose through your own reverse proxy only when protected by API keys and/or additional access controls
- use a dedicated ChatGPT browser profile
- keep the browser worker away from SSH keys, Hermes secrets, customer projects, and production secrets
- do not auto-confirm high-impact shell/MCP actions
- do not sell this as a public API
- review ChatGPT/OpenAI terms before use

This repo does not provide shell/MCP tools yet. Add those behind explicit allowlists,
argument schemas, approval gates, and audit logs.
