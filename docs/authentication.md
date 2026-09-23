# Authentication

Authenticated endpoints accept either header:

```http
Authorization: Bearer $CHATGPT_WEB_API_KEY
```

or:

```http
X-API-Key: $CHATGPT_WEB_API_KEY
```

If both are present, the server accepts any valid supplied token. This is useful
for Postman exports where `X-API-Key: $CHATGPT_WEB_API_KEY` may remain unresolved while
`Authorization: Bearer $CHATGPT_WEB_API_KEY` is valid.
