# Security Policy

## Supported versions

Security reports for this fork should be tested against the current `main`
branch when possible. Older revisions may not receive security backports.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability.

Use GitHub's private vulnerability reporting feature from the repository's
**Security** tab and select **Report a vulnerability**.

Please include, where applicable:

- the affected version or commit
- operating system and Python version
- browser and browser channel
- the affected provider, browser-session, or profile behavior
- reproduction steps or a minimal proof of concept
- the security impact
- any known mitigations or prerequisites

Do not include real account credentials, session tokens, browser cookies, or
unrelated private data in a report.

## Security boundaries

This project controls a local browser-backed provider and may use a persistent
browser profile. Security reports are particularly useful for issues such as:

- unintended disclosure of browser-session or profile data
- credential, cookie, or token exposure
- unintended access across local-user or profile boundaries
- command or code execution outside the intended browser-automation boundary
- network exposure that bypasses the documented local deployment boundary

Local control of an explicitly configured browser profile or configuration path
by the same authorized operating-system user is not by itself considered a
privilege-boundary violation.

## Upstream issues

This repository is a fork of `guberm/chatgpt-web-provider`.

If a vulnerability discovered here appears to originate in upstream code,
report it privately here rather than publishing technical details in an issue.
The maintainers can coordinate with upstream when appropriate.

## Disclosure

There is currently no fixed response or remediation SLA. Public disclosure of
an unresolved vulnerability should be coordinated through the private report.
