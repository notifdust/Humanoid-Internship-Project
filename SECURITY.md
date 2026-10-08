# Security policy

## Supported versions

This is a research / challenge repository. Only the latest `main` branch is maintained.

## Reporting a vulnerability

If you discover a security issue (e.g. accidental commit of credentials, unsafe deserialization of untrusted checkpoints):

1. **Do not** open a public GitHub issue.
2. Contact the repository owner privately via GitHub (Security Advisories if enabled, or a private message).
3. Include steps to reproduce and impact assessment if possible.

We will acknowledge the report and work on a fix or disclosure plan as appropriate.

## Secrets hygiene

Never commit `.env` files, API keys, cloud credentials, or private dataset URLs with embedded tokens. Use `.env.example` for required variable names only.
