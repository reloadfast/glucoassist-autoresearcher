# CLAUDE.md — glucoassist-autoresearcher

## Autonomy Rules
Proceed without asking for confirmation on all routine operations. Only stop for:
- Irreversible data loss (dropping DB tables, `rm -rf`, overwriting uncommitted work)
- Pushing to remote / opening PRs
- Breaking public API contracts that affect GlucoAssist integration
- Adding new external services or third-party dependencies not already in the manifest

Proceed freely without prompting for:
- Reading, creating, editing, or deleting files anywhere in this repo
- Running tests, linters, formatters, security scans
- Creating git commits (but not pushing)
- Installing packages into the local venv
- Creating branches
- Any action that is fully reversible with `git checkout` or `git reset`

## Token Efficiency Rules
- Be concise. No preamble, no summaries unless asked.
- Reference file:line instead of reproducing code blocks.
- Use bullet lists, not prose paragraphs.
- Skip "I will now..." or "Here is the..." phrases.
- When editing, show only changed lines with minimal context.
- Batch related file reads; avoid re-reading already-known files.

## Project Overview
- **What it is:** Standalone research sidecar for [GlucoAssist](https://github.com/reloadfast/glucoassist) — autonomously proposes, evaluates, and promotes improvements to the glucose forecasting model using an LLM
- **Relationship to GlucoAssist:** This container is a sidecar of GlucoAssist and exists only to serve it. Both repos must always be considered together. Any API contract change here breaks GlucoAssist's Research page. When in doubt, check the GlucoAssist repo before changing endpoints or database schema.
- **Target user:** Self-hosters running GlucoAssist on Unraid (or Docker Compose); the service is entirely optional — GlucoAssist works without it
- **Key constraints:** No UI; pure REST API. Shares a SQLite database volume with GlucoAssist. LLM calls are ad-hoc (never scheduled — Ollama may be offline)
- **Deployment:** Docker container, GHCR (`ghcr.io/reloadfast/glucoassist-autoresearcher`), port 8001 internal

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| API framework | FastAPI + uvicorn |
| Database | SQLite (shared volume with GlucoAssist) |
| ML | scikit-learn (Ridge), LightGBM |
| Data | pandas, numpy |
| LLM integration | requests → Ollama API or any OpenAI-compatible endpoint |
| Linting | ruff |
| Testing | pytest, pytest-asyncio, httpx |
| Container registry | GHCR (`ghcr.io/reloadfast/glucoassist-autoresearcher`) |
| CI/CD | GitHub Actions (ci.yml, release.yml) |

## Architecture

```
GlucoAssist container
  ├── HTTP → http://autoresearcher:8001  (AUTORESEARCHER_URL env var)
  └── shared volume: /data/glucoassist.db
                            │
              glucoassist-autoresearcher :8001
                            │
                Ollama (local) or OpenAI-compatible LLM
```

- GlucoAssist reads experiment results from `autoresearcher_log` table in the shared DB
- This service writes glucose readings from `glucose_readings` table (written by GlucoAssist)
- Promoted model configs are saved as `promoted_model_config.json` in the data volume

## Project Structure

```
autoresearch/
├── .github/
│   └── workflows/
│       ├── ci.yml          # test on push/PR
│       └── release.yml     # build + push to GHCR on tag push
├── app/
│   ├── main.py             # FastAPI app, routes, DB init
│   └── services/
│       ├── autoresearcher.py       # research loop, CV, LLM calls
│       └── autoresearcher_program.md  # prompt injected into LLM
├── tests/
│   ├── test_api.py
│   └── test_service.py
├── unraid/
│   └── ga-autoresearcher.xml  # gitignored — Unraid CA template
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```

## Sensitive Data Rules
- NEVER commit `.env`, secrets, tokens, API keys, or passwords
- All secrets via environment variables; document in `.env.example` with placeholder values only
- The SQLite database file is gitignored
- Deployment-specific config files are gitignored
- OpenAI/LLM API keys are passed at request time via the API body — never store them in the container environment or logs

## Environment Variables

```
DATABASE_PATH=/data/glucoassist.db  # path to the shared SQLite DB inside the container
```

## Unraid Template
The Unraid Community Applications template lives at `unraid/ga-autoresearcher.xml` (gitignored — internal use and local testing only).
- **Never reference this file in public-facing documentation** (README, CONTRIBUTING, or any user-visible content)
- Update the template as part of acceptance criteria whenever any of the following change: ports, env vars, volume mounts, container name
- All env vars in the template must stay in sync with `.env.example`
- After structural changes, test by importing the template into Unraid CA

## Testing Requirements
- Target: ≥80% line coverage; tests in `tests/`
- All tests must pass before merge to main
- CI runs `pytest tests/ -v` on every push and PR
- No `# noqa` without an inline justification comment (existing ones in main.py are acceptable)

## Security
- Fail CI on HIGH severity static analysis findings
- No hardcoded credentials anywhere in the codebase
- LLM API keys must never appear in logs, DB records, or error messages

## Version Visibility
This is an API-only service with no browser UI. The version is exposed via `GET /api/version` and consumed by GlucoAssist for compatibility checks.

**Source of truth:** `VERSION` constant in `app/main.py:22`. When a `[project]` section is added to `pyproject.toml`, migrate to `importlib.metadata.version("glucoassist-autoresearcher")` and remove the hardcoded constant.

**Acceptance criteria — every release PR must satisfy all of the following before merge:**
- [ ] `VERSION` in `app/main.py` bumped following semver (patch · minor · major)
- [ ] `GET /api/version` returns the new version string (covered by test or manually verified)
- [ ] No other version strings hardcoded anywhere (`grep -r 'version' app/` to check)
- [ ] `CHANGELOG.md` entry written (use [Keep a Changelog](https://keepachangelog.com) format)
- [ ] Git tag `vx.y.z` created and pushed after release PR merges (`git tag vx.y.z && git push origin vx.y.z`)

## Backlog / Roadmap Conventions
- All backlog, roadmap, and milestone tracking files must use codified IDs for every item (e.g. `SEC-001`, `BUG-003`, `UX-011`)
- ID format: `<CATEGORY>-<NNN>` — category is uppercase, number is zero-padded to 3 digits
- IDs must never be reused or renumbered once assigned; retired items stay in the file marked `[DONE]` or `[DROPPED]`
- When referencing a backlog item in a commit message, PR, or comment, always use its ID
- New items are appended at the bottom of their category table; do not insert mid-table to avoid ID churn
- Backlog files are gitignored — they are local working documents, not public artefacts

## Git Conventions
- Branch prefixes: `feature/`, `fix/`, `chore/`, `docs/`, `release/`
- Commits: conventional commits (`feat:`, `fix:`, `chore:`, `test:`, `docs:`, `release:`)
- PRs require CI green before merge
- main branch = deployable state at all times
- Version bump + CHANGELOG entry are **required acceptance criteria** for any release PR
- When creating GitHub issues, always add them to the project roadmap if one exists
- When merging a PR, close all issues it resolves

## Parallel Agents
Multiple Claude agents may work on this repo simultaneously on separate branches. To avoid cross-contamination:
- Before fixing any CI failure, run `git diff main...HEAD -- <file>` to confirm the offending code is within your branch's diff
- If the failure is in a file you did not touch, do NOT fix it in the current branch — create a separate `fix/` branch targeting main and open its own PR
- Each branch/PR must own only the changes scoped to its issue; never absorb unrelated fixes to make CI green
