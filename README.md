# glucoassist-autoresearcher

Standalone research engine for [GlucoAssist](https://github.com/reloadfast/glucoassist).

Autonomously proposes, evaluates, and promotes improvements to the glucose forecasting model
using a locally-hosted or commercial LLM. GlucoAssist calls this service over HTTP when
`AUTORESEARCHER_URL` is configured — it is entirely optional.

## Architecture

```
GlucoAssist container  ──HTTP──►  glucoassist-autoresearcher container
      │                                         │
      └──── shared volume: /data/glucoassist.db ┘
```

The autoresearcher reads glucose data from the shared SQLite file and writes experiment
results back to the same file. GlucoAssist reads those results and displays them in the
Research page.

## Quick start

```yaml
services:
  glucoassist:
    image: ghcr.io/reloadfast/glucoassist:latest
    volumes:
      - /path/to/data:/data
    environment:
      AUTORESEARCHER_URL: http://autoresearcher:8001

  autoresearcher:          # optional — remove to disable the research feature
    image: ghcr.io/reloadfast/glucoassist-autoresearcher:latest
    volumes:
      - /path/to/data:/data    # same volume as GlucoAssist
    environment:
      DATABASE_PATH: /data/glucoassist.db
```

## Deploying

### Publishing a new image (GHCR)

Images are hosted on the GitHub Container Registry — no Docker Hub required. The
`release.yml` workflow builds and pushes automatically when you tag a release:

```bash
git tag v0.1.0
git push origin v0.1.0
```

This publishes both `ghcr.io/reloadfast/glucoassist-autoresearcher:v0.1.0` and `:latest`.

### Keeping the image up to date

Use [Watchtower](https://containrrr.dev/watchtower/) to pull new releases automatically:

```yaml
watchtower:
  image: containrrr/watchtower
  volumes:
    - /var/run/docker.sock:/var/run/docker.sock
  command: --interval 86400 glucoassist autoresearcher
```

### Unraid setup

The autoresearcher is a second container alongside GlucoAssist — it is not part of the
main Community Applications template. Add it manually in Unraid's Docker UI:

| Setting | Value |
|---|---|
| Image | `ghcr.io/reloadfast/glucoassist-autoresearcher:latest` |
| Network | Same as GlucoAssist (e.g. `bridge` or a custom network) |
| Port | `8001` — internal only; GlucoAssist talks to it directly, no host exposure needed |
| Volume | `/data` → same host path as GlucoAssist's `/data` volume |
| Env vars | None required — LLM config is stored in the shared DB by GlucoAssist |

Then add one environment variable to your **GlucoAssist** container:

```
AUTORESEARCHER_URL=http://<autoresearcher-container-ip>:8001
```

### Wiring it up

1. Start both containers.
2. In GlucoAssist, go to **Settings → Research Service**.
3. Configure your LLM provider (Ollama or OpenAI-compatible), endpoint, and model.
4. Click **Test Connection** — you should see `✓ Connected — sidecar v0.1.0`.
5. Open the **Research** page and click **Run Now** to start your first experiment.

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/version | Service version |
| POST | /api/run | Start a research run |
| GET | /api/status | Current run state |
| DELETE | /api/run | Cancel current run |
| GET | /api/log | Experiment results |

### POST /api/run body

```json
{
  "n_experiments": 10,
  "llm_type": "ollama",
  "llm_endpoint": "http://ollama:11434",
  "llm_model": "llama3.1:8b",
  "llm_api_key": null,
  "program_md": "# Optional — omit to use the bundled default"
}
```

## LLM backends

- **Ollama (local)**: `llm_type=ollama` — requires Ollama reachable from the container.
- **OpenAI-compatible**: `llm_type=openai_compatible` — works with OpenAI, vLLM, LiteLLM,
  and any endpoint implementing `/v1/chat/completions`.

LLM credentials are stored encrypted in the shared database by GlucoAssist and passed in
the `/api/run` body — they are never stored in environment variables.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_PATH` | `/data/glucoassist.db` | Path to the shared SQLite database |

## Compatibility

GlucoAssist checks `GET /api/version` on every connection and warns if the sidecar version
is below `AUTORESEARCHER_MIN_VERSION`. See
[docs/autoresearcher.md](https://github.com/reloadfast/glucoassist/blob/main/docs/autoresearcher.md)
in the main repo for the compatibility matrix, upgrade checklist, and rollback instructions.
