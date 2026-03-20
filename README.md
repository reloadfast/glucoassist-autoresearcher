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
  "program_md": "...",
  "llm_provider": "ollama",
  "ollama_url": "http://localhost:11434",
  "ollama_model": "llama3.1:8b",
  "openai_url": "",
  "openai_api_key": "",
  "openai_model": "gpt-4o"
}
```

## LLM backends

- **Ollama (local)**: `llm_provider=ollama`
- **OpenAI-compatible**: `llm_provider=openai_compatible` — works with OpenAI, Anthropic
  (via proxy), Google Gemini, vLLM, LiteLLM, and others.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_PATH` | `/data/glucoassist.db` | Path to the shared SQLite database |

## Compatibility

GlucoAssist checks `GET /api/version` on every connection. See GlucoAssist release notes
for the minimum compatible version of this service.
