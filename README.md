<h1 align="center">🛹 ollie-rl</h1>

<p align="center">
  <strong>An OpenAI-compatible chat-completions server with a built-in online GRPO training loop.</strong><br/>
  Drop it in front of any agent, post rewards, get a fine-tuned policy.
</p>

<p align="center">
  <a href="https://github.com/wsxiaoys/ollie-rl/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/wsxiaoys/ollie-rl/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/wsxiaoys/ollie-rl/actions/workflows/docker-publish.yml"><img alt="Docker" src="https://github.com/wsxiaoys/ollie-rl/actions/workflows/docker-publish.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
  <a href="https://github.com/wsxiaoys/ollie-rl/pkgs/container/ollie-rl"><img alt="Container" src="https://img.shields.io/badge/ghcr.io-ollie--rl-2496ED?logo=docker&logoColor=white"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.14+-3776AB?logo=python&logoColor=white">
  <img alt="Status" src="https://img.shields.io/badge/status-experimental-orange">
</p>

---

## Why ollie-rl?

Most RL libraries (`trl`, `verl`, `OpenRLHF`, `tinker-cookbook`) are **training
scripts**: you batch up trajectories offline, run a script, and hope the loss
curve goes down. But modern agents already speak one universal protocol —
**`POST /v1/chat/completions`** — and they already have a notion of success
(test passed, task completed, user thumbs-up).

`ollie-rl` closes that loop. It is an HTTP server that:

1. Exposes an **OpenAI-compatible** `/v1/chat/completions` endpoint.
2. Tags each completion with an `X-Run-Id` header that the agent passes through.
3. Accepts a **scalar reward** per run via `PUT /tuners/{id}/runs/{run_id}/reward`.
4. Implicitly forms **GRPO groups**, computes advantages, and triggers
   `train_step`s on a pluggable trainer backend (currently `tinker` or custom backends).

The result: any agent framework — whether it's LangGraph, CrewAI, ACP, your
homebrew loop, or `inspect-ai` — can become an RL training driver by swapping
its OpenAI base URL.

## How it works

The server orchestrates the classic synchronous-RL lifecycle and hides
GRPO group bookkeeping behind a tiny HTTP surface.

```mermaid
sequenceDiagram
    participant C as Agent / Worker
    participant API as ollie-rl
    participant T as Trainer backend

    C->>API: POST /tuners { recipe, datum_ids }
    API-->>C: { tuner_id }

    loop training step
        C->>API: POST /tuners/{id}/runs
        API-->>C: 200 { run_id, datum_id } or 204 + Retry-After
        loop one or more LLM turns
            C->>API: POST /openai/v1/chat/completions<br/>X-Tuner-Id, X-Run-Id
            API->>T: sample(...)
            T-->>API: ChatCompletion
            API-->>C: ChatCompletion
        end
        C->>API: PUT /tuners/{id}/runs/{run_id}/reward { reward }
        Note over API,T: server collects runs and rewards<br/>fires train_step when batch is ready
    end
```

For the full data model (Rollout / Run / ChatCompletion / Batch / Advantage),
see [`docs/data-model.md`](./.agents/skills/dev/references/data-model.md).
For the client lifecycle, see
[`docs/sync-rl.md`](./.agents/skills/dev/references/sync-rl.md).

## Quickstart

### 1. Run the server

```bash
docker compose -f deploy/docker-compose.yaml up -d
# server is now live at http://localhost:8000  (Swagger UI at /docs)
```

Or run from source:

```bash
uv sync
uv run poe dev
```

### 2. Create a tuner

```bash
curl -X POST http://localhost:8000/tuners \
  -H 'Content-Type: application/json' \
  -d '{
        "name": "banana-policy",
        "recipe": "grpo_16x32",
        "datum_ids": ["prompt-1", "prompt-2", "prompt-3"]
      }'
# => { "tuner_id": "tnr_...", "name": "banana-policy", "recipe": "grpo_16x32" }
```

### 3. Run an agent loop

You can drive your agent loop using the open-source terminal agent, **OpenCode CLI**. Configure it to route requests through the `ollie-rl` sidecar and map the required custom headers to environment variables.

1. Add a custom model provider in your `opencode.json` (either in the project root or globally in `~/.config/opencode/opencode.json`):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "ollie/tinker",
  "provider": {
    "ollie": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "ollie-rl",
      "options": {
        "baseURL": "http://localhost:8000/openai/v1",
        "apiKey": "any-key",
        "headers": {
          "X-Tuner-Id": "{env:X_TUNER_ID}",
          "X-Run-Id": "{env:X_RUN_ID}"
        }
      },
      "models": {
        "tinker": {}
      }
    }
  }
}
```

2. Drive the loop in your terminal:

```bash
# 1. Request a run assignment from the tuner
RUN_INFO=$(curl -s -X POST http://localhost:8000/tuners/tnr_.../runs)
export X_TUNER_ID="tnr_..."
export X_RUN_ID=$(echo $RUN_INFO | jq -r '.run_id')
export DATUM_ID=$(echo $RUN_INFO | jq -r '.datum_id')

# 2. Run OpenCode to execute a task using the live policy
opencode run "Solve this task: $DATUM_ID"

# 3. Score the run and submit the scalar reward
curl -X PUT http://localhost:8000/tuners/$X_TUNER_ID/runs/$X_RUN_ID/reward \
  -H 'Content-Type: application/json' \
  -d '{"reward": 1.0}'
```

The server batches **16 runs per `datum_id`** into a GRPO group, waits until
**32 groups (512 runs)** are ready, and then automatically fires a
`train_step` on the configured backend.

## How does this compare to `trl` / `verl` / `tinker`?

| | `ollie-rl` | `trl` / `verl` / `OpenRLHF` | `tinker-cookbook` |
|---|---|---|---|
| **Interface** | HTTP, OpenAI-compatible | Python script | Python script |
| **Drives your agent loop** | ✅ yes | ❌ you write a rollout collector | ❌ you write a rollout collector |
| **Online (sample ↔ train)** | ✅ implicit GRPO | ✅ (with effort) | ✅ |
| **Pluggable backend** | ✅ via `TrainerFactory` | varies | tinker only |
| **Framework-agnostic clients** | ✅ any OpenAI client | ❌ Python only | ❌ Python only |
| **Status** | experimental | mature | mature |

`ollie-rl` is not a replacement for `trl` — it's the **sidecar layer above it**.
You can imagine plugging `trl`, `verl`, or any custom trainer in behind the
`Trainer` protocol.

## Architecture

```
src/
├── ollie_rl/
│   ├── server/         FastAPI HTTP surface
│   ├── service/        TunerService — dispense_run, advantage math, maybe_train
│   ├── trainer/        Pluggable Trainer / TrainerFactory protocol
│   │   ├── types.py    The plugin contract
│   │   └── factory.py  Registry of registered TrainerFactories
│   ├── cookbook/       Declarative `Recipe`s (group_size, batch shape, …)
│   ├── db/             SQLAlchemy async models (SQLite by default, Postgres-ready)
│   └── types.py        HTTP DTOs
```

Key concepts:

- **Tuner** — one live training job; owns a policy and a `datum_pool`.
- **Run** — one attempt at a `datum_id`; carries the scalar reward.
- **Rollout** — a GRPO group of K runs sharing the same `datum_id`.
- **Recipe** — declarative algorithm knobs (`group_size`, `num_groups_per_batch`).
- **Trainer** — pluggable backend (`tinker` or custom backends).

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `sqlite+aiosqlite:///./data/db.sqlite` | SQLAlchemy async URL. Switch to `postgresql+asyncpg://...` for prod. |

## Status & Roadmap

`ollie-rl` is **pre-1.0 / experimental**. The HTTP surface is intentionally
small and is still evolving.

Planned:

- [ ] A `tinker` backend.
- [ ] A runnable end-to-end `examples/` directory with reward curves.
- [ ] MkDocs Material documentation site.
- [ ] Lightweight `ollie-rl-client` Python SDK on PyPI.
- [ ] vLLM / SGLang trainer adapters.
- [ ] Multi-step scheduler + reward replay.

See [`ROADMAP.md`](./ROADMAP.md) once it lands, or browse
[the issues](https://github.com/wsxiaoys/ollie-rl/issues).

## Development

```bash
uv sync --all-groups
pre-commit install --hook-type pre-commit --hook-type pre-push

uv run poe test          # pytest
uv run poe check         # ty type-check
uv run poe lint          # ruff
uv run poe format        # ruff fix + uv format
uv run poe dev           # uvicorn reload server
```

See [`CONTRIBUTING.md`](./CONTRIBUTING.md) for the full contributor guide.

## License

[MIT](./LICENSE) © Meng Zhang and contributors.

## Acknowledgements

- The GRPO algorithm originates from the
  [DeepSeekMath](https://arxiv.org/abs/2402.03300) paper.
- The pluggable `Trainer` shape is heavily inspired by
  [`tinker-cookbook`](https://github.com/thinking-machines-lab/tinker-cookbook).
- The OpenAI-compatible HTTP surface is what makes any agent framework a
  potential RL driver — kudos to the OpenAI API team for the de-facto standard.
