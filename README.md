<h1 align="center">🛹 ollie-rl</h1>

<p align="center">
  <strong>Fine-tune the agent you already have — by pointing it at a new URL.</strong><br/>
  An OpenAI-compatible chat-completions server with a built-in online GRPO loop.
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

## Live demo

Explore the [`tuning-code-contests` demo](https://demo.ollie-rl.tabbyml.com),
a read-only deployment backed by a completed training run. The dashboard,
API documentation, and training results are available to browse; state-changing
requests and background training are disabled.

## Train your agent. Not your training loop.

Your agent already speaks one universal protocol — `POST /v1/chat/completions` —
and it already has a notion of success (test passed, task completed, user
thumbs-up). `ollie-rl` is the **drop-in sidecar** that turns those two things
into an online GRPO training signal. **Zero agent code changes.**

**What you don't have to write:**

- ❌ A rollout collector
- ❌ A dataset loader / replay buffer
- ❌ An offline training script
- ❌ A custom RL framework integration per agent

**What you write:**

- ✅ A reward function (one `PUT` per task)
- ✅ A list of `datum_id`s (your prompts / tasks)

That's it. Any agent that can change its OpenAI base URL — LangGraph, CrewAI,
OpenCode, `inspect-ai`, ACP, your homebrew loop — becomes an RL training
driver. The server forms GRPO groups, computes advantages, and fires
`train_step`s on a pluggable backend (`tinker` and custom backends) on your
behalf.

## 30-second demo: train OpenCode CLI

[OpenCode CLI](https://opencode.ai) is an open-source terminal agent. Without
patching a single line of OpenCode, you can fine-tune the policy it drives by
adding one provider block to `opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "ollie/tinker",
  "provider": {
    "ollie": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "ollie-rl",
      "options": {
        "baseURL": "http://localhost:8000/tuners/{env:TUNER_ID}/runs/{env:RUN_ID}/openai/v1",
        "apiKey": "any-key"
      },
      "models": { "tinker": {} }
    }
  }
}
```

Then drive the GRPO loop from your shell — create a tuner once, then request
a run, let the agent solve the task, and score it:

```bash
# One-time: create a tuner over your task list
TUNER_ID=$(curl -s -X POST http://localhost:8000/tuners \
  -H 'Content-Type: application/json' \
  -d '{"name":"my-policy","recipe":"grpo_16x32","trainer":"tinker","train_datum_ids":["task-1","task-2","task-3"]}' \
  | jq -r .tuner_id)

# Per run: request a run assignment
RUN=$(curl -s -X POST http://localhost:8000/tuners/$TUNER_ID/runs)
export TUNER_ID
export RUN_ID=$(jq -r .run_id  <<< "$RUN")
DATUM_ID=$(jq   -r .datum_id <<< "$RUN")

# Run the agent against the live (and learning) policy
opencode run "Solve this task: $DATUM_ID"

# Score the run — the server learns from it implicitly
curl -X PUT http://localhost:8000/tuners/$TUNER_ID/runs/$RUN_ID/reward \
  -H 'Content-Type: application/json' \
  -d '{"reward": 1.0}'
```

Loop the per-run block. Every **16** scored runs of a given prompt form a GRPO group;
every **32** groups (= 512 runs) trigger a `train_step` automatically. Your
agent is being fine-tuned while it's being used.

## How it works

```mermaid
sequenceDiagram
    participant C as Your agent
    participant API as ollie-rl
    participant T as Trainer backend

    C->>API: POST /tuners { name, recipe, trainer, train_datum_ids }
    API-->>C: { tuner_id, resolved recipe }

    loop training step
        C->>API: POST /tuners/{id}/runs
        API-->>C: 200 { run_id, datum_id }  or  204 + Retry-After
        loop one or more LLM turns
            C->>API: POST /tuners/{id}/runs/{run_id}/openai/v1/chat/completions
            API->>T: sample(...)
            T-->>API: ChatCompletion
            API-->>C: ChatCompletion
        end
        C->>API: PUT /tuners/{id}/runs/{run_id}/reward { reward }
        Note over API,T: server collects runs and rewards<br/>fires train_step when batch is ready
    end
```

Concepts the server hides for you:

- **Tuner** — one live training job; owns a policy and a `datum_pool`.
- **Run** — one rewarded attempt at a `datum_id`; it may contain multiple
  trajectories (for example, main-agent and subagent trajectories).
- **Rollout** — a GRPO group of K runs sharing the same `datum_id`.
- **Recipe** — immutable, tuner-level algorithm knobs such as `group_size` and
  `num_groups_per_batch`. Pass a preset name, or an object containing fields to
  override from the built-in defaults:

  ```json
  {
    "recipe": {
      "group_size": 8,
      "num_groups_per_batch": 16
    }
  }
  ```

  Ollie persists the fully resolved recipe, so later default or preset changes
  do not alter an existing tuner. To prevent workers from producing rollout
  data that would become stale before training can consume it, run dispensing
  is capped at:

  ```text
  max outstanding runs =
      (max_off_policy_generation + 1)
      * group_size
      * num_groups_per_batch
  ```

  Outstanding runs include active leases and rewarded-but-untrained runs;
  expired, rejected, trained, and evaluation runs do not consume this budget.
  Once the limit is reached, the run endpoint returns `204` with
  `Retry-After` until capacity becomes available.
- **Trainer** — the pluggable backend (`tinker`, or your own).

For the full data model, see
[`data-model.md`](./.agents/skills/dev/references/data-model.md).
For the wire protocol, see
[`sync-rl.md`](./.agents/skills/dev/references/sync-rl.md).

## Run the server

Boot ollie-rl on `http://localhost:8000` (Swagger UI at `/docs`) and the
demo above is ready to go:

```bash
docker compose -f deploy/docker-compose.yaml up -d
```

Or from source, for local development:

```bash
uv sync
uv run poe dev
```

## How does this compare to `trl` / `verl` / `OpenRLHF`?

| | `ollie-rl` | `trl` / `verl` / `OpenRLHF` |
|---|---|---|
| **Interface** | HTTP, OpenAI-compatible | Python script |
| **Drives your agent loop** | ✅ yes — bring your own | ❌ you write a rollout collector |
| **Online (sample ↔ train)** | ✅ implicit GRPO | ✅ (with effort) |
| **Pluggable backend** | ✅ via `TrainerFactory` | varies |
| **Framework-agnostic clients** | ✅ any OpenAI client | ❌ Python only |
| **Status** | experimental | mature |

`ollie-rl` is not a replacement for `trl` — it's the **sidecar layer above
it**. You can imagine plugging `trl`, `verl`, or any custom trainer in behind
the `Trainer` protocol.

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `sqlite+aiosqlite:///./data/db.sqlite` | SQLAlchemy async URL. Switch to `postgresql+asyncpg://...` for prod. |
| `OLLIE_READ_ONLY` | `false` | Run as a read-only demo: allow the dashboard, docs, and other reads while rejecting state-changing HTTP requests with `403` and disabling background training. |

Set `OLLIE_READ_ONLY` to `1`, `true`, `yes`, or `on` (case-insensitive) to
enable it. The mode is disabled by default, so existing deployments retain their
normal training behavior.

### Gemini MSRL authentication

`GeminiMsrlClient` calls the Google API directly and resolves authentication in
this order:

1. An explicitly supplied `token_source`.
2. The trusted custom broker configured by `GEMINI_MSRL_TOKEN_URL`.
3. A rotating access token in the file configured by `GEMINI_MSRL_ENV_FILE`.
4. Inline service-account JSON from `GEMINI_MSRL_GOOGLE_CREDENTIALS_JSON`.
5. Google Application Default Credentials (ADC).

The environment file must contain `GEMINI_MSRL_AUTH_TOKEN=...`. It is re-read
when its modification time changes, allowing an external token refresher to
rotate credentials without restarting the server. The broker takes precedence
when both `GEMINI_MSRL_TOKEN_URL` and `GEMINI_MSRL_ENV_FILE` are configured.

ADC produces short-lived OAuth access tokens with the Google Cloud Platform
scope. It supports local developer credentials, `GOOGLE_APPLICATION_CREDENTIALS`,
attached service accounts, and workload identity without treating the API URL as
an ID-token audience.

For local development, either run:

```bash
gcloud auth application-default login
```

or point ADC at a service-account JSON file:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/secure/path/service-account.json
```

For platforms such as Railway where mounting a credential file is inconvenient,
store the minified JSON as a secret variable instead:

```bash
export GEMINI_MSRL_GOOGLE_CREDENTIALS_JSON="$(jq -c . /secure/path/service-account.json)"
```

The JSON is parsed in memory and is not written to the container filesystem.
Treat this variable as a secret and never expose it in logs or deployment output.

An explicit source can also be injected:

```python
import os

from gemini_msrl import GeminiMsrlClient, GoogleAuthTokenSource

token_source = GoogleAuthTokenSource.from_service_account_file(
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
)
client = GeminiMsrlClient(token_source=token_source)
```

A service-account JSON file contains a long-lived private key. Store it in a
secret manager, mount it read-only with restrictive permissions, use a
least-privilege account, and rotate it regularly. Never commit it, bake it into
an image, or expose its contents or generated tokens in logs. Prefer an attached
service account or workload identity for production. The HTTP broker and token
environment file must likewise be secured with constrained access.

## Status & Roadmap

`ollie-rl` is **pre-1.0 / experimental**. The HTTP surface is intentionally
small and is still evolving.

Planned:

- [ ] A `tinker` trainer backend.
- [ ] An auto-research prompt-optimization backend — a `Trainer` that, instead
      of updating weights, evolves the system prompt from rewarded rollouts
      (think GEPA / OPRO / DSPy-style optimizers) behind the same HTTP surface.
- [ ] A runnable end-to-end `examples/` directory with reward curves.
- [ ] Documentation website.
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
