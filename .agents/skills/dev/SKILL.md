---
name: dev
description: Development guide for the Ollie RL api server
---

# References

- `references/data-model.md` — concepts of `Rollout` (the GRPO group) /
  `Run` / `Reward` / `ChatCompletion`, how they map to GRPO groups and
  batches, plus the end-to-end request lifecycle. Read this before
  touching `TunerService`, `RunModel`, `ChatCompletionModel`, or any
  rollout-collection / training-step code path.
- `references/sync-rl.md` — how a client interacts with the Ollie RL
  api server over its public HTTP surface for synchronous GRPO:
  bootstrapping a tuner via `POST /tuners`, iterating the dataset and
  fanning out agent runs (with bounded concurrency) via
  `POST /openai/v1/chat/completions` using `X-Tuner-Id` and `X-Run-Id`
  headers (the latter sent only on result-affecting completions), and
  posting per-run rewards via
  `PUT /tuners/{tuner_id}/runs/{run_id}/reward`. Training is applied
  implicitly by the server as rewards arrive. Read this when changing
  the HTTP surface in `server/app.py`, the rollout/reward contracts, or
  building a client SDK / cookbook recipe.
