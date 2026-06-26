---
name: dev
description: development on Ollie's RL api server
---

# References

- `references/data-model.md` — concepts of `Rollout` (the GRPO group) /
  `Reward` / `ChatCompletion`, how they map to GRPO groups and batches,
  plus the end-to-end request lifecycle. Read this before touching
  `TunerService`, `RewardModel`, `ChatCompletionModel`, or any
  rollout-collection / training-step code path.
- `references/sync-rl.md` — how a synchronous GRPO client drives the
  server: bootstrapping a tuner, fanning out rollouts via
  `/openai/v1/chat/completions` with `X-Tuner-Id` / `X-Run-Id` /
  `X-Datum-Id`, posting rewards, and triggering a train step. Read this
  when changing the HTTP surface in `server/app.py`, the rollout/reward
  contracts, or building a client SDK / cookbook recipe.
