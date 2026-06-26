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
