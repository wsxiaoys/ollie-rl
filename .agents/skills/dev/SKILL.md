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
- `references/tinker-trainer.md` — planning doc for integrating
  [Tinker](https://tinker-docs.thinkingmachines.ai/) as a real
  `Trainer` backend (registered as `"tinker"`) with first-class
  async-RL support. Reframes async RL as the natural mode of the
  existing `Trainer` protocol: the `is_training()` short-circuit in
  `TunerService.dispense_run` has no correctness role and is
  **deleted unconditionally** in Phase 1, `Example` gains a
  `policy_generation` field plumbed from `ChatCompletionModel`, and
  each trainer owns its own off-policy mechanics (Tinker filters
  stale `Example`s client-side using
  `TinkerTrainerConfig.max_steps_off_policy`; `gemini_msrl` already
  does the equivalent server-side). `Recipe` is unchanged. Phased
  delivery: (1) async-RL as natural mode (no tinker code; immediate
  consumer is `gemini_msrl`), (2) tinker skeleton trainer, (3) real
  tinker `train_step` with client-side staleness filter. Read this
  before touching `trainer/` to add tinker, deleting the dispense
  gate in `TunerService`, or plumbing `policy_generation` through
  `Example`. Reference: the
  [`tinker_cookbook/recipes/harbor_rl`](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/harbor_rl)
  recipe and the `do_async_training` loop in
  [`tinker_cookbook/rl/train.py`](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/rl/train.py).
- `references/malformed-samples.md` — design doc for handling
  malformed model output (e.g. broken tool calls) during GRPO
  training. Replaces the current `raise NotImplementedError` sites
  in `trainer/tinker.py` and `trainer/gemini_msrl.py` with a
  server-side flow: trainer flags the `Sample` as `malformed=True`,
  `TunerService.sample` records the bad completion, sets
  `reward = recipe.malformed_penalty` via `update_reward`, and
  raises `MalformedSampleError` (→ HTTP **409 Conflict**). The
  agent client's rollout loop exits naturally (same handling it
  already has for "reward already set" races). The bad token
  sequence still lands in `Trainer.train_step` as an `Example`
  with the most-negative GRPO advantage in its group — the model
  learns "don't do that" from the very rollout that caused the
  problem. Adds `Sample.malformed: bool` and
  `Recipe.malformed_penalty: float`. Read this before touching the
  parse-failure branches in `trainer/tinker.py` /
  `trainer/gemini_msrl.py`, the `Recipe` schema, or
  `TunerService.sample`.
