# Tinker-Cookbook Parity Tracking

This doc summarizes how `src/ollie_rl/trainer/tinker.py` (and the surrounding
ollie-rl service layer) compares against the reference implementation in
[`tinker-cookbook/recipes/harbor_rl`](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/harbor_rl)
(and the supporting `tinker_cookbook/rl/train.py` + `rl/data_processing.py`).

The two projects sit at **different layers of abstraction**, so the goal here
is not to mirror harbor_rl 1:1 — it is to track where ollie-rl's `tinker`
backend deliberately diverges, and where it is genuinely behind.

## Layering

| Layer | tinker-cookbook (`harbor_rl`) | ollie-rl |
|---|---|---|
| Owns the training loop | `tinker_cookbook.rl.train.main()` | External orchestrator (`TunerService.maybe_train`) drives `Trainer.train_step`. |
| Owns env / reward / tools / sandbox | Yes (`HarborEnvGroupBuilder`, `HarborBashTool`, `HarborReward`, Modal sandbox). | No — agent runs out-of-process, hits `/v1/chat/completions`, submits scalar reward via `PUT /reward`. |
| Multi-turn rollout representation | One `Trajectory` with N `Transition`s (one per turn). | One `Run` with N `ChatCompletion` rows; grouped into trajectory-level Datums by the trainer. |
| Config object | `chz.chz` dataclasses (`CLIConfig`, `Config`, `AsyncConfig`). | `pydantic.BaseModel` (`TinkerTrainerConfig`, `TinkerTrainerState`) — must JSON-serialize into `StateStore`. |
| State persistence | `CheckpointManager` writing to `log_path`, rolling vs periodic checkpoints, TTL. | Single JSON blob in `StateStore`; `optimizer_path` field declared but currently unused. |

## Advantage / group math

| Aspect | tinker-cookbook | ollie-rl |
|---|---|---|
| Where computed | `rl/data_processing.compute_advantages` — runs inside the trainer pipeline. | `TunerService._collect_consumable_batch` — runs in the service layer before `train_step`. |
| Per-trajectory return | `sum(transition.reward) + final_reward`. Per-step rewards in `Env.step`, optional group-level term from `EnvGroupBuilder.compute_group_rewards`. | Single scalar reward submitted via `PUT /reward` on the `Run`. |
| Group centering | `advantage_i = total_return_i - mean(group_returns)`. **No std normalization.** | `advantage_i = (reward_i - mean) / (std + eps)` with degenerate-std fallback to `0`. |
| Token-level layout | Broadcasts the trajectory scalar to every action token across **all turns** of the trajectory; observation tokens get `0` + `mask=0`. | We reconstruct multi-turn trajectories across the batch using `examples_to_data` (in `ollie_rl.trainer.tinker.accumulator`), laying out `advantage=0`/`mask=0` on prompt/observation tokens and `advantage=run_advantage`/`mask=1` on completion tokens. This is byte-identical to the cookbook's token-level layout. |
| Temporal credit assignment | None — flat scalar across the whole trajectory. | None — flat scalar across the whole run. |

**Net:** correctness-equivalent up to the std normalization choice. We have closed the compute-efficiency gap by implementing prefix-graph partitioning in `ollie_rl.trainer.tinker.accumulator` which merges prefix-extending turns into one `Datum` so the shared prefix is computed only once.

## `train_step` body

| Behaviour | cookbook (`rl.train.train_step`) | ollie-rl (`TinkerTrainer.train_step`) |
|---|---|---|
| Substep pipelining | Splits `data_D` into `num_substeps` batches; pipelines `forward_backward_async` + `optim_step_async` so they share clock cycles. | Single `forward_backward_async` + single `optim_step_async` per call. |
| `mask` field | Stripped with `_remove_mask` before `forward_backward` — mask is metadata for downstream KL. | Kept in `loss_fn_inputs`. |
| Adam params | `AdamParams(learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)`. | `AdamParams(learning_rate=lr)` — relies on tinker defaults. |
| Training logprobs | Returned by `_training_logprobs_from_fwd_bwd`; fed into KL metrics. | Discarded. |
| Optim metrics | Merged into the metrics dict. | Discarded. |

## Staleness handling

- **Cookbook (`AsyncConfig`):** stale trajectories are dropped, and the
  rollout pipeline keeps generating until `groups_per_batch` survive. Lag is
  bounded but training never stalls on a stale batch.
- **ollie-rl:** client-side `_filter_stale`; if dropped fraction exceeds
  `max_stale_fraction` (default 0.4), raises `StaleBatchError` and **refuses**
  the batch. No requeue/top-up loop. `max_stale_fraction` has no analog
  upstream.

## KL penalty / reference policy

| Feature | cookbook | ollie-rl |
|---|---|---|
| Reference policy client | Configurable via `KLReferenceConfig`. | None. |
| Sample/train KL gap | `compute_kl_sample_train`. | None. |
| Post-update KL | `compute_post_kl` opt-in. | None. |
| KL coefficient | `kl_penalty_coef`, plus `kl_discount_factor`. | `kl_penalty_coef` forwarded into `loss_fn_config`; no discount. |

## Sampler promotion / checkpoints

- **Cookbook:** `save_checkpoint_and_get_sampling_client` per iteration, with
  rolling and periodic cadences (`rolling_save_every`, `save_every`) and TTLs
  (`ttl_seconds`, `rolling_ttl_seconds`).
- **ollie-rl:** `_promote_sampler` controlled by `sampler_promotion_every`
  (default every step). No rolling vs periodic distinction, no TTL,
  `optimizer_path` never written.

## Sampling path

- **Cookbook:** `do_group_rollout` requests multiple samples per call, feeds
  them into multi-turn env loops, produces `Trajectory`/`TrajectoryGroup`.
- **ollie-rl:** hardcoded `num_samples=1`, single-turn per call:
  - Parses response with `renderer.parse_response`.
  - Wraps result as OpenAI `ChatCompletion` with synthesized tool-call IDs.
  - Raises `NotImplementedError` for malformed assistant responses unless
    `stop_reason == "length"`.
  - Tokenizer from `sampling_client.get_tokenizer()` (vs
    `tokenizer_utils.get_tokenizer(model_name)` upstream) and falls back to
    the `"role_colon"` renderer if `get_recommended_renderer_name` fails.

## Observability / orchestration features missing on the ollie-rl side

Cookbook has, ollie-rl does not:

- `eval_every` + parallel `SamplingClientEvaluator` runs.
- `logtree` HTML/JSON dumps and `RolloutSummaryGroup` JSONL exports.
- `wandb` integration (`wandb_project`, `wandb_name`).
- `trace` spans / Gantt charts (`enable_trace`, `span_chart_every`).
- `tqdm` progress bars.
- Per-iteration `print_group` debug dump.

ollie-rl uses only stdlib `logging` today.

## Config knobs with no analog on our side

From the cookbook `Config` (not exhaustive):

`num_substeps`, `compute_post_kl`, `remove_constant_reward_groups`,
`rollout_error_tolerance`, `stream_minibatch_config`, `kl_discount_factor`,
`evaluator_builders`, `recipe_name` (cookbook-side concept),
`load_checkpoint_path`, `renderer_name`, `rolling_save_every`, `ttl_seconds`,
`num_groups_to_log`, `rollout_json_export`, `max_steps`.

## Smaller deltas worth noting

- Sampler checkpoint names: we tag with `uuid.uuid4().hex[:8]`; cookbook uses
  the iteration index.
- `loss_fn` typing: we hold it as `str` and cast at the call boundary;
  cookbook types it as `LossFnType` (`Literal`) end-to-end.
- State persistence cadence: we persist on every successful `train_step`;
  cookbook persists at `save_every`.

## TL;DR

Our `TinkerTrainer` is a deliberately small slice of what cookbook does in
`rl/train.py` + `rl/data_processing.py`:

- **Same idea, different packaging** for multi-turn: cookbook bakes it into
  the per-token mask/advantage layout; we shard it into one `Example` per
  chat completion in the service layer.
- **Same idea, slightly different math** for advantages: cookbook centers
  only; we center *and* divide by std with an eps fallback.
- **Genuine gaps:** substep pipelining, KL reference / KL metrics, eval
  pipeline, checkpoint TTL and rolling cadence, observability hooks
  (wandb/logtree/trace), and the `AsyncConfig` requeue-on-stale behaviour
  (we currently refuse the batch instead).

Order in which it is likely worth closing the gap, increasing in scope:

1. Strip `mask` before `forward_backward`; set proper Adam params
   (`beta1=0.9, beta2=0.95, eps=1e-8`).
2. Plumb training logprobs and optimizer metrics back out of `train_step` so
   the service can log/expose them.
3. `num_substeps` pipelined `forward_backward`/`optim_step`.
4. KL reference client + post-KL metric.
5. Rolling vs periodic checkpoints with TTL.
6. Replace `StaleBatchError`-on-overflow with a requeue/top-up loop matching
   `AsyncConfig.groups_per_batch` semantics.
7. **[Completed]** Optional `trajectory_to_data`-style prefix-merge for multi-turn runs to
   avoid redundant forward/backward over shared prefixes (pure compute win,
   no semantic change). Implemented via prefix-graph partitioning in
   `ollie_rl.trainer.tinker.accumulator.examples_to_data`.

## `trajectory_to_data` in ollie-rl

**Status:** Completed. Implemented via prefix-graph partitioning in `ollie_rl.trainer.tinker.accumulator` and fully unit-tested.

Cookbook's `trajectory_to_data` walks a `Trajectory.transitions` list — an
ordered sequence of `(obs, action)` pairs that the rollout pipeline knows are
all part of one episode. ollie-rl does not have that information at the
trainer boundary: by the time examples reach `TinkerTrainer.train_step`, the
trajectory structure has been flattened into per-`ChatCompletion` rows.

### Why a `run_id` is *not* a single trajectory

From `data-model.md`:

> A **run** is the unit of reward / advantage. A single run may internally
> contain multiple trajectories (e.g. multi-step or agent-with-sub-agent
> setups); they all share the same `run_id`, reward, and advantage.

A `run_id` is therefore a **bag of trajectories that share an advantage**,
not a single trajectory. Concretely:

- **Sub-agent.** The main agent's thread and the sub-agent's thread don't
  share a system prompt — they are two disjoint token-prefix trees.
- **Multi-agent / committee.** N agents collaborating, each with its own
  thread.
- **Pure parallel attempts.** Multiple independent chat threads inside one
  run.

So naïvely sorting all completions by creation time and treating them as one
linear chain would produce semantic nonsense merges (e.g. attaching sub-agent
tokens onto the end of the main agent's trajectory).

### Reconstructing Trajectories from Token Sequences

We discover trajectory boundaries directly from the token sequences themselves:

```
Run (shared advantage)
├── Trajectory A   (main agent)        ┐
│   ├── ChatCompletion a1               ├─ same prefix tree
│   ├── ChatCompletion a2               │
│   └── ChatCompletion a3               ┘
└── Trajectory B   (sub-agent)         ┐
    ├── ChatCompletion b1               ├─ different prefix tree
    └── ChatCompletion b2               ┘
```

Two completions belong to the same trajectory iff one's prompt is a
**token-prefix extension** of the other's full token sequence. That is a
prefix-graph partitioning problem, not a simple sort.

### Implemented Partitioning Algorithm

To reconstruct multi-turn trajectories from a flat batch of examples:

1. **Sort by prompt length ascending:** Sort completions by `len(prompt_tokens)`
   ascending to make the partitioning greedy.
2. **Maintain a set of open trajectory accumulators:** Each accumulator holds
   `(full_seq, sampled_lp, advantages, mask)`.
3. **Greedy matching:** For each completion with prompt `ob` and completion `ac`:
   - Find the open accumulator whose `full_seq` is the **longest** prefix
     of `ob` (via a linear scan).
   - If one exists: attach.
     `delta_ob = ob[len(full_seq):]` → append with zeros for
     `lp`/`advantage`/`mask`; then append `ac` with `lp = ex.logprobs`,
     `advantage = [ex.advantage] * len(ac)`, `mask = [1.0] * len(ac)`.
   - If none exists: open a fresh accumulator seeded with `ob + ac` (zeros
     on `ob`, real values on `ac`).
4. **Flush to Datums:** Convert each open accumulator into its own `tinker.Datum`
   with the standard right-shift / left-shift slicing.

This correctly handles:

| Case | Outcome |
|---|---|
| Linear multi-turn (single agent) | One accumulator grows; **1 Datum** per trajectory. |
| Sub-agent | Main + sub each build their own accumulator; **2 Datums**. |
| Pure parallel attempts (N threads) | **N Datums**. |
| Renderer re-encodes history non-monotonically | Mismatch on every turn → degenerates to **1 Datum per completion** — safe fallback. |

### Implications for the Contract

- **No `run_id` required on `Example`:** Because prefix matching is based purely on token sequences, the unique paths of the conversations naturally partition different runs and trajectories without any extra metadata. The pre-computed `advantage` is carried directly on each `Example`.
- **Service stays trajectory-agnostic:** `_collect_consumable_batch` just emits standard `Example(advantage, tokens, logprobs)` per chat completion. Trajectory detection lives entirely in `TinkerTrainer._examples_to_data`.

### Equivalence to Cookbook

In cookbook, trajectory boundaries are *given* by the rollout pipeline.
ollie-rl's HTTP design trades that structural information for being
client-agnostic, so we **recover it from tokens** before applying the same
merge math. Once recovered, the per-token `(advantage, mask, logprobs)`
layout is byte-identical to cookbook's, and the resulting Datums are
drop-in compatible with the cookbook KL / `_remove_mask` flow we'll port
later.
