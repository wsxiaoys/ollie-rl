---
name: bug-unleashed
description: Operate the BugUnleashed code-review RL example — prepare the dataset, smoke and run the Ollie agent driver against a local or MSRL tuner, score rollouts, calibrate the judge, and debug a stuck tuner.
---

# BugUnleashed example

`examples/bug-unleashed/` trains an Ollie agent to review PRs: one example per
gold bug from the BugUnleashed dataset (201 examples: 166 bug + 35 clean), a
finding matches when its **path** matches the gold location and an LLM judge
agrees it is the same defect. Read `examples/bug-unleashed/README.md` for the
full spec; this skill is the operating manual.

| File | Role |
| --- | --- |
| `prepare_data.py` | fetch cases/PRs/diffs, materialize `.git`-free snapshots, write `data/instances/*.json` and `data/splits.json` |
| `reward.py` | parse `<review_findings>`, path gate, judge, score; also the judge-calibration CLI |
| `run_training.py` | the driver: create/resume a tuner, dispense runs, launch the Ollie CLI, submit rewards |

`data/` is gitignored and large (~31 GB). Never commit it.

## 1. Prepare data

```bash
GITHUB_TOKEN=$(gh auth token) uv run python examples/bug-unleashed/prepare_data.py --concurrency 5
```

Resumable — rerun after a failure and it skips completed repos/PRs. Expect
`cases: 200 (165 bug, 35 clean)`, `examples: 201`, `repos: 65`, `failures: []`.
Splits are repo-disjoint: train 168 examples / val 33.

## 2. Smoke before spending money

The cheapest thing that exercises the whole loop is the MSRL recipe in the
README (2 datums × 2 rollouts). Always pass `--run-timeout 900` — a CLI that
stops calling the model but never exits otherwise holds a worker slot forever
and the group never completes.

```bash
KEY=$(python3 -c "import json;print(json.load(open('$HOME/.pi/agent/models.json'))['providers']['kumo-deepinfra']['apiKey'])")
uv run python examples/bug-unleashed/run_training.py \
  --base-url https://ollie-rl-production.up.railway.app \
  --trainer gemini_msrl --base-model gemini-3.5-flash \
  --recipe '{"group_size": 2, "num_groups_per_batch": 2,
             "max_off_policy_generation": 0, "sampler_promotion_every": 1,
             "eval_group_size": 1, "eval_every_n_checkpoints": 1,
             "reward_normalizer": "centering"}' \
  --runs 6 --concurrency 2 --ollie-max-steps 12 --run-timeout 900 \
  --judge-base-url https://kumo.up.railway.app/v1 --judge-api-key "$KEY" \
  --judge-model deepseek-4.1-flash
```

Timings on the deployed server: ~5 min of TPU warm-up (dispense answers 204
until the job is `RUNNING`), then ~7.5 min per `trainStep`. Resume with
`--tuner-id <id>` to skip the warm-up; because the eval tier outranks training
once a checkpoint exists, a resumed run spends its `--runs` on val datums.

## 3. Full run

The run size is a decision made when the skill is triggered — ask, don't assume.
The two recipe knobs are the shape of a GRPO step:

- `group_size` — rollouts per **datum** (one PR example). The advantage is taken
  against the mean of that group, so larger groups give a steadier baseline and
  cost proportionally more.
- `num_groups_per_batch` — datums per **train step**. Averages the groups, so
  larger batches give a smoother gradient and fewer steps per dollar.

One step therefore needs `group_size × num_groups_per_batch` rewarded runs, and
with `max_off_policy_generation: 0` that product is also the outstanding-run
budget. `4 × 8 = 32` rollouts per step is the sane default; `2 × 2 = 4` is the
smoke shape, whose 2-sample baseline is too noisy to learn from.

```bash
uv run python examples/bug-unleashed/run_training.py \
  --base-url https://ollie-rl-production.up.railway.app \
  --trainer gemini_msrl --base-model gemini-3.5-flash \
  --recipe '{"group_size": 4, "num_groups_per_batch": 8, "reward_normalizer": "centering"}' \
  --runs 2000 --concurrency 8 \
  --judge-base-url https://kumo.up.railway.app/v1 --judge-api-key "$KEY" \
  --judge-model deepseek-4.1-flash
```

Local server instead: `uv run uvicorn ollie_rl.server.app:app --port 8000` and
`--base-url http://localhost:8000 --trainer fake`.

## Rules that are easy to get wrong

- **`reward_normalizer` must be `centering`.** The default `grpo` divides by
  the group std and erases the `-5.0` clean-PR penalty.
- **The prompt's key list is what the model follows.** `severity` appeared in
  the prose but not in the "every finding needs …" sentence, so
  `gemini-3.5-flash` emitted exactly the four listed keys and every finding was
  dropped (`severity is not one of ('major','minor'): None`) — a whole smoke of
  misses. Any new field must be added to that sentence.
- **`parse_findings` scans line by line.** The `<review_findings>` element must
  sit inside one physical line (one NDJSON event); a pretty-printed element
  spread over several lines is invisible.
- **Only `gemini-3.5-flash` backs the multi-step API.** `gemini-3.1-flash-lite`
  and `gemini-3.5-flash-lite` are rejected by Vertex itself
  (`Base model … is not supported for multi-step reinforcement tuning`) for
  every adapter size and model-id form. The docs' RLFT model list covers the
  dataset-based `reinforcementTuningSpec`, which does accept Flash-Lite — a
  different trainer, not this one.
- **`--base-model` is MSRL-only** and lands in `trainer_params.base_model`.
  `checkpoint_interval` is forced to `sampler_promotion_every` by the service.
- **Judge calls must not go through the tuner run endpoint** — they would
  pollute the policy context and usage. `reward.py` calls the gateway directly.
- **`PUT …/reward` 403** means the lease expired or the run is empty (the agent
  never called the model). An empty run is a driver bug, not a model failure.

## Debugging a tuner

```bash
curl -s $BASE/tuners/$TUNER | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["status"],d["policy_generation"],d["is_training"],d["trainer_state"].get("pending_train_op"))'
curl -s $BASE/tuners/$TUNER/runs | python3 -m json.tool | head -40   # per-run status: rewarded / expired / in_flight
```

- `is_training True` + a `pending_train_op` means a `trainStep` is in flight;
  `last_train_op_duration_seconds` and `policy_generation` tell you it landed.
- A datum needs exactly `group_size` rewarded runs before it joins a batch, and
  a step needs `num_groups_per_batch` datums. With
  `max_off_policy_generation: 0` the outstanding budget is
  `group_size × num_groups_per_batch`, so one dead rollout stalls the group
  until its 15-minute lease expires.
- **Vertex cancels an MSRL job idle for 3 hours** ("cancelled after being idle
  for 3 hours"). A tuner left between steps dies; resume creates a new job.
- Probe Vertex directly with the service account when the server's error is
  opaque: `railway variables -p 646e2805-30a3-49e9-bcc7-96701314b87c -e
  production -s ollie-rl --json` → `GEMINI_MSRL_GOOGLE_CREDENTIALS_JSON` →
  write to a 0600 temp file → `Credentials.from_service_account_info(...,
  scopes=["https://www.googleapis.com/auth/cloud-platform"])` → hit
  `v1beta1/projects/<p>/locations/us-central1/tuningJobs`. Delete the file
  afterwards.

## Scoring and calibration

```bash
# score a saved rollout (NDJSON, one event per line)
uv run python examples/bug-unleashed/reward.py --example-id <id> --output rollout.ndjson \
  --judge-base-url https://kumo.up.railway.app/v1 --judge-api-key "$KEY" \
  --judge-model deepseek-4.1-flash --verbose

# rebuild the calibration set, hand-label it, then measure
uv run python examples/bug-unleashed/reward.py --make-pairs data/calibration-pairs.jsonl
uv run python examples/bug-unleashed/reward.py --pairs data/calibration-pairs.jsonl --concurrency 8
```

Reward: bug `+1.0` major match / `+0.25` minor match / `-1.0` miss; clean `-5.0`
if any major, else `0`. Minors are never penalised. The judge is calibrated at
~0.94 on 44 hand-labelled adversarial pairs; `--make-pairs` rows score 1.000 by
construction (synthetic, tautological) — only the hand-labelled file means
anything. The gateway does not honour `seed` reliably, so `--judge-votes` buys
little.

Label a pair TRUE when the finding names the same file, mechanism and
consequence — even with a different trigger instance or a wrong stated cause.
FALSE for a different code path, a restatement of the PR, or a general risk.

## Deploy check

`docker-publish.yml` only builds on `main` and tags, so pushing a feature
branch never changes the Railway image. Compare the deployment's
`meta.imageDigest` (`railway deployment list -p … -e production -s ollie-rl
--json`) with the ghcr `:latest` manifest digest before blaming the server.
