# CodeContests — Harbor + ollie-rl RL Example

Reinforcement learning on containerized competitive-programming tasks using
[Harbor](https://www.harborframework.com/docs/training-workflows/rl) as the
rollout environment and `ollie-rl` as the tuner/trainer.

Each rollout is a full Harbor trial: a Terminus 2 agent solves the problem in
the selected container backend, and Harbor's verifier runs the task tests to
produce the reward. The Ollie-agent variant is maintained separately under
`examples/ollie-code-contests`.

```
examples/code-contests/
├── prepare_data.py
├── run_training.py
├── tasks/              # generated Harbor tasks
└── trials/             # generated Harbor trial output
```

## Prepare tasks

Every row of `open-thoughts/CodeContests` contains a gzipped tarball of a
complete Harbor task directory. Extract a local subset from the repository root:

```bash
uv run python examples/code-contests/prepare_data.py --limit 64
```

Each extracted task path is used verbatim as its ollie-rl `datum_id`.

## Run training

Start the ollie-rl server with `uv run poe dev`, then run:

```bash
uv run python examples/code-contests/run_training.py --runs 200 --concurrency 8
```

Each Terminus 2 trial samples through a run-specific endpoint:

```
http://<ollie-host>/tuners/<tuner-id>/runs/<run-id>/openai/v1
```

The tuner and run IDs in the request path allow ollie-rl to attribute every
completion before Harbor submits the verifier reward.

### Useful flags

| Flag | Default | Meaning |
|---|---|---|
| `--base-url` | `http://localhost:8000` | ollie-rl API URL. |
| `--recipe` | `grpo_16x32` | Tuning recipe. |
| `--trainer` | `fake` | Trainer factory. |
| `--name` | `tuning-code-contests` | Tuner name. |
| `--environment` | `docker` | Harbor environment backend. |
| `--runs` | `200` | Number of run/score iterations. |
| `--concurrency` | `8` | Parallel Harbor trials. |
| `--tuner-id` | unset | Resume an existing tuner. |

## Notes

- For local Docker, use a host-reachable `--base-url` rather than `localhost`
  when the agent container must connect back to the ollie-rl server.
- The driver targets `harbor==0.16.1`.
- Submit rewards before the run lease expires; expired leases return `409`.
