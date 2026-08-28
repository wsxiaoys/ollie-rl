# Code Search — Ollie Agent + ollie-rl

This example trains the Ollie agent to localize files, classes, and functions
that need modification for an issue. It uses
`OpenHands/SWE-rebench-code-search` for problem statements and multi-level
localization labels.

## Prepare data

```bash
uv run python examples/code-search/prepare_data.py --limit 64 --seed 42
```

`--limit` selects repositories, not individual instances. The preparer shuffles
all repositories deterministically, selects the requested count, and prepares
every instance belonging to those repositories. Its resumable index avoids
fetching all Hugging Face rows again on later runs.

Generated data is ignored by Git and stored under `prepared/`:

```text
prepared/
  github.com/<org>/<repo>/git/
  github.com/<org>/<repo>/<base-commit>/
  index/
  instances/<instance-id>.json
  manifest.json
```

## Run training

Start ollie-rl, then run:

```bash
uv run python examples/code-search/run_training.py \
  --runs 2000 \
  --concurrency 8
```

The driver creates or resumes an ollie-rl tuner, dispenses run-addressed model
endpoints, and launches `@getollie/cli@0.7.1` for every assignment. Each base
commit snapshot is mounted at `/workspace` with `:ro`, so concurrent agents
cannot alter prepared data.

The prompt asks the agent to return repository-relative file, class, and
function locations in a structured `localization_result` element. The driver
computes separate set-F1 rewards for files, modules/classes, and entities, sums
them, and submits the resulting scalar reward to ollie-rl.

Useful options include `--recipe`, `--trainer`, `--agent-model`,
`--ollie-executor`, `--ollie-max-steps`, and `--tuner-id`. The default executor
is `none`; `daytona` and `vercel` are also accepted by Ollie CLI 0.7.1.
