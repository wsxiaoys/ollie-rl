# Word Puzzle — Ollie Agent + ollie-rl

This example trains the **Ollie agent** to infer composed string
transformations from a few input/output pairs. The driver launches the Ollie CLI
for each dispensed run, routes all of its model calls through that run's
OpenAI-compatible endpoint, grades its final response by exact match, and sends
the scalar reward back to `ollie-rl`.

## Prerequisites

- Start the server with `uv run poe dev`.
- Install Node.js 22 or newer; the driver resolves the pinned Ollie CLI through
  `npx`.

## Generate a dataset

From the repository root:

```bash
uv run python examples/word-puzzle/prepare_data.py \
  --output examples/word-puzzle/data/puzzles.jsonl \
  --num-examples 1000 \
  --num-transformations 2 \
  --num-few-shot-examples 3 \
  --operator-set extended \
  --seed 42
```

`--flip-reverse-task` is enabled by default, so `--num-examples 1000` writes
2,000 tasks: 1,000 decrypt-direction and 1,000 encrypt-direction puzzles. Add
`--no-flip-reverse-task` when the requested count should equal the number of
output rows. Run `prepare_data.py --help` for all generation controls.

Generated JSONL files are ignored by Git. Each row contains a user prompt and
its exact reference answer.

## Run training

```bash
uv run python examples/word-puzzle/run_training.py \
  --dataset examples/word-puzzle/data/puzzles.jsonl \
  --runs 2000 \
  --concurrency 8
```

Unless `--tuner-id` is supplied, the driver first looks for an existing tuner
whose name matches `--name`. It creates a new tuner only when no match is found
or the tuner list cannot be queried. Before creating a tuner, it deterministically
shuffles the dataset and holds out 5% of puzzles for evaluation (at least one
when the dataset has two or more puzzles).

For every assignment, the driver launches `@getollie/cli` in NDJSON mode with a
temporary workspace. Its `OPENAI_BASE_URL` is set to:

```text
http://<ollie-host>/tuners/<tuner-id>/runs/<run-id>/openai/v1
```

This attributes every Ollie model turn to the dispensed run. Each agent run is
limited to 100 model/tool steps. The default `--ollie-executor none` is
sufficient because word puzzles do not require tool execution; `local` and
`daytona` remain available for experimentation.

The generated prompts require the final answer in this form:

```text
$\text{answer}$
```

The grader recursively reads Ollie's NDJSON events, extracts the last such
expression, and compares it to the reference exactly. Missing or incorrect
answers receive zero reward.

### Useful flags

| Flag | Default | Meaning |
|---|---|---|
| `--base-url` | `http://localhost:8000` | ollie-rl API URL. |
| `--dataset` | `examples/word-puzzle/data/puzzles.jsonl` | Generated JSONL dataset. |
| `--recipe` | `grpo_4x8` | Tuning recipe used when creating a tuner. |
| `--trainer` | `fake` | Trainer factory used when creating a tuner. |
| `--name` | `tuning-word-puzzle-ollie` | Tuner name to reuse or create. |
| `--agent-model` | `ollie` | Model passed to the Ollie CLI. |
| Ollie package | `0.6.1` | Pinned `@getollie/cli` version resolved by `npx`. |
| `--ollie-executor` | `none` | Ollie executor: `none`, `local`, or `daytona`. |
| `--runs` | `2000` | Number of run/score iterations. |
| `--concurrency` | `8` | Maximum simultaneous Ollie agents. |
| `--tuner-id` | unset | Resume an existing tuner. |
