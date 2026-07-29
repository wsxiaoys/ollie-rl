# Ollie Harbor code-contest tasks

These fixtures adapt `code_contests-0000` from the generated tasks under
`examples/code-contests`. Both ask the agent to solve the same
balanced-parentheses problem and run five groups of input/output tests in a
separate Daytona verifier.

- `code-contest` requests `/workspace/solution.py` and verifies it with Python.
- `code-contest-typescript` requests `/workspace/solution.ts`, asks the agent to
  test it with Bun, and verifies it with Bun.

Ollie 0.2.2 is invoked exclusively through `npx`; there is no global
installation or root-level host setup.

Prerequisites:

- Node.js 22 or newer and npm/npx on the host
- `OPENAI_API_KEY` for the agent
- `DAYTONA_API_KEY` (or Daytona JWT credentials) for Ollie's remote executor
  and the verifier sandbox

From the repository root, enter `src/ollie/testdata` and run the Python task:

```bash
cd src/ollie/testdata
uv run harbor run \
  --path code-contest \
  --env ollie.harbor_environment:OllieEnvironment \
  --environment-kwarg verifier_environment=daytona \
  --agent ollie.harbor_agent:OllieAgent \
  --model openai/gpt-5.6 \
  --agent-env OPENAI_API_KEY="$OPENAI_API_KEY" \
  --verifier harbor.verifier.verifier:Verifier \
  --n-concurrent 1 \
  --yes
```

Run the TypeScript/Bun variant with:

```bash
cd src/ollie/testdata
uv run harbor run \
  --path code-contest-typescript \
  --env ollie.harbor_environment:OllieEnvironment \
  --environment-kwarg verifier_environment=daytona \
  --agent ollie.harbor_agent:OllieAgent \
  --model openai/gpt-5.6 \
  --agent-env OPENAI_API_KEY="$OPENAI_API_KEY" \
  --verifier harbor.verifier.verifier:Verifier \
  --n-concurrent 1 \
  --yes
```

`OllieEnvironment` launches the Ollie CLI from the Harbor trial's host-local
workspace. It is **not a security boundary**, so use it only with trusted agent
code. `OllieAgent` defaults to the Daytona executor, which makes remote
execution available when the agent tries the TypeScript solution with Bun.

Each fixture declares the complete `/workspace` directory as an artifact so
its separate verifier receives the submitted solution and all other workspace
changes. The Python verifier uses `python:3.12-slim`; the TypeScript verifier
uses `oven/bun:1.2`. Neither test harness installs packages or requires network
access.

The explicit standard `Verifier` import makes Harbor upload each task's
`tests` directory to `/tests` instead of assuming the verifier image already
contains the tests. Harbor transfers the workspace artifact before running the
test script. Reward `1` is awarded only when every expected output matches.
The agent's NDJSON stream and stderr are retained as `ollie.ndjson` and
`ollie.stderr` in the trial's agent logs.
