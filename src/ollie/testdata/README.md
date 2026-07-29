# Ollie Harbor smoke task

This fixture runs the Ollie agent on the host and verifies its artifact in a
separate Daytona sandbox. Ollie 0.2.2 is invoked exclusively through `npx`;
there is no global installation or root-level host setup.

Prerequisites:

- Node.js 22 or newer and npm/npx on the host
- `OPENAI_API_KEY` for the agent
- `DAYTONA_API_KEY` (or Daytona JWT credentials) for the verifier sandbox

From the repository root, enter `src/ollie/testdata` and run:

```bash
cd src/ollie/testdata
uv run harbor run \
  --path smoke \
  --env ollie.harbor_environment:OllieEnvironment \
  --environment-kwarg verifier_environment=daytona \
  --agent ollie.harbor_agent:OllieAgent \
  --model openai/gpt-5.6 \
  --agent-env OPENAI_API_KEY="$OPENAI_API_KEY" \
  --verifier harbor.verifier.verifier:Verifier \
  --n-concurrent 1 \
  --yes
```

`OllieEnvironment` keeps the agent workspace under the Harbor trial directory
and runs agent commands as the current host user. It is **not a security
boundary**, so use it only with trusted agent code. For workspace access,
Ollie receives only `--cwd .`; its `/workspace` writes persist directly into
Harbor's host workspace. The fixture declares the complete `/workspace`
directory as an artifact so the separate verifier receives all agent changes.

When Harbor creates the separate verifier requested by `task.toml`, the hybrid
adapter delegates that environment to a prebuilt Ubuntu Daytona sandbox. The
explicit standard `Verifier` import makes Harbor upload `smoke/tests` to
`/tests` instead of assuming the verifier image already contains the tests.
Harbor also transfers collected artifacts before the tests run.

A successful trial receives reward `1` only when `answer.txt` has the exact
expected bytes. The agent's NDJSON stream and stderr are retained as
`ollie.ndjson` and `ollie.stderr` in the trial's agent logs.
