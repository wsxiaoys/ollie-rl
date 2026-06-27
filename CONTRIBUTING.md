# Contributing to ollie-rl

Thanks for your interest in `ollie-rl`! This project is in its early days, and
small contributions — bug reports, doc fixes, recipes, new trainer adapters —
are all extremely welcome.

By participating, you agree to abide by our
[Code of Conduct](./CODE_OF_CONDUCT.md).

## Quick reference

| | |
|---|---|
| **Bug report** | [Open an issue](https://github.com/wsxiaoys/ollie-rl/issues/new?template=bug_report.yml) |
| **Feature idea** | [Open an issue](https://github.com/wsxiaoys/ollie-rl/issues/new?template=feature_request.yml) |
| **Security report** | See [SECURITY.md](./SECURITY.md) — do **not** open a public issue |
| **Questions / show-and-tell** | [GitHub Discussions](https://github.com/wsxiaoys/ollie-rl/discussions) (once enabled) |

## Development setup

`ollie-rl` is a Python project managed with [`uv`](https://docs.astral.sh/uv/).

### Prerequisites

- **Python 3.14+** (the project pins a very recent Python; see
  [`.python-version`](./.python-version)). `uv` will install it for you.
- A recent `uv` (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
- Optional: Docker + Docker Compose for the integration smoke test.

### Clone and install

```bash
git clone https://github.com/wsxiaoys/ollie-rl
cd ollie-rl
uv sync --all-groups
pre-commit install --hook-type pre-commit --hook-type pre-push
```

### Common tasks

All workflows are exposed via [`poethepoet`](https://poethepoet.natn.io/) tasks
in `pyproject.toml`:

```bash
uv run poe dev             # run the FastAPI server with auto-reload
uv run poe test            # run the pytest suite
uv run poe check           # run the `ty` type checker
uv run poe lint            # run ruff lint (no autofix)
uv run poe format          # run ruff autofix + uv format
uv run poe docker-build    # build the Docker image locally
uv run poe docker-up       # docker compose up (production-style)
```

Before pushing, the pre-push hook will re-run `ty check`, `pytest`, and the
formatter automatically. If you want to run them manually:

```bash
uv run ty check
uv run pytest
uv run poe format
```

## How the codebase is organized

See the [README "Architecture" section](./README.md#architecture) for the
high-level layout. The two most important internal docs are:

- [`.agents/skills/dev/references/data-model.md`](./.agents/skills/dev/references/data-model.md)
  — the canonical explanation of `Rollout`, `Run`, `ChatCompletion`, batch
  formation, and advantage math. **Read this before touching `TunerService`.**
- [`.agents/skills/dev/references/sync-rl.md`](./.agents/skills/dev/references/sync-rl.md)
  — the HTTP-surface contract that any synchronous-RL client must respect.
  **Read this before changing `server/app.py` or building a client.**

## Adding things

### A new trainer backend

The `Trainer` / `TrainerFactory` contract lives in
[`src/ollie_rl/trainer/types.py`](./src/ollie_rl/trainer/types.py). The
shortest path to a new backend is:

1. Implement `Trainer` (`sample`, `train_step`, `in_flight_train_op`).
2. Implement `TrainerFactory.open(...)` to bootstrap or restore your trainer
   from a `StateStore`.
3. Register it via the factory registry in
   [`src/ollie_rl/trainer/factory.py`](./src/ollie_rl/trainer/factory.py).
4. Add a test that exercises both `sample` and `train_step` against a fake.

See `gemini_msrl.py` for a worked example.

### A new recipe

Recipes are declarative knobs the `TunerService` consults to form batches.
Add a frozen `Recipe(...)` constant in
[`src/ollie_rl/cookbook/recipes.py`](./src/ollie_rl/cookbook/recipes.py)
and reference it by name from `POST /tuners`.

## Commit style

We follow [Conventional Commits](https://www.conventionalcommits.org/) so the
changelog and (eventually) release-please can be generated automatically:

```
feat(trainer): add tinker backend
fix(server): return 409 on stale run_id
docs: fix typo in data-model.md
chore(ci): bump setup-uv to v4
```

Prefixes we use: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`,
`build`, `ci`.

## Pull request checklist

- [ ] Tests pass locally (`uv run pytest`).
- [ ] Type check passes (`uv run ty check`).
- [ ] Code is formatted (`uv run poe format`).
- [ ] Public-API or HTTP-surface changes are reflected in `README.md` and
      `.agents/skills/dev/references/*.md`.
- [ ] Commit messages follow the Conventional Commits style above.
- [ ] If the change is user-visible, a one-line entry is added to `CHANGELOG.md`
      (once it exists) under `[Unreleased]`.

Thanks again — happy hacking! 🛹
