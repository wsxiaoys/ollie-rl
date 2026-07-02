"""End-to-end RL training driver: CodeContests (Harbor tasks) + ollie-rl.

Each rollout runs as a containerized **Harbor trial**: a Terminus 2 agent is
dropped into a sandbox to solve one competitive-programming task, and Harbor's
verifier runs the task's unit tests to produce the reward. The driver just
orchestrates the loop — dispense a run from ollie-rl, run the trial, report the
reward — while ollie-rl handles GRPO grouping and training.

How Harbor and ollie-rl concepts line up:

    Harbor                         ollie-rl
    ------------------------------ ---------------------------------
    TaskConfig(path=...)           datum_id  (one CodeContests task)
    one trial -> (reward)          Run       (one attempt at a datum_id)
    K trials of the same task      Rollout   (a GRPO group)
    verifier_result.rewards        PUT /reward payload
    agent base_url (LLM endpoint)  ollie-rl's OpenAI-compatible proxy

Token collection is handled entirely by ollie-rl: Terminus 2 samples through the
shared endpoint below, tagged with this run's ``X-Run-Id`` header, so every
result-affecting completion is recorded under the dispensed ``run_id``.

Header-addressed endpoint
-------------------------
This driver uses ollie-rl's standard OpenAI-compatible endpoint and carries the
tuner_id / run_id in **HTTP headers**::

    POST /openai/v1/chat/completions
    X-Tuner-Id: {tuner_id}
    X-Run-Id:   {run_id}

so every agent shares one static ``base_url``::

    http://<ollie-host>/openai/v1

(An OpenAI-compatible client appends ``/chat/completions`` to that base_url.)
The per-run attribution comes from the ``X-Run-Id`` header, injected below via
Terminus 2's ``extra_headers`` LLM kwarg.

Prerequisites
-------------
* ``ollie-rl`` server running (``uv run poe dev`` from the repo root).
* ``harbor`` installed in the environment (``uv add harbor`` or ``pip install
  harbor``) with a working container backend (local ``docker`` by default).
* Tasks extracted locally first::

      uv run python examples/code-contests/prepare_data.py --limit 64

Run it (from the repo root)::

    uv run python examples/code-contests/run_training.py --runs 200
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import httpx

# --- Harbor imports (verified against harbor==0.16.1) ---------------------
from harbor.trial.trial import Trial
from harbor.models.environment_type import EnvironmentType
from harbor.models.trial.config import (
    TrialConfig,
    TaskConfig,
    AgentConfig,
    EnvironmentConfig,
)
from harbor.models.agent.name import AgentName

EXAMPLE_DIR = Path(__file__).resolve().parent
TASKS_DIR = EXAMPLE_DIR / "tasks"
TRIALS_DIR = EXAMPLE_DIR / "trials"

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_RECIPE = "grpo_16x32"
DEFAULT_TRAINER = "fake"
DEFAULT_TUNER_NAME = "tuning-code-contests"

# litellm (used internally by Terminus 2) needs a provider prefix to know how to
# route the request. ollie-rl exposes an OpenAI-compatible proxy, so the model is
# addressed as ``openai/<name>`` which tells litellm to treat the per-run
# ``base_url`` as an OpenAI-compatible server (otherwise it raises "LLM Provider
# NOT provided").
AGENT_MODEL_NAME = "openai/ollie"


# --------------------------------------------------------------------------
# ollie-rl HTTP helpers
# --------------------------------------------------------------------------
def discover_datum_ids() -> list[str]:
    """Every extracted task directory under ``tasks/`` is one datum_id."""
    if not TASKS_DIR.exists():
        raise SystemExit(
            f"No tasks found in {TASKS_DIR}. Run prepare_data.py first:\n"
            f"  uv run python {EXAMPLE_DIR / 'prepare_data.py'} --limit 64"
        )
    return sorted(
        p.name for p in TASKS_DIR.iterdir() if p.is_dir() and (p / "task.toml").exists()
    )


async def create_tuner(
    client: httpx.AsyncClient,
    *,
    name: str,
    recipe: str,
    trainer: str,
    datum_ids: list[str],
) -> str:
    resp = await client.post(
        "/tuners",
        json={
            "name": name,
            "recipe": recipe,
            "trainer": trainer,
            "datum_ids": datum_ids,
        },
    )
    resp.raise_for_status()
    tuner_id = resp.json()["tuner_id"]
    print(f"[driver] created tuner {tuner_id} ({len(datum_ids)} tasks)")
    return tuner_id


async def dispense_run(
    client: httpx.AsyncClient, tuner_id: str
) -> tuple[str, str] | None:
    """Return ``(run_id, datum_id)`` or ``None`` when the trainer is busy (204)."""
    resp = await client.post(f"/tuners/{tuner_id}/runs")
    if resp.status_code == 204:
        return None
    resp.raise_for_status()
    body = resp.json()
    return body["run_id"], body["datum_id"]


async def submit_reward(
    client: httpx.AsyncClient, tuner_id: str, run_id: str, reward: float
) -> None:
    resp = await client.put(
        f"/tuners/{tuner_id}/runs/{run_id}/reward", json={"reward": reward}
    )
    resp.raise_for_status()


# --------------------------------------------------------------------------
# The rollout: one Harbor trial == one ollie-rl Run
# --------------------------------------------------------------------------
def openai_base_url(base: str) -> str:
    """The shared OpenAI-compatible endpoint (ids travel in headers, not URL)."""
    return f"{base}/openai/v1"


async def run_rollout(
    *,
    base: str,
    tuner_id: str,
    run_id: str,
    datum_id: str,
    environment: str,
) -> float:
    """Execute one containerized Harbor trial and return its reward.

    A Harbor trial maps 1:1 onto one ollie-rl Run: a single agent attempt at a
    single task under a single ``run_id``. Terminus 2 samples through the shared
    ``/openai/v1`` endpoint, tagging every request with this run's ``X-Run-Id``
    header, so ollie-rl records every completion under ``run_id``. Harbor's
    verifier then runs the task's ``tests/test.sh`` and writes a reward file,
    surfaced here as a float.
    """
    config = TrialConfig(
        task=TaskConfig(path=TASKS_DIR / datum_id),
        trials_dir=TRIALS_DIR,
        agent=AgentConfig(
            name=AgentName.TERMINUS_2.value,
            model_name=AGENT_MODEL_NAME,
            # Terminus 2 forwards these kwargs to its LiteLLM backend:
            #   api_base    -> the shared OpenAI-compatible endpoint
            #   api_key     -> required by litellm's openai/ provider
            #   extra_headers -> X-Tuner-Id / X-Run-Id, attached to each
            #                    chat-completion request so ollie-rl attributes
            #                    every completion to this run
            kwargs={
                "api_base": openai_base_url(base),
                "llm_kwargs": {
                    "api_key": "ollie",
                    "extra_headers": {
                        "X-Tuner-Id": tuner_id,
                        "X-Run-Id": run_id,
                    },
                },
            },
        ),
        environment=EnvironmentConfig(type=EnvironmentType[environment.upper()]),
    )

    # `Trial` is abstract; `Trial.create()` loads the task and returns the right
    # concrete trial (single- vs multi-step). `run()` returns a `TrialResult`.
    trial = await Trial.create(config)
    trial_result = await trial.run()
    return extract_reward(trial_result)


def extract_reward(trial_result) -> float:
    """Pull the scalar reward out of a single Harbor trial result.

    ``VerifierResult.rewards`` is a ``dict[str, float | int] | None``. Harbor
    reads ``/logs/verifier/reward.json`` (multi-metric) and falls back to
    ``reward.txt`` (a single value). We prefer the ``"reward"`` key, then fall
    back to the sole value when the verifier emitted exactly one metric.
    """
    verifier_result = getattr(trial_result, "verifier_result", None)
    rewards = getattr(verifier_result, "rewards", None) if verifier_result else None
    if not rewards:
        return 0.0
    if "reward" in rewards:
        return float(rewards["reward"])
    if len(rewards) == 1:
        return float(next(iter(rewards.values())))
    return 0.0


# --------------------------------------------------------------------------
# Driver loop
# --------------------------------------------------------------------------
async def worker(
    name: int,
    client: httpx.AsyncClient,
    tuner_id: str,
    budget: asyncio.Queue[int],
    args: argparse.Namespace,
    stats: dict,
) -> None:
    while True:
        try:
            run = budget.get_nowait()
        except asyncio.QueueEmpty:
            return

        # Phase 1: dispense a run assignment (204 => training barrier, back off).
        assignment: tuple[str, str] | None = None
        while assignment is None:
            assignment = await dispense_run(client, tuner_id)
            if assignment is None:
                await asyncio.sleep(1.0)
        run_id, datum_id = assignment

        # Phase 2: execute the containerized Harbor trial.
        try:
            reward = await run_rollout(
                base=args.base_url,
                tuner_id=tuner_id,
                run_id=run_id,
                datum_id=datum_id,
                environment=args.environment,
            )
        except Exception as exc:  # a crashed trial scores the failure reward.
            print(f"[driver] run {run:04d} trial error ({datum_id}): {exc}")
            reward = 0.0

        # Phase 3: report the reward; the server groups/advantages/trains.
        await submit_reward(client, tuner_id, run_id, reward)

        stats["rewards"].append(reward)
        window = stats["rewards"][-32:]
        avg = sum(window) / len(window)
        print(
            f"[driver] run {run:04d} task={datum_id:<20} "
            f"reward={reward:+.1f} avg32={avg:.3f}"
        )


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--recipe", default=DEFAULT_RECIPE)
    parser.add_argument("--trainer", default=DEFAULT_TRAINER)
    parser.add_argument("--name", default=DEFAULT_TUNER_NAME)
    parser.add_argument(
        "--environment",
        default="docker",
        help="Harbor EnvironmentType (docker, daytona, modal, ...).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=200,
        help="How many run/score iterations to perform.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="How many Harbor trials to run in parallel.",
    )
    parser.add_argument(
        "--tuner-id",
        default=None,
        help="Reuse an existing tuner instead of creating a new one.",
    )
    args = parser.parse_args()

    TRIALS_DIR.mkdir(parents=True, exist_ok=True)
    datum_ids = discover_datum_ids()

    async with httpx.AsyncClient(base_url=args.base_url, timeout=30.0) as client:
        tuner_id = args.tuner_id or await create_tuner(
            client,
            name=args.name,
            recipe=args.recipe,
            trainer=args.trainer,
            datum_ids=datum_ids,
        )

        budget: asyncio.Queue[int] = asyncio.Queue()
        for run in range(args.runs):
            budget.put_nowait(run)

        stats: dict = {"rewards": []}
        await asyncio.gather(
            *(
                worker(i, client, tuner_id, budget, args, stats)
                for i in range(max(1, args.concurrency))
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
