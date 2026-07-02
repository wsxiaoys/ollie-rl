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
per-run ``base_url`` below, so every result-affecting completion is recorded
under the dispensed ``run_id`` — no ``token_ids`` / ``mask_ids`` plumbing needed.

Twin (base_url-addressed) endpoint
----------------------------------
This driver uses ollie-rl's *twin* of ``/openai/v1/chat/completions`` that
carries the tuner_id / run_id **in the URL path** instead of headers::

    POST /tuners/{tuner_id}/runs/{run_id}/openai/chat/completions

so the agent's ``base_url`` is simply::

    http://<ollie-host>/tuners/{tuner_id}/runs/{run_id}/openai

(An OpenAI-compatible client appends ``/chat/completions`` to that base_url.)

Prerequisites
-------------
* ``ollie-rl`` server running (``uv run poe dev`` from the repo root).
* ``harbor`` installed in the environment (``uv add harbor`` or ``pip install
  harbor``) with a working container backend (local ``docker`` by default).
* Tasks extracted locally first::

      uv run python examples/code-contests/prepare_data.py --limit 64

Run it (from the repo root)::

    uv run python examples/code-contests/run_training.py --steps 200
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
DEFAULT_MODEL = "hosted_vllm/ollie"  # policy is selected by tuner_id in the URL


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
def run_base_url(base: str, tuner_id: str, run_id: str) -> str:
    """The base_url-addressed twin endpoint for this specific run."""
    return f"{base}/tuners/{tuner_id}/runs/{run_id}/openai"


async def run_rollout(
    *,
    base: str,
    tuner_id: str,
    run_id: str,
    datum_id: str,
    environment: str,
    model: str,
) -> float:
    """Execute one containerized Harbor trial and return its reward.

    In Harbor's own words, "a trial is a rollout that produces a reward", so one
    trial maps 1:1 onto one ollie-rl Run: a single agent attempt at a single
    task under a single ``run_id``. (A Harbor Job is just the parallel
    orchestration of many such trials — unnecessary here since concurrency and
    dispatch are owned by the driver, and a Job shares one agent ``base_url``
    across its trials, which our per-run ``run_id`` embedding cannot use.)

    Terminus 2 samples exclusively through ``run_base_url`` (which carries this
    run's id), so ollie-rl records every completion under ``run_id``. Harbor's
    verifier then runs the task's ``tests/test.sh`` and writes a reward file,
    surfaced here as a float.
    """
    config = TrialConfig(
        task=TaskConfig(path=TASKS_DIR / datum_id),
        trials_dir=TRIALS_DIR,
        agent=AgentConfig(
            name=AgentName.TERMINUS_2.value,
            model_name=model,
            kwargs={"base_url": run_base_url(base, tuner_id, run_id)},
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
            step = budget.get_nowait()
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
                model=args.model,
            )
        except Exception as exc:  # a crashed trial scores the failure reward.
            print(f"[driver] step {step:04d} trial error ({datum_id}): {exc}")
            reward = 0.0

        # Phase 3: report the reward; the server groups/advantages/trains.
        await submit_reward(client, tuner_id, run_id, reward)

        stats["rewards"].append(reward)
        window = stats["rewards"][-32:]
        avg = sum(window) / len(window)
        print(
            f"[driver] step {step:04d} task={datum_id:<20} "
            f"reward={reward:+.1f} avg32={avg:.3f}"
        )


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--recipe", default=DEFAULT_RECIPE)
    parser.add_argument("--trainer", default=DEFAULT_TRAINER)
    parser.add_argument("--name", default=DEFAULT_TUNER_NAME)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--environment",
        default="docker",
        help="Harbor EnvironmentType (docker, daytona, modal, ...).",
    )
    parser.add_argument(
        "--steps",
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
        for step in range(args.steps):
            budget.put_nowait(step)

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
