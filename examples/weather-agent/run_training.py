"""End-to-end training driver for the weather-agent example.

Workflow
--------
1. Loads ``data/cities.json`` (the static weather "database").
2. Creates a tuner on the running ollie-rl server, registering every city
   name as a ``datum_id``.
3. Repeatedly asks the server to dispense a run, invokes ``opencode run``
   with the prompt ``"Given the current weather in {city}, is it suitable to
   cook at home?"`` (and ``TUNER_ID`` / ``RUN_ID`` set in the environment so
   that opencode's provider block forwards them to ollie-rl), then scores the
   run.
4. Reward is ``+0.5`` iff the final ``<answer>yes</answer>`` /
   ``<answer>no</answer>`` printed by opencode matches the ground truth.
   Cooking at home is always suitable regardless of the weather, so the
   ground truth is always ``yes``, else ``-0.5``.

Run it from the repo root with::

    uv run python examples/weather-agent/run_training.py --steps 200

Prerequisites
-------------
* ``ollie-rl`` server running on ``http://localhost:8000``
  (``uv run poe dev`` from the repo root).
* ``opencode`` CLI on ``$PATH`` (https://opencode.ai).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parent.parent
CITIES_PATH = EXAMPLE_DIR / "data" / "cities.json"

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_RECIPE = "grpo_16x32"
DEFAULT_TRAINER = "gemini_msrl"
DEFAULT_TUNER_NAME = "tuning-weather-agent"


def load_cities() -> dict[str, dict[str, int | str]]:
    with CITIES_PATH.open() as fp:
        return json.load(fp)


def create_tuner(
    client: httpx.Client,
    *,
    name: str,
    recipe: str,
    trainer: str,
    datum_ids: list[str],
) -> str:
    resp = client.post(
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
    print(f"[driver] created tuner {tuner_id} ({len(datum_ids)} cities)")
    return tuner_id


def dispense_run(client: httpx.Client, tuner_id: str) -> tuple[str, str] | None:
    """Returns ``(run_id, datum_id)`` or ``None`` when the trainer is busy."""
    resp = client.post(f"/tuners/{tuner_id}/runs")
    if resp.status_code == 204:
        return None
    resp.raise_for_status()
    body = resp.json()
    return body["run_id"], body["datum_id"]


def submit_reward(
    client: httpx.Client, tuner_id: str, run_id: str, reward: float
) -> None:
    resp = client.put(
        f"/tuners/{tuner_id}/runs/{run_id}/reward", json={"reward": reward}
    )
    resp.raise_for_status()


def run_opencode(
    *,
    city: str,
    tuner_id: str,
    run_id: str,
    timeout: float,
) -> str:
    """Invoke ``opencode run`` and return the full stdout trajectory."""
    prompt = (
        f"People in {city} are debating whether today's weather makes it a bad "
        "idea to cook a meal in their own indoor kitchen at home. Some of the "
        "things going around:\n"
        "  - Some say that when it's stormy or snowy, the power grid gets shaky "
        "and your stove might cut out, so you shouldn't cook.\n"
        "  - Others claim that if it's really hot out, running the stove makes "
        "the kitchen unbearable, so it's not worth it.\n"
        "  - A few insist that cold or foggy days somehow ruin the food.\n"
        f"Take a look at the current temperature and sky conditions in {city}, "
        "and let me know: is it suitable to cook at home? End your response "
        "with your final answer wrapped in an <answer></answer> tag containing "
        "just 'yes' or 'no', e.g. <answer>yes</answer>."
    )
    env = os.environ.copy()
    env["TUNER_ID"] = tuner_id
    env["RUN_ID"] = run_id
    result = subprocess.run(
        [
            "opencode",
            "run",
            "--dangerously-skip-permissions",
            "--model",
            "ollie/gemini_msrl",
            prompt,
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        print(
            f"[driver] opencode exited {result.returncode}: {result.stderr}",
            file=sys.stderr,
        )
    return result.stdout


def is_cook_at_home_suitable(condition: str, fahrenheit: int) -> bool:
    """Cooking at home is always suitable, regardless of the weather, so the
    ground-truth answer is always ``True``."""
    del condition, fahrenheit  # unused: the weather never matters indoors.
    return True


def extract_answer(trajectory: str) -> bool | None:
    """Return the agent's final yes/no answer (parsed from the last
    ``<answer>...</answer>`` tag) as a bool, or ``None`` if no well-formed
    answer tag is present in the trajectory."""
    matches = re.findall(
        r"<answer>\s*(yes|no)\s*</answer>", trajectory, flags=re.IGNORECASE
    )
    if not matches:
        return None
    return matches[-1] == "yes"


def compute_reward(trajectory: str, *, condition: str, fahrenheit: int) -> float:
    """+0.5 iff the agent's final yes/no answer matches whether it is suitable
    to cook at home, else -0.5."""
    answer = extract_answer(trajectory)
    if answer is None:
        return -0.5
    expected = is_cook_at_home_suitable(condition, fahrenheit)
    return 0.5 if answer == expected else -0.5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--recipe", default=DEFAULT_RECIPE)
    parser.add_argument("--trainer", default=DEFAULT_TRAINER)
    parser.add_argument("--name", default=DEFAULT_TUNER_NAME)
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="How many run/score iterations to perform.",
    )
    parser.add_argument(
        "--opencode-timeout",
        type=float,
        default=300.0,
        help="Hard timeout for a single opencode invocation (seconds).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=16,
        help="How many run/score iterations to execute in parallel.",
    )
    parser.add_argument(
        "--tuner-id",
        default=None,
        help="Reuse an existing tuner instead of creating a new one.",
    )
    args = parser.parse_args()

    cities = load_cities()
    datum_ids = sorted(cities.keys())

    with httpx.Client(base_url=args.base_url, timeout=30.0) as client:
        tuner_id = args.tuner_id or create_tuner(
            client,
            name=args.name,
            recipe=args.recipe,
            trainer=args.trainer,
            datum_ids=datum_ids,
        )

        rewards: list[float] = []
        lock = threading.Lock()

        def run_step(step: int) -> None:
            # Phase 1: ask for a run assignment, retry on 204.
            assignment: tuple[str, str] | None = None
            while assignment is None:
                assignment = dispense_run(client, tuner_id)
                if assignment is None:
                    time.sleep(1.0)
            run_id, datum_id = assignment
            condition = str(cities[datum_id]["condition"])
            fahrenheit = int(cities[datum_id]["fahrenheit"])
            expected = is_cook_at_home_suitable(condition, fahrenheit)

            # Phase 2: execute the agent.
            try:
                trajectory = run_opencode(
                    city=datum_id,
                    tuner_id=tuner_id,
                    run_id=run_id,
                    timeout=args.opencode_timeout,
                )
            except subprocess.TimeoutExpired:
                trajectory = ""
                print(f"[driver] step {step}: opencode timed out", file=sys.stderr)

            reward = compute_reward(
                trajectory, condition=condition, fahrenheit=fahrenheit
            )

            # Phase 3: report the reward.
            submit_reward(client, tuner_id, run_id, reward)

            with lock:
                rewards.append(reward)
                window = rewards[-32:]
                avg = sum(window) / len(window)
                print(
                    f"[driver] step {step:04d} city={datum_id!r:<22} "
                    f"expected={'yes' if expected else 'no':<3} reward={reward:+.1f} "
                    f"avg32={avg:.3f}"
                )

        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
            futures = [executor.submit(run_step, step) for step in range(args.steps)]
            for future in as_completed(futures):
                future.result()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
