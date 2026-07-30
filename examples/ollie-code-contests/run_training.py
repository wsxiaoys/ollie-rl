"""Train Ollie on Ollie-adapted CodeContests Harbor tasks.

Prepare tasks first, then run from the repository root::

    uv run python examples/ollie-code-contests/prepare_data.py --limit 64
    uv run python examples/ollie-code-contests/run_training.py --runs 200
"""

from __future__ import annotations

import argparse
import asyncio
import math
import random
import tomllib
from collections.abc import Awaitable, Callable
from pathlib import Path

import httpx
from harbor.models.environment_type import EnvironmentType
from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
    VerifierConfig,
)
from harbor.trial.hooks import TrialEvent, TrialHookEvent
from harbor.trial.trial import Trial

EXAMPLE_DIR = Path(__file__).resolve().parent
TASKS_DIR = EXAMPLE_DIR / "tasks"
TRIALS_DIR = EXAMPLE_DIR / "trials"

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_RECIPE = "grpo_16x32"
DEFAULT_TRAINER = "fake"
DEFAULT_TUNER_NAME = "tuning-ollie-code-contests"
DEFAULT_AGENT_MODEL = "openai/ollie"
DEFAULT_OLLIE_EXECUTOR = "daytona"
DEFAULT_OLLIE_WORKSPACE_PATH = "/app"
DEFAULT_DAYTONA_AUTO_STOP_MINS = 30
DEFAULT_DAYTONA_AUTO_DELETE_MINS = 5

OLLIE_AGENT_IMPORT_PATH = "ollie.harbor_agent:OllieAgent"
OLLIE_ENVIRONMENT_IMPORT_PATH = "ollie.harbor_environment:OllieEnvironment"
STANDARD_VERIFIER_IMPORT_PATH = "harbor.verifier.verifier:Verifier"

EVAL_FRACTION = 0.05
EVAL_SPLIT_SEED = 0xBADBEEF
CONTROL_PLANE_BACKOFF_BASE_SEC = 1.0
CONTROL_PLANE_BACKOFF_CAP_SEC = 30.0


def discover_datum_ids() -> list[str]:
    if not TASKS_DIR.exists():
        raise SystemExit(
            f"No tasks found in {TASKS_DIR}. Run:\n"
            f"  uv run python {EXAMPLE_DIR / 'prepare_data.py'} --limit 64"
        )
    datum_ids = sorted(
        path.name
        for path in TASKS_DIR.iterdir()
        if path.is_dir() and (path / "task.toml").exists()
    )
    if not datum_ids:
        raise SystemExit(f"No Harbor tasks found under {TASKS_DIR}")
    return datum_ids


def validate_ollie_tasks(datum_ids: list[str]) -> None:
    incompatible: list[str] = []
    for datum_id in datum_ids:
        config_path = TASKS_DIR / datum_id / "task.toml"
        config = tomllib.loads(config_path.read_text(encoding="utf-8"))
        verifier_config = config.get("verifier", {})
        verifier_environment = verifier_config.get("environment", {})
        if (
            config.get("artifacts") != ["/app"]
            or config.get("environment", {}).get("workdir") != "/app"
            or verifier_config.get("environment_mode") != "separate"
            or not verifier_environment.get("docker_image")
        ):
            incompatible.append(datum_id)
    if incompatible:
        sample = ", ".join(incompatible[:5])
        suffix = "..." if len(incompatible) > 5 else ""
        raise SystemExit(
            "Tasks need the current Ollie verifier configuration; repair them "
            f"with `uv run python {EXAMPLE_DIR / 'prepare_data.py'} "
            f"--repair-existing`. Incompatible tasks: {sample}{suffix}"
        )


def split_train_eval(datum_ids: list[str]) -> tuple[list[str], list[str]]:
    if len(datum_ids) < 2:
        return list(datum_ids), []
    eval_count = min(
        max(1, math.ceil(len(datum_ids) * EVAL_FRACTION)), len(datum_ids) - 1
    )
    shuffled = list(datum_ids)
    random.Random(EVAL_SPLIT_SEED).shuffle(shuffled)
    return shuffled[eval_count:], shuffled[:eval_count]


async def create_tuner(
    client: httpx.AsyncClient,
    *,
    name: str,
    recipe: str,
    trainer: str,
    train_datum_ids: list[str],
    eval_datum_ids: list[str],
) -> str:
    response = await client.post(
        "/tuners",
        json={
            "name": name,
            "recipe": recipe,
            "trainer": trainer,
            "train_datum_ids": train_datum_ids,
            "eval_datum_ids": eval_datum_ids,
        },
    )
    response.raise_for_status()
    tuner_id = response.json()["tuner_id"]
    print(
        f"[driver] created tuner {tuner_id} "
        f"({len(train_datum_ids)} train / {len(eval_datum_ids)} eval tasks)"
    )
    return tuner_id


async def _request_with_retry(
    description: str, send: Callable[[], Awaitable[httpx.Response]]
) -> httpx.Response:
    attempt = 0
    while True:
        try:
            return await send()
        except (httpx.TimeoutException, httpx.TransportError) as error:
            ceiling = min(
                CONTROL_PLANE_BACKOFF_CAP_SEC,
                CONTROL_PLANE_BACKOFF_BASE_SEC * (2**attempt),
            )
            delay = random.uniform(0, ceiling)
            print(
                f"[driver] {description}: transient {type(error).__name__}; "
                f"retrying in {delay:.1f}s"
            )
            await asyncio.sleep(delay)
            attempt += 1


async def dispense_runs(
    client: httpx.AsyncClient, tuner_id: str, batch_size: int
) -> list[tuple[str, str]]:
    while True:
        response = await _request_with_retry(
            f"batch dispense (tuner {tuner_id}, size {batch_size})",
            lambda: client.post(
                f"/tuners/{tuner_id}/runs", params={"batch_size": batch_size}
            ),
        )
        if response.status_code == 204:
            try:
                retry_after = max(0.0, float(response.headers.get("Retry-After", "1")))
            except ValueError:
                retry_after = 1.0
            await asyncio.sleep(retry_after)
            continue
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, list) or not body:
            raise ValueError("batch dispense response must be a non-empty list")
        return [(item["run_id"], item["datum_id"]) for item in body]


async def submit_reward(
    client: httpx.AsyncClient, tuner_id: str, run_id: str, reward: float
) -> bool:
    response = await _request_with_retry(
        f"reward (run {run_id})",
        lambda: client.put(
            f"/tuners/{tuner_id}/runs/{run_id}/reward", json={"reward": reward}
        ),
    )
    if response.status_code in (403, 409):
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        print(
            f"[driver] run {run_id} reward not accepted "
            f"({response.status_code} {response.reason_phrase}; likely already "
            f"rewarded, expired, or missing completions): {detail}"
        )
        return False
    response.raise_for_status()
    return True


def run_openai_base_url(base: str, tuner_id: str, run_id: str) -> str:
    return f"{base}/tuners/{tuner_id}/runs/{run_id}/openai/v1"


def agent_timed_out(trial_result) -> bool:
    info = getattr(trial_result, "exception_info", None)
    return bool(info and info.exception_type == "AgentTimeoutError")


def extract_reward(trial_result) -> float | None:
    verifier_result = getattr(trial_result, "verifier_result", None)
    rewards = getattr(verifier_result, "rewards", None) if verifier_result else None
    if not rewards:
        return None
    if "reward" in rewards:
        return float(rewards["reward"])
    if len(rewards) == 1:
        return float(next(iter(rewards.values())))
    return None


async def run_rollout(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    tuner_id: str,
    run_id: str,
    datum_id: str,
    verifier_environment: str,
    agent_model: str,
    ollie_executor: str,
    ollie_max_steps: int | None,
    ollie_command_timeout_ms: int | None,
    agent_timeout_multiplier: float | None,
) -> float | None:
    environment_kwargs: dict = {}
    if verifier_environment.lower() == "daytona":
        environment_kwargs = {
            "auto_stop_interval_mins": DEFAULT_DAYTONA_AUTO_STOP_MINS,
            "auto_delete_interval_mins": DEFAULT_DAYTONA_AUTO_DELETE_MINS,
        }

    agent_kwargs: dict = {
        "openai_base_url": run_openai_base_url(base_url, tuner_id, run_id),
        "openai_api_key": "ollie",
        "executor": ollie_executor,
        "workspace_path": DEFAULT_OLLIE_WORKSPACE_PATH,
    }
    optional_kwargs = {
        "max_steps": ollie_max_steps,
        "command_timeout_ms": ollie_command_timeout_ms,
    }
    agent_kwargs.update(
        {key: value for key, value in optional_kwargs.items() if value is not None}
    )

    config = TrialConfig(
        task=TaskConfig(path=TASKS_DIR / datum_id),
        trials_dir=TRIALS_DIR,
        agent=AgentConfig(
            import_path=OLLIE_AGENT_IMPORT_PATH,
            model_name=agent_model,
            kwargs=agent_kwargs,
        ),
        environment=EnvironmentConfig(
            import_path=OLLIE_ENVIRONMENT_IMPORT_PATH,
            kwargs={
                "verifier_environment": verifier_environment,
                **environment_kwargs,
            },
        ),
        verifier=VerifierConfig(import_path=STANDARD_VERIFIER_IMPORT_PATH),
        agent_timeout_multiplier=agent_timeout_multiplier,
    )
    trial = await Trial.create(config)

    async def skip_verifier_for_server_reward(_event: TrialHookEvent) -> None:
        """Avoid grading when sampling already finalized the run on the server."""
        try:
            response = await client.get(f"/tuners/{tuner_id}/runs/{run_id}")
            response.raise_for_status()
            reward = response.json().get("run", {}).get("reward")
        except Exception as error:
            # This check is only an optimization. Fail open so a temporary
            # observability/API issue does not suppress legitimate grading.
            print(
                f"[driver] run {run_id} could not check server reward before "
                f"verification; continuing with verifier: {error}"
            )
            return

        if reward is not None:
            trial.config.verifier.disable = True
            print(
                f"[driver] run {run_id} already has server reward={float(reward):+.1f}; "
                "skipping verifier"
            )

    trial.add_hook(TrialEvent.AGENT_END, skip_verifier_for_server_reward)
    result = await trial.run()
    if agent_timed_out(result):
        graded = result.verifier_result is not None
        print(
            f"[driver] run {run_id} hit the agent timeout "
            f"(graded={graded}); skipping reward"
        )
        return None
    if result.exception_info:
        raise RuntimeError(
            f"{result.exception_info.exception_type}: "
            f"{result.exception_info.exception_message}"
        )
    return extract_reward(result)


async def execute_run(
    index: int,
    client: httpx.AsyncClient,
    tuner_id: str,
    run_id: str,
    datum_id: str,
    args: argparse.Namespace,
    rewards: list[float],
) -> None:
    try:
        reward = await run_rollout(
            client=client,
            base_url=args.base_url,
            tuner_id=tuner_id,
            run_id=run_id,
            datum_id=datum_id,
            verifier_environment=args.verifier_environment,
            agent_model=args.agent_model,
            ollie_executor=args.ollie_executor,
            ollie_max_steps=args.ollie_max_steps,
            ollie_command_timeout_ms=args.ollie_command_timeout_ms,
            agent_timeout_multiplier=args.agent_timeout_multiplier,
        )
    except Exception as error:
        print(
            f"[driver] run {index:04d} trial crashed ({datum_id}); "
            f"skipping reward: {error}"
        )
        return
    if reward is None:
        print(
            f"[driver] run {index:04d} task={datum_id} no reward to submit "
            f"(server finalized the run or verifier produced no scalar reward)"
        )
        return
    if not await submit_reward(client, tuner_id, run_id, reward):
        print(
            f"[driver] run {index:04d} task={datum_id} reward submission "
            f"was rejected; excluding it from local stats"
        )
        return
    rewards.append(reward)
    window = rewards[-32:]
    print(
        f"[driver] run {index:04d} task={datum_id:<20} "
        f"reward={reward:+.1f} avg32={sum(window) / len(window):.3f}"
    )


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--recipe", default=DEFAULT_RECIPE)
    parser.add_argument("--trainer", default=DEFAULT_TRAINER)
    parser.add_argument("--name", default=DEFAULT_TUNER_NAME)
    parser.add_argument("--agent-model", default=DEFAULT_AGENT_MODEL)
    parser.add_argument(
        "--ollie-executor",
        choices=("none", "local", "daytona"),
        default=DEFAULT_OLLIE_EXECUTOR,
    )
    parser.add_argument("--ollie-max-steps", type=int, default=None)
    parser.add_argument("--ollie-command-timeout-ms", type=int, default=None)
    parser.add_argument(
        "--verifier-environment",
        "--environment",
        dest="verifier_environment",
        default="docker",
        help=(
            "Separate Harbor verifier backend (for example, docker or daytona). "
            "The --environment spelling is retained as a compatibility alias."
        ),
    )
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--agent-timeout-multiplier", type=float, default=None)
    parser.add_argument("--tuner-id", default=None)
    args = parser.parse_args()

    for option in ("ollie_max_steps", "ollie_command_timeout_ms"):
        value = getattr(args, option)
        if value is not None and value <= 0:
            parser.error(f"--{option.replace('_', '-')} must be positive")

    if ":" not in args.verifier_environment:
        try:
            EnvironmentType(args.verifier_environment)
        except ValueError:
            parser.error(
                "--verifier-environment must be a built-in Harbor environment "
                "name (for example, docker or daytona) or an import path; "
                f"received {args.verifier_environment!r}. Ollie is already used "
                "as the agent environment and cannot verify itself."
            )

    TRIALS_DIR.mkdir(parents=True, exist_ok=True)
    datum_ids = discover_datum_ids()
    validate_ollie_tasks(datum_ids)
    timeout = httpx.Timeout(connect=10.0, read=90.0, write=30.0, pool=30.0)
    async with httpx.AsyncClient(base_url=args.base_url, timeout=timeout) as client:
        tuner_id = args.tuner_id
        if tuner_id:
            try:
                response = await client.get(f"/tuners/{tuner_id}")
                response.raise_for_status()
                details = response.json()
                recipe_name = details.get("recipe", {}).get("name", "unknown")
                print(
                    f"[driver] adapting to existing tuner {tuner_id} "
                    f"(name={details.get('name')!r}, recipe={recipe_name!r})"
                )
            except httpx.HTTPStatusError as error:
                if error.response.status_code == 404:
                    print(f"[driver] error: tuner {tuner_id} not found")
                    return 1
                raise
        else:
            try:
                response = await client.get("/tuners")
                response.raise_for_status()
                tuner_id = next(
                    (
                        tuner["tuner_id"]
                        for tuner in response.json().get("tuners", [])
                        if tuner["name"] == args.name
                    ),
                    None,
                )
                if tuner_id:
                    print(
                        f"[driver] adapting to existing tuner {tuner_id} "
                        f"(name={args.name!r})"
                    )
            except Exception as error:
                print(
                    "[driver] warning: could not list tuners to check for "
                    f"existing name: {error}"
                )
        if not tuner_id:
            train_ids, eval_ids = split_train_eval(datum_ids)
            tuner_id = await create_tuner(
                client,
                name=args.name,
                recipe=args.recipe,
                trainer=args.trainer,
                train_datum_ids=train_ids,
                eval_datum_ids=eval_ids,
            )

        rewards: list[float] = []
        concurrency = max(1, args.concurrency)
        next_run = 0
        active: set[asyncio.Task[None]] = set()
        while next_run < args.runs or active:
            capacity = concurrency - len(active)
            if next_run < args.runs and capacity > 0:
                assignments = await dispense_runs(
                    client, tuner_id, min(capacity, args.runs - next_run, 1024)
                )
                for run_id, datum_id in assignments:
                    index = next_run
                    next_run += 1
                    active.add(
                        asyncio.create_task(
                            execute_run(
                                index,
                                client,
                                tuner_id,
                                run_id,
                                datum_id,
                                args,
                                rewards,
                            )
                        )
                    )
                continue
            completed, active = await asyncio.wait(
                active, return_when=asyncio.FIRST_COMPLETED
            )
            for task in completed:
                task.result()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
