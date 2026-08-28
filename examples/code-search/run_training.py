"""Train the Ollie agent on prepared SWE-rebench code-localization tasks.

Prepare repositories first, start ollie-rl, then run from the repository root::

    uv run python examples/code-search/prepare_data.py --limit 64
    uv run python examples/code-search/run_training.py --runs 2000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import httpx

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_PREPARED_DIR = EXAMPLE_DIR / "prepared"
DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_RECIPE = "grpo_4x8"
DEFAULT_TRAINER = "fake"
DEFAULT_TUNER_NAME = "tuning-code-search-ollie"
DEFAULT_AGENT_MODEL = "ollie"
OLLIE_CLI_VERSION = "0.7.1"
OLLIE_MAX_STEPS = 100
EVAL_FRACTION = 0.05
EVAL_SPLIT_SEED = 0xBADBEEF
LOCALIZATION_RESULT_RE = re.compile(
    r"<localization_result>\s*(\{.*?\})\s*</localization_result>", re.DOTALL
)


@dataclass(frozen=True)
class Location:
    file: str
    class_name: str | None = None
    function_name: str | None = None


@dataclass(frozen=True)
class CodeSearchInstance:
    datum_id: str
    repo: str
    base_commit: str
    problem_statement: str
    file_changes: tuple[dict[str, Any], ...]
    snapshot: Path


@dataclass(frozen=True)
class LocalizationResult:
    locations: tuple[Location, ...]
    errors: tuple[str, ...]


@dataclass(frozen=True)
class Reward:
    total: float
    file: float
    module: float
    entity: float


def load_instances(prepared_dir: Path) -> dict[str, CodeSearchInstance]:
    instances_dir = prepared_dir / "instances"
    if not instances_dir.is_dir():
        raise ValueError(
            f"prepared instances not found: {instances_dir}; run prepare_data.py first"
        )

    instances: dict[str, CodeSearchInstance] = {}
    for instance_path in sorted(instances_dir.glob("*.json")):
        try:
            item = json.loads(instance_path.read_text(encoding="utf-8"))
            repo = item["repo"]
            org, repo_name = repo.split("/")
            base_commit = item["base_commit"]
            problem_statement = item["problem_statement"]
            file_changes = item["file_changes"]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid prepared instance: {instance_path}") from error

        snapshot = prepared_dir / "github.com" / org / repo_name / base_commit
        if not snapshot.is_dir():
            raise ValueError(f"snapshot not found for {instance_path.stem}: {snapshot}")
        if not isinstance(problem_statement, str) or not isinstance(file_changes, list):
            raise TypeError(f"invalid prepared instance: {instance_path}")

        datum_id = instance_path.stem
        instances[datum_id] = CodeSearchInstance(
            datum_id=datum_id,
            repo=repo,
            base_commit=base_commit,
            problem_statement=problem_statement,
            file_changes=tuple(file_changes),
            snapshot=snapshot.resolve(),
        )

    if not instances:
        raise ValueError(f"no prepared instances found under {instances_dir}")
    return instances


def split_train_eval(datum_ids: list[str]) -> tuple[list[str], list[str]]:
    if len(datum_ids) < 2:
        return list(datum_ids), []
    shuffled = list(datum_ids)
    random.Random(EVAL_SPLIT_SEED).shuffle(shuffled)
    eval_count = min(
        max(1, math.ceil(len(shuffled) * EVAL_FRACTION)), len(shuffled) - 1
    )
    return shuffled[eval_count:], shuffled[:eval_count]


def build_prompt(instance: CodeSearchInstance) -> str:
    return f"""You are a specialized code-localization agent. Identify the files, classes, and functions in the repository that need modification to resolve the issue below. Do not solve or implement the issue.

The repository is mounted read-only at /workspace. Search it before answering. Prioritize precision and recall, and return only locations that need modification. Every file path must be relative to /workspace and must not begin with './'.

For each location, always provide `file`. Provide `class_name` when the change belongs to a class, and `function_name` when it belongs to a function or method. Omit optional names when the change is at file scope.

<issue_description>
{instance.problem_statement}
</issue_description>

Your final response must contain a `localization_result` XML element whose contents are one JSON object with a `locations` array. Do not include an implementation or explanation outside that element. Each location object may contain `file`, `class_name`, and `function_name`."""


def _string_values(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _string_values(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _string_values(nested)


def parse_location(value: Any) -> Location:
    if not isinstance(value, dict):
        raise TypeError("location is not an object")
    file = value.get("file")
    class_name = value.get("class_name")
    function_name = value.get("function_name")
    if not isinstance(file, str) or not file:
        raise TypeError("location.file is not a non-empty string")
    path = PurePosixPath(file)
    if path.is_absolute() or ".." in path.parts or file.startswith("./"):
        raise ValueError(f"location.file is not repository-relative: {file!r}")
    if class_name is not None and not isinstance(class_name, str):
        raise TypeError("location.class_name is not a string or null")
    if function_name is not None and not isinstance(function_name, str):
        raise TypeError("location.function_name is not a string or null")
    return Location(file, class_name or None, function_name or None)


def parse_agent_output(output: str) -> LocalizationResult:
    payloads: list[dict[str, Any]] = []
    errors: list[str] = []
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            values = [line]
        else:
            values = list(_string_values(event))
            if isinstance(event, dict) and event.get("type") == "error":
                errors.append(": ".join(values) if values else line)

        for value in values:
            for match in LOCALIZATION_RESULT_RE.finditer(value):
                try:
                    payload = json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    payloads.append(payload)

    if not payloads:
        return LocalizationResult((), tuple(errors))

    try:
        raw_locations = payloads[-1]["locations"]
        if not isinstance(raw_locations, list):
            raise TypeError("locations is not an array")
        locations = tuple(dict.fromkeys(parse_location(item) for item in raw_locations))
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f"invalid localization result: {error}")
        locations = ()
    return LocalizationResult(locations, tuple(errors))


def f1_score(predicted: set[str], expected: set[str]) -> float:
    if not predicted or not expected:
        return 0.0
    true_positives = len(predicted & expected)
    precision = true_positives / len(predicted)
    recall = true_positives / len(expected)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def expected_targets(
    instance: CodeSearchInstance,
) -> tuple[set[str], set[str], set[str]]:
    files: set[str] = set()
    modules: set[str] = set()
    entities: set[str] = set()
    for change in instance.file_changes:
        file = change["file"]
        changes = change["changes"]
        files.add(file)
        modules.update(changes.get("edited_modules") or ())
        entities.update(changes.get("edited_entities") or ())
    return files, modules, entities


def predicted_targets(
    locations: tuple[Location, ...],
) -> tuple[set[str], set[str], set[str]]:
    files: set[str] = set()
    modules: set[str] = set()
    entities: set[str] = set()
    for location in locations:
        files.add(location.file)
        if location.class_name:
            module = f"{location.file}:{location.class_name}"
            modules.add(module)
            if location.function_name:
                entities.add(f"{module}.{location.function_name}")
        elif location.function_name:
            symbol = f"{location.file}:{location.function_name}"
            modules.add(symbol)
            entities.add(symbol)
    return files, modules, entities


def grade(instance: CodeSearchInstance, locations: tuple[Location, ...]) -> Reward:
    predicted = predicted_targets(locations)
    expected = expected_targets(instance)
    file_reward, module_reward, entity_reward = (
        f1_score(predicted_level, expected_level)
        for predicted_level, expected_level in zip(predicted, expected, strict=True)
    )
    return Reward(
        total=file_reward + module_reward + entity_reward,
        file=file_reward,
        module=module_reward,
        entity=entity_reward,
    )


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
        f"({len(train_datum_ids)} train / {len(eval_datum_ids)} eval instances)"
    )
    return tuner_id


async def resolve_tuner(
    client: httpx.AsyncClient,
    *,
    tuner_id: str | None,
    name: str,
    recipe: str,
    trainer: str,
    train_datum_ids: list[str],
    eval_datum_ids: list[str],
) -> str:
    if tuner_id:
        response = await client.get(f"/tuners/{tuner_id}")
        response.raise_for_status()
        print(f"[driver] resuming tuner {tuner_id}")
        return tuner_id

    try:
        response = await client.get("/tuners")
        response.raise_for_status()
        existing_id = next(
            (
                tuner["tuner_id"]
                for tuner in response.json().get("tuners", [])
                if tuner["name"] == name
            ),
            None,
        )
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
        print(f"[driver] could not look up an existing tuner: {error}")
        existing_id = None

    if existing_id:
        print(f"[driver] resuming tuner {existing_id} (name={name!r})")
        return existing_id
    return await create_tuner(
        client,
        name=name,
        recipe=recipe,
        trainer=trainer,
        train_datum_ids=train_datum_ids,
        eval_datum_ids=eval_datum_ids,
    )


async def dispense_run(client: httpx.AsyncClient, tuner_id: str) -> tuple[str, str]:
    while True:
        response = await client.post(f"/tuners/{tuner_id}/runs")
        if response.status_code == 204:
            try:
                delay = max(0.0, float(response.headers.get("Retry-After", "1")))
            except ValueError:
                delay = 1.0
            await asyncio.sleep(delay)
            continue
        response.raise_for_status()
        body = response.json()
        return body["run_id"], body["datum_id"]


async def run_ollie_agent(
    *,
    base_url: str,
    tuner_id: str,
    run_id: str,
    instance: CodeSearchInstance,
    model: str,
    executor: str,
    max_steps: int,
) -> LocalizationResult:
    run_base_url = f"{base_url.rstrip('/')}/tuners/{tuner_id}/runs/{run_id}/openai/v1"
    environment = {
        **os.environ,
        "OPENAI_BASE_URL": run_base_url,
        "OPENAI_API_KEY": "ollie",
    }
    command = (
        "npx",
        "--yes",
        f"@getollie/cli@{OLLIE_CLI_VERSION}",
        "agent",
        "--workspace-path",
        "/workspace",
        "-v",
        f"{instance.snapshot}:/workspace:ro",
        "--mode",
        "sandbox",
        "--executor",
        executor,
        "--ndjson",
        "--no-color",
        "--model",
        model,
        "--max-steps",
        str(max_steps),
        build_prompt(instance),
    )
    process = await asyncio.create_subprocess_exec(
        *command,
        env=environment,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()

    output = stdout.decode(errors="replace")
    result = parse_agent_output(output)
    if process.returncode != 0:
        details = list(result.errors[-1:])
        if stderr_detail := stderr.decode(errors="replace").strip():
            details.append(stderr_detail)
        if not details and output.strip():
            details.append(output.strip())
        detail = " | ".join(details) if details else "no error output"
        raise RuntimeError(
            f"Ollie agent exited with status {process.returncode}: {detail[-1000:]}"
        )
    return result


async def submit_reward(
    client: httpx.AsyncClient, tuner_id: str, run_id: str, reward: float
) -> None:
    response = await client.put(
        f"/tuners/{tuner_id}/runs/{run_id}/reward",
        json={"reward": reward},
    )
    response.raise_for_status()


async def execute_run(
    number: int,
    client: httpx.AsyncClient,
    tuner_id: str,
    instance: CodeSearchInstance,
    run_id: str,
    args: argparse.Namespace,
    rewards: list[float],
) -> None:
    try:
        result = await run_ollie_agent(
            base_url=args.base_url,
            tuner_id=tuner_id,
            run_id=run_id,
            instance=instance,
            model=args.agent_model,
            executor=args.ollie_executor,
            max_steps=args.ollie_max_steps,
        )
        reward = grade(instance, result.locations)
        await submit_reward(client, tuner_id, run_id, reward.total)
    except Exception as error:  # noqa: BLE001 - one failed rollout must not stop peers
        print(f"[driver] run {number:04d} {instance.datum_id} failed: {error}")
        return

    rewards.append(reward.total)
    window = rewards[-32:]
    print(
        f"[driver] run {number:04d} task={instance.datum_id} "
        f"reward={reward.total:.3f} "
        f"(file={reward.file:.3f} module={reward.module:.3f} "
        f"entity={reward.entity:.3f}) avg32={sum(window) / len(window):.3f} "
        f"locations={len(result.locations)}"
    )
    for error in result.errors:
        print(f"[driver] run {number:04d} agent warning: {error[-500:]}")


async def async_main(args: argparse.Namespace) -> int:
    instances = load_instances(args.prepared_dir)
    train_datum_ids, eval_datum_ids = split_train_eval(list(instances))
    timeout = httpx.Timeout(connect=10.0, read=90.0, write=30.0, pool=30.0)
    async with httpx.AsyncClient(base_url=args.base_url, timeout=timeout) as client:
        tuner_id = await resolve_tuner(
            client,
            tuner_id=args.tuner_id,
            name=args.name,
            recipe=args.recipe,
            trainer=args.trainer,
            train_datum_ids=train_datum_ids,
            eval_datum_ids=eval_datum_ids,
        )
        rewards: list[float] = []
        slots = asyncio.Semaphore(args.concurrency)

        async def execute_and_release(
            number: int,
            instance: CodeSearchInstance,
            run_id: str,
        ) -> None:
            try:
                await execute_run(
                    number, client, tuner_id, instance, run_id, args, rewards
                )
            finally:
                slots.release()

        async with asyncio.TaskGroup() as tasks:
            for number in range(args.runs):
                await slots.acquire()
                try:
                    run_id, datum_id = await dispense_run(client, tuner_id)
                    instance = instances[datum_id]
                except BaseException:
                    slots.release()
                    raise
                tasks.create_task(execute_and_release(number, instance, run_id))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--prepared-dir", type=Path, default=DEFAULT_PREPARED_DIR)
    parser.add_argument("--recipe", default=DEFAULT_RECIPE)
    parser.add_argument("--trainer", default=DEFAULT_TRAINER)
    parser.add_argument("--name", default=DEFAULT_TUNER_NAME)
    parser.add_argument("--agent-model", default=DEFAULT_AGENT_MODEL)
    parser.add_argument(
        "--ollie-executor",
        choices=("none", "daytona", "vercel"),
        default="none",
    )
    parser.add_argument("--ollie-max-steps", type=int, default=OLLIE_MAX_STEPS)
    parser.add_argument("--runs", type=int, default=2000)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--tuner-id", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.runs < 1:
        raise SystemExit("--runs must be at least 1")
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be at least 1")
    if args.ollie_max_steps < 1:
        raise SystemExit("--ollie-max-steps must be at least 1")
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
