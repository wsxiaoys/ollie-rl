"""Train the Ollie agent on generated string-transformation puzzles.

Generate the dataset first::

    uv run python examples/word-puzzle/prepare_data.py \
      -o examples/word-puzzle/data/puzzles.jsonl --seed 42

Then start ollie-rl and run::

    uv run python examples/word-puzzle/run_training.py --runs 2000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = EXAMPLE_DIR / "data" / "puzzles.jsonl"
DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_RECIPE = "grpo_4x8"
DEFAULT_TRAINER = "fake"
DEFAULT_TUNER_NAME = "tuning-word-puzzle-ollie"
DEFAULT_AGENT_MODEL = "ollie"
OLLIE_MAX_STEPS = 100
EVAL_FRACTION = 0.05
EVAL_SPLIT_SEED = 0xBADBEEF
FINAL_ANSWER_RE = re.compile(r"\$\\text\{([^{}]*)\}\$")


@dataclass(frozen=True)
class Puzzle:
    datum_id: str
    prompt: str
    reference: str


@dataclass(frozen=True)
class AgentResult:
    answer: str | None
    errors: tuple[str, ...]


def load_dataset(path: Path) -> dict[str, Puzzle]:
    """Load generator JSONL and assign stable, line-number-based datum ids."""
    puzzles: dict[str, Puzzle] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                contents = item["contents"]
                reference = item["references"]["reference"]
                prompt = "\n\n".join(
                    "".join(part["text"] for part in content["parts"])
                    for content in contents
                    if content["role"] == "user"
                )
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"invalid dataset row at line {line_number}"
                ) from error
            if not isinstance(reference, str) or not prompt:
                raise ValueError(f"invalid dataset row at line {line_number}")
            datum_id = f"puzzle-{line_number:06d}"
            puzzles[datum_id] = Puzzle(datum_id, prompt, reference)
    if not puzzles:
        raise ValueError(f"dataset is empty: {path}")
    return puzzles


def split_train_eval(datum_ids: list[str]) -> tuple[list[str], list[str]]:
    """Shuffle deterministically and hold out five percent for evaluation."""
    if len(datum_ids) < 2:
        return list(datum_ids), []

    shuffled = list(datum_ids)
    random.Random(EVAL_SPLIT_SEED).shuffle(shuffled)
    eval_count = min(
        max(1, math.ceil(len(shuffled) * EVAL_FRACTION)), len(shuffled) - 1
    )
    return shuffled[eval_count:], shuffled[:eval_count]


def _string_values(value: Any):
    """Yield strings recursively from one Ollie NDJSON event."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _string_values(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _string_values(nested)


def parse_agent_output(output: str) -> AgentResult:
    """Extract the final answer and errors from Ollie's NDJSON event stream."""
    answers: list[str] = []
    errors: list[str] = []
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            values = [line]
        else:
            values = list(_string_values(event))
            if isinstance(event, dict) and event.get("type") == "error":
                details = [value for value in values if value not in {"error", "ERROR"}]
                errors.append(": ".join(details) if details else line)

        for value in values:
            answers.extend(
                answer.strip()
                for answer in FINAL_ANSWER_RE.findall(value)
                if answer.strip() != "your answer"
            )

    return AgentResult(
        answer=answers[-1] if answers else None,
        errors=tuple(errors),
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
        f"({len(train_datum_ids)} train / {len(eval_datum_ids)} eval puzzles)"
    )
    return tuner_id


async def dispense_run(client: httpx.AsyncClient, tuner_id: str) -> tuple[str, str]:
    """Wait until the tuner can dispense one run assignment."""
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
    prompt: str,
    model: str,
    executor: str,
) -> AgentResult:
    """Run one Ollie CLI agent and parse its NDJSON event stream."""
    # A trailing slash on --base-url would otherwise produce a `//tuners/...`
    # path. Railway does not normalize that path and returns 404 before the
    # request reaches ollie-rl's OpenAI-compatible endpoint.
    run_base_url = f"{base_url.rstrip('/')}/tuners/{tuner_id}/runs/{run_id}/openai/v1"
    environment = {
        **os.environ,
        "OPENAI_BASE_URL": run_base_url,
        "OPENAI_API_KEY": "ollie",
    }
    with tempfile.TemporaryDirectory(prefix=f"ollie-{run_id}-") as workspace:
        command = (
            "npx",
            "--yes",
            f"@getollie/cli@0.6.1",
            "agent",
            "--workspace-path",
            "/workspace",
            "-v",
            f"{workspace}:/workspace",
            "--ndjson",
            "--no-color",
            "--model",
            model,
            "--executor",
            executor,
            "--max-steps",
            str(OLLIE_MAX_STEPS),
            prompt,
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
        if details:
            detail = " | ".join(details)
            raise RuntimeError(
                f"Ollie agent exited with status {process.returncode}: {detail[-1000:]}"
            )
        raise RuntimeError(
            f"Ollie agent exited with status {process.returncode} without error output"
        )

    if result.errors and result.answer is None:
        raise RuntimeError(
            f"Ollie agent reported an error: {result.errors[-1][-1000:]}"
        )
    return result


async def submit_reward(
    client: httpx.AsyncClient, tuner_id: str, run_id: str, reward: float
) -> None:
    response = await client.put(
        f"/tuners/{tuner_id}/runs/{run_id}/reward",
        json={"reward": reward},
    )
    if response.status_code == 403:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"reward rejected (403 Forbidden): {detail}")
    response.raise_for_status()


async def execute_run(
    number: int,
    client: httpx.AsyncClient,
    tuner_id: str,
    puzzle: Puzzle,
    run_id: str,
    args: argparse.Namespace,
    rewards: list[float],
) -> None:
    try:
        result = await run_ollie_agent(
            base_url=args.base_url,
            tuner_id=tuner_id,
            run_id=run_id,
            prompt=puzzle.prompt,
            model=args.agent_model,
            executor=args.ollie_executor,
        )
        reward = float(result.answer == puzzle.reference)
        await submit_reward(client, tuner_id, run_id, reward)
    except Exception as error:
        # Do not invent a reward when the agent or submission fails. The lease
        # expires and the server can dispense this datum again later.
        print(f"[driver] run {number:04d} {puzzle.datum_id} failed: {error}")
        return

    rewards.append(reward)
    window = rewards[-32:]
    print(
        f"[driver] run {number:04d} task={puzzle.datum_id} "
        f"reward={reward:.0f} avg32={sum(window) / len(window):.3f} "
        f"answer={result.answer!r}"
    )


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
    """Resume an explicit or same-name tuner, or create a new one."""
    if tuner_id:
        response = await client.get(f"/tuners/{tuner_id}")
        response.raise_for_status()
        details = response.json()
        print(f"[driver] resuming tuner {tuner_id} (name={details.get('name')!r})")
        return tuner_id

    try:
        response = await client.get("/tuners")
        response.raise_for_status()
        tuner_id = next(
            (
                tuner["tuner_id"]
                for tuner in response.json().get("tuners", [])
                if tuner["name"] == name
            ),
            None,
        )
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
        print(
            "[driver] warning: could not find an existing tuner by name; "
            f"creating one instead: {error}"
        )

    if tuner_id:
        print(f"[driver] resuming tuner {tuner_id} (name={name!r})")
        return tuner_id

    return await create_tuner(
        client,
        name=name,
        recipe=recipe,
        trainer=trainer,
        train_datum_ids=train_datum_ids,
        eval_datum_ids=eval_datum_ids,
    )


async def async_main(args: argparse.Namespace) -> int:
    puzzles = load_dataset(args.dataset)
    train_datum_ids, eval_datum_ids = split_train_eval(list(puzzles))
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
        slots = asyncio.Semaphore(max(1, args.concurrency))

        async def execute_and_release(number: int, puzzle: Puzzle, run_id: str) -> None:
            try:
                await execute_run(
                    number,
                    client,
                    tuner_id,
                    puzzle,
                    run_id,
                    args,
                    rewards,
                )
            finally:
                slots.release()

        async with asyncio.TaskGroup() as tasks:
            for number in range(args.runs):
                await slots.acquire()
                try:
                    run_id, datum_id = await dispense_run(client, tuner_id)
                    try:
                        puzzle = puzzles[datum_id]
                    except KeyError as error:
                        raise ValueError(
                            f"tuner dispensed unknown datum {datum_id!r}; "
                            "resume with the dataset used to create it"
                        ) from error
                except BaseException:
                    slots.release()
                    raise
                tasks.create_task(execute_and_release(number, puzzle, run_id))

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--recipe", default=DEFAULT_RECIPE)
    parser.add_argument("--trainer", default=DEFAULT_TRAINER)
    parser.add_argument("--name", default=DEFAULT_TUNER_NAME)
    parser.add_argument("--agent-model", default=DEFAULT_AGENT_MODEL)
    parser.add_argument(
        "--ollie-executor",
        choices=("none", "local", "daytona"),
        default="none",
    )
    parser.add_argument("--runs", type=int, default=2000)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--tuner-id", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.runs < 1:
        raise SystemExit("--runs must be at least 1")
    if not args.dataset.is_file():
        raise SystemExit(
            f"dataset not found: {args.dataset}\n"
            "Generate it first with examples/word-puzzle/prepare_data.py."
        )
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
