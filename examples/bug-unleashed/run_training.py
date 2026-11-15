"""Train the Ollie agent to review pull requests on prepared BugUnleashed cases.

Run ``prepare_data.py`` first, start ollie-rl, then point this driver at it
(the README shows the full command). The judge needs an OpenAI-compatible
endpoint: pass ``--judge-base-url`` and ``--judge-api-key`` (or set
``OPENAI_API_KEY``). The reward table and the judge prompt live in
``reward.py`` and are described in the README.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reward


# NDJSON lines can carry a whole model turn, so the default 64 KiB StreamReader
# limit would raise instead of yielding the line.
_STREAM_LIMIT = 16 * 1024 * 1024


def _kill_process_group(process: asyncio.subprocess.Process) -> None:
    if hasattr(os, "killpg"):
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.kill()  # best-effort fallback where killpg is unavailable


EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = EXAMPLE_DIR / "data"
DEFAULT_BASE_URL = "http://localhost:8000"
# `RecipeInput` layers over `Recipe` defaults rather than over a named recipe,
# so the grpo_4x8 shape is spelled out next to the required override. Without
# `centering` the normalizer divides by the group std and the -2.0 clean-PR
# penalty loses its magnitude.
DEFAULT_RECIPE = (
    '{"group_size": 4, "num_groups_per_batch": 8, "reward_normalizer": "centering"}'
)
DEFAULT_TRAINER = "fake"
DEFAULT_TUNER_NAME = "tuning-bug-unleashed-ollie"
DEFAULT_AGENT_MODEL = "ollie"
OLLIE_CLI_VERSION = "0.9.1"
OLLIE_MAX_STEPS = 80
# `--ollie-max-steps` already bounds a rollout's work, so there is no
# wall-clock timeout by default: a fixed deadline only truncates slow-but-
# progressing runs and leaves them unrewardable (they lapse into the server's
# `expired` state). What still needs catching is a run that stops progressing --
# a wedged tool call, or a CLI orphaned by a crash that no longer calls the
# model at all -- so the driver watches stdout and kills the group after
# `DEFAULT_RUN_IDLE_TIMEOUT` seconds with no new event.
DEFAULT_RUN_TIMEOUT = 0
DEFAULT_RUN_IDLE_TIMEOUT = 10 * 60
# A run younger than this might still be worked by a predecessor driver's
# orphaned CLI, so the startup sweep leaves it alone (the server's run lease is
# 15 minutes).
DEFAULT_SALVAGE_MIN_AGE = 15 * 60
# Prepared diffs: p50 ~10k characters, p90 ~49k. The tail is truncated with a
# note; the agent can read the rest from the read-only checkout.
DEFAULT_MAX_DIFF_CHARS = 60_000
# `AdapterSize` enum accepted by the MSRL tuning API, smallest to largest.
ADAPTER_SIZES = (
    "ADAPTER_SIZE_ONE",
    "ADAPTER_SIZE_TWO",
    "ADAPTER_SIZE_FOUR",
    "ADAPTER_SIZE_EIGHT",
    "ADAPTER_SIZE_SIXTEEN",
    "ADAPTER_SIZE_THIRTY_TWO",
)

PROMPT_TEMPLATE = """You are reviewing a pull request for defects it introduces. The repository is
mounted read-only at /workspace and is the tree *after* the PR was applied.
Use it: read the changed files, grep for call sites, and follow the code paths
the change touches.

Report only defects this PR introduces. Do not report pre-existing bugs,
style or naming nits, missing tests, or general risks. A finding must name the
specific mechanism and the condition that triggers it — "may cause a null
dereference" is not a finding.

At most 3 findings, majors and minors combined. Use severity "major" only when
you are confident enough to bet that this is a real defect introduced by the
PR; use "minor" for a plausible but unconfirmed one. Prefer fewer, sharper
findings over guesses.

Every finding is an object with exactly these keys: `severity` ("major" or
"minor"), `path` (repository-relative, no leading "./"), `line` (in the post-PR
file), `title` (one line), and `body` (mechanism + condition).

Your final response must contain a `<review_findings>` element whose contents
are one JSON array of finding objects. Do not include explanation outside that
element.

<title>
{pr_title}
</title>

<description>
{pr_body}
</description>

<diff>
{diff}
</diff>"""


@dataclass(frozen=True)
class ReviewTask:
    """One dispensed example: the prompt inputs plus the scoring gold."""

    example_id: str
    split: str
    kind: str
    repo: str
    pr: int
    title: str
    body: str
    diff: str
    snapshot: Path
    example: reward.Example


@dataclass
class SplitStats:
    """Running eval summary for one split."""

    rewards: list[float] = field(default_factory=list)
    bug: int = 0
    bug_strict: int = 0
    clean: int = 0
    clean_quiet: int = 0

    def record(self, kind: str, result: reward.RewardResult) -> None:
        self.rewards.append(result.total)
        if kind == "bug":
            self.bug += 1
            self.bug_strict += int(result.strict_match)
        else:
            self.clean += 1
            self.clean_quiet += int(result.total == reward.CLEAN_REWARD)

    def summary(self) -> str:
        if not self.rewards:
            return "n=0"
        parts = [
            f"n={len(self.rewards)}",
            f"avg={sum(self.rewards) / len(self.rewards):.3f}",
        ]
        if self.bug:
            rate = self.bug_strict / self.bug
            parts.append(f"strict={self.bug_strict}/{self.bug} ({rate:.2f})")
        if self.clean:
            rate = self.clean_quiet / self.clean
            parts.append(f"quiet={self.clean_quiet}/{self.clean} ({rate:.2f})")
        return " ".join(parts)


def truncate_diff(diff: str, limit: int) -> str:
    """Cut the diff to `limit` characters, telling the agent what is missing."""
    if len(diff) <= limit:
        return diff
    omitted = len(diff) - limit
    return (
        f"{diff[:limit]}\n\n[diff truncated: {omitted} more characters. The full "
        "post-PR tree is in /workspace.]\n"
    )


def build_prompt(task: ReviewTask) -> str:
    return PROMPT_TEMPLATE.format(
        pr_title=task.title,
        pr_body=task.body or "(no description)",
        diff=task.diff,
    )


def load_tasks(
    data_dir: Path,
    examples: dict[str, reward.Example],
    max_diff_chars: int,
) -> dict[str, ReviewTask]:
    """Load one `ReviewTask` per prepared instance, keyed by example id."""
    instances_dir = data_dir / "instances"
    if not instances_dir.is_dir():
        raise ValueError(
            f"prepared instances not found: {instances_dir}; run prepare_data.py first"
        )

    tasks: dict[str, ReviewTask] = {}
    for path in sorted(instances_dir.glob("*.json")):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
            example_id = item["example_id"]
            case_id = item["case_id"]
            split = item["split"]
            kind = item["kind"]
            title = item["title"]
            repo = item["repo"]
            pr = int(item["pr"])
            snapshot = data_dir / item["snapshot"]
            diff_path = data_dir / item["diff"]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid prepared instance: {path}") from error

        example = examples.get(example_id)
        if example is None:
            raise ValueError(f"instance {path} has no matching example")
        if split not in {"train", "val"}:
            raise ValueError(f"instance {path} has unexpected split {split!r}")
        if not snapshot.is_dir():
            raise ValueError(f"instance {path} snapshot is missing: {snapshot}")

        body = ""
        pr_path = data_dir / "pr" / f"{case_id}.json"
        if pr_path.is_file():
            payload = json.loads(pr_path.read_text(encoding="utf-8"))
            body = (payload.get("body") or "").strip()
        diff = truncate_diff(
            diff_path.read_text(encoding="utf-8", errors="replace"),
            max_diff_chars,
        )

        tasks[example_id] = ReviewTask(
            example_id=example_id,
            split=split,
            kind=kind,
            repo=repo,
            pr=pr,
            title=title,
            body=body,
            diff=diff,
            snapshot=snapshot,
            example=example,
        )

    if not tasks:
        raise ValueError(f"no prepared instances found under {instances_dir}")
    return tasks


def parse_recipe(value: str) -> Any:
    if not value.lstrip().startswith("{"):
        return value
    try:
        recipe = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f"invalid recipe JSON: {error.msg}") from error
    if not isinstance(recipe, dict):
        raise argparse.ArgumentTypeError("recipe JSON must be an object")
    return recipe


async def create_tuner(
    client: httpx.AsyncClient,
    *,
    name: str,
    recipe: Any,
    trainer: str,
    train_datum_ids: list[str],
    eval_datum_ids: list[str],
    trainer_params: dict[str, Any] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "name": name,
        "recipe": recipe,
        "trainer": trainer,
        "train_datum_ids": train_datum_ids,
        "eval_datum_ids": eval_datum_ids,
    }
    if trainer_params:
        payload["trainer_params"] = trainer_params
    response = await client.post("/tuners", json=payload)
    if response.is_error:
        # Vertex rejects an unknown base model here; the body carries its list
        # of accepted models, which is the only place to learn it.
        raise RuntimeError(
            f"create tuner rejected ({response.status_code}): {response.text[:1000]}"
        )
    tuner_id = response.json()["tuner_id"]
    params = f" params={trainer_params}" if trainer_params else ""
    print(
        f"[driver] created tuner {tuner_id} "
        f"({len(train_datum_ids)} train / {len(eval_datum_ids)} eval examples){params}"
    )
    return tuner_id


async def resolve_tuner(
    client: httpx.AsyncClient,
    *,
    tuner_id: str | None,
    name: str,
    recipe: Any,
    trainer: str,
    train_datum_ids: list[str],
    eval_datum_ids: list[str],
    trainer_params: dict[str, Any] | None = None,
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
        trainer_params=trainer_params,
    )


async def warn_unless_centering(client: httpx.AsyncClient, tuner_id: str) -> None:
    """The reward table only works at absolute scale (see README → Reward)."""
    response = await client.get(f"/tuners/{tuner_id}")
    response.raise_for_status()
    normalizer = (response.json().get("recipe") or {}).get("reward_normalizer")
    if normalizer != "centering":
        print(
            f"[driver] WARNING: reward_normalizer={normalizer!r}; the -2.0 clean-PR "
            "penalty is divided away unless it is 'centering'"
        )
    else:
        print("[driver] recipe uses reward_normalizer=centering")


async def dispense_run(client: httpx.AsyncClient, tuner_id: str) -> tuple[str, str]:
    waiting = False
    while True:
        response = await client.post(f"/tuners/{tuner_id}/runs")
        if response.status_code == 204:
            try:
                delay = max(0.0, float(response.headers.get("Retry-After", "1")))
            except ValueError:
                delay = 1.0
            if not waiting:
                print(
                    f"[driver] waiting for a runnable assignment; retrying in {delay:g}s",
                    flush=True,
                )
                waiting = True
            await asyncio.sleep(delay)
            continue
        response.raise_for_status()
        body = response.json()
        if waiting:
            print("[driver] assignment available; resuming runs", flush=True)
        return body["run_id"], body["datum_id"]


async def run_ollie_agent(
    *,
    base_url: str,
    tuner_id: str,
    run_id: str,
    task: ReviewTask,
    model: str,
    executor: str,
    max_steps: int,
    idle_timeout: float,
    max_context_tokens: int | None = None,
    timeout: float | None = None,
) -> tuple[tuple[reward.Finding, ...], tuple[str, ...]]:
    """Run one Ollie CLI agent and parse its `<review_findings>` payload."""
    # A trailing slash on --base-url would otherwise produce a `//tuners/...`
    # path. Railway does not normalize that path and returns 404 before the
    # request reaches ollie-rl's OpenAI-compatible endpoint.
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
        f"{task.snapshot}:/workspace:ro",
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
        *(
            ("--max-context-tokens", str(max_context_tokens))
            if max_context_tokens
            else ()
        ),
        build_prompt(task),
    )
    process = await asyncio.create_subprocess_exec(
        *command,
        env=environment,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
        limit=_STREAM_LIMIT,
    )
    assert process.stdout is not None and process.stderr is not None
    stderr_task = asyncio.create_task(process.stderr.read())

    # Stream stdout and reset a timer on every NDJSON event. A run that stops
    # producing events is wedged (stuck tool, or a reparented CLI that no longer
    # calls the model); kill its whole group. `--run-timeout`, when set, is an
    # optional wall-clock backstop on top of the idle guard.
    output = bytearray()
    loop = asyncio.get_running_loop()
    started = loop.time()
    kill_reason: str | None = None
    while True:
        budget = idle_timeout
        if timeout:
            remaining = timeout - (loop.time() - started)
            if remaining <= 0:
                kill_reason = f"Ollie agent timed out after {timeout:.0f}s"
                break
            budget = min(budget, remaining)
        try:
            line = await asyncio.wait_for(process.stdout.readline(), timeout=budget)
        except TimeoutError:
            if timeout and loop.time() - started >= timeout:
                kill_reason = f"Ollie agent timed out after {timeout:.0f}s"
            else:
                kill_reason = f"Ollie agent made no progress for {idle_timeout:.0f}s"
            break
        if not line:
            break
        output.extend(line)

    if kill_reason is not None:
        # npx spawns the real CLI as a grandchild; killing only the wrapper
        # leaves that process calling the run's endpoint forever. The child is a
        # group leader (start_new_session=True), so kill the whole group. A hung
        # CLI otherwise keeps this worker slot forever and its group can never
        # complete, so the tuner stops dispensing; the server's run lease then
        # expires on its own.
        _kill_process_group(process)
    try:
        await asyncio.wait_for(process.wait(), timeout=30)
    except TimeoutError:
        _kill_process_group(process)
        await process.wait()
    try:
        stderr = await asyncio.wait_for(stderr_task, timeout=5)
    except TimeoutError:
        stderr_task.cancel()
        stderr = b""
    if kill_reason is not None:
        raise RuntimeError(kill_reason) from None

    output = bytes(output).decode(errors="replace")
    findings, errors = reward.parse_findings(output)
    if process.returncode != 0:
        details = list(errors[-1:])
        if stderr_detail := stderr.decode(errors="replace").strip():
            details.append(stderr_detail)
        if not details and output.strip():
            details.append(output.strip())
        detail = " | ".join(details) if details else "no error output"
        raise RuntimeError(
            f"Ollie agent exited with status {process.returncode}: {detail[-1000:]}"
        )
    return findings, errors


async def submit_reward(
    client: httpx.AsyncClient, tuner_id: str, run_id: str, reward_value: float
) -> None:
    response = await client.put(
        f"/tuners/{tuner_id}/runs/{run_id}/reward",
        json={"reward": reward_value},
    )
    if response.is_error:
        # 403 means the lease expired, the reward was already set, or the run
        # recorded no completion; the body says which.
        raise RuntimeError(
            f"reward rejected ({response.status_code}): {response.text[:300]}"
        )


async def list_runs(client: httpx.AsyncClient, tuner_id: str) -> list[dict[str, Any]]:
    """Every run under a tuner, following `next_cursor` pages of 500."""
    runs: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        params: dict[str, Any] = {"limit": 500}
        if cursor:
            params["cursor"] = cursor
        response = await client.get(f"/tuners/{tuner_id}/runs", params=params)
        response.raise_for_status()
        body = response.json()
        runs.extend(body.get("runs") or [])
        cursor = body.get("next_cursor")
        if not cursor:
            return runs


async def salvage_run_findings(
    client: httpx.AsyncClient, tuner_id: str, run_id: str
) -> tuple[tuple[reward.Finding, ...], tuple[str, ...]]:
    """Recover the last `<review_findings>` from a run's recorded completions.

    The CLI's model turns are persisted server-side, so a run whose process
    died can still be scored. Raises when the transcript cannot be fetched (so
    the caller can leave the run for a later retry); a transcript that holds no
    completions yields empty findings, and a malformed `<review_findings>` is
    reported by `reward.parse_findings`.
    """
    try:
        response = await client.get(
            f"/tuners/{tuner_id}/runs/{run_id}",
            params={"include_payloads": "true"},
        )
        response.raise_for_status()
        completions = response.json().get("completions") or []
        # One NDJSON line per completion so `parse_findings` sees the whole
        # transcript and picks the agent's final `<review_findings>`.
        lines: list[str] = []
        for completion in completions:
            choices = (completion.get("response") or {}).get("choices") or []
            message = choices[0].get("message") if choices else None
            content = (message or {}).get("content")
            if content:
                lines.append(json.dumps({"content": content}))
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"could not read run completions: {error}") from error

    if not lines:
        return (), ()
    return reward.parse_findings("\n".join(lines))


async def salvage_run(
    client: httpx.AsyncClient,
    tuner_id: str,
    task: ReviewTask,
    run_id: str,
    judge: reward.Judge,
) -> tuple[tuple[reward.Finding, ...], reward.RewardResult]:
    """Score and record a reward for a run from its recorded completions.

    The reward endpoint is write-once and the lease may have expired, so a
    rejected `PUT` (403) means this salvage is over, not that the driver should
    stop. A readable transcript with no (or malformed) `<review_findings>` is
    scored as produced, exactly like the normal path, so a killed or hung run
    earns its miss instead of lapsing into `expired`.
    """
    findings, _errors = await salvage_run_findings(client, tuner_id, run_id)
    result = await reward.score_example(task.example, findings, judge)
    await submit_reward(client, tuner_id, run_id, result.total)
    return findings, result


async def execute_run(
    number: int,
    client: httpx.AsyncClient,
    tuner_id: str,
    task: ReviewTask,
    run_id: str,
    args: argparse.Namespace,
    judge: reward.Judge,
    stats: dict[str, SplitStats],
) -> None:
    print(
        f"[driver] run {number:04d} {task.split}/{task.kind} "
        f"task={task.example_id} run={run_id} started",
        flush=True,
    )
    try:
        findings, parse_errors = await run_ollie_agent(
            base_url=args.base_url,
            tuner_id=tuner_id,
            run_id=run_id,
            task=task,
            model=args.agent_model,
            executor=args.ollie_executor,
            max_steps=args.ollie_max_steps,
            idle_timeout=args.run_idle_timeout,
            max_context_tokens=args.ollie_max_context_tokens,
            timeout=args.run_timeout or None,
        )
        result = await reward.score_example(task.example, findings, judge)
        await submit_reward(client, tuner_id, run_id, result.total)
    except Exception as error:  # noqa: BLE001 - one failed rollout must not stop peers
        # The CLI records every model turn before it dies, so a run that already
        # emitted `<review_findings>` can still be scored from its transcript
        # instead of squatting its group until the server lease expires.
        try:
            findings, result = await salvage_run(client, tuner_id, task, run_id, judge)
        except Exception:  # noqa: BLE001 - salvage is best-effort
            print(f"[driver] run {number:04d} {task.example_id} failed: {error}")
            return
        matched = (
            f"matched={result.matched_severity}" if result.matched else "matched=no"
        )
        print(
            f"[driver] run {number:04d} {task.example_id} salvaged "
            f"reward={result.total:.2f} {matched} findings={len(findings)} "
            f"| {error}"
        )
        return

    split_stats = stats[task.split]
    split_stats.record(task.kind, result)
    matched = f"matched={result.matched_severity}" if result.matched else "matched=no"
    print(
        f"[driver] run {number:04d} {task.split}/{task.kind} "
        f"task={task.example_id} reward={result.total:+.2f} {matched} "
        f"findings={len(findings)} judge_calls={result.judge_calls} "
        f"| {task.split} {split_stats.summary()}"
    )
    for error in (*parse_errors, *result.errors):
        print(f"[driver] run {number:04d} note: {error[-500:]}")


async def salvage_orphaned_runs(
    client: httpx.AsyncClient,
    tuner_id: str,
    tasks: dict[str, ReviewTask],
    judge: reward.Judge,
    min_age: float,
) -> None:
    """Reward runs a killed predecessor driver left `in_flight`.

    A driver that dies mid-flight leaves its Ollie CLIs running and its runs
    unrewarded; each one blocks its datum's group until the server lease
    expires, stalling the tuner. Salvage them the same way a failed rollout is
    salvaged. Best-effort: a failed sweep must not abort startup.
    """
    try:
        runs = await list_runs(client, tuner_id)
    except Exception as error:  # noqa: BLE001 - a failed sweep must not abort startup
        print(f"[driver] salvage sweep skipped: {error}")
        return

    now = datetime.now(timezone.utc)
    orphans: list[dict[str, Any]] = []
    for run in runs:
        if run.get("status") != "in_flight" or run.get("reward") is not None:
            continue
        if not run.get("completion_count"):
            continue
        created_at = run.get("created_at")
        if not isinstance(created_at, str):
            continue
        try:
            created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except ValueError:
            continue
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        if (now - created).total_seconds() > min_age:
            orphans.append(run)

    recovered = 0
    for run in orphans:
        run_id = run["run_id"]
        # `datum_id` is the prepared example id, the same key `load_tasks` uses.
        task = tasks.get(run.get("datum_id"))
        if task is None:
            print(
                f"[driver] salvage sweep: run {run_id} has no loaded example "
                f"for datum {run.get('datum_id')}"
            )
            continue
        try:
            findings, result = await salvage_run(client, tuner_id, task, run_id, judge)
        except Exception as error:  # noqa: BLE001 - one bad run must not stop the sweep
            print(f"[driver] salvage sweep: run {run_id} skipped: {error}")
            continue
        recovered += 1
        print(
            f"[driver] salvage sweep: run {run_id} {task.example_id} "
            f"recovered reward={result.total:.2f} findings={len(findings)}"
        )
    print(f"[driver] salvage sweep: {recovered} run(s) recovered")


def load_example_order(data_dir: Path) -> dict[str, list[str]]:
    """Prepared per-split example order from ``splits.json``, when present.

    Datasets prepared before the order was recorded omit ``order``; callers
    then fall back to sorted ids.
    """
    path = data_dir / "splits.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError, ValueError:
        return {}
    order = payload.get("order")
    if not isinstance(order, dict):
        return {}
    return {
        split: [str(example_id) for example_id in ids]
        for split, ids in order.items()
        if isinstance(ids, list)
    }


def order_datum_ids(available: Iterable[str], order: list[str] | None) -> list[str]:
    """Apply a prepared order, appending any ids it does not mention.

    The split's membership is authoritative; ``splits.json`` sets the sequence.
    Ids missing from the prepared order (a stale or partial file) are appended
    in sorted order so nothing is silently dropped.
    """
    remaining = set(available)
    ordered: list[str] = []
    for example_id in order or []:
        if example_id in remaining:
            ordered.append(example_id)
            remaining.discard(example_id)
    ordered.extend(sorted(remaining))
    return ordered


async def async_main(args: argparse.Namespace) -> int:
    data_dir = args.data_dir.resolve()
    examples = reward.load_examples(data_dir)
    tasks = load_tasks(data_dir, examples, args.max_diff_chars)
    order = load_example_order(data_dir)
    train_datum_ids = order_datum_ids(
        (example_id for example_id, task in tasks.items() if task.split == "train"),
        order.get("train"),
    )
    eval_datum_ids = order_datum_ids(
        (example_id for example_id, task in tasks.items() if task.split == "val"),
        order.get("val"),
    )
    if not train_datum_ids:
        raise SystemExit("no train examples; re-run prepare_data.py")

    def count(ids: list[str], kind: str) -> int:
        return sum(1 for datum_id in ids if tasks[datum_id].kind == kind)

    source = "splits.json order" if order else "sorted ids"
    print(
        f"[driver] examples ({source}): train {len(train_datum_ids)} "
        f"({count(train_datum_ids, 'bug')} bug / {count(train_datum_ids, 'clean')} clean), "
        f"val {len(eval_datum_ids)} "
        f"({count(eval_datum_ids, 'bug')} bug / {count(eval_datum_ids, 'clean')} clean)"
    )

    # MSRL config fields are passed through `trainer_params`; other trainers
    # ignore them. Only include values the caller set explicitly so the server
    # defaults apply otherwise.
    params: dict[str, Any] = {}
    if args.base_model:
        params["base_model"] = args.base_model
    if args.adapter_size:
        params["adapter_size"] = args.adapter_size
    trainer_params: dict[str, Any] | None = params or None

    timeout = httpx.Timeout(connect=10.0, read=90.0, write=30.0, pool=30.0)
    stats = {"train": SplitStats(), "val": SplitStats()}
    async with (
        reward.Judge(
            model=args.judge_model,
            base_url=args.judge_base_url,
            api_key=args.judge_api_key,
            seed=args.judge_seed if args.judge_seed >= 0 else None,
            votes=args.judge_votes,
        ) as judge,
        httpx.AsyncClient(base_url=args.base_url, timeout=timeout) as client,
    ):
        tuner_id = await resolve_tuner(
            client,
            tuner_id=args.tuner_id,
            name=args.name,
            recipe=args.recipe,
            trainer=args.trainer,
            train_datum_ids=train_datum_ids,
            eval_datum_ids=eval_datum_ids,
            trainer_params=trainer_params,
        )
        await warn_unless_centering(client, tuner_id)
        # Clear runs a killed predecessor driver abandoned before this driver
        # starts competing with them for group slots.
        await salvage_orphaned_runs(
            client, tuner_id, tasks, judge, args.salvage_min_age
        )

        slots = asyncio.Semaphore(args.concurrency)

        async def execute_and_release(
            number: int, task: ReviewTask, run_id: str
        ) -> None:
            try:
                await execute_run(
                    number, client, tuner_id, task, run_id, args, judge, stats
                )
            finally:
                slots.release()

        async with asyncio.TaskGroup() as group:
            for number in range(args.runs):
                await slots.acquire()
                try:
                    run_id, datum_id = await dispense_run(client, tuner_id)
                    task = tasks[datum_id]
                except BaseException:
                    slots.release()
                    raise
                group.create_task(execute_and_release(number, task, run_id))

    for split in ("train", "val"):
        print(f"[driver] {split}: {stats[split].summary()}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--recipe",
        type=parse_recipe,
        default=DEFAULT_RECIPE,
        help="Named recipe or an inline JSON object of recipe overrides.",
    )
    parser.add_argument("--trainer", default=DEFAULT_TRAINER)
    parser.add_argument(
        "--base-model",
        default=None,
        help=(
            "MSRL only: tuning base model (e.g. gemini-3.5-flash-lite), sent as "
            "trainer_params.base_model."
        ),
    )
    parser.add_argument(
        "--adapter-size",
        default=None,
        choices=ADAPTER_SIZES,
        help=(
            "MSRL only: LoRA adapter size, sent as trainer_params.adapter_size. "
            "Server default is ADAPTER_SIZE_SIXTEEN; ADAPTER_SIZE_THIRTY_TWO is "
            "the largest the API accepts."
        ),
    )
    parser.add_argument("--name", default=DEFAULT_TUNER_NAME)
    parser.add_argument("--agent-model", default=DEFAULT_AGENT_MODEL)
    parser.add_argument(
        "--ollie-executor",
        choices=("none", "daytona", "vercel"),
        default="none",
        help="Sandbox executor; 'none' runs the sandbox locally.",
    )
    parser.add_argument("--ollie-max-steps", type=int, default=OLLIE_MAX_STEPS)
    parser.add_argument(
        "--ollie-max-context-tokens",
        type=int,
        default=None,
        help=(
            "Pass --max-context-tokens to the CLI: the agent wraps up and "
            "answers at 90%% of this budget, and the loop hard-stops at it. "
            "Unset disables the limit."
        ),
    )
    parser.add_argument(
        "--run-timeout",
        type=float,
        default=DEFAULT_RUN_TIMEOUT,
        help=(
            "Optional wall-clock backstop in seconds; 0 (default) disables it. "
            "Prefer --run-idle-timeout: a fixed deadline truncates slow runs."
        ),
    )
    parser.add_argument(
        "--run-idle-timeout",
        type=float,
        default=DEFAULT_RUN_IDLE_TIMEOUT,
        help=(
            "Kill a rollout after this many seconds with no new agent event "
            "(the hang guard); --ollie-max-steps bounds work."
        ),
    )
    parser.add_argument(
        "--salvage-min-age",
        type=float,
        default=DEFAULT_SALVAGE_MIN_AGE,
        help=(
            "Startup orphan sweep only touches in-flight runs older than this "
            "many seconds, so a predecessor's still-running CLI can finish."
        ),
    )
    parser.add_argument("--max-diff-chars", type=int, default=DEFAULT_MAX_DIFF_CHARS)
    parser.add_argument("--runs", type=int, default=2000)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--tuner-id", default=None)
    parser.add_argument("--judge-model", default=reward.DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-base-url", default=reward.DEFAULT_JUDGE_BASE_URL)
    parser.add_argument("--judge-api-key", default=None)
    parser.add_argument(
        "--judge-seed",
        type=int,
        default=reward.DEFAULT_JUDGE_SEED,
        help="Request seed for reproducible verdicts; negative disables it.",
    )
    parser.add_argument(
        "--judge-votes",
        type=int,
        default=1,
        help="Majority vote over N judge calls per pair (odd N is best).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.runs < 1:
        raise SystemExit("--runs must be at least 1")
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be at least 1")
    if args.ollie_max_steps < 1:
        raise SystemExit("--ollie-max-steps must be at least 1")
    if args.run_idle_timeout <= 0:
        raise SystemExit("--run-idle-timeout must be positive")
    if args.max_diff_chars < 1:
        raise SystemExit("--max-diff-chars must be at least 1")
    if args.judge_votes < 1:
        raise SystemExit("--judge-votes must be at least 1")
    try:
        return asyncio.run(async_main(args))
    except ValueError as error:
        raise SystemExit(f"[driver] {error}") from error


if __name__ == "__main__":
    raise SystemExit(main())
