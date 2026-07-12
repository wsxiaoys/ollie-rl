"""End-to-end RL training driver: CodeContests (Harbor tasks) + ollie-rl.

Each rollout runs as a containerized **Harbor trial**: a Terminus 2 agent is
dropped into a sandbox to solve one competitive-programming task, and Harbor's
verifier runs the task's unit tests to produce the reward. The driver just
orchestrates the loop — dispense a run from ollie-rl, run the trial, report the
reward — while ollie-rl handles the training.

The agent samples through ollie-rl's OpenAI-compatible endpoint. Completions are
attributed to the dispensed run via a path-addressed ``base_url``
(``http://<ollie-host>/tuners/<tuner-id>/runs/<run-id>/openai/v1``) so the ids
show up in the request line — which keeps per-run completions searchable in log
aggregators (e.g. Railway) — instead of travelling in ``X-Tuner-Id`` /
``X-Run-Id`` headers.

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
import random
from collections.abc import Awaitable, Callable
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

# Per-completion (per-turn) wall-clock cap forwarded to litellm as ``timeout``.
# NOTE: Harbor wraps each LLM call in a tenacity ``retry(stop_after_attempt(3))``,
# and a timeout is a retryable exception, so the effective time before a turn
# gives up is roughly ``3 * timeout`` plus backoff. Set this comfortably above
# the normal upper body of the latency distribution (observed p95 ~360s) so it
# only kills genuine hangs rather than slow-but-healthy turns. ``None`` disables.
DEFAULT_LLM_TIMEOUT_SEC: int | None = 300

# Daytona sandbox lifecycle backstop. Harbor's ``stop(delete=True)`` cleanup is
# skipped when the driver is killed abruptly, which leaks sandboxes because the
# defaults (``auto_stop_interval_mins=0``) disable auto-stop entirely. Enabling
# an inactivity auto-stop plus a post-stop auto-delete lets orphaned sandboxes
# self-reclaim even after a hard crash. Only applied when ``--environment daytona``.
#
# ``auto_stop_interval_mins`` is Daytona's *inactivity* timer: the sandbox stops
# once it has seen no SDK events for this many minutes. "Activity" means an SDK
# interaction with the sandbox -- command exec, ``send_keys``, file up/downloads,
# lifecycle state changes -- each of which stamps ``last_activity_at`` and resets
# the countdown. Crucially, it is a *per-gap* limit, not cumulative: it trips
# only when a single idle gap between two sandbox calls exceeds the interval.
# The dominant gap is the agent waiting on an LLM completion -- that wait happens
# outside the sandbox, emits no event, and so does not refresh the timer. A slow
# turn longer than the interval therefore lets the sandbox auto-stop mid-run,
# invalidating its bearer token and crashing the trial with a Daytona auth error
# (observed on trial ``code_contests-0006__7DbYZo5``: a ~17-min completion tripped
# the old 10-min value). Keep this comfortably above the worst-case single-turn
# LLM latency (``DEFAULT_LLM_TIMEOUT_SEC`` can be ~3x with retries) so only real
# leaks -- not slow-but-healthy turns -- get reclaimed.
DEFAULT_DAYTONA_AUTO_STOP_MINS = 30
DEFAULT_DAYTONA_AUTO_DELETE_MINS = 5

# Terminus 2 routes through litellm, which needs a provider prefix. ollie-rl
# exposes an OpenAI-compatible endpoint, so the model is addressed as
# ``openai/<name>``.
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
    train_datum_ids: list[str],
    eval_datum_ids: list[str],
) -> str:
    resp = await client.post(
        "/tuners",
        json={
            "name": name,
            "recipe": recipe,
            "trainer": trainer,
            "train_datum_ids": train_datum_ids,
            "eval_datum_ids": eval_datum_ids,
        },
    )
    resp.raise_for_status()
    tuner_id = resp.json()["tuner_id"]
    print(
        f"[driver] created tuner {tuner_id} "
        f"({len(train_datum_ids)} train / {len(eval_datum_ids)} eval tasks)"
    )
    return tuner_id


# Fraction of discovered tasks held out as the eval split (scored per
# checkpoint, never trained on). A fixed-seed shuffle picks the eval subset so
# the split is random (not correlated with datum-id ordering) yet deterministic
# across runs.
EVAL_FRACTION = 0.05
EVAL_SPLIT_SEED = 0xBADBEEF


def split_train_eval(datum_ids: list[str]) -> tuple[list[str], list[str]]:
    """Deterministically hold out a random fraction of datums for eval.

    Shuffles a copy with a fixed seed and takes the first
    `ceil(n * EVAL_FRACTION)` as eval, so the split is reproducible run-to-run
    while decoupled from datum-id ordering. Reserves at least one eval datum
    when there are >=2 tasks (so eval is exercised), and always keeps at least
    one training datum.
    """
    import math
    import random

    n = len(datum_ids)
    if n < 2:
        # Too few to hold any out; train on everything, eval disabled.
        return list(datum_ids), []
    eval_count = max(1, math.ceil(n * EVAL_FRACTION))
    eval_count = min(eval_count, n - 1)  # keep at least one training datum

    shuffled = list(datum_ids)
    random.Random(EVAL_SPLIT_SEED).shuffle(shuffled)
    eval_datum_ids = shuffled[:eval_count]
    train_datum_ids = shuffled[eval_count:]
    return train_datum_ids, eval_datum_ids


# --------------------------------------------------------------------------
# Control-plane transport resilience
# --------------------------------------------------------------------------
# The dispense/reward calls are short JSON requests, but the ollie-rl server can
# briefly stall — e.g. a pooled Postgres connection went stale during an idle
# lull (no `pool_pre_ping`), or a train step momentarily tied up the connection
# pool — which surfaces on the client as an `httpx.ReadTimeout`/transport error.
# Those are transient: retrying after a short backoff succeeds, whereas letting
# the exception escape unwinds `asyncio.gather` and kills the whole run. So wrap
# these calls in an exponential-backoff-with-jitter retry.
#
# The retry is intentionally *unbounded*: a stall can coincide with a long
# all-runs-dispensed lull (the server is simply busy training / has nothing to
# hand out), which is a normal steady state rather than a failure, so there's no
# attempt count at which giving up and crashing the driver would be correct. We
# just keep backing off (capped) until the server responds again.
CONTROL_PLANE_BACKOFF_BASE_SEC = 1.0
CONTROL_PLANE_BACKOFF_CAP_SEC = 30.0


async def _request_with_retry(
    describe: str,
    send: Callable[[], Awaitable[httpx.Response]],
) -> httpx.Response:
    """Send one HTTP request, retrying transient timeouts / transport errors.

    ``send`` performs a single attempt and returns its response. On an
    ``httpx.TimeoutException`` or ``httpx.TransportError`` (connection reset,
    stale pooled connection, etc.) we back off — exponential with full jitter,
    capped at ``CONTROL_PLANE_BACKOFF_CAP_SEC`` — and retry indefinitely, since a
    stall may just be a busy/all-dispensed server that will recover. HTTP status
    errors are *not* retried here (the caller inspects the response and decides).
    """
    attempt = 0
    while True:
        try:
            return await send()
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            ceiling = min(
                CONTROL_PLANE_BACKOFF_CAP_SEC,
                CONTROL_PLANE_BACKOFF_BASE_SEC * (2**attempt),
            )
            delay = random.uniform(0, ceiling)  # full jitter
            print(
                f"[driver] {describe}: transient transport error "
                f"({type(exc).__name__}); retrying in {delay:.1f}s"
            )
            await asyncio.sleep(delay)
            attempt += 1


async def dispense_run(
    client: httpx.AsyncClient,
    tuner_id: str,
) -> tuple[str, str] | None:
    """Return ``(run_id, datum_id)`` or ``None`` when the trainer is busy (204).

    Datum quarantine (unhealthy-finish-rate / success-rate filtering) is now
    configured on the tuner's recipe (``max_unhealthy_finish_ratio`` /
    ``max_succeed_ratio``), not per request, so this call takes no quarantine
    params.

    Transient transport failures (server briefly unresponsive under load) are
    retried with backoff rather than crashing the driver; see
    :func:`_request_with_retry`.
    """
    resp = await _request_with_retry(
        f"dispense (tuner {tuner_id})",
        lambda: client.post(f"/tuners/{tuner_id}/runs"),
    )
    if resp.status_code == 204:
        return None
    resp.raise_for_status()
    body = resp.json()
    return body["run_id"], body["datum_id"]


async def submit_reward(
    client: httpx.AsyncClient, tuner_id: str, run_id: str, reward: float
) -> bool:
    """Report a run's reward. Returns ``False`` when the server rejects it.

    A ``409 Conflict`` is expected and non-fatal: it happens when the run has
    expired, already had its reward set (e.g. a malformed example the server
    already finalized), or produced no chat completions at all (a crashed trial
    that never sampled — a reward for it carries no training signal, so the
    server refuses it). We swallow it so the driver can keep going.

    Transient transport failures are retried with backoff (see
    :func:`_request_with_retry`) so a momentary server stall doesn't crash the
    driver.
    """
    resp = await _request_with_retry(
        f"reward (run {run_id})",
        lambda: client.put(
            f"/tuners/{tuner_id}/runs/{run_id}/reward", json={"reward": reward}
        ),
    )
    if resp.status_code == 409:
        print(
            f"[driver] run {run_id} reward rejected (409 Conflict; likely a "
            f"malformed example): {resp.json().get('detail', resp.text)}"
        )
        return False
    resp.raise_for_status()
    return True


# --------------------------------------------------------------------------
# The rollout: one Harbor trial == one ollie-rl Run
# --------------------------------------------------------------------------
def run_openai_base_url(base: str, tuner_id: str, run_id: str) -> str:
    """The per-run OpenAI-compatible endpoint (ids travel in the URL path).

    Encoding the tuner/run ids in the path (rather than ``X-Tuner-Id`` /
    ``X-Run-Id`` headers) keeps them in the request line, which makes each run's
    completions easy to search in log aggregators (e.g. Railway).
    """
    return f"{base}/tuners/{tuner_id}/runs/{run_id}/openai/v1"


async def run_rollout(
    *,
    base: str,
    tuner_id: str,
    run_id: str,
    datum_id: str,
    environment: str,
    agent_timeout_multiplier: float | None = None,
) -> float | None:
    """Execute one containerized Harbor trial and return its reward.

    The agent samples through this run's path-addressed ``/openai/v1`` endpoint
    so ollie-rl records completions under the run without needing per-request
    headers. Harbor's verifier then runs the task's tests and produces the
    reward.

    Returns the graded scalar reward, or ``None`` when the run carries no policy
    signal we want to train on. ``None`` is *not* a zero reward — it means the
    caller should skip submission and let the run's lease expire. This covers:

    * the verifier never ran (e.g. the trial crashed or was cancelled before
      grading), and
    * the agent hit its ``timeout_sec`` budget. Harbor still grades a timed-out
      agent, but that reward reflects a truncated (often empty) rollout rather
      than a policy decision to stop, so we deliberately drop it instead of
      training on a spurious ``0.0``.
    """
    # Per-turn timeout: forwarded to litellm's ``acompletion(timeout=...)`` via
    # the agent's ``llm_kwargs``. Caps how long a single completion can hang.
    # The tuner/run ids travel in the path-addressed ``api_base`` below, so no
    # ``X-Tuner-Id`` / ``X-Run-Id`` headers are needed.
    llm_kwargs: dict = {
        "api_key": "ollie",
    }
    if DEFAULT_LLM_TIMEOUT_SEC is not None:
        llm_kwargs["timeout"] = DEFAULT_LLM_TIMEOUT_SEC

    # Daytona-only lifecycle backstop so orphaned sandboxes self-reclaim if the
    # driver dies before Harbor's ``stop(delete=True)`` cleanup runs. These
    # kwargs are ignored by / invalid for other backends, so only set them for
    # the daytona environment.
    env_kwargs: dict = {}
    if environment.lower() == "daytona":
        env_kwargs = {
            "auto_stop_interval_mins": DEFAULT_DAYTONA_AUTO_STOP_MINS,
            "auto_delete_interval_mins": DEFAULT_DAYTONA_AUTO_DELETE_MINS,
        }

    config = TrialConfig(
        task=TaskConfig(path=TASKS_DIR / datum_id),
        trials_dir=TRIALS_DIR,
        agent=AgentConfig(
            name=AgentName.TERMINUS_2.value,
            model_name=AGENT_MODEL_NAME,
            # Point the agent at this run's path-addressed OpenAI-compatible
            # endpoint so ollie-rl attributes every completion to this run
            # (the tuner/run ids are baked into the URL, not sent as headers).
            kwargs={
                "api_base": run_openai_base_url(base, tuner_id, run_id),
                "llm_kwargs": llm_kwargs,
            },
        ),
        environment=EnvironmentConfig(
            type=EnvironmentType[environment.upper()],
            kwargs=env_kwargs,
        ),
        # Scale the agent phase budget. Harbor computes the effective timeout as
        # ``task.agent.timeout_sec * agent_timeout_multiplier`` (falling back to
        # the task default when the multiplier is None), so this stretches the
        # per-task budget without hardcoding an absolute seconds value.
        agent_timeout_multiplier=agent_timeout_multiplier,
    )

    # `Trial` is abstract; `Trial.create()` loads the task and returns the right
    # concrete trial (single- vs multi-step). `run()` returns a `TrialResult`.
    trial = await Trial.create(config)
    trial_result = await trial.run()
    if agent_timed_out(trial_result):
        # Harbor still grades a timed-out agent, but a timeout means the rollout
        # was truncated (often before writing any solution), so the graded
        # reward isn't a real policy signal. Skip it and let the lease expire.
        graded = getattr(trial_result, "verifier_result", None) is not None
        print(
            f"[driver] run {run_id} hit the agent timeout "
            f"(graded={graded}); skipping reward"
        )
        return None
    return extract_reward(trial_result)


def agent_timed_out(trial_result) -> bool:
    """Whether Harbor aborted the agent phase on its ``timeout_sec`` budget.

    Read straight off the structured ``TrialResult`` — ``exception_info.
    exception_type`` is ``"AgentTimeoutError"`` — so there's no need to scrape
    the ``exception.txt`` artifact. Note Harbor still runs the verifier after a
    timeout, so this can be ``True`` alongside a real ``verifier_result``.
    """
    info = getattr(trial_result, "exception_info", None)
    return bool(info and info.exception_type == "AgentTimeoutError")


def extract_reward(trial_result) -> float | None:
    """Pull the scalar reward out of a single Harbor trial result.

    ``VerifierResult.rewards`` is a ``dict[str, float | int] | None``. We prefer
    the ``"reward"`` key, then fall back to the sole value when the verifier
    emitted exactly one metric.

    Returns ``None`` when the verifier produced no graded reward (no
    ``verifier_result``/``rewards``, or an ambiguous multi-metric result we
    can't interpret). A ``None`` means "not graded" and is deliberately
    distinct from a genuine ``0.0`` so the driver can skip submitting a
    fabricated reward for un-graded runs.
    """
    verifier_result = getattr(trial_result, "verifier_result", None)
    rewards = getattr(verifier_result, "rewards", None) if verifier_result else None
    if not rewards:
        return None
    if "reward" in rewards:
        return float(rewards["reward"])
    if len(rewards) == 1:
        return float(next(iter(rewards.values())))
    return None


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
                agent_timeout_multiplier=args.agent_timeout_multiplier,
            )
        except Exception as exc:
            # The trial crashed before the verifier could grade it (e.g. the
            # inference endpoint dropped mid-run). There is no graded outcome to
            # attribute to the policy, so DON'T fabricate a 0.0 reward — skip
            # submission and let the run's lease expire for a clean re-dispense.
            print(
                f"[driver] run {run:04d} trial crashed ({datum_id}); "
                f"skipping reward: {exc}"
            )
            continue

        # A run with no policy signal (verifier never ran, or the agent hit its
        # timeout budget so the rollout was truncated) comes back as None. Skip
        # it rather than submit a spurious 0.0 and let the lease expire.
        if reward is None:
            print(
                f"[driver] run {run:04d} task={datum_id} no policy signal "
                f"(not graded or agent timed out); skipping reward"
            )
            continue

        # Phase 3: report the reward; the server groups/advantages/trains.
        # A rejected reward (409) is expected for malformed examples the server
        # already finalized, so skip recording stats for it.
        if not await submit_reward(client, tuner_id, run_id, reward):
            continue

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
        "--agent-timeout-multiplier",
        type=float,
        default=None,
        help=(
            "Scale each task's agent timeout budget. Harbor multiplies the "
            "task's [agent] timeout_sec by this factor (e.g. 2.0 doubles it). "
            "Defaults to the task-defined timeout when omitted."
        ),
    )
    # Datum quarantine (length-rate / success-rate filtering) is configured on
    # the tuner's recipe now, not via CLI flags / query params.
    parser.add_argument(
        "--tuner-id",
        default=None,
        help="Reuse an existing tuner instead of creating a new one.",
    )
    args = parser.parse_args()

    TRIALS_DIR.mkdir(parents=True, exist_ok=True)
    datum_ids = discover_datum_ids()

    # Granular timeouts (rather than a single 30s that collides with the
    # server's default SQLAlchemy `pool_timeout=30`): fail fast on a dead
    # connection, but give a healthy-but-busy server room to respond on read
    # before `_request_with_retry` backs off and retries.
    timeout = httpx.Timeout(connect=10.0, read=90.0, write=30.0, pool=30.0)
    async with httpx.AsyncClient(base_url=args.base_url, timeout=timeout) as client:
        tuner_id = args.tuner_id
        if tuner_id:
            # Confirm the explicitly provided tuner exists and fetch its name.
            try:
                resp = await client.get(f"/tuners/{tuner_id}")
                resp.raise_for_status()
                details = resp.json()
                print(
                    f"[driver] adapting to existing tuner {tuner_id} "
                    f"(name={details['name']!r}, recipe={details['recipe']['name']!r})"
                )
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404:
                    print(f"[driver] error: tuner {tuner_id} not found")
                    return 1
                raise
        else:
            # Try to find an existing tuner with the matching name.
            try:
                resp = await client.get("/tuners")
                resp.raise_for_status()
                tuners = resp.json().get("tuners", [])
                for t in tuners:
                    if t["name"] == args.name:
                        tuner_id = t["tuner_id"]
                        print(
                            f"[driver] adapting to existing tuner {tuner_id} (name={args.name!r})"
                        )
                        break
            except Exception as exc:
                print(
                    f"[driver] warning: could not list tuners to check for existing name: {exc}"
                )

        if not tuner_id:
            train_datum_ids, eval_datum_ids = split_train_eval(datum_ids)
            tuner_id = await create_tuner(
                client,
                name=args.name,
                recipe=args.recipe,
                trainer=args.trainer,
                train_datum_ids=train_datum_ids,
                eval_datum_ids=eval_datum_ids,
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
