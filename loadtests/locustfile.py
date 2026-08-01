"""Synthetic closed-loop rollout load test for Ollie RL.

The workload exercises the public batch-dispense -> completion -> reward flow
with the built-in fake trainer, avoiding Gemini quota and Harbor sandbox costs.
Each Locust process maintains a shared assignment queue and has at most one
batch request in flight. Configure its size with
``OLLIE_LOADTEST_DISPENSE_BATCH_SIZE`` (default and maximum: 1024).

By default, one tuner is created per Locust process. Set ``OLLIE_TUNER_ID`` to
make all processes in a distributed run target the same pre-created tuner.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
import zlib
from typing import Any

import gevent
from gevent.lock import Semaphore
from gevent.queue import Empty, Queue
from locust import FastHttpUser, constant, task

logger = logging.getLogger(__name__)

_TUNER_LOCK = Semaphore()
_TUNER_ID: str | None = os.getenv("OLLIE_TUNER_ID")
_DISPENSE_LOCK = Semaphore()
_DISPENSE_QUEUE: Queue[tuple[str, str]] = Queue()
_NEXT_DISPENSE_AT = 0.0


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


DATUM_COUNT = _positive_int("OLLIE_LOADTEST_DATUMS", 256)
TURNS_PER_RUN = _positive_int("OLLIE_LOADTEST_TURNS", 1)
DISPENSE_BATCH_SIZE = _positive_int("OLLIE_LOADTEST_DISPENSE_BATCH_SIZE", 1024)
if DISPENSE_BATCH_SIZE > 1024:
    raise ValueError("OLLIE_LOADTEST_DISPENSE_BATCH_SIZE must be at most 1024")
RECIPE = os.getenv("OLLIE_LOADTEST_RECIPE", "grpo_4x8")
THINK_TIME_SECONDS = max(
    0.0, float(os.getenv("OLLIE_LOADTEST_THINK_TIME_MS", "0")) / 1000
)


def _response_error(response: Any) -> str:
    try:
        return str(response.json())
    except TypeError, ValueError:
        return (response.text or "empty response")[:500]


def _retry_after(response: Any) -> float:
    try:
        return max(0.0, float(response.headers.get("Retry-After", "1")))
    except ValueError:
        return 1.0


def _reward_for(run_id: str) -> float:
    """Return deterministic, non-degenerate rewards."""
    return 0.5 if zlib.crc32(run_id.encode()) % 2 else 0.0


class RolloutUser(FastHttpUser):
    """A worker that owns at most one rollout at a time."""

    wait_time = constant(0)
    connection_timeout = 10.0
    network_timeout = 60.0

    def on_start(self) -> None:
        self.tuner_id = self._get_or_create_tuner()

    def _get_or_create_tuner(self) -> str:
        global _TUNER_ID

        if _TUNER_ID is not None:
            return _TUNER_ID

        with _TUNER_LOCK:
            if _TUNER_ID is not None:
                return _TUNER_ID

            payload = {
                "name": f"locust-fake-{uuid.uuid4().hex[:12]}",
                "recipe": RECIPE,
                "trainer": "fake",
                "train_datum_ids": [
                    f"loadtest-datum-{index}" for index in range(DATUM_COUNT)
                ],
                "eval_datum_ids": [],
            }
            with self.client.post(
                "/tuners",
                json=payload,
                name="/tuners [create fake]",
                catch_response=True,
            ) as response:
                if response.status_code != 200:
                    response.failure(
                        f"failed to create fake tuner: {_response_error(response)}"
                    )
                    raise RuntimeError("Unable to create the load-test tuner")

                try:
                    tuner_id = response.json()["tuner_id"]
                except (KeyError, TypeError, ValueError) as error:
                    response.failure(f"invalid tuner response: {error}")
                    raise RuntimeError("Invalid tuner creation response") from error

                if not isinstance(tuner_id, str) or not tuner_id:
                    response.failure("tuner_id is missing or invalid")
                    raise RuntimeError("Invalid tuner ID")

                response.success()
                _TUNER_ID = tuner_id
                logger.info("Created fake load-test tuner %s", tuner_id)
                return tuner_id

    @task
    def rollout(self) -> None:
        rollout_started = time.perf_counter()
        dispensed = self._dispense_run()
        if dispensed is None:
            return

        run_id, datum_id = dispensed
        if not self._complete_run(run_id, datum_id):
            return
        if not self._submit_reward(run_id):
            return

        self.environment.events.request.fire(
            request_type="ROLLOUT",
            name="dispense + completion + reward",
            response_time=(time.perf_counter() - rollout_started) * 1000,
            response_length=0,
            response=None,
            context={},
            exception=None,
        )

    def _dispense_run(self) -> tuple[str, str] | None:
        """Take one assignment from the process-wide batch dispense queue."""
        global _NEXT_DISPENSE_AT

        while True:
            try:
                return _DISPENSE_QUEUE.get_nowait()
            except Empty:
                pass

            delay = 0.0
            with _DISPENSE_LOCK:
                # Another greenlet may have refilled the queue while this one
                # waited for the single process-level batch requester.
                try:
                    return _DISPENSE_QUEUE.get_nowait()
                except Empty:
                    pass

                now = time.monotonic()
                if now < _NEXT_DISPENSE_AT:
                    delay = _NEXT_DISPENSE_AT - now
                else:
                    with self.client.post(
                        f"/tuners/{self.tuner_id}/runs",
                        params={"batch_size": DISPENSE_BATCH_SIZE},
                        name="/tuners/[id]/runs [batch dispense]",
                        catch_response=True,
                    ) as response:
                        if response.status_code == 204:
                            response.success()
                            delay = _retry_after(response)
                            _NEXT_DISPENSE_AT = time.monotonic() + delay
                        elif response.status_code != 200:
                            response.failure(
                                f"batch dispense failed: {_response_error(response)}"
                            )
                            return None
                        else:
                            try:
                                batch = response.json()
                                if not isinstance(batch, list) or not batch:
                                    raise ValueError(
                                        "batch response must be a non-empty list"
                                    )
                                assignments = [
                                    (item["run_id"], item["datum_id"]) for item in batch
                                ]
                            except (KeyError, TypeError, ValueError) as error:
                                response.failure(
                                    f"invalid batch dispense response: {error}"
                                )
                                return None

                            if any(
                                not isinstance(run_id, str)
                                or not isinstance(datum_id, str)
                                for run_id, datum_id in assignments
                            ):
                                response.failure("run_id or datum_id is invalid")
                                return None

                            if len({run_id for run_id, _ in assignments}) != len(
                                assignments
                            ):
                                response.failure(
                                    "batch dispense returned duplicate run IDs"
                                )
                                return None

                            for assignment in assignments:
                                _DISPENSE_QUEUE.put_nowait(assignment)
                            _NEXT_DISPENSE_AT = 0.0
                            response.success()

            if delay:
                gevent.sleep(delay)

    def _complete_run(self, run_id: str, datum_id: str) -> bool:
        messages: list[dict[str, str]] = []
        for turn in range(TURNS_PER_RUN):
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Synthetic rollout datum={datum_id} run={run_id} turn={turn}"
                    ),
                }
            )
            with self.client.post(
                f"/tuners/{self.tuner_id}/runs/{run_id}/openai/v1/chat/completions",
                json={"model": "fake-model", "messages": messages},
                name=("/tuners/[id]/runs/[id]/openai/v1/chat/completions [complete]"),
                catch_response=True,
            ) as response:
                if response.status_code != 200:
                    response.failure(f"completion failed: {_response_error(response)}")
                    return False

                try:
                    completion = response.json()
                    choice = completion["choices"][0]
                    content = choice["message"]["content"]
                except (IndexError, KeyError, TypeError, ValueError) as error:
                    response.failure(f"invalid completion response: {error}")
                    return False

                if not isinstance(content, str) or not content:
                    response.failure("completion content is empty or invalid")
                    return False

                response.success()
                messages.append({"role": "assistant", "content": content})

            if THINK_TIME_SECONDS:
                gevent.sleep(THINK_TIME_SECONDS)

        return True

    def _submit_reward(self, run_id: str) -> bool:
        reward = _reward_for(run_id)
        with self.client.put(
            f"/tuners/{self.tuner_id}/runs/{run_id}/reward",
            json={"reward": reward},
            name="/tuners/[id]/runs/[id]/reward [submit]",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"reward failed: {_response_error(response)}")
                return False

            try:
                data = response.json()
            except ValueError as error:
                response.failure(f"invalid reward response: {error}")
                return False

            if data.get("run_id") != run_id or data.get("reward") != reward:
                response.failure("reward response does not match the request")
                return False

            response.success()
            return True
