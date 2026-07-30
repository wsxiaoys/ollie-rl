"""One-shot concurrent Gemini MSRL completions against a single Ollie run.

This is a diagnostic workload, not a normal rollout workload. Every Locust user
sends exactly one request with a unique prompt to the same run, forcing distinct
Gemini sampling operations instead of exercising Ollie's idempotent replay.

Required environment variables:

    OLLIE_TUNER_ID   Existing Gemini MSRL tuner ID.
    OLLIE_RUN_ID     Existing, unexpired, unrewarded run ID.

Optional environment variables:

    OLLIE_EXPERIMENT_ID    Marker included in the request header and prompt.
    OLLIE_MODEL            Requested model name (default: gemini-3.5-flash).
    OLLIE_LOADTEST_PROMPT  Prompt prefix for every request.

Example (1024 concurrent requests):

    OLLIE_TUNER_ID=tuner_... OLLIE_RUN_ID=run_... \
      uv run locust -f loadtests/gemini_same_run_locustfile.py \
      --headless --host https://your-ollie-service.example \
      --users 1024 --spawn-rate 1024 --run-time 5m

The script deliberately does not submit a reward. Concurrent unique prompts on
one run fork its trajectory and should only be used for controlled diagnostics.
"""

from __future__ import annotations

import os
import uuid

from locust import FastHttpUser, constant, task
from locust.exception import StopUser


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} must be set")
    return value


TUNER_ID = _required_env("OLLIE_TUNER_ID")
RUN_ID = _required_env("OLLIE_RUN_ID")
MODEL = os.getenv("OLLIE_MODEL", "gemini-3.5-flash")
EXPERIMENT_ID = os.getenv("OLLIE_EXPERIMENT_ID", f"gemini-load-{uuid.uuid4().hex[:12]}")
PROMPT_PREFIX = os.getenv(
    "OLLIE_LOADTEST_PROMPT",
    "Gemini MSRL concurrency diagnostic; reply with exactly OK",
)


class SameRunCompletionUser(FastHttpUser):
    """Send one uniquely keyed completion and then stop this Locust user."""

    wait_time = constant(0)
    connection_timeout = 30.0
    network_timeout = 240.0

    @task
    def complete_once(self) -> None:
        request_id = uuid.uuid4().hex
        self.client.post(
            f"/tuners/{TUNER_ID}/runs/{RUN_ID}/openai/v1/chat/completions",
            headers={"X-Ollie-Loadtest-Id": EXPERIMENT_ID},
            json={
                "model": MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"{PROMPT_PREFIX}\nExperiment: {EXPERIMENT_ID}"
                            f"\nDiagnostic request: {request_id}"
                        ),
                    }
                ],
            },
            name="/tuners/[tuner_id]/runs/[run_id]/openai/v1/chat/completions",
        )
        raise StopUser()
