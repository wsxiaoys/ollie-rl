"""Harbor installed-agent integration for :mod:`@getollie/cli`.

Use the agent together with its required environment import path::

    harbor run -d "<dataset@version>" \
        --env ollie.harbor_environment:OllieEnvironment \
        --agent ollie.harbor_agent:OllieAgent \
        --model openai/gpt-5.6

The CLI's NDJSON event stream is retained in the Harbor agent log directory and
its final token usage is copied into :class:`AgentContext` after the run.
"""

from __future__ import annotations

import json
import shlex
from collections import Counter
from pathlib import Path
from typing import Any, ClassVar, override

from harbor.agents.installed.base import (
    BaseInstalledAgent,
    CliFlag,
    with_prompt_template,
)
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from ollie.harbor_environment import OllieEnvironment


class OllieAgent(BaseInstalledAgent):
    """Run Ollie using the required :class:`OllieEnvironment` adapter."""

    _DEFAULT_VERSION = "0.2.2"
    _OUTPUT_FILENAME = "ollie.ndjson"
    _STDERR_FILENAME = "ollie.stderr"

    CLI_FLAGS: ClassVar[list[CliFlag]] = [
        CliFlag(
            "executor",
            cli="--executor",
            type="enum",
            choices=["none", "local", "daytona"],
            default="none",
        ),
        CliFlag(
            "max_steps",
            cli="--max-steps",
            type="int",
        ),
        CliFlag(
            "command_timeout_ms",
            cli="--timeout",
            type="int",
        ),
        CliFlag(
            "max_output_bytes",
            cli="--max-output",
            type="int",
        ),
    ]

    def __init__(
        self,
        logs_dir: Path,
        prompt_template_path: Path | str | None = None,
        version: str | None = None,
        extra_env: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._package_version = version or self._DEFAULT_VERSION
        super().__init__(
            logs_dir=logs_dir,
            prompt_template_path=prompt_template_path,
            version=self._package_version,
            extra_env=extra_env,
            **kwargs,
        )

    @staticmethod
    @override
    def name() -> str:
        return "ollie"

    @staticmethod
    def _require_environment(environment: BaseEnvironment) -> OllieEnvironment:
        if not isinstance(environment, OllieEnvironment):
            raise TypeError(
                "OllieAgent requires --env ollie.harbor_environment:OllieEnvironment"
            )
        return environment

    def _package_spec(self) -> str:
        return f"@getollie/cli@{self._package_version}"

    @override
    def get_version_command(self) -> str | None:
        argv = ["npx", "--yes", self._package_spec(), "--version"]
        return " ".join(shlex.quote(part) for part in argv)

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        environment = self._require_environment(environment)
        # npx resolves the package into npm's user cache. Nothing is installed
        # globally and no root-level setup is performed.
        await self.exec_as_agent(
            environment,
            command=self.get_version_command() or "false",
        )

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        environment = self._require_environment(environment)
        # BaseInstalledAgent creates /installed-agent for container installs.
        # Ollie is resolved through npx, so no virtual installation directory
        # is needed by the host-local adapter.
        (self.logs_dir / "setup").mkdir(parents=True, exist_ok=True)
        await self.install(environment)

    def _model_args(self) -> list[str]:
        if not self.model_name:
            return []

        if "/" in self.model_name:
            provider, model = self.model_name.split("/", 1)
        else:
            provider, model = "openai", self.model_name

        if provider != "openai":
            raise ValueError(
                "@getollie/cli currently supports only the openai provider; "
                f"received {provider!r}"
            )
        return ["--provider", provider, "--model", model]

    def _cli_args(self) -> list[str]:
        args = ["--ndjson", "--no-color", *self._model_args()]
        for descriptor in self.CLI_FLAGS:
            value = self._resolved_flags.get(descriptor.kwarg)
            if value is not None:
                args.extend([descriptor.cli, str(value)])
        return args

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        environment = self._require_environment(environment)
        output_path = (self.logs_dir / self._OUTPUT_FILENAME).resolve()
        stderr_path = (self.logs_dir / self._STDERR_FILENAME).resolve()
        argv = [
            "npx",
            "--yes",
            self._package_spec(),
            "agent",
            "--cwd",
            ".",
            *self._cli_args(),
            instruction,
        ]
        command = " ".join(shlex.quote(part) for part in argv)

        await self.exec_as_agent(
            environment,
            command=(
                "set -o pipefail; "
                f"{command} 2> {shlex.quote(str(stderr_path))} "
                f"| tee {shlex.quote(str(output_path))}"
            ),
        )

    @staticmethod
    def _token_count(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, dict):
            for key in ("total", "totalTokens"):
                count = OllieAgent._token_count(value.get(key))
                if count is not None:
                    return count
        return None

    def _read_events(self) -> list[dict[str, Any]]:
        output_path = self.logs_dir / self._OUTPUT_FILENAME
        if not output_path.exists():
            return []

        events: list[dict[str, Any]] = []
        for line in output_path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
        return events

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        events = self._read_events()
        if not events:
            return

        done_event = next(
            (event for event in reversed(events) if event.get("type") == "done"),
            None,
        )
        usage = done_event.get("usage") if done_event else None
        if isinstance(usage, dict):
            context.n_input_tokens = self._token_count(usage.get("inputTokens"))
            context.n_output_tokens = self._token_count(usage.get("outputTokens"))

            cache_tokens = self._token_count(usage.get("cachedInputTokens"))
            input_detail = usage.get("inputTokens")
            if cache_tokens is None and isinstance(input_detail, dict):
                cache_tokens = self._token_count(
                    input_detail.get("cacheRead") or input_detail.get("cacheReadTokens")
                )
            context.n_cache_tokens = cache_tokens

        event_counts = Counter(str(event.get("type", "unknown")) for event in events)
        errors = [
            str(event.get("message"))
            for event in events
            if event.get("type") == "error" and event.get("message")
        ]
        context.metadata = {
            **(context.metadata or {}),
            "ollie": {
                "event_counts": dict(event_counts),
                "errors": errors,
                "output_file": self._OUTPUT_FILENAME,
                "stderr_file": self._STDERR_FILENAME,
            },
        }
