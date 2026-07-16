"""A host-side Harbor agent for persistent just-bash environments."""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, override

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.agent.context import AgentContext
from harbor.models.trajectories import (
    Agent,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageFunctionToolCall

from just_bash.env import JustBashEnvironment

_DEFAULT_SYSTEM_PROMPT = """You are solving a task in a restricted, persistent just-bash workspace.
Use read_file and write_file for direct UTF-8 file access, and use bash to inspect the
workspace or run commands. The filesystem persists between tool calls, but this is not a
full Linux machine: arbitrary native binaries, package installation, tmux, and system
services may be unavailable. Network access is disabled. Check command exit codes and
outputs and correct mistakes.

When the task is complete, call mark_task_complete. You will receive a final warning.
Review the workspace and call mark_task_complete again on the next turn only when you are
certain it is ready for grading. The verifier grades files, not your prose."""

_BASH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": (
            "Execute a shell command in the persistent just-bash workspace. "
            "Filesystem changes remain visible to later calls."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                }
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}
_READ_FILE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a UTF-8 text file from the persistent workspace.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
}
_WRITE_FILE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write UTF-8 text to a file in the persistent workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    },
}
_COMPLETE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "mark_task_complete",
        "description": "Request grading; call on two consecutive turns to confirm.",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
}
_TOOLS = [_BASH_TOOL, _READ_FILE_TOOL, _WRITE_FILE_TOOL, _COMPLETE_TOOL]
_COMPLETION_CONFIRMATION = (
    "Are you sure? Grading will begin and no further corrections can be made. "
    "Review the workspace, then call mark_task_complete again if it is ready."
)
_MISSING_ACTION_FEEDBACK = (
    "The task is not complete. Use bash, read_file, or write_file to continue, or "
    "call mark_task_complete when the workspace is ready for grading."
)


class JustBashAgent(BaseAgent):
    """Run an OpenAI tool-calling loop against a just-bash environment.

    The model runs on the host through an OpenAI-compatible endpoint. Only shell
    commands execute in the Harbor environment, so the agent requires no tmux,
    terminal emulation, or in-environment installation.
    """

    SUPPORTS_ATIF = True

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        api_base: str | None = None,
        api_key: str = "ollie",
        max_turns: int = 32,
        command_timeout_sec: int | None = None,
        completion_timeout_sec: float | None = None,
        max_observation_chars: int = 20_000,
        temperature: float | None = None,
        system_prompt: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(logs_dir, model_name, *args, **kwargs)

        if model_name is None:
            raise ValueError("model_name is required for JustBashAgent")
        if api_base is None:
            raise ValueError("api_base is required for JustBashAgent")
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        if command_timeout_sec is not None and command_timeout_sec <= 0:
            raise ValueError("command_timeout_sec must be positive")
        if completion_timeout_sec is not None and completion_timeout_sec <= 0:
            raise ValueError("completion_timeout_sec must be positive")
        if max_observation_chars < 1:
            raise ValueError("max_observation_chars must be at least 1")

        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "base_url": api_base,
        }
        if completion_timeout_sec is not None:
            client_kwargs["timeout"] = completion_timeout_sec

        self._client = AsyncOpenAI(**client_kwargs)
        self._model_name = model_name
        self._api_base = api_base
        self._max_turns = max_turns
        self._command_timeout_sec = command_timeout_sec
        self._max_observation_chars = max_observation_chars
        self._temperature = temperature
        self._system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT
        self._transcript_path = logs_dir / "just-bash-transcript.json"
        self._trajectory_path = logs_dir / "trajectory.json"

    @staticmethod
    @override
    def name() -> str:
        return "just-bash"

    @override
    def version(self) -> str:
        return "1"

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if not isinstance(environment, JustBashEnvironment):
            raise TypeError(
                "JustBashAgent requires a JustBashEnvironment, got "
                f"{type(environment).__name__}"
            )

        session_id = self.session_id or str(uuid.uuid4())
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": instruction},
        ]
        events: list[dict[str, Any]] = []
        steps = [
            Step(
                step_id=1,
                timestamp=self._timestamp(),
                source="user",
                message=instruction,
            )
        ]
        request_times_ms: list[float] = []
        total_input_tokens = 0
        total_output_tokens = 0
        total_cache_tokens: int | None = None
        command_count = 0
        final_content: str | None = None
        stop_reason = "max_turns"
        completion_pending = False

        self._update_context(
            context,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            cache_tokens=total_cache_tokens,
            turn_count=0,
            command_count=command_count,
            stop_reason="running",
            final_content=final_content,
            session_id=session_id,
            request_times_ms=request_times_ms,
        )
        self._write_transcript(
            messages=messages,
            events=events,
            stop_reason="running",
        )
        self._write_trajectory(
            session_id=session_id,
            steps=steps,
            input_tokens=0,
            output_tokens=0,
            cache_tokens=None,
            stop_reason="running",
        )

        try:
            for turn_index in range(1, self._max_turns + 1):
                completion_kwargs: dict[str, Any] = {
                    "model": self._model_name,
                    "messages": messages,
                    "tools": _TOOLS,
                    "tool_choice": "auto",
                }
                if self._temperature is not None:
                    completion_kwargs["temperature"] = self._temperature

                request_started = time.perf_counter()
                response = await self._client.chat.completions.create(
                    **completion_kwargs
                )
                request_time_ms = (time.perf_counter() - request_started) * 1000
                request_times_ms.append(request_time_ms)
                if not response.choices:
                    raise RuntimeError("chat completion returned no choices")

                choice = response.choices[0]
                assistant_message = choice.message.model_dump(exclude_none=True)
                messages.append(assistant_message)

                usage = response.usage
                input_tokens = usage.prompt_tokens if usage is not None else None
                output_tokens = usage.completion_tokens if usage is not None else None
                cached_tokens: int | None = None
                if usage is not None:
                    total_input_tokens += usage.prompt_tokens
                    total_output_tokens += usage.completion_tokens
                    prompt_details = usage.prompt_tokens_details
                    if (
                        prompt_details is not None
                        and prompt_details.cached_tokens is not None
                    ):
                        cached_tokens = prompt_details.cached_tokens
                        total_cache_tokens = (
                            total_cache_tokens or 0
                        ) + prompt_details.cached_tokens

                message_dump = choice.message.model_dump(exclude_none=True)
                reasoning_content = message_dump.get("reasoning_content")
                if not isinstance(reasoning_content, str):
                    reasoning_content = None
                raw_tool_calls = choice.message.tool_calls or []
                if not all(
                    isinstance(call, ChatCompletionMessageFunctionToolCall)
                    for call in raw_tool_calls
                ):
                    raise RuntimeError("model returned an unsupported custom tool call")
                tool_calls = [
                    call
                    for call in raw_tool_calls
                    if isinstance(call, ChatCompletionMessageFunctionToolCall)
                ]
                events.append(
                    {
                        "type": "completion",
                        "turn": turn_index,
                        "response_id": response.id,
                        "finish_reason": choice.finish_reason,
                        "usage": usage.model_dump(exclude_none=True)
                        if usage is not None
                        else None,
                        "message": assistant_message,
                    }
                )

                if not tool_calls:
                    completion_pending = False
                    final_content = choice.message.content
                    messages.append(
                        {"role": "user", "content": _MISSING_ACTION_FEEDBACK}
                    )
                    steps.append(
                        self._agent_step(
                            step_id=len(steps) + 1,
                            message=final_content,
                            reasoning_content=reasoning_content,
                            tool_calls=[],
                            observations=[
                                ObservationResult(content=_MISSING_ACTION_FEEDBACK)
                            ],
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            cached_tokens=cached_tokens,
                            request_time_ms=request_time_ms,
                            response_id=response.id,
                        )
                    )
                    self._write_trajectory(
                        session_id=session_id,
                        steps=steps,
                        input_tokens=total_input_tokens,
                        output_tokens=total_output_tokens,
                        cache_tokens=total_cache_tokens,
                        stop_reason="running",
                    )
                    self._update_context(
                        context,
                        input_tokens=total_input_tokens,
                        output_tokens=total_output_tokens,
                        cache_tokens=total_cache_tokens,
                        turn_count=turn_index,
                        command_count=command_count,
                        stop_reason="running",
                        final_content=final_content,
                        session_id=session_id,
                        request_times_ms=request_times_ms,
                    )
                    self._write_transcript(
                        messages=messages,
                        events=events,
                        stop_reason="running",
                    )
                    continue

                has_completion_call = any(
                    call.function.name == "mark_task_complete" for call in tool_calls
                )
                was_completion_pending = completion_pending
                completion_pending = has_completion_call
                turn_tool_calls: list[ToolCall] = []
                turn_observations: list[ObservationResult] = []
                completion_confirmed = False

                for tool_call in tool_calls:
                    name, arguments = self._parse_tool_call(tool_call)
                    turn_tool_calls.append(
                        ToolCall(
                            tool_call_id=tool_call.id,
                            function_name=name,
                            arguments=arguments,
                        )
                    )
                    if name == "mark_task_complete":
                        completion_confirmed = was_completion_pending
                        observation = {
                            "status": "confirmed"
                            if completion_confirmed
                            else "confirmation_required",
                            "message": "Task marked complete."
                            if completion_confirmed
                            else _COMPLETION_CONFIRMATION,
                        }
                        command = None
                    else:
                        if name == "bash":
                            command_count += 1
                        observation, command = await self._run_tool_call(
                            tool_call=tool_call,
                            environment=environment,
                        )
                    observation_text = json.dumps(
                        observation, ensure_ascii=False, separators=(",", ":")
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": observation_text,
                        }
                    )
                    turn_observations.append(
                        ObservationResult(
                            source_call_id=tool_call.id,
                            content=observation_text,
                        )
                    )
                    events.append(
                        {
                            "type": "tool",
                            "turn": turn_index,
                            "tool_call_id": tool_call.id,
                            "tool": name,
                            "command": command,
                            "observation": observation,
                        }
                    )

                steps.append(
                    self._agent_step(
                        step_id=len(steps) + 1,
                        message=choice.message.content,
                        reasoning_content=reasoning_content,
                        tool_calls=turn_tool_calls,
                        observations=turn_observations,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cached_tokens=cached_tokens,
                        request_time_ms=request_time_ms,
                        response_id=response.id,
                    )
                )
                stop_reason = "completed" if completion_confirmed else "running"
                self._write_trajectory(
                    session_id=session_id,
                    steps=steps,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    cache_tokens=total_cache_tokens,
                    stop_reason=stop_reason,
                )

                self._update_context(
                    context,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    cache_tokens=total_cache_tokens,
                    turn_count=turn_index,
                    command_count=command_count,
                    stop_reason=stop_reason,
                    final_content=final_content,
                    session_id=session_id,
                    request_times_ms=request_times_ms,
                )
                self._write_transcript(
                    messages=messages,
                    events=events,
                    stop_reason=stop_reason,
                )
                if completion_confirmed:
                    return
        except BaseException as exc:
            stop_reason = f"error:{type(exc).__name__}"
            self._update_context(
                context,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                cache_tokens=total_cache_tokens,
                turn_count=self._completed_turn_count(events),
                command_count=command_count,
                stop_reason=stop_reason,
                final_content=final_content,
                session_id=session_id,
                request_times_ms=request_times_ms,
            )
            self._write_transcript(
                messages=messages,
                events=events,
                stop_reason=stop_reason,
            )
            self._write_trajectory(
                session_id=session_id,
                steps=steps,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                cache_tokens=total_cache_tokens,
                stop_reason=stop_reason,
            )
            raise

        stop_reason = "max_turns"
        self._update_context(
            context,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            cache_tokens=total_cache_tokens,
            turn_count=self._max_turns,
            command_count=command_count,
            stop_reason=stop_reason,
            final_content=final_content,
            session_id=session_id,
            request_times_ms=request_times_ms,
        )
        self._write_transcript(
            messages=messages,
            events=events,
            stop_reason=stop_reason,
        )
        self._write_trajectory(
            session_id=session_id,
            steps=steps,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            cache_tokens=total_cache_tokens,
            stop_reason=stop_reason,
        )

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _parse_tool_call(
        tool_call: ChatCompletionMessageFunctionToolCall,
    ) -> tuple[str, dict[str, Any]]:
        name = tool_call.function.name
        raw_arguments = tool_call.function.arguments
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError, TypeError:
            return name, {"raw_arguments": raw_arguments}
        if not isinstance(arguments, dict):
            return name, {"raw_arguments": raw_arguments}
        return name, arguments

    def _agent_step(
        self,
        *,
        step_id: int,
        message: str | None,
        reasoning_content: str | None,
        tool_calls: list[ToolCall],
        observations: list[ObservationResult],
        input_tokens: int | None,
        output_tokens: int | None,
        cached_tokens: int | None,
        request_time_ms: float,
        response_id: str,
    ) -> Step:
        return Step(
            step_id=step_id,
            timestamp=self._timestamp(),
            source="agent",
            model_name=self._model_name,
            message=message or "",
            reasoning_content=reasoning_content,
            tool_calls=tool_calls or None,
            observation=Observation(results=observations),
            metrics=Metrics(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                cached_tokens=cached_tokens,
                extra={
                    "request_time_ms": request_time_ms,
                    "response_id": response_id,
                },
            ),
            llm_call_count=1,
        )

    def _write_trajectory(
        self,
        *,
        session_id: str,
        steps: list[Step],
        input_tokens: int,
        output_tokens: int,
        cache_tokens: int | None,
        stop_reason: str,
    ) -> None:
        trajectory = Trajectory(
            session_id=session_id,
            agent=Agent(
                name=self.name(),
                version=self.version(),
                model_name=self._model_name,
                tool_definitions=_TOOLS,
                extra={"api_base": self._api_base},
            ),
            steps=steps,
            final_metrics=FinalMetrics(
                total_prompt_tokens=input_tokens,
                total_completion_tokens=output_tokens,
                total_cached_tokens=cache_tokens,
                total_steps=len(steps),
                extra={"stop_reason": stop_reason},
            ),
        )
        self._atomic_write_json(
            self._trajectory_path,
            trajectory.to_json_dict(),
        )

    async def _run_tool_call(
        self,
        *,
        tool_call: ChatCompletionMessageFunctionToolCall,
        environment: JustBashEnvironment,
    ) -> tuple[dict[str, Any], str | None]:
        name = tool_call.function.name
        if name not in {"bash", "read_file", "write_file"}:
            return {"error": f"unsupported tool: {name}"}, None

        raw_arguments = tool_call.function.arguments
        try:
            arguments = json.loads(raw_arguments)
        except (json.JSONDecodeError, TypeError) as exc:
            detail = exc.msg if isinstance(exc, json.JSONDecodeError) else str(exc)
            return {"error": f"invalid JSON arguments: {detail}"}, None

        if not isinstance(arguments, dict):
            return {"error": "tool arguments must be a JSON object"}, None

        if name == "bash":
            command = arguments.get("command")
            if not isinstance(command, str) or not command.strip():
                return {"error": "command must be a non-empty string"}, None
            result = await environment.exec(
                command,
                timeout_sec=self._command_timeout_sec,
            )
            return self._observation(result), command

        path = arguments.get("path")
        if not isinstance(path, str) or not path.strip():
            return {"error": "path must be a non-empty string"}, None

        if name == "read_file":
            content = await environment.read_file(path)
            return {"content": self._truncate(content)}, None

        content = arguments.get("content")
        if not isinstance(content, str):
            return {"error": "content must be a string"}, None
        await environment.write_file(path, content)
        return {"status": "ok"}, None

    def _observation(self, result: ExecResult) -> dict[str, Any]:
        return {
            "stdout": self._truncate(result.stdout or ""),
            "stderr": self._truncate(result.stderr or ""),
            "exit_code": result.return_code,
        }

    def _truncate(self, value: str) -> str:
        limit = self._max_observation_chars
        if len(value) <= limit:
            return value

        omitted = len(value) - limit
        while True:
            marker = f"\n[... {omitted} characters omitted ...]\n"
            retained = max(0, limit - len(marker))
            updated_omitted = len(value) - retained
            if updated_omitted == omitted:
                break
            omitted = updated_omitted

        if len(marker) >= limit:
            return marker[:limit]
        head_length = retained // 2
        tail_length = retained - head_length
        return value[:head_length] + marker + value[-tail_length:]

    @staticmethod
    def _update_context(
        context: AgentContext,
        *,
        input_tokens: int,
        output_tokens: int,
        cache_tokens: int | None,
        turn_count: int,
        command_count: int,
        stop_reason: str,
        final_content: str | None,
        session_id: str | None = None,
        request_times_ms: list[float] | None = None,
    ) -> None:
        context.n_input_tokens = input_tokens
        context.n_output_tokens = output_tokens
        context.n_cache_tokens = cache_tokens
        context.metadata = {
            "session_id": session_id,
            "n_episodes": turn_count,
            "turn_count": turn_count,
            "command_count": command_count,
            "stop_reason": stop_reason,
            "final_content": final_content,
            "api_request_times_msec": request_times_ms or [],
        }

    @staticmethod
    def _completed_turn_count(events: list[dict[str, Any]]) -> int:
        return sum(event.get("type") == "completion" for event in events)

    def _write_transcript(
        self,
        *,
        messages: list[dict[str, Any]],
        events: list[dict[str, Any]],
        stop_reason: str,
    ) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "agent": self.name(),
            "version": self.version(),
            "model": self._model_name,
            "api_base": self._api_base,
            "stop_reason": stop_reason,
            "messages": messages,
            "events": events,
        }
        self._atomic_write_json(self._transcript_path, payload)

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(path)
