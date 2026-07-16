"""Harbor environment backed by a persistent just-bash worker process.

The worker is intentionally treated as a tiny ``JustBashLike`` service. It
accepts newline-delimited JSON requests on stdin and writes one response per
line to stdout. The only protocol methods are ``exec``, ``readFile``, and
``writeFile``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path, PurePosixPath
from typing import Any, override

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths


class JustBashWorkerError(RuntimeError):
    """Raised when the just-bash worker or its line protocol fails."""


class _JustBashWorker:
    """Async client for the three-method just-bash worker protocol."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        cwd: Path | None,
        env: Mapping[str, str],
        logger: logging.Logger,
    ) -> None:
        if not command:
            raise ValueError("worker_command must not be empty")

        self._command = tuple(command)
        self._cwd = cwd
        self._env = dict(env)
        self._logger = logger
        self._process: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._request_lock = asyncio.Lock()
        self._next_request_id = 1

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    async def start(self) -> None:
        if self.is_running:
            return

        self._process = await asyncio.create_subprocess_exec(
            *self._command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._cwd,
            env=self._env,
        )
        self._stderr_task = asyncio.create_task(self._drain_stderr())

    async def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return

        if process.stdin is not None:
            process.stdin.close()
            with suppress(BrokenPipeError, ConnectionResetError):
                await process.stdin.wait_closed()

        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except TimeoutError:
                process.kill()
                await process.wait()
        else:
            await process.wait()

        stderr_task = self._stderr_task
        self._stderr_task = None
        if stderr_task is not None:
            with suppress(asyncio.CancelledError):
                await stderr_task

    async def abort(self) -> None:
        """Kill the worker immediately after a timed-out request."""
        process = self._process
        self._process = None
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()

        stderr_task = self._stderr_task
        self._stderr_task = None
        if stderr_task is not None:
            with suppress(asyncio.CancelledError):
                await stderr_task

    async def exec(self, command: str) -> dict[str, str | int]:
        result = await self._request("exec", {"command": command})
        if not isinstance(result, dict):
            raise JustBashWorkerError("exec returned a non-object result")

        stdout = result.get("stdout")
        stderr = result.get("stderr")
        exit_code = result.get("exitCode")
        if not isinstance(stdout, str):
            raise JustBashWorkerError("exec result.stdout must be a string")
        if not isinstance(stderr, str):
            raise JustBashWorkerError("exec result.stderr must be a string")
        if not isinstance(exit_code, int) or isinstance(exit_code, bool):
            raise JustBashWorkerError("exec result.exitCode must be an integer")

        return {
            "stdout": stdout,
            "stderr": stderr,
            "exitCode": exit_code,
        }

    async def read_file(self, path: str) -> str:
        result = await self._request("readFile", {"path": path})
        if not isinstance(result, str):
            raise JustBashWorkerError("readFile returned a non-string result")
        return result

    async def write_file(self, path: str, content: str) -> None:
        result = await self._request(
            "writeFile",
            {"path": path, "content": content},
        )
        if result is not None:
            raise JustBashWorkerError("writeFile must return null")

    async def _request(self, method: str, params: dict[str, Any]) -> Any:
        async with self._request_lock:
            process = self._require_process()
            if process.stdin is None or process.stdout is None:
                raise JustBashWorkerError("worker stdio is unavailable")

            request_id = str(self._next_request_id)
            self._next_request_id += 1
            payload = json.dumps(
                {"id": request_id, "method": method, "params": params},
                separators=(",", ":"),
            )

            try:
                process.stdin.write(f"{payload}\n".encode())
                await process.stdin.drain()
                response_line = await process.stdout.readline()
            except (BrokenPipeError, ConnectionResetError) as exc:
                raise JustBashWorkerError("worker closed its protocol stream") from exc

            if not response_line:
                return_code = await process.wait()
                raise JustBashWorkerError(
                    f"worker exited before responding (exit code {return_code})"
                )

            try:
                response = json.loads(response_line)
            except json.JSONDecodeError as exc:
                raise JustBashWorkerError(
                    "worker returned malformed JSON: "
                    f"{response_line.decode(errors='replace').rstrip()}"
                ) from exc

            if not isinstance(response, dict):
                raise JustBashWorkerError("worker response must be a JSON object")
            if response.get("id") != request_id:
                raise JustBashWorkerError(
                    f"worker response id {response.get('id')!r} does not match "
                    f"request id {request_id!r}"
                )
            if "error" in response:
                error = response["error"]
                message = (
                    error.get("message") if isinstance(error, dict) else str(error)
                )
                raise JustBashWorkerError(f"{method} failed: {message}")
            if "result" not in response:
                raise JustBashWorkerError("worker response has no result")
            return response["result"]

    def _require_process(self) -> asyncio.subprocess.Process:
        if not self.is_running or self._process is None:
            raise JustBashWorkerError("worker is not running")
        return self._process

    async def _drain_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return

        while line := await process.stderr.readline():
            self._logger.debug(
                "just-bash worker: %s",
                line.decode(errors="replace").rstrip(),
            )


class JustBashEnvironment(BaseEnvironment):
    """A text-only Harbor environment backed by a just-bash worker.

    ``worker_command`` must start a process implementing the three-method line
    protocol described by :class:`_JustBashWorker`. File transfers intentionally
    support UTF-8 text only, matching the supplied TypeScript ``JustBashLike``
    interface.
    """

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        worker_command: Sequence[str] | str,
        worker_cwd: Path | str | None = None,
        worker_env: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            **kwargs,
        )

        command = (
            tuple(shlex.split(worker_command))
            if isinstance(worker_command, str)
            else tuple(worker_command)
        )
        process_env = {"PATH": os.environ.get("PATH", "")}
        if worker_env:
            process_env.update(worker_env)

        self._worker = _JustBashWorker(
            command,
            cwd=Path(worker_cwd) if worker_cwd is not None else None,
            env=process_env,
            logger=self.logger,
        )

    @staticmethod
    @override
    def type() -> str:
        return "just-bash"

    @property
    @override
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(disable_internet=True)

    @override
    def _validate_definition(self) -> None:
        # just-bash does not build a Dockerfile or container image. Harbor still
        # supplies environment_dir as task metadata, but no build definition is
        # required for this provider.
        return None

    @override
    async def start(self, force_build: bool) -> None:
        if force_build:
            self.logger.debug("force_build has no effect for just-bash")
        await self._worker.start()
        await self.ensure_dirs(self._mount_targets(writable_only=True))
        await self._upload_environment_dir_after_start()

    @override
    async def stop(self, delete: bool) -> None:
        if not delete:
            self.logger.debug(
                "just-bash environments are ephemeral and stop even when delete=False"
            )
        await self._worker.close()

    async def read_file(self, path: str) -> str:
        """Read one UTF-8 text file from the virtual filesystem."""
        return await self._worker.read_file(path)

    async def write_file(self, path: str, content: str) -> None:
        """Write one UTF-8 text file to the virtual filesystem."""
        parent = str(PurePosixPath(path).parent)
        if parent not in {"", "."}:
            result = await self.exec(f"mkdir -p -- {shlex.quote(parent)}")
            if result.return_code != 0:
                raise JustBashWorkerError(
                    f"failed to create parent directory {parent!r}: "
                    f"{result.stderr or result.stdout or 'unknown error'}"
                )
        await self._worker.write_file(path, content)

    @override
    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        source = Path(source_path)
        try:
            content = source.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"just-bash only supports UTF-8 text files: {source}"
            ) from exc
        await self.write_file(target_path, content)

    @override
    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        source_root = Path(source_dir)
        for source in source_root.rglob("*"):
            if source.is_symlink():
                raise ValueError(
                    f"just-bash directory upload rejects symlinks: {source}"
                )
            if not source.is_file():
                continue
            relative_path = source.relative_to(source_root)
            if "__pycache__" in relative_path.parts or source.suffix in {
                ".pyc",
                ".pyo",
            }:
                continue
            target = str(PurePosixPath(target_dir) / relative_path.as_posix())
            await self.upload_file(source, target)

    @override
    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(await self.read_file(source_path), encoding="utf-8")

    @override
    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        source_root = PurePosixPath(source_dir)
        target_root = Path(target_dir)
        target_root.mkdir(parents=True, exist_ok=True)

        result = await self.exec(f"find {shlex.quote(str(source_root))} -type f -print")
        if result.return_code != 0:
            raise JustBashWorkerError(
                f"failed to enumerate {source_dir!r}: "
                f"{result.stderr or result.stdout or 'unknown error'}"
            )

        for raw_path in (result.stdout or "").splitlines():
            source = PurePosixPath(raw_path)
            try:
                relative = source.relative_to(source_root)
            except ValueError as exc:
                raise JustBashWorkerError(
                    f"worker returned a path outside {source_dir!r}: {raw_path!r}"
                ) from exc
            await self.download_file(
                str(source),
                target_root / relative.as_posix(),
            )

    @override
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        self._resolve_user(user)  # Accepted for Harbor compatibility.
        merged_env = self._merge_env(env)
        wrapped_command = self._wrap_command(
            command,
            cwd=cwd or self.task_env_config.workdir,
            env=merged_env,
        )

        try:
            if timeout_sec is None:
                result = await self._worker.exec(wrapped_command)
            else:
                result = await asyncio.wait_for(
                    self._worker.exec(wrapped_command),
                    timeout=timeout_sec,
                )
        except TimeoutError:
            await self._worker.abort()
            raise

        stdout = str(result["stdout"])
        stderr = str(result["stderr"])
        callback = self._output_callback()
        if callback is not None:
            if stdout:
                await callback(stdout, "stdout")
            if stderr:
                await callback(stderr, "stderr")

        return ExecResult(
            stdout=stdout,
            stderr=stderr,
            return_code=int(result["exitCode"]),
        )

    @staticmethod
    def _wrap_command(
        command: str,
        *,
        cwd: str | None,
        env: Mapping[str, str] | None,
    ) -> str:
        if cwd is None and not env:
            return command

        inner_command = command
        if cwd is not None:
            inner_command = f"cd -- {shlex.quote(cwd)} && {inner_command}"

        assignments = ""
        if env:
            assignments = " ".join(
                f"{key}={shlex.quote(value)}" for key, value in env.items()
            )
        return f"env {assignments} bash -c {shlex.quote(inner_command)}".replace(
            "env  bash", "env bash"
        )
