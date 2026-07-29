"""Hybrid Harbor environment for Ollie agents and isolated verifiers.

Agent sessions run directly on the host in a trial-local workspace. Separate
verifier sessions can be delegated to a built-in or custom Harbor environment,
such as Daytona. Host execution is intended only for trusted agent commands and
is not an isolation boundary.
"""

from __future__ import annotations

import asyncio
import getpass
import os
import re
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path, PurePath, PurePosixPath
from typing import Any, ClassVar, override

from harbor.environments.base import BaseEnvironment, EnvironmentPath, ExecResult
from harbor.environments.capabilities import (
    EnvironmentCapabilities,
    EnvironmentResourceCapabilities,
)
from harbor.environments.factory import EnvironmentFactory
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import EnvironmentConfig, NetworkMode, TaskOS
from harbor.models.trial.paths import TrialPaths


class OllieEnvironment(BaseEnvironment):
    """Run the agent locally and delegate separate verifier environments.

    Harbor uses the same environment import path when it creates a separate
    verifier. Its verifier session IDs contain ``__verifier__``; those sessions
    are returned as an instance of ``verifier_environment`` instead. Agent
    sessions remain host-local and execute with the current user's permissions.
    """

    _VERIFIER_SESSION_MARKER = "__verifier__"
    _RESERVED_PATHS: ClassVar[tuple[str, ...]] = (
        "/logs/agent",
        "/logs/verifier",
        "/logs/artifacts",
        "/installed-agent",
        "/harbor/skills",
        "/solution",
        "/tests",
        "/workspace",
    )

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        session_id = kwargs.get("session_id")
        verifier_environment = kwargs.get("verifier_environment")
        if (
            cls is OllieEnvironment
            and isinstance(session_id, str)
            and cls._VERIFIER_SESSION_MARKER in session_id
            and verifier_environment
        ):
            delegate_kwargs = dict(kwargs)
            delegate_kwargs.pop("verifier_environment", None)
            environment_ref = str(verifier_environment)
            if ":" in environment_ref:
                if environment_ref == "ollie.harbor_environment:OllieEnvironment":
                    raise ValueError("OllieEnvironment cannot delegate to itself")
                EnvironmentFactory.run_preflight(None, environment_ref)
                return EnvironmentFactory.create_environment_from_import_path(
                    environment_ref,
                    **delegate_kwargs,
                )

            try:
                environment_type = EnvironmentType(environment_ref)
            except ValueError as error:
                raise ValueError(
                    "verifier_environment must be a built-in Harbor environment "
                    f"name or import path; received {environment_ref!r}"
                ) from error
            EnvironmentFactory.run_preflight(environment_type)
            return EnvironmentFactory.create_environment(
                type=environment_type,
                **delegate_kwargs,
            )
        return super().__new__(cls)

    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: EnvironmentConfig,
        *args: Any,
        verifier_environment: str | None = None,
        **kwargs: Any,
    ) -> None:
        del args, verifier_environment
        self._root = trial_paths.trial_dir / ".ollie-environment"
        self._host_workdir = self._root / "workspace"
        self._tests_dir = self._root / "tests"
        self._solution_dir = self._root / "solution"
        self._installed_agent_dir = self._root / "installed-agent"
        self._skills_dir = self._root / "skills"
        self._virtual_workdir = self._normalize_workdir(
            task_env_config.workdir or "/workspace"
        )
        super().__init__(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            **kwargs,
        )

    @staticmethod
    def _normalize_workdir(workdir: str) -> str:
        path = PurePosixPath(workdir)
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("Ollie environment workdir must be an absolute POSIX path")
        return str(path)

    @staticmethod
    @override
    def type() -> str:
        return "ollie"

    @classmethod
    @override
    def preflight(cls) -> None:
        node = shutil.which("node")
        npx = shutil.which("npx")
        if not node or not npx:
            raise SystemExit("OllieEnvironment requires host Node.js >=22 and npx")
        result = subprocess.run(
            [node, "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
        match = re.search(r"v?(\d+)", result.stdout)
        if result.returncode != 0 or not match or int(match.group(1)) < 22:
            raise SystemExit(
                "OllieEnvironment requires host Node.js >=22; "
                f"found {result.stdout.strip() or 'unknown'}"
            )

    @classmethod
    @override
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities:
        return EnvironmentResourceCapabilities()

    @property
    @override
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(mounted=True)

    @override
    def _validate_definition(self) -> None:
        if self.task_env_config.os != TaskOS.LINUX:
            raise ValueError("OllieEnvironment supports Linux-style tasks only")
        if self.task_env_config.network_mode != NetworkMode.PUBLIC:
            raise ValueError("The local agent environment cannot restrict networking")
        if self._virtual_workdir != "/workspace" and any(
            self._virtual_workdir == reserved
            or self._virtual_workdir.startswith(f"{reserved}/")
            or reserved.startswith(f"{self._virtual_workdir}/")
            for reserved in self._RESERVED_PATHS
        ):
            raise ValueError(
                "Ollie environment workdir overlaps a reserved Harbor path: "
                f"{self._virtual_workdir}"
            )
        if any(
            (self.environment_dir / name).exists()
            for name in ("docker-compose.yaml", "docker-compose.yml", "compose.yaml")
        ):
            raise ValueError("The local agent environment does not support Compose")

    @property
    def host_workdir(self) -> Path:
        """Return the host directory used as the agent workspace."""
        return self._host_workdir

    def _mapping(self) -> list[tuple[str, Path]]:
        mapping = [
            ("/logs/agent", self.trial_paths.agent_dir),
            ("/logs/verifier", self.trial_paths.verifier_dir),
            ("/logs/artifacts", self.trial_paths.artifacts_dir),
            ("/installed-agent", self._installed_agent_dir),
            ("/harbor/skills", self._skills_dir),
            ("/solution", self._solution_dir),
            ("/tests", self._tests_dir),
            ("/workspace", self._host_workdir),
            ("/logs", self.trial_paths.trial_dir),
        ]
        if self._virtual_workdir != "/workspace":
            mapping.append((self._virtual_workdir, self._host_workdir))
        return sorted(mapping, key=lambda item: len(item[0]), reverse=True)

    @staticmethod
    def _contained_path(root: Path, relative: PurePosixPath) -> Path:
        candidate = root.joinpath(*relative.parts)
        try:
            candidate.resolve(strict=False).relative_to(root.resolve(strict=False))
        except ValueError as error:
            raise ValueError(
                f"Path escapes the Ollie environment: {candidate}"
            ) from error
        return candidate

    def _virtual_to_host(self, value: str | PurePath) -> Path:
        path = PurePosixPath(value)
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsupported Ollie environment path: {str(value)!r}")
        normalized = str(path)
        for virtual, host in self._mapping():
            if normalized == virtual:
                return self._contained_path(host, PurePosixPath("."))
            prefix = f"{virtual}/"
            if normalized.startswith(prefix):
                relative = PurePosixPath(normalized[len(prefix) :])
                return self._contained_path(host, relative)
        raise ValueError(f"Path is outside the Ollie environment: {str(value)!r}")

    @override
    async def start(self, force_build: bool) -> None:
        del force_build
        for path in (
            self._host_workdir,
            self._tests_dir,
            self._solution_dir,
            self._installed_agent_dir,
            self._skills_dir,
            self.trial_paths.agent_dir,
            self.trial_paths.verifier_dir,
            self.trial_paths.artifacts_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    @override
    async def stop(self, delete: bool) -> None:
        del delete

    @override
    async def ensure_dirs(
        self,
        dirs: Sequence[EnvironmentPath],
        *,
        chmod: bool = True,
    ) -> ExecResult | None:
        del chmod
        for directory in dirs:
            self._virtual_to_host(directory).mkdir(parents=True, exist_ok=True)
        return None

    @override
    async def empty_dirs(
        self,
        dirs: Sequence[EnvironmentPath],
        *,
        chmod: bool = True,
    ) -> ExecResult | None:
        del chmod
        for directory in dirs:
            host_dir = self._virtual_to_host(directory)
            host_dir.mkdir(parents=True, exist_ok=True)
            for child in host_dir.iterdir():
                if child.is_symlink() or child.is_file():
                    child.unlink()
                else:
                    shutil.rmtree(child)
        return None

    @override
    async def reset_dirs(
        self,
        *,
        remove_dirs: Sequence[EnvironmentPath],
        create_dirs: Sequence[EnvironmentPath],
        chmod_dirs: Sequence[EnvironmentPath] | None = None,
    ) -> ExecResult:
        for directory in remove_dirs:
            host_dir = self._virtual_to_host(directory)
            if host_dir.is_symlink() or host_dir.is_file():
                host_dir.unlink(missing_ok=True)
            else:
                shutil.rmtree(host_dir, ignore_errors=True)
        await self.ensure_dirs(create_dirs, chmod=bool(chmod_dirs))
        return ExecResult(stdout=None, stderr=None, return_code=0)

    @override
    async def is_dir(self, path: str, user: str | int | None = None) -> bool:
        del user
        return self._virtual_to_host(path).is_dir()

    @override
    async def is_file(self, path: str, user: str | int | None = None) -> bool:
        del user
        return self._virtual_to_host(path).is_file()

    @override
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        effective_user = self._resolve_user(user)
        current_users = (None, "root", os.getuid(), os.geteuid(), getpass.getuser())
        if effective_user not in current_users:
            raise NotImplementedError(
                f"OllieEnvironment cannot switch to user {effective_user!r}"
            )
        if timeout_sec is not None and timeout_sec <= 0:
            raise ValueError("timeout_sec must be positive")

        host_cwd = self._virtual_to_host(cwd or self._virtual_workdir)
        host_cwd.mkdir(parents=True, exist_ok=True)
        process_env = os.environ.copy()
        process_env.update(self._merge_env(env) or {})

        process = await asyncio.create_subprocess_exec(
            "/bin/bash",
            "-lc",
            command,
            cwd=host_cwd,
            env=process_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            if timeout_sec is None:
                stdout_bytes, stderr_bytes = await process.communicate()
            else:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(), timeout=timeout_sec
                )
        except asyncio.CancelledError:
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.communicate(), timeout=5)
                except TimeoutError:
                    process.kill()
                    await process.communicate()
            raise
        except TimeoutError:
            process.terminate()
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(), timeout=5
                )
            except TimeoutError:
                process.kill()
                stdout_bytes, stderr_bytes = await process.communicate()
            return ExecResult(
                stdout=stdout_bytes.decode(errors="replace") or None,
                stderr=(
                    stderr_bytes.decode(errors="replace") + "\nCommand timed out"
                ).strip(),
                return_code=124,
            )

        stdout = stdout_bytes.decode(errors="replace") or None
        stderr = stderr_bytes.decode(errors="replace") or None
        callback = self._output_callback()
        if callback is not None:
            if stdout:
                await callback(stdout, "stdout")
            if stderr:
                await callback(stderr, "stderr")
        return ExecResult(
            stdout=stdout,
            stderr=stderr,
            return_code=process.returncode or 0,
        )

    @staticmethod
    def _reject_symlink(path: Path) -> None:
        if path.is_symlink():
            raise ValueError(f"Symlinks are not supported: {path}")

    @override
    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        source = Path(source_path)
        self._reject_symlink(source)
        target = self._virtual_to_host(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    @override
    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        source = Path(source_dir)
        self._reject_symlink(source)
        target = self._virtual_to_host(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        for path in source.rglob("*"):
            self._reject_symlink(path)
            relative = path.relative_to(source)
            destination = target / relative
            if path.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)

    @override
    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        source = self._virtual_to_host(source_path)
        self._reject_symlink(source)
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    @override
    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        source = self._virtual_to_host(source_dir)
        self._reject_symlink(source)
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        for path in source.rglob("*"):
            self._reject_symlink(path)
            relative = path.relative_to(source)
            destination = target / relative
            if path.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
