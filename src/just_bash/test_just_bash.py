from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import override

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
)
from harbor.trial.trial import Trial

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TEST_TASK_DIR = Path(__file__).parent / "testdata" / "code_contests-0000"
_DEFAULT_OLLIE_DIR = _REPO_ROOT.parent / "ollie"


class SolveBracketsAgent(BaseAgent):
    """Host-side test agent that solves the example balanced-brackets task."""

    @staticmethod
    @override
    def name() -> str:
        return "solve-brackets"

    @override
    def version(self) -> str:
        return "test"

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        return None

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        result = await environment.exec(
            """mkdir -p /app
cat > /app/solution.py <<'PY'
import sys


def is_balanced(text: str) -> bool:
    depth = 0
    for character in text:
        if character == "(":
            depth += 1
        else:
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


bracket_strings = sys.argv[1:] or sys.stdin.read().splitlines()[1:]
for bracket_string in bracket_strings:
    print("YES" if is_balanced(bracket_string) else "NO")
PY
"""
        )
        if result.return_code != 0:
            raise RuntimeError(result.stderr or result.stdout or "agent command failed")
        context.metadata = {"instruction": instruction}


class TestJustBashEnvironmentTrial(unittest.IsolatedAsyncioTestCase):
    async def test_harbor_trial_with_real_ollie_sandbox(self) -> None:
        ollie_dir = Path(os.environ.get("OLLIE_SANDBOX_DIR", _DEFAULT_OLLIE_DIR))
        if shutil.which("pnpm") is None:
            self.skipTest("pnpm is not installed")
        if not (ollie_dir / "package.json").is_file():
            self.skipTest("Ollie checkout not found; set OLLIE_SANDBOX_DIR to its root")

        workspace_dir = Path(tempfile.mkdtemp(prefix="ollie-sandbox-", dir="/tmp"))
        trials_dir = workspace_dir / "trials"
        trial_name = "real-just-bash-code-contest-test"
        trial_dir = trials_dir / trial_name

        config = TrialConfig(
            task=TaskConfig(path=_TEST_TASK_DIR),
            trial_name=trial_name,
            trials_dir=trials_dir,
            agent=AgentConfig(import_path=f"{__name__}:SolveBracketsAgent"),
            environment=EnvironmentConfig(
                import_path="just_bash:JustBashEnvironment",
                kwargs={
                    # Suppress pnpm lifecycle banners so stdout remains a clean
                    # NDJSON protocol stream from the sandbox worker.
                    "worker_command": [
                        "pnpm",
                        "--silent",
                        "run",
                        "ollie",
                        "sandbox",
                        "--cwd",
                        str(workspace_dir),
                    ],
                    "worker_cwd": str(ollie_dir),
                },
            ),
        )

        try:
            trial = await Trial.create(config)
            result = await trial.run()

            self.assertIsNone(result.exception_info)
            self.assertIsNotNone(result.verifier_result)
            assert result.verifier_result is not None
            self.assertEqual(result.verifier_result.rewards, {"reward": 1.0})
            self.assertEqual(
                (trial_dir / "verifier" / "reward.txt").read_text().strip(),
                "1",
            )
        finally:
            print(f"Harbor trial directory: {trial_dir}")
