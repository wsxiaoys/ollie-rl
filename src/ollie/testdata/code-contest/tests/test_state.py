# Adapted from open-thoughts/CodeContests task code_contests-0000.

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SOLUTION_PATH = Path("/workspace/solution.py")
TEST_DATA_PATH = Path("/tests/test_data.json")


def main() -> int:
    if not SOLUTION_PATH.is_file():
        print(f"solution not found: {SOLUTION_PATH}", file=sys.stderr)
        return 1

    test_data = json.loads(TEST_DATA_PATH.read_text(encoding="utf-8"))
    inputs = test_data["inputs"]
    expected_outputs = test_data["outputs"]
    if len(inputs) != len(expected_outputs):
        print("test input/output counts do not match", file=sys.stderr)
        return 1

    for index, (input_data, expected_output) in enumerate(
        zip(inputs, expected_outputs, strict=True),
        start=1,
    ):
        try:
            result = subprocess.run(
                [sys.executable, str(SOLUTION_PATH)],
                input=input_data,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except subprocess.TimeoutExpired:
            print(f"test {index} timed out", file=sys.stderr)
            return 1

        if result.returncode != 0:
            print(
                f"test {index} exited with code {result.returncode}: {result.stderr}",
                file=sys.stderr,
            )
            return 1

        expected_lines = expected_output.strip().splitlines()
        actual_lines = result.stdout.strip().splitlines()
        if actual_lines != expected_lines:
            print(
                f"test {index} failed: expected {expected_lines!r}, "
                f"got {actual_lines!r}",
                file=sys.stderr,
            )
            return 1

        print(f"test {index} passed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
