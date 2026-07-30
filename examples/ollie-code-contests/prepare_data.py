"""Extract CodeContests as Harbor tasks adapted for the Ollie agent.

Usage from the repository root::

    uv run python examples/ollie-code-contests/prepare_data.py --limit 64
"""

from __future__ import annotations

import argparse
import base64
import io
import re
import tarfile
import tomllib
from pathlib import Path

import httpx

EXAMPLE_DIR = Path(__file__).resolve().parent
TASKS_DIR = EXAMPLE_DIR / "tasks"
DATASET = "open-thoughts/CodeContests"
CONFIG = "default"
SPLIT = "train"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
MAX_PAGE_SIZE = 100


def fetch_rows(client: httpx.Client, offset: int, length: int) -> list[dict]:
    response = client.get(
        ROWS_URL,
        params={
            "dataset": DATASET,
            "config": CONFIG,
            "split": SPLIT,
            "offset": offset,
            "length": length,
        },
    )
    response.raise_for_status()
    return response.json()["rows"]


def extract_task(task_binary_b64: str, destination: Path) -> None:
    raw = base64.b64decode(task_binary_b64)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        archive.extractall(destination, filter="data")


def _set_toml_value(
    source: str, *, section: str | None, key: str, value: str
) -> str:
    lines = source.splitlines()
    if section is None:
        start = 0
        end = next(
            (i for i, line in enumerate(lines) if line.strip().startswith("[")),
            len(lines),
        )
    else:
        header = f"[{section}]"
        header_index = next(
            (i for i, line in enumerate(lines) if line.strip() == header), None
        )
        if header_index is None:
            if lines and lines[-1]:
                lines.append("")
            lines.extend([header, f"{key} = {value}"])
            return "\n".join(lines) + "\n"
        start = header_index + 1
        end = next(
            (i for i in range(start, len(lines)) if lines[i].strip().startswith("[")),
            len(lines),
        )

    key_pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
    for index in range(start, end):
        if key_pattern.match(lines[index]):
            lines[index] = f"{key} = {value}"
            return "\n".join(lines) + "\n"

    lines.insert(start, f"{key} = {value}")
    return "\n".join(lines) + "\n"


def adapt_task_for_ollie(task_dir: Path) -> None:
    """Configure artifact transfer and separate verification for Ollie."""
    config_path = task_dir / "task.toml"
    source = config_path.read_text(encoding="utf-8")
    config = tomllib.loads(source)
    artifacts = config.get("artifacts", [])
    if artifacts and artifacts != ["/app"]:
        raise ValueError(
            f"{config_path} declares unsupported artifacts; refusing to overwrite them"
        )

    source = _set_toml_value(
        source, section=None, key="artifacts", value='["/app"]'
    )
    source = _set_toml_value(
        source, section="environment", key="workdir", value='"/app"'
    )
    source = _set_toml_value(
        source,
        section="verifier",
        key="environment_mode",
        value='"separate"',
    )
    source = _set_toml_value(
        source,
        section="verifier.environment",
        key="docker_image",
        value='"python:3.12-slim"',
    )
    source = _set_toml_value(
        source,
        section="verifier.environment",
        key="workdir",
        value='"/app"',
    )
    tomllib.loads(source)
    config_path.write_text(source, encoding="utf-8")


def repair_existing_tasks() -> int:
    task_dirs = sorted(
        path
        for path in TASKS_DIR.iterdir()
        if path.is_dir() and (path / "task.toml").exists()
    )
    for index, task_dir in enumerate(task_dirs, start=1):
        adapt_task_for_ollie(task_dir)
        print(f"[prepare] repaired {index:04d}  {task_dir.name}")
    print(f"[prepare] done: repaired {len(task_dirs)} existing tasks")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--repair-existing",
        action="store_true",
        help="Re-adapt downloaded tasks without fetching the dataset again.",
    )
    args = parser.parse_args()

    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    if args.repair_existing:
        return repair_existing_tasks()

    extracted = 0
    offset = args.offset
    with httpx.Client(timeout=60.0) as client:
        while extracted < args.limit:
            rows = fetch_rows(
                client, offset, min(MAX_PAGE_SIZE, args.limit - extracted)
            )
            if not rows:
                print(f"[prepare] dataset exhausted after {extracted} tasks")
                break
            for item in rows:
                row = item["row"]
                destination = TASKS_DIR / row["path"]
                destination.mkdir(parents=True, exist_ok=True)
                extract_task(row["task_binary"], destination)
                adapt_task_for_ollie(destination)
                extracted += 1
                print(
                    f"[prepare] {extracted:04d}  {row['path']}  ->  {destination}"
                )
            offset += len(rows)

    print(f"[prepare] done: {extracted} Ollie Harbor tasks under {TASKS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
