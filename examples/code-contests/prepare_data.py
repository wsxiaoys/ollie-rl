"""Convert the ``open-thoughts/CodeContests`` dataset into local Harbor tasks.

Each row of the dataset is a pair ``{path, task_binary}`` where ``task_binary``
is a **gzip-compressed tarball of a complete Harbor task directory**::

    code_contests-0000/
    ├── task.toml
    ├── instruction.md
    ├── environment/
    │   └── Dockerfile
    └── tests/
        ├── test.sh
        ├── test_state.py
        └── test_data.json

So "converting to Harbor format" is really just *extracting* each tarball into
``examples/code-contests/tasks/<path>/``. Every extracted directory is one
Harbor task, and its ``path`` (e.g. ``code_contests-0000``) is used verbatim as
an ollie-rl ``datum_id`` by ``run_training.py``.

This script pulls rows over the public Hugging Face datasets-server HTTP API
(``/rows``), which returns ``task_binary`` as base64 — no ``datasets`` /
``huggingface_hub`` dependency required, just ``httpx`` (already a repo dep).

Usage (from the repo root)::

    uv run python examples/code-contests/prepare_data.py --limit 64
"""

from __future__ import annotations

import argparse
import base64
import io
import tarfile
from pathlib import Path

import httpx

EXAMPLE_DIR = Path(__file__).resolve().parent
TASKS_DIR = EXAMPLE_DIR / "tasks"

DATASET = "open-thoughts/CodeContests"
CONFIG = "default"
SPLIT = "train"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
MAX_PAGE_SIZE = 100  # datasets-server caps `length` at 100 rows per request.


def fetch_rows(client: httpx.Client, offset: int, length: int) -> list[dict]:
    resp = client.get(
        ROWS_URL,
        params={
            "dataset": DATASET,
            "config": CONFIG,
            "split": SPLIT,
            "offset": offset,
            "length": length,
        },
    )
    resp.raise_for_status()
    return resp.json()["rows"]


def extract_task(task_binary_b64: str, dest: Path) -> None:
    """Decode + gunzip + untar one task tarball into ``dest``."""
    raw = base64.b64decode(task_binary_b64)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        # ``filter="data"`` (py3.12+) blocks path traversal / unsafe members.
        tar.extractall(dest, filter="data")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=64,
        help="How many tasks to extract (dataset has thousands).",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Row offset to start from (useful for a held-out eval slice).",
    )
    args = parser.parse_args()

    TASKS_DIR.mkdir(parents=True, exist_ok=True)

    extracted = 0
    offset = args.offset
    with httpx.Client(timeout=60.0) as client:
        while extracted < args.limit:
            length = min(MAX_PAGE_SIZE, args.limit - extracted)
            rows = fetch_rows(client, offset, length)
            if not rows:
                print(f"[prepare] dataset exhausted after {extracted} tasks")
                break
            for item in rows:
                row = item["row"]
                path = row["path"]
                dest = TASKS_DIR / path
                dest.mkdir(parents=True, exist_ok=True)
                extract_task(row["task_binary"], dest)
                extracted += 1
                print(f"[prepare] {extracted:04d}  {path}  ->  {dest}")
            offset += len(rows)

    print(f"[prepare] done: {extracted} Harbor tasks under {TASKS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
