"""Prepare SWE-rebench code-search instances and repository snapshots.

Usage from the repository root::

    uv run python examples/code-search/prepare_data.py --limit 64

The generated data has this layout::

    prepared/
      github.com/<org>/<repo>/git/           # bare Git repository
      github.com/<org>/<repo>/<base_commit>/ # immutable source snapshot
      index/                                 # resumable dataset index
      instances/<instance_id>.json           # private prompt and reward data
      manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import httpx

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_PREPARED_DIR = EXAMPLE_DIR / "prepared"
DATASET = "OpenHands/SWE-rebench-code-search"
DATASET_REVISION = "b4e0432105eef4bf7b2f4cfad41b9167af3f418c"
CONFIG = "default"
SPLIT = "train"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
DATASET_API_URL = f"https://huggingface.co/api/datasets/{DATASET}"
MAX_PAGE_SIZE = 100
MAX_REQUEST_ATTEMPTS = 8
REQUEST_INTERVAL_SECONDS = 0.5
MAX_RETRY_DELAY_SECONDS = 60.0
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")
COMMIT_SHA = re.compile(r"^[0-9a-fA-F]{40}$")


def retry_delay(response: httpx.Response | None, attempt: int) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after is not None:
            try:
                return min(float(retry_after), MAX_RETRY_DELAY_SECONDS)
            except ValueError:
                pass
    return min(2.0**attempt, MAX_RETRY_DELAY_SECONDS)


def fetch_rows(
    client: httpx.Client, *, offset: int, length: int
) -> list[dict[str, Any]]:
    params = {
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "offset": offset,
        "length": length,
    }
    for attempt in range(1, MAX_REQUEST_ATTEMPTS + 1):
        response: httpx.Response | None = None
        try:
            response = client.get(ROWS_URL, params=params)
            if response.status_code != 429 and response.status_code < 500:
                response.raise_for_status()
                payload = response.json()
                return [item["row"] for item in payload["rows"]]
        except httpx.TransportError:
            if attempt == MAX_REQUEST_ATTEMPTS:
                raise
        else:
            if attempt == MAX_REQUEST_ATTEMPTS:
                response.raise_for_status()

        delay = retry_delay(response, attempt)
        status = response.status_code if response is not None else "network error"
        print(
            f"[prepare] rows request at offset {offset} returned {status}; "
            f"retrying in {delay:g}s ({attempt}/{MAX_REQUEST_ATTEMPTS})"
        )
        time.sleep(delay)

    raise AssertionError("unreachable")


def verify_dataset_revision(client: httpx.Client) -> None:
    response = client.get(DATASET_API_URL)
    response.raise_for_status()
    actual_revision = response.json()["sha"]
    if actual_revision != DATASET_REVISION:
        raise RuntimeError(
            "The dataset revision changed: "
            f"expected {DATASET_REVISION}, found {actual_revision}. "
            "Review the new dataset before updating DATASET_REVISION."
        )


def validate_component(value: str, *, label: str) -> str:
    if not SAFE_COMPONENT.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"unsafe {label}: {value!r}")
    return value


def validate_row(row: dict[str, Any]) -> tuple[str, str, str, str]:
    instance_id = validate_component(str(row["instance_id"]), label="instance_id")

    repo_parts = str(row["repo"]).split("/")
    if len(repo_parts) != 2:
        raise ValueError(f"repo must have the form <org>/<name>: {row['repo']!r}")
    org = validate_component(repo_parts[0], label="repository organization")
    repo = validate_component(repo_parts[1], label="repository name")

    base_commit = str(row["base_commit"]).lower()
    if not COMMIT_SHA.fullmatch(base_commit):
        raise ValueError(f"invalid base_commit for {instance_id}: {base_commit!r}")

    if not isinstance(row["problem_statement"], str):
        raise TypeError(f"problem_statement is not a string for {instance_id}")
    if not isinstance(row["file_changes"], list):
        raise TypeError(f"file_changes is not a list for {instance_id}")

    return instance_id, org, repo, base_commit


def run_git(
    *args: str, capture_output: bool = True
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *args],
            check=True,
            text=True,
            capture_output=capture_output,
        )
    except subprocess.CalledProcessError as error:
        stderr = error.stderr.strip() if error.stderr else ""
        command = subprocess.list2cmdline(error.cmd)
        raise RuntimeError(f"Git command failed: {command}\n{stderr}") from error


def commit_exists(git_dir: Path, base_commit: str) -> bool:
    result = subprocess.run(
        ["git", f"--git-dir={git_dir}", "cat-file", "-e", f"{base_commit}^{{commit}}"],
        check=False,
        text=True,
        capture_output=True,
    )
    return result.returncode == 0


def fetch_commit(git_dir: Path, base_commit: str) -> None:
    ref = f"refs/ollie/commits/{base_commit}"
    if not commit_exists(git_dir, base_commit):
        run_git(
            f"--git-dir={git_dir}",
            "fetch",
            "--depth=1",
            "--no-tags",
            "origin",
            f"+{base_commit}:{ref}",
        )
    else:
        run_git(f"--git-dir={git_dir}", "update-ref", ref, base_commit)


def prepare_git_repository(repo_root: Path, org: str, repo: str) -> Path:
    git_dir = repo_root / "git"
    if git_dir.exists():
        result = run_git(f"--git-dir={git_dir}", "rev-parse", "--is-bare-repository")
        if result.stdout.strip() != "true":
            raise RuntimeError(f"existing Git directory is not bare: {git_dir}")
        return git_dir

    repo_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".git.tmp-", dir=repo_root))
    try:
        run_git("init", "--bare", str(temporary))
        run_git(
            f"--git-dir={temporary}",
            "remote",
            "add",
            "origin",
            f"https://github.com/{org}/{repo}.git",
        )
        os.replace(temporary, git_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return git_dir


def materialize_snapshot(repo_root: Path, git_dir: Path, base_commit: str) -> Path:
    snapshot_dir = repo_root / base_commit
    if snapshot_dir.exists():
        if not snapshot_dir.is_dir():
            raise RuntimeError(f"snapshot path is not a directory: {snapshot_dir}")
        return snapshot_dir

    temporary = Path(tempfile.mkdtemp(prefix=f".{base_commit}.tmp-", dir=repo_root))
    archive_path = temporary.with_suffix(".tar")
    try:
        run_git(
            f"--git-dir={git_dir}",
            "archive",
            "--format=tar",
            f"--output={archive_path}",
            base_commit,
        )
        with tarfile.open(archive_path, mode="r:") as archive:
            archive.extractall(temporary, filter="data")
        os.replace(temporary, snapshot_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        archive_path.unlink(missing_ok=True)
    return snapshot_dir


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def instance_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "repo": row["repo"],
        "base_commit": str(row["base_commit"]).lower(),
        "problem_statement": row["problem_statement"],
        "file_changes": row["file_changes"],
    }


def indexed_row(row: dict[str, Any]) -> dict[str, Any]:
    return {"instance_id": row["instance_id"], **instance_payload(row)}


def load_or_build_index(
    client: httpx.Client, prepared_dir: Path
) -> list[dict[str, Any]]:
    index_dir = prepared_dir / "index"
    metadata_path = index_dir / "metadata.json"
    expected_metadata = {
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "config": CONFIG,
        "split": SPLIT,
    }

    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key, expected_value in expected_metadata.items():
            if metadata.get(key) != expected_value:
                raise RuntimeError(
                    f"cached index has unexpected {key}: "
                    f"expected {expected_value!r}, found {metadata.get(key)!r}"
                )
    else:
        metadata = {**expected_metadata, "complete": False, "num_instances": 0}
        write_json(metadata_path, metadata)

    pages_dir = index_dir / "pages"
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        page_path = pages_dir / f"{offset:08d}.json"
        if not page_path.exists():
            break
        page = json.loads(page_path.read_text(encoding="utf-8"))
        if not isinstance(page, list) or not page:
            raise RuntimeError(f"invalid cached index page: {page_path}")
        rows.extend(page)
        offset += len(page)

    if metadata.get("complete"):
        if metadata.get("num_instances") != len(rows):
            raise RuntimeError(
                "completed index metadata does not match its cached page count"
            )
        print(f"[prepare] loaded {len(rows)} instances from the completed index")
        return rows

    if rows:
        print(f"[prepare] resuming dataset index from {len(rows)} cached instances")

    while True:
        page = fetch_rows(client, offset=offset, length=MAX_PAGE_SIZE)
        if not page:
            write_json(
                metadata_path,
                {**expected_metadata, "complete": True, "num_instances": len(rows)},
            )
            return rows

        indexed_page = [indexed_row(row) for row in page]
        write_json(pages_dir / f"{offset:08d}.json", indexed_page)
        rows.extend(indexed_page)
        offset += len(indexed_page)
        print(f"[prepare] indexed {len(rows)} dataset instances")
        time.sleep(REQUEST_INTERVAL_SECONDS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=64,
        help="Number of shuffled repositories to prepare (default: 64).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used to shuffle repositories (default: 42).",
    )
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=DEFAULT_PREPARED_DIR,
        help=f"Output directory (default: {DEFAULT_PREPARED_DIR})",
    )
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be at least 1")

    prepared_dir = args.prepared_dir.resolve()
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        verify_dataset_revision(client)
        rows = load_or_build_index(client, prepared_dir)

    rows_by_repository: dict[tuple[str, str], list[tuple[dict[str, Any], str, str]]] = (
        defaultdict(list)
    )
    seen_instance_ids: set[str] = set()
    for row in rows:
        instance_id, org, repo, base_commit = validate_row(row)
        if instance_id in seen_instance_ids:
            raise ValueError(f"duplicate instance_id in dataset: {instance_id}")
        seen_instance_ids.add(instance_id)
        rows_by_repository[(org, repo)].append((row, instance_id, base_commit))

    repository_keys = sorted(rows_by_repository)
    random.Random(args.seed).shuffle(repository_keys)
    selected_repository_keys = repository_keys[: args.limit]
    if len(selected_repository_keys) < args.limit:
        print(
            f"[prepare] requested {args.limit} repositories, but the dataset only "
            f"contains {len(selected_repository_keys)}"
        )

    repositories: dict[tuple[str, str], set[str]] = defaultdict(set)
    validated_rows: list[tuple[dict[str, Any], str]] = []
    for org, repo in selected_repository_keys:
        for row, instance_id, base_commit in rows_by_repository[(org, repo)]:
            repositories[(org, repo)].add(base_commit)
            validated_rows.append((row, instance_id))

    github_dir = prepared_dir / "github.com"
    for repo_index, (org, repo) in enumerate(selected_repository_keys, start=1):
        commits = repositories[(org, repo)]
        repo_root = github_dir / org / repo
        print(
            f"[prepare] repository {repo_index}/{len(repositories)}: "
            f"{org}/{repo} ({len(commits)} commits)"
        )
        git_dir = prepare_git_repository(repo_root, org, repo)
        for base_commit in sorted(commits):
            fetch_commit(git_dir, base_commit)
            materialize_snapshot(repo_root, git_dir, base_commit)

    instances_dir = prepared_dir / "instances"
    for row, instance_id in validated_rows:
        write_json(instances_dir / f"{instance_id}.json", instance_payload(row))

    write_json(
        prepared_dir / "manifest.json",
        {
            "dataset": DATASET,
            "dataset_revision": DATASET_REVISION,
            "config": CONFIG,
            "split": SPLIT,
            "selection": {
                "seed": args.seed,
                "repository_limit": args.limit,
                "repositories": [
                    f"{org}/{repo}" for org, repo in selected_repository_keys
                ],
            },
            "prepared_instances": len(validated_rows),
            "prepared_repositories": len(repositories),
        },
    )

    print(
        f"[prepare] done: {len(validated_rows)} instances across "
        f"{len(repositories)} repositories under {prepared_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
