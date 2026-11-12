"""Prepare BugUnleashed code-review instances, repo snapshots, and splits.

Usage from the repository root::

    uv run python examples/bug-unleashed/prepare_data.py --limit 4

The generated data has this layout::

    data/
      cases.json                         # stripped dataset snapshot
      splits.json                        # repo-disjoint train/val assignment
      repos/<org>__<repo>/git/           # bare Git repository
      repos/<org>__<repo>/<head_sha>/    # clean snapshot, no .git
      diffs/<case_id>.diff               # PR diff used in the prompt
      instances/<example_id>.json        # one example per gold bug
      manifest.json

Each bug case becomes one example per gold bug (the case's other golds are
attached as siblings); clean cases become a single example with no target.
`--no-snapshots` prepares only the dataset, splits, and example metadata, which
is enough to inspect the pipeline without a GitHub token.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = EXAMPLE_DIR / "data"
CASES_URL = "https://bug-unleashed.netlify.app/data/cases.json"
GITHUB_API = "https://api.github.com"
GITHUB_DIFF_ACCEPT = "application/vnd.github.v3.diff"

# Fixed split parameters (see README "Sampling and splits").
SPLIT_SEED = 20261111
VAL_BUG_EXAMPLES = 28
VAL_CLEAN_EXAMPLES = 5
LANGUAGE_PENALTY = 3

MAX_REQUEST_ATTEMPTS = 5
MAX_RETRY_DELAY_SECONDS = 60.0
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")
COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")

# Fields kept in the committed-free snapshot; everything else is upstream
# model/evaluation data (see README "Dataset").
CASE_FIELDS = (
    "id",
    "repo",
    "pr",
    "title",
    "kind",
    "language",
    "origin",
    "tags",
    "cves",
    "hasCve",
    "advisoryIds",
    "gold",
)
PROVENANCE_FIELDS = (
    "selection",
    "description",
    "sourcePublic",
    "sourceGeneratedAt",
)


# --- Small helpers -----------------------------------------------------------


def log(message: str) -> None:
    print(f"[prepare] {message}", flush=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validate_component(value: str, *, label: str) -> str:
    if not SAFE_COMPONENT.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"unsafe {label}: {value!r}")
    return value


def parse_repo(repo: str) -> tuple[str, str]:
    parts = repo.split("/")
    if len(parts) != 2:
        raise ValueError(f"repo must have the form <org>/<name>: {repo!r}")
    return (
        validate_component(parts[0], label="repository organization"),
        validate_component(parts[1], label="repository name"),
    )


def repo_slug(repo: str) -> str:
    org, name = parse_repo(repo)
    return f"{org}__{name}"


def retry_delay(response: httpx.Response | None, attempt: int) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after is not None:
            try:
                return min(float(retry_after), MAX_RETRY_DELAY_SECONDS)
            except ValueError:
                pass
    return min(2.0**attempt, MAX_RETRY_DELAY_SECONDS)


def github_headers(*, accept: str | None = None) -> dict[str, str]:
    headers = {"Accept": accept or "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def github_get(
    client: httpx.Client, url: str, *, accept: str | None = None
) -> httpx.Response:
    """GET a GitHub API resource with retries and a clear rate-limit error."""
    for attempt in range(1, MAX_REQUEST_ATTEMPTS + 1):
        response: httpx.Response | None = None
        try:
            response = client.get(url, headers=github_headers(accept=accept))
            if response.status_code not in {403, 429} and response.status_code < 500:
                response.raise_for_status()
                return response
        except httpx.TransportError:
            if attempt == MAX_REQUEST_ATTEMPTS:
                raise

        if attempt == MAX_REQUEST_ATTEMPTS:
            if response is not None and response.status_code in {403, 429}:
                raise RuntimeError(
                    f"GitHub API rate limit or permission error for {url}. "
                    "Set GITHUB_TOKEN, or rerun with --no-snapshots."
                )
            if response is not None:
                response.raise_for_status()
            raise RuntimeError(f"GitHub request failed after retries: {url}")

        delay = retry_delay(response, attempt)
        status = response.status_code if response is not None else "network error"
        log(
            f"{url} returned {status}; retrying in {delay:g}s ({attempt}/{MAX_REQUEST_ATTEMPTS})"
        )
        time.sleep(delay)

    raise AssertionError("unreachable")


# --- Dataset -----------------------------------------------------------------


def strip_cases(raw: dict[str, Any]) -> dict[str, Any]:
    """Drop every upstream model/evaluation field from the dataset snapshot."""
    cases = []
    for case in raw["cases"]:
        missing = [field for field in CASE_FIELDS if field not in case]
        if missing:
            raise ValueError(f"case {case.get('id')!r} is missing {missing}")
        cases.append({field: case[field] for field in CASE_FIELDS})

    provenance = raw.get("provenance", {})
    return {
        "generatedAt": raw.get("generatedAt"),
        "provenance": {
            field: provenance[field]
            for field in PROVENANCE_FIELDS
            if field in provenance
        },
        "counts": raw.get("counts"),
        "cases": cases,
    }


def load_cases(
    client: httpx.Client, data_dir: Path, *, refresh: bool
) -> dict[str, Any]:
    path = data_dir / "cases.json"
    if path.exists() and not refresh:
        log(f"using cached {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    log(f"downloading {CASES_URL}")
    response = client.get(CASES_URL)
    response.raise_for_status()
    stripped = strip_cases(response.json())
    write_json(path, stripped)
    log(f"wrote {path} ({len(stripped['cases'])} cases)")
    return stripped


def validate_cases(cases: list[dict[str, Any]]) -> None:
    seen_ids: set[str] = set()
    for case in cases:
        case_id = validate_component(str(case["id"]), label="case id")
        if case_id in seen_ids:
            raise ValueError(f"duplicate case id: {case_id}")
        seen_ids.add(case_id)
        parse_repo(case["repo"])
        if not isinstance(case["pr"], int):
            raise TypeError(f"case {case_id} has a non-integer pr")
        if case["kind"] not in {"bug", "clean"}:
            raise ValueError(f"case {case_id} has unexpected kind {case['kind']!r}")
        gold = case["gold"]
        if not isinstance(gold, list):
            raise TypeError(f"case {case_id} gold is not a list")
        if case["kind"] == "bug" and not gold:
            raise ValueError(f"bug case {case_id} has no gold")
        if case["kind"] == "clean" and gold:
            raise ValueError(f"clean case {case_id} has gold")
        for item in gold:
            if not item.get("summary"):
                raise ValueError(f"case {case_id} has a gold without a summary")
            if not item.get("locations"):
                raise ValueError(f"case {case_id} has a gold without locations")


# --- Splits ------------------------------------------------------------------


def primary_language(case: dict[str, Any]) -> str:
    # `language` is a single string in this dataset; keep the helper tolerant of
    # a comma-separated value in case the upstream format changes.
    return case["language"].split(",")[0].strip()


def assign_splits(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Greedy repo-disjoint assignment, deterministic for a fixed case order."""
    # Counts are in *examples*: one per gold bug, one per clean case. This keeps
    # the split aligned with the README table and with what training sees.
    repos: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"bug": 0, "clean": 0, "languages": Counter()}
    )
    for case in cases:
        entry = repos[case["repo"]]
        entry[case["kind"]] += len(case["gold"]) if case["kind"] == "bug" else 1
        entry["languages"][primary_language(case)] += 1

    order = sorted(
        repos,
        key=lambda repo: (-(repos[repo]["bug"] + repos[repo]["clean"]), repo),
    )

    targets = {
        "val": {"bug": VAL_BUG_EXAMPLES, "clean": VAL_CLEAN_EXAMPLES},
        "train": {
            "bug": sum(entry["bug"] for entry in repos.values()) - VAL_BUG_EXAMPLES,
            "clean": sum(entry["clean"] for entry in repos.values())
            - VAL_CLEAN_EXAMPLES,
        },
    }
    current = {split: {"bug": 0, "clean": 0} for split in targets}
    language_counts: dict[str, Counter[str]] = {split: Counter() for split in targets}
    assignment: dict[str, str] = {}

    for repo in order:
        entry = repos[repo]
        language = entry["languages"].most_common(1)[0][0]
        best: tuple[float, str] | None = None
        for split in ("val", "train"):
            need = sum(
                targets[split][kind] - current[split][kind] for kind in targets[split]
            )
            if need <= 0:
                continue
            score = need - LANGUAGE_PENALTY * language_counts[split][language]
            if best is None or score > best[0]:
                best = (score, split)
        if best is None:
            raise RuntimeError(f"no split has capacity left for {repo}")
        split = best[1]
        assignment[repo] = split
        for kind in ("bug", "clean"):
            current[split][kind] += entry[kind]
        language_counts[split][language] += 1

    return {
        "seed": SPLIT_SEED,
        "strategy": "repo-disjoint greedy; largest repos first; fewest primary-language repos preferred",
        "targets": targets,
        "counts": current,
        "languages": {split: dict(counts) for split, counts in language_counts.items()},
        "repos": {
            split: sorted(repo for repo, value in assignment.items() if value == split)
            for split in targets
        },
        "assignment": dict(sorted(assignment.items())),
    }


# --- Git and GitHub ----------------------------------------------------------


# Git LFS would otherwise download the real binaries: `git archive` runs the
# smudge filter, so a 133-byte pointer blob becomes a 36 MB asset on disk. A
# reviewer only needs the source tree, so keep the pointers.
GIT_ENV = {
    **os.environ,
    "GIT_LFS_SKIP_SMUDGE": "1",
    "GIT_TERMINAL_PROMPT": "0",
}

# Belt and braces: disable the LFS filter for the archive itself, in case the
# environment variable is not honoured by the installed git-lfs version.
LFS_FILTER_OVERRIDES = (
    "-c",
    "filter.lfs.process=",
    "-c",
    "filter.lfs.smudge=",
    "-c",
    "filter.lfs.clean=",
    "-c",
    "filter.lfs.required=false",
)


def run_git(*args: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *args],
            check=True,
            text=True,
            capture_output=True,
            env=GIT_ENV,
        )
    except subprocess.CalledProcessError as error:
        stderr = error.stderr.strip() if error.stderr else ""
        command = subprocess.list2cmdline(error.cmd)
        raise RuntimeError(f"Git command failed: {command}\n{stderr}") from error


def commit_exists(git_dir: Path, commit: str) -> bool:
    result = subprocess.run(
        ["git", f"--git-dir={git_dir}", "cat-file", "-e", f"{commit}^{{commit}}"],
        check=False,
        text=True,
        capture_output=True,
    )
    return result.returncode == 0


def prepare_bare_repo(repo_root: Path, repo: str) -> Path:
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
            f"https://github.com/{repo}.git",
        )
        os.replace(temporary, git_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return git_dir


def fetch_commit(git_dir: Path, commit: str) -> None:
    if commit_exists(git_dir, commit):
        return
    run_git(
        f"--git-dir={git_dir}",
        "fetch",
        "--depth=1",
        "--no-tags",
        "origin",
        f"+{commit}:refs/ollie/commits/{commit}",
    )


def snapshot_member_filter(skipped: list[str]) -> Any:
    """Extraction filter that drops entries the `data` filter rejects.

    Some trees (e.g. next.js test fixtures) contain symlinks to absolute host
    paths. They are meaningless inside a snapshot, so skip them instead of
    failing the whole case.
    """

    def filter_member(member: tarfile.TarInfo, destination: str) -> Any:
        try:
            return tarfile.data_filter(member, destination)
        except tarfile.FilterError:
            skipped.append(member.name)
            return None

    return filter_member


def materialize_snapshot(repo_root: Path, git_dir: Path, commit: str) -> Path:
    snapshot_dir = repo_root / commit
    if snapshot_dir.is_dir():
        return snapshot_dir

    temporary = Path(tempfile.mkdtemp(prefix=f".{commit}.tmp-", dir=repo_root))
    archive_path = temporary.with_suffix(".tar")
    try:
        run_git(
            *LFS_FILTER_OVERRIDES,
            f"--git-dir={git_dir}",
            "archive",
            "--format=tar",
            f"--output={archive_path}",
            commit,
        )
        skipped: list[str] = []
        with tarfile.open(archive_path, mode="r:") as archive:
            archive.extractall(temporary, filter=snapshot_member_filter(skipped))
        if skipped:
            log(f"{commit[:12]}: skipped {len(skipped)} unsafe archive entries")
        os.replace(temporary, snapshot_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        archive_path.unlink(missing_ok=True)
    return snapshot_dir


def fetch_pr(client: httpx.Client, repo: str, pr: int) -> dict[str, Any]:
    """Return the PR metadata and diff the prompt needs."""
    metadata = github_get(client, f"{GITHUB_API}/repos/{repo}/pulls/{pr}").json()
    head_sha = str(metadata["head"]["sha"]).lower()
    base_sha = str(metadata["base"]["sha"]).lower()
    for sha, label in ((head_sha, "head"), (base_sha, "base")):
        if not COMMIT_SHA.fullmatch(sha):
            raise ValueError(f"invalid {label} sha for {repo}#{pr}: {sha!r}")

    diff = github_get(
        client, f"{GITHUB_API}/repos/{repo}/pulls/{pr}", accept=GITHUB_DIFF_ACCEPT
    ).text
    if not diff.strip():
        raise ValueError(f"empty diff for {repo}#{pr}")

    return {
        "title": metadata.get("title") or "",
        "body": metadata.get("body") or "",
        "head_sha": head_sha,
        "base_sha": base_sha,
        "diff": diff,
    }


# --- Examples ----------------------------------------------------------------


def gold_paths(gold: dict[str, Any]) -> list[str]:
    return sorted({location["path"] for location in gold["locations"]})


def build_examples(
    cases: list[dict[str, Any]], assignment: dict[str, str]
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for case in cases:
        split = assignment[case["repo"]]
        base = {
            "case_id": case["id"],
            "repo": case["repo"],
            "pr": case["pr"],
            "title": case["title"],
            "language": case["language"],
            "split": split,
            "kind": case["kind"],
        }
        if case["kind"] == "clean":
            examples.append(
                {
                    **base,
                    "example_id": case["id"],
                    "target": None,
                    "siblings": [],
                }
            )
            continue

        golds = case["gold"]
        for index, gold in enumerate(golds, start=1):
            siblings = [
                {
                    "gid": other["gid"],
                    "summary": other["summary"],
                    "paths": gold_paths(other),
                }
                for other in golds
                if other is not gold
            ]
            examples.append(
                {
                    **base,
                    "example_id": f"{case['id']}__g{index}",
                    "target": {
                        "gid": gold["gid"],
                        "summary": gold["summary"],
                        "paths": gold_paths(gold),
                    },
                    "siblings": siblings,
                }
            )
    return examples


class Progress:
    """Thread-safe case counter and failure log."""

    def __init__(self, total: int) -> None:
        self.total = total
        self.done = 0
        self.failures: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    def record(self, label: str, error: BaseException | None = None) -> None:
        with self._lock:
            self.done += 1
            if error is None:
                log(f"case {self.done}/{self.total}: {label}")
            else:
                self.failures.append((label, str(error)))
                log(f"case {self.done}/{self.total}: {label} FAILED: {error}")

    def record_repo(self, repo: str, cases: int, error: BaseException) -> None:
        with self._lock:
            self.done += cases
            self.failures.append((repo, str(error)))
            log(f"{repo}: repository setup FAILED ({cases} cases): {error}")


def prepare_repo(
    client: httpx.Client,
    data_dir: Path,
    repo: str,
    repo_cases: list[dict[str, Any]],
    examples_by_case: dict[str, list[dict[str, Any]]],
    progress: Progress,
) -> None:
    """Fetch metadata, diffs, and one snapshot per case for a single repo.

    Repos run in parallel; cases within a repo run sequentially because they
    share one bare Git directory.
    """
    repo_root = data_dir / "repos" / repo_slug(repo)
    git_dir = prepare_bare_repo(repo_root, repo)
    for case in repo_cases:
        label = f"{case['id']} ({repo}#{case['pr']})"
        try:
            pr = fetch_pr(client, repo, case["pr"])
            write_json(
                data_dir / "pr" / f"{case['id']}.json",
                {
                    "title": pr["title"],
                    "body": pr["body"],
                    "head_sha": pr["head_sha"],
                    "base_sha": pr["base_sha"],
                },
            )
            diff_path = data_dir / "diffs" / f"{case['id']}.diff"
            diff_path.parent.mkdir(parents=True, exist_ok=True)
            diff_path.write_text(pr["diff"], encoding="utf-8")

            fetch_commit(git_dir, pr["head_sha"])
            materialize_snapshot(repo_root, git_dir, pr["head_sha"])

            for example in examples_by_case[case["id"]]:
                example["head_sha"] = pr["head_sha"]
                example["base_sha"] = pr["base_sha"]
                example["snapshot"] = str(
                    (repo_root / pr["head_sha"]).relative_to(data_dir)
                )
                example["diff"] = str(diff_path.relative_to(data_dir))
        except Exception as error:  # noqa: BLE001 - reported per case
            progress.record(label, error)
        else:
            progress.record(label)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Output directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--refresh-cases",
        action="store_true",
        help="Re-download and re-strip cases.json even if a cached copy exists.",
    )
    parser.add_argument(
        "--no-snapshots",
        action="store_true",
        help="Skip PR metadata, diffs, and repository snapshots (no GitHub access).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Prepare only the first N cases (for a smoke run).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Repositories prepared in parallel (default: 5).",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")

    data_dir = args.data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        dataset = load_cases(client, data_dir, refresh=args.refresh_cases)
        cases = dataset["cases"]
        if args.limit is not None:
            cases = cases[: args.limit]
        validate_cases(cases)

        splits = assign_splits(cases)
        write_json(data_dir / "splits.json", splits)
        log(
            f"split {len(cases)} cases into train/val: "
            f"{splits['counts']['train']} / {splits['counts']['val']}"
        )

        examples = build_examples(cases, splits["assignment"])
        examples_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for example in examples:
            examples_by_case[example["case_id"]].append(example)

        prepared_repos: set[str] = set()
        failures: list[tuple[str, str]] = []
        if not args.no_snapshots:
            cases_by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for case in cases:
                cases_by_repo[case["repo"]].append(case)

            progress = Progress(len(cases))
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = {
                    pool.submit(
                        prepare_repo,
                        client,
                        data_dir,
                        repo,
                        repo_cases,
                        examples_by_case,
                        progress,
                    ): (repo, repo_cases)
                    for repo, repo_cases in sorted(cases_by_repo.items())
                }
                for future in as_completed(futures):
                    repo, repo_cases = futures[future]
                    try:
                        future.result()
                    except Exception as error:  # noqa: BLE001 - reported per repo
                        progress.record_repo(repo, len(repo_cases), error)

            failures = progress.failures
            failed_repos = {label for label, _ in failures}
            prepared_repos = {
                case["repo"]
                for case in cases
                if case["repo"] not in failed_repos and case["id"] not in failed_repos
            }

        instances_dir = data_dir / "instances"
        for example in examples:
            write_json(instances_dir / f"{example['example_id']}.json", example)

        write_json(
            data_dir / "manifest.json",
            {
                "generatedAt": dataset.get("generatedAt"),
                "cases": len(cases),
                "examples": len(examples),
                "splits": {
                    split: {
                        "repos": len(splits["repos"][split]),
                        **splits["counts"][split],
                    }
                    for split in splits["repos"]
                },
                "snapshots": not args.no_snapshots,
                "prepared_repos": len(prepared_repos),
                "failures": [
                    {"case": label, "error": error} for label, error in failures
                ],
            },
        )

    log(
        f"done: {len(examples)} examples across "
        f"{len({case['repo'] for case in cases})} repos under {data_dir}"
    )
    if failures:
        log(f"{len(failures)} case(s) failed; see manifest.json")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
