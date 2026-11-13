"""Judge and reward for the BugUnleashed code-review example.

The reward is example-level and has three parts, all in this module:

1. **Parse** the rollout's `<review_findings>` element into at most 3 valid
   findings (invalid findings are dropped, extras truncated).
2. **Gate** each finding against the example's target gold by file path only.
3. **Judge** the gated candidates with a small binary classifier and score
   the example with the fixed reward table.

Usage::

    # Score one recorded rollout (debugging).
    uv run python examples/bug-unleashed/reward.py \
        --example-id actualbudget__actual-6667__g1 --output rollout.ndjson

    # Build a starter calibration set, hand-label it, then check the judge.
    uv run python examples/bug-unleashed/reward.py --make-pairs pairs.jsonl
    uv run python examples/bug-unleashed/reward.py --pairs pairs.jsonl

`run_training.py` imports `load_examples`, `parse_findings`, `Judge` and
`score_example` from here.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import posixpath
import random
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import httpx

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = EXAMPLE_DIR / "data"
DEFAULT_JUDGE_MODEL = os.environ.get("BUG_UNLEASHED_JUDGE_MODEL", "gpt-4o-mini")
DEFAULT_JUDGE_BASE_URL = os.environ.get(
    "BUG_UNLEASHED_JUDGE_BASE_URL", "https://api.openai.com/v1"
)
JUDGE_TIMEOUT_SECONDS = 60.0
# The gateway honours `seed`, which makes verdicts reproducible; a negative
# value disables it (the model is otherwise non-deterministic even at
# temperature 0, by 1-2 pairs per 44).
DEFAULT_JUDGE_SEED = 0

# Fixed by the README reward table.
MAX_FINDINGS = 3
MAJOR_MATCH_REWARD = 1.0
MINOR_MATCH_REWARD = 0.25
MISS_REWARD = -1.0
CLEAN_MAJOR_PENALTY = -5.0
CLEAN_REWARD = 0.0

SEVERITIES = ("major", "minor")
FINDINGS_RE = re.compile(r"<review_findings>(.*?)</review_findings>", re.DOTALL)
JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}")

JUDGE_SYSTEM_PROMPT = """You decide whether a code-review finding identifies the same defect as a
known gold bug.

You will be given:
- GOLD: the gold bug's summary, plus the files and short notes where it lives.
- FINDING: one finding produced by a reviewer (file, line, title, body).

Answer true only if the finding describes the same defect as GOLD: the same
mechanism in the same file. A finding still matches when it names a narrower
or broader trigger instance of that mechanism, or explains the cause
differently, as long as the defect it describes is GOLD's.

Answer false if the finding:
- only names a category or a general risk ("may cause a null dereference",
  "possible race condition") without the specific mechanism;
- restates the PR's purpose, title, or description instead of identifying a
  defect;
- reports a different problem that happens to be in the same file;
- is a style, naming, or formatting nit;
- names the right bug but in a file not listed under GOLD.

A finding that identifies GOLD's defect but explains its cause incorrectly
still matches: the defect, not the explanation, is what counts. A finding
that only says the PR changes a check or a code path, without stating what
goes wrong as a result, is not a defect report.

Do not reward confidence, length, or specificity in general — only the match
to GOLD's mechanism and defect. If GOLD lists several files, a finding in
any one of them can match.

Reply with JSON only: {"same_bug": true} or {"same_bug": false}"""


# --- Data --------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    severity: str
    path: str
    line: int
    title: str
    body: str


@dataclass(frozen=True)
class Gold:
    gid: str
    summary: str
    paths: tuple[str, ...]
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Example:
    example_id: str
    kind: str
    target: Gold | None
    siblings: tuple[Gold, ...] = ()
    title: str = ""


@dataclass(frozen=True)
class RewardResult:
    total: float
    matched: bool
    matched_severity: str | None
    judge_calls: int
    errors: tuple[str, ...] = ()

    @property
    def strict_match(self) -> bool:
        """The reported eval metric: matched *and* committed as a major."""
        return self.matched and self.matched_severity == "major"


def _gold(payload: dict[str, Any]) -> Gold:
    return Gold(
        gid=payload["gid"],
        summary=payload["summary"],
        paths=tuple(payload["paths"]),
        notes=tuple(payload.get("notes") or ()),
    )


def load_examples(data_dir: Path = DEFAULT_DATA_DIR) -> dict[str, Example]:
    instances_dir = data_dir / "instances"
    if not instances_dir.is_dir():
        raise ValueError(
            f"prepared instances not found: {instances_dir}; run prepare_data.py first"
        )

    examples: dict[str, Example] = {}
    for path in sorted(instances_dir.glob("*.json")):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
            example_id = item["example_id"]
            kind = item["kind"]
            target = item["target"]
            siblings = item.get("siblings") or []
            title = item.get("title") or ""
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid prepared example: {path}") from error
        if kind not in {"bug", "clean"}:
            raise ValueError(f"example {example_id} has unexpected kind {kind!r}")
        if (kind == "bug") != isinstance(target, dict):
            raise ValueError(f"example {example_id} target does not match its kind")
        examples[example_id] = Example(
            example_id=example_id,
            kind=kind,
            target=_gold(target) if isinstance(target, dict) else None,
            siblings=tuple(_gold(sibling) for sibling in siblings),
            title=title,
        )

    if not examples:
        raise ValueError(f"no prepared examples found under {instances_dir}")
    return examples


# --- Parsing -----------------------------------------------------------------


def _string_values(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _string_values(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _string_values(nested)


def normalize_path(path: str) -> str:
    normalized = posixpath.normpath(path.strip())
    return normalized


def parse_finding(value: Any) -> Finding:
    """Validate one finding object; raises on anything the README rejects."""
    if not isinstance(value, dict):
        raise TypeError("finding is not an object")

    severity = value.get("severity")
    if severity not in SEVERITIES:
        raise ValueError(f"severity is not one of {SEVERITIES}: {severity!r}")

    path = value.get("path")
    if not isinstance(path, str) or not path.strip():
        raise TypeError("path is not a non-empty string")
    if posixpath.isabs(path) or ".." in path.split("/"):
        raise ValueError(f"path is not repository-relative: {path!r}")

    line = value.get("line")
    if isinstance(line, bool) or not isinstance(line, (int, float)):
        raise TypeError("line is not an integer")
    if isinstance(line, float):
        if not line.is_integer():
            raise TypeError("line is not an integer")
        line = int(line)
    if line < 1:
        raise ValueError(f"line is not positive: {line}")

    title = value.get("title")
    body = value.get("body")
    if not isinstance(title, str) or not title.strip():
        raise TypeError("title is not a non-empty string")
    if not isinstance(body, str) or not body.strip():
        raise TypeError("body is not a non-empty string")

    return Finding(
        severity=severity,
        path=normalize_path(path),
        line=line,
        title=title.strip(),
        body=body.strip(),
    )


def parse_findings(output: str) -> tuple[tuple[Finding, ...], tuple[str, ...]]:
    """Return the findings from the last `<review_findings>` element.

    Invalid findings are dropped (and reported) rather than failing the whole
    rollout; anything beyond the first `MAX_FINDINGS` valid findings is
    truncated in output order.
    """
    payloads: list[str] = []
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            values: Iterable[str] = (line,)
        else:
            values = _string_values(event)
        for value in values:
            payloads.extend(match.group(1) for match in FINDINGS_RE.finditer(value))

    errors: list[str] = []
    if not payloads:
        return (), ("no <review_findings> element in the agent output",)

    raw = payloads[-1]
    try:
        items = json.loads(raw)
    except json.JSONDecodeError as error:
        return (), (f"invalid <review_findings> JSON: {error.msg}",)
    if not isinstance(items, list):
        return (), ("<review_findings> contents are not a JSON array",)

    findings: list[Finding] = []
    for index, item in enumerate(items):
        if len(findings) >= MAX_FINDINGS:
            errors.append(
                f"dropped {len(items) - index} finding(s) beyond the {MAX_FINDINGS} cap"
            )
            break
        try:
            findings.append(parse_finding(item))
        except (TypeError, ValueError) as error:
            errors.append(f"dropped finding {index}: {error}")
    return tuple(findings), tuple(errors)


# --- Matching ----------------------------------------------------------------


def path_matches(finding: Finding, gold: Gold) -> bool:
    """The deterministic location gate: file-level, line numbers ignored."""
    return finding.path in {normalize_path(path) for path in gold.paths}


def gated_candidates(
    findings: Sequence[Finding], gold: Gold, severity: str
) -> list[Finding]:
    """Findings of `severity` that pass the path gate, deduplicated."""
    gated = (
        finding
        for finding in findings
        if finding.severity == severity and path_matches(finding, gold)
    )
    return list(dict.fromkeys(gated))


def judge_user_message(gold: Gold, finding: Finding) -> str:
    return (
        f"GOLD:\n{gold.summary}\n"
        f"Files: {', '.join(gold.paths)}\n"
        f"Notes: {'; '.join(gold.notes)}\n"
        f"\n"
        f"FINDING:\n"
        f"File: {finding.path}\n"
        f"Line: {finding.line}\n"
        f"Title: {finding.title}\n"
        f"Body: {finding.body}"
    )


def parse_verdict(text: str) -> bool | None:
    """Read `{"same_bug": bool}` out of a model reply, tolerating fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    payload: Any = None
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = JSON_OBJECT_RE.search(cleaned)
        if match is not None:
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                payload = None

    if not isinstance(payload, dict):
        return None
    verdict = payload.get("same_bug")
    return verdict if isinstance(verdict, bool) else None


# --- Judge -------------------------------------------------------------------


class Judge:
    """Small binary classifier over an OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_JUDGE_MODEL,
        base_url: str = DEFAULT_JUDGE_BASE_URL,
        api_key: str | None = None,
        temperature: float = 0.0,
        seed: int | None = DEFAULT_JUDGE_SEED,
        votes: int = 1,
        timeout: float = JUDGE_TIMEOUT_SECONDS,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "no judge API key: set OPENAI_API_KEY or pass --judge-api-key"
            )
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.votes = max(1, votes)
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        self.calls = 0
        self.failures = 0

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def _one_vote(self, gold: Gold, finding: Finding) -> tuple[bool, str | None]:
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": judge_user_message(gold, finding)},
            ],
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        try:
            response = await self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as error:
            self.failures += 1
            return False, f"judge call failed: {error}"

        if not isinstance(content, str):
            self.failures += 1
            return False, "judge reply has no text content"
        verdict = parse_verdict(content)
        if verdict is None:
            self.failures += 1
            return False, f"unparseable judge reply: {content[:200]!r}"
        return verdict, None

    async def same_bug(self, gold: Gold, finding: Finding) -> tuple[bool, str | None]:
        """Return `(verdict, error)`. Any failure counts as `False` (a miss).

        The backend is not deterministic even at temperature 0 (and `seed` is
        not always honoured), so `votes > 1` takes a majority over that many
        calls; a tie or any failure votes `False`.
        """
        self.calls += self.votes
        results = await asyncio.gather(
            *(self._one_vote(gold, finding) for _ in range(self.votes))
        )
        errors = [error for _, error in results if error]
        yes = sum(1 for verdict, _ in results if verdict)
        return yes * 2 > self.votes, (errors[0] if errors else None)


# --- Reward ------------------------------------------------------------------


async def score_example(
    example: Example, findings: Sequence[Finding], judge: Judge
) -> RewardResult:
    """Score one rollout against the example's target gold."""
    errors: list[str] = []

    if example.kind == "clean":
        majors = [finding for finding in findings if finding.severity == "major"]
        if majors:
            errors.append(f"{len(majors)} major finding(s) on a clean PR")
            return RewardResult(
                total=CLEAN_MAJOR_PENALTY,
                matched=False,
                matched_severity=None,
                judge_calls=0,
                errors=tuple(errors),
            )
        return RewardResult(
            total=CLEAN_REWARD,
            matched=False,
            matched_severity=None,
            judge_calls=0,
            errors=tuple(errors),
        )

    target = example.target
    if target is None:
        raise ValueError(f"bug example {example.example_id} has no target gold")

    start_calls = judge.calls

    # Majors first: a major match is worth 1.0 and needs no minor judging.
    for severity, reward in (
        ("major", MAJOR_MATCH_REWARD),
        ("minor", MINOR_MATCH_REWARD),
    ):
        for finding in gated_candidates(findings, target, severity):
            matched, error = await judge.same_bug(target, finding)
            if error:
                errors.append(error)
            if matched:
                return RewardResult(
                    total=reward,
                    matched=True,
                    matched_severity=severity,
                    judge_calls=judge.calls - start_calls,
                    errors=tuple(errors),
                )

    return RewardResult(
        total=MISS_REWARD,
        matched=False,
        matched_severity=None,
        judge_calls=judge.calls - start_calls,
        errors=tuple(errors),
    )


# --- Calibration helpers -----------------------------------------------------


GENERIC_TEMPLATES = (
    "This change may cause a null dereference at runtime.",
    "Possible race condition introduced by this change.",
    "The new code path is not covered by tests.",
    "This could break error handling in edge cases.",
)


def _short_title(text: str, limit: int = 120) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit].rsplit(" ", 1)[0] + "..."


def _other_gold(
    examples: dict[str, Example], gid: str, rng: random.Random
) -> Gold | None:
    candidates = [
        example.target
        for example in examples.values()
        if example.kind == "bug"
        and example.target is not None
        and example.target.gid != gid
    ]
    return rng.choice(candidates) if candidates else None


def candidate_pairs(
    examples: dict[str, Example], *, limit: int, seed: int
) -> list[dict[str, Any]]:
    """Build an unlabelled starter set of `(gold, finding)` pairs.

    Per sampled bug example, all placed on a gold path:

    - `gold_summary` — the gold summary itself (true);
    - `gold_title_only` — the same, truncated to a title-length body (true);
    - `generic` — a bare category statement (false);
    - `pr_title` — a restatement of the PR title (false);
    - `sibling_gold` — another gold of the same PR (false, same file);
    - `other_gold` — a different bug's summary (false, same file).

    `label` is left `null`; hand-label each row before running `--pairs`.
    """
    bugs = [example for example in examples.values() if example.kind == "bug"]
    rng = random.Random(seed)
    rng.shuffle(bugs)

    # Always include the multi-gold cases: their sibling is the hardest
    # negative (same file, different bug), so it must be in the sample.
    sampled = bugs[:limit]
    seen = {example.example_id for example in sampled}
    sampled = [
        example
        for example in bugs
        if example.siblings and example.example_id not in seen
    ] + sampled

    pairs: list[dict[str, Any]] = []
    for example in sampled:
        target = example.target
        if target is None:
            continue
        path = target.paths[0]
        title = _short_title(target.summary)

        candidates: list[tuple[str, Finding]] = [
            (
                "gold_summary",
                Finding(
                    severity="major",
                    path=path,
                    line=1,
                    title=title,
                    body=target.summary,
                ),
            ),
            (
                "gold_title_only",
                Finding(
                    severity="major",
                    path=path,
                    line=1,
                    title=title,
                    body=_short_title(target.summary, 200),
                ),
            ),
            (
                "generic",
                Finding(
                    severity="major",
                    path=path,
                    line=1,
                    title="Potential null dereference",
                    body=rng.choice(GENERIC_TEMPLATES),
                ),
            ),
            (
                "pr_title",
                Finding(
                    severity="minor",
                    path=path,
                    line=1,
                    title=_short_title(example.title or "This PR changes behavior"),
                    body=(
                        f"{example.title or 'This PR'} — the PR changes behavior "
                        "in this file."
                    ),
                ),
            ),
        ]
        if example.siblings:
            sibling = example.siblings[0]
            candidates.append(
                (
                    "sibling_gold",
                    Finding(
                        severity="major",
                        path=sibling.paths[0],
                        line=1,
                        title=_short_title(sibling.summary),
                        body=sibling.summary,
                    ),
                )
            )
        other = _other_gold(examples, target.gid, rng)
        if other is not None:
            candidates.append(
                (
                    "other_gold",
                    Finding(
                        severity="major",
                        path=path,
                        line=1,
                        title=_short_title(other.summary),
                        body=other.summary,
                    ),
                )
            )

        for kind, finding in candidates:
            pairs.append(
                {
                    "kind": kind,
                    "example_id": example.example_id,
                    "label": None,
                    "gold": {
                        "gid": target.gid,
                        "summary": target.summary,
                        "paths": list(target.paths),
                        "notes": list(target.notes),
                    },
                    "finding": {
                        "severity": finding.severity,
                        "path": finding.path,
                        "line": finding.line,
                        "title": finding.title,
                        "body": finding.body,
                    },
                }
            )
    return pairs


def _pair_gold(payload: dict[str, Any]) -> Gold:
    return Gold(
        gid=payload.get("gid", "pair"),
        summary=payload["summary"],
        paths=tuple(payload.get("paths") or ()),
        notes=tuple(payload.get("notes") or ()),
    )


def load_labelled_pairs(
    pairs_path: Path,
) -> tuple[list[tuple[dict[str, Any], bool]], int]:
    """Read `(pair, label)` rows from a calibration JSONL file."""
    labelled: list[tuple[dict[str, Any], bool]] = []
    unlabelled = 0
    for number, line in enumerate(
        pairs_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            pair = json.loads(line)
        except json.JSONDecodeError as error:
            print(f"[calibrate] line {number}: invalid JSON ({error.msg}), skipped")
            continue
        label = pair.get("label")
        if label is None:
            unlabelled += 1
            continue
        if not isinstance(label, bool):
            print(f"[calibrate] line {number}: label is not a boolean, skipped")
            continue
        labelled.append((pair, label))
    return labelled, unlabelled


@dataclass(frozen=True)
class PairOutcome:
    kind: str
    label: bool
    verdict: bool
    finding: Finding | None
    gold_summary: str
    error: str | None


async def judge_pairs(
    labelled: Sequence[tuple[dict[str, Any], bool]],
    judge: Judge,
    *,
    concurrency: int,
) -> list[PairOutcome]:
    """Judge every labelled pair, at most `concurrency` calls in flight."""
    slots = asyncio.Semaphore(concurrency)

    async def one(pair: dict[str, Any], label: bool) -> PairOutcome:
        kind = str(pair.get("kind", "?"))
        gold = _pair_gold(pair["gold"])
        try:
            finding = parse_finding(pair["finding"])
        except (TypeError, ValueError) as error:
            return PairOutcome(kind, label, False, None, gold.summary, str(error))
        async with slots:
            verdict, error = await judge.same_bug(gold, finding)
        return PairOutcome(kind, label, verdict, finding, gold.summary, error)

    return list(await asyncio.gather(*(one(pair, label) for pair, label in labelled)))


async def calibrate(
    labelled: Sequence[tuple[dict[str, Any], bool]],
    judge: Judge,
    *,
    verbose: bool,
    concurrency: int,
) -> int:
    """Run the judge over hand-labelled pairs and report agreement."""
    outcomes = await judge_pairs(labelled, judge, concurrency=concurrency)

    true_positives = false_positives = true_negatives = false_negatives = 0
    disagreements: list[str] = []
    for outcome in outcomes:
        if verbose:
            print(
                f"[calibrate] {outcome.kind:16s} "
                f"label={outcome.label!s:5s} judge={outcome.verdict!s:5s}"
            )
        if outcome.error:
            print(f"[calibrate] {outcome.error}")

        title = outcome.finding.title if outcome.finding else "<invalid finding>"
        if outcome.verdict and outcome.label:
            true_positives += 1
        elif outcome.verdict and not outcome.label:
            false_positives += 1
            disagreements.append(
                f"false positive ({outcome.kind}): "
                f"{title!r} vs {outcome.gold_summary[:100]!r}"
            )
        elif outcome.label:
            false_negatives += 1
            disagreements.append(
                f"false negative ({outcome.kind}): "
                f"{title!r} vs {outcome.gold_summary[:100]!r}"
            )
        else:
            true_negatives += 1

    total = true_positives + false_positives + true_negatives + false_negatives
    accuracy = (true_positives + true_negatives) / total if total else 0.0
    print(
        f"[calibrate] {total} pairs: accuracy={accuracy:.3f} "
        f"tp={true_positives} fp={false_positives} "
        f"tn={true_negatives} fn={false_negatives}"
    )
    if judge.failures:
        print(f"[calibrate] {judge.failures}/{judge.calls} judge calls failed")
    for line in disagreements:
        print(f"[calibrate] {line}")
    return 0


# --- CLI ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-base-url", default=DEFAULT_JUDGE_BASE_URL)
    parser.add_argument("--judge-api-key", default=None)
    parser.add_argument(
        "--judge-seed",
        type=int,
        default=DEFAULT_JUDGE_SEED,
        help="Request seed for reproducible verdicts; negative disables it.",
    )
    parser.add_argument(
        "--judge-votes",
        type=int,
        default=1,
        help="Majority vote over N judge calls per pair (odd N is best).",
    )
    parser.add_argument("--pairs", type=Path, default=None, help="Calibration JSONL.")
    parser.add_argument(
        "--make-pairs",
        type=Path,
        default=None,
        help="Write an unlabelled starter calibration JSONL and exit.",
    )
    parser.add_argument(
        "--limit", type=int, default=12, help="Examples for --make-pairs."
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    parser.add_argument("--example-id", default=None, help="Score one rollout.")
    parser.add_argument("--output", type=Path, default=None, help="Agent NDJSON file.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--concurrency", type=int, default=8, help="Judge calls in flight."
    )
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> int:
    if args.make_pairs is not None:
        examples = load_examples(args.data_dir)
        pairs = candidate_pairs(examples, limit=args.limit, seed=args.seed)
        args.make_pairs.parent.mkdir(parents=True, exist_ok=True)
        args.make_pairs.write_text(
            "".join(json.dumps(pair, ensure_ascii=False) + "\n" for pair in pairs),
            encoding="utf-8",
        )
        print(
            f"[reward] wrote {len(pairs)} unlabelled pairs to {args.make_pairs}; "
            'set "label" to true/false, then rerun with --pairs'
        )
        return 0

    if args.pairs is not None:
        labelled, unlabelled = load_labelled_pairs(args.pairs)
        if not labelled:
            print(
                f"[calibrate] no labelled pairs in {args.pairs} "
                f'({unlabelled} unlabelled); set "label": true/false and rerun'
            )
            return 1
        async with Judge(
            model=args.judge_model,
            base_url=args.judge_base_url,
            api_key=args.judge_api_key,
            seed=args.judge_seed if args.judge_seed >= 0 else None,
            votes=args.judge_votes,
        ) as judge:
            return await calibrate(
                labelled,
                judge,
                verbose=args.verbose,
                concurrency=args.concurrency,
            )

    if args.example_id is None or args.output is None:
        print("nothing to do: pass --pairs, --make-pairs, or --example-id/--output")
        return 1

    examples = load_examples(args.data_dir)
    if args.example_id not in examples:
        print(f"unknown example id: {args.example_id}")
        return 1
    findings, errors = parse_findings(args.output.read_text(encoding="utf-8"))
    for error in errors:
        print(f"[reward] parse: {error}")
    example = examples[args.example_id]
    async with Judge(
        model=args.judge_model,
        base_url=args.judge_base_url,
        api_key=args.judge_api_key,
        seed=args.judge_seed if args.judge_seed >= 0 else None,
        votes=args.judge_votes,
    ) as judge:
        result = await score_example(example, findings, judge)
    print(
        f"[reward] {example.example_id} kind={example.kind} "
        f"findings={len(findings)} reward={result.total:.3f} "
        f"matched={result.matched} severity={result.matched_severity} "
        f"strict_match={result.strict_match} judge_calls={result.judge_calls}"
    )
    for error in result.errors:
        print(f"[reward] {error}")
    return 0


def main() -> int:
    args = parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be at least 1")
    if args.judge_votes < 1:
        raise SystemExit("--judge-votes must be at least 1")
    try:
        return asyncio.run(async_main(args))
    except ValueError as error:
        raise SystemExit(f"[reward] {error}") from error


if __name__ == "__main__":
    raise SystemExit(main())
