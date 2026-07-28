#!/usr/bin/env python3
"""Fix HTTP 500 on every sample: TuningCandidate rejects backend's new
`promptTokenCount` field.

Root cause: the Vertex tuning-scope backend now returns `promptTokenCount` on
each `tuningCandidates[i]`, but `gemini_msrl.types.TuningCandidate` inherits
`BaseModelConfig` (extra="forbid") and doesn't declare it, so pydantic raises
`Extra inputs are not permitted [promptTokenCount]` and the WHOLE
GenerateContentTuningScopeResponse fails to parse -> server 500 -> every
rollout NaN.

Fix (option 1, precise): declare `prompt_token_count: Optional[int] = None` on
TuningCandidate (parallel to the existing generation_token_count /
thoughts_token_count; the camelCase alias `promptTokenCount` is handled by the
base alias_generator). Captures the value and satisfies extra="forbid" without
loosening the model.

Idempotent. Run on VM: python3 ~/patch-tuning-candidate-extra.py [--revert]
"""

import pathlib
import sys

TYPES = pathlib.Path.home() / "ollie-rl/src/gemini_msrl/types.py"

OLD = """class TuningCandidate(BaseModelConfig):
    candidate_id: str
    candidate: Candidate
    generation_token_count: Optional[int] = None
    thoughts_token_count: Optional[int] = None
"""
NEW = """class TuningCandidate(BaseModelConfig):
    candidate_id: str
    candidate: Candidate
    prompt_token_count: Optional[int] = None
    generation_token_count: Optional[int] = None
    thoughts_token_count: Optional[int] = None
"""


def apply(path, old, new, revert):
    src = path.read_text()
    a, b = (new, old) if revert else (old, new)
    if b in src:
        return "already " + ("reverted" if revert else "patched")
    if a not in src:
        return "ANCHOR NOT FOUND"
    path.write_text(src.replace(a, b, 1))
    return "REVERTED" if revert else "PATCHED"


revert = "--revert" in sys.argv
print("TuningCandidate:", apply(TYPES, OLD, NEW, revert))
