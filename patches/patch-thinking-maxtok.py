#!/usr/bin/env python3
"""HACK (throwaway, VM-local): hardcode thinking_level=MINIMAL and
max_output_tokens=8192 into the ollie server's generateContentTuningScope
request, to match the codesearch async operating point (the 4999 recipe).

Two edits:
  1. gemini_msrl/types.py: add `thinking_config` to GenerationConfig so the
     field is allowed (BaseModelConfig has extra='forbid'). to_camel alias +
     by_alias dump -> serializes as `thinkingConfig` on the wire.
  2. ollie_rl/trainer/gemini_msrl.py: build GenerationConfig with
     max_output_tokens=8192 (was clamped request.max_tokens) and
     thinking_config={"thinking_level": "MINIMAL"} (-> thinkingConfig.thinkingLevel).

Idempotent. Run on VM: python3 ~/patch-thinking-maxtok.py [--revert]
"""

import sys, pathlib

TYPES = pathlib.Path.home() / "ollie-rl/src/gemini_msrl/types.py"
TRAINER = pathlib.Path.home() / "ollie-rl/src/ollie_rl/trainer/gemini_msrl.py"

TYPES_ANCHOR = """class GenerationConfig(BaseModelConfig):
    max_output_tokens: Optional[int] = None
"""
TYPES_PATCHED = """class GenerationConfig(BaseModelConfig):
    max_output_tokens: Optional[int] = None
    thinking_config: Optional[dict] = None  # HACK: {"thinking_level": "MINIMAL"}
"""

TRAINER_ANCHOR = """                generation_config=GenerationConfig(
                    max_output_tokens=max_tokens,
                ),
"""
TRAINER_PATCHED = """                generation_config=GenerationConfig(
                    max_output_tokens=8192,  # HACK: 4999 recipe operating point
                    thinking_config={"thinking_level": "MINIMAL"},  # HACK
                ),
"""


def apply(path, anchor, patched, revert):
    src = path.read_text()
    if revert:
        if patched in src:
            path.write_text(src.replace(patched, anchor))
            return "REVERTED"
        return "nothing to revert"
    if patched in src:
        return "already patched"
    if anchor not in src:
        return "ANCHOR NOT FOUND"
    path.write_text(src.replace(anchor, patched, 1))
    return "PATCHED"


revert = "--revert" in sys.argv
print("types.py:", apply(TYPES, TYPES_ANCHOR, TYPES_PATCHED, revert))
print("gemini_msrl.py:", apply(TRAINER, TRAINER_ANCHOR, TRAINER_PATCHED, revert))
