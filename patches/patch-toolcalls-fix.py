#!/usr/bin/env python3
"""FIX (not diagnostic): materialize each assistant message's `tool_calls`
ValidatorIterator to a real list at the very top of TunerService.sample(),
BEFORE hash_request()/model_dump() can consume it.

Root cause: OpenAI's tool_calls is typed Iterable -> pydantic v2 exposes a
single-consumption ValidatorIterator. hash_request() (model_dump) runs before
the trainer reads tool_calls and exhausts it, so multi-turn tool calls vanish
and MSRL 400s on function_response.name. Materializing once up front makes every
downstream consumer see the data.

Idempotent. Run on VM: python3 ~/patch-toolcalls-fix.py [--revert]
"""

import sys, pathlib

F = pathlib.Path.home() / "ollie-rl/src/ollie_rl/service/tuner/sampling.py"
MARKER = "# FIX(tool_calls-iterator):"

# Anchor: the first executable line of sample(), right after the docstring.
ANCHOR = '''        """
        Generate a chat completion from the active policy of the requested model,
        and optionally record metadata if run_id is provided.
        """
        trainer = await self._get_trainer(tuner_id)
'''

INSERT = """        # FIX(tool_calls-iterator): OpenAI `tool_calls` is typed Iterable, so
        # pydantic v2 exposes it as a single-consumption ValidatorIterator.
        # hash_request()/model_dump() below would exhaust it before the trainer
        # reads it, silently dropping multi-turn tool calls (MSRL then 400s on
        # function_response.name). Materialize to a list ONCE up front so every
        # downstream consumer sees the data.
        for _m in request.messages:
            if isinstance(_m, dict) and _m.get("tool_calls") is not None:
                _m["tool_calls"] = list(_m["tool_calls"])
"""

src = F.read_text()

if "--revert" in sys.argv:
    if MARKER in src:
        F.write_text(src.replace(INSERT, ""))
        print("REVERTED")
    else:
        print("nothing to revert")
    sys.exit(0)

if MARKER in src:
    print("already patched")
    sys.exit(0)
if ANCHOR not in src:
    print("ANCHOR NOT FOUND — aborting")
    sys.exit(2)

# Insert INSERT right after the anchor's `trainer = await self._get_trainer(...)`
src = src.replace(ANCHOR, ANCHOR + INSERT, 1)
F.write_text(src)
print("PATCHED")
