#!/usr/bin/env python3
"""Insert a TEMP diagnostic dump into the VM's OllieSampler.sample so we can see
the exact OpenAI request body sent and response received per turn.
Dumps to /tmp/ollie_sampler.jsonl. Idempotent. Run: python3 ~/patch-sampler-debug.py [--revert]
"""

import sys, pathlib

F = (
    pathlib.Path.home()
    / "vmsrl-cs/internal_src/example_recipe/codesearch/ollie_sampler.py"
)
MARKER = "ollie_sampler.jsonl"

ANCHOR = """        data = await self._post_with_retries(body)
"""
INSERT = """        try:
            import json as _json, time as _time
            with open("/tmp/ollie_sampler.jsonl", "a") as _f:
                _f.write(_json.dumps({"ts": _time.time(), "request": body, "response": data}) + "\\n")
        except Exception:
            pass
"""

src = F.read_text()
if "--revert" in sys.argv:
    if MARKER in src:
        F.write_text(src.replace(ANCHOR + INSERT, ANCHOR))
        print("REVERTED")
    else:
        print("nothing to revert")
    sys.exit(0)
if MARKER in src:
    print("already patched")
    sys.exit(0)
if ANCHOR not in src:
    print("ANCHOR NOT FOUND")
    sys.exit(2)
F.write_text(src.replace(ANCHOR, ANCHOR + INSERT, 1))
print("PATCHED")
