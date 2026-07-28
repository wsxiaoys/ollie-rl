#!/usr/bin/env python3
"""Insert a TEMP diagnostic dump of generateContentTuningScope request bodies
into the VM's gemini_msrl/client.py, right after the client.request() call.
Idempotent: does nothing if the marker is already present.
Run on the VM: python3 ~/patch-client-debug.py [--revert]
"""

import sys, pathlib

CLIENT = pathlib.Path.home() / "ollie-rl/src/gemini_msrl/client.py"
MARKER = "gcts_requests.jsonl"

ANCHOR = """            response = await client.request(
                method, url, json=json_data, params=params, headers=request_headers
            )
"""

INSERT = """            if "generateContentTuningScope" in path:
                try:
                    import json as _json, time as _time
                    with open("/tmp/gcts_requests.jsonl", "a") as _f:
                        _f.write(_json.dumps({"ts": _time.time(), "status": response.status_code, "body": json_data}) + "\\n")
                except Exception:
                    pass
"""

src = CLIENT.read_text()

if "--revert" in sys.argv:
    if MARKER in src:
        src = src.replace(ANCHOR + INSERT, ANCHOR)
        CLIENT.write_text(src)
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

src = src.replace(ANCHOR, ANCHOR + INSERT, 1)
CLIENT.write_text(src)
print("PATCHED")
