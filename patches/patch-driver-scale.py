#!/usr/bin/env python3
"""Scale the driver for --concurrency 1024.
  1. ollie_sampler.py: each per-run OllieSampler owns its own httpx client.
     At concurrency 1024 there are ~1024 of these; each only needs a few conns
     (one in-flight completion at a time), but set explicit small Limits so 1024
     clients don't each spin up the default 100-conn pool. Also widen pool wait.
  2. codesearch_ollie_train.py: the SHARED dispense/reward client is hit by all
     1024 workers -> give it high Limits + a longer pool wait so dispense/reward
     don't starve.
Idempotent. Run on VM: python3 ~/patch-driver-scale.py [--revert]
"""

import sys, pathlib

BASE = pathlib.Path.home() / "vmsrl-cs/internal_src/example_recipe/codesearch"
SAMPLER = BASE / "ollie_sampler.py"
DRIVER = BASE / "codesearch_ollie_train.py"

SAMPLER_ANCHOR = (
    "        self._client = http_client or httpx.AsyncClient(timeout=timeout)\n"
)
SAMPLER_NEW = (
    "        self._client = http_client or httpx.AsyncClient(\n"
    "            timeout=timeout,\n"
    "            limits=httpx.Limits(max_connections=8, max_keepalive_connections=2),  # SCALE\n"
    "        )\n"
)

DRIVER_ANCHOR = (
    "    timeout = httpx.Timeout(connect=10.0, read=90.0, write=30.0, pool=30.0)\n"
    "    async with httpx.AsyncClient(base_url=args.base_url, timeout=timeout) as client:\n"
)
DRIVER_NEW = (
    "    timeout = httpx.Timeout(connect=10.0, read=90.0, write=30.0, pool=120.0)  # SCALE pool wait\n"
    "    _scale_limits = httpx.Limits(max_connections=2048, max_keepalive_connections=512)  # SCALE\n"
    "    async with httpx.AsyncClient(base_url=args.base_url, timeout=timeout, limits=_scale_limits) as client:\n"
)


def apply(path, anchor, new, revert):
    src = path.read_text()
    if revert:
        if new in src:
            path.write_text(src.replace(new, anchor))
            return "REVERTED"
        return "nothing to revert"
    if new in src:
        return "already patched"
    if anchor not in src:
        return "ANCHOR NOT FOUND"
    path.write_text(src.replace(anchor, new, 1))
    return "PATCHED"


revert = "--revert" in sys.argv
print("ollie_sampler.py:", apply(SAMPLER, SAMPLER_ANCHOR, SAMPLER_NEW, revert))
print("codesearch_ollie_train.py:", apply(DRIVER, DRIVER_ANCHOR, DRIVER_NEW, revert))
