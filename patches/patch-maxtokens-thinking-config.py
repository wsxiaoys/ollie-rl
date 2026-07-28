#!/usr/bin/env python3
"""Fix the gemini_msrl.py hardcode that DROPPED the caller's max_tokens.

Before (regression): the generation_config hardcoded
    max_output_tokens=8192            # ignored request.max_tokens entirely
    thinking_config={"thinkingLevel": "MINIMAL"}
The block ABOVE it already computed a properly clamped `max_tokens` from
`request.max_tokens` (capped at VERTEX_MAX_OUTPUT_TOKENS=32768) -- and then the
hardcode threw it away. So any caller-supplied budget was silently lost.

After (config with the run's operating point as defaults):
    max_output_tokens:
        - if request.max_tokens is set  -> use it (already clamped above)
        - else                          -> OLLIE_MAX_OUTPUT_TOKENS env (default 8192)
        (still clamped to VERTEX_MAX_OUTPUT_TOKENS)
    thinking_config:
        - OLLIE_THINKING_LEVEL env (default "MINIMAL")
        - set to None (omit) if OLLIE_THINKING_LEVEL="" / "none" / "off"

No wire-protocol or driver change. The VM keeps 8192 + MINIMAL via the env
defaults in start-ollie.sh, but a real caller-supplied max_tokens is now
honored instead of dropped, and the thinking level is configurable rather than
frozen. Committable as a real feature (env-driven operating point).

Idempotent. Run on VM: python3 ~/patch-maxtokens-thinking-config.py [--revert]
"""

import pathlib
import sys

TRAINER = pathlib.Path.home() / "ollie-rl/src/ollie_rl/trainer/gemini_msrl.py"

# 1. add `import os` (after `import logging`, a known-present line)
IMPORT_ANCHOR = "import logging\n"
IMPORT_NEW = "import logging\nimport os\n"

# 2. Replace the clamp + hardcoded generation_config with env-driven logic.
#    Anchor spans from the clamp comment through the GenerationConfig(...) call.
OLD = """        # Vertex caps max_output_tokens at 32768 for tuning-scope generations.
        # OpenAI clients (e.g. cloudcode) often send larger values (64000+);
        # clamp to the documented max so we don't 400 on perfectly valid
        # OpenAI-style requests.
        VERTEX_MAX_OUTPUT_TOKENS = 32768
        max_tokens = request.max_tokens
        if max_tokens is None or max_tokens > VERTEX_MAX_OUTPUT_TOKENS:
            max_tokens = VERTEX_MAX_OUTPUT_TOKENS

        tuning_job_id = self.tuning_job_name.split("/")[-1]
        scope_req = GenerateContentTuningScopeRequest(
            content_generation_parameters=ContentGenerationParameters(
                contents=contents,
                generation_config=GenerationConfig(
                    max_output_tokens=8192,  # HACK: 4999 recipe operating point
                    thinking_config={"thinkingLevel": "MINIMAL"},  # HACK
                ),
                system_instruction=system_instruction,
                tools=gemini_tools,
            )
        )"""

NEW = """        # Vertex caps max_output_tokens at 32768 for tuning-scope generations.
        # OpenAI clients (e.g. cloudcode) often send larger values (64000+);
        # clamp to the documented max so we don't 400 on perfectly valid
        # OpenAI-style requests.
        VERTEX_MAX_OUTPUT_TOKENS = 32768
        # Operating point is env-configurable so a deployment can pin a smaller
        # generation budget without a code change. Precedence: an explicit
        # request.max_tokens wins; otherwise fall back to OLLIE_MAX_OUTPUT_TOKENS
        # (default 8192). Either way the value is clamped to Vertex's cap. This
        # replaces a former hardcode that silently ignored request.max_tokens.
        _default_max_tokens = int(os.environ.get("OLLIE_MAX_OUTPUT_TOKENS", "8192"))
        max_tokens = request.max_tokens
        if max_tokens is None:
            max_tokens = _default_max_tokens
        if max_tokens > VERTEX_MAX_OUTPUT_TOKENS:
            max_tokens = VERTEX_MAX_OUTPUT_TOKENS

        # Thinking level is env-configurable (default MINIMAL, the current
        # operating point). Set OLLIE_THINKING_LEVEL to "" / "none" / "off" to
        # omit thinking_config entirely (backend default). Note the inner key is
        # camelCase `thinkingLevel` on purpose -- the tuning-scope API rejects
        # snake_case here.
        _thinking_level = os.environ.get("OLLIE_THINKING_LEVEL", "MINIMAL").strip()
        _thinking_config = (
            {"thinkingLevel": _thinking_level}
            if _thinking_level and _thinking_level.lower() not in ("none", "off")
            else None
        )

        tuning_job_id = self.tuning_job_name.split("/")[-1]
        scope_req = GenerateContentTuningScopeRequest(
            content_generation_parameters=ContentGenerationParameters(
                contents=contents,
                generation_config=GenerationConfig(
                    max_output_tokens=max_tokens,
                    thinking_config=_thinking_config,
                ),
                system_instruction=system_instruction,
                tools=gemini_tools,
            )
        )"""


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
print("import os:      ", apply(TRAINER, IMPORT_ANCHOR, IMPORT_NEW, revert))
print("config block:  ", apply(TRAINER, OLD, NEW, revert))
