#!/usr/bin/env python3
"""HACK (throwaway, VM-local): wire the codesearch async "4999 recipe"
operating point into the ollie driver on the VM.

Edits ~/vmsrl-cs/internal_src/example_recipe/codesearch/codesearch_ollie_train.py:
  1. Reward config: replace CodeSearchRewardConfig() with the 4999 values
     (lowest_reward=-0.1, bucketized F0.5, run-steps bonus, sum combine).
  2. Agent: pass turn_limit_prompt_override=5 in run_rollout.
  3. Defaults: --max-steps default 10, --max-datums default 1283 (whole train set).
Note: dataset path + thinking + max_tokens are handled elsewhere (CLI flag /
server hack), not here.

Idempotent. Run on VM: python3 ~/patch-driver-recipe.py [--revert]
"""

import sys, pathlib

F = (
    pathlib.Path.home()
    / "vmsrl-cs/internal_src/example_recipe/codesearch/codesearch_ollie_train.py"
)

EDITS = [
    # 1. Reward config -> 4999 recipe values.
    (
        "    reward_config = CodeSearchRewardConfig()\n",
        """    reward_config = CodeSearchRewardConfig(  # HACK: 4999 recipe operating point
        lowest_reward=-0.1,
        highest_reward=1.0,
        enable_conditional_reward=True,
        files_score_threshold=0.5,
        files_reward_multiplier=1.0,
        files_bucket_boundaries=[0.0, 0.6, 0.7, 0.8, 0.9, 1.0],
        lines_bucket_boundaries=[0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        run_steps_bonus_amount=0.05,
        run_steps_bonus_max_steps=5,
        run_steps_bonus_files_min=0.65,
        run_steps_bonus_lines_min=0.45,
    )
""",
    ),
    # 2. Pass turn_limit_prompt_override=5 to the agent.
    (
        """            tuning_job_id=tuner_id,
            max_steps=max_steps,
            lowest_reward=lowest_reward,
            sample_timeout_s=sample_timeout_s,
        )
""",
        """            tuning_job_id=tuner_id,
            max_steps=max_steps,
            lowest_reward=lowest_reward,
            turn_limit_prompt_override=5,  # HACK: 4999 recipe
            sample_timeout_s=sample_timeout_s,
        )
""",
    ),
    # 3a. --max-steps default 30 -> 10.
    (
        """        "--max-steps",
        type=int,
        default=30,
""",
        """        "--max-steps",
        type=int,
        default=10,  # HACK: 4999 recipe
""",
    ),
    # 3b. --max-datums default 32 -> 1283 (whole train set).
    (
        """        "--max-datums",
        type=int,
        default=32,
""",
        """        "--max-datums",
        type=int,
        default=1283,  # HACK: whole cognition_gemini4_train.jsonl
""",
    ),
]

MARKER = "HACK: 4999 recipe operating point"
src = F.read_text()
revert = "--revert" in sys.argv

if revert:
    if MARKER not in src:
        print("nothing to revert")
        sys.exit(0)
    for anchor, patched in EDITS:
        src = src.replace(patched, anchor)
    F.write_text(src)
    print("REVERTED")
    sys.exit(0)

if MARKER in src:
    print("already patched")
    sys.exit(0)

for i, (anchor, patched) in enumerate(EDITS):
    if anchor not in src:
        print(f"ANCHOR {i} NOT FOUND -- aborting")
        sys.exit(2)
    src = src.replace(anchor, patched, 1)
F.write_text(src)
print("PATCHED")
