from __future__ import annotations

import argparse
import base64
import codecs
import dataclasses
import json
import logging
import math
import random
import string
import sys
from collections.abc import Sequence
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_ASCII_LETTERS = string.ascii_letters
_NUM_LETTERS = len(_ASCII_LETTERS)


# --- Character-level transforms with parameters --------------------------------


def _rotate_char(s: str, rotate: int = 2) -> str:
    rotate = rotate % len(s)
    return s[rotate:] + s[:rotate]


def _mirror_char(s: str, copy_input: bool = False) -> str:
    return (s + s[::-1]) if copy_input else s[::-1]


def _shift_char(s: str, shift: int = 7) -> str:
    shift = shift % _NUM_LETTERS
    return "".join(
        _ASCII_LETTERS[(_ASCII_LETTERS.find(c) + shift) % _NUM_LETTERS] for c in s
    )


# --- Fixed (parameter-free) transforms; the "extended" training set ------------


def _swap_adj(s: str) -> str:
    return "".join(s[i + 1] + s[i] for i in range(0, len(s) - 1, 2)) + (
        s[-1] if len(s) % 2 else ""
    )


def _reverse_blocks_of_3(s: str) -> str:
    return "".join(s[i : i + 3][::-1] for i in range(0, len(s), 3))


def _weave_from_ends(s: str) -> str:
    return "".join(sum(zip(s[: len(s) // 2], s[::-1]), ())) + (
        s[len(s) // 2] if len(s) % 2 else ""
    )


def _move_middle_to_front(s: str) -> str:
    n = len(s)
    if n % 2:
        return s[n // 2 : n // 2 + 1] + s[: n // 2] + s[n // 2 + 1 :]
    return s[n // 2 - 1 : n // 2 + 1] + s[: n // 2 - 1] + s[n // 2 + 1 :]


def _move_middle_to_end(s: str) -> str:
    n = len(s)
    if n % 2:
        return s[: n // 2] + s[n // 2 + 1 :] + s[n // 2 : n // 2 + 1]
    return s[: n // 2 - 1] + s[n // 2 + 1 :] + s[n // 2 - 1 : n // 2 + 1]


def _interleave_split(s: str) -> str:
    return "".join(
        "".join(pair) for pair in zip(s[: (len(s) + 1) // 2], s[(len(s) + 1) // 2 :])
    ) + (s[len(s) // 2 - 1] if len(s) % 2 else "")


def _pull_outer_to_middle(s: str) -> str:
    return s[1 : len(s) // 2] + s[0] + s[-1] + s[len(s) // 2 : -1] if len(s) > 1 else s


TRANSFORMATION_FUNCTIONS: dict[str, Callable[[str], str]] = {
    "swap_adjacent_char_pairs": _swap_adj,
    "reverse_blocks_of_3": _reverse_blocks_of_3,
    "weave_from_ends": _weave_from_ends,
    "move_middle_to_front": _move_middle_to_front,
    "move_middle_to_end": _move_middle_to_end,
    "interleave_split_string": _interleave_split,
    "pull_outer_to_middle": _pull_outer_to_middle,
}


# --- Withhold (eval-only) transforms ------------------------------------------


def _atbash(s: str) -> str:
    out = []
    for c in s:
        if "a" <= c <= "z":
            out.append(chr(ord("a") + (ord("z") - ord(c))))
        elif "A" <= c <= "Z":
            out.append(chr(ord("A") + (ord("Z") - ord(c))))
        else:
            out.append(c)
    return "".join(out)


WITHHOLD_TRANSFORMATION_FUNCTIONS: dict[str, Callable[[str], str]] = {
    "base64_encoding": lambda s: base64.b64encode(s.encode("utf-8")).decode("utf-8"),
    "hex_encoding": lambda s: s.encode("utf-8").hex(),
    "rot13_cipher": lambda s: codecs.encode(s, "rot_13"),
    "word_to_binary_string": lambda s: " ".join(format(ord(c), "08b") for c in s),
    "atbash_cipher": _atbash,
    "hex_and_reverse": lambda s: s.encode("utf-8").hex()[::-1],
    "base64_and_rot13": lambda s: codecs.encode(
        base64.b64encode(s.encode("utf-8")).decode("utf-8"), "rot_13"
    ),
    "ascii_values_as_string": lambda s: "-".join(str(ord(c)) for c in s),
    "shift_chars_by_position": lambda s: "".join(
        chr(ord(s[i]) + i) for i in range(len(s))
    ),
    "alternate_ascii_add_subtract": lambda s: "".join(
        chr(ord(c) + (1 if i % 2 == 0 else -1)) for i, c in enumerate(s)
    ),
    "duplicate_vowels": lambda s: "".join(
        c * 2 if c.lower() in "aeiou" else c for c in s
    ),
    "reverse_vowel_order": (
        lambda s: (
            lambda vowels: "".join(
                vowels.pop(0) if c.lower() in "aeiou" else c for c in s
            )
        )([c for c in s if c.lower() in "aeiou"][::-1])
    ),
}


# --- Operator sets exposed via the CLI ----------------------------------------

BASIC_OPERATORS = ["shift", "mirror", "rotate"]
EXTENDED_OPERATORS = BASIC_OPERATORS + list(TRANSFORMATION_FUNCTIONS.keys())
WITHHOLD_OPERATORS = list(WITHHOLD_TRANSFORMATION_FUNCTIONS.keys())
ALL_OPERATORS = EXTENDED_OPERATORS + WITHHOLD_OPERATORS

# --- v2 operator sets: property-matched train/withhold split ----------------
# v1 split was confounded: all training ops were length-preserving + letter-only,
# all withhold ops were encodings (expanding + charset-changing). v2 reshuffles
# so both sets contain a mix of permutations and encodings.
#
# Train v2 (14 ops):
#   Class A — length-preserving, letter-only (10): the 3 basic + 7 transforms
#   Class B — expanding, letter-only (1): duplicate_vowels
#   Class D — expanding, charset-change (3): base64_encoding, hex_encoding,
#       ascii_values_as_string
#
# Withhold v2 (8 ops):
#   Class A — length-preserving, letter-only (4): rot13_cipher, atbash_cipher,
#       alternate_ascii_add_subtract, reverse_vowel_order
#   Class C — length-preserving, charset-change (1): shift_chars_by_position
#   Class D — expanding, charset-change (3): word_to_binary_string,
#       hex_and_reverse, base64_and_rot13

EXTENDED_OPERATORS_V2 = BASIC_OPERATORS + [
    # Class A (permutations) — same 7 as v1
    "swap_adjacent_char_pairs",
    "reverse_blocks_of_3",
    "weave_from_ends",
    "move_middle_to_front",
    "move_middle_to_end",
    "interleave_split_string",
    "pull_outer_to_middle",
    # Class B (expanding, letter-only)
    "duplicate_vowels",
    # Class D (expanding, charset-change)
    "base64_encoding",
    "hex_encoding",
    "ascii_values_as_string",
]

WITHHOLD_OPERATORS_V2 = [
    # Class A (length-preserving, letter-only)
    "rot13_cipher",
    "atbash_cipher",
    "alternate_ascii_add_subtract",
    "reverse_vowel_order",
    # Class C (length-preserving, charset-change)
    "shift_chars_by_position",
    # Class D (expanding, charset-change)
    "word_to_binary_string",
    "hex_and_reverse",
    "base64_and_rot13",
]

OPERATOR_SETS = {
    "basic": BASIC_OPERATORS,
    "extended": EXTENDED_OPERATORS,
    "withhold": WITHHOLD_OPERATORS,
    "extended_v2": EXTENDED_OPERATORS_V2,
    "withhold_v2": WITHHOLD_OPERATORS_V2,
    "all": ALL_OPERATORS,
}

# Operators whose output is longer than the input; composing multiple in
# one rule multiplies string length. Callers building deep rules cap them
# via ``generate_random_rule(max_expanding_ops=...)`` to keep prompt size
# bounded.
EXPANDING_OPERATORS: set[str] = {
    "word_to_binary_string",
    "ascii_values_as_string",
    "hex_encoding",
    "hex_and_reverse",
    "base64_encoding",
    "base64_and_rot13",
    "duplicate_vowels",
}


def apply_transform(word: str, rule: Sequence[Sequence[Any]]) -> str:
    """Apply a sequence of (op_name, param) transformations to ``word``.

    The signature accepts any 2-element iterable per rule step so
    JSON-round-tripped rules (``list[list[...]]``) work unchanged.
    """
    out = word
    for op, param in rule:
        if op == "shift":
            out = _shift_char(out, param)
        elif op == "mirror":
            out = _mirror_char(out, param)
        elif op == "rotate":
            out = _rotate_char(out, param)
        elif op in TRANSFORMATION_FUNCTIONS:
            out = TRANSFORMATION_FUNCTIONS[op](out)
        elif op in WITHHOLD_TRANSFORMATION_FUNCTIONS:
            out = WITHHOLD_TRANSFORMATION_FUNCTIONS[op](out)
        else:
            raise ValueError(f"unknown operator {op!r}")
    return out


def _random_op_param(op: str, max_rotate: int, rng: random.Random) -> tuple[str, Any]:
    """Sample a random parameter for one operator."""
    if op == "shift":
        return (op, rng.randint(1, _NUM_LETTERS - 1))
    if op == "mirror":
        return (op, rng.choice([True, False]))
    if op == "rotate":
        return (op, rng.randint(1, max(1, max_rotate)))
    if op in TRANSFORMATION_FUNCTIONS or op in WITHHOLD_TRANSFORMATION_FUNCTIONS:
        return (op, None)
    raise ValueError(f"unknown operator {op!r}")


def generate_random_rule(
    num_transformations: int,
    max_rotate: int,
    operators: Sequence[str] = BASIC_OPERATORS,
    rng: Optional[random.Random] = None,
    max_expanding_ops: Optional[int] = None,
) -> list[tuple[str, Any]]:
    """Build a random rule = list of ``(op_name, param)`` of length ``num_transformations``.

    Operators are cycled + shuffled so each op appears roughly equally often.

    ``max_expanding_ops`` caps the number of ``EXPANDING_OPERATORS`` in
    the rule to bound prompt length; sampling picks uniformly in
    ``[0, max_expanding_ops]`` positions before filling in the operator
    names.
    """
    r: random.Random = rng if rng is not None else random.Random()
    if max_expanding_ops is not None:
        expanding_in_pool = [op for op in operators if op in EXPANDING_OPERATORS]
        non_expanding_in_pool = [
            op for op in operators if op not in EXPANDING_OPERATORS
        ]
        if not non_expanding_in_pool and max_expanding_ops < num_transformations:
            raise ValueError(
                f"max_expanding_ops={max_expanding_ops} but every operator in "
                f"the pool is expanding; can't build a rule of length "
                f"{num_transformations}."
            )
        # Choose how many expanding positions first (uniform in
        # [0, cap]) so "cap of 1" means "0 or 1", not "always 1".
        upper = min(max_expanding_ops, len(expanding_in_pool), num_transformations)
        num_expanding = r.randint(0, upper) if upper > 0 else 0
        chosen_ops: list[str] = []
        if num_expanding > 0:
            chosen_ops.extend(r.choices(expanding_in_pool, k=num_expanding))
        remaining = num_transformations - len(chosen_ops)
        if remaining > 0:
            if not non_expanding_in_pool:
                raise ValueError(
                    f"Cannot fill {remaining} non-expanding slots: pool "
                    f"has no non-expanding operators."
                )
            chosen_ops.extend(r.choices(non_expanding_in_pool, k=remaining))
        r.shuffle(chosen_ops)
    else:
        pool = list(operators) * math.ceil(num_transformations / max(1, len(operators)))
        r.shuffle(pool)
        chosen_ops = pool[:num_transformations]
    return [_random_op_param(op, max_rotate, r) for op in chosen_ops]


_TEMPLATES = [
    # 1 -- Codebreaker
    """You are a master codebreaker, renowned for deciphering the most enigmatic patterns. The following messages have been intercepted, each containing a word that has been systematically scrambled. Your mission is to crack the code.

Here are the successful decryptions from your field notes:
{examples_block}

{main_question}

{explanation_request}Put your final answer in the following format:
$\\text{{your answer}}$""",
    # 2 -- Linguistic Analyst
    """You are a linguistic analyst specializing in algorithmic word transformations. Your task is to identify the pattern in the following examples and apply it to a new word.

Observe the following data points:
{examples_block}

{main_question}

{explanation_request}Put your final answer in the following format:
$\\text{{your answer}}$""",
    # 3 -- Direct Pattern Finder
    """A consistent rule was used to transform an original word into an encrypted word.

Here are {num_examples} examples of this transformation:
{examples_block}

{main_question}

{explanation_request}Put your final answer in the following format:
$\\text{{your answer}}$""",
    # 4 -- Software Engineer
    """You are a senior software engineer debugging a legacy system. You've found a function, `scramble_word()`, that is undocumented. By observing its input and output, you must determine the algorithm it uses.

Here is the data from your test runs:
{examples_block}

{main_question}

{explanation_request}Put your final answer in the following format:
$\\text{{your answer}}$""",
    # 5 -- Game Show
    """Hello and welcome back to "Word Wizards!" Our contestant has to figure out the secret rule that's changing our words!

Let's look at the board:
{examples_block}

{main_question}

{explanation_request}Put your final answer in the following format:
$\\text{{your answer}}$""",
    # 6 -- Historical Cryptographer
    """You are a historian specializing in historical cryptography. A newly discovered manuscript contains a peculiar substitution cipher that seems to operate on a character-level principle rather than simple letter replacement.

Your analysis has so far revealed these pairings:
{examples_block}

{main_question}

{explanation_request}Put your final answer in the following format:
$\\text{{your answer}}$""",
    # 7 -- AI Trainer
    """// SYSTEM PROMPT: PATTERN RECOGNITION TEST

The following examples demonstrate a consistent string transformation rule. Your task is to infer the rule and apply it.

// TRAINING DATA:
{examples_block}

{main_question}

Put your final answer in the following format:
$\\text{{your answer}}$""",
    # 8 -- Xenolinguist
    """You are a xenolinguist aboard the starship "Odyssey," deciphering a communication from a new non-humanoid species. The signal appears to be a language, but each word is passed through a strange structural filter.

Our translation matrix has established the following relationships:
{examples_block}

{main_question}

{explanation_request}Put your final answer in the following format:
$\\text{{your answer}}$""",
    # 9 -- Alchemist
    """You are an apprentice to a master alchemist. You've found a grimoire detailing a "Transmutation of Essence" spell, which changes the form of written words. To understand the spell, you are studying the master's notes.

The notes show the following transmutations:
{examples_block}

{main_question}

{explanation_request}Put your final answer in the following format:
$\\text{{your answer}}$""",
    # 10 -- Minimalist
    """Infer a function `f(x)` given the following examples.

{examples_block}

{main_question}

{explanation_request}Put your final answer in the following format:
$\\text{{your answer}}$""",
]


_QUESTION_ENCRYPT = {  # reverse_task=True (given original, find encrypted)
    1: "Your next mission is to encrypt the word **[{w}]**. What is the resulting coded message?",
    2: "Now, apply that rule to transform the original word `{w}`.",
    3: "What is the encrypted word for `{w}`?",
    4: "What would be the output if the input string was `{w}`?",
    5: "Alright, contestant, for the grand prize: if our wizards get the word `{w}`, what will they turn it into?",
    6: "You must now encode the original text `{w}` using the same principle. What is the resulting coded text?",
    8: "A new concept, `{w}`, must be transmitted. How will it be filtered into a signal?",
    9: "You must now perform a Transmutation of Essence on the substance `{w}`. What is its new form?",
    10: "Given `x = '{w}'`, find `f(x)`.",
}

_QUESTION_DECRYPT = {  # reverse_task=False (given encrypted, find original)
    1: "Now, a new coded message has arrived: **[{w}]**. What is the original word?",
    2: "Use the rule to determine the original word for `{w}`.",
    3: "What is the original word for `{w}`?",
    4: "A new output, `{w}`, was generated by the function. What was the original input?",
    5: "Alright, contestant, for the grand prize: if our wizards give you `{w}`, what word did they start with? Think fast!",
    6: "A key fragment of the coded text reads: `{w}`. What is the original text for this fragment?",
    8: "A priority message has just been received: `{w}`. The ship's survival depends on your translation. What is the original concept?",
    9: "You find a smudged recipe that requires the original essence of a substance written as `{w}`. What was the original substance?",
    10: "Given `y = '{w}'`, find `x` such that `f(x) = y`.",
}

_EXPLANATION_TAIL = {
    1: "Before you give the answer, briefly outline the method you've uncovered.",
    2: "Your response must first describe the transformation rule before providing the final answer.",
    3: "First, state the rule, then provide the answer.",
    4: "First, describe the algorithm you've reverse-engineered, then state the answer.",
    5: "First, tell us the secret rule, and then give us the word!",
    6: "First, describe the character-level principle you have deciphered, then state the answer.",
    8: "First, document the structural filter's logic, then provide the answer.",
    9: "First, describe the principle of this Transmutation of Essence, then reveal the answer.",
    10: "First, define the function, then provide the answer.",
}


def _example_block(examples: Sequence[tuple[str, str]], template_id: int) -> str:
    if template_id == 7:
        rows = [f'  {{ "input": "{raw}", "output": "{enc}" }}' for raw, enc in examples]
        return "[\n" + ",\n".join(rows) + "\n]"
    lines: list[str] = []
    if template_id == 1:
        for i, (raw, enc) in enumerate(examples):
            lines.append(
                f"- **Example {i}:** `[{raw}]` was found to be the original for `[{enc}]`."
            )
    elif template_id == 2:
        for i, (raw, enc) in enumerate(examples):
            lines.append(f"{i + 1}.  Original: `{raw}`, Transformed: `{enc}`")
    elif template_id == 3:
        for raw, enc in examples:
            lines.append(f"- `{raw}` -> `{enc}`")
    elif template_id == 4:
        for raw, enc in examples:
            lines.append(f"- Input: `{raw}` -> Output: `{enc}`")
    elif template_id == 5:
        for raw, enc in examples:
            lines.append(f"- Our wizards turned `{raw}` into `{enc}`!")
    elif template_id == 6:
        for raw, enc in examples:
            lines.append(f"- Original Text: `{raw}`, Coded Text: `{enc}`")
    elif template_id == 8:
        for raw, enc in examples:
            lines.append(f"- Concept `{raw}` is transmitted as signal `{enc}`.")
    elif template_id == 9:
        for raw, enc in examples:
            lines.append(
                f"- {raw.capitalize()} (`{raw}`) becomes {enc.capitalize()} (`{enc}`)."
            )
    elif template_id == 10:
        for raw, enc in examples:
            lines.append(f"f('{raw}') = '{enc}'")
    else:
        raise ValueError(f"invalid template_id {template_id}")
    return "\n".join(lines)


_ANSWER_FORMAT_TRAILER = (
    "Put your final answer in the following format:\n$\\text{your answer}$"
)


def generate_prompt(
    examples: Sequence[tuple[str, str]],
    task_word: str,
    force_template_id: Optional[int] = None,
    explain_encryption: bool = False,
    reverse_task: bool = False,
    rng: Optional[random.Random] = None,
    include_answer_format: bool = True,
) -> tuple[str, int]:
    """Format one prompt.

    Returns ``(prompt_text, template_id)``. Template id is 1..10.

    Args:
      examples: (original, encrypted) demonstration pairs.
      task_word: the input to include in the main question.
      force_template_id: pin a template (1..10) instead of sampling.
      explain_encryption: if True, ask the model to describe the rule
        before answering.
      reverse_task: if False, question is "given encrypted, find original";
        if True, "given original, find encrypted".
      rng: seedable random source.
      include_answer_format: if True (default), every template ends
        with the ``Put your final answer in the following format:
        $\\text{your answer}$`` trailer. Direct-answer eval callers
        MUST keep True; agent-mode callers (structured tool-call
        submit) SHOULD pass False.
    """
    r: random.Random = rng if rng is not None else random.Random()
    if force_template_id is not None:
        if not 1 <= force_template_id <= len(_TEMPLATES):
            raise ValueError(f"force_template_id must be in 1..{len(_TEMPLATES)}")
        tid = force_template_id
    else:
        tid = r.randint(1, len(_TEMPLATES))

    if tid == 7:
        # Template 7 embeds the explain-encryption instruction inline.
        instr = (
            "Determine the transformed output."
            if reverse_task
            else "Determine the original input."
        )
        if explain_encryption:
            expl = "// TASK: Your response should first describe the inferred rule, then provide the answer on a new line."
            main_q = (
                f"// TEST INPUT:\n`{task_word}`\n\n// REQUIRED OUTPUT:\n{expl}\n{instr}"
            )
        else:
            main_q = f"// TEST INPUT:\n`{task_word}`\n\n// REQUIRED OUTPUT:\n{instr}"
        explanation_tail = ""
    else:
        table = _QUESTION_ENCRYPT if reverse_task else _QUESTION_DECRYPT
        main_q = table[tid].format(w=task_word)
        explanation_tail = _EXPLANATION_TAIL[tid] + "\n\n" if explain_encryption else ""

    prompt = _TEMPLATES[tid - 1].format(
        examples_block=_example_block(examples, tid),
        main_question=main_q,
        explanation_request=explanation_tail,
        num_examples=len(examples),
    )
    if not include_answer_format:
        # Strip the trailing answer-format block; every template ends
        # with it, so locate + cut and rstrip to end cleanly.
        idx = prompt.rfind(_ANSWER_FORMAT_TRAILER)
        if idx >= 0:
            prompt = prompt[:idx].rstrip()
    return prompt, tid


@dataclasses.dataclass(kw_only=True)
class PuzzleTask:
    """One synthesized word-puzzle task."""

    prompt: str
    expected_answer: str
    original_word: str
    encrypted_word: str
    few_shot_examples: list[tuple[str, str]]  # list of (original, encrypted)
    rule: list[tuple[str, Any]]  # list of (op_name, param)
    template_id: int
    num_transformations: int
    num_few_shot_examples: int
    explain_encryption: bool
    reverse_task: bool  # False: encrypted->original; True: original->encrypted
    # Optional caller-supplied metadata serialized as top-level ``tags``
    # on the JSONL record; survives into eval output for per-slice
    # breakdowns. Values must be JSON-serializable.
    tags: dict[str, Any] = dataclasses.field(default_factory=dict)
    # Optional pool of extra words disjoint from ``few_shot_examples``
    # and ``original_word``; populated when the task is generated with
    # ``include_retry_metadata=True`` for the agent-mode retry feature.
    # Empty list => the JSONL record omits the ``retry_metadata`` block.
    extra_word_pool: list[str] = dataclasses.field(default_factory=list)

    @property
    def example_id(self) -> str:
        """Stable JSON-serialized id for this task."""
        return json.dumps(
            {
                "t": self.template_id,
                "n": self.num_transformations,
                "nf": self.num_few_shot_examples,
                "ee": self.explain_encryption,
                "rt": self.reverse_task,
                "rule": [{"tr": t, "p": p} for t, p in self.rule],
            }
        )

    def to_jsonl_record(self) -> dict[str, Any]:
        """Convert to the ``{contents, references}`` wire shape consumed
        by ``parse_word_puzzle_item`` in ``word_puzzle.py``.

        Non-empty ``self.tags`` are emitted as a top-level ``tags`` key.
        Non-empty ``self.extra_word_pool`` emits a top-level
        ``retry_metadata`` block used by the agent-mode retry feature;
        the block is omitted otherwise so direct-answer/training
        pipelines never see the ground-truth rule.
        """
        record: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": self.prompt}]}],
            "references": {"reference": self.expected_answer},
        }
        if self.tags:
            record["tags"] = dict(self.tags)
        if self.extra_word_pool:
            record["retry_metadata"] = {
                # list-of-lists (not tuples) for byte-identical json round-trip.
                "rule": [[op, param] for op, param in self.rule],
                "extra_word_pool": list(self.extra_word_pool),
                "reverse_task": self.reverse_task,
            }
        return record


def generate_one_task(
    *,
    num_transformations: int,
    num_few_shot_examples: int,
    word_candidates: Sequence[str],
    operators: Sequence[str] = BASIC_OPERATORS,
    explain_encryption: bool = False,
    reverse_task: bool = False,
    force_template_id: Optional[int] = None,
    rule: Optional[Sequence[tuple[str, Any]]] = None,
    tags: Optional[dict[str, Any]] = None,
    max_expanding_ops: Optional[int] = None,
    rng: Optional[random.Random] = None,
    include_answer_format: bool = True,
    include_retry_metadata: bool = False,
    retry_extra_pool_size: int = 20,
) -> PuzzleTask:
    """Generate one word-puzzle task.

    Example:
        >>> import random
        >>> from example_recipe.word_puzzle.dataset.puzzle_dataset_gen import (
        ...     generate_one_task, EXTENDED_OPERATORS)
        >>> task = generate_one_task(
        ...     num_transformations=10,
        ...     num_few_shot_examples=5,
        ...     word_candidates=["python", "javascript", "rust", "golang", ...],
        ...     operators=EXTENDED_OPERATORS,
        ...     rng=random.Random(42),
        ... )
        >>> task.rule           # 10 (op_name, param) tuples
        >>> task.expected_answer
        >>> task.to_jsonl_record()  # -> {"contents": [...], "references": {...}}

    Args:
        num_transformations: how many operations to compose into the rule.
        num_few_shot_examples: how many (original, encrypted) pairs to show.
        word_candidates: pool of words to sample from (uses ``num_few_shot_examples
            + 1`` distinct words: N for demonstrations, 1 for the task).
        operators: which operators are allowed in the rule (e.g.
            ``BASIC_OPERATORS`` for shift/mirror/rotate,  ``EXTENDED_OPERATORS``
            for the training-set operators, ``WITHHOLD_OPERATORS`` for
            eval-only operators, or ``ALL_OPERATORS``).
        explain_encryption: if True, ask the model to describe the rule before
            answering.
        reverse_task: if False (default), question is "given encrypted, find
            original"; if True, "given original, find encrypted".
        force_template_id: pin a prompt template (1..10) instead of sampling.
        rule: optionally pass a pre-built rule (skips ``generate_random_rule``).
            Useful for testing or when you want to reuse a rule across tasks.
        tags: optional caller-supplied metadata dict, serialized as a
            top-level ``tags`` key on the JSONL record and passed
            through into the eval output for group-by-tag analysis.
        max_expanding_ops: if set, caps the number of expanding operators
            (see ``EXPANDING_OPERATORS``) in a single rule. Ignored when
            ``rule`` is provided.
        rng: seedable ``random.Random`` instance. If None, uses a fresh
            (non-deterministic) one.
        include_answer_format: if True (default), the emitted prompt ends
            with the ``$\\text{your answer}$`` trailer. Set False for
            agent-mode (structured tool-call submit) eval generation.
        include_retry_metadata: if True, populate
            ``PuzzleTask.extra_word_pool`` and emit a ``retry_metadata``
            block on the JSONL record for the agent-mode retry feature.
            Default False to avoid leaking the ground-truth rule.
        retry_extra_pool_size: pool size when
            ``include_retry_metadata=True``. Ignored otherwise.

    Returns:
        A ``PuzzleTask`` dataclass.
    """
    if num_few_shot_examples < 0:
        raise ValueError("num_few_shot_examples must be >= 0")
    r: random.Random = rng if rng is not None else random.Random()

    picked = r.sample(list(word_candidates), num_few_shot_examples + 1)
    demo_words = picked[:num_few_shot_examples]
    task_word = picked[-1]

    if rule is None:
        max_rotate = min((len(w) - 1 for w in demo_words), default=len(task_word) - 1)
        max_rotate = max(1, max_rotate)
        rule = generate_random_rule(
            num_transformations=num_transformations,
            max_rotate=max_rotate,
            operators=operators,
            rng=r,
            max_expanding_ops=max_expanding_ops,
        )
    rule = list(rule)

    demos: list[tuple[str, str]] = [(w, apply_transform(w, rule)) for w in demo_words]
    encrypted = apply_transform(task_word, rule)

    prompt, tid = generate_prompt(
        examples=demos,
        task_word=task_word if reverse_task else encrypted,
        force_template_id=force_template_id,
        explain_encryption=explain_encryption,
        reverse_task=reverse_task,
        rng=r,
        include_answer_format=include_answer_format,
    )

    # Sample the retry-feature extra pool disjoint from demo + task
    # words; if candidates run short, take whatever's available.
    extra_word_pool: list[str] = []
    if include_retry_metadata:
        used = set(demo_words) | {task_word}
        pool_candidates = [w for w in word_candidates if w not in used]
        n_pool = min(retry_extra_pool_size, len(pool_candidates))
        extra_word_pool = r.sample(pool_candidates, n_pool)

    return PuzzleTask(
        prompt=prompt,
        expected_answer=encrypted if reverse_task else task_word,
        original_word=task_word,
        encrypted_word=encrypted,
        few_shot_examples=demos,
        rule=rule,
        template_id=tid,
        num_transformations=num_transformations,
        num_few_shot_examples=num_few_shot_examples,
        explain_encryption=explain_encryption,
        reverse_task=reverse_task,
        tags=dict(tags) if tags else {},
        extra_word_pool=extra_word_pool,
    )


def load_word_candidates(
    path: Optional[str] = None,
    *,
    min_length: int = 4,
    wordfreq_top_n: int = 150000,
    wordfreq_skip_top: int = 1000,
) -> list[str]:
    """Load ASCII-letter-only word candidates.

    Loading order:
      1. If ``path`` is given, read one word per line from that file.
      2. Otherwise, try to import ``wordfreq`` and use
         ``top_n_list('en', wordfreq_top_n, ascii_only=True)``, then drop
         the top ``wordfreq_skip_top`` most-common words (too easy /
         function-word-heavy).

    Results are filtered to lowercase-ASCII words of length
    ``>= min_length``, de-duplicated, and sorted for determinism.
    """
    if path is not None:
        with open(path, "r", encoding="utf-8") as f:
            raw = [line.strip() for line in f if line.strip()]
    else:
        try:
            from wordfreq import top_n_list
        except ImportError as e:
            raise RuntimeError(
                "No --word-list given and the `wordfreq` package is not "
                "installed. Either `pip install wordfreq` or pass "
                "--word-list PATH."
            ) from e
        raw = top_n_list("en", wordfreq_top_n, ascii_only=True)[wordfreq_skip_top:]

    keep: set[str] = set()
    for w in raw:
        if len(w) >= min_length and all(c in _ASCII_LETTERS for c in w):
            keep.add(w)
    return sorted(keep)


def generate_dataset(
    *,
    num_examples: int,
    num_transformations: int,
    num_few_shot_examples: int,
    word_candidates: Sequence[str],
    operators: Sequence[str] = BASIC_OPERATORS,
    explain_encryption: bool = True,
    flip_reverse_task: bool = True,
    rng: Optional[random.Random] = None,
    include_retry_metadata: bool = False,
    retry_extra_pool_size: int = 20,
) -> list[PuzzleTask]:
    """Generate ``num_examples`` tasks (or ``2 * num_examples`` if
    ``flip_reverse_task=True``, producing both encrypt- and
    decrypt-direction puzzles).

    Args:
        include_retry_metadata: if True, each task carries
            ``retry_metadata`` (extra word pool for the agent-mode
            retry feature). Forwarded to ``generate_one_task``.
        retry_extra_pool_size: pool size when
            ``include_retry_metadata=True``. Ignored otherwise.
    """
    r: random.Random = rng if rng is not None else random.Random()
    tasks: list[PuzzleTask] = []
    for _ in range(num_examples):
        tasks.append(
            generate_one_task(
                num_transformations=num_transformations,
                num_few_shot_examples=num_few_shot_examples,
                word_candidates=word_candidates,
                operators=operators,
                explain_encryption=explain_encryption,
                reverse_task=False,
                rng=r,
                include_retry_metadata=include_retry_metadata,
                retry_extra_pool_size=retry_extra_pool_size,
            )
        )
    if flip_reverse_task:
        for _ in range(num_examples):
            tasks.append(
                generate_one_task(
                    num_transformations=num_transformations,
                    num_few_shot_examples=num_few_shot_examples,
                    word_candidates=word_candidates,
                    operators=operators,
                    explain_encryption=explain_encryption,
                    reverse_task=True,
                    rng=r,
                    include_retry_metadata=include_retry_metadata,
                    retry_extra_pool_size=retry_extra_pool_size,
                )
            )
    return tasks


def write_jsonl(tasks: Sequence[PuzzleTask], path: str) -> None:
    """Write tasks to ``path`` in the ``{contents, references}`` wire shape."""
    with open(path, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t.to_jsonl_record()) + "\n")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="puzzle_dataset_gen",
        description=(
            "Generate a word-puzzle JSONL dataset compatible with "
            "WordPuzzleEnv (see word_puzzle.py). Purely offline: no LLM "
            "calls, no GCS I/O."
        ),
    )
    p.add_argument("--output", "-o", required=True, help="Path to output JSONL file.")
    p.add_argument(
        "--num-examples",
        type=int,
        default=100,
        help="Number of decrypt-direction examples (doubled if --flip-reverse-task).",
    )
    p.add_argument(
        "--num-transformations",
        type=int,
        default=2,
        help="How many operations composed into each puzzle's rule.",
    )
    p.add_argument(
        "--num-few-shot-examples",
        type=int,
        default=3,
        help="How many (original, encrypted) demo pairs shown in each prompt.",
    )
    p.add_argument(
        "--operator-set",
        choices=list(OPERATOR_SETS.keys()),
        default="extended",
        help="Operator pool: `basic` = shift/mirror/rotate; `extended` "
        "= basic + 7 fixed transforms; `withhold` = 12 eval-only "
        "encoding/cipher transforms; `all` = union.",
    )
    p.add_argument(
        "--word-list",
        default=None,
        help="Path to a plain-text word list (one word per line). If omitted, "
        "falls back to the `wordfreq` package.",
    )
    p.add_argument(
        "--min-word-length",
        type=int,
        default=5,
        help="Drop candidates shorter than this. 5 is the safer default "
        "for multi-op rules.",
    )
    p.add_argument(
        "--explain-encryption",
        action="store_true",
        default=True,
        help="Ask the model to describe the rule before answering (default: on).",
    )
    p.add_argument(
        "--no-explain-encryption",
        dest="explain_encryption",
        action="store_false",
        help="Do NOT include the 'first describe the rule' instruction.",
    )
    p.add_argument(
        "--flip-reverse-task",
        action="store_true",
        default=True,
        help="Also generate matching encrypt-direction tasks (doubles the "
        "output count). On by default.",
    )
    p.add_argument(
        "--no-flip-reverse-task",
        dest="flip_reverse_task",
        action="store_false",
        help="Only produce decrypt-direction tasks.",
    )
    p.add_argument(
        "--include-retry-metadata",
        action="store_true",
        default=False,
        help="Populate retry_metadata (extra_word_pool) on each task for "
        "agent-mode retry. Default off to avoid leaking ground-truth info.",
    )
    p.add_argument(
        "--retry-extra-pool-size",
        type=int,
        default=20,
        help="Number of extra words in the retry pool (only used with "
        "--include-retry-metadata). Default 20.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for the random number generator. If omitted, the run is "
        "non-deterministic.",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    operators = OPERATOR_SETS[args.operator_set]
    word_candidates = load_word_candidates(
        path=args.word_list, min_length=args.min_word_length
    )
    if not word_candidates:
        logger.error("word candidate list is empty.")
        return 2
    logger.info(
        "Loaded %d word candidates (min_length=%d, source=%s)",
        len(word_candidates),
        args.min_word_length,
        f"file:{args.word_list}" if args.word_list else "wordfreq",
    )
    logger.info("Operator set: %s (%d ops)", args.operator_set, len(operators))
    rng = random.Random(args.seed) if args.seed is not None else random.Random()
    tasks = generate_dataset(
        num_examples=args.num_examples,
        num_transformations=args.num_transformations,
        num_few_shot_examples=args.num_few_shot_examples,
        word_candidates=word_candidates,
        operators=operators,
        explain_encryption=args.explain_encryption,
        flip_reverse_task=args.flip_reverse_task,
        rng=rng,
        include_retry_metadata=args.include_retry_metadata,
        retry_extra_pool_size=args.retry_extra_pool_size,
    )
    write_jsonl(tasks, args.output)
    logger.info("Wrote %d tasks to %s", len(tasks), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
