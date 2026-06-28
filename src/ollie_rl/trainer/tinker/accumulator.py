import logging
from typing import List, Optional
import torch
import tinker

from ollie_rl.trainer.types import Example

logger = logging.getLogger(__name__)


class TrajectoryAccumulator:
    def __init__(
        self,
        full_tokens: List[int],
        sampled_logprobs: List[float],
        advantages: List[float],
        mask: List[float],
    ):
        self.full_tokens = list(full_tokens)
        self.sampled_logprobs = list(sampled_logprobs)
        self.advantages = list(advantages)
        self.mask = list(mask)

    def is_prefix_of(self, other_tokens: List[int]) -> bool:
        """Check if this accumulator's full token sequence is a prefix of other_tokens."""
        if len(self.full_tokens) > len(other_tokens):
            return False
        return other_tokens[: len(self.full_tokens)] == self.full_tokens

    def extend(
        self,
        new_prompt_tokens: List[int],
        completion_tokens: List[int],
        completion_logprobs: List[float],
        advantage: float,
    ):
        """Extend the trajectory with a new turn."""
        pad_len = len(new_prompt_tokens)
        self.full_tokens.extend(new_prompt_tokens)
        self.sampled_logprobs.extend([0.0] * pad_len)
        self.advantages.extend([0.0] * pad_len)
        self.mask.extend([0.0] * pad_len)

        self.full_tokens.extend(completion_tokens)
        self.sampled_logprobs.extend(completion_logprobs)
        self.advantages.extend([advantage] * len(completion_tokens))
        self.mask.extend([1.0] * len(completion_tokens))


class ParsedExample:
    def __init__(
        self,
        example: Example,
        prompt_tokens: List[int],
        completion_tokens: List[int],
        logprobs: List[float],
    ):
        self.example = example
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.logprobs = logprobs


def examples_to_data(examples: List[Example]) -> List[tinker.Datum]:
    """
    Groups examples into trajectory-level tinker.Datums by reconstructing
    the prefix-tree structure of their tokens.
    """
    parsed_examples: List[ParsedExample] = []
    for ex in examples:
        if ex.tokens is None or ex.logprobs is None:
            logger.warning(
                f"TinkerTrainer skipping example {ex.chat_completion_id}: "
                "no cached tokens/logprobs"
            )
            continue

        completion_len = len(ex.logprobs)
        prompt_len = len(ex.tokens) - completion_len
        if prompt_len < 1 or completion_len < 1:
            logger.warning(
                f"TinkerTrainer skipping example {ex.chat_completion_id}: "
                f"degenerate prompt_len={prompt_len}, completion_len={completion_len}"
            )
            continue

        parsed_examples.append(
            ParsedExample(
                example=ex,
                prompt_tokens=ex.tokens[:prompt_len],
                completion_tokens=ex.tokens[prompt_len:],
                logprobs=ex.logprobs,
            )
        )

    # Sort by prompt length ascending to make partitioning greedy
    parsed_examples.sort(key=lambda x: len(x.prompt_tokens))

    accumulators: List[TrajectoryAccumulator] = []

    for item in parsed_examples:
        prompt = item.prompt_tokens
        comp = item.completion_tokens
        lps = item.logprobs
        adv = item.example.advantage

        # Find the accumulator whose full sequence is the longest prefix of the current prompt
        best_acc: Optional[TrajectoryAccumulator] = None
        for acc in accumulators:
            if acc.is_prefix_of(prompt):
                if best_acc is None or len(acc.full_tokens) > len(best_acc.full_tokens):
                    best_acc = acc

        if best_acc is not None:
            # Extend existing trajectory
            delta_prompt = prompt[len(best_acc.full_tokens) :]
            best_acc.extend(delta_prompt, comp, lps, adv)
        else:
            # Create a new trajectory
            new_acc = TrajectoryAccumulator(
                full_tokens=prompt + comp,
                sampled_logprobs=[0.0] * len(prompt) + list(lps),
                advantages=[0.0] * len(prompt) + [adv] * len(comp),
                mask=[0.0] * len(prompt) + [1.0] * len(comp),
            )
            accumulators.append(new_acc)

    data: List[tinker.Datum] = []
    for acc in accumulators:
        full_tokens = acc.full_tokens
        sampled_logprobs = acc.sampled_logprobs
        advantages = acc.advantages
        mask = acc.mask

        # Right-shift/left-shift slicing to align loss inputs with target_tokens
        target_tokens = full_tokens[1:]
        sampled_logprobs = sampled_logprobs[1:]
        advantages = advantages[1:]
        mask = mask[1:]
        input_tokens = tinker.ModelInput.from_ints(tokens=full_tokens[:-1])

        data.append(
            tinker.Datum(
                model_input=input_tokens,
                loss_fn_inputs={
                    "target_tokens": tinker.TensorData.from_torch(
                        torch.tensor(target_tokens, dtype=torch.int64)
                    ),
                    "logprobs": tinker.TensorData.from_torch(
                        torch.tensor(sampled_logprobs, dtype=torch.float32)
                    ),
                    "advantages": tinker.TensorData.from_torch(
                        torch.tensor(advantages, dtype=torch.float32)
                    ),
                    "mask": tinker.TensorData.from_torch(
                        torch.tensor(mask, dtype=torch.float32)
                    ),
                },
            )
        )

    return data
