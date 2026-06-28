import unittest
import pytest
from ollie_rl.trainer.types import Example
from ollie_rl.trainer.tinker.accumulator import (
    TrajectoryAccumulator,
    examples_to_data,
)


class TestTinkerAccumulator(unittest.TestCase):
    def _make_example(
        self,
        chat_completion_id: str,
        tokens: list[int],
        logprobs: list[float],
        advantage: float = 1.0,
    ) -> Example:
        return Example(
            chat_completion_id=chat_completion_id,
            advantage=advantage,
            policy_generation=0,
            tokens=tokens,
            logprobs=logprobs,
        )

    def test_trajectory_accumulator_prefix_checking(self):
        # Full sequence is [1, 2, 3] (prompt) + [4, 5] (completion) = [1, 2, 3, 4, 5]
        acc = TrajectoryAccumulator(
            full_tokens=[1, 2, 3, 4, 5],
            sampled_logprobs=[0.0, 0.0, 0.0, -0.1, -0.2],
            advantages=[0.0, 0.0, 0.0, 1.0, 1.0],
            mask=[0.0, 0.0, 0.0, 1.0, 1.0],
        )

        # Prompt [1, 2, 3, 4, 5, 6, 7] has [1, 2, 3, 4, 5] as a prefix
        self.assertTrue(acc.is_prefix_of([1, 2, 3, 4, 5, 6, 7]))

        # Prompt [1, 2, 3, 4, 9, 6, 7] does NOT have [1, 2, 3, 4, 5] as a prefix
        self.assertFalse(acc.is_prefix_of([1, 2, 3, 4, 9, 6, 7]))

        # Shorter prompt [1, 2, 3] is NOT prefixed by the longer accumulator
        self.assertFalse(acc.is_prefix_of([1, 2, 3]))

    def test_single_turn_reconstruction(self):
        # Prompt: [1, 2, 3], Completion: [4, 5]
        ex = self._make_example(
            chat_completion_id="ex-1",
            tokens=[1, 2, 3, 4, 5],
            logprobs=[-0.1, -0.2],
            advantage=1.5,
        )

        data = examples_to_data([ex])
        self.assertEqual(len(data), 1)

        datum = data[0]
        # input_tokens should be full_tokens[:-1] = [1, 2, 3, 4]
        self.assertEqual(datum.model_input.to_ints(), [1, 2, 3, 4])

        # loss_fn_inputs are sliced [1:]
        target_tokens = datum.loss_fn_inputs["target_tokens"].to_torch()
        logprobs = datum.loss_fn_inputs["logprobs"].to_torch()
        advantages = datum.loss_fn_inputs["advantages"].to_torch()
        mask = datum.loss_fn_inputs["mask"].to_torch()

        # target_tokens should be [2, 3, 4, 5]
        self.assertEqual(target_tokens.tolist(), [2, 3, 4, 5])
        # mask should be [0.0, 0.0, 1.0, 1.0] (shifted prompt is 0, completion is 1.0)
        self.assertEqual(mask.tolist(), [0.0, 0.0, 1.0, 1.0])
        # advantages should be [0.0, 0.0, 1.5, 1.5]
        self.assertEqual(advantages.tolist(), [0.0, 0.0, 1.5, 1.5])
        # logprobs should be [0.0, 0.0, -0.1, -0.2]
        self.assertEqual(logprobs.tolist(), pytest.approx([0.0, 0.0, -0.1, -0.2]))

    def test_multi_turn_linear_reconstruction(self):
        # Turn 1: Prompt [1, 2], Completion [3, 4] (Full: [1, 2, 3, 4])
        ex1 = self._make_example(
            chat_completion_id="turn-1",
            tokens=[1, 2, 3, 4],
            logprobs=[-0.1, -0.2],
            advantage=1.0,
        )
        # Turn 2: Prompt [1, 2, 3, 4, 5, 6] (extends Turn 1 with observation [5, 6]), Completion [7, 8]
        ex2 = self._make_example(
            chat_completion_id="turn-2",
            tokens=[1, 2, 3, 4, 5, 6, 7, 8],
            logprobs=[-0.3, -0.4],
            advantage=2.0,
        )

        # Pass both to examples_to_data
        data = examples_to_data([ex1, ex2])
        # They should be aggregated into a single trajectory Datum!
        self.assertEqual(len(data), 1)

        datum = data[0]
        # full sequence = [1, 2, 3, 4] + [5, 6] + [7, 8] = [1, 2, 3, 4, 5, 6, 7, 8]
        # input_tokens = full_seq[:-1] = [1, 2, 3, 4, 5, 6, 7]
        self.assertEqual(datum.model_input.to_ints(), [1, 2, 3, 4, 5, 6, 7])

        target_tokens = datum.loss_fn_inputs["target_tokens"].to_torch().tolist()
        logprobs = datum.loss_fn_inputs["logprobs"].to_torch().tolist()
        advantages = datum.loss_fn_inputs["advantages"].to_torch().tolist()
        mask = datum.loss_fn_inputs["mask"].to_torch().tolist()

        # full targets = [2, 3, 4, 5, 6, 7, 8]
        self.assertEqual(target_tokens, [2, 3, 4, 5, 6, 7, 8])

        # Full layout before shift:
        # tokens:      1,   2,    3,    4,    5,   6,    7,    8
        # mask:        0,   0,   1.0,  1.0,   0,   0,   1.0,  1.0
        # advantages:  0,   0,   1.0,  1.0,   0,   0,   2.0,  2.0
        # logprobs:    0,   0,  -0.1, -0.2,   0,   0,  -0.3, -0.4
        #
        # After shift [1:]:
        # tokens:      2,    3,    4,    5,   6,    7,    8
        # mask:        0,   1.0,  1.0,   0,   0,   1.0,  1.0
        # advantages:  0,   1.0,  1.0,   0,   0,   2.0,  2.0
        # logprobs:    0,  -0.1, -0.2,   0,   0,  -0.3, -0.4

        self.assertEqual(mask, [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
        self.assertEqual(advantages, [0.0, 1.0, 1.0, 0.0, 0.0, 2.0, 2.0])
        self.assertEqual(
            logprobs, pytest.approx([0.0, -0.1, -0.2, 0.0, 0.0, -0.3, -0.4])
        )

    def test_parallel_branches_reconstruction(self):
        # Two parallel runs starting with prompt [1, 2]
        # Run A Turn 1: Prompt [1, 2], Completion [3, 4] (Full: [1, 2, 3, 4])
        ex_a1 = self._make_example(
            chat_completion_id="run-a-1",
            tokens=[1, 2, 3, 4],
            logprobs=[-0.1, -0.1],
            advantage=1.0,
        )
        # Run B Turn 1: Prompt [1, 2], Completion [5, 6] (Full: [1, 2, 5, 6])
        ex_b1 = self._make_example(
            chat_completion_id="run-b-1",
            tokens=[1, 2, 5, 6],
            logprobs=[-0.2, -0.2],
            advantage=1.5,
        )
        # Run A Turn 2: Prompt [1, 2, 3, 4, 7, 8], Completion [9]
        ex_a2 = self._make_example(
            chat_completion_id="run-a-2",
            tokens=[1, 2, 3, 4, 7, 8, 9],
            logprobs=[-0.3],
            advantage=2.0,
        )
        # Run B Turn 2: Prompt [1, 2, 5, 6, 7, 8], Completion [10]
        ex_b2 = self._make_example(
            chat_completion_id="run-b-2",
            tokens=[1, 2, 5, 6, 7, 8, 10],
            logprobs=[-0.4],
            advantage=2.5,
        )

        # Pass all 4 to examples_to_data
        data = examples_to_data([ex_a1, ex_b1, ex_a2, ex_b2])
        # They should form exactly 2 distinct trajectory Datums!
        self.assertEqual(len(data), 2)

        # Sort by input length to identify them easily
        data.sort(key=lambda d: d.model_input.length)

        # Both have length 6 (full sequence 7 - 1)
        self.assertEqual(data[0].model_input.length, 6)
        self.assertEqual(data[1].model_input.length, 6)

        inputs = {tuple(d.model_input.to_ints()) for d in data}
        # One should be [1, 2, 3, 4, 7, 8] and the other [1, 2, 5, 6, 7, 8]
        self.assertIn((1, 2, 3, 4, 7, 8), inputs)
        self.assertIn((1, 2, 5, 6, 7, 8), inputs)

    def test_identical_prefix_collision_resolution(self):
        # Run A and Run B are identical on Turn 1: Prompt [1, 2], Completion [3, 4]
        ex_a1 = self._make_example(
            chat_completion_id="run-a-1",
            tokens=[1, 2, 3, 4],
            logprobs=[-0.1, -0.1],
            advantage=1.0,
        )
        ex_b1 = self._make_example(
            chat_completion_id="run-b-1",
            tokens=[1, 2, 3, 4],
            logprobs=[-0.1, -0.1],
            advantage=1.0,
        )
        # Turn 2 branches:
        # Run A Turn 2: Prompt [1, 2, 3, 4, 5] (observation [5]), Completion [7]
        ex_a2 = self._make_example(
            chat_completion_id="run-a-2",
            tokens=[1, 2, 3, 4, 5, 7],
            logprobs=[-0.3],
            advantage=2.0,
        )
        # Run B Turn 2: Prompt [1, 2, 3, 4, 6] (observation [6]), Completion [8]
        ex_b2 = self._make_example(
            chat_completion_id="run-b-2",
            tokens=[1, 2, 3, 4, 6, 8],
            logprobs=[-0.4],
            advantage=2.5,
        )

        data = examples_to_data([ex_a1, ex_b1, ex_a2, ex_b2])
        # Should resolve into 2 distinct trajectory Datums
        self.assertEqual(len(data), 2)

        inputs = {tuple(d.model_input.to_ints()) for d in data}
        self.assertIn((1, 2, 3, 4, 5), inputs)
        self.assertIn((1, 2, 3, 4, 6), inputs)

    def test_missing_data_skipped(self):
        bad_ex = Example(
            chat_completion_id="bad",
            advantage=1.0,
            policy_generation=0,
            tokens=None,
            logprobs=None,
        )
        data = examples_to_data([bad_ex])
        self.assertEqual(len(data), 0)
