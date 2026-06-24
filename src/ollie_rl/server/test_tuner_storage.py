import unittest
import asyncio
from unittest.mock import AsyncMock, patch
from ollie_rl.server.tuner_storage import TunerStorage
from ollie_rl.cookbook import Recipe, Tuner


class DummyTuner(Tuner):
    def __init__(self, model_id: str, kind: str, state: str):
        self._model_id = model_id
        self._kind = kind
        self._state = state

    @property
    def kind(self) -> str:
        return self._kind

    async def sample(self, request):
        pass

    async def train_step(self, examples):
        pass

    async def save_state(self) -> str:
        return self._state


class DummyRecipe(Recipe):
    async def create(self, model_id: str) -> DummyTuner:
        return DummyTuner(model_id, "dummy", "state")

    async def restore(self, state: str) -> DummyTuner:
        return DummyTuner("dummy_model", "dummy", state)


class TestTunerStorage(unittest.TestCase):
    def setUp(self):
        self.storage = TunerStorage()
        asyncio.run(self.storage.client.flushall())

    def tearDown(self):
        asyncio.run(self.storage.close())

    def test_load_state_empty(self):
        records = asyncio.run(self.storage.load_state())
        self.assertEqual(records, {})

    def test_register_and_get_tuner(self):
        tuner = DummyTuner("model-1", "dummy_kind", "opaque_state_123")

        # Register the tuner (adds to memory & persists to Redis)
        asyncio.run(self.storage.register_tuner("model-1", tuner))

        # Check in-memory retrieval
        self.assertEqual(self.storage.get("model-1"), tuner)
        self.assertEqual(self.storage.list_keys(), ["model-1"])

        # Check persisted state in Redis
        records = asyncio.run(self.storage.load_state())
        self.assertIn("model-1", records)
        self.assertEqual(records["model-1"].kind, "dummy_kind")
        self.assertEqual(records["model-1"].state, "opaque_state_123")

    def test_save_all_tuners(self):
        tuner1 = DummyTuner("model-1", "dummy_kind_1", "state_1")
        tuner2 = DummyTuner("model-2", "dummy_kind_2", "state_2")

        self.storage.active_tuners = {
            "model-1": tuner1,
            "model-2": tuner2,
        }

        # Save all active tuners
        asyncio.run(self.storage.save_all_tuners())

        records = asyncio.run(self.storage.load_state())
        self.assertEqual(len(records), 2)
        self.assertEqual(records["model-1"].kind, "dummy_kind_1")
        self.assertEqual(records["model-1"].state, "state_1")
        self.assertEqual(records["model-2"].kind, "dummy_kind_2")
        self.assertEqual(records["model-2"].state, "state_2")

    def test_delete_tuner(self):
        tuner1 = DummyTuner("model-1", "dummy_kind_1", "state_1")
        tuner2 = DummyTuner("model-2", "dummy_kind_2", "state_2")

        asyncio.run(self.storage.register_tuner("model-1", tuner1))
        asyncio.run(self.storage.register_tuner("model-2", tuner2))

        # Delete model-1
        asyncio.run(self.storage.delete_tuner("model-1"))

        # Check in-memory state
        self.assertIsNone(self.storage.get("model-1"))
        self.assertEqual(self.storage.get("model-2"), tuner2)

        # Check persisted state in Redis
        records = asyncio.run(self.storage.load_state())
        self.assertNotIn("model-1", records)
        self.assertIn("model-2", records)

    @patch("ollie_rl.cookbook.Cookbook.restore", new_callable=AsyncMock)
    def test_restore_tuners(self, mock_restore):
        # Setup mock restore to return a dummy tuner
        mock_restore.side_effect = lambda kind, state: DummyTuner(
            "model-1", kind, state
        )

        # Pre-populate some records in Redis
        tuner = DummyTuner("model-1", "dummy_kind_1", "state_1")
        self.storage.active_tuners = {"model-1": tuner}
        asyncio.run(self.storage.save_all_tuners())

        # Clear in-memory state to simulate server restart
        self.storage.active_tuners = {}

        # Restore tuners from Redis
        asyncio.run(self.storage.restore_tuners())

        # Check in-memory state after restoration
        restored_tuner = self.storage.get("model-1")
        self.assertIsNotNone(restored_tuner)
        assert restored_tuner is not None
        self.assertEqual(restored_tuner.kind, "dummy_kind_1")
        self.assertEqual(asyncio.run(restored_tuner.save_state()), "state_1")
        mock_restore.assert_called_once_with("dummy_kind_1", "state_1")


if __name__ == "__main__":
    unittest.main()
