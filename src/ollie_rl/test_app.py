import os
import unittest
import httpx
from ollie_rl.db import init_db, shutdown_db
from ollie_rl.server.app import app


class TestAppSqlite(unittest.IsolatedAsyncioTestCase):
    db_file = "data/test_db.db"
    db_url = f"sqlite+aiosqlite:///{db_file}"

    async def asyncSetUp(self):
        # Clean up any leftover db file
        if os.path.exists(self.db_file):
            os.remove(self.db_file)

        # Initialize the database with a real SQLite file
        await init_db(self.db_url)

    async def asyncTearDown(self):
        # Shutdown database connection
        await shutdown_db()

    async def test_full_workflow(self):
        # Use httpx.AsyncClient with ASGI transport to call the FastAPI app directly
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # 0. Test redirect from / to /docs
            res = await client.get("/")
            self.assertEqual(res.status_code, 307)
            self.assertEqual(res.headers["location"], "/docs")

            # 1. Create a tuner
            create_tuner_payload = {
                "name": "integration-test-policy",
                "recipe": "grpo_16x32",
                "trainer": "fake",
                "datum_ids": ["task-1", "task-2", "task-3"],
            }
            res = await client.post("/tuners", json=create_tuner_payload)
            self.assertEqual(res.status_code, 200)
            tuner_data = res.json()
            self.assertIn("tuner_id", tuner_data)
            tuner_id = tuner_data["tuner_id"]
            self.assertEqual(tuner_data["name"], "integration-test-policy")
            self.assertEqual(tuner_data["recipe"], "grpo_16x32")

            # 1.5 Get tuner details
            res = await client.get(f"/tuners/{tuner_id}")
            self.assertEqual(res.status_code, 200)
            tuner_details = res.json()
            self.assertEqual(tuner_details["tuner_id"], tuner_id)
            self.assertEqual(tuner_details["name"], "integration-test-policy")
            self.assertEqual(tuner_details["recipe"], "grpo_16x32")
            self.assertEqual(tuner_details["trainer"], "fake")
            self.assertEqual(tuner_details["policy_generation"], 0)
            self.assertIsNone(tuner_details["state"])

            # Test 404 for non-existent tuner
            res = await client.get("/tuners/non-existent-id")
            self.assertEqual(res.status_code, 404)

            # 2. Dispense a run
            res = await client.post(f"/tuners/{tuner_id}/runs")
            self.assertEqual(res.status_code, 200)
            run_data = res.json()
            self.assertIn("run_id", run_data)
            self.assertIn("datum_id", run_data)
            run_id = run_data["run_id"]
            datum_id = run_data["datum_id"]
            self.assertIn(datum_id, ["task-1", "task-2", "task-3"])

            # 3. Request a chat completion
            completion_payload = {
                "model": "fake-model",
                "messages": [
                    {"role": "user", "content": f"Solve this task: {datum_id}"}
                ],
            }
            res = await client.post(
                "/openai/v1/chat/completions",
                json=completion_payload,
                headers={"X-Tuner-Id": tuner_id, "X-Run-Id": run_id},
            )
            self.assertEqual(res.status_code, 200)
            comp_data = res.json()
            self.assertIn("id", comp_data)
            self.assertEqual(
                comp_data["choices"][0]["message"]["content"],
                "This is a fake completion response from ollie-rl fake trainer.",
            )

            # 4. Submit a reward
            reward_payload = {"reward": 1.0}
            res = await client.put(
                f"/tuners/{tuner_id}/runs/{run_id}/reward", json=reward_payload
            )
            self.assertEqual(res.status_code, 200)
            reward_data = res.json()
            self.assertEqual(reward_data["run_id"], run_id)
            self.assertEqual(reward_data["reward"], 1.0)
