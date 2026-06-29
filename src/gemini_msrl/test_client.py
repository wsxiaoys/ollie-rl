import os
import tempfile
import unittest
from unittest.mock import patch, AsyncMock, MagicMock
from gemini_msrl.client import GeminiMsrlClient


class TestGeminiMsrlClient(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.env_file_path = os.path.join(self.temp_dir.name, ".env")
        with open(self.env_file_path, "w") as f:
            f.write("GEMINI_MSRL_AUTH_TOKEN=env-file-token\n")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_init_with_env_vars(self):
        with patch.dict(
            "os.environ",
            {
                "GEMINI_MSRL_PROJECT_ID": "env-project",
                "GEMINI_MSRL_ENV_FILE": self.env_file_path,
            },
        ):
            client = GeminiMsrlClient()
            self.assertEqual(client.project_id, "env-project")
            self.assertEqual(client.location, "us-central1")
            self.assertEqual(client._current_token(), "env-file-token")
            self.assertEqual(
                client.base_url,
                "https://us-central1-staging-aiplatform.sandbox.googleapis.com",
            )

    def test_init_with_forwarder_mode_env(self):
        with patch.dict(
            "os.environ",
            {
                "GEMINI_MSRL_PROJECT_ID": "env-project",
                "GEMINI_MSRL_ENV_FILE": self.env_file_path,
                "GEMINI_MSRL_FORWARDER_MODE": "real",
            },
        ):
            client = GeminiMsrlClient()
            self.assertEqual(client.headers["X-Forwarder-Mode"], "real")

    def test_init_with_custom_base_url_env(self):
        with patch.dict(
            "os.environ",
            {
                "GEMINI_MSRL_PROJECT_ID": "env-project",
                "GEMINI_MSRL_ENV_FILE": self.env_file_path,
                "GEMINI_MSRL_BASE_URL": "https://custom.api.com",
            },
        ):
            client = GeminiMsrlClient()
            self.assertEqual(client.base_url, "https://custom.api.com")

    async def test_cancel_tuning_job(self):
        with patch.dict(
            "os.environ",
            {
                "GEMINI_MSRL_PROJECT_ID": "env-project",
                "GEMINI_MSRL_ENV_FILE": self.env_file_path,
            },
        ):
            client = GeminiMsrlClient()
            with patch(
                "httpx.AsyncClient.request", new_callable=AsyncMock
            ) as mock_request:
                mock_response = MagicMock()
                mock_response.is_error = False
                mock_response.json.return_value = {}
                mock_request.return_value = mock_response

                await client.cancel_tuning_job("projects/p1/locations/l1/tuningJobs/j1")

                args, kwargs = mock_request.call_args
                method, url = args
                self.assertEqual(method, "POST")
                self.assertTrue(
                    url.endswith(
                        "/v1beta1/projects/p1/locations/l1/tuningJobs/j1:cancel"
                    )
                )

    def test_init_missing_env_file_raises_error(self):
        with patch.dict("os.environ", {"GEMINI_MSRL_PROJECT_ID": "env-project"}):
            with self.assertRaises(ValueError) as ctx:
                GeminiMsrlClient()
            self.assertIn(
                "GEMINI_MSRL_ENV_FILE environment variable must be set",
                str(ctx.exception),
            )

    def test_init_missing_project_id_raises_error(self):
        with patch.dict("os.environ", {"GEMINI_MSRL_ENV_FILE": self.env_file_path}):
            with self.assertRaises(ValueError) as ctx:
                GeminiMsrlClient()
            self.assertIn(
                "GEMINI_MSRL_PROJECT_ID environment variable must be set",
                str(ctx.exception),
            )
