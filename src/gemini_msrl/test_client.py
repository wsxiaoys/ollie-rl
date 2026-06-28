import os
import tempfile
import unittest
from unittest.mock import patch
from gemini_msrl.client import GeminiMsrlClient


class TestGeminiMsrlClientInit(unittest.TestCase):
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
