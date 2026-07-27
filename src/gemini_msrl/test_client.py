import os
import unittest
from unittest.mock import patch

import httpx

from gemini_msrl import GeminiMsrlClient, GeminiMsrlHttpError


class FakeTokenSource:
    def __init__(self, tokens: list[str] | None = None):
        self.tokens = tokens or ["test-token"]
        self.calls: list[bool] = []

    async def get_token(self, *, force_refresh: bool = False) -> str:
        self.calls.append(force_refresh)
        index = min(len(self.calls) - 1, len(self.tokens) - 1)
        return self.tokens[index]


class TestGeminiMsrlClient(unittest.IsolatedAsyncioTestCase):
    def _client(
        self,
        *,
        token_source: FakeTokenSource | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> tuple[GeminiMsrlClient, httpx.AsyncClient]:
        async_client = httpx.AsyncClient(transport=transport)
        client = GeminiMsrlClient(
            client=async_client,
            token_source=token_source or FakeTokenSource(),
        )
        return client, async_client

    def test_init_with_explicit_token_source(self):
        source = FakeTokenSource()
        with patch.dict(
            os.environ,
            {
                "GEMINI_MSRL_PROJECT_ID": "env-project",
                "GEMINI_MSRL_TOKEN_URL": "https://broker.invalid/token",
            },
            clear=True,
        ):
            client = GeminiMsrlClient(token_source=source)

        self.assertEqual(client.project_id, "env-project")
        self.assertEqual(client.location, "us-central1")
        self.assertIs(client._token_source, source)
        self.assertEqual(
            client.base_url,
            "https://us-central1-aiplatform.googleapis.com",
        )

    def test_init_uses_application_default_credentials(self):
        with (
            patch.dict(
                os.environ,
                {
                    "GEMINI_MSRL_PROJECT_ID": "env-project",
                    "GOOGLE_APPLICATION_CREDENTIALS": "/secrets/service-account.json",
                    "GEMINI_MSRL_BASE_URL": "https://google-api.invalid",
                },
                clear=True,
            ),
            patch("gemini_msrl.client.GoogleAuthTokenSource.default") as factory,
        ):
            source = FakeTokenSource()
            factory.return_value = source
            client = GeminiMsrlClient()

        factory.assert_called_once_with()
        self.assertIs(client._token_source, source)
        self.assertEqual(client.base_url, "https://google-api.invalid")

    def test_init_with_forwarder_mode(self):
        with patch.dict(
            os.environ,
            {
                "GEMINI_MSRL_PROJECT_ID": "env-project",
                "GEMINI_MSRL_FORWARDER_MODE": "real",
            },
            clear=True,
        ):
            client = GeminiMsrlClient(token_source=FakeTokenSource())

        self.assertEqual(client.headers["X-Forwarder-Mode"], "real")

    def test_init_defaults_to_adc_without_credential_environment(self):
        with (
            patch.dict(
                os.environ,
                {"GEMINI_MSRL_PROJECT_ID": "env-project"},
                clear=True,
            ),
            patch("gemini_msrl.client.GoogleAuthTokenSource.default") as factory,
        ):
            source = FakeTokenSource()
            factory.return_value = source
            client = GeminiMsrlClient()

        self.assertIs(client._token_source, source)
        factory.assert_called_once_with()

    def test_removed_env_file_auth_does_not_override_adc(self):
        with (
            patch.dict(
                os.environ,
                {
                    "GEMINI_MSRL_PROJECT_ID": "env-project",
                    "GEMINI_MSRL_ENV_FILE": "/tmp/legacy.env",
                    "GEMINI_MSRL_AUTH_TOKEN": "legacy-token",
                },
                clear=True,
            ),
            patch("gemini_msrl.client.GoogleAuthTokenSource.default") as factory,
        ):
            source = FakeTokenSource()
            factory.return_value = source
            client = GeminiMsrlClient()

        self.assertIs(client._token_source, source)
        factory.assert_called_once_with()

    def test_init_missing_project_id_raises_error(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            self.assertRaisesRegex(ValueError, "GEMINI_MSRL_PROJECT_ID"),
        ):
            GeminiMsrlClient(token_source=FakeTokenSource())

    async def test_request_injects_token(self):
        requests: list[httpx.Request] = []

        async def handle(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={}, request=request)

        with patch.dict(
            os.environ,
            {"GEMINI_MSRL_PROJECT_ID": "env-project"},
            clear=True,
        ):
            source = FakeTokenSource()
            client, async_client = self._client(
                token_source=source,
                transport=httpx.MockTransport(handle),
            )

        try:
            await client.cancel_tuning_job("projects/p1/locations/l1/tuningJobs/j1")
        finally:
            await async_client.aclose()

        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].method, "POST")
        self.assertTrue(
            str(requests[0].url).endswith(
                "/v1beta1/projects/p1/locations/l1/tuningJobs/j1:cancel"
            )
        )
        self.assertEqual(requests[0].headers["Authorization"], "Bearer test-token")
        self.assertEqual(source.calls, [False])

    async def test_401_forces_one_refresh_and_replay(self):
        requests: list[httpx.Request] = []

        async def handle(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            status = 401 if len(requests) == 1 else 200
            return httpx.Response(status, json={}, request=request)

        with patch.dict(
            os.environ,
            {"GEMINI_MSRL_PROJECT_ID": "env-project"},
            clear=True,
        ):
            source = FakeTokenSource(["old-token", "new-token"])
            client, async_client = self._client(
                token_source=source,
                transport=httpx.MockTransport(handle),
            )

        try:
            await client.cancel_tuning_job("projects/p1/locations/l1/tuningJobs/j1")
        finally:
            await async_client.aclose()

        self.assertEqual(source.calls, [False, True])
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0].headers["Authorization"], "Bearer old-token")
        self.assertEqual(requests[1].headers["Authorization"], "Bearer new-token")

    async def test_non_401_error_is_not_retried(self):
        calls = 0

        async def handle(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(
                403,
                json={"error": {"message": "forbidden"}},
                request=request,
            )

        with patch.dict(
            os.environ,
            {"GEMINI_MSRL_PROJECT_ID": "env-project"},
            clear=True,
        ):
            source = FakeTokenSource()
            client, async_client = self._client(
                token_source=source,
                transport=httpx.MockTransport(handle),
            )

        try:
            with self.assertRaises(GeminiMsrlHttpError) as context:
                await client.cancel_tuning_job("projects/p1/locations/l1/tuningJobs/j1")
        finally:
            await async_client.aclose()

        self.assertEqual(context.exception.status_code, 403)
        self.assertEqual(calls, 1)
        self.assertEqual(source.calls, [False])

    async def test_supplied_http_client_is_not_closed(self):
        with patch.dict(
            os.environ,
            {"GEMINI_MSRL_PROJECT_ID": "env-project"},
            clear=True,
        ):
            client, async_client = self._client()

        await client.close()
        self.assertFalse(async_client.is_closed)
        await async_client.aclose()
