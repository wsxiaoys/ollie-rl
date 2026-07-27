import asyncio
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx

from gemini_msrl import GeminiMsrlAuthError, GoogleAuthTokenSource
from gemini_msrl.auth import _HttpTokenSource


class TestGoogleAuthTokenSource(unittest.IsolatedAsyncioTestCase):
    async def test_service_account_loads_lazily_refreshes_and_caches(self):
        credentials = MagicMock()
        credentials.valid = False
        credentials.token = None

        def refresh(_request):
            credentials.token = "access-token"
            credentials.valid = True

        credentials.refresh.side_effect = refresh

        with patch(
            "gemini_msrl.auth.ServiceAccountCredentials.from_service_account_file",
            return_value=credentials,
        ) as load:
            source = GoogleAuthTokenSource.from_service_account_file(
                "/secrets/service-account.json"
            )
            load.assert_not_called()

            self.assertEqual(await source.get_token(), "access-token")
            self.assertEqual(await source.get_token(), "access-token")

        load.assert_called_once_with(
            Path("/secrets/service-account.json"),
            scopes=("https://www.googleapis.com/auth/cloud-platform",),
        )
        credentials.refresh.assert_called_once()

    async def test_service_account_info_loads_lazily(self):
        credentials = MagicMock()
        credentials.valid = False
        credentials.token = None

        def refresh(_request):
            credentials.token = "info-access-token"
            credentials.valid = True

        credentials.refresh.side_effect = refresh
        info = {
            "type": "service_account",
            "client_email": "ollie-msrl-dev@example.invalid",
            "private_key": "secret-key",
        }

        with patch(
            "gemini_msrl.auth.ServiceAccountCredentials.from_service_account_info",
            return_value=credentials,
        ) as load:
            source = GoogleAuthTokenSource.from_service_account_info(info)
            load.assert_not_called()
            self.assertEqual(await source.get_token(), "info-access-token")

        load.assert_called_once_with(
            info,
            scopes=("https://www.googleapis.com/auth/cloud-platform",),
        )
        credentials.refresh.assert_called_once()

    async def test_default_uses_application_default_credentials(self):
        credentials = MagicMock()
        credentials.valid = False
        credentials.token = None

        def refresh(_request):
            credentials.token = "adc-token"
            credentials.valid = True

        credentials.refresh.side_effect = refresh
        with patch(
            "gemini_msrl.auth.google.auth.default",
            return_value=(credentials, "credential-project"),
        ) as load:
            source = GoogleAuthTokenSource.default()
            load.assert_not_called()
            self.assertEqual(await source.get_token(), "adc-token")

        load.assert_called_once_with(
            scopes=("https://www.googleapis.com/auth/cloud-platform",)
        )

    async def test_force_refresh_bypasses_valid_cache(self):
        credentials = MagicMock()
        credentials.valid = True
        credentials.token = "first-token"

        def refresh(_request):
            credentials.token = "refreshed-token"

        credentials.refresh.side_effect = refresh
        source = GoogleAuthTokenSource.default()
        source._credentials = credentials

        self.assertEqual(
            await source.get_token(force_refresh=True),
            "refreshed-token",
        )
        credentials.refresh.assert_called_once()

    async def test_concurrent_initial_requests_share_refresh(self):
        credentials = MagicMock()
        credentials.valid = False
        credentials.token = None

        def refresh(_request):
            credentials.token = "shared-token"
            credentials.valid = True

        credentials.refresh.side_effect = refresh

        with patch(
            "gemini_msrl.auth.google.auth.default",
            return_value=(credentials, None),
        ) as load:
            source = GoogleAuthTokenSource.default()
            tokens = await asyncio.gather(*(source.get_token() for _ in range(8)))

        self.assertEqual(tokens, ["shared-token"] * 8)
        load.assert_called_once()
        credentials.refresh.assert_called_once()

    async def test_credential_error_is_sanitized_and_chained(self):
        secret = "private-key-material"
        with patch(
            "gemini_msrl.auth.google.auth.default",
            side_effect=ValueError(secret),
        ):
            source = GoogleAuthTokenSource.default()
            with self.assertRaises(GeminiMsrlAuthError) as context:
                await source.get_token()

        self.assertNotIn(secret, str(context.exception))
        self.assertIsInstance(context.exception.__cause__, ValueError)

    async def test_rejects_empty_scopes(self):
        with self.assertRaisesRegex(ValueError, "scope"):
            GoogleAuthTokenSource.default(scopes=())


class TestHttpTokenSource(unittest.IsolatedAsyncioTestCase):
    async def test_fetches_and_caches_broker_token(self):
        calls = 0

        async def handle(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(200, json={"token": "broker-token"}, request=request)

        source = _HttpTokenSource("https://broker.invalid/token")
        source._client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        try:
            self.assertEqual(await source.get_token(), "broker-token")
            self.assertEqual(await source.get_token(), "broker-token")
        finally:
            await source.close()

        self.assertEqual(calls, 1)

    async def test_refresh_failure_keeps_cached_token(self):
        calls = 0

        async def handle(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(
                    200,
                    json={"token": "cached-token"},
                    request=request,
                )
            raise httpx.ConnectError("broker unavailable", request=request)

        source = _HttpTokenSource("https://broker.invalid/token")
        source._client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        try:
            self.assertEqual(await source.get_token(), "cached-token")
            self.assertEqual(
                await source.get_token(force_refresh=True),
                "cached-token",
            )
        finally:
            await source.close()

        self.assertEqual(calls, 2)

    async def test_first_malformed_response_raises_auth_error(self):
        async def handle(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={}, request=request)

        source = _HttpTokenSource("https://broker.invalid/token")
        source._client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        try:
            with self.assertRaises(GeminiMsrlAuthError):
                await source.get_token()
        finally:
            await source.close()
