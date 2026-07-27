from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import google.auth
import httpx
from google.auth.credentials import Credentials
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials as ServiceAccountCredentials

logger = logging.getLogger(__name__)


class GeminiMsrlError(Exception):
    """Base exception for the Gemini MSRL API client."""


class GeminiMsrlAuthError(GeminiMsrlError):
    """Raised when an authentication token cannot be acquired."""


@runtime_checkable
class TokenSource(Protocol):
    """Asynchronous provider of bearer tokens for Gemini MSRL requests."""

    async def get_token(self, *, force_refresh: bool = False) -> str: ...


GOOGLE_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


class GoogleAuthTokenSource:
    """Produces scoped OAuth access tokens using Google Auth credentials.

    Application Default Credentials discovery and token refresh are lazy. Both
    operations use a worker thread because google-auth's supported requests
    transport is synchronous.
    """

    def __init__(self, credential_loader: Callable[[], Credentials]):
        self._credential_loader = credential_loader
        self._credentials: Credentials | None = None
        self._lock = asyncio.Lock()

    @staticmethod
    def _normalize_scopes(scopes: Sequence[str]) -> tuple[str, ...]:
        normalized = tuple(scope for scope in scopes if scope.strip())
        if not normalized:
            raise ValueError("At least one Google OAuth scope must be configured")
        return normalized

    @classmethod
    def default(
        cls,
        *,
        scopes: Sequence[str] = (GOOGLE_CLOUD_PLATFORM_SCOPE,),
    ) -> GoogleAuthTokenSource:
        """Create a source using Application Default Credentials."""
        normalized_scopes = cls._normalize_scopes(scopes)

        def load() -> Credentials:
            credentials, _ = google.auth.default(scopes=normalized_scopes)
            return credentials

        return cls(load)

    @classmethod
    def from_service_account_file(
        cls,
        filename: str | os.PathLike[str],
        *,
        scopes: Sequence[str] = (GOOGLE_CLOUD_PLATFORM_SCOPE,),
    ) -> GoogleAuthTokenSource:
        """Create a source backed by an explicit service-account JSON file."""
        if not str(filename):
            raise ValueError("Service-account credential filename must not be empty")
        credential_path = Path(filename)
        normalized_scopes = cls._normalize_scopes(scopes)

        def load() -> Credentials:
            return ServiceAccountCredentials.from_service_account_file(
                credential_path,
                scopes=normalized_scopes,
            )

        return cls(load)

    @classmethod
    def from_service_account_info(
        cls,
        info: Mapping[str, Any],
        *,
        scopes: Sequence[str] = (GOOGLE_CLOUD_PLATFORM_SCOPE,),
    ) -> GoogleAuthTokenSource:
        """Create a source backed by an in-memory service-account object."""
        credential_info = dict(info)
        if not credential_info:
            raise ValueError("Service-account credential information must not be empty")
        normalized_scopes = cls._normalize_scopes(scopes)

        def load() -> Credentials:
            return ServiceAccountCredentials.from_service_account_info(
                credential_info,
                scopes=normalized_scopes,
            )

        return cls(load)

    @staticmethod
    def _refresh(credentials: Credentials) -> None:
        credentials.refresh(Request())

    @staticmethod
    def _valid_token(credentials: Credentials | None) -> str | None:
        if credentials is None or not credentials.valid:
            return None
        token = credentials.token
        return token if isinstance(token, str) and token else None

    async def get_token(self, *, force_refresh: bool = False) -> str:
        if not force_refresh:
            token = self._valid_token(self._credentials)
            if token is not None:
                return token

        async with self._lock:
            if not force_refresh:
                token = self._valid_token(self._credentials)
                if token is not None:
                    return token

            try:
                if self._credentials is None:
                    self._credentials = await asyncio.to_thread(self._credential_loader)
                await asyncio.to_thread(self._refresh, self._credentials)
            except Exception as exc:
                raise GeminiMsrlAuthError(
                    "Failed to load or refresh Google credentials"
                ) from exc

            token = self._valid_token(self._credentials)
            if token is None:
                raise GeminiMsrlAuthError(
                    "Google authentication refresh returned no valid access token"
                )
            return token


def _parse_env_file(path: Path) -> dict[str, str]:
    """Parse the simple KEY=VALUE format used by token refresh scripts."""
    values: dict[str, str] = {}
    try:
        with path.open() as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                    value = value[1:-1]
                values[key.strip()] = value
    except FileNotFoundError:
        pass
    return values


class _EnvFileTokenSource:
    """Token source that reloads a token when its environment file changes."""

    def __init__(self, env_path: str, key: str):
        self._path = Path(env_path)
        self._key = key
        self._cached_token: str | None = None
        self._cached_mtime_ns: int | None = None
        self._lock = asyncio.Lock()

    async def get_token(self, *, force_refresh: bool = False) -> str:
        async with self._lock:
            try:
                mtime_ns = self._path.stat().st_mtime_ns
            except FileNotFoundError:
                mtime_ns = None

            if force_refresh or mtime_ns != self._cached_mtime_ns:
                values = _parse_env_file(self._path)
                token = values.get(self._key)
                if token:
                    if token != self._cached_token:
                        logger.info(
                            "GeminiMsrlClient: refreshed %s from %s",
                            self._key,
                            self._path,
                        )
                    self._cached_token = token
                elif self._cached_token is not None:
                    logger.warning(
                        "GeminiMsrlClient: %s not present in %s; keeping cached token",
                        self._key,
                        self._path,
                    )
                self._cached_mtime_ns = mtime_ns

            if self._cached_token is None:
                raise GeminiMsrlAuthError(
                    f"{self._key} not found in environment file {self._path}"
                )
            return self._cached_token


class _HttpTokenSource:
    """Compatibility token source backed by a trusted HTTP token broker."""

    def __init__(
        self,
        url: str,
        *,
        refresh_interval: float = 60.0,
        timeout: float = 10.0,
    ):
        self._url = url
        self._refresh_interval = refresh_interval
        self._timeout = timeout
        self._cached_token: str | None = None
        self._fetched_at = 0.0
        self._client: httpx.AsyncClient | None = None
        self._lock = asyncio.Lock()

    def _is_stale(self) -> bool:
        return self._cached_token is None or time.time() >= (
            self._fetched_at + self._refresh_interval
        )

    async def _fetch(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)

        try:
            response = await self._client.get(self._url)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            if self._cached_token is not None:
                logger.warning(
                    "GeminiMsrlClient: token broker refresh failed; keeping cached token"
                )
                return
            raise GeminiMsrlAuthError(
                "Failed to acquire an authentication token from the configured broker"
            ) from exc

        token = data.get("token") if isinstance(data, dict) else None
        if not isinstance(token, str) or not token:
            if self._cached_token is not None:
                logger.warning(
                    "GeminiMsrlClient: token broker returned no token; keeping cached token"
                )
                return
            raise GeminiMsrlAuthError(
                "The configured token broker returned no authentication token"
            )

        self._cached_token = token
        self._fetched_at = time.time()
        logger.info("GeminiMsrlClient: refreshed token from the configured broker")

    async def get_token(self, *, force_refresh: bool = False) -> str:
        if not force_refresh and not self._is_stale():
            assert self._cached_token is not None
            return self._cached_token

        async with self._lock:
            if force_refresh or self._is_stale():
                await self._fetch()

            if self._cached_token is None:
                raise GeminiMsrlAuthError(
                    "The configured token broker returned no authentication token"
                )
            return self._cached_token

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
