import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Union

import httpx

from .types import (
    ContentGenerationParameters,
    CreateTuningJobRequest,
    EndpointGenerateContentResponse,
    GenerateContentTuningScopeRequest,
    Operation,
    TrainStepRequest,
    TuningJob,
)

logger = logging.getLogger(__name__)


class GeminiMsrlError(Exception):
    """Base exception for the Gemini MSRL API client."""

    pass


class GeminiMsrlHttpError(GeminiMsrlError):
    """Exception raised when an API request returns an HTTP error status code."""

    def __init__(self, status_code: int, message: str, details: Optional[Any] = None):
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code
        self.message = message
        self.details = details


def _parse_env_file(path: Path) -> Dict[str, str]:
    """Minimal KEY=VALUE parser. Ignores blank lines and `#` comments. Strips
    surrounding single/double quotes from values."""
    out: Dict[str, str] = {}
    try:
        with path.open("r") as fp:
            for raw in fp:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                v = v.strip()
                if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
                    v = v[1:-1]
                out[k.strip()] = v
    except FileNotFoundError:
        pass
    return out


class _TokenSource(Protocol):
    """Anything that can hand out the current bearer token on demand."""

    def get(self) -> Optional[str]: ...


class _HttpTokenSource:
    """Token source that fetches a bearer token from an HTTP endpoint and
    refreshes it on a fixed interval.

    The endpoint is expected to return JSON of the form::

        {"token": "<bearer-token>"}

    The token is refreshed lazily on ``get()`` (which is called per outgoing
    request) once ``refresh_interval`` seconds have elapsed since the last
    successful fetch, so there is no background thread to manage. If a refresh
    fails while a cached token is still available, the cached token is kept and
    the error is logged rather than raised.
    """

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
        self._cached_token: Optional[str] = None
        self._fetched_at: float = 0.0  # unix seconds; 0 => must fetch

    def _is_stale(self) -> bool:
        if self._cached_token is None:
            return True
        return time.time() >= (self._fetched_at + self._refresh_interval)

    def _fetch(self) -> None:
        try:
            resp = httpx.get(self._url, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            if self._cached_token is not None:
                logger.warning(
                    "GeminiMsrlClient: failed to refresh token from %s: %s; "
                    "keeping cached token",
                    self._url,
                    exc,
                )
                return
            raise ValueError(
                f"Failed to fetch auth token from {self._url}: {exc}"
            ) from exc

        token = data.get("token") if isinstance(data, dict) else None
        if not token:
            if self._cached_token is not None:
                logger.warning(
                    "GeminiMsrlClient: token endpoint %s returned no 'token'; "
                    "keeping cached token",
                    self._url,
                )
                return
            raise ValueError(
                f"Token endpoint {self._url} response missing 'token' field"
            )

        self._cached_token = token
        self._fetched_at = time.time()
        logger.info("GeminiMsrlClient: refreshed token from %s", self._url)

    def get(self) -> Optional[str]:
        if self._is_stale():
            self._fetch()
        return self._cached_token


class _EnvFileTokenSource:
    """Token source that re-reads a key from an env file when its mtime changes.

    Lets the server pick up refreshed tokens without restarting. Updating .env
    on disk is enough; the next outgoing request will use the new value.
    """

    def __init__(self, env_path: str, key: str, initial: Optional[str] = None):
        self._path = Path(env_path)
        self._key = key
        self._cached_token: Optional[str] = initial
        self._cached_mtime: Optional[float] = None

    def get(self) -> Optional[str]:
        try:
            mtime = self._path.stat().st_mtime
        except FileNotFoundError:
            return self._cached_token
        if mtime != self._cached_mtime:
            parsed = _parse_env_file(self._path)
            new_token = parsed.get(self._key)
            if new_token and new_token != self._cached_token:
                logger.info(
                    "GeminiMsrlClient: refreshed %s from %s", self._key, self._path
                )
                self._cached_token = new_token
            elif new_token is None:
                logger.warning(
                    "GeminiMsrlClient: %s not present in %s; keeping cached token",
                    self._key,
                    self._path,
                )
            self._cached_mtime = mtime
        return self._cached_token


class GeminiMsrlClient:
    """
    Asynchronous API Client for the Gemini Multi-Step Reinforcement Learning (MSRL) Tuning Service.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        client: Optional[httpx.AsyncClient] = None,
    ):
        project_id = os.environ.get("GEMINI_MSRL_PROJECT_ID")
        if not project_id:
            raise ValueError("GEMINI_MSRL_PROJECT_ID environment variable must be set")

        self.project_id = project_id
        self.location = "us-central1"

        if base_url is None:
            base_url = os.environ.get(
                "GEMINI_MSRL_BASE_URL",
                "https://us-central1-staging-aiplatform.sandbox.googleapis.com",
            )
        self.base_url = base_url.rstrip("/")

        # Token source. Two mechanisms are supported (URL takes precedence):
        #
        # 1. GEMINI_MSRL_TOKEN_URL: fetch the token directly from an HTTP
        #    endpoint and refresh lazily just before it expires. This is the
        #    standard mechanism.
        # 2. GEMINI_MSRL_ENV_FILE: re-read the token from an env file when its
        #    mtime changes, so an external refresher (e.g. `gcloud auth ...`)
        #    can propagate updates without a server restart.
        #
        # In both cases the token is pulled per-request, so refreshes take
        # effect without restarting the server.
        token_url = os.environ.get("GEMINI_MSRL_TOKEN_URL")
        self._token_source: _TokenSource
        if token_url:
            self._token_source = _HttpTokenSource(token_url)
        else:
            env_file = os.environ.get("GEMINI_MSRL_ENV_FILE")
            if not env_file:
                raise ValueError(
                    "GEMINI_MSRL_ENV_FILE environment variable must be set "
                    "(or set GEMINI_MSRL_TOKEN_URL to fetch tokens over HTTP)"
                )
            self._token_source = _EnvFileTokenSource(env_file, "GEMINI_MSRL_AUTH_TOKEN")

        # Static headers (Authorization is injected per-request from the
        # current token, so it's intentionally NOT pinned here).
        self.headers = {
            "Content-Type": "application/json",
            "x-goog-user-project": self.project_id,
        }

        # X-Forwarder-Mode header (fixture vs real)
        # Defaults to 'fixture' if not set.
        forwarder_mode = os.environ.get("GEMINI_MSRL_FORWARDER_MODE", None)
        if forwarder_mode:
            self.headers["X-Forwarder-Mode"] = forwarder_mode

        # Use provided AsyncClient or manage one internally
        self._client = client
        self._owns_client = client is None

    def _current_token(self) -> str:
        tok = self._token_source.get()
        if not tok:
            raise ValueError("GEMINI_MSRL_AUTH_TOKEN not found in environment file")
        return tok

    async def get_client(self) -> httpx.AsyncClient:
        """Get or initialize the underlying httpx.AsyncClient.

        Note: Authorization is *not* set as a default header on the AsyncClient
        because the token may be refreshed between requests (see _request).
        """
        if self._client is None:
            self._client = httpx.AsyncClient(headers=self.headers)
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client if it was created internally."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "GeminiMsrlClient":
        await self.get_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def _request(
        self,
        method: str,
        path: str,
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Helper to perform requests and handle errors."""
        client = await self.get_client()
        url = f"{self.base_url}/{path.lstrip('/')}"

        # Pull a fresh token per-request so external refreshes (e.g. updates
        # to the env file by a `gcloud auth ...` cron) propagate without a
        # server restart.
        request_headers = {"Authorization": f"Bearer {self._current_token()}"}

        try:
            response = await client.request(
                method, url, json=json_data, params=params, headers=request_headers
            )
        except httpx.RequestError as exc:
            logger.error(f"An error occurred while requesting {exc.request.url!r}.")
            raise GeminiMsrlError(f"Request failed: {exc}") from exc

        if response.is_error:
            # Try to extract detailed error message from response JSON
            error_msg = response.text
            details = None
            try:
                err_json = response.json()
                if "error" in err_json:
                    error_msg = err_json["error"].get("message", error_msg)
                    details = err_json["error"].get("details", None)
            except Exception:
                pass

            logger.error(f"API Error {response.status_code}: {error_msg}")
            raise GeminiMsrlHttpError(response.status_code, error_msg, details)

        return response.json()

    # --- API Methods ---

    async def create_tuning_job(
        self, request: Union[CreateTuningJobRequest, Dict[str, Any]]
    ) -> TuningJob:
        """
        Creates a tuning job.

        Path: POST /v1beta1/projects/{project_id}/locations/{location}/tuningJobs
        """
        path = (
            f"v1beta1/projects/{self.project_id}/locations/{self.location}/tuningJobs"
        )

        if isinstance(request, CreateTuningJobRequest):
            # Exclude none so we don't send nulls, using camelCase alias
            json_data = request.model_dump(by_alias=True, exclude_none=True)
        else:
            json_data = request

        response_data = await self._request("POST", path, json_data=json_data)
        return TuningJob.model_validate(response_data)

    async def get_tuning_job(self, name: str) -> TuningJob:
        """
        Retrieves details of a tuning job.

        Path: GET /v1beta1/{name=projects/*/locations/*/tuningJobs/*}
        """
        # Ensure name is a path relative to v1beta1
        path = name if name.startswith("v1beta1/") else f"v1beta1/{name}"
        response_data = await self._request("GET", path)
        return TuningJob.model_validate(response_data)

    async def cancel_tuning_job(self, name: str) -> None:
        """
        Cancels a tuning job.

        Path: POST /v1beta1/{name=projects/*/locations/*/tuningJobs/*}:cancel
        """
        # Ensure name is a path relative to v1beta1
        path = name if name.startswith("v1beta1/") else f"v1beta1/{name}"
        if not path.endswith(":cancel"):
            path += ":cancel"
        await self._request("POST", path)

    async def generate_content_tuning_scope(
        self,
        tuning_job_id: str,
        request: Union[GenerateContentTuningScopeRequest, Dict[str, Any]],
    ) -> Operation:
        """
        Performs content generation within the scope of a tuning job.
        This creates a Long-Running Operation (LRO).

        Path: POST /v1beta1/projects/{project_id}/locations/{location}/tuningJobs/{tuning_job_id}:generateContentTuningScope
        """
        path = f"v1beta1/projects/{self.project_id}/locations/{self.location}/tuningJobs/{tuning_job_id}:generateContentTuningScope"

        if isinstance(request, GenerateContentTuningScopeRequest):
            json_data = request.model_dump(
                mode="json", by_alias=True, exclude_none=True
            )
        else:
            json_data = request

        response_data = await self._request("POST", path, json_data=json_data)
        return Operation.model_validate(response_data)

    async def train_step(
        self,
        tuning_job_id: str,
        request: Union[TrainStepRequest, Dict[str, Any]],
    ) -> Operation:
        """
        Performs a single step of training within the scope of a tuning job.
        This creates a Long-Running Operation (LRO).

        Path: POST /v1beta1/projects/{project_id}/locations/{location}/tuningJobs/{tuning_job_id}:trainStep
        """
        path = f"v1beta1/projects/{self.project_id}/locations/{self.location}/tuningJobs/{tuning_job_id}:trainStep"

        if isinstance(request, TrainStepRequest):
            json_data = request.model_dump(by_alias=True, exclude_none=True)
        else:
            json_data = request

        response_data = await self._request("POST", path, json_data=json_data)
        return Operation.model_validate(response_data)

    async def generate_content_endpoint(
        self,
        endpoint_name: str,
        request: Union[ContentGenerationParameters, Dict[str, Any]],
    ) -> EndpointGenerateContentResponse:
        """
        Performs standard content generation against a deployed endpoint.

        Path: POST /v1beta1/{endpoint=projects/*/locations/*/endpoints/*}:generateContent
        """
        path = endpoint_name.lstrip("/")
        if not path.startswith("v1beta1/"):
            path = f"v1beta1/{path}"
        if not path.endswith(":generateContent"):
            path += ":generateContent"

        if isinstance(request, ContentGenerationParameters):
            # Standard endpoint generation accepts the generate-content body
            # directly, not the MSRL tuning-scope wrapper.
            json_data = request.model_dump(
                mode="json", by_alias=True, exclude_none=True
            )
        else:
            json_data = request

        response_data = await self._request("POST", path, json_data=json_data)
        return EndpointGenerateContentResponse.model_validate(response_data)

    # --- Operation & Polling Helpers ---

    async def get_operation(self, name: str) -> Operation:
        """
        Retrieves the status of a Long-Running Operation.

        Path: GET /v1beta1/{name=projects/*/locations/*/operations/*}
        """
        path = name if name.startswith("v1beta1/") else f"v1beta1/{name}"
        response_data = await self._request("GET", path)
        return Operation.model_validate(response_data)

    async def wait_for_operation(
        self,
        operation_name: str,
        timeout_seconds: float = 300.0,
        poll_interval: float = 2.0,
    ) -> Operation:
        """
        Polls a Long-Running Operation until it is done or timeout is reached.
        """
        start_time = asyncio.get_event_loop().time()
        while True:
            operation = await self.get_operation(operation_name)
            if operation.done:
                return operation

            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= timeout_seconds:
                raise TimeoutError(
                    f"Operation '{operation_name}' timed out after {timeout_seconds} seconds."
                )

            await asyncio.sleep(poll_interval)

    async def wait_for_tuning_job_running(
        self,
        tuning_job_name: str,
        timeout_seconds: float = 600.0,
        poll_interval: float = 5.0,
    ) -> TuningJob:
        """
        Polls a TuningJob until its state is 'JOB_STATE_RUNNING' or terminal.
        """
        start_time = asyncio.get_event_loop().time()
        while True:
            job = await self.get_tuning_job(tuning_job_name)
            state = job.state

            if state == "JOB_STATE_RUNNING":
                return job

            # Check for terminal failure states
            if state in (
                "JOB_STATE_FAILED",
                "JOB_STATE_CANCELLED",
                "JOB_STATE_PAUSED",
            ):
                raise GeminiMsrlError(
                    f"Tuning job entered terminal non-running state: {state}. Error: {job.error}"
                )

            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= timeout_seconds:
                raise TimeoutError(
                    f"Tuning job '{tuning_job_name}' did not transition to JOB_STATE_RUNNING after {timeout_seconds} seconds. Current state: {state}"
                )

            await asyncio.sleep(poll_interval)
