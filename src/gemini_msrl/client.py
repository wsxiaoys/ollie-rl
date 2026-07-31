import asyncio
import json
import logging
import os
from typing import AbstractSet, Any, Self

import httpx

from .auth import (
    GeminiMsrlError,
    GoogleAuthTokenSource,
    TokenSource,
    _EnvFileTokenSource,
    _HttpTokenSource,
)
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


class GeminiMsrlHttpError(GeminiMsrlError):
    """Exception raised when an API request returns an HTTP error status code."""

    def __init__(self, status_code: int, message: str, details: Any | None = None):
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code
        self.message = message
        self.details = details


class GeminiMsrlClient:
    """
    Asynchronous API Client for the Gemini Multi-Step Reinforcement Learning (MSRL) Tuning Service.
    """

    def __init__(
        self,
        base_url: str | None = None,
        client: httpx.AsyncClient | None = None,
        *,
        token_source: TokenSource | None = None,
    ):
        project_id = os.environ.get("GEMINI_MSRL_PROJECT_ID")
        if not project_id:
            raise ValueError("GEMINI_MSRL_PROJECT_ID environment variable must be set")

        self.project_id = project_id
        self.location = "us-central1"

        if base_url is None:
            base_url = os.environ.get(
                "GEMINI_MSRL_BASE_URL",
                "https://us-central1-aiplatform.googleapis.com",
            )
        self.base_url = base_url.rstrip("/")

        self._owned_token_source: _HttpTokenSource | None = None
        if token_source is not None:
            self._token_source = token_source
        elif token_url := os.environ.get("GEMINI_MSRL_TOKEN_URL"):
            broker_source = _HttpTokenSource(token_url)
            self._token_source = broker_source
            self._owned_token_source = broker_source
        elif env_file := os.environ.get("GEMINI_MSRL_ENV_FILE"):
            self._token_source = _EnvFileTokenSource(
                env_file,
                "GEMINI_MSRL_AUTH_TOKEN",
            )
        elif credentials_json := os.environ.get("GEMINI_MSRL_GOOGLE_CREDENTIALS_JSON"):
            try:
                credentials_info = json.loads(credentials_json)
            except json.JSONDecodeError:
                raise ValueError(
                    "GEMINI_MSRL_GOOGLE_CREDENTIALS_JSON must contain valid JSON"
                ) from None
            if not isinstance(credentials_info, dict):
                raise ValueError(
                    "GEMINI_MSRL_GOOGLE_CREDENTIALS_JSON must contain a JSON object"
                )
            self._token_source = GoogleAuthTokenSource.from_service_account_info(
                credentials_info
            )
        else:
            self._token_source = GoogleAuthTokenSource.default()

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

    async def get_client(self) -> httpx.AsyncClient:
        """Get or initialize the underlying httpx.AsyncClient.

        Note: Authorization is *not* set as a default header on the AsyncClient
        because the token may be refreshed between requests (see _request).
        The larger active-connection pool prevents HTTPX's default 100-connection
        limit from serializing high-concurrency rollout sampling, while the
        smaller keep-alive cap avoids retaining every burst connection.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers=self.headers,
                limits=httpx.Limits(
                    max_connections=512,
                    max_keepalive_connections=100,
                ),
                timeout=httpx.Timeout(600, connect=30.0, pool=60.0),
            )
        return self._client

    async def close(self) -> None:
        """Close HTTP resources created internally by this client."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None
        if self._owned_token_source is not None:
            await self._owned_token_source.close()

    async def __aenter__(self) -> Self:
        await self.get_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def _request(
        self,
        method: str,
        path: str,
        json_data: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        *,
        base_url: str | None = None,
        timeout: float | httpx.Timeout | None = None,
        quiet_error_statuses: AbstractSet[int] = frozenset(),
    ) -> dict[str, Any]:
        """Helper to perform requests and handle errors."""
        client = await self.get_client()
        request_base_url = (base_url or self.base_url).rstrip("/")
        url = f"{request_base_url}/{path.lstrip('/')}"
        request_timeout = timeout if timeout is not None else client.timeout

        response: httpx.Response | None = None
        for attempt in range(2):
            token = await self._token_source.get_token(force_refresh=attempt == 1)
            request_headers = {"Authorization": f"Bearer {token}"}

            try:
                response = await client.request(
                    method,
                    url,
                    json=json_data,
                    params=params,
                    headers=request_headers,
                    timeout=request_timeout,
                )
            except httpx.RequestError as exc:
                logger.error("An error occurred while requesting %r.", exc.request.url)
                raise GeminiMsrlError(f"Request failed: {exc}") from exc

            if response.status_code != 401 or attempt == 1:
                break

        assert response is not None
        if response.is_error:
            # Try to extract detailed error information from a structured body.
            error_msg = response.text
            details = None
            try:
                err_json = response.json()
            except ValueError:
                err_json = None
            if isinstance(err_json, dict) and isinstance(err_json.get("error"), dict):
                error = err_json["error"]
                error_msg = error.get("message", error_msg)
                details = error.get("details")

            if response.status_code not in quiet_error_statuses:
                logger.error("API Error %s: %s", response.status_code, error_msg)
            raise GeminiMsrlHttpError(response.status_code, error_msg, details)

        return response.json()

    # --- API Methods ---

    async def create_tuning_job(
        self, request: CreateTuningJobRequest | dict[str, Any]
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

    async def get_tuning_job(
        self,
        name: str,
        *,
        quiet_error_statuses: AbstractSet[int] = frozenset(),
    ) -> TuningJob:
        """
        Retrieves details of a tuning job.

        Path: GET /v1beta1/{name=projects/*/locations/*/tuningJobs/*}
        """
        # Ensure name is a path relative to v1beta1
        path = name if name.startswith("v1beta1/") else f"v1beta1/{name}"
        response_data = await self._request(
            "GET", path, quiet_error_statuses=quiet_error_statuses
        )
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
        request: GenerateContentTuningScopeRequest | dict[str, Any],
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
        request: TrainStepRequest | dict[str, Any],
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
        request: ContentGenerationParameters | dict[str, Any],
    ) -> EndpointGenerateContentResponse:
        """
        Performs standard content generation against a deployed endpoint.

        Checkpoint endpoints are served by the location-specific replica API,
        rather than the control-plane API used for tuning jobs and operations.

        Path: POST /v1beta1/{endpoint=projects/*/locations/*/endpoints/*}:generateContent
        """
        path = endpoint_name.lstrip("/")
        if not path.startswith("v1beta1/"):
            path = f"v1beta1/{path}"
        if not path.endswith(":generateContent"):
            path += ":generateContent"

        endpoint_parts = path.split("/")
        try:
            location = endpoint_parts[endpoint_parts.index("locations") + 1]
        except ValueError, IndexError:
            raise ValueError(
                "endpoint_name must contain a projects/*/locations/*/endpoints/* "
                "resource name"
            ) from None
        if not location:
            raise ValueError("endpoint_name must contain a non-empty location")
        endpoint_base_url = f"https://aiplatform.{location}.rep.googleapis.com"

        if isinstance(request, ContentGenerationParameters):
            # Standard endpoint generation accepts the generate-content body
            # directly, not the MSRL tuning-scope wrapper.
            json_data = request.model_dump(
                mode="json", by_alias=True, exclude_none=True
            )
        else:
            json_data = request

        response_data = await self._request(
            "POST",
            path,
            json_data=json_data,
            base_url=endpoint_base_url,
            timeout=60.0,
        )
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
