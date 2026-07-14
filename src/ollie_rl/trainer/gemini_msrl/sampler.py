from __future__ import annotations

import asyncio
import logging
from typing import Optional

from gemini_msrl import GeminiMsrlClient

from ollie_rl.trainer.types import Sampler
from ollie_rl.types import ChatCompletionRequest

from .conversion import build_content_generation_parameters, sample_from_candidates
from .ops import GeminiMsrlEndpointSampleOp

logger = logging.getLogger(__name__)


class GeminiMsrlSampler(Sampler):
    def __init__(
        self,
        client: GeminiMsrlClient,
        endpoint_name: str,
        policy_generation: int,
    ):
        self.client = client
        self.endpoint_name = endpoint_name
        self.policy_generation = policy_generation

    async def sample(
        self,
        request: ChatCompletionRequest,
        *,
        restore_state: Optional[str] = None,
    ) -> GeminiMsrlEndpointSampleOp:
        if restore_state is not None:
            logger.debug(
                "Ignoring restore_state for non-resumable Gemini endpoint sample "
                "against %s",
                self.endpoint_name,
            )

        async def _run():
            params = build_content_generation_parameters(request)
            response = await self.client.generate_content_endpoint(
                self.endpoint_name,
                params,
            )
            if not response.candidates:
                raise RuntimeError(
                    "Failed to retrieve generated candidates from endpoint response"
                )
            return sample_from_candidates(
                candidates=response.candidates,
                usage_metadata=response.usage_metadata,
                model_name=request.model,
                policy_generation=self.policy_generation,
                id_prefix=f"checkpoint_{self.policy_generation}",
            )

        return GeminiMsrlEndpointSampleOp(asyncio.create_task(_run()))
