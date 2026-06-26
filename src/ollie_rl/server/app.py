import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import RedirectResponse

from ollie_rl.types import ChatCompletionRequest, CreateTunerRequest, CreateRewardRequest
from openai.types.chat import ChatCompletion
from ollie_rl.db import init_db, shutdown_db
from ollie_rl.service import TunerService

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Services:
    tuner = TunerService()


services = Services()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize database
    try:
        logger.info("Initializing database tables...")
        await init_db()
    except Exception as e:
        logger.exception("Failed to initialize database during startup")

    yield

    try:
        logger.info("Shutting down database tables...")
        await shutdown_db()
    except Exception as e:
        logger.exception("Failed to shutdown database during shutdown")


app = FastAPI(
    title="Ollie RL Server",
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(404)
async def custom_404_handler(request: Request, exc: Exception) -> RedirectResponse:
    """Redirect 404 Not Found errors to /docs."""
    return RedirectResponse("/docs")


@app.post("/tuners")
async def create_tuner(request: CreateTunerRequest):
    """
    Creates a new LoRA training client / model dynamically from a recipe template.
    """
    try:
        # Create, initialize, and register the tuner dynamically
        tuner_id = await services.tuner.create_tuner(request.recipe, request.name)

        logger.info(
            f"Dynamically created and initialized tuner: {tuner_id} (name: {request.name}) using recipe template: {request.recipe}"
        )
        return {
            "tuner_id": tuner_id,
            "name": request.name,
            "recipe": request.recipe,
        }
    except Exception as e:
        logger.exception(f"Failed to create tuner for name: {request.name}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tuners/{tuner_id}/rewards")
async def create_reward(
    tuner_id: str,
    request: CreateRewardRequest,
):
    """
    Sets the reward for a specific run under a tuner.
    """
    try:
        await services.tuner.create_reward(
            tuner_id=tuner_id,
            datum_id=request.datum_id,
            run_id=request.run_id,
            reward=request.reward,
        )
        return {"status": "success"}
    except Exception as e:
        logger.exception(f"Failed to set reward for run '{request.run_id}'")
        raise HTTPException(status_code=500, detail=str(e))


async def _generate_chat_completion(
    tuner_id: str,
    request: ChatCompletionRequest,
) -> ChatCompletion:
    tuner = await services.tuner.get(tuner_id)
    if not tuner:
        raise HTTPException(
            status_code=404,
            detail=f"Tuner '{tuner_id}' not found or not initialized.",
        )

    try:
        sample_op = await tuner.sample(request)
        sample = await sample_op.wait()
    except Exception as e:
        logger.exception(
            f"Failed to generate chat completion for model '{request.model}'"
        )
        raise HTTPException(status_code=500, detail=str(e))

    return sample.completion


@app.post("/openai/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    x_tuner_id: Annotated[str, Header()],
    x_datum_id: Annotated[str | None, Header()] = None,
    x_run_id: Annotated[str | None, Header()] = None,
) -> ChatCompletion:
    """Generate a chat completion from the active policy of the requested model."""
    completion = await _generate_chat_completion(x_tuner_id, request)

    # Record completion metadata via TunerService
    if x_run_id is not None and x_datum_id is not None:
        await services.tuner.record_chat_completion(
            completion_id=completion.id,
            tuner_id=x_tuner_id,
            run_id=x_run_id,
            datum_id=x_datum_id,
        )

    return completion
