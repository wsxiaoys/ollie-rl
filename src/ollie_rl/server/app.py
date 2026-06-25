import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import RedirectResponse

from ollie_rl.cookbook import Cookbook
from ollie_rl.types import ChatCompletionRequest, CreateTunerRequest
from openai.types.chat import ChatCompletion
from .tuner_storage import TunerStorage
from .completion_storage import CompletionStorage

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize tuner storage (handles both in-memory active tuners and persistence)
storage = TunerStorage()
completion_storage = CompletionStorage()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load and restore tuners from storage
    try:
        logger.info("Restoring persisted tuners...")
        await storage.restore_tuners()
        logger.info(f"Successfully restored active tuners: {storage.list_keys()}")
    except Exception as e:
        logger.exception("Failed to restore persisted tuners during startup")

    yield

    # Shutdown: persist all active tuners to storage and close completion storage
    try:
        logger.info("Persisting active tuners on shutdown...")
        await storage.save_all_tuners()
    except Exception as e:
        logger.exception("Failed to persist tuners during shutdown")

    try:
        logger.info("Closing tuner storage...")
        await storage.close()
    except Exception as e:
        logger.exception("Failed to close tuner storage during shutdown")

    try:
        logger.info("Closing completion storage...")
        await completion_storage.close()
    except Exception as e:
        logger.exception("Failed to close completion storage during shutdown")


app = FastAPI(
    title="Ollie RL Server",
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(404)
async def custom_404_handler(request: Request, exc: Exception) -> RedirectResponse:
    """Redirect 404 Not Found errors to /docs."""
    return RedirectResponse("/docs")


@app.post("/v1/tuners")
async def create_tuner(request: CreateTunerRequest):
    """
    Creates a new LoRA training client / model dynamically from a recipe template.
    """
    try:
        # Create and initialize the tuner using the Cookbook
        tuner = await Cookbook.create(request.recipe, tuner_id=request.tuner_id)

        # Register the tuner dynamically (keeps in memory and persists to disk)
        await storage.register_tuner(request.tuner_id, tuner)

        logger.info(
            f"Dynamically created and initialized tuner: {request.tuner_id} using recipe template: {request.recipe}"
        )
        return {
            "status": "success",
            "tuner_id": request.tuner_id,
            "recipe": request.recipe,
        }
    except Exception as e:
        logger.exception(f"Failed to create tuner: {request.tuner_id}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    chat_id: Optional[str] = Header(None, alias="x-chat-id"),
) -> ChatCompletion:
    """Generate a chat completion from the active policy of the requested model."""
    tuner = storage.get(request.model)
    if not tuner:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model}' not found or not initialized. Available models: {storage.list_keys()}",
        )

    try:
        if chat_id:
            logger.info(
                f"Generating chat completion for model '{request.model}' with chat ID: {chat_id}"
            )
        completion = await tuner.sample(request)
        if completion and getattr(completion, "id", None):
            await completion_storage.record_completion(
                completion_id=completion.id,
                chat_id=chat_id,
                tuner_id=request.model,
            )
        return completion
    except Exception as e:
        logger.exception(
            f"Failed to generate chat completion for model '{request.model}'"
        )
        raise HTTPException(status_code=500, detail=str(e))
