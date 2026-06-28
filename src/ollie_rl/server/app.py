import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import RedirectResponse

from ollie_rl.types import (
    ChatCompletionRequest,
    CreateTunerRequest,
    CreateTunerResponse,
    DispenseRun,
    PutRewardRequest,
    PutRewardResponse,
)
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
    except Exception:
        logger.exception("Failed to initialize database during startup")

    yield

    try:
        logger.info("Shutting down database tables...")
        await shutdown_db()
    except Exception:
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
async def create_tuner(request: CreateTunerRequest) -> CreateTunerResponse:
    """
    Creates a new LoRA training client / model dynamically from a recipe template.
    """
    try:
        if not request.datum_ids:
            raise HTTPException(status_code=400, detail="datum_ids must be non-empty")

        # Create, initialize, and register the tuner dynamically
        tuner_id = await services.tuner.create_tuner(
            recipe=request.recipe,
            name=request.name,
            datum_ids=request.datum_ids,
            trainer=request.trainer,
            bootstrap=request.bootstrap,
        )

        logger.info(
            f"Dynamically created and initialized tuner: {tuner_id} (name: {request.name}) using recipe template: {request.recipe}"
        )
        return CreateTunerResponse(
            tuner_id=tuner_id,
            name=request.name,
            recipe=request.recipe,
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception(f"Failed to create tuner for name: {request.name}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/openai/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    x_tuner_id: Annotated[str, Header()],
    x_run_id: Annotated[str | None, Header()] = None,
) -> ChatCompletion:
    """Generate a chat completion from the active policy of the requested model."""
    from ollie_rl.service.tuner_service import (
        TunerNotFoundError,
        RunNotFoundError,
        RunExpiredError,
        RewardAlreadySetError,
    )

    try:
        return await services.tuner.sample(
            tuner_id=x_tuner_id,
            request=request,
            run_id=x_run_id,
        )
    except (TunerNotFoundError, RunNotFoundError) as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )
    except (RunExpiredError, RewardAlreadySetError) as e:
        raise HTTPException(
            status_code=409,
            detail=str(e),
        )
    except Exception as e:
        logger.exception(
            f"Failed to generate chat completion for model '{request.model}'"
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tuners/{tuner_id}/runs")
async def dispense_run(tuner_id: str) -> DispenseRun:
    """
    Dispense a run assignment for the tuner.
    Returns 200 OK with run_id, datum_id, expires_at.
    Or 204 No Content with Retry-After header if no run can be dispensed.
    """
    from ollie_rl.service.tuner_service import TunerNotFoundError

    try:
        run_response = await services.tuner.dispense_run(tuner_id)
        if run_response is None:
            raise HTTPException(204, headers={"Retry-After": "1"})

        return run_response
    except TunerNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to dispense run for tuner '{tuner_id}'")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/tuners/{tuner_id}/runs/{run_id}/reward")
async def put_reward(
    tuner_id: str,
    run_id: str,
    request: PutRewardRequest,
) -> PutRewardResponse:
    """
    Sets the reward for a specific run under a tuner.
    """
    from ollie_rl.service.tuner_service import (
        RunNotFoundError,
        RunExpiredError,
        RewardAlreadySetError,
    )
    import asyncio

    try:
        await services.tuner.update_reward(
            tuner_id=tuner_id,
            run_id=run_id,
            reward=request.reward,
        )
        # Auto-train trigger (fire-and-forget)
        asyncio.create_task(services.tuner.maybe_train(tuner_id))

        return PutRewardResponse(run_id=run_id, reward=request.reward)
    except RunNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (RunExpiredError, RewardAlreadySetError) as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to record reward for run '{run_id}'")
        raise HTTPException(status_code=500, detail=str(e))
