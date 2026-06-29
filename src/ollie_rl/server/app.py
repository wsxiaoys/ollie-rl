import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import RedirectResponse, StreamingResponse

from ollie_rl.types import (
    ChatCompletionRequest,
    CreateTunerRequest,
    CreateTunerResponse,
    DispenseRun,
    PutRewardRequest,
    PutRewardResponse,
    GetTunerResponse,
    ListTunersResponse,
    ListRunsResponse,
    RunDetailResponse,
    ChatCompletionDetailResponse,
)
from ollie_rl.db import init_db, shutdown_db
from ollie_rl.service import TunerService
from ollie_rl.server.streaming import simulate_stream
from ollie_rl.server.webui import mount_webui

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


@app.get("/", include_in_schema=False)
async def redirect_to_docs() -> RedirectResponse:
    """Redirect root access to /docs."""
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
            trainer_params=request.trainer_params,
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


@app.get("/tuners")
async def list_tuners() -> ListTunersResponse:
    """
    Returns a list of all tuners, including id, name, recipe, trainer, and policy_generation.
    """
    try:
        tuners = await services.tuner.list_tuners()
        return ListTunersResponse(tuners=tuners)
    except Exception as e:
        logger.exception("Failed to list tuners")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tuners/{tuner_id}")
async def get_tuner(tuner_id: str, progress: bool = False) -> GetTunerResponse:
    """
    Returns information for a specific tuner, including policy_generation and
    stored trainer state. Pass `?progress=true` to also include a recipe-aware
    training-progress snapshot (batch readiness, run/group coverage, next pick).
    """
    from ollie_rl.service.tuner_service import TunerNotFoundError

    try:
        return await services.tuner.get_tuner_details(
            tuner_id, include_progress=progress
        )
    except TunerNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to fetch details for tuner '{tuner_id}'")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tuners/{tuner_id}/runs", response_model=ListRunsResponse)
async def list_runs(tuner_id: str) -> ListRunsResponse:
    """
    List all runs for a tuner (newest first) with their derived lifecycle
    status and recorded chat-completion counts.
    """
    from ollie_rl.service.tuner_service import TunerNotFoundError

    try:
        runs = await services.tuner.list_runs(tuner_id)
        return ListRunsResponse(runs=runs)
    except TunerNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to list runs for tuner '{tuner_id}'")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tuners/{tuner_id}/runs/{run_id}", response_model=RunDetailResponse)
async def get_run(tuner_id: str, run_id: str) -> RunDetailResponse:
    """
    Return a single run and its chat completions (oldest first) so the full
    request/response transcript can be visualized.
    """
    from ollie_rl.service.tuner_service import RunNotFoundError

    try:
        return await services.tuner.get_run_details(tuner_id, run_id)
    except RunNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(
            f"Failed to fetch run '{run_id}' for tuner '{tuner_id}'"
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/tuners/{tuner_id}/runs/{run_id}/completions/{completion_id}",
    response_model=ChatCompletionDetailResponse,
)
async def get_completion(
    tuner_id: str, run_id: str, completion_id: str
) -> ChatCompletionDetailResponse:
    """
    Return a single recorded chat completion (request, response, and any
    sample-time tensors) so it can be inspected in isolation.
    """
    from ollie_rl.service.tuner_service import ChatCompletionNotFoundError

    try:
        return await services.tuner.get_completion_details(
            tuner_id, run_id, completion_id
        )
    except ChatCompletionNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(
            f"Failed to fetch completion '{completion_id}' for run "
            f"'{run_id}' of tuner '{tuner_id}'"
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/openai/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    x_tuner_id: Annotated[str, Header()],
    x_run_id: Annotated[str | None, Header()] = None,
):
    """Generate a chat completion from the active policy of the requested model.

    Real token-by-token streaming is not supported. When ``stream=true`` is
    requested, the full completion is generated first and then replayed as a
    simulated SSE stream so that OpenAI-compatible clients keep working.
    """
    from ollie_rl.service.tuner_service import (
        TunerNotFoundError,
        RunNotFoundError,
        RunExpiredError,
        RewardAlreadySetError,
        MalformedSampleError,
    )

    try:
        completion = await services.tuner.sample(
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
    except MalformedSampleError as e:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "malformed_sample",
                "message": str(e),
                "raw_content": e.raw_content,
            },
        )
    except Exception as e:
        logger.exception(
            f"Failed to generate chat completion for model '{request.model}'"
        )
        raise HTTPException(status_code=500, detail=str(e))

    if request.stream:
        return StreamingResponse(
            simulate_stream(completion),
            media_type="text/event-stream",
        )
    return completion


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


# Serve the built web dashboard at /app (no-op until `web/dist` is built).
# Mounted last so it never shadows the API routes above.
mount_webui(app)
