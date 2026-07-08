import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated, Literal, Optional

from fastapi import FastAPI, Header, HTTPException, Query
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
    ListDatumsResponse,
    ListRunsResponse,
    RewardDistributionResponse,
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


def _train_loop_disabled() -> bool:
    """Return True when the background train loop is disabled via env var.

    Set ``OLLIE_DISABLE_TRAIN_LOOP`` to a truthy value (``1``, ``true``,
    ``yes``, or ``on``) to prevent the server from starting the background
    train loop.
    """
    value = os.environ.get("OLLIE_DISABLE_TRAIN_LOOP", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize database
    try:
        logger.info("Initializing database tables...")
        await init_db()
    except Exception:
        logger.exception("Failed to initialize database during startup")

    # Start the background train loop that periodically triggers train steps,
    # unless it has been disabled via the OLLIE_DISABLE_TRAIN_LOOP env var.
    train_loop_enabled = not _train_loop_disabled()
    if train_loop_enabled:
        services.tuner.start_train_loop()
    else:
        logger.info(
            "Train loop disabled via OLLIE_DISABLE_TRAIN_LOOP; "
            "skipping background train loop startup"
        )

    yield

    if train_loop_enabled:
        try:
            logger.info("Stopping train loop...")
            await services.tuner.stop_train_loop()
        except Exception:
            logger.exception("Failed to stop train loop during shutdown")

    try:
        logger.info("Shutting down database tables...")
        await shutdown_db()
    except Exception:
        logger.exception("Failed to shutdown database during shutdown")


app = FastAPI(
    title="Ollie RL Server",
    version="0.1.0",
    description="📊 [Web dashboard](/app)",
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
        # Datum-set validation (train non-empty, no train/eval overlap) lives
        # in `create_tuner`; surface its ValueError as a 400.
        tuner_id = await services.tuner.create_tuner(
            recipe=request.recipe,
            name=request.name,
            train_datum_ids=request.train_datum_ids,
            eval_datum_ids=request.eval_datum_ids,
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
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
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
async def get_tuner(
    tuner_id: str,
    progress: str = Query(
        default="",
        description=(
            "Comma-separated progress snapshots to attach: any of 'train' (a "
            "recipe-aware training-progress snapshot -- batch readiness, "
            "run/group coverage, next pick) and 'eval' (a per-eval-datum "
            "held-out status rollup). E.g. 'train', 'eval', or 'train,eval'. "
            "Empty (default) attaches neither."
        ),
    ),
) -> GetTunerResponse:
    """
    Returns information for a specific tuner, including policy_generation and
    stored trainer state. Pass `?progress=train` and/or `?progress=eval`
    (comma-separated) to also include the corresponding progress snapshot(s)
    under the response's `progress` object.
    """
    from ollie_rl.service.tuner import TunerNotFoundError

    kinds = [p.strip() for p in progress.split(",") if p.strip()]
    invalid = sorted(set(kinds) - {"train", "eval"})
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid progress kind(s): {invalid}. Allowed: 'train', 'eval'.",
        )

    try:
        return await services.tuner.get_tuner_details(tuner_id, progress=kinds)
    except TunerNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to fetch details for tuner '{tuner_id}'")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tuners/{tuner_id}/data", response_model=ListDatumsResponse)
async def list_data(
    tuner_id: str,
    split: Optional[Literal["train", "eval"]] = Query(
        default=None,
        description=(
            "Only return datums from this split: 'train' or 'eval'. Omit to "
            "return the full pool."
        ),
    ),
) -> ListDatumsResponse:
    """
    Return the datum-id pool registered for a tuner, so clients can build a
    filter dropdown for the runs list. Pass `split=train`/`eval` to scope the
    pool to a single split.
    """
    from ollie_rl.service.tuner import TunerNotFoundError

    try:
        return await services.tuner.list_datums(tuner_id, split=split)
    except TunerNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to list data for tuner '{tuner_id}'")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/tuners/{tuner_id}/reward-distribution",
    response_model=RewardDistributionResponse,
)
async def reward_distribution(
    tuner_id: str,
    datum_id: Optional[str] = Query(
        default=None,
        description="Only aggregate runs dispensed for this datum id.",
    ),
    kind: Literal["train", "eval"] = Query(
        default="train",
        description=(
            "Which runs to bucket: 'train' (rewarded training runs by "
            "completion generation) or 'eval' (held-out eval runs by the "
            "generation of the checkpoint they targeted)."
        ),
    ),
) -> RewardDistributionResponse:
    """
    Reward distribution bucketed by policy generation for a tuner.

    Reads only `(reward, generation)` per rewarded run and returns the finished
    per-generation histogram, so the dashboard doesn't fetch every run to
    aggregate client-side. Pass `datum_id` to scope to a single datum, and
    `kind=eval` to bucket the held-out eval split by checkpoint generation.
    """
    from ollie_rl.service.tuner import TunerNotFoundError

    try:
        return await services.tuner.reward_distribution(
            tuner_id, datum_id=datum_id, kind=kind
        )
    except TunerNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(
            f"Failed to compute reward distribution for tuner '{tuner_id}'"
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tuners/{tuner_id}/runs", response_model=ListRunsResponse)
async def list_runs(
    tuner_id: str,
    limit: Optional[int] = Query(
        default=None,
        ge=1,
        le=500,
        description="Max runs to return per page. Omit to return every run.",
    ),
    cursor: Optional[str] = Query(
        default=None,
        description=(
            "Opaque forward cursor from a previous response's `next_cursor`. "
            "Omit to fetch the first page."
        ),
    ),
    datum_id: Optional[str] = Query(
        default=None,
        description="Only return runs dispensed for this datum id.",
    ),
    kind: Optional[Literal["train", "eval"]] = Query(
        default=None,
        description=(
            "Only return runs of this kind: 'train' (runs on the training "
            "split) or 'eval' (held-out eval runs, i.e. those targeting a "
            "checkpoint). Omit to return both."
        ),
    ),
) -> ListRunsResponse:
    """
    List runs for a tuner (newest first) with their derived lifecycle status
    and recorded chat-completion counts.

    Supports cursor-based pagination via `limit`/`cursor`; the response returns
    a `next_cursor` when more runs are available. Pass `datum_id` to filter the
    listing to runs for a single datum, and `kind` to restrict to training vs
    held-out eval runs.
    """
    from ollie_rl.service.tuner import (
        InvalidRunCursorError,
        TunerNotFoundError,
    )

    try:
        return await services.tuner.list_runs(
            tuner_id, limit=limit, cursor=cursor, datum_id=datum_id, kind=kind
        )
    except TunerNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except InvalidRunCursorError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to list runs for tuner '{tuner_id}'")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tuners/{tuner_id}/runs/{run_id}", response_model=RunDetailResponse)
async def get_run(tuner_id: str, run_id: str) -> RunDetailResponse:
    """
    Return a single run and its chat completions (oldest first) so the full
    request/response transcript can be visualized.
    """
    from ollie_rl.service.tuner import RunNotFoundError

    try:
        return await services.tuner.get_run_details(tuner_id, run_id)
    except RunNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to fetch run '{run_id}' for tuner '{tuner_id}'")
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
    from ollie_rl.service.tuner import ChatCompletionNotFoundError

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


async def _generate_chat_completion(
    *,
    tuner_id: str,
    request: ChatCompletionRequest,
    run_id: str | None,
):
    """Shared handler for the header- and path-addressed completion endpoints.

    Real token-by-token streaming is not supported. When ``stream=true`` is
    requested, the full completion is generated first and then replayed as a
    simulated SSE stream so that OpenAI-compatible clients keep working.
    """
    from ollie_rl.service.tuner import (
        TunerNotFoundError,
        RunNotFoundError,
        RunExpiredError,
        RewardAlreadySetError,
        ContentFilterSampleError,
        LengthSampleError,
    )

    try:
        completion = await services.tuner.sample(
            tuner_id=tuner_id,
            request=request,
            run_id=run_id,
        )
    except (TunerNotFoundError, RunNotFoundError) as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )
    except (RunExpiredError, RewardAlreadySetError) as e:
        raise HTTPException(
            status_code=403,
            detail=str(e),
        )
    except ContentFilterSampleError as e:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "content_filter_sample",
                "message": str(e),
                "raw_content": e.raw_content,
            },
        )
    except LengthSampleError as e:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "length_sample",
                "message": str(e),
                "raw_content": e.raw_content,
            },
        )
    except Exception as e:
        # Include tuner/run ids and the exception string so failures (e.g. a
        # Gemini sampling operation timing out, whose message carries the
        # operation name) are correlatable to a run in the deploy logs. These
        # attempts never reach the DB, so this log is the only server-side
        # record of them.
        logger.exception(
            "Failed to generate chat completion (model=%s tuner=%s run=%s): %s",
            request.model,
            tuner_id,
            run_id,
            e,
        )
        raise HTTPException(status_code=500, detail=str(e))

    if request.stream:
        return StreamingResponse(
            simulate_stream(completion),
            media_type="text/event-stream",
        )
    return completion


@app.post("/openai/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    x_tuner_id: Annotated[str, Header()],
    x_run_id: Annotated[str | None, Header()] = None,
):
    """Generate a chat completion from the active policy of the requested model.

    The tuner/run this completion is attributed to travel in the ``X-Tuner-Id``
    / ``X-Run-Id`` headers. See the path-addressed twin
    (``/tuners/{tuner_id}/runs/{run_id}/openai/v1/chat/completions``) for a
    variant that carries the ids in the URL instead.
    """
    return await _generate_chat_completion(
        tuner_id=x_tuner_id,
        request=request,
        run_id=x_run_id,
    )


@app.post("/tuners/{tuner_id}/runs/{run_id}/openai/v1/chat/completions")
async def create_run_chat_completion(
    tuner_id: str,
    run_id: str,
    request: ChatCompletionRequest,
):
    """Path-addressed twin of ``/openai/v1/chat/completions``.

    Behaves identically to the header-based endpoint but carries the tuner and
    run ids in the URL instead of the ``X-Tuner-Id`` / ``X-Run-Id`` headers.
    Encoding the ids in the path keeps them in the request line, which makes
    per-run completions easy to search in log aggregators (e.g. Railway).
    """
    return await _generate_chat_completion(
        tuner_id=tuner_id,
        request=request,
        run_id=run_id,
    )


@app.post("/tuners/{tuner_id}/runs")
async def dispense_run(
    tuner_id: str,
) -> DispenseRun:
    """
    Dispense a run assignment for the tuner.
    Returns 200 OK with run_id, datum_id, expires_at.
    Or 204 No Content with Retry-After header if no run can be dispensed.

    Datum quarantine (length-rate / success-rate filtering plus the two-phase
    probe gate) is configured on the tuner's recipe, not per request.
    """
    from ollie_rl.service.tuner import TunerNotFoundError

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
    from ollie_rl.service.tuner import (
        RunNotFoundError,
        RunExpiredError,
        RewardAlreadySetError,
        EmptyRunError,
    )

    try:
        await services.tuner.update_reward(
            tuner_id=tuner_id,
            run_id=run_id,
            reward=request.reward,
        )
        # Training is driven by the background train loop (see
        # `TunerService.start_train_loop`), not triggered per reward.

        return PutRewardResponse(run_id=run_id, reward=request.reward)
    except RunNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (RunExpiredError, RewardAlreadySetError, EmptyRunError) as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.exception(f"Failed to record reward for run '{run_id}'")
        raise HTTPException(status_code=500, detail=str(e))


# Serve the built web dashboard at /app (no-op until `web/dist` is built).
# Mounted last so it never shadows the API routes above.
mount_webui(app)
