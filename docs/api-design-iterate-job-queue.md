# API Design — Server-Dispatched Runs

Working doc for the next round of changes to the Ollie RL api server.
Goal: make the public HTTP surface correct for **synchronous GRPO**
(sampling must pause while training is in flight) while keeping the
door open for future async / disaggregated-sampler recipes.

The core proposal: **invert dispatch.** Instead of the client minting
`datum_id` / `run_id` pairs and posting completions whenever, the server
dispenses run assignments via a queue endpoint and the client just
executes them. The synchronous barrier falls out of "the queue is empty
right now."

Status: **draft, awaiting sign-off before implementation**.

---

## 1. Problem

Today the loop is implicitly async:

- Clients pick `datum_id`s and mint `run_id`s out-of-band, then post
  chat completions and rewards whenever.
- `TunerService.train` opportunistically grabs `TARGET_GROUP_COUNT`
  ready rollouts and trains.
- Sampling and training run against the same `Tuner` object with **no
  coordination** — a chat completion can be served from a half-updated
  policy mid-step.
- The client has to know how many runs make one training step
  (`GROUP_SIZE`, `TARGET_GROUP_COUNT`); those constants live in
  `tuner_service.py` and are not on the wire.
- Multi-worker sampling has no coordination: two samplers picking
  `datum_id`s independently will collide or duplicate work.

A server-driven run queue collapses all of these into one mechanism.

---

## 2. Mental model

| Question                                          | Owner               | Mechanism                                              |
|---------------------------------------------------|---------------------|--------------------------------------------------------|
| When is a training step happening?                | recipe / service    | recipe's `TrainOp`, tracked in-process by `TunerService` |
| Who decides what gets sampled next?               | recipe              | `Tuner.dispense_run(ctx)` returns the next assignment  |
| How is work handed to a sampler?                  | server              | `POST /tuners/{id}/runs` returns a `(run_id, datum_id)`|
| Which policy did a sample come from?              | trainer (recorded)  | `Sample.policy_generation` → `ChatCompletionModel`     |
| Is sampling allowed during training?              | recipe              | `dispense_run` returns `None` (or not) while busy      |

We do **not** introduce an `IDLE / TRAINING` state column on
`TunerModel`. The recipe's `TrainOp` is the source of truth for "are we
training right now"; `TunerService` holds the in-flight op handle
in-process and tells the recipe via `DispenseContext.is_training`. No
duplicated state.

---

## 3. Decisions

| # | Decision                                                                                                   | Rationale                                                                                                |
|---|------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| 1 | Server dispenses runs via `POST /tuners/{id}/runs`. Response is `{ run_id, datum_id }`.                   | Leasing is state-mutating (inserts a row, allocates `run_id`) — POST is the right verb.                  |
| 2 | **One run per request.** No batching.                                                                      | Simplest contract; multi-worker fan-out is many parallel POSTs.                                          |
| 3 | Datum pool is **registered at `POST /tuners` creation time** (`datum_ids` in the request body).            | A tuner is useless without a corpus; making it required eliminates a "did you forget to register?" failure mode. |
| 4 | Rename `Sample.step_id` → `Sample.policy_generation`. Type stays `str` (opaque, recipe-defined).           | The name actually describes the concept (model weight version). Type stays opaque because recipes vary.   |
| 5 | Persist `policy_generation` on **`ChatCompletionModel`**, not on `RunModel`.                               | A single run may produce multiple chat completions (multi-step / tool-using trajectories), each at a different generation. |
| 6 | The barrier is implicit: while a `TrainOp` is in flight, `dispense_run` returns `None` ⇒ HTTP `204`.       | One endpoint, one mental model. No `423`, no separate `state` polling.                                   |
| 7 | **No** `Tuner.allows_sampling_during_training` property.                                                    | The "what do we do during training" policy lives inside `dispense_run` itself; that's expressive enough. |
| 8 | **No** `Tuner.restore_train_op` hook.                                                                      | The recipe's existing `save_state` / `Recipe.restore` is the persistence contract.                       |
| 9 | **No separate `RewardModel`.** Reward lives as a column on `RunModel` (which is the canonical run record). | A reward without a run is meaningless; a run with no reward is just unfinished. One row per run.         |
| 10| `group_size` / `batch_size` are **recipe-internal** (or recipe hparams), never on the wire.               | They're scheduling, not API contract.                                                                    |

---

## 4. Surface changes

### 4.1 `TunerModel` (DB)

Unchanged. The existing `state` column already stores recipe-serialized
state (e.g. `{"tuning_job_name": "..."}` for `gemini_msrl`) and we don't
add anything else. "Is this tuner currently training?" is tracked
**in-process** by `TunerService` via a per-tuner asyncio handle, not by
a DB flag.

### 4.2 `ChatCompletionModel` (DB)

Uncomment and rename the long-commented-out field:

```python
policy_generation: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
```

Stamped from `Sample.policy_generation` at completion-record time. One
`ChatCompletionModel` row per LLM round-trip; many rows per `RunModel`
in the multi-step / tool-using case.

### 4.3 New table — `RunModel`

Replaces the existing `RewardModel`. One row per **run** (a unit of
work dispensed by the server, possibly completed and rewarded).

```python
class RunModel(BaseModel):
    __tablename__ = "runs"

    id:          Mapped[str]                 = mapped_column(String(255), primary_key=True)
    tuner_id:    Mapped[str]                 = mapped_column(String(255), ForeignKey("tuners.id"), index=True)
    datum_id:    Mapped[str]                 = mapped_column(String(255), nullable=False, index=True)
    reward:      Mapped[Optional[float]]     = mapped_column(Float, nullable=True)
    train_count: Mapped[int]                 = mapped_column(Integer, nullable=False, default=0)
    expires_at:  Mapped[datetime]            = mapped_column(DateTime, nullable=False, index=True)
    created_at:  Mapped[datetime]            = mapped_column(DateTime, nullable=False, default=func.now())
    updated_at:  Mapped[datetime]            = mapped_column(DateTime, nullable=False, default=func.now(), onupdate=func.now())
```

Two independent bookkeeping fields, each with its own role:

**`train_count: int`** — consumed-by-training counter, ported verbatim
from `RewardModel.train_count` in
`src/ollie_rl/db/models.py`. A run is "consumable for training" iff
`reward IS NOT NULL AND train_count == 0`. After a successful
`train_step`, `_drive_step` bumps `train_count` to `1` so the same
run is never trained on twice. Future off-policy work can change the
threshold (`train_count <= K`) without schema churn.

**`expires_at: datetime`** — lease deadline for redispense, **not**
related to training consumption. Stamped at dispense time as
`NOW() + run_ttl` (recipe hparam, default `5 min` in v1). Its only
job is to give the dispenser a chance to re-distribute datums whose
runs were dispensed but never rewarded (sampler crashed, network
blip, etc.):

- If a run has `reward IS NULL AND expires_at <= NOW()`, its lease is
  expired. The next `POST /runs` is free to pick the same `datum_id`
  again (with a fresh `run_id`); the recipe's `dispense_run` is the
  one that decides whether to do so.
- An expired run **does not** block training of *other* runs against
  the same datum. The training-readiness query
  (`reward IS NOT NULL AND train_count == 0`) ignores `expires_at`
  entirely.
- `PUT /reward` rejects a reward posted against an expired-and-
  unrewarded run (`409 Conflict`) because the dispenser may have
  already re-issued that datum — accepting the late reward would
  double-count.
- A background sweeper could later hard-delete
  expired-and-unrewarded rows for tidiness, but it's not required
  for v1.

Chat completions reference the run via the existing
`ChatCompletionModel.run_id`.

### 4.4 New table — `datum_rows`

```python
class DatumRowModel(BaseModel):
    __tablename__ = "datum_rows"

    tuner_id: Mapped[str] = mapped_column(String(255), ForeignKey("tuners.id"), primary_key=True)
    datum_id: Mapped[str] = mapped_column(String(255), primary_key=True)
```

Server treats `datum_id` as opaque. Populated at `POST /tuners` time
from the request body. No streaming endpoint in v1 — re-create the
tuner if the corpus needs to grow.

Naming note: the table is `datum_rows` (one row per registered datum
reference) rather than `datums`, to stay consistent with the existing
SQLAlchemy models in `src/ollie_rl/db/models.py` where each model
represents a row-level record (`TunerModel` / `tuners`,
`ChatCompletionModel` / `chat_completions`, `RewardModel` / `rewards`)
and the `DatumRowModel` name makes the row-per-(tuner, datum_id)
contract explicit.

### 4.5 `Sample` (in-process)

```diff
 class Sample(BaseModel):
     completion: ChatCompletion
-    step_id: str
+    policy_generation: str
```

Pure rename. `gemini_msrl.sample()` already builds this from
`response.train_step_id`; `test_gemini_msrl.py` follows. Type stays
`str` so that recipes with opaque generation identifiers (hashes,
LRO names, semver-ish strings) keep working.

### 4.6 `Tuner` (in-process)

One new hook:

```python
@dataclass
class DispenseContext:
    is_training: bool
    datum_pool: list[str]

@dataclass
class RunAssignment:
    run_id: str
    datum_id: str

class Tuner(...):
    def dispense_run(self, ctx: DispenseContext) -> Optional[RunAssignment]:
        """
        Recipe-owned dispatch. Default implementation:
          - if ctx.is_training: return None  (sync-safe)
          - if not ctx.datum_pool: return None
          - pick a datum_id (recipe's choice of policy), mint a run_id,
            and return the assignment.
        """
        ...
```

Sync GRPO uses the default. Async recipes override and ignore
`ctx.is_training`, possibly capping in-flight runs etc.

No `allows_sampling_during_training` property — the logic lives inside
`dispense_run` itself.

One additional optional hook used for both restart recovery (§4.11)
and the live `is_training` check (§4.8):

```python
class Tuner(...):
    async def in_flight_train_op(self) -> Optional[TrainOp]:
        """
        Return the TrainOp captured in the last save_state() (i.e. the
        train op that was running when state was checkpointed), so
        TunerService can poll / await it. Returns None when no train
        op was in flight at checkpoint time. Default impl: return None.
        """
        return None
```

`TrainOp` and `SampleOp` inherit from a generic base class `Op[T]` to share a non-blocking `peek()` interface:

```python
class Op(ABC, Generic[T]):
    @abstractmethod
    async def wait(self) -> T:
        """Block and wait for the operation to complete."""
        pass

    @abstractmethod
    async def peek(self) -> bool:
        """Return True iff the op has reached a terminal state. Cheap;
        OK to call on every request."""
        pass

class TrainOp(Op[None]):
    pass

class SampleOp(Op[Sample]):
    pass
```

For `gemini_msrl`, `peek` is a `GetOperation` call against the LRO name (and the LRO API caches terminal state, so it stays cheap even after completion).

Recipes that persist `op.name` in their state use this to rehydrate a
`TrainOp` wrapper that polls the existing LRO.

### 4.7 HTTP endpoints

#### `POST /tuners` — extended

```jsonc
// request
{
  "name": "my-tuner",
  "recipe": "gemini_msrl",
  "datum_ids": ["d_001", "d_002", "..."],     // REQUIRED, non-empty
  "hparams": { ... }                          // recipe-defined; may include group_size/batch_size
}

// response
{ "tuner_id": "tuner_…", "name": "…", "recipe": "gemini_msrl" }
```

#### `POST /tuners/{id}/runs` — new

Allocates a `run_id`, picks a `datum_id` from the pool, inserts a
`runs` row, returns the assignment.

```jsonc
// request body: empty (room for future filters)

// response: 200 OK
{ "run_id": "run_…", "datum_id": "d_017" }

// response: 204 No Content
// — recipe's dispense_run returned None
//   (e.g. a TrainOp is in flight under sync policy).
// — Retry-After header set to a sensible default (1s in v1).
```

Server behavior:

1. Read `is_training` from `TunerService`'s in-process map.
2. Load the datum pool (`SELECT datum_id FROM datum_rows WHERE tuner_id=…`).
3. Call `tuner.dispense_run(DispenseContext(is_training, datum_pool))`.
4. If `None`: respond `204` with `Retry-After`.
5. Otherwise insert a `runs` row and return `{ run_id, datum_id }`.

One run per call. Workers wanting fan-out issue many parallel POSTs.

#### `POST /openai/v1/chat/completions` — behavior change

- **If `run_id` is present** and matches a row in `runs` for this
  tuner: server records a `ChatCompletionModel` row keyed to that
  `run_id`, stamping `policy_generation` from `Sample`. The `datum_id`
  comes from the existing `RunModel` row (client can't lie).
- **If `run_id` is absent**: behave like today — serve the completion,
  don't persist a `ChatCompletionModel` row, don't touch the run table.
  This keeps the fire-and-forget path for evals / exploration intact.
- An unknown `run_id` returns `409 Conflict`.
- `423 Locked` is no longer needed — barriers are expressed by the
  queue returning `204`, not by completions being refused.

#### `PUT /tuners/{id}/runs/{run_id}/reward` — replaces `POST /rewards`

```jsonc
// request
{ "reward": 0.75 }
// response
{ "run_id": "run_…", "reward": 0.75 }
```

```sql
UPDATE runs
   SET reward     = :reward,
       updated_at = NOW()
 WHERE tuner_id   = :tuner_id
   AND id        = :run_id
   AND reward IS NULL
   AND expires_at > NOW();
```

The `reward IS NULL` guard makes the call idempotent in the
single-write-wins sense (a duplicate PUT returns `409 Conflict`); the
`expires_at > NOW()` guard rejects late rewards on an expired lease
(also `409 Conflict`), so the dispenser is free to re-issue that
`datum_id` without fear of double-counting. An unknown `run_id`
returns `404 Not Found`.

`RewardModel` is **removed** from the schema in this iteration.

#### `GET /tuners/{id}` — optional, observability only

Returns `{ tuner_id, name, recipe, training: bool, pending_run_count,
last_recorded_policy_generation }`. Not on the critical path; clients
don't need it to drive the loop.

### 4.8 `TunerService.train()`

**The DB is the lock.** There is no in-memory `_train_tasks` map and
no per-process asyncio handle. `is_training` is computed from
`Tuner.in_flight_train_op().peek()` — a recipe-side call that, for
LRO-shaped backends like `gemini_msrl`, is just a cached
`GetOperation` against the persisted op name. This is cheap enough to
call on every `POST /runs`.

```python
class TunerService:
    async def is_training(self, tuner_id: str) -> bool:
        tuner = await self.get(tuner_id)
        if tuner is None:
            return False
        op = await tuner.in_flight_train_op()
        return op is not None and not await op.peek()

    async def maybe_train(self, tuner_id: str) -> None:
        # SELECT … FOR UPDATE on the tuner row serializes claim attempts
        # across drivers / processes. Whoever wins the row lock either
        # (a) sees an unfinished in_flight_train_op and bails, or
        # (b) commits a fresh one via _drive_step and releases the lock.
        async with self.async_session() as session:
            async with session.begin():
                record = (await session.execute(
                    select(TunerModel)
                    .where(TunerModel.id == tuner_id)
                    .with_for_update()
                )).scalar_one_or_none()
                if record is None:
                    return

                tuner = await self._materialize(tuner_id, record)
                op = await tuner.in_flight_train_op()
                if op is not None and not await op.peek():
                    return                                           # already training
                if not await self._has_enough_consumable_runs(tuner_id, session):
                    return

                batch, run_ids = await self._collect_consumable_batch(tuner_id, session)
                new_op = await tuner.train_step(batch)               # backend accepted
                await session.execute(                               # bump train_count
                    update(RunModel)
                    .where(RunModel.tuner_id == tuner_id)
                    .where(RunModel.id.in_(run_ids))
                    .values(train_count=RunModel.train_count + 1)
                )
                record.state = await tuner.save_state()              # persists new_op.name
                # commit releases the FOR UPDATE lock
```

`_collect_consumable_batch` is the direct successor of today's
`collect_rollout_ready_for_training` in
`src/ollie_rl/service/tuner_service.py` — same group-by-`datum_id`
logic, same `GROUP_SIZE` / `TARGET_GROUP_COUNT` constants (now
recipe-internal per §3 decision 10), and the same
`train_count <= TARGET_MAX_TRAIN_COUNT` (= 0 today) gate. The only
shape change is that it reads from `runs` (one row per run,
`reward IS NOT NULL AND train_count == 0` → ready) instead of from
`rewards` joined to `chat_completions`; the in-memory `Rollout` /
`RolloutRun` it produces stays identical.

Note `expires_at` is **not** in the consumable-batch query — that
field is solely for the leasing path (§4.3). A run that was
dispensed-and-rewarded after its lease expired is unreachable
anyway because `PUT /reward` rejected the late write.

The "did the LRO finish?" check is *also* DB-driven: anyone calling
`is_training(tuner_id)` materializes the tuner from
`TunerModel.state`, asks the recipe to rehydrate its `TrainOp`, and
peeks. There is no in-process latch to keep in sync.

#### Multi-driver / multi-process safety

This design gets multi-driver safety for free, because the
mutual-exclusion primitive is `SELECT … FOR UPDATE` on the
`tuners.id` row inside `maybe_train`:

- Two drivers calling `maybe_train(tuner_id)` simultaneously: one
  takes the row lock, decides whether to start a step, commits, and
  releases. The other sees the just-committed `in_flight_train_op`
  via `peek()` and bails.
- A driver restarting mid-step: the row lock is released by the DB
  on connection close; the next caller sees the persisted
  `active_train_op` in `state` and either waits for it to peek-done
  or starts a fresh step (recipe's choice — for `gemini_msrl` the
  LRO is still running server-side, so `peek()` returns `False` and
  we wait).
- `POST /runs` reads `is_training` without taking the lock; it can
  race with a `maybe_train` that's about to commit a new op, but the
  worst case is one extra `204` (or, conversely, one run dispensed
  against a tuner that's about to start training). That's the same
  race as the previous in-memory version and is benign — the next
  `POST /runs` reconciles.

Note we **do not** need a separate atomic-claim column on `TunerModel`
as `api-design-iterate.md` proposed; the recipe-owned
`active_train_op` field inside `TunerModel.state` is the claim, and
the row-level `FOR UPDATE` is the serialization point.

### 4.9 Auto-train trigger

`maybe_train` is called from the reward endpoint:

```python
@router.put("/tuners/{tuner_id}/runs/{run_id}/reward")
async def put_reward(...):
    await tuner_service.record_reward(tuner_id, run_id, reward)
    await tuner_service.maybe_train(tuner_id)   # fire-and-forget
```

No background sweeper in v1.

### 4.10 Startup recovery

There is no special restart path. Because `is_training` is computed
from `Tuner.in_flight_train_op().peek()` (§4.8) and the recipe state
in `TunerModel.state` already carries the in-flight op id (§4.11),
a fresh process picks up where the old one left off:

- `POST /runs` materializes the tuner from `state`, asks the recipe
  for `in_flight_train_op()`, peeks. If the LRO is still running,
  `is_training` is `True` and the queue stays closed (`204`).
- Once the LRO peeks done, the next `maybe_train` sees that and
  either starts a new step (if there are enough consumable runs) or
  no-ops.
- The batch that the LRO was consuming was marked consumed in the DB
  *before* the crash (`_drive_step` does that inside the same
  transaction that captures the op id), so there's nothing to redo
  on the rollout side.

Recipes that do **not** persist their in-flight op (default `Tuner`
returns `None` from `in_flight_train_op`) fall back to: the LRO may
still be running asynchronously on the recipe side, but the server
treats `is_training` as `False` and may dispense runs against a
half-trained policy. Recipes with cheap polling APIs (like
`gemini_msrl`'s LRO names) should implement `in_flight_train_op` so
the post-crash barrier is honored.

### 4.11 State persistence cadence

`TunerModel.state` is the durable mirror of `Tuner.save_state()` and
is also the multi-driver claim (§4.8). Today it is only written at
create time; that's not enough. New cadence — exactly **two** writes,
both driven by `TunerService` inside a `SELECT … FOR UPDATE`
transaction (recipes do not write to the DB directly):

| When                                            | Why                                                              |
|-------------------------------------------------|------------------------------------------------------------------|
| After `Recipe.create(...)` returns              | (existing) — first persistence of recipe-owned identifiers.       |
| Inside `maybe_train`, after `train_step(batch)` returns and runs are marked consumed, **before** committing the FOR-UPDATE transaction | Capture the in-flight op id (e.g. Vertex AI LRO name) so concurrent drivers see the claim and any future restart can poll/await it. |

There is no post-`wait()` checkpoint, and indeed `op.wait()` is **not
called** under the row lock — only `train_step()` (which just submits
the LRO) and the state checkpoint happen inside the transaction.
After commit, no one is blocking; the next `POST /runs` or
`maybe_train` will peek the LRO from the persisted state and decide
what to do. The active op id stays in the serialized state until the
next training step overwrites it; that's harmless because `peek()`
on a completed LRO is a cheap cached read.

Recipes opt in by widening their state model. For `gemini_msrl`:

```python
class GeminiMsrlRecipeState(BaseModel):
    tuning_job_name: str
    active_train_op: Optional[str] = None      # NEW: LRO resource name of in-flight TrainStep
```

with the tuner internally stamping `self._active_train_op = op.name`
on `train_step(...)`. We do **not** need to track `active_run_ids`
in the state, because `_drive_step` marks runs consumed *before*
checkpointing — the durable source of truth for "which runs were
consumed by which step" is the `runs` table (via a future
`consumed_at` / `consumed_by_policy_generation` column, see Q2),
not the recipe state.

`TunerService._persist_state` is a small helper:

```python
async def _persist_state(self, tuner_id: str, state_str: str) -> None:
    async with self.async_session() as session:
        async with session.begin():
            await session.execute(
                update(TunerModel)
                .where(TunerModel.id == tuner_id)
                .values(state=state_str)
            )
```

Why driven by `TunerService` rather than each recipe writing on its
own?

- **Single writer.** Only `TunerService` owns the SQLAlchemy session;
  recipes stay pure (no DB import).
- **Atomic with the other DB writes in `maybe_train`** (the
  consumed-marking write and the state checkpoint happen inside the
  same FOR-UPDATE transaction).
- **Uniform recovery path.** §4.10's behaviour falls out of `peek()`
  for every recipe that returns a non-`None` `in_flight_train_op`;
  the API contract is just "put it in your state and we'll persist
  it for you."

Trade-offs / caveats:

- A crash *between* `train_step(...)` succeeding on the recipe
  backend and the transaction commit leaks an LRO on the recipe side
  while the server's `state` still points at the previous one.
  Acceptable — that window is microseconds wide, and the next
  training step will create a fresh op (recipe backends like Vertex
  AI tolerate orphaned LROs).
- `peek()` must be idempotent and side-effect free. Vertex AI LROs
  already satisfy this; future recipes need to honor it.

---

## 5. Client flow (sync mode)

```mermaid
sequenceDiagram
    participant C as Sync RL Client
    participant API as Ollie RL API
    participant DB as DB

    C->>API: POST /tuners { recipe, datum_ids: [...] }
    API-->>C: { tuner_id }

    loop each training step
        loop fan out N samplers (parallel)
            C->>API: POST /tuners/{id}/runs
            alt queue open
                API->>API: dispense_run(is_training=False, datum_pool)
                API->>DB: INSERT runs
                API-->>C: 200 { run_id, datum_id }
                C->>API: POST /openai/v1/chat/completions { run_id, ... }
                API->>DB: INSERT chat_completions (policy_generation)
                API-->>C: ChatCompletion
                C->>API: PUT /tuners/{id}/runs/{run_id}/reward { reward }
                API->>DB: UPDATE runs SET reward=…
            else barrier closed
                API-->>C: 204 + Retry-After
                Note over C: backoff and retry
            end
        end

        Note over API: maybe_train sees ≥ batch_size consumable runs<br/>(reward IS NOT NULL AND train_count == 0)
        API->>DB: BEGIN; SELECT tuners WHERE id=? FOR UPDATE
        API->>API: tuner.train_step(batch) → new_op
        API->>DB: UPDATE runs SET train_count = train_count + 1
        API->>DB: UPDATE tuners SET state=tuner.save_state() (with op id)
        API->>DB: COMMIT
        Note over API: dispense_run now sees in_flight_train_op().peek()==False ⇒ 204
        Note over API: (no in-process latch — every POST /runs peeks the LRO via state)
        Note over C: next POST returns 200 ← barrier released once peek() flips True
    end
```

The async client flow is **literally the same diagram** — the recipe's
`dispense_run` simply ignores `ctx.is_training` and the client never
sees a `204`.

---

## 6. What gets simpler vs. the current repo

| Concern                                | Current                                              | Proposed                                                                       |
|----------------------------------------|------------------------------------------------------|--------------------------------------------------------------------------------|
| Sync barrier                           | none — completions served from half-trained policy   | implicit: `dispense_run` returns `None` during training                        |
| Run lifecycle                          | implicit, client-minted `run_id`                     | explicit `RunModel` row, server-allocated `run_id`                             |
| Sizing on the wire                     | `GROUP_SIZE`/`TARGET_GROUP_COUNT` private constants  | recipe-internal (or hparams at create time)                                    |
| Multi-worker fan-out                   | client coordinates `datum_id` picks                  | server arbitrates: each `POST /runs` returns a unique assignment               |
| Recipe-driven curriculum               | not possible without API change                      | recipe owns `dispense_run`                                                     |
| `policy_generation` capture            | already in `Sample`, dropped on the floor            | persisted on `ChatCompletionModel`                                             |
| Reward storage                         | separate `RewardModel`                               | merged onto `RunModel`                                                         |
| Datum pool                             | implicit (client picks at sample time)               | explicit, required at tuner creation                                           |
| Train-step mutual exclusion            | in-process only (single driver assumed)              | DB-driven via `SELECT … FOR UPDATE` on `tuners` row + `in_flight_train_op().peek()` (multi-driver safe) |
| Run lease / redispense                 | none — orphaned runs sit in `rewards` forever        | `runs.expires_at` (TTL) — dispenser may re-issue datum after lease expires; `PUT /reward` rejects late writes |
| Consumed-by-training tracking          | `RewardModel.train_count` bumped after train         | `RunModel.train_count` bumped after train (same semantics, new table)          |

What we don't take on:

- Streaming corpus updates after creation.
- Background hard-deletion of expired rows (a `WHERE expires_at >
  NOW()` filter is enough for v1).

---

## 7. Implementation plan

### 7.1 Ship together

1. **Rename**
   - `Sample.step_id` → `Sample.policy_generation` (stays `str`).
   - Update `gemini_msrl.sample()` and `test_gemini_msrl.py`.
2. **DB schema**
   - `ChatCompletionModel.policy_generation` (uncomment + rename).
   - `RunModel` (new, absorbs `RewardModel`).
   - `DatumRowModel` (new, table `datum_rows`).
   - Drop `RewardModel`.
3. **`Tuner` hooks**
   - `dispense_run(ctx) -> Optional[RunAssignment]` with a sane default
     implementation in the base class.
   - `in_flight_train_op() -> Optional[TrainOp]` with a default of
     `None` (recipes opt in for restart-safe training).
4. **`TunerService`**
   - `register_datums`, `dispense_run`, `record_chat_completion`,
     `record_reward`, `maybe_train`.
   - `is_training(tuner_id)` is computed by
     `(await tuner.in_flight_train_op()).peek()` — no in-memory
     latch.
   - `maybe_train` wraps `train_step` + `train_count += 1` + state
     checkpoint in a `SELECT … FOR UPDATE` transaction on `tuners`
     (§4.8 / §4.11). `op.wait()` is **not** called under the lock.
   - `collect_rollout_ready_for_training` becomes
     `_collect_consumable_batch`, reading from `runs` with
     `reward IS NOT NULL AND train_count == 0` (the existing
     `train_count` semantics, just on the new table — see §4.3).
5. **`gemini_msrl` state migration**
   - Add `active_train_op: Optional[str]` to `GeminiMsrlRecipeState`.
   - `GeminiMsrlTuner.train_step`: stamp `self._active_train_op =
     op.name` before returning.
   - `GeminiMsrlRecipe.restore`: when `active_train_op` is set,
     rehydrate a `GeminiMsrlTrainingOp(client, active_train_op,
     config)` and expose it via `Tuner.in_flight_train_op`.
6. **HTTP**
   - Extend `POST /tuners` to require `datum_ids`.
   - `POST /tuners/{id}/runs`.
   - `POST /openai/v1/chat/completions`: `run_id` optional; when
     present, validate against an existing run and stamp the chat
     completion row.
   - `PUT /tuners/{id}/runs/{run_id}/reward` (replaces `POST /rewards`).
   - Optionally extend `GET /tuners/{id}` for observability.
7. **Docs**
   - `.agents/skills/dev/references/sync-rl.md`: rewrite the barrier
     section around the queue.
   - `.agents/skills/dev/references/data-model.md`: add `RunModel`,
     `DatumRowModel`, `policy_generation`; remove `RewardModel`.
8. **Tests** (after sign-off, per AGENTS.md)
   - Run lifecycle: dispense → completion → reward → `train_count`
     bumped after training.
   - Barrier: while `is_training`, `POST /runs` returns `204`.
   - Multi-worker dispatch: concurrent `POST /runs` never share a
     `run_id`.
   - `policy_generation` from `Sample` round-trips through DB.
   - `chat/completions` with an unknown `run_id` returns `409`.
   - `chat/completions` without `run_id` succeeds, writes nothing.
   - Duplicate `PUT /reward` returns `409` (atomic guard).
   - Multi-completion runs: more than one `ChatCompletionModel` row
     per `RunModel`, each carrying its own `policy_generation`.
   - State checkpoint: after `train_step` returns and `train_count`
     is bumped, `TunerModel.state` contains the recipe's in-flight
     op id (single FOR-UPDATE transaction).
   - `is_training` is DB-derived: a fresh process that has never
     called `train_step` correctly reports `True` while a previously
     committed LRO is still running, and `False` after `peek()`
     flips.
   - Multi-driver safety: two concurrent `maybe_train` calls produce
     exactly one new train op (the loser sees the in-flight op via
     `peek()` and bails).
   - `train_count` gate: a run with `train_count > 0` is never
     re-included in a subsequent batch.
   - Lease expiration: `PUT /reward` against a run whose
     `expires_at <= NOW()` returns `409`; the dispenser is free to
     re-issue that `datum_id` with a fresh `run_id`.
   - `train_count` vs. `expires_at` are independent: a rewarded run
     with `train_count == 0` stays consumable for training even
     after `expires_at` has passed (the lease guard is only
     consulted on the rollout side, never on the training side).

### 7.2 Deferred

9. Streamed datum-pool extension after creation.
10. Server-side curriculum policies as first-class primitives.
11. Off-policy correction (importance ratios) inside `train_step`,
    keyed on `policy_generation`.
12. Background hard-deletion of expired-and-unrewarded `runs`
    rows (`WHERE expires_at <= NOW() AND reward IS NULL`).

---

## 8. Open questions

| #  | Question                                                                                                | Default if no answer                                                          |
|----|---------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| Q1 | `Retry-After` on the 204 — fixed (e.g. `1s`) or recipe-tunable?                                         | Fixed `1` in v1; recipe-tunable later via hparams.                            |
| Q2 | How do we mark a run as "consumed by training"? | `RunModel.train_count: int` (ported from `RewardModel.train_count`); ready iff `train_count == 0`; bumped to 1 after train. Promote to `consumed_by_policy_generation` later if off-policy work needs more than a counter. |
| Q3 | Should `POST /tuners` reject a *too-small* `datum_ids` array (e.g. < expected batch size)?              | Warn but accept; the recipe can always cycle.                                 |
| Q4 | Is `dispense_run` sync or async on the `Tuner`?                                                          | Sync; the DB layer can be async around it.                                    |
| Q5 | Does `PUT /reward` accept reward updates after a run is already consumed?                               | No — return `409 Conflict`; rewards are write-once.                           |
| Q6 | Should `GET /tuners/{id}` expose `training: bool` even though it's derived from a `peek()` poll? | Yes; same DB-derived value `POST /runs` uses, just exposed for debugging.     |
| Q7 | Should `TunerService` also checkpoint state after non-train events (e.g. after every reward) so opaque recipe counters stay fresh? | No in v1 — `train_step` boundaries are the only point where state actually changes for current recipes. |
| Q8 | Default `run_ttl` for `runs.expires_at`?                                                                | `5 min` in v1; recipe-tunable later via hparams.                              |
| Q9 | Does `peek()` need a result cache inside `TunerService` to avoid hammering the LRO backend on hot `POST /runs` paths? | Not in v1 — Vertex AI's `GetOperation` is cheap and itself caches the terminal state. Revisit if `peek` shows up in profiling. |

---

## 9. Non-goals

- **Server-owned corpora.** The server holds opaque `datum_id`s, not
  dataset rows. Clients still own data.
- **Per-rollout heartbeats.** Crash recovery for in-flight runs is
  TTL-based (`runs.expires_at`, §4.3), not heartbeat-based. A
  client that crashes mid-completion just lets its run expire.
- **Streamed lease delivery (WebSocket / SSE).** Polling-with-204 is
  simpler and matches the OpenAI-shaped completions endpoint right
  next door.
- **A separate `RewardModel`.** Reward is a column on `RunModel`, not
  its own entity.
- **Streaming corpus updates.** `POST /tuners/{id}/datum_pool` is
  intentionally not in this iteration.
- **Hard-deletion of expired rows.** A `WHERE expires_at > NOW()`
  predicate on the consumable-runs query is enough; a janitor job
  can come later.
