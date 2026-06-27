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

Status: **in progress.** A large chunk of the plumbing has already
landed in staging — see §7.1 for the per-step checklist. The pieces
still missing are the queue endpoint itself (`POST /tuners/{id}/runs`),
the recipe-side `dispense_run` / `in_flight_train_op` hooks, the
DB-driven `is_training` derivation, and the `maybe_train` barrier with
its `SELECT … FOR UPDATE` mutual exclusion. Everything else (schema,
rename, datum-pool registration, `policy_generation` capture on
completions, run-keyed reward endpoint, `GeminiMsrlRecipeState.last_train_op`,
`Op.peek()`) is already shipped.

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
| When is a training step happening?                | recipe              | recipe's `TrainOp`, polled via `Op.peek()`             |
| Who decides what gets sampled next?               | recipe              | `Tuner.dispense_run(ctx)` returns the next assignment  |
| How is work handed to a sampler?                  | server              | `POST /tuners/{id}/runs` returns a `(run_id, datum_id, expires_at)`|
| Which policy did a sample come from?              | trainer (recorded)  | `Sample.policy_generation` → `ChatCompletionModel`     |
| Is sampling allowed during training?              | recipe              | `dispense_run` returns `None` (or not) while busy      |
| Where does the in-flight train op live across restarts? | recipe state    | `GeminiMsrlRecipeState.last_train_op` (already shipped) |

We do **not** introduce an `IDLE / TRAINING` state column on
`TunerModel`. The recipe's `TrainOp` is the source of truth for "are we
training right now"; the server materializes the recipe from
`TunerModel.state` (already persisted via `StateStore`) and asks the
recipe to rehydrate its `TrainOp` so it can be `peek()`'d.

---

## 3. Decisions

| # | Decision                                                                                                   | Rationale                                                                                                |
|---|------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| 1 | Server dispenses runs via `POST /tuners/{id}/runs`. Response is `{ run_id, datum_id, expires_at }`.       | Leasing is state-mutating (inserts a row, allocates `run_id`) — POST is the right verb.                  |
| 2 | **One run per request.** No batching.                                                                      | Simplest contract; multi-worker fan-out is many parallel POSTs.                                          |
| 3 | Datum pool is **registered at `POST /tuners` creation time** (`datum_ids` in the request body).            | A tuner is useless without a corpus; making it required eliminates a "did you forget to register?" failure mode. |
| 4 | Rename `Sample.step_id` → `Sample.policy_generation`. Type stays `str` (opaque, recipe-defined).           | The name actually describes the concept (model weight version). Type stays opaque because recipes vary.   |
| 5 | Persist `policy_generation` on **`ChatCompletionModel`**, not on `RunModel`.                               | A single run may produce multiple chat completions (multi-step / tool-using trajectories), each at a different generation. |
| 6 | The barrier is implicit: while a `TrainOp` is in flight, `dispense_run` returns `None` ⇒ HTTP `204`.       | One endpoint, one mental model. No `423`, no separate `state` polling.                                   |
| 7 | **No** `Tuner.allows_sampling_during_training` property.                                                    | The "what do we do during training" policy lives inside `dispense_run` itself; that's expressive enough. |
| 8 | **State persistence is recipe-driven via `StateStore`.** The recipe calls `state_store.save(blob)` whenever its in-process state has meaningfully changed; `TunerService` only supplies the DB-backed implementation. | Earlier drafts of this doc made `TunerService` the single writer. The shipped `StateStore` Protocol is cleaner: recipes stay pure (no DB import) and own their cadence; the DB is just a key-value backend. |
| 9 | **No separate `RewardModel`.** Reward lives as a column on `RunModel` (which is the canonical run record). | A reward without a run is meaningless; a run with no reward is just unfinished. One row per run.         |
| 10| `group_size` / `batch_size` are **recipe-internal** (or recipe hparams), never on the wire.               | They're scheduling, not API contract.                                                                    |

---

## 4. Surface changes

### 4.1 `TunerModel` (DB) — shipped

```python
class TunerModel(BaseModel):
    __tablename__ = "tuners"

    id:    Mapped[str]           = mapped_column(String(36),  primary_key=True)
    name:  Mapped[str]           = mapped_column(String(255), nullable=False)
    kind:  Mapped[str]           = mapped_column(String(255), nullable=False, index=True)
    # NULL between row creation and the Tuner's first save; subsequently
    # whatever the recipe most recently passed to state_store.save(...).
    state: Mapped[Optional[str]] = mapped_column(Text,        nullable=True)
```

No `is_training` column — that lives in the recipe state.

### 4.2 `ChatCompletionModel` (DB) — shipped

```python
class ChatCompletionModel(BaseModel):
    __tablename__ = "chat_completions"

    id:                Mapped[str]      = mapped_column(String(255), primary_key=True)
    tuner_id:          Mapped[str]      = mapped_column(String(255), ForeignKey("tuners.id"), nullable=False)
    policy_generation: Mapped[str]      = mapped_column(String(255), nullable=False, index=True)
    run_id:            Mapped[Optional[str]] = mapped_column(String(255), nullable=False, index=True)
    datum_id:          Mapped[str]      = mapped_column(String(255), nullable=False, index=True)
    created_at:        Mapped[datetime] = mapped_column(DateTime,    nullable=False, default=func.now())
    updated_at:        Mapped[datetime] = mapped_column(DateTime,    nullable=False, default=func.now(), onupdate=func.now())
```

`policy_generation` is stamped from `Sample.policy_generation` at
completion-record time. One `ChatCompletionModel` row per LLM
round-trip; many rows per `RunModel` in the multi-step / tool-using
case.

### 4.3 `RunModel` (DB) — shipped

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

**`train_count: int`** — consumed-by-training counter. A run is
"consumable for training" iff `reward IS NOT NULL AND train_count == 0`.
After a successful `train_step`, `TunerService.train` bumps
`train_count` to `1` so the same run is never trained on twice. Future
off-policy work can change the threshold (`train_count <= K`) without
schema churn.

**`expires_at: datetime`** — lease deadline for redispense, **not**
related to training consumption. Stamped at dispense time as
`NOW() + run_ttl` (recipe hparam, default `5 min` in v1). Its only job
is to let the dispenser re-distribute datums whose runs were dispensed
but never rewarded (sampler crashed, network blip, etc.):

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
  double-count. **(shipped — `RunExpiredError`.)**
- A background sweeper could later hard-delete expired-and-unrewarded
  rows for tidiness, but it's not required for v1.

Chat completions reference the run via the existing
`ChatCompletionModel.run_id`.

### 4.4 `DatumRowModel` (DB) — shipped

```python
class DatumRowModel(BaseModel):
    __tablename__ = "datum_rows"

    tuner_id: Mapped[str] = mapped_column(String(255), ForeignKey("tuners.id"), primary_key=True)
    datum_id: Mapped[str] = mapped_column(String(255), primary_key=True)
```

Server treats `datum_id` as opaque. Populated at `POST /tuners` time
from the request body. No streaming endpoint in v1 — re-create the
tuner if the corpus needs to grow.

### 4.5 `Sample` (in-process) — shipped

```python
class Sample(BaseModel):
    completion: ChatCompletion
    policy_generation: str
```

`gemini_msrl.sample()` already builds this from `response.train_step_id`;
`test_gemini_msrl.py` already asserts on it. Type stays `str` so that
recipes with opaque generation identifiers (hashes, LRO names,
semver-ish strings) keep working.

### 4.6 `Tuner` (in-process)

The `cookbook/types.py` skeleton is already in place:

```python
@dataclass
class DispenseContext:
    is_training: bool
    datum_pool: List[str]

@dataclass
class RunAssignment:
    run_id: str
    datum_id: str

class Op(ABC, Generic[T]):
    @abstractmethod
    async def wait(self) -> T: ...
    @abstractmethod
    async def peek(self) -> bool: ...   # already implemented for gemini_msrl

class TrainOp(Op[None]):   ...
class SampleOp(Op[Sample]): ...

class Tuner(ABC):
    @property
    @abstractmethod
    def kind(self) -> str: ...
    @abstractmethod
    async def sample(self, request: ChatCompletionRequest) -> SampleOp: ...
    @abstractmethod
    async def train_step(self, examples: List[Example]) -> TrainOp: ...
```

Two hooks still need to be added (per §7.1):

```python
class Tuner(ABC):
    def dispense_run(self, ctx: DispenseContext) -> Optional[RunAssignment]:
        """
        Recipe-owned dispatch. Default implementation lives on the base
        class:
          - if ctx.is_training: return None  (sync-safe)
          - if not ctx.datum_pool: return None
          - pick a datum_id (recipe's choice of policy), mint a run_id,
            and return the assignment.
        """
        ...

    async def in_flight_train_op(self) -> Optional[TrainOp]:
        """
        Return the TrainOp captured in the most recently saved state
        (i.e. the train op that was running when state was last
        checkpointed), so TunerService can poll / await it. Returns
        None when no train op was in flight. Default impl: return None.
        """
        return None
```

Sync GRPO uses the default `dispense_run`. Async recipes override and
ignore `ctx.is_training`, possibly capping in-flight runs etc.

No `allows_sampling_during_training` property — the logic lives inside
`dispense_run` itself.

For `gemini_msrl`, `peek` is already a `GetOperation` call against the
LRO name (and the LRO API caches terminal state, so it stays cheap
even after completion). `in_flight_train_op` will wrap the persisted
`state.last_train_op` (already on `GeminiMsrlRecipeState`) in a fresh
`GeminiMsrlTrainingOp`.

#### `Recipe` / `Cookbook` (already aligned)

```python
class Recipe(ABC):
    @abstractmethod
    async def create(self, name: str, state_store: StateStore) -> Tuner: ...

class Cookbook:
    @classmethod
    async def open(cls, kind: str, name: str, state_store: StateStore) -> Tuner: ...
```

`Recipe.create` (renamed from `Recipe.open`) is the bootstrap-or-resume
entry point. The recipe inspects `await state_store.load()` to decide
which branch.

### 4.7 `StateStore` Protocol — shipped

The hidden hero of this iteration. Lives in `cookbook/types.py`:

```python
class StateStore(Protocol):
    async def load(self) -> Optional[str]: ...
    async def save(self, state: str) -> None: ...
```

`TunerService` supplies `_DbStateStore(tuner_id)`, which `UPDATE`s the
`tuners.state` column. `GeminiMsrlTuner._persist_state` serializes
`GeminiMsrlRecipeState` (which now carries `last_train_op:
Optional[str]`) and calls `state_store.save(...)` every time the
in-flight LRO name changes — currently:

- Once at bootstrap (after `create_tuning_job`).
- Once inside `train_step(...)`, right after the LRO is submitted and
  before returning. This is what makes the restart-recovery story work
  without an extra `_persist_state` checkpoint in `TunerService`.

Recipes that need finer cadence (e.g. checkpoint after every reward)
can just call `state_store.save` from wherever — there is no
TunerService coordination required.

### 4.8 HTTP endpoints

#### `POST /tuners` — shipped

```jsonc
// request
{
  "name": "my-tuner",
  "recipe": "gemini_msrl",
  "datum_ids": ["d_001", "d_002", "..."],     // REQUIRED, non-empty (400 if empty)
  "hparams": { ... }                          // recipe-defined; may include group_size/batch_size
}

// response
{ "tuner_id": "tuner_…", "name": "…", "recipe": "gemini_msrl" }
```

#### `POST /tuners/{id}/runs` — still TODO

Allocates a `run_id`, picks a `datum_id` from the pool, inserts a
`runs` row, returns the assignment.

```jsonc
// request body: empty (room for future filters)

// response: 200 OK
{ "run_id": "run_…", "datum_id": "d_017", "expires_at": "2026-06-26T18:00:00Z" }

// response: 204 No Content
// — recipe's dispense_run returned None
//   (e.g. a TrainOp is in flight under sync policy).
// — Retry-After header set to a sensible default (1s in v1).
```

Server behavior (to implement in `TunerService.dispense_run(tuner_id)`):

1. Read `is_training` by materializing the tuner from
   `TunerModel.state` and calling
   `(await tuner.in_flight_train_op()).peek()`. `None` ⇒ `False`.
2. Load the datum pool (`SELECT datum_id FROM datum_rows WHERE tuner_id=…`).
3. Call `tuner.dispense_run(DispenseContext(is_training, datum_pool))`.
4. If `None`: respond `204` with `Retry-After`.
5. Otherwise insert a `runs` row (with
   `expires_at = NOW() + run_ttl`) and return `{ run_id, datum_id, expires_at }`.

One run per call. Workers wanting fan-out issue many parallel POSTs.

#### `POST /openai/v1/chat/completions` — shipped (with caveats)

Current shipped behavior:

- **If `x_run_id` header is present** and matches a row in `runs` for
  this tuner: server overrides `datum_id` from the DB record (client
  can't lie), serves the completion, and persists a
  `ChatCompletionModel` row keyed to that `run_id`, stamping
  `policy_generation` from `Sample`.
- **If `x_run_id` is absent**: behave like today — serve the
  completion, don't persist a `ChatCompletionModel` row, don't touch
  the run table.
- An unknown `x_run_id` returns `409 Conflict`.
- `423 Locked` is no longer needed — barriers are expressed by the
  queue returning `204`, not by completions being refused.

Caveat to address before declaring this endpoint done: when
`x_run_id` is absent, the current handler still references
`x_datum_id` which is never assigned. Either reject the run-less path
or thread an explicit "evaluation" mode through.

#### `PUT /tuners/{id}/runs/{run_id}/reward` — shipped

```jsonc
// request
{ "reward": 0.75 }
// response
{ "run_id": "run_…", "reward": 0.75 }
```

Behind the scenes, `TunerService.update_reward`:

```sql
SELECT … FROM runs WHERE id = :run_id AND tuner_id = :tuner_id
  -- 404 RunNotFoundError if missing
  -- 409 RewardAlreadySetError if reward IS NOT NULL
  -- 409 RunExpiredError       if expires_at <= NOW()
UPDATE runs SET reward = :reward, updated_at = NOW() WHERE id = :run_id;
```

The reward-already-set and lease-expiration guards keep the dispenser
free to re-issue an expired `datum_id` without fear of double-counting.

`RewardModel` has been **removed** from the schema.

After the UPDATE, the endpoint fires-and-forgets
`asyncio.create_task(services.tuner.train(tuner_id))`. This is the
current trigger; once `maybe_train` lands (§4.10) it will replace
this call directly.

#### `GET /tuners/{id}` — optional, observability only (deferred)

Will return `{ tuner_id, name, recipe, training: bool,
pending_run_count, last_recorded_policy_generation }`. Not on the
critical path; clients don't need it to drive the loop.

### 4.9 `TunerService.train` (today) vs. `maybe_train` (target)

#### Today (shipped)

`TunerService.train(tuner_id)`:

1. `collect_rollout_ready_for_training(tuner_id)` — `SELECT * FROM runs
   WHERE tuner_id = ? AND train_count == 0 AND reward IS NOT NULL`,
   group by `datum_id`, drop groups whose size isn't exactly
   `GROUP_SIZE = 16`, compute GRPO advantages.
2. Require ≥ `TARGET_GROUP_COUNT = 32` groups, else bail.
3. Map runs → `Example(chat_completion_id, advantage)` by joining
   `ChatCompletionModel` on `run_id`.
4. `train_op = await tuner.train_step(examples)` — the recipe
   submits the LRO and persists `state.last_train_op = op.name` via
   `state_store.save(...)` before returning (already shipped on
   `GeminiMsrlTuner`).
5. `UPDATE runs SET train_count = train_count + 1 WHERE id IN (…)`.
6. `await train_op.wait()`.

Multi-driver safety: **none.** Two concurrent `train(tuner_id)` calls
both race against `tuner.train_step` — `GeminiMsrlTuner.train_step`
guards via `state.last_train_op.peek()` and raises `RuntimeError` on
the loser, but the loser's run-collection work and DB roundtrips are
wasted.

#### Target (TODO)

Rename to `maybe_train` and wrap the critical section in a row-level
lock on `tuners`:

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
        # across drivers / processes. Whoever wins either
        # (a) sees an unfinished in_flight_train_op and bails, or
        # (b) commits a fresh one via train_step and releases the lock.
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

                batch, run_ids = await self._collect_consumable_batch(tuner_id, session)
                if not batch:
                    return

                await tuner.train_step(batch)                        # submits LRO + state_store.save
                await session.execute(                               # bump train_count
                    update(RunModel)
                    .where(RunModel.tuner_id == tuner_id)
                    .where(RunModel.id.in_(run_ids))
                    .values(train_count=RunModel.train_count + 1)
                )
                # commit releases the FOR UPDATE lock
```

`_collect_consumable_batch` is the direct successor of today's
`collect_rollout_ready_for_training`: same group-by-`datum_id` logic,
same `GROUP_SIZE` / `TARGET_GROUP_COUNT` constants (now
recipe-internal per §3 decision 10), same
`train_count <= TARGET_MAX_TRAIN_COUNT` (= 0 today) gate. The shape
changes are:

- It returns `(rollouts, run_ids)` so the `train_count` UPDATE can run
  inside the FOR-UPDATE transaction.
- It takes the open `session` so it joins the same transaction.

Note `expires_at` is **not** in the consumable-batch query — that field
is solely for the leasing path (§4.3). A run that was
dispensed-and-rewarded after its lease expired is unreachable anyway
because `PUT /reward` rejected the late write.

The "did the LRO finish?" check is also DB-derived: anyone calling
`is_training(tuner_id)` materializes the tuner from
`TunerModel.state`, asks the recipe to rehydrate its `TrainOp`, and
peeks. There is no in-process latch to keep in sync.

#### Multi-driver / multi-process safety

This design gets multi-driver safety for free, because the
mutual-exclusion primitive is `SELECT … FOR UPDATE` on the `tuners.id`
row inside `maybe_train`:

- Two drivers calling `maybe_train(tuner_id)` simultaneously: one
  takes the row lock, decides whether to start a step, commits, and
  releases. The other sees the just-committed `last_train_op` via
  `peek()` and bails.
- A driver restarting mid-step: the row lock is released by the DB on
  connection close; the next caller sees the persisted
  `last_train_op` in `state` and either waits for it to peek-done or
  starts a fresh step (recipe's choice — for `gemini_msrl` the LRO is
  still running server-side, so `peek()` returns `False` and we
  return early without dispensing).
- `POST /runs` reads `is_training` without taking the lock; it can
  race with a `maybe_train` that's about to commit a new op, but the
  worst case is one extra `204` (or, conversely, one run dispensed
  against a tuner that's about to start training). That's the same
  race as the previous in-memory version and is benign — the next
  `POST /runs` reconciles.

Note we **do not** need a separate atomic-claim column on `TunerModel`
as `api-design-iterate.md` proposed; the recipe-owned `last_train_op`
field inside `TunerModel.state` is the claim, and the row-level `FOR
UPDATE` is the serialization point.

### 4.10 Auto-train trigger

Today (shipped):

```python
@app.put("/tuners/{tuner_id}/runs/{run_id}/reward")
async def put_reward(...):
    await services.tuner.update_reward(tuner_id, run_id, reward)
    asyncio.create_task(services.tuner.train(tuner_id))     # fire-and-forget
    ...
```

Target: just swap `services.tuner.train` for `services.tuner.maybe_train`
once it lands. No other call sites change. No background sweeper in v1.

### 4.11 Startup recovery

There is no special restart path. Because `is_training` is computed
from `Tuner.in_flight_train_op().peek()` (§4.9) and the recipe state
in `TunerModel.state` already carries the in-flight op id (already
true for `gemini_msrl` via `state.last_train_op`), a fresh process
picks up where the old one left off:

- `POST /runs` materializes the tuner from `state`, asks the recipe
  for `in_flight_train_op()`, peeks. If the LRO is still running,
  `is_training` is `True` and the queue stays closed (`204`).
- Once the LRO peeks done, the next `maybe_train` sees that and
  either starts a new step (if there are enough consumable runs) or
  no-ops.
- The batch that the LRO was consuming was marked consumed in the DB
  *before* the crash (the `train_count` UPDATE happens inside the
  same FOR-UPDATE transaction as `train_step`), so there's nothing to
  redo on the rollout side.

Recipes that do **not** persist their in-flight op (default `Tuner`
returns `None` from `in_flight_train_op`) fall back to: the LRO may
still be running asynchronously on the recipe side, but the server
treats `is_training` as `False` and may dispense runs against a
half-trained policy. Recipes with cheap polling APIs (like
`gemini_msrl`'s LRO names) should implement `in_flight_train_op` so
the post-crash barrier is honored.

### 4.12 State persistence cadence

Driven entirely by the recipe, via the `StateStore` Protocol.

For `GeminiMsrlTuner` today:

| When                                        | What gets saved                                                                  |
|---------------------------------------------|----------------------------------------------------------------------------------|
| `Recipe.create` (bootstrap branch)          | `GeminiMsrlRecipeState(tuning_job_name=job.name)` — first persistence.            |
| `train_step(...)`, after LRO submit         | `GeminiMsrlRecipeState(tuning_job_name=…, last_train_op=op.name)` — captures the in-flight op id so concurrent drivers see the claim and any future restart can poll/await it. |

No post-`wait()` checkpoint, and indeed `op.wait()` is not called
inside `train_step`. After the recipe's `state_store.save(...)`
returns, the LRO name is durable; the next `POST /runs` or
`maybe_train` will peek it from `TunerModel.state` and decide what to
do. The `last_train_op` field stays in the serialized state until the
next training step overwrites it; that's harmless because `peek()` on
a completed LRO is a cheap cached read.

Once `maybe_train` lands (§4.9), the `train_step(...)` call and the
`train_count` UPDATE will run inside the same FOR-UPDATE transaction
on `tuners`. The recipe's `state_store.save(...)` inside
`train_step(...)` uses its own short-lived session, but
`_DbStateStore.save` is itself transactional, so the visible-ordering
guarantee is: by the time `train_step(...)` returns, the new
`last_train_op` is durably committed.

Trade-offs / caveats:

- A crash *between* `train_step(...)` submitting the LRO and
  `state_store.save` committing leaks an LRO on the recipe side while
  the server's `state` still points at the previous one. Acceptable —
  that window is microseconds wide, and the next training step will
  create a fresh op (recipe backends like Vertex AI tolerate orphaned
  LROs).
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
                API-->>C: 200 { run_id, datum_id, expires_at }
                C->>API: POST /openai/v1/chat/completions { x_run_id, ... }
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
        API->>API: tuner.train_step(batch) → state_store.save(last_train_op)
        API->>DB: UPDATE runs SET train_count = train_count + 1
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

| Concern                                | Today (some shipped, some not)                       | Target                                                                         |
|----------------------------------------|------------------------------------------------------|--------------------------------------------------------------------------------|
| Sync barrier                           | none — completions served from half-trained policy   | implicit: `dispense_run` returns `None` during training                        |
| Run lifecycle                          | partial: `RunModel` exists, but server-allocated `run_id` and `POST /runs` are not wired yet | explicit `RunModel` row, server-allocated `run_id`                             |
| Sizing on the wire                     | `GROUP_SIZE`/`TARGET_GROUP_COUNT` private constants in `tuner_service.py` | recipe-internal (or hparams at create time)                                    |
| Multi-worker fan-out                   | client coordinates `datum_id` picks                  | server arbitrates: each `POST /runs` returns a unique assignment               |
| Recipe-driven curriculum               | not possible without API change                      | recipe owns `dispense_run`                                                     |
| `policy_generation` capture            | **shipped** — persisted on `ChatCompletionModel`     | (unchanged)                                                                    |
| Reward storage                         | **shipped** — merged onto `RunModel`                 | (unchanged)                                                                    |
| Datum pool                             | **shipped** — explicit, required at tuner creation   | (unchanged)                                                                    |
| State persistence                      | **shipped** — recipe-driven via `StateStore`         | (unchanged)                                                                    |
| Train-step mutual exclusion            | in-process only (single driver assumed; `train_step` guards via `state.last_train_op.peek()` but losers waste work) | DB-driven via `SELECT … FOR UPDATE` on `tuners` row + `in_flight_train_op().peek()` (multi-driver safe) |
| Run lease / redispense                 | partial: `runs.expires_at` exists, `PUT /reward` rejects late writes; **not used** by a dispenser yet | dispenser may re-issue datum after lease expires                               |
| Consumed-by-training tracking          | **shipped** — `RunModel.train_count` bumped after train | (unchanged)                                                                    |

What we don't take on:

- Streaming corpus updates after creation.
- Background hard-deletion of expired rows (a `WHERE expires_at >
  NOW()` filter is enough for v1).

---

## 7. Implementation plan

### 7.1 Ship together

Legend: `[x]` landed in staging, `[ ]` still to do.

1. **Rename**
   - `[x]` `Sample.step_id` → `Sample.policy_generation` (stays `str`).
   - `[x]` Update `gemini_msrl.sample()` and `test_gemini_msrl.py`.
   - `[x]` `Recipe.open` → `Recipe.create` (commit `f16b330`).
2. **DB schema**
   - `[x]` `ChatCompletionModel.policy_generation` (uncomment + rename).
   - `[x]` `RunModel` (new, absorbs `RewardModel`).
   - `[x]` `DatumRowModel` (new, table `datum_rows`).
   - `[x]` Drop `RewardModel`.
3. **`Tuner` hooks**
   - `[x]` Wire-types: `DispenseContext` and `RunAssignment` dataclasses
     live in `cookbook/types.py`; the abstract `Op[T]` base with `wait()`
     and `peek()` exists. `TrainOp` / `SampleOp` subclass `Op`.
   - `[x]` `Op.peek()` implemented on `GeminiMsrlOp` (and inherited by
     `GeminiMsrlTrainingOp` / `GeminiMsrlSamplingOp`). LRO `GetOperation`,
     terminal-cached. Covered by `test_sample_op_peek` /
     `test_train_op_peek`.
   - `[x]` `dispense_run(ctx) -> Optional[RunAssignment]` with a sane
     default implementation in the base class (returns `None` while
     `ctx.is_training`, otherwise mints `run_id` and picks a `datum_id`
     from `ctx.datum_pool`).
   - `[x]` `in_flight_train_op() -> Optional[TrainOp]` with a default
     of `None` (recipes opt in for restart-safe training). For
     `gemini_msrl`, wrap `state.last_train_op` (already persisted) in
     a fresh `GeminiMsrlTrainingOp(self.client, self.state.last_train_op)`.
4. **`StateStore` plumbing**
   - `[x]` `StateStore` Protocol in `cookbook/types.py`.
   - `[x]` `_DbStateStore` backing `tuners.state` via UPDATE inside
     `TunerService`.
   - `[x]` `Cookbook.open(kind, name, state_store)` delegates to
     `Recipe.create`.
   - `[x]` `GeminiMsrlTuner._persist_state` + initial save in
     `Recipe.create`'s bootstrap branch.
   - `[x]` `GeminiMsrlRecipeState.last_train_op: Optional[str]` (commit
     `8572371`), stamped inside `train_step(...)` before returning.
5. **`TunerService`**
   - `[x]` `create_tuner(recipe, name, datum_ids)` persists the datum
     pool into `datum_rows`.
   - `[x]` `record_chat_completion(..., policy_generation)` stamps the
     new column.
   - `[x]` `collect_rollout_ready_for_training` reads from `runs` with
     `train_count <= TARGET_MAX_TRAIN_COUNT AND reward IS NOT NULL`;
     in-memory `Rollout` / `RolloutRun` shape preserved.
   - `[x]` `train` bumps `RunModel.train_count` after `train_step`.
   - `[x]` `update_reward(tuner_id, run_id, reward)`: 404 on missing,
     409 on already-rewarded (`RewardAlreadySetError`), 409 on expired
     (`RunExpiredError`).
   - `[x]` Rename `collect_rollout_ready_for_training` →
     `_collect_consumable_batch` (private; called only from
     `maybe_train`) and return `run_ids` alongside the batch so the
     `train_count` UPDATE can run inside the FOR-UPDATE transaction.
     Take the session as a parameter.
   - `[x]` `dispense_run(tuner_id)`: load datum pool from `datum_rows`,
     materialize tuner, derive `is_training` via
     `(await tuner.in_flight_train_op()).peek()`, call
     `tuner.dispense_run(DispenseContext(...))`, `INSERT` into `runs`
     with `expires_at = NOW() + run_ttl`.
   - `[x]` `is_training(tuner_id)` computed by
     `(await tuner.in_flight_train_op()).peek()` — no in-memory latch.
   - `[x]` `maybe_train` wraps `train_step` + `train_count += 1` in a
     `SELECT … FOR UPDATE` transaction on `tuners` (§4.9 / §4.12).
     `op.wait()` is **not** called under the lock. Replaces today's
     fire-and-forget `train` task; rename `train` → `maybe_train` at
     the call site in `put_reward`.
6. **HTTP**
   - `[x]` Extend `POST /tuners` to require `datum_ids` (rejects empty
     arrays with `400`).
   - `[x]` `POST /openai/v1/chat/completions`: `x_run_id` optional
     header; when present, validate against an existing `runs` row
     (`409` on unknown), derive `datum_id` from the row (server doesn't
     trust the client), stamp the chat completion row with
     `policy_generation` from `Sample`.
   - `[x]` Fix the absent-`x_run_id` path in
     `POST /openai/v1/chat/completions`: `x_datum_id` is currently
     referenced but never bound. Either explicitly skip persistence
     in the run-less path, or require `x_run_id`.
   - `[x]` Remove legacy `POST /tuners/{id}/rewards`.
   - `[x]` `POST /tuners/{id}/runs` (dispense endpoint; `204 +
     Retry-After` when `dispense_run` returns `None`).
   - `[x]` `PUT /tuners/{id}/runs/{run_id}/reward` — replaces the
     dropped `POST /rewards`.
   - `[x]` Swap `services.tuner.train` → `services.tuner.maybe_train`
     in `put_reward` once `maybe_train` lands.
   - `[x]` Optionally extend `GET /tuners/{id}` for observability
     (`training`, `pending_run_count`, `last_recorded_policy_generation`).
7. **Docs**
   - `[x]` `.agents/skills/dev/references/sync-rl.md`: rewrite the
     barrier section around the queue.
   - `[x]` `.agents/skills/dev/references/data-model.md`: add
     `RunModel`, `DatumRowModel`, `policy_generation`; remove
     `RewardModel`.
8. **Tests** (after sign-off, per AGENTS.md)
   - Run lifecycle: dispense → completion → reward → `train_count`
     bumped after training.
   - Barrier: while `is_training`, `POST /runs` returns `204`.
   - Multi-worker dispatch: concurrent `POST /runs` never share a
     `run_id`.
   - `policy_generation` from `Sample` round-trips through DB
     (already partially covered by `test_gemini_msrl.py`).
   - `chat/completions` with an unknown `run_id` returns `409`.
   - `chat/completions` without `run_id` either explicitly succeeds
     and writes nothing, or is rejected — pick one and test it.
   - Duplicate `PUT /reward` returns `409` (atomic guard).
   - Multi-completion runs: more than one `ChatCompletionModel` row
     per `RunModel`, each carrying its own `policy_generation`.
   - State checkpoint: after `train_step` returns and `train_count` is
     bumped, `TunerModel.state` contains the recipe's in-flight op id
     (today: via `state_store.save` inside `train_step`; target: same,
     but now also inside the FOR-UPDATE transaction).
   - `is_training` is DB-derived: a fresh process that has never called
     `train_step` correctly reports `True` while a previously committed
     LRO is still running, and `False` after `peek()` flips.
   - Multi-driver safety: two concurrent `maybe_train` calls produce
     exactly one new train op (the loser sees the in-flight op via
     `peek()` and bails).
   - `train_count` gate: a run with `train_count > 0` is never
     re-included in a subsequent batch.
   - Lease expiration: `PUT /reward` against a run whose
     `expires_at <= NOW()` returns `409`; the dispenser is free to
     re-issue that `datum_id` with a fresh `run_id`.
   - `train_count` vs. `expires_at` are independent: a rewarded run
     with `train_count == 0` stays consumable for training even after
     `expires_at` has passed (the lease guard is only consulted on the
     rollout side, never on the training side).

### 7.2 Deferred

9. Streamed datum-pool extension after creation.
10. Server-side curriculum policies as first-class primitives.
11. Off-policy correction (importance ratios) inside `train_step`,
    keyed on `policy_generation`.
12. Background hard-deletion of expired-and-unrewarded `runs` rows
    (`WHERE expires_at <= NOW() AND reward IS NULL`).
13. Per-recipe-tunable `run_ttl` and `Retry-After` via hparams.

---

## 8. Open questions

| #  | Question                                                                                                | Default if no answer                                                          |
|----|---------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| Q1 | `Retry-After` on the 204 — fixed (e.g. `1s`) or recipe-tunable?                                         | Fixed `1` in v1; recipe-tunable later via hparams.                            |
| Q2 | How do we mark a run as "consumed by training"? | **shipped:** `RunModel.train_count: int`; ready iff `train_count == 0`; bumped to 1 after train. Promote to `consumed_by_policy_generation` later if off-policy work needs more than a counter. |
| Q3 | Should `POST /tuners` reject a *too-small* `datum_ids` array (e.g. < expected batch size)?              | Warn but accept; the recipe can always cycle. (Today: accepts as long as non-empty.) |
| Q4 | Is `dispense_run` sync or async on the `Tuner`?                                                          | Sync; the DB layer can be async around it.                                    |
| Q5 | Does `PUT /reward` accept reward updates after a run is already consumed?                               | **shipped:** No — `RewardAlreadySetError` → `409 Conflict`; rewards are write-once. |
| Q6 | Should `GET /tuners/{id}` expose `training: bool` even though it's derived from a `peek()` poll?         | Yes; same DB-derived value `POST /runs` uses, just exposed for debugging.     |
| Q7 | Should the recipe also checkpoint state after non-train events (e.g. after every reward) so opaque recipe counters stay fresh? | No in v1 — `train_step` boundaries are the only point where state actually changes for current recipes. |
| Q8 | Default `run_ttl` for `runs.expires_at`?                                                                | `5 min` in v1; recipe-tunable later via hparams.                              |
| Q9 | Does `peek()` need a result cache inside `TunerService` to avoid hammering the LRO backend on hot `POST /runs` paths? | Not in v1 — Vertex AI's `GetOperation` is cheap and itself caches the terminal state. Revisit if `peek` shows up in profiling. |
| Q10| Should `POST /openai/v1/chat/completions` require `x_run_id`, or keep the fire-and-forget path?         | Pick one before declaring §4.8 done — the current handler has a dangling reference on the run-less branch. Recommend: keep optional but explicitly skip persistence when absent. |

---

## 9. Non-goals

- **Server-owned corpora.** The server holds opaque `datum_id`s, not
  dataset rows. Clients still own data.
- **Per-rollout heartbeats.** Crash recovery for in-flight runs is
  TTL-based (`runs.expires_at`, §4.3), not heartbeat-based. A client
  that crashes mid-completion just lets its run expire.
- **Streamed lease delivery (WebSocket / SSE).** Polling-with-204 is
  simpler and matches the OpenAI-shaped completions endpoint right next
  door.
- **A separate `RewardModel`.** Reward is a column on `RunModel`, not
  its own entity.
- **Streaming corpus updates.** `POST /tuners/{id}/datum_pool` is
  intentionally not in this iteration.
- **Hard-deletion of expired rows.** A `WHERE expires_at > NOW()`
  predicate on the consumable-runs query is enough; a janitor job can
  come later.
- **`TunerService` as single state writer.** An earlier draft of this
  doc made the service own state persistence; the shipped `StateStore`
  Protocol cleanly inverts that: recipes own cadence, the service owns
  only the DB-backed implementation.
