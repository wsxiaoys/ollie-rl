# Refactor: Splitting `cookbook` into `trainer` + `cookbook`

## Naming

The pre-refactor `Tuner` ABC collides with the user-facing "tuner"
concept exposed by the server (`TunerModel`, `TunerService`, `/tuners`,
`tuner_id`). After this refactor:

| Layer            | Name                                                              | Meaning                                                        |
| ---------------- | ----------------------------------------------------------------- | -------------------------------------------------------------- |
| Backend driver   | `Trainer`                                                         | Live execution handle: `sample`, `train_step`, …               |
| Backend factory  | `TrainerFactory`                                                  | Opens a `Trainer` against a `StateStore`                       |
| Backend package  | `ollie_rl.trainer`                                                | Holds `Trainer`, `TrainerFactory`, concrete backends           |
| User-facing      | `TunerModel` / `TunerService` / `/tuners` / `tuner_id`            | The persisted aggregate the API talks about                    |

`Tuner` (server domain, the thing users create at `/tuners`) and
`Trainer` (backend driver, the thing that talks to Gemini MSRL / TRL /
…) are now lexically distinct everywhere.

## Motivation

The current `ollie_rl.cookbook` package tangles two unrelated concerns:

1. **Backend abstraction** — how to talk to a particular training stack
   (Gemini MSRL, in the future Tinker, vLLM+TRL, etc.). This is *behavioral*
   and async: `sample`, `train_step`, `in_flight_train_op`, plus a
   `StateStore` for persistence.
2. **Rollout / scheduling strategy** — how the RL loop drives the backend:
   on-policy gating, GRPO-style group formation, run lease duration,
   datum scheduling. This is *declarative* and algorithm-shaped.

Today both live on `Recipe`/`Tuner`, which forces every new backend
(`GeminiMsrlTuner`, future `TinkerTuner`, …) to also re-implement
scheduling logic that isn't backend-specific. Concretely,
`GeminiMsrlTuner.dispense_run` is just generic "pick the datum with
min(completed+in_flight) runs" — nothing about it is Gemini-specific —
and `TunerService` hard-codes `GROUP_SIZE`, `TARGET_GROUP_COUNT`, and the
on-policy gate, all of which are properly algorithm choices.

This refactor splits the two concerns into two **fully independent**
packages. `TunerService` is the single consumer that joins them.

## Target package layout

```
src/ollie_rl/
├── trainer/                     # NEW — backend abstraction layer
│   ├── __init__.py              # exports Trainer, TrainerFactory, StateStore, …
│   ├── types.py                 # Trainer ABC, Op/TrainOp/SampleOp, StateStore, Example, Sample
│   ├── factory.py               # TrainerFactory registry
│   └── gemini_msrl.py           # GeminiMsrlTrainer + GeminiMsrlTrainerFactory
│
├── cookbook/                    # CHANGED — pure rollout/scheduling strategy (data only)
│   ├── __init__.py              # exports Recipe, Cookbook
│   └── recipes.py               # Recipe + named instances (GRPO_16x32, …)
│
├── service/
│   └── tuner_service.py         # consumes both packages directly; joins them
└── …
```

## Conceptual model

```
Recipe                = scheduling policy        (data only: how the RL loop schedules itself)
TrainerFactory        = backend factory          (data + behavior: how to open a Trainer)

TunerService          = (Recipe × TrainerFactory) → orchestrates DB, dispatch, batches
```

Crucially: **a `Recipe` does not know what backend it runs on, and a
`TrainerFactory` does not know any scheduling policy.** The pairing
happens inside `TunerService` at tuner-creation and tuner-materialization
time. There is no extra "cookbook service" layer — `TunerService` is the
join point.

### `Recipe` naming note

The pre-refactor `Recipe` was an ABC that bundled scheduling +
backend-creation behavior. The post-refactor `Recipe` is a *frozen
pydantic value type* that holds only scheduling knobs — i.e. what an
earlier draft of this doc called `RecipeStrategy`. Since there is no
longer any *other* thing called "recipe" in the system, the value type
just takes the name `Recipe` directly. There is no separate
`RecipeStrategy` class.

| Concept            | Package                    | Responsibility                                                                       |
| ------------------ | -------------------------- | ------------------------------------------------------------------------------------ |
| `Trainer`          | `ollie_rl.trainer`         | `sample`, `train_step`, `in_flight_train_op`; persistence via `StateStore`           |
| `TrainerFactory`   | `ollie_rl.trainer`         | Async constructor that opens a `Trainer` against a `StateStore`. No scheduling opinions|
| `StateStore`       | `ollie_rl.trainer`         | Opaque-blob persistence protocol owned by the `Trainer`                              |
| `Recipe`           | `ollie_rl.cookbook`        | Declarative scheduling knobs (on_policy, group_size, lease_seconds, …)               |
| `Cookbook`         | `ollie_rl.cookbook`        | Registry / lookup of named `Recipe`s                                                 |
| `TunerService`     | `ollie_rl.service`         | Orchestrates DB rows, dispatch, batch formation; **joins** Recipe × TrainerFactory   |

`trainer` and `cookbook` have **no imports between them**. They both
expose pure values/ABCs. `TunerService` is the single consumer that
imports from both.

---

## Package 1: `ollie_rl.trainer`

### Purpose

Abstract over training backends. A `Trainer` is the minimal async
surface the rest of the system needs to drive a training job,
regardless of whether it's Gemini MSRL, a local TRL+vLLM stack, or any
future provider. **No scheduling concepts appear in this package.**

### `ollie_rl/trainer/types.py`

```python
from abc import ABC, abstractmethod
from typing import Generic, List, Optional, Protocol, TypeVar
from pydantic import BaseModel
from openai.types.chat import ChatCompletion

from ollie_rl.types import ChatCompletionRequest

T = TypeVar("T")


class Example(BaseModel):
    chat_completion_id: str
    advantage: float


class Sample(BaseModel):
    completion: ChatCompletion
    policy_generation: str


class StateStore(Protocol):
    """
    Bi-directional opaque-blob persistence handle owned by a Trainer.
    Read-your-writes semantics required.
    """
    async def load(self) -> Optional[str]: ...
    async def save(self, state: str) -> None: ...


class Op(ABC, Generic[T]):
    @abstractmethod
    async def wait(self) -> T: ...
    @abstractmethod
    async def peek(self) -> bool: ...


class TrainOp(Op[None]):
    pass


class SampleOp(Op[Sample]):
    pass


class Trainer(ABC):
    """
    A single, live training job against some backend.

    The Trainer owns its own persistence cadence via its StateStore.
    """

    @property
    @abstractmethod
    def kind(self) -> str: ...

    @abstractmethod
    async def sample(self, request: ChatCompletionRequest) -> SampleOp: ...

    @abstractmethod
    async def train_step(self, examples: List[Example]) -> TrainOp: ...

    async def in_flight_train_op(self) -> Optional[TrainOp]:
        return None


class TrainerFactory(ABC):
    """
    Async factory that bootstraps or restores a Trainer against a StateStore.

    Has no knowledge of recipes or scheduling. May accept backend-specific
    bootstrap kwargs (base_model, adapter_size, …) via `open(...)`.
    """

    @property
    @abstractmethod
    def kind(self) -> str:
        """Stable identifier (e.g. 'gemini_msrl')."""

    @abstractmethod
    async def open(
        self,
        name: str,
        state_store: StateStore,
        **bootstrap,
    ) -> Trainer: ...
```

### `ollie_rl/trainer/factory.py`

Self-registration registry:

```python
from typing import Dict
from .types import TrainerFactory

_REGISTRY: Dict[str, TrainerFactory] = {}

def register(factory: TrainerFactory) -> None: ...
def get(kind: str) -> TrainerFactory: ...
def available() -> list[str]: ...
```

### `ollie_rl/trainer/gemini_msrl.py`

Moves verbatim from `cookbook/gemini_msrl.py` with these diffs:

- Imports come from `ollie_rl.trainer.types`.
- `GeminiMsrlRecipe` → `GeminiMsrlTrainerFactory` (implements `TrainerFactory`).
- `GeminiMsrlTuner` → `GeminiMsrlTrainer`.
- `GeminiMsrlTuner.dispense_run` is **deleted**.
- At module bottom: `factory.register(GeminiMsrlTrainerFactory())`.

---

## Package 2: `ollie_rl.cookbook` (data-only)

### Purpose

Manage rollout strategy. A `Recipe` is a frozen pydantic value holding
the declarative scheduling knobs. The package never imports anything
from `ollie_rl.trainer`; it has no awareness of which backend executes
the loop.

### `ollie_rl/cookbook/recipes.py`

```python
from typing import Literal
from pydantic import BaseModel

Scheduler = Literal["fifo_epoch", "random"]


class Recipe(BaseModel, frozen=True):
    """
    Declarative algorithm-level knobs the TunerService needs to schedule
    runs and form training batches. Pure data; knows nothing about backends.
    """

    # ---- Policy gating --------------------------------------------------
    on_policy: bool = True
    max_trained_count: int = 0

    # ---- Batch formation (GRPO-style grouping) --------------------------
    group_size: int = 16
    target_group_count: int = 32

    # ---- Run lifecycle --------------------------------------------------
    lease_seconds: int = 7200

    # ---- Scheduling -----------------------------------------------------
    scheduler: Scheduler = "fifo_epoch"


# ---- Named recipe instances --------------------------------------------

GRPO_16x32 = Recipe(
    on_policy=True,
    group_size=16,
    target_group_count=32,
    max_trained_count=0,
    lease_seconds=7200,
    scheduler="fifo_epoch",
)
```

The name describes the *RL loop shape*, not the backend. The same
`GRPO_16x32` recipe can be paired with `gemini_msrl`, a future `tinker`
backend, etc. by `TunerService` at call time.

### `ollie_rl/cookbook/__init__.py`

```python
from typing import Dict

from .recipes import Recipe, GRPO_16x32

RECIPES: Dict[str, Recipe] = {
    "grpo_16x32": GRPO_16x32,
}


class Cookbook:
    """Lookup of named recipes."""

    @classmethod
    def get(cls, name: str) -> Recipe:
        recipe = RECIPES.get(name)
        if recipe is None:
            raise ValueError(
                f"Recipe '{name}' not found. Available: {list(RECIPES)}"
            )
        return recipe


__all__ = ["Cookbook", "Recipe"]
```

The cookbook package is **import-cycle-free** with respect to
`ollie_rl.trainer`: no `TrainerFactory`, no `Trainer`, no backend SDK.
It can be imported in tests, notebooks, or admin tools without dragging
in Gemini clients.

The name → `Recipe` binding lives in the `RECIPES` registry dict, not
on the `Recipe` value itself. That keeps `Recipe` a pure value and
lets the same `Recipe` be registered under multiple aliases if ever
needed.

---

## `ollie_rl.service.tuner_service` (revised)

`TunerService` is the only place that imports from both
`ollie_rl.trainer` and `ollie_rl.cookbook`. The join is a few lines —
no intermediate service needed.

### Imports

```python
from ollie_rl.cookbook import Cookbook, Recipe
from ollie_rl.trainer import Trainer, StateStore
from ollie_rl.trainer import factory as trainer_factory
```

### Persistence

`TunerModel` carries enough info to re-materialize:

- `recipe_name: str`        — which recipe (RL loop shape)
- `trainer_kind: str`       — which backend
- `bootstrap_json: str`     — backend kwargs (frozen at create time)
- `recipe_json: str`        — frozen snapshot of the `Recipe` value for
                              reproducibility (optional but recommended)

(Today `TunerModel.kind` doubles as both. Migrate to two columns;
retain `kind` as an alias for `trainer_kind` for one release.)

### `create_tuner`

```python
async def create_tuner(
    self,
    recipe_name: str,
    trainer_kind: str,
    name: str,
    datum_ids: List[str],
    bootstrap: Optional[Dict[str, Any]] = None,
) -> str:
    recipe = Cookbook.get(recipe_name)
    factory = trainer_factory.get(trainer_kind)    # validate now, fail fast
    bootstrap = bootstrap or {}

    tuner_id = f"tuner_{uuid.uuid4()}"
    async with self.async_session() as session:
        async with session.begin():
            session.add(TunerModel(
                id=tuner_id,
                name=name,
                recipe_name=recipe_name,
                trainer_kind=trainer_kind,
                bootstrap_json=json.dumps(bootstrap),
                recipe_json=recipe.model_dump_json(),
                state=None,
            ))
            for datum_id in datum_ids:
                session.add(DatumRowModel(tuner_id=tuner_id, datum_id=datum_id))

    state_store = _DbStateStore(tuner_id)
    trainer = await factory.open(name, state_store, **bootstrap)
    self.active_trainers[tuner_id] = trainer
    return tuner_id
```

(`TunerService.active_tuners` → `TunerService.active_trainers`: the
cached values are `Trainer` instances, not tuners.)

### `_materialize`

```python
async def _materialize(self, tuner_id: str, record: TunerModel) -> Trainer:
    factory = trainer_factory.get(record.trainer_kind)
    bootstrap = json.loads(record.bootstrap_json or "{}")
    trainer = await factory.open(record.name, _DbStateStore(tuner_id), **bootstrap)
    self.active_trainers[tuner_id] = trainer
    return trainer
```

### `dispense_run`

```python
async def dispense_run(self, tuner_id: str) -> Optional[RunModel]:
    record = await self._load_record(tuner_id)
    if record is None:
        return None
    recipe = self._recipe_for(record)

    if recipe.on_policy:
        trainer = await self.get(tuner_id)
        op = await trainer.in_flight_train_op() if trainer else None
        if op is not None and not await op.peek():
            return None

    async with self.async_session() as session:
        datum_pool, runs = await self._load_pool_and_runs(tuner_id, session)
        datum_id = self._pick_datum(datum_pool, runs, recipe)
        if datum_id is None:
            return None

        run_record = RunModel(
            id=f"run_{uuid.uuid4()}",
            tuner_id=tuner_id,
            datum_id=datum_id,
            reward=None,
            trained_count=0,
            expires_at=utcnow() + timedelta(seconds=recipe.lease_seconds),
        )
        async with session.begin():
            session.add(run_record)
    return run_record


def _recipe_for(self, record: TunerModel) -> Recipe:
    if record.recipe_json:
        return Recipe.model_validate_json(record.recipe_json)
    return Cookbook.get(record.recipe_name)
```

`_collect_consumable_batch` reads `recipe.group_size`,
`recipe.target_group_count`, `recipe.max_trained_count` instead of
module-level constants.

### `_pick_datum`

```python
def _pick_datum(
    self,
    datum_pool: list[str],
    runs: list[RunModel],
    recipe: Recipe,
) -> Optional[str]:
    if not datum_pool:
        return None
    if recipe.scheduler == "random":
        return random.choice(datum_pool)
    now = utcnow()
    score = {d: 0 for d in datum_pool}
    for r in runs:
        if r.datum_id not in score:
            continue
        if r.reward is not None or r.expires_at > now:
            score[r.datum_id] += 1
    return min(score, key=score.get)
```

---

## Server-side wiring

One `TunerService` instance at startup:

```python
# server/app.py (sketch)
tuner_service = TunerService()

@app.post("/tuners")
async def create_tuner(req: CreateTunerRequest) -> CreateTunerResponse:
    tuner_id = await tuner_service.create_tuner(
        recipe_name=req.recipe,           # e.g. "grpo_16x32"
        trainer_kind=req.trainer,         # e.g. "gemini_msrl"
        name=req.name,
        datum_ids=req.datum_ids,
        bootstrap=req.bootstrap or {},
    )
    return CreateTunerResponse(tuner_id=tuner_id)
```

The API exposes the `recipe`/`trainer` split explicitly. If a
deployment wants to hide that (always-Gemini setups), the server layer
can pin `trainer` to a config default and only expose `recipe`.

---

## Dependency direction

```
ollie_rl.trainer  ──┐
                    ├──> ollie_rl.service.tuner_service ──> server
ollie_rl.cookbook ──┘
```

`TunerService` is the **only** module that imports from both `trainer`
and `cookbook`. That invariant can be enforced with a one-line grep
check in CI.

---

## Migration plan

The split is mechanical and can be staged so the suite stays green.

### Step 1 — Carve out `ollie_rl.trainer`

- `mkdir src/ollie_rl/trainer`
- Move `Tuner` (rename to `Trainer`), `Op`, `TrainOp`, `SampleOp`,
  `Example`, `Sample`, `StateStore` from `cookbook/types.py` to
  `trainer/types.py`.
- Add `TrainerFactory` ABC + `trainer/factory.py` registry.
- Move `gemini_msrl.py` to `trainer/gemini_msrl.py`; rename
  `GeminiMsrlTuner` → `GeminiMsrlTrainer` and
  `GeminiMsrlRecipe` → `GeminiMsrlTrainerFactory`; delete
  `GeminiMsrlTrainer.dispense_run`.
- Keep a `cookbook/types.py` shim re-exporting moved symbols.

### Step 2 — Reduce `cookbook` to a pure `Recipe` value type

- Replace `cookbook/types.py` (the old `Recipe` ABC) with
  `cookbook/recipes.py` defining `Recipe` as a frozen pydantic model
  plus named instances (`GRPO_16x32`).
- Update `cookbook/__init__.py` to expose `Recipe` and `Cookbook.get(name)`.
- Remove `Cookbook.open(...)` and any imports of `ollie_rl.trainer`.

### Step 3 — Migrate `TunerService`

- Import from both `cookbook` (for `Recipe` / `Cookbook`) and `trainer`
  (for `TrainerFactory` / `factory.get`).
- Switch `_materialize` / `create_tuner` to call
  `trainer_factory.get(trainer_kind).open(name, state_store, **bootstrap)`.
- Rename `self.active_tuners` → `self.active_trainers` (now caches
  `Trainer` instances).
- Rewrite `dispense_run` to use `Recipe` + `_pick_datum`.
- Replace hard-coded constants in `_collect_consumable_batch` with
  `recipe.*` lookups.
- Drop `DispenseContext`, `RunAssignment`, `DatumMetric`.

### Step 4 — Schema + API

- Migrate `tuners` table: add `recipe_name`, `trainer_kind`,
  `bootstrap_json`, `recipe_json`. Backfill from existing `kind`.
- Update API request models to take `(recipe, trainer, bootstrap)`.

### Step 5 — Cleanup

- Delete `cookbook/types.py` shim.
- Delete `cookbook/gemini_msrl.py`.
- Update remaining imports across `server/`, `service/`, tests.

---

## What disappears

- `Tuner.dispense_run`, `DispenseContext`, `RunAssignment`, `DatumMetric`
- The old `Tuner` ABC (renamed to `Trainer`, with `dispense_run` removed)
- The old `Recipe` ABC with `create(...)` (replaced by the pydantic value type)
- Any direct import of `ollie_rl.trainer` from `ollie_rl.cookbook` (none allowed)
- Module-level constants `GROUP_SIZE`, `TARGET_GROUP_COUNT`,
  `TARGET_MAX_TRAIN_COUNT` in `tuner_service.py`
- Hard-coded `timedelta(minutes=120)` lease in `dispense_run`
- Hard-coded "block while training" branch (now gated on `recipe.on_policy`)

## What stays

- `Trainer.sample` / `train_step` / `in_flight_train_op` contracts
  (same shape as today's `Tuner.*`)
- `StateStore` protocol and the DB-backed implementation
- All DB models (with the small `tuners` schema addition)
- The global `_train_lock` and `maybe_train` orchestration shape
- The server-side `Tuner*` names: `TunerModel`, `TunerService`,
  `/tuners`, `tuner_id`. Only the *backend driver* is renamed.

---

## Why one layer is sufficient

An earlier draft proposed a separate `CookbookService` to perform the
`(Recipe × TrainerFactory)` join. Collapsing it back into `TunerService`
is the right call because:

1. **`TunerService` is the only consumer of the join.** Nobody else
   needs a "resolved trainer spec." Wrapping a two-line lookup in its
   own service just adds indirection without isolating anything.
2. **The join itself is trivial.** `Cookbook.get(name)` and
   `trainer_factory.get(kind).open(...)` are one line each. There's
   no shared state, no caching, no transactional logic to encapsulate.
3. **YAGNI on the abstraction.** If a second consumer ever appears
   (e.g. a "dry-run / validate-spec" endpoint), promoting the join to
   its own service is a 20-minute extraction. Don't pay that cost
   pre-emptively.

The invariant we actually want is "`cookbook` and `trainer` never
import each other." That's preserved with `TunerService` as the sole
join point.

---

## Open questions

1. **Recipe snapshotting.** Strongly recommended (`recipe_json` on the
   row) for reproducibility. Worth the migration cost?
   Preference: yes.
2. **Custom schedulers per recipe.** Stay enum-only (`fifo_epoch` |
   `random`) until a concrete recipe demands otherwise.
3. **Bootstrap typing.** `bootstrap: Dict[str, Any]` is dynamic. Should
   each `TrainerFactory` declare a typed pydantic model for its
   bootstrap kwargs? Preference: yes, in a follow-up
   (`TrainerFactory.bootstrap_schema -> Type[BaseModel]`).
4. **Default `trainer` at the server.** Should the API require
   `trainer`, or default it from server config? Preference: require
   it explicitly; allow a server-config override for single-backend
   deployments.
