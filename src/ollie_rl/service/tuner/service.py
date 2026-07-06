"""The composed :class:`TunerService`.

``TunerService`` is assembled from focused feature mixins layered on the shared
:class:`~ollie_rl.service.tuner.base.TunerServiceBase` (which holds the trainer
registry, concurrency locks, and shared DB-access helpers). Each mixin owns one
concern:

* :class:`~ollie_rl.service.tuner.training.TrainingMixin` -- background train
  loop and consumable-batch collection.
* :class:`~ollie_rl.service.tuner.sampling.SamplingMixin` -- generation,
  idempotent replay, completion recording, and reward updates.
* :class:`~ollie_rl.service.tuner.lifecycle.LifecycleMixin` -- tuner creation.
* :class:`~ollie_rl.service.tuner.queries.QueryMixin` -- read-only dashboard
  queries over tuners, runs, and completions.
* :class:`~ollie_rl.service.tuner.dispensing.DispenseMixin` -- run dispensing.
"""

from ollie_rl.service.tuner.base import TunerServiceBase
from ollie_rl.service.tuner.dispensing import DispenseMixin
from ollie_rl.service.tuner.lifecycle import LifecycleMixin
from ollie_rl.service.tuner.queries import QueryMixin
from ollie_rl.service.tuner.sampling import SamplingMixin
from ollie_rl.service.tuner.training import TrainingMixin


class TunerService(
    TrainingMixin,
    SamplingMixin,
    LifecycleMixin,
    QueryMixin,
    DispenseMixin,
    TunerServiceBase,
):
    """
    Handles both active in-memory trainers and their persistence to a database.
    Uses SQLAlchemy async engine and sessionmaker from the ollie_rl.db subpackage.
    """
