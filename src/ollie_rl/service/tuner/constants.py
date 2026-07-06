"""Tuning-run lifecycle constants shared across the tuner service modules."""

# Compute-based signal for the `expired` (vs `lost`) classification: a
# run that has accumulated at least this much total generation time (the summed
# `duration_ms` across its recorded completions) without ever earning a reward
# is treated as `expired` even if no in-flight op lingers. It burned real
# compute yet never finished -- the same waste the `expired` label flags -- so it
# should not be dismissed as merely `lost`. Measured in milliseconds.
RUN_EXPIRE_GENERATION_BUDGET_MS = 15 * 60 * 1000

# Time budget (seconds) granted to a run's lease. A run is dispensed with a
# deadline of `now + RUN_LEASE_SECONDS`, and every recorded completion extends
# the deadline to `RUN_LEASE_SECONDS` from that completion's time. This keeps an
# actively progressing multi-turn run alive turn-by-turn while still expiring
# runs whose generation genuinely stalled or was abandoned.
RUN_LEASE_SECONDS = 20 * 60
