"""Tuning-run lifecycle constants shared across the tuner service modules."""

# Compute-based signal for the `expired` (vs `lost`) classification: a
# run that has accumulated at least this much total generation time (the summed
# `duration_ms` across its recorded completions) without ever earning a reward
# is treated as `expired` even if no in-flight op lingers. It burned real
# compute yet never finished -- the same waste the `expired` label flags -- so it
# should not be dismissed as merely `lost`. Measured in milliseconds.
RUN_EXPIRE_GENERATION_BUDGET_MS = 2 * 60 * 1000

# Time budget (seconds) granted to a run's lease. A run is dispensed with a
# deadline of `now + RUN_LEASE_SECONDS`. Active HTTP completion requests
# heartbeat the deadline, then start a fresh full window when the request ends.
# A run can therefore expire only after no completion request has been in flight
# for this full interval.
RUN_LEASE_SECONDS = 15 * 60
