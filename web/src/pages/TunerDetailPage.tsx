import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "@tanstack/react-router";
import { tunerQuery } from "../api/queries";
import type { NextPickTier } from "../api/types";
import { DatumTable } from "../components/DatumTable";
import { Badge, Mono, Panel, ProgressBar, StatCard } from "../components/ui";

const TIER_TONE: Record<
  NextPickTier,
  "good" | "warn" | "info" | "default"
> = {
  incomplete: "warn",
  fresh: "info",
  saturated: "good",
  none: "default",
};

export function TunerDetailPage() {
  const { tunerId } = useParams({ from: "/tuners/$tunerId" });
  const { data, isLoading, isError, error, isFetching } = useQuery(
    tunerQuery(tunerId),
  );

  if (isLoading) {
    return <div className="placeholder">Loading tuner…</div>;
  }
  if (isError) {
    return (
      <div className="placeholder placeholder--error">
        Failed to load tuner: {(error as Error).message}
      </div>
    );
  }
  if (!data) return null;

  const { recipe, progress } = data;
  const isTraining = data.is_training;

  return (
    <div className="page">
      <header className="page__header detail-header">
        <div>
          <Link to="/tuners" className="link link--back">
            ← Tuners
          </Link>
          <h1>{data.name}</h1>
          <div className="detail-header__meta">
            <Mono>{data.tuner_id}</Mono>
            <Badge tone="info">{data.trainer}</Badge>
            <Badge tone="good">
              gen {data.policy_generation}
              {isTraining && (
                <span className="badge-training">
                  <span className="training-now__dot" />
                  training…
                </span>
              )}
            </Badge>
            {isFetching && <span className="live-dot">● live</span>}
          </div>
        </div>
        <div className="recipe-chips">
          <span className="chip">group_size {recipe.group_size}</span>
          <span className="chip">groups/batch {recipe.num_groups_per_batch}</span>
          <span className="chip">
            off-policy ≤ {recipe.max_off_policy_generation}
          </span>
          <span className="chip">malformed {recipe.malformed_penalty}</span>
        </div>
      </header>

      {!progress && (
        <div className="placeholder">
          No progress snapshot returned for this tuner.
        </div>
      )}

      {progress && (
        <>
          <div className="kpi-strip">
            <StatCard label="total runs" value={progress.runs.total} />
            <StatCard
              label="in flight"
              value={progress.runs.in_flight}
              tone="muted"
            />
            <StatCard label="rewarded" value={progress.runs.rewarded} />
            <StatCard
              label="consumable"
              value={progress.runs.consumable}
              tone="good"
            />
            <StatCard label="trained" value={progress.runs.trained} />
            <StatCard
              label="expired"
              value={progress.runs.expired}
              tone="warn"
            />
            <StatCard
              label="rejected"
              value={progress.runs.rejected}
              tone="warn"
            />
          </div>

          <div className="grid-2">
            <Panel title="Batch readiness">
              <div className="batch-readiness">
                <div className="batch-readiness__numbers">
                  <span className="big">{progress.batch.groups_ready}</span>
                  <span className="muted">
                    / {recipe.num_groups_per_batch} groups ready
                  </span>
                </div>
                <ProgressBar
                  value={progress.batch.groups_ready}
                  max={recipe.num_groups_per_batch}
                  tone="good"
                />
                <div className="batch-readiness__sub">
                  <span>
                    {progress.batch.groups_in_progress} groups in progress
                  </span>
                </div>
                <hr className="divider" />
                <div className="coverage">
                  <div>
                    <span className="big">
                      {progress.data.coverage.in_progress}
                    </span>
                    <span className="muted">datums in progress</span>
                  </div>
                  <div>
                    <span className="big">
                      {progress.data.coverage.never_trained}
                    </span>
                    <span className="muted">never trained</span>
                  </div>
                </div>
              </div>
            </Panel>

            <Panel title="Next pick">
              <div className="next-pick">
                <Badge tone={TIER_TONE[progress.next_pick.tier]}>
                  {progress.next_pick.tier}
                </Badge>
                <div className="next-pick__datum">
                  {progress.next_pick.datum_id ? (
                    <Mono>{progress.next_pick.datum_id}</Mono>
                  ) : (
                    <span className="muted">— nothing to dispense —</span>
                  )}
                </div>
                <p className="next-pick__reason">{progress.next_pick.reason}</p>
              </div>
            </Panel>
          </div>

          <Panel
            title="Datum pool"
            right={
              <span className="muted">
                {progress.data.items.length} active datums
              </span>
            }
          >
            <DatumTable
              items={progress.data.items}
              groupSize={recipe.group_size}
            />
          </Panel>
        </>
      )}
    </div>
  );
}
