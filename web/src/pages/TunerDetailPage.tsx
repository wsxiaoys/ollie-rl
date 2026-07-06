import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "@tanstack/react-router";
import { useState } from "react";
import { rewardDistributionQuery, tunerQuery } from "../api/queries";
import type { NextPickTier } from "../api/types";
import { computeQuarantine, DatumTable } from "../components/DatumTable";
import { RewardDistribution } from "../components/RewardDistribution";
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

/** Human-readable train op duration, e.g. 90.4 → "1m 30s", 7.2 → "7.2s". */
function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const mins = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60);
  return `${mins}m ${secs}s`;
}

export function TunerDetailPage() {
  const { tunerId } = useParams({ from: "/tuners/$tunerId" });
  const [hideExcluded, setHideExcluded] = useState(true);
  const { data, isLoading, isError, error, isFetching } = useQuery(
    tunerQuery(tunerId),
  );
  const rewardDistQ = useQuery(rewardDistributionQuery(tunerId));

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
  const lastTrainOpDuration = data.last_train_op_duration_seconds;

  const datumItems = progress?.data.items ?? [];
  const excludedCount = datumItems.filter(
    (item) =>
      computeQuarantine(item, {
        quarantineMinSamples: recipe.quarantine_min_samples,
        maxLengthRatio: recipe.max_length_ratio,
        maxSucceedRatio: recipe.max_succeed_ratio,
      }).quarantined,
  ).length;
  const activeCount = datumItems.length - excludedCount;

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
        <div className="detail-header__aside">
          <Link
            to="/runs"
            search={{ tuner: data.tuner_id }}
            className="link"
          >
            View runs →
          </Link>
          <div className="recipe-chips">
            <span className="chip">group_size {recipe.group_size}</span>
            <span className="chip">
              groups/batch {recipe.num_groups_per_batch}
            </span>
            <span className="chip">
              off-policy ≤ {recipe.max_off_policy_generation}
            </span>
            <span className="chip">
              content_filter {recipe.content_filter_penalty}
            </span>
            <span className="chip">length {recipe.length_penalty}</span>
          </div>
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
            {lastTrainOpDuration != null && (
              <StatCard
                label={`gen ${data.policy_generation} train time`}
                value={formatDuration(lastTrainOpDuration)}
                tone="good"
              />
            )}
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
              label="lost"
              value={progress.runs.lost}
              tone="danger"
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
                <div className="batch-readiness__runs muted">
                  = {progress.batch.groups_ready * recipe.group_size} /{" "}
                  {recipe.num_groups_per_batch * recipe.group_size} runs
                  <span className="batch-readiness__hint">
                    ({recipe.group_size} runs per group)
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
                    <span className="batch-readiness__hint">
                      (≈ {progress.batch.groups_in_progress * recipe.group_size}{" "}
                      runs)
                    </span>
                  </span>
                </div>
                <hr className="divider" />
                <div className="coverage">
                  <div>
                    <span className="big">
                      {progress.data.coverage.in_progress}
                    </span>
                    <span className="muted">data in progress</span>
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
            title="Reward distribution by generation"
            right={
              rewardDistQ.isFetching ? (
                <span className="live-dot">● live</span>
              ) : undefined
            }
          >
            {rewardDistQ.isError ? (
              <div className="placeholder placeholder--inset placeholder--error">
                Failed to load reward distribution:{" "}
                {(rewardDistQ.error as Error).message}
              </div>
            ) : !rewardDistQ.data ? (
              <div className="placeholder placeholder--inset">
                Loading reward distribution…
              </div>
            ) : (
              <RewardDistribution dist={rewardDistQ.data} />
            )}
          </Panel>

          <Panel
            title="Datum pool"
            right={
              <div className="datum-pool-header">
                <span className="muted">
                  {excludedCount > 0 ? (
                    <>
                      {activeCount} active · {excludedCount} excluded
                    </>
                  ) : (
                    <>{activeCount} active data</>
                  )}
                </span>
                <label className="datum-pool-toggle">
                  <input
                    type="checkbox"
                    checked={hideExcluded}
                    onChange={(e) => setHideExcluded(e.target.checked)}
                  />
                  Hide excluded
                </label>
              </div>
            }
          >
            <div className="datum-pool-scrollable">
              <DatumTable
                items={progress.data.items}
                groupSize={recipe.group_size}
                quarantineMinSamples={recipe.quarantine_min_samples}
                maxLengthRatio={recipe.max_length_ratio}
                maxSucceedRatio={recipe.max_succeed_ratio}
                hideExcluded={hideExcluded}
                tunerId={data.tuner_id}
              />
            </div>
          </Panel>
        </>
      )}
    </div>
  );
}
