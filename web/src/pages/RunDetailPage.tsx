import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "@tanstack/react-router";
import { useState } from "react";
import { runQuery } from "../api/queries";
import { ChatTranscript } from "../components/ChatTranscript";
import { RunStatusBadge } from "../components/RunStatusBadge";
import { Mono, Panel, StatCard } from "../components/ui";

export function RunDetailPage() {
  const { tunerId, runId } = useParams({
    from: "/tuners/$tunerId/runs/$runId",
  });
  const { data, isLoading, isError, error, isFetching } = useQuery(
    runQuery(tunerId, runId),
  );
  const [interestingOnly, setInterestingOnly] = useState(true);

  if (isLoading) {
    return <div className="placeholder">Loading run…</div>;
  }
  if (isError) {
    return (
      <div className="placeholder placeholder--error">
        Failed to load run: {(error as Error).message}
      </div>
    );
  }
  if (!data) return null;

  const { run, completions } = data;

  return (
    <div className="page">
      <header className="page__header">
        <Link
          to="/runs"
          search={{ tuner: tunerId }}
          className="link link--back"
        >
          ← Runs
        </Link>
        <h1>Run detail</h1>
        <div className="detail-header__meta">
          <Mono>{run.run_id}</Mono>
          <RunStatusBadge status={run.status} />
          {isFetching && <span className="live-dot">● live</span>}
        </div>
      </header>

      <div className="kpi-strip kpi-strip--runs">
        <StatCard label="datum" value={<Mono>{run.datum_id}</Mono>} />
        <StatCard
          label="reward"
          value={run.reward === null ? "—" : run.reward.toFixed(3)}
          tone={run.reward === null ? "muted" : "good"}
        />
        <StatCard label="completions" value={run.completion_count} />
        <StatCard
          label="duration"
          value={
            typeof run.duration_ms_total === "number"
              ? `${(run.duration_ms_total / 1000).toFixed(2)}s`
              : "—"
          }
          tone={typeof run.duration_ms_total === "number" ? "default" : "muted"}
          title="Sum of individual chat completion generation latencies — not the run's wall-clock duration."
        />
        <StatCard
          label="context"
          value={
            typeof run.context_window_tokens_max === "number"
              ? run.context_window_tokens_max.toLocaleString()
              : "—"
          }
          tone={
            typeof run.context_window_tokens_max === "number" ? "default" : "muted"
          }
          title="Max prompt + completion + reasoning tokens across this run's chat completions."
        />
        <StatCard label="trained" value={run.trained_count} />
        <StatCard
          label="rejected"
          value={run.rejected_count}
          tone={run.rejected_count > 0 ? "warn" : "default"}
        />
      </div>

      <Panel
        title="Chat completions"
        right={
          <label className="toggle">
            <input
              type="checkbox"
              checked={interestingOnly}
              onChange={(e) => setInterestingOnly(e.target.checked)}
            />
            Show interesting trajectories only
          </label>
        }
      >
        <ChatTranscript
          completions={completions}
          interestingOnly={interestingOnly}
          tunerId={tunerId}
          runId={runId}
        />
      </Panel>
    </div>
  );
}
