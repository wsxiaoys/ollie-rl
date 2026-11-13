import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "@tanstack/react-router";
import { useState } from "react";
import type { ChatCompletionPayload, RunDetailResponse } from "../api/types";
import { runQuery, runTranscriptQuery } from "../api/queries";
import { ChatTranscript } from "../components/ChatTranscript";
import { RunStatusBadge } from "../components/RunStatusBadge";
import { Mono, Panel, StatCard } from "../components/ui";

/**
 * Format a millisecond duration into a compact, human-readable string:
 * sub-second stays in `ms`, anything longer rolls up to seconds.
 */
function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(ms < 10_000 ? 2 : 1)}s`;
}

/** Format token counts compactly for the dense completions table. */
function formatTokens(tokens: number | null | undefined): string {
  if (typeof tokens !== "number") return "—";
  if (tokens < 1000) return tokens.toLocaleString();
  return `${(tokens / 1000).toFixed(tokens < 10_000 ? 1 : 0)}k`;
}

/**
 * Compact, payload-free view of a run's completions: one row per completion
 * with its summary fields and a link to the full completion detail. This is
 * what opening a run transfers by default — the transcript's per-turn
 * request/response payloads (which grow ~quadratically with turn count) are
 * fetched only when the user asks for them.
 */
function CompletionsTable({
  completions,
  tunerId,
  runId,
}: {
  completions: RunDetailResponse["completions"];
  tunerId: string;
  runId: string;
}) {
  if (completions.length === 0) {
    return (
      <div className="placeholder placeholder--inset">
        No chat completions recorded for this run yet.
      </div>
    );
  }

  return (
    <div className="table-scroll">
      <table className="table table--dense">
        <thead>
          <tr>
            <th>Completion ID</th>
            <th className="num">Gen</th>
            <th className="num">Messages</th>
            <th>Finish</th>
            <th className="num">In</th>
            <th className="num">Out</th>
            <th className="num">Reason</th>
            <th className="num">Duration</th>
            <th>Created</th>
          </tr>
        </thead>
        <tbody>
          {completions.map((c) => (
            <tr key={c.id}>
              <td>
                <Link
                  to="/tuners/$tunerId/runs/$runId/completions/$completionId"
                  params={{ tunerId, runId, completionId: c.id }}
                  className="link-plain"
                >
                  <Mono>{c.id}</Mono>
                </Link>
              </td>
              <td className="num">{c.policy_generation}</td>
              <td className="num">{c.message_count ?? "—"}</td>
              <td>{c.finish_reason ?? "—"}</td>
              <td className="num">{formatTokens(c.prompt_tokens)}</td>
              <td className="num">{formatTokens(c.completion_tokens)}</td>
              <td className="num">{formatTokens(c.reasoning_tokens)}</td>
              <td className="num">
                {typeof c.duration_ms === "number"
                  ? formatDuration(c.duration_ms)
                  : "—"}
              </td>
              <td>{new Date(c.created_at).toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function RunDetailPage() {
  const { tunerId, runId } = useParams({
    from: "/tuners/$tunerId/runs/$runId",
  });
  const { data, isLoading, isError, error, isFetching } = useQuery(
    runQuery(tunerId, runId),
  );
  const [showTranscript, setShowTranscript] = useState(false);
  const [interestingOnly, setInterestingOnly] = useState(true);

  // The full transcript is fetched only once the user asks for it, so simply
  // opening a run never transfers the per-turn request/response payloads.
  const transcriptQ = useQuery({
    ...runTranscriptQuery(tunerId, runId),
    enabled: showTranscript,
  });

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
          <div className="panel__controls">
            {showTranscript && (
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={interestingOnly}
                  onChange={(e) => setInterestingOnly(e.target.checked)}
                />
                Show interesting trajectories only
              </label>
            )}
            <label className="toggle">
              <input
                type="checkbox"
                checked={showTranscript}
                onChange={(e) => setShowTranscript(e.target.checked)}
              />
              Load full transcript
            </label>
          </div>
        }
      >
        {!showTranscript ? (
          <CompletionsTable
            completions={completions}
            tunerId={tunerId}
            runId={runId}
          />
        ) : transcriptQ.data ? (
          <ChatTranscript
            // `include_payloads=true` guarantees request/response are present.
            completions={
              transcriptQ.data.completions as ChatCompletionPayload[]
            }
            interestingOnly={interestingOnly}
            tunerId={tunerId}
            runId={runId}
          />
        ) : transcriptQ.isError ? (
          <div className="placeholder placeholder--error">
            Failed to load transcript: {(transcriptQ.error as Error).message}
          </div>
        ) : (
          <div className="placeholder placeholder--inset">
            Loading full transcript…
          </div>
        )}
      </Panel>
    </div>
  );
}
