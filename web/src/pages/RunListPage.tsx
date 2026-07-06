import { useInfiniteQuery, useQuery } from "@tanstack/react-query";
import { Link, useNavigate, useSearch } from "@tanstack/react-router";
import { useEffect, useMemo, useRef } from "react";
import { dataQuery, runsPageQuery, tunersQuery } from "../api/queries";
import { RunStatusBadge } from "../components/RunStatusBadge";
import { SearchableSelect } from "../components/SearchableSelect";
import { Mono } from "../components/ui";

/**
 * Format a millisecond duration into a compact, human-readable string:
 * sub-second stays in `ms`, anything longer rolls up to seconds.
 */
function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(ms < 10_000 ? 2 : 1)}s`;
}

/** Format token counts compactly for dense tables. */
function formatTokens(tokens: number): string {
  if (tokens < 1000) return tokens.toLocaleString();
  return `${(tokens / 1000).toFixed(tokens < 10_000 ? 1 : 0)}k`;
}

export function RunListPage() {
  const { tuner: tunerId, datum: datumId } = useSearch({ from: "/runs" });
  const navigate = useNavigate();

  const tunersQ = useQuery(tunersQuery);
  const dataQ = useQuery({
    ...dataQuery(tunerId ?? ""),
    enabled: Boolean(tunerId),
  });
  const runsQ = useInfiniteQuery({
    ...runsPageQuery(tunerId ?? "", datumId),
    enabled: Boolean(tunerId),
  });

  const runs = useMemo(
    () => runsQ.data?.pages.flatMap((p) => p.runs) ?? [],
    [runsQ.data],
  );

  // Infinite scroll: fetch the next page when the sentinel row near the bottom
  // of the table scrolls into view.
  const { fetchNextPage, hasNextPage, isFetchingNextPage } = runsQ;
  const sentinelRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const el = sentinelRef.current;
    if (!el || !hasNextPage) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0]?.isIntersecting && !isFetchingNextPage) {
          fetchNextPage();
        }
      },
      { rootMargin: "200px" },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [hasNextPage, isFetchingNextPage, fetchNextPage]);

  useEffect(() => {
    if (!tunerId && tunersQ.data?.tuners && tunersQ.data.tuners.length > 0) {
      const firstTunerId = tunersQ.data.tuners[0].tuner_id;
      navigate({ to: "/runs", search: { tuner: firstTunerId } });
    }
  }, [tunerId, tunersQ.data, navigate]);

  const onDatumChange = (value: string | null) => {
    navigate({
      to: "/runs",
      search: { tuner: tunerId, datum: value ?? undefined },
    });
  };

  return (
    <div className="page">
      <header className="page__header">
        <h1>Runs</h1>
        <p className="page__subtitle">
          Runs for the selected tuner and their chat completions.
        </p>
      </header>

      <div className="runs-picker">
        <label htmlFor="datum-filter">Datum ID</label>
        <SearchableSelect
          id="datum-filter"
          value={datumId ?? null}
          options={dataQ.data?.datum_ids ?? []}
          onChange={onDatumChange}
          placeholder="All data"
          searchPlaceholder="Search datum id…"
          disabled={!tunerId}
        />
        {runsQ.isFetching && <span className="live-dot">● live</span>}
      </div>

      {!tunerId && (
        <div className="placeholder">Select a tuner to view its runs.</div>
      )}

      {tunerId && runsQ.isLoading && (
        <div className="placeholder">Loading runs…</div>
      )}
      {tunerId && runsQ.isError && (
        <div className="placeholder placeholder--error">
          Failed to load runs: {(runsQ.error as Error).message}
        </div>
      )}

      {tunerId && runsQ.data && runs.length === 0 && (
        <div className="placeholder">
          {datumId
            ? `No runs found for datum "${datumId}".`
            : "No runs dispensed yet."}
        </div>
      )}

      {tunerId && runsQ.data && runs.length > 0 && (
        <table className="table table--dense">
          <thead>
            <tr>
              <th>Run ID</th>
              <th>Datum ID</th>
              <th>Status</th>
              <th className="num">Reward</th>
              <th className="num">Completions</th>
              <th className="num">Duration</th>
              <th className="num">Context</th>
              <th>Created</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => (
              <tr key={r.run_id}>
                <td>
                  <Link
                    to="/tuners/$tunerId/runs/$runId"
                    params={{ tunerId, runId: r.run_id }}
                    className="link-plain"
                  >
                    <Mono>{r.run_id}</Mono>
                  </Link>
                </td>
                <td>
                  {r.datum_id === datumId ? (
                    <Mono>{r.datum_id}</Mono>
                  ) : (
                    <button
                      type="button"
                      className="link-button"
                      title={`Filter runs by datum "${r.datum_id}"`}
                      onClick={() => onDatumChange(r.datum_id)}
                    >
                      <Mono>{r.datum_id}</Mono>
                    </button>
                  )}
                </td>
                <td>
                  <RunStatusBadge status={r.status} />
                </td>
                <td className="num">
                  {r.reward === null ? "—" : r.reward.toFixed(3)}
                </td>
                <td className="num">{r.completion_count}</td>
                <td
                  className="num"
                  title={
                    typeof r.duration_ms_total === "number"
                      ? `${r.duration_ms_total.toLocaleString()} ms — sum of chat completion generation latencies, not the run's wall-clock duration.`
                      : undefined
                  }
                >
                  {typeof r.duration_ms_total === "number"
                    ? formatDuration(r.duration_ms_total)
                    : "—"}
                </td>
                <td
                  className="num"
                  title={
                    typeof r.context_window_tokens_max === "number"
                      ? `${r.context_window_tokens_max.toLocaleString()} tokens — max prompt + completion + reasoning tokens across this run's chat completions.`
                      : undefined
                  }
                >
                  {typeof r.context_window_tokens_max === "number"
                    ? formatTokens(r.context_window_tokens_max)
                    : "—"}
                </td>
                <td
                  title={`Expires ${new Date(r.expires_at).toLocaleString()}`}
                >
                  {new Date(r.created_at).toLocaleString()}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {tunerId && runs.length > 0 && (
        <>
          {/* Sentinel observed to auto-load the next page on scroll. */}
          <div ref={sentinelRef} aria-hidden className="runs-sentinel" />
          <div className="runs-pager">
            {runsQ.isFetchingNextPage ? (
              <span className="muted">Loading more…</span>
            ) : runsQ.hasNextPage ? (
              <span className="muted">Scroll to load more…</span>
            ) : (
              <span className="muted">End of runs</span>
            )}
          </div>
        </>
      )}
    </div>
  );
}
