import { useInfiniteQuery, useQuery } from "@tanstack/react-query";
import { Link, useNavigate, useSearch } from "@tanstack/react-router";
import { useEffect, useMemo, useRef, type ChangeEvent } from "react";
import { runsPageQuery, tunersQuery } from "../api/queries";
import { RunStatusBadge } from "../components/RunStatusBadge";
import { Mono } from "../components/ui";

export function RunListPage() {
  const { tuner: tunerId } = useSearch({ from: "/runs" });
  const navigate = useNavigate();

  const tunersQ = useQuery(tunersQuery);
  const runsQ = useInfiniteQuery({
    ...runsPageQuery(tunerId ?? ""),
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

  const onSelect = (e: ChangeEvent<HTMLSelectElement>) => {
    const value = e.target.value;
    navigate({ to: "/runs", search: { tuner: value || undefined } });
  };

  return (
    <div className="page">
      <header className="page__header">
        <h1>Runs</h1>
        <p className="page__subtitle">
          Pick a tuner to inspect its runs and their chat completions.
        </p>
      </header>

      <div className="runs-picker">
        <label htmlFor="tuner-select">Tuner</label>
        <select id="tuner-select" value={tunerId ?? ""} onChange={onSelect}>
          <option value="">— select a tuner —</option>
          {tunersQ.data?.tuners.map((t) => (
            <option key={t.tuner_id} value={t.tuner_id}>
              {t.name} ({t.tuner_id})
            </option>
          ))}
        </select>
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
        <div className="placeholder">No runs dispensed yet.</div>
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
              <th>Created</th>
              <th>Expires</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => (
              <tr key={r.run_id}>
                <td>
                  <Link
                    to="/tuners/$tunerId/runs/$runId"
                    params={{ tunerId, runId: r.run_id }}
                    className="link"
                  >
                    <Mono>{r.run_id}</Mono>
                  </Link>
                </td>
                <td>
                  <Mono>{r.datum_id}</Mono>
                </td>
                <td>
                  <RunStatusBadge status={r.status} />
                </td>
                <td className="num">
                  {r.reward === null ? "—" : r.reward.toFixed(3)}
                </td>
                <td className="num">{r.completion_count}</td>
                <td>{new Date(r.created_at).toLocaleString()}</td>
                <td>{new Date(r.expires_at).toLocaleString()}</td>
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
