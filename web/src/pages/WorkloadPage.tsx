import { useInfiniteQuery } from "@tanstack/react-query";
import { Link, useParams } from "@tanstack/react-router";
import { useEffect, useMemo, useRef } from "react";
import { inFlightChatCompletionsQuery } from "../api/queries";
import { Badge, Mono, Panel, StatCard } from "../components/ui";

function formatAge(timestamp: string): string {
  const seconds = Math.max(
    0,
    Math.floor((Date.now() - new Date(timestamp).getTime()) / 1000),
  );
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ${seconds % 60}s`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${minutes % 60}m`;
}

export function WorkloadPage() {
  const { tunerId } = useParams({ from: "/tuners/$tunerId/workload" });
  const workloadQ = useInfiniteQuery(inFlightChatCompletionsQuery(tunerId));
  const items = useMemo(
    () => workloadQ.data?.pages.flatMap((page) => page.items) ?? [],
    [workloadQ.data],
  );
  const summary = workloadQ.data?.pages[0];

  const { fetchNextPage, hasNextPage, isFetchingNextPage } = workloadQ;
  const sentinelRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const element = sentinelRef.current;
    if (!element || !hasNextPage) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0]?.isIntersecting && !isFetchingNextPage) {
          fetchNextPage();
        }
      },
      { rootMargin: "200px" },
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, [hasNextPage, isFetchingNextPage, fetchNextPage]);

  return (
    <div className="page">
      <header className="page__header">
        <h1>Workload</h1>
        <p className="page__subtitle">
          Resumable chat-completion operations currently tracked for this tuner.
        </p>
      </header>

      {workloadQ.isError ? (
        <div className="placeholder placeholder--error">
          Failed to load workload: {(workloadQ.error as Error).message}
        </div>
      ) : !workloadQ.data ? (
        <div className="placeholder">Loading workload…</div>
      ) : (
        <>
          <div className="kpi-strip">
            <StatCard label="tracked operations" value={summary?.total ?? 0} />
            <StatCard
              label="active lease"
              value={summary?.active_lease_count ?? 0}
              tone="good"
            />
            <StatCard
              label="past lease"
              value={summary?.past_lease_count ?? 0}
              tone={(summary?.past_lease_count ?? 0) > 0 ? "warn" : "muted"}
            />
            <StatCard
              label="oldest operation"
              value={
                summary?.oldest_created_at
                  ? formatAge(summary.oldest_created_at)
                  : "—"
              }
            />
          </div>

          <Panel
            title="Tracked operations"
            right={
              workloadQ.isFetching ? (
                <span className="live-dot">● live</span>
              ) : (
                <span className="muted">refreshes every 2s</span>
              )
            }
          >
            {items.length === 0 ? (
              <div className="placeholder placeholder--inset">
                No resumable chat-completion operations are currently tracked.
              </div>
            ) : (
              <div className="table-scroll">
                <table className="table table--dense">
                  <thead>
                    <tr>
                      <th>Age</th>
                      <th>Run ID</th>
                      <th>Datum ID</th>
                      <th>Split</th>
                      <th>Lease</th>
                      <th className="num">Prior completions</th>
                      <th>Request key</th>
                    </tr>
                  </thead>
                  <tbody>
                    {items.map((item) => (
                      <tr key={`${item.run_id}:${item.request_hash}`}>
                        <td title={new Date(item.created_at).toLocaleString()}>
                          {formatAge(item.created_at)}
                        </td>
                        <td>
                          <Link
                            to="/tuners/$tunerId/runs/$runId"
                            params={{ tunerId, runId: item.run_id }}
                            className="link-plain"
                          >
                            <Mono>{item.run_id}</Mono>
                          </Link>
                        </td>
                        <td>
                          <Link
                            to="/datums"
                            search={{ tuner: tunerId, datum: item.datum_id }}
                            className="link-plain"
                          >
                            <Mono>{item.datum_id}</Mono>
                          </Link>
                        </td>
                        <td>
                          <Badge tone={item.kind === "eval" ? "info" : "default"}>
                            {item.kind === "eval" &&
                            item.checkpoint_generation != null
                              ? `eval · gen ${item.checkpoint_generation}`
                              : item.kind}
                          </Badge>
                        </td>
                        <td title={new Date(item.run_expires_at).toLocaleString()}>
                          <Badge tone={item.lease_expired ? "warn" : "good"}>
                            {item.lease_expired ? "past lease" : "active"}
                          </Badge>
                        </td>
                        <td className="num">
                          {item.recorded_completion_count}
                        </td>
                        <td title={item.request_hash}>
                          <Mono>{item.request_hash.slice(0, 10)}…</Mono>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            {items.length > 0 && (
              <>
                <div ref={sentinelRef} aria-hidden className="runs-sentinel" />
                <div className="runs-pager">
                  {workloadQ.isFetchingNextPage ? (
                    <span className="muted">Loading more…</span>
                  ) : workloadQ.hasNextPage ? (
                    <span className="muted">Scroll to load more…</span>
                  ) : (
                    <span className="muted">End of workload</span>
                  )}
                </div>
              </>
            )}
          </Panel>

          <p className="page-note muted">
            This view includes only resumable backend operations. Non-resumable
            requests are not represented, and an operation can remain listed
            after its run lease expires until a retry reconciles it.
          </p>
        </>
      )}
    </div>
  );
}
