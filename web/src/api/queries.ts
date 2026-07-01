import { infiniteQueryOptions, queryOptions } from "@tanstack/react-query";
import {
  getCompletion,
  getRun,
  getTuner,
  listRuns,
  listTuners,
} from "./client";

/** Page size for the cursor-paginated runs list. */
export const RUNS_PAGE_SIZE = 25;

// The server exposes only a live snapshot (no history), so we poll.
export const tunersQuery = queryOptions({
  queryKey: ["tuners"],
  queryFn: listTuners,
  refetchInterval: 5000,
});

export const tunerQuery = (tunerId: string) =>
  queryOptions({
    queryKey: ["tuner", tunerId],
    queryFn: () => getTuner(tunerId),
    refetchInterval: 2000,
  });

// Unbounded fetch of every run — used by the reward-distribution view, which
// needs the full history to bucket rewards by generation.
export const runsQuery = (tunerId: string) =>
  queryOptions({
    queryKey: ["runs", tunerId],
    queryFn: () => listRuns(tunerId),
    refetchInterval: 5000,
  });

// Cursor-paginated runs for the runs list page. Each page is `RUNS_PAGE_SIZE`
// runs; `next_cursor` drives `fetchNextPage`.
export const runsPageQuery = (tunerId: string) =>
  infiniteQueryOptions({
    queryKey: ["runs", "paged", tunerId],
    queryFn: ({ pageParam }) =>
      listRuns(tunerId, { limit: RUNS_PAGE_SIZE, cursor: pageParam }),
    initialPageParam: null as string | null,
    getNextPageParam: (lastPage) => lastPage.next_cursor ?? undefined,
    refetchInterval: 5000,
  });

export const runQuery = (tunerId: string, runId: string) =>
  queryOptions({
    queryKey: ["run", tunerId, runId],
    queryFn: () => getRun(tunerId, runId),
    refetchInterval: 5000,
  });

export const completionQuery = (
  tunerId: string,
  runId: string,
  completionId: string,
) =>
  queryOptions({
    queryKey: ["completion", tunerId, runId, completionId],
    queryFn: () => getCompletion(tunerId, runId, completionId),
    // A recorded completion is immutable, so no need to poll.
    staleTime: Infinity,
  });
