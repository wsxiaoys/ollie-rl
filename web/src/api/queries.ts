import { infiniteQueryOptions, queryOptions } from "@tanstack/react-query";
import {
  getCompletion,
  getRun,
  getTuner,
  listData,
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

// The datum pool is static for a tuner's lifetime, so fetch it once and cache
// it — used to populate the runs filter dropdown.
export const dataQuery = (tunerId: string) =>
  queryOptions({
    queryKey: ["data", tunerId],
    queryFn: () => listData(tunerId),
    staleTime: Infinity,
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
// runs; `next_cursor` drives `fetchNextPage`. An optional `datumId` narrows the
// listing to a single datum and is part of the query key so switching filters
// starts a fresh paginated fetch.
export const runsPageQuery = (tunerId: string, datumId?: string) =>
  infiniteQueryOptions({
    queryKey: ["runs", "paged", tunerId, datumId ?? null],
    queryFn: ({ pageParam }) =>
      listRuns(tunerId, {
        limit: RUNS_PAGE_SIZE,
        cursor: pageParam,
        datumId,
      }),
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
