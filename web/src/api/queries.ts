import { queryOptions } from "@tanstack/react-query";
import {
  getCompletion,
  getRun,
  getTuner,
  listRuns,
  listTuners,
} from "./client";

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

export const runsQuery = (tunerId: string) =>
  queryOptions({
    queryKey: ["runs", tunerId],
    queryFn: () => listRuns(tunerId),
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
