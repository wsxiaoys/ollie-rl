import type {
  ChatCompletionDetailResponse,
  EvalProgress,
  GetTunerResponse,
  ListDatumsResponse,
  ListRunsResponse,
  ListTunersResponse,
  RewardDistributionData,
  RunDetailResponse,
} from "./types";

// The dashboard is always served same-origin as the API: in prod FastAPI serves
// both the built SPA and the API; in dev FastAPI reverse-proxies /app to the
// Vite dev server. So every request uses a relative path.

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(path, {
    headers: { Accept: "application/json" },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      // ignore non-JSON error bodies
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

export function listTuners(): Promise<ListTunersResponse> {
  return get<ListTunersResponse>("/tuners");
}

export function getTuner(tunerId: string): Promise<GetTunerResponse> {
  return get<GetTunerResponse>(
    `/tuners/${encodeURIComponent(tunerId)}?progress=train`,
  );
}

// Datum-id pool for a tuner, optionally scoped to a single `split`
// ("train"/"eval") so a dropdown can list only the relevant datums.
export function listData(
  tunerId: string,
  split?: "train" | "eval" | null,
): Promise<ListDatumsResponse> {
  const params = new URLSearchParams();
  if (split) params.set("split", split);
  const qs = params.toString();
  return get<ListDatumsResponse>(
    `/tuners/${encodeURIComponent(tunerId)}/data${qs ? `?${qs}` : ""}`,
  );
}

export function listRuns(
  tunerId: string,
  opts: {
    limit?: number;
    cursor?: string | null;
    datumId?: string | null;
    kind?: "train" | "eval" | null;
  } = {},
): Promise<ListRunsResponse> {
  const params = new URLSearchParams();
  if (opts.limit != null) params.set("limit", String(opts.limit));
  if (opts.cursor) params.set("cursor", opts.cursor);
  if (opts.datumId) params.set("datum_id", opts.datumId);
  if (opts.kind) params.set("kind", opts.kind);
  const qs = params.toString();
  return get<ListRunsResponse>(
    `/tuners/${encodeURIComponent(tunerId)}/runs${qs ? `?${qs}` : ""}`,
  );
}

// Reward distribution bucketed by policy generation. `kind="eval"` buckets the
// held-out eval split by the generation of the checkpoint each eval run
// targeted; `kind="train"` (default) buckets training runs by completion
// generation. `datumId` optionally scopes to a single datum.
export function getRewardDistribution(
  tunerId: string,
  datumId?: string | null,
  kind: "train" | "eval" = "train",
): Promise<RewardDistributionData> {
  const params = new URLSearchParams();
  if (datumId) params.set("datum_id", datumId);
  if (kind !== "train") params.set("kind", kind);
  const qs = params.toString();
  return get<RewardDistributionData>(
    `/tuners/${encodeURIComponent(tunerId)}/reward-distribution${
      qs ? `?${qs}` : ""
    }`,
  );
}

// Per-eval-datum held-out status rollup (in-flight / completed), powering the
// Eval page's status table. Served as `progress.eval` on the tuner detail
// endpoint when fetched with `?progress=eval`.
export async function getEvalProgress(
  tunerId: string,
): Promise<EvalProgress | null> {
  const tuner = await get<GetTunerResponse>(
    `/tuners/${encodeURIComponent(tunerId)}?progress=eval`,
  );
  return tuner.progress?.eval ?? null;
}

export function getRun(
  tunerId: string,
  runId: string,
): Promise<RunDetailResponse> {
  return get<RunDetailResponse>(
    `/tuners/${encodeURIComponent(tunerId)}/runs/${encodeURIComponent(runId)}`,
  );
}

export function getCompletion(
  tunerId: string,
  runId: string,
  completionId: string,
): Promise<ChatCompletionDetailResponse> {
  return get<ChatCompletionDetailResponse>(
    `/tuners/${encodeURIComponent(tunerId)}/runs/${encodeURIComponent(
      runId,
    )}/completions/${encodeURIComponent(completionId)}`,
  );
}
