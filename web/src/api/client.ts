import type {
  GetTunerResponse,
  ListRunsResponse,
  ListTunersResponse,
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
    `/tuners/${encodeURIComponent(tunerId)}?progress=true`,
  );
}

export function listRuns(tunerId: string): Promise<ListRunsResponse> {
  return get<ListRunsResponse>(
    `/tuners/${encodeURIComponent(tunerId)}/runs`,
  );
}

export function getRun(
  tunerId: string,
  runId: string,
): Promise<RunDetailResponse> {
  return get<RunDetailResponse>(
    `/tuners/${encodeURIComponent(tunerId)}/runs/${encodeURIComponent(runId)}`,
  );
}
