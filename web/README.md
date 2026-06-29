# ollie-rl web

A small read-only dashboard for viewing **tuner stats** from the ollie-rl
FastAPI server. Built with Vite + React + TanStack Router / Query / Table.

The dashboard is served under the **`/app`** path and is always same-origin with
the API, so no CORS or API base URL configuration is needed.

## Pages

- **`/app/tuners`** — list of tuners (`GET /tuners`).
- **`/app/tuners/:tunerId`** — live training dashboard (`GET /tuners/{id}?progress=true`),
  polled every ~2s. Shows the run KPI strip, batch readiness, the scheduler's
  next pick, and the per-datum pool table.

The server only exposes a live snapshot (no history), so the UI polls rather
than charting trends over time.

## Develop

From the **repo root**, a single command runs both the FastAPI backend
(auto-reload) and the Vite dev server (HMR):

```bash
uv run poe dev
```

Then open **http://localhost:8000/app**. FastAPI serves the API directly and
reverse-proxies `/app/*` to the Vite dev server (HMR included — the websocket
connects straight to `:5173`), so the UI and API are a single origin. This is
enabled by the `OLLIE_WEB_DEV_SERVER=http://localhost:5173` env var that
`poe dev` sets automatically.

To run just the frontend (served by Vite on http://localhost:5173/app, but note
API calls then need the backend reachable on the same origin):

```bash
cd web
bun install
bun run dev
```

## Build & serve from FastAPI

```bash
cd web
bun run build      # type-check + production bundle into web/dist
```

FastAPI automatically serves `web/dist` at `/app` (with SPA fallback) when it
exists, so in production a single `uvicorn` process serves both the API and the
dashboard. Override the bundle location with the `OLLIE_WEB_DIST` env var.

## API types

The TypeScript API types are **generated from the server's OpenAPI schema**, not
hand-written. `src/api/schema.d.ts` is produced by
[`openapi-typescript`](https://openapi-ts.dev/), and `src/api/types.ts` exposes
friendly aliases over it. After changing the backend DTOs, regenerate from the
**repo root**:

```bash
uv run poe gen-web-types
```

This dumps the FastAPI OpenAPI document to `web/openapi.json` (gitignored) and
rewrites `src/api/schema.d.ts`. Do not edit `schema.d.ts` by hand.
