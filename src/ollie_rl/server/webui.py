"""Serve the web dashboard from FastAPI under ``/app``.

Two modes:

* **Production** — serve the built Vite bundle (``web/dist``) as static files,
  so a single ``uvicorn`` process serves both the API and the dashboard from the
  same origin. No-op until ``web/dist`` is built.
* **Development** — when ``OLLIE_WEB_DEV_SERVER`` is set (e.g.
  ``http://localhost:5173``), reverse-proxy ``/app/*`` to the running Vite dev
  server. This lets you open ``http://localhost:8000/app`` and get the live
  (HMR) frontend *and* the API on a single origin. The Vite HMR websocket
  connects directly to the dev server's port, so only HTTP needs proxying.

The dev proxy takes precedence over static files when configured.
"""

import os
from pathlib import Path

from fastapi import FastAPI
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.staticfiles import StaticFiles

# Headers that must not be forwarded verbatim by a proxy (RFC 7230 §6.1).
_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


class SPAStaticFiles(StaticFiles):
    """StaticFiles that falls back to index.html for client-side routes.

    Plain StaticFiles 404s on deep links like ``/app/tuners/abc``; an SPA needs
    those served the bundled ``index.html`` so the client router can take over.
    """

    async def get_response(self, path: str, scope):  # type: ignore[override]
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404:
                return await super().get_response("index.html", scope)
            raise


def _resolve_web_dist() -> Path:
    override = os.environ.get("OLLIE_WEB_DIST")
    if override:
        return Path(override)
    # src/ollie_rl/server/webui.py -> repo root -> web/dist
    return Path(__file__).resolve().parents[3] / "web" / "dist"


def _mount_dev_proxy(app: FastAPI, target: str) -> None:
    """Reverse-proxy ``/app/*`` to a running Vite dev server at ``target``."""
    import httpx
    from starlette.background import BackgroundTask
    from starlette.requests import Request
    from starlette.responses import RedirectResponse, Response, StreamingResponse

    client = httpx.AsyncClient(base_url=target, timeout=None)

    @app.get("/app", include_in_schema=False)
    async def _web_root() -> RedirectResponse:
        return RedirectResponse("/app/")

    @app.api_route(
        "/app/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
        include_in_schema=False,
    )
    async def _web_proxy(request: Request, path: str) -> Response:
        url = httpx.URL(
            path=f"/app/{path}",
            query=request.url.query.encode("utf-8"),
        )
        req_headers = [
            (k, v) for (k, v) in request.headers.raw if k.lower() != b"host"
        ]
        upstream_req = client.build_request(
            request.method,
            url,
            headers=req_headers,
            content=request.stream(),
        )
        upstream = await client.send(upstream_req, stream=True)
        resp_headers = [
            (k, v)
            for (k, v) in upstream.headers.multi_items()
            if k.lower() not in _HOP_BY_HOP
        ]
        return StreamingResponse(
            upstream.aiter_raw(),
            status_code=upstream.status_code,
            headers=dict(resp_headers),
            background=BackgroundTask(upstream.aclose),
        )


def mount_webui(app: FastAPI) -> None:
    """Mount the dashboard at ``/app`` (dev proxy if configured, else static)."""
    dev_server = os.environ.get("OLLIE_WEB_DEV_SERVER")
    if dev_server:
        _mount_dev_proxy(app, dev_server.rstrip("/"))
        return

    dist = _resolve_web_dist()
    if not dist.is_dir():
        return
    app.mount("/app", SPAStaticFiles(directory=dist, html=True), name="webui")
