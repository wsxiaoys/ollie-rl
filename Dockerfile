# syntax=docker/dockerfile:1.7

# --- Stage 1: build the web dashboard (Vite bundle) ---
FROM oven/bun:1-slim AS web-builder

WORKDIR /web

# Install dependencies first to leverage Docker layer caching.
COPY web/package.json web/bun.lock ./
RUN --mount=type=cache,target=/root/.bun/install/cache \
    bun install --frozen-lockfile

# Copy the rest of the web sources and build the production bundle into web/dist.
COPY web/ ./
RUN bun run build

# --- Stage 2: build the Python service ---
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Install dependencies first to leverage Docker layer caching.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Copy the project source and install it.
COPY src ./src
COPY README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Bundle the built web dashboard so the server can serve it at /app.
COPY --from=web-builder /web/dist ./web/dist

EXPOSE 8000

CMD ["uvicorn", "ollie_rl.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
