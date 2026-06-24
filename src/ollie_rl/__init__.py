def dev() -> None:
    import uvicorn

    """Run the development server."""
    uvicorn.run("ollie_rl.server.app:app", host="127.0.0.1", port=8000, reload=True)
