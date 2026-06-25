# Practices
* use `uv add` to install package, avoid editing pyproject.toml directly
* run `uv run ty check` and `uv run pytest` to check code quality and correctness when needed
  * in most cases, you don't have to run these check when implementing anything.
  * usually it's better to do these check before commiting / pushing / creating prs.
* DO NOT RUSH to write unit test, confirm user's good with current implementation before start writing tests.