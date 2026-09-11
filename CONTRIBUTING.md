# Contributing

Open an issue before large changes. Keep external adapters optional and add regression
tests that reproduce behavior changes.

1. Create a focused branch.
2. Install with `uv sync --locked --extra dev`.
3. Run `uv run --no-sync ruff check .`, `uv run --no-sync ruff format --check .`,
   `uv run --no-sync pyright`, and `uv run --no-sync pytest`.
4. For optional integrations, sync with `--extra openai --extra chroma` and rerun tests.
5. Build with `uv build` and describe the change and verification in your pull request.

Use `uv lock` when changing dependencies and commit the updated lockfile.
Tests must not make paid API calls. Inject providers/clients and use temporary directories.
The Chroma integration test runs locally when the extra is installed.

Never commit API keys or scraped content you lack permission to redistribute.
Contributions are licensed under MIT.
