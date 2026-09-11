# Changelog

## 0.1.1

- Fixed raw HTML handling, explicit target errors, resource cleanup, and bounded HTTP retries.
- Fixed non-terminating crawl and chunk settings; normalized URLs and bounded crawl scheduling.
- Preserved sidebar link discovery while extracting main content, Markdown links, and code.
- Added Unicode local embeddings, OpenAI batching, and client ownership.
- Added vector validation and idempotent chunk upserts.
- Matched memory/Chroma cosine scores; moved database operations off the event loop.
- Fixed source numbering and added structured citations and bounded answer context.
- Added offline CLI search, single-question chat, friendly errors, and consistent file formats.
- Added atomic UTF-8 files and versioned indexes storing embedding settings.
- Restored missing repository configuration and added regression and integration tests.

### Migration notes

Rebuild v0.1.0 JSON indexes: the schema and local tokenizer have changed. Chroma
collections now require cosine distance and store complete chunk metadata; use a new
collection name and reindex. Custom stores should implement ID upsert semantics.
Repeated indexing of changed pages does not remove obsolete chunks; rebuild a store
for full replacement. HTTP retries may increase request counts; set max_retries=0
for one attempt per fetch.
