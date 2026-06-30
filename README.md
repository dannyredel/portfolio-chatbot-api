# portfolio-chatbot-api

FastAPI backend for the RAG-powered chatbot on [dannyredel.github.io](https://dannyredel.github.io).
Companion to the Quarto site repo — see `../dannyredel.github.io/_project/` for full project docs.

## Stack

- **FastAPI** + `uvicorn` — async web framework
- **File-based index** — `data/index.pkl` holds chunks + embeddings, loaded into memory at startup; cosine search via `numpy`. No database.
- **OpenAI `text-embedding-3-small`** (1536 dims) — embeddings
- **Claude Sonnet 4.6** (`claude-sonnet-4-6`) — answer generation, streamed via SSE
- **slowapi** — 20 req/min per IP

Why no DB? Corpus is small and slow-changing. Committing the `.pkl` with the code gives
reproducible deploys and zero infra cost. See [`_project/DECISIONS.md`](../dannyredel.github.io/_project/DECISIONS.md) — ADR-008.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET  | `/api/health` | liveness + current model |
| POST | `/api/chat` | body `{"question": "…"}`, returns `text/event-stream` |
| POST | `/api/ingest` | admin-only (`X-Admin-Key` header) — re-reads knowledge, rewrites `index.pkl`, hot-reloads in memory |

### SSE events from `/api/chat`

```
data: {"type":"sources","sources":[{"source":"profile.md","section":"Languages","similarity":0.83}]}
data: {"type":"token","text":"Yes"}
data: {"type":"token","text":" — "}
...
data: {"type":"done"}
```

## Local setup

```bash
cp .env.example .env
# fill in ANTHROPIC_API_KEY, OPENAI_API_KEY, ADMIN_API_KEY

python -m venv .venv
source .venv/Scripts/activate    # Windows Git Bash
pip install -r requirements.txt

# Build the index (reads ../dannyredel.github.io/knowledge/*.md → writes data/index.pkl)
python -m scripts.ingest

# Run the API
uvicorn app.main:app --reload --port 8000
```

Smoke test:

```bash
curl http://localhost:8000/api/health

curl -N -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"What programming languages does Daniel use?"}'
```

## Updating knowledge

When you edit `../dannyredel.github.io/knowledge/*.md`:

```bash
python -m scripts.ingest     # rewrites data/index.pkl
git add data/index.pkl       # commit the regenerated index
git commit -m "chore: reindex"
git push                     # Railway redeploys, new index picked up at startup
```

## Deploy (Railway)

1. Push this repo to GitHub (make sure `data/index.pkl` is **committed**, not gitignored).
2. Create a new Railway project → deploy from GitHub.
3. Set env vars from `.env.example` (skip `KNOWLEDGE_DIR` / `INDEX_PATH` unless you want to override).
4. Railway auto-detects the `Procfile`.
5. Copy the public URL and set `window.CHATBOT_API_URL = 'https://<railway-url>'` in the widget.

## Layout

```
app/
  main.py         FastAPI app, routes, CORS, rate limiting
  config.py       pydantic-settings config
  embeddings.py   OpenAI embedding wrapper
  retrieval.py    In-memory numpy store + pickle save/load
  prompts.py      System prompt + context-message builder
  chat.py         SSE streaming generator
scripts/
  ingest.py       markdown -> chunks -> embeddings -> data/index.pkl
data/
  index.pkl       Committed artifact. Regenerate with `python -m scripts.ingest`.
```
