# AI Wisdom Router

A local-first life advisor that routes your question to the minds best equipped to answer it.

Instead of one flat chatbot persona, the router keeps a live **weight distribution** across a roster
of mentors — Paul Graham, Naval Ravikant, Warren Buffett, David Goggins, Charlie Munger. Every message
is embedded once, matched against each mentor's corpus of real quotes in a vector store, and scored
against their declared domain expertise. The blend shifts turn by turn as the conversation moves: ask
about runway and Paul Graham rises, pivot to your training schedule and Goggins takes over.

Everything runs on your machine — [Ollama](https://ollama.com) for generation and embeddings,
[Qdrant](https://qdrant.tech) for retrieval. No API keys, no data leaves the box.

```
you ──▶ embed ──▶ detect topics ─┐
                                 ├──▶ score mentors ──▶ blend weights ──▶ weighted prompt ──▶ answer
        retrieve their quotes ───┘         (per mentor)     (momentum)      (+ their quotes)
```

## Two modes

**Adaptive Mentor** — one synthesized voice. The system prompt is assembled from the top-weighted
mentors in proportion to their current weight, so the answer blends their styles rather than
impersonating any single one. Weights carry momentum between turns, so the voice drifts instead of
snapping, and staying on one topic for several turns gives that topic's specialists a bonus.

**Council Mode** — the top mentors each answer in their own voice, so you see the same problem framed
several ways. The calls are issued concurrently, and a mentor whose call fails is dropped from the
council instead of failing the turn.

In both modes the quotes retrieved for a mentor are injected into that mentor's section of the system
prompt, so the advice is grounded in what they actually said rather than the model's impression of them.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally
- Qdrant is **not** a separate install by default — the app uses an embedded on-disk store

## Setup

```bash
git clone https://github.com/SdSarthak/ai-wisdom-router.git
cd ai-wisdom-router

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r backend/requirements.txt

# Pull the models (~9 GB for qwen3:14b — see below for lighter options)
ollama pull qwen3:14b
ollama pull bge-m3

cp .env.example .env              # optional; every value has a working default
```

Then start it:

```bash
python -m backend.main
```

Open <http://localhost:8000>. The first launch embeds the quote corpus into Qdrant, which takes a
minute or two; later launches detect the corpus is present and skip straight to serving.

If Ollama is not running the server still starts — `/api/health` reports what is missing and the UI
shows a banner telling you which command to run.

### Lighter models

`qwen3:14b` wants roughly 10 GB of RAM. On a smaller machine:

```bash
ollama pull qwen3:4b
ollama pull nomic-embed-text
```

```dotenv
LLM_MODEL=qwen3:4b
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIM=768          # must match the model's output dimension
```

Changing the embedding model changes the vector size, so rebuild the collection:

```bash
python -m backend.vector_store.seeder --force
```

### Using a Qdrant server instead

The embedded store keeps a lock on its directory, so only one process can use it at a time. To run
Qdrant as a service instead:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

```dotenv
QDRANT_MODE=server
```

## Configuration

All settings are environment variables, read from the shell or from `.env`. See `.env.example` for the
full list. The ones worth touching:

| Variable | Default | What it does |
| --- | --- | --- |
| `LLM_MODEL` | `qwen3:14b` | Ollama model used for generation |
| `EMBEDDING_MODEL` | `bge-m3` | Ollama model used for embeddings |
| `EMBEDDING_DIM` | `1024` | Must match the embedding model's output size |
| `QDRANT_MODE` | `local` | `local` (on-disk), `server` (HTTP), or `memory` |
| `WEIGHT_MOMENTUM` | `0.35` | How much of the previous turn's weighting carries over |
| `MIN_WEIGHT_THRESHOLD` | `0.05` | Mentors below this share are dropped from the prompt |
| `TOPIC_SIMILARITY_THRESHOLD` | `0.45` | Similarity a message needs to count as "about" a topic |
| `EMBEDDING_SCORE_WEIGHT` | `0.6` | Pull of corpus similarity vs. declared expertise |
| `COUNCIL_TOP_N` | `4` | Seats on the council |
| `SEED_ON_STARTUP` | `true` | Set `false` once seeded for a faster boot |

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/chat` | Send a message. Body: `{session_id, message, mode}` where mode is `adaptive` or `council` |
| `GET` | `/api/session/{id}/weights` | Current mentor distribution for a session |
| `DELETE` | `/api/session/{id}` | Forget a conversation |
| `GET` | `/api/mentors` | The roster with display names, domains and colours |
| `GET` | `/api/health` | Ollama reachability, installed models, corpus size |

Interactive docs at <http://localhost:8000/docs>.

```bash
curl -X POST http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"demo","message":"I have 8 months of runway and no traction.","mode":"council"}'
```

`503` means the model backend is unavailable; the `detail` field says which command fixes it.

## How routing works

**Topic detection.** Ten topics each have a short anchor description, embedded once at startup. A
message is assigned every topic whose anchor it exceeds `TOPIC_SIMILARITY_THRESHOLD` against, and
always at least its single closest topic.

**Mentor scoring.** Each mentor gets `EMBEDDING_SCORE_WEIGHT × (mean similarity of their best matching
quotes) + DOMAIN_SCORE_WEIGHT × (their declared strength in the detected topics)`. The message is
embedded once per turn and that vector is reused for both topic detection and retrieval. If the vector
store is unavailable, scoring falls back to domain weights alone rather than collapsing to zero.

**Weight blending.** New scores are normalized, given a trajectory bonus if the conversation has stayed
on one topic for `TRAJECTORY_WINDOW` turns, then mixed with the previous distribution using
`WEIGHT_MOMENTUM`. Mentors under `MIN_WEIGHT_THRESHOLD` are dropped — but never all of them.

**Council selection.** The top `COUNCIL_TOP_N` mentors, plus one runner-up if they finished within
`COUNCIL_MIN_SCORE_GAP` of the cut line, so a near-tie is not decided by an arbitrary boundary.

## Adding a mentor

1. Add a `Mentor` to `MENTORS` in `backend/mentors/roster.py` with `domain_weights` over the topics in
   `TOPIC_DOMAINS`, a persona prompt and a colour.
2. Add their quotes to `SEED_DATA` in `backend/mentors/seed_data.py` as `(text, source, [topics])`.
3. `python -m backend.vector_store.seeder` — seeding is idempotent, so only the new mentor is embedded.

`tests/test_config.py` and `tests/test_seeder.py` will fail if a mentor declares a domain it has no
weight for, or tags a quote with an unknown topic.

## Tests

```bash
python -m pytest
```

122 tests, no network and no Ollama required — embeddings are replaced with a deterministic hash-based
stand-in, generation is stubbed, and Qdrant runs in-process.

## Project layout

```
backend/
  api/            routes and pydantic schemas
  graph/          per-turn orchestration (adaptive, council) and session state
  llm/            ollama http client, prompt assembly
  mentors/        roster definitions and the quote corpus
  scoring/        topic detection, mentor scoring, weight blending
  vector_store/   qdrant connection, embeddings, seeding
frontend/         static single-page UI served by the backend
tests/
```

## Notes

Sessions live in memory: they are bounded per conversation and evicted least-recently-used, and they do
not survive a restart. The browser keeps its session id in `localStorage` so a reload continues the same
conversation.

The mentors are personas built from public writing and talks. They are a way to get several honest
framings of a problem quickly — not the real people, and not professional advice.

## License

MIT
