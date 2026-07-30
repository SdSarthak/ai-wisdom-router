# AI Wisdom Router

A local-first life advisor that routes your question to the minds best equipped to answer it.

Instead of one flat chatbot persona, the router keeps a live **weight distribution** across a roster
of mentors (Paul Graham, Naval Ravikant, Warren Buffett, David Goggins, Charlie Munger). Every message
is embedded, matched against each mentor's corpus of real quotes in a vector store, and scored against
their domain expertise. The blend shifts turn by turn as the conversation moves — ask about runway and
Paul Graham rises, pivot to your training schedule and Goggins takes over.

Everything runs on your machine: [Ollama](https://ollama.com) for generation and embeddings,
[Qdrant](https://qdrant.tech) for retrieval. No API keys, no data leaves the box.

## Two modes

**Adaptive Mentor** — one synthesized voice. The system prompt is assembled from the top-weighted
mentors in proportion to their current weight, so the answer blends their styles rather than
impersonating any single one. Weights carry momentum between turns and get a trajectory bonus when
you stay on a topic.

**Council Mode** — the top mentors each answer in their own voice, in parallel, so you can see the
same problem framed five different ways.

## Setup

See the full guide below once you have Ollama and Qdrant available.
