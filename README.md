# TechJam Conversational Search

A conversational shopping agent that retrieves and ranks products from a 50,000-item
Amazon Clothing/Shoes/Jewelry catalog across multi-turn dialogue, combining sparse
(BM25) and dense (embedding) retrieval with a dialogue state tracker that detects
buying vs. browsing intent, accumulates disclosed constraints, and asks clarifying
questions to converge on the shopper's hidden target product.

## Project Overview

Each session gives the agent an anonymized `user_profile` and an opening customer
message. On every turn (up to 10), the agent may ask a clarification question
(`ask_attribute`) and/or return up to 10 ranked `parent_asin` candidates. A session
succeeds the moment the target product appears in the top-10.

Starting from the weak BM25-only starter (Hit Rate@10 0.125, MRR 0.068, MTTC 9.81 on
the public set), we built:

1. **Sparse retrieval** — SQLite FTS5 (BM25-scored) over title, categories, features,
   details, store, and description.
2. **Dense retrieval** — a `sentence-transformers` bi-encoder (`all-MiniLM-L6-v2`)
   embeds the full catalog once at startup; each turn's query is embedded and matched
   via cosine similarity, catching semantic matches BM25 misses on vague queries.
3. **Reciprocal Rank Fusion (RRF)** — merges sparse and dense ranked lists without
   needing to normalize their incompatible score scales.
4. **A dialogue state layer** (`state.py`, `signal_extractor.py`, `dialogue_manager.py`)
   that tracks what's been disclosed, detects intent overrides, and decides what to
   ask next.

### Dialogue state modules

- **`state.py`** — two dataclasses. `SessionState` holds per-session dialogue state:
  track (`browsing`/`buying`), buying confidence, confirmed constraints, and
  exhausted/asked buckets. `TurnSignal` holds what a single user turn implied.
- **`signal_extractor.py`** — regex-based turn classifier. Detects overrides
  ("actually", "never mind"), no-preference replies ("doesn't matter", "up to you"),
  vague/browsing language ("just looking", "not sure"), and which attribute bucket
  (material, colour, size, style, use_case) the message discloses a value for.
  Produces a `TurnSignal` and a `confidence_delta` used to move the session between
  "browsing" and "buying".
- **`dialogue_manager.py`** — pure state-transition logic. `update_state()` folds a
  `TurnSignal` into the prior `SessionState` (overrides replace a bucket's value,
  no-preference marks it exhausted, confidence is clamped to `[0, 1]`).
  `choose_next_attribute()` walks `DEFAULT_PRIORITY` to decide what to ask about next.
  `build_query()` turns confirmed constraints into a query fragment.

## Setup and Installation

Requires Python 3.10+ and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone <repo-url>
cd techjam-conversational-search
uv sync
```

Download the catalog per the challenge instructions (not committed to the repo):

```bash
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

Verify against the published `SHA256SUMS`.

First run downloads `sentence-transformers/all-MiniLM-L6-v2` (~80MB) from Hugging
Face, cached locally afterward (no `HF_TOKEN` needed for this public model).

> **WSL note:** keep the repo on the native Linux filesystem (e.g. `~/projects/...`)
> rather than a mounted Windows drive (`/mnt/c/`, `/mnt/d/`) — imports and file I/O
> are substantially slower across that boundary.

## Steps to Reproduce Results

```bash
uv run -m evaluator.local_evaluator
```

Builds the FTS5 index and embeds the full catalog, simulates all 200 public sessions,
writes `results.json`, and prints the summary metrics (`hit_rate_at_10`, `mrr`, `mttc`,
`efficiency`, `recommended_technical_score`, plus a `scenario_metrics` breakdown by
`boundary`/`browsing`/`buying`/`intent_override`).

For faster iteration during development, we built a smaller sample catalog guaranteed
to contain every ground-truth ASIN referenced in the public set:

```bash
uv run python scripts/build_sample_catalog.py
uv run -m evaluator.local_evaluator --catalog data/catalog_sample.jsonl
```

> Sample-catalog metrics aren't comparable to full-catalog runs (fewer distractors
> inflate hit rate) — use it only to sanity-check logic changes, then confirm final
> numbers on the full catalog.

## How the Solution Maps to the Four Pillars

**I. Core Architecture: Intent Routing & Hybrid Pipeline**
`dialogue_manager.py` tracks a `buying_confidence` score per session, updated each
turn by `signal_extractor.py`'s disclosed-bucket and override detection; the session
is classified `buying` once confidence exceeds a threshold, `browsing` otherwise. This
routing currently informs the *query construction* fed into retrieval (`build_query()`
folds confirmed constraints into the search string) rather than switching between two
structurally different retrieval tracks. Multi-route retrieval is implemented as
specified — keyword (BM25/FTS5) and vector similarity (dense bi-encoder), fused via
RRF. **Not currently implemented:** a category-matching route, and the LLM semantic
ranking stage — an LLM/cross-encoder reranking pass was prototyped but removed for
this submission due to per-turn latency at full catalog scale (see Limitations).

**II. Dialog Strategy: Multi-Turn Scenario Evolution**
Implemented. `SessionState.confirmed_constraints` accumulates disclosed slots
incrementally; `signal_extractor.py`'s override detection lets a new disclosure
overwrite a bucket's prior value rather than merge with it, handling abrupt intent
change. `choose_next_attribute()` selects an unasked, unexhausted bucket to ask about
each turn (gated to the first few turns and skipped when the shopper just disclosed
something or stated no preference), which is what drives proactive clarification.
**Partially implemented:** clarification is currently triggered by turn count and
prior disclosure, not by directly measuring candidate-pool size/over-generality —
tying `choose_next_attribute()` to actual retrieval ambiguity (e.g. asking only when
the fused candidate pool exceeds a size threshold) is the natural next step.

**III. Self-Evolution: Dynamic Context Programming**
**Partially implemented.** Short-term session state updates every turn via
`update_state()`, and retrieval queries are re-built each turn from accumulated
constraints (`build_query()`), so the agent does adapt within a session. **Not
implemented:** long-term user-profile usage — `user_profile` is passed into `reset()`
but currently unused — and there is no runtime re-orchestration of the pipeline's own
strategy beyond the fixed attribute-priority list; this is the pillar we'd invest in
most with more time.

**IV. Evaluation Matrix: Product & Efficiency Metrics**
Fully met by the provided evaluator, which we ran as-is without modification:
Hit Rate@10, MRR, and MTTC are computed exactly as specified, both overall and per
scenario, via `evaluator/local_evaluator.py`.

## Limitations & Future Improvements

**No LLM/cross-encoder reranking in this submission.** We built and tested a local
cross-encoder reranking pass (`cross-encoder/ms-marco-MiniLM-L-6-v2`) over the fused
candidate pool, which is the step the spec calls "LLM Semantic Ranking." At full
catalog scale (50,000 products × 200 sessions × up to 10 turns), the reranker's
per-(query, candidate) forward pass made full evaluation runs impractically slow, so
it's removed from this submission. Given more time, we would reintroduce it with a
tighter candidate pool, adaptive skip-when-unambiguous logic, and/or a distilled or
batched scoring approach to keep latency acceptable at scale.

**No category-matching retrieval route.** Only keyword and vector similarity are
implemented as retrieval routes; a dedicated category-tree route (as distinct from
category text folded into the BM25 fields) was not built separately.

**Clarification triggering is heuristic, not pool-size-driven.** `choose_next_attribute`
follows a fixed priority order gated by turn count, rather than directly measuring
candidate-pool ambiguity and triggering "Over-Generality" cutoffs as specified. A more
faithful implementation would compute candidate-pool size/diversity after each
retrieval pass and only ask when that pool is genuinely broad.

**No long-term personalization.** `user_profile` is accepted but unused; prior
purchase patterns and preference tags could inform initial retrieval ranking or which
attribute to ask about first.

**Attribute/override/no-preference detection is regex-based**, which is brittle to
phrasing our patterns don't anticipate. An LLM-based extractor would generalize
better to open-ended natural language, at the cost of latency and token spend.

**`intent_override` is the weakest-performing scenario** in our local testing —
correctly discarding a stale constraint while preserving unrelated ones is harder
than steady-state accumulation, and our confirmation flow is a first pass rather than
a fully robust solution.

## Team Member Contributions

| Member | Role | Contribution |
|---|---|---|
| PEARL, KM | Intent detection & routing | *(fill in specifics)* |
| WY | Retrieval pipeline | Sparse (BM25/FTS5) + dense (bi-encoder) hybrid retrieval, RRF fusion; prototyped and evaluated cross-encoder reranking (removed from final submission for latency) |
| MX | Conversation state & dialogue strategy | `state.py`, `signal_extractor.py`, `dialogue_manager.py` — *(fill in specifics)* |
| JY | Personalization & context programming | *(fill in specifics)* |
| *(unassigned)* | Evaluation, integration, infra | *(fill in)* |