from __future__ import annotations

from dataclasses import dataclass

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Cross-encoders run on-device with no per-call cost, so it's fine to score
# a wider pool than an LLM reranker would — cost/latency isn't the same
# constraint here.
DEFAULT_MAX_CANDIDATES = 50


@dataclass
class Candidate:
    parent_asin: str
    title: str
    details: str


class Reranker:
    """Local cross-encoder reranking pass over the fused sparse+dense candidate list.

    Runs entirely on-device — no network call, no API key, no per-call cost.
    Falls back to the incoming (fusion) order if the model can't be loaded
    (sentence-transformers missing, weights unavailable, etc.) or a scoring
    call fails, so reranking can only ever improve ordering, never break
    retrieval.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        cross_encoder: "CrossEncoder | None" = None,
    ) -> None:
        self.model_name = model
        self.max_candidates = max_candidates
        if cross_encoder is not None:
            self._model = cross_encoder
        elif CrossEncoder is not None:
            try:
                self._model = CrossEncoder(model)
            except Exception:
                # e.g. no internet to fetch weights, or a bad model name —
                # never let a load failure take retrieval down with it
                self._model = None
        else:
            self._model = None

    @property
    def enabled(self) -> bool:
        return self._model is not None

    def rerank(self, query: str, candidates: list[Candidate]) -> list[str]:
        """Return parent_asins re-ordered by relevance to `query`.

        Only the first `max_candidates` are scored; anything past that
        cutoff keeps its fused order and is appended after the reranked
        head untouched.
        """
        if not candidates:
            return []
        if self._model is None:
            return [c.parent_asin for c in candidates]

        head = candidates[: self.max_candidates]
        tail = candidates[self.max_candidates :]

        pairs = [(query, f"{c.title.strip()} — {c.details.strip()}") for c in head]
        try:
            scores = self._model.predict(pairs)
        except Exception:
            return [c.parent_asin for c in candidates]

        ranked_head = [
            c.parent_asin
            for c, _ in sorted(zip(head, scores), key=lambda pair: pair[1], reverse=True)
        ]
        return ranked_head + [c.parent_asin for c in tail]