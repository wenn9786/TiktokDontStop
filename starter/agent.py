from __future__ import annotations

import json
import re
import sqlite3
import math
from pathlib import Path
import numpy as np
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
from dialogue_manager import DEFAULT_PRIORITY, build_query, choose_next_attribute, update_state
from signal_extractor import extract_signal
from state import SessionState


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    # words that only show up as connective tissue inside OVERRIDE_SIGNALS
    # phrases ("actually", "instead of", "changed my mind", ...) — without
    # this they get misclassified into a bucket (defaulting to "feature")
    # and corrupt the override-detection/confirmation flow.
    "actually", "ignore", "earlier", "preference", "what", "need",
    "instead", "changed", "mind",
}

def _text(value: object) -> str:
    # {"color": "red", "size": "M"} → "color red size M"
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]

_CHEAPER_PATTERNS = (
    r"\bcheap", r"\blower price", r"\bless expensive", r"\baffordable",
    r"\bbudget\b", r"\bunder \$?\d+", r"\bsomething cheaper\b",
)

_PRICIER_PATTERNS = (
    r"\bmore premium", r"\bhigher end", r"\bexpensive", r"\bluxury",
    r"\bpremium\b", r"\bhigh quality",
)

def _extract_price_sentiment(messages: list[str]) -> float:

    """Returns a signal in [-1, 1]. Negative = wants cheaper, positive =
    wants pricier. Repetition saturates via tanh rather than clipping hard
    at one mention."""

    signal = 0.0

    for text in messages:
        low = text.lower()
        if any(re.search(pat, low) for pat in _CHEAPER_PATTERNS):
            signal -= 1.0
        if any(re.search(pat, low) for pat in _PRICIER_PATTERNS):
            signal += 1.0

    return math.tanh(signal / 2.0)

def _profile_price_signal(profile: dict) -> float:

    """Long-term-profile version of the same signal, scanning the
    user_profile's summary/preference_tags instead of live messages."""

    if not profile:
        return 0.0

    text = " ".join([
        str(profile.get("summary", "")),
        " ".join(str(tag) for tag in (profile.get("preference_tags") or [])),
    ])

    return _extract_price_sentiment([text])


class Agent:
    """Conversational retrieval agent backed by the local dialogue modules."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, SessionState] = {}
        self._embedder = SentenceTransformer("all-MiniLM-L6-v2") if SentenceTransformer else None
        self._asin_order: list[str] = []
        self._embeddings: np.ndarray | None = None
        self._prices: dict[str, float] = {}
        self._categories: dict[str, set[str]] = {}
        self._stores: dict[str, str] = {}
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        dense_texts: list[str] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                dense_texts.append(
                    f"{_text(product.get('title'))}. {_text(product.get('categories'))}. "
                    f"{_text(product.get('features'))}. {_text(product.get('description'))}"
                )
                self._asin_order.append(str(product["parent_asin"]))
                price = product.get("price")
                if price not in (None, ""):
                    try:
                        self._prices[str(product["parent_asin"])] = float(price)
                    except (TypeError, ValueError):
                        pass
                raw_categories = product.get("categories") or []
                if isinstance(raw_categories, list):
                    self._categories[str(product["parent_asin"])] = {
                        str(c).strip().lower() for c in raw_categories if c
                    }
                store = product.get("store")
                if store:
                    self._stores[str(product["parent_asin"])] = str(store).strip().lower()

                batch.append(
                    (   # fields doesn't fully match catalog.json1?
                        str(product["parent_asin"]),
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()
        if self._embedder:
            self._embeddings = self._embedder.encode(
                dense_texts,
                batch_size=128,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )

    def _dense_search(self, user_message: str, top_k: int) -> list[str]:
        if not user_message.strip() or self._embeddings is None:
            return []
        query_vec = self._embedder.encode(
            [user_message], normalize_embeddings=True, convert_to_numpy=True
        )[0]
        similarities = self._embeddings @ query_vec
        top_indices = np.argsort(-similarities)[:top_k]
        return [self._asin_order[i] for i in top_indices]

    def _reciprocal_rank_fusion(self, ranked_lists: list[list[str]], k: int = 60) -> list[str]:
        scores: dict[str, float] = {}
        for ranked in ranked_lists:
            for position, parent_asin in enumerate(ranked):
                scores[parent_asin] = scores.get(parent_asin, 0.0) + 1.0 / (k + position + 1)
        return sorted(scores, key=lambda asin: scores[asin], reverse=True)

    def _reciprocal_rank_fusion_scores(self, ranked_lists: list[list[str]], k: int = 60) -> dict[str, float]:

        """Same fusion math as the original _reciprocal_rank_fusion, but
        returns raw scores instead of an already-sorted list, so
        personalization can adjust scores before the final top-K cut."""

        scores: dict[str, float] = {}

        for ranked in ranked_lists:
            for position, parent_asin in enumerate(ranked):
                scores[parent_asin] = scores.get(parent_asin, 0.0) + 1.0 / (k + position + 1)
        return scores

    def _apply_own_price_detection(self, state: SessionState, signal, user_message: str) -> None:
        """
        Self-contained fallback: if signal_extractor.py already classified
        this turn as a 'budget' disclosure, do nothing (avoid double-
        counting the same mention). Otherwise, independently check the raw
        message for price language ourselves. This means price
        personalization keeps working even if signal_extractor's bucket
        classification handles "cheaper"/"premium"-style phrasing
        differently than expected.
        """
        if signal.disclosed_bucket == "budget":
            return
        direction = _extract_price_sentiment([user_message])
        if direction == 0.0:
            return

        state.confirmed_constraints["budget"] = user_message
        state.bucket_mention_counts["budget"] = state.bucket_mention_counts.get("budget", 0) + 1

    def _price_signal(self, session_id: str) -> float:
        """Combined price-sentiment signal in [-1, 1]: 70% weight on what the
        user has said this session (read off the shared SessionState), 30%
        weight on the long-term profile."""

        state = self._sessions.get(session_id)
        if state is None:
            return 0.0

        latest_budget_text = state.confirmed_constraints.get("budget", "")
        mentions = state.bucket_mention_counts.get("budget", 0)

        if not latest_budget_text or mentions == 0:
            history_signal = 0.0
        else:
            direction = _extract_price_sentiment([latest_budget_text])
            sign = 1.0 if direction > 0 else (-1.0 if direction < 0 else 0.0)
            history_signal = sign * math.tanh(mentions / 2.0)
        profile_signal = _profile_price_signal(state.user_profile)
        combined = 0.7 * history_signal + 0.3 * profile_signal
        return max(-1.0, min(1.0, combined))
    
    def _apply_price_preference(
        self, scores: dict[str, float], session_id: str, weight: float = 1.0
    ) -> dict[str, float]:

        """Boost cheaper candidates when the user wants cheaper (signal < 0),
        boost pricier ones when signal > 0. Boost magnitude is scaled
        relative to the existing score range."""

        signal_strength = self._price_signal(session_id)
        if signal_strength == 0.0 or not self._prices:
            return scores
        priced = [(asin, self._prices[asin]) for asin in scores if asin in self._prices]

        if not priced:
            return scores
        min_p = min(p for _, p in priced)
        max_p = max(p for _, p in priced)
        price_range = max(max_p - min_p, 1e-6)
        score_scale = max(scores.values()) if scores else 0.0
        adjusted = dict(scores)
        for asin, price in priced:
            normalized = (price - min_p) / price_range  # 0 = cheapest, 1 = priciest
            boost = -signal_strength * (0.5 - normalized) * weight * score_scale
            adjusted[asin] = adjusted.get(asin, 0.0) + boost
        return adjusted

    def _item_similarity(self, asin_a: str, asin_b: str) -> float:

        """Cheap, embedding-free similarity: category overlap (Jaccard) does
        most of the work, matching store adds a smaller bonus."""

        cats_a = self._categories.get(asin_a, set())
        cats_b = self._categories.get(asin_b, set())

        if cats_a and cats_b:
            union = len(cats_a | cats_b)
            cat_sim = len(cats_a & cats_b) / union if union else 0.0

        else:
            cat_sim = 0.0

        store_a = self._stores.get(asin_a)
        store_sim = 1.0 if store_a and store_a == self._stores.get(asin_b) else 0.0
        return min(1.0, 0.65 * cat_sim + 0.35 * store_sim)

    def _apply_negative_reweight(
        self,
        scores: dict[str, float],
        session_id: str,
        turn: int,
        similarity_weight: float = 0.6,
        exact_reshow_weight: float = 1.5,
        recency_decay: float = 0.85,
    ) -> dict[str, float]:

        """Penalize candidates similar to (or identical to) products already
        shown in earlier turns. Nothing shown in a prior turn was selected —
        the session is still going — so everything in state.shown_asins is,
        by construction, a rejection."""
        state = self._sessions.get(session_id)
        if state is None or not state.shown_asins:
            return scores
        score_scale = max(scores.values()) if scores else 0.0
        if score_scale == 0.0:
            return scores
        
        adjusted = dict(scores)
        n_shown = max(1, len(state.shown_asins))

        for candidate_asin in scores:
            total_penalty = 0.0
            for shown_asin, shown_turn in state.shown_asins.items():
                recency_weight = recency_decay ** max(0, turn - shown_turn - 1)
                if candidate_asin == shown_asin:
                    count = state.shown_counts.get(shown_asin, 1)
                    repeat_strength = math.tanh(count / 1.5)
                    total_penalty += exact_reshow_weight * repeat_strength * recency_weight
                else:
                    total_penalty += self._item_similarity(candidate_asin, shown_asin) * recency_weight

            total_penalty /= n_shown
            adjusted[candidate_asin] = adjusted[candidate_asin] - similarity_weight * score_scale * total_penalty
        return adjusted

    def reset(self, session_id: str, user_profile: dict) -> None:
        # The profile is anonymized and may be used for personalization.
        self._sessions[session_id] = SessionState()

    def respond(
        self,
        session_id: str,
        user_message: str, # query
        turn: int,
        top_k: int,
    ) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")
        prior_state = self._sessions[session_id]
        signal = extract_signal(user_message, prior_state)
        if signal.disclosed_bucket == "colour":
            signal.disclosed_bucket = "color"
        state = update_state(prior_state, signal)
        state.track = "buying" if state.buying_confidence > 0.8 else "browsing"
        self._apply_own_price_detection(state, signal, user_message)
        self._sessions[session_id] = state

        query_parts: list[str] = []
        if getattr(state, "category", ""):
            query_parts.append(state.category)

        built_query = build_query(state)
        if build_query:
            query_parts.append(built_query)

        query_parts.append(user_message)

        query = " ".join(part for part in (build_query(state), user_message) if part)
        unique_terms = list(dict.fromkeys(_terms(user_message)))[:40]
        expression = " OR ".join(f'"{term}"' for term in unique_terms)

        if expression:
            rows = self.connection.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? "
                "ORDER BY bm25(products, 0.0, 8.0, 5.0, 3.0, 2.0, 2.0, 1.0) LIMIT ? "
                (expression, max(100, top_k * 10)),
            ).fetchall()
            sparse_ranked = [str(row[0]) for row in rows]
        else:
            sparse_ranked = []

        dense_ranked = self._dense_search(user_message, top_k=50)
        scores = self._reciprocal_rank_fusion_scores([sparse_ranked, dense_ranked])
        scores = self._apply_price_preference(scores, session_id)
        scores = self._apply_negative_reweight(scores, session_id, turn)
        # fused = self._reciprocal_rank_fusion([sparse_ranked, dense_ranked])[:top_k]
        # recommendations = [{"parent_asin": asin} for asin in fused]

        # for asin in fused:
        #     state.shown_asins[asin] = turn
        #     state.shown_counts[asin] = state.shown_counts.get(asin, 0) + 1

        fused = self._reciprocal_rank_fusion(
            [sparse_ranked, dense_ranked],
            k=40,
        )
        candidate_ids = fused[:max(100, top_k * 10)]

        reranked: list[tuple[float, str]] = []
 
        try:
            if candidate_ids:
                placeholders = ",".join("?" for _ in candidate_ids)
                rows = self.connection.execute(
                    f"SELECT parent_asin, title, categories, features, details, "
                    f"store, description FROM products "
                    f"WHERE parent_asin IN ({placeholders})",
                    candidate_ids,
                ).fetchall()
 
                rrf_position = {
                    asin: position for position, asin in enumerate(fused)
                }
                query_terms = set(_terms(query))
 
                for row in rows:
                    asin = str(row[0])
                    fields = [str(value or "") for value in row[1:]]
                    corpus = " ".join(fields).lower()
                    product_terms = set(_terms(corpus))
 
                    position = rrf_position.get(asin, len(fused))
                    score = 1.0 / (1.0 + position)
 
                    # General lexical overlap.
                    overlap = len(query_terms & product_terms)
                    score += 0.025 * min(overlap, 15)
 
                    # Stronger weights for explicit user constraints.
                    for bucket, value in state.confirmed_constraints.items():
                        value_text = str(value).lower().strip()
                        value_terms = set(_terms(value_text))
 
                        if not value_terms:
                            continue
 
                        if value_text in corpus:
                            score += 0.28
                        elif value_terms.issubset(product_terms):
                            score += 0.20
                        elif value_terms & product_terms:
                            score += 0.06
 
                        # Common aliases.
                        if bucket == "color":
                            if "grey" in value_text and "gray" in corpus:
                                score += 0.08
                            if "gray" in value_text and "grey" in corpus:
                                score += 0.08
 
                        if bucket == "size":
                            size_aliases = {
                                "extra small": ("xs", "extra small"),
                                "extra large": ("xl", "extra large"),
                                "x-large": ("xl", "extra large"),
                            }
                            for source, targets in size_aliases.items():
                                if source in value_text and any(
                                    target in corpus for target in targets
                                ):
                                    score += 0.08
 
                    reranked.append((score, asin))
        except Exception:
            # Safety fallback: if the reranking stage hits a schema or data
            # problem, fall back to the RRF order instead of throwing and
            # losing the whole turn (matches what a friend's version does).
            reranked = [
                (1.0 / (1.0 + position), asin)
                for position, asin in enumerate(candidate_ids)
            ]
 
        reranked.sort(key=lambda item: item[0], reverse=True)
        recommendations = [
            {"parent_asin": asin}
            for _, asin in reranked[:top_k]
        ]

        ask_attribute = None
        if turn < 5 and not signal.disclosed_bucket and not signal.is_no_preference:
            if len(state.confirmed_constraints) < 4:
                ask_attribute = choose_next_attribute(state, DEFAULT_PRIORITY)

        return {
            "message": "Here are the closest matches I found.",
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
