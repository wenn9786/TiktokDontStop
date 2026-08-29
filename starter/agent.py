from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer


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

# STEP A: buckets classify_constraint() in the simulator can ever produce. 
# Asking about "category" or "brand" can never return a
# match, so we never ask about them — that would waste a turn for nothing.

ASKABLE_BUCKETS = ["material", "color", "budget", "size", "style", "use_case", "feature"]
 
BUCKET_KEYWORDS: dict[str, tuple[str, ...]] = {
    "budget": ("budget", "under", "cheap", "afford", "price", "$"),
    "material": (
        "cotton", "leather", "wool", "silk", "polyester", "denim", "linen",
        "suede", "nylon", "cashmere", "canvas", "fleece",
    ),
    "color": ("color", "black", "white", "blue", "red", "pink", "green", "grey", "gray", "yellow", "purple"),
    "size": ("size", "sizing", "width", "wide", "narrow", "small", "medium", "large", "xl"),
    "style": ("department", "style", "fit", "sleeve", "neck", "casual", "formal"),
    "use_case": ("hiking", "running", "gym", "winter", "outdoor", "work", "travel", "summer"),
}

# STEP B: phrases that signal the customer is replacing an earlier
# preference rather than adding to it.
OVERRIDE_SIGNALS = (
    "actually",
    "ignore my earlier preference",
    "what i need is",
    "instead of",
    "changed my mind",
)
 
# STEP B: yes/no vocabulary for resolving a pending override confirmation.
AFFIRMATIVE_SIGNALS = (
    "yes", "yeah", "yep", "yup", "sure", "correct", "right", "switch",
    "go ahead", "please do", "update it", "change it",
)
NEGATIVE_SIGNALS = (
    "no", "nah", "nope", "keep", "don't", "do not", "stay", "never mind",
    "leave it", "no thanks",
)

# STEP C: once this many buckets have been filled, stop asking follow-up
# questions no matter how broad the remaining candidate pool looks — we'd
# rather return imperfect recommendations than interrogate the customer
# forever.
MAX_BUCKETS_BEFORE_STOP_ASKING = 4

# STEP C: if an AND-across-buckets query still matches more than this many
# catalog rows, the pool is considered "too broad" and we ask another
# question instead of returning weak recommendations (as long as we're
# still under MAX_BUCKETS_BEFORE_STOP_ASKING).
BROAD_POOL_THRESHOLD = 25

def classify_constraint(text: str) -> dict[str, str]:
    """Return {bucket: matched keyword(s)} for every ASKABLE_BUCKETS bucket
    whose keywords appear in `text`. A single message can hit several
    buckets at once (e.g. "black leather boots under $50")."""
    term_set = set(_terms(text))
    hits: dict[str, str] = {}
    for bucket in ASKABLE_BUCKETS:
        matched = [kw for kw in BUCKET_KEYWORDS.get(bucket, ()) if kw in term_set]
        if matched:
            hits[bucket] = " ".join(matched)
    return hits


def _has_override_signal(text: str) -> bool:
    lowered = text.lower()
    return any(signal in lowered for signal in OVERRIDE_SIGNALS)


def _is_affirmative(text: str) -> bool:
    lowered = text.lower()
    return any(signal in lowered for signal in AFFIRMATIVE_SIGNALS)


def _is_negative(text: str) -> bool:
    lowered = text.lower()
    return any(signal in lowered for signal in NEGATIVE_SIGNALS)

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

class Agent:
    """Editable weak baseline: stateless BM25 retrieval with no LLM dependency."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: set[str] = set()
        self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self._asin_order: list[str] = []
        self._embeddings: np.ndarray | None = None
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

    def reset(self, session_id: str, user_profile: dict) -> None:
        # The profile is anonymized and may be used for personalization.
        self._sessions.add(session_id)

    def respond(
        self,
        session_id: str,
        user_message: str, # query
        turn: int,
        top_k: int,
    ) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")
        unique_terms = list(dict.fromkeys(_terms(user_message)))[:40]
        expression = " OR ".join(f'"{term}"' for term in unique_terms)

        if not expression:
            sparse_ranked: list[str] = []
        else:
            rows = self.connection.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? "
                "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
                (expression, top_k * 2),
            ).fetchall()
            sparse_ranked = [str(row[0]) for row in rows]

        dense_ranked = self._dense_search(user_message, top_k=50)
        fused = self._reciprocal_rank_fusion([sparse_ranked, dense_ranked])[:top_k]
        recommendations = [{"parent_asin": asin} for asin in fused]
        # unique_terms = list(dict.fromkeys(_terms(user_message)))[:40]
        # expression = " OR ".join(f'"{term}"' for term in unique_terms)
        # if not expression:
        #     recommendations: list[dict] = []
        # else:
        #     rows = self.connection.execute(
        #         "SELECT parent_asin FROM products WHERE products MATCH ? "
        #         "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
        #         (expression, top_k),
        #     ).fetchall()
        #     recommendations = [{"parent_asin": str(row[0])} for row in rows]
        return {
            "message": "Here are the closest matches I found.",
            "ask_attribute": None,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
