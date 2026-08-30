from __future__ import annotations
 
import json
import re
import sqlite3
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
    # words that only show up as connective tissue in override phrases
    # ("actually", "instead of", "changed my mind", ...) — without this
    # they get misclassified into a bucket and corrupt query-building.
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
 
class Agent:
    """Conversational retrieval agent backed by the local dialogue modules."""
 
    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, SessionState] = {}
        self._embedder = SentenceTransformer("all-MiniLM-L6-v2") if SentenceTransformer else None
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
        self._sessions[session_id] = state
 
        # Keep category/confirmed constraints in every turn.
        # Later user replies can be short ("black", "yes", etc.), so using
        # only the newest message loses important retrieval context.
        query_parts: list[str] = []
        if getattr(state, "category", ""):
            query_parts.append(state.category)
 
        built_query = build_query(state)
        if built_query:
            query_parts.append(built_query)
 
        query_parts.append(user_message)
        query = " ".join(part for part in query_parts if part)
 
        # ---- Stage 1: broad lexical retrieval ----
        unique_terms = list(dict.fromkeys(_terms(query)))[:60]
        expression = " OR ".join(f'"{term}"' for term in unique_terms)
 
        if expression:
            rows = self.connection.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? "
                "ORDER BY bm25(products, 0.0, 8.0, 5.0, 3.0, 2.0, 2.0, 1.0) LIMIT ?",
                (expression, max(100, top_k * 10)),
            ).fetchall()
            sparse_ranked = [str(row[0]) for row in rows]
        else:
            sparse_ranked = []
 
        # ---- Stage 2: broad semantic retrieval ----
        dense_ranked = self._dense_search(query, top_k=100)
 
        # ---- Stage 3: RRF candidate fusion ----
        fused = self._reciprocal_rank_fusion(
            [sparse_ranked, dense_ranked],
            k=40,
        )
        candidate_ids = fused[:max(100, top_k * 10)]
 
        # ---- Stage 4: explicit constraint-aware reranking ----
        # RRF alone does not know that "black leather" is more important than
        # a generic semantic match. The second stage explicitly rewards
        # products satisfying confirmed constraints.
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