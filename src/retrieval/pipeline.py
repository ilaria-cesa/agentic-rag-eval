"""Retrieval pipeline with optional filing filter, per-filing search and reranking.

Steps (each can be switched on in a RetrievalConfig):
  1. scope  : guess company and fiscal year from the question and keep only those filings;
  2. search : dense cosine search; with `per_filing` every filing in scope gets its own top list,
              so a question about two years cannot be answered from one year only;
  3. rerank : a cross-encoder reorders the candidate pool.
Without a reranker, per-filing lists are interleaved (best of filing A, best of filing B, ...).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from src.retrieval.scope import Scope, detect_scope, filings_of

MAX_PER_FILING_FILINGS = 4  # above this, one global ranking is used inside the scope


class Scorer(Protocol):
    def score(self, question: str, passages: list[str]) -> list[float]: ...


@dataclass(frozen=True)
class RetrievalConfig:
    name: str
    use_filter: bool = False
    per_filing: bool = False
    rerank: bool = False
    pool: int = 30
    top_k: int = 10


class Corpus:
    """Chunks plus unit-length embeddings, indexed by filing for fast filtered search."""

    def __init__(self, chunks: list[dict], embeddings: np.ndarray):
        self.chunks = chunks
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.vectors = (embeddings / np.clip(norms, 1e-12, None)).astype(np.float32)
        self.filings = filings_of(chunks)
        self.by_filing: dict[tuple[str, str], np.ndarray] = {}
        keys = [(c["ticker"], c["report_date"]) for c in chunks]
        for key in set(keys):
            self.by_filing[key] = np.array([i for i, k in enumerate(keys) if k == key])
        self.everything = np.arange(len(chunks))


def _top(corpus: Corpus, query: np.ndarray, idx: np.ndarray, k: int) -> list[tuple[int, float]]:
    scores = corpus.vectors[idx] @ query
    order = np.argsort(-scores)[:k]
    return [(int(idx[i]), float(scores[i])) for i in order]


def _interleave(lists: list[list[tuple[int, float]]]) -> list[int]:
    out, seen = [], set()
    for rank in range(max(len(x) for x in lists)):
        for items in lists:
            if rank < len(items) and items[rank][0] not in seen:
                seen.add(items[rank][0])
                out.append(items[rank][0])
    return out


def retrieve(
    corpus: Corpus,
    question: str,
    query_vector: np.ndarray,
    config: RetrievalConfig,
    reranker: Scorer | None = None,
) -> tuple[list[int], Scope | None]:
    """Return (chunk indices, best first, the scope that was used or None)."""
    query = query_vector / max(float(np.linalg.norm(query_vector)), 1e-12)
    scope = detect_scope(question, corpus.filings) if config.use_filter else None
    keys = [k for k in (scope.report_dates if scope else []) if k in corpus.by_filing]
    if not keys:
        scope = None
    if scope is None:
        candidates = _top(corpus, query, corpus.everything, config.pool)
        lists = [candidates]
    elif config.per_filing and len(keys) <= MAX_PER_FILING_FILINGS:
        per = math.ceil(config.pool / len(keys))
        lists = [_top(corpus, query, corpus.by_filing[k], per) for k in keys]
    else:
        idx = np.concatenate([corpus.by_filing[k] for k in keys])
        lists = [_top(corpus, query, idx, config.pool)]

    if config.rerank:
        if reranker is None:
            raise ValueError("config.rerank needs a reranker")
        pool = list(dict.fromkeys(i for items in lists for i, _ in items))
        scores = reranker.score(question, [corpus.chunks[i]["embed_text"] for i in pool])
        ranked = [i for _, i in sorted(zip(scores, pool), key=lambda t: -t[0])]
    elif len(lists) > 1:
        ranked = _interleave(lists)
    else:
        ranked = [i for i, _ in lists[0]]
    return ranked[: config.top_k], scope


def scoped_search(
    corpus: Corpus,
    query_vector: np.ndarray,
    query: str,
    ticker: str | None,
    year: int | None,
    k: int,
    reranker: Scorer,
    pool: int = 30,
) -> list[dict]:
    """Search with an explicit scope chosen by the caller (the agent): dense top-`pool`, then rerank.

    Unknown tickers or years fall back to a wider scope instead of returning nothing.
    """
    keys = list(corpus.by_filing)
    if ticker and any(t == ticker for t, _ in keys):
        keys = [key for key in keys if key[0] == ticker]
    if year and any(int(d[:4]) == year for _, d in keys):
        keys = [key for key in keys if int(key[1][:4]) == year]
    idx = np.concatenate([corpus.by_filing[key] for key in keys])
    unit = query_vector / max(float(np.linalg.norm(query_vector)), 1e-12)
    candidates = [i for i, _ in _top(corpus, unit, idx, pool)]
    scores = reranker.score(query, [corpus.chunks[i]["embed_text"] for i in candidates])
    order = sorted(zip(scores, candidates), key=lambda t: -t[0])
    return [corpus.chunks[i] for _, i in order[:k]]
