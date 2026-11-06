"""Pick the passages that go to the answering model from everything the agent collected."""

from __future__ import annotations

from collections.abc import Callable

ScoreFn = Callable[[str, list[dict]], list[float]]


def select_context(
    question: str, evidence: list[dict], score_fn: ScoreFn | None, max_chunks: int
) -> list[dict]:
    """Keep the best passage of every filing found, then fill with the best of the rest.

    Scores come from the reranker applied to the ORIGINAL question, so a query that was rewritten
    too loosely cannot push the useful passage out. Keeping one passage per filing guarantees that
    a comparison between two years still sees both years. The result is ordered best first.
    """
    if score_fn is None or len(evidence) <= 1:
        return evidence[:max_chunks]
    scores = score_fn(question, evidence)
    ranked = sorted(range(len(evidence)), key=lambda i: -scores[i])
    chosen: list[int] = []
    seen_filings: set[tuple[str, str]] = set()
    for i in ranked:
        filing = (evidence[i]["ticker"], evidence[i]["report_date"])
        if filing not in seen_filings and len(chosen) < max_chunks:
            seen_filings.add(filing)
            chosen.append(i)
    for i in ranked:
        if i not in chosen and len(chosen) < max_chunks:
            chosen.append(i)
    return [evidence[i] for i in sorted(chosen, key=lambda i: -scores[i])]
