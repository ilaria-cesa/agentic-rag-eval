"""Prompts and parsers for the planning and sufficiency-check steps of the agent.

The LLM is asked for small JSON objects. Small models sometimes add text around the JSON or
get it wrong, so every parser returns None (or a safe default) instead of raising, and the
agent falls back to a plain single search.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

MAX_SEARCHES = 4


@dataclass(frozen=True)
class Search:
    query: str
    ticker: str | None = None
    year: int | None = None


def company_table(filings: dict[str, list[str]], names: dict[str, str]) -> str:
    rows = []
    for ticker in sorted(filings):
        years = ", ".join(sorted({d[:4] for d in filings[ticker]}))
        rows.append(f"{ticker} = {names.get(ticker, ticker)} (fiscal years {years})")
    return "\n".join(rows)


def plan_messages(question: str, table: str) -> list[dict]:
    system = (
        "You plan searches over SEC Form 10-K filings. Companies and fiscal years available:\n"
        f"{table}\n\n"
        "Reply with ONLY a JSON object: "
        '{"searches": [{"query": "...", "ticker": "AAPL", "year": 2025}]}\n'
        "Rules: one search for every company AND fiscal year the question needs (a comparison "
        "between two years needs two searches); write each query as the filing itself would "
        "phrase it (financial terms, no question words); use null for ticker or year when the "
        f"question does not say; at most {MAX_SEARCHES} searches. If the question describes a "
        "company without naming it, use your own knowledge to pick the ticker from the list."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": question}]


def check_messages(question: str, evidence_lines: list[str]) -> list[dict]:
    system = (
        "You check whether retrieved passages from 10-K filings are enough to answer a question. "
        "Reply with ONLY a JSON object: "
        '{"enough": true} or {"enough": false, "search": {"query": "...", "ticker": "AAPL", '
        '"year": 2025}}. Say enough=false only if a specific fact needed for the answer is '
        "missing and a new search for it could plausibly find it; then describe that missing "
        "fact in the search query."
    )
    user = "Question: " + question + "\n\nRetrieved passages:\n" + "\n".join(evidence_lines)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _first_json_object(text: str) -> dict | None:
    start = text.find("{")
    while start != -1:
        depth = 0
        for end in range(start, len(text)):
            depth += {"{": 1, "}": -1}.get(text[end], 0)
            if depth == 0:
                try:
                    value = json.loads(text[start : end + 1])
                    return value if isinstance(value, dict) else None
                except json.JSONDecodeError:
                    break
        start = text.find("{", start + 1)
    return None


def _to_search(item: object, known: set[str]) -> Search | None:
    if not isinstance(item, dict) or not str(item.get("query", "")).strip():
        return None
    ticker = item.get("ticker")
    ticker = ticker.upper() if isinstance(ticker, str) and ticker.upper() in known else None
    year = item.get("year")
    year = (
        int(year) if isinstance(year, (int, str)) and re.fullmatch(r"20\d\d", str(year)) else None
    )
    return Search(str(item["query"]).strip(), ticker, year)


def parse_plan(text: str, known_tickers: set[str]) -> list[Search] | None:
    data = _first_json_object(text)
    if not data or not isinstance(data.get("searches"), list):
        return None
    searches = [_to_search(i, known_tickers) for i in data["searches"][:MAX_SEARCHES]]
    searches = [s for s in searches if s]
    return searches or None


def parse_check(text: str, known_tickers: set[str]) -> Search | None:
    """The follow-up search if the model says the evidence is not enough, otherwise None."""
    data = _first_json_object(text)
    if not data or data.get("enough") is not False:
        return None
    return _to_search(data.get("search"), known_tickers)
