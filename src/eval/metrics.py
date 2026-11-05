"""Metrics for the RAG baseline: retrieval quality and cheap, rule-based answer checks.

Nothing here uses an LLM judge. The checks are proxies:
  * retrieval: are the golden evidence chunks among the top-k retrieved chunks?
  * refusal: did the system refuse exactly when the golden set says it should?
  * citations: are the cited ids real (among the provided chunks) and do they hit the evidence?
  * number match: do the key numbers of the golden answer appear in the generated answer?
A "number match" only says the figures agree; a human (or a judge, later) must still read
the answers, in particular for false_premise questions that need a correction.
"""

from __future__ import annotations

import re
from collections import defaultdict

KS = (1, 3, 5, 10)
_NUMBER = re.compile(r"\d[\d,]*\.?\d*")


def retrieval_metrics(evidence: list[str], retrieved: list[str], ks: tuple[int, ...] = KS) -> dict:
    """Recall@k, hit@k (any evidence chunk), all@k (every evidence chunk) and reciprocal rank."""
    gold = set(evidence)
    out: dict[str, float] = {}
    for k in ks:
        found = gold & set(retrieved[:k])
        out[f"recall@{k}"] = len(found) / len(gold)
        out[f"hit@{k}"] = float(bool(found))
        out[f"all@{k}"] = float(found == gold)
    ranks = [i + 1 for i, chunk_id in enumerate(retrieved) if chunk_id in gold]
    out["mrr"] = 1 / ranks[0] if ranks else 0.0
    return out


def key_numbers(text: str) -> set[str]:
    """Numbers with at least two digits, without commas or trailing dots (years included)."""
    cleaned = (n.replace(",", "").rstrip(".") for n in _NUMBER.findall(text))
    return {n for n in cleaned if len(n.replace(".", "")) >= 2}


def numbers_match(gold_answer: str, answer: str) -> bool | None:
    """True when every key number of the gold answer is in the answer; None if it has none."""
    wanted = key_numbers(gold_answer)
    if not wanted:
        return None
    return wanted <= key_numbers(answer)


def score_row(question: dict, parsed: dict, retrieved: list[str]) -> dict:
    """Combine the golden row, the parsed model answer and the ranked retrieved ids."""
    should_refuse = question["expected"] == "refuse"
    row = {
        "id": question["id"],
        "type": question["type"],
        "refused": parsed["refused"],
        "behaviour_ok": should_refuse == parsed["refused"],
        "invalid_citation": bool(parsed["invalid_citations"]),
    }
    if should_refuse:
        return row
    cited = set(parsed["cited_ids"])
    row["cites_evidence"] = bool(cited & set(question["evidence_chunk_ids"]))
    row["uncited_answer"] = not parsed["refused"] and not cited
    row["numbers_match"] = (
        None if parsed["refused"] else numbers_match(question["answer"], parsed["answer"])
    )
    row.update(retrieval_metrics(question["evidence_chunk_ids"], retrieved))
    return row


def _mean(values: list) -> float | None:
    values = [float(v) for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else None


def summarise(rows: list[dict]) -> dict[str, dict]:
    """Mean of every numeric/bool column, overall and per question type."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups["all"].append(row)
        groups[row["type"]].append(row)
    summary = {}
    for name, items in groups.items():
        columns = sorted({key for item in items for key in item} - {"id", "type"})
        summary[name] = {"n": len(items)} | {
            col: _mean([item.get(col) for item in items]) for col in columns
        }
    return summary
