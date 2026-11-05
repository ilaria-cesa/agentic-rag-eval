"""Check a file of drafted golden-set questions and pick a balanced subset to review by hand.

Usage:
    python -m src.eval.draft_select --draft data/possible_question.json

Writes data/golden_review.xlsx (question, draft answer and the evidence text side by side)
and prints a report of the automatic checks. Nothing here proves an answer is correct:
the checks only catch the obvious problems, a human still has to read the evidence.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

QUOTAS = {
    "fact": 18,
    "numeric": 12,
    "comparison": 10,
    "multi_hop": 10,
    "unanswerable": 10,
    "false_premise": 5,
}
_YEAR = re.compile(r"20\d\d|FY|fiscal", re.IGNORECASE)


def number_variants(number: str) -> set[str]:
    """Spellings of a number as it may appear in a filing (billions -> millions, commas)."""
    base = number.rstrip(".").replace(",", "")
    out = {base}
    try:
        value = float(base)
    except ValueError:
        return out
    out |= {f"{value * 1000:.0f}", f"{value:.1f}", f"{value:.2f}"}
    return out


def unsupported_numbers(answer: str, evidence_text: str) -> list[str]:
    """Numbers written in the answer that cannot be found in the evidence (maybe computed)."""
    haystack = evidence_text.replace(",", "")
    found = re.findall(r"\d[\d,]*\.?\d*", answer)
    return [
        n
        for n in found
        if len(n.replace(",", "").rstrip(".")) > 1
        and not any(v in haystack for v in number_variants(n))
    ]


def is_adjacent_multi_hop(evidence: list[str]) -> bool:
    """True when all evidence chunks sit next to each other in one section of one filing."""
    keys = {e.rsplit("_", 1)[0] for e in evidence}
    if len(keys) != 1 or len(evidence) < 2:
        return False
    positions = sorted(int(e.rsplit("_", 1)[1]) for e in evidence)
    return positions[-1] - positions[0] <= len(positions)


def problems(q: dict, chunks: dict[str, str]) -> list[str]:
    found = []
    evidence = q.get("evidence", [])
    if q["type"] == "unanswerable":
        if q["answer"].strip() != "NOT_FOUND":
            found.append("typed unanswerable but has an answer")
        return found
    if not evidence:
        found.append("no evidence")
    if any(e not in chunks for e in evidence):
        found.append("unknown chunk id")
        return found
    if not _YEAR.search(q["question"]):
        found.append("no year in question")
    text = " ".join(chunks[e] for e in evidence)
    missing = unsupported_numbers(q["answer"], text)
    if missing:
        found.append(f"numbers not in evidence: {', '.join(missing)}")
    if q["type"] == "multi_hop" and is_adjacent_multi_hop(evidence):
        found.append("multi_hop on adjacent chunks")
    return found


def select(questions: list[dict], chunks: dict[str, str], forced: set[str]) -> list[dict]:
    """Round-robin by company inside each type, least used company first."""
    used: Counter = Counter()
    chosen: list[dict] = []
    for qtype, quota in QUOTAS.items():
        pool = [
            q
            for q in questions
            if q["type"] == qtype and (q["id"] in forced or not problems(q, chunks))
        ]
        if qtype == "false_premise":
            pool = [q for q in pool if q["id"] in forced]
        picked: list[dict] = []
        while pool and len(picked) < quota:
            pool.sort(key=lambda q: (used[q["company"]], q["id"]))
            best = pool.pop(0)
            picked.append(best)
            used[best["company"]] += 1
        chosen += picked
    return chosen


def write_xlsx(rows: list[dict], chunks: dict[str, str], path: Path) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment

    wb = Workbook()
    ws = wb.active
    ws.title = "review"
    ws.append(
        [
            "id",
            "company",
            "type",
            "question",
            "draft answer",
            "evidence ids",
            "evidence text",
            "verdict (ok / fix / drop)",
            "corrected answer",
            "notes",
        ]
    )
    for q in rows:
        ids = q.get("evidence", [])
        text = "\n\n".join(f"[{e}]\n{chunks[e]}" for e in ids if e in chunks)
        ws.append(
            [
                q["id"],
                q["company"],
                q["type"],
                q["question"],
                q["answer"],
                "\n".join(ids),
                text[:32000],
                "",
                "",
                "",
            ]
        )
    for col, width in zip("ABCDEFGHIJ", [7, 8, 12, 45, 45, 32, 90, 14, 40, 30]):
        ws.column_dimensions[col].width = width
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A2"
    wb.save(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, default=Path("data/possible_question.json"))
    parser.add_argument("--chunks", type=Path, default=Path("data/chunks_1600.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/golden_review.xlsx"))
    parser.add_argument(
        "--force",
        default="Q009,Q030,Q058,Q059,Q079",
        help="comma separated ids to take even if a check complains",
    )
    args = parser.parse_args(argv)

    questions = json.loads(args.draft.read_text(encoding="utf-8"))["questions"]
    chunks = {}
    with args.chunks.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            chunks[row["chunk_id"]] = row["text"]

    flagged = {q["id"]: problems(q, chunks) for q in questions}
    print(f"{len(questions)} drafted questions, {sum(1 for v in flagged.values() if v)} flagged")
    for reason, count in Counter(
        p.split(":")[0] for v in flagged.values() for p in v
    ).most_common():
        print(f"  {count:3d}  {reason}")

    chosen = select(questions, chunks, {x for x in args.force.split(",") if x})
    for q in chosen:
        if q["type"] == "unanswerable":
            q["evidence"] = []
    write_xlsx(chosen, chunks, args.out)
    print(f"\nselected {len(chosen)} -> {args.out}")
    print("by type:", dict(Counter(q["type"] for q in chosen)))
    print("by company:", dict(sorted(Counter(q["company"] for q in chosen).items())))
    in_corpus = {cid.split("_")[0] for cid in chunks}
    covered = {q["company"] for q in questions}
    print("corpus companies with no drafted question:", sorted(in_corpus - covered))


if __name__ == "__main__":
    main()
