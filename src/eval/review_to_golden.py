"""Turn the hand-checked review sheet into golden/golden_set.jsonl.

Usage:  python -m src.eval.review_to_golden [--review data/golden_review.xlsx]

Verdict column: "ok" keeps the draft answer, "fix" uses the corrected answer, "drop" skips the
row. Free-text notes in the sheet are never copied into the golden file.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.eval.golden import load_jsonl


def build_rows(sheet_rows: list[tuple], chunks: list[dict]) -> list[dict]:
    periods: dict[str, set[str]] = defaultdict(set)
    for chunk in chunks:
        periods[chunk["ticker"]].add(chunk["report_date"])
    rows = []
    for (
        qid,
        company,
        qtype,
        question,
        answer,
        evidence,
        _text,
        verdict,
        fixed,
        _notes,
    ) in sheet_rows:
        verdict = (verdict or "").strip().lower()
        if verdict not in {"ok", "fix"}:
            continue
        ids = [e for e in (evidence or "").split("\n") if e]
        refuse = qtype == "unanswerable"
        final = "" if refuse else (fixed if verdict == "fix" and fixed else answer)
        dates = sorted({e.split("_")[1] for e in ids}) if ids else sorted(periods[company])
        if qtype == "multi_hop" and len(set(ids)) < 2:
            print(f"{qid}: multi_hop with a single chunk, kept as 'fact'")
            qtype = "fact"
        rows.append(
            {
                "id": qid,
                "question": question.strip(),
                "language": "en",
                "type": qtype,
                "tickers": [company],
                "report_dates": dates,
                "expected": "refuse" if refuse else "answer",
                "answer": final.strip(),
                "evidence_chunk_ids": ids,
                "notes": "",
            }
        )
    return sorted(rows, key=lambda r: r["id"])


def main(argv: list[str] | None = None) -> None:
    from openpyxl import load_workbook

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, default=Path("data/golden_review.xlsx"))
    parser.add_argument("--chunks", type=Path, default=Path("data/chunks_1600.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("golden/golden_set.jsonl"))
    args = parser.parse_args(argv)

    sheet = load_workbook(args.review)["review"]
    rows = build_rows(list(sheet.iter_rows(min_row=2, values_only=True)), load_jsonl(args.chunks))
    args.out.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    print(f"wrote {len(rows)} questions to {args.out}")


if __name__ == "__main__":
    main()
