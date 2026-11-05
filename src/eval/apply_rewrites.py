"""Apply the approved question rewrites from data/golden_rewrites.xlsx to the golden set.

Usage:  python -m src.eval.apply_rewrites [--sheet data/golden_rewrites.xlsx]

Sheet columns: id, style, old question, new question, old answer, new answer, evidence text,
verdict (ok / keep / fix), final question, final answer, notes. Verdict "ok" takes the proposed
question and answer, "fix" takes the final columns, "keep" leaves the row as it was.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def apply(rows: list[dict], sheet_rows: list[tuple]) -> tuple[list[dict], int]:
    by_id = {r["id"]: r for r in rows}
    changed = 0
    for (
        qid,
        _style,
        _old_q,
        new_q,
        _old_a,
        new_a,
        _text,
        verdict,
        final_q,
        final_a,
        _n,
    ) in sheet_rows:
        verdict = (verdict or "").strip().lower()
        if verdict not in {"ok", "fix"} or qid not in by_id:
            continue
        question, answer = (
            (new_q, new_a) if verdict == "ok" else (final_q or new_q, final_a or new_a)
        )
        by_id[qid]["question"] = question.strip()
        by_id[qid]["answer"] = answer.strip()
        changed += 1
    return rows, changed


def main(argv: list[str] | None = None) -> None:
    from openpyxl import load_workbook

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet", type=Path, default=Path("data/golden_rewrites.xlsx"))
    parser.add_argument("--golden", type=Path, default=Path("golden/golden_set.jsonl"))
    args = parser.parse_args(argv)

    rows = [json.loads(x) for x in args.golden.read_text(encoding="utf-8").splitlines() if x]
    sheet = load_workbook(args.sheet).active
    rows, changed = apply(rows, list(sheet.iter_rows(min_row=2, values_only=True)))
    args.golden.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    print(f"rewrote {changed} questions in {args.golden}")


if __name__ == "__main__":
    main()
