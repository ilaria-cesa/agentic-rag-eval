"""Download 10-K filings from EDGAR and extract the sections we index.

Usage (from the repo root):
    python -m src.ingest.run                      # default companies, last 2 10-Ks each
    python -m src.ingest.run --tickers JPM MSFT --years 1

Output (git-ignored): data/raw/<TICKER>/<accession>.html, data/parsed/<TICKER>_<report_date>.json
and data/summary.csv. The script is resumable: files already on disk are not downloaded again.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from dotenv import load_dotenv

from src.ingest.companies import DEFAULT_TICKERS
from src.ingest.edgar import EdgarClient
from src.ingest.parse import WANTED_ITEMS, extract_sections, html_to_text


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--years", type=int, default=2, help="number of latest 10-Ks per company")
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--max-rps", type=float, default=5.0)
    args = parser.parse_args(argv)

    load_dotenv()
    client = EdgarClient(max_rps=args.max_rps)
    raw_dir, parsed_dir = args.out / "raw", args.out / "parsed"
    parsed_dir.mkdir(parents=True, exist_ok=True)

    companies = client.resolve_tickers(args.tickers)
    for missing in sorted({t.upper() for t in args.tickers} - set(companies)):
        print(f"WARNING: ticker not found on EDGAR: {missing}")

    rows = []
    for ticker, (cik, name) in companies.items():
        for filing in client.list_10k_filings(cik, args.years):
            parsed_path = parsed_dir / f"{ticker}_{filing.report_date}.json"
            if parsed_path.exists():
                record = json.loads(parsed_path.read_text(encoding="utf-8"))
            else:
                raw_path = raw_dir / ticker / f"{filing.accession}.html"
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                if not raw_path.exists():
                    raw_path.write_text(client.get_text(filing.url), encoding="utf-8")
                sections, warnings = extract_sections(
                    html_to_text(raw_path.read_text(encoding="utf-8"))
                )
                record = {
                    "ticker": ticker,
                    "company": name,
                    "cik": cik,
                    "accession": filing.accession,
                    "form": filing.form,
                    "filing_date": filing.filing_date,
                    "report_date": filing.report_date,
                    "url": filing.url,
                    "sections": sections,
                    "warnings": warnings,
                }
                parsed_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
            row = {"ticker": ticker, "report_date": record["report_date"]}
            row.update({n: len(record["sections"].get(n, "")) for n in WANTED_ITEMS.values()})
            row["warnings"] = "; ".join(record["warnings"])
            rows.append(row)
            print(row)

    if rows:
        with open(args.out / "summary.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
