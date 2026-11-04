"""Upload the chunk file to a PRIVATE Hugging Face dataset repo.

Usage (token from the environment, never from code):
    export HF_TOKEN=...      # PowerShell:  $env:HF_TOKEN = "..."
    python -m src.hub.push_dataset --repo-id <user>/<name>

Only data/chunks_<n>.jsonl is uploaded. Raw HTML and parsed filings stay on disk.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def push(repo_id: str, path: Path, api: HfApi | None = None, private: bool = True) -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Set the HF_TOKEN environment variable (a token with write access).")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.chunking.run` first.")
    api = api or HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=path.name,
        repo_id=repo_id,
        repo_type="dataset",
    )
    return f"https://huggingface.co/datasets/{repo_id}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="e.g. your-username/sec-10k-chunks")
    parser.add_argument("--file", type=Path, default=Path("data/chunks_1600.jsonl"))
    args = parser.parse_args(argv)
    print("uploaded to", push(args.repo_id, args.file))


if __name__ == "__main__":
    main()
