import pytest

from src.hub.push_dataset import push


class FakeApi:
    def __init__(self):
        self.calls = []

    def create_repo(self, **kwargs):
        self.calls.append(("create_repo", kwargs))

    def upload_file(self, **kwargs):
        self.calls.append(("upload_file", kwargs))


def test_push_creates_a_private_dataset_repo_and_uploads(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "dummy")
    f = tmp_path / "chunks_1600.jsonl"
    f.write_text("{}\n")
    api = FakeApi()
    url = push("me/sec-chunks", f, api=api)
    assert url.endswith("/datasets/me/sec-chunks")
    assert api.calls[0][1]["private"] is True and api.calls[0][1]["repo_type"] == "dataset"
    assert api.calls[1][1]["path_in_repo"] == "chunks_1600.jsonl"


def test_push_requires_a_token(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError):
        push("me/x", tmp_path / "f.jsonl", api=FakeApi())
