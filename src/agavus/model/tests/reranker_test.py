"""The pinned reranker: absent weights mean no reranker; the download is pinned and idempotent."""

from pathlib import Path

import pytest

from agavus.model import reranker
from agavus.model.reranker import MODEL_FILE, REVISION, TOKENIZER_FILE, load_reranker, reranker_dir


def test_missing_weights_mean_no_reranker(tmp_path: Path) -> None:
  assert load_reranker(tmp_path) is None
  (reranker_dir(tmp_path) / "onnx").mkdir(parents=True)
  (reranker_dir(tmp_path) / MODEL_FILE).write_bytes(b"")
  assert load_reranker(tmp_path) is None  # the tokenizer is still missing


def test_download_fetches_the_pinned_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  calls: list[tuple[str, str, str, Path]] = []

  def fake(repo: str, name: str, *, revision: str, local_dir: Path) -> str:
    calls.append((repo, name, revision, local_dir))
    return str(local_dir / name)

  monkeypatch.setattr(reranker, "hf_hub_download", fake)
  folder = reranker.download_reranker(tmp_path)
  assert folder == reranker_dir(tmp_path) and folder.name == REVISION
  assert [(name, revision) for _, name, revision, _ in calls] == [
    (MODEL_FILE, REVISION),
    (TOKENIZER_FILE, REVISION),
  ]
