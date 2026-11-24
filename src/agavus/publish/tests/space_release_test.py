"""Tests for Space staging and upload (synthetic demo tree, mocked hub)."""

from pathlib import Path

import pytest

from agavus.publish import space_release
from agavus.publish.space_release import AGAVUS_REPO, stage_space, upload_space


class FakeApi:
  """Records HfApi calls without touching the network."""

  def __init__(self) -> None:
    self.created: list[tuple[str, str, str | None]] = []
    self.uploaded: list[tuple[str, str, str, dict[str, object]]] = []

  def create_repo(
    self, repo: str, *, repo_type: str, space_sdk: str | None = None, exist_ok: bool = False
  ) -> None:
    self.created.append((repo, repo_type, space_sdk))

  def upload_folder(
    self, *, repo_id: str, folder_path: str, repo_type: str, **kwargs: object
  ) -> None:
    self.uploaded.append((repo_id, folder_path, repo_type, kwargs))


def _demo(tmp_path: Path) -> Path:
  space = tmp_path / "space"
  (space / "assets").mkdir(parents=True)
  (space / "app.py").write_text("print('demo')\n", encoding="utf-8")
  (space / "README.md").write_text("---\nsdk: gradio\n---\n", encoding="utf-8")
  (space / "assets" / "l11340.html").write_text("<p>Art. 1º</p>", encoding="utf-8")
  (space / "tests").mkdir()
  (space / "tests" / "app_test.py").write_text("x = 1\n", encoding="utf-8")
  return space


def test_stage_space_pins_revision(tmp_path: Path) -> None:
  staging = stage_space(_demo(tmp_path), tmp_path / "staging", "v0.1.0")
  requirements = (staging / "requirements.txt").read_text(encoding="utf-8")
  assert requirements == f"agavus @ git+{AGAVUS_REPO}@v0.1.0\ngradio\n"
  assert (staging / "app.py").exists()
  assert (staging / "assets" / "l11340.html").exists()
  assert not (staging / "tests").exists()


def test_stage_space_rejects_invalid_revision(tmp_path: Path) -> None:
  with pytest.raises(ValueError, match="invalid git revision"):
    stage_space(_demo(tmp_path), tmp_path / "staging", "v0.1.0; rm -rf /")


def test_upload_space_creates_gradio_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  staging = stage_space(_demo(tmp_path), tmp_path / "staging", "v0.1.0")
  fake = FakeApi()
  monkeypatch.setattr(space_release, "HfApi", lambda: fake)
  url = upload_space(staging, "Falconiere/agavus")
  assert url == "https://huggingface.co/spaces/Falconiere/agavus"
  assert fake.created == [("Falconiere/agavus", "space", "gradio")]
  assert fake.uploaded == [
    (
      "Falconiere/agavus",
      str(staging),
      "space",
      {
        "ignore_patterns": space_release.IGNORE_PATTERNS,
        "delete_patterns": space_release.DELETE_PATTERNS,
      },
    )
  ]
