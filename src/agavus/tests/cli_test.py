"""Tests for the CLI (tests folder; ingestion mocked, no network)."""

import json
from pathlib import Path

import pytest

from agavus import cli
from agavus.answer.compose import Answer
from agavus.corpus.batch import IngestBatch
from agavus.corpus.fetch import FetchError
from agavus.evals.run import EvalReport


class FakeModel:
  classes = ("F", "M", "O")


class FakeClassifier:
  model = FakeModel()

  def predict_proba(self, texts: list[str]) -> list[list[float]]:
    return [[0.3, 0.3, 0.4]]

  def top_features(self, text: str, k: int = 5) -> list[tuple[str, float]]:
    return [("protetiva", 1.0)]


def test_default_greets(capsys: pytest.CaptureFixture[str]) -> None:
  assert cli.main([]) == 0
  assert "Hello, world!" in capsys.readouterr().out


def test_ingest_datajud_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
  called: list[int] = []

  def fake(settings: object, limit: int, pages: int | None = None) -> IngestBatch:
    called.append(limit)
    assert pages is None
    return IngestBatch(3, False, "start")

  monkeypatch.setattr(cli, "ingest_datajud", fake)
  assert cli.main(["ingest", "--source", "datajud", "--limit", "7"]) == 0
  assert called == [7]


def test_ingest_statutes_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(cli, "ingest_statutes", lambda settings: 5)
  assert cli.main(["ingest", "--source", "statutes"]) == 0


def test_ingest_requires_source() -> None:
  with pytest.raises(SystemExit):
    cli.main(["ingest"])


def test_ingest_fetch_error_is_friendly(
  monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  def failing(settings: object, limit: int, pages: int | None = None) -> IngestBatch:
    raise FetchError("HTTP 429")

  monkeypatch.setattr(cli, "ingest_datajud", failing)
  assert cli.main(["ingest", "--source", "datajud"]) == 1
  assert "nothing saved" in capsys.readouterr().out


def _run_with_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, argv: list[str]) -> int:
  monkeypatch.setenv("AGAVUS_DATA_DIR", str(tmp_path))
  return cli.main(argv)


def test_label_empty_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  assert _run_with_data_dir(monkeypatch, tmp_path, ["label"]) == 0
  assert (tmp_path / "processed" / "corpus.db").exists()


def test_train_seeds_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  assert _run_with_data_dir(monkeypatch, tmp_path, ["train"]) == 0
  assert (tmp_path / "models" / "models.db").exists()


def test_classify_without_model_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  assert _run_with_data_dir(monkeypatch, tmp_path, ["classify", "texto"]) == 1


def _prepare_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  (tmp_path / "models").mkdir()
  (tmp_path / "models" / "models.db").touch()
  monkeypatch.setattr(cli, "load_model", lambda path, name: FakeClassifier())


def test_ask_exit_codes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  _prepare_model(tmp_path, monkeypatch)
  answers = {
    "accept-M": Answer("accept-M", "resposta", ("fonte",)),
    "defer": Answer("defer", "clarify", ()),
    "reject": Answer("reject", "fora", ()),
  }
  for decision, expected_code in (("accept-M", 0), ("defer", 10), ("reject", 20)):
    monkeypatch.setattr(cli, "compose", lambda q, p, s, a=answers[decision]: a)
    assert _run_with_data_dir(monkeypatch, tmp_path, ["-p", "pergunta"]) == expected_code


def test_ask_json_output(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  _prepare_model(tmp_path, monkeypatch)
  monkeypatch.setattr(
    cli, "compose", lambda q, p, s: Answer("accept-M", "resposta", ("Lei 11.340/2006, art. 22",))
  )
  code = _run_with_data_dir(monkeypatch, tmp_path, ["-p", "pergunta", "--json", "--explain"])
  assert code == 0
  payload = json.loads(capsys.readouterr().out)
  assert payload["decision"] == "accept-M"
  assert payload["area"] == "M"
  assert payload["sources"] == [{"ref": "Lei 11.340/2006, art. 22"}]
  assert "trace" in payload


def _fake_report() -> EvalReport:
  return EvalReport(
    accuracy=1.0,
    macro_f1=1.0,
    accepted=2,
    accepted_correct=2,
    deferred=1,
    rejected=1,
    precision_on_accepted=1.0,
    coverage=0.5,
    m_recall=1.0,
  )


def test_publish_dry_run(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  def fake_stage(settings: object, staging: Path) -> EvalReport:
    return _fake_report()

  monkeypatch.setattr(cli, "stage_release", fake_stage)
  assert _run_with_data_dir(monkeypatch, tmp_path, ["publish"]) == 0
  assert "dry run" in capsys.readouterr().out


def test_publish_bootstrap_and_uploads(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  calls: list[tuple[str, object]] = []

  def fake_bootstrap(settings: object, assets: Path) -> Path:
    calls.append(("bootstrap", assets))
    return tmp_path / "models" / "models.db"

  def fake_stage(settings: object, staging: Path) -> EvalReport:
    calls.append(("stage", staging))
    return _fake_report()

  def fake_upload(staging: Path, repo: str) -> str:
    calls.append(("model", repo))
    return "https://huggingface.co/model"

  def fake_stage_space(space_dir: Path, staging: Path, revision: str) -> Path:
    calls.append(("space-stage", revision))
    return staging

  def fake_upload_space(staging: Path, repo: str) -> str:
    calls.append(("space", repo))
    return "https://huggingface.co/spaces/demo"

  monkeypatch.setattr(cli, "bootstrap_corpus", fake_bootstrap)
  monkeypatch.setattr(cli, "stage_release", fake_stage)
  monkeypatch.setattr(cli, "upload", fake_upload)
  monkeypatch.setattr(cli, "stage_space", fake_stage_space)
  monkeypatch.setattr(cli, "upload_space", fake_upload_space)
  assets = tmp_path / "assets"
  code = _run_with_data_dir(
    monkeypatch,
    tmp_path,
    [
      "publish",
      "--bootstrap",
      str(assets),
      "--repo",
      "owner/model",
      "--space",
      "owner/space",
      "--revision",
      "v1.0.0",
    ],
  )
  assert code == 0
  assert calls == [
    ("bootstrap", assets),
    ("stage", tmp_path / "hf-staging"),
    ("model", "owner/model"),
    ("space-stage", "v1.0.0"),
    ("space", "owner/space"),
  ]
  output = capsys.readouterr().out
  assert "model live" in output
  assert "Space live" in output


def test_publish_bootstrap_failure(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  def failing(settings: object, assets: Path) -> Path:
    raise ValueError("non-public texts")

  monkeypatch.setattr(cli, "bootstrap_corpus", failing)
  code = _run_with_data_dir(monkeypatch, tmp_path, ["publish", "--bootstrap", "assets"])
  assert code == 1
  assert "bootstrap failed" in capsys.readouterr().out
