"""Space staging and upload for the public demo (docs/publishing.md).

Staging pins the engine to the released git ref, so the Space app, the
installed package, and the model artifact always match. Every staging tree
passes the pre-upload privacy scan.
"""

import re
import shutil
from pathlib import Path

from huggingface_hub import HfApi

from agavus.publish.privacy import ensure_no_personal_data

AGAVUS_REPO = "https://github.com/Falconiere/agavus.git"
SPACE_FILES = ("app.py", "README.md")
REVISION_RE = re.compile(r"[A-Za-z0-9._/-]+")
IGNORE_PATTERNS = ["__pycache__", "*.pyc", ".pytest_cache", "tests"]


def stage_space(space_dir: Path, staging: Path, revision: str) -> Path:
  """Copy the demo into a release folder with a pinned engine requirement."""
  if REVISION_RE.fullmatch(revision) is None:
    raise ValueError(f"invalid git revision: {revision!r}")
  if staging.exists():
    shutil.rmtree(staging)
  staging.mkdir(parents=True)
  for name in SPACE_FILES:
    shutil.copyfile(space_dir / name, staging / name)
  shutil.copytree(space_dir / "assets", staging / "assets")
  requirements = f"agavus @ git+{AGAVUS_REPO}@{revision}\ngradio\n"
  (staging / "requirements.txt").write_text(requirements, encoding="utf-8")
  ensure_no_personal_data(staging)
  return staging


def upload_space(staging: Path, repo: str) -> str:
  """Create/update the Gradio Space and upload the staged demo."""
  ensure_no_personal_data(staging)
  api = HfApi()
  api.create_repo(repo, repo_type="space", space_sdk="gradio", exist_ok=True)
  api.upload_folder(
    repo_id=repo,
    folder_path=str(staging),
    repo_type="space",
    ignore_patterns=IGNORE_PATTERNS,
  )
  return f"https://huggingface.co/spaces/{repo}"
