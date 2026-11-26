"""Regression coverage for the learned extra's default-import boundary."""

import subprocess
import sys


def test_default_application_imports_without_learned_dependencies() -> None:
  code = """
import importlib.abc
import sys

class OptionalDependencyBlocker(importlib.abc.MetaPathFinder):
  def find_spec(self, fullname, path=None, target=None):
    if fullname.split('.')[0] in {'torch', 'transformers', 'safetensors'}:
      raise ImportError(f'blocked optional dependency: {fullname}')
    return None

sys.meta_path.insert(0, OptionalDependencyBlocker())
import agavus.cli
"""
  result = subprocess.run(
    [sys.executable, "-c", code],
    check=False,
    capture_output=True,
    text=True,
  )
  assert result.returncode == 0, result.stderr
