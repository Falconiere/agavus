"""Typed surface of onnxruntime used by agavus (the package ships no types)."""

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

class InferenceSession:
  def __init__(self, path_or_bytes: str, *, providers: Sequence[str]) -> None: ...
  def run(
    self, output_names: Sequence[str] | None, input_feed: Mapping[str, NDArray[np.int64]]
  ) -> list[NDArray[np.float32]]: ...
