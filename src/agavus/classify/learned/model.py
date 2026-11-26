"""Two learned linear heads over raw-token, masked-mean pretrained features."""

import math
from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor, nn

from agavus.classify.learned.types import RELATIONS


def tensors(windows: Sequence[Sequence[int]], device: str) -> tuple[Tensor, Tensor]:
  """Right-pad with a masked token; padding cannot enter the mean."""
  if not windows or any(not row for row in windows):
    raise ValueError("need nonempty windows")
  width = max(map(len, windows))
  ids = torch.zeros((len(windows), width), dtype=torch.long, device=device)
  mask = torch.zeros_like(ids)
  for index, row in enumerate(windows):
    ids[index, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
    mask[index, : len(row)] = 1
  return ids, mask


class LearnedModel(nn.Module):
  """Fixed inference weights; only the explicit training command updates them."""

  def __init__(self, encoder: nn.Module, hidden_size: int, classes: tuple[str, ...]) -> None:
    super().__init__()
    self.encoder = encoder
    self.classes = classes
    self.area = nn.Linear(hidden_size, len(classes))
    self.relation = nn.Linear(hidden_size, len(RELATIONS))

  def features(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
    """Masked mean pooling shared by frozen and fine-tuned experiments."""
    hidden = cast(
      tuple[Tensor, ...],
      self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=False),
    )[0]
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

  def logits(self, input_ids: Tensor, attention_mask: Tensor, task: str) -> Tensor:
    """No lexicons, feature crosses or normalization in the learned path."""
    if task not in ("area", "relation"):
      raise ValueError("unknown learned task")
    head = self.area if task == "area" else self.relation
    return cast(Tensor, head(self.features(input_ids, attention_mask)))

  def document_logits(
    self, windows: Sequence[Sequence[int]], task: str, device: str = "cpu"
  ) -> Tensor:
    """Average window logits before temperature scaling, regardless of length."""
    rows = [self.logits(*tensors([row], device), task) for row in windows]
    if not rows:
      raise ValueError("need nonempty windows")
    return torch.cat(rows).mean(dim=0, keepdim=True)

  def probabilities(
    self, windows: Sequence[Sequence[int]], task: str, temperature: float = 1.0, device: str = "cpu"
  ) -> dict[str, float]:
    """Calibrated probabilities only; no generated legal explanation."""
    if not math.isfinite(temperature) or temperature <= 0:
      raise ValueError("temperature must be finite and positive")
    self.eval()
    with torch.no_grad():
      scores = self.document_logits(windows, task, device) / temperature
      probs = torch.softmax(scores, dim=1).cpu().tolist()[0]
    classes = self.classes if task == "area" else RELATIONS
    return {label: float(value) for label, value in zip(classes, probs, strict=True)}
