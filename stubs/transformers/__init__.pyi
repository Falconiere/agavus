"""Narrow typed surface used by the optional local experiment and CPU tests."""

from torch import nn

class PretrainedConfig:
  hidden_size: int

class PreTrainedModel(nn.Module):
  config: PretrainedConfig
  def save_pretrained(self, save_directory: str, *, safe_serialization: bool = True) -> None: ...

class AutoModel:
  @classmethod
  def from_pretrained(
    cls,
    pretrained_model_name_or_path: str,
    *,
    local_files_only: bool = False,
    trust_remote_code: bool = False,
  ) -> PreTrainedModel: ...

class BertConfig(PretrainedConfig):
  def __init__(
    self,
    vocab_size: int = 30522,
    hidden_size: int = 768,
    num_hidden_layers: int = 12,
    num_attention_heads: int = 12,
    intermediate_size: int = 3072,
    hidden_dropout_prob: float = 0.1,
    attention_probs_dropout_prob: float = 0.1,
    max_position_embeddings: int = 512,
    pad_token_id: int = 0,
  ) -> None: ...

class BertModel(PreTrainedModel):
  def __init__(self, config: BertConfig) -> None: ...
