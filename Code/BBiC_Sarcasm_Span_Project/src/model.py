"""BBiC architecture and supported architectural ablations."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

from .crf import LinearChainCRF
from .spans import LABELS, LABEL2ID

VARIANTS = {
    "bbic": (True, True),
    "bert_bilstm": (True, False),
    "bert_crf": (False, True),
    "bert_only": (False, False),
}


class BBiCModel(nn.Module):
    """BERT -> optional BiLSTM -> linear emissions -> optional CRF.

    Main BBiC variant uses both BiLSTM and CRF. Other variants support Table 10
    ablations. A local encoder config is used when loading a saved state dict so
    that the fine-tuned encoder weights are restored from the checkpoint itself.
    """

    def __init__(
        self,
        base_model: str = "bert-base-uncased",
        bilstm_hidden_size: int = 256,
        dropout: float = 0.30,
        variant: str = "bbic",
        encoder_config_path: str | None = None,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
        try:
            from transformers import AutoConfig, AutoModel
        except ImportError as exc:
            raise ImportError(
                "BBiC requires Hugging Face Transformers. Install requirements.txt first."
            ) from exc

        self.variant = variant
        self.use_bilstm, self.use_crf = VARIANTS[variant]
        self.base_model_name = base_model
        if pretrained:
            self.encoder = AutoModel.from_pretrained(base_model)
        else:
            if not encoder_config_path:
                raise ValueError("encoder_config_path is required for pretrained=False")
            encoder_config = AutoConfig.from_pretrained(encoder_config_path, local_files_only=True)
            self.encoder = AutoModel.from_config(encoder_config)
        encoder_dim = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(float(dropout))
        if self.use_bilstm:
            self.bilstm = nn.LSTM(
                input_size=encoder_dim,
                hidden_size=int(bilstm_hidden_size),
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            emission_dim = 2 * int(bilstm_hidden_size)
        else:
            self.bilstm = None
            emission_dim = encoder_dim
        self.classifier = nn.Linear(emission_dim, len(LABELS))
        self.crf = LinearChainCRF(len(LABELS), list(LABELS)) if self.use_crf else None

    def emissions(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask.long())
        hidden = self.dropout(encoded.last_hidden_state)
        if self.bilstm is not None:
            # Packing prevents batch-padding tokens from influencing the backward LSTM states.
            lengths = attention_mask.long().sum(dim=1).detach().cpu()
            packed = pack_padded_sequence(hidden, lengths, batch_first=True, enforce_sorted=False)
            packed_output, _ = self.bilstm(packed)
            hidden, _ = pad_packed_sequence(
                packed_output, batch_first=True, total_length=attention_mask.shape[1]
            )
            hidden = self.dropout(hidden)
        return self.classifier(hidden)

    def loss(self, emissions: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.crf is not None:
            return self.crf.neg_log_likelihood(emissions, labels, mask)
        # Independent token classification for non-CRF ablations; ignore padding.
        active_emissions = emissions[mask]
        active_labels = labels[mask]
        return F.cross_entropy(active_emissions, active_labels)

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        if self.crf is not None:
            return self.crf.decode(emissions, mask)
        paths: list[list[int]] = []
        for b in range(emissions.shape[0]):
            length = int(mask[b].sum().item())
            paths.append(emissions[b, :length].argmax(dim=-1).tolist())
        return paths

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        emission_scores = self.emissions(input_ids, attention_mask)
        loss_value = self.loss(emission_scores, labels, attention_mask.bool()) if labels is not None else None
        return {"loss": loss_value, "emissions": emission_scores}
