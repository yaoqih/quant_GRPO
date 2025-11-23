from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        x = x + self.pe[:, :length]
        return self.dropout(x)


class TransformerPolicy(nn.Module):
    """Cross-sectional policy network with a Transformer backbone."""

    def __init__(
        self,
        feature_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "gelu",
        ff_multiplier: int = 4,
        max_positions: int = 1024,
        use_value_head: bool = True,
        temporal_span: int = 1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.temporal_span = max(int(temporal_span), 1)
        self.feature_proj = nn.Linear(feature_dim, d_model)
        self.temporal_encoder = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            batch_first=True,
        )
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_positions, dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.batch_first = True

        encoder_kwargs = dict(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_multiplier * d_model,
            dropout=dropout,
            activation=activation,
        )

        try:
            encoder_layer = nn.TransformerEncoderLayer(batch_first=True, **encoder_kwargs)
        except TypeError:
            # For older PyTorch releases where batch_first is missing.
            self.batch_first = False
            encoder_layer = nn.TransformerEncoderLayer(**encoder_kwargs)

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.policy_head = nn.Linear(d_model, 1)
        self.value_head = nn.Linear(d_model, 1) if use_value_head else None

    def forward(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Parameters
        ----------
        features : torch.Tensor
            Shape (batch_size, instruments, temporal_span, feature_dim) for temporal training,
            or (batch_size, instruments, feature_dim) for legacy single-step inputs.
        mask : torch.Tensor, optional
            Boolean mask with True for valid instruments.
        """
        if features.dim() == 4:
            batch, instruments, timesteps, _ = features.shape
            projected = self.feature_proj(features)
            temporal_input = projected.view(batch * instruments, timesteps, self.d_model)
            temporal_encoded, _ = self.temporal_encoder(temporal_input)
            x = temporal_encoded[:, -1, :].view(batch, instruments, self.d_model)
        else:
            x = self.feature_proj(features)
        x = self.layer_norm(x)
        x = self.pos_encoder(x)

        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask

        if self.batch_first:
            encoded = self.encoder(x, src_key_padding_mask=key_padding_mask)
        else:
            encoded = self.encoder(x.transpose(0, 1), src_key_padding_mask=key_padding_mask).transpose(0, 1)

        logits = self.policy_head(encoded).squeeze(-1)
        values = self.value_head(encoded).squeeze(-1) if self.value_head is not None else None
        return logits, values

    def act(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        temperature: float = 1.0,
        greedy: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, _ = self.forward(features, mask)
        mask_value = torch.finfo(logits.dtype).min
        masked_logits = logits.masked_fill(~mask, mask_value)
        probs = F.softmax(masked_logits / max(temperature, 1e-4), dim=-1)
        if greedy:
            action = probs.argmax(dim=-1, keepdim=True)
        else:
            action = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(probs.size(0), 1)
        return action.squeeze(-1), probs, logits
