"""Transformer-based policy network for cross-sectional stock selection."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """标准正弦位置编码，仅用于时序编码器。"""

    def __init__(self, d_model: int, max_len: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        pe = self._generate_pe(max_len)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def _generate_pe(self, length: int) -> torch.Tensor:
        position = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(length, self.d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        if length > self.pe.size(1):
            new_pe = self._generate_pe(length).unsqueeze(0).to(x.device)
            self.pe = new_pe
        pe_slice = self.pe[:, :length]
        if pe_slice.device != x.device:
            pe_slice = pe_slice.to(x.device)
        return self.dropout(x + pe_slice)


class TemporalEncoder(nn.Module):
    """时序注意力编码器，使用CLS token聚合。"""

    def __init__(self, d_model: int, nhead: int = 4, num_layers: int = 1, dropout: float = 0.1, max_len: int = 64):
        super().__init__()
        self.d_model = d_model
        self.temporal_pe = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.attention = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch * instruments, temporal_span, d_model]
        Returns:
            [batch * instruments, d_model]
        """
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.temporal_pe(x)
        x = self.attention(x)
        return self.norm(x[:, 0, :])


class TransformerPolicy(nn.Module):
    """
    跨截面策略网络。

    关键修复：移除了截面层的Positional Encoding。
    原因：股票之间没有固定顺序，添加PE会引入错误的归纳偏置。
    """

    def __init__(
        self,
        feature_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        ff_multiplier: int = 4,
        temporal_span: int = 1,
        temporal_nhead: int = 4,
        temporal_layers: int = 1,
        **_kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.temporal_span = max(int(temporal_span), 1)
        
        # Feature normalization on the last dimension only (insensitive to padding)
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.feature_proj = nn.Linear(feature_dim, d_model)

        # 时序编码器（有位置编码，因为时间有顺序）
        self.temporal_encoder = TemporalEncoder(
            d_model=d_model, nhead=temporal_nhead, num_layers=temporal_layers,
            dropout=dropout, max_len=max(temporal_span + 1, 64),
        )

        # 截面编码器（无位置编码，因为股票没有固定顺序）
        self.layer_norm = nn.LayerNorm(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_multiplier * d_model,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出头
        self.policy_head = nn.Linear(d_model, 1)
        self._init_weights()
    def _init_weights(self):
        """Xavier 初始化，有助于模型快速收敛"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, None]:
        """
        Args:
            features: [batch, instruments, temporal_span, feature_dim] 或 [batch, instruments, feature_dim]
            mask: [batch, instruments] 有效股票的布尔掩码

        Returns:
            logits: [batch, instruments] 每只股票的得分
            None: 兼容性占位符
        """
        if features.dim() == 4:
            batch, instruments, timesteps, _ = features.shape

            normed_features = self.feature_norm(features)
            projected = self.feature_proj(normed_features)
            temporal_input = projected.view(batch * instruments, timesteps, self.d_model)
            temporal_out = self.temporal_encoder(temporal_input)
            x = temporal_out.view(batch, instruments, self.d_model)
        else:
            normed_features = self.feature_norm(features)
            x = self.feature_proj(normed_features)

        x = self.layer_norm(x)

        # 注意：这里不添加positional encoding！
        # 股票之间没有顺序关系，PE会引入错误的归纳偏置

        key_padding_mask = ~mask if mask is not None else None
        encoded = self.encoder(x, src_key_padding_mask=key_padding_mask)

        logits = self.policy_head(encoded).squeeze(-1)
        return logits, None
