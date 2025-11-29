"""Transformer-based policy network for cross-sectional stock selection."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class GatedFeatureExtractor(nn.Module):
    """在进入主干网络前抑制冗余特征噪声。"""

    def __init__(self, feature_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.linear_val = nn.Linear(feature_dim, d_model)
        self.linear_gate = nn.Linear(feature_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values = self.linear_val(x)
        gate = self.sigmoid(self.linear_gate(x))
        return self.dropout(values * gate)


class GRUTemporalEncoder(nn.Module):
    """针对短时序的轻量级编码器，天然具备时间先验。"""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch * instruments, temporal_span, input_dim]
        Returns:
            [batch * instruments, hidden_dim]
        """
        self.gru.flatten_parameters()
        _, h_n = self.gru(x)
        last_hidden = h_n[-1]
        return self.norm(self.dropout(last_hidden))


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
        temporal_nhead: int = 4,  # 兼容旧配置，保持接口不报错
        temporal_layers: int = 1,
        **_kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.temporal_span = max(int(temporal_span), 1)
        
        # Feature normalization on the last dimension only (insensitive to padding)
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.feature_gating = GatedFeatureExtractor(feature_dim, d_model, dropout=dropout)

        # 20日以内短序列用GRU更稳定
        self.temporal_encoder = GRUTemporalEncoder(
            input_dim=d_model,
            hidden_dim=d_model,
            num_layers=temporal_layers,
            dropout=dropout,
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
            gated = self.feature_gating(normed_features)
            temporal_input = gated.reshape(batch * instruments, timesteps, self.d_model)
            temporal_out = self.temporal_encoder(temporal_input)
            x = temporal_out.view(batch, instruments, self.d_model)
        else:
            batch, instruments, _ = features.shape
            normed_features = self.feature_norm(features)
            x = self.feature_gating(normed_features)

        x = self.layer_norm(x)

        # 构造市场上下文token，避免噪声股票彼此互扰
        if mask is not None:
            mask = mask.to(dtype=torch.bool)
            mask_float = mask.float()
            valid_count = mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)
            market_context = (x * mask_float.unsqueeze(-1)).sum(dim=1, keepdim=True) / valid_count.unsqueeze(-1)
            padding_mask = torch.cat(
                [torch.zeros(batch, 1, dtype=torch.bool, device=mask.device), ~mask],
                dim=1,
            )
        else:
            market_context = x.mean(dim=1, keepdim=True)
            padding_mask = None

        x_with_context = torch.cat([market_context, x], dim=1)

        key_padding_mask = padding_mask
        encoded = self.encoder(x_with_context, src_key_padding_mask=key_padding_mask)

        stock_embeddings = encoded[:, 1:, :]
        logits = self.policy_head(stock_embeddings).squeeze(-1)
        return logits, None
