"""Transformer-based policy network for cross-sectional stock selection."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class BottleneckFeatureExtractor(nn.Module):
    """GLU + SeLU selection block to sparsify useless factors before temporal modeling."""

    def __init__(self, feature_dim: int, d_model: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden_dim = max(d_model * expansion, d_model)
        self.input_linear = nn.Linear(feature_dim, hidden_dim * 2)
        self.glu = nn.GLU(dim=-1)
        self.activation = nn.SELU()
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_dim, d_model)
        self.residual = nn.Linear(feature_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = self.glu(self.input_linear(x))
        activated = self.activation(gated)
        transformed = self.proj(self.dropout(activated))
        return transformed + self.residual(x)

    def l1_penalty(self) -> torch.Tensor:
        penalty = self.input_linear.weight.abs().sum()
        penalty = penalty + self.proj.weight.abs().sum()
        penalty = penalty + self.residual.weight.abs().sum()
        return penalty


class GRUTemporalEncoder(nn.Module):
    """GRU encoder with attention pooling to retain the most informative day in the window."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
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
        outputs, _ = self.gru(x)
        attn_score = self.attn(outputs)  # [B*T, window, 1]
        attn_weights = torch.softmax(attn_score, dim=1)
        pooled = (outputs * attn_weights).sum(dim=1)
        return self.norm(self.dropout(pooled))


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
        nhead: int = 2,
        num_layers: int = 2,
        dropout: float = 0.1,
        ff_multiplier: int = 4,
        temporal_span: int = 1,
        temporal_nhead: int = 4,  # 兼容旧配置，保持接口不报错
        temporal_layers: int = 1,
        noise_std: float = 0.0,
        **_kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.temporal_span = max(int(temporal_span), 1)
        self.noise_std = float(noise_std)
        
        # Feature normalization on the last dimension only (insensitive to padding)
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.feature_extractor = BottleneckFeatureExtractor(feature_dim, d_model, dropout=dropout)

        # 20日以内短序列用GRU更稳定
        self.temporal_encoder = GRUTemporalEncoder(
            input_dim=d_model,
            hidden_dim=d_model,
            num_layers=temporal_layers,
            dropout=dropout,
        )

        # 市场上下文驱动的受限注意力
        self.layer_norm = nn.LayerNorm(d_model)
        self.market_token = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        fusion_dim = d_model * 2
        self.stock_fusion = nn.Sequential(
            nn.Linear(fusion_dim, ff_multiplier * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_multiplier * d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # 输出头
        self.policy_head = nn.Linear(d_model, 1)
        self.logit_scale = nn.Parameter(torch.zeros(1))
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
            gated = self.feature_extractor(normed_features)
            temporal_input = gated.reshape(batch * instruments, timesteps, self.d_model)
            temporal_out = self.temporal_encoder(temporal_input)
            x = temporal_out.view(batch, instruments, self.d_model)
        else:
            batch, instruments, _ = features.shape
            normed_features = self.feature_norm(features)
            x = self.feature_extractor(normed_features)

        x = self.layer_norm(x)

        if mask is None:
            mask = torch.ones(batch, instruments, dtype=torch.bool, device=x.device)
        mask = mask.to(dtype=torch.bool)

        mask_float = mask.float()
        valid_count = mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)
        market_context = (x * mask_float.unsqueeze(-1)).sum(dim=1, keepdim=True) / valid_count.unsqueeze(-1)
        market_context = self.market_token(market_context)

        padding_mask = ~mask
        context, _ = self.cross_attention(
            query=market_context,
            key=x,
            value=x,
            key_padding_mask=padding_mask,
        )
        context = context.expand(-1, instruments, -1)
        fusion_input = torch.cat([x, context], dim=-1)
        stock_embeddings = self.stock_fusion(fusion_input)

        if self.training and self.noise_std > 0:
            noise = torch.randn_like(stock_embeddings) * self.noise_std
            stock_embeddings = stock_embeddings + noise

        stock_embeddings = stock_embeddings * mask.unsqueeze(-1)
        logits = self.policy_head(stock_embeddings).squeeze(-1)
        logits = logits * torch.exp(self.logit_scale)
        return logits, None

    def feature_l1_penalty(self) -> torch.Tensor:
        return self.feature_extractor.l1_penalty()
