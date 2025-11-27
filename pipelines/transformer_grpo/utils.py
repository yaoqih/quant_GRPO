"""Utility functions for transformer GRPO pipeline."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple
import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .data_pipeline import DailyBatch


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def compute_topk_weights(
    logits: torch.Tensor,
    mask: torch.Tensor,
    top_k: int,
    temperature: float = 1.0,
    min_weight: float = 0.0,
    differentiable: bool = True,
) -> torch.Tensor:
    """
    计算 top-k 权重。

    关键修复：训练时使用完整的softmax让梯度流动，
    而不是用detach切断梯度。

    Args:
        logits: [batch, num_instruments] 或 [num_instruments]
        mask: [batch, num_instruments] 或 [num_instruments]
        top_k: 选择前 k 个股票
        temperature: softmax 温度
        min_weight: 最小权重阈值（仅在推理时生效）
        differentiable: True=训练模式，False=推理模式

    Returns:
        weights: 归一化的权重
    """
    squeeze_output = False
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
        mask = mask.unsqueeze(0)
        squeeze_output = True

    # 清理NaN/Inf
    logits = torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits))

    # Mask无效位置
    fill_value = -1e9
    masked_logits = logits.masked_fill(~mask, fill_value)

    # 温度缩放
    effective_temp = max(temperature, 1e-4)
    scaled = masked_logits / effective_temp

    # 数值稳定的softmax
    scaled_max = scaled.max(dim=-1, keepdim=True).values
    scaled = scaled - scaled_max.detach()

    if differentiable:
        # ====== 训练模式 ======
        # 直接使用softmax，让梯度完整流动到所有位置
        # 不做hard top-k选择，避免切断梯度
        probs = F.softmax(scaled, dim=-1)
        weights = probs * mask.float()
    else:
        # ====== 推理模式 ======
        # 使用hard top-k选择
        probs = F.softmax(scaled, dim=-1) * mask.float()
        k = min(top_k, logits.size(-1))

        if k > 0 and k < logits.size(-1):
            topk_values, topk_indices = torch.topk(probs, k, dim=-1)
            weights = torch.zeros_like(probs)
            weights.scatter_(1, topk_indices, topk_values)
        else:
            weights = probs

        # min_weight过滤
        if min_weight > 0:
            keep_mask = weights >= min_weight
            if not keep_mask.any(dim=-1).all():
                # 如果没有权重超过阈值，保留最大的
                max_idx = weights.argmax(dim=-1, keepdim=True)
                fallback_mask = torch.zeros_like(keep_mask).scatter_(1, max_idx, True)
                keep_mask = keep_mask | fallback_mask
            weights = weights * keep_mask.float()

    # 归一化
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    # 最终检查
    weights = torch.where(torch.isfinite(weights), weights, torch.zeros_like(weights))

    if squeeze_output:
        weights = weights.squeeze(0)

    return weights


def collate_daily_batches(samples: List[DailyBatch]) -> Dict[str, object]:
    """将多个DailyBatch合并为一个batch。"""
    if not samples:
        raise ValueError("Empty batch.")

    batch_size = len(samples)
    sample_shapes = [_infer_sample_shape(sample) for sample in samples]
    max_tokens = max(shape[0] for shape in sample_shapes)
    first_shape = sample_shapes[0]

    if len(first_shape) == 3:
        temporal_span = first_shape[1]
        feature_dim = first_shape[2]
    else:
        temporal_span = 1
        feature_dim = first_shape[1]

    features = torch.zeros(batch_size, max_tokens, temporal_span, feature_dim, dtype=torch.float32)
    rewards = torch.zeros(batch_size, max_tokens, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_tokens, dtype=torch.bool)
    metadata: List[Dict[str, object]] = []

    for idx, sample in enumerate(samples):
        feat, rew = sample.materialize()
        length = feat.shape[0]
        if feat.ndim == 2:
            feat = feat[:, np.newaxis, :]
        features[idx, :length] = torch.from_numpy(feat.astype(np.float32, copy=False))
        rewards[idx, :length] = torch.from_numpy(rew.astype(np.float32, copy=False))
        mask[idx, :length] = True
        metadata.append({"date": sample.date, "instruments": sample.instruments})

    return {
        "features": features,
        "rewards": rewards,
        "mask": mask,
        "meta": metadata,
    }


def _infer_sample_shape(sample: DailyBatch) -> Tuple[int, ...]:
    if sample.feature_shape:
        return sample.feature_shape
    if sample.features is not None:
        return sample.features.shape
    feat, _ = sample.materialize()
    return feat.shape


def compute_performance(trades: pd.DataFrame, risk_free: float = 0.02) -> Dict[str, float]:
    """计算回测性能指标。"""
    if trades.empty:
        return {
            "cumulative_return": 0.0,
            "annual_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "hit_ratio": 0.0,
            "avg_daily_return": 0.0,
            "switch_ratio": 0.0,
            "avg_turnover": 0.0,
        }

    daily_returns = trades["reward"].to_numpy()
    nav = trades["nav"].to_numpy() if "nav" in trades else np.cumprod(1.0 + daily_returns)
    total_return = float(nav[-1] - 1.0)
    trading_days = max(len(daily_returns), 1)
    ann_return = float((1.0 + total_return) ** (252.0 / trading_days) - 1.0)
    mean_daily = float(np.mean(daily_returns))
    std_daily = float(np.std(daily_returns) + 1e-9)
    sharpe = ((mean_daily - risk_free / 252.0) / std_daily) * math.sqrt(252.0)

    running_max = np.maximum.accumulate(nav)
    drawdown = (running_max - nav) / running_max
    max_dd = float(np.max(drawdown))

    hit_ratio = float((daily_returns > 0).mean())
    avg_turnover = float(trades["turnover"].mean()) if "turnover" in trades else 0.0

    return {
        "cumulative_return": total_return,
        "annual_return": ann_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "hit_ratio": hit_ratio,
        "avg_daily_return": mean_daily,
        "switch_ratio": avg_turnover,
        "avg_turnover": avg_turnover,
    }
