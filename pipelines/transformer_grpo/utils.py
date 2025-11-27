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
    Compute normalized weights for portfolio construction.

    Args:
        logits: [batch, num_instruments]
        mask: [batch, num_instruments]
        top_k: Target number of stocks
        temperature: Softmax temperature
        min_weight: Minimum weight threshold (inference only)
        differentiable: If True, uses soft attention. If False, uses hard Top-K.

    Returns:
        weights: [batch, num_instruments] Normalized weights summing to 1.
    """
    # Handle 1D input
    squeeze_output = False
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
        mask = mask.unsqueeze(0)
        squeeze_output = True

    # Basic masking
    fill_value = -1e9
    masked_logits = logits.masked_fill(~mask, fill_value)
    
    if differentiable:
        # Training: Softmax attention across ALL valid stocks
        # We generally do NOT enforce strict Top-K here to allow gradient flow to all
        # valid candidates. The temperature controls concentration.
        scores = masked_logits / max(temperature, 1e-4)
        scores = scores - scores.max(dim=-1, keepdim=True).values.detach()
        weights = F.softmax(scores, dim=-1)
        weights = weights * mask.float()
    else:
        # Inference: Hard Top-K selection
        # 1. Calculate probabilities
        scores = masked_logits / max(temperature, 1e-4)
        probs = F.softmax(scores, dim=-1)
        
        # 2. Select Top-K indices
        # Handle edge case where K > valid_count
        valid_count = mask.sum(dim=-1, keepdim=True)
        k_tensor = torch.min(torch.tensor(top_k, device=logits.device), valid_count).long()
        
        # We need a loop or careful broadcasting if K varies per batch (it usually doesn't here)
        # Assuming constant K for simplicity or taking min across batch
        eff_k = min(top_k, logits.size(-1))
        
        topk_vals, topk_inds = torch.topk(probs, eff_k, dim=-1)
        
        # 3. Create hard mask
        hard_weights = torch.zeros_like(probs)
        hard_weights.scatter_(1, topk_inds, topk_vals)
        
        # 4. Apply Min Weight Filter
        if min_weight > 0:
            keep = hard_weights >= min_weight
            # Fallback: if everything filtered, keep max
            fallback = torch.zeros_like(keep).scatter_(1, hard_weights.argmax(dim=-1, keepdim=True), True)
            hard_weights = hard_weights * (keep | fallback).float()
            
        weights = hard_weights

    # Normalize
    sum_w = weights.sum(dim=-1, keepdim=True)
    weights = weights / sum_w.clamp(min=1e-8)
    
    if squeeze_output:
        weights = weights.squeeze(0)
        
    return weights


def sample_topk_actions(
    logits: torch.Tensor,
    mask: torch.Tensor,
    top_k: int,
    group_size: int,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample 'group_size' actions for each batch item using Multinomial sampling.
    Robust version handling K > valid_count.
    """
    batch_size, num_inst = logits.shape
    
    # Flatten for sampling: [batch * group, num_inst]
    logits_expanded = logits.unsqueeze(1).expand(-1, group_size, -1)
    mask_expanded = mask.unsqueeze(1).expand(-1, group_size, -1)
    
    logits_flat = logits_expanded.reshape(-1, num_inst)
    mask_flat = mask_expanded.reshape(-1, num_inst)
    
    masked_logits = logits_flat.masked_fill(~mask_flat, -1e9)
    scores = masked_logits / max(temperature, 1e-4)
    probs = F.softmax(scores, dim=-1)
    
    # Handle variable valid counts across batch
    # Since torch.multinomial requires fixed num_samples, we set K = min(top_k, min_valid_count)
    # But wait, if one batch has 5 valid and another 100, we shouldn't limit both to 5.
    # Solution: Loop or just assume data is dense enough (usually > 300 stocks).
    # Given Qlib constraint "min_instruments=30", we are safe if top_k=10.
    # But to be safe, we cap K.
    
    valid_counts = mask_flat.sum(dim=1)
    min_valid = valid_counts.min().item()
    effective_k = min(top_k, int(min_valid))
    
    if effective_k < 1:
        # Should not happen if min_instruments >= 1
        # Fallback to taking top 1
        effective_k = 1

    action_indices = torch.multinomial(probs, num_samples=effective_k, replacement=False)
    
    sampled_masks_flat = torch.zeros_like(mask_flat, dtype=torch.bool)
    sampled_masks_flat.scatter_(1, action_indices, True)
    
    # Calculate surrogate log probs
    log_probs_all = F.log_softmax(scores, dim=-1)
    selected_log_probs = torch.gather(log_probs_all, 1, action_indices)
    sample_log_probs = selected_log_probs.sum(dim=-1)
    
    return sampled_masks_flat.view(batch_size, group_size, num_inst), sample_log_probs.view(batch_size, group_size)



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
