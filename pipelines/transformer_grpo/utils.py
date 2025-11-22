from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import torch

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


def collate_daily_batches(samples: List[DailyBatch]) -> Dict[str, object]:
    if not samples:
        raise ValueError("Empty batch.")
    batch_size = len(samples)
    max_tokens = max(s.features.shape[0] for s in samples)
    feature_dim = samples[0].features.shape[1]

    features = torch.zeros(batch_size, max_tokens, feature_dim, dtype=torch.float32)
    rewards = torch.zeros(batch_size, max_tokens, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_tokens, dtype=torch.bool)
    metadata: List[Dict[str, object]] = []

    for idx, sample in enumerate(samples):
        length = sample.features.shape[0]
        features[idx, :length] = torch.from_numpy(sample.features)
        rewards[idx, :length] = torch.from_numpy(sample.rewards)
        mask[idx, :length] = True
        metadata.append(
            {
                "date": sample.date,
                "instruments": sample.instruments,
            }
        )
    return {
        "features": features,
        "rewards": rewards,
        "mask": mask,
        "meta": metadata,
    }


def compute_group_advantage(
    rewards: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 1.0,
    baseline: torch.Tensor | None = None,
) -> torch.Tensor:
    mask_f = mask.float()
    count = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
    if baseline is not None:
        centered = (rewards - baseline) * mask_f
    else:
        mean = (rewards * mask_f).sum(dim=1, keepdim=True) / count
        centered = (rewards - mean) * mask_f
    mean_centered = (centered * mask_f).sum(dim=1, keepdim=True) / count
    centered = (centered - mean_centered) * mask_f
    var = (centered ** 2).sum(dim=1, keepdim=True) / count
    std = torch.sqrt(var + 1e-6)
    adv = centered / std
    if temperature != 1.0:
        adv = adv / max(temperature, 1e-4)
    return adv * mask_f


def compute_performance(trades: pd.DataFrame, risk_free: float = 0.02) -> Dict[str, float]:
    if trades.empty:
        return {
            "cumulative_return": 0.0,
            "annual_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "hit_ratio": 0.0,
            "avg_daily_return": 0.0,
            "switch_ratio": 0.0,
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
    switches = trades["instrument"].ne(trades["instrument"].shift()).sum()
    switch_ratio = float((switches / trading_days) if trading_days else 0.0)

    return {
        "cumulative_return": total_return,
        "annual_return": ann_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "hit_ratio": hit_ratio,
        "avg_daily_return": mean_daily,
        "switch_ratio": switch_ratio,
    }
