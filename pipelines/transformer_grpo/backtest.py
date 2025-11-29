"""Backtest module for evaluating trained policies."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from .data_pipeline import DailyBatchDataset
from .model import TransformerPolicy
from .utils import compute_performance, compute_topk_weights


def run_backtest(
    model: TransformerPolicy,
    dataset: DailyBatchDataset,
    device: torch.device,
    risk_free: float = 0.02,
    top_k: int = 1,
    temperature: float = 1.0,
    min_weight: float = 0.0,
    commission: float = 0.0,
    slippage: float = 0.0,
    **_kwargs,  # 忽略其他参数保持兼容性
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Run backtest on dataset using the trained model.

    Returns:
        trades: DataFrame with daily trading records
        summary: Performance metrics dictionary
    """
    model.eval()
    records = []
    nav = 1.0
    prev_weights: Optional[Dict[str, float]] = None
    per_unit_cost = commission + slippage

    with torch.no_grad():
        for batch in dataset:
            feat_np, reward_np = batch.materialize()
            features = torch.from_numpy(feat_np.astype(np.float32, copy=False)).unsqueeze(0).to(device)
            token_count = feat_np.shape[0]
            mask = torch.ones(1, token_count, dtype=torch.bool, device=device)

            # 获取模型 logits 并计算权重
            logits, _ = model(features, mask)
            weights_tensor = compute_topk_weights(
                logits=logits[0], mask=mask[0], top_k=top_k,
                temperature=temperature, min_weight=min_weight, differentiable=False,
            )

            # 转换为字典格式
            weights_np = weights_tensor.detach().cpu().numpy()[:token_count]
            effective_weights = np.where(weights_np > 1e-6, weights_np, 0.0)
            rewards_np = reward_np[:token_count]
            weights = {
                str(inst): float(weights_np[idx])
                for idx, inst in enumerate(batch.instruments[:token_count])
                if weights_np[idx] > 1e-6
            }

            # 计算收益和成本
            weighted_reward = float(np.dot(effective_weights, rewards_np))

            # 计算换手
            inst_list = [str(inst) for inst in batch.instruments[:token_count]]
            inst_set = set(inst_list)
            if prev_weights is None:
                turnover = float(np.abs(effective_weights).sum())
            else:
                prev_on_curr = np.array([prev_weights.get(inst, 0.0) for inst in inst_list], dtype=np.float32)
                shared_turnover = float(np.abs(effective_weights - prev_on_curr).sum())
                missing_prev = float(
                    sum(abs(prev_weights.get(inst, 0.0)) for inst in prev_weights.keys() - inst_set)
                )
                turnover = shared_turnover + missing_prev

            trade_cost = per_unit_cost * turnover
            net_reward = weighted_reward - trade_cost
            nav *= 1.0 + net_reward
            prev_weights = weights

            records.append({
                "date": batch.date,
                "instrument": ";".join(f"{k}:{v:.3f}" for k, v in sorted(weights.items())),
                "raw_reward": weighted_reward,
                "cost": trade_cost,
                "reward": net_reward,
                "nav": nav,
                "turnover": turnover,
            })

    trades = pd.DataFrame.from_records(records)
    summary = compute_performance(trades, risk_free=risk_free)
    return trades, summary


def save_trades(trades: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(path, index=False)
