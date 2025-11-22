from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch

from .data_pipeline import DailyBatchDataset
from .model import TransformerPolicy
from .utils import compute_performance


def run_policy_on_dataset(
    model: TransformerPolicy,
    dataset: DailyBatchDataset,
    device: torch.device,
    temperature: float = 1.0,
    greedy: bool = True,
    commission: float = 0.0,
    slippage: float = 0.0,
) -> pd.DataFrame:
    model.eval()
    records = []
    nav = 1.0
    prev_instrument = None

    with torch.no_grad():
        for batch in dataset:
            features = torch.from_numpy(batch.features).unsqueeze(0).to(device)
            mask = torch.ones(1, batch.features.shape[0], dtype=torch.bool, device=device)
            action, probs, logits = model.act(features, mask, temperature=temperature, greedy=greedy)
            idx = int(action.item())
            raw_reward = float(batch.rewards[idx])
            instrument = batch.instruments[idx]
            trade_cost = 0.0
            if prev_instrument is None or instrument != prev_instrument:
                trade_cost += (commission + slippage)
            net_reward = raw_reward - trade_cost
            nav *= 1.0 + net_reward
            prev_instrument = instrument
            records.append(
                {
                    "date": batch.date,
                    "instrument": instrument,
                    "raw_reward": raw_reward,
                    "cost": trade_cost,
                    "reward": net_reward,
                    "prob": float(probs[0, idx].item()),
                    "logit": float(logits[0, idx].item()),
                    "nav": nav,
                }
            )
    trades = pd.DataFrame.from_records(records)
    return trades


def run_backtest(
    model: TransformerPolicy,
    dataset: DailyBatchDataset,
    device: torch.device,
    risk_free: float = 0.02,
    temperature: float = 1.0,
    greedy: bool = True,
    commission: float = 0.0,
    slippage: float = 0.0,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    trades = run_policy_on_dataset(
        model=model,
        dataset=dataset,
        device=device,
        temperature=temperature,
        greedy=greedy,
        commission=commission,
        slippage=slippage,
    )
    summary = compute_performance(trades, risk_free=risk_free)
    return trades, summary


def save_trades(trades: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(path, index=False)
