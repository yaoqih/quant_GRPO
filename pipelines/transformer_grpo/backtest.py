from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from .data_pipeline import DailyBatchDataset
from .model import TransformerPolicy
from .utils import compute_performance


def _normalize_vector(probs: List[float]) -> List[float]:
    total = float(sum(probs))
    if total <= 1e-8:
        return [1.0 / max(len(probs), 1)] * len(probs)
    return [p / total for p in probs]


def _build_weights(
    inst_probs: Dict[str, float],
    top_k: int,
    min_weight: float,
    prev_weights: Optional[Dict[str, float]],
    hold_threshold: Optional[float],
) -> Tuple[Dict[str, float], bool]:
    if hold_threshold is not None and prev_weights:
        retained_prob = sum(inst_probs.get(inst, 0.0) for inst in prev_weights)
        if retained_prob >= hold_threshold:
            retained = {inst: weight for inst, weight in prev_weights.items() if inst in inst_probs}
            total = sum(retained.values())
            if total > 0:
                retained = {inst: weight / total for inst, weight in retained.items()}
                return retained, True

    sorted_pairs = sorted(inst_probs.items(), key=lambda kv: kv[1], reverse=True)
    k = max(1, min(top_k, len(sorted_pairs)))
    selected = sorted_pairs[:k]
    filtered = {inst: prob for inst, prob in selected if prob >= min_weight}
    if not filtered:
        inst, prob = sorted_pairs[0]
        filtered = {inst: prob}
    norm_weights = _normalize_vector(list(filtered.values()))
    return {inst: weight for (inst, _), weight in zip(filtered.items(), norm_weights)}, False


def _compute_turnover(
    prev_weights: Optional[Dict[str, float]],
    new_weights: Dict[str, float],
) -> float:
    if prev_weights is None:
        return sum(new_weights.values())
    instruments = set(prev_weights) | set(new_weights)
    turnover = 0.0
    for inst in instruments:
        turnover += abs(new_weights.get(inst, 0.0) - prev_weights.get(inst, 0.0))
    return turnover


def run_policy_on_dataset(
    model: TransformerPolicy,
    dataset: DailyBatchDataset,
    device: torch.device,
    temperature: float = 1.0,
    greedy: bool = True,
    commission: float = 0.0,
    slippage: float = 0.0,
    top_k: int = 1,
    hold_threshold: Optional[float] = None,
    min_weight: float = 0.0,
    cash_token: Optional[str] = None,
    max_gross_exposure: float = 1.0,
) -> pd.DataFrame:
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
            action, probs, logits = model.act(features, mask, temperature=temperature, greedy=greedy)
            valid_len = token_count
            probs_np = probs[0, :valid_len].detach().cpu().tolist()
            probs_np = _normalize_vector(probs_np)
            inst_probs = {inst: float(prob) for inst, prob in zip(batch.instruments[:valid_len], probs_np)}
            weights, held_prev = _build_weights(
                inst_probs,
                top_k=top_k,
                min_weight=min_weight,
                prev_weights=prev_weights,
                hold_threshold=hold_threshold,
            )
            rewards_map = {inst: float(reward_np[idx]) for idx, inst in enumerate(batch.instruments[:valid_len])}
            logits_map = {inst: float(logits[0, idx].item()) for idx, inst in enumerate(batch.instruments[:valid_len])}
            total_weight = sum(weights.values())
            limit = max(max_gross_exposure, 1e-6)
            if total_weight > limit:
                scale = limit / total_weight
                weights = {inst: weight * scale for inst, weight in weights.items()}
                total_weight = limit
            if cash_token is not None and cash_token in rewards_map:
                residual = max(limit - total_weight, 0.0)
                if residual > 0:
                    weights[cash_token] = weights.get(cash_token, 0.0) + residual
            weighted_reward = sum(weights.get(inst, 0.0) * rewards_map.get(inst, 0.0) for inst in weights)
            portfolio_score = sum(weights.get(inst, 0.0) * logits_map.get(inst, 0.0) for inst in weights)
            turnover = _compute_turnover(prev_weights, weights)
            trade_cost = per_unit_cost * turnover
            net_reward = weighted_reward - trade_cost
            nav *= 1.0 + net_reward
            prev_weights = weights
            instrument_str = ";".join(f"{inst}:{weight:.3f}" for inst, weight in sorted(weights.items()))
            records.append(
                {
                    "date": batch.date,
                    "instrument": instrument_str,
                    "raw_reward": weighted_reward,
                    "cost": trade_cost,
                    "reward": net_reward,
                    "prob": max(weights.values()) if weights else 0.0,
                    "logit": portfolio_score,
                    "nav": nav,
                    "turnover": turnover,
                    "held_previous": held_prev,
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
    top_k: int = 1,
    hold_threshold: Optional[float] = None,
    min_weight: float = 0.0,
    cash_token: Optional[str] = None,
    max_gross_exposure: float = 1.0,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    trades = run_policy_on_dataset(
        model=model,
        dataset=dataset,
        device=device,
        temperature=temperature,
        greedy=greedy,
        commission=commission,
        slippage=slippage,
        top_k=top_k,
        hold_threshold=hold_threshold,
        min_weight=min_weight,
        cash_token=cash_token,
        max_gross_exposure=max_gross_exposure,
    )
    summary = compute_performance(trades, risk_free=risk_free)
    return trades, summary


def save_trades(trades: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(path, index=False)
