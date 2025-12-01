"""Utility functions for transformer GRPO pipeline."""
from __future__ import annotations

import math
import random
import threading
from queue import Queue
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Sampler

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


class BackgroundGenerator:
    """
    Prefetch samples on a background thread to overlap CPU batch preparation with GPU work.
    Inspired by the strategy used in YOLO-style dataloaders.
    """

    def __init__(self, iterator: Iterator, max_prefetch: int = 1):
        self._iterator = iterator
        self._queue: "Queue[Any]" = Queue(max(1, max_prefetch))
        self._sentinel = object()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        try:
            for item in self._iterator:
                self._queue.put(item)
        finally:
            self._queue.put(self._sentinel)

    def __iter__(self) -> "BackgroundGenerator":
        return self

    def __next__(self) -> Any:
        item = self._queue.get()
        if item is self._sentinel:
            raise StopIteration
        return item


def _move_batch_to_device(data: Any, device: torch.device) -> Any:
    if torch.is_tensor(data):
        return data.to(device, non_blocking=True)
    if isinstance(data, dict):
        return {key: _move_batch_to_device(value, device) for key, value in data.items()}
    if isinstance(data, list):
        return [_move_batch_to_device(value, device) for value in data]
    if isinstance(data, tuple):
        return tuple(_move_batch_to_device(value, device) for value in data)
    return data


class PrefetchDataLoader:
    """
    Wrapper around DataLoader that (1) prefetches batches on a background thread/process and
    (2) optionally moves tensors to the target device asynchronously via a dedicated CUDA stream.
    """

    def __init__(
        self,
        loader: torch.utils.data.DataLoader,
        buffer_size: int = 2,
        device: Optional[torch.device] = None,
        async_transfer: bool = False,
    ):
        self.loader = loader
        self.buffer_size = max(int(buffer_size), 1)
        self.device = device
        self.async_transfer = async_transfer and device is not None and device.type == "cuda"

    def __iter__(self):
        iterator = iter(self.loader)
        if self.buffer_size > 1:
            iterator = BackgroundGenerator(iterator, max_prefetch=self.buffer_size)

        if not self.async_transfer:
            yield from iterator
            return

        stream = torch.cuda.Stream(device=self.device)

        def _next_prefetched():
            try:
                batch = next(iterator)
            except StopIteration:
                return None
            with torch.cuda.stream(stream):
                return _move_batch_to_device(batch, self.device)  # type: ignore[arg-type]

        next_batch = _next_prefetched()
        while next_batch is not None:
            torch.cuda.current_stream().wait_stream(stream)
            current = next_batch
            next_batch = _next_prefetched()
            yield current


class LengthAwareBatchSampler(Sampler[List[int]]):
    """Groups samples with similar instrument counts to reduce padding overhead."""

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        shuffle: bool = True,
        bucket_size_multiplier: float = 8.0,
        drop_last: bool = False,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.lengths: List[int] = list(lengths)
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        bucket_size_multiplier = max(float(bucket_size_multiplier), 1.0)
        self.bucket_size = max(self.batch_size, int(self.batch_size * bucket_size_multiplier))
        self._indices = list(range(len(self.lengths)))

    def __len__(self) -> int:
        if self.drop_last:
            return len(self._indices) // self.batch_size
        return math.ceil(len(self._indices) / max(self.batch_size, 1))

    def __iter__(self):  # type: ignore[override]
        indices = list(self._indices)
        if self.shuffle:
            random.shuffle(indices)

        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start : start + self.bucket_size]
            bucket.sort(key=lambda idx: self.lengths[idx], reverse=True)
            for chunk_start in range(0, len(bucket), self.batch_size):
                batch = bucket[chunk_start : chunk_start + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                if batch:
                    yield batch


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
    Sample `group_size` actions for each batch item (independent sets) via Gumbel-TopK.
    Fully vectorized to keep sampling on GPU.
    """
    batch_size, num_inst = logits.shape
    temperature = max(float(temperature), 1e-4)
    masked_logits = logits.masked_fill(~mask, -1e9)
    log_probs = F.log_softmax(masked_logits / temperature, dim=-1)  # [B, N]

    if top_k <= 0 or group_size <= 0:
        empty_mask = torch.zeros(batch_size, group_size, num_inst, dtype=torch.bool, device=logits.device)
        empty_logprob = torch.zeros(batch_size, group_size, device=logits.device)
        return empty_mask, empty_logprob

    gumbel_shape = (batch_size, group_size, num_inst)
    # Avoid log(0) by clamping to tiny positive number
    uniform = torch.rand(gumbel_shape, device=logits.device).clamp_(min=1e-9, max=1 - 1e-9)
    gumbels = -torch.log(-torch.log(uniform))

    scores = log_probs.unsqueeze(1) + gumbels  # [B, G, N]
    scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))

    k = min(int(top_k), num_inst)
    _, topk_idx = torch.topk(scores, k=k, dim=-1)

    repeated_mask = mask.unsqueeze(1).expand_as(scores)
    valid_flags = torch.gather(repeated_mask, 2, topk_idx)

    sampled_masks = torch.zeros_like(scores, dtype=torch.bool)
    sampled_masks.scatter_(2, topk_idx, valid_flags)

    expanded_log_probs = log_probs.unsqueeze(1).expand_as(scores)
    selected_log_probs = torch.gather(expanded_log_probs, 2, topk_idx)
    valid_float = valid_flags.float()
    counts = valid_float.sum(dim=-1).clamp_min(1.0)
    sample_log_probs = (selected_log_probs * valid_float).sum(dim=-1) / counts

    return sampled_masks, sample_log_probs



def collate_daily_batches(samples: List[DailyBatch]) -> Dict[str, object]:
    """将多个DailyBatch合并为一个batch，并保留输入特征的原始精度。"""
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

    np_feat_dtype = np.dtype(getattr(samples[0], "feature_dtype", np.float32))
    torch_feat_dtype = _numpy_dtype_to_torch(np_feat_dtype)

    features = torch.zeros(batch_size, max_tokens, temporal_span, feature_dim, dtype=torch_feat_dtype)
    rewards = torch.zeros(batch_size, max_tokens, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_tokens, dtype=torch.bool)
    metadata: List[Dict[str, object]] = []

    raw_rewards = torch.zeros(batch_size, max_tokens, dtype=torch.float32)

    for idx, sample in enumerate(samples):
        feat, rew, raw = sample.materialize()
        length = feat.shape[0]
        if feat.ndim == 2:
            feat = feat[:, np.newaxis, :]
        feat_np = np.asarray(feat, dtype=np_feat_dtype)
        feat_tensor = torch.from_numpy(feat_np)
        if feat_tensor.dtype != torch_feat_dtype:
            feat_tensor = feat_tensor.to(torch_feat_dtype)
        features[idx, :length] = feat_tensor
        rewards[idx, :length] = torch.from_numpy(rew.astype(np.float32, copy=False))
        raw_rewards[idx, :length] = torch.from_numpy(raw.astype(np.float32, copy=False))
        mask[idx, :length] = True
        metadata.append(
            {
                "date": sample.date.to_pydatetime() if hasattr(sample.date, "to_pydatetime") else sample.date,
                "instruments": [str(inst) for inst in sample.instruments.tolist()],
            }
        )

    return {
        "features": features,
        "rewards": rewards,
        "raw_rewards": raw_rewards,
        "mask": mask,
        "meta": metadata,
    }


def _infer_sample_shape(sample: DailyBatch) -> Tuple[int, ...]:
    if sample.feature_shape:
        return sample.feature_shape
    if sample.features is not None:
        return sample.features.shape
    feat, _, _ = sample.materialize()
    return feat.shape


def _numpy_dtype_to_torch(dtype: np.dtype) -> torch.dtype:
    if dtype == np.float16:
        return torch.float16
    if dtype == np.float64:
        return torch.float64
    return torch.float32


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
    final_nav = float(nav[-1])
    total_return = final_nav - 1.0
    trading_days = max(len(daily_returns), 1)
    if final_nav <= 0:
        ann_return = -1.0
    else:
        ann_return = float(final_nav ** (252.0 / trading_days) - 1.0)
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
