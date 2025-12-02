"""Trainer for cross-sectional stock selection with direct return optimization."""
from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml
import qlib
from tqdm import tqdm

from .backtest import run_backtest, save_trades
from .data_pipeline import DailyBatchDataset, DailyBatchFactory
from .logger import LoggerFactory, BaseLogger
from .model import TransformerPolicy
from .utils import (
    PrefetchDataLoader,
    collate_daily_batches,
    compute_topk_weights,
    ensure_dir,
    set_global_seed,
    sample_topk_actions,
    LengthAwareBatchSampler,
)


def compute_rank_correlation(pred_scores: torch.Tensor, actual_rewards: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """计算预测分数和实际收益的 Spearman 排名相关系数（向量化实现）"""
    batch_size = pred_scores.size(0)
    large_val = 1e9

    pred_masked = pred_scores.masked_fill(~mask, -large_val)
    actual_masked = actual_rewards.masked_fill(~mask, -large_val)

    pred_rank = pred_masked.argsort(dim=1).argsort(dim=1).float()
    actual_rank = actual_masked.argsort(dim=1).argsort(dim=1).float()

    n = mask.sum(dim=1).float()
    d = (pred_rank - actual_rank) * mask.float()
    d_squared_sum = (d ** 2).sum(dim=1)

    denom = n * (n * n - 1)
    denom = torch.clamp(denom, min=1.0)
    correlations = 1.0 - 6.0 * d_squared_sum / denom
    correlations = correlations.masked_fill(n < 3, 0.0)

    return correlations


def compute_listmle_loss_vectorized(logits: torch.Tensor, rewards: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    向量化的ListMLE loss实现。

    ListMLE: 给定真实排序，最大化按此排序采样的概率。
    """
    batch_size, num_inst = logits.shape
    device = logits.device

    # 按reward排序，获取排序索引
    # 将无效位置的reward设为-inf使其排在最后
    rewards_masked = rewards.masked_fill(~mask, -1e9)
    sorted_indices = rewards_masked.argsort(dim=1, descending=True)  # [batch, num_inst]

    # 按真实排序重排logits
    sorted_logits = torch.gather(logits, 1, sorted_indices)  # [batch, num_inst]

    # 将无效位置的logits设为-inf
    sorted_mask = torch.gather(mask, 1, sorted_indices)
    sorted_logits = sorted_logits.masked_fill(~sorted_mask, -1e9)

    # 计算从后向前的cumulative logsumexp
    # ListMLE: sum_i [log(exp(s_i) / sum_{j>=i} exp(s_j))]
    #        = sum_i [s_i - logsumexp(s_{i:n})]
    flipped = sorted_logits.flip(dims=[1])
    cum_logsumexp = torch.logcumsumexp(flipped, dim=1).flip(dims=[1])

    # loss = logsumexp - logits (对每个位置)
    loss_per_pos = cum_logsumexp - sorted_logits

    # 只对有效位置求和
    loss_per_pos = loss_per_pos.masked_fill(~sorted_mask, 0.0)

    # 每个batch的loss是有效位置的平均
    n_valid = mask.sum(dim=1).clamp(min=1).float()
    batch_loss = loss_per_pos.sum(dim=1) / n_valid
    avg_tokens = n_valid.mean().clamp(min=1.0)
    return batch_loss.mean() / avg_tokens

import math

class TopKScheduler:
    def __init__(self, start_k: int, end_k: int, total_epochs: int, strategy: str = "linear"):
        """
        Top-K 调度器
        
        Args:
            start_k: 初始 Top-K (例如 30)
            end_k: 最终 Top-K (例如 5)
            total_epochs: 总 Epoch 数
            strategy: 'linear' (线性) 或 'cosine' (余弦退火)
        """
        self.start_k = start_k
        self.end_k = max(1, end_k) # 保证至少为1
        self.total_epochs = total_epochs
        self.strategy = strategy

    def get_k(self, current_epoch: int) -> int:
        """根据当前 epoch (1-based) 返回当前的 k 值"""
        # 防止越界
        if current_epoch >= self.total_epochs:
            return self.end_k
        if current_epoch <= 0:
            return self.start_k

        # 进度 (0.0 -> 1.0)
        progress = (current_epoch - 1) / (self.total_epochs - 1)

        if self.strategy == "linear":
            # 线性衰减: k = start - (start - end) * progress
            k = self.start_k - (self.start_k - self.end_k) * progress
        
        elif self.strategy == "cosine":
            # 余弦衰减: 也就是先慢，中间快，最后慢
            cosine_progress = 0.5 * (1 + math.cos(progress * math.pi))
            # 注意：cosine 是从 1 到 -1 (或类似)，这里简化为权重混合
            # 修正公式：start + (end - start) * (1 - cosine_decay)
            # 简单写法：使用 standard cosine annealing logic
            k = self.end_k + 0.5 * (self.start_k - self.end_k) * (1 + math.cos(progress * math.pi))
            
        elif self.strategy == "step":
            # 阶梯衰减：每过 1/3 阶段减半，或者自定义
            # 这里简单实现：前50%用 start_k，后50%线性降到 end_k
            if progress < 0.5:
                k = self.start_k
            else:
                ratio = (progress - 0.5) * 2
                k = self.start_k - (self.start_k - self.end_k) * ratio
        else:
            k = self.start_k

        return max(1, int(round(k)))
class Trainer:
    """GRPO (Group Relative Policy Optimization) Trainer."""

    def __init__(self, config: Dict):
        self.config = config
        self.train_cfg = config.get("training", {})
        self.model_cfg = config.get("model", {})
        self.data_cfg = config.get("data", {})
        self.backtest_cfg = config.get("backtest", {})
        reward_cfg = self.data_cfg.get("reward", {}) or {}
        self.reward_scale = max(float(reward_cfg.get("scale", 1.0)), 1e-6)

        # 初始化路径和日志
        self.output_root = Path(self.train_cfg.get("checkpoint_root", "runs/transformer_grpo"))
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.work_dir = ensure_dir(self.output_root / self.timestamp)
        self.history_file = self.work_dir / "metrics.jsonl"

        self.logger_cfg = self.train_cfg.get("logger") or {}
        self.logger_split_runs = bool(self.logger_cfg.get("split_pretrain_run", False))
        pretrain_suffix = "pretrain" if self.logger_split_runs and self.pretrain_epochs > 0 else None
        self.logger = self._build_logger(pretrain_suffix)

        set_global_seed(int(self.train_cfg.get("seed", 42)))

        qlib.init(**config.get("qlib", {}))
        self.device = self._resolve_device(self.train_cfg.get("device", "auto"))

        self._build_datasets()
        self._loader_prefetches_device = False
        self._loader_pin_memory = False

        # 训练参数
        self.epochs = int(self.train_cfg.get("epochs", 40))
        self.batch_size = int(self.train_cfg.get("batch_size", 4))
        self.entropy_coef = float(self.train_cfg.get("entropy_coef", 0.01))
        self.rank_coef_start = float(self.train_cfg.get("rank_coef", 0.0))
        self.rank_coef_final = float(self.train_cfg.get("rank_coef_final", self.rank_coef_start))
        self.rank_coef_decay_epochs = max(1, int(self.train_cfg.get("rank_coef_decay_epochs", self.epochs)))
        self.grad_clip = float(self.train_cfg.get("grad_clip", 1.0))
        default_loss_weights = {"listmle": 1.0, "mse_norm": 0.005, "mse_raw": 0.002}
        cfg_loss_weights = self.train_cfg.get("pretrain_loss_weights") or {}
        self.pretrain_loss_weights = {
            "listmle": float(cfg_loss_weights.get("listmle", default_loss_weights["listmle"])),
            "mse_norm": float(cfg_loss_weights.get("mse_norm", default_loss_weights["mse_norm"])),
            "mse_raw": float(cfg_loss_weights.get("mse_raw", default_loss_weights["mse_raw"])),
        }
        self.pretrain_entropy_coef = float(self.train_cfg.get("pretrain_entropy_coef", 0.0))
        self.pretrain_listmle_warmup_steps = max(0, int(self.train_cfg.get("pretrain_listmle_warmup_steps", 0)))
        self.early_stop_patience = int(self.train_cfg.get("early_stop_patience", 20))
        self.log_interval = int(self.train_cfg.get("log_interval", 50))
        self.pretrain_epochs = int(self.train_cfg.get("pretrain_epochs", 5))
        self.continue_rl = bool(self.train_cfg.get("continue_rl_after_pretrain", True))
        self.skip_rl_training = False
        self.grad_accum_steps = max(1, int(self.train_cfg.get("grad_accum_steps", 1)))
        self._grad_accum_counter = 0
        default_pretrain_scale = 100.0 if self.reward_scale == 1.0 else 1.0
        self.pretrain_value_scale = float(self.train_cfg.get("pretrain_value_scale", default_pretrain_scale))

        # GRPO Params
        self.group_size = int(self.train_cfg.get("group_size", 4))
        self.ppo_epochs = max(1, int(self.train_cfg.get("ppo_epochs", 1)))
        self.clip_ratio = float(self.train_cfg.get("clip_ratio", 0.2))
        self.kl_coef = float(self.train_cfg.get("kl_coef", 0.0))
        self.target_kl = float(self.train_cfg.get("target_kl", 0.02))
        self.kl_adjust_rate = float(self.train_cfg.get("kl_adjust_rate", 1.5))
        self.min_kl_coef = float(self.train_cfg.get("min_kl_coef", 1e-5))
        self.max_kl_coef = float(self.train_cfg.get("max_kl_coef", 0.05))
        self.turnover_cost = float(self.train_cfg.get("turnover_cost", 0.0))

        # Top-k参数
        # --- Top-K Scheduler Params ---
        self.start_top_k = int(self.train_cfg.get("train_top_k", 10))
        # 如果没配置 min_train_top_k，默认不衰减 (end_k = start_k)
        self.end_top_k = int(self.train_cfg.get("min_train_top_k", self.start_top_k))
        self.k_decay_strategy = self.train_cfg.get("k_decay_strategy", "linear")

        # 初始化调度器
        self.k_scheduler = TopKScheduler(
            start_k=self.start_top_k,
            end_k=self.end_top_k,
            total_epochs=self.epochs,
            strategy=self.k_decay_strategy
        )
        
        # 初始 current_k
        self.current_train_k = self.start_top_k
        self.temperature = float(self.train_cfg.get("temperature", 1.0))

        self.global_step = 0
        self.no_improve_epochs = 0

        self._build_model()
        self._save_config()

        # 预训练
        if self.pretrain_epochs > 0:
            self._pretrain_policy()
            self._save_checkpoint("pretrain.pt", 0)
            if self.logger_split_runs and self.continue_rl:
                self.logger.close()
                self.logger = self._build_logger("rl")

        if self.pretrain_epochs > 0 and not self.continue_rl:
            print("[Trainer] RL training skipped per configuration `continue_rl_after_pretrain=False`.")
            self.skip_rl_training = True
            return

    def _resolve_device(self, requested: str) -> torch.device:
        if requested == "cpu":
            return torch.device("cpu")
        if torch.cuda.is_available() and requested in ("auto", "cuda", "gpu"):
            return torch.device("cuda")
        return torch.device("cpu")

    def _build_logger(self, run_suffix: Optional[str] = None) -> BaseLogger:
        if not self.logger_cfg:
            return LoggerFactory({}, self.config, self.work_dir).build()
        cfg = dict(self.logger_cfg)
        if run_suffix and (cfg.get("type") or "").lower() == "wandb":
            base_name = cfg.get("run_name")
            if base_name:
                cfg["run_name"] = f"{base_name}-{run_suffix}"
            else:
                cfg["run_name"] = run_suffix
        return LoggerFactory(cfg, self.config, self.work_dir).build()

    def _rank_coef_for_epoch(self, epoch: int) -> float:
        if self.rank_coef_start == self.rank_coef_final:
            return self.rank_coef_start
        span = max(self.rank_coef_decay_epochs - 1, 1)
        progress = min(max((epoch - 1) / span, 0.0), 1.0)
        return self.rank_coef_start + (self.rank_coef_final - self.rank_coef_start) * progress

    def _adjust_kl_coef(self, avg_kl: float) -> None:
        if avg_kl <= 0:
            return
        high_thr = self.target_kl * 1.5
        low_thr = self.target_kl / 1.5 if self.target_kl > 0 else 0.0
        if avg_kl > high_thr:
            self.kl_coef = min(self.kl_coef * self.kl_adjust_rate, self.max_kl_coef)
        elif low_thr > 0 and avg_kl < low_thr:
            self.kl_coef = max(self.kl_coef / self.kl_adjust_rate, self.min_kl_coef)

    def _build_datasets(self):
        """构建数据集，顺序处理以减少内存峰值。"""
        handler_cfg = self.data_cfg.get("handler")
        if handler_cfg is None:
            raise ValueError("`data.handler` must be provided in config.")

        segments = self.data_cfg.get("segments", {})
        reward_cfg = self.data_cfg.get("reward", {}) or {}
        reward_clip = reward_cfg.get("clip")
        if reward_clip is not None:
            reward_clip = (float(reward_clip[0]), float(reward_clip[1]))

        self.data_factory = DailyBatchFactory(
            handler_config=handler_cfg,
            segments=segments,
            feature_group=self.data_cfg.get("feature_group", "feature"),
            label_group=self.data_cfg.get("label_group", "label"),
            min_instruments=int(self.data_cfg.get("min_instruments", 30)),
            max_instruments=self.data_cfg.get("max_instruments"),
            reward_clip=reward_clip,
            reward_scale=float(reward_cfg.get("scale", 1.0)),
            reward_normalize=reward_cfg.get("normalize"),
            augment=self.data_cfg.get("augment"),
            feature_dtype=self.data_cfg.get("feature_dtype", "float32"),
        )

        # 顺序构建数据集，每个构建完后清理内存
        print("[Data] Building train dataset...")
        self.train_dataset = self.data_factory.build_segment("train")
        gc.collect()

        print("[Data] Building valid dataset...")
        self.valid_dataset = self.data_factory.build_segment("valid") if "valid" in segments else DailyBatchDataset([])
        gc.collect()

        print("[Data] Building test dataset...")
        self.test_dataset = self.data_factory.build_segment("test") if "test" in segments else DailyBatchDataset([])
        gc.collect()

        # 释放 handler 内部缓存
        if hasattr(self.data_factory, 'handler') and hasattr(self.data_factory.handler, '_data'):
            self.data_factory.handler._data = None
        gc.collect()

        if len(self.train_dataset) == 0:
            raise ValueError("Training segment produced zero samples.")

    def _build_model(self):
        feature_dim = self.train_dataset.feature_dim
        temporal_span = getattr(self.train_dataset, "temporal_span", 1)
        max_instruments = getattr(self.train_dataset, "max_instruments", 0) + 10

        self.model_cfg.setdefault("temporal_span", temporal_span)
        self.model_cfg.setdefault("max_positions", max_instruments)

        self.model = TransformerPolicy(feature_dim=feature_dim, **self.model_cfg).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.train_cfg.get("learning_rate", 1e-3)),
            weight_decay=float(self.train_cfg.get("weight_decay", 0.01)),
        )
        self.scheduler = None
        if self.train_cfg.get("use_cosine_lr", True):
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        self.optimizer.zero_grad(set_to_none=True)

    def _save_config(self):
        with (self.work_dir / "config.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.config, f, allow_unicode=True)

    def _build_loader(self, dataset: DailyBatchDataset, shuffle: bool) -> DataLoader:
        loader_cfg = self.train_cfg.get("dataloader", {})
        num_workers = int(loader_cfg.get("num_workers", 0))
        pin_memory_default = self.device.type == "cuda"
        pin_memory = bool(loader_cfg.get("pin_memory", pin_memory_default))
        drop_last = bool(loader_cfg.get("drop_last", False))
        loader_kwargs = {
            "dataset": dataset,
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "collate_fn": collate_daily_batches,
            "num_workers": num_workers,
            "drop_last": drop_last,
            "pin_memory": pin_memory,
        }

        bucket_sampler = None
        bucket_enabled = bool(loader_cfg.get("bucket_by_instrument_count", False))
        has_lengths = getattr(dataset, "instrument_counts", None)
        if bucket_enabled and shuffle and has_lengths:
            bucket_multiplier = float(loader_cfg.get("bucket_size_multiplier", 8.0))
            bucket_sampler = LengthAwareBatchSampler(
                lengths=dataset.instrument_counts,
                batch_size=self.batch_size,
                shuffle=True,
                bucket_size_multiplier=bucket_multiplier,
                drop_last=drop_last,
            )

        if bucket_sampler is not None:
            loader_kwargs.pop("batch_size", None)
            loader_kwargs.pop("shuffle", None)
            loader_kwargs.pop("drop_last", None)
            loader_kwargs["batch_sampler"] = bucket_sampler

        if num_workers > 0:
            prefetch_factor = loader_cfg.get("prefetch_factor")
            if prefetch_factor is not None:
                loader_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
            loader_kwargs["persistent_workers"] = bool(loader_cfg.get("persistent_workers", False))
        loader = DataLoader(**loader_kwargs)

        async_transfer = bool(loader_cfg.get("async_device_transfer", False)) and self.device.type == "cuda"
        buffer_size = int(loader_cfg.get("prefetch_batches", 0))
        if async_transfer:
            buffer_size = max(buffer_size, 2)
        should_wrap = async_transfer or buffer_size > 0
        if should_wrap:
            loader = PrefetchDataLoader(
                loader=loader,
                buffer_size=max(buffer_size, 1),
                device=self.device if async_transfer else None,
                async_transfer=async_transfer,
            )

        self._loader_pin_memory = pin_memory
        self._loader_prefetches_device = async_transfer
        return loader

    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._loader_prefetches_device:
            features = batch["features"]
            rewards = batch["rewards"]
            raw_rewards = batch["raw_rewards"]
            mask = batch["mask"]
        else:
            non_blocking = self.device.type == "cuda" and self._loader_pin_memory
            features = batch["features"].to(self.device, non_blocking=non_blocking)
            rewards = batch["rewards"].to(self.device, non_blocking=non_blocking)
            raw_rewards = batch["raw_rewards"].to(self.device, non_blocking=non_blocking)
            mask = batch["mask"].to(self.device, non_blocking=non_blocking)

        if features.dtype != torch.float32:
            features = features.float()

        return features, rewards, raw_rewards, mask

    def _backward_with_accum(self, loss: torch.Tensor, count_global_step: bool) -> None:
        scaled_loss = loss / self.grad_accum_steps
        scaled_loss.backward()
        self._grad_accum_counter += 1
        if self._grad_accum_counter >= self.grad_accum_steps:
            self._step_optimizer(count_global_step)

    def _step_optimizer(self, count_global_step: bool) -> None:
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self._grad_accum_counter = 0
        if count_global_step:
            self.global_step += 1

    def _flush_optimizer(self, count_global_step: bool) -> None:
        if self._grad_accum_counter > 0:
            self._step_optimizer(count_global_step)

    def _pretrain_policy(self):
        """Pretraining phase: ListMLE + MSE for stability."""
        loader = self._build_loader(self.train_dataset, shuffle=True)
        print(f"[Pretrain] Starting supervised warm-up for {self.pretrain_epochs} epoch(s)")
        
        mse_loss_fn = torch.nn.MSELoss()
        loss_w = self.pretrain_loss_weights
        pretrain_global_step = 0

        for epoch in range(1, self.pretrain_epochs + 1):
            self.model.train()
            total_loss = 0.0
            total_rank_corr = 0.0
            total_topk_return = 0.0
            total_entropy = 0.0
            steps = 0
            warn_flag = False

            for step, batch in enumerate(tqdm(loader, desc=f"[Pretrain] Epoch {epoch}"), start=1):
                features, rewards_norm, rewards_raw, mask = self._move_batch_to_device(batch)

                logits, _ = self.model(features, mask)

                # 1. ListMLE Loss (Ranking)
                listmle_loss = compute_listmle_loss_vectorized(logits, rewards_norm, mask)
                
                # 2. MSE Loss (Value stability on normed labels)
                valid_logits = logits[mask]
                masked_norm_rewards = rewards_norm[mask]
                target_rewards = masked_norm_rewards * self.pretrain_value_scale
                mse_norm = mse_loss_fn(valid_logits, target_rewards)

                # 3. Raw reward regression (anchors RL objective)
                masked_raw_rewards = rewards_raw[mask]
                mse_raw = mse_loss_fn(valid_logits, masked_raw_rewards)

                masked_logits_full = logits.masked_fill(~mask, -1e9)
                probs = F.softmax(masked_logits_full, dim=-1)
                entropy = -(probs * F.log_softmax(masked_logits_full, dim=-1)).sum(dim=-1).mean()

                warmup_scale = 1.0
                if self.pretrain_listmle_warmup_steps > 0:
                    warmup_scale = min(pretrain_global_step / self.pretrain_listmle_warmup_steps, 1.0)

                loss = (
                    loss_w["listmle"] * listmle_loss
                    + (loss_w["mse_norm"] * warmup_scale) * mse_norm
                    + (loss_w["mse_raw"] * warmup_scale) * mse_raw
                    - self.pretrain_entropy_coef * entropy
                )
                
                rank_corr = compute_rank_correlation(logits, rewards_norm, mask).mean()

                self._backward_with_accum(loss, count_global_step=False)

                # Diagnostics: entropy and top-k raw return
                entropy_value = entropy.detach().item()
                weights = compute_topk_weights(
                    logits=logits.detach(), mask=mask, top_k=self.start_top_k,
                    temperature=self.temperature, min_weight=0.0, differentiable=True,
                )
                avg_topk = (weights * rewards_raw).sum(dim=-1).mean()

                total_loss += loss.item()
                total_rank_corr += rank_corr.item()
                total_topk_return += avg_topk.item()
                total_entropy += entropy_value
                steps += 1
                if rank_corr.item() < -0.05:
                    warn_flag = True

                step_metrics = {
                    "epoch": epoch,
                    "step": step,
                    "loss": loss.item(),
                    "listmle_loss": listmle_loss.item(),
                    "mse_norm": mse_norm.item(),
                    "mse_raw": mse_raw.item(),
                    "rank_corr": rank_corr.item(),
                    "entropy": entropy_value,
                    "avg_topk_raw_return": avg_topk.item(),
                    "warmup_scale": warmup_scale,
                }
                self.logger.log_metrics("pretrain_step", step_metrics, step=pretrain_global_step)
                pretrain_global_step += 1

            self._flush_optimizer(count_global_step=False)
            avg_loss = total_loss / max(steps, 1)
            avg_rank_corr = total_rank_corr / max(steps, 1)
            avg_topk_return = total_topk_return / max(steps, 1)
            avg_entropy = total_entropy / max(steps, 1)
            metrics = {
                "epoch": epoch,
                "loss": avg_loss,
                "rank_corr": avg_rank_corr,
                "avg_topk_raw_return": avg_topk_return,
                "entropy": avg_entropy,
            }
            self.logger.log_metrics("pretrain_epoch", metrics)
            print(f"[Pretrain {epoch}] loss={avg_loss:.4f} rank_corr={avg_rank_corr:.4f} "
                  f"topk_ret={avg_topk_return:.5f} entropy={avg_entropy:.3f}")
            if warn_flag:
                print(f"[Pretrain {epoch}] Warning: rank_corr dropped below -0.05; consider checking labels/features.")
            if avg_topk_return < 0:
                print(f"[Pretrain {epoch}] Warning: average top-k raw return is negative ({avg_topk_return:.5f}).")

        print("[Pretrain] Completed")
        # Optional quick sanity backtest on validation segment
        if len(self.valid_dataset) > 0:
            print("[Pretrain] Running sanity backtest on validation split...")
            trades, summary = run_backtest(
                model=self.model,
                dataset=self.valid_dataset,
                device=self.device,
                top_k=int(self.backtest_cfg.get("top_k", self.start_top_k)),
                temperature=float(self.backtest_cfg.get("temperature", 1.0)),
                min_weight=float(self.backtest_cfg.get("min_weight", 0.0)),
                commission=float(self.backtest_cfg.get("commission", 0.0)),
                slippage=float(self.backtest_cfg.get("slippage", 0.0)),
                reward_scale=self.reward_scale,
            )
            pretrain_dir = ensure_dir(self.work_dir / "pretrain")
            save_trades(trades, pretrain_dir / "pretrain_trades.csv")
            (pretrain_dir / "pretrain_metrics.json").write_text(json.dumps(summary, indent=2))
            self.logger.log_metrics("pretrain_backtest", summary)
            print(f"[Pretrain] Sanity backtest sharpe={summary.get('sharpe', 0):.4f}, "
                  f"cum_return={summary.get('cumulative_return', 0):.4f}")

    def train(self):
        """主训练循环"""
        if getattr(self, "skip_rl_training", False):
            self.logger.close()
            return
        train_loader = self._build_loader(self.train_dataset, shuffle=True)
        best_metric = float("-inf")
        best_epoch = 0

        try:
            for epoch in range(1, self.epochs + 1):
                # --- [新增] 更新 Top-K ---
                self.current_train_k = self.k_scheduler.get_k(epoch)
                train_stats = self._run_epoch(train_loader, epoch)
                self._adjust_kl_coef(train_stats.get("kl", 0.0))
                self.logger.log_metrics("train_epoch", {"epoch": epoch, **train_stats})

                val_stats = {}
                if len(self.valid_dataset) > 0:
                    _, val_stats = self.evaluate(self.valid_dataset, "valid", epoch)
                    metric_val = val_stats.get("sharpe", float("-inf"))
                    if metric_val > best_metric:
                        best_metric = metric_val
                        best_epoch = epoch
                        self._save_checkpoint("best.pt", epoch)
                        self.no_improve_epochs = 0
                    else:
                        self.no_improve_epochs += 1
                    self.logger.log_metrics("valid_epoch", {"epoch": epoch, **val_stats})

                record = {"epoch": epoch, "train": train_stats, "valid": val_stats}
                with self.history_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, default=str) + "\n")

                print(f"[Epoch {epoch}] loss={train_stats['loss']:.4f} rank_corr={train_stats['rank_corr']:.4f} "
                      f"return={train_stats['avg_return']:.5f}")
                if val_stats:
                    print(f"[Epoch {epoch}] valid_sharpe={val_stats.get('sharpe', 0):.4f} "
                          f"valid_return={val_stats.get('cumulative_return', 0):.4f}")

                if self.scheduler:
                    self.scheduler.step()

                if self.early_stop_patience > 0 and self.no_improve_epochs >= self.early_stop_patience:
                    print(f"[EarlyStop] Stopping at epoch {epoch}")
                    break

            self._save_checkpoint("last.pt", epoch)

            if len(self.test_dataset) > 0:
                self.evaluate(self.test_dataset, "test", epoch + 1)

            if best_metric > float("-inf"):
                print(f"Best sharpe={best_metric:.4f} at epoch {best_epoch}")

        finally:
            self.logger.close()

    def _run_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        GRPO Training Epoch.
        Samples multiple actions (groups) per input and uses PPO-style clipped loss.
        """
        self.model.train()
        total_loss = 0.0
        total_pg_loss = 0.0
        total_rank_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        total_return = 0.0
        total_rank_corr = 0.0
        total_turnover_penalty = 0.0
        total_updates = 0
        batch_count = 0

        rank_coef_value = self._rank_coef_for_epoch(epoch)
        pbar = tqdm(loader, desc=f"[Train] Epoch {epoch}")
        for step, batch in enumerate(pbar, start=1):
            features, rewards_norm, rewards_raw, mask = self._move_batch_to_device(batch)
            batch_count += 1

            with torch.no_grad():
                logits, _ = self.model(features, mask)
                masked_logits = logits.masked_fill(~mask, -1e9)
                log_probs = F.log_softmax(masked_logits / self.temperature, dim=-1)
                sampled_masks, old_log_probs = sample_topk_actions(
                    logits,
                    mask,
                    top_k=self.current_train_k,
                    group_size=self.group_size,
                    temperature=self.temperature,
                )
                sampled_masks = sampled_masks.detach()
                sampled_mask_float = sampled_masks.float()
                old_log_probs = old_log_probs.detach()

                raw_rewards_exp = rewards_raw.unsqueeze(1).expand(-1, self.group_size, -1)
                selected_raw = (raw_rewards_exp * sampled_mask_float).sum(dim=-1)
                selected_counts = sampled_mask_float.sum(dim=-1).clamp(min=1.0)
                sqrt_counts = torch.sqrt(selected_counts).clamp(min=1.0)
                if self.turnover_cost > 0:
                    turnover_penalty = self.turnover_cost * (selected_counts / max(float(self.current_train_k), 1.0))
                else:
                    turnover_penalty = torch.zeros_like(selected_counts)
                sample_returns = (selected_raw - turnover_penalty) / sqrt_counts
                avg_batch_return = sample_returns.mean()

                mean_return = sample_returns.mean(dim=1, keepdim=True)
                std_return = sample_returns.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-3)
                advantages = (sample_returns - mean_return) / std_return

                rank_corr_value = compute_rank_correlation(logits, rewards_norm, mask).mean().item()

            advantages = advantages.detach()
            sampled_mask_float = sampled_mask_float.detach()
            selected_counts = selected_counts.detach()
            turnover_penalty_value = turnover_penalty.mean().item()
            adv_mean_value = advantages.mean().item()
            adv_std_value = advantages.std(unbiased=False).item()
            avg_return_value = (avg_batch_return / self.reward_scale).item()

            total_return += avg_return_value
            total_rank_corr += rank_corr_value
            total_turnover_penalty += turnover_penalty_value

            for ppo_epoch in range(self.ppo_epochs):
                logits, _ = self.model(features, mask)
                masked_logits = logits.masked_fill(~mask, -1e9)
                all_log_probs = F.log_softmax(masked_logits / self.temperature, dim=-1)
                all_log_probs_exp = all_log_probs.unsqueeze(1).expand(-1, self.group_size, -1)
                numerator = (all_log_probs_exp * sampled_mask_float).sum(dim=-1)
                new_log_probs = numerator / selected_counts

                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
                pg_loss = -torch.min(surr1, surr2).mean()

                kl_div = (old_log_probs - new_log_probs).mean()
                probs = all_log_probs.exp()
                entropy = -(probs * all_log_probs).sum(dim=-1).mean()

                rank_loss = None
                if rank_coef_value > 0:
                    rank_loss = compute_listmle_loss_vectorized(logits, rewards_norm, mask)

                loss = pg_loss - self.entropy_coef * entropy + self.kl_coef * kl_div
                if rank_loss is not None:
                    loss = loss + rank_coef_value * rank_loss

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"[Warning] NaN/Inf loss at step {step} (ppo_iter {ppo_epoch}), skipping")
                    continue

                self._backward_with_accum(loss, count_global_step=True)

                total_loss += loss.item()
                total_pg_loss += pg_loss.item()
                total_entropy += entropy.item()
                total_kl += kl_div.item()
                if rank_loss is not None:
                    total_rank_loss += rank_loss.item()
                total_updates += 1

                if ppo_epoch == self.ppo_epochs - 1:
                    rank_loss_value = rank_loss.item() if rank_loss is not None else 0.0
                    ratio_mean = ratio.mean().item()
                    ratio_std = ratio.std(unbiased=False).item()
                    step_metrics = {
                        "loss": loss.item(),
                        "pg_loss": pg_loss.item(),
                        "rank_loss": rank_loss_value,
                        "entropy": entropy.item(),
                        "kl": kl_div.item(),
                        "avg_return": avg_return_value,
                        "rank_corr": rank_corr_value,
                        "ratio_mean": ratio_mean,
                        "ratio_std": ratio_std,
                        "advantage_mean": adv_mean_value,
                        "advantage_std": adv_std_value,
                        "turnover_penalty": turnover_penalty_value,
                        "lr": self.optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                        "epoch_step": step,
                        "train_top_k": self.current_train_k,
                        "rank_coef": rank_coef_value,
                        "kl_coef": self.kl_coef,
                    }
                    self.logger.log_metrics("train_step", step_metrics, step=self.global_step)
                    pbar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        rank_corr=f"{rank_corr_value:.4f}",
                        ret=f"{avg_return_value:.5f}",
                    )

        self._flush_optimizer(count_global_step=True)
        n_updates = max(total_updates, 1)
        n_batches = max(batch_count, 1)
        avg_rank_loss = total_rank_loss / n_updates if rank_coef_value > 0 else 0.0
        return {
            "loss": total_loss / n_updates,
            "pg_loss": total_pg_loss / n_updates,
            "rank_loss": avg_rank_loss,
            "entropy": total_entropy / n_updates,
            "kl": total_kl / n_updates,
            "avg_return": total_return / n_batches,
            "rank_corr": total_rank_corr / n_batches,
            "turnover_penalty": total_turnover_penalty / n_batches,
        }

    def evaluate(self, dataset: DailyBatchDataset, stage: str, epoch: int) -> Tuple[pd.DataFrame, Dict[str, float]]:
        trades, summary = run_backtest(
            model=self.model, dataset=dataset, device=self.device,
            top_k=int(self.backtest_cfg.get("top_k", 10)),
            temperature=float(self.backtest_cfg.get("temperature", 1.0)),
            min_weight=float(self.backtest_cfg.get("min_weight", 0.0)),
            commission=float(self.backtest_cfg.get("commission", 0.0)),
            slippage=float(self.backtest_cfg.get("slippage", 0.0)),
            reward_scale=self.reward_scale,
        )

        stage_dir = ensure_dir(self.work_dir / stage)
        save_trades(trades, stage_dir / f"{stage}_trades.csv")
        self.logger.log_metrics(stage, {"epoch": epoch, **summary})

        with (stage_dir / f"{stage}_metrics_epoch_{epoch}.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return trades, summary

    def _save_checkpoint(self, name: str, epoch: int):
        ckpt = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": self.config,
            "global_step": self.global_step,
        }
        torch.save(ckpt, self.work_dir / name)


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train Transformer policy with direct return optimization")
    parser.add_argument("--config", type=Path, default=Path("pipelines/transformer_grpo/config_cn_t1.yaml"))
    args = parser.parse_args()
    trainer = Trainer(load_config(args.config))
    trainer.train()


if __name__ == "__main__":
    main()
