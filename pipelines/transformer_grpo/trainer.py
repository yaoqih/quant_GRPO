from __future__ import annotations

import argparse
import json
from datetime import datetime
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from .logger import LoggerFactory
from .model import TransformerPolicy
from .utils import collate_daily_batches, ensure_dir, set_global_seed


class GRPOTrainer:
    def __init__(self, config: Dict):
        self.config = config
        self.train_cfg = config.get("training", {})
        self.model_cfg = config.get("model", {})
        self.data_cfg = config.get("data", {})
        self.output_root = Path(self.train_cfg.get("checkpoint_root", "runs/transformer_grpo"))
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.work_dir = ensure_dir(self.output_root / self.timestamp)
        self.history_file = self.work_dir / "metrics.jsonl"
        self.backtest_cfg = config.get("backtest", {})
        logger_cfg = self.train_cfg.get("logger") or {}
        self.logger = LoggerFactory(logger_cfg, config, self.work_dir).build()

        seed = int(self.train_cfg.get("seed", 7))
        set_global_seed(seed)

        qlib.init(**config.get("qlib", {}))
        self.device = self._resolve_device(self.train_cfg.get("device", "auto"))

        handler_cfg = self.data_cfg.get("handler")
        if handler_cfg is None:
            raise ValueError("`data.handler` must be provided in the config file.")
        segments = self.data_cfg.get("segments", {})

        reward_clip = self.data_cfg.get("reward", {}).get("clip")
        if reward_clip is not None:
            reward_clip = (float(reward_clip[0]), float(reward_clip[1]))

        max_instruments = self.data_cfg.get("max_instruments")
        if max_instruments is not None:
            max_instruments = int(max_instruments)

        self.data_factory = DailyBatchFactory(
            handler_config=handler_cfg,
            segments=segments,
            feature_group=self.data_cfg.get("feature_group", "feature"),
            label_group=self.data_cfg.get("label_group", "label"),
            label_name=self.data_cfg.get("label_name", "LABEL0"),
            min_instruments=int(self.data_cfg.get("min_instruments", 10)),
            max_instruments=max_instruments,
            reward_clip=reward_clip,
            reward_scale=float(self.data_cfg.get("reward", {}).get("scale", 1.0)),
            instrument_universe=self.data_cfg.get("instrument_universe"),
            augment=self.data_cfg.get("augment"),
            cache_config=self.data_cfg.get("cache"),
            feature_dtype=self.data_cfg.get("feature_dtype", "float32"),
        )

        self.train_dataset = self.data_factory.build_segment("train")
        if len(self.train_dataset) == 0:
            raise ValueError("Training segment produced zero samples. Please check the data configuration.")
        self.valid_dataset = (
            self.data_factory.build_segment("valid") if "valid" in segments else DailyBatchDataset([])
        )
        self.test_dataset = (
            self.data_factory.build_segment("test") if "test" in segments else DailyBatchDataset([])
        )

        feature_dim = self.train_dataset.feature_dim
        if feature_dim == 0:
            raise ValueError("Unable to infer feature dimension from the dataset.")
        inferred_max_positions = int(getattr(self.train_dataset, "max_instruments", 0) or 0) + 10
        if inferred_max_positions > 0:
            current_max_positions = int(self.model_cfg.get("max_positions", 0) or 0)
            if current_max_positions <= 0 or inferred_max_positions > current_max_positions:
                self.model_cfg["max_positions"] = inferred_max_positions
        temporal_span = int(getattr(self.train_dataset, "temporal_span", 1) or 1)
        if temporal_span > 1 and "temporal_span" not in self.model_cfg:
            self.model_cfg["temporal_span"] = temporal_span

        self.model = TransformerPolicy(feature_dim=feature_dim, **self.model_cfg).to(self.device)
        self.optimizer = self._build_optimizer()
        self.scheduler = None
        self.entropy_coef = float(self.train_cfg.get("entropy_coef", 0.01))
        self.value_coef = float(self.train_cfg.get("value_coef", 0.5))
        self.grad_clip = float(self.train_cfg.get("grad_clip", 1.0))
        self.adv_temperature = float(self.train_cfg.get("adv_temperature", 1.0))
        self.clip_range = float(self.train_cfg.get("clip_range", 0.2))
        self.kl_coef = float(self.train_cfg.get("kl_coef", 0.0))
        self.kl_target = float(self.train_cfg.get("kl_target", 0.0))
        self.kl_beta = float(self.train_cfg.get("kl_beta", 1.5))
        self.log_ratio_clip = float(self.train_cfg.get("log_ratio_clip", 5.0))
        self.ref_sync_interval = int(self.train_cfg.get("ref_sync_interval", 0))
        self.ref_sync_alpha = float(self.train_cfg.get("ref_sync_alpha", 1.0))
        self.turnover_coef = float(self.train_cfg.get("turnover_coef", 0.0))
        self.turnover_quad_coef = float(self.train_cfg.get("turnover_quad_coef", 0.0))
        self.drawdown_coef = float(self.train_cfg.get("drawdown_coef", 0.0))
        self.volatility_coef = float(self.train_cfg.get("volatility_coef", 0.0))
        self.max_drawdown_target = float(self.train_cfg.get("max_drawdown_target", 1.0))
        self.early_stop_patience = int(self.train_cfg.get("early_stop_patience", 0))
        self.nav_window = int(self.train_cfg.get("nav_window", 32))
        self.pretrain_epochs = int(self.train_cfg.get("pretrain_epochs", 0))
        self.pretrain_temperature = float(self.train_cfg.get("pretrain_temperature", 0.1))
        self.greedy_eval = bool(self.train_cfg.get("greedy_eval", True))
        self.eval_temperature = float(self.train_cfg.get("eval_temperature", 1.0))
        self.risk_free = float(self.train_cfg.get("risk_free", 0.02))
        self.monitor_metric = self.train_cfg.get("monitor", "sharpe")
        self.log_interval = int(self.train_cfg.get("log_interval", 25))
        self.num_workers = int(self.train_cfg.get("num_workers", 0))
        self.global_step = 0
        self.use_cosine_lr = bool(self.train_cfg.get("use_cosine_lr", False))
        self.ema_decay = float(self.train_cfg.get("ema_decay", 1.0))
        self.eval_with_ema = bool(self.train_cfg.get("eval_with_ema", False))
        self.no_improve_epochs = 0
        self.drawdown_patience = 0
        self.chronological_batches = bool(self.train_cfg.get("chronological_batches", False))
        default_trade_cost = float(self.backtest_cfg.get("commission", 0.0)) + float(
            self.backtest_cfg.get("slippage", 0.0)
        )
        self.transaction_cost_rate = float(self.train_cfg.get("transaction_cost", default_trade_cost))

        self.reference_model = None
        if self.ref_sync_interval > 0 or self.kl_coef > 0:
            self.reference_model = TransformerPolicy(feature_dim=feature_dim, **self.model_cfg).to(self.device)
            for param in self.reference_model.parameters():
                param.requires_grad = False
            self._sync_reference(hard=True)

        self.ema_model = None
        if self.ema_decay < 1.0 or self.eval_with_ema:
            self.ema_model = TransformerPolicy(feature_dim=feature_dim, **self.model_cfg).to(self.device)
            self.ema_model.load_state_dict(self.model.state_dict())
            for param in self.ema_model.parameters():
                param.requires_grad = False

        self.walk_forward_sets: Dict[str, DailyBatchDataset] = {}
        for name, seg in self.data_cfg.get("walk_forward_segments", {}).items():
            try:
                dataset = self.data_factory.build_segment(seg)
            except Exception as exc:  # noqa: BLE001
                print(f"[WalkForward] skip segment {name}: {exc}")
                continue
            if len(dataset) > 0:
                self.walk_forward_sets[name] = dataset

        self.start_epoch = 1
        resume_path = self.train_cfg.get("resume_from")
        if resume_path:
            self._load_checkpoint(Path(resume_path))

        self._save_config()
        self._pretrain_policy()

    def _save_config(self) -> None:
        config_path = self.work_dir / "config.yaml"
        with config_path.open("w", encoding="utf-8") as fout:
            yaml.safe_dump(self.config, fout, allow_unicode=True)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.train_cfg.get("learning_rate", 3e-4)),
            weight_decay=float(self.train_cfg.get("weight_decay", 0.0)),
        )

    def _resolve_device(self, requested: str) -> torch.device:
        if requested == "cpu":
            return torch.device("cpu")
        if torch.cuda.is_available() and requested in ("auto", "cuda", "gpu"):
            return torch.device("cuda")
        return torch.device("cpu")

    def _build_loader(self, dataset: DailyBatchDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=int(self.train_cfg.get("batch_size", 16)),
            shuffle=shuffle,
            collate_fn=collate_daily_batches,
            num_workers=self.num_workers,
            drop_last=False,
        )

    def train(self) -> None:
        train_loader = self._build_loader(self.train_dataset, shuffle=not self.chronological_batches)
        epochs = int(self.train_cfg.get("epochs", 20))
        best_metric = float("-inf")
        best_epoch = 0
        last_epoch = self.start_epoch - 1
        if self.use_cosine_lr:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(epochs - self.start_epoch + 1, 1),
            )

        try:
            for epoch in range(self.start_epoch, epochs + 1):
                train_stats = self._run_epoch(train_loader, epoch)
                self.logger.log_metrics("train", {"epoch": epoch, **train_stats})
                val_stats = {}
                if len(self.valid_dataset) > 0:
                    _, val_stats = self.evaluate(
                        self.valid_dataset,
                        stage="valid",
                        epoch=epoch,
                        model=self._get_eval_model(),
                    )
                    metric_val = val_stats.get(self.monitor_metric)
                    if metric_val is not None and metric_val > best_metric:
                        best_metric = metric_val
                        best_epoch = epoch
                        self._save_checkpoint("best.pt", epoch)
                        self.no_improve_epochs = 0
                    else:
                        self.no_improve_epochs += 1
                    drawdown = val_stats.get("max_drawdown")
                    if drawdown is not None and drawdown > self.max_drawdown_target:
                        self.drawdown_patience += 1
                    else:
                        self.drawdown_patience = 0
                    self.logger.log_metrics("valid", {"epoch": epoch, **val_stats})
                else:
                    self.no_improve_epochs = 0
                    self.drawdown_patience = 0

                record = {
                    "epoch": epoch,
                    "train": train_stats,
                    "valid": val_stats,
                }
                with self.history_file.open("a", encoding="utf-8") as fout:
                    fout.write(json.dumps(record, default=str) + "\n")

                print(
                    f"[Epoch {epoch}] train_loss={train_stats['loss']:.4f} "
                    f"reward={train_stats['avg_reward']:.5f} "
                    f"nav_dd={train_stats['nav_max_drawdown']:.3f}"
                )
                if val_stats:
                    print(f"[Epoch {epoch}] valid_{self.monitor_metric}={val_stats.get(self.monitor_metric, 0):.4f}")
                if self.scheduler is not None:
                    self.scheduler.step()
                last_epoch = epoch
                if (
                    self.early_stop_patience > 0
                    and len(self.valid_dataset) > 0
                    and (
                        self.no_improve_epochs >= self.early_stop_patience
                        or self.drawdown_patience >= self.early_stop_patience
                    )
                ):
                    print(f"[EarlyStop] stopping at epoch {epoch} due to plateau/drawdown.")
                    break

            self._save_checkpoint("last.pt", last_epoch)
            if len(self.test_dataset) > 0:
                self.evaluate(self.test_dataset, stage="test", epoch=last_epoch + 1, model=self._get_eval_model())
            for idx, (name, dataset) in enumerate(self.walk_forward_sets.items(), start=1):
                self.evaluate(
                    dataset,
                    stage=f"walk_{name}",
                    epoch=last_epoch + 1 + idx,
                    model=self._get_eval_model(),
                )

            if best_metric > float("-inf"):
                print(f"Best {self.monitor_metric}={best_metric:.4f} achieved at epoch {best_epoch}")
        finally:
            self.logger.close()

    def _run_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_pg = 0.0
        total_value = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        total_reward = 0.0
        total_expected = 0.0
        total_steps = 0
        total_turnover = 0.0
        total_realized = 0.0
        total_drawdown_pen = 0.0
        total_vol_pen = 0.0
        nav_value = 1.0
        nav_peak = 1.0
        nav_history = [nav_value]
        portfolio_returns: List[float] = []
        prev_allocations_map: Optional[Dict[str, float]] = None
        total_cost_sum = 0.0
        total_cost_count = 0

        for step, batch in enumerate(loader, start=1):
            features = batch["features"].to(self.device)
            rewards = batch["rewards"].to(self.device)
            mask = batch["mask"].to(self.device)

            logits, values = self.model(features, mask)
            masked_logits = self._mask_invalid(logits, mask)
            mask_f = mask.float()
            count = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)

            log_probs = F.log_softmax(masked_logits, dim=-1)
            probs = log_probs.exp()
            expected_reward = ((probs * rewards) * mask_f).sum(dim=-1, keepdim=True) / count

            if values is not None:
                baseline = values.detach()
            else:
                baseline = expected_reward
            centered = (rewards - baseline) * mask_f
            centered = centered - (centered.sum(dim=1, keepdim=True) / count)
            var = (centered ** 2).sum(dim=1, keepdim=True) / count
            std = torch.sqrt(var + 1e-6)
            advantages = centered / std
            if self.adv_temperature != 1.0:
                advantages = advantages / max(self.adv_temperature, 1e-4)
            advantages = advantages.detach()

            if self.reference_model is not None:
                with torch.no_grad():
                    ref_logits, _ = self.reference_model(features, mask)
                ref_log_probs = F.log_softmax(self._mask_invalid(ref_logits, mask), dim=-1)
            else:
                ref_log_probs = log_probs.detach()

            ref_probs = ref_log_probs.exp()

            log_ratio_raw = log_probs - ref_log_probs
            log_ratio = log_ratio_raw.clamp(min=-self.log_ratio_clip, max=self.log_ratio_clip)
            ratio = torch.exp(log_ratio)
            clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
            weighted_adv = ratio * advantages
            clipped_adv = clipped_ratio * advantages
            policy_loss = -torch.min(weighted_adv, clipped_adv).masked_select(mask).mean()

            if values is not None:
                value_loss = F.mse_loss(values.masked_select(mask), rewards.masked_select(mask))
            else:
                value_loss = torch.zeros(1, device=self.device)

            entropy = -(probs * log_probs).masked_select(mask).sum() / mask.sum()
            if self.kl_coef > 0 and self.reference_model is not None:
                kl = ((probs * log_ratio_raw) * mask_f).sum(dim=-1)
                kl = kl.sum() / mask.sum()
            else:
                kl = torch.zeros(1, device=self.device)

            turnover_penalty = torch.zeros(1, device=self.device)
            if self.turnover_coef > 0 or self.turnover_quad_coef > 0:
                prob_delta = (probs - ref_probs) * mask_f
                linear = (prob_delta.abs().sum(dim=-1, keepdim=True) / count).mean()
                quadratic = ((prob_delta ** 2).sum(dim=-1, keepdim=True) / count).mean()
                turnover_penalty = self.turnover_coef * linear + self.turnover_quad_coef * quadratic

            normalized = (probs * mask_f).clamp_min(0.0)
            norm_sum = normalized.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            normalized = normalized / norm_sum
            portfolio_reward = ((normalized * rewards) * mask_f).sum(dim=-1, keepdim=True) / count
            transaction_costs = torch.zeros_like(portfolio_reward)
            if self.chronological_batches:
                for sample_idx in range(portfolio_reward.size(0)):
                    valid_count = int(mask_f[sample_idx].sum().item())
                    if valid_count == 0:
                        continue
                    inst_array = batch["meta"][sample_idx]["instruments"][:valid_count]
                    instruments = [str(inst) for inst in inst_array]
                    weights_vec = normalized[sample_idx, :valid_count]
                    if prev_allocations_map is None:
                        prev_vec = torch.zeros(valid_count, dtype=weights_vec.dtype, device=weights_vec.device)
                    else:
                        prev_vec = torch.tensor(
                            [prev_allocations_map.get(inst, 0.0) for inst in instruments],
                            dtype=weights_vec.dtype,
                            device=weights_vec.device,
                        )
                    turnover_actual = torch.abs(weights_vec - prev_vec).sum()
                    transaction_costs[sample_idx, 0] = self.transaction_cost_rate * turnover_actual
                    prev_allocations_map = {
                        inst: float(weights_vec[idx].detach().cpu()) for idx, inst in enumerate(instruments)
                    }
            portfolio_reward = portfolio_reward - transaction_costs
            cost_penalty = transaction_costs.mean()
            realized = portfolio_reward.mean()
            total_cost_sum += float(transaction_costs.sum().item())
            total_cost_count += transaction_costs.shape[0]

            if self.chronological_batches:
                realized_series = portfolio_reward.detach().view(-1)
                for value in realized_series:
                    nav_value *= float(1.0 + value.item())
                    nav_peak = max(nav_peak, nav_value)
                    nav_history.append(nav_value)
                    portfolio_returns.append(float(value.item()))
                drawdown = (nav_peak - nav_value) / max(nav_peak, 1e-8)
            else:
                nav_value *= float(1.0 + realized.item())
                nav_peak = max(nav_peak, nav_value)
                nav_history.append(nav_value)
                drawdown = (nav_peak - nav_value) / max(nav_peak, 1e-8)
                portfolio_returns.append(realized.item())

            drawdown_penalty = torch.zeros(1, device=self.device)
            if self.drawdown_coef > 0 and drawdown > self.max_drawdown_target:
                drawdown_penalty = torch.tensor(
                    (drawdown - self.max_drawdown_target) * self.drawdown_coef,
                    device=self.device,
                )

            volatility_penalty = torch.zeros(1, device=self.device)
            if self.volatility_coef > 0:
                centered_portfolio = portfolio_reward - portfolio_reward.mean()
                volatility_penalty = self.volatility_coef * centered_portfolio.pow(2).mean()

            loss = (
                policy_loss
                + self.value_coef * value_loss
                - self.entropy_coef * entropy
                + self.kl_coef * kl
                + turnover_penalty
                + drawdown_penalty
                + volatility_penalty
                + cost_penalty
            )

            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip is not None and self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.global_step += 1
            self._maybe_sync_reference()
            self._update_ema()
            self._adapt_kl_coef(float(kl.item()))

            total_loss += loss.item()
            total_pg += policy_loss.item()
            total_value += value_loss.item()
            total_entropy += entropy.item()
            total_kl += kl.item()
            total_reward += (rewards.masked_select(mask)).mean().item()
            total_expected += expected_reward.mean().item()
            total_turnover += turnover_penalty.item()
            total_realized += realized.item()
            total_drawdown_pen += drawdown_penalty.item()
            total_vol_pen += volatility_penalty.item()
            total_steps += 1

            if step % self.log_interval == 0:
                print(
                    f"Epoch {epoch} Step {step} "
                    f"loss={loss.item():.4f} pg={policy_loss.item():.4f} "
                    f"value={value_loss.item():.4f} entropy={entropy.item():.4f} "
                    f"kl={kl.item():.4f} turnover={turnover_penalty.item():.6f} "
                    f"cost={cost_penalty.item():.6f}"
                )
                if self.logger.supports_step_metrics:
                    step_metrics = {
                        "loss": float(loss.item()),
                        "policy_loss": float(policy_loss.item()),
                        "value_loss": float(value_loss.item()),
                        "entropy": float(entropy.item()),
                        "kl": float(kl.item()),
                        "turnover_penalty": float(turnover_penalty.item()),
                        "realized_reward": float(realized.item()),
                        "transaction_cost": float(cost_penalty.item()),
                    }
                    self.logger.log_metrics("train_step", step_metrics, step=self.global_step)

        avg = lambda x: x / max(total_steps, 1)
        nav_arr = np.array(nav_history, dtype=np.float32)
        running_max = np.maximum.accumulate(nav_arr)
        nav_drawdown = float(np.max((running_max - nav_arr) / np.clip(running_max, 1e-8, None)))
        if portfolio_returns:
            recent = np.array(portfolio_returns[-self.nav_window :], dtype=np.float32)
            nav_volatility = float(np.std(recent) * math.sqrt(252.0))
        else:
            nav_volatility = 0.0
        return {
            "loss": avg(total_loss),
            "policy_loss": avg(total_pg),
            "value_loss": avg(total_value),
            "entropy": avg(total_entropy),
            "kl": avg(total_kl),
            "expected_reward": avg(total_expected),
            "avg_reward": avg(total_reward),
            "turnover_penalty": avg(total_turnover),
            "realized_reward": avg(total_realized),
            "drawdown_penalty": avg(total_drawdown_pen),
            "volatility_penalty": avg(total_vol_pen),
            "transaction_cost": (total_cost_sum / max(total_cost_count, 1)),
            "nav_max_drawdown": nav_drawdown,
            "nav_volatility": nav_volatility,
        }

    def evaluate(
        self,
        dataset: DailyBatchDataset,
        stage: str,
        epoch: int,
        model: Optional[TransformerPolicy] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        if len(dataset) == 0:
            return pd.DataFrame(), {}
        eval_model = model or self._get_eval_model()
        trades, summary = run_backtest(
            model=eval_model,
            dataset=dataset,
            device=self.device,
            risk_free=self.risk_free,
            temperature=self.eval_temperature,
            greedy=self.greedy_eval,
            commission=float(self.backtest_cfg.get("commission", 0.0)),
            slippage=float(self.backtest_cfg.get("slippage", 0.0)),
            top_k=int(self.backtest_cfg.get("top_k", 1)),
            hold_threshold=(
                float(self.backtest_cfg.get("hold_threshold"))
                if self.backtest_cfg.get("hold_threshold") is not None
                else None
            ),
            min_weight=float(self.backtest_cfg.get("min_weight", 0.0)),
            cash_token=self.backtest_cfg.get("cash_token"),
            max_gross_exposure=float(self.backtest_cfg.get("max_gross_exposure", 1.0)),
        )
        stage_dir = ensure_dir(self.work_dir / stage)
        save_trades(trades, stage_dir / f"{stage}_trades.csv")
        payload = {"epoch": epoch, **summary}
        self.logger.log_metrics(stage, payload)
        self.logger.log_trades(stage, stage_dir / f"{stage}_trades.csv")
        metrics_path = stage_dir / f"{stage}_metrics_epoch_{epoch}.json"
        with metrics_path.open("w", encoding="utf-8") as fout:
            json.dump(summary, fout, indent=2)
        return trades, summary

    def _save_checkpoint(self, name: str, epoch: int) -> None:
        ckpt = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": self.config,
            "global_step": self.global_step,
        }
        if self.reference_model is not None:
            ckpt["reference_state"] = self.reference_model.state_dict()
        if self.ema_model is not None:
            ckpt["ema_state"] = self.ema_model.state_dict()
        torch.save(ckpt, self.work_dir / name)

    def _load_checkpoint(self, path: Path) -> None:
        ckpt = torch.load(path, map_location="cpu")
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.start_epoch = ckpt["epoch"] + 1
        self.global_step = ckpt.get("global_step", 0)
        if self.reference_model is not None and "reference_state" in ckpt:
            self.reference_model.load_state_dict(ckpt["reference_state"])
        if self.ema_model is not None and "ema_state" in ckpt:
            self.ema_model.load_state_dict(ckpt["ema_state"])

    @staticmethod
    def _mask_invalid(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        fill_value = torch.finfo(logits.dtype).min
        return logits.masked_fill(~mask, fill_value)

    def _maybe_sync_reference(self) -> None:
        if self.reference_model is None or self.ref_sync_interval <= 0:
            return
        if self.global_step % self.ref_sync_interval != 0:
            return
        self._sync_reference()

    def _sync_reference(self, hard: bool = False) -> None:
        if self.reference_model is None:
            return
        with torch.no_grad():
            if hard:
                self.reference_model.load_state_dict(self.model.state_dict())
            else:
                for ref_param, param in zip(self.reference_model.parameters(), self.model.parameters()):
                    ref_param.data.lerp_(param.data, self.ref_sync_alpha)
        self.reference_model.eval()

    def _pretrain_policy(self) -> None:
        if self.pretrain_epochs <= 0:
            return
        loader = self._build_loader(self.train_dataset, shuffle=True)
        temperature = max(self.pretrain_temperature, 1e-3)
        print(f"[Pretrain] start supervised warm-up for {self.pretrain_epochs} epoch(s)")
        for epoch in range(1, self.pretrain_epochs + 1):
            total_loss = 0.0
            total_policy = 0.0
            total_value = 0.0
            steps = 0
            self.model.train()
            for batch in tqdm(loader,desc=f"[Pretrain] training epoch {epoch}"):
                features = batch["features"].to(self.device)
                rewards = batch["rewards"].to(self.device)
                mask = batch["mask"].to(self.device)

                logits, values = self.model(features, mask)
                masked_logits = self._mask_invalid(logits, mask)
                log_probs = F.log_softmax(masked_logits, dim=-1)

                scaled_reward = rewards / temperature
                scaled_logits = self._mask_invalid(scaled_reward, mask)
                target_probs = F.softmax(scaled_logits, dim=-1)

                policy_loss = -(target_probs * log_probs).masked_select(mask).sum() / mask.sum()
                if values is not None:
                    value_loss = F.mse_loss(values.masked_select(mask), rewards.masked_select(mask))
                else:
                    value_loss = torch.zeros(1, device=self.device)
                loss = policy_loss + self.value_coef * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                if self.grad_clip is not None and self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                total_loss += loss.item()
                total_policy += policy_loss.item()
                total_value += value_loss.item()
                steps += 1
            avg_loss = total_loss / max(steps, 1)
            avg_policy = total_policy / max(steps, 1)
            avg_value = total_value / max(steps, 1)
            print(f"[Pretrain {epoch}] loss={avg_loss:.4f} policy={avg_policy:.4f} value={avg_value:.4f}")
            self.logger.log_metrics(
                "pretrain",
                {"loss": avg_loss, "policy_loss": avg_policy, "value_loss": avg_value, "epoch": epoch},
            )
        self.optimizer = self._build_optimizer()
        self._sync_reference(hard=True)

    def _adapt_kl_coef(self, kl_value: float) -> None:
        if self.kl_target <= 0 or self.kl_beta <= 1.0:
            return
        if kl_value > self.kl_target * 1.5:
            self.kl_coef = min(self.kl_coef * self.kl_beta, 1.0)
        elif kl_value < self.kl_target / 1.5:
            self.kl_coef = max(self.kl_coef / self.kl_beta, 1e-6)

    def _update_ema(self) -> None:
        if self.ema_model is None or self.ema_decay >= 1.0:
            return
        decay = self.ema_decay
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)

    def _get_eval_model(self) -> TransformerPolicy:
        if self.eval_with_ema and self.ema_model is not None:
            return self.ema_model
        return self.model


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fin:
        return yaml.safe_load(fin)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Transformer + GRPO policy.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("pipelines/transformer_grpo/config_cn_t1.yaml"),
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    trainer = GRPOTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
