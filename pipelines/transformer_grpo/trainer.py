from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml
import qlib

from .backtest import run_backtest, save_trades
from .data_pipeline import DailyBatchDataset, DailyBatchFactory
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
        self.pretrain_epochs = int(self.train_cfg.get("pretrain_epochs", 0))
        self.pretrain_temperature = float(self.train_cfg.get("pretrain_temperature", 0.1))
        self.greedy_eval = bool(self.train_cfg.get("greedy_eval", True))
        self.eval_temperature = float(self.train_cfg.get("eval_temperature", 1.0))
        self.risk_free = float(self.train_cfg.get("risk_free", 0.02))
        self.monitor_metric = self.train_cfg.get("monitor", "sharpe")
        self.log_interval = int(self.train_cfg.get("log_interval", 25))
        self.num_workers = int(self.train_cfg.get("num_workers", 0))
        self.global_step = 0

        self.reference_model = None
        if self.ref_sync_interval > 0 or self.kl_coef > 0:
            self.reference_model = TransformerPolicy(feature_dim=feature_dim, **self.model_cfg).to(self.device)
            for param in self.reference_model.parameters():
                param.requires_grad = False
            self._sync_reference(hard=True)

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
        train_loader = self._build_loader(self.train_dataset, shuffle=True)
        epochs = int(self.train_cfg.get("epochs", 20))
        best_metric = float("-inf")
        best_epoch = 0

        for epoch in range(self.start_epoch, epochs + 1):
            train_stats = self._run_epoch(train_loader, epoch)
            val_stats = {}
            if len(self.valid_dataset) > 0:
                _, val_stats = self.evaluate(self.valid_dataset, stage="valid", epoch=epoch)
                metric_val = val_stats.get(self.monitor_metric)
                if metric_val is not None and metric_val > best_metric:
                    best_metric = metric_val
                    best_epoch = epoch
                    self._save_checkpoint("best.pt", epoch)

            record = {
                "epoch": epoch,
                "train": train_stats,
                "valid": val_stats,
            }
            with self.history_file.open("a", encoding="utf-8") as fout:
                fout.write(json.dumps(record, default=str) + "\n")

            print(f"[Epoch {epoch}] train_loss={train_stats['loss']:.4f} reward={train_stats['avg_reward']:.5f}")
            if val_stats:
                print(f"[Epoch {epoch}] valid_{self.monitor_metric}={val_stats.get(self.monitor_metric, 0):.4f}")

        self._save_checkpoint("last.pt", epochs)
        if len(self.test_dataset) > 0:
            self.evaluate(self.test_dataset, stage="test", epoch=epochs + 1)

        if best_metric > float("-inf"):
            print(f"Best {self.monitor_metric}={best_metric:.4f} achieved at epoch {best_epoch}")

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
            if self.turnover_coef > 0:
                prob_delta = torch.abs(probs - ref_probs) * mask_f
                turnover_penalty = (prob_delta.sum(dim=-1, keepdim=True) / count).mean() * self.turnover_coef

            loss = (
                policy_loss
                + self.value_coef * value_loss
                - self.entropy_coef * entropy
                + self.kl_coef * kl
                + turnover_penalty
            )

            self.optimizer.zero_grad()
            loss.backward()
            if self.grad_clip is not None and self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.global_step += 1
            self._maybe_sync_reference()
            self._adapt_kl_coef(float(kl.item()))

            with torch.no_grad():
                normalized = (probs * mask_f).clamp_min(0.0)
                norm_sum = normalized.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                normalized = normalized / norm_sum
                actions = torch.multinomial(normalized, num_samples=1)
                realized = torch.gather(rewards, 1, actions).mean()

            total_loss += loss.item()
            total_pg += policy_loss.item()
            total_value += value_loss.item()
            total_entropy += entropy.item()
            total_kl += kl.item()
            total_reward += (rewards.masked_select(mask)).mean().item()
            total_expected += expected_reward.mean().item()
            total_turnover += turnover_penalty.item()
            total_realized += realized.item()
            total_steps += 1

            if step % self.log_interval == 0:
                print(
                    f"Epoch {epoch} Step {step} "
                    f"loss={loss.item():.4f} pg={policy_loss.item():.4f} "
                    f"value={value_loss.item():.4f} entropy={entropy.item():.4f} "
                    f"kl={kl.item():.4f} turnover={turnover_penalty.item():.6f}"
                )

        avg = lambda x: x / max(total_steps, 1)
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
        }

    def evaluate(self, dataset: DailyBatchDataset, stage: str, epoch: int) -> Tuple[pd.DataFrame, Dict[str, float]]:
        if len(dataset) == 0:
            return pd.DataFrame(), {}
        trades, summary = run_backtest(
            model=self.model,
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
        )
        stage_dir = ensure_dir(self.work_dir / stage)
        save_trades(trades, stage_dir / f"{stage}_trades.csv")
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
        torch.save(ckpt, self.work_dir / name)

    def _load_checkpoint(self, path: Path) -> None:
        ckpt = torch.load(path, map_location="cpu")
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.start_epoch = ckpt["epoch"] + 1
        self.global_step = ckpt.get("global_step", 0)
        if self.reference_model is not None and "reference_state" in ckpt:
            self.reference_model.load_state_dict(ckpt["reference_state"])

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
            for batch in loader:
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
        self.optimizer = self._build_optimizer()
        self._sync_reference(hard=True)

    def _adapt_kl_coef(self, kl_value: float) -> None:
        if self.kl_target <= 0 or self.kl_beta <= 1.0:
            return
        if kl_value > self.kl_target * 1.5:
            self.kl_coef = min(self.kl_coef * self.kl_beta, 1.0)
        elif kl_value < self.kl_target / 1.5:
            self.kl_coef = max(self.kl_coef / self.kl_beta, 1e-6)


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
