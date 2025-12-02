from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional


class BaseLogger:
    """No-op fallback when no experiment tracker is requested."""

    supports_step_metrics: bool = False

    def log_metrics(self, stage: str, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        return

    def log_trades(self, stage: str, trades_path: Path) -> None:
        return

    def close(self) -> None:
        return


class WandbLogger(BaseLogger):
    """Thin wrapper around wandb so training metrics can be visualised online/offline."""

    supports_step_metrics = True

    def __init__(self, cfg: Dict, full_config: Dict, work_dir: Path):
        try:
            import wandb
        except ImportError as exc:  # noqa: BLE001
            raise RuntimeError("wandb logger requested but wandb is not installed") from exc

        project = cfg.get("project", "transformer_grpo")
        run_name = cfg.get("run_name")
        mode = cfg.get("mode")  # online/offline/disabled
        entity = cfg.get("entity")
        tags = cfg.get("tags")
        settings = cfg.get("settings") or {}
        init_kwargs = {
            "project": project,
            "config": full_config,
            "dir": str(work_dir),
        }
        if run_name:
            init_kwargs["name"] = run_name
        if mode:
            init_kwargs["mode"] = mode
        if entity:
            init_kwargs["entity"] = entity
        if tags:
            init_kwargs["tags"] = tags
        if settings:
            init_kwargs["settings"] = settings
        self._wandb = wandb
        self._run = wandb.init(**init_kwargs)
        self._log_trades = bool(cfg.get("log_trades", False))
        self._last_step: int = -1

    def log_metrics(self, stage: str, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        if not metrics:
            return
        if self._run is None:
            raise RuntimeError(
                "wandb logger is enabled but wandb.init() did not return a run. "
                "Please ensure `wandb.login()` completes successfully before training."
            )
        payload = {f"{stage}/{key}": value for key, value in metrics.items()}
        prev_step = self._last_step
        if step is not None:
            payload["global_step"] = step
            log_step = step if step > prev_step else prev_step + 1
        else:
            log_step = prev_step + 1
        self._last_step = log_step
        self._run.log(payload, step=log_step)

    def log_trades(self, stage: str, trades_path: Path) -> None:
        if self._run is None or not self._log_trades or not trades_path.exists():
            return
        artifact = self._wandb.Artifact(name=f"{stage}_trades", type="trades")
        artifact.add_file(str(trades_path))
        self._run.log_artifact(artifact)

    def close(self) -> None:
        if self._run is not None:
            self._run.finish()


class TensorBoardLogger(BaseLogger):
    """TensorBoard SummaryWriter backed logger."""

    supports_step_metrics = True

    def __init__(self, cfg: Dict, work_dir: Path):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:  # noqa: BLE001
            raise RuntimeError("TensorBoard logger requested but torch.utils.tensorboard is unavailable") from exc
        log_dir = cfg.get("log_dir")
        if log_dir is None:
            log_dir = work_dir / "tensorboard"
        self._writer = SummaryWriter(log_dir=str(log_dir))

    def log_metrics(self, stage: str, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        if not metrics:
            return
        for key, value in metrics.items():
            self._writer.add_scalar(f"{stage}/{key}", value, global_step=step)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()


@dataclass
class LoggerFactory:
    config: Dict
    full_config: Dict
    work_dir: Path

    def build(self) -> BaseLogger:
        if not self.config:
            return BaseLogger()
        kind = (self.config.get("type") or "").lower()
        if kind == "wandb":
            return WandbLogger(self.config, self.full_config, self.work_dir)
        if kind in {"tensorboard", "tb"}:
            return TensorBoardLogger(self.config, self.work_dir)
        return BaseLogger()
