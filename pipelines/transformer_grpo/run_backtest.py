from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import qlib

from .backtest import run_backtest, save_trades
from .data_pipeline import DailyBatchFactory
from .model import TransformerPolicy
from .trainer import load_config
from .utils import ensure_dir
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved policy checkpoint.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("pipelines/transformer_grpo/config_cn_t1.yaml"),
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--segment", type=str, default=None)
    parser.add_argument("--out_dir", type=Path, default=Path("runs/eval"))
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--greedy", action="store_true", help="Disable sampling and pick the top action each day.")
    parser.add_argument("--top_k", type=int, default=None, help="Override backtest.top_k")
    parser.add_argument("--hold_threshold", type=float, default=None, help="Override backtest.hold_threshold")
    parser.add_argument("--min_weight", type=float, default=None, help="Override backtest.min_weight")
    parser.add_argument("--cash_token", type=str, default=None, help="Override backtest.cash_token")
    parser.add_argument(
        "--max_gross_exposure",
        type=float,
        default=None,
        help="Override backtest.max_gross_exposure",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.checkpoint is None:
        checkpoint=os.listdir(cfg['training']['checkpoint_root'][-1]/'best.pt')
        if not os.path.exists():
            raise ValueError("Must specify a checkpoint with --checkpoint")
    qlib.init(**cfg.get("qlib", {}))
    backtest_cfg = cfg.get("backtest", {})
    segment = args.segment or backtest_cfg.get("segment", "test")

    handler_cfg = cfg["data"]["handler"]
    segments = cfg["data"]["segments"]
    reward_clip = cfg["data"].get("reward", {}).get("clip")
    if reward_clip is not None:
        reward_clip = (float(reward_clip[0]), float(reward_clip[1]))
    max_instruments = cfg["data"].get("max_instruments")
    if max_instruments is not None:
        max_instruments = int(max_instruments)

    factory = DailyBatchFactory(
        handler_config=handler_cfg,
        segments=segments,
        feature_group=cfg["data"].get("feature_group", "feature"),
        label_group=cfg["data"].get("label_group", "label"),
        label_name=cfg["data"].get("label_name", "LABEL0"),
        min_instruments=int(cfg["data"].get("min_instruments", 10)),
        max_instruments=max_instruments,
        reward_clip=reward_clip,
        reward_scale=float(cfg["data"].get("reward", {}).get("scale", 1.0)),
        instrument_universe=cfg["data"].get("instrument_universe"),
        augment=cfg["data"].get("augment"),
    )

    dataset = factory.build_segment(segment)
    if len(dataset) == 0:
        raise ValueError(f"Segment '{segment}' produced zero samples.")

    feature_dim = dataset.feature_dim
    model = TransformerPolicy(feature_dim=feature_dim, **cfg.get("model", {}))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    top_k = int(args.top_k) if args.top_k is not None else int(backtest_cfg.get("top_k", 1))
    hold_threshold = (
        args.hold_threshold
        if args.hold_threshold is not None
        else backtest_cfg.get("hold_threshold")
    )
    if hold_threshold is not None:
        hold_threshold = float(hold_threshold)
    min_weight = (
        args.min_weight
        if args.min_weight is not None
        else float(backtest_cfg.get("min_weight", 0.0))
    )
    cash_token = args.cash_token if args.cash_token is not None else backtest_cfg.get("cash_token")
    max_gross_exposure = (
        args.max_gross_exposure
        if args.max_gross_exposure is not None
        else float(backtest_cfg.get("max_gross_exposure", 1.0))
    )

    trades, summary = run_backtest(
        model=model,
        dataset=dataset,
        device=device,
        risk_free=float(cfg.get("training", {}).get("risk_free", 0.02)),
        temperature=args.temperature,
        greedy=args.greedy,
        commission=float(backtest_cfg.get("commission", 0.0)),
        slippage=float(backtest_cfg.get("slippage", 0.0)),
        top_k=top_k,
        hold_threshold=hold_threshold,
        min_weight=min_weight,
        cash_token=cash_token,
        max_gross_exposure=max_gross_exposure,
    )

    out_dir = ensure_dir(args.out_dir)
    save_trades(trades, out_dir / f"{segment}_trades.csv")
    with (out_dir / f"{segment}_metrics.json").open("w", encoding="utf-8") as fout:
        json.dump(summary, fout, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
