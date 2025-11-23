from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from qlib.data.dataset.handler import DataHandlerLP
from qlib.utils import init_instance_by_config

IndexLike = Union[str, pd.Timestamp]


@dataclass
class DailyBatch:
    """Cross-sectional snapshot used by the GRPO trainer."""

    date: pd.Timestamp
    instruments: np.ndarray
    features: np.ndarray
    rewards: np.ndarray


class DailyBatchDataset(Dataset):
    """Thin wrapper so PyTorch can iterate over a list of DailyBatch samples."""

    def __init__(self, batches: Sequence[DailyBatch]):
        super().__init__()
        self._batches: List[DailyBatch] = sorted(batches, key=lambda b: b.date)
        first = self._batches[0].features if self._batches else None
        if first is not None and first.ndim == 3:
            self.feature_dim = first.shape[-1]
            self.temporal_span = first.shape[1]
        else:
            self.feature_dim = first.shape[1] if first is not None else 0
            self.temporal_span = 1
        self.max_instruments: int = max((b.features.shape[0] for b in self._batches), default=0)

    def __len__(self) -> int:
        return len(self._batches)

    def __getitem__(self, idx: int) -> DailyBatch:
        return self._batches[idx]

    @property
    def calendar(self) -> List[pd.Timestamp]:
        return [b.date for b in self._batches]


class DailyBatchFactory:
    """Builds DailyBatchDataset instances directly from qlib handlers."""

    def __init__(
        self,
        handler_config: Dict,
        segments: Optional[Dict[str, Sequence[IndexLike]]] = None,
        feature_group: str = "feature",
        label_group: str = "label",
        label_name: Optional[str] = None,
        min_instruments: int = 10,
        max_instruments: Optional[int] = None,
        reward_clip: Optional[Tuple[float, float]] = None,
        reward_scale: float = 1.0,
        instrument_universe: Optional[Iterable[str]] = None,
        augment: Optional[Dict] = None,
    ):
        self.handler: DataHandlerLP = init_instance_by_config(handler_config, accept_types=DataHandlerLP)
        self.segments = segments or {}
        self.feature_group = feature_group
        self.label_group = label_group
        self.label_name = label_name or _infer_label_name_from_config(handler_config, label_group)
        self.min_instruments = max(min_instruments, 1)
        self.max_instruments = max_instruments
        self.reward_clip = reward_clip
        self.reward_scale = reward_scale
        self.instrument_universe = set(instrument_universe) if instrument_universe is not None else None
        self.augment_cfg = augment or {}
        self.temporal_span = max(int(self.augment_cfg.get("temporal_span", 1)), 1)
        self.roll_vol_windows = sorted(set(int(x) for x in self.augment_cfg.get("rolling_vol_windows", [])))
        self.future_return_horizons = sorted(
            set(int(x) for x in self.augment_cfg.get("future_return_horizons", []))
        )
        self.instrument_emb_dim = int(self.augment_cfg.get("instrument_embedding_dim", 0))
        self.include_cash_token = bool(self.augment_cfg.get("include_cash_token", False))
        self.cash_return = float(self.augment_cfg.get("cash_return", 0.0))
        self.cash_feature_value = float(self.augment_cfg.get("cash_feature_value", 0.0))
        self.cash_token_name = self.augment_cfg.get("cash_token", "CASH")
        self._instrument_embed_cache: Dict[str, np.ndarray] = {}

    def build_segment(
        self,
        segment: Union[str, Sequence[IndexLike]],
        data_key: str = DataHandlerLP.DK_L,
    ) -> DailyBatchDataset:
        start, end = self._segment_to_pair(segment)
        raw_frame = self.handler.fetch(
            selector=slice(start, end),
            col_set=[self.feature_group, self.label_group],
            data_key=data_key,
        )
        batches = list(
            self._frame_to_batches(
                raw_frame,
                feature_group=self.feature_group,
                label_group=self.label_group,
                label_name=self.label_name,
            )
        )
        return DailyBatchDataset(batches)

    def _segment_to_pair(self, segment: Union[str, Sequence[IndexLike]]) -> Tuple[IndexLike, IndexLike]:
        if isinstance(segment, str):
            if segment not in self.segments:
                raise KeyError(f"Unknown segment '{segment}'. Available: {list(self.segments)}")
            seg = self.segments[segment]
        else:
            seg = segment
        if len(seg) != 2:
            raise ValueError(f"Segment must contain [start, end], got {seg}")
        start, end = seg
        start = pd.Timestamp(start) if start is not None else None
        end = pd.Timestamp(end) if end is not None else None
        return start, end

    def _frame_to_batches(
        self,
        frame: pd.DataFrame,
        feature_group: str,
        label_group: str,
        label_name: Optional[str],
    ) -> Iterable[DailyBatch]:
        if frame.empty:
            return []

        feature_view = _select_group(frame, feature_group).copy()
        feature_view = _flatten_columns(feature_view)
        label_view = _select_group(frame, label_group)

        if isinstance(label_view, pd.DataFrame):
            if label_name and label_name in label_view.columns:
                label_series = label_view[label_name]
            else:
                label_series = label_view.iloc[:, 0]
        else:
            label_series = label_view
        label_series = label_series.copy()

        feature_view = self._augment_feature_view(feature_view, label_series)
        grouped = feature_view.groupby(level="datetime", sort=True)
        batches: List[DailyBatch] = []
        feature_dim = len(feature_view.columns)
        buffer_map: Dict[str, deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=self.temporal_span))

        for date, feature_slice in grouped:
            label_slice = label_series.xs(date, level="datetime")
            slice_view = feature_slice.droplevel("datetime") if isinstance(feature_slice.index, pd.MultiIndex) else feature_slice
            inst_names: List[str] = []
            inst_features: List[np.ndarray] = []
            inst_rewards: List[float] = []

            for inst, row in slice_view.iterrows():
                inst_key = str(inst)
                buf = buffer_map[inst_key]
                buf.append(row.to_numpy(dtype=np.float32))
                if len(buf) < self.temporal_span:
                    continue
                reward_val = label_slice.get(inst, np.nan)
                if pd.isna(reward_val):
                    continue
                inst_names.append(inst_key)
                inst_features.append(np.stack(buf, axis=0))
                inst_rewards.append(float(reward_val))

            if not inst_names:
                continue

            instruments = np.array(inst_names, dtype=object)
            features = np.stack(inst_features).astype(np.float32)
            rewards = np.array(inst_rewards, dtype=np.float32)

            if self.instrument_universe is not None:
                mask = np.isin(instruments, list(self.instrument_universe))
                if not mask.any():
                    continue
                instruments = instruments[mask]
                features = features[mask]
                rewards = rewards[mask]

            if len(instruments) < self.min_instruments:
                continue

            if self.max_instruments and len(instruments) > self.max_instruments:
                features = features[: self.max_instruments]
                rewards = rewards[: self.max_instruments]
                instruments = instruments[: self.max_instruments]

            if self.reward_clip is not None:
                low, high = self.reward_clip
                rewards = np.clip(rewards, low, high)
            if self.reward_scale != 1.0:
                rewards = rewards * self.reward_scale

            if self.include_cash_token:
                cash_feat = np.full(
                    (self.temporal_span, feature_dim),
                    self.cash_feature_value,
                    dtype=np.float32,
                )
                features = np.vstack([features, cash_feat[np.newaxis, ...]])
                rewards = np.concatenate([rewards, np.array([self.cash_return], dtype=np.float32)])
                instruments = np.concatenate([instruments, np.array([self.cash_token_name], dtype=object)])

            if self.instrument_emb_dim > 0:
                embeddings = np.stack(
                    [self._get_instrument_embedding(str(inst)) for inst in instruments]
                ).astype(np.float32)
                embeddings = np.broadcast_to(
                    embeddings[:, np.newaxis, :],
                    (len(instruments), self.temporal_span, self.instrument_emb_dim),
                )
                features = np.concatenate([features, embeddings], axis=-1)

            batches.append(
                DailyBatch(
                    date=pd.Timestamp(date),
                    instruments=instruments,
                    features=features,
                    rewards=rewards,
                )
            )
        return batches

    def _augment_feature_view(self, feature_view: pd.DataFrame, label_series: pd.Series) -> pd.DataFrame:
        frames = [feature_view]
        if self.roll_vol_windows:
            grouped_label = label_series.groupby(level="instrument")
            for window in self.roll_vol_windows:
                rolled = grouped_label.rolling(window, min_periods=1).std().droplevel(0)
                shifted = rolled.groupby(level="instrument").shift(1)
                frames.append(shifted.reindex(feature_view.index).to_frame(name=f"label_std_{window}"))
        if self.future_return_horizons:
            grouped_label = label_series.groupby(level="instrument")
            for horizon in self.future_return_horizons:
                # Rolling product over past horizon days (include current), then shift forward so each row only sees past info.
                prod = (1.0 + grouped_label).rolling(horizon, min_periods=horizon).apply(np.prod, raw=True).droplevel(0) - 1.0
                past_future = prod.groupby(level="instrument").shift(1)
                frames.append(past_future.reindex(feature_view.index).to_frame(name=f"past_return_{horizon}"))
        augmented = pd.concat(frames, axis=1)
        augmented = augmented.fillna(0.0)
        return augmented

    def _get_instrument_embedding(self, instrument: str) -> np.ndarray:
        if self.instrument_emb_dim <= 0:
            return np.zeros((1,), dtype=np.float32)
        if instrument not in self._instrument_embed_cache:
            seed = (hash(instrument) & 0xFFFF_FFFF) or 7
            rng = np.random.default_rng(seed)
            self._instrument_embed_cache[instrument] = rng.standard_normal(self.instrument_emb_dim).astype(np.float32)
        return self._instrument_embed_cache[instrument]


def _select_group(frame: pd.DataFrame, group: str) -> Union[pd.DataFrame, pd.Series]:
    cols = frame.columns
    if isinstance(cols, pd.MultiIndex):
        if group not in cols.get_level_values(0):
            raise KeyError(f"Column group '{group}' not found in {cols.get_level_values(0).unique()}")
        return frame[group]

    # Fall back to flattened naming convention such as "feature::$close"
    prefix = f"{group}::"
    matched = [col for col in cols if str(col).startswith(prefix)]
    if matched:
        return frame[matched]
    raise KeyError(f"Unable to locate columns for group '{group}'.")


def _infer_label_name_from_config(handler_config: Dict, label_group: str) -> Optional[str]:
    """Best-effort extraction of the label column name from handler config."""
    if not isinstance(handler_config, dict):
        return None
    kwargs = handler_config.get("kwargs", {})
    label_spec = kwargs.get("label")
    if isinstance(label_spec, (list, tuple)) and label_spec:
        first = label_spec[0]
        if isinstance(first, str):
            return first
        if isinstance(first, (list, tuple)) and first:
            candidate = first[0]
            if isinstance(candidate, str):
                return candidate
    # Some handlers return alias list via `get_label_config`, already handled in data.
    return None


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    flat = frame.copy()
    if isinstance(flat.columns, pd.MultiIndex):
        flat.columns = ["__".join(str(part) for part in col if part is not None) for col in flat.columns.values]
    else:
        flat.columns = [str(col) for col in flat.columns]
    return flat
