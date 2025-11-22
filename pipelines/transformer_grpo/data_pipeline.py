from __future__ import annotations

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
        self.feature_dim: int = self._batches[0].features.shape[1] if self._batches else 0
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
        self.instrument_universe = (
            set(instrument_universe) if instrument_universe is not None else None
        )

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
        label_view = _select_group(frame, label_group)

        if isinstance(label_view, pd.DataFrame):
            if label_name and label_name in label_view.columns:
                label_series = label_view[label_name]
            else:
                label_series = label_view.iloc[:, 0]
        else:
            label_series = label_view

        grouped = feature_view.groupby(level="datetime", sort=True)
        batches: List[DailyBatch] = []

        for date, feature_slice in grouped:
            label_slice = label_series.xs(date, level="datetime")
            joined = feature_slice.join(label_slice.rename("reward"), how="inner")
            if self.instrument_universe is not None:
                joined = joined.loc[joined.index.intersection(self.instrument_universe)]
            joined = joined.dropna()
            if len(joined) < self.min_instruments:
                continue
            if self.max_instruments and len(joined) > self.max_instruments:
                joined = joined.iloc[: self.max_instruments]
            features = joined.drop(columns=["reward"]).to_numpy(dtype=np.float32)
            rewards = joined["reward"].to_numpy(dtype=np.float32)
            if self.reward_clip is not None:
                low, high = self.reward_clip
                rewards = np.clip(rewards, low, high)
            if self.reward_scale != 1.0:
                rewards = rewards * self.reward_scale
            instruments = joined.index.to_numpy()
            batches.append(
                DailyBatch(
                    date=pd.Timestamp(date),
                    instruments=instruments,
                    features=features,
                    rewards=rewards,
                )
            )
        return batches


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
