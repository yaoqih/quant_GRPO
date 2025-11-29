from __future__ import annotations

import gc
import re
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from qlib.data.dataset.handler import DataHandlerLP
from qlib.utils import init_instance_by_config

IndexLike = Union[str, pd.Timestamp]


@dataclass
class CrossSection:
    """Single trading day's cross-sectional view."""

    date: pd.Timestamp
    instruments: np.ndarray
    features: np.ndarray
    rewards: np.ndarray
    index_map: Dict[str, int]
    feature_dim: int


@dataclass
class DailyBatch:
    """Cross-sectional snapshot lazily materialized from rolling windows."""

    date: pd.Timestamp
    instruments: np.ndarray
    features: Optional[np.ndarray]
    rewards: Optional[np.ndarray]
    feature_shape: Tuple[int, ...] = field(default_factory=tuple)
    window_dates: Sequence[pd.Timestamp] = field(default_factory=list)
    index_slices: Optional[Sequence[np.ndarray]] = None
    data_instruments: Optional[np.ndarray] = None
    loader: Optional["RollingWindowLoader"] = None
    temporal_span: int = 1
    include_cash_token: bool = False
    cash_return: float = 0.0
    cash_feature_value: float = 0.0
    cash_token_name: str = "CASH"
    feature_dtype: np.dtype = np.float32

    def materialize(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.features is not None and self.rewards is not None:
            return self.features, self.rewards
        if self.loader is None:
            raise ValueError("DailyBatch is missing loader for dynamic materialization.")
        return self.loader.build_window(self)


class DailyBatchDataset(Dataset):
    """Thin wrapper so PyTorch can iterate over daily batches."""

    def __init__(self, batches: Sequence[DailyBatch]):
        super().__init__()
        self._batches: List[DailyBatch] = sorted(batches, key=lambda b: b.date)
        first_shape = _get_feature_shape(self._batches[0]) if self._batches else ()
        if len(first_shape) == 3:
            self.feature_dim = first_shape[-1]
            self.temporal_span = first_shape[1]
        elif len(first_shape) == 2:
            self.feature_dim = first_shape[-1]
            self.temporal_span = 1
        else:
            self.feature_dim = 0
            self.temporal_span = 1
        self.max_instruments: int = max((_num_tokens(b) for b in self._batches), default=0)

    def __len__(self) -> int:
        return len(self._batches)

    def __getitem__(self, idx: int) -> DailyBatch:
        return self._batches[idx]

    @property
    def calendar(self) -> List[pd.Timestamp]:
        return [b.date for b in self._batches]


def _get_feature_shape(batch: Optional[DailyBatch]) -> Tuple[int, ...]:
    if batch is None:
        return ()
    if batch.feature_shape:
        return batch.feature_shape
    if batch.features is not None:
        return batch.features.shape
    return ()


def _num_tokens(batch: DailyBatch) -> int:
    shape = _get_feature_shape(batch)
    if shape:
        return int(shape[0])
    return len(batch.instruments)


class RollingWindowLoader:
    """Build rolling windows directly from in-memory cross sections."""

    def __init__(self, cross_sections: "OrderedDict[pd.Timestamp, CrossSection]", feature_dtype: np.dtype):
        self.cross_sections = cross_sections
        self.feature_dtype = feature_dtype

    def build_window(self, batch: DailyBatch) -> Tuple[np.ndarray, np.ndarray]:
        if not batch.window_dates:
            raise ValueError("DailyBatch missing window_dates information.")

        base_instruments = batch.data_instruments if batch.data_instruments is not None else batch.instruments
        base_instruments = np.array(base_instruments, dtype=object)

        feature_slices: List[np.ndarray] = []
        for idx, date in enumerate(batch.window_dates):
            entry = self.cross_sections.get(pd.Timestamp(date))
            if entry is None:
                raise KeyError(f"Missing cross-section for {date}.")
            if batch.index_slices is not None and idx < len(batch.index_slices):
                take_idx = np.asarray(batch.index_slices[idx], dtype=np.int64)
            else:
                take_idx = self._indices_for(entry, base_instruments)
            feature_slices.append(np.asarray(entry.features[take_idx], dtype=self.feature_dtype, copy=False))

        stacked = np.stack(feature_slices, axis=1)  # [num_inst, T, feature_dim]
        stacked = np.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0)

        final_entry = self.cross_sections[pd.Timestamp(batch.window_dates[-1])]
        if batch.index_slices is not None and len(batch.index_slices) >= len(batch.window_dates):
            final_idx = np.asarray(batch.index_slices[-1], dtype=np.int64)
        else:
            final_idx = self._indices_for(final_entry, base_instruments)
        rewards = np.asarray(final_entry.rewards[final_idx], dtype=np.float32, copy=False)
        rewards = np.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)

        if batch.include_cash_token:
            cash_feat = np.full(
                (1, len(batch.window_dates), stacked.shape[-1]),
                batch.cash_feature_value,
                dtype=self.feature_dtype,
            )
            stacked = np.concatenate([stacked, cash_feat], axis=0)
            rewards = np.concatenate([rewards, np.array([batch.cash_return], dtype=np.float32)])

        return stacked, rewards

    @staticmethod
    def _indices_for(entry: CrossSection, instruments: np.ndarray) -> np.ndarray:
        idx = []
        for inst in instruments:
            inst_key = str(inst)
            if inst_key not in entry.index_map:
                raise KeyError(f"Instrument {inst_key} missing in cross section {entry.date}")
            idx.append(entry.index_map[inst_key])
        return np.array(idx, dtype=np.int64)


class DailyBatchFactory:
    """Build DailyBatchDataset directly from qlib handlers without disk cache."""

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
        feature_dtype: str = "float32",
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
        if instrument_universe is not None:
            inst_list = [str(inst) for inst in instrument_universe]
            self.instrument_universe = set(inst_list)
            self._instrument_universe_array = np.array(inst_list, dtype=object)
        else:
            self.instrument_universe = None
            self._instrument_universe_array = None
        self.augment_cfg = augment or {}
        self.temporal_span = max(int(self.augment_cfg.get("temporal_span", 1)), 1)

        self.label_expressions = _extract_label_expressions(handler_config)
        self.label_future_lookahead = _infer_label_future_lookahead(self.label_expressions)
        self.label_safe_shift = max(self.label_future_lookahead + 1, 1)
        self.include_cash_token = bool(self.augment_cfg.get("include_cash_token", False))
        self.cash_return = float(self.augment_cfg.get("cash_return", 0.0))
        self.cash_feature_value = float(self.augment_cfg.get("cash_feature_value", 0.0))
        self.cash_token_name = self.augment_cfg.get("cash_token", "CASH")
        try:
            self.feature_dtype = np.dtype(feature_dtype)
        except TypeError as exc:  # noqa: BLE001
            raise ValueError(f"Unsupported feature dtype '{feature_dtype}'.") from exc

    def build_segment(
        self,
        segment: Union[str, Sequence[IndexLike]],
        data_key: str = DataHandlerLP.DK_L,
    ) -> DailyBatchDataset:
        start, end = self._segment_to_pair(segment)
        cross_sections = self._collect_cross_sections(start, end, data_key)
        if not cross_sections:
            return DailyBatchDataset([])

        loader = RollingWindowLoader(cross_sections, self.feature_dtype)
        effective_end, has_targets = self._effective_segment_end(cross_sections, start, end)
        if not has_targets:
            return DailyBatchDataset([])
        batches = self._build_batches_from_cross_sections(cross_sections, loader, start, effective_end)
        return DailyBatchDataset(batches)

    def _collect_cross_sections(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        data_key: str,
    ) -> "OrderedDict[pd.Timestamp, CrossSection]":
        storage: "OrderedDict[pd.Timestamp, CrossSection]" = OrderedDict()
        for frame, _, _ in self._fetch_chunked(start, end, data_key):
            if frame.empty:
                continue
            feature_view = _select_group(frame, self.feature_group)
            feature_view = _flatten_columns(feature_view)
            label_view = _select_group(frame, self.label_group)

            if isinstance(label_view, pd.DataFrame):
                if self.label_name and self.label_name in label_view.columns:
                    label_series = label_view[self.label_name]
                else:
                    label_series = label_view.iloc[:, 0]
            else:
                label_series = label_view

            unique_dates = feature_view.index.get_level_values("datetime").unique()
            unique_dates = sorted(unique_dates)

            for target_date in unique_dates:
                current_date = pd.Timestamp(target_date)
                if start is not None and current_date < start - pd.Timedelta(days=self.temporal_span):
                    continue
                if end is not None and current_date > end:
                    continue
                if current_date in storage:
                    # Skip duplicate days introduced by chunk lookback windows.
                    continue

                try:
                    day_feat_df = feature_view.xs(current_date, level="datetime")
                    day_reward_series = label_series.xs(current_date, level="datetime")
                except KeyError:
                    continue

                day_feats = day_feat_df.to_numpy(dtype=self.feature_dtype, copy=False)
                day_rewards = day_reward_series.to_numpy(dtype=np.float32, copy=False)
                day_instruments = day_feat_df.index.get_level_values("instrument").to_numpy().astype(str)

                if day_feats.size == 0:
                    continue

                sort_idx = np.argsort(day_instruments)
                day_instruments = day_instruments[sort_idx]
                day_feats = day_feats[sort_idx]
                day_rewards = day_rewards[sort_idx]

                if self.instrument_universe is not None:
                    mask_universe = np.isin(day_instruments, self._instrument_universe_array)
                    if not mask_universe.any():
                        continue
                    day_instruments = day_instruments[mask_universe]
                    day_feats = day_feats[mask_universe]
                    day_rewards = day_rewards[mask_universe]

                if day_instruments.size == 0:
                    continue

                if self.reward_clip is not None:
                    day_rewards = np.clip(day_rewards, self.reward_clip[0], self.reward_clip[1])
                if self.reward_scale != 1.0:
                    day_rewards = day_rewards * self.reward_scale

                day_feats = np.nan_to_num(day_feats, nan=0.0, posinf=0.0, neginf=0.0)
                day_rewards = np.nan_to_num(day_rewards, nan=0.0, posinf=0.0, neginf=0.0)

                instruments = day_instruments.astype(object, copy=True)
                index_map = {inst: idx for idx, inst in enumerate(instruments)}

                storage[current_date] = CrossSection(
                    date=current_date,
                    instruments=instruments,
                    features=day_feats,
                    rewards=day_rewards,
                    index_map=index_map,
                    feature_dim=day_feats.shape[1],
                )

            del frame, feature_view, label_view
            gc.collect()

        return storage

    def _build_batches_from_cross_sections(
        self,
        cross_sections: "OrderedDict[pd.Timestamp, CrossSection]",
        loader: RollingWindowLoader,
        start: Optional[pd.Timestamp],
        end: Optional[pd.Timestamp],
    ) -> List[DailyBatch]:
        batches: List[DailyBatch] = []
        window: deque[CrossSection] = deque()
        for date, entry in cross_sections.items():
            window.append(entry)
            if len(window) > self.temporal_span:
                window.popleft()
            if len(window) < self.temporal_span:
                continue

            target_date = window[-1].date
            if start is not None and target_date < start:
                continue
            if end is not None and target_date > end:
                break

            batch = self._build_batch_from_window(window, loader)
            if batch is not None:
                batches.append(batch)

        return batches

    def _build_batch_from_window(
        self,
        window: Sequence[CrossSection],
        loader: RollingWindowLoader,
    ) -> Optional[DailyBatch]:
        metas = list(window)
        if not metas:
            return None

        common: Optional[set] = None
        for entry in metas:
            inst_set = set(entry.instruments)
            if common is None:
                common = inst_set
            else:
                common &= inst_set
        if common is None or not common:
            return None

        if self.instrument_universe is not None:
            common &= set(self.instrument_universe)
        if len(common) < self.min_instruments:
            return None

        target_entry = metas[-1]
        ordered = [inst for inst in target_entry.instruments if inst in common]
        if len(ordered) < self.min_instruments:
            return None
        if self.max_instruments and len(ordered) > self.max_instruments:
            ordered = ordered[: self.max_instruments]

        data_instruments = np.array(ordered, dtype=object)
        final_instruments = data_instruments
        if self.include_cash_token:
            final_instruments = np.append(final_instruments, self.cash_token_name)

        index_slices: List[np.ndarray] = []
        for entry in metas:
            index_slices.append(
                np.fromiter(
                    (entry.index_map[str(inst)] for inst in data_instruments),
                    dtype=np.int64,
                    count=len(data_instruments),
                )
            )

        feature_shape = (len(final_instruments), self.temporal_span, target_entry.feature_dim)

        return DailyBatch(
            date=target_entry.date,
            instruments=np.array(final_instruments, dtype=object),
            features=None,
            rewards=None,
            feature_shape=feature_shape,
            window_dates=[entry.date for entry in metas],
            index_slices=index_slices,
            data_instruments=data_instruments,
            loader=loader,
            temporal_span=self.temporal_span,
            include_cash_token=self.include_cash_token,
            cash_return=self.cash_return,
            cash_feature_value=self.cash_feature_value,
            cash_token_name=self.cash_token_name,
            feature_dtype=self.feature_dtype,
        )

    def _effective_segment_end(
        self,
        cross_sections: "OrderedDict[pd.Timestamp, CrossSection]",
        start: Optional[pd.Timestamp],
        end: Optional[pd.Timestamp],
    ) -> Tuple[Optional[pd.Timestamp], bool]:
        if not cross_sections:
            return end, False

        candidate_dates = [
            date
            for date in cross_sections.keys()
            if (start is None or date >= start) and (end is None or date <= end)
        ]

        if not candidate_dates:
            return end, False

        lookahead = max(self.label_future_lookahead, 0)
        if lookahead == 0:
            return end, True

        if len(candidate_dates) <= lookahead:
            return end, False

        guard_idx = len(candidate_dates) - lookahead - 1
        guard_idx = max(guard_idx, 0)
        return candidate_dates[guard_idx], True

    def _segment_to_pair(self, segment: Union[str, Sequence[IndexLike]]) -> Tuple[pd.Timestamp, pd.Timestamp]:
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

    def _fetch_chunked(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        data_key: str,
    ):
        if start is None and end is None:
            raise ValueError("At least one of start/end must be specified.")
        start_year = (start or end).year
        end_year = (end or start).year
        if start_year > end_year:
            start_year, end_year = end_year, start_year
        for year in range(start_year, end_year + 1):
            chunk_start = max(start or pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-01-01"))
            chunk_end = min(end or pd.Timestamp(f"{year}-12-31"), pd.Timestamp(f"{year}-12-31"))
            lookback = max(self.temporal_span + self.label_safe_shift, 1)
            fetch_start = chunk_start - pd.Timedelta(days=lookback)
            frame = self.handler.fetch(
                selector=slice(fetch_start, chunk_end),
                col_set=[self.feature_group, self.label_group],
                data_key=data_key,
            )
            yield frame, chunk_start, chunk_end


def _select_group(frame: pd.DataFrame, group: str) -> Union[pd.DataFrame, pd.Series]:
    cols = frame.columns
    if isinstance(cols, pd.MultiIndex):
        if group not in cols.get_level_values(0):
            raise KeyError(f"Column group '{group}' not found in {cols.get_level_values(0).unique()}")
        return frame[group]

    prefix = f"{group}::"
    matched = [col for col in cols if str(col).startswith(prefix)]
    if matched:
        return frame[matched]
    raise KeyError(f"Unable to locate columns for group '{group}'.")


def _infer_label_name_from_config(handler_config: Dict, _label_group: str) -> Optional[str]:
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
    return None


def _extract_label_expressions(handler_config: Dict) -> List[str]:
    if not isinstance(handler_config, dict):
        return []
    kwargs = handler_config.get("kwargs", {})
    label_spec = kwargs.get("label")
    if isinstance(label_spec, (list, tuple)):
        if label_spec and isinstance(label_spec[0], str):
            return list(label_spec)
        if label_spec and isinstance(label_spec[0], (list, tuple)):
            return [str(expr) for expr in label_spec[0] if isinstance(expr, str)]
    return []


def _infer_label_future_lookahead(label_expressions: Sequence[str]) -> int:
    pattern = re.compile(r"Ref\([^,]+,\s*(-?\d+)")
    max_lookahead = 0
    for expr in label_expressions:
        for match in pattern.findall(expr):
            try:
                offset = int(match)
            except ValueError:
                continue
            if offset < 0:
                max_lookahead = max(max_lookahead, abs(offset))
    return max_lookahead


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        new_cols = ["__".join(str(part) for part in col if part is not None) for col in frame.columns.values]
    else:
        new_cols = [str(col) for col in frame.columns]
    return frame.set_axis(new_cols, axis=1, copy=False)
