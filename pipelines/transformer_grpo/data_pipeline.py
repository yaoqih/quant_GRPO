from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import re
import shutil
from collections import defaultdict, deque
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from qlib.data.dataset.handler import DataHandlerLP
from qlib.utils import init_instance_by_config

IndexLike = Union[str, pd.Timestamp]

_CACHE_WORKER_OFFSET = 2


def _cpu_worker_count(offset: int = _CACHE_WORKER_OFFSET) -> int:
    try:
        total = multiprocessing.cpu_count() or 1
    except NotImplementedError:  # pragma: no cover - platform specific
        total = 1
    return max(1, total - max(offset, 0))


def _load_cached_batch_worker(date_str: str, meta_path: str, cache_dir: str) -> Optional[Dict[str, Any]]:
    meta_file = Path(meta_path)
    if not meta_file.exists():
        return None
    try:
        with meta_file.open("r", encoding="utf-8") as fin:
            meta = json.load(fin)
    except Exception:  # noqa: BLE001 - surface as cache miss
        return None
    cache_name = meta.get("cache_file") or f"{date_str}.npz"
    cache_path = Path(cache_dir) / cache_name
    if not cache_path.exists():
        return None
    instruments = meta.get("instruments", [])
    shape = meta.get("shape", [])
    return {
        "date": date_str,
        "cache_path": str(cache_path),
        "instruments": instruments,
        "shape": shape,
    }


def _write_npz_file(directory: str, filename: str, compress: bool, features: np.ndarray, rewards: np.ndarray) -> str:
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / filename
    tmp_path = path.with_suffix(".tmp")
    save_fn = np.savez_compressed if compress else np.savez
    with tmp_path.open("wb") as fout:
        save_fn(fout, features=features, rewards=rewards)
    os.replace(tmp_path, path)
    return str(path)


@dataclass
class _PendingCacheWrite:
    future: Future
    segment_name: str
    date: pd.Timestamp
    instruments: List[str]
    feature_shape: Tuple[int, ...]
    cache_file: str


@dataclass
class DailyBatch:
    """Cross-sectional snapshot used by the GRPO trainer."""

    date: pd.Timestamp
    instruments: np.ndarray
    features: Optional[np.ndarray]
    rewards: Optional[np.ndarray]
    feature_shape: Tuple[int, ...] = field(default_factory=tuple)
    cache_path: Optional[Path] = None

    def materialize(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return numpy arrays for features/rewards, loading from disk if needed."""
        if self.features is not None and self.rewards is not None:
            return self.features, self.rewards
        if self.cache_path is None:
            raise ValueError("No in-memory arrays or cache path available for DailyBatch.")
        with np.load(self.cache_path, allow_pickle=False) as data:
            features = data["features"]
            rewards = data["rewards"]
        return features, rewards


class DailyBatchDataset(Dataset):
    """Thin wrapper so PyTorch can iterate over a list of DailyBatch samples."""

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
        cache_config: Optional[Dict] = None,
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
        self.instrument_universe = set(instrument_universe) if instrument_universe is not None else None
        self.augment_cfg = augment or {}
        self.temporal_span = max(int(self.augment_cfg.get("temporal_span", 1)), 1)
        self.roll_vol_windows = sorted(set(int(x) for x in self.augment_cfg.get("rolling_vol_windows", [])))
        self.future_return_horizons = sorted(
            set(int(x) for x in self.augment_cfg.get("future_return_horizons", []))
        )
        self.label_expressions = _extract_label_expressions(handler_config)
        self.label_future_lookahead = _infer_label_future_lookahead(self.label_expressions)
        self.label_safe_shift = max(self.label_future_lookahead + 1, 1)
        self.instrument_emb_dim = int(self.augment_cfg.get("instrument_embedding_dim", 0))
        self.include_cash_token = bool(self.augment_cfg.get("include_cash_token", False))
        self.cash_return = float(self.augment_cfg.get("cash_return", 0.0))
        self.cash_feature_value = float(self.augment_cfg.get("cash_feature_value", 0.0))
        self.cash_token_name = self.augment_cfg.get("cash_token", "CASH")
        self._instrument_embed_cache: Dict[str, np.ndarray] = {}
        try:
            self.feature_dtype = np.dtype(feature_dtype)
        except TypeError as exc:  # noqa: BLE001
            raise ValueError(f"Unsupported feature dtype '{feature_dtype}'.") from exc
        cache_cfg = cache_config or {}
        cache_dtype = cache_cfg.get("feature_dtype")
        if cache_dtype is not None:
            try:
                self.feature_dtype = np.dtype(cache_dtype)
            except TypeError as exc:  # noqa: BLE001
                raise ValueError(f"Unsupported cache feature dtype '{cache_dtype}'.") from exc
        self.cache_enabled = bool(cache_cfg.get("enable", False))
        self.cache_reuse = bool(cache_cfg.get("reuse_existing", True))
        self.cache_force_refresh = bool(cache_cfg.get("force_refresh", False))
        cache_root = cache_cfg.get("root")
        self.cache_dir: Optional[Path] = None
        if self.cache_enabled:
            self.cache_dir = Path(cache_root or "runs/daily_batch_cache")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        worker_override = cache_cfg.get("num_workers")
        if worker_override is not None:
            self.cache_workers = max(int(worker_override), 1)
        else:
            self.cache_workers = _cpu_worker_count()
        self.cache_compress = bool(cache_cfg.get("compress", True))
        self._cache_segment_dirs: Dict[str, Path] = {}
        self.cache_signature = self._build_cache_signature(
            handler_config=handler_config,
            feature_group=feature_group,
            label_group=label_group,
            label_name=self.label_name,
            min_instruments=self.min_instruments,
            max_instruments=self.max_instruments,
            reward_clip=self.reward_clip,
            reward_scale=self.reward_scale,
            instrument_universe=self.instrument_universe,
            augment=self.augment_cfg,
            feature_dtype=str(self.feature_dtype),
            temporal_span=self.temporal_span,
        )

    def build_segment(
        self,
        segment: Union[str, Sequence[IndexLike]],
        data_key: str = DataHandlerLP.DK_L,
    ) -> DailyBatchDataset:
        start, end = self._segment_to_pair(segment)
        segment_name = self._segment_name(segment, start, end)
        if self.cache_enabled and self.cache_reuse and not self.cache_force_refresh:
            cached_batches = self._load_cached_batches(segment_name)
            if cached_batches is not None:
                return DailyBatchDataset(cached_batches)
        if self.cache_enabled and self.cache_force_refresh:
            self._clear_cache_segment(segment_name)
        raw_frame = self.handler.fetch(
            selector=slice(start, end),
            col_set=[self.feature_group, self.label_group],
            data_key=data_key,
        )
        batches, calendar = self._frame_to_batches(
            raw_frame,
            feature_group=self.feature_group,
            label_group=self.label_group,
            label_name=self.label_name,
            segment_name=segment_name,
        )
        if self.cache_enabled and calendar:
            self._write_segment_manifest(segment_name, calendar)
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

    def _segment_name(
        self,
        segment: Union[str, Sequence[IndexLike]],
        start: Optional[pd.Timestamp],
        end: Optional[pd.Timestamp],
    ) -> str:
        if isinstance(segment, str):
            return segment
        start_tag = start.strftime("%Y%m%d") if isinstance(start, pd.Timestamp) else "start"
        end_tag = end.strftime("%Y%m%d") if isinstance(end, pd.Timestamp) else "end"
        return f"{start_tag}_{end_tag}"

    def _cache_dir_for(self, segment_name: str, ensure: bool = True) -> Optional[Path]:
        if self.cache_dir is None:
            if ensure:
                raise RuntimeError("Cache directory is not configured.")
            return None
        if segment_name in self._cache_segment_dirs:
            return self._cache_segment_dirs[segment_name]
        seg_dir = self.cache_dir / segment_name
        if ensure:
            seg_dir.mkdir(parents=True, exist_ok=True)
        elif not seg_dir.exists():
            return None
        self._cache_segment_dirs[segment_name] = seg_dir
        return seg_dir

    def _segment_manifest_path(self, segment_name: str) -> Optional[Path]:
        seg_dir = self._cache_dir_for(segment_name, ensure=False)
        if seg_dir is None:
            return None
        return seg_dir / "manifest.json"

    def _write_segment_manifest(self, segment_name: str, calendar: Sequence[pd.Timestamp]) -> None:
        manifest_path = self._cache_dir_for(segment_name, ensure=True) / "manifest.json"
        tmp_path = manifest_path.with_suffix(".json.tmp")
        payload = {
            "signature": self.cache_signature,
            "calendar": [pd.Timestamp(ts).strftime("%Y-%m-%d") for ts in calendar],
            "feature_dtype": str(self.feature_dtype),
        }
        with tmp_path.open("w", encoding="utf-8") as fout:
            json.dump(payload, fout, sort_keys=True)
        os.replace(tmp_path, manifest_path)

    def _load_segment_manifest(self, segment_name: str) -> Optional[Dict]:
        manifest_path = self._segment_manifest_path(segment_name)
        if manifest_path is None or not manifest_path.exists():
            return None
        try:
            with manifest_path.open("r", encoding="utf-8") as fin:
                content = fin.read()
                if not content.strip():
                    # Empty file, treat as no cache
                    return None
                manifest = json.loads(content)
        except json.JSONDecodeError:
            # Corrupted cache file, treat as no cache
            return None
        if manifest.get("signature") != self.cache_signature:
            return None
        return manifest

    def _clear_cache_segment(self, segment_name: str) -> None:
        seg_dir = self._cache_dir_for(segment_name, ensure=False)
        if seg_dir is None or not seg_dir.exists():
            return
        shutil.rmtree(seg_dir)
        if segment_name in self._cache_segment_dirs:
            del self._cache_segment_dirs[segment_name]

    def _batch_meta_path(self, segment_name: str, date_str: str) -> Optional[Path]:
        seg_dir = self._cache_dir_for(segment_name, ensure=False)
        if seg_dir is None:
            return None
        return seg_dir / f"{date_str}.meta.json"

    def _write_batch_meta(
        self,
        segment_name: str,
        date: pd.Timestamp,
        instruments: np.ndarray,
        feature_shape: Tuple[int, ...],
        cache_file: str,
    ) -> None:
        meta_path = self._cache_dir_for(segment_name, ensure=True) / f"{date.strftime('%Y%m%d')}.meta.json"
        tmp_path = meta_path.with_suffix(".json.tmp")
        payload = {
            "date": date.strftime("%Y-%m-%d"),
            "cache_file": cache_file,
            "instruments": [str(inst) for inst in instruments],
            "shape": [int(dim) for dim in feature_shape],
        }
        with tmp_path.open("w", encoding="utf-8") as fout:
            json.dump(payload, fout, sort_keys=True)
        os.replace(tmp_path, meta_path)

    def _read_batch_meta(
        self,
        segment_name: str,
        date_str: str,
    ) -> Optional[Tuple[np.ndarray, Tuple[int, ...], Path]]:
        meta_path = self._batch_meta_path(segment_name, date_str)
        if meta_path is None or not meta_path.exists():
            return None
        try:
            with meta_path.open("r", encoding="utf-8") as fin:
                content = fin.read()
                if not content.strip():
                    return None
                meta = json.loads(content)
        except json.JSONDecodeError:
            return None
        instruments = np.array(meta.get("instruments", []), dtype=object)
        shape = tuple(int(x) for x in meta.get("shape", []))
        cache_name = meta.get("cache_file") or f"{date_str}.npz"
        seg_dir = self._cache_dir_for(segment_name, ensure=False)
        if seg_dir is None:
            return None
        cache_path = seg_dir / cache_name
        if not cache_path.exists():
            return None
        return instruments, shape, cache_path

    def _load_cached_batches(self, segment_name: str) -> Optional[List[DailyBatch]]:
        manifest = self._load_segment_manifest(segment_name)
        if manifest is None:
            return None
        calendar = manifest.get("calendar", [])
        if not calendar:
            return None
        seg_dir = self._cache_dir_for(segment_name, ensure=False)
        if seg_dir is None or not seg_dir.exists():
            return None
        tasks: List[Tuple[str, str]] = []
        for date_str in calendar:
            date = pd.Timestamp(date_str)
            meta_path = seg_dir / f"{date.strftime('%Y%m%d')}.meta.json"
            if not meta_path.exists():
                return None
            tasks.append((date_str, str(meta_path)))
        worker_count = max(1, min(self.cache_workers, len(tasks)))
        result_map: Dict[str, Dict[str, Any]] = {}
        cache_dir_str = str(seg_dir)
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            futures = {
                pool.submit(_load_cached_batch_worker, date_str, meta_path, cache_dir_str): date_str
                for date_str, meta_path in tasks
            }
            for future in as_completed(futures):
                date_key = futures[future]
                try:
                    payload = future.result()
                except Exception:  # noqa: BLE001
                    return None
                if payload is None:
                    return None
                result_map[date_key] = payload
        batches: List[DailyBatch] = []
        for date_str in calendar:
            payload = result_map.get(date_str)
            if payload is None:
                return None
            date = pd.Timestamp(payload.get("date", date_str))
            instruments = np.array(payload.get("instruments", []), dtype=object)
            shape_arr = tuple(int(x) for x in payload.get("shape", []))
            cache_path = Path(payload["cache_path"])
            batches.append(
                DailyBatch(
                    date=date,
                    instruments=instruments,
                    features=None,
                    rewards=None,
                    feature_shape=shape_arr,
                    cache_path=cache_path,
                )
            )
        return batches

    def _finalize_cache_writes(
        self,
        pending_writes: List[_PendingCacheWrite],
        executor: ProcessPoolExecutor,
    ) -> None:
        error: Optional[RuntimeError] = None
        for pending in pending_writes:
            try:
                pending.future.result()
            except Exception as exc:  # noqa: BLE001
                error = RuntimeError(
                    f"Failed to write cache for segment '{pending.segment_name}' on {pending.date.strftime('%Y-%m-%d')}"
                )
                error.__cause__ = exc
                break
            instruments = np.array(pending.instruments, dtype=object)
            self._write_batch_meta(
                segment_name=pending.segment_name,
                date=pending.date,
                instruments=instruments,
                feature_shape=pending.feature_shape,
                cache_file=pending.cache_file,
            )
        executor.shutdown(wait=True)
        if error is not None:
            raise error

    def _frame_to_batches(
        self,
        frame: pd.DataFrame,
        feature_group: str,
        label_group: str,
        label_name: Optional[str],
        segment_name: str,
    ) -> Tuple[List[DailyBatch], List[pd.Timestamp]]:
        if frame.empty:
            return [], []

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
        calendar: List[pd.Timestamp] = []
        feature_dim = len(feature_view.columns)
        buffer_map: Dict[str, deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=self.temporal_span))
        pending_cache_writes: List[_PendingCacheWrite] = []
        cache_executor: Optional[ProcessPoolExecutor] = None
        cache_dir_path: Optional[Path] = None
        if self.cache_enabled and self.cache_dir is not None:
            cache_dir_path = self._cache_dir_for(segment_name)

        try:
            for date, feature_slice in grouped:
                current_date = pd.Timestamp(date)
                label_slice = label_series.xs(date, level="datetime")
                slice_view = feature_slice.droplevel("datetime") if isinstance(feature_slice.index, pd.MultiIndex) else feature_slice

                # 向量化优化：一次性处理所有 instruments
                # 获取所有 instrument 的索引
                inst_index = slice_view.index
                n_inst = len(inst_index)

                if n_inst == 0:
                    continue

                # 将整个 slice 转为 numpy 数组（避免逐行 iterrows）
                slice_values = slice_view.to_numpy(dtype=np.float32)  # [n_inst, feature_dim]

                # 批量更新 buffer 并收集结果
                inst_names: List[str] = []
                inst_features: List[np.ndarray] = []
                inst_rewards: List[float] = []

                for idx, inst in enumerate(inst_index):
                    inst_key = str(inst)
                    buf = buffer_map[inst_key]
                    buf.append(slice_values[idx])  # 直接使用预提取的 numpy 数组
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
                features = np.stack(inst_features).astype(self.feature_dtype, copy=False)
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
                        dtype=self.feature_dtype,
                    )
                    features = np.vstack([features, cash_feat[np.newaxis, ...]])
                    rewards = np.concatenate([rewards, np.array([self.cash_return], dtype=np.float32)])
                    instruments = np.concatenate([instruments, np.array([self.cash_token_name], dtype=object)])

                if self.instrument_emb_dim > 0:
                    embeddings = np.stack(
                        [self._get_instrument_embedding(str(inst)) for inst in instruments]
                    ).astype(self.feature_dtype)
                    embeddings = np.broadcast_to(
                        embeddings[:, np.newaxis, :],
                        (len(instruments), self.temporal_span, self.instrument_emb_dim),
                    )
                    features = np.concatenate([features, embeddings], axis=-1)

                feature_shape = tuple(features.shape)
                cache_path: Optional[Path] = None
                if cache_dir_path is not None:
                    cache_file = f"{current_date.strftime('%Y%m%d')}.npz"
                    if cache_executor is None:
                        cache_executor = ProcessPoolExecutor(max_workers=self.cache_workers)
                    future = cache_executor.submit(
                        _write_npz_file,
                        str(cache_dir_path),
                        cache_file,
                        self.cache_compress,
                        features,
                        rewards,
                    )
                    pending_cache_writes.append(
                        _PendingCacheWrite(
                            future=future,
                            segment_name=segment_name,
                            date=current_date,
                            instruments=[str(inst) for inst in instruments],
                            feature_shape=feature_shape,
                            cache_file=cache_file,
                        )
                    )
                    cache_path = cache_dir_path / cache_file
                    features = None
                    rewards = None

                batches.append(
                    DailyBatch(
                        date=current_date,
                        instruments=instruments,
                        features=features,
                        rewards=rewards,
                        feature_shape=feature_shape,
                        cache_path=cache_path,
                    )
                )
                calendar.append(current_date)
        finally:
            if cache_executor is not None:
                self._finalize_cache_writes(pending_cache_writes, cache_executor)
        return batches, calendar

    def _augment_feature_view(self, feature_view: pd.DataFrame, label_series: pd.Series) -> pd.DataFrame:
        frames = [feature_view]
        safe_label_series = None
        if self.roll_vol_windows or self.future_return_horizons:
            # Shift label-derived data far enough into the past so no future price information leaks into features.
            safe_label_series = label_series.groupby(level="instrument").shift(self.label_safe_shift)
        if self.roll_vol_windows:
            grouped_label = safe_label_series.groupby(level="instrument")
            for window in self.roll_vol_windows:
                rolled = grouped_label.rolling(window, min_periods=1).std().droplevel(0)
                shifted = rolled.groupby(level="instrument").shift(1)
                frames.append(shifted.reindex(feature_view.index).to_frame(name=f"label_std_{window}"))
        if self.future_return_horizons:
            grouped_label = safe_label_series.groupby(level="instrument")
            for horizon in self.future_return_horizons:
                # Rolling product over past horizon days (include current), then shift forward so each row only sees past info.
                prod = (1.0 + grouped_label).rolling(horizon, min_periods=horizon).apply(np.prod, raw=True).droplevel(0) - 1.0
                past_future = prod.groupby(level="instrument").shift(1)
                frames.append(past_future.reindex(feature_view.index).to_frame(name=f"past_return_{horizon}"))
        augmented = pd.concat(frames, axis=1)
        augmented = augmented.fillna(0.0)
        return augmented

    def _build_cache_signature(
        self,
        handler_config: Dict,
        feature_group: str,
        label_group: str,
        label_name: Optional[str],
        min_instruments: int,
        max_instruments: Optional[int],
        reward_clip: Optional[Tuple[float, float]],
        reward_scale: float,
        instrument_universe: Optional[Iterable[str]],
        augment: Dict,
        feature_dtype: str,
        temporal_span: int,
    ) -> str:
        payload = {
            "handler": handler_config,
            "feature_group": feature_group,
            "label_group": label_group,
            "label_name": label_name,
            "min_instruments": min_instruments,
            "max_instruments": max_instruments,
            "reward_clip": list(reward_clip) if reward_clip is not None else None,
            "reward_scale": reward_scale,
            "instrument_universe": sorted(instrument_universe) if instrument_universe is not None else None,
            "augment": augment,
            "feature_dtype": feature_dtype,
            "temporal_span": temporal_span,
        }
        dump = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(dump.encode("utf-8")).hexdigest()

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


def _infer_label_name_from_config(handler_config: Dict, _label_group: str) -> Optional[str]:
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
    flat = frame.copy()
    if isinstance(flat.columns, pd.MultiIndex):
        flat.columns = ["__".join(str(part) for part in col if part is not None) for col in flat.columns.values]
    else:
        flat.columns = [str(col) for col in flat.columns]
    return flat
