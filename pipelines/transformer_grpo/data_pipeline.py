from __future__ import annotations

import gc
import hashlib
import json
# multiprocessing import removed - using synchronous cache writes
import os
import re
import shutil
from collections import defaultdict, deque
# ProcessPoolExecutor removed for memory optimization
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from qlib.data.dataset.handler import DataHandlerLP
from qlib.utils import init_instance_by_config

IndexLike = Union[str, pd.Timestamp]

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


# _PendingCacheWrite class removed - using synchronous cache writes now


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
        """Return numpy arrays for features/rewards, loading from disk if needed.

        Uses memory-mapped loading for better memory efficiency.
        """
        if self.features is not None and self.rewards is not None:
            return self.features, self.rewards
        if self.cache_path is None:
            raise ValueError("No in-memory arrays or cache path available for DailyBatch.")
        # 使用 mmap_mode='r' 延迟加载，减少峰值内存
        with np.load(self.cache_path, allow_pickle=False, mmap_mode='r') as data:
            # 复制为可写数组（训练需要）
            features = np.array(data["features"])
            rewards = np.array(data["rewards"])
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

        self.label_expressions = _extract_label_expressions(handler_config)
        self.label_future_lookahead = _infer_label_future_lookahead(self.label_expressions)
        self.label_safe_shift = max(self.label_future_lookahead + 1, 1)
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
        self.cache_compress = bool(cache_cfg.get("compress", True))
        # cache_workers removed - using synchronous writes now
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
                print(f"[Cache] Loaded {len(cached_batches)} batches from cache for '{segment_name}'")
                return DailyBatchDataset(cached_batches)
        if self.cache_enabled and self.cache_force_refresh:
            self._clear_cache_segment(segment_name)

        # 分块读取处理，降低峰值内存
        all_batches: List[DailyBatch] = []
        all_calendar: List[pd.Timestamp] = []

        for chunk_frame, chunk_start, chunk_end in self._fetch_chunked(start, end, data_key):
            print(f"[Chunk] Processing {chunk_start.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}...")
            batches, calendar = self._frame_to_batches_chunk(
                chunk_frame,
                feature_group=self.feature_group,
                label_group=self.label_group,
                label_name=self.label_name,
                segment_name=segment_name,
                valid_start=start,
                valid_end=end,
            )
            all_batches.extend(batches)
            all_calendar.extend(calendar)
            # 立即释放 chunk_frame
            del chunk_frame
            gc.collect()

        if self.cache_enabled and all_calendar:
            self._write_segment_manifest(segment_name, all_calendar)
        return DailyBatchDataset(all_batches)

    def _fetch_chunked(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        data_key: str,
    ):
        """按年份分块读取数据，降低峰值内存"""
        start_year = start.year
        end_year = end.year

        for year in range(start_year, end_year + 1):
            # 计算当前年份的有效范围
            chunk_start = max(start, pd.Timestamp(f"{year}-01-01"))
            chunk_end = min(end, pd.Timestamp(f"{year}-12-31"))

            # 需要额外读取 temporal_span 天的历史数据用于构建窗口
            lookback_days = self.temporal_span * 2  # 留余量
            fetch_start = chunk_start - pd.Timedelta(days=lookback_days)

            print(f"[Chunk] Fetching year {year}: {fetch_start.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}...")
            chunk_frame = self.handler.fetch(
                selector=slice(fetch_start, chunk_end),
                col_set=[self.feature_group, self.label_group],
                data_key=data_key,
            )

            yield chunk_frame, chunk_start, chunk_end

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
        """加载缓存的批次数据（优化版：顺序加载 meta 文件，避免多进程内存开销）。"""
        manifest = self._load_segment_manifest(segment_name)
        if manifest is None:
            return None
        calendar = manifest.get("calendar", [])
        if not calendar:
            return None
        seg_dir = self._cache_dir_for(segment_name, ensure=False)
        if seg_dir is None or not seg_dir.exists():
            return None

        batches: List[DailyBatch] = []
        for date_str in calendar:
            date = pd.Timestamp(date_str)
            meta_path = seg_dir / f"{date.strftime('%Y%m%d')}.meta.json"
            if not meta_path.exists():
                return None

            # 直接读取 meta 文件（小文件，无需多进程）
            try:
                with meta_path.open("r", encoding="utf-8") as fin:
                    meta = json.load(fin)
            except Exception:
                return None

            cache_name = meta.get("cache_file") or f"{date.strftime('%Y%m%d')}.npz"
            cache_path = seg_dir / cache_name
            if not cache_path.exists():
                return None

            instruments = np.array(meta.get("instruments", []), dtype=object)
            shape_arr = tuple(int(x) for x in meta.get("shape", []))

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

    # _finalize_cache_writes removed - using synchronous cache writes now

    def _frame_to_batches_chunk(
        self,
        frame: pd.DataFrame,
        feature_group: str,
        label_group: str,
        label_name: Optional[str],
        segment_name: str,
        valid_start: pd.Timestamp,
        valid_end: pd.Timestamp,
    ) -> Tuple[List[DailyBatch], List[pd.Timestamp]]:
        """
        分块流式处理：
        - 预处理为高效的 numpy 索引结构
        - 逐日期处理，每个日期只在内存中保留当天数据
        - 处理完立即写入缓存并释放
        - 仅处理 valid_start 到 valid_end 范围内的日期
        """
        if frame.empty:
            return [], []

        # 1. 准备特征和标签（移除不必要的.copy()以减少内存）
        feature_view = _select_group(frame, feature_group)
        feature_view = _flatten_columns(feature_view)
        label_view = _select_group(frame, label_group)

        if isinstance(label_view, pd.DataFrame):
            if label_name and label_name in label_view.columns:
                label_series = label_view[label_name]
            else:
                label_series = label_view.iloc[:, 0]
        else:
            label_series = label_view
        # 不再需要.copy()，后续只读取值

        T = self.temporal_span
        feature_dim = feature_view.shape[1]

        # 2. 按股票分组，构建高效索引（关键：避免后续的 pandas 操作）
        print("[Cache] Building instrument index...")
        datetime_level = frame.index.get_level_values("datetime")
        instrument_level = frame.index.get_level_values("instrument")
        unique_dates = sorted(datetime_level.unique())
        unique_instruments = list(set(instrument_level))

        # 每只股票：{inst: (features_array, labels_array, sorted_dates_list)}
        inst_data: Dict[str, Tuple[np.ndarray, np.ndarray, List]] = {}

        for inst in unique_instruments:
            mask = instrument_level == inst
            inst_df = feature_view.loc[mask]
            inst_labels = label_series.loc[mask]

            inst_dates = inst_df.index.get_level_values("datetime")
            sort_order = np.argsort(inst_dates)

            inst_data[inst] = (
                inst_df.values[sort_order].astype(self.feature_dtype),
                inst_labels.values[sort_order].astype(np.float32),
                [inst_dates[i] for i in sort_order],
            )

        # 释放原始 DataFrame
        del feature_view, label_series, label_view, frame
        gc.collect()

        # 3. 构建日期 -> 股票索引映射
        # date_inst_idx[date][inst] = idx_in_inst_array
        print("[Cache] Building date-instrument mapping...")
        date_inst_idx: Dict[Any, Dict[str, int]] = {d: {} for d in unique_dates}
        for inst, (_, _, dates_list) in inst_data.items():
            for idx, dt in enumerate(dates_list):
                if dt in date_inst_idx:
                    date_inst_idx[dt][inst] = idx

        # 4. 流式处理每个日期
        print(f"[Cache] Processing {len(unique_dates)} dates (streaming)...")
        batches: List[DailyBatch] = []
        calendar: List[pd.Timestamp] = []
        cache_dir_path: Optional[Path] = None

        if self.cache_enabled and self.cache_dir is not None:
            cache_dir_path = self._cache_dir_for(segment_name)

        try:
            for target_date in unique_dates:
                current_date = pd.Timestamp(target_date)

                # 只处理有效范围内的日期（分块读取会包含额外的历史数据）
                if current_date < valid_start or current_date > valid_end:
                    continue

                inst_idx_map = date_inst_idx[target_date]

                batch_insts = []
                batch_feats_list = []
                batch_rewards_list = []

                for inst, idx in inst_idx_map.items():
                    # 检查是否有足够历史数据
                    if idx < T - 1:
                        continue

                    features_arr, labels_arr, _ = inst_data[inst]

                    # 直接切片获取窗口 [T, F]
                    window = features_arr[idx - T + 1: idx + 1]
                    if window.shape[0] != T:
                        continue

                    batch_insts.append(inst)
                    batch_feats_list.append(window)
                    batch_rewards_list.append(labels_arr[idx])

                # 检查最小股票数
                if len(batch_insts) < self.min_instruments:
                    continue

                # 转换为 numpy
                batch_insts = np.array(batch_insts, dtype=object)
                batch_feats = np.stack(batch_feats_list, axis=0)  # [N, T, F]
                batch_rewards = np.array(batch_rewards_list, dtype=np.float32)

                # 立即释放临时列表
                del batch_feats_list, batch_rewards_list

                # Universe filter
                if self.instrument_universe is not None:
                    mask = np.isin(batch_insts, list(self.instrument_universe))
                    if not mask.any():
                        continue
                    batch_insts = batch_insts[mask]
                    batch_feats = batch_feats[mask]
                    batch_rewards = batch_rewards[mask]

                if len(batch_insts) < self.min_instruments:
                    continue

                # Reward Clipping/Scaling
                if self.reward_clip is not None:
                    batch_rewards = np.clip(batch_rewards, self.reward_clip[0], self.reward_clip[1])
                if self.reward_scale != 1.0:
                    batch_rewards = batch_rewards * self.reward_scale

                # Max instruments
                if self.max_instruments and len(batch_insts) > self.max_instruments:
                    batch_insts = batch_insts[:self.max_instruments]
                    batch_feats = batch_feats[:self.max_instruments]
                    batch_rewards = batch_rewards[:self.max_instruments]

                # Cash Token
                if self.include_cash_token:
                    cash_feat = np.full((1, T, feature_dim), self.cash_feature_value, dtype=self.feature_dtype)
                    batch_feats = np.concatenate([batch_feats, cash_feat], axis=0)
                    batch_rewards = np.append(batch_rewards, np.float32(self.cash_return))
                    batch_insts = np.append(batch_insts, self.cash_token_name)

                # NaN handling
                batch_feats = np.nan_to_num(batch_feats, nan=0.0, posinf=0.0, neginf=0.0)
                batch_rewards = np.nan_to_num(batch_rewards, nan=0.0, posinf=0.0, neginf=0.0)

                feature_shape = tuple(batch_feats.shape)

                # 写入缓存（同步写入，避免多进程内存开销）
                cache_path: Optional[Path] = None
                if cache_dir_path is not None:
                    cache_file = f"{current_date.strftime('%Y%m%d')}.npz"
                    # 同步写入缓存文件
                    _write_npz_file(
                        str(cache_dir_path),
                        cache_file,
                        self.cache_compress,
                        batch_feats,
                        batch_rewards,
                    )
                    # 立即写入元数据
                    self._write_batch_meta(
                        segment_name=segment_name,
                        date=current_date,
                        instruments=batch_insts,
                        feature_shape=feature_shape,
                        cache_file=cache_file,
                    )
                    cache_path = cache_dir_path / cache_file
                    # 释放内存，后续从缓存加载
                    batch_feats = None
                    batch_rewards = None

                batches.append(
                    DailyBatch(
                        date=current_date,
                        instruments=np.array(batch_insts, dtype=object) if batch_insts is not None else np.array([], dtype=object),
                        features=batch_feats,
                        rewards=batch_rewards,
                        feature_shape=feature_shape,
                        cache_path=cache_path,
                    )
                )
                calendar.append(current_date)

        finally:
            # 释放所有数据
            del inst_data, date_inst_idx
            gc.collect()

        print(f"[Cache] Built {len(batches)} batches for '{segment_name}'")
        return batches, calendar

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
    """扁平化列名（不复制数据，直接返回重命名后的视图）"""
    if isinstance(frame.columns, pd.MultiIndex):
        new_cols = ["__".join(str(part) for part in col if part is not None) for col in frame.columns.values]
    else:
        new_cols = [str(col) for col in frame.columns]
    # 使用 set_axis 避免复制数据
    return frame.set_axis(new_cols, axis=1, copy=False)
