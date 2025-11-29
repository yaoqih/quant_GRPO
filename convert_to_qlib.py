#!/usr/bin/env python3
"""Convert downloaded Parquet bars into Qlib binary dataset with resume support."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import logging
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
QLIB_SRC = ROOT_DIR / "qlib"
if QLIB_SRC.exists() and str(QLIB_SRC) not in sys.path:
    sys.path.insert(0, str(QLIB_SRC))

try:
    from qlib.scripts.dump_bin import DumpDataAll
except ModuleNotFoundError:  # pragma: no cover - fallback for source checkout
    scripts_dir = QLIB_SRC / "scripts"
    if scripts_dir.exists() and str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from dump_bin import DumpDataAll  # type: ignore

from download_stock import TradingCalendar, load_symbols_from_file, parse_symbols

load_dotenv()


@dataclasses.dataclass
class ConverterConfig:
    start_date: dt.date
    end_date: dt.date
    parquet_dir: Path
    qlib_dir: Path
    force: bool
    limit: Optional[int]


class ParquetDataset:
    def __init__(self, root: Path):
        self._root = root
        if not self._root.exists():
            raise FileNotFoundError(f"Parquet directory not found: {root}")

    def available_symbols(self) -> List[str]:
        return sorted({path.stem.upper() for path in self._root.glob("*.parquet")})

    def load(self, symbol: str) -> pd.DataFrame:
        path = self._root / f"{symbol.lower()}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Parquet file missing for {symbol}: {path}")
        return pd.read_parquet(path)


class QlibExporter:
    def __init__(self, config: ConverterConfig, calendar: Sequence[pd.Timestamp]):
        self._config = config
        self._calendar = list(calendar)
        if not self._calendar:
            raise RuntimeError("Trading calendar is empty; cannot build qlib dataset")
        self._dumper = DumpDataAll(
            data_path=str(config.parquet_dir),
            qlib_dir=str(config.qlib_dir),
            file_suffix=".parquet",
            date_field_name="date",
            symbol_field_name="symbol",
            include_fields="open,close,high,low,volume,money,turnover,factor",
            max_workers=1,
        )
        self._instrument_lock = threading.Lock()
        self._ranges: Dict[str, Tuple[dt.date, dt.date]] = self._load_ranges()
        self._instrument_file = (
            self._dumper._instruments_dir.joinpath(self._dumper.INSTRUMENTS_FILE_NAME)
        )
        self._ensure_calendar_file()

    def _ensure_calendar_file(self) -> None:
        self._dumper.save_calendars(self._calendar)

    def _load_ranges(self) -> Dict[str, Tuple[dt.date, dt.date]]:
        instrument_file = self._dumper._instruments_dir.joinpath(self._dumper.INSTRUMENTS_FILE_NAME)
        if not instrument_file.exists():
            return {}
        entries: Dict[str, Tuple[dt.date, dt.date]] = {}
        for line in instrument_file.read_text().splitlines():
            parts = line.strip().split(self._dumper.INSTRUMENTS_SEP)
            if len(parts) < 3:
                continue
            try:
                start = dt.datetime.strptime(parts[1], "%Y-%m-%d").date()
                end = dt.datetime.strptime(parts[2], "%Y-%m-%d").date()
            except ValueError:
                continue
            entries[parts[0].upper()] = (start, end)
        return entries

    def is_up_to_date(self, symbol: str, required_end: dt.date) -> bool:
        entry = self._ranges.get(symbol.upper())
        if not entry:
            return False
        return entry[1] >= required_end

    def export(self, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        working = df.copy()
        working["symbol"] = working["symbol"].str.upper()
        working.sort_values("date", inplace=True)
        self._dumper._dump_bin(working, self._calendar)
        start = working["date"].min().date()
        end = working["date"].max().date()
        self._update_range(working["symbol"].iat[0], start, end)

    def _update_range(self, symbol: str, start: dt.date, end: dt.date) -> None:
        key = symbol.upper()
        with self._instrument_lock:
            previous = self._ranges.get(key)
            if previous:
                start = min(start, previous[0])
                end = max(end, previous[1])
            self._ranges[key] = (start, end)
            lines = [
                f"{sym}{self._dumper.INSTRUMENTS_SEP}"
                f"{rng[0].strftime('%Y-%m-%d')}{self._dumper.INSTRUMENTS_SEP}"
                f"{rng[1].strftime('%Y-%m-%d')}"
                for sym, rng in sorted(self._ranges.items())
            ]
            self._instrument_file.parent.mkdir(parents=True, exist_ok=True)
            self._instrument_file.write_text("\n".join(lines))


class ParquetToQlibConverter:
    def __init__(self, config: ConverterConfig, dataset: ParquetDataset, exporter: QlibExporter):
        self._config = config
        self._dataset = dataset
        self._exporter = exporter
        self._logger = logging.getLogger("parquet_to_qlib")

    def run(self, requested_symbols: Sequence[str]) -> Dict[str, int]:
        symbols = self._normalize_symbols(requested_symbols)
        if self._config.limit is not None:
            symbols = symbols[: self._config.limit]
        if not symbols:
            raise RuntimeError("No symbols found for conversion")
        summary = {"processed": 0, "skipped": 0, "failed": 0}
        for symbol in symbols:
            try:
                if not self._config.force and self._exporter.is_up_to_date(symbol, self._config.end_date):
                    summary["skipped"] += 1
                    continue
                frame = self._load_frame(symbol)
                if frame.empty:
                    self._logger.info("No rows inside date range for %s", symbol)
                    summary["skipped"] += 1
                    continue
                self._exporter.export(frame)
                summary["processed"] += 1
            except Exception as exc:  # noqa: BLE001 - best effort logging
                summary["failed"] += 1
                self._logger.error("Failed to convert %s: %s", symbol, exc)
        return summary

    def _normalize_symbols(self, requested: Sequence[str]) -> List[str]:
        if requested:
            return sorted({symbol.upper() for symbol in requested})
        return self._dataset.available_symbols()

    def _load_frame(self, symbol: str) -> pd.DataFrame:
        raw = self._dataset.load(symbol)
        if raw.empty:
            return raw
        frame = raw.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.dropna(subset=["date"])
        mask = (frame["date"].dt.date >= self._config.start_date) & (
            frame["date"].dt.date <= self._config.end_date
        )
        frame = frame.loc[mask].copy()
        frame.sort_values("date", inplace=True)
        frame.reset_index(drop=True, inplace=True)
        frame["symbol"] = symbol.upper()
        return frame


def resolve_dates(start_str: str, end_str: Optional[str], calendar: TradingCalendar) -> tuple[dt.date, dt.date]:
    start = dt.datetime.strptime(start_str, "%Y-%m-%d").date()
    if end_str:
        end = dt.datetime.strptime(end_str, "%Y-%m-%d").date()
    else:
        end = calendar.latest_session()
    if end < start:
        raise ValueError("End date must be on or after start date")
    latest = calendar.latest_session(end)
    if end > latest:
        end = latest
    return start, end


def resolve_symbols(args, dataset: ParquetDataset) -> List[str]:
    if args.symbols:
        return [inst.symbol.upper() for inst in parse_symbols(args.symbols)]
    if args.symbols_file:
        entries = load_symbols_from_file(Path(args.symbols_file))
        return [inst.symbol.upper() for inst in entries]
    return dataset.available_symbols()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert per-symbol Parquet files into qlib format.")
    parser.add_argument("--parquet-dir", default="data/daily", help="Directory containing parquet files.")
    parser.add_argument("--qlib-dir", default="data/qlib_1d_c", help="Destination qlib directory.")
    parser.add_argument("--start-date", default="1990-01-01", help="Start date YYYY-MM-DD.")
    parser.add_argument("--end-date", default=None, help="End date YYYY-MM-DD (defaults to latest trading day).")
    parser.add_argument("--symbols", nargs="*", default=None, help="Restrict conversion to explicit symbols.")
    parser.add_argument("--symbols-file", default=None, help="File containing symbols (csv/json/plain).")
    parser.add_argument("--force", action="store_true", help="Rebuild qlib data even if already up to date.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of symbols for debugging.")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_argument_parser()
    args = parser.parse_args()

    calendar = TradingCalendar()
    start_date, end_date = resolve_dates(args.start_date, args.end_date, calendar)
    config = ConverterConfig(
        start_date=start_date,
        end_date=end_date,
        parquet_dir=Path(args.parquet_dir),
        qlib_dir=Path(args.qlib_dir),
        force=args.force,
        limit=args.limit,
    )
    config.qlib_dir.mkdir(parents=True, exist_ok=True)
    dataset = ParquetDataset(config.parquet_dir)
    symbol_list = resolve_symbols(args, dataset)
    calendar_slice = [ts for ts in calendar.dates if start_date <= ts.date() <= end_date]
    exporter = QlibExporter(config, calendar_slice)
    converter = ParquetToQlibConverter(config, dataset, exporter)
    stats = converter.run(symbol_list)
    logging.info("Qlib conversion finished: %s", stats)


if __name__ == "__main__":
    main()
