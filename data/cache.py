from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

from config.timeframes import Timeframe


class ParquetCache:
    """Caches OHLCV data as Parquet files with a SQLite metadata index."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = cache_dir / "index.db"
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_index (
                    symbol TEXT,
                    timeframe TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    file_path TEXT,
                    row_count INTEGER,
                    updated_at TEXT,
                    PRIMARY KEY (symbol, timeframe)
                )
            """)

    def _get_file_path(self, symbol: str, timeframe: Timeframe) -> Path:
        safe_symbol = symbol.replace(".", "_").replace("^", "IDX_")
        path = self.cache_dir / safe_symbol
        path.mkdir(parents=True, exist_ok=True)
        return path / f"{timeframe.value}.parquet"

    def get(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame | None:
        file_path = self._get_file_path(symbol, timeframe)
        if not file_path.exists():
            return None

        df = pd.read_parquet(file_path)
        mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
        filtered = df.loc[mask]
        if filtered.empty:
            return None
        filtered.attrs["symbol"] = symbol
        filtered.attrs["timeframe"] = timeframe.value
        return filtered

    def store(
        self,
        symbol: str,
        timeframe: Timeframe,
        df: pd.DataFrame,
    ) -> None:
        if df.empty:
            return

        file_path = self._get_file_path(symbol, timeframe)

        # Merge with existing data if present
        if file_path.exists():
            existing = pd.read_parquet(file_path)
            df = pd.concat([existing, df])
            df = df[~df.index.duplicated(keep="last")]
            df = df.sort_index()

        df.to_parquet(file_path, engine="pyarrow")

        # Update index
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO cache_index
                   (symbol, timeframe, start_date, end_date, file_path, row_count, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    symbol,
                    timeframe.value,
                    str(df.index.min()),
                    str(df.index.max()),
                    str(file_path),
                    len(df),
                    datetime.now().isoformat(),
                ),
            )

    def has_range(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """SELECT start_date, end_date FROM cache_index
                   WHERE symbol = ? AND timeframe = ?""",
                (symbol, timeframe.value),
            ).fetchone()

        if row is None:
            return False

        cached_start = pd.Timestamp(row[0])
        cached_end = pd.Timestamp(row[1])
        return cached_start <= pd.Timestamp(start) and cached_end >= pd.Timestamp(end)
