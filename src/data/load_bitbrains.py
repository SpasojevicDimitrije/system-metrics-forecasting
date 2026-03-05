from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd


@dataclass(frozen=True)
class BitbrainsColumns:
    timestamp_ms: str = 'Timestamp [ms]'
    cpu_usage_pct: str = 'CPU usage [%]'
    mem_usage_kb: str = 'Memory usage [KB]'
    disk_read_kbs: str = 'Disk read throughput [KB/s]'
    disk_write_kbs: str = 'Disk write throughput [KB/s]'
    net_recv_kbs: str = 'Network received throughput [KB/s]'
    net_trans_kbs: str = 'Network transmitted throughput [KB/s]'


def _clean_colname(col: str) -> str:
    # Example raw header tokens: "Timestamp [ms]" or "CPU usage [%]"
    col = col.strip()
    # Remove duplicate spaces
    col = re.sub(r'\s+', ' ', col)
    # Remove trailing semicolons if present
    col = col.rstrip(';').strip()
    return col


def load_bitbrains_csv(path: str | Path) -> pd.DataFrame:
    """
    Loads one Bitbrains VM trace CSV.
    Returns a DataFrame indexed by datetime (UTC) and sorted by time.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'File not found: {path}')

    # Use python engine for regex separators; handle ';' optionally followed by whitespace/tab
    df = pd.read_csv(
        path,
        sep=r';\s*',          # semicolon + optional whitespace (covers ";\t")
        engine='python',
        header=0,
        na_values=['', 'null', '[null]'],
    )

    # Drop trailing empty column if the file has a trailing separator
    if df.columns.size > 0 and (df.columns[-1] == '' or str(df.columns[-1]).startswith('Unnamed')):
        df = df.iloc[:, :-1]

    # Clean column names
    df.columns = [_clean_colname(c) for c in df.columns]

    # Ensure timestamp exists and parse it
    cols = BitbrainsColumns()
    if cols.timestamp_ms not in df.columns:
        raise ValueError(f"Expected column '{cols.timestamp_ms}' not found. Found: {list(df.columns)}")

    df[cols.timestamp_ms] = pd.to_numeric(df[cols.timestamp_ms], errors='coerce')
    df = df.dropna(subset=[cols.timestamp_ms])

    df['datetime'] = pd.to_datetime(df[cols.timestamp_ms].astype('int64'), unit='s', utc=True)
    df = df.drop(columns=[cols.timestamp_ms]).set_index('datetime').sort_index()

    # Convert all remaining columns to numeric where possible
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    return df


def infer_sampling_seconds(df: pd.DataFrame) -> float | None:
    """
    Returns the median sampling interval in seconds, or None if not enough points.
    """
    if df.index.size < 3:
        return None
    deltas = df.index.to_series().diff().dropna().dt.total_seconds()
    if deltas.empty:
        return None
    return float(deltas.median())


if __name__ == '__main__':
    # Quick local test
    raw_dir = Path('data/raw')
    csvs = sorted(raw_dir.glob('*.csv'))
    if not csvs:
        raise SystemExit('No CSV files found in data/raw/. Put your Bitbrains CSV there.')

    df0 = load_bitbrains_csv(csvs[0])
    print(df0.head())
    print('Rows:', len(df0), 'Cols:', len(df0.columns))
    print('Median sampling interval (s):', infer_sampling_seconds(df0))
    print('Columns:', list(df0.columns))
