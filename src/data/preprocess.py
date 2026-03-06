from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

from src.data.load_bitbrains import load_bitbrains_csv


@dataclass(frozen=True)
class PreprocessConfig:
    sampling_seconds: int = 300  # CSV is already sampled at 5 minutes
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    features: Tuple[str, ...] = (
        'CPU usage [%]',
        'Memory usage [KB]',
        'Disk read throughput [KB/s]',
        'Disk write throughput [KB/s]',
        'Network received throughput [KB/s]',
        'Network transmitted throughput [KB/s]',
    )


def _time_split_indices(
    n: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
) -> Tuple[slice, slice, slice]:
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError('train_frac + val_frac + test_frac must sum to 1.0')

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val

    train_sl = slice(0, n_train)
    val_sl = slice(n_train, n_train + n_val)
    test_sl = slice(n_train + n_val, n_train + n_val + n_test)

    return train_sl, val_sl, test_sl


def preprocess_vm_csv(
    csv_path: str | Path,
    cfg: PreprocessConfig,
) -> dict:
    """
    Load one VM trace CSV, select configured features, fill missing values,
    perform a chronological train/val/test split, and standardize using
    train statistics only.

    Returns a dict with:
      - df_features: filled selected features
      - X_train, X_val, X_test: scaled arrays
      - scaler: fitted StandardScaler
      - idx_train, idx_val, idx_test: DatetimeIndex splits
    """
    df = load_bitbrains_csv(csv_path)

    missing = [c for c in cfg.features if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required feature columns: {missing}. Found: {list(df.columns)}')

    df = df.loc[:, cfg.features].copy()

    # The CSV is already sampled correctly. We only handle missing values here.
    df_filled = df.ffill().fillna(0.0)

    n = len(df_filled)
    train_sl, val_sl, test_sl = _time_split_indices(n, cfg.train_frac, cfg.val_frac, cfg.test_frac)

    train_df = df_filled.iloc[train_sl]
    val_df = df_filled.iloc[val_sl]
    test_df = df_filled.iloc[test_sl]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df.values)
    X_val = scaler.transform(val_df.values)
    X_test = scaler.transform(test_df.values)

    return {
        'df_features': df_filled,
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'idx_train': train_df.index,
        'idx_val': val_df.index,
        'idx_test': test_df.index,
        'scaler': scaler,
        'feature_names': list(cfg.features),
        'missing_points': int(df.isna().any(axis=1).sum()),
        'total_points': int(n),
    }


def save_processed(out_dir: str | Path, vm_name: str, payload: dict) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / f'{vm_name}_X_train.npy', payload['X_train'])
    np.save(out_dir / f'{vm_name}_X_val.npy', payload['X_val'])
    np.save(out_dir / f'{vm_name}_X_test.npy', payload['X_test'])

    joblib.dump(payload['scaler'], out_dir / f'{vm_name}_scaler.joblib')

    (out_dir / f'{vm_name}_features.txt').write_text(
        '\n'.join(payload['feature_names']),
        encoding='utf-8',
    )

    pd.Series(payload['idx_train']).to_csv(out_dir / f'{vm_name}_idx_train.csv', index=False)
    pd.Series(payload['idx_val']).to_csv(out_dir / f'{vm_name}_idx_val.csv', index=False)
    pd.Series(payload['idx_test']).to_csv(out_dir / f'{vm_name}_idx_test.csv', index=False)

    meta = [
        f"total_points={payload['total_points']}",
        f"missing_points_before_fill={payload['missing_points']}",
        f"n_train={payload['X_train'].shape[0]}",
        f"n_val={payload['X_val'].shape[0]}",
        f"n_test={payload['X_test'].shape[0]}",
        f"n_features={payload['X_train'].shape[1]}",
    ]
    (out_dir / f'{vm_name}_meta.txt').write_text('\n'.join(meta), encoding='utf-8')


if __name__ == '__main__':
    cfg = PreprocessConfig()

    raw_dir = Path('data/raw')
    csvs = sorted(raw_dir.glob('*.csv'))
    if not csvs:
        raise SystemExit('No CSV files found in data/raw/. Put your Bitbrains CSV there.')

    csv_path = csvs[0]
    vm_name = csv_path.stem

    payload = preprocess_vm_csv(csv_path, cfg)
    print('Feature names:', payload['feature_names'])
    print('Total points:', payload['total_points'])
    print('Missing points before fill:', payload['missing_points'])
    print('Shapes:', payload['X_train'].shape, payload['X_val'].shape, payload['X_test'].shape)

    save_processed('data/processed', vm_name, payload)
    print('Saved to data/processed/')
