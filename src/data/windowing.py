from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class WindowConfig:
    lookback: int = 12  # number of timesteps in the input window
    horizon: int = 1    # predict t + horizon (1-step ahead)


def make_windows(
    X: np.ndarray,
    cfg: WindowConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a multivariate time series X (T, D) into supervised windows.

    Returns:
      - X_seq: (N, lookback, D)
      - y:     (N, D)  where y[i] = X[t + horizon - 1]
    """
    if X.ndim != 2:
        raise ValueError(f'Expected X to have shape (T, D). Got {X.shape}')

    T, D = X.shape
    L = cfg.lookback
    H = cfg.horizon

    if L <= 0 or H <= 0:
        raise ValueError('lookback and horizon must be positive integers')

    # last starting index so that window and target fit inside series
    last_start = T - (L + H) + 1
    if last_start <= 0:
        raise ValueError(
            f'Not enough timesteps: T={T}, lookback={L}, horizon={H}'
        )

    X_seq = np.zeros((last_start, L, D), dtype=np.float32)
    y = np.zeros((last_start, D), dtype=np.float32)

    for i in range(last_start):
        X_seq[i] = X[i:i + L]
        y[i] = X[i + L + H - 1]

    return X_seq, y


def flatten_windows(X_seq: np.ndarray) -> np.ndarray:
    """
    Flatten (N, L, D) -> (N, L*D) for models like linear regression.
    """
    if X_seq.ndim != 3:
        raise ValueError(f'Expected X_seq to have shape (N, L, D). Got {X_seq.shape}')
    N, L, D = X_seq.shape
    return X_seq.reshape(N, L * D)


if __name__ == '__main__':
    # quick sanity test
    X = np.arange(30, dtype=np.float32).reshape(10, 3)  # T=10, D=3
    cfg = WindowConfig(lookback=4, horizon=1)
    X_seq, y = make_windows(X, cfg)
    print('X_seq shape:', X_seq.shape)  # (10 - 5 + 1 = 6, 4, 3) => (6,4,3)
    print('y shape:', y.shape)          # (6,3)
    print('First window:\n', X_seq[0])
    print('First target:\n', y[0])
