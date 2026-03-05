from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from src.data.windowing import WindowConfig, make_windows, flatten_windows


def persistence_predict(X: np.ndarray, cfg: WindowConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    Persistence baseline:
    Predict y = last vector in the input window.

    Returns (y_true, y_pred) aligned with make_windows output.
    """
    X_seq, y_true = make_windows(X, cfg)  # X_seq: (N, L, D)
    y_pred = X_seq[:, -1, :]              # last observed state in window
    return y_true, y_pred


def ridge_train_predict(
    X_train: np.ndarray,
    X_eval: np.ndarray,
    cfg: WindowConfig,
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Ridge regression baseline:
    Flatten windows (L,D) -> (L*D) and fit multi-output ridge.

    Returns (y_true, y_pred) for X_eval.
    """
    X_seq_tr, y_tr = make_windows(X_train, cfg)
    X_tr = flatten_windows(X_seq_tr)

    X_seq_ev, y_true = make_windows(X_eval, cfg)
    X_ev = flatten_windows(X_seq_ev)

    model = Ridge(alpha=alpha)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_ev)
    return y_true, y_pred
