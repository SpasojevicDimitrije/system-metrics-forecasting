from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from src.data.windowing import WindowConfig, flatten_windows, make_windows


def make_eval_windows(
    history: np.ndarray,
    future: np.ndarray,
    cfg: WindowConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build evaluation windows for forecasting `future` using the most recent
    context from `history`.

    This makes window-based models comparable to recursive models like VAR.

    Parameters
    ----------
    history : np.ndarray
        Past observed sequence of shape (T_hist, D).
    future : np.ndarray
        Evaluation segment of shape (T_future, D).
    cfg : WindowConfig
        Window configuration.

    Returns
    -------
    X_seq : np.ndarray
        Input windows of shape (N, lookback, D).
    y_true : np.ndarray
        Targets of shape (N, D), aligned so that all targets belong to `future`.

    Notes
    -----
    For horizon=1, the first prediction in `future` uses the last `lookback`
    points from `history`.
    """
    if history.ndim != 2 or future.ndim != 2:
        raise ValueError(f'Expected 2D arrays, got history={history.shape}, future={future.shape}')

    L = cfg.lookback
    H = cfg.horizon

    if len(history) < L:
        raise ValueError(
            f'History must contain at least lookback={L} points. Got len(history)={len(history)}'
        )

    if len(future) < H:
        raise ValueError(
            f'Future segment must contain at least horizon={H} points. Got len(future)={len(future)}'
        )

    combined = np.vstack([history[-L:], future])
    X_seq, y_true = make_windows(combined, cfg)
    return X_seq, y_true


def persistence_predict(
    history: np.ndarray,
    future: np.ndarray,
    cfg: WindowConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Persistence baseline on a forecasting segment.

    Predict each target as the last observed vector in its input window.
    """
    X_seq, y_true = make_eval_windows(history, future, cfg)
    y_pred = X_seq[:, -1, :]
    return y_true, y_pred


def fit_ridge(
    X_train: np.ndarray,
    cfg: WindowConfig,
    alpha: float = 1.0,
) -> Ridge:
    """
    Fit a multi-output ridge regression model on train windows only.
    """
    X_seq_tr, y_tr = make_windows(X_train, cfg)
    X_tr = flatten_windows(X_seq_tr)

    model = Ridge(alpha=alpha)
    model.fit(X_tr, y_tr)
    return model


def ridge_predict(
    model: Ridge,
    history: np.ndarray,
    future: np.ndarray,
    cfg: WindowConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate a fitted ridge model on a forecasting segment.
    """
    X_seq_ev, y_true = make_eval_windows(history, future, cfg)
    X_ev = flatten_windows(X_seq_ev)

    y_pred = model.predict(X_ev)
    return y_true, y_pred
