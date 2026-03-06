from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from statsmodels.tsa.api import VAR


@dataclass
class FittedVAR:
    results: object
    lag_order: int


def fit_var(
    X_train: np.ndarray,
    maxlags: int = 12,
) -> FittedVAR:
    """
    Fit a VAR model on training data only.

    Returns:
      FittedVAR containing the fitted statsmodels results object
      and the selected lag order.
    """
    if X_train.ndim != 2:
        raise ValueError(f'Expected X_train to have shape (T, D). Got {X_train.shape}')

    model = VAR(X_train)
    results = model.fit(maxlags=maxlags, ic='aic')

    lag_order = results.k_ar
    if lag_order <= 0:
        raise ValueError('VAR selected lag order 0, which is not useful for forecasting.')

    return FittedVAR(results=results, lag_order=lag_order)


def var_predict_sequence(
    fitted: FittedVAR,
    history: np.ndarray,
    X_future: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Produce recursive one-step-ahead forecasts over X_future.

    Important:
    - fitted model is trained once on train data
    - history should initially be the training sequence
    - after each prediction step, the true observed point from X_future
      is appended to history (teacher forcing evaluation)

    Returns:
      y_true: (N, D)
      y_pred: (N, D)
    """
    if history.ndim != 2 or X_future.ndim != 2:
        raise ValueError(
            f'Expected 2D arrays, got history={history.shape}, X_future={X_future.shape}'
        )

    p = fitted.lag_order
    results = fitted.results

    hist = history.copy()
    y_true = []
    y_pred = []

    for t in range(len(X_future)):
        forecast_input = hist[-p:]
        pred = results.forecast(y=forecast_input, steps=1)[0]

        y_true.append(X_future[t])
        y_pred.append(pred)

        # append the true next observation before predicting the following step
        hist = np.vstack([hist, X_future[t]])

    return np.asarray(y_true), np.asarray(y_pred)
