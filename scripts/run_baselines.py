from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from src.data.windowing import WindowConfig
from src.evaluation.metrics import mae, per_feature_rmse, rmse
from src.models.baselines import fit_ridge, persistence_predict, ridge_predict
from src.models.var_model import fit_var, var_predict_sequence


FEATURE_NAMES = [
    'CPU usage [%]',
    'Memory usage [KB]',
    'Disk read throughput [KB/s]',
    'Disk write throughput [KB/s]',
    'Network received throughput [KB/s]',
    'Network transmitted throughput [KB/s]',
]


def _load_split(vm_prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    processed = Path('data/processed')
    X_train = np.load(processed / f'{vm_prefix}_X_train.npy')
    X_val = np.load(processed / f'{vm_prefix}_X_val.npy')
    X_test = np.load(processed / f'{vm_prefix}_X_test.npy')
    return X_train, X_val, X_test


def _load_scaler(vm_prefix: str):
    processed = Path('data/processed')
    scaler_path = processed / f'{vm_prefix}_scaler.joblib'
    if not scaler_path.exists():
        raise SystemExit(
            f'Missing scaler file: {scaler_path}. '
            'Run: python3 -m src.data.preprocess'
        )
    return joblib.load(scaler_path)


def _find_first_vm_prefix() -> str:
    processed = Path('data/processed')
    train_files = sorted(processed.glob('*_X_train.npy'))
    if not train_files:
        raise SystemExit('No processed arrays found. Run: python3 -m src.data.preprocess')
    return train_files[0].name.replace('_X_train.npy', '')


def _inverse_transform_targets(
    scaler,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_true_inv = scaler.inverse_transform(y_true)
    y_pred_inv = scaler.inverse_transform(y_pred)
    return y_true_inv, y_pred_inv


def _print_metrics(
    model_name: str,
    scaler,
    y_val_true: np.ndarray,
    y_val_pred: np.ndarray,
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
) -> None:
    print(f'\n{model_name}')

    # Scaled-space metrics: useful for fair model comparison
    print('  Val  scaled MAE:', mae(y_val_true, y_val_pred), 'scaled RMSE:', rmse(y_val_true, y_val_pred))
    print('  Test scaled MAE:', mae(y_test_true, y_test_pred), 'scaled RMSE:', rmse(y_test_true, y_test_pred))

    # Original-unit metrics: useful for interpretation
    y_val_true_inv, y_val_pred_inv = _inverse_transform_targets(scaler, y_val_true, y_val_pred)
    y_test_true_inv, y_test_pred_inv = _inverse_transform_targets(scaler, y_test_true, y_test_pred)

    print('  Val  original-unit MAE:', mae(y_val_true_inv, y_val_pred_inv), 'original-unit RMSE:', rmse(y_val_true_inv, y_val_pred_inv))
    print('  Test original-unit MAE:', mae(y_test_true_inv, y_test_pred_inv), 'original-unit RMSE:', rmse(y_test_true_inv, y_test_pred_inv))

    pf_rmse_scaled = per_feature_rmse(y_test_true, y_test_pred)
    print('  Test per-feature RMSE (scaled):')
    for name, value in zip(FEATURE_NAMES, pf_rmse_scaled):
        print(f'    {name}: {value:.6f}')

    pf_rmse_original = per_feature_rmse(y_test_true_inv, y_test_pred_inv)
    print('  Test per-feature RMSE (original units):')
    for name, value in zip(FEATURE_NAMES, pf_rmse_original):
        print(f'    {name}: {value:.6f}')


def main() -> None:
    vm_prefix = _find_first_vm_prefix()
    X_train, X_val, X_test = _load_split(vm_prefix)
    scaler = _load_scaler(vm_prefix)

    cfg = WindowConfig(lookback=12, horizon=1)

    print('VM prefix:', vm_prefix)
    print('Shapes (train/val/test):', X_train.shape, X_val.shape, X_test.shape)
    print('Window config:', cfg)

    val_history = X_train
    test_history = np.vstack([X_train, X_val])

    # Persistence baseline
    y_val_true, y_val_pred = persistence_predict(val_history, X_val, cfg)
    y_test_true, y_test_pred = persistence_predict(test_history, X_test, cfg)
    _print_metrics(
        'Persistence baseline',
        scaler,
        y_val_true,
        y_val_pred,
        y_test_true,
        y_test_pred,
    )

    # Ridge regression baseline
    ridge_model = fit_ridge(X_train, cfg, alpha=1.0)

    y_val_true, y_val_pred = ridge_predict(ridge_model, val_history, X_val, cfg)
    y_test_true, y_test_pred = ridge_predict(ridge_model, test_history, X_test, cfg)
    _print_metrics(
        'Ridge regression baseline (alpha=1.0)',
        scaler,
        y_val_true,
        y_val_pred,
        y_test_true,
        y_test_pred,
    )

    # VAR baseline
    fitted_var = fit_var(X_train, maxlags=12)

    y_val_true, y_val_pred = var_predict_sequence(
        fitted=fitted_var,
        history=val_history,
        X_future=X_val,
    )

    y_test_true, y_test_pred = var_predict_sequence(
        fitted=fitted_var,
        history=test_history,
        X_future=X_test,
    )

    print(f'\nVAR selected lag from train: {fitted_var.lag_order}')
    _print_metrics(
        'VAR baseline',
        scaler,
        y_val_true,
        y_val_pred,
        y_test_true,
        y_test_pred,
    )


if __name__ == '__main__':
    main()
