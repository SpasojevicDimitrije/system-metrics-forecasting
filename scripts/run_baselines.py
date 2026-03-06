from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.windowing import WindowConfig
from src.evaluation.metrics import mae, per_feature_rmse, rmse
from src.models.baselines import persistence_predict, ridge_train_predict
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


def _find_first_vm_prefix() -> str:
    processed = Path('data/processed')
    train_files = sorted(processed.glob('*_X_train.npy'))
    if not train_files:
        raise SystemExit('No processed arrays found. Run: python3 -m src.data.preprocess')
    return train_files[0].name.replace('_X_train.npy', '')


def _print_metrics(
    model_name: str,
    y_val_true: np.ndarray,
    y_val_pred: np.ndarray,
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
) -> None:
    print(f'\n{model_name}')
    print('  Val  MAE:', mae(y_val_true, y_val_pred), 'RMSE:', rmse(y_val_true, y_val_pred))
    print('  Test MAE:', mae(y_test_true, y_test_pred), 'RMSE:', rmse(y_test_true, y_test_pred))

    pf_rmse = per_feature_rmse(y_test_true, y_test_pred)
    print('  Test per-feature RMSE:')
    for name, value in zip(FEATURE_NAMES, pf_rmse):
        print(f'    {name}: {value:.6f}')


def main() -> None:
    vm_prefix = _find_first_vm_prefix()
    X_train, X_val, X_test = _load_split(vm_prefix)

    cfg = WindowConfig(lookback=12, horizon=1)

    print('VM prefix:', vm_prefix)
    print('Shapes (train/val/test):', X_train.shape, X_val.shape, X_test.shape)
    print('Window config:', cfg)

    # Persistence baseline
    y_val_true, y_val_pred = persistence_predict(X_val, cfg)
    y_test_true, y_test_pred = persistence_predict(X_test, cfg)
    _print_metrics(
        'Persistence baseline',
        y_val_true,
        y_val_pred,
        y_test_true,
        y_test_pred,
    )

    # Ridge regression baseline
    y_val_true, y_val_pred = ridge_train_predict(X_train, X_val, cfg, alpha=1.0)
    y_test_true, y_test_pred = ridge_train_predict(X_train, X_test, cfg, alpha=1.0)
    _print_metrics(
        'Ridge regression baseline (alpha=1.0)',
        y_val_true,
        y_val_pred,
        y_test_true,
        y_test_pred,
    )

    # VAR baseline: fit once on train, evaluate on val and test with same fitted model
    fitted_var = fit_var(X_train, maxlags=12)

    y_val_true, y_val_pred = var_predict_sequence(
        fitted=fitted_var,
        history=X_train,
        X_future=X_val,
    )

    y_test_true, y_test_pred = var_predict_sequence(
        fitted=fitted_var,
        history=X_train,
        X_future=X_test,
    )

    print(f'\nVAR selected lag from train: {fitted_var.lag_order}')
    _print_metrics(
        'VAR baseline',
        y_val_true,
        y_val_pred,
        y_test_true,
        y_test_pred,
    )


if __name__ == '__main__':
    main()
