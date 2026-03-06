from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from src.data.windowing import WindowConfig, make_windows
from src.evaluation.metrics import mae, per_feature_rmse, rmse
from src.models.baselines import make_eval_windows
from src.models.rnn_model import RNNConfig, RecurrentForecaster


FEATURE_NAMES = [
    'CPU usage [%]',
    'Memory usage [KB]',
    'Disk read throughput [KB/s]',
    'Disk write throughput [KB/s]',
    'Network received throughput [KB/s]',
    'Network transmitted throughput [KB/s]',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train recurrent forecaster on Bitbrains VM data.')
    parser.add_argument('--rnn-type', type=str, default='lstm', choices=['lstm', 'gru'])
    parser.add_argument('--lookback', type=int, default=12)
    parser.add_argument('--horizon', type=int, default=1)
    parser.add_argument('--hidden-size', type=int, default=32)
    parser.add_argument('--num-layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_first_vm_prefix() -> str:
    processed = Path('data/processed')
    train_files = sorted(processed.glob('*_X_train.npy'))
    if not train_files:
        raise SystemExit('No processed arrays found. Run: python3 -m src.data.preprocess')
    return train_files[0].name.replace('_X_train.npy', '')


def load_split(vm_prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    processed = Path('data/processed')
    x_train = np.load(processed / f'{vm_prefix}_X_train.npy')
    x_val = np.load(processed / f'{vm_prefix}_X_val.npy')
    x_test = np.load(processed / f'{vm_prefix}_X_test.npy')
    return x_train, x_val, x_test


def load_scaler(vm_prefix: str):
    processed = Path('data/processed')
    scaler_path = processed / f'{vm_prefix}_scaler.joblib'
    if not scaler_path.exists():
        raise SystemExit(
            f'Missing scaler file: {scaler_path}. '
            'Run: python3 -m src.data.preprocess'
        )
    return joblib.load(scaler_path)


def inverse_transform_targets(
    scaler,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_true_inv = scaler.inverse_transform(y_true)
    y_pred_inv = scaler.inverse_transform(y_pred)
    return y_true_inv, y_pred_inv


def make_loader(
    x_seq: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    x_tensor = torch.tensor(x_seq, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def predict_model(
    model: nn.Module,
    x_seq: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()

    preds = []
    x_tensor = torch.tensor(x_seq, dtype=torch.float32)
    loader = DataLoader(x_tensor, batch_size=batch_size, shuffle=False)

    for xb in loader:
        xb = xb.to(device)
        y_hat = model(xb)
        preds.append(y_hat.cpu().numpy())

    return np.vstack(preds)


def print_metrics(
    model_name: str,
    scaler,
    y_val_true: np.ndarray,
    y_val_pred: np.ndarray,
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
) -> None:
    print(f'\n{model_name}')
    print('  Val  scaled MAE:', mae(y_val_true, y_val_pred), 'scaled RMSE:', rmse(y_val_true, y_val_pred))
    print('  Test scaled MAE:', mae(y_test_true, y_test_pred), 'scaled RMSE:', rmse(y_test_true, y_test_pred))

    y_val_true_inv, y_val_pred_inv = inverse_transform_targets(scaler, y_val_true, y_val_pred)
    y_test_true_inv, y_test_pred_inv = inverse_transform_targets(scaler, y_test_true, y_test_pred)

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
    args = parse_args()
    set_seed(args.seed)

    vm_prefix = find_first_vm_prefix()
    x_train, x_val, x_test = load_split(vm_prefix)
    scaler = load_scaler(vm_prefix)

    window_cfg = WindowConfig(lookback=args.lookback, horizon=args.horizon)

    x_train_seq, y_train = make_windows(x_train, window_cfg)
    x_val_seq, y_val = make_eval_windows(history=x_train, future=x_val, cfg=window_cfg)
    x_test_seq, y_test = make_eval_windows(
        history=np.vstack([x_train, x_val]),
        future=x_test,
        cfg=window_cfg,
    )

    print('VM prefix:', vm_prefix)
    print('Shapes (train/val/test):', x_train.shape, x_val.shape, x_test.shape)
    print('Window config:', window_cfg)
    print('Model config:')
    print(
        f'  rnn_type={args.rnn_type}, hidden_size={args.hidden_size}, '
        f'num_layers={args.num_layers}, dropout={args.dropout}'
    )
    print('Training config:')
    print(
        f'  batch_size={args.batch_size}, lr={args.lr}, weight_decay={args.weight_decay}, '
        f'epochs={args.epochs}, patience={args.patience}, grad_clip={args.grad_clip}'
    )
    print('Windowed shapes:')
    print('  Train:', x_train_seq.shape, y_train.shape)
    print('  Val:  ', x_val_seq.shape, y_val.shape)
    print('  Test: ', x_test_seq.shape, y_test.shape)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    train_loader = make_loader(x_train_seq, y_train, batch_size=args.batch_size, shuffle=True)

    model_cfg = RNNConfig(
        input_size=x_train.shape[1],
        output_size=x_train.shape[1],
        rnn_type=args.rnn_type,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    model = RecurrentForecaster(model_cfg).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_rmse = float('inf')
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_samples = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            y_hat = model(xb)
            loss = criterion(y_hat, yb)
            loss.backward()

            clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

            optimizer.step()

            batch_n = xb.shape[0]
            running_loss += loss.item() * batch_n
            n_samples += batch_n

        train_loss = running_loss / n_samples

        y_val_pred = predict_model(model, x_val_seq, device=device)
        val_rmse = rmse(y_val, y_val_pred)
        val_mae = mae(y_val, y_val_pred)

        print(
            f'Epoch {epoch:03d} | '
            f'train_loss={train_loss:.6f} | '
            f'val_scaled_mae={val_mae:.6f} | '
            f'val_scaled_rmse={val_rmse:.6f}'
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f'\nEarly stopping triggered at epoch {epoch}.')
            break

    if best_state is None:
        raise RuntimeError('Training failed: no best model state was stored.')

    model.load_state_dict(best_state)
    print(f'\nBest epoch: {best_epoch} with val scaled RMSE: {best_val_rmse:.6f}')

    y_val_pred = predict_model(model, x_val_seq, device=device)
    y_test_pred = predict_model(model, x_test_seq, device=device)

    model_name = f'{args.rnn_type.upper()} baseline'
    print_metrics(
        model_name,
        scaler,
        y_val_true=y_val,
        y_val_pred=y_val_pred,
        y_test_true=y_test,
        y_test_pred=y_test_pred,
    )


if __name__ == '__main__':
    main()
