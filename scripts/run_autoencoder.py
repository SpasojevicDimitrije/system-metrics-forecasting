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
from src.models.autoencoder import AutoencoderConfig, FeedforwardAutoencoder
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
    parser = argparse.ArgumentParser(description='Train autoencoder + latent forecaster.')
    parser.add_argument('--lookback', type=int, default=24)
    parser.add_argument('--horizon', type=int, default=1)
    parser.add_argument('--latent-size', type=int, default=3)
    parser.add_argument('--ae-hidden-size', type=int, default=16)
    parser.add_argument('--rnn-type', type=str, default='gru', choices=['lstm', 'gru'])
    parser.add_argument('--rnn-hidden-size', type=int, default=32)

    parser.add_argument('--ae-batch-size', type=int, default=128)
    parser.add_argument('--ae-lr', type=float, default=1e-3)
    parser.add_argument('--ae-weight-decay', type=float, default=1e-5)
    parser.add_argument('--ae-epochs', type=int, default=100)
    parser.add_argument('--ae-patience', type=int, default=10)

    parser.add_argument('--rnn-batch-size', type=int, default=64)
    parser.add_argument('--rnn-lr', type=float, default=1e-3)
    parser.add_argument('--rnn-weight-decay', type=float, default=1e-5)
    parser.add_argument('--rnn-epochs', type=int, default=100)
    parser.add_argument('--rnn-patience', type=int, default=10)
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


def make_vector_loader(
    x: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    x_tensor = torch.tensor(x, dtype=torch.float32)
    ds = TensorDataset(x_tensor, x_tensor)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def make_sequence_loader(
    x_seq: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    x_tensor = torch.tensor(x_seq, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    ds = TensorDataset(x_tensor, y_tensor)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def reconstruct(
    model: FeedforwardAutoencoder,
    x: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(torch.tensor(x, dtype=torch.float32), batch_size=batch_size, shuffle=False)

    outputs = []
    for xb in loader:
        xb = xb.to(device)
        x_hat = model(xb)
        outputs.append(x_hat.cpu().numpy())

    return np.vstack(outputs)


@torch.no_grad()
def encode_array(
    model: FeedforwardAutoencoder,
    x: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(torch.tensor(x, dtype=torch.float32), batch_size=batch_size, shuffle=False)

    outputs = []
    for xb in loader:
        xb = xb.to(device)
        z = model.encode(xb)
        outputs.append(z.cpu().numpy())

    return np.vstack(outputs)


@torch.no_grad()
def decode_array(
    model: FeedforwardAutoencoder,
    z: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(torch.tensor(z, dtype=torch.float32), batch_size=batch_size, shuffle=False)

    outputs = []
    for zb in loader:
        zb = zb.to(device)
        x_hat = model.decode(zb)
        outputs.append(x_hat.cpu().numpy())

    return np.vstack(outputs)


@torch.no_grad()
def predict_latent_model(
    model: nn.Module,
    x_seq: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(torch.tensor(x_seq, dtype=torch.float32), batch_size=batch_size, shuffle=False)

    preds = []
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('VM prefix:', vm_prefix)
    print('Shapes (train/val/test):', x_train.shape, x_val.shape, x_test.shape)
    print('Device:', device)

    # Train autoencoder
    ae_cfg = AutoencoderConfig(
        input_size=x_train.shape[1],
        latent_size=args.latent_size,
        hidden_size=args.ae_hidden_size,
    )
    ae = FeedforwardAutoencoder(ae_cfg).to(device)

    ae_train_loader = make_vector_loader(x_train, batch_size=args.ae_batch_size, shuffle=True)
    ae_criterion = nn.MSELoss()
    ae_optimizer = torch.optim.Adam(
        ae.parameters(),
        lr=args.ae_lr,
        weight_decay=args.ae_weight_decay,
    )

    best_ae_val_loss = float('inf')
    best_ae_epoch = 0
    best_ae_state = None
    ae_bad_epochs = 0

    for epoch in range(1, args.ae_epochs + 1):
        ae.train()
        running_loss = 0.0
        n_samples = 0

        for xb, yb in ae_train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            ae_optimizer.zero_grad()
            x_hat = ae(xb)
            loss = ae_criterion(x_hat, yb)
            loss.backward()
            ae_optimizer.step()

            batch_n = xb.shape[0]
            running_loss += loss.item() * batch_n
            n_samples += batch_n

        train_loss = running_loss / n_samples

        x_val_recon = reconstruct(ae, x_val, device=device)
        val_loss = rmse(x_val, x_val_recon)

        print(f'AE Epoch {epoch:03d} | train_rmse={train_loss:.6f} | val_rmse={val_loss:.6f}')

        if val_loss < best_ae_val_loss:
            best_ae_val_loss = val_loss
            best_ae_epoch = epoch
            best_ae_state = copy.deepcopy(ae.state_dict())
            ae_bad_epochs = 0
        else:
            ae_bad_epochs += 1

        if ae_bad_epochs >= args.ae_patience:
            print(f'\nAE early stopping triggered at epoch {epoch}.')
            break

    if best_ae_state is None:
        raise RuntimeError('Autoencoder training failed: no best state stored.')

    ae.load_state_dict(best_ae_state)
    print(f'\nBest AE epoch: {best_ae_epoch} with val RMSE: {best_ae_val_loss:.6f}')

    # Encode full series
    z_train = encode_array(ae, x_train, device=device)
    z_val = encode_array(ae, x_val, device=device)
    z_test = encode_array(ae, x_test, device=device)

    window_cfg = WindowConfig(lookback=args.lookback, horizon=args.horizon)

    z_train_seq, z_train_target = make_windows(z_train, window_cfg)
    z_val_seq, z_val_target = make_eval_windows(history=z_train, future=z_val, cfg=window_cfg)
    z_test_seq, z_test_target = make_eval_windows(
        history=np.vstack([z_train, z_val]),
        future=z_test,
        cfg=window_cfg,
    )

    print('\nLatent shapes:')
    print('  z_train:', z_train.shape)
    print('  z_val:  ', z_val.shape)
    print('  z_test: ', z_test.shape)
    print('Window config:', window_cfg)
    print('Latent windowed shapes:')
    print('  Train:', z_train_seq.shape, z_train_target.shape)
    print('  Val:  ', z_val_seq.shape, z_val_target.shape)
    print('  Test: ', z_test_seq.shape, z_test_target.shape)

    # Forecast in latent space
    rnn_cfg = RNNConfig(
        input_size=args.latent_size,
        output_size=args.latent_size,
        rnn_type=args.rnn_type,
        hidden_size=args.rnn_hidden_size,
        num_layers=1,
        dropout=0.0,
    )
    latent_model = RecurrentForecaster(rnn_cfg).to(device)

    train_loader = make_sequence_loader(
        z_train_seq,
        z_train_target,
        batch_size=args.rnn_batch_size,
        shuffle=True,
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        latent_model.parameters(),
        lr=args.rnn_lr,
        weight_decay=args.rnn_weight_decay,
    )

    best_val_rmse = float('inf')
    best_epoch = 0
    best_state = None
    bad_epochs = 0

    for epoch in range(1, args.rnn_epochs + 1):
        latent_model.train()
        running_loss = 0.0
        n_samples = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            y_hat = latent_model(xb)
            loss = criterion(y_hat, yb)
            loss.backward()
            clip_grad_norm_(latent_model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            batch_n = xb.shape[0]
            running_loss += loss.item() * batch_n
            n_samples += batch_n

        train_loss = running_loss / n_samples

        z_val_pred = predict_latent_model(latent_model, z_val_seq, device=device)
        val_rmse = rmse(z_val_target, z_val_pred)
        val_mae = mae(z_val_target, z_val_pred)

        print(
            f'LAT Epoch {epoch:03d} | '
            f'train_loss={train_loss:.6f} | '
            f'val_latent_mae={val_mae:.6f} | '
            f'val_latent_rmse={val_rmse:.6f}'
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_state = copy.deepcopy(latent_model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= args.rnn_patience:
            print(f'\nLatent forecaster early stopping triggered at epoch {epoch}.')
            break

    if best_state is None:
        raise RuntimeError('Latent forecaster training failed: no best state stored.')

    latent_model.load_state_dict(best_state)
    print(f'\nBest latent epoch: {best_epoch} with val latent RMSE: {best_val_rmse:.6f}')

    # Decode predictions back to original space
    z_val_pred = predict_latent_model(latent_model, z_val_seq, device=device)
    z_test_pred = predict_latent_model(latent_model, z_test_seq, device=device)

    x_val_pred = decode_array(ae, z_val_pred, device=device)
    x_test_pred = decode_array(ae, z_test_pred, device=device)

    _, x_val_true = make_eval_windows(history=x_train, future=x_val, cfg=window_cfg)
    _, x_test_true = make_eval_windows(
        history=np.vstack([x_train, x_val]),
        future=x_test,
        cfg=window_cfg,
    )

    print_metrics(
        f'Autoencoder + {args.rnn_type.upper()} latent forecasting',
        scaler,
        y_val_true=x_val_true,
        y_val_pred=x_val_pred,
        y_test_true=x_test_true,
        y_test_pred=x_test_pred,
    )


if __name__ == '__main__':
    main()
