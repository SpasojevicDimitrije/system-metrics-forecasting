from pathlib import Path
import numpy as np

from src.data.windowing import WindowConfig, make_windows, flatten_windows


def main() -> None:
    processed = Path('data/processed')
    npys = sorted(processed.glob('*_X_train.npy'))
    if not npys:
        raise SystemExit('No processed train arrays found. Run: python3 -m src.data.preprocess')

    # use first VM found
    X_train_path = npys[0]
    vm_name = X_train_path.name.replace('_X_train.npy', '')
    X_train = np.load(X_train_path)

    cfg = WindowConfig(lookback=12, horizon=1)  # 1 hour history -> next step

    X_seq, y = make_windows(X_train, cfg)
    X_flat = flatten_windows(X_seq)

    print('VM:', vm_name)
    print('X_train:', X_train.shape)
    print('X_seq:', X_seq.shape)
    print('y:', y.shape)
    print('X_flat:', X_flat.shape)


if __name__ == '__main__':
    main()
