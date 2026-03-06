from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data.load_bitbrains import load_bitbrains_csv


FEATURES = [
    'CPU usage [%]',
    'Memory usage [KB]',
    'Disk read throughput [KB/s]',
    'Disk write throughput [KB/s]',
    'Network received throughput [KB/s]',
    'Network transmitted throughput [KB/s]',
]


def main() -> None:
    raw_dir = Path('data/raw')
    csvs = sorted(raw_dir.glob('*.csv'))

    if not csvs:
        raise SystemExit('No CSV files found in data/raw')

    csv_path = csvs[0]
    vm_name = csv_path.stem

    df = load_bitbrains_csv(csv_path)
    df = df[FEATURES]

    corr = df.corr()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    cax = ax.imshow(corr.values)

    ax.set_xticks(range(len(FEATURES)))
    ax.set_yticks(range(len(FEATURES)))

    ax.set_xticklabels(FEATURES, rotation=45, ha='right')
    ax.set_yticklabels(FEATURES)

    fig.colorbar(cax)

    ax.set_title("Metric Correlation Matrix")

    out_dir = Path("reports/figures") / vm_name
    out_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_dir / "correlation_matrix.png", dpi=150, bbox_inches="tight")

    print("Saved:", out_dir / "correlation_matrix.png")


if __name__ == "__main__":
    main()
