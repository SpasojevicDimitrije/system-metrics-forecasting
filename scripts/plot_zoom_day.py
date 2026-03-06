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
        raise SystemExit("No CSV files found")

    csv_path = csvs[0]
    vm_name = csv_path.stem

    df = load_bitbrains_csv(csv_path)
    df = df[FEATURES]

    # Pick one day with spikes
    day = df.index[0].date()
    zoom = df[df.index.date == day]

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    for feature in FEATURES:
        ax.plot(zoom.index, zoom[feature], label=feature)

    ax.legend()
    ax.set_title(f"Metrics for {day}")
    ax.set_xlabel("Time")

    out_dir = Path("reports/figures") / vm_name
    out_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_dir / "zoom_one_day.png", dpi=150, bbox_inches="tight")

    print("Saved:", out_dir / "zoom_one_day.png")


if __name__ == "__main__":
    main()
