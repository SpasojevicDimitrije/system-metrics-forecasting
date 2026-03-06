from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data.load_bitbrains import load_bitbrains_csv


CPU_MEMORY = [
    'CPU usage [%]',
    'Memory usage [KB]',
]

IO_FEATURES = [
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

    # Pick one day
    day = df.index[0].date()
    zoom = df[df.index.date == day]

    out_dir = Path("reports/figures") / vm_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # CPU + MEMORY plot
    fig1 = plt.figure(figsize=(10, 5))
    ax1 = fig1.add_subplot(111)

    # memory na levoj osi
    ax1.plot(zoom.index, zoom['Memory usage [KB]'], color="tab:orange", label="Memory usage [KB]")
    ax1.set_ylabel("Memory usage [KB]", color="tab:orange")

    # CPU na desnoj osi
    ax2 = ax1.twinx()
    ax2.plot(zoom.index, zoom['CPU usage [%]'], color="tab:blue", label="CPU usage [%]")
    ax2.set_ylabel("CPU usage [%]", color="tab:blue")

    ax1.set_title(f"CPU and Memory usage for {day}")
    ax1.set_xlabel("Time")

    fig1.savefig(
        out_dir / "zoom_cpu_memory.png",
        dpi=150,
        bbox_inches="tight",
    )

    print("Saved:", out_dir / "zoom_cpu_memory.png")

    # DISK + NETWORK plot
    fig2 = plt.figure(figsize=(10, 5))
    ax2 = fig2.add_subplot(111)

    for feature in IO_FEATURES:
        ax2.plot(zoom.index, zoom[feature], label=feature)

    ax2.legend()
    ax2.set_title(f"Disk and Network throughput for {day}")
    ax2.set_xlabel("Time")

    fig2.savefig(
        out_dir / "zoom_io.png",
        dpi=150,
        bbox_inches="tight",
    )

    print("Saved:", out_dir / "zoom_io.png")


if __name__ == "__main__":
    main()
