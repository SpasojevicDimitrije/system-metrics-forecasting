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


def _safe_name(s: str) -> str:
    return (
        s.replace(' ', '_')
        .replace('[', '')
        .replace(']', '')
        .replace('%', 'pct')
        .replace('/', '_per_')
        .replace('__', '_')
    )


def main() -> None:
    raw_dir = Path('data/raw')
    csvs = sorted(raw_dir.glob('*.csv'))
    if not csvs:
        raise SystemExit('No CSV files found in data/raw/. Put your Bitbrains CSV there.')

    csv_path = csvs[0]
    vm_name = csv_path.stem

    df = load_bitbrains_csv(csv_path)

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise SystemExit(f'Missing expected columns: {missing}\nFound: {list(df.columns)}')

    df = df[FEATURES].copy()

    out_dir = Path('reports/figures') / vm_name
    out_dir.mkdir(parents=True, exist_ok=True)

    df_plot = df

    # Plot each feature as its own figure
    for col in FEATURES:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(df_plot.index, df_plot[col])
        ax.set_title(f'{vm_name} — {col}')
        ax.set_xlabel('Time')
        ax.set_ylabel(col)
        fig.autofmt_xdate()

        fname = out_dir / f'{_safe_name(col)}.png'
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)

    # Extra: CPU vs Memory (same plot, 2 axes)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    ax1.plot(df_plot.index, df_plot['CPU usage [%]'])
    ax2.plot(df_plot.index, df_plot['Memory usage [KB]'])
    ax1.set_title(f'{vm_name} — CPU usage vs Memory usage')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('CPU usage [%]')
    ax2.set_ylabel('Memory usage [KB]')
    fig.autofmt_xdate()
    fig.savefig(out_dir / 'cpu_vs_memory.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Quick stats CSV
    stats = df.describe().T
    stats.to_csv(out_dir / 'summary_stats.csv')

    print(f'Saved plots to: {out_dir}')
    print('Saved summary stats to:', out_dir / 'summary_stats.csv')


if __name__ == '__main__':
    main()
