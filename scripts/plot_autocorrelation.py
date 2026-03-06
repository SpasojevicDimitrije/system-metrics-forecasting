from pathlib import Path

import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot

from src.data.load_bitbrains import load_bitbrains_csv


FEATURES = [
    'CPU usage [%]',
    'Memory usage [KB]',
    'Disk read throughput [KB/s]',
    'Disk write throughput [KB/s]',
    'Network received throughput [KB/s]',
    'Network transmitted throughput [KB/s]',
]


def main():
    raw_dir = Path("data/raw")
    csvs = sorted(raw_dir.glob("*.csv"))

    if not csvs:
        raise SystemExit("No CSV files found")

    csv_path = csvs[0]
    vm_name = csv_path.stem

    df = load_bitbrains_csv(csv_path)

    out_dir = Path("reports/figures") / vm_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for feature in FEATURES:

        fig = plt.figure()
        ax = fig.add_subplot(111)

        autocorrelation_plot(df[feature], ax=ax)

        ax.set_title(f"Autocorrelation — {feature}")

        safe_name = (
            feature.replace(" ", "_")
            .replace("[", "")
            .replace("]", "")
            .replace("/", "_per_")
            .replace("%", "pct")
        )

        fig.savefig(
            out_dir / f"autocorr_{safe_name}.png",
            dpi=150,
            bbox_inches="tight"
        )

        plt.close(fig)

        print("Saved autocorr plot:", feature)


if __name__ == "__main__":
    main()
