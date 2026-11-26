"""Plot the completed H100 prefill GEMM experiment from validated checkpoints."""

import argparse
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

from summarize import activity_coverage, read_results


HERE = Path(__file__).resolve().parent
INK, BLUE, MUTED = "#152733", "#126986", "#52636D"


def plot(tasks, records, output):
    planned = set(tasks.values())
    if len(planned) != 20 or set(records) != planned:
        raise ValueError(f"Final figure requires all 20 validated checkpoints; found {len(records)}.")
    seeds = sorted({seed for seed, _, _, _ in planned})
    rows = sorted({m for _, m, _, _ in planned})
    widths = {(k, n) for _, _, k, n in planned}
    if len(seeds) != 5 or len(rows) != 4 or len(widths) != 1:
        raise ValueError("The final figure expects four row counts, five seeds and fixed projection widths.")
    k, n = widths.pop()
    fig, axes = plt.subplots(1, len(rows), sharey=True, figsize=(10.4, 4.65))
    try:
        fig.subplots_adjust(left=0.089, right=0.977, top=0.68, bottom=0.24, wspace=0.18)
        fig.text(0.04, 0.91, "BF16 prefill projection on the NVIDIA H100",
                 fontsize=19, fontweight="semibold", color=INK)
        fig.text(0.04, 0.844, f"Input width {k:,}   ·   Projection width {n:,}   ·   One point per seed",
                 fontsize=10.5, color=MUTED)

        legend = [
            Line2D([], [], ls="", marker="o", color=BLUE, markerfacecolor=fill, label=label)
            for fill, label in [(BLUE, "Sufficient activity coverage"), ("white", "Insufficient activity coverage")]
        ]
        legend.append(Line2D([], [], ls="--", color=INK, lw=1.3, label="Median of five seeds"))
        fig.legend(handles=legend, loc="upper left", frameon=False, ncol=3,
                   bbox_to_anchor=(0.033, 0.817), columnspacing=2.1, handlelength=1.8)

        for ax, m in zip(axes, rows):
            values = []
            for pos, seed in enumerate(seeds):
                result = records[seed, m, k, n]
                value = result["timing"]["wall_tflops"]
                sufficient, _, _ = activity_coverage(result)
                values.append(value)
                ax.scatter(pos, value, s=49, facecolors=BLUE if sufficient else "white",
                           edgecolors=BLUE, linewidths=1.6, zorder=3)
            ax.axhline(statistics.median(values), ls="--", lw=1.2, color=INK, zorder=2)
            ax.set_title(f"{m:,} token rows", loc="left", color=INK, fontsize=11.5)
            ax.set_xticks(range(len(seeds)), [str(seed) for seed in seeds])
            ax.set_xlim(-0.48, len(seeds) - 0.52)
            ax.grid(axis="y", color="#DEE5E8", lw=0.8)
            ax.set_axisbelow(True)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.spines["bottom"].set_color("#B9C4C9")
            ax.tick_params(length=0, pad=5, labelcolor=INK)

        axes[0].set_ylabel("Wall throughput (TFLOP/s)", fontsize=10.5, color=INK)
        axes[0].yaxis.set_major_locator(MaxNLocator(5))
        axes[0].yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
        fig.text(0.53, 0.137, "Input seed", ha="center", fontsize=10.5, color=INK)
        fig.text(0.04, 0.052, "Throughput medians include every point. Marker fill indicates coverage of GPU activity samples.",
                 fontsize=9.5, color=MUTED)
        fig.savefig(output, dpi=180, facecolor="white")
    finally:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--output", type=Path, default=HERE / "results/prefill-throughput.png")
    args = parser.parse_args()
    try:
        plot(*read_results(args.results), args.output)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(1, f"Could not plot GEMM results: {exc}\n")
