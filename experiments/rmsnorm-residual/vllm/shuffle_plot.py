"""Render the verified three-way H100 latency comparison."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import run
from shuffle_report import BACKENDS, protocol, read_results


LABELS = ("vLLM 0.10.0", "Triton · grouped reduction", "Triton · ordered warp shuffles")
COLORS = ("#A84F2D", "#346675", "#55427F")
MARKERS = ("o", "s", "^")


def draw(data, output, *, backends=BACKENDS, labels=LABELS, seeds=protocol.TIMING_SEEDS,
         subtitle="1,024 token rows × 4,096 features  ·  BF16  ·  In-place"):
    fig, ax = plt.subplots(figsize=(9.4, 4.5))
    fig.subplots_adjust(left=.31, right=.95, top=.65, bottom=.24)
    ink, secondary = "#152733", "#50636D"
    fig.text(.054, .885, "Bit-matched residual addition and RMSNorm on H100", fontsize=16, weight="bold", color=ink)
    fig.text(.054, .816, subtitle, fontsize=10.5, color=secondary)
    for offset, seed, color, marker in zip((-.17, 0, .17), seeds, COLORS, MARKERS):
        for row, backend in enumerate(backends):
            stats = data[seed]["timing"]["statistics"][backend]
            median = stats["median_us"]
            ax.errorbar(median, row + offset, xerr=[[median-stats["q1_us"]], [stats["q3_us"]-median]],
                        lw=1.15, linestyle="none", capsize=3, color=color, marker=marker, ms=5)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(2.55, -.55)
    ax.set_xlabel("GPU latency per call (µs)", labelpad=10, color=ink)
    ax.xaxis.grid(color="#DFE7E9", linewidth=.8)
    ax.tick_params(axis="both", length=0, pad=7, colors=ink)
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.spines["bottom"].set_color("#BCCBCD")
    fig.legend([Line2D([], [], ls="", marker=m, color=c) for c, m in zip(COLORS, MARKERS)],
               [f"Seed {seed}" for seed in seeds], loc="upper left", bbox_to_anchor=(.047, .77),
               ncol=3, frameon=False, fontsize=10)
    fig.text(.054, .072, "Points show the median of 21 CUDA graph replays. Whiskers show the interquartile range.",
             fontsize=9.3, color=secondary)
    fig.savefig(output, dpi=190, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=run.HERE / "shuffle-latency.png")
    args = parser.parse_args()
    draw(read_results(run.HERE / "shuffle-results"), args.output)
