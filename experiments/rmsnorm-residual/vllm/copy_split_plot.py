"""Plot the operator rate and restoration share from verified CUDA events."""
import argparse
from pathlib import Path

from matplotlib.lines import Line2D

from copy_split_report import admissible, read_results
import copy_split_run as protocol
import run
from shuffle_plot import COLORS, MARKERS, plt


def draw(data, output):
    fig, (rate, share) = plt.subplots(1, 2, figsize=(10.5, 4.8), gridspec_kw={"width_ratios": [1.35, 1]})
    fig.subplots_adjust(left=.225, right=.95, top=.66, bottom=.225, wspace=.23)
    ink, muted = "#152733", "#50636D"
    fig.text(.052, .91, "Restoration and residual RMSNorm on H100", weight="bold", fontsize=17, color=ink)
    fig.text(.052, .84, "8 CUDA streams  ·  128 token rows × 4,096 features  ·  BF16", fontsize=10.5, color=muted)
    legend, labels = [], []
    for seed, offset, color, marker in zip(protocol.TIMING_SEEDS, (-.18, 0, .18), COLORS, MARKERS):
        result = data.get(seed)
        if not result or not admissible(result["timing"]):
            continue
        timing = result["timing"]
        legend.append(Line2D([], [], ls="", marker=marker, color=color))
        labels.append(f"Seed {seed}")
        for row, name in enumerate(protocol.BACKENDS):
            split = timing["rates"][f"split:{name}"]
            center = split["median_operator_rows_per_second"] / 1e6
            all_rates = [1000 * b["normalized_rows"] / b["operator_ms"] / 1e6
                         for b in timing["blocks"] if b["condition"] == f"split:{name}"]
            rate.errorbar(center, row + offset, xerr=[[center-min(all_rates)], [max(all_rates)-center]],
                          linestyle="none", lw=1.15, capsize=3, color=color, marker=marker, ms=5)
            share.plot(100 * split["restoration_fraction"], row + offset, ls="", marker=marker, color=color, ms=5)
    rate.set_yticks(range(3), ("vLLM 0.10.0", "Triton · 8 warps", "Triton · 16 warps"))
    share.set_yticks([])
    rate.set_xlabel("Operator stage (million rows/s)", labelpad=10, color=ink)
    share.set_xlabel("Time in restoration stage (%)", labelpad=10, color=ink)
    for axis in (rate, share):
        axis.set_ylim(2.55, -.55)
        axis.xaxis.grid(color="#DFE7E9", linewidth=.8)
        axis.tick_params(axis="both", length=0, pad=7, colors=ink)
        axis.spines[["left", "right", "top"]].set_visible(False)
        axis.spines["bottom"].set_color("#BCCBCD")
    if legend:
        fig.legend(legend, labels, loc="upper left", bbox_to_anchor=(.045, .805), ncol=len(labels),
                   frameon=False, fontsize=9.8)
    fig.text(.052, .052, "CUDA events on the coordinating stream bound both stages. Whiskers span three blocks.",
             fontsize=9.35, color=muted)
    fig.savefig(output, dpi=190, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=run.HERE / "copy-split.png")
    args = parser.parse_args()
    draw(read_results(run.HERE / "copy-split-results"), args.output)
