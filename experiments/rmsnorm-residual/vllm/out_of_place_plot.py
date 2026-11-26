"""Plot verified input-preserving throughput and NVML power."""
import argparse
from pathlib import Path

from matplotlib.lines import Line2D

from out_of_place_report import admissible, read_results
import out_of_place_run as protocol
import run
from shuffle_plot import COLORS, MARKERS, plt


def draw(data, output):
    fig, (rate, power) = plt.subplots(1, 2, figsize=(10.6, 5.1), gridspec_kw={"width_ratios": [1.35, 1]})
    fig.subplots_adjust(left=.25, right=.95, top=.66, bottom=.215, wspace=.24)
    ink, muted = "#152733", "#50636D"
    fig.text(.051, .912, "Input-preserving residual RMSNorm on H100", fontsize=17, color=ink, weight="bold")
    fig.text(.051, .845, "8 CUDA streams  ·  128 token rows × 4,096 features  ·  BF16", fontsize=10.5, color=muted)
    handles, labels = [], []
    for seed, offset, color, marker in zip(protocol.TIMING_SEEDS, (-.18, 0, .18), COLORS, MARKERS):
        result = data.get(seed)
        if not result or not admissible(result["timing"]):
            continue
        timing = result["timing"]
        handles.append(Line2D([], [], ls="", marker=marker, color=color))
        labels.append(f"Seed {seed}")
        for row, name in enumerate(protocol.BACKENDS):
            center = timing["median_rows_per_second"][name] / 1e6
            rates = [b["normalized_rows"] / b["wall_seconds"] / 1e6 for b in timing["blocks"] if b["backend"] == name]
            rate.errorbar(center, row + offset, xerr=[[center-min(rates)], [max(rates)-center]], ls="",
                          lw=1.1, capsize=3, color=color, marker=marker, ms=5)
            observed = timing["telemetry"]["by_backend"][name]["metrics"]["power_w"]["median"]
            if observed is not None:
                power.plot(observed, row + offset, ls="", marker=marker, color=color, ms=5)
    rate.set_yticks(range(4), ("vLLM + copies", "Triton 8 warps + copies", "Triton 16 warps + copies", "Triton direct output"))
    power.set_yticks([])
    rate.set_xlabel("Throughput (million rows/s)", labelpad=11, color=ink)
    power.set_xlabel("Median observed power (W)", labelpad=11, color=ink)
    for axis in (rate, power):
        axis.set_ylim(3.55, -.55)
        axis.xaxis.grid(color="#DFE7E9", linewidth=.8)
        axis.tick_params(axis="both", length=0, pad=7, colors=ink)
        axis.spines[["left", "right", "top"]].set_visible(False)
        axis.spines["bottom"].set_color("#BCCBCD")
    if handles:
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.044, .807), ncol=len(labels),
                   frameon=False, fontsize=9.8)
    fig.text(.051, .056, "Throughput whiskers span four measured blocks. Power samples cover the full device.",
             fontsize=9.35, color=muted)
    fig.savefig(output, dpi=190, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=run.HERE / "out-of-place-throughput.png")
    args = parser.parse_args()
    draw(read_results(run.HERE / "out-of-place-results"), args.output)
