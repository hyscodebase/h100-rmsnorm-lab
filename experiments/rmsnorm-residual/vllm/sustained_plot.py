"""Render throughput and device activity across the three independent H100 runs."""
import argparse
from pathlib import Path

from matplotlib.lines import Line2D

import run
from shuffle_plot import COLORS, MARKERS, plt
from sustained_report import admissible, read_results
import sustained_run as protocol


def draw(data, output):
    fig, (rate_axis, load_axis) = plt.subplots(1, 2, figsize=(10.5, 4.85), gridspec_kw={"width_ratios": [1.4, 1]})
    fig.subplots_adjust(left=.235, right=.95, top=.65, bottom=.22, wspace=.18)
    ink, secondary = "#152733", "#50636D"
    fig.text(.052, .9, "Sustained residual RMSNorm on H100", weight="bold", fontsize=17, color=ink)
    fig.text(.052, .827, "8 CUDA streams  ·  128 token rows × 4,096 features  ·  BF16", fontsize=10.5, color=secondary)
    markers, legend, activity = [], [], []
    for seed, offset, color, marker in zip(protocol.TIMING_SEEDS, (-.18, 0, .18), COLORS, MARKERS):
        timing = data[seed]["timing"]
        if not admissible(timing):
            continue
        markers.append(Line2D([], [], ls="", marker=marker, color=color))
        legend.append(f"Seed {seed}")
        for row, backend in enumerate(protocol.BACKENDS):
            center = timing["median_rows_per_second"][backend] / 1e6
            rates = [block["normalized_rows"] / block["wall_seconds"] / 1e6
                     for block in timing["blocks"] if block["backend"] == backend]
            rate_axis.errorbar(center, row + offset, xerr=[[center-min(rates)], [max(rates)-center]],
                               lw=1.15, linestyle="none", capsize=3, color=color, marker=marker, ms=5)
            load = timing["telemetry"]["by_backend"][backend]
            median = load["metrics"]["gpu_busy_pct"]["median"]
            if median is not None:
                activity.append(median)
                load_axis.plot(median, row + offset, ls="", marker=marker, color=color, ms=5,
                               markerfacecolor=color if load["high_load"] else "white")
    load_axis.axvline(80, ls="--", lw=1, c="#628B83")
    load_axis.set_xlim(max(0, min([80, *activity]) - 5), 102)
    markers.append(Line2D([], [], ls="--", color="#628B83"))
    legend.append("80% activity threshold")
    rate_axis.set_yticks(range(len(protocol.BACKENDS)), ("vLLM 0.10.0", "Triton · 8 warps", "Triton · 16 warps"))
    load_axis.set_yticks([])
    rate_axis.set_xlabel("Throughput (million normalized rows/s)", labelpad=12, color=ink)
    load_axis.set_xlabel("Median GPU activity (%)", labelpad=12, color=ink)
    for axis in (rate_axis, load_axis):
        axis.set_ylim(2.55, -.55)
        axis.xaxis.grid(color="#DFE7E9", linewidth=.8)
        axis.tick_params(axis="both", length=0, pad=7, colors=ink)
        axis.spines[["left", "right", "top"]].set_visible(False)
        axis.spines["bottom"].set_color("#BCCBCD")
    fig.legend(markers, legend, loc="upper left", bbox_to_anchor=(.045, .79), ncol=len(legend), frameon=False, fontsize=9.8)
    fig.text(.052, .047, "Throughput includes buffer copies and scheduling. Whiskers show the range across three blocks.\n"
             "Open activity markers indicate a missed device-load requirement.",
             fontsize=9.3, color=secondary)
    fig.savefig(output, dpi=190, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=run.HERE / "sustained-throughput.png")
    args = parser.parse_args()
    draw(read_results(run.HERE / "sustained-results"), args.output)
