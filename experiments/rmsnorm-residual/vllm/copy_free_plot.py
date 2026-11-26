"""Plot verified fusion throughput and directly observed NVML power."""
import argparse
from pathlib import Path

from matplotlib.lines import Line2D

from copy_free_report import admissible, read_results
import copy_free_run as protocol
import run
from shuffle_plot import COLORS, MARKERS, plt


def draw(data, output, *, workload=protocol.DEFAULT):
    fig, (rate, power) = plt.subplots(1, 2, figsize=(10.6, 4.8), gridspec_kw={"width_ratios": [1.3, 1]})
    fig.subplots_adjust(left=.225, right=.945, top=.69, bottom=.22, wspace=.27)
    ink, muted = "#152733", "#50636D"
    fig.text(.051, .905, "Copy-free residual RMSNorm on H100", fontsize=17, color=ink, weight="bold")
    subtitle = f"8 CUDA streams  ·  {workload.rows} token rows × 4,096 features  ·  BF16"
    fig.text(.051, .832, subtitle, fontsize=10.5, color=muted)
    handles, labels = [], []
    for seed, offset, color, marker in zip(workload.timing_seeds, (-.18, 0, .18), COLORS, MARKERS):
        timing = data.get(seed, {}).get("timing")
        if not admissible(timing):
            continue
        handles.append(Line2D([], [], ls="", marker=marker, color=color))
        labels.append(f"Seed {seed}")
        for row, backend in enumerate(protocol.BACKENDS):
            center = timing["median_rows_per_second"][backend] / 1e6
            rates = [b["normalized_rows"] / b["wall_seconds"] / 1e6
                     for b in timing["blocks"] if b["backend"] == backend]
            rate.errorbar(center, row + offset, xerr=[[center - min(rates)], [max(rates) - center]], ls="",
                          lw=1.1, capsize=3, color=color, marker=marker, ms=5)
            observed = timing["telemetry"]["by_backend"][backend]["metrics"]["power_w"]["median"]
            if observed is not None:
                power.plot(observed, row + offset, ls="", marker=marker, color=color, ms=5)
    rate.set_yticks(range(3), ("vLLM + copies", "Triton add → RMSNorm", "Triton fused output"))
    power.set_yticks([])
    rate.set_xlabel("Throughput (million rows/s)", labelpad=11, color=ink)
    power.set_xlabel("Median observed power (W)", labelpad=11, color=ink)
    for axis in (rate, power):
        axis.set_ylim(2.58, -.58)
        axis.xaxis.grid(color="#DFE7E9", linewidth=.8)
        axis.tick_params(axis="both", length=0, pad=7, colors=ink)
        axis.spines[["left", "right", "top"]].set_visible(False)
        axis.spines["bottom"].set_color("#BCCBCD")
    if handles:
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.044, .793), ncol=len(labels),
                   frameon=False, fontsize=9.8)
    fig.text(.051, .058, "Whiskers span three timed periods. NVML power samples describe the full device.",
             fontsize=9.35, color=muted)
    fig.savefig(output, dpi=190, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--rows", type=int, choices=protocol.WORKLOADS, default=protocol.ROWS)
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    workload = protocol.WORKLOADS[args.rows]
    prefix = "copy-free" if workload == protocol.DEFAULT else f"copy-free-{workload.rows}"
    output = args.output or run.HERE / f"{prefix}-throughput.png"
    draw(read_results(workload=workload), output, workload=workload)
