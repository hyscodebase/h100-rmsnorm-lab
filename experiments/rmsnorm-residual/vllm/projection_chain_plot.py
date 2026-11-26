"""Plot independent BF16 projection-chain throughput and the fusion effect."""
import argparse

from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from projection_chain_report import read_results
import projection_chain_run as protocol
import run
from shuffle_plot import plt


def draw(results, output, *, workload=protocol.SQUARE):
    measured = [(seed, result) for seed, result in sorted(results.items()) if result["effect"]]
    if not measured:
        raise ValueError("No exact timing checkpoints are available to plot.")
    ink, muted, green, slate = "#152733", "#50636D", "#236858", "#526C79"
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.4, 4.6), sharey=True)
    fig.subplots_adjust(left=.12, right=.954, top=.69, bottom=.26, wspace=.30)
    fig.text(.06, .91, "Fused residual RMSNorm between BF16 projections", fontsize=17, color=ink, weight="bold")
    subtitle = "H100  ·  512 rows × 4,096 features  ·  8 CUDA streams"
    if workload != protocol.SQUARE:
        subtitle = f"H100  ·  512 rows  ·  {protocol.WIDTH:,} → {workload.downstream_width:,} downstream  ·  8 streams"
    fig.text(.06, .834, subtitle, fontsize=10.5, color=muted)

    styles = [(protocol.SPLIT, -.12, slate, "o", "Tuned split"), (protocol.FUSED, .12, green, "D", "Fused")]
    for row, (_, result) in enumerate(measured):
        timing = result["timing"]
        for name, offset, color, marker, _ in styles:
            center = timing["median_rows_per_second"][name] / 1e6
            rates = [block["normalized_rows"] / block["wall_seconds"] / 1e6
                     for block in timing["blocks"] if block["backend"] == name]
            left.errorbar(center, row + offset, xerr=[[center - min(rates)], [max(rates) - center]],
                          ls="", lw=1.25, capsize=3, color=color, marker=marker, ms=4.8)
        right.plot(100 * result["effect"]["fused_gain"], row, color=green, marker="D", ms=5.5)

    threshold = 100 * (protocol.THRESHOLD - 1)
    right.axvline(0, color="#95A7AD", linewidth=1)
    right.axvline(threshold, color="#9B4F36", linestyle="--", linewidth=1.2)
    values = [100 * result["effect"]["fused_gain"] for _, result in measured]
    low, high = min(0, *values), max(threshold, *values)
    padding = max(.15, (high - low) * .12)
    right.set_xlim(low - padding, high + padding)
    right.xaxis.set_major_formatter(PercentFormatter(xmax=100))
    right.tick_params(axis="y", labelleft=False)
    left.set_yticks(range(len(measured)), [f"Seed {seed}" for seed, _ in measured])
    left.set_ylim(len(measured) - .43, -.43)
    left.set_title("Full-chain throughput", loc="left", color=ink, fontsize=11)
    right.set_title("Fusion gain over tuned split", loc="left", color=ink, fontsize=11)

    for axis in (left, right):
        axis.xaxis.grid(color="#DFE7E9", linewidth=.8)
        axis.tick_params(axis="both", length=0, pad=6, colors=ink)
        axis.spines[["left", "right", "top"]].set_visible(False)
        axis.spines["bottom"].set_color("#BCCBCD")
    left.set_xlabel("Throughput (million rows/s)", labelpad=10, color=ink)
    right.set_xlabel("Throughput change", labelpad=10, color=ink)
    handles = [Line2D([], [], ls="", marker=marker, color=color, label=label) for _, _, color, marker, label in styles]
    handles.append(Line2D([], [], ls="--", color="#9B4F36", label="Registered 1% threshold"))
    fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(.957, .031), ncol=3, frameon=False, fontsize=9)
    fig.text(.06, .058, "Points show period medians. Whiskers span the measured periods.", color=muted, fontsize=9)
    fig.savefig(output, dpi=190, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--downstream-width", type=int, choices=protocol.WORKLOADS, default=protocol.WIDTH)
    workload = protocol.WORKLOADS[cli.parse_args().downstream_width]
    draw(read_results(workload=workload), run.HERE / f"{workload.archive_name}-throughput.png", workload=workload)
