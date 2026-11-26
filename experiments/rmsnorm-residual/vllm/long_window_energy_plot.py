"""Plot measured padding and conditional energy savings for a verified archive."""
import argparse
from pathlib import Path

from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter, StrMethodFormatter

import long_window_energy_report as report
import long_window_energy_run as protocol
from shuffle_plot import plt


def draw(records, output):
    summary = report.summarize(records)["seeds"]
    seeds = [seed for seed, row in summary.items() if row["saved_periods"]]
    if not seeds:
        raise ValueError("A plot requires at least one saved timing seed.")
    ink, gray, green, blue, red = "#18303A", "#586B72", "#287461", "#3E617F", "#A34B34"
    fig, (padding, saving) = plt.subplots(1, 2, sharey=True, figsize=(11.6, 4.45))
    fig.subplots_adjust(left=.107, right=.952, top=.71, bottom=.236, wspace=.29)
    fig.text(.06, .916, "H100 energy savings under measured boundary padding", fontsize=17, color=ink, weight="bold")
    fig.text(.06, .84, "512 rows × 4,096 features  ·  BF16  ·  4,096 bursts per period", fontsize=10.5, color=gray)
    padding.set_xscale("symlog", linthresh=1)
    saving.axvline(0, linewidth=1, color="#AAB8BE")
    threshold = 100 * (1 - protocol.energy.ENERGY_RATIO)
    saving.axvline(threshold, linewidth=1.1, color=red, linestyle="--")
    for row, seed in enumerate(seeds):
        seed_windows = [record["window"] for record in records.values() if str(record["seed"]) == seed]
        for index, window in enumerate(seed_windows):
            if window is None or (padding_ns := window["sample_bounds"]["padding_upper_ns"]) is None:
                continue
            y = row + .12 * (index - (len(seed_windows) - 1) / 2)
            color = green if window["sample_bounds"]["status"] == "qualified" else red
            padding.plot(padding_ns / 1_000_000, y, "o", color=color, ms=4.6)
        if (comparison := summary[seed]["comparison"]) is None:
            accepted = len(summary[seed]["eligible_pair_indices"])
            saving.text(.18, row, f"{accepted}/2 eligible pairs", transform=saving.get_yaxis_transform(),
                        va="center", color=gray, fontsize=9)
            continue
        for wattage, color, shift in ((700, green, -.12), (1000, blue, .12)):
            bounds = comparison["padding_power_sensitivity"][str(wattage)]
            lower, upper = bounds["saving_lower"], bounds["saving_upper"]
            if lower is not None:
                saving.plot([100 * lower, 100 * upper], [row + shift, row + shift], color=color,
                            marker="|", lw=1.5, ms=6)
            else:
                saving.text(.18, row + shift, f"{wattage:,} W lower bound unresolved", color=color, fontsize=8.5,
                            transform=saving.get_yaxis_transform(), va="center")
        saving.plot(100 * comparison["gross_energy_saving"], row, "D", ms=5, color=ink)

    padding.set_yticks(range(len(seeds)), [f"Seed {seed}" for seed in seeds])
    padding.set_ylim(len(seeds) - .45, -.55)
    low_ms, high_ms = padding.get_xlim()
    ticks = [ms for ms in (0, 1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000)
             if low_ms <= ms <= high_ms]
    if len(ticks) >= 2:
        padding.set_xticks(ticks)
    padding.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    saving.xaxis.set_major_locator(MaxNLocator(nbins=6))
    saving.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=1))
    for axis, title, label in ((padding, "Measured periods", "Upper bound on combined padding (ms)"),
                               (saving, "Energy savings by seed", "Energy saved per row")):
        axis.set_title(title, loc="left", color=ink, fontsize=11)
        axis.set_xlabel(label, labelpad=10, color=ink)
        axis.xaxis.grid(color="#DFE7E9", linewidth=.8)
        axis.spines[["left", "right", "top"]].set_visible(False)
        axis.spines["bottom"].set_color("#BCCBCD")
        axis.tick_params(axis="both", length=0, pad=6, colors=ink)
    handles = [Line2D([], [], ls="", marker="D", color=ink, label="Gross counter delta"),
               Line2D([], [], color=green, marker="|", label="700 W padding scenario"),
               Line2D([], [], color=blue, marker="|", label="1,000 W padding scenario")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.055, .032), ncol=3, frameon=False, fontsize=9)
    fig.text(.95, .063, "Dashed line marks the 1.5% criterion.", ha="right", color=gray, fontsize=9)
    fig.savefig(output, dpi=190, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("directory", type=Path, nargs="?", default=report.RESULTS)
    cli.add_argument("--output", type=Path, default=protocol.run.HERE / "long-window-energy-512.png")
    args = cli.parse_args()
    draw(report.read_verified(args.directory), args.output)
