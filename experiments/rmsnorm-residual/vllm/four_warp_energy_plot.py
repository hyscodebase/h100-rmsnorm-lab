"""Plot the fixed-work energy saving and throughput change on H100."""
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from four_warp_energy_report import read_results
import four_warp_energy_run as protocol
import run
from shuffle_plot import plt


def draw(records, output):
    eligible = [(seed, record["effect"]) for seed, record in sorted(records.items()) if record["effect"]]
    if not eligible:
        raise ValueError("No exact energy comparisons are available to plot.")
    ink, muted = "#152733", "#50636D"
    fig, (energy, throughput) = plt.subplots(1, 2, figsize=(10.9, 4.45), sharey=True)
    fig.subplots_adjust(left=.124, right=.949, top=.69, bottom=.255, wspace=.31)
    fig.text(.06, .918, "Four vs. eight physical warps on H100", fontsize=17, color=ink, weight="bold")
    fig.text(.06, .842, "512 rows × 4,096 features  ·  BF16  ·  equal work per period", fontsize=10.5, color=muted)
    for axis, key, target, color, xlabel, title in (
        (energy, "energy_saving", 100 * (1 - protocol.ENERGY_RATIO), "#236858", "Energy saved", "Gross-window energy per row"),
        (throughput, "throughput_change", 100 * (protocol.THROUGHPUT_RATIO - 1), "#355F78",
         "Throughput change", "Wall-clock throughput"),
    ):
        values = [100 * effect[key] for _, effect in eligible]
        lower, upper = min(0, target, *values), max(0, target, *values)
        padding = max(.08, (upper - lower) * .16)
        axis.set_xlim(lower - padding, upper + padding)
        axis.axvline(0, color="#A9B6B9", lw=1.0)
        axis.axvline(target, color="#9B4F36", ls="--", lw=1.2)
        axis.plot(values, range(len(eligible)), ls="", marker="D", color=color, ms=6)
        axis.set_title(title, loc="left", color=ink, fontsize=11)
        axis.set_xlabel(xlabel, labelpad=11, color=ink)
        axis.xaxis.set_major_formatter(PercentFormatter(xmax=100))
        axis.xaxis.grid(color="#DFE7E9", lw=.8)
        axis.tick_params(axis="both", length=0, pad=6, colors=ink)
        axis.spines[["left", "right", "top"]].set_visible(False)
        axis.spines["bottom"].set_color("#BCCBCD")
    energy.set_yticks(range(len(eligible)), [f"Seed {seed}" for seed, _ in eligible])
    energy.set_ylim(len(eligible) - .46, -.46)
    fig.text(.06, .057, "Each point compares two fixed-work periods per implementation.", color=muted, fontsize=9.1)
    legend = Line2D([], [], ls="--", color="#9B4F36", label="Registered threshold")
    fig.legend(handles=[legend], loc="lower right", bbox_to_anchor=(.947, .034), frameon=False, fontsize=9)
    fig.savefig(output, dpi=190, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    draw(read_results(), run.HERE / "four-warp-energy-512.png")
