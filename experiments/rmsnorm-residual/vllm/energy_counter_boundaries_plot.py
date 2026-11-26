"""Plot counter boundary delays and eligible fixed-work energy effects."""
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter, StrMethodFormatter

from energy_counter_boundaries_report import read_results
import energy_counter_boundaries_run as protocol
import run
from shuffle_plot import plt


def draw(records, output):
    measured = [(seed, record) for seed, record in sorted(records.items()) if record["timing"]]
    if not measured:
        raise ValueError("No attempted timing pairs are available to plot.")
    ink, muted, accepted, rejected = "#152733", "#50636D", "#236858", "#9B4F36"
    fig, (boundary, energy) = plt.subplots(1, 2, figsize=(11.4, 4.5), sharey=True)
    fig.subplots_adjust(left=.115, right=.954, top=.69, bottom=.255, wspace=.32)
    fig.text(.06, .911, "Checking H100 energy at short counter boundaries", fontsize=17, color=ink, weight="bold")
    fig.text(.06, .835, "512 rows × 4,096 features  ·  BF16  ·  equal work per period", fontsize=10.5, color=muted)

    threshold = 100 * (1 - protocol.energy.ENERGY_RATIO)
    savings = [100 * d["effect"]["energy_saving"] for _, d in measured if d["effect"]]
    low, high = min([0, *savings]), max([threshold, *savings])
    margin = max(.1, (high - low) * .13)
    energy.set_xlim(low - margin, high + margin)
    energy.axvline(0, color="#9EACB1", lw=1)
    energy.axvline(threshold, color=rejected, ls="--", lw=1.2)
    boundary.set_xscale("symlog", linthresh=1)
    boundary.axvline(1000 * protocol.MAX_PADDING_S, color=rejected, ls="--", lw=1.2)
    for row, (_, record) in enumerate(measured):
        pairs = record["timing"]["pairs"]
        for index, pair in enumerate(pairs):
            color = accepted if pair["eligible"] else rejected
            for position, block in enumerate(pair["blocks"]):
                offset = .11 * (2 * index + position - (2 * len(pairs) - 1) / 2)
                boundary.plot(1000 * protocol.energy.boundary_padding_seconds(block), row + offset,
                              marker="o", color=color, ms=4.6)
        if effect := record["effect"]:
            energy.plot(100 * effect["energy_saving"], row, marker="D", color=accepted, ms=6)
        else:
            count = len(record["timing"]["accepted_pair_indices"])
            energy.text(.19, row, f"{count}/2 eligible pairs", transform=energy.get_yaxis_transform(),
                        color=muted, va="center", fontsize=9.5)

    boundary.set_yticks(range(len(measured)), [f"Seed {seed}" for seed, _ in measured])
    boundary.set_ylim(len(measured) - .45, -.45)
    boundary.set_title("All attempted periods", loc="left", color=ink, fontsize=11)
    energy.set_title("Energy saving from eligible pairs", loc="left", color=ink, fontsize=11)
    boundary.set_xlabel("Combined query padding (ms)", labelpad=10, color=ink)
    low_ms, high_ms = boundary.get_xlim()
    boundary.set_xticks([ms for ms in (0, 1, 5, 10, 50, 100, 500, 1000, 5000) if low_ms <= ms <= high_ms])
    boundary.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    energy.set_xlabel("Energy saved per row", labelpad=10, color=ink)
    energy.xaxis.set_major_formatter(PercentFormatter(xmax=100))
    for axis in (boundary, energy):
        axis.xaxis.grid(color="#DFE7E9", linewidth=.8)
        axis.tick_params(axis="both", length=0, pad=6, colors=ink)
        axis.spines[["left", "right", "top"]].set_visible(False)
        axis.spines["bottom"].set_color("#BCCBCD")
    fig.text(.06, .059, "Dashed lines mark 50 ms and 1.5%.", color=muted, fontsize=9.2)
    legend = [Line2D([], [], ls="", marker="o", color=accepted, label="Eligible pair"),
              Line2D([], [], ls="", marker="o", color=rejected, label="Rejected pair")]
    fig.legend(handles=legend, loc="lower right", bbox_to_anchor=(.948, .035), ncol=2, frameon=False, fontsize=9)
    fig.savefig(output, dpi=190, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    draw(read_results(), run.HERE / "energy-counter-boundaries-512.png")
