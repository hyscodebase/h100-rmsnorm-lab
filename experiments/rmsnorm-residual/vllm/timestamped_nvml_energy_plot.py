"""Plot verified call boundaries, field timestamp boundaries and energy effects."""
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter, StrMethodFormatter

import run
from shuffle_plot import plt
from timestamped_nvml_energy_report import read_results
import timestamped_nvml_energy_run as protocol


def draw(records, output):
    measured = [(seed, record) for seed, record in sorted(records.items()) if record["timing"]]
    if not measured:
        raise ValueError("No timing windows are available for this comparison.")
    ink, muted, accepted, rejected = "#152733", "#52646D", "#216856", "#A54E34"
    fig, (boundaries, energy) = plt.subplots(1, 2, figsize=(11.5, 4.45), sharey=True)
    fig.subplots_adjust(left=.11, right=.946, top=.704, bottom=.23, wspace=.29)
    fig.text(.06, .918, "Where H100 energy measurement boundaries fall", fontsize=17, color=ink, weight="bold")
    fig.text(.06, .84, "512 rows × 4,096 features  ·  BF16  ·  equal work", fontsize=10.5, color=muted)

    cutoff = protocol.MAX_PADDING_NS / 1_000_000
    threshold = 100 * (1 - protocol.energy.ENERGY_RATIO)
    savings = [100 * r["effect"]["energy_saving"] for _, r in measured if r["effect"]]
    low, high = min([0, *savings]), max([threshold, *savings])
    margin = max(.1, (high - low) * .13)
    energy.set_xlim(low - margin, high + margin)
    energy.axvline(0, color="#ACB9BD", lw=1)
    energy.axvline(threshold, color=rejected, linestyle="--", lw=1.15)
    boundaries.set_xscale("symlog", linthresh=1)
    boundaries.axvline(cutoff, color=rejected, linestyle="--", lw=1.15)

    for row, (_, record) in enumerate(measured):
        timing = record["timing"]
        windows = [(pair["eligible"], block) for pair in timing["pairs"] for block in pair["blocks"]]
        for index, (eligible, window) in enumerate(windows):
            y = row + .115 * (index - (len(windows) - 1) / 2)
            padding = window["sample_bounds"]["padding_upper_ns"]
            call_ms = 1000 * protocol.energy.boundary_padding_seconds(window)
            field_ms = None if padding is None else padding / 1_000_000
            color = accepted if eligible else rejected
            if field_ms is not None:
                boundaries.plot([field_ms, call_ms], [y, y], color=color, alpha=.55, lw=.9)
                boundaries.plot(field_ms, y, marker="o", color=color, ms=4)
            boundaries.plot(call_ms, y, marker="x" if field_ms is None else "o", mfc="white",
                            color=color, ms=4.7, linestyle="")
        if effect := record["effect"]:
            energy.plot(100 * effect["energy_saving"], row, marker="D", color=ink, ms=6)
        else:
            count = len(timing["accepted_pair_indices"])
            energy.text(.18, row, f"{count}/2 eligible pairs", transform=energy.get_yaxis_transform(),
                        va="center", color=muted, fontsize=9.3)

    boundaries.set_yticks(range(len(measured)), [f"Seed {seed}" for seed, _ in measured])
    boundaries.set_ylim(len(measured) - .51, -.51)
    boundaries.set(title="All attempted periods", xlabel="Combined padding (ms)")
    energy.set(title="Energy savings with both orders", xlabel="Energy saved per row")
    limits = boundaries.get_xlim()
    boundaries.set_xticks([ms for ms in (0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000)
                          if limits[0] <= ms <= limits[1]])
    boundaries.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    energy.xaxis.set_major_locator(MaxNLocator(nbins=6))
    energy.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=1))
    for axis in (boundaries, energy):
        axis.title.set_ha("left")
        axis.title.set_position((0, 1))
        axis.title.set_color(ink)
        axis.title.set_fontsize(11)
        axis.xaxis.grid(color="#DFE7E9", linewidth=.8)
        axis.tick_params(axis="both", length=0, pad=6, colors=ink)
        axis.spines[["left", "right", "top"]].set_visible(False)
        axis.spines["bottom"].set_color("#BCCBCD")
    marks = [Line2D([], [], ls="", marker="o", color=ink, mfc="white", label="API call bounds"),
             Line2D([], [], ls="", marker="o", color=ink, label="Reported field bounds"),
             Line2D([], [], ls="", marker="o", color=accepted, label="Eligible pair")]
    if any(not pair["eligible"] for _, r in measured for pair in r["timing"]["pairs"]):
        marks.append(Line2D([], [], ls="", marker="o", color=rejected, label="Rejected pair"))
    if any(b["sample_bounds"]["padding_upper_ns"] is None
           for _, r in measured for p in r["timing"]["pairs"] for b in p["blocks"]):
        marks.append(Line2D([], [], ls="", marker="x", color=rejected, label="Field bounds unresolved"))
    fig.legend(handles=marks, loc="lower left", bbox_to_anchor=(.056, .035), ncol=len(marks),
               frameon=False, fontsize=9, columnspacing=1.2)
    fig.text(.946, .065, "Dashed lines mark 50 ms and 1.5%.", ha="right", color=muted, fontsize=9)
    fig.savefig(output, dpi=190, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    draw(read_results(), run.HERE / "timestamped-nvml-energy-512.png")
