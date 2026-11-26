"""Plot tuning-seed tile rates and independent held-out throughput."""
from matplotlib.lines import Line2D

import run
from shuffle_plot import COLORS, MARKERS, plt
from split_add_tuning_report import read_results
import split_add_tuning_run as protocol


def draw_point(axis, timing, backend, y, *, color, marker):
    center = timing["median_rows_per_second"][backend] / 1e6
    rates = [block["normalized_rows"] / block["wall_seconds"] / 1e6
             for block in timing["blocks"] if block["backend"] == backend]
    axis.errorbar(center, y, xerr=[[center - min(rates)], [max(rates) - center]], ls="",
                  lw=1.15, capsize=3, color=color, marker=marker, ms=5)


def draw(data, output):
    tuning = data[protocol.NUMERIC_SEED]
    selected = tuning["selected_tile"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.4, 4.75))
    fig.subplots_adjust(left=.145, right=.957, top=.67, bottom=.236, wspace=.58)
    ink, muted = "#152733", "#50636D"
    fig.text(.06, .912, "Choosing the residual-add tile on H100", fontsize=17, color=ink, weight="bold")
    fig.text(.06, .839, "512 rows × 4,096 features  ·  BF16  ·  8 CUDA streams", fontsize=10.5, color=muted)

    for index, tile in enumerate(protocol.TILES):
        name = protocol.TILE_BACKENDS[tile]
        if name in tuning["timing"]["median_rows_per_second"]:
            draw_point(left, tuning["timing"], name, index,
                       color="#236858" if tile == selected else "#647780", marker="D" if tile == selected else "o")
    left.set_yticks(range(len(protocol.TILES)),
                    [f"{tile:,}" + (" selected" if tile == selected else "") for tile in protocol.TILES])
    left.set_ylim(len(protocol.TILES) - .5, -.5)
    left.set_title("Selection seed 173", loc="left", color=ink, fontsize=11)

    measured = protocol.held_out_backends(selected)
    names = {protocol.ORIGINAL: "Original split", protocol.FUSED: "Fused direct output"}
    if selected != protocol.BASE_TILE:
        names[protocol.TILE_BACKENDS[selected]] = "Selected split"
    handles, labels = [], []
    for seed, offset, color, marker in zip(protocol.TIMING_SEEDS, (-.17, 0, .17), COLORS, MARKERS):
        timing = data.get(seed, {}).get("timing")
        if not protocol.exact_timing(timing):
            continue
        handles.append(Line2D([], [], ls="", marker=marker, color=color))
        labels.append(str(seed))
        for index, name in enumerate(measured):
            draw_point(right, timing, name, index + offset, color=color, marker=marker)
    right.set_yticks(range(len(measured)), [names[name] for name in measured])
    right.set_ylim(len(measured) - .5, -.5)
    right.set_title("Independent seeds", loc="left", color=ink, fontsize=11)
    if handles:
        right.legend(handles, labels, loc="lower right", frameon=False, ncol=len(handles), fontsize=8.4,
                     bbox_to_anchor=(1, 1.06))

    for axis in (left, right):
        axis.set_xlabel("Throughput (million rows/s)", labelpad=10, color=ink)
        axis.xaxis.grid(color="#DFE7E9", linewidth=.8)
        axis.tick_params(axis="both", length=0, pad=6, colors=ink)
        axis.spines[["left", "right", "top"]].set_visible(False)
        axis.spines["bottom"].set_color("#BCCBCD")
    fig.text(.06, .055, "Points show period medians. Whiskers span the measured periods.", color=muted, fontsize=9.1)
    fig.savefig(output, dpi=190, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    draw(read_results(), run.HERE / "split-add-tiles-512-throughput.png")
