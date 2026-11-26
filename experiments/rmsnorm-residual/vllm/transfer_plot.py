"""Plot the validated latency reduction by token row count."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import row_transfer as protocol
import run
import shuffle_report
from transfer_report import read_results, timing_medians


def draw(results, baseline, output):
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    fig.subplots_adjust(left=.12, right=.96, top=.72, bottom=.22)
    ink, secondary, grid = "#152733", "#50636D", "#DFE7E9"
    colors, markers, offsets = ("#A84F2D", "#346675", "#55427F"), ("o", "s", "^"), (-.16, 0, .16)
    fig.text(.058, .9, "Residual RMSNorm speed across token row counts", weight="bold", fontsize=16, color=ink)
    fig.text(.058, .825, "NVIDIA H100  ·  4,096 features  ·  BF16  ·  Released vLLM 0.10.0 reference",
             fontsize=10.5, color=secondary)
    shapes = sorted((*protocol.ROWS, run.ROWS))
    reductions = []
    ax.axhspan(5, 100, color="#F2F7F6", zorder=0)
    ax.axhline(5, c="#628B83", lw=1.2, ls="--")
    ax.axhline(0, c="#81969A", lw=.8)
    for idx, rows in enumerate(shapes):
        if rows == run.ROWS:
            records = (baseline[seed] for seed in protocol.shuffle.TIMING_SEEDS)
        else:
            records = (results.get((rows, seed)) for seed in protocol.TIMING_SEEDS)
        row_values = []
        for record, color, marker, offset in zip(records, colors, markers, offsets):
            if record and record["timing"]:
                if rows == run.ROWS:
                    times = shuffle_report.timing_medians(record)
                    vllm, triton = times[protocol.shuffle.REFERENCE], times[protocol.shuffle.CANDIDATE]
                else:
                    vllm, triton = timing_medians(record)
                reduction = 100 * (1 - triton / vllm)
                row_values.append(reduction)
                ax.plot(idx + offset, reduction, ls="", marker=marker, color=color, ms=6, zorder=3)
        if not row_values:
            ax.axvspan(idx-.31, idx+.31, color="#F8EDE8", zorder=0)
            ax.text(idx, .5, "Excluded on\naccuracy", ha="center", va="center",
                    transform=ax.get_xaxis_transform(), fontsize=9.5, color=secondary, zorder=4)
        elif max(row_values) < 5:
            label = f"{min(row_values):.3f}–{max(row_values):.3f}%"
            ax.text(idx, min(row_values) - 1.15, label, ha="center", va="top", fontsize=9, color=secondary)
        reductions.extend(row_values)
    ax.set_ylim(min(0, *reductions) - 2, max(5, *reductions) + 3)
    ax.set_xlim(-.55, len(shapes)-.43)
    ax.set_xticks(range(len(shapes)), [f"{rows:,}" for rows in shapes])
    ax.set_xlabel("Token rows", labelpad=10, color=ink)
    ax.set_ylabel("Latency reduction (%)", labelpad=10, color=ink)
    ax.yaxis.grid(color=grid, linewidth=.8)
    ax.tick_params(axis="both", length=0, pad=7, colors=ink)
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.spines["bottom"].set_color("#BCCBCD")
    handles = [Line2D([], [], ls="", marker=m, color=c) for m, c in zip(markers, colors)]
    handles.append(Line2D([], [], ls="--", color="#628B83"))
    fig.legend(handles, ["Timing seed 1", "Timing seed 2", "Timing seed 3", "5% criterion"],
               loc="upper left", bbox_to_anchor=(.047, .79), ncol=4, frameon=False, fontsize=9.6)
    fig.text(.058, .066, "Each point uses the per-backend median of 21 CUDA graph replays. "
             "The 1,024-row results come from the prior study.", fontsize=9.25, color=secondary)
    fig.savefig(output, dpi=190, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=run.HERE / "transfer-latency.png")
    args = parser.parse_args()
    draw(read_results(run.HERE / "transfer-results"), shuffle_report.read_results(run.HERE / "shuffle-results"),
         args.output)
