"""Plot the precision-matched H100 latency comparison from verified measurements."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from report import HERE, protocol, read_records


ROW_LABELS = {
    protocol.TRITON: "Triton · fused",
    "compiled_materialized_max-autotune-no-cudagraphs": "PyTorch · two stages, autotuned",
    "compiled_materialized_default": "PyTorch · two stages, default",
    "compiled_bitcast_max-autotune-no-cudagraphs": "PyTorch · explicit BF16 bits, autotuned",
    "compiled_bitcast_default": "PyTorch · explicit BF16 bits, default",
}


def draw(records, output):
    seeds = protocol.SEEDS[1:]
    if not all(seed in records and records[seed]["gate_passed"] for seed in seeds):
        raise ValueError("All three eligible timing checkpoints are required for the final figure.")
    shown = [name for name in ROW_LABELS if all(name in records[s]["backends"] and
                                               records[s]["backends"][name]["status"] == "measured" for s in seeds)]
    colors = ("#9E4D22", "#22818C", "#534381")
    shapes = ("o", "s", "^")
    fig, ax = plt.subplots(figsize=(10.2, 5.1))
    fig.subplots_adjust(left=.33, right=.95, top=.63, bottom=.21)
    title, secondary = "#152733", "#50636D"
    fig.text(.05, .9, "Fused residual addition and RMSNorm on H100", color=title, fontsize=18, weight="bold")
    fig.text(.05, .831, "1,024 token rows × 4,096 features  ·  BF16  ·  Two verified outputs",
             color=secondary, fontsize=10.5)
    for offset, seed, color, marker in zip((-.17, 0, .17), seeds, colors, shapes):
        for row, name in enumerate(shown):
            timing = records[seed]["backends"][name]["timing"]
            median = timing["median_us"]
            ax.errorbar(median, row + offset, xerr=[[median-timing["q1_us"]], [timing["q3_us"]-median]],
                        lw=1.2, linestyle="none", capsize=3, color=color, marker=marker, ms=5)
    ax.set_yticks(range(len(shown)), [ROW_LABELS[name] for name in shown])
    ax.invert_yaxis()
    ax.set_ylim(len(shown)-.55, -.55)
    ax.set_xlabel("GPU latency per call (µs)", labelpad=10, color=title)
    ax.xaxis.grid(color="#DFE7E9", linewidth=.8)
    ax.tick_params(axis="both", length=0, pad=6, colors=title)
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.spines["bottom"].set_color("#B7C4C9")
    fig.legend([Line2D([], [], ls="", marker=marker, color=color) for color, marker in zip(colors, shapes)],
               [f"Seed {seed}" for seed in seeds], loc="upper left", bbox_to_anchor=(.042, .795),
               ncol=len(seeds), frameon=False, fontsize=10)
    fig.text(.05, .062, "Points show the median of 21 replays. Whiskers span the interquartile range.",
             color=secondary, fontsize=9.5)
    fig.savefig(output, dpi=190, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "latency.png")
    args = parser.parse_args()
    try:
        draw(read_records(HERE / "results"), args.output)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Plot failed: {exc}\n")
