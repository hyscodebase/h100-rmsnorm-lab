"""Plot the completed H100 observer comparison. Requires Matplotlib."""

import argparse
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter

import analyze


HERE = Path(__file__).resolve().parent
INK, MUTED, GRID = "#182D37", "#50646F", "#DCE4E7"
CLI, NVML = "#A24615", "#116984"


def plot(results, output):
    records = analyze.load(results)
    decision = analyze.decision(records)
    if decision is None:
        raise ValueError(f"Final figure requires all 18 validated checkpoints; found {len(records)}.")
    passes, ratios, _, _ = decision
    seeds = list(analyze.ORDER)
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(12, 5.4),
                             gridspec_kw={"width_ratios": [1.2, 1.05, 1]})
    try:
        fig.subplots_adjust(left=0.087, right=0.97, top=0.67, bottom=0.25, wspace=0.23)
        fig.text(0.045, 0.91, "GPU activity sampling during H100 BF16 GEMM",
                 fontsize=20, weight="semibold", color=INK)
        fig.text(0.045, 0.85, "M × K × N = 8,192 × 4,096 × 11,008  ·  Six input seeds  ·  45 scheduled polls per reader and run",
                 fontsize=11, color=MUTED)
        gap_ax, ratio_ax, pair_ax = axes
        for ax, title, summary, threshold in zip(axes,
                ("Largest activity gap", "Standalone throughput", "Simultaneous reads"),
                (f"Coverage passed   CLI {passes['A']}/6   ·   NVML {passes['B']}/6",
                 f"Median {statistics.median(ratios):.4f}×   ·   Threshold 0.98×", "Same slot, midpoints ≤ 0.5 s apart"),
                (5, .98, 20)):
            ax.set_title(title, fontsize=12.5, weight="semibold", loc="left", pad=26, color=INK)
            ax.text(0, 1.053, summary, transform=ax.transAxes, fontsize=9.5, color=MUTED)
            ax.axvline(threshold, color=MUTED, ls=(0, (4, 3)), lw=1.2)
            ax.grid(axis="y", color=GRID, lw=.8)
            ax.set_axisbelow(True)
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.spines["bottom"].set_color(GRID)
            ax.tick_params(length=0, pad=5, labelcolor=INK)

        for pos, seed in enumerate(seeds):
            gaps = []
            for arm, reader, marker, color, offset in (
                    ("A", "cli", "o", CLI, -.11), ("B", "nvml", "s", NVML, .11)):
                record = records[seed, arm]
                _, gap, passed = analyze.coverage(record["observers"][reader], record["timing"]["wall_elapsed_s"])
                gaps.append(gap)
                gap_ax.scatter(gap, pos + offset, marker=marker, s=47, edgecolor=color,
                               facecolor=color if passed else "white", linewidth=1.6, zorder=3)
            gap_ax.plot(gaps, [pos - .11, pos + .11], color="#AEBBC1", lw=1.4, zorder=2)
            matched, _, _ = analyze.agreement(records[seed, "C"]["observers"])
            pair_ax.scatter(matched, pos, s=37, color=INK, zorder=3)
            pair_ax.annotate(str(matched), (matched, pos), xytext=(6, 0),
                             textcoords="offset points", va="center", color=INK, fontsize=9)

        ratio_ax.scatter(ratios, range(len(seeds)), s=43, marker="D", color=NVML, zorder=3)
        ratio_min, ratio_max = min(*ratios, .98), max(*ratios, 1.0)
        pad = max(.002, (ratio_max - ratio_min) * .06)
        ratio_ax.set_xlim(ratio_min - pad, ratio_max + pad)
        ratio_ax.xaxis.set_major_locator(MaxNLocator(5))
        ratio_ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        gap_ax.set_xlim(left=0)
        gap_ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
        pair_ax.set_xlim(-2, 51)
        pair_ax.set_xticks([0, 10, 20, 30, 45])
        gap_ax.set_yticks(range(len(seeds)), seeds)
        gap_ax.set_ylim(len(seeds) - .5, -.5)
        gap_ax.set_ylabel("Input seed", color=INK, labelpad=6)
        for ax, label in zip(axes, ("Maximum gap, including window edges (s)", "Wall TFLOP/s ratio (NVML / CLI)",
                                    "Matched slots out of 45")):
            ax.set_xlabel(label, color=INK, labelpad=9)

        handles = [Line2D([], [], ls="", marker=m, color=c, label=l)
                   for m, c, l in (("o", CLI, "CLI (nvidia-smi)"), ("s", NVML, "Direct NVML"))]
        handles.append(Line2D([], [], ls="--", color=MUTED, lw=1.2, label="Registered threshold"))
        fig.legend(handles=handles, loc="center left", bbox_to_anchor=(.036, .094),
                   ncol=3, frameon=False, handlelength=1.6, columnspacing=1.5)
        fig.text(.045, .032, "Hollow markers indicate failed coverage (fewer than 34 samples or a gap over 5 s). "
                 "At least 20 matched slots are required to assess agreement.", color=MUTED, fontsize=9)
        fig.savefig(output, dpi=180, facecolor="white")
    finally:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--output", type=Path, default=HERE / "observer-comparison.png")
    args = parser.parse_args()
    try:
        plot(args.results, args.output)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Could not plot observer results: {exc}\n")
