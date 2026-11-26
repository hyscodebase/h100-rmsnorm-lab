"""Render the checked-in H100 benchmark figure using Matplotlib."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, StrMethodFormatter


RESULTS = Path(__file__).resolve().parent / "results"
WARPS = {"triton_4w": ("4 warps", "#126986"), "triton_8w": ("8 warps", "#BF531D")}


def ratios(path):
    records = json.loads(path.read_text())["results"]
    medians = {(r["width"], r["rows"], r["backend"]): r["median_us"] for r in records}
    return {
        (width, rows, backend): medians[width, rows, "torch_compile"] / latency
        for (width, rows, backend), latency in medians.items()
        if backend in WARPS
    }


def main():
    first = ratios(RESULTS / "h100.json")
    repeat = ratios(RESULTS / "h100-repeat.json")
    if first.keys() != repeat.keys():
        raise ValueError("The two runs must include the same shapes and backends.")
    widths = sorted({width for width, _, _ in first})
    fig, axes = plt.subplots(1, len(widths), sharey=True, figsize=(10, 4.8), squeeze=False)
    fig.subplots_adjust(left=0.095, right=0.98, top=0.68, bottom=0.18, wspace=0.13)
    title_color = "#152733"
    muted = "#52636D"
    fig.text(0.04, 0.94, "BF16 RMSNorm on the NVIDIA H100",
             fontsize=19, fontweight="semibold", color=title_color)
    fig.text(0.04, 0.875, "Median torch.compile latency / median Triton latency. "
             "Values above 1× favor Triton.", fontsize=10.7, color=muted)

    legend = [Line2D([], [], color=color, lw=2.3, label=label)
              for label, color in WARPS.values()]
    for label, fill in (("First run", title_color), ("Repeat", "white")):
        legend.append(Line2D([], [], ls="", marker="o", color=title_color,
                             markerfacecolor=fill, markersize=5.8, label=label))
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(0.034, 0.845),
               ncol=4, frameon=False, handlelength=1.8, columnspacing=2.0)

    for ax, width in zip(axes[0], widths):
        rows = sorted({r for w, r, _ in first if w == width})
        # Adjacent markers keep near-identical results from concealing each other.
        positions = range(len(rows))
        for backend, (_, color) in WARPS.items():
            a = [first[width, r, backend] for r in rows]
            b = [repeat[width, r, backend] for r in rows]
            ax.plot([i - 0.035 for i in positions], a, color=color, marker="o",
                    markersize=5.1, linewidth=1.65, zorder=3)
            ax.scatter([i + 0.035 for i in positions], b, edgecolors=color,
                       facecolors="white", linewidths=1.2, s=27, zorder=4)
        ax.axhline(1.0, color=muted, linestyle=(0, (4, 3)), linewidth=1.1)
        ax.set_title(f"Hidden width {width:,}", loc="left", fontsize=12, color=title_color)
        ax.set_xticks(list(positions), [f"{r:,}" for r in rows])
        ax.set_xlim(-0.17, len(rows) - 0.83)
        ax.set_xlabel("Token rows", color=muted)
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}×"))
        ax.grid(axis="y", color="#DFE5E7", linewidth=0.75)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#B8C3C8")
        ax.tick_params(length=0, pad=5, labelcolor=title_color)
    fig.text(0.04, 0.035, "Independent benchmark invocations with CUDA graph replay. "
             "Each marker uses the two backend medians from the same run.",
             fontsize=9, color=muted)
    fig.savefig(RESULTS / "warp-comparison.png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
