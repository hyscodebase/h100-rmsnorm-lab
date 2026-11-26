"""Plot actual time-stamped supplementary device observations inside verified GEMM windows."""
import argparse
import sys
from datetime import timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from export_evidence import existing_samples, utc
from summarize import read_results

HERE = Path(__file__).resolve().parent
INK, MUTED = "#182F36", "#4B626B"
COLORS = {2048: "#156879", 4096: "#804C94", 8192: "#B05820", 16384: "#3068A0"}


def read_windows(results, sidecars, require_complete=True):
    tasks, records = read_results(results)
    if require_complete and (len(tasks) != 20 or set(records) != set(tasks.values())):
        raise ValueError(f"The final usage plot requires 20 validated GEMM checkpoints; found {len(records)}.")
    windows = []
    for task, shape in tasks.items():
        if shape not in records:
            continue
        result = records[shape]
        start = utc(result["timed_start_utc"])
        end = start + timedelta(seconds=result["timing"]["wall_elapsed_s"])
        readings = existing_samples(sidecars / f"{task}.csv", start, end)
        windows.append((start, end, shape[1], readings))
    windows.sort(key=lambda window: window[0])
    if not windows or any(b[0] < a[1] for a, b in zip(windows, windows[1:])):
        raise ValueError("This timeline requires distinct, non-overlapping completed windows.")
    return windows


def plot(windows, output):
    start, end = windows[0][0], windows[-1][1]
    dates = start.strftime("%d %b %Y")
    if start.date() != end.date():
        dates += end.strftime(" – %d %b %Y")
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 5.8))
    try:
        fig.subplots_adjust(left=.095, right=.979, top=.75, bottom=.18, hspace=.19)
        fig.text(.04, .928, "H100 power and temperature during BF16 prefill GEMM",
                 color=INK, fontsize=19, fontweight="semibold")
        fig.text(.04, .868, f"Device-wide nvidia-smi readings   ·   {len(windows)} timed runs   ·   {dates} (UTC)",
                 color=MUTED, fontsize=10.5)
        handles = [Line2D([], [], marker="o", linestyle="", markerfacecolor=color, markeredgewidth=0,
                          label=f"{rows:,} rows") for rows, color in COLORS.items()]
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.033, .841), ncol=4,
                   frameon=False, handletextpad=.2, columnspacing=1.65, fontsize=9)

        coverage = []
        for ax, field, label in zip(axes, ("power_w", "temperature_c"), ("Power (W)", "Temperature (°C)")):
            run_count = 0
            for left, right, rows, data in windows:
                ax.axvspan(left, right, color="#EDF1F3", linewidth=0, zorder=0)
                points = [(stamp, values[field]) for stamp, values in data.items() if field in values]
                if points:
                    run_count += 1
                    times, values = zip(*points)
                    ax.scatter(times, values, color=COLORS[rows], s=9, alpha=.81, linewidths=0, zorder=3)
            coverage.append(run_count)
            ax.set_ylabel(label, color=INK, fontsize=10)
            ax.tick_params(colors=MUTED, labelsize=9, length=0, pad=6)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3, integer=True))
            ax.grid(axis="y", color="white", linewidth=1, zorder=1)
            ax.spines[["top", "right"]].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color("#BAC8CC")
        if not any(coverage):
            raise ValueError("No supplementary power or temperature samples exist inside these windows.")
        first_temperature = next((left for left, _, _, data in windows
                                  if any("temperature_c" in values for values in data.values())), None)
        if first_temperature is None or first_temperature - start > (end - start) * .16:
            empty_end = first_temperature or end
            axes[1].text(start + (empty_end - start) / 2, .50, "No temperature samples",
                         transform=axes[1].get_xaxis_transform(), ha="center", va="center", fontsize=9, color=MUTED,
                         bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 3})
        padding = (end - start) * .012
        axes[1].set_xlim(start - padding, end + padding)
        axes[1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8, tz=timezone.utc))
        fmt = "%H:%M" if start.date() == end.date() else "%d %b\n%H:%M"
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter(fmt, tz=timezone.utc))
        axes[1].set_xlabel("Time of day (UTC)", color=INK, labelpad=8, fontsize=10)
        fig.text(.04, .051, "Shading marks timed GEMM windows. Points show individual supplementary samples. "
                 f"Power was recorded in {coverage[0]}/{len(windows)} runs and temperature in {coverage[1]}/{len(windows)}.",
                 color=MUTED, fontsize=9)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180, facecolor="white")
    finally:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--sidecars", type=Path, default=HERE / "evidence" / "supplemental")
    parser.add_argument("--output", type=Path, default=HERE / "evidence" / "device-usage.png")
    args = parser.parse_args()
    try:
        plot(read_windows(args.results, args.sidecars), args.output)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        sys.exit(f"Usage plot failed: {exc}")
