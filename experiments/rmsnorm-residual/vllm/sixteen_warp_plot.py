"""Plot verified 128-row CUDA Graph latency for the three implementations."""
import argparse
from pathlib import Path

import run
from shuffle_plot import draw
from sixteen_warp_report import read_results
import sixteen_warp_run as protocol


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=run.HERE / "sixteen-warp-latency.png")
    args = parser.parse_args()
    draw(read_results(run.HERE / "sixteen-warp-results"), args.output, backends=protocol.BACKENDS,
         labels=("vLLM 0.10.0", "Triton · 8 physical warps", "Triton · 16 physical warps"),
         seeds=protocol.TIMING_SEEDS, subtitle="128 token rows × 4,096 features  ·  BF16  ·  In-place")
