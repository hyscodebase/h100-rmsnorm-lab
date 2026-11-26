"""Plot verified four- and eight-warp throughput and observed H100 power."""
from types import SimpleNamespace

from copy_free_plot import draw
from four_physical_warps_report import read_results
import four_physical_warps_run as protocol
import run


if __name__ == "__main__":
    workload = SimpleNamespace(rows=protocol.ROWS, timing_seeds=protocol.TIMING_SEEDS, backends=protocol.BACKENDS)
    draw(read_results(), run.HERE / "four-physical-warps-512-throughput.png", workload=workload,
         backend_labels=("Eight physical warps", "Four physical warps"))
