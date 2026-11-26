"""Plot the verified 512-row split scheduling throughput and device power."""
from types import SimpleNamespace

from copy_free_plot import draw
import run
from split_schedule_report import read_results
import split_schedule_run as protocol


if __name__ == "__main__":
    workload = SimpleNamespace(rows=protocol.ROWS, timing_seeds=protocol.TIMING_SEEDS, backends=protocol.BACKENDS)
    draw(read_results(), run.HERE / "split-schedule-512-throughput.png", workload=workload,
         backend_labels=("Fused direct output", "Staged split", "Adjacent split"))
