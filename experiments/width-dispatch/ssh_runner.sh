#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
[[ "$#" == 3 && "$1" == bench && "$2" =~ ^(53|54|55)$ && "$3" =~ ^(4102|4106|4107|4108|4109|4111|4113|4116)$ ]] || exit 2
seed=$2
width=$3
revision=$(printenv BENCH_REVISION || true)
[[ "$revision" =~ ^([0-9a-f]{40}|[0-9a-f]{64})$ ]] || exit 2
host=$(git config --local --get rmsnorm.host)
setup=$(git config --local --get rmsnorm.setup)
remote_total_timeout=$(python3 -c 'import json; print(json.load(open("plans/width-dispatch.json"))["remote_total_timeout_seconds"])')
[[ "$remote_total_timeout" =~ ^[0-9]+$ ]] || exit 2
remote_cleanup_reserve=$(python3 -c 'import json; print(json.load(open("plans/width-dispatch.json"))["remote_timeout_cleanup_reserve_seconds"])')
[[ "$remote_cleanup_reserve" =~ ^[0-9]+$ ]] || exit 2
files=(kernels.py experiments/width-dispatch/run.py experiments/width-dispatch/telemetry.py experiments/width-dispatch/PLAN.md plans/width-dispatch.json)
git diff --quiet "$revision" -- "${files[@]}" || exit 2
remote=$(cat <<'REMOTE'
set -euo pipefail
umask 077
SECONDS=0
work=$(mktemp -d)
candidate=""
tmpdir=$(printenv TMPDIR || true)
if [[ -z "$tmpdir" ]]; then tmpdir=/tmp; fi
cache_root=$(printenv XDG_CACHE_HOME || true)
if [[ -z "$cache_root" ]]; then cache_root="$HOME/.cache"; fi
trap 'rm -rf -- "$work"; if [ -n "$candidate" ]; then rm -f -- "$candidate"; fi' EXIT
tar --no-same-owner -xf - -C "$work"
cache_dir="$cache_root/h100-rmsnorm-lab/width-dispatch"
mkdir -p -- "$cache_dir"
exec 9>"$tmpdir/rmsnorm-vllm-$UID.lock"
flock -x -n -E 75 9
REMOTE
)
printf -v line '. %q' "$setup"
remote+=$'\n'"$line"
printf -v line 'record="$cache_dir"/width-dispatch-%q-s%q-w%q.json' "$revision" "$seed" "$width"
remote+=$'\n'"$line"
printf -v line 'revision=%q; seed=%q; width=%q' "$revision" "$seed" "$width"
remote+=$'\n'"$line"
remote+=$'\n''if [ -s "$record" ]; then cat "$record"; exit 0; fi'
remote+=$'\n''candidate=$(mktemp "$cache_dir/.width-dispatch.XXXXXX")'
remote+=$'\n''bench_started_utc=$(python3 -c '\''from datetime import datetime, timezone; print(datetime.now(timezone.utc).isoformat())'\'')'
remote+=$'\n''bench_started_monotonic_ns=$(python3 -c '\''import time; print(time.monotonic_ns())'\'')'
remote+=$'\n''remote_elapsed_seconds=$SECONDS'
printf -v line 'remote_total_timeout=%q' "$remote_total_timeout"
remote+=$'\n'"$line"
remote+=$'\n''remaining_seconds=$((remote_total_timeout - remote_elapsed_seconds))'
printf -v line 'cleanup_reserve_seconds=%q' "$remote_cleanup_reserve"
remote+=$'\n'"$line"
remote+=$'\n''if (( remaining_seconds <= cleanup_reserve_seconds )); then exit 124; fi'
remote+=$'\n''benchmark_budget_seconds=$((remaining_seconds - cleanup_reserve_seconds))'
remote+=$'\n''bench_timeout_seconds=$(( benchmark_budget_seconds < 300 ? benchmark_budget_seconds : 300 ))'
execution_manifest=$(printenv BENCH_EXECUTION_SOURCE_SHA256 || true)
[[ -n "$execution_manifest" ]] || exit 2
printf -v line 'export BENCH_EXECUTION_SOURCE_SHA256=%q' "$execution_manifest"
remote+=$'\n'"$line"
remote+=$'\n''export BENCH_STARTED_UTC="$bench_started_utc"'
remote+=$'\n''export BENCH_STARTED_MONOTONIC_NS="$bench_started_monotonic_ns"'
remote+=$'\n''export BENCH_TIMEOUT_SECONDS="$bench_timeout_seconds"'
printf -v command 'timeout -k 10s "${bench_timeout_seconds}s" env BENCH_REVISION=%q python3 "$work"/experiments/width-dispatch/run.py --seed %q --width %q' \
    "$revision" "$seed" "$width"
remote+=$'\n'"if $command > \"\$candidate\"; then bench_status=0; else bench_status=\$?; fi"
remote+=$'\n'$(cat <<'REMOTE'
if (( bench_status == 124 || bench_status == 137 )); then
  python3 - "$work" "$candidate" "$record" "$revision" "$seed" "$width" "$bench_status" <<'PY'
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
root = Path(sys.argv[1])
candidate, record_path, revision, seed, width, status = sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
compute_sources = ("kernels.py", "experiments/width-dispatch/run.py", "experiments/width-dispatch/telemetry.py", "experiments/width-dispatch/PLAN.md", "plans/width-dispatch.json")
def source_digest(name):
    with open(root / name, "rb") as source:
        return hashlib.sha256(source.read()).hexdigest()
with open(candidate, "w") as output:
    json.dump({"status": "execution_failed", "seed": seed, "rows": 32768, "width": width,
               "source_revision": revision,
               "source_sha256": {name: source_digest(name) for name in compute_sources},
               "execution_source_sha256": json.loads(os.environ["BENCH_EXECUTION_SOURCE_SHA256"]),
               "started_utc": os.environ["BENCH_STARTED_UTC"],
               "finished_utc": datetime.now(timezone.utc).isoformat(),
               "wall_seconds": max(0.0, (time.monotonic_ns() - int(os.environ["BENCH_STARTED_MONOTONIC_NS"])) / 1e9),
               "failure": {"stage": "remote_benchmark_timeout", "type": "TimeoutExpired",
                           "message": f"Benchmark exceeded its {os.environ['BENCH_TIMEOUT_SECONDS']}-second budget (timeout exit {status})."}},
              output, indent=2, allow_nan=False)
    output.write("\n")
    output.flush()
    os.fsync(output.fileno())
os.replace(candidate, record_path)
directory = os.open(os.path.dirname(record_path), os.O_RDONLY)
try:
    os.fsync(directory)
finally:
    os.close(directory)
PY
  candidate=""
  cat "$record"
  exit 0
fi
if (( bench_status != 0 )); then exit "$bench_status"; fi
REMOTE
)
remote+=$'\n'$(cat <<'REMOTE'
python3 - "$candidate" "$record" <<'PY'
import json
import os
import sys
with open(sys.argv[1]) as handle:
    record = json.load(handle)
record["execution_source_sha256"] = json.loads(os.environ["BENCH_EXECUTION_SOURCE_SHA256"])
with open(sys.argv[1], "w") as handle:
    json.dump(record, handle, indent=2, allow_nan=False)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(sys.argv[1], sys.argv[2])
directory = os.open(os.path.dirname(sys.argv[2]), os.O_RDONLY)
try:
    os.fsync(directory)
finally:
    os.close(directory)
PY
candidate=""
cat "$record"
REMOTE
)
printf -v deadline_command 'timeout -k 5s %ss bash -c %q' "$remote_total_timeout" "$remote"
printf -v wrapped_remote 'status=0; %s || status=$?; if (( status == 124 || status == 137 )); then exit 124; else exit "$status"; fi' "$deadline_command"
remote=$wrapped_remote
COPYFILE_DISABLE=1 tar --format=ustar -cf - "${files[@]}" |
    ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=12 \
        -o ServerAliveInterval=10 -o ServerAliveCountMax=3 "$host" "$remote"
