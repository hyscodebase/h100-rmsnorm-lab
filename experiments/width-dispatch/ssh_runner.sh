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
files=(kernels.py experiments/width-dispatch/run.py experiments/width-dispatch/telemetry.py experiments/width-dispatch/PLAN.md plans/width-dispatch.json)
git diff --quiet "$revision" -- "${files[@]}" || exit 2
remote=$(cat <<'REMOTE'
set -euo pipefail
umask 077
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
remote+=$'\n''if [ -s "$record" ]; then cat "$record"; exit 0; fi'
remote+=$'\n''candidate=$(mktemp "$cache_dir/.width-dispatch.XXXXXX")'
printf -v command 'timeout -k 10s 300s env BENCH_REVISION=%q python3 "$work"/experiments/width-dispatch/run.py --seed %q --width %q' "$revision" "$seed" "$width"
remote+=$'\n'"$command"' > "$candidate"'
remote+=$'\n'$(cat <<'REMOTE'
python3 - "$candidate" "$record" <<'PY'
import json
import os
import sys
with open(sys.argv[1], "rb+") as handle:
    json.load(handle)
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
COPYFILE_DISABLE=1 tar --format=ustar -cf - "${files[@]}" |
    ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=12 \
        -o ServerAliveInterval=10 -o ServerAliveCountMax=3 "$host" "$remote"
