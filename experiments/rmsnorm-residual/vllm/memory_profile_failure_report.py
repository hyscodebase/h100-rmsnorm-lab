"""Verify the publicly archived Nsight Compute qualification attempts."""
import argparse
import json
from pathlib import Path
import re

from copy_free_report import verify_supplemental_samples
import memory_profile as profile
import run
from transfer_report import timestamp, verify_revision

RESULTS = run.HERE / "memory-512-results"


def verify_failures(folder=RESULTS):
    records = json.loads((folder / "qualification-failures.json").read_bytes())
    attempts = {}
    for file in (folder / "attempts").glob("*.json"):
        row = json.loads(file.read_bytes())
        revision = row["code_revision"]
        if (file.stem != f"r{profile.ROWS}-s{profile.NUMERIC_SEED}-{row['attempt_id']}"
                or (row["rows"], row["seed"], row["outcome"], row["exit_code"])
                != (profile.ROWS, profile.NUMERIC_SEED, "failed", 1)
                or timestamp(row["started_utc"]) > timestamp(row["finished_utc"])
                or revision in attempts):
            raise ValueError("An attempt does not match the failed local submission.")
        verify_revision(run.ROOT, revision, ())
        attempts[revision] = row
    if list(folder.glob("s[0-9]*.json")) or len(records) != 2 or len(attempts) != 2:
        raise ValueError("The qualification archive must contain exactly two failed attempts.")
    for record, (reason, attached, error) in zip(records, (
        ("option_rejected", False, "Option profile-from-start is not supported during range replay."),
        ("counter_access_denied", True, "ERR_NVGPUCTRPERM"),
    ), strict=True):
        revision = record["code_revision"]
        if (revision not in attempts or record["seed"] != profile.NUMERIC_SEED
                or record["outcome"] != reason or record["profiler_joined_target"] is not attached
                or record["exported_profiles"] != 0 or record["complete_application_traces"] != 0
                or not re.fullmatch(r"[0-9a-f]{64}", record["private_profiler_log_sha256"])
                or not any(error in line for line in record["errors"])):
            raise ValueError("The profiler rejection differs from its registered attempt.")
        attempts.pop(revision)
    verify_supplemental_samples(folder)
    return records


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("directory", type=Path, nargs="?", default=RESULTS)
    for entry in verify_failures(cli.parse_args().directory):
        print(entry["seed"], entry["code_revision"][:7], entry["outcome"])
