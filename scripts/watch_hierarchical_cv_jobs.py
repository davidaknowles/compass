#!/usr/bin/env python
"""Release fold workers after their atomic hierarchical CV checkpoints complete."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from time import sleep

import numpy as np


def checkpoint_complete(path: Path) -> bool:
    if not path.exists():
        return False
    with np.load(path, allow_pickle=False) as checkpoint:
        completed = int(np.asarray(checkpoint["next_lambda_index"]).reshape(-1)[0])
        return completed == len(checkpoint["lambdas"])


def job_active(job_id: str) -> bool:
    result = subprocess.run(
        ["squeue", "-h", "-j", job_id],
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--job",
        action="append",
        nargs=2,
        metavar=("JOB_ID", "CHECKPOINT"),
        required=True,
    )
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    args = parser.parse_args()

    for job_id, checkpoint_value in args.job:
        checkpoint = Path(checkpoint_value).expanduser()
        while job_active(job_id):
            if checkpoint_complete(checkpoint):
                subprocess.run(["scancel", job_id], check=True)
                print(f"CV complete; stopped job {job_id}: {checkpoint}", flush=True)
                break
            sleep(args.poll_seconds)
        else:
            state = "complete" if checkpoint_complete(checkpoint) else "incomplete"
            print(f"Job {job_id} ended with checkpoint {state}: {checkpoint}", flush=True)


if __name__ == "__main__":
    main()
