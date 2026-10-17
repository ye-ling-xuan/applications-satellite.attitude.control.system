"""Retry transient Windows JSON rename locks; no learning changes.

This wrapper is used for the single-axis seed42 rerun after an I/O failure.
The original training source, initialization seed and fixed budget are retained.
The wrapper and its SHA are disclosed in the run manifests.
"""
from pathlib import Path
import hashlib
import sys
import time

from deadline_study import protocol

original_write_json = protocol.write_json
retries = []
wrapper = {
    "path": "03_SAC训练/io_retry_runner.py",
    "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "effect": "retry PermissionError in atomic JSON replacement up to30 times with0.1s delay; no change to dynamics/reward/RNG/optimizer",
}


def retry_write_json(path, value):
    if Path(path).name in ("training_manifest.json", "training_metadata.json"):
        value = {**value, "execution_io_wrapper": {**wrapper, "retries_so_far": len(retries)}}
    for attempt in range(30):
        try:
            return original_write_json(path, value)
        except PermissionError:
            retries.append({"path": str(path), "attempt": attempt+1})
            if attempt == 29:
                raise
            time.sleep(.1)


protocol.write_json = retry_write_json
from deadline_study import training

try:
    training.main(sys.argv[1:])
finally:
    original_write_json(Path(__file__).with_name("io_retry_execution_record.json"),
                        {**wrapper, "retry_events": retries})
