"""Run the dataset build and full validation with persistent progress/status logs."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOG = ROOT / "job.log"
STATUS = ROOT / "job_status.json"


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def status(state: str, **details: object) -> None:
    document = {"state": state, "updated_at_utc": timestamp(), "pid": os.getpid(), **details}
    STATUS.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def run(command: list[str], log) -> None:
    log.write(f"\n[{timestamp()}] Running: {' '.join(command)}\n")
    log.flush()
    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    status("building", log=str(LOG))
    with LOG.open("w", encoding="utf-8", buffering=1) as log:
        try:
            run([sys.executable, "-u", str(ROOT / "build_dataset.py")], log)
            status("validating", log=str(LOG))
            run([sys.executable, "-u", str(ROOT / "validate_dataset.py"), "--rebuild-check"], log)
            status("complete", log=str(LOG), validation="PASS", reproducibility="PASS")
            log.write(f"[{timestamp()}] COMPLETE: dataset and deterministic rebuild validated\n")
        except Exception as error:
            status("failed", log=str(LOG), error=str(error))
            log.write(f"[{timestamp()}] FAILED\n{traceback.format_exc()}\n")
            raise


if __name__ == "__main__":
    main()
