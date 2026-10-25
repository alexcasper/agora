"""Download run results from the `agora-runs` Modal volume, file by file.

`modal volume get` on a directory that is still being written can produce
zero-byte files, so this lists the volume, fetches each file individually,
verifies its size, and skips files already downloaded at the same size.

    python tools/pull_runs.py main-20261025 runs-modal/
    python tools/pull_runs.py main-20261025/phi-4-mini runs-modal/ --only summary.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import modal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", help="path inside the volume, e.g. main-20261025")
    parser.add_argument("dest", help="local directory")
    parser.add_argument("--volume", default="agora-runs")
    parser.add_argument("--only", action="append", help="only files with this basename (repeatable)")
    args = parser.parse_args()

    volume = modal.Volume.from_name(args.volume)
    dest = Path(args.dest)
    fetched = skipped = mismatched = 0
    for entry in volume.listdir(args.prefix, recursive=True):
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        if args.only and Path(entry.path).name not in args.only:
            continue
        target = dest / entry.path
        if target.exists() and target.stat().st_size == entry.size:
            skipped += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        data = b"".join(volume.read_file(entry.path))
        if len(data) != entry.size:
            # Written to while we read it; keep what we got and fetch again next time.
            mismatched += 1
        target.write_bytes(data)
        fetched += 1
    print(f"fetched {fetched}, unchanged {skipped}, size-mismatched {mismatched} -> {dest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
