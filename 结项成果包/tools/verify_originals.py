"""Capture/check original project file hashes; never write to original files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
PROJECT = PACKAGE.parent
SNAPSHOT = PACKAGE / "06_结项说明与报告" / "original_snapshot.json"
EXCLUDED = {".git", ".conda", "__pycache__", ".pytest_cache", PACKAGE.name}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect() -> dict:
    records = {}
    for path in sorted(PROJECT.rglob("*")):
        relative = path.relative_to(PROJECT)
        if any(part in EXCLUDED for part in relative.parts) or not path.is_file():
            continue
        records[relative.as_posix()] = {"sha256": digest(path), "size": path.stat().st_size}
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["capture", "check"])
    args = parser.parse_args()
    if args.action == "capture":
        if SNAPSHOT.exists():
            raise SystemExit("Snapshot already exists; refusing to replace the original baseline.")
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        records = collect()
        SNAPSHOT.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Captured {len(records)} original files.")
    else:
        expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
        changed = []
        for relative, record in expected.items():
            path = PROJECT / relative
            if not path.is_file() or digest(path) != record["sha256"]:
                changed.append(relative)
        report = {"original_file_count": len(expected), "changed_or_missing": changed,
                  "preserved": not changed, "excluded": sorted(EXCLUDED)}
        (SNAPSHOT.parent / "original_preservation_check.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False))
        if changed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
