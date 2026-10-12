"""Archive new research material without modifying any source file.

This archive has its own frozen snapshot and copy manifest. It does not change
the first archive's 67-item manifest or original-file preservation snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
SOURCE = WORKSPACE / "中期前任务收尾"
PROJECT = SOURCE / "大一立项AI在卫星姿态调整中的应用"
COPY_ROOT = HERE / "原样工程" / PROJECT.name
EXCLUDED_PARTS = {".git", "__pycache__", ".vscode"}
PRESERVATION = WORKSPACE / "结项成果包" / "06_结项说明与报告" / "new_source_preservation.json"


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def category(relative: Path) -> str:
    if relative.parts[0] == "pid":
        return "01_PID单轴与三轴参考"
    if relative.parts[0] == "单轴强化学习":
        return "02_单轴PPO历史模型与评测"
    if relative.parts[0] == "三轴强化学习":
        return "03_三轴PPO_SAC_TD3历史模型与评测"
    if relative.parts[0] == "RL与PID对比":
        return "04_历史横向对比与结果说明"
    return "00_项目入口与环境说明"


def source_files() -> list[Path]:
    return sorted((p for p in SOURCE.rglob("*") if p.is_file()), key=lambda p: p.as_posix())


def snapshot() -> dict:
    records = []
    for path in source_files():
        records.append({"source_relative": path.relative_to(WORKSPACE).as_posix(),
                        "size": path.stat().st_size, "sha256": digest(path)})
    return {"schema_version": 1, "snapshot_date": "2026-10-11",
            "source_root": SOURCE.relative_to(WORKSPACE).as_posix(),
            "scope": "all files, including nested Git metadata, editor settings and Python caches",
            "file_count": len(records), "total_bytes": sum(r["size"] for r in records),
            "files": records}


def archive() -> dict:
    frozen = HERE / "source_snapshot.json"
    if not frozen.exists():
        write_json(frozen, snapshot())
    records = []
    excluded = Counter()
    for path in source_files():
        relative = path.relative_to(PROJECT)
        omitted = next((part for part in relative.parts if part in EXCLUDED_PARTS), None)
        if omitted is not None:
            excluded[omitted] += 1
            continue
        if path.suffix.lower() in {".pyc", ".pyo"}:
            excluded["Python bytecode"] += 1
            continue
        copied = COPY_ROOT / relative
        copied.parent.mkdir(parents=True, exist_ok=True)
        original_sha = digest(path)
        if copied.exists() and digest(copied) != original_sha:
            raise RuntimeError(f"Existing archive differs from source: {copied}")
        if not copied.exists():
            shutil.copy2(path, copied)
        copied_sha = digest(copied)
        records.append({"category": category(relative),
                        "source_relative": path.relative_to(WORKSPACE).as_posix(),
                        "copy_relative": copied.relative_to(WORKSPACE).as_posix(),
                        "size": path.stat().st_size, "sha256": original_sha,
                        "copy_sha256": copied_sha, "verified": copied_sha == original_sha})
    result = {"schema_version": 1, "archive_date": "2026-10-11",
              "file_count": len(records), "total_bytes": sum(r["size"] for r in records),
              "category_counts": dict(Counter(r["category"] for r in records)),
              "excluded_file_counts": dict(excluded),
              "copies_identical": all(r["verified"] for r in records), "files": records}
    write_json(HERE / "copy_manifest.json", result)
    return check()


def check() -> dict:
    frozen = json.loads((HERE / "source_snapshot.json").read_text(encoding="utf-8"))
    manifest = json.loads((HERE / "copy_manifest.json").read_text(encoding="utf-8"))
    current = {p.relative_to(WORKSPACE).as_posix(): p for p in source_files()}
    expected = {r["source_relative"]: r for r in frozen["files"]}
    missing = sorted(set(expected) - set(current))
    added = sorted(set(current) - set(expected))
    changed = [name for name in sorted(set(expected) & set(current))
               if digest(current[name]) != expected[name]["sha256"]]
    mismatches = []
    for record in manifest["files"]:
        path = WORKSPACE / record["copy_relative"]
        if not path.exists() or digest(path) != record["sha256"]:
            mismatches.append(record["copy_relative"])
    result = {"check_date": "2026-10-11", "source_root": frozen["source_root"],
              "snapshot_file_count": frozen["file_count"],
              "snapshot_total_bytes": frozen["total_bytes"],
              "copied_file_count": manifest["file_count"],
              "copied_total_bytes": manifest["total_bytes"],
              "source_files_unchanged": not missing and not changed,
              "source_tree_identical": not missing and not changed and not added,
              "copies_identical": not mismatches,
              "missing": missing, "changed": changed, "added": added,
              "copy_mismatches": mismatches}
    result["passed"] = result["source_tree_identical"] and result["copies_identical"]
    write_json(PRESERVATION, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("archive", "check"), nargs="?", default="check")
    args = parser.parse_args()
    result = archive() if args.action == "archive" else check()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)
