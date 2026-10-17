"""Read-only integrity and key-table review for the seven-step teaching pack.

Default: verify this pack only. --check-originals additionally verifies the
original files. --seal creates a new reviewed manifest once, without touching
the original delivery. No option trains or selects a model.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from statistics import mean
from urllib.parse import unquote, urlsplit

PACK = Path(__file__).resolve().parent
PROJECT = PACK.parent
MANIFEST = PACK / "讲解包清单.json"
RECORD = PACK / "核验结果.json"
SKIP = {"__pycache__", "_构建临时"}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(2**20), b""):
            digest.update(part)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def require(value, message):
    if not value:
        raise ValueError(message)


def pack_files():
    return sorted(p for p in PACK.rglob("*") if p.is_file() and p not in {MANIFEST, RECORD}
                  and not any(part in SKIP for part in p.relative_to(PACK).parts))


def steps():
    result = sorted(p for p in PACK.iterdir() if p.is_dir() and re.match(r"^0[1-7]_", p.name))
    require(len(result) == 7, "Expected exactly seven research steps")
    return result


def copy_check(check_originals=False):
    copied = {}
    locations = {}
    counts = {}
    for step in steps():
        for name in ("本步讲解.md", "答辩与论文用稿.md", "支撑代码与数据/来源清单.json"):
            require((step / name).is_file(), f"Missing step component: {step.name}/{name}")
        listing = read(step / "支撑代码与数据/来源清单.json")["sources"]
        counts[step.name] = len(listing)
        for item in listing:
            target = (step / item["copy"]).resolve()
            require(target.is_relative_to(step), "Copy path outside research step")
            require(target.is_file(), "Missing copied evidence: " + str(target))
            require(sha(target) == item["sha256"], "Copied evidence SHA mismatch: " + str(target))
            if "bytes" in item:
                require(target.stat().st_size == item["bytes"], "Copied evidence byte count mismatch")
            if check_originals:
                original = (PROJECT / item["source"]).resolve()
                require(original.is_relative_to(PROJECT), "Source outside project")
                require(sha(original) == item["sha256"], "Original source SHA mismatch: " + item["source"])
            relative = target.relative_to(PACK).as_posix()
            require(relative not in copied, "Copy registered more than once: " + relative)
            copied[relative] = item
            locations.setdefault(item["source"], target)
    return {"registered_copies": len(copied), "by_step": counts, "original_sources_checked": check_originals}, copied, locations


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attrs):
        self.urls += [value for key, value in attrs if key in {"href", "src"} and value]


def link_check(copied):
    # Original source excerpts keep their historical links. Authored reading
    # material and the generated HTML must have usable local destinations.
    authored = [p for p in pack_files() if p.suffix in {".md", ".html"}
                and p.relative_to(PACK).as_posix() not in copied]
    total = 0
    outside = []
    missing = []
    for source in authored:
        content = source.read_text(encoding="utf-8-sig")
        if source.suffix == ".html":
            parser = Links()
            parser.feed(content)
            urls = parser.urls
        else:
            urls = re.findall(r"\[[^\]]*\]\(([^)]+)\)", content)
        for url in urls:
            url = url.strip().strip("<>")
            parts = urlsplit(url)
            if parts.scheme or not parts.path or url.startswith("//"):
                continue
            target = (source.parent / unquote(parts.path)).resolve()
            total += 1
            if not target.exists():
                missing.append({"source": source.relative_to(PACK).as_posix(), "link": url})
            elif not target.is_relative_to(PACK):
                outside.append({"source": source.relative_to(PACK).as_posix(), "link": url})
    require(not missing, "Broken authored reading links: " + json.dumps(missing, ensure_ascii=False))
    return {"authored_documents": len(authored), "local_links_checked": total, "missing": [],
            "references_to_full_project": outside,
            "original_copied_text_links": "Historical links retained in source copies; use step material notes for current evidence layout"}


def table(locations, original):
    require(original in locations, "Full supporting CSV not selected: " + original)
    with locations[original].open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def true(value):
    require(value in {"True", "False", "true", "false", "1", "0"}, "Unknown boolean in CSV")
    return value.lower() in {"true", "1"}


def rate(rows):
    require(bool(rows), "Empty comparison group")
    return sum(true(row["success"]) for row in rows) / len(rows)


def numeric_check(locations):
    v1 = table(locations, "结项成果包/04_实验数据与图表/episode_metrics.csv")
    v2 = table(locations, "结项成果包/07_单轴三轴多算法对比/results/episode_metrics.csv")
    v3 = table(locations, "结项成果包/08_惯量域随机化与泛化边界研究/03_测试数据与配对结果/episode_metrics.csv")
    require([len(v1), len(v2), len(v3)] == [5600, 7560, 10920], "Formal episode totals differ")
    require(len({r["scenario_id"] for r in v1}) == 800, "V1 scene total differs")
    v1_nominal = {}
    for label in ("PID", "PPO_nominal", "PPO_randomized"):
        rows = [r for r in v1 if r["controller"] == label and r["family"] == "nominal"]
        require(rate(rows) == 1, "V1 nominal success differs")
        v1_nominal[label] = {"n": len(rows), "success_percent": 100 * rate(rows), "control_cost": mean(float(r["control_cost"]) for r in rows)}
    expected_v2 = {
        "single": {"baseline_PID": 66.7, "tuned_PID": 71.1, "improved_PID": 71.1,
                   "PPO_calibrated": 68.9, "SAC_calibrated": 68.5, "TD3_calibrated": 64.1},
        "three": {"baseline_PID": 23.9, "tuned_PID": 71.1, "improved_PID": 72.8,
                  "PPO_calibrated": 55.9, "SAC_calibrated": 63.0, "TD3_calibrated": 24.8},
    }
    v2_rates = []
    for axis, groups in expected_v2.items():
        for label, expected in groups.items():
            rows = [r for r in v2 if r["axis"] == axis and r["controller"] == label]
            require(round(100 * rate(rows), 1) == expected, f"V2 success differs: {axis}/{label}")
            v2_rates.append({"axis": axis, "controller": label, "n": len(rows), "success_percent": round(100 * rate(rows), 1)})
    v3_groups = defaultdict(list)
    for row in v3:
        if row["domain"] == "out_of_domain" and row["experiment"] == "scalar":
            v3_groups[(row["axis"], row["controller"])].append(row)
    v3_main = []
    v3_costs = []
    for axis in ("single", "three"):
        for label in ("PID_N", "PID_R", "SAC_N", "SAC_R"):
            rows = v3_groups[(axis, label)]
            expected = 98.9 if (axis, label) == ("single", "SAC_R") else 84.4 if (axis, label) == ("three", "SAC_R") else 100.0
            require(round(100 * rate(rows), 1) == expected, f"V3 domain success differs: {axis}/{label}")
            v3_main.append({"axis": axis, "controller": label, "n": len(rows),
                            "successful": sum(true(r["success"]) for r in rows),
                            "success_percent": round(100 * rate(rows), 1)})
        pid = v3_groups[(axis, "PID_N")]
        sac = v3_groups[(axis, "SAC_N")]
        pid_cost = mean(float(r["control_cost"]) for r in pid)
        sac_cost = mean(float(r["control_cost"]) for r in sac)
        reduction = 100 * (1 - sac_cost / pid_cost)
        require(round(reduction, 1) == (54.7 if axis == "single" else 28.7), "V3 cost reduction differs")
        v3_costs.append({"axis": axis, "pid_cost": pid_cost, "sac_cost": sac_cost,
                         "reduction_percent": round(reduction, 1),
                         "pid_settling_s": mean(float(r["settling_time_s"]) for r in pid),
                         "sac_settling_s": mean(float(r["settling_time_s"]) for r in sac),
                         "pid_rms_deg": mean(float(r["rms_error_deg"]) for r in pid),
                         "sac_rms_deg": mean(float(r["rms_error_deg"]) for r in sac)})
    pairs = table(locations, "结项成果包/07_单轴三轴多算法对比/results/policy_calibration_pairs.csv")
    require(len(pairs) == 3240, "V2 calibration pair count differs")
    recovered = [r for r in pairs if r["axis"] == "three" and r["algorithm"] == "SAC"]
    require(sum(true(r["success_recovered"]) for r in recovered) == 340, "V2 SAC recovery count differs")
    require(sum(true(r["success_lost"]) for r in recovered) == 0, "V2 SAC lost success count differs")
    return {"formal_episode_counts": {"V1": len(v1), "V2": len(v2), "V3": len(v3), "total": len(v1) + len(v2) + len(v3)},
            "V1_nominal": v1_nominal, "V2_success": v2_rates, "V2_calibration_pairs": len(pairs),
            "V3_out_of_domain": v3_main, "V3_out_of_domain_tradeoff": v3_costs,
            "scope": "Recomputed from complete copied CSVs, including all successes and failures; no training or physical replay"}


def originals_check():
    baseline = read(PACK / "原成果保全基线.json")
    for item in baseline["files"]:
        path = PROJECT / item["source"]
        require(path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], "Original baseline differs: " + item["source"])
    counts = {}
    for name in ("最终交付清单.json", "08_惯量域随机化与泛化边界研究/05_研究分析与核验/技术研究冻结清单.json"):
        listing = read(PROJECT / "结项成果包" / name)
        for item in listing["files"]:
            path = PROJECT / "结项成果包" / item["relative"]
            require(path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], "Original reviewed file differs: " + item["relative"])
        counts[name] = len(listing["files"])
    return {"status": "passed", "baseline_files": len(baseline["files"]), "original_manifest_files_rehashed": counts,
            "originals_modified": False}


def check(seal=False, check_originals=False):
    if seal:
        require(not MANIFEST.exists(), "Reviewed pack manifest already exists; refusing to overwrite")
        save(RECORD, {"status": "pending", "scope": "Assembly checks have not yet completed"})
    source_status, copied, locations = copy_check(check_originals)
    data = numeric_check(locations)
    if seal:
        require(not MANIFEST.exists(), "Reviewed pack manifest already exists; refusing to overwrite")
        rows = [{"relative": p.relative_to(PACK).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)} for p in pack_files()]
        save(MANIFEST, {"client_date": "2026-10-16", "file_count": len(rows), "files": rows,
                        "exclusions": ["this manifest", "核验结果.json (avoid circular self-hash)", "__pycache__", "temporary construction files"],
                        "new_research_run": False, "school_submission_claimed": False})
    links = link_check(copied)
    listing = read(MANIFEST)
    bound = set()
    for item in listing["files"]:
        target = (PACK / item["relative"]).resolve()
        require(target.is_relative_to(PACK), "Manifest path outside teaching pack")
        require(target.stat().st_size == item["bytes"] and sha(target) == item["sha256"], "Pack manifest differs: " + item["relative"])
        bound.add(target)
    require(bound == {p.resolve() for p in pack_files()}, "Pack manifest membership differs")
    result = {"status": "passed", "client_date": "2026-10-16", "files": len(bound),
              "total_bytes": sum(item["bytes"] for item in listing["files"]), "manifest_sha256": sha(MANIFEST),
              "copies": source_status, "links": links, "recomputed_key_numbers": data,
              "new_training_or_experiments": False, "default_writes_files": False}
    if check_originals:
        result["original_preservation"] = originals_check()
    if seal:
        save(RECORD, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal", action="store_true", help="One-time assembly manifest, after review")
    parser.add_argument("--check-originals", action="store_true", help="Additionally read and hash original project evidence")
    args = parser.parse_args()
    print(json.dumps(check(args.seal, args.check_originals), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
