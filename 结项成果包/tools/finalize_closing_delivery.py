"""Freeze the completed research, bind final artifacts, and verify delivery.

This tool never trains, selects models, edits formal results or submits files.
The technical freeze is immutable through this tool. The independent final
coverage audit is kept outside the manifest to avoid a circular self-hash.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
import sys
import zipfile
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "08_惯量域随机化与泛化边界研究"
CLOSING = ROOT / "09_结项文书与答辩"
SUPPLEMENT = CLOSING / "08_补充阅读与团队复核"
ANALYSIS = STUDY / "05_研究分析与核验"
TECH = ANALYSIS / "技术研究冻结清单.json"
DELIVERY = ROOT / "最终交付清单.json"
CHECK = CLOSING / "05_来源与核验" / "最终整合核验.json"
ARCHIVE = ROOT / "最终阅读与展示包.zip"
SKIP = {"__pycache__", ".pytest_cache", "tmp", ".artifact_deps", "06_开发联调"}
AUDIT_NAMES = {"最终任务覆盖审查.md", "最终任务覆盖审查.json"}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def row(path):
    return {"relative": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)}


def files_under(directory):
    return [p for p in directory.rglob("*") if p.is_file() and not any(part in SKIP for part in p.relative_to(ROOT).parts)]


def technical_files():
    paths = [p for p in files_under(STUDY) if p != TECH]
    paths += files_under(ROOT / "inertia_generalization")
    paths += sorted((ROOT / "tests").glob("*.py"))
    paths += [ROOT / "run_inertia.py", ROOT / "tools/verify_inertia_results.py", ROOT / "requirements-tested.txt", ROOT / "requirements.txt"]
    return sorted(set(paths))


def final_files():
    paths = [p for p in files_under(CLOSING) if p != CHECK and p.name not in AUDIT_NAMES
             and ("_qa" not in p.parts or p.name == "visual_review.json")]
    paths += [TECH, ROOT / "README.md", ROOT / "工作总说明.md", ROOT / "最终交付总说明.md", ROOT / "结果总览.html", ROOT / ".gitignore"]
    paths += [ROOT / "tools" / name for name in (
        "build_closing_pdfs.py", "render_closing_pdfs.py", "build_closing_video.py", "build_closing_supplement.py",
        "artifact_runtime_diagnostic.md", "artifact_runtime_diagnostic.json", "artifact_runtime_diagnostic.py", "finalize_closing_delivery.py")]
    return sorted(set(paths))


def csv_count(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def assert_equal(actual, expected, name):
    if actual != expected:
        raise ValueError(f"{name}: expected {expected!r}, got {actual!r}")


def supplement_check():
    literature = SUPPLEMENT / "01_补充文献"
    acquisition = read(literature / "补充文献获取清单.json")
    assert_equal(acquisition["existing_main_text_academic_source_count"], 9, "existing scholarly sources")
    assert_equal(acquisition["new_full_text_academic_source_count"], 3, "supplementary scholarly sources")
    assert_equal(acquisition["total_academic_source_count"], 12, "combined scholarly sources")
    papers = acquisition["documents"]
    assert_equal(len(papers), 3, "supplementary papers")
    assert_equal(len({p["doi"].lower() for p in papers}), 3, "unique supplementary DOIs")
    from pypdf import PdfReader
    for paper in papers:
        assert_equal(paper["full_text_obtained"], True, paper["id"] + " full text")
        pdf = literature / paper["pdf_path"]
        assert_equal(pdf.stat().st_size, paper["pdf_bytes"], paper["id"] + " PDF size")
        assert_equal(sha(pdf), paper["pdf_sha256"], paper["id"] + " PDF SHA")
        assert_equal(len(PdfReader(pdf).pages), paper["pdf_pages"], paper["id"] + " PDF pages")
    audit = read(SUPPLEMENT / "04_附录独立审查/补充材料独立审查.json")
    assert_equal(audit["status"], "passed", "supplementary independent audit")
    assert_equal(audit["open_findings"], [], "supplementary open findings")
    assert_equal(sha(literature / "补充文献分类与阅读札记.md"), audit["literature"]["notes_sha256"], "audited reading notes")
    mini = SUPPLEMENT / "02_最小独立复核"
    run_record = read(mini / "运行核验记录.json")
    assert_equal(run_record["status"], "passed", "minimal execution record")
    for name, expected_sha in run_record["source_sha256"].items():
        assert_equal(sha(mini / name), expected_sha, "minimal source " + name)
    for execution in run_record["executions"]:
        assert_equal(execution["exit_code"], 0, "minimal exit code")
        assert_equal(sha(mini / execution["output_file"]), execution["output_sha256"], "minimal actual output")
    output = read(mini / "代理实际复核输出.json")
    assert_equal(output["status"], "passed", "minimal policy review")
    assert_equal(len(output["records"]), 8, "minimal instances")
    assert_equal(len({r["key"] for r in output["records"]}), 8, "unique minimal instances")
    assert_equal(output["writes_by_default"], False, "minimal default output")
    assert_equal(output["frozen_inputs_unchanged"], True, "minimal input preservation")
    assert_equal(output["checked_input_count"], 25, "minimal input count")
    assert_equal(sha(mini / output["fixture_file"]), output["fixture_sha256"], "minimal fixture SHA")
    assert_equal(sha(mini / output["expected_results_file"]), output["expected_results_sha256"], "minimal expectation SHA")
    for record in output["records"]:
        assert_equal(record["status"], "passed", record["key"])
        assert_equal(record["action_replay"]["status"], "passed", record["key"] + " action replay")
        rollout = record["short_rollout"]
        assert_equal(rollout["duration_s"], 20.0, record["key"] + " full duration")
        assert_equal(rollout["full_horizon_acceptance_matches_saved"], True, record["key"] + " acceptance")
    failure = read(mini / "代理实际失败教学输出.json")
    assert_equal(failure["status"], "passed", "failure teaching review")
    assert_equal(len(failure["records"]), 1, "failure teaching count")
    metrics = failure["records"][0]["metrics_recomputed_from_physical_states"]
    assert_equal(metrics["success"], False, "teaching example success")
    assert_equal(metrics["failed"], False, "teaching example physical divergence")
    for result in (output, failure):
        for relative, expected_sha in result["checked_input_sha256"].items():
            assert_equal(sha(ROOT / relative), expected_sha, "minimal bound input " + relative)
    pdf_dir = SUPPLEMENT / "03_附录阅读PDF"
    pdfs = read(pdf_dir / "生成清单.json")
    qa = read(pdf_dir / "visual_review.json")
    assert_equal(len(pdfs), 2, "appendix PDFs")
    assert_equal(qa["status"], "passed", "appendix visual review")
    for item in pdfs:
        assert_equal(item["visual_qa"], "passed", item["output"])
        assert_equal(sha(ROOT / item["source"]), item["source_sha256"], item["source"])
        assert_equal(sha(ROOT / item["output"]), item["pdf_sha256"], item["output"])
        assert_equal(len(PdfReader(ROOT / item["output"]).pages), item["pages"], item["output"])
    for record in qa["records"]:
        assert_equal(record["reviewed_pages"], list(range(1, record["pages"] + 1)), "appendix reviewed pages")
        for page in record["rendered_pages"]:
            assert_equal(sha(ROOT / page["relative"]), page["sha256"], "appendix rendered page")
    return {"combined_scholarly_sources": 12, "new_full_text_papers": 3,
            "minimal_instances": 8, "policy_replay_commands": sum(r["action_replay"]["commands_checked"] for r in output["records"]),
            "closed_loop_steps": sum(r["short_rollout"]["steps"] for r in output["records"]),
            "closed_loop_duration_s_each": 20, "failure_teaching_samples": 1,
            "appendix_pdf_pages": sum(p["pages"] for p in pdfs), "member_execution_confirmed": False}


def structural_check():
    model_dir = STUDY / "02_模型与验证日志"
    metas = sorted(model_dir.glob("*/*/seed_*/training_metadata.json"))
    assert_equal(len(metas), 12, "formal training models")
    steps = 0
    for p in metas:
        m = read(p)
        assert_equal(m["status"], "complete", str(p))
        assert_equal(m["actual_steps"], 1_000_000, str(p))
        assert_equal(m["development"], False, str(p))
        steps += m["actual_steps"]
        for filename in ("best_model.zip", "final_model.zip"):
            if not (p.parent / filename).is_file():
                raise FileNotFoundError(p.parent / filename)
    results = STUDY / "03_测试数据与配对结果"
    counts = {"episode_metrics": 10920, "training_mode_pairs": 4680, "calibration_pairs": 4680,
              "pid_pairs": 780, "boundary_points": 440, "empirical_boundaries": 40}
    for name, expected in counts.items():
        assert_equal(csv_count(results / f"{name}.csv"), expected, name)
    assert_equal(len(list(results.glob("*/raw_traces/*.npz"))), 10920, "formal NPZ traces")
    assert_equal(read(results / "evaluation_manifest.json")["episode_count"], 10920, "evaluation manifest")
    assert_equal(read(ANALYSIS / "独立核验记录.json")["status"], "passed", "full independent verification")
    figures = STUDY / "04_图表与结果页面"
    assert_equal(len(list(figures.glob("*.png"))), 5, "scientific PNG figures")
    assert_equal(len(list(figures.glob("*.pdf"))), 5, "scientific PDF figures")
    sys.path.insert(0, str(ROOT / ".artifact_deps"))
    from pypdf import PdfReader
    pdfs = read(CLOSING / "06_可打印PDF/生成清单.json")
    assert_equal(len(pdfs), 4, "A4 reading PDFs")
    for item in pdfs:
        assert_equal(item["visual_qa"], "passed", item["output"])
        assert_equal(sha(ROOT / item["source"]), item["source_sha256"], item["source"])
        assert_equal(sha(ROOT / item["output"]), item["pdf_sha256"], item["output"])
        assert_equal(len(PdfReader(ROOT / item["output"]).pages), item["pages"], item["output"])
    deck = CLOSING / "04_答辩演示"
    qa = read(deck / "_qa/visual_review.json")
    assert_equal(qa["pdf_reviewed_pages"], list(range(1, 13)), "defense PDF visual review")
    assert_equal(sha(deck / "答辩演示.html"), qa["html_sha256"], "defense HTML")
    assert_equal(sha(deck / "答辩演示阅读本.pdf"), qa["pdf_sha256"], "defense PDF")
    assert_equal(sha(deck / "defense_results.js"), qa["defense_results_js_sha256"], "defense JavaScript payload")
    assert_equal(sha(deck / "defense_results.json"), qa["defense_results_json_file_sha256"], "defense JSON payload")
    assert_equal(qa["defense_results_js_matches_json_payload"], True, "HTML-bound payload equality")
    video = read(CLOSING / "07_演示视频/视频生成清单.json")
    assert_equal(video["visual_review"], "passed", "video visual review")
    assert_equal(video["decode_validation"]["status"], "passed", "video full decode")
    assert_equal(video["decode_validation"]["frames"], 900, "video frames")
    assert_equal(sha(ROOT / video["path"]), video["sha256"], "video SHA")
    supplement = supplement_check()
    return {"formal_training_models": 12, "training_steps": steps, "csv_rows": counts,
            "raw_traces": 10920, "scientific_figures": 5, "a4_pdf_pages": sum(p["pages"] for p in pdfs),
            "defense_pdf_pages": 12, "video_frames": 900, "video_duration_s": 90,
            "full_independent_verification": "passed", "regression_tests": "63 passed in prior final test run; no research source changed since",
            "closing_supplement": supplement}


class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.urls = []

    def handle_starttag(self, tag, attrs):
        self.urls += [v for k, v in attrs if k in {"href", "src"} and v]


def link_check():
    primary = [ROOT / "README.md", ROOT / "最终交付总说明.md", ROOT / "结果总览.html",
               STUDY / "README.md", ANALYSIS / "研究执行与冻结.md", ANALYSIS / "研究分析.md",
               CLOSING / "README.md", STUDY / "04_图表与结果页面/结果总览.html", CLOSING / "04_答辩演示/答辩演示.html"]
    primary += sorted((CLOSING / "01_两篇学习综述").glob("*综述.md"))
    primary += sorted((CLOSING / "02_结项报告").glob("*.md"))
    primary += [SUPPLEMENT / "README.md"]
    primary += sorted((SUPPLEMENT / "01_补充文献").glob("*札记.md"))
    primary += sorted((SUPPLEMENT / "02_最小独立复核").glob("*说明.md"))
    missing = []; checked = 0
    for source in primary:
        content = source.read_text(encoding="utf-8")
        if source.suffix == ".html":
            parser = Links(); parser.feed(content); urls = parser.urls
        else:
            urls = re.findall(r"\[[^\]]*\]\(([^)]+)\)", content)
        for url in urls:
            url = url.strip().strip("<>")
            parts = urlsplit(url)
            if parts.scheme or not parts.path or url.startswith("//"):
                continue
            path = (source.parent / unquote(parts.path)).resolve()
            checked += 1
            if not path.exists():
                missing.append({"source": source.relative_to(ROOT).as_posix(), "link": url})
    if missing:
        raise ValueError("Broken primary links: " + json.dumps(missing, ensure_ascii=False))
    return {"primary_documents": len(primary), "local_targets_checked": checked, "missing": []}


def verify_manifest(path, current_paths):
    manifest = read(path); bound = set()
    for item in manifest["files"]:
        target = ROOT / item["relative"]
        assert_equal(target.stat().st_size, item["bytes"], item["relative"] + " size")
        assert_equal(sha(target), item["sha256"], item["relative"] + " SHA")
        bound.add(target)
    assert_equal(bound, set(current_paths), str(path) + " membership")
    return {"manifest": path.relative_to(ROOT).as_posix(), "sha256": sha(path), "files": len(bound),
            "bytes": sum(item["bytes"] for item in manifest["files"]), "status": "passed"}


def light_paths():
    paths = [ROOT / "最终交付总说明.md", ROOT / "README.md", STUDY / "README.md", STUDY / "研究计划.md"]
    for prefix in ("01_两篇学习综述", "02_结项报告", "03_答辩内容", "04_提交核对与真实贡献", "04_答辩演示", "06_可打印PDF", "07_演示视频"):
        paths += [p for p in files_under(CLOSING / prefix) if "_qa" not in p.parts]
    paths += [CLOSING / "README.md"]
    # Full texts are useful in the reading copy; large source-page QA images
    # stay in the complete delivery, whose manifest still binds them.
    paths += [p for p in files_under(SUPPLEMENT) if "视觉核对" not in p.parts]
    paths += [p for p in files_under(ANALYSIS) if p.name not in {"技术研究冻结清单.json", "测试前独立核验_attempt1.json"}]
    paths += files_under(STUDY / "04_图表与结果页面")
    # Keep every evidence link on the offline results page usable in the
    # reading package. Full trajectory directories remain in the full archive.
    paths += [p for p in (STUDY / "03_测试数据与配对结果").glob("*") if p.is_file() and p.suffix in {".csv", ".json"}]
    paths += [p for p in (CLOSING / "05_来源与核验").glob("*") if p.is_file() and p.suffix in {".md", ".json"} and p.name not in AUDIT_NAMES and p != CHECK]
    paths += [ROOT / "tools/artifact_runtime_diagnostic.md"]
    return sorted(set(paths))


def build():
    structural_check()
    if TECH.exists():
        verify_manifest(TECH, technical_files())
    else:
        rows = [row(p) for p in technical_files()]
        write(TECH, {"created_utc": datetime.now(timezone.utc).isoformat(), "scope": "post-research technical delivery freeze; not a replacement for pre-test protocol/deployment freezes",
                     "exclusions": sorted(SKIP) + ["this manifest"], "file_count": len(rows), "files": rows,
                     "legacy_preservation_baseline": "08_惯量域随机化与泛化边界研究/01_实验协议与配置/prior_assets_snapshot.json"})
    if DELIVERY.exists() or ARCHIVE.exists():
        raise FileExistsError("Final delivery already exists; use --verify. Do not overwrite a reviewed final delivery.")
    rows = [row(p) for p in final_files()]
    write(DELIVERY, {"created_utc": datetime.now(timezone.utc).isoformat(), "scope": "final documents and display artifacts plus referenced complete V3 technical freeze",
                     "file_count": len(rows), "files": rows, "technical_manifest": TECH.relative_to(ROOT).as_posix(),
                     "legacy_assets": "referenced through immutable prior_assets_snapshot.json; not duplicated here",
                     "exclusions": sorted(SKIP) + ["_qa screenshots (QA record retained)", "independent final coverage audit (no circular self-hash)", "this manifest", "ZIP", "integration verification record"],
                     "format_blockers": ["native DOCX", "native PPTX"], "school_submission_status": "not submitted by this agent"})
    with zipfile.ZipFile(ARCHIVE, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr("阅读说明.txt", "这是阅读与展示副本。完整代码、训练模型、replay、CSV/NPZ与哈希清单仍在结项成果包。请先读最终交付总说明.md。学校DOCX/PPTX尚未生成，本地文件不代表已正式提交。文中完整实验路径的复核请使用完整成果包。\n")
        for path in light_paths():
            z.write(path, path.relative_to(ROOT).as_posix())
    return verify()


def verify():
    stats = structural_check()
    manifests = [verify_manifest(TECH, technical_files()), verify_manifest(DELIVERY, final_files())]
    links = link_check()
    expected = {p.relative_to(ROOT).as_posix(): p for p in light_paths()}
    with zipfile.ZipFile(ARCHIVE) as z:
        assert_equal(z.testzip(), None, "ZIP CRC")
        assert_equal(set(z.namelist()), set(expected) | {"阅读说明.txt"}, "ZIP membership")
        for name, path in expected.items():
            assert_equal(hashlib.sha256(z.read(name)).hexdigest(), sha(path), name + " archive SHA")
    result = {"status": "passed", "checked_utc": datetime.now(timezone.utc).isoformat(), "scope": "artifact integrity, counts and primary local links; scientific replay already recorded separately",
              "structure": stats, "manifests": manifests, "links": links,
              "light_archive": {"relative": ARCHIVE.relative_to(ROOT).as_posix(), "entries": len(expected) + 1, "bytes": ARCHIVE.stat().st_size, "sha256": sha(ARCHIVE), "all_entries_match_source": True},
              "remaining": ["school template/deadline/project number", "personal reading/reproduction/contribution confirmation", "fees/opinions/signatures", "native DOCX/PPTX managed runtime", "actual submission and defense"],
              "no_new_experiments": True}
    write(CHECK, result)
    return result


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(); mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build", action="store_true"); mode.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = build() if args.build else verify()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
