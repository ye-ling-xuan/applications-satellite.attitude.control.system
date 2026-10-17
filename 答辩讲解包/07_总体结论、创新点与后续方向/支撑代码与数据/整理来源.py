"""Copy closing evidence without editing the original project or running research."""
from pathlib import Path
import hashlib
import json
import shutil

STEP = Path(__file__).resolve().parents[1]
PACK = STEP.parent
PROJECT = PACK.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    items = [
        ("结项成果包/10_学校结题报告附件3/01_可粘贴正文/项目研究结题报告.md", "报告与计划/项目研究结题报告.md"),
        ("结项成果包/10_学校结题报告附件3/01_可粘贴正文/项目成果.md", "报告与计划/项目成果.md"),
        ("结项成果包/10_学校结题报告附件3/01_可粘贴正文/后续大创工作方向与预期成果.md", "报告与计划/后续大创工作方向与预期成果.md"),
        ("结项成果包/08_惯量域随机化与泛化边界研究/05_研究分析与核验/核心结论.json", "结果证据/核心结论.json"),
        ("结项成果包/08_惯量域随机化与泛化边界研究/05_研究分析与核验/独立核验简要说明.md", "结果证据/独立核验简要说明.md"),
        ("结项成果包/09_结项文书与答辩/05_来源与核验/文献核对清单.md", "文献/文献核对清单.md"),
        ("结项成果包/09_结项文书与答辩/05_来源与核验/文献获取清单.json", "文献/文献获取清单.json"),
        ("结项成果包/09_结项文书与答辩/08_补充阅读与团队复核/01_补充文献/补充文献分类与阅读札记.md", "文献/补充文献分类与阅读札记.md"),
        ("结项成果包/09_结项文书与答辩/08_补充阅读与团队复核/01_补充文献/补充文献获取清单.json", "文献/补充文献获取清单.json"),
    ]
    rows = []
    for source_name, name in items:
        source = PROJECT / source_name
        target = STEP / "支撑代码与数据" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert digest(target) == digest(source), f"Refusing to overwrite differing copy: {target}"
        else:
            shutil.copyfile(source, target)
        rows.append({"source": source_name, "copy": target.relative_to(STEP).as_posix(),
                     "bytes": source.stat().st_size, "sha256": digest(source)})
    out = STEP / "支撑代码与数据/来源清单.json"
    out.write_text(json.dumps({"client_date": "2026-10-16", "scope": "selected closing text, planned future work and evidence; no new research", "sources": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    baseline_path = PACK / "原成果保全基线.json"
    if not baseline_path.exists():
        relative_names = ["结项成果包/README.md", "结项成果包/最终交付总说明.md", "结项成果包/最终交付清单.json", "结项成果包/最终阅读与展示包.zip", "结项成果包/08_惯量域随机化与泛化边界研究/05_研究分析与核验/技术研究冻结清单.json"]
        baseline = {"client_date": "2026-10-16", "scope": "Original reviewed deliverables retained; new teaching pack is outside original delivery", "files": [{"source": name, "bytes": (PROJECT / name).stat().st_size, "sha256": digest(PROJECT / name)} for name in relative_names]}
        baseline_path.write_text(json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"copied": len(rows), "originals_modified": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
