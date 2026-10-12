"""Verify portable local links and the final deliverable's required artifacts."""
from html.parser import HTMLParser
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


class LocalLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.images = 0

    def handle_starttag(self, tag, attrs):
        self.images += tag == "img"
        self.links.extend(value for key, value in attrs if key in ("src", "href") and value)


def main():
    parser = LocalLinks()
    parser.feed((ROOT / "结果总览.html").read_text(encoding="utf-8"))
    for target in parser.links:
        if "://" not in target:
            assert (ROOT / target).is_file(), f"Missing HTML resource: {target}"
    assert parser.images == 7, "Expected six benchmark figures plus one three-axis figure"
    assert len(list((ROOT / "04_实验数据与图表" / "figures").glob("*.png"))) == 6
    documents = ("工作总说明.md", "README.md", "06_结项说明与报告/结项技术报告.md",
                 "06_结项说明与报告/一个月收尾与交付清单.md")
    for name in documents:
        path = ROOT / name
        text = path.read_text(encoding="utf-8")
        assert text.strip(), f"Empty document: {name}"
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if "://" not in target:
                assert (path.parent / target).exists(), f"Missing Markdown resource: {name}: {target}"
    for name, key, expected in (
        ("result_verification.json", "status", "passed"),
        ("copy_verification.json", "all_passed", True),
        ("original_preservation_check.json", "preserved", True),
    ):
        record = json.loads((ROOT / "06_结项说明与报告" / name).read_text(encoding="utf-8"))
        assert record[key] == expected, f"Audit has not passed: {name}"
    extension = ROOT / "07_单轴三轴多算法对比"
    if extension.exists():
        page = extension / "结果总览.html"
        multi_parser = LocalLinks()
        multi_parser.feed(page.read_text(encoding="utf-8"))
        for target in multi_parser.links:
            if "://" not in target:
                assert (extension / target).is_file(), f"Missing multi-algorithm HTML resource: {target}"
        assert multi_parser.images == 8, "Expected eight multi-algorithm figures"
        for path in (extension / "README.md", extension / "新融合分析.md", extension / "实验约定.md",
                     ROOT / "06_结项说明与报告/答辩提纲与提交前核验.md",
                     ROOT / "05_三轴拓展演示/README.md"):
            for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
                if "://" not in target:
                    assert (path.parent / target).exists(), f"Missing Markdown resource: {path.name}: {target}"
        summary = json.loads((extension / "result_summary.json").read_text(encoding="utf-8"))
        manifest = json.loads((extension / "results/evaluation_manifest.json").read_text(encoding="utf-8"))
        assert summary["fresh_episode_count"] == manifest["episode_count"] == 7560
        assert summary["raw_episode_count"] == 4320
        assert summary["calibrated_episode_count"] == summary["policy_calibration_pair_count"] == 3240
        assert summary["historical_episode_count"] == manifest["historical"]["episode_count"] == 720
        audit = json.loads((extension / "核验记录.json").read_text(encoding="utf-8"))
        assert audit["status"] == "passed", "Multi-algorithm result audit has not passed"
        assert len(list((extension / "models").glob("*/*/seed_*/training_metadata.json"))) == 18
    print("Delivery links and required artifacts: passed; V1 seven figures, V2 eight figures and all audit records.")


if __name__ == "__main__":
    main()
