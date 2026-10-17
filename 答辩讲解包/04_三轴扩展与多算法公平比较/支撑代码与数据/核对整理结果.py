"""Check copied source hashes and newly authored local Markdown links."""
from pathlib import Path
import hashlib
import json
import re
from urllib.parse import unquote

PROJECT = Path(__file__).resolve().parents[3]
BASE = PROJECT / "答辩讲解包"


def main():
    reports = []
    for name in ("04_三轴扩展与多算法公平比较", "05_控制方法改进与效果分析"):
        step = BASE / name
        rows = json.loads((step / "支撑代码与数据/来源清单.json").read_text(encoding="utf-8"))["sources"]
        for row in rows:
            digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            assert digest(PROJECT / row["source"]) == row["sha256"] == digest(step / row["copy"]), row
        docs = [step / "本步讲解.md", step / "答辩与论文用稿.md", step / "支撑代码与数据/材料说明.md"]
        links = 0
        for doc in docs:
            for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", doc.read_text(encoding="utf-8")):
                target = unquote(target.split("#", 1)[0])
                if not target or "://" in target:
                    continue
                assert (doc.parent / target).exists(), (doc, target)
                links += 1
        reports.append({"step": name, "status": "passed", "original_copy_hashes_checked": len(rows),
                        "authored_document_links_checked": links,
                        "explanation_chinese_characters": len(re.findall("[\u4e00-\u9fff]", docs[0].read_text(encoding="utf-8"))),
                        "scope": "复制一致性与新编入口链接；不表示重新执行原全量仿真核验"})
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
