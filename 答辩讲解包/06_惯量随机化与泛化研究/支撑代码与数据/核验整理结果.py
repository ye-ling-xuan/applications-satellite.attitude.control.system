"""Read-only copy and local Markdown-link audit for this curated step."""
from pathlib import Path
import hashlib
import json
import re

SUPPORT = Path(__file__).resolve().parent
STEP = SUPPORT.parent
PROJECT = SUPPORT.parents[2]

def main():
    manifest = json.loads((SUPPORT / "来源清单.json").read_text(encoding="utf-8"))
    errors = []
    for record in manifest["sources"]:
        for path in (PROJECT / record["source"], STEP / record["copy"]):
            if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                errors.append(str(path))
    links = 0
    for path in STEP.rglob("*.md"):
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8-sig")):
            if target.startswith(("http:", "https:", "#")):
                continue
            if not (path.parent / target.split("#")[0]).exists():
                errors.append(f"{path}: {target}")
            links += 1
    text = (STEP / "本步讲解.md").read_text(encoding="utf-8")
    print(json.dumps({"copied_sources": len(manifest["sources"]), "local_md_links": links,
                      "main_chinese_characters": len(re.findall(r"[\u4e00-\u9fff]", text)), "errors": errors}, ensure_ascii=False))
    if errors:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
