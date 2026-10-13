"""Static HTML checks and evidence probes, without creating PPTX/DOCX.

System Node is used only for JavaScript syntax checking of an HTML script.
It is not used as a substitute for the mandatory managed artifact runtime.
"""
from html.parser import HTMLParser
from pathlib import Path
import json
import re
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT/"09_结项文书与答辩"/"04_答辩演示"/"答辩演示.html"


class CheckHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.slides, self.notes, self.seconds = 0, 0, 0

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if tag == "section" and "slide" in attrs.get("class", "").split():
            self.slides += 1
            self.seconds += int(attrs.get("data-seconds", 0))
        if tag == "aside" and attrs.get("class") == "notes":
            self.notes += 1


def main():
    content = HTML.read_text(encoding="utf-8")
    parser = CheckHTML()
    parser.feed(content)
    private = HTML.parent/"_qa"
    private.mkdir(exist_ok=True)
    script_path = private/"inline_script.js"
    script_path.write_text("\n".join(re.findall(r"<script>([\s\S]*?)</script>", content)), encoding="utf-8")
    node = shutil.which("node")
    syntax = subprocess.run([node, "--check", str(script_path)], capture_output=True, text=True) if node else None
    payload = json.loads((HTML.parent/"defense_results.json").read_text(encoding="utf-8"))
    result = {"slides": parser.slides, "notes": parser.notes, "total_seconds": parser.seconds,
              "print_css": "@media print" in content,
              "safe_fallback_without_results": "正式测试和独立核验完成后" in content,
              "actual_results_verified": payload.get("verified") is True and payload.get("status") == "verified",
              "actual_training_scale": payload.get("training"),
              "javascript_syntax_checked": syntax is not None,
              "javascript_syntax_valid": syntax.returncode == 0 if syntax is not None else None,
              "syntax_error": syntax.stderr if syntax is not None else "node unavailable",
              "qa_scope": "static HTML and script syntax; visual inspection recorded separately"}
    (private/"static_check.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["slides"] != 12 or result["notes"] != 12 or result["total_seconds"] != 480 or result["javascript_syntax_valid"] is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
