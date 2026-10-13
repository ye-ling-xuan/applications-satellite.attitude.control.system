"""Read-only local link/data checks for the final V3 result and defense pages."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlparse, unquote
import hashlib
import json


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
STUDY = PACKAGE/"08_惯量域随机化与泛化边界研究"


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self.ids = [], set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        for attr in ("href", "src"):
            if attrs.get(attr):
                self.links.append((tag, attrs[attr]))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    reports, problems = [], []
    pages = [HERE/"答辩演示.html", STUDY/"04_图表与结果页面"/"结果总览.html"]
    for page in pages:
        parsed = Links()
        parsed.feed(page.read_text(encoding="utf-8"))
        checked = []
        for tag, href in parsed.links:
            uri = urlparse(href)
            if uri.scheme in ("http", "https", "mailto", "data"):
                checked.append({"tag": tag, "href": href, "type": "external_not_network_checked"})
                continue
            if not uri.path:
                exists = not uri.fragment or unquote(uri.fragment) in parsed.ids
            else:
                exists = (page.parent/unquote(uri.path)).resolve().is_file()
            checked.append({"tag": tag, "href": href, "exists": exists})
            if not exists:
                problems.append(f"{page.name}: {href}")
        reports.append({"page": str(page.relative_to(PACKAGE)), "sha256": sha(page), "links": checked})
    payload_path = HERE/"defense_results.json"
    script_path = HERE/"defense_results.js"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    script = script_path.read_text(encoding="utf-8")
    try:
        assigned = script.split("window.INERTIA_DEFENSE_RESULTS =", 1)[1].strip()
        script_payload = json.loads(assigned.removesuffix(";"))
        script_matches_json = script_payload == payload
    except (IndexError, json.JSONDecodeError):
        script_matches_json = False
    if not script_matches_json:
        problems.append("HTML-bound defense_results.js differs from defense_results.json")
    for key, expected_path in (("source_core_sha256", STUDY/"05_研究分析与核验"/"核心结论.json"),
                               ("source_manifest_sha256", STUDY/"03_测试数据与配对结果"/"evaluation_manifest.json"),
                               ("source_final_audit_sha256", STUDY/"05_研究分析与核验"/"独立核验记录.json")):
        if payload[key] != sha(expected_path):
            problems.append("Defense source digest mismatch: "+key)
    for key, value in payload["figures"].items():
        path = (HERE/value).resolve()
        if not path.is_file():
            problems.append("Missing defense figure: "+key)
    audit = json.loads((STUDY/"05_研究分析与核验"/"独立核验记录.json").read_text(encoding="utf-8"))
    if payload["verified"] is not True or audit["status"] != "passed":
        problems.append("Final independent verification not published")
    result = {"status": "passed" if not problems else "failed", "pages": reports,
              "figure_count": len(payload["figures"]), "training": payload["training"],
              "defense_results_json_file_sha256": sha(payload_path),
              "defense_results_js_sha256": sha(script_path),
              "defense_results_js_matches_json_payload": script_matches_json,
              "final_audit_passed": audit["status"] == "passed", "problems": problems}
    target = HERE/"_qa"/"final_link_check.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "pages": len(reports), "figures": result["figure_count"], "problems": problems}, ensure_ascii=False))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
