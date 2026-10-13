"""Read-only extraction of original reports and retrieval of public references."""
from __future__ import annotations

import concurrent.futures
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET
import zipfile

import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = HERE / "文献缓存"
CACHE.mkdir(parents=True, exist_ok=True)


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts, self.skip = [], 0
    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1
        if tag in ("p", "div", "h1", "h2", "h3", "li", "br"):
            self.parts.append("\n")
    def handle_endtag(self, tag):
        if tag in ("script", "style") and self.skip:
            self.skip -= 1
    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


SOURCES = [
    ("N01", "NASA Guidance Navigation and Control", "https://www.nasa.gov/smallsat-institute/sst-soa/guidance-navigation-and-control/"),
    ("B01", "Markley Crassidis Fundamentals of Spacecraft Attitude Determination and Control", "https://api.crossref.org/works/10.1007/978-1-4939-0802-8"),
    ("B02", "Astrom Murray Feedback Systems open textbook", "https://www.cds.caltech.edu/~murray/books/AM08/pdf/fbs-public_24Jul2020.pdf"),
    ("B03", "Sutton Barto Reinforcement Learning 2018", "https://incompleteideas.net/book/RLbook2020.pdf"),
    ("A01", "PPO original paper", "https://arxiv.org/abs/1707.06347"),
    ("A02", "SAC original paper", "https://arxiv.org/abs/1801.01290"),
    ("A03", "TD3 original paper", "https://arxiv.org/abs/1802.09477"),
    ("A04", "Tobin domain randomization", "https://arxiv.org/abs/1703.06907"),
    ("A05", "Peng dynamics randomization", "https://arxiv.org/abs/1710.06537"),
    ("A06", "Zhao sim to real survey", "https://arxiv.org/abs/2009.13303"),
    ("D01", "Gymnasium environment API", "https://gymnasium.farama.org/api/env/"),
    ("D02", "Gymnasium handling time limits", "https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/"),
    ("D03", "SB3 reinforcement learning tips", "https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html"),
    ("D04", "SB3 SAC documentation", "https://stable-baselines3.readthedocs.io/en/master/modules/sac.html"),
    ("P01", "Raffin SB3 paper JMLR", "https://www.jmlr.org/papers/v22/20-1364.html"),
    ("C01", "Satellite attitude RL Crossref title search", "https://api.crossref.org/works?query.title=Satellite%20attitude%20control%20with%20deep%20reinforcement%20learning&rows=5"),
    ("C02", "Spacecraft inertia identification Crossref", "https://api.crossref.org/works?query.title=Adaptive%20asymptotic%20tracking%20of%20spacecraft%20attitude%20motion%20with%20inertia%20matrix%20identification&rows=3"),
]


def fetch(source):
    sid, label, url = source
    record = {"id": sid, "label": label, "requested_url": url, "retrieved_date": "2026-10-13"}
    try:
        r = requests.get(url, timeout=55, headers={"User-Agent": "SatelliteAttitudeStudentReview/1.0 (public-reference-verification)"})
        record.update(status=r.status_code, resolved_url=r.url, content_type=r.headers.get("Content-Type", ""))
        if r.status_code != 200:
            record.update(verified=False, error=f"HTTP {r.status_code}")
            return record
        is_pdf = r.content.startswith(b"%PDF")
        ext = ".pdf" if is_pdf else ".json" if "json" in record["content_type"] else ".html"
        path = CACHE / (sid + ext)
        path.write_bytes(r.content)
        record.update(cache_file=str(path.relative_to(HERE)), sha256=hashlib.sha256(r.content).hexdigest(), bytes=len(r.content))
        if is_pdf:
            out = CACHE / (sid + ".txt")
            subprocess.run(["pdftotext", "-layout", str(path), str(out)], check=True, capture_output=True)
        elif ext == ".json":
            (CACHE / (sid + ".txt")).write_text(json.dumps(r.json(), ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            parser = TextParser()
            parser.feed(r.content.decode(r.encoding if r.encoding and r.encoding.lower() != "iso-8859-1" else "utf-8", errors="replace"))
            plain = "".join(parser.parts)
            plain = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", plain)
            (CACHE / (sid + ".txt")).write_text(plain, encoding="utf-8")
        record["verified"] = True
    except Exception as e:
        record.update(verified=False, error=f"{type(e).__name__}: {e}")
    return record


def extract_originals():
    originals = ROOT.parent / "项目报告"
    pdf = next(originals.glob("*.pdf"))
    subprocess.run(["pdftotext", "-layout", str(pdf), str(HERE / "原立项全文提取.txt")], check=True)
    docx = originals / "中期检查报告.docx"
    with zipfile.ZipFile(docx) as z:
        tree = ET.fromstring(z.read("word/document.xml"))
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paras = ["".join(t.text or "" for t in p.findall(".//w:t", ns)) for p in tree.findall(".//w:p", ns)]
    (HERE / "原中期全文提取.txt").write_text("\n".join(paras), encoding="utf-8")
    records = []
    for path in (pdf, docx):
        copied = ROOT / "01_遴选原始参考" / "04_立项与中期报告参考" / "项目报告" / path.name
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append({"original": str(path), "sha256": sha, "reference_copy": str(copied), "copy_identical": copied.exists() and hashlib.sha256(copied.read_bytes()).hexdigest() == sha})
    (HERE / "原报告保全核对.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    extract_originals()
    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:
        records = list(pool.map(fetch, SOURCES))
    (HERE / "文献获取清单.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    for r in records:
        print(r["id"], r["verified"], r.get("status", ""), r.get("bytes", ""), r.get("error", ""))
