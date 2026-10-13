"""Build/render only the closing literature and team-reproduction appendix.

The four already reviewed main PDFs and the immutable research are untouched.
Visual pass must be recorded only after the generated pages have been opened.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/".artifact_deps"))
from tools.build_closing_pdfs import build
from PIL import Image,ImageDraw,ImageFont
from pypdf import PdfReader
SUPPLEMENT=ROOT/"09_结项文书与答辩/08_补充阅读与团队复核"
OUTPUT=SUPPLEMENT/"03_附录阅读PDF"
MANIFEST=OUTPUT/"生成清单.json"
RENDER=ROOT/"tmp/artifacts/supplement_pdf_review"


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")


def create():
    sources=sorted((SUPPLEMENT/"01_补充文献").glob("*札记.md"))
    sources+=sorted((SUPPLEMENT/"02_最小独立复核").glob("*说明.md"))
    if len(sources)!=2:raise ValueError(f"Expected two completed appendix sources, got {sources!r}")
    rows=[build(source,OUTPUT/(source.stem+".pdf")) for source in sources]
    save(MANIFEST,rows)
    print(json.dumps(rows,ensure_ascii=False,indent=2))


def render():
    rows=json.loads(MANIFEST.read_text(encoding="utf-8"));RENDER.mkdir(parents=True,exist_ok=True)
    records=[];images=[]
    for i,row in enumerate(rows,1):
        pdf=ROOT/row["output"];folder=RENDER/f"document_{i}";folder.mkdir(exist_ok=True)
        executable=Path("D:/LaTeX/texlive/2025/bin/windows/pdftoppm.exe")
        subprocess.run([str(executable),"-r","110","-png",str(pdf),str(folder/"page")],check=True,capture_output=True)
        pages=[folder/f"page-{j}.png" for j in range(1,len(PdfReader(pdf).pages)+1)]
        if not all(p.is_file() for p in pages):raise ValueError("Incomplete rendered appendix pages")
        images.extend((i,j,p) for j,p in enumerate(pages,1))
        records.append({"pdf":row["output"],"pdf_sha256":sha(pdf),"pages":len(pages),"rendered_pages":[{"relative":p.relative_to(ROOT).as_posix(),"sha256":sha(p)} for p in pages],"visual_review":"pending"})
    font=ImageFont.truetype("C:/Windows/Fonts/msyh.ttc",20)
    for start in range(0,len(images),2):
        sheet=Image.new("RGB",(1840,1340),"#dce2e6");draw=ImageDraw.Draw(sheet)
        for col,(doc,page,path) in enumerate(images[start:start+2]):
            with Image.open(path) as im:
                im.thumbnail((890,1270));sheet.paste(im,(15+920*col,50))
            draw.text((15+920*col,12),f"附录{doc} 第{page}页",font=font,fill="black")
        sheet.save(RENDER/f"contact_{start//2+1:02}.jpg",quality=92)
    save(RENDER/"render_manifest.json",records)
    print(json.dumps({"documents":len(rows),"pages":len(images),"contact_sheets":(len(images)+1)//2}))


def record_pass():
    rows=json.loads(MANIFEST.read_text(encoding="utf-8"))
    renders=json.loads((RENDER/"render_manifest.json").read_text(encoding="utf-8"))
    for row,record in zip(rows,renders,strict=True):
        if sha(ROOT/row["source"])!=row["source_sha256"]:raise ValueError("Source changed since rendering")
        if sha(ROOT/row["output"])!=record["pdf_sha256"]:raise ValueError("PDF changed since rendering")
        for page in record["rendered_pages"]:
            if sha(ROOT/page["relative"])!=page["sha256"]:raise ValueError("Page image changed")
        row["visual_qa"]="passed";record["visual_review"]="passed"
        record["reviewed_pages"]=list(range(1,record["pages"]+1))
    save(MANIFEST,rows);save(RENDER/"render_manifest.json",renders)
    save(OUTPUT/"visual_review.json",{"status":"passed","reviewer":"root","review_method":"All rendered appendix pages opened and visually inspected before --record-visual-pass; source and PDF hashes rechecked","documents":len(rows),"pages":sum(r["pages"] for r in rows),"records":renders})
    print("Recorded completed appendix page review")


def main():
    if hasattr(sys.stdout,"reconfigure"):sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build",action="store_true");mode.add_argument("--render",action="store_true")
    mode.add_argument("--record-visual-pass",action="store_true")
    args=parser.parse_args()
    create() if args.build else render() if args.render else record_pass()


if __name__=="__main__":main()
