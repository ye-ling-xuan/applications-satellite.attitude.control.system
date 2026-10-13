"""Render every final PDF page for visual review with local Poppler."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/".artifact_deps"))
from PIL import Image,ImageOps,ImageDraw,ImageFont
from pypdf import PdfReader


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--reviews-only",action="store_true");args=parser.parse_args()
    closing=ROOT/"09_结项文书与答辩";out=ROOT/"tmp/artifacts/pdf_review";out.mkdir(parents=True,exist_ok=True)
    executable=Path("D:/LaTeX/texlive/2025/bin/windows/pdftoppm.exe")
    pdfs=sorted((closing/"06_可打印PDF").glob("*.pdf"))
    if args.reviews_only:pdfs=[p for p in pdfs if "综述" in p.name]
    records=[];all_images=[]
    for index,pdf in enumerate(pdfs):
        target=out/f"document_{index+1}";target.mkdir(exist_ok=True)
        subprocess.run([str(executable),"-r","110","-png",str(pdf),str(target/"page")],check=True,capture_output=True)
        pages=sorted(target.glob("page-*.png"),key=lambda p:int(p.stem.split("-")[-1]))
        expected=len(PdfReader(pdf).pages)
        if len(pages)!=expected:raise ValueError("Page render count mismatch")
        records.append({"pdf":str(pdf.relative_to(ROOT)),"pages":expected,"rendered_pages":[str(p.relative_to(ROOT)) for p in pages],"visual_review":"pending"})
        all_images.extend((index+1,k+1,p) for k,p in enumerate(pages))
    # Two pages per sheet retains readable text at full tool image detail.
    font=ImageFont.truetype("C:/Windows/Fonts/msyh.ttc",20)
    for start in range(0,len(all_images),2):
        sample=all_images[start:start+2];sheet=Image.new("RGB",(1840,1340),"#dce2e6");draw=ImageDraw.Draw(sheet)
        for col,(doc,page,path) in enumerate(sample):
            with Image.open(path) as im:
                im.thumbnail((890,1270));sheet.paste(im,(15+col*920,50))
            draw.text((15+col*920,12),f"文档{doc} 第{page}页",font=font,fill="black")
        sheet.save(out/f"contact_{start//2+1:02}.jpg",quality=92)
    (out/"render_manifest.json").write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"documents":len(records),"pages":len(all_images),"contact_sheets":(len(all_images)+1)//2}))


if __name__=="__main__":main()
