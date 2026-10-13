"""Build A4 reading copies from the auditable Markdown closing materials.

These PDFs do not claim to be the school's unknown final DOCX template.
Task-local PDF dependencies are explicitly allowed by the PDF skill.
"""
from __future__ import annotations
import argparse
import hashlib
import html
import json
import re
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/".artifact_deps"))
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER,TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle,getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,Preformatted,KeepTogether
from pypdf import PdfReader


def inline(text):
    text=html.escape(text.strip(),quote=False)
    # SimSun lacks several mathematical characters. Keep their original
    # notation and use an embedded math font for the whole adjoining run,
    # including the base of combining accents.
    missing_math=set("\u0307\u1d40\u207b\u2080\u2081\u2082\u2083\u2212\u223c\u2297")
    text=re.sub(r"[A-Za-z0-9_α-ωΑ-Ω\u0307\u1d40\u207b\u2080-\u2089\u2212\u223c\u2297]+",
                lambda m: f"<font name='ClosingMath'>{m.group(0)}</font>" if any(c in missing_math for c in m.group(0)) else m.group(0),text)
    text=re.sub(r"\*\*(.+?)\*\*",r"<b>\1</b>",text)
    text=re.sub(r"`([^`]+)`",r"<font name='Closing'>\1</font>",text)
    def link(m):
        label,url=m.group(1),m.group(2)
        if url.startswith(("https://","http://")):return f'<link href="{html.escape(url,quote=True)}" color="#126773">{label}</link>'
        return label
    text=re.sub(r"\[([^\]]+)\]\(([^)]+)\)",link,text)
    return text


def build(source,output):
    font=Path("C:/Windows/Fonts/simsun.ttc")
    if not font.exists():raise FileNotFoundError("Verified Chinese font required: "+str(font))
    if "Closing" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("Closing",str(font),subfontIndex=0))
        # Use a true Chinese bold font when installed; keep readable Song otherwise.
        bold=Path("C:/Windows/Fonts/simhei.ttf")
        pdfmetrics.registerFont(TTFont("ClosingBold",str(bold if bold.exists() else font),subfontIndex=0))
        pdfmetrics.registerFontFamily("Closing",normal="Closing",bold="ClosingBold",italic="Closing",boldItalic="ClosingBold")
        from matplotlib import get_data_path
        math_font=Path(get_data_path())/"fonts/ttf/DejaVuSans.ttf"
        pdfmetrics.registerFont(TTFont("ClosingMath",str(math_font)))
        pdfmetrics.registerFont(TTFont("ClosingMathBold",str(math_font.with_name("DejaVuSans-Bold.ttf"))))
        pdfmetrics.registerFontFamily("ClosingMath",normal="ClosingMath",bold="ClosingMathBold",italic="ClosingMath",boldItalic="ClosingMathBold")
    styles={
        "body":ParagraphStyle("Body",fontName="Closing",fontSize=11,leading=18,spaceAfter=7,wordWrap="CJK",textColor=colors.black),
        "title":ParagraphStyle("Title",fontName="ClosingBold",fontSize=19,leading=29,spaceAfter=17,wordWrap="CJK",textColor=colors.black),
        "h2":ParagraphStyle("Heading2",fontName="ClosingBold",fontSize=14,leading=21,spaceBefore=16,spaceAfter=9,keepWithNext=True,wordWrap="CJK",textColor=colors.black),
        "h3":ParagraphStyle("Heading3",fontName="ClosingBold",fontSize=12,leading=19,spaceBefore=11,spaceAfter=7,keepWithNext=True,wordWrap="CJK",textColor=colors.black),
        "cell":ParagraphStyle("Cell",fontName="Closing",fontSize=8.5,leading=13,wordWrap="CJK",textColor=colors.black),
        "code":ParagraphStyle("Code",fontName="Closing",fontSize=8,leading=12,wordWrap="CJK",spaceAfter=8),
        "caption":ParagraphStyle("Caption",fontName="Closing",fontSize=9,leading=14,alignment=TA_CENTER,spaceAfter=10,wordWrap="CJK"),
    }
    width=A4[0]-46*mm
    title=next((line.lstrip("# ") for line in source.read_text(encoding="utf-8").splitlines() if line.startswith("# ")),source.stem)
    doc=SimpleDocTemplate(str(output),pagesize=A4,rightMargin=23*mm,leftMargin=23*mm,topMargin=21*mm,bottomMargin=21*mm,
                         title=title,author="卫星姿态控制项目组",allowSplitting=True)
    lines=source.read_text(encoding="utf-8").splitlines();story=[];i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line==r"\[":
            equation=[];i+=1
            while i<len(lines) and lines[i].strip()!=r"\]":equation.append(lines[i].strip());i+=1
            i+=1
            tex=" ".join(equation).replace(r"\tfrac12",r"\frac{1}{2}").replace(r"\frac12",r"\frac{1}{2}")
            tex=tex.replace(r"\boldsymbol",r"\mathbf")
            tex=re.sub(r"\\(mathbf|mathbb|mathcal)\s*(\\[a-zA-Z]+|[A-Za-z])",r"\\\1{\2}",tex)
            import matplotlib
            matplotlib.use("Agg")
            from matplotlib.mathtext import math_to_image
            path=ROOT/"tmp/artifacts/equations"/(hashlib.sha256(tex.encode()).hexdigest()[:20]+".png")
            path.parent.mkdir(parents=True,exist_ok=True)
            math_to_image("$"+tex+"$",str(path),dpi=240,format="png",color="black")
            from PIL import Image as PILImage
            with PILImage.open(path) as im:w,h=im.size
            ratio=min(72/240,width/w)
            story.extend([Image(str(path),width=w*ratio,height=h*ratio),Spacer(1,9)])
            continue
        if line.startswith("```"):
            code=[];i+=1
            while i<len(lines) and not lines[i].strip().startswith("```"):code.append(lines[i]);i+=1
            # Wrapping code as paragraphs prevents clipping long workspace paths.
            for row in code:story.append(Paragraph(html.escape(row).replace(" ","&#160;"),styles["code"]))
            i+=1;continue
        if line.startswith("|"):
            table=[]
            while i<len(lines) and lines[i].strip().startswith("|"):
                fields=lines[i].strip().strip("|").split("|")
                if not all(re.fullmatch(r"\s*:?-+:?\s*",v) for v in fields):table.append([v.strip() for v in fields])
                i+=1
            n=max(map(len,table));table=[r+[""]*(n-len(r)) for r in table]
            # Content-derived widths with minimum room for Chinese wrapping.
            scores=np_array([max(3,min(20,max(len(r[j]) for r in table))) for j in range(n)])
            if n>=6:scores=np_array([max(4,min(16,s)) for s in scores])
            widths=[width*s/sum(scores) for s in scores]
            data=[[Paragraph(inline(v),styles["cell"]) for v in r] for r in table]
            t=Table(data,colWidths=widths,repeatRows=1,hAlign="LEFT")
            t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#eaf0f4")),("GRID",(0,0),(-1,-1),.4,colors.HexColor("#d9d9d9")),
                                   ("VALIGN",(0,0),(-1,-1),"MIDDLE"),("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
                                   ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
                                   ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f8fafb")])]))
            story.extend([t,Spacer(1,10)]);continue
        image=re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)",line)
        if image:
            path=(source.parent/image.group(2)).resolve()
            if not path.exists():raise FileNotFoundError(path)
            from PIL import Image as PILImage
            with PILImage.open(path) as im:w,h=im.size
            ratio=min(width/w,170*mm/h)
            group=[Image(str(path),width=w*ratio,height=h*ratio),Paragraph(inline(image.group(1)),styles["caption"])]
            if story and isinstance(story[-1],Paragraph) and story[-1].style.name=="Heading3":
                group.insert(0,story.pop())
            story.append(KeepTogether(group))
            i+=1;continue
        heading=re.match(r"^(#{1,6})\s+(.+)$",line)
        if heading:
            level=len(heading.group(1));s="title" if level==1 else "h2" if level==2 else "h3"
            story.append(Paragraph(inline(heading.group(2)),styles[s]));i+=1;continue
        para=[line];i+=1
        while i<len(lines) and lines[i].strip() and lines[i].strip()!=r"\[" and not re.match(r"^(#|\||```|!\[|[-*]\s|\d+\.\s)",lines[i].strip()):
            para.append(lines[i].strip());i+=1
        text=" ".join(para)
        text=re.sub(r"^[-*]\s+\[ \]\s*","□ ",text)
        text=re.sub(r"^[-*]\s+\[[xX]\]\s*","■ ",text)
        text=re.sub(r"^[-*]\s+","• ",text)
        story.append(Paragraph(inline(text),styles["body"]))
    def footer(canvas,document):
        canvas.saveState();canvas.setFont("Closing",8);canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(23*mm,12*mm,"卫星姿态控制项目  结项阅读稿")
        canvas.drawRightString(A4[0]-23*mm,12*mm,str(document.page));canvas.restoreState()
    output.parent.mkdir(parents=True,exist_ok=True);doc.build(story,onFirstPage=footer,onLaterPages=footer)
    reader=PdfReader(output)
    return {"source":str(source.relative_to(ROOT)),"output":str(output.relative_to(ROOT)),"pages":len(reader.pages),
            "source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"pdf_sha256":hashlib.sha256(output.read_bytes()).hexdigest(),
            "text_characters":sum(len(p.extract_text() or "") for p in reader.pages),"visual_qa":"pending",
            "school_template_applied":False,"format":"A4 Chinese reading copy"}


def np_array(values):return list(values)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--reviews-only",action="store_true");args=parser.parse_args()
    closing=ROOT/"09_结项文书与答辩";sources=sorted((closing/"01_两篇学习综述").glob("*.md"))
    sources=[p for p in sources if "综述" in p.name]
    if not args.reviews_only:
        sources.extend(sorted((closing/"02_结项报告").glob("*报告*.md")))
        sources.append(ROOT/"08_惯量域随机化与泛化边界研究"/"05_研究分析与核验"/"研究分析.md")
    records=[]
    for source in sources:
        if not source.exists():raise FileNotFoundError(source)
        target=closing/"06_可打印PDF"/(source.stem+".pdf")
        records.append(build(source,target));print(json.dumps(records[-1],ensure_ascii=False),flush=True)
    (closing/"06_可打印PDF"/"生成清单.json").write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__=="__main__":main()
