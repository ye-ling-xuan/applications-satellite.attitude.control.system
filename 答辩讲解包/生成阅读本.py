"""Build the local HTML reading copy from the seven authored explanations."""
from __future__ import annotations
import html
from pathlib import Path
from urllib.parse import quote, urlsplit
import re

PACK = Path(__file__).resolve().parent


def relocate(text, parent):
    def replace(match):
        label, url = match.groups()
        if urlsplit(url).scheme or url.startswith(("#", "//")):
            return match.group(0)
        target = (parent / url.split("#", 1)[0]).resolve()
        try:
            local = target.relative_to(PACK).as_posix()
        except ValueError:
            # Original source excerpts may point back to the full workspace.
            import os
            local = Path(os.path.relpath(target, PACK)).as_posix()
        if "#" in url:
            local += "#" + url.split("#", 1)[1]
        return f"[{label}]({quote(local, safe='/#:.')})"
    return re.sub(r"\[([^\]]*)\]\(([^)]+)\)", replace, text)


def main():
    import markdown
    steps = sorted(p for p in PACK.iterdir() if p.is_dir() and re.match(r"^0[1-7]_", p.name))
    assert len(steps) == 7, "All seven steps must be complete before building the reading copy"
    links = []
    articles = []
    for number, step in enumerate(steps, 1):
        source = step / "本步讲解.md"
        assert source.is_file(), source
        text = relocate(source.read_text(encoding="utf-8"), step)
        rendered = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
        links.append(f'<a href="#step-{number}"><span>{number:02d}</span>{html.escape(step.name[3:])}</a>')
        script_link = quote((step / "答辩与论文用稿.md").relative_to(PACK).as_posix(), safe="/")
        evidence_link = quote((step / "支撑代码与数据/来源清单.json").relative_to(PACK).as_posix(), safe="/")
        articles.append(f'<article id="step-{number}">{rendered}<div class="next"><a href="{script_link}">查看本步答辩与论文用稿</a><a href="{evidence_link}">查看支撑材料来源</a><a href="#top">返回目录</a></div></article>')
    document = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>卫星姿态控制 · 答辩讲解包</title>
<style>
:root{color-scheme:light;--ink:#183044;--muted:#546677;--blue:#155c9c;--line:#dce5ed}*{box-sizing:border-box}body{margin:0;background:#eef3f7;color:var(--ink);font:17px/1.9 'Microsoft YaHei','Noto Sans CJK SC',sans-serif}header{background:#12334e;color:white;padding:48px max(24px,calc((100vw - 1100px)/2));}header h1{font-size:34px;line-height:1.4;margin:0 0 15px}header p{margin:6px 0;color:#d4e5f3}.tag{letter-spacing:2px;font-size:13px;color:#a8d5f5}.layout{max-width:1240px;margin:28px auto;display:grid;grid-template-columns:235px minmax(0,1fr);gap:24px;padding:0 20px}nav{position:sticky;top:16px;align-self:start;background:white;border:1px solid var(--line);border-radius:12px;padding:14px}nav a{display:block;color:var(--ink);text-decoration:none;font-size:14px;padding:9px;border-radius:6px;line-height:1.5}nav a:hover{background:#e9f3fc}nav span{color:var(--blue);font-weight:700;margin-right:10px}main{min-width:0}article{scroll-margin-top:16px;background:white;border:1px solid var(--line);border-radius:12px;padding:34px 40px;margin-bottom:25px;overflow-wrap:anywhere}h1{font-size:27px;line-height:1.5}h2{font-size:22px;line-height:1.5;margin-top:32px;color:#184f7c}h3{font-size:19px}p{margin:15px 0}a{color:var(--blue)}table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.7;display:block;overflow-x:auto;margin:20px 0}th,td{border:1px solid var(--line);padding:10px 12px;text-align:left;vertical-align:top}th{background:#eff5fa}tr:nth-child(even)td{background:#fafcfe}img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:6px}pre{background:#f1f5f8;padding:18px;white-space:pre-wrap;overflow:auto;font:14px/1.75 Consolas,'Microsoft YaHei',monospace}code{font-family:Consolas,'Microsoft YaHei',monospace;font-size:.92em}blockquote{margin:20px 0;padding:12px 20px;border-left:4px solid #3984ba;background:#f0f7fd}.intro{background:#e8f3fb;padding:20px;border-radius:8px;margin-bottom:25px;font-size:16px}.next{border-top:1px solid var(--line);margin-top:25px;padding-top:18px;display:flex;gap:20px;flex-wrap:wrap;font-size:14px}footer{max-width:900px;margin:10px auto 40px;color:var(--muted);font-size:14px;padding:0 20px}@media(max-width:850px){.layout{display:block}nav{position:static;margin-bottom:20px}article{padding:24px 20px}header h1{font-size:27px}}@media print{body{background:white;font-size:11pt}.layout{display:block;margin:0;padding:0}nav,.next{display:none}header{background:white;color:var(--ink);padding:10px 0}header p{color:var(--muted)}article{border:0;padding:10px 0;break-before:page}img,table{break-inside:avoid}}
</style></head><body><header id="top"><div class="tag">研究顺序 · 关键证据 · 通俗讲解</div><h1>卫星姿态控制：从单轴原型到惯量泛化</h1><p>按01—07阅读，理解每一步为什么做、怎么做、发现了什么。</p><p>编制日期：2026年10月16日 · 本次为整理讲解，未新增实验。</p></header><div class="layout"><nav>'''
    document += "".join(links)
    document += '</nav><main><div class="intro"><strong>先记住：</strong>我们控制的是卫星朝向。PID与强化学习在速度、误差、可靠性和控制代价上可能各有取舍。V1、V2、V3的条件不同，每一步都在自己的实验范围内解释结果。图表和表格后面都有支撑材料。</div>'
    document += "".join(articles)
    document += '</main></div><footer>本页由七步讲解文件生成。每一步的精华、口述稿与原样支撑副本仍保存在对应文件夹；完整原始工程保持原样。</footer></body></html>'
    (PACK / "按顺序阅读.html").write_text(document, encoding="utf-8")
    print(f"Built {len(steps)} steps into local reading HTML")


if __name__ == "__main__":
    main()
