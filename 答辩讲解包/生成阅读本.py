"""Build an offline reading copy; no experiment or source data is modified."""
from __future__ import annotations
import html
import os
from pathlib import Path
import re
import sys
from urllib.parse import quote, urlsplit

PACK = Path(__file__).resolve().parent
PROJECT = PACK.parent
# Optional local build dependency; the produced HTML needs no package or network.
for parent in PACK.parents:
    dependency = parent / '工作区临时/文档渲染依赖'
    if dependency.is_dir():
        sys.path.insert(0, str(dependency))
        break


def relocate(text, parent):
    def replace(match):
        label, url = match.groups()
        url = url.strip().strip('<>')
        parts = urlsplit(url)
        if parts.scheme or url.startswith(('#', '//')):
            return match.group(0)
        from urllib.parse import unquote
        target = (parent / unquote(parts.path)).resolve()
        local = Path(os.path.relpath(target, PACK)).as_posix()
        if parts.fragment:
            local += '#' + parts.fragment
        return f'[{label}]({quote(local, safe="/#:.")})'
    return re.sub(r'\[([^\]]*)\]\(([^)]+)\)', replace, text)


def render(text):
    import markdown
    from latex2mathml.converter import convert
    equations = []
    pattern = re.compile(r'\\\[(.*?)\\\]|\$\$(.*?)\$\$|\\\((.*?)\\\)|(?<!\$)\$([^\n$]+)\$(?!\$)', re.S)
    def equation(match):
        raw = next(value for value in match.groups() if value is not None)
        display = match.group(1) is not None or match.group(2) is not None
        markup = convert(raw.strip(), display='block' if display else 'inline')
        token = f'MATHEQUATIONPLACEHOLDER{len(equations)}TOKEN'
        equations.append(markup)
        return '\n\n' + token + '\n\n' if display else token
    text = pattern.sub(equation, text)
    output = markdown.markdown(text, extensions=['tables', 'fenced_code', 'sane_lists'])
    for index, equation_html in enumerate(equations):
        token = f'MATHEQUATIONPLACEHOLDER{index}TOKEN'
        output = output.replace('<p>' + token + '</p>', equation_html).replace(token, equation_html)
    return output


def main():
    steps = sorted(p for p in PACK.iterdir() if p.is_dir() and re.match(r'^\d{2}_', p.name))
    assert len(steps) == 8
    navigation, articles = [], []
    for number, step in enumerate(steps, 1):
        title = html.escape(step.name[3:])
        navigation.append(f'<a href="#step-{number}"><b>{number:02d}</b> {title}</a>')
        text = relocate((step / '本步讲解.md').read_text(encoding='utf-8'), step)
        use = quote((step / '答辩与论文用稿.md').relative_to(PACK).as_posix(), safe='/')
        evidence = quote((step / '支撑代码与数据/材料说明.md').relative_to(PACK).as_posix(), safe='/')
        articles.append(f'<article id="step-{number}">{render(text)}<div class="next"><a href="{use}">本步答辩与论文用稿</a><a href="{evidence}">支撑代码与数据</a><a href="#top">返回目录</a></div></article>')
    css = '''*{box-sizing:border-box}body{margin:0;color:#1a3046;background:#edf2f6;font:17px/1.85 "Microsoft YaHei",system-ui,sans-serif}header{background:#16374e;color:white;padding:40px max(24px,calc((100vw - 1200px)/2))}header h1{font-size:32px;line-height:1.45;margin:8px 0}header p{color:#dfebf3;margin:8px 0}.layout{max-width:1320px;margin:24px auto;display:grid;grid-template-columns:240px minmax(0,1fr);gap:22px;padding:0 20px}nav{position:sticky;top:16px;align-self:start;background:white;border:1px solid #d5e0e9;border-radius:10px;padding:12px}nav a{display:block;text-decoration:none;padding:9px;font-size:14px;line-height:1.5}nav b{color:#175d91}a{color:#155c91}article{background:white;padding:28px 36px;border:1px solid #d5e0e9;border-radius:10px;margin-bottom:24px;scroll-margin-top:12px;overflow-wrap:anywhere}h1{font-size:27px}h2{font-size:22px;color:#1d5278;margin-top:30px}h3{font-size:19px}h1,h2,h3{line-height:1.5}table{border-collapse:collapse;font-size:14px;display:block;overflow:auto;line-height:1.7;margin:20px 0}td,th{border:1px solid #d5e0e9;padding:9px 11px;text-align:left}th{background:#edf4f9}img{max-width:100%;height:auto}blockquote,.intro{background:#edf5fa;border-left:4px solid #4181ac;padding:12px 20px;margin:20px 0}pre{background:#f1f4f7;padding:16px;overflow:auto;font:14px/1.7 Consolas,monospace}code{font-size:.92em}math[display="block"]{overflow-x:auto;overflow-y:hidden;padding:12px 0;max-width:100%;font-size:1.04em}.next{border-top:1px solid #d5e0e9;padding-top:16px;display:flex;gap:20px;flex-wrap:wrap;font-size:14px}footer{max-width:1100px;margin:25px auto;padding:0 20px;color:#567;font-size:14px}@media(max-width:860px){.layout{display:block}nav{position:static;margin-bottom:20px}article{padding:22px 18px}header h1{font-size:26px}}@media print{nav,.next{display:none}.layout{display:block;padding:0}body{background:white;font-size:11pt}header{background:white;color:#123;padding:0}header p{color:#456}article{border:0;break-before:page;padding:10px 0}}'''
    document = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>答辩讲解包：卫星姿态控制与柔性附件振动抑制</title><style>{css}</style></head><body><header id="top"><div>论文版 · 2026年10月18日</div><h1>卫星姿态控制与柔性附件振动抑制</h1><p>前期探索说明研究为何转向柔性附件；最终实验检验PPO学习补偿的收益与边界。</p><p><a style="color:white" href="论文正文初稿.md">论文正文初稿</a> · <a style="color:white" href="论文大纲与证据对照.md">论文大纲与证据对照</a></p></header><div class="layout"><nav>{''.join(navigation)}</nav><main><div class="intro">阅读顺序：01—04建立问题与铺垫，05—07是论文重点，08给出结论与写作安排。这里报告相对所比较PID的结果，同时保留LQR、输入整形等传统方法；全部结论以具体仿真范围为限。</div>{''.join(articles)}</main></div><footer>公式已转为原生MathML，可离线阅读。大体积模型与完整轨迹通过来源清单连接原工程；请将答辩讲解包与结项成果包保存在同一项目目录。</footer></body></html>'''
    (PACK / '按顺序阅读.html').write_text(document, encoding='utf-8')
    print('Built 8 research steps with offline MathML.')


if __name__ == '__main__':
    main()
