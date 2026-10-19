"""Build an offline report and verify figure provenance; no experiment is run."""
from pathlib import Path
import hashlib
import html
import json
import re
from urllib.parse import unquote, urlsplit

import markdown
from PIL import Image


PROJECT = Path(__file__).resolve().parents[2]
PACKAGE = PROJECT / '结题报告配图'
FIGURES = PACKAGE / '05_完整图表版'
REPORT = PROJECT / '结题报告内容_20261019.md'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path):
    item = {'path': path.relative_to(PROJECT).as_posix(),
            'bytes': path.stat().st_size, 'sha256': digest(path)}
    if path.suffix == '.png':
        with Image.open(path) as picture:
            picture.verify()
        with Image.open(path) as picture:
            item['pixels'] = list(picture.size)
    return item


def main():
    text = REPORT.read_text(encoding='utf-8')
    assert '\ufffd' not in text
    images = re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', text)
    assert len(images) == 11
    assert [int(re.search(r'图(\d+)_', path).group(1)) for _, path in images] == list(range(1, 12))
    assert re.findall(r'^\*\*图(\d+) ', text, re.M) == [str(i) for i in range(1, 12)]
    for _, path in images:
        assert (PROJECT / path).is_file(), path
        assert (PROJECT / path).with_suffix('.svg').is_file(), path

    # Verify the source records saved by the three plotting workflows.
    source_hashes = {}
    training_manifest = PACKAGE / '06_绘图代码与数据/训练与方案/来源与哈希.json'
    early_manifest = PACKAGE / '06_绘图代码与数据/前期结果/来源与核对记录.json'
    flex_manifest = PACKAGE / '06_绘图代码与数据/柔性结果/来源与导出清单.json'
    source_hashes.update(read_json(training_manifest))
    early = read_json(early_manifest)
    flex = read_json(flex_manifest)
    for item in early['sources']:
        source_hashes[item['path']] = item['sha256']
    for item in flex['inputs']:
        source_hashes[item['source']] = item['sha256']
        local_copy = flex_manifest.parent / item['local_copy']
        assert digest(local_copy) == item['sha256'], str(local_copy)
    for group in (early, flex):
        for item in group['outputs']:
            path = PROJECT / item.get('path', item.get('file'))
            assert digest(path) == item['sha256'], str(path)
    old_manifest = read_json(PACKAGE / '配图来源清单.json')
    for item in old_manifest['items'][:2]:
        source_hashes[item['source']] = item['source_sha256']
        assert digest(PACKAGE / item['copied_file']) == item['copy_sha256']
        assert digest(PACKAGE / item['png_export']) == item['png_export_sha256']
    for path, expected in source_hashes.items():
        assert digest(PROJECT / path) == expected, path

    body = markdown.markdown(text, extensions=['tables', 'fenced_code', 'sane_lists'])
    # Preserve each original caption and give every chart a direct navigation link.
    pattern = r'<p>(<img [^>]+>)</p>\s*(<p><strong>图(\d+) .*?</p>)'
    def figure(match):
        tag, caption, number = match.groups()
        src = re.search(r'src="([^"]+)"', tag).group(1)
        caption = caption.replace('<p>', '<figcaption>', 1).replace('</p>', '</figcaption>')
        return (f'<figure id="figure-{number}">'
                f'<a class="figure-link" href="{src}" target="_blank" rel="noopener" '
                f'title="打开图{number}原图">{tag}</a>{caption}</figure>')
    body, wrapped = re.subn(pattern, figure, body, flags=re.S)
    assert wrapped == 11
    labels = [
        '训练与测试数量', '柔性附件模型', '补偿控制结构', '完整训练验证曲线',
        '参数与验收设计', '刚体多算法对比', '惯量与执行器探索',
        '三组成功率', '三组控制代价', '配对统计区间', '独立核验',
    ]
    navigation = ''.join(f'<a href="#figure-{i}">图{i} {html.escape(label)}</a>'
                         for i, label in enumerate(labels, 1))
    page = '''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>结题报告内容 · 完整配图阅读版</title>
<style>
:root { color-scheme: light; }
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body { margin: 0; background: #edf2f6; color: #243444;
       font: 17px/1.9 "Microsoft YaHei", "PingFang SC", sans-serif; }
main { max-width: 1160px; margin: 24px auto; padding: 36px 42px;
       background: white; border: 1px solid #dce4eb; border-radius: 12px; }
header { border-bottom: 2px solid #235678; padding-bottom: 20px; margin-bottom: 26px; }
h1 { font-size: 30px; margin: 0 0 8px; color: #173e5c; }
header p { color: #52697a; font-size: 15px; margin: 0; }
a { color: #1f6086; text-decoration-thickness: 1px; text-underline-offset: 3px; }
h2 { color: #173e5c; margin: 42px 0 20px; font-size: 25px; }
h3 { color: #235678; font-size: 22px; }
h4 { color: #235678; font-size: 20px; margin-top: 34px; }
p { margin: 16px 0; overflow-wrap: anywhere; }
details { margin-top: 18px; background: #f3f7fa; padding: 12px 18px; border-radius: 8px; }
summary { cursor: pointer; font-size: 16px; color: #244e6b; }
nav { display: grid; grid-template-columns: repeat(3, 1fr); gap: 5px 18px;
      padding: 10px 0; font-size: 14px; }
figure { margin: 26px 0 38px; scroll-margin-top: 18px; }
.figure-link { display: block; border: 1px solid #e1e7ed; background: white; }
img { display: block; width: 100%; height: auto; }
figcaption { font-size: 14px; line-height: 1.8; padding: 12px 4px 0; color: #465968; }
figcaption strong { color: #244c68; }
footer { border-top: 1px solid #dce4eb; margin-top: 32px; padding-top: 15px; font-size: 14px; }
@media (max-width: 700px) {
 body { font-size: 16px; } main { margin: 0; padding: 22px 16px; border-radius: 0; }
 nav { grid-template-columns: 1fr 1fr; } h1 { font-size: 25px; }
}
@media print {
 body { background: white; font-size: 11pt; } main { border: 0; margin: 0; padding: 0; max-width: none; }
 details, footer, header p { display: none; } figure { break-inside: avoid; }
 h2, h3, h4 { break-after: avoid; } a { color: inherit; text-decoration: none; }
}
</style></head><body><main><header>
<h1>项目结题报告</h1>
<p>完整配图阅读版 · 11张图按正文顺序插入 · 点击图片查看原图</p>
<details><summary>按图快速定位</summary><nav>__NAV__</nav></details>
</header>__BODY__
<footer>配图位置与使用说明：<a href="结题报告配图/配图与插入位置说明.md">配图与插入位置说明</a>
 · <a href="结题报告配图/正文数据与图表对应.md">正文数据与图表对应</a>
 · <a href="结题报告配图/完整图表来源清单.json">完整图表来源清单</a></footer>
</main></body></html>'''.replace('__NAV__', navigation).replace('__BODY__', body)
    target = REPORT.with_suffix('.html')
    target.write_text(page, encoding='utf-8')

    sections = [
        '四、项目成果：六次训练、交互步数及两批测试数量',
        '五、2：单轴双惯量柔性附件模型',
        '五、2：PD反馈与PPO连续力矩补偿设计',
        '五、2：算法开发、正式训练和验证选模',
        '五、2：物理参数、三组测试与持续验收规则',
        '五、3：单轴/三轴刚体多算法及部署修正',
        '五、3：惯量随机化、一阶滞后与纯延迟探索',
        '五、3：三组九方法全部测试结果与越界',
        '五、3：三组全部方法的平均力矩平方积分',
        '五、3：十八项主要比较的配对校正区间',
        '五、3：独立轨迹重放与指标复算',
    ]
    training_sources = list(read_json(training_manifest))
    original_diagrams = [item['source'] for item in old_manifest['items'][:2]]
    early_sources = [item['path'] for item in early['sources']]
    flex_sources = [item['source'] for item in flex['inputs']]
    select = {
        1: [p for p in training_sources if p.endswith(('metadata.json', '最终结果.json', '结果摘要.json'))],
        2: [original_diagrams[0]], 3: [original_diagrams[1]],
        4: [p for p in training_sources if p.endswith('metadata.json')],
        5: [p for p in training_sources if p.endswith(('执行配置.json', '预声明.json'))],
        6: [p for p in early_sources if '/07_单轴三轴' in p],
        7: [p for p in early_sources if '/07_单轴三轴' not in p],
        8: [p for p in flex_sources if p.endswith(('summary.json', 'freeze.json'))],
        9: [p for p in flex_sources if p.endswith(('summary.json', 'freeze.json'))],
        10: [p for p in flex_sources if p.endswith(('paired_statistics.json', 'freeze.json'))],
        11: [p for p in flex_sources if p.endswith(('verification.json', 'supplement.py'))],
    }
    items = []
    for number, ((alt, relative), section) in enumerate(zip(images, sections), 1):
        png = PROJECT / relative
        items.append({'figure': number, 'title': alt, 'report_section': section,
                      'png': record(png), 'svg': record(png.with_suffix('.svg')),
                      'data_sources': [record(PROJECT / p) for p in select[number]]})
    manifest = {
        'date': '2026-10-19', 'version': '完整11图版',
        'scope': '现有记录制图与报告排版；未新增训练、调参或测试；原答辩讲解包与成果包未修改。',
        'report': [record(REPORT), record(target)], 'figure_count': len(items), 'figures': items,
        'source_record_files': [record(p) for p in (training_manifest, early_manifest, flex_manifest)],
        'verified_source_file_count': len(source_hashes),
        'historical_selection_manifest': '结题报告配图/配图来源清单.json',
    }
    (PACKAGE / '完整图表来源清单.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

    # Validate every report/package Markdown and report HTML local link, including the instructions.
    checked_links = 0
    missing = []
    documents = [REPORT, target, *PACKAGE.rglob('*.md')]
    for document in documents:
        content = document.read_text(encoding='utf-8')
        assert '\ufffd' not in content, str(document)
        if document.suffix == '.html':
            links = re.findall(r'(?:src|href)="([^"]+)"', content)
        else:
            links = re.findall(r'\[[^\]]*\]\(([^)]+)\)', content)
        for link in links:
            link = html.unescape(link.strip('<>'))
            url = urlsplit(link)
            if url.scheme or url.netloc or not url.path:
                continue
            path = document.parent / unquote(url.path)
            if not path.exists():
                missing.append({'document': str(document), 'link': link})
            checked_links += 1
    assert not missing, json.dumps(missing, ensure_ascii=False)
    check = {
        'date': '2026-10-19', 'report_image_count': 11, 'caption_order': list(range(1, 12)),
        'png_count': len(list(FIGURES.glob('*.png'))),
        'svg_count': len(list(FIGURES.glob('*.svg'))),
        'source_hashes_verified': len(source_hashes),
        'local_links_checked': checked_links, 'missing_links': missing,
        'png_decode': '11张均通过', 'utf8_replacement_characters': 0,
        'experiment_actions': '未重新训练、调参或测试。仅验证文件、链接和来源哈希。',
    }
    assert check['png_count'] == check['svg_count'] == 11
    (PACKAGE / '核对结果.json').write_text(json.dumps(check, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(check, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
