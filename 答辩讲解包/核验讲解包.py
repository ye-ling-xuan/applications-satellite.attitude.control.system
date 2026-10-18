"""Read-only evidence, metric and link checks; --seal writes a NEW version manifest."""
from __future__ import annotations
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit
import zipfile

PACK = Path(__file__).resolve().parent
PROJECT = PACK.parent
MANIFEST = PACK / '讲解包清单_柔性附件论文版.json'
RECORD = PACK / '核验结果_柔性附件论文版.json'
BASE = PROJECT / '结项成果包/16_柔性附体振动抑制连续控制/07_训练修正与复核'
SUPPLEMENT = BASE / '10_补充实验与论文复核'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def files():
    return sorted(p for p in PACK.rglob('*') if p.is_file() and p not in {MANIFEST, RECORD}
                  and '__pycache__' not in p.parts)


def source_check():
    steps = sorted(p for p in PACK.iterdir() if p.is_dir() and re.match(r'^\d{2}_', p.name))
    require([p.name[:2] for p in steps] == [f'{i:02}' for i in range(1,9)], 'Expected exactly steps 01 to 08')
    copies, references, copied_paths = 0, 0, set()
    for step in steps:
        for name in ('本步讲解.md', '答辩与论文用稿.md', '支撑代码与数据/材料说明.md', '支撑代码与数据/来源清单.json'):
            require((step/name).is_file(), 'Missing component: '+str(step/name))
        require('## 论文正文' in (step/'答辩与论文用稿.md').read_text(encoding='utf-8'), 'Missing paper section')
        for item in read(step/'支撑代码与数据/来源清单.json')['sources']:
            source = (PROJECT/item['source']).resolve()
            require(source.is_relative_to(PROJECT), 'Source outside project')
            require(source.is_file(), 'Missing source '+str(source))
            require(sha(source)==item['sha256'] and source.stat().st_size==item['bytes'], 'Source changed '+str(source))
            if item.get('reference_only'):
                references += 1
            else:
                target = (step/item['copy']).resolve()
                require(target.is_relative_to(step), 'Copy outside step')
                require(target.is_file() and sha(target)==item['sha256'], 'Copy differs '+str(target))
                copied_paths.add(target)
                copies += 1
    return dict(steps=len(steps), registered_copies=copies, registered_references=references), copied_paths


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []
    def handle_starttag(self, tag, attrs):
        self.urls.extend(value for key,value in attrs if key in {'href','src'} and value)


def link_check(copied):
    count, external_project = 0, 0
    missing = []
    for source in files():
        if source in copied or source.suffix not in {'.md','.html','.svg'}:
            continue
        text = source.read_text(encoding='utf-8-sig')
        if source.suffix in {'.html','.svg'}:
            parser=Links();parser.feed(text);urls=parser.urls
        else:
            urls=re.findall(r'\[[^\]]*\]\(([^)]+)\)',text)
        for url in urls:
            parts=urlsplit(url.strip().strip('<>'))
            if parts.scheme or not parts.path or url.startswith('//'):
                continue
            count+=1
            target=(source.parent/unquote(parts.path)).resolve()
            if not target.exists():
                missing.append([source.relative_to(PACK).as_posix(),url])
            external_project+=not target.is_relative_to(PACK)
    require(not missing,'Broken links: '+json.dumps(missing,ensure_ascii=False))
    page=(PACK/'按顺序阅读.html').read_text(encoding='utf-8')
    require('<math' in page and 'MathJax' not in page,'Offline math missing')
    require(all(f'id="step-{n}"' in page for n in range(1,9)), 'HTML step missing')
    return dict(local_links=count, references_to_full_project=external_project, broken_links=0, offline_math=True)


def numerical_check():
    import numpy as np
    summary=read(SUPPLEMENT/'结果摘要.json')
    require(summary['verification']['passed'],'Supplement physics audit failed')
    total=0;details=[]
    limits=np.deg2rad([1,.5,.1,.5])
    for group,methods in summary['results'].items():
        for method,reported in methods.items():
            path=SUPPLEMENT/'02_新独立测试'/group/method/'responses.npz'
            with np.load(path) as data:
                states=data['state'];commands=data['command'];failure=data['failed_trace'][:,-1]
                bounds=np.deg2rad([170,300,10])
                exceeded=np.any(np.abs(states[:,:,:3])>bounds[None,None,:],axis=(1,2))
                require(np.array_equal(exceeded,failure),'Model range failure differs')
                require(np.max(np.abs(commands))<=.5+1e-12,'Torque bound differs')
                success=np.all(np.abs(states[:,160:,:]) <= limits[None,None,:]+1e-12,axis=(1,2)) & ~failure
                cost=np.sum(commands**2*.05,axis=1)
                require(states.shape==(640,441,4) and commands.shape==(640,440),'Trajectory shape changed')
                require(np.array_equal(success,data['success']),'True state acceptance differs')
                np.testing.assert_allclose(cost,data['cost'],atol=1e-12,rtol=1e-12)
                require(int(success.sum())==reported['passes'],'Pass count differs')
                require(int(failure.sum())==reported['physical_failures'],'Failure count differs')
                require(abs(float(cost.mean())-reported['cost'])<1e-12,'Mean effort differs')
                total+=len(success)
                details.append(dict(group=group,method=method,passes=int(success.sum()),count=len(success)))
    require(total==17280 and total==summary['new_test_episodes'],'Episode total differs')
    require(summary['new_training_steps']==0 and not summary['old_models_changed'],'Unexpected training/model edits')
    freeze=read(SUPPLEMENT/'01_方案与冻结/freeze.json')
    for field,root in [('source_sha256',SUPPLEMENT),('old_inputs_sha256',BASE),('scenario_and_noise_sha256',SUPPLEMENT)]:
        for relative,digest in freeze[field].items():
            require(sha(root/relative)==digest,'Frozen input changed: '+relative)
    require(all(x['all_three_seeds_beat_both_PID_references'] for x in summary['claims'].values()), 'Main superiority flag differs')
    old=read(BASE/'08_最终报告/最终结果.json')
    require(old['training_steps']==4718592 and old['total_test_episodes_including_post_hoc']==6400,'Original flexible totals differ')
    return dict(recomputed_episodes=total,scope='All new trajectories: true-state acceptance, bound-failure flags and full-window torque-square integral; physical audit read and frozen inputs hashed. No retraining.',details=details)


def original_check():
    baseline=read(PACK/'原成果保全基线.json')
    for item in baseline['files']:
        path=PROJECT/item['source']
        require(path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],'Original baseline changed: '+str(path))
    counts={}
    for relative in ('最终交付清单.json','08_惯量域随机化与泛化边界研究/05_研究分析与核验/技术研究冻结清单.json'):
        listing=read(PROJECT/'结项成果包'/relative)
        for item in listing['files']:
            path=PROJECT/'结项成果包'/item['relative']
            require(path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],'Original frozen file changed: '+str(path))
        counts[relative]=len(listing['files'])
    return dict(passed=True,baseline=len(baseline['files']),files_rehashed=counts)


def archive_check():
    index=read(PACK/'版本记录/重组前资料索引.json')
    path=PACK/'版本记录'/index['archive']
    require(sha(path)==index['archive_sha256'],'Archive changed')
    with zipfile.ZipFile(path) as archive:
        require(archive.testzip() is None,'Archive CRC failure')
        require(not any(re.match(r'^(10_|11_)',p) for p in archive.namelist()),'Deleted steps retained in archive')
        require('按顺序阅读.html' not in archive.namelist(),'Old reading HTML retained')
        for item in index['files']:
            raw=archive.read(item['relative'])
            require(len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256'],'Archived content differs')
    return dict(files=index['file_count'],sha256=index['archive_sha256'],old_steps_10_11_included=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seal',action='store_true');parser.add_argument('--check-originals',action='store_true');args=parser.parse_args()
    if args.seal:
        require(not MANIFEST.exists(),'Version manifest already exists; do not overwrite it')
        RECORD.write_text(json.dumps(dict(status='pending',scope='First-version checks are running; no pass claimed yet'),ensure_ascii=False,indent=2),encoding='utf-8')
    sources,copied=source_check()
    result=dict(status='passed',sources=sources,links=link_check(copied),numbers=numerical_check(),archive=archive_check())
    if args.check_originals:
        result['originals']=original_check()
    if args.seal:
        rows=[dict(relative=p.relative_to(PACK).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files()]
        require(all(x['bytes']<100*1024**2 for x in rows),'File exceeds GitHub 100MiB threshold')
        listing=dict(version='柔性附件论文版_20261018',files=rows,file_count=len(rows),scope='Eight-step teaching/paper package; full sources referenced in the same project',excluded=['this manifest','核验结果_柔性附件论文版.json','__pycache__'])
        MANIFEST.write_text(json.dumps(listing,ensure_ascii=False,indent=2),encoding='utf-8')
    listing=read(MANIFEST)
    require({x['relative'] for x in listing['files']}=={p.relative_to(PACK).as_posix() for p in files()},'Unmanifested or missing file')
    for item in listing['files']:
        path=PACK/item['relative']
        require(path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],'Sealed file changed: '+item['relative'])
    result['manifest_files']=listing['file_count']
    RECORD.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({key:result[key] for key in ('status','sources','links','manifest_files')},ensure_ascii=False))


if __name__=='__main__':
    main()
