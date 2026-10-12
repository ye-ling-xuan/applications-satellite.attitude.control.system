"""Copy selected historical project files verbatim and record SHA256 provenance.

Run from any directory. Existing source files are never modified.
"""
from pathlib import Path
import hashlib
import json
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def selection():
    groups = {}
    def add(title, sources):
        groups.setdefault(title, set()).update(p for p in sources if p.is_file())
    w3 = ROOT / 'satellite1PID/week3搭建单轴卫星PID最小仿真系统'
    w4 = ROOT / 'satellite1PID/week4系统扩展：对单轴PID加入干扰噪声等实际影响'
    w5 = ROOT / 'satellite1PID/week5：三轴卫星姿态控制'
    add('01_PID建模与控制参考', [p for p in w3.glob('*.py') if p.name != 'tempCodeRunnerFile.py'])
    add('01_PID建模与控制参考', list(w4.glob('*.py')) + [w4 / 'Figure_1.png', w4 / '本周内容笔记/系统扩展介绍.md'])
    add('01_PID建模与控制参考', [p for p in w5.iterdir() if p.suffix in {'.py', '.markdown', '.png'}])
    cms = ROOT / 'cms的代码批注'
    add('01_PID建模与控制参考', [cms / f for f in ['satellite.py', 'pid.py', 'metrics.py', 'pid_tuning.py', '测试.py']])
    add('02_强化学习环境与实验参考', [cms / f for f in ['satellite_env.py', 'train_ppo.py', 'test_env.py']])
    add('02_强化学习环境与实验参考', (ROOT / 'satellite2AI/week6创建卫星Gym环境').glob('*.py'))
    notes = ROOT / 'yxy的学习笔记'
    for name in ['gpt改良版', '接入gym的RL训练']:
        folder = notes / name
        add('02_强化学习环境与实验参考', folder.glob('*.py'))
        add('03_历史模型日志与图表', [p for p in folder.iterdir() if p.suffix in {'.zip', '.png'}])
    add('02_强化学习环境与实验参考', [notes / '卫星姿态描述动力学运动学学习笔记.markdown', notes / '卫星姿态描述和动力学模型搭建.markdown', notes / 'RL基础学习.md'])
    add('03_历史模型日志与图表', [ROOT / 'ppo_satellite.zip', ROOT / 'ppo_satellite_final.zip', ROOT / 'ppo_satellite_improved.zip', ROOT / 'attitude_response.png', ROOT / 'satellite_response.png'])
    add('03_历史模型日志与图表', [p for p in (ROOT / 'logs').rglob('*') if p.suffix == '.npz' or p.name == 'best_model.zip'])
    for folder in ['satellite_tensorboard', 'ppo_sat_tensorboard']:
        add('03_历史模型日志与图表', (ROOT / folder).rglob('events.out.tfevents.*'))
    add('04_立项与中期报告参考', [ROOT / '项目报告/中期检查报告.docx', ROOT / '项目报告/《人工智能在卫星姿态控制系统中的应用》大一年度项目立项报告_20251026163255.pdf'])
    return groups


def main():
    records = []
    groups = selection()
    for title, sources in groups.items():
        for source in sorted(sources):
            relative = source.relative_to(ROOT)
            target = HERE / title / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            before = digest(source)
            shutil.copy2(source, target)
            copied = digest(target)
            after = digest(source)
            if before != copied or before != after:
                raise RuntimeError(f'Copy verification failed: {relative}')
            records.append({'category': title, 'source_relative': relative.as_posix(), 'copy_relative': target.relative_to(ROOT).as_posix(), 'sha256': before, 'copy_sha256': copied, 'size': source.stat().st_size, 'verified': True})
    manifest = {'schema_version': 1, 'file_count': len(records), 'total_bytes': sum(r['size'] for r in records), 'source_files_modified': False, 'copies_identical': True, 'files': records}
    (HERE / 'copy_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    explanations = {
        '01_PID建模与控制参考': '保留从单轴到三轴的动力学、控制器、现实因素扩展及指标实现，以展示模型建立和控制基线的演进。原版部分指标和四元数误差有问题，正式实验使用新版主线。',
        '02_强化学习环境与实验参考': '保留卫星 Gym 环境、PPO/DDPG 探索、GUI、参数比较、泛化测试及动力学笔记，作为方法来源和学习过程证据。历史 DDPG 脚本不等于已完成 DDPG 训练。',
        '03_历史模型日志与图表': '保留卫星 PPO 历史权重、评估文件、TensorBoard 日志和结果图。各模型的观测、动作、奖励、归一化规则可能不同，不可直接用于新版统一环境，也不可当作新版实验结果。',
        '04_立项与中期报告参考': '仅保留大一立项与中期报告，以核对原始目标和学习任务。大创计划书不纳入本成果包依据。',
    }
    lines = ['# 原始参考遴选清单', '', '所有文件按来源原样复制；不删除、不改写原文件。每份副本保留原目录层级。`copy_manifest.json` 记录来源、目标、大小和 SHA256，复制后逐份核验。', '', '这些文件属于历史参考。正式运行与最终模型请使用成果包的新版主线及其运行说明；历史脚本不会自动执行或自动加载模型。', '']
    for title, sources in groups.items():
        lines += [f'## {title}', '', explanations[title], '', '| 原始相对路径 | 复制大小（字节） |', '|---|---:|']
        for source in sorted(sources):
            lines.append(f'| `{source.relative_to(ROOT).as_posix()}` | {source.stat().st_size} |')
        lines.append('')
    lines += ['## 未纳入的内容', '', '排除 `.git`、`.conda`、缓存、临时运行文件、CartPole/Blackjack 通用练习及大量重复教学代码；CartPole 模型不属于卫星姿态控制成果。没有纳入大创报名材料和大创计划书。', '', f'本次共复制 {len(records)} 份文件，总计 {manifest["total_bytes"]:,} 字节，逐份 SHA256 相同。', '']
    (HERE / '遴选清单.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'files'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
