"""Execute predeclared stages with bounded CPU concurrency and retained logs."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'代码'))
from continuous_study import protocol as p


def jobs(commands, maximum):
    pending = list(commands)
    active = []
    failed = []
    log_dir = ROOT/'执行日志'
    log_dir.mkdir(exist_ok=True)
    while pending or active:
        while pending and len(active) < maximum:
            name, arguments = pending.pop(0)
            path = log_dir/(name+'.log')
            log = path.open('a', encoding='utf-8')
            environment = dict(os.environ, PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1',
                               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            process = subprocess.Popen([sys.executable, '-X', 'utf8', '-B', str(ROOT/'运行研究.py'), *arguments],
                                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=environment,
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            active.append((name, process, log, time.monotonic()))
            print(json.dumps(dict(event='started', name=name, pid=process.pid), ensure_ascii=False), flush=True)
        retained = []
        for name, process, log, started in active:
            status = process.poll()
            if status is None:
                retained.append((name, process, log, started))
                continue
            log.close()
            print(json.dumps(dict(event='finished', name=name, status=status,
                                  wall_time_s=round(time.monotonic()-started, 2)), ensure_ascii=False), flush=True)
            if status:
                failed.append(name)
        active = retained
        if active:
            time.sleep(1)
    if failed:
        raise RuntimeError('failed jobs; outputs preserved: '+', '.join(failed))


def run(maximum):
    p.design()
    for kind in p.KINDS:
        if not (ROOT/'02_传统控制'/f'selected_{kind}.json').exists():
            raise RuntimeError('finish fresh traditional tuning before scheduling: '+kind)
    snapshot = p.source_hashes()
    jobs([(f'sanity_{algorithm}', ['train', '--stage', 'sanity', '--algorithm', algorithm])
          for algorithm in p.ALGORITHMS], maximum)
    jobs([(f'development_{kind}_{algorithm}_A', ['train', '--kind', kind, '--algorithm', algorithm,
                                               '--reward', 'A', '--stage', 'development'])
          for kind in p.KINDS for algorithm in p.ALGORITHMS], maximum)
    jobs([(f'decision_A_{kind}', ['decide', '--kind', kind]) for kind in p.KINDS], maximum)
    requires_b = [kind for kind in p.KINDS if
                  p.read_json(ROOT/'03_训练记录'/'decisions'/f'{kind}.json')['reward_B_required']]
    if requires_b:
        jobs([(f'development_{kind}_{algorithm}_B', ['train', '--kind', kind, '--algorithm', algorithm,
                                                   '--reward', 'B', '--stage', 'development'])
              for kind in requires_b for algorithm in p.ALGORITHMS], maximum)
        jobs([(f'decision_AB_{kind}', ['decide', '--kind', kind]) for kind in requires_b], maximum)
    eligible = [kind for kind in p.KINDS if
                p.read_json(ROOT/'03_训练记录'/'decisions'/f'{kind}.json')['formal_eligible']]
    jobs([(f'formal_{kind}_seed_{seed}', ['train', '--kind', kind, '--stage', 'formal', '--seed', str(seed)])
          for kind in eligible for seed in p.FORMAL_SEEDS], maximum)
    if snapshot != p.source_hashes():
        raise RuntimeError('production code changed during training; do not continue to independent tests')
    for command in ('freeze', 'test', 'verify', 'report'):
        jobs([(command, [command])], 1)
    result = p.read_json(ROOT/'06_结果与说明'/'结果摘要.json')
    print(json.dumps(dict(event='study_complete', **result), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, choices=(1, 2, 3), default=3)
    run(parser.parse_args().workers)
