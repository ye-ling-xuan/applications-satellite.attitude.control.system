"""Run one already declared formal seed in its own process.

This entry point only schedules the frozen train_run; it changes no model,
reward, validation rule, seed list, or step budget. Separate processes keep
PyTorch random streams independent when overlapping research stages.
"""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent/'代码'))
from actuator_study import protocol as p
from actuator_study.delivery import _completed, _metadata, selection_open
from actuator_study.learning import train_run

parser = argparse.ArgumentParser()
parser.add_argument('--kind', choices=p.KINDS, required=True)
parser.add_argument('--method', choices=p.METHODS, default='residual')
parser.add_argument('--seed', choices=p.FORMAL_SEEDS, type=int, required=True)
args = parser.parse_args()
selection_open()
decision = p.read_json(p.STUDY/'03_SAC训练'/'development_decisions'/f'{args.kind}_{args.method}.json')
if not decision['passed_gate']:
    raise RuntimeError('development gate did not pass')
path = _metadata(args.kind, args.method, decision['reward_name'], args.seed, 'formal')
if not _completed(path):
    train_run(args.kind, args.method, decision['reward_name'], args.seed, p.FORMAL_STEPS, 'formal')
