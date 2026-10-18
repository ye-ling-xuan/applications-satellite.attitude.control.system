"""Additional conservative PID search; development/validation only."""
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'代码'))
import flexible_study as study

def main():
    study.design()
    directory = HERE/'02_传统控制'/'补充搜索'
    declaration = dict(reason='original success-first PID has physical failures; add a zero-physical-failure conservative baseline',
                       source_hashes=study.sources(),held_out_test_used=False,
                       random_seed=16021,additional_random_candidates=2400,grid_candidates=225,
                       ranking='physical failures first; most passes; lowest penalized cost',
                       validation_shortlist=48)
    study.write(directory/'预声明.json',declaration)
    rng = np.random.default_rng(16021)
    originals = study.read(HERE/'02_传统控制'/'PID_search.json')['all_development']
    candidates = []
    for kp in np.geomspace(.025,6.,15):
        for kd in np.geomspace(.12,8.,15):
            candidates.append(dict(method='PID',kp=float(kp),kd=float(kd),ki=0.))
    old = study.read(HERE/'02_传统控制'/'selected.json')['PID']['settings']
    for j in range(2400):
        if j<1200:
            kp,kd = old['kp']*np.exp(rng.normal(0.,.65)),old['kd']*np.exp(rng.normal(0.,.65))
        else:
            kp,kd = np.exp(rng.uniform(np.log(.025),np.log(8.))),np.exp(rng.uniform(np.log(.12),np.log(10.)))
        ki = 0. if rng.random()<.8 else float(np.exp(rng.uniform(-9.,-3.)))
        candidates.append(dict(method='PID',kp=float(kp),kd=float(kd),ki=ki))
    dev,val = study.scenes('development'),study.scenes('validation')
    logs = []
    for first in range(0,len(candidates),24):
        chunk = candidates[first:first+24]
        settings = dict(method='PID',**{key:np.repeat([s[key] for s in chunk],len(dev)) for key in ('kp','kd','ki')})
        metrics = study.simulate(dev*len(chunk),settings)
        for j,candidate in enumerate(chunk):
            score = study.summary({key:value[j*len(dev):(j+1)*len(dev)] for key,value in metrics.items()})
            logs.append(dict(index=1200+first+j,settings=candidate,development=score))
    all_records = originals+logs
    rank = lambda value,split:(value[split]['physical_failures'],-value[split]['passes'],
                               value[split]['selection_cost'],value['index'])
    ranked = sorted(all_records,key=lambda value:rank(value,'development'))[:48]
    for value in ranked:
        value['validation'] = study.summary(study.simulate(val,value['settings']))
    best = min(ranked,key=lambda value:rank(value,'validation'))
    study.write(directory/'全部开发候选.json',logs)
    study.write(directory/'验证候选.json',ranked)
    study.write(directory/'selected_safe.json',best)
    print(best,flush=True)

if __name__=='__main__':main()
