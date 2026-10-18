"""Revalidate every earlier selected traditional candidate before any test.

The primary script and its frozen training sources are unchanged.  Earlier
development rounds must not be silently discarded by a wider but coarser gain
grid.  This script accesses only development/validation; the PID used by the
learning environment is required to remain exactly unchanged.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'代码'))
import disturbance_study as s

selected_path=s.STUDY/'02_传统控制'/'selected.json'
selected=s.read_json(selected_path)
original_pid=selected['pid']['parameters'].copy()
archives=sorted((s.STUDY/'02_开发历史').glob('*/02_传统控制/selected.json'))
rounds=[s.read_json(path) for path in archives]+[selected]
evidence={}
for family in ('observer','harmonic'):
    candidates=[];seen=set()
    for item in rounds:
        parameter=item[family]['parameters']
        key=str(sorted(parameter.items()))
        if key not in seen:seen.add(key);candidates.append(parameter)
    harmonic=selected[family].get('harmonic')
    scores=[]
    for parameter in candidates:
        dev=s.summary(s.evaluate(s.scenes('development'),s.Controller(parameter,family,harmonic))['rows'])
        val=s.summary(s.evaluate(s.scenes('validation'),s.Controller(parameter,family,harmonic))['rows'])
        scores.append(dict(parameters=parameter,development=dev,validation=val,harmonic=harmonic))
    best=min(scores,key=lambda row:s.rank(row['validation']))
    selected[family]=dict(best,development_candidates=selected[family]['development_candidates'],
        validation_candidates=selected[family]['validation_candidates'],
        retained_earlier_selected_candidates=len(candidates),selection_used_heldout=False)
    evidence[family]=scores
if selected['pid']['parameters']!=original_pid:raise RuntimeError('PID must remain unchanged during training')
selected['retained_prior_development_candidates']=True
selected['retained_candidates_evidence_sha256']=None
evidence_path=s.STUDY/'02_传统控制'/'全开发轮次候选复核.json'
s.write_json(evidence_path,dict(evidence=evidence,source_sha256=s.source_hashes(),
    caller_sha256=s.sha(__file__),input_sha256={split:s.sha(s.STUDY/'01_方案与场景'/f'{split}.json')
    for split in ('development','validation')},test_read=False,PID_unchanged=True))
selected['retained_candidates_evidence_sha256']=s.sha(evidence_path)
s.write_json(selected_path,selected)
print({family:selected[family]['validation'] for family in ('observer','harmonic')})
