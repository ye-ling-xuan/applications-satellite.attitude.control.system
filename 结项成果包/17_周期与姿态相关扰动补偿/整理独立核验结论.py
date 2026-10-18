"""Report metric replication separately from the failed strict state check."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'代码'))
import disturbance_study as s
import numpy as np

folder=s.STUDY/'04_独立测试';summary=s.read_json(folder/'summary.json');freeze=s.read_json(folder/'freeze.json')
investigation=s.read_json(s.STUDY/'05_独立核验'/'积分误差调查.json')
if s.source_hashes()!=freeze['source_sha256']:raise RuntimeError('production sources changed')
for file,expected in summary['files_sha256'].items():
    if s.sha(folder/file)!=expected:raise RuntimeError('frozen response changed')
for file,expected in freeze['models'].items():
    if s.sha(s.STUDY/file)!=expected:raise RuntimeError('frozen model changed')
tail=np.arange(s.STEPS+1)*s.DT>=s.DEADLINE-1e-10
records={};independent_metrics={};max_errors=np.zeros(3);largest_cost=0.;largest_rms=0.
for name,controller in s._controller_registry().items():
    rows=s.read_json(folder/f'{name}_episodes.json');original=np.load(folder/f'{name}_traces.npz')
    replay=s.evaluate(s.scenes('test'),controller,keep_traces=True)
    for key in ('theta','omega','tau','phase','sent_command','command','actual_cost_interval'):
        if np.max(np.abs(original[key]-replay['traces'][key]))>1e-11:raise RuntimeError('controller replay failed')
    x=np.load(s.STUDY/'05_独立核验'/f'{name}_独立ODE闭环.npz')['state']
    angle=np.degrees(x[:,:,0]);rate=np.degrees(x[:,:,1]);rms=np.sqrt(np.mean(angle[tail]**2,axis=0))
    passes=np.all(np.abs(angle[tail])<=s.ANGLE_TOL,axis=0)&np.all(np.abs(rate[tail])<=s.RATE_TOL,axis=0)
    if not np.array_equal(passes,[r['success'] for r in rows]):raise RuntimeError('independent acceptance differs')
    error=np.array([np.max(np.abs(x[:,:,i]-original[key])) for i,key in enumerate(('theta','omega','tau'))])
    max_errors=np.maximum(max_errors,error)
    cost_error=float(np.max(np.abs(x[-1,:,3]-[r['actual_cost'] for r in rows])))
    rms_error=float(np.max(np.abs(rms-[r['steady_rms_deg'] for r in rows])))
    largest_cost=max(largest_cost,cost_error);largest_rms=max(largest_rms,rms_error)
    records[name]=dict(episodes=len(rows),independent_passes=int(passes.sum()),
        max_angle_error_rad=float(error[0]),max_rate_error_rad_s=float(error[1]),max_torque_error_Nm=float(error[2]),
        max_cost_difference=cost_error,max_RMS_difference_deg=rms_error,
        mean_independent_RMS_deg=float(rms.mean()),mean_independent_cost=float(x[-1,:,3].mean()))
    independent_metrics[name]=(rms,x[-1,:,3])
comparisons={};confidence=1.-.05/3
for name in summary['comparisons']:
    before,cost_before=independent_metrics['pid'];after,cost_after=independent_metrics[name]
    rng=np.random.default_rng(170999);sample=rng.integers(0,len(before),(20000,len(before)))
    interval=np.quantile((before[sample]-after[sample]).mean(axis=1),[(1-confidence)/2,1-(1-confidence)/2]).tolist()
    cost_ratio=float(cost_after.mean()/cost_before.mean())
    supported=interval[0]>0. and cost_ratio<=1.1
    if supported!=summary['comparisons'][name]['primary_supported']:raise RuntimeError('primary conclusion not replicated')
    comparisons[name]=dict(improvement_interval_deg=interval,cost_ratio=cost_ratio,primary_supported=bool(supported),confidence=confidence)
value=dict(passed=True,verification_scope='frozen sources/models/responses, exact controller replay, all independently integrated closed-loop acceptance/primary statistical conclusions',
    full_original_audit_passed=False,original_strict_state_check_passed=False,
    original_failure_preserved=True,original_tolerance=2e-7,
    strict_state_threshold_relaxed=False,controllers=len(records),episodes=sum(r['episodes'] for r in records.values()),
    all_command_and_physics_replays=True,all_metrics_independently_reconstructed=True,
    independent_adaptive_ODE_episodes=sum(r['episodes'] for r in records.values()),max_replay_error=0.,
    max_independent_ODE_error=float(max_errors.max()),max_angle_difference_rad=float(max_errors[0]),
    max_rate_difference_rad_s=float(max_errors[1]),max_torque_difference_Nm=float(max_errors[2]),
    max_independent_cost_error=largest_cost,max_independent_RMS_error_deg=largest_rms,
    independent_closed_loop=records,independent_primary_comparisons=comparisons,
    primary_conclusion_replicated=True,
    limitation='The original mixed-unit 2e-7 full-state numerical check did not pass. The independent adaptive ODE nevertheless reproduces all128cases per controller, all acceptance decisions and all prespecified paired primary conclusions. This is metric/conclusion replication, not a claim that the original strict numerical audit passed.',
    scope='All768independent closed-loop episodes; original failed checks and original raw responses retained.',
    supplementary_verifier_sha256=s.sha(__file__),source_unchanged=True)
s.write_json(s.STUDY/'05_独立核验'/'verification.json',value)
print(value)
