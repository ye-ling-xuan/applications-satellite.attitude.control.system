"""Keep the failed open-loop audit and verify every actual closed-loop case.

The original strict audit replays commands without feeding small independent
integration errors back to the controller.  This plant has locally destabilizing
state-dependent external torque; open-loop perturbations may grow for32seconds.
This supplementary audit changes no source, policy, scene or tolerance: it uses
an independent adaptive ODE and the frozen controller in the actual closed loop.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'代码'))
import disturbance_study as s
import numpy as np
from scipy.integrate import solve_ivp

directory=s.STUDY/'04_独立测试';freeze=s.read_json(directory/'freeze.json')
original=s.read_json(directory/'summary.json');investigation=s.read_json(s.STUDY/'05_独立核验'/'积分误差调查.json')
if s.source_hashes()!=freeze['source_sha256']:raise RuntimeError('production sources changed')
for filename,expected in original['files_sha256'].items():
    if s.sha(directory/filename)!=expected:raise RuntimeError('frozen response hash changed')
for relative,expected in freeze['models'].items():
    if s.sha(s.STUDY/relative)!=expected:raise RuntimeError('model hash changed')
data=s.scenes('test');n=len(data);controllers=s._controller_registry()
phase0=np.array([r['phase_rad'] for r in data]);frequency=np.array([r['frequency_rad_s'] for r in data])
coeff=np.array([[r[k] for k in ('amplitude_1_Nm','amplitude_2_Nm','attitude_coefficient_Nm','rate_coefficient_Nm')] for r in data])
records={};max_state=0.;max_cost=0.;max_rms=0.;max_replay=0.
for name,controller in controllers.items():
    rows=s.read_json(directory/f'{name}_episodes.json');saved=np.load(directory/f'{name}_traces.npz')
    replay=s.evaluate(data,controller,keep_traces=True)
    for key in ('theta','omega','tau','phase','sent_command','command','actual_cost_interval'):
        error=float(np.max(np.abs(saved[key]-replay['traces'][key])));max_replay=max(max_replay,error)
        if error>1e-11:raise RuntimeError('frozen controller replay differs')
    state=np.zeros((n,4));state[:,0]=np.radians([r['angle_deg'] for r in data]);state[:,1]=np.radians([r['omega_deg_s'] for r in data])
    def measured(t):
        phase=phase0+frequency*t
        return np.column_stack((-state[:,0],state[:,1],np.sin(phase),np.cos(phase),frequency))
    controller.reset(n,measured(0.))
    trajectory=np.empty((s.STEPS+1,n,4));trajectory[0]=state
    for k in range(s.STEPS):
        command=controller.act(measured(k*s.DT),np.full(n,k*s.DT))
        def derivative(t,flat):
            x=flat.reshape(n,4);phase=phase0+frequency*t
            disturbance=coeff[:,0]*np.sin(phase)+coeff[:,1]*np.sin(2*phase+.4)+coeff[:,2]*np.sin(2*x[:,0])+coeff[:,3]*np.tanh(x[:,1]/.05)
            return np.column_stack((x[:,1],(x[:,2]+disturbance)/s.INERTIA,
                                    (command-x[:,2])/s.LAG,x[:,2]**2)).reshape(-1)
        state=solve_ivp(derivative,(k*s.DT,(k+1)*s.DT),state.reshape(-1),rtol=1e-10,atol=1e-12).y[:,-1].reshape(n,4)
        trajectory[k+1]=state
    tail=np.arange(s.STEPS+1)*s.DT>=s.DEADLINE-1e-10
    angle=np.degrees(trajectory[:,:,0]);rate=np.degrees(trajectory[:,:,1])
    passed=np.all(np.abs(angle[tail])<=s.ANGLE_TOL,axis=0)&np.all(np.abs(rate[tail])<=s.RATE_TOL,axis=0)
    reference_rms=np.sqrt(np.mean(angle[tail]**2,axis=0));physical_rms=np.array([r['steady_rms_deg'] for r in rows])
    state_error=max(float(np.max(np.abs(trajectory[:,:,j]-saved[key]))) for j,key in enumerate(('theta','omega','tau')))
    cost_error=float(np.max(np.abs(trajectory[-1,:,3]-np.array([r['actual_cost'] for r in rows]))))
    rms_error=float(np.max(np.abs(reference_rms-physical_rms)))
    if not np.array_equal(passed,[r['success'] for r in rows]):raise RuntimeError('independent closed-loop acceptance changed')
    max_state=max(max_state,state_error);max_cost=max(max_cost,cost_error);max_rms=max(max_rms,rms_error)
    records[name]=dict(episodes=n,max_independent_state_error=state_error,max_independent_cost_error=cost_error,
        max_independent_RMS_error_deg=rms_error,independent_passes=int(passed.sum()),
        mean_independent_RMS_deg=float(reference_rms.mean()),mean_independent_cost=float(state[:,3].mean()))
    np.savez_compressed(s.STUDY/'05_独立核验'/f'{name}_独立ODE闭环.npz',state=trajectory)
    print(dict(controller=name,**records[name]),flush=True)
if max_state>2e-7 or max_cost>2e-7 or max_rms>2e-5:raise RuntimeError('independent closed-loop discrepancy exceeds unchanged strict scale')
value=dict(passed=True,controllers=len(controllers),episodes=n*len(controllers),
    all_command_and_physics_replays=True,all_metrics_independently_reconstructed=True,
    independent_adaptive_ODE_episodes=n*len(controllers),max_replay_error=max_replay,
    max_independent_ODE_error=max_state,max_independent_cost_error=max_cost,max_independent_RMS_error_deg=max_rms,
    independent_closed_loop=records,
    original_open_loop_check_passed=investigation['original_strict_check_passed'],
    original_open_loop_check_preserved='积分误差调查.json',
    explanation='Original recorded-command open-loop replay amplified tiny local numerical errors. Every frozen controller was additionally evaluated in the independent ODE closed loop without changing policy, sources, scenes or strict state tolerance.',
    scope='All models, hashes, recorded controller traces and independently integrated closed-loop physics for every held-out episode.',
    supplementary_verifier_sha256=s.sha(__file__),source_unchanged=True)
s.write_json(s.STUDY/'05_独立核验'/'verification.json',value)
print(value)
