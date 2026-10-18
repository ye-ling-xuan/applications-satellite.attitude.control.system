"""Independent adaptive-ODE reevaluation of every frozen three-axis controller."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'代码'))
import joint_study as s
import numpy as np
from scipy.integrate import solve_ivp

folder=s.STUDY/'04_独立测试';freeze=s.read_json(folder/'freeze.json');original=s.read_json(folder/'summary.json')
if s.source_hashes()!=freeze['source_sha256']:raise RuntimeError('frozen source changed')
for path,h in original['files_sha256'].items():
    if s.sha(folder/path)!=h:raise RuntimeError('original response changed')
for path,h in freeze['models'].items():
    if s.sha(s.STUDY/path)!=h:raise RuntimeError('model changed')
data=s.scenes('test');n=len(data);phase0=np.array([r['phase_rad'] for r in data]);frequency=np.array([r['frequency_rad_s'] for r in data])
coeff={k:np.array([r[k] for r in data]) for k in ('amplitude_1_Nm','amplitude_2_Nm','attitude_coefficient_Nm','rate_coefficient_Nm')}
inertia=np.array([1.2,1.,.8]);tail=np.arange(s.STEPS+1)*s.DT>=s.DEADLINE-1e-10
def rotation(q):
    q=q/np.linalg.norm(q,axis=-1,keepdims=True);q=np.where(q[:,:1]<0.,-q,q)
    length=np.linalg.norm(q[:,1:],axis=1);angle=2*np.arctan2(length,q[:,0])
    scale=np.divide(angle,length,out=np.full_like(length,2.),where=length>1e-12)
    return q[:,1:]*scale[:,None]
records={};metrics={};max_angle=0.;max_rate=0.;max_tau=0.;max_cost=0.
for name,controller in s.registry().items():
    original_trace=np.load(folder/f'{name}_traces.npz');rows=s.read_json(folder/f'{name}_episodes.json')
    state=np.zeros((n,11));state[:,:4]=original_trace['quaternion'][0];state[:,4:7]=np.radians([r['omega_deg_s'] for r in data])
    def measurement(t):
        phase=phase0+frequency*t
        return np.column_stack((-rotation(state[:,:4]),state[:,4:7],np.sin(phase),np.cos(phase),frequency))
    controller.reset(n,measurement(0.));trajectory=np.empty((s.STEPS+1,n,11));trajectory[0]=state
    for k in range(s.STEPS):
        command=controller.act(measurement(k*s.DT),np.full(n,k*s.DT))
        def derivative(t,flat):
            x=flat.reshape(n,11);q=x[:,:4];w=x[:,4:7];tau=x[:,7:10];theta=rotation(q);phase=phase0+frequency*t
            disturb=coeff['amplitude_1_Nm']*np.sin(phase)+coeff['amplitude_2_Nm']*np.sin(2*phase+.4)+coeff['attitude_coefficient_Nm']*np.sin(2*theta)+coeff['rate_coefficient_Nm']*np.tanh(w/.05)
            qdot=.5*np.column_stack((-np.sum(q[:,1:]*w,axis=1),q[:,:1]*w+np.cross(q[:,1:],w)))
            wdot=(tau+disturb-np.cross(w,inertia*w))/inertia
            return np.column_stack((qdot,wdot,(command-tau)/.15,np.sum(tau*tau,axis=1))).reshape(-1)
        state=solve_ivp(derivative,(k*s.DT,(k+1)*s.DT),state.reshape(-1),rtol=1e-10,atol=1e-12).y[:,-1].reshape(n,11)
        state[:,:4]/=np.linalg.norm(state[:,:4],axis=1,keepdims=True);trajectory[k+1]=state
    q=trajectory[:,:,:4];angles=np.degrees(2*np.arctan2(np.linalg.norm(q[:,:,1:],axis=-1),np.abs(q[:,:,0])));rates=np.degrees(np.linalg.norm(trajectory[:,:,4:7],axis=-1))
    passed=np.all(angles[tail]<=2.,axis=0)&np.all(rates[tail]<=1.,axis=0)
    if not np.array_equal(passed,[r['success'] for r in rows]):raise RuntimeError('independent acceptance changed')
    rms=np.sqrt(np.mean(angles[tail]**2,axis=0));cost=state[:,-1]
    angle_error=float(np.max(np.abs(angles-np.degrees(s.quaternion_angle(original_trace['quaternion'])))))
    rate_error=float(np.max(np.abs(trajectory[:,:,4:7]-original_trace['omega'])))
    tau_error=float(np.max(np.abs(trajectory[:,:,7:10]-original_trace['tau'])))
    cost_error=float(np.max(np.abs(cost-[r['actual_cost'] for r in rows])))
    max_angle=max(max_angle,angle_error);max_rate=max(max_rate,rate_error);max_tau=max(max_tau,tau_error);max_cost=max(max_cost,cost_error)
    records[name]=dict(episodes=n,passes=int(passed.sum()),max_attitude_angle_difference_deg=angle_error,max_rate_difference_rad_s=rate_error,max_torque_difference_Nm=tau_error,max_cost_difference=cost_error,mean_independent_RMS_deg=float(rms.mean()),mean_independent_cost=float(cost.mean()))
    metrics[name]=(rms,cost);np.savez_compressed(s.STUDY/'05_独立核验'/f'{name}_独立闭环ODE.npz',state=trajectory);print(dict(controller=name,**records[name]),flush=True)
comparisons={};confidence=1.-.05/3
for name,old in original['comparisons'].items():
    a,ca=metrics['pid'];b,cb=metrics[name];idx=np.random.default_rng(190999).integers(0,len(a),(20000,len(a)));ci=np.quantile((a[idx]-b[idx]).mean(axis=1),[(1-confidence)/2,1-(1-confidence)/2]).tolist();ratio=float(cb.mean()/ca.mean());allpass=old['all_pass'];supported=bool(allpass and ci[0]>0 and ratio<=1.1)
    if supported!=old['primary_supported']:raise RuntimeError('independent primary conclusion differs')
    comparisons[name]=dict(improvement_interval_deg=ci,cost_ratio=ratio,primary_supported=supported)
value=dict(passed=True,episodes=n*len(records),records=records,independent_comparisons=comparisons,primary_conclusion_replicated=True,max_angle_difference_deg=max_angle,max_rate_difference_rad_s=max_rate,max_torque_difference_Nm=max_tau,max_cost_difference=max_cost,source_unchanged=True,verifier_sha256=s.sha(__file__))
s.write_json(s.STUDY/'05_独立核验'/'独立高精度闭环复核.json',value);print(value)
