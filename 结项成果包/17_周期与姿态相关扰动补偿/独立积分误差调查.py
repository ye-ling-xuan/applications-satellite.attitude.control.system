"""Diagnose a failed strict long-horizon independent ODE comparison."""
from pathlib import Path
import sys,math
sys.path.insert(0,str(Path(__file__).resolve().parent/'代码'))
import disturbance_study as s
import numpy as np
from scipy.integrate import solve_ivp

directory=s.STUDY/'04_独立测试';data=s.scenes('test')
freeze=s.read_json(directory/'freeze.json');records=[]
for name in freeze['controllers']:
    trace=np.load(directory/f'{name}_traces.npz')
    for i in (0,len(data)//2):
        r=data[i];x=np.array([np.radians(r['angle_deg']),np.radians(r['omega_deg_s']),0.])
        global_errors=[];local_errors=[];trajectory=[x.copy()]
        for k in range(s.STEPS):
            u=float(trace['command'][k,i])
            def rhs(t,x):
                phase=r['phase_rad']+r['frequency_rad_s']*t
                disturb=r['amplitude_1_Nm']*math.sin(phase)+r['amplitude_2_Nm']*math.sin(2*phase+.4)+r['attitude_coefficient_Nm']*math.sin(2*x[0])+r['rate_coefficient_Nm']*math.tanh(x[1]/.05)
                return [x[1],(x[2]+disturb)/s.INERTIA,(u-x[2])/s.LAG]
            initial=np.array([trace['theta'][k,i],trace['omega'][k,i],trace['tau'][k,i]])
            endpoint=np.array([trace['theta'][k+1,i],trace['omega'][k+1,i],trace['tau'][k+1,i]])
            x=solve_ivp(rhs,(k*s.DT,(k+1)*s.DT),x,rtol=1e-11,atol=1e-13).y[:,-1]
            local=solve_ivp(rhs,(k*s.DT,(k+1)*s.DT),initial,rtol=1e-11,atol=1e-13).y[:,-1]
            global_errors.append(np.abs(x-endpoint));local_errors.append(np.abs(local-endpoint));trajectory.append(x.copy())
        global_errors=np.array(global_errors);local_errors=np.array(local_errors);trajectory=np.array(trajectory)
        tail=np.arange(s.STEPS+1)*s.DT>=s.DEADLINE-1e-10
        original_rms=float(np.sqrt(np.mean(np.degrees(trace['theta'][tail,i])**2)))
        independent_rms=float(np.sqrt(np.mean(np.degrees(trajectory[tail,0])**2)))
        row=dict(controller=name,sample_id=i,max_global_state_error=global_errors.max(axis=0).tolist(),
            max_local_state_error=local_errors.max(axis=0).tolist(),
            max_global_error=float(global_errors.max()),max_local_error=float(local_errors.max()),
            original_steady_RMS_deg=original_rms,independent_steady_RMS_deg=independent_rms,
            absolute_RMS_difference_deg=abs(original_rms-independent_rms))
        records.append(row);print(row,flush=True)
result=dict(original_threshold=2e-7,original_strict_check_passed=all(r['max_global_error']<=2e-7 for r in records),
            records=records,max_global_error=max(r['max_global_error'] for r in records),
            max_local_error=max(r['max_local_error'] for r in records),
            max_absolute_RMS_difference_deg=max(r['absolute_RMS_difference_deg'] for r in records),
            sources_unmodified=s.source_hashes()==freeze['source_sha256'])
s.write_json(s.STUDY/'05_独立核验'/'积分误差调查.json',result)
print(result)
