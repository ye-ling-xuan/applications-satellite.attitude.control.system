"""Improve traditional references before held-out tests; learner base is fixed."""
from pathlib import Path
import sys
import numpy as np
from scipy.linalg import solve_continuous_are
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'代码'))
import flexible_study as study

def main():
    folder=HERE/'02_补充传统优化'
    study.write(folder/'预声明.json',dict(seed=16022,held_out_test_used=False,
                methods=['LQR_refined','ZVD_PID_refined'],ranking='fewest physical failures; most passes; lowest penalized cost',
                lqr_extra_random=500,lqr_extra_grid=75,zvd_extra_local=750,
                primary_family_comparisons_after_addition=36,learner_base_unchanged=True))
    rng=np.random.default_rng(16022)
    old=HERE.parent/'02_传统控制'
    selected=study.read(HERE/'02_传统控制'/'selected.json')
    wn,z=2*np.pi*.6,.025
    inv=1/study.JB+1/study.JF
    k,c=wn*wn/inv,2*z*wn/inv
    a=np.array([[0,1,0,0],[0,0,k/study.JB,c/study.JB],[0,0,0,1],[0,0,-wn*wn,-2*z*wn]],float)
    b=np.array([[0],[1/study.JB],[0],[-1/study.JB]])
    lqr=[]
    for qt in (1.,10.,100.):
        for qq in (1.,10.,100.,1000.,10000.):
            for r in (.1,1.,10.,100.,1000.):
                q=[qt,.3*qt,qq,.1*qq]
                p=solve_continuous_are(a,b,np.diag(q),[[r]])
                lqr.append(dict(method='LQR',gain=(b.T@p/r).ravel().tolist(),q_diag=q,r=r))
    for _ in range(500):
        q=np.exp(rng.uniform(np.log([.05,.005,.05,.001]),np.log([100.,100.,10000.,1000.])))
        r=float(np.exp(rng.uniform(np.log(.03),np.log(1000.))))
        p=solve_continuous_are(a,b,np.diag(q),[[r]])
        lqr.append(dict(method='LQR',gain=(b.T@p/r).ravel().tolist(),q_diag=q.tolist(),r=r))
    settings=selected['ZVD_PID']['settings']
    zvd=[]
    for _ in range(750):
        zvd.append(dict(method='ZVD_PID',kp=float(settings['kp']*np.exp(rng.normal(0.,.45))),
                        kd=float(settings['kd']*np.exp(rng.normal(0.,.45))),
                        ki=0. if rng.random()<.5 else float(settings['ki']*np.exp(rng.normal(0.,.7)))))
    dev,val=study.scenes('development'),study.scenes('validation')
    results={}
    for method,candidates in [('LQR',lqr),('ZVD_PID',zvd)]:
        original=study.read(old/f'{method}_search.json')['all_development']
        logs=[]
        for first in range(0,len(candidates),24):
            chunk=candidates[first:first+24]
            settings=dict(method=method)
            if method=='LQR':settings['gain']=np.repeat([s['gain'] for s in chunk],len(dev),axis=0)
            else:settings.update({name:np.repeat([s[name] for s in chunk],len(dev)) for name in ('kp','kd','ki')})
            metrics=study.simulate(dev*len(chunk),settings)
            for j,parameters in enumerate(chunk):
                score=study.summary({name:values[j*len(dev):(j+1)*len(dev)] for name,values in metrics.items()})
                logs.append(dict(index=len(original)+first+j,settings=parameters,development=score))
        rank=lambda value,split:(value[split]['physical_failures'],-value[split]['passes'],value[split]['selection_cost'],value['index'])
        ranked=sorted(original+logs,key=lambda value:rank(value,'development'))[:48]
        for value in ranked:value['validation']=study.summary(study.simulate(val,value['settings']))
        best=min(ranked,key=lambda value:rank(value,'validation'))
        results[method+'_refined']=best
        study.write(folder/f'{method}_search.json',dict(additional_development=logs,validation=ranked,selected=best))
        print(method,best,flush=True)
    study.write(folder/'selected.json',results)

if __name__=='__main__':main()
