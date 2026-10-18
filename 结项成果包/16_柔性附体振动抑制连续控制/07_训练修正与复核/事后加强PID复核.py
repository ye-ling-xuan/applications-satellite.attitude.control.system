"""Explicit post-test stronger-PID check; select only from development/validation."""
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'代码'))
import flexible_study as study


def main():
    folder=HERE/'09_事后加强PID复核'
    original=HERE.parent/'02_传统控制'/'PID_search.json'
    additional=HERE.parent/'02_传统控制'/'补充搜索'/'全部开发候选.json'
    study.write(folder/'复核规则.json',dict(post_hoc=True,main_test_already_seen=True,
                learner_parameters_changed=False,candidate_count=3825,validation_shortlist=48,
                selection='development most passes, then penalized cost, then index; top48 validation same rule',
                selection_uses_test_outcomes=False,original_candidates_sha256=study.digest(original),
                additional_candidates_sha256=study.digest(additional),script_sha256=study.digest(__file__)))
    candidates=study.read(original)['all_development']+study.read(additional)
    if len(candidates)!=3825:raise RuntimeError('candidate coverage differs')
    rank=lambda value,split:(-value[split]['passes'],value[split]['selection_cost'],value['index'])
    short=sorted(candidates,key=lambda value:rank(value,'development'))[:48]
    validation=study.scenes('validation')
    for item in short:item['validation']=study.summary(study.simulate(validation,item['settings']))
    best=min(short,key=lambda value:rank(value,'validation'))
    study.write(folder/'验证候选.json',short)
    study.write(folder/'selected.json',best)
    # The test split and all learned-model test responses are read only after
    # selecting and serializing the traditional controller above.
    data=study.scenes('test')
    study.write(folder/'freeze.json',dict(settings=best['settings'],selected_sha256=study.digest(folder/'selected.json'),
                  test_initial_sha256=study.digest(HERE/'01_实验方案'/'test.json'),
                  production_source_hashes=study.sources(),post_hoc=True))
    metric,states,commands=study.simulate(data,best['settings'],record=True)
    np.savez_compressed(folder/'responses.npz',state=states,command=commands,**metric)
    f,g=study.independent_transition_matrices(data)
    failed=np.zeros(len(data),bool)
    maximum=0.
    for step in range(study.STEPS):
        expected=np.einsum('nij,nj->ni',f,states[:,step])+g*commands[:,step,None]
        expected[failed]=states[failed,step]
        residual=float(np.max(np.abs(expected-states[:,step+1])))
        maximum=max(maximum,residual)
        if residual>2e-11:raise RuntimeError('post-hoc traditional physics replay differs')
        x=states[:,step+1]
        failed|=((np.abs(x[:,0])>np.deg2rad(170.))|(np.abs(x[:,1])>np.deg2rad(300.))|
                 (np.abs(x[:,2])>np.deg2rad(10.)))
    success=np.all(np.abs(states[:,160:,:])<=np.deg2rad([1.,.5,.1,.5]),axis=(1,2))&~failed
    cost=np.einsum('nt,nt->n',commands,commands)*.05
    np.testing.assert_array_equal(metric['success'],success)
    np.testing.assert_allclose(metric['cost'],cost,rtol=0.,atol=2e-13)
    result=study.summary(metric)
    study.write(folder/'metrics.json',result)
    comparisons=[]
    for seed in study.FORMAL_SEEDS:
        with np.load(HERE/'04_独立测试'/f'PID_PPO_{seed}'/'responses.npz') as archive:
            difference=(archive['success'].astype(float)-success.astype(float)).reshape(5,128).mean(axis=0)
            rng=np.random.default_rng(16500+seed)
            values=np.concatenate([difference[rng.integers(0,128,(1000,128))].mean(axis=1) for _ in range(20)])
            low,high=np.quantile(values,[.05/42/2,1-.05/42/2])
            comparisons.append(dict(seed=seed,mean_success_difference=float(difference.mean()),
                 lower=float(low),upper=float(high),confidence=1-.05/42,
                 post_hoc=True,learned_physical_failures=int(archive['failed'].sum())))
    study.write(folder/'配对复核.json',comparisons)
    study.write(folder/'verification.json',dict(passed=True,episodes=len(data),transitions=len(data)*study.STEPS,
                max_state_residual=maximum,responses_sha256=study.digest(folder/'responses.npz'),
                selected_sha256=study.digest(folder/'selected.json'),script_sha256=study.digest(__file__),
                selection_test_separated_in_code=True,post_hoc=True))
    text=['# 事后加强传统PID对照的复核','',
          '主独立测试已经查看后，发现3825个候选中只对安全优先的PID完成了补充选择；原成功数优先PID仍来自初始1200个候选。因此补做这一项，避免把较弱的传统参数当作全部PID的能力。它是明确标注的事后复核，不伪装成事前声明的主比较。','',
          '选择程序只读取3825个候选的开发指标，按成功数优先筛48个，使用验证初态选择参数；参数写入文件后，才读取测试初态。没有修改或重新训练学习模型，也没有使用测试响应挑PID参数。','',
          f'新PID验证：{best["validation"]["passes"]}/320成功、{best["validation"]["physical_failures"]}物理失败。',
          f'新PID独立测试：{result["passes"]}/640={result["success"]:.2%}成功、{result["physical_failures"]}物理失败；平均力矩平方积分{result["cost"]:.7f}。',
          '', '三个冻结PPO模型仍分别640/640成功、0物理失败。配对复核采用42项校正的保守区间，但本节保留事后属性，不能替代新的前瞻独立确认。',
          '', '10°变形越界表示超出本研究线性近似的有效范围，不代表已证明真实卫星结构损坏。','']
    (folder/'复核说明.md').write_text('\n'.join(text),encoding='utf-8')
    print(best['validation'],result,comparisons,flush=True)


if __name__=='__main__':main()
