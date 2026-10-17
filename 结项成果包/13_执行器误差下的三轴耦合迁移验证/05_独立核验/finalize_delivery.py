"""Assemble the final handoff from completed audits, without research execution."""
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
PACKAGE = STUDY.parent
AUDIT = STUDY / '05_独立核验'
TEST = STUDY / '04_独立测试'
MARKER = '\n## 最终交付复核与结果解释\n'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    freeze_path = TEST / 'freeze.json'
    frozen = read_json(freeze_path)
    semantic = read_json(AUDIT / 'summary_semantic_verification.json')
    saved = read_json(AUDIT / 'all_saved_verification.json')
    units = read_json(AUDIT / 'final_unit_verification.json')
    assert semantic['passed']
    assert semantic['expected_output_groups'] == semantic['verified_output_groups'] == 42
    assert saved['expected_output_groups'] == saved['verified_output_groups'] == 42
    assert all(row['passed'] for row in saved['all_saved_trajectories_metric_cost_limit_checks'])
    assert units['passed'] == units['tests'] == 17
    source_checks = {name: sha256(PACKAGE / name) == expected
                     for name, expected in frozen['source_and_input_hashes'].items()}
    assert all(source_checks.values())
    model_checks = {s['id']: sha256(Path(s['model_path'])) == s['model_sha256']
                    for s in frozen['controllers'] if s.get('model_path')}
    assert len(model_checks) == 8 and all(model_checks.values())
    single = next(PACKAGE.glob('12_*'))
    single_freeze_path = single / '04_独立测试' / 'freeze.json'
    assert sha256(single_freeze_path) == frozen['single_axis_freeze_sha256']
    single_checks = {name: sha256(single / name) == expected
                     for name, expected in read_json(single_freeze_path)['artifacts'].items()}
    assert all(single_checks.values())
    for spec in frozen['controllers']:
        if spec['method'] == 'retuned':
            selected = read_json(STUDY / '02_三轴PID整定' / spec['kind'] / 'selected.json')
            assert selected['parameters'] == spec['parameters']

    summaries = [read_json(TEST / spec['id'] / region / 'summary.json')
                 for spec in frozen['controllers']
                 for region in ('in_domain', 'ideal', 'out_of_domain')]
    episodes = sum(row['count'] for row in summaries)
    assert episodes == semantic['total_test_episodes'] == 17024
    assert sum(row['count'] for row in saved['all_saved_trajectories_metric_cost_limit_checks']) == episodes
    by_kind, by_region, by_stage = Counter(), Counter(), Counter()
    for row in summaries:
        by_kind[row['kind']] += row['count']
        by_region[row['region']] += row['count']
        by_stage[row['stage']] += row['count']
    searches = [r for kind in ('lag', 'delay') for r in read_json(STUDY / '02_三轴PID整定' / kind / 'all_attempts.json')]
    with (STUDY / '03_迁移验证' / 'traditional_validation_summaries.csv').open(encoding='utf-8-sig', newline='') as stream:
        validation_rows = list(csv.DictReader(stream))
    search_episodes = sum(int(r['count']) for r in searches)
    traditional_validation_episodes = sum(int(r['count']) for r in validation_rows)
    assert search_episodes == 314496 and traditional_validation_episodes == 1152
    domain = [r for r in summaries if r['region'] == 'in_domain']
    report_path = STUDY / '研究报告.md'
    appendix = [
        '',
        f'独立测试实际执行{episodes:,}回合，包含14个冻结控制器×3个区域，共42组；滞后8,064回合、纯延迟8,960回合。',
        '其中正式残差SAC两类执行器各三个种子、共六次运行对应7,296回合，传统控制器共7,296回合，预声明开发诊断模型共2,432回合。开发诊断模型不作为正式独立种子结论。',
        f'此前PID候选搜索记录{search_episodes:,}次开发/验证回合评估，六个传统控制器独立验证复评{traditional_validation_episodes:,}回合；它们与17,024回合测试分开计数。',
        '',
        '域内每个控制器均测试768回合。重新整定三轴PID在滞后组通过766/768、纯延迟组通过768/768，平均实际力矩平方积分分别为0.04113698、0.06967661 (N·m)²·s。滞后组两例未通过验收窗口，保留该结果，未据此重新调参。',
        '直接迁移原单轴PID在滞后/纯延迟域内分别通过504/768、494/768，模型补偿分别通过756/768、620/768。失败任务较低的全时域代价不能被解释为完成相同任务的节能收益。',
        '正式残差SAC的种子101/202/303在滞后域内分别通过36/768、768/768、701/768；在纯延迟域内分别通过119/768、3/768、0/768。三个种子均逐一报告，没有根据测试结果只保留优胜种子。',
        '滞后残差种子202虽在域内全部通过，平均实际代价为0.18301028 (N·m)²·s，约为重新整定PID的4.45倍；其理想执行器组为0/128、域外组为211/256。因此域内全通过不能推出跨区域的稳定迁移或节能优势。',
        '本次17,024回合均未触发预设物理失败阈值，但未通过规定的精度/速度/时间窗口仍计为任务失败。',
        '',
        '17项单元/独立物理测试通过；42组全部保存轨迹的指标、代价、指令限制与执行器时序复核通过。独立全动力学求解仅覆盖预定代表工况，并未逐条独立重积分全部研究轨迹。',
        '另以仅依赖Python标准库的脚本独立解析metrics.csv，用math.fsum与相对/绝对1e-12容差核对count、passes、physical_failures、success_rate、mean_actual_cost、mean_command_cost及max_settling_time_s，再核对每个summary.json和总表all_summaries.csv；42组均通过，分组及控制器-场景键无重复、场景覆盖完整。',
        '测试前冻结源码、场景、模型及12号研究冻结清单SHA在最终核验时仍一致。最终交付脚本只汇总已完成结果，没有执行新训练、搜索或测试。',
        '',
        '与原单轴研究相比，此处同时改变了惯量、初态维数、误差几何和整体范数验收要求，并引入三轴耦合。因此迁移变化不能单独归因于陀螺耦合项。本实验没有三轴联合训练策略，也不支持硬件部署或连续/无限时域稳定性的结论。',
        '',
        '复现时将test、verify分别作为独立Python命令运行：本机在SAC测试后同进程导入SciPy遇到OpenMP重复运行时；保留完整测试输出后，干净Python进程的verify已成功，无需允许重复运行时的环境变量。随后运行report、05_独立核验/verify_summary_semantics.py与05_独立核验/finalize_delivery.py重建最终说明及交付JSON。',
        ''
    ]
    base = report_path.read_text(encoding='utf-8').split(MARKER)[0].rstrip()
    report_path.write_text(base + '\n' + MARKER + '\n'.join(appendix), encoding='utf-8')
    verification_paths = [AUDIT / 'all_saved_verification.json', AUDIT / 'summary_semantic_verification.json',
                          AUDIT / 'final_unit_verification.json', AUDIT / 'postsearch_integration_verification.json',
                          AUDIT / 'verify_summary_semantics.py', Path(__file__), report_path,
                          freeze_path, TEST / 'all_summaries.csv']
    handoff = dict(
        study=STUDY.name, status='completed', created_utc=datetime.now(timezone.utc).isoformat(),
        joint_three_axis_training=False, further_tuning_after_test=False,
        controller_count=len(frozen['controllers']), expected_test_groups=42, verified_test_groups=42,
        actual_test_episodes=episodes, test_episodes_by_kind=dict(by_kind), test_episodes_by_region=dict(by_region),
        test_episodes_by_stage=dict(by_stage), total_physical_failures=sum(r['physical_failures'] for r in summaries),
        episode_accounting=dict(candidate_development_validation_evaluations=search_episodes,
            selected_traditional_validation_episodes=traditional_validation_episodes, held_out_test_episodes=episodes,
            total_recorded_research_episode_evaluations=search_episodes+traditional_validation_episodes+episodes,
            excludes_unit_test_fixtures=True, three_axis_RL_training_interactions=0),
        main_in_domain_results=domain,
        verification=dict(unit_tests_passed=17, all_saved_trajectory_metric_cost_limit_checks=True,
            all_summary_semantic_checks=True, complete_group_and_scene_coverage=True,
            duplicate_global_groups=False, all_sources_and_inputs_unchanged=all(source_checks.values()),
            all_eight_transferred_models_unchanged=all(model_checks.values()),
            source12_freeze_and_all_bound_artifacts_unchanged=all(single_checks.values()),
            independent_full_dynamics_scope='predeclared representative cases, not all saved research trajectories'),
        freeze_sha256=sha256(freeze_path), source12_freeze_sha256=sha256(single_freeze_path),
        scope_limitations=[
            'Independent axis deployment of frozen single-axis models; no joint three-axis RL training.',
            'Inertia, initial-state dimension, error geometry and overall-norm acceptance also changed; do not attribute degradation only to coupling.',
            'Seed variability is retained; development direct SAC is diagnostic only.',
            'Finite sampled acceptance window, not continuous-time or infinite-horizon stability or hardware qualification.',
            'Costs include unsuccessful tasks; lower unsuccessful-task cost is not a successful-task energy saving.'
        ],
        report_path=report_path.relative_to(STUDY).as_posix(),
        final_artifact_sha256={p.relative_to(STUDY).as_posix(): sha256(p) for p in verification_paths})
    write_json(STUDY / '交付汇总.json', handoff)
    print(json.dumps(dict(status=handoff['status'], episodes=episodes, groups=42,
                         report=str(report_path), handoff=str(STUDY / '交付汇总.json')), ensure_ascii=False))


if __name__ == '__main__':
    main()
