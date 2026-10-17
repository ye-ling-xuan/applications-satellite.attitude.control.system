"""Reproduce two scientific PNG figures from study 14's saved records/traces.

Only this new summary directory is written. No training, simulation, selection
of a model, or modification of any study 12/13/14 artifact is performed.

Case rule, fixed before reading individual trajectories: among natural-test
scenario IDs, keep direct_PID collision=True and analytic_route_PID plus all
three DQN seeds success=True; choose the lexicographically first scenario ID.
This explanatory case is deliberately conditional on outcomes. It is not an
estimate of general performance; Figure 1 retains all assessment responses.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'14_姿态禁入区下的学习路线选择'/'04_独立测试'/'responses'
FIGURES = HERE/'图表'
METHODS = ('direct_PID', 'analytic_route_PID', 'DQN_seed_101', 'DQN_seed_202', 'DQN_seed_303')
LABELS = ('直接目标 PID', '解析路线 + PID', 'DQN 101 + PID', 'DQN 202 + PID', 'DQN 303 + PID')
COLORS = ('#59636f', '#18836b', '#3e6fb0', '#9e4a83', '#397f9e')
REGIONS = (('test', '自然测试', 512), ('detour', '绕行压力测试', 128), ('edge', '边缘压力测试', 128))
SELECTION_RULE = ('自然测试集中，直接目标PID发生碰撞，且解析路线与DQN种子101、202、303均成功；'
                  '满足条件的scenario_id按字典序排序，选第一例。该条件选例只解释机制，不替代总体结果。')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def configure_plotting():
    candidates = ('C:/Windows/Fonts/msyh.ttc',
                  'C:/Windows/Fonts/Noto Sans SC (TrueType).otf',
                  '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    for filename in candidates:
        if Path(filename).exists():
            font_manager.fontManager.addfont(filename)
            plt.rcParams['font.family'] = font_manager.FontProperties(fname=filename).get_name()
            break
    else:
        plt.rcParams['font.family'] = ['Noto Sans CJK SC', 'Noto Sans SC', 'SimHei', 'DejaVu Sans']
    plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 11,
                         'axes.labelsize': 12, 'axes.titlesize': 14,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': '#c7cdd3', 'text.color': '#192936',
                         'axes.labelcolor': '#253541', 'xtick.color': '#4b5862',
                         'ytick.color': '#253541', 'savefig.facecolor': 'white'})


def load_records():
    records, inputs = {}, {}
    summary_path = SOURCE/'summary.csv'
    summary = csv_rows(summary_path)
    inputs[summary_path] = digest(summary_path)
    for region, _, count in REGIONS:
        common_ids = None
        for method in METHODS:
            path = SOURCE/method/region/'episodes.csv'
            rows = csv_rows(path)
            if len(rows) != count or len({row['scenario_id'] for row in rows}) != count:
                raise ValueError(f'invalid complete response count: {method}/{region}')
            ids = {row['scenario_id'] for row in rows}
            if common_ids is not None and ids != common_ids:
                raise ValueError('methods do not share the same paired scenarios')
            common_ids = ids
            successes = sum(row['success'] == 'True' for row in rows)
            collisions = sum(row['collision'] == 'True' for row in rows)
            saved = next(row for row in summary if row['method'] == method and row['region'] == region)
            if (int(saved['count']) != count or int(saved['successes']) != successes
                    or int(saved['collisions']) != collisions):
                raise ValueError('episode records differ from the saved summary')
            records[(method, region)] = rows
            inputs[path] = digest(path)
    return records, inputs


def success_figure(records):
    figure, axes = plt.subplots(1, 3, figsize=(14.4, 6.7), sharey=True)
    figure.subplots_adjust(left=.153, right=.985, top=.73, bottom=.22, wspace=.14)
    figure.suptitle('禁入区路线选择：全部测试初态的任务成功率', y=.965, fontsize=19, fontweight='bold')
    figure.text(.5, .89, '同一固定 PID · 每回合仅选择一次路线 · 所有未成功样本均计入分母',
                ha='center', fontsize=12, color='#53616c')
    y = np.arange(len(METHODS))
    for axis, (region, title, count) in zip(axes, REGIONS):
        successes = np.array([sum(row['success'] == 'True' for row in records[(method, region)])
                              for method in METHODS])
        percentage = 100.*successes/count
        axis.barh(y, 100., height=.62, color='#f4e6e3', edgecolor='none', zorder=1)
        axis.barh(y, percentage, height=.62, color=COLORS, edgecolor='none', zorder=2)
        for row_index, (value, success) in enumerate(zip(percentage, successes)):
            text = f'{value:.1f}%  ·  {success}/{count}'
            if value >= 55.:
                axis.text(min(value-2., 97.), row_index, text, ha='right', va='center',
                          fontsize=11, fontweight='bold', color='white', zorder=3)
            else:
                axis.text(2., row_index, text, ha='left', va='center', fontsize=11,
                          fontweight='bold', color='#754039', zorder=3)
        axis.set_title(f'{title}\n{count} 个独立初态', pad=16, fontsize=14, linespacing=1.6)
        axis.set_xlim(0., 100.)
        axis.set_xticks([0, 25, 50, 75, 100], ['0%', '25%', '50%', '75%', '100%'])
        axis.set_yticks(y, LABELS)
        axis.grid(axis='x', color='#dce2e6', alpha=.7, linewidth=.7, zorder=0)
        axis.tick_params(axis='y', length=0, pad=12)
        axis.tick_params(axis='x', length=0, pad=8)
        axis.spines['left'].set_visible(False)
        axis.spines['bottom'].set_visible(False)
    axes[0].invert_yaxis()
    figure.legend(handles=[Patch(facecolor='#f4e6e3', label='浅红段：未成功（含碰撞）')],
                  loc='lower center', bbox_to_anchor=(.57, .112), frameon=False, fontsize=11)
    figure.text(.5, .066, '条内显示成功率与成功数/总数；自然分布为主结果，压力集单独报告。',
                ha='center', fontsize=10.5, color='#53616c')
    figure.text(.5, .023, '绕行：展开角差 >180°；边缘：至少一个端点靠近采样边界。解析路线三组均全部成功，DQN仍有未成功记录。',
                ha='center', fontsize=10.5, color='#53616c')
    destination = FIGURES/'图1_禁入区各方法完整成功率.png'
    figure.savefig(destination, dpi=200)
    plt.close(figure)
    return destination


def load_case(records, inputs):
    by_id = {method: {row['scenario_id']: row for row in records[(method, 'test')]}
             for method in METHODS}
    eligible = sorted(identifier for identifier, row in by_id['direct_PID'].items()
                      if row['collision'] == 'True'
                      and all(by_id[method][identifier]['success'] == 'True' for method in METHODS[1:]))
    if not eligible:
        raise ValueError('no paired example satisfies the declared selection rule')
    identifier = eligible[0]
    traces, indices = {}, {}
    for method in METHODS:
        rows = records[(method, 'test')]
        index = next(i for i, row in enumerate(rows) if row['scenario_id'] == identifier)
        indices[method] = index
        path = SOURCE/method/'test'/'traces.npz'
        with np.load(path, allow_pickle=False) as archive:
            traces[method] = {name: archive[name][:, index].copy()
                              for name in ('theta', 'omega', 'collision', 'command', 'interval_cost')}
            traces[method]['time'] = archive['time'].copy()
            traces[method]['population'] = archive['population'][index].copy()
            traces[method]['action'] = int(archive['action'][index])
        inputs[path] = digest(path)
    reference = traces['analytic_route_PID']
    for method in METHODS:
        np.testing.assert_array_equal(traces[method]['population'], reference['population'])
        np.testing.assert_array_equal(traces[method]['time'], reference['time'])
    overlap = all(np.array_equal(traces[method][name], reference[name])
                  for method in METHODS[2:] for name in ('theta', 'omega', 'command'))
    direct_row = by_id['direct_PID'][identifier]
    contact = int(direct_row['collision_interval'])
    time = traces['direct_PID']['time']
    if contact < 0 or not traces['direct_PID']['collision'][contact+1]:
        raise ValueError('inconsistent direct-PID contact record')
    np.testing.assert_array_equal(traces['direct_PID']['theta'][contact+1:],
                                  np.full(len(time)-contact-1, traces['direct_PID']['theta'][contact+1]))
    np.testing.assert_array_equal(traces['direct_PID']['command'][contact+1:], 0.)
    metadata = dict(selection_rule=SELECTION_RULE, eligible_natural_scenarios=len(eligible),
                    scenario_id=identifier, method_indices=indices,
                    initial_angle_deg=float(np.rad2deg(reference['population'][0])),
                    target_angle_deg=float(np.rad2deg(reference['population'][1])),
                    initial_rate_deg_s=float(np.rad2deg(reference['omega'][0])),
                    direct_collision_interval=contact,
                    direct_collision_time_interval_s=[float(time[contact]), float(time[contact+1])],
                    successful_trajectories_overlap_exactly=overlap,
                    rows={method: by_id[method][identifier] for method in METHODS})
    return traces, metadata


def case_figure(traces, metadata):
    figure, (attitude, torque) = plt.subplots(2, 1, figsize=(14.4, 9.3), sharex=True,
                                            gridspec_kw={'height_ratios': [2.5, 1.]})
    figure.subplots_adjust(left=.085, right=.975, top=.81, bottom=.20, hspace=.17)
    figure.suptitle('配对轨迹代表案例（仅用于解释，不能代替总体结果）',
                   y=.967, fontsize=18, fontweight='bold')
    figure.text(.5, .915, f"{metadata['scenario_id']}  ·  初始 {metadata['initial_angle_deg']:.2f}°"
                f" → 目标 {metadata['target_angle_deg']:.2f}°  ·  初始角速度 0°/s", ha='center', fontsize=12)
    direct = traces['direct_PID']
    time = direct['time']
    contact = metadata['direct_collision_interval']
    completion = contact+1
    contact_start, contact_end = metadata['direct_collision_time_interval_s']
    reference = traces['analytic_route_PID']
    target = metadata['target_angle_deg']
    for lower, upper in [(-200., -160.), (160., 200.)]:
        attitude.axhspan(lower, upper, facecolor='#f7d9d4', alpha=.8, zorder=0)
        attitude.axhline(lower, color='#ce8d82', linewidth=.6, zorder=1)
        attitude.axhline(upper, color='#ce8d82', linewidth=.6, zorder=1)
        attitude.text(21.6, (lower+upper)/2., '禁入区', ha='right', va='center',
                      color='#9d4d40', fontsize=11)
    attitude.axhline(target, color='#375168', linewidth=1.1, linestyle=(0, (3, 3)), label='目标姿态')
    attitude.plot(time[:completion+1], np.rad2deg(direct['theta'][:completion+1]),
                  color=COLORS[0], linewidth=2.4, label='直接目标 PID：碰撞前及首碰间隔')
    attitude.plot(time[completion:], np.rad2deg(direct['theta'][completion:]),
                  color='#89939b', linewidth=1.7, linestyle=(0, (4, 3)), label='直接 PID：失败后记录填充')
    successful_label = '解析路线 + 三个 DQN：成功轨迹完全重合'
    if metadata['successful_trajectories_overlap_exactly']:
        attitude.plot(time, np.rad2deg(reference['theta']), color=COLORS[1], linewidth=2.4,
                      label=successful_label)
        torque.stairs(reference['command'], time, color=COLORS[1], linewidth=1.8,
                       label='解析路线 / DQN 101、202、303（重合）')
    else:
        for method, color, label in zip(METHODS[1:], COLORS[1:], LABELS[1:]):
            attitude.plot(time, np.rad2deg(traces[method]['theta']), color=color, linewidth=1.8, label=label)
            torque.stairs(traces[method]['command'], time, color=color, linewidth=1.4, label=label)
    collision_angle = float(np.rad2deg(direct['theta'][completion]))
    attitude.scatter([contact_end], [collision_angle], marker='x', color='#b84439',
                     linewidth=2., s=65, zorder=5)
    attitude.annotate(f'首次触碰位于 {contact_start:.2f}–{contact_end:.2f} s\n标记为完整首碰间隔的记录端点',
                      xy=(contact_end, collision_angle), xytext=(5.0, -80.),
                      arrowprops=dict(arrowstyle='->', color='#b84439', linewidth=1.),
                      fontsize=10.5, color='#9d4d40', ha='left', va='center')
    attitude.annotate('失败后状态冻结\n虚线为填充，不能解释为停稳',
                      xy=(15., collision_angle), xytext=(13.5, -65.),
                      arrowprops=dict(arrowstyle='->', color='#89939b', linewidth=1.),
                      fontsize=10.5, color='#69757e', ha='center', va='center')
    attitude.set_ylim(-215., 215.)
    attitude.set_yticks([-200., -160., -80., 0., 80., 160., 200.])
    attitude.set_ylabel('展开角度 / °')
    attitude.legend(loc='upper left', bbox_to_anchor=(0., 1.17), ncol=2, frameon=False,
                    fontsize=10.3, columnspacing=2., handlelength=2.8)
    torque.stairs(direct['command'][:completion], time[:completion+1], color=COLORS[0],
                  linewidth=1.9, label='直接目标 PID：实际传播区间')
    torque.stairs(direct['command'][completion:], time[completion:], color='#89939b',
                  linewidth=1.4, linestyle=(0, (4, 3)), label='失败后零命令填充')
    torque.axhline(0., color='#b9c3cb', linewidth=.8, zorder=0)
    torque.set_ylim(-1.1, 1.1)
    torque.set_yticks([-1., -.5, 0., .5, 1.])
    torque.set_ylabel('力矩命令 / N·m')
    torque.set_xlabel('时间 / s')
    torque.annotate('碰撞后命令归零，仅为失败填充', xy=(14., 0.), xytext=(14., -.62),
                    arrowprops=dict(arrowstyle='->', color='#89939b', linewidth=.9),
                    ha='center', fontsize=10.3, color='#69757e')
    for axis in (attitude, torque):
        axis.set_xlim(0., 22.)
        axis.set_xticks(np.arange(0., 23., 2.))
        axis.grid(axis='both', color='#dce2e6', linewidth=.7, alpha=.65, zorder=0)
    figure.text(.5, .123, f"选例规则：自然集里直接 PID 碰撞、其余四法均成功的 {metadata['eligible_natural_scenarios']} 例，按场景编号取第一例。",
                ha='center', fontsize=10.5, color='#53616c')
    figure.text(.5, .078, '四条成功轨迹完全相同，合并绘制且不人为错开；此例仅解释路线差异，总体证据见图1。',
                ha='center', fontsize=10.5, color='#53616c')
    figure.text(.5, .035, '该研究采用理想执行器，保存的命令即施加力矩（共同限幅 ±2 N·m）；碰撞终止后的低费用不代表节能。',
                ha='center', fontsize=10.5, color='#53616c')
    destination = FIGURES/'图2_禁入区配对代表轨迹.png'
    figure.savefig(destination, dpi=200)
    plt.close(figure)
    return destination


def main():
    configure_plotting()
    FIGURES.mkdir(parents=True, exist_ok=True)
    records, inputs = load_records()
    first = success_figure(records)
    traces, metadata = load_case(records, inputs)
    second = case_figure(traces, metadata)
    metadata['input_sha256'] = {path.relative_to(HERE.parent).as_posix(): value for path, value in inputs.items()}
    metadata['output_sha256'] = {path.relative_to(HERE).as_posix(): digest(path) for path in (first, second)}
    metadata['script_sha256'] = digest(__file__)
    metadata['source_artifacts_modified'] = False
    (HERE/'绘图数据与选例.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in metadata.items()
                      if key not in ('input_sha256', 'rows')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
