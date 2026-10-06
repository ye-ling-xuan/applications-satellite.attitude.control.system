# AI 卫星姿态控制

当前主线：**统一环境接口 → 单轴 PID + PPO → 三轴扩展 → 三种方法对比。**

目前已完成单轴 PID、独立 PPO 和奖励权重对照；已有组合初稿为 PD + PPO，完整 PID + PPO 和三轴统一训练尚待完成。

## 目录说明

| 位置 | 用途 |
|---|---|
| `attitude_control/` | 当前维护的仿真、控制、训练与评估代码 |
| `tests/` | 回归测试 |
| `artifacts/` | 选中的模型、配置、验证记录和关键实验结果 |
| `docs/` | 组内进展、实验说明和动力学参考 |
| `legacy/` | 旧 PPO 复现与三轴示例所需的少量文件 |
| `.vscode/` | 运行任务和调试配置 |
| `pyproject.toml` | Python 依赖与项目配置 |
| `AGENTS.md` | Codex 协作约定 |

根目录已移除旧学习文件夹、重复图片和模型、项目内旧环境以及临时输出。完整历史材料和原始轨迹保存在项目外的 `归档/2026-10-05目录整理/`，清单见其中的 `manifest.json`。缓存和空目录已删除。

## 运行

在 VS Code 中打开本项目目录，使用原项目的 `xingyu-part` 分支。本机解释器为 `D:\Users\admin\anaconda3\python.exe`。

在项目根目录的 PowerShell 终端运行：

```powershell
$satPython = 'D:\Users\admin\anaconda3\python.exe'
& $satPython -m pytest -q
& $satPython -m attitude_control.evaluate --tuning artifacts/tuning_v1/tuning.json --output artifacts/baselines_next_v1 --plot
& $satPython -m attitude_control.compare --left artifacts/ablation_r050_test10405_v1 --right artifacts/ablation_r005_test10405_v1 --output artifacts/torque_ablation_next_v1
```

已有非空输出目录不会被覆盖，再次运行请换新目录。其他电脑可创建 Python 环境后安装：`python -m pip install -e ".[rl,plot,dev]"`。

## 已有结果

PPO 结果为 5 次独立训练的平均值，每个模型测试 84 个场景。

| 方法 | 全部测试工况成功率 |
|---|---:|
| PID | 100% |
| PPO，力矩惩罚 0.5 | 42.9% |
| PPO，力矩惩罚 0.05 | 97.6% |

改进后的 PPO 在基础工况下平均约 4.95 秒稳定，PID 约 3.43 秒。当前还不能宣称 PPO 全面优于 PID。详细判据与统计在 `docs/实验说明.md`。

## 下一步

1. 完善共享接口，实现完整 PID + PPO 修正力矩。
2. 比较 PID、PPO、PID + PPO 在惯量变化和扰动下的表现。
3. 单轴验证后，接入三轴四元数模型。

组内同步材料：`docs/项目进展与后续推进方向.md`。
整理记录：`docs/目录整理说明.md`。
