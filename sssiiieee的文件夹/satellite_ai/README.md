# satellite_ai —— 强化学习卫星姿态控制（改进版）

本文件夹是从原项目强化学习代码**复制后改进**的版本。原代码位于 `satellite2AI`、
`cms的代码批注`、`yxy的学习笔记` 等目录，**均未做任何修改**。所有改进点都以
`ds:` 开头的注释标注，方便追溯。

项目推进逻辑链：**单轴（已收敛）→ 三轴（收敛不佳）→ 进一步改进 → 模型辨识 + MPC（补齐立项第②项）**。

## 目录结构

```
satellite_ai/
├── README.md / requirements.txt
├── explanation.ipynb            # 项目逻辑讲解（34 cells）
├── step.ipynb                   # 手动运行教程（12 cells）
├── common/
│   └── metrics.py               # 统一评估函数（单轴/三轴共用，ds10）
├── single_axis/                 # 单轴 RL 模块（已收敛）
│   ├── sat_env.py               # 环境（ds1~ds5）
│   ├── train.py                 # 训练（ds6~ds9）
│   ├── evaluate.py              # 评估 + OOD 测试（ds11、ds12）
│   └── output/                  # 单轴训练产出
│       ├── models/              # 模型（best_model.zip / sac_satellite_final.zip）
│       ├── tensorboard/         # 训练曲线
│       ├── train_log.txt
│       └── sac_ood_generalization.png
├── three_axis/                  # 三轴 RL 模块（改进中）
    ├── satellite3d.py           # 动力学（ds13，从 PID 侧复制）
    ├── quaternion_utils.py      # 四元数工具（ds13）
    ├── pid3d.py                 # 三轴 PID（ds13，用于对比）
    ├── sat_env3d.py             # 环境（ds14~ds17、ds21~ds23）
    ├── train3d.py               # 训练（ds18、ds22、ds24）
    ├── evaluate3d.py            # 评估 + PID vs RL 对比（ds19、ds20）
    └── output/
        ├── models/              # 三轴模型（线性奖励阶段）
        ├── models_v2/           # 三轴模型（课程学习+shaping 阶段）
        ├── tensorboard/
        └── sac_vs_pid_3d.png
├── mpc/                         # 模型辨识 + MPC 模块（ds27~ds29）
    ├── system_id.py             # 系统辨识（ds27）
    ├── mpc_controller.py        # MPC 控制器（ds28）
    ├── compare.py               # 三方对比 PID/MPC/SAC（ds29）
    └── identified_model.npz     # 辨识出的 A、B
└── visualization/               # 三轴姿态可视化（ds30）
    ├── animate_attitude.py      # 收敛动画脚本
    └── output/                  # sac_attitude.gif / pid_attitude.gif
```

## ds 改进点清单

**单轴（ds1~ds12）**

| 编号 | 内容 | 位置 |
|------|------|------|
| ds1 | 终止语义：超时用 `truncated` 而非 `terminated` | single_axis/sat_env.py |
| ds2 | 奖励系统化：去交叉项，加成功奖励 + 时间惩罚 | single_axis/sat_env.py |
| ds3 | 干扰真实化：偏置 + 周期正弦 + 噪声 | single_axis/sat_env.py |
| ds4 | 域随机化 | single_axis/sat_env.py |
| ds5 | 观测归一化 + info 保留真实物理量 | single_axis/sat_env.py |
| ds6 | 固定随机种子 | single_axis/train.py |
| ds7 | 新增 SAC 算法 | single_axis/train.py |
| ds8 | 训练/评估分离 + EvalCallback | single_axis/train.py |
| ds9 | VecNormalize 归一化奖励 | single_axis/train.py |
| ds10 | 统一评估指标模块 | common/metrics.py |
| ds11 | OOD 泛化评估 | single_axis/evaluate.py |
| ds12 | matplotlib 中文字体 | single_axis/evaluate.py |

**三轴（ds13~ds24）**

| 编号 | 内容 | 位置 |
|------|------|------|
| ds13 | 复制三轴动力学/四元数/PID（逻辑未改） | three_axis/satellite3d.py 等 |
| ds14 | 三轴 Gym 环境（误差四元数+角速度状态，三轴力矩动作） | three_axis/sat_env3d.py |
| ds15 | 三轴奖励（等效转角 + 角速度 + 能耗 + 成功/时间） | three_axis/sat_env3d.py |
| ds16 | 三轴域随机化 | three_axis/sat_env3d.py |
| ds17 | 三轴终止语义沿用 ds1 | three_axis/sat_env3d.py |
| ds18 | 三轴训练脚本（独立输出目录） | three_axis/train3d.py |
| ds19 | 三轴评估（等效转角复用 metrics） | three_axis/evaluate3d.py |
| ds20 | 三轴 PID vs RL 公平对比 | three_axis/evaluate3d.py |
| ds21 | 三轴奖励改【线性】惩罚（解决二次惩罚梯度消失） | three_axis/sat_env3d.py |
| ds22 | 课程学习：初始姿态 20°→90° 由易到难 | three_axis/train3d.py + sat_env3d.py |
| ds23 | potential-based shaping：李雅普诺夫势函数提供稠密梯度 | three_axis/sat_env3d.py |
| ds24 | 加大训练规模：100 万步 + 策略网络 [256,256] | three_axis/train3d.py |
| ds25 | 近似续训（SB3 无法无缝续训，结果退化，弃用） | three_axis/continue_train3d.py |
| ds26 | 单轴 PID vs RL 系统量化对比 | single_axis/compare_pid_rl.py |

**MPC（ds27~ds29）**

| 编号 | 内容 | 位置 |
|------|------|------|
| ds27 | 系统辨识：最小二乘拟合离散线性状态空间模型 A、B | mpc/system_id.py |
| ds28 | MPC 控制器：有限时域二次规划闭式解 + 滚动优化 + 饱和限幅 | mpc/mpc_controller.py |
| ds29 | 三方对比 PID vs MPC vs SAC | mpc/compare.py |

**可视化（ds30）**

| 编号 | 内容 | 位置 |
|------|------|------|
| ds30 | 三轴姿态收敛 3D 动画（SAC/PID 对比） | visualization/animate_attitude.py |

## 使用方法

虚拟环境在 `sssiiieee的文件夹/venv`，依赖见 `requirements.txt`。命令需在 `satellite_ai` 目录下运行：

```bash
cd satellite_ai
source ../venv/Scripts/activate        # 或直接用 ../venv/Scripts/python.exe

# ---- 单轴 ----
python single_axis/train.py sac        # 训练（约 10 分钟）
python single_axis/evaluate.py         # 评估（默认读 single_axis/output/models/）

# ---- 三轴 ----
python three_axis/train3d.py sac       # 训练（约 2 小时，100 万步）
python three_axis/evaluate3d.py three_axis/output/models_v2/sac_satellite3d_v2_final sac

# ---- 模型辨识 + MPC ----
python mpc/system_id.py               # 系统辨识
python mpc/compare.py                 # 三方对比 PID/MPC/SAC
python visualization/animate_attitude.py   # 三轴姿态收敛动画（GIF）
```

## 训练成果

**单轴（SAC，10 万步）**：所有初始条件（含 90°）均 `success=True`，稳定时间 2~4 秒。

| 初始条件 | 稳定时间 | 超调 | 成功 |
|----------|---------|------|------|
| 10°/0°/s | 1.98s | 17.8% | ✅ |
| 30°/0°/s | 2.68s | 6.1%  | ✅ |
| 60°/0°/s | 3.40s | 3.1%  | ✅ |
| 90°/0°/s | 3.94s | 2.0%  | ✅ |
| 30°/10°/s | 2.82s | 6.1%  | ✅ |

**三轴（迭代推进中）**，稳态误差（等效转角）：

| 阶段 | 做法 | 单轴稳态误差 | 多轴(45°/45°/45°) | 状态 |
|------|------|------------|-------------------|------|
| 阶段 1 | 二次惩罚，20 万步 | ~10.9° | ~10.9° | 收敛不佳 |
| 阶段 2 | 线性惩罚（ds21），30 万步 | ~2.7° | ~3.0° | 仍未能 <1° |
| 阶段 3 | 课程学习 + shaping（ds22~24），90.9 万步 | **~1.27°** | **177°（发散）** | 单轴逼近 <1°，多轴鲁棒性待解 |

> PID 参考（`pid3d.py`，Kp=3,Ki=0.5,Kd=1）：yaw/pitch 30° 能收敛（~0.4~0.8°），
> roll 30° 也无法 <1°；但 PID 在三轴 45°/45°/45°、60°/-60°/60° 上**不发散**（误差 1.8~2.9°），
> 鲁棒性优于当前 SAC。

**三轴 RL 结论（诚实）**：三轮迭代（二次→线性→课程+shaping+加训）单轴精度稳步改善
（10.9°→2.7°→1.27°），但多轴耦合大角度发散问题未解决，甚至因课程学习后期 ±90°/轴的
极端分布而加重。三轴 RL 仍未打赢 PID，瓶颈锁定在「多轴耦合大角度鲁棒性」。

**续训教训（ds25）**：SB3 的 `model.save()` 不保存回放缓冲区/VecNormalize 统计量，
从 `best_model.zip` 续训剩余 9% 后策略反而退化（单轴 1.27°→7.5°）。故续训产物
`sac_satellite3d_v2_final` 已弃用，**三轴最佳成果仍为 `models_v2/best_model.zip`**。

## 单轴 PID vs RL 系统量化对比（ds26）

脚本 `single_axis/compare_pid_rl.py`，在**同一动力学/初始条件/力矩限幅**下，对比网格整定出的
最优 PID（Kp=4, Ki=0, Kd=2）与训练好的 SAC。

**无干扰场景**：

| 指标 | PID | SAC | 谁更优 |
|------|-----|-----|--------|
| 稳定时间 90° | 4.56s | 3.94s | SAC |
| 超调量 90° | 14.2% | 2.0% | SAC |
| 控制能耗 90° | 5.30 | 1.91 | SAC |
| 稳态误差 | ~0.003° | ~0.43° | PID |

**结论（诚实）**：PID 赢在稳态精度（积分作用消除误差）；SAC 赢在速度、超调、能耗，
且更鲁棒（含周期干扰时 PID 稳态误差退化 250 倍，SAC 仅退化 2 倍）。
SAC 的 ~0.43° 残留误差源于二次惩罚接近目标时梯度消失（与三轴 ds21 同病根），
若单轴奖励也改线性惩罚，SAC 稳态误差有望逼近 0。

## 模型辨识 + MPC 三方对比（ds27~ds29）

脚本 `mpc/compare.py`：先由 `mpc/system_id.py` 用最小二乘辨识出离散线性模型
（A、B 误差 2e-16，机器精度），再基于该模型做有限时域 MPC，与 PID、SAC 三方对比。

**无干扰场景**（30° 初始）：

| 指标 | PID | MPC | SAC | 谁最优 |
|------|-----|-----|-----|--------|
| 稳定时间 | 2.72s | **1.66s** | 2.68s | MPC |
| 超调量 | 15.9% | **3.6%** | 6.1% | MPC |
| 稳态误差 | 0.002° | **0.0000°** | 0.43° | MPC |
| 控制能耗 | 1.12 | 1.87 | **0.90** | SAC |

**结论（诚实）**：MPC 基于精确辨识的模型，在速度、超调、稳态精度上全面最优，
代价是能耗略高（Q/R 权重可调）；SAC 最省能量但有 ~0.43° 残留；PID 简单但慢、超调大。
至此项目三条控制路线（经典 PID / 智能 RL / 预测 MPC）齐备，形成完整对比。

## 成果文件说明

| 文件 | 说明 |
|------|------|
| `single_axis/output/models/best_model.zip` | 单轴训练中评估得分最高的模型 |
| `single_axis/output/models/sac_satellite_final.zip` | 单轴最终模型 |
| `three_axis/output/models/` | 三轴线性奖励阶段的模型 |
| `three_axis/output/models_v2/` | 三轴课程学习 + shaping 阶段的模型（训练中） |
| `*/output/models/*/eval_logs/evaluations.npz` | 每次评估的奖励与回合长度记录 |
| `single_axis/output/sac_ood_generalization.png` | 单轴泛化测试曲线 |
| `three_axis/output/sac_vs_pid_3d.png` | 三轴 PID vs SAC 对比曲线 |
| `mpc/identified_model.npz` | 辨识出的离散模型 A、B |
| `mpc/pid_mpc_sac.png` | 单轴 PID vs MPC vs SAC 三方对比图 |
| `visualization/output/sac_attitude.gif` | 三轴 SAC 姿态收敛动画 |
| `visualization/output/pid_attitude.gif` | 三轴 PID 姿态收敛动画 |

## 下一步（尚未实现）

- 自动调参：Optuna 贝叶斯优化（升级随机搜索）
- 执行器模型：反作用轮（力矩限幅 + 速率限幅 + 动量饱和）
