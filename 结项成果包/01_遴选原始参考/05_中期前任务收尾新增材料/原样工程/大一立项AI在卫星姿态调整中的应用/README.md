# AI 在卫星姿态调整中的应用 —— 项目交接说明

本项目研究卫星姿态控制:一方面对比**传统 PID 控制**与**强化学习**两条技术路线,
另一方面在强化学习内部横向对比 **PPO / SAC / TD3** 三种算法。目前已完成:

- 单轴 / 三轴卫星姿态的 RL 环境搭建,以及 PPO / SAC / TD3 三种算法的训练;
- 0–90° 偏转纠正评测(测纠正时间、平均误差、稳态误差);
- **RL 与 PID 的对比**,包括无干扰干净对比和加干扰鲁棒性对比;
- **PPO / SAC / TD3 三种 RL 算法的对比**,详见
  [`RL与PID对比/三轴算法对比_结果与分析.md`](RL与PID对比/三轴算法对比_结果与分析.md)。

---

## 一、目前成果

### 1. 单轴强化学习(`单轴强化学习/`)

- **环境**:单轴卫星,状态 `[角度 θ, 角速度 ω]`,动作 `[力矩 τ]`,带可配置随机干扰与成功奖励。
- **模型**:PPO 训练 20 万步,`ppo_satellite_final.zip`。
- **评测**:初始偏角 **0–90°(步长 10°)** 全部纠正成功:

| 初始偏角 | 纠正时间 | 稳态平均误差 |
|---|---|---|
| 10° | 1.09 s | 0.74° |
| 30° | 1.68 s | 0.73° |
| 60° | 2.13 s | 0.73° |
| 90° | 1.77 s | 0.71° |

> 全程平均误差随初始角度增大(30° 时约 9.8°、90° 时约 39.9°),稳态误差稳定在 ~0.71–0.74°。

### 2. 三轴强化学习(`三轴强化学习/`)

- **环境**:三轴卫星,四元数姿态 + 欧拉动力学(含陀螺耦合项 `ω×Iω`),状态 `[误差旋转向量(3), 角速度(3)]`,动作 `[三轴力矩(3)]`。惯量 `I = diag(1.2, 1.0, 0.8)`,力矩饱和 ±2 N·m,`dt=0.01`。
- **模型**:三个算法各训练 80 万步——`ppo_satellite3d.zip`(收敛良好,`explained_variance ≈ 0.999`)、
  `sac_satellite3d.zip`、`td3_satellite3d.zip`。收敛监控记录见 `训练收敛监控.txt`。
- **说明**:**PPO 是唯一在本任务上真正收敛的算法**。SAC / TD3 的环境修复与调参过程、
  以及各自的失败原因分析,见 [`RL与PID对比/三轴算法对比_结果与分析.md`](RL与PID对比/三轴算法对比_结果与分析.md)。
- **评测(PPO)**:共 **29 个姿态**(roll/pitch/yaw 各自 0–90° 每 15° 一档 + 三轴对称 + 三轴耦合),**全部纠正成功**:

| 姿态类型 | 纠正时间范围 | 稳态平均误差范围 |
|---|---|---|
| roll / pitch / yaw 单轴 15–90° | 1.2 ~ 2.9 s | 0.53 ~ 0.93° |
| 三轴对称(15°~60°) | 1.8 ~ 2.4 s | 0.65 ~ 0.74° |
| 三轴耦合(非对称) | 2.2 ~ 2.6 s | 0.61 ~ 0.86° |

> 说明:纯单轴偏转时陀螺耦合项为零,近似退化为单轴问题;三轴对称 / 三轴耦合才真正考验三轴控制能力,均能稳定到 1° 以内。

### 3. RL 与 PID 对比(`RL与PID对比/`)

对比统一使用 `三轴强化学习/satellite_env3d.py` 的动力学作为唯一仿真器,RL 与 PID 在**完全相同的物理条件**下运行。PID 采用项目原参数(Kp=3、Ki=0.5、Kd=1,输出限幅 ±2 N·m)。

#### 3.1 无干扰干净对比(29 个姿态)

| | RL(PPO) | PID |
|---|---|---|
| 纠正时间 | **约 2 s(1.6~2.9 s)** | 6~12 s,其中 4/29 姿态未稳定到 1° |
| 稳态误差 | 0.5~0.9° | **0.35~0.81°(略优)** |

#### 3.2 加干扰鲁棒性对比(29 个姿态 × 4 个干扰幅值)

干扰为项目原正弦干扰 `d(t)=A·[sin(0.2Hz), 0.6·sin(0.3Hz), 0.8·sin(0.25Hz)]`,每条轨迹固定跑 20 s 观测稳态保持能力:

| 干扰幅值 A | RL 纠正时间 | PID 纠正时间 | RL 稳态残差 | PID 稳态残差 |
|---|---|---|---|---|
| 0.00 N·m | 1.80 s | 8.81 s | 0.510° | **0.174°** |
| 0.02 N·m | 1.81 s | 17.69 s | 0.511° | 0.805°(1/29 未稳定) |
| 0.05 N·m | 1.82 s | **全部未稳定** | 0.514° | 1.919°(29/29 未稳定) |
| 0.10 N·m | 1.82 s | **全部未稳定** | 0.526° | 3.793°(29/29 未稳定) |

### 4. RL 算法横向对比:PPO / SAC / TD3(`RL与PID对比/`)

在同一环境、同一奖励、同一随机初始化、**同样 80 万步预算**下,对 29 个姿态逐个评测
(均取各算法的**最终模型**,不挑 checkpoint),结果:

| 算法 | 纠正时间均值 | 末段平均误差 | 20 s 内未稳定 |
|---|---|---|---|
| **PPO** | **2.09 s** | **0.711°** | **0 / 25** |
| SAC | — | 2.771° | 25 / 25 |
| TD3 | — | 14.180° | 25 / 25 |

**只有 PPO 真正学会了任务。** 原因(详见
[`RL与PID对比/三轴算法对比_结果与分析.md`](RL与PID对比/三轴算法对比_结果与分析.md)):

- 环境原本有一个四元数双重覆盖的 bug(`_quat_error` 未处理 `q` 与 `-q` 同姿态),
  误差超过 180° 时方向会反。**已修复**;PPO 的既有结果不受影响(逐用例完全相同),
  SAC / TD3 则是主要受害者。
- **SAC**:自动熵系数塌缩到 ~4e-4、探索过早消失,卡在 ~2.7° 的平台;
  改用固定 `ent_coef=0.05` 后能下降,但平台仍在。
- **TD3**:SB3 的 TD3 是确定性策略,**必须显式给 `action_noise`** 才有探索;
  补上后开始下降但随即陷入 4–24° 的剧烈振荡,始终不收敛。

---

## 二、关键结论

1. **RL 纠正更快**:29 个姿态下 RL 纠正时间稳定在 ~2 s,远快于 PID(6~12 s)。
2. **无干扰下 PID 稳态精度略高**:PID 的积分项能消除静差(稳态误差 0.17~0.4°),优于 RL(~0.5°);但代价是更慢、且对扰动敏感。
3. **RL 抗干扰鲁棒性显著更强**:RL 训练时见过随机扰动,学出了抗扰能力——干扰从 0 增到 0.1 N·m,RL 的纠正时间和残差几乎不变(全部 29 姿态成功);而 PID 在 0.02 N·m 时已被拖慢、0.05 N·m(项目自带干扰)起 **29 个姿态全部无法稳定到 1° 以内**。
4. **RL 内部三种算法差别巨大**:同样的环境和预算下,**只有 PPO 收敛**(2.09 s / 0.711°,全部 25 个非平凡姿态稳定);SAC 卡在 ~2.7° 的平台,TD3 在 4–24° 之间剧烈振荡、始终不收敛。这说明"用 RL"不等于"能工作",**算法选择与超参配置同样关键**(详见 `RL与PID对比/三轴算法对比_结果与分析.md`)。
5. **结论一句话**:固定增益 PID 在理想条件下精度高但鲁棒性差;RL 用训练数据换来了更快的纠正和更强的抗扰能力——但在本任务上,只有 PPO 这一支 RL 路线真正兑现了这个优势。

> 公平性说明:RL 策略训练时用的是 ±0.005 N·m 的均匀随机干扰,因此鲁棒性对比考察的是 RL 在训练分布之外的表现;PID 增益本就是配合 0.05 N·m 正弦干扰选取的。

---

## 三、目录结构

```
├── pid/                        # PID 控制路线(参考实现)
│   ├── single_axis/            #   单轴 PID 基础仿真
│   ├── single_axis_disturbance/#   单轴 PID + 干扰/噪声/死区
│   └── three_axis/             #   三轴 PID(四元数 + 欧拉动力学)
│
├── 单轴强化学习/                # RL 路线 —— 单轴
│   ├── satellite_env.py        #   单轴 Gymnasium 环境
│   ├── train_satellite.py      #   PPO 训练(20 万步)
│   ├── evaluate_single_axis.py #   0–90° 评测出图
│   ├── test_env.py / test_satellite.py
│   ├── metrics.py              #   评测指标(纠正时间/平均误差/稳态误差)
│   ├── ppo_satellite_final.zip #   训练好的模型
│   └── 成果图/                  #   评测结果图 + CSV
│
├── 三轴强化学习/                # RL 路线 —— 三轴
│   ├── satellite_env3d.py      #   三轴 Gymnasium 环境(四元数)
│   ├── train_satellite3d.py    #   PPO 训练(80 万步)
│   ├── train_satellite3d_sac.py #  SAC 训练(80 万步)
│   ├── train_satellite3d_td3.py #  TD3 训练(80 万步)
│   ├── evaluate_three_axis_multi.py  # 29 姿态评测出图
│   ├── metrics.py
│   ├── 训练收敛监控.txt          #   SAC/TD3 每 2.5 万步的真实指标记录
│   ├── ppo_satellite3d.zip     #   训练好的模型
│   ├── sac_satellite3d.zip
│   ├── td3_satellite3d.zip
│   └── 成果图/
│
├── RL与PID对比/                 # 对比实验(RL vs PID;PPO/SAC/TD3)
│   ├── compare_rl_pid.py       #   无干扰干净对比(RL vs PID)
│   ├── compare_robustness.py   #   加干扰鲁棒性对比(RL vs PID)
│   ├── compare_rl_algorithms.py #  RL 算法横向对比(PPO/SAC/TD3)
│   ├── 三轴算法对比_结果与分析.md #   上面的结果、原因分析与局限
│   └── 成果图/                  #   对比结果图 + CSV
│
├── requirements.txt
└── README.md
```

> 成果图命名约定:`compare_*` / `robustness_*` 出自 **RL vs PID** 对比,
> `rl_algorithms_*` 出自 **PPO/SAC/TD3** 算法对比,两组不要混淆。

> `RL与PID对比/` 的脚本会复用 `三轴强化学习/satellite_env3d.py` 的动力学,保证两套控制器在完全相同条件下对比。

---

## 四、如何运行

### 环境

```bash
pip install -r requirements.txt
```

主要依赖:`numpy`、`gymnasium`、`stable-baselines3`、`matplotlib`。

### 训练(可选,模型已训练好)

```bash
cd 单轴强化学习 && python train_satellite.py       # 单轴 PPO,约 20 万步
cd 三轴强化学习 && python train_satellite3d.py     # 三轴 PPO,约 80 万步
cd 三轴强化学习 && python train_satellite3d_sac.py # 三轴 SAC,约 80 万步
cd 三轴强化学习 && python train_satellite3d_td3.py # 三轴 TD3,约 80 万步
```

> 断点续训:三轴脚本会自动从 `logs3d/checkpoints_*/` 里步数最大的 checkpoint 续训。
> **注意 `X.load()` 会连超参数一起恢复**——如果你改了脚本里的配置,必须先删掉
> 对应的 `logs3d/checkpoints_*` 目录,否则新配置不会生效。

### 评测出图

```bash
cd 单轴强化学习 && python evaluate_single_axis.py        # 单轴 0–90°
cd 三轴强化学习 && python evaluate_three_axis_multi.py   # 三轴 29 姿态
```

### 对比实验

```bash
cd RL与PID对比 && python compare_rl_pid.py         # RL vs PID,无干扰(约 1 分钟)
cd RL与PID对比 && python compare_robustness.py     # RL vs PID,加干扰鲁棒性(约 5 分钟)
cd RL与PID对比 && python compare_rl_algorithms.py  # PPO vs SAC vs TD3(约 2 分钟)
```

所有脚本会把结果图和 CSV 输出到各自文件夹下的 `成果图/` 目录。

---

## 五、遗留事项 / 下一步

1. **RL 域随机化**:当前 RL 训练只用 ±0.005 N·m 的随机干扰,若在训练时加大/多档随机扰动,有望进一步压低 RL 的 ~0.5° 稳态残差,让"鲁棒性"故事更完整。
2. **PID 自动调参**:做一次 PID 参数寻优(grid search),再与 RL 做"最优 PID vs RL"的公平对比。
3. **可视化补充(可选)**:给鲁棒性对比加一张"29 姿态 × 4 幅值"的热力图,把每个姿态在每个干扰档位下的表现一眼看清。
4. **TD3 探索噪声扫描**:TD3 的 `action_noise` σ 目前取 0.2 N·m,而完成姿态机动所需的
   力矩量级仅 ~0.01 N·m——噪声可能是它无法收敛的原因之一。**此猜想尚未做实验验证**,
   值得扫一遍 σ = 0.01 / 0.05 / 0.1 确认。
5. **SAC 平台问题**:SAC 固定卡在 ~2.7°,可尝试更大的 `learning_starts`、更慢的 `tau`,
   或"先用 PPO 热启动再 SAC 微调"的路线。

---

## 附:对比结果文件清单

`RL与PID对比/成果图/` 下:

**RL vs PID(`compare_rl_pid.py` / `compare_robustness.py`)**

| 文件 | 内容 |
|---|---|
| `compare_correction_curves.png` | 无干扰下 3 个代表姿态的纠正曲线(RL vs PID) |
| `compare_settling_time.png` | 纠正时间散点图(vs 初始误差) |
| `compare_steady_error.png` | 稳态误差散点图(vs 初始误差) |
| `compare_rl_pid.csv` | 无干扰对比原始数据(29 姿态) |
| `robustness_correction_time.png` | 纠正时间随干扰幅值变化 |
| `robustness_residual_error.png` | 稳态残差随干扰幅值变化 |
| `robustness_curves_A0.05.png` | 0.05 N·m 干扰下 3 个代表姿态的纠正曲线 |
| `compare_robustness.csv` | 鲁棒性对比原始数据(29 姿态 × 4 幅值) |

**PPO vs SAC vs TD3(`compare_rl_algorithms.py`)**

| 文件 | 内容 |
|---|---|
| `rl_algorithms_curves.png` | 3 个代表姿态的纠正曲线(三算法) |
| `rl_algorithms_settling_time.png` | 纠正时间散点图;未稳定的姿态画在 20 s 处(×) |
| `rl_algorithms_tail_error.png` | 末段平均误差散点图(轨迹最后 0.5 s 的均值) |
| `rl_algorithms_comparison.csv` | 原始数据(29 姿态 × 3 算法) |

> 为什么用"末段平均误差"而不是 `metrics.steady_state_error`:后者取"最后一次超过阈值
> 之后"的均值,对**始终不收敛**的策略会退化成只剩 1 个采样点的平均,数值虚低、
> 看起来反而比收敛的策略更好。末段均值对收敛与不收敛的策略都适用、可比。
