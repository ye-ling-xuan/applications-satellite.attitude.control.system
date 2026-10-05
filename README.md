# AI 卫星姿态控制：可复现实验工作区

本项目后续主线是统一环境接口、研究 PID + PPO 组合控制并扩展到三轴，保留 PID 和独立 PPO 作为对照。当前完成的是单轴 PID 与独立 PPO 实验；组合初稿为固定 PD + PPO，尚未验证完整 PID + PPO 的优势。当前维护代码是仓库根目录下的 `attitude_control/`；原有学习笔记、模型和脚本保留供参考。

组内同步材料见根目录的 `项目进展与后续推进方向.md`，包含当前成果、实验结论、下一阶段优先级与建议分工。

## 当前进度

当前项目根目录为原 GitHub 仓库 `applications-satellite.attitude.control.system` 的本地目录，个人分支为 `xingyu-part`。在 VS Code 中打开此目录。外层 `大一立项` 只作为存放项目的容器，避免在那里初始化另一个仓库。

- 已建立共享的单轴动力学、执行器限幅、扰动场景和指标计算。
- 已实现 PD/PID 基线、纯 PPO 和固定 PD + PPO 残差控制。
- 已增加回归测试及 VS Code 测试、调试和运行任务。
- 已在独立验证集上整定传统控制器；后续实验使用固定基线，避免在测试集调参。
- 已增加有版本记录的训练任务、独立验证集选模与多种子测试；实际结果见下方实验记录。
- 已完成两组各 5 个种子的独立 PPO 训练，以及力矩惩罚权重对照；49 项回归测试通过。

## 在这台电脑运行

已检查到可用解释器 `D:\Users\admin\anaconda3\python.exe`，包含 NumPy、PyTorch、Gymnasium、SB3、Matplotlib、pytest。

在 VS Code 中打开当前项目根目录，通过命令面板的 `Python: Select Interpreter` 选择这个解释器。工作区已设置默认路径，但以前手动选择的解释器可能仍需切换。

在项目根目录的 PowerShell 终端运行：

```powershell
$satPython = 'D:\Users\admin\anaconda3\python.exe'
& $satPython -m pytest -q
& $satPython -m attitude_control.tune --output artifacts/tuning_v1
& $satPython -m attitude_control.evaluate --tuning artifacts/tuning_v1/tuning.json --output artifacts/baselines_v1 --plot
& $satPython -m attitude_control.train --mode pure --steps 50000 --seed 7 --tuning artifacts/tuning_v1/tuning.json --output artifacts/pure_seed7_v1
& $satPython -m attitude_control.train --mode residual --steps 50000 --seed 7 --tuning artifacts/tuning_v1/tuning.json --output artifacts/residual_seed7_v1
& $satPython -m attitude_control.evaluate --tuning artifacts/tuning_v1/tuning.json --pure artifacts/pure_seed7_v1 --residual artifacts/residual_seed7_v1 --output artifacts/comparison_v1 --plot
```

上述为最初的 v1 先导实验流程，其 5 万步 PPO 没有达到成功标准。当前独立 PPO 的运行方式为：

```powershell
& $satPython -m attitude_control.train --profile precision_nominal --mode pure --steps 200000 --seed 7 --n-envs 4 --n-steps 256 --batch-size 256 --learning-rate 0.0001 --gamma 0.98 --ent-coef 0.01 --validation-interval 20000 --tuning artifacts/tuning_v1/tuning.json --output artifacts/precision_nominal_seed7_v2
```

本轮已完成的模型目录为 `artifacts/precision_nominal_seed{7,19,42,61,83}_v1`。每个目录中的 `model.zip` 为验证集选出的检查点，`final_model.zip` 为最后一次更新后的模型。评估不会依据测试集选择模型。

降低力矩惩罚的对照组已保存在 `artifacts/precision_lowtorque_seed{7,19,42,61,83}_v1`。名称中的 lowtorque 指较低的惩罚权重，并不表示输出力矩较低。重跑其中一个种子可使用新目录：

```powershell
& $satPython -m attitude_control.train --profile precision_nominal --torque-penalty 0.05 --mode pure --steps 200000 --seed 7 --n-envs 4 --n-steps 256 --batch-size 256 --learning-rate 0.0001 --gamma 0.98 --ent-coef 0.01 --validation-interval 20000 --tuning artifacts/tuning_v1/tuning.json --output artifacts/precision_lowtorque_seed7_v2
```

```powershell
& $satPython -m attitude_control.benchmark --models artifacts/precision_nominal_seed7_v1 artifacts/precision_nominal_seed19_v1 artifacts/precision_nominal_seed42_v1 artifacts/precision_nominal_seed61_v1 artifacts/precision_nominal_seed83_v1 --tuning artifacts/tuning_v1/tuning.json --output artifacts/benchmark_nominal_v2 --test-seed 9040 --cases-per-group 12
```

已有结果保存在 `artifacts/benchmark_nominal_v1`。重新运行同一测试集可检查复现性；后续若依据其结果修改算法，该集合应视为诊断/回归集合，最终结论另用新的预先固定测试集。

本轮奖励对照采用新的测试 seed=10405，两组各复用 84 个场景，结果分别在 `artifacts/ablation_r050_test10405_v1` 和 `artifacts/ablation_r005_test10405_v1`。核对配对条件并生成汇总：

```powershell
& $satPython -m attitude_control.compare --left artifacts/ablation_r050_test10405_v1 --right artifacts/ablation_r005_test10405_v1 --output artifacts/torque_ablation_v2
```

现成汇总在 `artifacts/torque_ablation_v1`。比较脚本检查场景、种子、物理配置、训练工况、超参数、预算及选模规则一致；调节时间差仅使用两组都成功的配对场景。

已有非空输出目录不会被覆盖。上述实验本轮已经运行过的目录请直接查看；再次运行时改用 `v2` 等新目录。

其他电脑可创建独立环境，再运行 `python -m pip install -e ".[rl,plot,dev]"`。这台电脑无需重新安装。训练模型目录记录实际依赖版本，`pyproject.toml` 给出支持范围。

## 代码入口

| 文件 | 作用 |
|---|---|
| `config.py` | 步长、惯量、力矩上限、控制器增益和指标阈值 |
| `core.py` | 单轴动力学、扰动序列、公共观测、奖励与回合推进 |
| `controllers.py` | PD/PID 与 PPO 动作到力矩的映射 |
| `env.py` | Gymnasium 环境，只适配公共仿真内核 |
| `tasks.py` | 训练工况、奖励定义和观测归一化；保存模型时记录任务版本 |
| `metrics.py` | 超调、持续稳定时间、成功率及控制代价 |
| `tune.py` | 使用验证场景整定传统控制器 |
| `train.py` | 训练纯 PPO 或残差 PPO，保存模型及配置 |
| `evaluate.py` | 在相同测试场景中运行所有控制器并导出结果 |
| `validation.py` | 在独立验证场景上选择训练检查点 |
| `benchmark.py` | 多种子独立 PPO 与 PID/PD 的共用场景测试 |
| `compare.py` | 检查奖励权重对照的配对条件，导出比较表和图 |
| `diagnose.py` | 复现旧版模型并检查先导策略在目标附近的输出 |
| `tests/` | 动力学、指标、随机种子、动作映射、模型契约测试 |

### 当前物理模型与控制律

内部采用弧度、rad/s、kg·m²、N·m；带 `_deg` 的配置及绘图字段采用度。单轴模型为 `I * omega_dot = torque + disturbance`，每个采样周期内假定力矩恒定，使用精确零阶保持更新。它与历史脚本的半隐式欧拉更新不同，旧模型不直接混入新实验。

纯 PPO：`torque_command = 2 * clip(action, -1, 1)`。

残差 PPO：`torque_command = kp * attitude_error - kd * omega + 0.5 * clip(action, -1, 1)`。

两者与 PD/PID 最终均经过同一个 ±2 N·m 的执行器限幅。残差使用固定 PD。观测为 `[error/angle_scale, omega/rate_scale, previous_applied_torque/max_torque]`，不包含真实惯量或真实扰动；v1 的观测角度尺度为 180°，precision 任务为 40°。加载模型时按其保存的任务转换观测，避免训练、测试归一化不一致。PID 使用角速度作微分反馈并采用条件积分防饱和。

当前是目标附近的单轴机动模型，不处理跨越 ±180°的角度环绕，也没有真实反作用轮、传感器噪声或通信延迟。随机力矩噪声属于外部扰动，不能解释为测量噪声。

### 奖励与成功标准

v1 奖励为角度误差、角速度、施加力矩和力矩变化的归一化平方代价，乘以 `dt`；失败附加 −100。

precision 任务参考原改良版的二次奖励：`-5*error² - omega² - torque_penalty*torque² + 2*error*omega`，单位采用 SI。`--torque-penalty` 默认为 0.5，本轮对照值为 0.05。当角度、角速度同时满足评价阈值时，每步另加 2；失败附加 −50。这与原脚本仅按角度发放奖励有所区别。precision 奖励不乘以 `dt`，因此其训练回报与 v1 不可直接比较。任务记录版本为 `explicit-ppo-task-v2`；旧版任务按原权重 0.5 加载。

两种任务共用同一动力学和成功判据。越过 90°误差边界是真正终止；正常达到 10 s 时间上限使用 `truncated=True`。训练不会因短暂经过目标提前终止。各控制器的测试表仍统一计算 `core.stage_cost` 的归一化代价及物理指标，不用不同训练奖励排名。

评估成功要求完整运行至 10 s，最后至少 1 s 同时满足 `|姿态误差| <= 1°` 和 `|角速度| <= 0.2°/s`，且没有失败。调节时间是最后一次离开条件区间之后的起点，并要求满足保持时间；失败场景的调节时间为缺失值，汇总明确标注仅对成功场景取平均。

超调是从初始误差一侧越过目标后的最大偏差。`torque_effort_nm2_s` 是力矩平方积分，单位 N²·m²·s，属于控制代价代理，不能称为实际电能。

### 实验划分和输出

- 基线整定：12 个验证场景，seed=101；网格搜索 PD/PID 增益，以平均归一化控制代价选择。
- v1 训练：初始误差 ±60°、角速度 ±10°/s，惯量 0.7–1.3、恒定扰动力矩 ±0.03 N·m；可用 `--nominal-only` 关闭惯量与扰动随机化。
- `precision_nominal`：初始误差 ±40°、角速度为 0、惯量固定、无扰动，先建立可学习的基础任务。
- `precision_robust`：保留 precision 奖励和观测，加入初始角速度、惯量和偏置力矩随机化；该配置已实现，本轮未训练。
- 验证：8 个固定独立场景；按成功率、失败率、末段误差、归一化代价依次选模。训练中每 2 万步检查一次，并保留最后模型用于诊断退化。
- 测试：seed=2026，6 组各 6 个场景，涵盖正负初始角度、不同初始角速度、惯量 0.5/1/1.5、±0.08 N·m 偏置、脉冲加随机力矩。每个场景对各方法复用同一扰动序列。
- 隐藏的随机惯量和扰动在启用时使训练属于部分可观测的鲁棒性任务；当前没有历史观测或在线参数辨识，不能宣称完成自适应辨识。
- 多种子测试：诊断 seed=9040，奖励对照 seed=10405，均为 7 组各 12 个场景；训练范围内一组，另含初始状态扩展、惯量低/高、正/负偏置、脉冲噪声。每个控制器复用相同场景，种子间标准差按独立训练模型计算，不把测试回合冒充独立训练次数。
- `metadata.json` 保存训练种子、控制模式、配置、实际训练步数、依赖版本、源码哈希及模型哈希。
- `manifest.json` 保存测试场景、配置、模型来源、测试种子和源码哈希。
- `episodes.csv` 保存每次实验指标；`summary.csv/json` 按方法和场景组汇总；`.npz` 保存逐步状态、实际力矩和扰动；`response.png` 展示其中一个场景。
- 可视化是单个场景，结论应基于完整统计；训练奖励不能跨不同奖励定义直接比较。

## 在 VS Code 与 Codex 协作

本机已发现 `openai.chatgpt`（Codex）扩展，首次使用需在扩展侧栏完成登录；本轮未检查账号登录状态。打开项目文件夹而不是单个历史脚本，在 Codex 侧栏明确指定任务、相关文件和验收条件。`AGENTS.md` 提供持续协作约定。

推荐一次完成一个可测试的改动，例如：

> 阅读 AGENTS.md、README.md 和 attitude_control/env.py。为单轴环境增加测量噪声；真实状态用于动力学和指标，带噪声观测只交给控制器。补充无噪声退化、种子复现、训练评估一致性测试，运行 pytest；先不要重新训练。

复查时可使用：

> 复查本次改动，重点检查单位、执行器限幅、观测与动作映射、时间截断、随机种子、PID 积分状态、训练和测试泄漏。先列出带代码位置的可复现问题，再提出修正；不能仅凭响应曲线宣称 RL 更优。

通过 `Terminal: Run Task` 可执行测试、基线整定、基线评估和独立 PPO 训练；训练任务可选择随机种子和力矩惩罚。调试面板可运行环境测试或评估。输出目录已存在时换新目录，保护既有实验。避免桌面 Codex 与 VS Code 中另一个活动会话同时修改同一组文件。

官方说明：[Codex IDE 扩展](https://learn.chatgpt.com/docs/codex/ide)、[AGENTS.md](https://learn.chatgpt.com/docs/agent-configuration/agents-md)。

## 本轮实验记录

2026-10-05，本机 CPU 实际完成以下工作：

1. 复现旧版已保存 PPO：原改良模型在所检查的 6 个单轴场景中可以稳定控制，证明之前调用 SB3 的训练确实产生了策略。原环境和新环境的复现分别记录在 `artifacts/diagnosis_v2`，不混用两种动力学结果。
2. 完成共享环境的 v1 先导实验：独立 PPO 和残差 PPO 各训练 50,176 步，在 36 个扩展场景中均未达到成功标准；该结果用来检查和诊断训练配置，不用于否定强化学习。
3. 完成 precision_nominal 基础训练：5 个种子各实际训练 200,704 步。部分模型在后期退化，因此按独立验证集保留检查点，而非直接采用最后模型。这次恢复同时调整了观测、奖励、工况和训练超参数，不能把效果单独归因于某一项。
4. 在相同 5 个种子和预算下，将力矩惩罚从 0.5 降为 0.05，其他训练因素配置保持一致，并对两组用新的 seed=10405 配对测试。每组评估 5×84=420 个 PPO 回合，PD/PID 各评估 84 个相同场景。

下表为完整测试集结果。PPO 数值为 5 个独立训练模型的均值 ± 模型间样本标准差，PD/PID 为固定控制器结果；误差取最后 1 秒的平均绝对误差。

| 控制器 | 训练范围内成功率 | 全部工况成功率 | 全部工况末段误差 / ° | 全部工况力矩平方积分 / N²·m²·s |
|---|---:|---:|---:|---:|
| PD | 100% | 100% | 0.220 | 1.627 |
| PID | 100% | 100% | 0.294 | 1.630 |
| PPO，惩罚 0.5 | 93.3% ± 14.9 个百分点 | 42.9% ± 10.0 个百分点 | 1.710 ± 0.421 | 0.192 ± 0.119 |
| PPO，惩罚 0.05 | 100% ± 0 个百分点 | 97.6% ± 5.3 个百分点 | 0.150 ± 0.057 | 0.828 ± 0.185 |

降低惩罚后，PPO 更积极地施加力矩，成功率和误差明显改善。训练范围内的 PPO 五个模型全部成功，平均调节时间为 4.95 s，PID 为 3.43 s，PD 为 3.41 s；PPO 仍较慢。扩展工况中有一个 PPO 种子在低惯量场景失去稳定成功率，需研究鲁棒训练。力矩平方积分较低也不能单独构成优势：应与成功率、精度和调整速度共同评价，不能把它当作实际能耗。

结果支持继续研究独立 PPO 的奖励设计和泛化，目前不支持“PPO 全面优于传统控制”的结论。当前只是单轴理想执行器仿真，传统基线采用有限网格整定；尚未对匹配调节时间的控制代价、真实执行器和三轴模型作充分比较。

- 配对结果：`artifacts/torque_ablation_v1/comparison.csv`、`checks.json`、`ablation.png`。
- 改进组逐回合结果与响应图：`artifacts/ablation_r005_test10405_v1/episodes.csv`、`responses.png`。
- 验证结果：`python -m pytest -q`，49 项测试通过；覆盖共享动力学、指标、模型加载、观测变换、任务版本、配对条件和移动目录后的结果加载。

## 下一步

1. 完善公共环境接口，先在单轴上实现完整 PID + PPO 修正力矩，处理积分状态的观测、重置和防饱和。组合策略需要重新训练。
2. 对比 PID、独立 PPO、PID + PPO，重点检查惯量变化、偏置扰动和速度与控制代价的权衡。先利用验证集确认训练，再冻结新的测试种子；10405 后续作为诊断/回归集合。
3. 单轴组合验证后，将已有三轴四元数动力学接入相同训练与评估接口，并重新适配三种控制方法。
4. 逐项加入测量噪声、延迟和反作用轮限制，每次增加一个因素并验证训练与评估一致性。
