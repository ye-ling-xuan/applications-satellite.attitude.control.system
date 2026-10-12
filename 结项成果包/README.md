# 卫星姿态控制结项成果包

本包包含前一版单轴 PID/PPO 随机化实验，以及新增的单轴、三轴 PID/PPO/SAC/TD3 同条件比较。新增材料融合入口是 [07 多算法融合分析](07_单轴三轴多算法对比/新融合分析.md) 和 [07 离线结果页](07_单轴三轴多算法对比/结果总览.html)。[工作总说明](工作总说明.md) 汇总两版工作；前一版[技术报告](06_结项说明与报告/结项技术报告.md)和数据保留。所有修订位于本成果包，来源文件未删改。

## 工作目录

| 标题 | 文件夹或入口 | 内容 |
|---|---|---|
| 01 原始资产遴选 | `01_遴选原始参考` | 有用代码、历史模型、曲线、笔记及两篇原报告的原样副本；来源和哈希清单 |
| 02 统一建模与实验标准 | `02_配置与方法`、`satellite_benchmark` | 单位、动力学、PID、训练、评估、绘图的统一实现及实际依赖记录 |
| 03 模型训练 | `03_训练模型与日志/正式训练_100万步` | 正式交付：普通和随机化 PPO 各三个独立种子，选中模型、最终模型、验证及训练日志；PID 整定记录 |
| 04 公平对比与鲁棒性 | `04_实验数据与图表` | 固定测试场景、逐回合指标、全部原始轨迹、汇总和六张核心图 |
| 05 三轴建模拓展 | `05_三轴拓展演示` | 四元数最短旋转、刚体耦合、RK4、PID、三轴曲线及轨迹 |
| 06 结项说明与核验 | `06_结项说明与报告` | 技术报告、答辩提纲、原文件保全检查和回归验证记录 |
| 07 单轴三轴多算法融合 | `07_单轴三轴多算法对比`、`multi_algorithm_benchmark` | PID整定/滤波、PPO/SAC/TD3新训练与同条件测试、零参考力矩校准、历史模型迁移部署、配对改进证据 |

## 各代码工作的标题与职责

| 工作标题 | 代码 | 职责 |
|---|---|---|
| 物理模型和约束配置 | `satellite_benchmark/config.py` | 惯量、步长、力矩、扰动噪声、随机化范围、稳定判据 |
| 统一姿态环境 | `satellite_benchmark/env.py` | Gymnasium 接口、精确常加速度步进、随机数、工程约束、真实状态记录 |
| PID 与抗积分饱和 | `satellite_benchmark/controllers.py` | 同观测 PID、积分限幅、条件积分、输出约束 |
| 正确性能指标 | `satellite_benchmark/metrics.py` | 持续稳定、成功率、误差、超调、控制代价 |
| 模型训练与选模 | `satellite_benchmark/training.py` | 普通/随机化 PPO、固定验证场景、独立种子和训练归档 |
| 参数搜索与配对实验 | `satellite_benchmark/evaluation.py` | 验证集 PID 搜索、批量模型预测、测试场景及全部轨迹保存 |
| 结项图表 | `satellite_benchmark/plotting.py` | 从保存的数据生成六张图，明确展示种子波动和失败 |
| 报告与离线展示 | `tools/build_report.py` | 从真实数据生成总说明、技术报告和可筛选的结果页面 |
| 原文件保全 | `tools/verify_originals.py` | 核验原项目123份文件的 SHA256，与整理前快照比较 |
| 正式结果核验 | `tools/verify_results.py` | 逐一重算5600回合指标与48组汇总，核验动力学、配对输入、力矩限幅及模型来源 |
| 回归检查 | `tests` | 动力学、随机数、积分饱和、稳定/超调与数据接口等检查 |
| 多算法统一管线 | `multi_algorithm_benchmark`、`run_multi.py` | 单轴/三轴PPO、SAC、TD3三种子训练，原始/整定/改进PID和策略校准的配对比较 |
| 三轴批量训练 | `multi_algorithm_benchmark/vector_training.py` | 与逐个Gym环境数值一致的批量RK4、独立重置及标准Monitor记录 |
| 历史模型兼容加载 | `multi_algorithm_benchmark/archive_compat.py` | 保留权重及Box边界，处理NumPy版本元数据兼容；仅用于历史模型推理 |
| 新实验独立核验 | `tools/verify_multi_results.py` | 轨迹物理重放、独立指标/汇总/配对差值复算、模型和两批来源保全 |

## 新融合实验入口

在本目录运行 `python run_multi.py check`。已保存模型的复测建议另设输出目录：

```powershell
python run_multi.py evaluate --samples 20 --output '07_单轴三轴多算法对比/另存复测'
python run_multi.py plot --results '07_单轴三轴多算法对比/另存复测'
python run_multi.py report --results '07_单轴三轴多算法对比/另存复测' --output '07_单轴三轴多算法对比/另存复测/报告'
```

从零训练、PID整定和策略校准的命令见07的README。原算法与校准后策略分别展示；校准不修改模型权重，也不等于积分抗扰。原始PID、整定PID、改进PID分别保存，Ki=0或alpha=1时按实际含义解释。

全部新结果的独立核验记录在 [07核验记录](07_单轴三轴多算法对比/核验记录.json)，可在本目录运行 `python tools/verify_multi_results.py` 重新核验。

## 安装与启动

实际验证环境为 Python 3.12.4、CPU PyTorch，版本见 `02_配置与方法/environment.json` 和 `requirements-tested.txt`。正常安装可使用 `requirements.txt`。推荐在独立 Python 环境内安装：

```powershell
cd 'D:\applications-satellite.attitude.control.system\结项成果包'
python -m pip install -r requirements.txt
python run.py check
```

本机已经验证可用的解释器是 `D:\Anaconda2024.06-1\python.exe`。如果 `python` 指向其他环境，可使用：

```powershell
& 'D:\Anaconda2024.06-1\python.exe' run.py check
```

## 使用已交付模型重新评估

不需要重新训练即可检查结果。正式模型位于 `03_训练模型与日志/正式训练_100万步`，`run.py` 也以此为默认模型路径。为保留交付数据，复现实验使用单独输出目录，下面显式指定模型来源：

```powershell
python run.py evaluate --models '03_训练模型与日志/正式训练_100万步' --samples 50 --output '04_实验数据与图表/复现实验'
python run.py plot --models '03_训练模型与日志/正式训练_100万步' --results '04_实验数据与图表/复现实验'
python run.py three-axis
```

正常工况与五类压力条件共16个case，每个case50组初态，共800场景；PID和两类PPO的三个训练种子共7个控制器，共5600回合。默认仿真20 s，dt=0.05 s，末5 s持续满足误差≤1°且角速度≤0.5°/s才成功。正式交付数量以 `evaluation_manifest.json` 为准。

## 从零复现训练和整定

六个正式模型各请求 1,000,000 交互步，实际各训练 1,001,472 步；总请求 6,000,000 步，实际共 6,008,832 步。`nominal` 和 `randomized` 各使用种子 11、22、33。正式训练每 50,000 步进行验证，依据验证集选择 `best_model.zip`，最终测试不参与选模。

新实验必须使用新输出目录，以免覆盖已交付模型。`--steps` 表示每份模型的请求交互预算，实际步数向完整 rollout 向上取整；`--jobs 3` 最多同时训练三个独立模型，六个模型分批完成。复现正式预算的命令为：

```powershell
python run.py train --mode both --seeds 11 22 33 --steps 1000000 --validation-interval 50000 --jobs 3 --output '03_训练模型与日志/重新训练_100万步'
python run.py tune-pid --output '03_训练模型与日志/重新训练_100万步'
python run.py evaluate --models '03_训练模型与日志/重新训练_100万步' --samples 50 --output '04_实验数据与图表/重新训练评估'
python run.py plot --models '03_训练模型与日志/重新训练_100万步' --results '04_实验数据与图表/重新训练评估'
```

每个模型的实际预算、配置、选中检查点和依赖版本记录于同目录 `training_metadata.json`；`best_model.zip` 可能来自较早检查点，不一定等于 `final_model.zip`。这里显式传入 `--validation-interval 50000`，因为 CLI 的验证间隔默认值为 20000。跨平台或不同库版本训练结果可能不同，评估已保存模型通常更容易复现。

恢复训练使用 `--resume`；恢复的是策略和优化器，不保证恢复中断时的环境随机数和未完成 rollout。追加预算、恢复过程和日志保存于模型旁；最终测试不用于选择检查点。

## 数据阅读

- `episode_metrics.csv`：每个模型、每个测试场景的真实结果；稳定时间为空表示未持续稳定。
- `summary.csv`：按具体case汇总；成功率误差线基于三个独立训练种子，PID没有训练种子误差线。
- `raw_traces/*.npz`：所有采样点的真实角度、角速度、实际及指令力矩、扰动、惯量和失败标记；内嵌JSON元信息，可用 `np.load(..., allow_pickle=False)` 读取。
- `test_scenarios.json`：固定场景参数和随机种子；不同case复用初态用于配对。
- `figures`：训练验证趋势、配对响应、鲁棒性、控制代价、失败案例、训练种子波动。
- `family_summary.json`：技术报告使用的六类工况汇总。
- `06_结项说明与报告/result_verification.json`：全部轨迹、指标、汇总和六份模型的独立核验记录；`run.py check`自动执行该核验。

## 结果解释与交付边界

历史模型的观测、动作映射和步长与新环境可能不同，不能直接加载成等预算正式对比模型。01目录存来源副本：首次67份的重新哈希核验见 `06_结项说明与报告/copy_verification.json`；新增60份及827份来源快照见 `01_遴选原始参考/05_中期前任务收尾新增材料`，两批共127份副本的联合核验见07核验记录。四份新增历史模型以原权重在统一环境中单独做迁移部署复测，不混入新模型排名。

模型目录的名称保留了工作过程：`03_训练模型与日志/正式训练_100万步` 是本次正式交付；旧目录 `03_训练模型与日志/正式训练` 为每个模型请求 200,000 步、实际 200,704 步的早期探索记录，现已由百万步版本替代；`03_训练模型与日志/nominal/seed_11` 为中断的早期联调记录。后两者不参与正式结果。详细分类见 [模型与日志说明](03_训练模型与日志/README.md)。

本包交付软件仿真证据，不保证学校“优秀”评定，不声称完成在轨产品。随机化训练、算法选型、滤波及策略校准效果以实际统计为准，可能存在退化。三轴强化学习与SAC/TD3对比见新增07；尚未实施模型辨识/MPC、STK联合验证或硬件在环。正式提交前应完成团队复现、文献核查、学校格式转换和答辩。

报告更新命令会从默认正式交付数据生成说明：

```powershell
python run.py report
```
