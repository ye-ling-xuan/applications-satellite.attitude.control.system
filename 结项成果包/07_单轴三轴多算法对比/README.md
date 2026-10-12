# 单轴与三轴多算法对比：新增成果入口

这里把新增的 PID/PPO/SAC/TD3 工作融入统一的结项实验。原文件与存档模型保留，新训练模型、正式测试和历史迁移复测分开保存。

先看 [新融合分析](新融合分析.md) 和 [离线结果总览](结果总览.html)。本次正式测试 7560 回合；历史策略迁移复测 720 回合。实际改进、失败与种子波动均由保存的数据生成。

| 工作 | 目录/文件 | 用途 |
|---|---|---|
| 01 数学模型与共同条件 | `../multi_algorithm_benchmark/envs.py` | 单轴刚体、三轴四元数与耦合动力学 |
| 02 PID 基线、整定、改进 | `../multi_algorithm_benchmark/controllers.py`、`models/pid_configs.json` | 验证集寻优、饱和积分保护与测量滤波 |
| 03 PPO/SAC/TD3 新训练 | `models/single`、`models/three` | 同条件、同交互预算、3 种子 |
| 04 正式对比 | `results/episode_metrics.csv`、`results/summary.csv` | 全回合和按工况统计 |
| 05 配对改进 | `results/paired_improvements.csv`、`results/pairwise_results.csv` | 相同初态、外扰和噪声下的效果变化 |
| 06 全部原始轨迹 | `results/single/raw_traces`、`results/three/raw_traces` | 独立复算指标和绘图 |
| 07 历史模型迁移部署 | `results/historical` | 单独评估，不混入新模型排名 |
| 08 图表与报告 | `results/figures`、`新融合分析.md`、`结果总览.html` | 答辩和结项素材 |
| 09 RL 部署校准 | `../multi_algorithm_benchmark/improvements.py`、`results/policy_calibration_pairs.csv` | 同模型、同种子、同配置比较零动作补偿前后 |

在结项成果包根目录运行：

```powershell
python run_multi.py check
python run_multi.py evaluate --samples 20
python run_multi.py plot
python run_multi.py report
```

正式模型已经保存，可直接复测。重新训练时请指定新的输出目录，防止把已发布实验覆盖：

```powershell
python run_multi.py tune-pid --output "07_单轴三轴多算法对比/另一次训练"
python run_multi.py train --steps 1000000 --jobs 3 --output "07_单轴三轴多算法对比/另一次训练"
python run_multi.py calibrate --models "07_单轴三轴多算法对比/另一次训练"
python run_multi.py evaluate --models "07_单轴三轴多算法对比/另一次训练" --output "07_单轴三轴多算法对比/另一次结果"
python run_multi.py plot --results "07_单轴三轴多算法对比/另一次结果"
python run_multi.py report --models "07_单轴三轴多算法对比/另一次训练" --results "07_单轴三轴多算法对比/另一次结果" --output "07_单轴三轴多算法对比/另一次结果/报告"
```

固定验收：完整部署 20 秒，末 5 秒每个样本误差 ≤1° 且角速度 ≤0.5°/s，无发散。此指标比一次入阈即提前终止更严格，历史结果与本次数字不能直接横向拼接。
