# 已保存的实验快照

本次提交包含已完成实验的配置、模型、验证记录、指标表和图片。逐步状态数据 `.npz`、缓存、失败的中间输出和临时检查结果保留在本地。

- `tuning_v1`：传统控制器整定。
- `pure_seed7_v1`、`residual_seed7_v1`、`comparison_v1`：约 5 万步先导实验。
- `diagnosis_v2`：旧版模型复现与策略检查。
- `precision_nominal_seed*_v1`：力矩惩罚 0.5 的五个独立模型。
- `precision_lowtorque_seed*_v1`：力矩惩罚 0.05 的五个独立模型。
- `ablation_r050_test10405_v1`、`ablation_r005_test10405_v1`：相同场景的两组评估。
- `torque_ablation_v1`：配对比较表、检查记录和对照图。

原始清单保留实验时的绝对路径和源码哈希。比较脚本支持移动或克隆后从相邻模型目录读取元数据，并检查模型标识一致；原始记录不被改写。重新运行实验请使用新的输出目录，避免覆盖这些快照。

当前 `model.zip` 是验证集选出的模型，`final_model.zip` 是训练最后的模型。查看项目根目录 README.md 获取完整运行命令。
