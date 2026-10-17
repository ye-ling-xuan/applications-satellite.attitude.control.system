# 一次 Windows 文件写入中断与恢复

正式训练的单轴种子 42，在第 100000 步验证时，更新 `best_checkpoint.json` 发生 `PermissionError: [WinError 5]`。错误出现在临时文件替换 JSON 的步骤，没有证据表明神经网络或动力学计算发生异常。

该次未完成运行的 23 个文件已保留在 `incomplete_runs/single_config_B_seed42_iofailure_100k/`。移动前后逐文件 SHA-256 完全一致，绝对路径均已检查位于本研究的 `03_SAC训练` 目录内。错误信息、步数、保留文件哈希和恢复方式记录在 `io_incident_record.json`。

随后，使用原种子 42、原奖励、原网络、原超参数和原来的 100 万步预算，从零重新训练。没有读取正式测试结果，没有挑换种子，也没有把中断时的模型继续训练后冒充从零完成的一次训练。

重跑通过 `io_retry_runner.py` 启动。包装器只在 JSON 原子替换出现 `PermissionError` 时，每隔 0.1 秒重试，最多 30 次；不改变物理、观测、奖励、随机数流或优化器。包装器的实际源码哈希和作用写入重跑的 `training_manifest.json`、`training_metadata.json`，源码副本保存在该运行的 `source_snapshot/execution_io_retry_runner.py`。实际重试事件保存在 `io_retry_execution_record.json`。

正式研究配置冻结文件和训练核心源码没有因为此次恢复而修改。其他 5 份正在执行的正式训练没有重启。该中断尝试的 10 万交互步应计入实际计算量，但不计作一份完成模型：本次新增研究计划完成的正式模型仍是 6 份、600 万步；加上 4 份开发训练的 80 万步和该次中断的 10 万步，实际总量预计为 690 万步。最终以所有运行的完成/中断记录核验。
