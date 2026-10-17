# 姿态禁入区下的学习路线选择

研究对象是单轴圆周姿态的固定禁入弧。每回合DQN只选择一次顺／逆时针，之后由同一固定PID执行。它属于单步上下文强化学习决策，不是连续深度反馈控制。直接目标PID、解析安全路线＋PID、DQN路线选择＋PID均公开保留。

当前实施阶段不等于已完成研究训练；实际状态以训练目录中的`training_metadata.json`与最终报告为准。新研究与12号执行器研究互不依赖，绝不修改12号源码。

使用已验证环境 `D:/Anaconda2024.06-1/python.exe -X utf8 -B`。依次运行：

```powershell
& D:/Anaconda2024.06-1/python.exe -X utf8 -B 运行研究.py design
& D:/Anaconda2024.06-1/python.exe -X utf8 -B -m unittest discover -s 代码 -v
& D:/Anaconda2024.06-1/python.exe -X utf8 -B 运行研究.py baseline
# 下列步骤会生成训练环境／启动研究，仅在统一启动通知后执行。
& D:/Anaconda2024.06-1/python.exe -X utf8 -B 运行研究.py prepare
& D:/Anaconda2024.06-1/python.exe -X utf8 -B 运行研究.py development
& D:/Anaconda2024.06-1/python.exe -X utf8 -B 运行研究.py formal
& D:/Anaconda2024.06-1/python.exe -X utf8 -B 运行研究.py freeze
& D:/Anaconda2024.06-1/python.exe -X utf8 -B 运行研究.py test
& D:/Anaconda2024.06-1/python.exe -X utf8 -B 运行研究.py verify
& D:/Anaconda2024.06-1/python.exe -X utf8 -B 运行研究.py report
```

开发64、验证128、自然测试512个独立初态；另外预先规定各128个绕行和边缘压力初态。训练使用独立生成的2万个初态，先对两动作真实仿真，并仅缓存几何初态和结果指标；学习器只看到自己选择动作的奖励。没有正确方向监督标签或规则动作修正。

一次DQN步对应一整条22秒反馈轨迹，不能称为一次0.05秒物理交互。缓存真实物理传播次数、在线决策数、重复状态动作重放数、验证物理仿真及优化器记录分别公开。研究预算最多开发2×1万、正式3×2万，共8万路线决策。

解析路线对单个固定禁区有解析解，是完整传统规划基准。若学习改善直接PID，只说明加入路线选择的收益；不将它写成学习低层反馈战胜PID，也不预设学习优于解析规划。碰撞失败的低力矩代价不算节能收益。每个输出文件保持小于100 MiB；测试轨迹按方法和集合拆分。
