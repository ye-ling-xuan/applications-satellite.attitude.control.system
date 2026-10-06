# AI 在卫星姿态调整中的应用

单轴 / 三轴卫星姿态控制的 PID 仿真,以及基于强化学习(PPO)的姿态控制实验。项目分两条路线:

- **PID 路线** —— `pid/`
- **强化学习路线** —— `rl/`

## 目录结构

```
├── pid/                              # PID 控制路线
│   ├── single_axis/                  # 单轴卫星 PID 最小仿真系统(基础版)
│   ├── single_axis_disturbance/      # 单轴 PID + 干扰 / 噪声 / 死区等实际影响
│   └── three_axis/                   # 三轴卫星姿态控制(四元数 + 欧拉动力学)
│
├── rl/                               # 强化学习路线
│   ├── satellite_env.py              # 单轴卫星 Gymnasium 环境
│   ├── train_satellite.py            # PPO 训练脚本
│   ├── test_env.py                   # 环境冒烟测试
│   └── test_satellite.py             # 训练结果测试
│
├── requirements.txt
└── README.md
```

## 运行

### 单轴 PID(基础版)

```bash
python pid/single_axis/main.py
```

### 单轴 PID(加干扰 / 噪声 / 死区)

```bash
python pid/single_axis_disturbance/main.py
```

### 三轴姿态控制

```bash
python pid/three_axis/main_3d.py
```

### 强化学习(PPO)

```bash
python rl/train_satellite.py
```

## 依赖

安装依赖:

```bash
pip install -r requirements.txt
```
