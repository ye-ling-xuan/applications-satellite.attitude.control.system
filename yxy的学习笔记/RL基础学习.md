以下是你项目的**强化学习基础学习文档**，内容紧扣卫星姿态控制，从零开始，包含重要公式和算法伪代码。你可以将其保存为 `RL_Basics_for_Satellite_Control.md`。

---

# 卫星姿态控制中的强化学习基础

## 1. 从PID到强化学习：控制思想的跃迁

| 控制方法 | 控制律来源 | 参数获取 | 优化目标 |
|---------|-----------|---------|---------|
| PID | 人工公式 \(\tau = K_p e + K_i \int e + K_d \dot{e}\) | 手动调参 | 间接通过三项增益影响响应 |
| 强化学习 | 非线性神经网络 \(\tau = \mu_\theta(s)\) | 从交互数据自动学习 | 直接最大化自定义奖励 |

- **PID** 是固定的线性映射，善于处理单变量线性系统，但难以应对耦合、非线性、多目标权衡。
- **强化学习** 将控制问题视为**序列决策问题**，让智能体自主发现从状态到动作的最优映射。

## 2. 核心框架：马尔可夫决策过程 (MDP)

MDP 将一切交互式问题抽象为五元组 \((S, A, R, P, \gamma)\)：

- **状态空间 \(S\)**：智能体感知的环境信息。  
  本项目单轴：\(s = [\theta, \omega]\)，三轴扩展为四元数 \(q\) 与角速度 \(\boldsymbol{\omega}\)。
- **动作空间 \(A\)**：智能体可执行的输出。连续力矩 \(\tau \in [-2, 2]\) Nm。
- **奖励函数 \(R(s, a, s')\)**：立即反馈的标量，表达控制目标。  
  本项目典型形式：  
  \[
  r = -\big( k_1 e^2 + k_2 \omega^2 + k_3 \tau^2 \big)
  \]
- **状态转移概率 \(P\)**：环境的动力学。本项目为确定性刚体运动学。
- **折扣因子 \(\gamma \in [0,1]\)**：权衡短期与长期奖励。

### 目标
寻找策略 \(\pi^*\) 最大化期望回报：
\[
J(\pi) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t r_t \right]
\]

## 3. 策略与价值函数

- **策略 \(\pi\)**：决定在状态 \(s\) 下如何选择动作。  
  确定性策略：\(a = \mu_\theta(s)\) （DDPG）  
  随机策略：\(a \sim \pi_\theta(\cdot|s)\)，通常使用高斯分布（PPO）。
- **状态价值函数**：从状态 \(s\) 出发，按策略 \(\pi\) 能获得的期望回报：  
  \[
  V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t r_t \bigg| s_0 = s \right]
  \]
- **动作价值函数**：在状态 \(s\) 执行动作 \(a\) 后，继续按策略 \(\pi\) 获得的期望回报：  
  \[
  Q^\pi(s, a) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t r_t \bigg| s_0 = s, a_0 = a \right]
  \]
- **优势函数**：衡量某动作相对于平均的好坏：  
  \[
  A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)
  \]


## 4. 演员-评论家 (Actor-Critic) 方法

本项目使用一类统一框架：**Actor-Critic**。

- **Actor (演员)**：策略网络 \(\pi_\theta\)，直接生成动作。
- **Critic (评论家)**：价值网络 \(Q_\phi\) 或 \(V_\phi\)，评判动作好坏。

训练时两者相互促进：
- Critic 用 TD 误差学习逼近真实价值。
- Actor 使用 Critic 的梯度更新，朝高价值方向调整动作。

## 5. 算法 1：DDPG (Deep Deterministic Policy Gradient)

**适用**：连续动作空间，离线策略，确定性控制。

### 核心要点
- Actor 输出确定性动作：\(a = \mu_\theta(s)\)。
- Critic 评估 Q 值：\(Q_\phi(s, a)\)。
- 使用**经验回放**和**目标网络**稳定训练。
- Actor 通过最大化 Q 值更新：\(\mathcal{L}_a = -Q_\phi(s, \mu_\theta(s))\)。

### 算法伪代码
```
初始化 Actor μ_θ， Critic Q_φ， 目标网络 μ_θ', Q_φ'， 经验池 D
for episode = 1 to M:
    接收初始状态 s
    for t = 1 to T:
        a = μ_θ(s) + 噪声 N_t
        执行 a，得到 r, s', done
        存储 (s, a, r, s', done) 到 D
        从 D 采样 batch B = {(s_i, a_i, r_i, s'_i, done_i)}
        
        # 更新 Critic
        y_i = r_i + γ (1 - done_i) Q_φ'(s'_i, μ_θ'(s'_i))
        L_c = 均方误差(Q_φ(s_i, a_i), y_i)
        梯度下降更新 φ
        
        # 更新 Actor
        L_a = -均值(Q_φ(s_i, μ_θ(s_i)))
        梯度下降更新 θ
        
        # 软更新目标网络
        θ' ← τθ + (1-τ)θ'
        φ' ← τφ + (1-τ)φ'
        
        if done: break
```

### 优点与缺陷
- 样本效率高 (off-policy)。
- 对超参数敏感，易 Q 值爆炸 (需梯度裁剪、奖励缩放)。
- 确定性策略可能探索不足。

## 6. 算法 2：PPO (Proximal Policy Optimization)

**适用**：连续动作空间，在线策略，随机控制。

### 核心思想
限制策略更新幅度，防止策略崩溃。使用**截断目标函数**：

\[
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) A_t, \ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
\]

其中 \(r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}\) 是新旧策略概率比，\(\epsilon\) 是裁剪范围（典型 0.2）。

### 算法伪代码
```
初始化策略 π_θ， 价值 V_φ
for iteration = 1 to K:
    用当前 π_θ 收集 N 步轨迹 {s_t, a_t, r_t, s_{t+1}}
    计算优势估计 Â_t (例如用 GAE)
    
    for epoch = 1 to M:
        从轨迹中取小批量
        计算比率 r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)
        L_clip = min(r_t * Â_t, clip(r_t, 1-ε, 1+ε) * Â_t)
        L_v = MSE(V_φ(s_t), 实际回报)
        L_total = -L_clip + c1 * L_v - c2 * 熵(π_θ)
        梯度下降更新 θ, φ
    end for
end for
```

### 优点与缺陷
- **极稳定**，超参数鲁棒。
- 随机策略自带探索，无需人工噪声设计。
- 需更多样本 (on-policy)，但仿真廉价，本任务不成问题。
- 适合多轴扩展，社区首选。

## 7. 关键技巧与实现细节

### 奖励缩放
为防止Q值或梯度爆炸，将奖励乘以缩放因子，使单步奖励大致在 [-10, 10] 内。

### 状态归一化
将角度除以 π，角速度除以 10 rad/s，使输入在 [-1,1] 附近，加速收敛。

### 梯度裁剪
对 Critic 和 Actor 的梯度使用 `torch.nn.utils.clip_grad_norm_(..., max_norm=1.0)`。

### 经验回放 (DDPG)
- 容量：10^5 条经验。
- batch size：64~128。
- 预热：先纯随机动作填充 1000 条再更新。

### 探索噪声 (DDPG)
- 初始标准差 0.3，衰减至 0.05 (带下限) 。
- 避免噪声过早消失导致策略停滞。

### GAE 优势估计 (PPO)
\[
\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + \cdots + (\gamma\lambda)^{T-t+1}\delta_{T-1}
\]
其中 \(\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)\)，\(\lambda\) 控制偏差-方差权衡。

## 8. 你的项目路线图

1. **单轴 DDPG** (已完成验证) → 理解 Actor-Critic 模型。
2. **单轴 PPO** (推荐稳定版) → 使用 `Stable-Baselines3` 快速训练，对比 PID。
3. **三轴 + 四元数 PPO** → 状态 = 四元数 (4) + 角速度 (3)，动作 = 三轴力矩 (3)，奖励 = 欧拉角误差 + 能耗。
4. **逐步加入真实因素**：飞轮一阶动态、角动量饱和、传感器噪声 → 重新训练策略。

## 9. 公式速查表

| 名称 | 公式 |
|------|------|
| PID 控制律 | \(u = K_p e + K_i \int e + K_d \frac{de}{dt}\) |
| 回报 | \(G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}\) |
| TD 误差 | \(\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)\) |
| 贝尔曼方程 (Q) | \(Q^\pi(s,a) = \mathbb{E}[r + \gamma Q^\pi(s', a')]\) |
| DDPG Actor 损失 | \(\mathcal{L}_a = -\frac{1}{N}\sum Q_\phi(s_i, \mu_\theta(s_i))\) |
| PPO 截断目标 | \(L^{\text{CLIP}} = \min\big(r(\theta)A, \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon)A\big)\) |
| 高斯策略对数概率 | \(\log \pi_\theta(a|s) \propto -\frac{\|a - \mu_\theta(s)\|^2}{2\sigma^2}\) |

---

**延伸阅读**：Sutton & Barto 《强化学习》（第二版），OpenAI Spinning Up 文档，PPO 原始论文（Schulman et al., 2017）。

这份文档应能支撑你从 PID 的理解平滑过渡到强化学习的实现，并直接指导项目中的算法选择与调参。