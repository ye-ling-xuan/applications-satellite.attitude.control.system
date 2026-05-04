import gymnasium as gym

# 1. 创建环境（带窗口显示）
env = gym.make("CartPole-v1", render_mode="human")

# 统计变量
total_episodes = 10  # 玩10局
all_rewards = []     # 保存每一局的总奖励

# 2. 循环玩多局游戏
for episode in range(total_episodes):
    obs, info = env.reset()  # 每局开始都重置
    done = False
    episode_reward = 0       # 本局奖励
    
    # 3. 单局循环
    while not done:
        action = env.action_space.sample()  # 随机动作
        obs, reward, terminated, truncated, info = env.step(action)
        
        episode_reward += reward  # 累加本局奖励
        done = terminated or truncated  # 判断是否结束
    
    # 一局结束，保存并打印
    all_rewards.append(episode_reward)
    print(f"第 {episode+1} 局 得分: {episode_reward}")

# 4. 最终统计
print("\n===== 统计结果 =====")
print(f"每局得分: {all_rewards}")
print(f"平均得分: {sum(all_rewards)/len(all_rewards):.2f}")

# 关闭环境
env.close()