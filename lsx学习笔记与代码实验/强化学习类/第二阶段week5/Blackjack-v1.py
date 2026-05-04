import gymnasium as gym

# 创建21点纸牌环境
env = gym.make("Blackjack-v1")
obs, info = env.reset()   # obs = (玩家点数, 庄家明牌, 是否有可用Ace)

print("初始状态:", obs)

# 执行一个随机动作（0: 要牌, 1: 停牌）
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
print(f"动作: {action}, 新状态: {obs}, 奖励: {reward}, 游戏结束: {terminated}")

print("观测空间:", env.observation_space)   # Tuple(Discrete(32), Discrete(11), Discrete(2))
print("动作空间:", env.action_space)        # Discrete(2)