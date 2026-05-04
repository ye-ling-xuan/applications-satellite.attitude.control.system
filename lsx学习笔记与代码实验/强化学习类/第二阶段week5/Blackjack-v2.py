import gymnasium as gym

# 1. 创建21点环境
env = gym.make("Blackjack-v1")

# 2. 定义一个随机策略（随便选动作）
def policy(state):
    return env.action_space.sample()

# 3. 套交互循环模板
state, info = env.reset()
done = False
total_reward = 0

while not done:
    action = policy(state)
    next_state, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    state = next_state  # 别忘了更新状态！
    done = terminated or truncated

print(f"这一局的总奖励: {total_reward}")