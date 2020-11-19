import gym
env = gym.make('CliffWalking-v0')

n_state = env.observation_space.n
print(n_state)

n_action = env.action_space.n
print(n_action)