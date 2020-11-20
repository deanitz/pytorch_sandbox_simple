import gym
from windy_gridworld import WindyGridwoldEnv

env = WindyGridwoldEnv()

env.reset()
env.render()

for i in range(5):
    print(env.step(1))
    env.render()