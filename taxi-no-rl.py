import gym
import numpy

env = gym.make('Taxi-v3').env
qTable = numpy.zeros([env.observation_space.n, env.action_space.n])
env.s = env.reset()

steps = 0
penalties = 0
reward = 0


done = False

while not done:
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    penalties += 1 if reward <= -10 else 0
    steps += 1
