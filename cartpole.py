import gym
import numpy

env = gym.make('CartPole-v1')

episodes = 50000
noise = 0.1
param = numpy.random.rand(4) * 2 - 1
bestReward = 0
for episode in range(episodes + 1):
	newparam = param + (numpy.random.rand(4) * 2 - 1) * noise
	reward = 0
	state = env.reset()
	for timestep in range(200):
		action = 0 if numpy.matmul(param, state) < 0 else 1
		state, reward, finish, info = env.step(action)
		env.render()
		if finish:
			break
	if reward == 200:
		break
	elif reward > bestReward:
		bestReward = reward
		param = newparam
