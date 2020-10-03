import gym
import numpy
import random

env = gym.make('Taxi-v3').env
qTable = numpy.zeros([env.observation_space.n, env.action_space.n])

episodes = 500
alpha = 0.3
gamma = 0.8
epsilon = 0.1

for episode in range(episodes + 1):
    state = env.reset()
    steps = 0
    penalties = 0
    reward = 0 
    finish = False
    while finish == False:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = numpy.argmax(qTable[state])
        nextState, reward, finish, info = env.step(action) 
        oldQVal = qTable[state, action]
        nextMaxQVal = numpy.max(qTable[nextState])
        newQVal = ((1 - alpha) * oldQVal) + (alpha * (reward + gamma * nextMaxQVal))
        qTable[state, action] = newQVal
        state = nextState
        penalties += 1 if reward <= -10 else 0
        steps += 1