#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import gym

# Hyper Parameters
BATCH_SIZE = 320
LR = 0.01                   # learning rate
EPSILON = 0.1               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 2000
env = gym.make('CartPole-v1')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape

class Net(nn.Module):

    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        # self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(50, N_ACTIONS)
        self.fc2.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action_value = self.fc2(x)
        return action_value

class DQN:
    def __init__(self):
        self.eval_net, self.target_net = Net().cuda(), Net().cuda()

        self.learn_step_counter = 0
        self.memorry_counter = 0
        self.memorry = np.zeros((MEMORY_CAPACITY, N_STATES*2 + 2))
        self.optimizor = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x).cuda(), 0)
        if np.random.uniform() < EPSILON:
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)

        else:
            action_value = self.eval_net.forward(x)
            action = torch.argmax(action_value, 1).cpu().numpy()
            # action = actopn
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)

        return action

    def store_memorry(self, s, a, r, s_):
        transition = np.concatenate((s, [a, r], s_), -1)
        index = self.memorry_counter % MEMORY_CAPACITY
        self.memorry[index, :] =transition
        self.memorry_counter += 1

    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memorry = self.memorry[sample_index, :]
        b_s = torch.FloatTensor(b_memorry[:, :N_STATES]).cuda()
        b_a = torch.LongTensor(b_memorry[:, N_STATES:N_STATES+1].astype('int')).cuda()
        b_r = torch.FloatTensor(b_memorry[:, N_STATES+1:N_STATES+2]).cuda()
        b_s_ = torch.FloatTensor(b_memorry[:, -N_STATES:]).cuda()

        q_eval =  self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA*q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizor.zero_grad()
        loss.backward()
        self.optimizor.step()

model = DQN()
print("\n收集经验")
for i_episode in range(400):
    s = env.reset()
    ep_r = 0

    while True:
        a = model.choose_action(s)

        s_, r, done, _ = env.step(a)

        # modify the reward
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        model.store_memorry(s, a, r, s_)

        ep_r += r

        if model.memorry_counter > BATCH_SIZE:
            model.learn()
            if done:
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(ep_r, 2))
        if done:
            break
        s = s_
