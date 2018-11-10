#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

alpha = 0.2
minGong = np.zeros((5,5))
REWARD = minGong - 1
REWARD[2, 4] = 10
REWARD[2, [0, 1]] = -10
REWARD[[0, 1], 3] = -10
REWARD[4, 2:5] = -10
over = REWARD != -1
# action_all = []
V = minGong
gamma = 0.9
actions_all = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
display = ['↑', '↓', '←', '→']
PI = minGong.astype('int').tolist()
n = dict()
qfunc = dict()
# PI
for s in np.argwhere(~over):
    for a in range(4):
        if any(s + actions_all[a] < 0) or any(s + actions_all[a] > 4):
            continue
        qfunc["{}_{}".format(s, display[a])] = .0
        # n["{}_{}".format(s, display[a])] = .0
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(25, 50)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(50, 4)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net().cuda(), Net().cuda()

        self.learn_step_counter = 0  # for target updating
        # self.memory_counter = 0  # for storing memory
        # self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))  # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters())
        self.loss_func = nn.MSELoss()

    # def choose_action(self, x):
    #     x = torch.unsqueeze(torch.FloatTensor(x).cuda(), 0)
    #     # input only one sample
    #     if np.random.uniform() < EPSILON:  # greedy
    #         actions_value = self.eval_net.forward(x)
    #         action = torch.max(actions_value, 1)[1].cpu().data.numpy()
    #         action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
    #     else:  # random
    #         action = np.random.randint(0, N_ACTIONS)
    #         action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
    #     return action
    def epsilon_greedy(self, s, epsilon):
        avalible_actions = []
        a__ = []
        for a in range(4):
            if any(s + actions_all[a] < 0) or any(s + actions_all[a] > 4):
                continue
            avalible_actions.append(display[a])
            a__.append(a)

        if np.random.rand() < epsilon:
            return avalible_actions[np.random.randint(len(avalible_actions))]
        actions_value = self.eval_net.forward(s)
        action = np.argmax([actions_value[x] for x in avalible_actions])
        return action
        # torch.argmax()
        # action = torch.max(actions_value, 1)[1].cpu().data.numpy()[0]
        # return avalible_actions[np.array([qfunc["{}_{}".format(s, x)] for x in avalible_actions]).argmax()]
    # if isinstance(PI[s[0]][s[1]], str):
    #     return PI[s[0]][s[1]]
    # return avalible_actions[np.random.randint(len(avalible_actions))]
    def learn(self):
        # target parameter update
        if self.learn_step_counter % 10 == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()  # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

for iter in range(100):
    seed = np.random.randint(over.size-over.sum()+1)-1
    s = np.argwhere(~over)[seed]
    t = False
    counted = 0


    s_sample, a_sample, r_sample = [], [], []
    while not t and counted <100:
        a = actions_all[display.index(epsilon_greedy(s, epsilon=0.1))] # 执行策略

        # s_sample.append(s)
        key_s = "{}_{}".format(s, display[np.apply_along_axis(np.array_equal, 1, actions_all, a).argmax()])

        s1 = s + a
        r = REWARD[s1[0], s1[1]]
        if over[s1[0], s1[1]] :
            qfunc[key_s] = qfunc[key_s] + alpha*(r - qfunc[key_s])
            t = True
            break
        a_ = actions_all[display.index(epsilon_greedy(s1, epsilon=0))] # 更新时在下一步中的a


        key_max = "{}_{}".format(s1, display[np.apply_along_axis(np.array_equal, 1, actions_all, a_).argmax()])
        qfunc[key_s] = qfunc[key_s] + alpha*(r + gamma*qfunc[key_max] - qfunc[key_s])
        counted += 1
        s = s1




for s in np.argwhere(~over):
    temp = []
    action_ = []
    for a in range(4):
        if any(s + actions_all[a] < 0) or any(s + actions_all[a] > 4):
            continue
        temp.append(qfunc["{}_{}".format(s, display[a])])
        action_.append(a)

    PI[s[0]][s[1]] = display[action_[np.array(temp).argmax()]]

# print(qfunc)
print(np.array(PI))