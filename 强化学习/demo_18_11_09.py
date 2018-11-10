#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

def epsilon_greedy(s, epsilon):
    avalible_actions = []
    for a in range(4):
        if any(s + actions_all[a] < 0) or any(s + actions_all[a] > 4):
            continue
        avalible_actions.append(display[a])

    if np.random.rand() < epsilon:
        return avalible_actions[np.random.randint(len(avalible_actions))]
    return avalible_actions[np.array([qfunc["{}_{}".format(s, x)] for x in avalible_actions]).argmax()]
    # if isinstance(PI[s[0]][s[1]], str):
    #     return PI[s[0]][s[1]]
    # return avalible_actions[np.random.randint(len(avalible_actions))]
for s in np.argwhere(~over):
    for a in range(4):
        if any(s + actions_all[a] < 0) or any(s + actions_all[a] > 4):
            continue
        qfunc["{}_{}".format(s, display[a])] = .0
        # n["{}_{}".format(s, display[a])] = .0
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
