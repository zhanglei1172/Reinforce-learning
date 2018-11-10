#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


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
    # if all(s == 0):
    #     print(1)

    avalible_actions = []
    for a in range(4):
        if any(s + actions_all[a] < 0) or any(s + actions_all[a] > 4):
            continue
        avalible_actions.append(display[a])
        # qfunc["{}_{}".format(s, display[a])] = .0
        # n["{}_{}".format(s, display[a])] = .0
    if np.random.rand() < epsilon:
        return avalible_actions[np.random.randint(len(avalible_actions))]
    if isinstance(PI[s[0]][s[1]], str):
        return PI[s[0]][s[1]]
    return avalible_actions[np.random.randint(len(avalible_actions))]
for s in np.argwhere(~over):
    for a in range(4):
        if any(s + actions_all[a] < 0) or any(s + actions_all[a] > 4):
            continue
        qfunc["{}_{}".format(s, display[a])] = .0
        n["{}_{}".format(s, display[a])] = .0
for iter in range(500):
    seed = np.random.randint(over.size-over.sum()+1)-1
    s = np.argwhere(~over)[seed]
    t = False
    counted = 0


    s_sample, a_sample, r_sample = [], [], []
    while not t and counted <100:
        a = actions_all[display.index(epsilon_greedy(s, epsilon=0.1))]
        s_sample.append(s)
        s = s + a
        r = REWARD[s[0], s[1]]
        a_sample.append(display[actions_all.tolist().index(a.tolist())])
        r_sample.append(r)
        counted += 1
        if over[s[0], s[1]] :
            t = True

    g = .0
    for i in range(len(s_sample)-1, -1 , -1):
        g = g*gamma
        g += r_sample[i]
    for i in range(len(s_sample)):
        key = "{}_{}".format(s_sample[i], a_sample[i])
        n[key] += 1.
        qfunc[key] = (qfunc[key]*(n[key]-1)+g)/n[key]
        g -= r_sample[i]
        g /= gamma

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
