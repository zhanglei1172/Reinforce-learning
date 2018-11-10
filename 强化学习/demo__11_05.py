#!/usr/bin/env python
# -*- coding: utf-8 -*-
# method: model-based and MC-error and value-iteration
import pickle
import numpy as np

BOARD_ROWS = 3
BOARD_COLS = 3
# 最多连续几个算赢
MAX_LINK = 3
BOARD_SIZE = BOARD_ROWS * BOARD_COLS
class State:

    def __init__(self):
        self.data = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.winner = None
        self.end = None
        self.hashVal = None

    # compute the hash value for one state, it's unique
    def getHash(self):
        if self.hashVal is None:
            self.hashVal = ''
            for i in self.data.reshape(BOARD_ROWS * BOARD_COLS):
                if i == -1:
                    i = 2
                self.hashVal += str(int(i))

                # self.hash_val = self.hash_val * 3 + i
        return self.hashVal

    def isEnd(self):
        if self.end is not None:
            return self.end
        results = []
        for i in range(BOARD_ROWS):     # 检查行
            results.append(np.sum(self.data[i, :]))
        for i in range(BOARD_COLS):     # 检查列
            results.append(np.sum(self.data[:, i]))
        results.append(0)
        # 检查对角线
        for i in range(BOARD_ROWS):
            results[-1] += self.data[i, i]
        results.append(0)
        for i in range(BOARD_ROWS):
            results[-1] += self.data[i, BOARD_ROWS-i-1]

        for i in results:
            if i == MAX_LINK:
                self.winner = 1
                self.end = True
                return self.end
            if i == -MAX_LINK:
                self.winner = -1
                self.end = True
                return self.end
        # 是否为平局
        sumed = np.sum(np.abs(self.data))
        if sumed == BOARD_ROWS * BOARD_COLS:
            self.winner =0
            self.end = True
            return self.end
        # 游戏继续
        self.end = False
        return self.end

    def show(self):
        for i in range(0, BOARD_ROWS):
            print('-------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                if self.data[i, j] == 1:
                    token = '*'
                elif self.data[i, j] == 0:
                    token = '0'
                elif self.data[i, j] == -1:
                    token = 'x'
                out += token + ' | '
            print(out)
        print('-------------')

        # @symbol 1 or -1
        # 把棋子放进位置(i, j)

    def nextState(self, i, j, symbol):
        newState = State()
        newState.data = np.copy(self.data)
        newState.data[i, j] = symbol
        return newState

def getAllStatesImpl(currentState, currentSymbol, allStates):
    for i in range(0, BOARD_ROWS):
        for j in range(0, BOARD_COLS):
            if currentState.data[i][j] == 0:
                newState = currentState.nextState(i, j, currentSymbol)
                newHash = newState.getHash()
                if newHash not in allStates.keys():
                    isEnd = newState.isEnd()
                    allStates[newHash] = (newState, isEnd)
                    if not isEnd:
                        getAllStatesImpl(newState, -currentSymbol, allStates) # next player

def getAllStates():
    currentSymbol = 1
    currentState = State()
    allStates = dict()
    allStates[currentState.getHash()] = (currentState, currentState.isEnd())
    getAllStatesImpl(currentState, currentSymbol, allStates)
    return allStates

# 所有可能的主板配置
allStates = getAllStates()

class AI_PLAYER:
    # @stepSize: 更新评估的步数
    # @exploreRate: 探索的概率
    def __init__(self, stepSize=0.1, exploreRate=0.1):
        self.allStates = allStates
        self.estimations = dict()
        self.stepSize = stepSize
        self.exploreRate = exploreRate
        self.states = []

    def reset(self):
        self.states = []

    def setSymbol(self, symbol):
        self.symbol = symbol
        for hash in self.allStates.keys():
            (state, isEnd) = self.allStates[hash]
            if isEnd:
                if state.winner == self.symbol:
                    self.estimations[hash] = 1.0
                else:
                    self.estimations[hash] = 0
            else:
                self.estimations[hash] = 0.5
        # 接受状态

    def feedState(self, state):
        self.states.append(state)

    def feedReward(self, reward):
        if len(self.states) == 0:
            return
        self.states = [state.getHash() for state in self.states]
        target = reward
        for latestState in reversed(self.states):
            value = self.estimations[latestState] + self.stepSize * (target - self.estimations[latestState])
            self.estimations[latestState] = value
            target = value   # MC-error
        self.states = []

    def takeAction(self):
        state = self.states[-1]
        nextStates = []
        nextPositions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if state.data[i, j] == 0:
                    nextPositions.append([i, j])
                    nextStates.append(state.nextState(i, j, self.symbol).getHash())
        if np.random.binomial(1, self.exploreRate): # greedy探索开启
            np.random.shuffle(nextPositions)
            # 不确定 截取法是否适合探索
            # 但总好过忘记所有步骤
            self.states = []
            action = nextPositions[0]
            action.append(self.symbol)  # size:([i, j, symnol])
            return action
        values = []
        for hash, pos in zip(nextStates, nextPositions): # 不探索
            values.append((self.estimations[hash], pos))
        np.random.shuffle(values)
        values.sort(key=lambda x: x[0], reverse=True)
        action = values[0][1]  # 取出最大的action里的位置
        action.append(self.symbol)
        return action

    def savePolicy(self):
        with open('optimal_policy_' + str(self.symbol), 'wb') as fw:
        # fw =
            pickle.dump(self.estimations, fw)
        # fw.close()

    def loadPolicy(self):
        with open('optimal_policy_' + str(self.symbol), 'rb') as fr:
            self.estimations = pickle.load(fr)
        # fr.close()

class Judger:
    # @player1: 第一个移动的玩家为1
    # @player2: 另一个玩家为-1
    # @feedback: 当为true的时候，两个玩家都会有reward，游戏结束
    def __init__(self, player1, player2, feedback=True):
        self.p1 = player1
        self.p2 = player2
        self.feedback = feedback
        self.currentPlayer = None
        self.p1Symbol = 1
        self.p2Symbol = -1
        self.p1.setSymbol(self.p1Symbol)
        self.p2.setSymbol(self.p2Symbol)
        self.currentState = State()
        self.allStates = allStates

        # 两个玩家的reward

    def giveReward(self):
        if self.currentState.winner == self.p1Symbol:
            self.p1.feedReward(1)
            self.p2.feedReward(0)
        elif self.currentState.winner == self.p2Symbol:
            self.p1.feedReward(0)
            self.p2.feedReward(1)
        else:
            self.p1.feedReward(0.1)
            self.p2.feedReward(0.5)

    def feedCurrentState(self):
        self.p1.feedState(self.currentState)
        self.p2.feedState(self.currentState)

    def reset(self):
        self.p1.reset()
        self.p2.reset()
        self.currentState = State()
        self.currentPlayer = None

    # @show: 如果为 True, 打印棋盘
    def play(self, show=False):
        self.reset()
        self.feedCurrentState()
        while True:
            # 设置当前玩家
            if self.currentPlayer == self.p1:
                self.currentPlayer = self.p2
            else:
                self.currentPlayer = self.p1
            if show:
                self.currentState.show()
            [i, j, symbol] = self.currentPlayer.takeAction()
            self.currentState = self.currentState.nextState(i, j, symbol)
            hashValue = self.currentState.getHash()
            self.currentState, isEnd = self.allStates[hashValue]
            self.feedCurrentState()
            if isEnd:
                if self.feedback:
                    self.giveReward()
                return self.currentState.winner

def train(epochs=2000):
    player1 = AI_PLAYER()#实例化玩家一
    player2 = AI_PLAYER()#实例化玩家二
    judger = Judger(player1, player2)#判断玩家一和玩家二胜负
    player1Win = 0.0#玩家一胜利场数
    player2Win = 0.0#玩家二胜利场数
    #epochs为训练次数
    for i in range(0, epochs):
        print("训练代数", i)
        winner = judger.play()
        if winner == 1:
            player1Win += 1
        if winner == -1:
            player2Win += 1
        judger.reset()#一盘对决结束就重置游戏
    print(player1Win / epochs)#玩家一胜率
    print(player2Win / epochs)#玩家二胜率
    player1.savePolicy()#保存策略
    player2.savePolicy()


class HumanPlayer:
    def __init__(self, stepSize = 0.1, exploreRate=0.1):
        self.symbol = None
        self.currentState = None
        return
    def reset(self):
        return
    def setSymbol(self, symbol):
        self.symbol = symbol
        return
    def feedState(self, state):
        self.currentState = state
        return
    def feedReward(self, reward):
        return
    def takeAction(self):
        data = int(input("输入你的位置:"))
        data -= 1
        i = data // int(BOARD_COLS)    #//表示整除，而/表示结果是浮点型
        j = data % BOARD_COLS          #余数
        if self.currentState.data[i, j] != 0:
            return self.takeAction()
        return (i, j, self.symbol)

def compete(turns=500):
    player1 = AI_PLAYER(exploreRate=0)#没有探索率
    player2 = AI_PLAYER(exploreRate=0)
    judger = Judger(player1, player2, False)
    player1.loadPolicy()#加载训练模型
    player2.loadPolicy()
    player1Win = 0.0#玩家一胜利场数
    player2Win = 0.0
    #tuens为测试回合数
    for i in range(0, turns):
        print("Epoch", i)
        winner = judger.play()
        if winner == 1:
            player1Win += 1
        if winner == -1:
            player2Win += 1
        judger.reset()
    print(player1Win / turns)#玩家一胜率
    print(player2Win / turns)

def play():
    while True:
        player1 = AI_PLAYER(exploreRate=0)
        player2 = HumanPlayer()#人类控制的玩家
        judger = Judger(player1, player2, False)
        player1.loadPolicy()
        winner = judger.play(True)
        if winner == player2.symbol:
            print("赢!")
        elif winner == player1.symbol:
            print("失败!")
        else:
            print("平!")

if __name__ == "__main__":
    train(5000)
    compete()
    play()