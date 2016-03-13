import numpy as np
from pyglet.window import key
import random
import util
import math


class WorldFeedback(object):
    def getreward(self):
        raise NotImplementedError


class QLearner(object):
    _moves = [key.MOTION_DOWN, key.MOTION_LEFT, key.MOTION_RIGHT, key.MOTION_UP]  # Do nothing, Move left, Move right

    def __init__(self, board, worldfeedback, learningrate=0.01, discountfactor=0.6, epsilon=0.1):
        # initialize policy
        self.board = board
        self.feedback = worldfeedback
        self.lr = learningrate
        self.gamma = discountfactor
        self.epsilon = epsilon

        self.lastState = None
        self.lastAction = 0
        self.reset()

        self.policy = {}
        self._createpolicyentry(self.lastState)

    def _createpolicyentry(self, state):
        self.policy[state] = np.random.rand(len(self._moves))

    def reset(self):
        self.lastState = self.board.encode()
        self.lastAction = 0

    def newpiece(self):
        # self.lastAction = 0
        pass

    def softmax(self, state):
        p = [math.exp(a) for a in self.policy[state]]
        s = sum(p)
        return [v/s for v in p]

    def _nextaction(self, state):
        # return util.choosewithprob(self.softmax(state))
        if random.random() < self.epsilon:
            return random.randint(0, len(self.policy[state])-1)
        else:
            return np.argmax(self.policy[state])

    def _updatevalue(self, state, action, reward):
        self.policy[self.lastState][self.lastAction] += self.lr * (reward + self.gamma * self.policy[state][action] - self.policy[self.lastState][self.lastAction])

    def step(self):
        # encode board into state
        state = self.board.encode()

        # choose action from policy table
        if state not in self.policy:
            self._createpolicyentry(state)

        action = self._nextaction(state)

        reward = self.feedback.getreward()

        # print(len(self.policy), state)
        # print(len(self.policy), self.policy[state], reward)

        self._updatevalue(state, action, reward)

        self.lastState = state
        self.lastAction = action

        return self._moves[action]


class SarsaLambdaLearner(QLearner):
    def __init__(self, board, worldfeedback, learningrate=0.01, discountfactor=0.6, epsilon=0.1, lam=0.9):
        self.lam = lam
        #self.e = {}
        self.track = []
        super(SarsaLambdaLearner, self).__init__(board, worldfeedback, learningrate, discountfactor, epsilon)

    def newpiece(self):
        super(SarsaLambdaLearner, self).newpiece()
        # self.track = []

    def reset(self):
        super(SarsaLambdaLearner, self).reset()
        self.track = []

    def _createpolicyentry(self, state):
        super(SarsaLambdaLearner, self)._createpolicyentry(state)
        #self.e[state] = np.zeros((len(self._moves),) )

    def _updatevalue(self, state, action, reward):
        delta = reward + self.gamma * self.policy[state][action] - self.policy[self.lastState][self.lastAction]

        hit = False
        for i in range(len(self.track)):
            if self.track[i][1] == self.lastAction and self.track[i][0] == self.lastState:
                self.track[i][2] += 1
                hit = True
                break

        if not hit:
            self.track.append([self.lastState, self.lastAction, 1])

        clearids = set()

        for i in range(len(self.track)):
            s, a, e = self.track[i]
            self.policy[s][a] += self.lr * delta * e
            self.track[i][2] *= self.gamma * self.lam

            if self.track[i][2] < 1e-6:
                clearids.add(i)

        if len(clearids) > 0:
            self.track = [t for i, t in enumerate(self.track) if i not in clearids]

        # for s in self.policy.keys():
        #     for ai in range(len(self._moves)):
        #         self.policy[s][ai] += self.lr * delta * self.e[s][ai]
        #         self.e[s][ai] *= self.gamma * self.lam
