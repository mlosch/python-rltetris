import numpy as np
from pyglet.window import key
import random


class WorldFeedback(object):
    def getreward(self):
        raise NotImplementedError


class QLearner(object):
    _moves = [key.MOTION_DOWN, key.MOTION_LEFT, key.MOTION_RIGHT, key.MOTION_UP]  # Do nothing, Move left, Move right

    def __init__(self, board, worldfeedback, learningrate=0.01, discountfactor=0.6):
        # initialize policy
        self.board = board
        self.feedback = worldfeedback
        self.lr = learningrate
        self.gamma = discountfactor

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

    def nextaction(self):
        # encode board into state
        state = self.board.encode()

        # choose action from policy table
        if state not in self.policy:
            self._createpolicyentry(state)

        if random.random() < 0.0:
            action = random.randint(0, len(self.policy[state])-1)
        else:
            action = np.argmax(self.policy[state])

        reward = self.feedback.getreward()

        # print(len(self.policy), state)
        # print(len(self.policy), self.policy[state], reward)

        self.policy[self.lastState][self.lastAction] += self.lr * (reward + self.gamma * self.policy[state][action] - self.policy[self.lastState][self.lastAction])

        self.lastState = state
        self.lastAction = action

        return self._moves[action]
