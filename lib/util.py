import matplotlib.pyplot as plt
import time
import random

_base36 = [ '0','1','2','3','4','5','6','7','8','9','A',
            'B','C','D','E','F','G','H','I','J','K','L',
            'M','N','O','P','Q','R','S','T','U','V','W',
            'X','Y','Z']

def num2base36(v):
    result = _base36[v % 36]

    while (v / 36) > 0:
        v /= 36
        result = _base36[v % 36] + result

    return result


def choosewithprob(p):
    dice = random.random() * sum(p)
    cum = 0

    for i in range(len(p)):
        if cum >= dice:
            return i

        cum += p[i]

    return len(p)-1


class Scoreplot(object):
    def __init__(self):
        self.scores = []
        self.x = []
        self.fig = plt.figure()
        plt.xlabel('Game')
        plt.ylabel('Avg Score')
        plt.ion()
        plt.show()

    def newscore(self, game, score):
        self.scores.append(score)
        #self.x.append(len(self.scores))
        self.x.append(game)

    def updatescore(self, game, score):
        self.scores[game] = score

    def plot(self):
        # plt.figure(self.fig.number)
        self.fig.clear()
        plt.xlabel('Game')
        plt.ylabel('Score')
        plt.plot(self.x, self.scores)
        plt.draw()
        time.sleep(0.01)
