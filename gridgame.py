from gym.envs.toy_text import discrete
import os
import sys
import numpy as np  

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class GridgameEnv(discrete.DiscreteEnv):
    def __init__(self, shape=[3,3]):
        self.shape = shape


        MAX_Y = shape[0]
        MAX_X = shape[1]

        nS = np.prod(shape)
        nA = 4

        grid = np.arange(nS).reshape(shape)
        it = np.nditer(grid, flags=['multi_index'])

        P = {}


        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            P[s] = {a : [] for a in range(nA)}

            is_done = lambda s: s == 0 or s == (nS - 1)
            reward = 0.0 if is_done(s) else -1.0

            # Get in a terminal state
            if is_done(s):
                P[s][UP] = [(1.0, s, reward, True)]
                P[s][RIGHT] = [(1.0, s, reward, True)]
                P[s][DOWN] = [(1.0, s, reward, True)]
                P[s][LEFT] = [(1.0, s, reward, True)]
            # Not a terminal state
            else:
                ns_up = s if y == 0 else s - MAX_X
                ns_right = s if x == (MAX_X - 1) else s + 1
                ns_down = s if y == (MAX_Y - 1) else s + MAX_X
                ns_left = s if x == 0 else s - 1
                P[s][UP] = [(1.0, ns_up, reward, is_done(ns_up))]
                P[s][RIGHT] = [(1.0, ns_right, reward, is_done(ns_right))]
                P[s][DOWN] = [(1.0, ns_down, reward, is_done(ns_down))]
                P[s][LEFT] = [(1.0, ns_left, reward, is_done(ns_left))]

            it.iternext()

            isd = np.ones(nS) / nS

            self.P = P

            super(GridgameEnv, self).__init__(nS, nA, P, isd)


