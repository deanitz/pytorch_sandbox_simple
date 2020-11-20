from operator import pos
from gym.envs.toy_text import discrete
import numpy as np
import sys

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class WindyGridwoldEnv(discrete.DiscreteEnv):
    def __init__(self) -> None:
        self.shape = (7, 10)
        nS = self.shape[0] * self.shape[1]
        nA = 4

        # Windy columns:
        winds = np.zeros(self.shape)
        winds[:,[3, 4, 5, 8]] = 1
        winds[:,[6, 7]] = 2

        self.goal = (3, 7)

        # Trans probs and rewards:
        P = {}
        for state in range(nS):
            position = np.unravel_index(state, self.shape)
            P[state] = { a: [] for a in range(nA) }
            P[state][UP] = self._calculate_trans_prob(position, [-1, 0], winds)
            P[state][RIGHT] = self._calculate_trans_prob(position, [0, 1], winds)
            P[state][DOWN] = self._calculate_trans_prob(position, [1, 0], winds)
            P[state][LEFT] = self._calculate_trans_prob(position, [0, -1], winds)

        # Start state:
        # agent in (3, 0)
        isd = np.zeros(nS)
        isd[np.ravel_multi_index((3, 0), self.shape)] = 1.0
        
        super(WindyGridwoldEnv, self).__init__(nS, nA, P, isd)

    def _calculate_trans_prob(self, current, delta, winds):
        # Prob is always 1.0!
        wind_delta = np.array([-1, 0])
        new_position = np.array(current) + np.array(delta) +\
            (wind_delta * winds[tuple(current)])
        new_position = self._limit_coordinates(new_position.astype(int))

        new_state = np.ravel_multi_index(tuple(new_position), self.shape)

        is_done = tuple(new_position) == self.goal
        return [(1.0, new_state, -1.0, is_done)]

    def _limit_coordinates(self, position):
        position[0] = min(position[0], self.shape[0] - 1)
        position[0] = max(position[0], 0)
        position[1] = min(position[1], self.shape[1] - 1)
        position[1] = max(position[1], 0)

        return position

    def render(self):
        outfile = sys.stdout
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            if self.s == s:
                output = ' x '
            elif position == self.goal:
                output = ' П '
            else:
                output = ' o '

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += '\n'
            outfile.write(output)
        outfile.write('\n')


