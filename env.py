import numpy as np
import gym
from gym import spaces
from copy import deepcopy
import random


class Connect4Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, board_shape=(6, 7)):
        super(Connect4Env, self).__init__()
        self.board_shape = board_shape
        self.action_space = spaces.Discrete(board_shape[1])
        self.observation_space = spaces.Box(low=-1,
                                            high=1,
                                            shape=board_shape,
                                            dtype=np.int8)
        self.state = np.zeros([6, 7], dtype=np.int8)
        self.turn = random.choice([-1, 1])

    def reset(self):
        self.state = np.zeros([6, 7], dtype=np.int8)
        self.turn = random.choice([-1, 1])
        return self.state

    def step(self, action):
        self.state = self.make_move(self.state, action, self.turn)
        self.turn *= -1
        done, reward = self.get_reward(self.state)

        return self.state, reward, done, None

    def render(self, mode='human'):
        state = deepcopy(self.state).astype('str')
        print('Current Board Position: \n')
        for (m, n), symbol in np.ndenumerate(state):
            if symbol == '0':
                state[m][n] = ' '
            elif symbol == '1':
                state[m][n] = 'X'
            elif symbol == '-1':
                state[m][n] = 'O'
        print('  0   1   2   3   4   5   6')
        print(
            '  ' + str(state[0][0]) + ' | ' + str(state[0][1]) + ' | ' + str(state[0][2]) + ' | ' + str(
                state[0][3]) + ' | ' + str(state[0][4]) + ' | ' + str(state[0][5]) + ' | ' + str(
                state[0][6]) + '  ')
        print('-----------------------------')
        print(
            '  ' + str(state[1][0]) + ' | ' + str(state[1][1]) + ' | ' + str(state[1][2]) + ' | ' + str(
                state[1][3]) + ' | ' + str(state[1][4]) + ' | ' + str(state[1][5]) + ' | ' + str(
                state[1][6]) + '  ')
        print('-----------------------------')
        print(
            '  ' + str(state[2][0]) + ' | ' + str(state[2][1]) + ' | ' + str(state[2][2]) + ' | ' + str(
                state[2][3]) + ' | ' + str(state[2][4]) + ' | ' + str(state[2][5]) + ' | ' + str(
                state[2][6]) + '  ')
        print('-----------------------------')
        print(
            '  ' + str(state[3][0]) + ' | ' + str(state[3][1]) + ' | ' + str(state[3][2]) + ' | ' + str(
                state[3][3]) + ' | ' + str(state[3][4]) + ' | ' + str(state[3][5]) + ' | ' + str(
                state[3][6]) + '  ')
        print('-----------------------------')
        print(
            '  ' + str(state[4][0]) + ' | ' + str(state[4][1]) + ' | ' + str(state[4][2]) + ' | ' + str(
                state[4][3]) + ' | ' + str(state[4][4]) + ' | ' + str(state[4][5]) + ' | ' + str(
                state[4][6]) + '  ')
        print('-----------------------------')
        print(
            '  ' + str(state[5][0]) + ' | ' + str(state[5][1]) + ' | ' + str(state[5][2]) + ' | ' + str(
                state[5][3]) + ' | ' + str(state[5][4]) + ' | ' + str(state[5][5]) + ' | ' + str(
                state[5][6]) + '  ')
        print('-----------------------------')
        print('  0   1   2   3   4   5   6')
        print('\n')

    def close(self):
        pass

    @staticmethod
    def make_move(state, action, turn):
        col = state[:, action]
        row_index = (col == 0).sum() - 1
        state[row_index, action] = turn
        return state

    @staticmethod
    def get_reward(state):
        reward = 1
        for m in range(6):
            if state[m][0] == state[m][1] == state[m][2] == state[m][3] and state[m][0] != 0:
                return True, reward
            if state[m][1] == state[m][2] == state[m][3] == state[m][4] and state[m][1] != 0:
                return True, reward
            if state[m][2] == state[m][3] == state[m][4] == state[m][5] and state[m][2] != 0:
                return True, reward
            if state[m][3] == state[m][4] == state[m][5] == state[m][6] and state[m][3] != 0:
                return True, reward

        # check vertical
        for n in range(7):
            if state[0][n] == state[1][n] == state[2][n] == state[3][n] and state[0][n] != 0:
                return True, reward
            if state[1][n] == state[2][n] == state[3][n] == state[4][n] and state[1][n] != 0:
                return True, reward
            if state[2][n] == state[3][n] == state[4][n] == state[5][n] and state[2][n] != 0:
                return True, reward

        # check diagonal
        if state[0][0] == state[1][1] == state[2][2] == state[3][3] and state[0][0] != 0:
            return True, reward
        if state[1][1] == state[2][2] == state[3][3] == state[4][4] and state[1][1] != 0:
            return True, reward
        if state[2][2] == state[3][3] == state[4][4] == state[5][5] and state[2][2] != 0:
            return True, reward
        if state[1][0] == state[2][1] == state[3][2] == state[4][3] and state[1][0] != 0:
            return True, reward
        if state[2][1] == state[3][2] == state[4][3] == state[5][4] and state[2][1] != 0:
            return True, reward
        if state[2][0] == state[3][1] == state[4][2] == state[5][3] and state[2][0] != 0:
            return True, reward
        if state[0][1] == state[1][2] == state[2][3] == state[3][4] and state[0][1] != 0:
            return True, reward
        if state[1][2] == state[2][3] == state[3][4] == state[4][5] and state[1][2] != 0:
            return True, reward
        if state[2][3] == state[3][4] == state[4][5] == state[5][6] and state[2][3] != 0:
            return True, reward
        if state[0][2] == state[1][3] == state[2][4] == state[3][5] and state[0][2] != 0:
            return True, reward
        if state[1][3] == state[2][4] == state[3][5] == state[4][6] and state[1][3] != 0:
            return True, reward
        if state[0][3] == state[1][4] == state[2][5] == state[3][6] and state[0][3] != 0:
            return True, reward

        if state[5][0] == state[4][1] == state[3][2] == state[2][3] and state[5][0] != 0:
            return True, reward
        if state[4][1] == state[3][2] == state[2][3] == state[1][4] and state[4][1] != 0:
            return True, reward
        if state[3][2] == state[2][3] == state[1][4] == state[0][5] and state[3][2] != 0:
            return True, reward
        if state[4][0] == state[3][1] == state[2][2] == state[1][3] and state[4][0] != 0:
            return True, reward
        if state[3][1] == state[2][2] == state[1][3] == state[0][4] and state[3][1] != 0:
            return True, reward
        if state[3][0] == state[2][1] == state[1][2] == state[0][3] and state[3][0] != 0:
            return True, reward
        if state[5][1] == state[4][2] == state[3][3] == state[2][4] and state[5][1] != 0:
            return True, reward
        if state[4][2] == state[3][3] == state[2][4] == state[1][5] and state[4][2] != 0:
            return True, reward
        if state[3][3] == state[2][4] == state[1][5] == state[0][6] and state[3][3] != 0:
            return True, reward
        if state[5][2] == state[4][3] == state[3][4] == state[2][5] and state[5][2] != 0:
            return True, reward
        if state[4][3] == state[3][4] == state[2][5] == state[1][6] and state[4][3] != 0:
            return True, reward
        if state[5][3] == state[4][4] == state[3][5] == state[2][6] and state[5][3] != 0:
            return True, reward

        # Check if Draw
        if (state == 0).sum() == 0:
            return True, 0.

        return False, 0


if __name__ == '__main__':
    env = Connect4Env()
    state, reward, done, _ = env.step(0)
    env.render()