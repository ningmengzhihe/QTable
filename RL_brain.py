"""
comment by lihan
Q learning with q_table algorithm
"""

import numpy as np
import pandas as pd


class QLearningTable:
    def __init__(self,
                 actions,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions)

    def choose_action(self, observation):
        # 首先检查observation是否在q_table当中，如果不再那么将observation加入到q_table当中
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)

        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action


    def learn(self, s, a, r, s_):
        """
        学习Q Table，也就是更新Q Table
        :param s:
        :param a:
        :param r:
        :param s_:
        :return:
        """
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            # next state
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            # next state is terminal
            q_target = r

        # update
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)


    def check_state_exist(self, state):
        """
        给q_table添加新的state
        :param state:
        :return:
        """
        if state not in self.q_table.index:
            # append new state with 0 value to q table
            df = pd.DataFrame(
                np.array([[0]*len(self.actions)]),
                index=[state],
                columns=self.q_table.columns
            )
            self.q_table = pd.concat([self.q_table, df])

            # 下面代码有个warning，所以改成concat
            # FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.self.q_table = self.q_table.append(
            # self.q_table = self.q_table.append(
            #     pd.Series(
            #         [0]*len(self.actions),
            #         index=self.q_table.columns,
            #         name=state,
            #     )
            # )

