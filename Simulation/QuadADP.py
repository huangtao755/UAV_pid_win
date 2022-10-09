import os

import numpy as np
import pandas as pd

from Comman import MemoryStore
from Evn import QuadrotorFlyModel as Qfm

D2R = Qfm.D2R
current_path = os.path.abspath(os.path.join(os.getcwd(), ".."))


class QuadADP(object):
    def __init__(self):
        """

        """
        structure_type = Qfm.StructureType.quad_x
        init_att = np.array([0, 0, 0])
        init_pos = np.array([0, 0, 0])
        name = 'quad'
        self.uav_para = Qfm.QuadParas(structure_type=structure_type)
        self.sim_para = Qfm.QuadSimOpt(init_mode=Qfm.SimInitType.fixed, init_att=init_att,
                                       init_pos=init_pos)
        self.quad = Qfm.QuadModel(self.uav_para, self.sim_para)

        # get state and action
        err_state = np.array(pd.read_csv(current_path + '//DataSave//QuadPid//quad1_state_err.csv'))[:, 1:]
        self.state_old = np.delete(err_state, -1, axis=0)
        self.state_new = np.delete(err_state, 0, axis=0)
        action = np.array(pd.read_csv(current_path + '//DataSave//QuadPid//quad1_action.csv'))[:, 1:]
        self.action = np.delete(action, -1, axis=0)

        self.get_reward(state=self.state_old, action=self.action, ts=self.uav_para.ts)
        self.reward = np.array(pd.read_csv(current_path + '//DataSave//QuadPid//quad1_reward.csv'))[:, 1:]
        print(len(self.state_old), len(self.state_new), len(self.action), len(self.reward))

        self.replay_buffer = MemoryStore.ReplayBuffer(buffer_size=4999)
        self.replay_buffer.clear()
        for i in range(len(self.reward)):
            self.replay_buffer.buffer_append((self.state_old, self.action, self.state_new, self.reward))
        self.replay_buffer.episode_append(rewards=None)
        print(self.replay_buffer.count)

    @staticmethod
    def calculate_reward(state, action):
        """

        :param state:
        :param action:
        :return:
        """
        Q = np.eye(len(state))
        R = 0.3 * np.eye(len(action))
        reward = state.dot(Q).dot(state.T) + action.dot(R).dot(action.T)
        return reward

    def get_reward(self, state, action, ts):
        """

        :return:
        """
        reward = []
        record_reward = MemoryStore.DataRecord()
        record_reward.clear()

        err_state = np.array(pd.read_csv(current_path + '//DataSave//QuadPid//quad1_state_err.csv'))[:, 1:]
        t = np.array(range(0, len(state))) * ts
        state = np.delete(err_state, -1, axis=0)
        action = np.array(pd.read_csv(current_path + '//DataSave//QuadPid//quad1_action.csv'))[:, 1:]
        action = np.delete(action, -1, axis=0)
        for i in range(len(state)):
            reward.append(self.calculate_reward(state[i], action[i]))
            record_reward.buffer_append(reward[i])
        record_reward.episode_append()
        record_reward.save_data(path=current_path + '//DataSave//QuadPid', data_name='quad1_reward', data=reward)

        # fig = plt.figure(1)
        # plt.plot(t, action)
        # fig2 = plt.figure(2)
        # plt.plot(t, reward)
        # plt.show()

    def get_replay_buffer(self):
        pass


if __name__ == "__main__":
    QuadADP()
