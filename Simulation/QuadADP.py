import os

import numpy as np
import pandas as pd

from Algorithm.ADP_siglenet_p import ADPSingleNet
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

        self.replay_buffer = MemoryStore.ReplayBuffer(buffer_size=4999)
        self.replay_buffer.clear()

        self.state_old, self.action, self.state_new, self.reward = self.get_reward()
        print(len(self.state_old), len(self.action), len(self.state_new), len(self.reward), 'length')

        self.adp = ADPSingleNet(evn=self.quad, replay_buffer=self.replay_buffer)

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

    def get_reward(self):
        """

        :return:
        """
        rewards = []
        err_state = np.array(pd.read_csv(current_path + '//DataSave//QuadPid//quad1_state_err.csv'))[:, 1:]
        state_old = np.delete(err_state, -1, axis=0)
        state_new = np.delete(err_state, 0, axis=0)
        action = np.array(pd.read_csv(current_path + '//DataSave//QuadPid//quad1_action.csv'))[:, 1:]
        action = np.delete(action, -1, axis=0)

        for i in range(len(state_old)):
            rewards.append(self.calculate_reward(state_old[i], action[i]))
            self.replay_buffer.buffer_append(np.hstack((state_old[i], action[i], state_new[i], rewards[i])))
        self.replay_buffer.episode_append(rewards=None)
        data = self.replay_buffer.buffer_sample_batch(batch_size=len(state_old))
        MemoryStore.DataRecord.save_data(path=current_path + '//DataSave//QuadPid',
                                         data_name='quad1_reward', data=rewards)
        self.replay_buffer.save_data(path=current_path + '//DataSave//QuadADP',
                                     data_name='replay_buffer', data=data)
        return state_old, action, state_new, rewards


if __name__ == "__main__":
    quad_adp = QuadADP()
    data = quad_adp.replay_buffer.buffer_sample_batch(batch_size=3)
