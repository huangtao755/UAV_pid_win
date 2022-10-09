import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from Algorithm.ClassicControl import PidControl
from Comman import MemoryStore
from Evn import QuadrotorFlyGui as Qgui
from Evn import QuadrotorFlyModel as Qfm

D2R = Qfm.D2R
current_path = os.path.abspath(os.path.join(os.getcwd(), ".."))


def cacualate_reward(state, action):
    """

    :param state:
    :param action:
    :return:
    """
    reward = 0
    return reward


def get_reward():
    """

    :return:
    """
    state = np.array(pd.read_csv(current_path + '\\DataSave\\QuadPid\\quad2_state.csv'))
    state_old = np.delete(state, -1, axis=0)
    state_new = np.delete(state, 0, axis=0)
    action = np.array(pd.read_csv(current_path + '\\DataSave\\QuadPid\\quad2_action.csv'))
    action = np.delete(action, -1, axis=0)
    reward = cacualate_reward(state=state_old, action=action)


if __name__ == "__main__":
    get_reward()