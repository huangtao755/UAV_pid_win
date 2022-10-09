#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author:
introduction:
cite:
"""
import random
from collections import deque

import numpy as np
import pandas as pd


class ReplayBuffer(object):
    """ storing data in order replaying for train algorithm"""

    def __init__(self, buffer_size, random_seed=123):
        """

        :param buffer_size:
        :param random_seed:
        """
        # size of minimize buffer is able to train
        self.buffer_size = buffer_size
        # counter for replay buffer
        self.count = 0
        # buffer, contain all data together
        self.buffer = deque()
        # used for random sampling
        random.seed(random_seed)
        # when count rise over the buffer_size, the train can begin
        self.isBufferFull = False
        # counter for episode
        self.episodeNum = 0
        # record the start position of each episode in buffer
        self.episodePos = deque()
        # record the sum rewards of steps for each episode
        self.episodeRewards = deque()

    def buffer_append(self, experience):
        """append data to buffer, should run each step after system update"""
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
            self.isBufferFull = False
        else:
            self.buffer.popleft()
            self.buffer.append(experience)
            self.isBufferFull = True

    def episode_append(self, rewards):
        """

        :param rewards:
        :return:
        """
        self.episodeNum += 1
        self.episodePos.append(self.count)
        self.episodeRewards.append(rewards)

    def size(self):
        """

        :return:
        """
        return self.count

    def buffer_sample_batch(self, batch_size):
        """

        :param batch_size:
        :return:
        """
        """sample a batch of data with size of batch_size"""
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        return batch

    def clear(self):
        """

        :return:
        """
        self.buffer.clear()
        self.count = 0
        self.episodeNum = 0
        self.episodePos.clear()
        self.episodeRewards.clear()


class DataRecord(object):
    """data record for show result"""

    def __init__(self, compatibility_mode=False):
        """

        :param compatibility_mode:
        """
        # new buffer, store data sepeartely with different episode, new in 0.3
        self.episodeList = list()
        self.bufferTemp = deque()
        self.compatibilityMode = compatibility_mode

        # counter for replay buffer
        self.count = 0

        # counter for episode
        self.episodeNum = 0

        # record the sum rewards of steps for each episode
        self.episodeRewards = deque()
        # record the average td error of steps for each episode
        self.episodeTdErr = deque()
        # record some sample of weights, once after episode
        self.episodeWeights = deque()
        # record some sample of weights, once each step, for observing the vary of weights
        self.weights = deque()

        if self.compatibilityMode:
            # buffer, contain all data together, discarded in 0.3
            self.buffer = deque()
            # record the start position of each episode in buffer, discarded in 0.3
            self.episodePos = deque()

    def buffer_append(self, experience, weights=0):
        """

        :param experience:
        :param weights:
        :return:
        """
        """append data to buffer, should run each step after system update"""
        self.bufferTemp.append(experience)
        self.count += 1
        self.weights.append(weights)

        if self.compatibilityMode:
            self.buffer.append(experience)
    #        if self.count < self.buffer_size:
    #            self.buffer.append(experience)
    #            self.count += 1
    #            self.isBufferFull = False
    #        else:
    #            self.buffer.popleft()
    #            self.buffer.append(experience)

    def episode_append(self, rewards=0, td_err=0, weights=0):
        """

        :param rewards:
        :param td_err:
        :param weights:
        :return:
        """
        """append data to episode buffer, should run each episode after episode finish"""
        self.episodeNum += 1
        self.episodeRewards.append(rewards)
        self.episodeTdErr.append(td_err)
        self.episodeWeights.append(weights)

        self.episodeList.append(self.bufferTemp)
        self.bufferTemp = deque()
        if self.compatibilityMode:
            self.episodePos.append(self.count)

    @staticmethod
    def save_data(path, data_name, data):
        """

        :param path:
        :param data_name:
        :param data:
        :return:
        """
        print('---------------------------------------')
        name = str(path + '//' + data_name + '.csv')
        print(name)
        pd.DataFrame(data).to_csv(name)
        print(data_name + 'data is saved')

    def get_episode_buffer(self, index=-1):
        """

        :param index:
        :return:
        """
        if index == -1:
            index = self.episodeNum - 1
        elif index > (self.episodeNum - 1):
            self.print_mess("Does not exist this episode!")
        else:
            # index = index
            return None

        buffer_temp = self.episodeList[index]
        data = list()
        item_len = len(buffer_temp[0])
        for ii in range(item_len):
            x = np.array([_[ii] for _ in buffer_temp])
            data.append(x)
        return data

    def size(self):
        """

        :return:
        """
        return self.count

    def clear(self):
        """

        :return:
        """
        self.count = 0
        self.episodeNum = 0
        self.episodeRewards.clear()
        self.bufferTemp.clear()
        self.episodeList.clear()
        if self.compatibilityMode:
            self.buffer.clear()
            self.episodePos.clear()

    @classmethod
    def print_mess(cls, mes=""):
        """

        :param mes:
        :return:
        """
        # implement with print or warning if the project exist
        print(mes)
