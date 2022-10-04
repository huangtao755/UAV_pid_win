# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The file used to implement the data store and replay

By xiaobo
Contact linxiaobo110@gmail.com
Created on Wed Jan 17 10:40:44 2018
"""
import numpy as np
import torch
import torch as t
from torch.autograd import Variable

# Copyright (C)
#
# This file is part of QuadrotorFly
#
# GWpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GWpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GWpy.  If not, see <http://www.gnu.org/licenses/>.

"""
********************************************************************************************************
**-------------------------------------------------------------------------------------------------------
**  Compiler   : python 3.6
**  Module Name: MemoryStore
**  Module Date: 2018-04-17
**  Module Auth: xiaobo
**  Version    : V0.1
**  Description: create the module
**-------------------------------------------------------------------------------------------------------
**  Reversion  : V0.2
**  Modified By: xiaobo
**  Date       : 2019-4-25
**  Content    : rewrite the module, add note
**  Notes      :
"""

"""
*********************************************************************************************************
Define nural network
*********************************************************************************************************
"""

t.manual_seed(4)

state_dim = 12
v_dim = 1
action_dim = 4
learning_rate = 0.005
learning_num = 1000
sim_num = 20
x0 = np.array([2, -1])
epislon = 0.0001
Fre_V1_paras = 5  # 考虑


############################################################################################################
# 定义网络
############################################################################################################
class Model(t.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lay1 = t.nn.Linear(state_dim, 10, bias=False)  # 线性层
        self.lay1.weight.data.normal_(0, 0.5)  # 权重初始化
        self.lay2 = t.nn.Linear(10, 1, bias=False)  # 线性层
        self.lay2.weight.data.normal_(0, 0.5)  # 权重初始化

    def forward(self, x):
        layer1 = self.lay1(x)  # 第一隐层
        layer1 = t.nn.functional.relu(layer1)  # relu激活函数
        output = self.lay2(layer1)  # 输出层
        return output


class HDP_P():
    def __init__(self, env):
        self.env = env

        self.V1_model = Model()  # 定义V1网络
        self.V2_model = Model()  # 定义V2网络
        self.A_model = Model()  # 定义A网络
        self.criterion = t.nn.MSELoss(reduction='mean')  # 平方误差损失

        # 训练一定次数，更新Critic Net的参数
        # 这里只需要定义A网络和V2网络的优化器
        self.optimizerV2 = t.optim.SGD(self.V2_model.parameters(), lr=learning_rate)  # 利用梯度下降算法优化model.parameters
        self.optimizerA = t.optim.SGD(self.A_model.parameters(), lr=learning_rate)  # 利用梯度下降算法优化model.parameters

        # 采样状态  将状态定义在x1 [-2,2]   x2 [-1,1]
        x = np.arange(-2, 2, 0.1)
        y = np.arange(-1, 1, 0.1)
        xx, yy = np.meshgrid(x, y)  # 为一维的矩阵
        self.state = np.transpose(np.array([xx.ravel(), yy.ravel()]))  # 所有状态
        self.state_num = self.state.shape[0]  # 状态个数

        # 动作采样  将输入定在[-10 10] 内
        self.action = np.arange(-10, 10, 0.1)
        self.cost = []  # 初始化误差矩阵

    def reward(self, sk, uk):
        Q = t.eye(sk.shape)
        R = t.eye(uk.shape)
        reward = t.mm(t.mm(sk.T, Q), sk) + t.mm(t.mm(uk.T, R), uk)
        return reward

    def J_loss(self, sk, uk, Vk_1):
        Vk = np.zeros(uk.shape[0])
        for i in range(uk.shape[0]):
            Vk[i] = self.reward(sk, uk) + Vk_1[i]
        return Vk

    def learning(self):
        for train_index in range(learning_num):
            print("the", train_index + 1, "--th learning start")

            last_V_value = self.V2_model(Variable(t.Tensor(self.state)))

            ############################################################################################################
            # 更新Critic网络
            ############################################################################################################
            V2_predict = self.V2_model(Variable(t.Tensor(self.state)))  # 估值网络的预测值

            la_u = self.A_model(Variable(t.Tensor(self.state)))  # 计算动作输出
            la_next_state = self.env(self.state, self.la_u)
            V2_target = np.zeros([self.state_num, 1])
            for index in range(self.state_num):
                next_V1 = self.V1_model(Variable(t.Tensor(la_next_state[index, :])))
                V2_target[index] = self.reward(self.state, la_u.dara) + next_V1

            V2_loss = self.criterion(V2_predict, Variable(t.tensor(V2_target)))
            self.optimizerV2.zero_grad()
            V2_loss.backward()
            self.optimizerV2.step()

            print('--------the', train_index + 1, 'Critic Net have updated---------')

            ############################################################################################################
            # 更新Actor网络
            ############################################################################################################
            A_predict = self.A_model(Variable(t.Tensor(self.state)))  # 网络输出值

            A_target = np.zeros([self.state_num, 1])
            for index in range(self.state_num):
                new_state = np.tile(self.state[index, :], (self.action.shape[0], 1))
                new_next_state = self.evn(new_state, self.action)
                next_V1 = self.V1_model(Variable(t.Tensor(new_next_state)))
                A1 = self.J_loss(self.state[index, :], self.action, next_V1.data)
                A_target_index = np.argmin(A1)
                A_target[index] = self.action[A_target_index]

            A_loss = self.criterion(A_predict, Variable(torch.Tensor(A_target)))
            self.optimizerA.zero_grad()
            A_loss.backward()
            self.optimizerA.step()

            print('--------the', train_index + 1, 'Actor Net have updated---------')

            if (train_index + 1) % Fre_V1_paras == 0:
                self.V1_model = self.V2_model
                print('-------Use V2 Net update V1 Net-------')

            print('A paras:\n', list(self.A_model.named_parameters()))
            print('V1 paras:\n', list(self.V1_model.named_parameters()))
            print('V2 paras:\n', list(self.V2_model.named_parameters()))

            V_value = self.V2_model(Variable(torch.Tensor(self.state))).data
            eor = np.abs(V_value) - np.abs(last_V_value)
            print('误差提升', eor)
            dis = np.sum(np.array(eor.reshape(self.state_num)))
            self.cost.append(np.abs(dis))
            print('-------deta(V)', np.abs(dis))
            if np.abs(dis) < epislon:
                print('Loss 小于阈值，退出训练')
                self.V1_model = self.V2_model
                break
        # torch.save(model_objec, 'model.pth')
        # model = torch.load('model.pth')
