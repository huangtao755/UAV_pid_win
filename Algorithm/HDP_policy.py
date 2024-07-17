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


class PIDSMADP(object):
    def __init__(self):
        self.kp = 0.
        self.kd = 0.

        self.u_kp = 0
        self.u_kd = 0
        self.u = np.array([self.u_kp, self.u_kd])

        self.r = np.diag([1, 1])

        self.w = 0.01*np.ones(6)
        self.phi = np.zeros(6)
        self.d_phi = np.zeros((3, 6))

    def get_phi(self, state):
        s = state[0]
        kp = state[1]
        kd = state[2]
        self.phi = np.array([s**2, kp**2, kd**2, s*kp, s*kd, kp*kd])

    def get_d_phi(self, state):
        s = state[0]
        kp = state[1]
        kd = state[2]
        self.d_phi = np.array([[2*s, 0, 0, kp, kd, 0],
                               [0, 2*kp, 0, s, 0, kd],
                               [0, 0, 2*kd, 0, s, kp]])

    def u_star(self, state, dt=0.01):
        self.get_d_phi(state)

        self.u_kp = -1/2 / self.r[0, 0] * self.d_phi[1, :].dot(self.w)
        self.u_kd = -1/2 / self.r[1, 1] * self.d_phi[2, :].dot(self.w)

        print(self.u_kp, 'u_kp')
        print(self.u_kp > 0)
        print(self.kp, 'kp_old')
        self.kp = self.kp + self.u_kp*dt

        print(self.kp, 'kpp')

        self.kd = self.kd + self.u_kd * dt
        self.u = np.array([self.u_kp, self.u_kd])
        return self.u_kp, self.u_kd

    def reset(self):
        self.kp = 0
        self.kd = 0

    def trainI(self, state, state_, r, v, dt=0.01, lr=0.001):
        r = np.array(r)
        v = np.array(v)
        r = r + v.dot(self.r).dot(v.T)

        self.get_phi(state)
        phi = self.phi
        self.get_phi(state_)
        phi_ = self.phi
        delta_phi = phi_ - phi
        print(delta_phi, 'delta_phi')
        e_h = r * dt + self.w @ delta_phi + np.random.randn(1)*0.00

        m = delta_phi.T@delta_phi + 1
        print(m, 'm')
        self.w = self.w - lr * delta_phi/m**2*e_h
        print(self.w, 'w')


class SMADP(object):
    def __init__(self):
        self.u = 0.

        self.v = 0.

        self.r = 0.1

        self.w = 0.0001*np.ones(3)
        self.phi = np.zeros(3)
        self.d_phi = np.zeros((2, 3))

    def get_phi(self, state):
        s = state[0]
        u = state[1]
        self.phi = np.array([s**2, u**2, s*u])

    def get_d_phi(self, state):
        s = state[0]
        u = state[1]
        self.d_phi = np.array([[2*s, 0, u],
                               [0, 2*u, s]])

    def u_star(self, state, dt=0.01):
        self.get_d_phi(state)

        self.v = -1/2 / self.r * self.d_phi[1, :].dot(self.w)

        print(self.u_kp, 'u_kp')
        print(self.u_kp > 0)
        print(self.kp, 'kp_old')
        self.kp = self.kp + self.u_kp*dt

        print(self.kp, 'kpp')

        self.kd = self.kd + self.u_kd * dt
        self.u = np.array([self.u_kp, self.u_kd])
        return self.u_kp, self.u_kd

    def reset(self):
        self.kp = 0
        self.kd = 0

    def trainI(self, state, state_, r, v, dt=0.01, lr=0.01):
        r = np.array(r)
        v = np.array(v)
        r = r + v.dot(self.r).dot(v.T)

        self.get_phi(state)
        phi = self.phi
        self.get_phi(state_)
        phi_ = self.phi
        delta_phi = phi_ - phi
        print(delta_phi, 'delta_phi')
        e_h = r * dt + self.w @ delta_phi + np.random.randn(1)*0.001

        m = delta_phi.T@delta_phi + 1
        print(m, 'm')
        self.w = self.w - lr * delta_phi/m**2*e_h
        print(self.w, 'w')


class PSMADP(object):
    def __init__(self):
        self.kp = 0.

        self.u_kp = 0
        self.u = np.array([self.u_kp])

        self.r = np.diag([1])

        self.w = 0.01*np.ones(3)
        self.phi = np.zeros(3)
        self.d_phi = np.zeros((3, 3))

    def get_phi(self, state):
        s = state[0]
        kp = state[1]
        self.phi = np.array([s**2, kp**2, s*kp])

    def get_d_phi(self, state):
        s = state[0]
        kp = state[1]
        self.d_phi = np.array([[2*s, 0, kp],
                               [0, 2*kp, s]])

    def u_star(self, state, dt=0.01):
        self.get_d_phi(state)
        print(self.r)
        print(self.d_phi[1, :])
        print(self.w)
        self.u_kp = -1/2 / self.r[0] * self.d_phi[1, :].dot(self.w)

        print(self.u_kp, 'u_kp')
        print(self.u_kp > 0)
        print(self.kp, 'kp_old')
        self.kp = self.kp + self.u_kp*dt

        print(self.kp, 'kpp')

        self.u = np.array([self.u_kp])
        return self.u_kp

    def reset(self):
        self.kp = 0

    def trainI(self, state, state_, r, v, dt=0.01, lr=0.001):
        r = np.array(r)
        v = np.array(v)
        r = r + v.dot(self.r).dot(v.T)

        self.get_phi(state)
        phi = self.phi
        self.get_phi(state_)
        phi_ = self.phi
        delta_phi = phi_ - phi
        print(delta_phi, 'delta_phi')
        e_h = r * dt + self.w @ delta_phi + np.random.randn(1)*0.00

        m = delta_phi.T@delta_phi + 1
        print(m, 'm')
        self.w = self.w - lr * delta_phi/m**2*e_h
        print(self.w, 'w')