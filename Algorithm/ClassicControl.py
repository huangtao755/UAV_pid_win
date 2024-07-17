#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch as t
import math
import Evn.QuadrotorFlyModel as Qfm

# import cubic_spline
import math
import matplotlib.pyplot as plt
# import scipy.linalg as la


class PidControl(object):
    def __init__(self,
                 uav_para=Qfm.QuadParas(structure_type=Qfm.StructureType.quad_x),
                 kp_pos=np.zeros(3),
                 ki_pos=np.zeros(3),
                 kd_pos=np.zeros(3),
                 kp_vel=np.zeros(3),
                 ki_vel=np.zeros(3),
                 kd_vel=np.zeros(3),
                 kp_att=np.zeros(3),
                 ki_att=np.zeros(3),
                 kd_att=np.zeros(3),
                 kp_att_v=np.zeros(3),
                 ki_att_v=np.zeros(3),
                 kd_att_v=np.zeros(3),
                 p_v=np.zeros(3),
                 p_a=np.zeros(3),
                 a_v=np.zeros(3),
                 a_a=np.zeros(3)):
        """

        :param uav_para:
        :param kp_pos:
        :param ki_pos:
        :param kd_pos:
        :param kp_vel:
        :param ki_vel:
        :param kd_vel:
        :param kp_att:
        :param ki_att:
        :param kd_att:
        :param kp_att_v:
        :param ki_att_v:
        :param kd_att_v:
        """
        " init model "
        self.uav_par = uav_para
        self.ts = uav_para.ts
        self.step_num = 0
        " init control para "
        self.kp_pos = kp_pos
        self.ki_pos = ki_pos
        self.kd_pos = kd_pos

        self.kp_vel = kp_vel
        self.ki_vel = ki_vel
        self.kd_vel = kd_vel

        self.kp_att = kp_att
        self.ki_att = ki_att
        self.kd_att = kd_att

        self.kp_att_v = kp_att_v
        self.ki_att_v = ki_att_v
        self.kd_att_v = kd_att_v
        " simulation state "
        self.err_p_pos = np.zeros(3)
        self.err_i_pos = np.zeros(3)
        self.err_d_pos = np.zeros(3)

        self.err_p_vel = np.zeros(3)
        self.err_i_vel = np.zeros(3)
        self.err_d_vel = np.zeros(3)

        self.err_p_att = np.zeros(3)
        self.err_i_att = np.zeros(3)
        self.err_d_att = np.zeros(3)

        self.err_p_att_v = np.zeros(3)
        self.err_i_att_v = np.zeros(3)
        self.err_d_att_v = np.zeros(3)

        self.p_v = p_v
        self.p_a = p_a
        self.a_v = a_v
        self.a_a = a_a

        self.err = np.zeros(12)

        self.ob_p = Observer(state_dim=3, dt=0.01)
        self.ob_q = Observer(state_dim=3, dt=0.01)
        self.ob_r = Observer(state_dim=3, dt=0.01)

        self.ref_att =np.zeros(3)

        self.ref_att_ob = np.zeros(9)

    def pid_control(self, state, ref_state, ref_a=None):
        """

        :param state:
        :param ref_state:
        :return:
        """
        print('________________________________step%d simulation____________________________' % self.step_num)
        action = np.zeros(4)

        " _______________position double loop_______________ "
        # ########position loop######## #
        pos = state[0:3]
        ref_pos = ref_state[0:3]
        err_p_pos_o = ref_pos - pos  # get new error of pos
        err_p_pos_ = np.array(8 * np.tanh(np.array(err_p_pos_o / 8)))
        # err_p_pos_ = err_p_pos_.clip(np.array([-8, -8, -8]), np.array([8, 8, 8]))

        if self.step_num == 0:
            self.err_d_pos = np.zeros(3)
        else:
            self.err_d_pos = (err_p_pos_ - self.err_p_pos) / self.ts  # get new error of pos-dot
        self.err_p_pos = err_p_pos_  # update pos error
        self.err_i_pos += self.err_p_pos * self.ts  # update pos integral

        ref_vel = self.kp_pos * self.err_p_pos \
                  + self.ki_pos * self.err_i_pos \
                  + self.kd_pos * self.err_d_pos  # get ref_v as input of velocity input

        # ########velocity loop######## #
        vel = state[3:6]
        err_p_vel_ = ref_vel - vel  # get new error of velocity
        if self.step_num == 0:
            self.err_d_vel = np.zeros(3)
        else:
            self.err_d_vel = (err_p_vel_ - self.err_p_vel) / self.ts  # get new error of vel-dot
        self.err_p_vel = err_p_vel_  # update vel error
        self.err_i_vel += self.err_p_vel * self.ts  # update vel integral

        a_pos = self.kp_vel * self.err_p_vel \
                + self.ki_vel * self.err_i_vel \
                + self.kd_vel * self.err_d_vel  # get the output u of 3D for position loop

        a_pos = a_pos.clip(np.array([-30, -30, -30]), np.array([30, 30, 30]))
        a_pos[2] += self.uav_par.g  # gravity compensation in z-axis
        a_pos[2] = max(0.0000000000001, a_pos[2])
        if ref_a is not None:
            a_pos[0] += ref_a[0]
            a_pos[1] += ref_a[1]

        " ________________attitude double loop_______________ "
        # ########attitude loop######## #
        phi = state[6]
        theta = state[7]
        phy = state[8]
        att = np.array([phi, theta, phy])

        # u1 = self.uav_par.uavM * a_pos[2] / (np.cos(phi) * np.cos(theta))
        # print('original_u1', u1)
        u1 = self.uav_par.uavM * np.sqrt(sum(np.square(a_pos)))

        # print('----------------------------------')
        # print('phi', phi)
        # print('theta', theta)
        # print('phy', phy)
        # print('__________________________________')

        ref_phy = ref_state[3]
        ref_phi = np.arcsin(self.uav_par.uavM * (a_pos[0] * np.sin(ref_phy) - a_pos[1] * np.cos(ref_phy)) / u1)
        # ref_phi = min(ref_phi, np.pi/3)
        # ref_phi = max(ref_phi, -np.pi/3)
        k = max(self.uav_par.uavM * (a_pos[0] * np.cos(ref_phy) + a_pos[1] * np.sin(ref_phy)) / (u1 * np.cos(ref_phi)), -0.8)
        k = min(k, 0.8)
        ref_theta = np.arcsin(self.uav_par.uavM * (a_pos[0] * np.cos(ref_phy) + a_pos[1] * np.sin(ref_phy)) / (u1 * np.cos(ref_phi)))
        ref_att = np.array([ref_phi, ref_theta, ref_phy])

        phi_r = self.ob_p.observer(ref_phi)
        theta_r = self.ob_p.observer(ref_theta)
        phy_r = self.ob_p.observer(ref_phy)

        self.ref_att = np.hstack((ref_phi, ref_theta, ref_phy))
        self.ref_att_ob = np.hstack((phi_r, theta_r, phy_r))
        # print('----------------------------------')
        # print('ref_phi', ref_phi)
        # print('ref_theta', ref_theta)
        # print('ref_phy', ref_phy)
        # print('__________________________________')

        err_p_att_ = ref_att - att

        if self.step_num == 0:
            self.err_d_att = np.zeros(3)
        else:
            self.err_d_att = (err_p_att_ - self.err_p_att) / self.ts
        self.err_p_att = err_p_att_
        self.err_i_att += self.err_p_att * self.ts

        ref_att_v = self.kp_att * self.err_p_att \
                    + self.ki_att * self.err_i_att \
                    + self.kd_att * self.err_d_att

        # print('----------------------------------')
        # print('err_p_att', self.err_p_att)
        # print('err_i_att', self.err_i_att)
        # print('err_d_att', self.err_d_att)
        # print('ref_att_v', ref_att_v)
        # print('__________________________________')
        # ########velocity of attitude loop######## #
        att_v = state[9:12]
        err_p_att_v_ = ref_att_v - att_v

        if self.step_num == 0:
            self.err_d_att_v = 0
        else:
            self.err_d_att_v = (err_p_att_v_ - self.err_p_att_v) / self.ts
        self.err_p_att_v = err_p_att_v_
        self.err_i_att_v += self.err_p_att_v * self.ts

        a_att = self.kp_att_v * self.err_p_att_v \
                + self.ki_att_v * self.err_i_att_v \
                + self.kd_att_v * self.err_d_att_v

        if ref_a is not None:
            a_att[0] -= ref_a[3]
            a_att[1] -= ref_a[4]
            a_att[2] -= ref_a[5]
        # a_att = a_att.clip([-25, -25, -25], [25, 25, 25])
        a_att = np.array(20 * np.tanh(np.array(a_att / 20)))

        u = a_att * self.uav_par.uavInertia
        u1 = max(u1, np.sqrt(np.linalg.norm(u)))
        action = np.array([u1, u[0], u[1], u[2]])
        self.err = np.array(np.hstack((err_p_pos_o, self.err_p_vel, self.err_p_att, self.err_p_att_v)))
        self.step_num += 1

        return action, self.ref_att_ob

    def fix_pid_control(self, state, ref_state, ref_a=None):
        """

        :param state:
        :param ref_state:
        :return:
        """
        print('________________________________step%d simulation____________________________' % self.step_num)
        action = np.zeros(4)

        " _______________position double loop_______________ "
        # ########position loop######## #
        pos = state[0:3]
        ref_pos = ref_state[0:3]
        err_p_pos_o = ref_pos - pos  # get new error of pos
        err_p_pos_ = np.array(8 * np.tanh(np.array(err_p_pos_o / 8)))
        # err_p_pos_ = err_p_pos_.clip(np.array([-8, -8, -8]), np.array([8, 8, 8]))

        if self.step_num == 0:
            self.err_d_pos = np.zeros(3)
        else:
            self.err_d_pos = (err_p_pos_ - self.err_p_pos) / self.ts  # get new error of pos-dot
        self.err_p_pos = err_p_pos_  # update pos error
        self.err_i_pos += self.err_p_pos * self.ts  # update pos integral

        ref_vel = np.zeros(3)
        for i in range(3):
            ref_vel[i] = self.kp_pos[i] * abs(self.err_p_pos[i])**self.p_v[i]*np.tanh(self.err_p_pos[i]*100) \
                  + self.ki_pos[i] * self.err_i_pos[i] \
                  + self.kd_pos[i] * abs(self.err_d_pos[i])**(2*self.p_v[i]/(1+self.p_v[i]))*np.tanh(self.err_d_pos[i]*100)  # get ref_v as input of velocity input

        # ########velocity loop######## #
        vel = state[3:6]
        err_p_vel_ = ref_vel - vel  # get new error of velocity
        if self.step_num == 0:
            self.err_d_vel = np.zeros(3)
        else:
            self.err_d_vel = (err_p_vel_ - self.err_p_vel) / self.ts  # get new error of vel-dot
        self.err_p_vel = err_p_vel_  # update vel error
        self.err_i_vel += self.err_p_vel * self.ts  # update vel integral

        a_pos = np.zeros(3)
        for i in range(3):
            a_pos[i] = self.kp_vel[i] * abs(self.err_p_vel[i])**(self.p_a[i])*np.tanh(300*self.err_p_vel[i]) \
                + self.ki_vel[i] * self.err_i_vel[i] \
                + self.kd_vel[i] * abs(self.err_d_vel[i])**(2*self.p_a[i]/(1+self.p_a[i]))*np.tanh(300*self.err_p_vel[i])  # get the output u of 3D for position loop

        a_pos = a_pos.clip(np.array([-30, -30, -30]), np.array([30, 30, 30]))
        a_pos[2] += self.uav_par.g  # gravity compensation in z-axis
        a_pos[2] = max(0.0000000000001, a_pos[2])
        if ref_a is not None:
            a_pos[0] += ref_a[0]
            a_pos[1] += ref_a[1]

        " ________________attitude double loop_______________ "
        # ########attitude loop######## #
        phi = state[6]
        theta = state[7]
        phy = state[8]
        att = np.array([phi, theta, phy])

        # u1 = self.uav_par.uavM * a_pos[2] / (np.cos(phi) * np.cos(theta))
        # print('original_u1', u1)
        u1 = self.uav_par.uavM * np.sqrt(sum(np.square(a_pos)))

        # print('----------------------------------')
        # print('phi', phi)
        # print('theta', theta)
        # print('phy', phy)
        # print('__________________________________')

        ref_phy = ref_state[3]
        ref_phi = np.arcsin(self.uav_par.uavM * (a_pos[0] * np.sin(ref_phy) - a_pos[1] * np.cos(ref_phy)) / u1)
        # ref_phi = min(ref_phi, np.pi/3)
        # ref_phi = max(ref_phi, -np.pi/3)
        k = max(self.uav_par.uavM * (a_pos[0] * np.cos(ref_phy) + a_pos[1] * np.sin(ref_phy)) / (u1 * np.cos(ref_phi)), -0.8)
        k = min(k, 0.8)
        ref_theta = np.arcsin(self.uav_par.uavM * (a_pos[0] * np.cos(ref_phy) + a_pos[1] * np.sin(ref_phy)) / (u1 * np.cos(ref_phi)))
        ref_att = np.array([ref_phi, ref_theta, ref_phy])

        phi_r = self.ob_p.observer(ref_phi)
        theta_r = self.ob_p.observer(ref_theta)
        phy_r = self.ob_p.observer(ref_phy)

        self.ref_att = np.hstack((ref_phi, ref_theta, ref_phy))
        self.ref_att_ob = np.hstack((phi_r, theta_r, phy_r))
        # print('----------------------------------')
        # print('ref_phi', ref_phi)
        # print('ref_theta', ref_theta)
        # print('ref_phy', ref_phy)
        # print('__________________________________')

        err_p_att_ = ref_att - att

        if self.step_num == 0:
            self.err_d_att = np.zeros(3)
        else:
            self.err_d_att = (err_p_att_ - self.err_p_att) / self.ts
        self.err_p_att = err_p_att_
        self.err_i_att += self.err_p_att * self.ts

        ref_att_v = np.zeros(3)
        for i in range(3):
            ref_att_v[i] = self.kp_att[i] * abs(self.err_p_att[i])**self.a_v[i]*np.tanh(self.err_p_att[i]*10) \
                    + self.ki_att[i] * self.err_i_att[i] \
                    + self.kd_att[i] * abs(self.err_d_att[i])**(2*self.a_v[i]/(1+self.a_v[i]))*np.tanh(self.err_d_att[i]*10)

        # print('----------------------------------')
        # print('err_p_att', self.err_p_att)
        # print('err_i_att', self.err_i_att)
        # print('err_d_att', self.err_d_att)
        # print('ref_att_v', ref_att_v)
        # print('__________________________________')
        # ########velocity of attitude loop######## #
        att_v = state[9:12]
        err_p_att_v_ = ref_att_v - att_v

        if self.step_num == 0:
            self.err_d_att_v = np.zeros(3)
        else:
            self.err_d_att_v = (err_p_att_v_ - self.err_p_att_v) / self.ts
        self.err_p_att_v = err_p_att_v_
        self.err_i_att_v += self.err_p_att_v * self.ts

        a_att = np.zeros(3)
        print(self.kd_att_v, self.err_d_att_v, self.a_a, 'test')
        for i in range(3):
            a_att[i] = self.kp_att_v[i] * abs(self.err_p_att_v[i])**self.a_a[i]*np.tanh(self.err_p_att_v[i]*1000) \
                + self.ki_att_v[i] * self.err_i_att_v[i] \
                + self.kd_att_v[i] * abs(self.err_d_att_v[i])**(2*self.a_a[i]/(1+self.a_a[i]))*np.tanh(self.err_d_att_v[i]*1000)

        if ref_a is not None:
            a_att[0] -= ref_a[3]
            a_att[1] -= ref_a[4]
            a_att[2] -= ref_a[5]
        # a_att = a_att.clip([-25, -25, -25], [25, 25, 25])
        a_att = np.array(20 * np.tanh(np.array(a_att / 20)))

        u = a_att * self.uav_par.uavInertia
        u1 = max(u1, np.sqrt(np.linalg.norm(u)))
        action = np.array([u1, u[0], u[1], u[2]])
        self.err = np.array(np.hstack((err_p_pos_o, self.err_p_vel, self.err_p_att, self.err_p_att_v)))
        self.step_num += 1

        return action, self.ref_att_ob


    def reset(self):
        self.step_num = 0
        " simulation state "
        self.err_p_pos = np.zeros(3)
        self.err_i_pos = np.zeros(3)
        self.err_d_pos = np.zeros(3)

        self.err_p_vel = np.zeros(3)
        self.err_i_vel = np.zeros(3)
        self.err_d_vel = np.zeros(3)

        self.err_p_att = np.zeros(3)
        self.err_i_att = np.zeros(3)
        self.err_d_att = np.zeros(3)

        self.err_p_att_v = np.zeros(3)
        self.err_i_att_v = np.zeros(3)
        self.err_d_att_v = np.zeros(3)


class SinglePidControl(object):
    def __init__(self,
                 uav_para=Qfm.QuadParas(structure_type=Qfm.StructureType.quad_x),
                 kp_pos=np.zeros(3),
                 ki_pos=np.zeros(3),
                 kd_pos=np.zeros(3),
                 kp_att=np.zeros(3),
                 ki_att=np.zeros(3),
                 kd_att=np.zeros(3)
                 ):
        """

        :param uav_para:
        :param kp_pos:
        :param ki_pos:
        :param kd_pos:
        :param kp_vel:
        :param ki_vel:
        :param kd_vel:
        :param kp_att:
        :param ki_att:
        :param kd_att:
        :param kp_att_v:
        :param ki_att_v:
        :param kd_att_v:
        """
        " init model "
        self.uav_par = uav_para
        self.ts = uav_para.ts
        self.step_num = 0
        " init control para "
        self.kp_pos = kp_pos
        self.ki_pos = ki_pos
        self.kd_pos = kd_pos

        self.kp_att = kp_att
        self.ki_att = ki_att
        self.kd_att = kd_att

        " simulation state "
        self.err_p_pos = np.zeros(3)
        self.err_i_pos = np.zeros(3)
        self.err_d_pos = np.zeros(3)

        self.err_p_att = np.zeros(3)
        self.err_i_att = np.zeros(3)
        self.err_d_att = np.zeros(3)

        self.err = np.zeros(12)

        self.ref_phy = 0
        self.ref_phi = 0
        self.ref_theta = 0

    def pid_control(self, state, ref_state, ref_a=None):
        """

        :param state:
        :param ref_state:
        :return:
        """
        print('________________________________step%d simulation____________________________' % self.step_num)
        action = np.zeros(4)

        " _______________position double loop_______________ "
        # ########position loop######## #
        pos = state[0:3]
        pos_v = state[3:6]
        ref_pos = ref_state[0:3]
        ref_v = ref_state[3:6]

        err_p_pos_o = ref_pos - pos  # get new error of pos
        err_p_pos_ = np.array(8 * np.tanh(np.array(err_p_pos_o / 8)))
        # err_p_pos_ = err_p_pos_.clip(np.array([-8, -8, -8]), np.array([8, 8, 8]))

        if self.step_num == 0:
            self.err_d_pos = np.zeros(3)
        else:
            self.err_d_pos = ref_v - pos_v  # get new error of pos-dot
        self.err_p_pos = err_p_pos_  # update pos error
        self.err_i_pos = self.err_i_pos*0.95 + self.err_p_pos * self.ts  # update pos integral

        a_pos = self.kp_pos * self.err_p_pos \
                + self.ki_pos * self.err_i_pos \
                + self.kd_pos * self.err_d_pos  # get the output u of 3D for position loop

        # a_pos = a_pos.clip(np.array([-30, -30, -30]), np.array([30, 30, 30]))
        a_pos[2] += self.uav_par.g  # gravity compensation in z-axis
        a_pos[2] = max(0.001, a_pos[2])
        if ref_a is not None:
            a_pos[0] += ref_a[0]
            a_pos[1] += ref_a[1]
        " ________________attitude double loop_______________ "
        # ########attitude loop######## #
        phi = state[6]
        theta = state[7]
        phy = state[8]
        att = np.array([phi, theta, phy])

        # u1 = self.uav_par.uavM * a_pos[2] / (np.cos(phi) * np.cos(theta))
        # print('original_u1', u1)
        u1 = self.uav_par.uavM * np.sqrt(sum(np.square(a_pos)))

        # print('----------------------------------')
        # print('phi', phi)
        # print('theta', theta)
        # print('phy', phy)
        # print('__________________________________')

        ref_phy = ref_state[3]
        self.ref_phy = ref_phy
        ref_phi = np.arcsin(self.uav_par.uavM * (a_pos[0] * np.sin(ref_phy) - a_pos[1] * np.cos(ref_phy)) / u1)
        self.ref_phi = ref_phi
        ref_theta = np.arcsin(
            self.uav_par.uavM * (a_pos[0] * np.cos(ref_phy) + a_pos[1] * np.sin(ref_phy)) / (u1 * np.cos(ref_phi)))
        self.ref_theta = ref_theta

        ref_att = np.array([ref_phi, ref_theta, ref_phy])

        print('----------------------------------')
        print('ref_phi', ref_phi)
        print('ref_theta', ref_theta)
        print('ref_phy', ref_phy)
        print('__________________________________')

        err_p_att_ = ref_att - att

        if self.step_num == 0:
            self.err_d_att = np.zeros(3)
        else:
            self.err_d_att = (err_p_att_ - self.err_p_att) / self.ts
        self.err_p_att = err_p_att_
        self.err_i_att += self.err_p_att * self.ts

        a_att = self.kp_att * self.err_p_att \
                    + self.ki_att * self.err_i_att \
                    + self.kd_att * self.err_d_att

        u = a_att * self.uav_par.uavInertia
        u1 = max(u1, np.sqrt(np.linalg.norm(u)))
        action = np.array([u1, u[0], u[1], u[2]])
        self.err = np.array(np.hstack((err_p_pos_o, self.err_d_pos, self.err_p_att, self.err_d_att)))
        self.step_num += 1

        return action

    def reset(self):
        self.step_num = 0
        " simulation state "
        self.err_p_pos = np.zeros(3)
        self.err_i_pos = np.zeros(3)
        self.err_d_pos = np.zeros(3)

        self.err_p_att = np.zeros(3)
        self.err_i_att = np.zeros(3)
        self.err_d_att = np.zeros(3)


class Observer(object):
    def __init__(self, state_dim, dt):
        self.z = np.zeros(state_dim+2)
        self.d_z = np.zeros(state_dim+2)
        self.v = np.zeros(state_dim+1)

        self.para = np.array([4, 2, 2, 2, 2])
        self.dt = dt

    def observer_dynamic(self, y):
        self.d_z[0] = self.v[0]
        self.v[0] = - self.para[0] * np.abs(self.z[0] - y)**(2/3) * np.sign((self.z[0] - y)) + self.z[1]
        self.d_z[1] = self.v[1]
        self.v[1] = - self.para[1] * np.abs(self.z[1] - self.v[0])**(1/2) * np.sign((self.z[1] - self.v[0])) + self.z[2]
        self.d_z[2] = self.v[2]
        self.v[2] = - self.para[2] * np.abs(self.z[2] - self.v[1]) ** (1 / 2) * np.sign(10*(self.z[2] - self.v[1])) + self.z[3]
        self.d_z[3] = self.v[3]
        self.v[3] = - self.para[3] * np.abs(self.z[3] - self.v[2]) ** (1 / 2) * np.sign(10*(self.z[3] - self.v[2])) + self.z[4]
        self.d_z[4] = - self.para[4] * np.tanh(10*(self.z[4] - self.v[3]))

        return self.d_z

    def observer(self, y):
        for i in range(2):
            self.d_z = self.observer_dynamic(y)
            self.z = self.z + self.d_z * self.dt/2
        return self.z
