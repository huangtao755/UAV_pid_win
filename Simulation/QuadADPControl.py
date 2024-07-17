#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
introduction:
"""
import matplotlib.pyplot as plt
import numpy as np
import time

from Algorithm.ClassicControl import SinglePidControl
from Algorithm.HDP_policy import PIDSMADP
from Algorithm.HDP_policy import PSMADP
from Comman import MemoryStore
from Evn import QuadrotorFlyGui as Qgui
from Evn import QuadrotorFlyModel as Qfm


D2R = Qfm.D2R


def get_reward(state, v):
    Q = np.diag([100, 1, 1])
    R = np.diag([0.01, 0.01])
    return state.dot(Q).dot(state.T) + v.dot(R).dot(v.T)

def get_reward_att(state, v):
    Q = np.diag([100, 1])
    R = np.diag([0.01])
    return state.dot(Q).dot(state.T) + v.dot(R).dot(v.T)


def get_sm(x, dx):
    return 20*np.tanh((4*x + 0.1*dx)/20)

def get_sm_att(x, dx):
    return 100*x + 1000*dx


def point_track():
    """

    :return:
    """
    print("ADPPID controller test")
    uav_para = Qfm.QuadParas(structure_type=Qfm.StructureType.quad_x)
    sim_para = Qfm.QuadSimOpt(init_mode=Qfm.SimInitType.fixed, init_att=np.array([0, 0, 0]),
                              init_pos=np.array([0, 0, 0]))
    quad = Qfm.QuadModel(uav_para, sim_para)
    record = MemoryStore.DataRecord()
    record.clear()

    # gui init
    gui = Qgui.QuadrotorFlyGui([quad])

    # init controller
    pid = SinglePidControl(uav_para=uav_para,
                           kp_pos=np.array([0.4, 0.4, 0.2]),
                           ki_pos=np.array([0.0, 0.0, 0.0]),
                           kd_pos=np.array([10.1, 10.1, 1.6]),

                           kp_att=np.array([0.15, 0.15, 0.15]),
                           ki_att=np.array([0., 0., 0]),
                           kd_att=np.array([1., 1., 1.8]))

    # init ADP
    adp_x = PIDSMADP()
    adp_y = PIDSMADP()
    adp_z = PIDSMADP()

    adp_p = PSMADP()
    adp_q = PSMADP()
    adp_r = PSMADP()

    # simulator init
    step_num = 0
    sim_time = 100
    ref = np.array([5, 5, -5, 0])

    state = np.zeros(12)
    state_ = np.zeros(12)
    print(quad.observe(), 'observe')

    state = quad.observe()
    sx = np.hstack((-get_sm(ref[0]-state[0], -state[3]), adp_x.kp, adp_x.kd))
    sy = np.hstack((-get_sm(ref[1]-state[1], -state[4]), adp_y.kp, adp_y.kd))
    sz = np.hstack((-get_sm(ref[2]-state[2], -state[5]), adp_z.kp, adp_z.kd))

    sp = np.hstack((get_sm_att(pid.ref_phy-state[6], state[9]), adp_p.kp))
    sq = np.hstack((get_sm_att(pid.ref_phi-state[7], state[10]), adp_q.kp))
    sr = np.hstack((get_sm_att(pid.ref_theta-state[8], state[11]), adp_r.kp))

    # simulate begin
    state = quad.observe()
    for i in range(int(sim_time/quad.uavPara.ts)):
        # print(time.sleep(0.1))

        action = pid.pid_control(state, ref)
        quad.step(action)

        state_ = quad.observe()
        sx_ = np.hstack((-get_sm(ref[0]-state_[0], -state_[3]), adp_x.kp, adp_x.kd))
        sy_ = np.hstack((-get_sm(ref[1]-state_[1], -state_[4]), adp_y.kp, adp_y.kd))
        sz_ = np.hstack((-get_sm(ref[2]-state_[2], -state_[5]), adp_x.kp, adp_x.kd))

        sp_ = np.hstack((-get_sm_att(pid.ref_phy - state[6], -state[9]), adp_p.kp))
        sq_ = np.hstack((-get_sm_att(pid.ref_phi - state[7], -state[10]), adp_q.kp))
        sr_ = np.hstack((-get_sm_att(pid.ref_theta - state[8], -state[11]), adp_r.kp))

        rx = get_reward(sx_, adp_x.u)
        ry = get_reward(sy_, adp_y.u)
        rz = get_reward(sz_, adp_z.u)

        rp = get_reward_att(sp_, adp_p.u)
        rq = get_reward_att(sq_, adp_q.u)
        rr = get_reward_att(sr_, adp_r.u)

        # print(rx, 'reward')
        # print(sx, sx_)
        adp_x.trainI(sx, sx_, rx, adp_x.u)
        adp_y.trainI(sy, sy_, ry, adp_y.u)
        adp_z.trainI(sz, sz_, rz, adp_z.u)

        adp_p.trainI(sp, sp_, rp, adp_p.u)
        adp_q.trainI(sq, sq_, rq, adp_q.u)
        adp_r.trainI(sr, sr_, rr, adp_r.u)

        u_kp_x, u_kd_x = adp_x.u_star(sx_)
        print(u_kp_x, 'get_ukp')
        pid.kp_pos[0] += u_kp_x * quad.uavPara.ts
        print(pid.kp_pos[0], 'kp_x')
        pid.kd_pos[0] -= u_kd_x * quad.uavPara.ts

        u_kp_q = adp_q.u_star(sp_)

        pid.kp_att[2] += u_kp_q * quad.uavPara.ts

        state = state_
        sx = np.hstack((-get_sm(ref[0]-state[0], -state[3]), adp_x.kp, adp_x.kd))
        sy = np.hstack((-get_sm(ref[1]-state[1], -state[4]), adp_y.kp, adp_y.kd))
        sz = np.hstack((-get_sm(ref[2]-state[2], -state[5]), adp_x.kp, adp_x.kd))

        sp = np.hstack((-get_sm_att(pid.ref_phy - state[6], -state[9]), adp_p.kp))
        print(-get_sm_att(pid.ref_phi - state[7], -state[10]), adp_q.kp)
        sq = np.hstack((-get_sm_att(pid.ref_phi - state[7], -state[10]), adp_q.kp))
        sr = np.hstack((-get_sm_att(pid.ref_theta - state[8], -state[11]), adp_r.kp))
        if i % 100 == 0:
            gui.quadGui.target = ref[0:3]
            gui.quadGui.sim_time = quad.ts
            gui.render()
        record.buffer_append((state, action))
        step_num += 1

    record.episode_append()
    data = record.get_episode_buffer()
    bs = data[0]
    ba = data[1]
    t = range(0, record.count)
    ts = np.array(t) * pid.ts
    # mpl.style.use('seaborn')

    fig1 = plt.figure(2)
    plt.clf()
    plt.subplot(4, 1, 1)
    plt.plot(ts, bs[t, 6] / D2R, label='roll')
    plt.plot(ts, bs[t, 7] / D2R, label='pitch')
    plt.plot(ts, bs[t, 8] / D2R, label='yaw')
    plt.ylabel('Attitude $(\circ)$', fontsize=15)
    plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
    plt.subplot(4, 1, 2)
    plt.plot(ts, bs[t, 0], label='x')
    plt.plot(ts, bs[t, 1], label='y')
    plt.ylabel('Position (m)', fontsize=15)
    plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
    plt.subplot(4, 1, 3)
    plt.plot(ts, bs[t, 2], label='z')
    plt.ylabel('Altitude (m)', fontsize=15)
    plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
    plt.subplot(4, 1, 4)
    plt.plot(ts, ba[t, 0], label='f')
    plt.plot(ts, ba[t, 1] / uav_para.uavInertia[0], label='t1')
    plt.plot(ts, ba[t, 2] / uav_para.uavInertia[1], label='t2')
    plt.plot(ts, ba[t, 3] / uav_para.uavInertia[2], label='t3')
    plt.ylabel('f (m/s^2)', fontsize=15)
    plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
    plt.show()


def point_adp_track():
    """

    :return:
    """
    print("ADPPID controller test")
    uav_para = Qfm.QuadParas(structure_type=Qfm.StructureType.quad_x)
    sim_para = Qfm.QuadSimOpt(init_mode=Qfm.SimInitType.fixed, init_att=np.array([0, 0, 0]),
                              init_pos=np.array([0, 0, 0]))
    quad = Qfm.QuadModel(uav_para, sim_para)
    record = MemoryStore.DataRecord()
    record.clear()

    # gui init
    gui = Qgui.QuadrotorFlyGui([quad])

    # init controller
    pid = SinglePidControl(uav_para=uav_para,
                           kp_pos=np.array([0.1, 0.1, 0.2]),
                           ki_pos=np.array([0.0, 0.0, 0.0]),
                           kd_pos=np.array([10.1, 10.1, 1.6]),

                           kp_att=np.array([0.45, 0.45, 0.35]),
                           ki_att=np.array([0., 0., 0]),
                           kd_att=np.array([2.8, 2.8, 3.2]))

    # init ADP
    adp_x = PIDSMADP()
    adp_y = PIDSMADP()
    adp_z = PIDSMADP()

    # simulator init
    step_num = 0
    sim_time = 150
    ref = np.array([10, 10, -10, 0])

    state = np.zeros(12)
    state_ = np.zeros(12)
    print(quad.observe(), 'observe')

    state = quad.observe()
    sx = np.hstack((-get_sm(ref[0]-state[0], -state[3]), adp_x.kp, adp_x.kd))
    sy = np.hstack((-get_sm(ref[1]-state[1], -state[4]), adp_y.kp, adp_y.kd))
    sz = np.hstack((-get_sm(ref[2]-state[2], -state[5]), adp_x.kp, adp_x.kd))

    # simulate begin
    state = quad.observe()
    for i in range(int(sim_time/quad.uavPara.ts)):
        # print(time.sleep(0.1))

        action = pid.pid_control(state, ref)
        quad.step(action)

        state_ = quad.observe()
        sx_ = np.hstack((-get_sm(ref[0]-state_[0], -state_[3]), adp_x.kp, adp_x.kd))
        sy_ = np.hstack((-get_sm(ref[1]-state_[1], -state_[4]), adp_y.kp, adp_y.kd))
        sz_ = np.hstack((-get_sm(ref[2]-state_[2], -state_[5]), adp_x.kp, adp_x.kd))

        rx = get_reward(sx_, adp_x.u)
        ry = get_reward(sy_, adp_y.u)
        rz = get_reward(sz_, adp_z.u)

        print(rx, 'reward')
        # print(sx, sx_)
        adp_x.trainI(sx, sx_, rx, adp_x.u)
        adp_y.trainI(sy, sy_, ry, adp_y.u)
        adp_z.trainI(sz, sz_, rz, adp_z.u)

        u_kp_x, u_kd_x = adp_x.u_star(sx_)
        print(u_kp_x, 'get_ukp')
        pid.kp_pos[0] += u_kp_x * quad.uavPara.ts
        print(pid.kp_pos[0], 'kp_x')
        pid.kd_pos[0] += u_kd_x * quad.uavPara.ts

        state = state_
        sx = np.hstack((-get_sm(ref[0]-state[0], -state[3]), adp_x.kp, adp_x.kd))
        sy = np.hstack((-get_sm(ref[1]-state[1], -state[4]), adp_y.kp, adp_y.kd))
        sz = np.hstack((-get_sm(ref[2]-state[2], -state[5]), adp_x.kp, adp_x.kd))
        if i % 10 == 0:
            gui.quadGui.target = ref[0:3]
            gui.quadGui.sim_time = quad.ts
            gui.render()
        record.buffer_append((state, action))
        step_num += 1

    record.episode_append()
    data = record.get_episode_buffer()
    bs = data[0]
    ba = data[1]
    t = range(0, record.count)
    ts = np.array(t) * pid.ts
    # mpl.style.use('seaborn')
    fig1 = plt.figure(2)
    plt.clf()
    plt.subplot(4, 1, 1)
    plt.plot(ts, bs[t, 6] / D2R, label='roll')
    plt.plot(ts, bs[t, 7] / D2R, label='pitch')
    plt.plot(ts, bs[t, 8] / D2R, label='yaw')
    plt.ylabel('Attitude $(\circ)$', fontsize=15)
    plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
    plt.subplot(4, 1, 2)
    plt.plot(ts, bs[t, 0], label='x')
    plt.plot(ts, bs[t, 1], label='y')
    plt.ylabel('Position (m)', fontsize=15)
    plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
    plt.subplot(4, 1, 3)
    plt.plot(ts, bs[t, 2], label='z')
    plt.ylabel('Altitude (m)', fontsize=15)
    plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
    plt.subplot(4, 1, 4)
    plt.plot(ts, ba[t, 0], label='f')
    plt.plot(ts, ba[t, 1] / uav_para.uavInertia[0], label='t1')
    plt.plot(ts, ba[t, 2] / uav_para.uavInertia[1], label='t2')
    plt.plot(ts, ba[t, 3] / uav_para.uavInertia[2], label='t3')
    plt.ylabel('f (m/s^2)', fontsize=15)
    plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
    plt.show()


if __name__ == "__main__":
    point_track()
