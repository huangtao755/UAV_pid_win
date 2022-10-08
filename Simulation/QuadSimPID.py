import matplotlib.pyplot as plt
import numpy as np

from Algorithm.ClassicControl import PidControl
from Comman import MemoryStore
from Evn import QuadrotorFlyGui as Qgui
from Evn import QuadrotorFlyModel as Qfm

D2R = Qfm.D2R


def point_track():
    print("PID controller test")
    uav_para = Qfm.QuadParas(structure_type=Qfm.StructureType.quad_x)
    sim_para = Qfm.QuadSimOpt(init_mode=Qfm.SimInitType.fixed,
                              init_att=np.array([0, 0, 0]), init_pos=np.array([-10, -20, -10]))
    quad = Qfm.QuadModel(uav_para, sim_para)
    record = MemoryStore.DataRecord()
    record.clear()

    # gui init
    gui = Qgui.QuadrotorFlyGui([quad])

    # init controller
    pid = PidControl(uav_para=uav_para,
                     kp_pos=np.array([0.9, 0.9, 1]),
                     ki_pos=np.array([0, 0., 0.0]),
                     kd_pos=np.array([0, 0, 0]),
                     kp_vel=np.array([2.9, 2.9, 3.0]),
                     ki_vel=np.array([0.01, 0.01, 0.01]),
                     kd_vel=np.array([0.06, 0.06, 0.05]),

                     kp_att=np.array([5.5, 5.5, 3]),
                     ki_att=np.array([0., 0, 0]),
                     kd_att=np.array([0, 0, 0]),
                     kp_att_v=np.array([30, 30, 10]),
                     ki_att_v=np.array([0.01, 0.01, 0.]),
                     kd_att_v=np.array([0.01, 0.01, 0.1]))

    # simulator init
    step_num = 0
    pos_x = []
    pos_y = []
    pos_z = []
    pos = []
    ref = np.array([15, 15, 15, 0])
    print(quad.observe())
    # simulate begin
    for i in range(1500):
        state_temp = quad.observe()
        action = pid.pid_control(state_temp, ref)
        quad.step(action)
        pos_x.append(state_temp[0])
        pos_y.append(state_temp[1])
        pos_z.append(state_temp[2])
        pos = [pos_x, pos_y, pos_z]

        if i % 10 == 0:
            gui.quadGui.target = ref[0:3]
            gui.quadGui.sim_time = quad.ts
            gui.render()
        record.buffer_append((state_temp, action))
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


def traject_track():
    print("PID controller test")
    uav_para = Qfm.QuadParas(structure_type=Qfm.StructureType.quad_x)
    sim_para = Qfm.QuadSimOpt(init_mode=Qfm.SimInitType.fixed,
                              init_att=np.array([0, 0, 0]), init_pos=np.array([-3, 0, 0]))
    quad = Qfm.QuadModel(uav_para, sim_para)
    record = MemoryStore.DataRecord()
    record.clear()

    # gui init
    gui = Qgui.QuadrotorFlyGui([quad])

    # init controller
    pid = PidControl(uav_para=uav_para,
                     kp_pos=np.array([0.9, 0.9, 1]),
                     ki_pos=np.array([0, 0., 0.0]),
                     kd_pos=np.array([0, 0, 0]),
                     kp_vel=np.array([2.9, 2.9, 3.0]),
                     ki_vel=np.array([0.01, 0.01, 0.01]),
                     kd_vel=np.array([0.06, 0.06, 0.05]),

                     kp_att=np.array([5.5, 5.5, 3]),
                     ki_att=np.array([0., 0, 0]),
                     kd_att=np.array([0, 0, 0]),
                     kp_att_v=np.array([30, 30, 10]),
                     ki_att_v=np.array([0.01, 0.01, 0.]),
                     kd_att_v=np.array([0.01, 0.01, 0.1]))

    # simulator init
    step_num = 0
    pos_x = []
    pos_y = []
    pos_z = []
    pos = []
    # ref = np.array([15, -15, -15, 0])
    print(quad.observe())
    # simulate begin
    for i in range(30000):
        ref = np.array([3 * np.cos(np.pi / 18 * quad.ts + np.pi),
                        3 * np.sin(np.pi / 18 * quad.ts + np.pi), 0.05 * quad.ts, np.pi / 18 * quad.ts])
        state_temp = quad.observe()
        action = pid.pid_control(state_temp, ref)
        quad.step(action)
        pos_x.append(state_temp[0])
        pos_y.append(state_temp[1])
        pos_z.append(state_temp[2])
        pos = [pos_x, pos_y, pos_z]

        if i % 20 == 0:
            gui.quadGui.target = ref[0:3]
            gui.quadGui.sim_time = quad.ts
            gui.render()
        state_temp[8] = state_temp[8] % (2 * np.pi)
        print(state_temp[8], ''' angle''')
        record.buffer_append((state_temp, action))
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
    traject_track()
