import matplotlib.pyplot as plt
import numpy as np

from Algorithm.ClassicControl import PidControl
from Comman import MemoryStore
from Evn import QuadrotorFlyGui as Qgui
from Evn import QuadrotorFlyModel as Qfm

D2R = Qfm.D2R


def test():
    print("PID controller test")
    uav_para = Qfm.QuadParas(structure_type=Qfm.StructureType.quad_x)
    sim_para = Qfm.QuadSimOpt(init_mode=Qfm.SimInitType.fixed,
                              init_att=np.array([0, 0, 0]), init_pos=np.array([0, 0, 0]))
    quad = Qfm.QuadModel(uav_para, sim_para)
    record = MemoryStore.DataRecord()
    record.clear()

    # gui init
    gui = Qgui.QuadrotorFlyGui([quad])

    # init controller
    pid = PidControl(kp_pos=np.array([0.5, 0.5, 0.5]),
                     ki_pos=np.array([0, 0, 0.0]),
                     kd_pos=np.array([0, 0, 0]),
                     kp_vel=np.array([3, 3, 3]),
                     ki_vel=np.array([0, 0, 0.01]),
                     kd_vel=np.array([0.1, 0.1, 0.1]),

                     kp_att=np.array([0.01, 0.01, 0.01]),
                     ki_att=np.array([0, 0, 0]),
                     kd_att=np.array([0, 0, 0]),
                     kp_att_v=np.array([0.1, 0.1, 0.1]),
                     ki_att_v=np.array([0, 0, 0]),
                     kd_att_v=np.array([0.1, 0.1, 0.1]))

    # simulator init
    step_num = 0
    pos_x = []
    pos_y = []
    pos_z = []
    pos = []
    ref = np.array([0., 5., 0., 0])

    # simulate begin
    for i in range(1600):
        state_temp = quad.observe()
        action = pid.pid_control(state_temp, ref)
        quad.step(action)
        pos_x.append(state_temp[0])
        pos_y.append(state_temp[1])
        pos_z.append(state_temp[2])
        pos = [pos_x, pos_y, pos_z]

        if i % 100 == 0:
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
    # mpl.style.use('seaborn')
    fig1 = plt.figure(2)
    plt.clf()
    plt.subplot(3, 1, 1)
    plt.plot(t, bs[t, 6] / D2R, label='roll')
    plt.plot(t, bs[t, 7] / D2R, label='pitch')
    plt.plot(t, bs[t, 8] / D2R, label='yaw')
    plt.ylabel('Attitude $(\circ)$', fontsize=15)
    plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
    plt.subplot(3, 1, 2)
    plt.plot(t, bs[t, 0], label='x')
    plt.plot(t, bs[t, 1], label='y')
    plt.ylabel('Position (m)', fontsize=15)
    plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
    plt.subplot(3, 1, 3)
    plt.plot(t, bs[t, 2], label='z')
    plt.ylabel('Altitude (m)', fontsize=15)
    plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
    plt.show()


if __name__ == "__main__":
    test()
