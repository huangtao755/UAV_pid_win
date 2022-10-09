import matplotlib.pyplot as plt
import numpy as np

from Algorithm.ClassicControl import PidControl
from Comman import MemoryStore
from Evn import QuadrotorFlyGui as Qgui
from Evn import QuadrotorFlyModel as Qfm

D2R = Qfm.D2R


class QuadControl(object):
    def __init__(self,
                 structure_type=Qfm.StructureType.quad_x,
                 init_att=np.array([0, 0, 0]),
                 init_pos=np.array([0, 0, 0])):
        """

        :param structure_type:
        :param init_att:
        :param init_pos:
        """
        self.uav_para = Qfm.QuadParas(structure_type=structure_type)
        self.sim_para = Qfm.QuadSimOpt(init_mode=Qfm.SimInitType.fixed, init_att=init_att,
                                       init_pos=init_pos)
        self.quad = Qfm.QuadModel(self.uav_para, self.sim_para)
        self.pid = PidControl(uav_para=self.uav_para,
                              kp_pos=np.array([0.5, 0.5, 0.45]),
                              ki_pos=np.array([0, 0., 0.0]),
                              kd_pos=np.array([0, 0, 0]),
                              kp_vel=np.array([1.4, 1.4, 1.6]),
                              ki_vel=np.array([0.01, 0.01, 0.01]),
                              kd_vel=np.array([0.15, 0.15, 0.]),

                              kp_att=np.array([2, 2, 2.]),
                              ki_att=np.array([0., 0, 0]),
                              kd_att=np.array([0, 0, 0]),
                              kp_att_v=np.array([14, 14, 10]),
                              ki_att_v=np.array([0.01, 0.01, 0.01]),
                              kd_att_v=np.array([0., 0., 0.01]))

        self.state_temp = self.quad.observe()
        self.record = MemoryStore.DataRecord()
        self.record.clear()
        self.step_num = 0

        self.ref = np.array([0, 0, 0, 0])
        self.ref_v = np.array([0, 0, 0, 0])
        self.fig1 = None
        self.fig2 = None

    def track(self, steps: int, ref, ref_v=np.array([0, 0, 0, 0])):
        """

        :param steps:
        :param ref:
        :param ref_v:
        :return:
        """
        self.state_temp = self.quad.observe()
        state_compensate = self.state_temp - np.array([0, 0, 0,
                                                       ref_v[0], ref_v[1], ref_v[2],
                                                       0, 0, 0,
                                                       0, 0, ref_v[3]])
        action = self.pid.pid_control(state_compensate, ref)
        self.quad.step(action)

        track_err = np.hstack([ref[0] - self.state_temp[0],
                               ref[1] - self.state_temp[1],
                               ref[2] - self.state_temp[2],
                               ref[3] - self.state_temp[8]])

        self.record.buffer_append((self.state_temp, action, track_err))

        self.step_num += 1
        print('steps_num', self.step_num)
        if self.step_num == steps:
            print('ending')
            self.record.episode_append()

    def reset(self, att, pos):
        """

        :param att:
        :param pos:
        :return:
        """
        self.quad.reset_states(att=att, pos=pos)
        self.step_num = 0
        self.record.clear()

    def fig_show(self, i):
        """

        :param i:
        :return:
        """
        data = self.record.get_episode_buffer()
        bs = data[0]
        ba = data[1]
        t = range(0, self.record.count)
        ts = np.array(t) * self.pid.ts

        self.fig1 = plt.figure(int(i * 3))
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
        plt.ylabel('Altitude $(\circ)$', fontsize=15)
        plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
        plt.subplot(4, 1, 4)
        plt.plot(ts, ba[t, 0], label='f')
        plt.plot(ts, ba[t, 1] / self.uav_para.uavInertia[0], label='t1')
        plt.plot(ts, ba[t, 2] / self.uav_para.uavInertia[1], label='t2')
        plt.plot(ts, ba[t, 3] / self.uav_para.uavInertia[2], label='t3')
        plt.ylabel('f (m/s^2)', fontsize=15)
        plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
        print(data)
        if len(data) == 3:
            err = data[2]
            self.fig1 = plt.figure(int(i * 3 + 1))
            plt.subplot(3, 1, 1)
            plt.plot(ts, bs[t, 3], label='x')
            plt.plot(ts, bs[t, 4], label='y')
            plt.plot(ts, bs[t, 5], label='z')
            plt.ylabel('v/m/s', fontsize=15)
            plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
            plt.subplot(3, 1, 2)
            plt.plot(ts, err[t, 0], label='x')
            plt.plot(ts, err[t, 1], label='y')
            plt.plot(ts, err[t, 2], label='z')
            plt.ylabel('error/m', fontsize=15)
            plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
            plt.subplot(3, 1, 3)
            plt.plot(ts, err[t, 3], label='err_phi')
            plt.ylabel('Altitude_err $(\circ)$', fontsize=15)
            plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
        # plt.show()


def main():
    quad1 = QuadControl(init_pos=np.array([-5, 0, 0]))
    quad2 = QuadControl(init_pos=np.array([-5, 0, 0]))
    gui = Qgui.QuadrotorFlyGui([quad1.quad, quad2.quad])
    steps = 1000
    ref = np.array([-15, -15, -15, 0])
    for i in range(steps):
        # ref = np.array([5 * np.cos(np.pi / 18 * quad1.quad.ts + np.pi),
        #                 5 * np.sin(np.pi / 18 * quad1.quad.ts + np.pi),
        #                 0.2 * quad1.quad.ts,
        #                 np.pi / 18 * quad1.quad.ts])

        # ref_v1 = np.array([-np.pi * np.sin(np.pi / 18 * (quad1.quad.ts) + np.pi) * 5 / 18,
        #                    np.pi * np.cos(np.pi / 18 * (quad1.quad.ts) + np.pi) * 5 / 18,
        #                    0.2,
        #                    np.pi / 18])  # target velocity
        ref_v1 = np.array([0, 0, 0, 0])
        quad2.state_temp = quad2.quad.observe()
        quad1.track(ref=ref, ref_v=ref_v1, steps=steps)
        state_compensate = quad2.state_temp - np.array([0, 0, 0,
                                                        ref_v1[0], ref_v1[1], ref_v1[2],
                                                        0, 0, 0,
                                                        0, 0, ref_v1[3]])
        action2 = quad2.quad.controller_pid(state=state_compensate, ref_state=ref)
        quad2.quad.step(action2)
        track_err = np.hstack([ref[0] - quad2.state_temp[0],
                               ref[1] - quad2.state_temp[1],
                               ref[2] - quad2.state_temp[2],
                               ref[3] - quad2.state_temp[8]])

        quad2.record.buffer_append((quad2.state_temp, action2, track_err))

        if i % 20 == 0:
            gui.quadGui.target = ref
            gui.quadGui.sim_time = quad1.quad.ts
            gui.render()
    quad2.record.episode_append()
    quad1.fig_show(1)
    quad2.fig_show(2)
    plt.show()


if __name__ == "__main__":
    main()
