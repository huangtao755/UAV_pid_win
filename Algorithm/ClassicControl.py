#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import Evn.QuadrotorFlyModel as Qfm


class PidControl(object):
    def __init__(self,
                 uav_para=Qfm.QuadParas(structure_type=Qfm.StructureType.quad_x),
                 kp_pos=np.array([0, 0, 0]),
                 ki_pos=np.array([0, 0, 0]),
                 kd_pos=np.array([0, 0, 0]),
                 kp_vel=np.array([0, 0, 0]),
                 ki_vel=np.array([0, 0, 0]),
                 kd_vel=np.array([0, 0, 0]),
                 kp_att=np.array([0, 0, 0]),
                 ki_att=np.array([0, 0, 0]),
                 kd_att=np.array([0, 0, 0]), ):
        self.ts = uav_para.ts
