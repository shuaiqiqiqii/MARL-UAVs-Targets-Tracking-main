from math import cos, sin, pi
import random

import numpy as np


#目标类
class TARGET:
    def __init__(self, x0: float, y0: float, h0: float, a0: float, v_max: float, h_max: float, dt):
        """
        :param x0: scalar
        :param y0: scalar
        :param h0: 航向角度
        :param v_max: 最大速度
        :param h_max: 最大航向角度
        :param dt: 时间步长
        """
        # the position, velocity and heading of this uav
        self.x = x0
        self.y = y0
        self.h = h0
        self.v_max = v_max

        # the max heading angular rate and the action of this uav
        self.h_max = h_max
        self.a = a0

        # time interval
        self.dt = dt

    def update_position(self, x_max, y_max) -> (float, float):
        """
        receive the action (heading angular rate), then update the current position
        :param y_max:
        :param x_max:
        :return:
        """
        self.a = random.uniform(-self.h_max, self.h_max)
        dx = self.dt * self.v_max * cos(self.h)  # x 方向位移
        dy = self.dt * self.v_max * sin(self.h)  # y 方向位移
        self.x += dx
        self.y += dy

        # if self.x > x_max:
        #     self.x = x_max
        # if self.x < 0:
        #     self.x = 0
        #
        # if self.y > y_max:
        #     self.y = y_max
        # if self.y < 0:
        #     self.y = 0

        # self.h += self.dt * self.a  # 更新朝向角度
        # self.h = (self.h + pi) % (2 * pi) - pi  # 确保朝向角度在 [-pi, pi) 范围内
        #添加了边界反弹

        #原先的版本，现进行修改
        # if 0 > self.y or self.y > y_max:
        #     self.h = -self.h
        # elif self.x < 0 or self.x > x_max:
        #     if self.h > 0:
        #         self.h = pi-self.h
        #     else:
        #         self.h = -pi-self.h
        #
        # return self.x, self.y
        # ===================== 【修复】边界反弹逻辑 =====================
        # 碰到上下边界 → 垂直反弹
        # --------------------- 安全边界限制（不反弹，永不越界） ---------------------
        # ===================== 真实物理反弹（自然流畅） =====================
        hit_boundary = False

        # 左右边界反弹（X 轴）
        if self.x <= 2 or self.x >= x_max - 2:
            self.h = pi - self.h  # 水平反弹
            hit_boundary = True

        # 上下边界反弹（Y 轴）
        if self.y <= 2 or self.y >= y_max - 2:
            self.h = -self.h  # 垂直反弹
            hit_boundary = True

        # ===================== 安全锁边（绝不越界，绝不卡 0 距离） =====================
        self.x = np.clip(self.x, 2, x_max - 2)
        self.y = np.clip(self.y, 2, y_max - 2)

        # 更新朝向
        self.h += self.dt * self.a
        self.h = (self.h + pi) % (2 * pi) - pi

        return self.x, self.y