import random
from math import cos, sin, pi, sqrt


class OBSTACLE:
    def __init__(self, x0: float, y0: float, h0 : float, a0:float,v_max: float, h_max: float,dt):
        """
        动态障碍物类
        :param x0: float, 初始 x 坐标
        :param y0: float, 初始 y 坐标
        :param v_max: float, 最大线速度
        :param h_max: float, 最大角速度
        :param dt: float, 时间间隔
        :param is_static: bool, 是否为静态障碍物
        """
        self.x = x0
        self.y = y0
        self.v_max = v_max
        self.h_max = h_max

        self.dt = dt
        self.a = a0
        self.h = h0

    def update_position(self, x_max: float, y_max: float) -> (float, float):
        """
        更新障碍物位置
        :param x_max: float, x 轴边界
        :param y_max: float, y 轴边界
        :return: (x, y) 新位置
        """
        self.a = random.uniform(-self.h_max, self.h_max) * 0.3

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
        #更新朝向
        self.h += self.dt * self.a  # 更新朝向角度
        self.h = (self.h + pi) % (2 * pi) - pi  # 确保朝向角度在 [-pi, pi) 范围内

        # if 0 > self.y or self.y > y_max:
        #     self.h = -self.h
        # elif self.x < 0 or self.x > x_max:
        #     if self.h > 0:
        #         self.h = pi - self.h
        #     else:
        #         self.h = -pi - self.h
        # ✅ 4. 标准弹性反弹（最自然）
        if self.x < 0 or self.x > x_max:
            self.h = pi - self.h
        if self.y < 0 or self.y > y_max:
            self.h = -self.h

        # ✅ 5. 防止卡墙
        self.x = max(0, min(self.x, x_max))
        self.y = max(0, min(self.y, y_max))

        return self.x, self.y

