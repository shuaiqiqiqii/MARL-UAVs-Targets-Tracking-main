import random
from math import cos, sin, pi, sqrt


class OBSTACLE:
    def __init__(self, x0: float, y0: float, v_max: float, h_max: float, dt,
                 move_range: float = 0, is_static: bool = False):
        """
        动态障碍物类
        :param x0: float, 初始 x 坐标
        :param y0: float, 初始 y 坐标
        :param v_max: float, 最大线速度
        :param h_max: float, 最大角速度
        :param dt: float, 时间间隔
        :param move_range: float, 移动范围半径（0 表示不限制）
        :param is_static: bool, 是否为静态障碍物
        """
        self.x = x0
        self.y = y0
        self.init_x = x0
        self.init_y = y0
        self.h = random.uniform(-pi, pi)
        self.v_max = v_max
        self.h_max = h_max
        self.dt = dt
        self.move_range = move_range
        self.is_static = is_static
        self.a = 0

    def update_position(self, x_max: float, y_max: float) -> (float, float):
        """
        更新障碍物位置
        :param x_max: float, x 轴边界
        :param y_max: float, y 轴边界
        :return: (x, y) 新位置
        """
        if self.is_static:
            return self.x, self.y

        self.a = random.uniform(-self.h_max, self.h_max)
        dx = self.dt * self.v_max * cos(self.h)
        dy = self.dt * self.v_max * sin(self.h)

        new_x = self.x + dx
        new_y = self.y + dy

        if self.move_range > 0:
            dist_from_init = sqrt((new_x - self.init_x) ** 2 + (new_y - self.init_y) ** 2)
            if dist_from_init > self.move_range:
                self.h = -self.h
                dx = self.dt * self.v_max * cos(self.h)
                dy = self.dt * self.v_max * sin(self.h)
                new_x = self.x + dx
                new_y = self.y + dy

        if 0 > new_y or new_y > y_max:
            self.h = -self.h
            new_y = max(0, min(y_max, new_y))
        elif new_x < 0 or new_x > x_max:
            if self.h > 0:
                self.h = pi - self.h
            else:
                self.h = -pi - self.h
            new_x = max(0, min(x_max, new_x))

        self.x = new_x
        self.y = new_y

        self.h = (self.h + pi) % (2 * pi) - pi

        return self.x, self.y

    def distance_to(self, x: float, y: float) -> float:
        """
        计算障碍物到某点的距离
        :param x: float, 点的 x 坐标
        :param y: float, 点的 y 坐标
        :return: float, 距离
        """
        return sqrt((self.x - x) ** 2 + (self.y - y) ** 2)

    def is_in_collision_range(self, x: float, y: float, collision_radius: float) -> bool:
        """
        判断是否在碰撞范围内
        :param x: float, 点的 x 坐标
        :param y: float, 点的 y 坐标
        :param collision_radius: float, 碰撞半径
        :return: bool
        """
        return self.distance_to(x, y) <= collision_radius
