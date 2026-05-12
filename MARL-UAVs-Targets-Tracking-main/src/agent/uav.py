import math
import random
import numpy as np
from math import cos, sin, sqrt, exp, pi, e, atan2
from typing import List, Tuple
from src.models.PMINet import PMINetwork
from src.agent.target import TARGET
from scipy.special import softmax
from src.agent.obstacle import OBSTACLE
from ..utils.data_util import clip_and_normalize


class UAV:
    def __init__(self, x0, y0, h0, a_idx, v_max, h_max, na, dc, dp, dt,do):
        """
        无人机初始化
        :param dt: float, 步长 采样的时间间隔
        :param x0: float, 坐标
        :param y0: float, 坐标
        :param h0: float, 朝向
        :param v_max: float, 最大线速度
        :param h_max: float, 最大角速度
        :param na: int, 动作空间的维度
        :param dc: float, 与无人机交流的最大距离
        :param dp: float, 观测目标的最大距离
        :param do: float, 碰撞目标的最大距离

        """
        # the position, velocity and heading of this uav
        self.x = x0
        self.y = y0
        self.h = h0
        self.v_max = v_max

        # the max heading angular rate and the action of this uav
        self.h_max = h_max
        self.Na = na

        # action
        self.a = a_idx

        # the maximum communication distance and maximum perception distance
        self.dc = dc
        self.dp = dp
        self.do = do

        # time interval
        self.dt = dt

        # set of local information
        # self.communication = []
        self.target_observation = [] #感知到的目标信息
        self.uav_communication = []  #通信范围内的队友信息
        #new
        self.obstacle_observation = []

        self.target_tracking_steps = {}  # 字典，key为目标id，value为连续观测步数

        # reward
        self.raw_reward = 0
        self.reward = 0

        self.last_closest_obstacle_dist = None  # 用于紧急避障奖励，记录上一时刻最近障碍物的距离

    def __distance(self, target) -> float:
        """
        计算无人机到目标/队友的欧氏距离
        calculate the distance from uav to target
        :param target: class UAV or class TARGET
        :return: scalar
        """
        return sqrt((self.x - target.x) ** 2 + (self.y - target.y) ** 2)

    @staticmethod
    def distance(x1, y1, x2, y2) -> float:
        """
        计算两点间欧氏距离
        calculate the distance from uav to target
        :param x2:
        :param y1:
        :param x1:
        :param y2:
        :return: scalar
        """
        return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def discrete_action(self, a_idx: int) -> float:
        """
        动作处理，将离散的动作索引转化为实际的转向角度
        from the action space index to the real difference
        :param a_idx: {0,1,...,Na - 1}
        :return: action : scalar 即角度改变量
        """
        # from action space to the real world action
        na = a_idx + 1  # 从 1 开始索引
        return (2 * na - self.Na - 1) * self.h_max / (self.Na - 1)

    def update_position(self, action: 'int',x_max,y_max) -> (float, float, float):
        """
         执行动作 → 更新位置和航向角
        receive the index from action space, then update the current position
        :param action: {0,1,...,Na - 1}
        :return:新坐标+航向角度
        """
        self.a = action
        a = self.discrete_action(action)  # 有可能把这行放到其他位置

        dx = self.dt * self.v_max * cos(self.h)  # x 方向位移
        dy = self.dt * self.v_max * sin(self.h)  # y 方向位移
        self.x += dx
        self.y += dy
        #新增碰撞反弹。模拟真实轨迹
        # 左右边界反弹
        if self.x <= 2 or self.x >=x_max - 2:
            self.h = pi - self.h

        # 上下边界反弹
        if self.y <= 2 or self.y >= y_max - 2:
            self.h = -self.h

        # ===================== 安全锁边（绝不越界） =====================
        self.x = np.clip(self.x, 2, x_max - 2)
        self.y = np.clip(self.y, 2, y_max - 2)



        self.h += self.dt * a  # 更新朝向角度
        self.h = (self.h + pi) % (2 * pi) - pi  # 确保朝向角度在 [-pi, pi) 范围内

        return self.x, self.y, self.h  # 返回agent的位置和朝向(heading/theta)

    # def observe_obstacle(self, obstacle_list: List['OBSTACLE'],relative = True) :
    #     pass


    def observe_target(self, targets_list: List['TARGET'], relative=True):
        """
        感知通信范围内的目标，存储相对/绝对状态
        :param relative: 是否将自身设为坐标原点
        :param targets_list: [class UAV]
        :return: None
        """
        self.target_observation = []  # Reset observed targets
        current_observed_ids = set()
        for target in targets_list:
            dist = self.__distance(target)
            if dist <= self.dp: #仅感知dp范围内的目标
                current_observed_ids.add(id(target))
                # add (x, y, vx, vy) information
                if relative: #如果使用的是相对坐标，相应的应该使用相对速度进行感知
                    self.target_observation.append(((target.x - self.x) / self.dp,
                                                    (target.y - self.y) / self.dp,
                                                    cos(target.h) * target.v_max / self.v_max - cos(self.h),
                                                    sin(target.h) * target.v_max / self.v_max - sin(self.h)))
                else:
                    self.target_observation.append((target.x / self.dp,
                                                    target.y / self.dp,
                                                    cos(target.h) * target.v_max / self.v_max,
                                                    sin(target.h) * target.v_max / self.v_max))
            # 更新连续观测步数
        for t_id in list(self.target_tracking_steps.keys()):
            if t_id in current_observed_ids:
                self.target_tracking_steps[t_id] += 1
            else:
                del self.target_tracking_steps[t_id]  # 一旦丢失，重置
        for t_id in current_observed_ids:
            if t_id not in self.target_tracking_steps:
                self.target_tracking_steps[t_id] = 1

    def observe_obstacle(self, obstacle_list: List['OBSTACLE'], relative = True):
        """
        感知障碍物信息 新增函数 可能存在修改
        :param obstacle_list:
        :param relative:
        :return:
        """
        self.obstacle_observation = []
        for obstacle in obstacle_list:
            dist = self.__distance(obstacle)
            if dist <=self.do:
                if relative:
                    self.obstacle_observation.append(((obstacle.x - self.x) / self.do,
                                                      (obstacle.y - self.y) / self.do,
                                                      cos(obstacle.h) * obstacle.v_max / self.v_max - cos(self.h),
                                                      sin(obstacle.h) * obstacle.v_max / self.v_max - sin(self.h)))
                else:
                    self.obstacle_observation.append((obstacle.x / self.do,
                                                      (obstacle.y - self.y) / self.do,
                                                      cos(obstacle.h) * obstacle.v_max / self.v_max - cos(self.h),
                                                      sin(obstacle.h) * obstacle.v_max / self.v_max - sin(self.h)))




    def observe_uav(self, uav_list: List['UAV'], relative=True):  # communication
        """
        与其他无人机进行通信，存储相对/绝对状态
        :param relative: relative to uav itself
        :param uav_list: [class UAV]
        :return:
        """
        self.uav_communication = []  # Reset observed targets
        for uav in uav_list:
            dist = self.__distance(uav)
            if dist <= self.dc and uav != self: #进通信dc范围内的其他队友
                # add (x, y, vx, vy, a) information
                if relative:
                    self.uav_communication.append(((uav.x - self.x) / self.dc,
                                                   (uav.y - self.y) / self.dc,
                                                   cos(uav.h) - cos(self.h),
                                                   sin(uav.h) - sin(self.h),
                                                   (uav.a - self.a) / self.Na))
                else:
                    self.uav_communication.append((uav.x / self.dc,
                                                   uav.y / self.dc,
                                                   cos(uav.h),
                                                   sin(uav.h),
                                                   uav.a / self.Na))

    def __get_all_local_state(self) -> (List[Tuple[float, float, float, float, float]],
                                        List[Tuple[float, float, float, float]], Tuple[float, float, float],
                                        List[Tuple[float, float, float, float]]):
        """
        将状态进行拆分，拆分为：队友通信5 + 目标感知4 + 自身状态3 +可能存在 （障碍感知 4维度）
        :return: [(x, y, vx, by, na),...] for uav, [(x, y, vx, vy)] for targets, (x, y, na) for itself
        """
        return self.uav_communication, self.target_observation, (self.x / self.dc, self.y / self.dc, self.a / self.Na),self.obstacle_observation

    def __get_local_state_by_weighted_mean(self) -> 'np.ndarray':
        """
        将多目标的感知信息和队友的通信信息加权平均为12维的向量  权重  = 1/距离
        :return: return weighted state: ndarray: (12)
        """
        communication, observation, state ,obstacle = self.__get_all_local_state()
        #如果是队友通信信息，加权平均（5维），多了个动作维度
        if communication:
            d_communication = []  # store the distance from each uav to itself
            for x, y, vx, vy, na in communication:
                d_communication.append(min(self.distance(x, y, self.x, self.y), 1))

            # regularization by the distance
            # communication = self.__transform_to_array2d(communication)
            communication = np.array(communication)
            communication_weighted = communication / np.array(d_communication)[:, np.newaxis]
            average_communication = np.mean(communication_weighted, axis=0)
        else:
            # average_communication = np.zeros(4 + self.Na)  # empty communication
            average_communication = -np.ones(4 + 1)  # empty communication  # TODO -1合法吗
        #对目标感知信息的加权平均（4维）
        if observation:
            d_observation = []  # store the distance from each target to itself
            for x, y, vx, vy in observation:
                d_observation.append(min(self.distance(x, y, self.x, self.y), 1))

            # regularization by the distance
            observation = np.array(observation)
            observation_weighted = observation / np.array(d_observation)[:, np.newaxis]
            average_observation = np.mean(observation_weighted, axis=0)
        else:
            average_observation = -np.ones(4)  # empty observation  # TODO -1合法吗

        if obstacle:
            d_obstacle = []
            for x, y, vx, vy in obstacle:
                d_obstacle.append(min(self.distance(x, y, self.x, self.y), 1))
            obstacle = np.array(obstacle)
            obstacle_weighted = obstacle / np.array(d_obstacle)[:, np.newaxis]
            average_obstacle = np.mean(obstacle_weighted, axis=0)
        else:
            average_obstacle = -np.ones(4)



        #将所有信息进行拼接（12维） 目标、队友、自身的感知信息 4+5+3
        state = np.array(state)
        result = np.hstack((average_communication, average_observation, state, average_obstacle))
        return result

    def get_local_state(self) -> 'np.ndarray':
        """
        对外提供状态接口，返回12维向量
        :return: np.ndarray
        """
        # using weighted mean method:
        return self.__get_local_state_by_weighted_mean()

    # def __calculate_multi_target_tracking_reward(self, target_list) -> float:
    #     """
    #      原先的追踪奖励函数，缺乏持续的奖励
    #     追踪奖励计算，距离目标越近奖励越高
    #     calculate multi target tracking reward
    #     :return: scalar [1, 2)
    #     """
    #     track_reward = 0
    #     for target  in target_list:
    #         if target != self:
    #             distance = self.__distance(target)
    #             if distance <= self.dp:
    #                 reward = 1 + (self.dp - distance) / self.dp
    #                 # track_reward += clip_and_normalize(reward, 1, 2, 0)
    #                 track_reward += reward  # 没有clip, 在调用时外部clip
    #     return track_reward
    def __calculate_multi_target_tracking_reward(self, target_list):
        track_reward = 0
        for target in target_list:
            dist = self.__distance(target)
            if dist <= self.dp:
                base_reward = 1 + (self.dp - dist) / self.dp
                # 额外连续性奖励：连续跟踪超过5步后，每步额外+0.1
                continuous_bonus = 0.0
                t_id = id(target)
                if t_id in self.target_tracking_steps:
                    steps = self.target_tracking_steps[t_id]
                    if steps > 5:
                        continuous_bonus = 0.1 * min(steps - 5, 20) / 20  # 上限1.0
                track_reward += base_reward + continuous_bonus
        return track_reward

#奖励原先的radio，原先为2，范围太多，距离的太远也会收到惩罚，不合理
    # def __calculate_duplicate_tracking_punishment(self, uav_list: List['UAV'], radio=1.2) -> float:
    #     """
    #     重复最终追踪惩罚，使用指数惩罚，距离越近惩罚越重
    #     calculate duplicate tracking punishment
    #     :param uav_list: [class UAV]
    #     :param radio: radio用来控制惩罚的范围, 超出多远才算入惩罚
    #     :return: scalar (-e/2, -1/2]
    #     """
    #     total_punishment = 0
    #     for other_uav in uav_list:
    #         if other_uav != self:
    #             distance = self.__distance(other_uav)
    #             if distance <= radio * self.dp:
    #                 punishment = -0.5 * exp((radio * self.dp - distance) / (radio * self.dp))
    #                 # total_punishment += clip_and_normalize(punishment, -e/2, -1/2, -1)
    #                 total_punishment += punishment  # 没有clip, 在调用时外部clip
    #     return total_punishment
    def __calculate_duplicate_tracking_punishment(self, uav_list, target_list):
        """
        只惩罚同时观测到相同目标的队友（真正扎堆），
        而不是惩罚所有在通信范围内的队友。
        """
        punishment = 0.0
        for target in target_list:
            # 统计所有能感知到该目标的无人机
            observers = [u for u in uav_list if u.distance(u.x, u.y, target.x, target.y) <= u.dp]
            if len(observers) > 1 and self in observers:
                # 越多人扎堆同一目标，惩罚越大（上限 -0.5）
                # punishment -= min(0.5, (len(observers) - 1) * 0.2)
                punishment -= min(0.2, (len(observers) - 1) * 0.05)

        return max(punishment, -1.0)  # 限制总惩罚不低于 -1.0

    def __calculate_boundary_punishment(self, x_max: float, y_max: float) -> float:
        # 安全区域扩展：距离边界 min_distance 以内才开始警告
        min_distance = 50.0  # 比 dp 小得多，释放探索空间
        x_to_0 = self.x
        x_to_max = x_max - self.x
        y_to_0 = self.y
        y_to_max = y_max - self.y
        d_bdr = min(x_to_0, x_to_max, y_to_0, y_to_max)

        if d_bdr < min_distance:
            # 接近边界：用平方函数平滑惩罚，最大惩罚 -0.5
            penalty = -0.5 * ((min_distance - d_bdr) / min_distance) ** 2
        else:
            penalty = 0.0
        return penalty




        # return clip_and_normalize(boundary_punishment, -1/2, 0, -1)

    #新增
    #计算障碍物惩罚
    #后续需要新添加一个属性  用于计算碰撞区域
    #可能是主要修改区域

    # def __calculate_obstacle_punishment(self, obstacle_list: List['OBSTACLE']) -> float:
    #     """
    #     计算靠近障碍物的惩罚（负值），距离越近惩罚越重。
    #     设计原则：
    #       - 惩罚范围：[0, -1]（0 表示无惩罚，-1 表示完全重叠）
    #       - 仅当 distance < self.do 时施加惩罚
    #       - 多个障碍物时取最大惩罚（最危险的一个），避免累加导致过度惩罚
    #     :param obstacle_list: 障碍物对象列表，每个对象需有 x, y 属性
    #     :return: 惩罚值，范围 [-1, 0]
    #     """
    #     max_punishment = 0.0
    #     for obs in obstacle_list:
    #         distance = self.__distance(obs)
    #         if distance < self.do:
    #             normalized = (self.do - distance) / self.do  # [0, 1]
    #             punish = - (normalized ** 2)
    #             if punish < max_punishment:
    #                 max_punishment = punish
    #     return max_punishment
    #
    def __calculate_obstacle_punishment(self, obstacle_list: List['OBSTACLE']) -> float:
        max_punishment = 0.0
        for obs in obstacle_list:
            distance = self.__distance(obs)
            if distance < self.do:
                punish = -0.5 * (self.do - distance) / self.do  # 线性，范围 [ -0.5, 0 ]
                if punish < max_punishment:
                    max_punishment = punish
        return max_punishment

    def __calculate_exclusive_tracking_bonus(self, target_list, uav_list):
        """
        独占追踪目标奖励
        如果某个目标只有本机在追踪（感知范围内仅本机），则给予奖励。
        奖励大小可调，此处设为每个独占目标给予 0.3。
        """
        bonus = 0.0
        for target in target_list:
            # 统计所有无人机中距离目标在 dp 内的数量
            observers = [u for u in uav_list if u.distance(u.x, u.y, target.x, target.y) <= u.dp]
            if len(observers) == 1 and observers[0] == self:
                bonus += 0.3  # 可调参数
        return bonus

    def __calculate_emergency_avoidance_bonus(self, obstacle_list, safe_dist=150.0, margin=2.0):
        """
        如果上一时刻处于危险距离内，且本时刻距离明显增大，则给予成功逃避奖励。
        safe_dist: 危险距离阈值
        margin: 最小距离增量才算成功逃离
        """
        if not obstacle_list:
            self.last_closest_obstacle_dist = None
            return 0.0

        current_closest = min(self.__distance(obs) for obs in obstacle_list)
        bonus = 0.0

        if self.last_closest_obstacle_dist is not None and self.last_closest_obstacle_dist < safe_dist:
            if current_closest > self.last_closest_obstacle_dist + margin:
                bonus = 0.3  # 成功逃离奖励

        self.last_closest_obstacle_dist = current_closest
        return bonus




    #TODO可能存在修改 封装障碍物惩罚
    def calculate_raw_reward(self, uav_list: List['UAV'], target__list: List['TARGET'],obstacle_list:List['OBSTACLE'], x_max, y_max):
        """
        封装三类奖励（追踪奖励+边界惩罚+重复追踪惩罚）
        calculate three parts of the reward/punishment for this uav
        :return: float, float, float
        """
        tracking_reward = self.__calculate_multi_target_tracking_reward(target__list)
        boundary_punishment = self.__calculate_boundary_punishment(x_max, y_max)
        # dup_punishment = self.__calculate_duplicate_tracking_punishment(uav_list)
        #新增惩罚
        obstacle_punishment = self.__calculate_obstacle_punishment(obstacle_list)
        # alloc = self.__calculate_target_allocation_punishment(target__list, uav_list)
        fovea = self.__calculate_fovea_bonus(target__list)  # 新增
        exclusive = self.__calculate_exclusive_tracking_bonus(target__list, uav_list)
        dup_punishment = self.__calculate_duplicate_tracking_punishment(uav_list, target__list)

        avoidance_bonus = self.__calculate_emergency_avoidance_bonus(obstacle_list, safe_dist=150.0, margin=2.0)

        return tracking_reward, boundary_punishment, dup_punishment,obstacle_punishment,fovea,exclusive,avoidance_bonus

    def __calculate_cooperative_reward_by_pmi(self, uav_list: List['UAV'], pmi_net: "PMINetwork", a) -> float:
        """
        计算协作奖励 = (1-a)×自身奖励 + a×(队友奖励×PMI依赖度)
        :param pmi_net: class PMINetwork
        :param uav_list: [class UAV]
        :param a:协作系数（0=自私，1=完全协作）
        :return:
        """
        if a == 0:  # 提前判断，节省计算的复杂度
            return self.raw_reward

        neighbor_rewards = []
        neighbor_dependencies = []
        la = self.get_local_state()

        for other_uav in uav_list:
            if other_uav != self and self.__distance(other_uav) <= self.dp:
                neighbor_rewards.append(other_uav.raw_reward)
                other_uav_la = other_uav.get_local_state()
                _input = la * other_uav_la
                neighbor_dependencies.append(pmi_net.inference(_input.squeeze()))

        if len(neighbor_rewards):
            neighbor_rewards = np.array(neighbor_rewards)
            neighbor_dependencies = np.array(neighbor_dependencies).astype(np.float32)
            softmax_values = softmax(neighbor_dependencies)
            reward = (1 - a) * self.raw_reward + a * np.sum(neighbor_rewards * softmax_values).item()
        else:
            reward = (1 - a) * self.raw_reward
        return reward

    def __calculate_cooperative_reward_by_mean(self, uav_list: List['UAV'], a) -> float:
        """
        计算无PMI的简单协作，取队友奖励的平均值
        :param uav_list: [class UAV]
        :param a: float, proportion of selfish and sharing
        :return:
        """
        if a == 0:  # 提前判断，节省计算的复杂度
            return self.raw_reward

        neighbor_rewards = []
        for other_uav in uav_list:
            if other_uav != self and self.__distance(other_uav) <= self.dp:
                neighbor_rewards.append(other_uav.raw_reward)
        # 没有加入PMI网络
        reward = (1 - a) * self.raw_reward + a * sum(neighbor_rewards) / len(neighbor_rewards) \
            if len(neighbor_rewards) else 0
        return reward
    def __calculate_target_allocation_punishment(self, target_list, uav_list):
        """
        对于每个目标，如果有多架无人机同时观测到它，则对观测该目标的无人机施加惩罚（越集中惩罚越大）
        不好用
        """
        punishment = 0.0
        for target in target_list:
            # 找到所有能看到该目标的无人机（包括自己）
            observers = []
            for uav in uav_list:
                if uav.distance(uav.x, uav.y, target.x, target.y) <= uav.dp:
                    observers.append(uav)
            n_obs = len(observers)
            if n_obs > 1 and self in observers:
                # 惩罚幅度：与观测者数量成正比，但最多惩罚到 -0.5
                punishment -= min(0.5, (n_obs - 1) * 0.15)
        return max(punishment, -1.0)

    def __calculate_fovea_bonus(self, target_list):
        """
        计算视野中心奖励：目标越靠近无人机正前方，奖励越高。
        """
        bonus = 0.0
        for target in target_list:
            if self.__distance(target) > self.dp:
                continue  # 不在感知范围内，无奖励
            # 计算目标相对于无人机的方位角
            dx = target.x - self.x
            dy = target.y - self.y
            angle_to_target = math.atan2(dy, dx)
            # 计算绝对角度差，并归一化到 [0, pi]
            angle_diff = abs(angle_to_target - self.h)
            angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi  # 保证在 [-pi, pi]
            angle_diff = abs(angle_diff)
            # 视野中心奖励：角度差小于 30°（pi/6）时给分，线性衰减，最大 0.5
            max_angle = math.pi / 6  # 30度
            if angle_diff <= max_angle:
                bonus += min(0.5, 0.5 * (1 - angle_diff / max_angle))
        return  min(bonus, 0.5)


    def calculate_cooperative_reward(self, uav_list: List['UAV'], pmi_net=None, a=0.5) -> float:
        """
        对外接口，使用PMI网络计算PMI协作奖励，否则使用均值协作奖励
        :param uav_list:
        :param pmi_net:
        :param a: 0: selfish, 1: completely shared
        :return:
        """
        if pmi_net:
            return self.__calculate_cooperative_reward_by_pmi(uav_list, pmi_net, a)
        else:
            return self.__calculate_cooperative_reward_by_mean(uav_list, a)



    def get_action_by_direction(self, target_list, uav_list):
        """
        基于epsilon贪心算法和最优目标选择的动作选择，用于后续的对比实验，无深度学习
        :param target_list:
        :param uav_list:
        :return:
        """
        def distance(x1, y1, x2, y2):
            return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        # 奖励和惩罚权重
        target_reward_weight = 1.0
        repetition_penalty_weight = 0.8
        self.epsilon = 0.25  #25的概率随机动作
        self.continue_tracing = 0.3 #30的概率保持当前动作
        
        best_score = float('-inf')
        best_angle = 0.0

        # 随机扰动：以epsilon的概率选择随机目标
        #小于epsilon则随机选择一个动作
        if random.random() < self.epsilon:
            return np.random.randint(0, self.Na)
        else:
            #便利所有目标，计算得分
            for target in target_list:
                target_x, target_y = target.x, target.y

                # 当前无人机到目标的距离
                dist_to_target = distance(self.x, self.y, target_x, target_y)

                # 重复追踪的惩罚，考虑其他无人机在重复追踪半径内是否在追踪同一目标
                repetition_penalty = 0.0
                for uav in uav_list:
                    uav_x, uav_y = uav.x, uav.y
                    if (uav_x, uav_y) != (self.x, self.y):
                        dist_to_target_from_other_uav = distance(uav_x, uav_y, target_x, target_y)
                        if dist_to_target_from_other_uav < self.dc:
                            repetition_penalty += repetition_penalty_weight

                # 计算当前目标的得分 = 距离奖励- 重复惩罚
                score = target_reward_weight / dist_to_target - repetition_penalty

                # 根据得分选择最优目标
                if score > best_score:
                    best_score = score
                    best_angle = np.arctan2(target_y - self.y, target_x - self.x) - self.h

        # 以continue_tracing的概率保持上一个动作
        if random.random() < self.continue_tracing:
            best_angle = 0
            
        actual_action = self.find_closest_a_idx(best_angle)
        return actual_action

    def find_closest_a_idx(self, angle):
        """
        通过输入的角度找到最接近的离散动作
        :param angle:
        :return:
        """
        best_idx = 0
        min_diff = float('inf')
        for idx in range(self.Na):
            act = self.discrete_action(idx)
            diff = abs(act - angle)
            if diff < min_diff:
                min_diff = diff
                best_idx = idx
        return best_idx