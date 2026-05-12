import os.path
from math import e

from PIL.ImageOps import cover

from src.utils.data_util import clip_and_normalize
from src.agent.uav import UAV
from src.agent.target import TARGET
import numpy as np
from math import pi
import random
from typing import List
from  src.agent.obstacle import OBSTACLE

class Environment:
    def __init__(self, n_uav: int, m_targets: int, x_max: float, y_max: float, na: int, n_obstacles: int = 0):
        """
        :param n_uav: scalar 无人机数量
        :param m_targets: scalar 目标数量
        :param x_max: scalar 环境x轴最大范围
        :param y_max: scalar y轴最大范围
        :param na: scalar 动作维度

        新增
        :param n_obstacles: 障碍物数量

        """
        # size of the environment
        self.x_max = x_max
        self.y_max = y_max

        # dim of action space and state space
        # communication(4 scalar, a), observation(4 scalar), boundary and state information(2 scalar, a)
        # self.state_dim = (4 + na) + 4 + (2 + na)
        #状态维度 =  通信 + 观测  +边界、状态
        self.state_dim = (4 + 1) + 4 + (2 + 1) + 4
        #动作维度
        self.action_dim = na

        # agents parameters in the environments
        self.n_uav = n_uav
        self.m_targets = m_targets
        self.n_obstacles = n_obstacles
        # agents
        self.uav_list = []
        self.target_list = []
        self.obstacle_list = []


        # position of uav and target ，记录每步无人机和目标的位置
        #新增障碍物位置
        self.position = {'all_uav_xs': [], 'all_uav_ys': [], 'all_target_xs': [], 'all_target_ys': [],'all_obstacle_xs': [], 'all_obstacle_ys': []}

        # 被无人机视野覆盖目标数量
        self.covered_target_num = []
        #新增被碰撞的数量
        self.covered_obstacle_num = []

    def __reset(self, t_v_max, t_h_max,u_v_max, u_h_max,o_v_max,o_h_max , na, dc, dp, dt,do,init_x, init_y):
        """
        作用是为了在每次的训练和评估之前 重置无人机数量和目标的初始状态，保证每轮的环境都是独立的
        :param t_v_max:  目标最大速度
        :param t_h_max: 目标最大转向角度
        :param u_v_max:  无人机最大速度
        :param u_h_max:  无人机的最大转向角度
        :param na:  动作维度
        :param dc:  无人机的通信距离
        :param dp:  无人机的追踪距离
        :param dt:  时间步长
        :param init_x:  无人机的初始坐标
        :param init_y:

        新增
        :param o_h_max: 障碍物最大转向速度
        :param o_v_max: 障碍物最大速度
        :param do: 无人机碰撞距离
        :return:
        """
        # 初始无人机的坐标
        if isinstance(init_x, List) and isinstance(init_y, List):
            #所有无人机都有独立的初始坐标
            self.uav_list = [UAV(init_x[i],
                                 init_y[i],
                                 random.uniform(-pi, pi), #随机的转向角度
                                 random.randint(0, self.action_dim - 1), # 初始为随机的动作
                                 u_v_max, u_h_max, na, dc, dp, dt,do) for i in range(self.n_uav)]
        elif not isinstance(init_x, List) and not isinstance(init_y, List):
            #所有无人机的初始坐标相同
            self.uav_list = [UAV(init_x,
                                 init_y,
                                 random.uniform(-pi, pi),
                                 random.randint(0, self.action_dim - 1),
                                 u_v_max, u_h_max, na, dc, dp, dt,do) for _ in range(self.n_uav)]
        elif isinstance(init_x, List):
            #x坐标统一
            self.uav_list = [UAV(init_x[i],
                                 init_y,
                                 random.uniform(-pi, pi),
                                 random.randint(0, self.action_dim - 1),
                                 u_v_max, u_h_max, na, dc, dp, dt,do) for i in range(self.n_uav)]
        elif isinstance(init_y, List):
            #y坐标统一
            self.uav_list = [UAV(init_x,
                                 init_y[i],
                                 random.uniform(-pi, pi),
                                 random.randint(0, self.action_dim - 1),
                                 u_v_max, u_h_max, na, dc, dp, dt,do) for i in range(self.n_uav)]
        else:
            print("wrong init position")

        #  初始化目标列表（随机位置、随机航向角、随机转向角）
        self.target_list = [TARGET(random.uniform(0, self.x_max),
                                   random.uniform(0, self.y_max),
                                   random.uniform(-pi, pi),
                                   random.uniform(-pi / 6, pi / 6),
                                   t_v_max, t_h_max, dt)
                            for _ in range(self.m_targets)]
        self.position = {'all_uav_xs': [], 'all_uav_ys': [], 'all_target_xs': [], 'all_target_ys': [],'all_obstacle_xs': [], 'all_obstacle_ys': []}
        self.covered_target_num = []



        #新增
        # 初始化障碍物列表 （随机位置、随机航向角、随机转向角）
        self.obstacle_list = [OBSTACLE(random.uniform(0, self.x_max),
                                       random.uniform(0, self.y_max),
                                       random.uniform(-pi, pi),
                                       random.uniform(-pi / 6, pi / 6),
                                       o_v_max, o_h_max, dt)
                              for _ in range(self.n_obstacles)]
        # self.position = {'all_'}
        #碰撞和覆盖集合
        self.covered_obstacle_num = []




    def reset(self, config):
        """
        从配置文件中提取参数进行重置
        :param config:
        :return:
        """
        # self.__reset(t_v_max=config["target"]["v_max"],
        #              t_h_max=pi / float(config["target"]["h_max"]),
        #              u_v_max=config["uav"]["v_max"],
        #              u_h_max=pi / float(config["uav"]["h_max"]),
        #              na=config["environment"]["na"],
        #              dc=config["uav"]["dc"],
        #              dp=config["uav"]["dp"],
        #              dt=config["uav"]["dt"],
        #              init_x=config['environment']['x_max']/2, init_y=config['environment']['y_max']/2)
        self.n_obstacles = config['environment']['n_obstacles']
        self.__reset(t_v_max=config["target"]["v_max"],
                     t_h_max=pi / float(config["target"]["h_max"]),
                     u_v_max=config["uav"]["v_max"],
                     u_h_max=pi / float(config["uav"]["h_max"]),
                     #新增
                     o_v_max=config["obstacle"]["v_max"],
                     o_h_max=pi / float(config["obstacle"]["h_max"]),
                     do=config["uav"]["do"],

                     na=config["environment"]["na"],
                     dc=config["uav"]["dc"],
                     dp=config["uav"]["dp"],
                     dt=config["uav"]["dt"],


                     init_x=[x * config['environment']['x_max'] / (config['environment']['n_uav'] + 1)
                             for x in range(1, config['environment']['n_uav']+1)],
                     init_y=config['environment']['y_max']/2)

    def get_states(self) -> (List['np.ndarray']):
        """
        获取无人机状态
        :return: list of np array, each element is a 1-dim array with size of 12
        """
        uav_states = []
        # collect the overall communication and target observation by each uav
        for uav in self.uav_list:
            uav_states.append(uav.get_local_state())
        return uav_states

    def step(self, config, pmi, actions):
        """
        状态转换函数:执行动作 -》更新位置 + 计算奖励 -》返回新状态 TODO: 新增碰撞
        :param config:
        :param pmi: PMI network
        :param actions: {0,1,...,Na - 1}
        :return: next states, rewards
        """
        # update the position of targets， 目标进行自主运动
        for i, target in enumerate(self.target_list):
            target.update_position(self.x_max, self.y_max)

        #新增
        for i,obstacle in enumerate(self.obstacle_list):
            obstacle.update_position(self.x_max, self.y_max)

        #更新所有无人机位置
        for i, uav in enumerate(self.uav_list):
            uav.update_position(actions[i],self.x_max,self.y_max) #执行动作，更新位置
            # #加入限制条件，防止无人机出框
            # uav.x = np.clip(uav.x, 10, self.x_max - 10)
            # uav.y = np.clip(uav.y, 10, self.y_max - 10)

            # observation and communication 观测与交流
            uav.observe_target(self.target_list) #观测目标
            uav.observe_uav(self.uav_list) #观测其他无人机

            #新增 观测障碍物
            uav.observe_obstacle(self.obstacle_list)


        #计算奖励 包含追踪奖励 边界惩罚 重复追踪惩罚
        #新增 碰撞惩罚
        (rewards,
         target_tracking_reward,
         boundary_punishment,
         duplicate_tracking_punishment,
         obstacle_punishment,
         fovea_bonus,
         exclusive_bonus,
         emergency_bonus) = self.calculate_rewards(config=config, pmi=pmi)

        #获取下一状态
        next_states = self.get_states()

        #计算当前无人机覆盖的目标数量
        covered_targets = self.calculate_covered_target()
        self.covered_target_num.append(covered_targets)
        #新增 计算当前无人机碰撞数
        covered_obstacle = self.calculate_covered_obstacle()
        self.covered_obstacle_num.append(covered_obstacle)


        # trace the position matrix，记录当前无人机和目标的位置
        target_xs, target_ys = self.__get_all_target_position()
        self.position['all_target_xs'].append(target_xs)
        self.position['all_target_ys'].append(target_ys)
        uav_xs, uav_ys = self.__get_all_uav_position()
        self.position['all_uav_xs'].append(uav_xs)
        self.position['all_uav_ys'].append(uav_ys)
        #新增 障碍物位置
        obstacle_xs,obstacle_ys = self.__get_all_obstacle_position()
        self.position['all_obstacle_xs'].append(obstacle_xs)
        self.position['all_obstacle_ys'].append(obstacle_ys)



        #封装奖励
        reward = {
            'rewards': rewards,
            'target_tracking_reward': target_tracking_reward,
            'boundary_punishment': boundary_punishment,
            'duplicate_tracking_punishment': duplicate_tracking_punishment,
            #新增
            'obstacle_punishment': obstacle_punishment,
            'fovea_bonus':fovea_bonus,
            'exclusive_bonus': exclusive_bonus,
            'emergency_bonus': emergency_bonus
        }

        return next_states, reward, covered_targets,covered_obstacle

    def __get_all_uav_position(self) -> (List[float], List[float]):
        """
        获取当前所有无人机的 x，y坐标
        :return: all the position of the uav through this epoch
        """
        uav_xs = []
        uav_ys = []
        for uav in self.uav_list:
            uav_xs.append(uav.x)
            uav_ys.append(uav.y)
        return uav_xs, uav_ys

    def __get_all_target_position(self) -> (List[float], List[float]):
        """
        获取当前目标的所有位置
        :return: all the position of the targets through this epoch
        """
        target_xs = []
        target_ys = []
        for target in self.target_list:
            target_xs.append(target.x)
            target_ys.append(target.y)
        return target_xs, target_ys

    #新增
    def __get_all_obstacle_position(self) -> (List[float], List[float]):
        """
        获取当前障碍物所有位置
        :return:
        """
        obstacle_xs = []
        obstacle_ys = []
        for obstacle in self.obstacle_list:
            obstacle_xs.append(obstacle.x)
            obstacle_ys.append(obstacle.y)
        return obstacle_xs, obstacle_ys



    def get_uav_and_target_position(self) -> (List[float], List[float], List[float], List[float]):
        """
        :return: both the uav and the target position matrix ，后续进行可视化
        """
        return (self.position['all_uav_xs'], self.position['all_uav_ys'],
                self.position['all_target_xs'], self.position['all_target_ys'])

    #TODO： 新增 后续可能会和上一函数合并
    def get_uav_obstacle_position(self) -> (List[float], List[float], List[float], List[float]):
        """
        获取全部障碍物位置
        :return:
        """
        return (self.position['all_uav_xs'], self.position['all_uav_ys'],
                self.position['all_obstacle_xs'], self.position['all_obstacle_ys'])

#待完善
    def calculate_rewards(self, config, pmi) -> ([float], float, float, float):
        """
        计算奖励 待完善 后续可能是主要完善点
        :param config:
        :param pmi:
        :return:
        """
        # raw reward first
        target_tracking_rewards = [] #追踪奖励
        boundary_punishments = [] #边界惩罚
        duplicate_tracking_punishments = [] #重复追踪惩罚

        #新增障碍物惩罚
        obstacle_punishments = []
        fovea_bonus_rewards = []
        #新增的追踪单一目标的奖励
        exclusive_rewards = []
        #新增的计算紧急避障奖励
        emergency_rewards = []

        #封装奖励与惩罚
        for uav in self.uav_list:
            # raw reward for each uav (not clipped)计算原始奖励（未裁剪/归一化）
            (target_tracking_reward,
             boundary_punishment,
             duplicate_tracking_punishment,
             obstacle_punishment,
             fovea_bonus,
             exclusive,
             emergency_bonus
             ) = uav.calculate_raw_reward(self.uav_list, self.target_list, self.obstacle_list,self.x_max, self.y_max)

            # 奖励裁剪+归一化（避免极端值影响训练）
            #追踪奖励：0~2×目标数 → 归一化到0~1
            target_tracking_reward = clip_and_normalize(target_tracking_reward,
                                                        0, 28, 0, name="track")
            #4 * config['environment']['m_targets']
            #重复追踪惩罚：-e/2×无人机数~0 → 归一化到-1~0
            # duplicate_tracking_punishment = clip_and_normalize(duplicate_tracking_punishment,
            #                                                    -e / 2 * config['environment']['n_uav'], 0, -1,name="duplicate")
            duplicate_tracking_punishment = clip_and_normalize(duplicate_tracking_punishment, -2.5, 0, -1,name='duplicate')
            #越界惩罚：-0.5~0 → 归一化到-1~0
            boundary_punishment = clip_and_normalize(boundary_punishment, -2.0, 0, -1,name="boundary")

            #新增 碰撞惩罚
            obstacle_punishment = clip_and_normalize(obstacle_punishment, -2.0, 0, -1,name="obstacle")
            fovea_bonus = clip_and_normalize(fovea_bonus, 0, 0.5, 0, name="fovea")

            exclusive_bonus = clip_and_normalize(exclusive, 0, 2.1, 0,name="exclusive")

            emergency_bonus = clip_and_normalize(emergency_bonus, 0, 0.3, 0, name="emergency")



            # 保存奖励和惩罚
            target_tracking_rewards.append(target_tracking_reward)
            boundary_punishments.append(boundary_punishment)
            duplicate_tracking_punishments.append(duplicate_tracking_punishment)
            #new
            obstacle_punishments.append(obstacle_punishment)
            fovea_bonus_rewards.append(fovea_bonus)
            exclusive_rewards.append(exclusive)

            emergency_rewards.append(emergency_bonus)



            # 计算原始总奖励 加权和
            uav.raw_reward = (config["uav"]["alpha"] * target_tracking_reward + config["uav"]["beta"] *
                              boundary_punishment + config["uav"]["gamma"] * duplicate_tracking_punishment+
                              #new
                              config["uav"]["omega"] * obstacle_punishment+ config["uav"].get("fovea_weight", 0.5) * fovea_bonus+config["uav"].get("exclusive_weight", 0.5) * exclusive_bonus+config["uav"].get("emergency_weight", 0.2) * emergency_bonus)

        rewards = []
        #计算协作的奖励
        #待重新构建
        for uav in self.uav_list:
            reward = uav.calculate_cooperative_reward(self.uav_list, pmi, config['cooperative'])
            # uav.reward = clip_and_normalize(reward, -1, 1,name="reward")
            uav.reward = reward
            rewards.append(uav.reward)
        return (rewards, target_tracking_rewards, boundary_punishments, duplicate_tracking_punishments ,obstacle_punishments,
                fovea_bonus_rewards,exclusive_rewards,emergency_rewards)


    def save_position(self, save_dir, epoch_i):
        """
        整理无人机/目标位置数据
        :param save_dir:
        :param epoch_i:
        :return:
        """
        # os.makedirs(os.path.join(save_dir, "o_xy"), exist_ok=True)
        u_xy = np.array([self.position["all_uav_xs"],
                         self.position["all_uav_ys"]]).transpose()  # n_uav * num_steps * 2
        t_xy = np.array([self.position["all_target_xs"],
                         self.position["all_target_ys"]]).transpose()  # m_target * num_steps * 2
        #new
        o_xy = np.array([self.position["all_obstacle_xs"],
                         self.position["all_obstacle_ys"]]).transpose()

        np.savetxt(os.path.join(save_dir, "u_xy", 'u_xy' + str(epoch_i) + '.csv'),
                   u_xy.reshape(-1, 2), delimiter=',', header='x,y', comments='')
        np.savetxt(os.path.join(save_dir, "t_xy", 't_xy' + str(epoch_i) + '.csv'),
                   t_xy.reshape(-1, 2), delimiter=',', header='x,y', comments='')
        #new
        np.savetxt(os.path.join(save_dir, "o_xy", 'o_xy' + str(epoch_i) + '.csv'),
                   o_xy.reshape(-1, 2), delimiter=',', header='x,y', comments='')

    def save_covered_num(self, save_dir, epoch_i):
        """
        整理覆盖目标数数据
        :param save_dir:
        :param epoch_i:
        :return:
        """
        covered_target_num_array = np.array(self.covered_target_num).reshape(-1, 1)

        np.savetxt(os.path.join(save_dir, "covered_target_num", 'covered_target_num' + str(epoch_i) + '.csv'),
                   covered_target_num_array, delimiter=',', header='covered_target_num', comments='')

    def save_obstacle_num(self, save_dir, epoch_i):
        """
        新增 存储每个轮次碰撞信息
        :param save_dir:
        :param epoch_i:
        :return:
        """
        os.makedirs(os.path.join(save_dir, "covered_obstacle_num"), exist_ok=True)
        covered_obstacle_num_array = np.array(self.covered_obstacle_num).reshape(-1, 1)

        np.savetxt(os.path.join(save_dir , "covered_obstacle_num", 'covered_obstacle_num' + str(epoch_i) + '.csv'),
                   covered_obstacle_num_array, delimiter=',', header='covered_obstacle_num', comments='')

    def calculate_covered_target(self):
        """
        计算当前步被覆盖的目标数（无人机距离<do视为覆盖）
        :return:
        """
        covered_target_num = 0
        for target in self.target_list:
            for uav in self.uav_list:
                if uav.distance(uav.x, uav.y, target.x, target.y) < uav.do:
                    covered_target_num += 1
                    break
        return covered_target_num

    def calculate_covered_obstacle(self):
        """
        新增 计算碰撞数量
        :return:
        """

        covered_obstacle_num = 0
        for obstacle in self.obstacle_list:
            for uav in self.uav_list:
                if uav.distance(uav.x,uav.y,obstacle.x,obstacle.y) < uav.do:
                    covered_obstacle_num += 1
                    break
        return covered_obstacle_num

    def update_size(self, x_max, y_max):
        """
        随着训练回合数，不断扩大可搜索区域
        :param x_max:
        :param y_max:
        :return:
        """
        self.x_max = x_max
        self.y_max = y_max
