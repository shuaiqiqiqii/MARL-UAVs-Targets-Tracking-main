import os.path
import csv
from tqdm import tqdm
import numpy as np
import torch
from utils.draw_util import draw_animation
from torch.utils.tensorboard import SummaryWriter
import random
import collections
import torch.nn.functional as F
from utils.exploration import GridExplorationBonus
# 强制使用CPU，避免50系显卡报错
torch.cuda.is_available = lambda : False

#训练结果存储类
class ReturnValueOfTrain:

    def __init__(self):
        self.return_list = [] #总奖励列表
        self.target_tracking_return_list = [] #目标追踪奖励列表
        self.boundary_punishment_return_list = [] #越界惩罚列表
        self.duplicate_tracking_punishment_return_list = [] #重复追踪惩罚

        self.average_covered_targets_list = [] #平均覆盖目标数量列表
        self.max_covered_targets_list = [] #最大覆盖目标数量列表
        #新增的碰撞惩罚列表 平均碰撞和最大碰撞列表
        self.obstacle_punishment_return_list = []
        self.average_covered_obstacles_list = []
        self.max_covered_obstacles_list = []
        self.fovea_bonus_return_list=[]


    def item(self):
        """
        将所有的指标封装成字典返回
        :return:
        """
        value_dict = {
            'return_list': self.return_list,
            'target_tracking_return_list': self.target_tracking_return_list,
            'boundary_punishment_return_list': self.boundary_punishment_return_list,
            'duplicate_tracking_punishment_return_list': self.duplicate_tracking_punishment_return_list,
            'average_covered_targets_list': self.average_covered_targets_list,
            'max_covered_targets_list': self.max_covered_targets_list,
            'obstacle_punishment_return_list': self.obstacle_punishment_return_list,
            'average_covered_obstacles_list': self.average_covered_obstacles_list,
            'max_covered_obstacles_list': self.max_covered_obstacles_list
        }
        return value_dict

    def save_epoch(self, reward, tt_return, bp_return, dtp_return, op_return,fovea_bonus_reward,average_targets,average_obstacle, max_targets,max_obstacle):
        """
        每轮训练后，保存当前轮次的所有指标
        :param max_obstacle: 一个轮次的最大碰撞数量
        :param average_obstacle: 平均碰撞数
        :param op_return: 碰撞惩罚返回值
        :param reward:奖励
        :param tt_return:
        :param bp_return:
        :param dtp_return:
        :param average_targets:
        :param max_targets:
        :return:
        """
        self.return_list.append(reward)
        self.target_tracking_return_list.append(tt_return)
        self.boundary_punishment_return_list.append(bp_return)
        self.duplicate_tracking_punishment_return_list.append(dtp_return)

        self.average_covered_targets_list.append(average_targets)
        self.max_covered_targets_list.append(max_targets)

        self.obstacle_punishment_return_list.append(op_return)
        self.average_covered_obstacles_list.append(average_obstacle)
        self.max_covered_obstacles_list.append(max_obstacle)
        self.fovea_bonus_return_list.append(fovea_bonus_reward)


#基础经验回访池，存储训练过程中的经验(状态-》动作-》奖励-》下一状态)，随机采样
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) #使用双端列表进行经验存储，队列满了之后自动删除最早的经验

    def add(self, transition_dict):
        # 从transition_dict中提取各个列表，状态、动作、奖励、下一状态
        states = transition_dict['states']
        actions = transition_dict['actions']
        rewards = transition_dict['rewards']
        next_states = transition_dict['next_states']

        # 将各个元素合并成元组，并添加到缓冲区中
        experiences = zip(states, actions, rewards, next_states)
        self.buffer.extend(experiences)

    def sample(self, batch_size):
        """
        随机采样指定数量的经验
        :param batch_size:
        :return:
        """
        transitions = random.sample(self.buffer, min(batch_size, self.size()))
        states, actions, rewards, next_states = zip(*transitions)

        # 构造返回的字典并返回
        sample_dict = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states
        }
        return sample_dict

    def size(self):
        return len(self.buffer)

#优先经验回放池
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        """
        初始化缓存池
        :param capacity: 缓冲区容量
        :param alpha: 优先级系数，0表示随机采样，1则表示完全按照优先级
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = collections.deque(maxlen=capacity) #缓冲区
        self.priorities = np.zeros((capacity,), dtype=np.float32) #每个经验的优先级
        self.pos = 0 #当前的存储位置

    def add(self, transition_dict):
        """
        提取经验并进行逐个相加
        :param transition_dict:
        :return:
        """
        states = transition_dict['states']
        actions = transition_dict['actions']
        rewards = transition_dict['rewards']
        next_states = transition_dict['next_states']
        #将状态、动作、奖励、下一状态封装

        experiences = zip(states, actions, rewards, next_states)

        for experience in experiences:
            #将新的经验优先级设为当前最大值，确保新的经验能被采样到
            max_priority = self.priorities.max() if self.buffer else 1.0
            #缓冲区满了就覆盖最早的，没满就进入队列
            if len(self.buffer) < self.capacity:
                self.buffer.append(experience)
            else:
                self.buffer[self.pos] = experience

            self.priorities[self.pos] = max_priority
            #位置循环
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """
        采样，让重要的经验比如奖励变化大、追踪目标成功、失败的经验被更多的采样
        :param batch_size:
        :param beta:
        :return:
        """
        if len(self.buffer) == 0:
            return dict(states=[], actions=[], rewards=[], next_states=[]), None, None

        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]
        #计算采样概率，优先级越高的采样的概率越大
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        #按照概率进行采样
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), p=probabilities)
        samples = [self.buffer[idx] for idx in indices]
        #计算权重，修改优先级采样导致的偏差
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        #拆分经验并封装
        batch = list(zip(*samples))
        states = np.array(batch[0])
        actions = np.array(batch[1])
        rewards = np.array(batch[2])
        next_states = np.array(batch[3])

        sample_dict = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
        }
        return sample_dict, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        """
        更新采样经验的优先级
        :param batch_indices:
        :param batch_priorities:
        :return:
        """
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority

    def size(self):
        return len(self.buffer)

def operate_epoch(config, env, agent, pmi, num_steps, cwriter_state=None, cwriter_prob=None,debug_print = False,explorer=None):
    transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'log_probs': []}
    log_prob_list = []
    episode_return = 0
    episode_target_tracking_return = 0
    episode_boundary_punishment_return = 0
    episode_duplicate_tracking_punishment_return = 0
    episode_obstacle_punishment_return = 0
    episode_fovea_bonus_return =0




    covered_targets_list = []
    covered_obstacles_list = []

    for i in range(num_steps):
        action_list = []
        all_states = []
        for uav in env.uav_list:
            st = uav.get_local_state()  # [16,] numpy
            all_states.append(st)

        # 使用 PMI 计算同一时间步所有无人机的内在奖励
        intrinsic_rewards = np.zeros(env.n_uav)
        if pmi is not None:
            with torch.no_grad():
                tensor_states = torch.FloatTensor(np.array(all_states)).to(agent.device)
                pmi_vals = pmi(tensor_states).squeeze()  # [N]
                # 简单内在奖励：PMI 值本身（可加权重）
                intrinsic_rewards = pmi_vals.cpu().numpy() * config.get("pmi_weight", 0.1)

        for idx, uav in enumerate(env.uav_list):
            state_np = all_states[idx]  # 保持 16 维，不拼接
            state_tensor = torch.FloatTensor(state_np).unsqueeze(0).to(agent.device)
            action, log_prob = agent.take_action(state_tensor)
            action_list.append(action)
            log_prob_list.append(log_prob)
            transition_dict['states'].append(state_np)

        next_state_list, reward_list, covered_targets, covered_obstacles = env.step(config, pmi, action_list)
        # 计算探索奖励（需要 explorer 对象）
        if explorer is not None:
            exp_bonus = np.array([explorer.get_bonus(uav.x, uav.y) for uav in env.uav_list])
            for uav in env.uav_list:
                explorer.update(uav.x, uav.y)
        else:
            exp_bonus = np.zeros(env.n_uav)

        # if debug_print:
        #     print(f"----- Episode step {i + 1} -----")
        #     for uav_idx in range(env.n_uav):
        #         tt = reward_list['target_tracking_reward'][uav_idx]
        #         bp = reward_list['boundary_punishment'][uav_idx]
        #         dp = reward_list['duplicate_tracking_punishment'][uav_idx]
        #         op = reward_list['obstacle_punishment'][uav_idx]
        #         total = reward_list['rewards'][uav_idx]
        #     #     print(
        #     #         f"  UAV {uav_idx}: track={tt:+.3f}, boundary={bp:+.3f}, dup={dp:+.3f}, obstacle={op:+.3f} → total={total:+.3f}")
        #     # print("")  # 空行分隔

        # 将环境奖励与内在奖励相加
        env_rewards = np.array(reward_list['rewards'])
        total_rewards = env_rewards + intrinsic_rewards  # 混合奖励

        transition_dict['actions'].extend(action_list)
        transition_dict['next_states'].extend(next_state_list)
        transition_dict['rewards'].extend(total_rewards.tolist())
        transition_dict['log_probs'] = log_prob_list

        episode_return += sum(total_rewards)
        episode_target_tracking_return += sum(reward_list['target_tracking_reward'])
        episode_boundary_punishment_return += sum(reward_list['boundary_punishment'])
        episode_duplicate_tracking_punishment_return += sum(reward_list['duplicate_tracking_punishment'])
        episode_obstacle_punishment_return += sum(reward_list['obstacle_punishment'])
        episode_fovea_bonus_return += sum(reward_list['fovea_bonus'])

        covered_targets_list.append(covered_targets)
        covered_obstacles_list.append(covered_obstacles)

    episode_return /= num_steps * env.n_uav
    episode_target_tracking_return /= num_steps * env.n_uav
    episode_boundary_punishment_return /= num_steps * env.n_uav
    episode_duplicate_tracking_punishment_return /= num_steps * env.n_uav
    episode_obstacle_punishment_return /= num_steps * env.n_uav
    episode_fovea_bonus_return /= num_steps * env.n_uav


    average_covered_targets = np.mean(covered_targets_list)
    average_covered_obstacles = np.mean(covered_obstacles_list)
    max_covered_targets = np.max(covered_targets_list)
    max_covered_obstacles = np.max(covered_obstacles_list)

    return (transition_dict, episode_return, episode_target_tracking_return,
            episode_boundary_punishment_return,episode_duplicate_tracking_punishment_return,episode_obstacle_punishment_return,episode_fovea_bonus_return,
            average_covered_targets, average_covered_obstacles,max_covered_targets, max_covered_obstacles)

def train(config, env, agent, pmi, num_episodes, num_steps, frequency):
    save_dir = os.path.join(config["save_dir"], "logs")
    writer = SummaryWriter(log_dir=save_dir)
    return_value = ReturnValueOfTrain()
    INIT_SIZE, FINAL_SIZE = 1000, 2000 #地图的初始大小和最终大小
    INIT_DP, FINAL_DP = 500, 300 #探索范围的初始大小和最终大小
    INIT_GAMMA, FINAL_GAMMA = 0.05, 0.15  # 重复惩罚权重
    INIT_TARGETS, FINAL_TARGETS = 7, 5 #目标数量
    INIT_OBS, FINAL_OBS = 1, 3  #障碍数量



    # 课程开始 / 结束的回合
    COURSE_START = 0
    COURSE_END = 2000  # 1500回合后达到最终难度
    explorer = GridExplorationBonus(INIT_SIZE, INIT_SIZE, grid_size=80, bonus_weight=0.00)
    # ===================== 修复：MAPPO不创建回放池 =====================
    buffer = None
    sample_size = 0
    if config['method'] != "MAPPO":
        buffer = PrioritizedReplayBuffer(config["actor_critic"]["buffer_size"])
        sample_size = config["actor_critic"]["sample_size"] if config["actor_critic"]["sample_size"] > 0 else config["environment"]["n_uav"] * num_steps

    with open(os.path.join(save_dir, 'state.csv'), mode='w', newline='') as state_file, \
            open(os.path.join(save_dir, 'prob.csv'), mode='w', newline='') as prob_file:
        cwriter_state = csv.writer(state_file)
        cwriter_prob = csv.writer(prob_file)
        cwriter_state.writerow(['state'])
        cwriter_prob.writerow(['prob'])



        with tqdm(total=num_episodes, desc='Episodes') as pbar:
            for i in range(num_episodes):
                progress = min(1.0, max(0.0, (i - COURSE_START) / (COURSE_END - COURSE_START)))

                current_size = int(INIT_SIZE + (FINAL_SIZE - INIT_SIZE) * progress)
                dp = int(INIT_DP + (FINAL_DP - INIT_DP) * progress)
                gamma_weight = INIT_GAMMA + (FINAL_GAMMA - INIT_GAMMA) * progress
                m_targets = int(INIT_TARGETS + (FINAL_TARGETS - INIT_TARGETS) * progress)
                n_obstacles = int(INIT_OBS + (FINAL_OBS - INIT_OBS) * progress)

                # 边界惩罚权重：前 800 回合为 0，之后线性增加到 0.05
                if i < 800:
                    beta = 0.0
                    explorer.bonus_weight = 0.0
                else:
                    beta = min(0.05, 0.05 * (i - 800) / 700)
                    explorer.bonus_weight = 0.005 * min(1.0, (i - 500) / 1000)  # 最高 0.005

                # 应用环境参数
                env.update_size(current_size, current_size) if hasattr(env, 'update_size') else None
                explorer.resize(current_size, current_size)
                config['environment']['x_max'] = current_size
                config['environment']['y_max'] = current_size
                config['uav']['dp'] = dp
                config['uav']['gamma'] = gamma_weight
                config['environment']['m_targets'] = m_targets
                config['environment']['n_obstacles'] = n_obstacles
                config['uav']['beta'] = beta


                # ------- PMI / cooperative 依旧按 schedule -------
                if  i < config["schedule"]["stage1_end"]:
                    config["pmi_weight"] = 0.0
                    config["cooperative"] = 0.0
                elif i < config["schedule"]["stage2_end"]:
                    config["pmi_weight"] = 0.005
                    config["cooperative"] = 0.0
                elif  i < config["schedule"]["stage3_end"]:
                    config["pmi_weight"] = 0.02
                    config["cooperative"] = 0.1
                else:
                    config["pmi_weight"] = 0.05
                    config["cooperative"] = 0.3



                env.reset(config=config)

                debug = False
                transition_dict, reward, tt_return, bp_return, \
                    dtp_return,op_return, fovea_bonus,average_targets, average_obstacles,max_targets,max_obstacles = operate_epoch(config, env, agent, pmi, num_steps,explorer=explorer)

                writer.add_scalar('reward', reward, i)
                writer.add_scalar('target_tracking_return', tt_return, i)
                writer.add_scalar('boundary_punishment', bp_return, i)
                writer.add_scalar('duplicate_tracking_punishment', dtp_return, i)
                writer.add_scalar('average_covered_targets', average_targets, i)
                writer.add_scalar('max_covered_targets', max_targets, i)
                writer.add_scalar('obstacle_punishment', op_return, i)
                writer.add_scalar('average_covered_obstacles', average_obstacles, i)
                writer.add_scalar('max_covered_obstacles', max_obstacles, i)
                writer.add_scalar('fovea_bonus', fovea_bonus, i)


                return_value.save_epoch(reward, tt_return, bp_return, dtp_return, op_return,fovea_bonus,average_targets, average_obstacles,max_targets,max_obstacles)

                # ===================== 修复：双模式 =====================
                if config['method'] != "MAPPO":
                    buffer.add(transition_dict)
                    sample_dict, indices, _ = buffer.sample(sample_size)
                    actor_loss, critic_loss, td_errors = agent.update(sample_dict)
                    buffer.update_priorities(indices, td_errors.abs().detach().cpu().numpy())
                else:
                    # ---------- 构建 MAPPO 需要的轨迹字典 ----------
                    T = num_steps
                    N = env.n_uav
                    states = np.array(transition_dict['states']).reshape(T, N, 16)  # [T, N, 16]
                    actions = np.array(transition_dict['actions']).reshape(T, N)  # [T, N]
                    log_probs_old = np.array(transition_dict['log_probs']).reshape(T, N)  # [T, N]
                    rewards = np.array(transition_dict['rewards']).reshape(T, N)  # [T, N]
                    dones = np.zeros_like(actions)  # [T, N]，无终止标志

                    trajectory = {
                        'states': states,
                        'actions': actions,
                        'log_probs': log_probs_old,
                        'rewards': rewards,
                        'dones': dones
                    }

                    actor_loss, critic_loss = agent.update(trajectory)

                    if pmi:
                        train_data = torch.tensor(np.array(transition_dict['states']), dtype=torch.float32).to(
                            agent.device)
                        avg_pmi_loss = pmi.train_pmi(config, train_data, env.n_uav)
                        writer.add_scalar('avg_pmi_loss', avg_pmi_loss, i)


                writer.add_scalar('actor_loss', actor_loss, i)
                writer.add_scalar('critic_loss', critic_loss, i)

                if (i + 1) % frequency == 0:
                    if pmi:
                        pbar.set_postfix({'episode': '%d' % (i + 1),
                                          'return': '%.3f' % np.mean(return_value.return_list[-frequency:]),
                                          'actor loss': '%f' % actor_loss,
                                          'critic loss': '%f' % critic_loss,
                                          'avg pmi loss': '%f' % avg_pmi_loss})
                    else:
                        pbar.set_postfix({'episode': '%d' % (i + 1),
                                          'return': '%.3f' % np.mean(return_value.return_list[-frequency:]),
                                          'actor loss': '%f' % actor_loss,
                                          'critic loss': '%f' % critic_loss})

                    draw_animation(config=config, env=env, num_steps=num_steps, ep_num=i)
                    agent.save(save_dir=config["save_dir"], epoch_i=i + 1)
                    if pmi:
                        pmi.save(save_dir=config["save_dir"], epoch_i=i + 1)
                    env.save_position(save_dir=config["save_dir"], epoch_i=i + 1)
                    env.save_covered_num(save_dir=config["save_dir"], epoch_i=i + 1)
                    env.save_obstacle_num(save_dir=config["save_dir"], epoch_i=i + 1)

                pbar.update(1)

    writer.close()
    return return_value.item()


def evaluate(config, env, agent, pmi, num_steps):
    return_value = ReturnValueOfTrain()
    env.reset(config=config)
    transition_dict, reward, tt_return, bp_return, dtp_return,op_return,average_targets,average_obstacles, max_targets,max_obstacles = operate_epoch(config, env, agent, pmi, num_steps)
    return_value.save_epoch(reward, tt_return, bp_return, dtp_return, op_return,average_targets,average_obstacles, max_targets,max_obstacles)
    draw_animation(config=config, env=env, num_steps=num_steps, ep_num=0)
    env.save_position(save_dir=config["save_dir"], epoch_i=0)
    env.save_covered_num(save_dir=config["save_dir"], epoch_i=0)
    env.save_obstacle_num(save_dir=config["save_dir"], epoch_i=0)
    return return_value.item()


def run_epoch(config, pmi, env, num_steps):
    transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': []}
    episode_return = 0
    episode_target_tracking_return = 0
    episode_boundary_punishment_return = 0
    episode_duplicate_tracking_punishment_return = 0
    episode_obstacle_punishment_return = 0
    covered_obstacle_list = []
    covered_targets_list = []

    for _ in range(num_steps):
        action_list = []
        for uav in env.uav_list:
            action = uav.get_action_by_direction(env.target_list, env.uav_list)
            action_list.append(action)

        next_state_list, reward_list, covered_targets ,covered_obstacles= env.step(config, pmi, action_list)

        transition_dict['actions'].extend(action_list)
        transition_dict['rewards'].extend(reward_list['rewards'])
        episode_return += sum(reward_list['rewards'])
        episode_target_tracking_return += sum(reward_list['target_tracking_reward'])
        episode_boundary_punishment_return += sum(reward_list['boundary_punishment'])
        episode_duplicate_tracking_punishment_return += sum(reward_list['duplicate_tracking_punishment'])
        covered_targets_list.append(covered_targets)
        episode_obstacle_punishment_return += sum(reward_list['obstacle_punishment'])
        covered_obstacle_list.append(covered_obstacles)

    average_covered_targets = np.mean(covered_targets_list)
    max_covered_targets = np.max(covered_targets_list)
    average_covered_obstacles = np.mean(covered_obstacle_list)
    max_covered_obstacles = np.max(covered_obstacle_list)

    return (transition_dict, episode_return, episode_target_tracking_return,
            episode_boundary_punishment_return, episode_duplicate_tracking_punishment_return,episode_obstacle_punishment_return,
            average_covered_targets,average_covered_obstacles, max_covered_targets,max_covered_obstacles)


def run(config, env, pmi, num_steps):
    return_value = ReturnValueOfTrain()
    env.reset(config=config)
    transition_dict, reward, tt_return, bp_return, dtp_return, op_return,average_targets, average_obstacles,max_targets,max_obstacles = run_epoch(config, pmi, env, num_steps)
    return_value.save_epoch(reward, tt_return, bp_return, dtp_return, op_return,average_targets, average_obstacles,max_targets,max_obstacles)
    draw_animation(config=config, env=env, num_steps=num_steps, ep_num=0)
    env.save_position(save_dir=config["save_dir"], epoch_i=0)
    env.save_covered_num(save_dir=config["save_dir"], epoch_i=0)
    env.save_obstacle_num(save_dir=config["save_dir"], epoch_i=0)
    return return_value.item()


def compute_gae(rewards, values, next_values, gamma, gae_lambda):
    advantages = []
    advantage = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] - values[t]
        advantage = delta + gamma * gae_lambda * advantage
        advantages.insert(0, advantage)
    returns = np.array(advantages) + np.array(values)
    return advantages, returns