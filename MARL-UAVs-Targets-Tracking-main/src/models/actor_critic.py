import os.path

import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
#Actor—Critic网络框架
#基础残差块
class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):

        super(ResidualBlock, self).__init__()
        #第一层1D卷积层，用于提取特征
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels) #批归一化
        #第二层1D卷积，用于特征细化
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.down_sample = None
        # 下采样层（当输入/输出通道数不同或步长≠1时，对齐维度）
        if stride != 1 or in_channels != out_channels:
            self.down_sample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        """
        前向传播过程，卷积→BN→ReLU→卷积→BN
        :param x:
        :return:
        """
        residual = x #使用残差网络，保存原始输入
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # 残差路径：需要时下采样对齐维度
        if self.down_sample is not None:
            residual = self.down_sample(x)
        out += residual
        out = self.relu(out)
        return out

#残差策略网络：根据输入状态，输出动作的概率分布
class ResPolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(ResPolicyNet, self).__init__()
        self.conv1 = nn.Conv1d(1, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.residual_block1 = ResidualBlock(hidden_dim, hidden_dim)
        self.residual_block2 = ResidualBlock(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = x.unsqueeze(1) # [batch, 12] → [batch, 1, 12]（适配1D卷积的通道维度）
        # 初始卷积+BN+激活
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # 残差块特征提取
        x = self.residual_block1(x)
        x = self.residual_block2(x)
        # 平均池化（降维，固定输出维度）
        x = f.avg_pool1d(x, x.size(2))  # 这里使用平均池化[batch, hidden_dim, 12] → [batch, hidden_dim, 1]
        x = x.view(x.size(0), -1) # 展平 → [batch, hidden_dim]
        # 全连接层+softmax（输出动作概率）
        x = self.fc(x)
        return f.softmax(x, dim=1)


#残差价值网络，输入状态，输出状态的价值
class ResValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ResValueNet, self).__init__()
        self.conv1 = nn.Conv1d(1, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.residual_block1 = ResidualBlock(hidden_dim, hidden_dim)
        self.residual_block2 = ResidualBlock(hidden_dim, hidden_dim)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.residual_block1(x)
        x = self.residual_block2(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.squeeze(1) # [batch, 1] → [batch]

#全连接网络，主要用于对比实验效果
class FnnPolicyNet(nn.Module):
    def __init__(self, n_states, n_hiddens, n_actions):
        super(FnnPolicyNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_actions)

    # 前向传播
    def forward(self, x):
        x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
        x = f.relu(x)
        x = self.fc2(x)  # [b,n_hiddens]-->[b,n_actions]
        # 每个状态对应的动作的概率
        x = f.softmax(x, dim=1)  # [b,n_actions]-->[b,n_actions]
        return x


class FnnValueNet(nn.Module):
    def __init__(self, n_states, n_hiddens):
        super(FnnValueNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, 1)

    # 前向传播
    def forward(self, x):
        x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
        x = f.relu(x)
        x = self.fc2(x)  # [b,n_hiddens]-->[b,1]
        return x.squeeze(1)


class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 gamma, device):
        """
        :param state_dim: 特征空间的维数
        :param hidden_dim: 隐藏层的维数
        :param action_dim: 动作空间的维数
        :param actor_lr: actor网络的学习率
        :param critic_lr: critic网络的学习率
        :param gamma: 经验回放参数
        :param device: 用于训练的设备
        """
        # 策略网络
        self.actor = FnnPolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = FnnValueNet(state_dim, hidden_dim).to(device)  # 价值网络
        # 策略网络优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)  # 价值网络优化器
        self.gamma = gamma
        self.device = device

    def take_action(self, states):
        """
        动作选择
        :param states: nparray, size(state_dim,) 代表无人机的状态
        :return: 动作索引 + 动作概率
        """
        states_np = np.array(states)[np.newaxis, :]  # 直接使用np.array来转换
        states_tensor = torch.tensor(states_np, dtype=torch.float).to(self.device)
        probs = self.actor(states_tensor)
        action_dist = torch.distributions.Categorical(probs)  # TODO ?
        action = action_dist.sample()
        return action, probs

    def update(self, transition_dict):
        """
        基于时序差分（TD）更新Actor和Critic
        :param transition_dict: dict, 包含状态,动作, 单个无人机的奖励, 下一个状态的四元组
        :return:Actor损失/Critic损失/TD误差
        """
        # 转换为Tensor并移到指定设备
        states = torch.tensor(np.array(transition_dict['states']),
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        # actions = actions.long()
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device).squeeze()
        next_states = torch.tensor(np.array(transition_dict['next_states']),
                                   dtype=torch.float).to(self.device)

        # 时序差分目标
        td_target = rewards + self.gamma * self.critic(next_states)
        td_delta = td_target - self.critic(states)  # 时序差分误差
        actions = actions.long()  # 关键：转为int64（long）
        # 3. 计算Actor损失（策略梯度
        log_probs = torch.log(self.actor(states).gather(1, actions))
        # log_probs = torch.log(self.actor(states).gather(1, actions))
        actor_loss = torch.mean(-log_probs * td_delta.detach())
        # 4. 计算Critic损失（MSE）
        critic_loss = torch.mean(f.mse_loss(self.critic(states), td_target.detach()))
        # 5. 反向传播+更新
        self.actor_optimizer.zero_grad()  # 清空梯度
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return actor_loss, critic_loss, td_delta

    def save(self, save_dir, epoch_i):
        """
        保存模型参数和优化器状态
        :param save_dir:
        :param epoch_i:
        :return:
        """
        torch.save({
            'model_state_dict': self.actor.state_dict(),
            'optimizer_state_dict': self.actor_optimizer.state_dict()
        }, os.path.join(save_dir, "actor", 'actor_weights_' + str(epoch_i) + '.pth'))
        torch.save({
            'model_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.critic_optimizer.state_dict()
        }, os.path.join(save_dir, "critic", 'critic_weights_' + str(epoch_i) + '.pth'))

    def load(self, actor_path, critic_path):
        """
        加载模型参数和优化器状态
        :param actor_path:
        :param critic_path:
        :return:
        """
        if actor_path and os.path.exists(actor_path):
            checkpoint = torch.load(actor_path)
            self.actor.load_state_dict(checkpoint['model_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if critic_path and os.path.exists(critic_path):
            checkpoint = torch.load(critic_path)
            self.critic.load_state_dict(checkpoint['model_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
