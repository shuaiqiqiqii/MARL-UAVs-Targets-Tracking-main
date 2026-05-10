import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import os

class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class MAPPO:
    def __init__(self, state_dim, action_dim, hidden_dim=128,
                 actor_lr=3e-4, critic_lr=1e-3, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, device='cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        self.actor = Actor(state_dim, hidden_dim, action_dim).to(device)
        self.actor_old = Actor(state_dim, hidden_dim, action_dim).to(device)
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic = Critic(state_dim, hidden_dim).to(device)

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon

    def take_action(self, state):
        """输入单个状态 tensor [state_dim]，返回 (action, log_prob)"""
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0).to(self.device)
            else:
                state = state.to(self.device)
            prob = self.actor(state)
            dist = Categorical(prob)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()

    def update(self, trajectory):
        T, N = trajectory['actions'].shape
        states = torch.FloatTensor(trajectory['states']).to(self.device)  # [T, N, D]
        actions = torch.LongTensor(trajectory['actions']).to(self.device)  # [T, N]
        old_log_probs = torch.FloatTensor(trajectory['log_probs']).to(self.device)  # [T, N]
        rewards = torch.FloatTensor(trajectory['rewards']).to(self.device)  # [T, N]
        dones = torch.FloatTensor(trajectory['dones']).to(self.device)  # [T, N]

        # 计算旧值（用于 GAE，只用一次）
        with torch.no_grad():
            old_values = self.critic(states).squeeze(-1)  # [T, N]

        # 计算 GAE 优势（基于旧值）
        advantages = torch.zeros_like(rewards)
        gae = 0
        for t in reversed(range(T - 1)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * old_values[t + 1] * mask - old_values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae

        # 展平所有数据
        states_flat = states.view(-1, self.state_dim)  # [T*N, D]
        actions_flat = actions.view(-1, 1)
        old_log_probs_flat = old_log_probs.view(-1, 1)
        advantages_flat = advantages.view(-1, 1)

        # PPO 多轮更新
        for _ in range(5):  # 可根据 config 调整
            # 关键：每个 epoch 重新计算当前网络输出
            values = self.critic(states_flat).squeeze(-1)  # [T*N]
            prob = self.actor(states_flat)
            dist = torch.distributions.Categorical(prob)
            new_log_probs = dist.log_prob(actions_flat.squeeze(-1)).unsqueeze(-1)
            ratio = torch.exp(new_log_probs - old_log_probs_flat)

            # Actor loss (PPO-Clip)
            surr1 = ratio * advantages_flat
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_flat
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss (MSE with detached target)
            td_target = (advantages_flat + old_values.view(-1, 1)).detach().squeeze(-1)  # 使用旧值作为目标
            critic_loss = F.mse_loss(values, td_target)

            # 梯度更新
            self.opt_actor.zero_grad()
            self.opt_critic.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.opt_actor.step()
            self.opt_critic.step()

        # 更新旧策略参数
        self.actor_old.load_state_dict(self.actor.state_dict())
        return actor_loss.item(), critic_loss.item()

    def save(self, save_dir, epoch_i):
        # 确保子目录存在（get_config 中已建，但加一句更安全）
        actor_dir = os.path.join(save_dir, "actor")
        critic_dir = os.path.join(save_dir, "critic")
        os.makedirs(actor_dir, exist_ok=True)
        os.makedirs(critic_dir, exist_ok=True)

        torch.save(self.actor.state_dict(), os.path.join(actor_dir, f'actor_{epoch_i}.pth'))
        torch.save(self.critic.state_dict(), os.path.join(critic_dir, f'critic_{epoch_i}.pth'))

    def load(self, actor_path, critic_path):
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))
            self.actor_old.load_state_dict(self.actor.state_dict())
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))