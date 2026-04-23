import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

####新增模型MAPPO TODO：待完善
class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)

class Critic(nn.Module):
    def __init__(self, global_state_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(global_state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class MAPPO:
    def __init__(self, state_dim, global_state_dim, action_dim, hidden_dim, actor_lr, critic_lr, gamma, gae_lambda, clip_epsilon, device):
        self.actor = Actor(state_dim, hidden_dim, action_dim).to(device)
        self.critic = Critic(global_state_dim, hidden_dim).to(device)
        self.actor_old = Actor(state_dim, hidden_dim, action_dim).to(device)
        self.actor_old.load_state_dict(self.actor.state_dict())

        self.opt_actor = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.device = device

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()

    def calc_gae(self, rewards, values, next_value):
        # GAE 优势计算
        advantages = []
        advantage = 0.0
        next_v = next_value
        for r, v in reversed(list(zip(rewards, values))):
            delta = r + self.gamma * next_v - v
            advantage = delta + self.gamma * self.gae_lambda * advantage
            advantages.insert(0, advantage)
            next_v = v
        return advantages

    def update(self, batch):
        states = torch.tensor(batch['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(batch['actions'], dtype=torch.int64).to(self.device)
        rewards = torch.tensor(batch['rewards'], dtype=torch.float).to(self.device)
        next_states = torch.tensor(batch['next_states'], dtype=torch.float).to(self.device)
        log_probs_old = torch.tensor(batch['log_probs'], dtype=torch.float).to(self.device)

        values = self.critic(states).squeeze()
        next_values = self.critic(next_states).squeeze()
        advantages = self.calc_gae(rewards, values, next_values)
        advantages = torch.tensor(advantages, dtype=torch.float).to(self.device)
        returns = advantages + values

        # PPO 更新
        for _ in range(10):
            probs = self.actor(states)
            dist = torch.distributions.Categorical(probs)
            log_probs_new = dist.log_prob(actions)
            ratio = torch.exp(log_probs_new - log_probs_old)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = F.mse_loss(self.critic(states).squeeze(), returns)

            self.opt_actor.zero_grad()
            actor_loss.backward()
            self.opt_actor.step()

            self.opt_critic.zero_grad()
            critic_loss.backward()
            self.opt_critic.step()

        self.actor_old.load_state_dict(self.actor.state_dict())
        return actor_loss.item(), critic_loss.item()

    def save(self, save_dir, epoch_i):
        torch.save(self.actor.state_dict(), f"{save_dir}/actor_{epoch_i}.pth")
        torch.save(self.critic.state_dict(), f"{save_dir}/critic_{epoch_i}.pth")