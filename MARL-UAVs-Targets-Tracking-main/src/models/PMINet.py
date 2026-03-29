import torch.optim as optim
import numpy as np
# 定义模型
import torch
import torch.nn as nn
import torch.nn.functional as f
import os
#PMi网络架构

#损失函数，用于最大化两个智能体之间的互信息
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    @staticmethod
    def forward(output1, output2):
        """
         互信息最大化损失：log(1+e^(-o1)) + log(1+e^(o2))
        :param output1:
        :param output2:
        :return:
        """
        loss = torch.mean(torch.log(1 + torch.exp(-output1)) + torch.log(1 + torch.exp(output2)))
        return loss

#PMI 网络的作用是将无人机的通信特征、观测特征、边界特征编码为统一的高维特征，并通过自定义损失最大化不同无人机之间的特征互信息。
class PMINetwork(nn.Module):
    def __init__(self, comm_dim=5, obs_dim=4, boundary_state_dim=3, hidden_dim=64, b2_size=3000):
        super(PMINetwork, self).__init__()
        self.comm_dim = comm_dim #通信特征维度5
        self.obs_dim = obs_dim #局部观测特征维度4
        self.boundary_state_dim = boundary_state_dim #边界状态特征3
        self.hidden_dim = hidden_dim
        self.b2_size = b2_size
        # 分支1：通信特征编码（带批归一化）
        self.fc_comm = nn.Linear(comm_dim, hidden_dim)
        self.bn_comm = nn.BatchNorm1d(hidden_dim)
        # 分支2：观测特征编码（带批归一化）
        self.fc_obs = nn.Linear(obs_dim, hidden_dim)
        self.bn_obs = nn.BatchNorm1d(hidden_dim)
        # 分支3：边界状态特征编码（带批归一化）
        self.fc_boundary_state = nn.Linear(boundary_state_dim, hidden_dim)
        self.bn_boundary_state = nn.BatchNorm1d(hidden_dim)
        # 融合层（3个分支拼接后降维）
        self.fc1 = nn.Linear(hidden_dim * 3, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        # 输出层（标量，用于计算互信息损失）
        self.fc2 = nn.Linear(hidden_dim, 1)
        # 优化器（Adam）
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.float()
        # 拆分输入特征（按维度拆分）
        comm = x[:, :self.comm_dim]
        obs = x[:, self.comm_dim:self.comm_dim + self.obs_dim]
        boundary_state = x[:, self.comm_dim + self.obs_dim:self.comm_dim + self.obs_dim + self.boundary_state_dim]

        # Process each part with BatchNorm （FC→BN→ReLU）
        comm_vec = self.fc_comm(comm)
        comm_vec = f.relu(self.bn_comm(comm_vec))
        obs_vec = self.fc_obs(obs)
        obs_vec = f.relu(self.bn_obs(obs_vec))
        boundary_state_vec = self.fc_boundary_state(boundary_state)
        boundary_state_vec = f.relu(self.bn_boundary_state(boundary_state_vec))

        # 特征拼接（3个分支的高维特征）
        combined = torch.cat((comm_vec, obs_vec, boundary_state_vec), dim=1)
        # 融合层（降维→BN→ReLU）
        x = self.fc1(combined)
        x = f.relu(self.bn1(x))
        # 输出标量（用于计算互信息损失）
        output = self.fc2(x)
        return output

    def inference(self, single_data):
        """
        推理函数：用于训练完成后的推理阶段，输入单个无人机的 12 维特征，输出 PMI 标量值
        :param single_data:
        :return:
        """
        self.eval()# 切换到评估模式（BN/ Dropout固定）
        if isinstance(single_data, np.ndarray):
            single_data = torch.tensor(single_data, dtype=torch.float32)

        if single_data.ndim == 1:
            single_data = single_data.unsqueeze(0)
        output = self.forward(single_data)
        return output.item()  # Extract and return the single scalar value

    def train_pmi(self, config, train_data, n_uav):
        self.train()# 切换到训练模式
        loss_function = CustomLoss()
        # train_data (timesteps*n_uav,12)
        # 数据重塑：(timesteps*n_uav, 12) → (timesteps, n_uav, 12)
        timesteps = train_data.size(0) // n_uav
        train_data = train_data.view(timesteps, n_uav, 12)
        # 随机采样训练样本（b2_size=3000）
        timestep_indices = torch.randint(low=0, high=timesteps, size=(self.b2_size,)) #随机时间步
        uav_indices = torch.randint(low=0, high=n_uav, size=(self.b2_size, 2)) #随机两架无人机
        selected_data = torch.zeros((self.b2_size, 2, 12))
        for i in range(self.b2_size):
            selected_data[i] = train_data[timestep_indices[i], uav_indices[i]]

        avg_loss = 0
        # 按批次训练
        for i in range(self.b2_size // config["pmi"]["batch_size"]):
            self.optimizer.zero_grad()
            batch_data = selected_data[i * config["pmi"]["batch_size"]:(i + 1) * config["pmi"]["batch_size"]]
            # 拆分样本对：第一个无人机/第二个无人机
            input_1_2 = batch_data[:, 0].squeeze(1)
            input_1_3 = batch_data[:, 1].squeeze(1)

            output_1_2 = self.forward(input_1_2)
            output_1_3 = self.forward(input_1_3)
            loss = loss_function(output_1_2, output_1_3)
            avg_loss += abs(loss.item())
            # 反向传播和优化
            loss.backward()
            self.optimizer.step()
        avg_loss /= (self.b2_size // config["pmi"]["batch_size"])
        return avg_loss

    def save(self, save_dir, epoch_i):
        """
        保存模型参数和优化器状态
        :param save_dir:
        :param epoch_i:
        :return:
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, os.path.join(save_dir, "pmi", 'pmi_weights_' + str(epoch_i) + '.pth'))

    def load(self, path):
        """
        加载模型参数和优化器状态
        :param path:
        :return:
        """
        if path and os.path.exists(path):
            checkpoint = torch.load(path)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
