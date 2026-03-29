import csv
import os.path
import numpy as np

#主要用于数据处理，实验结果保存和奖励数值预处理
def save_csv(config, return_list):
    """
     保存训练过程中的各类奖励/惩罚数据到CSV文件
    :param config:
    :param return_list:
        return_list = {
        'return_list': self.return_list, 总奖励列表
        'target_tracking_return_list' :target_tracking_return_list, 跟踪奖励列表
        'boundary_punishment_return_list':boundary_punishment_return_list, 边界惩罚
        'duplicate_tracking_punishment_return_list':duplicate_tracking_punishment_return_list 重复追踪惩罚
    }
    :return:
    """

    #对奖励和惩罚进行分别的保存操作
    with open(os.path.join(config["save_dir"], 'return_list.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Reward'])  # 写入表头
        for reward in return_list['return_list']:
            writer.writerow([reward])

    with open(os.path.join(config["save_dir"], 'target_tracking_return_list.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['target_tracking'])  # 写入表头
        for reward in return_list['target_tracking_return_list']:
            writer.writerow([reward])

    with open(os.path.join(config["save_dir"], 'boundary_punishment_return_list.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['boundary_punishment'])  # 写入表头
        for reward in return_list['boundary_punishment_return_list']:
            writer.writerow([reward])

    with open(os.path.join(config["save_dir"], 'duplicate_tracking_punishment_return_list.csv'), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['duplicate_tracking_punishment'])  # 写入表头
        for reward in return_list['duplicate_tracking_punishment_return_list']:
            writer.writerow([reward])


def clip_and_normalize(val, floor, ceil, choice=1):
    """
    裁剪并归一化
    :param val: 原始值
    :param floor:  最小值
    :param ceil: 最大值
    :param choice: 归一化模式
    :return:
    """
    if val < floor or val > ceil:
        val = max(val, floor)
        val = min(val, ceil)
        print("overstep in clip.")
    val = np.clip(val, floor, ceil)
    mid = (floor + ceil) / 2
    if choice == -1:
        val = (val - floor) / (ceil - floor) - 1  # (-1, 0)
    elif choice == 0:
        val = (val - floor) / (ceil - floor)  # (0, 1)
    else:
        val = (val - mid) / (mid - floor)  # (-1, 1)
    return val
