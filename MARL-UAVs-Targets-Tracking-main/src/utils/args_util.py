import os
import random
import time
import yaml
import numpy as np
import torch

#项目的初始化入口
def get_config(config_file):
    """
    解析配置文件-》设置种子-》创建实验目录-》配置设备-》保存配置
    :param config_file: str, 超参数所在的文件路径
    :return: dict, 解析后的超参数字典
    """
    #读取并解析yaml文件
    with open(config_file, 'r', encoding="UTF-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # 设置全局随机种子。保证每次运行实验的随机数一致，实验结果可复现。
    if 'seed' in config and config['seed'] is not None:
        np.random.seed(config['seed'])
        random.seed(config['seed'])
        torch.manual_seed(config['seed'])

    # 针对每次实验生成本实验的唯一实验名称
    run_id = str(os.getpid()) #获取进程ID
    exp_name = '_'.join([
        config['exp_name'],
        time.strftime('%Y-%b-%d-%H-%M-%S'), run_id
    ])

    #定义保存路径
    save_dir = os.path.join(config['result_dir'], exp_name)
    args_save_name = os.path.join(save_dir, 'args.yaml')
    config['save_dir'] = save_dir

    # 创建所有需要的文件夹
    mkdir(config['result_dir']) #所有结果的根目录
    mkdir(save_dir) #本次实验目录
    mkdir(os.path.join(save_dir, "actor")) #存储Actor模型
    mkdir(os.path.join(save_dir, "critic")) #存储Critic模型
    mkdir(os.path.join(save_dir, "pmi")) #保存PMI网络模型
    mkdir(os.path.join(save_dir, "animated")) #动画
    mkdir(os.path.join(save_dir, "t_xy")) #目标位置
    mkdir(os.path.join(save_dir, "u_xy")) #无人机位置
    mkdir(os.path.join(save_dir, "covered_target_num")) #覆盖目标数

    # create cuda devices
    set_device(config)
    # 将配置文件保存到实验目录（方便后续复现）
    with open(args_save_name, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return config


def mkdir(folder):
    """
    递归创建文件夹
    :param folder:
    :return:
    """
    if not os.path.isdir(folder):
        os.makedirs(folder)


def set_device(config):
    """
    配置运行设备
    :param config: dict
    :return: None
    """
    #使用cpu时，gpus = -1
    if config['gpus'] == -1 or not torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = "" #清空cuda可见设备
        print('use cpu')
        config['devices'] = [torch.device('cpu')]
    else:
        #设置cuda可见设备
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(i) for i in range(config['first_device'],
                                                                            config['first_device'] + config['gpus']))
        print('use gpus: {}'.format(config['gpus']))
        config['devices'] = [torch.device('cuda', i) for i in range(config['first_device'],
                                                                    config['first_device'] + config['gpus'])]


if __name__ == "__main__":
    example = get_config("../configs/MAAC.yaml") #加载配置文件
    print(type(example))
    print(example) #输出解析后的配置字典
