import argparse
import os.path
from src.environment import Environment
from models.actor_critic import ActorCritic
from utils.args_util import get_config
from train import train, evaluate, run
from models.PMINet import PMINetwork
from utils.data_util import save_csv
from utils.draw_util import plot_reward_curve


def print_config(vdict, name="config"):
    """
    辅助打印函数，作用是格式化打印配置文件，在运行时能查看当前配置
    :param vdict: dict, 待打印的字典
    :param name: str, 打印的字典名称
    :return: None
    """
    print("-----------------------------------------")
    print("|This is the summary of {}:".format(name))
    var = vdict
    for i in var:
        if var[i] is None:
            continue
        print("|{:11}\t: {}".format(i, var[i]))
    print("-----------------------------------------")



def print_args(args, name="args"):
    """
    格式化打印命令行参数
    :param args:
    :param name: str, 打印的字典名称
    :return: None
    """
    print("-----------------------------------------")
    print("|This is the summary of {}:".format(name))
    for arg in vars(args):
        print("| {:<11} : {}".format(arg, getattr(args, arg)))
    print("-----------------------------------------")

#融合配置文件参数和命令行参数，所有的参数统一存在config字典中
def add_args_to_config(config, args):
    for arg in vars(args):
        # print("| {:<11} : {}".format(arg, getattr(args, arg)))
        config[str(arg)] = getattr(args, arg)


def main(args):
    # 获取方法所用的参数
    config = get_config(os.path.join("configs", args.method + ".yaml"))
    add_args_to_config(config, args)
    print_config(config)
    print_args(args)

    # 初始化environment, agent
    env = Environment(n_uav=config["environment"]["n_uav"], #无人机数量
                      m_targets=config["environment"]["m_targets"], #目标数量
                      x_max=config["environment"]["x_max"], #环境x轴的最大范围
                      y_max=config["environment"]["y_max"], #y轴最大范围
                      na=config["environment"]["na"]) #动作维度-无人机的动作数，本文将动作设置为离散变量
    #初始化智能体，Actor-Critic，Actor负责输出无人机的动作，Critic负责评估动作的好坏
    if args.method == "C-METHOD": #如果实验采用对比方法，不初始化Actor-Critic
        agent = None
    else:
        agent = ActorCritic(state_dim=12, #状态维度（无人机位置+速度+目标位置等）
                            hidden_dim=config["actor_critic"]["hidden_dim"], #隐藏层维度
                            action_dim=config["environment"]["na"], #动作维度
                            actor_lr=float(config["actor_critic"]["actor_lr"]), #Actor学习率
                            critic_lr=float(config["actor_critic"]["critic_lr"]), #Critic学习率
                            gamma=float(config["actor_critic"]["gamma"]), #折扣因子，权衡即时奖励和未来奖励
                            device=config["devices"][0])  # 只用第一个device
        agent.load(args.actor_path, args.critic_path) #加载预训练的网络权重
    # 初始化 pmi ，pmi网络用于让无人机感知队友的状态，计算互惠奖励
    if args.method == "MAAC" or args.method == "MAAC-G" or args.method == "C-METHOD":
        pmi = None #
        if args.method == "MAAC":
            config["cooperative"] = 0  # MAAC只考虑无人机自己的奖励
    elif args.method == "MAAC-R":
        pmi = PMINetwork(hidden_dim=config["pmi"]["hidden_dim"],
                         b2_size=config["pmi"]["b2_size"])
        pmi.load(args.pmi_path)#加载pmi网络预训练权重
    else:
        return
#训练阶段
    if args.phase == "train":
        return_list = train(config=config,
                            env=env,
                            agent=agent,
                            pmi=pmi,
                            num_episodes=args.num_episodes,
                            num_steps=args.num_steps,
                            frequency=args.frequency)
#评估阶段，用训练好的模型测试效果
    elif args.phase == "evaluate":
        return_list = evaluate(config=config,
                               env=env,
                               agent=agent,
                               pmi=pmi,
                               num_steps=args.num_steps)
#仅运行，可视化追踪过程，不进行训练
    elif args.phase == "run":
        return_list = run(config=config,
                          env=env,
                          pmi=pmi,
                          num_steps=args.num_steps)
    else:
        return
#保存结果
    save_csv(config, return_list)
#绘制各种曲线
    plot_reward_curve(config, return_list['return_list'], "overall_return") #总奖励曲线
    plot_reward_curve(config, return_list["target_tracking_return_list"], #目标追踪奖励曲线
                      "target_tracking_return_list")
    plot_reward_curve(config, return_list["boundary_punishment_return_list"], #越界惩罚曲线
                      "boundary_punishment_return_list")
    plot_reward_curve(config, return_list["duplicate_tracking_punishment_return_list"], #重复追踪惩罚曲线
                      "duplicate_tracking_punishment_return_list")
    plot_reward_curve(config, return_list["average_covered_targets_list"],#平均覆盖目标数量曲线
                      "average_covered_targets_list")
    plot_reward_curve(config, return_list["max_covered_targets_list"],#最大覆盖目标数量曲线
                      "max_covered_targets_list")


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="")

    # 添加超参数
    parser.add_argument("--phase", type=str, default="train", choices=["train", "evaluate", "run"])
    parser.add_argument("-e", "--num_episodes", type=int, default=10000, help="训练轮数")
    parser.add_argument("-s", "--num_steps", type=int, default=200, help="每轮进行步数")
    parser.add_argument("-f", "--frequency", type=int, default=100, help="打印信息及保存的频率")
    parser.add_argument("-a", "--actor_path", type=str, default=None, help="actor网络权重的路径")
    parser.add_argument("-c", "--critic_path", type=str, default=None, help="critic网络权重的路径")
    parser.add_argument("-p", "--pmi_path", type=str, default=None, help="pmi网络权重的路径")
    parser.add_argument("-m", "--method", help="", default="MAAC-R", choices=["MAAC", "MAAC-R", "MAAC-G", "C-METHOD"])
    # 解析命令行参数
    main_args = parser.parse_args()

    # 调用主函数
    main(main_args)

#调用python main.py --phase train -m MAAC -e 5000 -s 300进行训练，训练5000轮 每个轮次300步

#python main.py --phase train -m MAAC-G -f 50 使用MAAC-G进行训练，50轮打印一次
#从预训练的权重继续训练
#python main.py --phase train -m MAAC-R -a ./checkpoints/actor_1000.pth -c ./checkpoints/critic_1000.pth -p ./checkpoints/pmi_1000.pth
