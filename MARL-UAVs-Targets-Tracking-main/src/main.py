import argparse
import os.path
from src.environment import Environment
from models.actor_critic import ActorCritic
from utils.args_util import get_config
from train import train, evaluate, run
from models.PMINet import PMINetwork
from utils.data_util import save_csv
from utils.draw_util import plot_reward_curve
from models.mappo import MAPPO


def print_config(vdict, name="config"):
    print("-----------------------------------------")
    print("|This is the summary of {}:".format(name))
    var = vdict
    for i in var:
        if var[i] is None:
            continue
        print("|{:11}\t: {}".format(i, var[i]))
    print("-----------------------------------------")


def print_args(args, name="args"):
    print("-----------------------------------------")
    print("|This is the summary of {}:".format(name))
    for arg in vars(args):
        print("| {:<11} : {}".format(arg, getattr(args, arg)))
    print("-----------------------------------------")


def add_args_to_config(config, args):
    for arg in vars(args):
        config[str(arg)] = getattr(args, arg)


def main(args):
    config = get_config(os.path.join("configs", args.method + ".yaml"))
    add_args_to_config(config, args)
    print_config(config)
    print_args(args)

    env = Environment(n_uav=config["environment"]["n_uav"],
                      m_targets=config["environment"]["m_targets"],
                      x_max=config["environment"]["x_max"],
                      y_max=config["environment"]["y_max"],
                      na=config["environment"]["na"],
                      n_obstacles=config["environment"]["n_obstacles"]
                      )

    agent = None
    pmi = None

    if args.method == "MAPPO":
        pmi = PMINetwork(
            hidden_dim=config["pmi"]["hidden_dim"],
            b2_size=config["pmi"]["b2_size"]
        )
        if args.pmi_path is not None:
            pmi.load(args.pmi_path)

        agent = MAPPO(
            state_dim=16,  # 直接使用 16 维
            action_dim=config["environment"]["na"],
            hidden_dim=config["actor_critic"]["hidden_dim"],
            actor_lr=float(config["actor_critic"]["actor_lr"]),
            critic_lr=float(config["actor_critic"]["critic_lr"]),
            gamma=float(config["actor_critic"]["gamma"]),
            gae_lambda=config["actor_critic"]["gae_lambda"],
            clip_epsilon=config["actor_critic"]["clip_epsilon"],
            device=config["devices"][0]
        )

    elif args.method == "C-METHOD":
        agent = None

    else:
        agent = ActorCritic(state_dim=16,
                            hidden_dim=config["actor_critic"]["hidden_dim"],
                            action_dim=config["environment"]["na"],
                            actor_lr=float(config["actor_critic"]["actor_lr"]),
                            critic_lr=float(config["actor_critic"]["critic_lr"]),
                            gamma=float(config["actor_critic"]["gamma"]),
                            device=config["devices"][0])
        agent.load(args.actor_path, args.critic_path)

        if args.method == "MAAC":
            pmi = None
            config["cooperative"] = 0
        elif args.method == "MAAC-R" or args.method == "MAAC-G":
            pmi = PMINetwork(hidden_dim=config["pmi"]["hidden_dim"],
                             b2_size=config["pmi"]["b2_size"])
            pmi.load(args.pmi_path)

    if args.phase == "train":
        return_list = train(config=config,
                            env=env,
                            agent=agent,
                            pmi=pmi,
                            num_episodes=args.num_episodes,
                            num_steps=args.num_steps,
                            frequency=args.frequency)
    elif args.phase == "evaluate":
        return_list = evaluate(config=config,
                               env=env,
                               agent=agent,
                               pmi=pmi,
                               num_steps=args.num_steps)
    elif args.phase == "run":
        return_list = run(config=config,
                          env=env,
                          pmi=pmi,
                          num_steps=args.num_steps)
    else:
        return

    save_csv(config, return_list)

    plot_reward_curve(config, return_list['return_list'], "overall_return")
    plot_reward_curve(config, return_list["target_tracking_return_list"],
                      "target_tracking_return_list")
    plot_reward_curve(config, return_list["boundary_punishment_return_list"],
                      "boundary_punishment_return_list")
    plot_reward_curve(config, return_list["duplicate_tracking_punishment_return_list"],
                      "duplicate_tracking_punishment_return_list")
    plot_reward_curve(config, return_list["average_covered_targets_list"],
                      "average_covered_targets_list")
    plot_reward_curve(config, return_list["max_covered_targets_list"],
                      "max_covered_targets_list")


    plot_reward_curve(config, return_list["obstacle_punishment_return_list"], "obstacle_punishment_return_list")
    plot_reward_curve(config, return_list["average_covered_obstacles_list"], "average_covered_obstacles_list")
    plot_reward_curve(config, return_list["max_covered_obstacles_list"], "max_covered_obstacles_list")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--phase", type=str, default="train", choices=["train", "evaluate", "run"])
    parser.add_argument("-e", "--num_episodes", type=int, default=5000)
    parser.add_argument("-s", "--num_steps", type=int, default=200)
    parser.add_argument("-f", "--frequency", type=int, default=100)
    parser.add_argument("-a", "--actor_path", type=str, default=None)
    parser.add_argument("-c", "--critic_path", type=str, default=None)
    parser.add_argument("-p", "--pmi_path", type=str, default=None)
    parser.add_argument("-m", "--method", default="MAAC-R", choices=["MAAC", "MAAC-R", "MAAC-G", "C-METHOD", "MAPPO"])
    main_args = parser.parse_args()
    main(main_args)