import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import imageio
from PIL import Image
import numpy as np
from matplotlib.lines import lineStyles

#绘制各种曲线和可视化操作

# 初始化文本对象为None
text_obj = None


# def update(ax, env, uav_plots, target_plots, uav_search_patches, frame, frames, num_steps, interval=1, paint_all=True):
#     global text_obj
#
#     # 彻底删掉 frame==0 的 return，让所有帧都执行绘图
#     # if frame == 0:
#     #     return
#
#     # 切片改为 0:frame+1:interval，保证每帧都有数据
#     end_idx = frame + 1
#     #更新无人机的位置和感知范围
#     for i, uav in enumerate(env.uav_list):
#         #获取当前帧之前所有无人机的xy坐标
#         uav_x = env.position['all_uav_xs'][0:end_idx:interval]
#         uav_y = env.position['all_uav_ys'][0:end_idx:interval]
#         #空值检查
#         if not uav_x or not uav_y:
#             print(f"Warning: UAV {i} empty at frame {frame}")
#             continue
#         #提取i个无人机的坐标
#         uav_x = [sublist[i] for sublist in uav_x]
#         uav_y = [sublist[i] for sublist in uav_y]
#         #更新无人机散点图
#         if uav_x and uav_y:
#             uav_plots[i].set_offsets(np.column_stack([uav_x, uav_y]))
#             uav_plots[i].set_color('blue')
#             uav_plots[i].set_sizes([20])  # 保证点足够大
#             uav_search_patches[i].center = (uav_x[-1], uav_y[-1])
#     #更新目标位置
#     for i in range(env.m_targets):
#         target_x = env.position['all_target_xs'][0:end_idx:interval]
#         target_y = env.position['all_target_ys'][0:end_idx:interval]
#
#         if not target_x or not target_y:
#             print(f"Warning: Target {i} empty at frame {frame}")
#             continue
#
#         target_x = [sublist[i] for sublist in target_x]
#         target_y = [sublist[i] for sublist in target_y]
#
#         if target_x and target_y:
#             target_plots[i].set_offsets(np.column_stack([target_x, target_y]))
#             target_plots[i].set_color('red')
#             target_plots[i].set_sizes([30])
#
#     # 避免 covered_target_num 越界
#     covered_num = env.covered_target_num[frame] if frame < len(env.covered_target_num) else 0
#     detect_rate = covered_num / env.m_targets * 100 if env.m_targets > 0 else 0.0
#     #更新统计文本
#     text_str = (
#         f"detected target num = {covered_num}\n"
#         f"detected target rate = {detect_rate:.2f}%"
#     )
#
#   #安全清理旧文本
#     if text_obj is not None:
#         try:
#             text_obj.remove()
#         except ValueError:
#             pass  # 防止对象已被销毁时报错
#     #绘制新文本
#     text_obj = ax.text(0.02, 0.98, text_str, transform=ax.transAxes,
#                        fontsize=10, verticalalignment='top', color='black',
#                        bbox=dict(facecolor='white', alpha=0.8))


def resize_image(image_path):
    """
    调整图片尺寸为16的倍数，避免视频编码器报错
    :param image_path:
    :return:
    """
    img = Image.open(image_path).convert('RGB')
    # Resize the image to be divisible by 16
    new_width = (img.width // 16) * 16
    new_height = (img.height // 16) * 16
    if new_width != img.width or new_height != img.height:
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)  # Updated to use Image.Resampling.LANCZOS
    return np.array(img)

def get_gradient_color(start_color, end_color, num_points, idx):
    """
    生成渐变颜色
    :param start_color:
    :param end_color:
    :param num_points:
    :param idx:
    :return:
    """
    start_rgba = np.array(mcolors.to_rgba(start_color))
    end_rgba = np.array(mcolors.to_rgba(end_color))
    ratio = idx / max(1, num_points - 1)
    gradient_rgba = start_rgba + (end_rgba - start_rgba) * ratio
    return mcolors.to_hex(gradient_rgba)

def update(ax, env, uav_plots, target_plots,obstacle_plots, uav_search_patches, frame, frames, num_steps, interval=2, paint_all=True):
    """
    每一帧的更新函数
    :param ax:
    :param env:
    :param uav_plots:
    :param target_plots:
    :param uav_search_patches:
    :param frame:
    :param frames:
    :param num_steps:
    :param interval:
    :param paint_all:
    :return:
    """

    global text_obj

    if frame == 0:
        return
    for i, uav in enumerate(env.uav_list):
        uav_x = env.position['all_uav_xs'][0: frame: interval]
        uav_y = env.position['all_uav_ys'][0: frame: interval]
        uav_x = [sublist[i] for sublist in uav_x]
        uav_y = [sublist[i] for sublist in uav_y]
        if uav_x and uav_y:  # Ensure the lists are not empty
            colors = [get_gradient_color('#E1FFFF', '#0000FF', frame, idx) for idx in range(len(uav_x))]
            #无人机的颜色表示为蓝色渐变
            uav_plots[i].set_offsets(np.column_stack([uav_x, uav_y]))
            uav_plots[i].set_color(colors)
            uav_search_patches[i].center = (uav_x[-1], uav_y[-1])
        else:
            print(f"Warning: UAV {i} position list is empty at frame {frame}.")

    for i in range(env.m_targets):
        target_x = env.position['all_target_xs'][0: frame: interval]
        target_y = env.position['all_target_ys'][0: frame: interval]
        target_x = [sublist[i] for sublist in target_x]
        target_y = [sublist[i] for sublist in target_y]
        if target_x and target_y:  # Ensure the lists are not empty
            colors = [get_gradient_color('#FFC0CB', '#DC143C', frame, idx) for idx in range(len(target_x))]
            #追踪目标的颜色为红色渐变
            target_plots[i].set_offsets(np.column_stack([target_x, target_y]))
            target_plots[i].set_color(colors)
        else:
            print(f"Warning: Target {i} position list is empty at frame {frame}.")

    #新增碰撞物的可视化操作
    #TODO 待完善
    for i in range(env.n_obstacles):
        obstacle_x = env.position['all_obstacle_xs'][0: frame: interval]
        obstacle_y = env.position['all_obstacle_ys'][0: frame: interval]
        obstacle_x = [sublist[i] for sublist in obstacle_x]
        obstacle_y = [sublist[i] for sublist in obstacle_y]
        if obstacle_x and obstacle_y:
            colors = [get_gradient_color('#E0E0E0','#333333',frame,idx) for idx in range(len(obstacle_x))]
            #障碍物表示为灰色渐变色
            obstacle_plots[i].set_offsets(np.column_stack([obstacle_x, obstacle_y]))
            obstacle_plots[i].set_color(colors)
        else:
            print(f"Warning: obstacle {i} position list is empty at frame {frame}.")


    text_str = (
        f"detected target num = {env.covered_target_num[frame]}\n"
        f"detected target rate = {env.covered_target_num[frame] / env.m_targets * 100:.2f}%"
        #new
        f"detected obstacle num = {env.covered_obstacle_num[frame]}\n"
        f"detected obstacle rate = {env.covered_obstacle_num[frame] / env.n_obstacles * 100:.2f}%"
    )

    # 清除之前的文本对象（如果存在）
    if text_obj is not None:
        text_obj.remove()

    # 绘制新的文本对象，没有边框，颜色为深蓝色
    text_obj = ax.text(0.02, 0.98, text_str, transform=ax.transAxes, fontsize=10, verticalalignment='top',
                       color='black')


def draw_animation(config, env, num_steps, ep_num, frames=100):
    plt.switch_backend('Agg')  # 无GUI环境推荐，有GUI可换'TkAgg'
    #固定画布大小和分辨率
    fig, ax = plt.subplots(figsize=(6, 6),dpi=60)
    # ax.set_xlim(-env.x_max / 3, env.x_max / 3 * 4)
    # ax.set_ylim(-env.y_max / 3, env.y_max / 3 * 4)
    ax.set_xlim(0, 2000)  # 强制固定x轴范围
    ax.set_ylim(0, 2000)  # 强制固定y轴范围
    ax.set_aspect('equal')  # 保持等比例
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')


    #绘制轨迹、感知圈、文本
    uav_plots = [ax.scatter([], [], marker='o', color='b', linestyle='None', s=2,alpha=1) for _ in range(env.n_uav)]
    target_plots = [ax.scatter([], [], marker='o', color='r', linestyle='None', s=3,alpha=1) for _ in range(env.m_targets)]
    #new
    obstacle_plots = [ax.scatter([],[],marker='o',color='grey',linestyle = 'None',s=4,alpha=1) for _ in range(env.n_obstacles)]
    uav_search_patches = [patches.Circle((0, 0), uav.dp, color='lightblue', alpha=0.2) for uav in env.uav_list]
    uav_search_obstacle = [patches.Circle((0,0), uav.do,color = 'lightgrey',alpha=0.2 ) for uav in env.uav_list]


    for patch in uav_search_patches:
        ax.add_patch(patch)
    for patch in uav_search_obstacle:
        ax.add_patch(patch)

    # save_dir = os.path.join(config["save_dir"], "frames")
    # os.makedirs(save_dir, exist_ok=True)
    save_dir = os.path.join(config["save_dir"], "frames", f"episode_{ep_num + 1}")
    os.makedirs(save_dir, exist_ok=True)  # 按episode分目录，避免覆盖
    saved_frames = []  # 记录保存的帧路径，避免漏帧
    # Save frames at intervals of 5 num_steps
    step_interval = 5

    for frame in range(0, num_steps, step_interval):
        update(ax, env, uav_plots, target_plots,obstacle_plots, uav_search_patches, frame, frames, num_steps)
        # 核心修复：强制渲染画布（没有这步就是空白！）
        fig.canvas.draw()  # 触发渲染
        fig.canvas.flush_events()  # 刷新事件

        # 保存帧（确保路径正确）
        frame_path = os.path.join(save_dir, f'frame_{frame:04d}.png')
        plt.savefig(frame_path, bbox_inches=None, pad_inches=0)  # 去除白边
        saved_frames.append(frame_path)
        # plt.pause(0.001)  # 确保渲染完成
        plt.draw()
        # plt.pause(0.001)  # Pause to ensure the plot updates visibly if needed

    plt.close(fig)

    # Generate MP4
    video_path = os.path.join(config["save_dir"], "animated", f'animated_plot_{ep_num + 1}.mp4')
    writer = imageio.get_writer(video_path, fps=5, codec='libx264', format='FFMPEG', pixelformat='yuv420p')

    for frame in range(0, num_steps, step_interval):
        frame_path = os.path.join(save_dir, f'frame_{frame:04d}.png')
        if os.path.exists(frame_path):
            img_array = resize_image(frame_path)
            writer.append_data(img_array)
    writer.close()

    # Optionally remove PNG files
    for frame in range(0, num_steps, step_interval):
        frame_path = os.path.join(save_dir, f'frame_{frame:04d}.png')
        if os.path.exists(frame_path):
            os.remove(frame_path)



# def draw_animation(config, env, num_steps, ep_num, frames=100):
#     fig, ax = plt.subplots(figsize=(8, 8))
#     ax.set_xlim(0, env.x_max)
#     ax.set_ylim(0, env.y_max)
#     # 初始化点大小
#     uav_plots = [ax.scatter([], [], marker='o', color='b', s=20, alpha=1) for _ in range(env.n_uav)]
#     target_plots = [ax.scatter([], [], marker='o', color='r', s=30, alpha=1) for _ in range(env.m_targets)]
#     #初始化无人机的感知范围
#     uav_search_patches = [patches.Circle((0, 0), uav.dp, color='lightblue', alpha=0.2) for uav in env.uav_list]
#     for patch in uav_search_patches:
#         ax.add_patch(patch)
#     #创建帧保存目录
#     save_dir = os.path.join(config["save_dir"], "frames")
#     os.makedirs(save_dir, exist_ok=True)
#
#     # step_interval 强制设为 1，保证每帧都有完整数据
#     step_interval = 1
#     for frame in range(0, num_steps, step_interval):
#         update(ax, env, uav_plots, target_plots, uav_search_patches,
#                frame, frames, num_steps, interval=step_interval)
#         plt.draw()
#         plt.savefig(os.path.join(save_dir, f'frame_{frame:04d}.png'), dpi=150)
#         # 每次 save 后重置 scatter，避免累积数据导致空白
#         for plot in uav_plots + target_plots:
#             plot.set_offsets(np.empty((0, 2)))  # 清空点数据，让下一帧重新画
#     plt.close(fig)

def plot_reward_curve(config, return_list, name):
    """
    绘制奖励曲线并存储
    :param config:
    :param return_list:
    :param name:
    :return:
    """
    plt.figure(figsize=(6, 6))
    plt.plot(return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Total Return')
    plt.title(name)
    plt.grid(True)
    plt.savefig(os.path.join(config["save_dir"], name + ".png"))
    # plt.show()

