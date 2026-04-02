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



#替换原先函数
#生成高精度图像

# def resize_image(image_path):
#     """
#     调整图片尺寸为16的倍数，避免视频编码器报错
#     :param image_path:
#     :return:
#     """
#     img = Image.open(image_path).convert('RGB')
#     # Resize the image to be divisible by 16
#     new_width = (img.width // 16) * 16
#     new_height = (img.height // 16) * 16
#     if new_width != img.width or new_height != img.height:
#         img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)  # Updated to use Image.Resampling.LANCZOS
#     return np.array(img)
def resize_image(image_path):
    """
    调整图片尺寸为16的倍数，避免视频编码器报错，同时保持高清
    """
    img = Image.open(image_path).convert('RGB')
    new_width = (img.width // 16) * 16
    new_height = (img.height // 16) * 16
    if new_width != img.width or new_height != img.height:
        # ✅ 用最高质量重采样，不损失清晰度
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
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
#替换原先的视频生成
#生成高精度图像

def draw_animation(config, env, num_steps, ep_num, frames=100):
    plt.switch_backend('Agg')
    # ✅ 论文级高清：画布放大 + 300 DPI（关键！）
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    ax.set_xlim(0, 2000)
    ax.set_ylim(0, 2000)
    ax.set_aspect('equal')
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.tick_params(labelsize=10)

    # 初始化绘图元素
    uav_plots = [ax.scatter([], [], marker='o', color='b', linestyle='None', s=20, alpha=1) for _ in range(env.n_uav)]
    target_plots = [ax.scatter([], [], marker='o', color='r', linestyle='None', s=30, alpha=1) for _ in range(env.m_targets)]
    obstacle_plots = [ax.scatter([], [], marker='s', color='grey', linestyle='None', s=40, alpha=1) for _ in range(env.n_obstacles)]
    uav_search_patches = [patches.Circle((0, 0), uav.dp, color='lightblue', alpha=0.2) for uav in env.uav_list]
    uav_obstacle_patches = [patches.Circle((0, 0), uav.do, color='lightgrey', alpha=0.2) for uav in env.uav_list]

    for patch in uav_search_patches:
        ax.add_patch(patch)
    for patch in uav_obstacle_patches:
        ax.add_patch(patch)

    # ✅ 按 episode 分目录，保留高清帧（不再自动删除）
    save_dir = os.path.join(config["save_dir"], "frames", f"episode_{ep_num + 1}")
    os.makedirs(save_dir, exist_ok=True)
    saved_frames = []
    step_interval = 5  # 每5步一帧，平衡流畅度和文件大小

    for frame in range(0, num_steps, step_interval):
        update(ax, env, uav_plots, target_plots, obstacle_plots, uav_search_patches, frame, frames, num_steps)
        fig.canvas.draw()
        fig.canvas.flush_events()

        # ✅ 论文级保存：高DPI、无白边、无损PNG
        frame_path = os.path.join(save_dir, f'frame_{frame:04d}.png')
        plt.savefig(frame_path, dpi=300, bbox_inches='tight', pad_inches=0.1, format='png')
        saved_frames.append(frame_path)
        plt.draw()

    plt.close(fig)

    # ✅ 生成高清MP4（可选，保留动画）
    video_path = os.path.join(config["save_dir"], "animated", f'animated_plot_{ep_num + 1}.mp4')
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    writer = imageio.get_writer(video_path, fps=10, codec='libx264', format='FFMPEG', pixelformat='yuv420p', quality=10)

    for frame_path in saved_frames:
        if os.path.exists(frame_path):
            img_array = resize_image(frame_path)
            writer.append_data(img_array)
    writer.close()

    # ✅ 【关键修改】注释掉自动删除，保留高清原图！
    # for frame_path in saved_frames:
    #     if os.path.exists(frame_path):
    #         os.remove(frame_path)


#可用版本 但是只能生成视频
# def draw_animation(config, env, num_steps, ep_num, frames=100):
#     plt.switch_backend('Agg')  # 无GUI环境推荐，有GUI可换'TkAgg'
#     #固定画布大小和分辨率
#     fig, ax = plt.subplots(figsize=(6, 6),dpi=60)
#     # ax.set_xlim(-env.x_max / 3, env.x_max / 3 * 4)
#     # ax.set_ylim(-env.y_max / 3, env.y_max / 3 * 4)
#     ax.set_xlim(0, 2000)  # 强制固定x轴范围
#     ax.set_ylim(0, 2000)  # 强制固定y轴范围
#     ax.set_aspect('equal')  # 保持等比例
#     ax.set_xlabel('X Position')
#     ax.set_ylabel('Y Position')
#
#
#     #绘制轨迹、感知圈、文本
#     uav_plots = [ax.scatter([], [], marker='o', color='b', linestyle='None', s=2,alpha=1) for _ in range(env.n_uav)]
#     target_plots = [ax.scatter([], [], marker='o', color='r', linestyle='None', s=3,alpha=1) for _ in range(env.m_targets)]
#     #new
#     obstacle_plots = [ax.scatter([],[],marker='o',color='grey',linestyle = 'None',s=4,alpha=1) for _ in range(env.n_obstacles)]
#     uav_search_patches = [patches.Circle((0, 0), uav.dp, color='lightblue', alpha=0.2) for uav in env.uav_list]
#     uav_search_obstacle = [patches.Circle((0,0), uav.do,color = 'lightgrey',alpha=0.2 ) for uav in env.uav_list]
#
#
#     for patch in uav_search_patches:
#         ax.add_patch(patch)
#     for patch in uav_search_obstacle:
#         ax.add_patch(patch)
#
#     # save_dir = os.path.join(config["save_dir"], "frames")
#     # os.makedirs(save_dir, exist_ok=True)
#     save_dir = os.path.join(config["save_dir"], "frames", f"episode_{ep_num + 1}")
#     os.makedirs(save_dir, exist_ok=True)  # 按episode分目录，避免覆盖
#     saved_frames = []  # 记录保存的帧路径，避免漏帧
#     # Save frames at intervals of 5 num_steps
#     step_interval = 5
#
#     for frame in range(0, num_steps, step_interval):
#         update(ax, env, uav_plots, target_plots,obstacle_plots, uav_search_patches, frame, frames, num_steps)
#         # 核心修复：强制渲染画布（没有这步就是空白！）
#         fig.canvas.draw()  # 触发渲染
#         fig.canvas.flush_events()  # 刷新事件
#
#         # 保存帧（确保路径正确）
#         frame_path = os.path.join(save_dir, f'frame_{frame:04d}.png')
#         plt.savefig(frame_path, bbox_inches=None, pad_inches=0)  # 去除白边
#         saved_frames.append(frame_path)
#         # plt.pause(0.001)  # 确保渲染完成
#         plt.draw()
#         # plt.pause(0.001)  # Pause to ensure the plot updates visibly if needed
#
#     plt.close(fig)
#
#     # Generate MP4
#     video_path = os.path.join(config["save_dir"], "animated", f'animated_plot_{ep_num + 1}.mp4')
#     writer = imageio.get_writer(video_path, fps=5, codec='libx264', format='FFMPEG', pixelformat='yuv420p')
#
#     for frame in range(0, num_steps, step_interval):
#         frame_path = os.path.join(save_dir, f'frame_{frame:04d}.png')
#         if os.path.exists(frame_path):
#             img_array = resize_image(frame_path)
#             writer.append_data(img_array)
#     writer.close()
#
#     # Optionally remove PNG files
#     for frame in range(0, num_steps, step_interval):
#         frame_path = os.path.join(save_dir, f'frame_{frame:04d}.png')
#         if os.path.exists(frame_path):
#             os.remove(frame_path)





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

