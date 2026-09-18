import os
import re
import sys
import yaml

import cv2
from copy import deepcopy

sys.path.append("./");

from envs._GLOBAL_CONFIGS import CONFIGS_PATH;
from collect_data import class_decorator,get_embodiment_config;
from sapien.utils.viewer import Viewer

#该函数目的是根据任务名称和任务配置，返回一个构建好的参数字典 args，用于后续的任务设置和执行。
'''
args = {
    "embodiment": ["aloha"],
    "render_freq": 20,
    "task_name": "task1",
    "left_robot_file": "path/to/left_robot_file",
    "right_robot_file": "path/to/right_robot_file",
    "dual_arm_embodied": True,
    "left_embodiment_config": {...},
    "right_embodiment_config": {...},
    "embodiment_name": "aloha",
    "task_config": "task_config_name",
    "save_path": "path/to/save/task_config_name/task1/aloha"
    }
'''
#字典的查找语法为args["key"]，返回键对应的值
def build_args(task_name, task_config):
    # 路径拼接 得到task_config.yml的完整路径
    config_path = os.path.join(
        CONFIGS_PATH,
        f"{task_config}.yml"
        )

    #读取任务配置文件并将其内容加载到args字典中
    '''
    args={
    "embodiment":["aloha"],
    "render_freq":20
    }
    '''
    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    #字典中增加一个键值对，键为"task_name"，值为传入的task_name参数
    args["task_name"] = task_name

    # 读取 embodiment 配置
    embodiment_type = args["embodiment"]

    embodiment_config_path = os.path.join(
        CONFIGS_PATH,
        "_embodiment_config.yml"
    )

    #读取机器人配置文件并将其内容加载到embodiment_types字典中
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        embodiment_types = yaml.load(
            f.read(),
            Loader=yaml.FullLoader
        )
    #定义内部函数，返回指定名称的机器人的配置文件路径
    def get_embodiment_file(name):
        return embodiment_types[name]["file_path"]

    #判断机器人的数量
    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(
            embodiment_type[0]
        )

        args["right_robot_file"] = get_embodiment_file(
            embodiment_type[0]
        )

        args["dual_arm_embodied"] = True

        embodiment_name = str(embodiment_type[0])
    #双机器人情况，第三个参数是距离
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(
            embodiment_type[0]
        )

        args["right_robot_file"] = get_embodiment_file(
            embodiment_type[1]
        )

        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False

        embodiment_name = (
            str(embodiment_type[0])
            + "+"
            + str(embodiment_type[1])
        )

    else:
        raise ValueError("Invalid embodiment config")

    args["left_embodiment_config"] = get_embodiment_config(
        args["left_robot_file"]
    )

    args["right_embodiment_config"] = get_embodiment_config(
        args["right_robot_file"]
    )

    args["embodiment_name"] = embodiment_name
    args["task_config"] = task_config

    embodiment_dir = re.sub(
        r"[^A-Za-z0-9_]+",
        "_",
        embodiment_name
    ).lower()

    args["save_path"] = os.path.join(
        args["save_path"],
        task_config,
        task_name,
        embodiment_dir,
    )

    return args

def reset_replay(task, initial_actor_states):
    """
    将当前 scene 恢复到 trajectory 开始前，
    用于下一轮循环播放。
    """
    print("正在恢复初始状态...")

    old_render_freq = task.render_freq

    # reset 过程中暂时不要刷新 Viewer
    task.render_freq = 0

    # --------------------------------------------------------
    # 1. 松开夹爪
    # --------------------------------------------------------

    task.together_open_gripper(
        save_freq=None
    )

    # --------------------------------------------------------
    # 2. 机器人回 home
    # --------------------------------------------------------

    task.robot.move_to_homestate()

    # --------------------------------------------------------
    # 3. hammer / block 等恢复初始位置
    # --------------------------------------------------------

    for actor, pose in initial_actor_states:

        actor.set_pose(
            deepcopy(pose)
        )

        # 消除上一轮运动留下来的速度
        if hasattr(actor, "set_velocity"):
            actor.set_velocity(
                [0, 0, 0]
            )

        if hasattr(actor, "set_angular_velocity"):
            actor.set_angular_velocity(
                [0, 0, 0]
            )

    # --------------------------------------------------------
    # 4. trajectory 读取指针回到第0段
    # --------------------------------------------------------

    task.left_cnt = 0
    task.right_cnt = 0

    task.plan_success = True

    # --------------------------------------------------------
    # 5. 让物理场景稳定几步
    # --------------------------------------------------------

    for _ in range(20):
        task.scene.step()

    # 恢复 Viewer
    task.render_freq = old_render_freq

    task._update_render()
    task.viewer.render()

    print("初始状态恢复完成")

def main():
#sys表示读取命令行参数，sys.argv[0]是脚本名称，sys.argv[1]是第一个参数，sys.argv[2]是第二个参数，sys.argv[3]是第三个参数
    if sys.argv[1] == "--help" or len(sys.argv) == 1:
        print(
            "用法：python scripts/replay_trajectory.py "
            "<task_name> <task_config> [episode] [render_freq]"
        )
        return

    task_name = sys.argv[1]
    task_config = sys.argv[2]

    episode_idx = (
        int(sys.argv[3])        #值1
        if len(sys.argv) >= 4   #条件，满足则为值1
        else 0                  #值2
    )

    args = build_args(task_name, task_config)

    task = class_decorator(task_name)

    # 读取这个 episode 对应的 seed , episode表示成功规划的轨迹的索引

    seed_path = os.path.join(
        args["save_path"],
        "seed.txt"
    )

    with open(seed_path, "r") as f:
        seed_list = [
            int(x)
            for x in f.read().split()
        ]

    if episode_idx >= len(seed_list):
        raise IndexError(
            f"episode {episode_idx} 没有对应 seed"
        )

    seed = seed_list[episode_idx]

    print("=" * 70)
    print(f"Task:       {task_name}")
    print(f"Episode:    {episode_idx}")
    print(f"Seed:       {seed}")
    print(f"Trajectory: episode{episode_idx}.pkl")
    print("=" * 70)

    # ------------------------------------
    # Replay 模式
    # ------------------------------------

    args["need_plan"] = False

    # 开启 Viewer
    args["render_freq"] = 0

    # 不生成新的训练数据
    args["save_data"] = False

    print("正在初始化场景...")

    # 使用相同 seed 重建场景
    task.setup_demo(
        now_ep_num=episode_idx,
        seed=seed,
        **args, #展开args字典的键值对进行传递
    )
    print("场景初始化完成")



    print("正在创建 Viewer...")

    task.viewer = Viewer(task.renderer)
    task.viewer.set_scene(task.scene)

    task.viewer.set_camera_xyz(
        x=0.4,
        y=0.22,
        z=1.5,
    )

    task.viewer.set_camera_rpy(
        r=0,
        p=-0.8,
        y=2.45,
    )

    # 从现在开始，每5个仿真step刷新一次Viewer
    render_freq = sys.argv[4] if len(sys.argv) >= 5 else 5
    task.render_freq = int(render_freq)

    # 先主动渲染一帧
    task.scene.update_render()
    task.viewer.render()

    print("Viewer 创建完成")


    # 三个机器人相机窗口
    camera_windows = {
        "head_camera": "RoboTwin - Head Camera",
        "left_camera": "RoboTwin - Left Wrist Camera",
        "right_camera": "RoboTwin - Right Wrist Camera",
    }

    for window_name in camera_windows.values():
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 640, 480)

    # 把三个窗口横向排开
    cv2.moveWindow("RoboTwin - Head Camera", 0, 0)
    cv2.moveWindow("RoboTwin - Left Wrist Camera", 650, 0)
    cv2.moveWindow("RoboTwin - Right Wrist Camera", 1300, 0)

    stop_requested = False

    # 保存 RoboTwin 原来的 _update_render()
    original_update_render = task._update_render


    def update_render_with_cameras():
        nonlocal stop_requested

        # RoboTwin 原本的渲染更新
        original_update_render()

        # 让三个相机真正拍一帧
        task.cameras.update_picture()

        rgb = task.cameras.get_rgb()

        for camera_name, window_name in camera_windows.items():

            if camera_name not in rgb:
                continue

            image = rgb[camera_name]["rgb"]

            # RoboTwin 是 RGB，OpenCV 显示使用 BGR
            image_bgr = cv2.cvtColor(
                image,
                cv2.COLOR_RGB2BGR
            )

            cv2.imshow(
                window_name,
                image_bgr
            )

        # 按 q 退出循环播放
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            stop_requested = True


    # 用我们扩展后的函数替换原函数
    task._update_render = update_render_with_cameras

    # 读取之前保存的 trajectory

    traj_data = task.load_tran_data(
        episode_idx
    )

    args["left_joint_path"] = (
        traj_data["left_joint_path"]
    )

    args["right_joint_path"] = (
        traj_data["right_joint_path"]
    )

    task.set_path_lst(args)

    # 保存 episode 开始时物体的初始状态

    initial_actor_states = []

    for actor in task.scene.get_all_actors():

        name = actor.get_name()

        # 桌子、墙、地面不会移动，不需要恢复
        if name in ["table", "wall", "ground"]:
            continue

        initial_actor_states.append(
            (
                actor,
                deepcopy(actor.get_pose())
            )
        )
    print(
        f"已记录 {len(initial_actor_states)} 个物体的初始状态"
    )

    # ============================================================
    # 循环播放
    # ============================================================

    loop_count = 0

    print()
    print("=" * 70)
    print("开始循环播放 trajectory")
    print("关闭 SAPIEN Viewer 或在相机窗口按 q 退出")
    print("=" * 70)

    try:

        while not task.viewer.closed and not stop_requested:

            loop_count += 1

            # ------------------------------------
            # 播放一次保存好的 trajectory
            # ------------------------------------

            task.play_once()

            if stop_requested or task.viewer.closed:
                break

            print(
                "Task success:",
                task.check_success()
            )

            # ------------------------------------
            # 恢复到初始状态
            # ------------------------------------

            reset_replay(task, initial_actor_states)

    except KeyboardInterrupt:
        pass

    finally:

        cv2.destroyAllWindows()

        if not task.viewer.closed:
            task.viewer.close()

        task.close_env()

        print("\nReplay 已结束")


if __name__ == "__main__":
    main()
