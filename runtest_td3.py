import numpy as np
import csv
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from model2.TD3train import CustomEnv

# 定义评估函数，用于评估已训练好的 DDPG 模型（未使用 VecNormalize）
def evaluate_trained_model_no_normalization(num_episodes=20):
    # 定义动作标签，表示环境中可以控制的变量
    ACTION_LABELS = [
        "空燃比",                 # 动作1
        "煤气中 CO 含量 (%)",    # 动作2
        "一级煤气流速 (m/s)",    # 动作3
        "二级煤气流速 (m/s)",    # 动作4
        "三级煤气流速 (m/s)",    # 动作5
        "空气预热温度 (℃)"       # 动作6
    ]
    # 定义观察标签，表示环境中可观测的状态变量
    OBS_LABELS = [
        "温度方差",              # 状态1
        "煤气用量 (立方米/秒)"   # 状态2
    ]

    # 创建评估环境，使用 DummyVecEnv 封装以适配稳定基线模型
    eval_env = DummyVecEnv([lambda: CustomEnv()])

    # 加载训练好的 DDPG 模型并绑定到评估环境
    model = DDPG.load("./logs/td3_model_1060000_steps.zip", env=eval_env)
    print("Loaded trained DDPG model (no VecNormalize).")

    # 打开 CSV 文件，用于保存评估日志
    with open("evaluation_log.csv", mode="w", newline="", encoding="utf-8") as csvfile:
        # 定义 CSV 文件的列名，涵盖每一步的详细数据
        fieldnames = (
            ["Episode", "Step"] +
            [f"Obs_before_{label}" for label in OBS_LABELS] +  # 动作前的观察值
            [f"Action_{label}" for label in ACTION_LABELS] +  # 动作值
            [f"Obs_after_{label}" for label in OBS_LABELS] +  # 动作后的观察值
            ["Reward", "Done"]                               # 奖励和回合完成标志
        )
        # 创建 CSV 写入器
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()  # 写入表头

        # 循环评估指定数量的回合
        for episode in range(num_episodes):
            obs = eval_env.reset()  # 重置环境，获取初始观察值
            done = [False]          # 标志当前回合是否完成
            step_count = 0          # 初始化步数
            episode_reward = 0.0    # 初始化累积奖励

            # 在当前回合未结束时，循环执行以下步骤
            while not done[0]:
                step_count += 1  # 增加步数计数
                # 使用模型预测动作，设置为确定性模式
                action, _states = model.predict(obs, deterministic=True)

                # 准备记录当前步的数据
                row_data = {
                    "Episode": episode + 1,  # 当前回合编号
                    "Step": step_count       # 当前步编号
                }
                # 记录动作前的观察值
                for label, value in zip(OBS_LABELS, obs[0]):
                    row_data[f"Obs_before_{label}"] = value
                # 记录动作值
                for label, value in zip(ACTION_LABELS, action[0]):
                    row_data[f"Action_{label}"] = value

                # 在环境中执行动作，获取新状态、奖励、完成标志和附加信息
                obs_new, reward, done, info = eval_env.step(action)
                # 记录动作后的观察值
                for label, value in zip(OBS_LABELS, obs_new[0]):
                    row_data[f"Obs_after_{label}"] = value
                # 记录奖励和回合完成标志
                row_data["Reward"] = reward[0]
                row_data["Done"] = done[0]

                # 将当前步的数据写入 CSV 文件
                writer.writerow(row_data)

                # 累积奖励
                episode_reward += reward[0]
                # 更新观察值
                obs = obs_new

            # 输出当前回合的评估结果
            print(f"Episode {episode + 1} finished | Total Steps: {step_count}, "
                  f"Episode Reward: {episode_reward:.4f}")

# 程序入口，调用评估函数并指定回合数量
if __name__ == "__main__":
    evaluate_trained_model_no_normalization(num_episodes=20)
    print("All evaluation logs have been written to 'evaluation_log.csv'.")
