import csv
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from model2.DDPGtrain import CustomEnv


# 定义一个函数，用于评估已经训练好的 DDPG 模型（不使用向量归一化的环境）
def evaluate_trained_model_no_normalization(num_episodes=20):
    # 定义动作标签，对应实际控制变量的名称
    ACTION_LABELS = [
        "空燃比",
        "煤气中 CO 含量 (%)",
        "一级煤气流速 (m/s)",
        "二级煤气流速 (m/s)",
        "三级煤气流速 (m/s)",
        "空气预热温度 (℃)"
    ]
    # 定义观察标签，对应状态变量的名称
    OBS_LABELS = [
        "温度方差",
        "煤气用量 (立方米/秒)"
    ]

    # 创建评估环境，使用 DummyVecEnv 封装 CustomEnv 以兼容 Stable-Baselines3 的模型
    eval_env = DummyVecEnv([lambda: CustomEnv()])
    # 加载训练好的 DDPG 模型，并绑定到评估环境
    model = DDPG.load("./modelstorge/best_model.zip", env=eval_env)
    print("Loaded trained DDPG model (no VecNormalize).")

    # 打开一个 CSV 文件，用于记录评估日志
    with open("evaluation_log_ddpg.csv", mode="w", newline="", encoding="utf-8") as csvfile:
        # 定义 CSV 文件的字段名称，包括观察值、动作值、奖励等
        fieldnames = (
                ["Episode", "Step"] +
                [f"Obs_before_{label}" for label in OBS_LABELS] +
                [f"Action_{label}" for label in ACTION_LABELS] +
                [f"Obs_after_{label}" for label in OBS_LABELS] +
                ["Reward", "Done"]
        )
        # 创建 CSV 写入器
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        # 写入表头
        writer.writeheader()

        # 逐个评估指定数量的回合
        for episode in range(num_episodes):
            # 重置环境，获取初始观察值
            obs = eval_env.reset()
            done = [False]  # 标志当前回合是否结束
            step_count = 0  # 记录当前回合的步数
            episode_reward = 0.0  # 累积奖励初始化

            # 当回合未结束时，执行以下操作
            while not done[0]:
                step_count += 1  # 步数加一
                # 使用模型预测动作，设置为确定性模式
                action, _states = model.predict(obs, deterministic=True)

                # 准备一行记录，存储当前步的信息
                row_data = {
                    "Episode": episode + 1,
                    "Step": step_count
                }
                # 添加当前观察值（动作之前）
                for label, value in zip(OBS_LABELS, obs[0]):
                    row_data[f"Obs_before_{label}"] = value
                # 添加动作值
                for label, value in zip(ACTION_LABELS, action[0]):
                    row_data[f"Action_{label}"] = value

                # 在环境中执行动作，获取新的状态、奖励、完成标志和额外信息
                obs_new, reward, done, info = eval_env.step(action)
                # 添加新的观察值（动作之后）
                for label, value in zip(OBS_LABELS, obs_new[0]):
                    row_data[f"Obs_after_{label}"] = value
                # 添加奖励和回合完成标志
                row_data["Reward"] = reward[0]
                row_data["Done"] = done[0]
                # 将当前记录写入 CSV 文件
                writer.writerow(row_data)

                # 累积当前奖励
                episode_reward += reward[0]
                # 更新当前观察值
                obs = obs_new

            # 输出当前回合的评估结果
            print(f"Episode {episode + 1} finished | Total Steps: {step_count}, "
                  f"Episode Reward: {episode_reward:.4f}")


# 程序入口，调用评估函数并指定评估回合数
if __name__ == "__main__":
    evaluate_trained_model_no_normalization(num_episodes=20)
    print("All evaluation logs have been written to 'evaluation_log_ddpg.csv'.")
