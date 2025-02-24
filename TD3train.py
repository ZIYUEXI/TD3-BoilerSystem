from stable_baselines3 import TD3
# 从 stable-baselines3 库中导入 TD3 算法，用于训练基于深度强化学习的智能体。

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
# 导入稳定基线中的回调函数类：
# - CheckpointCallback：用于保存模型的回调。
# - EvalCallback：用于评估模型表现的回调。
# - BaseCallback：所有自定义回调的基类。

from stable_baselines3.common.noise import NormalActionNoise
# 导入 NormalActionNoise，用于添加噪声到智能体的动作中，通常在训练连续动作空间时使用。

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
# 导入矢量化环境工具：
# - DummyVecEnv：将单个环境包装成支持并行的环境（尽管只有一个环境）。
# - VecNormalize：用于对环境观测值和奖励进行归一化处理。

from MyData import MetaData
# 从自定义模块 MyData 中导入 MetaData，可能包含与强化学习相关的元数据。

from xgbModel import ModelPredictor
# 从自定义模块 xgbModel 中导入 ModelPredictor，可能是一个基于 XGBoost 的模型预测器，用于辅助强化学习。

import gymnasium as gym
# 导入 gymnasium 库，创建和管理强化学习的仿真环境。

from gymnasium import spaces
# 从 gymnasium 中导入 spaces 模块，用于定义环境的动作和观察空间。

import numpy as np
# 导入 numpy 库，用于数值计算和数组操作。


fixed_nozzle_angle = 5
# 固定喷嘴的角度，单位为度（degrees），用于控制喷嘴的喷射方向。

fixed_preheat_nozzle_count = 16
# 预热喷嘴的固定数量，用于预热阶段的喷嘴数量。

fixed_heating_nozzle_count = 36
# 加热喷嘴的固定数量，用于主加热阶段的喷嘴数量。

fixed_equalizing_nozzle_count = 11
# 均热喷嘴的固定数量，用于均热阶段的喷嘴数量。

air_fuel_ratio_range = (0.3, 1.0)
# 空气与燃料的比值范围（Air-Fuel Ratio），用于控制燃烧混合比。
# 最小值为 0.3，最大值为 1.0。

gas_co_composition_range = (27.0, 31.0)
# 燃气中一氧化碳（CO）成分的百分比范围，用于定义燃气成分变化的范围。

first_stage_gas_velocity_range = (26.0, 26.6)
# 第一阶段燃气速度的范围，用于控制燃气流速。

second_stage_gas_velocity_range = (27.0, 27.6)
# 第二阶段燃气速度的范围。

third_stage_gas_velocity_range = (28.0, 28.6)
# 第三阶段燃气速度的范围。

air_fuel_preheat_temp_range = (950.0, 1030.0)
# 空气和燃料的预热温度范围，用于确保燃烧前的空气和燃料达到适当的预热温度。


class CustomEnv(gym.Env):
    def __init__(self):
        # 初始化自定义环境的构造函数。

        super(CustomEnv, self).__init__()
        # 调用父类的初始化方法，确保自定义环境继承了父类的功能。

        self.max_steps = 8200
        # 定义环境的最大步数限制，表示每一轮交互中最多允许执行的步数。

        self.current_observation = None
        # 当前的观测值（state），在环境初始化时设置为 None。

        self.action_space = spaces.Box(
            low=np.array([
                air_fuel_ratio_range[0],
                gas_co_composition_range[0],
                first_stage_gas_velocity_range[0],
                second_stage_gas_velocity_range[0],
                third_stage_gas_velocity_range[0],
                air_fuel_preheat_temp_range[0]
            ], dtype=np.float32),
            high=np.array([
                air_fuel_ratio_range[1],
                gas_co_composition_range[1],
                first_stage_gas_velocity_range[1],
                second_stage_gas_velocity_range[1],
                third_stage_gas_velocity_range[1],
                air_fuel_preheat_temp_range[1]
            ], dtype=np.float32),
            dtype=np.float32
        )
        # 定义动作空间（action space），即智能体可采取的动作范围。
        # 使用 gymnasium 的 Box 类来定义一个连续的动作空间，其中每个维度有上下限。
        # 动作空间的维度：
        # 1. 空气燃料比。
        # 2. 燃气中一氧化碳成分。
        # 3. 第一阶段燃气速度。
        # 4. 第二阶段燃气速度。
        # 5. 第三阶段燃气速度。
        # 6. 空气燃料的预热温度。

        self.observation_space = spaces.Box(
            low=np.array([0.0, 200.0, 0.0], dtype=np.float32),
            high=np.array([30.0, 560.0, self.max_steps], dtype=np.float32),
            dtype=np.float32
        )
        # 定义观测空间（observation space），即智能体接收到的环境状态。
        # 使用 gymnasium 的 Box 类来定义一个连续的观测空间。
        # 观测空间的维度：
        # 1. 某些环境变量的最小值（如燃烧相关的工艺参数）。
        # 2. 温度范围。
        # 3. 当前步数范围，最大值为 self.max_steps。

        self.step_count = 0
        # 初始化步数计数器，用于跟踪当前轮次中执行的步数。

        self.model = ModelPredictor()
        # 初始化一个自定义的模型预测器实例（可能基于 XGBoost），用于辅助智能体的决策。

    def reset(self, seed=None, options=None):
        # 重置环境到初始状态。
        # 参数：
        # - seed：可选参数，用于设置随机数种子，确保结果的可复现性。
        # - options：可选参数，用于传递其他选项信息（在此未使用）。

        print('环境重置')
        # 打印日志，表示环境正在被重置。

        super().reset(seed=seed)
        # 调用父类的 reset 方法，确保环境的基本功能被正确初始化，并应用随机种子。

        self.step_count = 0
        # 重置步数计数器，表示新一轮交互从第 0 步开始。

        self.current_observation = np.array([
            np.random.uniform(5.0, 10.0),
            # 随机生成观测值的第一个维度，范围为 [5.0, 10.0]，可能代表某些工艺参数。

            np.random.uniform(400.0, 500.0),
            # 随机生成观测值的第二个维度，范围为 [400.0, 500.0]，可能代表温度。

            self.step_count
            # 第三个维度为当前步数，初始值为 0。
        ], dtype=np.float32)
        # 初始化当前的环境观测值，包含三个维度的数据。

        return self.current_observation, {}
        # 返回当前的观测值和额外的信息（此处为空字典）。
        # 根据 gymnasium 的规范，`reset` 方法需要返回初始观测值和其他信息。

    def step(self, action):
        # 执行环境中一步操作。
        # 参数：
        # - action：智能体采取的动作，为一个数组，包含多个控制参数。

        action = np.clip(action, self.action_space.low, self.action_space.high)
        # 将动作限制在动作空间范围内，确保动作的每个维度都在定义的上下限之间。

        air_fuel_ratio, gas_co_composition, vel_1, vel_2, vel_3, preheat_temp = action
        # 将动作解包为具体的参数：
        # - air_fuel_ratio：空气燃料比。
        # - gas_co_composition：燃气中 CO 的比例。
        # - vel_1, vel_2, vel_3：第一、二、三阶段的燃气速度。
        # - preheat_temp：空气燃料的预热温度。

        metadata = MetaData(
            nozzle_angle=fixed_nozzle_angle,
            first_stage_gas_velocity=vel_1,
            second_stage_gas_velocity=vel_2,
            third_stage_gas_velocity=vel_3,
            gas_co_composition=gas_co_composition,
            air_fuel_ratio=air_fuel_ratio,
            air_fuel_preheat_temperature=preheat_temp,
            preheat_nozzle_count=fixed_preheat_nozzle_count,
            heating_nozzle_count=fixed_heating_nozzle_count,
            equalizing_nozzle_count=fixed_equalizing_nozzle_count
        )
        # 创建一个 MetaData 对象，包含当前动作参数和固定的喷嘴配置。
        # MetaData 的作用是将这些参数封装，用于后续的模型输入或计算。

        sample_array = np.array([metadata.use_data()])
        # 从 MetaData 中提取数据，将其封装为数组形式，用于输入模型预测。

        model_output = self.model.predict(sample_array, self.step_count)
        # 调用模型预测器，根据当前步数和输入数据进行预测。
        # `model_output` 是一个字典，包含预测结果。

        temperature_std = model_output.get('温度方差')
        # 从模型输出中获取温度方差，表示温度控制的稳定性。

        gas_consumption = model_output.get('煤气用量(立方米每秒）')
        # 从模型输出中获取煤气用量，表示资源消耗。

        self.step_count += 1
        # 更新步数计数器，表示已经执行了一步操作。

        self.current_observation = np.array([
            temperature_std,
            gas_consumption,
            self.step_count
        ], dtype=np.float32)
        # 更新当前观测值，包括温度方差、煤气用量和当前步数。

        reward = -np.power(temperature_std / 100, 2) - np.power(gas_consumption / 10.0, 2)
        # 根据温度方差和煤气用量计算奖励：
        # - 奖励的目标是最小化温度波动（稳定性）和资源消耗。

        done = (self.step_count >= self.max_steps)
        # 检查是否达到最大步数限制，决定本轮交互是否结束。

        if done:
            temp_penalty = np.power(temperature_std, 2)
            gas_penalty = np.power(gas_consumption, 2)
            reward = -(10.0 * temp_penalty + 1.0 * gas_penalty)
            # 如果交互结束，计算额外的惩罚，特别是对温度和资源消耗的严重波动进行加权惩罚。

        if self.step_count % 1000 == 0:
            print(f"Step {self.step_count}: 温度方差={temperature_std}, 煤气用量={gas_consumption}, 奖励={reward}")
            # 每 1000 步打印一次日志信息，便于跟踪训练过程中的关键数据。

        truncated = False
        # 是否截断，当前未启用此功能，默认为 False。

        info = {
            "step_count": self.step_count,
            "action_taken": action
        }
        # 附加信息，包括当前步数和采取的动作，便于调试或记录。

        return self.current_observation, reward, done, truncated, info
        # 返回值：
        # - 当前观测值。
        # - 奖励值。
        # - 是否完成。
        # - 是否截断（当前为 False）。
        # - 附加信息字典。


class SaveVecNormalizeCallback(BaseCallback):
    # 自定义回调类，继承自 stable-baselines3 的 BaseCallback。
    # 此回调用于定期保存 VecNormalize 的统计信息。

    def __init__(self, save_freq: int, save_path: str, verbose=0):
        # 初始化回调。
        # 参数：
        # - save_freq：保存频率（步数间隔），表示每隔多少步保存一次。
        # - save_path：保存路径，用于存储 VecNormalize 的统计信息。
        # - verbose：是否输出日志，0 表示无输出，>0 表示有输出。

        super(SaveVecNormalizeCallback, self).__init__(verbose)
        # 调用父类的初始化方法，确保继承基础功能。

        self.save_freq = save_freq
        # 保存频率，控制保存间隔。

        self.save_path = save_path
        # 保存路径，指定保存文件的位置。

    def _on_step(self) -> bool:
        # 回调方法，每一步训练后调用。
        # 返回值：
        # - True：继续训练。
        # - False：中断训练。

        if self.n_calls % self.save_freq == 0:
            # 检查当前步数是否满足保存条件（当前调用次数 n_calls 是否是 save_freq 的倍数）。

            self.training_env.save(self.save_path)
            # 调用训练环境的 save 方法，将 VecNormalize 的统计信息保存到指定路径。

            if self.verbose > 0:
                # 如果 verbose 参数大于 0，则打印保存日志。
                print(f"Saved VecNormalize stats to {self.save_path}")

        return True
        # 返回 True，表示训练可以继续。


def main():
    # 主函数，定义整个训练过程，包括环境设置、模型配置、训练与回调。

    train_env = CustomEnv()
    # 初始化自定义训练环境。

    vec_train_env = DummyVecEnv([lambda: train_env])
    # 使用 DummyVecEnv 包装训练环境，将其矢量化，以支持 stable-baselines3 的兼容性。

    vec_train_env = VecNormalize(vec_train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    # 对训练环境应用 VecNormalize，归一化观测值和奖励，并设置观测值的裁剪范围为 [-10, 10]。

    eval_env = CustomEnv()
    # 初始化自定义评估环境。

    vec_eval_env = DummyVecEnv([lambda: eval_env])
    # 使用 DummyVecEnv 包装评估环境。

    try:
        vec_eval_env = VecNormalize.load("./vec_normalize2.pkl", vec_eval_env)
        # 尝试加载之前保存的 VecNormalize 状态文件。
    except FileNotFoundError:
        vec_eval_env = VecNormalize(vec_eval_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        # 如果状态文件不存在，则重新初始化 VecNormalize。

    vec_eval_env.training = False
    # 设置评估环境为非训练模式（防止更新统计信息）。

    vec_eval_env.norm_reward = False
    # 关闭奖励归一化，评估时使用原始奖励值。

    n_actions = train_env.action_space.shape[-1]
    # 获取动作空间的维度，即智能体可以控制的参数数量。

    print(f"Number of actions: {n_actions}")
    # 打印动作维度信息。

    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))
    # 设置动作噪声，使用均值为 0、标准差为 0.2 的高斯噪声，帮助智能体在训练中探索。

    model = TD3(
        policy="MlpPolicy",
        env=vec_train_env,
        action_noise=action_noise,
        verbose=1,
        learning_rate=1e-3,
        buffer_size=1000000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        policy_delay=2,
        device="cuda",
        tensorboard_log="./td3_tensorboard/"
    )
    # 初始化 TD3 模型，参数配置包括：
    # - 使用多层感知器（MLP）作为策略网络。
    # - 指定训练环境。
    # - 设置动作噪声以增强探索。
    # - 学习率、缓冲区大小、批量大小、折扣因子（gamma）等超参数。
    # - 设置在 GPU 上运行（device="cuda"）。
    # - 配置 TensorBoard 日志目录。

    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./models/', name_prefix='td3_model')
    # 定义检查点回调，每隔 10000 步保存一次模型到指定路径。

    eval_callback = EvalCallback(
        eval_env=vec_eval_env,
        best_model_save_path='./logs/',
        log_path='./logs/',
        eval_freq=10000,
        deterministic=True,
        render=False,
        callback_on_new_best=None,
        verbose=1
    )
    # 定义评估回调，每隔 10000 步使用评估环境测试模型表现，并保存表现最好的模型。

    save_vec_normalize_callback = SaveVecNormalizeCallback(save_freq=10000, save_path="./vec_normalize.pkl", verbose=1)
    # 定义自定义回调，用于每隔 10000 步保存 VecNormalize 的统计信息。

    model.learn(
        total_timesteps=10000000000,
        callback=[checkpoint_callback, eval_callback, save_vec_normalize_callback],
        tb_log_name="TD3_run_1"
    )
    # 开始训练模型：
    # - 训练步数设置为 100 亿（可根据实际需求调整）。
    # - 注册回调函数列表，包含模型检查点、评估和 VecNormalize 保存。
    # - 设置 TensorBoard 日志名称为 "TD3_run_1"。

    obs, _ = train_env.reset()
    # 重置训练环境，获取初始观测值（此处为后续代码的起点）。


if __name__ == "__main__":
    main()