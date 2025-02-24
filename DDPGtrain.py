from stable_baselines3 import DDPG
# 从Stable-Baselines3库中导入深度确定性策略梯度（DDPG）算法，用于强化学习训练。

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
# 导入回调函数：CheckpointCallback用于定期保存模型，EvalCallback用于评估，BaseCallback是回调的基类。

from stable_baselines3.common.noise import NormalActionNoise
# 导入NormalActionNoise，用于在DDPG训练时为动作添加噪声，以促进探索。

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
# DummyVecEnv用于将环境包装成多进程向量化环境的形式，VecNormalize用于对环境观测值和奖励进行标准化处理。

from MyData import MetaData
# 导入自定义模块`MyData`中的`MetaData`类或方法，可能包含环境或模型相关的元数据。

from xgbModel import ModelPredictor
# 导入自定义模块`xgbModel`中的`ModelPredictor`类或方法，可能用于模型预测或结合强化学习的应用。

import gymnasium as gym
# 导入Gymnasium库，这是一个强化学习环境的标准化工具包。

from gymnasium import spaces
# 从Gymnasium库中导入`spaces`模块，用于定义动作空间和状态空间。

import numpy as np
# 导入NumPy库，用于高效的数组和数值计算。


fixed_nozzle_angle = 0
# 固定喷嘴角度，值为0，可能表示喷嘴为垂直方向或水平无偏角。

fixed_preheat_nozzle_count = 10
# 预热喷嘴的固定数量，设置为10个，用于初始预热阶段的燃料喷射。

fixed_heating_nozzle_count = 36
# 加热喷嘴的固定数量，设置为36个，用于燃烧加热阶段。

fixed_equalizing_nozzle_count = 10
# 均衡喷嘴的固定数量，设置为10个，用于燃烧均衡阶段或温度分布均匀化。

air_fuel_ratio_range = (0.3, 1.0)
# 空燃比范围，表示空气和燃料的混合比例范围在0.3到1.0之间。

gas_co_composition_range = (27.0, 31.0)
# 燃气中一氧化碳（CO）组成的百分比范围，在27.0%到31.0%之间。

first_stage_gas_velocity_range = (26.0, 26.6)
# 第一阶段燃气速度范围，单位可能是m/s，范围在26.0到26.6之间。

second_stage_gas_velocity_range = (27.0, 27.6)
# 第二阶段燃气速度范围，单位可能是m/s，范围在27.0到27.6之间。

third_stage_gas_velocity_range = (28.0, 28.6)
# 第三阶段燃气速度范围，单位可能是m/s，范围在28.0到28.6之间。

air_fuel_preheat_temp_range = (950.0, 1030.0)
# 空气和燃料预热温度范围，单位可能是摄氏度，范围在950.0到1030.0之间。



class CustomEnv(gym.Env):
    def __init__(self):
        # 初始化自定义强化学习环境。
        super(CustomEnv, self).__init__()
        # 调用父类的初始化方法，确保继承自基类的初始化行为。

        self.max_steps = 8200
        # 定义环境的最大步数限制，用于终止条件。

        self.current_observation = None
        # 当前观察值的占位变量，初始化为None。

        self.action_space = spaces.Box(
            low=np.array([
                air_fuel_ratio_range[0],  # 空燃比的最小值
                gas_co_composition_range[0],  # 燃气中CO成分的最小值
                first_stage_gas_velocity_range[0],  # 第一阶段燃气速度的最小值
                second_stage_gas_velocity_range[0],  # 第二阶段燃气速度的最小值
                third_stage_gas_velocity_range[0],  # 第三阶段燃气速度的最小值
                air_fuel_preheat_temp_range[0]  # 空气和燃料预热温度的最小值
            ], dtype=np.float32),
            high=np.array([
                air_fuel_ratio_range[1],  # 空燃比的最大值
                gas_co_composition_range[1],  # 燃气中CO成分的最大值
                first_stage_gas_velocity_range[1],  # 第一阶段燃气速度的最大值
                second_stage_gas_velocity_range[1],  # 第二阶段燃气速度的最大值
                third_stage_gas_velocity_range[1],  # 第三阶段燃气速度的最大值
                air_fuel_preheat_temp_range[1]  # 空气和燃料预热温度的最大值
            ], dtype=np.float32),
            dtype=np.float32
        )
        # 定义动作空间，表示智能体可以采取的动作范围，包括空燃比、燃气成分、各阶段燃气速度和预热温度。

        self.observation_space = spaces.Box(
            low=np.array([0.0, 200.0, 0.0], dtype=np.float32),
            high=np.array([30.0, 560.0, self.max_steps], dtype=np.float32),
            dtype=np.float32
        )

        self.step_count = 0
        # 初始化步数计数器，用于跟踪当前的环境步数。

        self.model = ModelPredictor()
        # 初始化自定义的预测模型`ModelPredictor`，可能用于辅助环境状态预测或奖励计算。

    def reset(self, seed=None, options=None):
        # 重置环境到初始状态，通常在强化学习训练的每个回合开始时调用。

        print('环境重置')
        # 输出提示信息，表明环境正在重置。

        super().reset(seed=seed)
        # 调用父类的reset方法，确保环境基础设置被正确初始化，同时支持设置随机种子（`seed`）。

        self.step_count = 0
        # 将步数计数器重置为0，表示新回合的开始。

        self.current_observation = np.array([
            np.random.uniform(5.0, 10.0),  # 第一个状态值，初始化为5.0到10.0之间的随机值，可能是某种浓度。
            np.random.uniform(400.0, 500.0),  # 第二个状态值，初始化为400.0到500.0之间的随机值，可能是温度或压力。
            self.step_count  # 第三个状态值，表示步数，初始化为0。
        ], dtype=np.float32)
        # 初始化观察值，定义了新的环境状态。

        return self.current_observation, {}

    def step(self, action):
        # 执行一步操作，根据动作更新环境状态，计算奖励，并返回下一状态及其他信息。

        action = np.clip(action, self.action_space.low, self.action_space.high)
        # 将传入的动作限制在定义的动作空间范围内，防止越界。

        air_fuel_ratio, gas_co_composition, vel_1, vel_2, vel_3, preheat_temp = action
        # 解包动作，提取空燃比、燃气成分、各阶段燃气速度和预热温度的值。

        metadata = MetaData(
            nozzle_angle=fixed_nozzle_angle,  # 固定喷嘴角度
            first_stage_gas_velocity=vel_1,  # 第一阶段燃气速度
            second_stage_gas_velocity=vel_2,  # 第二阶段燃气速度
            third_stage_gas_velocity=vel_3,  # 第三阶段燃气速度
            gas_co_composition=gas_co_composition,  # 燃气中CO成分
            air_fuel_ratio=air_fuel_ratio,  # 空燃比
            air_fuel_preheat_temperature=preheat_temp,  # 空气和燃料预热温度
            preheat_nozzle_count=fixed_preheat_nozzle_count,  # 预热喷嘴数量
            heating_nozzle_count=fixed_heating_nozzle_count,  # 加热喷嘴数量
            equalizing_nozzle_count=fixed_equalizing_nozzle_count  # 均衡喷嘴数量
        )
        # 创建一个`MetaData`对象，用于存储当前动作的元数据。

        sample_array = np.array([metadata.use_data()])
        # 调用`MetaData`的方法`use_data`生成样本数据，并转换为数组。

        model_output = self.model.predict(sample_array, self.step_count)
        # 使用预测模型`ModelPredictor`预测当前状态的输出值。

        temperature_std = model_output.get('温度方差')
        # 从模型输出中获取温度方差。

        gas_consumption = model_output.get('煤气用量(立方米每秒）')
        # 从模型输出中获取煤气消耗量。

        self.step_count += 1
        # 增加步数计数器。

        self.current_observation = np.array([
            temperature_std,  # 更新观察值中的温度方差
            gas_consumption,  # 更新观察值中的煤气消耗量
            self.step_count  # 当前步数
        ], dtype=np.float32)
        # 更新当前观察值。

        reward = -np.power(temperature_std / 100, 2) - np.power(gas_consumption / 10.0, 2)
        # 根据温度方差和煤气消耗量计算奖励，奖励为负值，表示偏差越小越好。

        done = (self.step_count >= self.max_steps)
        # 检查是否达到最大步数，从而结束回合。

        if done:
            reward = -np.power(temperature_std / 1, 2) - np.power(gas_consumption / 1, 2)
            # 如果回合结束，更新奖励公式。

        if self.step_count % 1000 == 0:
            print(f"Step {self.step_count}: 温度方差={temperature_std}, 煤气用量={gas_consumption}, 奖励={reward}")
            # 每1000步打印状态信息，便于监控训练过程。

        truncated = False
        # 指定是否因某些约束导致的非自然中止（未使用）。

        info = {
            "step_count": self.step_count,  # 当前步数
            "action_taken": action  # 当前采取的动作
        }
        # 额外信息字典，用于记录步数和动作。

        return self.current_observation, reward, done, truncated, info
        # 返回新的观察值、奖励值、是否结束、是否截断标志以及额外信息。


class SaveVecNormalizeCallback(BaseCallback):
    # 自定义回调类，用于定期保存VecNormalize环境的统计数据。

    def __init__(self, save_freq: int, save_path: str, verbose=0):
        # 初始化回调对象。
        # 参数：
        # - save_freq: 保存频率，表示每隔多少次调用保存一次统计数据。
        # - save_path: 保存路径，指定统计数据文件保存的位置。
        # - verbose: 冗长级别，0表示不打印信息，大于0时会打印日志信息。

        super(SaveVecNormalizeCallback, self).__init__(verbose)
        # 调用BaseCallback的初始化方法。

        self.save_freq = save_freq
        # 保存频率。

        self.save_path = save_path
        # 保存路径。

    def _on_step(self) -> bool:
        # 每次步数更新时执行的逻辑。
        if self.n_calls % self.save_freq == 0:
            # 如果当前调用次数能够被保存频率整除，则执行保存操作。

            self.training_env.save(self.save_path)
            # 调用训练环境的`save`方法，将VecNormalize的统计数据保存到指定路径。

            if self.verbose > 0:
                # 如果verbose参数大于0，则打印保存成功的日志信息。
                print(f"Saved VecNormalize stats to {self.save_path}")

        return True
        # 返回True，表示回调函数未中断训练过程。


def main():
    # 主函数，设置强化学习环境并开始训练。

    train_env = CustomEnv()
    # 创建自定义训练环境实例。

    vec_train_env = DummyVecEnv([lambda: train_env])
    # 将训练环境包装为DummyVecEnv，用于支持向量化处理。

    vec_train_env = VecNormalize(vec_train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    # 对训练环境进行归一化处理：
    # - norm_obs: 对观察值进行归一化。
    # - norm_reward: 对奖励值进行归一化。
    # - clip_obs: 限制观察值的范围为[-10.0, 10.0]。

    eval_env = CustomEnv()
    # 创建自定义评估环境实例。

    vec_eval_env = DummyVecEnv([lambda: eval_env])
    # 将评估环境包装为DummyVecEnv。

    try:
        vec_eval_env = VecNormalize.load("./vec_normalize2.pkl", vec_eval_env)
        # 尝试加载之前保存的归一化统计数据。
    except FileNotFoundError:
        vec_eval_env = VecNormalize(vec_eval_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        # 如果统计数据文件不存在，则创建新的归一化对象。

    vec_eval_env.training = False
    # 设置评估环境为非训练模式。

    vec_eval_env.norm_reward = False
    # 关闭评估环境的奖励归一化。

    n_actions = train_env.action_space.shape[-1]
    # 获取动作空间的维度（动作数量）。
    print(f"Number of actions: {n_actions}")
    # 打印动作数量。

    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))
    # 定义动作噪声，用于促进DDPG算法的探索。
    # mean: 噪声均值，初始化为零向量。
    # sigma: 噪声标准差，设置为0.2的常量数组。

    model = DDPG(
        policy="MlpPolicy",  # 使用多层感知器(Multi-layer Perceptron)策略。
        env=vec_train_env,  # 指定训练环境。
        action_noise=action_noise,  # 设置动作噪声。
        verbose=1,  # 冗长级别，1表示打印训练日志。
        learning_rate=1e-3,  # 学习率。
        buffer_size=1000000,  # 经验回放缓冲区大小。
        batch_size=256,  # 每次训练的批量大小。
        tau=0.003,  # 目标网络软更新系数。
        gamma=0.99,  # 折扣因子。
        device="cuda",  # 使用GPU进行训练。
        tensorboard_log="./ddpg_tensorboard/"  # TensorBoard日志路径。
    )
    # 初始化DDPG算法模型。

    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./models/', name_prefix='ddpg_model')
    # 创建检查点回调，用于每10000步保存模型到指定路径。

    eval_callback = EvalCallback(
        eval_env=vec_eval_env,  # 指定评估环境。
        best_model_save_path='./logs/',  # 最佳模型保存路径。
        log_path='./logs/',  # 日志路径。
        eval_freq=10000,  # 评估频率，每10000步进行一次评估。
        deterministic=True,  # 评估时使用确定性策略。
        render=False,  # 评估时不渲染环境。
        callback_on_new_best=None,  # 无额外回调。
        verbose=1  # 冗长级别。
    )
    # 创建评估回调，用于保存最佳模型和记录评估结果。

    save_vec_normalize_callback = SaveVecNormalizeCallback(save_freq=10000, save_path="./vec_normalize.pkl", verbose=1)
    # 创建自定义回调，用于定期保存VecNormalize的统计数据。

    model.learn(
        total_timesteps=10000000000,  # 总训练时间步数。
        callback=[checkpoint_callback, eval_callback, save_vec_normalize_callback],
        # 设置回调列表。
        tb_log_name="DDPG_run_2"  # TensorBoard日志名称。
    )
    # 开始模型训练。

    obs, _ = train_env.reset()
    # 重置训练环境，获取初始观察值。


if __name__ == "__main__":
    main()