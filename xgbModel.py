# 导入 Python 内置的序列化工具，支持将 Python 对象保存为二进制文件 (.pkl 格式)
import pickle

# 导入 Joblib 工具，提供高效的模型保存与加载功能，特别适合处理大型对象
import joblib

# 导入 os 模块，用于操作系统功能，如文件路径操作、文件检查等
import os

# 定义 ModelPredictor 类，用于加载模型和标量文件，以及对新样本进行预测和修正
class ModelPredictor:
    def __init__(self,
                 model_path_pickle='xgb_multi_output_model.pkl',  # 默认的模型文件路径（pickle 格式）
                 model_path_joblib='xgb_multi_output_model.joblib',  # 默认的模型文件路径（joblib 格式）
                 scaler_X_path='scaler_X.pkl',  # 默认的输入标量文件路径
                 scaler_Y_path='scaler_Y.pkl'):  # 默认的输出标量文件路径

        # 加载模型（优先加载 pickle 文件，如果不存在则加载 joblib 文件）
        self.model = self._load_model(model_path_pickle, model_path_joblib)
        # 加载输入数据的标准化器
        self.scaler_X = self._load_scaler(scaler_X_path, 'scaler_X')
        # 加载输出数据的标准化器
        self.scaler_Y = self._load_scaler(scaler_Y_path, 'scaler_Y')
        # 定义目标变量名称列表
        self.target_cols = [
            '温度最大值',
            '温度最小值',
            '温度方差',
            '煤气用量(立方米每秒）'
        ]

    # 私有方法，用于加载模型文件
    def _load_model(self, model_path_pickle, model_path_joblib):
        if os.path.exists(model_path_pickle):  # 如果 pickle 文件存在
            with open(model_path_pickle, 'rb') as f:
                model = pickle.load(f)  # 使用 pickle 加载模型
            print(f"成功加载模型文件 '{model_path_pickle}'。")
        elif os.path.exists(model_path_joblib):  # 如果 joblib 文件存在
            model = joblib.load(model_path_joblib)  # 使用 joblib 加载模型
            print(f"成功加载模型文件 '{model_path_joblib}'。")
        else:  # 如果两种文件都不存在，则抛出异常
            raise FileNotFoundError(f"无法找到模型文件 '{model_path_pickle}' 或 '{model_path_joblib}'。")
        return model

    # 私有方法，用于加载标量文件
    def _load_scaler(self, scaler_path, scaler_name):
        if os.path.exists(scaler_path):  # 如果标量文件存在
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)  # 使用 pickle 加载标量
            print(f"成功加载标量文件 '{scaler_path}'。")
        else:  # 如果文件不存在，则抛出异常
            raise FileNotFoundError(f"无法找到文件 '{scaler_path}'。")
        return scaler

    # 公有方法，用于对新样本进行预测
    def predict(self, new_sample, process_time):
        # 对新样本进行标准化处理
        X_test_scaled = self.scaler_X.transform(new_sample)
        # 使用模型进行预测（标准化尺度）
        Y_pred_scaled = self.model.predict(X_test_scaled)
        # 将预测结果还原为原始尺度
        Y_pred = self.scaler_Y.inverse_transform(Y_pred_scaled)
        # 将预测结果转换为目标变量字典
        Q = [{col: row[i] for i, col in enumerate(self.target_cols)} for row in Y_pred][0]
        # 对温度最大值和最小值进行修正
        temperature_max = self._temperature_change(Q['温度最大值'], process_time)
        temperature_min = self._temperature_change(Q['温度最小值'], process_time)
        # 更新修正后的温度值
        Q['温度最大值'] = temperature_max
        Q['温度最小值'] = temperature_min
        return Q

    # 用于修正温度值
    def _temperature_change(self, temperature, process_time, total_instances=8200):
        # 前 60% 的数据点数量
        num_first_phase = int(total_instances * 0.6)
        if process_time < num_first_phase:  # 如果在前 60% 的阶段
            if process_time == 0:  # 如果时间为 0，返回初始温度 20
                return 20
            # 计算当前时间点的权重
            weight = process_time / num_first_phase
            # 根据权重进行修正
            return 20 + (temperature - 20) * weight
        return temperature  # 如果在后 40% 的阶段，直接返回原温度
