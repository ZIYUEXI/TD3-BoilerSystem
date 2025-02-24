import random  # 导入随机数生成模块，用于生成随机数或执行随机操作
import numpy as np  # 导入NumPy模块，用于高效的数组和矩阵操作
import pandas as pd  # 导入Pandas模块，用于数据操作与分析
from model2.xgbModel import ModelPredictor  # 从自定义模块中导入ModelPredictor类，用于模型预测
from model2.MyData import MetaData  # 从自定义模块中导入MetaData类，用于管理或加载元数据
from pyecharts.charts import Line  # 导入pyecharts的Line类，用于创建折线图
from pyecharts import options as opts  # 导入pyecharts的选项模块，用于设置图表的样式和选项

def generate_metadata(num_instances):
    """
    生成指定数量的元数据实例。

    参数:
        num_instances (int): 要生成的元数据实例数量。

    返回:
        list: 包含生成的 MetaData 对象的列表。
    """
    # 初始化元数据列表，用于存储生成的 MetaData 对象
    metadata_list = []

    # 固定参数值
    fixed_nozzle_angle = 0  # 喷嘴角度固定为0
    fixed_preheat_nozzle_count = 10  # 预热喷嘴数量固定为10
    fixed_heating_nozzle_count = 36  # 加热喷嘴数量固定为36
    fixed_equalizing_nozzle_count = 10  # 均衡喷嘴数量固定为10

    # 参数范围，用于生成随机值
    air_fuel_ratio_range = (0.3, 1.0)  # 空燃比范围
    gas_co_composition_range = (27.0, 31.0)  # 气体中CO成分的百分比范围
    first_stage_gas_velocity_range = (26.0, 26.6)  # 第一阶段气体速度范围
    second_stage_gas_velocity_range = (27.0, 27.6)  # 第二阶段气体速度范围
    third_stage_gas_velocity_range = (28.0, 28.6)  # 第三阶段气体速度范围
    air_fuel_preheat_temp_range = (950.0, 1030.0)  # 空燃预热温度范围（单位: °C）

    # 生成指定数量的元数据实例
    for _ in range(num_instances):
        # 在指定范围内生成随机参数值，并进行精度格式化
        air_fuel_ratio = round(random.uniform(*air_fuel_ratio_range), 4)  # 随机生成空燃比，保留4位小数
        gas_co_composition = round(random.uniform(*gas_co_composition_range), 2)  # 随机生成CO成分，保留2位小数
        first_stage_gas_velocity = round(random.uniform(*first_stage_gas_velocity_range), 3)  # 第一阶段气体速度，保留3位小数
        second_stage_gas_velocity = round(random.uniform(*second_stage_gas_velocity_range), 3)  # 第二阶段气体速度
        third_stage_gas_velocity = round(random.uniform(*third_stage_gas_velocity_range), 3)  # 第三阶段气体速度
        air_fuel_preheat_temperature = round(random.uniform(*air_fuel_preheat_temp_range), 1)  # 空燃预热温度，保留1位小数

        # 创建一个 MetaData 对象并填充数据
        data = MetaData(
            nozzle_angle=fixed_nozzle_angle,  # 固定喷嘴角度
            first_stage_gas_velocity=first_stage_gas_velocity,  # 第一阶段气体速度
            second_stage_gas_velocity=second_stage_gas_velocity,  # 第二阶段气体速度
            third_stage_gas_velocity=third_stage_gas_velocity,  # 第三阶段气体速度
            gas_co_composition=gas_co_composition,  # CO成分百分比
            air_fuel_ratio=air_fuel_ratio,  # 空燃比
            air_fuel_preheat_temperature=air_fuel_preheat_temperature,  # 空燃预热温度
            preheat_nozzle_count=fixed_preheat_nozzle_count,  # 预热喷嘴数量
            heating_nozzle_count=fixed_heating_nozzle_count,  # 加热喷嘴数量
            equalizing_nozzle_count=fixed_equalizing_nozzle_count  # 均衡喷嘴数量
        )

        # 将生成的 MetaData 对象添加到列表中
        metadata_list.append(data)

    # 返回生成的元数据列表
    return metadata_list


def main():
    predictor = ModelPredictor()
    num_instances = 8200
    metadata_list = generate_metadata(num_instances)
    temperatures_max = []
    temperatures_min = []
    temperatures_std = []
    temperatures = []
    gas_consumptions = []
    i = 0
    for data in metadata_list:
        try:
            # 从 data 对象中提取数据，并创建一个 NumPy 数组，用作模型预测的输入
            sample_array = np.array([data.use_data()])

            # 使用预测器 predictor 的 predict 方法进行预测，并传入当前数据和步骤 i
            prediction = predictor.predict(sample_array, i)

            # 从预测结果中提取所需的温度和煤气用量相关信息
            temperature_max = prediction.get('温度最大值')  # 获取温度最大值
            temperature_min = prediction.get('温度最小值')  # 获取温度最小值
            temperature_std = prediction.get('温度方差')  # 获取温度的标准差

            # 检查温度相关数据是否完整，如果缺失则抛出异常
            if temperature_max is None or temperature_min is None:
                raise ValueError("预测结果中缺少温度数据")

            # 从预测结果中获取煤气用量信息
            gas_consumption = prediction.get('煤气用量(立方米每秒）')

            # 打印当前步骤的温度标准差和煤气用量
            print(f"Step {i}: 温度方差={temperature_std}, 煤气用量={gas_consumption}")

            # 如果煤气用量数据缺失，抛出异常
            if gas_consumption is None:
                raise ValueError("预测结果中缺少煤气用量数据")

            # 将预测的温度最大值、最小值、标准差分别添加到对应的列表中
            temperatures_max.append(temperature_max)
            temperatures_min.append(temperature_min)
            temperatures_std.append(temperature_std)

            # 计算平均温度（最大值与最小值的平均值），并添加到温度列表中
            temperatures.append((temperature_max + temperature_min) / 2)

            # 将煤气用量添加到煤气用量列表中
            gas_consumptions.append(gas_consumption)

            i = i + 1
        except Exception as e:
            print(f"预测时发生错误: {e}")
    data_df = pd.DataFrame({
        'Temperature': temperatures,
        'TemperatureMax':temperatures_max,
        'TemperatureMin': temperatures_min,
        'GasConsumption': gas_consumptions
    })
    if not data_df.empty:
        temp_stats = data_df['Temperature'].agg(['max', 'min', 'mean', 'var', 'std'])
        gas_stats = data_df['GasConsumption'].agg(['max', 'min', 'mean', 'var', 'std'])
        print("整体温度统计:")
        print(f"  整体最大温度: {temp_stats['max']}")
        print(f"  整体最小温度: {temp_stats['min']}")
        print(f"  整体平均温度: {temp_stats['mean']:.2f}")
        print(f"  整体温度方差: {temp_stats['var']:.2f}")
        print(f"  整体温度标准差: {temp_stats['std']:.2f}")
        print("煤气用量统计:")
        print(f"  最大煤气用量: {gas_stats['max']} 立方米每秒")
        print(f"  最小煤气用量: {gas_stats['min']} 立方米每秒")
        print(f"  平均煤气用量: {gas_stats['mean']:.2f} 立方米每秒")
        print(f"  煤气用量方差: {gas_stats['var']:.2f}")
        print(f"  煤气用量标准差: {gas_stats['std']:.2f}")
        print("出炉温度统计")
        print(f"  最大温度: {temperatures_max[-1]}")
        print(f"  最小温度: {temperatures_min[-1]}")
        print(f"  温度方差: {temperatures_std[-1]}")
    else:
        print("没有有效的数据用于统计。")
if __name__ == "__main__":
    main()