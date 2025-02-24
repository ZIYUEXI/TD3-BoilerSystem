class MetaData:
    def __init__(self, nozzle_angle, first_stage_gas_velocity, second_stage_gas_velocity,
                 third_stage_gas_velocity, gas_co_composition, air_fuel_ratio,
                 air_fuel_preheat_temperature, preheat_nozzle_count,
                 heating_nozzle_count, equalizing_nozzle_count):
        # 喷嘴角度（不变值）
        self.nozzle_angle = nozzle_angle

        # 第一段煤气速度
        self.first_stage_gas_velocity = first_stage_gas_velocity

        # 第二段煤气速度
        self.second_stage_gas_velocity = second_stage_gas_velocity

        # 第三段煤气速度
        self.third_stage_gas_velocity = third_stage_gas_velocity

        # 燃气CO成分
        self.gas_co_composition = gas_co_composition

        # 空燃比
        self.air_fuel_ratio = air_fuel_ratio

        # 空燃气预热温度
        self.air_fuel_preheat_temperature = air_fuel_preheat_temperature

        # 预热喷嘴数量（不变值）
        self.preheat_nozzle_count = preheat_nozzle_count

        # 加热喷嘴数量（不变值）
        self.heating_nozzle_count = heating_nozzle_count

        # 均热喷嘴数量（不变值）
        self.equalizing_nozzle_count = equalizing_nozzle_count

    def use_data(self):
        return [
            self.nozzle_angle,
            self.first_stage_gas_velocity,
            self.second_stage_gas_velocity,
            self.third_stage_gas_velocity,
            self.gas_co_composition,
            self.air_fuel_ratio,
            self.air_fuel_preheat_temperature,
            self.preheat_nozzle_count,
            self.heating_nozzle_count,
            self.equalizing_nozzle_count
        ]



