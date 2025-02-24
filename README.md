# TD3-BoilerSystem  
🔥 基于数字孪生的锅炉系统强化学习控制优化

## 📖 概述  
通过 **TD3/DDPG 强化学习算法**与 **XGBoost（贝叶斯优化）** 结合的框架，实现工业锅炉系统的智能控制：  
- 降低燃气消耗  
- 提升钢坯温度场分布的均匀性  
- 支持数字孪生环境下的动态仿真  

---

## 🛠️ 环境依赖  
- **Python 3.8+**  
- 核心库：  
  ```bash
  torch>=2.0.0          # PyTorch 最新版
  xgboost>=1.7.0        # 多输出XGBoost模型
  scikit-learn>=1.2.0   # 数据标准化
  gym>=0.26.0           # 环境接口
  ```

---

## 🚀 快速开始  
1. **克隆仓库**：  
   ```bash  
   git clone https://github.com/yourname/TD3-BoilerSystem.git  
   ```  

2. **安装依赖**：  
   ```bash  
   pip install -r requirements.txt  
   ```  

3. **训练TD3模型**（默认使用预配置数字孪生环境）：
使用TD3train.py脚本即可 

5. **测试预训练模型**：  
使用runtest_td3.py脚本即可

---

## ⚙️ 算法对比  
| 算法 | 优势 | 适用场景 |  
|------|------|----------|  
| **TD3** | 双Q网络解决高估问题，延迟策略更新 | 高维状态空间、复杂动态环境 |  
| **DDPG** | 简单直接，训练速度快 | 基础连续控制任务 |  

### 训练奖励值
![奖励值](https://github.com/user-attachments/assets/6259ec60-11a9-45ff-bd76-f0da509345aa)

### 温度方差对比
<img width="750" alt="936cb0266c1c01005d308598cc02587" src="https://github.com/user-attachments/assets/a75eb827-be21-43bb-a917-9c072ee1ac3c" />

### 煤气用量对比
<img width="750" alt="09820bf29ebb7f2ac0052f00b824c86" src="https://github.com/user-attachments/assets/b44f8ace-1226-4d77-b5e4-28182103d01e" />

---

## 📂 文件结构  
```
TD3-BoilerSystem/
├── models/                 # 预训练模型权重（TD3/DDPG）
├── ddpg_tensorboard/       # DDPG训练指标日志
├── td3_tensorboard/        # TD3训练指标日志
├── xgb_multi_output_model.pkl  # XGBoost数字孪生环境核心模型
├── MyData.py               # 内置数据接口（已集成标准化逻辑）
├── scaler_X.pkl            # 输入特征标准化器
├── TD3train.py             # TD3训练入口（优先使用）
└── runtest_td3.py          # TD3策略可视化测试
```

---

## 📊 典型结果  
![目标特征重要性](https://github.com/user-attachments/assets/6fc2938c-6007-465b-89c0-035b32924483)
![状态回归](https://github.com/user-attachments/assets/08993deb-6661-47e7-a797-35026d2d39e7)


---

## 📜 许可证  
**Unlicense** - 可自由修改、商用、无归属要求  

