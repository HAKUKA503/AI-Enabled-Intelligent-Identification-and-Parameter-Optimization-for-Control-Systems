import matplotlib as mpl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 读取数据并预处理
data = pd.read_csv('data.csv')
time = data['time'].values
temperature = data['temperature'].values
voltage = data['volte'].values

# 1. 手动估计阶跃时刻（需根据实际数据调整）
step_time = 190  # 手动调整，需通过数据图像估计
mask_initial = time < step_time
mask_steady = time > step_time + 9000  # 确保终值稳态（如阶跃后9000s）

# 最小二乘法估计参数
# 2. 计算稳态值（延长区间，抗噪声）
u0 = np.mean(voltage[mask_initial])
u1 = np.mean(voltage[mask_steady])
Δu = u1 - u0

y0 = np.mean(temperature[mask_initial])
y1 = np.mean(temperature[mask_steady])
Δy = y1 - y0
K = Δy / Δu if Δu != 0 else np.nan  # 防止除零错误

# 3. 提取响应段（相对时间和温度）
mask_response = (time >= step_time) & (time <= step_time + 5000)  # 取响应前5000s
t_rel = time[mask_response] - step_time  # 相对时间（从阶跃时刻开始）
y_rel = temperature[mask_response] - y0  # 相对温度（减去初始稳态值）

# 4. 切线法计算滞后时间 θ（鲁棒性更强）
# 提取初始上升段（前500s）
idx_tangent = np.where(t_rel < 500)[0]
if len(idx_tangent) < 2:
    print("错误：数据点不足，无法计算切线")
    θ = np.nan
else:
    slope, intercept = np.polyfit(t_rel[idx_tangent], y_rel[idx_tangent], 1)
    θ_rel = max(0, -intercept / slope)  # 相对滞后时间（避免负数）
    θ = step_time + θ_rel  # 绝对滞后时间

# 5. 计算时间常数 T（增加容错逻辑）
T = np.nan  # 初始化 T 为 NaN，确保未定义时可捕获错误

# 方法1：63.2%响应法（优先使用）
y63 = 0.632 * Δy
mask_63 = (t_rel >= θ_rel - 10) & (y_rel >= y63)  # 允许10s误差
if np.any(mask_63):
    T_index = np.where(mask_63)[0][0]
    T_rel = t_rel[T_index] - θ_rel
    T = max(0, T_rel)  # 确保 T 非负
else:
    # 方法2：对数拟合法（备用）
    print("警告：未找到63.2%响应点，尝试对数拟合法...")
    valid_mask = (y_rel > 0) & (y_rel < Δy)  # 排除无效数据（y_rel需在0~Δy之间）
    if not np.any(valid_mask):
        print("错误：无有效数据用于对数拟合")
    else:
        t_valid = t_rel[valid_mask]
        delta_y_valid = Δy - y_rel[valid_mask]
        if np.any(delta_y_valid <= 0):
            print("错误：Δy - y_rel 存在非正值，无法计算对数")
        else:
            log_y = np.log(delta_y_valid)
            # 检查数据是否足够拟合
            if len(t_valid) < 2:
                print("错误：数据点不足，无法拟合对数曲线")
            else:
                slope_log, _ = np.polyfit(t_valid, log_y, 1)
                if slope_log == 0:
                    print("错误：对数曲线斜率为0，无法计算 T")
                else:
                    T = max(0, -1 / slope_log)  # 确保 T 非负

# 6. 模型验证（若 T 未定义，提前终止）
if np.isnan(T):
    print("错误：无法计算时间常数 T，请检查数据或调整参数")
else:
    def model_response(t, K, T, θ):
        y = np.zeros_like(t)
        for i, t_val in enumerate(t):
            if t_val > θ:
                y[i] = y0 + K * Δu * (1 - np.exp(-(t_val - θ)/T))
            else:
                y[i] = y0
        return y

    y_model = model_response(time, K, T, θ)

    # 计算模型验证统计量
    residuals = temperature - y_model
    
    # 整体误差统计量
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    mape = np.mean(np.abs(residuals / temperature)) * 100  # 注意：若temperature含0需调整
    r2 = r2_score(temperature, y_model)
    
    # 分段误差统计量（响应阶段更重要）
    mask_evaluation = time >= step_time
    rmse_response = np.sqrt(np.mean(residuals[mask_evaluation]**2))
    mae_response = np.mean(np.abs(residuals[mask_evaluation]))
    mape_response = np.mean(np.abs(residuals[mask_evaluation] / temperature[mask_evaluation])) * 100
    
    # 稳态误差
    steady_state_error = np.abs(y1 - y_model[-1])  # 模型预测的稳态值与实际稳态值的差异
    
    # 绘图（增加残差图）
    plt.figure(figsize=(15, 10))
    
    # 子图1：温度拟合对比
    plt.subplot(2, 2, 1)
    plt.plot(time, temperature, label='实际温度', linewidth=2)
    plt.plot(time, y_model, '--', label=f'模型拟合 (K={K:.2f}, T={T:.0f}, θ={θ:.0f})', linewidth=2)
    plt.axvline(step_time, color='r', linestyle='--', label='阶跃时刻')
    plt.xlabel('时间 / s')
    plt.ylabel('温度 / °C')
    plt.title('实际温度与系统辨识拟合曲线')
    plt.legend()
    plt.grid(True)
    
    # 子图2：残差分析
    plt.subplot(2, 2, 2)
    plt.plot(time, residuals, 'b-', linewidth=0.8, label='残差')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('时间 / s')
    plt.ylabel('残差 / °C')
    plt.title(f'模型残差 (RMSE={rmse:.2f}°C, MAPE={mape:.2f}%)')
    plt.legend()
    plt.grid(True)
    
    # 子图3：残差分布直方图
    plt.subplot(2, 2, 3)
    plt.hist(residuals, bins=30, alpha=0.7, color='b')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('残差 / °C')
    plt.ylabel('频数')
    plt.title('残差分布')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 子图4：实际值 vs 预测值散点图
    plt.subplot(2, 2, 4)
    plt.scatter(temperature, y_model, alpha=0.5, color='b')
    plt.plot([min(temperature), max(temperature)], [min(temperature), max(temperature)], 'r--')
    plt.xlabel('实际温度 / °C')
    plt.ylabel('预测温度 / °C')
    plt.title(f'实际值与预测值比较 (R方={r2:.4f})')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

    # 打印统计量结果
    print(f"\n系统辨识得到参数：K={K:.2f} °C/V, T={T:.0f} s, θ={θ:.0f} s")
    print("\n=== 模型验证统计量 ===")
    print(f"整体误差:")
    print(f"  - RMSE (均方根误差): {rmse:.2f} °C")
    print(f"  - MAE (平均绝对误差): {mae:.2f} °C")
    print(f"  - MAPE (平均绝对百分比误差): {mape:.2f} %")
    print(f"  - R² (决定系数): {r2:.4f}")
    
    print(f"\n响应阶段误差 (t ≥ {step_time}s):")
    print(f"  - RMSE: {rmse_response:.2f} °C")
    print(f"  - MAE: {mae_response:.2f} °C")
    print(f"  - MAPE: {mape_response:.2f} %")
    
    print(f"\n稳态误差:")
    print(f"  - 稳态温度误差: {steady_state_error:.2f} °C")
    
    # 评估模型质量
    print("\n=== 模型质量评估 ===")
    if rmse < 2 and mape < 5:
        print("模型质量优秀：误差小，拟合度高")
    elif rmse < 5 and mape < 10:
        print("模型质量良好：误差可接受，能满足工程需求")
    elif rmse < 10 and mape < 20:
        print("模型质量一般：存在一定误差，需谨慎使用")
    else:
        print("模型质量较差：误差较大，建议优化模型结构或参数")