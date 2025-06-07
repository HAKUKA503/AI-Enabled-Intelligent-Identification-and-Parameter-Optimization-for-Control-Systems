import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from scipy.signal import find_peaks
from typing import Tuple, List

# 解决中文显示问题
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 系统参数
SYSTEM = {
    "K": 10,           # 系统增益 (根据实际调整)
    "T": 2902,         # 时间常数 (s)
    "theta": 190,      # 纯滞后时间 (s)
    "room_temp": 16.8, # 室温 (°C)
    "setpoint": 35,    # 设定值 (°C)
    "u_max": 10,       # 执行器最大电压 (V)
    "u_min": 0         # 执行器最小电压 (V)
}

# 遗传算法配置
GA_CONFIG = {
    "pop_size": 100,     # 种群大小
    "max_gen": 20,       # 最大进化代数
    "chrom_len": 3,      # 染色体长度（Kp, Ki, Kd）
    "bounds": [(0.1, 1000),  # Kp范围
               (1e-6, 0.001), # Ki范围
               (0.1, 100)],    # Kd范围
    "cross_rate": 0.8,   # 交叉概率
    "mutate_rate": 0.1,  # 变异概率
    "mutate_scale": 0.2  # 变异幅度
}

# PID控制器类
class PIDController:
    def __init__(self, Kp: float, Ki: float, Kd: float):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0.0
        self.prev_error = 0.0
        self.saturation_count = 0

    def update(self, error: float, dt: float) -> float:
        p_term = self.Kp * error
        if self.Ki != 0:
            self.integral += error * dt
            integral_max = (SYSTEM["u_max"] - p_term) / self.Ki
            integral_min = (SYSTEM["u_min"] - p_term) / self.Ki
            self.integral = np.clip(self.integral, integral_min, integral_max)
        i_term = self.Ki * self.integral
        d_term = self.Kd * (error - self.prev_error) / (dt + 1e-6) if dt > 0 else 0
        self.prev_error = error
        u_unclipped = p_term + i_term + d_term
        u_clipped = np.clip(u_unclipped, SYSTEM["u_min"], SYSTEM["u_max"])
        if u_clipped != u_unclipped:
            self.saturation_count += 1
            if self.saturation_count > 5:
                self.integral -= error * dt * 0.1 * self.saturation_count
        else:
            self.saturation_count = 0
        return u_clipped

# 系统仿真函数
def simulate_system(pid_params: List[float], sim_time: int, dt: float) -> float:
    Kp, Ki, Kd = pid_params
    controller = PIDController(Kp, Ki, Kd)
    time_steps = int(sim_time / dt) + 1
    time = np.linspace(0, sim_time, time_steps)
    temp = np.full(time_steps, SYSTEM["room_temp"])
    delay_steps = max(1, int(SYSTEM["theta"] / dt))
    delay_buffer = np.zeros(delay_steps)

    for i in range(1, time_steps):
        error = SYSTEM["setpoint"] - temp[i-1]
        u = controller.update(error, dt)
        delay_buffer = np.roll(delay_buffer, -1)
        delay_buffer[-1] = u
        u_delayed = delay_buffer[0] if i > delay_steps else 0.0
        temp[i] = temp[i-1] + (dt / SYSTEM["T"]) * (SYSTEM["K"] * u_delayed - (temp[i-1] - SYSTEM["room_temp"]))

    # 适应度函数（ITAE + 超调惩罚 + 上升时间惩罚）
    error = SYSTEM["setpoint"] - temp
    itae = trapezoid(np.abs(error) * time, time)
    max_overshoot = max(0, np.max(temp) - SYSTEM["setpoint"])
    rise_time = time[np.argmax(temp >= SYSTEM["setpoint"])] if np.any(temp >= SYSTEM["setpoint"]) else sim_time
    return itae + 1e4 * max_overshoot + 1e3 * rise_time  # 总适应度（越小越好）

# 遗传算法优化器
class GeneticAlgorithm:
    def __init__(self):
        # 初始化种群（在参数范围内随机生成）
        self.pop = np.array([
            [np.random.uniform(b[0], b[1]) for b in GA_CONFIG["bounds"]]
            for _ in range(GA_CONFIG["pop_size"])
        ])
        self.best_fitness = np.inf
        self.best_chromosome = self.pop[0].copy()
        self.fitness_history = []

    def _fitness(self, chromosome: np.ndarray) -> float:
        # 粗仿真加速评估（步长=10s）
        return simulate_system(chromosome, sim_time=5000, dt=10)

    def _selection(self) -> np.ndarray:
        # 锦标赛选择（每次选2个个体，保留更优的）
        selected = []
        for _ in range(GA_CONFIG["pop_size"]):
            idx1, idx2 = np.random.choice(GA_CONFIG["pop_size"], 2, replace=False)
            selected.append(self.pop[idx1] if self._fitness(self.pop[idx1]) < self._fitness(self.pop[idx2]) else self.pop[idx2])
        return np.array(selected)

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # 算术交叉（生成两个子个体）
        alpha = np.random.rand(GA_CONFIG["chrom_len"])
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2
        return child1, child2

    def _mutation(self, chromosome: np.ndarray) -> np.ndarray:
        # 高斯变异（在参数范围内扰动）
        mutated = chromosome.copy()
        for i in range(GA_CONFIG["chrom_len"]):
            if np.random.rand() < GA_CONFIG["mutate_rate"]:
                scale = (GA_CONFIG["bounds"][i][1] - GA_CONFIG["bounds"][i][0]) * GA_CONFIG["mutate_scale"]
                mutated[i] += np.random.normal(0, scale)
                mutated[i] = np.clip(mutated[i], GA_CONFIG["bounds"][i][0], GA_CONFIG["bounds"][i][1])
        return mutated

    def optimize(self):
        for gen in range(GA_CONFIG["max_gen"]):
            # 评估当前种群适应度
            fitness = np.array([self._fitness(chrom) for chrom in self.pop])
            current_best_idx = np.argmin(fitness)
            current_best_fitness = fitness[current_best_idx]
            current_best_chrom = self.pop[current_best_idx]
            Kp, Ki, Kd = current_best_chrom

            # 更新全局最优
            if current_best_fitness < self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_chromosome = current_best_chrom.copy()
            self.fitness_history.append(self.best_fitness)

            # 选择操作（保留精英个体+锦标赛选择）
            selected_pop = self._selection()
            selected_pop[0] = self.best_chromosome.copy()

            # 交叉操作
            new_pop = []
            for i in range(0, GA_CONFIG["pop_size"], 2):
                parent1 = selected_pop[i]
                parent2 = selected_pop[i+1] if i+1 < GA_CONFIG["pop_size"] else selected_pop[0]
                if np.random.rand() < GA_CONFIG["cross_rate"]:
                    child1, child2 = self._crossover(parent1, parent2)
                    new_pop.extend([child1, child2])
                else:
                    new_pop.extend([parent1, parent2])

            # 变异操作
            new_pop = [self._mutation(chrom) for chrom in new_pop[:GA_CONFIG["pop_size"]]]
            self.pop = np.array(new_pop)

            # 打印当前代最优信息（包含PID参数）
            print(f"代数 {gen+1:2d} | 最优适应度: {self.best_fitness:.2f} | Kp: {Kp:.3f}, Ki: {Ki:.8f}, Kd: {Kd:.3f}")

        return self.best_chromosome, self.best_fitness

# 动态指标计算函数（新增峰值时间、衰减比、振荡次数）
def calculate_metrics(time, temp, setpoint, room_temp):
    delta = setpoint - room_temp
    if delta <= 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, np.inf, 0, room_temp)  # 处理无效delta

    # 上升时间（10%Δ → 90%Δ）
    rise_10 = room_temp + 0.1 * delta
    rise_90 = room_temp + 0.9 * delta
    idx_10 = np.argmax(temp >= rise_10)
    idx_90 = np.argmax(temp >= rise_90)
    rise_time = time[idx_90] - time[idx_10] if idx_90 > idx_10 else time[-1]

    # 峰值时间与峰值
    max_temp = np.max(temp)
    peak_idx = np.argmax(temp)
    peak_time = time[peak_idx]

    # 超调量
    overshoot = max(0, (max_temp - setpoint) / delta * 100) if delta != 0 else 0

    # 调节时间（±2%Δ误差带）
    err_band = 0.02 * delta
    in_band = np.abs(temp - setpoint) <= err_band
    settling_idx = len(time) - 1 if np.all(in_band) else np.where(~in_band)[0][-1] + 1
    settling_time = time[settling_idx] if settling_idx < len(time) else time[-1]

    # 稳态误差
    steady_error = setpoint - temp[-1]

    # 衰减比（至少两个峰值）
    peaks, _ = find_peaks(temp)
    if len(peaks) < 2:
        decay_ratio = np.inf  # 无振荡
    else:
        peak_values = temp[peaks]
        decay_ratio = peak_values[1] / peak_values[0] if peak_values[0] != 0 else np.inf

    # 振荡次数（穿越设定值的次数//2）
    crossings = np.where(np.diff(np.sign(temp - setpoint)))[0]
    oscill_count = len(crossings) // 2

    return rise_time, overshoot, settling_time, steady_error, peak_time, decay_ratio, oscill_count, max_temp

# 主程序
if __name__ == "__main__":
    print("开始遗传算法优化")
    ga = GeneticAlgorithm()
    best_params, best_fitness = ga.optimize()
    Kp_opt, Ki_opt, Kd_opt = best_params

    print("\n最优PID参数：")
    print(f"Kp = {Kp_opt:.3f}, Ki = {Ki_opt:.8f}, Kd = {Kd_opt:.3f}")
    print(f"最优适应度 = {best_fitness:.2f}")

    # 精细仿真验证（保存控制量）
    sim_time_fine = 5 * SYSTEM["T"]
    dt_fine = 1
    temp_fine = [SYSTEM["room_temp"]]
    u_fine = []  # 保存控制量
    time_fine = [0]
    controller = PIDController(*best_params)
    delay_buffer = np.zeros(max(1, int(SYSTEM["theta"] / dt_fine)))

    for t in np.arange(dt_fine, sim_time_fine + dt_fine, dt_fine):
        error = SYSTEM["setpoint"] - temp_fine[-1]
        u = controller.update(error, dt_fine)
        u_fine.append(u)
        delay_buffer = np.roll(delay_buffer, -1)
        delay_buffer[-1] = u
        u_delayed = delay_buffer[0] if len(time_fine) > len(delay_buffer) else 0.0
        new_temp = temp_fine[-1] + (dt_fine / SYSTEM["T"]) * (SYSTEM["K"] * u_delayed - (temp_fine[-1] - SYSTEM["room_temp"]))
        temp_fine.append(new_temp)
        time_fine.append(t)

    time_fine = np.array(time_fine)
    temp_fine = np.array(temp_fine)
    u_fine = np.array(u_fine)

    # 计算动态指标
    rise_time, overshoot, settling_time, steady_error, peak_time, decay_ratio, oscill_count, max_temp = calculate_metrics(
        time_fine, temp_fine, SYSTEM["setpoint"], SYSTEM["room_temp"]
    )

    # 打印性能指标
    print("\n系统性能指标：")
    print(f"上升时间: {rise_time:.1f} 秒")
    print(f"峰值时间: {peak_time:.1f} 秒")
    print(f"超调量: {overshoot:.2f} %")
    print(f"调节时间: {settling_time:.1f} 秒")
    print(f"稳态误差: {steady_error:.3f} ℃")
    if np.isinf(decay_ratio):
        print("衰减比: 无振荡（单调响应）")
    else:
        print(f"衰减比: {decay_ratio:.2f}")
    print(f"振荡次数: {oscill_count} 次")

    # 可视化结果（3个子图：温度+指标、控制量、适应度）
    plt.figure(figsize=(12, 12))

    # 子图1：温度响应 + 指标文本框
    plt.subplot(3, 1, 1)
    plt.plot(time_fine, temp_fine, label='实际温度', linewidth=2)
    plt.axhline(SYSTEM["setpoint"], color='r', linestyle='--', label='设定值')
    plt.axhline(SYSTEM["room_temp"], color='g', linestyle=':', label='室温')
    plt.title('遗传算法优化PID控制响应')
    plt.ylabel('温度 (℃)')
    plt.legend()
    plt.grid()

    # 指标文本框（右上角）
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    text_str = f"""上升时间: {rise_time:.1f} s
                    峰值时间: {peak_time:.1f} s
                    超调量: {overshoot:.2f} %
                    调节时间: {settling_time:.1f} s
                    稳态误差: {steady_error:.3f} ℃
                    衰减比: {'无振荡' if np.isinf(decay_ratio) else f'{decay_ratio:.2f}'}
                    振荡次数: {oscill_count} 次"""
    plt.text(0.65, 0.95, text_str, transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='top', bbox=props)

    # 子图2：控制量输出
    plt.subplot(3, 1, 2)
    plt.plot(time_fine[:-1], u_fine, label='控制电压', color='orange', linewidth=2)
    plt.title('PID控制器输出')
    plt.ylabel('电压 (V)')
    plt.legend()
    plt.grid()

    # 子图3：适应度下降曲线
    plt.subplot(3, 1, 3)
    plt.plot(ga.fitness_history, linewidth=2)
    plt.title('遗传算法优化过程（适应度下降曲线）')
    plt.xlabel('代数')
    plt.ylabel('适应度')
    plt.grid()

    plt.tight_layout()
    plt.show()