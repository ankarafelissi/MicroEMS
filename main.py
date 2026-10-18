# -*- coding:utf-8 -*-
"""
% 输入: 
%   - 光伏历史数据（data/gansu_pv.csv）
%   - 负荷历史数据（data/quanzhou_load.csv）
%   - 储能系统参数（容量、充放电功率限制）
%   - 分时电价数据
%   - InfluxDB数据库连接配置
% 输出: 
%   - 光伏功率预测值（96点，15分钟间隔）
%   - 负荷功率预测值（96点，15分钟间隔）
%   - 储能充放电优化策略
%   - 实时运行数据写入InfluxDB

"""

import time
import logging
from influxdb import InfluxDBClient
import pandas as pd
import numpy as np
from predict import DEVICE, predict_load, predict_pv
from scipy import interpolate
import matplotlib
import warnings
import os
import sys
# Pyomo
from pyomo.core import ConcreteModel, Var, Constraint, Objective, ConstraintList
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# 中文配置
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['FangSong']
warnings.filterwarnings('ignore')

# ===================== 配置参数 =====================
class Config:
    """系统配置"""
    # InfluxDB配置
    INFLUX_HOST = 'localhost'
    INFLUX_PORT = 8086
    INFLUX_USER = 'admin'
    INFLUX_PASSWORD = 'admin'
    
    # 数据库名称 - 微电网EMS专用
    DB_EMS = 'microgrid_ems'
    
    # 数据表名（measurement）
    MEAS_NWP = 'nwp_data'
    MEAS_PV_HISTORY = 'pv_history'
    MEAS_LOAD_HISTORY = 'load_history'
    MEAS_PV_FORECAST = 'pv_forecast'
    MEAS_LOAD_FORECAST = 'load_forecast'
    MEAS_OPTIMIZATION = 'optimization'
    MEAS_SOC = 'battery_soc'
    MEAS_MODE = 'operation_mode'
    
    # 数据路径
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    PV_DATA_PATH = os.path.join(BASE_PATH, 'data', 'gansu_pv.csv')
    LOAD_DATA_PATH = os.path.join(BASE_PATH, 'data', 'quanzhou_load.csv')
    PRICE_PATH = os.path.join(BASE_PATH, 'data', 'electricity_prices.csv')
    
    # 储能系统参数（与光伏/负荷数据量级匹配，单位：W和Wh）
    BATTERY_CAPACITY = 5000  # Wh（对应5kWh）
    PMAX_CH = 2000  # W 充电功率限制
    PMAX_DH = 2000  # W 放电功率限制
    YITA = 0.95
    SOC_MIN = 0.1
    SOC_MAX = 0.9
    SOC_INIT = 0.5
    
    # 电网功率限制（与光伏/负荷数据量级匹配，单位：W）
    GRID_MAX_BUY = 5000   # W
    GRID_MAX_SELL = 3000  # W
    
    # 数据缩放参数
    PV_SCALE = 6.0    # 光伏缩放倍数
    LOAD_SCALE = 0.2  # 负荷缩放倍数
    
    # 时间参数
    T = 96
    DT = 0.25


config = Config()


# ===================== InfluxDB客户端 =====================
class InfluxDBManager:
    """InfluxDB管理器"""
    def __init__(self):
        self.client = InfluxDBClient(
            host=config.INFLUX_HOST,
            port=config.INFLUX_PORT,
            username=config.INFLUX_USER,
            password=config.INFLUX_PASSWORD
        )
        self._init_databases()
    
    def _init_databases(self):
        """初始化数据库 - 删除旧数据库并重新创建"""
        try:
            self.client.drop_database(config.DB_EMS)
            print(f"[数据库] 已删除旧数据库: {config.DB_EMS}")
        except:
            pass

        self.client.create_database(config.DB_EMS)
        print(f"[数据库] 已创建新数据库: {config.DB_EMS}")

    def write_point(self, database, measurement, fields, tags=None, time=None):
        """写入单条数据"""
        json_body = [{
            "measurement": measurement,
            "tags": tags or {},
            "time": time or pd.Timestamp.now(tz='Asia/Shanghai').isoformat(),
            "fields": fields
        }]
        try:
            self.client.switch_database(database)
            self.client.write_points(json_body)
            return True
        except Exception as e:
            logging.error(f'写入失败: {e}')
            return False


# ===================== 训练模型预测 =====================
class Forecaster:
    """使用最佳模型预测，保留调度所需的功率缩放。"""
    def __init__(self):
        self.db = InfluxDBManager()
    
    def run_pv_forecast(self):
        """使用前500天中的最后一天，预测其后一天的光伏功率。"""
        print("\n[光伏预测] 使用训练好的LSTM-Transformer模型...")
        
        # 加载甘肃光伏数据
        # 保留训练时使用的前500天，预测函数按保存的列名选择特征。
        history = pd.read_csv(config.PV_DATA_PATH, nrows=500 * 96)
        
        pv_forecast = predict_pv(history)
        
        # 量级缩放
        pv_forecast_scaled = pv_forecast * config.PV_SCALE
        
        print(f"[光伏预测] 完成，原始范围: {pv_forecast.min():.2f} - {pv_forecast.max():.2f}（原始数据单位）")
        print(f"[光伏预测] 缩放后范围: {pv_forecast_scaled.min():.2f} - {pv_forecast_scaled.max():.2f} W")
        
        # 写入数据库
        current_time = pd.Timestamp.now(tz='Asia/Shanghai')
        for i in range(96):
            self.db.write_point(
                config.DB_EMS, config.MEAS_PV_FORECAST,
                {"power": float(pv_forecast_scaled[i])},
                time=(current_time + pd.Timedelta(minutes=15*i)).isoformat()
            )
        
        return pv_forecast_scaled
    
    def run_load_forecast(self):
        """使用CSV末尾7天的数据，预测其后24小时的负荷。"""
        print("\n[负荷预测] 使用训练好的LSTM模型...")
        
        # 加载负荷数据
        df = pd.read_csv(config.LOAD_DATA_PATH)
        
        load_forecast = predict_load(df)
        
        # 量级缩放
        load_forecast_scaled = load_forecast * config.LOAD_SCALE
        
        print(f"[负荷预测] 完成，原始范围: {load_forecast.min():.2f} - {load_forecast.max():.2f}（原始数据单位）")
        print(f"[负荷预测] 缩放后范围: {load_forecast_scaled.min():.2f} - {load_forecast_scaled.max():.2f} W")
        
        # 写入数据库
        current_time = pd.Timestamp.now(tz='Asia/Shanghai')
        for i in range(96):
            self.db.write_point(
                config.DB_EMS, config.MEAS_LOAD_FORECAST,
                {"power": float(load_forecast_scaled[i])},
                time=(current_time + pd.Timedelta(minutes=15*i)).isoformat()
            )
        
        return load_forecast_scaled
    
    def run_forecast(self):
        """运行预测流程"""
        print("\n" + "="*60)
        print("  步骤1: 光伏与负荷预测")
        print("="*60)
        
        pv_forecast = self.run_pv_forecast()
        load_forecast = self.run_load_forecast()
        
        return pv_forecast, load_forecast


# ===================== 优化模块 =====================
class Optimizer:
    """优化器"""
    def __init__(self):
        self.db = InfluxDBManager()
        self.load_price_data()
    
    def load_price_data(self):
        """加载电价数据"""
        price_data = pd.read_csv(config.PRICE_PATH)
        price = price_data.to_numpy()
        price_buy_24 = price[:, 1]
        price_sell_24 = price[:, 2]
        
        t_24 = np.linspace(0, 23, 24)
        t_real = np.linspace(0, 23, 96)
        f = interpolate.interp1d(t_24, price_buy_24, kind='zero')
        self.price_buy = f(t_real)
        f = interpolate.interp1d(t_24, price_sell_24, kind='zero')
        self.price_sell = f(t_real)
        
        print(f"[电价] 购电范围: {self.price_buy.min():.4f} - {self.price_buy.max():.4f} 元/kWh")
        print(f"[电价] 售电范围: {self.price_sell.min():.4f} - {self.price_sell.max():.4f} 元/kWh")
    
    def build_optimization_model(self, x, k, soc_last_time, PV_change, LOAD_change):
        """构建优化模型"""
        m = ConcreteModel()
        
        m.pg = Var(range(6), range(k+1), within=pyo.NonNegativeReals)
        m.ug = Var(range(6), range(k+1), within=pyo.Binary)
        
        m.obj = Objective(expr=
            sum(
                self.price_buy[x+j-1] * m.pg[3,j] * config.DT / 1000 - 
                self.price_sell[x+j-1] * m.pg[4,j] * config.DT / 1000 +
                (m.ug[2,j] + m.ug[3,j]) * 0.01
                for j in range(k+1)
            )
        )
        
        m.constraints = ConstraintList()
        
        m.constraints.add(m.pg[0, 0] == 0)
        m.constraints.add(m.pg[1, 0] == 0)
        m.constraints.add(m.pg[2, 0] == soc_last_time)
        m.constraints.add(m.pg[3, 0] == 0)
        m.constraints.add(m.pg[4, 0] == 0)
        m.constraints.add(m.pg[5, 0] == 0)
        
        for j in range(1, k+1):
            if j == 1:
                m.constraints.add(
                    m.pg[1, j] + m.pg[3, j] + PV_change[j-1] == 
                    LOAD_change[j-1] + m.pg[0, j] + m.pg[4, j] + m.pg[5, j]
                )
            else:
                m.constraints.add(
                    m.pg[1, j] + m.pg[3, j] + PV_change[j-1] == 
                    LOAD_change[j-1] + m.pg[0, j] + m.pg[4, j] + m.pg[5, j]
                )
            
            m.constraints.add(
                config.BATTERY_CAPACITY * (m.pg[2, j] - m.pg[2, j-1]) == 
                config.DT * (m.pg[0, j] * config.YITA - m.pg[1, j] / config.YITA)
            )
            
            m.constraints.add(m.pg[0, j] <= m.ug[0, j] * config.PMAX_CH)
            m.constraints.add(m.pg[1, j] <= m.ug[1, j] * config.PMAX_DH)
            m.constraints.add(m.ug[0, j] + m.ug[1, j] <= 1)
            m.constraints.add(config.SOC_MIN <= m.pg[2, j])
            m.constraints.add(m.pg[2, j] <= config.SOC_MAX)
            m.constraints.add(m.pg[4, j] <= config.GRID_MAX_SELL * m.ug[5, j])
            m.constraints.add(m.pg[3, j] <= config.GRID_MAX_BUY * m.ug[4, j])
            m.constraints.add(m.ug[5, j] + m.ug[4, j] <= 1)
        
        return m
    
    def solve_single_step(self, x, soc_last_time, PV_15min, LOAD_15min):
        """求解单步优化"""
        k = config.T - x
        
        PV_change = PV_15min[x:config.T]
        LOAD_change = LOAD_15min[x:config.T]
        
        m = self.build_optimization_model(x, k, soc_last_time, PV_change, LOAD_change)
        
        solver = SolverFactory('gurobi')
        solver.options['TimeLimit'] = 30
        solver.options['MIPGap'] = 0.015
        solver.options['OutputFlag'] = 0
        
        results = solver.solve(m, tee=False)
        
        yy = pyo.value(m.pg[:, 1])
        
        return {
            'p_ch': yy[0],
            'p_dh': yy[1],
            'soc': yy[2],
            'p_b': yy[3],
            'p_s': yy[4],
            'p_ab': yy[5]
        }
    
    def run_rolling_optimization(self, PPV_15min, Pload_15min):
        """运行滚动优化"""
        print("\n" + "="*60)
        print("  步骤2: 日内滚动优化")
        print("="*60)
        
        # 使用预测数据作为实际数据（演示）
        PV_15min = PPV_15min * 0.9  # 实际值略低于预测值
        LOAD_15min = Pload_15min * 1.1  # 实际值略高于预测值
        
        p_ch = np.zeros(config.T)
        p_dh = np.zeros(config.T)
        p_b = np.zeros(config.T)
        p_s = np.zeros(config.T)
        p_ab = np.zeros(config.T)
        soc_realtime = np.zeros(config.T)
        
        soc_last_time = config.SOC_INIT
        
        for x in range(config.T):
            print(f'----------- t={x} -----------')
            
            start_time = time.time()
            
            result = self.solve_single_step(x, soc_last_time, PV_15min, LOAD_15min)
            
            solve_time = time.time() - start_time
            print(f'  [求解耗时] {solve_time:.2f}秒')
            
            p_ch[x] = result['p_ch']
            p_dh[x] = result['p_dh']
            p_b[x] = result['p_b']
            p_s[x] = result['p_s']
            p_ab[x] = result['p_ab']
            
            # SOC直接从优化结果获取
            soc_realtime[x] = result['soc']
            
            # 更新SOC
            soc_last_time = result['soc']
            
            # 确定运行模式
            if p_ch[x] > 10:
                mode = 3  # 充电模式
            elif p_dh[x] > 10:
                mode = 3  # 放电模式
            elif p_b[x] > 10:
                mode = 1  # 购电模式
            elif p_s[x] > 10:
                mode = 2  # 售电模式
            else:
                mode = 3  # 自发自用
            
            # 打印关键信息
            hour = int(x * 0.25)
            price = self.price_buy[x] if x < len(self.price_buy) else 0
            print(f'  [t={x:2d} | {hour:02d}:00] SOC={soc_realtime[x]*100:.1f}%, '
                  f'p_ch={p_ch[x]:.0f}W, p_dh={p_dh[x]:.0f}W, '
                  f'p_buy={p_b[x]:.0f}W, p_sell={p_s[x]:.0f}W, '
                  f'电价={price:.3f}')
            
            # 写入数据库
            self._write_optimization_results(
                x, result, soc_realtime[x], mode,
                PV_15min[x], PPV_15min[x], LOAD_15min[x], Pload_15min[x]
            )
            
            soc_last_time = soc_realtime[x]
            
            if x < config.T - 1:
                sleep_time = max(0, 5 - solve_time)
                if sleep_time > 0:
                    print(f'  [等待] {sleep_time:.2f}秒')
                    time.sleep(sleep_time)
        
        print("\n[统计] 优化完成")
        print(f'  平均SOC: {np.mean(soc_realtime)*100:.1f}%')
        print(f'  总购电量: {np.sum(p_b)*config.DT:.2f} kWh')
        print(f'  总售电量: {np.sum(p_s)*config.DT:.2f} kWh')
        
        return p_ch, p_dh, p_b, p_s, p_ab, soc_realtime
    
    def _write_optimization_results(self, x, result, soc, mode, PV_actual, PV_forecast, LOAD_actual, LOAD_forecast):
        """写入优化结果"""
        current_time = pd.Timestamp.now(tz='Asia/Shanghai')
        
        self.db.write_point(
            config.DB_EMS, config.MEAS_OPTIMIZATION,
            {
                "p_ch": float(result['p_ch']),
                "p_dh": float(result['p_dh']) * -1,
                "p_buy": float(result['p_b']),
                "p_sell": float(result['p_s']) * -1,  # 售电功率改为负数
                "p_ab": float(result['p_ab']),
                "soc": float(soc) * 100,
                "mode": int(mode),
                "time_slot": int(x),
                "pv_actual": float(PV_actual),
                "pv_forecast": float(PV_forecast),
                "load_actual": float(LOAD_actual),
                "load_forecast": float(LOAD_forecast)
            },
            time=current_time.isoformat()
        )
        
        self.db.write_point(
            config.DB_EMS, config.MEAS_SOC,
            {"soc": float(soc) * 100, "time_slot": int(x)},
            time=current_time.isoformat()
        )
        
        self.db.write_point(
            config.DB_EMS, config.MEAS_MODE,
            {"mode": int(mode), "time_slot": int(x)},
            time=current_time.isoformat()
        )
        
        print(f"  [数据写入] t={x}, SOC={soc*100:.1f}%, mode={mode}, p_buy={result['p_b']:.1f}, p_sell={result['p_s']:.1f}")


# ===================== 主程序 =====================
def main():
    """主程序"""
    print("\n" + "="*60)
    print("  光储微电网能量管理系统 - 最终版")
    print("="*60)
    print(f"  时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  设备: {DEVICE}")
    print("="*60)
    
    # 步骤1: 预测
    forecaster = Forecaster()
    PPV_15min, Pload_15min = forecaster.run_forecast()
    
    # 步骤2: 滚动优化
    optimizer = Optimizer()
    p_ch, p_dh, p_b, p_s, p_ab, soc_realtime = optimizer.run_rolling_optimization(
        PPV_15min, Pload_15min
    )
    
    print("\n" + "="*60)
    print("  系统运行完成！")
    print("="*60)


if __name__ == "__main__":
    main()
