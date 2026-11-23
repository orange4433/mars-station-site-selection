import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from matplotlib import colors, cm
import numpy as np
import matplotlib.pyplot as plt
from obspy import read, UTCDateTime,Trace, Stream
from scipy.signal import spectrogram, welch, windows,detrend
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import os
from matplotlib.gridspec import GridSpec
from scipy.optimize import minimize
from scipy.integrate import cumulative_trapezoid
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from obspy import read, UTCDateTime,Trace, Stream
from scipy.signal import spectrogram, welch, windows
from datetime import datetime, timedelta, timezone
from scipy.interpolate import interp1d
import os
import matplotlib.dates as mdates
import pytz
import urllib.request
import csv
from obspy import read, UTCDateTime,Trace, Stream
from scipy.signal import spectrogram, windows
from scipy.interpolate import interp1d
from typing import Tuple, Optional
from scipy.ndimage import gaussian_filter1d
from scipy.signal import istft, stft
from scipy.signal.windows import hamming

plt.rcParams['font.family'] = 'Arial'
scale = 24

def mars_sol2ls(sol):
    """
    将火星日(Sol)转换为太阳经度(Ls)
    使用精确的火星轨道参数计算
    """
    # 火星轨道参数
    year_day = 668.6  # 火星年长度（火星日）
    peri_day = 485.35  # 近日点对应的火星日
    e_elips = 0.09340  # 轨道偏心率
    timeperi = 1.90258341759902  # 近日点经度（弧度）
    
    # 计算平近点角
    M = 2 * np.pi * (sol - peri_day) / year_day
    M = M % (2 * np.pi)
    
    # 开普勒方程求解（牛顿迭代法）
    E = M
    for _ in range(10):
        delta_E = (E - e_elips * np.sin(E) - M) / (1 - e_elips * np.cos(E))
        E -= delta_E
        if abs(delta_E) < 1e-7:
            break
    
    # 计算真近点角
    true_anomaly = 2 * np.arctan(np.sqrt((1 + e_elips) / (1 - e_elips)) * np.tan(E / 2))
    
    # 计算太阳经度Ls
    Ls = (true_anomaly - timeperi) % (2 * np.pi)
    return np.degrees(Ls)

def insight_sol_to_mars_day(insight_sol):
    """
    将Insight Sol转换为火星年中的实际日数
    Insight着陆日对应火星年的第555天
    """
    # Insight Sol 0 对应火星年的第555天
    mars_day = insight_sol + 555
    
    # 火星年长度
    mars_year_length = 668.6
    
    # 如果超过一年，取模
    if mars_day > mars_year_length:
        mars_day = mars_day % mars_year_length
    
    return mars_day

freq_list = [0.05, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8]
freq_list2 = [0.1,1]
SEISMIC_PARAMS = {
    'channel': 'Z',          # 处理通道
    'spec_window': 512,      # 频谱分析窗口
    'noverlap': 256,         # 频谱重叠样本
    'clim': (-220, -140),    # 颜色映射范围
    'freq_ticks': [0.1, 0.3, 1, 3, 8]  # 频率刻度
}

class lmst(object):
    def __init__(self,sol=0,hour=0,minute=0,sec=0):
        self.sol = sol
        self.hour = hour
        self.min = minute
        self.sec = sec
    def __str__(self):
        return ('Sol-''%.5d''M''%.2d'':''%.2d'':''%05.2f') %(self.sol,self.hour,self.min,self.sec)
    __repr__ = __str__
    def __sub__(self,other):
        #time difference in seconds
        sol1_start = UTCDateTime("2018-11-27T05:50:25.580014Z")
        sol2_start = UTCDateTime("2018-11-28T06:30:00.823990Z")
        sec_m = sol2_start - sol1_start - 0.000005
        sec_e = 86400
        t1 = self.conv2utc()
        t2 = other.conv2utc()
        return (t1-t2)*sec_e/sec_m
    def __add__(self,dt_m):
        sol1_start = UTCDateTime("2018-11-27T05:50:25.580014Z")
        sol2_start = UTCDateTime("2018-11-28T06:30:00.823990Z")
        sec_m = sol2_start - sol1_start - 0.000005
        sec_e = 86400
        dt_e = dt_m*sec_m/sec_e
        t1 = self.conv2utc()
        t2 = t1+dt_e
        tmp = lmst()
        tmp.utc2lmst(t2)
        return tmp
    def _display(self):
        # = print
        print('Sol-''%.5d''M''%.2d'':''%.2d'':''%05.2f' %(self.sol,self.hour,self.min,self.sec))
    def get_sol_hour_min(self):
        return ('Sol''%.3d''\n''%.2d'':''%.2d') %(self.sol,self.hour,self.min)
    def get_hour(self):
        return self.hour
    def get_hour_min(self):
        return ('%.2d''.''%.2d') %(self.hour,self.min)
    def conv2utc(self):
        ref = UTCDateTime("2018-11-26T05:10:50.336037Z")
        sol1_start = UTCDateTime("2018-11-27T05:50:25.580014Z")
        sol2_start = UTCDateTime("2018-11-28T06:30:00.823990Z")
        sec_m = sol2_start - sol1_start - 0.000005
        sec_e = 86400
        # calculate secs on Mars
        dur_m = self.sol*24*3600+self.hour*3600+self.min*60+self.sec
        dur_e = dur_m*sec_m/sec_e
        return ref+dur_e
    def utc2lmst(self,utc_time):        
        sol1_start = UTCDateTime("2018-11-27T05:50:25.580014Z")
        sol2_start = UTCDateTime("2018-11-28T06:30:00.823990Z")
        sec_m = sol2_start - sol1_start - 0.000005
        sec_e = 86400
        dur_e = utc_time-UTCDateTime("2018-11-26T05:10:50.336037Z")
        dur_m = dur_e*sec_e/sec_m
        self.sol = dur_m // 86400
        mod = dur_m % 86400
        self.hour = mod // 3600
        mod = mod %3600
        self.min = mod // 60
        self.sec = mod % 60
    def check(self):
        if round(self.sec) == 60:
            self.sec = 0
            self.min += 1
        if self.min == 60:
            self.min = 0
            self.hour += 1
        if self.hour == 24:
            self.hour = 0
            self.sol += 1


def c2utc(a):
    bbcc=lmst()
    bbcc.sol=int(a[:5])
    bbcc.hour = int(a[6:8])
    bbcc.min=int(a[9:11])
    bbcc.sec = float(a[12:])
    return bbcc.conv2utc()

def c2lmst(a):
    aa=lmst()
    aa.utc2lmst(UTCDateTime(a))
    return aa


class WindProcessor:
    """风速数据时空处理器 - 适配mcd_data.nc格式"""
    
    def __init__(self, nc_path: str):
        self.dataset = nc.Dataset(nc_path)
        self.sol_min = 114
        self.sol_max = 794
        self._cache = {}  # 位置数据缓存
        self._precompute_sol_data()
    
    def _precompute_sol_data(self):
        """预计算所有Sol的风速数据"""
        # 计算每个Sol对应的Ls值
        self.sol_values = np.arange(self.sol_min, self.sol_max + 1)
        self.ls_values = []
        
        for sol in self.sol_values:
            mars_day = insight_sol_to_mars_day(sol)
            ls = mars_sol2ls(mars_day)
            self.ls_values.append(ls)
        
        # 原始数据的Ls值
        self.original_ls = self.dataset['Ls'][:]
    
    def get_wind_profile(self, lat: float, lon: float) -> Tuple[np.ndarray, np.ndarray]:
        """获取指定位置风速数据"""
        # 检查缓存
        cache_key = f"{lat}_{lon}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # 查找最近网格点
        lats = self.dataset['latitude'][:]
        lons = self.dataset['longitude'][:]
        
        lat_idx = np.abs(lats - lat).argmin()
        lon_idx = np.abs(lons - lon).argmin()
        
        # 获取所有Sol的风速数据
        u_data = np.zeros((len(self.sol_values), 48))
        v_data = np.zeros((len(self.sol_values), 48))
        
        for i, sol in enumerate(self.sol_values):
            # 找到最接近的Ls索引
            target_ls = self.ls_values[i]
            ls_idx = np.argmin(np.abs(self.original_ls - target_ls))
            
            # 获取该位置和Ls对应的风速数据
            u_data[i, :] = self.dataset['U'][lon_idx, lat_idx, :, ls_idx]
            v_data[i, :] = self.dataset['V'][lon_idx, lat_idx, :, ls_idx]
        
        # 应用地方时校正
        shift_hours = int(lon / 15)  # 经度转换为小时偏移
        shift_points = 0*shift_hours * 2  # 每半小时一个点
        
        # 对每个Sol的数据进行滚动
        adjusted_u = np.zeros_like(u_data)
        adjusted_v = np.zeros_like(v_data)
        
        for sol_idx in range(len(self.sol_values)):
            adjusted_u[sol_idx] = np.roll(u_data[sol_idx], -shift_points)
            adjusted_v[sol_idx] = np.roll(v_data[sol_idx], -shift_points)
        
        # 将数据展平为连续时间序列
        u = adjusted_u.flatten()
        v = adjusted_v.flatten()
        print(len(u))
        # 缓存结果
        self._cache[cache_key] = (u, v)
        return u, v

# 绘制单个点的风速变化热力图
def figure(wind_speeds, min_value, max_value, u, integral, title):
    data = wind_speeds.reshape(-1, 48)  # 假设每小时有48个时间步长
    
    # 创建一个更大的数组，其中包含你的数据，并在其他部分填充NaN值
    full_data = np.full((1001, 48), np.nan)  # 创建一个1300x48的全NaN数组
    full_data[114:114+len(data), :] = data  # 将你的数据复制到全NaN数组的相应位置
    
    # 使用matplotlib绘制热力图
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_ylim(0, 1000)
    
    # cmap参数代表色彩映射
    sol_range = [1, 1000]
    norm = colors.Normalize(vmin=min_value, vmax=max_value, clip=True)
    scalar_mappable = cm.ScalarMappable(norm=norm, cmap=plt.get_cmap('plasma'))
    scalar_mappable.set_array([])  # 设置一个空数组
    im = ax.imshow(full_data, norm=norm, cmap='plasma', aspect=u)
    
    # 添加Colorbar
    cbar = plt.colorbar(scalar_mappable, ax=ax)
    cbar.set_label('Wind Speed (m/s)', fontdict={'family': 'Arial', 'fontsize': scale})
    for label in cbar.ax.get_yticklabels():
        label.set_family('Times New Roman')
        label.set_fontsize(12)
    

    
    # 显示颜色条
    plt.xlabel("hour")
    plt.ylabel("sol")
    ticks = np.arange(0, 48, 2)
    plt.xticks(ticks=ticks, labels=[i-48 if i > 47 else i for i in ticks])
    plt.xlabel('LMST hour', fontdict={'family': 'Arial', 'fontsize': scale})
    plt.xlim(0, 47)  # 修改这里以适应48个点
    original_xticks = plt.xticks()[0]

    # 计算新的y轴刻度位置（乘以5倍）
    new_xticks1 = original_xticks / 2
    new_xticks = [int(new_xticks1[i]) for i in range(len(new_xticks1))]
    # 设置新的y轴刻度位置
    plt.xticks(original_xticks, new_xticks)
    
    plt.xticks(fontname="Times New Roman", fontsize=f1)
    plt.ylabel('Sol', fontdict={'family': 'Times New Roman', 'fontsize': f2})
    plt.yticks(fontname="Times New Roman", fontsize=f1)
    plt.ylim(1300, 16)
    
    plt.title(title, fontdict={'family': 'Times New Roman', 'fontsize': 16})
    plt.tight_layout()
    plt.show()

def calc_psd(data, fs):
    dt = 0.05
    segment_length = int(25.6 / dt)  # 例如，25.6秒的时间段长度
    noverlap = int(segment_length * 0.5)  # 50%的重叠
    #nfft = 4096  # 零填充到4096个样本
    window = windows.hann(segment_length)  # 汉宁窗
    
    freq, psd = welch(
        data,
        fs=fs,
        nperseg=segment_length,
        noverlap=noverlap,
        #nfft=nfft,
        window=window,
        scaling='density'
    )
    
    return freq, psd

class SeismicAdjuster:
    """基于多频段风速特征的地震信号修正器"""
    
    def __init__(self, ref_day: int, target_day: int):
        self.ref_day = ref_day
        self.target_day = target_day
        
        # 定义频率分界点
        self.freq_breaks = freq_list
        
        # 加载17个插值器
        self.interpolators = [[] for i in range(1)]
        for i in range(len(self.freq_breaks)-1):
            for j in range(1):
                low = self.freq_breaks[i]
                high = self.freq_breaks[i+1]
                data = np.loadtxt(f'./17/wind_speed_earthquake_signals_{low}-{high}_1_{j}.txt')
                self.interpolators[j].append(
                    interp1d(data[:,0], data[:,1], fill_value="extrapolate")
                )
        self.interpolators2 = [[] for i in range(1)]
        for j in range(1):
            low = 0.1
            high = 1
            data = np.loadtxt(f'./17/wind_speed_earthquake_signals_{low}-{high}_1_{j}.txt')
            self.interpolators2[j].append(
                interp1d(data[:,0], data[:,1], fill_value="extrapolate")
            )


    
def align_daily_signal(stream: Stream) -> Tuple[np.ndarray, float]:
    """对齐到火星日00:00-24:00"""
    # 获取信号基本信息
    tr = stream.select(channel='EHZ')[0]
    delta = tr.stats.delta
    starttime_lmst = c2lmst(tr.stats.starttime)
    start_time_utc = tr.stats.starttime
    # 计算当日火星时间范围
    # 创建一个lmst对象代表我们感兴趣的起始时间(当天的00:00 LMST)
    target_start_lmst = lmst(sol=starttime_lmst.sol, hour=17, minute=45, sec=0)
    target_start_utc = target_start_lmst.conv2utc()

    # 同样创建一个代表当天结束时间(24:00 LMST)的对象
    target_end_lmst = lmst(sol=starttime_lmst.sol, hour=18, minute=35, sec=0)
    target_end_utc = target_end_lmst.conv2utc()

    # 根据目标时间转换为样本索引
    start_idx = int((target_start_utc - start_time_utc) / delta)
    end_idx = int((target_end_utc - start_time_utc) / delta)
    
    # 截取并返回
    return tr.data[start_idx:end_idx], delta

  
ref_day=20

import math

# 火星的平均半径 (km)
R = 3389.5  # 平均半径

def haversine(point1, point2):
    """
    计算两个地理坐标点之间的距离（Haversine公式）
    
    参数:
        point1: (纬度1, 经度1) 单位：度
        point2: (纬度2, 经度2) 单位：度
        
    返回:
        distance: 两点之间的距离 (km)
        angle_distance: 两点之间的角度距离 (弧度)
    """
    lat1, lon1 = math.radians(point1[0]), math.radians(point1[1])
    lat2, lon2 = math.radians(point2[0]), math.radians(point2[1])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    distance = R * c
    angle_distance = c  # 角度距离（大圆距离的角度）
    
    return angle_distance

# 标签是地震仪可取点
seismometer_points = [
    (-48.75, -45.0),
    (-7.5, -5.625),
    (-37.5, 118.125),
    (-45.0, 151.875),
    (-75.0, 78.75),
    (22.5, 180)
]

# 地震可能发生的点
earthquake_points = [
    (10, -105),
'''
    (35, 90),
    (15, 60),
    (-15, 110),
    (10, 160),
    (-20, 150),
    (18.4, -134),
    (0, -115),
    (-10,-120),
    (-40, 60),
    (-50, -45),
    (-10, -75)
'''
]

class SpectrumVisualizer:
    """频谱对比可视化工具"""
    
    @staticmethod
    def plot_comparison(signals,signals1,ratio,ratio1,Mw, sample_rate=100):
        """四图对比显示"""
        
        fig = plt.figure(figsize=(14, 12))

        # 使用 GridSpec 定义布局：
        # 第一行分为两列，其中第一列占 1/3 宽度，第二列留空（只用第一列）
        # 第二行跨全部宽度
        gs = GridSpec(2, 2, figure=fig, width_ratios=[5, 1], height_ratios=[1, 2])

        # axes[0]：占据第一行的第一列（共三列中的一列 → 占 1/3 宽度）
        ax0 = fig.add_subplot(gs[0, 0])  # 只使用第一行第一列

        # axes[1]：占据第二行全部三列（全宽）
        ax1 = fig.add_subplot(gs[1, :])  # 跨越所有列
        
        # 加载地震数据（新增部分）
        seis_data = np.load('event_1133_spectrogram.npz')
        seis_f = seis_data['frequencies']  # 原始地震频率轴（Hz）
        seis_t = seis_data['times']        # 原始地震时间轴（秒）
        seis_Sxx = seis_data['power']      # 原始地震功率数据（dB/Hz）
        
        def overlay_seismic(sim_t, sim_f, sim_Sxx):
            """在指定小时位置叠加地震数据"""
            # 1. 计算地震数据的时间偏移
            target_origin = 600  # 转换为秒
            
            # 2. 计算地震数据在模拟时间轴中的位置
            seis_start = target_origin + seis_t[0]
            seis_end = target_origin + seis_t[-1]
            # 3. 确定叠加范围（仅处理重叠部分）
            overlap_start = max(sim_t[0], seis_start)
            overlap_end = min(sim_t[-1], seis_end)
            '''
            if overlap_start > overlap_end:
                print(1)
                return sim_t, sim_f, sim_Sxx  # 无重叠时返回原数据
            '''
            # 创建插值函数：输入 t_20，输出 Sxx_20Hz 的每一行
            seis_Sxx = seis_data['power']
           
            f_interp = interp1d(seis_t, seis_Sxx, kind='linear', axis=1,fill_value="extrapolate")
            
            # 将 Sxx_20Hz 插值到 t_100 的时间点上
            seis_Sxx = f_interp(sim_t)
            
            # 4. 时间轴索引计算
            #seis_idx_start = np.searchsorted(seis_t, overlap_start - target_origin)
            #seis_idx_end = np.searchsorted(seis_t, overlap_end - target_origin)
            sim_idx_start = np.searchsorted(sim_t, overlap_start)
            sim_idx_end = np.searchsorted(sim_t, overlap_end)
            seis_idx_start = sim_idx_start
            seis_idx_end = sim_idx_end
            
            
            
            # 5. 频率轴对齐（地震数据截断到模拟频率范围）
            freq_mask = (seis_f >= 0.1) & (seis_f <= 1)
            valid_seis_f = seis_f[freq_mask]
            valid_seis_Sxx = seis_Sxx[freq_mask, :]
            
            # 6. 创建插值函数（频率轴）
            interp_Sxx = interp1d(seis_f, seis_Sxx, axis=0, 
                                 bounds_error=False, 
                                 fill_value=np.nan)(sim_f)
            interp_Sxx = (10**(interp_Sxx / 10))*((ratio)**2)
            # 7. 数据叠加（直接覆盖）
            modified_Sxx = sim_Sxx.copy()

            modified_Sxx[:, sim_idx_start:sim_idx_end] = np.maximum(
                modified_Sxx[:, sim_idx_start:sim_idx_end],
                interp_Sxx[:, seis_idx_start:seis_idx_end]
            )
            
            return sim_t, sim_f, modified_Sxx
        
        
        # 计算频谱

        f, t, Sxx = SpectrumVisualizer._compute_spectrogram(signals1, sample_rate)
        mask = (f >= 0.04) & (f < 10)
        # 合并频谱
        merged_freq = f[mask]
        merged_Sxx = Sxx[mask, :]

        # 在合并频谱后添加：
        merged_t, merged_f, merged_Sxx = overlay_seismic(t, merged_freq, merged_Sxx)

        # 高斯平滑（保持原有逻辑）
        merged_Sxx = gaussian_filter1d(merged_Sxx, sigma=0.1, axis=0)
        
        # 定义目标时间范围（凌晨3:50到4:30）
        start_time = 600  # 4:00 AM in seconds
        end_time = 2400  # 4:30 AM in seconds
        '''
        # 找到目标时间范围内的索引
        time_mask = (merged_t >= start_time) & (merged_t <= end_time)
        merged_t = merged_t[time_mask]
        merged_Sxx = merged_Sxx[:, time_mask]
        '''
        
        key_times = {
            "P Wave": (start_time + 229)/3600,  
            "S Wave": (start_time + 405)/3600  
        }
        # 叠加所有信号
        combined_waveform = signals
        #combined_waveform = target_signals1[0][10]
        #combined_waveform = np.gradient(combined_waveform, 1/sample_rate)
        print(len(signals))
        # 绘制波形图
        waveform = combined_waveform[int((start_time)*100):int((end_time)*100)]
        print(len(waveform))
        
        waveform_t = np.linspace(0, (end_time-start_time)*100, len(waveform))
        
        loaded_data_detrended = np.load('data_detrended_1133.npy')
        

        # 合并两个处理后的数组
        scaled_data_detrended = loaded_data_detrended * ratio1
        #print(len(scaled_data_detrended),max(loaded_data_detrended))

        #1. 构造原始时间轴和目标时间轴
        x_old = np.arange(len(scaled_data_detrended))                # [0, 1, 2, ..., 23999]
        x_new = np.linspace(0, len(scaled_data_detrended) - 1, 120000)  # 在 0 到 23999 之间均匀取 120000 个点

        # 2. 创建线性插值函数并插值
        f_interp = interp1d(x_old, scaled_data_detrended, kind='linear', fill_value='extrapolate')
        array_120k = f_interp(x_new)  # shape: (120000,)

        # 3. 前 120000 个元素相加
        a = 5000
        result_ffirst_part = waveform[0:a]
        result_first_part = waveform[a:120000+a] + array_120k
        print(len(result_first_part))
        # 4. 后 60000 个元素保留不变
        result_second_part = waveform[120000+a:]
        print(result_second_part[:10])
        # 5. 合并结果
        waveform = np.concatenate([result_ffirst_part,result_first_part, result_second_part], axis=0)
        print(len(waveform))
        
        

        #ax0.set_title(f"Waveform Overlay with Seismic Signal (hour:18:00 , Mw:{Mw})", fontsize=20)
        #ax0.set_title(f"Waveform Overlay with S1222a Seismic Signal (hour:18:00)", fontsize=20)
        start_time_hours = 0# 4:00 AM in hours
        end_time_hours = 20 / 60     # 4:30 AM in hours
         # 反变换
         # 将 dB/Hz 转换回幅度平方
        ax0.set_xlim(start_time_hours, end_time_hours)
        
        # 设置x轴刻度和标签
        xticks = np.arange(start_time_hours, end_time_hours + 0.1, float(10/60))  # 每10分钟一标注
        
        
        ax0.set_xticks(xticks)
        ax0.set_xticklabels([f"{int(h+17.84)}:{int((h+0.917)%1 * 61):02d}" for h in xticks])
        ax0.plot(waveform_t / 360000, waveform, color='b')
        #ax0.axvline(x=float((229)/3600), color='g', linestyle=':', linewidth=2, label='P Wave Arrival')
        #ax0.axvline(x=float((405)/3600), color='orange', linestyle=':', linewidth=2, label='S Wave Arrival')
        ax0.axvline(x=float((279)/3600), color='g', linestyle=':', linewidth=2, label='P Wave Arrival')
        ax0.axvline(x=float((455)/3600), color='orange', linestyle=':', linewidth=2, label='S Wave Arrival')
        #ax0.legend(loc='upper right',fontsize = scale-4)
        ax0.set_ylabel("Amplitude(m/s)", fontsize=scale)
        ax0.tick_params(axis='both', which='major', labelsize=scale-2)
        ax0.yaxis.offsetText.set_fontsize(scale-2)
        
        combined_waveform = signals1

        # 绘制波形图
        waveform = combined_waveform[int((start_time)*100):int((end_time)*100)]
        loaded_data_detrended = np.load('data_detrended_1133.npy')
        

        # 合并两个处理后的数组
        scaled_data_detrended = loaded_data_detrended * ratio1
        #print(len(scaled_data_detrended),max(loaded_data_detrended))

        #1. 构造原始时间轴和目标时间轴
        x_old = np.arange(len(scaled_data_detrended))                # [0, 1, 2, ..., 23999]
        x_new = np.linspace(0, len(scaled_data_detrended) - 1, 120000)  # 在 0 到 23999 之间均匀取 120000 个点

        # 2. 创建线性插值函数并插值
        f_interp = interp1d(x_old, scaled_data_detrended, kind='linear', fill_value='extrapolate')
        array_120k = f_interp(x_new)  # shape: (120000,)

        # 3. 前 120000 个元素相加
        ra = 5000
        result_ffirst_part = waveform[0:a]
        result_first_part = waveform[a:120000+a] + array_120k
        print(len(result_first_part))
        # 4. 后 60000 个元素保留不变
        result_second_part = waveform[120000+a:]
        print(result_second_part[:10])
        # 5. 合并结果
        waveform = np.concatenate([result_ffirst_part,result_first_part, result_second_part], axis=0)
        
        waveform_pre = combined_waveform[int((start_time-20)*100):int(start_time*100)]
        waveform = np.concatenate((waveform_pre, waveform))
        
        dt = 0.05  # 您的采样间隔
        velocity_data = waveform  # 如果P_new_Z是速度数据

        # 积分得到位移
        t = np.arange(len(velocity_data)) * dt
        displacement = cumulative_trapezoid(velocity_data, t, initial=0)
        
        merged_f, merged_t, merged_Sxx = SpectrumVisualizer._compute_spectrogram(displacement, fs=100)        
        # 绘制频谱图
        
        merged_t -= 20
        
        SpectrumVisualizer._plot_spectrum(ax1, merged_t, merged_f, merged_Sxx)
        #ax1.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()

    def _get_freq_mask(self, f: np.ndarray) -> np.ndarray:
        """根据当前插值器的频率范围生成mask"""
        low = self.freq_breaks[i]
        high = self.freq_breaks[i+1]
        return (f >= low) & (f <= high)    
    @staticmethod
    def _compute_spectrogram(data: np.ndarray, fs: float) -> Tuple[np.ndarray, ...]:
        """计算频谱图"""
        nperseg = int(256 * 20)  # 256秒窗口
        noverlap = int(nperseg * 0.75)
        return spectrogram(data, fs=fs,
                          window=windows.hamming(nperseg),
                          nperseg=nperseg,
                          noverlap=noverlap,
                          scaling='density')

    @staticmethod
    def _plot_spectrum(ax: plt.Axes, t: np.ndarray, f: np.ndarray, Sxx: np.ndarray):
        """绘制单个频谱图"""
        db = 10 * np.log10(Sxx[1:])  # 排除0Hz
        pc = ax.pcolormesh(t/3600, np.log10(f[1:]), db, cmap='plasma', shading='auto')
        
        # 定义目标时间范围（凌晨3:50到4:30）
        start_time_hours = 0  # 3:50 AM in hours
        end_time_hours =1/3  # 4:30 AM in hours
        
        # 设置x轴范围
        ax.set_xlim(start_time_hours, end_time_hours)
        
        # 设置x轴刻度和标签
        xticks = np.arange(start_time_hours, end_time_hours + 0.1, float(10/60))  # 每10分钟一标注

        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{int(h+17.84)}:{int((h+0.917)%1 * 61):02d}" for h in xticks])
        
        ax.set_ylim(np.log10([SEISMIC_PARAMS['freq_ticks'][0], SEISMIC_PARAMS['freq_ticks'][-1]]))
        ax.set_yticks(np.log10(SEISMIC_PARAMS['freq_ticks']))
        ax.set_yticklabels([str(f) for f in SEISMIC_PARAMS['freq_ticks']])
        ax.set_xlabel("Local Hour (h)", fontsize=scale)
        ax.set_ylabel("Frequency (Hz)", fontsize=scale)
        ax.tick_params(axis='both', which='major', labelsize=scale-2)
        
        #ax.set_title(title, fontsize=20)
        pc.set_clim(-180,-140)
        
        # 添加颜色条（移动到右侧并缩小）
        cbar = plt.colorbar(pc, ax=ax, orientation='vertical', pad=0.04, shrink=0.6)
        cbar.set_label('Power Spectral Density (dB)', fontsize=scale, labelpad=15)
        cbar.ax.tick_params(labelsize=scale)

def compare_asd(nc_file_path, lat, lon,day,hour,earthquake,Mw):
    earthquake_lat = earthquake_points[earthquake][0]
    earthquake_lon = earthquake_points[earthquake][1]
    angle_distance = haversine((earthquake_lat,earthquake_lon), (lat,lon))
    
    
    Evt_name = str(ref_day).zfill(3)
    main_file = f'../../SP/Data_SP_EH/LMST_{Evt_name}.mseed'
    st_disp = read(main_file)
    channel = 'Z'
    ref_tr = st_disp.select(channel=f'EH{channel}')
    
    D_vel = ref_tr[0].data
    
    wind_proc = WindProcessor('../mcd_data.nc')#PCM风速
    
    ref_stream = st_disp
    ref_stream.filter('bandpass',freqmin=0.1,freqmax =0.8,corners=2,zerophase=True)
    ref_stream1 = read(main_file)
    adjusted,_ = align_daily_signal(ref_stream)
    adjusted1,_ = align_daily_signal(ref_stream1)
    print(len(adjusted))
    fig = plt.figure(figsize=(14, 10))


    ax2 = plt.subplot(1, 1, 1)
    #ax2.plot(np.log10(f_p[1:]), np.log10(np.sqrt(psd_p[1:])), label="noise")
    
    filename = 'model_asd_data_final.txt'
    data = np.loadtxt(filename)

    # 假设文件中有两行数据，第一行是频率，第二行是ASD for disp
    frequencies = data[0, :]
    asd_disp = data[1, :]
    ratio,asd_disp = calculate_theoretical_asd(asd_disp, angle_distance,Mw)
   
    # 确保只选择符合要求的数据范围
    mask = (frequencies >= 0.1) & (frequencies <= 10) & (asd_disp >= -15) & (asd_disp <= -6)
    filtered_frequencies = frequencies[mask]
    filtered_asd_disp = asd_disp[mask]
    
    #ratio1 = 10**(Mw-3.55-0.73*np.log10(math.degrees(angle_distance/30.2)))
    ratio1 = 10**(Mw-3.84-1.06*np.log10(math.degrees(angle_distance/30.2)))
    print(ratio,ratio1)
    ratio1 = ratio
    SpectrumVisualizer.plot_comparison(adjusted,adjusted1,ratio,ratio1,Mw)
    
    ax2.plot(np.log10(filtered_frequencies), filtered_asd_disp, 'r',label="seis")
    ax2.set_xlabel('Frequency (Hz)', fontsize=12)
    ax2.set_ylabel('ASD for DISP', fontsize=12)
    ax2.set_xlim(np.log10(0.1), np.log10(10))
    ax2.set_ylim(-15,-6)
    ax2.set_xticks(np.log10([0.1, 0.2, 0.5, 1, 2, 5, 10]))
    ax2.set_xticklabels(['0.1', '0.2', '0.5', '1.0', '2.0', '5.0', '10.0'])
    plt.legend()
    
    input_txt_file_processed = 'asd_25th_percentile.txt'  # 处理后的 TXT 文件路径
    # 读取处理后的 TXT 文件
    with open(input_txt_file_processed, 'r') as file:
        lines = file.readlines()
        frequencies_processed = lines[0].strip().split(' ')
        values_processed = lines[1].strip().split(' ')

    # 将字符串转换为浮点数以便计算
    frequencies_processed = np.array([float(f) for f in frequencies_processed])
    values_processed = np.array([float(v) for v in values_processed])
    ax2.plot(np.log10(frequencies_processed), np.log10(values_processed), label='Original_noise', marker='x')
    
    # 获取两条线的数据
    x1 = np.log10(filtered_frequencies)
    y1 = filtered_asd_disp
    x2 = np.log10(frequencies_processed)
    y2 = np.log10(values_processed)

    # 确保数据排序
    sort_idx1 = np.argsort(x1)
    x1_sorted = x1[sort_idx1]
    y1_sorted = y1[sort_idx1]

    sort_idx2 = np.argsort(x2)
    x2_sorted = x2[sort_idx2]
    y2_sorted = y2[sort_idx2]

    # 生成密集插值点
    x_min = max(x1_sorted.min(), x2_sorted.min())
    x_max = min(x1_sorted.max(), x2_sorted.max())
    x_interp = np.linspace(x_min, x_max, 10000)

    # 插值两条曲线
    y1_interp = np.interp(x_interp, x1_sorted, y1_sorted)
    y2_interp = np.interp(x_interp, x2_sorted, y2_sorted)

    # 寻找交点
    diff = y1_interp - y2_interp
    crossing_indices = np.where(np.diff(np.sign(diff)))[0]

    # 计算精确交点
    crossings = []
    for idx in crossing_indices:
        x0, x1 = x_interp[idx], x_interp[idx+1]
        y0, y1 = diff[idx], diff[idx+1]
        t = -y0 / (y1 - y0)
        x_cross = x0 + t*(x1 - x0)
        crossings.append(x_cross)

    # 确保有两个交点
    if len(crossings) >= 2:
        # 取前两个交点
        x_start, x_end = sorted(crossings[:2])
        
        # 生成填充区域数据
        mask = (x_interp >= x_start) & (x_interp <= x_end)
        x_fill = x_interp[mask]
        y1_fill = y1_interp[mask]
        y2_fill = y2_interp[mask]
        
        # 绘制填充区域
        ax2.fill_between(x_fill, y1_fill, y2_fill, 
                        where=y1_fill > y2_fill, 
                        color='skyblue', alpha=0.4,
                        label='Detection')

    # 修改图例显示
    ax2.legend(loc='best', fontsize=18)

    from matplotlib.ticker import FuncFormatter

    # 创建指数格式转换函数
    def log_label_formatter(val, pos):
        """将log10值转换为10的幂次格式"""
        return r"$10^{%.0f}$" % val

    # 应用在y轴上
    ax2.yaxis.set_major_formatter(FuncFormatter(log_label_formatter))

    # 设置坐标轴标签（保持数据为log10后的值）
    ax2.set_xlabel(r'$\mathrm{Frequency\ [Hz]}$', fontsize=18)
    ax2.set_ylabel(r'$\mathrm{Displacement\ ASD\ [m/\sqrt{Hz}]}$', fontsize=18)
    
    
    # 获取0.2-1Hz范围内的索引
    mask = (frequencies_processed >= 0.2) & (frequencies_processed <= 1)
    noise_mask = (filtered_frequencies >= 0.2) & (filtered_frequencies <= 1)
    #noise_freqs = 
    # 提取0.2-1Hz范围内的数据
    noise_freqs = frequencies_processed[mask]
    noise_amplitudes = values_processed[mask]
    seismic_freqs = filtered_frequencies[noise_mask]
    seismic_amplitudes = (10**np.array(filtered_asd_disp))[noise_mask]

    # 计算0.2-1Hz范围内噪声和地震信号的平均值
    mean_noise = np.mean(noise_amplitudes)
    mean_seismic = np.mean(seismic_amplitudes)

    # 检查地震信号的平均值的50%是否高于噪声信号的平均值
    threshold = mean_seismic * 0.5 > mean_noise

    print(f"Noise Mean Amplitude: {mean_noise}")
    print(f"Seismic Mean Amplitude: {mean_seismic}")
    print(f"Is Seismic Signal Higher than Noise by 80%? {'Yes' if threshold else 'No'}")
    print(Mw)    
    
    plt.title(f'ASD Curve — Seismic Signal (Mw: {Mw}) vs Simulated Noise (Hour: {hour}:00)',fontsize=20)
    #f'earthquake:({earthquake_points[earthquake][0],earthquake_points[earthquake][1]}),magnitude:{Mw},distance:{math.degrees(angle_distance):.2f},lat:{lat},lon:{lon},visible:{threshold}')
    plt.tight_layout()
    plt.show()
    return angle_distance,threshold

def calculate_theoretical_asd(base_asd, angle_distance,Mw):
    """基于振幅比例计算理论ASD"""
    ratio = (30.2 / math.degrees(angle_distance))*(10**((3/2)*(Mw-3.84)))
    print(ratio)
    return ratio,base_asd + np.log10(ratio)  # 因ASD与A²成正比

# 主程序
if __name__ == "__main__":
    nc_file_path = '../mcd_data.nc'
    lat, lon = 21.1, -43.2  # 示例经纬度
    min_value = 0  # 风速最小值
    max_value = 16  # 风速最大值
    u = 0.03  # 图像比例
    integral = 1  # 积分常数（根据需要调整
    ans = {}
    for i in range(len(earthquake_points)):
        Mw = 4.8
        visible = True
        while visible:
            visible = compare_asd(nc_file_path, lat, lon,249,1,i,Mw)
            Mw -= float(0.1)
        ans[earthquake_points[i]] = Mw
    print(ans)
        
    






