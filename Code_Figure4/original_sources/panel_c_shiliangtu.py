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
from pathlib import Path
from matplotlib.gridspec import GridSpec
from scipy.optimize import minimize
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
scale = 20
HERE = Path(__file__).resolve().parent
MCD_FILE = HERE.parent / "mcd_data.nc"

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
freq_list2 = [0.2,1]
SEISMIC_PARAMS = {
    'channel': 'Z',          # 处理通道
    'spec_window': 512,      # 频谱分析窗口
    'noverlap': 256,         # 频谱重叠样本
    'clim': (-220, -170),    # 颜色映射范围
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
        nc_path = Path(nc_path).resolve()
        original_directory = Path.cwd()
        try:
            os.chdir(nc_path.parent)
            self.dataset = nc.Dataset(nc_path.name)
        finally:
            os.chdir(original_directory)
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
    cbar.set_label('Wind Speed (m/s)', fontdict={'family': 'Times New Roman', 'fontsize': 14})
    for label in cbar.ax.get_yticklabels():
        label.set_family('Times New Roman')
        label.set_fontsize(12)
    
    f1 = 16
    f2 = 16
    
    # 显示颜色条
    plt.xlabel("hour")
    plt.ylabel("sol")
    ticks = np.arange(0, 48, 2)
    plt.xticks(ticks=ticks, labels=[i-48 if i > 47 else i for i in ticks])
    plt.xlabel('LMST hour', fontdict={'family': 'Arial', 'fontsize': f2})
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
    data = np.asarray(data)
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
        self.interpolators = [[] for i in range(3)]
        for i in range(len(self.freq_breaks)-1):
            for j in range(3):
                low = self.freq_breaks[i]
                high = self.freq_breaks[i+1]
                data = np.loadtxt(HERE / '17' / f'wind_speed_earthquake_signals_{low}-{high}_220_{j}.txt')
                self.interpolators[j].append(
                    interp1d(data[:,0], data[:,1], fill_value="extrapolate")
                )
        self.interpolators2 = [[] for i in range(3)]
        for j in range(3):
            low = 0.1
            high = 1
            data = np.loadtxt(HERE / '17' / f'wind_speed_earthquake_signals_{low}-{high}_220_{j}.txt')
            self.interpolators2[j].append(
                interp1d(data[:,0], data[:,1], fill_value="extrapolate")
            )

# 增强版地震信号修正器（包含时间截取）
# --------------------------
class EnhancedSeismicAdjuster(SeismicAdjuster):
    """包含时间对齐的增强修正器"""
    
    def align_daily_signal(self, stream: Stream) -> Tuple[np.ndarray, float]:
        """对齐到火星日00:00-24:00"""
        # 获取信号基本信息
        tr = stream.select(channel=f"BH{SEISMIC_PARAMS['channel']}")[0]
        delta = tr.stats.delta
        starttime_lmst = c2lmst(tr.stats.starttime)
        start_time_utc = tr.stats.starttime
        # 计算当日火星时间范围
        # 创建一个lmst对象代表我们感兴趣的起始时间(当天的00:00 LMST)
        target_start_lmst = lmst(sol=starttime_lmst.sol+1, hour=0, minute=0, sec=0)
        target_start_utc = target_start_lmst.conv2utc()

        # 同样创建一个代表当天结束时间(24:00 LMST)的对象
        target_end_lmst = lmst(sol=starttime_lmst.sol+1, hour=24, minute=0, sec=0)
        target_end_utc = target_end_lmst.conv2utc()

        # 根据目标时间转换为样本索引
        start_idx = int((target_start_utc - start_time_utc) / delta)
        end_idx = int((target_end_utc - start_time_utc) / delta)
        
        # 截取并返回
        return tr.data[start_idx:end_idx], delta

    def corrected_adjust(self, ref_stream: Stream,
                        ref_wind: np.ndarray, target_wind: np.ndarray, 
                        gap: float) -> np.ndarray:
        """多频段修正"""
        ref_signal, delta = self.align_daily_signal(ref_stream)
        
        adjusted_segments = [[] for i in range(len(freq_list)-1)]
        samples_per_segment = int(3600*gap / delta)
        
        # 每个时间窗口独立处理所有频段
        for seg in range(int(6/gap)):
            start = seg * samples_per_segment
            end = (seg + 1) * samples_per_segment
            
            # 初始化该时段的修正信号
            seg_adjusted = []
            # 遍历所有频段插值器
            for i, interp in enumerate(self.interpolators[0]):
                ref_factor = 10**interp(ref_wind[int(seg*len(ref_wind)/int(24/gap))])
                target_factor = 10**interp(target_wind[int(seg*len(target_wind)/int(24/gap))])
                seg_adjusted = list(ref_signal[start:end] * (target_factor / ref_factor))
                adjusted_segments[i] += seg_adjusted
        #print(len(adjusted_segments))
        for seg in range(int(6/gap),int(17/gap)):
            start = seg * samples_per_segment
            end = (seg + 1) * samples_per_segment
            
            # 初始化该时段的修正信号
            seg_adjusted = []
            # 遍历所有频段插值器
            for i, interp in enumerate(self.interpolators[1]):
                ref_factor = 10**interp(ref_wind[int(seg*len(ref_wind)/int(24/gap))])
                target_factor = 10**interp(target_wind[int(seg*len(target_wind)/int(24/gap))])
                seg_adjusted = list(ref_signal[start:end] * (target_factor / ref_factor))
                adjusted_segments[i] += seg_adjusted
        #print(len(adjusted_segments))
        for seg in range(int(17/gap),int(24/gap)):
            start = seg * samples_per_segment
            end = (seg + 1) * samples_per_segment
            
            # 初始化该时段的修正信号
            seg_adjusted = []
            # 遍历所有频段插值器
            for i, interp in enumerate(self.interpolators[2]):
                ref_factor = 10**interp(ref_wind[int(seg*len(ref_wind)/int(24/gap))])
                target_factor = 10**interp(target_wind[int(seg*len(target_wind)/int(24/gap))])
                seg_adjusted = list(ref_signal[start:end] * (target_factor / ref_factor))
                adjusted_segments[i] += seg_adjusted
        #print(len(adjusted_segments))
        return adjusted_segments
    
    def corrected_adjust2(self, ref_stream: Stream,
                        ref_wind: np.ndarray, target_wind: np.ndarray, 
                        gap: float) -> np.ndarray:
        """多频段修正"""
        ref_signal, delta = self.align_daily_signal(ref_stream)
        
        adjusted_segments = [[] for i in range(len(freq_list2)-1)]
        samples_per_segment = int(3600*gap / delta)
        
        # 每个时间窗口独立处理所有频段
        for seg in range(int(6/gap)):
            start = seg * samples_per_segment
            end = (seg + 1) * samples_per_segment
            
            # 初始化该时段的修正信号
            seg_adjusted = []
            # 遍历所有频段插值器
            for i, interp in enumerate(self.interpolators2[0]):
                ref_factor = 10**interp(ref_wind[int(seg*len(ref_wind)/int(24/gap))])
                target_factor = 10**interp(target_wind[int(seg*len(target_wind)/int(24/gap))])
                seg_adjusted = list(ref_signal[start:end] * (target_factor / ref_factor))
                print(target_factor / ref_factor)
                adjusted_segments[i] += seg_adjusted
        #print(len(adjusted_segments))
        for seg in range(int(6/gap),int(17/gap)):
            start = seg * samples_per_segment
            end = (seg + 1) * samples_per_segment
            
            # 初始化该时段的修正信号
            seg_adjusted = []
            # 遍历所有频段插值器
            for i, interp in enumerate(self.interpolators2[1]):
                ref_factor = 10**interp(ref_wind[int(seg*len(ref_wind)/int(24/gap))])
                target_factor = 10**interp(target_wind[int(seg*len(target_wind)/int(24/gap))])
                seg_adjusted = list(ref_signal[start:end] * (target_factor / ref_factor))
                adjusted_segments[i] += seg_adjusted
        #print(len(adjusted_segments))
        for seg in range(int(17/gap),int(24/gap)):
            start = seg * samples_per_segment
            end = (seg + 1) * samples_per_segment
            
            # 初始化该时段的修正信号
            seg_adjusted = []
            # 遍历所有频段插值器
            for i, interp in enumerate(self.interpolators2[2]):
                ref_factor = 10**interp(ref_wind[int(seg*len(ref_wind)/int(24/gap))])
                target_factor = 10**interp(target_wind[int(seg*len(target_wind)/int(24/gap))])
                seg_adjusted = list(ref_signal[start:end] * (target_factor / ref_factor))
                adjusted_segments[i] += seg_adjusted
        print(len(adjusted_segments[0]))
        return adjusted_segments
    
ref_day=641
SEISMIC_DATA_DIR = Path(r"C:\Data and Code\File 6—Data_Insight_Seismic")

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
]

class SpectrumVisualizer:
    """频谱对比可视化工具"""
    
    @staticmethod
    def plot_comparison(target_signals1: list,target_signals2,target_signals3,hour,ratio,ratio1,ratio2,Mw, sample_rate: float = 20.0):
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
        seis_data = np.load(HERE / 'event_1133_spectrogram.npz')
        seis_f = seis_data['frequencies']  # 原始地震频率轴（Hz）
        seis_t = seis_data['times']        # 原始地震时间轴（秒）
        seis_Sxx = seis_data['power']      # 原始地震功率数据（dB/Hz）
        print(seis_Sxx.shape) 
        def overlay_seismic(sim_t, sim_f, sim_Sxx):
            """在指定小时位置叠加地震数据"""
            # 1. 计算地震数据的时间偏移
            target_origin = hour * 3600 + 300  # 转换为秒
            
            # 2. 计算地震数据在模拟时间轴中的位置
            seis_start = target_origin + seis_t[0]
            seis_end = target_origin + seis_t[-1]
            
            # 3. 确定叠加范围（仅处理重叠部分）
            overlap_start = max(sim_t[0], seis_start)
            overlap_end = min(sim_t[-1], seis_end)
            
            if overlap_start > overlap_end:
                print(1)
                return sim_t, sim_f, sim_Sxx  # 无重叠时返回原数据
            
            # 4. 时间轴索引计算
            seis_idx_start = np.searchsorted(seis_t, overlap_start - target_origin)
            seis_idx_end = np.searchsorted(seis_t, overlap_end - target_origin)
            sim_idx_start = np.searchsorted(sim_t, overlap_start)
            sim_idx_end = np.searchsorted(sim_t, overlap_end)
            
            # 5. 频率轴对齐（地震数据截断到模拟频率范围）
            freq_mask = (seis_f >= sim_f[0]) & (seis_f <= sim_f[-1])
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
        
        
        # 计算模拟频谱2
        # 处理模拟信号
        all_freqs = []
        all_Sxx = []
        for sig_pair in target_signals1:
            for i in range(len(sig_pair)):
                sig = sig_pair[i]
                sig = np.array(sig)
                f, t, Sxx = SpectrumVisualizer._compute_spectrogram(sig, sample_rate)
                mask = (f >= freq_list[i]) & (f < freq_list[i + 1])  # 根据当前txt的频率范围生成mask
                all_freqs.append(f[mask])
                all_Sxx.append(Sxx[mask, :])
        
        # 合并频谱
        merged_freq = np.concatenate(all_freqs)
        merged_Sxx = np.vstack(all_Sxx)
        sorted_idx = np.argsort(merged_freq)
        merged_freq = merged_freq[sorted_idx]
        merged_Sxx = merged_Sxx[sorted_idx, :]
        
        # 在合并频谱后添加：
        merged_t, merged_f, merged_Sxx = overlay_seismic(t, merged_freq, merged_Sxx)
        
        # 高斯平滑（保持原有逻辑）
        merged_Sxx = gaussian_filter1d(merged_Sxx, sigma=0.1, axis=0)
        
        # 定义目标时间范围（凌晨3:50到4:30）
        start_time = hour * 3600  # 4:00 AM in seconds
        end_time = hour * 3600 + 20 * 60     # 4:30 AM in seconds
        
        # 找到目标时间范围内的索引
        time_mask = (merged_t >= start_time) & (merged_t <= end_time)
        merged_t = merged_t[time_mask]
        merged_Sxx = merged_Sxx[:, time_mask]
        print(merged_Sxx.shape)
        
        all_waveforms = []
        for sig_pair in target_signals2:
            for i in range(0,1):
                sig = np.array(sig_pair[i])
                # 对信号求导
                
                all_waveforms.append(sig)
        key_times = {
            "P Wave": (start_time + 229+300)/3600,  
            "S Wave": (start_time + 405+300)/3600  
        }
        # 叠加所有信号
        combined_waveform = np.sum(all_waveforms, axis=0)
        #combined_waveform = target_signals1[0][10]
        #combined_waveform = np.gradient(combined_waveform, 1/sample_rate)
        print(len(combined_waveform))
        # 绘制波形图
        waveform = combined_waveform[int((start_time+50)*20):int((end_time+50)*20)]
        print(len(waveform))
        waveform_t = np.linspace(start_time*20, end_time*20, len(waveform))
        
        loaded_data_detrended = np.load(HERE / 'data_detrended_1133.npy')
        split_point = 405 * 20 +50*20
        part1 = loaded_data_detrended[:split_point]
        part2 = loaded_data_detrended[split_point:]

        # 应用不同的比例因子
        print(ratio,"我")
        part1_scaled = part1 * ratio
        part2_scaled = part2 * ratio

        # 合并两个处理后的数组
        scaled_data_detrended = np.concatenate((part1_scaled, part2_scaled))
        
        waveform += scaled_data_detrended
        waveform_pre = combined_waveform[int(start_time*20):int((start_time+50)*20)]
        waveform = np.concatenate((waveform_pre, waveform))
        waveform = waveform[:int(end_time-start_time)*20]

        #ax0.set_title(f"Waveform Overlay with Seismic Signal (hour:{hour:02d}:00 , Mw:{Mw})", fontsize=20)
        start_time_hours = hour-1 + 60 / 60  # 4:00 AM in hours
        end_time_hours = hour + 20 / 60     # 4:30 AM in hours
         # 反变换
         # 将 dB/Hz 转换回幅度平方
        ax0.set_xlim(start_time_hours, end_time_hours)
        
        # 设置x轴刻度和标签
        xticks = np.arange(start_time_hours, end_time_hours + 0.1, float(10/60))  # 每10分钟一标注
        print(xticks)
        
        ax0.set_xticks(xticks)
        ax0.set_xticklabels([f"{int(h):02d}:{int((h % 1) * 61):02d}" for h in xticks])
        ax0.plot(waveform_t / 72000, waveform, color='b')
        ax0.axvline(x=float((hour*3600 + 229+50)/3600), color='g', linestyle=':', linewidth=2, label='P Wave Arrival')
        ax0.axvline(x=float((hour*3600 + 405+50)/3600), color='orange', linestyle=':', linewidth=2, label='S Wave Arrival')
        #ax0.legend(loc='upper right',fontsize = scale-4)
        ax0.set_ylabel("Amplitude(m/s)", fontsize=scale)
        # 设置 Y 轴 offset text（即 1e-9）的字体大小
        ax0.yaxis.offsetText.set_fontsize(scale-2)  # 或 set_size()
        # 设置刻度标签的字体大小
        ax0.tick_params(axis='both', which='major', labelsize=scale-2)  # 控制 x 和 y 的主刻度标签
        
        
        
        all_waveforms = []
        all_freqs = []
        all_Sxx = []
        for sig_pair in target_signals1:
            for i in range(0,17):
                sig = np.array(sig_pair[i])
                # 对信号求导
                combined_waveform = sig
                waveform = combined_waveform[int((start_time+50)*20):int((end_time+50)*20)]
                print(len(waveform))
                waveform_t = np.linspace(start_time*20, end_time*20, len(waveform))
                
                loaded_data_detrended = np.load(HERE / 'data_detrended_1133_disp.npy')
                split_point = 405 * 20 +300*20
                part1 = loaded_data_detrended[:split_point]
                part2 = loaded_data_detrended[split_point:]

                # 应用不同的比例因子
                part1_scaled = part1 * ratio
                part2_scaled = part2 * ratio

                # 合并两个处理后的数组
                scaled_data_detrended = np.concatenate((part1_scaled, part2_scaled))
                print(len(scaled_data_detrended))
                print('张雨晨猪猪猪')
                print(np.mean(waveform))
                waveform += scaled_data_detrended
                
                waveform_pre = combined_waveform[int(start_time*20):int((start_time+50)*20)]
                waveform = np.concatenate((waveform_pre, waveform))
                waveform = waveform[:int(end_time-start_time)*20]
                
                waveform_pre =combined_waveform[int((start_time-20)*20):int(start_time*20)]
                waveform_pro =combined_waveform[int(end_time*20):int((end_time+20)*20)]
                waveform1 = np.concatenate((waveform_pre, waveform))
                waveform1 = np.concatenate((waveform1, waveform_pro))
                f, t, Sxx = SpectrumVisualizer._compute_spectrogram(waveform1, fs=sample_rate)
                mask = (f >= 0.9*freq_list[i]) & (f < 1.1*freq_list[i + 1])   # 根据当前txt的频率范围生成mask
                all_freqs.append(f[mask])
                all_Sxx.append(Sxx[mask, :])
        
        
        # 合并频谱
        merged_freq = np.concatenate(all_freqs)
        merged_Sxx = np.vstack(all_Sxx)
        sorted_idx = np.argsort(merged_freq)
        merged_freq = merged_freq[sorted_idx]
        merged_Sxx = merged_Sxx[sorted_idx, :]
        # 叠加所有信号
        #combined_waveform = np.sum(all_waveforms, axis=0)
        #combined_waveform = target_signals1[0][10]
        #combined_waveform = np.gradient(combined_waveform, 1/sample_rate)
        
        #waveform2 = combined_waveform[int((start_time-20)*20):int((end_time+20)*20)]
        #print(waveform1[100:200]==waveform2[100:200])
        # 绘制频谱图
        #print(merged_t)
        merged_t = t+hour*3600 -20
        SpectrumVisualizer._plot_spectrum(ax1, merged_t, merged_freq, merged_Sxx, hour)
        #ax1.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    def _get_freq_mask(self, f: np.ndarray) -> np.ndarray:
        """根据当前插值器的频率范围生成mask"""
        low = self.freq_breaks[i]
        high = self.freq_breaks[i+1]
        return (f >= low) & (f <= high)    
    @staticmethod
    def _compute_spectrogram(data: np.ndarray, fs: float) -> Tuple[np.ndarray, ...]:
        """计算频谱图"""
        return spectrogram(data, fs=fs,
                          window=windows.hamming(SEISMIC_PARAMS['spec_window']),
                          nperseg=SEISMIC_PARAMS['spec_window'],
                          noverlap=SEISMIC_PARAMS['noverlap'],
                          scaling='density')

    @staticmethod
    def _plot_spectrum(ax: plt.Axes, t: np.ndarray, f: np.ndarray, Sxx: np.ndarray,hour):
        """绘制单个频谱图"""
        db = 10 * np.log10(Sxx[1:])  # 排除0Hz
        pc = ax.pcolormesh(t/3600, np.log10(f[1:]), db, cmap='plasma', shading='auto')
        
        # 定义目标时间范围（凌晨3:50到4:30）
        start_time_hours = hour-1 + 60 / 60  # 3:50 AM in hours
        end_time_hours = hour + 20 / 60     # 4:30 AM in hours
        
        # 设置x轴范围
        ax.set_xlim(start_time_hours, end_time_hours)
        
        # 设置x轴刻度和标签
        xticks = np.arange(start_time_hours, end_time_hours + 0.1, float(10/60))  # 每10分钟一标注
        #print(xticks)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{int(h):02d}:{int((h % 1) * 61):02d}" for h in xticks])
        
        ax.set_ylim(np.log10([SEISMIC_PARAMS['freq_ticks'][0], SEISMIC_PARAMS['freq_ticks'][-1]]))
        ax.set_yticks(np.log10(SEISMIC_PARAMS['freq_ticks']))
        ax.set_yticklabels([str(f) for f in SEISMIC_PARAMS['freq_ticks']])
        # 设置刻度标签的字体大小
        ax.tick_params(axis='both', which='major', labelsize=scale-2)  # 控制 x 和 y 的主刻度标签
        ax.set_xlabel("Local Hour (h)", fontsize=scale)
        ax.set_ylabel("Frequency (Hz)", fontsize=scale)
        #ax.set_title(title, fontsize=20)
        pc.set_clim(SEISMIC_PARAMS['clim'])
        
        # 添加颜色条（移动到右侧并缩小）
        cbar = plt.colorbar(pc, ax=ax, orientation='vertical', pad=0.04, shrink=0.6)
        cbar.set_label('Power Spectral Density (dB)', fontsize=scale, labelpad=15)
        cbar.ax.tick_params(labelsize=scale-2)

def compare_asd(nc_file_path, lat, lon,day,hour,earthquake,Mw):
    earthquake_lat = earthquake_points[earthquake][0]
    earthquake_lon = earthquake_points[earthquake][1]
    angle_distance = haversine((earthquake_lat,earthquake_lon), (lat,lon))
    
    Evt_name = str(ref_day).zfill(4)
    main_file = SEISMIC_DATA_DIR / "Data_Step5_Rotate_DISP" / f"LMST_{Evt_name}.mseed"
    st_disp = read(str(main_file))
    channel = 'Z'
    ref_tr = st_disp.select(channel=f'BH{channel}')
    D_vel = ref_tr[0].data
    wind_proc = WindProcessor(str(MCD_FILE))#PCM风速
    adjuster = EnhancedSeismicAdjuster(ref_day, day)
    ref_stream = read(str(main_file))
    velocity_file = SEISMIC_DATA_DIR / "Data_Step5_Rotate" / f"LMST_{ref_day:04d}.mseed"
    ref_stream2 = read(str(velocity_file))
    ref_stream2.filter('bandpass', freqmin=0.2, freqmax=1.0, corners=4, zerophase=True)
    ref_stream3 = read(str(velocity_file))
    ssst = ref_stream3.select(channel='BHZ')
    sst = ref_stream2.select(channel='BHZ')
    u_ref, v_ref = wind_proc.get_wind_profile(4.5, 135.6)
    u_target, v_target = wind_proc.get_wind_profile(lat, lon)
    # 执行信号修正
    adjusted = adjuster.corrected_adjust(
        ref_stream,
        np.sqrt(u_ref**2 + v_ref**2)[(ref_day-113)*48:(ref_day-113+1)*48],
        np.sqrt(u_target**2 + v_target**2)[(day-113)*48:(day-113+1)*48],
        0.5
    )#它有17个，分别是按照17个频段进行拟合的
    adjusted2 = adjuster.corrected_adjust2(
        sst,
        np.sqrt(u_ref**2 + v_ref**2)[(ref_day-113)*48:(ref_day-113+1)*48],
        np.sqrt(u_target**2 + v_target**2)[(day-113)*48:(day-113+1)*48],
        0.5
    )#它有1个，分别是按照1个频段进行拟合的(早中晚）
    
    print('开始')
    print(np.mean(adjusted2))
    print('结束')
    
    adjusted3 = adjuster.corrected_adjust2(
        ssst,
        np.sqrt(u_ref**2 + v_ref**2)[(ref_day-113)*48:(ref_day-113+1)*48],
        np.sqrt(u_target**2 + v_target**2)[(day-113)*48:(day-113+1)*48],
        0.5
    )#它有1个，分别是按照1个频段进行拟合的(早中晚）
    wind_speeds = np.sqrt(u_target**2 + v_target**2)[(day-113)*48:(day-113+1)*48]
    dt = 0.05
    Fs = 1/dt
    wind = wind_speeds[2*hour]
    # 合并插值结果，去掉重复的 b 点
    
    Fs = 1/dt
    segment_length = int(25.6 / 0.05)  # 512 points
    noverlap = int(segment_length * 0.5)
                                
    f_p = []
    psd_p = []
    freq_list = [0.05, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8]
    for i in range(len(adjusted)):
        data = adjusted[i][int(len(adjusted[i])*(hour-0.5)/24):int(len(adjusted[i])*(hour+0.5)/24)]
        f_pi, psd_pi = calc_psd(data, Fs)
        mask = (f_pi >= freq_list[i]) & (f_pi < freq_list[i + 1])
        
        # 截取频率和PSD
        f_segment = f_pi[mask]
        psd_segment = psd_pi[mask]
        
        # 合并到总的 f_p 和 psd_p
        f_p.extend(f_segment)
        psd_p.extend(psd_segment)
    f_p = np.array(f_p)
    psd_p = np.array(psd_p)
    
    
    #ax2 = plt.subplot(1, 1, 1)
    #ax2.plot(np.log10(f_p[1:]), np.log10(np.sqrt(psd_p[1:])), label="noise")
    
    filename = HERE / 'model_asd_data_final.txt'
    data = np.loadtxt(filename)

    # 假设文件中有两行数据，第一行是频率，第二行是ASD for disp
    frequencies = data[0, :]
    asd_disp = data[1, :]
    ratio,asd_disp = calculate_theoretical_asd(asd_disp, angle_distance,Mw)
    # 确保只选择符合要求的数据范围
    mask = (frequencies >= 0.1) & (frequencies <= 10) & (asd_disp >= -15) & (asd_disp <= -6)
    filtered_frequencies = frequencies[mask]
    filtered_asd_disp = asd_disp[mask]
    
    ratio1 = 10**(Mw-3.84-0.73*np.log10(math.degrees(angle_distance/30.2)))
    ratio2 = 10**(Mw-3.84-1.06*np.log10(math.degrees(angle_distance/30.2)))
    ratio1 = ratio2
    print(ratio1,ratio2)
    
    SpectrumVisualizer.plot_comparison([adjusted],[adjusted2],[adjusted3],hour,ratio,ratio1,ratio2,Mw)
    
    
    input_txt_file_processed = HERE / 'processed_noise.txt'  # 处理后的 TXT 文件路径
    # 读取处理后的 TXT 文件
    with open(input_txt_file_processed, 'r') as file:
        lines = file.readlines()
        frequencies_processed = lines[0].strip().split(',')
        values_processed = lines[1].strip().split(',')

    # 将字符串转换为浮点数以便计算
    frequencies_processed = np.array([float(f) for f in frequencies_processed])
    values_processed = np.array([float(v) for v in values_processed])
    #ax2.plot(np.log10(frequencies_processed), np.log10(values_processed), label='Original_noise', marker='x')

    
    
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

    # 检查地震信号的平均值的80%是否高于噪声信号的平均值
    threshold = mean_seismic * 0.5 > mean_noise

    print(f"Noise Mean Amplitude: {mean_noise}")
    print(f"Seismic Mean Amplitude: {mean_seismic}")
    print(f"Is Seismic Signal Higher than Noise by 80%? {'Yes' if threshold else 'No'}")
    print(Mw)    
    
    #plt.title(f'earthquake:({earthquake_points[earthquake][0],earthquake_points[earthquake][1]}),magnitude:{Mw},distance:{math.degrees(angle_distance):.2f},lat:{lat},lon:{lon},visible:{threshold}')

    return threshold

def calculate_theoretical_asd(base_asd, angle_distance,Mw):
    """基于振幅比例计算理论ASD"""
    ratio = (30.2 / math.degrees(angle_distance))*(10**((3/2)*(Mw-3.41)))
    print(ratio)
    return ratio,base_asd + np.log10(ratio)  # 因ASD与A²成正比

# 主程序
if __name__ == "__main__":
    nc_file_path = str(MCD_FILE)
    lat, lon = 21.1, -43.2  # 示例经纬度
    min_value = 0  # 风速最小值
    max_value = 16  # 风速最大值
    u = 0.03  # 图像比例
    integral = 1  # 积分常数（根据需要调整
    ans = {}
    for i in range(len(earthquake_points)):
        Mw = 2.9
        visible = True
        while visible:
            visible = compare_asd(nc_file_path, lat, lon,434,1,i,Mw)
            Mw -= float(0.1)
        ans[earthquake_points[i]] = Mw
    print(ans)
        
    




