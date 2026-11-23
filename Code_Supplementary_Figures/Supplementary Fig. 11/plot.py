import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from obspy import read, UTCDateTime
from scipy.signal import welch
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "on_deck_sp"
OUTPUT = ROOT / "output"
sys.path.insert(0, str(ROOT / "common"))
from si_tools import resize_like_si

class lmst(object):
    def __init__(self, sol=0, hour=0, minute=0, sec=0):
        self.sol = sol
        self.hour = hour
        self.min = minute
        self.sec = sec

    def __str__(self):
        return ('Sol-%.5dM%.2d:%.2d:%05.2f') % (self.sol, self.hour, self.min, self.sec)

    __repr__ = __str__

    def conv2utc(self):
        ref = UTCDateTime("2018-11-26T05:10:50.336037Z")
        sol1_start = UTCDateTime("2018-11-27T05:50:25.580014Z")
        sol2_start = UTCDateTime("2018-11-28T06:30:00.823990Z")
        sec_m = sol2_start - sol1_start - 0.000005
        sec_e = 86400
        dur_m = self.sol * 24 * 3600 + self.hour * 3600 + self.min * 60 + self.sec
        dur_e = dur_m * sec_m / sec_e
        return ref + dur_e

def c2utc(a):
    bbcc = lmst()
    bbcc.sol = int(a[:5])
    bbcc.hour = int(a[6:8])
    bbcc.min = int(a[9:11])
    bbcc.sec = float(a[12:])
    return bbcc.conv2utc()

def calc_asd(data, fs):
    nfft = 2 ** int(np.ceil(np.log2(len(data))))
    freq, psd = welch(data, fs=fs, nperseg=nfft//4, window='hann', scaling='density')
    return freq, np.sqrt(psd) * (2 * np.pi * freq)  # 转换到位移谱密度

def plot_combined_asd_heatmap(base_path, days):
    """EHZ通道分析函数"""
    # 初始化参数
    freq_min, freq_max = 0.1, 10
    asd_min, asd_max = 1e-11, 1e-2
    interval_minutes = 10
    
    # 创建频率和ASD的网格
    freq_bins = np.logspace(np.log10(freq_min), np.log10(freq_max), 200)
    asd_bins = np.logspace(np.log10(asd_min), np.log10(asd_max), 200)
    
    combined_heatmap_data = np.zeros((len(asd_bins)-1, len(freq_bins)-1))
    
    # 存储每个频率区间内的ASD值
    asd_per_freq_bin = [[] for _ in range(len(freq_bins)-1)]
    
    for day in days:
        filename = f"{base_path}{day}.mseed"
        try:
            st = read(filename)
            st = st.select(channel="EHZ")
            print(f"成功读取文件: {filename}")
        except Exception as e:
            print(f"文件读取失败: {e}")
            continue

        if len(st) == 0:
            print("错误：未找到EHZ通道数据")
            continue

        # 合并数据段
        st.merge(method=1, fill_value=0)
        tr = st[0]
        Fs = tr.stats.sampling_rate
        start_time = tr.stats.starttime
        end_time = tr.stats.endtime
        total_duration = end_time - start_time
        num_intervals = int(total_duration // (interval_minutes * 60))
        
        for i in range(num_intervals):
            current_start_time = start_time + i * interval_minutes * 60
            current_end_time = current_start_time + interval_minutes * 60
            interval_tr = tr.slice(starttime=current_start_time, endtime=current_end_time)
            
            if len(interval_tr.data) == 0:
                continue
                
            freq, asd = calc_asd(interval_tr.data, Fs)
            
            # 收集每个频率区间内的ASD值
            for j in range(len(freq)):
                if freq_min <= freq[j] <= freq_max:
                    # 找到频率所属的bin索引
                    freq_idx = np.searchsorted(freq_bins, freq[j], side='right') - 1
                    if 0 <= freq_idx < len(asd_per_freq_bin):
                        asd_per_freq_bin[freq_idx].append(asd[j])
            
            # 统计二维直方图（热力图）
            hist, _, _ = np.histogram2d(freq, asd, bins=[freq_bins, asd_bins])
            combined_heatmap_data += hist.T

    # 计算每个频率区间内的25%分位数
    percentile_25 = []
    for asd_values in asd_per_freq_bin:
        if len(asd_values) > 0:
            # 计算每个频率区间的25%分位数
            p25 = np.percentile(asd_values, 25)
            percentile_25.append(p25)
        else:
            percentile_25.append(np.nan)  # 无数据时设为nan
    
    # 计算频率bin的中心值
    freq_centers = [(freq_bins[i] + freq_bins[i+1]) / 2 for i in range(len(freq_bins)-1)]
    
    # 创建有效数据的索引
    valid_indices = [i for i, p in enumerate(percentile_25) if not np.isnan(p)]
    freq_valid = [freq_centers[i] for i in valid_indices]
    asd_valid = [percentile_25[i] for i in valid_indices]
    
    # 对ASD值进行变换：除以(2*pi*freq)**2
    # 这相当于恢复到位移谱密度的原始形式
    transformed_asd = []
    for freq_val, asd_val in zip(freq_valid, asd_valid):
        if freq_val > 0:  # 避免除以零
            # 注意：原始ASD是原始PSD的平方根乘以(2πf)，所以这里相当于除以(2πf)
            transformed_val = asd_val / ((2 * np.pi * freq_val) ** 2)
            transformed_asd.append(transformed_val)
        else:
            transformed_asd.append(np.nan)
    
    # 保存25%分位数数据到文本文件（两行格式）
    with (DATA / "asd_25th_percentile.txt").open("w") as f:
        # 第一行：所有频率值，空格分隔
        f.write(" ".join([str(freq) for freq in freq_valid]) + "\n")
        # 第二行：变换后的ASD值，空格分隔
        f.write(" ".join([str(asd) for asd in transformed_asd]) + "\n")
    print("25%分位数数据已保存到 asd_25th_percentile.txt (两行格式, 已变换)")
    
    # The SI panel was exported from a 15 x 6 inch canvas.
    plt.figure(figsize=(15, 6))
    
    # 绘制热力图
    heatmap = plt.pcolormesh(
        freq_bins, asd_bins, combined_heatmap_data + 0.1,
        cmap='inferno', shading='auto', norm=LogNorm(vmin=1)
    )
    plt.colorbar(heatmap, label='Count')
    
    # 绘制25%分位数曲线（使用原始ASD值）
    plt.plot(freq_centers, percentile_25, color='cyan', linestyle='--', 
             linewidth=2, label='25th Percentile per Frequency Bin')
    plt.legend()
    
    plt.title(f"Combined ASD Heatmap (Every {interval_minutes} Minutes) - Days {', '.join(days)}")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Displacement ASD [m/√Hz]")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(freq_min, freq_max)
    plt.ylim(asd_min, asd_max)
    plt.yticks([1e-11, 1e-9, 1e-7, 1e-5, 1e-3])
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.047, top=0.938)
    OUTPUT.mkdir(exist_ok=True)
    raw = OUTPUT / "_Supplementary Fig. 11.raw.png"
    plt.savefig(raw, dpi=100)
    plt.savefig(OUTPUT / "Supplementary Fig. 11.pdf")
    plt.close()
    resize_like_si(raw, 11, OUTPUT / "Supplementary Fig. 11.png")
    raw.unlink()

# ========== 使用示例 ==========
if __name__ == "__main__":
    days = ['010', '014', '016', '018', '020', '021']
    base_path = str(DATA / "LMST_")
    plot_combined_asd_heatmap(base_path, days)
