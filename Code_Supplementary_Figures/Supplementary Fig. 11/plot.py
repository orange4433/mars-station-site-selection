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
    return freq, np.sqrt(psd) * (2 * np.pi * freq)  # Convert to displacement spectral density.

def plot_combined_asd_heatmap(base_path, days):
    """Analyze the EHZ channel."""
    # Initialize parameters.
    freq_min, freq_max = 0.1, 10
    asd_min, asd_max = 1e-11, 1e-2
    interval_minutes = 10
    
    # Create the frequency-ASD grid.
    freq_bins = np.logspace(np.log10(freq_min), np.log10(freq_max), 200)
    asd_bins = np.logspace(np.log10(asd_min), np.log10(asd_max), 200)
    
    combined_heatmap_data = np.zeros((len(asd_bins)-1, len(freq_bins)-1))
    
    # Store ASD values in each frequency bin.
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

        # Merge data segments.
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
            
            # Collect ASD values in each frequency bin.
            for j in range(len(freq)):
                if freq_min <= freq[j] <= freq_max:
                    # Find the frequency-bin index.
                    freq_idx = np.searchsorted(freq_bins, freq[j], side='right') - 1
                    if 0 <= freq_idx < len(asd_per_freq_bin):
                        asd_per_freq_bin[freq_idx].append(asd[j])
            
            # Compute the two-dimensional histogram (heatmap).
            hist, _, _ = np.histogram2d(freq, asd, bins=[freq_bins, asd_bins])
            combined_heatmap_data += hist.T

    # Calculate the 25th percentile in each frequency bin.
    percentile_25 = []
    for asd_values in asd_per_freq_bin:
        if len(asd_values) > 0:
            # Calculate the 25th percentile for each frequency bin.
            p25 = np.percentile(asd_values, 25)
            percentile_25.append(p25)
        else:
            percentile_25.append(np.nan)  # Use NaN when no data are available.
    
    # Calculate frequency-bin centers.
    freq_centers = [(freq_bins[i] + freq_bins[i+1]) / 2 for i in range(len(freq_bins)-1)]
    
    # Identify valid data.
    valid_indices = [i for i, p in enumerate(percentile_25) if not np.isnan(p)]
    freq_valid = [freq_centers[i] for i in valid_indices]
    asd_valid = [percentile_25[i] for i in valid_indices]
    
    # Transform ASD by dividing by (2*pi*freq)**2.
    # This restores the original displacement spectral density.
    transformed_asd = []
    for freq_val, asd_val in zip(freq_valid, asd_valid):
        if freq_val > 0:  # Avoid division by zero.
            # Original ASD equals sqrt(PSD) times (2*pi*f), so this effectively divides by (2*pi*f).
            transformed_val = asd_val / ((2 * np.pi * freq_val) ** 2)
            transformed_asd.append(transformed_val)
        else:
            transformed_asd.append(np.nan)
    
    # Save the 25th-percentile data as two rows of text.
    with (DATA / "asd_25th_percentile.txt").open("w") as f:
        # First row: space-separated frequency values.
        f.write(" ".join([str(freq) for freq in freq_valid]) + "\n")
        # Second row: space-separated transformed ASD values.
        f.write(" ".join([str(asd) for asd in transformed_asd]) + "\n")
    print("25%分位数数据已保存到 asd_25th_percentile.txt (两行格式, 已变换)")
    
    # The SI panel was exported from a 15 x 6 inch canvas.
    plt.figure(figsize=(15, 6))
    
    # Plot the heatmap.
    heatmap = plt.pcolormesh(
        freq_bins, asd_bins, combined_heatmap_data + 0.1,
        cmap='inferno', shading='auto', norm=LogNorm(vmin=1)
    )
    plt.colorbar(heatmap, label='Count')
    
    # Plot the 25th-percentile curve using the original ASD values.
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

# ========== Usage example ==========
if __name__ == "__main__":
    days = ['010', '014', '016', '018', '020', '021']
    base_path = str(DATA / "LMST_")
    plot_combined_asd_heatmap(base_path, days)
