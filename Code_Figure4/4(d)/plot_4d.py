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
from matplotlib.patches import Rectangle
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
ROOT = Path(__file__).resolve().parents[1]
NOISE_DATA_DIR = ROOT / 'data' / 'on_deck'
EVENT_DATA_DIR = ROOT / 'data' / 'seismic'
OUTPUT_DIR = ROOT / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)
# Parameters used when the original Figure_4 base panel was exported.
# The panel annotation reports the inferred detection limit (Mw 4.4), whereas
# the first displayed realization in the legacy loop started from Mw 4.8.
S1133C_DISTANCE_DEG = 60.219214529308665
FIGURE_4_EXPORT_MW = 4.8

def mars_sol2ls(sol):
    """
    Convert Mars sol to solar longitude (Ls).
    Use precise Mars orbital parameters.
    """
    # Mars orbital parameters
    year_day = 668.6  # Length of a Mars year in sols
    peri_day = 485.35  # Sol at perihelion
    e_elips = 0.09340  # Orbital eccentricity
    timeperi = 1.90258341759902  # Longitude of perihelion in radians
    
    # Calculate the mean anomaly.
    M = 2 * np.pi * (sol - peri_day) / year_day
    M = M % (2 * np.pi)
    
    # Solve Kepler's equation with Newton iteration.
    E = M
    for _ in range(10):
        delta_E = (E - e_elips * np.sin(E) - M) / (1 - e_elips * np.cos(E))
        E -= delta_E
        if abs(delta_E) < 1e-7:
            break
    
    # Calculate the true anomaly.
    true_anomaly = 2 * np.arctan(np.sqrt((1 + e_elips) / (1 - e_elips)) * np.tan(E / 2))
    
    # Calculate solar longitude Ls.
    Ls = (true_anomaly - timeperi) % (2 * np.pi)
    return np.degrees(Ls)

def insight_sol_to_mars_day(insight_sol):
    """
    Convert InSight sol to the day within a Mars year.
    InSight landed on day 555 of the Mars year.
    """
    # InSight sol 0 corresponds to day 555 of the Mars year.
    mars_day = insight_sol + 555
    
    # Length of a Mars year
    mars_year_length = 668.6
    
    # Wrap values modulo one year.
    if mars_day > mars_year_length:
        mars_day = mars_day % mars_year_length
    
    return mars_day

freq_list = [0.05, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8]
freq_list2 = [0.1,1]
SEISMIC_PARAMS = {
    'channel': 'Z',          # Channel to process
    'spec_window': 512,      # Spectral analysis window
    'noverlap': 256,         # Number of overlapping samples
    'clim': (-220, -140),    # Colormap limits
    'freq_ticks': [0.1, 0.3, 1, 3, 8]  # Frequency ticks
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
    """Process wind-speed data in space and time using the mcd_data.nc format."""
    
    def __init__(self, nc_path: str):
        self.dataset = nc.Dataset(nc_path)
        self.sol_min = 114
        self.sol_max = 794
        self._cache = {}  # Cache for location-specific data
        self._precompute_sol_data()
    
    def _precompute_sol_data(self):
        """Precompute wind-speed data for all sols."""
        # Calculate Ls for each sol.
        self.sol_values = np.arange(self.sol_min, self.sol_max + 1)
        self.ls_values = []
        
        for sol in self.sol_values:
            mars_day = insight_sol_to_mars_day(sol)
            ls = mars_sol2ls(mars_day)
            self.ls_values.append(ls)
        
        # Ls values in the original data
        self.original_ls = self.dataset['Ls'][:]
    
    def get_wind_profile(self, lat: float, lon: float) -> Tuple[np.ndarray, np.ndarray]:
        """Get wind-speed data at the specified location."""
        # Check the cache.
        cache_key = f"{lat}_{lon}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Find the nearest grid point.
        lats = self.dataset['latitude'][:]
        lons = self.dataset['longitude'][:]
        
        lat_idx = np.abs(lats - lat).argmin()
        lon_idx = np.abs(lons - lon).argmin()
        
        # Get wind-speed data for all sols.
        u_data = np.zeros((len(self.sol_values), 48))
        v_data = np.zeros((len(self.sol_values), 48))
        
        for i, sol in enumerate(self.sol_values):
            # Find the nearest Ls index.
            target_ls = self.ls_values[i]
            ls_idx = np.argmin(np.abs(self.original_ls - target_ls))
            
            # Get wind-speed data for the location and Ls.
            u_data[i, :] = self.dataset['U'][lon_idx, lat_idx, :, ls_idx]
            v_data[i, :] = self.dataset['V'][lon_idx, lat_idx, :, ls_idx]
        
        # Apply the local-time correction.
        shift_hours = int(lon / 15)  # Convert longitude to an hour offset.
        shift_points = 0*shift_hours * 2  # One point per half hour
        
        # Roll the data for each sol.
        adjusted_u = np.zeros_like(u_data)
        adjusted_v = np.zeros_like(v_data)
        
        for sol_idx in range(len(self.sol_values)):
            adjusted_u[sol_idx] = np.roll(u_data[sol_idx], -shift_points)
            adjusted_v[sol_idx] = np.roll(v_data[sol_idx], -shift_points)
        
        # Flatten into a continuous time series.
        u = adjusted_u.flatten()
        v = adjusted_v.flatten()
        print(len(u))
        # Cache the result.
        self._cache[cache_key] = (u, v)
        return u, v

# Plot a wind-speed heatmap for one location.
def figure(wind_speeds, min_value, max_value, u, integral, title):
    data = wind_speeds.reshape(-1, 48)  # Assume 48 time steps per hour.
    
    # Create a larger array containing the data, with NaN elsewhere.
    full_data = np.full((1001, 48), np.nan)  # Create a 1300-by-48 array filled with NaN.
    full_data[114:114+len(data), :] = data  # Copy the data into the corresponding positions of the NaN array.
    
    # Plot the heatmap with matplotlib.
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_ylim(0, 1000)
    
    # cmap specifies the colormap.
    sol_range = [1, 1000]
    norm = colors.Normalize(vmin=min_value, vmax=max_value, clip=True)
    scalar_mappable = cm.ScalarMappable(norm=norm, cmap=plt.get_cmap('plasma'))
    scalar_mappable.set_array([])  # Initialize an empty array.
    im = ax.imshow(full_data, norm=norm, cmap='plasma', aspect=u)
    
    # Add a colorbar.
    cbar = plt.colorbar(scalar_mappable, ax=ax)
    cbar.set_label('Wind Speed (m/s)', fontdict={'family': 'Arial', 'fontsize': scale})
    for label in cbar.ax.get_yticklabels():
        label.set_family('Times New Roman')
        label.set_fontsize(12)
    

    
    # Show the colorbar.
    plt.xlabel("hour")
    plt.ylabel("sol")
    ticks = np.arange(0, 48, 2)
    plt.xticks(ticks=ticks, labels=[i-48 if i > 47 else i for i in ticks])
    plt.xlabel('LMST hour', fontdict={'family': 'Arial', 'fontsize': scale})
    plt.xlim(0, 47)  # Adjust for 48 points.
    original_xticks = plt.xticks()[0]

    # Calculate new y-axis tick positions by multiplying by 5.
    new_xticks1 = original_xticks / 2
    new_xticks = [int(new_xticks1[i]) for i in range(len(new_xticks1))]
    # Set the new y-axis tick positions.
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
    segment_length = int(25.6 / dt)  # For example, a 25.6-second window
    noverlap = int(segment_length * 0.5)  # 50% overlap
    #nfft = 4096  # Zero-pad to 4096 samples.
    window = windows.hann(segment_length)  # Hann window
    
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
    """Correct seismic signals using wind-speed features in multiple frequency bands."""
    
    def __init__(self, ref_day: int, target_day: int):
        self.ref_day = ref_day
        self.target_day = target_day
        
        # Define frequency-band boundaries.
        self.freq_breaks = freq_list
        
        # Load 17 interpolators.
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
    """Align data to the Mars local-time interval 00:00-24:00."""
    # Get basic signal information.
    tr = stream.select(channel='EHZ')[0]
    delta = tr.stats.delta
    starttime_lmst = c2lmst(tr.stats.starttime)
    start_time_utc = tr.stats.starttime
    # Calculate the Mars local-time range for the day.
    # Create an LMST object for the start of the day (00:00 LMST).
    target_start_lmst = lmst(sol=starttime_lmst.sol, hour=17, minute=45, sec=0)
    target_start_utc = target_start_lmst.conv2utc()

    # Create an LMST object for the end of the day (24:00 LMST).
    target_end_lmst = lmst(sol=starttime_lmst.sol, hour=18, minute=35, sec=0)
    target_end_utc = target_end_lmst.conv2utc()

    # Convert target times to sample indices.
    start_idx = int((target_start_utc - start_time_utc) / delta)
    end_idx = int((target_end_utc - start_time_utc) / delta)
    
    # Slice and return.
    return tr.data[start_idx:end_idx], delta

  
ref_day=20

import math

# Mean Mars radius in km
R = 3389.5  # Mean radius

def haversine(point1, point2):
    """
    Calculate the distance between two geographic points with the Haversine formula.

    Parameters:
        point1: (latitude1, longitude1), in degrees.
        point2: (latitude2, longitude2), in degrees.

    Returns:
        distance: Distance between the points in km.
        angle_distance: Angular distance in radians.
    """
    lat1, lon1 = math.radians(point1[0]), math.radians(point1[1])
    lat2, lon2 = math.radians(point2[0]), math.radians(point2[1])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    distance = R * c
    angle_distance = c  # Angular great-circle distance
    
    return angle_distance

# Candidate seismometer locations
seismometer_points = [
    (-48.75, -45.0),
    (-7.5, -5.625),
    (-37.5, 118.125),
    (-45.0, 151.875),
    (-75.0, 78.75),
    (22.5, 180)
]

# Possible marsquake locations
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
    """Visualize and compare spectra."""
    
    @staticmethod
    def plot_comparison(signals,signals1,ratio,ratio1,Mw, sample_rate=100):
        """Display a four-panel comparison."""
        
        fig = plt.figure(figsize=(14, 12))

        # Define the layout with GridSpec:
        # Use the left third of the first row; leave the rest empty.
        # The second row spans the full width.
        gs = GridSpec(2, 2, figure=fig, width_ratios=[5, 1], height_ratios=[1, 2])

        # axes[0] occupies the first of three columns in the first row (one-third width).
        ax0 = fig.add_subplot(gs[0, 0])  # Use only the first column of the first row.

        # axes[1] spans all three columns in the second row.
        ax1 = fig.add_subplot(gs[1, :])  # Span all columns.
        
        # Load the seismic data.
        seis_data = np.load(NOISE_DATA_DIR / 'event_1133_spectrogram.npz')
        seis_f = seis_data['frequencies']  # Original seismic frequency axis (Hz)
        seis_t = seis_data['times']        # Original seismic time axis (seconds)
        seis_Sxx = seis_data['power']      # Original seismic power data (dB/Hz)
        
        def overlay_seismic(sim_t, sim_f, sim_Sxx):
            """Overlay seismic data at the specified hour."""
            # 1. Calculate the seismic data time offset.
            target_origin = 600  # Convert to seconds.
            
            # 2. Locate the seismic data on the simulation time axis.
            seis_start = target_origin + seis_t[0]
            seis_end = target_origin + seis_t[-1]
            # 3. Determine the overlap interval.
            overlap_start = max(sim_t[0], seis_start)
            overlap_end = min(sim_t[-1], seis_end)
            '''
            if overlap_start > overlap_end:
                print(1)
                return sim_t, sim_f, sim_Sxx  # Return the original data if there is no overlap.
            '''
            # Create an interpolator from t_20 to each row of Sxx_20Hz.
            seis_Sxx = seis_data['power']
           
            f_interp = interp1d(seis_t, seis_Sxx, kind='linear', axis=1,fill_value="extrapolate")
            
            # Interpolate Sxx_20Hz onto the t_100 time points.
            seis_Sxx = f_interp(sim_t)
            
            # 4. Calculate time-axis indices.
            #seis_idx_start = np.searchsorted(seis_t, overlap_start - target_origin)
            #seis_idx_end = np.searchsorted(seis_t, overlap_end - target_origin)
            sim_idx_start = np.searchsorted(sim_t, overlap_start)
            sim_idx_end = np.searchsorted(sim_t, overlap_end)
            seis_idx_start = sim_idx_start
            seis_idx_end = sim_idx_end
            
            
            
            # 5. Trim the seismic frequencies to the simulated frequency range.
            freq_mask = (seis_f >= 0.1) & (seis_f <= 1)
            valid_seis_f = seis_f[freq_mask]
            valid_seis_Sxx = seis_Sxx[freq_mask, :]
            
            # 6. Create a frequency-axis interpolator.
            interp_Sxx = interp1d(seis_f, seis_Sxx, axis=0, 
                                 bounds_error=False, 
                                 fill_value=np.nan)(sim_f)
            interp_Sxx = (10**(interp_Sxx / 10))*((ratio)**2)
            # 7. Overlay data by direct replacement.
            modified_Sxx = sim_Sxx.astype(np.float64, copy=True)

            modified_Sxx[:, sim_idx_start:sim_idx_end] = np.maximum(
                modified_Sxx[:, sim_idx_start:sim_idx_end],
                interp_Sxx[:, seis_idx_start:seis_idx_end]
            )
            
            return sim_t, sim_f, modified_Sxx
        
        
        # Calculate the spectrum.

        f, t, Sxx = SpectrumVisualizer._compute_spectrogram(signals1, sample_rate)
        mask = (f >= 0.04) & (f < 10)
        # Merge spectra.
        merged_freq = f[mask]
        merged_Sxx = Sxx[mask, :]

        # After merging the spectra:
        merged_t, merged_f, merged_Sxx = overlay_seismic(t, merged_freq, merged_Sxx)

        # Apply Gaussian smoothing using the existing logic.
        merged_Sxx = gaussian_filter1d(merged_Sxx, sigma=0.1, axis=0)
        
        # Define the target time interval, 03:50 to 04:30.
        start_time = 600  # 4:00 AM in seconds
        end_time = 2400  # 4:30 AM in seconds
        '''
        # Find indices within the target time interval.
        time_mask = (merged_t >= start_time) & (merged_t <= end_time)
        merged_t = merged_t[time_mask]
        merged_Sxx = merged_Sxx[:, time_mask]
        '''
        
        key_times = {
            "P Wave": (start_time + 229)/3600,  
            "S Wave": (start_time + 405)/3600  
        }
        # Overlay all signals.
        combined_waveform = signals
        #combined_waveform = target_signals1[0][10]
        #combined_waveform = np.gradient(combined_waveform, 1/sample_rate)
        print(len(signals))
        # Plot the waveform.
        waveform = combined_waveform[int((start_time)*100):int((end_time)*100)]
        print(len(waveform))
        
        waveform_t = np.linspace(0, (end_time-start_time)*100, len(waveform))
        
        loaded_data_detrended = np.load(EVENT_DATA_DIR / 'data_detrended_1133.npy')
        

        # Merge the two processed arrays.
        scaled_data_detrended = loaded_data_detrended * ratio1
        #print(len(scaled_data_detrended),max(loaded_data_detrended))

        # 1. Construct the source and target time axes.
        x_old = np.arange(len(scaled_data_detrended))                # [0, 1, 2, ..., 23999]
        x_new = np.linspace(0, len(scaled_data_detrended) - 1, 120000)  # Sample 120000 evenly spaced points between 0 and 23999.

        # 2. Create a linear interpolator and interpolate.
        f_interp = interp1d(x_old, scaled_data_detrended, kind='linear', fill_value='extrapolate')
        array_120k = f_interp(x_new)  # shape: (120000,)

        # 3. Sum the first 120000 elements.
        a = 5000
        result_ffirst_part = waveform[0:a]
        result_first_part = waveform[a:120000+a] + array_120k
        print(len(result_first_part))
        # 4. Leave the final 60000 elements unchanged.
        result_second_part = waveform[120000+a:]
        print(result_second_part[:10])
        # 5. Merge the results.
        waveform = np.concatenate([result_ffirst_part,result_first_part, result_second_part], axis=0)
        print(len(waveform))
        
        

        #ax0.set_title(f"Waveform Overlay with Seismic Signal (hour:18:00 , Mw:{Mw})", fontsize=20)
        #ax0.set_title(f"Waveform Overlay with S1133c Seismic Signal (hour:18:00)", fontsize=20)
        start_time_hours = 0# 4:00 AM in hours
        end_time_hours = 20 / 60     # 4:30 AM in hours
         # Inverse transform
         # Convert dB/Hz back to squared amplitude.
        ax0.set_xlim(start_time_hours, end_time_hours)
        
        # Set x-axis ticks and labels.
        xticks = np.arange(start_time_hours, end_time_hours + 0.1, float(10/60))  # Label every 10 minutes.
        
        
        ax0.set_xticks(xticks)
        ax0.set_xticklabels([f"{int(h+17.84)}:{int((h+0.917)%1 * 61):02d}" for h in xticks])
        ax0.plot(waveform_t / 360000, waveform, color='b')
        #ax0.axvline(x=float((229)/3600), color='g', linestyle=':', linewidth=2, label='P Wave Arrival')
        #ax0.axvline(x=float((405)/3600), color='orange', linestyle=':', linewidth=2, label='S Wave Arrival')
        ax0.axvline(x=float((279)/3600), color='g', linestyle=':', linewidth=2, label='P Wave Arrival')
        ax0.axvline(x=float((455)/3600), color='orange', linestyle=':', linewidth=2, label='S Wave Arrival')
        ax0.text(float(279/3600), 1.01, 'P', transform=ax0.get_xaxis_transform(),
                 ha='center', va='bottom', fontsize=scale)
        ax0.text(float(455/3600), 1.01, 'S', transform=ax0.get_xaxis_transform(),
                 ha='center', va='bottom', fontsize=scale)
        #ax0.legend(loc='upper right',fontsize = scale-4)
        ax0.set_ylabel("Amplitude(m/s)", fontsize=scale)
        ax0.tick_params(axis='both', which='major', labelsize=scale-2)
        ax0.yaxis.offsetText.set_fontsize(scale-2)
        
        combined_waveform = signals1

        # Plot the waveform.
        waveform = combined_waveform[int((start_time)*100):int((end_time)*100)]
        loaded_data_detrended = np.load(EVENT_DATA_DIR / 'data_detrended_1133.npy')
        

        # Merge the two processed arrays.
        scaled_data_detrended = loaded_data_detrended * ratio1
        #print(len(scaled_data_detrended),max(loaded_data_detrended))

        # 1. Construct the source and target time axes.
        x_old = np.arange(len(scaled_data_detrended))                # [0, 1, 2, ..., 23999]
        x_new = np.linspace(0, len(scaled_data_detrended) - 1, 120000)  # Sample 120000 evenly spaced points between 0 and 23999.

        # 2. Create a linear interpolator and interpolate.
        f_interp = interp1d(x_old, scaled_data_detrended, kind='linear', fill_value='extrapolate')
        array_120k = f_interp(x_new)  # shape: (120000,)

        # 3. Sum the first 120000 elements.
        ra = 5000
        result_ffirst_part = waveform[0:a]
        result_first_part = waveform[a:120000+a] + array_120k
        print(len(result_first_part))
        # 4. Leave the final 60000 elements unchanged.
        result_second_part = waveform[120000+a:]
        print(result_second_part[:10])
        # 5. Merge the results.
        waveform = np.concatenate([result_ffirst_part,result_first_part, result_second_part], axis=0)
        
        waveform_pre = combined_waveform[int((start_time-20)*100):int(start_time*100)]
        waveform = np.concatenate((waveform_pre, waveform))
        
        dt = 0.05  # Sampling interval
        velocity_data = waveform  # If P_new_Z contains velocity data

        # Integrate to obtain displacement.
        t = np.arange(len(velocity_data)) * dt
        displacement = cumulative_trapezoid(velocity_data, t, initial=0)
        
        merged_f, merged_t, merged_Sxx = SpectrumVisualizer._compute_spectrogram(displacement, fs=100)        
        # Plot the spectrogram.
        
        merged_t -= 20
        
        SpectrumVisualizer._plot_spectrum(ax1, merged_t, merged_f, merged_Sxx)
        ax1.add_patch(Rectangle(
            (4/60, np.log10(0.2)), 11/60,
            np.log10(1.0) - np.log10(0.2),
            fill=False, edgecolor='white', linewidth=2.5,
            linestyle=(0, (6, 4)),
        ))
        ax1.text(
            0.99, 0.96, 'On Deck', transform=ax1.transAxes,
            ha='right', va='top', fontsize=scale-2, fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.5'),
        )
        ax1.text(
            0.01, 0.02, f'Mw {Mw:.1f}', transform=ax1.transAxes,
            ha='left', va='bottom', fontsize=scale-2, fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.5'),
        )
        #ax1.legend(loc='upper right')
        
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / '4d_on_deck_mw4.4.png', dpi=300, bbox_inches='tight')
        fig.savefig(OUTPUT_DIR / '4d_on_deck_mw4.4.pdf', bbox_inches='tight')
        plt.close(fig)

    def _get_freq_mask(self, f: np.ndarray) -> np.ndarray:
        """Build a mask for the frequency range of the current interpolator."""
        low = self.freq_breaks[i]
        high = self.freq_breaks[i+1]
        return (f >= low) & (f <= high)    
    @staticmethod
    def _compute_spectrogram(data: np.ndarray, fs: float) -> Tuple[np.ndarray, ...]:
        """Calculate the spectrogram."""
        nperseg = int(256 * 20)  # 256-second window
        noverlap = int(nperseg * 0.75)
        return spectrogram(data, fs=fs,
                          window=windows.hamming(nperseg),
                          nperseg=nperseg,
                          noverlap=noverlap,
                          scaling='density')

    @staticmethod
    def _plot_spectrum(ax: plt.Axes, t: np.ndarray, f: np.ndarray, Sxx: np.ndarray):
        """Plot a single spectrogram."""
        db = 10 * np.log10(Sxx[1:])  # Exclude 0 Hz.
        pc = ax.pcolormesh(t/3600, np.log10(f[1:]), db, cmap='plasma', shading='auto')
        
        # Define the target time interval, 03:50 to 04:30.
        start_time_hours = 0  # 3:50 AM in hours
        end_time_hours =1/3  # 4:30 AM in hours
        
        # Set the x-axis limits.
        ax.set_xlim(start_time_hours, end_time_hours)
        
        # Set x-axis ticks and labels.
        xticks = np.arange(start_time_hours, end_time_hours + 0.1, float(10/60))  # Label every 10 minutes.

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
        
        # Move the colorbar to the right and reduce its size.
        cbar = plt.colorbar(pc, ax=ax, orientation='vertical', pad=0.04, shrink=0.6)
        cbar.set_label('Power Spectral Density (dB)', fontsize=scale, labelpad=15)
        cbar.ax.tick_params(labelsize=scale)

def compare_asd(nc_file_path, lat, lon,day,hour,earthquake,Mw):
    earthquake_lat = earthquake_points[earthquake][0]
    earthquake_lon = earthquake_points[earthquake][1]
    angle_distance = math.radians(S1133C_DISTANCE_DEG)
    
    
    Evt_name = str(ref_day).zfill(3)
    main_file = NOISE_DATA_DIR / f'LMST_{Evt_name}.mseed'
    st_disp = read(main_file)
    channel = 'Z'
    ref_tr = st_disp.select(channel=f'EH{channel}')
    
    D_vel = ref_tr[0].data
    
    ref_stream = st_disp
    ref_stream.filter('bandpass',freqmin=0.1,freqmax =0.8,corners=2,zerophase=True)
    ref_stream1 = read(main_file)
    adjusted,_ = align_daily_signal(ref_stream)
    adjusted1,_ = align_daily_signal(ref_stream1)
    print(len(adjusted))
    fig = plt.figure(figsize=(14, 10))


    ax2 = plt.subplot(1, 1, 1)
    #ax2.plot(np.log10(f_p[1:]), np.log10(np.sqrt(psd_p[1:])), label="noise")
    
    filename = NOISE_DATA_DIR / 'model_asd_data_final.txt'
    data = np.loadtxt(filename)

    # The file has two rows: frequency, then displacement ASD.
    frequencies = data[0, :]
    asd_disp = data[1, :]
    ratio,asd_disp = calculate_theoretical_asd(asd_disp, angle_distance,FIGURE_4_EXPORT_MW)
   
    # Select only the required data range.
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
    
    input_txt_file_processed = NOISE_DATA_DIR / 'asd_25th_percentile.txt'
    # Read the processed text file.
    with open(input_txt_file_processed, 'r') as file:
        lines = file.readlines()
        frequencies_processed = lines[0].strip().split(' ')
        values_processed = lines[1].strip().split(' ')

    # Convert strings to floats for calculation.
    frequencies_processed = np.array([float(f) for f in frequencies_processed])
    values_processed = np.array([float(v) for v in values_processed])
    ax2.plot(np.log10(frequencies_processed), np.log10(values_processed), label='Original_noise', marker='x')
    
    # Get the data for both curves.
    x1 = np.log10(filtered_frequencies)
    y1 = filtered_asd_disp
    x2 = np.log10(frequencies_processed)
    y2 = np.log10(values_processed)

    # Ensure the data are sorted.
    sort_idx1 = np.argsort(x1)
    x1_sorted = x1[sort_idx1]
    y1_sorted = y1[sort_idx1]

    sort_idx2 = np.argsort(x2)
    x2_sorted = x2[sort_idx2]
    y2_sorted = y2[sort_idx2]

    # Generate dense interpolation points.
    x_min = max(x1_sorted.min(), x2_sorted.min())
    x_max = min(x1_sorted.max(), x2_sorted.max())
    x_interp = np.linspace(x_min, x_max, 10000)

    # Interpolate both curves.
    y1_interp = np.interp(x_interp, x1_sorted, y1_sorted)
    y2_interp = np.interp(x_interp, x2_sorted, y2_sorted)

    # Find intersections.
    diff = y1_interp - y2_interp
    crossing_indices = np.where(np.diff(np.sign(diff)))[0]

    # Calculate the exact intersections.
    crossings = []
    for idx in crossing_indices:
        x0, x1 = x_interp[idx], x_interp[idx+1]
        y0, y1 = diff[idx], diff[idx+1]
        t = -y0 / (y1 - y0)
        x_cross = x0 + t*(x1 - x0)
        crossings.append(x_cross)

    # Require two intersections.
    if len(crossings) >= 2:
        # Use the first two intersections.
        x_start, x_end = sorted(crossings[:2])
        
        # Generate data for the filled region.
        mask = (x_interp >= x_start) & (x_interp <= x_end)
        x_fill = x_interp[mask]
        y1_fill = y1_interp[mask]
        y2_fill = y2_interp[mask]
        
        # Draw the filled region.
        ax2.fill_between(x_fill, y1_fill, y2_fill, 
                        where=y1_fill > y2_fill, 
                        color='skyblue', alpha=0.4,
                        label='Detection')

    # Update the legend.
    ax2.legend(loc='best', fontsize=18)

    from matplotlib.ticker import FuncFormatter

    # Create an exponent-formatting function.
    def log_label_formatter(val, pos):
        """Format log10 values as powers of ten."""
        return r"$10^{%.0f}$" % val

    # Apply it to the y-axis.
    ax2.yaxis.set_major_formatter(FuncFormatter(log_label_formatter))

    # Set axis labels while keeping data in log10 units.
    ax2.set_xlabel(r'$\mathrm{Frequency\ [Hz]}$', fontsize=18)
    ax2.set_ylabel(r'$\mathrm{Displacement\ ASD\ [m/\sqrt{Hz}]}$', fontsize=18)
    
    
    # Find indices in the 0.2-1 Hz band.
    mask = (frequencies_processed >= 0.2) & (frequencies_processed <= 1)
    noise_mask = (filtered_frequencies >= 0.2) & (filtered_frequencies <= 1)
    #noise_freqs = 
    # Extract data in the 0.2-1 Hz band.
    noise_freqs = frequencies_processed[mask]
    noise_amplitudes = values_processed[mask]
    seismic_freqs = filtered_frequencies[noise_mask]
    seismic_amplitudes = (10**np.array(filtered_asd_disp))[noise_mask]

    # Calculate mean noise and seismic signal levels over 0.2-1 Hz.
    mean_noise = np.mean(noise_amplitudes)
    mean_seismic = np.mean(seismic_amplitudes)

    # Check whether 50% of the mean seismic signal exceeds the mean noise.
    threshold = mean_seismic * 0.5 > mean_noise

    print(f"Noise Mean Amplitude: {mean_noise}")
    print(f"Seismic Mean Amplitude: {mean_seismic}")
    print(f"Is Seismic Signal Higher than Noise by 80%? {'Yes' if threshold else 'No'}")
    print(Mw)    
    
    plt.title(f'ASD Curve — Seismic Signal (Mw: {Mw}) vs Simulated Noise (Hour: {hour}:00)',fontsize=20)
    #f'earthquake:({earthquake_points[earthquake][0],earthquake_points[earthquake][1]}),magnitude:{Mw},distance:{math.degrees(angle_distance):.2f},lat:{lat},lon:{lon},visible:{threshold}')
    plt.close(fig)
    return angle_distance,threshold

def calculate_theoretical_asd(base_asd, angle_distance,Mw):
    """Calculate theoretical ASD from the amplitude ratio."""
    ratio = (30.2 / math.degrees(angle_distance))*(10**((3/2)*(Mw-3.84)))
    print(ratio)
    return ratio,base_asd + np.log10(ratio)  # Because ASD is proportional to A^2.

# Main program
if __name__ == "__main__":
    nc_file_path = '../mcd_data.nc'
    lat, lon = 21.1, -43.2  # Example longitude and latitude
    min_value = 0  # Minimum wind speed
    max_value = 16  # Maximum wind speed
    u = 0.03  # Figure aspect ratio
    integral = 1  # Integration constant (adjust as needed)
    compare_asd(None, lat, lon, 249, 1, 0, 4.4)
    print(f"Saved: {OUTPUT_DIR / '4d_on_deck_mw4.4.png'}")
        
    





