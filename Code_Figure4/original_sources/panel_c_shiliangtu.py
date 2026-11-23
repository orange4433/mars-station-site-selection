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
freq_list2 = [0.2,1]
SEISMIC_PARAMS = {
    'channel': 'Z',          # Channel to process
    'spec_window': 512,      # Spectral analysis window
    'noverlap': 256,         # Number of overlapping samples
    'clim': (-220, -170),    # Colormap limits
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
        nc_path = Path(nc_path).resolve()
        original_directory = Path.cwd()
        try:
            os.chdir(nc_path.parent)
            self.dataset = nc.Dataset(nc_path.name)
        finally:
            os.chdir(original_directory)
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
    cbar.set_label('Wind Speed (m/s)', fontdict={'family': 'Times New Roman', 'fontsize': 14})
    for label in cbar.ax.get_yticklabels():
        label.set_family('Times New Roman')
        label.set_fontsize(12)
    
    f1 = 16
    f2 = 16
    
    # Show the colorbar.
    plt.xlabel("hour")
    plt.ylabel("sol")
    ticks = np.arange(0, 48, 2)
    plt.xticks(ticks=ticks, labels=[i-48 if i > 47 else i for i in ticks])
    plt.xlabel('LMST hour', fontdict={'family': 'Arial', 'fontsize': f2})
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
    data = np.asarray(data)
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

# Enhanced seismic signal corrector with time-window selection
# --------------------------
class EnhancedSeismicAdjuster(SeismicAdjuster):
    """Enhanced signal corrector with time alignment."""
    
    def align_daily_signal(self, stream: Stream) -> Tuple[np.ndarray, float]:
        """Align data to the Mars local-time interval 00:00-24:00."""
        # Get basic signal information.
        tr = stream.select(channel=f"BH{SEISMIC_PARAMS['channel']}")[0]
        delta = tr.stats.delta
        starttime_lmst = c2lmst(tr.stats.starttime)
        start_time_utc = tr.stats.starttime
        # Calculate the Mars local-time range for the day.
        # Create an LMST object for the start of the day (00:00 LMST).
        target_start_lmst = lmst(sol=starttime_lmst.sol+1, hour=0, minute=0, sec=0)
        target_start_utc = target_start_lmst.conv2utc()

        # Create an LMST object for the end of the day (24:00 LMST).
        target_end_lmst = lmst(sol=starttime_lmst.sol+1, hour=24, minute=0, sec=0)
        target_end_utc = target_end_lmst.conv2utc()

        # Convert target times to sample indices.
        start_idx = int((target_start_utc - start_time_utc) / delta)
        end_idx = int((target_end_utc - start_time_utc) / delta)
        
        # Slice and return.
        return tr.data[start_idx:end_idx], delta

    def corrected_adjust(self, ref_stream: Stream,
                        ref_wind: np.ndarray, target_wind: np.ndarray, 
                        gap: float) -> np.ndarray:
        """Apply correction in multiple frequency bands."""
        ref_signal, delta = self.align_daily_signal(ref_stream)
        
        adjusted_segments = [[] for i in range(len(freq_list)-1)]
        samples_per_segment = int(3600*gap / delta)
        
        # Process all frequency bands independently in each time window.
        for seg in range(int(6/gap)):
            start = seg * samples_per_segment
            end = (seg + 1) * samples_per_segment
            
            # Initialize the corrected signal for this time window.
            seg_adjusted = []
            # Iterate over all frequency-band interpolators.
            for i, interp in enumerate(self.interpolators[0]):
                ref_factor = 10**interp(ref_wind[int(seg*len(ref_wind)/int(24/gap))])
                target_factor = 10**interp(target_wind[int(seg*len(target_wind)/int(24/gap))])
                seg_adjusted = list(ref_signal[start:end] * (target_factor / ref_factor))
                adjusted_segments[i] += seg_adjusted
        #print(len(adjusted_segments))
        for seg in range(int(6/gap),int(17/gap)):
            start = seg * samples_per_segment
            end = (seg + 1) * samples_per_segment
            
            # Initialize the corrected signal for this time window.
            seg_adjusted = []
            # Iterate over all frequency-band interpolators.
            for i, interp in enumerate(self.interpolators[1]):
                ref_factor = 10**interp(ref_wind[int(seg*len(ref_wind)/int(24/gap))])
                target_factor = 10**interp(target_wind[int(seg*len(target_wind)/int(24/gap))])
                seg_adjusted = list(ref_signal[start:end] * (target_factor / ref_factor))
                adjusted_segments[i] += seg_adjusted
        #print(len(adjusted_segments))
        for seg in range(int(17/gap),int(24/gap)):
            start = seg * samples_per_segment
            end = (seg + 1) * samples_per_segment
            
            # Initialize the corrected signal for this time window.
            seg_adjusted = []
            # Iterate over all frequency-band interpolators.
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
        """Apply correction in multiple frequency bands."""
        ref_signal, delta = self.align_daily_signal(ref_stream)
        
        adjusted_segments = [[] for i in range(len(freq_list2)-1)]
        samples_per_segment = int(3600*gap / delta)
        
        # Process all frequency bands independently in each time window.
        for seg in range(int(6/gap)):
            start = seg * samples_per_segment
            end = (seg + 1) * samples_per_segment
            
            # Initialize the corrected signal for this time window.
            seg_adjusted = []
            # Iterate over all frequency-band interpolators.
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
            
            # Initialize the corrected signal for this time window.
            seg_adjusted = []
            # Iterate over all frequency-band interpolators.
            for i, interp in enumerate(self.interpolators2[1]):
                ref_factor = 10**interp(ref_wind[int(seg*len(ref_wind)/int(24/gap))])
                target_factor = 10**interp(target_wind[int(seg*len(target_wind)/int(24/gap))])
                seg_adjusted = list(ref_signal[start:end] * (target_factor / ref_factor))
                adjusted_segments[i] += seg_adjusted
        #print(len(adjusted_segments))
        for seg in range(int(17/gap),int(24/gap)):
            start = seg * samples_per_segment
            end = (seg + 1) * samples_per_segment
            
            # Initialize the corrected signal for this time window.
            seg_adjusted = []
            # Iterate over all frequency-band interpolators.
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
    """Visualize and compare spectra."""
    
    @staticmethod
    def plot_comparison(target_signals1: list,target_signals2,target_signals3,hour,ratio,ratio1,ratio2,Mw, sample_rate: float = 20.0):
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
        seis_data = np.load(HERE / 'event_1133_spectrogram.npz')
        seis_f = seis_data['frequencies']  # Original seismic frequency axis (Hz)
        seis_t = seis_data['times']        # Original seismic time axis (seconds)
        seis_Sxx = seis_data['power']      # Original seismic power data (dB/Hz)
        print(seis_Sxx.shape) 
        def overlay_seismic(sim_t, sim_f, sim_Sxx):
            """Overlay seismic data at the specified hour."""
            # 1. Calculate the seismic data time offset.
            target_origin = hour * 3600 + 300  # Convert to seconds.
            
            # 2. Locate the seismic data on the simulation time axis.
            seis_start = target_origin + seis_t[0]
            seis_end = target_origin + seis_t[-1]
            
            # 3. Determine the overlap interval.
            overlap_start = max(sim_t[0], seis_start)
            overlap_end = min(sim_t[-1], seis_end)
            
            if overlap_start > overlap_end:
                print(1)
                return sim_t, sim_f, sim_Sxx  # Return the original data if there is no overlap.
            
            # 4. Calculate time-axis indices.
            seis_idx_start = np.searchsorted(seis_t, overlap_start - target_origin)
            seis_idx_end = np.searchsorted(seis_t, overlap_end - target_origin)
            sim_idx_start = np.searchsorted(sim_t, overlap_start)
            sim_idx_end = np.searchsorted(sim_t, overlap_end)
            
            # 5. Trim the seismic frequencies to the simulated frequency range.
            freq_mask = (seis_f >= sim_f[0]) & (seis_f <= sim_f[-1])
            valid_seis_f = seis_f[freq_mask]
            valid_seis_Sxx = seis_Sxx[freq_mask, :]
            
            # 6. Create a frequency-axis interpolator.
            interp_Sxx = interp1d(seis_f, seis_Sxx, axis=0, 
                                 bounds_error=False, 
                                 fill_value=np.nan)(sim_f)
            interp_Sxx = (10**(interp_Sxx / 10))*((ratio)**2)
            # 7. Overlay data by direct replacement.
            modified_Sxx = sim_Sxx.copy()
            modified_Sxx[:, sim_idx_start:sim_idx_end] = np.maximum(
                modified_Sxx[:, sim_idx_start:sim_idx_end],
                interp_Sxx[:, seis_idx_start:seis_idx_end]
            )
            
            return sim_t, sim_f, modified_Sxx
        
        
        # Calculate the second simulated spectrum.
        # Process the simulated signal.
        all_freqs = []
        all_Sxx = []
        for sig_pair in target_signals1:
            for i in range(len(sig_pair)):
                sig = sig_pair[i]
                sig = np.array(sig)
                f, t, Sxx = SpectrumVisualizer._compute_spectrogram(sig, sample_rate)
                mask = (f >= freq_list[i]) & (f < freq_list[i + 1])  # Build a mask for the frequency range in the current text file.
                all_freqs.append(f[mask])
                all_Sxx.append(Sxx[mask, :])
        
        # Merge spectra.
        merged_freq = np.concatenate(all_freqs)
        merged_Sxx = np.vstack(all_Sxx)
        sorted_idx = np.argsort(merged_freq)
        merged_freq = merged_freq[sorted_idx]
        merged_Sxx = merged_Sxx[sorted_idx, :]
        
        # After merging the spectra:
        merged_t, merged_f, merged_Sxx = overlay_seismic(t, merged_freq, merged_Sxx)
        
        # Apply Gaussian smoothing using the existing logic.
        merged_Sxx = gaussian_filter1d(merged_Sxx, sigma=0.1, axis=0)
        
        # Define the target time interval, 03:50 to 04:30.
        start_time = hour * 3600  # 4:00 AM in seconds
        end_time = hour * 3600 + 20 * 60     # 4:30 AM in seconds
        
        # Find indices within the target time interval.
        time_mask = (merged_t >= start_time) & (merged_t <= end_time)
        merged_t = merged_t[time_mask]
        merged_Sxx = merged_Sxx[:, time_mask]
        print(merged_Sxx.shape)
        
        all_waveforms = []
        for sig_pair in target_signals2:
            for i in range(0,1):
                sig = np.array(sig_pair[i])
                # Differentiate the signal.
                
                all_waveforms.append(sig)
        key_times = {
            "P Wave": (start_time + 229+300)/3600,  
            "S Wave": (start_time + 405+300)/3600  
        }
        # Overlay all signals.
        combined_waveform = np.sum(all_waveforms, axis=0)
        #combined_waveform = target_signals1[0][10]
        #combined_waveform = np.gradient(combined_waveform, 1/sample_rate)
        print(len(combined_waveform))
        # Plot the waveform.
        waveform = combined_waveform[int((start_time+50)*20):int((end_time+50)*20)]
        print(len(waveform))
        waveform_t = np.linspace(start_time*20, end_time*20, len(waveform))
        
        loaded_data_detrended = np.load(HERE / 'data_detrended_1133.npy')
        split_point = 405 * 20 +50*20
        part1 = loaded_data_detrended[:split_point]
        part2 = loaded_data_detrended[split_point:]

        # Apply different scale factors.
        print(ratio,"我")
        part1_scaled = part1 * ratio
        part2_scaled = part2 * ratio

        # Merge the two processed arrays.
        scaled_data_detrended = np.concatenate((part1_scaled, part2_scaled))
        
        waveform += scaled_data_detrended
        waveform_pre = combined_waveform[int(start_time*20):int((start_time+50)*20)]
        waveform = np.concatenate((waveform_pre, waveform))
        waveform = waveform[:int(end_time-start_time)*20]

        #ax0.set_title(f"Waveform Overlay with Seismic Signal (hour:{hour:02d}:00 , Mw:{Mw})", fontsize=20)
        start_time_hours = hour-1 + 60 / 60  # 4:00 AM in hours
        end_time_hours = hour + 20 / 60     # 4:30 AM in hours
         # Inverse transform
         # Convert dB/Hz back to squared amplitude.
        ax0.set_xlim(start_time_hours, end_time_hours)
        
        # Set x-axis ticks and labels.
        xticks = np.arange(start_time_hours, end_time_hours + 0.1, float(10/60))  # Label every 10 minutes.
        print(xticks)
        
        ax0.set_xticks(xticks)
        ax0.set_xticklabels([f"{int(h):02d}:{int((h % 1) * 61):02d}" for h in xticks])
        ax0.plot(waveform_t / 72000, waveform, color='b')
        ax0.axvline(x=float((hour*3600 + 229+50)/3600), color='g', linestyle=':', linewidth=2, label='P Wave Arrival')
        ax0.axvline(x=float((hour*3600 + 405+50)/3600), color='orange', linestyle=':', linewidth=2, label='S Wave Arrival')
        #ax0.legend(loc='upper right',fontsize = scale-4)
        ax0.set_ylabel("Amplitude(m/s)", fontsize=scale)
        # Set the y-axis offset-text font size (for example, 1e-9).
        ax0.yaxis.offsetText.set_fontsize(scale-2)  # Or use set_size().
        # Set the tick-label font size.
        ax0.tick_params(axis='both', which='major', labelsize=scale-2)  # Configure major tick labels on both axes.
        
        
        
        all_waveforms = []
        all_freqs = []
        all_Sxx = []
        for sig_pair in target_signals1:
            for i in range(0,17):
                sig = np.array(sig_pair[i])
                # Differentiate the signal.
                combined_waveform = sig
                waveform = combined_waveform[int((start_time+50)*20):int((end_time+50)*20)]
                print(len(waveform))
                waveform_t = np.linspace(start_time*20, end_time*20, len(waveform))
                
                loaded_data_detrended = np.load(HERE / 'data_detrended_1133_disp.npy')
                split_point = 405 * 20 +300*20
                part1 = loaded_data_detrended[:split_point]
                part2 = loaded_data_detrended[split_point:]

                # Apply different scale factors.
                part1_scaled = part1 * ratio
                part2_scaled = part2 * ratio

                # Merge the two processed arrays.
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
                mask = (f >= 0.9*freq_list[i]) & (f < 1.1*freq_list[i + 1])   # Build a mask for the frequency range in the current text file.
                all_freqs.append(f[mask])
                all_Sxx.append(Sxx[mask, :])
        
        
        # Merge spectra.
        merged_freq = np.concatenate(all_freqs)
        merged_Sxx = np.vstack(all_Sxx)
        sorted_idx = np.argsort(merged_freq)
        merged_freq = merged_freq[sorted_idx]
        merged_Sxx = merged_Sxx[sorted_idx, :]
        # Overlay all signals.
        #combined_waveform = np.sum(all_waveforms, axis=0)
        #combined_waveform = target_signals1[0][10]
        #combined_waveform = np.gradient(combined_waveform, 1/sample_rate)
        
        #waveform2 = combined_waveform[int((start_time-20)*20):int((end_time+20)*20)]
        #print(waveform1[100:200]==waveform2[100:200])
        # Plot the spectrogram.
        #print(merged_t)
        merged_t = t+hour*3600 -20
        SpectrumVisualizer._plot_spectrum(ax1, merged_t, merged_freq, merged_Sxx, hour)
        #ax1.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    def _get_freq_mask(self, f: np.ndarray) -> np.ndarray:
        """Build a mask for the frequency range of the current interpolator."""
        low = self.freq_breaks[i]
        high = self.freq_breaks[i+1]
        return (f >= low) & (f <= high)    
    @staticmethod
    def _compute_spectrogram(data: np.ndarray, fs: float) -> Tuple[np.ndarray, ...]:
        """Calculate the spectrogram."""
        return spectrogram(data, fs=fs,
                          window=windows.hamming(SEISMIC_PARAMS['spec_window']),
                          nperseg=SEISMIC_PARAMS['spec_window'],
                          noverlap=SEISMIC_PARAMS['noverlap'],
                          scaling='density')

    @staticmethod
    def _plot_spectrum(ax: plt.Axes, t: np.ndarray, f: np.ndarray, Sxx: np.ndarray,hour):
        """Plot a single spectrogram."""
        db = 10 * np.log10(Sxx[1:])  # Exclude 0 Hz.
        pc = ax.pcolormesh(t/3600, np.log10(f[1:]), db, cmap='plasma', shading='auto')
        
        # Define the target time interval, 03:50 to 04:30.
        start_time_hours = hour-1 + 60 / 60  # 3:50 AM in hours
        end_time_hours = hour + 20 / 60     # 4:30 AM in hours
        
        # Set the x-axis limits.
        ax.set_xlim(start_time_hours, end_time_hours)
        
        # Set x-axis ticks and labels.
        xticks = np.arange(start_time_hours, end_time_hours + 0.1, float(10/60))  # Label every 10 minutes.
        #print(xticks)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{int(h):02d}:{int((h % 1) * 61):02d}" for h in xticks])
        
        ax.set_ylim(np.log10([SEISMIC_PARAMS['freq_ticks'][0], SEISMIC_PARAMS['freq_ticks'][-1]]))
        ax.set_yticks(np.log10(SEISMIC_PARAMS['freq_ticks']))
        ax.set_yticklabels([str(f) for f in SEISMIC_PARAMS['freq_ticks']])
        # Set the tick-label font size.
        ax.tick_params(axis='both', which='major', labelsize=scale-2)  # Configure major tick labels on both axes.
        ax.set_xlabel("Local Hour (h)", fontsize=scale)
        ax.set_ylabel("Frequency (Hz)", fontsize=scale)
        #ax.set_title(title, fontsize=20)
        pc.set_clim(SEISMIC_PARAMS['clim'])
        
        # Move the colorbar to the right and reduce its size.
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
    wind_proc = WindProcessor(str(MCD_FILE))# PCM wind speed
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
    # Apply the signal correction.
    adjusted = adjuster.corrected_adjust(
        ref_stream,
        np.sqrt(u_ref**2 + v_ref**2)[(ref_day-113)*48:(ref_day-113+1)*48],
        np.sqrt(u_target**2 + v_target**2)[(day-113)*48:(day-113+1)*48],
        0.5
    )# There are 17 fits, one for each frequency band.
    adjusted2 = adjuster.corrected_adjust2(
        sst,
        np.sqrt(u_ref**2 + v_ref**2)[(ref_day-113)*48:(ref_day-113+1)*48],
        np.sqrt(u_target**2 + v_target**2)[(day-113)*48:(day-113+1)*48],
        0.5
    )# One fit for a single frequency band, with morning/day/evening periods.
    
    print('开始')
    print(np.mean(adjusted2))
    print('结束')
    
    adjusted3 = adjuster.corrected_adjust2(
        ssst,
        np.sqrt(u_ref**2 + v_ref**2)[(ref_day-113)*48:(ref_day-113+1)*48],
        np.sqrt(u_target**2 + v_target**2)[(day-113)*48:(day-113+1)*48],
        0.5
    )# One fit for a single frequency band, with morning/day/evening periods.
    wind_speeds = np.sqrt(u_target**2 + v_target**2)[(day-113)*48:(day-113+1)*48]
    dt = 0.05
    Fs = 1/dt
    wind = wind_speeds[2*hour]
    # Merge interpolation results, omitting the duplicate b point.
    
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
        
        # Slice frequency and PSD.
        f_segment = f_pi[mask]
        psd_segment = psd_pi[mask]
        
        # Append to the combined f_p and psd_p arrays.
        f_p.extend(f_segment)
        psd_p.extend(psd_segment)
    f_p = np.array(f_p)
    psd_p = np.array(psd_p)
    
    
    #ax2 = plt.subplot(1, 1, 1)
    #ax2.plot(np.log10(f_p[1:]), np.log10(np.sqrt(psd_p[1:])), label="noise")
    
    filename = HERE / 'model_asd_data_final.txt'
    data = np.loadtxt(filename)

    # The file has two rows: frequency, then displacement ASD.
    frequencies = data[0, :]
    asd_disp = data[1, :]
    ratio,asd_disp = calculate_theoretical_asd(asd_disp, angle_distance,Mw)
    # Select only the required data range.
    mask = (frequencies >= 0.1) & (frequencies <= 10) & (asd_disp >= -15) & (asd_disp <= -6)
    filtered_frequencies = frequencies[mask]
    filtered_asd_disp = asd_disp[mask]
    
    ratio1 = 10**(Mw-3.84-0.73*np.log10(math.degrees(angle_distance/30.2)))
    ratio2 = 10**(Mw-3.84-1.06*np.log10(math.degrees(angle_distance/30.2)))
    ratio1 = ratio2
    print(ratio1,ratio2)
    
    SpectrumVisualizer.plot_comparison([adjusted],[adjusted2],[adjusted3],hour,ratio,ratio1,ratio2,Mw)
    
    
    input_txt_file_processed = HERE / 'processed_noise.txt'  # Path to the processed text file
    # Read the processed text file.
    with open(input_txt_file_processed, 'r') as file:
        lines = file.readlines()
        frequencies_processed = lines[0].strip().split(',')
        values_processed = lines[1].strip().split(',')

    # Convert strings to floats for calculation.
    frequencies_processed = np.array([float(f) for f in frequencies_processed])
    values_processed = np.array([float(v) for v in values_processed])
    #ax2.plot(np.log10(frequencies_processed), np.log10(values_processed), label='Original_noise', marker='x')

    
    
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

    # Check whether 80% of the mean seismic signal exceeds the mean noise.
    threshold = mean_seismic * 0.5 > mean_noise

    print(f"Noise Mean Amplitude: {mean_noise}")
    print(f"Seismic Mean Amplitude: {mean_seismic}")
    print(f"Is Seismic Signal Higher than Noise by 80%? {'Yes' if threshold else 'No'}")
    print(Mw)    
    
    #plt.title(f'earthquake:({earthquake_points[earthquake][0],earthquake_points[earthquake][1]}),magnitude:{Mw},distance:{math.degrees(angle_distance):.2f},lat:{lat},lon:{lon},visible:{threshold}')

    return threshold

def calculate_theoretical_asd(base_asd, angle_distance,Mw):
    """Calculate theoretical ASD from the amplitude ratio."""
    ratio = (30.2 / math.degrees(angle_distance))*(10**((3/2)*(Mw-3.41)))
    print(ratio)
    return ratio,base_asd + np.log10(ratio)  # Because ASD is proportional to A^2.

# Main program
if __name__ == "__main__":
    nc_file_path = str(MCD_FILE)
    lat, lon = 21.1, -43.2  # Example longitude and latitude
    min_value = 0  # Minimum wind speed
    max_value = 16  # Maximum wind speed
    u = 0.03  # Figure aspect ratio
    integral = 1  # Integration constant (adjust as needed)
    ans = {}
    for i in range(len(earthquake_points)):
        Mw = 2.9
        visible = True
        while visible:
            visible = compare_asd(nc_file_path, lat, lon,434,1,i,Mw)
            Mw -= float(0.1)
        ans[earthquake_points[i]] = Mw
    print(ans)
        
    




