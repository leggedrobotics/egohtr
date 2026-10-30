import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

from src.visualizer.utils import (
    plot_time_sync_drift, 
    plot_clap_detection, 
    replace_exceeding_with_prev, 
    plot_acceleration_data, 
    plot_audio_data,
)
from src.config.configs import PipelineConfig


def replace_exceeding_with_prev(arr, threshold):
    """
    Replaces outliers with the last known valid value.
    """
    mask = arr > threshold
    if not np.any(mask):
        return arr
    
    idx = np.arange(len(arr))
    idx[mask] = 0
    idx = np.maximum.accumulate(idx) # forward fill
    return arr[idx]

def get_data_peaks(acc: np.ndarray, sync_cfg: dict, clear_startup: int, peak_height: float, peak_distance: int) -> np.ndarray:
    """ Get acceleration peaks from data for time synchronization.
    Args:
        acc (np.ndarray): Acceleration data.
        sync_cfg (dict): Synchronization configuration.
        clear_startup (int): Number of initial frames to ignore.
        peak_height (float): Minimum height of peaks.
        peak_distance (int): Minimum distance between peaks.
    Returns:
        peaks (np.ndarray): Indices of detected peaks.
    """
    acc = replace_exceeding_with_prev(acc, sync_cfg.get('peak_max_threshold', 1000))
    acc = gaussian_filter1d(acc, sigma=1)
    
    peaks, dict_ret = find_peaks(acc, height = peak_height, distance=peak_distance)
    if peaks.size == 0:
        print("[WARNING]: No peaks found in data for time synchronization. Using start of the sequence as start time.")
        return 0

    if sync_cfg["first_peak"]: # take the first peak
        peak = peaks[0] + clear_startup
    else: #highest peak
        peak = peaks[np.argmax(dict_ret['peak_heights'])] + clear_startup
    
    return peak
    

def get_exomc_start(data: dict, sync_cfg: dict, pip_cfg: PipelineConfig, **overrides) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Detecting the start of the sequence by the acceleration peak.
    """
    clear_startup = sync_cfg.get('clear_startup', 30)
    peak_height = sync_cfg.get('peak_height', 15)
    peak_distance = sync_cfg.get('peak_distance', 200)
    clap_period = overrides.get('clap_period', sync_cfg.get('clap_period', 10)) 
    
    time = data["Time"]  # in seconds
    sl = slice(clear_startup, clap_period * sync_cfg["freq_raw_data"])
    acc_lh = data["Acc"][:, 0][sl]
    acc_rh = data["Acc"][:, 1][sl]
    
    peak_lh = get_data_peaks(acc_lh, sync_cfg, clear_startup, peak_height, peak_distance)
    peak_rh = get_data_peaks(acc_rh, sync_cfg, clear_startup, peak_height, peak_distance)
    
    start_time = time[int(np.abs(peak_lh + peak_rh) / 2)] + sync_cfg.get("time_offset", 0.0)
    
    if pip_cfg.debug:
        print(f"[DEBUG]: exomc Left Hand Clap Peak Indices: {peak_lh + clear_startup}")
        print(f"[DEBUG]: exomc Right Hand Clap Peak Indices: {peak_rh + clear_startup}")
        plot_acceleration_data(acc_rh, time, start_time, truncate=clear_startup, label="Exo Right Hand")
        plot_acceleration_data(acc_lh, time, start_time, truncate=clear_startup, label="Exo Left Hand")
        
    return start_time, peak_lh, peak_rh

def get_egomc_start(data: dict, sync_cfg: dict, pip_cfg: PipelineConfig, **overrides) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Detecting the start of the sequence by the acceleration peak.
    """
    clear_startup = sync_cfg.get('clear_startup', 30)
    peak_height = sync_cfg.get('peak_height', 15)
    peak_distance = sync_cfg.get('peak_distance', 200)
    clap_period = overrides.get('clap_period', sync_cfg.get('clap_period', 10)) 
    
    time = data["Timestamp"] * 1e-3
    sl = slice(clear_startup, clap_period * sync_cfg["freq_raw_data"])
    
    acc_lhx = data['LeftHand_acceleration_x'][sl]
    acc_lhy = data['LeftHand_acceleration_y'][sl]  
    acc_lhz = data['LeftHand_acceleration_z'][sl] 
    acc_rhx = data['RightHand_acceleration_x'][sl]  
    acc_rhy = data['RightHand_acceleration_y'][sl] 
    acc_rhz = data['RightHand_acceleration_z'][sl] 
    acc_lh = np.sqrt(
        acc_lhx**2 + acc_lhy**2 + acc_lhz**2
        )
    acc_rh = np.sqrt(
        acc_rhx**2 + acc_rhy**2 + acc_rhz**2
        )
    
    peak_lh = get_data_peaks(acc_lh, sync_cfg, clear_startup, peak_height, peak_distance)
    peak_rh = get_data_peaks(acc_rh, sync_cfg, clear_startup, peak_height, peak_distance)
    
    start_time = time[int(np.abs(peak_lh + peak_rh) / 2)]
    
    if pip_cfg.debug:
        print(f"[DEBUG]: egomc Left Hand Clap Peak Indices: {peak_lh + clear_startup}")
        print(f"[DEBUG]: egomc Right Hand Clap Peak Indices: {peak_rh + clear_startup}")
        plot_acceleration_data(acc_rh, time, start_time, clear_startup, label="Ego Right Hand")
        plot_acceleration_data(acc_lh, time, start_time, clear_startup, label="Ego Left Hand")
        
    return start_time, peak_lh, peak_rh

def get_aria_start(aria_audio: dict, sync_cfg: dict, pip_cfg: PipelineConfig, **overrides) -> tuple[int, np.ndarray]:
    """
    Find the starting point of the camera by detecting the first peak in the audio file.
    """
    clap_period = overrides.get('clap_period', sync_cfg.get('clap_period', 10))
    
    num_frames_clap_period = clap_period * aria_audio["sample_rate"]
    num_frames_data = aria_audio["data"].shape[0]
    
    if num_frames_data <= num_frames_clap_period:
        num_frames_clap_period = num_frames_data

    data = aria_audio["data"][:num_frames_clap_period, :].astype(np.float64)
    audio_mean = np.sqrt(np.nanmean(np.square(data), axis=1))
    peaks, dict_ret = find_peaks(audio_mean, height = sync_cfg["peak_height"], distance=sync_cfg["peak_distance"])
    
    if peaks.size == 0:
        print("[WARNING]: No peaks found in Aria audio data for time synchronization. Using start of the sequence as start time.")
        start_idx = aria_audio["sample_rate"]  # to ensure camera calibration, skip first 30 frames
        start_time = aria_audio["timestamp"][start_idx]
        plot_audio_data(audio_mean, aria_audio["timestamp"] * 1e-9, start_time * 1e-9, truncate=0)
        return start_time, np.array([start_idx])
    
    start_idx = peaks[0]  # take the first detected peak for synchronization 
    #TODO find a way to automatically determine the correct peak if multiple peaks are detected (start_idx = peaks[np.argmax(dict_ret['peak_heights'])])
    start_time = aria_audio["timestamp"][start_idx]
    
    if pip_cfg.debug:
        print(f"[DEBUG]: Aria Audio Clap Peak Indices: {peaks}, Goal Index: {start_idx}")
        plot_audio_data(audio_mean, aria_audio["timestamp"] * 1e-9, start_time * 1e-9, truncate=0)
    return start_time, peaks
    
def test_time_sync_drift(egomc_sync_cfg, egomc_acc, aria_sync_cfg, aria_audio, pip_cfg):
    """
    Calculate clock drift between egomcko and Aria.
    """
    _, egomc_idxs, egomc_idxs_rh = get_egomc_start(egomc_acc, egomc_sync_cfg, pip_cfg, clap_period=10000)
    _, aria_idxs = get_aria_start(aria_audio.copy(), aria_sync_cfg, pip_cfg, clap_period=10000)
    timestamp_aria = aria_idxs / aria_audio["sample_rate"]
    timestamp_egomc = egomc_idxs / egomc_sync_cfg["freq_raw_data"]
    timestamp_egomc_rh = egomc_idxs_rh / egomc_sync_cfg["freq_raw_data"]
    
    timestamp_aria_shift = timestamp_aria - timestamp_aria[0]
    timestamp_egomc_shift = timestamp_egomc - timestamp_egomc[0]
    timestamp_egomc_rh_shift = timestamp_egomc_rh - timestamp_egomc_rh[0]
    
    plot_clap_detection(egomc_acc, aria_audio, egomc_idxs, egomc_idxs_rh, aria_idxs, egomc_idxs[0], aria_idxs[0], peak_max_threshold=150)
    plot_time_sync_drift(timestamp_aria_shift, timestamp_egomc_shift, timestamp_egomc_rh_shift, timestamp_egomc_rh[0])