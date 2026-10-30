import os
import json
import numpy as np
from projectaria_tools.core.sophus import SE3
from scipy.spatial.transform import Rotation as R
from scipy.io import wavfile

from projectaria_tools.core.sensor_data import TimeDomain
from projectaria_tools.core.vrs import extract_audio_track

def get_aria_first_pose_transform(trajectory, T_device_RGB, idx):
    pose = trajectory[idx]
    #90° clockwise rotation around Z
    rotz = R.from_euler('z', -90, degrees=True).as_matrix()
    Rz_90_mat = np.eye(4)
    Rz_90_mat[:3, :3] = rotz
    Rz_90 = SE3.from_matrix(Rz_90_mat)
    
    #90 clockwise rotation around x (origin cam pose, points in negative z direction)
    rotx = R.from_euler('x', 90, degrees=True).as_matrix()  
    Rx_90_mat = np.eye(4)
    Rx_90_mat[:3, :3] = rotx
    Rx_90 = SE3.from_matrix(Rx_90_mat)
    
    T_w_rgb_inv = np.linalg.inv((pose.transform_world_device @ T_device_RGB @ Rz_90 @ Rx_90).to_matrix())
    return T_w_rgb_inv

def extract_audio(
        ctx,
        provider, 
        sensor_stream_id,
        vrs_file_path: str,
        multi: bool = False) -> dict:
    """Extract audio from a VRS file as a wav file."""
        
    # vrs audio extraction
    os.makedirs(ctx.aria_audio_dir, exist_ok=True)
    file_prefix = 'audio_2nd' if multi else 'audio'
    wavefile_path = os.path.join(ctx.aria_audio_dir, f"{file_prefix}.wav")
    wavejson_path = os.path.join(ctx.aria_audio_dir, f"{file_prefix}.wav.json")
    if not os.path.exists(wavefile_path):
        extract_audio_track(vrs_file_path, wavefile_path)
    sample_rate, data = wavfile.read(wavefile_path)
    with open(wavejson_path, "r") as f:
        audio_info = json.load(f)

    time_domain = TimeDomain.DEVICE_TIME 
    #first audio data recording start
    first_audio_ts = provider.get_first_time_ns(sensor_stream_id, time_domain)
    #after last audio data is saved 
    last_audio_ts = provider.get_last_time_ns(sensor_stream_id, time_domain) + audio_info.get("last_audio_record_duration") * 1e9
    dt = 1.0 / sample_rate * 1e9  # time step between samples
    timestamp = np.arange(first_audio_ts, last_audio_ts, dt)
    return {'sample_rate': sample_rate, 'data': data, "timestamp": timestamp[:-1]}
