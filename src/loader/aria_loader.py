import os
import json
import numpy as np

from typing import Optional
from pathlib import Path
from scipy.io import wavfile

from projectaria_tools.core import data_provider, mps, calibration
from projectaria_tools.core.mps.utils import filter_points_from_confidence
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core import calibration
from projectaria_tools.core.calibration import CameraCalibration, DeviceCalibration
from projectaria_tools.core.sensor_data import TimeDomain
from projectaria_tools.core.mps.utils import get_nearest_pose, bisection_timestamp_search

from src.loader.base_loader import DataLoader    
from src.loader.time_sync import get_aria_start
from src.loader.utils import save_image_to_dir, transform_points
from src.loader.helper.aria_helper import get_aria_first_pose_transform
from src.loader.helper.scene_helper import downsample_cloud   
from src.loader.helper.helper import get_fixed_glass_translation     
from projectaria_tools.core.vrs import extract_audio_track 


class AriaLoader(DataLoader):
    """
    Loader for ARIA data sequences. This loader is responsible for loading the raw ARIA data including point clouds, RGB images, trajectories, 
    and audio, processing it to extract relevant features such as undistorted and aligned images. It also handles time synchronization of the ARIA data based on the audio clap event and prepares the data for export.
    It handles both the subject and observer roles.
    """
    
    device_calibration: DeviceCalibration
    rgb_camera_calibration : CameraCalibration #undistorted and rotated
    
    aria_pcd : np.ndarray
    aria_trajectory : np.ndarray
    aria_audio : dict
    
    start_frame_idx : int
    start_frame_ns : float
            
    def load_data(self):
        """Load ARIA data from the specified sequence context."""
        print(f"[INFO]: Loading ARIA {self.ctx.role} data")
 
        # Create data provider
        self.provider = data_provider.create_vrs_data_provider(str(self.ctx.aria_vrs))
        self.mps_data_provider = mps.MpsDataProvider(mps.MpsDataPathsProvider(str(self.ctx.aria_mps_dir)).get_data_paths())
        
        self.device_calibration = self.provider.get_device_calibration()
        self.aria_mps_trajectory = mps.read_closed_loop_trajectory(str(self.ctx.aria_closed_loop_traj))
        self.aria_online_calibrations = mps.read_online_calibration(str(self.ctx.aria_online_calib))
        self._extract_audio(str(self.ctx.aria_vrs))
        
        if self.ctx.role == "sub":
            self.aria_T_glass_RGB = self.provider.get_device_calibration().get_transform_device_sensor(
                "camera-rgb"
            )
            self.aria_pcd = mps.read_global_point_cloud(str(self.ctx.global_pcd))


    def process(self):
        """Process the loaded ARIA data."""
        print(f"[INFO]: Processing ARIA {self.ctx.role} data")
        self.aria_pcd = filter_points_from_confidence(self.aria_pcd) #mandatory according to project aria
        aria_pcd_w = np.stack([it.position_world for it in self.aria_pcd])
        self.aria_pcd = downsample_cloud(aria_pcd_w, voxel_size=self.sensor_cfg["processing"]["voxel_size"]) # 2.5cm
        
        # Remove points that are not within a limiting box
        bounding_box_min = np.array([-10.0, -10.0, -2.0])  # example values
        bounding_box_max = np.array([10.0, 10.0, 4.0])  # example values
        within_box = np.all((self.aria_pcd >= bounding_box_min) & (self.aria_pcd <= bounding_box_max), axis=1)
        self.aria_pcd = self.aria_pcd[within_box]


    def _extract_audio(self, vrs_file_path: str) -> Optional[str]:
        """Extract audio from a VRS file as a wav file."""
        
        # vrs audio extraction
        os.makedirs(self.ctx.aria_audio_dir, exist_ok=True)
        file_prefix = 'audio_obs' if self.ctx.role == "obs" else 'audio'
        wavefile_path = os.path.join(self.ctx.aria_audio_dir, f"{file_prefix}.wav")
        wavejson_path = os.path.join(self.ctx.aria_audio_dir, f"{file_prefix}.wav.json")
        if not os.path.exists(wavefile_path):
            extract_audio_track(vrs_file_path, wavefile_path)
        sample_rate, data = wavfile.read(wavefile_path)
        with open(wavejson_path, "r") as f:
            audio_info = json.load(f)

        sensor_stream_id = StreamId(self.sensor_cfg["streams"]["audio"])
        time_domain = TimeDomain.DEVICE_TIME 
        #first audio data recording start
        first_audio_ts = self.provider.get_first_time_ns(sensor_stream_id, time_domain)
        #after last audio data is saved 
        last_audio_ts = self.provider.get_last_time_ns(sensor_stream_id, time_domain) + audio_info.get("last_audio_record_duration") * 1e9
        dt = 1.0 / sample_rate * 1e9  # time step between samples
        timestamp = np.arange(first_audio_ts, last_audio_ts, dt)
        self.aria_audio = {'sample_rate': sample_rate, 'data': data, "timestamp": timestamp[:-1]}
        

    def time_sync_sensor(self, preloaded: bool = False):
        print(f"[INFO]: Time synchronizing ARIA {self.ctx.role} data")
        sensor_stream_id = StreamId(self.sensor_cfg["streams"]["rgb"])
        
        if not preloaded:
            time_sync_config = self.sensor_cfg["time_sync"] if self.ctx.role == "sub" else self.sensor_cfg["time_sync_obs"]
            self.start_frame_ns, aria_audio_idx = get_aria_start(self.aria_audio, time_sync_config, self.pip_cfg)
            #compute start and end frame index for RGB stream
            self.start_frame_idx = self.provider.get_index_by_time_ns(sensor_stream_id, int(self.start_frame_ns), TimeDomain.DEVICE_TIME)
        
        self.end_frame_ns = self.provider.get_last_time_ns(sensor_stream_id, TimeDomain.DEVICE_TIME)
        self.end_frame_idx = self.provider.get_index_by_time_ns(sensor_stream_id, self.end_frame_ns, TimeDomain.DEVICE_TIME)
        self.duration = (self.end_frame_idx - self.start_frame_idx) / self.sensor_cfg["fps"]
    
        
    def get_synced_sensor_data(self):
        """Output sensor data synchronized based on clap timestamp."""
        time_domain = TimeDomain.DEVICE_TIME  # query data based on DEVICE_TIME
        sensor_stream_id = StreamId(self.sensor_cfg["streams"]["rgb"])
        timestamps = self.provider.get_timestamps_ns(sensor_stream_id, TimeDomain.DEVICE_TIME)
        start_time = self.provider.get_first_time_ns(sensor_stream_id, time_domain)
        start_time_offset = self.start_frame_ns - start_time
        options = (self.provider.get_default_deliver_queued_options())  # default options activates all streams
        options.set_truncate_first_device_time_ns(int(start_time_offset)) # truncate data before clap
        
        
        end_time = self.provider.get_last_time_ns(sensor_stream_id, time_domain)
        self.end_frame_ns = timestamps[self.end_frame_idx - 2] # todo: verify end frame index
        end_time_offset = end_time - self.end_frame_ns
        options.set_truncate_last_device_time_ns(int(np.ceil(end_time_offset)))  # truncate data after last frame
        options.deactivate_stream_all()
        options.activate_stream(sensor_stream_id)
        return self.provider.deliver_queued_sensor_data(options)
    
    
    def get_export_data(self):
        """Get processed data for export."""
        if self.ctx.role == "sub":
            payload = {
                "aria_pcd": self.aria_pcd,
                "aria_rgb_imgs": self.get_aligned_imgs(),
                "aria_rgb_img_width": self.rgb_camera_calibration.get_image_size()[0] / self.sensor_cfg["processing"]["downsampling_factor"],
                "aria_rgb_img_height": self.rgb_camera_calibration.get_image_size()[1] / self.sensor_cfg["processing"]["downsampling_factor"],
                "aria_rgb_focal_length": self.rgb_camera_calibration.get_focal_lengths()[0] / self.sensor_cfg["processing"]["downsampling_factor"],
                "aria_trajectory": self.aria_trajectory,
                "aria_glasses_outline": self.get_glasses_outline(),
                "aria_T_origin_device": self.T_origin_device,
                "aria_T_device_camera": self.get_camera2device_transform(),
                "aria_t_origin_deviceE": self.get_eyes2origin_transl(self.seq_cfg["eyes2device"]),
                "aria_timestamp": self.get_synced_timestamp()
                    
            }
        else:
            imgs = self.get_aligned_imgs()
            payload = {
                "aria_obs_rgb_imgs": imgs,
                "aria_obs_rgb_img_width": self.rgb_camera_calibration.get_image_size()[0] - (2 * self.sensor_cfg["processing"]["crop_px"]),
                "aria_obs_rgb_img_height": self.rgb_camera_calibration.get_image_size()[1] - (2 * self.sensor_cfg["processing"]["crop_px"]),
                "aria_obs_rgb_focal_length": self.rgb_camera_calibration.get_focal_lengths()[0],
                "aria_obs_trajectory": self.aria_trajectory,
                "aria_obs_T_origin_device": self.T_origin_device,
                "aria_obs_T_device_camera": self.get_camera2device_transform(),
            }
            
            # Save first image for sync purposes
            img_path = self.ctx.aria_imgs_dir 
            save_image_to_dir(imgs[0], img_path,  filename=f"aria_obs_init")
        return payload
    

    def _get_linear_calib(self, raw_calib, rotate: bool = True) -> CameraCalibration:
        """Converts any Aria Calibration (Factory or Online) to a Linear Pinhole model."""
        w, h = raw_calib.get_image_size()
        
        # Create the standard pinhole model (Rectified)
        linear_calib = calibration.get_linear_camera_calibration(
            int(w), int(h),
            raw_calib.get_focal_lengths()[0],
            self.sensor_cfg["calibration"]["model"],
            raw_calib.get_transform_device_camera(),
        )
        
        # Standardize rotation for head-mounted portrait-to-landscape conversion
        if rotate:
            return calibration.rotate_camera_calib_cw90deg(linear_calib)
        else:
            return linear_calib
    
    
    def set_camera_calibration(self, use_online: bool = True):
        """Sets the persistent camera models used for the sequence."""
        rgb_stream_id = StreamId(self.sensor_cfg["streams"]["rgb"])
        label = self.provider.get_label_from_stream_id(rgb_stream_id)
        
        # Prefer Online (MPS) calibration to handle dynamic flexing of the glasses
        if use_online:
            ts = self.provider.get_first_time_ns(rgb_stream_id, TimeDomain.DEVICE_TIME)
            raw_calib = self.mps_data_provider.get_online_calibration(ts).get_camera_calib(label)
        else:
            raw_calib = self.provider.get_device_calibration().get_camera_calib(label)

        self.rgb_camera_calibration = self._get_linear_calib(raw_calib, rotate=True) 
    

    def get_undistorted_img(self, data, down_sampling_factor: int = 1) -> np.ndarray:
        """
        Rectifies, rotates, and downsamples a raw sensor data object.
        Used for real-time processing during the main loop.
        """
        # Extract raw image and its specific capture timestamp
        img = data[0].to_numpy_array()
        ts = data[1].capture_timestamp_ns
        # img = data.image_data_and_record()[0].to_numpy_array()
        # ts = data.image_data_and_record()[1].capture_timestamp_ns
        
        label = self.provider.get_label_from_stream_id(StreamId(self.sensor_cfg["streams"]["rgb"]))
        online_calib = self.mps_data_provider.get_online_calibration(ts).get_camera_calib(label)
        
        # Rectify img using online calibration
        target_linear = self._get_linear_calib(online_calib, rotate=False)
        img = calibration.distort_by_calibration(img, target_linear, online_calib)
        img = np.rot90(img, k=self.sensor_cfg["calibration"]["rotation_k"])
        
        # Vectorized downsampling (zero-copy)
        if down_sampling_factor > 1:
            img = img[::down_sampling_factor, ::down_sampling_factor]

        if self.ctx.role == 'obs': #remove 10 pixels from each border to avoid distortion artifacts
            crop = self.sensor_cfg["processing"]["crop_px"]
            img = img[crop:-crop, crop:-crop]
                    
        return img


    def save_undistorted_img(self, img_path: Path, num_imgs: int = 5) -> int | None:
        """
        Saves a set of undistorted images around the sync point.
        
        returns:
            The timestamp index of the first saved image.
        """
        print(f"[INFO]: Saving {num_imgs} FP images to {img_path}")
        rgb_id = StreamId(self.sensor_cfg["streams"]["rgb"])
        
        fp_idx = None
        
        for i in range(num_imgs):
            # Sample images every second around the clap event
            frame_idx = int(self.start_frame_idx + self.sensor_cfg["fps"] * (i)) # start from a second after clap
            
            try:
                raw_data = self.provider.get_image_data_by_index(rgb_id, frame_idx)
                img = self.get_undistorted_img(raw_data)
                
                save_image_to_dir(img, img_path, filename=f"aria_init_{i}")
            except Exception as e:
                print(f"[WARNING]: Could not extract QC image at index {frame_idx}: {e}")
        
            if fp_idx is None:
                fp_idx = frame_idx #first pose timestamp index
                
        return fp_idx
        
    
    def get_cam_fp(self, timestamp_idx) -> np.ndarray:
        """Get the first pose of the ARIA camera at the first saved image."""
        return get_aria_first_pose_transform(self.aria_mps_trajectory, self.aria_T_glass_RGB, timestamp_idx)

    
    def get_device2origin_transform_aria(self, T_origin_ariaW: np.ndarray):
        """Get the device trajectory transforms from MPS data at aria frequency.""" 
        T_world_device = np.zeros((self.nframes, 4,4))
        self.set_camera_calibration()
        for idx, data in enumerate(self.get_synced_sensor_data()):
            aria_time_ns = data.get_time_ns(TimeDomain.DEVICE_TIME)
            pose_info = get_nearest_pose(self.aria_mps_trajectory, aria_time_ns)
            if pose_info is None:
                break
            T_world_device[idx] = pose_info.transform_world_device.to_matrix()
        self.T_origin_device = T_origin_ariaW @ T_world_device
    
    def get_device2origin_transform(self, T_origin_ariaW: np.ndarray, nframes: int, fps: int) -> np.ndarray:
        """Get the device trajectory transforms from MPS data at given frequency.""" 
        T_world_device = np.zeros((nframes, 4,4))
        for idx in range(nframes):
            aria_time_ns = self.start_frame_ns + idx * (1e9 / fps)
            pose_info = get_nearest_pose(self.aria_mps_trajectory, aria_time_ns)
            if pose_info is None:
                break
            T_world_device[idx] = pose_info.transform_world_device.to_matrix() 
        return T_origin_ariaW @ T_world_device

    
    def get_camera2device_transform(self) -> np.ndarray:
        """Get the camera to device transform."""
        return self.rgb_camera_calibration.get_transform_device_camera().to_matrix()
    
    
    def spatial_align_data(self, transform):
        self.aria_pcd = self.get_aligned_pcd(transform) if self.ctx.role == "sub" else None
        self.aria_trajectory = self.get_aligned_trajectory(transform)
            
            
    def get_aligned_pcd(self, transform: np.ndarray) -> np.ndarray:
        """Get the ARIA point cloud transformed to the scene world frame."""
        return transform_points(self.aria_pcd, transform)
        
        
    def get_aligned_imgs(self) -> np.ndarray:
        """Get undistorted and aligned images from ARIA data."""
        self.set_camera_calibration()
        h, w = self.rgb_camera_calibration.get_image_size()
        
        if self.ctx.role == "sub":
            factor = self.sensor_cfg["processing"]["downsampling_factor"]
            imgs = np.zeros((self.nframes,
                         h // factor,
                         w // factor,
                         3), dtype=np.uint8)
        else:
            factor = 1
            imgs = np.zeros((self.nframes,
                         h - (2 * self.sensor_cfg["processing"]["crop_px"]),
                         w - (2 * self.sensor_cfg["processing"]["crop_px"]),
                         3), dtype=np.uint8)
                   
        
        for idx, data in enumerate(self.get_synced_sensor_data()):
            imgs[idx] = self.get_undistorted_img(data.image_data_and_record(), factor)
        return imgs


    def get_aligned_trajectory(self, transform: np.ndarray) -> np.ndarray:
        """Get the ARIA camera trajectory transformed to the scene world frame."""
        start_idx = self.get_traj_start_idx()
        end_idx = self.get_traj_end_idx()
        trajectory_data = self.aria_mps_trajectory[start_idx:end_idx]
        trajectory = [it.transform_world_device.translation()[0] for it in trajectory_data][0::20]
        return transform_points(np.array(trajectory), transform)
    
    def get_freq_aligned_trajectory(self, nframes: int, fps: int) -> np.ndarray:
        """Get the ARIA camera trajectory at given frequency."""
        trajectory = np.zeros((nframes, 3))
        for idx in range(nframes):
            aria_time_ns = self.start_frame_ns + idx * (1e9 / fps)
            pose_info = get_nearest_pose(self.aria_mps_trajectory, aria_time_ns)
            trajectory[idx] = pose_info.transform_world_device.translation()[0]
        return trajectory
    
    def get_time_aligned_trajectory(self, timestamps: np.ndarray) -> np.ndarray:
        """Get the ARIA camera trajectory at given frequency."""
        trajectory = np.zeros((timestamps.shape[0], 3))
        for idx, time in enumerate(timestamps):
            aria_time_ns = self.start_frame_ns + time * 1e9
            pose_info = get_nearest_pose(self.aria_mps_trajectory, aria_time_ns)
            trajectory[idx] = pose_info.transform_world_device.translation()[0]
        return trajectory
    
    
    def get_traj_start_idx(self):
        """Get start and end index of the ARIA trajectory based on time sync."""
        return bisection_timestamp_search(self.aria_mps_trajectory, int(self.start_frame_ns))
    
    def get_traj_end_idx(self):
        """Get start and end index of the ARIA trajectory based on time sync."""
        return bisection_timestamp_search(self.aria_mps_trajectory, int(self.end_frame_ns))
    

    def get_glasses_outline(self, use_cad_calib: bool = True) -> [np.ndarray]:
        """Return a list of points to be used to draw the outline of the glasses (line strip)."""
        glasses_outline_indexes = [
            "mic5",
            "camera-slam-left",
            "mic2",
            "mic1",
            "baro0",
            "mic1",
            "camera-slam-left",
            "camera-slam-right",
            "mic0",
            "baro0",
            "camera-slam-right",
            "mic6",
        ]
        points = []
        for index in glasses_outline_indexes:
            T_device_sensor = self.device_calibration.get_transform_device_sensor(
                index, True if "mic" or "mag" or "baro" in index else use_cad_calib
            )
            points.append(T_device_sensor.translation()[0])
        return points
    

    def get_eyes2origin_transl(self, cfg: dict) -> np.ndarray:
        
        t_origin_device = np.repeat(self.T_origin_device[:, :3, 3][:, :, np.newaxis], 2, axis=-1)  # repeat for two eyes
        
        t_device2deviceE_local = get_fixed_glass_translation(cfg["dist2glass"], cfg["height_trans"], cfg["joint"])  # translation from device to eyes joint
        t_device2deviceE = self.T_origin_device[:, :3, :3] @ t_device2deviceE_local  # translation from device to eyes joint
        t_origin_deviceE = t_origin_device + t_device2deviceE
        return t_origin_deviceE


