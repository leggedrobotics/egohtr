import numpy as np
import json

from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, filtfilt

from src.loader.base_loader import DataLoader
from src.loader .time_sync import get_exomc_start
from src.loader.utils import transform_points

class ExoMoCapLoader(DataLoader):
    """
    Data loader for exocentric motion capture data (ExoMoCap). This loader is responsible for loading the raw JSON data, processing it to extract joint poses and marker trajectories,
    performing time synchronization based on clap events, and preparing the data for export.
    """
    start_frame_idx: int = None
    end_frame_idx: int = None
    duration: float = None

    def load_data(self):
        # Implement ExoMoCap data loading logic here
        print(f"[INFO]: Loading ExoMoCap data")
        self.unit_scale = self.sensor_cfg["unit_scale"]
        self.load_json()
        self.load_sensor_timestamp()

    def process(self):
        print("[INFO]: Processing ExoMoCap data")
        self.get_joint_poses()
        self.align_traj, self.align_traj_timestamp, _ = self.get_marker_traj(self.sensor_cfg['marker_align'], duration=50)
        self.get_headFP()
        
    
    def load_json(self):
        """Extract skeleton, joint poses, parents and joint names from ExoMoCap JSON."""
        with open(self.ctx.exomc_json, 'r') as f:
            self.global_data = json.load(f)
            

    def load_sensor_timestamp(self):
        fps = self.sensor_cfg["fps"]
        range = self.global_data["Timebase"]["Range"]
        self.start_frame = range["Start"]
        self.nframes = range["End"]
        self.timestamp = np.arange(self.nframes) / fps


    def time_sync_sensor(self):
        # Implement time synchronization logic here
        acc = self.get_acceleration()
        exomc_start_time, _, _= get_exomc_start(acc, self.sensor_cfg['time_sync'], self.pip_cfg)
        self.start_frame_idx = int(np.searchsorted(self.timestamp, exomc_start_time)) + self.start_frame - 1
        self.end_frame_idx = self.timestamp.shape[0]
        self.duration = (self.end_frame_idx - self.start_frame_idx) / self.sensor_cfg["fps"]
        
        try: 
            clap_time = self.get_clap_time(self.sensor_cfg["time_sync"])
        except Exception as e:
            print(f"[WARNING]: No Clap marker found in ExoMoCap data. Time sync verification skipped.")
            clap_time = exomc_start_time
        
        # assert np.isclose(clap_time, exomc_start_time, atol=0.05), "ExoMoCap clap time does not match time sync result."
    

    def get_joint_poses(self):
        """Extract joint poses, parents and joint names from ExoMoCap JSON."""
        skeletons_global = self.global_data.get('Skeletons')
        if not skeletons_global:
            raise ValueError("No 'Skeletons' entry found in ExoMoCap JSON.")

        pose_entries = skeletons_global[0].get('Pose', [])
        segment_entries_global = skeletons_global[0].get('Segments', [])
        if not pose_entries or not segment_entries_global:
            raise ValueError("Skeleton entry has no 'Pose' or 'Segments' data.")

        # Joint names (segments) from the pose list
        self.joint_names = [p['Name'] for p in pose_entries]
        name_to_idx = {name: idx for idx, name in enumerate(self.joint_names)}

        # Parents: map parent joint name to index, -1 for roots
        parents = []
        t_transforms = []
        self.global_pose = np.tile(np.eye(4, dtype=np.float32)[np.newaxis, np.newaxis, :, :], (self.timestamp.shape[0], len(self.joint_names), 1, 1))
        # zero pos and identity quat in case of no skeleton data
        self.global_pos = np.zeros((self.timestamp.shape[0], len(self.joint_names), 3), dtype=np.float32)
        self.global_quat = np.zeros((self.timestamp.shape[0], len(self.joint_names), 4), dtype=np.float32)
        self.global_quat[..., 3] = 1.0

        for pose_idx, (pose, seg_global) in enumerate(zip(pose_entries, segment_entries_global)):
            parent_name = pose.get('Parent')
            t_transforms.append(pose.get('Transform'))
            parts = seg_global.get('Parts')
            
            for part in parts:
                range = part.get('Range')
                start, end = range.get('Start'), range.get('End')
                values = np.array(part.get('Values'))
                self.global_pos[start-1:end, pose_idx, :] = values[:, :3]
                self.global_quat[start-1:end, pose_idx, :] = values[:, 3:]

            if parent_name is None:
                parents.append(-1)
                self.get_pose_mask(parts)
            else:
                parents.append(name_to_idx.get(parent_name, -1))

        sl = slice(self.start_frame_idx, self.end_frame_idx)
        self.parents = np.array(parents, dtype=np.int32)
        self.t_pose = np.array(t_transforms, dtype=np.float32)
        #generate global pose
        g_rot = R.from_quat(self.global_quat.reshape(-1,4)).as_matrix().reshape(self.global_quat.shape[0], len(self.joint_names), 3,3)
        self.global_pose[:, :, :3, 3] = self.global_pos * self.unit_scale
        self.global_pose[:, :, :3, :3] = g_rot
        self.global_pose = self.global_pose[sl]
        self.global_pos = self.global_pos[sl] * self.unit_scale
        self.global_quat = self.global_quat[sl]
    

    def get_pose_mask(self, parts: list):
        """ Get the valid pose mask for the root joint based on the parts range"""
        self.mask = np.zeros((self.timestamp.shape[0],), dtype=bool)
        for part in parts:
            range = part.get('Range')
            start, end = range.get('Start'), range.get('End')
            self.mask[start:end] = True
        self.mask = self.mask[self.start_frame_idx:self.end_frame_idx]
    
    
    def get_marker_traj(self, marker_postfix: str, duration: int) -> tuple[np.ndarray, np.ndarray]:
        """ Get marker trajectory with length defined by 'duration' starting at range 'Start' defined by exomc sensor caption.
        
        Returns: 
                points of trajectory (N,3), if marker is not detected, points are filtered out.
                time of trajectory (N,) 
                residuals of trajectory (N,)
        """

        fps = self.sensor_cfg["fps"]
        frames = int(duration * fps)
        points = np.zeros((frames, 3), dtype=np.float32)
        residuals = np.zeros((frames), dtype=np.float32)


        markers_parts = [m['Parts'] for m in self.global_data['Markers'] if m['Name'].endswith(marker_postfix)]
        if markers_parts == []:
            raise ValueError(f"No marker found with postfix '{marker_postfix}' in ExoMoCap JSON.")
        
        for part in markers_parts[0]:
            range = part.get('Range')
            start, end = range.get('Start') - self.start_frame, range.get('End') - self.start_frame
            if start > frames:
                break
            elif end > frames: 
                end = frames

            points[start:end, :] = np.array(part.get('Values'))[:(end-start), :3]
            residuals[start:end] = np.array(part.get('Values'))[:(end-start), 3]


        points *= self.unit_scale
        residuals *= self.unit_scale
        time = np.zeros_like(points) 
        time = np.arange(points.shape[0]) / fps

        #filter non detected points
        mask = np.any(points != 0, axis=-1)

        return points[mask], time[mask], residuals[mask]
        

    def get_marker_fp(self, marker_postfix: str) -> np.ndarray:
        """ 
        Get the markers first point at start_frame_idx.
        Returns: points of trajectory (1,3), if marker is not detected, points is defined as (0,0,0)
        """

        markers_parts = [m['Parts'] for m in self.global_data['Markers'] if m['Name'].endswith(marker_postfix)]
        if markers_parts == []:
            raise ValueError(f"No marker found with postfix '{marker_postfix}' in ExoMoCap JSON.")
        
        for part in markers_parts[0]:
            range = part.get('Range')
            start, end = range.get('Start'), range.get('End')
            if start <= self.start_frame_idx <= end+1:
                point = np.array(part.get('Values'))[(self.start_frame_idx-1 - start), :3]
                return point
                
        raise ValueError(f"Marker with postfix '{marker_postfix}' not found at start frame index {self.start_frame_idx}.")


        
    def get_marker_range(self, marker_postfix: str) -> tuple[np.ndarray, np.ndarray]:
        """ Get the min and max range idx and time of a marker"""
        markers_parts = [m['Parts'] for m in self.global_data['Markers'] if m['Name'].endswith(marker_postfix)]
        start = markers_parts[0][0]['Range']['Start']
        end = markers_parts[0][-1]['Range']['End']
        range_idx = np.array([start, end])
        range_time = range_idx / self.sensor_cfg["fps"]
        return range_idx, range_time
    
        
    def get_headFP(self):
        """ Load two head fixed points for initial heading direction """
        markers = self.sensor_cfg['marker_init']
        num_markers = len(markers)
        self.headFP = np.zeros((num_markers, 3))
        for i, marker in enumerate(markers):
            point = self.get_marker_fp(marker)    
            self.headFP[i, :] = point.squeeze() * self.unit_scale
    

    def pad_to_nframes(self, points: np.ndarray, n: int) -> np.ndarray:
        """Pad or truncate the input points to have exactly n frames along the time axis."""
        # points: (T, D), pad/truncate along time axis to length n
        if points.shape[0] >= n:
            return points[:n]

        pad_len = n - points.shape[0]
        last = points[-1:]                     # shape (1, D)
        pad = np.repeat(last, pad_len, axis=0)
        return np.concatenate([points, pad], axis=0)


    def get_acceleration(self, cutoff_hz: float = 5.0, order: int = 2) -> dict:
        """Low-pass filter LeftHand/RightHand trajectories and compute 2nd derivative."""
        fps = float(self.sensor_cfg["fps"])
        dt = 1.0 / fps

        # Design Butterworth low-pass filter
        nyq = 0.5 * fps
        norm_cutoff = cutoff_hz / nyq
        if not 0.0 < norm_cutoff < 1.0:
            raise ValueError("cutoff_hz must be between 0 and Nyquist frequency (fps/2).")
        b, a = butter(order, norm_cutoff, btype="low")

        # Load marker trajectories
        lhand_marker = self.pad_to_nframes(self.get_marker_traj("LHandOut", duration = 45)[0], self.timestamp.shape[0]) * self.unit_scale
        rhand_marker = self.pad_to_nframes(self.get_marker_traj("RHandOut", duration = 45)[0], self.timestamp.shape[0]) * self.unit_scale  # (T_r, 3)

        # Shape: (nframes, 2, 3) with axis 0 = time
        hand_marker = np.stack([lhand_marker, rhand_marker], axis=1)
        filt_pos = filtfilt(b, a, hand_marker, axis=0)

        # First and second temporal derivatives using central differences
        vel = np.gradient(filt_pos, dt, axis=0)
        acc = np.gradient(vel, dt, axis=0)
        return {"Time": self.timestamp, "Acc": np.linalg.norm(acc, axis=1)}
    
    
    def get_clap_time(self, sync_cfg: dict) -> float:
        clap_marker_range = self.get_marker_range("Clap")[1]
        clap_time_s = clap_marker_range[-1] + sync_cfg.get("time_offset", 0.0) #offset: covers marker slighly before clap sound
        clap_time_s -= self.start_frame / self.sensor_cfg["fps"] #add start time offset
        return clap_time_s
    
    
    def spatial_align_data(self, transform: np.ndarray, init_ego_target_pos: np.ndarray):
        """Align the Exomocap data to a given transformation."""
        print("[INFO]: Aligning ExoMocap data")
        self.global_pose = np.einsum('ik, fjkl -> fjil', transform, self.global_pose)
        # correct translational offset (TODO add flag to choose between manual implementation or correction by aligning the ego and exo)
        offset = init_ego_target_pos - self.get_joint_pos("Head")[0]  # translation from first joint (root) to target position
        self.global_pose[:, :, :3, 3] += offset  # apply the same offset to all joints and frames
        self.align_traj = transform_points(self.align_traj, transform)
        
    
    def get_export_data(self) -> dict:
        """Prepare ExoMoCap data for export."""
        payload = {
            "exomc_global_poses": self.global_pose[::10],
            "exomc_traj": self.align_traj,
            "exomc_skeleton": self.get_skeleton_bones(),
            "exomc_timestamp": self.get_synced_timestamp()[::10],
            "exomc_mask": self.mask[::10], #TODO, add to config the option to export mask or not, and if not, remove it from the payload
        }
        return payload
    

    def get_skeleton_bones(self) -> np.ndarray: 
        bones = [[idx, parent] for idx, parent in enumerate(self.parents) if parent != -1]
        return np.array(bones)
    
    def get_joint_pos(self, name:str) -> np.ndarray:
        """Get the global position of a joint by name."""
        idx = self.joint_names.index(name)
        return self.global_pose[:, idx, :3, 3]