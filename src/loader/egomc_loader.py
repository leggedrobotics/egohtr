import numpy as np
import pandas as pd

from typing import Dict
from bvh import Bvh

from src.loader.base_loader import DataLoader
from src.loader.time_sync import get_egomc_start
from src.loader.helper.egomc_helper import seperate_channels
from src.loader.helper.helper import get_rotation_chain_vectorized
from src.visualizer.utils import plot_vel
from src.loader.utils import euler_to_quat, remove_quat_discontinuities


class EgoMoCapLoader(DataLoader):
    """
    Loader of the egocentric motion capture data (EgoMoCap). This loader is responsible for loading the raw BVH and CSV data, processing it to extract joint poses,
    performing time synchronization based on clap events, and preparing the data for export.
    """
    
    egomc_bvh: Bvh
    egomc_raw_data: Dict[str, np.ndarray]
    
    timestamp: np.ndarray
    joints: list
    root_pos: np.ndarray
    local_poses: np.ndarray
    global_poses: np.ndarray
    
    footlock: Dict[str, np.ndarray]
    footlock_soft: Dict[str, np.ndarray]
    
    start_frame_idx: int
    
    def load_data(self):
        # Implement EgoMocap data loading logic here
        print(f"[INFO]: Loading EgoMocap data")
        self.egomc_bvh = Bvh(self.ctx.egomc_bvh.read_text())
        dataframe = pd.read_csv(self.ctx.egomc_csv)
        self.egomc_raw_data = {col : dataframe[col].to_numpy() for col in dataframe.columns}
        self.timestamp = np.arange(self.egomc_bvh.nframes-1) * self.egomc_bvh.frame_time
        
    def process(self):
        print("[INFO]: Processing EgoMocap data")
        self.get_joint_poses()
        self.get_wolrd_poses()
        self._get_footlock()
        
    
    def get_joint_poses(self):
        """Compute global (forward kinematics)and local joint poses over time for a specific joint."""
        self.joints = self.egomc_bvh.get_joints_names()
        frames = np.array(self.egomc_bvh.frames).astype('float32')
        
        #separate the rotation angles and the positions of joints (if any)
        local_joint_rot, joint_rot_order = seperate_channels(frames, self.joints, self.egomc_bvh.joint_channels, target_type='rotation', seq_cfg=self.seq_cfg)
        #convert euler angles to quaternions and remove discontinuities
        self.local_joint_rot_quat = euler_to_quat(np.radians(local_joint_rot), order=joint_rot_order['Hips'])
        self.local_joint_rot_quat = remove_quat_discontinuities(self.local_joint_rot_quat)
        #get root position (assuming it's in the first 3 channels of the root joint)
        self.root_pos = frames[:, 0:3] * self.sensor_cfg['distance_scale']
        
        joints_offsets = {}
        joints_hierarchy = {}
        self.parents = np.zeros(len(self.joints), dtype=int) - 1 # -1 for root
        num_joints = len(self.joints)
        num_frames = len(frames)
        self.local_poses = np.zeros((num_frames, num_joints, 4, 4))
        self.global_poses = np.zeros((num_frames, num_joints, 4, 4))

        for i, joint in enumerate(self.joints):
            #build hierachry
            parent = self.egomc_bvh.joint_parent(joint)
            joints_hierarchy[joint] = [parent.name] if parent is not None else []
            #get offsets
            joints_offsets[joint] = np.array(self.egomc_bvh.joint_offset(joint)) * self.sensor_cfg['distance_scale']
            
            #local pose
            R_local = get_rotation_chain_vectorized(joint_rot_order[joint], local_joint_rot[:, i, :])
            p_local = np.tile(joints_offsets[joint], (num_frames, 1))
            
            self.local_poses[:, i, :3, :3] = R_local
            self.local_poses[:, i, :3, 3] = p_local
            self.local_poses[:, i, 3, 3] = 1.0
            
            #global pose 
            if parent is None:
                self.global_poses[:, i] = self.local_poses[:, i]
            else:
                parent_idx = self.joints.index(parent.name)
                self.global_poses[:, i] = np.einsum('...ij,...jk->...ik', self.global_poses[:, parent_idx], self.local_poses[:, i])
            
            self.parents[i] = self.joints.index(parent.name) if parent is not None else -1

        #slice poses 
        sl = slice(self.start_frame_idx, self.end_frame_idx)
        self.local_poses = self.local_poses[sl]
        self.global_poses = self.global_poses[sl]
        self.root_pos = self.root_pos[sl]
        self.local_joint_rot_quat = self.local_joint_rot_quat[sl]
        

    def get_wolrd_poses(self):
        self.world_poses = self.global_poses.copy()
        
        
        root_pos_aligned = self.root_pos.copy()
        root_pos_aligned -= self.root_pos[0, np.newaxis, :] # align the first 
        # root_pos_aligned += self.global_poses[0, 0, :3, 3][np.newaxis, :]
        self.world_poses[:, :, :3, 3] += root_pos_aligned[:, np.newaxis, :] #add global transl to the root 

    

    def get_skeleton_bones(self) -> np.ndarray: 
        bones = [] 
        for joint in self.joints:
            try: 
                parent = self.egomc_bvh.joint_parent(joint)
                i = self.joints.index(parent.name)
                j = self.joints.index(joint)
                bones.append([i, j])
            except: 
                continue
        return np.array(bones)
            

    def _get_footlock(self):
        timestamp_raw = np.array(self.egomc_raw_data["Timestamp"]) * self.sensor_cfg['time_scale']
        plot_data = {"ts": timestamp_raw}
        
        self.footlock = {}
        self.footlock_soft = {}
        for joint in ["LeftFoot", "RightFoot"]:
            vel_x = np.array(self.egomc_raw_data[f"{joint}_velocity_x"])
            vel_y = np.array(self.egomc_raw_data[f"{joint}_velocity_y"])
            vel_z = np.array(self.egomc_raw_data[f"{joint}_velocity_z"])
            speed = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)

            # Hard lock
            threshold = self.sensor_cfg['footlock']['threshold_hard']  # m/s
            footlock_mask = speed < threshold  # Boolean array: True = locked
            footlock_mask_30 = np.interp(self.timestamp, timestamp_raw, footlock_mask.astype(float)) > 0.5
            #Soft lock
            threshold_soft = self.sensor_cfg['footlock']['threshold_soft']
            footlock_soft = np.exp(-(speed/threshold_soft)**2)  # Boolean array: True = locked
            footlock_soft_30 = np.interp(self.timestamp, timestamp_raw, footlock_soft)

            idx = max(0, self.start_frame_idx)
            sl = slice(idx, self.end_frame_idx)
            self.footlock[f"{joint}"] = footlock_mask_30[sl]
            self.footlock_soft[f"{joint}"] = footlock_soft_30[sl]
                
            if self.pip_cfg.debug:
                plot_data[joint] = {
                    "vel": (vel_x, vel_y, vel_z),
                    "speed": speed,
                    "mask": footlock_mask,
                    "soft": footlock_soft
                }
        
        if self.pip_cfg.debug:
            plot_vel(
                plot_data["ts"],
                plot_data["LeftFoot"]["vel"][0], plot_data["LeftFoot"]["vel"][1], plot_data["LeftFoot"]["vel"][2],
                plot_data["LeftFoot"]["speed"],
                plot_data["LeftFoot"]["mask"],
                plot_data["LeftFoot"]["soft"],
                plot_data["RightFoot"]["vel"][0], plot_data["RightFoot"]["vel"][1], plot_data["RightFoot"]["vel"][2],
                plot_data["RightFoot"]["speed"],
                plot_data["RightFoot"]["mask"],
                plot_data["RightFoot"]["soft"],
            )
        
    
    def time_sync_sensor(self):
        """Determine the starting point based on clap event to synchronize sensor data.""" 
        #get clap time/idx
        egomc_start_time, _, _= get_egomc_start(self.egomc_raw_data, self.sensor_cfg['time_sync'], self.pip_cfg)
        self.start_frame_idx = int(np.searchsorted(self.timestamp, egomc_start_time))
        
        #set last frame idx
        self.end_frame_idx = self.egomc_bvh.nframes - 1 # will be updated later based on available data
        self.duration = (self.end_frame_idx - self.start_frame_idx) / self.sensor_cfg["fps"]
    
    def get_synced_time_idx(self, idx) -> int:
        return self.start_frame_idx + idx
    
    def joint_idx(self, joint_name: str) -> int:
        """Get the index of a joint by name."""
        return self.joints.index(joint_name)
    
    def spatial_align_data(self, transform: np.ndarray):
        """Align the Egomocap data to a given transformation."""
        print("[INFO]: Aligning EgoMocap data")
        self.global_poses = np.einsum('fik,fjkl->fjil', transform, self.global_poses)
    
    def spatial_align_world_pose(self, transform: np.ndarray):
        """Align the Egomocap world poses to a given first pose transformation."""
        print("[INFO]: Aligning EgoMocap world poses")
        self.world_poses = np.einsum('ik,fjkl->fjil', transform, self.world_poses)

    
        
    def get_export_data(self) -> dict:
        """Prepare data for export."""
        payload = {
            "egomc_root_pos": self.root_pos,
            "egomc_local_poses": self.local_poses,
            "egomc_global_poses": self.global_poses,
            "egomc_skeleton": self.get_skeleton_bones(),
            "egomc_timestamp": self.get_synced_timestamp(),
        }
        # flat dict
        for joint, mask in self.footlock.items():
            payload[f"egomc_footlock_{joint}"] = mask   
            payload[f"egomc_footlock_soft_{joint}"] = self.footlock_soft[joint]

        if self.pip_cfg.ablation:
            payload.update({"egomc_world_poses": self.world_poses})
    
        return payload 
    
    def get_joint_pos(self, name:str) -> np.ndarray:
        """Get the global position of a joint by name."""
        idx = self.joints.index(name)
        return self.global_poses[:, idx, :3, 3]