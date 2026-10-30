import torch
import numpy as np
from abc import ABC

from src.loader.helper.helper import (
    dump_config,
    p2p_vggt,
    p2p_classic, 
    get_fixed_glass_translation

)
from src.loader.utils import (
    get_rotation_transform,
)
from src.visualizer.helper import draw_registration_result
from src.loader.utils import transform_points
from src.config.configs import SequenceContext
from src.optimizer.data_optimizer import EgoHTGOptimizer

class EgoHTGGenerator(ABC):
    """
    Class to handle data generation for the EgoHumanTerrainGeneration dataset. This includes loading, processing, temporal and spatial alignment, 
    and exporting of data from various sensors and body models. The generator is designed to be flexible and extensible, 
    allowing for different types of data loaders and body models to be integrated as needed.
    """
    
    def __init__(self, pip_cfg, sensor_cfg):
        self.pip_cfg = pip_cfg
        self.sensor_cfg = sensor_cfg
   
    def run(self, loader: dict, ctx: SequenceContext, seq_cfg: dict, body_models: dict, optimizer: EgoHTGOptimizer):
        """Generate data for a specific sequence."""
        
        self._load_data(loader)
        self._temporal_align(loader, ctx, seq_cfg)
        self._process_data(loader, ctx, seq_cfg, body_models)
        self._spatial_align(loader, ctx, seq_cfg, body_models, optimizer)
        self._export_data(loader, ctx, seq_cfg, body_models)
        
    def _load_data(self, loader_dict: dict):
        """Load data from all initialized loaders."""
        for loader in loader_dict.values():
            if loader is None:
                continue
            loader.load_data()
        
    def _process_data(self, loader_dict: dict, ctx: SequenceContext, seq_cfg: dict, body_models: dict):
        """Process data from all initialized loaders."""
        for name,loader in loader_dict.items():
            if loader is None or name == 'aria_obs':
                continue
            loader.process()
            
        #retarget egomocap to smplx
        for body_model in body_models.values():
            body_model.generate_from_mocap(ctx, loader_dict[body_model.sensor_type])
        
        

    def _temporal_align(self, loader_dict: dict, ctx: SequenceContext, seq_cfg: dict):
        """Temporally align data from different loaders."""
        print("[INFO]: Time Sync dynamic sensor data")
        durations = []
        for name, loader in loader_dict.items():
            if loader is None or name == 'scene':
                continue
            # use provided time sync info if available, otherwise perform time sync and save the info to config
            elif seq_cfg.get(f"{name}_time_sync", None) is not None:
                loader.start_frame_idx = seq_cfg[f"{name}_time_sync"]["start_frame_idx"]
                loader.duration = seq_cfg[f"{name}_time_sync"]["duration"]
                if name == 'aria_obs' or name == 'aria':
                    loader.time_sync_sensor(preloaded=True)
                    loader.start_frame_ns = seq_cfg[f"{name}_time_sync"]["start_frame_ns"]
                    if name=="aria" and seq_cfg[f"{name}_time_sync"].get(f"traj_start_frame_idx", None) is None: #TODO: can be removed after all sequences have been generated with traj_start_frame_idx
                        seq_cfg[f"{name}_time_sync"]['traj_start_frame_idx'] = int(loader.get_traj_start_idx()) 
                print(f"[INFO]: Using provided {name} time sync from config.yaml")

            else:
                loader.time_sync_sensor()
                
                # save time sync info
                seq_cfg[f"{name}_time_sync"] = {
                    "start_frame_idx": loader.start_frame_idx,
                    "duration": loader.duration
                }
                # update cfg with additional time sync info for Aria data
                if name == 'aria_obs':
                    seq_cfg[f"{name}_time_sync"]['start_frame_ns'] = float(loader.start_frame_ns)
                if name == 'aria':
                    seq_cfg[f"{name}_time_sync"]['start_frame_ns'] = float(loader.start_frame_ns)
                    seq_cfg[f"{name}_time_sync"]['traj_start_frame_idx'] = int(loader.get_traj_start_idx())

            
            durations.append(loader.duration)
            
        # align to the shortest duration
        min_duration = min(durations)
        if self.pip_cfg.duration is None or self.pip_cfg.duration > min_duration:
            seq_cfg["duration"] = min_duration
        else: 
            seq_cfg["duration"] = self.pip_cfg.duration
            
        #update end frame for each loader
        for name, loader in loader_dict.items():
            if loader is None or name == 'scene':
                continue
            loader.set_nframes_by_duration(seq_cfg["duration"])
        #dump the updated config with time sync info and duration
        dump_config(seq_cfg, ctx.config_yaml, 'generator_config')


    def _spatial_align(self, loader_dict: dict, ctx: SequenceContext, seq_cfg: dict, body_models: dict, optimizer: EgoHTGOptimizer):
        """Spatially align data from different loaders."""
        print("[INFO]: Spatially Align sensor data")
        T_origin_sceneW = loader_dict["scene"].get_scene2origin_transform() if loader_dict["scene"] else np.eye(4)
        self.T_origin_ariaW = self.get_aria2origin_transform(T_origin_sceneW, loader_dict, ctx, seq_cfg, self.pip_cfg.multi_aria)
        
        # Aligne Aria data
        if self.pip_cfg.multi_aria:
            loader_dict["aria_obs"].get_device2origin_transform_aria(self.T_origin_ariaW) 
            loader_dict["aria_obs"].spatial_align_data(self.T_origin_ariaW)

        loader_dict["aria"].get_device2origin_transform_aria(self.T_origin_ariaW)
        loader_dict["aria"].spatial_align_data(self.T_origin_ariaW)

        #Align Scene data
        if loader_dict["scene"]:
            loader_dict["scene"].spatial_align_data(T_origin_sceneW)
        
        # Align Egomocap data
        if ctx.egomc_exists:
            nframes = loader_dict["egomc"].nframes
            ego_body_model = body_models["egomc"]
            T_origin_device = loader_dict["aria"].get_device2origin_transform(self.T_origin_ariaW, nframes, self.sensor_cfg["egomc"]["fps"])
            T_origin_sroot = self.get_sroot2origin_transform(T_origin_device, seq_cfg["head2device"], nframes, loader_dict["aria"].get_camera2device_transform())
            T_origin_smplxHead = self.get_head2origin_transform(T_origin_sroot, ego_body_model.get_joint_pos("head"), nframes)
            T_origin_egomcHead = self.get_head2origin_transform(T_origin_sroot, loader_dict["egomc"].global_poses[:, loader_dict["egomc"].joint_idx("Head"), :3, 3], nframes) 
            loader_dict["egomc"].spatial_align_data(T_origin_egomcHead)
            ego_body_model.spatial_align_ego(transform=torch.from_numpy(T_origin_smplxHead).float().to(ego_body_model.device), init_aria_height=T_origin_device[0,2,3], scene=ctx.scene_exists)
            ego_body_model.set_config(ctx)

            if self.pip_cfg.ablation:
                print("[INFO]: Align EgoMocap standalone")
                loader_dict["egomc"].spatial_align_world_pose(T_origin_egomcHead[0])
                ego_body_model.set_standalone_root(torch.from_numpy(loader_dict["egomc"].world_poses[:,0, :3, 3]).float().to(ego_body_model.device))

            # Align Exomocap data
            if ctx.exomc_exists:
                nframes = loader_dict["exomc"].nframes
                exo_body_model = body_models["exomc"]
                T_origin_device = loader_dict["aria"].get_device2origin_transform(self.T_origin_ariaW, nframes, self.sensor_cfg["exomc"]["fps"]) # only need the first frame
                T_origin_exomcW = self.get_exo2origin_transform(T_origin_device[0], loader_dict, ctx, seq_cfg, self.T_origin_ariaW, self.pip_cfg.multi_aria)
                loader_dict["exomc"].spatial_align_data(T_origin_exomcW, loader_dict["egomc"].global_poses[0, loader_dict["egomc"].joint_idx("Head"), :3, 3])
                exo_body_model.spatial_align_exo(torch.from_numpy(T_origin_exomcW).float().to(exo_body_model.device), torch.from_numpy(loader_dict["exomc"].get_joint_pos('Head')).float().to(exo_body_model.device), cfg=self.sensor_cfg["exomc"])
                optimizer.run_ik(ctx, seq_cfg["ik_config"], exo_body_model, loader_dict["exomc"].global_pose)
            
    
    def _export_data(self, loader_dict: dict, ctx: SequenceContext, seq_cfg: dict, body_models: dict):
        """Export data from all initialized loaders."""
        payload = {}
        for name, loader in loader_dict.items():
            if loader is None:
                continue
            payload.update(loader.get_export_data())
        #update smplx data
        for body_model in body_models.values():
            payload.update(body_model.get_export_data())
            # body_model.cleanup()
        
        print("[INFO]: Export generated data")
        ctx.gen_seq.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(ctx.gen_seq, **payload)
        

    def get_aria2origin_transform(self, T_origin_sceneW: np.ndarray, loader_dict: dict, ctx: SequenceContext, seq_cfg: dict, aria_multi_flag: bool) -> np.ndarray:
        """Get the transformation Aria to Scene coordinate system."""
        name = "T_sceneW_ariamultiW" if aria_multi_flag else "T_sceneW_ariaW"

        if name in seq_cfg:
            T_sceneW_ariaW = np.array(seq_cfg[name])
            print(f"[INFO]: Using provided {name} from config.yaml")
        elif ctx.scene_exists is True:
            
            fp_timestamp_idx = loader_dict["aria"].save_undistorted_img(ctx.aria_subject.aria_imgs_dir)
            T_cam_fp = loader_dict["aria"].get_cam_fp(fp_timestamp_idx)
            T_sceneW_ariaW = p2p_vggt(loader_dict["aria"].aria_pcd, loader_dict["scene"].scene_pcd_box, T_cam_fp, ctx.scene_dir, ctx.aria_dir, self.pip_cfg.root, seq_cfg)
            
            if self.pip_cfg.debug:
                print("[DEBUG]: Visualze P2P alignment result")
                draw_registration_result(loader_dict["aria"].aria_pcd, loader_dict["scene"].scene_pcd_box, T_sceneW_ariaW)
            
            # save transform
            seq_cfg[name] = T_sceneW_ariaW.astype(float).tolist()
            dump_config(seq_cfg, ctx.config_yaml, 'generator_config')
            
        else:
            print("[INFO]: Using Aria World Coordinate as origin, since no scene data is available.")
            T_sceneW_ariaW = np.eye(4)
        return T_origin_sceneW @ T_sceneW_ariaW
    
        
    def get_sroot2origin_transform(self, T_origin_device: np.ndarray, cfg: dict, nframes: int, T_device_camera: np.ndarray) -> np.ndarray:
        """
        Get the transformation EgomocapHead to origin coordinate system serie.
        The head joint of the Egomocap skeleton is aligned to the device (Aria) coordinate system via fixed translation from device to head joint
        
        The head rotation is decoupled from the device rotation, defined by the egocentric mocap suit data. 
        """
        ## extract yaw rotation from device trajectory and apply it to the head orientation, to preserve the head facing direction in the world coordinate system.
        T_origin_camera_init = T_origin_device[0] @ T_device_camera
        yaw = np.rad2deg(np.arctan2(T_origin_camera_init[1, 0], T_origin_camera_init[0, 0])) # extract yaw from device rotation
        rot_euler = cfg['rot_euler_global']
        rot_euler[2] += yaw
        ### rotation
        R_origin_device = np.repeat(T_origin_device[0, :3, :3][np.newaxis, :, :], nframes, axis=0) # fixed rotation
        R_origin_device = np.repeat(get_rotation_transform(rot_euler, order='xyz')[np.newaxis, :3, :3], nframes, axis=0)
        
        ### translation
        t_origin_device = T_origin_device[:, :3, 3]  
        t_device2deviceH_local = np.repeat(get_fixed_glass_translation(cfg["dist2glass"], cfg["height_trans"], cfg["joint"])[np.newaxis, :], nframes, axis=0) # translation from device to head joint
        t_device2deviceH = T_origin_device[:, :3, :3] @ t_device2deviceH_local  # rotate to world coordinate
        t_origin_deviceH = t_origin_device + t_device2deviceH.squeeze(-1)
        
        
        T_origin_deviceH = np.repeat(np.eye(4)[np.newaxis, :, :], nframes, axis=0)
        T_origin_deviceH[:, :3, :3] = R_origin_device  # rotation from device to world
        T_origin_deviceH[:, :3, 3] = t_origin_deviceH  # translation from device to world
        
        
        return T_origin_deviceH #@ T_deviceH_sroot
    
    def get_head2origin_transform(self, T_origin_sroot: np.ndarray, head_joint: np.ndarray, nframes: int) -> np.ndarray:
        
        T_sroot_head = np.repeat(np.eye(4)[np.newaxis, :, :], nframes, axis=0)
        T_sroot_head[:, :3, 3] = -head_joint[:nframes, :]  # translation from skeleton root to head joint
        
        return T_origin_sroot @ T_sroot_head
    
    def get_exo2origin_transform(self, T_origin_device: np.ndarray, loader_dict: dict, ctx: SequenceContext, seq_cfg: dict, T_origin_ariaW, aria_multi_flag: bool) -> np.ndarray:
        """
        Get the transformation Exomocap to origin coordinate system.
        The Exomocap data is aligned to the Aria trajectory via point-to-point registration between the 'aria-marker' trajectory from Exomocap and the device trajectory from Aria, 
        initalised by a transformation with:
         - translation from Aria first frame to the Exomocap first frame and 
         - rotation from Aria first frame heading to exomocap first frame 'head-marker' heading.
        """
        name = "T_origin_exomultiW" if aria_multi_flag else "T_origin_exoW"
        
        if name in seq_cfg:
            print(f"[INFO]: Using provided {name} from config.yaml")
            T_origin_exoW = np.array(seq_cfg[name])
        else:
            loader_exo = loader_dict["exomc"]
            loader_aria = loader_dict["aria"]

            #get exo trajectory
            exomc_traj = loader_exo.align_traj
            # aria_traj = loader_aria.get_freq_aligned_trajectory(exomc_traj.shape[0], loader_exo.sensor_cfg['fps'])
            aria_traj = transform_points(loader_aria.get_time_aligned_trajectory(loader_exo.align_traj_timestamp), T_origin_ariaW)

            #get heading vector
            head_points = loader_exo.headFP
            heading_vec = head_points[1, 0:2] - head_points[0, 0:2]

            #get exomc first pose
            T_exomcFP_exomcW = np.eye(4)
            T_exomcFP_exomcW[:3, 3] = -loader_exo.align_traj[0]
            #get aria first pose
            T_device_camera = loader_aria.get_camera2device_transform()
            T_origin_ariaFP = T_origin_device @ T_device_camera
            
            # align exomocap to aria trajectory
            T_origin_exoW_off = p2p_classic(exomc_traj, aria_traj, T_origin_ariaFP, T_exomcFP_exomcW, heading_vec)
            
            if self.pip_cfg.debug:
                print("[DEBUG]: Visualze P2P alignment result")
                draw_registration_result(exomc_traj, aria_traj, T_origin_exoW_off)

            # offset exomocap trajectory according to measured hardware dispolacment in real
            offset = T_origin_device[:3, :3] @ np.array([seq_cfg['exomc2device_offset']]).T.squeeze(-1)
            T_origin_exoW = T_origin_exoW_off.copy()
            T_origin_exoW[:3, 3] = T_origin_exoW[:3, 3] + offset
        
            # save transform
            seq_cfg[name] = T_origin_exoW.astype(float).tolist()
            dump_config(seq_cfg, ctx.config_yaml, 'generator_config')
   
        return T_origin_exoW
