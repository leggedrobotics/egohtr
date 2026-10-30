import numpy as np
import gc
import torch
from abc import ABC
from typing import Union
from types import SimpleNamespace

from smplx.body_models import SMPLXLayer, SMPLXOutput, SMPLX
from src.body_model.smplx_helper import SMPLX_JOINT_NAMES
from src.retargeter.motion_retargeter import EgoHTGRetargeter

from src.loader.helper.helper import dump_config

class SMPLXHandler(ABC):
    
    smplx_poses: dict
    
    JOINT_GROUPS = {
        "hips": [0, 1], "knees": [3, 4], "ankles": [6, 7], "feet": [9, 10],
        "spine": [2, 5, 8, 11, 14], "collar": [12, 13],
        "arms": list(range(15, 21))
    }
    
    def __init__(self, pip_cfg, retargeter: EgoHTGRetargeter = None, sensor_type: str = "egomc"):
        self.pip_cfg = pip_cfg
        self.retargeter = retargeter
        self.sensor_type = sensor_type
    
    def initialize(self, seq_cfg: dict):
        self.seq_cfg = seq_cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.smplx_layer = SMPLXLayer(
            model_path=self.pip_cfg.smplx_models, 
            model_type='smplx',
            gender=self.seq_cfg["gender"],
            use_rodrigues=False,
            use_pca=False,
            ext="npz",
        )
        self.smplx_layer.to(self.device)
    
    def spatial_align_ego(self, transform: torch.Tensor, init_aria_height: float, root: str = "pelvis", scene: bool = False):
        """
        Applies the external transforms_tensor (N, 4, 4) to the SMPL-X pose (global_orient, trans). Shifting pivot to the root joint.
        """
        global_orient = self.smplx_poses["global_orient"]
        trans = self.smplx_poses["transl"]
        N = global_orient.shape[0]
        root_pos = self.get_joint_pos(root, torch=True)
        
        trans += root_pos
        
        T_smplx = torch.eye(4, dtype=global_orient.dtype, device=global_orient.device).unsqueeze(0).repeat(N, 1, 1)
        T_smplx[:, :3, :3] = global_orient
        T_smplx[:, :3, 3] = trans

        T_transformed = torch.matmul(transform, T_smplx)
        trans_transformed = T_transformed[:, :3, 3]
        global_orient_transformed = T_transformed[:, :3, :3]
        
        trans_transformed -= root_pos
        
        self.smplx_poses["global_orient"] = global_orient_transformed
        self.smplx_poses["transl"] = trans_transformed
        
        # self.set_smplx_height(init_aria_height, root) if scene else None
        
    def spatial_align_exo(self, transform: torch.Tensor, init_aria_height, cfg: dict):
        """
        Applies the external transforms_tensor (N, 4, 4) to the SMPL-X pose (global_orient, trans). Shifting pivot to the root joint.
        """
        global_orient = self.smplx_poses["global_orient"]
        trans = self.smplx_poses["transl"]
        N = global_orient.shape[0]
        
        T_smplx = torch.eye(4, dtype=global_orient.dtype, device=global_orient.device).unsqueeze(0).repeat(N, 1, 1)
        T_smplx[:, :3, :3] = global_orient
        T_smplx[:, :3, 3] = trans

        T_transformed = torch.matmul(transform, T_smplx)
        trans_transformed = T_transformed[:, :3, 3]
        global_orient_transformed = T_transformed[:, :3, :3]
        
        self.smplx_poses["global_orient"] = global_orient_transformed
        self.smplx_poses["transl"] = trans_transformed
        
        self.apply_transl_offset(cfg["offset"], global_orient_transformed, init_aria_height)
        
    def set_smplx_height(self, init_aria_height: float, root: str):
        """Set SMPLX body height based on floor height."""
        root_before = self.get_joint_pos(root, torch=True)
        #init
        output = self.forward(self.smplx_poses)
        left_heel_height = output.joints[0, self.joint_idx("left_heel"), 2].detach().cpu().numpy()
        left_toe_height = output.joints[0, self.joint_idx("left_small_toe"), 2].detach().cpu().numpy()
        left_eye_height = output.joints[0, self.joint_idx("left_eye_smplhf"), 2].detach().cpu().numpy()
        height_diff = left_eye_height - (0.7 * left_toe_height + 0.3 * left_heel_height) 
        diff =  height_diff - init_aria_height
        sign = np.sign(diff)
        counter = 0
        
        while not np.isclose(height_diff, init_aria_height, atol=1e-3):
            self.smplx_poses["betas"][:,0] = self.smplx_poses["betas"][:,0] + sign * 0.01
            output = self.forward(self.smplx_poses)
            left_heel_height = output.joints[0, self.joint_idx("left_heel"), 2].detach().cpu().numpy()
            left_toe_height = output.joints[0, self.joint_idx("left_small_toe"), 2].detach().cpu().numpy()
            left_eye_height = output.joints[0, self.joint_idx("left_eye_smplhf"), 2].detach().cpu().numpy()
            height_diff = left_eye_height - (0.7 * left_toe_height + 0.3 * left_heel_height) 
            # change objective sign if overshoot
            if np.abs(diff) < np.abs(height_diff - init_aria_height) and counter == 2:
                sign *= -1
                
            # safety break
            if counter > 100:
                print("[WARNING] Could not adjust SMPLX height to match floor height.")
                break
            elif counter % 10 == 0:
                print(f"[INFO] Adjusting SMPLX height. Current Height: {height_diff:.4f}, Target Height: {init_aria_height}, Beta[0]: {self.smplx_poses['betas'][0,0].item():.4f}")
                counter += 1
            else: 
                counter += 1
            
            torch.cuda.empty_cache()
            gc.collect()
        
        print("[INFO]Adjusting betas to match floor height. Init Height:", diff + init_aria_height, "Final Height:", height_diff)
        #revert generated root offset
        height_offset = left_eye_height - init_aria_height
        self.smplx_poses["transl"][:,2] -= height_offset
        root_after = self.get_joint_pos(root, torch=True)
        root_offset = root_after - root_before
        self.smplx_poses["transl"][:,0:2] -= root_offset[:, 0:2]
        
    def apply_transl_offset(self, cfg: dict, global_orient_transformed: np.ndarray, init_aria_height, joint_name: str = "pelvis"):
        """Return SMPLX translation with offset applied due to the introduced pivot change."""
        offset = torch.tensor(cfg["translation"], device=self.device)
        # offset[1] = init_aria_height - smplx_output.joints[0, self.joint_idx("left_eye_smplhf"), 2] 
        
        #zero trans offset
        zero_transl = torch.zeros_like(self.smplx_poses["transl"]).to(self.device)
        smplx_output = self.forward({**self.smplx_poses, "transl": zero_transl}, return_verts=False)
        zero_transl_offset = smplx_output.joints[:, self.joint_idx(joint_name), :] - global_orient_transformed @ offset
        self.smplx_poses["transl"] -= zero_transl_offset
        
        #align with ego smplx
        smplx_output = self.forward(self.smplx_poses, return_verts=False)
        self.transl_offset = smplx_output.joints[:, self.joint_idx("head"), :] - init_aria_height
        self.smplx_poses["transl"] -= self.transl_offset
        
        #clean up
        del smplx_output
        torch.cuda.empty_cache()
        gc.collect()
    
        
    def set_config(self, ctx):
        """Set SMPLX configuration parameters."""
        self.seq_cfg["betas"] = self.smplx_poses["betas"][0].detach().cpu().numpy().tolist()
        dump_config(self.seq_cfg, ctx.config_yaml, 'smplx_config')


    def generate_from_mocap(self, ctx, loader):
        """Generate SMPLX parameters from mocap data."""
        if self.sensor_type == "egomc":
            self.generate_from_egomc(ctx, loader)
        elif self.sensor_type == "exomc":
            self.generate_from_exomc(ctx, loader)
        else:
            raise ValueError(f"Unsupported sensor type: {self.sensor_type}")

    def generate_from_egomc(self, ctx, loader):
        """Generate SMPLX parameters from egomocap data."""
        
        #retarget egomocap to smplx
        self.smplx_poses = self.retargeter.egomc_to_smplx(
            ctx.egomc_bvh, 
            loader.start_frame_idx, 
            loader.end_frame_idx, 
            self.seq_cfg, 
            loader.sensor_cfg,
            loader.seq_cfg,
        )
        
    def generate_from_exomc(self, ctx, loader):
        """Generate SMPLX parameters from exomocap data."""
        
        #retarget exomocap to smplx
        self.smplx_poses = self.retargeter.exomc_to_smplx(
            loader.global_pos,
            loader.global_quat,
            loader.joint_names, 
            loader.parents, 
            self.seq_cfg, 
            loader.sensor_cfg,
        )
    
    def forward_grad(self, smplx_params: dict) -> SMPLXOutput:
        """Forward pass through the SMPLX model.
        
        :param smplx_params: Dictionary containing SMPLX parameters in rotation matrix format:
            - 'global_orient': Global orientation (B, 3, 3)
            - 'trans': Translation (B, 3)
            - 'body_pose': Body pose (B, 21, 3, 3)
            - 'left_hand_pose': Left hand pose (B, 15, 3, 3)
            - 'right_hand_pose': Right hand pose (B, 15, 3, 3)
            - 'jaw_pose': Jaw pose (B, 3)
            - 'leye_pose': Left eye pose (B, 3)
            - 'reye_pose': Right eye pose (B, 3)
            
        :return: Dictionary containing output vertices and joints.
        """

        smplx_output = self.smplx_layer(
            global_orient=smplx_params.get('global_orient'),
            body_pose=smplx_params.get('body_pose'),
            left_hand_pose=smplx_params.get('left_hand_pose'),
            right_hand_pose=smplx_params.get('right_hand_pose'),
            jaw_pose=smplx_params.get('jaw_pose'),
            leye_pose=smplx_params.get('leye_pose'),
            reye_pose=smplx_params.get('reye_pose'),
            transl=smplx_params.get('transl'),
            return_full_pose=True,
        )
        
        return smplx_output
      

    def forward(self, smplx_params: dict, max_batch: int = 1000, return_verts: bool = True) -> SMPLXOutput:
        """
        Forward pass through the SMPLX model with automatic batch chunking to prevent OOM.
        """
        total_frames = smplx_params.get('transl').shape[0]
        
        # We store chunks on CPU (System RAM)
        combined_results = {
            'vertices': [], 'joints': [], 'full_pose': [], 
            'body_pose': [], 'global_orient': [], 'transl': [], 
            'left_hand_pose': [], 'right_hand_pose': [], 
            'jaw_pose': [], 'leye_pose': [], 'reye_pose': [],
            'betas': []
        }

        for i in range(0, total_frames, max_batch):
            end_idx = min(i + max_batch, total_frames)
            
            chunk_params = {k: v[i:end_idx] for k, v in smplx_params.items() if v is not None}

            # Disable gradient calculation to save massive amounts of VRAM
            with torch.no_grad():
                chunk_output = self.smplx_layer(
                    global_orient=chunk_params.get('global_orient'),
                    body_pose=chunk_params.get('body_pose'),
                    left_hand_pose=chunk_params.get('left_hand_pose'),
                    right_hand_pose=chunk_params.get('right_hand_pose'),
                    jaw_pose=chunk_params.get('jaw_pose'),
                    leye_pose=chunk_params.get('leye_pose'),
                    reye_pose=chunk_params.get('reye_pose'),
                    transl=chunk_params.get('transl'),
                    betas=chunk_params.get('betas'),
                    return_full_pose=True,
                    return_verts=return_verts, # Set to False in loops to save 90% memory
                )
                
                # IMMEDIATE OFFLOAD TO CPU
                for key in combined_results.keys():
                    val = getattr(chunk_output, key, None)
                    if val is not None:
                        # .detach().cpu() is the life-saver here
                        combined_results[key].append(val.detach().cpu())
                
                # Cleanup GPU memory immediately after chunk
                del chunk_output
                torch.cuda.empty_cache()

        # Concatenate on CPU
        for key in combined_results.keys():
            if combined_results[key]:
                combined_results[key] = torch.cat(combined_results[key], dim=0).to(self.device)
            else:
                combined_results[key] = None

        # Return a namespace so existing code doesn't break
        from types import SimpleNamespace
        return SimpleNamespace(**combined_results)
    

    def get_joint_pos(self, name: str, torch: bool = False) -> Union[np.ndarray, torch.Tensor]:
        """Return joint position."""
        smplx_output = self.forward(self.smplx_poses)
        joint_idx = self.joint_idx(name)
        joint_pos = smplx_output.joints[:, joint_idx, :]
        return joint_pos.detach().cpu().numpy() if not torch else joint_pos.detach()
    
    def joint_idx(self, name):
        """Return joint idx."""
        return SMPLX_JOINT_NAMES.index(name)
    
    @property
    def joints(self) -> torch.Tensor:
        """Return SMPLX joints."""
        smplx_output = self.forward(self.smplx_poses)
        return smplx_output.joints

    def get_export_data(self) -> dict:
        """Export SMPLX data as a dictionary."""
        smplx_output = self.forward(self.smplx_poses, return_verts=False)

        if self.sensor_type == "egomc":
            export_data = {
                f"smplx_{self.sensor_type}_body_pose": smplx_output.body_pose.detach().cpu().numpy(),
                # f"smplx_{self.sensor_type}_left_hand_pose": smplx_output.left_hand_pose.detach().cpu().numpy(),
                # f"smplx_{self.sensor_type}_right_hand_pose": smplx_output.right_hand_pose.detach().cpu().numpy(),
                # f"smplx_{self.sensor_type}_jaw_pose": smplx_output.jaw_pose.detach().cpu().numpy(),
                f"smplx_{self.sensor_type}_global_orient": smplx_output.global_orient.detach().cpu().numpy(),
                f"smplx_{self.sensor_type}_transl": smplx_output.transl.detach().cpu().numpy(),
                f"smplx_{self.sensor_type}_joints": smplx_output.joints.detach().cpu().numpy(),
                # f"smplx_{self.sensor_type}_vertices": smplx_output.vertices.detach().cpu().numpy(),
                f"smplx_{self.sensor_type}_betas": smplx_output.betas.detach().cpu().numpy(),
                f"smplx_{self.sensor_type}_faces": self.smplx_layer.faces,
                # f"smplx_{self.sensor_type}_leye_pose": smplx_output.full_pose[:,23,:,:].detach().cpu().numpy(),
                # f"smplx_{self.sensor_type}_reye_pose": smplx_output.full_pose[:,24,:,:].detach().cpu().numpy(),
                f"smplx_skeleton": self.get_skeleton_bones(),
            }
        elif self.sensor_type == "exomc": #every 10 frames to save space
            export_data = {
                f"smplx_{self.sensor_type}_body_pose": smplx_output.body_pose[::10].detach().cpu().numpy(),
                # f"smplx_{self.sensor_type}_left_hand_pose": smplx_output.left_hand_pose[::10].detach().cpu().numpy(),
                # f"smplx_{self.sensor_type}_right_hand_pose": smplx_output.right_hand_pose[::10].detach().cpu().numpy(),
                # f"smplx_{self.sensor_type}_jaw_pose": smplx_output.jaw_pose[::10].detach().cpu().numpy(),
                f"smplx_{self.sensor_type}_global_orient": smplx_output.global_orient[::10].detach().cpu().numpy(),
                f"smplx_{self.sensor_type}_transl": smplx_output.transl[::10].detach().cpu().numpy(),
                f"smplx_{self.sensor_type}_joints": smplx_output.joints[::10].detach().cpu().numpy(),
                # f"smplx_{self.sensor_type}_vertices": smplx_output.vertices[::10].detach().cpu().numpy(),
                f"smplx_{self.sensor_type}_betas": smplx_output.betas[::10].detach().cpu().numpy(),  
                f"smplx_{self.sensor_type}_faces": self.smplx_layer.faces,  
                # f"smplx_{self.sensor_type}_leye_pose": smplx_output.full_pose[::10,23,:,:].detach().cpu().numpy(),
                # f"smplx_{self.sensor_type}_reye_pose": smplx_output.full_pose[::10,24,:,:].detach().cpu().numpy(),
            }
        
        if self.pip_cfg.ablation and self.sensor_type == "egomc":
            export_data.update(self.get_egomc_standalone_data())
        
        return export_data
            
    
    def get_egomc_standalone_data(self) -> dict:
        """Export SMPLX data for standalone egomocap."""
        
        smplx_egomc_standalone_poses = self.smplx_poses.copy()
        smplx_egomc_standalone_poses["transl"] = self.standalone_root_pos
        smplx_output = self.forward(smplx_egomc_standalone_poses)
        export_data = {
            f"smplx_{self.sensor_type}_standalone_joints": smplx_output.joints.detach().cpu().numpy(),
            f"smplx_{self.sensor_type}_standalone_vertices": smplx_output.vertices.detach().cpu().numpy(),
        }

        return export_data
    
    def get_smplx_vertices(self, smplx_poses: dict) -> np.ndarray:
        """Return SMPLX vertices."""
        # convert smplx poses dict with np.arrays to torch tensors on the correct device
        smplx_poses = {k: torch.from_numpy(v).float().to(self.device) for k, v in smplx_poses.items()}
        smplx_output = self.forward(smplx_poses)
        return smplx_output.vertices.detach().cpu().numpy()
    
    def load_data(self, data: dict):
        """Import SMPLX data from a dictionary."""
        self.smplx_poses = {
            "body_pose": torch.from_numpy(data[f"smplx_{self.sensor_type}_body_pose"]).float().to(self.device),
            # "left_hand_pose": torch.from_numpy(data[f"smplx_{self.sensor_type}_left_hand_pose"]).float().to(self.device),
            # "right_hand_pose": torch.from_numpy(data[f"smplx_{self.sensor_type}_right_hand_pose"]).float().to(self.device),
            # "jaw_pose": torch.from_numpy(data[f"smplx_{self.sensor_type}_jaw_pose"]).float().to(self.device),
            "global_orient": torch.from_numpy(data[f"smplx_{self.sensor_type}_global_orient"]).float().to(self.device),
            "transl": torch.from_numpy(data[f"smplx_{self.sensor_type}_transl"]).float().to(self.device),
            "betas": torch.from_numpy(data[f"smplx_{self.sensor_type}_betas"]).float().to(self.device),
            # "leye_pose": torch.from_numpy(data[f"smplx_{self.sensor_type}_leye_pose"]).float().to(self.device),
            # "reye_pose": torch.from_numpy(data[f"smplx_{self.sensor_type}_reye_pose"]).float().to(self.device),
        }
        
        
    def update_poses(self, param_dict: dict, start: int, end: int):
        """Updates the internal poses with optimized values for a specific window."""
        with torch.no_grad():
            for key, val in param_dict.items():
                if val is not None:
                    self.smplx_poses[key][start:end] = val.detach()
                    
    def get_poses_window(self, start: int, end: int) -> dict:
        """Returns a dictionary of batched SMPLX poses for a specific window."""
        batched_poses = {}
        for key, val in self.smplx_poses.items():
            batched_poses[key] = val[start:end]
        return batched_poses
    

    def get_skeleton_bones(self) -> np.ndarray:
        """
        Return SMPL-X skeleton bones as (N_bones, 2) array of (child_idx, parent_idx),
        using the kinematic tree from the SMPL-X model.
        """
        parents = self.smplx_layer.parents.detach().cpu().numpy()  # shape (n_joints,)
        bones = [[idx, parent] for idx, parent in enumerate(parents) if parent != -1]
        return np.array(bones, dtype=np.int32)
    
    def set_standalone_root(self, transl: torch.Tensor):
        """Set SMPLX world translations for egomc standalone."""
        #align with the first frame
        aligned_transl = transl - transl[0, np.newaxis, :] + self.smplx_poses["transl"][0, np.newaxis, :]
        self.standalone_root_pos = aligned_transl
         
    def cleanup(self):
        """Free up GPU resources by moving model and data to CPU."""
        print("[INFO]: Cleaning up SMPLX GPU resources.")
        
        if hasattr(self, 'smplx_layer'):
            self.smplx_layer = self.smplx_layer.cpu()
        
        if hasattr(self, 'smplx_poses'):
            for k, v in self.smplx_poses.items():
                if isinstance(v, torch.Tensor):
                    self.smplx_poses[k] = v.cpu()

        torch.cuda.empty_cache()
        gc.collect()