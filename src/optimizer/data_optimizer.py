import torch
import time
import numpy as np
import math
import wandb 
from abc import ABC
from tqdm import tqdm
from datetime import datetime

from src.optimizer.losses import (
    get_pose_loss,
    get_head_pos_loss,
    get_contact_loss,
    get_smooth_loss,
    get_eye_loss,
)
from src.body_model.smplx_helper import (
    CONTACT_CANDIDATES,
)

from src.config.configs import PipelineConfig, SequenceContext
from src.body_model.smplx import SMPLXHandler


    
class EgoHTGOptimizer(ABC):
    """
    EgoHTGOptimizer is responsible for optimizing the SMPLX body model parameters to better fit the observed data and satisfy physical constraints. 
    It uses PyTorch for optimization and can log the process with Weights & Biases (wandb) if enabled in the configuration.
    """
    
    def __init__(self, pip_cfg: PipelineConfig):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pip_cfg = pip_cfg
        self.opt_payload = {}
    
        print("[INFO]: CUDA available:", torch.cuda.is_available())
        print("[INFO]: Number of GPUs:", torch.cuda.device_count())
        
            
    def run(self, ctx: SequenceContext, opt_cfg: dict, body_model: SMPLXHandler):
        """Optimize data for a specific sequence."""
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
        
        if self.pip_cfg.wandb:
            wandb.init(project="human_pose_opt",
                name=f"{ctx.name}_{timestamp}",
                config=opt_cfg,             
                )
        self._load_data(ctx, body_model)
        self._optimize(opt_cfg, body_model)
        self._export_data(ctx, body_model)
        
        
    def run_ik(self, ctx: SequenceContext, opt_cfg: dict, body_model: SMPLXHandler, global_poses: np.ndarray):
        """Optimize data for a specific sequence."""
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
        
        if self.pip_cfg.wandb:
            wandb.init(project="human_pose_opt",
                name=f"{ctx.name}_IK_{timestamp}",
                config=opt_cfg,             
                )
        self.ik(opt_cfg, body_model, global_poses)
            
    def _load_data(self, ctx: SequenceContext, body_model: SMPLXHandler):
        """Load data from all initialized loaders.""" 
        with np.load(ctx.gen_seq, allow_pickle=True) as loader:
            self.full_sequence_data = dict(loader)
        body_model.load_data(self.full_sequence_data)
        
    
    def _export_data(self, ctx: SequenceContext, body_model: SMPLXHandler):
        """Export optimized data to npz file."""
        print("[INFO]: Exporting optimized data")
        opt_data_sequence = {}
        opt_data_sequence.update(body_model.get_export_data())
        opt_data_sequence.update(self.opt_payload)
    
        ctx.opt_seq.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(ctx.opt_seq, **opt_data_sequence)
        
    def _optimize(self, opt_cfg: dict, body_model: SMPLXHandler):
        """Optimize data from all initialized loaders."""
        if self.device.type == "cpu":
            print("[WARNING]: CUDA not available, optimization will be omitted.")
            return
        
        #init
        nframes = self.full_sequence_data["smplx_egomc_global_orient"].shape[0]
        self.opt_payload = {
            "contact_dist": np.zeros((nframes , 2)),
            "nopen": np.zeros((nframes, 2)),
            "contact_pt": np.zeros((nframes, 10, 2))
        }
        footlock_soft = {
            "LeftFoot": self.full_sequence_data["egomc_footlock_soft_LeftFoot"],
            "RightFoot": self.full_sequence_data["egomc_footlock_soft_RightFoot"]
        }
        
        
        total_steps = math.ceil(nframes / opt_cfg["batch_size"])
        for start in tqdm(range(0, nframes, opt_cfg["batch_size"]), total=total_steps, desc="Batch Optimization"):
            end = min(start+opt_cfg["batch_size"], nframes)
            batch_size = end - start

            #init scene
            scene_mesh_torch, scene_vertices, scene_normals = self._prepare_scene_mesh(opt_cfg, batch_size)
            #init poses batch
            poses_batch = body_model.get_poses_window(start, end)
            poses_batch = self.set_opt_variables(opt_cfg["opt_vars"], poses_batch, body_model)
            optimizer = torch.optim.Adam([
                poses_batch["body_pose"], 
                poses_batch["global_orient"], 
                poses_batch["transl"], 
                ], opt_cfg["learning_rate"], betas=(0.9, 0.999))
             
            for it in range(0, opt_cfg["max_iterations"]):
                optimizer.zero_grad()
                str_time = time.time()
                
                smplx_out = body_model.forward_grad(poses_batch)
     
                #### LOSSES ####
                
                total_loss = 0.0 
                print_str = ''
                loss_logs = {}
                active_weights = {k: v for k, v in opt_cfg["weights"].items() if v > 0}
                
                if "contact" in active_weights or "penetration" in active_weights:
                    contact_loss, penetration_loss, distance, nopen_mask, contact_vertex, contact_dist = get_contact_loss(
                        scene_vertices, scene_normals, scene_mesh_torch, smplx_out.vertices, 
                        footlock_soft, CONTACT_CANDIDATES, [start,end], 
                        batch_size, self.device
                        )
                    if "contact" in active_weights:
                        print_str = print_str + 'Contact loss: {} \n'.format(contact_loss.item())
                        loss_logs["contact_loss"] = contact_loss.item()
                        total_loss += float(active_weights["contact"]) * contact_loss
                    if "penetration" in active_weights:
                        print_str = print_str + 'Penetration loss: {} \n'.format(penetration_loss.item())
                        loss_logs["penetration_loss"] = penetration_loss.item()
                        total_loss += float(active_weights["penetration"]) * penetration_loss

                if "head_pos" in active_weights:
                    head_pos_loss = get_head_pos_loss(
                        smplx_out.joints[:,body_model.joint_idx("head"),:], 
                        self.full_sequence_data["smplx_joints"][start:end, body_model.joint_idx("head"),:], 
                        self.device
                    )
                    print_str = print_str + 'Head position loss: {} \n'.format(head_pos_loss.item())
                    loss_logs["head_pos_loss"] = head_pos_loss.item()
                    total_loss += float(active_weights["head_pos"]) * head_pos_loss
                
                if "smooth" in active_weights:
                    smooth_loss = get_smooth_loss(smplx_out.joints, body_model.joints, start, batch_size, self.device)
                    print_str = print_str + 'Smoothing loss: {} \n'.format(smooth_loss.item())
                    loss_logs["smooth_loss"] = smooth_loss.item()
                    total_loss += float(active_weights["smooth"]) * smooth_loss
                    
                if "eye_pos" in active_weights:
                    eye_loss = get_eye_loss(
                        smplx_out.joints[:, [body_model.joint_idx("left_eye_smplhf"), body_model.joint_idx("right_eye_smplhf")], :], 
                        self.full_sequence_data["aria_t_origin_deviceE"][start:end], 
                        self.device
                    )
                    print_str = print_str + 'Eye Position loss: {} \n'.format(eye_loss.item())
                    loss_logs["eye_loss"] = eye_loss.item()
                    total_loss += float(active_weights["eye_pos"]) * eye_loss

                if "pose_prior" in active_weights:
                    pose_loss = get_pose_loss(smplx_out.joints, self.full_sequence_data["egomc_global_poses"][start:end, :, :3, 3], 'egomc')
                    print_str = print_str + 'Pose Prior : {} \n'.format(pose_loss.item())
                    loss_logs["pose_loss"] = pose_loss.item()
                    total_loss += float(active_weights["pose_prior"])* pose_loss

        
                print_str = 'Total loss: {} \n'.format(total_loss.item()) + print_str
                total_loss.backward()
                optimizer.step()
                end_time = time.time()
                
                if self.pip_cfg.debug:
                    print('Window Start: {} Iter: {}, time taken {}'.format(start, it, end_time - str_time))
                    print(print_str)
                
                #Visalisation 
                if self.pip_cfg.wandb:
                    wandb.log({
                        "total_loss": total_loss.item(),
                        **loss_logs,
                    })
            
            if "contact" in active_weights or "penetration" in active_weights:
                params_batch = {
                    "contact_dist": contact_dist,
                    "nopen": nopen_mask,
                    "contact_vertex": contact_vertex
                }
                self.update_payload(params_batch, start, end)
              
            body_model.update_poses(poses_batch, start, end)
    
    
    
    
    def ik(self, opt_cfg: dict, body_model: SMPLXHandler, global_poses: np.ndarray):
        """Optimize data from all initialized loaders."""
        if self.device.type == "cpu":
            print("[WARNING]: CUDA not available, optimization will be omitted.")
        
        #init
        nframes = body_model.smplx_poses["global_orient"].shape[0]
        total_steps = math.ceil(nframes / opt_cfg["batch_size"])
        for start in tqdm(range(0, nframes, opt_cfg["batch_size"]), total=total_steps, desc="Batch IK"):
            end = min(start+opt_cfg["batch_size"], nframes)
            batch_size = end - start
            
            #init poses batch
            poses_batch = body_model.get_poses_window(start, end)
            poses_batch = self.set_opt_variables(opt_cfg["opt_vars"], poses_batch, body_model)
            optimizer = torch.optim.Adam([
                poses_batch["body_pose"],
                ], opt_cfg["learning_rate"], betas=(0.9, 0.999))
             
            for it in range(0, opt_cfg["max_iterations"]):
                optimizer.zero_grad()
                str_time = time.time()
                
                smplx_out = body_model.forward_grad(poses_batch)
     
                #### LOSSES ####
                
                total_loss = 0.0 
                print_str = ''
                loss_logs = {}

                pose_loss = get_pose_loss(smplx_out.joints, global_poses[start:end, :, :3, 3], body_model.sensor_type)
                print_str = print_str + 'Pose Prior : {} \n'.format(pose_loss.item())
                loss_logs["pose_loss"] = pose_loss.item()
                total_loss += float(opt_cfg["weights"]["pose_prior"]) * pose_loss

        
                print_str = 'Total loss: {} \n'.format(total_loss.item()) + print_str
                total_loss.backward()
                optimizer.step()
                end_time = time.time()
                
                if self.pip_cfg.debug:
                    print('Window Start: {} Iter: {}, time taken {}'.format(start, it, end_time - str_time))
                    print(print_str)
                
                #Visalisation 
                if self.pip_cfg.wandb:
                    wandb.log({
                        "total_loss": total_loss.item(),
                        **loss_logs,
                    })
            body_model.update_poses(poses_batch, start, end)
    
    
    
    def update_payload(self, param_dict: dict, start: int, end: int):
        for key, val in param_dict.items():
                if val is not None and val.dtype == torch.dtype:
                    self.opt_payload[key][start:end] = val.detach().numpy()

       
       
    def _prepare_scene_mesh(self, opt_cfg: dict, batch_size: int):
        """
        Initializes the scene mesh as a batched PyTorch3D Meshes object.
        Optimized to avoid redundant memory copies.
        """
    
        try:
            from pytorch3d.structures import Meshes
        except Exception as e:
            raise RuntimeError(
                "pytorch3d is required to prepare the scene mesh (Meshes import failed). "
                "Install pytorch3d compatible with your PyTorch/CUDA setup: {}".format(e)
            ) from e
        scene_verts = torch.from_numpy(self.full_sequence_data["scene_vertices"]).float().to(self.device)
        scene_normals = torch.from_numpy(self.full_sequence_data["scene_normals"]).float().to(self.device)
        scene_faces = torch.from_numpy(self.full_sequence_data["scene_faces"]).long().to(self.device)

        verts_list = [scene_verts] * batch_size
        faces_list = [scene_faces] * batch_size

        scene_mesh_torch = Meshes(verts=verts_list, faces=faces_list)
        
        return scene_mesh_torch, scene_verts, scene_normals
 
    def set_opt_variables(self, opt_cfg, params_dict, body_model: SMPLXHandler):
        """Sets the optimization variables based on the configuration."""
        # Body Pose
        body_opt = opt_cfg.get("body_pose", [])
        body_pose = params_dict["body_pose"]
        
        if isinstance(body_opt, list) and len(body_opt) > 0:
            body_pose.requires_grad = True
            mask = torch.zeros_like(body_pose)
            for group in body_opt:
                if group in body_model.JOINT_GROUPS:
                    mask[:, body_model.JOINT_GROUPS[group], :] = 1.0
            body_pose.register_hook(lambda grad: grad * mask)
        elif body_opt is True:
            body_pose.requires_grad = True
        else:
            body_pose.requires_grad = False

        # Translation
        trans_opt = opt_cfg.get("trans", [])
        trans = params_dict["transl"]
        
        if isinstance(trans_opt, list) and len(trans_opt) > 0:
            trans.requires_grad = True
            t_mask = torch.zeros_like(trans)
            if "x" in trans_opt: t_mask[:, 0] = 1.0
            if "y" in trans_opt: t_mask[:, 1] = 1.0
            if "z" in trans_opt:  t_mask[:, 2] = 1.0
            trans.register_hook(lambda grad: grad * t_mask)
        elif trans_opt is True:
            trans.requires_grad = True
        else:
            trans.requires_grad = False

        # Misc Variables
        params_dict["global_orient"].requires_grad = (opt_cfg.get("global_orient") is True)
        return params_dict