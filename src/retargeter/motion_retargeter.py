import torch
import os
import numpy as np

from abc import ABC

from scipy.spatial.transform import Rotation as R
from smplx.joint_names import JOINT_NAMES

from external.GMR.general_motion_retargeting.utils.lafan1 import (
    load_bvh_file
)

from src.config.configs import SequenceContext
from src.retargeter.config_types.retargeting import RetargetingConfig
from src.retargeter.config_types.robot import RobotConfig
from src.retargeter.config_types.task import TaskConfig
from src.retargeter.config_types.retargeter import RetargeterConfig
from src.retargeter.scripts.robot_retarget import robot_retarget
from src.retargeter.retarget_helper import (
    coord_offset,
    coord_offset,
    smplx_to_egomc_map,
    smplx_to_exomc_map,
    get_smplx_local_joint,
    convexify_mesh,
    export_decomposed_mesh,
    generate_xml,
)
from src.body_model.smplx_helper import SMPLX_BODY_JOINTS

class EgoHTGRetargeter(ABC):
    
    def __init__(self, pip_cfg):
        self.pip_cfg = pip_cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_body_joints = len(SMPLX_BODY_JOINTS)
        
        
    def run(self, ctx: SequenceContext, seq_cfg: dict):
        
        self._load_data(ctx)
        if self.pip_cfg.overwrite_retarget or not ctx.scene_decomposed.exists() or not ctx.combined_mesh.exists():
            self._decompose_scene(ctx)
        if self.pip_cfg.overwrite_retarget or not ctx.box_assets_xml.exists() or not ctx.box_body_xml.exists():
            self._generate_assets(ctx)
        self._retarget_sequence(ctx, seq_cfg)        
       
    
    def _load_data(self, ctx: SequenceContext):
        """Load data from all initialized loaders and slice it to a given time period."""
        with np.load(ctx.gen_seq, allow_pickle=True) as loader:
            full_sequence_data = dict(loader)

        start_time, end_time = self.pip_cfg.period_retarget if self.pip_cfg.period_retarget is not None else [None, None]
        if start_time is None and end_time is None:
            self.full_sequence_data = full_sequence_data
            return

        # Use a primary timestamp array to determine the slice indices. 'egomc_timestamp' is a good candidate.
        if 'egomc_timestamp' in full_sequence_data:
            timestamps = full_sequence_data['egomc_timestamp']
        elif 'aria_timestamp' in full_sequence_data:
            timestamps = full_sequence_data['aria_timestamp']
        else:
            # If no timestamp is found, we can't slice.
            print("[WARNING]: No timestamp array found in data. Cannot slice sequence.")
            self.full_sequence_data = full_sequence_data
            return

        start_idx = 0
        if start_time is not None:
            start_idx = np.searchsorted(timestamps, start_time, side='left')

        end_idx = len(timestamps)
        if end_time is not None:
            end_idx = np.searchsorted(timestamps, end_time, side='right')

        sliced_data = {}
        for key, value in full_sequence_data.items():
            if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == len(timestamps):
                sliced_data[key] = value[start_idx:end_idx]
            else:
                # For arrays that don't match the timestamp length (like scene geometry) or are scalars, keep them as is.
                sliced_data[key] = value
        
        self.full_sequence_data = sliced_data
            
    def _decompose_scene(self, ctx: SequenceContext):
        print("[INFO]: Decomposing scene mesh for sequence.")
        output_folder = ctx.scene_decomposed
        output_combined= ctx.combined_mesh
        
        os.makedirs(output_folder, exist_ok=True)
        for f in os.listdir(output_folder):
            os.remove(os.path.join(output_folder, f))

        coacd_parts = convexify_mesh(self.full_sequence_data["scene_vertices"], self.full_sequence_data["scene_faces"])
        export_decomposed_mesh(coacd_parts, output_folder, output_combined)
    
    def _generate_assets(self, ctx: SequenceContext):
        print("[INFO]: Generating retarget XML files.")
        mesh_folder = ctx.scene_decomposed
        relative_mesh_folder = ctx.relative_scene_decomposed
        output_assets_xml = ctx.box_assets_xml
        output_body_xml = ctx.box_body_xml
        generate_xml(mesh_folder, relative_mesh_folder, output_assets_xml, output_body_xml)
    
    def _retarget_sequence(self, ctx: SequenceContext, seq_cfg: dict):
        rt_cfg = RetargetingConfig(
            task_name=ctx.name,
            data_path=self.pip_cfg.data_path,
            save_dir=ctx.seq_dir,
            robot_config=RobotConfig(
                robot_type=seq_cfg["robot_type"],
                robot_urdf_file=str(ctx.robot_models / seq_cfg["robot_type"] / f"{seq_cfg['robot_assets']}.urdf"),
            ),
            retargeter=RetargeterConfig(
                debug=self.pip_cfg.debug,
                visualize=True,
            ),
            task_config=TaskConfig(
                object_dir=ctx.retarget_dir,
            ),
        )
        robot_retarget(rt_cfg, self.full_sequence_data)
        
    def exomc_to_smplx(self, global_pos, global_quat, joint_names, parents, seq_cfg, sensor_cfg) -> dict:
        print("[INFO]: Retargeting exomocap to SMPLX.")
        n_frames = global_pos.shape[0]
        scalar_first = sensor_cfg["scalar_first"]
        
        betas = torch.tensor(seq_cfg["betas"], device=self.device).unsqueeze(0).repeat(n_frames, 1)
        trans_tensor = torch.zeros((n_frames, 3), device=self.device, dtype=torch.float32)
        global_orient_tensor = torch.zeros((n_frames, 3, 3), device=self.device, dtype=torch.float32)
        body_pose_tensor = torch.zeros(
            (n_frames, self.num_body_joints, 3, 3), device=self.device, dtype=torch.float32
        )
        basis_change = coord_offset["exomc"]["joint_basis_change"]
        world_offset = coord_offset["exomc"]["world"]

        for frame in range(n_frames):
            #orientation
            global_orient = R.from_quat(global_quat[frame, joint_names.index("Hips"), :], scalar_first=scalar_first)
            global_orient = global_orient * world_offset
            global_orient_tensor[frame] = torch.tensor(global_orient.as_matrix(), device=self.device).float()
            #translation
            trans_tensor[frame] = torch.tensor(global_pos[frame, joint_names.index("Hips"), :], device=self.device).float()
            # body pose
            body_pose = torch.zeros(
                (self.num_body_joints, 3, 3), device=self.device, dtype=torch.float32
            )
            for i, joint_name in enumerate(
                JOINT_NAMES[1:(self.num_body_joints + 1)]
            ):  # Joint names for SMPL and SMPL-X are equal
                if joint_name not in smplx_to_exomc_map:
                    continue
                exomc_joint_name = smplx_to_exomc_map[joint_name]
                exomc_joint_idx = joint_names.index(exomc_joint_name)
                parent_name = joint_names[parents[exomc_joint_idx]]
                body_pose[i] = torch.from_numpy(
                    get_smplx_local_joint(
                        exomc_joint_name,
                        global_quat[frame, exomc_joint_idx, :],
                        parent_name,
                        global_quat[frame, joint_names.index(parent_name), :],
                        basis_change,
                        sensor="exomc",
                        scalar_first=scalar_first,
                    )
                )
            body_pose_tensor[frame] = body_pose.clone()
        
        return {
            "body_pose": body_pose_tensor.detach(),
            "global_orient": global_orient_tensor.detach(),
            "transl": trans_tensor.detach(),
            "betas": betas,
        }
    
    def egomc_to_smplx(self, bvhfile, start_idx, end_idx, seq_cfg, sensor_cfg, loader_cfg) -> dict:
        print("[INFO]: Retargeting egomocap to SMPLX.")
        # Load the LAFAN BVH file
        data, frames, _ = load_bvh_file(bvhfile, loader_cfg, format=sensor_cfg["format"])
        frames = frames[start_idx:end_idx]
        n_frames = len(frames)
        
        betas = torch.tensor(seq_cfg["betas"], device=self.device).unsqueeze(0).repeat(n_frames, 1)
        trans_tensor = torch.zeros((n_frames, 3), device=self.device, dtype=torch.float32)
        global_orient_tensor = torch.zeros((n_frames, 3, 3), device=self.device, dtype=torch.float32)
        body_pose_tensor = torch.zeros(
            (n_frames, self.num_body_joints, 3, 3), device=self.device, dtype=torch.float32
        )
        basis_change = coord_offset["egomc"]["joint_basis_change"]
        offset_world = coord_offset["egomc"]["world"]
        for frame in range(n_frames):
            # Get the root orientation and body pose for the current frame
            global_orient = (
                R.from_quat(frames[frame]["Hips"][1], scalar_first=True)
            )
            global_orient = offset_world * global_orient
            global_orient_tensor[frame] = torch.tensor(global_orient.as_matrix(), device=self.device).float()
            body_pose = torch.zeros(
                (self.num_body_joints, 3, 3), device=self.device, dtype=torch.float32
            )
            for i, joint_name in enumerate(
                JOINT_NAMES[1:(self.num_body_joints + 1)]
            ):  # Joint names for SMPL and SMPL-X are equal
                if joint_name not in smplx_to_egomc_map:
                    continue
                egomc_joint_name = smplx_to_egomc_map[joint_name]
                parent_name = data.bones[data.parents[data.bones.index(egomc_joint_name)]]
                body_pose[i] = torch.from_numpy(
                    get_smplx_local_joint(
                        egomc_joint_name,
                        frames[frame][egomc_joint_name][1],
                        parent_name,
                        frames[frame][parent_name][1],
                        basis_change,
                        sensor="egomc"
                    )
                )
            body_pose_tensor[frame] = body_pose.clone()
            
        return {
            "body_pose": body_pose_tensor.detach(),
            "global_orient": global_orient_tensor.detach(),
            "transl": trans_tensor.detach(),
            "betas": betas,
        }
    