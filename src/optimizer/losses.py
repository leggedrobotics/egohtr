import torch
import numpy as np

try:
    # Try importing the CUDA version first
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        raise ImportError("CUDA not available, falling back to Python version")
    from pytorch3d.structures import Pointclouds
    from src.optimizer.losses_helper import point_mesh_face_distance 
    from external.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist
    Chamfer3D = chamfer_3DDist()
    
except ImportError:
    # Fallback: Python implementation
    from external.ChamferDistancePytorch.chamfer_python import distChamfer as Chamfer3D
    print("Using Python (CPU) ChamferDistance")

from src.optimizer.helper import get_retarget_idxs

RETARGET_EGO_IDX, SMPLX_JOINTS_IDX = get_retarget_idxs('egomc')
RETARGET_EXO_IDX, _ = get_retarget_idxs('exomc')

def get_pose_loss(joints_pos, joints_pos_target, type) -> torch.Tensor:
    joints_pos_target_torch = torch.from_numpy(joints_pos_target).to(joints_pos.device).type(torch.float32)
    # Mean Squared Error (L2 Loss)
    if type == 'egomc':
        retarget_idxs = RETARGET_EGO_IDX
    else:
        retarget_idxs = RETARGET_EXO_IDX

    # retarget_idxs = RETARGET_EGO_IDX if type == 'egomc' else RETARGET_EXO_IDX
    joint_position_loss = torch.nn.functional.mse_loss(
        joints_pos[:, SMPLX_JOINTS_IDX, :],
        joints_pos_target_torch[:, retarget_idxs, :],
    )
    return joint_position_loss


def get_contact_loss(scene_vertices, scene_normals, scene_mesh, body_vertices, footlock, candidates, window, batch_size, device):
    """
    Compute contact loss for front and heel foot regions, and select the one with higher penetration loss. 
    Uses same contact vertices for the whole batch (the one in contact at the start frame)

    Args:
        scene_vertices: (N, 3)
        scene_normals: (N, 3)
        body_vertices: (B, V, 3)
        contact_indices_dict: dict
        config: configuration object (optional)

    Returns:
        final_contact_loss: torch.Tensor
    """

    part_loss = torch.zeros(batch_size, 2)
    mask_penetration = torch.zeros(batch_size, 2)
    contact_dist = torch.zeros(batch_size, 2)
    contact_vertex = np.zeros((batch_size, 10, 2))
    idx = 0
    frame_contact_loss = torch.zeros((batch_size, 1)).to(device)
    total_contact_loss = torch.zeros(1).to(device)
    frame_penetration_loss = torch.zeros((batch_size, 1)).to(device)

    for part_name, contact_indices in candidates.items():
        # Select body contact vertices
        contact_indices_torch = torch.from_numpy(contact_indices).to(device)
        body_contact_vertices = body_vertices[:, contact_indices, :]
        scene_vertices_batched = scene_vertices.unsqueeze(0).expand(body_contact_vertices.shape[0], -1, -1)

        # Compute Chamfer distance
        body2floor_dist, floor2body_dist, body2floor_idxs, floor2body_idxs = Chamfer3D(
            body_contact_vertices, scene_vertices_batched
        )

        # Signed penetration term
        floor_normals = scene_normals[body2floor_idxs]
        vec = body_contact_vertices - scene_vertices[body2floor_idxs]
        signed_dists = torch.sum(vec * floor_normals, dim=-1) + 0.02
        part_penetration_loss = torch.mean(torch.nn.functional.softplus(signed_dists, beta=5e2)**2, dim=1)
        frame_penetration_loss = frame_penetration_loss + part_penetration_loss.unsqueeze(-1)
        eps = 1e-3
        no_pen = torch.abs(part_penetration_loss) < eps

        if torch.cuda.is_available():
            min_dists_z, min_dists_z_idx = torch.topk(body_contact_vertices[:,:,2], k=10, dim=1, largest=False) 
            min_dist_vertices_torch = contact_indices_torch[min_dists_z_idx]
            min_dist_vertices = contact_indices[min_dists_z_idx.cpu().numpy()]

            body_pts = torch.gather(input=body_vertices, dim=1, index=min_dist_vertices_torch.unsqueeze(2).repeat(1,1,3))
            body_pts_list = list(torch.unbind(body_pts, dim=0))
            body_pcd = Pointclouds(points=body_pts_list)

            # contact loss
            part_contact_loss = torch.mean(torch.gather(input=body2floor_dist, dim=1, index=min_dists_z_idx), dim=1).unsqueeze(-1)
            if part_name in ["RightFoot"]:
                mask = torch.tensor(footlock["RightFoot"][window[0]:window[1]]).unsqueeze(-1).to(device) #& no_pen
                contact_loss, dist_r = point_mesh_face_distance(meshes=scene_mesh, pcls=body_pcd, mask=mask)
                frame_contact_loss = frame_contact_loss + part_contact_loss * mask
                total_contact_loss = total_contact_loss + contact_loss
                contact_dist[:, idx] = dist_r.squeeze()

            
            if part_name in ["LeftFoot"]:
                mask = torch.tensor(footlock["LeftFoot"][window[0]:window[1]]).unsqueeze(-1).to(device) #& no_pen
                contact_loss, dist_l = point_mesh_face_distance(meshes=scene_mesh, pcls=body_pcd, mask=mask)
                frame_contact_loss = frame_contact_loss + part_contact_loss * mask
                total_contact_loss = total_contact_loss + contact_loss
                contact_dist[:, idx] = dist_l.squeeze()

            part_loss[:, idx] = part_contact_loss.squeeze() 
            contact_vertex[:, :, idx] = min_dist_vertices
        mask_penetration[:, idx] = no_pen.squeeze()
        idx += 1

    
    final_contact_loss = total_contact_loss #torch.mean(frame_contact_loss)
    final_penetration_loss = torch.mean(frame_penetration_loss)
    return final_contact_loss, final_penetration_loss, part_loss, mask_penetration, contact_vertex, contact_dist

def get_head_pos_loss(joint_pos, joint_pos_prior, device) -> torch.Tensor:
    """L2 loss between current and prior head positions.

    Ensures both tensors are float32 on the given device to avoid dtype
    mismatches (e.g., NumPy float64 arrays vs. model float32 tensors).
    """

    if isinstance(joint_pos_prior, np.ndarray):
        joint_pos_prior = torch.from_numpy(joint_pos_prior)
    joint_pos_prior = joint_pos_prior.to(device=device, dtype=torch.float32)
    joint_pos = joint_pos.to(device=device, dtype=torch.float32)

    head_pos_loss = torch.mean((joint_pos - joint_pos_prior) ** 2)
    return head_pos_loss

def get_foot_pos_loss(joint_pos, joint_pos_prior, device) -> torch.Tensor:
    if isinstance(joint_pos_prior, np.ndarray):
        joint_pos_prior = torch.from_numpy(joint_pos_prior)
    joint_pos_prior = joint_pos_prior.to(device=device, dtype=torch.float32)
    joint_pos = joint_pos.to(device=device, dtype=torch.float32)

    foot_pos_loss = torch.mean((joint_pos - joint_pos_prior) ** 2)
    return foot_pos_loss


def get_smooth_loss(joint_pos, prior_joint_pos, start, batch_size, device) -> torch.Tensor:
    frame_smooth_loss = torch.zeros((batch_size, 1)).to(device)

    if start:
        #add last from previous batch
        prev = torch.from_numpy(prior_joint_pos[start-1]).to(device=device, dtype=joint_pos.dtype)
        extended_joint_pos = torch.cat((prev.unsqueeze(0), joint_pos), dim=0)
    else: 
        prev = joint_pos[-1, :, :].detach().unsqueeze(0)
        extended_joint_pos = torch.cat((prev, joint_pos), dim=0)
    
    vel = torch.diff(extended_joint_pos, dim=0)
    speed = torch.norm(vel, dim=-1)
    frame_smooth_loss = torch.mean(speed ** 2)     
    return frame_smooth_loss


def get_eye_loss(eye_pos, eye_pos_aria, device): 
    if isinstance(eye_pos_aria, np.ndarray):
        eye_pos_aria = torch.from_numpy(eye_pos_aria)
    eye_pos_aria = eye_pos_aria.to(device=device, dtype=torch.float32)
    eye_pos = eye_pos.to(device=device, dtype=torch.float32)

    eye_loss = torch.mean((eye_pos - eye_pos_aria)**2)
    return eye_loss


