import yaml

import open3d as o3d
import numpy as np

from pathlib import Path

from scipy.spatial.transform import Rotation as R

from src.loader.vggt_wrapper import execute_vggt
from src.loader.utils import get_image_names
from src.loader.helper.scene_helper import downsample_cloud


def load_config(path: Path):
    with open(str(path), "r") as f:
        return yaml.safe_load(f)

def dump_config(config: dict, path: Path, type: str):
    with open(str(path), "r") as f:
        cfg = yaml.safe_load(f)
    cfg[type].update(config)
    with open(str(path), "w") as f:
        yaml.dump(cfg, f)

def icp(source_pcd : np.ndarray , target_pcd, init_T=np.eye(4), max_iterations=20000, threshold=0.05):
    """
    Perform point-to-point ICP between two sets of points A and B.
    Args:
        A: Nx3 numpy array of source points
        B: Nx3 numpy array of target points
        init_T: Initial transformation (4x4 numpy array) to start the alignment
        max_iterations: Maximum number of ICP iterations
        tolerance: Convergence tolerance
    Returns:
        final_transform: Final transformation matrix (4x4 numpy array)
    """
    print("[INFO]: Starting ICP registration...")
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_pcd)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_pcd)
    T_target_source = o3d.pipelines.registration.registration_icp(
        source, target, threshold, init_T,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
    )
    print(f"[INFO]: ICP registration completed:\n{T_target_source}")
    return T_target_source



def p2p_vggt(source_pcd: np.ndarray, target_pcd: np.ndarray, T_sourceFP_sourceW, scene_dir: Path, aria_dir: Path, root_dir: Path, seq_config: dict):
    """ Perform Point cloud to point cloud registration using vggt for a feature-based global alignment and ICP to fintune.
    Args:
        source_pcd: Nx3 numpy array of source points
        target_pcd: Nx3 numpy array of target points
        T_sourceFP_sourceW: Transformation matrix from source world to source first pose (4x4 numpy array)
        seq_path: Path to the sequence directory containing images for VGGt
    Returns:
        T_target_source: Final transformation matrix (4x4 numpy array)
    """

    image_names = get_image_names(scene_dir, aria_dir)
    T_targetW_sourceFP = execute_vggt(image_names, root_dir)


    T_targetW_sourceW = T_targetW_sourceFP @ T_sourceFP_sourceW #transform to first pose and then to target world

    #sweep
    # source_pcd, target_pcd_down, best_final_init = sweep_icp(source_pcd, target_pcd, T_targetW_sourceW)
    if "p2p" not in seq_config:
        return T_targetW_sourceW
    else:
        #manual adjustment if errorneous for some sequences (TODO: increase reliability of vggt initialization to avoid manual adjustments)
        best_final_init = manual_init(T_targetW_sourceW, seq_config["p2p"])
    
    final_result = icp(
        source_pcd, 
        target_pcd, 
        init_T=best_final_init,
        max_iterations=50000, 
        threshold=0.05
    )
    return final_result.transformation


def manual_init(T_targetW_sourceW: np.ndarray, seq_config: dict):
    """ Perform Point cloud to point cloud registration using manual initialization based on the first pose and ICP to fintune."""
    
    # add rotation in z axis
    rot_x = R.from_euler('x', seq_config["rot_euler"][0], degrees=True).as_matrix()
    T_rot = np.eye(4)
    T_rot[:3, :3] = rot_x
    T_targetW_sourceW = T_rot @ T_targetW_sourceW

    rot_y = R.from_euler('y', seq_config["rot_euler"][1], degrees=True).as_matrix()
    T_rot = np.eye(4)
    T_rot[:3, :3] = rot_y
    T_targetW_sourceW = T_rot @ T_targetW_sourceW

    rot_z = R.from_euler('z', seq_config["rot_euler"][2], degrees=True).as_matrix()
    T_rot = np.eye(4)
    T_rot[:3, :3] = rot_z
    T_targetW_sourceW = T_rot @ T_targetW_sourceW

    # add offset transformation in x and y direction
    T_targetW_sourceW[0,3] += seq_config["trans"][0] #lateral
    T_targetW_sourceW[1,3] += seq_config["trans"][1] #forward
    T_targetW_sourceW[2,3] += seq_config["trans"][2] # height

    return T_targetW_sourceW


def sweep_icp(source_pcd: np.ndarray, target_pcd: np.ndarray, T_targetW_sourceW: np.ndarray):
    target_pcd_down = downsample_cloud(target_pcd, voxel_size=0.05)
    angles = np.linspace(-np.pi/6, np.pi/6, 5)

    # 2. Generate the three cases (one axis fixed at 0)
    # Case 1: X is fixed at 0
    case_x = [(0.0, 0.0, rz) for ry in angles for rz in angles]
    # Case 2: Y is fixed at 0
    case_y = [(rx, 0.0, 0.0) for rx in angles for rz in angles]
    # Case 3: Z is fixed at 0
    case_z = [(0.0, ry, 0.0) for rx in angles for ry in angles]

    # 3. Combine and remove duplicates using a set
    unique_combinations = list(set(case_x + case_y + case_z))

    # 4. Sort to start at (0,0,0) and expand outwards based on total rotation magnitude
    unique_combinations.sort(key=lambda rot: abs(rot[0]) + abs(rot[1]) + abs(rot[2]))

    best_fitness = -1.0
    best_rot_init = T_targetW_sourceW

    print(f"Starting sweep over {len(unique_combinations)} unique angle combinations...")

    # 5. Fast Sweep Loop
    for rx, ry, rz in unique_combinations:
        T_init = np.copy(T_targetW_sourceW)
        print(f"Testing rotation (rx={rx:.2f}, ry={ry:.2f}, rz={rz:.2f}) on top of VGGt initialization...")

        # Apply rotation
        rot_mat = R.from_euler('xyz', [rx, ry, rz]).as_matrix()
        T_rot = np.eye(4)
        T_rot[:3, :3] = rot_mat
        T_init = T_rot @ T_init

        # 3. FAST Evaluation (Low max_iterations!)
        # We just want to see if this initialization "catches" the geometry quickly
        sweep_result = icp(
            source_pcd, 
            target_pcd,  # Use downsampled target for faster evaluation
            init_T=T_init, 
            max_iterations=100,  # <-- Significantly lowered for the sweep
            threshold=0.05
        )

        # 4. Early Stop Condition
        if sweep_result.fitness > 0.5:
            print(f"Early stop! Good initialization found with rough fitness: {sweep_result.fitness:.4f}")
            best_rot_init = sweep_result.transformation
            best_fitness = sweep_result.fitness
            break  # Break out of the itertools loop entirely

        # Track the best rough initialization just in case we never hit 0.3
        if sweep_result.fitness > best_fitness:
            best_fitness = sweep_result.fitness
            best_rot_init= sweep_result.transformation

    # --- 2. TRANSLATION SWEEP ---
    # Start the next sweep using the BEST transformation found above
    print(f"\nStarting TRANSLATION sweep from best rotation (fitness: {best_fitness:.4f})...")

    translations = np.linspace(-0.4, 0.4, 3) # E.g., 5 steps between -0.5 and 0.5

    # Generate cases (sweeping one axis at a time)
    trans_x = [(tx, 0.0, 0.0) for tx in translations]
    trans_y = [(0.0, ty, 0.0) for ty in translations]
    trans_z = [(0.0, 0.0, tz) for tz in translations]

    unique_translations = list(set(trans_x + trans_y + trans_z))
    unique_translations.sort(key=lambda t: abs(t[0]) + abs(t[1]) + abs(t[2]))

    best_final_init = best_rot_init

    for tx, ty, tz in unique_translations:
        # IMPORTANT: Copy from best_rot_init, not the original T_targetW_sourceW
        T_init = np.copy(best_rot_init) 
        print(f"Testing translation (tx={tx:.2f}, ty={ty:.2f}, tz={tz:.2f})...")

        # Apply translation offset
        T_init[0, 3] += tx
        T_init[1, 3] += ty
        T_init[2, 3] += tz

        sweep_result = icp(
            source_pcd, target_pcd, init_T=T_init, max_iterations=10, threshold=0.05
        )

        if sweep_result.fitness > 0.5:
            print(f"Early stop! Good translation found with fitness: {sweep_result.fitness:.4f}")
            best_final_init = sweep_result.transformation
            break

        if sweep_result.fitness > best_fitness:
            best_fitness = sweep_result.fitness
            best_final_init = sweep_result.transformation

    return source_pcd, target_pcd, best_final_init


    # return icp(source_pcd, target_pcd, init_T=T_targetW_sourceW, max_iterations=500, threshold=0.05).transformation #Finetune


def p2p_classic(source_pcd: np.ndarray, target_pcd: np.ndarray, T_targetW_targetFP, T_sourceFP_sourceW, vec_source_FP):
    """Perform Point cloud to point cloud registration using classic pose alignement for a global alignment and ICP to fintune.
    Args:
        source_pcd: Nx3 numpy array of source points
        target_pcd: Nx3 numpy array of target points
        T_targetW_targetFP: Transformation matrix from target first pose to target world (4x4 numpy array)
        T_sourceFP_sourceW: Transformation matrix from source world to source first pose (4x4 numpy array)
        vec_source_FP: headinig vector from source first pose (np.ndarray of shape (2,) )
    Returns:
        T_target_source: Final transformation matrix (4x4 numpy array)"""
    
    # get xy-plane rotation from source FP vector to target FP vector
    yaw_source = np.arctan2(vec_source_FP[1], vec_source_FP[0])
    rot_source = R.from_euler('z', yaw_source)
    z_axis_target = T_targetW_targetFP[:3, :3][:,2]
    yaw_target = np.arctan2(z_axis_target[1], z_axis_target[0])
    rot_target = R.from_euler('z', yaw_target)
    R_targetFP_sourceFP = rot_target * rot_source.inv()
    #get transformation matrix
    T_targetFP_sourceFP = np.eye(4)
    T_targetFP_sourceFP[:3, :3] = R.from_euler('z', R_targetFP_sourceFP.as_euler('xyz')[2]).as_matrix()
    #T_targetFP_sourceFP[:3, :3] = R.from_euler('z', np.pi/2).as_matrix()    
    T_world_initAria = np.eye(4)
    T_world_initAria[:3, 3] = T_targetW_targetFP[:3, 3]

    T_targetW_sourceW = T_world_initAria @ T_targetFP_sourceFP @ T_sourceFP_sourceW
    
    return icp(source_pcd, target_pcd, init_T=T_targetW_sourceW, max_iterations=50000, threshold=0.03).transformation #Finetune


def get_rotation_chain_vectorized(joint_channels: list, joint_rot: np.ndarray) -> np.ndarray:
    """
    Args:
        joint_channels: List of strings (e.g., ['Xrotation', 'Yrotation', 'Zrotation'])
        joint_rotations_tensor: np.ndarray of shape (Frames, 3)
    Returns:
        rotations_matrices: np.ndarray of shape (Frames, 3, 3)
    """
    # extract the order string (e.g., 'xyz')
    order = "".join([c[0].lower() for c in joint_channels])
    # compute the entire batch of matrices at once.
    rot = R.from_euler(order.upper(), joint_rot, degrees=True)
    
    return rot.as_matrix()


def get_fixed_glass_translation(dist2glass = 0.08, height_trans = 0.05, joint = 'Head'):
    """
    Fixed translation from Aria device to head joint or eye joint in device coordinate system.
    Args:
        dist2glass: distance from joint to glass along x axis
        height_trans: height translation along z axis
        joint: 'Head' or 'Eyes'
    Returns:
        joint_pos: np.ndarray of shape (3, N)
    """
    
    glass_phi = np.deg2rad(-52.1175)
    if joint == 'Eyes':
        glass_center_front = np.array([[height_trans, -0.05478/2, -0.0419775/2], [height_trans, -0.05478 * 3/2, -0.0419775 * 3/2]])
    elif joint == 'Head':
        glass_center_front = np.array([[height_trans, -0.05478, -0.0419775]])
    joint_pos = np.zeros((glass_center_front.shape[0], 3))
    joint_pos[:,0] = glass_center_front[:,0]
    joint_pos[:,1] = np.cos(glass_phi) * dist2glass + glass_center_front[:,1]
    joint_pos[:,2] = np.sin(glass_phi) * dist2glass + glass_center_front[:,2]
    return joint_pos.T
