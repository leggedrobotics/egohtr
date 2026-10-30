import trimesh
import numpy as np
from typing import Dict

import open3d as o3d

from pye57 import E57
from pathlib import Path
from scipy.spatial import KDTree

def load_point_cloud(pcd_path: Path) -> np.ndarray:
    """
    Loads an E57 point cloud and returns an Nx3 numpy array.
    
    Args:
        pcd_path: Path to the .e57 file.
    """          
    blk_scan = E57(str(pcd_path)).read_scan(0, intensity=True, colors=True, ignore_missing_fields=True)
    
    x = blk_scan["cartesianX"]
    y = blk_scan["cartesianY"]
    z = blk_scan["cartesianZ"]

    # Stack into Nx3 numpy array
    return np.vstack((x, y, z)).T


def load_obj(obj_file, flag=False) -> Dict[str, np.ndarray]:
    """
    Load OBJ file and return a dictionary of geometric data.
    
    Args:
        obj_file: Path to the .obj file.
        process_mesh: If True, trimesh will attempt to repair/merge vertices.
    """
    # Load meshes
    mesh = trimesh.load(str(obj_file), process=False, force='mesh')
    color_visual = mesh.visual.to_color()
    return {
        "vertices": np.array(mesh.vertices, dtype=np.float32),
        "faces": np.array(mesh.faces, dtype=np.int32),
        "normals": np.array(-mesh.vertex_normals, dtype=np.float32),  #invert mesh normals
        "vertex_colors": np.array(color_visual.vertex_colors, dtype=np.float32) / 255.0,
    }

def downsample_cloud(points: np.ndarray, voxel_size=0.25) -> np.ndarray:
    """
    Downsample a point cloud using a voxel grid filter.
    
    Args:
        points: Nx3 input array.
        voxel_size: Size of the voxel cube (in meters). 0.015 = 1.5cm.
    """
    if points.size == 0:
        return points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.voxel_down_sample(voxel_size)
    return np.array(pcd.points).astype(np.float32)


def get_global_height(points: np.ndarray, cfg: dict) -> float:
    """
    Estimate the global floor height using RANSAC plane fitting with pre-filtering and normal checks.
    Args:
        points: Nx3 array of point cloud data.
        cfg: Configuration dictionary containing RANSAC parameters and thresholds.
    Returns:
        Estimated floor height (z-value of the plane).
    """
    # 1. Pre-filtering (Passthrough)
    mask = (points[:, 2] > cfg['z_min']) & \
           (points[:, 2] < cfg['z_max'])
    filtered_points = points[mask]

    if len(filtered_points) < cfg['ransac_n']:
        raise ValueError("Not enough points remaining after pre-filtering.")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_points)

    # 2. Iterative RANSAC with Normal Check
    for _ in range(cfg['max_attempts']):
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=cfg['distance_threshold'], 
            ransac_n=cfg['ransac_n'], 
            num_iterations=cfg['num_iterations']
        )
        
        [a, b, c, d] = plane_model
        
        # Check if normal is roughly vertical (Z-axis)
        if abs(c) > cfg['normal_threshold']:
            return -d / c

        # Remove the detected non-floor plane and try again
        pcd = pcd.select_by_index(inliers, invert=True)
        if len(pcd.points) < cfg['ransac_n']:
            break

    raise RuntimeError("Could not find a valid floor plane.")

def get_local_height(point2d: np.ndarray, points: np.ndarray, cfg: dict) -> float:
    pcd_tree = KDTree(points[:, :2])
    _, idx = pcd_tree.query(point2d, k=cfg['points'])
    return float(points[idx, 2].mean())

def transform_obj(obj: Dict[str, np.ndarray], transform: np.ndarray) -> Dict[str, np.ndarray]:
    """Apply a 4x4 transformation matrix to an OBJ dictionary."""
    vertices_hom = np.hstack((obj['vertices'], np.ones((obj['vertices'].shape[0], 1))))
    transformed_vertices = (transform @ vertices_hom.T).T[:, :3]
    return {
        "vertices": transformed_vertices,
        "faces": obj['faces'],
        "normals": obj['normals'],
        "vertex_colors": obj['vertex_colors'],
    }