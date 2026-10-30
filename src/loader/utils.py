import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from projectaria_tools.core.sophus import SE3
from scipy.spatial.transform import Rotation as R
    
def get_homog(pose):
    """Convert 3x4 pose to 4x4 homogeneous matrix"""
    bottom_row = np.array([[0,0,0,1]], dtype=pose.dtype)
    return np.vstack([pose, bottom_row])


def save_image_to_dir(image: np.ndarray, save_dir: Path, filename: str):
    """
    Save an image to a directory using matplotlib's savefig, creating the directory if it does not exist.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / filename
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def transform_points(points: np.ndarray, transform):
    """Apply a 4x4 transformation matrix to a Nx3 point cloud.
    Args:
        points (np.ndarray): Nx3 array of 3D points.
        transform (np.ndarray): 4x4 transformation matrix.
    Returns:
        np.ndarray: Transformed Nx3 array of 3D points.
    """
    # Convert points to homogeneous coordinates
    ones = np.ones((points.shape[0], 1))
    points_homogeneous = np.hstack((points, ones))
    transformed_points = (transform @ points_homogeneous.T).T
    return transformed_points[:, :3]


def transform_normals(normals: np.ndarray, transform):
    """Apply a 4x4 transformation matrix to a Nx3 point cloud."""
    R = transform[:3, :3]
    normals_transformed = normals @ R.T
    normals_transformed_normed = normals_transformed / np.linalg.norm(normals_transformed, axis=1, keepdims=True) + 1e-8
    return normals_transformed_normed 


def get_image_names(scene_dir: Path, aria_dir: Path): 
    aria_img_path = aria_dir / "imgs"
    blk_img_path = scene_dir / "imgs"
     # list img path
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}
    
    import re
    def extract_number(p):
        m = re.search(r'(\d+)', p.stem)
        return int(m.group(1)) if m else -1

    image_names = [str(p) for p in sorted(
    [p for p in aria_img_path.iterdir() if p.suffix.lower() in image_extensions and p.is_file()],
    key=extract_number
    )]
    blk_img_names = [str(p) for p in blk_img_path.iterdir() if p.suffix.lower() in image_extensions and p.is_file()]
    image_names.extend(blk_img_names)
    return image_names

def transform_vggt_correction():
    #90° clockwise rotation around Z
    rotz = R.from_euler('z', -180, degrees=True).as_matrix()
    Rz_90_mat = np.eye(4)
    Rz_90_mat[:3, :3] = rotz
    Rz_90 = SE3.from_matrix(Rz_90_mat)
    T = (Rz_90).to_matrix()
    return T

def get_rotation_transform(rpy_deg: list, order: str = 'xyz') -> np.ndarray:
    """Get a 4x4 rotation transform from roll, pitch, yaw angles in degrees."""
    rot = R.from_euler(order, rpy_deg, degrees=True)
    T = np.eye(4)
    T[:3, :3] = rot.as_matrix()
    return T


def remove_quat_discontinuities(rotations):
    """

    Removing quat discontinuities on the time dimension (removing flips)

    :param rotations: Array of quaternions of shape (T, J, 4)
    :return: The processed array without quaternion inversion.
    """
    rots_inv = -rotations

    for i in range(1, rotations.shape[0]):
        # Compare dot products
        replace_mask = np.sum(rotations[i - 1: i] * rotations[i: i + 1], axis=-1) < np.sum(
            rotations[i - 1: i] * rots_inv[i: i + 1], axis=-1)
        replace_mask = replace_mask[..., np.newaxis]
        rotations[i] = replace_mask * rots_inv[i] + (1.0 - replace_mask) * rotations[i]

    return rotations


def quat_mul(x, y):
    """
    Performs quaternion multiplication on arrays of quaternions

    :param x: tensor of quaternions of shape (..., Nb of joints, 4)
    :param y: tensor of quaternions of shape (..., Nb of joints, 4)
    :return: The resulting quaternions
    """
    x0, x1, x2, x3 = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]
    y0, y1, y2, y3 = y[..., 0:1], y[..., 1:2], y[..., 2:3], y[..., 3:4]

    res = np.concatenate([
        y0 * x0 - y1 * x1 - y2 * x2 - y3 * x3,
        y0 * x1 + y1 * x0 - y2 * x3 + y3 * x2,
        y0 * x2 + y1 * x3 + y2 * x0 - y3 * x1,
        y0 * x3 - y1 * x2 + y2 * x1 + y3 * x0], axis=-1)

    return res


def angle_axis_to_quat(angle, axis):
    """
    Converts from and angle-axis representation to a quaternion representation

    :param angle: angles tensor
    :param axis: axis tensor
    :return: quaternion tensor
    """
    c = np.cos(angle / 2.0)[..., np.newaxis]
    s = np.sin(angle / 2.0)[..., np.newaxis]
    q = np.concatenate([c, s * axis], axis=-1)
    return q

def euler_to_quat(e, order):
    """

    Converts from an euler representation to a quaternion representation

    :param e: euler tensor
    :param order: order of euler rotations
    :return: quaternion tensor
    """
    axis = {
        'Xrotation': np.asarray([1, 0, 0], dtype=np.float32),
        'Yrotation': np.asarray([0, 1, 0], dtype=np.float32),
        'Zrotation': np.asarray([0, 0, 1], dtype=np.float32)}

    q0 = angle_axis_to_quat(e[..., 0], axis[order[0]])
    q1 = angle_axis_to_quat(e[..., 1], axis[order[1]])
    q2 = angle_axis_to_quat(e[..., 2], axis[order[2]])

    return quat_mul(q0, quat_mul(q1, q2))