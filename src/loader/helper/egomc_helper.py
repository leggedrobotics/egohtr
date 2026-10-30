import numpy as np

from sklearn.cluster import KMeans
from pathlib import Path
from typing import Callable, Literal


def get_vertex_indices(first_mesh):
    """ Assumes standing person at initalisation."""
    idx = 0
    while True: 
        if first_mesh[idx]:
            verts = first_mesh[idx]["body"]["vertices"]
            return get_feet_vertex_indices(verts)
        idx += 1 
        
def get_feet_vertex_indices(vertices) -> dict[str, np.ndarray]:
    """ Extract lowest vertices in z direction """
    lowest_idx = np.where(vertices[:, 2] < np.percentile(vertices[:, 2], 3))[0]
    lowest = vertices[lowest_idx]

    # Cluster into two feet
    kmeans = KMeans(n_clusters=2, n_init='auto').fit(lowest)
    labels = kmeans.labels_

    # Get indices corresponding to each cluster
    feet_vertex_indices= {
        "LeftFoot": lowest_idx[labels == 0],
        "RightFoot": lowest_idx[labels == 1],
    }
    return feet_vertex_indices


def get_contact_vertex_candidates(vertices) -> dict[str, np.ndarray]:
    """Extracts lowest vertices and clusters into Left/Right × Front/Heel."""
    # Select lowest vertices in z-direction
    lowest_idx = np.where(vertices[:, 2] < np.percentile(vertices[:, 2], 3))[0]
    lowest = vertices[lowest_idx]

    # Cluster into 4 groups (2 feet × 2 subregions)
    kmeans = KMeans(n_clusters=4, n_init='auto').fit(lowest)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # Identify left/right using x-coordinate of cluster centers
    left_clusters = centers[:, 0] < np.mean(centers[:, 0])
    right_clusters = ~left_clusters

    # Within each foot, identify front/heel using y (or whichever axis points forward)
    front_heel_split = np.mean(centers[:, 1])
    foot_labels = {}
    for i, (is_left, center) in enumerate(zip(left_clusters, centers)):
        foot_name = "Left" if is_left else "Right"
        part_name = "Front" if center[1] > front_heel_split else "Heel"
        foot_labels[f"{foot_name}{part_name}"] = lowest_idx[labels == i]

    return foot_labels
    
def load_vertice_indices(mesh_folder_path: Path):
    group_vertices = {}
    group_names = []
    with open(mesh_folder_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue  # skip malformed lines
            group_name = row[0].strip()
            # convert space-separated indices to int list
            vertex_indices = list(map(int, row[1].split()))
            group_vertices[group_name] = vertex_indices
            group_names.append(group_name)
            
    return group_vertices, group_names



def seperate_channels(frames, joints, joints_channel_order: Callable, target_type: Literal['rotation', 'position'], seq_cfg):
    """
    Separate rotation or position channels for given joints from BVH frames.
    
    :param frames: np.ndarray of shape (n_frames, n_channels)
    :param joints: dict of joint names
    :param joints_channel_order: dict mapping joint names to their channel order
    :param target_type: Type of channels to separate, either 'rotation' or 'position'
    :return: np.ndarray joint data of shape (n_frames, n_joints, 3) and joints order dict
    """
    indices = []
    joints_order = {}

    current_idx = 0
    for joint in joints:
        channels = joints_channel_order(joint)
        target_channels = [c for c in channels if target_type in c.lower()]
        joints_order[joint] = target_channels # e.g ['Xrotation', 'Yrotation', 'Zrotation'])
    
        for i, chan in enumerate(channels):
            if target_type in chan.lower():
                indices.append(current_idx + i)
        current_idx += len(channels)
    #extract rotation and reshape  
    data = frames[:, indices]

    if len(indices) != 3*len(joints):
        print(f"[WARNING] Not all joints have saved {target_type}s")
        return np.array([]), joints_order

    data = data.reshape(frames.shape[0], len(joints), 3)

    #set init yaw orientation to zero (TODO: check for coord system consistency)
    data[:, 0, 0] -= data[0, 0, 0]

    return data, joints_order
