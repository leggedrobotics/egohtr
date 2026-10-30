import numpy as np
import matplotlib.pyplot as plt


def to_homogeneous(pose):
    """Convert 3x4 pose to 4x4 homogeneous matrix"""
    bottom_row = np.array([[0,0,0,1]], dtype=pose.dtype)
    return np.vstack([pose, bottom_row])

# Define a simple frustum geometry in camera space
def create_frustum(scale=0.1):
    """Create a pyramid frustum in camera coordinates."""
    vertices = np.array([
        [0, 0, 0],  # camera origin
        [-1, -1, -1], [-1, -1, 1],
        [1,  -1, 1], [1,  -1, -1]
    ]) * scale
    edges = [
        (0,1), (0,2), (0,3), (0,4),
        (1,2), (2,3), (3,4), (4,1)
    ]
    return vertices, edges

def transform_vertices(vertices, pose):
    """Apply a 4x4 transform to vertices"""
    v_h = np.hstack([vertices, np.ones((vertices.shape[0],1))])  # to homogeneous
    v_t = (pose @ v_h.T).T
    return v_t[:,:3]

def plot_cam_poses(camera_poses: np.ndarray):
    # Convert to numpy
    # camera_poses = camera_poses.cpu().numpy()  # (1, N, 3, 4)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #ax.set_title('VGGT Camera Poses')

    frustum_vertices, frustum_edges = create_frustum()
    label = [f"aria{i+1}" for i in range(camera_poses.shape[1]-1)] + ["blk"] 
    # Loop through all cameras
    for i in range(camera_poses.shape[1]):
        pose4x4 = to_homogeneous(camera_poses[0,i])
        t = pose4x4[:3,3]

        # Draw camera center
        ax.scatter(t[0], t[1], t[2],s=50, label = label[i])

        # Draw frustum
        verts_world = transform_vertices(frustum_vertices, pose4x4)
        
        for edge in frustum_edges:
            ax.plot(*zip(verts_world[edge[0]], verts_world[edge[1]]), c='b')

    # Equal aspect ratio
    max_range = np.ptp(np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])).max() / 1.0
    mid_x = np.mean(ax.get_xlim())
    mid_y = np.mean(ax.get_ylim())
    mid_z = np.mean(ax.get_zlim())
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    #persepctive
    ax.legend()
    # ax.view_init(elev=0, azim=0)
    plt.tight_layout
    plt.savefig(f"logs/camera_poses.png")
