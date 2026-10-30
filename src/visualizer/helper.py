import numpy as np
import copy
import open3d as o3d
import rerun as rr

from projectaria_tools.core.sophus import SE3


def numpy_to_pcd(points: np.ndarray) -> o3d.geometry.PointCloud:
    """Convert Nx3 numpy array to Open3D point cloud."""    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def draw_registration_result(source, target, transformation):
    # Example usage before draw_registration_result
    source = numpy_to_pcd(source)   # convert numpy array
    target = numpy_to_pcd(target)         # convert numpy array
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])



def ToTransform3D(pose: SE3, from_parent: bool = False) -> rr.Transform3D:
    """
    Helper function to convert Sophus SE3D pose to a Rerun Transform3D
    """
    quat_xyzw = np.roll(
        pose.rotation().to_quat()[0], -1
    )  # change from w,x,y,z to x,y,z,w
    return rr.Transform3D(
        translation=pose.translation()[0],
        rotation=rr.Quaternion(xyzw=quat_xyzw),
        from_parent=from_parent,
    )

