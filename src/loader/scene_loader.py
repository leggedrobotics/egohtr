import numpy as np

from typing import Dict

from src.loader.base_loader import DataLoader
from src.loader.helper.scene_helper import (
    load_point_cloud,
    load_obj,
    downsample_cloud,
    get_local_height,
    get_global_height,
    transform_obj,
)
from src.loader.utils import transform_points


class SceneLoader(DataLoader):
    """
    Data loader for BLK data. This loader is responsible for loading the raw point cloud and mesh data of the scene, processing it to extract relevant features such as floor height,
    performing spatial alignment to the scene origin, and preparing the data for export.
    """
    scene_pcd : np.ndarray
    scene_obj : Dict[str, np.ndarray]
      
    def load_data(self):
        # Implement scene data loading logic here
        print(f"[INFO]: Loading Scene data from {self.ctx.name}")
    
        self.scene_pcd = load_point_cloud(self.ctx.scene_pcd)
        self.scene_obj = load_obj(self.ctx.scene_obj)

    def process(self): 
        print("[INFO]: Processing Scene data")
        self.scene_pcd = downsample_cloud(self.scene_pcd, self.sensor_cfg['voxel_size'])

        # Remove points that are not within a limiting box
        min = self.seq_cfg["scene_box"]["min"] if "scene_box" in self.seq_cfg else [-5, -5, -2]
        max = self.seq_cfg["scene_box"]["max"] if "scene_box" in self.seq_cfg else [5, 5, 2]

        bounding_box_min = np.array(min)
        bounding_box_max = np.array(max)
        within_box = np.all((self.scene_pcd >= bounding_box_min) & (self.scene_pcd <= bounding_box_max), axis=1)
        self.scene_pcd_box = self.scene_pcd[within_box]

    def get_floor_height(self, **kwargs) -> float:
        """Get the floor height of the scene."""
        if kwargs.get('pos_xy') is not None:
            return get_local_height(
                point2d=kwargs['pos_xy'],
                points=self.scene_pcd,
                cfg=self.sensor_cfg['k_nn']
            ) # local height at (x,y)
        return get_global_height(
            points=self.scene_pcd,
            cfg=self.sensor_cfg['plane_fit']
        ) # global height of the scene ground plane (preferred for mainly flat terrain)
        
    def get_scene2origin_transform(self) -> np.ndarray:
        """Get the transformation matrix to the scene origin defined at ground level."""
        floor_height = self.get_floor_height()
        transform = np.eye(4)
        transform[2, 3] = -floor_height
        return transform
    
    def spatial_align_data(self, transform):
        self.scene_obj = transform_obj(self.scene_obj, transform)
        self.scene_pcd = transform_points(self.scene_pcd, transform)
        
    def get_export_data(self) -> dict:
        """Prepare data for export."""
        payload = {
            # "scene_pcd": self.scene_pcd,
            "scene_vertices": self.scene_obj['vertices'],
            "scene_faces": self.scene_obj['faces'],
            "scene_normals": self.scene_obj['normals'],
            "scene_vertex_colors": self.scene_obj['vertex_colors'],
        }
        return payload