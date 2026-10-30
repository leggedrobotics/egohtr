import numpy as np
import rerun as rr
import trimesh
import time

from abc import ABC
from projectaria_tools.core.sophus import SE3

from src.visualizer.helper import ToTransform3D
from src.body_model.smplx_helper import SMPLH_SKELETON
from src.body_model.smplx import SMPLXHandler

from src.config.configs import SequenceContext, PipelineConfig

class EgoHTGVisualizer(ABC):
    """
    Base class for visualizing the Ego Human Terrain Generation (EgoHTG) pipeline outputs.
    
    This visualizer manages the loading, processing, and rendering of multi-sensor data, 
    scene meshes, and generated/optimized SMPL-X body model sequences using the Rerun viewer.

    The rerun viewer can be spawned in either web or desktop mode based on the pipeline configuration.
    """

    gen_data: dict | None = None
    opt_data: dict | None = None
    egoallo_data: dict | None = None
    josh_data: dict | None = None
    
    def __init__(self, pip_cfg: PipelineConfig):
        self.pip_cfg = pip_cfg
    
    def run(self, loader: dict, ctx: SequenceContext, seq_cfg: dict, gender: str = "male"):
        #init seq config
        self.ctx = ctx
        self.loader = loader
        self.seq_cfg = seq_cfg
        self.body_model = SMPLXHandler(self.pip_cfg)
        
        # visualize
        self.init_rr(gender)
        self.load_data(self.ctx)
        self.visualize_static_data() 
        self.visualize_dynamic_data()
        self.save(ctx.rerun)
        
    def init_rr(self, gender: str):
        rr.init("EgoHTG Data Visualizer", spawn=(not self.pip_cfg.web))
        # Run the viewer in the web browser or desktop app
        if self.pip_cfg.web:
            self.server_uri = rr.serve_grpc(grpc_port=self.seq_cfg["visualizer_config"]["grcp_port"], server_memory_limit="100%") #grpc_port=grpc_port, 
            rr.serve_web_viewer(web_port=self.seq_cfg["visualizer_config"]["web_port"], connect_to=self.server_uri, open_browser=True) #web_port=web_port, open_browser=open_browser, 
        else:
            rr.spawn()

        if self.ctx.blueprint_path.exists():
            # Send the blueprint to the viewer
            rr.log_file_from_path(self.ctx.blueprint_path)
            print(f"Blueprint loaded from {self.ctx.blueprint_path}")
        else:
            print(f"Blueprint file not found at {self.ctx.blueprint_path}. Using default layout.")
    
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rr.log(
            "world/xyz",
            rr.Arrows3D(
                vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            ),
        )
        self.body_model.initialize(seq_cfg={'gender': gender})
    
    def load_data(self, ctx: SequenceContext):
        print("[INFO]: Loading data for visualization")
        if ctx.gen_seq.exists():
            self.gen_data = np.load(ctx.gen_seq, allow_pickle=True)
            self._print_npz_summary(self.gen_data, "generated data")
        if ctx.opt_seq.exists():
            self.opt_data = np.load(ctx.opt_seq, allow_pickle=True)
            self._print_npz_summary(self.opt_data, "optimized data")

            
    def _get_smplx_vertices(self, sensor_type: str):
        if sensor_type == 'opt': 
            smplx_poses = {
                "body_pose": self.opt_data[f"smplx_egomc_body_pose"],
                "global_orient": self.opt_data[f"smplx_egomc_global_orient"],
                "betas": self.opt_data[f"smplx_egomc_betas"],
                "transl": self.opt_data[f"smplx_egomc_transl"],
            }
        else:
            step = 1 if sensor_type == 'exomc' else 1 # for egomc we visualize all frames, for exomc we sub-sample with a:
            smplx_poses = {
                "body_pose": self.gen_data[f"smplx_{sensor_type}_body_pose"][::step],
                "global_orient": self.gen_data[f"smplx_{sensor_type}_global_orient"][::step],
                "betas": self.gen_data[f"smplx_{sensor_type}_betas"][::step],
                "transl": self.gen_data[f"smplx_{sensor_type}_transl"][::step],
            }
        return self.body_model.get_smplx_vertices(smplx_poses)
        

    def _print_npz_summary(self, npz_obj: np.lib.npyio.NpzFile, name: str = "data"):
        print("----------------------------------------")
        print(f"[INFO]: Summary of {name}:")
        # Collect rows first
        rows = []
        for key in npz_obj.files:
            value = npz_obj[key]
            vtype = type(value).__name__
            shape = getattr(value, "shape", None)
            shape_str = str(shape) if shape is not None else ""
            rows.append((key, vtype, shape_str))

        if not rows:
            print("  (no entries)")
            return

        # Compute column widths
        key_width   = max(len(r[0]) for r in rows)
        type_width  = max(len(r[1]) for r in rows)
        # shape column can vary; no strict width needed, but we can align a bit
        print(f"  {'key'.ljust(key_width)}  {'type'.ljust(type_width)}  shape")
        print(f"  {'-' * key_width}  {'-' * type_width}  {'-' * 20}")

        # Print aligned rows
        for key, vtype, shape_str in rows:
            print(f"  {key.ljust(key_width)}  {vtype.ljust(type_width)}  {shape_str}")
        print("----------------------------------------")
        
    def visualize_static_data(self):
        """Visualize static data"""
        self.scene_mesh() if self.loader['scene'] is not None else None
        # self.aria_semi_dense()
        self.trajectory()
        self.aria_RGB_cam_calibration()
        self.aria_glasses_outline()
    
    def visualize_dynamic_data(self):
        """Visualize dynamic data"""
        assert self.gen_data is not None, "Raw data not loaded."

        # Initialize optional sources
        egomc_skeleton = None
        egomc_global_poses = None
        smplx_egomc_joints = None
        smplx_egomc_vertices = None
        smplx_egomc_faces = None
        smplx_egomc_normals = None
        exomc_global_poses = None
        exomc_skeleton = None
        smplx_exomc_joints = None
        smplx_exomc_vertices = None
        smplx_exomc_faces = None
        smplx_exomc_normals = None
        smplx_skeleton = self.gen_data["smplx_skeleton"]
        egomc_world_poses = None
        smplx_egomc_standalone_joints = None
        smplx_egomc_standalone_vertices = None    
        smplx_vertices_opt = None
        smplx_faces_opt = None
        smplx_joints_opt = None
        
        aria_T_origin_device = self.gen_data["aria_T_origin_device"]
        aria_T_device_camera = self.gen_data["aria_T_device_camera"]
        aria_rgb_imgs = self.gen_data["aria_rgb_imgs"] if "aria_rgb_imgs" in self.gen_data else None

            
        if self.ctx.egomc_exists:
            egomc_skeleton = self.gen_data["egomc_skeleton"]
            egomc_global_poses = self.gen_data["egomc_global_poses"]
            smplx_egomc_joints = self.gen_data["smplx_egomc_joints"]
            smplx_egomc_vertices = self._get_smplx_vertices('egomc')
            smplx_egomc_faces = self.gen_data["smplx_egomc_faces"]
            mesh = trimesh.Trimesh(vertices=smplx_egomc_vertices[0], faces=smplx_egomc_faces)
            smplx_egomc_normals = mesh.vertex_normals
            
        if self.ctx.exomc_exists:
            exomc_global_poses = self.gen_data["exomc_global_poses"] # sub-sample exomc frames for visualization
            exomc_skeleton = self.gen_data["exomc_skeleton"]
            smplx_exomc_joints = self.gen_data["smplx_exomc_joints"]
            smplx_exomc_vertices = self._get_smplx_vertices('exomc')
            smplx_exomc_faces = self.gen_data["smplx_exomc_faces"]
            mesh = trimesh.Trimesh(vertices=smplx_exomc_vertices[0], faces=smplx_exomc_faces)
            smplx_exomc_normals = mesh.vertex_normals
        
        if self.pip_cfg.ablation:
            egomc_world_poses = self.gen_data["egomc_world_poses"]
            smplx_egomc_standalone_joints = self.gen_data["smplx_egomc_standalone_joints"]
            smplx_egomc_standalone_vertices = self.gen_data["smplx_egomc_standalone_vertices"]

        
        if self.opt_data is not None:
            smplx_vertices_opt = self._get_smplx_vertices('opt')
            smplx_faces_opt = self.opt_data["smplx_egomc_faces"]
            smplx_joints_opt = self.opt_data["smplx_egomc_joints"]

            
        if self.pip_cfg.multi_aria:
            aria_obs_T_origin_device = self.gen_data["aria_obs_T_origin_device"]
            aria_obs_T_device_camera = self.gen_data["aria_obs_T_device_camera"]
            aria_obs_rgb_imgs = self.gen_data["aria_obs_rgb_imgs"]
              
        # create timestamp 
        fps_visu = self.seq_cfg["visualizer_config"]["fps"]
        duration = self.seq_cfg["generator_config"]["duration"]
        timestamps = np.linspace(0, duration, int(duration * fps_visu))
        frame_duration = 1.0 / fps_visu
        
        start_idx, end_idx = 0, -1
        for idx, ts in enumerate(timestamps[start_idx:end_idx]):
            idx = idx + start_idx
            start_time = time.perf_counter()
            
            rr.set_time("time", duration=ts)
            
            #aria
            self.aria_cam_pose(SE3.from_matrix(aria_T_origin_device[idx, :, :]), SE3.from_matrix(aria_T_device_camera), device="aria")
            self.aria_RGB_image(aria_rgb_imgs[idx], device="aria")
            if self.pip_cfg.multi_aria:
                self.aria_cam_pose(SE3.from_matrix(aria_obs_T_origin_device[idx, :, :]), SE3.from_matrix(aria_obs_T_device_camera), device="aria_obs")
                self.aria_RGB_image(aria_obs_rgb_imgs[idx], device="aria_obs")


            # human pose
            self.human_pose(
                idx,
                egomc_skeleton,
                exomc_skeleton,
                smplx_skeleton,
                egomc_global_poses,
                exomc_global_poses,
                egomc_world_poses,
                smplx_egomc_joints,
                smplx_egomc_vertices,
                smplx_egomc_faces,
                smplx_egomc_normals,
                smplx_egomc_standalone_joints,
                smplx_egomc_standalone_vertices,
                smplx_exomc_joints,
                smplx_exomc_vertices,
                smplx_exomc_faces,
                smplx_exomc_normals,
                smplx_joints_opt,
                smplx_vertices_opt,
                smplx_faces_opt,
            )
                
            elapsed_time = time.perf_counter() - start_time
            sleep_time = max(0.0, frame_duration - elapsed_time) 
            
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        
    
    def trajectory(self) -> None:
        assert self.gen_data is not None, "Raw data not loaded."
        rr.log(
            "world/aria/trajectory",
            rr.LineStrips3D(self.gen_data["aria_trajectory"], radii=0.008, colors=[255,0,0]),
            static=True,
        )
        if self.ctx.exomc_exists:
            rr.log(
                "world/exomc/trajectory",
                rr.LineStrips3D(self.gen_data["exomc_traj"], radii=0.008, colors=[100,0,0]),
                static=True,
            )
        if self.pip_cfg.multi_aria:
            rr.log(
                "world/aria_obs/trajectory",
                rr.LineStrips3D(self.gen_data["aria_obs_trajectory"], radii=0.008, colors=[150,0,0]),
                static=True,
            )

    def aria_semi_dense(self) -> None:
        assert self.gen_data is not None, "Raw data not loaded."
        rr.log(
            "world/aria/pcd",
            rr.Points3D(self.gen_data["aria_pcd"], radii=0.006),
            static=True,
        )

    def aria_RGB_cam_calibration(self) -> None:
        assert self.gen_data is not None, "Raw data not loaded."
        rr.log(
            f"world/aria/device/camera-rgb",
            rr.Pinhole(
                resolution=[
                    self.gen_data["aria_rgb_img_width"],
                    self.gen_data["aria_rgb_img_height"],
                ],
                focal_length=float(
                    self.gen_data["aria_rgb_focal_length"]
                ),
                image_plane_distance=0.5,
            ),
            static=True,
        )
        
        if self.pip_cfg.multi_aria:
            rr.log(
                f"world/aria_obs/device/camera-rgb",
                rr.Pinhole(
                    resolution=[
                        self.gen_data["aria_obs_rgb_img_width"],
                        self.gen_data["aria_obs_rgb_img_height"],
                    ],
                    focal_length=float(
                        self.gen_data["aria_obs_rgb_focal_length"]
                    ),
                    image_plane_distance=0.5,
                ),
                static=True,
            )
        
    def aria_glasses_outline(self) -> None:
        assert self.gen_data is not None, "Raw data not loaded."
        rr.log(
            f"world/aria/device/glasses_outline",
            rr.LineStrips3D([self.gen_data["aria_glasses_outline"]]),
            static=True,
        )
        
        if self.pip_cfg.multi_aria:
            rr.log(
                f"world/aria_obs/device/glasses_outline",
                rr.LineStrips3D([self.gen_data["aria_glasses_outline"]]),
                static=True,
            )
    
    def aria_cam_pose(self, T_world_device_transformed, T_device_camera, device: str = "aria") -> None:
        if T_world_device_transformed and T_device_camera:
            rr.log(
                f"world/{device}/device",
                ToTransform3D(T_world_device_transformed, False),
            )
            rr.log(
                f"world/{device}/device/camera-rgb",
                ToTransform3D(T_device_camera, False),
            )
            rr.log(
                f"world/{device}/device/camera-rgb/axes",
                rr.Arrows3D(
                    vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    colors=[[200, 0, 0], [0, 200, 0], [0, 0, 200]],
                ),
            )
    
            

        
    def aria_RGB_image(self, img, jpeg_quality: int = 75, device: str = "aria") -> None:
        rr.log(
            f"world/{device}/device/camera-rgb",
            rr.Image(img).compress(jpeg_quality=jpeg_quality),
        )
    
    
    def scene_mesh(self) -> None:
        assert self.gen_data is not None, "Raw data not loaded."
        rr.log(
            f"world/scene/mesh",
            rr.Mesh3D(
                vertex_positions=self.gen_data["scene_vertices"],
                triangle_indices=self.gen_data["scene_faces"],
                vertex_normals=self.gen_data["scene_normals"],
                vertex_colors=self.gen_data["scene_vertex_colors"],
            ),
        )
        # rr.log(f"world/scene/points", rr.Points3D(self.gen_data["scene_vertices"], colors=[0,255,0], radii=0.02), static=False)


    def human_pose(self, 
            idx, 
            egomc_skeleton, exomc_skeleton, smplx_skeleton, egomc_global_poses, exomc_global_poses, egomc_world_poses,
            smplx_egomc_joints, smplx_egomc_vertices, smplx_egomc_faces, smplx_egomc_normals,
            smplx_egomc_standalone_joints, smplx_egomc_standalone_vertices,
            smplx_exomc_joints, smplx_exomc_vertices, smplx_exomc_faces, smplx_exomc_normals,
            smplx_joints_opt, smplx_vertices_opt, smplx_faces_opt,
        ) -> None:
    
        if self.ctx.egomc_exists:
            rr.log(
                "world/egomc/sensor/skeleton",
                rr.LineStrips3D(
                    egomc_global_poses[idx, egomc_skeleton, :3, 3].reshape(-1, 2, 3),
                    colors=[100, 150, 150],
                ),
            )
            rr.log(
                "world/egomc/sensor/joints",
                rr.Points3D(
                    egomc_global_poses[idx, :, :3, 3],
                    colors=[100, 150, 150], 
                    radii=0.006,
                ),
                static=False,
            )
            rr.log(
                "world/egomc/smplx/skeleton",
                rr.LineStrips3D(
                    smplx_egomc_joints[idx, smplx_skeleton, :],
                    colors=[200, 150, 150],
                ),
            )
            rr.log(
                "world/egomc/smplx/joints",
                rr.Points3D(
                    smplx_egomc_joints[idx],
                    colors=[200, 150, 150],
                    radii=0.006,
                ),
                static=False,
            )
            rr.log(
                "world/egomc/smplx/mesh",
                rr.Mesh3D(
                    vertex_positions=smplx_egomc_vertices[idx],
                    triangle_indices=smplx_egomc_faces,
                    albedo_factor=[255, 150, 150],
                    vertex_normals=smplx_egomc_normals,
                )
            )

            if self.pip_cfg.ablation:
                rr.log(
                    "world/egomc_standalone/sensor/skeleton",
                    rr.LineStrips3D(
                        egomc_world_poses[idx, egomc_skeleton, :3, 3].reshape(-1, 2, 3),
                        colors=[150, 100, 100],
                    ),
                )
                rr.log(
                    "world/egomc_standalone/smplx/skeleton",
                    rr.LineStrips3D(
                        smplx_egomc_standalone_joints[idx, smplx_skeleton, :],
                        colors=[255, 100, 100],
                    ),
                )   
                rr.log(
                    "world/egomc_standalone/smplx/mesh",
                    rr.Mesh3D(
                        vertex_positions=smplx_egomc_standalone_vertices[idx],
                        triangle_indices=smplx_egomc_faces,
                        albedo_factor=[255, 100, 100],
                        vertex_normals=smplx_egomc_normals,
                    )   
                )
                
            if self.opt_data is not None:
                rr.log(
                    "world/smplx_opt/joints",
                    rr.Points3D(smplx_joints_opt[idx], colors = [255, 200, 150], radii=0.006),
                    static=False,
                )
                rr.log(
                    "world/smplx_opt/mesh",
                    rr.Mesh3D(
                        vertex_positions=smplx_vertices_opt[idx],
                        triangle_indices=smplx_faces_opt,
                        albedo_factor=[255, 150, 255],
                        vertex_normals=smplx_egomc_normals,
                    )
                )
            
        if self.ctx.exomc_exists:
            rr.log(
                "world/exomc/sensor/skeleton",
                rr.LineStrips3D(
                    exomc_global_poses[idx, exomc_skeleton, :3, 3].reshape(-1, 2, 3),
                    colors=[100, 200, 200],
                ),
            )
            rr.log(
                "world/exomc/sensor/joints",
                rr.Points3D(
                    exomc_global_poses[idx, :, :3, 3],
                    colors=[100, 200, 200],
                    radii=0.006,
                ),
                static=False,
            )
            rr.log(
                "world/exomc/smplx/skeleton",
                rr.LineStrips3D(
                    smplx_exomc_joints[idx, smplx_skeleton, :],
                    colors=[200, 200, 200],
                ),
            )
            rr.log(
                "world/exomc/smplx/joints",
                rr.Points3D(
                    smplx_exomc_joints[idx],
                    colors=[200, 200, 200],
                    radii=0.006,
                ),
                static=False,
            )
            rr.log(
                "world/exomc/smplx/joints",
                rr.Points3D(
                    exomc_global_poses[idx, [58,62], :3, 3],
                    colors=[100, 100, 100],
                    radii=0.006,
                ),
                static=False,
            )
            rr.log(
                "world/exomc/smplx/joints",
                rr.Points3D(
                    smplx_exomc_joints[idx, [51,57], :],
                    colors=[100, 100, 100],
                    radii=0.006,
                ),
                static=False,
            )
            rr.log(
                "world/exomc/smplx/mesh",
                rr.Mesh3D(
                    vertex_positions=smplx_exomc_vertices[idx],
                    triangle_indices=smplx_exomc_faces,
                    albedo_factor=[255, 200, 200],
                    vertex_normals=smplx_exomc_normals,
                )
            )
    def human_pose_egoallo(self, idx, egoallo_joints, smplx_skeleton):
        rr.log(
            "world/egoallo/device/camera-rgb/egoallo_joints",
            rr.Points3D(
                egoallo_joints[idx],
                colors=[200, 200, 100],
                radii=0.006,
            ),
            static=False,
        )
        rr.log(
                "world/egoallo/device/camera-rgb/skeleton",
                rr.LineStrips3D(
                    egoallo_joints[idx, SMPLH_SKELETON, :],
                    colors=[200, 200, 200],
                ),
            )


    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        rr.save(path)
        print(f"[INFO]: Visualization saved to {path}")
