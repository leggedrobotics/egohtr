import numpy as np
import coacd
import trimesh
import os

from scipy.spatial.transform import Rotation as R

coord_offset = { #GLOBAL (HumanIK to SMPL-X)
    "egomc": {
        "world": R.from_euler("x", -np.pi / 2),
        "joint_basis_change": R.from_euler('x', 0),
        
        "Hips": R.from_euler("x", 0),
        "Spine": R.from_euler("z", -np.pi / 2),
        "Spine1": R.from_euler("z", -np.pi / 2),
        "Spine2": R.from_euler("z", -np.pi / 2),
        "Spine3": R.from_euler("z", -np.pi / 2),
        "Neck": R.from_euler("z", -np.pi / 2), # turn, x pitch
        "Head": R.from_euler("z", 0),
        

        "LeftUpLeg": R.from_euler("z", np.pi / 2) * R.from_euler("x", - np.pi / 16) ,  # inout, forback
        "LeftLeg": R.from_euler("z", np.pi / 2 - np.pi / 64),
        "LeftFoot": R.from_euler("z", np.pi / 2) * R.from_euler("x", np.pi / 2) * R.from_euler("x", -np.pi / 8), # turn, pitch
        "LeftToeBase": R.from_euler("z", np.pi / 2) * R.from_euler("x", np.pi / 2),
        "RightUpLeg": R.from_euler("z", np.pi / 2) * R.from_euler("x", - np.pi / 16),
        "RightLeg":  R.from_euler("z", np.pi / 2 + np.pi / 64),
        "RightFoot": R.from_euler("z", np.pi / 2) * R.from_euler("x", np.pi / 2) *  R.from_euler("x", -np.pi / 8), # turn, pitch
        "RightToeBase": R.from_euler("z", np.pi / 2) * R.from_euler("x", np.pi / 2),

        "LeftShoulder": R.from_euler("z", -np.pi / 10) * R.from_euler("y", -np.pi / 10), # adjustments: shoulder down, shoulder forward
        "LeftArm": R.from_euler("z", np.pi / 10), #* R.from_euler("x", -np.pi / 4),
        "LeftForeArm": R.from_euler("x", 0),
        "LeftHand": R.from_euler("x", 0),
        "RightShoulder": R.from_euler("z", np.pi - np.pi / 10) * R.from_euler("x", -np.pi) * R.from_euler("y", np.pi / 10), # adjustments: shoulder down, shoulder forward
        "RightArm": R.from_euler("z", np.pi + np.pi / 10) * R.from_euler("x", -np.pi),
        "RightForeArm": R.from_euler("z", np.pi) * R.from_euler("x", -np.pi),
        "RightHand": R.from_euler("z", np.pi) * R.from_euler("x", np.pi),
    },
    "exomc": { #GLOBAL (QUALISYS to SMPL-X)
        
        "world": R.from_euler("z", np.pi) * R.from_euler("x", np.pi/2),
        
        "joint_basis_change": R.from_euler('x', np.pi/2) * R.from_euler('y', np.pi), # Z-up to Y-down

        "Hips": R.from_euler("x", 0),
        "Spine": R.from_euler("x", 0),
        "Spine1": R.from_euler("x", 0),
        "Spine2": R.from_euler("x", 0),
        "Spine3": R.from_euler("x", 0),
        "Neck": R.from_euler("x", 0),
        "Head": R.from_euler("x", 0),
        

        "LeftUpLeg": R.from_euler("z", 0),
        "LeftLeg": R.from_euler("z", 0),
        "LeftFoot": R.from_euler("z", 0),
        "LeftToeBase": R.from_euler("z", 0),
        "RightUpLeg": R.from_euler("z", 0),
        "RightLeg":  R.from_euler("z", 0),
        "RightFoot": R.from_euler("z", 0),
        "RightToeBase": R.from_euler("z", 0),

        "LeftShoulder": R.from_euler("z", 0) , # adjustments: shoulder down, shoulder forward
        "LeftArm": R.from_euler("z", 0),
        "LeftForeArm": R.from_euler("z", 0),
        "LeftForeArmRoll": R.from_euler("z", 0),
        "LeftHand": R.from_euler("z", 0),
        "RightShoulder": R.from_euler("z", 0), # adjustments: shoulder down, shoulder forward
        "RightArm": R.from_euler("z", 0) ,
        "RightForeArm": R.from_euler("z", 0) ,
        "RightForeArmRoll": R.from_euler("z", 0),
        "RightHand": R.from_euler("z", 0),
    },
}

smplx_to_egomc_map = {
    "pelvis": "Hips",
    "left_hip": "LeftUpLeg",
    "right_hip": "RightUpLeg",
    "spine1": "Spine",
    "left_knee": "LeftLeg",
    "right_knee": "RightLeg",
    "spine2": "Spine1",
    "left_ankle": "LeftFoot",
    "right_ankle": "RightFoot",
    "spine3": "Spine2",
    "left_foot": "LeftToeBase",
    "right_foot": "RightToeBase",
    "neck": "Neck",
    "left_collar": "LeftShoulder",
    "right_collar": "RightShoulder",
    "head": "Head",
    "left_shoulder": "LeftArm",
    "right_shoulder": "RightArm",
    "left_elbow": "LeftForeArm",
    "right_elbow": "RightForeArm",
    "left_wrist": "LeftHand",
    "right_wrist": "RightHand",
}

smplx_to_exomc_map = {
    "pelvis": "Hips",
    "left_hip": "LeftUpLeg",
    "right_hip": "RightUpLeg",
    "spine1": "Spine",
    "left_knee": "LeftLeg",
    "right_knee": "RightLeg",
    "spine2": "Spine1",
    "left_ankle": "LeftFoot",
    "right_ankle": "RightFoot",
    "spine3": "Spine2",
    "left_foot": "LeftToeBase",
    "right_foot": "RightToeBase",
    "neck": "Neck",
    "left_collar": "LeftShoulder",
    "right_collar": "RightShoulder",
    "head": "Head",
    "left_shoulder": "LeftArm",
    "right_shoulder": "RightArm",
    "left_elbow": "LeftForeArm",
    "right_elbow": "RightForeArm",
    "left_wrist": "LeftHand",
    "right_wrist": "RightHand",
}


def get_smplx_local_joint(
    frame_name, frame_quat, parent_frame_name, parent_frame_quat, basis_change, sensor="egomc", scalar_first=True
):
    parent_frame_global_rot = (
        R.from_quat(parent_frame_quat, scalar_first=scalar_first) * coord_offset[sensor][parent_frame_name]
    )
    frame_global_quat = (
        R.from_quat(frame_quat, scalar_first=scalar_first)
        * coord_offset[sensor][frame_name]
    )

    frame_local_quat = basis_change * parent_frame_global_rot.inv() * frame_global_quat * basis_change.inv()
    
    # to account for the missing spine joints of the smplx
    if frame_name in ["Spine", "Spine1", "Spine2"]: 
        factor = 1.25 # 0.0 = Stiff/No movement, 1.0 = Full movement TODO: find approach for better generalization
        
        # Convert to Rotation Vector (Axis-Angle representation)
        rot_vec = frame_local_quat.as_rotvec()
        
        # Scale the magnitude of the rotation
        damped_rot_vec = rot_vec * factor
        
        # Convert back to Rotation object
        frame_local_quat = R.from_rotvec(damped_rot_vec)
    
    return frame_local_quat.as_matrix()



def convexify_mesh(vertex_array: np.ndarray, face_array: np.ndarray) -> list:
    print("[INFO]: Decomposing scene mesh for sequence.")
    
    mesh = trimesh.Trimesh(vertices=vertex_array, faces=face_array)
    mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    parts = coacd.run_coacd(
        mesh, 
        threshold=0.01,             # Lower this for more detail (default is 0.05)
        preprocess_resolution=100,  # Increase for better initial geometry (default 50)
        resolution=2000,            # Sampling resolution for concavity (default 2000)
        # mcts_iteration=100,       # Number of search iterations (default 100)
        # mcts_depth=3,             # Search depth (default 3)
        pca=False,                  # Set to True if you want axis-aligned parts
        merge=True                  # Set to False (equivalent to -nm) to keep all tiny details
    )
    print(f"Decomposed into {len(parts)} parts.")
    
    return parts

def export_decomposed_mesh(parts, output_folder, output_combined):
    vs_all = []
    fs_all = []
    vs_counter = 0

    for i, part_mesh in enumerate(parts):
        vs, fs = part_mesh
        part_trimesh = trimesh.Trimesh(vertices=vs, faces=fs)
        output_path = os.path.join(output_folder, f"part_{i:03d}.obj")
        part_trimesh.export(output_path)
        
        vs_all.append(vs)
        fs_all.append(fs + vs_counter)
        vs_counter += len(vs)

    # Save combined mesh as well
    combined_mesh = trimesh.Trimesh(vertices=np.vstack(vs_all), faces=np.vstack(fs_all))
    combined_mesh.export(output_combined)
        


def generate_xml(mesh_folder, relative_mesh_folder, output_assets_xml, output_body_xml):
    print("[INFO]: Generating retarget XML files.")

    # Parameters TODO: make configurable
    scale_str = " 0.781065 0.781065 0.781065" #"0.7415730337078652 0.7415730337078652 0.7415730337078652"
    mat_rgba = "0.3 0.7 0.9 0.5"

    assets_lines = ["<mujocoinclude>"]
    body_lines = ["<mujocoinclude>"]
    files = sorted([f for f in os.listdir(mesh_folder) if f.endswith(".obj")])

    for filename in files:
    
        asset_name = os.path.splitext(filename)[0]
        
        # Generate Asset Lines
        assets_lines.append(f'    <mesh name="{asset_name}" file="{relative_mesh_folder}/{filename}" scale="{scale_str}"/>')
        assets_lines.append(f'    <material name="{asset_name}_material" rgba="{mat_rgba}"/>')

        # Generate Body Lines
        body_lines.append(f'    <body name="multi_boxes_{asset_name}_link" pos="0 0 0" quat="1 0 0 0">')
        body_lines.append(f'        <geom name="multi_boxes_{asset_name}_link" type="mesh" mesh="{asset_name}" '
                        f'pos="0 0 0" quat="1 0 0 0" material="{asset_name}_material" '
                        f'contype="1" conaffinity="1"/>')
        body_lines.append('    </body>')
        body_lines.append('') 

    assets_lines.append("</mujocoinclude>")
    body_lines.append("</mujocoinclude>")

    with open(output_assets_xml, "w") as f:
        f.write("\n".join(assets_lines))

    with open(output_body_xml, "w") as f:
        f.write("\n".join(body_lines))
