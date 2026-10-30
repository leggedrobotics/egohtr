import coacd
import trimesh
import numpy as np
import os

input_file = "demo_data/scene/sequence_07/box_models/meshes_all.obj"
output_folder = "demo_data/climb/mocap_climb_seq_8/box_models"

os.makedirs(output_folder, exist_ok=True)
for f in os.listdir(output_folder):
    os.remove(os.path.join(output_folder, f))

output_combined_folder = "demo_data/climb/mocap_climb_seq_8" #os.path.join(output_folder, "combined")
os.makedirs(output_combined_folder, exist_ok=True)

mesh = trimesh.load(input_file, force="mesh")
mesh = coacd.Mesh(mesh.vertices, mesh.faces)
parts = coacd.run_coacd(
    mesh, 
    threshold=0.01,             # Lower this for more detail (default is 0.05)
    preprocess_resolution=100,   # Increase for better initial geometry (default 50)
    resolution=2000,            # Sampling resolution for concavity (default 2000)
    # mcts_iteration=100,        # Number of search iterations (default 100)
    # mcts_depth=3,               # Search depth (default 3)
    pca=False,                  # Set to True if you want axis-aligned parts
    merge=True                  # Set to False (equivalent to -nm) to keep all tiny details
)
print(f"Decomposed into {len(parts)} parts.")

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
combined_output_path = os.path.join(output_combined_folder, "multi_boxes.obj")
combined_mesh.export(combined_output_path)
    
