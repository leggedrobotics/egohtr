import os

# 1. Settings
mesh_folder = "demo_data/climb/mocap_climb_seq_8/box_models"
relative_mesh_folder = "../../../demo_data/climb/mocap_climb_seq_8/box_models"
output_assets_xml = "demo_data/climb/mocap_climb_seq_8/box_assets.xml"
output_body_xml = "demo_data/climb/mocap_climb_seq_8/box_body.xml"

# Parameters
scale_str = "0.7415730337078652 0.7415730337078652 0.7415730337078652"
mat_rgba = "0.3 0.7 0.9 0.5"

# 2. Start the XML lists
assets_lines = ["<mujocoinclude>"]
body_lines = ["<mujocoinclude>"]

# 3. Loop through files
files = sorted([f for f in os.listdir(mesh_folder) if f.endswith(".obj")])

for filename in files:
    # Get the name without extension (e.g. "box1" from "box1.obj")
    asset_name = os.path.splitext(filename)[0]
    
    # --- A. Generate Asset Lines ---
    # 1. Mesh definition
    assets_lines.append(f'    <mesh name="{asset_name}" file="{relative_mesh_folder}/{filename}" scale="{scale_str}"/>')
    # 2. Material definition
    assets_lines.append(f'    <material name="{asset_name}_material" rgba="{mat_rgba}"/>')

    # --- B. Generate Body Lines ---
    # Using the structure: body -> geom
    body_lines.append(f'    <body name="multi_boxes_{asset_name}_link" pos="0 0 0" quat="1 0 0 0">')
    body_lines.append(f'        <geom name="multi_boxes_{asset_name}_link" type="mesh" mesh="{asset_name}" '
                      f'pos="0 0 0" quat="1 0 0 0" material="{asset_name}_material" '
                      f'contype="1" conaffinity="1"/>')
    body_lines.append('    </body>')
    body_lines.append('') # Empty line for readability

# 4. Close tags
assets_lines.append("</mujocoinclude>")
body_lines.append("</mujocoinclude>")

# 5. Save Asset File
with open(output_assets_xml, "w") as f:
    f.write("\n".join(assets_lines))

# 6. Save Body File
with open(output_body_xml, "w") as f:
    f.write("\n".join(body_lines))

print(f"Done!")
print(f"1. Assets saved to: {output_assets_xml}")
print(f"2. Bodies saved to: {output_body_xml}")