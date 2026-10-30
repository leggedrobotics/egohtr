import sys
import json
import numpy as np
import torch
import json
import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

print("HEy")


from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from src.loader.vggt_utils import plot_cam_poses
from src.loader.utils import (
    get_homog,
    transform_vggt_correction,
)


def vggt(image_names: list) -> np.ndarray:
    print("[INFO] Run vggt")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # Initialize the model and load the pretrained weights.
    # This will automatically download the model weights the first time it's run, which may take a while.
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

    # Load and preprocess example images (replace with your own image paths)
    images = load_and_preprocess_images(image_names).to(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            # Predict attributes including cameras, depth maps, and point maps.
            predictions = model(images)
            camera_poses, _ = pose_encoding_to_extri_intri(predictions["pose_enc"], build_intrinsics=False)

    # Optionally plot the camera poses
    plot_cam_poses(camera_poses.cpu().numpy().copy())
    
    T = transform_vggt_correction()
    camera_poses = camera_poses.cpu().numpy()


    # TODO: errorneous for multi aria sequences. Has to be readapted
    # Compute the transformation from the first to the last camera pose
    T_targetFP_sourceFP = (
        np.linalg.inv(get_homog(camera_poses[0, 0])) @ get_homog(camera_poses[0, -1])
    )
    T_targetFP_sourceFP[:2,3] = (
        np.linalg.inv(get_homog(camera_poses[0, 0])) @ get_homog(camera_poses[0, -1])
    )[:2,3]
    T_targetFP_sourceFP[2,3] = -1 * (
        np.linalg.inv(get_homog(camera_poses[0, 0])) @ get_homog(camera_poses[0, -1])
    )[2,3]
    
    return T_targetFP_sourceFP


#wrapper to run vggt in vggt venv
image_names = json.loads(sys.argv[1])
output_file = sys.argv[2]
T_targetW_sourceFP = vggt(image_names)

# save as npy
np.save(output_file, T_targetW_sourceFP)
